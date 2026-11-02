"""
FP16 linear layers on the P100 (compute capability 6.0): HFMA2 with chunked accumulation (docs/PLAN.md P8).

torch runs FP16 nn.Linear on SM 6.0 through cuBLAS with FP32 accumulation, at about half the card's HFMA2 rate. The
kernel (csrc/kernels/linear_pascal.cu, torch.ops.flash_attn_legacy.linear) multiplies FP16 pairs along K and folds
each lane into FP32 every 64 K elements: the stock FP16 accuracy, faster from a few hundred rows on. Off by default:

    fused_linear(x, W, bias)      F.linear(x, W, bias) on the kernel when it applies, else F.linear itself
    patch_linear(model)           the FP16 nn.Linear modules of a model on an SM 6.0 GPU run fused_linear
    FLASH_ATTN_LEGACY_LINEAR=1    patch_linear on HF models loaded with attn_implementation="flash_attn_legacy"

The kernel is real SM 6.0 code only: FLASH_ATTN_LEGACY_FORCE_ARCH does not apply (fp16 arithmetic runs at 1/64 of
the fp32 rate on SM 6.1, SM 7.x has tensor cores).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flash_attn import _ops
from .integrations import _record_fallback
from .utils import cuda_autocast_dtype, is_compiling

# Fewer rows than this go to F.linear: decode and small batches, where cuBLAS is faster
MIN_M = 128
_INT32 = 1 << 31
_PLAIN_TYPES = (torch.Tensor, nn.Parameter)
# The extension has the op (an extension built before P8 does not: fused_linear is then F.linear, with a warning)
_HAS_OP = hasattr(_ops, "linear")

# Device index -> compute capability 6.0. Read through torch.cuda.get_device_properties, which torch.compile
# folds into a constant.
_SM60 = {}


def _sm60(index):
    ok = _SM60.get(index)
    if ok is None:
        props = torch.cuda.get_device_properties(index)
        ok = _SM60[index] = (props.major, props.minor) == (6, 0)
    return ok


def _int32_ok(M, N, K):
    """y [M, N] = x [M, K] W [N, K]^T fits the kernel's int32 element offsets: the limits of the op's TORCH_CHECK
    (linear_forward in csrc/flash_attn_legacy.cpp), keep them in sync."""
    return M * N < _INT32 and M * K + 32 < _INT32 and N * K + 32 < _INT32


def _fallback_reason(x, W, bias):
    """None when fused_linear(x, W, bias) runs the kernel, else the key of its reason in the fallback registry
    (integrations.FALLBACK_REASONS), or "linear_small_m": fewer than MIN_M rows, the designed decode path (not a
    registry entry: never counted). Shapes, dtypes, devices and strides only, so that torch.compile traces it:
    the addresses (16-byte alignment) and the tensor types are checked outside torch.compile only; a misaligned
    input that reaches the op under torch.compile is copied there (contiguous_aligned), not sent to F.linear."""
    if x.dim() == 0 or x.numel() < MIN_M * x.shape[-1]:
        return "linear_small_m"  # first: decode, the cheapest exit of the hot path
    if not _HAS_OP:
        return "linear_unavailable"
    tensors = (x, W) if bias is None else (x, W, bias)
    if (not is_compiling() and any(type(t) not in _PLAIN_TYPES for t in tensors)) \
            or any(t.layout != torch.strided for t in tensors):
        return "linear_layout"
    if not x.is_cuda or any(t.device != x.device for t in tensors):
        return "linear_device"
    if any(t.dtype != torch.float16 for t in tensors) or cuda_autocast_dtype() not in (None, torch.float16):
        return "linear_dtype"
    if not _sm60(x.device.index):
        return "linear_arch"
    if (W.dim() != 2 or x.shape[-1] != W.shape[1] or W.shape[1] == 0 or W.shape[1] % 8
            or (bias is not None and (bias.dim() != 1 or bias.shape[0] != W.shape[0]))):
        return "linear_shape"
    N, K = W.shape
    M = x.numel() // K
    if not (x.is_contiguous() and W.is_contiguous()) or (
            not is_compiling() and (x.data_ptr() % 16 or W.data_ptr() % 16)):
        return "linear_layout"
    if not _int32_ok(M, N, K):
        return "linear_int32"
    if not _HAS_AUTOGRAD and torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        return "linear_autograd"
    return None


def fused_linear(x, W, bias=None):
    """F.linear(x, W, bias) (x [..., K], W [N, K], bias [N]) on the HFMA2 kernel of compute capability 6.0 GPUs.

    Falls back to F.linear itself (bitwise the stock result) when the kernel does not apply: not CUDA, not SM 6.0,
    not float16 (or autocast to another dtype), fewer than MIN_M rows (decode), K % 8 != 0, non-contiguous,
    misaligned or tensor-subclass inputs, beyond int32 offsets. The reasons are counted in fallback_stats(), except
    the small-M one (the designed decode path) and not under torch.compile (this code only runs while tracing).
    Differentiable: dX on the kernel (W^T copied for it), dW and db stock; no double backward."""
    reason = _fallback_reason(x, W, bias)
    if reason is not None:
        # not counted: small M (the designed decode path) and a last dim that does not match W (F.linear raises)
        mismatch = W.dim() == 2 and x.dim() > 0 and x.shape[-1] != W.shape[1]
        if reason != "linear_small_m" and not mismatch and not is_compiling():
            _record_fallback(reason)
        return F.linear(x, W, bias)
    K = W.shape[1]
    # The op takes and returns 2-D tensors (autograd sees those): the reshapes stay outside, so that in-place ops
    # on the output (activations) never modify a view created inside the op
    return _ops.linear(x.reshape(-1, K), W, bias).reshape(*x.shape[:-1], W.shape[0])


# ---------------------------------------------------------------------------
# Autograd of torch.ops.flash_attn_legacy.linear (2-D x, W, bias)
# ---------------------------------------------------------------------------

def _grad_input(grad, W):
    """dX [M, K] = dY [M, N] W [N, K]. On the kernel as dY (W^T)^T: it reduces along contiguous rows, so it takes
    a transposed copy of W, made only when the shapes allow the kernel (else stock)."""
    M, N = grad.shape
    K = W.shape[1]
    if grad.dtype == torch.float16 and M >= MIN_M and N > 0 and N % 8 == 0 and _int32_ok(M, K, N):
        return _ops.linear(grad.contiguous(), W.t().contiguous(), None)
    return grad.mm(W)


def _setup_context(ctx, inputs, output):
    x, W, bias = inputs
    ctx.has_bias = bias is not None
    # x only for dW, W only for dX
    ctx.save_for_backward(x if W.requires_grad else None, W if x.requires_grad else None)


def _backward(ctx, grad):
    if torch.is_grad_enabled():
        raise RuntimeError("flash_attn_legacy linear: double backward is not supported")
    x, W = ctx.saved_tensors
    dx = dW = db = None
    if ctx.needs_input_grad[0]:
        dx = _grad_input(grad, W)
    if ctx.needs_input_grad[1]:
        dW = grad.t().mm(x)  # stock cuBLAS (fp32 accumulation)
    if ctx.has_bias and ctx.needs_input_grad[2]:
        db = grad.sum(0)
    return dx, dW, db


_HAS_AUTOGRAD = hasattr(torch.library, "register_autograd")  # torch >= 2.4; else gradients go to F.linear
if _HAS_AUTOGRAD and _HAS_OP:
    torch.library.register_autograd("flash_attn_legacy::linear", _backward, setup_context=_setup_context)


# ---------------------------------------------------------------------------
# nn.Linear modules
# ---------------------------------------------------------------------------

class _FusedLinear(nn.Linear):
    """An nn.Linear whose forward is fused_linear: patch_linear swaps the class of the module (no instance
    attribute: nn.DataParallel replicas, deepcopy, pickling and torch.compile see a plain module)."""

    def forward(self, input):
        # decode (fewer than MIN_M rows) straight to F.linear: an HF decode step calls hundreds of these and is
        # bound by the host, where fused_linear's checks cost 1-2% of the decode rate
        if input.numel() < MIN_M * self.in_features:
            return F.linear(input, self.weight, self.bias)
        return fused_linear(input, self.weight, self.bias)


def _patchable(module):
    """An nn.Linear (exactly) with no instance forward (e.g. accelerate hooks), plain nn.Parameter weight / bias
    (no DTensor / FSDP2 or other subclasses) and an FP16 weight on a compute capability 6.0 GPU."""
    weight, bias = module.weight, module.bias
    return (type(module) is nn.Linear and "forward" not in module.__dict__ and type(weight) is nn.Parameter
            and (bias is None or type(bias) is nn.Parameter) and weight.dtype == torch.float16 and weight.is_cuda
            and _sm60(weight.device.index))


def patch_linear(model: nn.Module, enable: bool = True) -> int:
    """Route the forward of every nn.Linear of `model` (itself included) with an FP16 weight on a compute
    capability 6.0 GPU through fused_linear (the module's class becomes a private nn.Linear subclass);
    enable=False restores them all. Subclasses of nn.Linear, modules that already carry an instance forward
    (e.g. accelerate hooks) and weights that are not plain nn.Parameter are left alone; so are other dtypes and
    GPUs (a no-op there), and everything when the extension has no linear op. Idempotent. Returns the number of
    patched modules after the call.

    The check is made once, here: a patched module moved to another device or dtype later falls back to F.linear
    per call (fused_linear). Changing the model's attention implementation later (set_attn_implementation) does
    not unpatch: call patch_linear(model, enable=False)."""
    patched = 0
    for module in model.modules():
        if type(module) is _FusedLinear:
            if enable:
                patched += 1
            else:
                module.__class__ = nn.Linear
        elif enable and _HAS_OP and isinstance(module, nn.Linear) and _patchable(module):
            module.__class__ = _FusedLinear
            patched += 1
    return patched
