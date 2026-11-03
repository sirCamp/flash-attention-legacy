"""
Gated DeltaNet layers of HF hybrid models (Qwen3.5, Qwen3.5-MoE, Qwen3-Next, OLMo-hybrid) on pre-Ampere GPUs
(docs/PLAN.md P6.10 a, docs/analysis/kernels/qwen35_fla_guard.md).

transformers binds the layers' delta-rule functions at import time (use_kernel_func_from_hub_with_fallback): to
`fla`'s Triton kernels when `fla` imports, else to its PyTorch reference. `fla`'s chunk kernel (prefill, training)
fails to compile below compute capability 8.0 (fla 0.6.0, Triton 3.3.1: "PassManager::run failed" on SM 6.0, 6.1,
7.0 and 7.5), so installing `fla` breaks these models on every GPU this project targets. For a model built with
attn_implementation="flash_attn_legacy", guard_gated_delta routes the chunk function of its modeling module back to
the reference for CUDA tensors below SM 8.0; `fla`'s recurrent kernel (decode) runs there and stays (from the
reference chunk's state it matches the reference recurrent in fp16 / bf16 / fp32 on all five test GPUs).

The guarded function is a plain function: below SM 8.0 it also bypasses a hub kernel requested for the chunk
function (use_kernels=True). FLASH_ATTN_LEGACY_FLA_CHUNK=fla, set before the model is built, leaves `fla`'s chunk
kernel in place (a newer fla / Triton that compiles it); a guard already installed stays.

The chunk function also runs the reference with its sequential scan on our kernel (P6.10 c1 / c3,
docs/analysis/kernels/qwen35_delta_rule.md: torch.ops.flash_attn_legacy.gdn_chunk_scan, with its backward as a
reverse scan; the function 1.4-3.4x faster at 4k tokens, forward + backward 5-8x on the P100) and its prologue and
intra-chunk phase in one kernel (c2: gdn_chunk_prep, with an analytic backward in PyTorch; another ~2x for inference)
wherever `fla`'s chunk does not run, for the reference's supported arguments; only for the reference source it was
copied from (transformers 5.20.2, by fingerprint: another version runs the reference, with a warning).
FLASH_ATTN_LEGACY_GDN_SCAN=0 opts out; FLASH_ATTN_LEGACY_GDN_PREP=0 keeps the reference's lines for the prologue and
the intra-chunk phase. torch.func transforms and forward-mode AD are not supported on this path (an old-style autograd Function).
Like the guard, the wrapper is process-wide for the modeling module: other models of its classes get it too.

The layers' depthwise causal conv1d (P6.10 b, docs/analysis/kernels/qwen35_conv1d.md): transformers' reference
(F.conv1d on a transposing copy, then SiLU: 7-34x above the bandwidth floor on these GPUs) or the `causal-conv1d`
package, which has no code below SM 7.5. patch_conv1d replaces the module's causal_conv1d_fn / causal_conv1d_update
with our kernels (torch.ops.flash_attn_legacy.causal_conv1d*: conv and SiLU in fp32, one rounding, channel-last in and
out), keeping the package where it is bound and has code (SM 7.5 and newer). FLASH_ATTN_LEGACY_CONV1D=0 opts out.
"""

import functools
import hashlib
import inspect
import logging
import os
import sys

import torch

from .flash_attn import _ops
from .integrations import _record_fallback, _warn_once
from .utils import device_capability, is_compiling

logger = logging.getLogger(__name__)

# Below this compute capability `fla`'s chunk kernel does not run (measured: fla 0.6.0, Triton 3.3.1, torch 2.7.1)
FLA_CHUNK_MIN_CC = (8, 0)
_FLA_CHUNK_ENV = "FLASH_ATTN_LEGACY_FLA_CHUNK"
_CHUNK = "torch_chunk_gated_delta_rule"
_GUARDED = "_flash_attn_legacy_guarded"
# The closure of transformers' use_kernel_func_from_hub_with_fallback wrapper
_DECORATOR_CELLS = {"implementation", "torch_function", "is_new_implementation"}


def _cells(fn):
    """The closure of fn as {name: value}, or None (no closure, or an empty cell)."""
    code = getattr(fn, "__code__", None)
    if code is None:
        return None
    try:
        return {name: cell.cell_contents for name, cell in zip(code.co_freevars, fn.__closure__ or ())}
    except ValueError:  # an empty cell
        return None


def _decorator_cells(fn):
    """The closure of transformers' use_kernel_func_from_hub_with_fallback wrapper: fn itself, or the function inside
    the nn.Module that the `kernels` package's use_kernel_forward_from_hub makes of it (its forward's `func` cell).
    None for anything else, e.g. a third-party wrapper around transformers' function (left alone)."""
    if isinstance(fn, torch.nn.Module):
        outer = _cells(type(fn).forward)
        fn = outer.get("func") if outer else None
    cells = _cells(fn)
    return cells if cells is not None and _DECORATOR_CELLS <= cells.keys() else None


def _query(args, kwargs):
    if args:
        return args[0]
    return kwargs.get("query", kwargs.get("q"))


# ---------------------------------------------------------------------------
# The chunk delta rule: one wrapper decides between fla, our scan (P6.10 c, c1) and the reference
# ---------------------------------------------------------------------------

_SCAN_ENV = "FLASH_ATTN_LEGACY_GDN_SCAN"
_HAS_SCAN_OP = hasattr(_ops, "gdn_chunk_scan")  # an extension built before P6.10 c has no scan op
# ... and its backward (c3): gradient calls take the scan too
_HAS_SCAN_AUTOGRAD = (_HAS_SCAN_OP and hasattr(_ops, "gdn_chunk_scan_backward")
                      and hasattr(torch.library, "register_autograd"))
# ... and the prologue and intra-chunk phase in one op (c2); FLASH_ATTN_LEGACY_GDN_PREP=0 keeps their PyTorch lines
_PREP_ENV = "FLASH_ATTN_LEGACY_GDN_PREP"
_HAS_PREP = _HAS_SCAN_AUTOGRAD and hasattr(_ops, "gdn_chunk_prep")
# sha1 prefixes of the reference our copy below follows (and of its l2norm): transformers 5.20.2, the same in
# modeling_qwen3_5 / qwen3_5_moe / qwen3_next / olmo_hybrid. Another source: the reference itself runs.
_REFERENCE_SHA1 = {"d2f00f2b07183377"}
_L2NORM_SHA1 = {"757e6bd6fc8af596"}
_SCAN_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _sha1(fn):
    try:
        return hashlib.sha1(inspect.getsource(fn).encode()).hexdigest()[:16]
    except (OSError, TypeError):
        return None


def _scan_matches(reference):
    """Our copy of the reference (_chunk_scan) applies: the installed reference is the one it was written from."""
    l2norm = getattr(reference, "__globals__", {}).get("l2norm")
    return _sha1(reference) in _REFERENCE_SHA1 and l2norm is not None and _sha1(l2norm) in _L2NORM_SHA1


def _scan_supported(a):
    """The reference's arguments (bound) our scan takes: chunks of 64, l2norm in the kernel, K 64 / 128, V a multiple
    of 16, one head count for q / k / v, floating CUDA tensors on one device; gradients too with its backward (c3)."""
    q, k, v, g, beta, s0 = (a["query"], a["key"], a["value"], a["g"], a["beta"], a.get("initial_state"))
    tensors = [q, k, v, g, beta] + ([s0] if s0 is not None else [])
    if s0 is not None and (not isinstance(s0, torch.Tensor) or s0.shape != (q.shape[0], v.shape[2], q.shape[3],
                                                                          v.shape[3])):
        return False  # the reference broadcasts other shapes; the op takes [B, H, K, V]
    if not (a.get("chunk_size", 64) == 64 and a.get("use_qk_l2norm_in_kernel") is True
            and all(isinstance(t, torch.Tensor) and t.is_cuda and t.device == q.device for t in tensors)
            and q.dtype in _SCAN_DTYPES and q.dim() == 4 and q.shape == k.shape and v.dim() == 4
            and v.shape[:3] == q.shape[:3] and q.shape[-1] in (64, 128) and v.shape[-1] % 16 == 0
            and g.shape == q.shape[:3] and beta.shape == q.shape[:3]):
        return False
    return _HAS_SCAN_AUTOGRAD or not (torch.is_grad_enabled() and any(t.requires_grad for t in tensors))


def _chunk_scan(reference, query, key, value, g, beta, chunk_size=64, initial_state=None, output_final_state=False,
                use_qk_l2norm_in_kernel=False, kernel=True):
    """transformers' torch_chunk_gated_delta_rule (5.20.2, _REFERENCE_SHA1) with its sequential scan on our kernel
    (gdn_chunk_scan: the state at each chunk's start and the corrected values; the outputs then two batched matmuls,
    parallel over the chunks). kernel=False runs the reference's own loop instead (a test checks that this copy is then
    bitwise the reference). The prologue and the intra-chunk phase are one op (gdn_chunk_prep, c2), or the reference's
    lines (kernel=False, FLASH_ATTN_LEGACY_GDN_PREP=0, q / k / v of mixed dtypes)."""
    F = torch.nn.functional
    l2norm = reference.__globals__["l2norm"]
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    recurrent_state_shape = (batch_size, num_v_heads, k_head_dim, v_head_dim)
    padded_output_shape = (batch_size, num_v_heads, -1, v_head_dim)
    prep = (kernel and _HAS_PREP and use_qk_l2norm_in_kernel and chunk_size == 64
            and os.environ.get(_PREP_ENV) != "0" and key.dtype == query.dtype and value.dtype == query.dtype)
    if prep:
        # c2: the prologue and the intra-chunk phase in one op (fp32, row-major; its backward: _prep_backward)
        query, key, k_cumdecay, new_values, chunk_decay, intra_chunk_attn, _ = _ops.gdn_chunk_prep(
            query, key, value, g, beta, k_head_dim ** -0.5)
        num_chunks = chunk_decay.shape[-1]
        chunk_decay = chunk_decay[..., None, None]
    else:
        decay = g
        query, key, value, beta, decay = [
            x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
            for x in (query, key, value, beta, decay)
        ]
        if use_qk_l2norm_in_kernel:
            query = l2norm(query, dim=-1, eps=1e-6)
            key = l2norm(key, dim=-1, eps=1e-6)
        scaling = query.shape[-1] ** -0.5
        query = query * scaling
        pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
        query, key, value = (F.pad(x, (0, 0, 0, pad_size)) for x in (query, key, value))
        beta, decay = (F.pad(x, (0, pad_size)) for x in (beta, decay))
        total_sequence_length = sequence_length + pad_size
        num_chunks = total_sequence_length // chunk_size
        v_beta = value * beta.unsqueeze(-1)
        k_beta = key * beta.unsqueeze(-1)
        query, key, k_beta, v_beta = [
            x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, k_beta, v_beta)
        ]
        decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)
        strictly_upper_mask = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(1)
        cum_decay = decay.cumsum(dim=3)
        pairwise_decay = cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)
        pairwise_decay = pairwise_decay.masked_fill(strictly_upper_mask, float("-inf"))
        pairwise_decay = pairwise_decay.exp()
        ut_system = (k_beta @ key.transpose(-1, -2)) * pairwise_decay
        intra_chunk_attn = (query @ key.transpose(-1, -2)) * pairwise_decay
        decayed_k_beta = k_beta * cum_decay.exp().unsqueeze(-1)
        new_values = torch.linalg.solve_triangular(ut_system, v_beta, upper=False, unitriangular=True)
        k_cumdecay = torch.linalg.solve_triangular(ut_system, decayed_k_beta, upper=False, unitriangular=True)
        if kernel:  # the reference keeps these alive to its end; ours frees them before the scan's own buffers
            del ut_system, pairwise_decay, decayed_k_beta, k_beta, v_beta
        query = query * cum_decay.exp().unsqueeze(-1)
        key = key * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
        chunk_decay = cum_decay[..., -1].exp()[..., None, None]
    if initial_state is None:
        last_recurrent_state = torch.zeros(recurrent_state_shape, dtype=new_values.dtype, device=new_values.device)
    else:
        last_recurrent_state = initial_state.to(new_values)
    if kernel:
        # solve_triangular's results are column-major: the op copies them to rows (transient; the prep op's are rows);
        # under autograd it keeps what its backward needs (w, k~, d and its outputs h, v_new), w being the producer's
        # own saved result
        states, v_new, last_recurrent_state = _ops.gdn_chunk_scan(
            k_cumdecay, new_values, key, chunk_decay[..., 0, 0], last_recurrent_state)
        del k_cumdecay, new_values
        core_attn_out = torch.matmul(query, states)  # matmul's backward needs its inputs, not its output: in place
        del states
        core_attn_out += intra_chunk_attn @ v_new
    else:
        core_attn_out = torch.zeros_like(new_values)
        for i in range(num_chunks):
            v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ last_recurrent_state
            inter_chunk_attn = query[:, :, i] @ last_recurrent_state
            core_attn_out[:, :, i] = inter_chunk_attn + intra_chunk_attn[:, :, i] @ v_new
            last_recurrent_state = last_recurrent_state * chunk_decay[:, :, i] + key[:, :, i].transpose(-1, -2) @ v_new
    last_recurrent_state = None if not output_final_state else last_recurrent_state
    core_attn_out = core_attn_out.reshape(padded_output_shape)
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).to(initial_dtype, memory_format=torch.contiguous_format)
    return core_attn_out, last_recurrent_state


def _chunk_wrapper(fn, cells):
    """The module's chunk function: `fla` where it runs (SM 8.0 and newer, or FLASH_ATTN_LEGACY_FLA_CHUNK=fla), our
    scan where it applies (_scan_supported, the reference unchanged), else transformers' reference."""
    reference = cells["torch_function"]
    fla_bound = cells["is_new_implementation"] is True
    signature = inspect.signature(reference)
    named = tuple(signature.parameters)  # the keywords transformers' wrapper passes to its reference
    scan = _HAS_SCAN_OP and _scan_matches(reference)
    if _HAS_SCAN_OP and not scan:
        _warn_once("gdn_scan_fingerprint", "flash_attn_legacy: %s's chunk gated delta rule is not the transformers "
                   "reference (5.20.2) the scan kernel was written for: the reference runs (no scan kernel)",
                   getattr(reference, "__module__", "?"))

    @functools.wraps(reference)
    def chunk(*args, **kwargs):
        query = _query(args, kwargs)
        cuda = isinstance(query, torch.Tensor) and query.is_cuda
        below = cuda and device_capability(query.device.index) < FLA_CHUNK_MIN_CC
        if fla_bound and (not below or os.environ.get(_FLA_CHUNK_ENV) == "fla"):
            return fn(*args, **kwargs)
        if kwargs.get("cu_seqlens") is not None or kwargs.get("seq_idx") is not None:
            if not fla_bound:
                return fn(*args, **kwargs)  # transformers' own error for packed sequences without fla
            # as transformers' wrapper does for its reference: packed sequences cannot be kept apart there
            raise ValueError("flash_attn_legacy: packed sequences (cu_seqlens / seq_idx) in a Gated DeltaNet layer "
                             "need fla's chunk kernel, which does not run below compute capability 8.0; the "
                             "PyTorch reference would mix the sequences (the conv needs causal-conv1d too)")
        kwargs = {k: v for k, v in kwargs.items() if k in named}
        a = None
        if scan and cuda and os.environ.get(_SCAN_ENV) != "0":
            try:
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                a = dict(bound.arguments)
                a.pop("kwargs", None)  # the reference's var-keyword (its source is pinned by _REFERENCE_SHA1)
            except TypeError:
                a = None
            if a is not None and _scan_supported(a):
                return _chunk_scan(reference, **a)
        if not is_compiling():
            if fla_bound:
                _record_fallback("gated_delta_chunk_arch")  # fla's chunk does not run here: the reference does
            elif a is not None and (_HAS_SCAN_AUTOGRAD or not (torch.is_grad_enabled() and any(
                    isinstance(t, torch.Tensor) and t.requires_grad for t in a.values()))):
                _record_fallback("gated_delta_scan_input")  # a call the scan does not take
        return reference(*args, **kwargs)

    setattr(chunk, _GUARDED, True)
    return chunk


def guard_gated_delta(model) -> bool:
    """Install the chunk wrapper (_chunk_wrapper) on `model`'s modeling module (the first class of its MRO whose
    module defines the chunk function) when transformers bound it to its reference or to `fla` (left alone: another
    package, an unknown wrapper, and `fla` with FLASH_ATTN_LEGACY_FLA_CHUNK=fla). Process-wide for that module,
    idempotent. True when the module carries the wrapper (now or before)."""
    for cls in type(model).__mro__:
        module = sys.modules.get(cls.__module__)
        fn = getattr(module, _CHUNK, None)
        if fn is not None:
            break
    else:
        return False
    if getattr(fn, _GUARDED, False):
        return True
    cells = _decorator_cells(fn)
    if cells is None:
        _warn_once("gated_delta_unknown", "flash_attn_legacy: %s.%s is not transformers' kernel wrapper as known "
                   "(another transformers version, or replaced): left as it is (no fla guard, no scan kernel)",
                   module.__name__, _CHUNK)
        return False
    package = getattr(cells["implementation"], "__module__", "") or ""
    fla_bound = cells["is_new_implementation"] is True
    if fla_bound and (package.partition(".")[0] != "fla" or os.environ.get(_FLA_CHUNK_ENV) == "fla"):
        return False  # another package, or fla left in place by request
    if not fla_bound and not (_HAS_SCAN_OP and os.environ.get(_SCAN_ENV) != "0"):
        return False  # the reference stays bound: nothing to add without the scan
    try:
        wrapper = _chunk_wrapper(fn, cells)
    except (TypeError, ValueError) as e:  # a reference without a readable signature
        logger.warning("flash_attn_legacy: not wrapping %s.%s: %s", module.__name__, _CHUNK, e)
        return False
    setattr(module, _CHUNK, wrapper)
    return True


# ---------------------------------------------------------------------------
# The causal conv1d (P6.10 b)
# ---------------------------------------------------------------------------

_CONV_ENV = "FLASH_ATTN_LEGACY_CONV1D"
_CONV_FN, _CONV_UPDATE = "causal_conv1d_fn", "causal_conv1d_update"
# causal-conv1d's wheels and default build: SM 7.5 and newer (below, a package that imported has no kernel image)
CONV_PACKAGE_MIN_CC = (7, 5)
_HAS_CONV_OP = hasattr(_ops, "causal_conv1d")  # an extension built before P6.10 b has no conv ops
_CONV_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _conv_inputs_ok(x, weight, bias, activation):
    """The kernels take x [B, C, L] channel-last on CUDA, weight [C, 2..4] (x's dtype or float32), bias in the
    weight's dtype, SiLU or no activation."""
    return (_HAS_CONV_OP and isinstance(x, torch.Tensor) and x.is_cuda and x.dim() == 3 and x.dtype in _CONV_DTYPES
            and (x.stride(1) == 1 or x.shape[1] == 1) and activation in (None, "silu", "swish")
            and isinstance(weight, torch.Tensor) and weight.dim() == 2 and weight.shape[0] == x.shape[1]
            and 2 <= weight.shape[1] <= 4 and weight.dtype in (x.dtype, torch.float32) and weight.device == x.device
            and (bias is None or (bias.dtype == weight.dtype and bias.device == x.device)))


def _packed_error():
    return ValueError("flash_attn_legacy: packed sequences (cu_seqlens / seq_idx) in a Gated DeltaNet conv need "
                      "causal-conv1d (and fla's chunk kernel for the delta rule), which do not run below compute "
                      "capability 7.5 / 8.0; the reference would mix the sequences")


def _conv_targets(fn):
    """(the function to call where the package runs, the reference) of transformers' decorated conv global; the
    package only when transformers bound one."""
    cells = _decorator_cells(fn)
    if cells is None:
        return None, None
    return (fn if cells["is_new_implementation"] is True else None), cells["torch_function"]


def _uses_package(package, x):
    """The bound causal-conv1d runs: CUDA (it has no CPU code) with code for the device (SM 7.5 and newer)."""
    return (package is not None and isinstance(x, torch.Tensor) and x.is_cuda
            and device_capability(x.device.index) >= CONV_PACKAGE_MIN_CC)


def _conv_fn(fn):
    package, reference = _conv_targets(fn)

    @functools.wraps(reference)
    def causal_conv1d_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
        if _uses_package(package, hidden_states):
            return package(hidden_states, weight, bias, activation=activation, **kwargs)
        if kwargs.get("seq_idx") is not None or kwargs.get("cu_seqlens") is not None:
            raise _packed_error()
        if not _conv_inputs_ok(hidden_states, weight, bias, activation):
            if not is_compiling():
                _record_fallback("conv1d_input")
            return reference(hidden_states, weight, bias, activation=activation)
        return _ops.causal_conv1d(hidden_states, weight, bias, activation is not None)

    setattr(causal_conv1d_fn, _GUARDED, True)
    return causal_conv1d_fn


def _conv_update(fn):
    package, reference = _conv_targets(fn)

    @functools.wraps(reference)
    def causal_conv1d_update(hidden_states, conv_state, weight, bias=None, activation=None, **kwargs):
        if _uses_package(package, hidden_states):
            return package(hidden_states, conv_state, weight, bias, activation, **kwargs)
        grads = torch.is_grad_enabled() and any(isinstance(t, torch.Tensor) and t.requires_grad
                                                for t in (hidden_states, weight, bias))
        if (grads or not _conv_inputs_ok(hidden_states, weight, bias, activation)
                or not (isinstance(conv_state, torch.Tensor) and conv_state.device == hidden_states.device
                        and conv_state.dim() == 3 and conv_state.dtype in _CONV_DTYPES
                        and conv_state.shape[2] >= weight.shape[1] - 1)):
            if not is_compiling():
                _record_fallback("conv1d_input")
            return reference(hidden_states, conv_state, weight, bias, activation)
        return _ops.causal_conv1d_update(hidden_states, conv_state, weight, bias, activation is not None)

    setattr(causal_conv1d_update, _GUARDED, True)
    return causal_conv1d_update


def patch_conv1d(model) -> bool:
    """Route the causal conv1d of `model`'s modeling module (causal_conv1d_fn / causal_conv1d_update, the first class
    of its MRO whose module defines them) to our kernels, keeping a bound `causal-conv1d` package where it has code
    (SM 7.5 and newer). Process-wide for that module, idempotent. True when the module is patched (now or before)."""
    if os.environ.get(_CONV_ENV) == "0" or not _HAS_CONV_OP:
        return False
    for cls in type(model).__mro__:
        module = sys.modules.get(cls.__module__)
        if getattr(module, _CONV_FN, None) is not None and getattr(module, _CONV_UPDATE, None) is not None:
            break
    else:
        return False
    fns = {name: getattr(module, name) for name in (_CONV_FN, _CONV_UPDATE)}
    if all(getattr(fn, _GUARDED, False) for fn in fns.values()):
        return True
    if any(_decorator_cells(fn) is None for fn in fns.values()):
        _warn_once("conv1d_unknown", "flash_attn_legacy: %s's causal conv1d functions are not transformers' kernel "
                   "wrappers as known (another transformers version, or replaced): left as they are",
                   module.__name__)
        return False
    setattr(module, _CONV_FN, _conv_fn(fns[_CONV_FN]))
    setattr(module, _CONV_UPDATE, _conv_update(fns[_CONV_UPDATE]))
    return True


# Autograd of torch.ops.flash_attn_legacy.causal_conv1d (x, weight, bias, silu): dx, and dW / db when asked for
def _conv_setup_context(ctx, inputs, output):
    x, weight, bias, silu = inputs
    ctx.silu, ctx.has_bias = silu, bias is not None
    ctx.save_for_backward(x, weight, bias)


def _conv_backward(ctx, dy):
    if torch.is_grad_enabled():
        raise RuntimeError("flash_attn_legacy causal_conv1d: double backward is not supported")
    x, weight, bias = ctx.saved_tensors
    need_dw = ctx.needs_input_grad[1] or (ctx.has_bias and ctx.needs_input_grad[2])
    dx, dw, db = _ops.causal_conv1d_backward(dy, x, weight, bias, ctx.silu, need_dw)
    return (dx if ctx.needs_input_grad[0] else None, dw if ctx.needs_input_grad[1] else None,
            db if ctx.has_bias and ctx.needs_input_grad[2] else None, None)


if _HAS_CONV_OP and hasattr(torch.library, "register_autograd"):
    torch.library.register_autograd("flash_attn_legacy::causal_conv1d", _conv_backward,
                                    setup_context=_conv_setup_context)


# Autograd of torch.ops.flash_attn_legacy.gdn_chunk_scan (w, u, k~, d, initial_state) -> (h, v_new, final_state):
# the reverse scan (gdn_chunk_scan_backward) gives G (the state's gradient after each chunk), dv^ (= du) and the
# initial state's gradient; dw, dk~ and dd are batched products over the chunks (docs/analysis/kernels/
# qwen35_delta_rule.md section 6)
def _scan_setup_context(ctx, inputs, output):
    w, u, kt, d, s0 = inputs
    h, v_new, _ = output
    ctx.has_s0 = s0 is not None
    ctx.set_materialize_grads(False)  # unused outputs' gradients stay None (no zero-filled [B, H, NC, K, V])
    ctx.save_for_backward(w, kt, d, h, v_new)


def _scan_backward(ctx, dh, dv_new, d_final):
    if torch.is_grad_enabled():
        raise RuntimeError("flash_attn_legacy gdn_chunk_scan: double backward is not supported")
    w, kt, d, h, v_new = ctx.saved_tensors
    need_w, need_u, need_kt, need_d, need_s0 = ctx.needs_input_grad
    g, dvh, ds0 = _ops.gdn_chunk_scan_backward(w, kt, d, dh, dv_new, d_final, v_new.shape[-1])
    dkt = torch.matmul(v_new, g.transpose(-1, -2)) if need_kt else None
    # sum G * h per chunk as a batched dot product (no [B, H, NC, K, V] temporary)
    dd = torch.matmul(g.flatten(-2).unsqueeze(-2), h.flatten(-2).unsqueeze(-1))[..., 0, 0] if need_d else None
    del g  # freed before dw is allocated (the incoming gradients stay alive until return: autograd holds them)
    dw = -torch.matmul(dvh, h.transpose(-1, -2)) if need_w else None
    return dw, (dvh if need_u else None), dkt, dd, (ds0 if ctx.has_s0 and need_s0 else None)


if _HAS_SCAN_AUTOGRAD:
    torch.library.register_autograd("flash_attn_legacy::gdn_chunk_scan", _scan_backward,
                                    setup_context=_scan_setup_context)


# Autograd of torch.ops.flash_attn_legacy.gdn_chunk_prep (q, k, v, g, beta, scale) -> (q~, k~, w, u, d, A, T): the
# analytic backward of the prologue and the intra-chunk phase (docs/analysis/kernels/qwen35_delta_rule.md section 8,
# checked against autograd of the reference's lines in qwen35_delta_c2_backward.py), from the inputs and the saved w and
# T; u = T (beta v) is recomputed. In the op's padded chunk layout: a padded row's dc reaches the real tokens through
# the cumulative sum's reverse.
def _chunks(x, nc):
    """[B, L, H] or [B, L, H, X] in any dtype -> fp32 [B, H, NC, 64(, X)], zero-padded (the op's layout)"""
    x = x.transpose(1, 2).to(torch.float32)
    pad = nc * 64 - x.shape[2]
    if x.dim() == 3:
        return torch.nn.functional.pad(x, (0, pad)).reshape(x.shape[0], x.shape[1], nc, 64)
    return torch.nn.functional.pad(x, (0, 0, 0, pad)).reshape(x.shape[0], x.shape[1], nc, 64, x.shape[-1])


def _unchunk(x, like):
    """fp32 [B, H, NC, 64(, X)] -> `like`'s [B, L, H(, X)] and dtype"""
    B, H, nc = x.shape[:3]
    x = x.reshape(B, H, nc * 64, *x.shape[4:])[:, :, :like.shape[1]]
    return x.transpose(1, 2).to(like.dtype)


def _prep_grads(q, k, v, g, beta, w, T, scale, dqt, dkt, dw, du, dd, dA):
    """(dq, dk, dv or None, dg, dbeta) from the gradients of (q~, k~, w, u, d, A), each possibly None (zeros)"""
    nc = T.shape[2]
    qc, kc, vc, gc, bc = (_chunks(x, nc) for x in (q, k, v, g, beta))
    rq = torch.rsqrt((qc * qc).sum(-1, keepdim=True) + 1e-6)
    rk = torch.rsqrt((kc * kc).sum(-1, keepdim=True) + 1e-6)
    qn, kn = qc * (rq * scale), kc * rk
    c = gc.double().cumsum(-1)  # in fp64, as the op: its differences, under strong decays (gdn_prep.cu)
    ec, el = c.float().exp(), (c[..., -1:] - c).float().exp()
    low = torch.ones(64, 64, dtype=torch.bool, device=q.device).tril()
    P = (c.unsqueeze(-1) - c.unsqueeze(-2)).float().masked_fill(~low, float("-inf")).exp()
    c_end = c[..., -1].float()
    del c
    Tt = T.transpose(-1, -2)
    dc, dqn, dkn, dbeta = torch.zeros_like(gc), torch.zeros_like(qn), torch.zeros_like(kn), torch.zeros_like(bc)
    dv = dM = E = None
    # u = T (beta v), w = T (beta e^c kn): the right-hand sides' gradients T^T dY; M's -T^T dY Y^T (below)
    if du is not None:
        u = torch.matmul(T, vc * bc.unsqueeze(-1))
        dvb = torch.matmul(Tt, du)
        dv = dvb * bc.unsqueeze(-1)
        dbeta += (dvb * vc).sum(-1)
        dM = torch.matmul(du, u.transpose(-1, -2))
        del u, dvb
    if dw is not None:
        dkbe = torch.matmul(Tt, dw)
        t = (dkbe * kn).sum(-1) * ec
        dbeta += t
        dc += t * bc
        dkn += dkbe * (bc * ec).unsqueeze(-1)
        del dkbe
        dww = torch.matmul(dw, w.transpose(-1, -2))
        dM = dww if dM is None else dM + dww
        del dww
    if dM is not None:  # M_ij = beta_i kn_i.kn_j P_ij (j < i)
        dM = -torch.matmul(Tt, dM).tril(-1)
        KK = torch.matmul(kn, kn.transpose(-1, -2))
        S = dM * P * bc.unsqueeze(-1)
        dbeta += (dM * P * KK).sum(-1)
        dkn += torch.matmul(S, kn) + torch.matmul(S.transpose(-1, -2), kn)
        E = S * KK  # dM_ij M_ij: P's share, below
        del dM, KK, S
    if dA is not None:  # A_ij = qn_i.kn_j P_ij (j <= i)
        R = dA * P
        dqn += torch.matmul(R, kn)
        dkn += torch.matmul(R.transpose(-1, -2), qn)
        RA = R * torch.matmul(qn, kn.transpose(-1, -2))
        E = RA if E is None else E + RA
        del R, RA
    if E is not None:  # P_ij = exp(c_i - c_j)
        dc += E.sum(-1) - E.sum(-2)
        del E
    if dqt is not None:  # q~ = qn e^c
        dqn += dqt * ec.unsqueeze(-1)
        dc += (qn * dqt).sum(-1) * ec
    if dkt is not None:  # k~ = kn e^(c_63 - c)
        dkn += dkt * el.unsqueeze(-1)
        kdk = (kn * dkt).sum(-1) * el
        dc -= kdk
        dc[..., -1] += kdk.sum(-1)
    if dd is not None:  # d = e^(c_63)
        dc[..., -1] += dd * c_end.exp()
    dg = dc.flip(-1).cumsum(-1).flip(-1)  # the cumulative sum's backward
    # transformers' l2norm, x r with r = rsqrt(|x|^2 + eps): dx = r dx^ - r^3 x (x . dx^); q's scale
    dq = scale * (rq * dqn - rq ** 3 * qc * (qc * dqn).sum(-1, keepdim=True))
    dk = rk * dkn - rk ** 3 * kc * (kc * dkn).sum(-1, keepdim=True)
    return (_unchunk(dq, q), _unchunk(dk, k), None if dv is None else _unchunk(dv, v), _unchunk(dg, g),
            _unchunk(dbeta, beta))


def _prep_setup_context(ctx, inputs, output):
    q, k, v, g, beta, scale = inputs
    w, T = output[2], output[6]
    ctx.scale = scale
    ctx.mark_non_differentiable(T)
    ctx.set_materialize_grads(False)  # unused outputs' gradients stay None
    ctx.save_for_backward(q, k, v, g, beta, w, T)


def _prep_backward(ctx, dqt, dkt, dw, du, dd, dA, dT):
    if torch.is_grad_enabled():
        raise RuntimeError("flash_attn_legacy gdn_chunk_prep: double backward is not supported")
    q, k, v, g, beta, w, T = ctx.saved_tensors
    grads = _prep_grads(q, k, v, g, beta, w, T, ctx.scale, dqt, dkt, dw, du, dd, dA)
    return tuple(x if need else None for x, need in zip(grads, ctx.needs_input_grad[:5])) + (None,)


if _HAS_PREP:
    torch.library.register_autograd("flash_attn_legacy::gdn_chunk_prep", _prep_backward,
                                    setup_context=_prep_setup_context)
