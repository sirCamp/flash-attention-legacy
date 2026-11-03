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

The layers' depthwise causal conv1d (P6.10 b, docs/analysis/kernels/qwen35_conv1d.md): transformers' reference
(F.conv1d on a transposing copy, then SiLU: 7-34x above the bandwidth floor on these GPUs) or the `causal-conv1d`
package, which has no code below SM 7.5. patch_conv1d replaces the module's causal_conv1d_fn / causal_conv1d_update
with our kernels (torch.ops.flash_attn_legacy.causal_conv1d*: conv and SiLU in fp32, one rounding, channel-last in and
out), keeping the package where it is bound and has code (SM 7.5 and newer). FLASH_ATTN_LEGACY_CONV1D=0 opts out.
"""

import functools
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


def _guarded_chunk(fn, reference):
    # the keywords transformers' wrapper passes to its reference: the named parameters only
    named = tuple(inspect.signature(reference).parameters)

    @functools.wraps(reference)
    def chunk(*args, **kwargs):
        query = _query(args, kwargs)
        if isinstance(query, torch.Tensor) and query.is_cuda and (
                device_capability(query.device.index) < FLA_CHUNK_MIN_CC):
            # as transformers' wrapper does for its reference: packed sequences cannot be kept apart there
            if kwargs.get("cu_seqlens") is not None or kwargs.get("seq_idx") is not None:
                raise ValueError("flash_attn_legacy: packed sequences (cu_seqlens / seq_idx) in a Gated DeltaNet layer "
                                 "need fla's chunk kernel, which does not run below compute capability 8.0; the "
                                 "PyTorch reference would mix the sequences (the conv needs causal-conv1d too)")
            if not is_compiling():
                _record_fallback("gated_delta_chunk_arch")
            return reference(*args, **{k: v for k, v in kwargs.items() if k in named})
        return fn(*args, **kwargs)

    setattr(chunk, _GUARDED, True)
    return chunk


def guard_gated_delta(model) -> bool:
    """Route the chunk delta rule of `model`'s modeling module (the first class of its MRO whose module defines it)
    to transformers' PyTorch reference for CUDA tensors below compute capability 8.0, when transformers bound it to
    `fla`. Process-wide for that module, idempotent. True when the module is guarded (now or before)."""
    if os.environ.get(_FLA_CHUNK_ENV) == "fla":
        return False
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
                   "(another transformers version, or replaced): left as it is, not guarded for fla below SM 8.0",
                   module.__name__, _CHUNK)
        return False
    package = getattr(cells["implementation"], "__module__", "") or ""
    if cells["is_new_implementation"] is not True or package.partition(".")[0] != "fla":
        return False  # bound to the reference (no fla), or to another package
    try:
        guarded = _guarded_chunk(fn, cells["torch_function"])
    except (TypeError, ValueError) as e:  # a reference without a readable signature
        logger.warning("flash_attn_legacy: not guarding %s.%s: %s", module.__name__, _CHUNK, e)
        return False
    setattr(module, _CHUNK, guarded)
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
    return package is not None and (not x.is_cuda or device_capability(x.device.index) >= CONV_PACKAGE_MIN_CC)


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
