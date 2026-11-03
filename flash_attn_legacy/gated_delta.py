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
"""

import functools
import inspect
import logging
import os
import sys

import torch

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
