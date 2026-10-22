"""
HuggingFace Transformers integration for flash_attn_legacy.

Registers "flash_attn_legacy" as an attention implementation backend,
so users can load models with:

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        attn_implementation="flash_attn_legacy",
        torch_dtype=torch.float16,
        device_map="cuda",
    )

Requires transformers >= 5.0 (uses ALL_ATTENTION_FUNCTIONS registry).
For transformers < 5.0, use the monkey-patch approach in examples/hf_llama_patch.py.
"""

import logging
import math
from collections import Counter
from typing import Dict, NamedTuple, Optional

import torch
import torch.nn as nn

from .utils import SUPPORTED_HEAD_DIMS, fp16_autocast_active

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fallback registry — the single place that lists why a call does not run our
# kernel. Each reason either falls back to torch SDPA (correct result, no
# kernel) or raises, when no available path would give a correct result
# (better an explicit error than a silently wrong output).
#
# Counters and the warn-once set are plain module state: updates from several
# threads may at worst lose a count or log a warning twice. Under torch.compile
# they only run at trace time (see docs/PLAN.md, P6.5).
# ---------------------------------------------------------------------------


class FallbackReason(NamedTuple):
    message: str
    # "sdpa": torch SDPA, with position_bias added to its mask; a call with a
    # softcap or sinks (which SDPA would ignore) goes to _eager_attention
    # instead, which applies them and position_bias. "raise": NotImplementedError
    action: str = "sdpa"
    expected: bool = False   # expected by design: counted but not logged
    hint: str = ""           # what to use instead (for action="raise")


FALLBACK_REASONS: Dict[str, FallbackReason] = {
    # -- no correct interpretation: explicit error --------------------------
    "cu_seq_lens": FallbackReason(
        "inconsistent cu_seq_lens (q/k differ, batch > 1, or not 0 < ... < q_len == kv_len)",
        "raise", hint="segments must be the same for queries and keys, in a batch of 1, "
                      "strictly increasing from 0 to the sequence length"),
    # -- correct on SDPA, the kernel does not handle them (yet) ---------------
    "position_bias": FallbackReason("additive position_bias (T5 family)"),
    "decode": FallbackReason("q_len == 1 (decode)", expected=True),
    "attention_mask": FallbackReason(
        "explicit attention mask (padding inside a sequence, bidirectional sliding window, "
        "custom masks, ...)",
        expected=True),
    "q_longer_than_kv": FallbackReason("q_len > kv_len"),
    "head_dim": FallbackReason(f"head_dim not in {SUPPORTED_HEAD_DIMS}"),
    "v_head_dim": FallbackReason("V head_dim != Q head_dim (e.g. MLA)"),
    "dropout": FallbackReason("attention dropout in training mode"),
    "fp16_overflow": FallbackReason(
        "non-fp16 inputs (bf16/fp32) exceed the fp16 range the kernel computes in"),
    "non_fp16_training": FallbackReason(
        "bf16/fp32 training: kernel gradients are fp16 and may underflow without loss "
        "scaling (docs/PLAN.md P6.2)"),
}

# 2D masks follow HF's flash_attention_2 convention: a [B, n] mask means only the
# first n key slots hold tokens (the rest are empty slots of a pre-allocated
# cache) and zeros inside it are padding. They come from _flash_attn_legacy_mask
# or straight from models that pass HF's padding mask to "flash" backends.
# _flash_attn_legacy_mask tags its all-True masks with this attribute so the
# forward can skip a device sync; the attribute is only a fast path (it is lost
# if transformers copies the mask), never needed for correctness.
_N_VALID_ATTR = "_flash_attn_legacy_n_valid"
# The 2D masks _flash_attn_legacy_mask builds for causal sliding-window layers
# carry the window W (HF sliding_window: query i sees keys j with i - j < W) in
# their values: int32, W + 1 on kept slots and 0 on padding (W + 1 > 1, so never
# read from a plain 0/1 mask). Some models pass no `sliding_window` kwarg, so the
# window must survive copies of the mask (e.g. accelerate's .to(device) between
# GPUs); _WINDOW_ATTR is only the sync-free fast path, as _N_VALID_ATTR.
_WINDOW_ATTR = "_flash_attn_legacy_window"
_WINDOW_DTYPE = torch.int32

_fallback_counts: Counter = Counter()
_warned_reasons: set = set()


def _warn_once(key: str, message: str, *args) -> None:
    """Log a warning once per key (reset by reset_fallback_stats)."""
    if key not in _warned_reasons:
        _warned_reasons.add(key)
        logger.warning(message, *args)


def _record_fallback(reason: str, detail: str = "") -> str:
    """Count a fallback; warn once per reason (unless expected); raise if the
    reason has no correct fallback. Returns the action ("sdpa" or "eager")."""
    if reason not in FALLBACK_REASONS:
        raise KeyError(f"unknown fallback reason {reason!r}")
    info = FALLBACK_REASONS[reason]
    _fallback_counts[reason] += 1
    if info.action == "raise":
        raise NotImplementedError(
            f"flash_attn_legacy: {info.message}{detail} is not supported yet, and the SDPA "
            f"fallback would silently ignore it: {info.hint}.")
    if not info.expected:
        _warn_once(reason, "flash_attn_legacy: falling back to %s attention: %s%s",
                   info.action, info.message, detail)
    return info.action


def fallback_stats() -> Dict[str, int]:
    """Number of attention calls that did not run the kernel, per reason
    (keys of FALLBACK_REASONS)."""
    return dict(_fallback_counts)


def reset_fallback_stats() -> None:
    """Reset the counters returned by fallback_stats() and the warn-once state."""
    _fallback_counts.clear()
    _warned_reasons.clear()


def _kernel_fallback_reason(module, query, key, value, attention_mask, dropout, kwargs):
    """Decide per call (on the real tensors and kwargs, never on the model
    config) whether the kernel can run. Returns (reason, detail) or (None, "")."""
    if kwargs.get("position_bias") is not None:
        return "position_bias", ""

    q_len, kv_len, d = query.shape[2], key.shape[2], query.shape[-1]
    if q_len == 1:
        return "decode", ""
    if q_len > kv_len:
        return "q_longer_than_kv", f" (q_len={q_len}, kv_len={kv_len})"
    if d not in SUPPORTED_HEAD_DIMS:
        return "head_dim", f" (head_dim={d})"
    if value.shape[-1] != d:
        return "v_head_dim", f" (q/k head_dim={d}, v head_dim={value.shape[-1]})"
    if attention_mask is not None:
        return "attention_mask", ""
    if dropout > 0.0 and module.training:
        return "dropout", ""
    if query.dtype != torch.float16 and torch.is_grad_enabled() and (
            query.requires_grad or key.requires_grad or value.requires_grad) and not fp16_autocast_active():
        return "non_fp16_training", f" ({query.dtype})"
    return None, ""


def _causal_mask(q_len: int, kv_len: int, device, window: Optional[int] = None) -> torch.Tensor:
    """Bottom-right aligned causal mask (True = keep): query i sees keys
    j <= i + (kv_len - q_len), the convention of the kernel; with a sliding
    `window` only the last `window` of them (HF sliding_window)."""
    keep = torch.ones(q_len, kv_len, dtype=torch.bool, device=device).tril(diagonal=kv_len - q_len)
    if window is not None:
        keep = keep.triu(diagonal=kv_len - q_len - window + 1)
    return keep


# Upper bound for the fp32 logits of one query block in _eager_attention (the
# block holds ~3 live copies of it: logits, probabilities, a masked temporary).
_EAGER_BLOCK_BYTES = 256 * 1024 * 1024


def _eager_attention(query, key, value, mask, scaling, dropout_p, softcap=None,
                     sinks=None, position_bias=None):
    """Reference attention for features SDPA cannot express, in fp32, one block
    of query rows at a time (block size from _EAGER_BLOCK_BYTES).

    Semantics follow transformers' eager implementations: logits = q k^T * scale;
    softcap * tanh(logits / softcap) (Gemma 2); + position_bias (additive, must
    NOT include the mask; 2D-4D, broadcastable to [B, H, q, kv]); the mask (2D-4D,
    bool: True = keep, or additive float); softmax with an extra per-head sink logit that takes
    probability mass but contributes no value (gpt-oss); dropout.

    q: [B, H, q, d]; k, v: [B, H_kv, kv, d] (GQA not expanded: query heads are
    grouped per KV head, so K/V are never copied H/H_kv times). Returns
    [B, H, q, d_v] in the query dtype. Inference-oriented: under autograd every
    block's intermediates are kept. Temporary: replaced by the kernel
    (docs/PLAN.md §3b)."""
    B, H, q_len, d = query.shape
    H_kv, kv_len = key.shape[1], key.shape[2]
    rep = H // H_kv
    lowest = torch.finfo(torch.float32).min  # like HF eager: no -inf, so no NaN rows

    def grouped(t):
        """[*, H or 1, rows, kv] -> [*, H_kv or 1, rep or 1, rows, kv] (query-head groups)."""
        if t.shape[1] == 1:
            return t.unsqueeze(2)
        return t.reshape(t.shape[0], H_kv, rep, *t.shape[2:])

    def rows(t, r0, r1):  # slice the query dimension when it is not broadcast
        return t if t.shape[-2] == 1 else t[..., r0:r1, :]

    def as_4d(t, name):
        if t.dim() not in (2, 3, 4):
            raise ValueError(f"flash_attn_legacy: {name} must be 2D, 3D or 4D, got {tuple(t.shape)}")
        while t.dim() < 4:
            t = t.unsqueeze(0)
        return grouped(t[..., :kv_len].expand(-1, -1, q_len, -1))

    if mask is not None:
        mask = as_4d(mask, "attention mask")
    if position_bias is not None:
        position_bias = as_4d(position_bias, "position_bias")
    if sinks is not None:
        sinks = sinks.float().reshape(1, H_kv, rep, 1, 1)

    # The `rep` query heads of a KV head are folded into the row dimension so that
    # both matmuls have identical batch dims ([B, H_kv]): a broadcast matmul would
    # materialise K and V `rep` times per block.
    k32, v32 = key.float(), value.float()      # [B, H_kv, kv, d]
    q = query.reshape(B, H_kv, rep, q_len, d)
    d_v = value.shape[-1]
    out = torch.empty(B, H_kv, rep, q_len, d_v, dtype=query.dtype, device=query.device)
    block = max(1, _EAGER_BLOCK_BYTES // (3 * 4 * B * H * max(kv_len, 1)))
    for r0 in range(0, q_len, block):
        r1 = min(r0 + block, q_len)
        n = r1 - r0
        qb = q[:, :, :, r0:r1].float().reshape(B, H_kv, rep * n, d)
        logits = (qb @ k32.transpose(-2, -1)).view(B, H_kv, rep, n, kv_len) * scaling
        if softcap:
            logits = softcap * torch.tanh(logits / softcap)
        if position_bias is not None:
            logits = logits + rows(position_bias, r0, r1).float()
        if mask is not None:
            m = rows(mask, r0, r1)
            logits = logits.masked_fill(~m, lowest) if m.dtype == torch.bool else logits + m.float()
        logits = logits.clamp_min(lowest)  # -inf (or overflow) from additive inputs -> finite
        if sinks is not None:
            logits = torch.cat([logits, sinks.expand(B, -1, -1, r1 - r0, 1)], dim=-1)
        probs = torch.softmax(logits, dim=-1)
        if sinks is not None:
            probs = probs[..., :-1]  # the sink absorbs probability, contributes no value
        probs = torch.nn.functional.dropout(probs, p=dropout_p)
        o = probs.reshape(B, H_kv, rep * n, kv_len) @ v32
        out[:, :, :, r0:r1] = o.view(B, H_kv, rep, n, d_v).to(query.dtype)
    return out.reshape(B, H, q_len, d_v)


# Segments derived from a cu_seq_lens / position_ids tensor, and unpadding plans
# derived from a padding mask, are cached on that tensor: transformers passes
# the same object to every layer, so the host sync needed to read them happens
# once per forward, not once per layer. The key includes the tensor version
# (bumped by every in-place write, e.g. a reused buffer) and the device. An
# inference-mode tensor has no version counter: it is cached only when it is a
# padding mask made by our own mask builder (_PADDING_ATTR: a fresh copy on every
# builder call, never written), else recomputed. The attribute is a fast path, never needed for
# correctness.
_SEGMENTS_ATTR = "_flash_attn_legacy_segments"
_UNPAD_ATTR = "_flash_attn_legacy_unpad"
# Set by _flash_attn_legacy_mask on the 2D padding masks it returns (they do
# contain padding: no per-layer check needed).
_PADDING_ATTR = "_flash_attn_legacy_padding"


def _cached_segments(source, q_len, device, compute, attr=_SEGMENTS_ATTR):
    if source.is_inference():
        if not getattr(source, _PADDING_ATTR, False):
            return compute()  # no version counter: no caching
        version = 0  # our own padding mask: never modified
    else:
        version = source._version
    key = (q_len, version, device)
    cached = getattr(source, attr, None)
    if cached is not None and cached[0] == key:
        return cached[1]
    result = compute()
    try:
        setattr(source, attr, (key, result))
    except (AttributeError, RuntimeError):
        pass
    return result


def _packed_segments(query, key, attention_mask, padding_2d, kwargs):
    """(cu_seqlens int32 [n_segments + 1], max_seqlen) when the batch-1 sequence is
    a packing of independent segments, else None.

    Segments come from cu_seq_lens_q/k (VLM vision encoders, explicit packing),
    validated, or — as in HF flash_attention_2 — from position_ids returning to
    their minimum inside the sequence (DataCollatorWithFlattening; the first
    segment may start anywhere, e.g. a document split mid-way): with a cache in
    use transformers builds no packing mask at all (B4), and on sliding-window
    layers its mask carries the window only. Other position jumps (e.g.
    PoSE-style skipped positions) are not a packing: no split, warn once."""
    q_len, kv_len = query.shape[2], key.shape[2]
    cu_q, cu_k = kwargs.get("cu_seq_lens_q"), kwargs.get("cu_seq_lens_k")
    if cu_q is not None or cu_k is not None:
        if cu_q is None or cu_k is None or query.shape[0] != 1 or (
                cu_q is not cu_k and not torch.equal(cu_q, cu_k)):
            _record_fallback("cu_seq_lens")  # raises

        if q_len != kv_len:
            _record_fallback("cu_seq_lens", f" (q_len={q_len}, kv_len={kv_len})")  # raises

        def from_cu():
            edges = cu_q.tolist()
            if (edges[0] != 0 or edges[-1] != q_len
                    or any(b <= a for a, b in zip(edges, edges[1:]))):
                _record_fallback("cu_seq_lens", f" (cu_seq_lens={edges[:6]}..., "
                                                f"q_len={q_len}, kv_len={kv_len})")  # raises
            return (cu_q.to(device=query.device, dtype=torch.int32),
                    max(b - a for a, b in zip(edges, edges[1:])))
        return _cached_segments(cu_q, q_len, query.device, from_cu)

    position_ids = kwargs.get("position_ids")
    batch1_mask = attention_mask is None or (attention_mask.dim() == 4 and attention_mask.shape[0] == 1)
    if (position_ids is None or padding_2d or not batch1_mask or query.shape[0] != 1
            or q_len != kv_len or q_len < 2 or position_ids.numel() != q_len):
        return None  # (padded rows carry filler positions: not a packing)

    def from_positions():
        pos = position_ids.reshape(-1)
        jumps = pos[1:] != pos[:-1] + 1
        restarts = jumps & (pos[1:] == pos.min())
        n_jumps, n_restarts = torch.stack([jumps.sum(), restarts.sum()]).tolist()
        if n_restarts == 0 or n_jumps != n_restarts:
            if n_jumps:
                _warn_once("ambiguous_positions",
                           "flash_attn_legacy: position_ids jump without restarting at their "
                           "initial value; treated as one sequence (not a packing)")
            return None
        edges = [0] + (restarts.nonzero().flatten() + 1).tolist() + [q_len]
        cu = torch.tensor(edges, dtype=torch.int32, device=query.device)
        return cu, max(b - a for a, b in zip(edges, edges[1:]))
    return _cached_segments(position_ids, q_len, query.device, from_positions)


def _unpad_plan(key_mask, q_len, causal, device):
    """Native padding (docs/PLAN.md P3.1): rows of the flattened [B * n] batch for
    the varlen kernel, as (q_idx, k_idx, cu_q, cu_k, max_q, max_k), or None when
    the key padding of some row is not one contiguous block (fallback).

    Keys: the valid slots of the [B, kv_len] key mask (True = keep). Queries:
      - causal (self-attention only, the builder never gives a 2D mask
        otherwise): the valid ones among the last q_len key slots — HF's layout
        for prefill and chunked prefill; with contiguous key padding they end at
        the last valid key, so per-sequence bottom-right causality is exact;
      - bidirectional (encoders, cross-attention, whose queries are unrelated to
        the key mask): every query row, so padded query rows attend the valid
        keys exactly as in SDPA.
    The plan's host sync happens once per mask tensor (cached on it); the varlen
    kernel is called with validate=False (cu_seqlens built here), so no other."""
    def compute():
        B, kv_len = key_mask.shape
        kmask = key_mask.to(device=device, dtype=torch.bool)
        pos = torch.arange(kv_len, device=device)
        nk = kmask.sum(1)
        first = torch.where(kmask, pos, kv_len).amin(1)
        last = torch.where(kmask, pos, -1).amax(1)
        qmask = kmask[:, kv_len - q_len:]
        nq = qmask.sum(1)
        n_k, f_k, l_k, n_q = torch.stack([nk, first, last, nq]).tolist()  # the host sync
        if any(n and l - f + 1 != n for n, f, l in zip(n_k, f_k, l_k)):
            return None  # padding inside a row

        def cu(n):
            return torch.nn.functional.pad(n.cumsum(0, dtype=torch.int32), (1, 0))
        keys = (kmask.reshape(-1).nonzero().flatten(), cu(nk), max(max(n_k), 1))
        causal_q = (qmask.reshape(-1).nonzero().flatten(), cu(nq), max(max(n_q), 1))
        all_q = (torch.arange(B * q_len, device=device),
                 torch.arange(0, (B + 1) * q_len, q_len, dtype=torch.int32, device=device), q_len)
        return keys, causal_q, all_q

    cached = _cached_segments(key_mask, q_len, device, compute, attr=_UNPAD_ATTR)
    if cached is None:
        return None
    (k_idx, cu_k, max_k), (q_idx, cu_q, max_q) = cached[0], cached[1 if causal else 2]
    return q_idx, k_idx, cu_q, cu_k, max_q, max_k


def _is_sliding_causal(mask_function, window: int) -> bool:
    """`mask_function` behaves as transformers' sliding_window_causal_mask_function
    (window) on probe points around the window edges of two consecutive rows with
    more than `window` keys before them (rules out chunked attention, chunk size
    2 included, and other overlays)."""
    t = torch.tensor
    for q in (2 * window + 3, 2 * window + 4):
        expected = {0: False, q - window: False, q - window + 1: True, q: True, q + 1: False}
        try:
            if not all(bool(mask_function(t(0), t(0), t(q), t(kv))) == keep for kv, keep in expected.items()):
                return False
        except (TypeError, IndexError, RuntimeError):  # not index-based: not ours to interpret
            return False
    return True


def _mask_window(attention_mask) -> Optional[int]:
    """The sliding window W a 2D mask of _flash_attn_legacy_mask carries
    (see _WINDOW_ATTR), or None. Reads the values (a device sync) only when the
    tag was lost in a copy."""
    if attention_mask is None or attention_mask.dim() != 2:
        return None
    window = getattr(attention_mask, _WINDOW_ATTR, None)
    if window is None and attention_mask.dtype == _WINDOW_DTYPE and attention_mask.numel() > 0:
        top = int(attention_mask.max())
        window = top - 1 if top > 1 else None
    return window


def _segment_mask(cu_seqlens, n, causal, device) -> torch.Tensor:
    """[1, 1, n, n] boolean mask (True = keep) of a packing: tokens attend within
    their own segment only (and causally within it when `causal`)."""
    seg = torch.searchsorted(cu_seqlens[1:].to(device=device, dtype=torch.long),
                             torch.arange(n, device=device), right=True)
    mask = seg[:, None] == seg[None, :]
    if causal:
        mask = mask & _causal_mask(n, n, device)
    return mask[None, None]


def flash_attn_legacy_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    **kwargs,
):
    """Attention function compatible with transformers 5.x AttentionInterface.

    Signature matches what ALL_ATTENTION_FUNCTIONS expects.
    Runs the flash_attn_legacy kernel when _kernel_fallback_reason allows it
    (no mask other than padding, packing or a static cache's valid prefix,
    1 < q_len <= kv_len, supported head_dim, no unsupported kwargs).
    Otherwise the reason is recorded in FALLBACK_REASONS (see fallback_stats())
    and the call runs on PyTorch SDPA (position_bias added to its mask), on
    _eager_attention (softcap / sinks, which SDPA would ignore, e.g. on decode
    steps) or raises NotImplementedError.

    Returns (attn_output, attn_weights) where attn_weights is always None.
    """
    if fp16_autocast_active() and query.dtype != torch.float16:
        # fp16 autocast (e.g. q / k fp32 after HF's rotary, v fp16): attention
        # computes in fp16, as SDPA does under autocast (no range check)
        query, key, value = query.half(), key.half(), value.half()
    elif not key.dtype == value.dtype == query.dtype:
        # e.g. an fp32 static cache: compute in the query dtype
        key, value = key.to(query.dtype), value.to(query.dtype)

    # Causal sliding window (HF sliding_window = W: query i sees keys i - j < W):
    # from our mask builder (_WINDOW_ATTR), else from the layer's kwarg as in HF's
    # flash_attention_2. The kernel gets window_size=(W - 1, W - 1), right side 0
    # when causal.
    window = _mask_window(attention_mask)
    if window is None:
        window = kwargs.get("sliding_window")

    padding_2d = False
    key_mask_2d = None  # [B, kv] padding mask (True = keep) when padding_2d
    if attention_mask is not None and attention_mask.dim() == 2:
        # [B, n] mask (see _N_VALID_ATTR): keep the first n key slots only.
        n_valid = attention_mask.shape[-1]
        if n_valid > key.shape[2]:
            raise RuntimeError(f"flash_attn_legacy: 2D mask covers {n_valid} keys, "
                               f"but only {key.shape[2]} keys were passed")
        key, value = key[:, :, :n_valid], value[:, :, :n_valid]
        if getattr(attention_mask, _N_VALID_ATTR, None) == n_valid or (
                not getattr(attention_mask, _PADDING_ATTR, False) and bool(attention_mask.all())):
            attention_mask = None  # no padding
        else:
            # Padding (1 = keep): unpadded onto the varlen kernel (_unpad_plan), or a
            # broadcastable boolean mask for the fallbacks (causality added there).
            key_mask_2d = attention_mask
            attention_mask = attention_mask[:, None, None, :].bool()
            padding_2d = True

    # query/key/value come in as [B, H, N, d] from the model's attention layer
    B, H_q, q_len, d = query.shape
    kv_len = key.shape[2]

    if scaling is None:
        scaling = 1.0 / math.sqrt(d)

    # Detect causal: kwargs override (None means "not given", as in HF's SDPA),
    # else module.is_causal (set by the model's attention layer).
    if kwargs.get("is_causal") is not None:
        is_causal = bool(kwargs["is_causal"])
    elif hasattr(module, 'is_causal'):
        is_causal = module.is_causal
    elif attention_mask is None and q_len > 1:
        is_causal = True
    else:
        is_causal = False

    # Packed independent segments (batch of 1): varlen kernel, or a block-diagonal
    # mask in the fallbacks.
    segments = _packed_segments(query, key, attention_mask, padding_2d, kwargs)

    # The flash kernel knows "plain causal (bottom-right aligned) / no mask",
    # including q_len < kv_len (chunked prefill with a KV cache), and padding via
    # unpadding onto the varlen kernel. Everything else goes to SDPA, with the
    # reason recorded in the fallback registry.
    reason, detail = _kernel_fallback_reason(module, query, key, value,
                                             None if padding_2d else attention_mask, dropout, kwargs)
    if reason is None and window is not None and padding_2d and not is_causal:
        # unpadding shifts bidirectional rows against their keys
        reason, detail = "attention_mask", " (padding with a non-causal sliding window)"
    window_size = (-1, -1) if window is None else (window - 1, window - 1)
    softcap = float(kwargs.get("softcap") or 0.0)  # None or 0: no softcapping (Gemma 2: c > 0)
    sinks = kwargs.get("s_aux")  # [H_q] attention sinks (gpt-oss) or None
    unpad = None
    if reason is None and padding_2d:  # the plan's host sync only when the kernel can run
        unpad = _unpad_plan(key_mask_2d, q_len, is_causal, query.device)
        if unpad is None:
            reason, detail = "attention_mask", " (padding inside a sequence)"

    attn_output = None
    if reason is None:
        # Our flash kernel (O(N) memory), batched or varlen. bf16 / fp32 inputs are
        # converted to fp16 by the API, which raises FP16RangeError on overflow.
        from .flash_attn import FP16RangeError, flash_attention, flash_attn_varlen_func

        try:
            if unpad is not None:
                # [B, H, n, d] -> rows [B * n, H, d] for the varlen kernel; query rows
                # left out of the plan (causal padding) get 0 (index_copy keeps autograd)
                q_idx, k_idx, cu_q, cu_k, max_q, max_k = unpad

                def rows(t, idx):
                    return t.transpose(1, 2).reshape(-1, t.shape[1], t.shape[3]).index_select(0, idx)
                out = flash_attn_varlen_func(rows(query, q_idx), rows(key, k_idx), rows(value, k_idx),
                                             cu_q, cu_k, max_q, max_k, softmax_scale=scaling,
                                             causal=is_causal, window_size=window_size, softcap=softcap,
                                             sinks=sinks, validate=False)  # built by _unpad_plan
                attn_output = (query.new_zeros(B * q_len, H_q, out.shape[-1]).index_copy(0, q_idx, out)
                               .view(B, q_len, H_q, -1).transpose(1, 2))
            elif segments is None:
                attn_output = flash_attention(query, key, value, softmax_scale=scaling, causal=is_causal,
                                              window_size=window_size, softcap=softcap, sinks=sinks)
            else:
                cu_seqlens, max_len = segments
                packed = [t[0].transpose(0, 1) for t in (query, key, value)]  # [N, H, d]
                attn_output = flash_attn_varlen_func(
                    *packed, cu_seqlens, cu_seqlens, max_len, max_len,
                    softmax_scale=scaling, causal=is_causal, window_size=window_size, softcap=softcap,
                    sinks=sinks, validate=False,  # validated / built by _packed_segments
                ).transpose(0, 1).unsqueeze(0)  # -> [1, H, N, d]
        except FP16RangeError:
            reason, detail = "fp16_overflow", f" ({query.dtype})"
        if attn_output is not None and query.dtype == torch.float32:
            _warn_once("fp32_compute", "flash_attn_legacy: float32 inputs are computed in "
                                       "float16 by the kernel (inference only)")

    if attn_output is None:
        action = _record_fallback(reason, detail)  # raises if no correct fallback exists
        if action == "sdpa" and (softcap or sinks is not None):
            action = "eager"  # SDPA would ignore the softcap / sinks
        if reason == "non_fp16_training" and not module.training:
            _warn_once("grad_in_eval", "flash_attn_legacy: %s inference with gradients enabled "
                       "runs on a fallback; wrap it in torch.inference_mode() to use the kernel",
                       query.dtype)

        # One mask for both fallbacks: boolean (True = keep) or additive float,
        # with bottom-right causality; `causal_only` = plain causal, q_len == kv_len.
        mask, causal_only = attention_mask, False
        # The window of a 2D-mask / kwarg layer (4D masks already carry theirs)
        band = window if (attention_mask is None or padding_2d) else None
        if band is not None and not is_causal:
            band_mask = (_causal_mask(q_len, kv_len, query.device, band)
                         | _causal_mask(kv_len, q_len, query.device, band).T)  # |i - j| < W
        elif band is not None or (is_causal and q_len > 1):
            band_mask = _causal_mask(q_len, kv_len, query.device, band)
        else:
            band_mask = None
        if mask is not None:
            mask = mask[..., :kv_len]
            if padding_2d and band_mask is not None:
                mask = mask & band_mask
            if padding_2d:
                # Query rows that see no key (left padding) attend everything instead,
                # as HF's SDPA mask does: fully masked rows give NaN on torch < 2.5,
                # and NaN in padded rows can reach valid ones in the next layer.
                mask = mask | ~mask.any(-1, keepdim=True)
        elif is_causal and q_len == kv_len and band is None:
            causal_only = True  # no N x N mask materialised for SDPA
        elif band_mask is not None:
            mask = band_mask
        if segments is not None:
            # Intersect, never replace: the existing mask may carry a sliding window.
            seg = _segment_mask(segments[0], q_len, is_causal, query.device)
            if mask is None:
                mask, causal_only = seg, False
            elif mask.dtype == torch.bool:
                mask = mask & seg
            else:  # additive (possibly broadcast-shaped): where() broadcasts both
                mask = torch.where(seg, mask, torch.finfo(mask.dtype).min)

        dropout_p = dropout if module.training else 0.0
        if action == "eager":
            if causal_only:
                mask = _causal_mask(q_len, kv_len, query.device)
            attn_output = _eager_attention(
                query, key, value, mask, scaling, dropout_p,
                softcap=softcap, sinks=sinks,
                position_bias=kwargs.get("position_bias"))
        else:
            # GQA via repeat_interleave, not enable_gqa: on sm_6x/7x there is no
            # flash SDPA backend and enable_gqa would push mem-efficient SDPA to
            # the O(N^2) math backend.
            if key.shape[1] != H_q:
                rep = H_q // key.shape[1]
                key = key.repeat_interleave(rep, dim=1)
                value = value.repeat_interleave(rep, dim=1)
            bias = kwargs.get("position_bias")
            if bias is not None:
                mask, causal_only = _add_bias_to_mask(
                    bias, _causal_mask(q_len, kv_len, query.device) if causal_only else mask,
                    kv_len, query.dtype)
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=mask, dropout_p=dropout_p,
                is_causal=causal_only, scale=scaling)

    # Transpose from [B, H, N, d] to [B, N, H, d] — matches SDPA output format
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None


def _add_bias_to_mask(bias, mask, kv_len, dtype):
    """SDPA attn_mask = additive position_bias (2D-4D, broadcastable to [B, H, q, kv],
    as T5's relative bias) + the mask (bool: True = keep, or additive), as HF's
    SDPA does for these models. Masked slots get the lowest finite value, not
    -inf (rows of only masked slots stay finite, as in eager). Differentiable in
    the bias. Returns (attn_mask, is_causal=False)."""
    lowest = torch.finfo(dtype).min
    bias = bias[..., :kv_len].to(dtype).clamp_min(lowest)  # -inf in the bias: finite too
    if mask is None:
        return bias, False
    if mask.dtype == torch.bool:
        return torch.where(mask, bias, lowest), False
    return (bias + mask.to(dtype)).clamp_min(lowest), False


def _flash_attn_legacy_mask(batch_size, q_length, kv_length, q_offset=0, kv_offset=0,
                            mask_function=None, attention_mask=None, local_size=None,
                            allow_is_causal_skip=True, allow_is_bidirectional_skip=False,
                            **kwargs):
    """Mask builder for transformers' ALL_MASK_ATTENTION_FUNCTIONS.

    For plain causal / bidirectional attention and causal sliding windows (when
    transformers allows skipping the mask) it returns:
      - None when every key slot is attended (-> kernel), or
      - causal only: a [batch, n_valid] mask tagged with _N_VALID_ATTR when only
        the first n_valid key slots hold tokens (pre-allocated static caches) ->
        flash_attn_legacy_forward slices K/V to n_valid and runs the kernel;
      - with padding: the [batch, n_valid] padding mask (untagged) -> unpadded
        onto the varlen kernel when the padding is contiguous per row;
      - causal sliding window (P4.1): always a [batch, n_valid] mask, as above,
        holding the window in its values (_WINDOW_ATTR) -> the kernel's window_size.
    Anything else gets the SDPA builder's boolean 4D mask (-> SDPA fallback).
    The SDPA builder is always called with both skips disabled: an SDPA `None`
    means "slice K/V to q_len" (causal, top-left) or "no mask at all", which is
    not what our kernel does with a pre-allocated cache (B1).
    """
    from transformers import masking_utils as mu

    if mask_function is None:
        mask_function = mu.causal_mask_function

    def sdpa_mask():
        return mu.ALL_MASK_ATTENTION_FUNCTIONS["sdpa"](
            batch_size=batch_size, q_length=q_length, kv_length=kv_length,
            q_offset=q_offset, kv_offset=kv_offset, mask_function=mask_function,
            attention_mask=attention_mask, local_size=local_size,
            allow_is_causal_skip=False, allow_is_bidirectional_skip=False, **kwargs)

    window = None
    if (allow_is_causal_skip and local_size is not None and mask_function is not mu.causal_mask_function
            and _is_sliding_causal(mask_function, local_size)):
        window = int(local_size)
    causal = mask_function is mu.causal_mask_function or window is not None
    bidirectional = mask_function is mu.bidirectional_mask_function
    skip_allowed = (causal and allow_is_causal_skip) or (bidirectional and allow_is_bidirectional_skip)
    if not skip_allowed:
        return sdpa_mask()  # packing, overlays, custom mask functions, bidirectional windows
    if window is None and local_size is not None and kv_length >= local_size:
        return sdpa_mask()  # a window we did not recognise
    # Static caches may pass offsets as 0-d tensors; reading them syncs (see P6.5).
    q_off, kv_off = int(q_offset), int(kv_offset)
    if causal:
        # Self-attention: key slots [kv_off, q_off + q_length) hold tokens.
        n_valid = q_off + q_length - kv_off
        if not (q_length <= n_valid <= kv_length):
            return sdpa_mask()  # unexpected layout: let SDPA's mask decide
    else:
        # Bidirectional (encoders, cross-attention): all keys count, exactly as
        # SDPA's "no mask"; never slice K/V to q_length here.
        n_valid = kv_length
    if attention_mask is not None:
        pad = attention_mask[:, kv_off:kv_off + n_valid]
        if not bool(pad.all()):
            if pad.shape[-1] != n_valid:
                return sdpa_mask()  # mask shorter than the key slots: SDPA's builder pads it
            # Padding: a copy of the [batch, n_valid] mask (FA2's 2D convention),
            # marked as padded -> unpadded onto the varlen kernel (_unpad_plan), or
            # the fallback. Models that index the mask as 4D (e.g. Doge) do not
            # follow this convention: see README "Known integration corner cases".
            if window is None:
                pad = pad.to(torch.bool, copy=True)
            else:
                pad = pad.to(torch.bool).to(_WINDOW_DTYPE) * (window + 1)
                setattr(pad, _WINDOW_ATTR, window)
            setattr(pad, _PADDING_ATTR, True)
            return pad
    if n_valid == kv_length and window is None:
        return None
    device = kwargs.get("device", "cpu")
    if window is None:
        valid = torch.ones(1, 1, dtype=torch.bool, device=device).expand(batch_size, n_valid)
    else:
        valid = torch.full((1, 1), window + 1, dtype=_WINDOW_DTYPE, device=device).expand(batch_size, n_valid)
        setattr(valid, _WINDOW_ATTR, window)
    setattr(valid, _N_VALID_ATTR, n_valid)
    return valid


def register():
    """Register flash_attn_legacy in transformers' attention function registry.

    This does three things:
    1. Registers our forward function in ALL_ATTENTION_FUNCTIONS so models
       can dispatch to it via config._attn_implementation.
    1b. Registers a mask builder (_flash_attn_legacy_mask) so padded batches
       get a real attention mask.
    2. Patches _check_and_adjust_attn_implementation to skip the flash
       preload step for our implementation. It does not look at head_dim:
       kernel support is decided per attention call (_kernel_fallback_reason).
       The patch is needed because transformers assumes any implementation with "flash" in the name
       needs lazy_import_flash_attention(), which tries to load hub kernels
       or the flash_attn package — neither of which we use.

    After registration, users can do:
        model = AutoModelForCausalLM.from_pretrained(
            "model_name",
            attn_implementation="flash_attn_legacy",
            torch_dtype=torch.float16,
            device_map="cuda",
        )

    Returns True if registration succeeded, False otherwise.
    """
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel

        # Step 1: Register our attention function
        ALL_ATTENTION_FUNCTIONS.register("flash_attn_legacy", flash_attn_legacy_forward)

        # Step 1b: Register a mask builder. Without one, transformers builds no
        # mask at all for an unknown implementation and passes attention_mask=None
        # even for padded batches (padding silently ignored).
        try:
            from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
            ALL_MASK_ATTENTION_FUNCTIONS.register("flash_attn_legacy", _flash_attn_legacy_mask)
        except (ImportError, AttributeError):
            pass  # older transformers: masks are built per model

        # Step 2: Patch the validation to skip flash preload for our implementation.
        #
        # The issue: _check_and_adjust_attn_implementation calls
        # is_flash_attention_requested() which returns True for any name
        # containing "flash". This triggers lazy_import_flash_attention()
        # which tries to load the flash_attn pip package or a hub kernel,
        # neither of which we provide. We just need to bypass that preload.
        original_check = PreTrainedModel._check_and_adjust_attn_implementation

        def _patched_check(self, attn_implementation, is_init_check=False, allow_all_kernels=False):
            if attn_implementation == "flash_attn_legacy":
                return "flash_attn_legacy"
            return original_check(self, attn_implementation, is_init_check, allow_all_kernels)

        PreTrainedModel._check_and_adjust_attn_implementation = _patched_check
        return True
    except (ImportError, AttributeError):
        return False
