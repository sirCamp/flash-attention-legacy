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

from .utils import SUPPORTED_HEAD_DIMS

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
    # "sdpa": torch SDPA; "eager": _eager_attention (applies softcap / sinks /
    # position_bias, which SDPA would ignore); "raise": NotImplementedError
    action: str = "sdpa"
    expected: bool = False   # expected by design: counted but not logged
    hint: str = ""           # what to use instead (for action="raise")


FALLBACK_REASONS: Dict[str, FallbackReason] = {
    # -- not supported anywhere yet: SDPA would silently ignore them ----------
    "cu_seq_lens": FallbackReason(
        "packed/segmented attention via cu_seq_lens_q/k (e.g. VLM vision encoders)", "raise",
        hint="attn_implementation='sdpa' handles it (for a VLM it can be set for the vision "
             "sub-model only)"),
    # -- SDPA would ignore them: eager fallback (temporary, docs/PLAN.md §3b) --
    "softcap": FallbackReason("attention logit softcapping (e.g. Gemma 2)", "eager"),
    "s_aux": FallbackReason("attention sinks s_aux (e.g. gpt-oss)", "eager"),
    "position_bias": FallbackReason("additive position_bias", "eager"),
    # -- correct on SDPA, the kernel does not handle them (yet) ---------------
    "decode": FallbackReason("q_len == 1 (decode)", expected=True),
    "attention_mask": FallbackReason(
        "explicit attention mask (padding, sliding window, packing, ...)", expected=True),
    "q_longer_than_kv": FallbackReason("q_len > kv_len"),
    "head_dim": FallbackReason(f"head_dim not in {SUPPORTED_HEAD_DIMS}"),
    "v_head_dim": FallbackReason("V head_dim != Q head_dim (e.g. MLA)"),
    "dropout": FallbackReason("attention dropout in training mode"),
}

# 2D masks follow HF's flash_attention_2 convention: a [B, n] mask means only the
# first n key slots hold tokens (the rest are empty slots of a pre-allocated
# cache) and zeros inside it are padding. They come from _flash_attn_legacy_mask
# or straight from models that pass HF's padding mask to "flash" backends.
# _flash_attn_legacy_mask tags its all-True masks with this attribute so the
# forward can skip a device sync; the attribute is only a fast path (it is lost
# if transformers copies the mask), never needed for correctness.
_N_VALID_ATTR = "_flash_attn_legacy_n_valid"

_fallback_counts: Counter = Counter()
_warned_reasons: set = set()


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
    if not info.expected and reason not in _warned_reasons:
        _warned_reasons.add(reason)
        logger.warning("flash_attn_legacy: falling back to %s attention: %s%s",
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
    for name in ("cu_seq_lens_q", "cu_seq_lens_k"):
        if kwargs.get(name) is not None:
            return "cu_seq_lens", ""
    softcap = kwargs.get("softcap")
    if softcap is not None and float(softcap) != 0.0:  # None or 0.0: no softcapping
        return "softcap", f" (softcap={softcap})"
    if kwargs.get("s_aux") is not None:
        return "s_aux", ""
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
    return None, ""


def _causal_mask(q_len: int, kv_len: int, device) -> torch.Tensor:
    """Bottom-right aligned causal mask (True = keep): query i sees keys
    j <= i + (kv_len - q_len), the convention of the kernel."""
    return torch.ones(q_len, kv_len, dtype=torch.bool, device=device).tril(diagonal=kv_len - q_len)


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
    (no mask, 1 < q_len <= kv_len, supported head_dim, no unsupported kwargs).
    Otherwise the reason is recorded in FALLBACK_REASONS (see fallback_stats())
    and the call runs on PyTorch SDPA, on _eager_attention (features SDPA would
    ignore: softcap, sinks, position_bias) or raises NotImplementedError.

    Returns (attn_output, attn_weights) where attn_weights is always None.
    """
    from .flash_attn import flash_attention

    padding_2d = False
    if attention_mask is not None and attention_mask.dim() == 2:
        # [B, n] mask (see _N_VALID_ATTR): keep the first n key slots only.
        n_valid = attention_mask.shape[-1]
        if n_valid > key.shape[2]:
            raise RuntimeError(f"flash_attn_legacy: 2D mask covers {n_valid} keys, "
                               f"but only {key.shape[2]} keys were passed")
        key, value = key[:, :, :n_valid], value[:, :, :n_valid]
        if getattr(attention_mask, _N_VALID_ATTR, None) == n_valid or bool(attention_mask.all()):
            attention_mask = None  # no padding
        else:
            # Padding (1 = keep): broadcastable boolean mask; causality is added in
            # the fallback (native unpadding comes with docs/PLAN.md P3.1).
            attention_mask = attention_mask[:, None, None, :].bool()
            padding_2d = True

    # query/key/value come in as [B, H, N, d] from the model's attention layer
    B, H_q, q_len, d = query.shape
    kv_len = key.shape[2]

    if scaling is None:
        scaling = 1.0 / math.sqrt(d)

    # Detect causal: use module.is_causal (set by the model's attention layer),
    # with kwargs override. Matches SDPA behavior.
    if "is_causal" in kwargs:
        is_causal = kwargs["is_causal"]
    elif hasattr(module, 'is_causal'):
        is_causal = module.is_causal
    elif attention_mask is None and q_len > 1:
        is_causal = True
    else:
        is_causal = False

    # The flash kernel knows "plain causal (bottom-right aligned) / no mask",
    # including q_len < kv_len (chunked prefill with a KV cache). Everything
    # else goes to SDPA, with the reason recorded in the fallback registry.
    reason, detail = _kernel_fallback_reason(module, query, key, value, attention_mask,
                                             dropout, kwargs)

    if reason is None:
        # Prefill: use our flash kernel (O(N) memory)
        q_fp16 = query.half() if query.dtype != torch.float16 else query
        k_fp16 = key.half() if key.dtype != torch.float16 else key
        v_fp16 = value.half() if value.dtype != torch.float16 else value

        attn_output = flash_attention(
            q_fp16, k_fp16, v_fp16,
            softmax_scale=scaling,
            causal=is_causal,
        )

        if query.dtype != torch.float16:
            attn_output = attn_output.to(query.dtype)
    else:
        action = _record_fallback(reason, detail)  # raises if no correct fallback exists

        # One mask for both fallbacks: boolean (True = keep) or additive float,
        # with bottom-right causality; `causal_only` = plain causal, q_len == kv_len.
        mask, causal_only = attention_mask, False
        if mask is not None:
            mask = mask[..., :kv_len]
            if padding_2d and is_causal and q_len > 1:
                mask = mask & _causal_mask(q_len, kv_len, query.device)
        elif is_causal and q_len == kv_len:
            causal_only = True  # no N x N mask materialised for SDPA
        elif is_causal and q_len > 1:
            mask = _causal_mask(q_len, kv_len, query.device)

        dropout_p = dropout if module.training else 0.0
        if action == "eager":
            if causal_only:
                mask = _causal_mask(q_len, kv_len, query.device)
            attn_output = _eager_attention(
                query, key, value, mask, scaling, dropout_p,
                softcap=kwargs.get("softcap"), sinks=kwargs.get("s_aux"),
                position_bias=kwargs.get("position_bias"))
        else:
            # GQA via repeat_interleave, not enable_gqa: on sm_6x/7x there is no
            # flash SDPA backend and enable_gqa would push mem-efficient SDPA to
            # the O(N^2) math backend.
            if key.shape[1] != H_q:
                rep = H_q // key.shape[1]
                key = key.repeat_interleave(rep, dim=1)
                value = value.repeat_interleave(rep, dim=1)
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=mask, dropout_p=dropout_p,
                is_causal=causal_only, scale=scaling)

    # Transpose from [B, H, N, d] to [B, N, H, d] — matches SDPA output format
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None


def _flash_attn_legacy_mask(batch_size, q_length, kv_length, q_offset=0, kv_offset=0,
                            mask_function=None, attention_mask=None, local_size=None,
                            allow_is_causal_skip=True, allow_is_bidirectional_skip=False,
                            **kwargs):
    """Mask builder for transformers' ALL_MASK_ATTENTION_FUNCTIONS.

    For plain causal / bidirectional attention with no padding and no sliding
    window (and when transformers allows skipping the mask) it returns:
      - None when every key slot is attended (-> kernel), or
      - causal only: a [batch, n_valid] mask tagged with _N_VALID_ATTR when only
        the first n_valid key slots hold tokens (pre-allocated static caches) ->
        flash_attn_legacy_forward slices K/V to n_valid and runs the kernel.
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

    causal = mask_function is mu.causal_mask_function
    bidirectional = mask_function is mu.bidirectional_mask_function
    skip_allowed = (causal and allow_is_causal_skip) or (bidirectional and allow_is_bidirectional_skip)
    if not skip_allowed:
        return sdpa_mask()  # sliding window, packing, overlays, custom mask functions
    if local_size is not None and kv_length >= local_size:
        return sdpa_mask()  # sliding window
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
            return sdpa_mask()  # padding
    if n_valid == kv_length:
        return None
    device = kwargs.get("device", "cpu")
    valid = torch.ones(1, 1, dtype=torch.bool, device=device).expand(batch_size, n_valid)
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
