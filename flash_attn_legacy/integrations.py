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
    action: str = "sdpa"     # "sdpa": run SDPA instead; "raise": NotImplementedError
    expected: bool = False   # expected by design: counted but not logged
    hint: str = ""           # what to use instead (for action="raise")


FALLBACK_REASONS: Dict[str, FallbackReason] = {
    # -- not supported anywhere yet: SDPA would silently ignore them ----------
    "cu_seq_lens": FallbackReason(
        "packed/segmented attention via cu_seq_lens_q/k (e.g. VLM vision encoders)", "raise",
        hint="attn_implementation='sdpa' handles it (for a VLM it can be set for the vision "
             "sub-model only)"),
    "softcap": FallbackReason("attention logit softcapping (e.g. Gemma 2)", "raise",
                              hint="use attn_implementation='eager'"),
    "s_aux": FallbackReason("attention sinks s_aux (e.g. gpt-oss)", "raise",
                            hint="use attn_implementation='eager'"),
    "position_bias": FallbackReason("additive position_bias", "raise",
                                    hint="use attn_implementation='eager'"),
    # -- correct on SDPA, the kernel does not handle them (yet) ---------------
    "decode": FallbackReason("q_len == 1 (decode)", expected=True),
    "attention_mask": FallbackReason(
        "explicit attention mask (padding, sliding window, packing, ...)", expected=True),
    "q_longer_than_kv": FallbackReason("q_len > kv_len"),
    "head_dim": FallbackReason(f"head_dim not in {SUPPORTED_HEAD_DIMS}"),
    "v_head_dim": FallbackReason("V head_dim != Q head_dim (e.g. MLA)"),
    "dropout": FallbackReason("attention dropout in training mode"),
}

_fallback_counts: Counter = Counter()
_warned_reasons: set = set()


def _record_fallback(reason: str, detail: str = "") -> None:
    """Count a fallback; warn once per reason (unless expected); raise if the
    reason has no correct fallback."""
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
        logger.warning("flash_attn_legacy: falling back to SDPA: %s%s", info.message, detail)


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
    and the call either runs on PyTorch SDPA or raises NotImplementedError.

    Returns (attn_output, attn_weights) where attn_weights is always None.
    """
    from .flash_attn import flash_attention

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
        _record_fallback(reason, detail)  # raises for reasons without a correct fallback
        # GQA via repeat_interleave, not enable_gqa: on sm_6x/7x there is no flash
        # SDPA backend and enable_gqa would push mem-efficient SDPA to the O(N^2)
        # math backend.
        H_kv = key.shape[1]
        if H_kv != H_q:
            rep = H_q // H_kv
            key = key.repeat_interleave(rep, dim=1)
            value = value.repeat_interleave(rep, dim=1)

        mask, sdpa_causal = attention_mask, False
        if mask is not None:
            mask = mask[..., :kv_len]
        elif is_causal and q_len == kv_len:
            sdpa_causal = True  # no N x N mask materialised
        elif is_causal and q_len > 1:
            # Bottom-right aligned causal mask: query i sees keys <= i + (kv_len - q_len)
            mask = torch.ones(q_len, kv_len, dtype=torch.bool, device=query.device).tril(
                diagonal=kv_len - q_len
            )

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value,
            attn_mask=mask,
            dropout_p=dropout if module.training else 0.0,
            is_causal=sdpa_causal,
            scale=scaling,
        )

    # Transpose from [B, H, N, d] to [B, N, H, d] — matches SDPA output format
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None


def _flash_attn_legacy_mask(batch_size, q_length, kv_length, q_offset=0, kv_offset=0,
                            mask_function=None, attention_mask=None, local_size=None,
                            allow_is_causal_skip=True, **kwargs):
    """Mask builder for transformers' ALL_MASK_ATTENTION_FUNCTIONS.

    Returns None (-> flash kernel) when the mask is plain causal or plain
    bidirectional with no padding, no sliding window, and queries aligned to the
    end of the keys (this covers chunked prefill with a KV cache). Otherwise it
    delegates to the SDPA builder, whose boolean mask makes
    flash_attn_legacy_forward fall back to SDPA.
    """
    from transformers import masking_utils as mu

    sdpa_mask = mu.ALL_MASK_ATTENTION_FUNCTIONS["sdpa"]
    if mask_function is None:
        mask_function = mu.causal_mask_function
    call = dict(batch_size=batch_size, q_length=q_length, kv_length=kv_length,
                q_offset=q_offset, kv_offset=kv_offset, mask_function=mask_function,
                attention_mask=attention_mask, local_size=local_size,
                allow_is_causal_skip=allow_is_causal_skip, **kwargs)

    plain = mask_function in (mu.causal_mask_function,
                              getattr(mu, "bidirectional_mask_function", None))
    ints = all(isinstance(x, int) for x in (q_length, kv_length, q_offset, kv_offset))
    if not (allow_is_causal_skip and plain and ints and q_length <= kv_length):
        return sdpa_mask(**call)
    if local_size is not None and kv_length >= local_size:
        return sdpa_mask(**call)  # sliding window
    if q_offset + q_length != kv_offset + kv_length:
        return sdpa_mask(**call)  # queries not at the end of the keys
    if attention_mask is not None:
        pad = attention_mask[:, kv_offset:kv_offset + kv_length]
        if not bool(pad.all()):
            return sdpa_mask(**call)  # padding
    return None


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
