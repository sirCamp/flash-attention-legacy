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

import math
import torch
import torch.nn as nn
from typing import Optional


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
    Uses flash_attn_legacy for prefill (q_len == kv_len) and falls back
    to standard attention for decode (q_len == 1, kv_len > 1) since our
    kernel requires matching sequence lengths.

    Returns (attn_output, attn_weights) where attn_weights is always None.
    """
    from .flash_attn import flash_attention

    # query/key/value come in as [B, H, N, d] from the model's attention layer
    B, H_q, q_len, d = query.shape
    kv_len = key.shape[2]

    if scaling is None:
        scaling = 1.0 / math.sqrt(d)

    if q_len == kv_len and d in (64, 128):
        # Prefill: use our flash kernel (O(N) memory)
        q_fp16 = query.half() if query.dtype != torch.float16 else query
        k_fp16 = key.half() if key.dtype != torch.float16 else key
        v_fp16 = value.half() if value.dtype != torch.float16 else value

        attn_output = flash_attention(
            q_fp16, k_fp16, v_fp16,
            softmax_scale=scaling,
            is_causal=True,
        )

        if query.dtype != torch.float16:
            attn_output = attn_output.to(query.dtype)
    else:
        # Decode (q_len=1) or unsupported head_dim: standard attention
        H_kv = key.shape[1]
        if H_kv != H_q:
            rep = H_q // H_kv
            key = key.repeat_interleave(rep, dim=1)
            value = value.repeat_interleave(rep, dim=1)

        s = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scaling

        if attention_mask is not None:
            s = s + attention_mask

        attn_output = torch.matmul(
            torch.softmax(s, dim=-1), value.float()
        ).to(query.dtype)

    return attn_output, None


def register():
    """Register flash_attn_legacy in transformers' attention function registry.

    This does two things:
    1. Registers our forward function in ALL_ATTENTION_FUNCTIONS so models
       can dispatch to it via config._attn_implementation.
    2. Patches _check_and_adjust_attn_implementation to skip the flash
       preload step for our implementation. This is needed because
       transformers assumes any implementation with "flash" in the name
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
                # Our implementation is already registered and ready — skip
                # the preload step that would fail looking for hub kernels
                return "flash_attn_legacy"
            return original_check(self, attn_implementation, is_init_check, allow_all_kernels)

        PreTrainedModel._check_and_adjust_attn_implementation = _patched_check
        return True
    except (ImportError, AttributeError):
        return False
