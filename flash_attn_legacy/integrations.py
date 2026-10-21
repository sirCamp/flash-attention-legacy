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

from .utils import SUPPORTED_HEAD_DIMS


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
    Uses flash_attn_legacy for unmasked prefill (1 < q_len <= kv_len, no
    attention_mask, no attention dropout) and falls back to PyTorch SDPA
    otherwise (decode, padded/masked batches).

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
    # including q_len < kv_len (chunked prefill with a KV cache). Decode (q_len=1)
    # goes to SDPA: a single query row would not fill the tiles anyway.
    # Any explicit attention_mask (padding, sliding window, ...) or a dropout
    # during training must NOT be silently dropped, so those go to SDPA.
    use_flash = (
        q_len > 1
        and q_len <= kv_len
        and d in SUPPORTED_HEAD_DIMS
        and attention_mask is None
        and not (dropout > 0.0 and module.training)
    )

    if use_flash:
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
        # Decode (q_len=1), padded/masked batches, chunked prefill or an
        # unsupported head_dim: use PyTorch SDPA (memory-efficient, mask-aware).
        H_kv = key.shape[1]
        if H_kv != H_q:
            rep = H_q // H_kv
            key = key.repeat_interleave(rep, dim=1)
            value = value.repeat_interleave(rep, dim=1)

        mask = attention_mask
        if mask is not None:
            mask = mask[..., :kv_len]
        elif is_causal and q_len > 1:
            # Bottom-right aligned causal mask: query i sees keys <= i + (kv_len - q_len)
            mask = torch.ones(q_len, kv_len, dtype=torch.bool, device=query.device).tril(
                diagonal=kv_len - q_len
            )

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value,
            attn_mask=mask,
            dropout_p=dropout if module.training else 0.0,
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
                # Check if model's head_dim is supported by our kernels.
                # If not, silently fall back to SDPA so mask preprocessing works correctly.
                head_dim = getattr(self.config, 'head_dim', None)
                if head_dim is None:
                    hs = getattr(self.config, 'hidden_size', None)
                    nh = getattr(self.config, 'num_attention_heads', None)
                    if hs and nh:
                        head_dim = hs // nh
                if head_dim is not None and head_dim not in SUPPORTED_HEAD_DIMS:
                    return original_check(self, "sdpa", is_init_check, allow_all_kernels)
                return "flash_attn_legacy"
            return original_check(self, attn_implementation, is_init_check, allow_all_kernels)

        PreTrainedModel._check_and_adjust_attn_implementation = _patched_check
        return True
    except (ImportError, AttributeError):
        return False
