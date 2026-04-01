"""
Example: Integrating flash_attn_legacy with HuggingFace Transformers

This shows how to monkey-patch a LLaMA model to use flash_attn_legacy
for inference on V100/P100 GPUs where the standard FlashAttention2 backend
doesn't work.

Usage:
    python examples/hf_llama_patch.py --model meta-llama/Llama-2-7b-hf --prompt "Hello"
"""

import argparse
import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


def patch_llama_attention(model):
    """
    Replaces LLaMA's attention forward with flash_attn_legacy.
    Handles GQA (LLaMA 2/3 uses num_kv_heads < num_heads).
    """
    from flash_attn_legacy import flash_attention

    def make_flash_forward(original_self):
        """Create a patched forward method for one attention layer."""

        def flash_forward(
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value=None,
            past_key_values=None,
            output_attentions: bool = False,
            use_cache: bool = False,
            position_embeddings: Optional[Tuple] = None,
            **kwargs,
        ):
            bsz, q_len, _ = hidden_states.size()
            attn = original_self

            # Resolve KV cache — transformers <5 uses past_key_value,
            # transformers >=5 uses past_key_values (DynamicCache)
            cache = past_key_values if past_key_values is not None else past_key_value

            q = attn.q_proj(hidden_states)
            k = attn.k_proj(hidden_states)
            v = attn.v_proj(hidden_states)

            # Compatible with both transformers <5 and >=5
            num_heads = getattr(attn, 'num_heads', None) or attn.config.num_attention_heads
            num_kv_heads = getattr(attn, 'num_key_value_heads', None) or attn.config.num_key_value_heads
            head_dim = getattr(attn, 'head_dim', None) or (attn.config.hidden_size // num_heads)

            q = q.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
            v = v.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

            # Apply rotary embeddings — transformers >=5 passes position_embeddings
            if position_embeddings is not None:
                cos, sin = position_embeddings
                from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
            elif hasattr(attn, 'rotary_emb'):
                cos, sin = attn.rotary_emb(v, position_ids)
                from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
                q, k = apply_rotary_pos_emb(q, k, cos, sin)

            # KV cache — handle both DynamicCache (v5) and tuple (v4)
            if cache is not None and hasattr(cache, 'update'):
                # transformers >=5 DynamicCache
                k, v = cache.update(k, v, attn.layer_idx)
            elif cache is not None:
                # transformers <5 tuple cache
                k = torch.cat([cache[0], k], dim=2)
                v = torch.cat([cache[1], v], dim=2)
                if use_cache:
                    cache = (k, v)

            scale = 1.0 / math.sqrt(head_dim)
            kv_len = k.shape[2]

            if q_len == kv_len:
                # Prefill: use flash attention (O(N) memory)
                q_fp16 = q.half() if q.dtype != torch.float16 else q
                k_fp16 = k.half() if k.dtype != torch.float16 else k
                v_fp16 = v.half() if v.dtype != torch.float16 else v

                attn_output = flash_attention(
                    q_fp16, k_fp16, v_fp16,
                    softmax_scale=scale,
                    is_causal=True,
                )
            else:
                # Decode (q_len=1): standard attention is fine, no N² issue
                # Expand KV heads for GQA
                if num_kv_heads != num_heads:
                    rep = num_heads // num_kv_heads
                    k = k.repeat_interleave(rep, dim=1)
                    v = v.repeat_interleave(rep, dim=1)
                s = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
                attn_output = torch.matmul(torch.softmax(s, dim=-1), v.float()).to(q.dtype)

            if hidden_states.dtype != torch.float16:
                attn_output = attn_output.to(hidden_states.dtype)

            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
            attn_output = attn.o_proj(attn_output)

            # transformers >=5 expects (attn_output, attn_weights)
            # transformers <5 expects (attn_output, attn_weights, past_key_value)
            return attn_output, None

        return flash_forward

    # Find and patch all attention layers
    patched = 0
    for name, module in model.named_modules():
        if module.__class__.__name__ in ('LlamaAttention', 'LlamaSdpaAttention', 'LlamaFlashAttention2'):
            module.forward = make_flash_forward(module)
            patched += 1

    print(f"Patched {patched} attention layers with flash_attn_legacy")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--prompt', type=str, default='The meaning of life is')
    parser.add_argument('--max-new-tokens', type=int, default=50)
    args = parser.parse_args()

    from flash_attn_legacy import check_gpu_compatibility
    check_gpu_compatibility()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map='cuda',
    )

    # Patch attention
    model = patch_llama_attention(model)

    # Generate
    inputs = tokenizer(args.prompt, return_tensors='pt').to('cuda')
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7,
        )
    print(f"\n{tokenizer.decode(outputs[0], skip_special_tokens=True)}")


if __name__ == '__main__':
    main()
