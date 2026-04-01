"""
Integration tests with HuggingFace Transformers.

Tests that flash_attn_legacy can be patched into real transformer models.
Uses tiny randomly-initialized models — no downloads required.

Run:  pytest tests/test_transformers.py -v
Requires: pip install -e ".[transformers,test]"
"""

import math
import pytest
import torch

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


def _get_device_sm():
    """Return SM version of current CUDA device."""
    if not torch.cuda.is_available():
        return 0
    props = torch.cuda.get_device_properties(0)
    return props.major * 10 + props.minor


# ---------------------------------------------------------------------------
# Tiny LLaMA config for testing (no download needed)
# ---------------------------------------------------------------------------

def _make_tiny_llama():
    """Create a tiny LLaMA model for testing. ~2M params, fits on any GPU."""
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=256,
        hidden_size=512,         # must be divisible by num_heads
        intermediate_size=1024,
        num_hidden_layers=2,
        num_attention_heads=8,   # head_dim = 512/8 = 64
        num_key_value_heads=2,   # GQA: 8 Q heads, 2 KV heads
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        torch_dtype=torch.float16,
    )
    model = LlamaForCausalLM(config).cuda().half()
    return model, config


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
class TestTransformersIntegration:

    def test_patch_llama_forward(self):
        """Patch LLaMA attention and run a forward pass."""
        from examples.hf_llama_patch import patch_llama_attention

        model, config = _make_tiny_llama()
        model.eval()

        # Forward BEFORE patch (reference)
        input_ids = torch.randint(0, config.vocab_size, (1, 64), device='cuda')
        with torch.no_grad():
            ref_out = model(input_ids).logits.clone()

        # Patch and forward again
        patch_llama_attention(model)
        with torch.no_grad():
            patched_out = model(input_ids).logits

        # Both should produce valid output (not NaN/Inf)
        assert not torch.isnan(patched_out).any(), "NaN in patched output"
        assert not torch.isinf(patched_out).any(), "Inf in patched output"

        # Logits should be close (not identical due to FP16 attention vs FP32 native)
        # Allow larger tolerance because native HF uses FP32 attention internally
        cos_sim = torch.nn.functional.cosine_similarity(
            ref_out.float().view(1, -1),
            patched_out.float().view(1, -1),
        ).item()
        assert cos_sim > 0.95, f"Cosine similarity too low: {cos_sim:.4f}"

    def test_patch_llama_generate(self):
        """Patch LLaMA and run generate (autoregressive)."""
        from examples.hf_llama_patch import patch_llama_attention

        model, config = _make_tiny_llama()
        model.eval()
        patch_llama_attention(model)

        input_ids = torch.randint(0, config.vocab_size, (1, 8), device='cuda')
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=16, do_sample=False)

        assert out.shape[1] == 8 + 16, f"Expected 24 tokens, got {out.shape[1]}"
        assert not torch.isnan(out.float()).any()

    def test_patch_llama_backward(self):
        """Patch LLaMA and verify gradients flow through."""
        from examples.hf_llama_patch import patch_llama_attention

        model, config = _make_tiny_llama()
        model.train()
        patch_llama_attention(model)

        input_ids = torch.randint(0, config.vocab_size, (1, 32), device='cuda')
        labels = input_ids.clone()

        output = model(input_ids, labels=labels)
        loss = output.loss
        assert loss.item() > 0, "Loss should be positive"
        loss.backward()

        # Check that at least some parameters got gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad, "No gradients found — backward is broken"

    def test_gqa_head_mapping(self):
        """Verify GQA head mapping works with the tiny LLaMA config (8Q, 2KV)."""
        from flash_attn_legacy import flash_attention

        B, N, d = 1, 64, 64
        H_q, H_kv = 8, 2
        scale = 1.0 / math.sqrt(d)

        q = torch.randn(B, H_q, N, d, device='cuda', dtype=torch.float16) * 0.3
        k = torch.randn(B, H_kv, N, d, device='cuda', dtype=torch.float16) * 0.3
        v = torch.randn(B, H_kv, N, d, device='cuda', dtype=torch.float16) * 0.3

        out = flash_attention(q, k, v, softmax_scale=scale, is_causal=True)
        assert out.shape == (B, H_q, N, d)
        assert not torch.isnan(out).any()

    def test_different_seq_lengths(self):
        """Verify attention works with sequence lengths typical in LLaMA inference."""
        from examples.hf_llama_patch import patch_llama_attention

        model, config = _make_tiny_llama()
        model.eval()
        patch_llama_attention(model)

        for seq_len in [1, 16, 64, 128, 256]:
            input_ids = torch.randint(0, config.vocab_size, (1, seq_len), device='cuda')
            with torch.no_grad():
                out = model(input_ids)
            assert out.logits.shape == (1, seq_len, config.vocab_size), \
                f"Wrong output shape for seq_len={seq_len}"
            assert not torch.isnan(out.logits).any(), \
                f"NaN for seq_len={seq_len}"
