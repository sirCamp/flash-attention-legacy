"""
Tests for flash_attn_legacy — correctness against PyTorch reference.

Run:  pytest tests/test_flash_attn.py -v
      pytest tests/test_flash_attn.py -v -k "gqa"     # GQA tests only
"""

import math
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------

def ref_attention(q, k, v, scale, is_causal=False):
    """Standard attention in FP32. Handles GQA by repeating K/V heads."""
    H_q, H_kv = q.shape[1], k.shape[1]
    if H_kv < H_q:
        reps = H_q // H_kv
        k = k.repeat_interleave(reps, dim=1)
        v = v.repeat_interleave(reps, dim=1)
    s = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    if is_causal:
        N = q.shape[-2]
        mask = torch.triu(torch.ones(N, N, device=q.device, dtype=torch.bool), diagonal=1)
        s.masked_fill_(mask, float('-inf'))
    p = torch.softmax(s, dim=-1)
    return torch.matmul(p, v.float()).half()


def ref_backward(q, k, v, grad_out, scale, is_causal=False):
    """Reference backward in FP32. Returns dQ [H_q], dK [H_kv], dV [H_kv]."""
    H_q, H_kv = q.shape[1], k.shape[1]
    reps = H_q // H_kv

    q_f = q.float().detach().requires_grad_(True)
    # Expand K/V for GQA
    k_f = k.float().detach().repeat_interleave(reps, dim=1).requires_grad_(True)
    v_f = v.float().detach().repeat_interleave(reps, dim=1).requires_grad_(True)

    s = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale
    if is_causal:
        N = q.shape[-2]
        mask = torch.triu(torch.ones(N, N, device=q.device, dtype=torch.bool), diagonal=1)
        s.masked_fill_(mask, float('-inf'))
    p = torch.softmax(s, dim=-1)
    o = torch.matmul(p, v_f)
    o.backward(grad_out.float())

    dq = q_f.grad.half()
    # Sum expanded dK/dV back to H_kv heads
    dk = k_f.grad.view(k.shape[0], H_kv, reps, k.shape[2], k.shape[3]).sum(dim=2).half()
    dv = v_f.grad.view(v.shape[0], H_kv, reps, v.shape[2], v.shape[3]).sum(dim=2).half()
    return dq, dk, dv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=[64, 128], ids=["d64", "d128"])
def head_dim(request):
    return request.param


@pytest.fixture(params=[False, True], ids=["non_causal", "causal"])
def is_causal(request):
    return request.param


# ---------------------------------------------------------------------------
# Forward — standard MHA
# ---------------------------------------------------------------------------

class TestForwardMHA:

    @pytest.mark.parametrize("N", [32, 64, 128, 256, 512])
    def test_seq_lengths(self, head_dim, is_causal, N):
        from flash_attn_legacy import flash_attention
        B, H, d = 2, 4, head_dim
        scale = 1.0 / math.sqrt(d)
        q = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16) * 0.3
        k, v = torch.randn_like(q) * 0.3, torch.randn_like(q) * 0.3

        out = flash_attention(q, k, v, softmax_scale=scale, is_causal=is_causal)
        ref = ref_attention(q, k, v, scale, is_causal)
        torch.testing.assert_close(out, ref, rtol=2e-2, atol=5e-3)

    @pytest.mark.parametrize("N", [33, 65, 100, 200])
    def test_non_power_of_2(self, head_dim, N):
        from flash_attn_legacy import flash_attention
        B, H, d = 1, 2, head_dim
        scale = 1.0 / math.sqrt(d)
        q = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16) * 0.2
        k, v = torch.randn_like(q) * 0.2, torch.randn_like(q) * 0.2
        out = flash_attention(q, k, v, softmax_scale=scale)
        ref = ref_attention(q, k, v, scale)
        torch.testing.assert_close(out, ref, rtol=3e-2, atol=1e-2)

    def test_deterministic(self, head_dim):
        from flash_attn_legacy import flash_attention
        B, H, N, d = 1, 4, 256, head_dim
        q = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16)
        k, v = torch.randn_like(q), torch.randn_like(q)
        assert torch.equal(flash_attention(q, k, v), flash_attention(q, k, v))

    def test_single_token(self, head_dim):
        from flash_attn_legacy import flash_attention
        B, H, d = 2, 4, head_dim
        q = torch.randn(B, H, 1, d, device='cuda', dtype=torch.float16)
        k, v = torch.randn_like(q), torch.randn_like(q)
        out = flash_attention(q, k, v)
        torch.testing.assert_close(out, v, rtol=1e-2, atol=1e-3)

    def test_uniform_attention(self, head_dim):
        from flash_attn_legacy import flash_attention
        B, H, N, d = 1, 1, 64, head_dim
        q = torch.zeros(B, H, N, d, device='cuda', dtype=torch.float16)
        k = torch.zeros(B, H, N, d, device='cuda', dtype=torch.float16)
        v = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16) * 0.1
        out = flash_attention(q, k, v)
        expected = v.float().mean(dim=2, keepdim=True).expand_as(v).half()
        torch.testing.assert_close(out, expected, rtol=2e-2, atol=5e-3)

    def test_batch_independence(self, head_dim):
        from flash_attn_legacy import flash_attention
        B, H, N, d = 4, 2, 128, head_dim
        scale = 1.0 / math.sqrt(d)
        q = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16) * 0.3
        k, v = torch.randn_like(q) * 0.3, torch.randn_like(q) * 0.3
        full_out = flash_attention(q, k, v, softmax_scale=scale)
        for b in range(B):
            single = flash_attention(q[b:b+1], k[b:b+1], v[b:b+1], softmax_scale=scale)
            torch.testing.assert_close(full_out[b:b+1], single, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Forward — GQA and MQA
# ---------------------------------------------------------------------------

class TestForwardGQA:

    @pytest.mark.parametrize("H_q,H_kv", [(8, 2), (8, 4), (8, 1), (4, 1), (4, 2)])
    def test_gqa_correctness(self, head_dim, H_q, H_kv):
        """GQA forward matches reference with repeated K/V heads."""
        from flash_attn_legacy import flash_attention
        B, N, d = 2, 128, head_dim
        scale = 1.0 / math.sqrt(d)

        q = torch.randn(B, H_q, N, d, device='cuda', dtype=torch.float16) * 0.3
        k = torch.randn(B, H_kv, N, d, device='cuda', dtype=torch.float16) * 0.3
        v = torch.randn(B, H_kv, N, d, device='cuda', dtype=torch.float16) * 0.3

        out = flash_attention(q, k, v, softmax_scale=scale)
        ref = ref_attention(q, k, v, scale)
        torch.testing.assert_close(out, ref, rtol=3e-2, atol=1e-2)

    @pytest.mark.parametrize("H_q,H_kv", [(8, 2), (8, 1)])
    def test_gqa_causal(self, head_dim, H_q, H_kv):
        """GQA with causal mask."""
        from flash_attn_legacy import flash_attention
        B, N, d = 1, 256, head_dim
        scale = 1.0 / math.sqrt(d)

        q = torch.randn(B, H_q, N, d, device='cuda', dtype=torch.float16) * 0.3
        k = torch.randn(B, H_kv, N, d, device='cuda', dtype=torch.float16) * 0.3
        v = torch.randn(B, H_kv, N, d, device='cuda', dtype=torch.float16) * 0.3

        out = flash_attention(q, k, v, softmax_scale=scale, is_causal=True)
        ref = ref_attention(q, k, v, scale, is_causal=True)
        torch.testing.assert_close(out, ref, rtol=3e-2, atol=1e-2)

    def test_mqa_equals_broadcast(self, head_dim):
        """MQA (H_kv=1) should equal broadcasting K/V to all Q heads."""
        from flash_attn_legacy import flash_attention
        B, H_q, N, d = 2, 8, 128, head_dim
        scale = 1.0 / math.sqrt(d)

        q = torch.randn(B, H_q, N, d, device='cuda', dtype=torch.float16) * 0.3
        k = torch.randn(B, 1, N, d, device='cuda', dtype=torch.float16) * 0.3
        v = torch.randn(B, 1, N, d, device='cuda', dtype=torch.float16) * 0.3

        out_mqa = flash_attention(q, k, v, softmax_scale=scale)
        # Compare with explicit broadcast
        k_exp = k.expand(B, H_q, N, d)
        v_exp = v.expand(B, H_q, N, d)
        out_mha = flash_attention(q, k_exp, v_exp, softmax_scale=scale)

        torch.testing.assert_close(out_mqa, out_mha, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Backward — MHA
# ---------------------------------------------------------------------------

class TestBackwardMHA:

    def test_gradient_flow(self, head_dim, is_causal):
        from flash_attn_legacy import flash_attention
        B, H, N, d = 1, 2, 64, head_dim

        q = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16, requires_grad=True)
        k = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16, requires_grad=True)
        v = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16, requires_grad=True)

        flash_attention(q, k, v, is_causal=is_causal).sum().backward()
        assert q.grad is not None and not torch.all(q.grad == 0)
        assert k.grad is not None and not torch.all(k.grad == 0)
        assert v.grad is not None and not torch.all(v.grad == 0)

    @pytest.mark.parametrize("N", [32, 64, 128])
    def test_gradient_correctness(self, head_dim, is_causal, N):
        from flash_attn_legacy import flash_attention
        B, H, d = 1, 2, head_dim
        scale = 1.0 / math.sqrt(d)

        q = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16) * 0.2
        k = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16) * 0.2
        v = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16) * 0.2
        go = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16) * 0.1

        q.requires_grad_(True); k.requires_grad_(True); v.requires_grad_(True)
        flash_attention(q, k, v, softmax_scale=scale, is_causal=is_causal).backward(go)
        dq_f, dk_f, dv_f = q.grad.clone(), k.grad.clone(), v.grad.clone()

        dq_r, dk_r, dv_r = ref_backward(q.detach(), k.detach(), v.detach(), go, scale, is_causal)

        torch.testing.assert_close(dq_f, dq_r, rtol=0.1, atol=0.02)
        torch.testing.assert_close(dk_f, dk_r, rtol=0.1, atol=0.02)
        torch.testing.assert_close(dv_f, dv_r, rtol=0.1, atol=0.02)


# ---------------------------------------------------------------------------
# Backward — GQA
# ---------------------------------------------------------------------------

class TestBackwardGQA:

    @pytest.mark.parametrize("H_q,H_kv", [(8, 2), (4, 1)])
    def test_gqa_gradient_shapes(self, head_dim, H_q, H_kv):
        """dQ has H_q heads, dK/dV have H_kv heads."""
        from flash_attn_legacy import flash_attention
        B, N, d = 1, 64, head_dim

        q = torch.randn(B, H_q, N, d, device='cuda', dtype=torch.float16, requires_grad=True)
        k = torch.randn(B, H_kv, N, d, device='cuda', dtype=torch.float16, requires_grad=True)
        v = torch.randn(B, H_kv, N, d, device='cuda', dtype=torch.float16, requires_grad=True)

        flash_attention(q, k, v).sum().backward()

        assert q.grad.shape == (B, H_q, N, d)
        assert k.grad.shape == (B, H_kv, N, d)
        assert v.grad.shape == (B, H_kv, N, d)

    @pytest.mark.parametrize("H_q,H_kv", [(8, 2), (4, 1)])
    def test_gqa_gradient_correctness(self, head_dim, H_q, H_kv):
        """GQA backward matches reference."""
        from flash_attn_legacy import flash_attention
        B, N, d = 1, 64, head_dim
        scale = 1.0 / math.sqrt(d)

        q = torch.randn(B, H_q, N, d, device='cuda', dtype=torch.float16) * 0.2
        k = torch.randn(B, H_kv, N, d, device='cuda', dtype=torch.float16) * 0.2
        v = torch.randn(B, H_kv, N, d, device='cuda', dtype=torch.float16) * 0.2
        go = torch.randn(B, H_q, N, d, device='cuda', dtype=torch.float16) * 0.1

        q.requires_grad_(True); k.requires_grad_(True); v.requires_grad_(True)
        flash_attention(q, k, v, softmax_scale=scale).backward(go)
        dq_f, dk_f, dv_f = q.grad.clone(), k.grad.clone(), v.grad.clone()

        dq_r, dk_r, dv_r = ref_backward(q.detach(), k.detach(), v.detach(), go, scale)

        torch.testing.assert_close(dq_f, dq_r, rtol=0.15, atol=0.03)
        torch.testing.assert_close(dk_f, dk_r, rtol=0.15, atol=0.03)
        torch.testing.assert_close(dv_f, dv_r, rtol=0.15, atol=0.03)


# ---------------------------------------------------------------------------
# Module tests
# ---------------------------------------------------------------------------

class TestModules:

    def test_mha_module(self):
        from flash_attn_legacy import FlashMultiHeadAttention
        mha = FlashMultiHeadAttention(512, num_heads=8, causal=True).cuda().half()
        x = torch.randn(2, 128, 512, device='cuda', dtype=torch.float16)
        out = mha(x)
        assert out.shape == x.shape

    def test_gqa_module(self):
        from flash_attn_legacy import FlashMultiHeadAttention
        gqa = FlashMultiHeadAttention(512, num_heads=8, num_kv_heads=2, causal=True).cuda().half()
        x = torch.randn(2, 128, 512, device='cuda', dtype=torch.float16)
        out = gqa(x)
        assert out.shape == x.shape

    def test_mqa_module(self):
        from flash_attn_legacy import FlashMultiHeadAttention
        mqa = FlashMultiHeadAttention(512, num_heads=8, num_kv_heads=1).cuda().half()
        x = torch.randn(2, 128, 512, device='cuda', dtype=torch.float16)
        out = mqa(x)
        assert out.shape == x.shape

    def test_module_backward(self):
        from flash_attn_legacy import FlashMultiHeadAttention
        mha = FlashMultiHeadAttention(512, num_heads=8, num_kv_heads=2).cuda().half()
        x = torch.randn(2, 64, 512, device='cuda', dtype=torch.float16, requires_grad=True)
        mha(x).sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------

class TestUtils:

    def test_device_info(self):
        from flash_attn_legacy import get_device_info
        info = get_device_info()
        assert 'name' in info and info['sm_version'] >= 30

    def test_compat_check(self):
        from flash_attn_legacy import check_gpu_compatibility
        assert isinstance(check_gpu_compatibility(verbose=False), bool)


# ---------------------------------------------------------------------------
# Long sequence tests — the real use case for Flash Attention
# ---------------------------------------------------------------------------

class TestLongSequences:
    """Flash Attention's value is on long sequences. Test them explicitly."""

    @pytest.mark.parametrize("N", [1024, 2048, 4096])
    def test_long_seq_forward(self, head_dim, N):
        from flash_attn_legacy import flash_attention
        B, H, d = 1, 4, head_dim
        scale = 1.0 / math.sqrt(d)
        q = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16) * 0.3
        k, v = torch.randn_like(q) * 0.3, torch.randn_like(q) * 0.3

        out = flash_attention(q, k, v, softmax_scale=scale, is_causal=True)
        assert out.shape == (B, H, N, d)
        assert not torch.isnan(out).any(), f"NaN at N={N}, d={d}"
        assert not torch.isinf(out).any(), f"Inf at N={N}, d={d}"

        # Spot check: compare first 256 positions against reference
        ref = ref_attention(
            q[:, :, :256], k[:, :, :256], v[:, :, :256], scale, is_causal=True
        )
        torch.testing.assert_close(out[:, :, :256], ref, rtol=3e-2, atol=1e-2)

    @pytest.mark.parametrize("N", [1024, 2048])
    def test_long_seq_backward(self, head_dim, N):
        from flash_attn_legacy import flash_attention
        B, H, d = 1, 2, head_dim
        scale = 1.0 / math.sqrt(d)

        q = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16) * 0.2
        k = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16) * 0.2
        v = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16) * 0.2
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)

        out = flash_attention(q, k, v, softmax_scale=scale, is_causal=True)
        out.sum().backward()

        assert q.grad is not None and not torch.isnan(q.grad).any()
        assert k.grad is not None and not torch.isnan(k.grad).any()
        assert v.grad is not None and not torch.isnan(v.grad).any()

    @pytest.mark.slow
    def test_very_long_seq(self):
        """8K sequence — should work without OOM thanks to O(N) memory."""
        from flash_attn_legacy import flash_attention
        B, H, N, d = 1, 4, 8192, 64

        q = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16) * 0.2
        k, v = torch.randn_like(q) * 0.2, torch.randn_like(q) * 0.2

        torch.cuda.reset_peak_memory_stats()
        out = flash_attention(q, k, v, is_causal=True)
        torch.cuda.synchronize()

        assert out.shape == (B, H, N, d)
        assert not torch.isnan(out).any()

        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        # Standard attention would need N*N*H*B*4 = 8192^2*4*1*4 ~ 1 GB for S matrix
        # Flash should use much less — we just check it didn't blow up
        assert peak_mb < 500, f"Peak memory {peak_mb:.0f} MB seems too high for flash attention"


# ---------------------------------------------------------------------------
# Memory usage — prove O(N) vs O(N²)
# ---------------------------------------------------------------------------

class TestMemoryScaling:
    """Verify Flash Attention's O(N) memory claim by comparing two sequence lengths."""

    def test_memory_subquadratic(self):
        from flash_attn_legacy import flash_attention
        B, H, d = 2, 8, 64

        def measure_peak(N):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            q = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16)
            k, v = torch.randn_like(q), torch.randn_like(q)
            baseline = torch.cuda.memory_allocated()
            flash_attention(q, k, v, is_causal=True)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            # Memory beyond input tensors
            extra = (peak - baseline) / 1e6
            del q, k, v
            torch.cuda.empty_cache()
            return extra

        mem_1k = measure_peak(1024)
        mem_4k = measure_peak(4096)

        # If O(N²): mem_4k / mem_1k ≈ 16  (4096²/1024² = 16)
        # If O(N):  mem_4k / mem_1k ≈ 4   (4096/1024 = 4)
        # Allow some overhead — ratio should be well under 10
        ratio = mem_4k / max(mem_1k, 0.1)
        assert ratio < 10, (
            f"Memory scaling looks quadratic: "
            f"N=1024 → {mem_1k:.1f} MB, N=4096 → {mem_4k:.1f} MB, ratio={ratio:.1f}"
        )


# ---------------------------------------------------------------------------
# Numerical stability — edge cases that break naive softmax
# ---------------------------------------------------------------------------

class TestNumericalStability:

    def test_large_values(self, head_dim):
        """Large Q/K values → large dot products. Online softmax should handle this."""
        from flash_attn_legacy import flash_attention
        B, H, N, d = 1, 2, 128, head_dim
        scale = 1.0 / math.sqrt(d)

        # Scale up Q and K so dot products are ~100-1000x normal
        q = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16) * 5.0
        k = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16) * 5.0
        v = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16) * 0.1

        out = flash_attention(q, k, v, softmax_scale=scale)
        assert not torch.isnan(out).any(), "NaN with large Q/K"
        assert not torch.isinf(out).any(), "Inf with large Q/K"

        # Output should still be bounded (softmax weights sum to 1)
        v_max = v.float().abs().max().item()
        out_max = out.float().abs().max().item()
        assert out_max < v_max * 2, f"Output {out_max:.3f} exceeds V range {v_max:.3f}"

    def test_small_values(self, head_dim):
        """Very small Q/K → near-uniform attention. Should not produce NaN."""
        from flash_attn_legacy import flash_attention
        B, H, N, d = 1, 2, 128, head_dim

        q = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16) * 1e-3
        k = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16) * 1e-3
        v = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16) * 0.5

        out = flash_attention(q, k, v)
        assert not torch.isnan(out).any(), "NaN with tiny Q/K"
        # Near-uniform attention → output should be close to mean(V)
        v_mean = v.float().mean(dim=2, keepdim=True).half()
        torch.testing.assert_close(out, v_mean.expand_as(out), rtol=0.1, atol=0.05)

    def test_custom_scale(self, head_dim):
        """Non-standard softmax_scale should produce different (valid) results."""
        from flash_attn_legacy import flash_attention
        B, H, N, d = 1, 2, 64, head_dim

        q = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16) * 0.3
        k, v = torch.randn_like(q) * 0.3, torch.randn_like(q) * 0.3

        out_default = flash_attention(q, k, v)  # scale = 1/sqrt(d)
        out_custom = flash_attention(q, k, v, softmax_scale=0.01)

        assert not torch.isnan(out_custom).any()
        # Different scale → different output
        assert not torch.equal(out_default, out_custom)

    def test_causal_first_row(self, head_dim):
        """First row with causal mask attends only to position 0 → O[0] = V[0]."""
        from flash_attn_legacy import flash_attention
        B, H, d = 1, 2, head_dim

        q = torch.randn(B, H, 64, d, device='cuda', dtype=torch.float16) * 0.3
        k = torch.randn(B, H, 64, d, device='cuda', dtype=torch.float16) * 0.3
        v = torch.randn(B, H, 64, d, device='cuda', dtype=torch.float16)

        out = flash_attention(q, k, v, is_causal=True)
        # First position can only attend to itself → softmax([s]) = [1.0] → O[0] = V[0]
        torch.testing.assert_close(out[:, :, 0, :], v[:, :, 0, :], rtol=1e-2, atol=1e-3)

    def test_backward_gradient_magnitude(self, head_dim):
        """Gradients should have reasonable magnitude — no explosion or vanishing."""
        from flash_attn_legacy import flash_attention
        B, H, N, d = 1, 4, 256, head_dim
        scale = 1.0 / math.sqrt(d)

        q = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16) * 0.3
        k = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16) * 0.3
        v = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16) * 0.3
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)

        flash_attention(q, k, v, softmax_scale=scale).sum().backward()

        for name, grad in [("dQ", q.grad), ("dK", k.grad), ("dV", v.grad)]:
            g_abs = grad.float().abs()
            assert g_abs.max() < 1000, f"{name} gradient exploded: max={g_abs.max():.1f}"
            assert g_abs.mean() > 1e-6, f"{name} gradient vanished: mean={g_abs.mean():.2e}"
