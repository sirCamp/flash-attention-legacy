"""
Tests for flash_attn_legacy — correctness against PyTorch reference.

Run:  pytest tests/test_flash_attn.py -v
      pytest tests/test_flash_attn.py -v -k "gqa"     # GQA tests only
"""

import contextlib
import math
import warnings

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@contextlib.contextmanager
def no_host_sync():
    """Fail on any host-device synchronization inside the block (torch's sync
    debug mode; its "prototype feature" warning silenced)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Synchronization debug mode")
        torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Synchronization debug mode")
            torch.cuda.set_sync_debug_mode("default")


def kernel_ops():
    """The kernels as dispatcher ops (importing the package registers them)."""
    import flash_attn_legacy  # noqa: F401
    return torch.ops.flash_attn_legacy


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------

def ref_hidden(Nq, Nk, is_causal, window, device):
    """[Nq, Nk] True where the key is hidden. Bottom-right aligned: query i sits at
    key i + (Nk - Nq) and sees keys [i + off - left, i + off + right] (-1 =
    unbounded side; causal: right = 0), as upstream flash-attn."""
    left, right = window
    if is_causal:
        right = 0
    rel = (torch.arange(Nk, device=device)[None, :]
           - torch.arange(Nq, device=device)[:, None] - (Nk - Nq))  # key - row position
    hidden = torch.zeros(Nq, Nk, dtype=torch.bool, device=device)
    if left >= 0:
        hidden |= rel < -left
    if right >= 0:
        hidden |= rel > right
    return hidden


def ref_logits(q, k, scale, is_causal, window, softcap):
    """FP32 logits (K already expanded to the Q heads): scaled scores, softcapped
    as c * tanh(x / c) when softcap > 0 (Gemma 2), -inf where hidden."""
    s = torch.matmul(q, k.transpose(-2, -1)) * scale
    if softcap > 0:
        s = softcap * torch.tanh(s / softcap)
    return s.masked_fill(ref_hidden(q.shape[-2], k.shape[-2], is_causal, window, q.device), float('-inf'))


def ref_probs(s, sinks=None):
    """Softmax of logits [B, H, N_q, N_k]. With sinks [H]: one more logit per row,
    without value (gpt-oss), dropped after the softmax. Without: rows that see no
    key -> 0, like the kernel."""
    if sinks is None:
        return torch.nan_to_num(torch.softmax(s, dim=-1))
    col = sinks.float().view(1, -1, 1, 1).expand(*s.shape[:-1], 1)
    return torch.softmax(torch.cat([s, col], dim=-1), dim=-1)[..., :-1]


def ref_attention(q, k, v, scale, is_causal=False, window=(-1, -1), softcap=0.0, sinks=None):
    """Standard attention in FP32. Handles GQA by repeating K/V heads."""
    H_q, H_kv = q.shape[1], k.shape[1]
    if H_kv < H_q:
        reps = H_q // H_kv
        k = k.repeat_interleave(reps, dim=1)
        v = v.repeat_interleave(reps, dim=1)
    s = ref_logits(q.float(), k.float(), scale, is_causal, window, softcap)
    return torch.matmul(ref_probs(s, sinks), v.float()).half()


def ref_backward(q, k, v, grad_out, scale, is_causal=False, window=(-1, -1), softcap=0.0, sinks=None):
    """Reference backward in FP32. Returns dQ [H_q], dK [H_kv], dV [H_kv], and
    d sinks (fp32 [H_q]) when sinks are given."""
    H_q, H_kv = q.shape[1], k.shape[1]
    reps = H_q // H_kv

    q_f = q.float().detach().requires_grad_(True)
    # Expand K/V for GQA
    k_f = k.float().detach().repeat_interleave(reps, dim=1).requires_grad_(True)
    v_f = v.float().detach().repeat_interleave(reps, dim=1).requires_grad_(True)

    sinks_f = None if sinks is None else sinks.float().detach().requires_grad_(True)
    s = ref_logits(q_f, k_f, scale, is_causal, window, softcap)
    o = torch.matmul(ref_probs(s, sinks_f), v_f)
    o.backward(grad_out.float())

    dq = q_f.grad.half()
    # Sum expanded dK/dV back to H_kv heads
    dk = k_f.grad.view(k.shape[0], H_kv, reps, k.shape[2], k.shape[3]).sum(dim=2).half()
    dv = v_f.grad.view(v.shape[0], H_kv, reps, v.shape[2], v.shape[3]).sum(dim=2).half()
    return (dq, dk, dv) if sinks is None else (dq, dk, dv, sinks_f.grad)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=[32, 64, 72, 80, 96, 128, 256], ids=["d32", "d64", "d72", "d80", "d96", "d128", "d256"])
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

        out = flash_attention(q, k, v, softmax_scale=scale, causal=is_causal)
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

        out = flash_attention(q, k, v, softmax_scale=scale, causal=True)
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

        flash_attention(q, k, v, causal=is_causal).sum().backward()
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
        flash_attention(q, k, v, softmax_scale=scale, causal=is_causal).backward(go)
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

# ---------------------------------------------------------------------------
# Variable-length (packed sequences)
# ---------------------------------------------------------------------------

class TestVarlen:
    """Test flash_attn_varlen_func with packed sequences."""

    def _ref_varlen(self, q, k, v, cu_seqlens_q, cu_seqlens_k, scale, causal=False):
        """Reference: loop over sequences, compute standard attention per seq."""
        batch_size = cu_seqlens_q.shape[0] - 1
        outputs = []
        for i in range(batch_size):
            qs = cu_seqlens_q[i].item()
            qe = cu_seqlens_q[i + 1].item()
            ks = cu_seqlens_k[i].item()
            ke = cu_seqlens_k[i + 1].item()
            # [seq, H, d] -> [1, H, seq, d]
            qi = q[qs:qe].permute(1, 0, 2).unsqueeze(0)
            ki = k[ks:ke].permute(1, 0, 2).unsqueeze(0)
            vi = v[ks:ke].permute(1, 0, 2).unsqueeze(0)
            oi = ref_attention(qi, ki, vi, scale, is_causal=causal)
            outputs.append(oi.squeeze(0).permute(1, 0, 2))
        return torch.cat(outputs, dim=0)

    @pytest.mark.parametrize("d", [32, 64, 72, 80, 96, 128, 256])
    @pytest.mark.parametrize("causal", [False, True], ids=["non_causal", "causal"])
    def test_varlen_correctness(self, d, causal):
        """Varlen matches per-sequence reference attention."""
        from flash_attn_legacy import flash_attn_varlen_func

        H = 4
        seq_lens = [32, 64, 48]
        total = sum(seq_lens)
        scale = 1.0 / math.sqrt(d)

        q = torch.randn(total, H, d, device='cuda', dtype=torch.float16) * 0.3
        k = torch.randn(total, H, d, device='cuda', dtype=torch.float16) * 0.3
        v = torch.randn(total, H, d, device='cuda', dtype=torch.float16) * 0.3

        cu = torch.tensor([0] + list(torch.tensor(seq_lens).cumsum(0).tolist()),
                          dtype=torch.int32, device='cuda')

        out = flash_attn_varlen_func(q, k, v, cu, cu, max(seq_lens), max(seq_lens),
                                     softmax_scale=scale, causal=causal)
        ref = self._ref_varlen(q, k, v, cu, cu, scale, causal)

        assert out.shape == (total, H, d)
        torch.testing.assert_close(out, ref, rtol=2e-2, atol=5e-3)

    @pytest.mark.parametrize("d", [32, 64, 72, 80, 96, 128, 256])
    def test_varlen_gqa(self, d):
        """Varlen with GQA (different Q and KV head counts)."""
        from flash_attn_legacy import flash_attn_varlen_func

        H_q, H_kv = 8, 2
        seq_lens = [64, 32]
        total = sum(seq_lens)
        scale = 1.0 / math.sqrt(d)

        q = torch.randn(total, H_q, d, device='cuda', dtype=torch.float16) * 0.3
        k = torch.randn(total, H_kv, d, device='cuda', dtype=torch.float16) * 0.3
        v = torch.randn(total, H_kv, d, device='cuda', dtype=torch.float16) * 0.3

        cu = torch.tensor([0, 64, 96], dtype=torch.int32, device='cuda')

        out = flash_attn_varlen_func(q, k, v, cu, cu, 64, 64,
                                     softmax_scale=scale, causal=True)
        assert out.shape == (total, H_q, d)
        assert not torch.isnan(out).any()

    def test_varlen_single_sequence(self):
        """Single sequence should match regular flash_attention."""
        from flash_attn_legacy import flash_attention, flash_attn_varlen_func

        H, N, d = 4, 128, 64
        scale = 1.0 / math.sqrt(d)

        q = torch.randn(N, H, d, device='cuda', dtype=torch.float16) * 0.3
        k = torch.randn(N, H, d, device='cuda', dtype=torch.float16) * 0.3
        v = torch.randn(N, H, d, device='cuda', dtype=torch.float16) * 0.3

        cu = torch.tensor([0, N], dtype=torch.int32, device='cuda')
        out_varlen = flash_attn_varlen_func(q, k, v, cu, cu, N, N,
                                            softmax_scale=scale, causal=True)

        # Compare with batched version
        q_b = q.permute(1, 0, 2).unsqueeze(0)  # [1, H, N, d]
        k_b = k.permute(1, 0, 2).unsqueeze(0)
        v_b = v.permute(1, 0, 2).unsqueeze(0)
        out_batch = flash_attention(q_b, k_b, v_b, softmax_scale=scale, causal=True)
        out_batch = out_batch.squeeze(0).permute(1, 0, 2)  # [N, H, d]

        torch.testing.assert_close(out_varlen, out_batch, rtol=1e-4, atol=1e-4)

    def test_varlen_backward(self):
        """Gradients flow through varlen."""
        from flash_attn_legacy import flash_attn_varlen_func

        H, d = 4, 64
        seq_lens = [32, 64]
        total = sum(seq_lens)

        q = torch.randn(total, H, d, device='cuda', dtype=torch.float16, requires_grad=True)
        k = torch.randn(total, H, d, device='cuda', dtype=torch.float16, requires_grad=True)
        v = torch.randn(total, H, d, device='cuda', dtype=torch.float16, requires_grad=True)

        cu = torch.tensor([0, 32, 96], dtype=torch.int32, device='cuda')

        out = flash_attn_varlen_func(q, k, v, cu, cu, 64, 64, causal=True)
        out.sum().backward()

        assert q.grad is not None and not torch.all(q.grad == 0)
        assert k.grad is not None and not torch.all(k.grad == 0)
        assert v.grad is not None and not torch.all(v.grad == 0)

    def test_varlen_many_short_seqs(self):
        """Many short sequences packed together."""
        from flash_attn_legacy import flash_attn_varlen_func

        H, d = 2, 64
        seq_lens = [16, 8, 32, 4, 24]
        total = sum(seq_lens)

        q = torch.randn(total, H, d, device='cuda', dtype=torch.float16) * 0.3
        k = torch.randn(total, H, d, device='cuda', dtype=torch.float16) * 0.3
        v = torch.randn(total, H, d, device='cuda', dtype=torch.float16) * 0.3

        cu = torch.tensor([0] + list(torch.tensor(seq_lens).cumsum(0).tolist()),
                          dtype=torch.int32, device='cuda')

        out = flash_attn_varlen_func(q, k, v, cu, cu, max(seq_lens), max(seq_lens),
                                     causal=True)
        assert out.shape == (total, H, d)
        assert not torch.isnan(out).any()


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

    @pytest.mark.parametrize("embed_dim,num_heads,head_dim", [
        (384, 12, 32),   # MiniLM-style
        (768, 8, 96),    # Phi-3-mini-style
        (1152, 16, 72),  # SigLIP-so400m-style (zero-padded tiles)
        (1280, 16, 80),  # Qwen2-VL vision-style
        (512, 2, 256),   # GPT-J-style
    ])
    def test_module_new_head_dims(self, embed_dim, num_heads, head_dim):
        from flash_attn_legacy import FlashMultiHeadAttention
        mha = FlashMultiHeadAttention(embed_dim, num_heads=num_heads, causal=True).cuda().half()
        assert mha.head_dim == head_dim
        x = torch.randn(2, 64, embed_dim, device='cuda', dtype=torch.float16)
        out = mha(x)
        assert out.shape == x.shape


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

        out = flash_attention(q, k, v, softmax_scale=scale, causal=True)
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

        out = flash_attention(q, k, v, softmax_scale=scale, causal=True)
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
        out = flash_attention(q, k, v, causal=True)
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
            flash_attention(q, k, v, causal=True)
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

        out = flash_attention(q, k, v, causal=True)
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


# ---------------------------------------------------------------------------
# Input validation & unsupported configurations
# ---------------------------------------------------------------------------

class TestCrossLength:
    """N_q != N_kv: forward + backward vs FP32 reference (bottom-right causal)."""

    @pytest.mark.parametrize("n_q,n_k", [(64, 128), (128, 64), (1, 100), (17, 200), (100, 37)])
    @pytest.mark.parametrize("causal", [False, True], ids=["non_causal", "causal"])
    @pytest.mark.parametrize("H_q,H_kv", [(4, 4), (8, 2)])
    def test_forward_backward(self, head_dim, n_q, n_k, causal, H_q, H_kv):
        from flash_attn_legacy import flash_attention
        torch.manual_seed(0)
        scale = 1.0 / math.sqrt(head_dim)
        q = torch.randn(2, H_q, n_q, head_dim, device="cuda", dtype=torch.float16, requires_grad=True)
        k = torch.randn(2, H_kv, n_k, head_dim, device="cuda", dtype=torch.float16, requires_grad=True)
        v = torch.randn(2, H_kv, n_k, head_dim, device="cuda", dtype=torch.float16, requires_grad=True)
        go = torch.randn(2, H_q, n_q, head_dim, device="cuda", dtype=torch.float16)

        out = flash_attention(q, k, v, softmax_scale=scale, causal=causal)
        out.backward(go)
        ref = ref_attention(q.detach(), k.detach(), v.detach(), scale, is_causal=causal)
        rdq, rdk, rdv = ref_backward(q.detach(), k.detach(), v.detach(), go, scale, is_causal=causal)

        assert out.shape == (2, H_q, n_q, head_dim)
        assert k.grad.shape == k.shape and v.grad.shape == v.shape
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(q.grad, rdq, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(k.grad, rdk, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(v.grad, rdv, atol=5e-2, rtol=5e-2)

    def test_varlen_unequal_lengths_backward(self):
        from flash_attn_legacy import flash_attn_varlen_func
        torch.manual_seed(0)
        H, d = 4, 64
        lens_q, lens_k = [32, 64], [64, 40]
        cu_q = torch.tensor([0, 32, 96], dtype=torch.int32, device="cuda")
        cu_k = torch.tensor([0, 64, 104], dtype=torch.int32, device="cuda")
        q = torch.randn(sum(lens_q), H, d, device="cuda", dtype=torch.float16, requires_grad=True)
        k = torch.randn(sum(lens_k), H, d, device="cuda", dtype=torch.float16, requires_grad=True)
        v = torch.randn(sum(lens_k), H, d, device="cuda", dtype=torch.float16, requires_grad=True)
        scale = 1.0 / math.sqrt(d)
        for causal in (False, True):
            for t in (q, k, v):
                t.grad = None
            out = flash_attn_varlen_func(q, k, v, cu_q, cu_k, 64, 64, softmax_scale=scale, causal=causal)
            out.sum().backward()
            qs = k_s = 0
            for lq, lk in zip(lens_q, lens_k):
                qi = q[qs:qs + lq].detach().permute(1, 0, 2).unsqueeze(0)
                ki = k[k_s:k_s + lk].detach().permute(1, 0, 2).unsqueeze(0)
                vi = v[k_s:k_s + lk].detach().permute(1, 0, 2).unsqueeze(0)
                ref = ref_attention(qi, ki, vi, scale, is_causal=causal).squeeze(0).permute(1, 0, 2)
                torch.testing.assert_close(out[qs:qs + lq], ref, atol=2e-2, rtol=2e-2)
                go = torch.ones_like(qi)
                rdq, rdk, rdv = ref_backward(qi, ki, vi, go, scale, is_causal=causal)
                torch.testing.assert_close(q.grad[qs:qs + lq], rdq.squeeze(0).permute(1, 0, 2), atol=5e-2, rtol=5e-2)
                torch.testing.assert_close(k.grad[k_s:k_s + lk], rdk.squeeze(0).permute(1, 0, 2), atol=5e-2, rtol=5e-2)
                torch.testing.assert_close(v.grad[k_s:k_s + lk], rdv.squeeze(0).permute(1, 0, 2), atol=5e-2, rtol=5e-2)
                qs += lq
                k_s += lk


class TestWindow:
    """window_size=(left, right) (docs/PLAN.md P4.1): forward + backward vs the
    FP32 reference, batched and varlen, with blocks outside the window skipped."""

    WINDOWS = [(0, 0), (16, 0), (63, -1), (-1, 40), (100, 100), (3, 7), (200, -1)]

    @pytest.mark.parametrize("n_q,n_k", [(256, 256), (100, 300), (300, 100)])
    @pytest.mark.parametrize("window", WINDOWS, ids=str)
    @pytest.mark.parametrize("causal", [False, True], ids=["non_causal", "causal"])
    def test_forward_backward(self, head_dim, n_q, n_k, window, causal):
        from flash_attn_legacy import flash_attention
        torch.manual_seed(0)
        scale = 1.0 / math.sqrt(head_dim)
        H_q, H_kv = 4, 2
        q = torch.randn(2, H_q, n_q, head_dim, device="cuda", dtype=torch.float16, requires_grad=True)
        k = torch.randn(2, H_kv, n_k, head_dim, device="cuda", dtype=torch.float16, requires_grad=True)
        v = torch.randn(2, H_kv, n_k, head_dim, device="cuda", dtype=torch.float16, requires_grad=True)
        go = torch.randn(2, H_q, n_q, head_dim, device="cuda", dtype=torch.float16)

        out = flash_attention(q, k, v, softmax_scale=scale, causal=causal, window_size=window)
        out.backward(go)
        args = (q.detach(), k.detach(), v.detach())
        ref = ref_attention(*args, scale, is_causal=causal, window=window)
        rdq, rdk, rdv = ref_backward(*args, go, scale, is_causal=causal, window=window)
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(q.grad, rdq, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(k.grad, rdk, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(v.grad, rdv, atol=5e-2, rtol=5e-2)

    @pytest.mark.parametrize("window", [(31, 0), (0, 31), (5, 70)], ids=str)
    def test_window_is_not_shifted(self, window):
        """Mutation check: the kernel matches its own window, not the window +- 1."""
        from flash_attn_legacy import flash_attention
        torch.manual_seed(1)
        q, k, v = (torch.randn(1, 2, 192, 64, device="cuda", dtype=torch.float16) for _ in range(3))
        out = flash_attention(q, k, v, window_size=window)
        scale = 1.0 / math.sqrt(64)
        torch.testing.assert_close(out, ref_attention(q, k, v, scale, window=window), atol=2e-2, rtol=2e-2)
        for shifted in ((window[0] + 1, window[1]), (window[0], window[1] + 1)):
            assert (out - ref_attention(q, k, v, scale, window=shifted)).abs().max() > 0.05

    def test_wide_window_is_unbounded(self, head_dim):
        """A window reaching every key is exactly no window (normalised to -1 on
        the host: same kernel work)."""
        from flash_attn_legacy import flash_attention
        torch.manual_seed(0)
        q, k, v = (torch.randn(2, 4, 200, head_dim, device="cuda", dtype=torch.float16) for _ in range(3))
        for causal in (False, True):
            ref = flash_attention(q, k, v, causal=causal)
            assert torch.equal(flash_attention(q, k, v, causal=causal, window_size=(199, 199)), ref)
            assert torch.equal(flash_attention(q, k, v, causal=causal, window_size=(10**9, 10**9)), ref)

    @pytest.mark.parametrize("window", [(15, 0), (5, 20), (40, -1), (-1, 7)], ids=str)
    @pytest.mark.parametrize("causal", [False, True], ids=["non_causal", "causal"])
    def test_varlen(self, window, causal):
        """Per-sequence windows, forward + backward, sequences longer and shorter
        than the window, Q shorter and longer than K (rows with no visible key)."""
        from flash_attn_legacy import flash_attn_varlen_func
        torch.manual_seed(0)
        H_q, H_kv, d = 4, 2, 64
        lens_q, lens_k = [1, 70, 200, 33, 128, 90], [5, 70, 260, 33, 300, 20]
        cu_q = torch.tensor([0] + list(torch.tensor(lens_q).cumsum(0)), dtype=torch.int32, device="cuda")
        cu_k = torch.tensor([0] + list(torch.tensor(lens_k).cumsum(0)), dtype=torch.int32, device="cuda")
        q = torch.randn(sum(lens_q), H_q, d, device="cuda", dtype=torch.float16, requires_grad=True)
        k = torch.randn(sum(lens_k), H_kv, d, device="cuda", dtype=torch.float16, requires_grad=True)
        v = torch.randn(sum(lens_k), H_kv, d, device="cuda", dtype=torch.float16, requires_grad=True)
        go = torch.randn_like(q)
        scale = 1.0 / math.sqrt(d)
        out = flash_attn_varlen_func(q, k, v, cu_q, cu_k, max(lens_q), max(lens_k),
                                     softmax_scale=scale, causal=causal, window_size=window)
        out.backward(go)
        qs = ks = 0
        for lq, lk in zip(lens_q, lens_k):
            def seq(t, s0, n):
                return t[s0:s0 + n].detach().permute(1, 0, 2).unsqueeze(0)

            def tok(t):
                return t.squeeze(0).permute(1, 0, 2)
            qi, ki, vi, goi = seq(q, qs, lq), seq(k, ks, lk), seq(v, ks, lk), seq(go, qs, lq)
            ref = ref_attention(qi, ki, vi, scale, is_causal=causal, window=window)
            rdq, rdk, rdv = ref_backward(qi, ki, vi, goi, scale, is_causal=causal, window=window)
            torch.testing.assert_close(out[qs:qs + lq], tok(ref), atol=2e-2, rtol=2e-2)
            torch.testing.assert_close(q.grad[qs:qs + lq], tok(rdq), atol=5e-2, rtol=5e-2)
            torch.testing.assert_close(k.grad[ks:ks + lk], tok(rdk), atol=5e-2, rtol=5e-2)
            torch.testing.assert_close(v.grad[ks:ks + lk], tok(rdv), atol=5e-2, rtol=5e-2)
            qs += lq
            ks += lk

    @pytest.mark.parametrize("window", [(-2, 0), (0, -5), (1,), "12", None, (3.7, 0), (True, 0)])
    def test_invalid_window_raises(self, window):
        from flash_attn_legacy import flash_attention
        q = torch.randn(1, 1, 8, 64, device="cuda", dtype=torch.float16)
        with pytest.raises(ValueError, match="window_size"):
            flash_attention(q, q, q, window_size=window)


class TestSoftcap:
    """Logit softcapping c * tanh(x / c) (docs/PLAN.md P4.2): forward + backward
    vs the FP32 reference, batched and varlen. Inputs scaled so that many logits
    sit where tanh saturates (a wrong (1 - t^2) factor shows in the gradients)."""

    @staticmethod
    def _inputs(shape_q, shape_kv, amp=3.0, requires_grad=True):
        mk = lambda shape: (torch.randn(*shape, device="cuda") * amp).half().requires_grad_(requires_grad)
        return mk(shape_q), mk(shape_kv), mk(shape_kv)

    @pytest.mark.parametrize("n_q,n_k", [(256, 256), (100, 300), (300, 100)])  # 300/100: rows see no key
    @pytest.mark.parametrize("softcap", [5.0, 50.0])
    @pytest.mark.parametrize("causal,window", [(False, (-1, -1)), (True, (-1, -1)), (True, (16, 0))],
                             ids=["full", "causal", "causal_window16"])
    def test_forward_backward(self, head_dim, n_q, n_k, softcap, causal, window):
        from flash_attn_legacy import flash_attention
        torch.manual_seed(0)
        scale = 1.0 / math.sqrt(head_dim)
        q, k, v = self._inputs((2, 4, n_q, head_dim), (2, 2, n_k, head_dim))
        go = torch.randn(2, 4, n_q, head_dim, device="cuda", dtype=torch.float16)
        out = flash_attention(q, k, v, softmax_scale=scale, causal=causal, window_size=window, softcap=softcap)
        out.backward(go)
        args = (q.detach(), k.detach(), v.detach())
        kw = dict(is_causal=causal, window=window, softcap=softcap)
        rdq, rdk, rdv = ref_backward(*args, go, scale, **kw)
        torch.testing.assert_close(out, ref_attention(*args, scale, **kw), atol=2e-2, rtol=2e-2)
        # |dQ|, |dK| grow with the input scale: tolerance relative to the largest entry
        for got, ref in ((q.grad, rdq), (k.grad, rdk), (v.grad, rdv)):
            tol = 2e-2 * ref.float().abs().max().item()
            torch.testing.assert_close(got.float(), ref.float(), atol=tol, rtol=2e-2)

    def test_softcap_is_applied_with_its_value(self):
        """Mutation check: forward matches its own cap, not no cap or another cap;
        gradients match the softcapped reference, not the uncapped chain rule."""
        from flash_attn_legacy import flash_attention
        torch.manual_seed(1)
        q, k, v = self._inputs((1, 2, 192, 64), (1, 2, 192, 64))
        go = torch.randn(1, 2, 192, 64, device="cuda", dtype=torch.float16)
        scale, cap = 1.0 / 8, 4.0
        out = flash_attention(q, k, v, softcap=cap)
        out.backward(go)
        args = (q.detach(), k.detach(), v.detach())
        torch.testing.assert_close(out, ref_attention(*args, scale, softcap=cap), atol=2e-2, rtol=2e-2)
        for other in (0.0, 2 * cap):
            assert (out - ref_attention(*args, scale, softcap=other)).abs().max() > 0.1
        rdq = ref_backward(*args, go, scale, softcap=cap)[0]
        # dS * (1 - t^2) left out: the same P and dP, the uncapped factor `scale`
        assert (q.grad.float() - rdq.float()).abs().max() < 0.05 * rdq.float().abs().max()
        with torch.no_grad():
            s = (args[0].float() @ args[1].float().transpose(-1, -2)) * scale
            p = torch.softmax(cap * torch.tanh(s / cap), -1)
            dp = go.float() @ args[2].float().transpose(-1, -2)
            ds = p * (dp - (dp * p).sum(-1, keepdim=True)) * scale  # no (1 - t^2)
            wrong_dq = ds @ args[1].float()
        assert (q.grad.float() - wrong_dq).abs().max() > 0.2 * rdq.float().abs().max()

    @pytest.mark.parametrize("causal", [False, True], ids=["non_causal", "causal"])
    def test_varlen(self, causal):
        from flash_attn_legacy import flash_attn_varlen_func
        torch.manual_seed(0)
        H_q, H_kv, d, cap = 4, 2, 64, 10.0
        lens = [1, 70, 200, 33]
        cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device="cuda")
        q, k, v = self._inputs((sum(lens), H_q, d), (sum(lens), H_kv, d))
        go = torch.randn_like(q)
        scale = 1.0 / math.sqrt(d)
        out = flash_attn_varlen_func(q, k, v, cu, cu, max(lens), max(lens), softmax_scale=scale,
                                     causal=causal, softcap=cap)
        out.backward(go)
        s0 = 0
        for n in lens:
            def seq(t):
                return t[s0:s0 + n].detach().permute(1, 0, 2).unsqueeze(0)

            def tok(t):
                return t.squeeze(0).permute(1, 0, 2)
            args = (seq(q), seq(k), seq(v))
            ref = ref_attention(*args, scale, is_causal=causal, softcap=cap)
            refs = ref_backward(*args, seq(go), scale, is_causal=causal, softcap=cap)
            torch.testing.assert_close(out[s0:s0 + n], tok(ref), atol=2e-2, rtol=2e-2)
            for got, r in zip((q.grad, k.grad, v.grad), refs):
                r = tok(r).float()
                torch.testing.assert_close(got[s0:s0 + n].float(), r, atol=2e-2 * r.abs().max().item() + 1e-3,
                                           rtol=2e-2)
            s0 += n

    def test_huge_softcap_is_no_softcap(self):
        """c far above every |x|: c * tanh(x / c) ~ x, so the softcapped kernel
        must match the uncapped reference (forward and backward)."""
        from flash_attn_legacy import flash_attention
        torch.manual_seed(0)
        q, k, v = self._inputs((2, 4, 130, 64), (2, 4, 130, 64), amp=1.0)
        go = torch.randn_like(q)
        out = flash_attention(q, k, v, causal=True, softcap=1e4)
        out.backward(go)
        args = (q.detach(), k.detach(), v.detach())
        torch.testing.assert_close(out, ref_attention(*args, 1 / 8, is_causal=True), atol=2e-2, rtol=2e-2)
        for got, ref in zip((q.grad, k.grad, v.grad), ref_backward(*args, go, 1 / 8, is_causal=True)):
            torch.testing.assert_close(got, ref, atol=5e-2, rtol=5e-2)

    @pytest.mark.parametrize("softcap", [-1.0, float("nan"), float("inf"), True, "30", None, 1e-50, 1e39, 1e-40])
    def test_invalid_softcap_raises(self, softcap):
        from flash_attn_legacy import flash_attention
        q = torch.randn(1, 1, 8, 64, device="cuda", dtype=torch.float16)
        with pytest.raises(ValueError, match="softcap"):
            flash_attention(q, q, q, softcap=softcap)


class TestSinks:
    """Attention sinks (gpt-oss s_aux, docs/PLAN.md P4.3): one logit per Q head in
    every row's softmax, without value. Forward, logsumexp and the gradients of
    Q / K / V / sinks vs the FP32 reference, batched and varlen."""

    @staticmethod
    def _inputs(shape_q, shape_kv, n_heads, seed=0):
        torch.manual_seed(seed)
        mk = lambda shape: torch.randn(*shape, device="cuda", dtype=torch.float16).requires_grad_(True)
        # sinks around the logits' range: they take a visible share of each row
        sinks = (torch.randn(n_heads, device="cuda") * 2 + 1).requires_grad_(True)
        return mk(shape_q), mk(shape_kv), mk(shape_kv), sinks

    @staticmethod
    def _assert_grads(got, ref):
        for g, r in zip(got, ref):
            r = r.float()
            scale = r.abs().max().item() if r.numel() else 0.0
            torch.testing.assert_close(g.float(), r, atol=2e-2 * scale + 1e-3, rtol=2e-2)

    @pytest.mark.parametrize("n_q,n_k", [(256, 256), (100, 300), (300, 100)])  # 300/100 causal: rows see no key
    @pytest.mark.parametrize("causal,window", [(False, (-1, -1)), (True, (-1, -1)), (True, (16, 0))],
                             ids=["full", "causal", "causal_window16"])
    def test_forward_backward(self, head_dim, n_q, n_k, causal, window):
        from flash_attn_legacy import flash_attention
        scale = 1.0 / math.sqrt(head_dim)
        q, k, v, sinks = self._inputs((2, 4, n_q, head_dim), (2, 2, n_k, head_dim), 4)
        go = torch.randn(2, 4, n_q, head_dim, device="cuda", dtype=torch.float16)
        out = flash_attention(q, k, v, softmax_scale=scale, causal=causal, window_size=window, sinks=sinks)
        out.backward(go)
        args = (q.detach(), k.detach(), v.detach())
        kw = dict(is_causal=causal, window=window, sinks=sinks.detach())
        torch.testing.assert_close(out, ref_attention(*args, scale, **kw), atol=2e-2, rtol=2e-2)
        self._assert_grads((q.grad, k.grad, v.grad, sinks.grad), ref_backward(*args, go, scale, **kw))

    @pytest.mark.parametrize("n_q,n_k", [(65, 65), (200, 5), (5, 200)])
    def test_lse_includes_the_sink(self, head_dim, n_q, n_k):
        """L = logsumexp(logits, sink); (200, 5) causal: rows with no visible key
        give O = 0 and L = sink (the backward rebuilds P from L)."""
        scale = 1.0 / math.sqrt(head_dim)
        q, k, v, sinks = (t.detach() for t in self._inputs((2, 4, n_q, head_dim), (2, 2, n_k, head_dim), 4))
        out, lse = kernel_ops().forward(q, k, v, scale, True, -1, -1, 0.0, sinks)
        torch.testing.assert_close(out, ref_attention(q, k, v, scale, True, sinks=sinks), atol=2e-2, rtol=2e-2)
        s = ref_logits(q.float(), k.float().repeat_interleave(2, dim=1), scale, True, (-1, -1), 0.0)
        ref = torch.logsumexp(torch.cat([s, sinks.view(1, -1, 1, 1).expand(*s.shape[:-1], 1)], -1), -1)
        torch.testing.assert_close(lse, ref, atol=2e-3, rtol=1e-3)

    def test_no_keys(self):
        """N_k = 0: O = 0, L = sink; zero gradients."""
        from flash_attn_legacy import flash_attention
        q, k, v, sinks = self._inputs((1, 4, 8, 64), (1, 4, 0, 64), 4)
        out, lse = kernel_ops().forward(q.detach(), k.detach(), v.detach(), 0.125, False, -1, -1, 0.0,
                                        sinks.detach())
        assert (out == 0).all() and torch.equal(lse, sinks.detach().view(1, 4, 1).expand_as(lse))
        flash_attention(q, k, v, sinks=sinks).sum().backward()
        assert (q.grad == 0).all() and (sinks.grad == 0).all()

    def test_sinks_are_applied_with_their_values(self):
        """Mutation check: each head uses its own sink (not none, not another
        head's) and the sink gradient is not zero."""
        from flash_attn_legacy import flash_attention
        q, k, v, sinks = self._inputs((1, 4, 128, 64), (1, 4, 128, 64), 4, seed=1)
        out = flash_attention(q, k, v, causal=True, sinks=sinks)
        out.sum().backward()
        args = (q.detach(), k.detach(), v.detach())
        torch.testing.assert_close(out, ref_attention(*args, 0.125, True, sinks=sinks.detach()),
                                   atol=2e-2, rtol=2e-2)
        for other in (None, sinks.detach().flip(0)):
            assert (out - ref_attention(*args, 0.125, True, sinks=other)).abs().max() > 0.05
        assert (sinks.grad.abs() > 1e-3).all()

    def test_minus_inf_sink_is_no_sink(self):
        """sink = -inf on some heads: those heads match attention without sink,
        with a zero sink gradient; the others keep theirs."""
        from flash_attn_legacy import flash_attention
        q, k, v, sinks = self._inputs((2, 4, 100, 64), (2, 4, 100, 64), 4)
        with torch.no_grad():
            sinks[::2] = float("-inf")
        go = torch.randn_like(q)
        out = flash_attention(q, k, v, causal=True, sinks=sinks)
        out.backward(go)
        args = (q.detach(), k.detach(), v.detach())
        torch.testing.assert_close(out[:, ::2], ref_attention(*args, 0.125, True)[:, ::2], atol=2e-2, rtol=2e-2)
        ref = ref_backward(*args, go, 0.125, True, sinks=sinks.detach())
        self._assert_grads((q.grad, k.grad, v.grad), ref[:3])
        assert (sinks.grad[::2] == 0).all()
        torch.testing.assert_close(sinks.grad[1::2], ref[3][1::2], atol=1e-2, rtol=2e-2)

    def test_with_softcap_and_window(self):
        from flash_attn_legacy import flash_attention
        q, k, v, sinks = self._inputs((2, 4, 150, 64), (2, 2, 150, 64), 4)
        go = torch.randn_like(q)
        out = flash_attention(q, k, v, causal=True, window_size=(31, 0), softcap=8.0, sinks=sinks)
        out.backward(go)
        args = (q.detach(), k.detach(), v.detach())
        kw = dict(is_causal=True, window=(31, 0), softcap=8.0, sinks=sinks.detach())
        torch.testing.assert_close(out, ref_attention(*args, 0.125, **kw), atol=2e-2, rtol=2e-2)
        self._assert_grads((q.grad, k.grad, v.grad, sinks.grad), ref_backward(*args, go, 0.125, **kw))

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_sink_dtype(self, dtype):
        """Any float dtype; the gradient comes back in it, with the reference value."""
        from flash_attn_legacy import flash_attention
        q, k, v, sinks = self._inputs((1, 2, 64, 64), (1, 2, 64, 64), 2)
        sinks = sinks.detach().to(dtype).requires_grad_(True)
        go = torch.randn_like(q)
        flash_attention(q, k, v, sinks=sinks).backward(go)
        ref = ref_backward(q.detach(), k.detach(), v.detach(), go, 0.125, sinks=sinks.detach())[3]
        assert sinks.grad.dtype == dtype
        torch.testing.assert_close(sinks.grad.float(), ref, atol=3e-2 * ref.abs().max().item(), rtol=3e-2)

    def test_frozen_sinks(self):
        """Sinks without requires_grad: applied in the forward, no gradient."""
        from flash_attn_legacy import flash_attention
        q, k, v, sinks = self._inputs((1, 2, 64, 64), (1, 2, 64, 64), 2)
        sinks = sinks.detach()
        go = torch.randn_like(q)
        out = flash_attention(q, k, v, sinks=sinks)
        out.backward(go)
        args = (q.detach(), k.detach(), v.detach())
        torch.testing.assert_close(out, ref_attention(*args, 0.125, sinks=sinks), atol=2e-2, rtol=2e-2)
        self._assert_grads((q.grad, k.grad, v.grad), ref_backward(*args, go, 0.125, sinks=sinks)[:3])
        assert sinks.grad is None

    @pytest.mark.parametrize("lens_q,lens_k", [
        ([1, 70, 200, 33], [1, 70, 200, 33]),
        ([5, 0, 35, 1, 20], [0, 7, 33, 20, 1]),  # no keys, no queries, rows that see no key
    ], ids=["same", "uneven"])
    @pytest.mark.parametrize("causal", [False, True], ids=["non_causal", "causal"])
    def test_varlen(self, lens_q, lens_k, causal):
        """Packed sequences of 0 to 200 tokens; d sinks sums over all of them."""
        from flash_attn_legacy import flash_attn_varlen_func
        H_q, H_kv, d = 4, 2, 64
        cu = lambda lens: torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32,  # noqa: E731
                                       device="cuda")
        q, k, v, sinks = self._inputs((sum(lens_q), H_q, d), (sum(lens_k), H_kv, d), H_q)
        go = torch.randn_like(q)
        out = flash_attn_varlen_func(q, k, v, cu(lens_q), cu(lens_k), max(lens_q), max(lens_k),
                                     causal=causal, sinks=sinks)
        out.backward(go)
        dsinks = torch.zeros(H_q, device="cuda")
        q0 = k0 = 0
        for n_q, n_k in zip(lens_q, lens_k):
            def seq(t, s0, n):
                return t[s0:s0 + n].detach().permute(1, 0, 2).unsqueeze(0)

            def tok(t):
                return t.squeeze(0).permute(1, 0, 2)
            args = (seq(q, q0, n_q), seq(k, k0, n_k), seq(v, k0, n_k))
            kw = dict(is_causal=causal, sinks=sinks.detach())
            torch.testing.assert_close(out[q0:q0 + n_q], tok(ref_attention(*args, 0.125, **kw)),
                                       atol=2e-2, rtol=2e-2)
            *refs, ds = ref_backward(*args, seq(go, q0, n_q), 0.125, **kw)
            self._assert_grads([q.grad[q0:q0 + n_q], k.grad[k0:k0 + n_k], v.grad[k0:k0 + n_k]],
                               [tok(r) for r in refs])
            dsinks += ds
            q0, k0 = q0 + n_q, k0 + n_k
        assert torch.isfinite(sinks.grad).all()
        self._assert_grads([sinks.grad], [dsinks])

    @pytest.mark.parametrize("sinks", ["list", "str", "shape", "int", "cpu"])
    def test_invalid_sinks_raise(self, sinks):
        from flash_attn_legacy import flash_attention
        q = torch.randn(1, 4, 8, 64, device="cuda", dtype=torch.float16)
        sinks = {"list": [0.0] * 4, "str": "x", "shape": torch.zeros(3, device="cuda"),
                 "int": torch.zeros(4, dtype=torch.int32, device="cuda"), "cpu": torch.zeros(4)}[sinks]
        with pytest.raises(ValueError, match="sinks"):
            flash_attention(q, q, q, sinks=sinks)


class TestKVCache:
    """flash_attn_with_kvcache (decode, docs/analysis/kernels/decode_design.md) vs
    the FP32 reference row by row: ragged cache_seqlens (0, 1, block edges, full),
    GQA, q_len > 1 with causal, window, softcap, sinks, fp16 / bf16, contiguous
    BSHD caches and HuggingFace's BHSD layout as a transposed view (no copy)."""

    @staticmethod
    def _case(B, q_len, H_q, H_kv, d, max_len, dtype, layout, seed=0):
        torch.manual_seed(seed)
        q = torch.randn(B, q_len, H_q, d, device="cuda", dtype=dtype)
        if layout == "bshd":
            k, v = (torch.randn(B, max_len, H_kv, d, device="cuda", dtype=dtype) for _ in range(2))
        else:  # HF cache [B, H_kv, N, d], passed as a view
            k, v = (torch.randn(B, H_kv, max_len, d, device="cuda", dtype=dtype).transpose(1, 2) for _ in range(2))
        return q, k, v

    @staticmethod
    def _poison(k, v, seqlens):
        """NaN past each row's valid keys, as an uninitialised cache: never read."""
        for b, n in enumerate(seqlens):
            k[b, n:] = float("nan")
            v[b, n:] = float("nan")

    @staticmethod
    def _ref(q, k, v, seqlens, scale, causal=False, window=(-1, -1), softcap=0.0, sinks=None, leftpad=None,
             key_mask=None):
        """Per row b: reference over the keys [leftpad[b], seqlens[b]) (causal /
        window aligned to seqlens[b]) kept by key_mask[b]; out BSHD, lse [B, H_q, q_len]."""
        outs, lses = [], []
        for b, n in enumerate(seqlens):
            qb = q[b:b + 1].transpose(1, 2).float()        # [1, H_q, q_len, d]
            kb = k[b:b + 1, :n].transpose(1, 2).float()    # [1, H_kv, n, d]
            vb = v[b:b + 1, :n].transpose(1, 2).float()
            rep = qb.shape[1] // kb.shape[1]
            kb, vb = kb.repeat_interleave(rep, dim=1), vb.repeat_interleave(rep, dim=1)
            logits = ref_logits(qb, kb, scale, causal, window, softcap)
            if leftpad is not None:
                logits[..., :leftpad[b]] = float("-inf")
            if key_mask is not None:
                logits = logits.masked_fill(~key_mask[b, :n], float("-inf"))
            outs.append((ref_probs(logits, sinks) @ vb).transpose(1, 2))
            if sinks is not None:
                logits = torch.cat([logits, sinks.float().view(1, -1, 1, 1).expand(*logits.shape[:-1], 1)], -1)
            lses.append(torch.logsumexp(logits, -1))
        return torch.cat(outs), torch.cat(lses)

    def _check(self, out, lse, ref_out, ref_lse, dtype):
        tol = 2e-2 if dtype == torch.float16 else 4e-2
        torch.testing.assert_close(out.float(), ref_out, atol=tol, rtol=tol)
        finite = torch.isfinite(ref_lse)
        assert torch.equal(torch.isfinite(lse), finite)
        torch.testing.assert_close(lse[finite], ref_lse[finite], atol=2e-3, rtol=1e-3)

    # query rows per KV head (H_q / H_kv * q_len): 1, 4, 4, 4, 2, 3, 3, 8 (gpt-oss), 7 (Qwen2), 8, 6
    @pytest.mark.parametrize("q_len,H_q,H_kv", [(1, 4, 4), (1, 8, 2), (2, 4, 2), (4, 2, 2), (1, 2, 1), (3, 1, 1),
                                                (1, 3, 1), (1, 16, 2), (1, 7, 1), (2, 8, 2), (2, 3, 1), (1, 5, 1)])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
    @pytest.mark.parametrize("layout", ["bshd", "bhsd_view"])
    @pytest.mark.parametrize("num_splits", [1, 4], ids=["one_split", "splits4"])  # 4: empty splits for short rows
    def test_decode(self, head_dim, q_len, H_q, H_kv, dtype, layout, num_splits):
        from flash_attn_legacy import flash_attn_with_kvcache
        max_len = 300
        q, k, v = self._case(5, q_len, H_q, H_kv, head_dim, max_len, dtype, layout)
        seqlens = [300, 1, 0, 128, 77]
        self._poison(k, v, seqlens)
        scale = 1.0 / math.sqrt(head_dim)
        cs = torch.tensor(seqlens, dtype=torch.int32, device="cuda")
        with torch.no_grad():
            out, lse = flash_attn_with_kvcache(q, k, v, cache_seqlens=cs, causal=True, num_splits=num_splits,
                                               return_softmax_lse=True)
        ref_out, ref_lse = self._ref(q, k, v, seqlens, scale, causal=True)
        self._check(out, lse, ref_out, ref_lse, dtype)
        assert (out[2] == 0).all()  # no key: O = 0, lse = -inf

    @pytest.mark.parametrize("feature", ["window", "window_right", "softcap", "sinks", "non_causal", "int_seqlens"])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
    @pytest.mark.parametrize("num_splits", [1, 3], ids=["one_split", "splits3"])  # sinks once across splits
    @pytest.mark.parametrize("H_q", [4, 8], ids=["rows4", "rows8"])  # rows8: q in shared memory (gpt-oss: 8)
    def test_features(self, head_dim, feature, dtype, num_splits, H_q):
        from flash_attn_legacy import flash_attn_with_kvcache
        q, k, v = self._case(3, 2, H_q, 2, head_dim, 200, dtype, "bshd", seed=1)
        seqlens = [200, 50, 1]
        kw, ref_kw = {}, {"causal": True}
        if feature == "window":
            kw, ref_kw = {"window_size": (16, 0)}, {"causal": True, "window": (16, 0)}
        elif feature == "window_right":  # non-causal window around each query's position
            kw, ref_kw = {"window_size": (8, 1)}, {"window": (8, 1)}
        elif feature == "softcap":
            q = q * 4
            kw, ref_kw = {"softcap": 5.0}, {"causal": True, "softcap": 5.0}
        elif feature == "sinks":
            sinks = torch.randn(H_q, device="cuda") + 1
            kw, ref_kw = {"sinks": sinks}, {"causal": True, "sinks": sinks}
        elif feature == "non_causal":
            ref_kw = {}
        causal = feature not in ("non_causal", "window_right")
        if feature == "int_seqlens":
            seqlens, cs = [120, 120, 120], 120
        else:
            cs = torch.tensor(seqlens, dtype=torch.int32, device="cuda")
        self._poison(k, v, seqlens)
        with torch.no_grad():
            out, lse = flash_attn_with_kvcache(q, k, v, cache_seqlens=cs, causal=causal, num_splits=num_splits,
                                               return_softmax_lse=True, **kw)
        ref_out, ref_lse = self._ref(q, k, v, seqlens, 1.0 / math.sqrt(head_dim), **ref_kw)
        self._check(out, lse, ref_out, ref_lse, dtype)

    def test_seqlens_beyond_the_cache(self):
        """Tensor cache_seqlens are clamped to [0, max_len] on the GPU (never read
        past the cache); an int outside the range raises."""
        from flash_attn_legacy import flash_attn_with_kvcache
        q, k, v = self._case(2, 1, 4, 4, 64, 64, torch.float16, "bshd")
        with torch.no_grad():
            out = flash_attn_with_kvcache(q, k, v, cache_seqlens=torch.tensor([500, -3], dtype=torch.int32,
                                                                              device="cuda"))
            ref_out, _ = self._ref(q, k, v, [64, 0], 0.125)
            torch.testing.assert_close(out.float(), ref_out, atol=2e-2, rtol=2e-2)
            for bad in (65, -1):
                with pytest.raises(ValueError, match="cache_seqlens"):
                    flash_attn_with_kvcache(q, k, v, cache_seqlens=bad)

    @pytest.mark.parametrize("num_splits", [1, 5])
    def test_sinks_without_keys(self, num_splits):
        """A row whose window is empty keeps the sink only: O = 0, lse = sink
        (also when every split is empty)."""
        from flash_attn_legacy import flash_attn_with_kvcache
        q, k, v = self._case(1, 1, 2, 2, 64, 32, torch.float16, "bshd")
        sinks = torch.tensor([0.5, -1.0], device="cuda")
        with torch.no_grad():
            out, lse = flash_attn_with_kvcache(q, k, v, cache_seqlens=0, sinks=sinks, num_splits=num_splits,
                                               return_softmax_lse=True)
        assert (out == 0).all()
        torch.testing.assert_close(lse.view(-1), sinks)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
    def test_splits_long_context(self, dtype):
        """Batch 1 with a long cache: the automatic split count (> 1 here), the cap
        (128, values above it clamp) and one split agree with the reference."""
        import flash_attn_legacy_cuda
        from flash_attn_legacy import flash_attn_with_kvcache
        sms = torch.cuda.get_device_properties(0).multi_processor_count
        assert flash_attn_legacy_cuda.decode_num_splits(1 * 2, sms, 8192) > 1
        q, k, v = self._case(1, 1, 4, 2, 128, 8192, dtype, "bhsd_view")
        ref_out, ref_lse = self._ref(q, k, v, [8000], 1.0 / math.sqrt(128))
        with torch.no_grad():
            one, lse_1 = flash_attn_with_kvcache(q, k, v, cache_seqlens=8000, num_splits=1, return_softmax_lse=True)
            for n in (0, 128, 1000):
                out, lse = flash_attn_with_kvcache(q, k, v, cache_seqlens=8000, num_splits=n, return_softmax_lse=True)
                self._check(out, lse, ref_out, ref_lse, dtype)
                torch.testing.assert_close(out, one, atol=4e-3, rtol=4e-3)
                torch.testing.assert_close(lse, lse_1, atol=1e-3, rtol=1e-4)

    def test_window_with_many_splits(self):
        """Visible range starting past key 0 (sliding window) cut into many splits."""
        from flash_attn_legacy import flash_attn_with_kvcache
        q, k, v = self._case(2, 2, 4, 2, 64, 256, torch.float16, "bshd")
        seqlens = [200, 256]
        self._poison(k, v, seqlens)
        cs = torch.tensor(seqlens, dtype=torch.int32, device="cuda")
        with torch.no_grad():
            out, lse = flash_attn_with_kvcache(q, k, v, cache_seqlens=cs, causal=True, window_size=(64, 0),
                                               num_splits=16, return_softmax_lse=True)
        ref_out, ref_lse = self._ref(q, k, v, seqlens, 0.125, causal=True, window=(64, 0))
        self._check(out, lse, ref_out, ref_lse, torch.float16)

    @pytest.mark.parametrize("bad", [-1, 1.5, True])
    def test_invalid_num_splits(self, bad):
        from flash_attn_legacy import flash_attn_with_kvcache
        q, k, v = self._case(1, 1, 4, 4, 64, 32, torch.float16, "bshd")
        with torch.no_grad(), pytest.raises(ValueError, match="num_splits"):
            flash_attn_with_kvcache(q, k, v, num_splits=bad)

    def test_bf16_beyond_fp16_range(self):
        """bf16 caches are read as they are: values beyond the fp16 range are fine."""
        from flash_attn_legacy import flash_attn_with_kvcache
        q, k, v = self._case(2, 1, 4, 4, 64, 64, torch.bfloat16, "bshd")
        v = v * 1e6
        with torch.no_grad():
            out = flash_attn_with_kvcache(q, k, v, cache_seqlens=64)
        ref_out, _ = self._ref(q, k, v, [64, 64], 0.125)
        assert torch.isfinite(out).all()
        torch.testing.assert_close(out.float(), ref_out, atol=4e-2 * 1e6, rtol=4e-2)

    @pytest.mark.parametrize("num_splits", [0, 1, 4])
    @pytest.mark.parametrize("mode", ["causal", "window40", "non_causal", "sinks"])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
    def test_leftpad(self, head_dim, num_splits, mode, dtype):
        """HF left padding: keys [leftpad, seqlens), causal / window aligned to the
        end; leftpad == seqlens leaves a row with no key (only the sink)."""
        from flash_attn_legacy import flash_attn_with_kvcache
        q, k, v = self._case(4, 2, 4, 2, head_dim, 160, dtype, "bhsd_view")
        seqlens, leftpad = [160, 100, 30, 90], [0, 37, 30, 89]
        self._poison(k, v, seqlens)
        for b, lp in enumerate(leftpad):  # padded slots: never read either
            k[b, :lp] = float("nan")
            v[b, :lp] = float("nan")
        cs, lp_t = (torch.tensor(x, dtype=torch.int32, device="cuda") for x in (seqlens, leftpad))
        causal, window, sinks = mode != "non_causal", (40, 0) if mode == "window40" else (-1, -1), None
        if mode == "sinks":
            sinks = torch.randn(4, device="cuda")
        with torch.no_grad():
            out, lse = flash_attn_with_kvcache(q, k, v, cache_seqlens=cs, cache_leftpad=lp_t, causal=causal,
                                               window_size=window, num_splits=num_splits, sinks=sinks,
                                               return_softmax_lse=True)
        ref_out, ref_lse = self._ref(q, k.nan_to_num(), v.nan_to_num(), seqlens, 1.0 / math.sqrt(head_dim),
                                     causal=causal, window=window, sinks=sinks, leftpad=leftpad)
        self._check(out, lse, ref_out, ref_lse, dtype)
        assert (out[2] == 0).all()

    @pytest.mark.parametrize("num_splits", [0, 1, 4])
    @pytest.mark.parametrize("mode", ["causal", "window40", "sinks", "leftpad", "paged"])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
    def test_key_mask(self, head_dim, num_splits, mode, dtype):
        """key_mask: holes, left padding, a row with no kept key (O = 0 or only the
        sink), a full row; combined with seqlens / leftpad / window / paging;
        masked slots hold NaN (never read)."""
        from flash_attn_legacy import flash_attn_with_kvcache
        B, max_len = 5, 192
        q, k, v = self._case(B, 1, 8, 2, head_dim, max_len, dtype, "bshd" if mode == "paged" else "bhsd_view")
        seqlens = [192, 150, 60, 192, 33]
        keep = torch.ones(B, max_len, dtype=torch.bool, device="cuda")
        keep[0, 10:20] = keep[0, 100] = keep[0, 191] = False  # holes, last key masked
        keep[1, :70] = False
        keep[1, 120:125] = False                              # left padding + a hole
        keep[2] = False                                       # no key
        keep[4, ::3] = False                                  # every third key
        leftpad = [0, 0, 0, 30, 5] if mode == "leftpad" else None
        causal, window = True, (40, 0) if mode == "window40" else (-1, -1)
        sinks = torch.randn(8, device="cuda") if mode == "sinks" else None
        cs = torch.tensor(seqlens, dtype=torch.int32, device="cuda")
        kw = dict(cache_seqlens=cs, causal=causal, window_size=window, num_splits=num_splits, sinks=sinks,
                  key_mask=keep, return_softmax_lse=True)
        if leftpad is not None:
            kw["cache_leftpad"] = torch.tensor(leftpad, dtype=torch.int32, device="cuda")
        ref_out, ref_lse = self._ref(q, k, v, seqlens, 1.0 / math.sqrt(head_dim), causal=causal, window=window,
                                     sinks=sinks, leftpad=leftpad, key_mask=keep)
        self._poison(k, v, seqlens)
        k.masked_fill_(~keep[:, :, None, None], float("nan"))
        v.masked_fill_(~keep[:, :, None, None], float("nan"))
        kc, vc = k, v
        if mode == "paged":  # blocks made from the poisoned caches
            kc, vc, kw["block_table"] = self._paged(k, v, 64)
        with torch.no_grad():
            out, lse = flash_attn_with_kvcache(q, kc, vc, **kw)
        self._check(out, lse, ref_out, ref_lse, dtype)
        if sinks is None:
            assert (out[2] == 0).all()

    def test_key_mask_layouts_and_checked(self):
        """A [1, max_len] mask expanded to the batch (stride 0), a strided view
        (copied), q_len 3 with causal rows; wrong masks raise."""
        from flash_attn_legacy import flash_attn_with_kvcache
        q, k, v = self._case(3, 3, 4, 2, 64, 64, torch.float16, "bshd")
        keep = torch.rand(1, 64, device="cuda") > 0.3
        strided = (torch.rand(3, 128, device="cuda") > 0.3)[:, ::2]
        with torch.no_grad():
            for mask in (keep.expand(3, -1), strided):
                out = flash_attn_with_kvcache(q, k, v, cache_seqlens=60, causal=True, key_mask=mask)
                ref_out, _ = self._ref(q, k, v, [60] * 3, 0.125, causal=True, key_mask=mask)
                torch.testing.assert_close(out.float(), ref_out, atol=2e-2, rtol=2e-2)
            for bad in (keep.int(), keep[:, :63].expand(3, -1), keep.expand(2, -1), keep.cpu().expand(3, -1)):
                with pytest.raises(RuntimeError, match="key_mask"):
                    flash_attn_with_kvcache(q, k, v, key_mask=bad)

    def test_leftpad_clamped_and_checked(self):
        """leftpad outside [0, seqlens] is clamped; an int cache_seqlens with a tensor
        leftpad works; wrong shapes or types raise."""
        from flash_attn_legacy import flash_attn_with_kvcache
        q, k, v = self._case(3, 1, 4, 4, 64, 64, torch.float16, "bshd")
        lp = torch.tensor([-5, 100, 10], dtype=torch.int32, device="cuda")
        with torch.no_grad():
            out = flash_attn_with_kvcache(q, k, v, cache_seqlens=50, cache_leftpad=lp)
            ref_out, _ = self._ref(q, k, v, [50, 50, 50], 0.125, leftpad=[0, 50, 10])
            torch.testing.assert_close(out.float(), ref_out, atol=2e-2, rtol=2e-2)
            with pytest.raises(RuntimeError, match="cache_leftpad"):
                flash_attn_with_kvcache(q, k, v, cache_leftpad=torch.zeros(4, dtype=torch.int32, device="cuda"))
            for bad in ([0, 0, 0], torch.zeros(3, device="cuda")):
                with pytest.raises(ValueError, match="cache_leftpad"):
                    flash_attn_with_kvcache(q, k, v, cache_leftpad=bad)

    @staticmethod
    def _paged(k, v, block_size, layout="nhd", seed=0):
        """Contiguous caches [B, max_len, H_kv, d] scattered into shuffled blocks:
        (k_blocks, v_blocks [num_blocks, block_size, H_kv, d], block_table [B, nb]).
        layout "hnd": blocks stored [num_blocks, H_kv, block_size, d], passed as a
        transposed view; "vllm": k and v as slices of one [2, num_blocks, ...]."""
        B, max_len, H_kv, d = k.shape
        nb = max_len // block_size
        n_blocks = B * nb + 3  # spare blocks (NaN): never read
        perm = torch.randperm(n_blocks, generator=torch.Generator().manual_seed(seed))[:B * nb]
        table = perm.view(B, nb).to(torch.int32).cuda()
        if layout == "hnd":
            kb, vb = (torch.full((n_blocks, H_kv, block_size, d), float("nan"), device="cuda",
                                 dtype=k.dtype).transpose(1, 2) for _ in range(2))
        elif layout == "vllm":
            kv = torch.full((2, n_blocks, block_size, H_kv, d), float("nan"), device="cuda", dtype=k.dtype)
            kb, vb = kv[0], kv[1]
        else:
            kb, vb = (torch.full((n_blocks, block_size, H_kv, d), float("nan"), device="cuda", dtype=k.dtype)
                      for _ in range(2))
        kb[table.flatten().long()] = k.reshape(B * nb, block_size, H_kv, d)
        vb[table.flatten().long()] = v.reshape(B * nb, block_size, H_kv, d)
        return kb, vb, table

    @pytest.mark.parametrize("block_size", [16, 64, 256])
    @pytest.mark.parametrize("num_splits", [1, 0, 3, 5])  # 3 at block 256: split edges inside blocks
    @pytest.mark.parametrize("q_len,H_q", [(1, 4), (1, 8), (2, 8)], ids=["rows2", "rows4", "rows8"])
    def test_paged(self, head_dim, block_size, num_splits, q_len, H_q):
        """block_table (paged cache) gives the contiguous cache's results."""
        from flash_attn_legacy import flash_attn_with_kvcache
        q, k, v = self._case(3, q_len, H_q, 2, head_dim, 512, torch.float16, "bshd")
        seqlens = [512, 200, 1]
        cs = torch.tensor(seqlens, dtype=torch.int32, device="cuda")
        kb, vb, table = self._paged(k, v, block_size)
        self._poison(k, v, seqlens)
        sinks = torch.randn(H_q, device="cuda")
        with torch.no_grad():
            paged, lse_p = flash_attn_with_kvcache(q, kb, vb, cache_seqlens=cs, block_table=table, causal=True,
                                                   window_size=(300, 0), num_splits=num_splits, sinks=sinks,
                                                   return_softmax_lse=True)
        ref_out, ref_lse = self._ref(q, k, v, seqlens, 1.0 / math.sqrt(head_dim), causal=True, window=(300, 0),
                                     sinks=sinks)
        self._check(paged, lse_p, ref_out, ref_lse, torch.float16)

    @pytest.mark.parametrize("layout", ["hnd", "vllm"])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
    def test_paged_layouts(self, layout, dtype):
        """Paged caches as views (HND blocks, vLLM's [2, num_blocks, ...] slices),
        an int64 table with padding entries (-1) past the valid blocks, and
        cache_seqlens past the cache (clamped)."""
        from flash_attn_legacy import flash_attn_with_kvcache
        q, k, v = self._case(2, 1, 8, 2, 128, 256, dtype, "bshd")
        kb, vb, table = self._paged(k, v, 64, layout)
        seqlens = [256, 70]
        table = table.long()
        table[1, 2:] = -1  # padding entries: never read (keys >= 128 of row 1)
        cs = torch.tensor([10 ** 6, 70], dtype=torch.int32, device="cuda")
        with torch.no_grad():
            out, lse = flash_attn_with_kvcache(q, kb, vb, cache_seqlens=cs, block_table=table,
                                               return_softmax_lse=True)
        ref_out, ref_lse = self._ref(q, k, v, seqlens, 1.0 / math.sqrt(128))
        self._check(out, lse, ref_out, ref_lse, dtype)

    def test_paged_rejects(self):
        from flash_attn_legacy import flash_attn_with_kvcache
        q, k, v = self._case(2, 1, 4, 4, 64, 64, torch.float16, "bshd")
        kb, vb, table = self._paged(k, v, 16)
        with torch.no_grad():
            with pytest.raises(RuntimeError, match="leftpad"):
                flash_attn_with_kvcache(q, kb, vb, block_table=table,
                                        cache_leftpad=torch.zeros(2, dtype=torch.int32, device="cuda"))
            with pytest.raises(RuntimeError, match="multiple of 16"):
                flash_attn_with_kvcache(q, kb[:, :8].contiguous(), vb[:, :8].contiguous(), block_table=table)
            with pytest.raises(RuntimeError, match="no blocks"):
                flash_attn_with_kvcache(q, kb[:0], vb[:0], block_table=table)
            with pytest.raises(RuntimeError, match="block_table"):
                flash_attn_with_kvcache(q, kb, vb, block_table=table[:1])
            with pytest.raises(ValueError, match="block_table"):
                flash_attn_with_kvcache(q, kb, vb, block_table=table[0])
            with pytest.raises(ValueError, match="cache_seqlens"):
                flash_attn_with_kvcache(q, kb, vb, block_table=table, cache_seqlens=10 ** 6)

    def test_rejects(self):
        from flash_attn_legacy import flash_attn_with_kvcache
        q, k, v = self._case(1, 1, 4, 4, 64, 32, torch.float16, "bshd")
        with torch.no_grad():
            for kw in ({"k": k}, {"cache_batch_idx": torch.zeros(1, dtype=torch.int32, device="cuda")}):
                with pytest.raises(NotImplementedError):
                    flash_attn_with_kvcache(q, k, v, **kw)
            with pytest.raises(NotImplementedError):  # 16 query rows per KV head: not yet
                flash_attn_with_kvcache(torch.randn(1, 4, 4, 64, device="cuda", dtype=torch.float16),
                                        k[:, :, :1], v[:, :, :1])
            buf = torch.randn(32 * 260, device="cuda", dtype=torch.float16)
            k_odd = buf.as_strided((1, 32, 4, 64), (32 * 260, 260, 64, 1))  # key rows 260 elements apart
            with pytest.raises(RuntimeError, match="multiples of 8"):
                flash_attn_with_kvcache(q, k_odd, v)
        q.requires_grad_(True)
        with pytest.raises(RuntimeError, match="inference only"):
            flash_attn_with_kvcache(q, k, v)


class TestValidation:
    """Unsupported inputs must raise (not assert, not read out of bounds)."""

    def _qkv(self, n_q=64, n_k=64, d=64, dtype=torch.float16, requires_grad=False):
        mk = lambda n: torch.randn(1, 4, n, d, device="cuda", dtype=dtype, requires_grad=requires_grad)
        return mk(n_q), mk(n_k), mk(n_k)

    def test_wrong_dtype_raises(self):
        from flash_attn_legacy import flash_attention
        q, k, v = self._qkv(dtype=torch.float64)
        with pytest.raises(TypeError):
            flash_attention(q, k, v)

    def test_mixed_dtypes_raise(self):
        from flash_attn_legacy import flash_attention
        q, k, v = self._qkv()
        with pytest.raises(TypeError):
            flash_attention(q, k.bfloat16(), v)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
    @pytest.mark.parametrize("qk_mag,v_mag", [(1.0, 1.0), (1e3, 1e6), (1e-3, 1e-6), (300.0, 7e4)],
                             ids=["unit", "huge", "tiny", "beyond_fp16"])
    def test_non_fp16_inference_scaled(self, dtype, qk_mag, v_mag):
        """bf16 / fp32 inputs of any magnitude: scaled by powers of 2 into fp16 on
        the device (no host sync), output in dtype, relative accuracy of fp16."""
        from flash_attn_legacy import flash_attention
        torch.manual_seed(0)
        q, k, v = (torch.randn(2, 4, 130, 64, device="cuda", dtype=dtype) for _ in range(3))
        q, k, v = q * qk_mag, k * qk_mag, v * v_mag
        scale = 0.125 / (qk_mag * qk_mag)  # keeps the logits O(1) whatever the input magnitude
        with no_host_sync():
            out = flash_attention(q, k, v, softmax_scale=scale, causal=True)
        s = (q.double() @ k.double().transpose(-1, -2)) * scale
        s = s.masked_fill(ref_hidden(130, 130, True, (-1, -1), "cuda"), float("-inf"))
        ref = torch.softmax(s, -1) @ v.double()
        assert out.dtype == dtype
        torch.testing.assert_close(out.double(), ref, atol=2e-2 * v_mag, rtol=2e-2)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
    def test_non_fp16_varlen_scaled(self, dtype):
        from flash_attn_legacy import flash_attn_varlen_func
        torch.manual_seed(0)
        lens = [5, 70, 33]
        cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device="cuda")
        q, k, v = (torch.randn(sum(lens), 4, 64, device="cuda", dtype=dtype) for _ in range(3))
        v = v * 1e5  # beyond fp16
        out = flash_attn_varlen_func(q, k, v, cu, cu, max(lens), max(lens), causal=True)
        s0 = 0
        for n in lens:
            def seq(t):
                return t[s0:s0 + n].permute(1, 0, 2).unsqueeze(0).float()
            ref = ref_attention(seq(q), seq(k), seq(v) / 1e5, 0.125, True).float() * 1e5
            torch.testing.assert_close(out[s0:s0 + n].float(), ref.squeeze(0).permute(1, 0, 2),
                                       atol=2e-2 * 1e5, rtol=2e-2)
            s0 += n

    @pytest.mark.parametrize("n_q,n_k", [(0, 16), (16, 0), (0, 0)])
    def test_empty_sequences(self, n_q, n_k):
        """No 0-sized kernel launch: empty output, or zeros when there are no keys."""
        from flash_attn_legacy import flash_attention
        q, k, v = self._qkv(n_q=n_q, n_k=n_k, requires_grad=True)
        out = flash_attention(q, k, v)
        assert out.shape == (1, 4, n_q, 64) and (out == 0).all()
        out.sum().backward()
        for t in (q, k, v):
            assert t.grad.shape == t.shape and (t.grad == 0).all()

    @pytest.mark.parametrize("bad_k", [False, True], ids=["q_and_k", "k_only"])
    @pytest.mark.parametrize("edges,total", [([0, 10, 20], 30), ([2, 10, 30], 30), ([0, 20, 10, 30], 30)])
    def test_varlen_invalid_cu_seqlens_raise(self, edges, total, bad_k):
        from flash_attn_legacy import flash_attn_varlen_func
        q = torch.randn(total, 4, 64, device="cuda", dtype=torch.float16)
        cu = torch.tensor(edges, dtype=torch.int32, device="cuda")
        good = torch.tensor([0] + [total * (i + 1) // (len(edges) - 1) for i in range(len(edges) - 1)],
                            dtype=torch.int32, device="cuda")
        with pytest.raises(RuntimeError, match="cu_seqlens"):
            flash_attn_varlen_func(q, q, q, good if bad_k else cu, cu, 30, 30)

    def test_varlen_no_query_tokens(self):
        from flash_attn_legacy import flash_attn_varlen_func
        q = torch.randn(0, 4, 64, device="cuda", dtype=torch.float16)
        k = torch.randn(10, 4, 64, device="cuda", dtype=torch.float16)
        cu_q = torch.tensor([0, 0], dtype=torch.int32, device="cuda")
        cu_k = torch.tensor([0, 10], dtype=torch.int32, device="cuda")
        assert flash_attn_varlen_func(q, k, k, cu_q, cu_k, 0, 10).shape == (0, 4, 64)

    @pytest.mark.parametrize("arch", ["native", "pascal"])
    def test_no_tile_fits_raises_clear_error(self, monkeypatch, arch):
        """FLASH_ATTN_LEGACY_SMEM_LIMIT below every d=256 tile: explicit error, no failed launch."""
        from flash_attn_legacy import flash_attention
        if arch == "pascal":
            monkeypatch.setenv("FLASH_ATTN_LEGACY_FORCE_ARCH", "pascal")
        monkeypatch.setenv("FLASH_ATTN_LEGACY_SMEM_LIMIT", "32768")
        q, k, v = self._qkv(d=256)
        with pytest.raises(RuntimeError, match="no .* tile for head_dim 256 fits 32768"):
            flash_attention(q, k, v)

    @pytest.mark.parametrize("bad", [float("inf"), float("nan")], ids=["inf", "nan"])
    def test_non_finite_stays_local(self, bad):
        """An inf / NaN in one head of bf16 inputs: that tensor gets a plain cast
        (no scale computed from a non-finite peak), the other heads are exact."""
        from flash_attn_legacy import flash_attention
        torch.manual_seed(0)
        q, k, v = (torch.randn(1, 4, 64, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
        k[0, 2, 5, 3] = bad
        out = flash_attention(q, k, v, causal=True)
        ref = ref_attention(q.float(), k.float(), v.float(), 0.125, True).float()
        keep = [0, 1, 3]
        torch.testing.assert_close(out[:, keep].float(), ref[:, keep], atol=2e-2, rtol=2e-2)

    @pytest.mark.parametrize("what", ["zero_qk", "zero_v", "softcap_sinks"])
    def test_non_fp16_special_inputs(self, what):
        """All-zero tensors (scale 2^126, still zero) and softcap / sinks, whose
        logits use the corrected softmax scale, with fp32 inputs at large scale."""
        from flash_attn_legacy import flash_attention
        torch.manual_seed(0)
        q, k, v = (torch.randn(2, 4, 50, 64, device="cuda", dtype=torch.float32) for _ in range(3))
        kw, ref_kw = {}, {}
        if what == "zero_qk":
            q, k = q * 0, k * 0
        elif what == "zero_v":
            v = v * 0
        else:
            q, k = q * 100, k * 100
            sinks = torch.randn(4, device="cuda")
            kw = dict(softcap=5.0, sinks=sinks)
            ref_kw = dict(softcap=5.0, sinks=sinks)
        scale = 0.125 / (1e4 if what == "softcap_sinks" else 1)
        out = flash_attention(q, k, v, softmax_scale=scale, causal=True, **kw)
        ref = ref_attention(q, k, v, scale, True, **ref_kw).float()
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)

    def test_non_fp16_varlen_no_sync(self):
        from flash_attn_legacy import flash_attn_varlen_func
        q = torch.randn(50, 4, 64, device="cuda", dtype=torch.bfloat16)
        cu = torch.tensor([0, 20, 50], dtype=torch.int32, device="cuda")
        with no_host_sync():
            flash_attn_varlen_func(q, q, q, cu, cu, 30, 30, causal=True, validate=False)

    def test_non_fp16_sink_gradient_raises(self):
        """bf16 inputs with trainable sinks is training: not supported (no silent
        backward through scaled tensors)."""
        from flash_attn_legacy import flash_attention
        q = torch.randn(1, 4, 16, 64, device="cuda", dtype=torch.bfloat16)
        sinks = torch.zeros(4, device="cuda", requires_grad=True)
        with pytest.raises(NotImplementedError):
            flash_attention(q, q, q, sinks=sinks)

    def test_nan_propagates(self):
        """A NaN input gives NaN outputs, as eager attention (no exception)."""
        from flash_attn_legacy import flash_attention
        q, k, v = self._qkv(dtype=torch.float32)
        k[0, 0, 0, 0], k[0, 0, 1, 1] = float("nan"), 1e5
        assert torch.isnan(flash_attention(q, k, v)).any()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
    def test_mha_module_non_fp16(self, dtype):
        """FlashMultiHeadAttention: any magnitude in inference, warns in training."""
        from flash_attn_legacy import FlashMultiHeadAttention
        mha = FlashMultiHeadAttention(256, num_heads=4, causal=True).cuda().to(dtype)
        x = torch.randn(1, 32, 256, device="cuda", dtype=dtype)
        with torch.no_grad():
            assert mha(x).dtype == dtype
            assert torch.isfinite(mha(x * 1e6)).all()
        with pytest.warns(UserWarning, match="GradScaler"):
            mha(x).sum().backward()

    def test_non_fp16_training_raises(self):
        from flash_attn_legacy import flash_attention
        q, k, v = self._qkv(dtype=torch.bfloat16, requires_grad=True)
        with pytest.raises(NotImplementedError):
            flash_attention(q, k, v)

    @pytest.mark.parametrize("arch", ["pascal", "volta", "native"])
    def test_large_qk_products_do_not_overflow(self, monkeypatch, arch):
        """q_i * k_i > 65504 for in-range fp16 inputs: products must not be formed in
        fp16 (the Pascal path used __hmul2). FORCE_ARCH runs either path on any GPU;
        native = the GPU's own family (mma.sync on a T4)."""
        from flash_attn_legacy import flash_attention
        if arch == "native":
            monkeypatch.delenv("FLASH_ATTN_LEGACY_FORCE_ARCH", raising=False)
        else:
            monkeypatch.setenv("FLASH_ATTN_LEGACY_FORCE_ARCH", arch)
        torch.manual_seed(0)
        q, k, v = self._qkv(n_q=32, n_k=32)
        q[..., 0], k[..., 0] = 300.0, 300.0  # one product of 9e4 per (q, k) pair
        scale = 1e-4  # keeps the scores themselves moderate
        if arch == "volta" and torch.cuda.get_device_capability()[0] < 7:
            with pytest.raises(RuntimeError, match="tensor cores"):
                flash_attention(q, k, v, softmax_scale=scale)
            return
        out = flash_attention(q, k, v, softmax_scale=scale)
        ref = ref_attention(q, k, v, scale)
        assert torch.isfinite(out).all()
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)

    def test_bad_head_dim_raises(self):
        from flash_attn_legacy import flash_attention
        q, k, v = self._qkv(d=48)
        with pytest.raises(ValueError):
            flash_attention(q, k, v)

    @pytest.mark.parametrize("N", [1, 17, 33, 100])
    def test_backward_ragged_tail_no_nan(self, N):
        """Q-row padding in the last tile (N not a multiple of Br) must not leak NaN into dK/dV."""
        from flash_attn_legacy import flash_attention
        q, k, v = self._qkv(n_q=N, n_k=N, requires_grad=True)
        flash_attention(q, k, v, causal=True).sum().backward()
        for t in (q.grad, k.grad, v.grad):
            assert not torch.isnan(t).any()

    def test_is_causal_alias_deprecated(self):
        """`is_causal` still works but warns; result equals `causal`."""
        from flash_attn_legacy import flash_attention
        q, k, v = self._qkv()
        with pytest.warns(DeprecationWarning):
            old = flash_attention(q, k, v, is_causal=True)
        new = flash_attention(q, k, v, causal=True)
        assert torch.equal(old, new)


# ---------------------------------------------------------------------------
# Forward contract (docs/PLAN.md P2.8, step 0e): what any forward kernel must
# keep — the logsumexp the backward consumes, ragged tile edges, bands of rows
# that see no key, short varlen segments.
# ---------------------------------------------------------------------------

def ref_lse(q, k, scale, causal):
    """FP32 logsumexp per row [B, H_q, N_q]; -inf for rows that see no key."""
    reps = q.shape[1] // k.shape[1]
    s = q.float() @ k.float().repeat_interleave(reps, dim=1).transpose(-2, -1) * scale
    if causal:
        n_q, n_k = q.shape[-2], k.shape[-2]
        s.masked_fill_(torch.ones(n_q, n_k, device=q.device, dtype=torch.bool)
                       .triu(1 + n_k - n_q), float("-inf"))
    return torch.logsumexp(s, dim=-1)


def assert_lse_close(lse, ref):
    finite = torch.isfinite(ref)
    assert torch.equal(torch.isfinite(lse), finite), "rows with no visible key must give lse = -inf"
    assert (lse[~finite] == float("-inf")).all()
    torch.testing.assert_close(lse[finite], ref[finite], atol=2e-3, rtol=1e-3)


class TestForwardContract:

    @pytest.mark.parametrize("n_q,n_k", [(15, 15), (17, 17), (63, 63), (65, 65), (127, 127),
                                         (200, 5), (5, 200), (130, 17)])
    @pytest.mark.parametrize("causal", [False, True], ids=["non_causal", "causal"])
    def test_output_and_lse(self, head_dim, n_q, n_k, causal):
        """Output and raw logsumexp vs FP32 at ragged lengths; (200, 5) causal
        leaves whole tiles of rows with no visible key (O = 0, lse = -inf)."""
        torch.manual_seed(0)
        scale = 1.0 / math.sqrt(head_dim)
        q = torch.randn(2, 4, n_q, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(2, 2, n_k, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(2, 2, n_k, head_dim, device="cuda", dtype=torch.float16)
        out, lse = kernel_ops().forward(q, k, v, scale, causal)
        torch.testing.assert_close(out, ref_attention(q, k, v, scale, causal), atol=2e-2, rtol=2e-2)
        assert_lse_close(lse, ref_lse(q, k, scale, causal))

    @pytest.mark.parametrize("causal", [False, True], ids=["non_causal", "causal"])
    def test_varlen_short_segments(self, head_dim, causal):
        """Segments shorter than, equal to and just above a tile, mixed in one pack."""
        torch.manual_seed(0)
        scale = 1.0 / math.sqrt(head_dim)
        lens = [1, 15, 17, 65, 16, 64, 2]
        cu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32, device="cuda")
        q, k, v = (torch.randn(sum(lens), 4, head_dim, device="cuda", dtype=torch.float16) for _ in range(3))
        out, lse = kernel_ops().forward_varlen(q, k, v, cu, cu, max(lens), max(lens), scale, causal)
        for s, e in zip(cu[:-1].tolist(), cu[1:].tolist()):
            qi, ki, vi = (t[s:e].transpose(0, 1).unsqueeze(0) for t in (q, k, v))
            ref = ref_attention(qi, ki, vi, scale, causal)[0].transpose(0, 1)
            torch.testing.assert_close(out[s:e], ref, atol=2e-2, rtol=2e-2)
            assert_lse_close(lse[s:e].transpose(0, 1), ref_lse(qi, ki, scale, causal)[0])

    @pytest.mark.parametrize("d", [128, 256])
    def test_64k_fallback_tile(self, monkeypatch, d):
        """The 64 KB fallback tiles (what a 64 KB GPU runs) also on GPUs with more
        shared memory, so every tile of every list is exercised (the WMMA ones on
        a V100, or on a T4 with FORCE_ARCH=volta)."""
        monkeypatch.setenv("FLASH_ATTN_LEGACY_SMEM_LIMIT", "65536")
        self.test_output_and_lse(d, 130, 200, True)
        self.test_varlen_short_segments(d, True)

    @pytest.mark.parametrize("causal", [False, True], ids=["non_causal", "causal"])
    def test_misaligned_views(self, causal):
        """Contiguous views starting 2 bytes into their storage (not 16-byte
        aligned): the host copies them, results are unchanged."""
        from flash_attn_legacy import flash_attention
        torch.manual_seed(0)
        shape, n = (2, 4, 77, 64), 2 * 4 * 77 * 64
        q, k, v, go = (torch.randn(n + 1, device="cuda", dtype=torch.float16)[1:].view(shape)
                       for _ in range(4))
        assert q.data_ptr() % 16 != 0 and q.is_contiguous()
        q, k, v = (t.requires_grad_() for t in (q, k, v))
        out = flash_attention(q, k, v, causal=causal)
        out.backward(go)
        ref = ref_attention(q.detach(), k.detach(), v.detach(), 1 / 8, causal)
        rdq, rdk, rdv = ref_backward(q.detach(), k.detach(), v.detach(), go, 1 / 8, causal)
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
        for g, r in ((q.grad, rdq), (k.grad, rdk), (v.grad, rdv)):
            torch.testing.assert_close(g, r, atol=5e-2, rtol=5e-2)

    def test_misaligned_varlen(self):
        """Packed varlen input starting 2 bytes into its storage, with GQA."""
        from flash_attn_legacy import flash_attn_varlen_func
        torch.manual_seed(0)
        lens = [5, 40, 17]
        total = sum(lens)
        cu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32, device="cuda")

        def packed(h):
            return torch.randn(total * h * 64 + 1, device="cuda", dtype=torch.float16)[1:].view(total, h, 64)
        q, k, v = packed(4), packed(2), packed(2)
        assert q.data_ptr() % 16 != 0
        q, k, v = (t.requires_grad_() for t in (q, k, v))
        out = flash_attn_varlen_func(q, k, v, cu, cu, max(lens), max(lens), softmax_scale=1 / 8, causal=True)
        go = torch.randn_like(out)
        out.backward(go)
        for s, e in zip(cu[:-1].tolist(), cu[1:].tolist()):
            qi, ki, vi, gi = (t.detach()[s:e].transpose(0, 1).unsqueeze(0) for t in (q, k, v, go))
            ref = ref_attention(qi, ki, vi, 1 / 8, True)
            rdq, rdk, rdv = ref_backward(qi, ki, vi, gi, 1 / 8, True)
            torch.testing.assert_close(out[s:e], ref[0].transpose(0, 1), atol=2e-2, rtol=2e-2)
            for g, r in ((q.grad[s:e], rdq), (k.grad[s:e], rdk), (v.grad[s:e], rdv)):
                torch.testing.assert_close(g, r[0].transpose(0, 1), atol=5e-2, rtol=5e-2)

    @pytest.mark.parametrize("causal", [False, True], ids=["non_causal", "causal"])
    def test_varlen_backward_mixed_segments(self, head_dim, causal):
        """One-launch varlen backward vs per-segment FP32: empty segments, Q and K
        of different lengths in a segment (bottom-right causal), GQA."""
        from flash_attn_legacy import flash_attn_varlen_func
        torch.manual_seed(0)
        lens_q = [3, 0, 17, 65, 1, 40, 70]
        lens_k = [5, 4, 17, 33, 1, 0, 130]  # 130: several key blocks for every tile
        cu_q = torch.tensor([0] + torch.tensor(lens_q).cumsum(0).tolist(), dtype=torch.int32, device="cuda")
        cu_k = torch.tensor([0] + torch.tensor(lens_k).cumsum(0).tolist(), dtype=torch.int32, device="cuda")
        scale = 1.0 / math.sqrt(head_dim)
        q = torch.randn(sum(lens_q), 4, head_dim, device="cuda", dtype=torch.float16, requires_grad=True)
        k = torch.randn(sum(lens_k), 2, head_dim, device="cuda", dtype=torch.float16, requires_grad=True)
        v = torch.randn(sum(lens_k), 2, head_dim, device="cuda", dtype=torch.float16, requires_grad=True)
        out = flash_attn_varlen_func(q, k, v, cu_q, cu_k, max(lens_q), max(lens_k),
                                     softmax_scale=scale, causal=causal)
        go = torch.randn_like(out)
        out.backward(go)
        for i in range(len(lens_q)):
            qs, qe, ks, ke = cu_q[i].item(), cu_q[i + 1].item(), cu_k[i].item(), cu_k[i + 1].item()
            if qe == qs or ke == ks:  # nothing to attend: zero gradients for the segment
                assert not q.grad[qs:qe].any() and not k.grad[ks:ke].any() and not v.grad[ks:ke].any()
                continue
            qi, ki, vi, gi = (t.detach()[a:b].transpose(0, 1).unsqueeze(0)
                              for t, (a, b) in ((q, (qs, qe)), (k, (ks, ke)), (v, (ks, ke)), (go, (qs, qe))))
            rdq, rdk, rdv = ref_backward(qi, ki, vi, gi, scale, causal)
            for g, r in ((q.grad[qs:qe], rdq), (k.grad[ks:ke], rdk), (v.grad[ks:ke], rdv)):
                torch.testing.assert_close(g, r[0].transpose(0, 1), atol=5e-2, rtol=5e-2)

    @pytest.mark.parametrize("empty", ["keys", "queries"])
    def test_varlen_backward_no_keys_or_queries(self, empty):
        """total_k == 0 (no keys anywhere) or total_q == 0: zero gradients, no launch."""
        from flash_attn_legacy import flash_attn_varlen_func
        lens_q, lens_k = ([4, 7], [0, 0]) if empty == "keys" else ([0, 0], [5, 3])
        cu_q = torch.tensor([0] + torch.tensor(lens_q).cumsum(0).tolist(), dtype=torch.int32, device="cuda")
        cu_k = torch.tensor([0] + torch.tensor(lens_k).cumsum(0).tolist(), dtype=torch.int32, device="cuda")
        q = torch.randn(sum(lens_q), 4, 64, device="cuda", dtype=torch.float16, requires_grad=True)
        k, v = (torch.randn(sum(lens_k), 2, 64, device="cuda", dtype=torch.float16, requires_grad=True)
                for _ in range(2))
        out = flash_attn_varlen_func(q, k, v, cu_q, cu_k, max(lens_q), max(lens_k), causal=True)
        out.sum().backward()
        for t in (q, k, v):
            assert t.grad.shape == t.shape and not t.grad.any()

    def test_many_heads_grid(self):
        """B * H = 66000 > 65535 (the old gridDim.y limit), forward and backward,
        several query blocks per head."""
        from flash_attn_legacy import flash_attention
        torch.manual_seed(0)
        q, k, v = (torch.randn(1100, 60, 200, 32, device="cuda", dtype=torch.float16, requires_grad=True)
                   for _ in range(3))  # N=200: two query blocks per head
        out = flash_attention(q, k, v, causal=True)
        out.backward(torch.ones_like(out))
        scale = 32 ** -0.5
        for b in (0, 1099):  # spot-check the first and last batch
            sl = slice(b, b + 1)
            torch.testing.assert_close(out[sl], ref_attention(q[sl].detach(), k[sl].detach(), v[sl].detach(),
                                                              scale, True), atol=2e-2, rtol=2e-2)
            rdq, rdk, rdv = ref_backward(q[sl].detach(), k[sl].detach(), v[sl].detach(),
                                         torch.ones_like(out[sl]), scale, True)
            for g, r in ((q.grad[sl], rdq), (k.grad[sl], rdk), (v.grad[sl], rdv)):
                torch.testing.assert_close(g, r, atol=5e-2, rtol=5e-2)

    def test_over_2_31_elements_varlen(self):
        """Packed varlen tensors with more than 2^31 elements (64-bit offsets);
        checks the last sequences, whose offsets overflow 32-bit math."""
        from flash_attn_legacy import flash_attn_varlen_func
        free, _ = torch.cuda.mem_get_info()
        if free < 10 * 2**30:
            pytest.skip("needs about 10 GB of free GPU memory")
        H, d, seq = 32, 64, 16
        total = (2**31 // (H * d)) + 4 * seq  # just past 2^31 elements per tensor
        total -= total % seq
        cu = torch.arange(0, total + 1, seq, dtype=torch.int32, device="cuda")
        torch.manual_seed(0)
        q = torch.randn(total, H, d, device="cuda", dtype=torch.float16)
        out = flash_attn_varlen_func(q, q, q, cu, cu, seq, seq, causal=True)
        tail = q[-2 * seq:].view(2, seq, H, d).transpose(1, 2)  # last two sequences as [2, H, seq, d]
        ref = ref_attention(tail, tail, tail, d ** -0.5, True).transpose(1, 2).reshape(2 * seq, H, d)
        torch.testing.assert_close(out[-2 * seq:], ref, atol=2e-2, rtol=2e-2)

    def test_over_2_31_elements_batched(self):
        """Batched Q / O beyond 2^31 elements (64-bit batch / head offsets) with a
        short K/V; checks the last batch and head."""
        from flash_attn_legacy import flash_attention
        free, _ = torch.cuda.mem_get_info()
        if free < 10 * 2**30:
            pytest.skip("needs about 10 GB of free GPU memory")
        B, H, d, n_k = 2, 32, 128, 16
        n_q = 2**31 // (B * H * d) + 128  # just past 2^31 elements
        torch.manual_seed(0)
        q = torch.randn(B, H, n_q, d, device="cuda", dtype=torch.float16)
        k, v = (torch.randn(B, H, n_k, d, device="cuda", dtype=torch.float16) for _ in range(2))
        out = flash_attention(q, k, v)
        sl = (slice(B - 1, B), slice(H - 1, H), slice(n_q - 300, n_q))
        ref = ref_attention(q[sl], k[sl[:2]], v[sl[:2]], d ** -0.5)
        torch.testing.assert_close(out[sl], ref, atol=2e-2, rtol=2e-2)

    @pytest.mark.parametrize("causal", [False, True], ids=["non_causal", "causal"])
    def test_varlen_without_validation_matches(self, causal):
        """validate=False (no host sync; the grid is an upper bound whose extra
        blocks exit at once): same output and gradients as the validated call."""
        from flash_attn_legacy import flash_attn_varlen_func
        torch.manual_seed(0)
        lens = [3, 0, 200, 17, 1, 64]
        cu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32, device="cuda")
        q, k, v = (torch.randn(sum(lens), 4, 64, device="cuda", dtype=torch.float16, requires_grad=True)
                   for _ in range(3))
        go = torch.randn(sum(lens), 4, 64, device="cuda", dtype=torch.float16)
        results = []
        for validate in (True, False):
            out = flash_attn_varlen_func(q, k, v, cu, cu, max(lens), max(lens), causal=causal, validate=validate)
            grads = torch.autograd.grad(out, (q, k, v), go)
            results.append((out, *grads))
        torch.testing.assert_close(results[0][0], results[1][0], atol=0, rtol=0)  # forward: exact
        for a, b in zip(results[0][1:], results[1][1:]):  # dQ accumulates with atomics: order varies
            torch.testing.assert_close(a, b, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("causal", [False, True], ids=["non_causal", "causal"])
    def test_varlen_without_validation_cross_lengths(self, causal):
        """validate=False with cu_q != cu_k (q shorter than k, a sequence with keys
        but no queries, one with queries but no keys) and no queries at all."""
        from flash_attn_legacy import flash_attn_varlen_func
        torch.manual_seed(0)
        lens_q, lens_k = [5, 0, 70, 3], [9, 4, 130, 0]
        cu_q = torch.tensor([0] + torch.tensor(lens_q).cumsum(0).tolist(), dtype=torch.int32, device="cuda")
        cu_k = torch.tensor([0] + torch.tensor(lens_k).cumsum(0).tolist(), dtype=torch.int32, device="cuda")
        q = torch.randn(sum(lens_q), 4, 64, device="cuda", dtype=torch.float16, requires_grad=True)
        k, v = (torch.randn(sum(lens_k), 2, 64, device="cuda", dtype=torch.float16, requires_grad=True)
                for _ in range(2))
        go = torch.randn(sum(lens_q), 4, 64, device="cuda", dtype=torch.float16)
        res = []
        for validate in (True, False):
            out = flash_attn_varlen_func(q, k, v, cu_q, cu_k, max(lens_q), max(lens_k),
                                         causal=causal, validate=validate)
            res.append((out, *torch.autograd.grad(out, (q, k, v), go)))
        torch.testing.assert_close(res[0][0], res[1][0], atol=0, rtol=0)
        for a, b in zip(res[0][1:], res[1][1:]):
            torch.testing.assert_close(a, b, atol=1e-3, rtol=1e-3)
        empty_q = q[:0].detach()
        cu0 = torch.zeros(5, dtype=torch.int32, device="cuda")
        out = flash_attn_varlen_func(empty_q, k.detach(), v.detach(), cu0, cu_k, 1, max(lens_k),
                                     causal=causal, validate=False)
        assert out.shape == (0, 4, 64)

    def test_fp32_training_under_autocast(self):
        """The direct API accepts fp32 inputs with gradients under fp16 autocast."""
        from flash_attn_legacy import flash_attention
        q, k, v = (torch.randn(1, 2, 40, 64, device="cuda", requires_grad=True) for _ in range(3))
        with torch.autocast("cuda", dtype=torch.float16):
            out = flash_attention(q, k, v, causal=True)
        out.float().sum().backward()
        ref = ref_attention(q.detach().half(), k.detach().half(), v.detach().half(), 1 / 8, True)
        torch.testing.assert_close(out.half(), ref, atol=2e-2, rtol=2e-2)
        assert q.grad.dtype == torch.float32 and torch.isfinite(q.grad).all()

    def test_long_d256(self):
        """d=256 at 4096 tokens: the largest register / smem footprint."""
        torch.manual_seed(0)
        q, k, v = (torch.randn(1, 2, 4096, 256, device="cuda", dtype=torch.float16) for _ in range(3))
        scale = 1.0 / 16
        out, lse = kernel_ops().forward(q, k, v, scale, True)
        torch.testing.assert_close(out, ref_attention(q, k, v, scale, True), atol=2e-2, rtol=2e-2)
        assert_lse_close(lse, ref_lse(q, k, scale, True))


compile_ready = pytest.mark.skipif(not hasattr(torch.library, "opcheck"), reason="torch >= 2.4 required")


@compile_ready
class TestCompile:
    """torch.compile (docs/analysis/compile_design.md): the kernels are dispatcher
    ops with fake implementations, traced as opaque nodes. Backend aot_eager: it
    traces through the fakes and AOT autograd on every GPU (inductor's Triton needs
    SM >= 7.0); fullgraph=True fails on any graph break."""

    @staticmethod
    def _compiled(fn):
        torch._dynamo.reset()
        return torch.compile(fn, backend="aot_eager", fullgraph=True)

    @staticmethod
    def _qkv(shape_q, shape_kv, dtype=torch.float16, grad=False):
        torch.manual_seed(0)
        return tuple(torch.randn(s, device="cuda", dtype=dtype, requires_grad=grad)
                     for s in (shape_q, shape_kv, shape_kv))

    def _op_samples(self):
        ops = kernel_ops()
        q, k, v = self._qkv((2, 4, 70, 64), (2, 2, 70, 64))
        sinks = torch.randn(4, device="cuda")
        o, lse = ops.forward(q, k, v, 0.125, True, -1, -1, 0.0, sinks)
        lens = [5, 64, 1]
        cu = torch.tensor([0, 5, 69, 70], dtype=torch.int32, device="cuda")
        qp, kp, vp = (t[0].transpose(0, 1).contiguous() for t in (q, k, v))
        op, lsep = ops.forward_varlen(qp, kp, vp, cu, cu, max(lens), max(lens), 0.125, True)
        qd = torch.randn(2, 1, 4, 64, device="cuda", dtype=torch.float16)
        cu0 = torch.zeros(2, dtype=torch.int32, device="cuda")  # one empty sequence
        seqlens = torch.tensor([70, 3], dtype=torch.int32, device="cuda")
        return [
            (ops.forward.default, (q, k, v, 0.125, True, -1, -1, 0.0, sinks, None)),
            (ops.backward.default, (torch.randn_like(o), q, k, v, o, lse, 0.125, True, -1, -1, 0.0, sinks)),
            (ops.backward.default, (torch.randn_like(o), q, k, v, o, lse, 0.125, True, 31, 0, 0.0, None)),
            (ops.forward_varlen.default, (qp, kp, vp, cu, cu, 64, 64, 0.125, True, -1, -1, 0.0, None, None, False)),
            (ops.backward_varlen.default, (torch.randn_like(op), qp, kp, vp, op, lsep, cu, cu, 64, 64, 0.125, True,
                                           -1, -1, 0.0, None, False)),
            (ops.with_kvcache.default, (qd, k.transpose(1, 2), v.transpose(1, 2), seqlens, 0.125, True, -1, -1,
                                        0.0, sinks, 0, None, None)),
            # early returns: no keys (dsinks [H_q]), no batch rows, an empty pack
            (ops.forward.default, (q, k[:, :, :0], v[:, :, :0], 0.125, False, -1, -1, 0.0, sinks, None)),
            (ops.backward.default, (torch.randn_like(o), q, k[:, :, :0].contiguous(), v[:, :, :0].contiguous(), o,
                                    lse, 0.125, False, -1, -1, 0.0, sinks)),
            (ops.forward.default, (q[:0], k[:0], v[:0], 0.125, True, -1, -1, 0.0, None, None)),
            (ops.forward_varlen.default, (qp[:0], kp[:0], vp[:0], cu0, cu0, 0, 0, 0.125, True, -1, -1, 0.0,
                                          None, None, False)),
            (ops.with_kvcache.default, (qd, k.transpose(1, 2), v.transpose(1, 2), None, 0.125, True, -1, -1, 0.0,
                                        None, 0, None, None, None, 30)),  # int cache_seqlens: no tensor
            (ops.with_kvcache.default, (qd[:0], k[:0].transpose(1, 2), v[:0].transpose(1, 2), seqlens[:0], 0.125,
                                        True, -1, -1, 0.0, None, 0, None, None)),
        ]

    def test_opcheck(self):
        """Schema (no input mutated, no aliasing) and fake implementations match
        the kernels' outputs (shapes, strides, dtypes)."""
        for op, args in self._op_samples():
            torch.library.opcheck(op, args, test_utils=("test_schema", "test_faketensor"))

    @pytest.mark.parametrize("sinks", [False, True], ids=["plain", "sinks"])
    def test_flash_attention_fwd_bwd(self, sinks):
        from flash_attn_legacy import flash_attention
        q, k, v = self._qkv((2, 4, 100, 64), (2, 2, 100, 64), grad=True)
        s = torch.randn(4, device="cuda", requires_grad=True) if sinks else None

        def step(q, k, v, s):
            out = flash_attention(q, k, v, causal=True, window_size=(40, 0), softcap=30.0, sinks=s)
            return out.float().square().sum()

        results = []
        for fn in (step, self._compiled(step)):
            params = [t for t in (q, k, v, s) if t is not None]
            loss = fn(q, k, v, s)
            results.append((loss, *torch.autograd.grad(loss, params)))
        for a, b in zip(*results):
            torch.testing.assert_close(a, b, atol=1e-3, rtol=1e-3)

    def test_varlen_fwd_bwd(self):
        from flash_attn_legacy import flash_attn_varlen_func
        q, k, v = self._qkv((120, 4, 64), (120, 2, 64), grad=True)
        cu = torch.tensor([0, 7, 100, 120], dtype=torch.int32, device="cuda")

        def step(q, k, v):
            out = flash_attn_varlen_func(q, k, v, cu, cu, 93, 93, causal=True, validate=False)
            return out.float().square().sum()

        results = []
        for fn in (step, self._compiled(step)):
            loss = fn(q, k, v)
            results.append((loss, *torch.autograd.grad(loss, (q, k, v))))
        for a, b in zip(*results):
            torch.testing.assert_close(a, b, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
    def test_kvcache(self, dtype):
        from flash_attn_legacy import flash_attn_with_kvcache
        q = torch.randn(3, 1, 8, 64, device="cuda", dtype=dtype)
        k, v = (torch.randn(3, 2, 200, 64, device="cuda", dtype=dtype).transpose(1, 2) for _ in range(2))
        seqlens = torch.tensor([200, 17, 0], dtype=torch.int32, device="cuda")
        leftpad = torch.tensor([3, 0, 0], dtype=torch.int32, device="cuda")

        def decode(q, k, v, seqlens, leftpad):
            out, lse = flash_attn_with_kvcache(q, k, v, cache_seqlens=seqlens, cache_leftpad=leftpad, causal=True,
                                               return_softmax_lse=True)
            plain = flash_attn_with_kvcache(q, k, v, cache_seqlens=150)
            return out, lse, plain

        with torch.inference_mode():
            expected = decode(q, k, v, seqlens, leftpad)
            got = self._compiled(decode)(q, k, v, seqlens, leftpad)
        for a, b in zip(expected, got):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_bf16_inference(self):
        """bf16 inputs scaled into fp16 on the device: traceable (no host read)."""
        from flash_attn_legacy import flash_attention
        q, k, v = self._qkv((1, 4, 80, 128), (1, 4, 80, 128), dtype=torch.bfloat16)
        with torch.inference_mode():
            expected = flash_attention(q, k, v, causal=True)
            got = self._compiled(lambda q, k, v: flash_attention(q, k, v, causal=True))(q, k, v)
        torch.testing.assert_close(expected, got, atol=0, rtol=0)
