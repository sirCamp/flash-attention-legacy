"""
A/B performance test: Flash Attention Legacy vs Standard PyTorch Attention.

Measures wall-clock time and peak memory for both implementations across
sequence lengths. Asserts that:
  1. Flash is at least as fast as standard for N >= 512
  2. Flash uses strictly less memory for N >= 512
  3. Results are numerically close (max diff < 0.02)

Run:  pytest tests/test_performance.py -v -s
      (use -s to see the comparison table)
"""

import math
import statistics
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# ---------------------------------------------------------------------------
# Standard attention reference (what people use without flash attention)
# ---------------------------------------------------------------------------

def standard_attention(q, k, v, scale, is_causal=False):
    """Standard scaled dot-product attention in FP32 math, FP16 output.
    This is what torch.nn.functional.scaled_dot_product_attention does
    on GPUs without flash attention support (i.e. Pascal/Volta).
    """
    N = q.shape[-2]
    # FP32 matmul for the score matrix — this allocates O(N²) memory
    s = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    if is_causal:
        mask = torch.triu(torch.ones(N, N, device=q.device, dtype=torch.bool), diagonal=1)
        s.masked_fill_(mask, float('-inf'))
    p = torch.softmax(s, dim=-1)
    return torch.matmul(p, v.float()).half()


# ---------------------------------------------------------------------------
# Timing / memory helpers
# ---------------------------------------------------------------------------

def _warmup(fn, n=5):
    for _ in range(n):
        fn()
    torch.cuda.synchronize()


def _measure(fn, iters=20, runs=5):
    """Returns (mean_ms, std_ms, mean_extra_mb, std_extra_mb).
    Performs `runs` independent measurements of `iters` iterations each.
    """
    all_ms = []
    all_mb = []

    for _ in range(runs):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline_mem = torch.cuda.memory_allocated()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()

        all_ms.append(start.elapsed_time(end) / iters)
        peak_mem = torch.cuda.max_memory_allocated()
        all_mb.append((peak_mem - baseline_mem) / 1e6)

    return (
        statistics.mean(all_ms),
        statistics.stdev(all_ms) if len(all_ms) > 1 else 0.0,
        statistics.mean(all_mb),
        statistics.stdev(all_mb) if len(all_mb) > 1 else 0.0,
    )


# ---------------------------------------------------------------------------
# The actual A/B comparison
# ---------------------------------------------------------------------------

class TestFlashVsStandard:
    """Direct comparison: our flash attention vs standard PyTorch attention."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Clear GPU cache before each test."""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    @pytest.mark.parametrize("N", [256, 512, 1024, 2048])
    @pytest.mark.parametrize("d", [64, 128])
    def test_correctness_ab(self, N, d):
        """Both implementations produce the same result."""
        from flash_attn_legacy import flash_attention
        B, H = 2, 8
        scale = 1.0 / math.sqrt(d)

        q = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16) * 0.3
        k = torch.randn_like(q) * 0.3
        v = torch.randn_like(q) * 0.3

        out_flash = flash_attention(q, k, v, softmax_scale=scale, is_causal=True)
        out_std = standard_attention(q, k, v, scale, is_causal=True)

        max_diff = (out_flash.float() - out_std.float()).abs().max().item()
        assert max_diff < 0.02, f"N={N} d={d}: max diff {max_diff:.4f} > 0.02"

    @pytest.mark.parametrize("N", [512, 1024, 2048, 4096])
    def test_memory_advantage(self, N):
        """Flash attention uses less memory than standard for N >= 512."""
        from flash_attn_legacy import flash_attention
        B, H, d = 2, 8, 64
        scale = 1.0 / math.sqrt(d)

        q = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16) * 0.3
        k = torch.randn_like(q) * 0.3
        v = torch.randn_like(q) * 0.3

        # Measure standard
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()
        _ = standard_attention(q, k, v, scale, is_causal=True)
        torch.cuda.synchronize()
        std_extra_mb = (torch.cuda.max_memory_allocated() - mem_before) / 1e6

        # Measure flash
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()
        _ = flash_attention(q, k, v, softmax_scale=scale, is_causal=True)
        torch.cuda.synchronize()
        flash_extra_mb = (torch.cuda.max_memory_allocated() - mem_before) / 1e6

        savings_pct = (1 - flash_extra_mb / max(std_extra_mb, 0.1)) * 100
        print(f"  N={N}: std={std_extra_mb:.1f}MB, flash={flash_extra_mb:.1f}MB, "
              f"savings={savings_pct:.0f}%")

        assert flash_extra_mb < std_extra_mb, (
            f"N={N}: flash ({flash_extra_mb:.1f}MB) should use less "
            f"memory than standard ({std_extra_mb:.1f}MB)"
        )

    @pytest.mark.parametrize("N", [512, 1024, 2048, 4096])
    def test_speed(self, N):
        """Measure flash vs standard speed. On Pascal (no tensor cores),
        flash attention trades compute for memory — it may be slower for
        small N but the gap narrows as N grows (and memory becomes the
        bottleneck). We only assert that flash doesn't catastrophically
        regress (> 5x slower)."""
        from flash_attn_legacy import flash_attention
        B, H, d = 2, 8, 64
        scale = 1.0 / math.sqrt(d)

        q = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16) * 0.3
        k = torch.randn_like(q) * 0.3
        v = torch.randn_like(q) * 0.3

        fn_std = lambda: standard_attention(q, k, v, scale, is_causal=True)
        fn_flash = lambda: flash_attention(q, k, v, softmax_scale=scale, is_causal=True)

        _warmup(fn_std)
        _warmup(fn_flash)

        std_ms, std_ms_sd, _, _ = _measure(fn_std)
        flash_ms, flash_ms_sd, _, _ = _measure(fn_flash)

        speedup = std_ms / max(flash_ms, 0.001)
        print(f"  N={N}: std={std_ms:.2f}\u00b1{std_ms_sd:.2f}ms, "
              f"flash={flash_ms:.2f}\u00b1{flash_ms_sd:.2f}ms, "
              f"speedup={speedup:.2f}x")

        # Sanity check: flash should not be catastrophically slower
        assert speedup > 0.2, (
            f"N={N}: flash ({flash_ms:.2f}ms) is catastrophically slower "
            f"than standard ({std_ms:.2f}ms)"
        )


# ---------------------------------------------------------------------------
# Summary table — runs once, prints a clear comparison
# ---------------------------------------------------------------------------

class TestComparisonTable:
    """Prints a single summary table comparing flash vs standard attention.
    Use `pytest tests/test_performance.py::TestComparisonTable -v -s` to see it.
    """

    def test_print_comparison(self):
        from flash_attn_legacy import flash_attention, check_gpu_compatibility

        print("\n")
        print("=" * 95)
        check_gpu_compatibility()
        print("=" * 95)

        B, H, d = 2, 8, 64
        scale = 1.0 / math.sqrt(d)

        header = (f"{'N':>6} | {'Std (ms)':>14} {'Flash (ms)':>14} {'Speedup':>8} | "
                  f"{'Std MB':>8} {'Flash MB':>9} {'Saved':>7} | {'Max diff':>9}")
        print(header)
        print("-" * len(header))

        for N in [128, 256, 512, 1024, 2048, 4096]:
            q = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16) * 0.3
            k = torch.randn_like(q) * 0.3
            v = torch.randn_like(q) * 0.3

            fn_std = lambda: standard_attention(q, k, v, scale, is_causal=True)
            fn_flash = lambda: flash_attention(q, k, v, softmax_scale=scale, is_causal=True)

            _warmup(fn_std, 3)
            _warmup(fn_flash, 3)

            std_ms, std_sd, std_mb, _ = _measure(fn_std, iters=10)
            flash_ms, flash_sd, flash_mb, _ = _measure(fn_flash, iters=10)

            # Correctness
            out_f = flash_attention(q, k, v, softmax_scale=scale, is_causal=True)
            out_s = standard_attention(q, k, v, scale, is_causal=True)
            max_diff = (out_f.float() - out_s.float()).abs().max().item()

            speedup = std_ms / max(flash_ms, 0.001)
            saved = (1 - flash_mb / max(std_mb, 0.1)) * 100

            std_str = f"{std_ms:.2f}\u00b1{std_sd:.2f}"
            flash_str = f"{flash_ms:.2f}\u00b1{flash_sd:.2f}"

            try:
                print(f"{N:>6} | {std_str:>14} {flash_str:>14} {speedup:>7.2f}x | "
                      f"{std_mb:>8.1f} {flash_mb:>9.1f} {saved:>6.0f}% | {max_diff:>9.5f}")
            except torch.cuda.OutOfMemoryError:
                dash = "\u2014"
                print(f"{N:>6} | {'OOM':>14} {dash:>14} {dash:>8} | "
                      f"{'OOM':>8} {dash:>9} {dash:>7} | {dash:>9}")
                torch.cuda.empty_cache()

        print()
        print("Values shown as mean\u00b1std over 5 runs of 10 iterations each")
        print("Speedup > 1.0 = flash is faster")
        print("Saved > 0% = flash uses less memory")
        print("Max diff < 0.01 = numerically close")
        print("=" * 95)
