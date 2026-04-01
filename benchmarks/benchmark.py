#!/usr/bin/env python3
"""
Benchmark: Flash Attention Legacy vs Standard PyTorch Attention

Usage:
    python benchmarks/benchmark.py
    python benchmarks/benchmark.py --seq-lens 512 1024 2048 4096
"""

import argparse
import math
import sys
import torch

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))


def std_attention(q, k, v, scale, is_causal=False):
    N = q.shape[-2]
    s = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    if is_causal:
        mask = torch.triu(torch.ones(N, N, device=q.device, dtype=torch.bool), diagonal=1)
        s.masked_fill_(mask, float('-inf'))
    return torch.matmul(torch.softmax(s, dim=-1).half(), v)


def bench_one(B, H, N, d, is_causal, iters=100):
    from flash_attn_legacy import flash_attention
    scale = 1.0 / math.sqrt(d)
    dev = torch.device('cuda')

    q = torch.randn(B, H, N, d, device=dev, dtype=torch.float16)
    k, v = torch.randn_like(q), torch.randn_like(q)

    # Warmup
    for _ in range(20):
        flash_attention(q, k, v, softmax_scale=scale, is_causal=is_causal)
        std_attention(q, k, v, scale, is_causal)
    torch.cuda.synchronize()

    def time_fn(fn, iters):
        torch.cuda.reset_peak_memory_stats()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters, torch.cuda.max_memory_allocated() / 1e6

    flash_ms, flash_mem = time_fn(
        lambda: flash_attention(q, k, v, softmax_scale=scale, is_causal=is_causal), iters)
    std_ms, std_mem = time_fn(
        lambda: std_attention(q, k, v, scale, is_causal), iters)

    # Quick correctness check
    out_f = flash_attention(q, k, v, softmax_scale=scale, is_causal=is_causal)
    out_s = std_attention(q, k, v, scale, is_causal)
    max_diff = (out_f.float() - out_s.float()).abs().max().item()

    return {
        'N': N, 'd': d, 'causal': is_causal,
        'flash_ms': flash_ms, 'std_ms': std_ms,
        'speedup': std_ms / flash_ms if flash_ms > 0 else 0,
        'flash_mb': flash_mem, 'std_mb': std_mem,
        'mem_save': (1 - flash_mem / std_mem) * 100 if std_mem > 0 else 0,
        'max_diff': max_diff,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--seq-lens', type=int, nargs='+', default=[128, 256, 512, 1024, 2048, 4096])
    parser.add_argument('--head-dim', type=int, nargs='+', default=[64, 128])
    parser.add_argument('--causal', action='store_true')
    parser.add_argument('--iters', type=int, default=100)
    args = parser.parse_args()

    from flash_attn_legacy import check_gpu_compatibility
    print("=" * 90)
    check_gpu_compatibility()
    print("=" * 90)

    hdr = f"{'N':>6} {'d':>4} {'caus':>5} | {'Flash ms':>9} {'Std ms':>9} {'Speed':>7} | {'Fl MB':>7} {'St MB':>7} {'Save':>6} | {'Diff':>8}"
    print(hdr)
    print("-" * len(hdr))

    for d in args.head_dim:
        for N in args.seq_lens:
            try:
                r = bench_one(args.batch, args.heads, N, d, args.causal, args.iters)
                print(f"{r['N']:>6} {r['d']:>4} {str(r['causal']):>5} | "
                      f"{r['flash_ms']:>9.3f} {r['std_ms']:>9.3f} {r['speedup']:>6.2f}x | "
                      f"{r['flash_mb']:>7.1f} {r['std_mb']:>7.1f} {r['mem_save']:>5.1f}% | "
                      f"{r['max_diff']:>8.5f}")
            except torch.cuda.OutOfMemoryError:
                print(f"{N:>6} {d:>4} {'OOM':>5}")
                torch.cuda.empty_cache()

    print("\nHigher speedup + lower memory + small diff (<0.01) = good")


if __name__ == '__main__':
    main()
