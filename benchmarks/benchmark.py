#!/usr/bin/env python3
"""Kernel benchmark: flash_attn_legacy vs SDPA backends (docs/PLAN.md, P1.1).

The main reference is SDPA memory-efficient, selected explicitly: it is what
PyTorch already runs on these GPUs, so it is the bar to beat. SDPA math and
eager attention are reported for context. Each configuration is timed with CUDA
events over --reps repetitions (mean and standard deviation) after a warmup.
Before every repetition the L2 cache is flushed and the GPU is kept busy (so
the host-side launch overhead of every implementation is hidden, as in
triton's do_bench); the reps are queued without synchronizing in between.
Peak memory is measured on a separate run. Implementations run in a rotated
order per configuration, so clock/thermal drift does not always hit the same
one. The forward outputs are checked against SDPA math.

Usage:
    python benchmarks/benchmark.py                        # default grid, print table
    python benchmarks/benchmark.py --save                 # also write docs/bench/<gpu>.json
    python benchmarks/benchmark.py --seq-lens 1024 4096 --head-dim 64 --no-causal
    python benchmarks/benchmark.py --window 1024           # causal sliding window (P4.1)
    python benchmarks/benchmark.py --softcap 50            # logit softcapping (P4.2)
    python benchmarks/benchmark.py --sinks                 # attention sinks (P4.3)

With --window W every query sees its last W keys (HF sliding_window = W): the
kernel gets window_size=(W - 1, -1) (right side 0 when causal), SDPA and eager
a boolean mask built once per shape, during the warmup (it stays cached, so it
is neither timed nor counted in the peak memory).
With --softcap C the logits x become C * tanh(x / C) (Gemma 2). SDPA cannot
do that (reported as unsupported): the reference is eager, which is what the
HF integration falls back to for softcapping layers. Likewise --sinks (one
logit per head in every row's softmax, without value, as gpt-oss; trainable,
so fwd_bwd includes their gradient).
"""

import argparse
import contextlib
import dataclasses
import functools
import json
import math
import pathlib
import platform
import statistics
from datetime import datetime, timezone

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from flash_attn_legacy import flash_attention

ROOT = pathlib.Path(__file__).resolve().parent.parent


@functools.lru_cache(maxsize=4)
def keep_mask(n, causal, window, device):
    """[n, n] boolean mask (True = keep): causal and / or the last `window` keys."""
    rel = torch.arange(n, device=device)[None, :] - torch.arange(n, device=device)[:, None]
    keep = torch.ones(n, n, dtype=torch.bool, device=device)
    if causal:
        keep &= rel <= 0
    if window is not None:
        keep &= rel > -window
    return keep


@dataclasses.dataclass(frozen=True)
class Variant:
    """The attention computed: causal, sliding window W (None: none), softcap C
    (0: none), attention sinks."""
    causal: bool
    window: int = None
    softcap: float = 0.0
    sinks: bool = False


@functools.lru_cache(maxsize=4)
def sink_logits(n_heads, device):
    """Fixed sinks of --sinks, one per head, trainable (gradients accumulate, unread)."""
    return torch.linspace(-1.0, 2.0, n_heads, device=device).requires_grad_(True)


class Unsupported(Exception):
    """The implementation cannot compute this Variant."""


# Shapes here always have n_q == n_k, where top-left (SDPA) and bottom-right
# (flash_attn_legacy) causal alignment coincide.
def eager_attention(q, k, v, var):
    assert q.shape[-2] == k.shape[-2]
    s = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(q.shape[-1]))
    if var.softcap:
        s = var.softcap * torch.tanh(s / var.softcap)
    if var.causal or var.window is not None:
        s = s.masked_fill(~keep_mask(q.shape[-2], var.causal, var.window, q.device), float("-inf"))
    s = s.float()
    if var.sinks:
        col = sink_logits(q.shape[1], q.device).view(1, -1, 1, 1).expand(*s.shape[:-1], 1)
        return torch.softmax(torch.cat([s, col], dim=-1), dim=-1)[..., :-1].to(q.dtype) @ v
    return torch.softmax(s, dim=-1).to(q.dtype) @ v


def sdpa_attention(q, k, v, var):
    if var.softcap or var.sinks:
        raise Unsupported("SDPA has no logit softcapping / attention sinks")
    if var.window is None:
        return F.scaled_dot_product_attention(q, k, v, is_causal=var.causal)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=keep_mask(q.shape[-2], var.causal, var.window, q.device))


def legacy_attention(q, k, v, var):
    window = (-1, -1) if var.window is None else (var.window - 1, -1)
    sinks = sink_logits(q.shape[1], q.device) if var.sinks else None
    return flash_attention(q, k, v, causal=var.causal, window_size=window, softcap=var.softcap, sinks=sinks)


# name -> (function, SDPA backend to select around it or None)
IMPLS = {
    "flash_attn_legacy": (legacy_attention, None),
    "sdpa_mem_efficient": (sdpa_attention, SDPBackend.EFFICIENT_ATTENTION),
    "sdpa_math": (sdpa_attention, SDPBackend.MATH),
    "eager": (eager_attention, None),
}

_FLUSH = None  # buffer larger than L2, zeroed before every repetition


def time_ms(fn, warmup, reps):
    global _FLUSH
    if _FLUSH is None:
        l2 = torch.cuda.get_device_properties(torch.cuda.current_device()).L2_cache_size
        _FLUSH = torch.empty(max(4 * l2, 64 * 2**20), dtype=torch.uint8, device="cuda")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(reps)]
    for start, end in events:
        _FLUSH.zero_()
        torch.cuda._sleep(2_000_000)  # ~1 ms of GPU work: the host enqueues fn meanwhile
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    times = [start.elapsed_time(end) for start, end in events]
    return statistics.mean(times), statistics.stdev(times)


def peak_mb(fn, tensors):
    """Peak memory of one fn() above the inputs (gradients included)."""
    for t in tensors:
        t.grad = None
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20


def bench_config(impl, B, H, N, d, var, mode, warmup, reps):
    """One (impl, shape, mode) measurement; mode is 'fwd' or 'fwd_bwd'."""
    attn, backend = IMPLS[impl]
    grad = mode == "fwd_bwd"
    q = k = v = dout = None

    def step():
        if grad:
            q.grad = k.grad = v.grad = None
            attn(q, k, v, var).backward(dout)
        else:
            attn(q, k, v, var)

    try:
        q, k, v = (torch.randn(B, H, N, d, device="cuda", dtype=torch.float16, requires_grad=grad)
                   for _ in range(3))
        dout = torch.randn_like(q) if grad else None
        with sdpa_kernel(backend) if backend is not None else contextlib.nullcontext():
            mean, std = time_ms(step, warmup, reps)
            mem = peak_mb(step, (q, k, v))
    except torch.cuda.OutOfMemoryError:
        return {"status": "oom"}
    except Unsupported as e:
        return {"status": "unsupported", "error": str(e)}
    except RuntimeError as e:
        # Only "this SDPA backend cannot run this input" is a legitimate gap; any
        # error of our kernel must fail loudly, not be reported as unsupported.
        if backend is None or "No available kernel" not in str(e):
            raise
        return {"status": "unsupported", "error": str(e).splitlines()[0][:200]}
    finally:
        del q, k, v, dout
        torch.cuda.empty_cache()
    return {"status": "ok", "mean_ms": mean, "std_ms": std, "peak_mb": mem}


def max_errors(B, H, N, d, var, impls):
    """Max abs error of each forward vs eager in FP32 (None if it did not fit or
    the implementation cannot compute the variant)."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(B, H, N, d, device="cuda", dtype=torch.float16) for _ in range(3))
    errors = {}
    try:
        with torch.no_grad():
            ref = eager_attention(q.float(), k.float(), v.float(), var)
        for impl in impls:
            attn, backend = IMPLS[impl]
            with torch.no_grad(), sdpa_kernel(backend) if backend is not None else contextlib.nullcontext():
                with contextlib.suppress(Unsupported):
                    errors[impl] = (attn(q, k, v, var).float() - ref).abs().max().item()
    except torch.cuda.OutOfMemoryError:
        pass
    del q, k, v
    torch.cuda.empty_cache()
    return errors


def environment():
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "gpu": props.name,
        "sm": f"{props.major}.{props.minor}",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--seq-lens", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    p.add_argument("--head-dim", type=int, nargs="+", default=[64, 128])
    p.add_argument("--no-causal", action="store_true", help="non-causal only (default: causal only)")
    p.add_argument("--window", type=int, default=None,
                   help="sliding window: each query sees its last W keys (default: none)")
    p.add_argument("--softcap", type=float, default=0.0,
                   help="logit softcapping C * tanh(x / C) (default: 0, none)")
    p.add_argument("--sinks", action="store_true", help="attention sinks, one per head (gpt-oss)")
    p.add_argument("--modes", nargs="+", default=["fwd", "fwd_bwd"], choices=["fwd", "fwd_bwd"])
    p.add_argument("--impls", nargs="+", default=list(IMPLS), choices=list(IMPLS))
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--save", action="store_true", help="write docs/bench/<gpu>.json")
    args = p.parse_args()
    if args.reps < 10:
        p.error("--reps must be at least 10 (mean and std are reported)")
    if args.window is not None and args.window < 1:
        p.error("--window must be at least 1")
    if not (math.isfinite(args.softcap) and args.softcap >= 0):
        p.error("--softcap must be a finite number >= 0")
    if args.warmup < 1:
        p.error("--warmup must be at least 1 (the first call pays one-time setup)")

    env = environment()
    causal = not args.no_causal
    var = Variant(causal, args.window, args.softcap, args.sinks)
    # what the kernel replaces: SDPA mem-efficient, or eager for softcap / sinks (HF fallback)
    ref_impl = "eager" if args.softcap or args.sinks else "sdpa_mem_efficient"
    print(f"{env['gpu']} (sm_{env['sm'].replace('.', '')}), torch {env['torch']}, "
          f"B={args.batch} H={args.heads} causal={causal} window={args.window} softcap={args.softcap} sinks={args.sinks}, "
          f"{args.reps} reps")
    hdr = (f"{'mode':>7} {'d':>4} {'N':>6} | " + " | ".join(f"{i:>22}" for i in args.impls)
           + f" | legacy/{ref_impl}")
    print(hdr)
    print("-" * len(hdr))

    results, rotation = [], 0
    for mode in args.modes:
        for d in args.head_dim:
            for N in args.seq_lens:
                errors = max_errors(args.batch, args.heads, N, d, var, args.impls) if mode == "fwd" else {}
                row = {}
                order = args.impls[rotation:] + args.impls[:rotation]
                rotation = (rotation + 1) % len(args.impls)
                for impl in order:
                    row[impl] = bench_config(impl, args.batch, args.heads, N, d, var, mode,
                                             args.warmup, args.reps)
                for impl in args.impls:
                    results.append({"impl": impl, "mode": mode, "batch": args.batch, "heads": args.heads,
                                    "seq_len": N, "head_dim": d, "causal": causal, "window": args.window,
                                    "softcap": args.softcap, "sinks": args.sinks, "max_abs_error": errors.get(impl), **row[impl]})
                bad = [i for i, e in errors.items() if e > 1e-2]
                if bad:
                    print(f"WARNING d={d} N={N}: max abs error > 1e-2 vs FP32 math for {bad}: "
                          + ", ".join(f"{i}={errors[i]:.2e}" for i in bad))
                cells = []
                for impl in args.impls:
                    r = row[impl]
                    cells.append(f"{r['mean_ms']:8.2f}±{r['std_ms']:5.2f} {r['peak_mb']:5.0f}MB"
                                 if r["status"] == "ok" else f"{r['status']:>22}")
                ref, ours = row.get(ref_impl), row.get("flash_attn_legacy")
                ratio = (f"{ours['mean_ms'] / ref['mean_ms']:6.2f}x"
                         if ref and ours and ref["status"] == ours["status"] == "ok" else "")
                print(f"{mode:>7} {d:>4} {N:>6} | " + " | ".join(cells) + f" | {ratio}")

    print(f"cells: mean±std ms, peak memory above inputs and gradients; legacy/{ref_impl} < 1 means faster")
    print("note: the flash_attn_legacy forward always writes the logsumexp (B*H*N floats), SDPA skips it without grad")
    if args.save:
        suffix = ("" if args.window is None else f"_window{args.window}") + \
                 (f"_softcap{args.softcap:g}" if args.softcap else "") + ("_sinks" if args.sinks else "")
        out = ROOT / "docs" / "bench" / (env["gpu"].replace(" ", "_") + suffix + ".json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"environment": env, "args": vars(args), "results": results}, indent=1) + "\n")
        print(f"saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
