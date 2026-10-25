#!/usr/bin/env python3
"""Serving via transformers' continuous batching (docs/PLAN.md P6.9): generate_batch throughput.

  --requests concurrent requests, each a prompt of --prompt-len tokens (random tokens of the
  model's vocabulary, lengths spread +-25% so the batches mix prefill and decode) and
  --new-tokens greedy tokens; per attention implementation and CUDA graphs off / on (each pair
  in its own process): a persistent manager reruns the workload until a run is no longer 5% faster
  than the one before (each new batch shape captures a CUDA graph, so long prompts take several
  runs; a fresh manager recaptures every graph, tens of seconds); the fastest run is reported:
  wall time, generated tokens/s, failed requests (steady-state serving). transformers chooses the paging and
  the batch sizes itself; its paged cache takes --max-memory-percent of the free memory, the
  same for every implementation, so memory is not compared here. A failed request carries its
  error (eager / SDPA materialize the scores and can run out of memory); the first is printed.

Usage:
    python benchmarks/serving.py --model Qwen/Qwen2.5-0.5B --requests 64 --prompt-len 512 --new-tokens 128
"""

import argparse
import subprocess
import sys
import time

IMPLS = ("flash_attn_legacy", "sdpa", "eager")


def run_impl(args):
    import torch
    from transformers import AutoModelForCausalLM, GenerationConfig
    import flash_attn_legacy

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype),
                                                 attn_implementation=args.impl).cuda().eval()
    g = torch.Generator().manual_seed(0)
    vocab = model.config.vocab_size
    lens = torch.randint(int(args.prompt_len * 0.75), int(args.prompt_len * 1.25) + 1, (args.requests,), generator=g)
    prompts = [torch.randint(0, vocab, (int(n),), generator=g).tolist() for n in lens]
    gc = GenerationConfig(max_new_tokens=args.new_tokens, do_sample=False, eos_token_id=-1, pad_token_id=0)
    from transformers.generation.configuration_utils import ContinuousBatchingConfig
    flash_attn_legacy.reset_fallback_stats()
    graph = args.graphs == "on"
    cb = ContinuousBatchingConfig(use_cuda_graph=graph, max_memory_percent=args.max_memory_percent)
    row = f"| {args.impl} | {args.graphs} | {args.requests} | {args.prompt_len} | {args.new_tokens} |"
    try:
        with torch.no_grad():  # not inference_mode: generate_batch's thread updates the cache in place
            prev, best = float("inf"), None
            for _ in range(args.max_runs):  # new batch shapes keep capturing graphs for a few runs
                torch.cuda.synchronize()
                t = time.perf_counter()
                out = model.generate_batch(prompts, generation_config=gc, continuous_batching_config=cb,
                                           persistent_manager=True)
                torch.cuda.synchronize()
                run = time.perf_counter() - t
                if best is None or run < best[0]:
                    best = (run, out)
                if run > 0.95 * prev:  # steady state: no longer 5% faster than the run before
                    break
                prev = run
            dt, out = best
    except torch.cuda.OutOfMemoryError:
        print(f"{row} OOM | | |", flush=True)
        return 0
    finally:
        model.destroy_cached_continuous_batching_manager()
    failed = sum(o.error is not None for o in out.values())
    if failed:
        print(f"{args.impl}: first error: {next(o.error for o in out.values() if o.error is not None)}",
              file=sys.stderr, flush=True)
    tokens = sum(len(o.generated_tokens) for o in out.values())
    print(f"{row} {dt:.2f} | {tokens / dt:.0f} | {failed} |", flush=True)
    stats = flash_attn_legacy.fallback_stats()
    if stats:
        print(f"| {args.impl} | fallbacks | {stats} | | | | | |", flush=True)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                allow_abbrev=False)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--dtype", default="float16", choices=("float16", "bfloat16"))
    p.add_argument("--impls", nargs="+", default=list(IMPLS), choices=IMPLS)
    p.add_argument("--requests", type=int, default=64)
    p.add_argument("--prompt-len", type=int, default=512)
    p.add_argument("--new-tokens", type=int, default=128)
    p.add_argument("--max-memory-percent", type=float, default=0.7,
                   help="share of the free memory for the paged cache, the same for every implementation "
                        "(transformers' 0.9 leaves too little for the CUDA graph pools and cuBLAS on 16 GB cards)")
    p.add_argument("--max-runs", type=int, default=10, help="cap on the runs to reach the steady state")
    p.add_argument("--impl", help=argparse.SUPPRESS)  # worker: one implementation in this process
    p.add_argument("--graphs", choices=("off", "on"), help=argparse.SUPPRESS)  # worker: CUDA graphs
    args = p.parse_args()
    if args.max_runs < 1:
        p.error("--max-runs must be at least 1")
    if args.impl:
        return run_impl(args)
    import torch
    print(f"#### {torch.cuda.get_device_name()}: {args.model}, {args.dtype}, generate_batch\n", flush=True)
    print("| implementation | CUDA graphs | requests | prompt tokens | new tokens | s | tok/s | failed |", flush=True)
    print("|---|---|---|---|---|---|---|---|", flush=True)
    failed = False
    for impl in args.impls:
        for graphs in ("off", "on"):
            if subprocess.run([sys.executable, __file__, *sys.argv[1:], "--impl", impl, "--graphs", graphs]).returncode:
                print(f"FAILED: {impl}, CUDA graphs {graphs}", file=sys.stderr, flush=True)
                failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
