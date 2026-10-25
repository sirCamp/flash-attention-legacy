#!/usr/bin/env python3
"""Short queries against a long cache: speculative / assisted decoding (docs/analysis/kernels/spec_decode_design.md).

  kernel    one attention call of a verification step (q_len new queries against a cache of N keys,
            d=128, B=1), per path, us (median of CUDA events, the host launch hidden behind a GPU
            sleep): `forward` = the plain forward kernel on the cache (HF's path before P6.8),
            `kvcache` = flash_attn_with_kvcache (the decode kernel up to 8 query rows per KV head, the
            forward with key splits and packed GQA rows beyond), `1tok x q_len` = q_len one-token
            decode calls (what plain decoding pays for the same tokens), `sdpa` = HF's SDPA (KV
            repeated, a boolean causal mask)
  generate  HF greedy generate of --new-tokens after a long prompt (a piece of this repository's code
            to rewrite, so prompt lookup finds candidates): plain, assisted (--draft model) and prompt
            lookup, per attention implementation (the draft uses the same one), new tokens over the whole
            generate time (prefill included) and whether the output equals plain greedy; each
            implementation in its own process

Usage:
    python benchmarks/speculative.py kernel
    python benchmarks/speculative.py generate --model Qwen/Qwen2.5-3B --draft Qwen/Qwen2.5-0.5B
"""

import argparse
import pathlib
import statistics
import subprocess
import sys
import time

IMPLS = ("flash_attn_legacy", "sdpa", "eager")


def timed(fn, reps=50):
    import torch
    for _ in range(3):
        fn()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(2_000_000)  # the GPU busy while the host enqueues: launch overhead hidden
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1e3)
    return statistics.median(ts)


def kernel_case(H_q, H_kv, N, q_len, d=128):
    import torch
    import torch.nn.functional as F
    import flash_attn_legacy as fal
    from flash_attn_legacy.flash_attn import DECODE_MAX_ROWS
    q = torch.randn(1, H_q, q_len, d, device="cuda", dtype=torch.float16)
    k = torch.randn(1, H_kv, N, d, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    qd, kc, vc = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # HF's cache as [B, N, H_kv, d] views
    out = {"forward": timed(lambda: fal.flash_attention(q, k, v, causal=True)) if q_len > 1 else None,
           "kvcache": timed(lambda: fal.flash_attn_with_kvcache(qd, kc, vc, causal=True))}
    one = timed(lambda: fal.flash_attn_with_kvcache(qd[:, :1], kc, vc, causal=True)) if H_q // H_kv <= DECODE_MAX_ROWS else None
    out["1tok x q_len"] = one * q_len if one else None
    mask = torch.ones(q_len, N, dtype=torch.bool, device="cuda").tril(N - q_len)
    kr, vr = k.repeat_interleave(H_q // H_kv, 1), v.repeat_interleave(H_q // H_kv, 1)
    out["sdpa"] = timed(lambda: F.scaled_dot_product_attention(q, kr, vr, attn_mask=mask))
    ref = F.scaled_dot_product_attention(q.float(), kr.float(), vr.float(), attn_mask=mask)
    err = (fal.flash_attn_with_kvcache(qd, kc, vc, causal=True).transpose(1, 2).float() - ref).abs().max().item()
    assert err < 2e-2, f"flash_attn_with_kvcache vs SDPA math: {err}"
    return out


def run_kernel(args):
    import torch
    print(f"#### {torch.cuda.get_device_name()}: one layer, d=128, B=1, us\n")
    cols = ("forward", "kvcache", "1tok x q_len", "sdpa")
    print("| H_q / H_kv | N | q_len | rows / KV head | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 4) + "|")
    for H_q, H_kv in args.heads:
        for N in args.cache_lens:
            for q_len in args.q_lens:
                r = kernel_case(H_q, H_kv, N, q_len)
                cells = ["-" if r[c] is None else f"{r[c]:.0f}" for c in cols]
                print(f"| {H_q}/{H_kv} | {N} | {q_len} | {H_q // H_kv * q_len} | " + " | ".join(cells) + " |", flush=True)
                torch.cuda.empty_cache()


def run_generate_impl(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import flash_attn_legacy
    load = lambda name, **kw: AutoModelForCausalLM.from_pretrained(  # noqa: E731
        name, dtype=torch.float16, attn_implementation=args.impl, **kw).cuda().eval()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = load(args.model)
    draft = load(args.draft, cache_dir=args.draft_cache_dir) if args.draft else None
    code = (pathlib.Path(flash_attn_legacy.__file__).parent / "integrations.py").read_text()[:args.prompt_chars]
    prompt = "Here is a Python module:\n\n" + code + "\n\nRewrite the first function above, unchanged, with a docstring:\n"
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    kw = dict(max_new_tokens=args.new_tokens, min_new_tokens=args.new_tokens, do_sample=False,
              attention_mask=torch.ones_like(ids))
    modes = {"plain": {}, "prompt lookup": {"prompt_lookup_num_tokens": 10}}
    if draft is not None:
        modes["assisted"] = {"assistant_model": draft}
    flash_attn_legacy.reset_fallback_stats()
    ref = None
    for name, extra in modes.items():
        with torch.inference_mode():
            model.generate(ids, **kw, **extra)  # warmup
            torch.cuda.synchronize()
            t = time.perf_counter()
            out = model.generate(ids, **kw, **extra)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t
        new = out[0, ids.shape[1]:]
        ref = new if ref is None else ref
        # fp16 near-ties: a verification step (q_len > 1) may round a logit differently than one-token decode
        differ = (new != ref).nonzero()
        same = "yes" if differ.numel() == 0 else f"first {differ[0].item()} tokens"
        print(f"| {args.impl} | {ids.shape[1]} | {name} | {args.new_tokens / dt:.1f} | {same} |", flush=True)
    stats = flash_attn_legacy.fallback_stats()
    if stats:
        print(f"| {args.impl} | fallbacks | {stats} | | |", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                allow_abbrev=False)
    sub = p.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("kernel")
    k.add_argument("--heads", nargs="+", type=lambda s: tuple(map(int, s.split("/"))), default=[(32, 8), (24, 2)],
                   help="H_q/H_kv pairs, e.g. 32/8 24/2")
    k.add_argument("--cache-lens", nargs="+", type=int, default=[1024, 4096, 16384])
    k.add_argument("--q-lens", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    g = sub.add_parser("generate")
    g.add_argument("--model", default="Qwen/Qwen2.5-3B")
    g.add_argument("--draft", default=None, help="assistant model (same tokenizer), e.g. Qwen/Qwen2.5-0.5B")
    g.add_argument("--draft-cache-dir", default=None)
    g.add_argument("--impls", nargs="+", default=list(IMPLS), choices=IMPLS)
    g.add_argument("--new-tokens", type=int, default=128)
    g.add_argument("--prompt-chars", type=int, default=6000)
    g.add_argument("--impl", help=argparse.SUPPRESS)  # worker: one implementation in this process
    args = p.parse_args()
    if args.cmd == "kernel":
        return run_kernel(args)
    if args.impl:
        return run_generate_impl(args)
    import torch
    print(f"#### {torch.cuda.get_device_name()}: {args.model}, fp16, {args.new_tokens} new tokens (tok/s, prefill included)\n")
    print("| implementation | prompt tokens | mode | tok/s | output = plain greedy |")
    print("|---|---|---|---|---|", flush=True)  # before the workers' rows
    failed = False
    for impl in args.impls:
        if subprocess.run([sys.executable, __file__, *sys.argv[1:], "--impl", impl]).returncode:
            print(f"FAILED: {impl}", file=sys.stderr, flush=True)
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
