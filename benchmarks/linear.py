#!/usr/bin/env python3
"""FP16 linear layers on the P100 (docs/PLAN.md, P8): fused_linear (HFMA2, FP32 fold every 128 K-elements) vs
F.linear (cuBLAS, FP32 accumulation) on the layer shapes y = x W^T of real models.

Each shape is timed with CUDA events in ROUNDS rounds, every round timing both implementations back to back
(batches of ~15 ms), the median per implementation; the GPU is cooled below --cool C before each model and after
a shape that ends hotter than --hot C (the P100 throttles its clocks from about 80 C). The error columns are the
relative Frobenius error of each output against an FP32 product of the same FP16 inputs (first 2048 rows).

--accuracy HF_ID...: on wikitext-2 validation (16 x 1024 tokens), each model's top-1 agreement with itself in FP32
(FP32 weights, SDPA) and perplexity, for FP16 with stock F.linear (SDPA attention, then flash_attn_legacy's: the noise
floor stock FP16 already has) and FP16 with patch_linear (needs `datasets`).
--decode HF_ID...: decode steps alone, stock vs patch_linear in one process: a B x 2048 prefill fills a DynamicCache
(not timed), then 64 one-token forwards are timed; 5 rounds alternating the two (decode has fewer than 128 rows per
call: F.linear in both, what differs is the patched module's forward).

Usage:
    python benchmarks/linear.py                                  # every model, M = 512 2048 8192
    python benchmarks/linear.py --models qwen2.5-1.5b --ms 128 512
    python benchmarks/linear.py --accuracy Qwen/Qwen2.5-0.5B TinyLlama/TinyLlama-1.1B-Chat-v1.0
    python benchmarks/linear.py --decode TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

import argparse
import statistics
import subprocess
import time

SHAPES = {  # name: (hidden, intermediate, q_dim, kv_dim, vocab)
    "qwen2.5-0.5b": (896, 4864, 896, 128, 151936),
    "qwen2.5-1.5b": (1536, 8960, 1536, 256, 151936),
    "qwen2.5-3b": (2048, 11008, 2048, 256, 151936),
    "llama3.1-8b": (4096, 14336, 4096, 1024, 128256),
}
ROUNDS = 5


def layers(h, i, q, kv, vocab):
    """(N, K) of each projection: qkv fused, o, gate+up fused, down, lm_head."""
    return {"qkv": (q + 2 * kv, h), "o": (h, q), "gateup": (2 * i, h), "down": (h, i), "lm_head": (vocab, h)}


def temperature():
    import torch
    # by UUID: the same GPU whatever CUDA_DEVICE_ORDER / CUDA_VISIBLE_DEVICES (lines of unreadable GPUs skipped)
    uuid = f"GPU-{torch.cuda.get_device_properties(torch.cuda.current_device()).uuid}"
    out = subprocess.run(["nvidia-smi", "--query-gpu=uuid,temperature.gpu", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    return next(int(line.split(",")[1]) for line in out.splitlines() if line.startswith(uuid))


def cool(limit, timeout=600):
    end = time.monotonic() + timeout
    while temperature() >= limit and time.monotonic() < end:
        time.sleep(5)


def bench(M, N, K):
    """Median ms of (stock, fused), their relative errors against FP32 and the reason fused_linear runs F.linear
    (None: the kernel ran)."""
    import torch
    import torch.nn.functional as F
    from flash_attn_legacy import fused_linear
    from flash_attn_legacy.linear import _fallback_reason
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    W = torch.randn(N, K, device="cuda", dtype=torch.float16) * K ** -0.5
    reason = _fallback_reason(x, W, None)
    fns = {"stock": lambda: F.linear(x, W), "fused": lambda: fused_linear(x, W)}
    rows = x[:2048]  # the error on at most 2048 rows: an FP32 lm_head output of 8192 rows does not fit next to it
    ref = F.linear(rows.float(), W.float())
    err = {k: ((f(rows, W).float() - ref).norm() / ref.norm()).item() for k, f in
           {"stock": F.linear, "fused": fused_linear}.items()}
    del ref
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    fns["fused"]()
    e1.record()
    torch.cuda.synchronize()
    reps = int(min(200, max(2, 15.0 / max(e0.elapsed_time(e1), 1e-3))))
    times = {k: [] for k in fns}
    for _ in range(ROUNDS):
        for k, f in fns.items():
            e0.record()
            for _ in range(reps):
                f()
            e1.record()
            torch.cuda.synchronize()
            times[k].append(e0.elapsed_time(e1) / reps)
    return {k: statistics.median(v) for k, v in times.items()}, err, reason


def accuracy(name, seqs=16, T=1024):
    """Top-1 agreement with FP32 and perplexity on wikitext-2 validation: stock FP16 (SDPA, flash_attn_legacy) and
    FP16 with patch_linear."""
    import math
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import flash_attn_legacy
    tok = AutoTokenizer.from_pretrained(name)
    text = "\n\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")["text"]
                       if t.strip())
    ids = torch.tensor(tok(text)["input_ids"])
    if ids.numel() < seqs * T:
        raise RuntimeError(f"{name}: the tokenizer gave {ids.numel()} tokens, {seqs * T} needed")
    ids = ids[: seqs * T].view(seqs, T)

    @torch.inference_mode()
    def run(dtype, impl, fused):
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation=impl).cuda().eval()
        if fused:
            flash_attn_legacy.patch_linear(model)
        tops, nll = [], 0.0
        for s in range(seqs):
            x = ids[s:s + 1].cuda()
            logits = model(input_ids=x).logits[0, :-1].float()
            tops.append(logits.argmax(-1).cpu())
            nll += F.cross_entropy(logits, x[0, 1:], reduction="sum").item()
        del model
        torch.cuda.empty_cache()
        return torch.cat(tops), math.exp(nll / (seqs * (T - 1)))

    ref, ppl32 = run(torch.float32, "sdpa", False)
    cells = [f"{ppl32:.4f}"]
    tops = {}
    for key, impl, fused in (("sdpa", "sdpa", False), ("stock", "flash_attn_legacy", False),
                             ("fused", "flash_attn_legacy", True)):
        tops[key], ppl = run(torch.float16, impl, fused)
        cells += [f"{(tops[key] == ref).float().mean().item():.3%}", f"{ppl:.4f}"]
    cells.append(f"{(tops['fused'] == tops['stock']).float().mean().item():.3%}")
    print(f"| {name.split('/')[-1]} | " + " | ".join(cells) + " |", flush=True)


def decode(name, steps=64, rounds=5, P=2048):
    """Decode step time, stock vs patch_linear, alternated in one process."""
    import torch
    from transformers import AutoModelForCausalLM, DynamicCache
    import flash_attn_legacy
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16,
                                                 attn_implementation="flash_attn_legacy").cuda().eval()

    @torch.inference_mode()
    def step_ms(B, fused):
        flash_attn_legacy.patch_linear(model, enable=fused)
        ids = torch.randint(100, 30000, (B, P), generator=torch.Generator().manual_seed(0)).cuda()
        cache = DynamicCache(config=model.config)
        tok = model(input_ids=ids, past_key_values=cache, logits_to_keep=1).logits[:, -1:].argmax(-1)
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        e0.record()
        for _ in range(steps):
            tok = model(input_ids=tok, past_key_values=cache).logits[:, -1:].argmax(-1)
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) / steps

    for B in (1, 8):
        times = {False: [], True: []}
        step_ms(B, False), step_ms(B, True)  # warmup
        for _ in range(rounds):
            for fused in (False, True):
                times[fused].append(step_ms(B, fused))
        stock, fused = statistics.median(times[False]), statistics.median(times[True])
        print(f"| {name.split('/')[-1]} | {B} | {stock:.2f} | {fused:.2f} | {stock / fused:.3f}x |", flush=True)
    flash_attn_legacy.patch_linear(model, enable=False)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=list(SHAPES), choices=list(SHAPES))
    p.add_argument("--ms", nargs="+", type=int, default=[512, 2048, 8192], help="tokens per call (B x T)")
    p.add_argument("--cool", type=int, default=60)
    p.add_argument("--hot", type=int, default=72)
    p.add_argument("--accuracy", nargs="+", metavar="HF_ID", help="top-1 vs FP32 and perplexity on wikitext-2")
    p.add_argument("--decode", nargs="+", metavar="HF_ID", help="decode step time, stock vs patch_linear")
    args = p.parse_args()
    import os
    import torch
    import flash_attn_legacy
    if os.environ.get("FLASH_ATTN_LEGACY_LINEAR"):
        p.error("unset FLASH_ATTN_LEGACY_LINEAR: it would patch the stock runs too")
    if args.accuracy:
        print("| model | fp32 ppl | top-1 vs fp32, fp16 sdpa | ppl | fp16 stock | ppl | fp16 fused | ppl | "
              "fused vs stock |")
        print("|---" * 9 + "|")
        for name in args.accuracy:
            accuracy(name)
        return
    if args.decode:
        print("| model | B | stock ms / step | fused ms / step | speed |")
        print("|---|---|---|---|---|")
        for name in args.decode:
            cool(args.cool)
            decode(name)
        return
    if torch.cuda.get_device_capability() != (6, 0):
        print(f"{torch.cuda.get_device_name()}: fused_linear runs F.linear here (compute capability 6.0 only)")
    print(f"{torch.cuda.get_device_name()}, torch {torch.__version__}, "
          f"flash_attn_legacy {flash_attn_legacy.__version__}")
    print("\n| model | layer | M | N | K | stock TFLOPS | fused TFLOPS | speedup | error stock | error fused |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    speedups = []
    for name in args.models:
        cool(args.cool)
        for layer, (N, K) in layers(*SHAPES[name]).items():
            for M in args.ms:
                t, err, reason = bench(M, N, K)
                tflops = {k: 2 * M * N * K / ms / 1e9 for k, ms in t.items()}
                speedup = t["stock"] / t["fused"]
                if reason is None:  # in the geomean only the shapes the kernel ran
                    speedups.append(speedup)
                print(f"| {name} | {layer} | {M} | {N} | {K} | {tflops['stock']:.2f} | {tflops['fused']:.2f} | "
                      f"{speedup:.2f}x" + (f" (F.linear: {reason})" if reason else "") +
                      f" | {err['stock']:.1e} | {err['fused']:.1e} |", flush=True)
                if temperature() >= args.hot:
                    cool(args.cool)
    if speedups:
        print(f"\ngeomean speedup {statistics.geometric_mean(speedups):.2f}x over the {len(speedups)} shapes on the "
              f"kernel (min {min(speedups):.2f}x, max {max(speedups):.2f}x)")


if __name__ == "__main__":
    main()
