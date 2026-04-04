"""
Cross-encoder benchmark: flash_attn_legacy vs eager/sdpa.

Compares reranking speed, peak memory, and ranking quality (Spearman correlation)
across attention backends using a real cross-encoder model on STS Benchmark.

Usage:
    python scripts/test_crossencoder_benchmark.py
    python scripts/test_crossencoder_benchmark.py --model cross-encoder/ms-marco-MiniLM-L-6-v2
    python scripts/test_crossencoder_benchmark.py --model BAAI/bge-reranker-v2-m3 --fp16
    python scripts/test_crossencoder_benchmark.py --max_pairs 500 --batch_size 64
"""

import argparse
import time
import logging
import torch
import numpy as np

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    p.add_argument("--dataset", default="mteb/stsbenchmark-sts",
                   help="HuggingFace dataset for evaluation")
    p.add_argument("--split", default="test")
    p.add_argument("--max_pairs", type=int, default=1000,
                   help="Max sentence pairs to evaluate")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--warmup_pairs", type=int, default=50)
    p.add_argument("--fp16", action="store_true", help="Load model in FP16")
    p.add_argument("--backends", nargs="+",
                   default=["eager", "sdpa", "flash_attn_legacy"],
                   help="Attention backends to compare")
    return p.parse_args()


def load_dataset_pairs(dataset_name, split, max_pairs):
    """Load sentence pairs and gold scores from a STS-style dataset."""
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)

    pairs = []
    gold_scores = []

    # Try common column name patterns
    s1_col = next((c for c in ["sentence1", "text1", "query"] if c in ds.column_names), None)
    s2_col = next((c for c in ["sentence2", "text2", "passage"] if c in ds.column_names), None)
    score_col = next((c for c in ["score", "label", "similarity_score"] if c in ds.column_names), None)

    if not all([s1_col, s2_col, score_col]):
        raise ValueError(
            f"Cannot find sentence pair columns in dataset. "
            f"Available columns: {ds.column_names}"
        )

    for i, row in enumerate(ds):
        if i >= max_pairs:
            break
        pairs.append((row[s1_col], row[s2_col]))
        gold_scores.append(float(row[score_col]))

    return pairs, gold_scores


def load_crossencoder(model_name, attn_impl, use_fp16):
    from sentence_transformers import CrossEncoder

    model_kwargs = {"attn_implementation": attn_impl}
    if use_fp16:
        model_kwargs["torch_dtype"] = torch.float16

    try:
        import flash_attn_legacy  # noqa: F401
    except ImportError:
        pass

    model = CrossEncoder(
        model_name,
        device="cuda",
        model_kwargs=model_kwargs,
    )
    return model


def spearman_correlation(x, y):
    """Compute Spearman rank correlation without scipy dependency."""
    def rankdata(a):
        arr = np.asarray(a)
        sorter = np.argsort(arr)
        ranks = np.empty_like(sorter, dtype=float)
        ranks[sorter] = np.arange(1, len(arr) + 1, dtype=float)
        return ranks

    rx = rankdata(x)
    ry = rankdata(y)
    d = rx - ry
    n = len(x)
    return 1 - (6 * np.sum(d ** 2)) / (n * (n ** 2 - 1))


def pearson_correlation(x, y):
    """Compute Pearson correlation."""
    x = np.asarray(x)
    y = np.asarray(y)
    mx, my = x.mean(), y.mean()
    num = np.sum((x - mx) * (y - my))
    den = np.sqrt(np.sum((x - mx) ** 2) * np.sum((y - my) ** 2))
    return num / den if den > 0 else 0.0


def benchmark_predict(model, pairs, gold_scores, batch_size, warmup_pairs):
    """Benchmark cross-encoder prediction."""
    # Warmup
    if warmup_pairs > 0:
        warmup = pairs[:warmup_pairs]
        model.predict(warmup, batch_size=batch_size, show_progress_bar=False)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated()

    t0 = time.perf_counter()
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    total_ms = (t1 - t0) * 1000
    pairs_per_sec = len(pairs) / (t1 - t0)
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    baseline_mb = mem_before / (1024 ** 2)
    extra_mb = peak_mb - baseline_mb

    scores = np.asarray(scores).flatten()
    gold = np.asarray(gold_scores)

    spearman = spearman_correlation(scores, gold)
    pearson = pearson_correlation(scores, gold)

    return {
        "total_ms": total_ms,
        "pairs_per_sec": pairs_per_sec,
        "extra_mb": extra_mb,
        "spearman": spearman,
        "pearson": pearson,
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
    }


def main():
    args = parse_args()
    props = torch.cuda.get_device_properties(0)

    print(f"\n{'=' * 75}")
    print(f"  CROSS-ENCODER BENCHMARK")
    print(f"  Model:   {args.model}")
    print(f"  Dataset: {args.dataset} ({args.split})")
    print(f"  GPU:     {props.name} ({props.total_memory / (1024**3):.1f} GB)")
    print(f"  Config:  max_pairs={args.max_pairs}, batch={args.batch_size}, "
          f"dtype={'fp16' if args.fp16 else 'fp32'}")
    print(f"  Backends: {', '.join(args.backends)}")
    print(f"{'=' * 75}")

    # Load dataset
    print(f"\n  Loading dataset...")
    pairs, gold_scores = load_dataset_pairs(args.dataset, args.split, args.max_pairs)
    print(f"  Loaded {len(pairs)} sentence pairs")

    results = []

    for backend in args.backends:
        print(f"\n  [{backend}] Loading model...")
        try:
            model = load_crossencoder(args.model, backend, args.fp16)
        except Exception as e:
            print(f"  [{backend}] Failed to load: {e}")
            results.append({"backend": backend, "error": str(e)})
            continue

        print(f"  [{backend}] Predicting {len(pairs)} pairs...")
        try:
            res = benchmark_predict(
                model, pairs, gold_scores, args.batch_size, args.warmup_pairs
            )
            res["backend"] = backend
            results.append(res)
            print(f"  [{backend}] {res['pairs_per_sec']:.0f} pairs/s, "
                  f"{res['extra_mb']:.0f}MB extra, "
                  f"spearman={res['spearman']:.4f}")
        except torch.cuda.OutOfMemoryError:
            print(f"  [{backend}] OOM!")
            results.append({"backend": backend, "oom": True})
        finally:
            del model
            torch.cuda.empty_cache()

    # Results table
    valid = [r for r in results if "total_ms" in r]
    if not valid:
        print("\n  No successful runs.")
        return

    print(f"\n{'=' * 75}")
    print(f"  RESULTS ({len(pairs)} pairs)")
    print(f"{'=' * 75}")

    header = (f"  {'backend':>20} | {'pairs/s':>10} | {'total (ms)':>12} | "
              f"{'extra (MB)':>10} | {'spearman':>9} | {'pearson':>9}")
    print(f"\n{header}")
    print(f"  {'-' * (len(header) - 2)}")

    for r in results:
        if r.get("oom"):
            print(f"  {r['backend']:>20} | {'OOM':>10} | {'OOM':>12} | "
                  f"{'OOM':>10} | {'':>9} | {'':>9}")
        elif r.get("error"):
            print(f"  {r['backend']:>20} | {'ERR':>10} | {'ERR':>12} | "
                  f"{'ERR':>10} | {'':>9} | {'':>9}")
        else:
            print(f"  {r['backend']:>20} | {r['pairs_per_sec']:>10.1f} | "
                  f"{r['total_ms']:>12.1f} | {r['extra_mb']:>10.0f} | "
                  f"{r['spearman']:>9.4f} | {r['pearson']:>9.4f}")

    # Comparison
    if len(valid) >= 2:
        base = valid[0]
        print(f"\n  vs {base['backend']}:")
        for r in valid[1:]:
            speedup = r["pairs_per_sec"] / base["pairs_per_sec"]
            mem_diff = base["extra_mb"] - r["extra_mb"]
            sign = "+" if mem_diff > 0 else ""
            score_diff = abs(r["spearman"] - base["spearman"])
            print(f"    {r['backend']}: {speedup:.2f}x throughput, "
                  f"{sign}{mem_diff:.0f}MB memory, "
                  f"spearman diff={score_diff:.4f}")

    # Quality check
    if len(valid) >= 2:
        spearman_vals = [r["spearman"] for r in valid]
        max_diff = max(spearman_vals) - min(spearman_vals)
        print(f"\n  Quality: max spearman difference across backends = {max_diff:.4f}")
        if max_diff < 0.01:
            print(f"  All backends produce equivalent ranking quality.")
        else:
            print(f"  WARNING: ranking quality differs across backends (> 0.01)")

    print(f"\n{'=' * 75}")


if __name__ == "__main__":
    main()
