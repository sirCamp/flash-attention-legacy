"""
Embedding benchmark: sentence-transformers with flash_attn_legacy vs eager/sdpa.

Compares encoding speed and peak memory across attention backends
using a real embedding model (e.g. Qwen3-Embedding-4B).

Usage:
    python scripts/test_embedding_benchmark.py
    python scripts/test_embedding_benchmark.py --model Qwen/Qwen3-Embedding-0.6B
    python scripts/test_embedding_benchmark.py --batch_size 32 --seq_len 512 --fp16
"""

import argparse
import time
import logging
import torch

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-Embedding-4B")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--fp16", action="store_true", help="Load model in FP16")
    p.add_argument("--backends", nargs="+",
                   default=["eager", "sdpa", "flash_attn_legacy"],
                   help="Attention backends to compare")
    return p.parse_args()


def make_sentences(batch_size, seq_len):
    """Generate sentences of roughly target token length."""
    # ~1.3 tokens per word on average, so we aim for seq_len/1.3 words
    words_per_sent = max(10, seq_len // 2)
    base_words = ("the quick brown fox jumps over the lazy dog and "
                  "explores the vast landscape of natural language processing "
                  "while considering the implications of modern deep learning "
                  "architectures for embedding generation tasks in production "
                  "systems that require low latency and high throughput ").split()
    sentences = []
    for i in range(batch_size):
        sent = []
        for j in range(words_per_sent):
            sent.append(base_words[(i * 7 + j) % len(base_words)])
        sentences.append(" ".join(sent))
    return sentences


def load_model(model_name, attn_impl, use_fp16):
    from sentence_transformers import SentenceTransformer

    model_kwargs = {"attn_implementation": attn_impl}
    if use_fp16:
        model_kwargs["torch_dtype"] = torch.float16

    try:
        import flash_attn_legacy  # noqa: F401 — ensure registered
    except ImportError:
        pass

    model = SentenceTransformer(
        model_name,
        device="cuda",
        model_kwargs=model_kwargs,
    )
    return model


def benchmark_encode(model, sentences, warmup, runs):
    """Benchmark encoding, return avg time (ms) and peak memory (MB)."""
    # Warmup
    for _ in range(warmup):
        model.encode(sentences, show_progress_bar=False)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated()

    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        embeddings = model.encode(sentences, show_progress_bar=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    baseline_mb = mem_before / (1024 ** 2)
    extra_mb = peak_mb - baseline_mb
    avg_ms = sum(times) / len(times)
    std_ms = (sum((t - avg_ms) ** 2 for t in times) / len(times)) ** 0.5
    embed_dim = embeddings.shape[-1] if embeddings is not None else 0

    return {
        "avg_ms": avg_ms,
        "std_ms": std_ms,
        "peak_mb": peak_mb,
        "extra_mb": extra_mb,
        "embed_dim": embed_dim,
    }


def main():
    args = parse_args()
    props = torch.cuda.get_device_properties(0)
    sentences = make_sentences(args.batch_size, args.seq_len)

    print(f"\n{'=' * 75}")
    print(f"  EMBEDDING BENCHMARK")
    print(f"  Model:  {args.model}")
    print(f"  GPU:    {props.name} ({props.total_memory / (1024**3):.1f} GB)")
    print(f"  Config: batch={args.batch_size}, ~seq_len={args.seq_len}, "
          f"warmup={args.warmup}, runs={args.runs}, "
          f"dtype={'fp16' if args.fp16 else 'fp32'}")
    print(f"  Backends: {', '.join(args.backends)}")
    print(f"{'=' * 75}")

    results = []

    for backend in args.backends:
        print(f"\n  [{backend}] Loading model...")
        try:
            model = load_model(args.model, backend, args.fp16)
        except Exception as e:
            print(f"  [{backend}] Failed to load: {e}")
            results.append({"backend": backend, "error": str(e)})
            continue

        print(f"  [{backend}] Encoding {args.batch_size} sentences x {args.runs} runs...")
        try:
            res = benchmark_encode(model, sentences, args.warmup, args.runs)
            res["backend"] = backend
            results.append(res)
            print(f"  [{backend}] {res['avg_ms']:.1f}ms avg, "
                  f"{res['extra_mb']:.0f}MB extra, dim={res['embed_dim']}")
        except torch.cuda.OutOfMemoryError:
            print(f"  [{backend}] OOM!")
            results.append({"backend": backend, "oom": True})
        finally:
            del model
            torch.cuda.empty_cache()

    # Results table
    valid = [r for r in results if "avg_ms" in r]
    if not valid:
        print("\n  No successful runs.")
        return

    print(f"\n{'=' * 75}")
    print(f"  RESULTS")
    print(f"{'=' * 75}")

    header = f"  {'backend':>20} | {'avg (ms)':>12} | {'extra mem (MB)':>14} | {'embed dim':>9}"
    print(f"\n{header}")
    print(f"  {'-' * (len(header) - 2)}")

    baseline_ms = None
    for r in results:
        if r.get("oom"):
            print(f"  {r['backend']:>20} | {'OOM':>12} | {'OOM':>14} | {'':>9}")
        elif r.get("error"):
            err = r['error'][:30]
            print(f"  {r['backend']:>20} | {'ERR':>12} | {err:>14} | {'':>9}")
        else:
            if baseline_ms is None:
                baseline_ms = r["avg_ms"]
            speedup = baseline_ms / r["avg_ms"] if r["avg_ms"] > 0 else 0
            time_str = f"{r['avg_ms']:.1f}±{r['std_ms']:.1f}"
            print(f"  {r['backend']:>20} | {time_str:>12} | {r['extra_mb']:>14.0f} | {r['embed_dim']:>9}")

    # Speedup comparison
    if len(valid) >= 2:
        print(f"\n  Speedup vs {valid[0]['backend']}:")
        for r in valid[1:]:
            speedup = valid[0]["avg_ms"] / r["avg_ms"]
            mem_diff = valid[0]["extra_mb"] - r["extra_mb"]
            sign = "+" if mem_diff > 0 else ""
            print(f"    {r['backend']}: {speedup:.2f}x speed, "
                  f"{sign}{mem_diff:.0f}MB memory")

    print(f"\n{'=' * 75}")


if __name__ == "__main__":
    main()
