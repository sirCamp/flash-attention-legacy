"""
Prefill & generation benchmarks with real HuggingFace models.

Compares eager (O(N^2)) vs sdpa vs flash_attn_legacy (O(N)) on:
  - TinyLlama 1.1B  (~2.2GB FP16)
  - Qwen2.5 3B      (~6GB FP16)

Run:  pytest tests/test_generation_benchmark.py -v -s
      pytest tests/test_generation_benchmark.py -v -s -k "prefill"
      pytest tests/test_generation_benchmark.py -v -s -k "generation"
Requires: pip install -e ".[transformers,test]"
"""

import logging
import statistics
import pytest
import torch
import time

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

BACKENDS = ["eager", "sdpa", "flash_attn_legacy"]

MODELS = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "qwen3b": "Qwen/Qwen2.5-3B",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(model_name, attn_implementation="eager"):
    """Load model in FP16 on CUDA, suppressing noisy logs."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Suppress loading noise
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("accelerate").setLevel(logging.ERROR)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='cuda',
        attn_implementation=attn_implementation,
    )
    model.eval()

    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("accelerate").setLevel(logging.WARNING)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def _make_input_ids(tokenizer, seq_len):
    """Create input_ids of exactly seq_len tokens.

    Repeats a base sentence enough times, then truncates to exact length.
    """
    base = "The quick brown fox jumps over the lazy dog and returns home. "
    # Each sentence is ~13 tokens, overshoot then truncate
    text = base * (seq_len // 10 + 10)
    ids = tokenizer(text, return_tensors='pt', truncation=True,
                    max_length=seq_len).to('cuda')
    return ids['input_ids']


def _prefill_memory(model, input_ids, warmup=2, repeats=5):
    """Measure peak memory during forward pass (prefill only).

    Returns dict with mean/std for time (ms) and extra memory (MB).
    Returns None on OOM.
    """
    try:
        for _ in range(warmup):
            with torch.no_grad():
                model(input_ids)
            torch.cuda.synchronize()

        times_ms = []
        extra_mbs = []

        for _ in range(repeats):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            baseline = torch.cuda.memory_allocated()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            with torch.no_grad():
                model(input_ids)
            end.record()
            torch.cuda.synchronize()

            times_ms.append(start.elapsed_time(end))
            peak = torch.cuda.max_memory_allocated()
            extra_mbs.append((peak - baseline) / (1024 ** 2))

        return {
            'time_mean': statistics.mean(times_ms),
            'time_std': statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
            'mem_mean': statistics.mean(extra_mbs),
            'mem_std': statistics.stdev(extra_mbs) if len(extra_mbs) > 1 else 0.0,
        }
    except torch.cuda.OutOfMemoryError:
        return None


def _run_prefill_benchmark(model_name, seq_lengths, backends=BACKENDS, repeats=5):
    """Run prefill benchmark for all backends and sequence lengths.

    Returns list of dicts: [{N, backend, stats_or_None}, ...]
    All loading/measurement happens here; printing happens later.
    """
    import flash_attn_legacy  # ensures register() was called

    rows = []

    for N in seq_lengths:
        for backend in backends:
            model = None
            try:
                model, tokenizer = _load_model(model_name,
                                               attn_implementation=backend)
                input_ids = _make_input_ids(tokenizer, N)
                actual_len = input_ids.shape[1]

                torch.cuda.empty_cache()
                stats = _prefill_memory(model, input_ids, repeats=repeats)
                rows.append({'N': actual_len, 'backend': backend, 'stats': stats})
            except Exception:
                rows.append({'N': N, 'backend': backend, 'stats': None})
            finally:
                del model
                torch.cuda.empty_cache()

    return rows


def _print_prefill_table(model_name, rows, backends=BACKENDS):
    """Print a clean comparison table from collected results."""
    # Group by N
    by_n = {}
    for r in rows:
        by_n.setdefault(r['N'], {})[r['backend']] = r['stats']

    props = torch.cuda.get_device_properties(0)

    W = 95
    print(f"\n{'=' * W}")
    print(f"  PREFILL MEMORY BENCHMARK")
    print(f"  Model: {model_name}")
    print(f"  GPU:   {props.name} ({props.total_memory / (1024**3):.1f} GB)")
    print(f"  Values: mean\u00b1std over multiple runs")
    print(f"{'=' * W}")

    # --- Memory table ---
    print(f"\n  EXTRA MEMORY (MB above model weights)")
    cols = "  ".join(f"{b:>16}" for b in backends)
    header = f"  {'N':>6} | {cols} |"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for N in sorted(by_n.keys()):
        parts = []
        for b in backends:
            s = by_n[N].get(b)
            if s:
                parts.append(f"{s['mem_mean']:.1f}\u00b1{s['mem_std']:.1f}")
            else:
                parts.append("OOM")
        cols = "  ".join(f"{p:>16}" for p in parts)
        print(f"  {N:>6} | {cols} |")

    # --- Memory savings vs eager ---
    print(f"\n  MEMORY SAVINGS vs eager")
    savings_backends = [b for b in backends if b != "eager"]
    cols = "  ".join(f"{b:>16}" for b in savings_backends)
    header = f"  {'N':>6} | {cols} |"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for N in sorted(by_n.keys()):
        eager = by_n[N].get("eager")
        parts = []
        for b in savings_backends:
            s = by_n[N].get(b)
            if eager and s:
                pct = (1 - s['mem_mean'] / max(eager['mem_mean'], 0.1)) * 100
                parts.append(f"{pct:+.1f}%")
            elif eager is None and s:
                parts.append("eager OOM!")
            elif s is None:
                parts.append("OOM")
            else:
                parts.append("n/a")
        cols = "  ".join(f"{p:>16}" for p in parts)
        print(f"  {N:>6} | {cols} |")

    # --- Time table ---
    print(f"\n  PREFILL TIME (ms)")
    cols = "  ".join(f"{b:>16}" for b in backends)
    header = f"  {'N':>6} | {cols} |"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for N in sorted(by_n.keys()):
        parts = []
        for b in backends:
            s = by_n[N].get(b)
            if s:
                parts.append(f"{s['time_mean']:.1f}\u00b1{s['time_std']:.1f}")
            else:
                parts.append("OOM")
        cols = "  ".join(f"{p:>16}" for p in parts)
        print(f"  {N:>6} | {cols} |")

    print(f"\n  eager            = standard O(N\u00b2) attention (full score matrix)")
    print(f"  sdpa             = torch scaled_dot_product_attention")
    print(f"  flash_attn_legacy = our O(N) CUDA kernel for Pascal/Volta")
    print(f"{'=' * W}")


# ---------------------------------------------------------------------------
# Prefill benchmarks
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
class TestPrefillTinyLlama:
    """Prefill memory benchmark on TinyLlama 1.1B."""

    MODEL = MODELS["tinyllama"]

    @pytest.fixture(autouse=True)
    def _setup(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def test_prefill_memory(self):
        """3-way prefill comparison: eager vs sdpa vs flash_attn_legacy."""
        rows = _run_prefill_benchmark(
            self.MODEL,
            seq_lengths=[256, 512, 1024, 2048],
        )
        _print_prefill_table(self.MODEL, rows)

        # Assert: flash uses less memory than eager for longest sequence
        by_n = {}
        for r in rows:
            by_n.setdefault(r['N'], {})[r['backend']] = r['stats']

        for N in sorted(by_n.keys()):
            e = by_n[N].get("eager")
            f = by_n[N].get("flash_attn_legacy")
            if e and f and N >= 512:
                assert f['mem_mean'] < e['mem_mean'], (
                    f"N={N}: flash ({f['mem_mean']:.1f}MB) should use less "
                    f"memory than eager ({e['mem_mean']:.1f}MB)"
                )


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
class TestPrefillQwen3B:
    """Prefill memory benchmark on Qwen2.5 3B.
    Larger model = more heads/layers = bigger attention matrices = more savings.
    """

    MODEL = MODELS["qwen3b"]

    @pytest.fixture(autouse=True)
    def _setup(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def _check_gpu(self):
        props = torch.cuda.get_device_properties(0)
        gpu_gb = props.total_memory / (1024**3)
        if gpu_gb < 10:
            pytest.skip(f"Need >= 10GB GPU for 3B model, have {gpu_gb:.1f}GB")

    def test_prefill_memory(self):
        """3-way prefill comparison on Qwen2.5 3B."""
        self._check_gpu()

        rows = _run_prefill_benchmark(
            self.MODEL,
            seq_lengths=[256, 512, 1024],
        )
        _print_prefill_table(self.MODEL, rows)


# ---------------------------------------------------------------------------
# End-to-end generation benchmarks
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
class TestGeneration:
    """End-to-end generation (prefill + decode).

    Flash attention only helps during prefill. Decode (q_len=1) is identical
    for all backends. This test shows correctness and practical speed.
    """

    MODEL = MODELS["tinyllama"]

    @pytest.fixture(autouse=True)
    def _setup(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def test_generation_correctness(self):
        """Verify flash_attn_legacy generates coherent text."""
        import flash_attn_legacy

        model, tokenizer = _load_model(self.MODEL,
                                       attn_implementation="flash_attn_legacy")
        try:
            stats = _generate_timed(model, tokenizer, "The meaning of life is",
                                    max_new_tokens=32, warmup=1, repeats=1)
        finally:
            del model
            torch.cuda.empty_cache()

        assert stats['tokens'] > 0, "No tokens generated"
        print(f"\n  flash_attn_legacy output: {stats['text'][:200]}")

    def test_generation_speed(self):
        """Compare generation speed: eager vs sdpa vs flash_attn_legacy."""
        import flash_attn_legacy

        prompt = "Explain how transformers work in deep learning."
        results = []

        for backend in BACKENDS:
            model = None
            try:
                model, tokenizer = _load_model(self.MODEL,
                                               attn_implementation=backend)
                stats = _generate_timed(model, tokenizer, prompt,
                                        max_new_tokens=64, warmup=2, repeats=3)
                results.append((backend, stats))
            except torch.cuda.OutOfMemoryError:
                results.append((backend, None))
            finally:
                del model
                torch.cuda.empty_cache()

        # Print table at the end
        W = 80
        print(f"\n{'=' * W}")
        print(f"  GENERATION SPEED — {self.MODEL}")
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {props.name} ({props.total_memory / (1024**3):.1f} GB)")
        print(f"{'=' * W}")

        header = f"  {'backend':>20} | {'time (s)':>14} {'tok/s':>8} | {'tokens':>6}"
        print(header)
        print(f"  {'-' * (len(header) - 2)}")

        for backend, stats in results:
            if stats:
                t = f"{stats['time_mean']:.2f}\u00b1{stats['time_std']:.2f}"
                tps = stats['tokens'] / stats['time_mean']
                print(f"  {backend:>20} | {t:>14} {tps:>7.1f} | {stats['tokens']:>6}")
            else:
                print(f"  {backend:>20} | {'OOM':>14} {'':>8} | {'':>6}")

        print(f"{'=' * W}")
