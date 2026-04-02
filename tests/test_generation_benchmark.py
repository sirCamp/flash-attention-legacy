"""
Prefill & generation benchmarks with real HuggingFace models.

Compares eager (O(N^2)) vs flash_attn_legacy (O(N)) on:
  - TinyLlama 1.1B       (~2.2GB FP16, head_dim=64)
  - Llama 3.1 8B Instruct (~16GB FP16, head_dim=128, GQA)

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

BACKENDS = ["eager", "flash_attn_legacy"]

MODELS = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "llama3_8b": "meta-llama/Llama-3.1-8B-Instruct",
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


def _generate_timed(model, tokenizer, prompt, max_new_tokens=64,
                    warmup=2, repeats=5):
    """Generate multiple times and return timing/memory stats."""
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')

    for _ in range(warmup):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=4, max_length=None,
                           do_sample=False, pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize()

    times = []
    peak_mems = []
    out = None

    for _ in range(repeats):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=max_new_tokens,
                max_length=None,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        peak_mems.append(torch.cuda.max_memory_allocated() / (1024**3))

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    num_tokens = out.shape[1] - inputs['input_ids'].shape[1]

    return {
        'time_mean': statistics.mean(times),
        'time_std': statistics.stdev(times) if len(times) > 1 else 0.0,
        'mem_mean': statistics.mean(peak_mems),
        'mem_std': statistics.stdev(peak_mems) if len(peak_mems) > 1 else 0.0,
        'tokens': num_tokens,
        'text': text,
    }


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

    # --- Combined table: memory + time + savings ---
    print(f"\n  {'N':>6} | {'eager (MB)':>14} {'flash (MB)':>14} {'saved':>8} | {'eager (ms)':>14} {'flash (ms)':>14} {'speedup':>8}")
    print(f"  {'-' * 85}")

    for N in sorted(by_n.keys()):
        e = by_n[N].get("eager")
        f = by_n[N].get("flash_attn_legacy")

        e_mem = f"{e['mem_mean']:.1f}\u00b1{e['mem_std']:.1f}" if e else "OOM"
        f_mem = f"{f['mem_mean']:.1f}\u00b1{f['mem_std']:.1f}" if f else "OOM"
        e_ms = f"{e['time_mean']:.1f}\u00b1{e['time_std']:.1f}" if e else "OOM"
        f_ms = f"{f['time_mean']:.1f}\u00b1{f['time_std']:.1f}" if f else "OOM"

        if e and f:
            saved = (1 - f['mem_mean'] / max(e['mem_mean'], 0.1)) * 100
            saved_str = f"{saved:+.0f}%"
            spd = e['time_mean'] / max(f['time_mean'], 0.001)
            spd_str = f"{spd:.2f}x"
        elif e is None and f:
            saved_str = "eager OOM"
            spd_str = "n/a"
        else:
            saved_str = "n/a"
            spd_str = "n/a"

        print(f"  {N:>6} | {e_mem:>14} {f_mem:>14} {saved_str:>8} | {e_ms:>14} {f_ms:>14} {spd_str:>8}")

    print(f"\n  eager = standard O(N\u00b2) attention (full score matrix)")
    print(f"  flash = flash_attn_legacy (O(N) tiled, no score matrix)")
    print(f"  saved = memory reduction vs eager | speedup > 1.0 = flash is faster")
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
            seq_lengths=[256, 512, 1024, 2048, 4096, 8192],
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
class TestPrefillLlama3:
    """Prefill memory benchmark on Llama 3.1 8B Instruct.
    head_dim=128, GQA (32 Q heads, 8 KV heads), ~16GB FP16.
    Needs >= 20GB GPU memory.
    """

    MODEL = MODELS["llama3_8b"]

    @pytest.fixture(autouse=True)
    def _setup(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def _check_gpu(self):
        props = torch.cuda.get_device_properties(0)
        gpu_gb = props.total_memory / (1024**3)
        if gpu_gb < 20:
            pytest.skip(f"Need >= 20GB GPU for Llama 3.1 8B, have {gpu_gb:.1f}GB")

    def test_prefill_memory(self):
        """Prefill comparison on Llama 3.1 8B."""
        self._check_gpu()

        rows = _run_prefill_benchmark(
            self.MODEL,
            seq_lengths=[256, 512, 1024, 2048, 4096],
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
        """Compare generation speed: eager vs flash_attn_legacy."""
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


# ---------------------------------------------------------------------------
# Variable-length (packed sequences) benchmarks
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestVarlenBenchmark:
    """Compare varlen (packed) vs padded attention at kernel level.

    Shows memory and speed impact of packing variable-length sequences
    instead of padding to max length.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def _measure(self, fn, warmup=3, repeats=10):
        """Measure time (ms) and peak extra memory (MB) for a function."""
        import math as _math
        try:
            for _ in range(warmup):
                fn()
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
                fn()
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
            torch.cuda.empty_cache()
            return None

    def test_varlen_vs_padded(self):
        """Compare packed (varlen) vs padded attention.

        Simulates a batch with variable-length sequences. Padded = pad all
        to max_len and use standard flash_attention. Packed = concatenate
        and use flash_attn_varlen_func.
        """
        from flash_attn_legacy import flash_attention, flash_attn_varlen_func
        import math

        H, d = 8, 64
        scale = 1.0 / math.sqrt(d)

        scenarios = [
            ("uniform_short",       [64] * 16),
            ("uniform_medium",      [256] * 8),
            ("mixed_short",         [32, 64, 48, 16, 96, 80, 24, 128]),
            ("mixed_heavy",         [512, 64, 256, 32, 128, 64, 512, 32]),
            ("one_long_rest_short", [1024] + [32] * 15),
            ("all_long",            [1024] * 4),
        ]

        props = torch.cuda.get_device_properties(0)
        W = 105
        print(f"\n{'=' * W}")
        print(f"  VARLEN (PACKED) vs PADDED BENCHMARK")
        print(f"  GPU:  {props.name} ({props.total_memory / (1024**3):.1f} GB)")
        print(f"  Config: H={H}, d={d}, FP16, causal=True")
        print(f"{'=' * W}")
        print(f"\n  {'scenario':>25} | {'seqs':>4} {'max_len':>7} {'total':>6} {'pad_waste':>9} |"
              f" {'padded (MB)':>12} {'packed (MB)':>12} {'mem_saved':>9} |"
              f" {'padded (ms)':>12} {'packed (ms)':>12} {'speedup':>8}")
        print(f"  {'-' * 100}")

        for name, seq_lens in scenarios:
            batch = len(seq_lens)
            max_len = max(seq_lens)
            total_tokens = sum(seq_lens)
            padded_tokens = batch * max_len
            waste_pct = (1 - total_tokens / padded_tokens) * 100

            # --- Padded: pad all to max_len, single kernel call ---
            q_pad = torch.randn(batch, H, max_len, d, device='cuda', dtype=torch.float16) * 0.3
            k_pad = torch.randn(batch, H, max_len, d, device='cuda', dtype=torch.float16) * 0.3
            v_pad = torch.randn(batch, H, max_len, d, device='cuda', dtype=torch.float16) * 0.3

            def run_padded(q=q_pad, k=k_pad, v=v_pad):
                with torch.no_grad():
                    flash_attention(q, k, v, softmax_scale=scale, is_causal=True)

            # --- Packed: concatenate, use varlen ---
            q_pack = torch.randn(total_tokens, H, d, device='cuda', dtype=torch.float16) * 0.3
            k_pack = torch.randn(total_tokens, H, d, device='cuda', dtype=torch.float16) * 0.3
            v_pack = torch.randn(total_tokens, H, d, device='cuda', dtype=torch.float16) * 0.3

            offsets = [0]
            for sl in seq_lens:
                offsets.append(offsets[-1] + sl)
            cu = torch.tensor(offsets, dtype=torch.int32, device='cuda')

            def run_packed(q=q_pack, k=k_pack, v=v_pack, c=cu, ml=max_len):
                with torch.no_grad():
                    flash_attn_varlen_func(q, k, v, c, c, ml, ml,
                                          softmax_scale=scale, causal=True)

            padded_stats = self._measure(run_padded)
            packed_stats = self._measure(run_packed)

            del q_pad, k_pad, v_pad, q_pack, k_pack, v_pack
            torch.cuda.empty_cache()

            p_mem = f"{padded_stats['mem_mean']:.1f}" if padded_stats else "OOM"
            k_mem = f"{packed_stats['mem_mean']:.1f}" if packed_stats else "OOM"
            p_ms = f"{padded_stats['time_mean']:.2f}" if padded_stats else "OOM"
            k_ms = f"{packed_stats['time_mean']:.2f}" if packed_stats else "OOM"

            if padded_stats and packed_stats:
                saved = (1 - packed_stats['mem_mean'] / max(padded_stats['mem_mean'], 0.1)) * 100
                saved_str = f"{saved:+.0f}%"
                spd = padded_stats['time_mean'] / max(packed_stats['time_mean'], 0.001)
                spd_str = f"{spd:.2f}x"
            elif padded_stats is None and packed_stats:
                saved_str = "pad OOM"
                spd_str = "n/a"
            else:
                saved_str = "n/a"
                spd_str = "n/a"

            print(f"  {name:>25} | {batch:>4} {max_len:>7} {total_tokens:>6} {waste_pct:>8.0f}% |"
                  f" {p_mem:>12} {k_mem:>12} {saved_str:>9} |"
                  f" {p_ms:>12} {k_ms:>12} {spd_str:>8}")

        print(f"\n  padded = all sequences padded to max_len, single flash_attention call")
        print(f"  packed = concatenated sequences, flash_attn_varlen_func (per-seq kernel calls)")
        print(f"  pad_waste = % of padded tokens that are padding")
        print(f"  mem_saved = memory reduction of packed vs padded")
        eq_line = "=" * W
        print(eq_line)
