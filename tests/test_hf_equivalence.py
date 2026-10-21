"""
End-to-end equivalence: attn_implementation="flash_attn_legacy" vs "sdpa" on the
same weights (prefill logits, greedy generation, padded batches, chunked prefill).

Also checks *which* path ran: the integration silently falls back to SDPA in
some cases, so a "passing" generation could otherwise never touch our kernel.

Run:
    pytest tests/test_hf_equivalence.py -v -s
    FA_HF_MODELS="TinyLlama/TinyLlama-1.1B-Chat-v1.0,meta-llama/Llama-3.1-8B-Instruct" \
        pytest tests/test_hf_equivalence.py -v -s
"""

import os
import logging

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

transformers = pytest.importorskip("transformers")

from .hf_helpers import KernelCounter, assert_within_noise, compare, logits, set_impl  # noqa: E402

MODELS = [m for m in os.environ.get(
    "FA_HF_MODELS", "TinyLlama/TinyLlama-1.1B-Chat-v1.0").split(",") if m]

PROMPT = ("Flash attention computes exact attention in tiles, keeping the running "
          "softmax statistics in registers. Explain step by step why this saves memory:")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", params=MODELS)
def model_and_tok(request):
    import flash_attn_legacy  # noqa: F401  (registers the backend)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logging.getLogger("transformers").setLevel(logging.ERROR)
    name = request.param
    try:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.float16, device_map="cuda",
            attn_implementation="flash_attn_legacy",
        )
    except OSError as e:
        pytest.skip(f"cannot load {name}: {e}")
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    yield name, model, tok
    del model
    torch.cuda.empty_cache()


def _num_layers(model):
    """Layers whose attention runs on the kernel: all of them, except sliding-window
    layers (e.g. Gemma 3, 5 of every 6), which fall back until docs/PLAN.md P4.1."""
    types = getattr(model.config, "layer_types", None)
    if types:
        return sum(t == "full_attention" for t in types)
    return model.config.num_hidden_layers


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_tokens", [16, 128, 1000])
def test_prefill_logits_match_and_use_kernel(model_and_tok, monkeypatch, n_tokens):
    name, model, tok = model_and_tok
    counter = KernelCounter(monkeypatch)
    ids = tok(PROMPT * 200, return_tensors="pt").input_ids[:, :n_tokens].cuda()

    ref = logits(model, ids, "sdpa")
    eager = logits(model, ids, "eager")
    counter.reset()
    out = logits(model, ids, "flash_attn_legacy")

    n_calls = len(counter.calls)
    diff, cos, top1 = compare(out, ref)
    _, cos_eager, _ = compare(eager, ref)
    print(f"\n  [{name}] N={ids.shape[1]}: kernel calls={n_calls}/{_num_layers(model)} "
          f"max|dlogit|={diff:.4f} cos={cos:.6f} top1={top1:.4f}")

    assert n_calls == _num_layers(model), "prefill did not run the flash kernel"
    # top-1: every position where flash and the reference disagree must be an
    # FP16 near-tie of the reference (top-2 logits within 1% of the top one,
    # e.g. Llama-3.1-8B, 16 tokens: two logits 1 ulp apart); all others agree.
    mismatch = (out.argmax(-1) != ref.argmax(-1)).nonzero().flatten()
    top2 = ref[mismatch].topk(2, dim=-1).values
    margins = (top2[:, 0] - top2[:, 1]) / top2[:, 0].abs().clamp_min(1.0)
    assert bool((margins <= 1e-2).all()), f"non-tie top-1 mismatches: {margins.tolist()}"
    assert_within_noise(cos, cos_eager, 1.0, "prefill vs sdpa")


def test_greedy_generation_matches_sdpa(model_and_tok, monkeypatch):
    name, model, tok = model_and_tok
    counter = KernelCounter(monkeypatch)
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    gen = dict(max_new_tokens=64, do_sample=False, min_new_tokens=64)

    set_impl(model, "sdpa")
    ref = model.generate(ids, **gen)
    counter.reset()
    set_impl(model, "flash_attn_legacy")
    out = model.generate(ids, **gen)

    new_ref, new_out = ref[0, ids.shape[1]:], out[0, ids.shape[1]:]
    same = (new_ref == new_out).int()
    first_div = len(same) if same.all() else int((same == 0).nonzero()[0])
    print(f"\n  [{name}] kernel calls={len(counter.calls)}, identical tokens="
          f"{same.sum().item()}/{len(same)}, first divergence at {first_div}")
    print(f"  sdpa : {tok.decode(new_ref)!r}")
    print(f"  flash: {tok.decode(new_out)!r}")

    assert len(counter.calls) >= _num_layers(model), "generation never ran the flash kernel"
    # FP16 near-ties can flip a token late in a long greedy run; require a long common prefix.
    assert first_div >= 32


def test_left_padded_batch(model_and_tok, monkeypatch):
    """Padded batches run unpadded on the varlen kernel -> must match SDPA and the
    single-prompt run."""
    name, model, tok = model_and_tok
    counter = KernelCounter(monkeypatch)
    prompts = [PROMPT, "The capital of France is"]
    tok.padding_side = "left"
    batch = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    gen = dict(max_new_tokens=24, do_sample=False, min_new_tokens=24)

    set_impl(model, "sdpa")
    ref = model.generate(**batch, **gen)
    counter.reset()
    set_impl(model, "flash_attn_legacy")
    out = model.generate(**batch, **gen)
    print(f"\n  [{name}] padded batch: varlen (unpadded) kernel calls={len(counter.varlen_calls)}")
    assert counter.varlen_calls, "the padded prefill must run unpadded on the kernel (P3.1)"

    L = batch.input_ids.shape[1]
    for i in range(len(prompts)):
        a, b = ref[i, L:], out[i, L:]
        print(f"  [{i}] sdpa : {tok.decode(a)!r}\n  [{i}] flash: {tok.decode(b)!r}")
        assert torch.equal(a[:16], b[:16]), f"sample {i} diverges from sdpa"

    # The short (heavily padded) prompt must give the same tokens as when run alone.
    single = tok(prompts[1:], return_tensors="pt").to("cuda")
    alone = model.generate(**single, **gen)[0, single.input_ids.shape[1]:]
    assert torch.equal(alone[:16], out[1, L:][:16]), "padding changed the output"


def test_chunked_prefill_with_cache(model_and_tok, monkeypatch):
    """Prefill in two chunks with a KV cache (q_len < kv_len) == single-pass prefill."""
    from transformers import DynamicCache

    name, model, tok = model_and_tok
    counter = KernelCounter(monkeypatch)
    ids = tok(PROMPT * 20, return_tensors="pt").input_ids[:, :400].cuda()
    split = 300

    @torch.no_grad()
    def chunked(impl):
        set_impl(model, impl)
        cache = DynamicCache()
        model(input_ids=ids[:, :split], past_key_values=cache, use_cache=True)
        return model(input_ids=ids[:, split:], past_key_values=cache, use_cache=True).logits.float()

    ref_full = logits(model, ids, "sdpa")[split:]
    eager_chunk = chunked("eager")
    counter.reset()
    flash_chunk = chunked("flash_attn_legacy")
    second = [c for c in counter.calls if c[0] < c[1]]
    diff, cos, top1 = compare(flash_chunk[0], ref_full)
    _, cos_eager, _ = compare(eager_chunk[0], ref_full)
    print(f"\n  [{name}] chunked: kernel calls with q_len<kv_len={len(second)} "
          f"max|dlogit|={diff:.4f} cos={cos:.6f} top1={top1:.4f}")

    assert len(second) == _num_layers(model), "second chunk did not use the flash kernel"
    assert_within_noise(cos, cos_eager, top1, "chunked vs single-pass sdpa")
