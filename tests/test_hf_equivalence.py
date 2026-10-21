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

MODELS = [m for m in os.environ.get(
    "FA_HF_MODELS", "TinyLlama/TinyLlama-1.1B-Chat-v1.0").split(",") if m]

PROMPT = ("Flash attention computes exact attention in tiles, keeping the running "
          "softmax statistics in registers. Explain step by step why this saves memory:")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class KernelCounter:
    """Counts calls to the flash kernel made through the HF integration."""

    def __init__(self, monkeypatch):
        import flash_attn_legacy.flash_attn as fa
        self.calls = []
        orig = fa.flash_attention

        def wrapped(q, k, v, *args, **kwargs):
            self.calls.append((q.shape[2], k.shape[2]))  # (q_len, kv_len)
            return orig(q, k, v, *args, **kwargs)

        monkeypatch.setattr(fa, "flash_attention", wrapped)

    def reset(self):
        self.calls.clear()


def _set_impl(model, impl):
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation(impl)
    else:  # older transformers
        model.config._attn_implementation = impl
    assert model.config._attn_implementation == impl, (
        f"requested {impl}, model uses {model.config._attn_implementation}")


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
    return model.config.num_hidden_layers


@torch.no_grad()
def _logits(model, impl, input_ids, **kw):
    _set_impl(model, impl)
    return model(input_ids=input_ids, **kw).logits.float()


def _compare(a, b):
    """Return (max_abs_diff, mean_cosine, top1_agreement) over positions."""
    diff = (a - b).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()
    top1 = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    return diff, cos, top1


def _assert_within_noise(flash_cos, eager_cos, top1, label):
    """FP16 logits of large models differ between *any* two attention backends
    (eager vs sdpa included). Judge flash against that noise floor, measured on the
    same input, instead of a fixed threshold: its error (1 - cos) may be at most 3x
    eager's error w.r.t. the same reference."""
    err_flash, err_eager = 1.0 - flash_cos, 1.0 - eager_cos
    print(f"  {label}: 1-cos flash={err_flash:.2e} eager(noise floor)={err_eager:.2e} top1={top1:.4f}")
    assert err_flash <= 3.0 * err_eager + 1e-5, f"{label}: flash error above FP16 noise floor"
    assert top1 >= 0.99


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_tokens", [16, 128, 1000])
def test_prefill_logits_match_and_use_kernel(model_and_tok, monkeypatch, n_tokens):
    name, model, tok = model_and_tok
    counter = KernelCounter(monkeypatch)
    ids = tok(PROMPT * 200, return_tensors="pt").input_ids[:, :n_tokens].cuda()

    ref = _logits(model, "sdpa", ids)
    eager = _logits(model, "eager", ids)
    counter.reset()
    out = _logits(model, "flash_attn_legacy", ids)

    n_calls = len(counter.calls)
    diff, cos, top1 = _compare(out[0], ref[0])
    _, cos_eager, _ = _compare(eager[0], ref[0])
    print(f"\n  [{name}] N={ids.shape[1]}: kernel calls={n_calls}/{_num_layers(model)} "
          f"max|dlogit|={diff:.4f} cos={cos:.6f} top1={top1:.4f}")

    assert n_calls == _num_layers(model), "prefill did not run the flash kernel"
    _assert_within_noise(cos, cos_eager, top1, "prefill vs sdpa")


def test_greedy_generation_matches_sdpa(model_and_tok, monkeypatch):
    name, model, tok = model_and_tok
    counter = KernelCounter(monkeypatch)
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    gen = dict(max_new_tokens=64, do_sample=False, min_new_tokens=64)

    _set_impl(model, "sdpa")
    ref = model.generate(ids, **gen)
    counter.reset()
    _set_impl(model, "flash_attn_legacy")
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
    """Padded batches carry an attention_mask -> must match SDPA and the unpadded run."""
    name, model, tok = model_and_tok
    counter = KernelCounter(monkeypatch)
    prompts = [PROMPT, "The capital of France is"]
    tok.padding_side = "left"
    batch = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    gen = dict(max_new_tokens=24, do_sample=False, min_new_tokens=24)

    _set_impl(model, "sdpa")
    ref = model.generate(**batch, **gen)
    counter.reset()
    _set_impl(model, "flash_attn_legacy")
    out = model.generate(**batch, **gen)
    print(f"\n  [{name}] padded batch: kernel calls={len(counter.calls)} "
          f"(shapes: {sorted(set(counter.calls))[:4]})")

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
        _set_impl(model, impl)
        cache = DynamicCache()
        model(input_ids=ids[:, :split], past_key_values=cache, use_cache=True)
        return model(input_ids=ids[:, split:], past_key_values=cache, use_cache=True).logits.float()

    ref_full = _logits(model, "sdpa", ids)[:, split:]
    eager_chunk = chunked("eager")
    counter.reset()
    flash_chunk = chunked("flash_attn_legacy")
    second = [c for c in counter.calls if c[0] < c[1]]
    diff, cos, top1 = _compare(flash_chunk[0], ref_full[0])
    _, cos_eager, _ = _compare(eager_chunk[0], ref_full[0])
    print(f"\n  [{name}] chunked: kernel calls with q_len<kv_len={len(second)} "
          f"max|dlogit|={diff:.4f} cos={cos:.6f} top1={top1:.4f}")

    assert len(second) == _num_layers(model), "second chunk did not use the flash kernel"
    _assert_within_noise(cos, cos_eager, top1, "chunked vs single-pass sdpa")
