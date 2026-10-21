"""
Fast HuggingFace integration tests on tiny random-weight models (no download).

Reference ("gold") is the same weights in FP32 with eager attention; FP16 eager
on the same input gives the noise floor (see hf_helpers.assert_close_to_gold).
Each regression test here targets one item of docs/PLAN.md (P0.x).

Run:  pytest tests/test_hf_integration.py -v
"""

import logging

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

transformers = pytest.importorskip("transformers")

from flash_attn_legacy import fallback_stats, integrations, reset_fallback_stats  # noqa: E402
from .hf_helpers import (  # noqa: E402
    TINY_MODELS, KernelCounter, assert_close_to_gold, load_impl, logits, tiny_llama,
)


@pytest.fixture(autouse=True)
def _clean_fallback_stats():
    reset_fallback_stats()
    yield
    reset_fallback_stats()


def _qkv(q_len=16, kv_len=16, d=64, d_v=None, heads=4, dtype=torch.float16):
    mk = lambda n, dd: torch.randn(1, heads, n, dd, device="cuda", dtype=dtype)  # noqa: E731
    return mk(q_len, d), mk(kv_len, d), mk(kv_len, d_v or d)


class _Attn(nn.Module):
    """Stand-in for an HF attention module (only `is_causal` and `training` are read)."""

    def __init__(self, is_causal):
        super().__init__()
        self.is_causal = is_causal


# ---------------------------------------------------------------------------
# P0.1 — kernel support is decided per call, never on the model config
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("head_dim", [64, 80])
def test_backend_kept_for_any_head_dim(head_dim):
    """B6: the old config-based check switched whole models to sdpa (and used
    hidden_size / num_heads, wrong for sub-models). The backend must stay ours."""
    model = load_impl(tiny_llama(head_dim=head_dim), "flash_attn_legacy")
    assert model.config._attn_implementation == "flash_attn_legacy"


@pytest.mark.parametrize("head_dim,expect_kernel", [(64, True), (128, True), (80, False), (48, False)])
def test_per_call_decision_and_registry(monkeypatch, head_dim, expect_kernel):
    base = tiny_llama(head_dim=head_dim)
    ids = torch.randint(0, base.config.vocab_size, (1, 200))
    gold = logits(load_impl(base, "eager", torch.float32), ids)
    noise = logits(load_impl(base, "eager"), ids)

    counter = KernelCounter(monkeypatch)
    out = logits(load_impl(base, "flash_attn_legacy"), ids)
    n_layers = base.config.num_hidden_layers

    if expect_kernel:
        assert len(counter.calls) == n_layers and fallback_stats() == {}
    else:
        assert len(counter.calls) == 0 and fallback_stats() == {"head_dim": n_layers}
    assert_close_to_gold(out, gold, noise, f"d={head_dim}")


def test_decode_reason():
    base = tiny_llama()
    model = load_impl(base, "flash_attn_legacy")
    ids = torch.randint(0, base.config.vocab_size, (1, 32), device="cuda")
    model.generate(ids, max_new_tokens=4, min_new_tokens=4, do_sample=False)
    # 4 new tokens: the 1st comes from the prefill (kernel), 3 decode steps fall back.
    assert fallback_stats() == {"decode": 3 * base.config.num_hidden_layers}


def test_attention_mask_reason():
    base = tiny_llama()
    model = load_impl(base, "flash_attn_legacy")
    ids = torch.randint(0, base.config.vocab_size, (2, 24), device="cuda")
    mask = torch.ones_like(ids)
    mask[1, :10] = 0  # left padding on the second row
    with torch.no_grad():
        model(input_ids=ids, attention_mask=mask)
    assert fallback_stats() == {"attention_mask": base.config.num_hidden_layers}


def test_dropout_reason():
    base = tiny_llama(attention_dropout=0.1)
    model = load_impl(base, "flash_attn_legacy").train()
    ids = torch.randint(0, base.config.vocab_size, (1, 24), device="cuda")
    model(input_ids=ids)
    assert fallback_stats() == {"dropout": base.config.num_hidden_layers}


def test_q_longer_than_kv_reason():
    q, k, v = _qkv(q_len=24, kv_len=16)
    out, _ = integrations.flash_attn_legacy_forward(_Attn(is_causal=False), q, k, v, None)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v).transpose(1, 2)
    assert fallback_stats() == {"q_longer_than_kv": 1}
    torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-3)


def test_v_head_dim_reason():
    q, k, v = _qkv(d=64, d_v=32)
    out, _ = integrations.flash_attn_legacy_forward(_Attn(is_causal=True), q, k, v, None)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2)
    assert fallback_stats() == {"v_head_dim": 1} and out.shape == (1, 16, 4, 32)
    torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("kwarg,value,reason", [
    ("cu_seq_lens_q", torch.tensor([0, 8, 16], dtype=torch.int32), "cu_seq_lens"),
    ("cu_seq_lens_k", torch.tensor([0, 8, 16], dtype=torch.int32), "cu_seq_lens"),
    ("softcap", 50.0, "softcap"),
    ("s_aux", torch.zeros(4), "s_aux"),
    ("position_bias", torch.zeros(1, 4, 16, 16), "position_bias"),
])
@pytest.mark.parametrize("q_len", [16, 1], ids=["prefill", "decode"])
def test_unsupported_kwargs_raise(kwarg, value, reason, q_len):
    """No correct path exists yet (SDPA would silently ignore them): must raise,
    on the kernel path and on the fallback path alike."""
    q, k, v = _qkv(q_len=q_len)
    with pytest.raises(NotImplementedError):
        integrations.flash_attn_legacy_forward(_Attn(is_causal=True), q, k, v, None,
                                               **{kwarg: value})
    assert fallback_stats() == {reason: 1}


@pytest.mark.parametrize("value", [None, 0.0])
def test_softcap_disabled_values_do_not_raise(value):
    q, k, v = _qkv()
    integrations.flash_attn_legacy_forward(_Attn(is_causal=True), q, k, v, None, softcap=value)
    assert fallback_stats() == {}


def test_warn_once_per_reason_and_not_for_expected(caplog):
    base = tiny_llama(head_dim=80)
    model = load_impl(base, "flash_attn_legacy")
    ids = torch.randint(0, base.config.vocab_size, (1, 16), device="cuda")
    with caplog.at_level(logging.WARNING, logger=integrations.__name__):
        model.generate(ids, max_new_tokens=3, min_new_tokens=3, do_sample=False)
        model.generate(ids, max_new_tokens=3, min_new_tokens=3, do_sample=False)
    warnings = [r for r in caplog.records
                if r.name == integrations.__name__ and r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "head_dim=80" in warnings[0].getMessage()
    # Reasons are checked in order: decode steps count as "decode", prefills as "head_dim".
    n_layers = base.config.num_hidden_layers
    assert fallback_stats() == {"head_dim": 2 * n_layers, "decode": 2 * 2 * n_layers}


# ---------------------------------------------------------------------------
# P0.2 — caches whose kv_len is not "all tokens so far" (B1: StaticCache)
# ---------------------------------------------------------------------------

# prefill, a multi-token chunk with an offset (as in assisted decoding),
# two decode steps, a longer chunk
_STEPS = [20, 5, 1, 1, 13]


@torch.no_grad()
def _stepwise_logits(model, ids, cache):
    out, pos = [], 0
    for n in _STEPS:
        res = model(input_ids=ids[:, pos:pos + n], past_key_values=cache, use_cache=True,
                    cache_position=torch.arange(pos, pos + n, device="cuda"))
        out.append(res.logits.float()[0])
        pos += n
    return torch.cat(out)


@pytest.mark.parametrize("offloading", [False, True], ids=["static", "offloaded_static"])
@pytest.mark.parametrize("arch", ["llama", "gemma3"])  # gemma3: sliding + full layers (hybrid)
def test_static_cache_stepwise(monkeypatch, arch, offloading):
    from transformers import StaticCache

    base = TINY_MODELS[arch]()
    ids = torch.randint(0, base.config.vocab_size, (1, sum(_STEPS)), device="cuda")
    gold = logits(load_impl(base, "eager", torch.float32), ids)

    def run(impl):
        model = load_impl(base, impl)
        cache = StaticCache(config=model.config, max_cache_len=64, offloading=offloading)
        return _stepwise_logits(model, ids, cache)

    noise = run("eager")
    counter = KernelCounter(monkeypatch)
    out = run("flash_attn_legacy")
    assert_close_to_gold(out, gold, noise, f"{arch} static")

    # The kernel must run on the multi-token steps of every full-attention layer,
    # on K/V sliced to the filled part of the cache (kv == tokens so far).
    full_layers = sum(t == "full_attention" for t in getattr(
        base.config, "layer_types", None) or ["full_attention"] * base.config.num_hidden_layers)
    ends = [sum(_STEPS[:i + 1]) for i in range(len(_STEPS))]
    expected = [(n, end) for n, end in zip(_STEPS, ends) if n > 1 for _ in range(full_layers)]
    assert sorted(counter.calls) == sorted(expected)


def _beam_rows_error(x, y):
    """Worst 1-cos between each row of x and its closest row of y: beams with
    near-equal scores can swap order between backends. The matching must be
    one-to-one (a duplicated beam is an error, reported as 1-cos = 1)."""
    x = torch.nn.functional.normalize(x.float(), dim=-1)
    y = torch.nn.functional.normalize(y.float(), dim=-1)
    best = (x @ y.T).max(dim=1)
    if best.indices.unique().numel() != x.shape[0]:
        return 1.0
    return 1.0 - best.values.min().item()


@pytest.mark.parametrize("arch", ["llama", "gemma3"])
@pytest.mark.parametrize("cache_implementation", ["static", "offloaded_static", "dynamic"])
@pytest.mark.parametrize("mode", ["greedy", "beam", "assisted"])
def test_generate_matches_gold(arch, cache_implementation, mode):
    """End-to-end generate vs FP32 eager (gold), FP16 eager as noise floor:
    every step's logits (all beams) and the final sequences."""
    if mode == "assisted" and cache_implementation != "dynamic":
        pytest.skip("assisted decoding needs a dynamic cache")
    # Peaked but not saturated logits (Gemma scales embeddings by sqrt(hidden)).
    base = TINY_MODELS[arch](initializer_range={"llama": 0.5, "gemma3": 0.05}[arch])
    ids = torch.randint(0, base.config.vocab_size, (1, 24), device="cuda")
    # disable_compile: HF auto-compiles static-cache generation and Triton does not
    # support sm < 7.0; torch.compile is covered separately (docs/PLAN.md, P6.5).
    kw = dict(max_new_tokens=16, min_new_tokens=16, do_sample=False,
              cache_implementation=cache_implementation, disable_compile=True,
              return_dict_in_generate=True, output_logits=True)
    if mode == "beam":
        kw["num_beams"] = 3

    def gen(impl, dtype=torch.float16):
        extra = {}
        if mode == "assisted":
            extra["assistant_model"] = load_impl(base, "sdpa", dtype)
            extra["cache_implementation"] = None  # assisted decoding manages its own cache
        return load_impl(base, impl, dtype).generate(ids, **{**kw, **extra})

    gold, noise, out = gen("eager", torch.float32), gen("eager"), gen("flash_attn_legacy")
    label = f"{arch}/{cache_implementation}/{mode}"
    for step, (o, n, g) in enumerate(zip(out.logits, noise.logits, gold.logits)):
        assert torch.isfinite(g).all(), f"{label}: non-finite reference logits"
        err, err_noise = _beam_rows_error(o, g), _beam_rows_error(n, g)
        # same rule as hf_helpers.assert_within_noise; 1e-3 floor for tiny models
        assert err <= max(3.0 * err_noise, 1e-3), \
            f"{label}: step {step} 1-cos={err:.2e} (noise {err_noise:.2e})"
    if torch.equal(noise.sequences, gold.sequences):
        assert torch.equal(out.sequences, gold.sequences), label


# -- mask builder unit tests --------------------------------------------------

def _build(**kw):
    from transformers import masking_utils as mu

    args = dict(batch_size=2, q_length=5, kv_length=64, q_offset=20, kv_offset=0,
                mask_function=mu.causal_mask_function, attention_mask=None,
                allow_is_causal_skip=True, device="cuda")
    args.update(kw)
    return integrations._flash_attn_legacy_mask(**args)


def test_builder_static_cache_tags_valid_keys():
    m = _build()
    assert m.shape == (2, 25) and getattr(m, integrations._N_VALID_ATTR) == 25
    # padding mask zeros beyond the filled part of the cache are irrelevant
    pad = torch.ones(2, 64, dtype=torch.bool, device="cuda")
    pad[:, 40:] = False
    assert getattr(_build(attention_mask=pad), integrations._N_VALID_ATTR) == 25


def test_builder_dynamic_cache_and_padding():
    assert _build(kv_length=25) is None
    pad = torch.ones(2, 25, dtype=torch.bool, device="cuda")  # HF passes it as bool
    pad[1, :3] = False
    m = _build(kv_length=25, attention_mask=pad)
    assert m.dim() == 4 and m.dtype == torch.bool  # SDPA's mask -> fallback


def test_builder_never_returns_sdpa_none_for_causal_skip_disabled():
    """With skipping disabled the SDPA builder must not return None (B1)."""
    m = _build(allow_is_causal_skip=False)
    assert m is not None and m.dim() == 4


def test_builder_bidirectional_never_slices():
    """Encoders / cross-attention: all keys count (q_length != kv_length is normal)."""
    from transformers import masking_utils as mu

    kw = dict(mask_function=mu.bidirectional_mask_function, q_length=5, kv_length=12,
              q_offset=0, allow_is_causal_skip=False)
    assert _build(allow_is_bidirectional_skip=True, **kw) is None
    m = _build(allow_is_bidirectional_skip=False, **kw)
    assert m is not None and m.dim() == 4


def test_foreign_2d_padding_mask_is_honoured():
    """Some models pass HF's raw [B, kv] padding mask straight to 'flash' backends:
    it must be applied (SDPA fallback), never treated as 'first n keys valid'."""
    q, k, v = _qkv(q_len=16, kv_len=16)
    q, k, v = (t.expand(2, -1, -1, -1).contiguous() for t in (q, k, v))
    pad = torch.ones(2, 16, dtype=torch.long, device="cuda")
    pad[1, :6] = 0  # left padding on row 1
    out, _ = integrations.flash_attn_legacy_forward(_Attn(is_causal=True), q, k, v, pad)
    causal = torch.ones(16, 16, dtype=torch.bool, device="cuda").tril()
    ref_mask = pad[:, None, None, :].bool() & causal
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=ref_mask)
    assert fallback_stats() == {"attention_mask": 1}
    assert torch.isfinite(out).all()  # also fully padded query rows: no NaN
    # rows of the padded sequence that see at least one key must match the reference
    torch.testing.assert_close(out.transpose(1, 2)[:, :, 6:], ref[:, :, 6:], atol=2e-3, rtol=2e-3)


def test_untagged_copy_of_builder_mask_still_runs_kernel(monkeypatch):
    """The tag is only a fast path: a copy (tag lost) must give the same result."""
    q, k, v = _qkv(q_len=5, kv_len=64)
    m = _build(batch_size=1, q_length=5, q_offset=20)
    counter = KernelCounter(monkeypatch)
    out_tagged, _ = integrations.flash_attn_legacy_forward(_Attn(is_causal=True), q, k, v, m)
    out_copy, _ = integrations.flash_attn_legacy_forward(_Attn(is_causal=True), q, k, v, m.clone())
    assert counter.calls == [(5, 25), (5, 25)] and fallback_stats() == {}
    torch.testing.assert_close(out_tagged, out_copy, atol=0, rtol=0)


def test_tagged_mask_longer_than_keys_raises():
    q, k, v = _qkv(q_len=4, kv_len=8)
    m = _build(q_length=4, q_offset=6)  # tags 10 valid keys, only 8 passed
    with pytest.raises(RuntimeError):
        integrations.flash_attn_legacy_forward(_Attn(is_causal=True), q, k, v, m)
