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
    TINY_MODELS, KernelCounter, assert_close_to_gold, compare, load_impl, logits, tiny_llama,
    tiny_vision_tower,
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


# initializer_range per tiny model for end-to-end tests: logits peaked enough to
# avoid FP16 near-ties and to make softcap/sinks matter, without saturating them
# (Gemma scales embeddings by sqrt(hidden), hence the smaller range).
_INIT = {"llama": 0.5, "gemma3": 0.05, "gemma2": 0.05, "gpt_oss": 0.5}


# ---------------------------------------------------------------------------
# P0.1 — kernel support is decided per call, never on the model config
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("head_dim", [64, 48])
def test_backend_kept_for_any_head_dim(head_dim):
    """B6: the old config-based check switched whole models to sdpa (and used
    hidden_size / num_heads, wrong for sub-models). The backend must stay ours."""
    model = load_impl(tiny_llama(head_dim=head_dim), "flash_attn_legacy")
    assert model.config._attn_implementation == "flash_attn_legacy"


@pytest.mark.parametrize("head_dim,expect_kernel", [(64, True), (128, True), (80, True), (72, True), (48, False)])
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


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("arch", ["llama", "gemma3"])
def test_padded_batch_runs_on_kernel(monkeypatch, arch, side):
    """P3.1: a padded batch is unpadded onto the varlen kernel in every layer;
    logits of the real tokens within the noise of FP32 eager."""
    base = TINY_MODELS[arch](initializer_range=_INIT[arch])
    ids = torch.randint(0, base.config.vocab_size, (3, 40), device="cuda")
    mask = torch.ones_like(ids)
    if side == "left":
        mask[1, :10] = 0
        mask[2, :25] = 0
    else:
        mask[1, 30:] = 0
        mask[2, 12:] = 0

    def run(impl, dtype=torch.float16):
        with torch.no_grad():
            return load_impl(base, impl, dtype)(input_ids=ids, attention_mask=mask).logits.float()

    gold, noise = run("eager", torch.float32), run("eager")
    counter = KernelCounter(monkeypatch)
    out = run("flash_attn_legacy")
    full_layers = sum(t == "full_attention" for t in getattr(
        base.config, "layer_types", None) or ["full_attention"] * base.config.num_hidden_layers)
    assert len(counter.varlen_calls) == full_layers
    keep = mask.bool()
    assert_close_to_gold(out[keep], gold[keep], noise[keep], f"{arch} {side} padding")


@pytest.mark.parametrize("cache_implementation", ["dynamic", "static"])
@pytest.mark.parametrize("arch", ["llama", "gemma3"])
def test_generate_left_padded_batch(monkeypatch, arch, cache_implementation):
    """Batched generation with left padding: the prefill runs unpadded on the
    kernel; every step's logits of every row within the noise of FP32 eager."""
    base = TINY_MODELS[arch](initializer_range=_INIT[arch])
    ids = torch.randint(0, base.config.vocab_size, (2, 30), device="cuda")
    mask = torch.ones_like(ids)
    mask[0, :12] = 0
    kw = dict(attention_mask=mask, max_new_tokens=8, min_new_tokens=8, do_sample=False,
              return_dict_in_generate=True, output_logits=True, pad_token_id=0,
              cache_implementation=cache_implementation, disable_compile=True)

    def gen(impl, dtype=torch.float16):
        return torch.stack(load_impl(base, impl, dtype).generate(ids, **kw).logits).float()

    gold, noise = gen("eager", torch.float32), gen("eager")
    counter = KernelCounter(monkeypatch)
    out = gen("flash_attn_legacy")
    assert counter.varlen_calls  # the padded prefill ran on the kernel
    assert_close_to_gold(out, gold, noise, f"{arch} left-padded generate ({cache_implementation})")


def test_cross_attention_with_padded_encoder(monkeypatch):
    """Bidirectional 2D mask over the keys only (cross-attention: decoder queries
    unrelated to the encoder padding): every decoder row must match SDPA."""
    torch.manual_seed(0)
    q = torch.randn(2, 4, 8, 64, device="cuda", dtype=torch.float16)  # 8 decoder tokens
    k, v = (torch.randn(2, 4, 24, 64, device="cuda", dtype=torch.float16) for _ in range(2))
    pad = torch.ones(2, 24, dtype=torch.bool, device="cuda")
    pad[1, 15:] = False  # encoder padding on row 1
    counter = KernelCounter(monkeypatch)
    out, _ = integrations.flash_attn_legacy_forward(_Attn(is_causal=False), q, k, v, pad)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=pad[:, None, None, :])
    assert len(counter.varlen_calls) == 1 and fallback_stats() == {}
    torch.testing.assert_close(out.transpose(1, 2), ref, atol=2e-3, rtol=2e-3)


def test_bart_padded_encoder_teacher_forcing(monkeypatch):
    """Encoder-decoder (BART) with a padded encoder batch and q_len > 1 in the
    decoder: encoder and cross-attention unpadded on the kernel, logits of every
    decoder position within the noise of FP32 eager."""
    from transformers import BartConfig, BartForConditionalGeneration

    torch.manual_seed(0)
    cfg = BartConfig(vocab_size=256, d_model=256, encoder_layers=2, decoder_layers=2,
                     encoder_attention_heads=4, decoder_attention_heads=4,
                     encoder_ffn_dim=512, decoder_ffn_dim=512, max_position_embeddings=128,
                     init_std=0.1)
    base = BartForConditionalGeneration(cfg).eval()
    src = torch.randint(3, 256, (2, 30), device="cuda")
    src_mask = torch.ones_like(src)
    src_mask[1, 18:] = 0
    tgt = torch.randint(3, 256, (2, 12), device="cuda")

    def run(impl, dtype=torch.float16):
        with torch.no_grad():
            return load_impl(base, impl, dtype)(input_ids=src, attention_mask=src_mask,
                                                decoder_input_ids=tgt).logits.float()

    gold, noise = run("eager", torch.float32), run("eager")
    counter = KernelCounter(monkeypatch)
    out = run("flash_attn_legacy")
    assert counter.varlen_calls  # encoder self-attention and cross-attention
    assert_close_to_gold(out, gold, noise, "bart padded encoder")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_unpad_non_fp16_inference(monkeypatch, dtype):
    """bf16 / fp32 padded inference: unpadded onto the kernel (fp16 compute)."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 20, 64, device="cuda", dtype=dtype) for _ in range(3))
    pad = torch.ones(2, 20, dtype=torch.bool, device="cuda")
    pad[0, :7] = False
    counter = KernelCounter(monkeypatch)
    with torch.no_grad():
        out, _ = integrations.flash_attn_legacy_forward(_Attn(is_causal=True), q, k, v, pad)
    mask = pad[:, None, None, :] & torch.ones(20, 20, dtype=torch.bool, device="cuda").tril()
    ref = torch.nan_to_num(torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), attn_mask=mask)).transpose(1, 2)
    assert len(counter.varlen_calls) == 1 and out.dtype == dtype
    real = pad.clone()  # left padding: padded rows see no key
    torch.testing.assert_close(out.float()[real], ref[real], atol=2e-2, rtol=2e-2)


def test_unpad_fp16_overflow_falls_back_finite():
    """fp32 padded inputs beyond the fp16 range: SDPA fallback with the padding and
    causal mask; left-padded query rows (no visible key) stay finite."""
    q, k, v = (torch.randn(2, 4, 16, 64, device="cuda", dtype=torch.float32) for _ in range(3))
    v[0, 0, 10, 0] = 1e6
    pad = torch.ones(2, 16, dtype=torch.bool, device="cuda")
    pad[1, :5] = False
    with torch.no_grad():
        out, _ = integrations.flash_attn_legacy_forward(_Attn(is_causal=True), q, k, v, pad)
    assert fallback_stats() == {"fp16_overflow": 1}
    assert torch.isfinite(out).all()


def test_attention_mask_reason_for_holes():
    """Padding that is not one contiguous block per row (rare) keeps the fallback."""
    q, k, v = _qkv(q_len=16, kv_len=16)
    q, k, v = (t.expand(2, -1, -1, -1).contiguous() for t in (q, k, v))
    pad = torch.ones(2, 16, dtype=torch.bool, device="cuda")
    pad[1, 4:7] = False  # a hole in the middle of row 1
    out, _ = integrations.flash_attn_legacy_forward(_Attn(is_causal=False), q, k, v, pad)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=pad[:, None, None, :])
    assert fallback_stats() == {"attention_mask": 1}
    torch.testing.assert_close(out.transpose(1, 2), ref, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("causal", [True, False])
# right padding only without a cache prefix (q_len == kv_len): the layouts HF produces
@pytest.mark.parametrize("side,q_len,kv_len", [("left", 24, 24), ("right", 24, 24), ("left", 8, 24)])
def test_unpad_matches_masked_reference(monkeypatch, causal, side, q_len, kv_len):
    """Unpadded varlen vs the masked SDPA reference on every row that sees a key,
    forward and gradients; GQA; chunked prefill (q_len < kv_len)."""
    torch.manual_seed(0)
    B, H, H_kv, d = 3, 4, 2, 64
    q = torch.randn(B, H, q_len, d, device="cuda", dtype=torch.float16, requires_grad=True)
    k, v = (torch.randn(B, H_kv, kv_len, d, device="cuda", dtype=torch.float16, requires_grad=True)
            for _ in range(2))
    pad = torch.ones(B, kv_len, dtype=torch.bool, device="cuda")
    if side == "left":
        pad[1, :5] = False
        pad[2, :20] = False
    else:
        pad[1, 19:] = False
        pad[2, 6:] = False
    counter = KernelCounter(monkeypatch)
    out, _ = integrations.flash_attn_legacy_forward(_Attn(is_causal=causal), q, k, v, pad)
    assert len(counter.varlen_calls) == 1 and fallback_stats() == {}

    mask = pad[:, None, None, :]
    if causal:
        mask = mask & torch.ones(q_len, kv_len, dtype=torch.bool, device="cuda").tril(kv_len - q_len)
    rep = lambda t: t.repeat_interleave(H // H_kv, dim=1)  # noqa: E731
    qr, kr, vr = (t.detach().float().requires_grad_() for t in (q, k, v))
    ref = torch.nn.functional.scaled_dot_product_attention(qr, rep(kr), rep(vr), attn_mask=mask)
    ref = torch.nan_to_num(ref).transpose(1, 2)  # rows that see no key: 0, like the kernel
    # Query rows that see at least one key; causal: real rows only (padded rows
    # are 0 by design, as in FA2's unpadding); bidirectional: every row
    sees = mask.expand(B, 1, q_len, kv_len).any(-1)[:, 0]
    if causal:
        sees &= pad[:, kv_len - q_len:]
    torch.testing.assert_close(out.float()[sees], ref[sees], atol=2e-2, rtol=2e-2)
    go = torch.randn_like(out) * sees[:, :, None, None]
    out.backward(go)
    ref.backward(go.float())
    for g, r in ((q.grad, qr.grad), (k.grad, kr.grad), (v.grad, vr.grad)):
        torch.testing.assert_close(g.float(), r, atol=5e-2, rtol=5e-2)


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


@pytest.mark.parametrize("kwarg", ["cu_seq_lens_q", "cu_seq_lens_k"])
@pytest.mark.parametrize("q_len", [16, 1], ids=["prefill", "decode"])
def test_cu_seq_lens_one_sided_raises(kwarg, q_len):
    """cu_seq_lens given for queries only or keys only (no consistent segments):
    must raise, on the kernel path and on the fallback path alike."""
    value, reason = torch.tensor([0, 8, 16], dtype=torch.int32), "cu_seq_lens"
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
    base = tiny_llama(head_dim=48)
    model = load_impl(base, "flash_attn_legacy")
    ids = torch.randint(0, base.config.vocab_size, (1, 16), device="cuda")
    with caplog.at_level(logging.WARNING, logger=integrations.__name__):
        model.generate(ids, max_new_tokens=3, min_new_tokens=3, do_sample=False)
        model.generate(ids, max_new_tokens=3, min_new_tokens=3, do_sample=False)
    warnings = [r for r in caplog.records
                if r.name == integrations.__name__ and r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "head_dim=48" in warnings[0].getMessage()
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


def _beam_rows_error(x, y, ignore_worst=0):
    """Worst 1-cos of a one-to-one matching between the rows (beams) of x and y:
    beams with near-equal scores can swap order between backends. Greedy on the
    most similar pairs first, so two nearly identical beams cannot both claim
    the same reference row; a duplicated beam is left with a dissimilar row.
    `ignore_worst` drops that many worst-matched rows (beam forks, see caller)."""
    x = torch.nn.functional.normalize(x.float(), dim=-1)
    y = torch.nn.functional.normalize(y.float(), dim=-1)
    cos = (x @ y.T).cpu()
    free_x, free_y, errors = set(range(x.shape[0])), set(range(y.shape[0])), []
    for flat in torch.argsort(cos.flatten(), descending=True).tolist():
        i, j = divmod(flat, cos.shape[1])
        if i in free_x and j in free_y:
            errors.append(1.0 - cos[i, j].item())
            free_x.discard(i)
            free_y.discard(j)
    errors = sorted(errors)[:len(errors) - ignore_worst] or [0.0]
    return errors[-1]


# Assisted decoding manages its own (dynamic) cache: only that combination exists.
_GENERATE_CASES = [(c, m) for c in ("static", "offloaded_static", "dynamic")
                   for m in ("greedy", "beam", "assisted")
                   if m != "assisted" or c == "dynamic"]


@pytest.mark.parametrize("arch", list(_INIT))
@pytest.mark.parametrize("cache_implementation,mode", _GENERATE_CASES)
def test_generate_matches_gold(arch, cache_implementation, mode):
    """End-to-end generate vs FP32 eager (gold), FP16 eager as noise floor:
    every step's logits (all beams) and the final sequences."""
    base = TINY_MODELS[arch](initializer_range=_INIT[arch])
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
            extra["assistant_model"] = load_impl(base, "eager", dtype)  # any arch supports eager
            extra["cache_implementation"] = None  # assisted decoding manages its own cache
        return load_impl(base, impl, dtype).generate(ids, **{**kw, **extra})

    gold, noise, out = gen("eager", torch.float32), gen("eager"), gen("flash_attn_legacy")
    label = f"{arch}/{cache_implementation}/{mode}"
    # Beam search: an FP16 near-tie in the beam scores can keep a different
    # hypothesis in one beam for one step (a fork), with every other beam
    # unchanged; a wrong attention would affect all beams and all steps. So a
    # single beam may differ at a single step of the whole generation; the final
    # sequences are still checked below.
    forked_steps = []
    for step, (o, n, g) in enumerate(zip(out.logits, noise.logits, gold.logits)):
        assert torch.isfinite(g).all(), f"{label}: non-finite reference logits"
        err, err_noise = _beam_rows_error(o, g), _beam_rows_error(n, g)
        bound = max(3.0 * err_noise, 1e-3)  # same rule as hf_helpers.assert_within_noise
        if err > bound and mode == "beam" and not forked_steps:
            err = _beam_rows_error(o, g, ignore_worst=1)
            forked_steps.append(step)
        assert err <= bound, f"{label}: step {step} 1-cos={err:.2e} (noise {err_noise:.2e})"
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
    assert m.shape == (2, 25) and m.dtype == torch.bool  # padding mask -> unpad


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
    it must be applied (unpadded onto the kernel), never treated as 'first n keys valid'."""
    q, k, v = _qkv(q_len=16, kv_len=16)
    q, k, v = (t.expand(2, -1, -1, -1).contiguous() for t in (q, k, v))
    pad = torch.ones(2, 16, dtype=torch.long, device="cuda")
    pad[1, :6] = 0  # left padding on row 1
    out, _ = integrations.flash_attn_legacy_forward(_Attn(is_causal=True), q, k, v, pad)
    causal = torch.ones(16, 16, dtype=torch.bool, device="cuda").tril()
    ref_mask = pad[:, None, None, :].bool() & causal
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=ref_mask)
    assert fallback_stats() == {}  # contiguous padding: unpadded onto the kernel
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


# ---------------------------------------------------------------------------
# P0.3 — softcap / sinks / position_bias: eager fallback that applies them
# (temporary until the kernel supports them, docs/PLAN.md §3b)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch,reason", [("gemma2", "softcap"), ("gpt_oss", "s_aux")])
@pytest.mark.parametrize("n_tokens", [24, 300])  # both beyond the sliding window (8)
def test_eager_feature_models_vs_gold(monkeypatch, arch, reason, n_tokens):
    base = TINY_MODELS[arch](initializer_range=_INIT[arch])
    ids = torch.randint(0, base.config.vocab_size, (1, n_tokens))
    gold = logits(load_impl(base, "eager", torch.float32), ids)
    # Guard: the feature must matter on this model, or the test proves nothing.
    without = {"gemma2": dict(attn_logit_softcapping=None), "gpt_oss": {}}[arch]
    plain = TINY_MODELS[arch](initializer_range=_INIT[arch], **without)
    if arch == "gpt_oss":
        with torch.no_grad():
            for layer in plain.model.layers:
                layer.self_attn.sinks.fill_(-1e4)  # sinks with no mass
    _, cos_plain, _ = compare(logits(load_impl(plain, "eager", torch.float32), ids), gold)
    assert 1.0 - cos_plain > 1e-3, f"{reason} does not change the gold logits"
    noise = logits(load_impl(base, "eager"), ids)
    counter = KernelCounter(monkeypatch)
    out = logits(load_impl(base, "flash_attn_legacy"), ids)
    assert counter.calls == [] and fallback_stats() == {reason: base.config.num_hidden_layers}
    assert_close_to_gold(out, gold, noise, f"{arch} N={n_tokens}")


def _reference(q, k, v, scaling, mask=None, softcap=None, sinks=None, bias=None):
    rep = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    s = (q.float() @ k.float().transpose(-2, -1)) * scaling
    if softcap:
        s = softcap * torch.tanh(s / softcap)
    if bias is not None:
        s = s + bias
    if mask is not None:
        s = s.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else s + mask
    if sinks is not None:
        s = torch.cat([s, sinks.reshape(1, -1, 1, 1).expand(*s.shape[:-1], 1)], dim=-1)
    p = torch.softmax(s, dim=-1)
    if sinks is not None:
        p = p[..., :-1]
    return p @ v.float()


@pytest.mark.parametrize("q_len,kv_len", [(16, 16), (5, 40), (1, 40), (700, 700)])
@pytest.mark.parametrize("kv_heads", [4, 2], ids=["mha", "gqa"])
def test_eager_attention_unit(monkeypatch, q_len, kv_len, kv_heads):
    """_eager_attention (grouped GQA, query blocks) vs a direct fp32 formula with
    every feature on. Softcap 2.0 and scale 0.5 saturate the logits, so an ignored
    or misplaced softcap is far above the tolerance (checked below)."""
    monkeypatch.setattr(integrations, "_EAGER_BLOCK_BYTES", 4 * 1024 * 1024)  # several blocks
    torch.manual_seed(0)
    q, k, v = _qkv(q_len=q_len, kv_len=kv_len)
    k, v = k[:, :kv_heads].contiguous(), v[:, :kv_heads].contiguous()
    sinks = torch.randn(4, device="cuda")
    bias = torch.randn(1, 4, q_len, kv_len, device="cuda")
    causal = integrations._causal_mask(q_len, kv_len, "cuda")
    additive = torch.where(causal, 0.0, float("-inf")).float()  # -inf must not give NaN
    for mask in (causal, additive[None, None]):
        out = integrations._eager_attention(q, k, v, mask, 0.5, 0.0, softcap=2.0,
                                            sinks=sinks, position_bias=bias)
        ref = _reference(q, k, v, 0.5, mask.bool() if mask.dim() == 2 else mask,
                         softcap=2.0, sinks=sinks, bias=bias)
        no_cap = _reference(q, k, v, 0.5, causal, sinks=sinks, bias=bias)
        assert (ref - no_cap).abs().max() > 0.05  # softcap matters here
        assert torch.isfinite(out).all()
        torch.testing.assert_close(out.float(), ref, atol=2e-3, rtol=2e-3)


def test_eager_attention_gqa_per_head_mask_and_3d_bias():
    """Per-head (head dim H) mask under GQA and a 3D bias go through grouped()."""
    torch.manual_seed(0)
    q, k, v = _qkv(q_len=12, kv_len=20)
    k, v = k[:, :2].contiguous(), v[:, :2].contiguous()
    mask = torch.rand(1, 4, 12, 20, device="cuda") > 0.3
    mask[..., 0] = True  # every row keeps at least one key
    bias = torch.randn(4, 12, 20, device="cuda")
    out = integrations._eager_attention(q, k, v, mask, 0.5, 0.0, position_bias=bias)
    ref = _reference(q, k, v, 0.5, mask, bias=bias[None])
    torch.testing.assert_close(out.float(), ref, atol=2e-3, rtol=2e-3)
    with pytest.raises(ValueError):
        integrations._eager_attention(q, k, v, mask[0, 0, 0], 0.5, 0.0)  # 1D mask


def test_eager_attention_gqa_does_not_copy_kv_per_query_head():
    """Folding query heads into rows: peak memory stays far below rep copies of K/V."""
    B, H, H_kv, kv_len, d = 1, 16, 2, 8192, 64
    q = torch.randn(B, H, 1, d, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H_kv, kv_len, d, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    kv_fp32_bytes = 2 * k.numel() * 4  # K and V once in fp32
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    integrations._eager_attention(q, k, v, None, 0.125, 0.0)
    peak = torch.cuda.max_memory_allocated() - base
    rep = H // H_kv
    assert peak < kv_fp32_bytes * rep / 2, f"peak {peak} B suggests K/V copied per query head"


@pytest.mark.parametrize("kwarg,reason", [("softcap", "softcap"), ("s_aux", "s_aux"),
                                          ("position_bias", "position_bias")])
@pytest.mark.parametrize("q_len", [16, 1], ids=["prefill", "decode"])
def test_eager_kwargs_routed_and_applied(kwarg, reason, q_len):
    torch.manual_seed(0)
    q, k, v = _qkv(q_len=q_len, kv_len=16)
    value = {"softcap": 2.0, "s_aux": torch.randn(4, device="cuda"),
             "position_bias": torch.randn(1, 4, q_len, 16, device="cuda")}[kwarg]
    out, _ = integrations.flash_attn_legacy_forward(_Attn(is_causal=True), q, k, v, None,
                                                    scaling=0.5, **{kwarg: value})
    ref = _reference(q, k, v, 0.5, integrations._causal_mask(q_len, 16, "cuda"),
                     softcap=value if kwarg == "softcap" else None,
                     sinks=value if kwarg == "s_aux" else None,
                     bias=value if kwarg == "position_bias" else None)
    assert fallback_stats() == {reason: 1}
    torch.testing.assert_close(out.transpose(1, 2).float(), ref, atol=2e-3, rtol=2e-3)


# ---------------------------------------------------------------------------
# P0.4 — packed segments: cu_seq_lens (VLM vision encoders) and position_ids
# restarting inside a sequence (packing, B4)
# ---------------------------------------------------------------------------

_CU = [0, 7, 40, 41, 90]  # uneven segments, one of length 1


def _per_segment_reference(q, k, v, scaling, causal):
    outs = []
    for a, b in zip(_CU[:-1], _CU[1:]):
        outs.append(_reference(q[:, :, a:b], k[:, :, a:b], v[:, :, a:b], scaling,
                               integrations._causal_mask(b - a, b - a, "cuda") if causal else None))
    return torch.cat(outs, dim=2)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("kv_heads", [4, 2], ids=["mha", "gqa"])
@pytest.mark.parametrize("head_dim,path", [(64, "varlen"), (80, "varlen"), (48, "sdpa")])
def test_cu_seq_lens_direct(monkeypatch, causal, kv_heads, head_dim, path):
    torch.manual_seed(0)
    q, k, v = _qkv(q_len=_CU[-1], kv_len=_CU[-1], d=head_dim)
    k, v = k[:, :kv_heads].contiguous(), v[:, :kv_heads].contiguous()
    cu = torch.tensor(_CU, dtype=torch.int32, device="cuda")
    counter = KernelCounter(monkeypatch)
    out, _ = integrations.flash_attn_legacy_forward(
        _Attn(is_causal=causal), q, k, v, None, scaling=0.125,
        cu_seq_lens_q=cu, cu_seq_lens_k=cu, is_causal=causal)
    ref = _per_segment_reference(q, k, v, 0.125, causal)
    if path == "varlen":
        assert counter.varlen_calls == [(_CU[-1], len(_CU) - 1)] and fallback_stats() == {}
    else:
        assert counter.varlen_calls == [] and fallback_stats() == {"head_dim": 1}
    torch.testing.assert_close(out.transpose(1, 2).float(), ref, atol=2e-3, rtol=2e-3)


def test_cu_seq_lens_with_softcap_uses_segment_mask_in_eager():
    torch.manual_seed(0)
    q, k, v = _qkv(q_len=_CU[-1], kv_len=_CU[-1])
    cu = torch.tensor(_CU, dtype=torch.int32, device="cuda")
    out, _ = integrations.flash_attn_legacy_forward(
        _Attn(is_causal=True), q, k, v, None, scaling=0.5, softcap=2.0,
        cu_seq_lens_q=cu, cu_seq_lens_k=cu)
    outs = []
    for a, b in zip(_CU[:-1], _CU[1:]):
        outs.append(_reference(q[:, :, a:b], k[:, :, a:b], v[:, :, a:b], 0.5,
                               integrations._causal_mask(b - a, b - a, "cuda"), softcap=2.0))
    assert fallback_stats() == {"softcap": 1}
    torch.testing.assert_close(out.transpose(1, 2).float(), torch.cat(outs, dim=2), atol=2e-3, rtol=2e-3)


def test_cu_seq_lens_mismatch_raises():
    q, k, v = _qkv(q_len=_CU[-1], kv_len=_CU[-1])
    cu = torch.tensor(_CU, dtype=torch.int32, device="cuda")
    with pytest.raises(NotImplementedError):
        integrations.flash_attn_legacy_forward(_Attn(is_causal=False), q, k, v, None,
                                               cu_seq_lens_q=cu, cu_seq_lens_k=cu.flip(0))


def _packed_batch(base, lengths, seed=0):
    g = torch.Generator().manual_seed(seed)
    seqs = [torch.randint(0, base.config.vocab_size, (n,), generator=g) for n in lengths]
    ids = torch.cat(seqs)[None]
    position_ids = torch.cat([torch.arange(n) for n in lengths])[None]
    return seqs, ids, position_ids


@pytest.mark.parametrize("use_cache", [True, False])
@pytest.mark.parametrize("arch", ["llama", "gemma3"])  # gemma3: sliding-window layers
def test_packed_position_ids_forward(monkeypatch, arch, use_cache):
    """B4: with a cache transformers builds no packing mask (and only the window
    on sliding layers); our forward must still keep the segments apart (varlen
    kernel, or the segment mask intersected with the window). Without a cache it
    builds a block mask (-> SDPA fallback). All must equal each sequence alone."""
    base = TINY_MODELS[arch](initializer_range=_INIT[arch])
    lengths = [30, 17, 45]
    seqs, ids, position_ids = _packed_batch(base, lengths)
    gold = torch.cat([logits(load_impl(base, "eager", torch.float32), s_[None]) for s_ in seqs])
    noise = torch.cat([logits(load_impl(base, "eager"), s_[None]) for s_ in seqs])
    counter = KernelCounter(monkeypatch)
    out = logits(load_impl(base, "flash_attn_legacy"), ids,
                 position_ids=position_ids.cuda(), use_cache=use_cache)
    n_layers = base.config.num_hidden_layers
    full = sum(t == "full_attention" for t in getattr(base.config, "layer_types", None)
               or ["full_attention"] * n_layers)
    if use_cache:
        assert counter.varlen_calls == [(sum(lengths), 3)] * full
        assert fallback_stats() == ({"attention_mask": n_layers - full} if full < n_layers else {})
    else:
        assert counter.varlen_calls == [] and fallback_stats() == {"attention_mask": n_layers}
    assert counter.calls == []
    assert_close_to_gold(out, gold, noise, f"{arch} packed use_cache={use_cache}",
                         near_tie_slack=True)


@pytest.mark.parametrize("positions,why", [
    (lambda n: torch.arange(10, 10 + n), "offset arange: a plain sequence"),
    (lambda n: torch.cat([torch.arange(20), torch.arange(25, 25 + n - 20)]), "PoSE-style jump"),
])
def test_position_ids_not_a_packing(monkeypatch, caplog, positions, why):
    """Only restarts at the initial position split; anything else is one sequence."""
    base = tiny_llama()
    ids = torch.randint(0, base.config.vocab_size, (1, 48))
    counter = KernelCounter(monkeypatch)
    with caplog.at_level(logging.WARNING, logger=integrations.__name__):
        logits(load_impl(base, "flash_attn_legacy"), ids,
               position_ids=positions(48)[None].cuda(), use_cache=True)
    assert counter.varlen_calls == [], why
    assert counter.calls == [(48, 48)] * base.config.num_hidden_layers, why
    jump_warnings = [r for r in caplog.records if "jump" in r.getMessage()]
    assert len(jump_warnings) == (1 if "jump" in why else 0)


def test_packing_first_segment_mid_document(monkeypatch):
    """A first piece starting at position 37 (document split by the collator),
    then segments restarting at 0: still a packing of 3 segments."""
    q, k, v = _qkv(q_len=60, kv_len=60)
    pos = torch.cat([torch.arange(37, 57), torch.arange(25), torch.arange(15)])[None].cuda()
    counter = KernelCounter(monkeypatch)
    integrations.flash_attn_legacy_forward(_Attn(is_causal=True), q, k, v, None, position_ids=pos)
    assert counter.varlen_calls == [(60, 3)]


def test_segment_cache_follows_in_place_writes(monkeypatch):
    """A reused position_ids buffer refilled in place must not reuse old segments."""
    q, k, v = _qkv(q_len=40, kv_len=40)
    pos = torch.cat([torch.arange(20), torch.arange(20)])[None].cuda()
    counter = KernelCounter(monkeypatch)
    integrations.flash_attn_legacy_forward(_Attn(is_causal=True), q, k, v, None, position_ids=pos)
    pos.copy_(torch.arange(40)[None])  # same object, same length, no packing anymore
    integrations.flash_attn_legacy_forward(_Attn(is_causal=True), q, k, v, None, position_ids=pos)
    assert counter.varlen_calls == [(40, 2)] and counter.calls == [(40, 40)]


def test_cu_seq_lens_cached_but_kv_len_checked_every_call():
    q, k, v = _qkv(q_len=90, kv_len=90)
    cu = torch.tensor(_CU, dtype=torch.int32, device="cuda")
    integrations.flash_attn_legacy_forward(_Attn(is_causal=False), q, k, v, None,
                                           cu_seq_lens_q=cu, cu_seq_lens_k=cu)
    _, k2, v2 = _qkv(q_len=90, kv_len=120)
    with pytest.raises(NotImplementedError):
        integrations.flash_attn_legacy_forward(_Attn(is_causal=False), q, k2, v2, None,
                                               cu_seq_lens_q=cu, cu_seq_lens_k=cu)


def test_segments_intersect_broadcast_additive_mask():
    q, k, v = _qkv(q_len=_CU[-1], kv_len=_CU[-1])
    cu = torch.tensor(_CU, dtype=torch.int32, device="cuda")
    additive = torch.zeros(1, 1, 1, _CU[-1], device="cuda", dtype=torch.float16)
    out, _ = integrations.flash_attn_legacy_forward(
        _Attn(is_causal=False), q, k, v, additive, scaling=0.125,
        cu_seq_lens_q=cu, cu_seq_lens_k=cu)
    torch.testing.assert_close(out.transpose(1, 2).float(),
                               _per_segment_reference(q, k, v, 0.125, False), atol=2e-3, rtol=2e-3)


def test_padded_batch1_no_ambiguous_positions_warning(caplog):
    q, k, v = _qkv(q_len=5, kv_len=5)
    pad = torch.tensor([[0, 0, 1, 1, 1]], device="cuda")
    pos = torch.tensor([[1, 1, 0, 1, 2]], device="cuda")  # HF: cumsum - 1, pads filled with 1
    with caplog.at_level(logging.WARNING, logger=integrations.__name__):
        integrations.flash_attn_legacy_forward(_Attn(is_causal=True), q, k, v, pad, position_ids=pos)
    assert not [r for r in caplog.records if "jump" in r.getMessage()]


@pytest.mark.parametrize("edges,why", [([0, 7, 40, 80], "does not reach the end"),
                                       ([0, 7, 7, 90], "zero-length segment"),
                                       ([3, 40, 90], "does not start at 0")])
def test_cu_seq_lens_invalid_raises(edges, why):
    q, k, v = _qkv(q_len=90, kv_len=90)
    cu = torch.tensor(edges, dtype=torch.int32, device="cuda")
    with pytest.raises(NotImplementedError):
        integrations.flash_attn_legacy_forward(_Attn(is_causal=False), q, k, v, None,
                                               cu_seq_lens_q=cu, cu_seq_lens_k=cu)


def test_packed_position_ids_training_grads():
    """Packed training (DataCollatorWithFlattening layout, use_cache=True as in a
    custom loop): loss and gradients vs FP32 eager on the sequences run alone."""
    base = tiny_llama(initializer_range=0.1)
    lengths = [30, 17, 45]
    seqs, ids, position_ids = _packed_batch(base, lengths)

    def loss_and_grads(model, packed):
        model.train()
        if packed:
            out = model(input_ids=ids.cuda(), position_ids=position_ids.cuda(), use_cache=True)
            lg = out.logits.float()[0]
        else:
            lg = torch.cat([model(input_ids=s_[None].cuda()).logits.float()[0] for s_ in seqs])
        # next-token loss inside each segment only (no cross-boundary targets)
        tgt = torch.cat([torch.cat([s_[1:], torch.tensor([-100])]) for s_ in seqs]).cuda()
        loss = torch.nn.functional.cross_entropy(lg, tgt, ignore_index=-100)
        loss.backward()
        g = torch.cat([getattr(layer.self_attn, proj).weight.grad.float().flatten()
                       for layer in (model.model.layers[0], model.model.layers[-1])
                       for proj in ("q_proj", "k_proj", "v_proj")])
        return loss.item(), g

    gold_loss, gold_g = loss_and_grads(load_impl(base, "eager", torch.float32), packed=False)
    noise_loss, noise_g = loss_and_grads(load_impl(base, "eager"), packed=False)
    loss, g = loss_and_grads(load_impl(base, "flash_attn_legacy"), packed=True)
    err = lambda x: 1 - torch.nn.functional.cosine_similarity(x, gold_g, dim=0).item()  # noqa: E731
    assert abs(loss - gold_loss) <= max(3 * abs(noise_loss - gold_loss), 1e-3), (loss, gold_loss)
    assert err(g) <= max(3 * err(noise_g), 1e-5), (err(g), err(noise_g))


@pytest.mark.parametrize("kind,head_dim,expect_varlen", [
    ("qwen2_vl", 64, True), ("qwen2_vl", 80, True), ("qwen2_vl", 48, False), ("qwen2_5_vl", 64, True)])
def test_vision_tower_multiple_images(monkeypatch, kind, head_dim, expect_varlen):
    """VLM vision encoders: patches of several images attend within their image
    (and window, Qwen2.5-VL) via cu_seq_lens."""
    tower = tiny_vision_tower(kind, head_dim=head_dim)
    cfg = tower.config
    grid = torch.tensor([[1, 8, 8], [1, 12, 8], [1, 4, 4]])
    patch = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size ** 2
    x = torch.randn(int(grid.prod(-1).sum()), patch, generator=torch.Generator().manual_seed(0))

    def run(impl, dtype=torch.float16):
        m = load_impl(tower, impl, dtype)
        with torch.no_grad():
            return m(x.to("cuda", dtype), grid_thw=grid.cuda()).last_hidden_state.float()

    gold, noise = run("eager", torch.float32), run("eager")
    counter = KernelCounter(monkeypatch)
    out = run("flash_attn_legacy")
    if expect_varlen:
        assert len(counter.varlen_calls) == cfg.depth and fallback_stats() == {}
    else:
        assert counter.varlen_calls == [] and fallback_stats() == {"head_dim": cfg.depth}
    assert_close_to_gold(out, gold, noise, f"{kind} d={head_dim}")


# ---------------------------------------------------------------------------
# P0.5 — non-fp16 inputs: bf16/fp32 run on the kernel after a checked fp16
# conversion; values beyond the fp16 range fall back (never a silent inf)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("arch", ["llama", "gemma3"])
def test_non_fp16_models_run_kernel(monkeypatch, arch, dtype):
    base = TINY_MODELS[arch](initializer_range=_INIT[arch])
    ids = torch.randint(0, base.config.vocab_size, (1, 200))
    gold = logits(load_impl(base, "eager", torch.float32), ids)
    # noise floor: eager in the model dtype for bf16; fp16 (our compute dtype) for fp32
    noise = logits(load_impl(base, "eager", torch.bfloat16 if dtype == torch.bfloat16
                             else torch.float16), ids)
    counter = KernelCounter(monkeypatch)
    out = logits(load_impl(base, "flash_attn_legacy", dtype), ids)
    full = sum(t == "full_attention" for t in getattr(base.config, "layer_types", None)
               or ["full_attention"] * base.config.num_hidden_layers)
    assert len(counter.calls) == full and "fp16_overflow" not in fallback_stats()
    assert_close_to_gold(out, gold, noise, f"{arch} {dtype}")


def test_bf16_training_falls_back_and_matches_gold():
    """bf16 training: kernel gradients would be fp16 without loss scaling (underflow),
    so the call goes to SDPA in bf16 (reason non_fp16_training)."""
    base = tiny_llama(initializer_range=0.1)
    ids = torch.randint(0, base.config.vocab_size, (1, 64))

    def grads(model):
        model.train()
        out = model(input_ids=ids.cuda(), labels=ids.cuda())
        out.loss.backward()
        return out.loss.item(), torch.cat([getattr(model.model.layers[0].self_attn, p).weight.grad
                                           .float().flatten() for p in ("q_proj", "k_proj", "v_proj")])

    gold_loss, gold_g = grads(load_impl(base, "eager", torch.float32))
    noise_loss, noise_g = grads(load_impl(base, "eager", torch.bfloat16))
    loss, g = grads(load_impl(base, "flash_attn_legacy", torch.bfloat16))
    assert fallback_stats() == {"non_fp16_training": base.config.num_hidden_layers}
    err = lambda x: 1 - torch.nn.functional.cosine_similarity(x, gold_g, dim=0).item()  # noqa: E731
    assert abs(loss - gold_loss) <= max(3 * abs(noise_loss - gold_loss), 1e-3), (loss, gold_loss)
    assert err(g) <= max(3 * err(noise_g), 1e-5), (err(g), err(noise_g))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("where", ["v", "k", "segments"])
def test_fp16_overflow_falls_back_in_original_dtype(dtype, where):
    torch.manual_seed(0)
    q, k, v = _qkv(q_len=_CU[-1], kv_len=_CU[-1], dtype=dtype)
    (k if where == "k" else v)[0, 0, 3, 5] = 1e5  # beyond fp16 (65504)
    kw = {}
    if where == "segments":
        cu = torch.tensor(_CU, dtype=torch.int32, device="cuda")
        kw = dict(cu_seq_lens_q=cu, cu_seq_lens_k=cu)
    out, _ = integrations.flash_attn_legacy_forward(_Attn(is_causal=True), q, k, v, None,
                                                    scaling=0.125, **kw)
    if where == "segments":
        ref = _per_segment_reference(q, k, v, 0.125, True).transpose(1, 2)
    else:
        ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=0.125).transpose(1, 2)
    assert fallback_stats() == {"fp16_overflow": 1}
    assert out.dtype == dtype and torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), ref.float(), atol=3e-2, rtol=3e-2)


def test_bf16_eval_with_grad_enabled_warns(caplog):
    """eval() without inference_mode: weights require grad -> SDPA; must be visible."""
    base = tiny_llama()
    model = load_impl(base, "flash_attn_legacy", torch.bfloat16).eval()
    ids = torch.randint(0, base.config.vocab_size, (1, 32), device="cuda")
    with caplog.at_level(logging.WARNING, logger=integrations.__name__):
        model(input_ids=ids)
    assert any("inference_mode" in r.getMessage() for r in caplog.records)
    reset_fallback_stats()
    with torch.inference_mode():
        model(input_ids=ids)
    assert fallback_stats() == {}  # kernel in every layer


def test_mixed_kv_dtype_computed_in_query_dtype(monkeypatch):
    q, k, v = _qkv()
    counter = KernelCounter(monkeypatch)
    out, _ = integrations.flash_attn_legacy_forward(_Attn(is_causal=True), q, k.float(), v.float(), None)
    assert out.dtype == torch.float16 and counter.calls == [(16, 16)]


# ---------------------------------------------------------------------------
# P0.7 — edge cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module_causal,kwarg,expected", [
    (True, None, True), (False, None, False),   # None: "not given" -> module attribute
    (True, False, False), (False, True, True),  # explicit kwarg overrides the module
])
def test_is_causal_kwarg_vs_module_attribute(module_causal, kwarg, expected):
    q, k, v = _qkv()
    out, _ = integrations.flash_attn_legacy_forward(_Attn(is_causal=module_causal), q, k, v, None,
                                                    scaling=0.125, is_causal=kwarg)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=expected,
                                                           scale=0.125).transpose(1, 2)
    torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-3)
