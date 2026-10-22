"""
Fast HuggingFace integration tests on tiny random-weight models (no download).

Reference ("gold") is the same weights in FP32 with eager attention; FP16 eager
on the same input gives the noise floor (see hf_helpers.assert_close_to_gold).
Each regression test here targets one item of docs/PLAN.md (P0.x).

Run:  pytest tests/test_hf_integration.py -v
"""

import logging
from collections import Counter

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

transformers = pytest.importorskip("transformers")

from flash_attn_legacy import fallback_stats, integrations, reset_fallback_stats  # noqa: E402
from .hf_helpers import (  # noqa: E402
    TINY_MODELS, KernelCounter, assert_close_to_gold, compare, layer_types, load_impl, logits,
    tiny_llama, tiny_vision_tower,
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
_INIT = {"llama": 0.5, "gemma3": 0.05, "gemma2": 0.05, "gpt_oss": 0.5, "mistral": 0.5, "phi3": 0.5}


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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_decode_on_kernel(monkeypatch, dtype):
    """Decode steps run on the decode kernel (P6.1), bf16 caches read as they are."""
    base = tiny_llama()
    model = load_impl(base, "flash_attn_legacy", dtype)
    ids = torch.randint(0, base.config.vocab_size, (1, 32), device="cuda")
    counter = KernelCounter(monkeypatch)
    with torch.no_grad():
        model.generate(ids, max_new_tokens=4, min_new_tokens=4, do_sample=False)
    # 4 new tokens: the 1st comes from the prefill, 3 decode steps per layer
    n_layers = base.config.num_hidden_layers
    assert len(counter.decode_calls) == 3 * n_layers and fallback_stats() == {}


@pytest.mark.parametrize("case", ["broadcast_batch", "all_true", "holes", "window", "too_many_rows"])
def test_decode_4d_mask(monkeypatch, case):
    """q_len == 1 with a boolean [B or 1, 1, 1, kv] mask: the decode kernel with
    the mask (holes included); more query heads per KV head than the kernel
    handles go to SDPA with the mask as it is; all within FP32."""
    torch.manual_seed(0)
    B, kv = 3, 40
    rows = integrations.DECODE_MAX_ROWS + 1 if case == "too_many_rows" else 4
    q = torch.randn(B, rows, 1, 64, device="cuda", dtype=torch.float16)
    k, v = (torch.randn(B, 1, kv, 64, device="cuda", dtype=torch.float16) for _ in range(2))
    keep = torch.ones(B, kv, dtype=torch.bool, device="cuda")
    if case != "all_true":
        keep[:, :5] = False  # left padding
        keep[:, 30:] = False  # empty slots of a static cache
    if case == "holes":
        keep[1, 12] = False
    mask = keep[:1, None, None, :] if case == "broadcast_batch" else keep[:, None, None, :]
    kw, ref_mask = {}, mask.expand(B, -1, -1, -1)
    if case == "window":  # sliding_window 8 ends at the query (key 29), not at the cache end
        kw["sliding_window"] = 8
        ref_mask = ref_mask & (torch.arange(kv, device="cuda") > 29 - 8)
    counter = KernelCounter(monkeypatch)
    out, _ = integrations.flash_attn_legacy_forward(_Attn(is_causal=True), q, k, v, mask, scaling=0.125, **kw)
    ref = _reference(q, k, v, 0.125, ref_mask)
    torch.testing.assert_close(out.transpose(1, 2).float(), ref, atol=2e-3, rtol=2e-3)
    kernel = case != "too_many_rows"
    assert bool(counter.decode_calls) == kernel
    expected = {} if kernel else {"decode": 1}
    assert fallback_stats() == expected


def test_decode_fp32_falls_back():
    base = tiny_llama()
    model = load_impl(base, "flash_attn_legacy", torch.float32)
    ids = torch.randint(0, base.config.vocab_size, (1, 16), device="cuda")
    with torch.no_grad():
        model.generate(ids, max_new_tokens=3, min_new_tokens=3, do_sample=False)
    assert fallback_stats().get("decode") == 2 * base.config.num_hidden_layers


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("arch", ["llama", "gemma3", "mistral", "gpt_oss"])
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
    assert len(counter.varlen_calls) == base.config.num_hidden_layers  # sliding layers too (P4.1)
    keep = mask.bool()
    assert_close_to_gold(out[keep], gold[keep], noise[keep], f"{arch} {side} padding")


@pytest.mark.parametrize("pad", [12, 26], ids=["pad12", "pad26"])  # 26: padding inside the window (8)
@pytest.mark.parametrize("cache_implementation", ["dynamic", "static"])
@pytest.mark.parametrize("arch", ["llama", "gemma3", "mistral", "gpt_oss"])
def test_generate_left_padded_batch(monkeypatch, arch, cache_implementation, pad):
    """Batched generation with left padding: the prefill runs unpadded on the
    kernel, the decode steps on the decode kernel (key_mask); every step's
    logits of every row within the noise of FP32 eager."""
    base = TINY_MODELS[arch](initializer_range=_INIT[arch])
    ids = torch.randint(0, base.config.vocab_size, (2, 30), device="cuda")
    mask = torch.ones_like(ids)
    mask[0, :pad] = 0
    kw = dict(attention_mask=mask, max_new_tokens=8, min_new_tokens=8, do_sample=False,
              return_dict_in_generate=True, output_logits=True, pad_token_id=0,
              cache_implementation=cache_implementation, disable_compile=True)

    def gen(impl, dtype=torch.float16):
        return torch.stack(load_impl(base, impl, dtype).generate(ids, **kw).logits).float()

    gold, noise = gen("eager", torch.float32), gen("eager")
    counter = KernelCounter(monkeypatch)
    out = gen("flash_attn_legacy")
    assert counter.varlen_calls  # the padded prefill ran on the kernel
    assert counter.decode_calls and "decode" not in fallback_stats() and "attention_mask" not in fallback_stats()
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


def test_unpad_beyond_fp16_on_kernel(monkeypatch):
    """fp32 padded inputs beyond the fp16 range: on the varlen kernel (scaled into
    fp16 on the device), every valid row within fp16 accuracy of SDPA in fp32."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 16, 64, device="cuda", dtype=torch.float32) for _ in range(3))
    v[0, 0, 10, 0] = 1e6
    pad = torch.ones(2, 16, dtype=torch.bool, device="cuda")
    pad[1, :5] = False
    counter = KernelCounter(monkeypatch)
    with torch.no_grad():
        out, _ = integrations.flash_attn_legacy_forward(_Attn(is_causal=True), q, k, v, pad)
    mask = pad[:, None, None, :] & torch.ones(16, 16, dtype=torch.bool, device="cuda").tril()
    ref = torch.nan_to_num(torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)).transpose(1, 2)
    assert counter.varlen_calls and fallback_stats() == {}
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[pad], ref[pad], atol=2e-2 * 1e6 / 16, rtol=2e-2)


def test_fp16_autocast_training_runs_on_kernel(monkeypatch):
    """Mixed precision (fp32 weights, fp16 autocast): HF's rotary makes q / k fp32;
    the attention still computes in fp16 on the kernel, as SDPA does under
    autocast. Packed input; loss and gradients vs FP32 eager on the sequences run
    alone, noise floor: fp16-autocast eager on the sequences alone."""
    base = tiny_llama(initializer_range=0.1)
    lengths = [30, 17, 45]
    seqs, ids, position_ids = _packed_batch(base, lengths)
    tgt = torch.cat([torch.cat([s_[1:], torch.tensor([-100])]) for s_ in seqs]).cuda()

    def loss_and_grads(impl, packed, autocast):
        model = load_impl(base, impl, torch.float32).train()
        with torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
            if packed:
                lg = model(input_ids=ids.cuda(), position_ids=position_ids.cuda()).logits.float()[0]
            else:
                lg = torch.cat([model(input_ids=s_[None].cuda()).logits.float()[0] for s_ in seqs])
            loss = torch.nn.functional.cross_entropy(lg, tgt, ignore_index=-100)
        loss.backward()
        return loss.item(), torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])

    gold_loss, gold_g = loss_and_grads("eager", packed=False, autocast=False)
    noise_loss, noise_g = loss_and_grads("eager", packed=False, autocast=True)
    counter = KernelCounter(monkeypatch)
    loss, g = loss_and_grads("flash_attn_legacy", packed=True, autocast=True)
    assert counter.varlen_calls and fallback_stats() == {}
    err = lambda x: 1 - torch.nn.functional.cosine_similarity(x, gold_g, dim=0).item()  # noqa: E731
    assert abs(loss - gold_loss) <= max(3 * abs(noise_loss - gold_loss), 1e-3), (loss, gold_loss)
    assert err(g) <= max(3 * err(noise_g), 1e-5), (err(g), err(noise_g))


@pytest.mark.parametrize("padded", [False, True])
def test_autocast_mixed_dtypes_run_on_kernel(monkeypatch, padded):
    """Under fp16 autocast q / k may be fp32 (rotary) and v fp16: all cast to fp16
    and run on the kernel (batched, or unpadded varlen), fp16 output, gradients."""
    q, k, _ = (t.float().requires_grad_() for t in _qkv(q_len=24, kv_len=24))
    v = _qkv(q_len=24, kv_len=24)[2].requires_grad_()  # fp16
    pad = None
    if padded:
        pad = torch.ones(1, 24, dtype=torch.bool, device="cuda")
        pad[0, :5] = False
    counter = KernelCounter(monkeypatch)
    with torch.autocast("cuda", dtype=torch.float16):
        out, _ = integrations.flash_attn_legacy_forward(_Attn(is_causal=True), q, k, v, pad)
    out.float().sum().backward()
    assert len(counter.calls) + len(counter.varlen_calls) == 1 and fallback_stats() == {}
    assert out.dtype == torch.float16 and q.grad is not None and v.grad is not None


def test_autocast_skips_range_check():
    """fp16 autocast semantics (as SDPA there): values beyond fp16 are cast, not
    rejected (no fallback); the GradScaler handles inf."""
    q, k, v = (t.float() for t in _qkv())
    v[0, 0, 3, 0] = 1e6
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        integrations.flash_attn_legacy_forward(_Attn(is_causal=True), q, k, v, None)
    assert fallback_stats() == {}


def test_fp32_training_without_autocast_still_falls_back():
    q, k, v = (t.float().requires_grad_() for t in _qkv())
    integrations.flash_attn_legacy_forward(_Attn(is_causal=True), q, k, v, None)
    assert fallback_stats() == {"non_fp16_training": 1}


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
    # Every call (prefill and the 2 decode steps of each generate) has head_dim 48.
    n_layers = base.config.num_hidden_layers
    assert fallback_stats() == {"head_dim": 2 * 3 * n_layers}


# ---------------------------------------------------------------------------
# P0.2 — caches whose kv_len is not "all tokens so far" (B1: StaticCache)
# ---------------------------------------------------------------------------

# transformers 5.20: an offloaded static cache offloads only full-attention
# layers and its prefetch raises (ValueError) when there is none, with any backend.
_ALL_SLIDING_OFFLOAD = "transformers: offloaded static cache needs a full-attention layer"


def _all_sliding(arch):
    return "full_attention" not in layer_types(TINY_MODELS[arch]().config)


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
@pytest.mark.parametrize("arch", ["llama", "gemma3", "mistral"])  # gemma3: sliding + full (hybrid)
def test_static_cache_stepwise(monkeypatch, arch, offloading):
    from transformers import StaticCache

    if offloading and _all_sliding(arch):
        pytest.skip(_ALL_SLIDING_OFFLOAD)

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

    # The kernel must run on the multi-token steps of every layer: full-attention
    # layers on K/V sliced to the filled part of the cache (kv == tokens so far),
    # sliding layers (P4.1) on the filled slots of their sliding cache (the last
    # W - 1 tokens plus the new ones), with the window.
    types = layer_types(base.config)
    W = getattr(base.config, "sliding_window", None) or 0  # 0: no sliding layers (llama)
    expected = Counter()
    for i, n in enumerate(_STEPS):
        if n > 1:
            total = sum(_STEPS[:i + 1])
            expected[(n, total)] += types.count("full_attention")
            expected[(n, min(total, W - 1 + n))] += types.count("sliding_attention")
    assert Counter(counter.calls) == +expected
    sliding_calls = sum(n > 1 for n in _STEPS) * types.count("sliding_attention")
    assert [w for w in counter.windows if w != (-1, -1)] == [(W - 1, W - 1)] * sliding_calls
    # single-token steps on the decode kernel, sliding layers with the window
    decode_steps = sum(n == 1 for n in _STEPS)
    assert len(counter.decode_calls) == decode_steps * len(types) and fallback_stats() == {}
    assert [w for w in counter.decode_windows if w != (-1, -1)] == \
        [(W - 1, W - 1)] * decode_steps * types.count("sliding_attention")


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
    """End-to-end generate vs FP32 eager (gold), with the noise floor of FP16
    eager and FP16 SDPA (the worse of the two: FP16 near-ties fork beams on one
    backend and not on another): every step's
    logits (all beams) and the final sequences."""
    if cache_implementation == "offloaded_static" and _all_sliding(arch):
        pytest.skip(_ALL_SLIDING_OFFLOAD)
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

    gold, out = gen("eager", torch.float32), gen("flash_attn_legacy")
    noises = [gen("eager")] + ([gen("sdpa")] if getattr(base, "_supports_sdpa", False) else [])  # gpt-oss: no SDPA
    label = f"{arch}/{cache_implementation}/{mode}"
    # Beam search: an FP16 near-tie in the beam scores can keep a different
    # hypothesis in one beam for one step (a fork), with every other beam
    # unchanged; a wrong attention would affect all beams and all steps. So a
    # single beam may differ at a single step of the whole generation; the final
    # sequences are still checked below.
    forked_steps = []
    for step, (o, g, *ns) in enumerate(zip(out.logits, gold.logits, *(n.logits for n in noises))):
        assert torch.isfinite(g).all(), f"{label}: non-finite reference logits"
        err, err_noise = _beam_rows_error(o, g), max(_beam_rows_error(n, g) for n in ns)
        bound = max(3.0 * err_noise, 1e-3)  # same rule as hf_helpers.assert_within_noise
        if err > bound and mode == "beam" and not forked_steps:
            err = _beam_rows_error(o, g, ignore_worst=1)
            forked_steps.append(step)
        assert err <= bound, f"{label}: step {step} 1-cos={err:.2e} (noise {err_noise:.2e})"
    if all(torch.equal(n.sequences, gold.sequences) for n in noises):
        assert torch.equal(out.sequences, gold.sequences), label


# ---------------------------------------------------------------------------
# P4.1 — sliding window on the kernel
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_tokens", [24, 300])  # both beyond the sliding window (8)
@pytest.mark.parametrize("arch", list(TINY_MODELS))
def test_model_matrix_on_kernel(monkeypatch, arch, n_tokens):
    """P4.4: every model's prefill with every layer on the kernel and no
    fallback, sliding window (Mistral, Gemma 2/3, Phi-3, gpt-oss), softcap
    (Gemma 2) and sinks (gpt-oss) included; sliding layers get window_size=
    (W-1, W-1); logits within the noise of FP32 eager."""
    base = TINY_MODELS[arch](initializer_range=_INIT[arch])
    ids = torch.randint(0, base.config.vocab_size, (1, n_tokens))
    gold = logits(load_impl(base, "eager", torch.float32), ids)
    noise = logits(load_impl(base, "eager"), ids)
    counter = KernelCounter(monkeypatch)
    out = logits(load_impl(base, "flash_attn_legacy"), ids)
    assert len(counter.calls) == base.config.num_hidden_layers and fallback_stats() == {}
    n_sliding = layer_types(base.config).count("sliding_attention")
    expected = [(base.config.sliding_window - 1,) * 2] * n_sliding if n_sliding else []  # Llama: none
    assert [w for w in counter.windows if w != (-1, -1)] == expected
    assert_close_to_gold(out, gold, noise, f"{arch}, {n_tokens} tokens")


def test_sliding_window_mutation():
    """The window must be W keys (HF: i - j < W), not W +- 1: the kernel output
    of a sliding layer matches SDPA with the HF mask and differs from W +- 1."""
    torch.manual_seed(0)
    q, k, v = _qkv(40, 40)
    W = 8
    hf = integrations._causal_mask(40, 40, "cuda", W)
    assert hf.sum(-1).max() == W and bool(hf[10, 3]) and not bool(hf[10, 2])
    out, _ = integrations.flash_attn_legacy_forward(_Attn(True), q, k, v, None, sliding_window=W)
    out = out.transpose(1, 2)
    for w, same in ((W, True), (W - 1, False), (W + 1, False)):
        ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=integrations._causal_mask(40, 40, "cuda", w))
        assert torch.allclose(out, ref, atol=2e-3, rtol=2e-3) == same, w


@pytest.mark.parametrize("q_len,kv_len,causal,padded", [
    (1, 40, True, False), (1, 40, True, True), (5, 40, True, True),
    (40, 40, False, False), (40, 40, False, True)])
def test_sliding_window_fallbacks_apply_window(q_len, kv_len, causal, padded):
    """Decode and padding with a non-causal window (fallbacks), causal padding
    (unpadded onto the varlen kernel): all apply the window, vs SDPA with the
    explicit band mask."""
    torch.manual_seed(0)
    q, k, v = _qkv(q_len, kv_len)
    W = 8
    pad = torch.ones(1, kv_len, dtype=torch.bool, device="cuda")
    if padded:
        pad[0, :3] = False
    off = kv_len - q_len
    rel = torch.arange(kv_len, device="cuda")[None] - torch.arange(q_len, device="cuda")[:, None] - off
    band = (rel <= 0) & (rel > -W) if causal else rel.abs() < W
    keep = band & pad[0]
    keep = keep | ~keep.any(-1, keepdim=True)
    out, _ = integrations.flash_attn_legacy_forward(_Attn(causal), q, k, v, pad if padded else None,
                                                    sliding_window=W)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=keep)
    torch.testing.assert_close(out.transpose(1, 2), ref, atol=2e-3, rtol=2e-3)


def test_builder_sliding_window_tags_window():
    from transformers import masking_utils as mu

    sliding = mu.sliding_window_causal_mask_function(8)
    m = _build(kv_length=25, mask_function=sliding, local_size=8)
    assert m.shape == (2, 25) and getattr(m, integrations._WINDOW_ATTR) == 8
    pad = torch.ones(2, 25, dtype=torch.bool, device="cuda")
    pad[1, :3] = False
    m = _build(kv_length=25, mask_function=sliding, local_size=8, attention_mask=pad)
    assert getattr(m, integrations._WINDOW_ATTR) == 8 and getattr(m, integrations._PADDING_ATTR)
    assert torch.equal(m.bool(), pad)
    # skipping not allowed (packing, overlays): SDPA's 4D mask, window included
    assert _build(kv_length=25, mask_function=sliding, local_size=8, allow_is_causal_skip=False).dim() == 4


def test_builder_chunked_attention_is_not_a_window():
    """local_size also comes with chunked attention (Llama 4): never read as a window."""
    from transformers import masking_utils as mu

    chunked = mu.chunked_causal_mask_function(8, torch.zeros(2, dtype=torch.long, device="cuda"))
    assert not integrations._is_sliding_causal(chunked, 8)
    # chunk 2 matches a window of 2 on any single row: two rows are probed
    assert not integrations._is_sliding_causal(mu.chunked_causal_mask_function(2, torch.zeros(2, dtype=torch.long)), 2)
    assert integrations._is_sliding_causal(mu.sliding_window_causal_mask_function(8), 8)
    assert not integrations._is_sliding_causal(mu.sliding_window_causal_mask_function(9), 8)
    m = _build(kv_length=25, mask_function=chunked, local_size=8)
    assert m.dim() == 4


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


@pytest.mark.parametrize("padded", [False, True])
def test_window_survives_mask_copies(monkeypatch, padded):
    """Layers without a sliding_window kwarg (e.g. Qwen2-MoE) get the window only
    from the builder's mask: a copy that drops the tag (accelerate's .to(device)
    across GPUs) must keep it, read from the values; a plain 0/1 mask has none."""
    from transformers import masking_utils as mu

    q, k, v = _qkv(q_len=40, kv_len=40)
    pad = torch.ones(1, 40, dtype=torch.bool, device="cuda")
    if padded:
        pad[0, :3] = False
    m = _build(batch_size=1, q_length=40, kv_length=40, q_offset=0,
               mask_function=mu.sliding_window_causal_mask_function(8), local_size=8,
               attention_mask=pad if padded else None)
    copy = m.clone()
    assert getattr(copy, integrations._WINDOW_ATTR, None) is None
    assert integrations._mask_window(copy) == 8 and integrations._mask_window(pad.int()) is None
    counter = KernelCounter(monkeypatch)
    out_tagged, _ = integrations.flash_attn_legacy_forward(_Attn(is_causal=True), q, k, v, m)
    out_copy, _ = integrations.flash_attn_legacy_forward(_Attn(is_causal=True), q, k, v, copy)
    assert counter.windows == [(7, 7), (7, 7)] and fallback_stats() == {}
    torch.testing.assert_close(out_tagged, out_copy, atol=0, rtol=0)


def test_tagged_mask_longer_than_keys_raises():
    q, k, v = _qkv(q_len=4, kv_len=8)
    m = _build(q_length=4, q_offset=6)  # tags 10 valid keys, only 8 passed
    with pytest.raises(RuntimeError):
        integrations.flash_attn_legacy_forward(_Attn(is_causal=True), q, k, v, m)


# ---------------------------------------------------------------------------
# P0.3 / P4.2 / P4.3 / P4.4 — softcap and sinks on the kernel; position_bias
# (T5 family) on SDPA, added to its mask
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch", ["gemma2", "gpt_oss"])
def test_model_features_matter(arch):
    """Guard for test_model_matrix_on_kernel: softcap (Gemma 2) and sinks
    (gpt-oss) change the gold logits of these models, or matching them proves
    nothing about the feature."""
    base = TINY_MODELS[arch](initializer_range=_INIT[arch])
    ids = torch.randint(0, base.config.vocab_size, (1, 24))
    gold = logits(load_impl(base, "eager", torch.float32), ids)
    without = {"gemma2": dict(attn_logit_softcapping=None), "gpt_oss": {}}[arch]
    plain = TINY_MODELS[arch](initializer_range=_INIT[arch], **without)
    if arch == "gpt_oss":
        with torch.no_grad():
            for layer in plain.model.layers:
                layer.self_attn.sinks.fill_(-1e4)  # sinks with no mass
    _, cos_plain, _ = compare(logits(load_impl(plain, "eager", torch.float32), ids), gold)
    assert 1.0 - cos_plain > 1e-3, f"{arch}: the feature does not change the gold logits"


def test_gpt_oss_training_sink_gradients(monkeypatch):
    """gpt-oss fine-tuning under fp16 autocast on the kernel: loss and gradients,
    the sinks' own included, vs FP32 eager; noise floor: fp16-autocast eager."""
    base = TINY_MODELS["gpt_oss"](initializer_range=_INIT["gpt_oss"])
    ids = torch.randint(0, base.config.vocab_size, (2, 40), device="cuda")

    def grads(impl, autocast):
        model = load_impl(base, impl, torch.float32).train()
        with torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
            model(input_ids=ids, labels=ids).loss.backward()
        sinks = torch.cat([layer.self_attn.sinks.grad for layer in model.model.layers])
        return sinks, torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])

    gold_s, gold_g = grads("eager", autocast=False)
    noise_s, noise_g = grads("eager", autocast=True)
    counter = KernelCounter(monkeypatch)
    sinks, g = grads("flash_attn_legacy", autocast=True)
    assert len(counter.calls) == base.config.num_hidden_layers and fallback_stats() == {}
    assert gold_s.abs().min() > 0  # every sink is trained
    for got, gold, noise in ((sinks, gold_s, noise_s), (g, gold_g, noise_g)):
        err = lambda x: 1 - torch.nn.functional.cosine_similarity(x, gold, dim=0).item()  # noqa: E731
        assert err(got) <= max(3 * err(noise), 1e-5), (err(got), err(noise))


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


@pytest.mark.parametrize("kwarg,reason", [("softcap", None), ("s_aux", None),
                                          ("position_bias", "position_bias")])
@pytest.mark.parametrize("q_len", [16, 1], ids=["prefill", "decode"])
def test_feature_kwargs_routed_and_applied(kwarg, reason, q_len):
    """softcap / sinks: kernel on prefill and on decode (decode kernel);
    position_bias: SDPA. Softcap 2.0 with scale 0.5 saturates."""
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
    assert fallback_stats() == ({reason: 1} if reason else {})
    torch.testing.assert_close(out.transpose(1, 2).float(), ref, atol=2e-3, rtol=2e-3)


def _no_eager(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("position_bias must run on SDPA, not on the eager fallback")
    monkeypatch.setattr(integrations, "_eager_attention", fail)


@pytest.mark.parametrize("bias_kind", ["4d", "3d_longer", "minus_inf"])
@pytest.mark.parametrize("mask_kind", ["causal", "padding", "bool4d", "additive4d", "none"])
def test_position_bias_on_sdpa(monkeypatch, mask_kind, bias_kind):
    """position_bias added to SDPA's mask with every mask form the integration
    passes on (causal flag, 2D padding, 4D bool / additive at the fp16 minimum,
    none) and bias forms (4D, 3D wider than the keys, -inf entries); output and
    the gradients of q / k / v / bias vs the fp32 formula, on the rows that see
    a key (left-padded rows: don't care, as in HF)."""
    torch.manual_seed(0)
    B, n = 2, 24
    q, k, v = (torch.randn(B, 4, n, 64, device="cuda", dtype=torch.float16, requires_grad=True)
               for _ in range(3))
    bias = torch.randn(1, 4, n, n + (3 if bias_kind == "3d_longer" else 0), device="cuda")
    if bias_kind == "minus_inf":
        bias[..., 3] = float("-inf")  # a key no row may see, in every row
    bias = (bias[0] if bias_kind == "3d_longer" else bias).half().requires_grad_(True)
    causal = integrations._causal_mask(n, n, "cuda")
    pad = torch.ones(B, n, dtype=torch.bool, device="cuda")
    pad[1, :5] = False
    lowest = torch.finfo(torch.float16).min
    mask, ref_mask, is_causal = {
        "causal": (None, causal, True),
        "padding": (pad, pad[:, None, None, :] & causal, True),
        "bool4d": (causal[None, None], causal, False),
        "additive4d": (torch.where(causal, 0.0, lowest).half()[None, None], causal, False),
        "none": (None, None, False),
    }[mask_kind]
    _no_eager(monkeypatch)
    out, _ = integrations.flash_attn_legacy_forward(_Attn(is_causal=is_causal), q, k, v, mask,
                                                    scaling=0.125, position_bias=bias)
    assert fallback_stats() == {"position_bias": 1}
    assert torch.isfinite(out).all()
    seen = (pad if mask_kind == "padding" else torch.ones_like(pad))[:, :, None, None]  # [B, n, 1, 1]
    go = torch.randn_like(out) * seen  # left-padded rows carry no gradient
    out.backward(go)
    got = [t.grad for t in (q, k, v, bias)]
    for t in (q, k, v, bias):
        t.grad = None
    if ref_mask is not None:  # rows with no key attend everything, as the fallback (no NaN)
        ref_mask = ref_mask | ~ref_mask.any(-1, keepdim=True)
    ref = _reference(q, k, v, 0.125, ref_mask, bias=bias.float()[..., :n]).transpose(1, 2)
    ref.backward(go.float())
    torch.testing.assert_close(torch.where(seen, out.float(), 0.0), torch.where(seen, ref.detach(), 0.0),
                               atol=2e-3, rtol=2e-3)
    for g, t in zip(got, (q, k, v, bias)):
        torch.testing.assert_close(g.float(), t.grad.float(), atol=1e-2, rtol=1e-2)


def _tiny_t5():
    from transformers import T5Config, T5ForConditionalGeneration

    torch.manual_seed(0)
    cfg = T5Config(vocab_size=256, d_model=256, d_kv=64, num_heads=4, num_layers=2, d_ff=512,
                   initializer_factor=0.5, decoder_start_token_id=0, pad_token_id=0)
    src = torch.randint(1, 256, (2, 30), device="cuda")
    src_mask = torch.ones_like(src)
    src_mask[1, 20:] = 0  # padded encoder row
    return T5ForConditionalGeneration(cfg).eval(), src, src_mask


def test_t5_runs_on_sdpa_with_position_bias(monkeypatch):
    """T5 (relative position bias in every layer: encoder, decoder, cross):
    SDPA, never eager; logits within the noise of FP32 eager."""
    base, src, src_mask = _tiny_t5()
    cfg = base.config
    tgt = torch.randint(0, 256, (2, 12), device="cuda")

    def run(impl, dtype=torch.float16):
        with torch.no_grad():
            return load_impl(base, impl, dtype)(input_ids=src, attention_mask=src_mask,
                                                decoder_input_ids=tgt).logits.float()

    gold, noise = run("eager", torch.float32), run("eager")
    _no_eager(monkeypatch)
    out = run("flash_attn_legacy")
    assert fallback_stats() == {"position_bias": 3 * cfg.num_layers}  # self, decoder self, cross
    assert_close_to_gold(out.flatten(0, 1), gold.flatten(0, 1), noise.flatten(0, 1), "t5")


@pytest.mark.parametrize("cache_implementation", ["dynamic", "static"])
def test_t5_generate_with_position_bias(monkeypatch, cache_implementation):
    """T5 generate: decode steps (q_len 1, the bias's last row) and a static cache
    (keys trimmed to the valid prefix, the bias sliced to them) on SDPA; every
    step's logits within the noise of FP32 eager."""
    base, src, src_mask = _tiny_t5()
    kw = dict(attention_mask=src_mask, max_new_tokens=8, min_new_tokens=8, do_sample=False,
              return_dict_in_generate=True, output_logits=True,
              cache_implementation=cache_implementation, disable_compile=True)

    def gen(impl, dtype=torch.float16):
        return torch.stack(load_impl(base, impl, dtype).generate(src, **kw).logits).float()

    gold, noise = gen("eager", torch.float32), gen("eager")
    _no_eager(monkeypatch)
    out = gen("flash_attn_legacy")
    assert set(fallback_stats()) == {"position_bias"}
    assert_close_to_gold(out.flatten(0, 1), gold.flatten(0, 1), noise.flatten(0, 1),
                         f"t5 generate ({cache_implementation})")


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


@pytest.mark.parametrize("kwarg", ["softcap", "s_aux"])
def test_cu_seq_lens_with_features_on_varlen_kernel(monkeypatch, kwarg):
    torch.manual_seed(0)
    q, k, v = _qkv(q_len=_CU[-1], kv_len=_CU[-1])
    cu = torch.tensor(_CU, dtype=torch.int32, device="cuda")
    value = {"softcap": 2.0, "s_aux": torch.randn(4, device="cuda")}[kwarg]
    counter = KernelCounter(monkeypatch)
    out, _ = integrations.flash_attn_legacy_forward(
        _Attn(is_causal=True), q, k, v, None, scaling=0.5,
        cu_seq_lens_q=cu, cu_seq_lens_k=cu, **{kwarg: value})
    outs = []
    for a, b in zip(_CU[:-1], _CU[1:]):
        outs.append(_reference(q[:, :, a:b], k[:, :, a:b], v[:, :, a:b], 0.5,
                               integrations._causal_mask(b - a, b - a, "cuda"),
                               softcap=value if kwarg == "softcap" else None,
                               sinks=value if kwarg == "s_aux" else None))
    assert counter.varlen_calls == [(_CU[-1], len(_CU) - 1)] and fallback_stats() == {}
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
@pytest.mark.parametrize("arch", ["llama", "gemma3", "mistral"])  # sliding layers: gemma3, mistral
def test_packed_position_ids_forward(monkeypatch, arch, use_cache):
    """B4: with a cache transformers builds no packing mask (and only the window
    on sliding layers); our forward must still keep the segments apart (varlen
    kernel, with the window on sliding layers). Without a cache it builds a block
    mask (-> SDPA fallback). All must equal each sequence alone."""
    base = TINY_MODELS[arch](initializer_range=_INIT[arch])
    lengths = [30, 17, 45]
    seqs, ids, position_ids = _packed_batch(base, lengths)
    gold = torch.cat([logits(load_impl(base, "eager", torch.float32), s_[None]) for s_ in seqs])
    noise = torch.cat([logits(load_impl(base, "eager"), s_[None]) for s_ in seqs])
    counter = KernelCounter(monkeypatch)
    out = logits(load_impl(base, "flash_attn_legacy"), ids,
                 position_ids=position_ids.cuda(), use_cache=use_cache)
    n_layers = base.config.num_hidden_layers
    if use_cache:  # segments longer than the window (8): a missing window_size shows
        W = getattr(base.config, "sliding_window", None) or 0  # 0: no sliding layers (llama)
        assert counter.varlen_calls == [(sum(lengths), 3)] * n_layers and fallback_stats() == {}
        n_sliding = layer_types(base.config).count("sliding_attention")
        assert [w for w in counter.windows if w != (-1, -1)] == [(W - 1, W - 1)] * n_sliding
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


_GRIDS = [[1, 8, 8], [1, 12, 8], [1, 4, 4], [1, 8, 4], [1, 4, 12], [1, 8, 8], [1, 4, 4], [1, 12, 12]]


@pytest.mark.parametrize("n_images", [1, 3, 8])
@pytest.mark.parametrize("kind,head_dim,expect_varlen", [
    ("qwen2_vl", 64, True), ("qwen2_vl", 80, True), ("qwen2_vl", 48, False), ("qwen2_5_vl", 64, True)])
def test_vision_tower_multiple_images(monkeypatch, kind, head_dim, expect_varlen, n_images):
    """VLM vision encoders: patches of several images attend within their image
    (and window, Qwen2.5-VL) via cu_seq_lens; 1, 3 and 8 images."""
    tower = tiny_vision_tower(kind, head_dim=head_dim)
    cfg = tower.config
    grid = torch.tensor(_GRIDS[:n_images])
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
    assert_close_to_gold(out, gold, noise, f"{kind} d={head_dim} x{n_images}")


@pytest.mark.parametrize("n_images", [1, 3, 8])
@pytest.mark.parametrize("kind,head_dim", [("siglip", 72), ("clip", 64), ("pixtral", 64)])
def test_image_encoders_on_kernel(monkeypatch, kind, head_dim, n_images):
    """P3.3: SigLIP (so400m-style d=72), CLIP, Pixtral (images of different sizes in
    one sequence) with 1, 3, 8 images: every attention layer on the kernel, output
    within the noise of FP32 eager."""
    from .hf_helpers import tiny_image_encoder

    enc = tiny_image_encoder(kind, head_dim=head_dim)
    g = torch.Generator().manual_seed(0)
    if kind == "pixtral":
        sizes = [(64, 64), (48, 32), (32, 64), (16, 16), (64, 48), (32, 32), (48, 48), (16, 32)][:n_images]
        pixels = [torch.randn(3, h, w, generator=g) for h, w in sizes]
        sizes_t = torch.tensor(sizes)

        def run(impl, dtype=torch.float16):
            m = load_impl(enc, impl, dtype)
            x = torch.stack([torch.nn.functional.pad(p, (0, 64 - p.shape[2], 0, 64 - p.shape[1])) for p in pixels])
            with torch.no_grad():
                return m(x.to("cuda", dtype), image_sizes=sizes_t.cuda()).last_hidden_state.float()
    else:
        x = torch.randn(n_images, 3, 64, 64, generator=g)

        def run(impl, dtype=torch.float16):
            with torch.no_grad():
                return load_impl(enc, impl, dtype)(pixel_values=x.to("cuda", dtype)).last_hidden_state.float()

    gold, noise = run("eager", torch.float32), run("eager")
    counter = KernelCounter(monkeypatch)
    out = run("flash_attn_legacy")
    n_layers = enc.config.num_hidden_layers
    assert len(counter.calls) + len(counter.varlen_calls) == n_layers and fallback_stats() == {}, \
        (counter.calls, counter.varlen_calls, fallback_stats())
    # random encoders, 16-64 positions per image: one FP16 near-tie flip of the
    # hidden-state argmax is noise (seen once on P100, SigLIP x1: 63/64)
    flat = lambda t: t.flatten(0, -2)  # noqa: E731
    assert_close_to_gold(flat(out), flat(gold), flat(noise), f"{kind} x{n_images}", near_tie_slack=True)


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
    # every layer, sliding ones included (P4.1)
    assert len(counter.calls) == base.config.num_hidden_layers and fallback_stats() == {}
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
def test_beyond_fp16_on_kernel(dtype, where):
    """bf16 / fp32 values beyond the fp16 range run on the kernel (scaled into fp16
    on the device), within fp16 accuracy of SDPA in the original dtype."""
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
    assert fallback_stats() == {}
    assert out.dtype == dtype and torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), ref.float(), atol=3e-2 * ref.float().abs().max().item(), rtol=3e-2)


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


@pytest.mark.skipif(not hasattr(torch, "compiler"), reason="torch.compile required")
@pytest.mark.parametrize("cache_type", ["static", "dynamic"])
def test_compiled_decode(monkeypatch, cache_type):
    """P6.5: a compiled decode step with left padding runs the decode kernel with
    the step's mask (static: HF's 4D mask for compileable caches; dynamic: our
    builder's compiled branch), matches eager, and breaks the graph no more often
    than SDPA (none for either)."""
    import copy
    from torch._dynamo.utils import counters
    from transformers import DynamicCache, StaticCache
    base = tiny_llama()
    ids = torch.randint(0, base.config.vocab_size, (2, 10), device="cuda")
    mask = torch.ones_like(ids)
    mask[1, :3] = 0
    breaks, logits = {}, {}
    for impl in ("sdpa", "flash_attn_legacy"):
        model = load_impl(base, impl)
        cache = StaticCache(config=model.config, max_cache_len=32) if cache_type == "static" else DynamicCache()
        with torch.no_grad():
            nxt = model(input_ids=ids, attention_mask=mask, past_key_values=cache).logits[:, -1:].argmax(-1)
            step_mask = torch.cat([mask, torch.ones_like(nxt)], 1)
            pos = torch.tensor([ids.shape[1]], device="cuda")

            def step(c):
                return model(input_ids=nxt, attention_mask=step_mask, past_key_values=c, cache_position=pos).logits

            eager = step(copy.deepcopy(cache))
            counter = KernelCounter(monkeypatch)
            torch._dynamo.reset()
            counters.clear()
            try:
                compiled = torch.compile(step, backend="aot_eager")(copy.deepcopy(cache))
                breaks[impl] = sum(counters["graph_break"].values())
            finally:
                torch._dynamo.reset()
                counters.clear()
        torch.testing.assert_close(compiled.float(), eager.float(), atol=2e-2, rtol=2e-2)
        logits[impl] = eager.float()
        if impl == "flash_attn_legacy":
            assert len(counter.decode_calls) == base.config.num_hidden_layers and fallback_stats() == {}
    assert breaks["flash_attn_legacy"] <= breaks["sdpa"], breaks
    torch.testing.assert_close(logits["flash_attn_legacy"], logits["sdpa"], atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not hasattr(torch, "compiler"), reason="torch.compile required")
def test_compiled_decode_steps_do_not_recompile():
    """Consecutive compiled decode steps on a DynamicCache (kv_len grows every
    step) recompile no more often than SDPA: no per-step int reaches the ops."""
    from torch._dynamo.utils import counters
    base = tiny_llama()
    ids = torch.randint(0, base.config.vocab_size, (1, 10), device="cuda")
    graphs = {}
    for impl in ("sdpa", "flash_attn_legacy"):
        model = load_impl(base, impl)
        torch._dynamo.reset()
        counters.clear()
        try:
            step = torch.compile(lambda **kw: model(**kw), backend="aot_eager")
            with torch.no_grad():
                out = model(input_ids=ids, use_cache=True)
                cache, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
                for _ in range(5):
                    out = step(input_ids=nxt, past_key_values=cache, use_cache=True)
                    nxt = out.logits[:, -1:].argmax(-1)
            graphs[impl] = counters["stats"]["unique_graphs"]
        finally:
            torch._dynamo.reset()
            counters.clear()
    assert graphs["flash_attn_legacy"] <= graphs["sdpa"], graphs
