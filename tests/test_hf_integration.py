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
    KernelCounter, assert_close_to_gold, load_impl, logits, tiny_llama,
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
