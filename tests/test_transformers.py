"""
Monkey-patch integration (examples/hf_llama_patch.py, for transformers < 5.0)
on tiny random-weight Llama models: no downloads required.

Reference ("gold") is the same weights in FP32 with eager attention; FP16 eager
on the same input gives the noise floor (shared helpers in hf_helpers.py).

Run:  pytest tests/test_transformers.py -v
Requires: pip install -e ".[transformers,test]"
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

transformers = pytest.importorskip("transformers")

from .hf_helpers import assert_close_to_gold, load_impl, logits, tiny_llama  # noqa: E402


def _patched(base):
    from examples.hf_llama_patch import patch_llama_attention

    model = load_impl(base, "eager")
    patch_llama_attention(model)
    return model


@pytest.fixture
def base():
    # head_dim 64, GQA 8 query heads / 2 KV heads
    return tiny_llama(head_dim=64, n_heads=8, n_kv_heads=2, vocab=256, initializer_range=0.5)


@pytest.mark.parametrize("seq_len", [1, 16, 64, 256])
def test_patched_forward_vs_gold(base, seq_len):
    ids = torch.randint(0, base.config.vocab_size, (1, seq_len))
    gold = logits(load_impl(base, "eager", torch.float32), ids)
    noise = logits(load_impl(base, "eager"), ids)
    out = logits(_patched(base), ids)
    assert_close_to_gold(out, gold, noise, f"patched N={seq_len}")


def test_patched_generate_matches_eager(base):
    ids = torch.randint(0, base.config.vocab_size, (1, 8), device="cuda")
    kw = dict(max_new_tokens=16, min_new_tokens=16, do_sample=False)
    ref = load_impl(base, "eager").generate(ids, **kw)
    out = _patched(base).generate(ids, **kw)
    assert torch.equal(out, ref)


def test_patched_backward_vs_gold(base):
    ids = torch.randint(0, base.config.vocab_size, (1, 32), device="cuda")

    def grad(model):
        model.train()
        model(ids, labels=ids).loss.backward()
        return model.model.layers[0].self_attn.q_proj.weight.grad.float().flatten()

    gold, noise, out = grad(load_impl(base, "eager", torch.float32)), grad(load_impl(base, "eager")), grad(_patched(base))
    err = lambda g: 1 - torch.nn.functional.cosine_similarity(g, gold, dim=0).item()  # noqa: E731
    assert err(out) <= max(3 * err(noise), 1e-5), (err(out), err(noise))
