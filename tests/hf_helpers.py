"""
Shared helpers for the HuggingFace integration tests (one implementation for
all HF test files: kernel-call counting, backend switching, logits comparison
against the FP16 noise floor, tiny random models).
"""

import torch


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


def set_impl(model, impl):
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation(impl)
    else:  # older transformers
        model.config._attn_implementation = impl
    assert model.config._attn_implementation == impl, (
        f"requested {impl}, model uses {model.config._attn_implementation}")


def compare(a, b):
    """Return (max_abs_diff, mean_cosine, top1_agreement) over positions."""
    diff = (a - b).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()
    top1 = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    return diff, cos, top1


def assert_within_noise(flash_cos, eager_cos, top1, label, top1_noise=None):
    """FP16 logits differ between *any* two attention backends (eager vs sdpa
    included). Judge flash against that noise floor, measured on the same input,
    instead of a fixed threshold: its error (1 - cos) may be at most 3x the noise
    error w.r.t. the same reference. The 1e-6 floor sits just above the resolution
    of 1 - cos computed in FP32 (~1e-7), relevant only for tiny models."""
    err_flash, err_eager = 1.0 - flash_cos, 1.0 - eager_cos
    print(f"  {label}: 1-cos flash={err_flash:.2e} eager(noise floor)={err_eager:.2e} top1={top1:.4f}")
    assert err_flash <= max(3.0 * err_eager, 1e-6), f"{label}: flash error above FP16 noise floor"
    # top-1 agreement: relative to the noise reference when known (near-ties in
    # random models), else absolute.
    assert top1 >= (top1_noise - 0.01 if top1_noise is not None else 0.99), f"{label}: top1 {top1}"


def assert_close_to_gold(out, gold, noise, label):
    """`gold` is an FP32 eager reference; `noise` is FP16 eager on the same input."""
    _, cos, top1 = compare(out, gold)
    _, cos_noise, top1_noise = compare(noise, gold)
    assert_within_noise(cos, cos_noise, top1, label, top1_noise=top1_noise)


@torch.no_grad()
def logits(model, input_ids, impl=None, **kw):
    """FP32 logits of the first batch row; optionally switch attention impl first."""
    if impl is not None:
        set_impl(model, impl)
    return model(input_ids=input_ids.to(model.device), **kw).logits.float()[0]


def tiny_llama(head_dim=64, n_heads=4, n_kv_heads=2, n_layers=2, vocab=512, seed=0, **extra):
    """Random-weight Llama with the given attention geometry (no download)."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(
        hidden_size=head_dim * n_heads, num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads, head_dim=head_dim,
        intermediate_size=2 * head_dim * n_heads, num_hidden_layers=n_layers,
        vocab_size=vocab, max_position_embeddings=4096, **extra,
    )
    return LlamaForCausalLM(cfg).eval()


def tiny_gemma3(head_dim=64, n_heads=4, n_kv_heads=2, n_layers=4, sliding_window=8, vocab=512,
                seed=0, **extra):
    """Random-weight Gemma 3 text model alternating sliding / full attention layers."""
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    torch.manual_seed(seed)
    cfg = Gemma3TextConfig(
        hidden_size=head_dim * n_heads, num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads, head_dim=head_dim, query_pre_attn_scalar=head_dim,
        intermediate_size=2 * head_dim * n_heads, num_hidden_layers=n_layers,
        vocab_size=vocab, max_position_embeddings=4096, sliding_window=sliding_window,
        layer_types=["sliding_attention" if i % 2 == 0 else "full_attention" for i in range(n_layers)],
        **extra,
    )
    return Gemma3ForCausalLM(cfg).eval()


TINY_MODELS = {"llama": tiny_llama, "gemma3": tiny_gemma3}


def load_impl(model, impl, dtype=torch.float16):
    """Copy of `model` on CUDA with the given attention implementation and dtype."""
    import copy

    m = copy.deepcopy(model).to(device="cuda", dtype=dtype)
    set_impl(m, impl)
    return m
