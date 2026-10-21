"""
Shared helpers for the HuggingFace integration tests (one implementation for
all HF test files: kernel-call counting, backend switching, logits comparison
against the FP16 noise floor, tiny random models).
"""

import torch


class KernelCounter:
    """Counts calls to our kernels made through the HF integration:
    `calls` gets (q_len, kv_len) per batched call, `varlen_calls` gets
    (total_tokens, n_segments) per varlen call."""

    def __init__(self, monkeypatch):
        import flash_attn_legacy.flash_attn as fa
        self.calls, self.varlen_calls = [], []
        batched, varlen = fa.flash_attention, fa.flash_attn_varlen_func

        def wrapped(q, k, v, *args, **kwargs):
            self.calls.append((q.shape[2], k.shape[2]))
            return batched(q, k, v, *args, **kwargs)

        def wrapped_varlen(q, k, v, cu_q, *args, **kwargs):
            self.varlen_calls.append((q.shape[0], cu_q.numel() - 1))
            return varlen(q, k, v, cu_q, *args, **kwargs)

        monkeypatch.setattr(fa, "flash_attention", wrapped)
        monkeypatch.setattr(fa, "flash_attn_varlen_func", wrapped_varlen)

    def reset(self):
        self.calls.clear()
        self.varlen_calls.clear()


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


def assert_within_noise(flash_cos, eager_cos, top1, label, top1_noise=None, near_tie_positions=None):
    """FP16 logits differ between *any* two attention backends (eager vs sdpa
    included). Judge flash against that noise floor, measured on the same input,
    instead of a fixed threshold: its error (1 - cos) may be at most 3x the noise
    error w.r.t. the same reference. The 1e-6 floor sits just above the resolution
    of 1 - cos computed in FP32 (~1e-7), relevant only for tiny models."""
    err_flash, err_eager = 1.0 - flash_cos, 1.0 - eager_cos
    print(f"  {label}: 1-cos flash={err_flash:.2e} eager(noise floor)={err_eager:.2e} top1={top1:.4f}")
    assert err_flash <= max(3.0 * err_eager, 1e-6), f"{label}: flash error above FP16 noise floor"
    # top-1 agreement: relative to the noise reference when known, else absolute.
    # Slack 1%; with near_tie_positions=n (opt-in, short sequences of random
    # models with FP16 near-ties) at least 1.5 positions: one extra near-tie flip
    # must not fail the test, two must.
    slack = max(0.01, 1.5 / near_tie_positions) if near_tie_positions else 0.01
    assert top1 >= (top1_noise - slack if top1_noise is not None else 0.99), f"{label}: top1 {top1}"


def assert_close_to_gold(out, gold, noise, label, near_tie_slack=False):
    """`gold` is an FP32 eager reference; `noise` is FP16 eager on the same input."""
    _, cos, top1 = compare(out, gold)
    _, cos_noise, top1_noise = compare(noise, gold)
    assert_within_noise(cos, cos_noise, top1, label, top1_noise=top1_noise,
                        near_tie_positions=out.shape[0] if near_tie_slack else None)


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


def _alternating(n_layers):
    return ["sliding_attention" if i % 2 == 0 else "full_attention" for i in range(n_layers)]


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
        layer_types=_alternating(n_layers),
        **extra,
    )
    return Gemma3ForCausalLM(cfg).eval()


def tiny_gemma2(head_dim=64, n_heads=4, n_kv_heads=2, n_layers=4, sliding_window=8, vocab=512,
                seed=0, **extra):
    """Random-weight Gemma 2: attention logit softcapping + sliding window layers.
    Softcap 1.0 (real models: 50) so that it actually changes these small logits."""
    from transformers import Gemma2Config, Gemma2ForCausalLM

    torch.manual_seed(seed)
    cfg = Gemma2Config(
        hidden_size=head_dim * n_heads, num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads, head_dim=head_dim, query_pre_attn_scalar=head_dim,
        intermediate_size=2 * head_dim * n_heads, num_hidden_layers=n_layers,
        vocab_size=vocab, max_position_embeddings=4096, sliding_window=sliding_window,
        layer_types=_alternating(n_layers), **{"attn_logit_softcapping": 1.0, **extra},
    )
    return Gemma2ForCausalLM(cfg).eval()


def tiny_gpt_oss(head_dim=64, n_heads=4, n_kv_heads=2, n_layers=4, sliding_window=8, vocab=512,
                 seed=0, **extra):
    """Random-weight gpt-oss: attention sinks (s_aux), sliding window layers, small MoE."""
    from transformers import GptOssConfig, GptOssForCausalLM

    torch.manual_seed(seed)
    cfg = GptOssConfig(
        hidden_size=head_dim * n_heads, num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads, head_dim=head_dim,
        intermediate_size=2 * head_dim * n_heads, num_hidden_layers=n_layers,
        num_local_experts=2, num_experts_per_tok=1, vocab_size=vocab,
        max_position_embeddings=4096, sliding_window=sliding_window,
        layer_types=_alternating(n_layers), **extra,
    )
    model = GptOssForCausalLM(cfg).eval()
    with torch.no_grad():  # sinks are zero-initialised: make them matter
        for layer in model.model.layers:
            layer.self_attn.sinks.normal_(0.0, 2.0)
    return model


TINY_MODELS = {"llama": tiny_llama, "gemma3": tiny_gemma3, "gemma2": tiny_gemma2,
               "gpt_oss": tiny_gpt_oss}


def load_impl(model, impl, dtype=torch.float16):
    """Copy of `model` on CUDA with the given attention implementation and dtype."""
    import copy

    m = copy.deepcopy(model).to(device="cuda", dtype=dtype)
    set_impl(m, impl)
    return m


def tiny_vision_tower(kind="qwen2_vl", head_dim=64, n_heads=2, depth=2, seed=0):
    """Random-weight VLM vision encoder (attention over patches of several images
    via cu_seq_lens). qwen2_5_vl adds window attention on all but the last block."""
    torch.manual_seed(seed)
    if kind == "qwen2_vl":
        from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig
        from transformers.models.qwen2_vl.modeling_qwen2_vl import (
            Qwen2VisionTransformerPretrainedModel as Tower)
        cfg = Qwen2VLVisionConfig(depth=depth, embed_dim=head_dim * n_heads, num_heads=n_heads,
                                  hidden_size=64, mlp_ratio=2)
    else:
        from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VisionTransformerPretrainedModel as Tower)
        cfg = Qwen2_5_VLVisionConfig(depth=depth, hidden_size=head_dim * n_heads, num_heads=n_heads,
                                     intermediate_size=128, out_hidden_size=64, window_size=56,
                                     fullatt_block_indexes=[depth - 1])
    return Tower(cfg).eval()
