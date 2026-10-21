import sys
exec(open("/workspace/analysis/hf/probe.py").read())
from transformers import Gemma2ForCausalLM
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
ids = torch.randint(0, 100, (2, 20))
with torch.no_grad():
    m, r = build("gemma2", Gemma2ForCausalLM, vocab_size=100, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                 num_attention_heads=2, num_key_value_heads=1, head_dim=64, query_pre_attn_scalar=64, sliding_window=64,
                 attn_logit_softcapping=2.0, final_logit_softcapping=None)
    for n, p in m.named_parameters():
        if "q_proj" in n or "k_proj" in n: p.mul_(8)
    r.load_state_dict(m.state_dict())
    summarize("gemma2 softcap=2.0 (q/k weights x8), no pad", m(ids, use_cache=False).logits, r(ids, use_cache=False).logits)
    m.set_attn_implementation("sdpa"); LOG.clear()
    summarize("  HF sdpa on the same gemma2 (reference point)", m(ids, use_cache=False).logits, r(ids, use_cache=False).logits)
    vc = dict(depth=2, hidden_size=64, intermediate_size=128, num_heads=2, out_hidden_size=64, patch_size=14, spatial_merge_size=2,
              temporal_patch_size=2, window_size=56, fullatt_block_indexes=[1], in_channels=3)
    for name in (NAME,):
        c = Qwen2_5_VLVisionConfig(**vc); c._attn_implementation = "eager"
        ref = Qwen2_5_VisionTransformerPretrainedModel(c).eval()
        c2 = Qwen2_5_VLVisionConfig(**vc); c2._attn_implementation = NAME
        mm = Qwen2_5_VisionTransformerPretrainedModel(c2).eval(); mm.load_state_dict(ref.state_dict())
        grid = torch.tensor([[1, 8, 8], [1, 4, 4]])
        n = int(grid.prod(-1).sum())
        px = torch.randn(n, 3 * 2 * 14 * 14)
        o1 = mm(px, grid_thw=grid); o2 = ref(px, grid_thw=grid)
        o1 = getattr(o1, "last_hidden_state", o1); o2 = getattr(o2, "last_hidden_state", o2)
        if isinstance(o1, tuple): o1, o2 = o1[0], o2[0]
        summarize(f"qwen2.5-vl vision (2 images, window attn) name={NAME}", o1, o2)
