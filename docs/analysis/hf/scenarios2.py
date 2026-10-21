import sys
exec(open("/workspace/analysis/hf/probe.py").read())
from transformers import Gemma2ForCausalLM, MistralForCausalLM, ModernBertModel, GptOssForCausalLM, Qwen3ForCausalLM
ids = torch.randint(0, 100, (2, 20))
with torch.no_grad():
    m, r = build("gemma2", Gemma2ForCausalLM, vocab_size=100, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                 num_attention_heads=2, num_key_value_heads=1, head_dim=64, query_pre_attn_scalar=64, sliding_window=64,
                 attn_logit_softcapping=0.5, final_logit_softcapping=None)
    summarize("gemma2 softcap=0.5, seq<window, no pad", m(ids, use_cache=False).logits, r(ids, use_cache=False).logits)
    m, r = build("mistral", MistralForCausalLM, vocab_size=100, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                 num_attention_heads=2, num_key_value_heads=1, sliding_window=8)
    summarize("mistral sliding_window=8, seq=20", m(ids, use_cache=False).logits, r(ids, use_cache=False).logits)
    g1 = m.generate(ids[:, :6], max_new_tokens=10, do_sample=False, pad_token_id=0); LOG.clear()
    g2 = r.generate(ids[:, :6], max_new_tokens=10, do_sample=False, pad_token_id=0)
    print("### mistral generate past window (DynamicCache) token match:", bool((g1 == g2).all())); LOG.clear()
    m, r = build("qwen3", Qwen3ForCausalLM, vocab_size=100, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                 num_attention_heads=2, num_key_value_heads=1, head_dim=64, use_sliding_window=True, sliding_window=8, max_window_layers=0,
                 layer_types=["sliding_attention","full_attention"])
    summarize("qwen3 sliding layer (window 8), seq=20", m(ids, use_cache=False).logits, r(ids, use_cache=False).logits)
    m, r = build("modernbert", ModernBertModel, vocab_size=100, hidden_size=128, intermediate_size=256, num_hidden_layers=3,
                 num_attention_heads=2, local_attention=8, global_attn_every_n_layers=3, pad_token_id=0)
    am = torch.ones(2, 20, dtype=torch.long); am[1, 15:] = 0
    summarize("modernbert local_attention=8, no pad", m(ids, attention_mask=torch.ones_like(am)).last_hidden_state, r(ids, attention_mask=torch.ones_like(am)).last_hidden_state)
    summarize("modernbert right padding", m(ids, attention_mask=am).last_hidden_state, r(ids, attention_mask=am).last_hidden_state, am.bool())
    m, r = build("modernbert", ModernBertModel, vocab_size=100, hidden_size=128, intermediate_size=256, num_hidden_layers=3,
                 num_attention_heads=2, local_attention=64, global_attn_every_n_layers=3, pad_token_id=0)
    summarize("modernbert local_attention=64 (>seq), no pad", m(ids).last_hidden_state, r(ids).last_hidden_state)
    try:
        m, r = build("gpt_oss", GptOssForCausalLM, vocab_size=100, hidden_size=128, intermediate_size=128, num_hidden_layers=2,
                     num_attention_heads=2, num_key_value_heads=1, head_dim=64, num_local_experts=2, num_experts_per_tok=1, sliding_window=64)
        for mod in list(m.modules()) + list(r.modules()):
            if hasattr(mod, "sinks"): torch.nn.init.normal_(mod.sinks, std=2.0)
        r.load_state_dict(m.state_dict())
        summarize("gpt_oss sinks, no pad", m(ids, use_cache=False).logits, r(ids, use_cache=False).logits)
    except Exception as e:
        print("gpt_oss failed:", repr(e)[:300])
