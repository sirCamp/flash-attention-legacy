import sys; sys.argv = sys.argv[:2]
exec(open("/workspace/analysis/hf/probe.py").read())
from transformers import LlamaForCausalLM, DynamicCache, StaticCache
T = dict(vocab_size=100, hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=2,
         num_key_value_heads=1, max_position_embeddings=256)
m, r = build("llama", LlamaForCausalLM, **T)
ids = torch.randint(0, 100, (2, 12))
with torch.no_grad():
    # 1 plain
    summarize("llama no padding, no cache", m(ids, use_cache=False).logits, r(ids, use_cache=False).logits)
    # 2 left padding
    am = torch.ones(2, 12, dtype=torch.long); am[0, :4] = 0
    summarize("llama left padding", m(ids, attention_mask=am, use_cache=False).logits, r(ids, attention_mask=am, use_cache=False).logits, am.bool())
    # 3 packed via position_ids, no mask (DataCollatorWithFlattening style)
    pid = torch.tensor([[0,1,2,3,4,0,1,2,3,0,1,2]]); one = ids[:1]
    summarize("llama packed position_ids, no attention_mask", m(one, position_ids=pid, use_cache=False).logits, r(one, position_ids=pid, use_cache=False).logits)
    # 4 packed position_ids + all-ones attention_mask
    ones = torch.ones(1, 12, dtype=torch.long)
    # reference = per-sequence eager
    refp = torch.cat([r(one[:, s:e], use_cache=False).logits for s, e in [(0,5),(5,9),(9,12)]], 1)
    summarize("llama packed position_ids + all-ones mask (vs per-seq eager)", m(one, attention_mask=ones, position_ids=pid, use_cache=False).logits, refp)
    LOG.clear()
    summarize("  same, eager-with-same-inputs (HF sdpa/eager semantics)", r(one, attention_mask=ones, position_ids=pid, use_cache=False).logits, refp)
    # 5 packed with cu_seq_lens only (return_flash_attn_kwargs=True, return_position_ids=False)
    cu = torch.tensor([0,5,9,12], dtype=torch.int32)
    summarize("llama packed cu_seq_lens kwargs only", m(one, cu_seq_lens_q=cu, cu_seq_lens_k=cu, max_length_q=5, max_length_k=5, use_cache=False).logits, refp)
    # 6 dynamic-cache chunked prefill
    c1, c2 = DynamicCache(config=m.config), DynamicCache(config=r.config)
    m(ids[:, :7], past_key_values=c1); r(ids[:, :7], past_key_values=c2); LOG.clear()
    summarize("llama DynamicCache chunked prefill (q=5, past=7)", m(ids[:, 7:], past_key_values=c1).logits, r(ids[:, 7:], past_key_values=c2).logits)
    # 7 static cache prefill
    s1 = StaticCache(config=m.config, max_cache_len=32); s2 = StaticCache(config=r.config, max_cache_len=32)
    summarize("llama StaticCache prefill (q=12, max_cache_len=32)", m(ids, past_key_values=s1).logits, r(ids, past_key_values=s2).logits)
    # 8 static cache second chunk
    summarize("llama StaticCache 2nd chunk (q=4)", m(ids[:, :4], past_key_values=s1).logits, r(ids[:, :4], past_key_values=s2).logits)
    # 9 generate with static cache
    g1 = m.generate(ids, max_new_tokens=6, do_sample=False, cache_implementation="static", pad_token_id=0)
    LOG.clear(); g2 = r.generate(ids, max_new_tokens=6, do_sample=False, cache_implementation="static", pad_token_id=0)
    print("\n### generate(static) token match:", bool((g1 == g2).all()), g1[:, 12:].tolist(), g2[:, 12:].tolist()); LOG.clear()
    g1 = m.generate(ids, max_new_tokens=6, do_sample=False, pad_token_id=0); LOG.clear()
    g2 = r.generate(ids, max_new_tokens=6, do_sample=False, pad_token_id=0)
    print("### generate(dynamic) token match:", bool((g1 == g2).all())); LOG.clear()
