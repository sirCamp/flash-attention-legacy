from huggingface_hub import HfApi, hf_hub_download
api=HfApi()
ms=["Qwen/Qwen2.5-0.5B-Instruct","Qwen/Qwen3-0.6B","Qwen/Qwen3-1.7B","mistralai/Mistral-7B-Instruct-v0.3","mistralai/Mistral-7B-v0.1","mistralai/Mistral-7B-Instruct-v0.1","google/gemma-2-2b-it","google/gemma-3-1b-it","microsoft/Phi-3-mini-4k-instruct","microsoft/Phi-3.5-mini-instruct","answerdotai/ModernBERT-base","google-bert/bert-base-uncased","Qwen/Qwen3-Embedding-0.6B","openai/gpt-oss-20b"]
import json
for m in ms:
    try:
        p=hf_hub_download(m,"config.json")
        c=json.load(open(p))
        print("OK",m,{k:c.get(k) for k in ["head_dim","hidden_size","num_attention_heads","num_key_value_heads","sliding_window","use_sliding_window","attn_logit_softcapping","final_logit_softcapping","layer_types","torch_dtype","local_attention","global_attn_every_n_layers","max_window_layers","query_pre_attn_scalar"] if k in c and k!="layer_types"}, "layer_types:", (c.get("layer_types") or [])[:6])
    except Exception as e:
        print("FAIL",m,type(e).__name__,str(e)[:150])
