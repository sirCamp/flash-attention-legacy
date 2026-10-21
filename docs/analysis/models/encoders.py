"""Encoder / embedding survey: BERT, ModernBERT, Qwen3-Embedding (AutoModel hidden states).

usage: python encoders.py MODEL --tag TAG [--n 1000] [--pool last|cls] [--gold]
"""
import argparse
import gc
import logging
import sys
import time

import torch

sys.path.insert(0, "/workspace/analysis/models")
from common import OUT, Spy, cmp, dump, long_text, set_impl  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

logging.getLogger("transformers").setLevel(logging.ERROR)
ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--tag", required=True)
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--pool", default="cls")
ap.add_argument("--gold", action="store_true")
args = ap.parse_args()
R = {"model": args.model, "tag": args.tag, "scenarios": {}, "errors": {}}
spy = Spy()
tok = AutoTokenizer.from_pretrained(args.model)
TEXT = long_text()


def load(dtype, impl):
    m = AutoModel.from_pretrained(args.model, dtype=dtype, device_map="cuda", attn_implementation=impl)
    return m.eval()


model = load(torch.float16, "flash_attn_legacy")
cfg = model.config
R["config"] = dict(
    head_dim=getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads,
    n_heads=cfg.num_attention_heads, n_kv_heads=getattr(cfg, "num_key_value_heads", None),
    layers=cfg.num_hidden_layers, is_causal_cfg=getattr(cfg, "is_causal", "n/a"),
    is_decoder=getattr(cfg, "is_decoder", "n/a"),
    local_attention=getattr(cfg, "local_attention", None), sliding_window=getattr(cfg, "sliding_window", None),
    layer_types=(sorted(set(cfg.layer_types)) if getattr(cfg, "layer_types", None) else None),
    attn_impl_after_load=cfg._attn_implementation, max_pos=getattr(cfg, "max_position_embeddings", None),
    padding_side=tok.padding_side)
print(R["config"], flush=True)
N = min(args.n, (R["config"]["max_pos"] or 10**9))
ids_all = tok(TEXT, return_tensors="pt", add_special_tokens=False).input_ids[0]


def pool(h, am):
    if args.pool == "last":  # Qwen3-Embedding: last token (left padding => position -1)
        if tok.padding_side == "left":
            e = h[:, -1]
        else:
            idx = am.sum(1) - 1
            e = h[torch.arange(h.shape[0]), idx]
    else:
        e = h[:, 0]
    return torch.nn.functional.normalize(e.float(), dim=-1)


inputs = {}
inputs["single_with_mask"] = {"input_ids": ids_all[None, :N].cuda(), "attention_mask": torch.ones(1, N, dtype=torch.long).cuda()}
inputs["single_no_mask"] = {"input_ids": ids_all[None, :N].cuda()}
inputs["single_short64"] = {"input_ids": ids_all[None, :64].cuda()}
texts = [tok.decode(ids_all[:300]), "What is the capital of France?", tok.decode(ids_all[1000:1100]), "Hello"]
inputs["padded_batch4"] = {k: v.cuda() for k, v in tok(texts, return_tensors="pt", padding=True).items()}

outs = {}
with torch.no_grad():
    for name, inp in inputs.items():
        res = {}
        for impl in ("flash_attn_legacy", "sdpa", "eager"):
            try:
                set_impl(model, impl)
                if impl == "flash_attn_legacy":
                    spy.start()
                h = model(**inp).last_hidden_state
                if impl == "flash_attn_legacy":
                    res["path"] = spy.stop()
                    res["first_record"] = spy.records[:1]
                outs[(name, impl)] = h.float().cpu()
            except Exception as e:
                R["errors"][f"{name}/{impl}"] = f"{type(e).__name__}: {str(e)[:300]}"
                spy.active = False
        am = inp.get("attention_mask", torch.ones_like(inp["input_ids"])).bool().cpu()
        for a, b in (("flash_attn_legacy", "sdpa"), ("eager", "sdpa")):
            if (name, a) in outs and (name, b) in outs:
                res[f"{a}_vs_{b}_tokens"] = cmp(outs[(name, a)][am], outs[(name, b)][am])
                ea, eb = pool(outs[(name, a)], am), pool(outs[(name, b)], am)
                res[f"{a}_vs_{b}_pooled_cos"] = torch.nn.functional.cosine_similarity(ea, eb, dim=-1).tolist()
        R["scenarios"][name] = res
        print(name, {k: v for k, v in res.items() if k not in ("first_record",)}, flush=True)
        dump(R, f"{OUT}/results/{args.tag}.json")

    # padded vs alone for each text (flash)
    set_impl(model, "flash_attn_legacy")
    b = inputs["padded_batch4"]
    hb = model(**b).last_hidden_state
    eb = pool(hb, b["attention_mask"])
    al = []
    for i, t in enumerate(texts):
        s = {k: v.cuda() for k, v in tok([t], return_tensors="pt").items()}
        ha = model(**s).last_hidden_state
        al.append(torch.nn.functional.cosine_similarity(pool(ha, s["attention_mask"]), eb[i:i + 1]).item())
    R["scenarios"]["pooled_padded_vs_alone_flash"] = al
    print("padded vs alone", al, flush=True)

if args.gold:
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model = load(torch.float32, "eager")
    G = {}
    with torch.no_grad():
        for name, inp in inputs.items():
            g = model(**inp).last_hidden_state.float().cpu()
            am = inp.get("attention_mask", torch.ones_like(inp["input_ids"])).bool().cpu()
            G[name] = {impl: cmp(outs[(name, impl)][am], g[am]) for impl in ("flash_attn_legacy", "sdpa", "eager")
                       if (name, impl) in outs}
            G[name + "_pooled_cos"] = {impl: torch.nn.functional.cosine_similarity(pool(outs[(name, impl)], am), pool(g, am), dim=-1).tolist()
                                       for impl in ("flash_attn_legacy", "sdpa", "eager") if (name, impl) in outs}
    R["gold_fp32_eager"] = G
    print("GOLD", G, flush=True)
dump(R, f"{OUT}/results/{args.tag}.json")
print("DONE", flush=True)
