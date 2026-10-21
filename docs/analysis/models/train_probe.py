"""Training probe on Qwen2.5-0.5B: forward+backward, flash_attn_legacy vs sdpa vs eager (fp16), fp32 eager gold.

(a) no padding (attention_mask all ones, as Trainer passes it)
(a2) no padding, no attention_mask at all
(b) right padding, labels -100 on pads
(c) packed via DataCollatorWithFlattening (position_ids restart, no attention_mask)
(c2) same + return_flash_attn_kwargs=True (cu_seq_lens_* kwargs)
For (c) also: loss of each sequence run alone, and logits contamination check.
"""
import gc
import logging
import sys

import torch

sys.path.insert(0, "/workspace/analysis/models")
from common import OUT, Spy, dump, long_text, set_impl  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithFlattening  # noqa: E402

logging.getLogger("transformers").setLevel(logging.ERROR)
NAME = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
TAG = sys.argv[2] if len(sys.argv) > 2 else "train_probe_qwen25"
tok = AutoTokenizer.from_pretrained(NAME)
spy = Spy()
ids = tok(long_text(), return_tensors="pt", add_special_tokens=False).input_ids[0]
PARAMS = ["model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.k_proj.weight",
          "model.layers.12.self_attn.v_proj.weight", "model.layers.23.mlp.down_proj.weight"]

seqs = [ids[0:300], ids[300:450], ids[450:512]]  # 3 sequences, total 512
batches = {}
batches["a_nopad_mask1"] = dict(input_ids=torch.stack([ids[:512], ids[512:1024]]),
                                attention_mask=torch.ones(2, 512, dtype=torch.long),
                                labels=torch.stack([ids[:512], ids[512:1024]]))
batches["a2_nopad_nomask"] = dict(input_ids=torch.stack([ids[:512], ids[512:1024]]),
                                  labels=torch.stack([ids[:512], ids[512:1024]]))
x = torch.full((2, 512), tok.pad_token_id or 0)
am = torch.zeros(2, 512, dtype=torch.long)
x[0], am[0] = ids[:512], 1
x[1, :300], am[1, :300] = ids[512:812], 1
lab = x.clone()
lab[am == 0] = -100
batches["b_rightpad"] = dict(input_ids=x, attention_mask=am, labels=lab)
feats = [{"input_ids": s.tolist()} for s in seqs]
c = DataCollatorWithFlattening(return_position_ids=True)(feats)
batches["c_packed"] = {k: v for k, v in c.items()}
try:
    c2 = DataCollatorWithFlattening(return_position_ids=True, return_flash_attn_kwargs=True)(feats)
    batches["c2_packed_fa_kwargs"] = {k: v for k, v in c2.items()}
except TypeError as e:
    print("no return_flash_attn_kwargs:", e)
print({k: {kk: tuple(vv.shape) if hasattr(vv, "shape") else vv for kk, vv in v.items()} for k, v in batches.items()}, flush=True)
print("packed position_ids head:", batches["c_packed"]["position_ids"][0, 295:305].tolist(),
      "labels at boundaries:", batches["c_packed"]["labels"][0, [0, 300, 450]].tolist(), flush=True)


def to_cuda(b):
    return {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}


def run(model, impl, b, want_logits=False):
    set_impl(model, impl)
    model.zero_grad(set_to_none=True)
    out = model(**to_cuda(b))
    out.loss.backward()
    g = {p: model.get_parameter(p).grad.float().norm().item() for p in PARAMS}
    gv = {p: model.get_parameter(p).grad.float().flatten()[:200000].cpu() for p in PARAMS[1:3]}
    lg = out.logits.detach().float().cpu() if want_logits else None
    return out.loss.item(), g, gv, lg


R = {"model": NAME, "results": {}}


def probe(model, label, impls):
    res = {}
    for bn, b in batches.items():
        r = {}
        for impl in impls:
            if impl == "flash_attn_legacy":
                spy.start()
            loss, g, gv, lg = run(model, impl, b, want_logits=bn.startswith("c"))
            if impl == "flash_attn_legacy":
                path = spy.stop()
                r["path"] = {k: path.get(k) for k in ("calls", "kernel_calls", "prefill(q>1)", "kwargs_seen")}
            r[impl] = dict(loss=loss, grad_norms=g)
            STORE[(label, bn, impl)] = (gv, lg)
        res[bn] = r
        print(label, bn, {k: (v["loss"] if isinstance(v, dict) and "loss" in v else "") for k, v in r.items()}, flush=True)
    return res


STORE = {}
model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float16, device_map="cuda",
                                             attn_implementation="flash_attn_legacy")
model.train()
R["config_attention_dropout"] = getattr(model.config, "attention_dropout", None)
R["results"]["fp16"] = probe(model, "fp16", ("flash_attn_legacy", "sdpa", "eager"))


# packed: each sequence alone (sdpa and flash, no grad) -> loss and logits
@torch.no_grad()
def alone(model, impl):
    set_impl(model, impl)
    tot, n, lgs = 0.0, 0, []
    for s in seqs:
        o = model(input_ids=s[None].cuda(), labels=s[None].cuda())
        k = len(s) - 1
        tot += o.loss.item() * k
        n += k
        lgs.append(o.logits[0].float().cpu())
    return tot / n, torch.cat(lgs)


model.eval()
for impl in ("flash_attn_legacy", "sdpa"):
    l, lg = alone(model, impl)
    STORE[("fp16", "alone", impl)] = lg
    R["results"].setdefault("alone_each_sequence", {})[impl] = dict(token_weighted_loss=l)


def contamination(lg_packed, lg_alone):
    """1-cos per packed segment vs logits of that sequence alone."""
    out = {}
    o = 0
    for i, s in enumerate(seqs):
        a, b = lg_packed[o:o + len(s)], lg_alone[o:o + len(s)]
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
        out[f"seq{i}"] = dict(one_minus_cos=float(1 - cos.mean()), top1=float((a.argmax(-1) == b.argmax(-1)).float().mean()))
        o += len(s)
    return out


for bn in [k for k in batches if k.startswith("c")]:
    for impl in ("flash_attn_legacy", "sdpa", "eager"):
        _, lg = STORE[("fp16", bn, impl)]
        R["results"]["fp16"][bn][f"{impl}_packed_vs_alone_sdpa"] = contamination(lg[0], STORE[("fp16", "alone", "sdpa")])


def gcos(a, b):
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


for bn in batches:
    gs = STORE[("fp16", bn, "sdpa")][0]
    R["results"]["fp16"][bn]["grad_cos_vs_sdpa"] = {
        impl: {p: gcos(STORE[("fp16", bn, impl)][0][p], gs[p]) for p in gs} for impl in ("flash_attn_legacy", "eager")}

del model
gc.collect()
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32, device_map="cuda", attn_implementation="eager")
model.train()
R["results"]["fp32_eager_gold"] = probe(model, "fp32", ("eager",))
for bn in batches:
    gg = STORE[("fp32", bn, "eager")][0]
    R["results"]["fp16"][bn]["grad_cos_vs_fp32_gold"] = {
        impl: {p: gcos(STORE[("fp16", bn, impl)][0][p], gg[p]) for p in gg} for impl in ("flash_attn_legacy", "sdpa", "eager")}
dump(R, f"{OUT}/results/{TAG}.json")
print("DONE", flush=True)
