"""Follow-ups to train_probe.py on Qwen2.5-0.5B.

1. Packed sequences (DataCollatorWithFlattening) with use_cache=False vs default (use_cache=True in train mode):
   which path runs and is there cross-sequence contamination (logits vs each sequence run alone)?
2. Backward accuracy of the kernel vs fp16 SDPA / fp16 math, both vs an fp32 reference, on
   (a) random q/k/v and (b) real layer-0 q/k/v captured from the model.
"""
import logging
import sys

import torch

sys.path.insert(0, "/workspace/analysis/models")
from common import OUT, Spy, dump, long_text, set_impl  # noqa: E402
from flash_attn_legacy.flash_attn import flash_attention  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithFlattening  # noqa: E402

logging.getLogger("transformers").setLevel(logging.ERROR)
NAME = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(NAME)
spy = Spy()
ids = tok(long_text(), return_tensors="pt", add_special_tokens=False).input_ids[0]
seqs = [ids[0:300], ids[300:450], ids[450:512]]
feats = [{"input_ids": s.tolist()} for s in seqs]
packed = DataCollatorWithFlattening(return_position_ids=True)(feats)
packed_fa = DataCollatorWithFlattening(return_position_ids=True, return_flash_attn_kwargs=True)(feats)
model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float16, device_map="cuda",
                                             attn_implementation="flash_attn_legacy")
R = {"packed": {}, "backward": {}}


@torch.no_grad()
def alone_logits():
    set_impl(model, "sdpa")
    return torch.cat([model(input_ids=s[None].cuda(), use_cache=False).logits[0].float() for s in seqs])


def contamination(lg, ref):
    out, o = {}, 0
    for i, s in enumerate(seqs):
        a, b = lg[o:o + len(s)], ref[o:o + len(s)]
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
        out[f"seq{i}"] = dict(one_minus_cos=float(1 - cos.mean()), top1=float((a.argmax(-1) == b.argmax(-1)).float().mean()))
        o += len(s)
    return out


model.eval()
ref = alone_logits()
model.train()
for bname, b in (("packed", packed), ("packed_fa_kwargs", packed_fa)):
    for uc in (None, False):
        for impl in ("flash_attn_legacy", "sdpa", "eager"):
            set_impl(model, impl)
            kw = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
            if uc is not None:
                kw["use_cache"] = uc
            if impl == "flash_attn_legacy":
                spy.start()
            out = model(**kw)
            out.loss.backward()
            model.zero_grad(set_to_none=True)
            row = dict(loss=out.loss.item(), contamination=contamination(out.logits[0].float().detach(), ref))
            if impl == "flash_attn_legacy":
                p = spy.stop()
                row["path"] = dict(kernel=p["kernel_calls"], calls=p["calls"], masks=p.get("prefill(q>1)", {}).get("masks"))
            key = f"{bname}/use_cache={'default(True)' if uc is None else uc}/{impl}"
            R["packed"][key] = row
            print(key, row, flush=True)
model.eval()
with torch.no_grad():
    o = 0
    tot = 0.0
    n = 0
    for s in seqs:
        r = model(input_ids=s[None].cuda(), labels=s[None].cuda(), use_cache=False)
        tot += r.loss.item() * (len(s) - 1)
        n += len(s) - 1
    R["packed"]["alone_token_weighted_loss_sdpa"] = tot / n
print("alone loss", tot / n, flush=True)

# ---- capture real layer-0 q/k/v -----------------------------------------------
CAP = {}
import flash_attn_legacy.integrations as integ  # noqa: E402
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402


def cap_fwd(module, q, k, v, mask, *a, **kw):
    if getattr(module, "layer_idx", None) in (0, 12) and module.layer_idx not in CAP:
        CAP[module.layer_idx] = (q.detach().clone(), k.detach().clone(), v.detach().clone(), kw.get("scaling"))
    return integ.flash_attn_legacy_forward(module, q, k, v, mask, *a, **kw)


ALL_ATTENTION_FUNCTIONS.register("flash_attn_legacy", cap_fwd)
set_impl(model, "flash_attn_legacy")
with torch.no_grad():
    model(input_ids=ids[None, :512].cuda())


def bwd_case(q, k, v, scale, label):
    torch.manual_seed(1)
    do = torch.randn(q.shape[0], q.shape[2], q.shape[1], q.shape[3], device="cuda", dtype=torch.float16)  # [B,N,H,d]
    rep = q.shape[1] // k.shape[1]
    res = {}
    grads = {}
    for impl in ("fp32_math", "flash", "sdpa16", "math16"):
        dt = torch.float32 if impl == "fp32_math" else torch.float16
        qq, kk, vv = (t.detach().to(dt).requires_grad_(True) for t in (q, k, v))
        if impl == "flash":
            o = flash_attention(qq, kk, vv, softmax_scale=scale, causal=True)
        else:
            k2, v2 = kk.repeat_interleave(rep, 1), vv.repeat_interleave(rep, 1)
            if impl == "sdpa16":
                o = torch.nn.functional.scaled_dot_product_attention(qq, k2, v2, is_causal=True, scale=scale)
            else:
                s = (qq.float() @ k2.float().transpose(-1, -2)) * scale if impl == "fp32_math" else (qq @ k2.transpose(-1, -2)) * scale
                n = q.shape[2]
                m = torch.ones(n, n, dtype=torch.bool, device="cuda").tril()
                s = s.masked_fill(~m, float("-inf"))
                p = torch.softmax(s.float(), -1).to(dt)
                o = p @ v2
        o.backward(do.transpose(1, 2).to(dt))
        grads[impl] = (qq.grad.float(), kk.grad.float(), vv.grad.float(), o.detach().float())
    g0 = grads["fp32_math"]
    for impl in ("flash", "sdpa16", "math16"):
        res[impl] = {n: dict(rel=float((g - r).norm() / r.norm()), cos=float(torch.nn.functional.cosine_similarity(g.flatten(), r.flatten(), dim=0)))
                     for n, g, r in zip(("dq", "dk", "dv", "out"), grads[impl], g0)}
    R["backward"][label] = res
    print(label, {i: {n: f"rel={x['rel']:.2e}" for n, x in d.items()} for i, d in res.items()}, flush=True)


torch.manual_seed(0)
q = torch.randn(1, 14, 512, 64, device="cuda", dtype=torch.float16)
k = torch.randn(1, 2, 512, 64, device="cuda", dtype=torch.float16)
v = torch.randn(1, 2, 512, 64, device="cuda", dtype=torch.float16)
bwd_case(q, k, v, 0.125, "random_N512_d64_gqa14:2")
for li, (q, k, v, sc) in CAP.items():
    bwd_case(q, k, v, sc, f"real_qwen25_layer{li}_N512 (q_absmax={q.abs().max().item():.1f}, k_absmax={k.abs().max().item():.1f})")
dump(R, f"{OUT}/results/train_probe2_qwen25.json")
print("DONE")
