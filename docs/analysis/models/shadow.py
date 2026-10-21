"""Shadow check: on every attention call with the flash_attn_legacy backend, recompute the attention
in fp32 *with every feature the model passed* (softcap, sliding window, sinks, mask) and record the
error of what flash_attn_legacy returned. Also report generation divergence on a short prompt.

usage: python shadow.py MODEL [--n 26,1000] [--bf16]
"""
import argparse
import logging
import sys

import torch

sys.path.insert(0, "/workspace/analysis/models")
from common import OUT, PROMPT, cmp, dump, long_text  # noqa: E402
import flash_attn_legacy  # noqa: E402,F401
import flash_attn_legacy.flash_attn as fa  # noqa: E402
import flash_attn_legacy.integrations as integ  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402

logging.getLogger("transformers").setLevel(logging.ERROR)
ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--n", default="26,1000")
ap.add_argument("--tag", default=None)
ap.add_argument("--encoder", action="store_true")
args = ap.parse_args()
tag = args.tag or ("shadow_" + args.model.split("/")[-1])

REC = []
STATE = {"hit": False}
orig_kernel = fa.flash_attention


def kernel(q, k, v, *a, **kw):
    STATE["hit"] = True
    return orig_kernel(q, k, v, *a, **kw)


fa.flash_attention = kernel


def full_ref(module, q, k, v, mask, scaling, kwargs):
    """fp32 attention honoring causal / sliding_window / softcap / s_aux / mask.
    Returns (ref, sdpa16) where sdpa16 = plain fp16 SDPA on the same inputs+mask (per-call noise floor),
    or None when the call carries softcap / sinks (SDPA cannot express them)."""
    Hq, Hk = q.shape[1], k.shape[1]
    k16 = k.repeat_interleave(Hq // Hk, 1)
    v16 = v.repeat_interleave(Hq // Hk, 1)
    k, v = k16.float(), v16.float()
    nq, nk = q.shape[2], k.shape[2]
    s = (q.float() @ k.transpose(-1, -2)) * scaling
    softcap = kwargs.get("softcap")
    if softcap:
        s = torch.tanh(s / softcap) * softcap
    if mask is not None:
        allowed = mask[..., :nk]
        s = s.masked_fill(~allowed, float("-inf")) if allowed.dtype == torch.bool else s + allowed.float()
    else:
        allowed = torch.ones(nq, nk, dtype=torch.bool, device=q.device)
        if getattr(module, "is_causal", True):
            allowed = allowed.tril(nk - nq)
        sw = kwargs.get("sliding_window")
        if sw:
            qi = torch.arange(nq, device=q.device)[:, None] + (nk - nq)
            ki = torch.arange(nk, device=q.device)[None]
            allowed &= (qi - ki) < sw
        s = s.masked_fill(~allowed, float("-inf"))
    sd = None
    if not softcap and kwargs.get("s_aux") is None:
        sd = torch.nn.functional.scaled_dot_product_attention(q, k16, v16, attn_mask=allowed, scale=scaling).transpose(1, 2).float()
    sinks = kwargs.get("s_aux")
    if sinks is not None:
        sk = sinks.float().reshape(1, -1, 1, 1).expand(s.shape[0], -1, nq, 1)
        s = torch.cat([s, sk], -1)
        p = torch.softmax(s, -1)[..., :-1]
    else:
        p = torch.softmax(s, -1)
    return (p @ v).transpose(1, 2), sd


def fwd(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    STATE["hit"] = False
    out, _ = integ.flash_attn_legacy_forward(module, query, key, value, attention_mask, dropout=dropout,
                                             scaling=scaling, **kwargs)
    if STATE.get("on"):
        sc = scaling if scaling is not None else query.shape[-1] ** -0.5
        r, sd = full_ref(module, query, key, value, attention_mask, sc, kwargs)
        err = (out.float() - r).abs().max().item()
        rel = ((out.float() - r).norm() / r.norm()).item()
        sd_err = None if sd is None else (sd - r).abs().max().item()
        sd_rel = None if sd is None else ((sd - r).norm() / r.norm()).item()
        REC.append(dict(layer=getattr(module, "layer_idx", None), q=query.shape[2], k=key.shape[2],
                        kernel=STATE["hit"], max_err=err, rel_err=rel, sdpa16_max_err=sd_err, sdpa16_rel_err=sd_rel,
                        softcap=kwargs.get("softcap"), sliding_window=kwargs.get("sliding_window"),
                        sinks=kwargs.get("s_aux") is not None, mask=None if attention_mask is None else tuple(attention_mask.shape),
                        q_absmax=query.abs().max().item(), k_absmax=key.abs().max().item()))
    return out, None


ALL_ATTENTION_FUNCTIONS.register("flash_attn_legacy", fwd)
tok = AutoTokenizer.from_pretrained(args.model)
from transformers import AutoModel  # noqa: E402
Cls = AutoModel if args.encoder else AutoModelForCausalLM
model = Cls.from_pretrained(args.model, dtype=torch.float16, device_map="cuda",
                                             attn_implementation="flash_attn_legacy").eval()
R = {"model": args.model, "cases": {}}
ids_all = tok(long_text(), return_tensors="pt", add_special_tokens=not args.encoder).input_ids.cuda()
inputs = {f"text{n}": ids_all[:, :int(n)] for n in args.n.split(",")}
inputs["prompt"] = tok(PROMPT, return_tensors="pt").input_ids.cuda()
with torch.no_grad():
    for name, ids in inputs.items():
        REC.clear()
        STATE["on"] = True
        model.set_attn_implementation("flash_attn_legacy")
        get = (lambda o: o.last_hidden_state[0].float()) if args.encoder else (lambda o: o.logits[0].float())
        fl = get(model(input_ids=ids))
        STATE["on"] = False
        model.set_attn_implementation("sdpa")
        sd = get(model(input_ids=ids))
        model.set_attn_implementation("eager")
        ea = get(model(input_ids=ids))
        worst = sorted(REC, key=lambda r: -r["rel_err"])[:4]
        R["cases"][name] = dict(N=ids.shape[1], flash_vs_sdpa=cmp(fl, sd), eager_vs_sdpa=cmp(ea, sd), flash_vs_eager=cmp(fl, ea),
                                last_pos_top5_flash=fl[-1].topk(5).indices.tolist(), last_pos_top5_sdpa=sd[-1].topk(5).indices.tolist(),
                                last_pos_top2_margin_sdpa=float(sd[-1].topk(2).values.diff().abs()),
                                per_layer=[{k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()} for r in REC],
                                worst_layers=worst,
                                max_rel_err_kernel_layers=max([r["rel_err"] for r in REC if r["kernel"]], default=None),
                                max_rel_err_fallback_layers=max([r["rel_err"] for r in REC if not r["kernel"]], default=None))
        c = R["cases"][name]
        print(name, "N", c["N"], "flash_vs_sdpa", c["flash_vs_sdpa"], "eager_vs_sdpa", c["eager_vs_sdpa"],
              "\n   max rel err (vs full-feature fp32 ref): kernel layers", c["max_rel_err_kernel_layers"],
              "fallback layers", c["max_rel_err_fallback_layers"],
              "| sdpa16 same inputs max rel", max([r["sdpa16_rel_err"] for r in REC if r["sdpa16_rel_err"] is not None], default=None), "\n   worst", worst[:2], flush=True)
    if args.encoder:
        dump(R, f"{OUT}/results/{tag}.json")
        sys.exit(0)
    # step-by-step greedy decode comparison on the prompt
    gen = dict(max_new_tokens=16, min_new_tokens=16, do_sample=False, output_scores=True, return_dict_in_generate=True)
    outs = {}
    for impl in ("flash_attn_legacy", "sdpa", "eager"):
        model.set_attn_implementation(impl)
        outs[impl] = model.generate(inputs["prompt"], **gen)
    steps = []
    for t in range(16):
        s = {impl: outs[impl].scores[t][0].float() for impl in outs}
        steps.append(dict(step=t, tok={i: int(outs[i].sequences[0, inputs['prompt'].shape[1] + t]) for i in outs},
                          top2_margin_sdpa=float(s["sdpa"].topk(2).values.diff().abs()),
                          flash_vs_sdpa_maxabs=float((s["flash_attn_legacy"] - s["sdpa"]).abs().max()),
                          eager_vs_sdpa_maxabs=float((s["eager"] - s["sdpa"]).abs().max())))
    R["gen_steps"] = steps
    for st in steps[:8]:
        print(st, flush=True)
dump(R, f"{OUT}/results/{tag}.json")
print("DONE")
