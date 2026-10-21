"""Per-model survey of the flash_attn_legacy HF backend (causal LMs).

usage: python survey.py MODEL [--n 1000] [--long 5000] [--gold] [--override '{"k": v}'] [--tag name]
Writes results/<tag>.json. Phase A: fp16 model (flash/sdpa/eager). Phase B (--gold): fp32 eager gold.
"""
import argparse
import gc
import json
import logging
import math
import sys
import time

import torch

sys.path.insert(0, "/workspace/analysis/models")
from common import (OUT, PROMPT, Spy, cmp, dump, first_div, long_text, set_impl)  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache  # noqa: E402

logging.getLogger("transformers").setLevel(logging.ERROR)
ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--long", type=int, default=0)
ap.add_argument("--gold", action="store_true")
ap.add_argument("--override", default="{}")
ap.add_argument("--tag", default=None)
ap.add_argument("--skip_eager_long", action="store_true")
args = ap.parse_args()
tag = args.tag or args.model.split("/")[-1]
override = json.loads(args.override)
R = {"model": args.model, "tag": tag, "override": override, "scenarios": {}, "errors": {}}
T0 = time.time()


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def load(dtype, impl):
    cfg = AutoConfig.from_pretrained(args.model)
    for k, v in override.items():
        setattr(cfg, k, v)
    if override and hasattr(cfg, "layer_types") and "use_sliding_window" in override:
        # Qwen2: recompute layer_types from the overridden window settings
        cfg.layer_types = [
            "sliding_attention" if cfg.use_sliding_window and i >= cfg.max_window_layers else "full_attention"
            for i in range(cfg.num_hidden_layers)]
    m = AutoModelForCausalLM.from_pretrained(args.model, config=cfg, dtype=dtype, device_map="cuda",
                                             attn_implementation=impl)
    m.eval()
    return m


tok = AutoTokenizer.from_pretrained(args.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
TEXT = long_text()
spy = Spy()

model = load(torch.float16, "flash_attn_legacy")
cfg = model.config
L = cfg.num_hidden_layers
head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
R["config"] = dict(
    head_dim=head_dim, n_heads=cfg.num_attention_heads, n_kv_heads=getattr(cfg, "num_key_value_heads", None),
    layers=L, sliding_window=getattr(cfg, "sliding_window", None),
    use_sliding_window=getattr(cfg, "use_sliding_window", "n/a"),
    layer_types=(sorted(set(cfg.layer_types)) if getattr(cfg, "layer_types", None) else None),
    n_sliding_layers=(sum(t == "sliding_attention" for t in cfg.layer_types) if getattr(cfg, "layer_types", None) else None),
    attn_logit_softcapping=getattr(cfg, "attn_logit_softcapping", None),
    final_logit_softcapping=getattr(cfg, "final_logit_softcapping", None),
    is_causal=getattr(cfg, "is_causal", "n/a"),
    attn_impl_after_load=cfg._attn_implementation,
    max_pos=getattr(cfg, "max_position_embeddings", None),
    gen_cache_impl=getattr(model.generation_config, "cache_implementation", None),
    has_sinks=any("sinks" in n for n, _ in model.named_parameters()),
)
log("config", R["config"])
R["first_records"] = {}

store = {}  # (scenario, impl) -> cpu tensor


@torch.no_grad()
def fwd(impl, ids, keep=None, **kw):
    set_impl(model, impl)
    kk = {} if keep is None else {"logits_to_keep": keep}
    return model(input_ids=ids, **kk, **kw).logits[0].float()


def scen(name, fn):
    log("scenario", name)
    try:
        fn()
    except Exception as e:  # record and continue
        import traceback
        traceback.print_exc()
        R["errors"][name] = f"{type(e).__name__}: {str(e)[:400]}"
        spy.active = False
    torch.cuda.empty_cache()
    dump(R, f"{OUT}/results/{tag}.json")


all_ids = tok(TEXT, return_tensors="pt").input_ids.cuda()
N = min(args.n, (R["config"]["max_pos"] or 10**9) - 64)
ids1 = all_ids[:, :N]


# ---- S1: unpadded prefill -----------------------------------------------------
def s1():
    spy.start()
    fl = fwd("flash_attn_legacy", ids1)
    R["first_records"]["prefill"] = spy.records[:2]
    path = spy.stop()
    sd = fwd("sdpa", ids1)
    ea = fwd("eager", ids1)
    store[("prefill", "flash")] = fl.half().cpu()
    store[("prefill", "sdpa")] = sd.half().cpu()
    store[("prefill", "eager")] = ea.half().cpu()
    R["scenarios"]["prefill"] = dict(N=N, path=path, flash_vs_sdpa=cmp(fl, sd), eager_vs_sdpa=cmp(ea, sd),
                                     flash_vs_eager=cmp(fl, ea))
    log(R["scenarios"]["prefill"])


scen("prefill", s1)


# ---- S2: left-padded batch of 2 -----------------------------------------------
def s2():
    long_prompt = tok.decode(all_ids[0, :200])
    prompts = [long_prompt, "The capital of France is"]
    tok.padding_side = "left"
    batch = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    am = batch.attention_mask.bool()
    res = {}
    with torch.no_grad():
        outs = {}
        for impl in ("flash_attn_legacy", "sdpa", "eager"):
            set_impl(model, impl)
            if impl == "flash_attn_legacy":
                spy.start()
            lg = model(**batch).logits.float()
            if impl == "flash_attn_legacy":
                res["path_forward"] = spy.stop()
            outs[impl] = lg
        for i in range(2):
            sel = am[i]
            res[f"seq{i}_flash_vs_sdpa"] = cmp(outs["flash_attn_legacy"][i][sel], outs["sdpa"][i][sel])
            res[f"seq{i}_eager_vs_sdpa"] = cmp(outs["eager"][i][sel], outs["sdpa"][i][sel])
        # padded short prompt vs the same prompt alone (flash)
        single = tok(prompts[1:], return_tensors="pt").to("cuda")
        set_impl(model, "flash_attn_legacy")
        alone = model(**single).logits.float()[0]
        res["short_padded_vs_alone_flash"] = cmp(outs["flash_attn_legacy"][1][am[1]], alone)
    gen = dict(max_new_tokens=24, min_new_tokens=24, do_sample=False)
    toks = {}
    for impl in ("flash_attn_legacy", "sdpa"):
        set_impl(model, impl)
        if impl == "flash_attn_legacy":
            spy.start()
        toks[impl] = model.generate(**batch, **gen)[:, batch.input_ids.shape[1]:]
        if impl == "flash_attn_legacy":
            res["path_generate"] = spy.stop()
    res["gen_first_div"] = [first_div(toks["flash_attn_legacy"][i], toks["sdpa"][i]) for i in range(2)]
    res["gen_text_flash"] = [tok.decode(t) for t in toks["flash_attn_legacy"]]
    res["gen_text_sdpa"] = [tok.decode(t) for t in toks["sdpa"]]
    tok.padding_side = "right"
    R["scenarios"]["left_padded"] = res
    log({k: v for k, v in res.items() if not k.startswith("gen_text")})


scen("left_padded", s2)


# ---- S3: chunked prefill with DynamicCache -----------------------------------
def s3():
    ids = all_ids[:, :400]
    split = 300

    @torch.no_grad()
    def chunked(impl):
        set_impl(model, impl)
        cache = DynamicCache(config=model.config)
        model(input_ids=ids[:, :split], past_key_values=cache, use_cache=True)
        return model(input_ids=ids[:, split:], past_key_values=cache, use_cache=True).logits[0].float()

    ref = fwd("sdpa", ids)[split:]
    spy.start()
    fl = chunked("flash_attn_legacy")
    path = spy.stop()
    ea = chunked("eager")
    sd = chunked("sdpa")
    R["scenarios"]["chunked"] = dict(split=[split, 100], path=path, flash_chunk_vs_sdpa_full=cmp(fl, ref),
                                     eager_chunk_vs_sdpa_full=cmp(ea, ref), sdpa_chunk_vs_sdpa_full=cmp(sd, ref))
    store[("chunked", "flash")] = fl.half().cpu()
    log(R["scenarios"]["chunked"])


scen("chunked", s3)


# ---- S4: greedy generate 64 tokens ------------------------------------------
gen_prompt = tok(PROMPT, return_tensors="pt").input_ids.cuda()


def s4():
    gen = dict(max_new_tokens=64, min_new_tokens=64, do_sample=False)
    toks = {}
    res = {}
    for impl in ("flash_attn_legacy", "sdpa", "eager"):
        set_impl(model, impl)
        if impl == "flash_attn_legacy":
            spy.start()
        t0 = time.time()
        toks[impl] = model.generate(gen_prompt, **gen)[0, gen_prompt.shape[1]:]
        res[f"wall_s_{impl}"] = round(time.time() - t0, 2)
        if impl == "flash_attn_legacy":
            res["path"] = spy.stop()
            R["first_records"]["generate_decode"] = [r for r in spy.records if r["q"][2] == 1][:1]
    res["first_div_flash_vs_sdpa"] = first_div(toks["flash_attn_legacy"], toks["sdpa"])
    res["first_div_eager_vs_sdpa"] = first_div(toks["eager"], toks["sdpa"])
    res["first_div_flash_vs_eager"] = first_div(toks["flash_attn_legacy"], toks["eager"])
    res["text_flash"] = tok.decode(toks["flash_attn_legacy"])
    res["text_sdpa"] = tok.decode(toks["sdpa"])
    res["text_eager"] = tok.decode(toks["eager"])
    store[("gen", "flash")] = toks["flash_attn_legacy"].cpu()
    store[("gen", "sdpa")] = toks["sdpa"].cpu()
    store[("gen", "eager")] = toks["eager"].cpu()
    R["scenarios"]["generate"] = res
    log(res)


scen("generate", s4)

# ---- S5: long context beyond the sliding window ------------------------------
LONG = 0
if args.long:
    LONG = min(args.long, (R["config"]["max_pos"] or 10**9) - 64, all_ids.shape[1])
CH = 256
if LONG:
    step = max(1, math.ceil(LONG / 1024))
    keep = torch.unique(torch.cat([torch.arange(0, LONG, step), torch.arange(LONG - CH, LONG)])).cuda()
    idsL = all_ids[:, :LONG]
    W = R["config"]["sliding_window"] or LONG


def split_metrics(a, b, pos):
    out = {"all": cmp(a, b)}
    lo, hi = pos < W, pos >= W
    if lo.any():
        out[f"pos<{W}"] = cmp(a[lo], b[lo])
    if hi.any():
        out[f"pos>={W}"] = cmp(a[hi], b[hi])
    return out


def s5():
    pos = keep.cpu()
    spy.start()
    fl = fwd("flash_attn_legacy", idsL, keep=keep)
    path = spy.stop()
    sd = fwd("sdpa", idsL, keep=keep)
    res = dict(N=LONG, window=W, path=path, flash_vs_sdpa=split_metrics(fl, sd, pos))
    store[("long", "flash")] = fl.half().cpu()
    store[("long", "sdpa")] = sd.half().cpu()
    if not args.skip_eager_long:
        ea = fwd("eager", idsL, keep=keep)
        res["eager_vs_sdpa"] = split_metrics(ea, sd, pos)
        res["flash_vs_eager"] = split_metrics(fl, ea, pos)
        store[("long", "eager")] = ea.half().cpu()
        del ea
    # chunked over the window: cache LONG-CH tokens, then a CH-token chunk
    with torch.no_grad():
        set_impl(model, "flash_attn_legacy")
        cache = DynamicCache(config=model.config)
        model(input_ids=idsL[:, :-CH], past_key_values=cache, use_cache=True, logits_to_keep=1)
        spy.start()
        flc = model(input_ids=idsL[:, -CH:], past_key_values=cache, use_cache=True).logits[0].float()
        res["chunk_path"] = spy.stop()
        del cache
    tail = torch.isin(keep, torch.arange(LONG - CH, LONG, device=keep.device))
    res["flash_chunk_vs_sdpa_full"] = cmp(flc, sd[tail])
    store[("longchunk", "flash")] = flc.half().cpu()
    # generation after the long prompt (decode attends beyond window)
    gen = dict(max_new_tokens=32, min_new_tokens=32, do_sample=False)
    toks = {}
    for impl in ("flash_attn_legacy", "sdpa"):
        set_impl(model, impl)
        if impl == "flash_attn_legacy":
            spy.start()
        toks[impl] = model.generate(idsL, **gen)[0, LONG:]
        if impl == "flash_attn_legacy":
            res["gen_path"] = spy.stop()
    res["gen_first_div_flash_vs_sdpa"] = first_div(toks["flash_attn_legacy"], toks["sdpa"])
    res["gen_text_flash"] = tok.decode(toks["flash_attn_legacy"])
    res["gen_text_sdpa"] = tok.decode(toks["sdpa"])
    store[("longgen", "flash")] = toks["flash_attn_legacy"].cpu()
    store[("longgen", "sdpa")] = toks["sdpa"].cpu()
    R["scenarios"]["long"] = res
    log({k: v for k, v in res.items() if "text" not in k})


if LONG:
    scen("long", s5)

# ---- Phase B: fp32 eager gold -------------------------------------------------
if args.gold:
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model = load(torch.float32, "eager")
    G = {}

    def gold():
        with torch.no_grad():
            g = model(input_ids=ids1).logits[0].float()
            G["prefill"] = {k: cmp(store[("prefill", k)].cuda().float(), g) for k in ("flash", "sdpa", "eager")}
            g = model(input_ids=all_ids[:, :400]).logits[0, 300:].float()
            G["chunked_flash"] = cmp(store[("chunked", "flash")].cuda().float(), g)
            del g
        gt = model.generate(gen_prompt, max_new_tokens=64, min_new_tokens=64, do_sample=False)[0, gen_prompt.shape[1]:].cpu()
        G["gen_first_div_vs_gold"] = {k: first_div(store[("gen", k)], gt) for k in ("flash", "sdpa", "eager")}
        G["gen_text_gold"] = tok.decode(gt)
        if LONG:
            pos = keep.cpu()
            with torch.no_grad():
                g = model(input_ids=idsL, logits_to_keep=keep).logits[0].float()
            G["long"] = {k: split_metrics(store[("long", k)].cuda().float(), g, pos)
                         for k in ("flash", "sdpa", "eager") if ("long", k) in store}
            tail = torch.isin(keep, torch.arange(LONG - CH, LONG, device=keep.device))
            G["longchunk_flash"] = cmp(store[("longchunk", "flash")].cuda().float(), g[tail])
            del g
            torch.cuda.empty_cache()
            gt = model.generate(idsL, max_new_tokens=32, min_new_tokens=32, do_sample=False)[0, LONG:].cpu()
            G["longgen_first_div_vs_gold"] = {k: first_div(store[("longgen", k)], gt) for k in ("flash", "sdpa")}
            G["longgen_text_gold"] = tok.decode(gt)
        R["gold_fp32_eager"] = G
        log(G)

    scen("gold", gold)

R["elapsed_s"] = round(time.time() - T0, 1)
dump(R, f"{OUT}/results/{tag}.json")
log("DONE")
