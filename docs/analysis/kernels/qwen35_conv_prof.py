"""Qwen3.5 on the P100 (P6.10 b): share of the step in the linear-attention layers' causal conv1d (transformers'
F.conv1d fallback and its decode update), in the chunk / recurrent gated delta rule (torch reference) and the rest.
Each function wrapped with CUDA events (forward calls; under training the backward shows in the profiler table).

Usage: python docs/analysis/kernels/qwen35_conv_prof.py MODEL {prefill|lora|decode} N
"""
import collections
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForCausalLM
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

import flash_attn_legacy  # noqa: F401  (registers attn_implementation="flash_attn_legacy")

name, mode, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
acc = collections.defaultdict(list)


def timed(fname):
    fn = getattr(mq, fname)

    def wrapper(*a, **k):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        out = fn(*a, **k)
        e.record()
        acc[fname].append((s, e))
        return out
    setattr(mq, fname, wrapper)


for f in ("causal_conv1d_fn", "causal_conv1d_update", "torch_chunk_gated_delta_rule",
          "torch_recurrent_gated_delta_rule"):
    timed(f)

model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16, attn_implementation="flash_attn_legacy").cuda()
ids = torch.randint(100, 30000, (1, N), generator=torch.Generator().manual_seed(0)).cuda()
conv = next(m.conv1d for m in model.modules() if hasattr(m, "conv1d"))
print("conv1d:", conv, "weight", tuple(conv.weight.shape), conv.weight.dtype)
if mode == "lora":
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()


def step():
    if mode == "prefill":
        with torch.inference_mode():
            model(input_ids=ids, logits_to_keep=1)
    elif mode == "decode":
        with torch.inference_mode():
            model.generate(ids[:, :N], max_new_tokens=64, min_new_tokens=64, do_sample=False)
    else:
        model.train()
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        model.zero_grad(set_to_none=True)


for _ in range(2):
    step()
torch.cuda.synchronize()
acc.clear()
t0 = time.perf_counter()
step()
torch.cuda.synchronize()
wall = (time.perf_counter() - t0) * 1e3
print(f"{name} {mode} N={N}: wall {wall:.1f} ms")
for f, evs in acc.items():
    ms = sum(s.elapsed_time(e) for s, e in evs)
    print(f"  {f:34s} {len(evs):5d} calls {ms:9.1f} ms  {ms / wall:6.1%} of wall")

with profile(activities=[ProfilerActivity.CUDA]) as prof:
    step()
    torch.cuda.synchronize()
tot, allt = collections.Counter(), 0
for e in prof.key_averages():
    t = getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0)
    tot[e.key] += t
    allt += t
print(f"  CUDA total {allt / 1e3:.1f} ms; top kernels:")
for k, t in tot.most_common(14):
    print(f"    {t / 1e3:8.1f} ms {t / allt:6.1%}  {k[:120]}")
conv_t = sum(t for k, t in tot.items() if "conv" in k.lower() or "cudnn" in k.lower())
print(f"  kernels named conv / cudnn: {conv_t / 1e3:.1f} ms {conv_t / allt:.1%}")
