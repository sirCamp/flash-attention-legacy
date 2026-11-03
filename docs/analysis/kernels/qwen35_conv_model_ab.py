"""P6.10 b A/B, one process per run: Qwen3.5-2B fp16 on our attention, prefill of 4096 tokens, generate 2048 + 64
(greedy, 64 forced) and a LoRA step at 1024 tokens (peft r16 on the projections, gradient checkpointing, fp16
autocast, forward + backward), wall clock medians after a warm-up, and the peak memory of a prefill and of a step.
The arms are set by the caller's environment (P6.10 b: FLASH_ATTN_LEGACY_CONV1D=0, transformers' reference conv,
against our kernels; c1 / c3: FLASH_ATTN_LEGACY_GDN_SCAN=0; c2: FLASH_ATTN_LEGACY_GDN_PREP=0). Prints one JSON line.

Usage: python docs/analysis/kernels/qwen35_conv_model_ab.py LABEL (NO_LORA=1: prefill and generate only)
"""
import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM

import flash_attn_legacy

name = "Qwen/Qwen3.5-2B"
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16, attn_implementation="flash_attn_legacy").cuda()
gen = torch.Generator().manual_seed(0)
prefill_ids = torch.randint(100, 30000, (1, 4096), generator=gen).cuda()
prompt = torch.randint(100, 30000, (1, 2048), generator=gen).cuda()
train_ids = torch.randint(100, 30000, (1, 1024), generator=gen).cuda()


def timed(fn, reps):
    out = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) * 1e3)
    return sorted(out)[len(out) // 2]


res = {"label": sys.argv[1]}
with torch.inference_mode():
    prefill = lambda: model(input_ids=prefill_ids, logits_to_keep=1)  # noqa: E731
    decode = lambda: model.generate(prompt, max_new_tokens=64, min_new_tokens=64, do_sample=False)  # noqa: E731
    prefill(), decode()
    res["prefill_ms"], res["generate_ms"] = timed(prefill, 3), timed(decode, 3)
    torch.cuda.reset_peak_memory_stats()
    prefill()
    res["prefill_peak_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)

import os  # noqa: E402

if os.environ.get("NO_LORA") == "1":  # a GPU shared with another tenant: no room for the LoRA step
    res["fallbacks"] = flash_attn_legacy.fallback_stats()
    print(json.dumps(res))
    sys.exit(0)
from peft import LoraConfig, get_peft_model  # noqa: E402

model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=[
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
for p in model.parameters():
    if p.requires_grad:
        p.data = p.data.float()
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.train()


def step():
    with torch.autocast("cuda", dtype=torch.float16):
        loss = model(input_ids=train_ids, labels=train_ids).loss
    loss.backward()
    model.zero_grad(set_to_none=True)


step()
res["lora_ms"] = timed(step, 3)
torch.cuda.reset_peak_memory_stats()
step()
res["lora_peak_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
res["fallbacks"] = flash_attn_legacy.fallback_stats()
print(json.dumps(res))
