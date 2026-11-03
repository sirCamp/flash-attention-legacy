"""P8.10 first step (docs/analysis/kernels/p8_followups.md 2b.2): would dW = dY^T X on HFMA2 lanes overflow or underflow
on a real full fine-tune? Qwen2.5-0.5B, FP32 weights, fp16 autocast, GradScaler (its dynamic scale, after the first
skips settle), 2048 tokens of wikitext-2. For every nn.Linear's backward: dY (scaled, fp16) and X (fp16 cast); per layer
  - the largest single product |dY[m, n] X[m, k]| (exact: per token, max_n |dY| * max_k |X|),
  - the largest 32-term lane partial over tokens |sum_{m in chunk} dY[m, n] X[m, k]| (chunks of 32 tokens, the kernel's
    fp16 chain along the reduction; computed in fp32),
  - the share of products whose magnitude is below fp16's smallest normal (2^-14) among the nonzero ones, and of
    dY values below it,
against fp16's max 65504.
Usage: python docs/analysis/kernels/p8_dw_hist.py STEPS"""
import sys

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import flash_attn_legacy  # noqa: F401

name, steps = "Qwen/Qwen2.5-0.5B", int(sys.argv[1]) if len(sys.argv) > 1 else 6
tok = AutoTokenizer.from_pretrained(name)
text = "\n\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"][:4000]
                   if t.strip())
ids = torch.tensor(tok(text)["input_ids"][: 2048 * steps]).view(steps, 1, 2048).cuda()
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32, attn_implementation="flash_attn_legacy").cuda()
model.gradient_checkpointing_enable()
opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
scaler = torch.amp.GradScaler("cuda")
FP16_MAX, TINY = 65504.0, 2.0 ** -14
stats = {}
inputs = {}
record = [False]


def fwd_hook(mod, args, out):
    if record[0] and torch.is_grad_enabled():
        inputs[mod] = args[0].detach().to(torch.float16).reshape(-1, args[0].shape[-1])


def bwd_hook(mod, grad_in, grad_out):
    if not record[0] or mod not in inputs:
        return
    x = inputs[mod].float()                                  # [M, K]
    dy = grad_out[0].detach().reshape(-1, grad_out[0].shape[-1]).float()  # [M, N], scaled
    M = x.shape[0]
    single = (dy.abs().amax(1) * x.abs().amax(1)).max().item()
    lane = 0.0
    for m0 in range(0, M, 32):
        lane = max(lane, (dy[m0:m0 + 32].t() @ x[m0:m0 + 32]).abs().max().item())
    nz_dy = dy[dy != 0].abs()
    # products below 2^-14: estimated from a 64 x 64 sample of (n, k) per token block
    n_idx = torch.randint(0, dy.shape[1], (64,), device=dy.device)
    k_idx = torch.randint(0, x.shape[1], (64,), device=x.device)
    prods = (dy[:, n_idx, None] * x[:, None, k_idx]).abs()
    prods = prods[prods != 0]
    s = stats.setdefault(mod.qualname, {"single": 0.0, "lane": 0.0, "tiny_prod": [], "tiny_dy": []})
    s["single"] = max(s["single"], single)
    s["lane"] = max(s["lane"], lane)
    s["tiny_prod"].append((prods < TINY).float().mean().item() if prods.numel() else 0.0)
    s["tiny_dy"].append((nz_dy < TINY).float().mean().item() if nz_dy.numel() else 0.0)


for n, m in model.named_modules():
    if isinstance(m, nn.Linear):
        m.qualname = n
        m.register_forward_hook(fwd_hook)
        m.register_full_backward_hook(bwd_hook)

model.train()
scales = []
for step in range(steps):
    record[0] = step >= 2  # the first steps: GradScaler's scale settles (skips)
    with torch.autocast("cuda", dtype=torch.float16):
        loss = model(input_ids=ids[step], labels=ids[step]).loss
    scaler.scale(loss).backward()
    scales.append(scaler.get_scale())
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
    inputs.clear()
print("GradScaler scales per step:", scales)
rows = sorted(stats.items(), key=lambda kv: -kv[1]["lane"])
print(f"{'layer':45s} {'max |dY*X|':>11s} {'max 32-lane':>11s} {'prod<2^-14':>10s} {'dY<2^-14':>9s}")
for n, s in rows[:12] + rows[-3:]:
    print(f"{n:45s} {s['single']:11.3g} {s['lane']:11.3g} {sum(s['tiny_prod']) / len(s['tiny_prod']):10.2%} "
          f"{sum(s['tiny_dy']) / len(s['tiny_dy']):9.2%}")
over = [n for n, s in rows if s["lane"] > FP16_MAX]
print(f"layers with a 32-term lane above 65504: {len(over)} of {len(rows)}; max lane {rows[0][1]['lane']:.3g}; "
      f"max single product {max(s['single'] for s in stats.values()):.3g}")
tp = [x for s in stats.values() for x in s["tiny_prod"]]
print(f"products below 2^-14 (subnormal / lost): median layer {sorted(tp)[len(tp) // 2]:.2%}, max {max(tp):.2%}")
