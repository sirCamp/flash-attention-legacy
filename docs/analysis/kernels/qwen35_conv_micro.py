"""Qwen3.5 causal conv1d alone on one GPU (P6.10 b): transformers' reference causal_conv1d_fn (F.conv1d depthwise,
padding then slice, SiLU) on the layout the model passes (mixed_qkv [B, L, C] transposed to [B, C, L]: channel-last
memory), forward and forward + backward (dx; weight frozen as under LoRA, and weight trainable), against the bandwidth
floor (read x once, write y once; backward: read dy and x, write dx). Also the decode update (one token, conv state
[B, C, 4]).

Usage: python docs/analysis/kernels/qwen35_conv_micro.py
"""
import statistics

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

C, W = 6144, 4  # Qwen3.5-2B: conv dim 6144 (2 x 2048 key + 2048 value), kernel 4
dev = "cuda"
bw = {"Tesla P100-PCIE-16GB": 732, "Tesla V100-SXM2-32GB": 900, "Tesla T4": 320,
      "NVIDIA GeForce GTX 1080 Ti": 484, "NVIDIA GeForce RTX 2080 Ti": 616}.get(torch.cuda.get_device_name(), 0)
print(torch.cuda.get_device_name(), f"nominal {bw} GB/s")
ref = mq.causal_conv1d_fn.__wrapped__ if hasattr(mq.causal_conv1d_fn, "__wrapped__") else mq.causal_conv1d_fn


def timed(fn, iters=20, reps=5):
    out = []
    for _ in range(reps):
        fn()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e) / iters)
    return statistics.median(out)


weight = (torch.randn(C, W, device=dev) * 0.5).half()
for B, L in ((1, 1024), (1, 4096), (4, 2048), (1, 16384)):
    x = torch.randn(B, L, C, device=dev, dtype=torch.float16).transpose(1, 2)  # as the model: channel-last memory
    floor = 2 * x.numel() * 2 / (bw * 1e9) * 1e3 if bw else float("nan")
    f = timed(lambda: ref(x, weight, None, activation="silu"))
    xg = x.detach().requires_grad_(True)
    dy = torch.randn(B, C, L, device=dev, dtype=torch.float16)

    def fb(wt):
        y = ref(xg, wt, None, activation="silu")
        y.backward(dy)
    wtr = weight.clone().requires_grad_(True)
    fb_frozen = timed(lambda: fb(weight), 10, 3)
    fb_train = timed(lambda: fb(wtr), 10, 3)
    print(f"B={B} L={L:5d}: fwd {f:7.3f} ms (floor {floor:.3f}, {f / floor:5.1f}x) | fwd+bwd frozen W {fb_frozen:7.3f} ms, "
          f"trainable W {fb_train:7.3f} ms (floor ~{2.5 * floor:.3f})", flush=True)

upd = mq.causal_conv1d_update.__wrapped__ if hasattr(mq.causal_conv1d_update, "__wrapped__") else mq.causal_conv1d_update
for B in (1, 8):
    x = torch.randn(B, 1, C, device=dev, dtype=torch.float16).transpose(1, 2)
    state = torch.randn(B, C, W, device=dev, dtype=torch.float16)
    t = timed(lambda: upd(x, state, weight, None, "silu"), 50, 5)
    import time
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(200):
        upd(x, state, weight, None, "silu")
    host = (time.perf_counter() - t0) / 200 * 1e3
    torch.cuda.synchronize()
    print(f"update B={B}: GPU {t * 1e3:.1f} us per call, host {host * 1e3:.1f} us per call (launch-bound)")
print("conv_micro done")
