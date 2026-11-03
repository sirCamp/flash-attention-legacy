"""P6.10 b: the conv alone, ours (torch.ops.flash_attn_legacy.causal_conv1d*) vs transformers' reference, on the
model's layout (Qwen3.5-2B: C = 6144, W = 4, SiLU, fp16), CUDA events, median; the floor is one read of x and one
write of y at the nominal bandwidth (forward), dy and x read and dx written (backward, frozen weight; with dW one
more read of dy and x is not needed: fused). Plus the decode update: GPU time and host time per call."""
import statistics
import time

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

import flash_attn_legacy  # noqa: F401
from flash_attn_legacy import gated_delta

ops = torch.ops.flash_attn_legacy
ref_fn = gated_delta._decorator_cells(mq.causal_conv1d_fn)["torch_function"]
ref_upd = gated_delta._decorator_cells(mq.causal_conv1d_update)["torch_function"]
bw = {"Tesla P100-PCIE-16GB": 732, "Tesla V100-SXM2-32GB": 900, "Tesla T4": 320,
      "NVIDIA GeForce GTX 1080 Ti": 484, "NVIDIA GeForce RTX 2080 Ti": 616}.get(torch.cuda.get_device_name(), 0)
print(torch.cuda.get_device_name(), f"nominal {bw} GB/s")
C, W = 6144, 4


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


w = (torch.randn(C, W, device="cuda") * 0.5).half()
print(f"{'B':>2} {'L':>6} | {'fwd ref':>8} {'fwd ours':>8} {'x':>6} {'vs floor':>8} | {'f+b ref':>8} {'f+b ours':>8} "
      f"{'x':>6} | {'f+b+dW ours':>11}")
for B, L in ((1, 1024), (1, 4096), (4, 2048), (1, 16384)):
    x = torch.randn(B, L, C, device="cuda", dtype=torch.float16).transpose(1, 2)
    floor = 2 * x.numel() * 2 / (bw * 1e9) * 1e3
    fr = timed(lambda: ref_fn(x, w, None, activation="silu"))
    fo = timed(lambda: ops.causal_conv1d(x, w, None, True))
    xg = x.detach().requires_grad_(True)
    dy = torch.randn(B, L, C, device="cuda", dtype=torch.float16).transpose(1, 2)
    wt = w.clone().requires_grad_(True)

    def fb(fn, weight):
        fn(xg, weight).backward(dy)

    br = timed(lambda: fb(lambda a, b: ref_fn(a, b, None, activation="silu"), w), 10, 3)
    bo = timed(lambda: fb(lambda a, b: ops.causal_conv1d(a, b, None, True), w), 10, 3)
    bow = timed(lambda: fb(lambda a, b: ops.causal_conv1d(a, b, None, True), wt), 10, 3)
    print(f"{B:>2} {L:>6} | {fr:7.3f}ms {fo:7.3f}ms {fr / fo:5.1f}x {fo / floor:7.2f}x | {br:7.3f}ms {bo:7.3f}ms "
          f"{br / bo:5.1f}x | {bow:10.3f}ms", flush=True)

for B in (1, 8):
    x = torch.randn(B, 1, C, device="cuda", dtype=torch.float16).transpose(1, 2)
    res = {}
    for name, fn in (("ref", lambda s: ref_upd(x, s, w, None, "silu")),
                     ("ours", lambda s: ops.causal_conv1d_update(x, s, w, None, True))):
        state = torch.zeros(B, C, W, device="cuda", dtype=torch.float16)
        gpu = timed(lambda: fn(state), 50, 5)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            fn(state)
        host = (time.perf_counter() - t0) / 200 * 1e3
        torch.cuda.synchronize()
        res[name] = (gpu, host)
    print(f"update B={B}: ref {res['ref'][0] * 1e3:.1f} us GPU / {res['ref'][1] * 1e3:.1f} us host, ours "
          f"{res['ours'][0] * 1e3:.1f} us / {res['ours'][1] * 1e3:.1f} us")
print("DONE rc=0")
