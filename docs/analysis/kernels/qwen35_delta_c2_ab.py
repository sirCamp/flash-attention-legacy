"""P6.10 c2 A/B of the chunk gated delta rule itself (`gated_delta._chunk_scan`), in one process, the arms alternated
per round: A = FLASH_ATTN_LEGACY_GDN_PREP=0 (c3: the prologue and intra-chunk phase as PyTorch lines), B = the prep op.
Qwen3.5-2B's shapes (B = 1, H = 16, K = V = 128, fp16 inputs) at L = 4096: an inference call, and a training forward +
backward (the output's sum); medians of 4 rounds x 5 calls, and the peak memory of each."""
import os
import statistics
import sys

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

from flash_attn_legacy import gated_delta

L = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
H = int(sys.argv[2]) if len(sys.argv) > 2 else 16
reference = gated_delta._decorator_cells(mq.torch_chunk_gated_delta_rule)["torch_function"]
print(torch.cuda.get_device_name(), f"L={L} H={H}", flush=True)
gen = torch.Generator("cuda").manual_seed(0)
q, k, v = (torch.randn(1, L, H, 128, device="cuda", generator=gen).half() for _ in range(3))
g = -torch.rand(1, L, H, device="cuda", generator=gen) * 0.5
beta = torch.rand(1, L, H, device="cuda", generator=gen).half()
train = [t.clone().requires_grad_(True) for t in (q, k, v, g, beta)]


def infer():
    with torch.no_grad():
        gated_delta._chunk_scan(reference, q, k, v, g, beta, use_qk_l2norm_in_kernel=True)


def step():
    o, _ = gated_delta._chunk_scan(reference, *train, use_qk_l2norm_in_kernel=True)
    o.float().sum().backward()


def run(fn, n=5):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n


def peak(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20


times = {(arm, kind): [] for arm in "AB" for kind in ("infer", "train")}
mem = {}
for arm in "AB":  # warm-up (the T4's clocks ramp up) and the peaks
    os.environ["FLASH_ATTN_LEGACY_GDN_PREP"] = "0" if arm == "A" else "1"
    for _ in range(10):
        infer()
    step()
    mem[arm] = (peak(infer), peak(step))
for arm in "ABBAABBA":
    os.environ["FLASH_ATTN_LEGACY_GDN_PREP"] = "0" if arm == "A" else "1"
    times[arm, "infer"].append(run(infer))
    times[arm, "train"].append(run(step))
for kind in ("infer", "train"):
    a, b = statistics.median(times["A", kind]), statistics.median(times["B", kind])
    print(f"{kind}: A (c3) {a:.2f} ms [{min(times['A', kind]):.2f}-{max(times['A', kind]):.2f}], B (c2) {b:.2f} ms "
          f"[{min(times['B', kind]):.2f}-{max(times['B', kind]):.2f}], {a / b:.2f}x")
print(f"peak MiB: inference A {mem['A'][0]:.0f} B {mem['B'][0]:.0f}; training A {mem['A'][1]:.0f} B {mem['B'][1]:.0f}")
print("DONE rc=0")
