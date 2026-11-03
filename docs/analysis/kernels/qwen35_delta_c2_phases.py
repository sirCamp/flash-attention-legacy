"""P6.10 c2 analysis: where our chunk gated delta rule (c1 scan + c3 backward, `gated_delta._chunk_scan`) spends its
time, after c1 / c3. Qwen3.5-2B's shapes: B = 1, H = 16, K = V = 128, chunk 64, fp16 inputs, L = 4096.

1. Inference (no grad): a copy of `_chunk_scan`'s kernel path with CUDA events between its phases, median of 7.
2. Training (forward + backward of the function, the loss a sum of the output): GPU time per autograd node from the
   profiler (`autograd::engine::evaluate_function: <Node>`), and the forward under grad by phase."""
import statistics
import sys

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

from flash_attn_legacy import gated_delta
from flash_attn_legacy.flash_attn import _ops

L = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
H = int(sys.argv[2]) if len(sys.argv) > 2 else 16
reference = gated_delta._decorator_cells(mq.torch_chunk_gated_delta_rule)["torch_function"]
l2norm = reference.__globals__["l2norm"]
print(torch.cuda.get_device_name(), f"L={L} H={H}", flush=True)
NAMES = ["prologue", "decays", "ut+attn mm", "solves", "q/k scale", "scan op", "out mm", "epilogue"]


def phases(query, key, value, g, beta, cs=64):
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(len(NAMES) + 1)]
    ev[0].record()
    B, Lq, _, Kd = key.shape
    Hv, Vd = value.shape[-2:]
    query, key, value, beta, decay = [x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
                                      for x in (query, key, value, beta, g)]
    query, key = l2norm(query, dim=-1, eps=1e-6), l2norm(key, dim=-1, eps=1e-6)
    query = query * Kd ** -0.5
    pad = (cs - Lq % cs) % cs
    query, key, value = (F.pad(x, (0, 0, 0, pad)) for x in (query, key, value))
    beta, decay = (F.pad(x, (0, pad)) for x in (beta, decay))
    v_beta, k_beta = value * beta.unsqueeze(-1), key * beta.unsqueeze(-1)
    query, key, k_beta, v_beta = [x.reshape(B, Hv, -1, cs, x.shape[-1]) for x in (query, key, k_beta, v_beta)]
    decay = decay.reshape(B, Hv, -1, cs)
    ev[1].record()
    mask = torch.ones(cs, cs, dtype=torch.bool, device=query.device).triu(1)
    cum = decay.cumsum(dim=3)
    pw = (cum.unsqueeze(4) - cum.unsqueeze(3)).masked_fill(mask, float("-inf")).exp()
    dkb = k_beta * cum.exp().unsqueeze(-1)
    ev[2].record()
    ut = (k_beta @ key.transpose(-1, -2)) * pw
    intra = (query @ key.transpose(-1, -2)) * pw
    ev[3].record()
    nv = torch.linalg.solve_triangular(ut, v_beta, upper=False, unitriangular=True)
    kc = torch.linalg.solve_triangular(ut, dkb, upper=False, unitriangular=True)
    del ut, pw, dkb, k_beta, v_beta
    ev[4].record()
    state = torch.zeros(B, Hv, Kd, Vd, device=query.device)
    query = query * cum.exp().unsqueeze(-1)
    key = key * (cum[..., -1:] - cum).exp().unsqueeze(-1)
    cd = cum[..., -1].exp()
    ev[5].record()
    states, v_new, state = _ops.gdn_chunk_scan(kc, nv, key, cd, state)
    ev[6].record()
    out = torch.matmul(query, states)
    del states
    out += intra @ v_new
    ev[7].record()
    out = out.reshape(B, Hv, -1, Vd)[:, :, :Lq].transpose(1, 2).to(torch.float16, memory_format=torch.contiguous_format)
    ev[8].record()
    torch.cuda.synchronize()
    return [ev[i].elapsed_time(ev[i + 1]) for i in range(len(NAMES))]


def inputs(grad=False):
    gen = torch.Generator("cuda").manual_seed(0)
    q, k, v = (torch.randn(1, L, H, 128, device="cuda", generator=gen).half() for _ in range(3))
    gate = -torch.rand(1, L, H, device="cuda", generator=gen) * 0.5
    beta = torch.rand(1, L, H, device="cuda", generator=gen).half()
    return [t.requires_grad_(grad) for t in (q, k, v, gate, beta)]


a = inputs()
with torch.no_grad():
    for _ in range(10):  # the T4 ramps its clocks up over the first calls
        phases(*a)
    runs = [phases(*a) for _ in range(7)]
med = [statistics.median(r[i] for r in runs) for i in range(len(NAMES))]
tot = sum(med)
print(f"inference: total {tot:.2f} ms | " + " | ".join(f"{n} {m:.2f} ({m / tot:.0%})" for n, m in zip(NAMES, med)))

# training: the function's forward + backward, GPU time per autograd node
a = inputs(grad=True)


def step():
    o, _ = gated_delta._chunk_scan(reference, *a[:3], g=a[3], beta=a[4], use_qk_l2norm_in_kernel=True)
    o.float().sum().backward()


for _ in range(3):
    step()
torch.cuda.synchronize()
s, m, e = (torch.cuda.Event(enable_timing=True) for _ in range(3))
fw, bw = [], []
for _ in range(5):
    s.record()
    o, _ = gated_delta._chunk_scan(reference, *a[:3], g=a[3], beta=a[4], use_qk_l2norm_in_kernel=True)
    m.record()
    o.float().sum().backward()
    e.record()
    torch.cuda.synchronize()
    fw.append(s.elapsed_time(m))
    bw.append(m.elapsed_time(e))
print(f"training: forward {statistics.median(fw):.2f} ms, backward {statistics.median(bw):.2f} ms")
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
    step()
    torch.cuda.synchronize()
rows = []
for ev_ in prof.key_averages():
    if ev_.key.startswith("autograd::engine::evaluate_function:"):
        t = getattr(ev_, "device_time_total", None) or getattr(ev_, "cuda_time_total", 0)
        rows.append((t / 1000, ev_.key.split(": ", 1)[1], ev_.count))
rows.sort(reverse=True)
btot = sum(r[0] for r in rows)
print(f"backward nodes (GPU ms, of {btot:.2f}):")
for t, k, c in rows[:14]:
    print(f"  {t:7.2f} {t / btot:5.0%} x{c:<3d} {k}")
print("DONE rc=0")
