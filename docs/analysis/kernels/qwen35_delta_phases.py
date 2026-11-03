"""P6.10 (c) analysis: where transformers' reference chunk gated delta rule spends its time, phase by phase (a copy
of torch_chunk_gated_delta_rule with CUDA events between its phases), on Qwen3.5-2B's shapes: B = 1, H = 16 value
heads (key heads repeated), K = V = 128, chunk 64, fp16 inputs; L = 1024 / 4096 / 16384. Also the whole function and
the recurrent reference per token (decode), and the forward + backward of the chunk (training)."""
import statistics

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

from flash_attn_legacy import gated_delta

chunk_ref = gated_delta._decorator_cells(mq.torch_chunk_gated_delta_rule)["torch_function"]
rec_ref = gated_delta._decorator_cells(mq.torch_recurrent_gated_delta_rule)["torch_function"]
print(torch.cuda.get_device_name())


def phases(query, key, value, g, beta, chunk_size=64):
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
    ev[0].record()
    B, L, _, Kd = key.shape
    H, Vd = value.shape[-2:]
    query, key, value, beta, decay = [x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
                                      for x in (query, key, value, beta, g)]
    query, key = mq.l2norm(query, dim=-1, eps=1e-6), mq.l2norm(key, dim=-1, eps=1e-6)
    query = query * Kd ** -0.5
    pad = (chunk_size - L % chunk_size) % chunk_size
    query, key, value = (F.pad(x, (0, 0, 0, pad)) for x in (query, key, value))
    beta, decay = (F.pad(x, (0, pad)) for x in (beta, decay))
    nc = (L + pad) // chunk_size
    v_beta, k_beta = value * beta.unsqueeze(-1), key * beta.unsqueeze(-1)
    query, key, k_beta, v_beta = [x.reshape(B, H, -1, chunk_size, x.shape[-1]) for x in (query, key, k_beta, v_beta)]
    decay = decay.reshape(B, H, -1, chunk_size)
    mask = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(1)
    cum = decay.cumsum(dim=3)
    ev[1].record()  # prologue: casts, l2norm, pads, reshapes
    pw = (cum.unsqueeze(4) - cum.unsqueeze(3)).masked_fill(mask, float("-inf")).exp()
    ut = (k_beta @ key.transpose(-1, -2)) * pw
    intra = (query @ key.transpose(-1, -2)) * pw
    dk = k_beta * cum.exp().unsqueeze(-1)
    nv = torch.linalg.solve_triangular(ut, v_beta, upper=False, unitriangular=True)
    kc = torch.linalg.solve_triangular(ut, dk, upper=False, unitriangular=True)
    ev[2].record()  # intra-chunk phase (parallel over chunks)
    state = torch.zeros(B, H, Kd, Vd, device=query.device)
    out = torch.zeros_like(nv)
    query = query * cum.exp().unsqueeze(-1)
    key = key * (cum[..., -1:] - cum).exp().unsqueeze(-1)
    cd = cum[..., -1].exp()[..., None, None]
    for i in range(nc):
        v_new = nv[:, :, i] - kc[:, :, i] @ state
        out[:, :, i] = query[:, :, i] @ state + intra[:, :, i] @ v_new
        state = state * cd[:, :, i] + key[:, :, i].transpose(-1, -2) @ v_new
    ev[3].record()  # the sequential scan
    out = out.reshape(B, H, -1, Vd)[:, :, :L].transpose(1, 2).to(torch.float16, memory_format=torch.contiguous_format)
    ev[4].record()
    torch.cuda.synchronize()
    return [ev[i].elapsed_time(ev[i + 1]) for i in range(4)]


def inputs(L, H=16, K=128, dtype=torch.float16, grad=False):
    g = torch.Generator("cuda").manual_seed(0)
    q, k = (torch.randn(1, L, H, K, device="cuda", generator=g).to(dtype) for _ in range(2))
    v = torch.randn(1, L, H, K, device="cuda", generator=g).to(dtype)
    gate = -torch.rand(1, L, H, device="cuda", generator=g) * 0.5
    beta = torch.rand(1, L, H, device="cuda", generator=g).to(dtype)
    return [t.requires_grad_(grad) for t in (q, k, v, gate, beta)]


for L in (1024, 4096, 16384):
    args = inputs(L)
    phases(*args)
    runs = [phases(*args) for _ in range(5)]
    med = [statistics.median(r[i] for r in runs) for i in range(4)]
    tot = sum(med)
    print(f"L={L:5d}: total {tot:7.1f} ms | prologue {med[0]:6.1f} ({med[0] / tot:4.0%}) | intra-chunk {med[1]:6.1f} "
          f"({med[1] / tot:4.0%}) | scan {med[2]:6.1f} ({med[2] / tot:4.0%}) | epilogue {med[3]:5.1f}", flush=True)
    if L == 4096:
        a = inputs(L, grad=True)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        o, _ = chunk_ref(*a[:3], g=a[3], beta=a[4], use_qk_l2norm_in_kernel=True)
        s.record()
        o, _ = chunk_ref(*a[:3], g=a[3], beta=a[4], use_qk_l2norm_in_kernel=True)
        m = torch.cuda.Event(enable_timing=True)
        m.record()
        o.float().sum().backward()
        e.record()
        torch.cuda.synchronize()
        print(f"   fwd + bwd at 4096: forward {s.elapsed_time(m):.1f} ms, backward {m.elapsed_time(e):.1f} ms")

state = torch.zeros(1, 16, 128, 128, device="cuda")
a = inputs(1)
for _ in range(3):
    rec_ref(*a[:3], g=a[3], beta=a[4], initial_state=state, output_final_state=True, use_qk_l2norm_in_kernel=True)
torch.cuda.synchronize()
import time  # noqa: E402
t0 = time.perf_counter()
for _ in range(100):
    rec_ref(*a[:3], g=a[3], beta=a[4], initial_state=state, output_final_state=True, use_qk_l2norm_in_kernel=True)
torch.cuda.synchronize()
print(f"recurrent (one decode token, 16 heads): {(time.perf_counter() - t0) * 10:.3f} ms per call (host + GPU)")
print("DONE rc=0")
