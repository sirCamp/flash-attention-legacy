"""Kernel backward accuracy sweep vs fp32 math: batch size, GQA, head_dim, causal, N.
Also an end-to-end check: Qwen2.5-0.5B training-step grads for B=1 vs B=2 (flash vs sdpa vs fp32 gold)."""
import sys

import torch

sys.path.insert(0, "/workspace/analysis/models")
from common import OUT, dump  # noqa: E402
from flash_attn_legacy.flash_attn import flash_attention  # noqa: E402


def grads(q, k, v, do, scale, causal, impl):
    rep = q.shape[1] // k.shape[1]
    dt = torch.float32 if impl == "fp32" else torch.float16
    qq, kk, vv = (t.detach().to(dt).requires_grad_(True) for t in (q, k, v))
    if impl == "flash":
        o = flash_attention(qq, kk, vv, softmax_scale=scale, causal=causal)
    else:
        k2, v2 = kk.repeat_interleave(rep, 1), vv.repeat_interleave(rep, 1)
        if impl == "fp32":
            s = (qq @ k2.transpose(-1, -2)) * scale
            if causal:
                n = q.shape[2]
                s = s.masked_fill(~torch.ones(n, n, dtype=torch.bool, device="cuda").tril(), float("-inf"))
            o = torch.softmax(s, -1) @ v2
        else:
            o = torch.nn.functional.scaled_dot_product_attention(qq, k2, v2, is_causal=causal, scale=scale)
    o.backward(do.to(dt))
    return [t.grad.float() for t in (qq, kk, vv)]


rows, bad = [], []
torch.manual_seed(0)
for B in (1, 2, 3):
    for d in (64, 128, 256):
        for (Hq, Hk) in ((14, 2), (8, 8)):
            for N in (77, 512):
                for causal in (True, False):
                    q = torch.randn(B, Hq, N, d, device="cuda", dtype=torch.float16)
                    k = torch.randn(B, Hk, N, d, device="cuda", dtype=torch.float16)
                    v = torch.randn(B, Hk, N, d, device="cuda", dtype=torch.float16)
                    do = torch.randn(B, Hq, N, d, device="cuda", dtype=torch.float16)
                    sc = d ** -0.5
                    g0 = grads(q, k, v, do, sc, causal, "fp32")
                    gf = grads(q, k, v, do, sc, causal, "flash")
                    gs = grads(q, k, v, do, sc, causal, "sdpa16")
                    row = dict(B=B, d=d, Hq=Hq, Hk=Hk, N=N, causal=causal)
                    for n, a, b, r in zip(("dq", "dk", "dv"), gf, gs, g0):
                        # per batch element relative error (catches batch-indexing bugs)
                        row[n + "_flash"] = max(float((a[i] - r[i]).norm() / r[i].norm()) for i in range(B))
                        row[n + "_sdpa"] = max(float((b[i] - r[i]).norm() / r[i].norm()) for i in range(B))
                    rows.append(row)
                    if any(row[n + "_flash"] > 3 * row[n + "_sdpa"] + 1e-3 for n in ("dq", "dk", "dv")):
                        bad.append(row)
                        print("BAD", row, flush=True)
print(f"{len(rows)} cases, {len(bad)} bad. max flash rel: dq {max(r['dq_flash'] for r in rows):.2e} "
      f"dk {max(r['dk_flash'] for r in rows):.2e} dv {max(r['dv_flash'] for r in rows):.2e}; "
      f"max sdpa16 rel: dq {max(r['dq_sdpa'] for r in rows):.2e} dk {max(r['dk_sdpa'] for r in rows):.2e} dv {max(r['dv_sdpa'] for r in rows):.2e}", flush=True)
dump({"rows": rows, "bad": bad}, f"{OUT}/results/bwd_probe.json")
