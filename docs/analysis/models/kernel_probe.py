"""Direct kernel sweep vs fp32 math reference: head_dim x seq len x GQA x causal x q<kv.

Reports cases where the kernel's error is far above SDPA-fp16's error (both vs fp32 reference).
"""
import itertools
import sys

import torch

sys.path.insert(0, "/workspace/analysis/models")
from common import OUT, dump  # noqa: E402
from flash_attn_legacy.flash_attn import flash_attention  # noqa: E402

torch.manual_seed(0)


def ref(q, k, v, causal, scale):
    Hq, Hk = q.shape[1], k.shape[1]
    k = k.repeat_interleave(Hq // Hk, 1).float()
    v = v.repeat_interleave(Hq // Hk, 1).float()
    s = (q.float() @ k.transpose(-1, -2)) * scale
    if causal:
        nq, nk = q.shape[2], k.shape[2]
        m = torch.ones(nq, nk, dtype=torch.bool, device=q.device).tril(nk - nq)
        s = s.masked_fill(~m, float("-inf"))
    return torch.softmax(s, -1) @ v


rows = []
bad = []
dims = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [32, 64, 96, 128, 256]
for d in dims:
    for (Hq, Hk) in [(4, 1), (8, 8), (14, 2)]:
        for nq, nk in [(2, 2), (5, 5), (16, 16), (17, 17), (26, 26), (31, 31), (33, 33), (63, 63), (64, 64), (65, 65),
                       (100, 100), (127, 127), (129, 129), (257, 257), (1000, 1000), (26, 30), (5, 40), (100, 355), (64, 200)]:
            for causal in (True, False):
                scale = d ** -0.5
                q = torch.randn(1, Hq, nq, d, device="cuda", dtype=torch.float16)
                k = torch.randn(1, Hk, nk, d, device="cuda", dtype=torch.float16)
                v = torch.randn(1, Hk, nk, d, device="cuda", dtype=torch.float16)
                r = ref(q, k, v, causal, scale)
                o = flash_attention(q, k, v, softmax_scale=scale, causal=causal).float()
                kk = k.repeat_interleave(Hq // Hk, 1)
                vv = v.repeat_interleave(Hq // Hk, 1)
                m = None
                if causal:
                    m = torch.ones(nq, nk, dtype=torch.bool, device="cuda").tril(nk - nq)
                s = torch.nn.functional.scaled_dot_product_attention(q, kk, vv, attn_mask=m, scale=scale).float()
                ef, es = (o - r).abs().max().item(), (s - r).abs().max().item()
                row = dict(d=d, Hq=Hq, Hk=Hk, nq=nq, nk=nk, causal=causal, err_flash=ef, err_sdpa=es,
                           finite=bool(torch.isfinite(o).all()))
                rows.append(row)
                if not row["finite"] or ef > max(10 * es, 2e-3):
                    bad.append(row)
                    print("BAD", row, flush=True)
print(f"{len(rows)} cases, {len(bad)} bad; max err_flash={max(r['err_flash'] for r in rows):.3e} "
      f"max err_sdpa={max(r['err_sdpa'] for r in rows):.3e}")
dump({"rows": rows, "bad": bad}, f"{OUT}/results/kernel_probe.json")
