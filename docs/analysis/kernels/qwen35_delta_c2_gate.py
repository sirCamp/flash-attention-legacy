"""P6.10 c2 gate (the plan review's HIGH 2): before the kernel, time the c2 contract in PyTorch on the GPU.
- reference: transformers' lines (prologue + intra-chunk phase, up to the scan's inputs) forward, and their autograd
  backward for random gradients of (q~, k~, w, u, d, A);
- prototype: the same outputs with an explicit T = (I + M)^-1 (solve_triangular against I), and the analytic backward
  of `qwen35_delta_c2_backward.py` from the inputs and the saved T / u / w.
Qwen3.5-2B's shapes: B = 1, H = 16, K = V = 128, fp16 inputs, L = 4096; fp32 math. Also the gradients' agreement."""
import statistics
import sys

import torch
import torch.nn.functional as F

L = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
H = int(sys.argv[2]) if len(sys.argv) > 2 else 16
B, K, V, CS = 1, 128, 128, 64
dev = "cuda"
print(torch.cuda.get_device_name(), f"L={L} H={H}", flush=True)


def l2n(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def chunked(q, k, v, g, beta):
    q, k, v, beta, g = [x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
                        for x in (q, k, v, beta, g)]
    pad = (CS - L % CS) % CS
    q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
    beta, g = (F.pad(x, (0, pad)) for x in (beta, g))
    q, k, v = [x.reshape(B, H, -1, CS, x.shape[-1]) for x in (q, k, v)]
    return q, k, v, g.reshape(B, H, -1, CS), beta.reshape(B, H, -1, CS)


def reference(q, k, v, g, beta):
    q, k, v, beta, g = [x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
                        for x in (q, k, v, beta, g)]
    q, k = l2n(q), l2n(k)
    q = q * K ** -0.5
    pad = (CS - L % CS) % CS
    q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
    beta, g = (F.pad(x, (0, pad)) for x in (beta, g))
    vb, kb = v * beta.unsqueeze(-1), k * beta.unsqueeze(-1)
    q, k, kb, vb = [x.reshape(B, H, -1, CS, x.shape[-1]) for x in (q, k, kb, vb)]
    g = g.reshape(B, H, -1, CS)
    mask = torch.ones(CS, CS, dtype=torch.bool, device=dev).triu(1)
    c = g.cumsum(-1)
    pw = (c.unsqueeze(-1) - c.unsqueeze(-2)).masked_fill(mask, float("-inf")).exp()
    ut = (kb @ k.transpose(-1, -2)) * pw
    A = (q @ k.transpose(-1, -2)) * pw
    dkb = kb * c.exp().unsqueeze(-1)
    u = torch.linalg.solve_triangular(ut, vb, upper=False, unitriangular=True)
    w = torch.linalg.solve_triangular(ut, dkb, upper=False, unitriangular=True)
    qt = q * c.exp().unsqueeze(-1)
    kt = k * (c[..., -1:] - c).exp().unsqueeze(-1)
    return qt, kt, w, u, c[..., -1].exp(), A


EYE = torch.eye(CS, device=dev)
LOW = torch.ones(CS, CS, dtype=torch.bool, device=dev).tril()


def proto_forward(q, k, v, g, beta):
    q, k, v, g, beta = chunked(q, k, v, g, beta)
    qn, kn = l2n(q) * K ** -0.5, l2n(k)
    c = g.cumsum(-1)
    P = (c.unsqueeze(-1) - c.unsqueeze(-2)).masked_fill(~LOW, float("-inf")).exp()
    M = (beta.unsqueeze(-1) * (kn @ kn.transpose(-1, -2)) * P).tril(-1)
    T = torch.linalg.solve_triangular(M + EYE, EYE.expand_as(M), upper=False, unitriangular=True)
    ec = c.exp()
    u, w = T @ (v * beta.unsqueeze(-1)), T @ (kn * (beta * ec).unsqueeze(-1))
    A = (qn @ kn.transpose(-1, -2)) * P
    return qn * ec.unsqueeze(-1), kn * (c[..., -1:] - c).exp().unsqueeze(-1), w, u, c[..., -1].exp(), A, T


def proto_backward(q, k, v, g, beta, T, u, w, dqt, dkt, dw, du, dd, dA):
    q, k, v, g, beta = chunked(q, k, v, g, beta)
    rq = torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
    rk = torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    s = K ** -0.5
    qn, kn = q * rq * s, k * rk
    c = g.cumsum(-1)
    ec, el = c.exp(), (c[..., -1:] - c).exp()
    P = (c.unsqueeze(-1) - c.unsqueeze(-2)).masked_fill(~LOW, float("-inf")).exp()
    KK = kn @ kn.transpose(-1, -2)
    Tt = T.transpose(-1, -2)
    dvb, dkbe = Tt @ du, Tt @ dw
    dM = -(Tt @ (du @ u.transpose(-1, -2) + dw @ w.transpose(-1, -2))).tril(-1)
    dv = dvb * beta.unsqueeze(-1)
    dbeta = (dvb * v).sum(-1) + (dkbe * kn).sum(-1) * ec
    dc = (dkbe * kn).sum(-1) * beta * ec
    dkn = dkbe * (beta * ec).unsqueeze(-1)
    GP = dM * P
    dbeta += (GP * KK).sum(-1)
    S = GP * beta.unsqueeze(-1)
    dkn += S @ kn + S.transpose(-1, -2) @ kn
    QK = qn @ kn.transpose(-1, -2)
    R = dA * P
    E = GP * KK * beta.unsqueeze(-1) + R * QK
    dqn = R @ kn + dqt * ec.unsqueeze(-1)
    dkn += R.transpose(-1, -2) @ qn + dkt * el.unsqueeze(-1)
    dc += E.sum(-1) - E.sum(-2) + (qn * dqt).sum(-1) * ec
    kdk = (kn * dkt).sum(-1) * el
    dc -= kdk
    dc[..., -1] += kdk.sum(-1) + dd * c[..., -1].exp()
    dg = dc.flip(-1).cumsum(-1).flip(-1)
    dq = s * (rq * dqn - rq ** 3 * q * (q * dqn).sum(-1, keepdim=True))
    dk = rk * dkn - rk ** 3 * k * (k * dkn).sum(-1, keepdim=True)
    out = [x.reshape(B, H, -1, x.shape[-1])[:, :, :L].transpose(1, 2) for x in (dq, dk, dv)]
    out += [x.reshape(B, H, -1)[:, :, :L].transpose(1, 2) for x in (dg, dbeta)]
    return out


gen = torch.Generator(dev).manual_seed(0)
q, k, v = (torch.randn(B, L, H, n, device=dev, generator=gen).half() for n in (K, K, V))
g = -torch.rand(B, L, H, device=dev, generator=gen) * 0.5
beta = torch.rand(B, L, H, device=dev, generator=gen).half()
ins = [t.requires_grad_() for t in (q, k, v, g, beta)]
with torch.no_grad():
    outs = reference(*ins)
douts = [torch.randn(o.shape, device=dev, generator=gen) for o in outs]


def timed(fn, n=7):
    for _ in range(5):
        fn()
    ts = []
    for _ in range(n):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def ref_fwd_bwd():
    o = reference(*ins)
    return torch.autograd.grad(o, ins, douts)


with torch.no_grad():
    t_ref_fwd = timed(lambda: reference(*ins))
    t_proto_fwd = timed(lambda: proto_forward(*ins))
    qt, kt, w, u, d, A, T = proto_forward(*ins)
    t_proto_bwd = timed(lambda: proto_backward(*[x.detach() for x in ins], T, u, w, *douts))
t_ref_fwd_bwd = timed(ref_fwd_bwd)
print(f"reference: forward {t_ref_fwd:.2f} ms, backward (autograd) {t_ref_fwd_bwd - t_ref_fwd:.2f} ms")
print(f"prototype: forward (explicit T) {t_proto_fwd:.2f} ms, analytic backward {t_proto_bwd:.2f} ms")
ref = ref_fwd_bwd()
with torch.no_grad():
    ours = proto_backward(*[x.detach() for x in ins], T, u, w, *douts)
for name, a, b in zip("q k v g beta".split(), ref, ours):
    print(f"  d{name}: relative {((a.float() - b.float()).norm() / a.float().norm()).item():.1e}")
torch.cuda.reset_peak_memory_stats()
base = torch.cuda.memory_allocated()
o = reference(*ins)
print(f"reference graph after forward: {(torch.cuda.memory_allocated() - base) / 2**20:.0f} MiB")
del o
print("DONE rc=0")
