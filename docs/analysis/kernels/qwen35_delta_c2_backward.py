"""P6.10 c2 plan check: the analytic backward of the chunk gated delta rule's intra-chunk phase (the prologue's l2norm
and scale included), against autograd of transformers' reference lines, in fp64 on the CPU.

Per chunk (rows i, j of 64 tokens), with qn / kn the l2-normalised q / k (q scaled by K^-1/2) and c the in-chunk
cumulative sum of g:
    P_ij = exp(c_i - c_j) (j <= i, else 0),  M = strict_lower(beta_i kn_i.kn_j P_ij),  T = (I + M)^-1
    u = T (beta v),  w = T (beta e^c kn),  A = (qn kn^T) P (lower, diagonal included)
    q~ = qn e^c,  k~ = kn e^(c_last - c),  d = e^(c_last)
The outputs the scan and the output products take; their gradients are random here."""
import torch

torch.manual_seed(0)
dt = torch.float64
B, L, H, K, V, CS = 2, 150, 3, 16, 12, 64


def l2n(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def prep_reference(q, k, v, g, beta):
    """transformers' lines (torch_chunk_gated_delta_rule, 5.20.2) up to the scan, plus q~ / k~ / d"""
    F = torch.nn.functional
    q, k, v, beta, g = [x.transpose(1, 2).contiguous() for x in (q, k, v, beta, g)]
    q, k = l2n(q), l2n(k)
    q = q * K ** -0.5
    pad = (CS - L % CS) % CS
    q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
    beta, g = (F.pad(x, (0, pad)) for x in (beta, g))
    vb, kb = v * beta.unsqueeze(-1), k * beta.unsqueeze(-1)
    q, k, kb, vb = [x.reshape(B, H, -1, CS, x.shape[-1]) for x in (q, k, kb, vb)]
    g = g.reshape(B, H, -1, CS)
    mask = torch.ones(CS, CS, dtype=torch.bool).triu(1)
    c = g.cumsum(-1)
    pw = (c.unsqueeze(-1) - c.unsqueeze(-2)).masked_fill(mask, float("-inf")).exp()
    ut = (kb @ k.transpose(-1, -2)) * pw
    A = (q @ k.transpose(-1, -2)) * pw
    dkb = kb * c.exp().unsqueeze(-1)
    u = torch.linalg.solve_triangular(ut, vb, upper=False, unitriangular=True)
    w = torch.linalg.solve_triangular(ut, dkb, upper=False, unitriangular=True)
    qt = q * c.exp().unsqueeze(-1)
    kt = k * (c[..., -1:] - c).exp().unsqueeze(-1)
    d = c[..., -1].exp()
    return qt, kt, w, u, d, A


def prep_backward(q, k, v, g, beta, dqt, dkt, dw, du, dd, dA):
    """the analytic backward (the plan's formulas), from the inputs and the outputs' gradients"""
    F = torch.nn.functional
    q, k, v, beta, g = [x.transpose(1, 2) for x in (q, k, v, beta, g)]
    pad = (CS - L % CS) % CS
    q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
    beta, g = (F.pad(x, (0, pad)) for x in (beta, g))
    q, k, v = [x.reshape(B, H, -1, CS, x.shape[-1]) for x in (q, k, v)]
    beta, g = beta.reshape(B, H, -1, CS), g.reshape(B, H, -1, CS)
    rq = torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
    rk = torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    s = K ** -0.5
    qn, kn = q * rq * s, k * rk
    c = g.cumsum(-1)
    ec, el = c.exp(), (c[..., -1:] - c).exp()
    low = torch.ones(CS, CS, dtype=torch.bool).tril()
    P = (c.unsqueeze(-1) - c.unsqueeze(-2)).masked_fill(~low, float("-inf")).exp()
    KK = kn @ kn.transpose(-1, -2)
    M = (beta.unsqueeze(-1) * KK * P).tril(-1)
    T = torch.linalg.inv(torch.eye(CS, dtype=dt) + M)
    vb, kbe = v * beta.unsqueeze(-1), kn * (beta * ec).unsqueeze(-1)
    u, w = T @ vb, T @ kbe
    # Y = T X: dX = T^T dY, dM = -strict_lower(T^T dY Y^T)
    Tt = T.transpose(-1, -2)
    dvb, dkbe = Tt @ du, Tt @ dw
    dM = -(Tt @ (du @ u.transpose(-1, -2) + dw @ w.transpose(-1, -2))).tril(-1)
    dv = dvb * beta.unsqueeze(-1)
    dbeta = (dvb * v).sum(-1)
    kbe_dkbe = (dkbe * kbe).sum(-1)               # beta e^c kn . dkbe
    dkn = dkbe * (beta * ec).unsqueeze(-1)
    dbeta = dbeta + (dkbe * kn).sum(-1) * ec
    dc = kbe_dkbe.clone()
    # M_ij = beta_i KK_ij P_ij (j < i)
    GP = dM * P                                   # strictly lower
    dbeta = dbeta + (GP * KK).sum(-1)
    S = GP * beta.unsqueeze(-1)
    dkn = dkn + S @ kn + S.transpose(-1, -2) @ kn
    E = dM * M                                    # dP_ij P_ij
    # A_ij = qn_i.kn_j P_ij (j <= i)
    R = dA * P
    dqn = R @ kn
    dkn = dkn + R.transpose(-1, -2) @ qn
    A = (qn @ kn.transpose(-1, -2)) * P
    E = E + dA * A
    dc = dc + E.sum(-1) - E.sum(-2)
    # q~ = qn e^c, k~ = kn e^(c_last - c), d = e^(c_last)
    dqn = dqn + dqt * ec.unsqueeze(-1)
    dc = dc + ((qn * ec.unsqueeze(-1)) * dqt).sum(-1)
    dkn = dkn + dkt * el.unsqueeze(-1)
    kdk = ((kn * el.unsqueeze(-1)) * dkt).sum(-1)
    dc = dc - kdk
    dc[..., -1] += kdk.sum(-1) + dd * c[..., -1].exp()
    dg = dc.flip(-1).cumsum(-1).flip(-1)          # cumsum's backward
    # l2norm (and the scale): x^ = x r, dx = r dx^ - r^3 x (x . dx^)
    dq = s * (rq * dqn - rq ** 3 * q * (q * dqn).sum(-1, keepdim=True))
    dk = rk * dkn - rk ** 3 * k * (k * dkn).sum(-1, keepdim=True)
    out = []
    for x in (dq, dk, dv):
        out.append(x.reshape(B, H, -1, x.shape[-1])[:, :, :L].transpose(1, 2))
    for x in (dg, dbeta):
        out.append(x.reshape(B, H, -1)[:, :, :L].transpose(1, 2))
    return out


q, k, v = (torch.randn(B, L, H, n, dtype=dt) for n in (K, K, V))
g = -torch.rand(B, L, H, dtype=dt) * 0.5
beta = torch.rand(B, L, H, dtype=dt)
beta[0, 3, 1] = 0.0  # a zero beta
ins = [t.requires_grad_() for t in (q, k, v, g, beta)]
outs = prep_reference(*ins)
douts = [torch.randn_like(o) for o in outs]
ref = torch.autograd.grad(outs, ins, douts)
ours = prep_backward(*[t.detach() for t in ins], *douts)
worst = 0.0
for name, a, b in zip("q k v g beta".split(), ref, ours):
    err = ((a - b).norm() / a.norm()).item()
    worst = max(worst, err)
    print(f"d{name}: relative error {err:.2e}")
print("PASS" if worst < 1e-10 else "FAIL", f"worst {worst:.2e}")
