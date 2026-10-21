"""Minimal repro: flash_attn_legacy_forward silently ignores `softcap` (Gemma-2) and `s_aux`
(gpt-oss attention sinks), on BOTH the kernel path (mask=None) and the SDPA fallback path (bool mask).
No error, no warning. Compare with an fp32 reference that honours the feature.
"""
import warnings

import torch

import flash_attn_legacy  # noqa: F401
from flash_attn_legacy.integrations import flash_attn_legacy_forward

torch.manual_seed(0)


class Mod(torch.nn.Module):
    is_causal = True
    training = False


B, Hq, Hk, N, d = 1, 8, 4, 256, 256
q = torch.randn(B, Hq, N, d, device="cuda", dtype=torch.float16) * 2
k = torch.randn(B, Hk, N, d, device="cuda", dtype=torch.float16) * 2
v = torch.randn(B, Hk, N, d, device="cuda", dtype=torch.float16)
scale = d ** -0.5
causal = torch.ones(N, N, dtype=torch.bool, device="cuda").tril()


def ref(softcap=None, sinks=None):
    kk, vv = k.repeat_interleave(Hq // Hk, 1).float(), v.repeat_interleave(Hq // Hk, 1).float()
    s = q.float() @ kk.transpose(-1, -2) * scale
    if softcap:
        s = torch.tanh(s / softcap) * softcap
    s = s.masked_fill(~causal, float("-inf"))
    if sinks is not None:
        s = torch.cat([s, sinks.float().view(1, -1, 1, 1).expand(B, Hq, N, 1)], -1)
        p = torch.softmax(s, -1)[..., :-1]
    else:
        p = torch.softmax(s, -1)
    return (p @ vv).transpose(1, 2)


def rel(a, b):
    return ((a.float() - b).norm() / b.norm()).item()


sinks = torch.randn(Hq, device="cuda", dtype=torch.float16) * 2
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    for path, mask in (("kernel (mask=None)", None), ("fallback (bool mask)", causal[None, None])):
        base = flash_attn_legacy_forward(Mod(), q, k, v, mask, scaling=scale)[0]
        print(f"[{path}] no feature      : rel err vs ref = {rel(base, ref()):.2e}   (noise floor)")
        o = flash_attn_legacy_forward(Mod(), q, k, v, mask, scaling=scale, softcap=20.0)[0]
        print(f"[{path}] softcap=20      : rel err vs softcapped ref = {rel(o, ref(softcap=20.0)):.2e}; "
              f"identical to no-softcap output: {torch.equal(o, base)}")
        o = flash_attn_legacy_forward(Mod(), q, k, v, mask, scaling=scale, s_aux=sinks)[0]
        print(f"[{path}] s_aux=sinks     : rel err vs sink ref = {rel(o, ref(sinks=sinks)):.2e}; "
              f"identical to no-sink output: {torch.equal(o, base)}")
    print("warnings raised:", [str(x.message) for x in w])
