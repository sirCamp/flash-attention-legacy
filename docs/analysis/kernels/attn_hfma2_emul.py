"""P8.7: the emulation gate of an HFMA2 attention forward on the P100 (docs/analysis/kernels/p8_followups.md, 3.0).

Emulated (bit-faithful to the arithmetic, not to the tiling):
  S   Q prescaled by softmax_scale * log2(e) and rounded to fp16; for every 64 dims of d (a chunk) two fp16 lanes (even
      dims, odd dims), each a chain of 32 products: the first an fp16 multiply, then fp16 FMAs (single rounding: the
      product and sum exact in fp64, then rounded to fp16); at the end of the chunk both lanes widened and added in fp32
      (the fold), accumulated over chunks in fp32. Log2 domain.
  P   online softmax over key blocks of BC keys in fp32 (running max, alpha = exp2(m_old - m_new), l summed in fp32
      from the fp32 p), p rounded to fp16 for PV.
  PV  per output column, an fp16 chain over the BC keys of a block (fp16(p) * v: first a multiply, then FMAs), folded
      into the fp32 O with alpha once per block: "PV along output columns, half2(p, p)"; O = O / l at the end.
Note: fp64 -> fp16 goes through fp32 in torch (a double rounding on exact fp32 / fp16 midpoints only: rare, ignored).

Modes:
  tensors MODEL   q / k / v captured from the model on wikitext-2 (2 sequences x 1024): per layer the error of S, P and
                  O vs an fp64 reference, for the emulation, SDPA fp16 and flash_attn_legacy (today's fp32 path); the
                  largest |fp16 lane| seen (S and PV chains)
  model MODEL     top-1 agreement with the FP32 model and perplexity (16 x 1024 tokens, P8's --accuracy setting) of the
                  FP16 model with SDPA, flash_attn_legacy and the emulated attention

Usage: python docs/analysis/kernels/attn_hfma2_emul.py {tensors|model} HF_ID [BC]
"""
import math
import sys

import torch
import torch.nn.functional as F

import os

LOG2E = 1.4426950408889634
# Which parts are emulated in fp16: "s", "pv" or "s,pv" (the others in fp32, exact up to fp32 rounding)
PARTS = os.environ.get("EMUL_PARTS", "s,pv").split(",")
# S variants (exact in real arithmetic): K centered per head (K - mean over the keys: every row of S shifts by the
# constant q.c, softmax unchanged), and Q prescaled by a power of 2 only (exact; the rest of scale * log2(e) applied
# in fp32 after the fold)
K_CENTER = os.environ.get("EMUL_K_CENTER") == "1"
POW2 = os.environ.get("EMUL_POW2") == "1"
CHUNK = 64
STATS = {"s_lane": 0.0, "pv_lane": 0.0}


def r16(x):
    return x.to(torch.float16).to(torch.float64)


def emul_scores(q, k, scale):
    """q [H, L, d], k [H, Lk, d] fp16 -> S [H, L, Lk] fp32, log2 domain."""
    H, L, d = q.shape
    if "s" not in PARTS:
        return (q.float() @ k.float().transpose(1, 2)) * (scale * LOG2E)
    if K_CENTER:
        k = (k.float() - k.float().mean(1, keepdim=True)).half()
    f = scale * LOG2E
    p2 = 2.0 ** math.floor(math.log2(f)) if POW2 else f
    qs = (q.float() * p2).half().double()
    kd = k.double()
    S = torch.zeros(H, L, k.shape[1], dtype=torch.float32, device=q.device)
    for c0 in range(0, d, CHUNK):
        acc = None
        for t in range(min(CHUNK, d - c0) // 2):
            i = c0 + 2 * t
            prod = torch.stack([qs[:, :, i, None] * kd[:, None, :, i], qs[:, :, i + 1, None] * kd[:, None, :, i + 1]])
            acc = r16(prod) if acc is None else r16(acc + prod)
        STATS["s_lane"] = max(STATS["s_lane"], acc.abs().max().item())
        S += acc[0].float() + acc[1].float()
    return S * (f / p2) if POW2 else S


def emul_attention(q, k, v, scale, bc):
    """q [H, L, d], k / v [H, Lk, d] fp16, causal (queries aligned to the last keys) -> O [H, L, d] fp32."""
    H, L, d = q.shape
    Lk = k.shape[1]
    S = emul_scores(q, k, scale)
    qi = torch.arange(L, device=q.device)[:, None] + (Lk - L)
    S = S.masked_fill(torch.arange(Lk, device=q.device)[None, :] > qi, float("-inf"))
    m = torch.full((H, L, 1), float("-inf"), device=q.device)
    l = torch.zeros(H, L, 1, device=q.device)
    O = torch.zeros(H, L, d, device=q.device)
    vd = v.double()
    for j0 in range(0, Lk, bc):
        Sb = S[:, :, j0:j0 + bc]
        m_new = torch.maximum(m, Sb.amax(-1, keepdim=True))
        live = torch.isfinite(m_new)
        m_safe = torch.where(live, m_new, torch.zeros_like(m_new))
        alpha = torch.where(live, torch.exp2(m - m_safe), torch.ones_like(m))
        p = torch.exp2(Sb - m_safe)  # fp32; masked keys: 0
        l = l * alpha + p.sum(-1, keepdim=True)
        if "pv" not in PARTS:
            O = O * alpha + p @ v[:, j0:j0 + bc].float()
            m = m_new
            continue
        p16 = p.half().double()
        acc = None
        for t in range(Sb.shape[-1]):
            prod = p16[:, :, t, None] * vd[:, None, j0 + t, :]
            acc = r16(prod) if acc is None else r16(acc + prod)
        STATS["pv_lane"] = max(STATS["pv_lane"], acc.abs().max().item())
        O = O * alpha + acc.float()
        m = m_new
    return O / l


def ref_attention(q, k, v, scale):
    """fp64 reference: the exact attention of the fp16 inputs."""
    L, Lk = q.shape[1], k.shape[1]
    S = (q.double() @ k.double().transpose(1, 2)) * scale
    qi = torch.arange(L, device=q.device)[:, None] + (Lk - L)
    S = S.masked_fill(torch.arange(Lk, device=q.device)[None, :] > qi, float("-inf"))
    return torch.softmax(S, -1) @ v.double(), S * LOG2E


def _expand_kv(module, key, value):
    g = module.num_key_value_groups
    return key.repeat_interleave(g, 1), value.repeat_interleave(g, 1)


BC = 64


def hfma2_emul_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """HF attention interface: query [B, H, L, d] -> ([B, L, H, d], None). Causal, no padding (batch rows whole)."""
    key, value = _expand_kv(module, key, value)
    scale = scaling if scaling is not None else query.shape[-1] ** -0.5
    out = torch.stack([emul_attention(query[b], key[b], value[b], scale, BC) for b in range(query.shape[0])])
    return out.to(query.dtype).transpose(1, 2).contiguous(), None


def relf(a, ref):
    a, ref = a.double(), ref.double()
    return ((a - ref).norm() / ref.norm()).item()


def load_ids(name, seqs, T):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    text = "\n\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")["text"]
                       if t.strip())
    ids = torch.tensor(tok(text)["input_ids"])
    assert ids.numel() >= seqs * T, f"{name}: {ids.numel()} tokens"
    return ids[: seqs * T].view(seqs, T)


def tensors(name, seqs=2, T=1024):
    from transformers import AutoModelForCausalLM
    import flash_attn_legacy
    ids = load_ids(name, seqs, T)
    rows = []

    def capture(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
        key_e, value_e = _expand_kv(module, key, value)
        scale = scaling if scaling is not None else query.shape[-1] ** -0.5
        q, k, v = query[0], key_e[0], value_e[0]
        o64, s64 = ref_attention(q, k, v, scale)
        STATS["s_lane"] = STATS["pv_lane"] = 0.0
        S_em = emul_scores(q, k, scale)
        L = q.shape[1]
        causal = torch.ones(L, L, dtype=torch.bool, device=q.device).tril()
        s_err = ((S_em.double() - s64).abs()[:, causal]).max().item()
        # near each row's max (within 16 log2 units: the keys exp2 does not flush to ~0), with K centering the
        # rows shift by q.c: compare after removing each row's max
        s64m = s64.masked_fill(~causal, float("-inf"))
        s64m = s64m - s64m.amax(-1, keepdim=True)
        sem = S_em.double().masked_fill(~causal, float("-inf"))
        sem = sem - sem.amax(-1, keepdim=True)
        near = s64m >= -16
        s_err_near = (sem - s64m).abs()[near].max().item()
        s_near_abs = s64.abs()[near].max().item()
        s32 = (q.float() @ k.float().transpose(1, 2)) * (scale * LOG2E)
        s32_err = ((s32.double() - s64).abs()[:, causal]).max().item()
        p64 = torch.softmax(s64.masked_fill(~causal, float("-inf")) / LOG2E, -1)
        p_em = torch.softmax(S_em.double().masked_fill(~causal, float("-inf")) / LOG2E, -1)
        heavy = p64 > 0.1
        p_rel = ((p_em - p64).abs() / p64)[heavy].max().item() if heavy.any() else 0.0
        o_em = emul_attention(q, k, v, scale, BC)
        o_sdpa = F.scaled_dot_product_attention(query, key_e, value_e, is_causal=True, scale=scale)[0]
        o_fa = flash_attn_legacy.flash_attention(query, key, value, softmax_scale=scale, causal=True)[0]
        rows.append((len(rows), relf(o_em.half(), o64), relf(o_sdpa, o64), relf(o_fa, o64), s_err, s32_err, p_rel,
                     s64[:, causal].abs().max().item(), STATS["s_lane"], STATS["pv_lane"], s_err_near, s_near_abs))
        return o_fa.unsqueeze(0).transpose(1, 2).contiguous(), None

    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
    AttentionInterface.register("capture", capture)
    AttentionMaskInterface.register("capture", sdpa_mask)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16, attn_implementation="capture").cuda().eval()
    with torch.inference_mode():
        for s in range(seqs):
            model(input_ids=ids[s:s + 1].cuda())
    print(f"### {name} [{'+'.join(PARTS)}{' k-center' if K_CENTER else ''}{' pow2' if POW2 else ''}]: per layer call (2 x {T} tokens), O relF vs fp64; S abs error max (log2 units); P rel error on "
          f"weights > 0.1; |S| max; largest fp16 lane (S, PV)")
    print("| call | O emul | O sdpa fp16 | O flash_attn_legacy | S err emul | S err fp32 | P rel err emul | |S| max "
          "| S lane max | PV lane max |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join([str(r[0])] + [f"{x:.2e}" for x in r[1:7]] + [f"{x:.1f}" for x in r[7:]]) + " |")
    import statistics
    for i, label in ((1, "O emul"), (2, "O sdpa"), (3, "O fa_legacy"), (6, "P rel"), (4, "S err"),
                     (10, "S err near max"), (11, "|S| near max")):
        vals = [r[i] for r in rows]
        print(f"{label}: median {statistics.median(vals):.2e}, max {max(vals):.2e}")
    print(f"S lane max {max(r[8] for r in rows):.1f}, PV lane max {max(r[9] for r in rows):.1f} (fp16 max 65504)")


def model_accuracy(name, seqs=16, T=1024):
    from transformers import AttentionInterface, AutoModelForCausalLM
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
    import flash_attn_legacy  # noqa: F401
    AttentionInterface.register("hfma2_emul", hfma2_emul_attention)
    AttentionMaskInterface.register("hfma2_emul", sdpa_mask)
    ids = load_ids(name, seqs, T)

    @torch.inference_mode()
    def run(dtype, impl):
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation=impl).cuda().eval()
        tops, nll = [], 0.0
        for s in range(seqs):
            x = ids[s:s + 1].cuda()
            logits = model(input_ids=x).logits[0, :-1].float()
            tops.append(logits.argmax(-1).cpu())
            nll += F.cross_entropy(logits, x[0, 1:], reduction="sum").item()
        del model
        torch.cuda.empty_cache()
        return torch.cat(tops), math.exp(nll / (seqs * (T - 1)))

    ref, ppl32 = run(torch.float32, "sdpa")
    cells = [f"{ppl32:.4f}"]
    tops = {}
    for impl in ("sdpa", "flash_attn_legacy", "hfma2_emul"):
        tops[impl], ppl = run(torch.float16, impl)
        cells += [f"{(tops[impl] == ref).float().mean().item():.3%}", f"{ppl:.4f}"]
    cells.append(f"{(tops['hfma2_emul'] == tops['flash_attn_legacy']).float().mean().item():.3%}")
    print("| model | fp32 ppl | top-1 vs fp32, fp16 sdpa | ppl | fp16 flash_attn_legacy | ppl | fp16 HFMA2 emulated | ppl "
          "| emulated vs flash_attn_legacy |")
    print(f"| {name.split('/')[-1]} ({'+'.join(PARTS)}{' k-center' if K_CENTER else ''}{' pow2' if POW2 else ''}"
          f", BC {BC}) | " + " | ".join(cells) + " |", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 3:
        BC = int(sys.argv[3])
    {"tensors": tensors, "model": model_accuracy}[sys.argv[1]](sys.argv[2])
    print("attn_emul done")
