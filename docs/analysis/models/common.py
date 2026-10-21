"""Shared helpers for the flash_attn_legacy model survey (spy, metrics, text)."""
import json
import math
import os

import torch

import flash_attn_legacy  # noqa: F401  registers the backend
import flash_attn_legacy.flash_attn as fa
import flash_attn_legacy.integrations as integ
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

OUT = "/workspace/analysis/models"
PROMPT = ("Flash attention computes exact attention in tiles, keeping the running "
          "softmax statistics in registers. Explain step by step why this saves memory:")


def long_text():
    import transformers
    base = os.path.dirname(transformers.__file__)
    parts = []
    for f in ("generation/utils.py", "modeling_utils.py", "trainer.py"):
        p = os.path.join(base, f)
        if os.path.exists(p):
            parts.append(open(p).read())
    return "\n".join(parts)


def _small(v):
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, torch.Tensor):
        if v.numel() == 1:
            return f"tensor({v.item():.4g})"
        return f"tensor{tuple(v.shape)}:{str(v.dtype).replace('torch.', '')}"
    if isinstance(v, (tuple, list)) and len(v) <= 4:
        return [_small(x) for x in v]
    return type(v).__name__


class Spy:
    """Wraps the registered flash_attn_legacy attention function and the kernel."""

    def __init__(self):
        self.records = []
        self.active = False
        self._hit = False
        orig_kernel = fa.flash_attention
        orig_fwd = integ.flash_attn_legacy_forward
        spy = self

        def kernel(q, k, v, *a, **kw):
            spy._hit = True
            spy._kernel_args = dict(causal=kw.get("causal"), scale=kw.get("softmax_scale"))
            return orig_kernel(q, k, v, *a, **kw)

        def fwd(module, query, key, value, attention_mask, *args, **kwargs):
            spy._hit = False
            spy._kernel_args = None
            out = orig_fwd(module, query, key, value, attention_mask, *args, **kwargs)
            if spy.active:
                m = attention_mask
                if m is None:
                    mdesc = None
                else:
                    mdesc = dict(shape=tuple(m.shape), dtype=str(m.dtype).replace("torch.", ""))
                kw = {k: _small(v) for k, v in kwargs.items()}
                if args:
                    kw["_positional"] = [_small(a) for a in args]
                spy.records.append(dict(
                    layer=getattr(module, "layer_idx", None),
                    module_is_causal=getattr(module, "is_causal", "n/a"),
                    module_sliding_window=getattr(module, "sliding_window", "n/a"),
                    q=tuple(query.shape), k=tuple(key.shape), dtype=str(query.dtype),
                    mask=mdesc, kwargs=kw, kernel=spy._hit, kernel_args=spy._kernel_args,
                ))
            return out

        fa.flash_attention = kernel
        ALL_ATTENTION_FUNCTIONS.register("flash_attn_legacy", fwd)

    def start(self):
        self.records = []
        self.active = True

    def stop(self):
        self.active = False
        return summarize(self.records)


def summarize(recs):
    """Compact per-scenario summary of which path ran."""
    out = {"calls": len(recs), "kernel_calls": sum(r["kernel"] for r in recs)}
    for phase, sel in (("prefill(q>1)", lambda r: r["q"][2] > 1), ("decode(q=1)", lambda r: r["q"][2] == 1)):
        rs = [r for r in recs if sel(r)]
        if not rs:
            continue
        kern_layers = sorted({r["layer"] for r in rs if r["kernel"]}, key=lambda x: (x is None, x))
        fb_layers = sorted({r["layer"] for r in rs if not r["kernel"]}, key=lambda x: (x is None, x))
        masks = sorted({str(r["mask"]) for r in rs})
        out[phase] = dict(calls=len(rs), kernel=sum(r["kernel"] for r in rs),
                          kernel_layers=kern_layers, fallback_layers=fb_layers,
                          masks=masks[:6],
                          qk_shapes=sorted({(r["q"][2], r["k"][2]) for r in rs})[:6],
                          kernel_causal=sorted({str(r["kernel_args"]["causal"]) for r in rs if r["kernel_args"]}))
    keys = {}
    for r in recs:
        for k, v in r["kwargs"].items():
            keys.setdefault(k, set()).add(json.dumps(v, default=str)[:80])
    out["kwargs_seen"] = {k: sorted(v)[:4] for k, v in keys.items()}
    out["module_is_causal"] = sorted({str(r["module_is_causal"]) for r in recs})
    out["module_sliding_window"] = sorted({str(r["module_sliding_window"]) for r in recs})
    return out


def first_record_dump(recs, n=3):
    return recs[:n]


def cmp(a, b):
    """a, b: [N, V] float tensors. Returns metrics dict."""
    a, b = a.float(), b.float()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    finite = torch.isfinite(a).all().item() and torch.isfinite(b).all().item()
    return dict(one_minus_cos=float(1 - cos.mean().item()),
                worst_pos_one_minus_cos=float(1 - cos.min().item()),
                top1=float((a.argmax(-1) == b.argmax(-1)).float().mean().item()),
                max_abs=float((a - b).abs().max().item()), finite=finite)


def set_impl(model, impl):
    model.set_attn_implementation(impl)
    assert model.config._attn_implementation == impl, (impl, model.config._attn_implementation)


def first_div(a, b):
    same = (a == b)
    return int(len(same)) if bool(same.all()) else int((~same).nonzero()[0])


def dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=str)
