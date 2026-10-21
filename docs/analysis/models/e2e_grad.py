"""Isolate the attention kernel's contribution to end-to-end gradient error.

fp32 Qwen2.5-0.5B everywhere; only attention differs:
  gold      : eager fp32
  flash     : flash_attn_legacy (q/k/v cast to fp16 -> kernel -> cast back)
  sdpa16cast: q/k/v cast to fp16 -> torch SDPA(is_causal) -> cast back
  math16cast: q/k/v cast to fp16 -> fp16 matmul/softmax(fp32) -> cast back
Grad cosine vs gold for several params, B=2 x 512, no padding.
"""
import sys

import torch

sys.path.insert(0, "/workspace/analysis/models")
from common import OUT, dump, long_text  # noqa: E402
import flash_attn_legacy  # noqa: E402,F401
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402


def sdpa16cast(module, q, k, v, mask, dropout=0.0, scaling=None, **kw):
    rep = q.shape[1] // k.shape[1]
    o = torch.nn.functional.scaled_dot_product_attention(q.half(), k.half().repeat_interleave(rep, 1),
                                                         v.half().repeat_interleave(rep, 1), is_causal=True, scale=scaling)
    return o.to(q.dtype).transpose(1, 2).contiguous(), None


def math16cast(module, q, k, v, mask, dropout=0.0, scaling=None, **kw):
    rep = q.shape[1] // k.shape[1]
    q, k2, v2 = q.half(), k.half().repeat_interleave(rep, 1), v.half().repeat_interleave(rep, 1)
    s = (q @ k2.transpose(-1, -2)) * scaling
    n = q.shape[2]
    s = s.masked_fill(~torch.ones(n, n, dtype=torch.bool, device=q.device).tril(), float("-inf"))
    o = torch.softmax(s.float(), -1).half() @ v2
    return o.float().transpose(1, 2).contiguous(), None


ALL_ATTENTION_FUNCTIONS.register("sdpa16cast", sdpa16cast)
ALL_ATTENTION_FUNCTIONS.register("math16cast", math16cast)
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS  # noqa: E402
ALL_MASK_ATTENTION_FUNCTIONS.register("sdpa16cast", ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])
ALL_MASK_ATTENTION_FUNCTIONS.register("math16cast", ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])

NAME = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(NAME)
ids = tok(long_text(), return_tensors="pt", add_special_tokens=False).input_ids[0]
x = torch.stack([ids[:512], ids[512:1024]]).cuda()
model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32, device_map="cuda", attn_implementation="eager")
model.train()
P = [n for n, _ in model.named_parameters() if any(s in n for s in ("layers.0.self_attn.q_proj.weight", "layers.0.self_attn.k_proj.weight",
                                                                      "layers.0.self_attn.v_proj.weight", "layers.12.self_attn.q_proj.weight",
                                                                      "layers.23.self_attn.k_proj.weight", "embed_tokens"))]
G, R = {}, {}
for impl in ("eager", "flash_attn_legacy", "sdpa16cast", "math16cast", "sdpa"):
    model.config._attn_implementation = impl
    assert model.model.layers[0].self_attn.config._attn_implementation == impl
    model.zero_grad(set_to_none=True)
    out = model(input_ids=x, labels=x, use_cache=False)
    out.loss.backward()
    G[impl] = {p: model.get_parameter(p).grad.detach().clone().flatten() for p in P}
    R[impl] = dict(loss=out.loss.item())
for impl in G:
    R[impl]["grad_cos_vs_fp32_eager"] = {p.replace("model.", ""): round(float(torch.nn.functional.cosine_similarity(G[impl][p], G["eager"][p], dim=0)), 6) for p in P}
    R[impl]["grad_norm"] = {p.replace("model.", ""): round(float(G[impl][p].norm()), 5) for p in P}
    print(impl, R[impl], flush=True)
dump(R, f"{OUT}/results/e2e_grad.json")
