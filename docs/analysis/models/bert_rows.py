"""BERT N=512: per-query-row attention error of the kernel vs fp16 SDPA (both vs fp32), on captured real q/k/v,
plus which token position carries the end-to-end outlier."""
import sys

import torch

sys.path.insert(0, "/workspace/analysis/models")
from common import OUT, dump, long_text  # noqa: E402
import flash_attn_legacy.integrations as integ  # noqa: E402
from flash_attn_legacy.flash_attn import flash_attention  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402

name = "google-bert/bert-base-uncased"
tok = AutoTokenizer.from_pretrained(name)
CAP = []


def cap(module, q, k, v, mask, *a, **kw):
    CAP.append((q.detach().clone(), k.detach().clone(), v.detach().clone(), kw.get("scaling")))
    return integ.flash_attn_legacy_forward(module, q, k, v, mask, *a, **kw)


ALL_ATTENTION_FUNCTIONS.register("flash_attn_legacy", cap)
m = AutoModel.from_pretrained(name, dtype=torch.float16, device_map="cuda", attn_implementation="flash_attn_legacy").eval()
ids = tok(long_text(), return_tensors="pt", add_special_tokens=False).input_ids[:, :512].cuda()
with torch.no_grad():
    hf = m(input_ids=ids).last_hidden_state[0].float()
    m.set_attn_implementation("sdpa")
    hs = m(input_ids=ids).last_hidden_state[0].float()
cos = torch.nn.functional.cosine_similarity(hf, hs, dim=-1)
worst = int(cos.argmin())
R = {"worst_pos": worst, "worst_1mcos": float(1 - cos[worst]), "token": tok.decode(ids[0, worst]),
     "n_pos_1mcos_gt_1e-5": int((1 - cos > 1e-5).sum()), "layers": []}
print(R, flush=True)
for li, (q, k, v, sc) in enumerate(CAP):
    with torch.no_grad():
        ref = torch.softmax((q.float() @ k.float().transpose(-1, -2)) * sc, -1) @ v.float()
        fl = flash_attention(q, k, v, softmax_scale=sc, causal=False).float()
        sd = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=sc).float()
        ef = (fl - ref).abs().amax(dim=(0, 1, 3))  # per query row
        es = (sd - ref).abs().amax(dim=(0, 1, 3))
        logits_max = ((q.float() @ k.float().transpose(-1, -2)) * sc).amax().item()
    row = dict(layer=li, kernel_max=float(ef.max()), sdpa_max=float(es.max()), kernel_row_worst=int(ef.argmax()),
               kernel_err_at_worst_pos=float(ef[worst]), sdpa_err_at_worst_pos=float(es[worst]),
               max_row_ratio=float((ef / es.clamp_min(1e-6)).max()), max_scaled_logit=logits_max)
    R["layers"].append(row)
    print(row, flush=True)
dump(R, f"{OUT}/results/bert_rows.json")
