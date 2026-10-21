"""Minimal repro: packed sequences (DataCollatorWithFlattening -> position_ids restart, no attention_mask)
in train mode with the default use_cache=True: flash_attn_legacy runs the kernel as ONE causal sequence
(cross-contamination), and ignores the cu_seq_lens_* kwargs when return_flash_attn_kwargs=True.
(sdpa/eager behave the same with use_cache=True: transformers skips packed-sequence detection when a cache
object exists. HF flash_attention_2 would use position_ids / cu_seq_lens -> varlen, i.e. no contamination.)
With use_cache=False the packed mask is built, flash_attn_legacy falls back to SDPA, results are correct.
"""
import torch

import flash_attn_legacy  # noqa: F401
import flash_attn_legacy.flash_attn as fa
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithFlattening

NAME = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float16, device_map="cuda",
                                             attn_implementation="flash_attn_legacy")
calls = []
orig = fa.flash_attention
fa.flash_attention = lambda q, k, v, *a, **kw: (calls.append(q.shape[2]), orig(q, k, v, *a, **kw))[1]

texts = ["The quick brown fox jumps over the lazy dog. " * 8, "In 1492 Columbus sailed the ocean blue. " * 6]
feats = [{"input_ids": tok(t).input_ids} for t in texts]
for fa_kwargs in (False, True):
    batch = DataCollatorWithFlattening(return_position_ids=True, return_flash_attn_kwargs=fa_kwargs)(feats)
    batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
    n0 = len(feats[0]["input_ids"])
    with torch.no_grad():
        alone = model(input_ids=batch["input_ids"][:, n0:], use_cache=False).logits[0].float()
    model.train()
    for use_cache in (None, False):
        calls.clear()
        kw = {} if use_cache is None else {"use_cache": use_cache}
        with torch.no_grad():
            out = model(**batch, **kw).logits[0, n0:].float()
        cos = torch.nn.functional.cosine_similarity(out, alone, dim=-1).mean().item()
        top1 = (out.argmax(-1) == alone.argmax(-1)).float().mean().item()
        print(f"fa_kwargs={fa_kwargs} use_cache={'default' if use_cache is None else use_cache}: "
              f"kernel calls={len(calls)}  2nd sequence vs alone: 1-cos={1 - cos:.3e} top1={top1:.3f}")
    model.eval()
