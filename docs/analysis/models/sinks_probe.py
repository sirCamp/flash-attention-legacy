"""Minimal repro: attention sinks (gpt-oss s_aux) are silently ignored by the flash kernel path.

Builds a tiny random-init GptOss model (2 layers: sliding + full, head_dim 64), gives the sink
logits non-trivial values, and compares flash_attn_legacy vs eager (fp16) and fp32 eager gold.
"""
import logging
import sys

import torch

sys.path.insert(0, "/workspace/analysis/models")
from common import OUT, Spy, cmp, dump  # noqa: E402
from transformers import GptOssConfig, GptOssForCausalLM  # noqa: E402

logging.getLogger("transformers").setLevel(logging.ERROR)
torch.manual_seed(0)
cfg = GptOssConfig(vocab_size=1000, hidden_size=256, intermediate_size=256, num_hidden_layers=2,
                   num_attention_heads=4, num_key_value_heads=2, head_dim=64, sliding_window=128,
                   num_local_experts=4, num_experts_per_tok=2, max_position_embeddings=4096,
                   layer_types=["sliding_attention", "full_attention"])
spy = Spy()
model = GptOssForCausalLM(cfg).to("cuda", torch.float32).eval()
with torch.no_grad():
    for layer in model.model.layers:
        layer.self_attn.sinks.normal_(0.0, 2.0)  # trained gpt-oss sinks are O(1) logits
R = {"config": dict(head_dim=64, layer_types=cfg.layer_types, sliding_window=128,
                    sinks=[l.self_attn.sinks[:4].tolist() for l in model.model.layers]), "cases": {}}
for N in (64, 100, 512):
    ids = torch.randint(0, 1000, (1, N), device="cuda")
    with torch.no_grad():
        model.float()
        model.set_attn_implementation("eager")
        gold = model(input_ids=ids).logits[0].float()
        model.half()
        out = {}
        for impl in ("eager", "flash_attn_legacy"):
            model.set_attn_implementation(impl)
            if impl == "flash_attn_legacy":
                spy.start()
            out[impl] = model(input_ids=ids).logits[0].float()
            if impl == "flash_attn_legacy":
                path = spy.stop()
                rec = spy.records[:2]
        # also: what if sinks are zeroed? (then the sink still adds exp(0) to the denominator)
        case = dict(path={k: path.get(k) for k in ("calls", "kernel_calls", "prefill(q>1)", "kwargs_seen")},
                    first_records=rec,
                    flash_vs_fp32gold=cmp(out["flash_attn_legacy"], gold),
                    eager16_vs_fp32gold=cmp(out["eager"], gold),
                    flash_vs_eager16=cmp(out["flash_attn_legacy"], out["eager"]))
    R["cases"][f"N={N}"] = case
    print(N, {k: v for k, v in case.items() if k != "first_records"}, flush=True)
dump(R, f"{OUT}/results/sinks_probe.json")
print("DONE")
