"""Rough timing: prefill ms (N in list) and greedy decode tok/s, flash_attn_legacy vs sdpa (fp16).

usage: python timing.py MODEL TAG [N1,N2,...]
"""
import logging
import sys
import time

import torch

sys.path.insert(0, "/workspace/analysis/models")
from common import OUT, Spy, dump, long_text, set_impl  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

logging.getLogger("transformers").setLevel(logging.ERROR)
name, tag = sys.argv[1], sys.argv[2]
Ns = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "512,1024,2048,4096").split(",")]
spy = Spy()
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16, device_map="cuda",
                                             attn_implementation="flash_attn_legacy").eval()
ids_all = tok(long_text(), return_tensors="pt").input_ids.cuda()
R = {"model": name, "prefill_ms": {}, "decode": {}, "kernel_calls": {}}


@torch.no_grad()
def timeit(fn, it=5):
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(it):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


for N in Ns:
    ids = ids_all[:, :N]
    row = {}
    for impl in ("flash_attn_legacy", "sdpa"):
        set_impl(model, impl)
        try:
            torch.cuda.reset_peak_memory_stats()
            if impl == "flash_attn_legacy":
                spy.start()
            row[impl] = round(timeit(lambda: model(input_ids=ids, logits_to_keep=1)), 2)
            if impl == "flash_attn_legacy":
                R["kernel_calls"][N] = spy.stop()["kernel_calls"]
            row[impl + "_peak_GB"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        except torch.cuda.OutOfMemoryError:
            row[impl] = "OOM"
            spy.active = False
            torch.cuda.empty_cache()
    if isinstance(row.get("sdpa"), float) and isinstance(row.get("flash_attn_legacy"), float):
        row["speedup"] = round(row["sdpa"] / row["flash_attn_legacy"], 3)
    R["prefill_ms"][N] = row
    print("prefill", N, row, flush=True)

for P in (32, 1000):
    ids = ids_all[:, :P]
    for impl in ("flash_attn_legacy", "sdpa"):
        set_impl(model, impl)
        g = dict(max_new_tokens=128, min_new_tokens=128, do_sample=False)
        model.generate(ids, max_new_tokens=8, do_sample=False)
        torch.cuda.synchronize()
        t = time.time()
        model.generate(ids, **g)
        torch.cuda.synchronize()
        dt = time.time() - t
        R["decode"][f"prompt{P}_{impl}"] = dict(s=round(dt, 3), tok_per_s=round(128 / dt, 1))
        print("generate", P, impl, R["decode"][f"prompt{P}_{impl}"], flush=True)
dump(R, f"{OUT}/results/timing_{tag}.json")
print("DONE")
