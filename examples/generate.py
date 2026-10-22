"""Generation with HuggingFace transformers on flash_attn_legacy (or SDPA / eager,
to compare): prefill on the flash kernel, every decode step on the decode kernel.

    python examples/generate.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
    python examples/generate.py --model Qwen/Qwen2.5-3B --impl sdpa
    python examples/generate.py --model Qwen/Qwen2.5-3B --compare      # all impls, same prompts

Reports tokens per second (prefill and decode), peak GPU memory and which
attention calls fell back (flash_attn_legacy.fallback_stats()).
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import flash_attn_legacy

PROMPTS = [
    "Explain in a few sentences why attention needs O(N^2) memory when computed naively.",
    "Write a short poem about an old GPU that still trains models.",
    "List three differences between the Pascal and the Volta GPU architectures.",
    "What is the capital of Italy? Answer in one word.",
]


def run(model_name, impl, args, tokenizer):
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, attn_implementation=impl).cuda().eval()
    prompts = (PROMPTS * ((args.batch + len(PROMPTS) - 1) // len(PROMPTS)))[:args.batch]
    if tokenizer.chat_template:
        prompts = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                                 add_generation_prompt=True) for p in prompts]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")  # left padding
    kw = dict(max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id,
              cache_implementation=args.cache, disable_compile=True)

    with torch.inference_mode():
        model.generate(**inputs, max_new_tokens=4, do_sample=False, pad_token_id=tokenizer.pad_token_id)  # warmup
        flash_attn_legacy.reset_fallback_stats()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        model(**inputs)  # prefill alone, for its timing
        torch.cuda.synchronize()
        t_prefill = time.perf_counter() - t0
        t0 = time.perf_counter()
        out = model.generate(**inputs, **kw)
        torch.cuda.synchronize()
        t_total = time.perf_counter() - t0
    new = out[:, inputs["input_ids"].shape[1]:]
    n_new = int((new != tokenizer.pad_token_id).sum())
    texts = tokenizer.batch_decode(new, skip_special_tokens=True)
    print(f"\n== {impl}: prefill {inputs['input_ids'].numel() / t_prefill:,.0f} tok/s, generate {n_new / t_total:,.1f} "
          f"new tok/s ({t_total:.2f} s), peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, "
          f"fallbacks {flash_attn_legacy.fallback_stats() if impl == 'flash_attn_legacy' else '-'}")
    for t in texts[:2]:
        print("  >", t.strip().replace("\n", " ")[:200])
    del model
    torch.cuda.empty_cache()
    return texts


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--impl", default="flash_attn_legacy", choices=["flash_attn_legacy", "sdpa", "eager"])
    p.add_argument("--compare", action="store_true", help="run flash_attn_legacy, sdpa and eager")
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--cache", default="dynamic", choices=["dynamic", "static"])
    args = p.parse_args()

    print(flash_attn_legacy.get_device_info())
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    impls = ["flash_attn_legacy", "sdpa", "eager"] if args.compare else [args.impl]
    outputs = {impl: run(args.model, impl, args, tokenizer) for impl in impls}
    if args.compare:
        ref = outputs["eager"]
        for impl in impls[:-1]:
            same = sum(a == b for a, b in zip(outputs[impl], ref))
            print(f"{impl}: {same}/{len(ref)} generations identical to eager "
                  "(fp16 near-ties may legitimately fork a long greedy generation)")


if __name__ == "__main__":
    main()
