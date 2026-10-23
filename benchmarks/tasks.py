#!/usr/bin/env python3
"""Task accuracy on HuggingFace models: flash_attn_legacy vs SDPA vs eager.

What the attention implementation changes in a model's answers, on standard benchmarks run by
lm-evaluation-harness (the model is loaded here with each implementation and handed to it):
  arc_easy, arc_challenge, hellaswag  multiple choice by log-likelihood (the prefill path): acc, acc_norm
  gsm8k                               5-shot greedy generation, batched with left padding (prefill and
                                      decode paths): exact match (strict and flexible answer extraction)
  passkey_<N>                         a 5-digit key hidden at a spread of depths in N tokens of filler,
                                      greedy answer: retrieval accuracy, also where only some
                                      implementations fit (OOM is a result)
Every document's prediction is saved, so the report gives each implementation's score and how many
answers differ from eager's. Each implementation runs in its own process.

Usage:
    python benchmarks/tasks.py --model meta-llama/Llama-3.1-8B-Instruct --save
    python benchmarks/tasks.py --model Qwen/Qwen2.5-3B --tasks arc_easy --limit 50
Results print as JSON lines; --save appends them to docs/bench/tasks_<gpu>.jsonl; --report FILE...
turns saved files into markdown tables.
"""

import argparse
import json
import pathlib
import random
import subprocess
import sys

from models import IMPLS, ROOT, is_oom, load  # one way to load a model and to recognise an OOM

LM_EVAL_TASKS = ("arc_easy", "arc_challenge", "hellaswag", "gsm8k")
FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "


def lm_eval_rows(model, tok, tasks, batch_size, limit):
    """One row per (task, metric[, filter]) with the per-document values and predictions."""
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    out = simple_evaluate(model=HFLM(pretrained=model, tokenizer=tok, batch_size=batch_size), tasks=list(tasks),
                          limit=limit, log_samples=True)
    rows = []
    for task, res in out["results"].items():
        samples = sorted(out["samples"][task], key=lambda s: s["doc_id"])
        for key, value in res.items():
            if not isinstance(value, float) or "_stderr" in key:
                continue
            metric, _, filt = key.partition(",")
            sel = [s for s in samples if s.get("filter", "none") == filt]
            preds = [pred_of(s) for s in sel]
            rows.append({"task": task, "metric": metric, "filter": filt, "value": round(value, 4),
                         "stderr": round(res.get(f"{metric}_stderr,{filt}", float("nan")), 4), "n": len(sel),
                         "preds": preds, "correct": [s.get(metric) for s in sel]})
    return rows


def pred_of(s):
    """The answer a sample's metric judged: the chosen option (log-likelihood argmax) or the extracted text."""
    resps = s["filtered_resps"]
    if resps and isinstance(resps[0], (list, tuple)):  # [(loglik, is_greedy)] per option
        return max(range(len(resps)), key=lambda i: resps[i][0])
    return resps[0] if resps else None


def passkey_rows(model, tok, lengths, n):
    """Greedy answer to "what is the pass key" after N tokens of filler with the key at depth (i + 0.5) / n."""
    import torch
    rng = random.Random(0)
    filler_ids = tok(FILLER, add_special_tokens=False).input_ids
    rows = []
    for N in lengths:
        preds, correct = [], []
        try:
            for i in range(n):
                key = str(rng.randint(10000, 99999))
                needle = tok(f" The pass key is {key}. Remember it. {key} is the pass key. ", add_special_tokens=False).input_ids
                body = (filler_ids * (N // len(filler_ids) + 1))[:N - len(needle) - 64]
                at = int(len(body) * (i + 0.5) / n)
                text = tok.decode(body[:at] + needle + body[at:])
                msgs = [{"role": "user", "content": text + "\nWhat is the pass key? Answer with the number only."}]
                ids = (tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"]
                       if tok.chat_template else tok(text + "\nThe pass key is", return_tensors="pt").input_ids).cuda()
                with torch.inference_mode():
                    out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=8, do_sample=False)
                answer = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
                preds.append(answer.strip())
                correct.append(float(key in answer))
            rows.append({"task": f"passkey_{N}", "metric": "acc", "filter": "none", "value": round(sum(correct) / n, 4),
                         "n": n, "tokens": int(ids.shape[1]), "preds": preds, "correct": correct})
        except Exception as err:  # noqa: BLE001 - only OOMs are recorded, the rest re-raised
            if not is_oom(err):
                raise
            rows.append({"task": f"passkey_{N}", "metric": "acc", "filter": "none", "oom": True, "n": n})
            torch.cuda.empty_cache()
    return rows


def run_worker(args):
    import torch
    import flash_attn_legacy  # registers the attention implementation
    from transformers import AutoTokenizer
    gpu = torch.cuda.get_device_name()
    out = ROOT / "docs" / "bench" / f"tasks_{gpu.replace(' ', '_')}.jsonl"
    model = load(args.model, args.impl, torch.float16).eval()
    tok = AutoTokenizer.from_pretrained(args.model)
    tasks = [t for t in args.tasks if t in LM_EVAL_TASKS]
    rows = lm_eval_rows(model, tok, tasks, args.batch_size, args.limit) if tasks else []
    if "passkey" in args.tasks:
        rows += passkey_rows(model, tok, args.passkey_lens, args.passkey_n)
    for r in rows:
        r.update(model=args.model, impl=args.impl, gpu=gpu, dtype="fp16", limit=args.limit,
                 fallbacks=flash_attn_legacy.fallback_stats())
        print(json.dumps({k: v for k, v in r.items() if k not in ("preds", "correct")}), flush=True)
        if args.save:
            with out.open("a") as f:
                f.write(json.dumps(r) + "\n")


def report(paths):
    """Per GPU and model: each implementation's score per (task, metric, filter), and how many
    predictions differ from eager's (same documents)."""
    rows = [json.loads(line) for path in paths for line in pathlib.Path(path).read_text().splitlines() if line]
    latest = {(r["gpu"], r["model"], r["impl"], r["task"], r["metric"], r["filter"]): r for r in rows}
    rows = list(latest.values())
    for gpu, model in dict.fromkeys((r["gpu"], r["model"]) for r in rows):
        sel = [r for r in rows if (r["gpu"], r["model"]) == (gpu, model)]
        impls = [i for i in IMPLS if any(r["impl"] == i for r in sel)]
        print(f"\n#### {gpu}: {model} (fp16; score, answers different from eager)\n")
        print("| task | metric | n | " + " | ".join(impls) + " |")
        print("|---" * (len(impls) + 3) + "|")
        for task, metric, filt in dict.fromkeys((r["task"], r["metric"], r["filter"]) for r in sel):
            got = {r["impl"]: r for r in sel if (r["task"], r["metric"], r["filter"]) == (task, metric, filt)}
            ref = got.get("eager")
            cells = []
            for i in impls:
                r = got.get(i)
                if r is None or r.get("oom"):
                    cells.append("-" if r is None else "OOM")
                    continue
                cell = f"{r['value']:.4f}"
                if ref is not None and not ref.get("oom") and i != "eager":
                    cell += f" ({sum(a != b for a, b in zip(r['preds'], ref['preds']))})"
                cells.append(cell)
            n = next(r["n"] for r in got.values())
            name = metric + ("" if filt == "none" else f" ({filt})")
            print(f"| {task} | {name} | {n} | " + " | ".join(cells) + " |")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                allow_abbrev=False)
    p.add_argument("--report", nargs="+", metavar="JSONL", help="print markdown tables from saved results and exit")
    p.add_argument("--model")
    p.add_argument("--impls", nargs="+", default=list(IMPLS), choices=IMPLS)
    p.add_argument("--tasks", nargs="+", default=[*LM_EVAL_TASKS, "passkey"], choices=[*LM_EVAL_TASKS, "passkey"])
    p.add_argument("--limit", type=int, default=None, help="first N documents of each lm-eval task")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--passkey-lens", nargs="+", type=int, default=[4096, 8192, 16384, 32768])
    p.add_argument("--passkey-n", type=int, default=20)
    p.add_argument("--save", action="store_true")
    p.add_argument("--impl", help=argparse.SUPPRESS)  # worker: one implementation in this process
    args = p.parse_args()
    if args.report:
        return report(args.report)
    if not args.model:
        p.error("--model is required")
    if args.impl:
        return run_worker(args)
    failed = False
    for impl in args.impls:  # own process each: no compiled or cached state shared between the backends
        if subprocess.run([sys.executable, __file__, *sys.argv[1:], "--impl", impl]).returncode:
            print(f"FAILED: {args.model} {impl}", file=sys.stderr, flush=True)
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
