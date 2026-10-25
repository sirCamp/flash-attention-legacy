#!/usr/bin/env python3
"""End-to-end benchmark on HuggingFace models: flash_attn_legacy vs SDPA vs eager.

What a user sees, per model and attention implementation, on the same inputs:
  prefill   forward of B x N random tokens (inference, filling a DynamicCache as
            the prefill of a generation does): ms, tokens/s, peak memory
  generate  greedy generation of --new-tokens after a B x P prompt with a
            DynamicCache: decode tokens/s (prefill time subtracted), peak memory;
            --static adds a StaticCache reused across calls and, with --compile,
            HF's compiled decode steps (SM >= 7.0: inductor needs Triton)
  train     one optimizer step on B x N tokens with fp16 autocast + GradScaler,
            the attention projections trainable in fp32 and the rest frozen in
            fp16 (as examples/finetune.py: fits 16 GB without extra libraries):
            ms, tokens/s, peak memory, loss and loss scale (the step must have
            run: finite loss, AdamW state allocated); with --checkpointing
            (gradient checkpointing) the rows are train_ckpt; with --lora R
            (peft) only rank-R adapters of the projections are trainable, in
            fp32, and the rows are train_lora(_ckpt); --train-batch B > 1 is a
            padded batch (lengths from N down to N/2, right padding with an
            attention mask, as a fine-tuning batch: the varlen path), tokens/s
            counting the real tokens
An out-of-memory result is reported as OOM (the longest length that fits is a
result in itself: attention memory is O(N) on the kernel, O(N^2) in eager).
--check saves the top-1 agreement of the prefill logits with eager (512 tokens).

Every (model, implementation) runs in its own processes, one for prefill and generate
and one for train: compiled graphs (which keep their StaticCache alive), dynamo's
recompile budget and the CUDA caching allocator are not shared between the backends
being compared nor counted in the train memory; within a process every compiled case
is released (dynamo's caches and the CUDA graph pools) before the next one. Each row records the memory
allocated before the case (base_gib: the weights; the optimizer state of train is
created in the case, so it is in peak_gib) and, under torch.compile, dynamo's graph
breaks and unique graphs.

Usage:
    python benchmarks/models.py --models Qwen/Qwen2.5-3B --modes prefill train
    python benchmarks/models.py --models TinyLlama/TinyLlama-1.1B-Chat-v1.0 --check --save
Results print as JSON lines; --save appends them to docs/bench/models_<gpu>.jsonl;
--report FILE... turns saved files into markdown tables (one per GPU and mode; a case
saved more than once shows its last result).
--summary FILE... prints the README's overview: a row per model and GPU.
"""

import argparse
import contextlib
import gc
import json
import math
import pathlib
import statistics
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
IMPLS = ("flash_attn_legacy", "sdpa", "eager")
ATTN_PROJ = tuple(f".{proj}_proj." for proj in ("q", "k", "v", "o"))


def timed(fn, reps):
    """Median wall time (ms) of fn() over reps runs after one warmup, synchronized."""
    import torch
    fn()
    times = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1e3)
    return statistics.median(times)


@contextlib.contextmanager
def peak_memory(result):
    import torch
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    yield
    torch.cuda.synchronize()
    result["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)


def is_oom(err):
    """CUDA out of memory, also when wrapped (inductor, cuBLAS allocation)."""
    import torch
    text = str(err)
    return isinstance(err, torch.cuda.OutOfMemoryError) or "out of memory" in text or "ALLOC_FAILED" in text


def run_case(fn, result, on_oom=None):
    """fn(result) filling `result`, with the memory allocated before it, the
    implementation's fallbacks and dynamo's counters of this case; OOM recorded
    instead of raised (on_oom() then cleans the case's state)."""
    import torch
    import flash_attn_legacy
    from torch._dynamo.utils import counters
    flash_attn_legacy.reset_fallback_stats()
    counters.clear()
    result["base_gib"] = round(torch.cuda.memory_allocated() / 2**30, 2)
    try:
        fn(result)
    except Exception as err:  # noqa: BLE001 - only OOMs are recorded, the rest re-raised
        if not is_oom(err):
            raise
        for key in ("ms", "tok_s", "decode_tok_s", "peak_gib", "loss", "loss_scale"):
            result.pop(key, None)
        result["oom"] = True
        if on_oom:
            on_oom()
    result["fallbacks"] = flash_attn_legacy.fallback_stats()
    if counters["stats"]["unique_graphs"]:
        result["dynamo"] = {"unique_graphs": counters["stats"]["unique_graphs"],
                            "graph_breaks": sum(counters["graph_break"].values())}
        # the compiled graphs keep their inputs (a StaticCache) alive, and HF's reduce-overhead mode its CUDA graph pools
        torch.compiler.reset()
        torch._inductor.cudagraph_trees.reset_cudagraph_trees()
    gc.collect()
    torch.cuda.empty_cache()
    return result


def load(name, impl, dtype):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation=impl).cuda()


def tokens(model, B, N, seed=0):
    import torch
    g = torch.Generator().manual_seed(seed)
    vocab = model.config.get_text_config().vocab_size
    return torch.randint(100, min(vocab, 30000), (B, N), generator=g).cuda()


def bench_prefill(model, B, N, reps):
    """Forward of B x N tokens, last position's logits only."""
    import torch

    def fn(r):
        ids = tokens(model, B, N)
        with torch.inference_mode(), peak_memory(r):
            r["ms"] = round(timed(lambda: model(input_ids=ids, logits_to_keep=1), reps), 2)
        r["tok_s"] = round(B * N / r["ms"] * 1e3)
    return run_case(fn, {"mode": "prefill", "B": B, "N": N})


def bench_generate(model, B, P, new, reps, static, compile_):
    """Greedy generation of `new` tokens after a B x P prompt; the decode rate
    counts the new - 1 decode steps (the first token comes from the prefill,
    timed separately with the same cache type and subtracted)."""
    import torch
    from transformers import DynamicCache, StaticCache

    def fn(r):
        ids = tokens(model, B, P)
        kw = dict(max_new_tokens=new, min_new_tokens=new, do_sample=False, disable_compile=not compile_)
        static_cache = StaticCache(config=model.config, max_cache_len=P + new) if static else None

        def cache():  # a fresh DynamicCache (not the generation config's default, e.g. a hybrid one)
            if static:
                static_cache.reset()
                return static_cache
            return DynamicCache(config=model.config)

        def gen():
            return model.generate(ids, attention_mask=torch.ones_like(ids), past_key_values=cache(), **kw)

        def prefill():  # the same prefill as gen's: same cache type and mask
            return model(input_ids=ids, attention_mask=torch.ones_like(ids), past_key_values=cache(), logits_to_keep=1)

        with torch.inference_mode():
            if compile_:
                gen()  # compilation outside the timing and the peak memory
        with torch.inference_mode(), peak_memory(r):
            t_gen = timed(gen, reps)
            t_pre = timed(prefill, reps)
        r["ms"] = round(t_gen, 1)
        if t_gen <= t_pre:
            raise RuntimeError(f"generate {t_gen:.1f} ms not longer than its prefill {t_pre:.1f} ms: more --new-tokens")
        r["decode_tok_s"] = round(B * (new - 1) / (t_gen - t_pre) * 1e3, 1)
    mode = "generate" + ("_static" if static else "") + ("_compiled" if compile_ else "")
    return run_case(fn, {"mode": mode, "B": B, "N": P, "new": new})


def train_model(name, impl, lora):
    """Trainable in fp32 (AdamW states): the attention projections, or with lora > 0 rank-`lora`
    adapters on them; the rest frozen in fp16. Loaded in bf16 (checkpoints are bf16: the fp32
    parameters are exact) to keep host memory at 2 bytes per parameter."""
    import torch
    model = load(name, impl, torch.bfloat16)
    if lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=lora, lora_alpha=2 * lora,
                                                 target_modules=[p.strip(".") for p in ATTN_PROJ]))
    for n, p in model.named_parameters():
        trainable = "lora_" in n if lora else any(proj in n for proj in ATTN_PROJ)
        p.data = p.data.float() if trainable else p.data.half()
        p.requires_grad_(trainable)
    return model


def train_row(B, N, checkpointing, lora):
    """The keys of a train case (the LoRA rank too: two ranks are two cases)."""
    mode = "train" + ("_lora" if lora else "") + ("_ckpt" if checkpointing else "")
    return {"mode": mode, "B": B, "N": N, "dtype": "fp16", **({"lora": lora} if lora else {})}


def train_batch(model, B, N):
    """B x N tokens, labels, attention mask and the number of real tokens: one full row (no mask),
    or with B > 1 lengths from N down to N/2, right padded (pad positions masked and out of the loss)."""
    import torch
    ids = tokens(model, B, N)
    lengths = [N - (N // 2) * b // max(B - 1, 1) for b in range(B)]
    mask = torch.arange(N, device=ids.device) < torch.tensor(lengths, device=ids.device)[:, None]
    return ids.masked_fill(~mask, 0), ids.masked_fill(~mask, -100), mask.long() if B > 1 else None, sum(lengths)


def bench_train(model, B, N, reps, checkpointing, lora):
    """One AdamW step (fp16 autocast, GradScaler); the step must really run."""
    import torch
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-6)
    scaler = torch.amp.GradScaler("cuda", init_scale=2.0**10)  # no overflow-skipped steps in the timing
    if checkpointing:  # non-reentrant: the frozen embeddings give inputs without grad
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    last = {}

    def step(ids, labels, mask):
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(input_ids=ids, labels=labels, attention_mask=mask).loss
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        last["loss"] = loss.detach()

    def fn(r):
        *batch, real = train_batch(model, B, N)
        with peak_memory(r):
            r["ms"] = round(timed(lambda: step(*batch), reps), 1)
        r["tok_s"] = round(real / r["ms"] * 1e3)
        r["loss"], r["loss_scale"] = round(last["loss"].item(), 4), scaler.get_scale()
        if not math.isfinite(r["loss"]) or not opt.state:
            raise RuntimeError(f"train step did not run: loss {r['loss']}, optimizer state {len(opt.state)}")

    result = run_case(fn, train_row(B, N, checkpointing, lora), on_oom=lambda: opt.zero_grad(set_to_none=True))
    model.eval()
    return result


def check(name, impls, dtype, n=512):
    """Top-1 agreement of each implementation's prefill logits with eager."""
    import torch
    logits = {}
    for impl in ("eager", *[i for i in impls if i != "eager"]):
        model = load(name, impl, dtype)
        with torch.inference_mode():
            logits[impl] = model(input_ids=tokens(model, 1, n)).logits.float()
        del model
        torch.cuda.empty_cache()
    ref = logits["eager"].argmax(-1)
    return {impl: round((logits[impl].argmax(-1) == ref).float().mean().item(), 4) for impl in logits if impl != "eager"}


def report(paths):
    """Markdown tables from saved JSON lines: per GPU and mode, one row per
    (model, B, N) with each implementation's throughput and peak memory; against
    the faster of SDPA and eager, flash_attn_legacy's speedup, its time (ours /
    theirs: below 1 we are faster) and the peak memory it saves against the
    smaller of their peaks; then, per GPU, the longest N each implementation ran."""
    rows = latest_rows(paths)
    for gpu in dict.fromkeys(r["gpu"] for r in rows):
        for mode in dict.fromkeys(r["mode"] for r in rows if r["gpu"] == gpu):
            key = rate_key(mode)
            sel = [r for r in rows if r["gpu"] == gpu and r["mode"] == mode]
            impls = [i for i in IMPLS if any(r["impl"] == i for r in sel)]
            print(f"\n#### {gpu}: {mode} ({key.replace('_', ' ')}, peak GiB)\n")
            print("| model | dtype | B | N | " + " | ".join(impls) + " | speedup | time | memory saved |")
            print("|---" * (len(impls) + 7) + "|")
            for case in dict.fromkeys((*case_of(r), r["N"]) for r in sel):
                got = {r["impl"]: r for r in sel if (*case_of(r), r["N"]) == case}
                cells = []
                for i in impls:
                    r = got.get(i)
                    cells.append("-" if r is None else "OOM" if r.get("oom") else f"{r[key]:,} ({r['peak_gib']})")
                vs = versus(got, key)
                cmp = (["only ours" if got.get("flash_attn_legacy", {}).get(key) else "-", "-", "-"] if vs is None else
                       [f"{vs[0]:.2f}x", f"{1 / vs[0]:.2f}x", f"{vs[1]:.2f} GiB ({vs[2]:.0%})"])
                print(f"| {model_name(case)} | {case[1]} | {case[2]} | {case[-1]} | " + " | ".join(cells + cmp) + " |")
        longest(gpu, rows)


def latest_rows(paths):
    """The saved rows but the checks; a case measured again (e.g. after a kernel change) keeps its last result."""
    rows = [json.loads(line) for path in paths for line in pathlib.Path(path).read_text().splitlines() if line]
    latest = {}
    for r in rows:
        if r["mode"] != "check":
            latest[(r["gpu"], r["impl"], r["mode"], r["N"], *case_of(r))] = r
    return list(latest.values())


def rate_key(mode):
    return "decode_tok_s" if mode.startswith("generate") else "tok_s"


def versus(got, key):
    """flash_attn_legacy against the faster of SDPA and eager in one case (impl -> row): (speedup, GiB saved
    against the smaller of their peaks, its share), or None when ours or both of them did not run."""
    ours = got.get("flash_attn_legacy", {})
    ran = [got[i] for i in ("sdpa", "eager") if i in got and not got[i].get("oom")]
    if not ours.get(key) or not ran:
        return None
    low = min(r["peak_gib"] for r in ran)
    return ours[key] / max(r[key] for r in ran), low - ours["peak_gib"], (low - ours["peak_gib"]) / low


# The README overview: (column, modes tried in order, B, N)
SUMMARY = (("prefill, 4k", ("prefill",), 1, 4096),
           ("decode, 2k prompt", ("generate",), 1, 2048),
           ("compiled decode, 2k prompt", ("generate_static_compiled",), 1, 2048),
           ("train step, 2k", ("train_ckpt", "train_lora_ckpt", "train"), 1, 2048))
SUMMARY_TAG = {"train_lora_ckpt": " (LoRA)", "train": " (no ckpt)"}


def summary(paths):
    """One markdown table, a row per model (by weight size) and GPU: speedup against the faster of SDPA and
    eager and the peak memory saved in the SUMMARY cases (train: the first of its modes ours ran), and the
    longest prefill of ours / of the better of the others (>= N: no longer N measured; full tables: --report)."""
    rows = latest_rows(paths)
    gpus = sorted(dict.fromkeys(r["gpu"] for r in rows), key=lambda g: [k in g for k in GPU_ORDER].index(True))
    weights = lambda m: min((r["base_gib"] for r in rows if r["model"] == m and "base_gib" in r), default=0)  # noqa: E731
    print("| model | GPU | " + " | ".join(c[0] for c in SUMMARY) + " | longest prefill, ours / others |")
    print("|---" * (len(SUMMARY) + 3) + "|")
    for model in sorted(dict.fromkeys(r["model"] for r in rows), key=weights):
        for gpu in [g for g in gpus if any(r["model"] == model and r["gpu"] == g for r in rows)]:
            sel = [r for r in rows if r["model"] == model and r["gpu"] == gpu]
            cells = []
            for _, modes, B, N in SUMMARY:
                case = lambda m: {r["impl"]: r for r in sel if r["mode"] == m and r["B"] == B and r["N"] == N}  # noqa: E731
                tried = [m for m in modes if case(m)]
                mode = next((m for m in tried if not case(m).get("flash_attn_legacy", {"oom": True}).get("oom")),
                            tried[0] if tried else None)
                got, key = case(mode), rate_key(mode or "")
                vs = versus(got, key)
                if vs is not None:
                    cell = f"{vs[0]:.2f}x" + (f", -{vs[1]:.1f} GiB" if vs[1] >= 0.05 else "")
                elif got.get("flash_attn_legacy", {}).get(key):
                    cell = "only ours"
                else:
                    cell = "OOM" if got and all(r.get("oom") for r in got.values()) else "-"
                cells.append(cell + (SUMMARY_TAG.get(mode, "") if cell != "-" else ""))
            cells.append(" / ".join(longest_prefill(sel, impls) for impls in (("flash_attn_legacy",), ("sdpa", "eager"))))
            print(f"| {model.split('/')[-1]} | {short_gpu(gpu)} | " + " | ".join(cells) + " |")


GPU_ORDER = ("V100", "T4", "P100", "1080", "2080")


def longest_prefill(sel, impls):
    ran = [r["N"] for r in sel if r["mode"] == "prefill" and r["impl"] in impls and not r.get("oom")]
    oom = [r["N"] for r in sel if r["mode"] == "prefill" and r["impl"] in impls and r.get("oom")]
    if not ran:
        return "OOM" if oom else "-"
    return ("" if any(n > max(ran) for n in oom) else ">= ") + f"{max(ran) // 1024}k"


def short_gpu(gpu):
    for prefix in ("Tesla ", "NVIDIA GeForce GTX ", "NVIDIA GeForce RTX "):
        gpu = gpu.removeprefix(prefix)
    return gpu.split("-")[0] + (" 32 GB" if "32GB" in gpu else "")


def case_of(r):
    """What makes two rows of a mode the same case, but N."""
    return r["model"], r["dtype"], r["B"], r.get("new"), r.get("lora")


def model_name(case):
    return case[0].split("/")[-1] + (f" (LoRA r{case[4]})" if case[4] else "")


def longest(gpu, rows):
    """Per case, the longest N each implementation ran on `gpu`: a limit when it was out of memory at
    a longer N, else at least that (no longer N measured)."""
    sel = [r for r in rows if r["gpu"] == gpu]
    impls = [i for i in IMPLS if any(r["impl"] == i for r in sel)]
    print(f"\n#### {gpu}: longest N that fits (>= N: no longer N measured)\n")
    print("| model | mode | dtype | B | " + " | ".join(impls) + " |")
    print("|---" * (len(impls) + 4) + "|")
    for mode, case in dict.fromkeys((r["mode"], case_of(r)) for r in sel):
        same = [r for r in sel if (r["mode"], case_of(r)) == (mode, case)]
        cells = []
        for i in impls:
            ran = [r["N"] for r in same if r["impl"] == i and not r.get("oom")]
            oom = [r["N"] for r in same if r["impl"] == i and r.get("oom")]
            if not ran:
                cells.append("OOM" if oom else "-")
            else:
                cells.append(("" if any(n > max(ran) for n in oom) else ">= ") + f"{max(ran):,}")
        print(f"| {model_name(case)} | {mode} | {case[1]} | {case[2]} | " + " | ".join(cells) + " |")


def run_worker(args, name, impl):
    """The cases of one (model, implementation), in this process."""
    import torch
    import flash_attn_legacy
    gpu = torch.cuda.get_device_name()
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    out = ROOT / "docs" / "bench" / f"models_{gpu.replace(' ', '_')}.jsonl"

    def emit(r):  # printed and saved as soon as measured: a crash loses one case
        r.update(model=name, gpu=gpu)
        r.setdefault("dtype", args.dtype)  # train rows are fp16 autocast whatever the inference weights
        r.setdefault("impl", impl)
        print(json.dumps(r), flush=True)
        if args.save:
            with out.open("a") as f:
                f.write(json.dumps(r) + "\n")

    if args.check:
        print(f"{gpu}, torch {torch.__version__}, {flash_attn_legacy.get_device_info()}", flush=True)
        emit({"mode": "check", "impl": "-", "B": 1, "N": 512, "top1_vs_eager": check(name, args.impls, dtype)})
        return

    def loaded(fn, oom_rows):  # the model, or OOM rows for its cases when its weights alone do not fit
        try:
            return fn()
        except Exception as err:
            if not is_oom(err):
                raise
            for r in oom_rows:
                emit({**r, "oom": True})
            return None

    gen_modes = ["generate"] + (["generate_static" + ("_compiled" if args.compile else "")] if args.static else [])
    if "prefill" in args.modes or "generate" in args.modes:
        model = loaded(lambda: load(name, impl, dtype).eval(),
                       [{"mode": "prefill", "B": 1, "N": n} for n in args.prefill_lens if "prefill" in args.modes]
                       + [{"mode": m, "B": B, "N": P, "new": args.new_tokens} for B in args.gen_batch
                          for P in args.prompt_lens for m in gen_modes if "generate" in args.modes])
    if ("prefill" in args.modes or "generate" in args.modes) and model is not None:
        if "prefill" in args.modes:
            for n in args.prefill_lens:
                emit(bench_prefill(model, 1, n, args.reps))
        if "generate" in args.modes:
            for B in args.gen_batch:
                for P in args.prompt_lens:
                    emit(bench_generate(model, B, P, args.new_tokens, args.reps, False, False))
                    if args.static:
                        emit(bench_generate(model, B, P, args.new_tokens, args.reps, True, args.compile))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if "train" in args.modes:
        model = loaded(lambda: train_model(name, impl, args.lora),
                       [train_row(B, n, args.checkpointing, args.lora) for B in args.train_batch for n in args.train_lens])
        if model is not None:
            for B in args.train_batch:
                for n in args.train_lens:
                    emit(bench_train(model, B, n, args.reps, args.checkpointing, args.lora))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                allow_abbrev=False)  # the workers' command lines filter --check by name
    p.add_argument("--report", nargs="+", metavar="JSONL", help="print markdown tables from saved results and exit")
    p.add_argument("--summary", nargs="+", metavar="JSONL", help="print the README's per-model overview table and exit")
    p.add_argument("--models", nargs="+")
    p.add_argument("--impls", nargs="+", default=list(IMPLS), choices=IMPLS)
    p.add_argument("--modes", nargs="+", default=["prefill", "generate", "train"],
                   choices=["prefill", "generate", "train"])
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"], help="inference weights")
    p.add_argument("--prefill-lens", nargs="+", type=int, default=[1024, 4096, 8192, 16384])
    p.add_argument("--prompt-lens", nargs="+", type=int, default=[2048, 8192])
    p.add_argument("--gen-batch", nargs="+", type=int, default=[1, 8])
    p.add_argument("--new-tokens", type=int, default=128)
    p.add_argument("--train-lens", nargs="+", type=int, default=[1024, 2048, 4096, 8192])
    p.add_argument("--train-batch", nargs="+", type=int, default=[1], help="B > 1: a padded batch")
    p.add_argument("--checkpointing", action="store_true", help="gradient checkpointing in train")
    p.add_argument("--lora", type=int, default=0, metavar="R", help="train rank-R LoRA adapters (needs peft)")
    p.add_argument("--static", action="store_true", help="generate also with a reused StaticCache")
    p.add_argument("--compile", action="store_true", help="with --static: HF's compiled decode steps (SM >= 7.0)")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--check", action="store_true", help="save the top-1 agreement with eager on 512 tokens")
    p.add_argument("--save", action="store_true")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)  # one (model, impl) in this process
    args = p.parse_args()
    if args.report:
        return report(args.report)
    if args.summary:
        return summary(args.summary)
    if not args.models:
        p.error("--models is required")
    if args.compile and not args.static:
        p.error("--compile needs --static (HF compiles the decode step of a StaticCache)")
    if args.reps < 1 or args.lora < 0 or min(args.train_batch) < 1:
        p.error("--reps and --train-batch must be >= 1, --lora >= 0")
    if args.new_tokens < 2:
        p.error("--new-tokens must be >= 2 (the decode rate counts new - 1 steps)")
    if args.worker:
        return run_worker(args, args.models[0], args.impls[0])
    # One process per (model, implementation, inference or train), and one for the agreement check:
    # nothing the inference cases leave behind (dynamo's compiled graphs hold their StaticCache)
    # is counted in the train memory
    base = [a for a in sys.argv[1:] if a != "--check"]
    inference = [m for m in args.modes if m != "train"]
    phases = ([inference] if inference else []) + ([["train"]] if "train" in args.modes else [])
    failed = False
    for name in args.models:
        jobs = ([["--check", "--impls", *args.impls]] if args.check else []) + [
            ["--impls", impl, "--modes", *modes] for impl in args.impls for modes in phases]
        for job in jobs:
            cmd = [sys.executable, __file__, *base, "--worker", "--models", name, *job]
            if subprocess.run(cmd).returncode:  # reported, the other implementations still run
                print(f"FAILED: {name} {' '.join(job)}", file=sys.stderr, flush=True)
                failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
