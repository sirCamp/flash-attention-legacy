"""
Quick training test: eager vs flash_attn_legacy on TinyLlama 1.1B.

Runs a few gradient steps on random token sequences and compares:
  - Loss curve (both should decrease similarly)
  - Step time
  - Peak GPU memory

Usage:
    python scripts/test_training.py
    python scripts/test_training.py --steps 20 --seq_len 512
"""

import argparse
import time
import logging
import torch
import torch.nn.functional as F

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--mixed_lengths", action="store_true",
                   help="Use variable-length sequences (pad to max in batch)")
    p.add_argument("--no_grad_ckpt", action="store_true",
                   help="Disable gradient checkpointing (uses more memory, shows attention diff)")
    return p.parse_args()


def load_model(model_name, attn_impl):
    from transformers import AutoModelForCausalLM
    import flash_attn_legacy  # ensure register()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        attn_implementation=attn_impl,
    ).cuda()
    model.train()
    return model


def run_training(model, args, label):
    vocab_size = model.config.vocab_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")

    # Warmup
    dummy = torch.randint(0, vocab_size, (1, 32), device="cuda")
    with torch.amp.autocast("cuda", dtype=torch.float16):
        out = model(dummy, labels=dummy)
    scaler.scale(out.loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated()

    losses = []
    times = []

    for step in range(args.steps):
        if args.mixed_lengths:
            # Variable lengths: each sequence is 25%-100% of seq_len
            min_len = max(16, args.seq_len // 4)
            lengths = [torch.randint(min_len, args.seq_len + 1, (1,)).item()
                       for _ in range(args.batch_size)]
            max_len = max(lengths)
            input_ids = torch.full((args.batch_size, max_len), 0, dtype=torch.long, device="cuda")
            for b, l in enumerate(lengths):
                input_ids[b, :l] = torch.randint(0, vocab_size, (l,), device="cuda")
        else:
            input_ids = torch.randint(0, vocab_size, (args.batch_size, args.seq_len), device="cuda")
        labels = input_ids.clone()

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(input_ids, labels=labels)
            loss = out.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        losses.append(loss.item())
        times.append(t1 - t0)

    peak_total_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    baseline_mb = mem_before / (1024 ** 2)
    peak_extra_mb = peak_total_mb - baseline_mb
    avg_time = sum(times) / len(times)
    avg_time_no_first = sum(times[1:]) / max(len(times) - 1, 1)

    return {
        "label": label,
        "losses": losses,
        "avg_step_ms": avg_time * 1000,
        "avg_step_ms_no_first": avg_time_no_first * 1000,
        "baseline_mb": baseline_mb,
        "peak_total_mb": peak_total_mb,
        "peak_extra_mb": peak_extra_mb,
    }


def main():
    args = parse_args()
    props = torch.cuda.get_device_properties(0)

    print(f"\n{'=' * 70}")
    print(f"  TRAINING TEST")
    print(f"  Model: {args.model}")
    print(f"  GPU:   {props.name} ({props.total_memory / (1024**3):.1f} GB)")
    mode = "mixed_lengths" if args.mixed_lengths else "fixed"
    print(f"  Config: B={args.batch_size}, seq_len={args.seq_len}, "
          f"steps={args.steps}, lr={args.lr}, mode={mode}")
    print(f"{'=' * 70}")

    backends = ["eager", "flash_attn_legacy"]
    results = []

    for backend in backends:
        print(f"\n  [{backend}] Loading model...")
        model = load_model(args.model, backend)
        if not args.no_grad_ckpt:
            model.gradient_checkpointing_enable()

        print(f"  [{backend}] Training {args.steps} steps...")
        try:
            res = run_training(model, args, backend)
            results.append(res)
        except torch.cuda.OutOfMemoryError:
            print(f"  [{backend}] OOM!")
            results.append({"label": backend, "losses": [], "avg_step_ms": 0,
                            "avg_step_ms_no_first": 0, "peak_extra_mb": 0, "oom": True})
        finally:
            del model
            torch.cuda.empty_cache()

    # Print results
    print(f"\n{'=' * 70}")
    print(f"  RESULTS")
    print(f"{'=' * 70}")

    # Loss curves side by side
    print(f"\n  {'step':>4} |", end="")
    for r in results:
        print(f" {r['label']:>20} |", end="")
    print()
    print(f"  {'-' * (6 + 23 * len(results))}")

    max_steps = max(len(r["losses"]) for r in results) if results else 0
    for s in range(max_steps):
        print(f"  {s:>4} |", end="")
        for r in results:
            if s < len(r["losses"]):
                print(f" {r['losses'][s]:>20.4f} |", end="")
            else:
                print(f" {'OOM':>20} |", end="")
        print()

    # Summary
    print(f"\n  {'metric':>25} |", end="")
    for r in results:
        print(f" {r['label']:>20} |", end="")
    print()
    print(f"  {'-' * (27 + 23 * len(results))}")

    for metric, key, fmt in [
        ("avg step (ms)", "avg_step_ms", ".1f"),
        ("avg step excl 1st (ms)", "avg_step_ms_no_first", ".1f"),
        ("baseline mem (MB)", "baseline_mb", ".0f"),
        ("peak total mem (MB)", "peak_total_mb", ".0f"),
        ("peak extra mem (MB)", "peak_extra_mb", ".0f"),
        ("final loss", None, ".4f"),
    ]:
        print(f"  {metric:>25} |", end="")
        for r in results:
            if r.get("oom"):
                print(f" {'OOM':>20} |", end="")
            elif key:
                print(f" {r[key]:>20{fmt}} |", end="")
            else:
                val = r["losses"][-1] if r["losses"] else 0
                print(f" {val:>20{fmt}} |", end="")
        print()

    # Loss coherence check
    valid = [r for r in results if not r.get("oom") and r["losses"]]
    if len(valid) == 2:
        l0 = valid[0]["losses"][-1]
        l1 = valid[1]["losses"][-1]
        diff = abs(l0 - l1)
        rel = diff / max(abs(l0), 1e-6) * 100
        trend0 = "decreasing" if valid[0]["losses"][-1] < valid[0]["losses"][0] else "NOT decreasing"
        trend1 = "decreasing" if valid[1]["losses"][-1] < valid[1]["losses"][0] else "NOT decreasing"
        print(f"\n  Loss diff (final): {diff:.4f} ({rel:.1f}% relative)")
        print(f"  {valid[0]['label']}: {trend0} ({valid[0]['losses'][0]:.4f} -> {valid[0]['losses'][-1]:.4f})")
        print(f"  {valid[1]['label']}: {trend1} ({valid[1]['losses'][0]:.4f} -> {valid[1]['losses'][-1]:.4f})")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
