"""Fine-tuning with HuggingFace transformers on flash_attn_legacy (or SDPA / eager,
to compare): mixed precision (fp32 weights, fp16 autocast + GradScaler), packed
sequences of `--seq-len` tokens. To fit a 16 GB GPU without extra libraries the
attention projections (q, k, v, o) are trained and the rest is frozen; the
gradients always go through the attention backward.

    python examples/finetune.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
    python examples/finetune.py --compare --steps 30          # flash_attn_legacy vs sdpa
    python examples/finetune.py --text my_corpus.txt --seq-len 4096 --batch 1

Text: a local file (--text), default the repository's documentation. Reports
tokens per second, peak memory and the loss curve.
"""

import argparse
import pathlib
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import flash_attn_legacy

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_text(path):
    if path:
        return pathlib.Path(path).read_text()
    return "\n\n".join(p.read_text() for p in sorted([ROOT / "README.md", *(ROOT / "docs").rglob("*.md")]))


def batches(ids, seq_len, batch, steps, seed=0):
    """Random windows of seq_len + 1 tokens (inputs and shifted labels)."""
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        starts = torch.randint(0, len(ids) - seq_len - 1, (batch,), generator=g)
        chunk = torch.stack([ids[s:s + seq_len + 1] for s in starts.tolist()])
        yield chunk[:, :-1].cuda(), chunk[:, 1:].cuda()


def run(impl, ids, args):
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation=impl).cuda()
    model.gradient_checkpointing_enable() if args.checkpointing else None
    trainable = []
    for name, p in model.named_parameters():
        p.requires_grad_(any(f"{proj}_proj" in name for proj in ("q", "k", "v", "o")))
        if p.requires_grad:
            trainable.append(p)
        else:
            p.data = p.data.half()  # frozen weights in fp16 (autocast computes in fp16 anyway)
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")
    print(f"\n== {impl}: {sum(p.numel() for p in trainable) / 1e6:.0f}M trainable parameters")

    flash_attn_legacy.reset_fallback_stats()
    torch.cuda.reset_peak_memory_stats()
    losses, t_start, n_tokens = [], None, 0
    model.train()
    for step, (x, y) in enumerate(batches(ids, args.seq_len, args.batch, args.steps)):
        if step == args.warmup:
            torch.cuda.synchronize()
            t_start, n_tokens = time.perf_counter(), 0
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(input_ids=x).logits
            loss = torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), y.flatten())
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
        n_tokens += x.numel()
        if step % 10 == 0:
            print(f"  step {step:4d}  loss {losses[-1]:.4f}")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_start
    print(f"   {n_tokens / elapsed:,.0f} tok/s, {elapsed / (args.steps - args.warmup) * 1e3:.0f} ms/step, peak "
          f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, final loss {sum(losses[-5:]) / 5:.4f}, "
          f"fallbacks {flash_attn_legacy.fallback_stats() if impl == 'flash_attn_legacy' else '-'}")
    del model, opt
    torch.cuda.empty_cache()
    return losses


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--impl", default="flash_attn_legacy", choices=["flash_attn_legacy", "sdpa", "eager"])
    p.add_argument("--compare", action="store_true", help="run flash_attn_legacy and sdpa on the same batches")
    p.add_argument("--text", default=None, help="training text file (default: this repo's documentation)")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5, help="steps excluded from the timing")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--checkpointing", action="store_true", help="gradient checkpointing (longer sequences)")
    args = p.parse_args()

    print(flash_attn_legacy.get_device_info())
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ids = torch.tensor(tokenizer(load_text(args.text))["input_ids"])
    print(f"{len(ids):,} tokens of text")
    impls = ["flash_attn_legacy", "sdpa"] if args.compare else [args.impl]
    curves = {impl: run(impl, ids, args) for impl in impls}
    if args.compare:
        a, b = curves["flash_attn_legacy"], curves["sdpa"]
        print(f"\nmax |loss difference| over {len(a)} steps: {max(abs(x - y) for x, y in zip(a, b)):.4f}")


if __name__ == "__main__":
    main()
