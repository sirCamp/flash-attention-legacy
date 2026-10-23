# End-to-end benchmarks on HuggingFace models

`python benchmarks/models.py` (raw results: `models_<gpu>.jsonl` in this folder; these tables: `--report`). transformers 5.20.2,
torch 2.7.1, fp16 weights, kernels at commit 5ad45ca (decode kernel of step h, GQA backward split). GPUs: V100 32 GB, T4 16 GB,
P100 16 GB, GTX 1080 Ti 11 GB, RTX 2080 Ti 11 GB. Models: TinyLlama 1.1B, Qwen2.5-3B / 14B, Gemma 2 2B, Mistral 7B,
Llama 3.1 8B, Qwen3.5 2B / 4B / 9B (not every model on every GPU). Each model and implementation runs in its own processes (one
for prefill / generate, one for train), so nothing one backend compiles or caches is counted in another.

Each cell: throughput (tokens/s; decode tokens/s for generate) and peak GPU memory in GiB. Against the faster of SDPA and eager:
`speedup` = our throughput / theirs, `time` = our time / theirs (below 1 we are faster), `memory saved` = the smaller of their
peaks minus ours (and its share of theirs). OOM = out of memory on that GPU; `only ours` = only flash_attn_legacy fits.
The last table of each GPU gives, per case, the longest N each implementation ran: a plain number is a limit (it was out of
memory at the next length), `>= N` means no longer N was measured.

- prefill: forward of B=1 x N tokens (`logits_to_keep=1`).
- generate: 128 greedy new tokens after a B x N prompt, prefill time subtracted; `generate` = DynamicCache, eager steps;
  `generate_static_compiled` = a reused StaticCache with HF's compiled decode step (SM >= 7.0: inductor needs Triton). Every
  compiled step is one graph, no graph break, for all three backends.
- train: one AdamW step, fp16 autocast + GradScaler; trainable in fp32 the attention projections (`train`), or rank-16 LoRA
  adapters on them (`train_lora`, peft), the rest frozen in fp16; `_ckpt` = gradient checkpointing. B=1 is one full row;
  B=4 is a padded batch (lengths N down to N/2, right padding, attention mask: our varlen path), tokens/s counting real tokens.
- Beyond the main lengths (16k prefill, 8k prompts, 8k train), our kernel alone was run up to 128k prefill, 64k prompts and
  64k train with checkpointing, one repetition each, to find its limit.

Correctness.
- The loss of the train step agrees across the backends in all 71 train cases where more than one fits: largest difference
  0.0026 (Llama 3.1 8B LoRA, B=4, 2k: 11.7434 vs 11.7460), fp16 noise; padded batches and LoRA included.
- Top-1 agreement of the prefill logits with eager (512 tokens): 0.992-1.000 for ours, 0.988-1.000 for SDPA (25 model / GPU
  pairs).
- No flash_attn_legacy row has a fallback.

What bounds the length.
- Inference: the attention, O(N) memory on our kernel and O(N^2) in SDPA / eager here, so we run up to 32x longer prefills
  and prompts (TinyLlama on the T4 / P100: >= 128k against 4k; Llama 3.1 8B on the V100: >= 16k against 4k, on the P100
  2k against 1k), and the same length where the weights or the KV cache, not the attention, set the limit.
- Train without checkpointing: the activations of every layer, for all backends; ours still reaches 2-4x longer on TinyLlama
  and Qwen2.5-3B (the same length on Gemma 2, the Qwen3.5 models and padded batches).
- Train with checkpointing: the attention still decides on TinyLlama (32k vocabulary: ours 8-32k, the others 2-4k). On the
  large-vocabulary models (Qwen2.5, Llama 3.1: vocabularies of 128k-152k) the logits take most of the memory: HF computes
  the full-vocabulary logits and the loss in fp32, several GiB at 8k, so where all three backends fit their peaks are nearly
  equal and the length gain shrinks (LoRA on the V100: Qwen2.5-3B 8k for all, Llama 3.1 8B 8k against 4k). A chunked
  cross-entropy would move that limit; it is in the model, not in the attention. The speedup remains (LoRA Qwen2.5-3B 4k:
  1.53x on the V100, 2.69x on the T4).
- Weights: Qwen2.5-14B (27.75 GiB) leaves no room for LoRA training on 32 GB, Llama 3.1 8B (14.97 GiB) almost none on 16 GB.

Reading the numbers.
- Rows bounded by the host (decode with eager steps, the Qwen3.5 models) follow the load of the shared cloud box while they run,
  by up to ~30% between runs (seen when cases were measured again, e.g. V100 Gemma 2 2B decode at 8k and 1080 Ti Qwen3.5-2B
  at 2k). The tables keep the last run of each case, so a ratio within ~0.8-1.2x on these rows is not a kernel result.
  The compiled decode steps (CUDA graphs) and the long prefill and train
  cases are not host bound.
- The OOM boundary is where a case needs about the whole GPU: there it depends on the allocator's fragmentation, and a case one
  step from the boundary may fit in one run and not in another (e.g. on the 2080 Ti Qwen2.5-3B fits a 32k prompt in generate
  but not a 32k prefill).
- Qwen3.5 (Gated DeltaNet + 1 full-attention layer in 4) runs its linear layers on transformers' PyTorch reference here
  (flash-linear-attention's Triton kernels do not compile for SM < 8.0): host bound (~4900 kernel launches per 1k prefill), so
  its rows measure mostly the memory saving and the longer context.
- SDPA on these GPUs has no flash backend: it runs the memory-efficient or the math kernel (with a mask and GQA, HF repeats the
  KV heads).

#### NVIDIA GeForce GTX 1080 Ti: prefill (tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 1024 | 2,724 (2.13) | 2,149 (2.41) | 2,301 (2.36) | 1.18x | 0.84x | 0.23 GiB (10%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 4096 | 1,903 (2.34) | 1,110 (6.9) | 1,022 (6.27) | 1.71x | 0.58x | 3.93 GiB (63%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | 1,538 (2.62) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 16384 | 1,247 (3.17) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 1024 | 1,007 (5.98) | 919 (6.09) | 939 (6.05) | 1.07x | 0.93x | 0.07 GiB (1%) |
| Qwen2.5-3B | fp16 | 1 | 4096 | 723 (6.32) | 572 (8.51) | 610 (8.13) | 1.19x | 0.84x | 1.81 GiB (22%) |
| Qwen2.5-3B | fp16 | 1 | 8192 | 585 (6.78) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 16384 | 424 (7.69) | OOM | OOM | only ours | - | - |
| Qwen3.5-2B | fp16 | 1 | 1024 | 1,781 (3.66) | 1,768 (3.67) | 1,732 (3.66) | 1.01x | 0.99x | 0.00 GiB (0%) |
| Qwen3.5-2B | fp16 | 1 | 4096 | 1,367 (4.04) | 1,295 (4.99) | 1,320 (4.72) | 1.04x | 0.97x | 0.68 GiB (14%) |
| Qwen3.5-2B | fp16 | 1 | 8192 | 1,239 (4.55) | 1,192 (8.81) | 1,192 (7.97) | 1.04x | 0.96x | 3.42 GiB (43%) |
| Qwen3.5-2B | fp16 | 1 | 16384 | 1,047 (5.57) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 32768 | 847 (4.29) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 65536 | 533 (6.51) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 131072 | OOM | - | - | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 32768 | 323 (9.52) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 65536 | OOM | - | - | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 131072 | OOM | - | - | - | - | - |

#### NVIDIA GeForce GTX 1080 Ti: train (tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 1024 | 919 (6.23) | 904 (9.26) | 909 (10.38) | 1.01x | 0.99x | 3.03 GiB (33%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 2048 | 878 (7.95) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 1024 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 2048 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 1024 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 2048 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 512 | 943 (7.85) | 314 (8.12) | OOM | 3.00x | 0.33x | 0.27 GiB (3%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 1024 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 2048 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 4096 | OOM | OOM | OOM | - | - | - |

#### NVIDIA GeForce GTX 1080 Ti: train_ckpt (tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 2048 | 720 (5.56) | 588 (6.85) | 534 (6.58) | 1.22x | 0.82x | 1.02 GiB (16%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 4096 | 495 (6.0) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | 345 (7.68) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 2048 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 2048 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 32768 | OOM | - | - | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 65536 | OOM | - | - | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 32768 | OOM | - | - | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 65536 | OOM | - | - | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 1024 | 556 (6.0) | 145 (6.0) | 470 (6.65) | 1.18x | 0.85x | 0.00 GiB (0%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 2048 | 459 (7.68) | 81 (7.7) | OOM | 5.67x | 0.18x | 0.02 GiB (0%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 4096 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 8192 | OOM | OOM | OOM | - | - | - |

#### NVIDIA GeForce GTX 1080 Ti: generate (decode tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 2048 | 37.3 (2.2) | 31.3 (3.34) | 32.8 (3.16) | 1.14x | 0.88x | 0.96 GiB (30%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | 37.9 (2.62) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 8 | 2048 | 222.2 (3.17) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 8 | 8192 | 151.5 (6.51) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 2048 | 21.1 (6.09) | 18.9 (6.61) | 19.2 (6.49) | 1.10x | 0.91x | 0.40 GiB (6%) |
| Qwen2.5-3B | fp16 | 1 | 8192 | 21.2 (6.78) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 8 | 2048 | 87.1 (7.69) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 8 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 2048 | 18.7 (3.79) | 18.0 (3.96) | 18.1 (3.87) | 1.03x | 0.97x | 0.08 GiB (2%) |
| Qwen3.5-2B | fp16 | 1 | 8192 | 18.7 (4.55) | 17.3 (8.82) | 18.1 (7.97) | 1.03x | 0.97x | 3.42 GiB (43%) |
| Qwen3.5-2B | fp16 | 8 | 2048 | 132.0 (5.71) | 98.2 (6.99) | 117.8 (6.35) | 1.12x | 0.89x | 0.64 GiB (10%) |
| Qwen3.5-2B | fp16 | 8 | 8192 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 16384 | 31.5 (3.17) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 32768 | 34.9 (4.29) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 65536 | 29.6 (6.51) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 16384 | 16.8 (7.69) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 32768 | 17.0 (9.52) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 65536 | OOM | - | - | - | - | - |

#### NVIDIA GeForce GTX 1080 Ti: train_lora_ckpt (tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-3B (LoRA r16) | fp16 | 1 | 2048 | 251 (9.73) | 246 (9.73) | 247 (9.74) | 1.02x | 0.98x | 0.00 GiB (0%) |
| Qwen2.5-3B (LoRA r16) | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B (LoRA r16) | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B (LoRA r16) | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B (LoRA r16) | fp16 | 1 | 32768 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B (LoRA r16) | fp16 | 4 | 2048 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B (LoRA r16) | fp16 | 4 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B (LoRA r16) | fp16 | 4 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B (LoRA r16) | fp16 | 4 | 16384 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B (LoRA r16) | fp16 | 4 | 32768 | OOM | OOM | OOM | - | - | - |

#### NVIDIA GeForce GTX 1080 Ti: longest N that fits (>= N: no longer N measured)

| model | mode | dtype | B | flash_attn_legacy | sdpa | eager |
|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | prefill | fp16 | 1 | 65,536 | 4,096 | 4,096 |
| TinyLlama-1.1B-Chat-v1.0 | train | fp16 | 1 | 2,048 | 1,024 | 1,024 |
| Qwen2.5-3B | prefill | fp16 | 1 | 32,768 | 4,096 | 4,096 |
| Qwen2.5-3B | train | fp16 | 1 | OOM | OOM | OOM |
| Qwen3.5-2B | prefill | fp16 | 1 | >= 16,384 | 8,192 | 8,192 |
| Qwen3.5-2B | train | fp16 | 1 | OOM | OOM | OOM |
| TinyLlama-1.1B-Chat-v1.0 | train_ckpt | fp16 | 1 | 8,192 | 2,048 | 2,048 |
| Qwen2.5-3B | train_ckpt | fp16 | 1 | OOM | OOM | OOM |
| Qwen3.5-2B | train_ckpt | fp16 | 1 | OOM | OOM | OOM |
| TinyLlama-1.1B-Chat-v1.0 | generate | fp16 | 1 | >= 65,536 | 2,048 | 2,048 |
| TinyLlama-1.1B-Chat-v1.0 | generate | fp16 | 8 | >= 8,192 | OOM | OOM |
| Qwen2.5-3B | generate | fp16 | 1 | 32,768 | 2,048 | 2,048 |
| Qwen2.5-3B | generate | fp16 | 8 | 2,048 | OOM | OOM |
| Qwen3.5-2B | generate | fp16 | 1 | >= 8,192 | >= 8,192 | >= 8,192 |
| Qwen3.5-2B | generate | fp16 | 8 | 2,048 | 2,048 | 2,048 |
| TinyLlama-1.1B-Chat-v1.0 | train | fp16 | 4 | 512 | 512 | OOM |
| TinyLlama-1.1B-Chat-v1.0 | train_ckpt | fp16 | 4 | 2,048 | 2,048 | 1,024 |
| Qwen2.5-3B (LoRA r16) | train_lora_ckpt | fp16 | 1 | 2,048 | 2,048 | 2,048 |
| Qwen2.5-3B (LoRA r16) | train_lora_ckpt | fp16 | 4 | OOM | OOM | OOM |

#### NVIDIA GeForce RTX 2080 Ti: prefill (tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 1024 | 16,126 (2.13) | 7,022 (2.41) | 10,246 (2.36) | 1.57x | 0.64x | 0.23 GiB (10%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 4096 | 15,555 (2.34) | 2,797 (6.9) | 4,748 (6.27) | 3.28x | 0.31x | 3.93 GiB (63%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | 13,803 (2.62) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 16384 | 11,047 (3.17) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 1024 | 6,261 (5.98) | 4,206 (6.09) | 5,336 (6.05) | 1.17x | 0.85x | 0.07 GiB (1%) |
| Qwen2.5-3B | fp16 | 1 | 4096 | 6,344 (6.32) | 2,236 (8.51) | 3,610 (8.13) | 1.76x | 0.57x | 1.81 GiB (22%) |
| Qwen2.5-3B | fp16 | 1 | 8192 | 5,872 (6.78) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 16384 | 5,025 (7.69) | OOM | OOM | only ours | - | - |
| Qwen3.5-2B | fp16 | 1 | 1024 | 7,392 (3.66) | 6,939 (3.67) | 7,267 (3.66) | 1.02x | 0.98x | 0.00 GiB (0%) |
| Qwen3.5-2B | fp16 | 1 | 4096 | 7,567 (4.04) | 6,023 (4.99) | 6,988 (4.72) | 1.08x | 0.92x | 0.68 GiB (14%) |
| Qwen3.5-2B | fp16 | 1 | 8192 | 7,344 (4.55) | 4,545 (8.81) | 6,404 (7.97) | 1.15x | 0.87x | 3.42 GiB (43%) |
| Qwen3.5-2B | fp16 | 1 | 16384 | 7,034 (5.57) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 32768 | 8,090 (4.29) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 65536 | 5,009 (6.51) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 131072 | OOM | - | - | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 32768 | OOM | - | - | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 65536 | OOM | - | - | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 131072 | OOM | - | - | - | - | - |

#### NVIDIA GeForce RTX 2080 Ti: generate (decode tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 2048 | 66.0 (2.2) | 57.6 (3.34) | 61.8 (3.16) | 1.07x | 0.94x | 0.96 GiB (30%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | 66.5 (2.62) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 8 | 2048 | 508.7 (3.17) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 8 | 8192 | 444.2 (6.51) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 2048 | 39.2 (6.09) | 35.3 (6.61) | 35.9 (6.49) | 1.09x | 0.92x | 0.40 GiB (6%) |
| Qwen2.5-3B | fp16 | 1 | 8192 | 39.2 (6.78) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 8 | 2048 | 296.2 (7.69) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 8 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 2048 | 33.8 (3.79) | 33.1 (3.96) | 33.6 (3.87) | 1.01x | 0.99x | 0.08 GiB (2%) |
| Qwen3.5-2B | fp16 | 1 | 8192 | 33.1 (4.55) | 31.9 (8.82) | 32.6 (7.97) | 1.02x | 0.98x | 3.42 GiB (43%) |
| Qwen3.5-2B | fp16 | 8 | 2048 | 248.9 (5.71) | 239.6 (6.99) | 251.1 (6.35) | 0.99x | 1.01x | 0.64 GiB (10%) |
| Qwen3.5-2B | fp16 | 8 | 8192 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 16384 | 65.9 (3.17) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 32768 | 66.1 (4.29) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 65536 | 57.3 (6.51) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 16384 | 40.0 (7.69) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 32768 | 38.6 (9.52) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 65536 | OOM | - | - | - | - | - |

#### NVIDIA GeForce RTX 2080 Ti: generate_static_compiled (decode tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 2048 | 174.7 (2.21) | 101.1 (3.35) | 133.5 (3.23) | 1.31x | 0.76x | 1.02 GiB (32%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | 154.2 (2.62) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 8 | 2048 | 1,080.1 (3.19) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 8 | 8192 | 821.6 (6.54) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 2048 | 69.5 (6.12) | 51.4 (6.63) | 59.0 (6.54) | 1.18x | 0.85x | 0.42 GiB (6%) |
| Qwen2.5-3B | fp16 | 1 | 8192 | 67.3 (6.78) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 8 | 2048 | 477.3 (7.73) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 8 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 2048 | 103.9 (3.79) | 90.6 (3.96) | 99.6 (3.89) | 1.04x | 0.96x | 0.10 GiB (3%) |
| Qwen3.5-2B | fp16 | 1 | 8192 | 102.9 (4.57) | 62.8 (8.82) | 79.9 (8.04) | 1.29x | 0.78x | 3.47 GiB (43%) |
| Qwen3.5-2B | fp16 | 8 | 2048 | 681.8 (5.76) | 541.0 (7.01) | 542.3 (6.5) | 1.26x | 0.80x | 0.74 GiB (11%) |
| Qwen3.5-2B | fp16 | 8 | 8192 | OOM | OOM | OOM | - | - | - |

#### NVIDIA GeForce RTX 2080 Ti: train (tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 1024 | 4,907 (6.23) | 2,893 (9.25) | OOM | 1.70x | 0.59x | 3.02 GiB (33%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 2048 | 5,123 (7.95) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 1024 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 2048 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 1024 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 2048 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 512 | 4,490 (7.85) | 4,238 (8.12) | OOM | 1.06x | 0.94x | 0.27 GiB (3%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 1024 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 2048 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 4096 | OOM | OOM | OOM | - | - | - |

#### NVIDIA GeForce RTX 2080 Ti: train_ckpt (tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 2048 | 4,049 (5.56) | 1,514 (6.85) | 1,991 (6.58) | 2.03x | 0.49x | 1.02 GiB (16%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 4096 | 3,864 (6.0) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | 3,181 (7.68) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 2048 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 2048 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 32768 | OOM | - | - | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 65536 | OOM | - | - | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 32768 | OOM | - | - | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 65536 | OOM | - | - | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 1024 | 3,674 (6.0) | 3,190 (6.0) | 2,320 (6.65) | 1.15x | 0.87x | 0.00 GiB (0%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 2048 | 3,626 (7.68) | 2,726 (7.7) | OOM | 1.33x | 0.75x | 0.02 GiB (0%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 4096 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 8192 | OOM | OOM | OOM | - | - | - |

#### NVIDIA GeForce RTX 2080 Ti: train_lora_ckpt (tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-3B (LoRA r16) | fp16 | 1 | 2048 | 1,776 (9.73) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B (LoRA r16) | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B (LoRA r16) | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B (LoRA r16) | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B (LoRA r16) | fp16 | 1 | 32768 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B (LoRA r16) | fp16 | 4 | 2048 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B (LoRA r16) | fp16 | 4 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B (LoRA r16) | fp16 | 4 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B (LoRA r16) | fp16 | 4 | 16384 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B (LoRA r16) | fp16 | 4 | 32768 | OOM | OOM | OOM | - | - | - |

#### NVIDIA GeForce RTX 2080 Ti: longest N that fits (>= N: no longer N measured)

| model | mode | dtype | B | flash_attn_legacy | sdpa | eager |
|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | prefill | fp16 | 1 | 65,536 | 4,096 | 4,096 |
| TinyLlama-1.1B-Chat-v1.0 | generate | fp16 | 1 | >= 65,536 | 2,048 | 2,048 |
| TinyLlama-1.1B-Chat-v1.0 | generate_static_compiled | fp16 | 1 | >= 8,192 | 2,048 | 2,048 |
| TinyLlama-1.1B-Chat-v1.0 | generate | fp16 | 8 | >= 8,192 | OOM | OOM |
| TinyLlama-1.1B-Chat-v1.0 | generate_static_compiled | fp16 | 8 | >= 8,192 | OOM | OOM |
| TinyLlama-1.1B-Chat-v1.0 | train | fp16 | 1 | 2,048 | 1,024 | OOM |
| TinyLlama-1.1B-Chat-v1.0 | train_ckpt | fp16 | 1 | 8,192 | 2,048 | 2,048 |
| Qwen2.5-3B | prefill | fp16 | 1 | 16,384 | 4,096 | 4,096 |
| Qwen2.5-3B | generate | fp16 | 1 | 32,768 | 2,048 | 2,048 |
| Qwen2.5-3B | generate_static_compiled | fp16 | 1 | >= 8,192 | 2,048 | 2,048 |
| Qwen2.5-3B | generate | fp16 | 8 | 2,048 | OOM | OOM |
| Qwen2.5-3B | generate_static_compiled | fp16 | 8 | 2,048 | OOM | OOM |
| Qwen2.5-3B | train | fp16 | 1 | OOM | OOM | OOM |
| Qwen2.5-3B | train_ckpt | fp16 | 1 | OOM | OOM | OOM |
| Qwen3.5-2B | prefill | fp16 | 1 | >= 16,384 | 8,192 | 8,192 |
| Qwen3.5-2B | generate | fp16 | 1 | >= 8,192 | >= 8,192 | >= 8,192 |
| Qwen3.5-2B | generate_static_compiled | fp16 | 1 | >= 8,192 | >= 8,192 | >= 8,192 |
| Qwen3.5-2B | generate | fp16 | 8 | 2,048 | 2,048 | 2,048 |
| Qwen3.5-2B | generate_static_compiled | fp16 | 8 | 2,048 | 2,048 | 2,048 |
| Qwen3.5-2B | train | fp16 | 1 | OOM | OOM | OOM |
| Qwen3.5-2B | train_ckpt | fp16 | 1 | OOM | OOM | OOM |
| TinyLlama-1.1B-Chat-v1.0 | train | fp16 | 4 | 512 | 512 | OOM |
| TinyLlama-1.1B-Chat-v1.0 | train_ckpt | fp16 | 4 | 2,048 | 2,048 | 1,024 |
| Qwen2.5-3B (LoRA r16) | train_lora_ckpt | fp16 | 1 | 2,048 | OOM | OOM |
| Qwen2.5-3B (LoRA r16) | train_lora_ckpt | fp16 | 4 | OOM | OOM | OOM |

#### Tesla P100-PCIE-16GB: prefill (tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 1024 | 3,259 (2.13) | 2,590 (2.41) | 2,778 (2.36) | 1.17x | 0.85x | 0.23 GiB (10%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 4096 | 3,096 (2.34) | 1,523 (6.9) | 1,804 (6.27) | 1.72x | 0.58x | 3.93 GiB (63%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | 2,505 (2.62) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 16384 | 1,858 (3.17) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 1024 | 1,058 (5.98) | 1,028 (6.09) | 1,056 (6.05) | 1.00x | 1.00x | 0.07 GiB (1%) |
| Qwen2.5-3B | fp16 | 1 | 4096 | 1,077 (6.32) | 871 (8.51) | 946 (8.13) | 1.14x | 0.88x | 1.81 GiB (22%) |
| Qwen2.5-3B | fp16 | 1 | 8192 | 929 (6.78) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 16384 | 709 (7.69) | OOM | OOM | only ours | - | - |
| Mistral-7B-Instruct-v0.2 | fp16 | 1 | 1024 | 460 (13.74) | 451 (14.01) | 456 (13.92) | 1.01x | 0.99x | 0.18 GiB (1%) |
| Mistral-7B-Instruct-v0.2 | fp16 | 1 | 4096 | 461 (14.45) | OOM | OOM | only ours | - | - |
| Mistral-7B-Instruct-v0.2 | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 1024 | 2,034 (3.66) | 2,029 (3.67) | 2,039 (3.66) | 1.00x | 1.00x | 0.00 GiB (0%) |
| Qwen3.5-2B | fp16 | 1 | 4096 | 2,138 (4.04) | 2,114 (4.99) | 2,158 (4.72) | 0.99x | 1.01x | 0.68 GiB (14%) |
| Qwen3.5-2B | fp16 | 1 | 8192 | 1,897 (4.55) | 1,828 (8.81) | 1,903 (7.97) | 1.00x | 1.00x | 3.42 GiB (43%) |
| Qwen3.5-2B | fp16 | 1 | 16384 | 1,613 (5.57) | OOM | OOM | only ours | - | - |
| Llama-3.1-8B-Instruct | fp16 | 1 | 512 | 459 (15.08) | 464 (15.15) | 469 (15.11) | 0.98x | 1.02x | 0.03 GiB (0%) |
| Llama-3.1-8B-Instruct | fp16 | 1 | 1024 | 459 (15.2) | 453 (15.48) | 462 (15.39) | 0.99x | 1.01x | 0.19 GiB (1%) |
| Llama-3.1-8B-Instruct | fp16 | 1 | 2048 | 530 (15.44) | OOM | OOM | only ours | - | - |
| Llama-3.1-8B-Instruct | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Llama-3.1-8B-Instruct | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Llama-3.1-8B-Instruct | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-4B | fp16 | 1 | 1024 | 767 (8.26) | 765 (8.27) | 776 (8.26) | 0.99x | 1.01x | 0.00 GiB (0%) |
| Qwen3.5-4B | fp16 | 1 | 4096 | 822 (9.03) | 801 (10.82) | 813 (10.33) | 1.01x | 0.99x | 1.30 GiB (13%) |
| Qwen3.5-4B | fp16 | 1 | 8192 | 759 (10.08) | OOM | OOM | only ours | - | - |
| Qwen3.5-4B | fp16 | 1 | 16384 | 643 (12.17) | OOM | OOM | only ours | - | - |
| Qwen3.5-4B | fp16 | 1 | 32768 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 32768 | 1,298 (4.29) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 65536 | 786 (6.51) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 131072 | 441 (10.97) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 32768 | 500 (9.52) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 65536 | 314 (13.18) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 131072 | OOM | - | - | - | - | - |

#### Tesla P100-PCIE-16GB: train (tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 1024 | 1,183 (6.23) | 1,085 (9.26) | 1,104 (10.38) | 1.07x | 0.93x | 3.03 GiB (33%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 2048 | 1,116 (7.95) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 4096 | 995 (11.53) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 1024 | 430 (14.84) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 2048 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 1024 | 638 (11.49) | 642 (11.76) | 644 (11.78) | 0.99x | 1.01x | 0.27 GiB (2%) |
| Qwen3.5-2B | fp16 | 1 | 2048 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Mistral-7B-Instruct-v0.2 | fp16 | 1 | 1024 | OOM | OOM | OOM | - | - | - |
| Mistral-7B-Instruct-v0.2 | fp16 | 1 | 2048 | OOM | OOM | OOM | - | - | - |
| Mistral-7B-Instruct-v0.2 | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Mistral-7B-Instruct-v0.2 | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-4B | fp16 | 1 | 1024 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-4B | fp16 | 1 | 2048 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-4B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-4B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 512 | 1,056 (7.85) | 978 (8.12) | 991 (12.2) | 1.07x | 0.94x | 0.27 GiB (3%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 1024 | 1,071 (11.35) | 898 (11.96) | OOM | 1.19x | 0.84x | 0.61 GiB (5%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 2048 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 4096 | OOM | OOM | OOM | - | - | - |

#### Tesla P100-PCIE-16GB: train_ckpt (tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 2048 | 886 (5.56) | 692 (6.85) | 705 (6.58) | 1.26x | 0.80x | 1.02 GiB (16%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 4096 | 790 (6.0) | 496 (12.94) | 537 (11.92) | 1.47x | 0.68x | 5.92 GiB (50%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | 589 (7.68) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 16384 | 386 (11.36) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 2048 | 345 (12.86) | 333 (12.86) | 335 (12.87) | 1.03x | 0.97x | 0.00 GiB (0%) |
| Qwen2.5-3B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 2048 | 445 (10.23) | 449 (10.23) | 449 (10.24) | 0.99x | 1.01x | 0.00 GiB (0%) |
| Qwen3.5-2B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-4B | fp16 | 1 | 2048 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-4B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-4B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-4B | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 32768 | OOM | - | - | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 65536 | OOM | - | - | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 32768 | OOM | - | - | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 65536 | OOM | - | - | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 1024 | 826 (6.0) | 699 (6.0) | 691 (6.65) | 1.18x | 0.85x | 0.00 GiB (0%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 2048 | 769 (7.68) | 555 (7.7) | 534 (12.05) | 1.39x | 0.72x | 0.02 GiB (0%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 4096 | 658 (11.36) | 395 (11.42) | OOM | 1.67x | 0.60x | 0.06 GiB (1%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 8192 | OOM | OOM | OOM | - | - | - |

#### Tesla P100-PCIE-16GB: generate (decode tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 2048 | 79.4 (2.2) | 54.7 (3.34) | 63.3 (3.16) | 1.25x | 0.80x | 0.96 GiB (30%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | 77.8 (2.62) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 8 | 2048 | 285.1 (3.17) | 96.6 (12.2) | 122.6 (10.84) | 2.33x | 0.43x | 7.67 GiB (71%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 8 | 8192 | 193.7 (6.51) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 2048 | 37.4 (6.09) | 25.1 (6.61) | 29.3 (6.49) | 1.28x | 0.78x | 0.40 GiB (6%) |
| Qwen2.5-3B | fp16 | 1 | 8192 | 33.8 (6.78) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 8 | 2048 | 110.7 (7.69) | 52.1 (11.73) | 63.3 (10.87) | 1.75x | 0.57x | 3.18 GiB (29%) |
| Qwen2.5-3B | fp16 | 8 | 8192 | 84.0 (13.18) | OOM | OOM | only ours | - | - |
| Mistral-7B-Instruct-v0.2 | fp16 | 1 | 2048 | 21.1 (13.97) | 14.9 (15.09) | 16.8 (14.85) | 1.26x | 0.80x | 0.88 GiB (6%) |
| Mistral-7B-Instruct-v0.2 | fp16 | 8 | 2048 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 2048 | 39.7 (3.79) | 38.8 (3.96) | 39.0 (3.87) | 1.02x | 0.98x | 0.08 GiB (2%) |
| Qwen3.5-2B | fp16 | 1 | 8192 | 39.8 (4.55) | 33.2 (8.82) | 38.0 (7.97) | 1.05x | 0.95x | 3.42 GiB (43%) |
| Qwen3.5-2B | fp16 | 8 | 2048 | 171.5 (5.71) | 130.5 (6.99) | 143.9 (6.35) | 1.19x | 0.84x | 0.64 GiB (10%) |
| Qwen3.5-2B | fp16 | 8 | 8192 | 147.4 (11.83) | OOM | OOM | only ours | - | - |
| Llama-3.1-8B-Instruct | fp16 | 1 | 512 | 21.4 (15.08) | 18.8 (15.15) | 19.9 (15.11) | 1.08x | 0.93x | 0.03 GiB (0%) |
| Llama-3.1-8B-Instruct | fp16 | 1 | 2048 | 20.3 (15.44) | OOM | OOM | only ours | - | - |
| Llama-3.1-8B-Instruct | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Llama-3.1-8B-Instruct | fp16 | 8 | 512 | OOM | OOM | OOM | - | - | - |
| Llama-3.1-8B-Instruct | fp16 | 8 | 2048 | OOM | OOM | OOM | - | - | - |
| Llama-3.1-8B-Instruct | fp16 | 8 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-4B | fp16 | 1 | 2048 | 25.2 (8.51) | 21.7 (8.83) | 23.3 (8.65) | 1.08x | 0.92x | 0.14 GiB (2%) |
| Qwen3.5-4B | fp16 | 1 | 8192 | 23.9 (10.08) | OOM | OOM | only ours | - | - |
| Qwen3.5-4B | fp16 | 8 | 2048 | 71.7 (12.52) | 53.5 (14.95) | 59.8 (13.62) | 1.20x | 0.83x | 1.10 GiB (8%) |
| Qwen3.5-4B | fp16 | 8 | 8192 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 16384 | 63.9 (3.17) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 32768 | 49.2 (4.29) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 65536 | 40.8 (6.51) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 16384 | 29.5 (7.69) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 32768 | 23.8 (9.52) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 65536 | 16.9 (13.18) | - | - | only ours | - | - |

#### Tesla P100-PCIE-16GB: train_lora_ckpt (tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct (LoRA r16) | fp16 | 1 | 2048 | OOM | OOM | OOM | - | - | - |
| Llama-3.1-8B-Instruct (LoRA r16) | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Llama-3.1-8B-Instruct (LoRA r16) | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Llama-3.1-8B-Instruct (LoRA r16) | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |
| Llama-3.1-8B-Instruct (LoRA r16) | fp16 | 1 | 32768 | OOM | OOM | OOM | - | - | - |
| Llama-3.1-8B-Instruct (LoRA r16) | fp16 | 4 | 2048 | OOM | OOM | OOM | - | - | - |
| Llama-3.1-8B-Instruct (LoRA r16) | fp16 | 4 | 4096 | OOM | OOM | OOM | - | - | - |
| Llama-3.1-8B-Instruct (LoRA r16) | fp16 | 4 | 8192 | OOM | OOM | OOM | - | - | - |
| Llama-3.1-8B-Instruct (LoRA r16) | fp16 | 4 | 16384 | OOM | OOM | OOM | - | - | - |
| Llama-3.1-8B-Instruct (LoRA r16) | fp16 | 4 | 32768 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B (LoRA r16) | fp16 | 1 | 2048 | 352 (9.73) | 339 (9.73) | 343 (9.74) | 1.03x | 0.97x | 0.00 GiB (0%) |
| Qwen2.5-3B (LoRA r16) | fp16 | 1 | 4096 | 312 (13.5) | 286 (13.5) | OOM | 1.09x | 0.92x | 0.00 GiB (0%) |
| Qwen2.5-3B (LoRA r16) | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B (LoRA r16) | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B (LoRA r16) | fp16 | 1 | 32768 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B (LoRA r16) | fp16 | 4 | 2048 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B (LoRA r16) | fp16 | 4 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B (LoRA r16) | fp16 | 4 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B (LoRA r16) | fp16 | 4 | 16384 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B (LoRA r16) | fp16 | 4 | 32768 | OOM | OOM | OOM | - | - | - |
| Llama-3.1-8B-Instruct (LoRA r16) | fp16 | 1 | 512 | OOM | OOM | OOM | - | - | - |
| Llama-3.1-8B-Instruct (LoRA r16) | fp16 | 1 | 1024 | OOM | OOM | OOM | - | - | - |

#### Tesla P100-PCIE-16GB: longest N that fits (>= N: no longer N measured)

| model | mode | dtype | B | flash_attn_legacy | sdpa | eager |
|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | prefill | fp16 | 1 | >= 131,072 | 4,096 | 4,096 |
| TinyLlama-1.1B-Chat-v1.0 | train | fp16 | 1 | 4,096 | 1,024 | 1,024 |
| Qwen2.5-3B | prefill | fp16 | 1 | 65,536 | 4,096 | 4,096 |
| Qwen2.5-3B | train | fp16 | 1 | 1,024 | OOM | OOM |
| Mistral-7B-Instruct-v0.2 | prefill | fp16 | 1 | 4,096 | 1,024 | 1,024 |
| Qwen3.5-2B | prefill | fp16 | 1 | >= 16,384 | 8,192 | 8,192 |
| Qwen3.5-2B | train | fp16 | 1 | 1,024 | 1,024 | 1,024 |
| Mistral-7B-Instruct-v0.2 | train | fp16 | 1 | OOM | OOM | OOM |
| TinyLlama-1.1B-Chat-v1.0 | train_ckpt | fp16 | 1 | 16,384 | 4,096 | 4,096 |
| Qwen2.5-3B | train_ckpt | fp16 | 1 | 2,048 | 2,048 | 2,048 |
| Qwen3.5-2B | train_ckpt | fp16 | 1 | 2,048 | 2,048 | 2,048 |
| TinyLlama-1.1B-Chat-v1.0 | generate | fp16 | 1 | >= 65,536 | 2,048 | 2,048 |
| TinyLlama-1.1B-Chat-v1.0 | generate | fp16 | 8 | >= 8,192 | 2,048 | 2,048 |
| Qwen2.5-3B | generate | fp16 | 1 | >= 65,536 | 2,048 | 2,048 |
| Qwen2.5-3B | generate | fp16 | 8 | >= 8,192 | 2,048 | 2,048 |
| Mistral-7B-Instruct-v0.2 | generate | fp16 | 1 | >= 2,048 | >= 2,048 | >= 2,048 |
| Mistral-7B-Instruct-v0.2 | generate | fp16 | 8 | OOM | OOM | OOM |
| Qwen3.5-2B | generate | fp16 | 1 | >= 8,192 | >= 8,192 | >= 8,192 |
| Qwen3.5-2B | generate | fp16 | 8 | >= 8,192 | 2,048 | 2,048 |
| Llama-3.1-8B-Instruct | prefill | fp16 | 1 | 2,048 | 1,024 | 1,024 |
| Llama-3.1-8B-Instruct | generate | fp16 | 1 | 2,048 | 512 | 512 |
| Llama-3.1-8B-Instruct | generate | fp16 | 8 | OOM | OOM | OOM |
| Qwen3.5-4B | prefill | fp16 | 1 | 16,384 | 4,096 | 4,096 |
| Qwen3.5-4B | generate | fp16 | 1 | >= 8,192 | 2,048 | 2,048 |
| Qwen3.5-4B | generate | fp16 | 8 | 2,048 | 2,048 | 2,048 |
| Qwen3.5-4B | train | fp16 | 1 | OOM | OOM | OOM |
| Qwen3.5-4B | train_ckpt | fp16 | 1 | OOM | OOM | OOM |
| Llama-3.1-8B-Instruct (LoRA r16) | train_lora_ckpt | fp16 | 1 | OOM | OOM | OOM |
| Llama-3.1-8B-Instruct (LoRA r16) | train_lora_ckpt | fp16 | 4 | OOM | OOM | OOM |
| TinyLlama-1.1B-Chat-v1.0 | train | fp16 | 4 | 1,024 | 1,024 | 512 |
| TinyLlama-1.1B-Chat-v1.0 | train_ckpt | fp16 | 4 | 4,096 | 4,096 | 2,048 |
| Qwen2.5-3B (LoRA r16) | train_lora_ckpt | fp16 | 1 | 4,096 | 4,096 | 2,048 |
| Qwen2.5-3B (LoRA r16) | train_lora_ckpt | fp16 | 4 | OOM | OOM | OOM |

#### Tesla T4: prefill (tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 1024 | 8,760 (2.13) | 3,727 (2.41) | 5,466 (2.36) | 1.60x | 0.62x | 0.23 GiB (10%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 4096 | 8,639 (2.34) | 1,311 (6.9) | 2,223 (6.27) | 3.89x | 0.26x | 3.93 GiB (63%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | 7,747 (2.62) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 16384 | 6,135 (3.17) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 1024 | 3,523 (5.98) | 2,316 (6.09) | 2,987 (6.05) | 1.18x | 0.85x | 0.07 GiB (1%) |
| Qwen2.5-3B | fp16 | 1 | 4096 | 3,383 (6.32) | 1,079 (8.51) | 1,712 (8.13) | 1.98x | 0.51x | 1.81 GiB (22%) |
| Qwen2.5-3B | fp16 | 1 | 8192 | 3,229 (6.78) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 16384 | 2,524 (7.69) | OOM | OOM | only ours | - | - |
| Qwen3.5-2B | fp16 | 1 | 1024 | 4,224 (3.66) | 3,886 (3.67) | 4,192 (3.66) | 1.01x | 0.99x | 0.00 GiB (0%) |
| Qwen3.5-2B | fp16 | 1 | 4096 | 4,129 (4.04) | 3,158 (4.99) | 3,761 (4.72) | 1.10x | 0.91x | 0.68 GiB (14%) |
| Qwen3.5-2B | fp16 | 1 | 8192 | 3,965 (4.55) | 2,444 (8.81) | 3,392 (7.97) | 1.17x | 0.86x | 3.42 GiB (43%) |
| Qwen3.5-2B | fp16 | 1 | 16384 | 3,627 (5.57) | OOM | OOM | only ours | - | - |
| Qwen3.5-4B | fp16 | 1 | 1024 | 1,724 (8.26) | 1,579 (8.27) | 1,673 (8.26) | 1.03x | 0.97x | 0.00 GiB (0%) |
| Qwen3.5-4B | fp16 | 1 | 4096 | 1,685 (9.03) | 1,251 (10.82) | 1,520 (10.33) | 1.11x | 0.90x | 1.30 GiB (13%) |
| Qwen3.5-4B | fp16 | 1 | 8192 | 1,561 (10.08) | OOM | OOM | only ours | - | - |
| Qwen3.5-4B | fp16 | 1 | 16384 | 1,431 (12.17) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 32768 | 4,322 (4.29) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 65536 | 2,533 (6.51) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 131072 | 1,432 (10.97) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 32768 | 1,969 (9.52) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 65536 | OOM | - | - | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 131072 | OOM | - | - | - | - | - |

#### Tesla T4: train (tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 1024 | 2,713 (6.23) | 1,523 (9.25) | 1,871 (10.37) | 1.45x | 0.69x | 3.02 GiB (33%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 2048 | 2,897 (7.95) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 4096 | 2,739 (11.53) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 1024 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 2048 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 1024 | 919 (11.49) | 896 (11.76) | 918 (11.78) | 1.00x | 1.00x | 0.27 GiB (2%) |
| Qwen3.5-2B | fp16 | 1 | 2048 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-4B | fp16 | 1 | 1024 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-4B | fp16 | 1 | 2048 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-4B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-4B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 512 | 2,541 (7.85) | 2,373 (8.12) | 1,969 (12.2) | 1.07x | 0.93x | 0.27 GiB (3%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 1024 | 2,632 (11.35) | 2,302 (11.96) | OOM | 1.14x | 0.87x | 0.61 GiB (5%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 2048 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 4096 | OOM | OOM | OOM | - | - | - |

#### Tesla T4: train_ckpt (tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 2048 | 2,330 (5.56) | 762 (6.85) | 995 (6.58) | 2.34x | 0.43x | 1.02 GiB (16%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 4096 | 2,129 (6.0) | OOM | 661 (11.92) | 3.22x | 0.31x | 5.92 GiB (50%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | 1,720 (7.68) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 16384 | 1,256 (11.36) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 2048 | 975 (12.86) | 523 (12.86) | 702 (12.87) | 1.39x | 0.72x | 0.00 GiB (0%) |
| Qwen2.5-3B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 2048 | 529 (10.23) | 505 (10.23) | 526 (10.24) | 1.01x | 0.99x | 0.00 GiB (0%) |
| Qwen3.5-2B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 32768 | OOM | - | - | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 65536 | OOM | - | - | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 32768 | OOM | - | - | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 65536 | OOM | - | - | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 1024 | 2,038 (6.0) | 1,736 (6.0) | 1,223 (6.65) | 1.17x | 0.85x | 0.00 GiB (0%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 2048 | 1,987 (7.68) | 1,416 (7.7) | 763 (12.05) | 1.40x | 0.71x | 0.02 GiB (0%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 4096 | 1,798 (11.36) | 1,055 (11.42) | OOM | 1.70x | 0.59x | 0.06 GiB (1%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 8192 | OOM | OOM | OOM | - | - | - |

#### Tesla T4: generate (decode tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 2048 | 48.8 (2.2) | 42.6 (3.34) | 43.9 (3.16) | 1.11x | 0.90x | 0.96 GiB (30%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | 48.2 (2.62) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 8 | 2048 | 364.9 (3.17) | 77.4 (12.2) | OOM | 4.71x | 0.21x | 9.03 GiB (74%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 8 | 8192 | 211.1 (6.51) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 2048 | 28.4 (6.09) | 20.1 (6.61) | 25.5 (6.49) | 1.11x | 0.90x | 0.40 GiB (6%) |
| Qwen2.5-3B | fp16 | 1 | 8192 | 28.1 (6.78) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 8 | 2048 | 183.0 (7.69) | 43.1 (11.73) | 83.3 (10.87) | 2.20x | 0.46x | 3.18 GiB (29%) |
| Qwen2.5-3B | fp16 | 8 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 2048 | 24.2 (3.79) | 23.3 (3.96) | 23.5 (3.87) | 1.03x | 0.97x | 0.08 GiB (2%) |
| Qwen3.5-2B | fp16 | 1 | 8192 | 23.5 (4.55) | 22.3 (8.82) | 22.8 (7.97) | 1.03x | 0.97x | 3.42 GiB (43%) |
| Qwen3.5-2B | fp16 | 8 | 2048 | 178.2 (5.71) | 120.4 (6.99) | 170.9 (6.35) | 1.04x | 0.96x | 0.64 GiB (10%) |
| Qwen3.5-2B | fp16 | 8 | 8192 | 169.5 (11.83) | OOM | OOM | only ours | - | - |
| Qwen3.5-4B | fp16 | 1 | 2048 | 17.8 (8.51) | 17.1 (8.83) | 17.6 (8.65) | 1.01x | 0.99x | 0.14 GiB (2%) |
| Qwen3.5-4B | fp16 | 1 | 8192 | 17.2 (10.08) | OOM | OOM | only ours | - | - |
| Qwen3.5-4B | fp16 | 8 | 2048 | 102.1 (12.52) | OOM | OOM | only ours | - | - |
| Qwen3.5-4B | fp16 | 8 | 8192 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 16384 | 50.3 (3.17) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 32768 | 41.6 (4.29) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 65536 | 27.3 (6.51) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 16384 | 24.5 (7.69) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 32768 | 19.6 (9.52) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 65536 | OOM | - | - | - | - | - |

#### Tesla T4: generate_static_compiled (decode tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 2048 | 91.2 (2.21) | 60.4 (3.35) | 68.5 (3.23) | 1.33x | 0.75x | 1.02 GiB (32%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | 81.3 (2.62) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 8 | 2048 | 547.4 (3.19) | 171.2 (12.23) | OOM | 3.20x | 0.31x | 9.04 GiB (74%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 8 | 8192 | 397.7 (6.54) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 2048 | 34.4 (6.12) | 27.4 (6.63) | 29.4 (6.54) | 1.17x | 0.85x | 0.42 GiB (6%) |
| Qwen2.5-3B | fp16 | 1 | 8192 | 32.5 (6.78) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 8 | 2048 | 224.6 (7.73) | 106.1 (11.76) | 103.3 (11.17) | 2.12x | 0.47x | 3.44 GiB (31%) |
| Qwen2.5-3B | fp16 | 8 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 2048 | 53.8 (3.79) | 48.5 (3.96) | 50.9 (3.89) | 1.06x | 0.95x | 0.10 GiB (3%) |
| Qwen3.5-2B | fp16 | 1 | 8192 | 53.0 (4.57) | 36.0 (8.82) | 40.4 (8.04) | 1.31x | 0.76x | 3.47 GiB (43%) |
| Qwen3.5-2B | fp16 | 8 | 2048 | 325.9 (5.76) | 241.5 (7.01) | 249.9 (6.5) | 1.30x | 0.77x | 0.74 GiB (11%) |
| Qwen3.5-2B | fp16 | 8 | 8192 | 301.6 (11.98) | OOM | OOM | only ours | - | - |
| Qwen3.5-4B | fp16 | 1 | 2048 | 25.2 (8.52) | 23.1 (8.83) | 23.7 (8.68) | 1.06x | 0.94x | 0.16 GiB (2%) |
| Qwen3.5-4B | fp16 | 1 | 8192 | 24.6 (10.12) | OOM | OOM | only ours | - | - |
| Qwen3.5-4B | fp16 | 8 | 2048 | 152.9 (12.63) | OOM | OOM | only ours | - | - |
| Qwen3.5-4B | fp16 | 8 | 8192 | OOM | OOM | OOM | - | - | - |

#### Tesla T4: train_lora_ckpt (tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-3B (LoRA r16) | fp16 | 1 | 2048 | 994 (9.73) | 526 (9.73) | 706 (9.74) | 1.41x | 0.71x | 0.00 GiB (0%) |
| Qwen2.5-3B (LoRA r16) | fp16 | 1 | 4096 | 907 (13.5) | 337 (13.5) | OOM | 2.69x | 0.37x | 0.00 GiB (0%) |
| Qwen2.5-3B (LoRA r16) | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B (LoRA r16) | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B (LoRA r16) | fp16 | 1 | 32768 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B (LoRA r16) | fp16 | 4 | 2048 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B (LoRA r16) | fp16 | 4 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B (LoRA r16) | fp16 | 4 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B (LoRA r16) | fp16 | 4 | 16384 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B (LoRA r16) | fp16 | 4 | 32768 | OOM | OOM | OOM | - | - | - |

#### Tesla T4: longest N that fits (>= N: no longer N measured)

| model | mode | dtype | B | flash_attn_legacy | sdpa | eager |
|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | prefill | fp16 | 1 | >= 131,072 | 4,096 | 4,096 |
| Qwen2.5-3B | prefill | fp16 | 1 | 32,768 | 4,096 | 4,096 |
| Qwen3.5-2B | prefill | fp16 | 1 | >= 16,384 | 8,192 | 8,192 |
| Qwen3.5-4B | prefill | fp16 | 1 | >= 16,384 | 4,096 | 4,096 |
| TinyLlama-1.1B-Chat-v1.0 | train | fp16 | 1 | 4,096 | 1,024 | 1,024 |
| Qwen2.5-3B | train | fp16 | 1 | OOM | OOM | OOM |
| Qwen3.5-2B | train | fp16 | 1 | 1,024 | 1,024 | 1,024 |
| Qwen3.5-4B | train | fp16 | 1 | OOM | OOM | OOM |
| TinyLlama-1.1B-Chat-v1.0 | train_ckpt | fp16 | 1 | 16,384 | 2,048 | 4,096 |
| Qwen2.5-3B | train_ckpt | fp16 | 1 | 2,048 | 2,048 | 2,048 |
| Qwen3.5-2B | train_ckpt | fp16 | 1 | 2,048 | 2,048 | 2,048 |
| TinyLlama-1.1B-Chat-v1.0 | generate | fp16 | 1 | >= 65,536 | 2,048 | 2,048 |
| TinyLlama-1.1B-Chat-v1.0 | generate_static_compiled | fp16 | 1 | >= 8,192 | 2,048 | 2,048 |
| TinyLlama-1.1B-Chat-v1.0 | generate | fp16 | 8 | >= 8,192 | 2,048 | OOM |
| TinyLlama-1.1B-Chat-v1.0 | generate_static_compiled | fp16 | 8 | >= 8,192 | 2,048 | OOM |
| Qwen2.5-3B | generate | fp16 | 1 | 32,768 | 2,048 | 2,048 |
| Qwen2.5-3B | generate_static_compiled | fp16 | 1 | >= 8,192 | 2,048 | 2,048 |
| Qwen2.5-3B | generate | fp16 | 8 | 2,048 | 2,048 | 2,048 |
| Qwen2.5-3B | generate_static_compiled | fp16 | 8 | 2,048 | 2,048 | 2,048 |
| Qwen3.5-2B | generate | fp16 | 1 | >= 8,192 | >= 8,192 | >= 8,192 |
| Qwen3.5-2B | generate_static_compiled | fp16 | 1 | >= 8,192 | >= 8,192 | >= 8,192 |
| Qwen3.5-2B | generate | fp16 | 8 | >= 8,192 | 2,048 | 2,048 |
| Qwen3.5-2B | generate_static_compiled | fp16 | 8 | >= 8,192 | 2,048 | 2,048 |
| Qwen3.5-4B | generate | fp16 | 1 | >= 8,192 | 2,048 | 2,048 |
| Qwen3.5-4B | generate_static_compiled | fp16 | 1 | >= 8,192 | 2,048 | 2,048 |
| Qwen3.5-4B | generate | fp16 | 8 | 2,048 | OOM | OOM |
| Qwen3.5-4B | generate_static_compiled | fp16 | 8 | 2,048 | OOM | OOM |
| TinyLlama-1.1B-Chat-v1.0 | train | fp16 | 4 | 1,024 | 1,024 | 512 |
| TinyLlama-1.1B-Chat-v1.0 | train_ckpt | fp16 | 4 | 4,096 | 4,096 | 2,048 |
| Qwen2.5-3B (LoRA r16) | train_lora_ckpt | fp16 | 1 | 4,096 | 4,096 | 2,048 |
| Qwen2.5-3B (LoRA r16) | train_lora_ckpt | fp16 | 4 | OOM | OOM | OOM |

#### Tesla V100-SXM2-32GB: prefill (tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 1024 | 20,237 (2.13) | 10,231 (2.41) | 14,457 (2.36) | 1.40x | 0.71x | 0.23 GiB (10%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 4096 | 21,277 (2.34) | 4,067 (6.9) | 6,229 (6.27) | 3.42x | 0.29x | 3.93 GiB (63%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | 18,061 (2.62) | 1,852 (20.87) | 3,736 (18.55) | 4.83x | 0.21x | 15.93 GiB (86%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 16384 | 13,108 (3.17) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 1024 | 8,605 (5.98) | 6,299 (6.09) | 8,253 (6.05) | 1.04x | 0.96x | 0.07 GiB (1%) |
| Qwen2.5-3B | fp16 | 1 | 4096 | 9,109 (6.32) | 3,241 (8.51) | 5,231 (8.13) | 1.74x | 0.57x | 1.81 GiB (22%) |
| Qwen2.5-3B | fp16 | 1 | 8192 | 8,129 (6.78) | 1,712 (15.79) | 3,531 (14.46) | 2.30x | 0.43x | 7.68 GiB (53%) |
| Qwen2.5-3B | fp16 | 1 | 16384 | 6,243 (7.69) | OOM | OOM | only ours | - | - |
| gemma-2-2b-it | fp16 | 1 | 1024 | 10,393 (5.05) | 8,760 (5.11) | 10,508 (5.08) | 0.99x | 1.01x | 0.03 GiB (1%) |
| gemma-2-2b-it | fp16 | 1 | 4096 | 10,910 (5.57) | 6,573 (6.72) | 7,789 (6.45) | 1.40x | 0.71x | 0.88 GiB (14%) |
| gemma-2-2b-it | fp16 | 1 | 8192 | 10,248 (6.26) | 4,202 (10.96) | 5,571 (10.15) | 1.84x | 0.54x | 3.89 GiB (38%) |
| gemma-2-2b-it | fp16 | 1 | 16384 | 8,913 (7.64) | 2,139 (26.67) | 3,249 (23.92) | 2.74x | 0.36x | 16.28 GiB (68%) |
| Qwen3.5-2B | fp16 | 1 | 1024 | 9,093 (3.66) | 9,038 (3.67) | 9,817 (3.66) | 0.93x | 1.08x | 0.00 GiB (0%) |
| Qwen3.5-2B | fp16 | 1 | 4096 | 11,155 (4.04) | 8,916 (4.99) | 10,410 (4.72) | 1.07x | 0.93x | 0.68 GiB (14%) |
| Qwen3.5-2B | fp16 | 1 | 8192 | 11,049 (4.55) | 7,012 (8.81) | 9,639 (7.97) | 1.15x | 0.87x | 3.42 GiB (43%) |
| Qwen3.5-2B | fp16 | 1 | 16384 | 9,697 (5.57) | 4,507 (23.6) | 7,455 (20.66) | 1.30x | 0.77x | 15.09 GiB (73%) |
| Qwen3.5-4B | fp16 | 1 | 1024 | 4,616 (8.26) | 4,339 (8.27) | 4,607 (8.26) | 1.00x | 1.00x | 0.00 GiB (0%) |
| Qwen3.5-4B | fp16 | 1 | 4096 | 4,821 (9.03) | 3,710 (10.83) | 4,463 (10.33) | 1.08x | 0.93x | 1.30 GiB (13%) |
| Qwen3.5-4B | fp16 | 1 | 8192 | 4,617 (10.08) | 2,730 (18.29) | 3,975 (16.73) | 1.16x | 0.86x | 6.65 GiB (40%) |
| Qwen3.5-4B | fp16 | 1 | 16384 | 4,241 (12.17) | OOM | OOM | only ours | - | - |
| Mistral-7B-Instruct-v0.2 | fp16 | 1 | 1024 | 4,453 (13.74) | 3,249 (14.01) | 4,043 (13.92) | 1.10x | 0.91x | 0.18 GiB (1%) |
| Mistral-7B-Instruct-v0.2 | fp16 | 1 | 4096 | 4,272 (14.45) | 1,713 (18.97) | 2,582 (18.22) | 1.65x | 0.60x | 3.77 GiB (21%) |
| Mistral-7B-Instruct-v0.2 | fp16 | 1 | 8192 | 3,783 (15.41) | OOM | OOM | only ours | - | - |
| Mistral-7B-Instruct-v0.2 | fp16 | 1 | 16384 | 3,027 (17.32) | OOM | OOM | only ours | - | - |
| Llama-3.1-8B-Instruct | fp16 | 1 | 1024 | 4,482 (15.2) | 3,231 (15.48) | 4,028 (15.39) | 1.11x | 0.90x | 0.19 GiB (1%) |
| Llama-3.1-8B-Instruct | fp16 | 1 | 4096 | 4,288 (15.92) | 1,713 (20.44) | 2,589 (19.69) | 1.66x | 0.60x | 3.77 GiB (19%) |
| Llama-3.1-8B-Instruct | fp16 | 1 | 8192 | 3,803 (16.88) | OOM | OOM | only ours | - | - |
| Llama-3.1-8B-Instruct | fp16 | 1 | 16384 | 3,018 (18.79) | OOM | OOM | only ours | - | - |
| Qwen3.5-9B | fp16 | 1 | 512 | 2,986 (16.87) | 2,910 (16.87) | 2,967 (16.87) | 1.01x | 0.99x | 0.00 GiB (0%) |
| Qwen3.5-9B | fp16 | 1 | 1024 | 3,312 (17.01) | 3,154 (17.02) | 3,276 (17.01) | 1.01x | 0.99x | 0.00 GiB (0%) |
| Qwen3.5-9B | fp16 | 1 | 2048 | 3,385 (17.28) | 3,057 (17.59) | 3,249 (17.41) | 1.04x | 0.96x | 0.13 GiB (1%) |
| Qwen3.5-9B | fp16 | 1 | 4096 | 3,405 (17.82) | 2,794 (19.61) | 3,190 (19.11) | 1.07x | 0.94x | 1.29 GiB (7%) |
| Qwen3.5-9B | fp16 | 1 | 8192 | 3,286 (18.9) | 2,205 (27.11) | 2,940 (25.55) | 1.12x | 0.89x | 6.65 GiB (26%) |
| Qwen3.5-9B | fp16 | 1 | 16384 | 3,097 (21.06) | OOM | OOM | only ours | - | - |
| Qwen3.5-9B | fp16 | 1 | 32768 | 2,777 (25.39) | OOM | OOM | only ours | - | - |
| Qwen2.5-14B | fp16 | 1 | 512 | 2,319 (27.82) | 1,976 (27.91) | 2,266 (27.86) | 1.02x | 0.98x | 0.04 GiB (0%) |
| Qwen2.5-14B | fp16 | 1 | 1024 | 2,466 (27.97) | 1,754 (28.33) | 2,200 (28.22) | 1.12x | 0.89x | 0.25 GiB (1%) |
| Qwen2.5-14B | fp16 | 1 | 2048 | 2,470 (28.27) | 1,351 (29.71) | 1,733 (29.41) | 1.43x | 0.70x | 1.14 GiB (4%) |
| Qwen2.5-14B | fp16 | 1 | 4096 | 2,295 (28.89) | OOM | OOM | only ours | - | - |
| Qwen2.5-14B | fp16 | 1 | 8192 | 2,014 (30.11) | OOM | OOM | only ours | - | - |
| Qwen2.5-14B | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-14B | fp16 | 1 | 32768 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 32768 | 8,416 (4.29) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 65536 | 4,860 (6.51) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 131072 | 2,662 (10.97) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 32768 | 4,281 (9.52) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 65536 | 2,745 (13.18) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 131072 | 1,615 (20.49) | - | - | only ours | - | - |

#### Tesla V100-SXM2-32GB: train (tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 1024 | 6,804 (6.23) | 4,176 (9.25) | 5,079 (10.37) | 1.34x | 0.75x | 3.02 GiB (33%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 2048 | 7,209 (7.95) | 3,124 (20.27) | 3,891 (24.76) | 1.85x | 0.54x | 12.32 GiB (61%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 4096 | 6,529 (11.53) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | 5,163 (18.68) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 1024 | 3,162 (14.84) | 2,400 (17.62) | 2,895 (18.32) | 1.09x | 0.92x | 2.78 GiB (16%) |
| Qwen2.5-3B | fp16 | 1 | 2048 | 3,327 (20.05) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| gemma-2-2b-it | fp16 | 1 | 1024 | 3,061 (15.22) | 2,705 (16.37) | 3,089 (16.85) | 0.99x | 1.01x | 1.15 GiB (7%) |
| gemma-2-2b-it | fp16 | 1 | 2048 | 3,146 (21.45) | 2,499 (25.31) | 3,055 (27.96) | 1.03x | 0.97x | 3.86 GiB (15%) |
| gemma-2-2b-it | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| gemma-2-2b-it | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 1024 | 1,758 (11.49) | 1,803 (11.76) | 1,391 (11.78) | 0.98x | 1.03x | 0.27 GiB (2%) |
| Qwen3.5-2B | fp16 | 1 | 2048 | 1,729 (18.51) | 1,708 (19.43) | 1,737 (19.66) | 1.00x | 1.00x | 0.92 GiB (5%) |
| Qwen3.5-2B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-4B | fp16 | 1 | 1024 | 1,021 (24.41) | 1,004 (25.12) | 947 (25.19) | 1.02x | 0.98x | 0.71 GiB (3%) |
| Qwen3.5-4B | fp16 | 1 | 2048 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-4B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-4B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Mistral-7B-Instruct-v0.2 | fp16 | 1 | 1024 | OOM | OOM | OOM | - | - | - |
| Mistral-7B-Instruct-v0.2 | fp16 | 1 | 2048 | OOM | OOM | OOM | - | - | - |
| Mistral-7B-Instruct-v0.2 | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Mistral-7B-Instruct-v0.2 | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 512 | 6,552 (7.85) | 6,183 (8.12) | 5,423 (12.2) | 1.06x | 0.94x | 0.27 GiB (3%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 1024 | 6,998 (11.35) | 6,002 (11.96) | 4,682 (28.27) | 1.17x | 0.86x | 0.61 GiB (5%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 2048 | 7,016 (18.3) | 5,085 (19.88) | OOM | 1.38x | 0.72x | 1.58 GiB (8%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 4096 | OOM | OOM | OOM | - | - | - |

#### Tesla V100-SXM2-32GB: train_ckpt (tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 2048 | 5,683 (5.56) | 2,172 (6.85) | 2,758 (6.58) | 2.06x | 0.49x | 1.02 GiB (16%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 4096 | 5,175 (6.0) | 1,327 (12.94) | 1,903 (11.92) | 2.72x | 0.37x | 5.92 GiB (50%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | 4,088 (7.68) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 16384 | 2,692 (11.36) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 2048 | 2,612 (12.86) | 1,485 (12.86) | 1,962 (12.87) | 1.33x | 0.75x | 0.00 GiB (0%) |
| Qwen2.5-3B | fp16 | 1 | 4096 | 2,402 (16.57) | 993 (16.57) | 1,549 (16.61) | 1.55x | 0.64x | 0.00 GiB (0%) |
| Qwen2.5-3B | fp16 | 1 | 8192 | 1,970 (24.12) | 544 (26.63) | OOM | 3.62x | 0.28x | 2.51 GiB (9%) |
| Qwen2.5-3B | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |
| gemma-2-2b-it | fp16 | 1 | 2048 | 2,475 (15.4) | 1,854 (15.4) | 2,294 (15.42) | 1.08x | 0.93x | 0.00 GiB (0%) |
| gemma-2-2b-it | fp16 | 1 | 4096 | 2,322 (22.48) | 1,604 (22.5) | 2,033 (22.55) | 1.14x | 0.88x | 0.02 GiB (0%) |
| gemma-2-2b-it | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| gemma-2-2b-it | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 2048 | 1,119 (10.23) | 1,198 (10.23) | 1,186 (10.24) | 0.93x | 1.07x | 0.00 GiB (0%) |
| Qwen3.5-2B | fp16 | 1 | 4096 | 889 (16.12) | 891 (16.12) | 918 (16.15) | 0.97x | 1.03x | 0.00 GiB (0%) |
| Qwen3.5-2B | fp16 | 1 | 8192 | 594 (27.89) | 563 (27.89) | OOM | 1.06x | 0.95x | 0.00 GiB (0%) |
| Qwen3.5-2B | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-4B | fp16 | 1 | 2048 | 610 (16.7) | 571 (16.69) | 619 (16.7) | 0.99x | 1.01x | -0.01 GiB (-0%) |
| Qwen3.5-4B | fp16 | 1 | 4096 | 426 (22.71) | 389 (22.71) | 424 (22.74) | 1.00x | 1.00x | 0.00 GiB (0%) |
| Qwen3.5-4B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-4B | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 1024 | 5,445 (6.0) | 4,677 (6.0) | 3,501 (6.65) | 1.16x | 0.86x | 0.00 GiB (0%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 2048 | 5,449 (7.68) | 3,950 (7.7) | 2,206 (12.05) | 1.38x | 0.72x | 0.02 GiB (0%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 4096 | 4,837 (11.36) | 2,844 (11.42) | OOM | 1.70x | 0.59x | 0.06 GiB (1%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 8192 | 3,874 (18.72) | 1,801 (18.97) | OOM | 2.15x | 0.46x | 0.25 GiB (1%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 32768 | 1,614 (18.73) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 65536 | OOM | - | - | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 32768 | OOM | - | - | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 65536 | OOM | - | - | - | - | - |

#### Tesla V100-SXM2-32GB: generate (decode tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 2048 | 52.0 (2.2) | 47.5 (3.34) | 49.7 (3.16) | 1.05x | 0.96x | 0.96 GiB (30%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | 52.1 (2.62) | 43.7 (20.87) | 48.9 (18.55) | 1.07x | 0.94x | 15.93 GiB (86%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 8 | 2048 | 390.9 (3.17) | 216.7 (12.2) | 365.4 (10.84) | 1.07x | 0.93x | 7.67 GiB (71%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 8 | 8192 | 396.3 (6.51) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 2048 | 32.8 (6.09) | 28.6 (6.61) | 29.8 (6.49) | 1.10x | 0.91x | 0.40 GiB (6%) |
| Qwen2.5-3B | fp16 | 1 | 8192 | 32.8 (6.78) | 22.9 (15.79) | 30.0 (14.46) | 1.09x | 0.91x | 7.68 GiB (53%) |
| Qwen2.5-3B | fp16 | 8 | 2048 | 248.3 (7.69) | 125.8 (11.73) | 211.4 (10.87) | 1.17x | 0.85x | 3.18 GiB (29%) |
| Qwen2.5-3B | fp16 | 8 | 8192 | 244.7 (13.18) | OOM | OOM | only ours | - | - |
| gemma-2-2b-it | fp16 | 1 | 2048 | 32.4 (5.23) | 30.2 (5.5) | 21.5 (5.4) | 1.07x | 0.93x | 0.17 GiB (3%) |
| gemma-2-2b-it | fp16 | 1 | 8192 | 31.9 (6.26) | 27.0 (10.96) | 21.7 (10.15) | 1.18x | 0.85x | 3.89 GiB (38%) |
| gemma-2-2b-it | fp16 | 8 | 2048 | 250.2 (7.64) | 123.4 (9.68) | 171.0 (9.04) | 1.46x | 0.68x | 1.40 GiB (15%) |
| gemma-2-2b-it | fp16 | 8 | 8192 | 162.2 (15.94) | OOM | OOM | only ours | - | - |
| Qwen3.5-2B | fp16 | 1 | 2048 | 27.1 (3.79) | 26.7 (3.96) | 27.5 (3.87) | 0.99x | 1.01x | 0.08 GiB (2%) |
| Qwen3.5-2B | fp16 | 1 | 8192 | 27.2 (4.55) | 26.2 (8.82) | 26.8 (7.97) | 1.01x | 0.99x | 3.42 GiB (43%) |
| Qwen3.5-2B | fp16 | 8 | 2048 | 207.8 (5.71) | 194.7 (6.99) | 198.3 (6.35) | 1.05x | 0.95x | 0.64 GiB (10%) |
| Qwen3.5-2B | fp16 | 8 | 8192 | 206.6 (11.83) | OOM | OOM | only ours | - | - |
| Qwen3.5-4B | fp16 | 1 | 2048 | 21.1 (8.51) | 19.8 (8.83) | 20.7 (8.65) | 1.02x | 0.98x | 0.14 GiB (2%) |
| Qwen3.5-4B | fp16 | 1 | 8192 | 20.7 (10.08) | 18.8 (18.29) | 20.2 (16.73) | 1.02x | 0.98x | 6.65 GiB (40%) |
| Qwen3.5-4B | fp16 | 8 | 2048 | 153.6 (12.52) | 129.8 (14.95) | 151.5 (13.62) | 1.01x | 0.99x | 1.10 GiB (8%) |
| Qwen3.5-4B | fp16 | 8 | 8192 | 152.8 (25.09) | OOM | OOM | only ours | - | - |
| Mistral-7B-Instruct-v0.2 | fp16 | 1 | 2048 | 34.1 (13.97) | 24.7 (15.09) | 29.6 (14.85) | 1.15x | 0.87x | 0.88 GiB (6%) |
| Mistral-7B-Instruct-v0.2 | fp16 | 1 | 8192 | 29.3 (15.41) | OOM | OOM | only ours | - | - |
| Mistral-7B-Instruct-v0.2 | fp16 | 8 | 2048 | 184.3 (17.32) | 64.7 (26.15) | 107.5 (24.32) | 1.71x | 0.58x | 7.00 GiB (29%) |
| Mistral-7B-Instruct-v0.2 | fp16 | 8 | 8192 | OOM | OOM | OOM | - | - | - |
| Llama-3.1-8B-Instruct | fp16 | 1 | 2048 | 32.9 (15.44) | 24.1 (16.56) | 28.5 (16.32) | 1.15x | 0.87x | 0.88 GiB (5%) |
| Llama-3.1-8B-Instruct | fp16 | 1 | 8192 | 28.5 (16.88) | OOM | OOM | only ours | - | - |
| Llama-3.1-8B-Instruct | fp16 | 8 | 2048 | 179.4 (18.79) | 64.3 (27.61) | 105.9 (25.79) | 1.69x | 0.59x | 7.00 GiB (27%) |
| Llama-3.1-8B-Instruct | fp16 | 8 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-9B | fp16 | 1 | 512 | 19.4 (16.87) | 19.2 (16.87) | 19.7 (16.87) | 0.98x | 1.02x | 0.00 GiB (0%) |
| Qwen3.5-9B | fp16 | 1 | 2048 | 18.9 (17.28) | 18.3 (17.59) | 19.0 (17.41) | 0.99x | 1.01x | 0.13 GiB (1%) |
| Qwen3.5-9B | fp16 | 1 | 8192 | 19.0 (18.9) | 18.4 (27.11) | 18.7 (25.55) | 1.02x | 0.98x | 6.65 GiB (26%) |
| Qwen3.5-9B | fp16 | 8 | 512 | 147.7 (18.17) | 144.6 (18.17) | 147.2 (18.17) | 1.00x | 1.00x | 0.00 GiB (0%) |
| Qwen3.5-9B | fp16 | 8 | 2048 | 146.8 (21.42) | 110.1 (23.84) | 135.6 (22.51) | 1.08x | 0.92x | 1.09 GiB (5%) |
| Qwen3.5-9B | fp16 | 8 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-14B | fp16 | 1 | 512 | 19.6 (27.82) | 17.5 (27.91) | 18.7 (27.86) | 1.05x | 0.95x | 0.04 GiB (0%) |
| Qwen2.5-14B | fp16 | 1 | 2048 | 19.1 (28.27) | 13.9 (29.71) | 16.6 (29.41) | 1.15x | 0.87x | 1.14 GiB (4%) |
| Qwen2.5-14B | fp16 | 1 | 8192 | 16.6 (30.11) | OOM | OOM | only ours | - | - |
| Qwen2.5-14B | fp16 | 8 | 512 | 128.9 (28.89) | 77.9 (29.62) | 103.3 (29.28) | 1.25x | 0.80x | 0.39 GiB (1%) |
| Qwen2.5-14B | fp16 | 8 | 2048 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-14B | fp16 | 8 | 8192 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 16384 | 52.7 (3.17) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 32768 | 54.5 (4.29) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 65536 | 54.3 (6.51) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 16384 | 33.0 (7.69) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 32768 | 33.0 (9.52) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 65536 | 32.3 (13.18) | - | - | only ours | - | - |

#### Tesla V100-SXM2-32GB: generate_static_compiled (decode tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 2048 | 165.3 (2.21) | 86.8 (3.35) | 139.6 (3.23) | 1.18x | 0.84x | 1.02 GiB (32%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | 150.5 (2.62) | 36.6 (20.88) | 93.1 (18.8) | 1.62x | 0.62x | 16.18 GiB (86%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 8 | 2048 | 973.5 (3.19) | 338.5 (12.23) | 495.9 (11.38) | 1.96x | 0.51x | 8.19 GiB (72%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 8 | 8192 | 808.2 (6.54) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 2048 | 72.6 (6.12) | 51.0 (6.63) | 66.4 (6.54) | 1.09x | 0.91x | 0.42 GiB (6%) |
| Qwen2.5-3B | fp16 | 1 | 8192 | 70.8 (6.78) | 25.1 (15.79) | 46.4 (14.59) | 1.53x | 0.66x | 7.81 GiB (54%) |
| Qwen2.5-3B | fp16 | 8 | 2048 | 471.7 (7.73) | 263.1 (11.76) | 273.0 (11.17) | 1.73x | 0.58x | 3.44 GiB (31%) |
| Qwen2.5-3B | fp16 | 8 | 8192 | 400.5 (13.21) | OOM | OOM | only ours | - | - |
| gemma-2-2b-it | fp16 | 1 | 2048 | 80.5 (5.25) | 50.2 (5.51) | 74.4 (5.44) | 1.08x | 0.92x | 0.19 GiB (3%) |
| gemma-2-2b-it | fp16 | 1 | 8192 | 70.5 (6.06) | 27.1 (10.76) | 54.3 (10.02) | 1.30x | 0.77x | 3.96 GiB (40%) |
| gemma-2-2b-it | fp16 | 8 | 2048 | 504.0 (7.75) | 286.9 (9.79) | 313.8 (9.29) | 1.61x | 0.62x | 1.54 GiB (17%) |
| gemma-2-2b-it | fp16 | 8 | 8192 | 279.1 (14.37) | OOM | OOM | only ours | - | - |
| Qwen3.5-2B | fp16 | 1 | 2048 | 104.0 (3.79) | 86.1 (3.96) | 100.1 (3.89) | 1.04x | 0.96x | 0.10 GiB (3%) |
| Qwen3.5-2B | fp16 | 1 | 8192 | 101.6 (4.57) | 57.7 (8.82) | 87.5 (8.04) | 1.16x | 0.86x | 3.47 GiB (43%) |
| Qwen3.5-2B | fp16 | 8 | 2048 | 591.8 (5.76) | 491.1 (7.01) | 509.9 (6.5) | 1.16x | 0.86x | 0.74 GiB (11%) |
| Qwen3.5-2B | fp16 | 8 | 8192 | 560.8 (11.98) | OOM | OOM | only ours | - | - |
| Qwen3.5-4B | fp16 | 1 | 2048 | 55.7 (8.52) | 48.5 (8.83) | 53.5 (8.68) | 1.04x | 0.96x | 0.16 GiB (2%) |
| Qwen3.5-4B | fp16 | 1 | 8192 | 54.1 (10.12) | 34.3 (18.29) | 45.4 (16.86) | 1.19x | 0.84x | 6.74 GiB (40%) |
| Qwen3.5-4B | fp16 | 8 | 2048 | 312.5 (12.63) | 252.2 (14.98) | 255.6 (13.92) | 1.22x | 0.82x | 1.29 GiB (9%) |
| Qwen3.5-4B | fp16 | 8 | 8192 | 288.9 (25.38) | OOM | OOM | only ours | - | - |
| Mistral-7B-Instruct-v0.2 | fp16 | 1 | 2048 | 37.8 (13.99) | 30.2 (15.11) | 33.4 (14.93) | 1.13x | 0.88x | 0.94 GiB (6%) |
| Mistral-7B-Instruct-v0.2 | fp16 | 1 | 8192 | 35.5 (15.42) | OOM | OOM | only ours | - | - |
| Mistral-7B-Instruct-v0.2 | fp16 | 8 | 2048 | 250.5 (17.44) | 131.2 (26.27) | 144.6 (24.96) | 1.73x | 0.58x | 7.52 GiB (30%) |
| Mistral-7B-Instruct-v0.2 | fp16 | 8 | 8192 | OOM | OOM | OOM | - | - | - |
| Llama-3.1-8B-Instruct | fp16 | 1 | 2048 | 36.4 (15.46) | 29.3 (16.58) | 32.2 (16.4) | 1.13x | 0.88x | 0.94 GiB (6%) |
| Llama-3.1-8B-Instruct | fp16 | 1 | 8192 | 34.3 (16.89) | OOM | OOM | only ours | - | - |
| Llama-3.1-8B-Instruct | fp16 | 8 | 2048 | 241.3 (18.91) | 128.4 (27.74) | 141.6 (26.43) | 1.70x | 0.59x | 7.52 GiB (28%) |
| Llama-3.1-8B-Instruct | fp16 | 8 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-9B | fp16 | 1 | 512 | 35.7 (16.88) | 34.8 (16.88) | 35.6 (16.88) | 1.00x | 1.00x | 0.00 GiB (0%) |
| Qwen3.5-9B | fp16 | 1 | 2048 | 35.5 (17.29) | 32.6 (17.6) | 34.7 (17.45) | 1.02x | 0.98x | 0.16 GiB (1%) |
| Qwen3.5-9B | fp16 | 1 | 8192 | 35.0 (18.94) | 25.6 (27.12) | 31.1 (25.68) | 1.13x | 0.89x | 6.74 GiB (26%) |
| Qwen3.5-9B | fp16 | 8 | 512 | 212.2 (18.23) | 202.8 (18.23) | 202.5 (18.24) | 1.05x | 0.96x | 0.00 GiB (0%) |
| Qwen3.5-9B | fp16 | 8 | 2048 | 208.2 (21.53) | 181.2 (23.87) | 182.6 (22.82) | 1.14x | 0.88x | 1.29 GiB (6%) |
| Qwen3.5-9B | fp16 | 8 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-14B | fp16 | 1 | 512 | 20.9 (27.84) | 19.9 (27.93) | 20.4 (27.91) | 1.02x | 0.98x | 0.07 GiB (0%) |
| Qwen2.5-14B | fp16 | 1 | 2048 | 20.7 (28.3) | 17.1 (29.73) | 18.4 (29.52) | 1.12x | 0.89x | 1.22 GiB (4%) |
| Qwen2.5-14B | fp16 | 1 | 8192 | 19.6 (30.14) | OOM | OOM | only ours | - | - |
| Qwen2.5-14B | fp16 | 8 | 512 | 144.9 (29.07) | 115.0 (29.81) | 117.7 (29.64) | 1.23x | 0.81x | 0.57 GiB (2%) |
| Qwen2.5-14B | fp16 | 8 | 2048 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-14B | fp16 | 8 | 8192 | OOM | OOM | OOM | - | - | - |

#### Tesla V100-SXM2-32GB: train_lora_ckpt (tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-3B (LoRA r16) | fp16 | 1 | 2048 | 2,603 (9.73) | 1,476 (9.73) | 1,945 (9.74) | 1.34x | 0.75x | 0.00 GiB (0%) |
| Qwen2.5-3B (LoRA r16) | fp16 | 1 | 4096 | 2,341 (13.5) | 985 (13.5) | 1,528 (13.53) | 1.53x | 0.65x | 0.00 GiB (0%) |
| Qwen2.5-3B (LoRA r16) | fp16 | 1 | 8192 | 1,899 (21.05) | 527 (23.46) | 1,038 (21.49) | 1.83x | 0.55x | 0.44 GiB (2%) |
| Qwen2.5-3B (LoRA r16) | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B (LoRA r16) | fp16 | 1 | 32768 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B (LoRA r16) | fp16 | 4 | 2048 | 2,212 (21.05) | 1,850 (21.07) | 1,532 (21.08) | 1.20x | 0.84x | 0.02 GiB (0%) |
| Qwen2.5-3B (LoRA r16) | fp16 | 4 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B (LoRA r16) | fp16 | 4 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B (LoRA r16) | fp16 | 4 | 16384 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B (LoRA r16) | fp16 | 4 | 32768 | OOM | OOM | OOM | - | - | - |
| Llama-3.1-8B-Instruct (LoRA r16) | fp16 | 1 | 2048 | 1,282 (18.59) | 770 (18.59) | 993 (18.6) | 1.29x | 0.77x | 0.00 GiB (0%) |
| Llama-3.1-8B-Instruct (LoRA r16) | fp16 | 1 | 4096 | 1,169 (22.06) | 530 (24.5) | 790 (23.44) | 1.48x | 0.68x | 1.38 GiB (6%) |
| Llama-3.1-8B-Instruct (LoRA r16) | fp16 | 1 | 8192 | 974 (29.0) | OOM | OOM | only ours | - | - |
| Llama-3.1-8B-Instruct (LoRA r16) | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |
| Llama-3.1-8B-Instruct (LoRA r16) | fp16 | 1 | 32768 | OOM | OOM | OOM | - | - | - |
| Llama-3.1-8B-Instruct (LoRA r16) | fp16 | 4 | 2048 | 1,073 (28.99) | 904 (29.01) | OOM | 1.19x | 0.84x | 0.02 GiB (0%) |
| Llama-3.1-8B-Instruct (LoRA r16) | fp16 | 4 | 4096 | OOM | OOM | OOM | - | - | - |
| Llama-3.1-8B-Instruct (LoRA r16) | fp16 | 4 | 8192 | OOM | OOM | OOM | - | - | - |
| Llama-3.1-8B-Instruct (LoRA r16) | fp16 | 4 | 16384 | OOM | OOM | OOM | - | - | - |
| Llama-3.1-8B-Instruct (LoRA r16) | fp16 | 4 | 32768 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-14B (LoRA r16) | fp16 | 1 | 2048 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-14B (LoRA r16) | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-14B (LoRA r16) | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-14B (LoRA r16) | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-14B (LoRA r16) | fp16 | 1 | 32768 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-14B (LoRA r16) | fp16 | 4 | 2048 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-14B (LoRA r16) | fp16 | 4 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-14B (LoRA r16) | fp16 | 4 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-14B (LoRA r16) | fp16 | 4 | 16384 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-14B (LoRA r16) | fp16 | 4 | 32768 | OOM | OOM | OOM | - | - | - |

#### Tesla V100-SXM2-32GB: longest N that fits (>= N: no longer N measured)

| model | mode | dtype | B | flash_attn_legacy | sdpa | eager |
|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | prefill | fp16 | 1 | >= 131,072 | 8,192 | 8,192 |
| Qwen2.5-3B | prefill | fp16 | 1 | >= 131,072 | 8,192 | 8,192 |
| gemma-2-2b-it | prefill | fp16 | 1 | >= 16,384 | >= 16,384 | >= 16,384 |
| Qwen3.5-2B | prefill | fp16 | 1 | >= 16,384 | >= 16,384 | >= 16,384 |
| Qwen3.5-4B | prefill | fp16 | 1 | >= 16,384 | 8,192 | 8,192 |
| Mistral-7B-Instruct-v0.2 | prefill | fp16 | 1 | >= 16,384 | 4,096 | 4,096 |
| TinyLlama-1.1B-Chat-v1.0 | train | fp16 | 1 | >= 8,192 | 2,048 | 2,048 |
| Qwen2.5-3B | train | fp16 | 1 | 2,048 | 1,024 | 1,024 |
| gemma-2-2b-it | train | fp16 | 1 | 2,048 | 2,048 | 2,048 |
| Qwen3.5-2B | train | fp16 | 1 | 2,048 | 2,048 | 2,048 |
| Qwen3.5-4B | train | fp16 | 1 | 1,024 | 1,024 | 1,024 |
| Mistral-7B-Instruct-v0.2 | train | fp16 | 1 | OOM | OOM | OOM |
| TinyLlama-1.1B-Chat-v1.0 | train_ckpt | fp16 | 1 | 32,768 | 4,096 | 4,096 |
| Qwen2.5-3B | train_ckpt | fp16 | 1 | 8,192 | 8,192 | 4,096 |
| gemma-2-2b-it | train_ckpt | fp16 | 1 | 4,096 | 4,096 | 4,096 |
| Qwen3.5-2B | train_ckpt | fp16 | 1 | 8,192 | 8,192 | 4,096 |
| Qwen3.5-4B | train_ckpt | fp16 | 1 | 4,096 | 4,096 | 4,096 |
| TinyLlama-1.1B-Chat-v1.0 | generate | fp16 | 1 | >= 65,536 | >= 8,192 | >= 8,192 |
| TinyLlama-1.1B-Chat-v1.0 | generate_static_compiled | fp16 | 1 | >= 8,192 | >= 8,192 | >= 8,192 |
| TinyLlama-1.1B-Chat-v1.0 | generate | fp16 | 8 | >= 8,192 | 2,048 | 2,048 |
| TinyLlama-1.1B-Chat-v1.0 | generate_static_compiled | fp16 | 8 | >= 8,192 | 2,048 | 2,048 |
| Qwen2.5-3B | generate | fp16 | 1 | >= 65,536 | >= 8,192 | >= 8,192 |
| Qwen2.5-3B | generate_static_compiled | fp16 | 1 | >= 8,192 | >= 8,192 | >= 8,192 |
| Qwen2.5-3B | generate | fp16 | 8 | >= 8,192 | 2,048 | 2,048 |
| Qwen2.5-3B | generate_static_compiled | fp16 | 8 | >= 8,192 | 2,048 | 2,048 |
| gemma-2-2b-it | generate | fp16 | 1 | >= 8,192 | >= 8,192 | >= 8,192 |
| gemma-2-2b-it | generate_static_compiled | fp16 | 1 | >= 8,192 | >= 8,192 | >= 8,192 |
| gemma-2-2b-it | generate | fp16 | 8 | >= 8,192 | 2,048 | 2,048 |
| gemma-2-2b-it | generate_static_compiled | fp16 | 8 | >= 8,192 | 2,048 | 2,048 |
| Qwen3.5-2B | generate | fp16 | 1 | >= 8,192 | >= 8,192 | >= 8,192 |
| Qwen3.5-2B | generate_static_compiled | fp16 | 1 | >= 8,192 | >= 8,192 | >= 8,192 |
| Qwen3.5-2B | generate | fp16 | 8 | >= 8,192 | 2,048 | 2,048 |
| Qwen3.5-2B | generate_static_compiled | fp16 | 8 | >= 8,192 | 2,048 | 2,048 |
| Qwen3.5-4B | generate | fp16 | 1 | >= 8,192 | >= 8,192 | >= 8,192 |
| Qwen3.5-4B | generate_static_compiled | fp16 | 1 | >= 8,192 | >= 8,192 | >= 8,192 |
| Qwen3.5-4B | generate | fp16 | 8 | >= 8,192 | 2,048 | 2,048 |
| Qwen3.5-4B | generate_static_compiled | fp16 | 8 | >= 8,192 | 2,048 | 2,048 |
| Mistral-7B-Instruct-v0.2 | generate | fp16 | 1 | >= 8,192 | 2,048 | 2,048 |
| Mistral-7B-Instruct-v0.2 | generate_static_compiled | fp16 | 1 | >= 8,192 | 2,048 | 2,048 |
| Mistral-7B-Instruct-v0.2 | generate | fp16 | 8 | 2,048 | 2,048 | 2,048 |
| Mistral-7B-Instruct-v0.2 | generate_static_compiled | fp16 | 8 | 2,048 | 2,048 | 2,048 |
| Llama-3.1-8B-Instruct | prefill | fp16 | 1 | >= 16,384 | 4,096 | 4,096 |
| Llama-3.1-8B-Instruct | generate | fp16 | 1 | >= 8,192 | 2,048 | 2,048 |
| Llama-3.1-8B-Instruct | generate_static_compiled | fp16 | 1 | >= 8,192 | 2,048 | 2,048 |
| Llama-3.1-8B-Instruct | generate | fp16 | 8 | 2,048 | 2,048 | 2,048 |
| Llama-3.1-8B-Instruct | generate_static_compiled | fp16 | 8 | 2,048 | 2,048 | 2,048 |
| Qwen3.5-9B | prefill | fp16 | 1 | >= 32,768 | 8,192 | 8,192 |
| Qwen3.5-9B | generate | fp16 | 1 | >= 8,192 | >= 8,192 | >= 8,192 |
| Qwen3.5-9B | generate_static_compiled | fp16 | 1 | >= 8,192 | >= 8,192 | >= 8,192 |
| Qwen3.5-9B | generate | fp16 | 8 | 2,048 | 2,048 | 2,048 |
| Qwen3.5-9B | generate_static_compiled | fp16 | 8 | 2,048 | 2,048 | 2,048 |
| Qwen2.5-14B | prefill | fp16 | 1 | 8,192 | 2,048 | 2,048 |
| Qwen2.5-14B | generate | fp16 | 1 | >= 8,192 | 2,048 | 2,048 |
| Qwen2.5-14B | generate_static_compiled | fp16 | 1 | >= 8,192 | 2,048 | 2,048 |
| Qwen2.5-14B | generate | fp16 | 8 | 512 | 512 | 512 |
| Qwen2.5-14B | generate_static_compiled | fp16 | 8 | 512 | 512 | 512 |
| TinyLlama-1.1B-Chat-v1.0 | train | fp16 | 4 | 2,048 | 2,048 | 1,024 |
| TinyLlama-1.1B-Chat-v1.0 | train_ckpt | fp16 | 4 | >= 8,192 | >= 8,192 | 2,048 |
| Qwen2.5-3B (LoRA r16) | train_lora_ckpt | fp16 | 1 | 8,192 | 8,192 | 8,192 |
| Qwen2.5-3B (LoRA r16) | train_lora_ckpt | fp16 | 4 | 2,048 | 2,048 | 2,048 |
| Llama-3.1-8B-Instruct (LoRA r16) | train_lora_ckpt | fp16 | 1 | 8,192 | 4,096 | 4,096 |
| Llama-3.1-8B-Instruct (LoRA r16) | train_lora_ckpt | fp16 | 4 | 2,048 | 2,048 | OOM |
| Qwen2.5-14B (LoRA r16) | train_lora_ckpt | fp16 | 1 | OOM | OOM | OOM |
| Qwen2.5-14B (LoRA r16) | train_lora_ckpt | fp16 | 4 | OOM | OOM | OOM |
