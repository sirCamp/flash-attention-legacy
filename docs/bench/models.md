# End-to-end benchmarks on HuggingFace models

`python benchmarks/models.py` (raw results: `models_<gpu>.jsonl` in this folder; these tables: `--report`). transformers 5.20.2,
torch 2.7.1, fp16 weights, kernels at commit 5d1da96 (the earlier runs, at 5ad45ca, agree within the noise: median time
ratio new / old 0.995-1.014 per GPU and backend). GPUs: V100 32 GB, T4 16 GB, P100 16 GB, GTX 1080 Ti 11 GB, RTX 2080 Ti 11 GB. Models: TinyLlama 1.1B, Qwen2.5-3B / 14B, Gemma 2 2B, Mistral 7B,
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
- Beyond the main lengths (16k prefill, 8k prompts, 8k train), our kernel alone was run up to 128k prefill, 64k prompts (the
  compiled decode too) and 64k train with checkpointing, one repetition each, to find its limit.

Correctness.
- The loss of the train step agrees across the backends in all 71 train cases where more than one fits: largest difference
  0.0007 (Gemma 2 2B on the V100, B=1, 2k: 12.0065 vs 12.0072), fp16 noise; padded batches and LoRA included.
- Top-1 agreement of the prefill logits with eager (512 tokens): 0.992-1.000 for ours, 0.988-1.000 for SDPA (24 model / GPU
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
  (flash-linear-attention is not installed: its Triton kernels run without tensor cores below SM 8.0 and do not compile
  in fp16 on Pascal): host bound (~4900 kernel launches per 1k prefill), so
  its rows measure mostly the memory saving and the longer context.
- SDPA on these GPUs has no flash backend: it runs the memory-efficient or the math kernel (with a mask and GQA, HF repeats the
  KV heads).

#### NVIDIA GeForce GTX 1080 Ti: prefill (tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 1024 | 2,704 (2.13) | 2,118 (2.41) | 2,304 (2.36) | 1.17x | 0.85x | 0.23 GiB (10%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 4096 | 1,852 (2.34) | 1,079 (6.9) | 1,026 (6.27) | 1.72x | 0.58x | 3.93 GiB (63%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | 1,545 (2.62) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 16384 | 1,241 (3.17) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 1024 | 990 (5.98) | 904 (6.09) | 928 (6.05) | 1.07x | 0.94x | 0.07 GiB (1%) |
| Qwen2.5-3B | fp16 | 1 | 4096 | 699 (6.32) | 571 (8.51) | 604 (8.13) | 1.16x | 0.86x | 1.81 GiB (22%) |
| Qwen2.5-3B | fp16 | 1 | 8192 | 578 (6.78) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 16384 | 423 (7.69) | OOM | OOM | only ours | - | - |
| Qwen3.5-2B | fp16 | 1 | 1024 | 1,729 (3.66) | 1,703 (3.67) | 1,729 (3.66) | 1.00x | 1.00x | 0.00 GiB (0%) |
| Qwen3.5-2B | fp16 | 1 | 4096 | 1,310 (4.04) | 1,262 (4.99) | 1,322 (4.72) | 0.99x | 1.01x | 0.68 GiB (14%) |
| Qwen3.5-2B | fp16 | 1 | 8192 | 1,218 (4.55) | 1,163 (8.81) | 1,181 (7.97) | 1.03x | 0.97x | 3.42 GiB (43%) |
| Qwen3.5-2B | fp16 | 1 | 16384 | 1,031 (5.57) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 32768 | 852 (4.29) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 65536 | 540 (6.51) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 131072 | OOM | - | - | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 32768 | 321 (9.52) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 65536 | OOM | - | - | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 131072 | OOM | - | - | - | - | - |

#### NVIDIA GeForce GTX 1080 Ti: generate (decode tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 2048 | 35.8 (2.2) | 23.8 (3.34) | 31.8 (3.16) | 1.13x | 0.89x | 0.96 GiB (30%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | 36.5 (2.62) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 8 | 2048 | 214.6 (3.17) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 8 | 8192 | 151.8 (6.51) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 2048 | 21.3 (6.09) | 18.6 (6.61) | 19.7 (6.49) | 1.08x | 0.92x | 0.40 GiB (6%) |
| Qwen2.5-3B | fp16 | 1 | 8192 | 21.3 (6.78) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 8 | 2048 | 88.6 (7.69) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 8 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 2048 | 18.9 (3.79) | 14.1 (3.96) | 18.8 (3.87) | 1.01x | 0.99x | 0.08 GiB (2%) |
| Qwen3.5-2B | fp16 | 1 | 8192 | 19.2 (4.55) | 13.9 (8.82) | 18.8 (7.97) | 1.02x | 0.98x | 3.42 GiB (43%) |
| Qwen3.5-2B | fp16 | 8 | 2048 | 132.5 (5.71) | 96.1 (6.99) | 117.2 (6.35) | 1.13x | 0.88x | 0.64 GiB (10%) |
| Qwen3.5-2B | fp16 | 8 | 8192 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 16384 | 28.8 (3.17) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 32768 | 27.2 (4.29) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 65536 | 31.9 (6.51) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 16384 | 20.8 (7.69) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 32768 | 23.1 (9.52) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 65536 | OOM | - | - | - | - | - |

#### NVIDIA GeForce GTX 1080 Ti: train (tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 1024 | 920 (6.23) | 909 (9.25) | 914 (10.37) | 1.01x | 0.99x | 3.02 GiB (33%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 2048 | 876 (7.95) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 512 | 934 (7.85) | 313 (8.12) | OOM | 2.98x | 0.34x | 0.27 GiB (3%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 1024 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 2048 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 1024 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 2048 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 1024 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 2048 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |

#### NVIDIA GeForce GTX 1080 Ti: train_ckpt (tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 2048 | 715 (5.56) | 589 (6.85) | 535 (6.58) | 1.21x | 0.82x | 1.02 GiB (16%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 4096 | 489 (6.0) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | 346 (7.68) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 1024 | 553 (6.0) | 144 (6.0) | 469 (6.65) | 1.18x | 0.85x | 0.00 GiB (0%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 2048 | 458 (7.68) | 81 (7.7) | OOM | 5.65x | 0.18x | 0.02 GiB (0%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 4096 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 8192 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 32768 | OOM | - | - | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 65536 | OOM | - | - | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 2048 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 32768 | OOM | - | - | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 65536 | OOM | - | - | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 2048 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |

#### NVIDIA GeForce GTX 1080 Ti: train_lora_ckpt (tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-3B (LoRA r16) | fp16 | 1 | 2048 | 250 (9.73) | 248 (9.73) | 246 (9.74) | 1.01x | 0.99x | 0.00 GiB (0%) |
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
| TinyLlama-1.1B-Chat-v1.0 | generate | fp16 | 1 | >= 65,536 | 2,048 | 2,048 |
| TinyLlama-1.1B-Chat-v1.0 | generate | fp16 | 8 | >= 8,192 | OOM | OOM |
| TinyLlama-1.1B-Chat-v1.0 | train | fp16 | 1 | 2,048 | 1,024 | 1,024 |
| TinyLlama-1.1B-Chat-v1.0 | train | fp16 | 4 | 512 | 512 | OOM |
| TinyLlama-1.1B-Chat-v1.0 | train_ckpt | fp16 | 1 | 8,192 | 2,048 | 2,048 |
| TinyLlama-1.1B-Chat-v1.0 | train_ckpt | fp16 | 4 | 2,048 | 2,048 | 1,024 |
| Qwen2.5-3B | prefill | fp16 | 1 | 32,768 | 4,096 | 4,096 |
| Qwen2.5-3B | generate | fp16 | 1 | 32,768 | 2,048 | 2,048 |
| Qwen2.5-3B | generate | fp16 | 8 | 2,048 | OOM | OOM |
| Qwen2.5-3B | train | fp16 | 1 | OOM | OOM | OOM |
| Qwen2.5-3B | train_ckpt | fp16 | 1 | OOM | OOM | OOM |
| Qwen2.5-3B (LoRA r16) | train_lora_ckpt | fp16 | 1 | 2,048 | 2,048 | 2,048 |
| Qwen2.5-3B (LoRA r16) | train_lora_ckpt | fp16 | 4 | OOM | OOM | OOM |
| Qwen3.5-2B | prefill | fp16 | 1 | >= 16,384 | 8,192 | 8,192 |
| Qwen3.5-2B | generate | fp16 | 1 | >= 8,192 | >= 8,192 | >= 8,192 |
| Qwen3.5-2B | generate | fp16 | 8 | 2,048 | 2,048 | 2,048 |
| Qwen3.5-2B | train | fp16 | 1 | OOM | OOM | OOM |
| Qwen3.5-2B | train_ckpt | fp16 | 1 | OOM | OOM | OOM |

#### NVIDIA GeForce RTX 2080 Ti: prefill (tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 1024 | 15,631 (2.13) | 7,045 (2.41) | 10,272 (2.36) | 1.52x | 0.66x | 0.23 GiB (10%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 4096 | 15,248 (2.34) | 2,797 (6.9) | 4,749 (6.27) | 3.21x | 0.31x | 3.93 GiB (63%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | 13,552 (2.62) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 16384 | 10,854 (3.17) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 1024 | 6,092 (5.98) | 4,202 (6.09) | 5,358 (6.05) | 1.14x | 0.88x | 0.07 GiB (1%) |
| Qwen2.5-3B | fp16 | 1 | 4096 | 6,208 (6.32) | 2,235 (8.51) | 3,605 (8.13) | 1.72x | 0.58x | 1.81 GiB (22%) |
| Qwen2.5-3B | fp16 | 1 | 8192 | 5,742 (6.78) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 16384 | 4,951 (7.69) | OOM | OOM | only ours | - | - |
| Qwen3.5-2B | fp16 | 1 | 1024 | 7,263 (3.66) | 6,861 (3.67) | 7,203 (3.66) | 1.01x | 0.99x | 0.00 GiB (0%) |
| Qwen3.5-2B | fp16 | 1 | 4096 | 7,439 (4.04) | 5,978 (4.99) | 6,956 (4.72) | 1.07x | 0.94x | 0.68 GiB (14%) |
| Qwen3.5-2B | fp16 | 1 | 8192 | 7,214 (4.55) | 4,517 (8.81) | 6,374 (7.97) | 1.13x | 0.88x | 3.42 GiB (43%) |
| Qwen3.5-2B | fp16 | 1 | 16384 | 6,923 (5.57) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 32768 | 7,859 (4.29) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 65536 | 4,913 (6.51) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 131072 | OOM | - | - | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 32768 | OOM | - | - | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 65536 | OOM | - | - | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 131072 | OOM | - | - | - | - | - |

#### NVIDIA GeForce RTX 2080 Ti: generate (decode tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 2048 | 66.6 (2.2) | 57.9 (3.34) | 56.5 (3.16) | 1.15x | 0.87x | 0.96 GiB (30%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | 66.1 (2.62) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 8 | 2048 | 519.5 (3.17) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 8 | 8192 | 446.2 (6.51) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 2048 | 39.9 (6.09) | 35.1 (6.61) | 35.6 (6.49) | 1.12x | 0.89x | 0.40 GiB (6%) |
| Qwen2.5-3B | fp16 | 1 | 8192 | 39.8 (6.78) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 8 | 2048 | 308.2 (7.69) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 8 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 2048 | 33.8 (3.79) | 33.3 (3.96) | 33.8 (3.87) | 1.00x | 1.00x | 0.08 GiB (2%) |
| Qwen3.5-2B | fp16 | 1 | 8192 | 33.5 (4.55) | 31.8 (8.82) | 33.1 (7.97) | 1.01x | 0.99x | 3.42 GiB (43%) |
| Qwen3.5-2B | fp16 | 8 | 2048 | 265.5 (5.71) | 242.9 (6.99) | 256.1 (6.35) | 1.04x | 0.96x | 0.64 GiB (10%) |
| Qwen3.5-2B | fp16 | 8 | 8192 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 16384 | 67.5 (3.17) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 32768 | 65.4 (4.29) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 65536 | 56.8 (6.51) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 16384 | 39.4 (7.69) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 32768 | 38.9 (9.52) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 65536 | OOM | - | - | - | - | - |

#### NVIDIA GeForce RTX 2080 Ti: generate_static_compiled (decode tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 2048 | 174.8 (2.21) | 100.9 (3.35) | 134.9 (3.23) | 1.30x | 0.77x | 1.02 GiB (32%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | 153.7 (2.62) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 8 | 2048 | 1,070.0 (3.19) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 8 | 8192 | 809.5 (6.54) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 2048 | 69.9 (6.12) | 51.5 (6.63) | 59.1 (6.54) | 1.18x | 0.85x | 0.42 GiB (6%) |
| Qwen2.5-3B | fp16 | 1 | 8192 | 67.4 (6.78) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 8 | 2048 | 477.3 (7.73) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 8 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 2048 | 103.7 (3.79) | 90.7 (3.96) | 100.4 (3.89) | 1.03x | 0.97x | 0.10 GiB (3%) |
| Qwen3.5-2B | fp16 | 1 | 8192 | 102.2 (4.57) | 62.2 (8.82) | 79.8 (8.04) | 1.28x | 0.78x | 3.47 GiB (43%) |
| Qwen3.5-2B | fp16 | 8 | 2048 | 690.3 (5.76) | 539.6 (7.0) | 545.5 (6.5) | 1.27x | 0.79x | 0.74 GiB (11%) |
| Qwen3.5-2B | fp16 | 8 | 8192 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 16384 | 139.6 (3.18) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 32768 | 125.7 (4.29) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 65536 | 103.4 (6.52) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 16384 | 62.7 (7.69) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 32768 | OOM | - | - | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 65536 | OOM | - | - | - | - | - |

#### NVIDIA GeForce RTX 2080 Ti: train (tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 1024 | 4,940 (6.23) | 2,892 (9.25) | OOM | 1.71x | 0.59x | 3.02 GiB (33%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 2048 | 5,100 (7.95) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 512 | 4,385 (7.85) | 4,163 (8.12) | OOM | 1.05x | 0.95x | 0.27 GiB (3%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 1024 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 2048 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 1024 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 2048 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 1024 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 2048 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |

#### NVIDIA GeForce RTX 2080 Ti: train_ckpt (tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 2048 | 4,030 (5.56) | 1,515 (6.85) | 1,992 (6.58) | 2.02x | 0.49x | 1.02 GiB (16%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 4096 | 3,851 (6.0) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | 3,162 (7.68) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 1024 | 3,611 (6.0) | 3,165 (6.0) | 2,324 (6.65) | 1.14x | 0.88x | 0.00 GiB (0%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 2048 | 3,568 (7.68) | 2,713 (7.7) | OOM | 1.32x | 0.76x | 0.02 GiB (0%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 4096 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 8192 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 32768 | OOM | - | - | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 65536 | OOM | - | - | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 2048 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 32768 | OOM | - | - | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 65536 | OOM | - | - | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 2048 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |

#### NVIDIA GeForce RTX 2080 Ti: train_lora_ckpt (tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-3B (LoRA r16) | fp16 | 1 | 2048 | 1,768 (9.73) | OOM | OOM | only ours | - | - |
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
| TinyLlama-1.1B-Chat-v1.0 | generate_static_compiled | fp16 | 1 | >= 65,536 | 2,048 | 2,048 |
| TinyLlama-1.1B-Chat-v1.0 | generate | fp16 | 8 | >= 8,192 | OOM | OOM |
| TinyLlama-1.1B-Chat-v1.0 | generate_static_compiled | fp16 | 8 | >= 8,192 | OOM | OOM |
| TinyLlama-1.1B-Chat-v1.0 | train | fp16 | 1 | 2,048 | 1,024 | OOM |
| TinyLlama-1.1B-Chat-v1.0 | train | fp16 | 4 | 512 | 512 | OOM |
| TinyLlama-1.1B-Chat-v1.0 | train_ckpt | fp16 | 1 | 8,192 | 2,048 | 2,048 |
| TinyLlama-1.1B-Chat-v1.0 | train_ckpt | fp16 | 4 | 2,048 | 2,048 | 1,024 |
| Qwen2.5-3B | prefill | fp16 | 1 | 16,384 | 4,096 | 4,096 |
| Qwen2.5-3B | generate | fp16 | 1 | 32,768 | 2,048 | 2,048 |
| Qwen2.5-3B | generate_static_compiled | fp16 | 1 | 16,384 | 2,048 | 2,048 |
| Qwen2.5-3B | generate | fp16 | 8 | 2,048 | OOM | OOM |
| Qwen2.5-3B | generate_static_compiled | fp16 | 8 | 2,048 | OOM | OOM |
| Qwen2.5-3B | train | fp16 | 1 | OOM | OOM | OOM |
| Qwen2.5-3B | train_ckpt | fp16 | 1 | OOM | OOM | OOM |
| Qwen2.5-3B (LoRA r16) | train_lora_ckpt | fp16 | 1 | 2,048 | OOM | OOM |
| Qwen2.5-3B (LoRA r16) | train_lora_ckpt | fp16 | 4 | OOM | OOM | OOM |
| Qwen3.5-2B | prefill | fp16 | 1 | >= 16,384 | 8,192 | 8,192 |
| Qwen3.5-2B | generate | fp16 | 1 | >= 8,192 | >= 8,192 | >= 8,192 |
| Qwen3.5-2B | generate_static_compiled | fp16 | 1 | >= 8,192 | >= 8,192 | >= 8,192 |
| Qwen3.5-2B | generate | fp16 | 8 | 2,048 | 2,048 | 2,048 |
| Qwen3.5-2B | generate_static_compiled | fp16 | 8 | 2,048 | 2,048 | 2,048 |
| Qwen3.5-2B | train | fp16 | 1 | OOM | OOM | OOM |
| Qwen3.5-2B | train_ckpt | fp16 | 1 | OOM | OOM | OOM |

#### Tesla P100-PCIE-16GB: prefill (tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 1024 | 3,255 (2.13) | 2,592 (2.41) | 2,777 (2.36) | 1.17x | 0.85x | 0.23 GiB (10%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 4096 | 3,082 (2.34) | 1,523 (6.9) | 1,805 (6.27) | 1.71x | 0.59x | 3.93 GiB (63%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | 2,469 (2.62) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 16384 | 1,837 (3.17) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 1024 | 1,058 (5.98) | 1,028 (6.09) | 1,055 (6.05) | 1.00x | 1.00x | 0.07 GiB (1%) |
| Qwen2.5-3B | fp16 | 1 | 4096 | 1,062 (6.32) | 869 (8.51) | 951 (8.13) | 1.12x | 0.90x | 1.81 GiB (22%) |
| Qwen2.5-3B | fp16 | 1 | 8192 | 919 (6.78) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 16384 | 708 (7.69) | OOM | OOM | only ours | - | - |
| Qwen3.5-2B | fp16 | 1 | 1024 | 2,033 (3.66) | 2,027 (3.67) | 2,039 (3.66) | 1.00x | 1.00x | 0.00 GiB (0%) |
| Qwen3.5-2B | fp16 | 1 | 4096 | 2,134 (4.04) | 2,114 (4.99) | 2,158 (4.72) | 0.99x | 1.01x | 0.68 GiB (14%) |
| Qwen3.5-2B | fp16 | 1 | 8192 | 1,865 (4.55) | 1,830 (8.81) | 1,903 (7.97) | 0.98x | 1.02x | 3.42 GiB (43%) |
| Qwen3.5-2B | fp16 | 1 | 16384 | 1,595 (5.57) | OOM | OOM | only ours | - | - |
| Llama-3.1-8B-Instruct | fp16 | 1 | 512 | 469 (15.08) | 464 (15.15) | 470 (15.11) | 1.00x | 1.00x | 0.03 GiB (0%) |
| Llama-3.1-8B-Instruct | fp16 | 1 | 1024 | 467 (15.2) | 453 (15.48) | 462 (15.39) | 1.01x | 0.99x | 0.19 GiB (1%) |
| Llama-3.1-8B-Instruct | fp16 | 1 | 2048 | 531 (15.44) | OOM | OOM | only ours | - | - |
| Llama-3.1-8B-Instruct | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Llama-3.1-8B-Instruct | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Llama-3.1-8B-Instruct | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-4B | fp16 | 1 | 1024 | 768 (8.26) | 764 (8.27) | 774 (8.26) | 0.99x | 1.01x | 0.00 GiB (0%) |
| Qwen3.5-4B | fp16 | 1 | 4096 | 799 (9.03) | 791 (10.82) | 807 (10.33) | 0.99x | 1.01x | 1.30 GiB (13%) |
| Qwen3.5-4B | fp16 | 1 | 8192 | 734 (10.08) | OOM | OOM | only ours | - | - |
| Qwen3.5-4B | fp16 | 1 | 16384 | 629 (12.17) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 32768 | 1,258 (4.29) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 65536 | 758 (6.51) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 131072 | 428 (10.97) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 32768 | 500 (9.52) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 65536 | 314 (13.18) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 131072 | OOM | - | - | - | - | - |
| Qwen3.5-4B | fp16 | 1 | 32768 | OOM | OOM | OOM | - | - | - |
| Mistral-7B-Instruct-v0.2 | fp16 | 1 | 1024 | 468 (13.74) | 451 (14.01) | 456 (13.92) | 1.03x | 0.97x | 0.18 GiB (1%) |
| Mistral-7B-Instruct-v0.2 | fp16 | 1 | 4096 | 467 (14.45) | OOM | OOM | only ours | - | - |
| Mistral-7B-Instruct-v0.2 | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |

#### Tesla P100-PCIE-16GB: generate (decode tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 2048 | 78.1 (2.2) | 54.6 (3.34) | 63.2 (3.16) | 1.24x | 0.81x | 0.96 GiB (30%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | 77.8 (2.62) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 8 | 2048 | 284.2 (3.17) | 96.7 (12.2) | 123.0 (10.84) | 2.31x | 0.43x | 7.67 GiB (71%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 8 | 8192 | 188.2 (6.51) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 2048 | 37.4 (6.09) | 25.1 (6.61) | 29.3 (6.49) | 1.28x | 0.78x | 0.40 GiB (6%) |
| Qwen2.5-3B | fp16 | 1 | 8192 | 34.0 (6.78) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 8 | 2048 | 112.7 (7.69) | 52.2 (11.73) | 63.8 (10.87) | 1.77x | 0.57x | 3.18 GiB (29%) |
| Qwen2.5-3B | fp16 | 8 | 8192 | 88.9 (13.18) | OOM | OOM | only ours | - | - |
| Qwen3.5-2B | fp16 | 1 | 2048 | 39.8 (3.79) | 38.9 (3.96) | 39.2 (3.87) | 1.02x | 0.98x | 0.08 GiB (2%) |
| Qwen3.5-2B | fp16 | 1 | 8192 | 39.9 (4.55) | 33.3 (8.82) | 38.5 (7.97) | 1.04x | 0.96x | 3.42 GiB (43%) |
| Qwen3.5-2B | fp16 | 8 | 2048 | 171.9 (5.71) | 131.0 (6.99) | 145.4 (6.35) | 1.18x | 0.85x | 0.64 GiB (10%) |
| Qwen3.5-2B | fp16 | 8 | 8192 | 152.1 (11.83) | OOM | OOM | only ours | - | - |
| Llama-3.1-8B-Instruct | fp16 | 1 | 512 | 21.4 (15.08) | 18.9 (15.15) | 19.9 (15.11) | 1.08x | 0.93x | 0.03 GiB (0%) |
| Llama-3.1-8B-Instruct | fp16 | 1 | 2048 | 20.3 (15.44) | OOM | OOM | only ours | - | - |
| Llama-3.1-8B-Instruct | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Llama-3.1-8B-Instruct | fp16 | 8 | 512 | OOM | OOM | OOM | - | - | - |
| Llama-3.1-8B-Instruct | fp16 | 8 | 2048 | OOM | OOM | OOM | - | - | - |
| Llama-3.1-8B-Instruct | fp16 | 8 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-4B | fp16 | 1 | 2048 | 25.2 (8.51) | 21.7 (8.83) | 23.3 (8.65) | 1.08x | 0.92x | 0.14 GiB (2%) |
| Qwen3.5-4B | fp16 | 1 | 8192 | 24.0 (10.08) | OOM | OOM | only ours | - | - |
| Qwen3.5-4B | fp16 | 8 | 2048 | 71.6 (12.52) | 53.6 (14.95) | 60.6 (13.62) | 1.18x | 0.85x | 1.10 GiB (8%) |
| Qwen3.5-4B | fp16 | 8 | 8192 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 16384 | 65.7 (3.17) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 32768 | 48.9 (4.29) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 65536 | 38.7 (6.51) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 16384 | 29.3 (7.69) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 32768 | 23.6 (9.52) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 65536 | 16.7 (13.18) | - | - | only ours | - | - |
| Mistral-7B-Instruct-v0.2 | fp16 | 1 | 2048 | 21.1 (13.97) | 14.9 (15.09) | 16.8 (14.85) | 1.26x | 0.80x | 0.88 GiB (6%) |
| Mistral-7B-Instruct-v0.2 | fp16 | 8 | 2048 | OOM | OOM | OOM | - | - | - |

#### Tesla P100-PCIE-16GB: train (tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 1024 | 1,180 (6.23) | 1,085 (9.25) | 1,103 (10.37) | 1.07x | 0.93x | 3.02 GiB (33%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 2048 | 1,117 (7.95) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 4096 | 993 (11.53) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 512 | 1,049 (7.85) | 978 (8.12) | 990 (12.2) | 1.06x | 0.94x | 0.27 GiB (3%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 1024 | 1,066 (11.35) | 895 (11.96) | OOM | 1.19x | 0.84x | 0.61 GiB (5%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 2048 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 1024 | 432 (14.84) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 2048 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 1024 | 639 (11.49) | 644 (11.76) | 645 (11.78) | 0.99x | 1.01x | 0.27 GiB (2%) |
| Qwen3.5-2B | fp16 | 1 | 2048 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-4B | fp16 | 1 | 1024 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-4B | fp16 | 1 | 2048 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-4B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-4B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Mistral-7B-Instruct-v0.2 | fp16 | 1 | 1024 | OOM | OOM | OOM | - | - | - |
| Mistral-7B-Instruct-v0.2 | fp16 | 1 | 2048 | OOM | OOM | OOM | - | - | - |
| Mistral-7B-Instruct-v0.2 | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Mistral-7B-Instruct-v0.2 | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |

#### Tesla P100-PCIE-16GB: train_ckpt (tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 2048 | 885 (5.56) | 692 (6.85) | 705 (6.58) | 1.26x | 0.80x | 1.02 GiB (16%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 4096 | 789 (6.0) | 496 (12.94) | 537 (11.92) | 1.47x | 0.68x | 5.92 GiB (50%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | 583 (7.68) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 16384 | 384 (11.36) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 1024 | 825 (6.0) | 699 (6.0) | 692 (6.65) | 1.18x | 0.85x | 0.00 GiB (0%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 2048 | 759 (7.68) | 556 (7.7) | 535 (12.05) | 1.37x | 0.73x | 0.02 GiB (0%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 4096 | 649 (11.36) | 395 (11.42) | OOM | 1.64x | 0.61x | 0.06 GiB (1%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 8192 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 32768 | OOM | - | - | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 65536 | OOM | - | - | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 2048 | 345 (12.86) | 333 (12.86) | 336 (12.87) | 1.03x | 0.97x | 0.00 GiB (0%) |
| Qwen2.5-3B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 32768 | OOM | - | - | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 65536 | OOM | - | - | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 2048 | 445 (10.23) | 449 (10.23) | 449 (10.24) | 0.99x | 1.01x | 0.00 GiB (0%) |
| Qwen3.5-2B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-4B | fp16 | 1 | 2048 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-4B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-4B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-4B | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |

#### Tesla P100-PCIE-16GB: train_lora_ckpt (tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-3B (LoRA r16) | fp16 | 1 | 2048 | 352 (9.73) | 340 (9.73) | 343 (9.74) | 1.03x | 0.97x | 0.00 GiB (0%) |
| Qwen2.5-3B (LoRA r16) | fp16 | 1 | 4096 | 310 (13.5) | 287 (13.5) | OOM | 1.08x | 0.93x | 0.00 GiB (0%) |
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

#### Tesla P100-PCIE-16GB: longest N that fits (>= N: no longer N measured)

| model | mode | dtype | B | flash_attn_legacy | sdpa | eager |
|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | prefill | fp16 | 1 | >= 131,072 | 4,096 | 4,096 |
| TinyLlama-1.1B-Chat-v1.0 | generate | fp16 | 1 | >= 65,536 | 2,048 | 2,048 |
| TinyLlama-1.1B-Chat-v1.0 | generate | fp16 | 8 | >= 8,192 | 2,048 | 2,048 |
| TinyLlama-1.1B-Chat-v1.0 | train | fp16 | 1 | 4,096 | 1,024 | 1,024 |
| TinyLlama-1.1B-Chat-v1.0 | train | fp16 | 4 | 1,024 | 1,024 | 512 |
| TinyLlama-1.1B-Chat-v1.0 | train_ckpt | fp16 | 1 | 16,384 | 4,096 | 4,096 |
| TinyLlama-1.1B-Chat-v1.0 | train_ckpt | fp16 | 4 | 4,096 | 4,096 | 2,048 |
| Qwen2.5-3B | prefill | fp16 | 1 | 65,536 | 4,096 | 4,096 |
| Qwen2.5-3B | generate | fp16 | 1 | >= 65,536 | 2,048 | 2,048 |
| Qwen2.5-3B | generate | fp16 | 8 | >= 8,192 | 2,048 | 2,048 |
| Qwen2.5-3B | train | fp16 | 1 | 1,024 | OOM | OOM |
| Qwen2.5-3B | train_ckpt | fp16 | 1 | 2,048 | 2,048 | 2,048 |
| Qwen2.5-3B (LoRA r16) | train_lora_ckpt | fp16 | 1 | 4,096 | 4,096 | 2,048 |
| Qwen2.5-3B (LoRA r16) | train_lora_ckpt | fp16 | 4 | OOM | OOM | OOM |
| Qwen3.5-2B | prefill | fp16 | 1 | >= 16,384 | 8,192 | 8,192 |
| Qwen3.5-2B | generate | fp16 | 1 | >= 8,192 | >= 8,192 | >= 8,192 |
| Qwen3.5-2B | generate | fp16 | 8 | >= 8,192 | 2,048 | 2,048 |
| Qwen3.5-2B | train | fp16 | 1 | 1,024 | 1,024 | 1,024 |
| Qwen3.5-2B | train_ckpt | fp16 | 1 | 2,048 | 2,048 | 2,048 |
| Llama-3.1-8B-Instruct | generate | fp16 | 1 | 2,048 | 512 | 512 |
| Llama-3.1-8B-Instruct | generate | fp16 | 8 | OOM | OOM | OOM |
| Llama-3.1-8B-Instruct | prefill | fp16 | 1 | 2,048 | 1,024 | 1,024 |
| Llama-3.1-8B-Instruct (LoRA r16) | train_lora_ckpt | fp16 | 1 | OOM | OOM | OOM |
| Llama-3.1-8B-Instruct (LoRA r16) | train_lora_ckpt | fp16 | 4 | OOM | OOM | OOM |
| Qwen3.5-4B | prefill | fp16 | 1 | 16,384 | 4,096 | 4,096 |
| Qwen3.5-4B | generate | fp16 | 1 | >= 8,192 | 2,048 | 2,048 |
| Qwen3.5-4B | generate | fp16 | 8 | 2,048 | 2,048 | 2,048 |
| Qwen3.5-4B | train | fp16 | 1 | OOM | OOM | OOM |
| Qwen3.5-4B | train_ckpt | fp16 | 1 | OOM | OOM | OOM |
| Mistral-7B-Instruct-v0.2 | prefill | fp16 | 1 | 4,096 | 1,024 | 1,024 |
| Mistral-7B-Instruct-v0.2 | generate | fp16 | 1 | >= 2,048 | >= 2,048 | >= 2,048 |
| Mistral-7B-Instruct-v0.2 | generate | fp16 | 8 | OOM | OOM | OOM |
| Mistral-7B-Instruct-v0.2 | train | fp16 | 1 | OOM | OOM | OOM |

#### Tesla T4: prefill (tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 1024 | 8,851 (2.13) | 3,740 (2.41) | 5,531 (2.36) | 1.60x | 0.62x | 0.23 GiB (10%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 4096 | 8,881 (2.34) | 1,312 (6.9) | 2,239 (6.27) | 3.97x | 0.25x | 3.93 GiB (63%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | 7,831 (2.62) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 16384 | 6,129 (3.17) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 1024 | 3,519 (5.98) | 2,351 (6.09) | 3,017 (6.05) | 1.17x | 0.86x | 0.07 GiB (1%) |
| Qwen2.5-3B | fp16 | 1 | 4096 | 3,370 (6.32) | 1,098 (8.51) | 1,718 (8.13) | 1.96x | 0.51x | 1.81 GiB (22%) |
| Qwen2.5-3B | fp16 | 1 | 8192 | 3,224 (6.78) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 16384 | 2,522 (7.69) | OOM | OOM | only ours | - | - |
| Qwen3.5-2B | fp16 | 1 | 1024 | 4,175 (3.66) | 3,926 (3.67) | 4,196 (3.66) | 0.99x | 1.01x | 0.00 GiB (0%) |
| Qwen3.5-2B | fp16 | 1 | 4096 | 4,132 (4.04) | 3,158 (4.99) | 3,753 (4.72) | 1.10x | 0.91x | 0.68 GiB (14%) |
| Qwen3.5-2B | fp16 | 1 | 8192 | 3,965 (4.55) | 2,450 (8.81) | 3,413 (7.97) | 1.16x | 0.86x | 3.42 GiB (43%) |
| Qwen3.5-2B | fp16 | 1 | 16384 | 3,627 (5.57) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 32768 | 4,325 (4.29) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 65536 | 2,540 (6.51) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 131072 | 1,430 (10.97) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 32768 | 1,940 (9.52) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 65536 | OOM | - | - | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 131072 | OOM | - | - | - | - | - |
| Qwen3.5-4B | fp16 | 1 | 1024 | 1,709 (8.26) | 1,578 (8.27) | 1,664 (8.26) | 1.03x | 0.97x | 0.00 GiB (0%) |
| Qwen3.5-4B | fp16 | 1 | 4096 | 1,663 (9.03) | 1,263 (10.82) | 1,514 (10.33) | 1.10x | 0.91x | 1.30 GiB (13%) |
| Qwen3.5-4B | fp16 | 1 | 8192 | 1,556 (10.08) | OOM | OOM | only ours | - | - |
| Qwen3.5-4B | fp16 | 1 | 16384 | 1,431 (12.17) | OOM | OOM | only ours | - | - |

#### Tesla T4: generate (decode tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 2048 | 49.3 (2.2) | 42.6 (3.34) | 44.0 (3.16) | 1.12x | 0.89x | 0.96 GiB (30%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | 48.9 (2.62) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 8 | 2048 | 377.5 (3.17) | 77.5 (12.2) | OOM | 4.87x | 0.21x | 9.03 GiB (74%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 8 | 8192 | 207.2 (6.51) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 2048 | 28.4 (6.09) | 20.1 (6.61) | 25.7 (6.49) | 1.11x | 0.90x | 0.40 GiB (6%) |
| Qwen2.5-3B | fp16 | 1 | 8192 | 28.1 (6.78) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 8 | 2048 | 183.0 (7.69) | 43.0 (11.73) | 83.7 (10.87) | 2.19x | 0.46x | 3.18 GiB (29%) |
| Qwen2.5-3B | fp16 | 8 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 2048 | 24.4 (3.79) | 23.5 (3.96) | 23.9 (3.87) | 1.02x | 0.98x | 0.08 GiB (2%) |
| Qwen3.5-2B | fp16 | 1 | 8192 | 23.9 (4.55) | 22.3 (8.82) | 23.2 (7.97) | 1.03x | 0.97x | 3.42 GiB (43%) |
| Qwen3.5-2B | fp16 | 8 | 2048 | 185.1 (5.71) | 121.0 (6.99) | 175.4 (6.35) | 1.06x | 0.95x | 0.64 GiB (10%) |
| Qwen3.5-2B | fp16 | 8 | 8192 | 169.8 (11.83) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 16384 | 48.6 (3.17) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 32768 | 42.2 (4.29) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 65536 | 25.3 (6.51) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 16384 | 24.5 (7.69) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 32768 | 19.8 (9.52) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 65536 | OOM | - | - | - | - | - |
| Qwen3.5-4B | fp16 | 1 | 2048 | 18.0 (8.51) | 17.1 (8.83) | 17.7 (8.65) | 1.02x | 0.98x | 0.14 GiB (2%) |
| Qwen3.5-4B | fp16 | 1 | 8192 | 17.4 (10.08) | OOM | OOM | only ours | - | - |
| Qwen3.5-4B | fp16 | 8 | 2048 | 102.4 (12.52) | OOM | 80.0 (13.62) | 1.28x | 0.78x | 1.10 GiB (8%) |
| Qwen3.5-4B | fp16 | 8 | 8192 | OOM | OOM | OOM | - | - | - |

#### Tesla T4: generate_static_compiled (decode tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 2048 | 91.4 (2.21) | 60.6 (3.35) | 69.0 (3.23) | 1.32x | 0.75x | 1.02 GiB (32%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | 81.8 (2.62) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 8 | 2048 | 556.7 (3.19) | 172.0 (12.23) | OOM | 3.24x | 0.31x | 9.04 GiB (74%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 8 | 8192 | 396.4 (6.54) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 2048 | 34.7 (6.12) | 27.6 (6.63) | 29.6 (6.54) | 1.17x | 0.85x | 0.42 GiB (6%) |
| Qwen2.5-3B | fp16 | 1 | 8192 | 32.6 (6.78) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 8 | 2048 | 230.3 (7.73) | 106.1 (11.76) | 103.9 (11.17) | 2.17x | 0.46x | 3.44 GiB (31%) |
| Qwen2.5-3B | fp16 | 8 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 2048 | 54.1 (3.79) | 48.6 (3.96) | 51.6 (3.89) | 1.05x | 0.95x | 0.10 GiB (3%) |
| Qwen3.5-2B | fp16 | 1 | 8192 | 52.8 (4.57) | 36.1 (8.82) | 40.3 (8.04) | 1.31x | 0.76x | 3.47 GiB (43%) |
| Qwen3.5-2B | fp16 | 8 | 2048 | 330.0 (5.76) | 245.1 (7.0) | 250.4 (6.5) | 1.32x | 0.76x | 0.74 GiB (11%) |
| Qwen3.5-2B | fp16 | 8 | 8192 | 299.6 (11.98) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 16384 | 76.2 (3.18) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 32768 | 69.0 (4.29) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 65536 | 45.9 (6.52) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 16384 | 30.8 (7.69) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 32768 | 28.0 (9.52) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 65536 | OOM | - | - | - | - | - |
| Qwen3.5-4B | fp16 | 1 | 2048 | 25.3 (8.52) | 23.2 (8.83) | 23.8 (8.68) | 1.06x | 0.94x | 0.16 GiB (2%) |
| Qwen3.5-4B | fp16 | 1 | 8192 | 24.7 (10.12) | OOM | OOM | only ours | - | - |
| Qwen3.5-4B | fp16 | 8 | 2048 | 155.2 (12.63) | OOM | OOM | only ours | - | - |
| Qwen3.5-4B | fp16 | 8 | 8192 | OOM | OOM | OOM | - | - | - |

#### Tesla T4: train (tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 1024 | 2,687 (6.23) | 1,514 (9.25) | 1,879 (10.37) | 1.43x | 0.70x | 3.02 GiB (33%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 2048 | 2,868 (7.95) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 4096 | 2,736 (11.53) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 512 | 2,491 (7.85) | 2,348 (8.12) | 1,980 (12.2) | 1.06x | 0.94x | 0.27 GiB (3%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 1024 | 2,619 (11.35) | 2,267 (11.96) | OOM | 1.16x | 0.87x | 0.61 GiB (5%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 2048 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 1024 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 2048 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 1024 | 920 (11.49) | 896 (11.76) | 917 (11.78) | 1.00x | 1.00x | 0.27 GiB (2%) |
| Qwen3.5-2B | fp16 | 1 | 2048 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-4B | fp16 | 1 | 1024 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-4B | fp16 | 1 | 2048 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-4B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-4B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |

#### Tesla T4: train_ckpt (tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 2048 | 2,298 (5.56) | 762 (6.85) | 998 (6.58) | 2.30x | 0.43x | 1.02 GiB (16%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 4096 | 2,106 (6.0) | OOM | 661 (11.92) | 3.19x | 0.31x | 5.92 GiB (50%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | 1,699 (7.68) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 16384 | 1,250 (11.36) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 1024 | 2,023 (6.0) | 1,742 (6.0) | 1,229 (6.65) | 1.16x | 0.86x | 0.00 GiB (0%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 2048 | 1,960 (7.68) | 1,422 (7.7) | 768 (12.05) | 1.38x | 0.73x | 0.02 GiB (0%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 4096 | 1,789 (11.36) | 1,051 (11.42) | OOM | 1.70x | 0.59x | 0.06 GiB (1%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 8192 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 32768 | OOM | - | - | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 65536 | OOM | - | - | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 2048 | 969 (12.86) | 522 (12.86) | 701 (12.87) | 1.38x | 0.72x | 0.00 GiB (0%) |
| Qwen2.5-3B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 32768 | OOM | - | - | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 65536 | OOM | - | - | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 2048 | 530 (10.23) | 505 (10.23) | 526 (10.24) | 1.01x | 0.99x | 0.00 GiB (0%) |
| Qwen3.5-2B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |

#### Tesla T4: train_lora_ckpt (tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-3B (LoRA r16) | fp16 | 1 | 2048 | 992 (9.73) | 529 (9.73) | 711 (9.74) | 1.40x | 0.72x | 0.00 GiB (0%) |
| Qwen2.5-3B (LoRA r16) | fp16 | 1 | 4096 | 903 (13.5) | 337 (13.5) | OOM | 2.68x | 0.37x | 0.00 GiB (0%) |
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
| TinyLlama-1.1B-Chat-v1.0 | generate | fp16 | 1 | >= 65,536 | 2,048 | 2,048 |
| TinyLlama-1.1B-Chat-v1.0 | generate_static_compiled | fp16 | 1 | >= 65,536 | 2,048 | 2,048 |
| TinyLlama-1.1B-Chat-v1.0 | generate | fp16 | 8 | >= 8,192 | 2,048 | OOM |
| TinyLlama-1.1B-Chat-v1.0 | generate_static_compiled | fp16 | 8 | >= 8,192 | 2,048 | OOM |
| TinyLlama-1.1B-Chat-v1.0 | train | fp16 | 1 | 4,096 | 1,024 | 1,024 |
| TinyLlama-1.1B-Chat-v1.0 | train | fp16 | 4 | 1,024 | 1,024 | 512 |
| TinyLlama-1.1B-Chat-v1.0 | train_ckpt | fp16 | 1 | 16,384 | 2,048 | 4,096 |
| TinyLlama-1.1B-Chat-v1.0 | train_ckpt | fp16 | 4 | 4,096 | 4,096 | 2,048 |
| Qwen2.5-3B | prefill | fp16 | 1 | 32,768 | 4,096 | 4,096 |
| Qwen2.5-3B | generate | fp16 | 1 | 32,768 | 2,048 | 2,048 |
| Qwen2.5-3B | generate_static_compiled | fp16 | 1 | 32,768 | 2,048 | 2,048 |
| Qwen2.5-3B | generate | fp16 | 8 | 2,048 | 2,048 | 2,048 |
| Qwen2.5-3B | generate_static_compiled | fp16 | 8 | 2,048 | 2,048 | 2,048 |
| Qwen2.5-3B | train | fp16 | 1 | OOM | OOM | OOM |
| Qwen2.5-3B | train_ckpt | fp16 | 1 | 2,048 | 2,048 | 2,048 |
| Qwen2.5-3B (LoRA r16) | train_lora_ckpt | fp16 | 1 | 4,096 | 4,096 | 2,048 |
| Qwen2.5-3B (LoRA r16) | train_lora_ckpt | fp16 | 4 | OOM | OOM | OOM |
| Qwen3.5-2B | prefill | fp16 | 1 | >= 16,384 | 8,192 | 8,192 |
| Qwen3.5-2B | generate | fp16 | 1 | >= 8,192 | >= 8,192 | >= 8,192 |
| Qwen3.5-2B | generate_static_compiled | fp16 | 1 | >= 8,192 | >= 8,192 | >= 8,192 |
| Qwen3.5-2B | generate | fp16 | 8 | >= 8,192 | 2,048 | 2,048 |
| Qwen3.5-2B | generate_static_compiled | fp16 | 8 | >= 8,192 | 2,048 | 2,048 |
| Qwen3.5-2B | train | fp16 | 1 | 1,024 | 1,024 | 1,024 |
| Qwen3.5-2B | train_ckpt | fp16 | 1 | 2,048 | 2,048 | 2,048 |
| Qwen3.5-4B | prefill | fp16 | 1 | >= 16,384 | 4,096 | 4,096 |
| Qwen3.5-4B | generate | fp16 | 1 | >= 8,192 | 2,048 | 2,048 |
| Qwen3.5-4B | generate_static_compiled | fp16 | 1 | >= 8,192 | 2,048 | 2,048 |
| Qwen3.5-4B | generate | fp16 | 8 | 2,048 | OOM | 2,048 |
| Qwen3.5-4B | generate_static_compiled | fp16 | 8 | 2,048 | OOM | OOM |
| Qwen3.5-4B | train | fp16 | 1 | OOM | OOM | OOM |

#### Tesla V100-SXM2-32GB: prefill (tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 1024 | 20,992 (2.13) | 10,175 (2.41) | 14,675 (2.36) | 1.43x | 0.70x | 0.23 GiB (10%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 4096 | 21,236 (2.34) | 4,064 (6.9) | 6,291 (6.27) | 3.38x | 0.30x | 3.93 GiB (63%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | 17,953 (2.62) | 1,851 (20.87) | 3,732 (18.55) | 4.81x | 0.21x | 15.93 GiB (86%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 16384 | 12,987 (3.17) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 1024 | 8,961 (5.98) | 6,288 (6.09) | 8,108 (6.05) | 1.11x | 0.90x | 0.07 GiB (1%) |
| Qwen2.5-3B | fp16 | 1 | 4096 | 9,181 (6.32) | 3,248 (8.51) | 5,232 (8.13) | 1.75x | 0.57x | 1.81 GiB (22%) |
| Qwen2.5-3B | fp16 | 1 | 8192 | 8,220 (6.78) | 1,712 (15.79) | 3,530 (14.46) | 2.33x | 0.43x | 7.68 GiB (53%) |
| Qwen2.5-3B | fp16 | 1 | 16384 | 6,388 (7.69) | OOM | OOM | only ours | - | - |
| gemma-2-2b-it | fp16 | 1 | 1024 | 10,382 (5.05) | 8,681 (5.11) | 10,133 (5.08) | 1.02x | 0.98x | 0.03 GiB (1%) |
| gemma-2-2b-it | fp16 | 1 | 4096 | 10,873 (5.57) | 6,602 (6.72) | 7,807 (6.45) | 1.39x | 0.72x | 0.88 GiB (14%) |
| gemma-2-2b-it | fp16 | 1 | 8192 | 10,207 (6.26) | 4,200 (10.96) | 5,551 (10.15) | 1.84x | 0.54x | 3.89 GiB (38%) |
| gemma-2-2b-it | fp16 | 1 | 16384 | 8,928 (7.64) | 2,138 (26.67) | 3,249 (23.92) | 2.75x | 0.36x | 16.28 GiB (68%) |
| Qwen3.5-2B | fp16 | 1 | 1024 | 9,342 (3.66) | 9,298 (3.67) | 9,487 (3.66) | 0.98x | 1.02x | 0.00 GiB (0%) |
| Qwen3.5-2B | fp16 | 1 | 4096 | 11,115 (4.04) | 8,897 (4.99) | 10,430 (4.72) | 1.07x | 0.94x | 0.68 GiB (14%) |
| Qwen3.5-2B | fp16 | 1 | 8192 | 11,000 (4.55) | 7,013 (8.81) | 9,581 (7.97) | 1.15x | 0.87x | 3.42 GiB (43%) |
| Qwen3.5-2B | fp16 | 1 | 16384 | 9,563 (5.57) | 4,506 (23.6) | 7,348 (20.66) | 1.30x | 0.77x | 15.09 GiB (73%) |
| Qwen3.5-9B | fp16 | 1 | 1024 | 3,301 (17.01) | 3,155 (17.02) | 3,282 (17.01) | 1.01x | 0.99x | 0.00 GiB (0%) |
| Qwen3.5-9B | fp16 | 1 | 512 | 2,990 (16.87) | 2,930 (16.87) | 2,943 (16.87) | 1.02x | 0.98x | 0.00 GiB (0%) |
| Qwen3.5-9B | fp16 | 1 | 2048 | 3,369 (17.28) | 3,064 (17.59) | 3,259 (17.41) | 1.03x | 0.97x | 0.13 GiB (1%) |
| Qwen3.5-9B | fp16 | 1 | 4096 | 3,385 (17.82) | 2,799 (19.61) | 3,187 (19.11) | 1.06x | 0.94x | 1.29 GiB (7%) |
| Qwen3.5-9B | fp16 | 1 | 8192 | 3,273 (18.9) | 2,205 (27.11) | 2,939 (25.55) | 1.11x | 0.90x | 6.65 GiB (26%) |
| Qwen3.5-9B | fp16 | 1 | 16384 | 3,091 (21.06) | OOM | OOM | only ours | - | - |
| Qwen2.5-14B | fp16 | 1 | 1024 | 2,458 (27.97) | 1,753 (28.33) | 2,193 (28.22) | 1.12x | 0.89x | 0.25 GiB (1%) |
| Qwen2.5-14B | fp16 | 1 | 512 | 2,311 (27.82) | 1,983 (27.91) | 2,273 (27.86) | 1.02x | 0.98x | 0.04 GiB (0%) |
| Qwen2.5-14B | fp16 | 1 | 2048 | 2,471 (28.27) | 1,349 (29.71) | 1,728 (29.41) | 1.43x | 0.70x | 1.14 GiB (4%) |
| Qwen2.5-14B | fp16 | 1 | 4096 | 2,307 (28.89) | OOM | OOM | only ours | - | - |
| Qwen2.5-14B | fp16 | 1 | 8192 | 2,024 (30.11) | OOM | OOM | only ours | - | - |
| Qwen2.5-14B | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 32768 | 8,351 (4.29) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 65536 | 4,845 (6.51) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 131072 | 2,662 (10.97) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 32768 | 4,413 (9.52) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 65536 | 2,838 (13.18) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 131072 | 1,665 (20.49) | - | - | only ours | - | - |
| Qwen3.5-9B | fp16 | 1 | 32768 | 2,773 (25.39) | OOM | OOM | only ours | - | - |
| Qwen2.5-14B | fp16 | 1 | 32768 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-4B | fp16 | 1 | 1024 | 4,574 (8.26) | 4,337 (8.27) | 4,585 (8.26) | 1.00x | 1.00x | 0.00 GiB (0%) |
| Qwen3.5-4B | fp16 | 1 | 4096 | 4,829 (9.03) | 3,698 (10.83) | 4,455 (10.33) | 1.08x | 0.92x | 1.30 GiB (13%) |
| Qwen3.5-4B | fp16 | 1 | 8192 | 4,622 (10.08) | 2,734 (18.29) | 3,966 (16.73) | 1.17x | 0.86x | 6.65 GiB (40%) |
| Qwen3.5-4B | fp16 | 1 | 16384 | 4,220 (12.17) | OOM | OOM | only ours | - | - |
| Llama-3.1-8B-Instruct | fp16 | 1 | 1024 | 4,473 (15.2) | 3,228 (15.48) | 4,014 (15.39) | 1.11x | 0.90x | 0.19 GiB (1%) |
| Llama-3.1-8B-Instruct | fp16 | 1 | 4096 | 4,278 (15.92) | 1,710 (20.44) | 2,585 (19.69) | 1.65x | 0.60x | 3.77 GiB (19%) |
| Llama-3.1-8B-Instruct | fp16 | 1 | 8192 | 3,827 (16.88) | OOM | OOM | only ours | - | - |
| Llama-3.1-8B-Instruct | fp16 | 1 | 16384 | 3,094 (18.79) | OOM | OOM | only ours | - | - |
| Mistral-7B-Instruct-v0.2 | fp16 | 1 | 1024 | 4,450 (13.74) | 3,232 (14.01) | 4,033 (13.92) | 1.10x | 0.91x | 0.18 GiB (1%) |
| Mistral-7B-Instruct-v0.2 | fp16 | 1 | 4096 | 4,276 (14.45) | 1,711 (18.97) | 2,582 (18.22) | 1.66x | 0.60x | 3.77 GiB (21%) |
| Mistral-7B-Instruct-v0.2 | fp16 | 1 | 8192 | 3,821 (15.41) | OOM | OOM | only ours | - | - |
| Mistral-7B-Instruct-v0.2 | fp16 | 1 | 16384 | 3,076 (17.32) | OOM | OOM | only ours | - | - |

#### Tesla V100-SXM2-32GB: generate (decode tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 2048 | 52.7 (2.2) | 47.1 (3.34) | 49.2 (3.16) | 1.07x | 0.93x | 0.96 GiB (30%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | 52.3 (2.62) | 44.2 (20.87) | 49.3 (18.55) | 1.06x | 0.94x | 15.93 GiB (86%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 8 | 2048 | 398.4 (3.17) | 216.6 (12.2) | 372.1 (10.84) | 1.07x | 0.93x | 7.67 GiB (71%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 8 | 8192 | 406.2 (6.51) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 2048 | 32.6 (6.09) | 28.3 (6.61) | 29.4 (6.49) | 1.11x | 0.90x | 0.40 GiB (6%) |
| Qwen2.5-3B | fp16 | 1 | 8192 | 32.7 (6.78) | 23.0 (15.79) | 29.5 (14.46) | 1.11x | 0.90x | 7.68 GiB (53%) |
| Qwen2.5-3B | fp16 | 8 | 2048 | 251.8 (7.69) | 125.9 (11.73) | 209.8 (10.87) | 1.20x | 0.83x | 3.18 GiB (29%) |
| Qwen2.5-3B | fp16 | 8 | 8192 | 247.4 (13.18) | OOM | OOM | only ours | - | - |
| gemma-2-2b-it | fp16 | 1 | 2048 | 32.2 (5.24) | 29.2 (5.5) | 29.7 (5.4) | 1.08x | 0.92x | 0.16 GiB (3%) |
| gemma-2-2b-it | fp16 | 1 | 8192 | 31.9 (6.26) | 26.7 (10.96) | 29.7 (10.15) | 1.07x | 0.93x | 3.89 GiB (38%) |
| gemma-2-2b-it | fp16 | 8 | 2048 | 248.2 (7.64) | 123.0 (9.68) | 204.2 (9.04) | 1.22x | 0.82x | 1.40 GiB (15%) |
| gemma-2-2b-it | fp16 | 8 | 8192 | 162.4 (15.94) | OOM | OOM | only ours | - | - |
| Qwen3.5-2B | fp16 | 1 | 2048 | 27.8 (3.79) | 27.9 (3.96) | 28.3 (3.87) | 0.98x | 1.02x | 0.08 GiB (2%) |
| Qwen3.5-2B | fp16 | 1 | 8192 | 28.4 (4.55) | 27.6 (8.82) | 27.1 (7.97) | 1.03x | 0.97x | 3.42 GiB (43%) |
| Qwen3.5-2B | fp16 | 8 | 2048 | 218.0 (5.71) | 205.4 (6.99) | 205.5 (6.35) | 1.06x | 0.94x | 0.64 GiB (10%) |
| Qwen3.5-2B | fp16 | 8 | 8192 | 210.8 (11.83) | OOM | OOM | only ours | - | - |
| Qwen3.5-9B | fp16 | 1 | 512 | 19.6 (16.87) | 19.4 (16.87) | 20.5 (16.87) | 0.96x | 1.05x | 0.00 GiB (0%) |
| Qwen3.5-9B | fp16 | 1 | 2048 | 19.5 (17.28) | 18.8 (17.59) | 19.9 (17.41) | 0.98x | 1.02x | 0.13 GiB (1%) |
| Qwen3.5-9B | fp16 | 1 | 8192 | 19.2 (18.9) | 18.5 (27.11) | 19.9 (25.55) | 0.96x | 1.04x | 6.65 GiB (26%) |
| Qwen3.5-9B | fp16 | 8 | 512 | 161.9 (18.17) | 147.1 (18.17) | 156.8 (18.17) | 1.03x | 0.97x | 0.00 GiB (0%) |
| Qwen3.5-9B | fp16 | 8 | 2048 | 157.0 (21.42) | 110.1 (23.84) | 136.9 (22.51) | 1.15x | 0.87x | 1.09 GiB (5%) |
| Qwen3.5-9B | fp16 | 8 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-14B | fp16 | 1 | 512 | 19.7 (27.82) | 17.5 (27.91) | 18.7 (27.86) | 1.05x | 0.95x | 0.04 GiB (0%) |
| Qwen2.5-14B | fp16 | 1 | 2048 | 19.1 (28.27) | 13.9 (29.71) | 16.5 (29.41) | 1.16x | 0.86x | 1.14 GiB (4%) |
| Qwen2.5-14B | fp16 | 1 | 8192 | 16.7 (30.11) | OOM | OOM | only ours | - | - |
| Qwen2.5-14B | fp16 | 8 | 512 | 128.3 (28.89) | 77.8 (29.62) | 102.9 (29.28) | 1.25x | 0.80x | 0.39 GiB (1%) |
| Qwen2.5-14B | fp16 | 8 | 2048 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-14B | fp16 | 8 | 8192 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 16384 | 53.6 (3.17) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 32768 | 52.9 (4.29) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 65536 | 53.1 (6.51) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 16384 | 32.4 (7.69) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 32768 | 32.0 (9.52) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 65536 | 32.4 (13.18) | - | - | only ours | - | - |
| Qwen3.5-4B | fp16 | 1 | 2048 | 21.5 (8.51) | 14.4 (8.83) | 19.6 (8.65) | 1.10x | 0.91x | 0.14 GiB (2%) |
| Qwen3.5-4B | fp16 | 1 | 8192 | 20.9 (10.08) | 18.7 (18.29) | 19.2 (16.73) | 1.09x | 0.92x | 6.65 GiB (40%) |
| Qwen3.5-4B | fp16 | 8 | 2048 | 153.4 (12.52) | 131.2 (14.95) | 149.9 (13.62) | 1.02x | 0.98x | 1.10 GiB (8%) |
| Qwen3.5-4B | fp16 | 8 | 8192 | 149.5 (25.09) | OOM | OOM | only ours | - | - |
| Llama-3.1-8B-Instruct | fp16 | 1 | 2048 | 32.9 (15.44) | 24.1 (16.56) | 25.0 (16.32) | 1.32x | 0.76x | 0.88 GiB (5%) |
| Llama-3.1-8B-Instruct | fp16 | 1 | 8192 | 28.5 (16.88) | OOM | OOM | only ours | - | - |
| Llama-3.1-8B-Instruct | fp16 | 8 | 2048 | 179.9 (18.79) | 64.3 (27.61) | 105.9 (25.79) | 1.70x | 0.59x | 7.00 GiB (27%) |
| Llama-3.1-8B-Instruct | fp16 | 8 | 8192 | OOM | OOM | OOM | - | - | - |
| Mistral-7B-Instruct-v0.2 | fp16 | 1 | 2048 | 34.3 (13.97) | 24.7 (15.09) | 29.5 (14.85) | 1.16x | 0.86x | 0.88 GiB (6%) |
| Mistral-7B-Instruct-v0.2 | fp16 | 1 | 8192 | 29.3 (15.41) | OOM | OOM | only ours | - | - |
| Mistral-7B-Instruct-v0.2 | fp16 | 8 | 2048 | 184.8 (17.32) | 64.9 (26.15) | 107.2 (24.32) | 1.72x | 0.58x | 7.00 GiB (29%) |
| Mistral-7B-Instruct-v0.2 | fp16 | 8 | 8192 | OOM | OOM | OOM | - | - | - |

#### Tesla V100-SXM2-32GB: generate_static_compiled (decode tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 2048 | 164.5 (2.21) | 86.6 (3.35) | 135.1 (3.23) | 1.22x | 0.82x | 1.02 GiB (32%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | 146.6 (2.62) | 36.6 (20.88) | 91.3 (18.8) | 1.61x | 0.62x | 16.18 GiB (86%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 8 | 2048 | 967.3 (3.19) | 342.1 (12.23) | 500.3 (11.38) | 1.93x | 0.52x | 8.19 GiB (72%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 8 | 8192 | 778.7 (6.54) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 2048 | 73.3 (6.12) | 50.7 (6.63) | 66.2 (6.54) | 1.11x | 0.90x | 0.42 GiB (6%) |
| Qwen2.5-3B | fp16 | 1 | 8192 | 71.4 (6.78) | 25.2 (15.79) | 46.4 (14.59) | 1.54x | 0.65x | 7.81 GiB (54%) |
| Qwen2.5-3B | fp16 | 8 | 2048 | 476.6 (7.73) | 265.3 (11.76) | 275.8 (11.17) | 1.73x | 0.58x | 3.44 GiB (31%) |
| Qwen2.5-3B | fp16 | 8 | 8192 | 402.3 (13.21) | OOM | OOM | only ours | - | - |
| gemma-2-2b-it | fp16 | 1 | 2048 | 81.8 (5.24) | 50.3 (5.51) | 74.5 (5.44) | 1.10x | 0.91x | 0.20 GiB (4%) |
| gemma-2-2b-it | fp16 | 1 | 8192 | 71.6 (6.06) | 27.1 (10.76) | 54.6 (10.02) | 1.31x | 0.76x | 3.96 GiB (40%) |
| gemma-2-2b-it | fp16 | 8 | 2048 | 513.0 (7.75) | 288.8 (9.79) | 314.8 (9.29) | 1.63x | 0.61x | 1.54 GiB (17%) |
| gemma-2-2b-it | fp16 | 8 | 8192 | 284.0 (14.37) | OOM | OOM | only ours | - | - |
| Qwen3.5-2B | fp16 | 1 | 2048 | 103.2 (3.79) | 86.6 (3.96) | 101.6 (3.89) | 1.02x | 0.98x | 0.10 GiB (3%) |
| Qwen3.5-2B | fp16 | 1 | 8192 | 100.7 (4.57) | 58.5 (8.82) | 87.8 (8.04) | 1.15x | 0.87x | 3.47 GiB (43%) |
| Qwen3.5-2B | fp16 | 8 | 2048 | 593.5 (5.76) | 497.4 (7.0) | 512.2 (6.5) | 1.16x | 0.86x | 0.74 GiB (11%) |
| Qwen3.5-2B | fp16 | 8 | 8192 | 558.7 (11.98) | OOM | OOM | only ours | - | - |
| Qwen3.5-9B | fp16 | 1 | 512 | 35.7 (16.88) | 34.9 (16.88) | 35.7 (16.88) | 1.00x | 1.00x | 0.00 GiB (0%) |
| Qwen3.5-9B | fp16 | 1 | 2048 | 35.6 (17.29) | 32.7 (17.6) | 34.8 (17.45) | 1.02x | 0.98x | 0.16 GiB (1%) |
| Qwen3.5-9B | fp16 | 1 | 8192 | 35.0 (18.94) | 25.4 (27.12) | 31.2 (25.68) | 1.12x | 0.89x | 6.74 GiB (26%) |
| Qwen3.5-9B | fp16 | 8 | 512 | 213.4 (18.23) | 203.1 (18.23) | 203.6 (18.24) | 1.05x | 0.95x | 0.00 GiB (0%) |
| Qwen3.5-9B | fp16 | 8 | 2048 | 210.0 (21.53) | 181.1 (23.87) | 183.1 (22.82) | 1.15x | 0.87x | 1.29 GiB (6%) |
| Qwen3.5-9B | fp16 | 8 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-14B | fp16 | 1 | 512 | 21.0 (27.84) | 19.9 (27.93) | 20.4 (27.91) | 1.03x | 0.97x | 0.07 GiB (0%) |
| Qwen2.5-14B | fp16 | 1 | 2048 | 20.8 (28.3) | 17.1 (29.73) | 18.5 (29.52) | 1.12x | 0.89x | 1.22 GiB (4%) |
| Qwen2.5-14B | fp16 | 1 | 8192 | 19.7 (30.14) | OOM | OOM | only ours | - | - |
| Qwen2.5-14B | fp16 | 8 | 512 | 145.4 (29.07) | 115.1 (29.81) | 117.6 (29.64) | 1.24x | 0.81x | 0.57 GiB (2%) |
| Qwen2.5-14B | fp16 | 8 | 2048 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-14B | fp16 | 8 | 8192 | OOM | OOM | OOM | - | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 16384 | 133.8 (3.18) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 32768 | 122.8 (4.29) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 65536 | 110.7 (6.52) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 16384 | 68.0 (7.69) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 32768 | 62.4 (9.52) | - | - | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 65536 | 53.2 (13.18) | - | - | only ours | - | - |
| Qwen3.5-4B | fp16 | 1 | 2048 | 55.6 (8.52) | 48.3 (8.83) | 53.1 (8.68) | 1.05x | 0.96x | 0.16 GiB (2%) |
| Qwen3.5-4B | fp16 | 1 | 8192 | 54.4 (10.12) | 34.2 (18.29) | 45.1 (16.86) | 1.21x | 0.83x | 6.74 GiB (40%) |
| Qwen3.5-4B | fp16 | 8 | 2048 | 312.7 (12.63) | 253.8 (14.98) | 258.1 (13.92) | 1.21x | 0.83x | 1.29 GiB (9%) |
| Qwen3.5-4B | fp16 | 8 | 8192 | 287.9 (25.38) | OOM | OOM | only ours | - | - |
| Llama-3.1-8B-Instruct | fp16 | 1 | 2048 | 36.5 (15.46) | 29.3 (16.58) | 32.2 (16.4) | 1.13x | 0.88x | 0.94 GiB (6%) |
| Llama-3.1-8B-Instruct | fp16 | 1 | 8192 | 34.4 (16.89) | OOM | OOM | only ours | - | - |
| Llama-3.1-8B-Instruct | fp16 | 8 | 2048 | 244.2 (18.91) | 128.7 (27.74) | 141.8 (26.43) | 1.72x | 0.58x | 7.52 GiB (28%) |
| Llama-3.1-8B-Instruct | fp16 | 8 | 8192 | OOM | OOM | OOM | - | - | - |
| Mistral-7B-Instruct-v0.2 | fp16 | 1 | 2048 | 37.8 (13.99) | 30.2 (15.11) | 33.4 (14.93) | 1.13x | 0.88x | 0.94 GiB (6%) |
| Mistral-7B-Instruct-v0.2 | fp16 | 1 | 8192 | 35.4 (15.42) | OOM | OOM | only ours | - | - |
| Mistral-7B-Instruct-v0.2 | fp16 | 8 | 2048 | 253.2 (17.44) | 131.0 (26.27) | 144.9 (24.96) | 1.75x | 0.57x | 7.52 GiB (30%) |
| Mistral-7B-Instruct-v0.2 | fp16 | 8 | 8192 | OOM | OOM | OOM | - | - | - |

#### Tesla V100-SXM2-32GB: train (tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 1024 | 6,781 (6.23) | 4,180 (9.25) | 5,077 (10.37) | 1.34x | 0.75x | 3.02 GiB (33%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 2048 | 7,206 (7.95) | 3,120 (20.27) | 3,888 (24.76) | 1.85x | 0.54x | 12.32 GiB (61%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 4096 | 6,498 (11.53) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | 5,144 (18.68) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 512 | 6,538 (7.85) | 6,160 (8.12) | 5,410 (12.2) | 1.06x | 0.94x | 0.27 GiB (3%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 1024 | 6,981 (11.35) | 6,008 (11.96) | 4,679 (28.27) | 1.16x | 0.86x | 0.61 GiB (5%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 2048 | 7,005 (18.3) | 5,098 (19.88) | OOM | 1.37x | 0.73x | 1.58 GiB (8%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 1024 | 3,160 (14.84) | 2,395 (17.62) | 2,896 (18.32) | 1.09x | 0.92x | 2.78 GiB (16%) |
| Qwen2.5-3B | fp16 | 1 | 2048 | 3,306 (20.05) | OOM | OOM | only ours | - | - |
| Qwen2.5-3B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| gemma-2-2b-it | fp16 | 1 | 1024 | 3,032 (15.22) | 2,693 (16.37) | 3,083 (16.85) | 0.98x | 1.02x | 1.15 GiB (7%) |
| gemma-2-2b-it | fp16 | 1 | 2048 | 3,111 (21.45) | 2,481 (25.31) | 3,041 (27.96) | 1.02x | 0.98x | 3.86 GiB (15%) |
| gemma-2-2b-it | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| gemma-2-2b-it | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 1024 | 1,855 (11.49) | 1,806 (11.76) | 1,834 (11.78) | 1.01x | 0.99x | 0.27 GiB (2%) |
| Qwen3.5-2B | fp16 | 1 | 2048 | 1,732 (18.51) | 1,708 (19.43) | 1,726 (19.66) | 1.00x | 1.00x | 0.92 GiB (5%) |
| Qwen3.5-2B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-4B | fp16 | 1 | 1024 | 1,015 (24.41) | 1,012 (25.12) | 1,027 (25.19) | 0.99x | 1.01x | 0.71 GiB (3%) |
| Qwen3.5-4B | fp16 | 1 | 2048 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-4B | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-4B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Mistral-7B-Instruct-v0.2 | fp16 | 1 | 1024 | OOM | OOM | OOM | - | - | - |
| Mistral-7B-Instruct-v0.2 | fp16 | 1 | 2048 | OOM | OOM | OOM | - | - | - |
| Mistral-7B-Instruct-v0.2 | fp16 | 1 | 4096 | OOM | OOM | OOM | - | - | - |
| Mistral-7B-Instruct-v0.2 | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |

#### Tesla V100-SXM2-32GB: train_ckpt (tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 2048 | 5,694 (5.56) | 2,171 (6.85) | 2,760 (6.58) | 2.06x | 0.48x | 1.02 GiB (16%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 4096 | 5,165 (6.0) | 1,327 (12.94) | 1,897 (11.92) | 2.72x | 0.37x | 5.92 GiB (50%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 8192 | 4,058 (7.68) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 16384 | 2,694 (11.36) | OOM | OOM | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 1024 | 5,460 (6.0) | 4,673 (6.0) | 3,511 (6.65) | 1.17x | 0.86x | 0.00 GiB (0%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 2048 | 5,427 (7.68) | 3,954 (7.7) | 2,209 (12.05) | 1.37x | 0.73x | 0.02 GiB (0%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 4096 | 4,837 (11.36) | 2,842 (11.42) | OOM | 1.70x | 0.59x | 0.06 GiB (1%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 4 | 8192 | 3,879 (18.72) | 1,799 (18.97) | OOM | 2.16x | 0.46x | 0.25 GiB (1%) |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 32768 | 1,616 (18.73) | - | - | only ours | - | - |
| TinyLlama-1.1B-Chat-v1.0 | fp16 | 1 | 65536 | OOM | - | - | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 2048 | 2,615 (12.86) | 1,488 (12.86) | 1,964 (12.87) | 1.33x | 0.75x | 0.00 GiB (0%) |
| Qwen2.5-3B | fp16 | 1 | 4096 | 2,412 (16.57) | 994 (16.57) | 1,552 (16.61) | 1.55x | 0.64x | 0.00 GiB (0%) |
| Qwen2.5-3B | fp16 | 1 | 8192 | 1,978 (24.12) | 544 (26.63) | OOM | 3.64x | 0.28x | 2.51 GiB (9%) |
| Qwen2.5-3B | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 32768 | OOM | - | - | - | - | - |
| Qwen2.5-3B | fp16 | 1 | 65536 | OOM | - | - | - | - | - |
| gemma-2-2b-it | fp16 | 1 | 2048 | 2,479 (15.4) | 1,866 (15.4) | 2,307 (15.42) | 1.07x | 0.93x | 0.00 GiB (0%) |
| gemma-2-2b-it | fp16 | 1 | 4096 | 2,314 (22.48) | 1,606 (22.5) | 2,028 (22.55) | 1.14x | 0.88x | 0.02 GiB (0%) |
| gemma-2-2b-it | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| gemma-2-2b-it | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-2B | fp16 | 1 | 2048 | 1,136 (10.23) | 1,169 (10.23) | 1,221 (10.24) | 0.93x | 1.07x | 0.00 GiB (0%) |
| Qwen3.5-2B | fp16 | 1 | 4096 | 912 (16.12) | 891 (16.12) | 910 (16.15) | 1.00x | 1.00x | 0.00 GiB (0%) |
| Qwen3.5-2B | fp16 | 1 | 8192 | 608 (27.89) | 561 (27.89) | OOM | 1.08x | 0.92x | 0.00 GiB (0%) |
| Qwen3.5-2B | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-4B | fp16 | 1 | 2048 | 620 (16.7) | 596 (16.69) | 620 (16.7) | 1.00x | 1.00x | -0.01 GiB (-0%) |
| Qwen3.5-4B | fp16 | 1 | 4096 | 424 (22.71) | 397 (22.71) | 422 (22.74) | 1.00x | 1.00x | 0.00 GiB (0%) |
| Qwen3.5-4B | fp16 | 1 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen3.5-4B | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |

#### Tesla V100-SXM2-32GB: train_lora_ckpt (tok s, peak GiB)

| model | dtype | B | N | flash_attn_legacy | sdpa | eager | speedup | time | memory saved |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-3B (LoRA r16) | fp16 | 1 | 2048 | 2,594 (9.73) | 1,477 (9.73) | 1,947 (9.74) | 1.33x | 0.75x | 0.00 GiB (0%) |
| Qwen2.5-3B (LoRA r16) | fp16 | 1 | 4096 | 2,358 (13.5) | 985 (13.5) | 1,529 (13.53) | 1.54x | 0.65x | 0.00 GiB (0%) |
| Qwen2.5-3B (LoRA r16) | fp16 | 1 | 8192 | 1,914 (21.05) | 527 (23.46) | 1,039 (21.49) | 1.84x | 0.54x | 0.44 GiB (2%) |
| Qwen2.5-3B (LoRA r16) | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B (LoRA r16) | fp16 | 1 | 32768 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B (LoRA r16) | fp16 | 4 | 2048 | 2,213 (21.05) | 1,860 (21.07) | 1,530 (21.08) | 1.19x | 0.84x | 0.02 GiB (0%) |
| Qwen2.5-3B (LoRA r16) | fp16 | 4 | 4096 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B (LoRA r16) | fp16 | 4 | 8192 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B (LoRA r16) | fp16 | 4 | 16384 | OOM | OOM | OOM | - | - | - |
| Qwen2.5-3B (LoRA r16) | fp16 | 4 | 32768 | OOM | OOM | OOM | - | - | - |
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
| Llama-3.1-8B-Instruct (LoRA r16) | fp16 | 1 | 2048 | 1,284 (18.59) | 770 (18.59) | 996 (18.6) | 1.29x | 0.78x | 0.00 GiB (0%) |
| Llama-3.1-8B-Instruct (LoRA r16) | fp16 | 1 | 4096 | 1,176 (22.06) | 530 (24.5) | 793 (23.44) | 1.48x | 0.67x | 1.38 GiB (6%) |
| Llama-3.1-8B-Instruct (LoRA r16) | fp16 | 1 | 8192 | 978 (29.0) | OOM | OOM | only ours | - | - |
| Llama-3.1-8B-Instruct (LoRA r16) | fp16 | 1 | 16384 | OOM | OOM | OOM | - | - | - |
| Llama-3.1-8B-Instruct (LoRA r16) | fp16 | 1 | 32768 | OOM | OOM | OOM | - | - | - |
| Llama-3.1-8B-Instruct (LoRA r16) | fp16 | 4 | 2048 | 1,071 (28.99) | 906 (29.01) | OOM | 1.18x | 0.85x | 0.02 GiB (0%) |
| Llama-3.1-8B-Instruct (LoRA r16) | fp16 | 4 | 4096 | OOM | OOM | OOM | - | - | - |
| Llama-3.1-8B-Instruct (LoRA r16) | fp16 | 4 | 8192 | OOM | OOM | OOM | - | - | - |
| Llama-3.1-8B-Instruct (LoRA r16) | fp16 | 4 | 16384 | OOM | OOM | OOM | - | - | - |
| Llama-3.1-8B-Instruct (LoRA r16) | fp16 | 4 | 32768 | OOM | OOM | OOM | - | - | - |

#### Tesla V100-SXM2-32GB: longest N that fits (>= N: no longer N measured)

| model | mode | dtype | B | flash_attn_legacy | sdpa | eager |
|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | prefill | fp16 | 1 | >= 131,072 | 8,192 | 8,192 |
| TinyLlama-1.1B-Chat-v1.0 | generate | fp16 | 1 | >= 65,536 | >= 8,192 | >= 8,192 |
| TinyLlama-1.1B-Chat-v1.0 | generate_static_compiled | fp16 | 1 | >= 65,536 | >= 8,192 | >= 8,192 |
| TinyLlama-1.1B-Chat-v1.0 | generate | fp16 | 8 | >= 8,192 | 2,048 | 2,048 |
| TinyLlama-1.1B-Chat-v1.0 | generate_static_compiled | fp16 | 8 | >= 8,192 | 2,048 | 2,048 |
| TinyLlama-1.1B-Chat-v1.0 | train | fp16 | 1 | >= 8,192 | 2,048 | 2,048 |
| TinyLlama-1.1B-Chat-v1.0 | train | fp16 | 4 | 2,048 | 2,048 | 1,024 |
| TinyLlama-1.1B-Chat-v1.0 | train_ckpt | fp16 | 1 | 32,768 | 4,096 | 4,096 |
| TinyLlama-1.1B-Chat-v1.0 | train_ckpt | fp16 | 4 | >= 8,192 | >= 8,192 | 2,048 |
| Qwen2.5-3B | prefill | fp16 | 1 | >= 131,072 | 8,192 | 8,192 |
| Qwen2.5-3B | generate | fp16 | 1 | >= 65,536 | >= 8,192 | >= 8,192 |
| Qwen2.5-3B | generate_static_compiled | fp16 | 1 | >= 65,536 | >= 8,192 | >= 8,192 |
| Qwen2.5-3B | generate | fp16 | 8 | >= 8,192 | 2,048 | 2,048 |
| Qwen2.5-3B | generate_static_compiled | fp16 | 8 | >= 8,192 | 2,048 | 2,048 |
| Qwen2.5-3B | train | fp16 | 1 | 2,048 | 1,024 | 1,024 |
| Qwen2.5-3B | train_ckpt | fp16 | 1 | 8,192 | 8,192 | 4,096 |
| Qwen2.5-3B (LoRA r16) | train_lora_ckpt | fp16 | 1 | 8,192 | 8,192 | 8,192 |
| Qwen2.5-3B (LoRA r16) | train_lora_ckpt | fp16 | 4 | 2,048 | 2,048 | 2,048 |
| gemma-2-2b-it | prefill | fp16 | 1 | >= 16,384 | >= 16,384 | >= 16,384 |
| gemma-2-2b-it | generate | fp16 | 1 | >= 8,192 | >= 8,192 | >= 8,192 |
| gemma-2-2b-it | generate_static_compiled | fp16 | 1 | >= 8,192 | >= 8,192 | >= 8,192 |
| gemma-2-2b-it | generate | fp16 | 8 | >= 8,192 | 2,048 | 2,048 |
| gemma-2-2b-it | generate_static_compiled | fp16 | 8 | >= 8,192 | 2,048 | 2,048 |
| gemma-2-2b-it | train | fp16 | 1 | 2,048 | 2,048 | 2,048 |
| gemma-2-2b-it | train_ckpt | fp16 | 1 | 4,096 | 4,096 | 4,096 |
| Qwen3.5-2B | prefill | fp16 | 1 | >= 16,384 | >= 16,384 | >= 16,384 |
| Qwen3.5-2B | generate | fp16 | 1 | >= 8,192 | >= 8,192 | >= 8,192 |
| Qwen3.5-2B | generate_static_compiled | fp16 | 1 | >= 8,192 | >= 8,192 | >= 8,192 |
| Qwen3.5-2B | generate | fp16 | 8 | >= 8,192 | 2,048 | 2,048 |
| Qwen3.5-2B | generate_static_compiled | fp16 | 8 | >= 8,192 | 2,048 | 2,048 |
| Qwen3.5-2B | train | fp16 | 1 | 2,048 | 2,048 | 2,048 |
| Qwen3.5-2B | train_ckpt | fp16 | 1 | 8,192 | 8,192 | 4,096 |
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
| Qwen2.5-14B (LoRA r16) | train_lora_ckpt | fp16 | 1 | OOM | OOM | OOM |
| Qwen2.5-14B (LoRA r16) | train_lora_ckpt | fp16 | 4 | OOM | OOM | OOM |
| Qwen3.5-4B | prefill | fp16 | 1 | >= 16,384 | 8,192 | 8,192 |
| Qwen3.5-4B | generate | fp16 | 1 | >= 8,192 | >= 8,192 | >= 8,192 |
| Qwen3.5-4B | generate_static_compiled | fp16 | 1 | >= 8,192 | >= 8,192 | >= 8,192 |
| Qwen3.5-4B | generate | fp16 | 8 | >= 8,192 | 2,048 | 2,048 |
| Qwen3.5-4B | generate_static_compiled | fp16 | 8 | >= 8,192 | 2,048 | 2,048 |
| Qwen3.5-4B | train | fp16 | 1 | 1,024 | 1,024 | 1,024 |
| Qwen3.5-4B | train_ckpt | fp16 | 1 | 4,096 | 4,096 | 4,096 |
| Llama-3.1-8B-Instruct | prefill | fp16 | 1 | >= 16,384 | 4,096 | 4,096 |
| Llama-3.1-8B-Instruct | generate | fp16 | 1 | >= 8,192 | 2,048 | 2,048 |
| Llama-3.1-8B-Instruct | generate_static_compiled | fp16 | 1 | >= 8,192 | 2,048 | 2,048 |
| Llama-3.1-8B-Instruct | generate | fp16 | 8 | 2,048 | 2,048 | 2,048 |
| Llama-3.1-8B-Instruct | generate_static_compiled | fp16 | 8 | 2,048 | 2,048 | 2,048 |
| Llama-3.1-8B-Instruct (LoRA r16) | train_lora_ckpt | fp16 | 1 | 8,192 | 4,096 | 4,096 |
| Llama-3.1-8B-Instruct (LoRA r16) | train_lora_ckpt | fp16 | 4 | 2,048 | 2,048 | OOM |
| Mistral-7B-Instruct-v0.2 | prefill | fp16 | 1 | >= 16,384 | 4,096 | 4,096 |
| Mistral-7B-Instruct-v0.2 | generate | fp16 | 1 | >= 8,192 | 2,048 | 2,048 |
| Mistral-7B-Instruct-v0.2 | generate_static_compiled | fp16 | 1 | >= 8,192 | 2,048 | 2,048 |
| Mistral-7B-Instruct-v0.2 | generate | fp16 | 8 | 2,048 | 2,048 | 2,048 |
| Mistral-7B-Instruct-v0.2 | generate_static_compiled | fp16 | 8 | 2,048 | 2,048 | 2,048 |
| Mistral-7B-Instruct-v0.2 | train | fp16 | 1 | OOM | OOM | OOM |
