# Serving: transformers' continuous batching

`python benchmarks/serving.py --model <model> --requests R --prompt-len P --new-tokens N` (PLAN P6.9). transformers' own
serving path, `model.generate_batch` (the engine behind `init_continuous_batching` and `transformers serve`), on its paged KV
cache: decode-only batches call `flash_attn_with_kvcache` with the block table (the new keys appended by our kernel), the other
batches our varlen kernel on the keys `cache.update` gathers out of the cache.

- Workload: R concurrent requests, each a prompt of random tokens (lengths P +-25%, so prefill and decode batches mix) and N
  greedy new tokens (no EOS); fp16 weights. transformers chooses the batches and the paging (pages of 256 tokens); its paged
  cache takes 70% of the free memory for every implementation (`--max-memory-percent`; transformers' default 90% leaves too
  little for the CUDA graph pools and cuBLAS on the 16 GB cards).
- Each implementation and CUDA graphs off / on in its own process; a persistent manager reruns the workload until a run is no
  longer 5% faster than the one before (every new batch shape captures a graph: long prompts take several runs; Qwen2.5-3B,
  8 x 8192 on the T4: 60, 17, 7.2, 5.2, 4.2, 4.0 s), the fastest run is reported. Cells: generated tokens/s (wall s).
- `ours / SDPA`: the faster of our two cells over the faster of SDPA's. `eager`: transformers' `paged|eager` materializes the
  scores of a whole batch (up to 8192 query tokens x the cache) and runs out of memory on every workload and GPU (one request
  failing takes its batch down: every request fails).

Where the time goes (Qwen2.5-3B, 8 x 8192, T4, graphs off, GPU time of one run): GEMMs 57% (decode steps read the weights:
bandwidth bound), our attention kernels 32% (decode 25%, varlen prefill 5%), the gather of the cache in `cache.update` 2.5%,
the rest 9%. A prefill reading K / V from the paged cache (no gather) would save at most those 2.5%: not done.

Kernels at 5d1da96 (the decode steps here are one token per request, at most 8 query rows per KV head: the decode kernel,
which the cache-forward fix after 5d1da96 does not touch); torch 2.7.1, transformers 5.20.2; the V100 is the SXM2 32 GB box of `models.md`. On the 1080 Ti the 3B 8 x 8192
workload with SDPA and graphs on was stopped by the 2-hour limit of the run (the eager cells after it did not run).

#### Tesla V100-SXM2-32GB: generated tokens/s (wall s)

| model | requests | prompt | new | ours, graphs off | ours, graphs on | SDPA, graphs off | SDPA, graphs on | ours / SDPA | eager |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 64 | 512 | 128 | 1,812 (4.5) | 4,624 (1.8) | 438 (18.7) | 345 (23.7) | 10.6x | OOM |
| Qwen2.5-3B | 32 | 1024 | 128 | 675 (6.1) | 1,094 (3.7) | 167 (24.6) | 126 (32.5) | 6.6x | OOM |
| Qwen2.5-3B | 8 | 8192 | 64 | 161 (3.2) | 250 (2.0) | 23 (22.4) | 20 (25.7) | 10.9x | OOM |

#### Tesla T4: generated tokens/s (wall s)

| model | requests | prompt | new | ours, graphs off | ours, graphs on | SDPA, graphs off | SDPA, graphs on | ours / SDPA | eager |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 64 | 512 | 128 | 1,623 (5.0) | 3,210 (2.5) | 392 (20.9) | 310 (26.4) | 8.2x | OOM |
| Qwen2.5-3B | 32 | 1024 | 128 | 499 (8.2) | 508 (8.1) | 103 (39.8) | 77 (52.9) | 4.9x | OOM |
| Qwen2.5-3B | 8 | 8192 | 64 | 124 (4.1) | 126 (4.0) | 15 (34.4) | 12 (41.2) | 8.4x | OOM |

#### Tesla P100-PCIE-16GB: generated tokens/s (wall s)

| model | requests | prompt | new | ours, graphs off | ours, graphs on | SDPA, graphs off | SDPA, graphs on | ours / SDPA | eager |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 64 | 512 | 128 | 1,798 (4.6) | 1,865 (4.4) | 162 (50.5) | 125 (65.3) | 11.5x | OOM |
| Qwen2.5-3B | 32 | 1024 | 128 | 353 (11.6) | 356 (11.5) | 52 (79.4) | 39 (105.7) | 6.8x | OOM |
| Qwen2.5-3B | 8 | 8192 | 64 | 69 (7.4) | 69 (7.4) | 7 (73.5) | 6 (81.3) | 9.9x | OOM |

#### NVIDIA GeForce GTX 1080 Ti: generated tokens/s (wall s)

| model | requests | prompt | new | ours, graphs off | ours, graphs on | SDPA, graphs off | SDPA, graphs on | ours / SDPA | eager |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 64 | 512 | 128 | 1,042 (7.9) | 1,569 (5.2) | 26 (309.6) | 20 (407.8) | 60.3x | OOM |
| Qwen2.5-3B | 32 | 1024 | 128 | 267 (15.3) | 280 (14.6) | 9 (477.3) | 4 (1001.3) | 31.1x | OOM |
| Qwen2.5-3B | 8 | 8192 | 64 | 56 (9.2) | 56 (9.2) | 0.5 (1073.0) | stopped (2 h limit) | 117x | not run |

#### NVIDIA GeForce RTX 2080 Ti: generated tokens/s (wall s)

| model | requests | prompt | new | ours, graphs off | ours, graphs on | SDPA, graphs off | SDPA, graphs on | ours / SDPA | eager |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 64 | 512 | 128 | 2,226 (3.7) | 5,120 (1.6) | 578 (14.2) | 451 (18.1) | 8.9x | OOM |
| Qwen2.5-3B | 32 | 1024 | 128 | 765 (5.3) | 992 (4.1) | 179 (22.9) | 137 (29.9) | 5.5x | OOM |
| Qwen2.5-3B | 8 | 8192 | 64 | 115 (4.5) | 130 (3.9) | 8 (61.2) | 7 (72.4) | 16.2x | OOM |
