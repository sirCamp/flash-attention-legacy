# flash-attention-legacy — Analysis and work plan

Status: **draft v2, updated after the independent review**; each item carries its own status (done / in progress / open).

Sources: analysis on Tesla V100 (sm_70, CUDA 12.6, torch 2.7.1, transformers 5.20.2):
- [`docs/analysis/hf/report.md`](analysis/hf/report.md) — contract of the HF integration (transformers source with file:line references, proofs on CPU).
- [`docs/analysis/models/report.md`](analysis/models/report.md) — 10 real models measured on the V100, with reproduction scripts.
- [`docs/analysis/kernels/report.md`](analysis/kernels/report.md) — registers, spills, shared memory and occupancy from `ptxas -v` and SASS for sm_60, 61, 70 and 75.
- Review of plan v1: results integrated below, items marked with **[R#]**.

Size: **S** = a few hours, **M** = 1–3 days, **L** = more than 3 days.

---

## 1. Summary of results

**The kernel is correct.**
- Forward: 570 configurations, backward: 72, no failures; error equal to SDPA FP16.
- 377 unit tests pass.
- Isolating the attention, the gradients are like those of SDPA FP16: 1-cos against the FP32 gold of 0.879/0.887, versus 0.866/0.875.

**Silent bugs, all in the HF integration.**

| # | Bug | Affected | Status |
|---|---|---|---|
| B1 | Prefill with `StaticCache`: the kernel includes the empty cache slots, degenerate output (`"is,,,,,,,"`) | everything with `cache_implementation="static"`; to be verified on hybrid and sliding caches (Gemma 2/3), offloaded, assisted decoding, beam search | **confirmed on V100** |
| B2 | `softcap` ignored, both in the kernel and in the fallback | Gemma 2 | confirmed on V100 |
| B3 | `s_aux` sinks ignored | gpt-oss | confirmed on a reduced model |
| B4 | Packed sequences with `use_cache=True` or `cu_seq_lens` treated as a single sequence | training with packing | confirmed on V100 |
| B5 | `cu_seq_lens` ignored **both in the kernel and in the SDPA fallback**: attention crosses images | Qwen2-VL-2B (see B6), Qwen3-VL and Pixtral (d=64, kernel path). Qwen2.5-VL is **not** affected today (everything on SDPA). | confirmed on CPU [R1] |
| B6 | The head_dim check is done on the **config** (`hidden_size / num_attention_heads`), not on the individual call. It is wrong for sub-models: in Qwen2-VL-2B it computes 96 instead of 80 and lets our backend through. It also moves entire sub-models to SDPA | VLM; all models with custom head_dim | confirmed from code and config [R2] |
| B7 | BF16 or FP32 inputs converted to FP16 without warning in the kernel path: possible overflow above 65504 | BF16 models (Gemma, …) | **closed in P0.5** |
| B8 | On T4 the forward d=64 and d=128 raises an exception (shared memory above 64 KB per block): BERT and Llama do not start | sm_75 | confirmed by ptxas, to be verified on the T4 [R4] |

**Missing features** (today on SDPA: correct, but without a kernel):
- padding;
- packing;
- sliding window: Gemma 3 22 layers out of 26, Gemma 2 13 out of 26, Phi-3 and Mistral v0.1 all;
- decode (`q_len=1`);
- softcap and sinks in the kernel;
- dropout;
- BF16;
- head_dim 72 (SigLIP) and 80 (Qwen2-VL / 2.5-VL);
- `torch.compile` [R10].

**Architectures.**
- **Turing**: see B8; the d=256 backward also fails (66.5 KB).
- **Occupancy**: 4–8 warps per SM out of 64 everywhere (`my_O[d]` per thread, about 255 registers).
- **Spill**:
  - fwd d=256 on sm_70: stack of 1768 B;
  - CUDA-core backward at d≥96: `my_dK`/`my_dV` entirely in local memory.
- **GTX 1080 (sm_61)**: forward limited by FP16 `HMUL2`, about 30 times.
- **Pascal backward**: one global `atomicAdd` per dQ element and per pair of rows.
- **Build**: about 8 minutes.

**Current performance on the V100.**
- Prefill at 4096 tokens: 1.3 to 1.55 times faster than HF SDPA, with about half the memory. The comparison must however be redone against SDPA mem-efficient with the backend explicitly selected, because HF SDPA with GQA on sm_70 may use the "math" backend and inflate the gain [R12].

---

## 1b. Measured performance (2026-10-21)

The forward is correct but **about 6× to 10× slower than PyTorch's SDPA memory-efficient**, on all architectures (even after the Bc=16 tiles of P2.3). The README benchmarks show it as "faster" because they compare against eager, which materializes the N×N matrix.

| Causal forward, B=2, H=16, N=4096 | ours (initial tiles) | ours (Bc=16, P2.3) | ours (padding, P5.3) | SDPA |
|---|---|---|---|---|
| V100, d=64 | 22.1 ms | 13.4 ms | **10.1 ms** | 2.34 ms |
| V100, d=128 | 33.0 ms | 29.0 ms | **24.3 ms** | 3.74 ms |
| T4, d=64 | 53.6 ms | 37.4 ms | **26.1 ms** | 4.59 ms |
| T4, d=128 | 64.6 ms | 64.5 ms | **48.8 ms** | 10.4 ms |
| P100, d=64 | 68.1 ms | — | **30.9 ms** | 15.7 ms |
| P100, d=128 | 127 ms | — | **121 ms** | 28.9 ms |
| GTX 1080 Ti, d=64 (H=8) | 151 ms | — | **40.8 ms** | — |

After P5.3 (Nsight Compute, T4, d=64): shared-memory read bank conflicts from 389 million to **0**, forward from 37.4 to 26.1 ms. Occupancy remains at 12.5% (1 block per SM) and stalls on shared memory and barriers: this is the limit of the "one thread per row" design, addressed by P2.8.

**Pascal backward** (P5.2, done): fwd+bwd with d=128 at 4096 tokens from 3.1–3.8 s to 0.27–0.43 s (P100 / 1080 Ti), i.e. 9 to 11 times faster.

Nsight Compute on the T4 (d=64, N=4096):
- tensor cores active at **2.1%** of peak;
- **92%** of shared-memory reads and writes in bank conflict;
- main stall: waiting on shared memory (44%);
- occupancy 12.5% (1 block of 4 warps per SM, 168 registers per thread).

Cause: the "one thread per row" design. Softmax, conversion of P and accumulation of O read and write their own row of S/P/O in shared memory with a stride of 32 floats, so all threads of the warp hit the same bank.

Changes:
1. **P5.3, brought forward**: padding and 128-bit accesses, a contained fix.
2. **P2.8**: forward in the FA2 style, with S/P in registers and the warps splitting the rows; it is the step needed to reach the level of SDPA.

## 2. Principles

1. **No wrong result silently.** Every unsupported case goes to a fallback **that applies it correctly** or raises an explicit error. The decision is made per call, never on the config.
2. **Gold reference: eager FP32.** The criteria are based on the logits within FP16 noise (eager FP16 against eager FP32 on the same input), not on token identity, which breaks with FP16 near-ties [R11].
3. **FA2-compatible API** (`flash_attn_func`, `flash_attn_varlen_func`, `window_size`, `softcap`, `causal`, `deterministic`). We draw inspiration from FA2 and xformers/CUTLASS; if we copy code we keep the BSD-3 license.
4. **Validation on every architecture**:
   - V100;
   - Pascal path forced on the V100, **in a test build that keeps it** and with parametrized tests [R5];
   - GTX 1080 Ti;
   - T4.
5. **Code homogeneity: everything is done in one way, in one place.**
   - **A single source of truth** for head_dim, tile and shared memory (`constexpr` function used by launch and checks), masking conventions, per-architecture dispatch and parameter names, the same from the kernel to the Python API.
   - **No parallel copies**: one tile loader, one set of WMMA helpers, one kernel with policies for batched and varlen. Features extend these structures.
   - **Same scheme** for validation (exceptions, not `assert`), errors, fallbacks (single registry of reasons) and tests (same reference, same helpers, same tolerances).
   - **Review checklist** for every PR: does a function or pattern already exist? Does the name follow the conventions? Is the fallback reason in the registry?
6. **Structure first, then features.** The kernel work (varlen backward, window, softcap, sink) comes **after** the P2.1 refactor [R6].

---

## 3. Phases

### P0 — Integration correctness (Python, small) — **M in total**

Goal: **no wrong output** on any architecture. Only the wrong cases are fixed, without making slower those that are correct today [R8].

| ID | Change | Size | Acceptance criterion |
|---|---|---|---|
| P0.1 | **Single fallback registry** and **per-call decision**. The head_dim check is removed from the config (B6); the fallback is decided based on dtype, actual head_dim, kwargs (`softcap`, `s_aux`, `cu_seq_lens`, `position_bias`), mask, and whether the tile fits the architecture. Every fallback records the reason, with a one-time warning. | S | Qwen2-VL-2B: the check uses the real d=80. Test that verifies the recorded reason for every case. |
| P0.2 | **B1, StaticCache**: when we delegate to the SDPA mask, we call it with `allow_is_causal_skip=False`; `None` remains only in the case we verified. | S | Logits within noise relative to eager FP32 with static, hybrid and sliding (Gemma 2/3) cache, offloaded, assisted decoding and beam search, on TinyLlama, Llama 8B and Gemma 3. |
| P0.3 | **B2, B3, MLA**: with `softcap` or `s_aux` a fallback that applies them is used (eager with mask). With a V head_dim different from Q's, SDPA is used, which supports it [R16]. | S | Gemma 2 and reduced gpt-oss within noise relative to eager FP32. |
| P0.4 | **B4 and B5**: with `cu_seq_lens_q/k`, `flash_attn_varlen_func` is used if the head_dim is supported. Otherwise a fallback that **respects the segments** (block-diagonal mask or loop per segment) [R1]. Packing derived from `position_ids` with `use_cache=False` stays on today's SDPA fallback, which is already correct, until P1.3 makes it fast [R8]. | S–M | Packed batch: loss equal to that of the sequences run individually. Vision encoder of Qwen2-VL-2B, Qwen3-VL and Pixtral (2 and 8 images) within noise. Gradient of the q/k proj (1-cos relative to the gold) no worse than SDPA FP16 mem-efficient [R7]. |
| P0.5 | **B7, dtype** (done; the `FP16RangeError` range check and the `fp16_overflow` fallback were later replaced in P6.2 by power-of-2 scaling on the GPU): BF16/FP32 in inference run on the kernel after a controlled conversion to FP16 in the API (`FP16RangeError` if a value exceeds 65504 → `fp16_overflow` fallback in the original dtype). In training (gradients required) they go to SDPA (`non_fp16_training`): without loss scaling FP16 gradients would underflow. FP32 in inference: one-time warning that the computation is in FP16. In addition: the Pascal forward dot product is no longer formed in FP16 (overflow with products > 65504; also 2.3–2.9× faster on the 1080 Ti). | S | Done: overflow tests, BF16 training and large products fail on the previous code. |
| P0.6 | **Postponed and merged into P2.2/P2.3** (decision 2026-10-21: no T4 available, so the guard would not be verifiable). Until then on Turing the forward d=64/128 raises an explicit error (`cudaFuncSetAttribute`), it does not give wrong results. | — | See P2.2/P2.3. |
| P0.7 | **Edge cases**: `is_causal=None` in the kwargs is interpreted as a module attribute, as SDPA does. Validation of `cu_seqlens` (monotonicity, last value equal to the total). Zero-length sequences without launches with grid 0. | S | Dedicated tests. |
| P0.8 | **Documentation of the known limits found in P0.1–P0.2** (README):<br>• Pascal + `cache_implementation="static"`: HF compiles on its own and Triton does not support sm<7, so `disable_compile=True` is needed.<br>• Bidirectional decoder (`is_causal=False`) with static cache: the empty keys are included, as HF's SDPA does.<br>• Doge with static cache: indexes the mask as 4D and raises an error. | S | "Known limitations" section updated. |
| P0.8b | **Tests rewritten against eager FP32**: model matrix as an optional slow test; each bug B1–B8 becomes a regression test that **fails on the current code**. Tests parametrized with `FLASH_ATTN_LEGACY_FORCE_ARCH`. Training: comparison of the loss over 50 steps and of the 1-cos of the gradients with SDPA mem-efficient FP16. | M | The tests fail before P0 and pass after, on V100, T4 and 1080 Ti. |
| P0.9 | **Backend name: decided to keep `flash_attn_legacy`** (2026-10-21). With P0.4 the name "flash" makes the VLMs' `cu_seq_lens` reach us, so a single varlen call on the kernel instead of the per-image Python loop. The monkey-patch of `_check_and_adjust_attn_implementation` stays, and Llama 4 (which raises for "flash" names) arrives with P6.7. | — | No change. |

### P1 — Benchmark infrastructure and varlen host — **M**

| ID | Change | Size | Acceptance criterion |
|---|---|---|---|
| P1.1 | **Done (kernel; branch p2-turing)**: `benchmarks/benchmark.py` measures forward and fwd+bwd against SDPA memory-efficient and math (backends explicitly selected) and eager, with mean and standard deviation over 20 repetitions (minimum 10) and peak memory; `--save` writes `docs/bench/<gpu>.json`. Baseline recorded on V100, T4, P100 and 1080 Ti (causal, B=2 H=16, N=512–4096, d=64/128). Legacy/mem-efficient ratio at N=4096: forward V100 4.0×/6.2× (d=64/128), T4 4.9×/4.7×, P100 1.9×/4.6×, 1080 Ti 0.29×/0.37× (SDPA mem-efficient on sm_61 is extremely slow: FP16 at 1/64); fwd+bwd V100 2.8×/4.3×, T4 3.5×/4.5×, P100 0.88×/1.25×, 1080 Ti 0.13×/0.15×. Remaining: model-level benchmark (prefill, training), today in `tests/test_generation_benchmark.py`. | M | Done: baseline on 4 GPUs before P2.8. |
| P1.2 | **Partly done (branch p2-turing)**: Q block offsets computed on the device (no host loop nor synchronous host→device copy), Br from the same tile selection as the launch (no per-SM table). Validation of `cu_seqlens` (P0.7) with a device→host copy, now optional: with `validate=False` (used by the HF integration, P3.2) no sync, grid with an upper bound. Original plan: **Host varlen**: grid `(cdiv(max_seqlen_q, Br), num_seqs, H)` without `block_offsets`, without `.to(CPU)` and without the hardcoded per-SM Br table. Touches only the launch, not the kernel logic. | S–M | No CPU-GPU sync in the varlen forward (profiler); tests green. |

### P2 — Kernel foundations, Turing and build — **L**

| ID | Change | Size | Acceptance criterion |
|---|---|---|---|
| P2.1 | **Refactor**: one single forward kernel and one backward with `SeqInfo` policy (batched/varlen) and `Mma` policy (WMMA sm_70/75, SIMT sm_60/61). Single `constexpr` function for shared memory and common helpers. Features become uniform runtime branches, not template flags, to contain instantiations. | L | Forward bit-identical before and after; no regression beyond the noise measured with the P1.1 harness. |
| P2.2 | **Done (branch p2-turing)**: tiles chosen at runtime with `with_first_fitting_tile` on the device's per-block shared memory (`cudaDevAttrMaxSharedMemoryPerBlockOptin`, with cache; test hook `FLASH_ATTN_LEGACY_SMEM_LIMIT`). Tile template with a single `smem_bytes()`; varlen takes `Br` from the same selection. | M | T4: 525 tests green (previously 198 failures); V100 with 64 KB limit: 397 tests green. |
| P2.3 | **Done (branch p2-turing)**: forward tiles measured and unified to **Bc=16 for every head_dim** (`<128,16,d>` for d=32–128, `<64,16,256>`): faster than the previous ones on V100 and T4 (causal forward, B=2 H=16 N=4096, V100: d=32 from 13.3 to 8.95 ms, d=64 from 21.0 to 13.4, d=96 from 23.4 to 15.3, d=128 from 33.0 to 29.0; T4: d=32 from 38.3 to 22.7, d=64 from 53.5 to 37.4, d=96 from 55.2 to 43.1) and all within Turing's 64 KB. Backward with dK/dV accumulators in WMMA fragments in registers (d=256 from 66.5 to ~38.5 KB; 0 measured spills on sm_70/75). The rest is superseded by P2.8 / P2.9. | M | Done: T4, V100 and 1080 Ti green. |
| P2.4 | **Done (branch p2-turing)**: varlen backward in a single launch (`backward_varlen`): the two backward bodies take `BwdBlockArgs` built from a batched or varlen grid (sequence lookup in `varlen_seq_of_block`, shared with the forward), D with L's layout. `cu_seqlens` validation with sync is optional (`validate=False`, P3.2). Original plan: **Native varlen backward** in a single launch, on top of the unified kernel of P2.1 [R6]. | M | Gradients as the reference; no sync; time measured. |
| P2.5 | **Done (branch p2-turing)**: 64-bit strides and offsets (`index_t`) in all parameter structs and block arguments (tile copies keep an `int` row stride: 64-bit in the loop cost 30% on Pascal at d=256), 64-bit `precompute_D`, batched grids flattened into `gridDim.x` (no 65535 limit on B·H), size checks on the host. Tests: B·H = 66000, tensors above 2^31 elements batched and varlen (V100 32 GB both; T4 and P100 one of the two). Original plan: **int64 and grid**: int64 offsets, B·H outside `gridDim.y`, `#include <mma.h>` outside the namespace. | S | Tests with B·H > 65535 and, where memory allows, above 2^31 elements. |
| P2.6 | **Build** — *configurable architectures brought forward (user request 2026-10-21)*: `TORCH_CUDA_ARCH_LIST` / `FLASH_ATTN_LEGACY_ARCHS` to choose the architectures; by default only those supported by the installed `nvcc` (`nvcc --list-gpu-arch`), with a warning for the excluded ones (CUDA 13 no longer compiles sm_60/61/70). Then: files split by (family, head_dim); the release build compiles Pascal only for sm_60/61, a **test build** keeps Pascal also for sm_70/75 so the forced path stays testable [R5]. | S–M | CUDA 13 compiles for the T4 only without errors; with `TORCH_CUDA_ARCH_LIST="7.5"` build in a few minutes; release build under 3 minutes; tests with forced Pascal in the test build. |
| P2.7 | **Done (branch p2-turing)**: d=72/80 on the 80 (WMMA) and 96 (CUDA-core, which require d multiple of 32) tiles with zero columns: tile copies receive the valid columns (`tile_valid_cols`, compile-time constant for tiles without padding), the O/dQ/dK/dV writes are limited to the real columns. N=4096 causal, against SDPA mem-efficient: fwd V100 0.92×, T4 0.71–0.73×, P100 0.94–0.98×; fwd+bwd 0.62–0.74×. Qwen2-VL vision encoder (d=80) on the varlen kernel in the HF test; SigLIP-style module (d=72). Build time: 4 extra instances per WMMA architecture (fwd/bwd × batched/varlen), not measured separately. Original plan: **head_dim 72 and 80** (verified on the configs). d=80 is already a multiple of 16 (Bc=16); d=72 is padded to 80 with zeros in shared memory for WMMA. On Pascal multiples of 8 are needed for vector loads [R14]. The build-time cost must be measured. | M | Unit tests for d=72 and d=80; Qwen2-VL and SigLIP vision encoders on the kernel. |
| P2.8 | **Done (branch p2-turing): steps 0, 1, 2, 5, 6 done, target reached on V100 and T4; step 3 (`mma.sync` on Turing) remains optional.** Step 6: `occupancy.py` regenerates `occupancy_table.md` reading the tiles from the kernel names and `kNumWarps` / `smem_bytes()` from the headers (no duplicated formula); small residual spills, the largest on the T4 WMMA forward d=256 (98/220 B), possible follow-up. Causal forward N=4096 against SDPA mem-efficient: V100 0.67–1.03×, T4 0.57–0.89× (previously 4–6.6× and 4.7–5.5×). Done: band body with WMMA and per-warp staging, 8-warp tiles with fallback to 64 KB, 128-bit copies everywhere and K/V prefetch in registers. Step 5 also done: Pascal forward with fp32 micro-tiles in the shared body `flash_attn_fwd_band.h` (P2.1 for the forward: a single K/V loop, masking, softmax, batched/varlen kernel and launcher for all families). P100 N=4096: d=64 27→12.2 ms (0.78× SDPA), d=128 81→52, d=256 1768→285; 1080 Ti d=64 74→15.4, d=128 176→41. Remaining: d=128/256 on Pascal (TN=1 tiles limited by shared memory, d=256 at one warp per block), step 3 (`mma.sync` on Turing) optional. **Forward redesigned FA2-style** (merged with P2.1, replaces P6.6). S and P remain in the fragment registers (`mma`/WMMA), each warp handles a band of 16 rows instead of one thread per row, softmax with warp-level reductions, O accumulated in fragments, no round trip through shared memory. On Pascal the same scheme at warp level with FP32 FMA. Reason: see §1b. Design and step order (0: preparation and new tests; 1: band body + WMMA with per-warp staging on sm_70/75; 2: K/V prefetch in registers; 3: `mma.sync` m16n8k8 on Turing; 4: tuning; 5: Pascal fp32 micro-tiles; 6: cleanup) in `docs/analysis/kernels/p2_8_design.md`. | L | Forward at 4096 tokens within 1.5× of SDPA memory-efficient on V100 and T4 (before the redesign 4–6.6×); agreed ambitious target: about 1×. Same correctness tests. |
| P2.9 | **Done (branch p2-turing)**: WMMA backward FA2-style: each warp has a band of 16 keys (and a d-slice for d≥96) and computes Sᵀ, Pᵀ, dPᵀ, dSᵀ as fragments, with element-wise steps on the fragments (row/column map from `wmma_acc_layout`); dK/dV in registers; shared dSᵀ in shared memory for dQ = dS·K across all warps, with coalesced atomics. Two barriers per Q block (previously 9+). fwd+bwd causal N=4096 against SDPA mem-efficient: V100 from 1.9–2.6× to 0.65–1.15× (d=64 20.4→11 ms, d=128 47→20.9), T4 from 2.4–2.8× to 0.74–1.23× (d=64 54→20, d=128 124→54). Remaining: with the d-slice (kSplitD>1) warps of the same band recompute Sᵀ and dPᵀ (+50% MMA at d=96/128, 2.5× at d=256): split the reduction over d or share Pᵀ/dSᵀ; high registers (220–255, 16 B of spill at d=64 on sm_75); Q/dO prefetch. | L | Done: tests green on V100, T4, P100, 1080 Ti. |

### P3 — Padding, packing and native VLMs (after P2) — **M–L**

| ID | Change | Size | Acceptance criterion |
|---|---|---|---|
| P3.1 | **Done (branch p2-turing)**: option (a) extended to all cases, now that varlen has native forward and backward: the mask builder returns the 2D padding mask (FA2 convention), the attention function unpads onto the varlen kernel and repads with `index_copy` (differentiable: training included). Condition: contiguous padding per row (otherwise fallback). Query: causal (self-attention only) the valid ones among the last q_len key slots, with padding rows at 0 as in FA2; bidirectional (encoder, cross-attention, where queries do not depend on the key mask) all rows, as in SDPA. Unpad plan with one sync per mask (cached, also under inference_mode for the builder's masks); the varlen validation sync per call remains (P6.5). In the fallback with padding, rows with no visible keys are made safe as in HF (NaN on torch < 2.5). A first attempt was wrong for cross-attention with padded encoder (blocked in review, mutation tests added). Tests: Llama and Gemma 3 with left/right padding within noise, generate with left padding, unpad against masked SDPA with gradients, GQA and chunked prefill. Original plan: **Padding**: to choose between (a) unpad, varlen and repad with the HF helpers, and (b) **key padding in the kernel**: `seqlens_k` and per-batch start offsets in the parameters, which avoids unpad and repad in generation with left padding (size S in the report) [R9]. Proposal: (b) for generation, (a) for training and right padding; to be decided after P2.1. | M | Padded batch: kernel in all layers and logits within noise on 3 models; gradients as in P0.4. |
| P3.2 | **Done (branch p2-turing)**: packing and padding in training on the native varlen (P2.4). Two corrections emerged while measuring: (1) with fp16 autocast (fp32 weights + GradScaler, the standard AMP setup) HF's rotary makes q/k fp32 and all training fell back to `non_fp16_training`: under fp16 autocast the attention now computes in fp16 as SDPA does (cast without range check, gradients allowed); (2) `validate=False` in the varlen (grid with upper bound, no sync) for the `cu_seqlens` built by the integration: no per-layer sync. Llama d=128, 8 layers, 3.6k packed tokens on V100: 214 ms/step and 8.9 GiB versus 542 ms and 20 GiB of the SDPA fallback with block mask; at 9k tokens SDPA goes OOM (ours 16.5 GiB); loss as SDPA (first-step difference 0.002). Original plan: **Native packing** from `position_ids` and `cu_seq_lens` on the P2.4 varlen backward. | S | Loss over 50 steps as SDPA; faster than the fallback (harness). |
| P3.3 | **Done (branch p2-turing)**: Qwen2-VL (d=64/80), Qwen2.5-VL (windows), SigLIP (d=72), CLIP, Pixtral (images of different sizes) with 1, 3 and 8 images: all layers on the kernel (batched or varlen), output within noise of eager FP32. Smaller Br for the 64-token windows of Qwen2.5-VL evaluated and discarded: varlen with segments of 64 (16k tokens, H=16, d=80) V100 0.47 ms against 0.63 for batched SDPA, T4 1.28 against 1.15; with 64-row tiles the T4 drops only to 1.21 (−6%) and gets worse on long segments. Original plan: **Vision encoder** on the kernel: Qwen2-VL/2.5-VL (with the 64-token windows roughly, which waste Br=128 tiles: evaluate a smaller Br), SigLIP, CLIP, Pixtral, with 1, 3 and 8 images. | M | Kernel in all vision layers; output within noise. |

### P4 — Kernel features (after P2) — **M–L**

| ID | Change | Size | Acceptance criterion |
|---|---|---|---|
| P4.1 | **Done (branch p2-turing)**: `window_size=(left, right)` with upstream semantics (rows aligned bottom-right, −1 unlimited) in the forward and backward, batched and varlen; out-of-window blocks are skipped, inner ones without mask. HF integration: the window comes from the mask builder (`local_size` verified on the `mask_function`) or from the `sliding_window` kwarg; the fallbacks apply it too. P100, window 1024, causal: at 8k forward 11.6 ms against 121.8 of SDPA mem-efficient, fwd+bwd 71 against 844, less memory. Reordering of the QK FMAs on Pascal (bank conflicts): d=128 causal forward from 51.8 to 34 ms. Original plan: Sliding window `window_size=(left, right)` in the forward and backward, skipping the blocks | S–M | Gemma 3, Mistral v0.1 and Phi-3 on the kernel beyond the window, within noise; timings against SDPA mem-efficient (at 8k SDPA may go OOM: compare at 4k, more memory at 8k) [R12] |
| P4.2 | **Done (branch p2-turing)**: `softcap` (upstream API) in the forward and backward of all families, batched and varlen. A single score map (`ScoreMap`): x = c·tanh(s·scale/c) in the log2 domain, dx/ds = scale·(1−t²); the branch is uniform and outside the per-element loops (with the branch inside, the WMMA backward lost up to 14% even without softcap). Cost of softcap: P100 +3%, V100 +8–15%. Against eager FP16 with softcap: V100 0.13–0.68×, P100 d=128 0.40–0.72×, memory 30–150× lower; **P100 d=256 2.4× slower than eager** (the known Pascal d=256 limit, which is the head_dim of Gemma 2 2B/9B). Integration: Gemma 2 on the kernel in all layers (sliding with the window); decode with softcap goes to eager, never to SDPA. Original plan: Softcap: `cap·tanh(s/cap)`, with the backward multiplying by `(1−t²)` | S–M | Gemma 2 on the kernel, within noise |
| P4.3 | **Done (branch p2-turing)**: `sinks` (one logit per Q head, HF kwarg `s_aux`) in the forward and backward of all families, batched and varlen. The online softmax starts from the state of a key with logit = sink and value 0 (`m = sink`, `l = 1`), so L includes the sink and dQ/dK/dV do not change; d_sink[h] = −Σ exp(sink − L)·D, reduced in `precompute_D` (one atomic per warp). −inf = no sink. Cost: V100 fwd unchanged, fwd+bwd +0–3%; against the eager fallback: V100 0.05–0.32×, memory at 4k 32 MB against 6 GB. Integration: gpt-oss on the kernel in all layers (sliding with the window); decode with sinks goes to eager, never to SDPA. Original plan: Sink (`row_max=sink`, `row_sum=1`) and reduction for `d_sink` | S–M | reduced gpt-oss: top1 as eager FP16 |
| P4.4 | **Done (branch p2-turing)**: window, softcap and sinks on the kernel in all models of the matrix (Llama, Mistral, Phi-3, Gemma 2, Gemma 3, gpt-oss: prefill without fallback, a single `test_model_matrix_on_kernel` test). `position_bias` is no longer handled defensively: in transformers 5.20 the T5 family (T5, mT5, LongT5, UDOP, Switch Transformers, ...) passes it to the attention interface, and all its layers went to eager FP32; now it goes to SDPA with the bias in the mask (like HF's SDPA), differentiable in the bias. Eager remains only for softcap and sinks outside the kernel (decode, until P6.1). Original plan: Integration: the P0.3 fallback remains only for unsupported cases | S | Model matrix without fallback for window, softcap and sinks |

### P5 — Pascal performance (GTX 1080 Ti) — **M–L**

| ID | Change | Size | Acceptance criterion |
|---|---|---|---|
| P5.1 | **Forward done (P2.8 step 5)**: Q, K, V in FP32 in shared memory (converted once) and P@V in FP32 on sm_60 and sm_61. Still open, to be discussed: the CUDA-core backward still reads half from shared memory and converts in the products (already faster than SDPA mem-efficient at d=128: P100 fwd+bwd 150 against 214 ms); FP16x2 (HFMA2, double speed) only on sm_60, with the accumulation-precision problem. Original plan: FP32 path for sm_61. **Partly brought forward in P0.5**: the Q·K product of the Pascal forward is in FP32 (FFMA), 2.3–2.9× faster on the 1080 Ti. Remains: K/V in FP32 in shared memory (fewer conversions), P@V, runtime selection for sm_60 vs sm_61, measured also on P100 | M | 1080 Ti and P100: forward measured before and after with the harness |
| P5.2 | **Done (branch p2-turing)**: CUDA-core backward rewritten with tiling like FA2: S/P/dP/dS of the [Br,Bc] tile computed cooperatively with 128-bit products on padded rows, P and dS in shared memory with stride Bc+1 (no conflicts for rows and columns), dK/dV in registers per (KV row, column slice), dQ summed over the whole KV block with **one** `atomicAdd` per element and block (previously one per q/kv pair: Bc times more). 0 spills on sm_60/61 (previously 2–4.7 KB of stack at d≥96). fwd+bwd causal B=2 H=16 N=4096: P100 d=64 788→93 ms, d=128 3094→270 ms; 1080 Ti d=64 802→187 ms, d=128 3830→431 ms. Tile loader and 128-bit dot product now unique in the header (previously three copies). To measure later: `atomicAdd` of dQ with the faster columns (fewer L2 sectors per instruction) and constant trip count in step 1. | M | Done: tests green on P100, 1080 Ti, V100/T4 (forced Pascal). |
| P5.3 | **Done (branch p2-turing)**: WMMA forward with padded rows of S/P (+4 float / +8 half) and 128-bit accesses per row, softmax writing P directly in half; CUDA-core forward with padded Q (+8 half), 128-bit dot product and P in registers (no P buffer). Block body shared between batched and varlen kernels (`volta_fwd_block`, `pascal_fwd_block`, brought forward from P2.1). Br=64 tried: ±10% differences with opposite sign between GPUs, kept Br=128. | S | Done: read conflicts at 0 (Nsight, T4); timings in §1b. |
| P5.4 | **Done (branch p2-turing)**: Pascal d=256 (Gemma 2/3): the fp32 shared memory left only one 16-row block, i.e. **one warp per SM** on the P100 (half of the FMA units idle, latencies exposed). Now the warps of a band split the columns (`kSplit` in the tile): partial S per column slice, summed in shared memory with a named barrier per band, same softmax, O per slice (16 registers instead of 128). Partial S in shared memory per lane (no bank conflicts). Causal B=2 H=16 N=4096: P100 288 → 99 ms (kSplit 2/4/8 = 153/130/120 before the conflict fix), 1080 Ti 196 → 121; SDPA mem-efficient 58 / 1049, eager 110 / 141. The forward alone remains above SDPA (P100 99 against 58 ms) because of the Bc=8 limit (little work per K/V block): possible next steps, more invasive, K and V not resident together (Bc=16) or Q and K in column slices like the cutlass kernels. Backward d=256: same limit (36 KB, one block per SM on the P100), 8-warp tile instead of 4: fwd+bwd P100 706 → 398 ms (SDPA mem-efficient 443, math 271), 1080 Ti 664 → 566 (math 353). | M | P100 d=256 at least on par with SDPA mem-efficient |
| P5.5 | **Open**: Pascal forward d=128 tile per SM. `FwdTileSimt<32, 8, 128, 2>` (the warps of a band split the columns, as P5.4) against today's `<32, 8, 128>`, causal B=2 H=16 N=1k–8k, measured twice alternated (2026-10-23): P100 −10% (4k 35.8 → 32.1 ms, SDPA mem-efficient 28.9: 1.24× → 1.11×), 1080 Ti +11% (42.5 → 47.0 ms). One tile cannot serve both: the CUDA-core launcher takes the SM (`get_sm_version()`, 60 also under `FLASH_ATTN_LEGACY_FORCE_ARCH=pascal`) and picks the split tile on sm_60 only. Cost: one more d=128 instance (batched + varlen). Tests: the split tile on the P100 and on forced Pascal (V100 / T4 / 2080 Ti), today's on the 1080 Ti. | S | P100 d=128 forward −10%, 1080 Ti unchanged; tests green on all GPUs |

### P7 — Backward on tensor cores and Turing `mma.sync` (order chosen 2026-10-22: this, then P6.1, P6.2, P6.5)

| ID | Change | Size | Acceptance criterion |
|---|---|---|---|
| P7.1 | **Done**: coalesced `precompute_D` (8 lanes per row): V100 0.76 → 0.09 ms, T4 2.62 → 0.28 | S | — |
| P7.2 | **In progress**: Turing backward with `mma.sync` m16n8k8 and `ldmatrix` (`BwdMma`, design and measurements in `docs/analysis/kernels/bwd_design.md`): P^T and dS^T remain in registers as A operands, no per-warp buffer, d=128 with 8 warps on the T4. T4 backward d=128 53.7 → 32.3 ms, d=64 18.2 → 16.6; then S^T / dP^T split per query among the warps of a band (P^T / dS^T exchanged in shared memory with a band barrier): d=128 26.9, d=256 168 → 136, d=64 14.6 (kSplitD 2). T4 fwd+bwd N=4096 against SDPA mem-efficient: d=64 0.86×, d=128 0.83×, d=256 0.91×. Family chosen by the arch of the running code (`compiled_arch_for`); sm_8x uses `BwdMma` via PTX 7.5 (untested). | L | T4 fwd+bwd within ~1× SDPA mem-efficient |
| P7.3 | **Done**: Turing forward with `mma.sync` (`TuringFwd`, `flash_attn_fwd_turing.cu`; P2.8 step 3): S and O in m16n8 accumulators, Q in registers up to d=128, online softmax on the accumulators (`BandSoftmax<2,4>`), P as A operand of P V, no staging. T4 causal forward N=4096: d=32 2.74 → 1.19 ms, d=64 3.64 → 2.14, d=96 6.14 → 2.95, d=128 7.12 → 4.13, d=256 18.4 → 16.4. WMMA remains for sm_70 (and FORCE_ARCH=volta). | M | — |
| P7.5 | **Done**: GQA backward on small grids: the Q heads of a KV head split over S blocks (the smallest divisor of the group that gives 2 blocks per SM), fp16 partials of dK / dV summed after the kernel (like upstream's `dk_expanded`). V100 GQA 16/2 d=128 N=1024: 2.27 → 0.78 ms (SDPA 0.77); TinyLlama training at 1k on V100 1.34× eager (the faster of the two) and 1.63× SDPA (`docs/bench/models.md`). | S | GQA backward as MHA at equal blocks |
| P7.4 | **In progress**: V100, WMMA with the same per-query split of S^T / dP^T among the warps of a band (P^T in the band buffer, dS^T from the shared buffer, band barrier). V100 N=4096 backward kernel: d=96 16.6 → 12.5 ms, d=128 19.3 → 14.6, d=64 12.0 → 11.3, d=32 5.6 → 5.2, d=256 50.8 → 49.4; fwd+bwd against SDPA mem-efficient d=128 1.30× → 1.05×, d=256 1.02×; d=64 with `<64,64,64,2>` (8 warps, split per query) 13.7 → 10.3 ms, 0.96×. Alternative `mma.m8n8k4` if needed. | M | fwd+bwd V100 within ~1× SDPA |

### P6 — Extensions (to prioritize)

| ID | Change | Size |
|---|---|---|
| P6.1 | **Done** (design `docs/analysis/kernels/decode_design.md`, steps a–g): `flash_attn_with_kvcache` with upstream's positional order, one CUDA-core kernel for all architectures (decode is bandwidth-bound), fp16/bf16 cache read without copies (also the HF layout as a transposed view). Step a done: contiguous cache, R ≤ 4 rows per KV head, causal / window / softcap / sinks, lse. Step b done: key split (automatic without device reads, or forced up to 128) and combination with the same `decode_merge`. T4 GQA decode 32/8 d=128: B=1 context 16k 3.27 → 0.32 ms (SDPA 3.86), B=8 4k 0.58 ms (SDPA 6.05), ~200 GB/s out of 320. Step c done: up to 8 rows per KV head (q in shared memory interleaved; gpt-oss 64/8, Qwen2 28/4); T4 64/8 d=64: 0.06–2.8 ms against 0.19–30 ms of SDPA, but ~60–100 GB/s: limited by per-row instructions (step g). Step d done: `cache_leftpad`. Step f done: HF's decode steps on the kernel (dynamic, static and offloaded caches, left padding from the 2D mask or from the 4D `[B,1,1,kv]` of static caches, window, softcap, sinks), one host sync per step with padding; remain on SDPA FP32, more than 8 rows per KV head, active gradients. Step g: measurements and attempts in `decode_design.md` (≤ 4 rows near the bandwidth limit; 8 rows ~100 GB/s). Step e done: paged cache (`block_table`, blocks multiple of 16, NHD/HND/vLLM layouts), contiguous path unchanged. **P6.1 done.** Step h (later, from the benchmarks on real models): the key step without per-row branches (the rows ran one after the other), split of 128 keys, parallel combine: kernel + combine 1.4–5× faster on all GPUs (GQA 16/2 at 2k: V100 146 → 28.5 µs, P100 185 → 37), V100 up to 680 GB/s; Gemma 2 at 8k on V100 in generate (eager steps, DynamicCache) 1.18× the faster of SDPA and eager (eager-step decode rows are host bound and move by up to ~30% between runs, `docs/bench/models.md`). | M–L |
| P6.2 | **Inference done**: BF16/FP32 weights on the kernel without model conversions: q, k, v scaled to powers of 2 in the FP16 range on the GPU (exact, no sync, no overflow fallback; the kernel reads 1/(s_q s_k) from memory, the output is multiplied by 1/s_v); decode reads BF16 natively. BF16 training remains on SDPA (before Ampere the HF Trainer requires fp16 AMP, which is supported). Original plan: BF16/FP32 training on the kernel: power-of-2 scaling of the gradient dO before conversion to FP16 (exact, the backward is linear in dO), dQ/dK/dV returned in FP32 or in the original dtype, handling of overflow of the scaled gradients. Closes the `non_fp16_training` fallback | M |
| P6.3 | Deterministic dQ (separate dQ kernel, see P5.2) | M |
| P6.4 | Dropout with Philox | L |
| P6.5 | **Done** (design in `docs/analysis/compile_design.md`): step a done, the kernels are dispatcher operators (`TORCH_LIBRARY`, `torch.ops.flash_attn_legacy.*`) with fake implementations; `flash_attention` (fwd+bwd), varlen and `flash_attn_with_kvcache` compile with `fullgraph=True`; cost +6 µs of host per call. Step b done: key mask in the decode kernel (`key_mask`, each split restricted to the first / last kept key), HF decode with padding no longer syncs and holes in the mask go to the kernel. Step c done: a compiled decode no longer breaks the graph (static cache: the 4D mask that transformers builds for compilable caches goes to the decode kernel; the import of `masking_utils` inside the builder broke the graph and is moved into `register()`; dynamic cache: the builder returns the 4D SDPA mask under tracing); T4, small model with reused static cache: compiled generate 103 ms against 259 eager (SDPA 102 against 254). Step d done: measurements on real models (`benchmarks/models.py`, one process per model and implementation, tables in `docs/bench/models.md`, 5 GPUs, 6 model families up to 14B, prefill / generate / train incl. padded batches and LoRA): compiled decode with static cache, speedup against the faster of SDPA and eager, V100 1.00–1.96× (TinyLlama B=8 2k 973 tok/s against 496 eager, Qwen2.5-3B 8k 1.53×, Gemma 2 2B 8k 1.30×), T4 1.06–3.20×, 2080 Ti 1.04–1.31× where the others fit; at B=8 with 8k only ours fits in memory for the 1-4B models; each compiled step is a single graph, with no breaks and no fallback. **P6.5 done.** Original plan: `torch.compile` and CUDA graph: register the operations with `torch.library` and remove host synchronizations: in the mask builder `bool(pad.all())` and `int(q_offset)`; in the attention function `bool(mask.all())` when the 2D mask marker is lost (one sync per layer, e.g. Gemma 3 with static cache); the FP16 range check for BF16/FP32 inputs (`amax`, one sync per call on the kernel path); the reading of segments (one sync per forward) [R10] | M |
| P6.6 | ~~`mma.sync` on sm_75~~ → merged into P2.8 | — |
| P6.7 | **Llama 4: chunked attention** (block-local attention; transformers raises an error for backends with "flash" in the name, so an adaptation in the HF registry is also needed), after P4.1 window in the kernel | M |
| P6.8 | **Decode beyond 8 rows per KV head.** **Steps a, b done** (8af1c40: key splits in the forward behind `flash_attn_with_kvcache` for R > 8; 4b98060: HF short queries on it; step b: packed GQA rows, split count re-measured; a 16-token verification on a 16k cache: V100 1202 -> 181 us, T4 592 -> 296, P100 7547 -> 1184; HF prompt lookup ours fastest on every GPU; step c: GQA > 8:1 decode in HF on it (FP16, no padding); beyond 8 rows, done since: `cache_leftpad`, sinks with packed rows, `key_mask` (HF padding on the kernel), paged caches, BF16 (the CUDA-core body read as FP32 on every GPU: equal to FP16 on Pascal, 1.1-2.6x faster than before on SM 7.x for verifications, MQA decode 7-30x faster than SDPA); open refinements: BF16 on the tensor-core bodies of SM 7.x, a per-block key-mask bitmask, a direct one-split write on short caches). **Design reviewed** (`docs/analysis/kernels/spec_decode_design.md`, 2026-10-23): measured, a verification step on the P100 costs ~30x a one-token decode (4k cache, q_len 2: 1925 vs 64 us) and HF assisted generation / prompt lookup run 20% / 41% slower than plain decoding with our backend (T4: ~2x left at 16k); plan: split-KV, then packed GQA rows, in the forward bodies behind `flash_attn_with_kvcache` for R > 8. Today the decode kernel handles up to `DECODE_MAX_ROWS` = 8 rows ((H_q / H_kv) · q_len) and HF uses it only at q_len = 1 (`integrations.py`): MQA or GQA > 8:1 models go to SDPA in decode, and the verifications of assisted / speculative decoding (q_len 2–20 on a long cache) go to the forward, which with few query rows launches few blocks and does not split the keys. First, a measurement: assisted generate (e.g. Llama 8B with a small draft, prompt lookup) and an MQA model, kernel / combine against SDPA. Then more rows per block or more blocks per KV head (registers and shared memory, Pascal included), q_len > 1 in the HF decode path. Also useful for serving (speculative verifications in vLLM, decode batches with q_len > 1) | M |
| P6.9 | **Serving via transformers** (continuous batching: `generate_batch`, `init_continuous_batching`, `transformers serve`; vLLM is out of the main path, its feasibility study stays as a reference: `docs/analysis/vllm_feasibility.md`). transformers 5.x accepts any implementation whose name contains "flash" and calls it with a `PagedAttentionCache`: decode-only batches pass `block_table` + the paged `k_cache` / `v_cache` and expect the kernel to write the new keys (`flash_attn_with_kvcache(q, k_cache, v_cache, k, v, ...)`), other batches get each sequence's keys packed by `cache.update` (varlen, q shorter than k per sequence), both inside CUDA graphs. Before: every request failed with our backend (`cu_seq_lens_k` per layer type). **Step a done**: `flash_attn_with_kvcache` takes `k` / `v` (a `kvcache_append` op: new keys written at cache_seqlens, paged too; rows with a negative cache_seqlens, the padding sequences of the CUDA graphs, neither written nor attended), the HF integration routes the paged cache (varlen / decode with append), and `register()` opens transformers' decode fast path to our name (offered to FA2 / FA3 names only; without it every decode step gathers each sequence's keys out of the cache; transformers keeps it off for sliding-window models); generated tokens are FP32's greedy choice within fp16 / bf16 noise (Llama, Gemma 2, gpt-oss, Mistral; eager and CUDA graphs; Qwen2.5-0.5B: 22 / 24 requests equal plain greedy, the 2 others diverge on fp32 near-ties of 0.01-0.02 logits). Next: step b, throughput of `generate_batch` against SDPA and `paged|eager` on 5 GPUs (concurrent requests, short / long prompts, how far each fits in memory); then, only if measured useful: prefill reading K / V from the paged cache (no gather in `cache.update`), one grid for mixed prefill + decode batches, the append inside the attention kernel | M |

---

## 3b. Temporary debt (to remove)

Provisional solutions introduced during P0, each with the entry that replaces it. An entry is closed only when the provisional code is **removed**, not merely placed alongside.

| Provisional | Introduced in | Replaced by | Closed when |
|---|---|---|---|
| ~~Eager fallback (FP32, in query blocks, without tensor cores, inference-oriented) for `softcap`~~ **resolved by P4.2** (remains only for decode with softcap) | P0.3 | P4.2 softcap in the kernel, plus P4.1 window (Gemma 2's sliding layers today also go to eager) | Gemma 2 on the kernel in all layers; the `softcap` reason disappears from the registry |
| ~~Eager fallback for the `s_aux` sinks~~ **resolved by P4.3** (remains only for decode with sinks) | P0.3 | P4.3 sinks in the kernel, plus P4.1 window (gpt-oss alternates sliding layers) | gpt-oss on the kernel in all layers; the `s_aux` reason disappears from the registry |
| ~~Eager fallback for `position_bias`~~ **resolved by P4.4**: SDPA with the bias in the mask (the T5 family uses it in all layers; the kernel with additive bias and its gradient remains a possible extension, P6) | P0.3 | P4.4 | No registry reason with action `eager` |
| BF16/FP32 training → SDPA fallback (`non_fp16_training`) | P0.5 | P6.2 backward with dO scaling | No `non_fp16_training` reason; BF16 gradients within noise even with loss × 1e-6 |
| `_eager_attention` itself: no registry reason has the `eager` action any more; after P6.1 (decode on the kernel) it remains only for softcap and sinks in the real fallbacks (FP32, unsupported head_dim, masks with holes) | P0.3 | P6.1 decode with softcap and sinks (done) | To be decided: removing it requires softcap/sinks also in those cases |
| ~~`cu_seq_lens` raises `NotImplementedError`~~ | P0.1 | P0.4 | **Closed in P0.4**: varlen in the kernel or segment mask in the fallbacks; `raise` remains only for inconsistent `cu_seq_lens` |
| ~~Varlen backward as a Python loop per segment~~ **resolved by P2.4** (one launch) | before P0, used by P0.4 | P2.4 native varlen backward | A single kernel launch, no host synchronization per segment. Already today faster than the block SDPA fallback: V100 7.6–15.9 ms against 24, 1080 Ti 36–267 ms against 5260, at 8192 tokens |
| ~~Vision encoder with head_dim 72/80 (Qwen2-VL, SigLIP) → segment fallback~~ **resolved by P2.7 and P3.3** | P0.4 | P2.7 new head_dims | Qwen2-VL and SigLIP vision encoders on the varlen kernel |
| ~~Padding (2D mask with zeros or 4D mask) → SDPA fallback~~ **resolved by P3.1** (the fallback remains for holes inside a row) | before P0 | P3.1 native padding | Padded batch on the kernel in all layers |
| ~~Sliding window → SDPA fallback~~ **resolved by P4.1** | before P0 | P4.1 window in the kernel | Gemma 3, Mistral v0.1 and Phi-3 without an `attention_mask` reason due to the window |
| ~~Query-block loop of the backward duplicated between `BwdWmma` and `BwdMma`~~ **resolved**: `bwd_q_loop` and `bwd_p_ds` shared by the three backwards, single `band_barrier` in `flash_attn_common.h`. It was: loop duplicated between `BwdWmma` and `BwdMma` (loads of Q/dO/L/D, mask and ScoreMap per element, dQ step), and `band_barrier` repeated in the Pascal forward and the Turing backward | P7.2 | One shared loop like `fwd_band_loop` (body per family), a single band barrier in `flash_attn_common.h` | A single copy of the loop and the barrier; tests green on all GPUs |
| ~~Guard on the T4 (tiles that don't fit → fallback)~~ **resolved by P2.2/P2.3**: tiles chosen at runtime with `with_first_fitting_tile`, no fallback reason for shared memory | P0.6 | P2.2/P2.3 runtime tiles | No shared-memory fallback on T4 |

## 4. Test strategy

- **Kernel** (`tests/test_flash_attn.py`): FP32 reference and current tolerances; parametrized on `FORCE_ARCH` in the test build.
- **HF integration** (`tests/test_hf_equivalence.py`):
  - eager FP32 reference, with eager FP16 as background noise;
  - count of kernel calls and reason for each fallback;
  - model matrix as a slow test.
- **Training**: loss over 50 steps and 1-cos of the gradients against SDPA mem-efficient FP16.
- **Benchmark**: P1.1 harness, before and after each phase.
- **Architecture matrix**, for each phase:

| | V100 (sm_70) | Forced Pascal on V100 | T4 (sm_75) | GTX 1080 Ti (sm_61) | P100 (sm_60) |
|---|---|---|---|---|---|
| Correctness | ✓ | ✓ (test build) | ✓ | ✓ | ✓ manual (user) |
| Performance | ✓ | — | ✓ | ✓ | ✓ manual (user) |

The P100 (sm_60, fast FP16) is supported, but the user tests it manually: for each phase touching Pascal a test and benchmark script must be prepared to be run in one go. Pascal optimizations are chosen at runtime per SM, so they do not silently worsen sm_60. The T4 was later added to the matrix; an RTX 2080 Ti (sm_75) ran the end-to-end benchmarks and the full test suite.

---

## 5. Decisions

**Approved by the user (2026-10-21)**:
- All the proposals below: rename now with a deprecated alias, padding as in P3.1, P6 priorities as listed, one branch per phase.
- The **P100 is supported**: manual test by the user on their own hardware.
- **Method**: one change at a time; each change is followed by tests (on the available architectures) and a code review before moving on to the next.

Original text of the proposals:

1. **Backend name.** *Superseded: decided to keep the name "flash" (see P0.9).* The review proposes renaming right away (e.g. `legacy_attn`), because until d=72 and d=80 (P2.7) arrive the main VLMs cannot use the kernel anyway. The rename brings:
   - no monkey-patch;
   - B5 closed for free, because the VLMs go through HF's per-image loop;
   - Llama 4 unblocked;
   - compatibility with `fullgraph`.

   We would return to the `cu_seq_lens` path (name with "flash") with P3.3. The alternative is to keep the name and handle `cu_seq_lens` already in P0.4. **Proposal: rename now**, keeping `flash_attn_legacy` as a deprecated alias, and re-evaluate in P3.3.
2. **Padding**: unpad with the HF helpers (private: pin the version or copy them, about 150 Apache-2.0 lines), key padding in the kernel, or both (proposal in P3.1).
3. **Hardware**: V100, T4 and GTX 1080 Ti. Is a P100 also needed?
4. **Priority of P6**.
5. **Workflow**: one branch per phase with PR and review; the current uncommitted changes go into a first base commit.
6. **Order**: Turing (P2.2–P2.3) before native padding and packing (P3). The plan already foresees this. P0.6 makes the T4 safe right away.
