# P8 follow-ups on the P100: where the time goes, and what to change next

Status: analysis and proposal (2026-11-02), revised after an independent review; to be decided before any
implementation. Follow-up 1 (fold every 64 K elements) is done and measured (`docs/bench/linear.md`, "Fold interval").
This note covers the other three:

- 2: the backward of the linear layers: dX without the W^T copy (2a); fp32 weights under autocast and dW on the
  kernel (2b);
- 3: attention's QK^T and PV on HFMA2;
- 4: a hand-scheduled SASS main loop for the linear kernel.

Everything below is for compute capability 6.0 only (Tesla P100). Every other GPU keeps its current code paths, and so
do sm_61 and the `FLASH_ATTN_LEGACY_FORCE_ARCH=pascal` path on newer GPUs.

## 1. Where the time goes today

`prof_split.py` (scratch, torch.profiler) measures the summed CUDA kernel time of one step, after two warm-up steps, on
mew's P100 with `patch_linear` on. The setup:
- our attention kernels, C = 64, fp16 autocast;
- gradient checkpointing for the training steps;
- "LoRA": rank 16 on q/k/v/o, fp32 adapters;
- "full": every weight trainable **in fp16**. This is a profile of the kernels, not a training recipe (see 2b).

The percentages are of kernel time, not wall-clock time. In host-bound steps the end-to-end gains are smaller.

| case | CUDA ms | linear kernel | attention | copies | cuBLAS GEMM | other |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B prefill 4k | 411 | 54.5% | 34.5% (fwd) | 3.5% | - | 7.5% |
| Qwen2.5-1.5B prefill 4k | 1283 | 62.5% | 30.6% (fwd) | 2.3% | - | 4.7% |
| Qwen2.5-1.5B LoRA 2k | 2310 | 58.9% | 25.3% (bwd 15.8%, fwd 9.5%) | 7.3% | 1.9% | 6.6% |
| Qwen2.5-0.5B full 2k | 1148 | 36.7% | 28.4% (bwd 21.3%, fwd 7.0%) | 5.0% | 22.5% | 7.5% |
| Qwen2.5-1.5B full 1k | 1483 | 46.0% | 14.6% (bwd 10.1%, fwd 4.4%) | 7.6% | 26.4% | 5.4% |

What is in the buckets:

- **Copies, W^T for dX.** The W^T copy of dX is a non-vectorised strided `direct_copy`, 101 ms in the LoRA step
  (4.4%), at ~61 GB/s (8% of the card's bandwidth).
- **Copies, everything else.** The remaining copies are `torch.cat` of rotary embeddings, other `direct_copy`s and
  autocast's fp16 casts. They also appear in prefill, which has no W^T.
- **cuBLAS GEMM, full.** This is all dW = dY^T X: cuBLAS `hgemm_*_TN_vec` for the layers and `maxwell_fp16_sgemm` for
  the tied lm_head, at ~7.8-8.0 TFLOPS. A probe (`dw_probe.py`) gives 6.6-8.5 TFLOPS on layer shapes and a relative
  error of 2.1e-4 against FP64, the same as stock forward (FP32 accumulation).
- **cuBLAS GEMM, LoRA.** Here this bucket is the fp32 adapters.

Upper bounds of what each follow-up removes, from kernel time:

| follow-up | LoRA step | full fine-tune step | prefill |
|---|---|---|---|
| 2a dX without W^T | ~3.5% (the copy, minus ~1% kernel cost) | ~3-6% | - |
| 2b dW on the kernel (planning 1.5x; up to ~1.7x) | - (base frozen) | ~7-11% | - |
| 3 attention fwd on HFMA2 (assume 1.5x, unproven, see 3) | ~3% | ~1.5-2.5% | ~10-12% |
| 3' attention bwd on HFMA2 (assume 1.5x) | ~5% | ~3-7% | - |
| 4 SASS main loop (assume +10% on the linear kernel) | ~5% | ~3-4% | ~5-6% |

## 2. Follow-up 2: the backward of the linear layers

### 2a. dX without the W^T copy

**Today.**
- dX [M, K] = dY [M, N] · W [N, K], reduced over N.
- dY is N-contiguous. W is K-contiguous, and the kernel pairs adjacent elements of the reduction dimension in one 32-bit
  word, so it receives `W.t().contiguous()`.
- That costs 2·N·K bytes moved per call (4.4% of the LoRA step).
- It also needs a transient of N·K·2 bytes. For the lm_head of Qwen2.5-1.5B (151936 x 1536) that is 467 MB, on a 16 GB
  card.

**Baseline alternative.** A tiled transpose kernel instead of torch's strided copy would recover most of the 4.4% with no
GEMM change, but it keeps the 467 MB transient. 2a removes both at the same cost (< 1%), so it is preferred. The tiled
transpose is the fallback if 2a's staging turns out slower than measured here.

**Change.** Add a B-operand layout where the reduction dimension is the slow one ([reduction, outputs] row-major).

- **Global loads.** These stay 16-byte vectors along the outputs. Each thread loads 2 rows x 8 outputs per stage
  (kLoads = 2 at 256 threads). The thread mapping is `tid & 15` → column group, so each row is a coalesced 256-byte
  segment.
- **Shared store.** It pairs the two rows with `__byte_perm` (0x5410 / 0x7632): word c = (W[2kp][c], W[2kp+1][c]), which
  is exactly the word today's transposed copy produces.
  - The stores are 2x STS.128 per thread.
  - Their 2-way bank conflict matches today's STS.32 pattern, which is already 2-way, so it is no regression.
  - `stage()`, the fold and the epilogue are unchanged.
- **Kernel.** A template parameter (`kBRowsK`) on `linear_kernel`.
- **Host.** The kernel is reached through a layout flag on `torch.ops.flash_attn_legacy.linear`, or a sibling op with its
  own `register_fake`. `_grad_input` passes W as is.
- **Constraints.**
  - N % 8 == 0 is already required for dX on the kernel, and it keeps reduction rows in whole pairs and stages.
  - K % 8 == 0 and 16-byte-aligned rows for the loads along K, as in the forward.
  - The reduction tail (N not a multiple of 32) zero-fills per row, as the transposed copy's K tail does today.

**Acceptance.**
- dX is bitwise equal to today's path on the 98 edge shapes and the real layer shapes (same words, same chain, same
  folds).
- The dX GEMM's kernel time is ≥ 0.98x of today's kernel-only time, and the copy is gone from the profile.
- The LoRA step is ≥ 1.03x on Qwen2.5-1.5B at 2k (wall clock, A/B alternated).
- Peak memory of `_grad_input` on the lm_head, measured in isolation (peak reset before the call), is lower by N·K·2
  bytes.
- `ptxas -v` reports 0 spills.
- opcheck / `torch.compile(fullgraph=True)` pass for the new op or flag.
- The concurrent-streams and CUDA-graph tests pass with the new layout.
- Non-sm_60 GPUs are unchanged (the full suite on the four other boxes).

### 2b. fp32 weights under autocast, then dW on the kernel

**Who it is for.** `patch_linear` takes only fp16 weights, and `_fallback_reason` sends any other W to `F.linear`. The
common full fine-tuning recipe is autocast fp16 with fp32 weights (HF Trainer `fp16=True`, `torch.amp.GradScaler`), so
today **none** of it reaches the kernel, in either the forward or the backward. Pure fp16 weights are used only with an
external master copy and its own loss scaler (DeepSpeed fp16, apex O2); `GradScaler` refuses fp16 gradients.

So 2b has two steps:

- **2b.1 fp32 weights under autocast.** When autocast to fp16 is on and W is fp32, the op takes a cached fp16 cast of W,
  as autocast already does for `F.linear`. Then:
  - dX goes on the kernel (with 2a);
  - dW comes back through the cast as fp32, as stock does.

  This puts the forward and dX of the standard recipe on the kernel. It is probably a larger gain than dW itself. It
  needs its own profile ("full" with fp32 weights) before any estimate is written down.
- **2b.2 dW on the kernel.** dW [N, K] = dY^T [N, M] · X [M, K], reduced over M (the tokens). Both operands are M-major,
  so both use the 2a staging.
  - **Reduction tail.** M is arbitrary (odd, packed varlen batches), so the staging needs per-row predicates. Row r+1 is
    zero-filled when r is the last valid row.
  - **Output dtype.** The output is fp32 when W is fp32 (2b.1), which saves the cast back, and fp16 for fp16 weights.
  - **Small outputs.** The k/v projections (e.g. 128 x 896 = 7 output tiles on 56 SMs) rely on stream-K over M, so the
    planning figure is 1.5x, not the forward's 1.7x.

**Overflow (to measure before designing 2b.2).**
- dY carries the loss scale (2^16 at the start).
- A single fp16 product dY·X, or a 32-term lane, can overflow where cuBLAS's FP32 accumulation does not. The new risk is
  specifically a partial sum that overflows and would have cancelled later.
- Activations of Qwen-class models reach 1e2-1e3 in the down_proj inputs (massive activations).
- A power-of-2 prescale of dY is exact only above fp16's subnormal range, so it trades overflow for underflow, which is
  what loss scaling is there to prevent.
- First step: a histogram of max |dY·X| and max |lane partial| on real layers under GradScaler.

**Acceptance (2b.1 and 2b.2).**
- dW error against FP64 ≤ 2.5e-3 relative Frobenius on edge shapes, including M odd and M % 32 != 0.
- A 300-step full fine-tune of Qwen2.5-0.5B with the standard recipe (autocast fp16, fp32 weights, GradScaler):
  - validation loss within the two-seed noise of stock;
  - skipped (inf) steps ≤ stock's.
- Full fine-tune step ≥ 1.07x on Qwen2.5-0.5B at 2k and Qwen2.5-1.5B at 1k (wall clock, A/B).
- opcheck, compile, streams, CUDA graph, 0 spills, and non-sm_60 unchanged, as in 2a.

## 3. Follow-up 3: attention on HFMA2

**Today.**
- The Pascal forward (`csrc/kernels/flash_attn_fwd_pascal.cu`, `PascalFwd::run`) computes S = QK^T and O += PV with
  fp32 FMA, from Q/K/V converted to fp32 in shared memory.
- At d = 128 the tile is `FwdTileSimt<32, 8, 128>` (Bc = 8 keys), 26 KB of shared memory. Shared memory is not what
  limits Bc; the registers are (64 fp32 O values per lane at d = 128).
- The same code also runs on sm_61, on newer GPUs under `FORCE_ARCH=pascal`, and on bf16 caches. All of those stay on
  fp32.

**Why it is not obviously a win.**

- **The fold dominates at small key blocks.** A half → float conversion costs about 4 FMA slots on sm_6x, so a fold
  (2 conversions + adds) is ~9-10 FMA-slot equivalents.
  - PV paired along keys with a fold per block adds ~56% at Bc = 32 and ~110% at Bc = 16, which would be slower than
    fp32.
  - S at d = 128 (two chunks of 64 per score) adds ~28%.

  Two layouts avoid most of this, and the static SASS count must pick between them before any claim of a gain:
  - **PV paired along output columns.** P is broadcast as half2(p, p) and V stays in its natural row-major fp16. This
    halves the half2 accumulator registers and roughly halves the fold cost.
  - **A half2 O accumulator rescaled by the online-softmax alpha (≤ 1)** and folded every 64 keys across blocks,
    independently of Bc.

- **exp2 turns S's absolute error into P's relative error.**
  - An fp16 lane holding |s| ~ 16-32 (log2 domain: attention sinks, peaky heads) has an ulp of 2^-6 to 2^-5.
  - Over 32 roundings that can reach an absolute error of ~0.02-0.1, i.e. 1.5-7% on the heaviest P weights.
  - This differs in kind from the linear kernel, whose error is relative to its output.

**Gate before any kernel work (3.0).** A PyTorch emulation of the chunked fp16 QK^T (pairs along d, fp16 lanes of 32
products, fp32 fold) and of the proposed PV on real q / k / v. The q / k / v are captured from Qwen2.5-0.5B / 1.5B,
TinyLlama and Qwen3-1.7B on wikitext-2, with their real scales and softcap where present.

It measures:
- the error of S, P and O against the fp32 path;
- the maximum |lane| before the fold (the overflow margin);
- top-1 and perplexity with the emulated attention swapped in.

3 goes ahead only if the top-1 flips and perplexity are at the stock-FP16 noise floor, with the margins of P8's
`--accuracy` table.

**Design, if the gate passes** (sm_60 fp16 inputs only):

- **Q, K and V.** They stay fp16 in shared memory. Q is prescaled by softmax_scale·log2(e) (≈ 0.128 at d = 128) before
  the fp16 store.
  - With softcap, the prescale is only scale·log2(e). `ScoreMap`'s multiplier is adjusted so that it is not applied
    twice, and small q are not pushed into subnormals by the 1/softcap factor.
- **Tiles.** These are derived from a register count (the O accumulator layout chosen above), with Bc ≥ 32 at d = 128,
  and shown with `ptxas -v` before the gain is estimated.
- **Head dims.** 64 and 128 first. The d = 256 split tile (32 dims per split, half a chunk), d = 32 and d = 96 stay
  fp32 until measured.
- **Overflow.**
  - The check runs on the raw folded S, before masking: masking writes -inf, so an overflowing lane must never be
    mistaken for a masked key.
  - A block that sees a non-finite folded score while its inputs are finite recomputes its rows on the fp32 path,
    inside the kernel.
  - This needs no host sync, so it is valid under CUDA-graph capture, deterministic, and safe across concurrent streams.
  - A device counter, read lazily by `fallback_stats()`, records it.
  - Non-finite inputs behave as today (no recompute).
- **Training consistency.** The Pascal backward recomputes S in fp32 but uses the forward's LSE and O. With the HFMA2
  forward, exp2(S_fp32 - LSE) no longer sums exactly to 1, and D = rowsum(dO∘O) uses the HFMA2 O. Until 3' lands, the
  HFMA2 forward is either:
  - inference-only (`torch.is_grad_enabled()` false or no saved state), or
  - shipped with a gradient-accuracy acceptance for "forward HFMA2, backward fp32".

  Gradient-checkpoint recompute must reproduce the forward bitwise, the in-kernel fallback included.

**Acceptance.**
- The 3.0 gate passes.
- Forward error against the fp32 reference ≤ 2x the current Pascal forward's on the existing grid, or the measured stock
  FP16 floor if that is tighter than reachable.
- Top-1 and perplexity at the noise floor on the four `--accuracy` models.
- Forward ≥ 1.3x at d = 64 / 128 on 1k-8k; prefill ≥ 1.08x end to end (A/B alternated).
- An adversarial Q·K test triggers the in-kernel fallback, and the fallback counter shows it.
- CUDA-graph, concurrent-stream and run-to-run determinism tests pass, with `ptxas -v` 0 spills.
- sm_61, `FORCE_ARCH=pascal` and bf16 caches are bitwise unchanged.

**3' (backward).** It reuses the scheme for dS, dV, dK and dQ, with its own emulation gate and accuracy run. Gradients
reduce over many keys or queries, and the per-block fold still applies.

## 4. Follow-up 4: hand-scheduled SASS main loop

**Today.** The kernel reaches 64-72% of the HFMA2 peak from M = 512. cuBLAS's `maxwell_hgemm` (COMPUTE_16F) reaches
~89%, but it has no fold. The fold at C = 64 costs ~256 instructions per 2048 HFMA2, so a folding kernel's ceiling is
about 89% / 1.125 ≈ 79% of peak. The possible gain is therefore 1.10-1.23x on the GEMM; the planning figure is +10%.

The prototype's static count places the losses in:
- dual-issue of LDS.128, STS and the address math (~90 extra issue slots per 1024 HFMA2);
- ~65 register-bank conflicts that a post-ptxas renaming cannot remove (the renamer, +1.6%, was not shipped);
- one barrier per stage with 8 warps resident.

**Change.** Write the main loop (stage loads, the 8 x 8 HFMA2 block, the fold) in SASS. The register allocation is
explicit, with bank-conflict-free operand reuse, and so are the dual-issue pairing and the stall/yield control codes.
Stream-K and the epilogue stay as they are.

**Tooling: a decision for the maintainer.** sm_60 uses the Maxwell/Pascal encoding family. Any assembler must first be
verified on two points:
- that it accepts sm_60 cubins;
- that it encodes **HFMA2, HMUL2 and F2F.F32.F16** with correct control codes.

The original maxas grammar targets sm_5x and, as far as known, lacks the fp16 instructions. Nervana's fp16 kernels used
fp32 math on fp16 storage, not HFMA2.

| option | build | repo | risk |
|---|---|---|---|
| A. prebuilt cubin, assembled offline, loaded with `cuModuleLoadData` | no new build dependency | the cubin + its SASS source + the assembly script | not rebuildable without the assembler; kept in sync by hand |
| B. assembler run by `setup.py` for sm_60 | the assembler at build time | SASS source only | build fragility wherever sm_60 is built |
| C. our post-ptxas patcher (`regperm.py`) extended from renaming to rescheduling | Python only | the patcher | bound by ptxas's allocation; rescheduling ptxas output is harder than writing the loop |
| D. CuAssembler (Python; its sm_6x support to be verified) as the assembler for A or B | Python only | as A or B | coverage of HFMA2 and control codes unverified |

**Acceptance.**
- Outputs are bitwise equal to the CUDA kernel's on every test shape.
- GEMM ≥ 1.08x on the 80 shapes (geomean, A/B alternated).
- `tests/test_linear.py` passes; the build on non-sm_60 GPUs is unchanged.

## 5. Proposed order

1. **P8.6 (2a)**: dX without the W^T copy. It is small, bitwise-verifiable and removes a 467 MB transient.
2. **P8.7 (3.0)**: the emulation gate for HFMA2 attention. It is cheap (PyTorch only) and decides whether 3 exists.
3. **P8.8 (2b.1)**: fp32 weights under autocast. It brings the standard full fine-tuning recipe onto the kernel. It is
   profiled first, then estimated.
4. **P8.9 (3)**: the HFMA2 attention forward, if the gate passed.
5. **P8.10 (2b.2)**: dW on the kernel, after the overflow histogram.
6. **P8.11 (3')**: the HFMA2 attention backward.
7. **P8.12 (4)**: the SASS main loop, after the tooling decision (A-D) and the assembler verification.

Each one follows the usual process: analysis, plan, review, implement. Tests run on all boxes and get an independent
review before the next one, and perf claims are confirmed with alternated A/B against a HEAD build.

## Sources

- **Profile and probe:** `prof_split.py` and `dw_probe.py` (scratch scripts, not in the repo), run on mew's P100 on
  2026-11-02, with torch 2.7.1+cu126 and transformers 5.20.2.
- **Linear kernel, its API and measurements:** `csrc/kernels/linear_pascal.cu`, `flash_attn_legacy/linear.py`
  (`_patchable`, `_fallback_reason`, `_grad_input`), `docs/bench/linear.md`, `docs/PLAN.md` P8.
- **Pascal attention forward:** `csrc/kernels/flash_attn_fwd_pascal.cu`, `csrc/include/flash_attn_common.h` (tiles,
  `ScoreMap`).
- **Static SASS analysis and the register renamer:** the P8 prototype work on mew (`~/fa-legacy-test/pascal_probe/`,
  2026-10-29/30).
- **Chunked accumulation:** IBM, arXiv 1812.08011; the DeepSeek-V3 technical report (FP32 promotion every 128 elements).
- **Independent review of this note (2026-11-02):** all profile figures re-derived and confirmed; its findings on 2b's
  target recipe, the attention fold cost and error, the fallback design and the SASS ceiling are folded in above.
