# The chunk gated delta rule of Qwen3.5-class models on pre-Ampere GPUs (P6.10 c)

Status: analysis and plan for the first step (2026-11-03), reviewed; section 3 revised after the review. Last of the order
decided on 2026-11-03: (a) the `fla` guard, done; (b) the causal conv1d, done; (c) this.

## 1. Today

Without `fla`'s kernels (always below SM 8.0, the guard of P6.10 a), transformers 5.20.2 runs its PyTorch reference
`torch_chunk_gated_delta_rule` for prefill and training. Decode runs `torch_recurrent_gated_delta_rule`, or `fla`'s
recurrent kernel when `fla` is installed (the guard keeps it).

The reference, per call (inputs [B, L, H, K|V], H value heads with the key heads repeated, chunks of 64 tokens):
1. **Prologue.** fp32 casts and transposes, l2norm of q and k, the query scale, pads to a multiple of 64.
2. **Intra-chunk phase, parallel over the chunks.**
   - The pairwise decays exp(cum_i - cum_j), masked.
   - The UT system (β k)·kᵀ ⊙ decay.
   - The intra-chunk attention q·kᵀ ⊙ decay.
   - Two batched 64x64 unit-lower-triangular solves (`solve_triangular`) for the "new values" u and the decayed keys
     w.
3. **Scan, sequential over the chunks.** A Python loop with four small matmuls per chunk, on a state S [B, H, K, V] in
   fp32:
   - v_new = u - w·S;
   - out = q̃·S + A·v_new;
   - S = S·exp(g_chunk) + k̃ᵀ·v_new.
4. **Epilogue.** The output back to [B, L, H, V] in the input dtype.

Qwen3.5-0.8B / 2B have 16 key and 16 value heads, 4B has 16 key and 32 value heads, all with K = V = 128; 18 / 18 / 24
such layers.

## 2. Where the time goes

Measured with `qwen35_delta_phases.py` in this directory: a copy of the reference with CUDA events between its phases,
B = 1, H = 16, K = V = 128, fp16 inputs; median of 5, one layer.

| GPU | L = 4096, total | prologue | intra-chunk | scan | forward + backward (autograd), L = 4096 |
|---|---|---|---|---|---|
| P100 | 14.7 ms | 16% | 28% | 53% | 21 + 231 ms |
| V100 | 19.4 ms | 7% | 11% | 81% | 37 + 171 ms |
| 2080 Ti | 25.7 ms | 7% | 15% | 77% | 53 + 235 ms |
| T4 | 21.7 ms | 20% | 37% | 40% | 48 + 528 ms |

The shares are the same at L = 1024 and 16384 (time is linear in L).

- **The scan is host-bound.** It makes ~6 launches of tiny kernels per chunk, 64 chunks at 4k. Measured in isolation,
  the V100 spends 18.5 ms per call; inside the model's prefill it was 8.1 ms per call (`qwen35_conv1d.md` section 1),
  because there the host runs ahead of the GPU while the big kernels of the other layers execute.
  - Either way, one kernel for the whole scan removes both costs: the launches, and the low occupancy of 16 heads x 64
    x 128 x 128 matmuls.
- **In the model** this function is 13% of Qwen3.5-2B's 4k prefill on the P100, 39.5% on the V100 and 43% on the T4.
  It is 5% of the P100's LoRA step at 2k. (`qwen35_conv1d.md` section 1, measured with the reference conv.)
- **The backward through autograd is 8-24x the forward**: the 64-step loop's graph plus `solve_triangular`'s backward.
  It dominates training of these models on these GPUs.
- **Decode** (the recurrent reference) costs 0.27 / 0.48 / 0.79 / 0.53 ms per token and layer on the P100 / V100 /
  2080 Ti / T4, host-bound. `fla`'s recurrent kernel, through the guard, already runs on all five GPUs.

## 3. Plan: three steps, each measured before the next

- **(c1) The scan as one CUDA kernel, forward only (inference and prefill).** The prologue and the intra-chunk phase
  stay in PyTorch, unchanged. Under autograd (training), the reference runs as today.
- **(c2) The intra-chunk phase fused.** The pairwise decay, the UT system, the 64x64 triangular solve and the
  intra-chunk attention per chunk, in one kernel, plus the prologue's l2norm and scale.
- **(c3) The backward**: a reverse scan with the state's gradient, plus the intra-chunk phase's backward. This is the
  large training win, but also the largest piece of work. It is planned after c1 / c2 have fixed the forward's layout.

**c1 in detail** (revised after the independent review, 2026-11-03).

- **Split like `fla`.** A sequential kernel computes only the recurrence, per chunk c (state S_c at the chunk's
  start):
  - v_new_c = u_c - w_c·S_c;
  - S_{c+1} = d_c·S_c + k̃_cᵀ·v_new_c;
  - it stores every S_c (h [B, H, NC, K, V] fp32) and v_new.

  The output, out_c = q̃_c·S_c + A_c·v_new_c, is then two batched matmuls, parallel over the chunks (cuBLAS).
  - This puts ~57% of the per-chunk work on the critical path instead of all of it.
  - It is the layout c3's backward needs (the per-chunk states).
  - Cost: h is a transient of 64 MB per layer at 4k (B = 1, H = 16), 1 GB for the 4B model at 32k.
- **Grid.** One block per (batch row, head, slice of VS columns of V). The state's columns are independent (exact,
  initial state included), so slicing V is exact. VS = 32, or 16 when B·H·V/32 is below the SM count (P100 / T4 /
  1080 Ti at B = 1).
- **Shared memory: ≤ 48 KB static on every arch** (the review found the first plan's ~136 KB fits none).
  - The state slice (K x VS fp32, 16 KB at VS = 32) and v_new (64 x VS, 8 KB) stay in shared memory.
  - w_c and k̃_c are streamed in K sub-tiles of 32 columns (64 x 32 fp32, 8 KB, padded against bank conflicts):
    v_new accumulates over the sub-tiles, then the state rows of each sub-tile are updated.
  - 256 threads, register double-buffering of the next sub-tile (no `cp.async` before Ampere).
- **The rest stays PyTorch**, copied from the installed reference: the prologue, the intra-chunk phase (u, w, A), q̃,
  k̃, the decays, and the output's matmuls.
  - A test checks that the copy, with a Python scan, is bitwise the installed reference on the same device.
  - The route is gated on a fingerprint of the reference's source, recorded for transformers 5.20.2. Another
    version means the reference, with a one-time warning.
- **Routing**: one wrapper over (the bound implementation, the reference), installed whenever transformers' wrapper
  is recognised, which replaces P6.10 (a)'s fla-only install. It decides in this order:
  1. packed sequences below SM 8.0 raise;
  2. `fla` bound and SM ≥ 8.0 (or `FLASH_ATTN_LEGACY_FLA_CHUNK=fla`): `fla`;
  3. our scan, when all of these hold:
     - CUDA;
     - `FLASH_ATTN_LEGACY_GDN_SCAN` is not 0;
     - chunk_size 64 and `use_qk_l2norm_in_kernel`;
     - K in {64, 128}, V a multiple of 16;
     - the reference fingerprint matches;
     - no gradient: not (grad enabled and any input or the initial state requires grad);
  4. otherwise the reference, with its named keywords only.

  The training route (gradient → reference) is not counted as a fallback until c3.
  - Under reentrant checkpointing the first forward runs our scan and the recompute runs the reference: equal up to
    fp32 noise, documented.
  - The op raises if it is ever reached with inputs that require grad.
- **Expected**, corrected:
  - the scan from 40-81% of the function to ~5-10%;
  - the function ~1.35x faster (T4), ~1.6x (P100), ~3-4x (V100 / 2080 Ti);
  - Qwen3.5-2B 4k prefill ~1.1-1.15x (T4), ~1.3-1.4x (V100);
  - on the P100 and T4 the intra-chunk phase then dominates, which is c2's target.
- **Training share**, measured on the P100: the chunk rule's forward + backward at 2k is ~72 ms per layer call, ~1.3 s
  of a 4.8 s LoRA step over 18 layers (~27%). So c3 is the training lever. The order after c1 is c3, then c2.

## 4. Acceptance (c1)

1. **Kernel against the reference's scan** (same inputs, fp32): out and final state within fp32 reassociation of the
   reference, relative Frobenius ≤ 1e-5.
   - Each error, per chunk and per head, is at most 2x the reference's against fp64.
   - The cases include slow decays (g ≈ -1e-3 and g = 0, with β near 1), where the state carries across chunks, and
     nearly parallel keys.
   - L ∈ {1, 63, 64, 65, 4096}, B ∈ {1, 2}, H ∈ {1, 16}.
   - With and without an initial state; K = V = 128 and K = 64 ≠ V = 128.
2. **The whole function against the reference**: the same within fp32 noise, then the outputs rounded to the input
   dtype.
   - The routing: no gradient → our scan; gradient → the reference, bitwise; `fla` at SM ≥ 8.0 untouched.
   - Packed sequences raise; the opt-out works; fallbacks are counted.
3. **Model** (tiny Qwen3.5, K ≠ V, our attention):
   - prefill logits within fp16 noise of the reference path;
   - greedy decode unchanged until a near-tie (the chunk state seeds the recurrent decode);
   - chunked prefill (an initial state) the same.
4. **Speed**:
   - the function at 4k on P100 / V100 / 2080 Ti / T4, ours against the reference;
   - Qwen3.5-2B prefill and generate A/B (4 rounds per arm), no regression anywhere.
5. **Suite**: `ptxas` 0 spills; opcheck / compile of the op; the full suite green on all five GPUs.
6. **Also**:
   - fallbacks for `output_final_state=False` (still our scan), for `use_qk_l2norm_in_kernel=False` and for
     chunk_size ≠ 64;
   - non-contiguous q / k / v; bf16 / fp32 inputs; H = 32 (4B);
   - B > 1 left-padded generate;
   - routing at SM ≥ 8.0 with and without `fla`, and on CPU;
   - the frozen-prefix case (grad enabled, nothing requires grad) on our scan;
   - LoRA with gradient checkpointing bitwise the reference;
   - run-to-run determinism; peak memory no higher than the reference (h counted);
   - 64-bit offsets;
   - the 1080 Ti's speed is not measurable while its GPU is shared with another tenant: correctness only there.
