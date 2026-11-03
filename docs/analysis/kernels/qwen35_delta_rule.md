# The chunk gated delta rule of Qwen3.5-class models on pre-Ampere GPUs (P6.10 c)

Status: c1, c3 and c2 done (2026-11-03): each planned, reviewed, implemented, reviewed again, measured (sections 5, 7
and 9). c2b (a fused backward kernel) open. Last of the order decided on 2026-11-03: (a) the `fla` guard, done; (b) the causal conv1d, done; (c) this.

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
  - w_c and k̃_c are streamed in K sub-tiles of 32 columns (64 x 32 fp32, 8 KB; unpadded as implemented: neither
    access pattern conflicts):
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

## 5. c1 as implemented, and the results

**Kernel** (`csrc/kernels/gdn_scan.cu`, op `gdn_chunk_scan(w, u, k̃, d, initial_state) -> (h, v_new, final_state)`).
- One block per (batch row, head, slice of VS = 16 or 32 state columns), 256 threads.
  - VS = 32 when B·H·V/32 reaches twice the SM count, else 16.
  - 75-96 registers on sm_60; shared memory 16-32 KB. (Section 7 corrects the "no spills" first recorded here.)
- The w / k̃ sub-tiles are 64 x 32 fp32, loaded as float4 into registers one tile ahead.
- Each tap's state value read from shared memory is reused for a thread's rows. The first version, with scalar
  dependent loads, ran at 7.4 ms on the P100 at 4k; the final one runs at 1.97 ms.
- A `__launch_bounds__(256, 2)` hint was tried after the review and reverted: it let ptxas take 96-124 registers,
  2 blocks per SM instead of 3, and 2.77 ms.
- The op raises if its inputs require grad.

**Python** (`flash_attn_legacy/gated_delta.py`).
- `_chunk_scan` is the reference's lines with the loop replaced. With `kernel=False` it is bitwise the installed
  reference (tested).
- On the kernel path it frees the intra-chunk intermediates before the scan and adds the two output products in
  place. Its peak memory is no higher than the reference's (tested at 2k).
- The route is pinned to the reference's source by sha1, with a warning when it differs.
- Inference calls the scan does not take are counted (`gated_delta_scan_input`); training calls go to the reference
  without a count until c3.
- Deviation from section 3: under reentrant gradient checkpointing the first forward takes the scan and the
  recompute the reference. They are equal up to fp32 rounding, not bitwise; this is documented in the module.

**Accuracy**: fp32 out / final state relF 5.5e-7 / 5.0e-7 at 4k on all GPUs measured. The tests' bound is 1e-5 per
head, over the fast, slow, flat and parallel-keys regimes, up to 4096 tokens, at K 64 / 128, VS 16 / 32 and V 48 / 128,
in fp16 / bf16 / fp32.

**Speed of the function** at 4096 tokens (`qwen35_delta_phases.py`-style timing, ours vs the reference, one layer):

| GPU | 16 heads (0.8B / 2B) | 32 heads (4B) | the scan kernel alone, 16 heads |
|---|---|---|---|
| V100 | 18.9 -> 5.6 ms (3.35x) | 18.5 -> 10.6 ms (1.74x) | 1.19 ms |
| 2080 Ti | 25.2 -> 9.1 ms (2.76x) | 26.6 -> 14.8 ms (1.79x) | 2.15 ms |
| P100 | 14.5 -> 9.7 ms (1.50x) | 22.2 -> 19.5 ms (1.14x) | 1.97 ms |
| T4 | 25.0 -> 18.2 ms (1.37x) | 36.5 -> 34.8 ms (1.05x) | 4.28 ms |

**In the model** (Qwen3.5-2B fp16 on our attention and conv, `qwen35_conv_model_ab.py NO_LORA=1`, A =
`FLASH_ATTN_LEGACY_GDN_SCAN=0`, 4 rounds per arm alternated):

| GPU | prefill 4096 | generate 2048 + 64 |
|---|---|---|
| V100 | 357 -> 307 ms (1.163x) | 2496 -> 2530 ms (0.99x, overlapping ranges) |
| 2080 Ti | 509 -> 466 ms (1.094x) | 3190 -> 3161 ms (1.009x) |
| P100 | 1867 -> 1795 ms (1.040x) | 2446 -> 2406 ms (1.017x) |
| T4 | 1144 -> 1142 ms (1.00x) | 3501 -> 3499 ms (1.00x) |

The 1080 Ti is not measurable (its GPU is shared with another tenant).

- Generate does not change beyond noise: its prefill is 2k of a 64-token decode, and decode runs the recurrent rule.
- **On the T4 the prefill does not move.** Measured inside the model in one process, its 18 chunk calls take 435 ->
  401 ms: the remaining time is the PyTorch prologue and intra-chunk phase (57% of the function there), and the T4's
  sustained clocks. That is c2's target, and the P100's too.

## 6. c3: the backward (plan, 2026-11-03)

**Today.** A call that needs gradients runs the reference, whose autograd graph holds the 64-step loop. Its
backward is 8-24x the forward (section 2): ~64 ms per layer at 2k on the P100, ~27% of a LoRA step.

**Approach.** Register autograd on `gdn_chunk_scan`, so the whole `_chunk_scan` path is differentiable.
- PyTorch's autograd keeps the parallel part: the prologue, the intra-chunk phase with its two batched triangular
  solves, and the two output matmuls. These are batched ops with no per-chunk loop.
- The op's backward is a reverse scan.
- The routing then sends gradient calls to the scan too (c1 sends them to the reference).

**The op's backward.** In the forward, per chunk c (S_c = h_c, the state at the chunk's start):

    v_c = u_c - w_c S_c,   S_{c+1} = d_c S_c + k̃_cᵀ v_c

Given dh_c, dv_c (from the output matmuls) and dS_NC (the final state's gradient), let G_{c+1} = ∂L/∂S_{c+1}, with
G_NC = dS_NC. Then, for c = NC-1 .. 0:

    dv̂_c = dv_c + k̃_c G_{c+1}                    (all of v_c's gradient: through S_{c+1} too)
    G_c  = d_c G_{c+1} - w_cᵀ dv̂_c + dh_c

and the inputs' gradients:

    du_c = dv̂_c,   dw_c = -dv̂_c S_cᵀ,   dk̃_c = v_c G_{c+1}ᵀ,   dd_c = Σ G_{c+1} ⊙ S_c,   dS_0 = G_0

- **The recurrence in G has the forward's shape.** It goes in reverse chunk order, uses k̃ in place of w and w in
  place of k̃, has signs flipped, and adds dh_c. Its columns are independent, as the state's are.
  - So it is the same kernel, with a template flag: blocks over state-column slices, w / k̃ streamed in sub-tiles.
  - It stores G_{c+1} per chunk ([B, H, NC, K, V]) and dv̂ ([B, H, NC, 64, V]).
- **The rest is batched matmuls in PyTorch, parallel over the chunks**: dw = -dv̂·hᵀ, dk̃ = v·Gᵀ, dd = Σ G ⊙ h.
  These reduce over V, which crosses the column slices, so they are not done in the kernel: no atomics, deterministic.
- **Saved for the backward**: h and v_new, the op's outputs; w, k̃ and d, its inputs.
- **Memory**: G is another [B, H, NC, K, V] fp32 transient in the backward, 64 MB per layer at 4k (B = 1, 16 heads).
  The reference's graph saves the state at every chunk too.

**Routing.** Gradient calls take the scan when their arguments are supported; the training route is no longer
counted. The reentrant-checkpointing note of c1 goes away: both passes take the scan.

**Acceptance.**
1. **Gradients against the reference's autograd, fp32**: dq, dk, dv, dg, dβ and d(initial state) within 1e-4
   relative, per head, over the four regimes of the c1 tests, at K 64 / 128 and V 48 / 128. Also with the final state
   in the loss (dS_NC), and with the output alone.
2. **The op's backward against fp64**: the Python reverse loop in fp64 (autograd of the op's math). `gradcheck` is not
   usable on an fp32-only op.
3. **Model.** A tiny Qwen3.5 (K ≠ V): a LoRA step's gradients within fp16 noise of the reference path, and a few
   optimizer steps whose losses agree. Gradient checkpointing, reentrant and not.
4. **Speed.** Forward + backward of the function at 2k / 4k on the four measurable GPUs, against the reference's.
   Qwen3.5-2B LoRA step at 1k (2k where it fits) A/B, 4 rounds per arm: expected 1.2-1.3x on the P100 (from section
   2's share), to be measured.
5. **Peak memory** of a training forward + backward no higher than the reference's.
6. **Suite.** ptxas 0 spills; opcheck with autograd registration; the full suite green on all five GPUs.

## 7. c3 as implemented, and the results

**Plan review points applied.**
- The op saves solve_triangular's own result for w, not a contiguous copy. The copy for the kernel's row loads is a
  transient inside the op.
- `set_materialize_grads(False)`: unused outputs' gradients reach the backward op as None, treated as zeros in the
  kernel, with no zero-filled [B, H, NC, K, V] tensor.
- dd is a batched dot product (no G ⊙ h temporary).
- Double backward raises.
- `_scan_supported` takes gradient calls when the backward is registered; unsupported calls are counted with or
  without gradients.
- The derivation was checked in fp64 against autograd of the recurrence: every gradient to ~1e-16.
- The plan review's doubt about the gain (whether `solve_triangular`'s backward would dominate) was settled by
  measuring before committing to the design. Forward + backward of the function on the P100 went from 71 to 15 ms at
  2k and from 228 to 29 ms at 4k.

**Code review points applied.**
- Tests with VS = 32 (B·H·V/32 above twice the SM count) and K = 128 against fp64, a 4096-token gradient case, five
  AdamW steps against the reference, and `test_aot_dispatch_dynamic`.
- The comment on freeing the incoming gradients early was wrong (autograd holds them until the backward returns) and
  was dropped; G is freed before dw is allocated.

**Accuracy.**
- Gradients of q, k, v, g, β and the initial state against the reference's autograd: ~5e-7 relative in fp32, under a
  per-head bound of 1e-4 in all four regimes, with and without the final state in the loss.
- The op's backward against fp64 autograd of the recurrence: within 1e-5.
- Losses of a tiny Qwen3.5 over five AdamW steps: within 1e-3 of the reference path.

**In the model** (Qwen3.5-2B fp16, our attention and conv, `qwen35_conv_model_ab.py`: A =
`FLASH_ATTN_LEGACY_GDN_SCAN=0`, B = the scan with its backward, 4 rounds per arm alternated; the LoRA step is a
forward + backward at 1024 tokens, rank-16 adapters on the projections, gradient checkpointing, fp16 autocast):

| GPU | LoRA step | prefill 4096 | generate 2048 + 64 |
|---|---|---|---|
| V100 | 1213 -> 405 ms (2.99x) | 1.170x | 1.081x |
| 2080 Ti | 1735 -> 603 ms (2.88x) | 1.154x | 1.031x |
| T4 | 1671 -> 1203 ms (1.39x) | 0.985x (overlapping ranges) | 0.986x (overlapping) |
| P100 | 2251 -> 1911 ms (1.18x) | 1.040x | 1.013x |
| 1080 Ti | 3156 -> 2866 ms (1.10x) | 0.989x | 1.001x |

- The 1080 Ti's GPU was not shared during this run: its ranges are tight.
- Its prefill is 1.1% slower with the scan, with non-overlapping ranges: at B = 1 its 28 SMs take the 64 blocks of
  VS = 32 in two waves. This is a known, small regression, left for c2, which changes that phase.
- The training gain is the largest of P6.10. On the V100 and 2080 Ti the delta rule's reference backward was most of
  the LoRA step.

**ptxas** (the 8 instantiations, sm_60 / 70 / 75). Acceptance 6 asked for no spills; two remain, both of one
register.
- sm_70, the backward at K 128 / VS 32: 8 bytes. ptxas keeps sm_70 at 80 registers, for 3 blocks per SM.
- sm_75, the forward at K 128 / VS 16: 4 bytes.
- The c1 kernel already had 8-byte sm_70 spills, at VS 32 for both K: section 5's "no spills" was wrong.
- Cost: the forward op on the T4 (B = 1, 16 heads, K = V = 128, 64 chunks, VS 16), c1 against c3, 5 alternated runs per
  arm: median 4.42 against 4.41 ms. Not measurable; left as is.

**Suite.** test_gated_delta and test_causal_conv1d, with and without fla, are green on all five GPUs: 179 passed
without fla, 177 with. Opcheck covers the autograd registration (`test_aot_dispatch_dynamic`).

## 8. c2: the intra-chunk phase (analysis and plan, 2026-11-03)

**Where the time goes now.** Measured with `qwen35_delta_c2_phases.py` (this directory): our path (`_chunk_scan` with
c1 / c3), B = 1, H = 16, K = V = 128, fp16 inputs, L = 4096, one layer. Inference is a no-gradient call, 10 warm-up
calls then the median of 7, CUDA events between phases. Training is the function's forward + backward.

| GPU | inference | prologue | decays | UT + attn matmuls | 2 solves | q / k scale | scan op | out matmuls | training fwd + bwd |
|---|---|---|---|---|---|---|---|---|---|
| P100 | 10.5 ms | 22% | 4% | 8% | 27% | 3% | 27% | 7% | 9.9 + 20.0 ms |
| V100 | 5.8 ms | 22% | 5% | 10% | 23% | 3% | 27% | 9% | 5.8 + 11.5 ms |
| 2080 Ti | 9.3 ms | 20% | 4% | 8% | 29% | 3% | 28% | 7% | 9.3 + 16.9 ms |
| T4 | 18.5 ms | 21% | 4% | 9% | 26% | 3% | 28% | 8% | 18.9 + 35.2 ms |
| 1080 Ti | 13.3 ms | 22% | 4% | 8% | 23% | 3% | 31% | 7% | 13.3 + 26.9 ms |

The epilogue is 1% everywhere.
- The T4 needs warm-up: its clocks ramp up over the first calls. With one warm-up call it read 24-30 ms, with the
  scan op at 8-11 ms.
- **The prologue and the intra-chunk phase are 59-65% of the inference call**: casts, transposes, l2norm, pads, the
  decay matrix, two Gram matmuls, two batched `solve_triangular`, and the q / k scaling. All are many full passes over
  [B, H, NC, 64, 128] fp32 tensors.
- **Training backward**, by autograd node (profiler, GPU time):
  - the 12 elementwise multiplies' backward (`MulBackward0`): 27-32%;
  - the scan op's backward (c3): 19-23%;
  - `solve_triangular`'s backward: 16-19%;
  - the matmuls' backward: 10-13%;
  - views, pads, casts and the rest: ~20%.

**In the model**, estimated from these shares and section 5's measurements:
- The function was 35% of Qwen3.5-2B's 4k prefill on the T4 (401 of 1142 ms, 18 calls).
- Removing ~55% of its time gives ~1.15-1.2x prefill on the T4, ~1.2x on the V100 and ~1.05x on the P100 (where the
  function is a smaller share).
- The 1080 Ti's 1.1% prefill regression (section 7) is inside the same function and would be recovered.

**Gate before the kernel** (the plan review's HIGH 2; `qwen35_delta_c2_gate.py`, this directory). The c2 contract was
prototyped in PyTorch and timed against the reference's lines, at L = 4096, 16 heads, fp16 inputs, median of 7:
- **forward**: the explicit T, then u = T·βv and w = T·βe^c·kn;
- **backward**: the analytic one, from the inputs and T / u / w; the reference's is autograd.

| GPU | reference forward | prototype forward | reference backward (autograd) | analytic backward |
|---|---|---|---|---|
| P100 | 6.72 ms | 4.91 ms | 12.71 ms | 11.11 ms |
| V100 | 3.65 ms | 2.88 ms | 7.06 ms | 6.86 ms |
| 2080 Ti | 5.92 ms | 4.19 ms | 9.45 ms | 9.48 ms |
| T4 | 14.35 ms | 9.67 ms | 20.34 ms | 20.86 ms |
| 1080 Ti | 8.11 ms | 6.50 ms | 17.00 ms | 14.75 ms |

- The gradients agree: dg (fp32) within 6.6e-7. dq / dk / dv / dβ are within 2.1e-4, which is the reference's fp16
  rounding of its gradients.
- The reference's graph holds 402 MiB per call after its forward.
- **Conclusion.**
  - In PyTorch the analytic backward is as fast as autograd's (0.98-1.15x). c2's training gain comes from the fused
    forward, which runs twice per step under gradient checkpointing, and from memory: inputs and T saved instead of
    402 MiB.
  - The backward's elementwise passes are what a fused backward kernel (c2b) would remove. It is decided after c2 is
    measured.

**Plan (c2): one fused "prep" kernel for the prologue and the intra-chunk phase, with an analytic backward in PyTorch**
(revised after the independent review, 2026-11-03).

The op: `gdn_chunk_prep(q, k, v, g, beta) -> (q̃, k̃, w, u, d, A, T)`.
- **Inputs.**
  - q / k / v: the model's [B, L, H, K | V] tensors in their dtype (fp16 / bf16 / fp32) and strides. The last
    dimension must be contiguous; otherwise the op makes it so.
  - g and β [B, L, H]: cast to fp32 on the host when they are not (tensors of B·L·H elements).
  - The tail chunk is zero-padded in the kernel: q, k, v and β are 0 and g is 0, as the reference's pad. This removes
    the prologue's casts, transposes and pads.
- **Outputs.**
  - q̃, k̃, w [B, H, NC, 64, K] and u [B, H, NC, 64, V], all fp32 and row-major. Today `solve_triangular` returns w
    and u column-major, and the scan op copies them to rows; its `contiguous` calls become no-ops.
  - d [B, H, NC] and A [B, H, NC, 64, 64]: A's upper triangle is written as exact zeros, because the output matmul
    reads all 64 x 64.
  - T = (I + M)⁻¹ [B, H, NC, 64, 64], marked non-differentiable, for the backward.
- **The kernel: one block per (b, h, chunk), 256 threads.**
  - Shared memory target ≤ 30 KB static: 2 blocks per SM on the P100 / T4 / 1080 Ti (64 KB per SM), 3 on the V100.
  - Shared memory, registers and blocks per SM are reported per arch and instantiation.
  1. **One pass over K** in 64 x 16 sub-tiles of q and k (4 KB each).
     - It accumulates both Gram matrices, k·kᵀ and q·kᵀ, in registers: 16 + 16 per thread, a fixed 4 x 4 patch of each
       64 x 64.
     - It also accumulates the row norms: k's are the Gram diagonal, q's are summed in the same pass.
     - The products are normalised afterwards (by r_i r_j, with q's scale). This differs from the reference, which
       normalises first, in fp32 rounding only.
  2. The cumulative sum of g (fp32, in shared memory) and the decays.
     - P_ij = exp(c_i - c_j), never e^{c_i}·e^{-c_j}: once c < -88 that product is inf·0.
     - M = strict_lower(β_i kn_i·kn_j P_ij) goes to shared memory (16 KB).
     - A = (qn·knᵀ) ⊙ P goes straight from registers to global memory.
  3. **T = (I + M)⁻¹ by forward substitution, row by row, in place of M**: T_i,<i = -M_i,<i · T_<i,<i.
     - For row i, the 256 threads take its i entries, the j-sum of each split over 4 threads with a shuffle reduction.
     - That is 63 `__syncthreads`, ~10 µs per block, overlapped by the SM's other block.
     - This is the order whose fp32 accuracy was checked (below). The `fla`-style blocked inverse is not taken.
  4. **A second pass over K and V** in 64 x 16 sub-tiles:
     - w = T(β e^c kn) and u = T(βv): T times the tile, 4 outputs per thread;
     - q̃ = qn e^c and k̃ = kn e^(c_last - c) from the same q / k tiles;
     - d = e^(c_last).
     - V needs only to be a multiple of 16.
  - **Traffic**: q and k are read twice, v once. Per call at 4k with 16 heads that is ~100 MB read and ~165 MB written,
    and ~4.6 GFLOP.
  - **Estimate** (from the review's count): the kernel ~1-1.5 ms on the P100 and ~2 ms on the T4, against the 6.7 / 11.8
    ms of the phases it replaces.
- **Accuracy of the inverse** (checked in fp32 against fp64 on 256 chunks per regime):
  - with the explicit inverse, u and w are within 3.8e-8 / 7.3e-8 / 7.5e-8 / 3.7e-7 (fast / slow / flat / parallel
    keys);
  - `solve_triangular` in fp32 gives 1.6-2.1e-7;
  - max |T| is 1.0 in all four regimes. `fla` takes the inverse too.
- **Backward: analytic, in PyTorch** (registered autograd on the op, traceable like c3's).
  - Saved: the op's inputs, T, and w (the same tensor the scan op saves). u is recomputed as T·βv (one batched
    matmul, 32 MB not kept per layer at 4k). kn, qn, c and P are recomputed elementwise.
  - Then:
    - dX = Tᵀ dY for u and w;
    - dM = -strict_lower(Tᵀ (du uᵀ + dw wᵀ));
    - the Gram and decay terms, the l2norm, and the cumsum's reverse sum, in the padded chunk layout (a padded row's dc
      reaches the real tokens through the reverse sum; the review checked this case).
  - The formulas are checked against autograd of the reference's lines in fp64 (`qwen35_delta_c2_backward.py`):
    every gradient within 1.5e-15.
  - The review's variants, run in its scratchpad and not added to the repo, also pass:
    - strong decay (g x 20) and flat (g = 0, β = 1): ~2e-15 in fp64;
    - fp32 at g x 0.5 / x 20 / x 60: ≤ 7e-7;
    - fp32 parallel keys at K = V = 128: worst 1.2e-5 (dg).
  - The alternatives were rejected:
    - **Recompute under autograd** (`torch.autograd.grad` on the reference's lines inside the registered backward). It
      traces; the review checked it with opcheck and `torch.compile`. But it rebuilds the reference's graph in the
      backward, which is today's whole backward cost plus a third forward.
    - **The fused forward for inference only**: two forward paths for one function, and training keeps today's
      forward and memory.
    - **One `solve_triangular` with the right-hand sides concatenated**: it keeps the prologue and the elementwise
      passes, and it would be a third variant of the same lines.
- **Python.** `_chunk_scan` keeps the reference's lines for `kernel=False` (the bitwise copy test). On the kernel path
  the prep op replaces the prologue and the intra-chunk lines. The scan op and the output products stay as they are.
  The routing and the fingerprint gating are unchanged.
  - `FLASH_ATTN_LEGACY_GDN_SCAN=0` still opts out of both ops (the reference).
  - `FLASH_ATTN_LEGACY_GDN_PREP=0` keeps c3's path: the scan with the PyTorch lines. It is for the A/B and as a way out
    if the prep op misbehaves.
  - The gate's PyTorch prototype (explicit T plus the analytic backward) is the fp32 reference in the tests, not a
    route.
- **Memory.**
  - The forward no longer materialises the fp32 copies of q / k / v, the decay matrix or the UT system.
  - Under autograd the op saves its inputs (fp16) and T (16 MB per layer at 4k, 16 heads), against the reference
    graph's 402 MiB.
- **Expected.**
  - Inference: the function ~2x faster against c3 (P100 10.5 -> ~5 ms, T4 18.5 -> ~8.7 ms, V100 5.8 -> ~2.6 ms).
  - Qwen3.5-2B 4k prefill: ~1.15-1.2x on the T4 and V100, ~1.05x on the P100.
  - LoRA step: the backward unchanged; the forward's gain twice under checkpointing. ~2% on the P100, ~5-9% on the T4,
    plus the memory.

**Acceptance (c2).**
1. **The op against the reference's lines, fp32**:
   - q̃, k̃, w, u, d and A within 1e-5 relative per head;
   - A's upper triangle exactly 0;
   - T within 1e-6 of fp64 `inv(I + M)`.
   - Regimes: fast, slow, flat, parallel keys, and strong decay (g down to -20 per token, e^c underflowing in fp32):
     every output finite.
   - L ∈ {1, 63, 64, 65, 4096}; K 64 / 128; V 48 / 128; fp16 / bf16 / fp32 q / k / v; g and β in fp16 / fp32;
     non-contiguous q / k / v (last dimension too); B 2; H 32.
   - 64-bit offsets; run-to-run bitwise determinism.
2. **The whole function**: the c1 / c3 tests unchanged, now through the prep op: output, final state, and gradients
   against the reference within their bounds.
3. **Backward.**
   - The op's gradients against autograd of the reference's lines (fp32, 1e-4 per head) and against fp64, in the five
     regimes at K = V = 128 (the backward script extended to them).
   - T non-differentiable; double backward raises.
   - opcheck with `test_faketensor`, `test_autograd_registration` and `test_aot_dispatch_dynamic`.
   - `torch.compile` of the whole function, forward and backward.
4. **Routing**: `FLASH_ATTN_LEGACY_GDN_PREP=0` gives c3's path bitwise; `FLASH_ATTN_LEGACY_GDN_SCAN=0` gives the
   reference.
5. **Model**: the tiny Qwen3.5 tests unchanged: prefill, decode, chunked prefill, LoRA with checkpointing, and 5 AdamW
   steps.
6. **Speed**, with pass marks:
   - The prep op at 4k with 16 heads ≤ 2 ms on the P100 and ≤ 3 ms on the T4.
   - The function's inference call ≥ 1.6x against c3 on the P100 and T4.
   - Qwen3.5-2B prefill / generate / LoRA A/B, `FLASH_ATTN_LEGACY_GDN_PREP=0` against c2, 4 rounds per arm
     alternated, all five GPUs:
     - no regression with non-overlapping ranges;
     - the 1080 Ti's prefill at least back to the reference's (section 7's 1.1%).
   - If a mark is missed: profile and revise before committing. If it is still missed, report to the user, not ship.
7. **Peak memory** of inference and of a training forward + backward at 4k, as numbers: no higher than c3's.
8. **Suite.**
   - ptxas: spills, shared memory, registers and blocks per SM reported per arch and instantiation. Spills only if the
     A/B shows no cost.
   - The full suite green on all five GPUs, with and without `fla`.

## 9. c2 as implemented, and the results

**Kernel** (`csrc/kernels/gdn_prep.cu`, op `gdn_chunk_prep(q, k, v, g, beta, scale) -> (q̃, k̃, w, u, d, A, T)`).
- As planned (section 8): one block of 256 threads per chunk, two passes over K (and V) in 64 x 16 sub-tiles, the
  Gram matrices in registers, T by forward substitution in shared memory.
- 27 KB static shared memory and 82-111 registers, without spills on sm_60 / 61 / 70 / 75. It is register-bound at two
  blocks per SM on every target; section 8 expected three on the V100, which is wrong.
- **The cumulative decay c is summed in fp64** (64 values per chunk, a warp scan), and its differences are taken in
  fp64 before `expf`. This was not in the plan.
  - Found by the strong-decay tests. With c ~ -700 by a chunk's end, the fp32 differences of prefix sums lose their
    low bits. Our first version's final state was then 20x further from the fp64 rule than the reference's.
  - With c in fp64 it is 10-100x closer than the reference: final state 1.7e-7 against 1.8e-5 at (2, 63, 16, 128).
  - It costs nothing measurable. The backward takes c in fp64 too.
- **Pass 2 reads T's rows as float4** (row stride 68). After the first A/B the T4 missed the op's 3 ms mark, at
  3.45 ms. The scalar reads of T (five shared loads per four FMAs) were the limit. With float4 reads: 2.85-3.02 ms
  on the T4 (two runs) and 1.64 ms on the P100 (was 1.79).
- The conversions `to_f` / `from_f` moved from `causal_conv1d.cu` into `flash_attn_common.h` (one copy).
  `conv_dtype` became `dtype_code`, shared by the two ops.

**Python** (`flash_attn_legacy/gated_delta.py`).
- `_chunk_scan` takes the op when q / k / v share a dtype and `FLASH_ATTN_LEGACY_GDN_PREP` is not 0. Otherwise it
  takes the reference's lines (c3's path).
- `_prep_grads` is the analytic backward of section 8, from the inputs, w and T (u recomputed), with the None
  gradients skipped term by term.

**Accuracy.**
- **The op against transformers' lines in fp64**, per head: each output within 3x the fp32 lines' own error, or within
  5e-7 (T: 1e-6), whichever is larger.
  - w and u are T·X where the lines substitute. With nearly parallel keys they come to 2.0-2.2x the lines' error,
    3.7e-6 absolute.
  - Elsewhere the op is at the lines' level or better.
- **The whole function**: the c1 / c3 tests unchanged, plus the strong-decay regime. Where the fp32 reference is
  itself off the fp64 rule (strong decays), the bound is no further from the fp64 rule than twice the reference.
- **The backward** against autograd of the lines, in fp32 and fp64: within 1e-4 per head in the five regimes, for all
  six outputs' gradients and for subsets.
- **The tiny Qwen3.5 training test in fp16** (model level).
  - A_log's and dt_bias's gradients differ from the reference's by up to 2.3e-2 on the 2080 Ti, against c3's 2.8e-4.
  - The cause is the gradient itself: it is a sum of dg·g over all tokens with up to 330x cancellation (g down to -27
    per token in the tiny model).
  - On layer 0's real inputs, dg of the reference, c3 and c2 is equally far from the fp64 rule (2.4e-4). In that sum
    the reference is 5.9e-3 from the fp64 rule and c2 5.4e-3.
  - c3 matched the reference only because it runs the same operations. The test's bound for these two parameters
    is 5e-2.

**Speed of the function** (`qwen35_delta_c2_ab.py`, L = 4096, 16 heads, fp16 inputs; A = c3's path, B = c2,
alternated, medians):

| GPU | prep op | inference A -> B | training fwd + bwd A -> B |
|---|---|---|---|
| P100 | 1.64 ms | 9.68 -> 4.50 ms (2.15x) | 29.5 -> 24.3 ms (1.21x) |
| V100 | 0.85 ms | 5.32 -> 2.54 ms (2.09x) | 16.2 -> 14.7 ms (1.11x) |
| 2080 Ti | 1.53 ms | 7.93 -> 3.58 ms (2.22x) | 23.4 -> 20.0 ms (1.17x) |
| T4 | 2.85-3.02 ms | 19.1 -> 9.28 ms (2.06x) | 55.0 -> 46.9 ms (1.17x) |
| 1080 Ti | 2.51 ms | 13.4 -> 7.29 ms (1.84x) | 40.5 -> 35.3 ms (1.15x) |

Peak memory over the call: inference 339 -> 258 MiB, training 739 -> 626 MiB.

**In the model** (Qwen3.5-2B fp16, `qwen35_conv_model_ab.py`; A = `FLASH_ATTN_LEGACY_GDN_PREP=0`, B = c2; 4 rounds per
arm alternated, medians; LoRA as in section 7):

| GPU | prefill 4096 | generate 2048 + 64 | LoRA step |
|---|---|---|---|
| V100 | 307 -> 258 ms (1.19x) | 1.03x (overlapping) | 411 -> 373 ms (1.10x) |
| 2080 Ti | 465 -> 388 ms (1.20x) | 1.04x (overlapping) | 536 -> 490 ms (1.09x) |
| T4 | 1123 -> 972 ms (1.16x) | 1.02x (overlapping) | 1164 -> 1118 ms (1.04x, overlapping) |
| P100 | 1796 -> 1704 ms (1.05x) | 2406 -> 2359 ms (1.02x) | 1913 -> 1856 ms (1.03x) |
| 1080 Ti | 3546 -> 3391 ms (1.05x, overlapping) | 1.02x (overlapping) | 2878 -> 2780 ms (1.04x, overlapping) |

- No regression anywhere. The 1080 Ti's prefill (1.1% slower with c3, section 7) is now ahead of c3's.
- The prefill gains match section 8's estimate: ~1.15-1.2x on the T4 / V100 / 2080 Ti, ~1.05x on the P100.
- Peak memory: prefill 4108 -> 4027 MiB. The LoRA step's peak is unchanged (6662 MiB): under gradient checkpointing
  the delta rule's graph is not what sets it.

**Plan review and code review points applied.**
- The plan review's changes are in section 8.
- From the code review:
  - a test of the prep op's own double-backward guard;
  - the flat regime in the backward test, and a K = 48 / V = 16 case;
  - a 5e-7 floor in place of 2e-6;
  - `empty_cache` before the 64-bit offset test;
  - the occupancy comment;
  - `cuda_bf16.h` included at the top of the common header.
- Left as they are:
  - the op writes T even for no-gradient calls (16 MB at 4k with 16 heads);
  - `_prep_grads` computes every input's gradient before the mask;
  - the tests' fp64 copy of the reference's lines, which is the independent truth.

**Suite.** The full suite is green on all five GPUs, apart from the A_log bound above, which was fixed and re-run:
P100 4906 passed, V100 4815, T4 4813, 1080 Ti 4796, 2080 Ti 4793.
- After the float4 change, test_gated_delta and test_causal_conv1d are green everywhere: 302 passed without `fla`, 300
  with.
- ptxas: gdn_prep has no spills on any arch. gdn_scan keeps c1 / c3's two one-register spills (section 7).
- c2b (a fused backward kernel) remains open. The training backward's elementwise passes and the solve's
  backward-shaped matmuls are its target, and the training gain above (1.11-1.21x on the function) is all from the
  forward.
