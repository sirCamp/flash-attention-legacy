# Backward on tensor cores: measurements and Turing `mma.sync` design

Status: step 2 implemented (`BwdMma` in `csrc/kernels/flash_attn_bwd.cu`, primitives in `csrc/include/flash_attn_mma.h`); the code is the
reference where this text differs. T4 causal B=2 H=16 N=4096, backward kernel: d=64 18.2 -> 16.6 ms, d=128 53.7 -> 32.3, d=256
167.7 -> 168.4 (`<32,16,256,4>`: every warp of a band recomputes S^T / dP^T, 4x at kSplitD 4 — next step).

## Where the time goes (V100 / T4, causal B=2 H=16 N=4096)

fwd+bwd vs SDPA mem-efficient: V100 1.29x (d=64) / 1.30x (d=128) / 1.03x (d=256), T4 1.02x / 1.48x / 1.15x;
the forward alone is at or below SDPA. Per kernel (torch profiler, d=128):

| kernel | V100 ms | T4 ms |
|---|---|---|
| backward (`bwd_batched_kernel`) | 21.4 | 53.8 |
| forward (recompute in the bench) | 4.8 | 7.2 |
| `precompute_D` | 0.76 -> 0.09 (coalesced, d48dfdc) | 2.62 -> 0.28 |
| dQ zero-fill + fp32 -> fp16 | 0.57 | 1.9 |

Ablations of the backward kernel (variants built aside, wrong results, timing only; d=128):

| variant | V100 ms | T4 ms |
|---|---|---|
| base | 21.3 | 53.7 |
| no dQ step | 15.6 (-27%) | 37.3 (-31%) |
| dQ without atomics (plain stores) | 18.4 | 52.0 |
| Q / dO loaded once (no per-q-block loads) | 18.7 | 49.2 |
| dQ atomics straight from the fragments (no staging) | 23.8 (d=64: -19%) | 55.2 |
| dS row-major for the dQ A operand | 20.4 | 53.7 |
| Q / dO register prefetch | 19.5 | 52.9 |
| prefetch + dS row-major | 18.9 (base 18.5 in that run) | 53.0 |

The V100 base moves by +-10% between runs (20.6-21.6, then 18.5): the small variants are within noise.

Conclusions:
- The main loop without dQ (S, dP, dV, dK: 4 x 68.7 GFLOP) runs at 17.6 TFLOPS on V100, the forward at 32.6. Per MMA the backward
  loads ~1.5 WMMA fragments from shared memory (K / V band reloaded every query block, P^T and dS^T stored to a per-warp buffer and
  reloaded as matrix_a, dQ staged through shared memory), the forward ~1.0 (Q in registers).
- The dQ step costs 27-31% for 20% of the FLOPs; atomics are a small part (T4 1.7 ms): the cost is the step's structure (2-4 MMA
  per output tile, staging, column-major dS loads).
- T4 d=128 runs the 64 KB fallback tile `<32,32,128,2>`: 4 warps per SM (V100: `<32,64,128,2>`, 8 warps). The per-warp buffers
  (10 KB) are what keeps `<32,64,128,2>` (67.8 KB) out of 64 KB.
- WMMA fragment layouts are opaque (the accumulator layout is probed, the matrix_a layout differs): an accumulator cannot become
  the next MMA's operand without a shared-memory round trip. That is the FA2 trick, and it needs `mma.sync` with known layouts.

## Turing: `BwdMma` (mma.sync.m16n8k8.f32.f16.f16.f32 + ldmatrix, sm_75)

Same block geometry as `BwdWmma` (grid over key blocks of Bc, loop over the Q heads of the KV head and the query blocks of Br;
warp w owns the 16 keys of band w / kSplitD and the d / kSplitD columns of slice w % kSplitD of dK / dV). Lane `g = lane >> 2`,
`t = lane & 3`; an m16n8 accumulator holds rows g, g + 8 and columns 2t, 2t + 1 (c0, c1 / c2, c3), and its half2 packing
`{c0c1, c2c3}` is the m16n8k8 A operand of the same 16 rows and 8 k (the reason for k8 MMAs).

Per query block, per warp, all in registers:
1. S^T [16 keys, Br] = K_band Q^T and dP^T = V_band dO^T: A = K_band / V_band (`ldmatrix.x4`, rows of K / V), B = Q^T / dO^T: an
   m16n8k8 B fragment holds (k = 2t, 2t + 1; n = g), consecutive d of one query row, so `ldmatrix` (no trans) on Q / dO rows.
   Accumulators Br / 2 floats each.
2. P^T = exp2(x(S^T) - L), dS^T = P^T (dP^T - D) dx/ds with the element positions known (no probe); masking, ScoreMap, sinks
   (through L) as today.
3. dV += P^T dO: A = P^T straight from the S^T accumulators (k = queries), B = dO [query, column] with `ldmatrix.trans`.
4. dK += dS^T Q: A = dS^T from registers, B = Q with `ldmatrix.trans`.
5. dS^T (key-major, half2 stores) to shared memory [Bc, Br + 8] by the slice-0 warp of each band; one block barrier.
6. dQ [Br, D] += dS K over the whole key block: units of 16 queries x 16 columns spread over the warps, A = dS from dS^T
   (`ldmatrix.trans`), B = K (`ldmatrix.trans`), atomics straight from the accumulators (known rows / columns, no staging).

Shared memory: K, V [Bc, D + 8], Q, dO [Br, D + 8], L, D [Br], dS^T [Bc, Br + 8] (half). No per-warp buffers, no staging, no layout
probe. d=128: `<32,64,128,2>` = 57 KB fits the T4's 64 KB (8 warps per SM instead of 4). dK / dV leave the registers as half2
stores at their known positions.

Registers (d=128, `<32,64,128,2>`): S^T + dP^T 32, dK + dV 64 (DP = 64), fragments ~16, addressing ~30: ~150, no spills
expected.

Barriers per query block: two, as today (after Q / dO / L / D are in shared memory; after dS is complete). The next block's Q / dO
writes do not race the dQ step (it reads dS and K only); dS is rewritten only after the next first barrier.

Shared parts: `BwdBlockArgs`, the loaders, `KeyWindow`, `ScoreMap`, the dQ fp32 buffer and its conversion are already common. The
query-block loop is the same in `BwdWmma` and `BwdMma`; factor it (like `fwd_band_loop`) once the second body exists.

Dispatch: by the arch of the code that runs (`compiled_arch_for`: SASS of the device's major with the highest minor not above it,
else PTX): >= 75 -> `BwdMma`, 70 -> `BwdWmma`, 6x -> `BwdSimt`; a T4 with a 7.0-only build uses WMMA. `FLASH_ATTN_LEGACY_FORCE_ARCH=volta`
keeps WMMA testable on a T4. Device code for `mma.sync` / `ldmatrix` exists only for `__CUDA_ARCH__ >= 750`; the stub bodies (as all
stubs now) trap. sm_8x runs `BwdMma` through the PTX 7.5 JIT (untested there).

## Steps (one change at a time: tests on all GPUs, benchmark, ptxas, review)
1. Done: `flash_attn_mma.h`: m16n8k8 MMA, `ldmatrix` x4 (+ trans), half2 packing.
2. Done: `BwdMma` + tiles + dispatch.
2b. Design review follow-ups, by measurement: S^T / dP^T split by queries across the warps of a band (no recompute at kSplitD > 1;
    d=256), K_band / V_band A fragments kept in registers (one block per SM: `__launch_bounds__(threads, 1)`), dQ work as 16-query
    strips (A loaded once), the shared query-block loop of `BwdWmma` / `BwdMma` factored like `fwd_band_loop`.
3. Turing forward on the same primitives (P2.8 step 3): Q in registers, P straight from S.
4. V100: either `mma.m8n8k4` (known layouts, quad-pair geometry) or keep WMMA; decide on the T4 results.
