# P2.8 design: FA2-style warp-band forward

Status: steps 0, 1, 2, 3, 5 and 6 done (see the step list; step 3 as P7.3); the code is the reference where this design differs. Baseline: `docs/bench/*.json` in git history (commit 81e45f9); current numbers in the same files.

**As implemented** (differs from the original plan below): the shared part is `csrc/include/flash_attn_fwd_band.h` (`FwdBlockArgs`, batched/varlen kernels, `fwd_key_masked`, `BandSoftmax<ROWS, ROW_LANES>`, `fwd_band_loop`, `fwd_band_run` with the Q/K/V copies and optional K/V register prefetch, `fwd_batched_kernel` / `fwd_varlen_kernel`, host launchers). Each family is a `Body` with `run<Tile>(const FwdBlockArgs&)` (now `run<Tile, kCache, kBf16, kRows>`): `TuringFwd` (`mma.sync` m16n8k8, `flash_attn_fwd_turing.cu`, step 3), `VoltaFwd` (WMMA, staged S/P) and `PascalFwd` (fp32 register micro-tile), instead of a per-op `Mma` policy. Tiles: `TileTuring_*` / `TileVolta_*` / `TilePascal_*` in `flash_attn_common.h`, chosen by measurement (comments there).

## Constraints from the code of the time (original plan)
- One block body per family (`volta_fwd_block`, `pascal_fwd_block`; now the `VoltaFwd` / `PascalFwd` bodies) shared by the batched and varlen kernels; today they return a per-thread lse (thread = row).
- Varlen `block_offsets` come from `flash_attn_fwd_{volta,pascal}_block_rows(d)` on the host: any new family must go through the same selection (now also `flash_attn_fwd_turing_block_rows`, selected by `fwd_block_rows(sm, d)` in `csrc/flash_attn_legacy.cpp`).
- `FLASH_ATTN_LEGACY_FORCE_ARCH=volta` runs the WMMA path on a T4; an `mma.sync m16n8k8` path runs only on sm_75+.
- lse convention: natural log, scale applied, `row_max + logf(fmaxf(row_sum, 1e-20f))`; fully masked rows: lse = -inf, O = 0. The backward depends on it.

## Architecture: one body, three MMA policies
`fwd_band_block<Tile, Mma>` in `csrc/include/flash_attn_fwd_block.h` (original plan; now `fwd_band_run` / `fwd_band_loop` / `fwd_batched_kernel` in `csrc/include/flash_attn_fwd_band.h`, one `Body` per family). Each warp owns a band of 16 query rows (`Br = 16 * warps`). The body owns block geometry, the K/V loop (with register prefetch), the unmasked/masked causal split, the online softmax (log2 domain, as FA2), and the L write (`L_ptr`, `l_stride`: 1 batched, `num_heads` varlen). The policy provides the fragments and `qk`, `for_each_s(row, col, float&)`, `make_p`, `rescale_o`, `pv`, `store_o`, plus `kRowsPerThread` / `kRowLanes` for the generic xor-shuffle row reductions. `row_sum` stays a per-thread partial and is reduced once at the end.

Softmax: `scale_log2 = scale * log2(e)`; per KV block `m_new = max(m_old, rowmax)`, `m_use = m_new == -inf ? 0 : m_new`, `alpha = exp2(m_old - m_use)`, `p = exp2(s*scale_log2 - m_use)`; final `lse = m*ln2 + log(max(sum, 1e-20))`. Output is no longer bit-identical to the old kernel (tests use tolerances).

### Turing: `MmaSm75` (original plan; now `TuringFwd`) (mma.sync.m16n8k8 f32.f16.f16.f32 + ldmatrix)
Lane `g = lane>>2`, `t = lane&3`: per 8-column n-tile a thread holds rows g, g+8, cols 2t, 2t+1 (c0,c1 / c2,c3); `kRowsPerThread = 2`, `kRowLanes = 4`. S = Bc/2 regs, O = d/2 regs, Q in registers (`ldmatrix.x4`, d/4 regs; Q kept in smem at d=256). K via `ldmatrix` (no trans), V via `ldmatrix.trans`. P as the A operand straight from S: for k-step kk, `A = {half2(S[kk].c0,c1), half2(S[kk].c2,c3)}` (the m16n8k8 C layout equals the A layout; the reason to use k8 rather than k16). Smem rows stride d+8 halves.

### Volta: `MmaWmmaStaged` (WMMA 16x16x16 with per-warp staging)
S fragments stored to a per-warp `[16][Bc+8]` float staging (Bc+8, not Bc+4: with Bc+4 at Bc=64 lanes (r=0,h=1) and (r=1,h=0) hit the same banks), softmax with 2 lanes per row (interleaved float4 chunks), P written as half in place (ldm `2*(Bc+8)`), reloaded as matrix_a. O rescale without assuming the WMMA layout: probe the accumulator row map once per block through `store_matrix_sync` (a bijection), then apply per-row alpha read from per-warp smem. No block-wide barrier for softmax. SDPA mem-efficient also stages through smem on sm_70, so the target is reachable. `mma.m8n8k4` is a later optional step (P2.8.7) only if V100 stays above 1.5x.

### Pascal: `MmaSimt` (fp32 FMA register micro-tile)
Today's forward is conversion bound (`half_dot`: ~2 cvt per FMA, F2F at 1/4 FFMA rate on sm_6x). Convert once at the smem store (fp32 smem). Lane `= rg*8 + cg`: rows 4rg..4rg+3 (`kRowsPerThread = 4`, `kRowLanes = 8`), keys `cg + 8j`, O columns `4cg + 32i..+3`. S with LDS.128 of Q (broadcast) and K: 64 FFMA per 8 LDS.128 at TN=4, no cvt. P stored transposed per warp `[Bc][16]` fp32. O = d/2 regs (vs `my_O[d]` today).

## Tiles (`<Br, Bc, d, warps>`, in `with_first_fitting_tile` order; sizes computed, speeds to measure)
| d | Turing mma | V100 WMMA-staged | Pascal SIMT fp32 |
|---|---|---|---|
| 32 | `<64,64,32,4>` 15 KB | `<64,64,32,4>` 33 KB | `<64,32,32,4>` 26 KB |
| 64 | `<64,64,64,4>` 27 KB | `<64,64,64,4>` 45 KB, `<64,32,64,4>` 29 KB | `<64,16,64,4>` 30 KB |
| 96 | `<64,64,96,4>` 39 KB, `<64,32,96,4>` 26 KB | `<64,32,96,4>` 36 KB | `<64,16,96,4>` 40 KB, `<32,16,96,2>` 27.6 KB |
| 128 | `<64,64,128,4>` 51 KB, `<64,32,128,4>` 34 KB | `<64,64,128,4>` 70 KB, `<64,32,128,4>` 44 KB | `<32,16,128,2>` 35 KB |
| 256 | `<64,32,256,4>` 66 KB, `<64,16,256,4>` 50.7 KB | `<64,32,256,4>` 77 KB, `<64,16,256,4>` 56 KB | `<16,16,256,1>` 50 KB, `<16,8,256,1>` 34 KB |

Register budget (tensor cores): O d/2 + S Bc/2 + Q d/4 + prefetch + ~30. `__launch_bounds__(threads, kMinBlocks)` per tile; 0 spills checked with `build_ptxas.sh` at every step. `smem_bytes()` stays the single source of truth (staging and aliasing included).

## K/V register prefetch
Split `load_tile_g2s` into `load_tile_g2r` + `store_tile_r2s<DstT>` (128-bit, converting store for Pascal fp32); `load_tile_g2s` becomes their composition. Loop with single K/V buffers and two barriers per iteration (today 2 + d/Bc):
```
prologue: g2s(K_0); sync
for j:
  rV = g2r(V_j); S = Q K_j^T; mask (diag only); softmax; rescale O
  r2s(rV); if (j+1<n) rK = g2r(K_{j+1}); sync
  O += P V_j; if (j+1<n) r2s(rK); sync
```

## Causal: masking only on diagonal blocks
`fwd_kv_nomask_limit(causal, q_row_start, Bc, seq_len_q, seq_len_k) = min(seq_len_k / Bc, causal ? max(0, (q_row_start + seq_len_k - seq_len_q + 1) / Bc) : INT_MAX)` next to `fwd_kv_limit`; `iter<false>` then `iter<true>` (one source, two instantiations) (original plan; now `fwd_band_loop`'s `nomask_first` / `nomask_last` from `KeyWindow::first_key` / `last_key`, masked and unmasked steps of one loop). Masked: `col >= seq_len_k`, `causal && col > row + seq_len_k - seq_len_q`. Fully masked bands keep m = -inf (O = 0, lse = -inf) and still hit every barrier.

## Implementation order (one change at a time; tests on all GPUs, benchmark, ptxas, review)
0. **Done**: groundwork, behaviour neutral: (a) shared WMMA helpers in `flash_attn_wmma.h` (forward + backward); (b) g2r/r2s split; (c) blocks write L; (d) `fwd_kv_nomask_limit` + split loop in the current kernels; (e) tests: raw lse vs FP32, unaligned N (15, 17, 63, 65, 127), causal n_q >> n_k (200, 5), varlen segments 1/15/17/65, d=256 N=4096.
1. **Done**: warp-band WMMA body (staged) for sm_70/75 replaces the thread-per-row `volta_fwd_block` (V100 fwd N=4096 d=64 9.5 -> 5.6 ms, d=128 24.2 -> 11.6; T4 26.1 -> 14.0, 48.6 -> 27.9). The `Mma` policy split (shared `fwd_band_block` in its own header) is deliberately done in step 3, when the second policy exists, instead of abstracting around a single user; the loop, masking and softmax must then live once in the shared body.
2. **Done**: register prefetch in the body.
3. **Done** (P7.3): `MmaSm75` (Turing; now `TuringFwd`, `flash_attn_fwd_turing.cu`); `FORCE_ARCH=volta` keeps WMMA testable on T4; `sm` passed to `*_block_rows` so launch and varlen offsets share one selection.
4. Tuning (staged O stores, per-warp n-tile skipping on the diagonal, tile sweep). Acceptance: forward N=4096 within 1.5x of SDPA mem-efficient on V100 and T4.
5. **Done**: `PascalFwd` (fp32 register micro-tile, fp32 shared memory) replaces the thread-per-row forward, inside the shared band header (P2.1 for the forward). N=4096 causal, P100 / 1080 Ti: d=64 27 -> 12.2 / 74 -> 15.4 ms (P100 now 0.78x SDPA mem-efficient), d=128 81 -> 52 / 176 -> 41, d=256 1768 -> 285 (P100). Open: d=128/256 on Pascal (TN=1 tiles are shared-memory bound, d=256 runs one-warp blocks).
6. **Done**: cleanup (old bodies and tiles), PLAN §1b/§3b, regenerate `occupancy_table.md`.
7. Optional: Volta `mma.m8n8k4`.

## Risks
Register pressure at d=128/256 (smaller Bc candidates, Q in smem, `kMinBlocks`, ptxas check); WMMA layout opacity (runtime probe); Turing path testable only on sm_75+; 16-byte alignment for ldmatrix / 128-bit loads (host check or clone on misaligned views); numerics (exp2, P in fp16 on tensor cores, guarded by the new lse test); edge semantics (step 0e first); varlen Br consistency; build time (instantiations x archs); bank conflicts in the new layouts (Nsight on T4/V100 as in P5.3).
