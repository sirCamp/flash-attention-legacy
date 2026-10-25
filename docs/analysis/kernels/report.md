# flash_attn_legacy: CUDA kernel gap analysis (Pascal / Volta / Turing)

**Historical snapshot (before P2.8)**: measurements and code references are of that time; current state in docs/PLAN.md.

Analysed source: /workspace/analysis/kernels/src, copied from /workspace/fa. Its md5 matches the local working tree for all 7 relevant files. (/workspace paths: the test box of the time, not committed.)
Toolchain: nvcc 12.6. SASS was dumped with the triton-bundled cuobjdump 12.8, because /usr/local/cuda is CUDA 13 and cannot disassemble sm_6x/70.
No GPU was used.

Legend: [M] = measured (ptxas / SASS / compile timing). [R] = reasoned (arithmetic, CUDA docs, design). Treat [R] performance claims as hypotheses to benchmark.

Artifacts in /workspace/analysis/kernels/ (on the test box; `sass/`, `ctime/` and `raw/compile_times.txt` are not committed: the
tracked equivalents are docs/analysis/kernels/raw/ptxas_*.txt and the .txt files in docs/analysis/kernels/):

| file | content |
|---|---|
| build_ptxas.sh | per-TU, per-arch nvcc -cubin -Xptxas -v. Flags = setup.py + torch -D__CUDA_NO_HALF* defines |
| raw/ptxas_<tu>_sm<arch>.txt | RAW ptxas output, kept verbatim |
| raw/compile_times.txt | wall time per TU per arch |
| occupancy.py -> occupancy_table.md (full table incl. varlen), occupancy.json | parsed ptxas + dynamic smem from the launch macros + occupancy model |
| sass/*.sass, sass_opcount.py -> sass_opcount.txt | SASS dumps + static opcode histograms |
| turing_tiles.py -> turing_tiles.txt | tile enumeration for sm_75 |
| ctime/ | per-instantiation compile-time experiment (Pascal fwd, sm_61) |

[M] Build fragility: compiling the .cu files without torch's -D__CUDA_NO_HALF_OPERATORS__/-D__CUDA_NO_HALF_CONVERSIONS__/-D__CUDA_NO_BFLOAT16_CONVERSIONS__/-D__CUDA_NO_HALF2_OPERATORS__ fails in cuda_bf16.h. The cause is "#include <mma.h>" placed inside namespace flash_attn_legacy (fwd_volta.cu:30, bwd.cu:324). Move it to file scope; this matters for BF16 work. (Fixed: file scope in csrc/include/flash_attn_wmma.h.)

## 1. Registers / spills / smem / occupancy

Occupancy model (cuda_occupancy.h style):
- Registers are allocated in 256-register units per warp, over 2 sub-partitions on sm_60 and 4 elsewhere.
- Shared memory is allocated in 256 B units.
- Per-arch limits:

| arch | smem/SM | smem/block | warps/SM | blocks/SM |
|---|---|---|---|---|
| sm_60 (P100) | 64 KB | 48 KB | 64 | 32 |
| sm_61 (GTX 1080) | 96 KB | 48 KB | 64 | 32 |
| sm_70 (V100) | 96 KB | 96 KB opt-in | 64 | 32 |
| sm_75 (T4) | 64 KB | 64 KB opt-in | 32 | 16 |

Dynamic smem uses exactly the launch-macro formulas. Static smem is 0 everywhere [M]. Volta kernels compiled for sm_6x are stubs and are omitted. Pascal kernels are also built for sm_70/75 (reachable via FORCE_ARCH=pascal). Varlen instantiations are within ±10 registers of the batched ones with identical smem; see occupancy_table.md (92 rows at the time).

| kernel | Br,Bc,d,warps | arch | regs | stack B | spill st/ld B | dyn smem B (KB) | blk/SM | warps/SM | blk limit regs/smem |
|---|---|---|---|---|---|---|---|---|---|
| fwd_volta | 128,32,32,4 | sm_70 | 128 | 0 | 0/0 | 36864 (36) | 2 | 8/64 | 4/2 |
| fwd_volta | 128,32,32,4 | sm_75 | 128 | 0 | 0/0 | 36864 (36) | 1 | 4/32 | 4/1 |
| fwd_volta | 128,64,64,4 | sm_70 | 255 | 0 | 0/0 | 81920 (80) | 1 | 4/64 | 2/1 |
| fwd_volta | 128,64,64,4 | sm_75 | 255 | 0 | 0/0 | 81920 (80) | FAIL | - | >64KB |
| fwd_volta | 128,32,96,4 | sm_70 | 254 | 0 | 0/0 | 61440 (60) | 1 | 4/64 | 2/1 |
| fwd_volta | 128,32,96,4 | sm_75 | 254 | 0 | 0/0 | 61440 (60) | 1 | 4/32 | 2/1 |
| fwd_volta | 128,32,128,4 | sm_70 | 255 | 0 | 0/0 | 73728 (72) | 1 | 4/64 | 2/1 |
| fwd_volta | 128,32,128,4 | sm_75 | 255 | 0 | 0/0 | 73728 (72) | FAIL | - | >64KB |
| fwd_volta | 64,16,256,2 | sm_70 | 64 | 1768 | 5708/4800 | 55296 (54) | 1 | 2/64 | 16/1 |
| fwd_volta | 64,16,256,2 | sm_75 | 255 | 424 | 1592/1200 | 55296 (54) | 1 | 2/32 | 4/1 |
| fwd_pascal | 128,32,32,4 | sm_60 | 119 | 0 | 0/0 | 28672 (28) | 2 | 8/64 | 4/2 |
| fwd_pascal | 128,32,32,4 | sm_61 | 125 | 0 | 0/0 | 28672 (28) | 3 | 12/64 | 4/3 |
| fwd_pascal | 128,32,64,4 | sm_60 | 150 | 0 | 0/0 | 40960 (40) | 1 | 4/64 | 3/1 |
| fwd_pascal | 128,32,64,4 | sm_61 | 159 | 0 | 0/0 | 40960 (40) | 2 | 8/64 | 3/2 |
| fwd_pascal | 64,16,96,2 | sm_60 | 168 | 0 | 0/0 | 22528 (22) | 2 | 4/64 | 6/2 |
| fwd_pascal | 64,16,96,2 | sm_61 | 168 | 0 | 0/0 | 22528 (22) | 4 | 8/64 | 6/4 |
| fwd_pascal | 64,16,128,2 | sm_60 | 231 | 0 | 0/0 | 28672 (28) | 2 | 4/64 | 4/2 |
| fwd_pascal | 64,16,128,2 | sm_61 | 235 | 0 | 0/0 | 28672 (28) | 3 | 6/64 | 4/3 |
| fwd_pascal | 32,16,256,1 | sm_60 | 255 | 256 | 2220/1476 | 34816 (34) | 1 | 1/64 | 8/1 |
| fwd_pascal | 32,16,256,1 | sm_61 | 255 | 256 | 2220/1476 | 34816 (34) | 2 | 2/64 | 8/2 |
| bwd_volta | 32,32,32,4 | sm_70 | 128 | 0 | 0/0 | 26624 (26) | 3 | 12/64 | 4/3 |
| bwd_volta | 32,32,32,4 | sm_75 | 79 | 0 | 0/0 | 26624 (26) | 2 | 8/32 | 6/2 |
| bwd_volta | 32,32,64,4 | sm_70 | 219 | 0 | 0/0 | 43008 (42) | 2 | 8/64 | 2/2 |
| bwd_volta | 32,32,64,4 | sm_75 | 162 | 0 | 0/0 | 43008 (42) | 1 | 4/32 | 3/1 |
| bwd_volta | 32,16,96,4 | sm_70 | 168 | 8 | 4/4 | 35840 (35) | 2 | 8/64 | 3/2 |
| bwd_volta | 32,16,96,4 | sm_75 | 150 | 0 | 0/0 | 35840 (35) | 1 | 4/32 | 3/1 |
| bwd_volta | 16,16,128,4 | sm_70 | 168 | 8 | 8/12 | 35328 (34.5) | 2 | 8/64 | 3/2 |
| bwd_volta | 16,16,128,4 | sm_75 | 128 | 16 | 12/12 | 35328 (34.5) | 1 | 4/32 | 4/1 |
| bwd_volta | 16,16,256,4 | sm_70 | 145 | 0 | 0/0 | 68096 (66.5) | 1 | 4/64 | 3/1 |
| bwd_volta | 16,16,256,4 | sm_75 | 145 | 0 | 0/0 | 68096 (66.5) | FAIL | - | >64KB |
| bwd_cudacore | 32,32,32,4 | sm_60 | 197 | 0 | 0/0 | 8192 (8) | 2 | 8/64 | 2/8 |
| bwd_cudacore | 32,32,32,4 | sm_61 | 206 | 0 | 0/0 | 8192 (8) | 2 | 8/64 | 2/12 |
| bwd_cudacore | 32,32,64,4 | sm_60 | 255 | 184 | 324/384 | 16384 (16) | 2 | 8/64 | 2/4 |
| bwd_cudacore | 32,32,64,4 | sm_61 | 255 | 184 | 332/392 | 16384 (16) | 2 | 8/64 | 2/6 |
| bwd_cudacore | 32,16,96,4 | sm_60 | 56 | 2232 | 5456/7296 | 18432 (18) | 3 | 12/64 | 9/3 |
| bwd_cudacore | 32,16,96,4 | sm_61 | 40 | 2472 | 6048/8452 | 18432 (18) | 5 | 20/64 | 12/5 |
| bwd_cudacore | 16,16,128,4 | sm_60 | 32 | 2368 | 6868/7092 | 16384 (16) | 4 | 16/64 | 16/4 |
| bwd_cudacore | 16,16,128,4 | sm_61 | 40 | 2248 | 6492/6716 | 16384 (16) | 6 | 24/64 | 12/6 |
| bwd_cudacore | 16,16,256,4 | sm_60 | 32 | 4696 | 13892/13980 | 32768 (32) | 2 | 8/64 | 16/2 |
| bwd_cudacore | 16,16,256,4 | sm_61 | 48 | 4360 | 13052/13140 | 32768 (32) | 3 | 12/64 | 10/3 |
| precompute_D | 256 thr | all | 31-57 | 0 | 0/0 | 0 | 8 (4 on sm_75) | full | - |

Key observations:

- [M] **Occupancy is very low.** Forward kernels run 4–8 warps/SM (6–12% on 64-warp SMs).
  - "1 thread = 1 Q row + float my_O[d]" pushes registers to about 255 for d ≥ 64, which caps the forward at 8 warps/SM on every arch.
  - Smem then cuts this to 4 warps/SM on V100 d=64/96/128 and P100 d=64.
  - Pascal d=256 runs 1 warp/SM on P100; Volta d=256 runs 2 warps/SM.
- [M] **Volta fwd d=256 on sm_70:** ptxas uses only 64 registers with a 1768 B stack and 5.7 KB static spill stores, so my_O[256] effectively lives in local memory (1200 LDL / 1427 STL in SASS).
  - sm_75: 255 registers + 424 B stack (300 LDL / 398 STL).
  - [R] Likely cause: the 16-step dn PV loop (WMMA + 2 barriers each) is not fully unrolled on sm_70, so my_O[dn+j] becomes dynamically indexed.
- [M] **CUDA-core bwd:** for d ≥ 96 the my_dK[D]/my_dV[D] arrays live entirely in local memory (32–64 registers, 2.0–4.7 KB stack, 1.4k–3.8k static LDL plus a similar number of STL). d=64 already spills 184 B.
- [M] **CUDA-core bwd atomics:** one global RED per dQ element per (q-row, kv-row) pair, i.e. N_q*N_k*d atomics per head. The WMMA bwd issues N_q*(N_k/Bc)*d, which is Bc times fewer.
  - [R] At N=4096, d=128 that is about 2.1e9 atomics/head. This is the dominant cost of the Pascal bwd.
- [R, from SASS + layout] **Smem bank conflicts:**
  - Row-per-thread access to the [Br,Bc] float buffers (smem_P on Pascal; smem_S / smem_PV on Volta) uses a stride of Bc words per thread. Scalar STS/LDS are 32-way conflicted (Bc=32/64) or 16-way (Bc=16); LDS.128 row reads are 8-way.
  - Volta d=64 has 256 static scalar STS per tile loop.
  - The Pascal Q row (smem_Q + tid*d, LDS.U.128) is 8-way conflicted and re-read for every key column (1288 LDS.128 static at d=64).
  - Fix: pad the row stride (Bc+1 for SIMT buffers; Bc+4 for WMMA float stores, ldm multiple of 4, plus float4 reads).
- [M] **WMMA on sm_75 lowers to native Turing HMMA.1688.F32** plus LDSM.16.M88/MT88 (ldmatrix). On sm_70 it lowers to HMMA.884.F32.F32.STEP0-3.

## 2. Turing (sm_75 / T4)

### 2.1 Launch failures on T4: verified [M arithmetic on launch macros + R device limit]

T4 MaxSharedMemoryPerBlockOptin = 65536 B. Each of these makes cudaFuncSetAttribute return cudaErrorInvalidValue, and FLASH_ATTN_CHECK_CUDA then throws (loud failure):

| instantiation | dynamic smem |
|---|---|
| fwd Volta d=64, batched AND varlen | 2*(128*64 + 2*64*64) + 4*128*64 + 2*128*64 = 81920 B (80 KB) |
| fwd Volta d=128, batched AND varlen | 2*(128*128 + 2*32*128) + 4*128*32 + 2*128*32 = 73728 B (72 KB) |
| bwd Volta-WMMA d=256 | 2*(4*16*256) + 4*(2*16*256 + 2*16*16) + 2*16*16 = 68096 B (66.5 KB) |

Everything else fits on T4: fwd d32 36 KB, d96 60 KB, d256 54 KB; bwd d32–128 26–42 KB; forced Pascal kernels at most 40 KB. But all of them run at only 1 block/SM.

Net effect: T4 currently fails the forward for the two most common head dims (BERT d=64, Llama d=128).

### 2.2 T4 tile proposal (current kernel structure) [R, turing_tiles.txt]

Constraints of the current code: Br = threads, Br and Bc multiples of 16, d % Bc == 0. Registers are assumed equal to today's (about 255 for d ≥ 64), so 8 warps/SM is the ceiling without an algorithmic change.

Optional enabler, **K/V aliasing** (one [Bc,d] buffer): it is safe because K is dead after S=QK^T and its barrier, V is loaded after the softmax barrier, and the PV loop ends with __syncthreads. It saves Bc*d*2 B and adds no barrier.

| d | today on T4 | proposal (no alias) | with K/V alias | notes |
|---|---|---|---|---|
| 32 | 128,32 · 36 KB · 1 blk · 4 w | 128,16 · 22 KB · 2 blk · 8 w | 128,16 · 21 KB · 3 blk · 12 w | |
| 64 | FAIL | 128,16 · 32 KB · 2 blk · 8 w (reg cap) | 128,16 · 30 KB · 2 · 8 w | Alt: 128,32 + alias = 44 KB, 1 blk, fewer iterations. Benchmark both |
| 96 | 128,32 · 60 KB · 1 · 4 w | keep | 64,16 · 21 KB · 3 blk · 6 w | |
| 128 | FAIL | 128,16 · 52 KB · 1 · 4 w (or 64,16 · 30 KB · 2 blk · 4 w) | 128,32 · 64 KB exactly · 1 · 4 w | Br=128 halves K/V global traffic |
| 256 | 64,16 · 54 KB · 1 · 2 w | keep | 64,16 · 46 KB | Needs an algorithmic fix (3.2) |

**Bwd d=256 cannot be fixed by tiles.** Br=Bc=16 is already the WMMA minimum, and K, V, Q, dO (32 KB) plus the dK/dV fp32 accumulators (32 KB) already total 64 KB.

Fix: keep the dK/dV accumulators as **persistent WMMA accumulator fragments in registers**. The tile-to-warp mapping in bwd_wmma_gemm_AtB_accum is static. Cost: +8 fragments/warp (+64 registers) at d=256, and smem drops to 34.5 KB.
- This also helps V100: d=64 goes 42 -> 26 KB; d=256 66.5 -> 34.5 KB, giving 2 blocks/SM.
- Interim fallback: route sm_75 d=256 bwd to the CUDA-core kernel (32 KB; works but spills).

[R] **V100 side finding:** d=64 (128,64) at 80 KB gives 1 block/SM. (128,32) is exactly 48 KB, giving 2 blocks/SM: double the occupancy at the cost of twice the KV iterations. Benchmark it.

### 2.3 Same WMMA code on sm_75

Yes [M]. The `__CUDA_ARCH__ >= 700` guard includes 750, and the SASS is native HMMA.1688 + LDSM. Only the tile/smem budget changes.

### 2.4 Runtime tile selection [R]

Recommended. Select once per device and cache:
- Inputs: MaxSharedMemoryPerBlockOptin (P100/1080 48 KB, V100 96 KB, T4 64 KB, sm_86/89 through the PTX JIT 99 KB), MaxSharedMemoryPerMultiprocessor, and SM version (still needed for "tensor cores" ≥70 and "slow fp16" ==61).
- A `constexpr smem_bytes()` per tile type is the single source of truth, used by launches and by a pre-launch "smem <= optin" check with a clear error.
- A host `select_tile(props, d)` returns an enum, and a `switch` instantiates the right kernel.
- T4-only tiles are compiled only for sm_75 (stub elsewhere, like the existing Volta stubs).

### 2.5 Varlen host code [R]

The .cpp hard-codes Br by SM. Any tile change silently produces wrong block_offsets, i.e. wrong results.

Fix: drop block_offsets. Use grid (cdiv(max_seqlen_q, Br), num_seqs, H) with an early exit, as upstream FA2 does; max_seqlen_q is already a parameter and currently unused. This also removes the blocking cu_seqlens_q.to(kCPU) sync on every call (it breaks CUDA graphs).

Also: the batched grid uses gridDim.y = B*H, which must be ≤ 65535 (large-batch decode can exceed it). Use a 3-D grid.

### 2.6 mma.sync m16n8k8 vs WMMA [R]

There is no raw throughput gain: WMMA already emits HMMA.1688.

The gain is the documented fragment layout:
- S, P and O stay in registers (warp owns 16 rows, FA2 style).
- This removes the smem_S -> smem_P -> smem_PV round trips (on sm_75 d=64: 256 STS, 71 LDS, 16 LDS.128 per tile, all bank-conflicted) and removes my_O[d] (e.g. 64 floats/thread at Br=64, d=128).
- It frees the S/P smem, lifting both occupancy ceilings.

Cost: an L-size rewrite. Volta would need m8n8k4 (different layout, slow on sm_75+), or keep WMMA there.

Plan:
- Phase 1: T4 tiles + runtime selection + bwd fragment accumulators (S–M).
- Phase 2: an mma.sync forward for sm_75, only if T4 benchmarks show it is smem/latency bound.

## 3. Pascal

### 3.1 sm_60 vs sm_61 fp16 [M SASS + R throughput]

Static SASS, fwd_pascal d=64:

| | HMUL2 | conversions | FADD | FFMA |
|---|---|---|---|---|
| sm_60 | 2048 | 6144 × HADD2.F32 (fp16 pipe) | 4161 | 2050 |
| sm_61 | 2048 | 6144 × F2F.F32.F16 (conversion unit) | 4161 | 2050 |

Dynamic work per thread per KV tile (d=64, Bc=32); the static S loop is duplicated once:
- S phase: 1024 HMUL2 + 2048 conversions + 2048 FADD.
- PV phase: 2048 conversions (V converted per row per column) + 2048 FFMA.

Rates per SM per clock (CUDA Programming Guide throughput table) [R]: sm_61 does 2 fp16 results, 128 FP32, about 32 conversions; sm_60 has fp16 at 2× its FP32 rate.

sm_61 cost for a 4-warp block per tile:

| work | cost |
|---|---|
| HMUL2 | ≈131k clk |
| conversions | ≈16k clk |
| FP32 math | ≈4k clk |

So the GTX 1080 forward is HMUL2-bound by about 30×, and conversions alone are 4× the FP32 cost. On P100, fp16 is cheap, but the conversions still cost about as much as the FP32 math.

Recommendations:
1. **sm_61 fp32 path, selected at runtime** (sm==61; template flag kFp32Math, compiled only into the sm_61 cubin, stubs elsewhere):
   - Convert K and V tiles to fp32 once at smem-load time: 2*Bc*d/Br = 32 conversions per thread per tile instead of 4096.
   - Use pure FFMA dot products with broadcast LDS.128 of the K row.
   - Q row: for d ≤ 64, keep it as fp32 in registers (also removes the conflicted Q reads); for d ≥ 96, fp32 smem with a padded stride.
   - Smem example, d=64 with Br=64, Bc=32: Q f32 16 KB + aliased K/V f32 8 KB + P (stride 33) 8.25 KB ≈ 32 KB, giving 3 blocks/SM on 96 KB.
2. **sm_60:** store V (optionally K) as fp32 in smem to drop the per-element HADD2.F32 in PV. Opt-in: HFMA2 chunked accumulation (precision risk on large logits).
3. Pad the P and Q smem rows (bank conflicts) on both archs.
4. P100 d=64 is at 40 KB, i.e. 1 block/SM. Use (128,16) = 28 KB, or alias + Bc=16 = 26 KB, to get 2 blocks/SM.

### 3.2 Register pressure [M]

**Forward my_O[d]:**
- d=128: 231–242 registers, no spill, but 4–6 warps/SM.
- d=256: 255 registers + 256 B stack; spill stores/loads 2220/1476 B; 369 LDL / 555 STL in SASS; 1 warp/block.
- Fix [R]: split d across 2–4 threads per row (shuffle-reduce partial dots), or keep O in fp32 smem for d=256.

**Backward 2*D floats/thread:**
- D=64: 255 registers + 184 B stack.
- D ≥ 96: arrays fully local (stack 2.0–4.7 KB, 5.4–15 KB static spill traffic).
- Fix [R]: change the THREADS_PER_KV split from rows to **columns**. Each of the TPK threads owns D/TPK columns of dK/dV (d=128, TPK=8 -> 16+16 floats); S and dP partials are reduced with log2(TPK) shuffles; the final dK/dV reduction disappears.
- Also accumulate dQ per Q block in smem across the Bc kv rows, then one atomic per element. That is Bc× fewer atomics.
- Long term: a SIMT register-blocked GEMM backward (L).

## 4. Missing features

Effort scale: S ≤ 3 days, M ≈ 1–2 weeks, L > 2 weeks.

**(a) Key padding / per-batch kv lengths** — effort S fwd, M with bwd + HF glue; risk low; arch concerns: none.
- Add `const int* seqlens_k` (plus optional `seqstart_k` for left padding) to the fwd/bwd params.
- In the kernel, `seq_len_k = seqlens_k ? seqlens_k[b] : seq_len_k`. Masking and fwd_kv_limit already use the variable.
- Bwd: KV blocks beyond the length must write ZERO dK/dV, because the outputs come from torch::empty.
- Causal becomes per-batch bottom-right.
- Unpad + varlen keeps working but pays gather/scatter plus the Python bwd loop.
- Files: common.h, fwd_*.cu, bwd.cu, .cpp, integrations.py.

**(b) Native varlen backward** — effort M; risk low-med.
- Add precompute_D_varlen ([total,H]).
- Bwd gets a VarlenSeq policy: grid (cdiv(max_seqlen_k,Bc), num_seqs*H_kv) with early exit, token strides, L/D at [tok*H+h], one dQ fp32 buffer plus a cast.
- Removes num_seqs*4 .item() syncs and about 10 copies/launches per sequence.
- The Turing d=256 bwd smem fix applies here too.
- Files: bwd.cu, .cpp, flash_attn.py.

**(c) Sliding window (left W)** — effort S–M; risk low; all archs.
- Mask: `j > i+off || j < i+off-W`.
- Block skipping is mandatory for any speedup: fwd kv_begin = max(0, (q_row_start+off-W)/Bc); bwd q_end_blk = min(nq, (kv_start+Bc-1-off+W)/Br + 1).
- Fully masked rows are already guarded (new_max == -inf).

**(d) Logit softcap (tanh)** — effort S fwd / M bwd; risk low.
- Fwd: `s = cap*tanh(s*scale/cap)`. Bwd: recompute t, then `ds *= (1 - t*t)`.
- MUFU.TANH (tanh.approx) exists only on sm_75+. sm_6x/70 use software tanhf (about 2× an exp per element), so the Pascal S phase slows down noticeably.

**(e) Attention sinks (gpt-oss)** — effort S fwd / S–M bwd; risk low.
- Fwd: init row_max = sink[h] and row_sum = 1; online softmax handles the rest, and L includes the sink.
- Bwd: dQ/dK/dV unchanged. New d_sink[h] = -sum_i exp(sink_h - L_i)*D_i in a small reduction kernel next to precompute_D.
- Pair with (c): gpt-oss alternates 128-token windows. d=64 works on all archs.

**(f) Dropout** — effort L; risk med-high.
- Philox4x32-10 keyed by (seed, offset), with a tile-independent counter (bh, i, j/4). Fwd and bwd tiles differ, and so do Pascal and Volta.
- Fwd: drop P only in PV; row_sum uses the undropped P.
- Bwd: regenerate the mask; dV += (P∘M/(1-p))^T dO; dS = P∘(dP∘M/(1-p) - D).
- Seed and offset come from torch's CUDA generator.
- Pascal has no fast 32-bit IMUL (XMAD sequences), so Philox costs more there.

**(g) Decode kernel (q_len 1..16, long KV)** — effort M–L; risk med.
- New flash-decoding kernel: grid (num_splits, B*H_kv). Each block covers one KV chunk for all (H_q/H_kv)×q_len rows (GQA packing fills the rows) and writes partial O fp32 + LSE; a combine kernel reduces.
- Needs cache_seqlens (ties to (a)), HF-cache strides, paged KV later.
- Memory-bound, so CUDA cores suffice and one kernel serves all three archs; sm_61 must use fp32 math.
- Today q_len=1 uses 1/128 of the threads and B*H blocks, each doing a serial full-KV scan.

**(h) BF16** — effort M; risk med (range/precision; needs large-magnitude tests).
- No bf16 math before sm_80. bf16->fp32 is a 16-bit shift (cheap everywhere).
- Fwd: template the Element type and convert bf16->fp16 on smem load. This is lossless inside fp16's normal range but **overflows above 65504**: add a saturate + debug check (LLM Q/K/V usually stay below 1e3). Write O as bf16 from fp32.
- Bwd: dO below about 6e-8 flushes in fp16. Since dQ/dK/dV and D are linear in dO, pre-scale dO by 2^k (from amax) and unscale in the epilogue; this is exact. Write dK/dV as fp32 or bf16, not half.
- Pascal: convert straight to fp32.
- Volta/Turing keep their tensor cores through the fp16 conversion.

**(i) Deterministic dQ** — effort M; risk low.
- Optional separate Q-outer dQ kernel that recomputes S, P, dP and writes dQ once. Costs about +40% bwd FLOPs.
- dK/dV are already deterministic (one block per KV block, GQA heads looped in order).
- Per-kv-block partial buffers are infeasible (512 MB/head at N=4096, Bc=16, d=128).
- On Pascal it also avoids the 2e9-atomic pattern, so it may even be faster.

**(j) int64 offsets** — effort S; risk low.
- `b_idx*q_batch_stride` and `(q_start_tok+q_row_start)*token_stride` are int×int and overflow beyond 2^31 elements (4 GB fp16; dQ_fp32 at the same count).
- Make the strides int64_t, compute base pointers in 64-bit, keep intra-tile offsets in int32.
- Also move B*H off gridDim.y (65535 limit).
- Cost: a few IMAD.WIDE per kernel.

## 5. Code quality and refactor

### Duplication [M]

- Batched vs varlen kernels are about 95% identical (fwd_volta 175+175 lines, fwd_pascal 150+160); only the prologue and the L index differ.
- The tile loader exists 3 times.
- The WMMA GEMMs are duplicated in bwd.cu.
- The smem formula is duplicated in 6 launch macros. The varlen Br table is duplicated in the .cpp, which is a latent bug.
- The softmax/mask/epilogue logic is duplicated between the volta and pascal forwards.

### Proposal [R]

- **kernel_traits.h:** `FwdTile<D, Arch>{Br, Bc, warps, constexpr smem()}` plus a host select_tile().
- **seqinfo.h:** BatchedSeq (optional seqlens_k[b]) and VarlenSeq (cu_seqlens, no block_offsets) policies with int64 base pointers, seqlen_q/k and causal_off.
- **mask.h:** runtime MaskParams{causal, window_left}, kv_block_range / q_block_range / visible(i,j).
- **mma_policy.h:** MmaWmma (sm_70/75) and MmaSimt<kFp32Math> (sm_60/61) exposing gemm_QKt / gemm_PV, so a single `flash_fwd_kernel<Traits, SeqInfo, Mma, Element>` shares the softmax/epilogue.
- Softcap and sinks become runtime-uniform branches.
- One copy each of load.h and wmma_gemm.h.

### Instantiation count [R]

- Today: 20 real instantiations × 4 archs, some of them stubs.
- Proposed fwd: d(5) × SeqInfo(2) × Element(2) = 20 per family. WMMA is built only for sm_70/75 and SIMT only for sm_60/61, giving 80 arch-cubins. Bwd is similar (80). T4-only tiles add +4 on sm_75.
- Making causal/window/softcap/sinks template flags would multiply this by 16 (about 1300). Do not do that.

### Compile time [M]

CPU time per TU per arch:

| TU | sm_60 | sm_61 | sm_70 | sm_75 |
|---|---|---|---|---|
| fwd_pascal | 121 s | 116 s | 118 s | 123 s |
| bwd | 13 s | 11 s | 24 s | 27 s |
| fwd_volta | stub | stub | 13 s | 11 s |

fwd_pascal is 83% of the ~580 s total, and as one TU it is the critical path of the ~8 min build.

Per instantiation (sm_61, batched): d32 3.5 s, d64 9.2 s, d96 5.8 s, d128 8.8 s, **d256 30.6 s** (21 s cicc + 9 s ptxas). Full unrolling over d dominates.

Quick wins:
1. Compile the Pascal kernels only for sm_60/61 (stub on ≥700; keep FORCE_ARCH=pascal behind a build flag). Saves about 240 s CPU.
2. Split TUs per (family, d, SeqInfo) so ninja parallelises. Wall time becomes the slowest TU, about 1 min.

[R] With 160 instantiations including bf16, CPU time is about 2× today's useful part, but wall time on the 64-core box is about 1–2 min, versus 8 min now.

### Other items [M/R]

- Move #include <mma.h> out of the namespace (fixed).
- No K/V prefetch / double buffering (register-staged prefetch is possible on all archs; no cp.async before sm_80).
- 32-bit half2 global loads; 128-bit loads would cut instructions 4×.
- O is stored row-per-thread with a 128–512 B lane stride (uncoalesced; epilogue only, minor).

## Priorities

1. **T4 enablement:** tiles (2.2), runtime selection, varlen grid without block_offsets, bwd dK/dV in fragments (S–M).
2. **sm_61 fp32 path** + Pascal bwd column split (M).
3. **Build:** Pascal kernels only on sm_6x + per-d TUs (S; about 8 min -> 1–2 min).
4. **Features:** varlen bwd, key padding, window + sinks; then decode, BF16, deterministic dQ, int64; dropout last.
