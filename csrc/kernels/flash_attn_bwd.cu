// ============================================================================
// Flash Attention v2 — Backward Kernel
// Works on both Pascal and Volta (CUDA core based).
//
// Key optimization vs v1: multi-thread per KV row
//   THREADS_PER_KV = NUM_THREADS / Bc threads collaborate on each KV row.
//   The Q-row inner loop is split across these threads (stride pattern).
//   dK/dV are accumulated in per-thread registers, then reduced via warp
//   shuffle (__shfl_down_sync with width = THREADS_PER_KV).
//   dQ is accumulated in a separate FP32 buffer with atomicAdd (unchanged).
//
//   d=64  → THREADS_PER_KV = 128/32 = 4  (4× speedup on Q-row loop)
//   d=128 → THREADS_PER_KV = 128/16 = 8  (8× speedup on Q-row loop)
//
// Algorithm (Flash Attn v2 backward — KV-outer loop):
//   For each KV block (j):
//     Load K_j, V_j into shared
//     Init dK_j = 0, dV_j = 0 in registers
//     For each Q head (GQA):
//       For each Q block (i):
//         Load Q_i, dO_i into shared
//         For qi in stride pattern (split across THREADS_PER_KV):
//           Recompute S_ij = Q_i @ K_j^T * scale
//           Recompute P_ij = exp(S_ij - L_i)
//           dV_j += P_ij * dO_i
//           dP_ij = dO_i @ V_j^T
//           dS_ij = P_ij * (dP_ij - D_i)
//           dQ_i += dS_ij @ K_j * scale     (atomicAdd to FP32 buffer)
//           dK_j += dS_ij * Q_i * scale
//     Reduce dK_j, dV_j across threads via warp shuffle
//     Store dK_j, dV_j
// ============================================================================

#include "flash_attn_common.h"

#include <mma.h>  // file scope: inside a namespace it breaks cuda_bf16.h

#include <string>

namespace flash_attn_legacy {

// ---------------------------------------------------------------------------
// Vectorized load
// ---------------------------------------------------------------------------
template <int ROWS, int COLS, int NUM_THREADS>
__device__ __forceinline__ void bwd_load_tile(
    const half* __restrict__ src,
    int src_row_stride,
    half* __restrict__ dst,
    int num_valid_rows
) {
    static_assert(COLS % 2 == 0, "COLS must be even");
    constexpr int COLS2 = COLS / 2;
    constexpr int TOTAL2 = ROWS * COLS2;
    constexpr int PER_THREAD = (TOTAL2 + NUM_THREADS - 1) / NUM_THREADS;
    const int tid = threadIdx.x;
    const half2 zero2 = {__float2half(0.0f), __float2half(0.0f)};
    half2* dst2 = reinterpret_cast<half2*>(dst);

    #pragma unroll
    for (int i = 0; i < PER_THREAD; i++) {
        int idx = tid + i * NUM_THREADS;
        if (idx < TOTAL2) {
            int row = idx / COLS2;
            int col2 = idx % COLS2;
            if (row < num_valid_rows) {
                const half2* src2 = reinterpret_cast<const half2*>(src + row * src_row_stride);
                dst2[row * COLS2 + col2] = src2[col2];
            } else {
                dst2[row * COLS2 + col2] = zero2;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Precompute D = rowsum(dO * O) — launched as a separate kernel
// D is [B, H, N] float
// ---------------------------------------------------------------------------
__global__ void precompute_D_kernel(
    const half* __restrict__ dO,   // [B, H, N, d]
    const half* __restrict__ O,    // [B, H, N, d]
    float* __restrict__ D,         // [B, H, N]
    int batch_size, int num_heads, int seq_len, int head_dim,
    int o_batch_stride, int o_head_stride, int o_seq_stride,
    int l_batch_stride, int l_head_stride
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * num_heads * seq_len;
    if (idx >= total) return;

    const int n = idx % seq_len;
    const int h = (idx / seq_len) % num_heads;
    const int b = idx / (num_heads * seq_len);

    const half* dO_row = dO + b * o_batch_stride + h * o_head_stride + n * o_seq_stride;
    const half* O_row  = O  + b * o_batch_stride + h * o_head_stride + n * o_seq_stride;

    float sum = 0.0f;
    const half2* dO2 = reinterpret_cast<const half2*>(dO_row);
    const half2* O2  = reinterpret_cast<const half2*>(O_row);
    const int d2 = head_dim / 2;
    for (int j = 0; j < d2; j++) {
        half2 a = dO2[j];
        half2 b_val = O2[j];
        sum += __half2float(a.x) * __half2float(b_val.x)
             + __half2float(a.y) * __half2float(b_val.y);
    }

    D[b * l_batch_stride + h * l_head_stride + n] = sum;
}

// ---------------------------------------------------------------------------
// Backward kernel — KV-outer loop, multi-thread per KV row
// ---------------------------------------------------------------------------
template <typename Tile>
__global__ void __launch_bounds__(Tile::kNumWarps * 32)
flash_attn_bwd_kernel(FlashAttnBwdParams params) {

    constexpr int Br = Tile::Br;
    constexpr int Bc = Tile::Bc;
    constexpr int D  = Tile::d;
    constexpr int NUM_THREADS = Tile::kNumWarps * 32;
    // How many threads collaborate on each KV row
    constexpr int THREADS_PER_KV = NUM_THREADS / Bc;
    static_assert(THREADS_PER_KV >= 1 && (THREADS_PER_KV & (THREADS_PER_KV - 1)) == 0,
                  "THREADS_PER_KV must be a power of 2");

    // Thread-to-KV-row mapping
    const int tid = threadIdx.x;
    const int kv_row = tid / THREADS_PER_KV;   // which KV row this thread owns
    const int lane   = tid % THREADS_PER_KV;   // position within the KV row group

    // GQA: grid is over (B * num_heads_k), each block handles one KV head
    // and accumulates from all Q heads that map to it.
    const int bh_idx = blockIdx.y;
    const int b_idx = bh_idx / params.num_heads_k;
    const int kv_h_idx = bh_idx % params.num_heads_k;
    const int heads_per_kv = params.num_heads / params.num_heads_k;
    const int kv_blk = blockIdx.x;
    const int kv_start = kv_blk * Bc;
    if (kv_start >= params.seq_len_k) return;
    const int valid_kv = min(Bc, params.seq_len_k - kv_start);

    const bool owns_kv_row = (kv_row < Bc) && (kv_row < valid_kv);

    // Base pointers for K, V (using kv_h_idx)
    const half* K_blk = params.K + b_idx * params.k_batch_stride
                                 + kv_h_idx * params.k_head_stride
                                 + kv_start * params.k_seq_stride;
    const half* V_blk = params.V + b_idx * params.v_batch_stride
                                 + kv_h_idx * params.v_head_stride
                                 + kv_start * params.v_seq_stride;

    // Shared memory layout:
    //   smem_K  [Bc, D]  half
    //   smem_V  [Bc, D]  half
    //   smem_Q  [Br, D]  half   — reloaded each Q block
    //   smem_dO [Br, D]  half   — reloaded each Q block
    extern __shared__ __align__(128) char smem_raw[];  // aligned for vector (128-bit) accesses
    half* smem_K  = reinterpret_cast<half*>(smem_raw);
    half* smem_V  = smem_K + Bc * D;
    half* smem_Q  = smem_V + Bc * D;
    half* smem_dO = smem_Q + Br * D;

    // dK, dV accumulators in registers (each thread has partial sums)
    float my_dK[D];
    float my_dV[D];
    #pragma unroll
    for (int j = 0; j < D; j++) { my_dK[j] = 0.0f; my_dV[j] = 0.0f; }

    // Load K, V (stay for entire block)
    bwd_load_tile<Bc, D, NUM_THREADS>(K_blk, params.k_seq_stride, smem_K, valid_kv);
    bwd_load_tile<Bc, D, NUM_THREADS>(V_blk, params.v_seq_stride, smem_V, valid_kv);
    __syncthreads();

    // Loop over all Q heads that map to this KV head
    const int num_q_blocks = cdiv(params.seq_len_q, Br);

    for (int qh = 0; qh < heads_per_kv; qh++) {
        const int h_idx = kv_h_idx * heads_per_kv + qh;
        // Causal optimization: skip Q blocks that are entirely before this KV block
        const int q_start_blk = bwd_first_q_block(params.is_causal, kv_start, Br, params.seq_len_q, params.seq_len_k);

    for (int q_blk = q_start_blk; q_blk < num_q_blocks; q_blk++) {
        const int q_start = q_blk * Br;
        const int valid_q = min(Br, params.seq_len_q - q_start);

        // Load Q, dO for this (q_head, q_block)
        const half* Q_blk_ptr = params.Q + b_idx * params.q_batch_stride
                                         + h_idx * params.q_head_stride
                                         + q_start * params.q_seq_stride;
        const half* dO_blk_ptr = params.dO + b_idx * params.o_batch_stride
                                           + h_idx * params.o_head_stride
                                           + q_start * params.o_seq_stride;

        bwd_load_tile<Br, D, NUM_THREADS>(Q_blk_ptr, params.q_seq_stride, smem_Q, valid_q);
        bwd_load_tile<Br, D, NUM_THREADS>(dO_blk_ptr, params.o_seq_stride, smem_dO, valid_q);
        __syncthreads();

        // L, D, dQ indexed by Q head
        const float* L_ptr = params.L + b_idx * params.l_batch_stride
                                      + h_idx * params.l_head_stride
                                      + q_start;
        const float* D_ptr = params.D + b_idx * params.l_batch_stride
                                      + h_idx * params.l_head_stride
                                      + q_start;
        float* dQ_ptr = params.dQ + b_idx * params.dq_batch_stride
                                  + h_idx * params.dq_head_stride
                                  + q_start * params.dq_seq_stride;

        // Each thread in the KV row group processes a strided subset of Q rows
        if (owns_kv_row) {
            for (int qi = lane; qi < Br && (q_start + qi) < params.seq_len_q; qi += THREADS_PER_KV) {
                // Causal check
                if (params.is_causal && (kv_start + kv_row) > (q_start + qi + (params.seq_len_k - params.seq_len_q))) continue;

                // Recompute S = Q[qi] . K[kv_row] * scale
                float s = 0.0f;
                const half2* q2 = reinterpret_cast<const half2*>(smem_Q + qi * D);
                const half2* k2 = reinterpret_cast<const half2*>(smem_K + kv_row * D);
                constexpr int D2 = D / 2;
                #pragma unroll
                for (int j = 0; j < D2; j++) {
                    half2 a = q2[j], b_val = k2[j];
                    s += __half2float(a.x) * __half2float(b_val.x)
                       + __half2float(a.y) * __half2float(b_val.y);
                }
                s *= params.softmax_scale;

                // P = exp(S - L[qi])
                float li = L_ptr[qi];
                float p = expf(s - li);

                // D_i (precomputed)
                float di = D_ptr[qi];

                // dP = dO[qi] . V[kv_row]
                float dp = 0.0f;
                const half2* do2 = reinterpret_cast<const half2*>(smem_dO + qi * D);
                const half2* v2  = reinterpret_cast<const half2*>(smem_V + kv_row * D);
                #pragma unroll
                for (int j = 0; j < D2; j++) {
                    half2 a = do2[j], b_val = v2[j];
                    dp += __half2float(a.x) * __half2float(b_val.x)
                        + __half2float(a.y) * __half2float(b_val.y);
                }

                // dS = P * (dP - D_i)
                float ds = p * (dp - di);

                // dS_scaled = dS * softmax_scale
                float ds_scaled = ds * params.softmax_scale;

                // Accumulate dV[kv_row] += P * dO[qi]
                // Accumulate dK[kv_row] += dS_scaled * Q[qi]
                #pragma unroll
                for (int j = 0; j < D2; j++) {
                    half2 do_val = do2[j];
                    my_dV[j * 2]     += p * __half2float(do_val.x);
                    my_dV[j * 2 + 1] += p * __half2float(do_val.y);

                    half2 q_val = q2[j];
                    my_dK[j * 2]     += ds_scaled * __half2float(q_val.x);
                    my_dK[j * 2 + 1] += ds_scaled * __half2float(q_val.y);
                }

                // dQ[qi] += dS_scaled * K[kv_row] — atomicAdd to FP32 buffer
                float* dq_row = dQ_ptr + qi * params.dq_seq_stride;
                #pragma unroll
                for (int j = 0; j < D2; j++) {
                    half2 k_val = k2[j];
                    atomicAdd(&dq_row[j * 2],     ds_scaled * __half2float(k_val.x));
                    atomicAdd(&dq_row[j * 2 + 1], ds_scaled * __half2float(k_val.y));
                }
            }
        }
        __syncthreads();
    }  // end q_blk loop
    }  // end qh (GQA Q-head) loop

    // -----------------------------------------------------------------------
    // Reduce dK, dV across lanes within each KV row group via warp shuffle.
    // THREADS_PER_KV is always a power of 2 and <= 32, so all threads in a
    // group are within the same warp — no __syncthreads needed.
    // -----------------------------------------------------------------------
    if (THREADS_PER_KV > 1) {
        #pragma unroll
        for (int offset = THREADS_PER_KV / 2; offset > 0; offset /= 2) {
            #pragma unroll
            for (int j = 0; j < D; j++) {
                my_dK[j] += __shfl_down_sync(0xFFFFFFFF, my_dK[j], offset, THREADS_PER_KV);
                my_dV[j] += __shfl_down_sync(0xFFFFFFFF, my_dV[j], offset, THREADS_PER_KV);
            }
        }
    }

    // Write dK, dV — only lane 0 of each valid group writes to global memory
    if (lane == 0 && owns_kv_row) {
        half* dK_out = params.dK + b_idx * params.dk_batch_stride
                                 + kv_h_idx * params.dk_head_stride
                                 + (kv_start + kv_row) * params.dk_seq_stride;
        half* dV_out = params.dV + b_idx * params.dv_batch_stride
                                 + kv_h_idx * params.dv_head_stride
                                 + (kv_start + kv_row) * params.dv_seq_stride;

        half2* dk2 = reinterpret_cast<half2*>(dK_out);
        half2* dv2 = reinterpret_cast<half2*>(dV_out);
        constexpr int D2 = D / 2;
        #pragma unroll
        for (int j = 0; j < D2; j++) {
            half2 dk_val, dv_val;
            dk_val.x = __float2half(my_dK[j * 2]);
            dk_val.y = __float2half(my_dK[j * 2 + 1]);
            dv_val.x = __float2half(my_dV[j * 2]);
            dv_val.y = __float2half(my_dV[j * 2 + 1]);
            dk2[j] = dk_val;
            dv2[j] = dv_val;
        }
    }
}

// ============================================================================
// VOLTA WMMA Backward — SM >= 7.0
// ============================================================================
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700

using namespace nvcuda;

// ---------------------------------------------------------------------------
// WMMA templates (duplicated from forward — needed in this compilation unit)
// ---------------------------------------------------------------------------

// C[M,N] = A[M,K] @ B^T[N,K]  (B loaded as col_major)
template <int M, int N, int K_DIM, int NUM_WARPS>
__device__ __forceinline__ void bwd_wmma_gemm_ABt(
    const half* smem_A, const half* smem_B,
    float* smem_C, int warp_id
) {
    constexpr int T = 16;
    constexpr int ntm = M / T, ntn = N / T, total = ntm * ntn;
    constexpr int per_warp = (total + NUM_WARPS - 1) / NUM_WARPS;
    #pragma unroll
    for (int t = 0; t < per_warp; t++) {
        int idx = warp_id + t * NUM_WARPS;
        if (idx >= total) break;
        int tm = idx / ntn, tn = idx % ntn;
        wmma::fragment<wmma::matrix_a, T,T,T, half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, T,T,T, half, wmma::col_major> b;
        wmma::fragment<wmma::accumulator, T,T,T, float> c;
        wmma::fill_fragment(c, 0.0f);
        #pragma unroll
        for (int k = 0; k < K_DIM; k += T) {
            wmma::load_matrix_sync(a, smem_A + tm*T*K_DIM + k, K_DIM);
            wmma::load_matrix_sync(b, smem_B + tn*T*K_DIM + k, K_DIM);
            wmma::mma_sync(c, a, b, c);
        }
        wmma::store_matrix_sync(smem_C + tm*T*N + tn*T, c, N, wmma::mem_row_major);
    }
}

// C[M,N_OUT] = A[M,K] @ B[K,N_FULL] (B row-major, pre-offset to column chunk)
template <int M, int N_OUT, int K_DIM, int NUM_WARPS>
__device__ __forceinline__ void bwd_wmma_gemm_AB(
    const half* smem_A, const half* smem_B, int B_stride,
    float* smem_C, int warp_id
) {
    constexpr int T = 16;
    constexpr int ntm = M / T, ntn = N_OUT / T, total = ntm * ntn;
    constexpr int per_warp = (total + NUM_WARPS - 1) / NUM_WARPS;
    #pragma unroll
    for (int t = 0; t < per_warp; t++) {
        int idx = warp_id + t * NUM_WARPS;
        if (idx >= total) break;
        int tm = idx / ntn, tn = idx % ntn;
        wmma::fragment<wmma::matrix_a, T,T,T, half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, T,T,T, half, wmma::row_major> b;
        wmma::fragment<wmma::accumulator, T,T,T, float> c;
        wmma::fill_fragment(c, 0.0f);
        #pragma unroll
        for (int k = 0; k < K_DIM; k += T) {
            wmma::load_matrix_sync(a, smem_A + tm*T*K_DIM + k, K_DIM);
            wmma::load_matrix_sync(b, smem_B + k*B_stride + tn*T, B_stride);
            wmma::mma_sync(c, a, b, c);
        }
        wmma::store_matrix_sync(smem_C + tm*T*N_OUT + tn*T, c, N_OUT, wmma::mem_row_major);
    }
}

// Accumulator fragments of a [OUT_R, OUT_C] float matrix spread over the warps
// of a block: tile idx = warp_id + t * NUM_WARPS (t < PER_WARP), row-major tiles.
template <int OUT_R, int OUT_C, int NUM_WARPS>
struct WarpAccum {
    static constexpr int T = 16;
    static constexpr int kCols = OUT_C / T;
    static constexpr int kTiles = (OUT_R / T) * kCols;
    static constexpr int kPerWarp = (kTiles + NUM_WARPS - 1) / NUM_WARPS;
    wmma::fragment<wmma::accumulator, T, T, T, float> frag[kPerWarp];

    __device__ __forceinline__ void zero() {
        #pragma unroll
        for (int t = 0; t < kPerWarp; t++) wmma::fill_fragment(frag[t], 0.0f);
    }

    // frag += A^T @ B, A [M, OUT_R] row-major (stride A_stride, loaded col_major
    // to transpose), B [M, OUT_C] row-major (stride B_stride).
    template <int M>
    __device__ __forceinline__ void add_AtB(const half* smem_A, int A_stride,
                                            const half* smem_B, int B_stride, int warp_id) {
        #pragma unroll
        for (int t = 0; t < kPerWarp; t++) {
            const int idx = warp_id + t * NUM_WARPS;
            if (idx >= kTiles) break;  // warp-uniform
            const int tr = idx / kCols, tc = idx % kCols;
            wmma::fragment<wmma::matrix_a, T, T, T, half, wmma::col_major> a;
            wmma::fragment<wmma::matrix_b, T, T, T, half, wmma::row_major> b;
            #pragma unroll
            for (int m = 0; m < M; m += T) {
                wmma::load_matrix_sync(a, smem_A + m * A_stride + tr * T, A_stride);
                wmma::load_matrix_sync(b, smem_B + m * B_stride + tc * T, B_stride);
                wmma::mma_sync(frag[t], a, b, frag[t]);
            }
        }
    }

    // Write the first `valid_rows` rows as half to out (row stride out_stride),
    // through a per-warp 16x16 float staging tile in shared memory.
    __device__ __forceinline__ void store_half(half* out, int out_stride, int valid_rows,
                                               float* warp_staging, int warp_id, int lane) {
        #pragma unroll
        for (int t = 0; t < kPerWarp; t++) {
            const int idx = warp_id + t * NUM_WARPS;
            if (idx >= kTiles) break;
            const int tr = idx / kCols, tc = idx % kCols;
            wmma::store_matrix_sync(warp_staging, frag[t], T, wmma::mem_row_major);
            __syncwarp();
            #pragma unroll
            for (int e = lane; e < T * T; e += 32) {
                const int row = tr * T + e / T;
                if (row < valid_rows) {
                    out[row * out_stride + tc * T + e % T] = __float2half(warp_staging[e]);
                }
            }
            __syncwarp();
        }
    }
};

// ---------------------------------------------------------------------------
// Backward kernel — Volta WMMA
// ---------------------------------------------------------------------------
template <typename Tile>
__global__ void __launch_bounds__(Tile::kNumWarps * 32)
flash_attn_bwd_volta_kernel(FlashAttnBwdParams params) {

    constexpr int Br = Tile::Br;
    constexpr int Bc = Tile::Bc;
    constexpr int D  = Tile::d;
    constexpr int NUM_THREADS = Tile::kNumWarps * 32;
    constexpr int NUM_WARPS = Tile::kNumWarps;
    constexpr int DQ_CHUNK = Bc;  // dQ computed in chunks of Bc columns

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;

    // Grid: (num_kv_blocks, B * H_kv)
    const int bh_idx = blockIdx.y;
    const int b_idx = bh_idx / params.num_heads_k;
    const int kv_h_idx = bh_idx % params.num_heads_k;
    const int heads_per_kv = params.num_heads / params.num_heads_k;
    const int kv_blk = blockIdx.x;
    const int kv_start = kv_blk * Bc;
    if (kv_start >= params.seq_len_k) return;
    const int valid_kv = min(Bc, params.seq_len_k - kv_start);

    // K, V base pointers
    const half* K_blk = params.K + b_idx * params.k_batch_stride
                                 + kv_h_idx * params.k_head_stride
                                 + kv_start * params.k_seq_stride;
    const half* V_blk = params.V + b_idx * params.v_batch_stride
                                 + kv_h_idx * params.v_head_stride
                                 + kv_start * params.v_seq_stride;

    // Shared memory layout (size: Tile::smem_bytes()):
    //   smem_K:    [Bc, D] half
    //   smem_V:    [Bc, D] half
    //   smem_Q:    [Br, D] half   (reloaded per Q block)
    //   smem_dO:   [Br, D] half   (reloaded per Q block)
    //   smem_S:    [Br, Bc] float (S → P, then reused for dQ chunks)
    //   smem_dP:   [Br, Bc] float (dP → dS)
    //   smem_half: [Br, Bc] half  (P_half → dS_half)
    //   staging:   [NUM_WARPS, 16*16] float (dK/dV write-back)
    // dK, dV are accumulated in registers (WarpAccum fragments) for the whole block.
    extern __shared__ __align__(128) char smem_raw[];  // WMMA needs 32-byte alignment
    half*  smem_K    = reinterpret_cast<half*>(smem_raw);
    half*  smem_V    = smem_K + Bc * D;
    half*  smem_Q    = smem_V + Bc * D;
    half*  smem_dO   = smem_Q + Br * D;
    float* smem_S    = reinterpret_cast<float*>(smem_dO + Br * D);
    float* smem_dP   = smem_S + Br * Bc;
    half*  smem_half = reinterpret_cast<half*>(smem_dP + Br * Bc);
    float* staging   = reinterpret_cast<float*>(smem_half + Br * Bc);

    WarpAccum<Bc, D, NUM_WARPS> dK_acc, dV_acc;
    dK_acc.zero();
    dV_acc.zero();

    // Load K, V (stay for entire block)
    bwd_load_tile<Bc, D, NUM_THREADS>(K_blk, params.k_seq_stride, smem_K, valid_kv);
    bwd_load_tile<Bc, D, NUM_THREADS>(V_blk, params.v_seq_stride, smem_V, valid_kv);
    __syncthreads();

    const int num_q_blocks = cdiv(params.seq_len_q, Br);

    // Loop over Q heads that map to this KV head
    for (int qh = 0; qh < heads_per_kv; qh++) {
        const int h_idx = kv_h_idx * heads_per_kv + qh;
        const int q_start_blk = bwd_first_q_block(params.is_causal, kv_start, Br, params.seq_len_q, params.seq_len_k);

    for (int q_blk = q_start_blk; q_blk < num_q_blocks; q_blk++) {
        const int q_start = q_blk * Br;
        const int valid_q = min(Br, params.seq_len_q - q_start);

        // Load Q, dO
        const half* Q_ptr = params.Q + b_idx * params.q_batch_stride
                                     + h_idx * params.q_head_stride
                                     + q_start * params.q_seq_stride;
        const half* dO_ptr = params.dO + b_idx * params.o_batch_stride
                                       + h_idx * params.o_head_stride
                                       + q_start * params.o_seq_stride;

        bwd_load_tile<Br, D, NUM_THREADS>(Q_ptr, params.q_seq_stride, smem_Q, valid_q);
        bwd_load_tile<Br, D, NUM_THREADS>(dO_ptr, params.o_seq_stride, smem_dO, valid_q);
        __syncthreads();

        // L, D pointers (Q head indexed)
        const float* L_ptr = params.L + b_idx * params.l_batch_stride
                                      + h_idx * params.l_head_stride + q_start;
        const float* D_ptr = params.D + b_idx * params.l_batch_stride
                                      + h_idx * params.l_head_stride + q_start;

        // ---- Step 1: S = Q @ K^T via WMMA ----
        bwd_wmma_gemm_ABt<Br, Bc, D, NUM_WARPS>(smem_Q, smem_K, smem_S, warp_id);
        __syncthreads();

        // ---- Step 2: P = exp(S * scale - L), with masking ----
        {
            constexpr int total_elems = Br * Bc;
            #pragma unroll
            for (int i = tid; i < total_elems; i += NUM_THREADS) {
                int qi = i / Bc;
                int kj = i % Bc;
                float s = smem_S[i] * params.softmax_scale;
                bool invalid = (q_start + qi >= params.seq_len_q) ||
                               (kv_start + kj >= params.seq_len_k) ||
                               (params.is_causal &&
                                (kv_start + kj) > (q_start + qi + (params.seq_len_k - params.seq_len_q)));
                float p = invalid ? 0.0f : expf(s - L_ptr[qi]);
                smem_S[i] = p;  // S now holds P
            }
        }
        __syncthreads();

        // ---- Step 3: P_half for WMMA ----
        {
            constexpr int total_P = Br * Bc;
            #pragma unroll
            for (int i = tid; i < total_P; i += NUM_THREADS)
                smem_half[i] = __float2half(smem_S[i]);
        }
        __syncthreads();

        // ---- Step 4: dV_acc += P_half^T @ dO via WMMA (registers) ----
        dV_acc.template add_AtB<Br>(smem_half, Bc,   // A = P_half [Br, Bc], stride Bc
                           smem_dO, D,      // B = dO [Br, D], stride D
                           warp_id);
        __syncthreads();

        // ---- Step 5: dP = dO @ V^T via WMMA ----
        bwd_wmma_gemm_ABt<Br, Bc, D, NUM_WARPS>(smem_dO, smem_V, smem_dP, warp_id);
        __syncthreads();

        // ---- Step 6: dS_scaled = P * (dP - D) * scale ----
        {
            constexpr int total_elems = Br * Bc;
            #pragma unroll
            for (int i = tid; i < total_elems; i += NUM_THREADS) {
                int qi = i / Bc;
                float p = smem_S[i];     // P from step 2
                float dp = smem_dP[i];
                // Padding rows (qi >= valid_q) have P == 0; don't read D past the
                // sequence end (OOB, and 0 * NaN would poison dK).
                float ds = (qi < valid_q) ? p * (dp - D_ptr[qi]) : 0.0f;
                smem_dP[i] = ds * params.softmax_scale;  // dS_scaled
            }
        }
        // smem_S (P) is now free!
        __syncthreads();

        // ---- Step 7: dS_half ----
        {
            constexpr int total_dS = Br * Bc;
            #pragma unroll
            for (int i = tid; i < total_dS; i += NUM_THREADS)
                smem_half[i] = __float2half(smem_dP[i]);
        }
        __syncthreads();

        // ---- Step 8: dK_acc += dS_half^T @ Q via WMMA (registers) ----
        dK_acc.template add_AtB<Br>(smem_half, Bc,   // A = dS_half [Br, Bc], stride Bc
                           smem_Q, D,       // B = Q [Br, D], stride D
                           warp_id);
        __syncthreads();

        // ---- Step 9: dQ += dS @ K (atomicAdd to global FP32 buffer) ----
        float* dQ_ptr = params.dQ + b_idx * params.dq_batch_stride
                                  + h_idx * params.dq_head_stride
                                  + q_start * params.dq_seq_stride;

        #pragma unroll
        for (int dn = 0; dn < D; dn += DQ_CHUNK) {
            // dQ_chunk[Br, DQ_CHUNK] = dS_half[Br, Bc] @ K[Bc, dn:dn+DQ_CHUNK]
            bwd_wmma_gemm_AB<Br, DQ_CHUNK, Bc, NUM_WARPS>(
                smem_half,          // A = dS_half [Br, Bc]
                smem_K + dn, D,     // B = K[:, dn:], stride D
                smem_S,             // C = reuse smem_S [Br, DQ_CHUNK]
                warp_id);
            __syncthreads();

            // AtomicAdd to global dQ
            constexpr int chunk_elems = Br * DQ_CHUNK;
            #pragma unroll
            for (int i = tid; i < chunk_elems; i += NUM_THREADS) {
                int qi = i / DQ_CHUNK;
                int dj = i % DQ_CHUNK;
                if (qi < valid_q)
                    ::atomicAdd(&dQ_ptr[qi * params.dq_seq_stride + dn + dj], smem_S[i]);
            }
            __syncthreads();
        }
    }  // end q_blk
    }  // end qh (GQA)

    // Write dK, dV from the register accumulators to global memory
    {
        half* dK_out = params.dK + b_idx * params.dk_batch_stride
                                 + kv_h_idx * params.dk_head_stride
                                 + kv_start * params.dk_seq_stride;
        half* dV_out = params.dV + b_idx * params.dv_batch_stride
                                 + kv_h_idx * params.dv_head_stride
                                 + kv_start * params.dv_seq_stride;
        float* warp_staging = staging + warp_id * 16 * 16;
        const int lane = tid % 32;
        dK_acc.store_half(dK_out, params.dk_seq_stride, valid_kv, warp_staging, warp_id, lane);
        dV_acc.store_half(dV_out, params.dv_seq_stride, valid_kv, warp_staging, warp_id, lane);
    }
}

#else  // __CUDA_ARCH__ < 700
template <typename Tile>
__global__ void __launch_bounds__(Tile::kNumWarps * 32)
flash_attn_bwd_volta_kernel(FlashAttnBwdParams params) {}
#endif  // __CUDA_ARCH__

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------

void precompute_D(
    const half* dO, const half* O, float* D,
    int B, int H, int N, int d,
    int o_batch_stride, int o_head_stride, int o_seq_stride,
    int l_batch_stride, int l_head_stride,
    cudaStream_t stream
) {
    const int total = B * H * N;
    const int block_size = 256;
    const int grid_size = cdiv(total, block_size);
    precompute_D_kernel<<<grid_size, block_size, 0, stream>>>(
        dO, O, D, B, H, N, d,
        o_batch_stride, o_head_stride, o_seq_stride,
        l_batch_stride, l_head_stride
    );
}

// Same dispatch scheme as the forward launchers, with one tile per head_dim,
// checked against the device's shared memory per block (clear error instead of
// a failed launch). With dK/dV in registers every WMMA backward tile fits 48 KB.
template <bool kWmma, typename F>
static void dispatch_bwd_tile(int head_dim, F&& f) {
    const int limit = max_dynamic_smem_per_block();
    bool ok = false;
    switch (head_dim) {
        case 32:  ok = with_first_fitting_tile<BwdTile<kWmma, 32, 32, 32>>(limit, f); break;
        case 64:  ok = with_first_fitting_tile<BwdTile<kWmma, 32, 32, 64>>(limit, f); break;
        case 96:  ok = with_first_fitting_tile<BwdTile<kWmma, 32, 16, 96>>(limit, f); break;
        case 128: ok = with_first_fitting_tile<BwdTile<kWmma, 16, 16, 128>>(limit, f); break;
        case 256: ok = with_first_fitting_tile<BwdTile<kWmma, 16, 16, 256>>(limit, f); break;
        default:  throw std::runtime_error("Unsupported head_dim for the backward");
    }
    if (!ok) throw_no_tile(kWmma ? "WMMA backward" : "CUDA-core backward", head_dim, limit);
}

template <typename Tile, typename Kernel>
static void launch_bwd(Kernel kernel, FlashAttnBwdParams& params, cudaStream_t stream) {
    constexpr int smem = Tile::smem_bytes();
    dim3 grid(cdiv(params.seq_len_k, Tile::Bc), params.batch_size * params.num_heads_k);
    dim3 block(Tile::kNumWarps * 32);
    FLASH_ATTN_CHECK_CUDA(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    kernel<<<grid, block, smem, stream>>>(params);
}

void flash_attn_bwd_volta_launch(FlashAttnBwdParams& params, cudaStream_t stream) {
    dispatch_bwd_tile<true>(params.head_dim, [&](auto tile) {
        using T = decltype(tile);
        launch_bwd<T>(flash_attn_bwd_volta_kernel<T>, params, stream);
    });
}

void flash_attn_bwd_pascal_launch(FlashAttnBwdParams& params, cudaStream_t stream) {
    dispatch_bwd_tile<false>(params.head_dim, [&](auto tile) {
        using T = decltype(tile);
        launch_bwd<T>(flash_attn_bwd_kernel<T>, params, stream);
    });
}

}  // namespace flash_attn_legacy
