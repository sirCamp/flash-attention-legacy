// ============================================================================
// Flash Attention v2 — Backward kernels
//   - flash_attn_bwd_kernel: CUDA cores (Pascal; FORCE_ARCH=pascal elsewhere)
//   - flash_attn_bwd_volta_kernel: WMMA tensor cores (Volta / Turing)
//   - precompute_D_kernel: D = rowsum(dO * O)
// Both backward kernels use the KV-outer loop of FlashAttention-2: one block
// per KV block, dK/dV accumulated in registers for the whole block, dQ
// accumulated in an FP32 buffer with one atomicAdd per element and KV block.
// ============================================================================

#include "flash_attn_common.h"

#include <mma.h>  // file scope: inside a namespace it breaks cuda_bf16.h

#include <string>

namespace flash_attn_legacy {

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
// Backward kernel — CUDA cores (Pascal), KV-outer loop, tiled like FA2.
//
// One block per (KV block of Bc rows, batch x KV head). For every Q head of
// the group (GQA) and every Q block of Br rows:
//   1. S, P = exp(S*scale - L), dP = dO @ V^T, dS = P * (dP - D) * scale for
//      the [Br, Bc] tile: each thread computes some entries (128-bit dot
//      products on padded rows: bank-conflict free) and stores P, dS in shared
//      memory (row stride Bc + 1: conflict free by rows and by columns).
//   2. dV[j] += P^T dO, dK[j] += dS^T Q: each thread owns one KV row j and a
//      slice of columns; the accumulators stay in registers for the whole block.
//   3. dQ[i] += dS K: each thread computes one Q row i and a slice of columns
//      over the whole KV block, then ONE atomicAdd per element and KV block
//      (the previous kernel issued one per element and (q, kv) pair).
//
// Shared memory (Tile::smem_bytes()):
//   K, V [Bc, kLdH] half; Q, dO [Br, kLdH] half (kLdH = d + 8)
//   P, dS [Br, kLdF] float (kLdF = Bc + 1); L, D [Br] float
// ---------------------------------------------------------------------------
template <typename Tile>
__global__ void __launch_bounds__(Tile::kNumWarps * 32)
flash_attn_bwd_kernel(FlashAttnBwdParams params) {
    constexpr int Br = Tile::Br;
    constexpr int Bc = Tile::Bc;
    constexpr int D  = Tile::d;
    constexpr int NT = Tile::kNumWarps * 32;
    constexpr int LDH = Tile::kLdH;
    constexpr int LDF = Tile::kLdF;
    // dK/dV ownership: KV row j = tid % Bc, columns [kv_g * KV_COLS, +KV_COLS)
    constexpr int KV_GROUPS = NT / Bc;
    constexpr int KV_COLS = D / KV_GROUPS;
    // dQ ownership: Q row i = tid % Br, columns [q_g * Q_COLS, +Q_COLS)
    constexpr int Q_GROUPS = NT / Br;
    constexpr int Q_COLS = D / Q_GROUPS;
    static_assert(NT % Bc == 0 && D % KV_GROUPS == 0 && KV_COLS % 2 == 0, "dK/dV split");
    static_assert(NT % Br == 0 && D % Q_GROUPS == 0 && Q_COLS % 2 == 0, "dQ split");

    const int tid = threadIdx.x;
    const int bh_idx = blockIdx.y;
    const int b_idx = bh_idx / params.num_heads_k;
    const int kv_h_idx = bh_idx % params.num_heads_k;
    const int heads_per_kv = params.num_heads / params.num_heads_k;
    const int kv_start = blockIdx.x * Bc;
    if (kv_start >= params.seq_len_k) return;
    const int valid_kv = min(Bc, params.seq_len_k - kv_start);
    const int causal_off = params.seq_len_k - params.seq_len_q;

    extern __shared__ __align__(128) char smem_raw[];  // aligned for 128-bit accesses
    half*  smem_K  = reinterpret_cast<half*>(smem_raw);
    half*  smem_V  = smem_K + Bc * LDH;
    half*  smem_Q  = smem_V + Bc * LDH;
    half*  smem_dO = smem_Q + Br * LDH;
    float* smem_P  = reinterpret_cast<float*>(smem_dO + Br * LDH);
    float* smem_dS = smem_P + Br * LDF;
    float* smem_L  = smem_dS + Br * LDF;
    float* smem_D  = smem_L + Br;

    load_tile_g2s<Bc, D, NT, LDH>(params.K + b_idx * params.k_batch_stride + kv_h_idx * params.k_head_stride
                                           + kv_start * params.k_seq_stride,
                                  params.k_seq_stride, smem_K, valid_kv);
    load_tile_g2s<Bc, D, NT, LDH>(params.V + b_idx * params.v_batch_stride + kv_h_idx * params.v_head_stride
                                           + kv_start * params.v_seq_stride,
                                  params.v_seq_stride, smem_V, valid_kv);

    const int kv_j = tid % Bc, kv_c0 = (tid / Bc) * KV_COLS;
    const int q_i = tid % Br, q_c0 = (tid / Br) * Q_COLS;
    float acc_dK[KV_COLS], acc_dV[KV_COLS];
    #pragma unroll
    for (int c = 0; c < KV_COLS; c++) { acc_dK[c] = 0.0f; acc_dV[c] = 0.0f; }

    const int num_q_blocks = cdiv(params.seq_len_q, Br);
    const int q_start_blk = bwd_first_q_block(params.is_causal, kv_start, Br,
                                              params.seq_len_q, params.seq_len_k);

    for (int qh = 0; qh < heads_per_kv; qh++) {
        const int h_idx = kv_h_idx * heads_per_kv + qh;
        for (int q_blk = q_start_blk; q_blk < num_q_blocks; q_blk++) {
            const int q_start = q_blk * Br;
            const int valid_q = min(Br, params.seq_len_q - q_start);

            // The previous iteration's readers of Q/dO/P/dS passed the barrier
            // at the end of the loop body.
            load_tile_g2s<Br, D, NT, LDH>(params.Q + b_idx * params.q_batch_stride + h_idx * params.q_head_stride
                                                   + q_start * params.q_seq_stride,
                                          params.q_seq_stride, smem_Q, valid_q);
            load_tile_g2s<Br, D, NT, LDH>(params.dO + b_idx * params.o_batch_stride + h_idx * params.o_head_stride
                                                    + q_start * params.o_seq_stride,
                                          params.o_seq_stride, smem_dO, valid_q);
            if (tid < Br) {
                const int lrow = b_idx * params.l_batch_stride + h_idx * params.l_head_stride + q_start + tid;
                smem_L[tid] = (tid < valid_q) ? params.L[lrow] : 0.0f;
                smem_D[tid] = (tid < valid_q) ? params.D[lrow] : 0.0f;
            }
            __syncthreads();

            // 1. P and dS for the [Br, Bc] tile (j fastest across threads)
            #pragma unroll
            for (int e = tid; e < Br * Bc; e += NT) {
                const int i = e / Bc, j = e % Bc;
                float p = 0.0f, ds = 0.0f;
                const bool valid = i < valid_q && j < valid_kv &&
                    !(params.is_causal && (kv_start + j) > (q_start + i + causal_off));
                if (valid) {
                    const float s = half_dot<D>(smem_Q + i * LDH, smem_K + j * LDH) * params.softmax_scale;
                    p = expf(s - smem_L[i]);
                    const float dp = half_dot<D>(smem_dO + i * LDH, smem_V + j * LDH);
                    ds = p * (dp - smem_D[i]) * params.softmax_scale;
                }
                smem_P[i * LDF + j] = p;
                smem_dS[i * LDF + j] = ds;
            }
            __syncthreads();

            // 2. dV[j] += P^T dO, dK[j] += dS^T Q (registers; row j, column slice)
            #pragma unroll 4
            for (int i = 0; i < Br; i++) {
                const float p = smem_P[i * LDF + kv_j];
                const float ds = smem_dS[i * LDF + kv_j];
                const half2* do2 = reinterpret_cast<const half2*>(smem_dO + i * LDH + kv_c0);
                const half2* q2  = reinterpret_cast<const half2*>(smem_Q + i * LDH + kv_c0);
                #pragma unroll
                for (int c = 0; c < KV_COLS / 2; c++) {
                    const float2 dof = __half22float2(do2[c]);
                    const float2 qf  = __half22float2(q2[c]);
                    acc_dV[2 * c]     = fmaf(p, dof.x, acc_dV[2 * c]);
                    acc_dV[2 * c + 1] = fmaf(p, dof.y, acc_dV[2 * c + 1]);
                    acc_dK[2 * c]     = fmaf(ds, qf.x, acc_dK[2 * c]);
                    acc_dK[2 * c + 1] = fmaf(ds, qf.y, acc_dK[2 * c + 1]);
                }
            }

            // 3. dQ[i] += dS K over the whole KV block, one atomicAdd per element
            if (q_i < valid_q) {
                float acc_dQ[Q_COLS];
                #pragma unroll
                for (int c = 0; c < Q_COLS; c++) acc_dQ[c] = 0.0f;
                #pragma unroll 4
                for (int j = 0; j < Bc; j++) {
                    const float ds = smem_dS[q_i * LDF + j];
                    const half2* k2 = reinterpret_cast<const half2*>(smem_K + j * LDH + q_c0);
                    #pragma unroll
                    for (int c = 0; c < Q_COLS / 2; c++) {
                        const float2 kf = __half22float2(k2[c]);
                        acc_dQ[2 * c]     = fmaf(ds, kf.x, acc_dQ[2 * c]);
                        acc_dQ[2 * c + 1] = fmaf(ds, kf.y, acc_dQ[2 * c + 1]);
                    }
                }
                float* dq_row = params.dQ + b_idx * params.dq_batch_stride + h_idx * params.dq_head_stride
                                          + (q_start + q_i) * params.dq_seq_stride + q_c0;
                #pragma unroll
                for (int c = 0; c < Q_COLS; c++) atomicAdd(&dq_row[c], acc_dQ[c]);
            }
            __syncthreads();  // P/dS/Q/dO are overwritten by the next iteration
        }
    }

    // Write dK, dV (row j of this KV block, column slice)
    if (kv_j < valid_kv) {
        half2* dk2 = reinterpret_cast<half2*>(params.dK + b_idx * params.dk_batch_stride
            + kv_h_idx * params.dk_head_stride + (kv_start + kv_j) * params.dk_seq_stride + kv_c0);
        half2* dv2 = reinterpret_cast<half2*>(params.dV + b_idx * params.dv_batch_stride
            + kv_h_idx * params.dv_head_stride + (kv_start + kv_j) * params.dv_seq_stride + kv_c0);
        #pragma unroll
        for (int c = 0; c < KV_COLS / 2; c++) {
            dk2[c] = __floats2half2_rn(acc_dK[2 * c], acc_dK[2 * c + 1]);
            dv2[c] = __floats2half2_rn(acc_dV[2 * c], acc_dV[2 * c + 1]);
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
    load_tile_g2s<Bc, D, NUM_THREADS>(K_blk, params.k_seq_stride, smem_K, valid_kv);
    load_tile_g2s<Bc, D, NUM_THREADS>(V_blk, params.v_seq_stride, smem_V, valid_kv);
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

        load_tile_g2s<Br, D, NUM_THREADS>(Q_ptr, params.q_seq_stride, smem_Q, valid_q);
        load_tile_g2s<Br, D, NUM_THREADS>(dO_ptr, params.o_seq_stride, smem_dO, valid_q);
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
