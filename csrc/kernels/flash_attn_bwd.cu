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
#include "flash_attn_wmma.h"


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

// ---------------------------------------------------------------------------
// Backward kernel — WMMA (Volta / Turing), FA2-style.
// Grid: (key blocks of Bc, B * H_kv). The block keeps its K, V and loops over
// the Q heads of its KV head (GQA) and the query blocks of Br rows that can see
// its keys. Warp w owns the 16 keys of band w / kSplitD and the d / kSplitD
// columns of slice w % kSplitD of dK / dV, accumulated in WMMA fragments. Per
// query block, each warp on its own computes, as transposed [16 keys, Br]
// fragments,
//   S^T = K_band Q^T,  P^T = exp(S^T * scale - L)  (masked to 0)
//   dV += P^T dO,  dP^T = V_band dO^T,  dS^T = P^T (dP^T - D) * scale,  dK += dS^T Q
// with the element-wise steps done on the fragments themselves (the row and
// column of each element come from wmma_acc_layout). P^T / dS^T reach the
// next WMMA as half through the warp's buffer, and dS^T also goes to a shared
// [Bc, Br] buffer: after one block barrier all warps compute dQ = dS K for the
// query block and add it to the fp32 dQ buffer with coalesced atomics (one per
// element and key block). Two block barriers per query block.
// ---------------------------------------------------------------------------
template <typename Tile>
__global__ void __launch_bounds__(Tile::kNumWarps * 32)
flash_attn_bwd_volta_kernel(FlashAttnBwdParams params) {
    constexpr int Br = Tile::Br;
    constexpr int Bc = Tile::Bc;
    constexpr int D  = Tile::d;
    constexpr int NUM_THREADS = Tile::kNumWarps * 32;
    constexpr int NUM_WARPS = Tile::kNumWarps;
    constexpr int SPLIT = Tile::kSplitD;
    constexpr int LDH = Tile::kLdH;
    constexpr int LDB = Tile::kLdB;
    constexpr int T = 16;
    constexpr int QT = Br / T;            // query tiles of a band's S^T / dP^T
    constexpr int DP = D / SPLIT;         // dK / dV columns of a warp
    constexpr int DPT = DP / T;           // dK / dV fragments of a warp
    constexpr int NE = kAccElems;
    using Acc = wmma::fragment<wmma::accumulator, T, T, T, float>;
    using FragA = wmma::fragment<wmma::matrix_a, T, T, T, half, wmma::row_major>;
    using FragAt = wmma::fragment<wmma::matrix_a, T, T, T, half, wmma::col_major>;
    using FragB = wmma::fragment<wmma::matrix_b, T, T, T, half, wmma::row_major>;
    using FragBt = wmma::fragment<wmma::matrix_b, T, T, T, half, wmma::col_major>;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    const int band = warp_id / SPLIT;      // keys [16 band, 16 band + 16) of the block
    const int part = warp_id % SPLIT;      // dK / dV columns [part DP, part DP + DP)

    const int bh_idx = blockIdx.y;
    const int b_idx = bh_idx / params.num_heads_k;
    const int kv_h_idx = bh_idx % params.num_heads_k;
    const int heads_per_kv = params.num_heads / params.num_heads_k;
    const int kv_start = blockIdx.x * Bc;
    if (kv_start >= params.seq_len_k) return;
    const int valid_kv = min(Bc, params.seq_len_k - kv_start);
    const int causal_off = params.seq_len_k - params.seq_len_q;
    const float scale_log2 = params.softmax_scale * kLog2e;

    extern __shared__ __align__(128) char smem_raw[];  // WMMA needs 32-byte alignment
    half*  smem_K  = reinterpret_cast<half*>(smem_raw);
    half*  smem_V  = smem_K + Bc * LDH;
    half*  smem_Q  = smem_V + Bc * LDH;
    half*  smem_dO = smem_Q + Br * LDH;
    float* smem_L  = reinterpret_cast<float*>(smem_dO + Br * LDH);  // L * log2(e)
    float* smem_D  = smem_L + Br;
    half*  smem_dSt = reinterpret_cast<half*>(smem_D + Br);           // dS^T [Bc, LDB]
    half*  wbuf = smem_dSt + Bc * LDB + warp_id * (16 * LDB);         // this warp's [16, LDB]
    float* wstage = reinterpret_cast<float*>(wbuf);                   // same bytes, as 16x16 float

    int e_row[NE], e_col[NE];  // position of each accumulator element in its 16x16 tile
    wmma_acc_layout(wstage, lane, e_row, e_col);

    Acc dK[DPT], dV[DPT];
    #pragma unroll
    for (int t = 0; t < DPT; t++) {
        wmma::fill_fragment(dK[t], 0.0f);
        wmma::fill_fragment(dV[t], 0.0f);
    }

    load_tile_g2s<Bc, D, NUM_THREADS, LDH>(params.K + b_idx * params.k_batch_stride + kv_h_idx * params.k_head_stride
                                           + kv_start * params.k_seq_stride, params.k_seq_stride, smem_K, valid_kv);
    load_tile_g2s<Bc, D, NUM_THREADS, LDH>(params.V + b_idx * params.v_batch_stride + kv_h_idx * params.v_head_stride
                                           + kv_start * params.v_seq_stride, params.v_seq_stride, smem_V, valid_kv);

    const int num_q_blocks = cdiv(params.seq_len_q, Br);
    const int q_first_blk = bwd_first_q_block(params.is_causal, kv_start, Br, params.seq_len_q, params.seq_len_k);
    const half* K_band = smem_K + band * T * LDH;
    const half* V_band = smem_V + band * T * LDH;

    for (int qh = 0; qh < heads_per_kv; qh++) {
        const int h_idx = kv_h_idx * heads_per_kv + qh;
        for (int q_blk = q_first_blk; q_blk < num_q_blocks; q_blk++) {
            const int q_start = q_blk * Br;
            const int valid_q = min(Br, params.seq_len_q - q_start);
            load_tile_g2s<Br, D, NUM_THREADS, LDH>(params.Q + b_idx * params.q_batch_stride + h_idx * params.q_head_stride
                                                   + q_start * params.q_seq_stride, params.q_seq_stride, smem_Q, valid_q);
            load_tile_g2s<Br, D, NUM_THREADS, LDH>(params.dO + b_idx * params.o_batch_stride + h_idx * params.o_head_stride
                                                   + q_start * params.o_seq_stride, params.o_seq_stride, smem_dO, valid_q);
            const int lse_off = b_idx * params.l_batch_stride + h_idx * params.l_head_stride + q_start;
            for (int i = tid; i < Br; i += NUM_THREADS) {
                // Rows past the sequence: P = 0 through the mask, D never used
                smem_L[i] = (i < valid_q) ? params.L[lse_off + i] * kLog2e : 0.0f;
                smem_D[i] = (i < valid_q) ? params.D[lse_off + i] : 0.0f;
            }
            __syncthreads();

            // S^T and dP^T for the band: [16 keys, Br queries] as QT fragments
            Acc S[QT], dP[QT];
            #pragma unroll
            for (int qt = 0; qt < QT; qt++) {
                wmma::fill_fragment(S[qt], 0.0f);
                wmma::fill_fragment(dP[qt], 0.0f);
            }
            #pragma unroll
            for (int k = 0; k < D; k += T) {
                FragA ka, va;
                wmma::load_matrix_sync(ka, K_band + k, LDH);
                wmma::load_matrix_sync(va, V_band + k, LDH);
                #pragma unroll
                for (int qt = 0; qt < QT; qt++) {
                    FragBt qb, ob;  // Q^T, dO^T: [k, queries] = rows of Q / dO read column-major
                    wmma::load_matrix_sync(qb, smem_Q + qt * T * LDH + k, LDH);
                    wmma::load_matrix_sync(ob, smem_dO + qt * T * LDH + k, LDH);
                    wmma::mma_sync(S[qt], ka, qb, S[qt]);
                    wmma::mma_sync(dP[qt], va, ob, dP[qt]);
                }
            }

            // P^T (in S) and dS^T (in dP), element-wise on the fragments; P^T as
            // half into the warp buffer for dV += P^T dO
            #pragma unroll
            for (int qt = 0; qt < QT; qt++) {
                #pragma unroll
                for (int i = 0; i < NE; i++) {
                    const int key = kv_start + band * T + e_row[i];
                    const int q = qt * T + e_col[i];  // row within the query block
                    const bool masked = q >= valid_q || key >= params.seq_len_k ||
                                        (params.is_causal && key > q_start + q + causal_off);
                    const float p = masked ? 0.0f : exp2f(S[qt].x[i] * scale_log2 - smem_L[q]);
                    S[qt].x[i] = p;
                    dP[qt].x[i] = masked ? 0.0f : p * (dP[qt].x[i] - smem_D[q]) * params.softmax_scale;
                    wbuf[e_row[i] * LDB + q] = __float2half(p);
                }
            }
            __syncwarp();
            #pragma unroll
            for (int k = 0; k < Br; k += T) {
                FragA pa;
                wmma::load_matrix_sync(pa, wbuf + k, LDB);
                #pragma unroll
                for (int t = 0; t < DPT; t++) {
                    FragB ob;
                    wmma::load_matrix_sync(ob, smem_dO + k * LDH + part * DP + t * T, LDH);
                    wmma::mma_sync(dV[t], pa, ob, dV[t]);
                }
            }
            __syncwarp();  // P^T reads done before dS^T overwrites the buffer

            // dS^T: into the warp buffer (dK += dS^T Q) and, once per band, into the
            // shared dS^T for dQ
            #pragma unroll
            for (int qt = 0; qt < QT; qt++) {
                #pragma unroll
                for (int i = 0; i < NE; i++) {
                    const half ds = __float2half(dP[qt].x[i]);
                    wbuf[e_row[i] * LDB + qt * T + e_col[i]] = ds;
                    if (part == 0) smem_dSt[(band * T + e_row[i]) * LDB + qt * T + e_col[i]] = ds;
                }
            }
            __syncwarp();
            #pragma unroll
            for (int k = 0; k < Br; k += T) {
                FragA dsa;
                wmma::load_matrix_sync(dsa, wbuf + k, LDB);
                #pragma unroll
                for (int t = 0; t < DPT; t++) {
                    FragB qb;
                    wmma::load_matrix_sync(qb, smem_Q + k * LDH + part * DP + t * T, LDH);
                    wmma::mma_sync(dK[t], dsa, qb, dK[t]);
                }
            }
            __syncthreads();  // shared dS^T complete; warp buffers free for staging

            // dQ[Br, D] += dS K: output tiles spread over the warps, staged per warp
            // and added with coalesced atomics
            float* dQ_blk = params.dQ + b_idx * params.dq_batch_stride + h_idx * params.dq_head_stride
                                      + q_start * params.dq_seq_stride;
            constexpr int DQ_TILES = QT * (D / T);
            #pragma unroll
            for (int tile = warp_id; tile < DQ_TILES; tile += NUM_WARPS) {
                const int qr = tile / (D / T), dc = tile % (D / T);
                Acc acc;
                wmma::fill_fragment(acc, 0.0f);
                #pragma unroll
                for (int k = 0; k < Bc; k += T) {
                    FragAt dsa;  // dS [queries, keys] = shared dS^T read column-major
                    FragB kb;
                    wmma::load_matrix_sync(dsa, smem_dSt + k * LDB + qr * T, LDB);
                    wmma::load_matrix_sync(kb, smem_K + k * LDH + dc * T, LDH);
                    wmma::mma_sync(acc, dsa, kb, acc);
                }
                wmma::store_matrix_sync(wstage, acc, T, wmma::mem_row_major);
                __syncwarp();
                #pragma unroll
                for (int e = lane; e < T * T; e += 32) {
                    const int row = qr * T + e / T;
                    if (row < valid_q) ::atomicAdd(dQ_blk + row * params.dq_seq_stride + dc * T + e % T, wstage[e]);
                }
                __syncwarp();
            }
            // No barrier here: the next query block first writes Q, dO, L, D (not
            // read by the dQ step) and writes dS^T only after its first barrier.
        }
    }

    // dK, dV of the warp's band and slice, through the 16x16 float staging
    half* dK_out = params.dK + b_idx * params.dk_batch_stride + kv_h_idx * params.dk_head_stride
                             + (kv_start + band * T) * params.dk_seq_stride + part * DP;
    half* dV_out = params.dV + b_idx * params.dv_batch_stride + kv_h_idx * params.dv_head_stride
                             + (kv_start + band * T) * params.dv_seq_stride + part * DP;
    const int band_valid = valid_kv - band * T;  // may be <= 0 for the last block
    auto store = [&](Acc& frag, half* out, int stride) {
        wmma::store_matrix_sync(wstage, frag, T, wmma::mem_row_major);
        __syncwarp();
        #pragma unroll
        for (int e = lane; e < T * T; e += 32) {
            if (e / T < band_valid) out[(e / T) * stride + e % T] = __float2half(wstage[e]);
        }
        __syncwarp();
    };
    #pragma unroll
    for (int t = 0; t < DPT; t++) {
        store(dK[t], dK_out + t * T, params.dk_seq_stride);
        store(dV[t], dV_out + t * T, params.dv_seq_stride);
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

// Tiles per head_dim. CUDA-core: sizes measured in P5.2. WMMA: kSplitD keeps
// dK / dV in registers (d / kSplitD columns per warp); lists in order of
// preference, the first that fits the device's shared memory per block is used.
template <typename F>
static void dispatch_bwd_simt_tile(int head_dim, F&& f) {
    const int limit = max_dynamic_smem_per_block();
    bool ok = false;
    switch (head_dim) {
        case 32:  ok = with_first_fitting_tile<BwdTileSimt<32, 32, 32>>(limit, f); break;
        case 64:  ok = with_first_fitting_tile<BwdTileSimt<32, 32, 64>>(limit, f); break;
        case 96:  ok = with_first_fitting_tile<BwdTileSimt<32, 16, 96>>(limit, f); break;
        case 128: ok = with_first_fitting_tile<BwdTileSimt<16, 16, 128>>(limit, f); break;
        case 256: ok = with_first_fitting_tile<BwdTileSimt<16, 16, 256>>(limit, f); break;
        default:  throw std::runtime_error("Unsupported head_dim for the backward");
    }
    if (!ok) throw_no_tile("CUDA-core backward", head_dim, limit);
}

template <typename F>
static void dispatch_bwd_wmma_tile(int head_dim, F&& f) {
    const int limit = max_dynamic_smem_per_block();
    bool ok = false;
    switch (head_dim) {
        case 32:  ok = with_first_fitting_tile<BwdTileWmma<64, 64, 32>>(limit, f); break;
        case 64:  ok = with_first_fitting_tile<BwdTileWmma<64, 64, 64>>(limit, f); break;
        case 96:  ok = with_first_fitting_tile<BwdTileWmma<32, 64, 96, 2>, BwdTileWmma<32, 32, 96, 2>>(limit, f); break;
        case 128: ok = with_first_fitting_tile<BwdTileWmma<32, 64, 128, 2>, BwdTileWmma<32, 32, 128, 2>>(limit, f); break;
        case 256: ok = with_first_fitting_tile<BwdTileWmma<32, 32, 256, 4>, BwdTileWmma<32, 16, 256, 4>>(limit, f); break;
        default:  throw std::runtime_error("Unsupported head_dim for the backward");
    }
    if (!ok) throw_no_tile("WMMA backward", head_dim, limit);
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
    dispatch_bwd_wmma_tile(params.head_dim, [&](auto tile) {
        using T = decltype(tile);
        launch_bwd<T>(flash_attn_bwd_volta_kernel<T>, params, stream);
    });
}

void flash_attn_bwd_pascal_launch(FlashAttnBwdParams& params, cudaStream_t stream) {
    dispatch_bwd_simt_tile(params.head_dim, [&](auto tile) {
        using T = decltype(tile);
        launch_bwd<T>(flash_attn_bwd_kernel<T>, params, stream);
    });
}

}  // namespace flash_attn_legacy
