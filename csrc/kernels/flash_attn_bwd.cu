// ============================================================================
// Flash Attention v2 — Backward kernels
//   - BwdSimt: CUDA cores (Pascal; FORCE_ARCH=pascal elsewhere)
//   - BwdWmma: WMMA tensor cores (Volta / Turing)
//   - bwd_batched_kernel / bwd_varlen_kernel: locate the key block, run a Body
//   - precompute_D_kernel: D = rowsum(dO * O), and the gradient of the sinks
// Both backward kernels use the KV-outer loop of FlashAttention-2: one block
// per KV block (of one sequence, for varlen), dK/dV accumulated in registers for the whole block, dQ
// accumulated in an FP32 buffer with one atomicAdd per element and KV block.
// ============================================================================

#include "flash_attn_common.h"
#include "flash_attn_wmma.h"
#include "flash_attn_mma.h"


#include <string>

namespace flash_attn_legacy {

// ---------------------------------------------------------------------------
// Precompute D = rowsum(dO * O) — launched as a separate kernel
// D has the layout of L: [B, H, N] (batched) or [total, H] (varlen, B = 1).
// Each row is read by kDLanes consecutive lanes in 16-byte chunks (coalesced:
// a warp reads 4 rows; rows have d % 8 == 0 halves and 16-byte aligned starts).
// With sinks (one logit per head, probability p_sink = exp(sink - L) in every
// row): dsinks[h] += -p_sink * D over the rows of head h, as only the sink's
// share of the normalization depends on it (dO . dO/dsink = -p_sink * D).
// ---------------------------------------------------------------------------
constexpr int kDLanes = 8;  // lanes per row
constexpr int kDBlock = 256;  // threads per block: whole warps (full-mask shuffles)
static_assert(kDBlock % 32 == 0 && 32 % kDLanes == 0, "row groups must not straddle warps");

__global__ void precompute_D_kernel(
    const half* __restrict__ dO,   // [B, H, N, d]
    const half* __restrict__ O,    // [B, H, N, d]
    float* __restrict__ D,         // [B, H, N]
    const float* __restrict__ L,   // layout of D; read with sinks only
    const float* __restrict__ sinks,  // [H] or nullptr
    float* __restrict__ dsinks,       // [H] fp32, zeroed; with sinks only
    int batch_size, int num_heads, int seq_len, int head_dim,
    index_t o_batch_stride, index_t o_head_stride, index_t o_seq_stride,
    index_t l_batch_stride, index_t l_head_stride, index_t l_seq_stride
) {
    const int64_t row = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / kDLanes;
    const int sub = threadIdx.x % kDLanes;  // lane within the row
    const bool valid = row < int64_t(batch_size) * num_heads * seq_len;

    // n fastest: the rows of a warp are consecutive, mostly of one head
    const int n = valid ? static_cast<int>(row % seq_len) : 0;
    const int h = valid ? static_cast<int>((row / seq_len) % num_heads) : 0;
    const int b = valid ? static_cast<int>(row / (int64_t(num_heads) * seq_len)) : 0;
    const index_t l_idx = b * l_batch_stride + h * l_head_stride + n * l_seq_stride;

    float sum = 0.0f;
    if (valid) {
        const index_t off = b * o_batch_stride + h * o_head_stride + n * o_seq_stride;
        const uint4* dO4 = reinterpret_cast<const uint4*>(dO + off);
        const uint4* O4  = reinterpret_cast<const uint4*>(O + off);
        for (int c = sub; c < head_dim / 8; c += kDLanes) {
            const uint4 x = dO4[c], y = O4[c];
            const half2* x2 = reinterpret_cast<const half2*>(&x);
            const half2* y2 = reinterpret_cast<const half2*>(&y);
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                const float2 xf = __half22float2(x2[k]), yf = __half22float2(y2[k]);
                sum = fmaf(xf.x, yf.x, fmaf(xf.y, yf.y, sum));
            }
        }
    }
    #pragma unroll
    for (int o = kDLanes / 2; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    const bool leader = valid && sub == 0;  // one lane per row writes / contributes
    if (leader) D[l_idx] = sum;

    if (sinks == nullptr) return;  // uniform over the grid
    float g = 0.0f;
    if (leader && sinks[h] > -INFINITY) g = -expf(sinks[h] - L[l_idx]) * sum;  // -inf: no sink
    // One atomic per warp when its rows share a head, else one per row.
    const int h0 = __shfl_sync(0xffffffffu, h, 0);
    if (__all_sync(0xffffffffu, !valid || h == h0)) {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) g += __shfl_xor_sync(0xffffffffu, g, o);
        if ((threadIdx.x & 31) == 0 && valid) atomicAdd(&dsinks[h0], g);
    } else if (leader) {
        atomicAdd(&dsinks[h], g);
    }
}

// ---------------------------------------------------------------------------
// One key block of the backward, from the batched or the varlen grid: pointers
// at the first row of the sequence, for the first Q head of the KV head's group
// (Q, dO, L, D, dQ; add qh * *_head_stride for the others, GQA) and for the KV
// head (K, V, dK, dV); row strides; key block start and sequence lengths.
// ---------------------------------------------------------------------------
struct BwdBlockArgs {
    const half* __restrict__ Q; const half* __restrict__ dO;
    const float* __restrict__ L; const float* __restrict__ D; float* __restrict__ dQ;
    const half* __restrict__ K; const half* __restrict__ V;
    half* __restrict__ dK; half* __restrict__ dV;
    index_t q_head_stride, o_head_stride, l_head_stride, dq_head_stride;
    index_t q_stride, o_stride, l_stride, dq_stride, k_stride, v_stride, dk_stride, dv_stride;
    int heads_per_kv, kv_start, seq_len_q, seq_len_k;
    int head_dim;  // real head_dim (<= Tile::d: extra tile columns are zero, never written)
    float softmax_scale, softcap;
    int window_left, window_right;  // KeyWindow of the sequence

    __device__ __forceinline__ KeyWindow window() const {
        return KeyWindow(window_left, window_right, seq_len_q, seq_len_k);
    }
    __device__ __forceinline__ ScoreMap score_map() const { return ScoreMap(softmax_scale, softcap); }
    // Query blocks of Br rows [x, y) with some row that sees a key of the key
    // block [kv_start, kv_start + Bc) (first / last row grow with the key)
    template <int Br, int Bc>
    __device__ __forceinline__ int2 q_blocks() const {
        const KeyWindow w = window();
        const int first_row = w.first_row(kv_start);
        const int last_row = w.last_row(min(kv_start + Bc, seq_len_k) - 1);
        if (last_row < first_row) return make_int2(0, 0);
        return make_int2(first_row / Br, last_row / Br + 1);
    }
};

// b: batch (0 for varlen); q_tok0 / k_tok0: first row of the sequence (0 for batched).
__device__ __forceinline__ BwdBlockArgs bwd_block_args(const FlashAttnBwdParams& p, int b, int q_tok0,
                                                       int k_tok0, int seq_len_q, int seq_len_k,
                                                       int kv_h, int kv_start) {
    const int heads_per_kv = p.num_heads / p.num_heads_k;
    const int h0 = kv_h * heads_per_kv;
    BwdBlockArgs a;
    a.Q  = p.Q  + b * p.q_batch_stride + h0 * p.q_head_stride + q_tok0 * p.q_seq_stride;
    a.dO = p.dO + b * p.o_batch_stride + h0 * p.o_head_stride + q_tok0 * p.o_seq_stride;
    a.L  = p.L  + b * p.l_batch_stride + h0 * p.l_head_stride + q_tok0 * p.l_seq_stride;
    a.D  = p.D  + b * p.l_batch_stride + h0 * p.l_head_stride + q_tok0 * p.l_seq_stride;
    a.dQ = p.dQ + b * p.dq_batch_stride + h0 * p.dq_head_stride + q_tok0 * p.dq_seq_stride;
    a.K  = p.K  + b * p.k_batch_stride + kv_h * p.k_head_stride + k_tok0 * p.k_seq_stride;
    a.V  = p.V  + b * p.v_batch_stride + kv_h * p.v_head_stride + k_tok0 * p.v_seq_stride;
    a.dK = p.dK + b * p.dk_batch_stride + kv_h * p.dk_head_stride + k_tok0 * p.dk_seq_stride;
    a.dV = p.dV + b * p.dv_batch_stride + kv_h * p.dv_head_stride + k_tok0 * p.dv_seq_stride;
    a.q_head_stride = p.q_head_stride; a.o_head_stride = p.o_head_stride;
    a.l_head_stride = p.l_head_stride; a.dq_head_stride = p.dq_head_stride;
    a.q_stride = p.q_seq_stride; a.o_stride = p.o_seq_stride; a.l_stride = p.l_seq_stride;
    a.dq_stride = p.dq_seq_stride; a.k_stride = p.k_seq_stride; a.v_stride = p.v_seq_stride;
    a.dk_stride = p.dk_seq_stride; a.dv_stride = p.dv_seq_stride;
    a.heads_per_kv = heads_per_kv;
    a.kv_start = kv_start;
    a.seq_len_q = seq_len_q;
    a.seq_len_k = seq_len_k;
    a.head_dim = p.head_dim;  // kernels narrow it with tile_valid_cols
    a.softmax_scale = p.softmax_scale;
    a.softcap = p.softcap;
    a.window_left = p.window_left;
    a.window_right = p.window_right;
    return a;
}

// Flat grid: x = (batch * KV heads) * key_blocks + key_block (as the forward).
template <typename Tile, typename Body>
__global__ void __launch_bounds__(Tile::kNumWarps * 32)
bwd_batched_kernel(FlashAttnBwdParams p) {
    const int key_blocks = cdiv(p.seq_len_k, Tile::Bc);
    const int bh = blockIdx.x / key_blocks;
    const int b = bh / p.num_heads_k;
    const int kv_h = bh % p.num_heads_k;
    const int kv_start = (blockIdx.x % key_blocks) * Tile::Bc;
    if (kv_start >= p.seq_len_k) return;
    BwdBlockArgs a = bwd_block_args(p, b, 0, 0, p.seq_len_q, p.seq_len_k, kv_h, kv_start);
    a.head_dim = tile_valid_cols<Tile::d>(p.head_dim);
    Body::template run<Tile>(a);
}

// Grid (key blocks of all sequences, H_kv): the sequence comes from
// key_block_offsets (varlen_seq_of_block).
template <typename Tile, typename Body>
__global__ void __launch_bounds__(Tile::kNumWarps * 32)
bwd_varlen_kernel(FlashAttnBwdParams p) {
    const int seq = varlen_seq_of_block(p.key_block_offsets, p.num_seqs, blockIdx.x);
    if (seq >= p.num_seqs) return;
    const int q_tok0 = p.cu_seqlens_q[seq], k_tok0 = p.cu_seqlens_k[seq];
    const int seq_len_k = p.cu_seqlens_k[seq + 1] - k_tok0;
    const int kv_start = (blockIdx.x - p.key_block_offsets[seq]) * Tile::Bc;
    if (kv_start >= seq_len_k) return;
    BwdBlockArgs a = bwd_block_args(p, 0, q_tok0, k_tok0, p.cu_seqlens_q[seq + 1] - q_tok0,
                                    seq_len_k, blockIdx.y, kv_start);
    a.head_dim = tile_valid_cols<Tile::d>(p.head_dim);
    Body::template run<Tile>(a);
}

// ---------------------------------------------------------------------------
// Backward kernel — CUDA cores (Pascal), KV-outer loop, tiled like FA2.
//
// One block per (KV block of Bc rows, batch x KV head). For every Q head of
// the group (GQA) and every Q block of Br rows:
//   1. S, P = exp(x(S) - L), dP = dO @ V^T, dS = P * (dP - D) * dx/dS for
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
struct BwdSimt {
template <typename Tile>
static __device__ __forceinline__ void run(const BwdBlockArgs& a) {
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
    const int kv_start = a.kv_start;
    const int valid_kv = min(Bc, a.seq_len_k - kv_start);
    const KeyWindow w = a.window();
    const ScoreMap score_map = a.score_map();

    extern __shared__ __align__(128) char smem_raw[];  // aligned for 128-bit accesses
    half*  smem_K  = reinterpret_cast<half*>(smem_raw);
    half*  smem_V  = smem_K + Bc * LDH;
    half*  smem_Q  = smem_V + Bc * LDH;
    half*  smem_dO = smem_Q + Br * LDH;
    float* smem_P  = reinterpret_cast<float*>(smem_dO + Br * LDH);
    float* smem_dS = smem_P + Br * LDF;
    float* smem_L  = smem_dS + Br * LDF;  // L * log2(e)
    float* smem_D  = smem_L + Br;

    load_tile_g2s<Bc, D, NT, LDH>(a.K + kv_start * a.k_stride, a.k_stride, smem_K, valid_kv, a.head_dim);
    load_tile_g2s<Bc, D, NT, LDH>(a.V + kv_start * a.v_stride, a.v_stride, smem_V, valid_kv, a.head_dim);

    const int kv_j = tid % Bc, kv_c0 = (tid / Bc) * KV_COLS;
    const int q_i = tid % Br, q_c0 = (tid / Br) * Q_COLS;
    float acc_dK[KV_COLS], acc_dV[KV_COLS];
    #pragma unroll
    for (int c = 0; c < KV_COLS; c++) { acc_dK[c] = 0.0f; acc_dV[c] = 0.0f; }

    const int2 q_blks = a.template q_blocks<Br, Bc>();

    for (int qh = 0; qh < a.heads_per_kv; qh++) {
        for (int q_blk = q_blks.x; q_blk < q_blks.y; q_blk++) {
            const int q_start = q_blk * Br;
            const int valid_q = min(Br, a.seq_len_q - q_start);

            // The previous iteration's readers of Q/dO/P/dS passed the barrier
            // at the end of the loop body.
            load_tile_g2s<Br, D, NT, LDH>(a.Q + qh * a.q_head_stride + q_start * a.q_stride, a.q_stride,
                                          smem_Q, valid_q, a.head_dim);
            load_tile_g2s<Br, D, NT, LDH>(a.dO + qh * a.o_head_stride + q_start * a.o_stride, a.o_stride,
                                          smem_dO, valid_q, a.head_dim);
            if (tid < Br) {
                const index_t lrow = qh * a.l_head_stride + (q_start + tid) * a.l_stride;
                smem_L[tid] = (tid < valid_q) ? a.L[lrow] * kLog2e : 0.0f;
                smem_D[tid] = (tid < valid_q) ? a.D[lrow] : 0.0f;
            }
            __syncthreads();

            // 1. P and dS for the [Br, Bc] tile (j fastest across threads)
            score_map.dispatch([&](auto softcap) {
                #pragma unroll
                for (int e = tid; e < Br * Bc; e += NT) {
                    const int i = e / Bc, j = e % Bc;
                    float p = 0.0f, ds = 0.0f;
                    const bool valid = i < valid_q && !w.masked(q_start + i, kv_start + j);
                    if (valid) {
                        float dx_ds;
                        const float x = score_map.template log2_logit<decltype(softcap)::value>(
                            half_dot<D>(smem_Q + i * LDH, smem_K + j * LDH), dx_ds);
                        p = exp2f(x - smem_L[i]);
                        const float dp = half_dot<D>(smem_dO + i * LDH, smem_V + j * LDH);
                        ds = p * (dp - smem_D[i]) * dx_ds;
                    }
                    smem_P[i * LDF + j] = p;
                    smem_dS[i * LDF + j] = ds;
                }
            });
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
                float* dq_row = a.dQ + qh * a.dq_head_stride + (q_start + q_i) * a.dq_stride + q_c0;
                #pragma unroll
                for (int c = 0; c < Q_COLS; c++) {
                    if (q_c0 + c < a.head_dim) atomicAdd(&dq_row[c], acc_dQ[c]);
                }
            }
            __syncthreads();  // P/dS/Q/dO are overwritten by the next iteration
        }
    }

    // Write dK, dV (row j of this KV block, column slice)
    if (kv_j < valid_kv) {
        #pragma unroll
        for (int c = 0; c < KV_COLS / 2; c++) {
            const int col = kv_c0 + 2 * c;
            if (col < a.head_dim) {  // not a zero-padded column (d=72/80)
                *reinterpret_cast<half2*>(a.dK + (kv_start + kv_j) * a.dk_stride + col) =
                    __floats2half2_rn(acc_dK[2 * c], acc_dK[2 * c + 1]);
                *reinterpret_cast<half2*>(a.dV + (kv_start + kv_j) * a.dv_stride + col) =
                    __floats2half2_rn(acc_dV[2 * c], acc_dV[2 * c + 1]);
            }
        }
    }
}
};

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
//   S^T = K_band Q^T,  P^T = exp(x(S^T) - L)  (ScoreMap; masked to 0)
//   dV += P^T dO,  dP^T = V_band dO^T,  dS^T = P^T (dP^T - D) * dx/dS,  dK += dS^T Q
// with the element-wise steps done on the fragments themselves (the row and
// column of each element come from wmma_acc_layout). P^T / dS^T reach the
// next WMMA as half through the warp's buffer, and dS^T also goes to a shared
// [Bc, Br] buffer: after one block barrier all warps compute dQ = dS K for the
// query block and add it to the fp32 dQ buffer with coalesced atomics (one per
// element and key block). Two block barriers per query block.
// ---------------------------------------------------------------------------
struct BwdWmma {
template <typename Tile>
static __device__ __forceinline__ void run(const BwdBlockArgs& a) {
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

    const int kv_start = a.kv_start;
    const int valid_kv = min(Bc, a.seq_len_k - kv_start);
    const KeyWindow w = a.window();
    const ScoreMap score_map = a.score_map();

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

    load_tile_g2s<Bc, D, NUM_THREADS, LDH>(a.K + kv_start * a.k_stride, a.k_stride, smem_K, valid_kv, a.head_dim);
    load_tile_g2s<Bc, D, NUM_THREADS, LDH>(a.V + kv_start * a.v_stride, a.v_stride, smem_V, valid_kv, a.head_dim);

    const int2 q_blks = a.template q_blocks<Br, Bc>();
    const half* K_band = smem_K + band * T * LDH;
    const half* V_band = smem_V + band * T * LDH;

    for (int qh = 0; qh < a.heads_per_kv; qh++) {
        for (int q_blk = q_blks.x; q_blk < q_blks.y; q_blk++) {
            const int q_start = q_blk * Br;
            const int valid_q = min(Br, a.seq_len_q - q_start);
            load_tile_g2s<Br, D, NUM_THREADS, LDH>(a.Q + qh * a.q_head_stride + q_start * a.q_stride, a.q_stride,
                                                   smem_Q, valid_q, a.head_dim);
            load_tile_g2s<Br, D, NUM_THREADS, LDH>(a.dO + qh * a.o_head_stride + q_start * a.o_stride, a.o_stride,
                                                   smem_dO, valid_q, a.head_dim);
            for (int i = tid; i < Br; i += NUM_THREADS) {
                // Rows past the sequence: P = 0 through the mask, D never used
                const index_t lrow = qh * a.l_head_stride + (q_start + i) * a.l_stride;
                smem_L[i] = (i < valid_q) ? a.L[lrow] * kLog2e : 0.0f;
                smem_D[i] = (i < valid_q) ? a.D[lrow] : 0.0f;
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
            score_map.dispatch([&](auto softcap) {
                #pragma unroll
                for (int qt = 0; qt < QT; qt++) {
                    #pragma unroll
                    for (int i = 0; i < NE; i++) {
                        const int key = kv_start + band * T + e_row[i];
                        const int q = qt * T + e_col[i];  // row within the query block
                        // selects, not a branch: the map of masked elements is
                        // cheaper than diverging (V100 fwd+bwd -3%)
                        const bool masked = q >= valid_q || w.masked(q_start + q, key);
                        float dx_ds;
                        const float x = score_map.template log2_logit<decltype(softcap)::value>(S[qt].x[i], dx_ds);
                        const float p = masked ? 0.0f : exp2f(x - smem_L[q]);
                        S[qt].x[i] = p;
                        dP[qt].x[i] = masked ? 0.0f : p * (dP[qt].x[i] - smem_D[q]) * dx_ds;
                        wbuf[e_row[i] * LDB + q] = __float2half(p);
                    }
                }
            });
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
            float* dQ_blk = a.dQ + qh * a.dq_head_stride + q_start * a.dq_stride;
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
                    if (row < valid_q && dc * T + e % T < a.head_dim) {
                        ::atomicAdd(dQ_blk + row * a.dq_stride + dc * T + e % T, wstage[e]);
                    }
                }
                __syncwarp();
            }
            // No barrier here: the next query block first writes Q, dO, L, D (not
            // read by the dQ step) and writes dS^T only after its first barrier.
        }
    }

    // dK, dV of the warp's band and slice, through the 16x16 float staging
    half* dK_out = a.dK + (kv_start + band * T) * a.dk_stride + part * DP;
    half* dV_out = a.dV + (kv_start + band * T) * a.dv_stride + part * DP;
    const int band_valid = valid_kv - band * T;  // may be <= 0 for the last block
    auto store = [&](Acc& frag, half* out, index_t stride, int col0) {
        wmma::store_matrix_sync(wstage, frag, T, wmma::mem_row_major);
        __syncwarp();
        #pragma unroll
        for (int e = lane; e < T * T; e += 32) {
            if (e / T < band_valid && col0 + e % T < a.head_dim) {
                out[(e / T) * stride + e % T] = __float2half(wstage[e]);
            }
        }
        __syncwarp();
    };
    #pragma unroll
    for (int t = 0; t < DPT; t++) {
        store(dK[t], dK_out + t * T, a.dk_stride, part * DP + t * T);
        store(dV[t], dV_out + t * T, a.dv_stride, part * DP + t * T);
    }
}
};

#else  // __CUDA_ARCH__ < 700
// Stub body for SM < 7.0 device code — never launched there: the host selects the
// family from the arch of the code that runs (compiled_arch_for).
struct BwdWmma {
    template <typename Tile>
    static __device__ __forceinline__ void run(const BwdBlockArgs&) { __trap(); }
};
#endif  // __CUDA_ARCH__

// ============================================================================
// TURING mma.sync Backward — SM >= 7.5 (docs/analysis/kernels/bwd_design.md)
// ============================================================================
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 750

// ---------------------------------------------------------------------------
// Same block geometry as BwdWmma (key blocks of Bc, loop over the Q heads of the
// KV head and the query blocks of Br; warp w owns the 16 keys of band w / kSplitD
// and the DP = d / kSplitD columns of slice w % kSplitD of dK / dV), with m16n8k8
// MMAs whose register layout is known (flash_attn_mma.h). Per query block each
// warp computes, in registers,
//   S^T = K_band Q^T, dP^T = V_band dO^T  ([16 keys, Br]: Br / 8 accumulators each)
//   P^T = exp2(x(S^T) - L), dS^T = P^T (dP^T - D) dx/dS  (ScoreMap; masked to 0)
//   dV += P^T dO, dK += dS^T Q  (P^T / dS^T are the A operands as they are)
// and the slice-0 warp of each band writes dS^T to shared memory; after one block
// barrier the warps compute dQ = dS K for the query block in 16 x 16 output tiles
// and add it to the fp32 dQ buffer with atomics straight from the accumulators.
// Two block barriers per query block.
// ---------------------------------------------------------------------------
struct BwdMma {
template <typename Tile>
static __device__ __forceinline__ void run(const BwdBlockArgs& a) {
    constexpr int Br = Tile::Br;
    constexpr int Bc = Tile::Bc;
    constexpr int D  = Tile::d;
    constexpr int NUM_THREADS = Tile::kNumWarps * 32;
    constexpr int NUM_WARPS = Tile::kNumWarps;
    constexpr int SPLIT = Tile::kSplitD;
    constexpr int LDH = Tile::kLdH;
    constexpr int LDS = Tile::kLdS;
    constexpr int QN = Br / 8;     // query n-tiles of S^T / dP^T (k-steps of dV / dK)
    constexpr int DP = D / SPLIT;  // dK / dV columns of a warp
    constexpr int DN = DP / 8;     // dK / dV n-tiles of a warp

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    const int g = lane >> 2, t = lane & 3;  // accumulator rows g, g + 8; columns 2t, 2t + 1
    const int lr = lane & 7, lm = lane >> 3;  // ldmatrix: row within matrix lm
    const int band = warp_id / SPLIT;
    const int part = warp_id % SPLIT;

    const int kv_start = a.kv_start;
    const int valid_kv = min(Bc, a.seq_len_k - kv_start);
    const KeyWindow w = a.window();
    const ScoreMap score_map = a.score_map();

    extern __shared__ __align__(128) char smem_raw[];
    half*  smem_K  = reinterpret_cast<half*>(smem_raw);
    half*  smem_V  = smem_K + Bc * LDH;
    half*  smem_Q  = smem_V + Bc * LDH;
    half*  smem_dO = smem_Q + Br * LDH;
    float* smem_L  = reinterpret_cast<float*>(smem_dO + Br * LDH);  // L * log2(e)
    float* smem_D  = smem_L + Br;
    half*  smem_dSt = reinterpret_cast<half*>(smem_D + Br);           // dS^T [Bc, LDS]

    float dK[DN][4], dV[DN][4];
    #pragma unroll
    for (int n = 0; n < DN; n++) {
        #pragma unroll
        for (int e = 0; e < 4; e++) { dK[n][e] = 0.0f; dV[n][e] = 0.0f; }
    }

    load_tile_g2s<Bc, D, NUM_THREADS, LDH>(a.K + kv_start * a.k_stride, a.k_stride, smem_K, valid_kv, a.head_dim);
    load_tile_g2s<Bc, D, NUM_THREADS, LDH>(a.V + kv_start * a.v_stride, a.v_stride, smem_V, valid_kv, a.head_dim);

    const int2 q_blks = a.template q_blocks<Br, Bc>();
    // A operand rows of the band (ldmatrix x4: rows 0-7 / 8-15, k and k + 8)
    const half* K_rows = smem_K + (band * 16 + (lm & 1) * 8 + lr) * LDH + (lm >> 1) * 8;
    const half* V_rows = smem_V + (band * 16 + (lm & 1) * 8 + lr) * LDH + (lm >> 1) * 8;

    for (int qh = 0; qh < a.heads_per_kv; qh++) {
        for (int q_blk = q_blks.x; q_blk < q_blks.y; q_blk++) {
            const int q_start = q_blk * Br;
            const int valid_q = min(Br, a.seq_len_q - q_start);
            load_tile_g2s<Br, D, NUM_THREADS, LDH>(a.Q + qh * a.q_head_stride + q_start * a.q_stride, a.q_stride,
                                                   smem_Q, valid_q, a.head_dim);
            load_tile_g2s<Br, D, NUM_THREADS, LDH>(a.dO + qh * a.o_head_stride + q_start * a.o_stride, a.o_stride,
                                                   smem_dO, valid_q, a.head_dim);
            for (int i = tid; i < Br; i += NUM_THREADS) {
                // Rows past the sequence: P = 0 through the mask, D never used
                const index_t lrow = qh * a.l_head_stride + (q_start + i) * a.l_stride;
                smem_L[i] = (i < valid_q) ? a.L[lrow] * kLog2e : 0.0f;
                smem_D[i] = (i < valid_q) ? a.D[lrow] : 0.0f;
            }
            __syncthreads();

            // S^T and dP^T for the band: [16 keys, Br queries], n-tile j = queries 8j..8j+7
            float S[QN][4], dP[QN][4];
            #pragma unroll
            for (int j = 0; j < QN; j++) {
                #pragma unroll
                for (int e = 0; e < 4; e++) { S[j][e] = 0.0f; dP[j][e] = 0.0f; }
            }
            #pragma unroll
            for (int k = 0; k < D; k += 16) {
                uint32_t ka[4], va[4];  // {rows 0-7, 8-15} x {k, k + 8}
                ldmatrix_x4(ka, K_rows + k);
                ldmatrix_x4(va, V_rows + k);
                #pragma unroll
                for (int j = 0; j < QN; j += 2) {
                    // B = Q^T / dO^T from query rows: matrices (j, k), (j, k + 8), (j + 1, k), (j + 1, k + 8)
                    const int row = (j + (lm >> 1)) * 8 + lr, col = k + (lm & 1) * 8;
                    uint32_t qb[4], ob[4];
                    ldmatrix_x4(qb, smem_Q + row * LDH + col);
                    ldmatrix_x4(ob, smem_dO + row * LDH + col);
                    mma_m16n8k8(S[j], ka[0], ka[1], qb[0]);
                    mma_m16n8k8(S[j], ka[2], ka[3], qb[1]);
                    mma_m16n8k8(S[j + 1], ka[0], ka[1], qb[2]);
                    mma_m16n8k8(S[j + 1], ka[2], ka[3], qb[3]);
                    mma_m16n8k8(dP[j], va[0], va[1], ob[0]);
                    mma_m16n8k8(dP[j], va[2], va[3], ob[1]);
                    mma_m16n8k8(dP[j + 1], va[0], va[1], ob[2]);
                    mma_m16n8k8(dP[j + 1], va[2], va[3], ob[3]);
                }
            }

            // P^T (in S) and dS^T (in dP), element-wise: e = 0, 1 key g, e = 2, 3 key g + 8;
            // query 8j + 2t + (e & 1)
            score_map.dispatch([&](auto softcap) {
                #pragma unroll
                for (int j = 0; j < QN; j++) {
                    #pragma unroll
                    for (int e = 0; e < 4; e++) {
                        const int key = kv_start + band * 16 + g + (e >> 1) * 8;
                        const int q = j * 8 + 2 * t + (e & 1);  // row within the query block
                        // selects, not a branch (as BwdWmma)
                        const bool masked = q >= valid_q || w.masked(q_start + q, key);
                        float dx_ds;
                        const float x = score_map.template log2_logit<decltype(softcap)::value>(S[j][e], dx_ds);
                        const float p = masked ? 0.0f : exp2f(x - smem_L[q]);
                        S[j][e] = p;
                        dP[j][e] = masked ? 0.0f : p * (dP[j][e] - smem_D[q]) * dx_ds;
                    }
                }
            });

            // dV += P^T dO and dK += dS^T Q over this warp's columns: A from the
            // accumulators (k-step j = queries 8j..8j+7), B = dO / Q [query, column]
            // with ldmatrix.trans: matrices (j, c), (j, c + 8), (j + 1, c), (j + 1, c + 8)
            #pragma unroll
            for (int j = 0; j < QN; j += 2) {
                const uint32_t p0a = pack_half2(S[j][0], S[j][1]), p0b = pack_half2(S[j][2], S[j][3]);
                const uint32_t p1a = pack_half2(S[j + 1][0], S[j + 1][1]), p1b = pack_half2(S[j + 1][2], S[j + 1][3]);
                const uint32_t s0a = pack_half2(dP[j][0], dP[j][1]), s0b = pack_half2(dP[j][2], dP[j][3]);
                const uint32_t s1a = pack_half2(dP[j + 1][0], dP[j + 1][1]), s1b = pack_half2(dP[j + 1][2], dP[j + 1][3]);
                const int row = (j + (lm >> 1)) * 8 + lr;
                #pragma unroll
                for (int n = 0; n < DN; n += 2) {
                    const int col = part * DP + n * 8 + (lm & 1) * 8;
                    uint32_t ob[4], qb[4];
                    ldmatrix_x4_trans(ob, smem_dO + row * LDH + col);
                    ldmatrix_x4_trans(qb, smem_Q + row * LDH + col);
                    mma_m16n8k8(dV[n], p0a, p0b, ob[0]);
                    mma_m16n8k8(dV[n + 1], p0a, p0b, ob[1]);
                    mma_m16n8k8(dV[n], p1a, p1b, ob[2]);
                    mma_m16n8k8(dV[n + 1], p1a, p1b, ob[3]);
                    mma_m16n8k8(dK[n], s0a, s0b, qb[0]);
                    mma_m16n8k8(dK[n + 1], s0a, s0b, qb[1]);
                    mma_m16n8k8(dK[n], s1a, s1b, qb[2]);
                    mma_m16n8k8(dK[n + 1], s1a, s1b, qb[3]);
                }
                // dS^T to shared memory, key-major (one warp per band): half2 of queries 2t, 2t + 1
                if (part == 0) {
                    #pragma unroll
                    for (int h = 0; h < 2; h++) {
                        const int qcol = (j + h) * 8 + 2 * t;
                        const uint32_t lo = h ? s1a : s0a, hi = h ? s1b : s0b;
                        *reinterpret_cast<uint32_t*>(smem_dSt + (band * 16 + g) * LDS + qcol) = lo;
                        *reinterpret_cast<uint32_t*>(smem_dSt + (band * 16 + g + 8) * LDS + qcol) = hi;
                    }
                }
            }
            __syncthreads();  // dS^T complete

            // dQ[Br, D] += dS K: units of 16 queries x 16 columns over the warps.
            // A = dS [query, key] from the key-major dS^T (ldmatrix.trans: matrices
            // (keys k, queries q0), (k, q0 + 8), (k + 8, q0), (k + 8, q0 + 8));
            // B = K [key, column] (ldmatrix.trans: (k, c), (k + 8, c), (k, c + 8), (k + 8, c + 8))
            float* dQ_blk = a.dQ + qh * a.dq_head_stride + q_start * a.dq_stride;
            constexpr int UNITS = (Br / 16) * (D / 16);
            #pragma unroll
            for (int u = warp_id; u < UNITS; u += NUM_WARPS) {
                const int q0 = (u / (D / 16)) * 16, c0 = (u % (D / 16)) * 16;
                float acc[2][4] = {};
                #pragma unroll
                for (int k = 0; k < Bc; k += 16) {
                    uint32_t da[4], kb[4];
                    ldmatrix_x4_trans(da, smem_dSt + (k + (lm >> 1) * 8 + lr) * LDS + q0 + (lm & 1) * 8);
                    ldmatrix_x4_trans(kb, smem_K + (k + (lm & 1) * 8 + lr) * LDH + c0 + (lm >> 1) * 8);
                    mma_m16n8k8(acc[0], da[0], da[1], kb[0]);
                    mma_m16n8k8(acc[0], da[2], da[3], kb[1]);
                    mma_m16n8k8(acc[1], da[0], da[1], kb[2]);
                    mma_m16n8k8(acc[1], da[2], da[3], kb[3]);
                }
                #pragma unroll
                for (int n = 0; n < 2; n++) {
                    #pragma unroll
                    for (int e = 0; e < 4; e++) {
                        const int row = q0 + g + (e >> 1) * 8, col = c0 + n * 8 + 2 * t + (e & 1);
                        if (row < valid_q && col < a.head_dim) ::atomicAdd(dQ_blk + row * a.dq_stride + col, acc[n][e]);
                    }
                }
            }
            // No barrier here: the next query block first writes Q, dO, L, D (not
            // read by the dQ step) and writes dS^T only after its first barrier.
        }
    }

    // dK, dV of the warp's band and slice: half2 at their accumulator positions
    #pragma unroll
    for (int n = 0; n < DN; n++) {
        const int col = part * DP + n * 8 + 2 * t;
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            const int key = band * 16 + g + h * 8;  // within the block
            if (key < valid_kv && col < a.head_dim) {
                *reinterpret_cast<half2*>(a.dK + (kv_start + key) * a.dk_stride + col) =
                    __floats2half2_rn(dK[n][2 * h], dK[n][2 * h + 1]);
                *reinterpret_cast<half2*>(a.dV + (kv_start + key) * a.dv_stride + col) =
                    __floats2half2_rn(dV[n][2 * h], dV[n][2 * h + 1]);
            }
        }
    }
}
};

#else  // __CUDA_ARCH__ < 750
// Stub body for older device code — never launched there: the host selects the
// family from the arch of the code that runs (compiled_arch_for).
struct BwdMma {
    template <typename Tile>
    static __device__ __forceinline__ void run(const BwdBlockArgs&) { __trap(); }
};
#endif  // __CUDA_ARCH__

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------

void precompute_D(
    const half* dO, const half* O, float* D,
    const float* L, const float* sinks, float* dsinks,
    int B, int H, int N, int d,
    index_t o_batch_stride, index_t o_head_stride, index_t o_seq_stride,
    index_t l_batch_stride, index_t l_head_stride, index_t l_seq_stride,
    cudaStream_t stream
) {
    const int64_t total = int64_t(B) * H * N * kDLanes;  // threads
    const int block_size = kDBlock;
    const int64_t grid_size = (total + block_size - 1) / block_size;
    if (grid_size > INT32_MAX) throw std::runtime_error("flash_attn_legacy: too many rows for precompute_D");
    precompute_D_kernel<<<grid_size, block_size, 0, stream>>>(
        dO, O, D, L, sinks, dsinks, B, H, N, d,
        o_batch_stride, o_head_stride, o_seq_stride,
        l_batch_stride, l_head_stride, l_seq_stride
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
        case 72:
        case 80:  // zero-padded to the d=96 tile (CUDA-core tiles need d % 32 == 0)
        case 96:  ok = with_first_fitting_tile<BwdTileSimt<32, 16, 96>>(limit, f); break;
        case 128: ok = with_first_fitting_tile<BwdTileSimt<16, 16, 128>>(limit, f); break;  // 8 warps: same
        // d=256: 36 KB, one block per SM on P100: 8 warps instead of 4. fwd+bwd causal
        // N=4096: P100 517 -> 398 ms (16 warps 439; SDPA mem-efficient 443), 1080 Ti 581 -> 566
        case 256: ok = with_first_fitting_tile<BwdTileSimt<16, 16, 256, 8>>(limit, f); break;
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
        case 72:
        case 80:  ok = with_first_fitting_tile<BwdTileWmma<32, 64, 80>>(limit, f); break;  // 72 zero-padded;
                  // fwd+bwd N=4096 d=72/80: V100 11.9 ms (SDPA 16.0), T4 26.2 (37.5)
        case 96:  ok = with_first_fitting_tile<BwdTileWmma<32, 64, 96, 2>, BwdTileWmma<32, 32, 96, 2>>(limit, f); break;
        case 128: ok = with_first_fitting_tile<BwdTileWmma<32, 64, 128, 2>, BwdTileWmma<32, 32, 128, 2>>(limit, f); break;
        case 256: ok = with_first_fitting_tile<BwdTileWmma<32, 32, 256, 4>, BwdTileWmma<32, 16, 256, 4>>(limit, f); break;
        default:  throw std::runtime_error("Unsupported head_dim for the backward");
    }
    if (!ok) throw_no_tile("WMMA backward", head_dim, limit);
}

template <typename F>
static void dispatch_bwd_mma_tile(int head_dim, F&& f) {
    const int limit = max_dynamic_smem_per_block();
    bool ok = false;
    switch (head_dim) {
        case 32:  ok = with_first_fitting_tile<BwdTileMma<64, 64, 32>>(limit, f); break;
        case 64:  ok = with_first_fitting_tile<BwdTileMma<64, 64, 64>>(limit, f); break;
        case 72:
        case 80:  ok = with_first_fitting_tile<BwdTileMma<32, 64, 80>>(limit, f); break;  // 72 zero-padded
        // one tile each: they fit the 64 KB of every sm >= 7.5
        case 96:  ok = with_first_fitting_tile<BwdTileMma<32, 64, 96, 2>>(limit, f); break;
        case 128: ok = with_first_fitting_tile<BwdTileMma<32, 64, 128, 2>>(limit, f); break;
        case 256: ok = with_first_fitting_tile<BwdTileMma<32, 32, 256, 4>, BwdTileMma<32, 16, 256, 4>>(limit, f); break;
        default:  throw std::runtime_error("Unsupported head_dim for the backward");
    }
    if (!ok) throw_no_tile("mma backward", head_dim, limit);
}

// Backward of one kernel family (Body) with its tile: batched grid, or varlen
// grid over total_k_blocks key blocks when params.cu_seqlens_q is set.
template <typename Tile, typename Body>
static void launch_bwd(FlashAttnBwdParams& params, int total_k_blocks, cudaStream_t stream) {
    constexpr int smem = Tile::smem_bytes();
    const bool varlen = params.cu_seqlens_q != nullptr;
    auto kernel = varlen ? bwd_varlen_kernel<Tile, Body> : bwd_batched_kernel<Tile, Body>;
    const int64_t batched_blocks = int64_t(cdiv(params.seq_len_k, Tile::Bc)) * params.batch_size * params.num_heads_k;
    if (!varlen && batched_blocks > INT32_MAX) {
        throw std::runtime_error("flash_attn_legacy: batch * heads * key blocks exceeds the grid limit");
    }
    const dim3 grid = varlen ? dim3(total_k_blocks, params.num_heads_k) : dim3(static_cast<unsigned>(batched_blocks));
    FLASH_ATTN_CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    kernel<<<grid, Tile::kNumWarps * 32, smem, stream>>>(params);
}

// Kernel family by (possibly forced) SM: mma.sync (sm >= 75), WMMA (sm 7.0, or
// FLASH_ATTN_LEGACY_FORCE_ARCH=volta), CUDA cores (sm 6.x or FORCE_ARCH=pascal).
template <typename F>
static void dispatch_bwd(int head_dim, int sm, F&& f) {
    if (sm >= 75)      dispatch_bwd_mma_tile(head_dim, [&](auto tile) { f(tile, BwdMma{}); });
    else if (sm >= 70) dispatch_bwd_wmma_tile(head_dim, [&](auto tile) { f(tile, BwdWmma{}); });
    else               dispatch_bwd_simt_tile(head_dim, [&](auto tile) { f(tile, BwdSimt{}); });
}

void flash_attn_bwd_launch(FlashAttnBwdParams& params, int sm, int total_k_blocks, cudaStream_t stream) {
    dispatch_bwd(params.head_dim, sm, [&](auto tile, auto body) {
        launch_bwd<decltype(tile), decltype(body)>(params, total_k_blocks, stream);
    });
}

// Keys per block of the backward tile selected for head_dim on this device.
int flash_attn_bwd_block_cols(int head_dim, int sm) {
    int cols = 0;
    dispatch_bwd(head_dim, sm, [&](auto tile, auto) { cols = decltype(tile)::Bc; });
    return cols;
}

}  // namespace flash_attn_legacy
