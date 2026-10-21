#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>

// ============================================================================
// Flash Attention Legacy — Common Types & Utilities
// Targets: Pascal (SM 6.x) and Volta (SM 7.0)
// ============================================================================

namespace flash_attn_legacy {

// ---------------------------------------------------------------------------
// Compile-time architecture detection
// ---------------------------------------------------------------------------
enum class Arch { Pascal, Volta, Unknown };

// ---------------------------------------------------------------------------
// Forward kernel launch parameters
// ---------------------------------------------------------------------------
struct FlashAttnParams {
    // Pointers (device)
    const half* __restrict__ Q;   // [B, H_q, N, d]
    const half* __restrict__ K;   // [B, H_kv, N, d]  (H_kv <= H_q for GQA/MQA)
    const half* __restrict__ V;   // [B, H_kv, N, d]
    half* __restrict__ O;         // [B, H_q, N, d]
    float* __restrict__ L;        // [B, H_q, N]  — logsumexp for backward

    // Dimensions
    int batch_size;
    int num_heads;       // H_q — query heads
    int num_heads_k;     // H_kv — key/value heads (== num_heads for MHA, 1 for MQA)
    int seq_len_q;       // N_q
    int seq_len_k;       // N_kv (may differ from N_q; causal is bottom-right aligned)
    int head_dim;        // 32, 64, 96, 128, or 256

    // Strides (in elements) for Q
    int q_batch_stride;
    int q_head_stride;
    int q_seq_stride;    // == head_dim for contiguous

    // Strides for K
    int k_batch_stride;
    int k_head_stride;
    int k_seq_stride;

    // Strides for V
    int v_batch_stride;
    int v_head_stride;
    int v_seq_stride;

    // Strides for O
    int o_batch_stride;
    int o_head_stride;
    int o_seq_stride;

    // Strides for L (logsumexp)
    int l_batch_stride;
    int l_head_stride;

    // Softmax scale  (typically 1/sqrt(d))
    float softmax_scale;

    // Causal mask
    bool is_causal;
};

// ---------------------------------------------------------------------------
// Forward kernel launch parameters — variable-length (packed sequences)
// ---------------------------------------------------------------------------
struct FlashAttnVarlenParams {
    // Pointers (device) — packed format: [total_tokens, H, d]
    const half* __restrict__ Q;   // [total_q, H_q, d]
    const half* __restrict__ K;   // [total_k, H_kv, d]
    const half* __restrict__ V;   // [total_k, H_kv, d]
    half* __restrict__ O;         // [total_q, H_q, d]
    float* __restrict__ L;        // [total_q, H_q] — logsumexp for backward

    // Sequence boundaries
    const int* __restrict__ cu_seqlens_q;   // [num_seqs + 1]
    const int* __restrict__ cu_seqlens_k;   // [num_seqs + 1]
    const int* __restrict__ block_offsets;   // [num_seqs + 1] — cumulative Q block counts

    int num_seqs;
    int num_heads;       // H_q
    int num_heads_k;     // H_kv
    int head_dim;        // 32, 64, 96, 128, or 256

    // Strides (in elements) — layout is [total, H, d]
    int q_token_stride;  // H_q * d
    int q_head_stride;   // d
    int k_token_stride;  // H_kv * d
    int k_head_stride;   // d
    int v_token_stride;
    int v_head_stride;
    int o_token_stride;
    int o_head_stride;

    float softmax_scale;
    bool is_causal;
};

// ---------------------------------------------------------------------------
// Backward kernel launch parameters — shared between all kernels
// ---------------------------------------------------------------------------
struct FlashAttnBwdParams {
    // Forward inputs
    const half* __restrict__ Q;
    const half* __restrict__ K;
    const half* __restrict__ V;
    const half* __restrict__ O;
    const float* __restrict__ L;     // logsumexp from forward
    const float* __restrict__ D;     // rowsum(dO * O), precomputed [B, H, N]

    // Gradient of output
    const half* __restrict__ dO;

    // Gradient outputs
    float* __restrict__ dQ;    // FP32 accumulator
    half* __restrict__ dK;
    half* __restrict__ dV;

    // Dimensions
    int batch_size, num_heads, num_heads_k, seq_len_q, seq_len_k, head_dim;
    float softmax_scale;
    bool is_causal;

    // Strides for Q/dQ
    int q_batch_stride, q_head_stride, q_seq_stride;
    // Strides for K/dK
    int k_batch_stride, k_head_stride, k_seq_stride;
    // Strides for V/dV
    int v_batch_stride, v_head_stride, v_seq_stride;
    // Strides for O/dO
    int o_batch_stride, o_head_stride, o_seq_stride;
    // Strides for L and D
    int l_batch_stride, l_head_stride;
    // Strides for dQ (FP32, same layout as Q but float)
    int dq_batch_stride, dq_head_stride, dq_seq_stride;
    // Strides for dK (same layout as K)
    int dk_batch_stride, dk_head_stride, dk_seq_stride;
    // Strides for dV (same layout as V)
    int dv_batch_stride, dv_head_stride, dv_seq_stride;
};

// ---------------------------------------------------------------------------
// Forward tile sizes
//
// One template per kernel family; smem_bytes() is the single source of truth
// for the dynamic shared memory of a tile (used by the launches and by the
// runtime tile selection). Threads per block: 32 * kNumWarps of each tile.
// ---------------------------------------------------------------------------

// WMMA forward (Volta SM 7.0, Turing SM 7.5), FA2-style: each warp owns a band
// of 16 query rows, O stays in WMMA accumulator fragments (registers).
// Shared memory: Q[Br,kLdH], K[Bc,kLdH], V[Bc,kLdH] (half, rows padded by 8
// halves: valid WMMA leading dimension, conflict-free fragment loads), then per
// warp an S/P staging tile [16,kLdS] float (P is written in place as half, row
// stride 2*kLdS) and 16 per-row scales. kLdS = Bc+8 makes the softmax reads (two
// lanes per row, interleaved float4 chunks) bank-conflict free.
template <int Br_, int Bc_, int d_>
struct FwdTileWmma {
    static constexpr int Br = Br_;
    static constexpr int Bc = Bc_;
    static constexpr int d  = d_;
    static constexpr int kNumWarps = Br_ / 16;
    static constexpr int kLdH = d_ + 8;
    static constexpr int kLdS = Bc_ + 8;
    static constexpr int kStageFloats = 16 * kLdS;  // per warp
    static constexpr int smem_bytes() {
        return (Br + 2 * Bc) * kLdH * 2 + kNumWarps * (kStageFloats + 16) * 4;
    }
    static_assert(Br_ % 16 == 0 && Bc_ % 16 == 0 && d_ % 16 == 0, "invalid WMMA forward tile");
    // The staging tile also holds the one-time accumulator layout probe
    // (256 floats + a 256-byte table).
    static_assert(kStageFloats >= 256 + 64, "WMMA forward staging too small");
};

// CUDA-core forward (Pascal SM 6.x): Q[Br,kLdQ] + K[Bc,d] + V[Bc,d] (half); P
// stays in registers. Q rows are padded by 8 halves so that the per-row 128-bit
// reads are bank-conflict free.
template <int Br_, int Bc_, int d_>
struct FwdTileSimt {
    static constexpr int Br = Br_;
    static constexpr int Bc = Bc_;
    static constexpr int d  = d_;
    static constexpr int kNumWarps = Br_ / 32;
    static constexpr int kLdQ = d_ + 8;
    static constexpr int smem_bytes() {
        return (Br * kLdQ + 2 * Bc * d) * 2;
    }
    static_assert(Br_ % 32 == 0 && d_ % 8 == 0, "invalid CUDA-core forward tile");
    static_assert(kLdQ % 8 == 0, "128-bit row reads need 16-byte aligned Q rows");
};

// WMMA forward tiles, in order of preference (the first that fits the device's
// shared memory per block is used: V100 96 KB, T4 64 KB). Measured, forward
// causal B=2 H=16 N=4096 (P2.8): 8 warps (128 rows) beat 4 warps everywhere
// (more warps share each K/V tile load); V100 / T4 in ms vs <64,Bc,d>:
// d=32 2.2 / 4.1 vs 3.1 / 7.9; d=64 3.0 / 6.8 vs 5.6 / 14.2; d=96 4.6 / 10.0
// vs 8.4 / 20.7; d=128 8.2 (Bc=32) / 12.5 (Bc=16) vs 11.6 / 27.9; d=256 15.2
// (V100) vs 35.9; on T4 d=256 stays at 4 warps (55 ms; 8 warps do not fit).
using TileVolta_d32  = FwdTileWmma<128, 32, 32>;    // 36 KB
using TileVolta_d64  = FwdTileWmma<128, 32, 64>;    // 48 KB
using TileVolta_d96  = FwdTileWmma<128, 16, 96>;    // 45 KB
using TileVolta_d128 = FwdTileWmma<128, 32, 128>;   // 72 KB (V100)
using TileVolta_d128_64k = FwdTileWmma<128, 16, 128>;  // 55 KB
using TileVolta_d256 = FwdTileWmma<128, 16, 256>;   // 95 KB (V100)
using TileVolta_d256_64k = FwdTileWmma<64, 16, 256>;   // 56 KB

using TilePascal_d32  = FwdTileSimt<128, 32, 32>;   // 14 KB
using TilePascal_d64  = FwdTileSimt<128, 32, 64>;   // 26 KB
using TilePascal_d96  = FwdTileSimt<64, 16, 96>;    // 19 KB
using TilePascal_d128 = FwdTileSimt<64, 16, 128>;   // 25 KB
using TilePascal_d256 = FwdTileSimt<32, 16, 256>;   // 33 KB

// First tile of the list whose smem_bytes() fits `smem_limit`: calls f(Tile{})
// and returns true, or returns false if none fits. Shared by every launcher and
// by the host code that needs the selected Br (varlen block offsets), so the
// selection can never disagree between them. Today every list has one tile
// (it fits all supported GPUs); the lists exist for per-device candidates
// (e.g. the forward redesign, docs/PLAN.md P2.8).
template <typename First, typename... Rest, typename F>
inline bool with_first_fitting_tile(int smem_limit, F& f) {
    if (First::smem_bytes() <= smem_limit) {
        f(First{});
        return true;
    }
    if constexpr (sizeof...(Rest) > 0) {
        return with_first_fitting_tile<Rest...>(smem_limit, f);
    } else {
        return false;
    }
}

// ---------------------------------------------------------------------------
// Backward tile sizes — same template scheme as the forward.
// WMMA: Br, Bc, d multiples of 16; d % Bc == 0 (dQ in chunks of Bc).
// CUDA-core: threads % Br == 0 and % Bc == 0, with an even number of columns
// per thread for dK/dV and dQ (static_asserts in the kernel).
// ---------------------------------------------------------------------------

// WMMA backward (Volta / Turing): K, V [Bc,d], Q, dO [Br,d] (half); S/P and
// dP/dS [Br,Bc] (float); P/dS [Br,Bc] (half); one 16x16 float staging tile per
// warp for writing back dK/dV, which are accumulated in registers (WMMA
// accumulator fragments) for the whole block, not in shared memory.
template <int Br_, int Bc_, int d_, int kNumWarps_ = 4>
struct BwdTileWmma {
    static constexpr int Br = Br_;
    static constexpr int Bc = Bc_;
    static constexpr int d  = d_;
    static constexpr int kNumWarps = kNumWarps_;
    static constexpr int smem_bytes() {
        return (2 * Bc * d + 2 * Br * d) * 2 + 2 * Br * Bc * 4 + Br * Bc * 2 + kNumWarps * 256 * 4;
    }
    static_assert(Br_ % 16 == 0 && Bc_ % 16 == 0 && d_ % Bc_ == 0, "invalid WMMA backward tile");
};

// CUDA-core backward (Pascal): K, V [Bc,kLdH], Q, dO [Br,kLdH] (half, rows padded
// for conflict-free 128-bit reads); P, dS [Br,kLdF] (float, odd stride: column
// reads in the dK/dV and dQ steps are conflict free); L, D [Br] (float). dK/dV
// in registers.
template <int Br_, int Bc_, int d_, int kNumWarps_ = 4>
struct BwdTileSimt {
    static constexpr int Br = Br_;
    static constexpr int Bc = Bc_;
    static constexpr int d  = d_;
    static constexpr int kNumWarps = kNumWarps_;
    static constexpr int kLdH = d_ + 8;
    static constexpr int kLdF = Bc_ + 1;
    static constexpr int smem_bytes() {
        return (2 * Bc * kLdH + 2 * Br * kLdH) * 2 + 2 * Br * kLdF * 4 + 2 * Br * 4;
    }
    static_assert(d_ % 8 == 0 && kLdH % 8 == 0, "128-bit row reads need 16-byte aligned rows");
    static_assert((kNumWarps_ * 32) % Br_ == 0 && (kNumWarps_ * 32) % Bc_ == 0,
                  "invalid CUDA-core backward tile");
};

// Backward tile of either family with the same sizes (the sizes per head_dim
// are listed once, in the backward dispatcher).
template <bool kWmma, int Br, int Bc, int d>
using BwdTile = std::conditional_t<kWmma, BwdTileWmma<Br, Bc, d>, BwdTileSimt<Br, Bc, d>>;

// ---------------------------------------------------------------------------
// Error checking macro
// ---------------------------------------------------------------------------
#define FLASH_ATTN_CHECK_CUDA(call)                                          \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                  \
            throw std::runtime_error(cudaGetErrorString(err));                \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// Maximum dynamic shared memory per block of the current device (opt-in limit:
// 48 KB Pascal, 96 KB Volta, 64 KB Turing). Queried per launch: a host-side
// attribute read, no device sync, and no shared cache to race on.
// FLASH_ATTN_LEGACY_SMEM_LIMIT=<bytes> lowers it (testing only: exercises the
// selection and its "no tile fits" error on any GPU).
// ---------------------------------------------------------------------------
inline int max_dynamic_smem_per_block() {
    int dev = 0;
    FLASH_ATTN_CHECK_CUDA(cudaGetDevice(&dev));
    int limit = 0;
    FLASH_ATTN_CHECK_CUDA(cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    if (const char* env = std::getenv("FLASH_ATTN_LEGACY_SMEM_LIMIT")) {
        const int forced = std::atoi(env);
        if (forced > 0 && forced < limit) limit = forced;
    }
    return limit;
}

// The single error for "no candidate tile fits this device" (all dispatchers).
[[noreturn]] inline void throw_no_tile(const char* kernel, int head_dim, int smem_limit) {
    throw std::runtime_error(std::string("flash_attn_legacy: no ") + kernel + " tile for head_dim " +
                             std::to_string(head_dim) + " fits " + std::to_string(smem_limit) +
                             " bytes of shared memory per block");
}

// ---------------------------------------------------------------------------
// Utility: ceiling division
// ---------------------------------------------------------------------------
__host__ __device__ __forceinline__ int cdiv(int a, int b) {
    return (a + b - 1) / b;
}

// ---------------------------------------------------------------------------
// Device helpers shared by every kernel (one implementation each).
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
// ---------------------------------------------------------------------------
// Vectorized global → shared memory load using half2 (4-byte transactions).
// Destination rows have stride DST_STRIDE (>= COLS, padded to avoid bank
// conflicts on per-row accesses); rows >= num_valid_rows are zero-filled.
// ---------------------------------------------------------------------------
template <int ROWS, int COLS, int NUM_THREADS, int DST_STRIDE = COLS>
__device__ __forceinline__ void load_tile_g2s(
    const half* __restrict__ src,
    int src_row_stride,
    half* __restrict__ dst,
    int num_valid_rows
) {
    static_assert(COLS % 2 == 0 && DST_STRIDE % 2 == 0, "half2 loads need even sizes");
    constexpr int COLS2 = COLS / 2;
    constexpr int TOTAL2 = ROWS * COLS2;
    constexpr int PER_THREAD = (TOTAL2 + NUM_THREADS - 1) / NUM_THREADS;

    const int tid = threadIdx.x;
    const half2 zero2 = {__float2half(0.0f), __float2half(0.0f)};

    #pragma unroll
    for (int i = 0; i < PER_THREAD; i++) {
        int idx = tid + i * NUM_THREADS;
        if (idx < TOTAL2) {
            int row = idx / COLS2;
            int col2 = idx % COLS2;
            half2* dst2 = reinterpret_cast<half2*>(dst + row * DST_STRIDE);
            if (row < num_valid_rows) {
                const half2* src2 = reinterpret_cast<const half2*>(src + row * src_row_stride);
                dst2[col2] = src2[col2];
            } else {
                dst2[col2] = zero2;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// fp16 dot product with fp32 products and accumulation, 128-bit loads.
// The products are NOT formed in fp16 (__hmul2): a single |q_i * k_i| > 65504
// would overflow to inf even with in-range inputs, and on sm_61 fp16 math runs
// at 1/64 of the fp32 rate.
// ---------------------------------------------------------------------------
template <int LEN>
__device__ __forceinline__ float half_dot(
    const half* __restrict__ a,
    const half* __restrict__ b
) {
    static_assert(LEN % 8 == 0, "LEN must be a multiple of 8 (128-bit loads)");
    const uint4* a4 = reinterpret_cast<const uint4*>(a);
    const uint4* b4 = reinterpret_cast<const uint4*>(b);
    float acc = 0.0f;
    #pragma unroll
    for (int i = 0; i < LEN / 8; i++) {
        const uint4 x = a4[i], y = b4[i];
        const half2* xh = reinterpret_cast<const half2*>(&x);
        const half2* yh = reinterpret_cast<const half2*>(&y);
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const float2 xf = __half22float2(xh[j]);
            const float2 yf = __half22float2(yh[j]);
            acc = fmaf(xf.x, yf.x, acc);
            acc = fmaf(xf.y, yf.y, acc);
        }
    }
    return acc;
}
#endif  // __CUDACC__

// ---------------------------------------------------------------------------
// Causal masking with N_q != N_kv is bottom-right aligned (same as upstream
// flash-attn): query row i may attend to keys j <= i + (N_kv - N_q).
// For N_q == N_kv this is the usual j <= i.
// ---------------------------------------------------------------------------

// Number of K/V blocks a Q block starting at q_row_start has to visit (forward).
__host__ __device__ __forceinline__ int fwd_kv_limit(
    bool causal, int q_row_start, int Br, int Bc, int seq_len_q, int seq_len_k
) {
    const int nkv = cdiv(seq_len_k, Bc);
    if (!causal) return nkv;
    const int last_visible_key = q_row_start + Br - 1 + (seq_len_k - seq_len_q);
    const int lim = (last_visible_key < 0) ? 0 : cdiv(last_visible_key + 1, Bc);
    return lim < nkv ? lim : nkv;
}

// Number of leading K/V blocks that need no masking for any row of the Q block
// [q_row_start, q_row_start + Br): fully inside seq_len_k and, if causal, fully
// visible to the first row (bottom-right alignment). Always <= fwd_kv_limit.
__host__ __device__ __forceinline__ int fwd_kv_nomask_limit(
    bool causal, int q_row_start, int Bc, int seq_len_q, int seq_len_k
) {
    const int full = seq_len_k / Bc;
    if (!causal) return full;
    const int visible = q_row_start + 1 + (seq_len_k - seq_len_q);  // keys seen by the first row
    const int lim = (visible <= 0) ? 0 : visible / Bc;
    return lim < full ? lim : full;
}

// First Q block that can see any key of the K/V block starting at kv_start (backward).
__host__ __device__ __forceinline__ int bwd_first_q_block(
    bool causal, int kv_start, int Br, int seq_len_q, int seq_len_k
) {
    if (!causal) return 0;
    const int first_q = kv_start - (seq_len_k - seq_len_q);
    return first_q <= 0 ? 0 : first_q / Br;
}

}  // namespace flash_attn_legacy
