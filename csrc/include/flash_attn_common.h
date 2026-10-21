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
// runtime tile selection). Br = threads per block: 1 thread owns 1 Q row.
// ---------------------------------------------------------------------------

// WMMA forward (Volta SM 7.0, Turing SM 7.5): Q[Br,d] + K[Bc,d] + V[Bc,d] (half),
// S[Br,Bc] (float, reused for the P@V chunks), P[Br,Bc] (half).
// Constraints: Br, Bc multiples of 16; d % Bc == 0 (P@V in chunks of Bc).
template <int Br_, int Bc_, int d_>
struct FwdTileWmma {
    static constexpr int Br = Br_;
    static constexpr int Bc = Bc_;
    static constexpr int d  = d_;
    static constexpr int kNumWarps = Br_ / 32;
    static constexpr int smem_bytes() {
        return (Br * d + 2 * Bc * d) * 2 + Br * Bc * 4 + Br * Bc * 2;
    }
    static_assert(Br_ % 32 == 0 && Bc_ % 16 == 0 && d_ % Bc_ == 0, "invalid WMMA forward tile");
};

// CUDA-core forward (Pascal SM 6.x): Q[Br,d] + K[Bc,d] + V[Bc,d] (half), P[Br,Bc] (float).
template <int Br_, int Bc_, int d_>
struct FwdTileSimt {
    static constexpr int Br = Br_;
    static constexpr int Bc = Bc_;
    static constexpr int d  = d_;
    static constexpr int kNumWarps = Br_ / 32;
    static constexpr int smem_bytes() {
        return (Br * d + 2 * Bc * d) * 2 + Br * Bc * 4;
    }
    static_assert(Br_ % 32 == 0, "invalid CUDA-core forward tile");
};

// Forward tiles: Bc = 16 for every head_dim, measured fastest on V100 and T4
// (smaller tiles -> more blocks per SM; see the WMMA launcher). Every WMMA tile
// fits the 64 KB per block of Turing.
using TileVolta_d32  = FwdTileWmma<128, 16, 32>;    // 22 KB
using TileVolta_d64  = FwdTileWmma<128, 16, 64>;    // 32 KB
using TileVolta_d96  = FwdTileWmma<128, 16, 96>;    // 43 KB
using TileVolta_d128 = FwdTileWmma<128, 16, 128>;   // 52 KB
using TileVolta_d256 = FwdTileWmma<64, 16, 256>;    // 54 KB

using TilePascal_d32  = FwdTileSimt<128, 32, 32>;   // 28 KB
using TilePascal_d64  = FwdTileSimt<128, 32, 64>;   // 40 KB
using TilePascal_d96  = FwdTileSimt<64, 16, 96>;    // 22 KB
using TilePascal_d128 = FwdTileSimt<64, 16, 128>;   // 28 KB
using TilePascal_d256 = FwdTileSimt<32, 16, 256>;   // 34 KB

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
// Br, Bc, d multiples of 16 (WMMA); d % Bc == 0 (dQ in chunks of Bc);
// THREADS_PER_KV = (kNumWarps*32) / Bc must be a power of 2 (CUDA-core kernel).
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

// CUDA-core backward (Pascal): K, V [Bc,d], Q, dO [Br,d] (half); dK/dV in registers.
template <int Br_, int Bc_, int d_, int kNumWarps_ = 4>
struct BwdTileSimt {
    static constexpr int Br = Br_;
    static constexpr int Bc = Bc_;
    static constexpr int d  = d_;
    static constexpr int kNumWarps = kNumWarps_;
    static constexpr int smem_bytes() { return (2 * Bc * d + 2 * Br * d) * 2; }
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

// First Q block that can see any key of the K/V block starting at kv_start (backward).
__host__ __device__ __forceinline__ int bwd_first_q_block(
    bool causal, int kv_start, int Br, int seq_len_q, int seq_len_k
) {
    if (!causal) return 0;
    const int first_q = kv_start - (seq_len_k - seq_len_q);
    return first_q <= 0 ? 0 : first_q / Br;
}

}  // namespace flash_attn_legacy
