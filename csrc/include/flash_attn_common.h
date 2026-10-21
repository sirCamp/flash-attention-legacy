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

// Element offsets and strides: 64-bit, so that tensors beyond 2^31 elements
// (large packed varlen batches) address correctly; products of an int row
// index and an index_t stride are computed in 64 bits.
using index_t = int64_t;

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
    int head_dim;        // 32, 64, 72, 80, 96, 128, or 256

    // Strides (in elements) for Q
    index_t q_batch_stride;
    index_t q_head_stride;
    index_t q_seq_stride;    // == head_dim for contiguous

    // Strides for K
    index_t k_batch_stride;
    index_t k_head_stride;
    index_t k_seq_stride;

    // Strides for V
    index_t v_batch_stride;
    index_t v_head_stride;
    index_t v_seq_stride;

    // Strides for O
    index_t o_batch_stride;
    index_t o_head_stride;
    index_t o_seq_stride;

    // Strides for L (logsumexp)
    index_t l_batch_stride;
    index_t l_head_stride;

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
    int head_dim;        // 32, 64, 72, 80, 96, 128, or 256

    // Strides (in elements) — layout is [total, H, d]
    index_t q_token_stride;  // H_q * d
    index_t q_head_stride;   // d
    index_t k_token_stride;  // H_kv * d
    index_t k_head_stride;   // d
    index_t v_token_stride;
    index_t v_head_stride;
    index_t o_token_stride;
    index_t o_head_stride;

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
    index_t q_batch_stride, q_head_stride, q_seq_stride;
    // Strides for K/dK
    index_t k_batch_stride, k_head_stride, k_seq_stride;
    // Strides for V/dV
    index_t v_batch_stride, v_head_stride, v_seq_stride;
    // Strides for O/dO
    index_t o_batch_stride, o_head_stride, o_seq_stride;
    // Strides for L and D (batched [B, H, N]: seq stride 1; varlen [total, H]: num_heads)
    index_t l_batch_stride, l_head_stride, l_seq_stride;
    // Strides for dQ (FP32, same layout as Q but float)
    index_t dq_batch_stride, dq_head_stride, dq_seq_stride;
    // Strides for dK (same layout as K)
    index_t dk_batch_stride, dk_head_stride, dk_seq_stride;
    // Strides for dV (same layout as V)
    index_t dv_batch_stride, dv_head_stride, dv_seq_stride;

    // Varlen only (nullptr / 0 for batched): packed [total, H, d] tensors with
    // batch strides 0; sequence i spans rows [cu_seqlens[i], cu_seqlens[i+1]);
    // key_block_offsets[i] = key blocks of the sequences before i.
    const int* __restrict__ cu_seqlens_q;
    const int* __restrict__ cu_seqlens_k;
    const int* __restrict__ key_block_offsets;
    int num_seqs;
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
    static constexpr bool kPrefetch = true;  // K/V register prefetch (measured faster for every tile)
    static constexpr int smem_bytes() {
        return (Br + 2 * Bc) * kLdH * 2 + kNumWarps * (kStageFloats + 16) * 4;
    }
    static_assert(Br_ % 16 == 0 && Bc_ % 16 == 0 && d_ % 16 == 0, "invalid WMMA forward tile");
    // The staging tile also holds the one-time accumulator layout probe
    // (256 floats + a 256-byte table).
    static_assert(kStageFloats >= 256 + 64, "WMMA forward staging too small");
};

// CUDA-core forward (Pascal SM 6.x), same band scheme as the WMMA one: each
// warp owns 16 query rows; lane = 8 * rg + cg holds rows 4rg..4rg+3 of the band,
// keys cg + 8j and O columns 4cg + 32i..+3, all in fp32 registers.
// Shared memory, fp32 (converted once when stored, no conversion in the inner
// products): Q[Br,kLdF], K[Bc,kLdF], V[Bc,kLdF] (rows padded by 4 floats: the 8
// K rows read at once start in different 4-bank groups), then per warp P^T
// [Bc,kLdP] (key-major; stride 20 makes the 8 float4 stores conflict free).
template <int Br_, int Bc_, int d_>
struct FwdTileSimt {
    static constexpr int Br = Br_;
    static constexpr int Bc = Bc_;
    static constexpr int d  = d_;
    static constexpr int kNumWarps = Br_ / 16;
    static constexpr int kLdF = d_ + 4;
    static constexpr int kLdP = 20;
    // Prefetch the next K/V block into registers only when it costs at most two
    // uint4 per thread and tile (else plain shared-memory loads); same count as
    // TileRegs<Bc, d, threads>::kPerThread, which is device-only.
    static constexpr bool kPrefetch = (Bc_ * d_ / 8 + kNumWarps * 32 - 1) / (kNumWarps * 32) <= 2;
    static constexpr int smem_bytes() {
        return ((Br + 2 * Bc) * kLdF + kNumWarps * Bc * kLdP) * 4;
    }
    static_assert(Br_ % 16 == 0 && Bc_ % 8 == 0 && d_ % 32 == 0, "invalid CUDA-core forward tile");
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
using TileVolta_d80  = FwdTileWmma<128, 16, 80>;    // 40 KB, also d=72 (zero-padded columns):
                                                    // N=4096 V100 3.1 ms (SDPA 3.4), T4 5.9 (8.2)
using TileVolta_d96  = FwdTileWmma<128, 16, 96>;    // 45 KB
using TileVolta_d128 = FwdTileWmma<128, 32, 128>;   // 72 KB (V100)
using TileVolta_d128_64k = FwdTileWmma<128, 16, 128>;  // 55 KB
using TileVolta_d256 = FwdTileWmma<128, 16, 256>;   // 95 KB (V100)
using TileVolta_d256_64k = FwdTileWmma<64, 16, 256>;   // 56 KB

// CUDA-core forward tiles (48 KB per block on Pascal). Measured, forward causal
// B=2 H=16 N=4096, P100 / 1080 Ti in ms (thread-per-row kernel before P2.8):
// d=32 6.1 / 7.3 (13.4 / 24); d=64 12.2 / 15.4 (27 / 74); d=96 21.8 / 23.8
// (47 / 118); d=128 52 / 41 (81 / 176); d=256 285 / 189 (1768 on P100). fp32
// shared memory limits d=256 to one-warp blocks (docs/PLAN.md P2.8).
using TilePascal_d32  = FwdTileSimt<64, 32, 32>;    // 28 KB
using TilePascal_d64  = FwdTileSimt<64, 16, 64>;    // 31 KB
using TilePascal_d96  = FwdTileSimt<64, 16, 96>;    // 43 KB
using TilePascal_d128 = FwdTileSimt<32, 8, 128>;    // 26 KB
using TilePascal_d256 = FwdTileSimt<16, 8, 256>;    // 33 KB

// First tile of the list whose smem_bytes() fits `smem_limit`: calls f(Tile{})
// and returns true, or returns false if none fits. Shared by every launcher and
// by the host code that needs the selected Br (varlen block offsets), so the
// selection can never disagree between them. Lists with more than one tile
// give devices with more shared memory per block a larger tile (e.g. the WMMA
// forward at d=128/256: V100 96 KB, T4 64 KB).
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
// Backward tile sizes — same template scheme as the forward. A block owns Bc
// keys (dK/dV for them) and loops over the query blocks of Br rows.
// ---------------------------------------------------------------------------

// WMMA backward (Volta / Turing), FA2-style: warp w owns the band of 16 keys
// (w / kSplitD) and the slice (w % kSplitD) of d / kSplitD columns of dK / dV,
// accumulated in WMMA fragments (registers). Shared memory: K, V [Bc,kLdH],
// Q, dO [Br,kLdH] (half, rows padded by 8), L, D [Br] (float), dS^T [Bc,kLdB]
// (half, key-major, for dQ = dS K), and per warp a [16,kLdB] half buffer (P^T,
// then dS^T, as WMMA operands; also the float staging of dQ / dK / dV tiles and
// the one-time fragment layout probe, hence Br >= 32).
template <int Br_, int Bc_, int d_, int kSplitD_ = 1>
struct BwdTileWmma {
    static constexpr int Br = Br_;
    static constexpr int Bc = Bc_;
    static constexpr int d  = d_;
    static constexpr int kSplitD = kSplitD_;
    static constexpr int kNumWarps = (Bc_ / 16) * kSplitD_;
    static constexpr int kLdH = d_ + 8;
    static constexpr int kLdB = Br_ + 8;
    static constexpr int kWarpBufBytes = 16 * kLdB * 2;
    static constexpr int smem_bytes() {
        return (2 * Bc + 2 * Br) * kLdH * 2 + 2 * Br * 4 + Bc * kLdB * 2 + kNumWarps * kWarpBufBytes;
    }
    static_assert(Br_ % 16 == 0 && Bc_ % 16 == 0 && d_ % (16 * kSplitD_) == 0, "invalid WMMA backward tile");
    static_assert(kWarpBufBytes >= 256 * 4 + 256, "warp buffer too small for the layout probe (Br >= 32)");
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

// Softmax in the log2 domain (exp2 is one instruction): scores * scale * log2(e).
constexpr float kLog2e = 1.4426950408889634f;
constexpr float kLn2 = 0.6931471805599453f;

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
// Global -> shared tile copies with 128-bit transactions, split in two halves so
// that a kernel can issue the global loads of the next tile early (into
// registers) and store them to shared memory later: load_tile_g2r reads a
// [ROWS, COLS] half tile (row stride src_row_stride) into TileRegs, rows >=
// num_valid_rows and columns >= num_valid_cols (a multiple of 8: head_dim 72 /
// 80 on a tile of 80 / 96 columns) as zeros; store_tile_r2s writes it with row stride DST_STRIDE
// (>= COLS, padded against bank conflicts), as half or converted to fp32.
// load_tile_g2s does both directly.
// Rows must start 16-byte aligned (COLS, strides multiples of 8 halves;
// contiguous_aligned() on the host). The row stride is an int (one row of a
// tensor fits in 32 bits; 64-bit math here costs a lot in small-tile loops):
// callers advance `src` with 64-bit offsets first.
// ---------------------------------------------------------------------------
template <int ROWS, int COLS, int NUM_THREADS>
struct TileRegs {
    static_assert(COLS % 8 == 0, "128-bit tile copies need COLS % 8 == 0");
    static constexpr int kVecPerRow = COLS / 8;
    static constexpr int kVecs = ROWS * kVecPerRow;
    static constexpr int kThreads = NUM_THREADS;
    static constexpr int kPerThread = (kVecs + NUM_THREADS - 1) / NUM_THREADS;
    uint4 v[kPerThread];
};

// Vector idx of a tile copy: row idx / kVecPerRow, 8 halves at column 8 * (idx % kVecPerRow).
template <typename R>
__device__ __forceinline__ uint4 tile_vec_load(const half* __restrict__ src, int src_row_stride,
                                               int idx, int num_valid_rows, int num_valid_cols) {
    const int row = idx / R::kVecPerRow;
    const int col = (idx % R::kVecPerRow) * 8;
    if ((R::kVecs % R::kThreads != 0 && idx >= R::kVecs) || row >= num_valid_rows || col >= num_valid_cols) {
        return make_uint4(0u, 0u, 0u, 0u);
    }
    return *reinterpret_cast<const uint4*>(src + row * src_row_stride + col);
}

template <typename R, int DST_STRIDE>
__device__ __forceinline__ void tile_vec_store(half* __restrict__ dst, int idx, uint4 v) {
    static_assert(DST_STRIDE % 8 == 0, "128-bit tile copies need DST_STRIDE % 8 == 0");
    if (R::kVecs % R::kThreads == 0 || idx < R::kVecs) {
        *reinterpret_cast<uint4*>(dst + (idx / R::kVecPerRow) * DST_STRIDE
                                  + (idx % R::kVecPerRow) * 8) = v;
    }
}

// Columns of a tile of d columns that hold data for this head_dim: tiles of 80
// and 96 columns also run head_dim 72 / 80 with zero-padded columns, every
// other tile has head_dim == d (a compile-time constant: no runtime checks).
template <int d>
__host__ __device__ __forceinline__ int tile_valid_cols(int head_dim) {
    return (d == 80 || d == 96) ? head_dim : d;
}

// Varlen grids: sequence of a flat block index, from the cumulative per-sequence
// block counts offsets[0..num_seqs] (upper-bound binary search; sequences with no
// block are skipped). Returns num_seqs for a block past the last sequence.
__device__ __forceinline__ int varlen_seq_of_block(const int* __restrict__ offsets, int num_seqs, int block) {
    int lo = 0, hi = num_seqs;
    while (lo < hi) {
        const int mid = (lo + hi) / 2;
        if (offsets[mid + 1] <= block) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// fp32 destination: the 8 halves are converted once, when stored (the CUDA-core
// kernels compute in fp32 from shared memory).
template <typename R, int DST_STRIDE>
__device__ __forceinline__ void tile_vec_store(float* __restrict__ dst, int idx, uint4 v) {
    static_assert(DST_STRIDE % 4 == 0, "128-bit tile copies need DST_STRIDE % 4 == 0 (floats)");
    if (R::kVecs % R::kThreads == 0 || idx < R::kVecs) {
        const half2* h = reinterpret_cast<const half2*>(&v);
        const float2 f0 = __half22float2(h[0]), f1 = __half22float2(h[1]);
        const float2 f2 = __half22float2(h[2]), f3 = __half22float2(h[3]);
        float4* d4 = reinterpret_cast<float4*>(dst + (idx / R::kVecPerRow) * DST_STRIDE
                                               + (idx % R::kVecPerRow) * 8);
        d4[0] = make_float4(f0.x, f0.y, f1.x, f1.y);
        d4[1] = make_float4(f2.x, f2.y, f3.x, f3.y);
    }
}

template <int ROWS, int COLS, int NUM_THREADS>
__device__ __forceinline__ void load_tile_g2r(
    const half* __restrict__ src, int src_row_stride,
    TileRegs<ROWS, COLS, NUM_THREADS>& regs, int num_valid_rows, int num_valid_cols = COLS
) {
    using R = TileRegs<ROWS, COLS, NUM_THREADS>;
    #pragma unroll
    for (int i = 0; i < R::kPerThread; i++) {
        regs.v[i] = tile_vec_load<R>(src, src_row_stride, threadIdx.x + i * NUM_THREADS, num_valid_rows,
                                     num_valid_cols);
    }
}

template <int ROWS, int COLS, int NUM_THREADS, int DST_STRIDE = COLS, typename DstT>
__device__ __forceinline__ void store_tile_r2s(
    const TileRegs<ROWS, COLS, NUM_THREADS>& regs, DstT* __restrict__ dst
) {
    using R = TileRegs<ROWS, COLS, NUM_THREADS>;
    #pragma unroll
    for (int i = 0; i < R::kPerThread; i++) {
        tile_vec_store<R, DST_STRIDE>(dst, threadIdx.x + i * NUM_THREADS, regs.v[i]);
    }
}

// One vector at a time (no tile-sized register buffer).
template <int ROWS, int COLS, int NUM_THREADS, int DST_STRIDE = COLS, typename DstT>
__device__ __forceinline__ void load_tile_g2s(
    const half* __restrict__ src, int src_row_stride,
    DstT* __restrict__ dst, int num_valid_rows, int num_valid_cols = COLS
) {
    using R = TileRegs<ROWS, COLS, NUM_THREADS>;
    #pragma unroll
    for (int i = 0; i < R::kPerThread; i++) {
        const int idx = threadIdx.x + i * NUM_THREADS;
        tile_vec_store<R, DST_STRIDE>(dst, idx, tile_vec_load<R>(src, src_row_stride, idx, num_valid_rows,
                                                                 num_valid_cols));
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
