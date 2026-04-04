#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>
#include <stdexcept>

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
    int seq_len;
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
    int batch_size, num_heads, num_heads_k, seq_len, head_dim;
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
// Tile sizes — tuned per architecture and head dimension
// ---------------------------------------------------------------------------

// VOLTA (SM 7.0): tensor cores (wmma 16x16x16), up to 96 KB shared
// Br = NUM_THREADS so every thread owns exactly one Q row →
// 100% utilization during softmax and P@V phases.

struct TileVolta_d32 {
    static constexpr int Br = 128;   // = 4 warps × 32 → 100% thread utilization
    static constexpr int Bc = 32;    // Must be <= d for PV_CHUNK loop (PV_CHUNK = Bc)
    static constexpr int d  = 32;
    // Shared: Q[128,32] + K[32,32] + V[32,32] (half) + S[128,32](float) + P[128,32](half)
    //       = (128*32 + 2*32*32)*2 + 128*32*4 + 128*32*2 = 12288 + 16384 + 8192 = 36 KB
    static constexpr int kNumWarps = 4;
};

struct TileVolta_d64 {
    static constexpr int Br = 128;   // = 4 warps × 32 → 100% thread utilization
    static constexpr int Bc = 64;
    static constexpr int d  = 64;
    // Shared: Q[128,64] + K[64,64] + V[64,64] + S[128,64]float
    //       = (128*64 + 2*64*64)*2 + 128*64*4 = 32768 + 32768 = 64 KB
    static constexpr int kNumWarps = 4;
};

struct TileVolta_d96 {
    static constexpr int Br = 128;   // = 4 warps × 32 → 100% thread utilization
    static constexpr int Bc = 32;
    static constexpr int d  = 96;
    // Shared: Q[128,96] + K[32,96] + V[32,96] (half) + S[128,32](float) + P[128,32](half)
    //       = (128*96 + 2*32*96)*2 + 128*32*4 + 128*32*2 = 36864 + 16384 + 8192 = 60 KB
    static constexpr int kNumWarps = 4;
};

struct TileVolta_d128 {
    static constexpr int Br = 128;   // = 4 warps × 32 → 100% thread utilization
    static constexpr int Bc = 32;
    static constexpr int d  = 128;
    // Shared: Q[128,128] + K[32,128] + V[32,128] + S[128,32]float
    //       = (128*128 + 2*32*128)*2 + 128*32*4 = 49152 + 16384 = 64 KB
    static constexpr int kNumWarps = 4;
};

struct TileVolta_d256 {
    static constexpr int Br = 64;    // = 2 warps × 32
    static constexpr int Bc = 16;
    static constexpr int d  = 256;
    // Shared: Q[64,256] + K[16,256] + V[16,256] (half) + S[64,16](float) + P[64,16](half)
    //       = (64*256 + 2*16*256)*2 + 64*16*4 + 64*16*2 = 49152 + 4096 + 2048 = 54 KB
    static constexpr int kNumWarps = 2;
};

// PASCAL (SM 6.x): no tensor cores, 48 KB shared memory
// Use half2 packed FP16 on CUDA cores.
// Br = NUM_THREADS for 100% thread utilization during softmax/P@V.

struct TilePascal_d32 {
    static constexpr int Br = 128;   // = 4 warps × 32 → 100% thread utilization
    static constexpr int Bc = 32;
    static constexpr int d  = 32;
    // Shared: Q[128,32] + K[32,32] + V[32,32] + P[128,32]float
    //       = (128*32 + 2*32*32)*2 + 128*32*4 = 12288 + 16384 = 28 KB
    static constexpr int kNumWarps = 4;
};

struct TilePascal_d64 {
    static constexpr int Br = 128;   // = 4 warps × 32 → 100% thread utilization
    static constexpr int Bc = 32;
    static constexpr int d  = 64;
    // Shared: Q[128,64] + K[32,64] + V[32,64] + P[128,32]float
    //       = (128*64 + 2*32*64)*2 + 128*32*4 = 24576 + 16384 = 40 KB
    static constexpr int kNumWarps = 4;
};

struct TilePascal_d96 {
    static constexpr int Br = 64;    // = 2 warps × 32
    static constexpr int Bc = 16;
    static constexpr int d  = 96;
    // Shared: Q[64,96] + K[16,96] + V[16,96] + P[64,16]float
    //       = (64*96 + 2*16*96)*2 + 64*16*4 = 18432 + 4096 = 22 KB
    static constexpr int kNumWarps = 2;
};

struct TilePascal_d128 {
    static constexpr int Br = 64;    // = 2 warps × 32 → 100% thread utilization
    static constexpr int Bc = 16;
    static constexpr int d  = 128;
    // Shared: Q[64,128] + K[16,128] + V[16,128] + P[64,16]float
    //       = (64*128 + 2*16*128)*2 + 64*16*4 = 24576 + 4096 = 28 KB
    static constexpr int kNumWarps = 2;  // 64 threads — keeps register pressure manageable
};

struct TilePascal_d256 {
    static constexpr int Br = 32;    // = 1 warp × 32
    static constexpr int Bc = 16;
    static constexpr int d  = 256;
    // Shared: Q[32,256] + K[16,256] + V[16,256] + P[32,16]float
    //       = (32*256 + 2*16*256)*2 + 32*16*4 = 32768 + 2048 = 34 KB
    static constexpr int kNumWarps = 1;
};

// ---------------------------------------------------------------------------
// Backward tile sizes — same for both archs, conservative
// All dimensions must be multiples of 16 for WMMA compatibility.
// THREADS_PER_KV = (kNumWarps*32) / Bc must be a power of 2.
// ---------------------------------------------------------------------------
struct TileBwd_d32 {
    static constexpr int Br = 32;
    static constexpr int Bc = 32;    // Must be <= d for DQ_CHUNK loop (DQ_CHUNK = Bc)
    static constexpr int d  = 32;
    static constexpr int kNumWarps = 4;
    // THREADS_PER_KV = 128/32 = 4
};

struct TileBwd_d64 {
    static constexpr int Br = 32;
    static constexpr int Bc = 32;
    static constexpr int d  = 64;
    static constexpr int kNumWarps = 4;
};

struct TileBwd_d96 {
    static constexpr int Br = 32;
    static constexpr int Bc = 16;
    static constexpr int d  = 96;
    static constexpr int kNumWarps = 4;
    // THREADS_PER_KV = 128/16 = 8
};

struct TileBwd_d128 {
    static constexpr int Br = 16;
    static constexpr int Bc = 16;
    static constexpr int d  = 128;
    static constexpr int kNumWarps = 4;
};

struct TileBwd_d256 {
    static constexpr int Br = 16;
    static constexpr int Bc = 16;
    static constexpr int d  = 256;
    static constexpr int kNumWarps = 4;
    // THREADS_PER_KV = 128/16 = 8
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
// Utility: ceiling division
// ---------------------------------------------------------------------------
__host__ __device__ __forceinline__ int cdiv(int a, int b) {
    return (a + b - 1) / b;
}

}  // namespace flash_attn_legacy
