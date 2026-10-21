#pragma once

// WMMA building blocks shared by the tensor-core forward and backward kernels
// (sm_70 / sm_75). Device code only exists for __CUDA_ARCH__ >= 700; the
// including .cu files provide stubs for older architectures.

#include <mma.h>  // file scope: inside a namespace it breaks cuda_bf16.h

#include "flash_attn_common.h"

namespace flash_attn_legacy {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700

using namespace nvcuda;

// ---------------------------------------------------------------------------
// WMMA matmul: C[M,N] = A[M,K] * B^T[N,K]
// A is [M, K_DIM] row-major in shared, B is [N, K_DIM] row-major in shared.
// We want A * B^T, so load B as col_major (which transposes it).
// Result stored to smem_C [M, N] as float with row stride LDC (default dense).
// All warps cooperate to cover all 16×16 output tiles.
// ---------------------------------------------------------------------------
template <int M, int N, int K_DIM, int NUM_WARPS, int LDC = N>
__device__ __forceinline__ void wmma_gemm_ABt(
    const half* __restrict__ smem_A,   // [M, K_DIM]
    const half* __restrict__ smem_B,   // [N, K_DIM]
    float*      __restrict__ smem_C,   // [M, N], row stride LDC
    int warp_id
) {
    constexpr int T = 16;
    constexpr int ntiles_n = N / T;
    constexpr int total_tiles = (M / T) * ntiles_n;
    constexpr int tiles_per_warp = (total_tiles + NUM_WARPS - 1) / NUM_WARPS;

    #pragma unroll
    for (int t = 0; t < tiles_per_warp; t++) {
        int tile_idx = warp_id + t * NUM_WARPS;
        if (tile_idx >= total_tiles) break;
        int tm = tile_idx / ntiles_n;
        int tn = tile_idx % ntiles_n;

        wmma::fragment<wmma::matrix_a, T, T, T, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, T, T, T, half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, T, T, T, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);
        #pragma unroll
        for (int k = 0; k < K_DIM; k += T) {
            wmma::load_matrix_sync(a_frag, smem_A + tm * T * K_DIM + k, K_DIM);
            wmma::load_matrix_sync(b_frag, smem_B + tn * T * K_DIM + k, K_DIM);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(smem_C + tm * T * LDC + tn * T, c_frag, LDC, wmma::mem_row_major);
    }
}

// ---------------------------------------------------------------------------
// WMMA matmul: C[M,N_OUT] = A[M,K] * B[K,N_OUT]  (NOT transposed)
// A is [M, K_DIM] row-major (row stride LDA, default dense) in shared mem.
// B is [K_DIM, N_FULL] row-major (stride B_stride) in shared mem;
//   pointer is pre-offset to the desired column chunk.
// Result stored to smem_C [M, N_OUT] as float (row stride LDC).
// ---------------------------------------------------------------------------
template <int M, int N_OUT, int K_DIM, int NUM_WARPS, int LDA = K_DIM, int LDC = N_OUT>
__device__ __forceinline__ void wmma_gemm_AB(
    const half* __restrict__ smem_A,
    const half* __restrict__ smem_B,
    int B_stride,
    float*      __restrict__ smem_C,
    int warp_id
) {
    constexpr int T = 16;
    constexpr int ntiles_n = N_OUT / T;
    constexpr int total_tiles = (M / T) * ntiles_n;
    constexpr int tiles_per_warp = (total_tiles + NUM_WARPS - 1) / NUM_WARPS;

    #pragma unroll
    for (int t = 0; t < tiles_per_warp; t++) {
        int tile_idx = warp_id + t * NUM_WARPS;
        if (tile_idx >= total_tiles) break;
        int tm = tile_idx / ntiles_n;
        int tn = tile_idx % ntiles_n;

        wmma::fragment<wmma::matrix_a, T, T, T, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, T, T, T, half, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, T, T, T, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);
        #pragma unroll
        for (int k = 0; k < K_DIM; k += T) {
            wmma::load_matrix_sync(a_frag, smem_A + tm * T * LDA + k, LDA);
            wmma::load_matrix_sync(b_frag, smem_B + k * B_stride + tn * T, B_stride);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(smem_C + tm * T * LDC + tn * T, c_frag, LDC, wmma::mem_row_major);
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

#endif  // __CUDA_ARCH__ >= 700

}  // namespace flash_attn_legacy
