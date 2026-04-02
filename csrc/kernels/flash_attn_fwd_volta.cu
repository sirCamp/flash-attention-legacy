// ============================================================================
// Flash Attention v2 — Forward Kernel for Volta (SM 7.0)
// Uses WMMA tensor cores for both Q@K^T and P@V matmuls (16×16×16 FP16)
//
// Key design:
//   - WMMA for S = Q @ K^T  → biggest compute bottleneck
//   - WMMA for O += P @ V   → second biggest compute bottleneck
//   - P stored in shared memory (float) for softmax, converted to half for WMMA
//   - Br = NUM_THREADS → 100% thread utilization during softmax
//   - 1 thread = 1 Q row for softmax and per-row O accumulation
//
// Architecture guard:
//   WMMA requires SM >= 7.0. This file is compiled for SM 6.x too
//   (gencode flags), but the kernel body is #ifdef'd to a stub for SM < 7.0.
//   Runtime dispatch in flash_attn_legacy.cpp ensures the Volta kernel is
//   never launched on Pascal.
// ============================================================================

#include "flash_attn_common.h"

namespace flash_attn_legacy {

// Guard all device code that uses WMMA intrinsics.
// Host compilation: __CUDA_ARCH__ is undefined → condition true → real code.
// SM 6.x device compilation: __CUDA_ARCH__ < 700 → stub.
// SM 7.0 device compilation: __CUDA_ARCH__ >= 700 → real code.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700

// WMMA types only available for SM >= 7.0
#include <mma.h>
using namespace nvcuda;

// ---------------------------------------------------------------------------
// Vectorized load (same as Pascal)
// ---------------------------------------------------------------------------
template <int ROWS, int COLS, int NUM_THREADS>
__device__ __forceinline__ void volta_load_tile(
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
// WMMA matmul: C[M,N] = A[M,K] * B^T[N,K]
// A is [M, K_DIM] row-major in shared, B is [N, K_DIM] row-major in shared.
// We want A * B^T, so load B as col_major (which transposes it).
// Result stored to smem_C [M, N] as float.
// All warps cooperate to cover all 16×16 output tiles.
// ---------------------------------------------------------------------------
template <int M, int N, int K_DIM, int NUM_WARPS>
__device__ __forceinline__ void wmma_gemm_ABt(
    const half* __restrict__ smem_A,   // [M, K_DIM]
    const half* __restrict__ smem_B,   // [N, K_DIM]
    float*      __restrict__ smem_C,   // [M, N]
    int warp_id
) {
    constexpr int TILE_M = 16;
    constexpr int TILE_N = 16;
    constexpr int TILE_K = 16;
    constexpr int ntiles_m = M / TILE_M;
    constexpr int ntiles_n = N / TILE_N;
    constexpr int total_tiles = ntiles_m * ntiles_n;
    constexpr int tiles_per_warp = (total_tiles + NUM_WARPS - 1) / NUM_WARPS;

    #pragma unroll
    for (int t = 0; t < tiles_per_warp; t++) {
        int tile_idx = warp_id + t * NUM_WARPS;
        if (tile_idx >= total_tiles) break;

        int tm = tile_idx / ntiles_n;
        int tn = tile_idx % ntiles_n;

        wmma::fragment<wmma::matrix_a, TILE_M, TILE_N, TILE_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, TILE_M, TILE_N, TILE_K, half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, TILE_M, TILE_N, TILE_K, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        // Accumulate over K dimension
        #pragma unroll
        for (int k = 0; k < K_DIM; k += TILE_K) {
            wmma::load_matrix_sync(a_frag, smem_A + tm * TILE_M * K_DIM + k, K_DIM);
            wmma::load_matrix_sync(b_frag, smem_B + tn * TILE_N * K_DIM + k, K_DIM);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        wmma::store_matrix_sync(smem_C + tm * TILE_M * N + tn * TILE_N, c_frag, N, wmma::mem_row_major);
    }
}

// ---------------------------------------------------------------------------
// WMMA matmul: C[M,N_OUT] = A[M,K] * B[K,N_OUT]  (NOT transposed)
// A is [M, K_DIM] row-major (stride K_DIM) in shared mem.
// B is [K_DIM, N_FULL] row-major (stride B_stride) in shared mem;
//   pointer is pre-offset to the desired column chunk.
// Result stored to smem_C [M, N_OUT] as float (stride N_OUT).
// All warps cooperate to cover all 16×16 output tiles.
// ---------------------------------------------------------------------------
template <int M, int N_OUT, int K_DIM, int NUM_WARPS>
__device__ __forceinline__ void wmma_gemm_AB(
    const half* __restrict__ smem_A,   // [M, K_DIM] row-major, stride K_DIM
    const half* __restrict__ smem_B,   // [K_DIM, ...] row-major, stride B_stride, pre-offset
    int B_stride,                       // leading dim of B (= full N_FULL, e.g. head_dim)
    float*      __restrict__ smem_C,   // [M, N_OUT] output, stride N_OUT
    int warp_id
) {
    constexpr int TILE_M = 16;
    constexpr int TILE_N = 16;
    constexpr int TILE_K = 16;
    constexpr int ntiles_m = M / TILE_M;
    constexpr int ntiles_n = N_OUT / TILE_N;
    constexpr int total_tiles = ntiles_m * ntiles_n;
    constexpr int tiles_per_warp = (total_tiles + NUM_WARPS - 1) / NUM_WARPS;

    #pragma unroll
    for (int t = 0; t < tiles_per_warp; t++) {
        int tile_idx = warp_id + t * NUM_WARPS;
        if (tile_idx >= total_tiles) break;

        int tm = tile_idx / ntiles_n;
        int tn = tile_idx % ntiles_n;

        wmma::fragment<wmma::matrix_a, TILE_M, TILE_N, TILE_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, TILE_M, TILE_N, TILE_K, half, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, TILE_M, TILE_N, TILE_K, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        // Accumulate over K dimension
        #pragma unroll
        for (int k = 0; k < K_DIM; k += TILE_K) {
            // A tile: rows [tm*16, tm*16+16), cols [k, k+16)
            wmma::load_matrix_sync(a_frag, smem_A + tm * TILE_M * K_DIM + k, K_DIM);
            // B tile: rows [k, k+16), cols [tn*16, tn*16+16) of the chunk
            // smem_B is pre-offset, stride = B_stride (full width of V)
            wmma::load_matrix_sync(b_frag, smem_B + k * B_stride + tn * TILE_N, B_stride);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        wmma::store_matrix_sync(smem_C + tm * TILE_M * N_OUT + tn * TILE_N, c_frag, N_OUT, wmma::mem_row_major);
    }
}

// ---------------------------------------------------------------------------
// Forward kernel — Volta
// ---------------------------------------------------------------------------
template <typename Tile>
__global__ void __launch_bounds__(Tile::kNumWarps * 32)
flash_attn_fwd_volta_kernel(FlashAttnParams params) {

    constexpr int Br = Tile::Br;
    constexpr int Bc = Tile::Bc;
    constexpr int d  = Tile::d;
    constexpr int NUM_THREADS = Tile::kNumWarps * 32;
    constexpr int NUM_WARPS = Tile::kNumWarps;

    // P@V output chunk size: process d in chunks of Bc columns.
    // smem_PV reuses smem_S space, both are [Br, Bc] in their respective types.
    // d=64, Bc=64 → 1 pass.  d=128, Bc=32 → 4 passes.
    constexpr int PV_CHUNK = Bc;

    const int bh_idx = blockIdx.y;
    const int b_idx = bh_idx / params.num_heads;
    const int h_idx = bh_idx % params.num_heads;
    const int kv_h_idx = h_idx / (params.num_heads / params.num_heads_k);
    const int q_row_start = blockIdx.x * Br;
    if (q_row_start >= params.seq_len) return;

    const int valid_q = min(Br, params.seq_len - q_row_start);

    // Base pointers
    const half* Q_ptr = params.Q + b_idx * params.q_batch_stride
                                 + h_idx * params.q_head_stride
                                 + q_row_start * params.q_seq_stride;
    const half* K_ptr = params.K + b_idx * params.k_batch_stride
                                 + kv_h_idx * params.k_head_stride;
    const half* V_ptr = params.V + b_idx * params.v_batch_stride
                                 + kv_h_idx * params.v_head_stride;
    half* O_ptr = params.O + b_idx * params.o_batch_stride
                           + h_idx * params.o_head_stride
                           + q_row_start * params.o_seq_stride;
    float* L_ptr = params.L + b_idx * params.l_batch_stride
                            + h_idx * params.l_head_stride
                            + q_row_start;

    // Shared memory layout:
    //   smem_Q:  [Br, d]  half
    //   smem_K:  [Bc, d]  half
    //   smem_V:  [Bc, d]  half
    //   smem_S:  [Br, Bc] float  (also reused as smem_PV after P→half conversion)
    //   smem_P:  [Br, Bc] half   (P converted to half for WMMA P@V)
    extern __shared__ char smem_raw[];
    half*  smem_Q = reinterpret_cast<half*>(smem_raw);
    half*  smem_K = smem_Q + Br * d;
    half*  smem_V = smem_K + Bc * d;
    float* smem_S = reinterpret_cast<float*>(smem_V + Bc * d);
    half*  smem_P = reinterpret_cast<half*>(smem_S + Br * Bc);
    // smem_PV overlaps smem_S — safe because smem_S is free after P conversion
    float* smem_PV = smem_S;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const bool owns_row = (tid < Br) && (tid < valid_q);

    // Per-row state
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float my_O[d];
    #pragma unroll
    for (int j = 0; j < d; j++) my_O[j] = 0.0f;

    // Load Q
    volta_load_tile<Br, d, NUM_THREADS>(Q_ptr, params.q_seq_stride, smem_Q, valid_q);
    __syncthreads();

    // Main loop
    const int nkv = cdiv(params.seq_len, Bc);
    const int kv_limit = params.is_causal ? min(nkv, cdiv(q_row_start + Br, Bc)) : nkv;

    for (int kv_blk = 0; kv_blk < kv_limit; kv_blk++) {
        const int kv_start = kv_blk * Bc;
        const int valid_kv = min(Bc, params.seq_len - kv_start);

        // Load K
        volta_load_tile<Bc, d, NUM_THREADS>(
            K_ptr + kv_start * params.k_seq_stride,
            params.k_seq_stride, smem_K, valid_kv);
        __syncthreads();

        // S = Q @ K^T via WMMA (all warps cooperate)
        wmma_gemm_ABt<Br, Bc, d, NUM_WARPS>(smem_Q, smem_K, smem_S, warp_id);
        __syncthreads();

        // Softmax: scale, mask, online update → P stored in smem_S as float
        if (owns_row) {
            float local_max = -INFINITY;

            #pragma unroll
            for (int c = 0; c < Bc; c++) {
                float s = smem_S[tid * Bc + c] * params.softmax_scale;
                if ((kv_start + c) >= params.seq_len ||
                    (params.is_causal && (kv_start + c) > (q_row_start + tid))) {
                    s = -INFINITY;
                }
                smem_S[tid * Bc + c] = s;
                local_max = fmaxf(local_max, s);
            }

            float new_max = fmaxf(row_max, local_max);
            float rescale = expf(row_max - new_max);
            row_max = new_max;
            row_sum *= rescale;
            #pragma unroll
            for (int j = 0; j < d; j++) my_O[j] *= rescale;

            float local_sum = 0.0f;
            #pragma unroll
            for (int c = 0; c < Bc; c++) {
                float s = smem_S[tid * Bc + c];
                float p = (s == -INFINITY) ? 0.0f : expf(s - new_max);
                smem_S[tid * Bc + c] = p;
                local_sum += p;
            }
            row_sum += local_sum;
        }
        __syncthreads();

        // Load V
        volta_load_tile<Bc, d, NUM_THREADS>(
            V_ptr + kv_start * params.v_seq_stride,
            params.v_seq_stride, smem_V, valid_kv);

        // Convert P from float (smem_S) to half (smem_P) for WMMA
        {
            constexpr int total_P = Br * Bc;
            #pragma unroll
            for (int idx = tid; idx < total_P; idx += NUM_THREADS) {
                smem_P[idx] = __float2half(smem_S[idx]);
            }
        }
        __syncthreads();

        // O += P @ V via WMMA
        // Process d in chunks of PV_CHUNK columns.
        // smem_PV (= smem_S, now free) holds the [Br, PV_CHUNK] float output.
        #pragma unroll
        for (int dn = 0; dn < d; dn += PV_CHUNK) {
            wmma_gemm_AB<Br, PV_CHUNK, Bc, NUM_WARPS>(
                smem_P,              // A = P [Br, Bc] half
                smem_V + dn,         // B = V[:, dn:dn+PV_CHUNK], stride = d
                d,                   // B stride (full head_dim)
                smem_PV,             // C = output [Br, PV_CHUNK] float
                warp_id
            );
            __syncthreads();

            // Each thread accumulates its row from the WMMA output
            if (owns_row) {
                #pragma unroll
                for (int j = 0; j < PV_CHUNK; j++) {
                    my_O[dn + j] += smem_PV[tid * PV_CHUNK + j];
                }
            }
            __syncthreads();
        }
    }

    // Finalize
    if (owns_row) {
        float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
        half2* o_dst2 = reinterpret_cast<half2*>(O_ptr + tid * params.o_seq_stride);
        constexpr int d2 = d / 2;
        #pragma unroll
        for (int j = 0; j < d2; j++) {
            half2 val;
            val.x = __float2half(my_O[j * 2] * inv_sum);
            val.y = __float2half(my_O[j * 2 + 1] * inv_sum);
            o_dst2[j] = val;
        }
        L_ptr[tid] = row_max + logf(fmaxf(row_sum, 1e-20f));
    }
}

#else  // __CUDA_ARCH__ < 700

// Stub kernel for SM < 7.0 — never launched at runtime (dispatch in .cpp).
template <typename Tile>
__global__ void __launch_bounds__(Tile::kNumWarps * 32)
flash_attn_fwd_volta_kernel(FlashAttnParams params) {}

#endif  // __CUDA_ARCH__

// ---------------------------------------------------------------------------
// Host launcher (always compiled — runtime dispatch ensures correctness)
// ---------------------------------------------------------------------------
void flash_attn_fwd_volta(FlashAttnParams& params, cudaStream_t stream) {
    const int nbh = params.batch_size * params.num_heads;

    if (params.head_dim == 64) {
        using T = TileVolta_d64;
        // Shared: Q[Br,d] + K[Bc,d] + V[Bc,d] (half) + S[Br,Bc] (float) + P[Br,Bc] (half)
        constexpr int smem = (T::Br * T::d + 2 * T::Bc * T::d) * sizeof(half)
                           + T::Br * T::Bc * sizeof(float)
                           + T::Br * T::Bc * sizeof(half);
        dim3 grid(cdiv(params.seq_len, T::Br), nbh);
        dim3 block(T::kNumWarps * 32);

        FLASH_ATTN_CHECK_CUDA(cudaFuncSetAttribute(flash_attn_fwd_volta_kernel<T>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        flash_attn_fwd_volta_kernel<T><<<grid, block, smem, stream>>>(params);
    } else {
        using T = TileVolta_d128;
        constexpr int smem = (T::Br * T::d + 2 * T::Bc * T::d) * sizeof(half)
                           + T::Br * T::Bc * sizeof(float)
                           + T::Br * T::Bc * sizeof(half);
        dim3 grid(cdiv(params.seq_len, T::Br), nbh);
        dim3 block(T::kNumWarps * 32);

        FLASH_ATTN_CHECK_CUDA(cudaFuncSetAttribute(flash_attn_fwd_volta_kernel<T>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        flash_attn_fwd_volta_kernel<T><<<grid, block, smem, stream>>>(params);
    }
}

}  // namespace flash_attn_legacy
