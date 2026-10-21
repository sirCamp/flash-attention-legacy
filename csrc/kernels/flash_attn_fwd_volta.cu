// ============================================================================
// Flash Attention v2 — Forward Kernel for Volta (SM 7.0) and Turing (SM 7.5)
// Uses WMMA tensor cores for both Q@K^T and P@V matmuls (16×16×16 FP16)
//
// Key design:
//   - WMMA for S = Q @ K^T and O += P @ V
//   - 1 thread = 1 Q row (Br = threads) for the softmax and the O accumulation;
//     each thread reads its S row (float) and writes its P row (half) with
//     128-bit accesses on padded rows (bank-conflict free)
//   - one block body (volta_fwd_block) shared by the batched and varlen kernels
//
// Architecture guard:
//   WMMA requires SM >= 7.0. This file is compiled for SM 6.x too
//   (gencode flags), but the kernel body is #ifdef'd to a stub for SM < 7.0.
//   Runtime dispatch in flash_attn_legacy.cpp ensures the Volta kernel is
//   never launched on Pascal.
// ============================================================================

#include "flash_attn_common.h"

#include <mma.h>  // file scope: inside a namespace it breaks cuda_bf16.h
#include <cstring>

#include <string>

namespace flash_attn_legacy {

// Guard all device code that uses WMMA intrinsics.
// Host compilation: __CUDA_ARCH__ is undefined → condition true → real code.
// SM 6.x device compilation: __CUDA_ARCH__ < 700 → stub.
// SM 7.0 device compilation: __CUDA_ARCH__ >= 700 → real code.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700

// WMMA types only available for SM >= 7.0
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
// Result stored to smem_C [M, N] as float with row stride LDC.
// All warps cooperate to cover all 16×16 output tiles.
// ---------------------------------------------------------------------------
template <int M, int N, int K_DIM, int NUM_WARPS, int LDC>
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
// A is [M, K_DIM] row-major (row stride LDA) in shared mem.
// B is [K_DIM, N_FULL] row-major (stride B_stride) in shared mem;
//   pointer is pre-offset to the desired column chunk.
// Result stored to smem_C [M, N_OUT] as float (row stride LDC).
// ---------------------------------------------------------------------------
template <int M, int N_OUT, int K_DIM, int NUM_WARPS, int LDA, int LDC>
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

// ---------------------------------------------------------------------------
// One Q block of the WMMA forward, shared by the batched and varlen kernels
// (they only differ in how they locate Q/K/V/O and where they write L).
//
// Shared memory (Tile::smem_bytes()):
//   smem_Q [Br, d] half, smem_K [Bc, d] half, smem_V [Bc, d] half
//   smem_S [Br, LDS] float  — S, then reused for the P@V chunks (LDS = Bc + 4)
//   smem_P [Br, LDP] half   — P for the P@V WMMA (LDP = Bc + 8)
// Each thread owns one Q row and reads/writes its row of S/P/PV with 128-bit
// accesses. The padded row strides make those accesses bank-conflict free (a
// quarter-warp reads 8 rows that start in 8 different 4-bank groups); without
// padding all 32 threads of a warp hit the same bank (docs/PLAN.md §1b).
// Returns the row's logsumexp (only meaningful for threads that own a row).
// ---------------------------------------------------------------------------
template <typename Tile>
__device__ __forceinline__ float volta_fwd_block(
    const half* __restrict__ Q_ptr, int q_stride,
    const half* __restrict__ K_ptr, int k_stride,
    const half* __restrict__ V_ptr, int v_stride,
    half* __restrict__ O_ptr, int o_stride,
    int q_row_start, int valid_q, int seq_len_q, int seq_len_k,
    float softmax_scale, bool is_causal
) {
    constexpr int Br = Tile::Br;
    constexpr int Bc = Tile::Bc;
    constexpr int d  = Tile::d;
    constexpr int NUM_THREADS = Tile::kNumWarps * 32;
    constexpr int NUM_WARPS = Tile::kNumWarps;
    constexpr int LDS = Tile::kLdS;
    constexpr int LDP = Tile::kLdP;
    constexpr int PV_CHUNK = Bc;  // P@V output in d/Bc chunks of Bc columns
    static_assert(Bc % 8 == 0, "128-bit row accesses need Bc % 8 == 0");

    extern __shared__ __align__(128) char smem_raw[];  // WMMA needs 32-byte alignment
    half*  smem_Q = reinterpret_cast<half*>(smem_raw);
    half*  smem_K = smem_Q + Br * d;
    half*  smem_V = smem_K + Bc * d;
    float* smem_S = reinterpret_cast<float*>(smem_V + Bc * d);
    half*  smem_P = reinterpret_cast<half*>(smem_S + Br * LDS);
    float* smem_PV = smem_S;  // free once P is written

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const bool owns_row = (tid < Br) && (tid < valid_q);
    const int causal_off = seq_len_k - seq_len_q;

    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float my_O[d];
    #pragma unroll
    for (int j = 0; j < d; j++) my_O[j] = 0.0f;

    volta_load_tile<Br, d, NUM_THREADS>(Q_ptr, q_stride, smem_Q, valid_q);
    __syncthreads();

    const int kv_limit = fwd_kv_limit(is_causal, q_row_start, Br, Bc, seq_len_q, seq_len_k);
    for (int kv_blk = 0; kv_blk < kv_limit; kv_blk++) {
        const int kv_start = kv_blk * Bc;
        const int valid_kv = min(Bc, seq_len_k - kv_start);

        volta_load_tile<Bc, d, NUM_THREADS>(K_ptr + kv_start * k_stride, k_stride, smem_K, valid_kv);
        __syncthreads();

        // S = Q @ K^T (tensor cores, all warps)
        wmma_gemm_ABt<Br, Bc, d, NUM_WARPS, LDS>(smem_Q, smem_K, smem_S, warp_id);
        __syncthreads();

        // Softmax of the own row (online update), P written straight as half
        if (owns_row) {
            float sv[Bc];
            const float4* s4 = reinterpret_cast<const float4*>(smem_S + tid * LDS);
            #pragma unroll
            for (int c4 = 0; c4 < Bc / 4; c4++) {
                const float4 v = s4[c4];
                sv[4 * c4] = v.x; sv[4 * c4 + 1] = v.y; sv[4 * c4 + 2] = v.z; sv[4 * c4 + 3] = v.w;
            }
            float local_max = -INFINITY;
            #pragma unroll
            for (int c = 0; c < Bc; c++) {
                float s = sv[c] * softmax_scale;
                if ((kv_start + c) >= seq_len_k ||
                    (is_causal && (kv_start + c) > (q_row_start + tid + causal_off))) {
                    s = -INFINITY;
                }
                sv[c] = s;
                local_max = fmaxf(local_max, s);
            }

            const float new_max = fmaxf(row_max, local_max);
            // new_max == -inf: row sees no key yet (fully masked) -> avoid exp(-inf - -inf) = NaN
            const float rescale = (new_max == -INFINITY) ? 1.0f : expf(row_max - new_max);
            row_max = new_max;
            row_sum *= rescale;
            #pragma unroll
            for (int j = 0; j < d; j++) my_O[j] *= rescale;

            float local_sum = 0.0f;
            uint4* p_dst = reinterpret_cast<uint4*>(smem_P + tid * LDP);
            #pragma unroll
            for (int c8 = 0; c8 < Bc / 8; c8++) {
                half2 h[4];
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    const float s0 = sv[8 * c8 + 2 * i], s1 = sv[8 * c8 + 2 * i + 1];
                    const float p0 = (s0 == -INFINITY) ? 0.0f : expf(s0 - new_max);
                    const float p1 = (s1 == -INFINITY) ? 0.0f : expf(s1 - new_max);
                    local_sum += p0 + p1;
                    h[i] = __floats2half2_rn(p0, p1);
                }
                uint4 packed;
                memcpy(&packed, h, sizeof(packed));
                p_dst[c8] = packed;
            }
            row_sum += local_sum;
        }

        volta_load_tile<Bc, d, NUM_THREADS>(V_ptr + kv_start * v_stride, v_stride, smem_V, valid_kv);
        __syncthreads();

        // O += P @ V (tensor cores) in chunks of PV_CHUNK columns, via smem_PV
        #pragma unroll
        for (int dn = 0; dn < d; dn += PV_CHUNK) {
            wmma_gemm_AB<Br, PV_CHUNK, Bc, NUM_WARPS, LDP, LDS>(smem_P, smem_V + dn, d, smem_PV, warp_id);
            __syncthreads();
            if (owns_row) {
                const float4* pv4 = reinterpret_cast<const float4*>(smem_PV + tid * LDS);
                #pragma unroll
                for (int j4 = 0; j4 < PV_CHUNK / 4; j4++) {
                    const float4 v = pv4[j4];
                    my_O[dn + 4 * j4]     += v.x;
                    my_O[dn + 4 * j4 + 1] += v.y;
                    my_O[dn + 4 * j4 + 2] += v.z;
                    my_O[dn + 4 * j4 + 3] += v.w;
                }
            }
            // Before the next chunk overwrites smem_PV. After the last chunk the
            // barrier following the next K load orders these reads before the
            // next S store into the same buffer.
            if (dn + PV_CHUNK < d) __syncthreads();
        }
    }

    // Finalize: normalised O row, logsumexp
    if (owns_row) {
        const float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
        half2* o_dst2 = reinterpret_cast<half2*>(O_ptr + tid * o_stride);
        #pragma unroll
        for (int j = 0; j < d / 2; j++) {
            o_dst2[j] = __floats2half2_rn(my_O[2 * j] * inv_sum, my_O[2 * j + 1] * inv_sum);
        }
    }
    return row_max + logf(fmaxf(row_sum, 1e-20f));
}

// ---------------------------------------------------------------------------
// Forward kernel — WMMA (Volta / Turing), batched [B, H, N, d]
// ---------------------------------------------------------------------------
template <typename Tile>
__global__ void __launch_bounds__(Tile::kNumWarps * 32)
flash_attn_fwd_volta_kernel(FlashAttnParams params) {
    constexpr int Br = Tile::Br;
    const int bh_idx = blockIdx.y;
    const int b_idx = bh_idx / params.num_heads;
    const int h_idx = bh_idx % params.num_heads;
    const int kv_h_idx = h_idx / (params.num_heads / params.num_heads_k);
    const int q_row_start = blockIdx.x * Br;
    if (q_row_start >= params.seq_len_q) return;
    const int valid_q = min(Br, params.seq_len_q - q_row_start);

    const float lse = volta_fwd_block<Tile>(
        params.Q + b_idx * params.q_batch_stride + h_idx * params.q_head_stride
                 + q_row_start * params.q_seq_stride, params.q_seq_stride,
        params.K + b_idx * params.k_batch_stride + kv_h_idx * params.k_head_stride, params.k_seq_stride,
        params.V + b_idx * params.v_batch_stride + kv_h_idx * params.v_head_stride, params.v_seq_stride,
        params.O + b_idx * params.o_batch_stride + h_idx * params.o_head_stride
                 + q_row_start * params.o_seq_stride, params.o_seq_stride,
        q_row_start, valid_q, params.seq_len_q, params.seq_len_k,
        params.softmax_scale, params.is_causal);

    const int tid = threadIdx.x;
    if (tid < valid_q) {
        params.L[b_idx * params.l_batch_stride + h_idx * params.l_head_stride + q_row_start + tid] = lse;
    }
}

// ---------------------------------------------------------------------------
// Forward kernel — WMMA (Volta / Turing), varlen (packed [total, H, d])
// ---------------------------------------------------------------------------
template <typename Tile>
__global__ void __launch_bounds__(Tile::kNumWarps * 32)
flash_attn_fwd_volta_varlen_kernel(FlashAttnVarlenParams params) {
    constexpr int Br = Tile::Br;
    const int h_idx = blockIdx.y;
    const int block_x = blockIdx.x;

    // Binary search: find which sequence this block belongs to
    int lo = 0, hi = params.num_seqs;
    while (lo < hi) {
        const int mid = (lo + hi) / 2;
        if (params.block_offsets[mid + 1] <= block_x) lo = mid + 1;
        else hi = mid;
    }
    const int seq_idx = lo;

    const int local_block = block_x - params.block_offsets[seq_idx];
    const int q_start_tok = params.cu_seqlens_q[seq_idx];
    const int k_start_tok = params.cu_seqlens_k[seq_idx];
    const int seq_len_q = params.cu_seqlens_q[seq_idx + 1] - q_start_tok;
    const int seq_len_k = params.cu_seqlens_k[seq_idx + 1] - k_start_tok;
    const int q_row_start = local_block * Br;
    if (q_row_start >= seq_len_q) return;
    const int valid_q = min(Br, seq_len_q - q_row_start);
    const int kv_h_idx = h_idx / (params.num_heads / params.num_heads_k);

    const float lse = volta_fwd_block<Tile>(
        params.Q + (q_start_tok + q_row_start) * params.q_token_stride + h_idx * params.q_head_stride,
        params.q_token_stride,
        params.K + k_start_tok * params.k_token_stride + kv_h_idx * params.k_head_stride,
        params.k_token_stride,
        params.V + k_start_tok * params.v_token_stride + kv_h_idx * params.v_head_stride,
        params.v_token_stride,
        params.O + (q_start_tok + q_row_start) * params.o_token_stride + h_idx * params.o_head_stride,
        params.o_token_stride,
        q_row_start, valid_q, seq_len_q, seq_len_k,
        params.softmax_scale, params.is_causal);

    const int tid = threadIdx.x;
    if (tid < valid_q) {
        // L is [total_q, H_q]: stride between tokens is num_heads
        params.L[(q_start_tok + q_row_start + tid) * params.num_heads + h_idx] = lse;
    }
}

#else  // __CUDA_ARCH__ < 700

// Stub kernels for SM < 7.0 — never launched at runtime (dispatch in .cpp).
template <typename Tile>
__global__ void __launch_bounds__(Tile::kNumWarps * 32)
flash_attn_fwd_volta_kernel(FlashAttnParams params) {}

template <typename Tile>
__global__ void __launch_bounds__(Tile::kNumWarps * 32)
flash_attn_fwd_volta_varlen_kernel(FlashAttnVarlenParams params) {}

#endif  // __CUDA_ARCH__

// ---------------------------------------------------------------------------
// Host launchers (always compiled — runtime dispatch ensures correctness)
//
// Tile per head_dim, chosen by measurement (docs/PLAN.md P2.3): Bc=16 tiles
// beat the previous Bc=32/64 ones on V100 and T4 (forward, causal, B=2 H=16
// N=4096; V100 / T4 in ms: d=32 8.95 vs 13.3 / 22.7 vs 38.3; d=64 13.4 vs 21.0
// / 37.4 vs 53.5; d=96 15.3 vs 23.4 / 43.1 vs 55.2; d=128 29.0 vs 33.0 / 64.6).
// The dispatch checks the tile against the device's shared memory (clear
// error) and gives the varlen host code its Br, so launch and block offsets
// always agree.
// ---------------------------------------------------------------------------
template <typename F>
static void dispatch_volta_fwd_tile(int head_dim, F&& f) {
    const int limit = max_dynamic_smem_per_block();
    bool ok = false;
    switch (head_dim) {
        case 32:  ok = with_first_fitting_tile<TileVolta_d32>(limit, f); break;
        case 64:  ok = with_first_fitting_tile<TileVolta_d64>(limit, f); break;
        case 96:  ok = with_first_fitting_tile<TileVolta_d96>(limit, f); break;
        case 128: ok = with_first_fitting_tile<TileVolta_d128>(limit, f); break;
        case 256: ok = with_first_fitting_tile<TileVolta_d256>(limit, f); break;
        default:  throw std::runtime_error("Unsupported head_dim for the WMMA forward");
    }
    if (!ok) throw_no_tile("WMMA forward", head_dim, limit);
}

template <typename Tile>
static void launch_volta_fwd(FlashAttnParams& params, cudaStream_t stream) {
    constexpr int smem = Tile::smem_bytes();
    dim3 grid(cdiv(params.seq_len_q, Tile::Br), params.batch_size * params.num_heads);
    dim3 block(Tile::kNumWarps * 32);
    FLASH_ATTN_CHECK_CUDA(cudaFuncSetAttribute(
        flash_attn_fwd_volta_kernel<Tile>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    flash_attn_fwd_volta_kernel<Tile><<<grid, block, smem, stream>>>(params);
}

template <typename Tile>
static void launch_volta_fwd_varlen(FlashAttnVarlenParams& params, int total_q_blocks,
                                    cudaStream_t stream) {
    constexpr int smem = Tile::smem_bytes();
    dim3 grid(total_q_blocks, params.num_heads);
    dim3 block(Tile::kNumWarps * 32);
    FLASH_ATTN_CHECK_CUDA(cudaFuncSetAttribute(
        flash_attn_fwd_volta_varlen_kernel<Tile>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    flash_attn_fwd_volta_varlen_kernel<Tile><<<grid, block, smem, stream>>>(params);
}

void flash_attn_fwd_volta(FlashAttnParams& params, cudaStream_t stream) {
    dispatch_volta_fwd_tile(params.head_dim, [&](auto tile) {
        launch_volta_fwd<decltype(tile)>(params, stream);
    });
}

void flash_attn_fwd_volta_varlen(FlashAttnVarlenParams& params,
                                 int total_q_blocks, cudaStream_t stream) {
    dispatch_volta_fwd_tile(params.head_dim, [&](auto tile) {
        launch_volta_fwd_varlen<decltype(tile)>(params, total_q_blocks, stream);
    });
}

// Q rows per block of the forward tile selected for head_dim on this device.
int flash_attn_fwd_volta_block_rows(int head_dim) {
    int rows = 0;
    dispatch_volta_fwd_tile(head_dim, [&](auto tile) { rows = decltype(tile)::Br; });
    return rows;
}

}  // namespace flash_attn_legacy
