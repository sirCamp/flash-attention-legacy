// ============================================================================
// Flash Attention v2 — Forward Kernel for Volta (SM 7.0) and Turing (SM 7.5)
// Uses WMMA tensor cores for both Q@K^T and P@V matmuls (16×16×16 FP16)
//
// Key design (FA2-style, docs/analysis/kernels/p2_8_design.md):
//   - each warp owns a band of 16 query rows: S = Q @ K^T, online softmax and
//     O += P @ V without block-wide barriers; O in WMMA accumulator fragments
//   - masking only on the K/V blocks that need it (tail, causal diagonal);
//     causal blocks a band cannot see are skipped by that warp
//   - one block body (volta_fwd_block) shared by the batched and varlen kernels
//
// Architecture guard:
//   WMMA requires SM >= 7.0. This file is compiled for SM 6.x too
//   (gencode flags), but the kernel body is #ifdef'd to a stub for SM < 7.0.
//   Runtime dispatch in flash_attn_legacy.cpp ensures the Volta kernel is
//   never launched on Pascal.
// ============================================================================

#include "flash_attn_common.h"
#include "flash_attn_wmma.h"

#include <cstring>

#include <string>

namespace flash_attn_legacy {

// Guard all device code that uses WMMA intrinsics.
// Host compilation: __CUDA_ARCH__ is undefined → condition true → real code.
// SM 6.x device compilation: __CUDA_ARCH__ < 700 → stub.
// SM 7.0 device compilation: __CUDA_ARCH__ >= 700 → real code.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700

// ---------------------------------------------------------------------------
// One Q block of the WMMA forward, shared by the batched and varlen kernels
// (they only differ in how they locate Q/K/V/O/L). FA2-style: warp w owns the
// band of query rows [16w, 16w+16) and runs S = Q K^T, the online softmax and
// O += P V on its own, with O in WMMA accumulator fragments. The only block-wide
// barriers are around the cooperative K/V tile loads.
//
// Per warp, S goes through a [16, LDS] float staging tile (the WMMA accumulator
// layout is opaque): two lanes per row (lane = 2*row + h) read interleaved float4
// chunks, and P is written back in place as half (row stride 2*LDS) for the
// P @ V fragments. To rescale O rows without assuming the accumulator layout,
// each lane learns once which row each of its fragment elements belongs to
// (store_matrix_sync of a fragment holding its own element ids).
//
// Softmax in the log2 domain (FA2): p = exp2(s * scale * log2e - m). Rows that
// see no key keep m = -inf and give O = 0, L = -inf, as the backward expects.
// L is the natural-log logsumexp of the scaled scores, written to
// L_ptr[row * l_stride].
// ---------------------------------------------------------------------------
template <typename Tile>
__device__ __forceinline__ void volta_fwd_block(
    const half* __restrict__ Q_ptr, int q_stride,
    const half* __restrict__ K_ptr, int k_stride,
    const half* __restrict__ V_ptr, int v_stride,
    half* __restrict__ O_ptr, int o_stride,
    float* __restrict__ L_ptr, int l_stride,
    int q_row_start, int valid_q, int seq_len_q, int seq_len_k,
    float softmax_scale, bool is_causal
) {
    constexpr int Br = Tile::Br;
    constexpr int Bc = Tile::Bc;
    constexpr int d  = Tile::d;
    constexpr int NUM_THREADS = Tile::kNumWarps * 32;
    constexpr int LDH = Tile::kLdH;
    constexpr int LDS = Tile::kLdS;
    constexpr int T = 16;
    constexpr int S_TILES = Bc / T;   // S fragments per warp
    constexpr int O_TILES = d / T;    // O fragments per warp
    constexpr int CHUNKS = Bc / 8;    // float4 chunks of S per lane (two lanes per row)
    using Acc = wmma::fragment<wmma::accumulator, T, T, T, float>;
    constexpr int NE = Acc::num_elements;
    static_assert(NE * 32 == T * T, "unexpected WMMA accumulator size");

    extern __shared__ __align__(128) char smem_raw[];  // WMMA needs 32-byte alignment
    half*  smem_Q = reinterpret_cast<half*>(smem_raw);
    half*  smem_K = smem_Q + Br * LDH;
    half*  smem_V = smem_K + Bc * LDH;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    float* stage = reinterpret_cast<float*>(smem_V + Bc * LDH) + warp_id * (Tile::kStageFloats + 16);
    float* row_scale = stage + Tile::kStageFloats;  // 16 per-row factors
    half*  stage_p = reinterpret_cast<half*>(stage);

    const int r = lane >> 1;            // softmax row of this lane within the band
    const int h = lane & 1;             // which interleaved half of the row
    const int band_row0 = warp_id * T;  // first row of the band within the Q block
    const bool band_active = band_row0 < valid_q;
    const int causal_off = seq_len_k - seq_len_q;
    const int q_row = q_row_start + band_row0 + r;  // sequence row of this lane's softmax row
    const float scale_log2 = softmax_scale * 1.4426950408889634f;

    // Row of each accumulator element of this lane (identical for every fragment).
    int row_of[NE];
    {
        Acc probe;
        #pragma unroll
        for (int i = 0; i < NE; i++) probe.x[i] = static_cast<float>(lane * NE + i);
        wmma::store_matrix_sync(stage, probe, T, wmma::mem_row_major);
        __syncwarp();
        unsigned char* table = reinterpret_cast<unsigned char*>(stage + T * T);
        #pragma unroll
        for (int k = 0; k < NE; k++) {
            const int e = lane + 32 * k;
            table[static_cast<int>(stage[e])] = static_cast<unsigned char>(e / T);
        }
        __syncwarp();
        #pragma unroll
        for (int i = 0; i < NE; i++) row_of[i] = table[lane * NE + i];
        __syncwarp();
    }

    Acc O[O_TILES];
    #pragma unroll
    for (int t = 0; t < O_TILES; t++) wmma::fill_fragment(O[t], 0.0f);
    float m = -INFINITY;   // running max of this row, log2 domain
    float row_sum = 0.0f;  // partial sum of this lane's columns

    load_tile_g2s<Br, d, NUM_THREADS, LDH>(Q_ptr, q_stride, smem_Q, valid_q);

    // One K/V block for this warp's band; kMask: block needs tail/causal masking.
    auto band_step = [&](auto mask_tag, int kv_start) {
        constexpr bool kMask = decltype(mask_tag)::value;

        // S = Q K^T for the band
        Acc S[S_TILES];
        #pragma unroll
        for (int tn = 0; tn < S_TILES; tn++) wmma::fill_fragment(S[tn], 0.0f);
        #pragma unroll
        for (int k = 0; k < d; k += T) {
            wmma::fragment<wmma::matrix_a, T, T, T, half, wmma::row_major> a;
            wmma::load_matrix_sync(a, smem_Q + band_row0 * LDH + k, LDH);
            #pragma unroll
            for (int tn = 0; tn < S_TILES; tn++) {
                wmma::fragment<wmma::matrix_b, T, T, T, half, wmma::col_major> b;
                wmma::load_matrix_sync(b, smem_K + tn * T * LDH + k, LDH);
                wmma::mma_sync(S[tn], a, b, S[tn]);
            }
        }
        #pragma unroll
        for (int tn = 0; tn < S_TILES; tn++) {
            wmma::store_matrix_sync(stage + tn * T, S[tn], LDS, wmma::mem_row_major);
        }
        __syncwarp();

        // Online softmax: this lane's half of row r
        float sv[CHUNKS * 4];
        float local_max = -INFINITY;
        #pragma unroll
        for (int c = 0; c < CHUNKS; c++) {
            const int col0 = 4 * (2 * c + h);
            const float4 v = *reinterpret_cast<const float4*>(stage + r * LDS + col0);
            const float e[4] = {v.x, v.y, v.z, v.w};
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                float s = e[i] * scale_log2;
                if constexpr (kMask) {
                    const int key = kv_start + col0 + i;
                    if (key >= seq_len_k || (is_causal && key > q_row + causal_off)) s = -INFINITY;
                }
                sv[4 * c + i] = s;
                local_max = fmaxf(local_max, s);
            }
        }
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, 1));
        const float m_new = fmaxf(m, local_max);
        const float m_use = (m_new == -INFINITY) ? 0.0f : m_new;  // no (-inf) - (-inf)
        const float alpha = exp2f(m - m_use);
        m = m_new;
        float local_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < CHUNKS * 4; i++) {
            sv[i] = exp2f(sv[i] - m_use);
            local_sum += sv[i];
        }
        row_sum = row_sum * alpha + local_sum;
        if (h == 0) row_scale[r] = alpha;
        __syncwarp();  // every lane has read S before P overwrites it

        #pragma unroll
        for (int c = 0; c < CHUNKS; c++) {
            const half2 p01 = __floats2half2_rn(sv[4 * c], sv[4 * c + 1]);
            const half2 p23 = __floats2half2_rn(sv[4 * c + 2], sv[4 * c + 3]);
            uint2 packed;
            memcpy(&packed.x, &p01, 4);
            memcpy(&packed.y, &p23, 4);
            *reinterpret_cast<uint2*>(stage_p + r * 2 * LDS + 4 * (2 * c + h)) = packed;
        }
        __syncwarp();

        // O = O * alpha(row) + P V
        #pragma unroll
        for (int i = 0; i < NE; i++) {
            const float alpha_row = row_scale[row_of[i]];
            #pragma unroll
            for (int t = 0; t < O_TILES; t++) O[t].x[i] *= alpha_row;
        }
        #pragma unroll
        for (int k = 0; k < Bc; k += T) {
            wmma::fragment<wmma::matrix_a, T, T, T, half, wmma::row_major> a;
            wmma::load_matrix_sync(a, stage_p + k, 2 * LDS);
            #pragma unroll
            for (int t = 0; t < O_TILES; t++) {
                wmma::fragment<wmma::matrix_b, T, T, T, half, wmma::row_major> b;
                wmma::load_matrix_sync(b, smem_V + k * LDH + t * T, LDH);
                wmma::mma_sync(O[t], a, b, O[t]);
            }
        }
        __syncwarp();  // P and row_scale reads done before the next block reuses them
    };

    const int kv_limit = fwd_kv_limit(is_causal, q_row_start, Br, Bc, seq_len_q, seq_len_k);
    // Per band: every row of the band is at or below its first row, so blocks
    // before this limit need no masking for any of them.
    const int nomask_limit = fwd_kv_nomask_limit(is_causal, q_row_start + band_row0, Bc,
                                                  seq_len_q, seq_len_k);
    // Last key any row of this band can see (causal): later blocks are skipped by the warp.
    const int band_last_key = q_row_start + band_row0 + T - 1 + causal_off;
    for (int kv_blk = 0; kv_blk < kv_limit; kv_blk++) {
        const int kv_start = kv_blk * Bc;
        const int valid_kv = min(Bc, seq_len_k - kv_start);
        load_tile_g2s<Bc, d, NUM_THREADS, LDH>(K_ptr + kv_start * k_stride, k_stride, smem_K, valid_kv);
        load_tile_g2s<Bc, d, NUM_THREADS, LDH>(V_ptr + kv_start * v_stride, v_stride, smem_V, valid_kv);
        __syncthreads();
        if (band_active && !(is_causal && kv_start > band_last_key)) {  // warp-uniform
            if (kv_blk < nomask_limit) band_step(std::false_type{}, kv_start);
            else                       band_step(std::true_type{}, kv_start);
        }
        __syncthreads();  // all warps done with K/V before the next load
    }
    if (!band_active) return;

    // Finalize: O / row_sum, logsumexp, O written through the staging tile
    row_sum += __shfl_xor_sync(0xffffffffu, row_sum, 1);
    const int band_valid = min(T, valid_q - band_row0);
    if (h == 0) {
        row_scale[r] = (row_sum > 0.0f) ? 1.0f / row_sum : 0.0f;
        if (r < band_valid) {
            L_ptr[(band_row0 + r) * l_stride] = m * 0.6931471805599453f + logf(fmaxf(row_sum, 1e-20f));
        }
    }
    __syncwarp();
    #pragma unroll
    for (int i = 0; i < NE; i++) {
        const float inv_row = row_scale[row_of[i]];
        #pragma unroll
        for (int t = 0; t < O_TILES; t++) O[t].x[i] *= inv_row;
    }
    constexpr int GROUP = (S_TILES < O_TILES) ? S_TILES : O_TILES;  // O tiles per staging pass
    static_assert(O_TILES % GROUP == 0, "O tiles must split into staging passes");
    constexpr int PASS_CHUNKS = T * GROUP * T / 8;  // 8-half output chunks per pass
    #pragma unroll
    for (int t0 = 0; t0 < O_TILES; t0 += GROUP) {
        #pragma unroll
        for (int g = 0; g < GROUP; g++) {
            wmma::store_matrix_sync(stage + g * T, O[t0 + g], LDS, wmma::mem_row_major);
        }
        __syncwarp();
        #pragma unroll
        for (int c = lane; c < PASS_CHUNKS; c += 32) {
            const int row = c / (GROUP * 2), col = (c % (GROUP * 2)) * 8;
            if (row < band_valid) {
                const float4 lo = *reinterpret_cast<const float4*>(stage + row * LDS + col);
                const float4 hi = *reinterpret_cast<const float4*>(stage + row * LDS + col + 4);
                const half2 o[4] = {__floats2half2_rn(lo.x, lo.y), __floats2half2_rn(lo.z, lo.w),
                                    __floats2half2_rn(hi.x, hi.y), __floats2half2_rn(hi.z, hi.w)};
                uint4 packed;
                memcpy(&packed, o, sizeof(packed));
                *reinterpret_cast<uint4*>(O_ptr + (band_row0 + row) * o_stride + t0 * T + col) = packed;
            }
        }
        __syncwarp();
    }
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

    volta_fwd_block<Tile>(
        params.Q + b_idx * params.q_batch_stride + h_idx * params.q_head_stride
                 + q_row_start * params.q_seq_stride, params.q_seq_stride,
        params.K + b_idx * params.k_batch_stride + kv_h_idx * params.k_head_stride, params.k_seq_stride,
        params.V + b_idx * params.v_batch_stride + kv_h_idx * params.v_head_stride, params.v_seq_stride,
        params.O + b_idx * params.o_batch_stride + h_idx * params.o_head_stride
                 + q_row_start * params.o_seq_stride, params.o_seq_stride,
        params.L + b_idx * params.l_batch_stride + h_idx * params.l_head_stride + q_row_start, 1,
        q_row_start, valid_q, params.seq_len_q, params.seq_len_k,
        params.softmax_scale, params.is_causal);
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

    volta_fwd_block<Tile>(
        params.Q + (q_start_tok + q_row_start) * params.q_token_stride + h_idx * params.q_head_stride,
        params.q_token_stride,
        params.K + k_start_tok * params.k_token_stride + kv_h_idx * params.k_head_stride,
        params.k_token_stride,
        params.V + k_start_tok * params.v_token_stride + kv_h_idx * params.v_head_stride,
        params.v_token_stride,
        params.O + (q_start_tok + q_row_start) * params.o_token_stride + h_idx * params.o_head_stride,
        params.o_token_stride,
        // L is [total_q, H_q]: stride between tokens is num_heads
        params.L + (q_start_tok + q_row_start) * params.num_heads + h_idx, params.num_heads,
        q_row_start, valid_q, seq_len_q, seq_len_k,
        params.softmax_scale, params.is_causal);
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
// Tile per head_dim: see the TileVolta_* aliases (flash_attn_common.h). The
// dispatch checks the tile against the device's shared memory (clear error)
// and gives the varlen host code its Br, so launch and block offsets always
// agree.
// ---------------------------------------------------------------------------
template <typename F>
static void dispatch_volta_fwd_tile(int head_dim, F&& f) {
    const int limit = max_dynamic_smem_per_block();
    bool ok = false;
    switch (head_dim) {
        case 32:  ok = with_first_fitting_tile<TileVolta_d32>(limit, f); break;
        case 64:  ok = with_first_fitting_tile<TileVolta_d64>(limit, f); break;
        case 96:  ok = with_first_fitting_tile<TileVolta_d96>(limit, f); break;
        case 128: ok = with_first_fitting_tile<TileVolta_d128, TileVolta_d128_64k>(limit, f); break;
        case 256: ok = with_first_fitting_tile<TileVolta_d256, TileVolta_d256_64k>(limit, f); break;
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
