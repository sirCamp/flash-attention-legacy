#pragma once

// ============================================================================
// FA2-style forward, shared by every kernel family (docs/PLAN.md P2.1 / P2.8).
//
// A Q block of Br rows is split in bands of 16 rows, one per warp. The parts
// that do not depend on how S = Q K^T and O += P V are computed live here, once:
//   - FwdBlockArgs: pointers and lengths of one Q block, from the batched or
//     the varlen grid (fwd_batched_kernel / fwd_varlen_kernel)
//   - fwd_key_masked: tail and bottom-right causal masking
//   - BandSoftmax: online softmax in the log2 domain for the rows a thread holds,
//     reduced over the lanes that share a row
//   - fwd_band_loop: the K/V loop (causal limit, blocks without masking, blocks
//     a band cannot see, K/V prefetch, the two block-wide barriers)
//   - fwd_band_run: fwd_band_loop with the Q/K/V tile copies (half or fp32
//     shared memory, optional K/V register prefetch); launch_fwd(_varlen)
// A family provides a Body with `template <class Tile> static void run(const
// FwdBlockArgs&)` that calls fwd_band_run with its own S / P / O code (WMMA:
// flash_attn_fwd_volta.cu, CUDA cores: flash_attn_fwd_pascal.cu) and a tile
// dispatcher per head_dim.
// ============================================================================

#include "flash_attn_common.h"

namespace flash_attn_legacy {

#ifdef __CUDACC__

constexpr int kBandRows = 16;  // query rows per warp

// One Q block: its rows start at Q / O / L (row strides q_stride / o_stride /
// l_stride); K and V start at key 0 of the sequence.
struct FwdBlockArgs {
    const half* __restrict__ Q; int q_stride;
    const half* __restrict__ K; int k_stride;
    const half* __restrict__ V; int v_stride;
    half* __restrict__ O; int o_stride;
    float* __restrict__ L; int l_stride;
    int q_row_start, valid_q, seq_len_q, seq_len_k;
    float softmax_scale;
    bool is_causal;
};

// Key `key` is not visible to sequence row `q_row`: past the end, or (causal,
// bottom-right aligned) after row + seq_len_k - seq_len_q.
__device__ __forceinline__ bool fwd_key_masked(const FwdBlockArgs& a, int q_row, int key) {
    return key >= a.seq_len_k || (a.is_causal && key > q_row + a.seq_len_k - a.seq_len_q);
}

// Online softmax state of the ROWS rows a thread holds; each row is spread over
// ROW_LANES lanes (lane ids differing only in the low log2(ROW_LANES) bits).
// Rows that see no key keep m = -inf and end with O = 0, L = -inf.
template <int ROWS, int ROW_LANES>
struct BandSoftmax {
    static_assert((ROW_LANES & (ROW_LANES - 1)) == 0 && ROW_LANES <= 32, "ROW_LANES must be a power of 2");
    float m[ROWS];    // running max, log2 domain (scores * scale * log2e)
    float sum[ROWS];  // partial sum of this lane's columns

    __device__ __forceinline__ void init() {
        #pragma unroll
        for (int r = 0; r < ROWS; r++) { m[r] = -INFINITY; sum[r] = 0.0f; }
    }

    // s: this lane's scores of the block, already scaled to the log2 domain and
    // -inf where masked. Turns them into p = exp2(s - m) in place and returns the
    // factor alpha[r] the row's O accumulator must be multiplied by.
    template <int COLS>
    __device__ __forceinline__ void update(float (&s)[ROWS][COLS], float (&alpha)[ROWS]) {
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            float block_max = -INFINITY;
            #pragma unroll
            for (int c = 0; c < COLS; c++) block_max = fmaxf(block_max, s[r][c]);
            #pragma unroll
            for (int o = 1; o < ROW_LANES; o <<= 1) {
                block_max = fmaxf(block_max, __shfl_xor_sync(0xffffffffu, block_max, o));
            }
            const float m_new = fmaxf(m[r], block_max);
            const float m_use = (m_new == -INFINITY) ? 0.0f : m_new;  // no (-inf) - (-inf)
            alpha[r] = exp2f(m[r] - m_use);
            m[r] = m_new;
            float block_sum = 0.0f;
            #pragma unroll
            for (int c = 0; c < COLS; c++) {
                s[r][c] = exp2f(s[r][c] - m_use);
                block_sum += s[r][c];
            }
            sum[r] = sum[r] * alpha[r] + block_sum;
        }
    }

    // inv[r]: factor for the final O row (0 for rows that saw no key);
    // lse[r]: natural-log logsumexp of the scaled scores (-inf for those rows).
    __device__ __forceinline__ void finalize(float (&inv)[ROWS], float (&lse)[ROWS]) {
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            float total = sum[r];
            #pragma unroll
            for (int o = 1; o < ROW_LANES; o <<= 1) total += __shfl_xor_sync(0xffffffffu, total, o);
            inv[r] = (total > 0.0f) ? 1.0f / total : 0.0f;
            lse[r] = m[r] * kLn2 + logf(fmaxf(total, 1e-20f));
        }
    }
};

// The K/V loop of one Q block. The caller supplies:
//   load_q()          global -> shared Q (published by the first barrier)
//   fetch(kv_blk)     start loading K/V block kv_blk into registers, or just
//                     record which block the next commit() loads
//   commit()          leave the fetched block fully in shared memory (the
//                     barrier that follows publishes it)
//   step(mask, kv_start)  this warp's band on the block in shared memory;
//                     mask is std::true_type when tail/causal masking is needed
// fetch(j + 1) is issued right after block j is committed, so its global loads
// overlap the computation of block j. Bands skip blocks they cannot see
// (causal) and only mask the blocks that need it. Inactive bands (rows past
// valid_q) still take part in every barrier.
template <int Br, int Bc, typename LoadQ, typename Fetch, typename Commit, typename Step>
__device__ __forceinline__ void fwd_band_loop(const FwdBlockArgs& a, int band_row0, bool band_active,
                                              LoadQ&& load_q, Fetch&& fetch, Commit&& commit, Step&& step) {
    const int kv_limit = fwd_kv_limit(a.is_causal, a.q_row_start, Br, Bc, a.seq_len_q, a.seq_len_k);
    // Every row of the band is at or below its first row: blocks before this
    // limit need no masking for any of them.
    const int nomask_limit = fwd_kv_nomask_limit(a.is_causal, a.q_row_start + band_row0, Bc,
                                                 a.seq_len_q, a.seq_len_k);
    const int band_last_key = a.q_row_start + band_row0 + kBandRows - 1 + (a.seq_len_k - a.seq_len_q);
    if (kv_limit > 0) fetch(0);  // in flight while Q is loaded
    load_q();
    for (int kv_blk = 0; kv_blk < kv_limit; kv_blk++) {
        const int kv_start = kv_blk * Bc;
        commit();
        __syncthreads();
        if (kv_blk + 1 < kv_limit) fetch(kv_blk + 1);
        if (band_active && !(a.is_causal && kv_start > band_last_key)) {  // warp-uniform
            if (kv_blk < nomask_limit) step(std::false_type{}, kv_start);
            else                       step(std::true_type{}, kv_start);
        }
        __syncthreads();  // all warps done with K/V before the next commit
    }
}

// fwd_band_loop with the Q/K/V tile copies of every family: rows of LD elements
// of DstT in shared memory (half for the tensor cores, float for the CUDA
// cores: converted once, when stored). With Tile::kPrefetch the next K/V block
// is loaded into registers while the current one is computed; otherwise each
// block is copied straight to shared memory at its turn.
template <typename Tile, int LD, typename DstT, typename Step>
__device__ __forceinline__ void fwd_band_run(const FwdBlockArgs& a, int band_row0, bool band_active,
                                             DstT* smem_Q, DstT* smem_K, DstT* smem_V, Step&& step) {
    constexpr int Br = Tile::Br, Bc = Tile::Bc, d = Tile::d, NT = Tile::kNumWarps * 32;
    auto load_q = [&]() { load_tile_g2s<Br, d, NT, LD>(a.Q, a.q_stride, smem_Q, a.valid_q); };
    if constexpr (Tile::kPrefetch) {
        TileRegs<Bc, d, NT> next_K, next_V;
        fwd_band_loop<Br, Bc>(a, band_row0, band_active, load_q,
            [&](int kv_blk) {
                const int kv_start = kv_blk * Bc;
                const int valid_kv = min(Bc, a.seq_len_k - kv_start);
                load_tile_g2r(a.K + kv_start * a.k_stride, a.k_stride, next_K, valid_kv);
                load_tile_g2r(a.V + kv_start * a.v_stride, a.v_stride, next_V, valid_kv);
            },
            [&]() {
                store_tile_r2s<Bc, d, NT, LD>(next_K, smem_K);
                store_tile_r2s<Bc, d, NT, LD>(next_V, smem_V);
            },
            step);
    } else {
        int pending = 0;  // block the next commit() loads
        fwd_band_loop<Br, Bc>(a, band_row0, band_active, load_q,
            [&](int kv_blk) { pending = kv_blk; },
            [&]() {
                const int kv_start = pending * Bc;
                const int valid_kv = min(Bc, a.seq_len_k - kv_start);
                load_tile_g2s<Bc, d, NT, LD>(a.K + kv_start * a.k_stride, a.k_stride, smem_K, valid_kv);
                load_tile_g2s<Bc, d, NT, LD>(a.V + kv_start * a.v_stride, a.v_stride, smem_V, valid_kv);
            },
            step);
    }
}

// ---------------------------------------------------------------------------
// Kernels: locate the Q block of this CUDA block and run the family's Body.
// ---------------------------------------------------------------------------
template <typename Tile, typename Body>
__global__ void __launch_bounds__(Tile::kNumWarps * 32)
fwd_batched_kernel(FlashAttnParams p) {
    const int bh_idx = blockIdx.y;
    const int b_idx = bh_idx / p.num_heads;
    const int h_idx = bh_idx % p.num_heads;
    const int kv_h_idx = h_idx / (p.num_heads / p.num_heads_k);
    const int q_row_start = blockIdx.x * Tile::Br;
    if (q_row_start >= p.seq_len_q) return;

    FwdBlockArgs a;
    a.Q = p.Q + b_idx * p.q_batch_stride + h_idx * p.q_head_stride + q_row_start * p.q_seq_stride;
    a.q_stride = p.q_seq_stride;
    a.K = p.K + b_idx * p.k_batch_stride + kv_h_idx * p.k_head_stride;
    a.k_stride = p.k_seq_stride;
    a.V = p.V + b_idx * p.v_batch_stride + kv_h_idx * p.v_head_stride;
    a.v_stride = p.v_seq_stride;
    a.O = p.O + b_idx * p.o_batch_stride + h_idx * p.o_head_stride + q_row_start * p.o_seq_stride;
    a.o_stride = p.o_seq_stride;
    a.L = p.L + b_idx * p.l_batch_stride + h_idx * p.l_head_stride + q_row_start;
    a.l_stride = 1;
    a.q_row_start = q_row_start;
    a.valid_q = min(Tile::Br, p.seq_len_q - q_row_start);
    a.seq_len_q = p.seq_len_q;
    a.seq_len_k = p.seq_len_k;
    a.softmax_scale = p.softmax_scale;
    a.is_causal = p.is_causal;
    Body::template run<Tile>(a);
}

template <typename Tile, typename Body>
__global__ void __launch_bounds__(Tile::kNumWarps * 32)
fwd_varlen_kernel(FlashAttnVarlenParams p) {
    const int h_idx = blockIdx.y;
    const int block_x = blockIdx.x;
    const int seq_idx = varlen_seq_of_block(p.block_offsets, p.num_seqs, block_x);
    if (seq_idx >= p.num_seqs) return;
    const int q_start_tok = p.cu_seqlens_q[seq_idx];
    const int k_start_tok = p.cu_seqlens_k[seq_idx];
    const int seq_len_q = p.cu_seqlens_q[seq_idx + 1] - q_start_tok;
    const int q_row_start = (block_x - p.block_offsets[seq_idx]) * Tile::Br;
    if (q_row_start >= seq_len_q) return;
    const int kv_h_idx = h_idx / (p.num_heads / p.num_heads_k);

    FwdBlockArgs a;
    a.Q = p.Q + (q_start_tok + q_row_start) * p.q_token_stride + h_idx * p.q_head_stride;
    a.q_stride = p.q_token_stride;
    a.K = p.K + k_start_tok * p.k_token_stride + kv_h_idx * p.k_head_stride;
    a.k_stride = p.k_token_stride;
    a.V = p.V + k_start_tok * p.v_token_stride + kv_h_idx * p.v_head_stride;
    a.v_stride = p.v_token_stride;
    a.O = p.O + (q_start_tok + q_row_start) * p.o_token_stride + h_idx * p.o_head_stride;
    a.o_stride = p.o_token_stride;
    a.L = p.L + (q_start_tok + q_row_start) * p.num_heads + h_idx;  // L is [total_q, H_q]
    a.l_stride = p.num_heads;
    a.q_row_start = q_row_start;
    a.valid_q = min(Tile::Br, seq_len_q - q_row_start);
    a.seq_len_q = seq_len_q;
    a.seq_len_k = p.cu_seqlens_k[seq_idx + 1] - k_start_tok;
    a.softmax_scale = p.softmax_scale;
    a.is_causal = p.is_causal;
    Body::template run<Tile>(a);
}

// ---------------------------------------------------------------------------
// Host launchers of a family's Body with the tile chosen by its dispatcher.
// ---------------------------------------------------------------------------
template <typename Tile, typename Body>
void launch_fwd(FlashAttnParams& params, cudaStream_t stream) {
    constexpr int smem = Tile::smem_bytes();
    dim3 grid(cdiv(params.seq_len_q, Tile::Br), params.batch_size * params.num_heads);
    FLASH_ATTN_CHECK_CUDA(cudaFuncSetAttribute(
        fwd_batched_kernel<Tile, Body>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    fwd_batched_kernel<Tile, Body><<<grid, Tile::kNumWarps * 32, smem, stream>>>(params);
}

template <typename Tile, typename Body>
void launch_fwd_varlen(FlashAttnVarlenParams& params, int total_q_blocks, cudaStream_t stream) {
    constexpr int smem = Tile::smem_bytes();
    dim3 grid(total_q_blocks, params.num_heads);
    FLASH_ATTN_CHECK_CUDA(cudaFuncSetAttribute(
        fwd_varlen_kernel<Tile, Body>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    fwd_varlen_kernel<Tile, Body><<<grid, Tile::kNumWarps * 32, smem, stream>>>(params);
}

#endif  // __CUDACC__

}  // namespace flash_attn_legacy
