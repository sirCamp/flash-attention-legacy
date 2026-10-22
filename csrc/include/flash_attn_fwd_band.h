#pragma once

// ============================================================================
// FA2-style forward, shared by every kernel family (docs/PLAN.md P2.1 / P2.8).
//
// A Q block of Br rows is split in bands of 16 rows, one per warp. The parts
// that do not depend on how S = Q K^T and O += P V are computed live here, once:
//   - FwdBlockArgs: pointers and lengths of one Q block, from the batched or
//     the varlen grid (fwd_batched_kernel / fwd_varlen_kernel)
//   - fwd_key_masked: the KeyWindow of the sequence (causal, sliding window,
//     tail of the keys); score_map(): scale and softcap (ScoreMap)
//   - BandSoftmax: online softmax in the log2 domain for the rows a thread holds,
//     reduced over the lanes that share a row, started from the head's sink
//   - fwd_band_loop: the K/V loop (the blocks the Q block can see, blocks
//     without masking, blocks a band cannot see, K/V prefetch, the two
//     block-wide barriers)
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
    const half* __restrict__ Q; index_t q_stride;
    const half* __restrict__ K; index_t k_stride;
    const half* __restrict__ V; index_t v_stride;
    half* __restrict__ O; index_t o_stride;
    float* __restrict__ L; index_t l_stride;
    int q_row_start, valid_q, seq_len_q, seq_len_k;
    int head_dim;  // real head_dim (<= Tile::d: extra tile columns are zero)
    float softmax_scale, softcap;
    float sink_log2;                // sink logit of the head * log2(e), -inf = none
    int window_left, window_right;  // KeyWindow of the sequence

    __device__ __forceinline__ KeyWindow window() const {
        return KeyWindow(window_left, window_right, seq_len_q, seq_len_k);
    }
    __device__ __forceinline__ ScoreMap score_map() const { return ScoreMap(softmax_scale, softcap); }
};

// Key `key` is not visible to sequence row `q_row` (KeyWindow; includes keys
// past the end).
__device__ __forceinline__ bool fwd_key_masked(const FwdBlockArgs& a, int q_row, int key) {
    return a.window().masked(q_row, key);
}

// Raw scores s[r][c] of a block -> log2-domain logits (ScoreMap), -inf for keys
// outside the KeyWindow when kMask (masked after the map: c * tanh(-inf) would be
// finite). row(r) / key(c): sequence row / key of s[r][c].
template <bool kMask, int R, int C, typename RowOf, typename KeyOf>
__device__ __forceinline__ void fwd_logits(const FwdBlockArgs& a, float (&s)[R][C], RowOf&& row, KeyOf&& key) {
    a.score_map().log2_logits(s);
    if constexpr (kMask) {
        #pragma unroll
        for (int r = 0; r < R; r++) {
            #pragma unroll
            for (int c = 0; c < C; c++) {
                if (fwd_key_masked(a, row(r), key(c))) s[r][c] = -INFINITY;
            }
        }
    }
}

// Online softmax state of the ROWS rows a thread holds; each row is spread over
// ROW_LANES lanes (lane ids differing only in the low log2(ROW_LANES) bits).
// A sink (logit without value) is the state of a row that has seen one key of
// logit sink and value 0: m = sink, sum = 1 (1 / ROW_LANES per lane). Rows that
// see no key end with O = 0 and L = sink, or m = -inf, L = -inf without sink.
template <int ROWS, int ROW_LANES>
struct BandSoftmax {
    static_assert((ROW_LANES & (ROW_LANES - 1)) == 0 && ROW_LANES <= 32, "ROW_LANES must be a power of 2");
    float m[ROWS];    // running max, log2 domain (scores * scale * log2e)
    float sum[ROWS];  // partial sum of this lane's columns

    __device__ __forceinline__ void init(float sink_log2) {
        const float share = sink_log2 > -INFINITY ? 1.0f / ROW_LANES : 0.0f;  // exact: power of 2
        #pragma unroll
        for (int r = 0; r < ROWS; r++) { m[r] = sink_log2; sum[r] = share; }
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

    // inv[r]: factor for the final O row (0 for rows that saw no key nor sink);
    // lse[r]: natural-log logsumexp of the logits and the sink (-inf for those rows
    // without sink).
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
//                     mask is std::true_type when some key of the block is
//                     outside the KeyWindow of some row of the band
// The loop visits the K/V blocks some row of the Q block can see (KeyWindow:
// causal, sliding window); fetch(j + 1) is issued right after block j is
// committed, so its global loads overlap the computation of block j. Bands skip
// blocks they cannot see and only mask the blocks that need it. Inactive bands
// (rows past valid_q) still take part in every barrier.
template <int Br, int Bc, typename LoadQ, typename Fetch, typename Commit, typename Step>
__device__ __forceinline__ void fwd_band_loop(const FwdBlockArgs& a, int band_row0, bool band_active,
                                              LoadQ&& load_q, Fetch&& fetch, Commit&& commit, Step&& step) {
    // First / last key of a row grow with the row: the Q block sees keys
    // [first_key(first row), last_key(last row)], a band likewise, and a block
    // inside [first_key(last row), last_key(first row)] of the band needs no mask.
    const KeyWindow w = a.window();
    const int q_first = a.q_row_start, q_last = a.q_row_start + a.valid_q - 1;
    const int kv_begin = w.first_key(q_first) / Bc;
    const int last_key = w.last_key(q_last);
    const int kv_end = last_key < 0 ? 0 : last_key / Bc + 1;
    const int band_first = a.q_row_start + band_row0, band_last = band_first + kBandRows - 1;
    const int band_first_key = w.first_key(band_first), band_last_key = w.last_key(band_last);
    const int nomask_first = w.first_key(band_last), nomask_last = w.last_key(band_first);
    if (kv_begin < kv_end) fetch(kv_begin);  // in flight while Q is loaded
    load_q();
    for (int kv_blk = kv_begin; kv_blk < kv_end; kv_blk++) {
        const int kv_start = kv_blk * Bc;
        commit();
        __syncthreads();
        if (kv_blk + 1 < kv_end) fetch(kv_blk + 1);
        if (band_active && kv_start <= band_last_key && kv_start + Bc - 1 >= band_first_key) {  // warp-uniform
            if (kv_start >= nomask_first && kv_start + Bc - 1 <= nomask_last) step(std::false_type{}, kv_start);
            else                                                              step(std::true_type{}, kv_start);
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
    auto load_q = [&]() { load_tile_g2s<Br, d, NT, LD>(a.Q, a.q_stride, smem_Q, a.valid_q, a.head_dim); };
    if constexpr (Tile::kPrefetch) {
        TileRegs<Bc, d, NT> next_K, next_V;
        fwd_band_loop<Br, Bc>(a, band_row0, band_active, load_q,
            [&](int kv_blk) {
                const int kv_start = kv_blk * Bc;
                const int valid_kv = min(Bc, a.seq_len_k - kv_start);
                load_tile_g2r(a.K + kv_start * a.k_stride, a.k_stride, next_K, valid_kv, a.head_dim);
                load_tile_g2r(a.V + kv_start * a.v_stride, a.v_stride, next_V, valid_kv, a.head_dim);
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
                load_tile_g2s<Bc, d, NT, LD>(a.K + kv_start * a.k_stride, a.k_stride, smem_K, valid_kv, a.head_dim);
                load_tile_g2s<Bc, d, NT, LD>(a.V + kv_start * a.v_stride, a.v_stride, smem_V, valid_kv, a.head_dim);
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
    // Flat grid: x = (batch * heads) * q_blocks + q_block (no 65535 limit on
    // B * H; the Q blocks of one head are adjacent, sharing its K/V in L2).
    const int q_blocks = cdiv(p.seq_len_q, Tile::Br);
    const int bh_idx = blockIdx.x / q_blocks;
    const int b_idx = bh_idx / p.num_heads;
    const int h_idx = bh_idx % p.num_heads;
    const int kv_h_idx = h_idx / (p.num_heads / p.num_heads_k);
    const int q_row_start = (blockIdx.x % q_blocks) * Tile::Br;
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
    a.head_dim = tile_valid_cols<Tile::d>(p.head_dim);
    a.softmax_scale = p.softmax_scale;
    a.softcap = p.softcap;
    a.sink_log2 = p.sinks ? p.sinks[h_idx] * kLog2e : -INFINITY;
    a.window_left = p.window_left;
    a.window_right = p.window_right;
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
    a.L = p.L + index_t(q_start_tok + q_row_start) * p.num_heads + h_idx;  // L is [total_q, H_q]
    a.l_stride = p.num_heads;
    a.q_row_start = q_row_start;
    a.valid_q = min(Tile::Br, seq_len_q - q_row_start);
    a.seq_len_q = seq_len_q;
    a.seq_len_k = p.cu_seqlens_k[seq_idx + 1] - k_start_tok;
    a.head_dim = tile_valid_cols<Tile::d>(p.head_dim);
    a.softmax_scale = p.softmax_scale;
    a.softcap = p.softcap;
    a.sink_log2 = p.sinks ? p.sinks[h_idx] * kLog2e : -INFINITY;
    a.window_left = p.window_left;
    a.window_right = p.window_right;
    Body::template run<Tile>(a);
}

// ---------------------------------------------------------------------------
// Host launchers of a family's Body with the tile chosen by its dispatcher.
// ---------------------------------------------------------------------------
template <typename Tile, typename Body>
void launch_fwd(FlashAttnParams& params, cudaStream_t stream) {
    constexpr int smem = Tile::smem_bytes();
    const int64_t blocks = int64_t(cdiv(params.seq_len_q, Tile::Br)) * params.batch_size * params.num_heads;
    if (blocks > INT32_MAX) throw std::runtime_error("flash_attn_legacy: batch * heads * query blocks exceeds the grid limit");
    dim3 grid(static_cast<unsigned>(blocks));
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
