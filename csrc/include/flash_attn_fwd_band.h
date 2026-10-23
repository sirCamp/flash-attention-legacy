#pragma once

// ============================================================================
// FA2-style forward, shared by every kernel family (docs/PLAN.md P2.1 / P2.8).
//
// A Q block of Br rows is split in bands of 16 rows, one or more warps each. The parts
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
// FwdBlockArgs&)` that calls fwd_band_run with its own S / P / O code (mma.sync:
// flash_attn_fwd_turing.cu, WMMA: flash_attn_fwd_volta.cu, CUDA cores:
// flash_attn_fwd_pascal.cu) and a tile dispatcher per head_dim.
// ============================================================================

#include "flash_attn_common.h"

namespace flash_attn_legacy {

#ifdef __CUDACC__

constexpr int kBandRows = 16;  // query rows per band (of one or more warps)

// One Q block: its rows start at Q / O / L (row strides q_stride / o_stride /
// l_stride); K and V start at key 0 of the sequence. The cache kernel (kCache)
// takes split `split` of the block's visible K/V blocks and writes the split's
// partial rows at O_part (fp32, row stride head_dim) and L_part (log2) instead of
// O / L; there Q, O_part and L_part start at token 0 of query head h0, and with
// packed GQA rows (rows_per_token = H_q / H_kv) block row r is token
// (q_row_start + r) / rows_per_token of query head h0 + (q_row_start + r) %
// rows_per_token (Q: token stride q_stride, head stride q_head_stride; partial
// row (head offset) * seq_len_q + token); seq_len_q counts tokens.
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
    int split, num_splits;          // num_splits 1: no split
    float* __restrict__ O_part;
    float* __restrict__ L_part;
    int rows_per_token;             // the cache kernel: 1, or H_q / H_kv with packed GQA rows
    index_t q_head_stride;

    __device__ __forceinline__ KeyWindow window() const {
        return KeyWindow(window_left, window_right, seq_len_q, seq_len_k);
    }
    __device__ __forceinline__ ScoreMap score_map() const { return ScoreMap(softmax_scale, softcap); }
};

// Token and query-head offset (from h0) of an absolute row of the cache kernel's
// rows (packed GQA rows; unpacked: the row itself and 0). The one place rows split.
struct FwdRow { int token, head; };
__device__ __forceinline__ FwdRow fwd_row(const FwdBlockArgs& a, int abs_row) {
    return {abs_row / a.rows_per_token, abs_row % a.rows_per_token};
}

// Token of an absolute row: the cache kernel's split, or the row itself.
template <bool kCache>
__device__ __forceinline__ int fwd_token(const FwdBlockArgs& a, int abs_row) {
    return kCache ? fwd_row(a, abs_row).token : abs_row;
}

// Partial row (the cache kernel) of block row `row`: (head offset) * seq_len_q + token.
__device__ __forceinline__ index_t fwd_part_row(const FwdBlockArgs& a, int row) {
    const FwdRow r = fwd_row(a, a.q_row_start + row);
    return index_t(r.head) * a.seq_len_q + r.token;
}

// The final values of N consecutive columns from `col` of block row `row` (O
// already multiplied by the row's inv): half into O, or with kCache fp32 into
// the split's partial. Every family writes its output through these two.
template <bool kCache, int N>
__device__ __forceinline__ void fwd_store_o(const FwdBlockArgs& a, int row, int col, const float (&v)[N]) {
    static_assert(N == 2 || N == 4 || N == 8, "2, 4 or 8 columns");
    if constexpr (kCache) {  // fp32, 8-byte aligned (col even, head_dim % 8 == 0)
        float* dst = a.O_part + fwd_part_row(a, row) * a.head_dim + col;
        #pragma unroll
        for (int i = 0; i < N; i += 2) *reinterpret_cast<float2*>(dst + i) = make_float2(v[i], v[i + 1]);
    } else {
        half2 h[N / 2];
        #pragma unroll
        for (int i = 0; i < N / 2; i++) h[i] = __floats2half2_rn(v[2 * i], v[2 * i + 1]);
        half* dst = a.O + row * a.o_stride + col;
        if constexpr (N == 2) *reinterpret_cast<half2*>(dst) = h[0];
        else if constexpr (N == 4) { uint2 u; memcpy(&u, h, sizeof(u)); *reinterpret_cast<uint2*>(dst) = u; }
        else { uint4 u; memcpy(&u, h, sizeof(u)); *reinterpret_cast<uint4*>(dst) = u; }
    }
}

// lse: natural log (BandSoftmax::finalize); the split partial keeps it in log2.
template <bool kCache>
__device__ __forceinline__ void fwd_store_lse(const FwdBlockArgs& a, int row, float lse) {
    if constexpr (kCache) a.L_part[fwd_part_row(a, row)] = lse * kLog2e;  // -inf stays -inf
    else                  a.L[row * a.l_stride] = lse;
}

// Key `key` is not visible to sequence row `q_row` (KeyWindow over the tokens;
// includes keys past the end).
template <bool kCache>
__device__ __forceinline__ bool fwd_key_masked(const FwdBlockArgs& a, int q_row, int key) {
    return a.window().masked(fwd_token<kCache>(a, q_row), key);
}

// Raw scores s[r][c] of a block -> log2-domain logits (ScoreMap), -inf for keys
// outside the KeyWindow when kMask (masked after the map: c * tanh(-inf) would be
// finite). row(r) / key(c): absolute row of the block's rows (a sequence row, or
// a packed GQA row of the cache kernel) / key of s[r][c].
template <bool kMask, bool kCache, int R, int C, typename RowOf, typename KeyOf>
__device__ __forceinline__ void fwd_logits(const FwdBlockArgs& a, float (&s)[R][C], RowOf&& row, KeyOf&& key) {
    a.score_map().log2_logits(s);
    if constexpr (kMask) {
        #pragma unroll
        for (int r = 0; r < R; r++) {
            #pragma unroll
            for (int c = 0; c < C; c++) {
                if (fwd_key_masked<kCache>(a, row(r), key(c))) s[r][c] = -INFINITY;
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
// (rows past valid_q) still take part in every barrier. Shared memory written
// by step() for block j is not rewritten by step() for block j + 1 before every
// warp is done with block j: the block-wide barrier closing each iteration
// (the column-split Pascal forward relies on it for its partial S).
template <int Br, int Bc, bool kCache, typename LoadQ, typename Fetch, typename Commit, typename Step>
__device__ __forceinline__ void fwd_band_loop(const FwdBlockArgs& a, int band_row0, bool band_active,
                                              LoadQ&& load_q, Fetch&& fetch, Commit&& commit, Step&& step) {
    // First / last key of a row grow with the row (its token: packed GQA rows of
    // a token share them): the Q block sees keys [first_key(first row),
    // last_key(last row)], a band likewise, and a block inside
    // [first_key(last row), last_key(first row)] of the band needs no mask.
    const KeyWindow w = a.window();
    const int q_first = fwd_token<kCache>(a, a.q_row_start);
    const int q_last = fwd_token<kCache>(a, a.q_row_start + a.valid_q - 1);
    int kv_begin = w.first_key(q_first) / Bc;
    const int last_key = w.last_key(q_last);
    int kv_end = last_key < 0 ? 0 : last_key / Bc + 1;
    if (a.num_splits > 1) {  // this split's share of the visible blocks (may be empty)
        const int chunk = cdiv(max(kv_end - kv_begin, 0), a.num_splits);
        kv_begin += a.split * chunk;
        kv_end = min(kv_end, kv_begin + chunk);
    }
    const int band_first = fwd_token<kCache>(a, a.q_row_start + band_row0);
    const int band_last = fwd_token<kCache>(a, a.q_row_start + band_row0 + kBandRows - 1);
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
template <typename Tile, int LD, bool kCache, typename DstT, typename Step>
__device__ __forceinline__ void fwd_band_run(const FwdBlockArgs& a, int band_row0, bool band_active,
                                             DstT* smem_Q, DstT* smem_K, DstT* smem_V, Step&& step) {
    constexpr int Br = Tile::Br, Bc = Tile::Bc, d = Tile::d, NT = Tile::kNumWarps * 32;
    auto load_q = [&]() {
        if constexpr (kCache) {  // rows from token 0 of head h0: (token, head offset), packed or not
            load_tile_g2s_rows<Br, d, NT, LD>([&](int r) {
                const FwdRow x = fwd_row(a, a.q_row_start + r);
                return a.Q + index_t(x.token) * a.q_stride + index_t(x.head) * a.q_head_stride;
            }, smem_Q, a.valid_q, a.head_dim);
        } else {
            load_tile_g2s<Br, d, NT, LD>(a.Q, a.q_stride, smem_Q, a.valid_q, a.head_dim);
        }
    };
    if constexpr (Tile::kPrefetch) {
        TileRegs<Bc, d, NT> next_K, next_V;
        fwd_band_loop<Br, Bc, kCache>(a, band_row0, band_active, load_q,
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
        fwd_band_loop<Br, Bc, kCache>(a, band_row0, band_active, load_q,
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
// kCache: the variant of flash_attn_with_kvcache (per-batch key lengths, key
// splits, the splits' partials for the combine). A template parameter: the
// plain forward carries none of it (a runtime branch or even the per-batch
// length load made the compiler lay the kernel out differently: Pascal d=128
// +23% on the P100, d=64 +9% on the 1080 Ti).
template <typename Tile, typename Body, bool kCache>
__global__ void __launch_bounds__(Tile::kNumWarps * 32)
fwd_batched_kernel(FlashAttnParams p) {
    // Flat grid: x = (batch * heads) * q_blocks + q_block (no 65535 limit on
    // B * H; the Q blocks of one head are adjacent, sharing its K/V in L2), y =
    // key split (the cache kernel). The cache kernel with packed GQA rows: heads =
    // KV heads, rows = q_len * (H_q / H_kv) (token, query head of the group);
    // unpacked, the query heads of a KV head are adjacent within a split.
    FwdBlockArgs a;
    if constexpr (kCache) {
        const int rpt = gqa_rows_per_token(p.pack_gqa, p.num_heads, p.num_heads_k);
        const int heads = p.num_heads / rpt, rows = p.seq_len_q * rpt;
        const int q_blocks = cdiv(rows, Tile::Br);
        const int split = blockIdx.y;
        const int bh_idx = blockIdx.x / q_blocks;
        const int b_idx = bh_idx / heads;
        const int h0 = (bh_idx % heads) * rpt;  // first query head of the block's rows
        const int kv_h_idx = h0 / (p.num_heads / p.num_heads_k);
        const int q_row_start = (blockIdx.x % q_blocks) * Tile::Br;
        if (q_row_start >= rows) return;
        a.Q = p.Q + b_idx * p.q_batch_stride + h0 * p.q_head_stride;  // token 0 (rows: fwd_band_run)
        a.q_stride = p.q_seq_stride;
        a.q_head_stride = p.q_head_stride;
        a.K = p.K + b_idx * p.k_batch_stride + kv_h_idx * p.k_head_stride;
        a.k_stride = p.k_seq_stride;
        a.V = p.V + b_idx * p.v_batch_stride + kv_h_idx * p.v_head_stride;
        a.v_stride = p.v_seq_stride;
        a.O = nullptr;
        a.o_stride = 0;
        a.L = nullptr;
        a.l_stride = 0;
        a.q_row_start = q_row_start;
        a.valid_q = min(Tile::Br, rows - q_row_start);
        a.seq_len_q = p.seq_len_q;  // tokens
        a.seq_len_k = p.seqlens_k ? min(max(p.seqlens_k[b_idx], 0), p.seq_len_k) : p.seq_len_k;
        a.head_dim = tile_valid_cols<Tile::d>(p.head_dim);
        a.softmax_scale = p.score_scale ? p.softmax_scale * __ldg(p.score_scale) : p.softmax_scale;
        a.softcap = p.softcap;
        a.sink_log2 = p.sinks && split == 0 ? p.sinks[h0] * kLog2e : -INFINITY;  // once per row (no packing with sinks)
        a.window_left = p.window_left;
        a.window_right = p.window_right;
        a.split = split;
        a.num_splits = max(p.num_splits, 1);
        a.rows_per_token = rpt;
        // partials of the split from (b, h0, token 0): rows (head offset) * q_len + token
        const index_t prow = (index_t(split) * p.batch_size + b_idx) * p.num_heads * p.seq_len_q + index_t(h0) * p.seq_len_q;
        a.O_part = p.O_part + prow * p.head_dim;
        a.L_part = p.L_part + prow;
    } else {
        const int q_blocks = cdiv(p.seq_len_q, Tile::Br);
        const int bh_idx = blockIdx.x / q_blocks;
        const int b_idx = bh_idx / p.num_heads;
        const int h_idx = bh_idx % p.num_heads;
        const int kv_h_idx = h_idx / (p.num_heads / p.num_heads_k);
        const int q_row_start = (blockIdx.x % q_blocks) * Tile::Br;
        if (q_row_start >= p.seq_len_q) return;

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
        a.softmax_scale = p.score_scale ? p.softmax_scale * __ldg(p.score_scale) : p.softmax_scale;
        a.softcap = p.softcap;
        a.sink_log2 = p.sinks ? p.sinks[h_idx] * kLog2e : -INFINITY;
        a.window_left = p.window_left;
        a.window_right = p.window_right;
        a.split = 0;
        a.num_splits = 1;
        a.O_part = nullptr;
        a.L_part = nullptr;
        a.rows_per_token = 1;
        a.q_head_stride = 0;
    }
    Body::template run<Tile, kCache>(a);
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
    a.softmax_scale = p.score_scale ? p.softmax_scale * __ldg(p.score_scale) : p.softmax_scale;
    a.softcap = p.softcap;
    a.sink_log2 = p.sinks ? p.sinks[h_idx] * kLog2e : -INFINITY;
    a.window_left = p.window_left;
    a.window_right = p.window_right;
    a.split = 0;
    a.num_splits = 1;
    a.O_part = nullptr;
    a.L_part = nullptr;
    a.rows_per_token = 1;
    a.q_head_stride = 0;
    Body::template run<Tile, false>(a);
}

// ---------------------------------------------------------------------------
// Host launchers of a family's Body with the tile chosen by its dispatcher.
// ---------------------------------------------------------------------------
template <typename Tile, typename Body>
void launch_fwd(FlashAttnParams& params, cudaStream_t stream) {
    constexpr int smem = Tile::smem_bytes();
    const int rpt = gqa_rows_per_token(params.O_part && params.pack_gqa, params.num_heads, params.num_heads_k);
    const int64_t blocks = int64_t(cdiv(params.seq_len_q * rpt, Tile::Br)) * params.batch_size * (params.num_heads / rpt);
    if (blocks > INT32_MAX) throw std::runtime_error("flash_attn_legacy: batch * heads * query blocks exceeds the grid limit");
    const dim3 grid(static_cast<unsigned>(blocks), max(params.num_splits, 1));
    auto launch = [&](auto cache) {
        const auto kernel = fwd_batched_kernel<Tile, Body, decltype(cache)::value>;
        FLASH_ATTN_CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        kernel<<<grid, Tile::kNumWarps * 32, smem, stream>>>(params);
    };
    if (params.O_part) launch(std::true_type{});  // flash_attn_with_kvcache: partials, even of one split
    else               launch(std::false_type{});
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
