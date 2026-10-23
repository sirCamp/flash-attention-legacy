// ============================================================================
// Flash Attention — decode with a KV cache (flash_attn_with_kvcache), CUDA
// cores, every arch (docs/analysis/kernels/decode_design.md).
//
// Decode reads each K / V element once and does ~R FLOPs per byte (R = query
// rows per KV head): bandwidth bound for R <= 8 on every supported GPU, so one
// fp32 CUDA-core kernel serves SM 6.0-7.5. The caches are read in their own
// dtype (fp16 or bf16) and converted in registers: no copy of the cache.
//
// One block per (batch row, KV head, key split), 4 warps; the block computes the
// R = rep x q_len query rows of that KV head (GQA: K / V read once for the group)
// over its split of the rows' visible keys. With more than one split each block
// writes its normalized O (fp32) and lse, and split_combine_kernel
// (flash_attn_split.h) merges them. Each lane loads one 16-byte chunk of a key
// row (q in registers up to 4 rows, in shared memory up to 8): G =
// next_pow2(d / 8) lanes per key, 32 / G keys per warp step; the group that
// scores key j also reads V row j and owns output columns [8 * lane_in_group,
// +8). Each group keeps an online softmax state (m, l, O) per row over its
// keys; groups and warps are merged with the same decode_merge at the end (and
// splits in the combine). The sink of a row enters once (group 0 of warp 0 of
// split 0 starts at m = sink, l = 1).
// An optional key mask (padding of any shape): each block shrinks its split to
// the first / last kept key (a scan of its own keys), masked keys are not read.
// ============================================================================

#include "flash_attn_split.h"

#include <climits>
#include <stdexcept>

namespace flash_attn_legacy {

namespace {

constexpr int kDecodeWarps = 4;

template <typename Elem, int D, int R, bool kPaged>
__global__ void __launch_bounds__(kDecodeWarps * 32)
decode_kernel(DecodeParams p) {
    constexpr int G = D / 8 <= 4 ? 4 : D / 8 <= 8 ? 8 : D / 8 <= 16 ? 16 : 32;  // lanes per key
    constexpr int GW = 32 / G;                    // key groups per warp
    constexpr int NG = kDecodeWarps * GW;         // key groups per block
    static_assert(D / 8 <= 32, "one 16-byte chunk per lane");

    const int b = blockIdx.x / p.num_heads_k, hk = blockIdx.x % p.num_heads_k;
    const int split = blockIdx.y;
    const int rep = p.num_heads / p.num_heads_k;
    const int nrows = rep * p.q_len;              // <= R (host check)
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int li = lane % G;                      // lane within the key group
    const int grp = warp * GW + lane / G;         // key group within the block
    const bool active = li < p.head_dim / 8;      // owns columns [8 li, 8 li + 8)
    const int seqlen = min(max(p.cache_seqlens ? p.cache_seqlens[b] : p.uniform_seqlen, 0), p.max_len);  // in the cache
    const KeyWindow w(p.window_left, p.window_right, p.q_len, seqlen);
    const ScoreMap sm(p.softmax_scale, p.softcap);

    // Row r = query qi * rep + head offset: head hk * rep + r % rep, query r / rep.
    // q in fp32: registers for R <= 4, shared memory beyond, interleaved as
    // [row][element i][chunk c] so the G lanes of a group read consecutive words
    // (no bank conflicts) and the groups broadcast.
    constexpr bool kQSmem = R > 4;
    __shared__ float s_q[kQSmem ? R : 1][8][kQSmem ? G : 1];  // zeros past head_dim: read by every lane
    float q[kQSmem ? 1 : R][8], o[R][8], m[R], l[R];
    auto q_chunk = [&](int r, int c) {  // 8 elements of row r from column 8c (zeros past the rows)
        uint4 u = make_uint4(0, 0, 0, 0);
        if (r < nrows) {
            u = *reinterpret_cast<const uint4*>(static_cast<const typename Elem::T*>(p.q) + b * p.q_batch_stride +
                                                (r / rep) * p.q_seq_stride + (hk * rep + r % rep) * p.q_head_stride +
                                                8 * c);
        }
        return u;
    };
    if constexpr (kQSmem) {
        const int vecs = p.head_dim / 8;
        for (int e = threadIdx.x; e < R * G; e += kDecodeWarps * 32) {
            const int r = e / G, c = e % G;
            float f[8];
            Elem::to_float(c < vecs ? q_chunk(r, c) : make_uint4(0, 0, 0, 0), f);
            #pragma unroll
            for (int i = 0; i < 8; i++) s_q[r][i][c] = f[i];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int r = 0; r < R; r++) {
        const int h = hk * rep + r % rep;
        if constexpr (!kQSmem) Elem::to_float(active ? q_chunk(r, li) : make_uint4(0, 0, 0, 0), q[r]);
        #pragma unroll
        for (int i = 0; i < 8; i++) o[r][i] = 0.0f;
        const bool sink = p.sinks != nullptr && grp == 0 && split == 0 && r < nrows;
        m[r] = sink ? p.sinks[h] * kLog2e : -INFINITY;  // the sink once per row
        l[r] = sink ? 1.0f : 0.0f;
    }

    // Row j of the cache: contiguous (batch row b, kPaged = false: the address of
    // the unpaged kernel) or paged (block table[j / block_size], slot j % block_size)
    const typename Elem::T* K = static_cast<const typename Elem::T*>(p.k) + hk * p.k_head_stride +
                                (kPaged ? 0 : b * p.k_batch_stride);
    const typename Elem::T* V = static_cast<const typename Elem::T*>(p.v) + hk * p.v_head_stride +
                                (kPaged ? 0 : b * p.v_batch_stride);
    const int* table = kPaged ? p.block_table + index_t(b) * p.max_blocks : nullptr;
    auto key_offset = [&](int j, index_t batch_stride, index_t seq_stride) -> index_t {
        if constexpr (kPaged) {
            const int blk = p.block_shift >= 0 ? j >> p.block_shift : j / p.block_size;
            const int slot = p.block_shift >= 0 ? j & (p.block_size - 1) : j % p.block_size;
            return index_t(table[blk]) * batch_stride + slot * seq_stride;
        }
        else return j * seq_stride;
    };
    // The rows see keys [first_key(0), last_key(q_len - 1)] (causal / window
    // aligned to the end, seqlen), none before the left padding; this split takes
    // `chunk` of them (a multiple of the block's NG keys per step)
    const int leftpad = p.cache_leftpad ? min(max(p.cache_leftpad[b], 0), seqlen) : 0;
    const int key_first = max(w.first_key(0), leftpad), key_last = w.last_key(p.q_len - 1);
    const int range = max(key_last - key_first + 1, 0);
    const int chunk = (cdiv(range, p.num_splits) + NG - 1) / NG * NG;
    int key_begin = key_first + split * chunk;
    int key_end = min(key_last, key_begin + chunk - 1);  // inclusive (< key_begin: empty)
    const bool* mrow = p.key_mask ? p.key_mask + b * p.mask_batch_stride : nullptr;
    if (mrow) {  // shrink this split's keys to its first / last kept one
        int lo = INT_MAX, hi = -1;
        for (int j = key_begin + threadIdx.x; j <= key_end; j += kDecodeWarps * 32) {
            if (mrow[j]) { lo = min(lo, j); hi = max(hi, j); }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            lo = min(lo, __shfl_xor_sync(0xffffffffu, lo, off));
            hi = max(hi, __shfl_xor_sync(0xffffffffu, hi, off));
        }
        __shared__ int s_lo[kDecodeWarps], s_hi[kDecodeWarps];
        if (lane == 0) { s_lo[warp] = lo; s_hi[warp] = hi; }
        __syncthreads();
        #pragma unroll
        for (int wp = 0; wp < kDecodeWarps; wp++) { lo = min(lo, s_lo[wp]); hi = max(hi, s_hi[wp]); }
        key_begin = max(key_begin, min(lo, key_end + 1));  // no kept key: empty (no INT_MAX overflow)
        key_end = min(key_end, hi);
    }
    // Keys each row sees (causal / window), read in the key loop as registers
    int row_first[R], row_last[R];
    #pragma unroll
    for (int r = 0; r < R; r++) {
        row_first[r] = w.first_key(r / rep);
        row_last[r] = w.last_key(r / rep);
    }
    // Same trip count for every lane of a warp (the shuffles below need all 32):
    // the warp steps over GW keys at a time, keys past key_end are masked. Every
    // row is computed (q = 0 past nrows) without branches, phase by phase (the
    // scores, their reductions, the merges), so the rows interleave.
    sm.dispatch([&](auto softcapped) {
        constexpr bool kSoftcap = decltype(softcapped)::value;
        // the next step's key (mask byte, K and V rows) is loaded before this one is scored
        auto load = [&](int j, bool& in_range, uint4& ku, uint4& vu) {
            in_range = j <= key_end && (mrow == nullptr || mrow[j]);
            ku = make_uint4(0, 0, 0, 0); vu = make_uint4(0, 0, 0, 0);
            if (active && in_range) {
                ku = *reinterpret_cast<const uint4*>(K + key_offset(j, p.k_batch_stride, p.k_seq_stride) + 8 * li);
                vu = *reinterpret_cast<const uint4*>(V + key_offset(j, p.v_batch_stride, p.v_seq_stride) + 8 * li);
            }
        };
        bool next_in_range;
        uint4 next_ku, next_vu;
        load(key_begin + warp * GW + lane / G, next_in_range, next_ku, next_vu);
        for (int j0 = key_begin + warp * GW; j0 <= key_end; j0 += NG) {
            const int j = j0 + lane / G;
            const bool in_range = next_in_range;
            float kf[8], vf[8];
            Elem::to_float(next_ku, kf);
            Elem::to_float(next_vu, vf);
            load(j + NG, next_in_range, next_ku, next_vu);
            float s[R];
            #pragma unroll
            for (int r = 0; r < R; r++) {
                s[r] = 0.0f;
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    if constexpr (kQSmem) s[r] = fmaf(s_q[r][i][li], kf[i], s[r]);
                    else                  s[r] = fmaf(q[r][i], kf[i], s[r]);
                }
            }
            #pragma unroll
            for (int off = 1; off < G; off <<= 1) {
                #pragma unroll
                for (int r = 0; r < R; r++) s[r] += __shfl_xor_sync(0xffffffffu, s[r], off);
            }
            #pragma unroll
            for (int r = 0; r < R; r++) {
                float unused;
                float x = sm.log2_logit<kSoftcap>(s[r], unused);
                if (!in_range || j < row_first[r] || j > row_last[r]) x = -INFINITY;
                decode_merge(m[r], l[r], o[r], x, 1.0f, vf);
            }
        }
    });

    // Merge the groups of a warp (lanes with the same li own the same columns),
    // then the warps through shared memory
    #pragma unroll
    for (int r = 0; r < R; r++) {
        #pragma unroll
        for (int off = G; off < 32; off <<= 1) {
            const float m2 = __shfl_xor_sync(0xffffffffu, m[r], off);
            const float l2 = __shfl_xor_sync(0xffffffffu, l[r], off);
            float o2[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) o2[i] = __shfl_xor_sync(0xffffffffu, o[r][i], off);
            decode_merge(m[r], l[r], o[r], m2, l2, o2);
        }
    }
    __shared__ float s_m[kDecodeWarps][R], s_l[kDecodeWarps][R];
    __shared__ float s_o[kDecodeWarps][R][G][8];
    if (lane < G) {
        #pragma unroll
        for (int r = 0; r < R; r++) {
            if (li == 0) { s_m[warp][r] = m[r]; s_l[warp][r] = l[r]; }
            #pragma unroll
            for (int i = 0; i < 8; i++) s_o[warp][r][li][i] = o[r][i];
        }
    }
    __syncthreads();
    if (warp != 0 || lane >= G) return;
    #pragma unroll
    for (int r = 0; r < R; r++) {
        if (r >= nrows) break;
        for (int wp = 1; wp < kDecodeWarps; wp++) {
            float o2[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) o2[i] = s_o[wp][r][li][i];
            decode_merge(m[r], l[r], o[r], s_m[wp][r], s_l[wp][r], o2);
        }
        const int qi = r / rep, h = hk * rep + r % rep;
        const float inv = l[r] > 0.0f ? 1.0f / l[r] : 0.0f;  // no visible key nor sink: O = 0
        const float lse2 = l[r] > 0.0f ? m[r] + log2f(l[r]) : -INFINITY;  // log2 domain
        float f[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) f[i] = o[r][i] * inv;
        const index_t row = (index_t(b) * p.num_heads + h) * p.q_len + qi;  // [B, H_q, q_len]
        if (p.num_splits == 1) {
            if (active) {
                *reinterpret_cast<uint4*>(static_cast<typename Elem::T*>(p.out) + b * p.o_batch_stride +
                                          qi * p.o_seq_stride + h * p.o_head_stride + 8 * li) = Elem::from_float(f);
            }
            if (li == 0) p.lse[row] = lse2 * kLn2;
        } else {  // partials of this split (empty splits too: O = 0, lse = -inf)
            const index_t prow = index_t(split) * p.batch_size * p.num_heads * p.q_len + row;
            if (active) {
                float4* dst = reinterpret_cast<float4*>(p.o_part + prow * p.head_dim + 8 * li);
                dst[0] = make_float4(f[0], f[1], f[2], f[3]);
                dst[1] = make_float4(f[4], f[5], f[6], f[7]);
            }
            if (li == 0) p.lse_part[prow] = lse2;
        }
    }
}

template <typename Elem, int R, bool kPaged>
void launch_decode_d(DecodeParams& p, cudaStream_t stream) {
    const dim3 grid(p.batch_size * p.num_heads_k, p.num_splits);
    switch (p.head_dim) {
        case 32:  decode_kernel<Elem, 32, R, kPaged><<<grid, kDecodeWarps * 32, 0, stream>>>(p); break;
        case 64:  decode_kernel<Elem, 64, R, kPaged><<<grid, kDecodeWarps * 32, 0, stream>>>(p); break;
        case 72:
        case 80:  decode_kernel<Elem, 80, R, kPaged><<<grid, kDecodeWarps * 32, 0, stream>>>(p); break;  // 72: 9 lanes
        case 96:  decode_kernel<Elem, 96, R, kPaged><<<grid, kDecodeWarps * 32, 0, stream>>>(p); break;
        case 128: decode_kernel<Elem, 128, R, kPaged><<<grid, kDecodeWarps * 32, 0, stream>>>(p); break;
        case 256: decode_kernel<Elem, 256, R, kPaged><<<grid, kDecodeWarps * 32, 0, stream>>>(p); break;
        default:  throw std::runtime_error("Unsupported head_dim for decode");
    }
}

template <typename Elem>
void launch_decode_r(DecodeParams& p, cudaStream_t stream) {
    const int rows = p.num_heads / p.num_heads_k * p.q_len;
    auto launch = [&](auto paged) {
        constexpr bool kPaged = decltype(paged)::value;
        if (rows <= 1)      launch_decode_d<Elem, 1, kPaged>(p, stream);
        else if (rows <= 2) launch_decode_d<Elem, 2, kPaged>(p, stream);
        else if (rows <= 4) launch_decode_d<Elem, 4, kPaged>(p, stream);
        else                launch_decode_d<Elem, 8, kPaged>(p, stream);  // rows <= 8 (host check)
    };
    if (p.block_table) launch(std::true_type{});
    else               launch(std::false_type{});
    if (p.num_splits > 1) {
        launch_split_combine<Elem>({p.o_part, p.lse_part, p.out, p.lse, p.o_batch_stride, p.o_seq_stride, p.o_head_stride,
                                    p.batch_size, p.q_len, p.num_heads, p.head_dim, p.num_splits}, stream);
    }
}

}  // namespace

// Query rows per KV head the decode kernel handles in one block.
int flash_attn_decode_max_rows() { return 8; }

// Shared by decode and the forward with key splits (flash_attn_split.h): the cap
// and the automatic count of splits, and the host entry of the combine.
constexpr int kMaxSplits = 128;
int flash_attn_max_splits() { return kMaxSplits; }

// Key splits (decode kernel, forward against a cache) for a grid of `blocks`
// blocks without splits, `resident` of them per SM (decode: 4 blocks of 4
// warps; the cache forward: 4 on Pascal, compute bound on small tiles, 1 on SM
// 7.x, bandwidth bound on large ones), on `sms` SMs, over a cache of max_len keys
// of which each block reads K / V for `rows` query rows (no device read:
// CUDA-graph friendly): ~2 waves of blocks; at least max(128, rows) keys per
// split (the partials, 8 * rows * d bytes per split written and read, stay
// within ~2x of the K / V read, 4 * max_len * d: filling the GPU pays more, as
// measured on packed speculative shapes); at most kMaxSplits. Sized by the
// capacity: a large preallocated cache holding few keys gets more (empty)
// splits than it needs.
int flash_attn_num_splits(int blocks, int sms, int resident, int max_len, int rows) {
    const int want = cdiv(2 * sms * resident, blocks);
    const int by_len = max(max_len / max(128, rows), 1);
    return max(1, min(min(want, by_len), kMaxSplits));
}

void flash_attn_split_combine(const SplitCombineParams& p, bool bf16, cudaStream_t stream) {
    if (bf16) launch_split_combine<Bf16Elem>(p, stream);
    else      launch_split_combine<HalfElem>(p, stream);
}

void flash_attn_decode(DecodeParams& p, bool bf16, cudaStream_t stream) {
    if (bf16) launch_decode_r<Bf16Elem>(p, stream);
    else      launch_decode_r<HalfElem>(p, stream);
}

}  // namespace flash_attn_legacy
