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
// writes its normalized O (fp32) and lse, and decode_combine_kernel merges them. Each
// lane loads one 16-byte chunk of a key row (q in registers up to 4 rows, in
// shared memory up to 8): G = next_pow2(d / 8) lanes per key,
// 32 / G keys per warp step; the group that scores key j also reads V row j and
// owns output columns [8 * lane_in_group, +8). Each group keeps an online
// softmax state (m, l, O) per row over its keys; groups and warps are merged
// with the same decode_merge at the end (and splits in the combine). The sink of
// a row enters once (group 0 of warp 0 of split 0 starts at m = sink, l = 1).
// ============================================================================

#include "flash_attn_common.h"

#include <cuda_bf16.h>
#include <stdexcept>
#include <string>

namespace flash_attn_legacy {

namespace {

// 8 elements of a 16-byte chunk <-> fp32
struct HalfElem {
    using T = half;
    static __device__ __forceinline__ void to_float(const uint4& u, float (&f)[8]) {
        const half2* h = reinterpret_cast<const half2*>(&u);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const float2 x = __half22float2(h[i]);
            f[2 * i] = x.x; f[2 * i + 1] = x.y;
        }
    }
    static __device__ __forceinline__ uint4 from_float(const float (&f)[8]) {
        uint4 u;
        half2* h = reinterpret_cast<half2*>(&u);
        #pragma unroll
        for (int i = 0; i < 4; i++) h[i] = __floats2half2_rn(f[2 * i], f[2 * i + 1]);
        return u;
    }
};

struct Bf16Elem {  // bf16 -> fp32 is a shift: no conversion instruction needed on any arch
    using T = __nv_bfloat16;
    static __device__ __forceinline__ void to_float(const uint4& u, float (&f)[8]) {
        const uint32_t w[4] = {u.x, u.y, u.z, u.w};
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            f[2 * i] = __uint_as_float(w[i] << 16);
            f[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
        }
    }
    static __device__ __forceinline__ uint4 from_float(const float (&f)[8]) {
        uint4 u;
        __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&u);
        #pragma unroll
        for (int i = 0; i < 4; i++) h[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
        return u;
    }
};

// (m, l, o) <- merge with (m2, l2, o2): online-softmax states in the log2 domain
// (m: running max, l: sum of exp2(x - m), o: sum of exp2(x - m) v). The per-key
// update is a merge with (x, 1, v); groups, warps and splits merge the same way.
__device__ __forceinline__ void decode_merge(float& m, float& l, float (&o)[8], float m2, float l2,
                                             const float (&o2)[8]) {
    const float mn = fmaxf(m, m2);
    if (mn == -INFINITY) return;  // both empty (no (-inf) - (-inf))
    const float a1 = exp2f(m - mn), a2 = exp2f(m2 - mn);
    l = l * a1 + l2 * a2;
    #pragma unroll
    for (int i = 0; i < 8; i++) o[i] = o[i] * a1 + o2[i] * a2;
    m = mn;
}

constexpr int kDecodeWarps = 4;

template <typename Elem, int D, int R>
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
    const int seqlen = min(max(p.cache_seqlens[b], 0), p.max_len);  // never past the cache
    const KeyWindow w(p.window_left, p.window_right, p.q_len, seqlen);
    const ScoreMap sm(p.softmax_scale, p.softcap);

    // Row r = query qi * rep + head offset: head hk * rep + r % rep, query r / rep.
    // q in fp32: registers for R <= 4, shared memory beyond, interleaved as
    // [row][element i][chunk c] so the G lanes of a group read consecutive words
    // (no bank conflicts) and the groups broadcast.
    constexpr bool kQSmem = R > 4;
    __shared__ float s_q[kQSmem ? R : 1][8][kQSmem ? D / 8 : 1];
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
        for (int e = threadIdx.x; e < R * vecs; e += kDecodeWarps * 32) {
            const int r = e / vecs, c = e % vecs;
            float f[8];
            Elem::to_float(q_chunk(r, c), f);
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

    const typename Elem::T* K = static_cast<const typename Elem::T*>(p.k) + b * p.k_batch_stride + hk * p.k_head_stride;
    const typename Elem::T* V = static_cast<const typename Elem::T*>(p.v) + b * p.v_batch_stride + hk * p.v_head_stride;
    // The rows see keys [first_key(0), last_key(q_len - 1)] (causal / window
    // aligned to the end, seqlen), none before the left padding; this split takes
    // `chunk` of them (a multiple of the block's NG keys per step)
    const int leftpad = p.cache_leftpad ? min(max(p.cache_leftpad[b], 0), seqlen) : 0;
    const int key_first = max(w.first_key(0), leftpad), key_last = w.last_key(p.q_len - 1);
    const int range = max(key_last - key_first + 1, 0);
    const int chunk = (cdiv(range, p.num_splits) + NG - 1) / NG * NG;
    const int key_begin = key_first + split * chunk;
    const int key_end = min(key_last, key_begin + chunk - 1);  // inclusive (< key_begin: empty)
    // Same trip count for every lane of a warp (the shuffles below need all 32):
    // the warp steps over GW keys at a time, keys past key_end are masked.
    for (int j0 = key_begin + warp * GW; j0 <= key_end; j0 += NG) {
        const int j = j0 + lane / G;
        const bool in_range = j <= key_end;
        uint4 ku = make_uint4(0, 0, 0, 0), vu = make_uint4(0, 0, 0, 0);
        if (active && in_range) {
            ku = *reinterpret_cast<const uint4*>(K + j * p.k_seq_stride + 8 * li);
            vu = *reinterpret_cast<const uint4*>(V + j * p.v_seq_stride + 8 * li);
        }
        float kf[8], vf[8];
        Elem::to_float(ku, kf);
        Elem::to_float(vu, vf);
        #pragma unroll
        for (int r = 0; r < R; r++) {
            if (r >= nrows) break;  // uniform over the block: the shuffles are skipped together
            float s = 0.0f;
            if constexpr (kQSmem) {
                if (active) {  // lanes past head_dim / 8 (head_dim < 8 G) have no slot
                    #pragma unroll
                    for (int i = 0; i < 8; i++) s = fmaf(s_q[r][i][li], kf[i], s);
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 8; i++) s = fmaf(q[r][i], kf[i], s);
            }
            #pragma unroll
            for (int off = 1; off < G; off <<= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
            float unused, x;
            if (sm.softcapped()) x = sm.log2_logit<true>(s, unused);
            else                 x = sm.log2_logit<false>(s, unused);
            if (!in_range || w.masked(r / rep, j)) x = -INFINITY;
            decode_merge(m[r], l[r], o[r], x, 1.0f, vf);
        }
    }

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

// Merge of the splits: one thread per 8 columns of a (row, head, query); each
// split's normalized (O_s, lse_s in log2) is the state (m = lse_s, l = 1, O_s).
template <typename Elem>
__global__ void decode_combine_kernel(DecodeParams p) {
    const int vecs = p.head_dim / 8;
    const index_t rows = index_t(p.batch_size) * p.num_heads * p.q_len;
    const index_t idx = index_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= rows * vecs) return;
    const index_t row = idx / vecs;
    const int c = static_cast<int>(idx % vecs);
    float m = -INFINITY, l = 0.0f, o[8] = {};
    for (int s = 0; s < p.num_splits; s++) {
        const index_t prow = index_t(s) * rows + row;
        const float4* src = reinterpret_cast<const float4*>(p.o_part + prow * p.head_dim + 8 * c);
        const float4 lo = src[0], hi = src[1];
        const float o2[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
        decode_merge(m, l, o, p.lse_part[prow], 1.0f, o2);
    }
    const float inv = l > 0.0f ? 1.0f / l : 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; i++) o[i] *= inv;
    const int qi = static_cast<int>(row % p.q_len);
    const int h = static_cast<int>((row / p.q_len) % p.num_heads);
    const int b = static_cast<int>(row / (index_t(p.num_heads) * p.q_len));
    *reinterpret_cast<uint4*>(static_cast<typename Elem::T*>(p.out) + b * p.o_batch_stride + qi * p.o_seq_stride +
                              h * p.o_head_stride + 8 * c) = Elem::from_float(o);
    if (c == 0) p.lse[row] = l > 0.0f ? (m + log2f(l)) * kLn2 : -INFINITY;
}

template <typename Elem, int R>
void launch_decode_d(DecodeParams& p, cudaStream_t stream) {
    const dim3 grid(p.batch_size * p.num_heads_k, p.num_splits);
    switch (p.head_dim) {
        case 32:  decode_kernel<Elem, 32, R><<<grid, kDecodeWarps * 32, 0, stream>>>(p); break;
        case 64:  decode_kernel<Elem, 64, R><<<grid, kDecodeWarps * 32, 0, stream>>>(p); break;
        case 72:
        case 80:  decode_kernel<Elem, 80, R><<<grid, kDecodeWarps * 32, 0, stream>>>(p); break;  // 72: 9 lanes
        case 96:  decode_kernel<Elem, 96, R><<<grid, kDecodeWarps * 32, 0, stream>>>(p); break;
        case 128: decode_kernel<Elem, 128, R><<<grid, kDecodeWarps * 32, 0, stream>>>(p); break;
        case 256: decode_kernel<Elem, 256, R><<<grid, kDecodeWarps * 32, 0, stream>>>(p); break;
        default:  throw std::runtime_error("Unsupported head_dim for decode");
    }
}

template <typename Elem>
void launch_decode_r(DecodeParams& p, cudaStream_t stream) {
    const int rows = p.num_heads / p.num_heads_k * p.q_len;
    if (rows <= 1)      launch_decode_d<Elem, 1>(p, stream);
    else if (rows <= 2) launch_decode_d<Elem, 2>(p, stream);
    else if (rows <= 4) launch_decode_d<Elem, 4>(p, stream);
    else                launch_decode_d<Elem, 8>(p, stream);  // rows <= 8 (host check)
    if (p.num_splits > 1) {
        const index_t threads = index_t(p.batch_size) * p.num_heads * p.q_len * (p.head_dim / 8);
        decode_combine_kernel<Elem><<<static_cast<unsigned>((threads + 255) / 256), 256, 0, stream>>>(p);
    }
}

}  // namespace

// Query rows per KV head the decode kernel handles in one block.
int flash_attn_decode_max_rows() { return 8; }

constexpr int kMaxDecodeSplits = 128;
int flash_attn_decode_max_splits() { return kMaxDecodeSplits; }

// Key splits for a grid of `blocks` (B * H_kv) on `sms` SMs and a cache of
// max_len keys (no device read: CUDA-graph friendly): ~2 waves of blocks, at
// least 256 keys per split, at most kMaxDecodeSplits. Sized by the capacity:
// a large preallocated cache holding few keys gets more (empty) splits than
// it needs.
int flash_attn_decode_num_splits(int blocks, int sms, int max_len) {
    const int want = cdiv(2 * sms * 4, blocks);  // ~4 resident blocks of 4 warps per SM, 2 waves
    const int by_len = max(max_len / 256, 1);
    return max(1, min(min(want, by_len), kMaxDecodeSplits));
}

void flash_attn_decode(DecodeParams& p, bool bf16, cudaStream_t stream) {
    if (bf16) launch_decode_r<Bf16Elem>(p, stream);
    else      launch_decode_r<HalfElem>(p, stream);
}

}  // namespace flash_attn_legacy
