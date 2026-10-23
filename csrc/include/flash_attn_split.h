// ============================================================================
// Flash Attention — key splits: the partial state of a split and their merge,
// shared by the decode kernel (docs/analysis/kernels/decode_design.md) and the
// forward with key splits (spec_decode_design.md).
//
// With more than one split, each split writes per (batch row, query head,
// query) its normalized O (fp32) and its lse in the log2 domain;
// split_combine_kernel merges the splits into the output (q's dtype) and the
// lse (natural log), with the same decode_merge the kernels use inside
// (SplitCombineParams in flash_attn_common.h; host entry flash_attn_split_combine).
// ============================================================================
#pragma once

#include "flash_attn_common.h"

#include <cuda_bf16.h>

namespace flash_attn_legacy {

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
// Branchless: a non-finite logit or V row propagates whether or not the state is
// still empty (only non-finite cache values are affected).
__device__ __forceinline__ void decode_merge(float& m, float& l, float (&o)[8], float m2, float l2,
                                             const float (&o2)[8]) {
    const float mn = fmaxf(m, m2);
    const float ms = mn == -INFINITY ? 0.0f : mn;  // both empty: a1 = a2 = 0, no (-inf) - (-inf); no branch
    const float a1 = exp2f(m - ms), a2 = exp2f(m2 - ms);
    l = l * a1 + l2 * a2;
    #pragma unroll
    for (int i = 0; i < 8; i++) o[i] = o[i] * a1 + o2[i] * a2;
    m = mn;
}

// Merge of the splits: L lanes (num_splits rounded up to a power of 2, at most
// 32) per 8 columns of a (row, head, query), lane s merging splits s, s + L, ...
// (their loads in flight together), then the L lanes by shuffles; each split's
// normalized (O_s, lse_s in log2) is the state (m = lse_s, l = 1, O_s).
template <typename Elem>
__global__ void split_combine_kernel(SplitCombineParams p, int L) {
    const int vecs = p.head_dim / 8;
    const index_t rows = index_t(p.batch_size) * p.num_heads * p.q_len;
    const index_t t = index_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const index_t idx = t / L;  // uniform over the L lanes (aligned groups of a warp)
    const int lane = static_cast<int>(t % L);
    const bool valid = idx < rows * vecs;  // the others still take part in the shuffles (whole warps)
    const index_t row = idx / vecs;
    const int c = static_cast<int>(idx % vecs);
    float m = -INFINITY, l = 0.0f, o[8] = {};
    for (int s = lane; valid && s < p.num_splits; s += L) {
        const index_t prow = index_t(s) * rows + row;
        const float4* src = reinterpret_cast<const float4*>(p.o_part + prow * p.head_dim + 8 * c);
        const float4 lo = src[0], hi = src[1];
        const float o2[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
        decode_merge(m, l, o, p.lse_part[prow], 1.0f, o2);
    }
    for (int off = L / 2; off > 0; off >>= 1) {
        const float m2 = __shfl_xor_sync(0xffffffffu, m, off);
        const float l2 = __shfl_xor_sync(0xffffffffu, l, off);
        float o2[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) o2[i] = __shfl_xor_sync(0xffffffffu, o[i], off);
        decode_merge(m, l, o, m2, l2, o2);
    }
    if (!valid || lane != 0) return;
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

template <typename Elem>
void launch_split_combine(const SplitCombineParams& p, cudaStream_t stream) {
    int L = 1;
    while (L < p.num_splits && L < 32) L *= 2;
    const index_t threads = index_t(p.batch_size) * p.num_heads * p.q_len * (p.head_dim / 8) * L;
    split_combine_kernel<Elem><<<static_cast<unsigned>((threads + 255) / 256), 256, 0, stream>>>(p, L);
}

}  // namespace flash_attn_legacy
