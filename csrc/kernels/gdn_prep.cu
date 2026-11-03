// ============================================================================
// The intra-chunk phase of the chunk gated delta rule (Qwen3.5-class Gated DeltaNet layers; docs/PLAN.md P6.10 c,
// docs/analysis/kernels/qwen35_delta_rule.md section 8, c2): GdnPrepParams. Per (batch row b, head h, chunk of 64
// tokens; the tail chunk zero-padded as transformers' reference pads it), from q, k [B, L, H, K] and v [B, L, H, V] in
// their dtype and g, beta [B, L, H] fp32, with qn / kn the l2-normalised q / k (q times scale) and c_i the cumulative
// sum of g in the chunk:
//
//   P_ij = exp(c_i - c_j) (j <= i, else 0)
//   M    = strict_lower(beta_i kn_i.kn_j P_ij),   T = (I + M)^-1
//   w    = T (beta e^c kn),   u = T (beta v)             ([.., 64, K], [.., 64, V])
//   A    = (qn kn^T) P (zeros above the diagonal)       ([.., 64, 64])
//   q~   = qn e^c,   k~ = kn e^(c_63 - c),   d = e^(c_63)
//
// and T itself (the backward's, gated_delta.py). fp32 throughout, but for c: its sums and differences in fp64 (under
// strong decays the fp32 differences of prefix sums lose their low bits). The reference normalises q / k before the
// Gram products; here the products of the raw rows are scaled after (fp32 rounding only).
//
// One block of 256 threads per chunk, grid (NC, H, B):
//   1. one pass over K in 64 x 16 sub-tiles of q and k (shared, column-major): both Gram matrices accumulated in
//      registers (a 4 x 4 patch of each per thread) and the rows' squared norms;
//   2. the cumulative decay, M into shared memory (diagonal 1), A to global memory;
//   3. T by forward substitution in place of M, a row per step (T_i,<i = -M_i,<i T_<i,<i: the order whose fp32
//      accuracy the plan checked), each entry's sum split over 4 threads;
//   4. a second pass over K and V in 64 x 16 sub-tiles: w and u as T times the scaled tile, q~ and k~.
// Static shared memory ~26 KB; 82-97 registers: two blocks per SM on every target (register-bound: 64K registers per
// SM). The next sub-tile is loaded into registers while the current one is used (no cp.async before SM 8.0).
// ============================================================================

#include "flash_attn_common.h"  // to_f

namespace flash_attn_legacy {

namespace {

constexpr int kChunk = 64;            // tokens per chunk (transformers' default chunk_size)
constexpr int kSub = 16;              // columns per sub-tile
constexpr int kThreads = 256;
constexpr int kLdT = kChunk + 4;      // M / T row stride: float4 rows in pass 2, few conflicts in the substitution
constexpr int kLdX = kSub + 4;        // pass 2's tile row stride (float4 rows, fewer bank conflicts)
constexpr int kBuf = 2 * kSub * kChunk;
static_assert(kChunk * kLdX <= kBuf, "pass 2's tile must fit the pass 1 buffer");

// Row r (of the chunk; zeros at or past `rows`), columns col..col+3 of a [64, *] slice with row stride `ld` as fp32
template <typename T>
__device__ __forceinline__ void load4(const T* __restrict__ src, index_t ld, int r, int rows, int col, float (&f)[4]) {
    const T* row = src + index_t(r) * ld + col;
    #pragma unroll
    for (int x = 0; x < 4; x++) f[x] = r < rows ? to_f(row[x]) : 0.0f;
}

template <typename T>
__global__ void __launch_bounds__(kThreads) gdn_prep_kernel(const GdnPrepParams p) {
    __shared__ __align__(16) float Ts[kChunk][kLdT];   // M, then T
    __shared__ __align__(16) float buf[kBuf];          // pass 1: q and k sub-tiles [kSub][64]; pass 2: [64][kLdX]
    // c in fp64: under strong decays (g ~ -20 per token, c ~ -700 by the chunk's end) the differences c_i - c_j of
    // fp32 prefix sums lose their low bits (an fp32 ulp of 700 is 6e-5); the 64 sums and the differences are cheap
    __shared__ double cs[kChunk];
    __shared__ float bs[kChunk], rq[kChunk], rk[kChunk];
    __shared__ float mrow[2][kChunk];                  // the substitution's current row of M, and the next

    const int tid = threadIdx.x;
    const int c = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
    const int K = p.k_dim, V = p.v_dim;
    const int t0 = c * kChunk;
    const int rows = min(kChunk, p.seqlen - t0);       // the chunk's tokens (the rest is padding)
    const T* q = static_cast<const T*>(p.q) + index_t(b) * p.q_batch_stride + index_t(t0) * p.q_seq_stride +
                 index_t(h) * p.q_head_stride;
    const T* k = static_cast<const T*>(p.k) + index_t(b) * p.k_batch_stride + index_t(t0) * p.k_seq_stride +
                 index_t(h) * p.k_head_stride;
    const T* v = static_cast<const T*>(p.v) + index_t(b) * p.v_batch_stride + index_t(t0) * p.v_seq_stride +
                 index_t(h) * p.v_head_stride;
    const index_t bhc = (index_t(b) * p.heads + h) * p.chunks + c;
    const index_t gi = (index_t(b) * p.seqlen + t0) * p.heads + h;

    // The cumulative decay (one warp, two tokens per lane) and beta; padded tokens have g = beta = 0
    if (tid < 32) {
        double g0 = tid < rows ? p.g[gi + index_t(tid) * p.heads] : 0.0;
        double g1 = tid + 32 < rows ? p.g[gi + index_t(tid + 32) * p.heads] : 0.0;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const double a0 = __shfl_up_sync(0xffffffffu, g0, o), a1 = __shfl_up_sync(0xffffffffu, g1, o);
            if (tid >= o) { g0 += a0; g1 += a1; }
        }
        g1 += __shfl_sync(0xffffffffu, g0, 31);
        cs[tid] = g0;
        cs[tid + 32] = g1;
    } else if (tid < 32 + kChunk) {
        const int i = tid - 32;
        bs[i] = i < rows ? p.beta[gi + index_t(i) * p.heads] : 0.0f;
    }

    // 1. The Gram matrices k k^T and q k^T of the raw rows, and the rows' squared norms
    const int lr = tid / 4, lc = (tid % 4) * 4;        // loads: row lr, columns lc..lc+3 of a sub-tile
    const int pr = (tid / 16) * 4, pc = (tid % 16) * 4;  // the thread's 4 x 4 patch: rows pr.., columns pc..
    float (*qs)[kChunk] = reinterpret_cast<float (*)[kChunk]>(buf);
    float (*ks)[kChunk] = reinterpret_cast<float (*)[kChunk]>(buf + kSub * kChunk);
    float kk[4][4], qk[4][4];
    #pragma unroll
    for (int a = 0; a < 4; a++) {
        #pragma unroll
        for (int e = 0; e < 4; e++) kk[a][e] = qk[a][e] = 0.0f;
    }
    float ssq = 0.0f;                                  // threads < 64: q row tid's; < 128: k row tid - 64's
    float fq[4], fk[4];
    load4(q, p.q_seq_stride, lr, rows, lc, fq);
    load4(k, p.k_seq_stride, lr, rows, lc, fk);
    for (int k0 = 0; k0 < K; k0 += kSub) {
        #pragma unroll
        for (int x = 0; x < 4; x++) { qs[lc + x][lr] = fq[x]; ks[lc + x][lr] = fk[x]; }
        __syncthreads();
        if (k0 + kSub < K) {
            load4(q, p.q_seq_stride, lr, rows, k0 + kSub + lc, fq);
            load4(k, p.k_seq_stride, lr, rows, k0 + kSub + lc, fk);
        }
        if (tid < 2 * kChunk) {
            const float (*src)[kChunk] = tid < kChunk ? qs : ks;
            const int r = tid % kChunk;
            #pragma unroll
            for (int x = 0; x < kSub; x++) ssq = fmaf(src[x][r], src[x][r], ssq);
        }
        #pragma unroll
        for (int x = 0; x < kSub; x++) {
            const float4 qa = *reinterpret_cast<const float4*>(&qs[x][pr]);
            const float4 ka = *reinterpret_cast<const float4*>(&ks[x][pr]);
            const float4 kb = *reinterpret_cast<const float4*>(&ks[x][pc]);
            const float qi[4] = {qa.x, qa.y, qa.z, qa.w}, ki[4] = {ka.x, ka.y, ka.z, ka.w};
            const float kj[4] = {kb.x, kb.y, kb.z, kb.w};
            #pragma unroll
            for (int a = 0; a < 4; a++) {
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    kk[a][e] = fmaf(ki[a], kj[e], kk[a][e]);
                    qk[a][e] = fmaf(qi[a], kj[e], qk[a][e]);
                }
            }
        }
        __syncthreads();  // the sub-tiles free for the next store
    }
    // transformers' l2norm: x * rsqrt(sum x^2 + 1e-6); q then times the scale
    if (tid < kChunk) rq[tid] = rsqrtf(ssq + 1e-6f) * p.scale;
    else if (tid < 2 * kChunk) rk[tid - kChunk] = rsqrtf(ssq + 1e-6f);
    __syncthreads();

    // 2. M (shared, diagonal 1, zeros above) and A (global)
    float* A = p.A + bhc * kChunk * kChunk;
    #pragma unroll
    for (int a = 0; a < 4; a++) {
        const int i = pr + a;
        const double ci = cs[i];
        const float bi = bs[i], rqi = rq[i], rki = rk[i];
        float arow[4];
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            const int j = pc + e;
            // exp(c_i - c_j), never e^c_i e^-c_j (inf * 0 once c < -88); none above the diagonal (it may overflow)
            const float pij = j <= i ? expf(static_cast<float>(ci - cs[j])) : 0.0f;
            const float rkj = rk[j];
            arow[e] = qk[a][e] * rqi * rkj * pij;
            Ts[i][j] = j < i ? bi * (kk[a][e] * rki * rkj) * pij : (j == i ? 1.0f : 0.0f);
        }
        *reinterpret_cast<float4*>(A + i * kChunk + pc) = make_float4(arow[0], arow[1], arow[2], arow[3]);
    }
    __syncthreads();

    // 3. T = (I + M)^-1 in place: row i from M's row i (mrow, copied a step ahead) and T's rows above it
    if (tid < kChunk) mrow[1][tid] = Ts[1][tid];
    __syncthreads();
    {
        const int j = tid / 4, part = tid % 4;
        for (int i = 1; i < kChunk; i++) {
            const float* m = mrow[i & 1];
            float s = 0.0f;
            if (j < i) {
                for (int mm = j + part; mm < i; mm += 4) s = fmaf(m[mm], Ts[mm][j], s);
            }
            s += __shfl_xor_sync(0xffffffffu, s, 1);
            s += __shfl_xor_sync(0xffffffffu, s, 2);
            if (j < i && part == 0) Ts[i][j] = -s;
            if (i + 1 < kChunk && tid < kChunk) mrow[(i + 1) & 1][tid] = Ts[i + 1][tid];
            __syncthreads();
        }
    }
    float* Tg = p.T + bhc * kChunk * kChunk;
    for (int e = tid; e < kChunk * kChunk; e += kThreads) Tg[e] = Ts[e / kChunk][e % kChunk];
    if (tid == 0) p.d[bhc] = expf(static_cast<float>(cs[kChunk - 1]));

    // 4. w = T (beta e^c kn), q~ and k~ over K's sub-tiles; then u = T (beta v) over V's
    float (*xs)[kLdX] = reinterpret_cast<float (*)[kLdX]>(buf);
    const int oc = tid % kSub, orow = tid / kSub;      // the products: column oc, rows orow + 16 x
    const float ec = expf(static_cast<float>(cs[lr])), el = expf(static_cast<float>(cs[kChunk - 1] - cs[lr]));
    const float br = bs[lr], rqr = rq[lr], rkr = rk[lr];
    const index_t row0 = bhc * kChunk;                 // the chunk's first row of the [.., 64, K | V] outputs
    const int nk = K / kSub, nt = nk + V / kSub;
    load4(q, p.q_seq_stride, lr, rows, lc, fq);
    load4(k, p.k_seq_stride, lr, rows, lc, fk);
    for (int n = 0; n < nt; n++) {
        const bool key_tile = n < nk;
        const int col0 = (key_tile ? n : n - nk) * kSub;
        float xv[4];
        if (key_tile) {
            float qo[4], ko[4];
            #pragma unroll
            for (int x = 0; x < 4; x++) {
                const float kn = fk[x] * rkr;
                qo[x] = fq[x] * rqr * ec;
                ko[x] = kn * el;
                xv[x] = kn * (br * ec);
            }
            const index_t o = (row0 + lr) * K + col0 + lc;
            *reinterpret_cast<float4*>(p.qt + o) = make_float4(qo[0], qo[1], qo[2], qo[3]);
            *reinterpret_cast<float4*>(p.kt + o) = make_float4(ko[0], ko[1], ko[2], ko[3]);
        } else {
            #pragma unroll
            for (int x = 0; x < 4; x++) xv[x] = fq[x] * br;  // fq holds v here
        }
        *reinterpret_cast<float4*>(&xs[lr][lc]) = make_float4(xv[0], xv[1], xv[2], xv[3]);
        __syncthreads();
        if (n + 1 < nk) {
            load4(q, p.q_seq_stride, lr, rows, (n + 1) * kSub + lc, fq);
            load4(k, p.k_seq_stride, lr, rows, (n + 1) * kSub + lc, fk);
        } else if (n + 1 < nt) {
            load4(v, p.v_seq_stride, lr, rows, (n + 1 - nk) * kSub + lc, fq);
        }
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll 4
        for (int m = 0; m < kChunk; m += 4) {  // T is zero above the diagonal: the full row, no bounds
            const float x0 = xs[m][oc], x1 = xs[m + 1][oc], x2 = xs[m + 2][oc], x3 = xs[m + 3][oc];
            #pragma unroll
            for (int x = 0; x < 4; x++) {  // 4 taps of a T row per shared load
                const float4 t = *reinterpret_cast<const float4*>(&Ts[orow + 16 * x][m]);
                acc[x] = fmaf(t.w, x3, fmaf(t.z, x2, fmaf(t.y, x1, fmaf(t.x, x0, acc[x]))));
            }
        }
        float* out = key_tile ? p.w : p.u;
        const int ld = key_tile ? K : V;
        #pragma unroll
        for (int x = 0; x < 4; x++) out[(row0 + orow + 16 * x) * ld + col0 + oc] = acc[x];
        __syncthreads();  // the tile free for the next store
    }
}

}  // namespace

// K and V multiples of 16 (the host checked); dtype: q / k / v's code (0 float16, 1 bfloat16, 2 float32)
void gdn_prep_launch(const GdnPrepParams& p, cudaStream_t stream) {
    if (p.batch == 0 || p.heads == 0 || p.chunks == 0) return;
    const dim3 grid(p.chunks, p.heads, p.batch);
    if (p.dtype == 0)      gdn_prep_kernel<half><<<grid, kThreads, 0, stream>>>(p);
    else if (p.dtype == 1) gdn_prep_kernel<__nv_bfloat16><<<grid, kThreads, 0, stream>>>(p);
    else                   gdn_prep_kernel<float><<<grid, kThreads, 0, stream>>>(p);
}

}  // namespace flash_attn_legacy
