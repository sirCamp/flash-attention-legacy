// ============================================================================
// The sequential scan of the chunk gated delta rule (Qwen3.5-class Gated DeltaNet
// layers; docs/PLAN.md P6.10 c, docs/analysis/kernels/qwen35_delta_rule.md, c1):
// GdnScanParams. Per (batch row b, head h), over the chunks c of 64 tokens, with
// S_0 the initial state (or 0):
//
//   h_c         = S_c                              (stored: [B, H, NC, K, V])
//   v_new_c     = u_c - w_c S_c                    (stored: [B, H, NC, 64, V])
//   S_{c+1}     = d_c S_c + k~_c^T v_new_c
//
// and the final state S_NC. The output (q~_c h_c + A_c v_new_c) is two batched
// matmuls in PyTorch, parallel over the chunks (the split of `fla`). fp32
// throughout, as transformers' reference.
//
// The columns of S are independent (every step multiplies S from the left), so
// a block owns a slice of VS columns of one head's state: grid (V / VS, H, B).
// Shared memory (<= 48 KB static, every arch): the state slice S [K][VS], v_new
// [64][VS], and one 64 x 32 tile of w_c or k~_c at a time (K sub-tiles of 32),
// so v_new accumulates over the sub-tiles and the state rows of a sub-tile are
// updated from it. 256 threads; a thread owns fixed (row, column) outputs.
// ============================================================================

#include "flash_attn_common.h"

namespace flash_attn_legacy {

namespace {

constexpr int kChunk = 64;     // tokens per chunk (transformers' default chunk_size)
constexpr int kSub = 32;       // K columns per streamed tile
constexpr int kThreads = 256;

// A 64 x 32 fp32 tile of a row-major [64, K] chunk matrix, columns k0..k0+31, as kTileVecs float4 per thread
constexpr int kTileVecs = kChunk * kSub / 4 / kThreads;

template <int K>
__device__ __forceinline__ void load_tile(float4 (&r)[kTileVecs], const float* src, int k0, int tid) {
    #pragma unroll
    for (int n = 0; n < kTileVecs; n++) {
        const int f = tid + n * kThreads, t = f / (kSub / 4), c4 = f % (kSub / 4);
        r[n] = __ldg(reinterpret_cast<const float4*>(src + index_t(t) * K + k0) + c4);
    }
}

__device__ __forceinline__ void store_tile(float (&tile)[kChunk][kSub], const float4 (&r)[kTileVecs], int tid) {
    #pragma unroll
    for (int n = 0; n < kTileVecs; n++) {
        const int f = tid + n * kThreads, t = f / (kSub / 4), c4 = f % (kSub / 4);
        reinterpret_cast<float4*>(&tile[t][0])[c4] = r[n];
    }
}

template <int K, int VS>
// No minimum-blocks hint: at (256, 2) ptxas takes up to 128 registers (96-124), leaving 2 blocks per SM where 75-96
// registers fit 3 (VS = 16: 128 blocks resident at once on the P100; 2.77 against 1.97 ms at 4k tokens)
__global__ void __launch_bounds__(kThreads) gdn_scan_kernel(const GdnScanParams p) {
    static_assert(K % kSub == 0 && kThreads % VS == 0 && kTileVecs * kThreads * 4 == kChunk * kSub, "tile mismatch");
    static_assert(kChunk % (kThreads / VS) == 0 && kSub % (kThreads / VS) == 0, "rows per pass: every row covered");
    constexpr int kRowsPerPass = kThreads / VS;          // rows a pass of the block covers
    constexpr int kOut = kChunk / kRowsPerPass;          // v_new outputs per thread
    constexpr int kSOut = kSub / kRowsPerPass;           // state outputs per thread and sub-tile
    constexpr int kSubs = K / kSub;                      // sub-tiles of w, then of k~, per chunk
    __shared__ float S[K][VS];
    __shared__ float vn[kChunk][VS];
    // Unpadded: the w phase reads rows (float4, a row per half-warp: broadcast), the k~ phase reads columns at two rows
    // per warp (adjacent banks): no conflicts either way
    __shared__ __align__(16) float tile[kChunk][kSub];

    const int tid = threadIdx.x, j = tid % VS, r0 = tid / VS;
    const int v0 = blockIdx.x * VS, h = blockIdx.y, b = blockIdx.z;
    const index_t bh = index_t(b) * p.heads + h;
    const int V = p.v_dim;
    // The tiles of a chunk in order: w's kSubs sub-tiles, then k~'s; the next one is prefetched into registers while
    // the current one is used (no cp.async before SM 8.0)
    auto tile_src = [&](int c, int n) {
        const index_t bhc = bh * p.chunks + c;
        return (n < kSubs ? p.w : p.kt) + bhc * kChunk * K;
    };
    float4 next[kTileVecs];

    for (int e = tid; e < K * VS; e += kThreads) {
        const int i = e / VS, jj = e % VS;
        S[i][jj] = p.s0 != nullptr ? p.s0[(bh * K + i) * V + v0 + jj] : 0.0f;
    }
    if (p.chunks > 0) load_tile<K>(next, tile_src(0, 0), 0, tid);
    __syncthreads();

    for (int c = 0; c < p.chunks; c++) {
        const index_t bhc = bh * p.chunks + c;
        // h_c = S_c
        float* hc = p.h + bhc * K * V + v0;
        for (int e = tid; e < K * VS; e += kThreads) {
            const int i = e / VS, jj = e % VS;
            hc[index_t(i) * V + jj] = S[i][jj];
        }
        const float* u = p.u + bhc * kChunk * V + v0;
        float acc[kOut];
        #pragma unroll
        for (int o = 0; o < kOut; o++) acc[o] = u[index_t(r0 + o * kRowsPerPass) * V + j];
        const float d = p.d[bhc];
        #pragma unroll 1
        for (int n = 0; n < 2 * kSubs; n++) {
            store_tile(tile, next, tid);
            // the next tile: this chunk's, or the next chunk's first
            if (n + 1 < 2 * kSubs) load_tile<K>(next, tile_src(c, n + 1), ((n + 1) % kSubs) * kSub, tid);
            else if (c + 1 < p.chunks) load_tile<K>(next, tile_src(c + 1, 0), 0, tid);
            __syncthreads();  // the tile (and, before the first k~ tile, v_new) visible
            const int k0 = (n % kSubs) * kSub;
            if (n < kSubs) {  // v_new = u - w S, accumulated over w's sub-tiles
                #pragma unroll 2
                for (int i = 0; i < kSub; i += 4) {
                    // one shared load of the state per tap for the thread's kOut rows, the tile 4 taps at a time
                    const float s0 = S[k0 + i][j], s1 = S[k0 + i + 1][j], s2 = S[k0 + i + 2][j], s3 = S[k0 + i + 3][j];
                    #pragma unroll
                    for (int o = 0; o < kOut; o++) {
                        const float4 wv = *reinterpret_cast<const float4*>(&tile[r0 + o * kRowsPerPass][i]);
                        float x = acc[o];
                        x = fmaf(-wv.x, s0, x); x = fmaf(-wv.y, s1, x); x = fmaf(-wv.z, s2, x); x = fmaf(-wv.w, s3, x);
                        acc[o] = x;
                    }
                }
                if (n == kSubs - 1) {
                    float* vg = p.v_new + bhc * kChunk * V + v0;
                    #pragma unroll
                    for (int o = 0; o < kOut; o++) {
                        const int t = r0 + o * kRowsPerPass;
                        vn[t][j] = acc[o];
                        vg[index_t(t) * V + j] = acc[o];
                    }
                }
            } else {  // S = d S + k~^T v_new for this sub-tile's state rows
                float a[kSOut];
                #pragma unroll
                for (int o = 0; o < kSOut; o++) a[o] = 0.0f;
                #pragma unroll 8
                for (int t = 0; t < kChunk; t++) {
                    const float vv = vn[t][j];  // one shared load for the thread's kSOut state rows
                    #pragma unroll
                    for (int o = 0; o < kSOut; o++) a[o] = fmaf(tile[t][r0 + o * kRowsPerPass], vv, a[o]);
                }
                #pragma unroll
                for (int o = 0; o < kSOut; o++) {
                    const int i = r0 + o * kRowsPerPass;
                    S[k0 + i][j] = fmaf(d, S[k0 + i][j], a[o]);
                }
            }
            __syncthreads();  // the tile buffer free for the next store; S / v_new complete
        }
    }
    for (int e = tid; e < K * VS; e += kThreads) {
        const int i = e / VS, jj = e % VS;
        p.s_final[(bh * K + i) * V + v0 + jj] = S[i][jj];
    }
}

template <int K>
void launch_vs(const GdnScanParams& p, int vs, cudaStream_t stream) {
    if (vs == 16) gdn_scan_kernel<K, 16><<<dim3(p.v_dim / 16, p.heads, p.batch), kThreads, 0, stream>>>(p);
    else          gdn_scan_kernel<K, 32><<<dim3(p.v_dim / 32, p.heads, p.batch), kThreads, 0, stream>>>(p);
}

}  // namespace

// vs: the state columns per block (16 or 32, the host chose; V % vs == 0); K 64 or 128 (the host checked)
void gdn_scan_launch(const GdnScanParams& p, int vs, cudaStream_t stream) {
    if (p.batch == 0 || p.heads == 0) return;
    if (p.k_dim == 64) launch_vs<64>(p, vs, stream);
    else               launch_vs<128>(p, vs, stream);
}

}  // namespace flash_attn_legacy
