// ============================================================================
// FP16 linear layer for the P100 (SM 6.0): HFMA2 with chunked accumulation
// (docs/PLAN.md P8). y [M, N] = x [M, K] W [N, K]^T (+ bias [N]), fp16 in and
// out, x and W K-contiguous (the nn.Linear layout).
//
// torch runs fp16 GEMMs on SM 6.0 with cuBLAS COMPUTE_32F, at about half the
// card's HFMA2 rate; COMPUTE_16F accumulates the whole K in fp16 (top-1 tokens
// flip). Here:
//   - inner products with HFMA2 paired along K: acc += half2(x[k], x[k+1]) *
//     half2(w[k], w[k+1]), each fp16 lane a chain of at most kChunk / 2 products;
//   - every kChunk = 64 K elements (chunks aligned to k = 0) both lanes are
//     widened to fp32 and added to an fp32 accumulator (a fold in fp32: two
//     lanes of e.g. 40000 do not overflow, as their fp16 sum would), and the
//     half2 accumulator restarts (the first K pair of a chunk is an HMUL2: no
//     reset moves); the epilogue adds the bias in fp32 and rounds to fp16 once.
// Structure: a register-blocked SGEMM where one "float" is one K pair (half2):
// block tile 128 x 128, 256 threads with 8 x 8 outputs each, stages of 32
// halves (16 K pairs), shared-memory double buffer (one __syncthreads per
// stage), global -> register -> shared staging (no cp.async on SM 6.0).
//
// Schedules (LinearPlan, chosen on the host by linear_pascal_plan):
//   data-parallel: one block per output tile, the whole K;
//   stream-K: gridDim.x blocks share the (tile, stage) units of the tiles
//     [tile0, tiles) in even, contiguous ranges (the host runs the full waves of
//     tiles but the last data-parallel and stream-K over the rest). A block
//     writes the tiles it covers whole directly. For a split tile EVERY
//     contributor (the one holding its first stages included) writes its fp32
//     partial to its own workspace slot, fences, and increments the tile's
//     counter once; the block that sees count == contributors - 1 (the "last
//     arriver") adds all partials in ascending contributor order (deterministic:
//     independent of who arrives last), runs the epilogue and resets the counter
//     to 0, so counters are 0 again after every completed launch. Fast path: the
//     first contributor, which usually arrives last (the tile is at the end of
//     its range), checks the counter before publishing and, if everyone else is
//     in, reduces from its registers in the same order (same bits, no workspace
//     round trip). No block ever waits for another: no co-residency requirement
//     (busy SMs, MPS, any grid size). Stream-K index math is 32-bit (the host
//     guarantees sk_blocks * units < 2^31). A segment boundary may fall inside a
//     chunk: that fp16 chain is cut earlier (never longer than kChunk / 2
//     products), never less accurate.
//
// Architecture guard: the body is SM 6.0 device code only (fp16 arithmetic runs
// at 1/64 of the fp32 rate on SM 6.1, SM 7.x has tensor cores); other device
// code gets a trapping stub, never launched (the host requires compute
// capability 6.0).
// ============================================================================

#include "flash_attn_common.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace flash_attn_legacy {

namespace {

constexpr int kTile = kLinearTile;                   // block tile: rows of x and of W
constexpr int kThreads = (kTile / 8) * (kTile / 8);  // 8 x 8 outputs per thread
constexpr int kStagePairs = 16;                      // K pairs per stage (32 halves)
// K elements per fp16 chain, then the fp32 fold. 64 (measured on the P100 against 128 and 32, docs/bench/linear.md):
// top-1 flips against FP32 1.1-1.3x stock FP16's (128: 1.3-1.7x), 4% slower than 128; 32 is 8% slower, not measurably better
constexpr int kChunk = 64;
constexpr int kStagesPerChunk = kChunk / (2 * kStagePairs);
constexpr int kStride = kTile + 4;                   // shared-memory row stride in half2 (rows stay 16-byte aligned)
constexpr int kLoads = kTile * 4 / kThreads;         // 16-byte global chunks per thread, stage and operand
// Stream-K: each block gets at least this many stages of work (fewer blocks than
// resident ones when the stream-K region is small; measured on the P100)
constexpr int kStreamKMinStages = 4;
static_assert(kChunk % (2 * kStagePairs) == 0, "the chunk must be a multiple of the stage depth");
static_assert(kLoads * kThreads == kTile * 4 && kThreads % 4 == 0, "tile / thread mismatch");

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 600

__device__ __forceinline__ uint32_t hfma2(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
    uint32_t d;
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}

// The two fp16 lanes widened and added in fp32 (an fp16 add overflows for two lanes > 32752)
__device__ __forceinline__ float fold(uint32_t h2) {
    const half2 v = *reinterpret_cast<const half2*>(&h2);
    return __low2float(v) + __high2float(v);
}

// Stream-K: first unit of block h (32-bit; the host guarantees blocks * units < 2^31)
__device__ __forceinline__ uint32_t sk_first_unit(uint32_t h, uint32_t units, uint32_t blocks) {
    return h * units / blocks;
}

// Block whose unit range [sk_first_unit(h), sk_first_unit(h + 1)) holds unit u (u < units):
// the largest h with h * units <= (u + 1) * blocks - 1
__device__ __forceinline__ uint32_t sk_block_of(uint32_t u, uint32_t units, uint32_t blocks) {
    return ((u + 1) * blocks - 1) / units;
}

#endif  // __CUDA_ARCH__ == 600

// kStreamK = false: one block per output tile (blockIdx.x, row-major tile order), tile0 unused.
// kStreamK = true: stream-K over the tiles [tile0, tiles).
template <bool kStreamK>
__global__ void __launch_bounds__(kThreads, 1)
linear_kernel(const LinearParams p, const int tile0) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 600
    const half* __restrict__ X = p.x;
    const half* __restrict__ W = p.w;
    const half* __restrict__ bias = p.bias;
    half* __restrict__ Y = p.y;
    const int M = p.M, N = p.N, K = p.K;

    __shared__ __align__(16) uint32_t As[2][kStagePairs][kStride];
    __shared__ __align__(16) uint32_t Bs[2][kStagePairs][kStride];

    constexpr int kThreadsN = kTile / 8;  // threads along N
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int tn = (lane & 7) | ((warp % (kThreadsN / 8)) << 3);
    const int tm = (lane >> 3) | ((warp / (kThreadsN / 8)) << 2);
    const int nt = (K + 2 * kStagePairs - 1) / (2 * kStagePairs);  // stages over the whole K
    const int kc = (tid & 3) * 8;  // same K offset for every 16-byte chunk of this thread (kThreads % 4 == 0)
    const int s0 = ((tid & 3) * 4) * kStride + (tid >> 2);

    float facc[8][8];

    // Accumulates the stages [st0, st1) of tile (m0, n0) into facc
    auto segment = [&](int m0, int n0, int st0, int st1) {
        // 32-bit element offsets (the host checks M * K, N * K < 2^31): 4 registers instead of
        // 4 64-bit pointers, so ptxas does not rematerialise the row * K products in the main loop
        int oa[kLoads], ob[kLoads];
        uint32_t okA = 0, okB = 0;
        #pragma unroll
        for (int l = 0; l < kLoads; ++l) {
            const int r = (tid + l * kThreads) >> 2;
            okA |= (m0 + r < M) ? (1u << l) : 0u;
            oa[l] = min(m0 + r, M - 1) * K + kc;
        }
        #pragma unroll
        for (int l = 0; l < kLoads; ++l) {
            const int r = (tid + l * kThreads) >> 2;
            okB |= (n0 + r < N) ? (1u << l) : 0u;
            ob[l] = min(n0 + r, N - 1) * K + kc;
        }
        uint4 ra[kLoads], rb[kLoads];
        auto gload = [&](int t) {
            const int ko = t * 2 * kStagePairs;
            const bool kok = ko + kc < K;
            #pragma unroll
            for (int l = 0; l < kLoads; ++l) {
                ra[l] = (kok && (okA >> l & 1)) ? __ldg(reinterpret_cast<const uint4*>(X + (oa[l] + ko)))
                                                : make_uint4(0, 0, 0, 0);
            }
            #pragma unroll
            for (int l = 0; l < kLoads; ++l) {
                rb[l] = (kok && (okB >> l & 1)) ? __ldg(reinterpret_cast<const uint4*>(W + (ob[l] + ko)))
                                                : make_uint4(0, 0, 0, 0);
            }
        };
        auto sstore = [&](int buf) {
            uint32_t* a = &As[buf][0][0] + s0;
            uint32_t* b = &Bs[buf][0][0] + s0;
            #pragma unroll
            for (int l = 0; l < kLoads; ++l) {
                const int o = l * (kThreads / 4);
                a[o] = ra[l].x; a[o + kStride] = ra[l].y; a[o + 2 * kStride] = ra[l].z; a[o + 3 * kStride] = ra[l].w;
            }
            #pragma unroll
            for (int l = 0; l < kLoads; ++l) {
                const int o = l * (kThreads / 4);
                b[o] = rb[l].x; b[o + kStride] = rb[l].y; b[o + 2 * kStride] = rb[l].z; b[o + 3 * kStride] = rb[l].w;
            }
        };
        uint32_t acc[8][8];
        // One stage of 16 K pairs from buffer buf; kFirst: its first K pair starts a chunk (HMUL2)
        auto stage = [&](int buf, auto first_tag) {
            constexpr bool kFirst = decltype(first_tag)::value;
            #pragma unroll
            for (int kp = 0; kp < kStagePairs; ++kp) {
                const uint4 a0 = *reinterpret_cast<const uint4*>(&As[buf][kp][tm * 4]);
                const uint4 a1 = *reinterpret_cast<const uint4*>(&As[buf][kp][tm * 4 + kTile / 2]);
                const uint4 b0 = *reinterpret_cast<const uint4*>(&Bs[buf][kp][tn * 4]);
                const uint4 b1 = *reinterpret_cast<const uint4*>(&Bs[buf][kp][tn * 4 + kTile / 2]);
                const uint32_t a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
                const uint32_t b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
                // j-major order of the 8 x 8 HFMA2 block (the order the kernel was tuned with)
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        if (kFirst && kp == 0) acc[i][j] = hmul2(a[i], b[j]);
                        else                   acc[i][j] = hfma2(a[i], b[j], acc[i][j]);
                    }
                }
            }
        };
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) facc[i][j] = 0.f;
        }

        gload(st0);
        sstore(0);
        __syncthreads();
        for (int t = st0; t < st1; ++t) {
            const int buf = (t - st0) & 1;
            gload(t + 1 < st1 ? t + 1 : t);  // unconditional (the last stage reloads itself): ptxas hoists the loads
            if (t % kStagesPerChunk == 0 || t == st0) stage(buf, std::true_type{});
            else                                      stage(buf, std::false_type{});
            sstore(buf ^ 1);  // safe: buf ^ 1 was last read before the previous barrier
            if ((t + 1) % kStagesPerChunk == 0 || t + 1 == st1) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) facc[i][j] += fold(acc[i][j]);
                }
            }
            __syncthreads();
        }
    };

    // fp32 (+ bias) -> fp16 output tile
    auto store_out = [&](int m0, int n0) {
        const bool vec = (N & 3) == 0;
        #pragma unroll
        for (int jg = 0; jg < 2; ++jg) {
            const int col = n0 + tn * 4 + jg * (kTile / 2);
            float bv[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) bv[j] = (bias != nullptr && col + j < N) ? __half2float(bias[col + j]) : 0.f;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = m0 + tm * 4 + (i & 3) + (i >> 2) * (kTile / 2);
                if (row >= M) continue;
                half h[4];
                #pragma unroll
                for (int j = 0; j < 4; ++j) h[j] = __float2half_rn(facc[i][jg * 4 + j] + bv[j]);
                half* yp = Y + size_t(row) * N + col;
                if (vec && col + 3 < N) {
                    uint2 v;
                    v.x = uint32_t(__half_as_ushort(h[0])) | (uint32_t(__half_as_ushort(h[1])) << 16);
                    v.y = uint32_t(__half_as_ushort(h[2])) | (uint32_t(__half_as_ushort(h[3])) << 16);
                    *reinterpret_cast<uint2*>(yp) = v;
                } else {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        if (col + j < N) yp[j] = h[j];
                    }
                }
            }
        }
    };

    // Adds the fp32 partial tile at w (row-major [kTile, kTile]) to facc
    auto add_partial = [&](const float* w) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int r = tm * 4 + (i & 3) + (i >> 2) * (kTile / 2);
            #pragma unroll
            for (int jg = 0; jg < 2; ++jg) {
                const float4 v = __ldcg(reinterpret_cast<const float4*>(w + r * kTile + tn * 4 + jg * (kTile / 2)));
                facc[i][jg * 4] += v.x; facc[i][jg * 4 + 1] += v.y;
                facc[i][jg * 4 + 2] += v.z; facc[i][jg * 4 + 3] += v.w;
            }
        }
    };

    const int tiles_n = (N + kTile - 1) / kTile, tiles_m = (M + kTile - 1) / kTile;
    if (!kStreamK) {
        const int t = blockIdx.x;
        const int m0 = (t / tiles_n) * kTile, n0 = (t % tiles_n) * kTile;
        segment(m0, n0, 0, nt);
        store_out(m0, n0);
        return;
    }

    // ---- stream-K over the tiles [tile0, tiles_m * tiles_n); unit = one stage of one tile
    constexpr int kTileFloats = kTile * kTile;
    __shared__ int s_last;
    const uint32_t Q = uint32_t(nt);
    const uint32_t U = uint32_t(tiles_m * tiles_n - tile0) * Q;
    const uint32_t g = blockIdx.x, G = gridDim.x;  // host: G <= U, every block has >= 1 unit
    const uint32_t u0 = sk_first_unit(g, U, G), u1 = sk_first_unit(g + 1, U, G);
    uint32_t u = u0;
    while (u < u1) {
        const uint32_t tl = u / Q, q0 = u - tl * Q;
        const uint32_t q1 = min(Q, q0 + (u1 - u));
        const int t = tile0 + int(tl);
        const int m0 = (t / tiles_n) * kTile, n0 = (t % tiles_n) * kTile;
        segment(m0, n0, int(q0), int(q1));
        if (q0 == 0 && q1 == Q) {  // the whole tile in this block
            store_out(m0, n0);
        } else {
            const uint32_t tb = tl * Q;
            const uint32_t hf = sk_block_of(tb, U, G), hl = sk_block_of(tb + Q - 1, U, G);  // contributors hf..hl
            bool done = false;
            // Owner fast path: hf (whose partial comes first in the reduction order) usually arrives last, as the
            // tile is at the END of its range. If every other contributor has arrived (nobody else can change the
            // counter then), it reduces straight from its registers: same order, same bits, no workspace round trip.
            if (g == hf) {
                if (tid == 0) {
                    const int last = *reinterpret_cast<volatile int*>(p.counters + tl) == int(hl - hf);
                    if (last) {
                        __threadfence();     // the counter observation before the partial reads below
                        p.counters[tl] = 0;  // self-reset
                    }
                    s_last = last;
                }
                __syncthreads();
                if (s_last) {
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) facc[i][j] = 0.f + facc[i][j];  // as the slow path (0 + p_hf + ...): -0 -> +0
                    }
                    for (uint32_t h = hf + 1; h <= hl; ++h) {  // h > hf enter the tile with their first segment: slot 0
                        add_partial(p.ws + size_t(h) * 2 * kTileFloats);
                    }
                    store_out(m0, n0);
                    done = true;
                }
            }
            if (!done) {
                // Publish this block's partial. Slot 0: the first segment of the block's range, slot 1: the last
                // (a block contributes to at most two split tiles: the one it starts in and the one it ends in).
                float* w = p.ws + (size_t(g) * 2 + (u == u0 ? 0 : 1)) * kTileFloats;
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int r = tm * 4 + (i & 3) + (i >> 2) * (kTile / 2);
                    #pragma unroll
                    for (int jg = 0; jg < 2; ++jg) {
                        __stcg(reinterpret_cast<float4*>(w + r * kTile + tn * 4 + jg * (kTile / 2)),
                               make_float4(facc[i][jg * 4], facc[i][jg * 4 + 1], facc[i][jg * 4 + 2],
                                           facc[i][jg * 4 + 3]));
                    }
                }
                __threadfence();  // every thread's partial visible device-wide before the counter increment
                __syncthreads();
                if (tid == 0) {
                    const int old = atomicAdd(p.counters + tl, 1);
                    const int last = old == int(hl - hf);
                    if (last) {
                        p.counters[tl] = 0;  // self-reset (every contributor has arrived)
                        __threadfence();
                    }
                    s_last = last;
                }
                __syncthreads();
                if (s_last) {  // last arriver: deterministic sum in contributor order, then the epilogue
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) facc[i][j] = 0.f;
                    }
                    for (uint32_t h = hf; h <= hl; ++h) {
                        const int slot = sk_first_unit(h, U, G) < tb ? 1 : 0;  // only hf enters after its first segment
                        add_partial(p.ws + (size_t(h) * 2 + slot) * kTileFloats);
                    }
                    store_out(m0, n0);
                }
            }
        }
        u += q1 - q0;
    }
#else
    // Never launched: the host requires compute capability 6.0
    __trap();
#endif  // __CUDA_ARCH__ == 600
}

// Stream-K blocks resident at once on the current device (SMs x resident blocks of
// the stream-K kernel per SM): a constant of the device and the build, cached per device.
int linear_resident_blocks() {
    constexpr int kMaxDevices = 64;
    static std::atomic<int> cache[kMaxDevices];  // static storage: zero (not queried yet)
    int dev = 0;
    FLASH_ATTN_CHECK_CUDA(cudaGetDevice(&dev));
    if (dev < kMaxDevices) {
        const int cached = cache[dev].load(std::memory_order_relaxed);
        if (cached > 0) return cached;
    }
    int sms = 0, per_sm = 0;
    FLASH_ATTN_CHECK_CUDA(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    FLASH_ATTN_CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, linear_kernel<true>, kThreads, 0));
    if (per_sm < 1) throw std::runtime_error("flash_attn_legacy: the linear kernel cannot be resident on this GPU");
    const int resident = sms * per_sm;
    if (dev < kMaxDevices) cache[dev].store(resident, std::memory_order_relaxed);
    return resident;
}

}  // namespace

// The schedule of y [M, N] = x [M, K] W^T on the current device. Hybrid: the full
// waves of tiles but the last data-parallel, stream-K over the remaining resident
// .. 2 * resident - 1 tiles (none when the tiles fill whole waves).
// FLASH_ATTN_LEGACY_LINEAR_STREAM_K=0: data-parallel only (testing: the reference
// of the stream-K tiles).
LinearPlan linear_pascal_plan(int M, int N, int K) {
    LinearPlan plan;
    plan.tiles = cdiv(M, kTile) * cdiv(N, kTile);
    plan.tile0 = plan.tiles;
    plan.sk_blocks = 0;
    plan.resident = linear_resident_blocks();
    const char* env = std::getenv("FLASH_ATTN_LEGACY_LINEAR_STREAM_K");
    if (env != nullptr && std::strcmp(env, "0") == 0) return plan;
    const int64_t resident = plan.resident;
    const int64_t tiles = plan.tiles;
    const int64_t stages = cdiv(K, 2 * kStagePairs);
    const int64_t tile0 = tiles % resident ? std::max<int64_t>(0, (tiles / resident - 1) * resident) : tiles;
    if (tile0 < tiles) {
        const int64_t units = (tiles - tile0) * stages;
        const int64_t blocks = std::min(resident, units / kStreamKMinStages);
        // Below one wave stream-K pays only with more blocks than tiles; the 32-bit unit
        // math of the kernel needs blocks * units < 2^31
        if (blocks >= 1 && (tiles >= resident || blocks > tiles) && blocks * units < (int64_t(1) << 31)) {
            plan.tile0 = int(tile0);
            plan.sk_blocks = int(blocks);
        }
    }
    return plan;
}

// p.ws / p.counters: plan.sk_blocks * 2 * kLinearTile^2 floats and plan.tiles - plan.tile0
// zeroed counters when plan.tile0 < plan.tiles (the host op allocates them).
void linear_pascal_launch(const LinearParams& p, const LinearPlan& plan, cudaStream_t stream) {
    if (plan.tile0 > 0) linear_kernel<false><<<plan.tile0, kThreads, 0, stream>>>(p, 0);
    if (plan.tile0 < plan.tiles) linear_kernel<true><<<plan.sk_blocks, kThreads, 0, stream>>>(p, plan.tile0);
}

}  // namespace flash_attn_legacy
