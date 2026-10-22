// ============================================================================
// Flash Attention v2 — Forward Kernel for Pascal (SM 6.0/6.1), CUDA cores
// (also runs on newer GPUs with FLASH_ATTN_LEGACY_FORCE_ARCH=pascal).
//
// Key design (FA2-style, docs/analysis/kernels/p2_8_design.md):
//   - the band forward of flash_attn_fwd_band.h (K/V loop, masking, online
//     softmax, batched/varlen kernels), with fp32 FMA register micro-tiles
//   - each band of 16 query rows belongs to Tile::kSplit warps, one per slice of
//     d / kSplit columns; lane = 8 * rg + cg holds rows 4rg..4rg+3, keys cg + 8j
//     (j < Bc/8) and O columns 4cg + 32i..+3 of its slice (i < d/kSplit/32).
//     With kSplit > 1 the warps of a band sum their partial S in shared memory
//     (one named barrier per band and block) and run the same softmax
//   - Q, K, V converted to fp32 once, when stored to shared memory: the inner
//     products run without conversions (on sm_6x a half->float conversion
//     costs about four FMAs) and without fp16 products (no overflow)
// ============================================================================

#include "flash_attn_fwd_band.h"

#include <cstring>

namespace flash_attn_legacy {

struct PascalFwd {
template <typename Tile>
static __device__ __forceinline__ void run(const FwdBlockArgs& a) {
    constexpr int Br = Tile::Br;
    constexpr int Bc = Tile::Bc;
    constexpr int d  = Tile::d;
    constexpr int NUM_THREADS = Tile::kNumWarps * 32;
    constexpr int LDF = Tile::kLdF;
    constexpr int LDP = Tile::kLdP;
    constexpr int kSplit = Tile::kSplit;
    constexpr int DS = d / kSplit;  // columns of this warp's slice
    constexpr int TN = Bc / 8;      // keys per lane
    constexpr int TD = DS / 32;     // float4 column groups of O per lane

    extern __shared__ __align__(128) char smem_raw[];  // aligned for 128-bit accesses
    float* smem_Q = reinterpret_cast<float*>(smem_raw);
    float* smem_K = smem_Q + Br * LDF;
    float* smem_V = smem_K + Bc * LDF;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    float* smem_Pt = smem_V + Bc * LDF + warp_id * Bc * LDP;  // this warp's P^T [Bc, LDP]
    const int band = warp_id / kSplit, part = warp_id % kSplit;
    const int col0 = part * DS;               // first column of this warp's slice
    // partial S of the band's warps (kSplit > 1): [kSplit][4][TN][32 lanes] per
    // band, lane-major (each lane reads back only its own slots: no bank conflicts)
    float* smem_S = smem_V + Bc * LDF + Tile::kNumWarps * Bc * LDP + (kSplit > 1 ? band * kSplit * 4 * TN * 32 : 0);

    const int rg = lane >> 3;                 // row group: rows 4rg..4rg+3 of the band
    const int cg = lane & 7;                  // column group (keys cg + 8j, O cols col0 + 4cg + 32i)
    const int band_row0 = band * kBandRows;
    const bool band_active = band_row0 < a.valid_q;
    const int row0 = band_row0 + 4 * rg;      // first of this lane's rows, within the block

    float4 O[4][TD];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int t = 0; t < TD; t++) O[i][t] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    BandSoftmax<4, 8> softmax;
    softmax.init(a.sink_log2);

    auto step = [&](auto mask_tag, int kv_start) {
        constexpr bool kMask = decltype(mask_tag)::value;

        // S = Q K^T for 4 rows x TN keys, k in chunks of 4 (float4 reads: Q rows
        // broadcast within the 8 lanes of a row group, the 8 K rows in different
        // bank groups thanks to the padded stride)
        float s[4][TN];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < TN; j++) s[i][j] = 0.0f;
        }
        #pragma unroll 4
        for (int k = col0; k < col0 + DS; k += 4) {
            float4 q[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) q[i] = *reinterpret_cast<const float4*>(smem_Q + (row0 + i) * LDF + k);
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                const float4 kk = *reinterpret_cast<const float4*>(smem_K + (cg + 8 * j) * LDF + k);
                // One K component over the 4 rows at a time: it stays in the operand
                // reuse cache, so sm_6x register bank conflicts no longer depend on
                // the register allocation (it swung d=96 / d=128 by +-30%, P4.1)
                #pragma unroll
                for (int i = 0; i < 4; i++) s[i][j] = fmaf(q[i].x, kk.x, s[i][j]);
                #pragma unroll
                for (int i = 0; i < 4; i++) s[i][j] = fmaf(q[i].y, kk.y, s[i][j]);
                #pragma unroll
                for (int i = 0; i < 4; i++) s[i][j] = fmaf(q[i].z, kk.z, s[i][j]);
                #pragma unroll
                for (int i = 0; i < 4; i++) s[i][j] = fmaf(q[i].w, kk.w, s[i][j]);
            }
        }
        if constexpr (kSplit > 1) {  // S = sum of the band's partial S, same order in every warp
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) smem_S[((part * 4 + i) * TN + j) * 32 + lane] = s[i][j];
            }
            band_barrier(band, kSplit * 32);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    float sum = 0.0f;
                    #pragma unroll
                    for (int p = 0; p < kSplit; p++) sum += smem_S[((p * 4 + i) * TN + j) * 32 + lane];
                    s[i][j] = sum;
                }
            }
            // smem_S is rewritten only in the next K/V block, after fwd_band_loop's
            // block-wide barrier (its contract)
        }
        fwd_logits<kMask>(a, s, [&](int i) { return a.q_row_start + row0 + i; },
                          [&](int j) { return kv_start + cg + 8 * j; });
        float alpha[4];
        softmax.update(s, alpha);

        // P^T for the band: key-major, so P @ V reads one float4 per key
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            *reinterpret_cast<float4*>(smem_Pt + (cg + 8 * j) * LDP + 4 * rg) =
                make_float4(s[0][j], s[1][j], s[2][j], s[3][j]);
        }
        __syncwarp();

        // O = O * alpha + P V
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int t = 0; t < TD; t++) {
                O[i][t].x *= alpha[i]; O[i][t].y *= alpha[i]; O[i][t].z *= alpha[i]; O[i][t].w *= alpha[i];
            }
        }
        #pragma unroll 4
        for (int c = 0; c < Bc; c++) {  // no p == 0 skip: 0 * inf must give NaN
            const float4 p = *reinterpret_cast<const float4*>(smem_Pt + c * LDP + 4 * rg);
            const float pr[4] = {p.x, p.y, p.z, p.w};
            #pragma unroll
            for (int t = 0; t < TD; t++) {
                const float4 v = *reinterpret_cast<const float4*>(smem_V + c * LDF + col0 + 4 * cg + 32 * t);
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    O[i][t].x = fmaf(pr[i], v.x, O[i][t].x);
                    O[i][t].y = fmaf(pr[i], v.y, O[i][t].y);
                    O[i][t].z = fmaf(pr[i], v.z, O[i][t].z);
                    O[i][t].w = fmaf(pr[i], v.w, O[i][t].w);
                }
            }
        }
        __syncwarp();  // P^T reads done before the next block overwrites it
    };

    fwd_band_run<Tile, LDF>(a, band_row0, band_active, smem_Q, smem_K, smem_V, step);
    if (!band_active) return;

    float inv[4], lse[4];
    softmax.finalize(inv, lse);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const int row = row0 + i;
        if (row >= a.valid_q) break;
        if (cg == 0 && part == 0) a.L[row * a.l_stride] = lse[i];
        #pragma unroll
        for (int t = 0; t < TD; t++) {
            const int col = col0 + 4 * cg + 32 * t;
            if (col < a.head_dim) {  // not a zero-padded column (d=72/80)
                const half2 lo = __floats2half2_rn(O[i][t].x * inv[i], O[i][t].y * inv[i]);
                const half2 hi = __floats2half2_rn(O[i][t].z * inv[i], O[i][t].w * inv[i]);
                uint2 packed;
                memcpy(&packed.x, &lo, 4);
                memcpy(&packed.y, &hi, 4);
                *reinterpret_cast<uint2*>(a.O + row * a.o_stride + col) = packed;
            }
        }
    }
}
};

// ---------------------------------------------------------------------------
// Host side: tile per head_dim (TilePascal_*, flash_attn_common.h), launched
// with the shared launch_fwd / launch_fwd_varlen.
// ---------------------------------------------------------------------------
template <typename F>
static void dispatch_pascal_fwd_tile(int head_dim, F&& f) {
    const int limit = max_dynamic_smem_per_block();
    bool ok = false;
    switch (head_dim) {
        case 32:  ok = with_first_fitting_tile<TilePascal_d32>(limit, f); break;
        case 64:  ok = with_first_fitting_tile<TilePascal_d64>(limit, f); break;
        case 72:
        case 80:
        case 96:  ok = with_first_fitting_tile<TilePascal_d96>(limit, f); break;
        // 72/80 zero-padded (CUDA-core tiles need d % 32 == 0): fwd N=4096 P100 22.7 ms (SDPA 23.2-24.1)
        case 128: ok = with_first_fitting_tile<TilePascal_d128>(limit, f); break;
        case 256: ok = with_first_fitting_tile<TilePascal_d256>(limit, f); break;
        default:  throw std::runtime_error("Unsupported head_dim for the CUDA-core forward");
    }
    if (!ok) throw_no_tile("CUDA-core forward", head_dim, limit);
}

void flash_attn_fwd_pascal(FlashAttnParams& params, cudaStream_t stream) {
    dispatch_pascal_fwd_tile(params.head_dim, [&](auto tile) {
        launch_fwd<decltype(tile), PascalFwd>(params, stream);
    });
}

void flash_attn_fwd_pascal_varlen(FlashAttnVarlenParams& params,
                                  int total_q_blocks, cudaStream_t stream) {
    dispatch_pascal_fwd_tile(params.head_dim, [&](auto tile) {
        launch_fwd_varlen<decltype(tile), PascalFwd>(params, total_q_blocks, stream);
    });
}

// Q rows per block of the forward tile selected for head_dim.
int flash_attn_fwd_pascal_block_rows(int head_dim) {
    int rows = 0;
    dispatch_pascal_fwd_tile(head_dim, [&](auto tile) { rows = decltype(tile)::Br; });
    return rows;
}

}  // namespace flash_attn_legacy
