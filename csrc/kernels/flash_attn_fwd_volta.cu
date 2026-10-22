// ============================================================================
// Flash Attention v2 — Forward Kernel for Volta (SM 7.0) and Turing (SM 7.5)
// Uses WMMA tensor cores for both Q@K^T and P@V matmuls (16×16×16 FP16)
//
// Key design (FA2-style, docs/analysis/kernels/p2_8_design.md):
//   - the band forward of flash_attn_fwd_band.h (K/V loop, masking, online
//     softmax, batched/varlen kernels), with WMMA for S = Q @ K^T and O += P @ V
//   - each warp owns a band of 16 query rows, O in WMMA accumulator fragments
//
// Architecture guard:
//   WMMA requires SM >= 7.0. This file is compiled for SM 6.x too
//   (gencode flags), but the body is #ifdef'd to a stub for SM < 7.0.
//   Runtime dispatch in flash_attn_legacy.cpp ensures the Volta kernel is
//   never launched on Pascal.
// ============================================================================

#include "flash_attn_fwd_band.h"
#include "flash_attn_wmma.h"

#include <cstring>

namespace flash_attn_legacy {

// Guard all device code that uses WMMA intrinsics.
// Host compilation: __CUDA_ARCH__ is undefined → condition true → real code.
// SM 6.x device compilation: __CUDA_ARCH__ < 700 → stub.
// SM 7.0 device compilation: __CUDA_ARCH__ >= 700 → real code.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700

// ---------------------------------------------------------------------------
// WMMA body of the shared band forward (flash_attn_fwd_band.h). Warp w owns the
// query rows [16w, 16w+16) of the block, with O in WMMA accumulator fragments.
//
// Per warp, S goes through a [16, LDS] float staging tile (the WMMA accumulator
// layout is opaque): two lanes per row (lane = 2*row + h) read interleaved float4
// chunks, and P is written back in place as half (row stride 2*LDS) for the
// P @ V fragments. To rescale O rows without assuming the accumulator layout,
// each lane learns once which row each of its fragment elements belongs to
// (wmma_acc_layout).
// ---------------------------------------------------------------------------
struct VoltaFwd {
template <typename Tile>
static __device__ __forceinline__ void run(const FwdBlockArgs& a) {
    constexpr int Br = Tile::Br;
    constexpr int Bc = Tile::Bc;
    constexpr int d  = Tile::d;
    constexpr int NUM_THREADS = Tile::kNumWarps * 32;
    constexpr int LDH = Tile::kLdH;
    constexpr int LDS = Tile::kLdS;
    constexpr int T = 16;
    static_assert(T == kBandRows, "one WMMA row tile per band");
    constexpr int S_TILES = Bc / T;   // S fragments per warp
    constexpr int O_TILES = d / T;    // O fragments per warp
    constexpr int CHUNKS = Bc / 8;    // float4 chunks of S per lane (two lanes per row)
    using Acc = wmma::fragment<wmma::accumulator, T, T, T, float>;
    constexpr int NE = kAccElems;

    extern __shared__ __align__(128) char smem_raw[];  // WMMA needs 32-byte alignment
    half*  smem_Q = reinterpret_cast<half*>(smem_raw);
    half*  smem_K = smem_Q + Br * LDH;
    half*  smem_V = smem_K + Bc * LDH;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    float* stage = reinterpret_cast<float*>(smem_V + Bc * LDH) + warp_id * (Tile::kStageFloats + 16);
    float* row_scale = stage + Tile::kStageFloats;  // 16 per-row factors
    half*  stage_p = reinterpret_cast<half*>(stage);

    const int r = lane >> 1;            // softmax row of this lane within the band
    const int h = lane & 1;             // which interleaved half of the row
    const int band_row0 = warp_id * T;  // first row of the band within the Q block
    const bool band_active = band_row0 < a.valid_q;
    const int q_row = a.q_row_start + band_row0 + r;  // sequence row of this lane's softmax row

    // Row of each accumulator element of this lane (identical for every fragment).
    int row_of[NE], col_unused[NE];
    wmma_acc_layout(stage, lane, row_of, col_unused);

    Acc O[O_TILES];
    #pragma unroll
    for (int t = 0; t < O_TILES; t++) wmma::fill_fragment(O[t], 0.0f);
    BandSoftmax<1, 2> softmax;
    softmax.init();

    // O rows times per-row factors published in row_scale
    auto scale_o = [&]() {
        #pragma unroll
        for (int i = 0; i < NE; i++) {
            const float f = row_scale[row_of[i]];
            #pragma unroll
            for (int t = 0; t < O_TILES; t++) O[t].x[i] *= f;
        }
    };

    auto step = [&](auto mask_tag, int kv_start) {
        constexpr bool kMask = decltype(mask_tag)::value;

        // S = Q K^T for the band, staged for the softmax
        Acc S[S_TILES];
        #pragma unroll
        for (int tn = 0; tn < S_TILES; tn++) wmma::fill_fragment(S[tn], 0.0f);
        #pragma unroll
        for (int k = 0; k < d; k += T) {
            wmma::fragment<wmma::matrix_a, T, T, T, half, wmma::row_major> qa;
            wmma::load_matrix_sync(qa, smem_Q + band_row0 * LDH + k, LDH);
            #pragma unroll
            for (int tn = 0; tn < S_TILES; tn++) {
                wmma::fragment<wmma::matrix_b, T, T, T, half, wmma::col_major> kb;
                wmma::load_matrix_sync(kb, smem_K + tn * T * LDH + k, LDH);
                wmma::mma_sync(S[tn], qa, kb, S[tn]);
            }
        }
        #pragma unroll
        for (int tn = 0; tn < S_TILES; tn++) {
            wmma::store_matrix_sync(stage + tn * T, S[tn], LDS, wmma::mem_row_major);
        }
        __syncwarp();

        // Online softmax: this lane's half of row r
        float s[1][CHUNKS * 4];
        #pragma unroll
        for (int c = 0; c < CHUNKS; c++) {
            const float4 v = *reinterpret_cast<const float4*>(stage + r * LDS + 4 * (2 * c + h));
            s[0][4 * c] = v.x; s[0][4 * c + 1] = v.y; s[0][4 * c + 2] = v.z; s[0][4 * c + 3] = v.w;
        }
        // element 4c + i of the row: key 4 (2c + h) + i of the block
        fwd_logits<kMask>(a, s, [&](int) { return q_row; },
                          [&](int e) { return kv_start + 4 * (2 * (e / 4) + h) + e % 4; });
        float alpha[1];
        softmax.update(s, alpha);
        if (h == 0) row_scale[r] = alpha[0];
        __syncwarp();  // every lane has read S before P overwrites it

        #pragma unroll
        for (int c = 0; c < CHUNKS; c++) {
            const half2 p01 = __floats2half2_rn(s[0][4 * c], s[0][4 * c + 1]);
            const half2 p23 = __floats2half2_rn(s[0][4 * c + 2], s[0][4 * c + 3]);
            uint2 packed;
            memcpy(&packed.x, &p01, 4);
            memcpy(&packed.y, &p23, 4);
            *reinterpret_cast<uint2*>(stage_p + r * 2 * LDS + 4 * (2 * c + h)) = packed;
        }
        __syncwarp();

        // O = O * alpha(row) + P V
        scale_o();
        #pragma unroll
        for (int k = 0; k < Bc; k += T) {
            wmma::fragment<wmma::matrix_a, T, T, T, half, wmma::row_major> pa;
            wmma::load_matrix_sync(pa, stage_p + k, 2 * LDS);
            #pragma unroll
            for (int t = 0; t < O_TILES; t++) {
                wmma::fragment<wmma::matrix_b, T, T, T, half, wmma::row_major> vb;
                wmma::load_matrix_sync(vb, smem_V + k * LDH + t * T, LDH);
                wmma::mma_sync(O[t], pa, vb, O[t]);
            }
        }
        __syncwarp();  // P and row_scale reads done before the next block reuses them
    };

    fwd_band_run<Tile, LDH>(a, band_row0, band_active, smem_Q, smem_K, smem_V, step);
    if (!band_active) return;

    // Finalize: O / row_sum, logsumexp, O written through the staging tile
    float inv[1], lse[1];
    softmax.finalize(inv, lse);
    const int band_valid = min(T, a.valid_q - band_row0);
    if (h == 0) {
        row_scale[r] = inv[0];
        if (r < band_valid) a.L[(band_row0 + r) * a.l_stride] = lse[0];
    }
    __syncwarp();
    scale_o();
    constexpr int GROUP = (S_TILES < O_TILES) ? S_TILES : O_TILES;  // O tiles per staging pass
    static_assert(O_TILES % GROUP == 0, "O tiles must split into staging passes");
    constexpr int PASS_CHUNKS = T * GROUP * T / 8;  // 8-half output chunks per pass
    #pragma unroll
    for (int t0 = 0; t0 < O_TILES; t0 += GROUP) {
        #pragma unroll
        for (int g = 0; g < GROUP; g++) {
            wmma::store_matrix_sync(stage + g * T, O[t0 + g], LDS, wmma::mem_row_major);
        }
        __syncwarp();
        #pragma unroll
        for (int c = lane; c < PASS_CHUNKS; c += 32) {
            const int row = c / (GROUP * 2), col = (c % (GROUP * 2)) * 8;
            if (row < band_valid && t0 * T + col < a.head_dim) {
                const float4 lo = *reinterpret_cast<const float4*>(stage + row * LDS + col);
                const float4 hi = *reinterpret_cast<const float4*>(stage + row * LDS + col + 4);
                const half2 o[4] = {__floats2half2_rn(lo.x, lo.y), __floats2half2_rn(lo.z, lo.w),
                                    __floats2half2_rn(hi.x, hi.y), __floats2half2_rn(hi.z, hi.w)};
                uint4 packed;
                memcpy(&packed, o, sizeof(packed));
                *reinterpret_cast<uint4*>(a.O + (band_row0 + row) * a.o_stride + t0 * T + col) = packed;
            }
        }
        __syncwarp();
    }
}
};

#else  // __CUDA_ARCH__ < 700

// Stub body for SM < 7.0 device code — never launched there (dispatch in .cpp).
struct VoltaFwd {
    template <typename Tile>
    static __device__ __forceinline__ void run(const FwdBlockArgs&) {}
};

#endif  // __CUDA_ARCH__


// ---------------------------------------------------------------------------
// Host launchers (always compiled — runtime dispatch ensures correctness)
//
// Tile per head_dim: see the TileVolta_* aliases (flash_attn_common.h). The
// dispatch checks the tile against the device's shared memory (clear error)
// and gives the varlen host code its Br, so launch and block offsets always
// agree.
// ---------------------------------------------------------------------------
template <typename F>
static void dispatch_volta_fwd_tile(int head_dim, F&& f) {
    const int limit = max_dynamic_smem_per_block();
    bool ok = false;
    switch (head_dim) {
        case 32:  ok = with_first_fitting_tile<TileVolta_d32>(limit, f); break;
        case 64:  ok = with_first_fitting_tile<TileVolta_d64>(limit, f); break;
        case 72:
        case 80:  ok = with_first_fitting_tile<TileVolta_d80>(limit, f); break;  // 72 zero-padded
        case 96:  ok = with_first_fitting_tile<TileVolta_d96>(limit, f); break;
        case 128: ok = with_first_fitting_tile<TileVolta_d128, TileVolta_d128_64k>(limit, f); break;
        case 256: ok = with_first_fitting_tile<TileVolta_d256, TileVolta_d256_64k>(limit, f); break;
        default:  throw std::runtime_error("Unsupported head_dim for the WMMA forward");
    }
    if (!ok) throw_no_tile("WMMA forward", head_dim, limit);
}

void flash_attn_fwd_volta(FlashAttnParams& params, cudaStream_t stream) {
    dispatch_volta_fwd_tile(params.head_dim, [&](auto tile) {
        launch_fwd<decltype(tile), VoltaFwd>(params, stream);
    });
}

void flash_attn_fwd_volta_varlen(FlashAttnVarlenParams& params,
                                 int total_q_blocks, cudaStream_t stream) {
    dispatch_volta_fwd_tile(params.head_dim, [&](auto tile) {
        launch_fwd_varlen<decltype(tile), VoltaFwd>(params, total_q_blocks, stream);
    });
}

// Q rows per block of the forward tile selected for head_dim on this device.
int flash_attn_fwd_volta_block_rows(int head_dim) {
    int rows = 0;
    dispatch_volta_fwd_tile(head_dim, [&](auto tile) { rows = decltype(tile)::Br; });
    return rows;
}

}  // namespace flash_attn_legacy
