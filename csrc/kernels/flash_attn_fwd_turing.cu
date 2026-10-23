// ============================================================================
// Flash Attention v2 — Forward Kernel for Turing (SM 7.5): mma.sync m16n8k8 +
// ldmatrix (flash_attn_mma.h), the band forward of flash_attn_fwd_band.h.
//
// Key design (FA2-style, docs/analysis/kernels/p2_8_design.md, MmaSm75):
//   - each warp owns a band of 16 query rows; lane = 4 g + t holds rows g and
//     g + 8 of S (keys 8n + 2t, +1) and of O (columns 8n + 2t, +1)
//   - S = Q K^T with Q as the A operand (registers when d <= 128, else ldmatrix
//     from shared memory) and K rows through ldmatrix
//   - online softmax on the S accumulators (BandSoftmax<2, 4>: the 4 lanes of a
//     row); P packed to half2 is the A operand of O += P V as it is, V through
//     ldmatrix.trans: no shared-memory staging, no layout probe
//
// Architecture guard: mma.sync / ldmatrix need SM >= 7.5; older device code
// gets a trapping stub (the host selects the family from the arch of the code
// that runs).
// ============================================================================

#include "flash_attn_fwd_band.h"
#include "flash_attn_mma.h"

namespace flash_attn_legacy {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 750

struct TuringFwd {
template <typename Tile, bool kCache>
static __device__ __forceinline__ void run(const FwdBlockArgs& a) {
    constexpr int Br = Tile::Br;
    constexpr int Bc = Tile::Bc;
    constexpr int d  = Tile::d;
    constexpr int LDH = Tile::kLdH;
    constexpr int SN = Bc / 8;   // S n-tiles (keys) per warp
    constexpr int ON = d / 8;    // O n-tiles (columns) per warp
    constexpr int QK = d / 16;   // k-steps of 16 for S = Q K^T
    static_assert(SN % 2 == 0 && ON % 2 == 0, "n-tiles are loaded in pairs");

    extern __shared__ __align__(128) char smem_raw[];
    half* smem_Q = reinterpret_cast<half*>(smem_raw);
    half* smem_K = smem_Q + Br * LDH;
    half* smem_V = smem_K + Bc * LDH;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int g = lane >> 2, t = lane & 3;    // accumulator rows g, g + 8; columns 2t, 2t + 1
    const int lr = lane & 7, lm = lane >> 3;  // ldmatrix: row within matrix lm
    const int band_row0 = warp_id * kBandRows;
    const bool band_active = band_row0 < a.valid_q;

    // A operand rows of the band's Q (ldmatrix x4: rows 0-7 / 8-15, k and k + 8)
    const half* Q_rows = smem_Q + (band_row0 + (lm & 1) * 8 + lr) * LDH + (lm >> 1) * 8;
    uint32_t qa[Tile::kQInRegs ? QK : 1][4];  // filled on the first step (Q published by then)
    bool q_ready = false;

    float O[ON][4];
    #pragma unroll
    for (int n = 0; n < ON; n++) {
        #pragma unroll
        for (int e = 0; e < 4; e++) O[n][e] = 0.0f;
    }
    BandSoftmax<2, 4> softmax;
    softmax.init(a.sink_log2);

    auto step = [&](auto mask_tag, int kv_start) {
        constexpr bool kMask = decltype(mask_tag)::value;
        if constexpr (Tile::kQInRegs) {
            if (!q_ready) {
                #pragma unroll
                for (int k = 0; k < QK; k++) ldmatrix_x4(qa[k], Q_rows + k * 16);
                q_ready = true;
            }
        }

        // S = Q K^T: B from K rows (ldmatrix, no trans): matrices (keys n, k),
        // (n, k + 8), (n + 1, k), (n + 1, k + 8)
        float S[SN][4];
        #pragma unroll
        for (int n = 0; n < SN; n++) {
            #pragma unroll
            for (int e = 0; e < 4; e++) S[n][e] = 0.0f;
        }
        #pragma unroll
        for (int k = 0; k < QK; k++) {
            uint32_t q[4];
            if constexpr (Tile::kQInRegs) {
                #pragma unroll
                for (int i = 0; i < 4; i++) q[i] = qa[k][i];
            } else {
                ldmatrix_x4(q, Q_rows + k * 16);
            }
            #pragma unroll
            for (int n = 0; n < SN; n += 2) {
                uint32_t kb[4];
                ldmatrix_x4(kb, smem_K + ((n + (lm >> 1)) * 8 + lr) * LDH + k * 16 + (lm & 1) * 8);
                mma_m16n8k8(S[n], q[0], q[1], kb[0]);
                mma_m16n8k8(S[n], q[2], q[3], kb[1]);
                mma_m16n8k8(S[n + 1], q[0], q[1], kb[2]);
                mma_m16n8k8(S[n + 1], q[2], q[3], kb[3]);
            }
        }

        // Online softmax on rows g (r = 0) and g + 8 (r = 1): element c of a row is
        // key 8 (c / 2) + 2t + (c & 1) of the block
        float s[2][2 * SN];
        #pragma unroll
        for (int n = 0; n < SN; n++) {
            s[0][2 * n] = S[n][0]; s[0][2 * n + 1] = S[n][1];
            s[1][2 * n] = S[n][2]; s[1][2 * n + 1] = S[n][3];
        }
        fwd_logits<kMask, kCache>(a, s, [&](int r) { return a.q_row_start + band_row0 + g + 8 * r; },
                          [&](int c) { return kv_start + 8 * (c / 2) + 2 * t + (c & 1); });
        float alpha[2];
        softmax.update(s, alpha);

        // O = O * alpha(row) + P V: A = P (k-step j = keys 8j..8j+7) from the
        // softmax registers; B = V [key, column] with ldmatrix.trans: matrices
        // (keys j, columns n), (j, n + 1), (j + 1, n), (j + 1, n + 1)
        #pragma unroll
        for (int n = 0; n < ON; n++) {
            O[n][0] *= alpha[0]; O[n][1] *= alpha[0];
            O[n][2] *= alpha[1]; O[n][3] *= alpha[1];
        }
        #pragma unroll
        for (int j = 0; j < SN; j += 2) {
            const uint32_t p0a = pack_half2(s[0][2 * j], s[0][2 * j + 1]);
            const uint32_t p0b = pack_half2(s[1][2 * j], s[1][2 * j + 1]);
            const uint32_t p1a = pack_half2(s[0][2 * j + 2], s[0][2 * j + 3]);
            const uint32_t p1b = pack_half2(s[1][2 * j + 2], s[1][2 * j + 3]);
            const int row = (j + (lm >> 1)) * 8 + lr;
            #pragma unroll
            for (int n = 0; n < ON; n += 2) {
                uint32_t vb[4];
                ldmatrix_x4_trans(vb, smem_V + row * LDH + (n + (lm & 1)) * 8);
                mma_m16n8k8(O[n], p0a, p0b, vb[0]);
                mma_m16n8k8(O[n + 1], p0a, p0b, vb[1]);
                mma_m16n8k8(O[n], p1a, p1b, vb[2]);
                mma_m16n8k8(O[n + 1], p1a, p1b, vb[3]);
            }
        }
    };

    fwd_band_run<Tile, LDH, kCache>(a, band_row0, band_active, smem_Q, smem_K, smem_V, step);
    if (!band_active) return;

    float inv[2], lse[2];
    softmax.finalize(inv, lse);
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        const int row = band_row0 + g + 8 * r;  // within the Q block
        if (row >= a.valid_q) continue;
        if (t == 0) fwd_store_lse<kCache>(a, row, lse[r]);
        #pragma unroll
        for (int n = 0; n < ON; n++) {
            const int col = 8 * n + 2 * t;
            if (col < a.head_dim) {  // not a zero-padded column (d=72/80)
                fwd_store_o<kCache>(a, row, col, {O[n][2 * r] * inv[r], O[n][2 * r + 1] * inv[r]});
            }
        }
    }
}
};

#else  // __CUDA_ARCH__ < 750

// Stub body for older device code — never launched there: the host selects the
// family from the arch of the code that runs (compiled_arch_for).
struct TuringFwd {
    template <typename Tile, bool kCache>
    static __device__ __forceinline__ void run(const FwdBlockArgs&) { __trap(); }
};

#endif  // __CUDA_ARCH__

// ---------------------------------------------------------------------------
// Host launchers: tile per head_dim (TileTuring_*, flash_attn_common.h, with
// the measurements); the dispatch checks the tile against the device's shared
// memory and gives the varlen host code its Br.
// ---------------------------------------------------------------------------
template <typename F>
static void dispatch_turing_fwd_tile(int head_dim, F&& f) {
    const int limit = max_dynamic_smem_per_block();
    bool ok = false;
    switch (head_dim) {
        case 32:  ok = with_first_fitting_tile<TileTuring_d32>(limit, f); break;
        case 64:  ok = with_first_fitting_tile<TileTuring_d64>(limit, f); break;
        case 72:
        case 80:  ok = with_first_fitting_tile<TileTuring_d80>(limit, f); break;  // 72 zero-padded
        case 96:  ok = with_first_fitting_tile<TileTuring_d96>(limit, f); break;
        case 128: ok = with_first_fitting_tile<TileTuring_d128>(limit, f); break;
        case 256: ok = with_first_fitting_tile<TileTuring_d256>(limit, f); break;
        default:  throw std::runtime_error("Unsupported head_dim for the mma forward");
    }
    if (!ok) throw_no_tile("mma forward", head_dim, limit);
}

void flash_attn_fwd_turing(FlashAttnParams& params, cudaStream_t stream) {
    dispatch_turing_fwd_tile(params.head_dim, [&](auto tile) {
        launch_fwd<decltype(tile), TuringFwd>(params, stream);
    });
}

void flash_attn_fwd_turing_varlen(FlashAttnVarlenParams& params, int total_q_blocks, cudaStream_t stream) {
    dispatch_turing_fwd_tile(params.head_dim, [&](auto tile) {
        launch_fwd_varlen<decltype(tile), TuringFwd>(params, total_q_blocks, stream);
    });
}

// Q rows per block of the forward tile selected for head_dim on this device.
int flash_attn_fwd_turing_block_rows(int head_dim) {
    int rows = 0;
    dispatch_turing_fwd_tile(head_dim, [&](auto tile) { rows = decltype(tile)::Br; });
    return rows;
}

}  // namespace flash_attn_legacy
