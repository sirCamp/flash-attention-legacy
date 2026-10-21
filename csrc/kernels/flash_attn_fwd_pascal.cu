// ============================================================================
// Flash Attention v2 — Forward Kernel for Pascal (SM 6.0/6.1), CUDA cores
// (also runs on newer GPUs with FLASH_ATTN_LEGACY_FORCE_ARCH=pascal).
//
// Key design:
//   - fp16 storage, fp32 math (products and sums)
//   - 1 thread = 1 Q row (Br = threads); P stays in registers (the owner of a
//     row of P multiplies it by V, whose rows every thread reads at once)
//   - Q rows padded in shared memory, 128-bit loads (bank-conflict free)
//   - one block body (pascal_fwd_block) shared by the batched and varlen kernels
// ============================================================================

#include "flash_attn_common.h"

#include <string>

namespace flash_attn_legacy {

// ---------------------------------------------------------------------------
// Vectorized global → shared memory load using half2 (4-byte transactions).
// Destination rows have stride DST_STRIDE (>= COLS, padded to avoid bank
// conflicts on per-row accesses); rows >= num_valid_rows are zero-filled.
// ---------------------------------------------------------------------------
template <int ROWS, int COLS, int NUM_THREADS, int DST_STRIDE = COLS>
__device__ __forceinline__ void load_tile_g2s(
    const half* __restrict__ src,
    int src_row_stride,
    half* __restrict__ dst,
    int num_valid_rows
) {
    static_assert(COLS % 2 == 0 && DST_STRIDE % 2 == 0, "half2 loads need even sizes");
    constexpr int COLS2 = COLS / 2;
    constexpr int TOTAL2 = ROWS * COLS2;
    constexpr int PER_THREAD = (TOTAL2 + NUM_THREADS - 1) / NUM_THREADS;

    const int tid = threadIdx.x;
    const half2 zero2 = {__float2half(0.0f), __float2half(0.0f)};

    #pragma unroll
    for (int i = 0; i < PER_THREAD; i++) {
        int idx = tid + i * NUM_THREADS;
        if (idx < TOTAL2) {
            int row = idx / COLS2;
            int col2 = idx % COLS2;
            half2* dst2 = reinterpret_cast<half2*>(dst + row * DST_STRIDE);
            if (row < num_valid_rows) {
                const half2* src2 = reinterpret_cast<const half2*>(src + row * src_row_stride);
                dst2[col2] = src2[col2];
            } else {
                dst2[col2] = zero2;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// fp16 dot product with fp32 products and accumulation, 128-bit loads.
// The products are NOT formed in fp16 (__hmul2): a single |q_i * k_i| > 65504
// would overflow to inf even with in-range inputs, and on sm_61 fp16 math runs
// at 1/64 of the fp32 rate.
// ---------------------------------------------------------------------------
template <int LEN>
__device__ __forceinline__ float half_dot(
    const half* __restrict__ a,
    const half* __restrict__ b
) {
    static_assert(LEN % 8 == 0, "LEN must be a multiple of 8 (128-bit loads)");
    const uint4* a4 = reinterpret_cast<const uint4*>(a);
    const uint4* b4 = reinterpret_cast<const uint4*>(b);
    float acc = 0.0f;
    #pragma unroll
    for (int i = 0; i < LEN / 8; i++) {
        const uint4 x = a4[i], y = b4[i];
        const half2* xh = reinterpret_cast<const half2*>(&x);
        const half2* yh = reinterpret_cast<const half2*>(&y);
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const float2 xf = __half22float2(xh[j]);
            const float2 yf = __half22float2(yh[j]);
            acc = fmaf(xf.x, yf.x, acc);
            acc = fmaf(xf.y, yf.y, acc);
        }
    }
    return acc;
}

// ---------------------------------------------------------------------------
// One Q block of the CUDA-core forward, shared by the batched and varlen
// kernels (same structure as volta_fwd_block).
//
// Shared memory (Tile::smem_bytes()):
//   smem_Q [Br, LDQ] half (LDQ = d + 8), smem_K [Bc, d] half, smem_V [Bc, d] half
// Each thread owns one Q row: it reads its own Q row with 128-bit loads (the
// padded stride makes them bank-conflict free) and K/V rows that every thread
// reads at the same time (broadcast). P stays in registers: the thread that
// computes a row of P is the one that multiplies it by V.
// Returns the row's logsumexp (only meaningful for threads that own a row).
// ---------------------------------------------------------------------------
template <typename Tile>
__device__ __forceinline__ float pascal_fwd_block(
    const half* __restrict__ Q_ptr, int q_stride,
    const half* __restrict__ K_ptr, int k_stride,
    const half* __restrict__ V_ptr, int v_stride,
    half* __restrict__ O_ptr, int o_stride,
    int q_row_start, int valid_q, int seq_len_q, int seq_len_k,
    float softmax_scale, bool is_causal
) {
    constexpr int Br = Tile::Br;
    constexpr int Bc = Tile::Bc;
    constexpr int d  = Tile::d;
    constexpr int NUM_THREADS = Tile::kNumWarps * 32;
    constexpr int LDQ = Tile::kLdQ;

    extern __shared__ __align__(128) char smem_raw[];  // aligned for 128-bit accesses
    half* smem_Q = reinterpret_cast<half*>(smem_raw);
    half* smem_K = smem_Q + Br * LDQ;
    half* smem_V = smem_K + Bc * d;

    const int tid = threadIdx.x;
    const bool owns_row = (tid < Br) && (tid < valid_q);
    const int causal_off = seq_len_k - seq_len_q;

    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float my_O[d];
    #pragma unroll
    for (int j = 0; j < d; j++) my_O[j] = 0.0f;

    load_tile_g2s<Br, d, NUM_THREADS, LDQ>(Q_ptr, q_stride, smem_Q, valid_q);
    __syncthreads();

    const int kv_limit = fwd_kv_limit(is_causal, q_row_start, Br, Bc, seq_len_q, seq_len_k);
    for (int kv_blk = 0; kv_blk < kv_limit; kv_blk++) {
        const int kv_start = kv_blk * Bc;
        const int valid_kv = min(Bc, seq_len_k - kv_start);

        // All threads finished the previous S (they passed the barrier after the
        // previous V load), so K can be overwritten; P@V does not read K.
        load_tile_g2s<Bc, d, NUM_THREADS>(K_ptr + kv_start * k_stride, k_stride, smem_K, valid_kv);
        __syncthreads();

        float p[Bc];
        if (owns_row) {
            float local_max = -INFINITY;
            #pragma unroll
            for (int c = 0; c < Bc; c++) {
                float s;
                if ((kv_start + c) >= seq_len_k ||
                    (is_causal && (kv_start + c) > (q_row_start + tid + causal_off))) {
                    s = -INFINITY;
                } else {
                    s = half_dot<d>(smem_Q + tid * LDQ, smem_K + c * d) * softmax_scale;
                }
                p[c] = s;
                local_max = fmaxf(local_max, s);
            }
            const float new_max = fmaxf(row_max, local_max);
            // new_max == -inf: row sees no key yet (fully masked) -> avoid exp(-inf - -inf) = NaN
            const float rescale = (new_max == -INFINITY) ? 1.0f : expf(row_max - new_max);
            row_max = new_max;
            row_sum *= rescale;
            #pragma unroll
            for (int j = 0; j < d; j++) my_O[j] *= rescale;
            float local_sum = 0.0f;
            #pragma unroll
            for (int c = 0; c < Bc; c++) {
                p[c] = (p[c] == -INFINITY) ? 0.0f : expf(p[c] - new_max);
                local_sum += p[c];
            }
            row_sum += local_sum;
        }

        // All threads finished reading K (S above) before V is written over the
        // previous V; the barrier also publishes V.
        load_tile_g2s<Bc, d, NUM_THREADS>(V_ptr + kv_start * v_stride, v_stride, smem_V, valid_kv);
        __syncthreads();

        // O += P @ V: own row of P (registers) x V rows (broadcast reads)
        if (owns_row) {
            #pragma unroll
            for (int c = 0; c < Bc; c++) {  // no p == 0 skip: divergent, and 0 * inf must give NaN
                const half2* v2 = reinterpret_cast<const half2*>(smem_V + c * d);
                #pragma unroll
                for (int j = 0; j < d / 2; j++) {
                    const float2 v = __half22float2(v2[j]);
                    my_O[2 * j]     = fmaf(p[c], v.x, my_O[2 * j]);
                    my_O[2 * j + 1] = fmaf(p[c], v.y, my_O[2 * j + 1]);
                }
            }
        }
    }

    if (owns_row) {
        const float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
        half2* o_dst2 = reinterpret_cast<half2*>(O_ptr + tid * o_stride);
        #pragma unroll
        for (int j = 0; j < d / 2; j++) {
            o_dst2[j] = __floats2half2_rn(my_O[2 * j] * inv_sum, my_O[2 * j + 1] * inv_sum);
        }
    }
    return row_max + logf(fmaxf(row_sum, 1e-20f));
}

// ---------------------------------------------------------------------------
// Forward kernel — CUDA cores (Pascal), batched [B, H, N, d]
// ---------------------------------------------------------------------------
template <typename Tile>
__global__ void __launch_bounds__(Tile::kNumWarps * 32)
flash_attn_fwd_pascal_kernel(FlashAttnParams params) {
    constexpr int Br = Tile::Br;
    const int bh_idx = blockIdx.y;
    const int b_idx = bh_idx / params.num_heads;
    const int h_idx = bh_idx % params.num_heads;
    const int kv_h_idx = h_idx / (params.num_heads / params.num_heads_k);
    const int q_row_start = blockIdx.x * Br;
    if (q_row_start >= params.seq_len_q) return;
    const int valid_q = min(Br, params.seq_len_q - q_row_start);

    const float lse = pascal_fwd_block<Tile>(
        params.Q + b_idx * params.q_batch_stride + h_idx * params.q_head_stride
                 + q_row_start * params.q_seq_stride, params.q_seq_stride,
        params.K + b_idx * params.k_batch_stride + kv_h_idx * params.k_head_stride, params.k_seq_stride,
        params.V + b_idx * params.v_batch_stride + kv_h_idx * params.v_head_stride, params.v_seq_stride,
        params.O + b_idx * params.o_batch_stride + h_idx * params.o_head_stride
                 + q_row_start * params.o_seq_stride, params.o_seq_stride,
        q_row_start, valid_q, params.seq_len_q, params.seq_len_k,
        params.softmax_scale, params.is_causal);

    const int tid = threadIdx.x;
    if (tid < valid_q) {
        params.L[b_idx * params.l_batch_stride + h_idx * params.l_head_stride + q_row_start + tid] = lse;
    }
}

// ---------------------------------------------------------------------------
// Forward kernel — CUDA cores (Pascal), varlen (packed [total, H, d])
// ---------------------------------------------------------------------------
template <typename Tile>
__global__ void __launch_bounds__(Tile::kNumWarps * 32)
flash_attn_fwd_pascal_varlen_kernel(FlashAttnVarlenParams params) {
    constexpr int Br = Tile::Br;
    const int h_idx = blockIdx.y;
    const int block_x = blockIdx.x;

    // Binary search: find which sequence this block belongs to
    int lo = 0, hi = params.num_seqs;
    while (lo < hi) {
        const int mid = (lo + hi) / 2;
        if (params.block_offsets[mid + 1] <= block_x) lo = mid + 1;
        else hi = mid;
    }
    const int seq_idx = lo;

    const int local_block = block_x - params.block_offsets[seq_idx];
    const int q_start_tok = params.cu_seqlens_q[seq_idx];
    const int k_start_tok = params.cu_seqlens_k[seq_idx];
    const int seq_len_q = params.cu_seqlens_q[seq_idx + 1] - q_start_tok;
    const int seq_len_k = params.cu_seqlens_k[seq_idx + 1] - k_start_tok;
    const int q_row_start = local_block * Br;
    if (q_row_start >= seq_len_q) return;
    const int valid_q = min(Br, seq_len_q - q_row_start);
    const int kv_h_idx = h_idx / (params.num_heads / params.num_heads_k);

    const float lse = pascal_fwd_block<Tile>(
        params.Q + (q_start_tok + q_row_start) * params.q_token_stride + h_idx * params.q_head_stride,
        params.q_token_stride,
        params.K + k_start_tok * params.k_token_stride + kv_h_idx * params.k_head_stride,
        params.k_token_stride,
        params.V + k_start_tok * params.v_token_stride + kv_h_idx * params.v_head_stride,
        params.v_token_stride,
        params.O + (q_start_tok + q_row_start) * params.o_token_stride + h_idx * params.o_head_stride,
        params.o_token_stride,
        q_row_start, valid_q, seq_len_q, seq_len_k,
        params.softmax_scale, params.is_causal);

    const int tid = threadIdx.x;
    if (tid < valid_q) {
        params.L[(q_start_tok + q_row_start + tid) * params.num_heads + h_idx] = lse;
    }
}

// ---------------------------------------------------------------------------
// Host launchers — same dispatch scheme as the WMMA forward (one candidate per
// head_dim: every Pascal tile fits the 48 KB of shared memory per block).
// ---------------------------------------------------------------------------
template <typename F>
static void dispatch_pascal_fwd_tile(int head_dim, F&& f) {
    const int limit = max_dynamic_smem_per_block();
    bool ok = false;
    switch (head_dim) {
        case 32:  ok = with_first_fitting_tile<TilePascal_d32>(limit, f); break;
        case 64:  ok = with_first_fitting_tile<TilePascal_d64>(limit, f); break;
        case 96:  ok = with_first_fitting_tile<TilePascal_d96>(limit, f); break;
        case 128: ok = with_first_fitting_tile<TilePascal_d128>(limit, f); break;
        case 256: ok = with_first_fitting_tile<TilePascal_d256>(limit, f); break;
        default:  throw std::runtime_error("Unsupported head_dim for the CUDA-core forward");
    }
    if (!ok) throw_no_tile("CUDA-core forward", head_dim, limit);
}

template <typename Tile>
static void launch_pascal_fwd(FlashAttnParams& params, cudaStream_t stream) {
    constexpr int smem = Tile::smem_bytes();
    dim3 grid(cdiv(params.seq_len_q, Tile::Br), params.batch_size * params.num_heads);
    dim3 block(Tile::kNumWarps * 32);
    flash_attn_fwd_pascal_kernel<Tile><<<grid, block, smem, stream>>>(params);
}

template <typename Tile>
static void launch_pascal_fwd_varlen(FlashAttnVarlenParams& params, int total_q_blocks,
                                     cudaStream_t stream) {
    constexpr int smem = Tile::smem_bytes();
    dim3 grid(total_q_blocks, params.num_heads);
    dim3 block(Tile::kNumWarps * 32);
    flash_attn_fwd_pascal_varlen_kernel<Tile><<<grid, block, smem, stream>>>(params);
}

void flash_attn_fwd_pascal(FlashAttnParams& params, cudaStream_t stream) {
    dispatch_pascal_fwd_tile(params.head_dim, [&](auto tile) {
        launch_pascal_fwd<decltype(tile)>(params, stream);
    });
}

void flash_attn_fwd_pascal_varlen(FlashAttnVarlenParams& params,
                                  int total_q_blocks, cudaStream_t stream) {
    dispatch_pascal_fwd_tile(params.head_dim, [&](auto tile) {
        launch_pascal_fwd_varlen<decltype(tile)>(params, total_q_blocks, stream);
    });
}

// Q rows per block of the forward tile selected for head_dim.
int flash_attn_fwd_pascal_block_rows(int head_dim) {
    int rows = 0;
    dispatch_pascal_fwd_tile(head_dim, [&](auto tile) { rows = decltype(tile)::Br; });
    return rows;
}

}  // namespace flash_attn_legacy
