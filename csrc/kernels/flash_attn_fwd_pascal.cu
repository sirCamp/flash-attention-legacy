// ============================================================================
// Flash Attention v2 — Forward Kernel for Pascal (SM 6.0/6.1)
// No tensor cores — uses half2 packed FP16 arithmetic on CUDA cores.
//
// Key design:
//   - P buffer in shared memory → no recomputation of S for P@V
//   - 1 thread = 1 Q row (NUM_THREADS >= Br always)
//   - Vectorized half2 loads/stores for 2× memory bandwidth
//   - Tiles: Br=32,Bc=32 for d=64; Br=16,Bc=16 for d=128
// ============================================================================

#include "flash_attn_common.h"

namespace flash_attn_legacy {

// ---------------------------------------------------------------------------
// Vectorized global → shared memory load using half2 (4-byte transactions)
// ---------------------------------------------------------------------------
template <int ROWS, int COLS, int NUM_THREADS>
__device__ __forceinline__ void load_tile_g2s(
    const half* __restrict__ src,
    int src_row_stride,
    half* __restrict__ dst,
    int num_valid_rows
) {
    static_assert(COLS % 2 == 0, "COLS must be even for half2 loads");
    constexpr int COLS2 = COLS / 2;
    constexpr int TOTAL2 = ROWS * COLS2;
    constexpr int PER_THREAD = (TOTAL2 + NUM_THREADS - 1) / NUM_THREADS;

    const int tid = threadIdx.x;
    const half2 zero2 = {__float2half(0.0f), __float2half(0.0f)};
    half2* dst2 = reinterpret_cast<half2*>(dst);

    #pragma unroll
    for (int i = 0; i < PER_THREAD; i++) {
        int idx = tid + i * NUM_THREADS;
        if (idx < TOTAL2) {
            int row = idx / COLS2;
            int col2 = idx % COLS2;
            if (row < num_valid_rows) {
                const half2* src2 = reinterpret_cast<const half2*>(src + row * src_row_stride);
                dst2[row * COLS2 + col2] = src2[col2];
            } else {
                dst2[row * COLS2 + col2] = zero2;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// half2 dot product — compile-time length for perfect unrolling
// ---------------------------------------------------------------------------
template <int LEN>
__device__ __forceinline__ float half2_dot(
    const half* __restrict__ a,
    const half* __restrict__ b
) {
    static_assert(LEN % 2 == 0, "LEN must be even");
    constexpr int LEN2 = LEN / 2;
    const half2* a2 = reinterpret_cast<const half2*>(a);
    const half2* b2 = reinterpret_cast<const half2*>(b);

    float acc = 0.0f;
    #pragma unroll
    for (int i = 0; i < LEN2; i++) {
        half2 prod = __hmul2(a2[i], b2[i]);
        acc += __half2float(prod.x) + __half2float(prod.y);
    }
    return acc;
}

// ---------------------------------------------------------------------------
// Forward kernel — Pascal
// ---------------------------------------------------------------------------
template <typename Tile>
__global__ void __launch_bounds__(Tile::kNumWarps * 32)
flash_attn_fwd_pascal_kernel(FlashAttnParams params) {

    constexpr int Br = Tile::Br;
    constexpr int Bc = Tile::Bc;
    constexpr int d  = Tile::d;
    constexpr int NUM_THREADS = Tile::kNumWarps * 32;

    const int bh_idx = blockIdx.y;
    const int b_idx = bh_idx / params.num_heads;
    const int h_idx = bh_idx % params.num_heads;
    // GQA: map query head to KV head
    const int kv_h_idx = h_idx / (params.num_heads / params.num_heads_k);
    const int q_row_start = blockIdx.x * Br;
    if (q_row_start >= params.seq_len) return;

    const int valid_q = min(Br, params.seq_len - q_row_start);

    // Base pointers
    const half* Q_ptr = params.Q + b_idx * params.q_batch_stride
                                 + h_idx * params.q_head_stride
                                 + q_row_start * params.q_seq_stride;
    const half* K_ptr = params.K + b_idx * params.k_batch_stride
                                 + kv_h_idx * params.k_head_stride;
    const half* V_ptr = params.V + b_idx * params.v_batch_stride
                                 + kv_h_idx * params.v_head_stride;
    half* O_ptr = params.O + b_idx * params.o_batch_stride
                           + h_idx * params.o_head_stride
                           + q_row_start * params.o_seq_stride;
    float* L_ptr = params.L + b_idx * params.l_batch_stride
                            + h_idx * params.l_head_stride
                            + q_row_start;

    // Shared memory
    extern __shared__ char smem_raw[];
    half*  smem_Q = reinterpret_cast<half*>(smem_raw);
    half*  smem_K = smem_Q + Br * d;
    half*  smem_V = smem_K + Bc * d;
    float* smem_P = reinterpret_cast<float*>(smem_V + Bc * d);  // [Br, Bc]

    const int tid = threadIdx.x;
    const bool owns_row = (tid < Br) && (tid < valid_q);

    // Per-row registers (only active threads use these)
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float my_O[d];
    #pragma unroll
    for (int j = 0; j < d; j++) my_O[j] = 0.0f;

    // Load Q (stays for entire block)
    load_tile_g2s<Br, d, NUM_THREADS>(Q_ptr, params.q_seq_stride, smem_Q, valid_q);
    __syncthreads();

    // Main loop over K/V blocks
    const int nkv = cdiv(params.seq_len, Bc);
    const int kv_limit = params.is_causal ? min(nkv, cdiv(q_row_start + Br, Bc)) : nkv;

    for (int kv_blk = 0; kv_blk < kv_limit; kv_blk++) {
        const int kv_start = kv_blk * Bc;
        const int valid_kv = min(Bc, params.seq_len - kv_start);

        // Load K
        load_tile_g2s<Bc, d, NUM_THREADS>(
            K_ptr + kv_start * params.k_seq_stride,
            params.k_seq_stride, smem_K, valid_kv);
        __syncthreads();

        // Compute S, softmax → P in shared memory
        if (owns_row) {
            float local_max = -INFINITY;

            // Compute S and find row max
            #pragma unroll
            for (int c = 0; c < Bc; c++) {
                float s;
                if ((kv_start + c) >= params.seq_len ||
                    (params.is_causal && (kv_start + c) > (q_row_start + tid))) {
                    s = -INFINITY;
                } else {
                    s = half2_dot<d>(smem_Q + tid * d, smem_K + c * d) * params.softmax_scale;
                }
                smem_P[tid * Bc + c] = s;
                local_max = fmaxf(local_max, s);
            }

            // Online softmax rescale
            float new_max = fmaxf(row_max, local_max);
            float rescale = expf(row_max - new_max);
            row_max = new_max;
            row_sum *= rescale;
            #pragma unroll
            for (int j = 0; j < d; j++) my_O[j] *= rescale;

            // Compute P = exp(S - max), store in shared, accumulate sum
            float local_sum = 0.0f;
            #pragma unroll
            for (int c = 0; c < Bc; c++) {
                float s = smem_P[tid * Bc + c];
                float p = (s == -INFINITY) ? 0.0f : expf(s - new_max);
                smem_P[tid * Bc + c] = p;
                local_sum += p;
            }
            row_sum += local_sum;
        }
        __syncthreads();

        // Load V
        load_tile_g2s<Bc, d, NUM_THREADS>(
            V_ptr + kv_start * params.v_seq_stride,
            params.v_seq_stride, smem_V, valid_kv);
        __syncthreads();

        // O += P @ V (P from shared, V from shared)
        if (owns_row) {
            #pragma unroll
            for (int c = 0; c < Bc; c++) {
                float p = smem_P[tid * Bc + c];
                if (p == 0.0f) continue;
                const half2* v2 = reinterpret_cast<const half2*>(smem_V + c * d);
                constexpr int d2 = d / 2;
                #pragma unroll
                for (int j = 0; j < d2; j++) {
                    half2 v = v2[j];
                    my_O[j * 2]     += p * __half2float(v.x);
                    my_O[j * 2 + 1] += p * __half2float(v.y);
                }
            }
        }
        __syncthreads();
    }

    // Finalize
    if (owns_row) {
        float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
        half2* o_dst2 = reinterpret_cast<half2*>(O_ptr + tid * params.o_seq_stride);
        constexpr int d2 = d / 2;
        #pragma unroll
        for (int j = 0; j < d2; j++) {
            half2 val;
            val.x = __float2half(my_O[j * 2] * inv_sum);
            val.y = __float2half(my_O[j * 2 + 1] * inv_sum);
            o_dst2[j] = val;
        }
        L_ptr[tid] = row_max + logf(fmaxf(row_sum, 1e-20f));
    }
}

// ---------------------------------------------------------------------------
// Forward kernel — Pascal — Variable-length (packed sequences)
// ---------------------------------------------------------------------------
template <typename Tile>
__global__ void __launch_bounds__(Tile::kNumWarps * 32)
flash_attn_fwd_pascal_varlen_kernel(FlashAttnVarlenParams params) {

    constexpr int Br = Tile::Br;
    constexpr int Bc = Tile::Bc;
    constexpr int d  = Tile::d;
    constexpr int NUM_THREADS = Tile::kNumWarps * 32;

    const int h_idx = blockIdx.y;
    const int block_x = blockIdx.x;

    // Binary search: find which sequence this block belongs to
    int seq_idx = 0;
    {
        int lo = 0, hi = params.num_seqs;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (params.block_offsets[mid + 1] <= block_x)
                lo = mid + 1;
            else
                hi = mid;
        }
        seq_idx = lo;
    }

    const int local_block = block_x - params.block_offsets[seq_idx];
    const int q_start_tok = params.cu_seqlens_q[seq_idx];
    const int k_start_tok = params.cu_seqlens_k[seq_idx];
    const int seq_len_q = params.cu_seqlens_q[seq_idx + 1] - q_start_tok;
    const int seq_len_k = params.cu_seqlens_k[seq_idx + 1] - k_start_tok;

    const int q_row_start = local_block * Br;
    if (q_row_start >= seq_len_q) return;
    const int valid_q = min(Br, seq_len_q - q_row_start);

    const int kv_h_idx = h_idx / (params.num_heads / params.num_heads_k);

    // Pointers into packed [total, H, d] tensors
    const half* Q_ptr = params.Q + (q_start_tok + q_row_start) * params.q_token_stride
                                 + h_idx * params.q_head_stride;
    const half* K_ptr = params.K + k_start_tok * params.k_token_stride
                                 + kv_h_idx * params.k_head_stride;
    const half* V_ptr = params.V + k_start_tok * params.v_token_stride
                                 + kv_h_idx * params.v_head_stride;
    half* O_ptr = params.O + (q_start_tok + q_row_start) * params.o_token_stride
                           + h_idx * params.o_head_stride;

    // Shared memory
    extern __shared__ char smem_raw[];
    half*  smem_Q = reinterpret_cast<half*>(smem_raw);
    half*  smem_K = smem_Q + Br * d;
    half*  smem_V = smem_K + Bc * d;
    float* smem_P = reinterpret_cast<float*>(smem_V + Bc * d);

    const int tid = threadIdx.x;
    const bool owns_row = (tid < Br) && (tid < valid_q);

    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float my_O[d];
    #pragma unroll
    for (int j = 0; j < d; j++) my_O[j] = 0.0f;

    // Load Q — stride is token_stride
    load_tile_g2s<Br, d, NUM_THREADS>(Q_ptr, params.q_token_stride, smem_Q, valid_q);
    __syncthreads();

    // Main loop over K/V blocks (within this sequence)
    const int nkv = cdiv(seq_len_k, Bc);
    const int kv_limit = params.is_causal ? min(nkv, cdiv(q_row_start + Br, Bc)) : nkv;

    for (int kv_blk = 0; kv_blk < kv_limit; kv_blk++) {
        const int kv_start = kv_blk * Bc;
        const int valid_kv = min(Bc, seq_len_k - kv_start);

        // Load K
        load_tile_g2s<Bc, d, NUM_THREADS>(
            K_ptr + kv_start * params.k_token_stride,
            params.k_token_stride, smem_K, valid_kv);
        __syncthreads();

        // Compute S, softmax → P in shared memory
        if (owns_row) {
            float local_max = -INFINITY;

            #pragma unroll
            for (int c = 0; c < Bc; c++) {
                float s;
                if ((kv_start + c) >= seq_len_k ||
                    (params.is_causal && (kv_start + c) > (q_row_start + tid))) {
                    s = -INFINITY;
                } else {
                    s = half2_dot<d>(smem_Q + tid * d, smem_K + c * d) * params.softmax_scale;
                }
                smem_P[tid * Bc + c] = s;
                local_max = fmaxf(local_max, s);
            }

            float new_max = fmaxf(row_max, local_max);
            float rescale = expf(row_max - new_max);
            row_max = new_max;
            row_sum *= rescale;
            #pragma unroll
            for (int j = 0; j < d; j++) my_O[j] *= rescale;

            float local_sum = 0.0f;
            #pragma unroll
            for (int c = 0; c < Bc; c++) {
                float s = smem_P[tid * Bc + c];
                float p = (s == -INFINITY) ? 0.0f : expf(s - new_max);
                smem_P[tid * Bc + c] = p;
                local_sum += p;
            }
            row_sum += local_sum;
        }
        __syncthreads();

        // Load V
        load_tile_g2s<Bc, d, NUM_THREADS>(
            V_ptr + kv_start * params.v_token_stride,
            params.v_token_stride, smem_V, valid_kv);
        __syncthreads();

        // O += P @ V
        if (owns_row) {
            #pragma unroll
            for (int c = 0; c < Bc; c++) {
                float p = smem_P[tid * Bc + c];
                if (p == 0.0f) continue;
                const half2* v2 = reinterpret_cast<const half2*>(smem_V + c * d);
                constexpr int d2 = d / 2;
                #pragma unroll
                for (int j = 0; j < d2; j++) {
                    half2 v = v2[j];
                    my_O[j * 2]     += p * __half2float(v.x);
                    my_O[j * 2 + 1] += p * __half2float(v.y);
                }
            }
        }
        __syncthreads();
    }

    // Finalize — write O and L with packed strides
    if (owns_row) {
        float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
        half2* o_dst2 = reinterpret_cast<half2*>(O_ptr + tid * params.o_token_stride);
        constexpr int d2 = d / 2;
        #pragma unroll
        for (int j = 0; j < d2; j++) {
            half2 val;
            val.x = __float2half(my_O[j * 2] * inv_sum);
            val.y = __float2half(my_O[j * 2 + 1] * inv_sum);
            o_dst2[j] = val;
        }
        params.L[(q_start_tok + q_row_start + tid) * params.num_heads + h_idx] =
            row_max + logf(fmaxf(row_sum, 1e-20f));
    }
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------
// Helper macro to launch Pascal forward kernel
#define PASCAL_FWD_LAUNCH(TileType, params, stream, nbh)                          \
    do {                                                                           \
        using T = TileType;                                                        \
        constexpr int smem = (T::Br * T::d + 2 * T::Bc * T::d) * sizeof(half)    \
                           + T::Br * T::Bc * sizeof(float);                        \
        dim3 grid(cdiv((params).seq_len, T::Br), nbh);                            \
        dim3 block(T::kNumWarps * 32);                                             \
        flash_attn_fwd_pascal_kernel<T><<<grid, block, smem, stream>>>(params);    \
    } while (0)

void flash_attn_fwd_pascal(FlashAttnParams& params, cudaStream_t stream) {
    const int nbh = params.batch_size * params.num_heads;

    switch (params.head_dim) {
        case 32:  PASCAL_FWD_LAUNCH(TilePascal_d32,  params, stream, nbh); break;
        case 64:  PASCAL_FWD_LAUNCH(TilePascal_d64,  params, stream, nbh); break;
        case 96:  PASCAL_FWD_LAUNCH(TilePascal_d96,  params, stream, nbh); break;
        case 128: PASCAL_FWD_LAUNCH(TilePascal_d128, params, stream, nbh); break;
        case 256: PASCAL_FWD_LAUNCH(TilePascal_d256, params, stream, nbh); break;
        default:  throw std::runtime_error("Unsupported head_dim for Pascal forward");
    }
}

#undef PASCAL_FWD_LAUNCH

// ---------------------------------------------------------------------------
// Host launcher — variable-length
// ---------------------------------------------------------------------------
// Helper macro to launch Pascal varlen forward kernel
#define PASCAL_FWD_VARLEN_LAUNCH(TileType, params, stream, total_q_blocks)               \
    do {                                                                                  \
        using T = TileType;                                                               \
        constexpr int smem = (T::Br * T::d + 2 * T::Bc * T::d) * sizeof(half)           \
                           + T::Br * T::Bc * sizeof(float);                               \
        dim3 grid(total_q_blocks, (params).num_heads);                                    \
        dim3 block(T::kNumWarps * 32);                                                    \
        flash_attn_fwd_pascal_varlen_kernel<T><<<grid, block, smem, stream>>>(params);    \
    } while (0)

void flash_attn_fwd_pascal_varlen(FlashAttnVarlenParams& params,
                                   int total_q_blocks, cudaStream_t stream) {
    switch (params.head_dim) {
        case 32:  PASCAL_FWD_VARLEN_LAUNCH(TilePascal_d32,  params, stream, total_q_blocks); break;
        case 64:  PASCAL_FWD_VARLEN_LAUNCH(TilePascal_d64,  params, stream, total_q_blocks); break;
        case 96:  PASCAL_FWD_VARLEN_LAUNCH(TilePascal_d96,  params, stream, total_q_blocks); break;
        case 128: PASCAL_FWD_VARLEN_LAUNCH(TilePascal_d128, params, stream, total_q_blocks); break;
        case 256: PASCAL_FWD_VARLEN_LAUNCH(TilePascal_d256, params, stream, total_q_blocks); break;
        default:  throw std::runtime_error("Unsupported head_dim for Pascal varlen forward");
    }
}

#undef PASCAL_FWD_VARLEN_LAUNCH

}  // namespace flash_attn_legacy
