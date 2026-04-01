// ============================================================================
// Flash Attention v2 — Backward Kernel
// Works on both Pascal and Volta (CUDA core based).
//
// Key optimization vs v1: multi-thread per KV row
//   THREADS_PER_KV = NUM_THREADS / Bc threads collaborate on each KV row.
//   The Q-row inner loop is split across these threads (stride pattern).
//   dK/dV are accumulated in per-thread registers, then reduced via warp
//   shuffle (__shfl_down_sync with width = THREADS_PER_KV).
//   dQ is accumulated in a separate FP32 buffer with atomicAdd (unchanged).
//
//   d=64  → THREADS_PER_KV = 128/32 = 4  (4× speedup on Q-row loop)
//   d=128 → THREADS_PER_KV = 128/16 = 8  (8× speedup on Q-row loop)
//
// Algorithm (Flash Attn v2 backward — KV-outer loop):
//   For each KV block (j):
//     Load K_j, V_j into shared
//     Init dK_j = 0, dV_j = 0 in registers
//     For each Q head (GQA):
//       For each Q block (i):
//         Load Q_i, dO_i into shared
//         For qi in stride pattern (split across THREADS_PER_KV):
//           Recompute S_ij = Q_i @ K_j^T * scale
//           Recompute P_ij = exp(S_ij - L_i)
//           dV_j += P_ij * dO_i
//           dP_ij = dO_i @ V_j^T
//           dS_ij = P_ij * (dP_ij - D_i)
//           dQ_i += dS_ij @ K_j * scale     (atomicAdd to FP32 buffer)
//           dK_j += dS_ij * Q_i * scale
//     Reduce dK_j, dV_j across threads via warp shuffle
//     Store dK_j, dV_j
// ============================================================================

#include "flash_attn_common.h"

namespace flash_attn_legacy {

// ---------------------------------------------------------------------------
// Vectorized load
// ---------------------------------------------------------------------------
template <int ROWS, int COLS, int NUM_THREADS>
__device__ __forceinline__ void bwd_load_tile(
    const half* __restrict__ src,
    int src_row_stride,
    half* __restrict__ dst,
    int num_valid_rows
) {
    static_assert(COLS % 2 == 0, "COLS must be even");
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
// Precompute D = rowsum(dO * O) — launched as a separate kernel
// D is [B, H, N] float
// ---------------------------------------------------------------------------
__global__ void precompute_D_kernel(
    const half* __restrict__ dO,   // [B, H, N, d]
    const half* __restrict__ O,    // [B, H, N, d]
    float* __restrict__ D,         // [B, H, N]
    int batch_size, int num_heads, int seq_len, int head_dim,
    int o_batch_stride, int o_head_stride, int o_seq_stride,
    int l_batch_stride, int l_head_stride
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * num_heads * seq_len;
    if (idx >= total) return;

    const int n = idx % seq_len;
    const int h = (idx / seq_len) % num_heads;
    const int b = idx / (num_heads * seq_len);

    const half* dO_row = dO + b * o_batch_stride + h * o_head_stride + n * o_seq_stride;
    const half* O_row  = O  + b * o_batch_stride + h * o_head_stride + n * o_seq_stride;

    float sum = 0.0f;
    const half2* dO2 = reinterpret_cast<const half2*>(dO_row);
    const half2* O2  = reinterpret_cast<const half2*>(O_row);
    const int d2 = head_dim / 2;
    for (int j = 0; j < d2; j++) {
        half2 a = dO2[j];
        half2 b_val = O2[j];
        sum += __half2float(a.x) * __half2float(b_val.x)
             + __half2float(a.y) * __half2float(b_val.y);
    }

    D[b * l_batch_stride + h * l_head_stride + n] = sum;
}

// ---------------------------------------------------------------------------
// Backward kernel — KV-outer loop, multi-thread per KV row
// ---------------------------------------------------------------------------
template <typename Tile>
__global__ void __launch_bounds__(Tile::kNumWarps * 32)
flash_attn_bwd_kernel(FlashAttnBwdParams params) {

    constexpr int Br = Tile::Br;
    constexpr int Bc = Tile::Bc;
    constexpr int D  = Tile::d;
    constexpr int NUM_THREADS = Tile::kNumWarps * 32;
    // How many threads collaborate on each KV row
    constexpr int THREADS_PER_KV = NUM_THREADS / Bc;
    static_assert(THREADS_PER_KV >= 1 && (THREADS_PER_KV & (THREADS_PER_KV - 1)) == 0,
                  "THREADS_PER_KV must be a power of 2");

    // Thread-to-KV-row mapping
    const int tid = threadIdx.x;
    const int kv_row = tid / THREADS_PER_KV;   // which KV row this thread owns
    const int lane   = tid % THREADS_PER_KV;   // position within the KV row group

    // GQA: grid is over (B * num_heads_k), each block handles one KV head
    // and accumulates from all Q heads that map to it.
    const int bh_idx = blockIdx.y;
    const int b_idx = bh_idx / params.num_heads_k;
    const int kv_h_idx = bh_idx % params.num_heads_k;
    const int heads_per_kv = params.num_heads / params.num_heads_k;
    const int kv_blk = blockIdx.x;
    const int kv_start = kv_blk * Bc;
    if (kv_start >= params.seq_len) return;
    const int valid_kv = min(Bc, params.seq_len - kv_start);

    const bool owns_kv_row = (kv_row < Bc) && (kv_row < valid_kv);

    // Base pointers for K, V (using kv_h_idx)
    const half* K_blk = params.K + b_idx * params.k_batch_stride
                                 + kv_h_idx * params.k_head_stride
                                 + kv_start * params.k_seq_stride;
    const half* V_blk = params.V + b_idx * params.v_batch_stride
                                 + kv_h_idx * params.v_head_stride
                                 + kv_start * params.v_seq_stride;

    // Shared memory layout:
    //   smem_K  [Bc, D]  half
    //   smem_V  [Bc, D]  half
    //   smem_Q  [Br, D]  half   — reloaded each Q block
    //   smem_dO [Br, D]  half   — reloaded each Q block
    extern __shared__ char smem_raw[];
    half* smem_K  = reinterpret_cast<half*>(smem_raw);
    half* smem_V  = smem_K + Bc * D;
    half* smem_Q  = smem_V + Bc * D;
    half* smem_dO = smem_Q + Br * D;

    // dK, dV accumulators in registers (each thread has partial sums)
    float my_dK[D];
    float my_dV[D];
    #pragma unroll
    for (int j = 0; j < D; j++) { my_dK[j] = 0.0f; my_dV[j] = 0.0f; }

    // Load K, V (stay for entire block)
    bwd_load_tile<Bc, D, NUM_THREADS>(K_blk, params.k_seq_stride, smem_K, valid_kv);
    bwd_load_tile<Bc, D, NUM_THREADS>(V_blk, params.v_seq_stride, smem_V, valid_kv);
    __syncthreads();

    // Loop over all Q heads that map to this KV head
    const int num_q_blocks = cdiv(params.seq_len, Br);

    for (int qh = 0; qh < heads_per_kv; qh++) {
        const int h_idx = kv_h_idx * heads_per_kv + qh;
        // Causal optimization: skip Q blocks that are entirely before this KV block
        const int q_start_blk = params.is_causal ? (kv_start / Br) : 0;

    for (int q_blk = q_start_blk; q_blk < num_q_blocks; q_blk++) {
        const int q_start = q_blk * Br;
        const int valid_q = min(Br, params.seq_len - q_start);

        // Load Q, dO for this (q_head, q_block)
        const half* Q_blk_ptr = params.Q + b_idx * params.q_batch_stride
                                         + h_idx * params.q_head_stride
                                         + q_start * params.q_seq_stride;
        const half* dO_blk_ptr = params.dO + b_idx * params.o_batch_stride
                                           + h_idx * params.o_head_stride
                                           + q_start * params.o_seq_stride;

        bwd_load_tile<Br, D, NUM_THREADS>(Q_blk_ptr, params.q_seq_stride, smem_Q, valid_q);
        bwd_load_tile<Br, D, NUM_THREADS>(dO_blk_ptr, params.o_seq_stride, smem_dO, valid_q);
        __syncthreads();

        // L, D, dQ indexed by Q head
        const float* L_ptr = params.L + b_idx * params.l_batch_stride
                                      + h_idx * params.l_head_stride
                                      + q_start;
        const float* D_ptr = params.D + b_idx * params.l_batch_stride
                                      + h_idx * params.l_head_stride
                                      + q_start;
        float* dQ_ptr = params.dQ + b_idx * params.dq_batch_stride
                                  + h_idx * params.dq_head_stride
                                  + q_start * params.dq_seq_stride;

        // Each thread in the KV row group processes a strided subset of Q rows
        if (owns_kv_row) {
            for (int qi = lane; qi < Br && (q_start + qi) < params.seq_len; qi += THREADS_PER_KV) {
                // Causal check
                if (params.is_causal && (kv_start + kv_row) > (q_start + qi)) continue;

                // Recompute S = Q[qi] . K[kv_row] * scale
                float s = 0.0f;
                const half2* q2 = reinterpret_cast<const half2*>(smem_Q + qi * D);
                const half2* k2 = reinterpret_cast<const half2*>(smem_K + kv_row * D);
                constexpr int D2 = D / 2;
                #pragma unroll
                for (int j = 0; j < D2; j++) {
                    half2 a = q2[j], b_val = k2[j];
                    s += __half2float(a.x) * __half2float(b_val.x)
                       + __half2float(a.y) * __half2float(b_val.y);
                }
                s *= params.softmax_scale;

                // P = exp(S - L[qi])
                float li = L_ptr[qi];
                float p = expf(s - li);

                // D_i (precomputed)
                float di = D_ptr[qi];

                // dP = dO[qi] . V[kv_row]
                float dp = 0.0f;
                const half2* do2 = reinterpret_cast<const half2*>(smem_dO + qi * D);
                const half2* v2  = reinterpret_cast<const half2*>(smem_V + kv_row * D);
                #pragma unroll
                for (int j = 0; j < D2; j++) {
                    half2 a = do2[j], b_val = v2[j];
                    dp += __half2float(a.x) * __half2float(b_val.x)
                        + __half2float(a.y) * __half2float(b_val.y);
                }

                // dS = P * (dP - D_i)
                float ds = p * (dp - di);

                // dS_scaled = dS * softmax_scale
                float ds_scaled = ds * params.softmax_scale;

                // Accumulate dV[kv_row] += P * dO[qi]
                // Accumulate dK[kv_row] += dS_scaled * Q[qi]
                #pragma unroll
                for (int j = 0; j < D2; j++) {
                    half2 do_val = do2[j];
                    my_dV[j * 2]     += p * __half2float(do_val.x);
                    my_dV[j * 2 + 1] += p * __half2float(do_val.y);

                    half2 q_val = q2[j];
                    my_dK[j * 2]     += ds_scaled * __half2float(q_val.x);
                    my_dK[j * 2 + 1] += ds_scaled * __half2float(q_val.y);
                }

                // dQ[qi] += dS_scaled * K[kv_row] — atomicAdd to FP32 buffer
                float* dq_row = dQ_ptr + qi * params.dq_seq_stride;
                #pragma unroll
                for (int j = 0; j < D2; j++) {
                    half2 k_val = k2[j];
                    atomicAdd(&dq_row[j * 2],     ds_scaled * __half2float(k_val.x));
                    atomicAdd(&dq_row[j * 2 + 1], ds_scaled * __half2float(k_val.y));
                }
            }
        }
        __syncthreads();
    }  // end q_blk loop
    }  // end qh (GQA Q-head) loop

    // -----------------------------------------------------------------------
    // Reduce dK, dV across lanes within each KV row group via warp shuffle.
    // THREADS_PER_KV is always a power of 2 and <= 32, so all threads in a
    // group are within the same warp — no __syncthreads needed.
    // -----------------------------------------------------------------------
    if (THREADS_PER_KV > 1) {
        #pragma unroll
        for (int offset = THREADS_PER_KV / 2; offset > 0; offset /= 2) {
            #pragma unroll
            for (int j = 0; j < D; j++) {
                my_dK[j] += __shfl_down_sync(0xFFFFFFFF, my_dK[j], offset, THREADS_PER_KV);
                my_dV[j] += __shfl_down_sync(0xFFFFFFFF, my_dV[j], offset, THREADS_PER_KV);
            }
        }
    }

    // Write dK, dV — only lane 0 of each valid group writes to global memory
    if (lane == 0 && owns_kv_row) {
        half* dK_out = params.dK + b_idx * params.dk_batch_stride
                                 + kv_h_idx * params.dk_head_stride
                                 + (kv_start + kv_row) * params.dk_seq_stride;
        half* dV_out = params.dV + b_idx * params.dv_batch_stride
                                 + kv_h_idx * params.dv_head_stride
                                 + (kv_start + kv_row) * params.dv_seq_stride;

        half2* dk2 = reinterpret_cast<half2*>(dK_out);
        half2* dv2 = reinterpret_cast<half2*>(dV_out);
        constexpr int D2 = D / 2;
        #pragma unroll
        for (int j = 0; j < D2; j++) {
            half2 dk_val, dv_val;
            dk_val.x = __float2half(my_dK[j * 2]);
            dk_val.y = __float2half(my_dK[j * 2 + 1]);
            dv_val.x = __float2half(my_dV[j * 2]);
            dv_val.y = __float2half(my_dV[j * 2 + 1]);
            dk2[j] = dk_val;
            dv2[j] = dv_val;
        }
    }
}

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------

void precompute_D(
    const half* dO, const half* O, float* D,
    int B, int H, int N, int d,
    int o_batch_stride, int o_head_stride, int o_seq_stride,
    int l_batch_stride, int l_head_stride,
    cudaStream_t stream
) {
    const int total = B * H * N;
    const int block_size = 256;
    const int grid_size = cdiv(total, block_size);
    precompute_D_kernel<<<grid_size, block_size, 0, stream>>>(
        dO, O, D, B, H, N, d,
        o_batch_stride, o_head_stride, o_seq_stride,
        l_batch_stride, l_head_stride
    );
}

void flash_attn_bwd_launch(FlashAttnBwdParams& params, cudaStream_t stream) {
    // Grid Y = B * num_heads_k (one block per KV head, loops over Q heads internally)
    const int nbh = params.batch_size * params.num_heads_k;

    if (params.head_dim == 64) {
        using T = TileBwd_d64;
        constexpr int smem = (T::Bc * T::d + T::Bc * T::d + T::Br * T::d + T::Br * T::d) * sizeof(half);
        dim3 grid(cdiv(params.seq_len, T::Bc), nbh);
        dim3 block(T::kNumWarps * 32);
        flash_attn_bwd_kernel<T><<<grid, block, smem, stream>>>(params);
    } else {
        using T = TileBwd_d128;
        constexpr int smem = (T::Bc * T::d + T::Bc * T::d + T::Br * T::d + T::Br * T::d) * sizeof(half);
        dim3 grid(cdiv(params.seq_len, T::Bc), nbh);
        dim3 block(T::kNumWarps * 32);

        FLASH_ATTN_CHECK_CUDA(cudaFuncSetAttribute(flash_attn_bwd_kernel<T>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        flash_attn_bwd_kernel<T><<<grid, block, smem, stream>>>(params);
    }
}

}  // namespace flash_attn_legacy
