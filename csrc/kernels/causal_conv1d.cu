// ============================================================================
// Depthwise causal conv1d of Gated DeltaNet layers (Qwen3.5, Qwen3-Next, ...;
// docs/PLAN.md P6.10 b, docs/analysis/kernels/qwen35_conv1d.md): ConvParams.
//
//   forward:  y[t] = act(z[t]), z[t] = sum_k w[k] x[t - (W-1) + k] + bias
//             (per channel, x = 0 before the first token), act = SiLU or none
//   backward: g[t] = dy[t] act'(z[t]) (z recomputed), dx[s] = sum_k w[k]
//             g[s + (W-1) - k] (g = 0 past the last token), dW[k] = sum_t g[t]
//             x[t - (W-1) + k], db = sum_t g[t]
//   update:   one decode step on a conv state [B, C, state_len] (transformers'
//             causal_conv1d_update): y for the new tokens, the state shifted in
//             place
//
// Memory-bound (W multiply-adds per element read): the forward and backward
// read x / dy channel-last with 16-byte loads (8 fp16 / bf16 or 4 fp32
// channels) along runs of kRun tokens per thread, the window of the previous
// W-1 tokens in registers; the first W-1 tokens of a run are re-read from
// global memory (a halo, no dependency between threads). Accumulation in fp32,
// in the same order in the forward and the backward's recompute; one rounding
// of each output. dW / db: per-run fp32 partials reduced in run order by a
// second kernel (deterministic; the summation order does not depend on the
// GPU, the SFU's exp in SiLU' may). No arch-specific code.
// ============================================================================

#include "flash_attn_common.h"

#include <cuda_bf16.h>

namespace flash_attn_legacy {

namespace {

// Tokens per thread (forward, backward). 32: the forward at 1.82x the bandwidth floor at 4k tokens on the P100 (64:
// 2.01x, 16: 1.83x and a slower dW pass)
constexpr int kRun = 32;
constexpr int kThreads = 128;   // channel groups per block

template <typename T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f<half>(half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float to_f<float>(float v) { return v; }

template <typename T> __device__ __forceinline__ T from_f(float v);
template <> __device__ __forceinline__ half from_f<half>(float v) { return __float2half_rn(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }
template <> __device__ __forceinline__ float from_f<float>(float v) { return v; }

// N consecutive elements at p as fp32 (N * sizeof(T) == 16: one 16-byte load; N == 1: one element)
template <typename T, int N>
__device__ __forceinline__ void load(const T* p, float (&f)[N]) {
    if constexpr (N * sizeof(T) == 16) {
        alignas(16) T v[N];
        *reinterpret_cast<uint4*>(v) = __ldg(reinterpret_cast<const uint4*>(p));
        #pragma unroll
        for (int i = 0; i < N; i++) f[i] = to_f(v[i]);
    } else {
        #pragma unroll
        for (int i = 0; i < N; i++) f[i] = to_f(p[i]);
    }
}

template <typename T, int N>
__device__ __forceinline__ void store(T* p, const float (&f)[N]) {
    if constexpr (N * sizeof(T) == 16) {
        alignas(16) T v[N];
        #pragma unroll
        for (int i = 0; i < N; i++) v[i] = from_f<T>(f[i]);
        *reinterpret_cast<uint4*>(p) = *reinterpret_cast<const uint4*>(v);
    } else {
        #pragma unroll
        for (int i = 0; i < N; i++) p[i] = from_f<T>(f[i]);
    }
}

__device__ __forceinline__ float sigmoid(float z) { return 1.0f / (1.0f + __expf(-z)); }

// z = sum_k w[k] x[k] + b over the window x[0..W-1] (oldest first): the one order of the forward and the recompute
template <int W>
__device__ __forceinline__ float conv_z(const float (&w)[W], const float (&x)[W], float b) {
    float z = 0.0f;
    #pragma unroll
    for (int k = 0; k < W; k++) z = fmaf(w[k], x[k], z);
    return z + b;
}

// The weights and bias of N channels from c0 (WT: the weight's dtype)
template <typename WT, int N, int W>
__device__ __forceinline__ void load_weights(const ConvParams& p, int c0, float (&w)[N][W], float (&b)[N]) {
    const WT* wp = static_cast<const WT*>(p.weight);
    const WT* bp = static_cast<const WT*>(p.bias);
    #pragma unroll
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int k = 0; k < W; k++) w[i][k] = to_f(wp[index_t(c0 + i) * W + k]);
        b[i] = bp != nullptr ? to_f(bp[c0 + i]) : 0.0f;
    }
}

// Thread (channel group, run of kRun tokens, batch row): blockIdx.x * kThreads + threadIdx.x, blockIdx.y, blockIdx.z
template <typename T, typename WT, int N, int W>
__global__ void __launch_bounds__(kThreads) conv_fwd_kernel(const ConvParams p) {
    const int c0 = (blockIdx.x * kThreads + threadIdx.x) * N;
    if (c0 >= p.channels) return;
    const int t0 = blockIdx.y * kRun, t1 = min(t0 + kRun, p.seqlen), b = blockIdx.z;
    const T* x = static_cast<const T*>(p.x) + index_t(b) * p.x_batch_stride + c0;
    T* y = static_cast<T*>(p.y) + index_t(b) * p.y_batch_stride + c0;
    float w[N][W], bias[N];
    load_weights<WT, N, W>(p, c0, w, bias);
    float win[W][N];  // x[t - (W-1) .. t], oldest first
    #pragma unroll
    for (int k = 0; k < W - 1; k++) {
        const int t = t0 - (W - 1) + k;
        if (t >= 0) load<T, N>(x + index_t(t) * p.x_seq_stride, win[k + 1]);
        else {
            #pragma unroll
            for (int i = 0; i < N; i++) win[k + 1][i] = 0.0f;
        }
    }
    for (int t = t0; t < t1; t++) {
        #pragma unroll
        for (int k = 0; k < W - 1; k++) {
            #pragma unroll
            for (int i = 0; i < N; i++) win[k][i] = win[k + 1][i];
        }
        load<T, N>(x + index_t(t) * p.x_seq_stride, win[W - 1]);
        float out[N];
        #pragma unroll
        for (int i = 0; i < N; i++) {
            float xs[W];
            #pragma unroll
            for (int k = 0; k < W; k++) xs[k] = win[k][i];
            const float z = conv_z<W>(w[i], xs, bias[i]);
            out[i] = p.silu ? z * sigmoid(z) : z;
        }
        store<T, N>(y + index_t(t) * p.y_seq_stride, out);
    }
}

// The backward of a run [t0, t1): g over [t0, t1 + W - 1) (z recomputed from x[t - (W-1) .. t]), dx over [t0, t1),
// dW / db partials over [t0, t1) into partial[(b * runs + run), c, 0..W] (when p.partial)
template <typename T, typename WT, int N, int W>
__global__ void __launch_bounds__(kThreads) conv_bwd_kernel(const ConvParams p) {
    const int c0 = (blockIdx.x * kThreads + threadIdx.x) * N;
    if (c0 >= p.channels) return;
    const int run = blockIdx.y, t0 = run * kRun, t1 = min(t0 + kRun, p.seqlen), b = blockIdx.z;
    const T* x = static_cast<const T*>(p.x) + index_t(b) * p.x_batch_stride + c0;
    const T* dy = static_cast<const T*>(p.dy) + index_t(b) * p.dy_batch_stride + c0;
    T* dx = static_cast<T*>(p.dx) + index_t(b) * p.y_batch_stride + c0;
    float w[N][W], bias[N];
    load_weights<WT, N, W>(p, c0, w, bias);
    float xw[W][N];  // x[t - (W-1) .. t]
    float gw[W][N];  // g[t - (W-1) .. t]
    #pragma unroll
    for (int k = 0; k < W - 1; k++) {
        const int t = t0 - (W - 1) + k;
        if (t >= 0) load<T, N>(x + index_t(t) * p.x_seq_stride, xw[k + 1]);
        else {
            #pragma unroll
            for (int i = 0; i < N; i++) xw[k + 1][i] = 0.0f;
        }
    }
    float dw[N][W], db[N];
    #pragma unroll
    for (int i = 0; i < N; i++) {
        db[i] = 0.0f;
        #pragma unroll
        for (int k = 0; k < W; k++) gw[k][i] = 0.0f;
        #pragma unroll
        for (int k = 0; k < W; k++) dw[i][k] = 0.0f;
    }
    // t runs to t1 - 1 + (W-1): the last dx of the run needs g up to there (0 past the sequence)
    for (int t = t0; t < t1 + W - 1; t++) {
        #pragma unroll
        for (int k = 0; k < W - 1; k++) {
            #pragma unroll
            for (int i = 0; i < N; i++) { xw[k][i] = xw[k + 1][i]; gw[k][i] = gw[k + 1][i]; }
        }
        if (t < p.seqlen) {
            float d[N];
            load<T, N>(x + index_t(t) * p.x_seq_stride, xw[W - 1]);
            load<T, N>(dy + index_t(t) * p.dy_seq_stride, d);
            #pragma unroll
            for (int i = 0; i < N; i++) {
                float xs[W];
                #pragma unroll
                for (int k = 0; k < W; k++) xs[k] = xw[k][i];
                float g = d[i];
                if (p.silu) {
                    const float z = conv_z<W>(w[i], xs, bias[i]), s = sigmoid(z);
                    g *= s * (1.0f + z * (1.0f - s));
                }
                gw[W - 1][i] = g;
                if (t < t1) {
                    db[i] += g;
                    #pragma unroll
                    for (int k = 0; k < W; k++) dw[i][k] = fmaf(g, xs[k], dw[i][k]);
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < N; i++) gw[W - 1][i] = 0.0f;
        }
        const int s = t - (W - 1);  // dx[s] = sum_k w[k] g[s + (W-1) - k], g[s + (W-1) - k] = g[t - k] = gw[W-1-k]
        if (s >= t0) {
            float out[N];
            #pragma unroll
            for (int i = 0; i < N; i++) {
                float acc = 0.0f;
                #pragma unroll
                for (int k = 0; k < W; k++) acc = fmaf(w[i][k], gw[W - 1 - k][i], acc);
                out[i] = acc;
            }
            store<T, N>(dx + index_t(s) * p.y_seq_stride, out);
        }
    }
    if (p.partial != nullptr) {
        float* part = p.partial + (index_t(b) * p.runs + run) * p.channels * (W + 1);
        #pragma unroll
        for (int i = 0; i < N; i++) {
            #pragma unroll
            for (int k = 0; k < W; k++) part[index_t(c0 + i) * (W + 1) + k] = dw[i][k];
            part[index_t(c0 + i) * (W + 1) + W] = db[i];
        }
    }
}

// dW / db: the partials summed over (batch row, run) in order, one thread per (channel, tap or bias)
template <typename WT>
__global__ void conv_reduce_kernel(const ConvParams p) {
    const int W = p.width, e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= p.channels * (W + 1)) return;
    const index_t n = index_t(p.batch) * p.runs, stride = index_t(p.channels) * (W + 1);
    float acc = 0.0f;
    for (index_t r = 0; r < n; r++) acc += p.partial[r * stride + e];
    const int c = e / (W + 1), k = e % (W + 1);
    if (k < W) {
        if (p.dweight != nullptr) static_cast<WT*>(p.dweight)[index_t(c) * W + k] = from_f<WT>(acc);
    } else if (p.dbias != nullptr) {
        static_cast<WT*>(p.dbias)[c] = from_f<WT>(acc);
    }
}

__device__ __forceinline__ float load_any(const void* p, index_t i, int dtype) {
    switch (dtype) {
        case 0: return __half2float(static_cast<const half*>(p)[i]);
        case 1: return __bfloat162float(static_cast<const __nv_bfloat16*>(p)[i]);
        default: return static_cast<const float*>(p)[i];
    }
}

__device__ __forceinline__ void store_any(void* p, index_t i, int dtype, float v) {
    switch (dtype) {
        case 0: static_cast<half*>(p)[i] = __float2half_rn(v); break;
        case 1: static_cast<__nv_bfloat16*>(p)[i] = __float2bfloat16_rn(v); break;
        default: static_cast<float*>(p)[i] = v;
    }
}

// One decode step per (batch row, channel): the sequence is the state then the S new tokens; outputs for the new
// tokens, then the state shifted in place (state[i] = seq[S + i], read ahead of every write: ascending i). Tiny and
// launch-bound: runtime dtypes.
__global__ void conv_update_kernel(const ConvParams p) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, b = blockIdx.y;
    if (c >= p.channels) return;
    const int W = p.width, sl = p.state_len, S = p.seqlen;
    const index_t st = index_t(b) * p.state_batch_stride + index_t(c) * p.state_channel_stride;
    const index_t xb = index_t(b) * p.x_batch_stride + c, yb = index_t(b) * p.y_batch_stride + c;
    auto seq = [&](int i) {
        return i < sl ? load_any(p.state, st + index_t(i) * p.state_seq_stride, p.state_dtype)
                      : load_any(p.x, xb + index_t(i - sl) * p.x_seq_stride, p.x_dtype);
    };
    const float bias = p.bias != nullptr ? load_any(p.bias, c, p.w_dtype) : 0.0f;
    for (int j = 0; j < S; j++) {
        float z = 0.0f;
        for (int k = 0; k < W; k++) {
            z = fmaf(load_any(p.weight, index_t(c) * W + k, p.w_dtype), seq(sl + j - (W - 1) + k), z);
        }
        z += bias;
        store_any(p.y, yb + index_t(j) * p.y_seq_stride, p.x_dtype, p.silu ? z * sigmoid(z) : z);
    }
    for (int i = 0; i < sl; i++) store_any(p.state, st + index_t(i) * p.state_seq_stride, p.state_dtype, seq(S + i));
}

template <typename T, typename WT, int N, int W>
void launch(const ConvParams& p, bool backward, cudaStream_t stream) {
    const dim3 grid(cdiv(cdiv(p.channels, N), kThreads), cdiv(p.seqlen, kRun), p.batch);
    if (!backward) {
        conv_fwd_kernel<T, WT, N, W><<<grid, kThreads, 0, stream>>>(p);
        return;
    }
    conv_bwd_kernel<T, WT, N, W><<<grid, kThreads, 0, stream>>>(p);
    if (p.partial != nullptr) conv_reduce_kernel<WT><<<cdiv(p.channels * (W + 1), 256), 256, 0, stream>>>(p);
}

template <typename T, typename WT, int N>
void dispatch_width(const ConvParams& p, bool backward, cudaStream_t stream) {
    switch (p.width) {
        case 2: launch<T, WT, N, 2>(p, backward, stream); break;
        case 3: launch<T, WT, N, 3>(p, backward, stream); break;
        default: launch<T, WT, N, 4>(p, backward, stream);
    }
}

// vec: 16-byte loads (the host checked C % N == 0 and the alignment), else one element per thread
template <typename T, typename WT>
void dispatch_vec(const ConvParams& p, bool vec, bool backward, cudaStream_t stream) {
    if (vec) dispatch_width<T, WT, 16 / sizeof(T)>(p, backward, stream);
    else     dispatch_width<T, WT, 1>(p, backward, stream);
}

template <typename T>
void dispatch_weight(const ConvParams& p, bool vec, bool backward, cudaStream_t stream) {
    if (p.w_dtype == 2) dispatch_vec<T, float>(p, vec, backward, stream);
    else                dispatch_vec<T, T>(p, vec, backward, stream);  // the host checked: the weight is x's dtype
}

}  // namespace

// The runs of kRun tokens a sequence is split into (the backward's dW / db partials per run)
int causal_conv1d_runs(int seqlen) { return cdiv(seqlen, kRun); }

void causal_conv1d_launch(const ConvParams& p, bool vec, bool backward, cudaStream_t stream) {
    if (p.batch == 0 || p.channels == 0 || p.seqlen == 0) return;
    switch (p.x_dtype) {
        case 0: dispatch_weight<half>(p, vec, backward, stream); break;
        case 1: dispatch_weight<__nv_bfloat16>(p, vec, backward, stream); break;
        default: dispatch_weight<float>(p, vec, backward, stream);
    }
}

void causal_conv1d_update_launch(const ConvParams& p, cudaStream_t stream) {
    if (p.batch == 0 || p.channels == 0) return;
    conv_update_kernel<<<dim3(cdiv(p.channels, 128), p.batch), 128, 0, stream>>>(p);
}

}  // namespace flash_attn_legacy
