// ============================================================================
// Flash Attention Legacy — PyTorch C++ Extension Entry Point
// Runtime dispatch: Pascal vs Volta
// Supports MHA, GQA (grouped-query), and MQA (multi-query attention)
// ============================================================================

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <tuple>
#include "include/flash_attn_common.h"

namespace flash_attn_legacy {

// Forward declarations — batched
void flash_attn_fwd_turing(FlashAttnParams& params, cudaStream_t stream);
void flash_attn_fwd_volta(FlashAttnParams& params, cudaStream_t stream);
void flash_attn_fwd_pascal(FlashAttnParams& params, cudaStream_t stream);
// Forward declarations — variable-length
void flash_attn_fwd_turing_varlen(FlashAttnVarlenParams& params, int total_q_blocks, cudaStream_t stream);
void flash_attn_fwd_volta_varlen(FlashAttnVarlenParams& params, int total_q_blocks, cudaStream_t stream);
void flash_attn_fwd_pascal_varlen(FlashAttnVarlenParams& params, int total_q_blocks, cudaStream_t stream);
int flash_attn_fwd_turing_block_rows(int head_dim);
int flash_attn_fwd_volta_block_rows(int head_dim);
int flash_attn_fwd_pascal_block_rows(int head_dim);
void precompute_D(
    const half* dO, const half* O, float* D,
    const float* L, const float* sinks, float* dsinks,
    int B, int H, int N, int d,
    index_t o_batch_stride, index_t o_head_stride, index_t o_seq_stride,
    index_t l_batch_stride, index_t l_head_stride, index_t l_seq_stride,
    cudaStream_t stream);
void flash_attn_bwd_launch(FlashAttnBwdParams& params, int sm, int total_k_blocks, cudaStream_t stream);
int flash_attn_bwd_block_cols(int head_dim, int sm);
void flash_attn_decode(DecodeParams& params, bool bf16, cudaStream_t stream);
int flash_attn_decode_max_rows();
int flash_attn_decode_num_splits(int blocks, int sms, int max_len);
int flash_attn_decode_max_splits();

// Contiguous with a 16-byte aligned start: the kernels move rows with 128-bit
// accesses (row strides are multiples of 8 halves since head_dim is). A view
// whose start is not 16-byte aligned (e.g. offset by an odd number of halves)
// is cloned.
static torch::Tensor contiguous_aligned(const torch::Tensor& t) {
    torch::Tensor c = t.contiguous();
    if (reinterpret_cast<uintptr_t>(c.data_ptr()) % 16 != 0) c = c.clone();
    return c;
}

// FLASH_ATTN_LEGACY_FORCE_ARCH=pascal|volta forces the kernel family regardless of
// the GPU (testing only: e.g. run the Pascal CUDA-core kernels on a T4/Ampere card;
// correctness is checked, performance numbers are not representative).
// Architectures this build contains (from setup.py: FLASH_ATTN_LEGACY_ARCHS).
#ifndef FLASH_ATTN_LEGACY_SASS
#define FLASH_ATTN_LEGACY_SASS 60, 61, 70, 75
#endif
#ifndef FLASH_ATTN_LEGACY_PTX
#define FLASH_ATTN_LEGACY_PTX 75
#endif
static constexpr int kCompiledSass[] = {FLASH_ATTN_LEGACY_SASS};
static constexpr int kCompiledPtx = FLASH_ATTN_LEGACY_PTX;

static std::string compiled_archs_str() {
    std::string s;
    for (int a : kCompiledSass) s += (s.empty() ? "" : ", ") + std::to_string(a / 10) + "." + std::to_string(a % 10);
    if (kCompiledPtx) s += " (+PTX " + std::to_string(kCompiledPtx / 10) + "." + std::to_string(kCompiledPtx % 10) + ")";
    return s;
}

// A GPU runs the SASS of its major with the highest minor not above its own,
// else the PTX of a lower-or-equal arch (JIT). Returns the arch of the code that
// runs (e.g. 70 on a T4 built for 7.0 only: no sm_75 instructions), or fails
// with a clear message instead of CUDA's "no kernel image is available".
static int compiled_arch_for(int sm) {
    int best = 0;
    for (int a : kCompiledSass) {
        if (a / 10 == sm / 10 && a <= sm && a > best) best = a;
    }
    if (best) return best;
    if (kCompiledPtx && kCompiledPtx <= sm) return kCompiledPtx;
    TORCH_CHECK(false, "flash_attn_legacy was built for SM ", compiled_archs_str(),
                ", this GPU is SM ", sm / 10, ".", sm % 10, ". Rebuild with "
                "FLASH_ATTN_LEGACY_ARCHS=\"", sm / 10, ".", sm % 10, "\" pip install -e . --no-build-isolation");
}

std::vector<std::string> compiled_archs() {
    std::vector<std::string> out;
    for (int a : kCompiledSass) out.push_back(std::to_string(a / 10) + "." + std::to_string(a % 10));
    if (kCompiledPtx) out.push_back(std::to_string(kCompiledPtx / 10) + "." + std::to_string(kCompiledPtx % 10) + "+PTX");
    return out;
}

static int get_sm_version(int device_id = -1) {
    if (device_id < 0) FLASH_ATTN_CHECK_CUDA(cudaGetDevice(&device_id));
    int major, minor;
    FLASH_ATTN_CHECK_CUDA(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id));
    FLASH_ATTN_CHECK_CUDA(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id));
    const int sm = major * 10 + minor;
    const int code_sm = compiled_arch_for(sm);  // selects the kernel family below
    if (const char* force = std::getenv("FLASH_ATTN_LEGACY_FORCE_ARCH")) {
        const std::string f(force);
        if (f == "pascal") return 60;  // CUDA-core kernels run on any SM >= 6.0
        if (f == "volta") {
            TORCH_CHECK(code_sm >= 70, "FLASH_ATTN_LEGACY_FORCE_ARCH=volta needs tensor cores "
                        "(SM >= 7.0) and a build for them; this GPU is SM ", sm, ", built for ",
                        compiled_archs_str());
            return 70;
        }
        TORCH_CHECK(f.empty(), "FLASH_ATTN_LEGACY_FORCE_ARCH must be 'pascal' or 'volta' (unset: the GPU's own "
                    "family, mma.sync on SM >= 7.5), got '", f, "'");
    }
    return code_sm;
}

// Head dimensions with a kernel tile (72 and 80 run on the 80 / 96 tiles with
// zero-padded columns).
static void check_head_dim(int d) {
    TORCH_CHECK(d == 32 || d == 64 || d == 72 || d == 80 || d == 96 || d == 128 || d == 256,
                "head_dim must be 32, 64, 72, 80, 96, 128, or 256, got ", d);
}

// (window_left, window_right) of the kernels' KeyWindow from the API arguments:
// -1 = unbounded, causal sets right = 0 (as upstream flash-attn). A side that
// reaches every key of the longest possible sequence (max_q query rows, max_k
// keys) is unbounded: no masking work for it, and no int overflow in KeyWindow.
static std::pair<int, int> key_window(bool is_causal, int64_t left, int64_t right,
                                      int64_t max_q, int64_t max_k) {
    TORCH_CHECK(left >= -1 && right >= -1,
                "window_size entries must be >= -1 (-1 = unbounded), got (", left, ", ", right, ")");
    if (is_causal) right = 0;
    if (left >= max_k - 1) left = -1;   // first key of every row: <= N_kv - 1 - left <= 0
    if (right >= max_q - 1) right = -1; // last key of every row: >= N_kv - N_q + right >= N_kv - 1
    return {int(left), int(right)};
}

// Logit softcap of the kernels' ScoreMap: c * tanh(x / c), 0 = none (Gemma 2).
// It must stay a normal float with a finite 1 / c (ScoreMap divides by it).
static float logit_softcap(double softcap) {
    TORCH_CHECK(std::isfinite(softcap) && softcap >= 0.0,
                "softcap must be finite and >= 0 (0 = no softcapping), got ", softcap);
    const float c = static_cast<float>(softcap);
    TORCH_CHECK(softcap == 0.0 || (std::isnormal(c) && std::isfinite(1.0f / c)),
                "softcap out of the float range, got ", softcap);
    return c;
}

// Attention sinks (gpt-oss s_aux): one logit per Q head, any float dtype on Q's
// device -> fp32 [H_q] contiguous; undefined = none. Values are not checked
// (no host sync): a finite logit, or -inf for a head without sink.
static torch::Tensor sink_logits(const c10::optional<torch::Tensor>& sinks, const torch::Tensor& Q, int H_q) {
    if (!sinks.has_value() || !sinks->defined()) return {};
    TORCH_CHECK(sinks->is_floating_point() && sinks->device() == Q.device(),
                "sinks must be a floating-point tensor on the device of Q");
    TORCH_CHECK(sinks->dim() == 1 && sinks->size(0) == H_q,
                "sinks must be [num_heads_q] = [", H_q, "], got ", sinks->sizes());
    return sinks->to(torch::kFloat32).contiguous();
}

static float* data_or_null(const torch::Tensor& t) { return t.defined() ? t.data_ptr<float>() : nullptr; }

// Input checks shared by the varlen entry points: Q [total_q, H_q, d],
// K / V [total_k, H_kv, d] fp16 CUDA, cu_seqlens_q / k int32 on the same device.
static void check_varlen_inputs(const torch::Tensor& Q, const torch::Tensor& K, const torch::Tensor& V,
                                const torch::Tensor& cu_seqlens_q, const torch::Tensor& cu_seqlens_k) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Q, K, V must be on CUDA");
    TORCH_CHECK(Q.device() == K.device() && Q.device() == V.device(), "Q, K, V must be on the same device");
    TORCH_CHECK(Q.dtype() == torch::kFloat16 && K.dtype() == torch::kFloat16 &&
                V.dtype() == torch::kFloat16, "Q, K, V must be float16");
    TORCH_CHECK(Q.dim() == 3 && K.dim() == 3 && V.dim() == 3, "Q, K, V must be [total, H, d]");
    const int d = Q.size(2), H_q = Q.size(1), H_kv = K.size(1);
    check_head_dim(d);
    TORCH_CHECK(K.size(2) == d && V.sizes() == K.sizes(), "K, V shape mismatch");
    TORCH_CHECK(H_kv > 0, "num_heads_kv must be > 0");
    TORCH_CHECK(H_q % H_kv == 0, "H_q (", H_q, ") must be divisible by H_kv (", H_kv, ")");
    TORCH_CHECK(cu_seqlens_q.is_cuda() && cu_seqlens_k.is_cuda() &&
                cu_seqlens_q.device() == Q.device() && cu_seqlens_k.device() == Q.device(),
                "cu_seqlens_q/k must be on the same CUDA device as Q");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32 && cu_seqlens_k.dtype() == torch::kInt32,
                "cu_seqlens_q/k must be int32");
    TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_q.size(0) >= 2, "cu_seqlens must have at least 2 entries");
    // Token counts must fit the int32 cu_seqlens; heads go in gridDim.y
    TORCH_CHECK(Q.size(0) <= INT32_MAX && K.size(0) <= INT32_MAX, "total tokens must fit in int32");
    TORCH_CHECK(H_q <= 65535, "num_heads must be <= 65535 for varlen, got ", H_q);
    TORCH_CHECK(cu_seqlens_k.sizes() == cu_seqlens_q.sizes(), "cu_seqlens_k length mismatch");
}

// Validates the varlen boundaries on one host copy (a wrong cu_seqlens would
// otherwise read out of bounds or leave rows uncomputed) and returns the host
// copies {cu_seqlens_q, cu_seqlens_k}. Removing this sync (CUDA graphs,
// torch.compile) is docs/PLAN.md P6.5.
static std::pair<torch::Tensor, torch::Tensor> validated_cu_seqlens(
    const torch::Tensor& cu_seqlens_q, const torch::Tensor& cu_seqlens_k, int total_q, int total_k) {
    const int num_seqs = cu_seqlens_q.size(0) - 1;
    auto cu_cpu = torch::cat({cu_seqlens_q, cu_seqlens_k}).to(torch::kCPU);  // one host sync
    auto cu_q_cpu = cu_cpu.narrow(0, 0, num_seqs + 1);
    auto cu_k_cpu = cu_cpu.narrow(0, num_seqs + 1, num_seqs + 1);
    auto cu_q_acc = cu_q_cpu.accessor<int, 1>();
    auto cu_k_acc = cu_k_cpu.accessor<int, 1>();
    TORCH_CHECK(cu_q_acc[0] == 0 && cu_k_acc[0] == 0, "cu_seqlens must start at 0");
    TORCH_CHECK(cu_q_acc[num_seqs] == total_q, "cu_seqlens_q must end at total_q (", total_q,
                "), got ", cu_q_acc[num_seqs]);
    TORCH_CHECK(cu_k_acc[num_seqs] == total_k, "cu_seqlens_k must end at total_k (", total_k,
                "), got ", cu_k_acc[num_seqs]);
    for (int i = 0; i < num_seqs; i++) {
        TORCH_CHECK(cu_q_acc[i + 1] >= cu_q_acc[i] && cu_k_acc[i + 1] >= cu_k_acc[i],
                    "cu_seqlens must be non-decreasing (sequence ", i, ")");
    }
    return {cu_q_cpu, cu_k_cpu};
}

// Blocks of `rows` rows per sequence: cumulative counts [0, cdiv(len_0, rows),
// ...] on the device (int32, no host loop or host-to-device copy) for the
// kernels' varlen_seq_of_block, and the number of blocks to launch: exact from
// the validated host copy of the same cu_seqlens when there is one, else (no
// host sync) the upper bound cdiv(total_rows, rows) + sequences — blocks past
// the last sequence exit at once.
struct VarlenBlocks {
    torch::Tensor offsets;
    int total;
};
static VarlenBlocks varlen_blocks(const torch::Tensor& cu_seqlens, const torch::Tensor& cu_host,
                                  int rows, int total_rows) {
    const int n = cu_seqlens.size(0) - 1;
    const torch::Tensor blocks = (cu_seqlens.narrow(0, 1, n) - cu_seqlens.narrow(0, 0, n) + (rows - 1))
                                     .div(rows, "floor");
    int total = cdiv(total_rows, rows) + n;
    if (cu_host.defined()) {
        auto acc = cu_host.accessor<int, 1>();
        total = 0;
        for (int i = 0; i < n; i++) total += cdiv(acc[i + 1] - acc[i], rows);
    }
    return {torch::cat({torch::zeros({1}, cu_seqlens.options()), blocks.cumsum(0).to(torch::kInt32)}), total};
}

// Data pointers of the backward (strides differ between batched and varlen and
// are set by each entry point).
static void set_bwd_pointers(FlashAttnBwdParams& b, const torch::Tensor& dO, const torch::Tensor& Q,
                             const torch::Tensor& K, const torch::Tensor& V, const torch::Tensor& O,
                             const torch::Tensor& L, const torch::Tensor& D, torch::Tensor& dQ_fp32,
                             torch::Tensor& dK, torch::Tensor& dV) {
    b.Q  = reinterpret_cast<const half*>(Q.data_ptr<at::Half>());
    b.K  = reinterpret_cast<const half*>(K.data_ptr<at::Half>());
    b.V  = reinterpret_cast<const half*>(V.data_ptr<at::Half>());
    b.O  = reinterpret_cast<const half*>(O.data_ptr<at::Half>());
    b.L  = L.data_ptr<float>();
    b.D  = D.data_ptr<float>();
    b.dO = reinterpret_cast<const half*>(dO.data_ptr<at::Half>());
    b.dQ = dQ_fp32.data_ptr<float>();
    b.dK = reinterpret_cast<half*>(dK.data_ptr<at::Half>());
    b.dV = reinterpret_cast<half*>(dV.data_ptr<at::Half>());
}

// ---------------------------------------------------------------------------
// Forward — supports GQA and N_q != N_kv:
//   Q is [B, H_q, N_q, d], K/V are [B, H_kv, N_kv, d]; H_q % H_kv == 0.
// Causal masking is bottom-right aligned (query i sees keys j <= i + N_kv - N_q);
// window_left / window_right restrict each row further (KeyWindow, -1 = unbounded).
// sinks: optional logit per Q head in every row's softmax, without value (gpt-oss).
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> flash_attn_forward(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    float softmax_scale, bool is_causal, int64_t window_left, int64_t window_right, double softcap,
    c10::optional<torch::Tensor> sinks
) {
    Q = contiguous_aligned(Q); K = contiguous_aligned(K); V = contiguous_aligned(V);

    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Q, K, V must be on CUDA");
    TORCH_CHECK(Q.device() == K.device() && Q.device() == V.device(),
                "Q, K, V must be on the same device");
    TORCH_CHECK(Q.dtype() == torch::kFloat16 && K.dtype() == torch::kFloat16 &&
                V.dtype() == torch::kFloat16, "Q, K, V must be float16");
    TORCH_CHECK(Q.dim() == 4, "Q must be [B, H_q, N, d]");
    TORCH_CHECK(K.dim() == 4 && V.dim() == 4, "K,V must be 4D");

    TORCH_CHECK(Q.size(0) <= INT32_MAX && Q.size(2) <= INT32_MAX && K.size(2) <= INT32_MAX,
                "batch and sequence lengths must fit in int32");
    const int B = Q.size(0), H_q = Q.size(1), N = Q.size(2), d = Q.size(3);
    const int H_kv = K.size(1);
    const int Nk = K.size(2);

    check_head_dim(d);
    TORCH_CHECK(K.size(0) == B && K.size(3) == d, "K shape mismatch");
    TORCH_CHECK(V.sizes() == K.sizes(), "V shape must match K");
    TORCH_CHECK(H_kv > 0, "num_heads_kv must be > 0");
    TORCH_CHECK(H_q % H_kv == 0, "num_heads_q (", H_q, ") must be divisible by num_heads_kv (", H_kv, ")");
    const torch::Tensor sink = sink_logits(sinks, Q, H_q);

    auto opts_half = Q.options();
    auto opts_float = Q.options().dtype(torch::kFloat32);
    torch::Tensor O = torch::empty({B, H_q, N, d}, opts_half);
    torch::Tensor L = torch::empty({B, H_q, N}, opts_float);
    if (O.numel() == 0) return {O, L};  // nothing to compute (a 0-sized grid is invalid)
    if (Nk == 0) {                       // no keys: output 0, logsumexp -inf or the sink (as masked rows)
        O.zero_();
        if (sink.defined()) L.copy_(sink.view({1, H_q, 1}).expand_as(L));
        else                L.fill_(-INFINITY);
        return {O, L};
    }

    const at::cuda::CUDAGuard device_guard(Q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    FlashAttnParams p = {};
    p.Q = reinterpret_cast<const half*>(Q.data_ptr<at::Half>());
    p.K = reinterpret_cast<const half*>(K.data_ptr<at::Half>());
    p.V = reinterpret_cast<const half*>(V.data_ptr<at::Half>());
    p.O = reinterpret_cast<half*>(O.data_ptr<at::Half>());
    p.L = L.data_ptr<float>();

    p.batch_size = B; p.num_heads = H_q; p.num_heads_k = H_kv;
    p.seq_len_q = N; p.seq_len_k = Nk; p.head_dim = d;
    p.softmax_scale = softmax_scale;
    std::tie(p.window_left, p.window_right) = key_window(is_causal, window_left, window_right, N, Nk);
    p.softcap = logit_softcap(softcap);
    p.sinks = data_or_null(sink);

    p.q_batch_stride = Q.stride(0); p.q_head_stride = Q.stride(1); p.q_seq_stride = Q.stride(2);
    p.k_batch_stride = K.stride(0); p.k_head_stride = K.stride(1); p.k_seq_stride = K.stride(2);
    p.v_batch_stride = V.stride(0); p.v_head_stride = V.stride(1); p.v_seq_stride = V.stride(2);
    p.o_batch_stride = O.stride(0); p.o_head_stride = O.stride(1); p.o_seq_stride = O.stride(2);
    p.l_batch_stride = index_t(H_q) * N; p.l_head_stride = N;

    int sm = get_sm_version();
    if (sm >= 75) {
        flash_attn_fwd_turing(p, stream);
    } else if (sm >= 70) {
        flash_attn_fwd_volta(p, stream);
    } else if (sm >= 60) {
        flash_attn_fwd_pascal(p, stream);
    } else {
        TORCH_CHECK(false, "flash_attn_legacy requires SM >= 6.0. Got SM ", sm);
    }

    FLASH_ATTN_CHECK_CUDA(cudaGetLastError());
    return {O, L};
}

// ---------------------------------------------------------------------------
// Backward — GQA aware
// dQ has same shape as Q [B, H_q, N, d]
// dK, dV have same shape as K, V [B, H_kv, N, d]
// dsinks: fp32 [H_q] with sinks (undefined -> None without)
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> flash_attn_backward(
    torch::Tensor dO,
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor O, torch::Tensor L,
    float softmax_scale, bool is_causal, int64_t window_left, int64_t window_right, double softcap,
    c10::optional<torch::Tensor> sinks
) {
    dO = contiguous_aligned(dO);
    Q = contiguous_aligned(Q); K = contiguous_aligned(K); V = contiguous_aligned(V);
    O = contiguous_aligned(O);

    for (const auto* t : {&dO, &Q, &K, &V, &O}) {
        TORCH_CHECK(t->is_cuda() && t->dtype() == torch::kFloat16,
                    "dO, Q, K, V, O must be CUDA float16");
        TORCH_CHECK(t->device() == Q.device(), "dO, Q, K, V, O must be on the same device");
    }
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4 && V.dim() == 4, "Q, K, V must be 4D");
    TORCH_CHECK(dO.sizes() == Q.sizes() && O.sizes() == Q.sizes(), "dO and O shapes must match Q");

    TORCH_CHECK(Q.size(0) <= INT32_MAX && Q.size(2) <= INT32_MAX && K.size(2) <= INT32_MAX,
                "batch and sequence lengths must fit in int32");
    const int B = Q.size(0), H_q = Q.size(1), N = Q.size(2), d = Q.size(3);
    const int H_kv = K.size(1);
    check_head_dim(d);
    TORCH_CHECK(H_kv > 0, "num_heads_kv must be > 0");
    TORCH_CHECK(H_q % H_kv == 0, "H_q must be divisible by H_kv for GQA");
    const int Nk = K.size(2);
    TORCH_CHECK(K.size(0) == B && K.size(3) == d, "K shape mismatch");
    TORCH_CHECK(V.sizes() == K.sizes(), "V shape must match K");

    TORCH_CHECK(L.is_cuda() && L.dtype() == torch::kFloat32 && L.device() == Q.device(),
                "L must be CUDA float32 on the same device as Q");
    L = L.contiguous();
    TORCH_CHECK(L.dim() == 3 && L.size(0) == B && L.size(1) == H_q && L.size(2) == N,
                "L must be [B, H_q, N]");
    const torch::Tensor sink = sink_logits(sinks, Q, H_q);
    // fp32 gradient of the sinks (undefined without sinks), cast by the caller
    torch::Tensor dsink = sink.defined() ? torch::zeros_like(sink) : torch::Tensor();

    const at::cuda::CUDAGuard device_guard(Q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // dQ in FP32 (atomicAdd accumulator) → convert to FP16 at end
    if (Q.numel() == 0 || Nk == 0) {
        // nothing to propagate: dQ is 0 without keys, dK/dV are 0 without queries
        return {torch::zeros_like(Q), torch::zeros_like(K), torch::zeros_like(V), dsink};  // O = 0: dsink = 0
    }
    torch::Tensor dQ_fp32 = torch::zeros({B, H_q, N, d}, Q.options().dtype(torch::kFloat32));
    // dK, dV have KV head shape — each element written exactly once (no atomics)
    torch::Tensor dK = torch::empty({B, H_kv, Nk, d}, K.options());
    torch::Tensor dV = torch::empty({B, H_kv, Nk, d}, V.options());

    // Precompute D = rowsum(dO * O) → [B, H_q, N]
    const index_t l_batch_stride = index_t(H_q) * N;
    const index_t l_head_stride = N;
    torch::Tensor D_tensor = torch::empty({B, H_q, N}, Q.options().dtype(torch::kFloat32));

    TORCH_CHECK(dO.strides() == O.strides(), "dO and O must share strides (precompute_D)");
    precompute_D(
        reinterpret_cast<const half*>(dO.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(O.data_ptr<at::Half>()),
        D_tensor.data_ptr<float>(), L.data_ptr<float>(), data_or_null(sink), data_or_null(dsink),
        B, H_q, N, d,
        O.stride(0), O.stride(1), O.stride(2),
        l_batch_stride, l_head_stride, 1,
        stream);

    FlashAttnBwdParams bwd = {};
    set_bwd_pointers(bwd, dO, Q, K, V, O, L, D_tensor, dQ_fp32, dK, dV);
    bwd.batch_size = B; bwd.num_heads = H_q; bwd.num_heads_k = H_kv;
    bwd.seq_len_q = N; bwd.seq_len_k = Nk; bwd.head_dim = d;
    bwd.softmax_scale = softmax_scale;
    std::tie(bwd.window_left, bwd.window_right) = key_window(is_causal, window_left, window_right, N, Nk);
    bwd.softcap = logit_softcap(softcap);

    bwd.q_batch_stride = Q.stride(0); bwd.q_head_stride = Q.stride(1); bwd.q_seq_stride = Q.stride(2);
    bwd.k_batch_stride = K.stride(0); bwd.k_head_stride = K.stride(1); bwd.k_seq_stride = K.stride(2);
    bwd.v_batch_stride = V.stride(0); bwd.v_head_stride = V.stride(1); bwd.v_seq_stride = V.stride(2);
    bwd.o_batch_stride = O.stride(0); bwd.o_head_stride = O.stride(1); bwd.o_seq_stride = O.stride(2);
    bwd.l_batch_stride = l_batch_stride; bwd.l_head_stride = l_head_stride; bwd.l_seq_stride = 1;
    bwd.dq_batch_stride = dQ_fp32.stride(0);
    bwd.dq_head_stride  = dQ_fp32.stride(1);
    bwd.dq_seq_stride   = dQ_fp32.stride(2);
    bwd.dk_batch_stride = dK.stride(0);
    bwd.dk_head_stride  = dK.stride(1);
    bwd.dk_seq_stride   = dK.stride(2);
    bwd.dv_batch_stride = dV.stride(0);
    bwd.dv_head_stride  = dV.stride(1);
    bwd.dv_seq_stride   = dV.stride(2);

    flash_attn_bwd_launch(bwd, get_sm_version(), 0, stream);
    FLASH_ATTN_CHECK_CUDA(cudaGetLastError());

    torch::Tensor dQ = dQ_fp32.to(torch::kFloat16);
    return {dQ, dK, dV, dsink};
}

// ---------------------------------------------------------------------------
// Forward — Variable-length (packed sequences)
// Q [total_q, H_q, d], K [total_k, H_kv, d], V [total_k, H_kv, d]
// cu_seqlens_q [num_seqs+1], cu_seqlens_k [num_seqs+1]
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> flash_attn_forward_varlen(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor cu_seqlens_q, torch::Tensor cu_seqlens_k,
    int max_seqlen_q, int max_seqlen_k,
    float softmax_scale, bool is_causal, int64_t window_left, int64_t window_right, double softcap,
    c10::optional<torch::Tensor> sinks, bool validate
) {
    Q = contiguous_aligned(Q); K = contiguous_aligned(K); V = contiguous_aligned(V);
    check_varlen_inputs(Q, K, V, cu_seqlens_q, cu_seqlens_k);
    const int total_q = Q.size(0), H_q = Q.size(1), d = Q.size(2);
    const torch::Tensor sink = sink_logits(sinks, Q, H_q);
    const int total_k = K.size(0), H_kv = K.size(1);
    const int num_seqs = cu_seqlens_q.size(0) - 1;

    auto opts_half = Q.options();
    auto opts_float = Q.options().dtype(torch::kFloat32);
    torch::Tensor O = torch::empty({total_q, H_q, d}, opts_half);
    torch::Tensor L = torch::empty({total_q, H_q}, opts_float);

    const at::cuda::CUDAGuard device_guard(Q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // validate=false: the caller guarantees valid cu_seqlens (no host sync)
    const auto cu_host = validate ? validated_cu_seqlens(cu_seqlens_q, cu_seqlens_k, total_q, total_k)
                                  : std::pair<torch::Tensor, torch::Tensor>();
    if (total_q == 0 || H_q == 0) return {O, L};  // no query tokens: nothing to launch

    int sm = get_sm_version();
    // Br of the tile the launcher will select (same dispatch: they cannot disagree)
    const int Br = sm >= 75 ? flash_attn_fwd_turing_block_rows(d)
                 : sm >= 70 ? flash_attn_fwd_volta_block_rows(d) : flash_attn_fwd_pascal_block_rows(d);
    const VarlenBlocks q_blocks = varlen_blocks(cu_seqlens_q, cu_host.first, Br, total_q);
    if (q_blocks.total == 0) return {O, L};  // only empty sequences: nothing to launch
    const int total_q_blocks = q_blocks.total;
    const torch::Tensor& block_offsets_t = q_blocks.offsets;

    FlashAttnVarlenParams p = {};
    p.Q = reinterpret_cast<const half*>(Q.data_ptr<at::Half>());
    p.K = reinterpret_cast<const half*>(K.data_ptr<at::Half>());
    p.V = reinterpret_cast<const half*>(V.data_ptr<at::Half>());
    p.O = reinterpret_cast<half*>(O.data_ptr<at::Half>());
    p.L = L.data_ptr<float>();

    p.cu_seqlens_q = cu_seqlens_q.data_ptr<int>();
    p.cu_seqlens_k = cu_seqlens_k.data_ptr<int>();
    p.block_offsets = block_offsets_t.data_ptr<int>();

    p.num_seqs = num_seqs;
    p.num_heads = H_q;
    p.num_heads_k = H_kv;
    p.head_dim = d;
    p.softmax_scale = softmax_scale;
    std::tie(p.window_left, p.window_right) = key_window(is_causal, window_left, window_right, total_q, total_k);
    p.softcap = logit_softcap(softcap);
    p.sinks = data_or_null(sink);

    p.q_token_stride = Q.stride(0); p.q_head_stride = Q.stride(1);
    p.k_token_stride = K.stride(0); p.k_head_stride = K.stride(1);
    p.v_token_stride = V.stride(0); p.v_head_stride = V.stride(1);
    p.o_token_stride = O.stride(0); p.o_head_stride = O.stride(1);

    if (sm >= 75) {
        flash_attn_fwd_turing_varlen(p, total_q_blocks, stream);
    } else if (sm >= 70) {
        flash_attn_fwd_volta_varlen(p, total_q_blocks, stream);
    } else if (sm >= 60) {
        flash_attn_fwd_pascal_varlen(p, total_q_blocks, stream);
    } else {
        TORCH_CHECK(false, "flash_attn_legacy requires SM >= 6.0. Got SM ", sm);
    }

    FLASH_ATTN_CHECK_CUDA(cudaGetLastError());
    return {O, L};
}

// ---------------------------------------------------------------------------
// Backward — variable-length packed sequences, one launch (GQA aware).
// Q, O, dO are [total_q, H_q, d], K, V [total_k, H_kv, d], L [total_q, H_q].
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> flash_attn_backward_varlen(
    torch::Tensor dO,
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor O, torch::Tensor L,
    torch::Tensor cu_seqlens_q, torch::Tensor cu_seqlens_k,
    int max_seqlen_q, int max_seqlen_k,
    float softmax_scale, bool is_causal, int64_t window_left, int64_t window_right, double softcap,
    c10::optional<torch::Tensor> sinks, bool validate
) {
    dO = contiguous_aligned(dO);
    Q = contiguous_aligned(Q); K = contiguous_aligned(K); V = contiguous_aligned(V);
    O = contiguous_aligned(O);
    check_varlen_inputs(Q, K, V, cu_seqlens_q, cu_seqlens_k);
    TORCH_CHECK(dO.device() == Q.device() && O.device() == Q.device() &&
                dO.dtype() == torch::kFloat16 && O.dtype() == torch::kFloat16,
                "dO, O must be CUDA float16 on the same device as Q");
    TORCH_CHECK(dO.sizes() == Q.sizes() && O.sizes() == Q.sizes(), "dO and O shapes must match Q");
    const int total_q = Q.size(0), H_q = Q.size(1), d = Q.size(2);
    const int total_k = K.size(0), H_kv = K.size(1);
    const int num_seqs = cu_seqlens_q.size(0) - 1;
    TORCH_CHECK(L.is_cuda() && L.dtype() == torch::kFloat32 && L.device() == Q.device(),
                "L must be CUDA float32 on the same device as Q");
    L = L.contiguous();
    TORCH_CHECK(L.dim() == 2 && L.size(0) == total_q && L.size(1) == H_q, "L must be [total_q, H_q]");
    const torch::Tensor sink = sink_logits(sinks, Q, H_q);
    torch::Tensor dsink = sink.defined() ? torch::zeros_like(sink) : torch::Tensor();
    const auto cu_host = validate ? validated_cu_seqlens(cu_seqlens_q, cu_seqlens_k, total_q, total_k)
                                  : std::pair<torch::Tensor, torch::Tensor>();

    const at::cuda::CUDAGuard device_guard(Q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (total_q == 0 || total_k == 0 || H_q == 0) {  // O = 0: dsink = 0
        return {torch::zeros_like(Q), torch::zeros_like(K), torch::zeros_like(V), dsink};
    }

    const int sm = get_sm_version();
    const int Bc = flash_attn_bwd_block_cols(d, sm);  // same selection as the launch
    const VarlenBlocks k_blocks = varlen_blocks(cu_seqlens_k, cu_host.second, Bc, total_k);

    torch::Tensor dQ_fp32 = torch::zeros({total_q, H_q, d}, Q.options().dtype(torch::kFloat32));
    // Every key row belongs to one sequence and one key block: written exactly once
    torch::Tensor dK = torch::empty_like(K);
    torch::Tensor dV = torch::empty_like(V);
    torch::Tensor D_tensor = torch::empty({total_q, H_q}, Q.options().dtype(torch::kFloat32));
    TORCH_CHECK(dO.strides() == O.strides(), "dO and O must share strides (precompute_D)");
    precompute_D(
        reinterpret_cast<const half*>(dO.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(O.data_ptr<at::Half>()),
        D_tensor.data_ptr<float>(), L.data_ptr<float>(), data_or_null(sink), data_or_null(dsink),
        1, H_q, total_q, d,
        0, O.stride(1), O.stride(0),
        0, 1, H_q,
        stream);

    FlashAttnBwdParams bwd = {};
    set_bwd_pointers(bwd, dO, Q, K, V, O, L, D_tensor, dQ_fp32, dK, dV);
    // seq_len_q / seq_len_k stay 0: the varlen kernel takes each sequence's lengths
    // from cu_seqlens (max_seqlen_* are accepted for API symmetry, unused).
    bwd.batch_size = 1; bwd.num_heads = H_q; bwd.num_heads_k = H_kv; bwd.head_dim = d;
    bwd.softmax_scale = softmax_scale;
    std::tie(bwd.window_left, bwd.window_right) = key_window(is_causal, window_left, window_right, total_q, total_k);
    bwd.softcap = logit_softcap(softcap);
    // Packed layout: no batch stride, "seq" stride = token stride
    bwd.q_head_stride = Q.stride(1); bwd.q_seq_stride = Q.stride(0);
    bwd.k_head_stride = K.stride(1); bwd.k_seq_stride = K.stride(0);
    bwd.v_head_stride = V.stride(1); bwd.v_seq_stride = V.stride(0);
    bwd.o_head_stride = dO.stride(1); bwd.o_seq_stride = dO.stride(0);
    bwd.l_head_stride = 1; bwd.l_seq_stride = H_q;
    bwd.dq_head_stride = dQ_fp32.stride(1); bwd.dq_seq_stride = dQ_fp32.stride(0);
    bwd.dk_head_stride = dK.stride(1); bwd.dk_seq_stride = dK.stride(0);
    bwd.dv_head_stride = dV.stride(1); bwd.dv_seq_stride = dV.stride(0);
    bwd.cu_seqlens_q = cu_seqlens_q.data_ptr<int>();
    bwd.cu_seqlens_k = cu_seqlens_k.data_ptr<int>();
    bwd.key_block_offsets = k_blocks.offsets.data_ptr<int>();
    bwd.num_seqs = num_seqs;

    if (k_blocks.total > 0) flash_attn_bwd_launch(bwd, sm, k_blocks.total, stream);
    FLASH_ATTN_CHECK_CUDA(cudaGetLastError());
    return {dQ_fp32.to(torch::kFloat16), dK, dV, dsink};
}

// ---------------------------------------------------------------------------
// Decode with a KV cache (upstream flash_attn_with_kvcache subset): q [B, q_len,
// H_q, d], caches [B, max_len, H_kv, d] with any 16-byte aligned strides (no
// copy: HF's [B, H_kv, N, d] cache passes as a transposed view), cache_seqlens
// [B] int32 = end of the valid keys of each row (clamped to [0, max_len] in the
// kernel: never read past the cache), cache_leftpad [B] int32 = first valid key
// (left padding; clamped to [0, seqlen]). Paged: caches [num_blocks, block_size,
// H_kv, d] and block_table [B, max_blocks] int32 (max_len = max_blocks *
// block_size; not with cache_leftpad); the table entries of the keys
// < cache_seqlens must be in [0, num_blocks): not validated (no sync), as
// upstream. fp16 or bf16, read natively.
// Returns out like q (BSHD) and lse [B, H_q, q_len] (natural log; -inf for rows
// with no visible key nor sink).
// ---------------------------------------------------------------------------
// int32 [B] on q's device (cache_seqlens, cache_leftpad), made contiguous.
static torch::Tensor check_batch_int32(const torch::Tensor& t, const torch::Tensor& q, int B, const char* name) {
    TORCH_CHECK(t.is_cuda() && t.device() == q.device() && t.dtype() == torch::kInt32 && t.dim() == 1 &&
                t.size(0) == B, name, " must be int32 [B] on q's device");
    return t.contiguous();
}

static void check_decode_layout(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.stride(-1) == 1, name, ": the last dimension must be contiguous");
    for (int i = 0; i < t.dim() - 1; i++) {
        TORCH_CHECK(t.size(i) == 1 || t.stride(i) % 8 == 0, name,
                    ": strides must be multiples of 8 elements (16-byte rows)");
    }
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name, ": data must be 16-byte aligned");
}

std::vector<torch::Tensor> flash_attn_with_kvcache(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor cache_seqlens,
    float softmax_scale, bool is_causal, int64_t window_left, int64_t window_right, double softcap,
    c10::optional<torch::Tensor> sinks, int64_t num_splits, c10::optional<torch::Tensor> cache_leftpad,
    c10::optional<torch::Tensor> block_table
) {
    TORCH_CHECK(q.dim() == 4 && k_cache.dim() == 4 && v_cache.dim() == 4,
                "q must be [B, q_len, H_q, d], k_cache / v_cache [B, max_len, H_kv, d] "
                "(paged: [num_blocks, block_size, H_kv, d])");
    TORCH_CHECK(q.is_cuda() && k_cache.device() == q.device() && v_cache.device() == q.device(),
                "q, k_cache, v_cache must be on the same CUDA device");
    TORCH_CHECK(q.dtype() == torch::kFloat16 || q.dtype() == torch::kBFloat16, "q must be float16 or bfloat16");
    TORCH_CHECK(k_cache.dtype() == q.dtype() && v_cache.dtype() == q.dtype(), "k_cache / v_cache must have q's dtype");
    TORCH_CHECK(k_cache.sizes() == v_cache.sizes(), "v_cache shape must match k_cache");
    TORCH_CHECK(q.size(0) <= INT32_MAX && q.size(1) <= INT32_MAX && k_cache.size(1) <= INT32_MAX,
                "batch and sequence lengths must fit in int32");
    const int B = q.size(0), q_len = q.size(1), H_q = q.size(2), d = q.size(3);
    const int H_kv = k_cache.size(2);
    check_head_dim(d);
    const bool paged = block_table.has_value() && block_table->defined();
    torch::Tensor table;
    int max_len = k_cache.size(1);
    if (paged) {
        // [num_blocks, block_size, H_kv, d] caches, block_table [B, max_blocks] int32
        table = *block_table;
        TORCH_CHECK(table.is_cuda() && table.device() == q.device() && table.dtype() == torch::kInt32 &&
                    table.dim() == 2 && table.size(0) == B, "block_table must be int32 [B, max_blocks] on q's device");
        TORCH_CHECK(k_cache.size(0) > 0, "paged KV cache: no blocks");
        TORCH_CHECK(k_cache.size(1) > 0 && k_cache.size(1) % 16 == 0,
                    "paged KV cache: block_size must be a positive multiple of 16, got ", k_cache.size(1));
        TORCH_CHECK(!(cache_leftpad.has_value() && cache_leftpad->defined()),
                    "cache_leftpad is not supported with a paged KV cache (block_table)");
        TORCH_CHECK(int64_t(table.size(1)) * k_cache.size(1) <= INT32_MAX, "paged KV cache too long");
        table = table.contiguous();
        max_len = table.size(1) * k_cache.size(1);
    } else {
        TORCH_CHECK(k_cache.size(0) == B, "k_cache must be [B, max_len, H_kv, d]");
    }
    TORCH_CHECK(k_cache.size(3) == d, "k_cache head_dim must match q");
    TORCH_CHECK(H_kv > 0 && H_q % H_kv == 0, "num_heads_q (", H_q, ") must be divisible by num_heads_kv (", H_kv, ")");
    TORCH_CHECK_NOT_IMPLEMENTED(H_q / H_kv * q_len <= flash_attn_decode_max_rows(),
                "flash_attn_with_kvcache: (num_heads_q / num_heads_kv) * q_len = ", H_q / H_kv * q_len,
                " query rows per KV head; at most ", flash_attn_decode_max_rows(), " for now");
    q = contiguous_aligned(q);
    check_decode_layout(k_cache, "k_cache");
    check_decode_layout(v_cache, "v_cache");
    cache_seqlens = check_batch_int32(cache_seqlens, q, B, "cache_seqlens");
    torch::Tensor leftpad;
    if (cache_leftpad.has_value() && cache_leftpad->defined()) {
        leftpad = check_batch_int32(*cache_leftpad, q, B, "cache_leftpad");
    }
    const torch::Tensor sink = sink_logits(sinks, q, H_q);

    torch::Tensor out = torch::empty_like(q);
    torch::Tensor lse = torch::empty({B, H_q, q_len}, q.options().dtype(torch::kFloat32));
    if (out.numel() == 0) return {out, lse};

    const at::cuda::CUDAGuard device_guard(q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    get_sm_version();  // a kernel image for this GPU

    DecodeParams p = {};
    p.q = q.data_ptr(); p.k = k_cache.data_ptr(); p.v = v_cache.data_ptr(); p.out = out.data_ptr();
    p.lse = lse.data_ptr<float>();
    p.cache_seqlens = cache_seqlens.data_ptr<int>();
    p.cache_leftpad = leftpad.defined() ? leftpad.data_ptr<int>() : nullptr;
    p.block_table = paged ? table.data_ptr<int>() : nullptr;
    p.max_blocks = paged ? table.size(1) : 0;
    p.block_size = paged ? k_cache.size(1) : 0;
    p.block_shift = -1;
    if (paged && (p.block_size & (p.block_size - 1)) == 0) {
        p.block_shift = 0;
        while ((1 << p.block_shift) < p.block_size) p.block_shift++;
    }
    p.sinks = data_or_null(sink);
    p.q_batch_stride = q.stride(0); p.q_seq_stride = q.stride(1); p.q_head_stride = q.stride(2);
    p.k_batch_stride = k_cache.stride(0); p.k_seq_stride = k_cache.stride(1); p.k_head_stride = k_cache.stride(2);
    p.v_batch_stride = v_cache.stride(0); p.v_seq_stride = v_cache.stride(1); p.v_head_stride = v_cache.stride(2);
    p.o_batch_stride = out.stride(0); p.o_seq_stride = out.stride(1); p.o_head_stride = out.stride(2);
    p.batch_size = B; p.q_len = q_len; p.num_heads = H_q; p.num_heads_k = H_kv; p.head_dim = d;
    p.max_len = max_len;
    // Key splits: num_splits > 0 forced (capped), 0 = chosen from the grid
    TORCH_CHECK(num_splits >= 0, "num_splits must be >= 0 (0 = automatic)");
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    p.num_splits = num_splits > 0 ? int(std::min<int64_t>(num_splits, flash_attn_decode_max_splits()))
                                  : flash_attn_decode_num_splits(B * H_kv, sms, max_len);
    torch::Tensor o_part, lse_part;
    if (p.num_splits > 1) {
        o_part = torch::empty({p.num_splits, B, H_q, q_len, d}, q.options().dtype(torch::kFloat32));
        lse_part = torch::empty({p.num_splits, B, H_q, q_len}, q.options().dtype(torch::kFloat32));
        p.o_part = o_part.data_ptr<float>();
        p.lse_part = lse_part.data_ptr<float>();
    }
    p.softmax_scale = softmax_scale;
    p.softcap = logit_softcap(softcap);
    std::tie(p.window_left, p.window_right) = key_window(is_causal, window_left, window_right, q_len, max_len);

    flash_attn_decode(p, q.dtype() == torch::kBFloat16, stream);
    FLASH_ATTN_CHECK_CUDA(cudaGetLastError());
    return {out, lse};
}

}  // namespace flash_attn_legacy

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compiled_archs", &flash_attn_legacy::compiled_archs,
          "CUDA architectures this build contains, e.g. ['6.0', '7.0', '7.5+PTX']");
    m.def("forward", &flash_attn_legacy::flash_attn_forward,
          "Flash Attention v2 forward (Pascal/Volta, GQA/MQA)",
          py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("softmax_scale"), py::arg("is_causal") = false,
          py::arg("window_left") = -1, py::arg("window_right") = -1, py::arg("softcap") = 0.0, py::arg("sinks") = py::none());
    m.def("forward_varlen", &flash_attn_legacy::flash_attn_forward_varlen,
          "Flash Attention v2 forward — variable-length packed sequences",
          py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("cu_seqlens_q"), py::arg("cu_seqlens_k"),
          py::arg("max_seqlen_q"), py::arg("max_seqlen_k"),
          py::arg("softmax_scale"), py::arg("is_causal") = false,
          py::arg("window_left") = -1, py::arg("window_right") = -1, py::arg("softcap") = 0.0, py::arg("sinks") = py::none(), py::arg("validate") = true);
    m.def("backward_varlen", &flash_attn_legacy::flash_attn_backward_varlen,
          "Flash Attention v2 backward — variable-length packed sequences, one launch",
          py::arg("dO"), py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("O"), py::arg("L"),
          py::arg("cu_seqlens_q"), py::arg("cu_seqlens_k"),
          py::arg("max_seqlen_q"), py::arg("max_seqlen_k"),
          py::arg("softmax_scale"), py::arg("is_causal") = false,
          py::arg("window_left") = -1, py::arg("window_right") = -1, py::arg("softcap") = 0.0, py::arg("sinks") = py::none(), py::arg("validate") = true);
    m.def("with_kvcache", &flash_attn_legacy::flash_attn_with_kvcache,
          "Decode with a KV cache (flash_attn_with_kvcache subset)",
          py::arg("q"), py::arg("k_cache"), py::arg("v_cache"), py::arg("cache_seqlens"),
          py::arg("softmax_scale"), py::arg("is_causal") = false,
          py::arg("window_left") = -1, py::arg("window_right") = -1, py::arg("softcap") = 0.0,
          py::arg("sinks") = py::none(), py::arg("num_splits") = 0, py::arg("cache_leftpad") = py::none(),
          py::arg("block_table") = py::none());
    m.def("decode_max_rows", &flash_attn_legacy::flash_attn_decode_max_rows,
          "Query rows per KV head ((H_q / H_kv) * q_len) flash_attn_with_kvcache handles");
    m.def("decode_num_splits", &flash_attn_legacy::flash_attn_decode_num_splits,
          "Automatic key splits of flash_attn_with_kvcache for (B * H_kv, SMs, max_len)",
          py::arg("blocks"), py::arg("sms"), py::arg("max_len"));
    m.def("backward", &flash_attn_legacy::flash_attn_backward,
          "Flash Attention v2 backward (Pascal/Volta, GQA/MQA)",
          py::arg("dO"), py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("O"), py::arg("L"),
          py::arg("softmax_scale"), py::arg("is_causal") = false,
          py::arg("window_left") = -1, py::arg("window_right") = -1, py::arg("softcap") = 0.0, py::arg("sinks") = py::none());
}
