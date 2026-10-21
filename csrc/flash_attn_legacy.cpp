// ============================================================================
// Flash Attention Legacy — PyTorch C++ Extension Entry Point
// Runtime dispatch: Pascal vs Volta
// Supports MHA, GQA (grouped-query), and MQA (multi-query attention)
// ============================================================================

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <string>
#include "include/flash_attn_common.h"

namespace flash_attn_legacy {

// Forward declarations — batched
void flash_attn_fwd_volta(FlashAttnParams& params, cudaStream_t stream);
void flash_attn_fwd_pascal(FlashAttnParams& params, cudaStream_t stream);
// Forward declarations — variable-length
void flash_attn_fwd_volta_varlen(FlashAttnVarlenParams& params, int total_q_blocks, cudaStream_t stream);
void flash_attn_fwd_pascal_varlen(FlashAttnVarlenParams& params, int total_q_blocks, cudaStream_t stream);
int flash_attn_fwd_volta_block_rows(int head_dim);
int flash_attn_fwd_pascal_block_rows(int head_dim);
void precompute_D(
    const half* dO, const half* O, float* D,
    int B, int H, int N, int d,
    int o_batch_stride, int o_head_stride, int o_seq_stride,
    int l_batch_stride, int l_head_stride,
    cudaStream_t stream);
void flash_attn_bwd_volta_launch(FlashAttnBwdParams& params, cudaStream_t stream);
void flash_attn_bwd_pascal_launch(FlashAttnBwdParams& params, cudaStream_t stream);

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

// A GPU can run SASS of the same major and lower-or-equal minor, or PTX of a
// lower-or-equal arch (JIT). Fail with a clear message instead of CUDA's
// "no kernel image is available for execution on the device".
static void check_compiled_for(int sm) {
    for (int a : kCompiledSass) {
        if (a / 10 == sm / 10 && a <= sm) return;
    }
    if (kCompiledPtx && kCompiledPtx <= sm) return;
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
    check_compiled_for(sm);
    if (const char* force = std::getenv("FLASH_ATTN_LEGACY_FORCE_ARCH")) {
        const std::string f(force);
        if (f == "pascal") return 60;  // CUDA-core kernels run on any SM >= 6.0
        if (f == "volta") {
            TORCH_CHECK(sm >= 70, "FLASH_ATTN_LEGACY_FORCE_ARCH=volta needs tensor cores "
                        "(SM >= 7.0), this GPU is SM ", sm);
            return 70;
        }
        TORCH_CHECK(f.empty(), "FLASH_ATTN_LEGACY_FORCE_ARCH must be 'pascal' or 'volta', got '", f, "'");
    }
    return sm;
}

// ---------------------------------------------------------------------------
// Forward — supports GQA and N_q != N_kv:
//   Q is [B, H_q, N_q, d], K/V are [B, H_kv, N_kv, d]; H_q % H_kv == 0.
// Causal masking is bottom-right aligned (query i sees keys j <= i + N_kv - N_q).
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> flash_attn_forward(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    float softmax_scale, bool is_causal
) {
    Q = contiguous_aligned(Q); K = contiguous_aligned(K); V = contiguous_aligned(V);

    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Q, K, V must be on CUDA");
    TORCH_CHECK(Q.device() == K.device() && Q.device() == V.device(),
                "Q, K, V must be on the same device");
    TORCH_CHECK(Q.dtype() == torch::kFloat16 && K.dtype() == torch::kFloat16 &&
                V.dtype() == torch::kFloat16, "Q, K, V must be float16");
    TORCH_CHECK(Q.dim() == 4, "Q must be [B, H_q, N, d]");
    TORCH_CHECK(K.dim() == 4 && V.dim() == 4, "K,V must be 4D");

    const int B = Q.size(0), H_q = Q.size(1), N = Q.size(2), d = Q.size(3);
    const int H_kv = K.size(1);
    const int Nk = K.size(2);

    TORCH_CHECK(d == 32 || d == 64 || d == 96 || d == 128 || d == 256,
                "head_dim must be 32, 64, 96, 128, or 256, got ", d);
    TORCH_CHECK(K.size(0) == B && K.size(3) == d, "K shape mismatch");
    TORCH_CHECK(V.sizes() == K.sizes(), "V shape must match K");
    TORCH_CHECK(H_kv > 0, "num_heads_kv must be > 0");
    TORCH_CHECK(H_q % H_kv == 0, "num_heads_q (", H_q, ") must be divisible by num_heads_kv (", H_kv, ")");

    auto opts_half = Q.options();
    auto opts_float = Q.options().dtype(torch::kFloat32);
    torch::Tensor O = torch::empty({B, H_q, N, d}, opts_half);
    torch::Tensor L = torch::empty({B, H_q, N}, opts_float);
    if (O.numel() == 0) return {O, L};  // nothing to compute (a 0-sized grid is invalid)
    if (Nk == 0) {                       // no keys: output 0, logsumexp -inf (as masked rows)
        O.zero_();
        L.fill_(-INFINITY);
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
    p.softmax_scale = softmax_scale; p.is_causal = is_causal;

    p.q_batch_stride = Q.stride(0); p.q_head_stride = Q.stride(1); p.q_seq_stride = Q.stride(2);
    p.k_batch_stride = K.stride(0); p.k_head_stride = K.stride(1); p.k_seq_stride = K.stride(2);
    p.v_batch_stride = V.stride(0); p.v_head_stride = V.stride(1); p.v_seq_stride = V.stride(2);
    p.o_batch_stride = O.stride(0); p.o_head_stride = O.stride(1); p.o_seq_stride = O.stride(2);
    p.l_batch_stride = H_q * N; p.l_head_stride = N;

    int sm = get_sm_version();
    if (sm >= 70) {
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
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> flash_attn_backward(
    torch::Tensor dO,
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor O, torch::Tensor L,
    float softmax_scale, bool is_causal
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

    const int B = Q.size(0), H_q = Q.size(1), N = Q.size(2), d = Q.size(3);
    const int H_kv = K.size(1);
    TORCH_CHECK(d == 32 || d == 64 || d == 96 || d == 128 || d == 256,
                "head_dim must be 32, 64, 96, 128, or 256, got ", d);
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

    const at::cuda::CUDAGuard device_guard(Q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // dQ in FP32 (atomicAdd accumulator) → convert to FP16 at end
    if (Q.numel() == 0 || Nk == 0) {
        // nothing to propagate: dQ is 0 without keys, dK/dV are 0 without queries
        return {torch::zeros_like(Q), torch::zeros_like(K), torch::zeros_like(V)};
    }
    torch::Tensor dQ_fp32 = torch::zeros({B, H_q, N, d}, Q.options().dtype(torch::kFloat32));
    // dK, dV have KV head shape — each element written exactly once (no atomics)
    torch::Tensor dK = torch::empty({B, H_kv, Nk, d}, K.options());
    torch::Tensor dV = torch::empty({B, H_kv, Nk, d}, V.options());

    // Precompute D = rowsum(dO * O) → [B, H_q, N]
    const int l_batch_stride = H_q * N;
    const int l_head_stride = N;
    torch::Tensor D_tensor = torch::empty({B, H_q, N}, Q.options().dtype(torch::kFloat32));

    precompute_D(
        reinterpret_cast<const half*>(dO.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(O.data_ptr<at::Half>()),
        D_tensor.data_ptr<float>(),
        B, H_q, N, d,
        O.stride(0), O.stride(1), O.stride(2),
        l_batch_stride, l_head_stride,
        stream);

    FlashAttnBwdParams bwd = {};
    bwd.Q  = reinterpret_cast<const half*>(Q.data_ptr<at::Half>());
    bwd.K  = reinterpret_cast<const half*>(K.data_ptr<at::Half>());
    bwd.V  = reinterpret_cast<const half*>(V.data_ptr<at::Half>());
    bwd.O  = reinterpret_cast<const half*>(O.data_ptr<at::Half>());
    bwd.L  = L.data_ptr<float>();
    bwd.D  = D_tensor.data_ptr<float>();
    bwd.dO = reinterpret_cast<const half*>(dO.data_ptr<at::Half>());
    bwd.dQ = dQ_fp32.data_ptr<float>();
    bwd.dK = reinterpret_cast<half*>(dK.data_ptr<at::Half>());
    bwd.dV = reinterpret_cast<half*>(dV.data_ptr<at::Half>());

    bwd.batch_size = B; bwd.num_heads = H_q; bwd.num_heads_k = H_kv;
    bwd.seq_len_q = N; bwd.seq_len_k = Nk; bwd.head_dim = d;
    bwd.softmax_scale = softmax_scale; bwd.is_causal = is_causal;

    bwd.q_batch_stride = Q.stride(0); bwd.q_head_stride = Q.stride(1); bwd.q_seq_stride = Q.stride(2);
    bwd.k_batch_stride = K.stride(0); bwd.k_head_stride = K.stride(1); bwd.k_seq_stride = K.stride(2);
    bwd.v_batch_stride = V.stride(0); bwd.v_head_stride = V.stride(1); bwd.v_seq_stride = V.stride(2);
    bwd.o_batch_stride = O.stride(0); bwd.o_head_stride = O.stride(1); bwd.o_seq_stride = O.stride(2);
    bwd.l_batch_stride = l_batch_stride; bwd.l_head_stride = l_head_stride;
    bwd.dq_batch_stride = dQ_fp32.stride(0);
    bwd.dq_head_stride  = dQ_fp32.stride(1);
    bwd.dq_seq_stride   = dQ_fp32.stride(2);
    bwd.dk_batch_stride = dK.stride(0);
    bwd.dk_head_stride  = dK.stride(1);
    bwd.dk_seq_stride   = dK.stride(2);
    bwd.dv_batch_stride = dV.stride(0);
    bwd.dv_head_stride  = dV.stride(1);
    bwd.dv_seq_stride   = dV.stride(2);

    int sm = get_sm_version();
    if (sm >= 70) {
        flash_attn_bwd_volta_launch(bwd, stream);
    } else {
        flash_attn_bwd_pascal_launch(bwd, stream);
    }
    FLASH_ATTN_CHECK_CUDA(cudaGetLastError());

    torch::Tensor dQ = dQ_fp32.to(torch::kFloat16);
    return {dQ, dK, dV};
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
    float softmax_scale, bool is_causal
) {
    Q = contiguous_aligned(Q); K = contiguous_aligned(K); V = contiguous_aligned(V);

    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Q, K, V must be on CUDA");
    TORCH_CHECK(cu_seqlens_q.is_cuda() && cu_seqlens_k.is_cuda() &&
                cu_seqlens_q.device() == Q.device() && cu_seqlens_k.device() == Q.device(),
                "cu_seqlens_q/k must be on the same CUDA device as Q");
    TORCH_CHECK(Q.device() == K.device() && Q.device() == V.device(),
                "Q, K, V must be on the same device");
    TORCH_CHECK(Q.dtype() == torch::kFloat16 && K.dtype() == torch::kFloat16 &&
                V.dtype() == torch::kFloat16, "Q, K, V must be float16");
    TORCH_CHECK(Q.dim() == 3, "Q must be [total_q, H_q, d]");
    TORCH_CHECK(K.dim() == 3 && V.dim() == 3, "K,V must be 3D");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must be int32");
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must be int32");

    const int total_q = Q.size(0), H_q = Q.size(1), d = Q.size(2);
    const int total_k = K.size(0), H_kv = K.size(1);
    const int num_seqs = cu_seqlens_q.size(0) - 1;

    TORCH_CHECK(d == 32 || d == 64 || d == 96 || d == 128 || d == 256,
                "head_dim must be 32, 64, 96, 128, or 256, got ", d);
    TORCH_CHECK(K.size(2) == d && V.size(2) == d, "K,V head_dim mismatch");
    TORCH_CHECK(V.size(1) == H_kv, "V num_heads mismatch");
    TORCH_CHECK(H_kv > 0, "num_heads_kv must be > 0");
    TORCH_CHECK(H_q % H_kv == 0, "H_q (", H_q, ") must be divisible by H_kv (", H_kv, ")");
    TORCH_CHECK(cu_seqlens_k.size(0) == num_seqs + 1, "cu_seqlens_k length mismatch");

    auto opts_half = Q.options();
    auto opts_float = Q.options().dtype(torch::kFloat32);
    torch::Tensor O = torch::empty({total_q, H_q, d}, opts_half);
    torch::Tensor L = torch::empty({total_q, H_q}, opts_float);

    const at::cuda::CUDAGuard device_guard(Q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Compute block_offsets on CPU: cumulative number of Q blocks per sequence.
    // The CPU copies are also used to validate the boundaries (a wrong cu_seqlens
    // would otherwise read out of bounds or leave rows uncomputed).
    TORCH_CHECK(num_seqs >= 1, "cu_seqlens must have at least 2 entries");
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

    int sm = get_sm_version();
    // Br of the tile the launcher will select (same dispatch: they cannot disagree)
    const int Br = (sm >= 70) ? flash_attn_fwd_volta_block_rows(d) : flash_attn_fwd_pascal_block_rows(d);

    std::vector<int> block_offsets_vec(num_seqs + 1);
    block_offsets_vec[0] = 0;
    for (int i = 0; i < num_seqs; i++) {
        int seq_len = cu_q_acc[i + 1] - cu_q_acc[i];
        block_offsets_vec[i + 1] = block_offsets_vec[i] + cdiv(seq_len, Br);
    }
    int total_q_blocks = block_offsets_vec[num_seqs];
    if (total_q_blocks == 0 || H_q == 0) return {O, L};  // no query tokens: nothing to launch

    // Copy block_offsets to GPU
    torch::Tensor block_offsets_t = torch::from_blob(
        block_offsets_vec.data(), {num_seqs + 1}, torch::kInt32
    ).to(Q.device());

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
    p.is_causal = is_causal;

    p.q_token_stride = Q.stride(0); p.q_head_stride = Q.stride(1);
    p.k_token_stride = K.stride(0); p.k_head_stride = K.stride(1);
    p.v_token_stride = V.stride(0); p.v_head_stride = V.stride(1);
    p.o_token_stride = O.stride(0); p.o_head_stride = O.stride(1);

    if (sm >= 70) {
        flash_attn_fwd_volta_varlen(p, total_q_blocks, stream);
    } else if (sm >= 60) {
        flash_attn_fwd_pascal_varlen(p, total_q_blocks, stream);
    } else {
        TORCH_CHECK(false, "flash_attn_legacy requires SM >= 6.0. Got SM ", sm);
    }

    FLASH_ATTN_CHECK_CUDA(cudaGetLastError());
    return {O, L};
}

}  // namespace flash_attn_legacy

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compiled_archs", &flash_attn_legacy::compiled_archs,
          "CUDA architectures this build contains, e.g. ['6.0', '7.0', '7.5+PTX']");
    m.def("forward", &flash_attn_legacy::flash_attn_forward,
          "Flash Attention v2 forward (Pascal/Volta, GQA/MQA)",
          py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("softmax_scale"), py::arg("is_causal") = false);
    m.def("forward_varlen", &flash_attn_legacy::flash_attn_forward_varlen,
          "Flash Attention v2 forward — variable-length packed sequences",
          py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("cu_seqlens_q"), py::arg("cu_seqlens_k"),
          py::arg("max_seqlen_q"), py::arg("max_seqlen_k"),
          py::arg("softmax_scale"), py::arg("is_causal") = false);
    m.def("backward", &flash_attn_legacy::flash_attn_backward,
          "Flash Attention v2 backward (Pascal/Volta, GQA/MQA)",
          py::arg("dO"), py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("O"), py::arg("L"),
          py::arg("softmax_scale"), py::arg("is_causal") = false);
}
