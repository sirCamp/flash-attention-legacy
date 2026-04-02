// ============================================================================
// Flash Attention Legacy — PyTorch C++ Extension Entry Point
// Runtime dispatch: Pascal vs Volta
// Supports MHA, GQA (grouped-query), and MQA (multi-query attention)
// ============================================================================

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "include/flash_attn_common.h"

namespace flash_attn_legacy {

// Forward declarations — batched
void flash_attn_fwd_volta(FlashAttnParams& params, cudaStream_t stream);
void flash_attn_fwd_pascal(FlashAttnParams& params, cudaStream_t stream);
// Forward declarations — variable-length
void flash_attn_fwd_volta_varlen(FlashAttnVarlenParams& params, int total_q_blocks, cudaStream_t stream);
void flash_attn_fwd_pascal_varlen(FlashAttnVarlenParams& params, int total_q_blocks, cudaStream_t stream);
void precompute_D(
    const half* dO, const half* O, float* D,
    int B, int H, int N, int d,
    int o_batch_stride, int o_head_stride, int o_seq_stride,
    int l_batch_stride, int l_head_stride,
    cudaStream_t stream);
void flash_attn_bwd_volta_launch(FlashAttnBwdParams& params, cudaStream_t stream);
void flash_attn_bwd_pascal_launch(FlashAttnBwdParams& params, cudaStream_t stream);

static int get_sm_version(int device_id = -1) {
    if (device_id < 0) FLASH_ATTN_CHECK_CUDA(cudaGetDevice(&device_id));
    int major, minor;
    FLASH_ATTN_CHECK_CUDA(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id));
    FLASH_ATTN_CHECK_CUDA(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id));
    return major * 10 + minor;
}

// ---------------------------------------------------------------------------
// Forward — supports GQA: Q is [B, H_q, N, d], K/V are [B, H_kv, N, d]
// H_q must be divisible by H_kv
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> flash_attn_forward(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    float softmax_scale, bool is_causal
) {
    Q = Q.contiguous(); K = K.contiguous(); V = V.contiguous();

    TORCH_CHECK(Q.is_cuda(), "Q must be on CUDA");
    TORCH_CHECK(Q.dtype() == torch::kFloat16, "Q must be float16");
    TORCH_CHECK(Q.dim() == 4, "Q must be [B, H_q, N, d]");
    TORCH_CHECK(K.dim() == 4 && V.dim() == 4, "K,V must be 4D");

    const int B = Q.size(0), H_q = Q.size(1), N = Q.size(2), d = Q.size(3);
    const int H_kv = K.size(1);

    TORCH_CHECK(d == 64 || d == 128, "head_dim must be 64 or 128, got ", d);
    TORCH_CHECK(K.size(0) == B && K.size(2) == N && K.size(3) == d, "K shape mismatch");
    TORCH_CHECK(V.size(0) == B && V.size(1) == H_kv && V.size(2) == N && V.size(3) == d, "V shape mismatch");
    TORCH_CHECK(H_q % H_kv == 0, "num_heads_q (", H_q, ") must be divisible by num_heads_kv (", H_kv, ")");

    auto opts_half = Q.options();
    auto opts_float = Q.options().dtype(torch::kFloat32);
    torch::Tensor O = torch::empty({B, H_q, N, d}, opts_half);
    torch::Tensor L = torch::empty({B, H_q, N}, opts_float);

    const at::cuda::CUDAGuard device_guard(Q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    FlashAttnParams p = {};
    p.Q = reinterpret_cast<const half*>(Q.data_ptr<at::Half>());
    p.K = reinterpret_cast<const half*>(K.data_ptr<at::Half>());
    p.V = reinterpret_cast<const half*>(V.data_ptr<at::Half>());
    p.O = reinterpret_cast<half*>(O.data_ptr<at::Half>());
    p.L = L.data_ptr<float>();

    p.batch_size = B; p.num_heads = H_q; p.num_heads_k = H_kv;
    p.seq_len = N; p.head_dim = d;
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
    dO = dO.contiguous();
    Q = Q.contiguous(); K = K.contiguous(); V = V.contiguous();
    O = O.contiguous();

    TORCH_CHECK(Q.is_cuda() && Q.dtype() == torch::kFloat16, "Q must be CUDA float16");
    TORCH_CHECK(dO.sizes() == Q.sizes(), "dO shape must match Q");

    const int B = Q.size(0), H_q = Q.size(1), N = Q.size(2), d = Q.size(3);
    const int H_kv = K.size(1);
    TORCH_CHECK(H_q % H_kv == 0, "H_q must be divisible by H_kv for GQA");

    const at::cuda::CUDAGuard device_guard(Q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // dQ in FP32 (atomicAdd accumulator) → convert to FP16 at end
    torch::Tensor dQ_fp32 = torch::zeros({B, H_q, N, d}, Q.options().dtype(torch::kFloat32));
    // dK, dV have KV head shape — each element written exactly once (no atomics)
    torch::Tensor dK = torch::empty({B, H_kv, N, d}, K.options());
    torch::Tensor dV = torch::empty({B, H_kv, N, d}, V.options());

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
    bwd.seq_len = N; bwd.head_dim = d;
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
    Q = Q.contiguous(); K = K.contiguous(); V = V.contiguous();

    TORCH_CHECK(Q.is_cuda(), "Q must be on CUDA");
    TORCH_CHECK(Q.dtype() == torch::kFloat16, "Q must be float16");
    TORCH_CHECK(Q.dim() == 3, "Q must be [total_q, H_q, d]");
    TORCH_CHECK(K.dim() == 3 && V.dim() == 3, "K,V must be 3D");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must be int32");
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must be int32");

    const int total_q = Q.size(0), H_q = Q.size(1), d = Q.size(2);
    const int total_k = K.size(0), H_kv = K.size(1);
    const int num_seqs = cu_seqlens_q.size(0) - 1;

    TORCH_CHECK(d == 64 || d == 128, "head_dim must be 64 or 128, got ", d);
    TORCH_CHECK(K.size(2) == d && V.size(2) == d, "K,V head_dim mismatch");
    TORCH_CHECK(V.size(1) == H_kv, "V num_heads mismatch");
    TORCH_CHECK(H_q % H_kv == 0, "H_q (", H_q, ") must be divisible by H_kv (", H_kv, ")");
    TORCH_CHECK(cu_seqlens_k.size(0) == num_seqs + 1, "cu_seqlens_k length mismatch");

    auto opts_half = Q.options();
    auto opts_float = Q.options().dtype(torch::kFloat32);
    torch::Tensor O = torch::empty({total_q, H_q, d}, opts_half);
    torch::Tensor L = torch::empty({total_q, H_q}, opts_float);

    const at::cuda::CUDAGuard device_guard(Q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Compute block_offsets on CPU: cumulative number of Q blocks per sequence
    auto cu_q_cpu = cu_seqlens_q.to(torch::kCPU);
    auto cu_q_acc = cu_q_cpu.accessor<int, 1>();

    int sm = get_sm_version();
    // Br depends on arch and head_dim
    int Br;
    if (sm >= 70) {
        Br = 128;  // Volta: Br=128 for both d=64 and d=128
    } else {
        Br = (d == 64) ? 128 : 64;  // Pascal: Br=128 for d=64, Br=64 for d=128
    }

    std::vector<int> block_offsets_vec(num_seqs + 1);
    block_offsets_vec[0] = 0;
    for (int i = 0; i < num_seqs; i++) {
        int seq_len = cu_q_acc[i + 1] - cu_q_acc[i];
        block_offsets_vec[i + 1] = block_offsets_vec[i] + cdiv(seq_len, Br);
    }
    int total_q_blocks = block_offsets_vec[num_seqs];

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
