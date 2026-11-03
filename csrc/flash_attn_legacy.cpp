// ============================================================================
// Flash Attention Legacy — PyTorch C++ Extension Entry Point
// Runtime dispatch: Pascal vs Volta
// Supports MHA, GQA (grouped-query), and MQA (multi-query attention)
// ============================================================================

#include <torch/extension.h>
#include <torch/library.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <mutex>
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
void flash_attn_split_combine(const SplitCombineParams& params, bool bf16, cudaStream_t stream);
void flash_attn_kvcache_append_launch(const KvAppendParams& params, cudaStream_t stream);
int flash_attn_decode_max_rows();
int flash_attn_num_splits(int blocks, int sms, int resident, int max_len, int rows);
int flash_attn_max_splits();
LinearPlan linear_pascal_plan(int M, int N, int K);
void linear_pascal_launch(const LinearParams& p, const LinearPlan& plan, cudaStream_t stream);
int causal_conv1d_runs(int seqlen);
void causal_conv1d_launch(const ConvParams& p, bool vec, bool backward, cudaStream_t stream);
void causal_conv1d_update_launch(const ConvParams& p, cudaStream_t stream);
void gdn_scan_launch(const GdnScanParams& p, int vs, bool backward, cudaStream_t stream);
void gdn_prep_launch(const GdnPrepParams& p, cudaStream_t stream);

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

// Compute capability of a device, e.g. 60 (no FLASH_ATTN_LEGACY_FORCE_ARCH).
static int device_sm(int device_id) {
    int major, minor;
    FLASH_ATTN_CHECK_CUDA(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id));
    FLASH_ATTN_CHECK_CUDA(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id));
    return major * 10 + minor;
}

static int get_sm_version(int device_id = -1) {
    if (device_id < 0) FLASH_ATTN_CHECK_CUDA(cudaGetDevice(&device_id));
    const int sm = device_sm(device_id);
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

// The forward of the kernel family that runs on `sm` (get_sm_version), and the
// query rows of the tile its launcher selects (same dispatch: they cannot disagree).
static void run_fwd(FlashAttnParams& p, int sm, cudaStream_t stream) {
    TORCH_CHECK(sm >= 60, "flash_attn_legacy requires SM >= 6.0. Got SM ", sm);
    if (sm >= 75)      flash_attn_fwd_turing(p, stream);
    else if (sm >= 70) flash_attn_fwd_volta(p, stream);
    else               flash_attn_fwd_pascal(p, stream);
}

static int fwd_block_rows(int sm, int d) {
    return sm >= 75 ? flash_attn_fwd_turing_block_rows(d)
         : sm >= 70 ? flash_attn_fwd_volta_block_rows(d) : flash_attn_fwd_pascal_block_rows(d);
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

// softmax_scale multiplier on the device (fp32 [1]) or undefined: the power-of-2
// scales of bf16 / fp32 inputs converted to fp16 without a host sync.
static torch::Tensor device_score_scale(const c10::optional<torch::Tensor>& t, const torch::Tensor& Q) {
    if (!t.has_value() || !t->defined()) return {};
    TORCH_CHECK(t->is_cuda() && t->device() == Q.device() && t->dtype() == torch::kFloat32 && t->numel() == 1,
                "score_scale must be one fp32 value on the device of Q");
    return t->contiguous();
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

// fp32 gradient of the sinks, zeroed for the kernels' atomics (cast by the caller);
// [0] without sinks: the ops never return undefined tensors.
static torch::Tensor sink_grad(const torch::Tensor& sink, const torch::Tensor& Q) {
    return sink.defined() ? torch::zeros_like(sink) : torch::empty({0}, Q.options().dtype(torch::kFloat32));
}

static float* data_or_null(const torch::Tensor& t) {
    return t.defined() && t.numel() > 0 ? t.data_ptr<float>() : nullptr;
}

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

// GQA backward: when the blocks of B * H_kv heads and their key blocks would leave
// SMs idle (one block loops over all H_q / H_kv query heads of its KV head), the
// group is split over S blocks, the smallest divisor of it giving 2 blocks per SM.
// Each slice writes its own fp16 dK / dV head (h * S + s), summed in fp32 after the
// kernel (upstream FA2's dk_expanded). 1 without GQA or with enough blocks.
// FLASH_ATTN_LEGACY_BWD_GROUP_SPLITS=n (tests): the smallest divisor >= min(n, group).
static int bwd_group_splits(int H_q, int H_kv, int64_t blocks) {
    const int rep = H_q / H_kv;
    if (const char* env = std::getenv("FLASH_ATTN_LEGACY_BWD_GROUP_SPLITS")) {
        const int n = std::atoi(env);
        TORCH_CHECK(n >= 1, "FLASH_ATTN_LEGACY_BWD_GROUP_SPLITS must be an integer >= 1");
        int S = std::min(n, rep);
        while (rep % S != 0) S++;
        return S;
    }
    const int64_t want = 2 * int64_t(at::cuda::getCurrentDeviceProperties()->multiProcessorCount);
    for (int S = 1; S < rep; S++) {
        if (rep % S == 0 && blocks * S >= want) return S;
    }
    return rep;
}

// dK / dV of K's layout with heads [.., H_kv * S, ..] (dim 1) -> the sum of the S slices.
static torch::Tensor sum_group_slices(const torch::Tensor& part, int S) {
    if (S == 1) return part;
    std::vector<int64_t> sizes = part.sizes().vec();
    sizes[1] /= S;
    sizes.insert(sizes.begin() + 2, S);
    return part.view(sizes).sum(2);  // fp16 sum: accumulated in fp32, rounded once
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
std::tuple<torch::Tensor, torch::Tensor> flash_attn_forward(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    double softmax_scale, bool is_causal, int64_t window_left, int64_t window_right, double softcap,
    c10::optional<torch::Tensor> sinks, c10::optional<torch::Tensor> score_scale
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
    const torch::Tensor scale_dev = device_score_scale(score_scale, Q);
    p.score_scale = data_or_null(scale_dev);

    p.q_batch_stride = Q.stride(0); p.q_head_stride = Q.stride(1); p.q_seq_stride = Q.stride(2);
    p.k_batch_stride = K.stride(0); p.k_head_stride = K.stride(1); p.k_seq_stride = K.stride(2);
    p.v_batch_stride = V.stride(0); p.v_head_stride = V.stride(1); p.v_seq_stride = V.stride(2);
    p.o_batch_stride = O.stride(0); p.o_head_stride = O.stride(1); p.o_seq_stride = O.stride(2);
    p.l_batch_stride = index_t(H_q) * N; p.l_head_stride = N;

    run_fwd(p, get_sm_version(), stream);
    FLASH_ATTN_CHECK_CUDA(cudaGetLastError());
    return {O, L};
}

// ---------------------------------------------------------------------------
// Backward — GQA aware
// dQ has same shape as Q [B, H_q, N, d]
// dK, dV have same shape as K, V [B, H_kv, N, d]
// dsinks: fp32 [H_q] with sinks ([0] without)
// ---------------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> flash_attn_backward(
    torch::Tensor dO,
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor O, torch::Tensor L,
    double softmax_scale, bool is_causal, int64_t window_left, int64_t window_right, double softcap,
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
    torch::Tensor dsink = sink_grad(sink, Q);

    const at::cuda::CUDAGuard device_guard(Q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // dQ in FP32 (atomicAdd accumulator) → convert to FP16 at end
    if (Q.numel() == 0 || Nk == 0) {
        // nothing to propagate: dQ is 0 without keys, dK/dV are 0 without queries
        return {torch::zeros_like(Q), torch::zeros_like(K), torch::zeros_like(V), dsink};  // O = 0: dsink = 0
    }
    torch::Tensor dQ_fp32 = torch::zeros({B, H_q, N, d}, Q.options().dtype(torch::kFloat32));
    // dK, dV: [B, H_kv * S, Nk, d], each element written exactly once (no atomics)
    const int sm = get_sm_version();
    const int S = bwd_group_splits(H_q, H_kv, int64_t(cdiv(Nk, flash_attn_bwd_block_cols(d, sm))) * B * H_kv);
    torch::Tensor dK = torch::empty({B, H_kv * S, Nk, d}, K.options());
    torch::Tensor dV = torch::empty({B, H_kv * S, Nk, d}, V.options());

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
    bwd.batch_size = B; bwd.num_heads = H_q; bwd.num_heads_k = H_kv; bwd.kv_group_splits = S;
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

    flash_attn_bwd_launch(bwd, sm, 0, stream);
    FLASH_ATTN_CHECK_CUDA(cudaGetLastError());

    torch::Tensor dQ = dQ_fp32.to(torch::kFloat16);
    return {dQ, sum_group_slices(dK, S), sum_group_slices(dV, S), dsink};
}

// ---------------------------------------------------------------------------
// Forward — Variable-length (packed sequences)
// Q [total_q, H_q, d], K [total_k, H_kv, d], V [total_k, H_kv, d]
// cu_seqlens_q [num_seqs+1], cu_seqlens_k [num_seqs+1]
// ---------------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> flash_attn_forward_varlen(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor cu_seqlens_q, torch::Tensor cu_seqlens_k,
    int64_t max_seqlen_q, int64_t max_seqlen_k,
    double softmax_scale, bool is_causal, int64_t window_left, int64_t window_right, double softcap,
    c10::optional<torch::Tensor> sinks, c10::optional<torch::Tensor> score_scale, bool validate
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
    const int Br = fwd_block_rows(sm, d);
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
    const torch::Tensor scale_dev = device_score_scale(score_scale, Q);
    p.score_scale = data_or_null(scale_dev);

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
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> flash_attn_backward_varlen(
    torch::Tensor dO,
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor O, torch::Tensor L,
    torch::Tensor cu_seqlens_q, torch::Tensor cu_seqlens_k,
    int64_t max_seqlen_q, int64_t max_seqlen_k,
    double softmax_scale, bool is_causal, int64_t window_left, int64_t window_right, double softcap,
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
    torch::Tensor dsink = sink_grad(sink, Q);
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
    // ([total_k, H_kv * S, d])
    const int S = bwd_group_splits(H_q, H_kv, int64_t(k_blocks.total) * H_kv);
    torch::Tensor dK = torch::empty({total_k, H_kv * S, d}, K.options());
    torch::Tensor dV = torch::empty({total_k, H_kv * S, d}, V.options());
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
    bwd.batch_size = 1; bwd.num_heads = H_q; bwd.num_heads_k = H_kv; bwd.kv_group_splits = S; bwd.head_dim = d;
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
    return {dQ_fp32.to(torch::kFloat16), sum_group_slices(dK, S), sum_group_slices(dV, S), dsink};
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
// upstream. key_mask: optional bool [B, max_len] (True = attend, any batch stride):
// masked keys are skipped (padding of any shape, no host sync). fp16 or bf16, read
// natively.
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

// A paged cache [num_blocks, block_size, H_kv, d] and its block_table [B, max_blocks] int32: the
// table made contiguous, max_len = max_blocks * block_size, log2(block_size) for a power of 2
// (shifts instead of divisions; else -1).
struct PagedCache {
    torch::Tensor table;
    int max_len, block_shift;
};

static PagedCache check_paged_cache(const torch::Tensor& k_cache, const torch::Tensor& table, const torch::Tensor& q,
                                    int B) {
    TORCH_CHECK(table.is_cuda() && table.device() == q.device() && table.dtype() == torch::kInt32 &&
                table.dim() == 2 && table.size(0) == B, "block_table must be int32 [B, max_blocks] on q's device");
    TORCH_CHECK(k_cache.size(0) > 0, "paged KV cache: no blocks");
    TORCH_CHECK(k_cache.size(1) > 0 && k_cache.size(1) % 16 == 0,
                "paged KV cache: block_size must be a positive multiple of 16, got ", k_cache.size(1));
    TORCH_CHECK(int64_t(table.size(1)) * k_cache.size(1) <= INT32_MAX, "paged KV cache too long");
    int block_shift = -1;
    if ((k_cache.size(1) & (k_cache.size(1) - 1)) == 0) {
        block_shift = 0;
        while ((int64_t(1) << block_shift) < k_cache.size(1)) block_shift++;
    }
    return {table.contiguous(), int(table.size(1) * k_cache.size(1)), block_shift};
}

std::tuple<torch::Tensor, torch::Tensor> flash_attn_with_kvcache(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache, c10::optional<torch::Tensor> cache_seqlens,
    double softmax_scale, bool is_causal, int64_t window_left, int64_t window_right, double softcap,
    c10::optional<torch::Tensor> sinks, int64_t num_splits, c10::optional<torch::Tensor> cache_leftpad,
    c10::optional<torch::Tensor> block_table, c10::optional<torch::Tensor> key_mask, int64_t uniform_seqlen
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
    int block_shift = -1;
    if (paged) {
        TORCH_CHECK(!(cache_leftpad.has_value() && cache_leftpad->defined()),
                    "cache_leftpad is not supported with a paged KV cache (block_table)");
        PagedCache pc = check_paged_cache(k_cache, *block_table, q, B);
        table = pc.table; max_len = pc.max_len; block_shift = pc.block_shift;
    } else {
        TORCH_CHECK(k_cache.size(0) == B, "k_cache must be [B, max_len, H_kv, d]");
    }
    TORCH_CHECK(k_cache.size(3) == d, "k_cache head_dim must match q");
    TORCH_CHECK(H_kv > 0 && H_q % H_kv == 0, "num_heads_q (", H_q, ") must be divisible by num_heads_kv (", H_kv, ")");
    // More query rows per KV head than the decode kernel takes (speculative
    // verification, MQA): the forward kernels with key splits, on the same cache
    const int rows = H_q / H_kv * q_len;
    const bool on_forward = rows > flash_attn_decode_max_rows();
    q = contiguous_aligned(q);
    check_decode_layout(k_cache, "k_cache");
    check_decode_layout(v_cache, "v_cache");
    torch::Tensor seqlens;  // undefined: every row ends at uniform_seqlen (-1: max_len)
    if (cache_seqlens.has_value() && cache_seqlens->defined()) {
        seqlens = check_batch_int32(*cache_seqlens, q, B, "cache_seqlens");
    }
    TORCH_CHECK(uniform_seqlen >= -1 && uniform_seqlen <= max_len, "uniform_seqlen=", uniform_seqlen,
                " outside [0, max_len=", max_len, "] (-1: max_len)");
    torch::Tensor leftpad;
    if (cache_leftpad.has_value() && cache_leftpad->defined()) {
        leftpad = check_batch_int32(*cache_leftpad, q, B, "cache_leftpad");
    }
    const torch::Tensor sink = sink_logits(sinks, q, H_q);
    torch::Tensor mask;
    if (key_mask.has_value() && key_mask->defined()) {
        mask = *key_mask;
        TORCH_CHECK(mask.is_cuda() && mask.device() == q.device() && mask.dtype() == torch::kBool && mask.dim() == 2 &&
                    mask.size(0) == B && mask.size(1) == max_len,
                    "key_mask must be bool [B, max_len] = [", B, ", ", max_len, "] on q's device");
        if (mask.stride(1) != 1) mask = mask.contiguous();  // any batch stride (0: broadcast)
    }

    torch::Tensor out = torch::empty_like(q);
    torch::Tensor lse = torch::empty({B, H_q, q_len}, q.options().dtype(torch::kFloat32));
    if (out.numel() == 0) return {out, lse};

    const at::cuda::CUDAGuard device_guard(q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int sm = get_sm_version();  // a kernel image for this GPU
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    TORCH_CHECK(num_splits >= 0, "num_splits must be >= 0 (0 = automatic)");
    if (on_forward) {  // q, out [B, q_len, H_q, d] and the cache [B, max_len, H_kv, d] read through their strides
        FlashAttnParams f = {};
        // fp16, or bf16 read as fp32 by the CUDA-core body (exact: no fp16 range limit) on any GPU
        const bool bf16 = q.dtype() == torch::kBFloat16;
        f.bf16 = bf16;
        const int fam = bf16 ? 60 : sm;  // the kernel family that runs
        f.Q = reinterpret_cast<const half*>(q.data_ptr());
        f.K = reinterpret_cast<const half*>(k_cache.data_ptr());
        f.V = reinterpret_cast<const half*>(v_cache.data_ptr());
        f.O = nullptr;  // the cache kernel writes partials; the combine writes out
        f.L = lse.data_ptr<float>();
        f.batch_size = B; f.num_heads = H_q; f.num_heads_k = H_kv; f.head_dim = d;
        f.seq_len_q = q_len;
        f.seq_len_k = uniform_seqlen >= 0 ? int(uniform_seqlen) : max_len;
        f.seqlens_k = seqlens.defined() ? seqlens.data_ptr<int>() : nullptr;
        f.leftpad_k = leftpad.defined() ? leftpad.data_ptr<int>() : nullptr;
        f.key_mask = mask.defined() ? mask.data_ptr<bool>() : nullptr;
        f.mask_batch_stride = mask.defined() ? mask.stride(0) : 0;
        f.block_table = paged ? table.data_ptr<int>() : nullptr;
        f.max_blocks = paged ? table.size(1) : 0;
        f.block_size = paged ? k_cache.size(1) : 0;
        f.block_shift = block_shift;
        f.q_batch_stride = q.stride(0); f.q_seq_stride = q.stride(1); f.q_head_stride = q.stride(2);
        f.k_batch_stride = k_cache.stride(0); f.k_seq_stride = k_cache.stride(1); f.k_head_stride = k_cache.stride(2);
        f.v_batch_stride = v_cache.stride(0); f.v_seq_stride = v_cache.stride(1); f.v_head_stride = v_cache.stride(2);
        f.o_batch_stride = out.stride(0); f.o_seq_stride = out.stride(1); f.o_head_stride = out.stride(2);
        f.l_batch_stride = index_t(H_q) * q_len; f.l_head_stride = q_len;
        f.softmax_scale = softmax_scale;
        f.softcap = logit_softcap(softcap);
        f.sinks = data_or_null(sink);
        std::tie(f.window_left, f.window_right) = key_window(is_causal, window_left, window_right, q_len, max_len);
        // Packed GQA rows: the q_len tokens x H_q / H_kv query heads of a KV head in one block (its K / V
        // read once for them; sinks one per row)
        f.pack_gqa = H_q > H_kv;
        const int rpt = gqa_rows_per_token(f.pack_gqa, H_q, H_kv);
        // Splits: each block reads its KV head's cache for its q_len * rpt rows (resident blocks per SM: 4 on
        // Pascal, compute bound on small tiles; 1 on SM 7.x, bandwidth bound on large ones)
        const int blocks = cdiv(q_len * rpt, fwd_block_rows(fam, d)) * B * (H_q / rpt);
        f.num_splits = num_splits > 0 ? int(std::min<int64_t>(num_splits, flash_attn_max_splits()))
                                      : flash_attn_num_splits(blocks, sms, fam < 70 ? 4 : 1, max_len, q_len * rpt);
        // Partials of every split, even of one (the kernel variant of the cache; see fwd_batched_kernel)
        torch::Tensor o_part = torch::empty({f.num_splits, B, H_q, q_len, d}, q.options().dtype(torch::kFloat32));
        torch::Tensor lse_part = torch::empty({f.num_splits, B, H_q, q_len}, q.options().dtype(torch::kFloat32));
        f.O_part = o_part.data_ptr<float>();
        f.L_part = lse_part.data_ptr<float>();
        run_fwd(f, fam, stream);
        flash_attn_split_combine({f.O_part, f.L_part, out.data_ptr(), lse.data_ptr<float>(), out.stride(0),
                                  out.stride(1), out.stride(2), B, q_len, H_q, d, f.num_splits}, bf16, stream);
        FLASH_ATTN_CHECK_CUDA(cudaGetLastError());
        return {out, lse};
    }

    DecodeParams p = {};
    p.q = q.data_ptr(); p.k = k_cache.data_ptr(); p.v = v_cache.data_ptr(); p.out = out.data_ptr();
    p.lse = lse.data_ptr<float>();
    p.cache_seqlens = seqlens.defined() ? seqlens.data_ptr<int>() : nullptr;
    p.uniform_seqlen = uniform_seqlen >= 0 ? int(uniform_seqlen) : max_len;
    p.cache_leftpad = leftpad.defined() ? leftpad.data_ptr<int>() : nullptr;
    p.block_table = paged ? table.data_ptr<int>() : nullptr;
    p.max_blocks = paged ? table.size(1) : 0;
    p.block_size = paged ? k_cache.size(1) : 0;
    p.block_shift = block_shift;
    p.sinks = data_or_null(sink);
    p.key_mask = mask.defined() ? mask.data_ptr<bool>() : nullptr;
    p.mask_batch_stride = mask.defined() ? mask.stride(0) : 0;
    p.q_batch_stride = q.stride(0); p.q_seq_stride = q.stride(1); p.q_head_stride = q.stride(2);
    p.k_batch_stride = k_cache.stride(0); p.k_seq_stride = k_cache.stride(1); p.k_head_stride = k_cache.stride(2);
    p.v_batch_stride = v_cache.stride(0); p.v_seq_stride = v_cache.stride(1); p.v_head_stride = v_cache.stride(2);
    p.o_batch_stride = out.stride(0); p.o_seq_stride = out.stride(1); p.o_head_stride = out.stride(2);
    p.batch_size = B; p.q_len = q_len; p.num_heads = H_q; p.num_heads_k = H_kv; p.head_dim = d;
    p.max_len = max_len;
    // Key splits: num_splits > 0 forced (capped), 0 = chosen from the grid
    p.num_splits = num_splits > 0 ? int(std::min<int64_t>(num_splits, flash_attn_max_splits()))
                                  : flash_attn_num_splits(B * H_kv, sms, 4, max_len, rows);
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

// k, v [B, seqlen_new, H_kv, d] written into the caches at cache_seqlens (or uniform_seqlen) per row
// (KvAppendParams): flash_attn_with_kvcache's k / v, before its attention.
void flash_attn_kvcache_append(
    const torch::Tensor& k_cache, const torch::Tensor& v_cache, torch::Tensor k, torch::Tensor v,
    c10::optional<torch::Tensor> cache_seqlens, int64_t uniform_seqlen, c10::optional<torch::Tensor> block_table
) {
    TORCH_CHECK(k.dim() == 4 && k_cache.dim() == 4 && k.sizes() == v.sizes() && k_cache.sizes() == v_cache.sizes(),
                "k / v must be [B, seqlen_new, H_kv, d] and k_cache / v_cache 4D of one shape");
    TORCH_CHECK(k.is_cuda() && k_cache.device() == k.device() && v_cache.device() == k.device() &&
                v.device() == k.device(), "k, v, k_cache, v_cache must be on the same CUDA device");
    TORCH_CHECK(k_cache.dtype() == k.dtype() && v_cache.dtype() == k.dtype() && v.dtype() == k.dtype() &&
                (k.dtype() == torch::kFloat16 || k.dtype() == torch::kBFloat16),
                "k, v, k_cache, v_cache must be float16 or bfloat16, one dtype");
    TORCH_CHECK(k.size(0) <= INT32_MAX && k.size(1) <= INT32_MAX && k_cache.size(1) <= INT32_MAX,
                "batch and sequence lengths must fit in int32");
    const int B = k.size(0), n = k.size(1), H_kv = k.size(2), d = k.size(3);
    TORCH_CHECK(k_cache.size(2) == H_kv && k_cache.size(3) == d, "k / v heads and head_dim must match the caches");
    check_head_dim(d);
    const bool paged = block_table.has_value() && block_table->defined();
    KvAppendParams p = {};
    p.max_len = k_cache.size(1);
    torch::Tensor table;
    if (paged) {
        PagedCache pc = check_paged_cache(k_cache, *block_table, k, B);
        table = pc.table;
        p.block_table = table.data_ptr<int>();
        p.max_blocks = table.size(1); p.block_size = k_cache.size(1); p.block_shift = pc.block_shift;
        TORCH_CHECK(k_cache.size(0) <= INT32_MAX, "paged KV cache: too many blocks");
        p.num_blocks = k_cache.size(0);
        p.max_len = pc.max_len;
    } else {
        TORCH_CHECK(k_cache.size(0) == B, "k_cache must be [B, max_len, H_kv, d]");
    }
    check_decode_layout(k_cache, "k_cache");
    check_decode_layout(v_cache, "v_cache");
    k = contiguous_aligned(k);
    v = contiguous_aligned(v);
    torch::Tensor seqlens;
    if (cache_seqlens.has_value() && cache_seqlens->defined()) {
        seqlens = check_batch_int32(*cache_seqlens, k, B, "cache_seqlens");
        p.cache_seqlens = seqlens.data_ptr<int>();
    } else {
        TORCH_CHECK(uniform_seqlen >= 0 && uniform_seqlen + n <= p.max_len, "uniform_seqlen=", uniform_seqlen,
                    " + seqlen_new=", n, " outside [0, max_len=", p.max_len, "]");
        p.uniform_seqlen = uniform_seqlen;
    }
    p.k_cache = k_cache.data_ptr(); p.v_cache = v_cache.data_ptr(); p.k = k.data_ptr(); p.v = v.data_ptr();
    p.kc_batch_stride = k_cache.stride(0); p.kc_seq_stride = k_cache.stride(1); p.kc_head_stride = k_cache.stride(2);
    p.vc_batch_stride = v_cache.stride(0); p.vc_seq_stride = v_cache.stride(1); p.vc_head_stride = v_cache.stride(2);
    p.k_batch_stride = k.stride(0); p.k_seq_stride = k.stride(1); p.k_head_stride = k.stride(2);
    p.v_batch_stride = v.stride(0); p.v_seq_stride = v.stride(1); p.v_head_stride = v.stride(2);
    p.batch_size = B; p.seqlen_new = n; p.num_heads_k = H_kv; p.head_dim = d;
    if (k.numel() == 0) return;
    const at::cuda::CUDAGuard device_guard(k.device());
    flash_attn_kvcache_append_launch(p, at::cuda::getCurrentCUDAStream());
    FLASH_ATTN_CHECK_CUDA(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// FP16 linear layer on the P100 (docs/PLAN.md P8, linear_pascal.cu): y [M, N] =
// x [M, K] W [N, K]^T (+ bias [N]), HFMA2 with chunked accumulation. Compute
// capability 6.0 only, real hardware (FLASH_ATTN_LEGACY_FORCE_ARCH does not
// apply: on other GPUs the kernel is a stub). Inputs that are not contiguous or
// not 16-byte aligned are copied (contiguous_aligned; the Python API sends them
// to F.linear instead, outside torch.compile).
//
// Stream-K workspace: the partials (two fp32 tiles per stream-K block) come from
// the caching allocator on the current stream, per call, uninitialized. The
// arrival counters (one per stream-K tile) must be 0 at launch, and the kernel
// leaves them at 0 (the last arriver of a tile resets its counter):
//   - eager launches: one persistent buffer per (device, stream), zeroed when it
//     is created. Launches on one stream never overlap, launches on different
//     streams use different buffers; no per-call memset (a separate memset
//     launch measured 10-20% of a small stream-K GEMM on the P100);
//   - under CUDA graph capture: counters per call from the graph's private pool,
//     zeroed by a captured memset: a graph never shares counters with eager
//     launches or other graphs, and every replay starts from 0.
// Known limit: a persistent buffer is keyed by the stream handle; a stream
// destroyed with work pending and a new stream reusing its handle could overlap
// (PyTorch's pool streams are never destroyed).
// ---------------------------------------------------------------------------
static std::mutex g_linear_counters_mutex;
// Leaked on purpose: no tensor destructor runs after the CUDA driver has shut down at exit
static auto* g_linear_counters = new std::map<std::pair<int, cudaStream_t>, torch::Tensor>();

// The device runs the linear kernel: compute capability 6.0 and a build with SM 6.0 code.
static void check_linear_device(int device_id) {
    const int sm = device_sm(device_id);
    TORCH_CHECK(sm == 60, "flash_attn_legacy linear: compute capability 6.0 (P100) only, this GPU is SM ",
                sm / 10, ".", sm % 10);
    compiled_arch_for(sm);  // a clear error when the build has no SM 6.0 code
}

// y = x W^T (W [N, K]), or y = x W (w_kn: W [K, N], N % 8 == 0)
torch::Tensor linear_forward(torch::Tensor x, torch::Tensor W, c10::optional<torch::Tensor> bias, bool w_kn) {
    TORCH_CHECK(x.is_cuda() && W.device() == x.device(), "x and W must be on the same CUDA device");
    TORCH_CHECK(x.dtype() == torch::kFloat16 && W.dtype() == torch::kFloat16, "x and W must be float16");
    TORCH_CHECK(x.dim() == 2 && W.dim() == 2 && W.size(w_kn ? 0 : 1) == x.size(1),
                "x must be [M, K] and W ", w_kn ? "[K, N]" : "[N, K]", ", got ", x.sizes(), " and ", W.sizes());
    const int64_t M = x.size(0), K = x.size(1), N = W.size(w_kn ? 1 : 0);
    TORCH_CHECK(K % 8 == 0, "K must be a multiple of 8 (128-bit loads), got ", K);
    TORCH_CHECK(!w_kn || N % 8 == 0, "N must be a multiple of 8 with W [K, N] (128-bit loads), got ", N);
    // 32-bit element offsets in the kernel: row * K + k stays below M * K + 32 (N * K + 32 for W)
    // (w_kn: the last stage addresses W rows up to K rounded up to 32, its loads beyond K masked)
    const int64_t w_elems = w_kn ? N * ((K + 31) / 32 * 32) : N * K + 32;
    TORCH_CHECK(M * N < (int64_t(1) << 31) && M * K + 32 < (int64_t(1) << 31) && w_elems < (int64_t(1) << 31),
                "M * N, M * K and N * K must fit the kernel's int32 offsets",
                w_kn ? " (W [K, N]: N * K rounded up to 32)" : "",
                ", got M=", M, " N=", N, " K=", K);
    torch::Tensor b;
    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->device() == x.device() && bias->dtype() == torch::kFloat16 && bias->dim() == 1 &&
                    bias->size(0) == N, "bias must be float16 [N] = [", N, "] on the device of x");
        b = bias->contiguous();
    }
    const at::cuda::CUDAGuard device_guard(x.device());
    check_linear_device(x.get_device());
    if (M == 0 || N == 0 || K == 0) {  // no product (a 0-sized grid is invalid): y = bias, or 0
        torch::Tensor y = torch::zeros({M, N}, x.options());
        if (b.defined()) y.copy_(b.expand({M, N}));
        return y;
    }
    x = contiguous_aligned(x);
    W = contiguous_aligned(W);
    torch::Tensor y = torch::empty({M, N}, x.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    LinearParams p = {};
    p.x = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
    p.w = reinterpret_cast<const half*>(W.data_ptr<at::Half>());
    p.bias = b.defined() ? reinterpret_cast<const half*>(b.data_ptr<at::Half>()) : nullptr;
    p.y = reinterpret_cast<half*>(y.data_ptr<at::Half>());
    p.M = int(M); p.N = int(N); p.K = int(K);
    p.w_kn = w_kn;
    const LinearPlan plan = linear_pascal_plan(p.M, p.N, p.K);
    torch::Tensor ws, counters;  // held until after the launch (the caching allocator orders their reuse)
    if (plan.tile0 < plan.tiles) {
        const int64_t ws_floats = int64_t(plan.sk_blocks) * 2 * kLinearTile * kLinearTile;
        const int64_t n_counters = plan.tiles - plan.tile0;
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        FLASH_ATTN_CHECK_CUDA(cudaStreamIsCapturing(stream, &capture));
        if (capture == cudaStreamCaptureStatusNone) {
            ws = torch::empty({ws_floats}, x.options().dtype(torch::kFloat32));
            std::lock_guard<std::mutex> lock(g_linear_counters_mutex);
            torch::Tensor& c = (*g_linear_counters)[{int(x.get_device()), stream}];
            // (Re)created zeroed on this stream; a replaced buffer is freed stream-ordered
            if (!c.defined() || c.numel() < n_counters) {
                c = torch::zeros({std::max<int64_t>(n_counters, 1024)}, x.options().dtype(torch::kInt32));
            }
            counters = c;  // a handle: the buffer stays alive through this launch even if replaced meanwhile
            p.counters = counters.data_ptr<int>();
        } else {
            ws = torch::empty({ws_floats + n_counters}, x.options().dtype(torch::kFloat32));
            p.counters = reinterpret_cast<int*>(ws.data_ptr<float>() + ws_floats);
            FLASH_ATTN_CHECK_CUDA(cudaMemsetAsync(p.counters, 0, n_counters * sizeof(int), stream));
        }
        p.ws = ws.data_ptr<float>();
    }
    linear_pascal_launch(p, plan, stream);
    FLASH_ATTN_CHECK_CUDA(cudaGetLastError());
    return y;
}

// (tiles, first stream-K tile, stream-K blocks, resident blocks) of linear_forward on the current device (tests).
std::vector<int64_t> linear_plan(int64_t M, int64_t N, int64_t K) {
    int dev = 0;
    FLASH_ATTN_CHECK_CUDA(cudaGetDevice(&dev));
    check_linear_device(dev);
    TORCH_CHECK(M > 0 && N > 0 && K > 0 && M * N < (int64_t(1) << 31) && K <= INT32_MAX, "invalid linear shape");
    const LinearPlan plan = linear_pascal_plan(int(M), int(N), int(K));
    return {plan.tiles, plan.tile0, plan.sk_blocks, plan.resident};
}

// ---------------------------------------------------------------------------
// Depthwise causal conv1d of Gated DeltaNet layers (causal_conv1d.cu, ConvParams;
// docs/PLAN.md P6.10 b): x [B, C, L] channel-last (stride 1 along C: transformers
// passes mixed_qkv [B, L, C] transposed), weight [C, W] (W = 2..4, x's dtype or
// float32), bias [C] (the weight's dtype); float16 / bfloat16 / float32. Outputs
// channel-last too ([B, L, C] memory, returned as [B, C, L]): transformers
// transposes them back. The Python wrapper (gated_delta.py) sends other inputs to
// transformers' reference.
// ---------------------------------------------------------------------------
// The dtype code of the fp16 / bf16 / fp32 kernels (the conv, the delta rule's prep): 0, 1, 2
static int dtype_code(const torch::Tensor& t, const char* name) {
    if (t.dtype() == torch::kFloat16) return 0;
    if (t.dtype() == torch::kBFloat16) return 1;
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float16, bfloat16 or float32, got ", t.dtype());
    return 2;
}

// Checks x / weight / bias; returns (weight, bias) contiguous and fills the shape, dtypes and pointers of p
static std::pair<torch::Tensor, torch::Tensor> conv_inputs(
    const torch::Tensor& x, torch::Tensor weight, c10::optional<torch::Tensor> bias, bool silu, ConvParams& p
) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 3, "x must be a CUDA tensor [B, C, L], got ", x.sizes());
    TORCH_CHECK(x.stride(1) == 1 || x.size(1) == 1, "x must be channel-last (stride 1 along C), got strides ",
                x.strides());
    TORCH_CHECK(weight.device() == x.device() && weight.dim() == 2 && weight.size(0) == x.size(1) &&
                weight.size(1) >= 2 && weight.size(1) <= 4, "weight must be [C, W] (W = 2..4) on x's device, got ",
                weight.sizes());
    TORCH_CHECK(x.size(1) <= INT32_MAX && x.size(2) <= INT32_MAX, "x too large, got ", x.sizes());
    // grid: (channel groups, runs of tokens, batch rows): y and z at most 65535
    TORCH_CHECK(x.size(0) <= 65535 && causal_conv1d_runs(x.size(2)) <= 65535,
                "x too large for the conv kernels' grid (B and the runs of tokens at most 65535), got ", x.sizes());
    p.x_dtype = dtype_code(x, "x");
    p.w_dtype = dtype_code(weight, "weight");
    TORCH_CHECK(p.w_dtype == p.x_dtype || p.w_dtype == 2, "weight must be x's dtype or float32");
    weight = weight.contiguous();
    torch::Tensor b;
    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->device() == x.device() && bias->dim() == 1 && bias->size(0) == x.size(1) &&
                    bias->dtype() == weight.dtype(), "bias must be [C] in the weight's dtype on x's device");
        b = bias->contiguous();
    }
    p.batch = x.size(0); p.channels = x.size(1); p.seqlen = x.size(2); p.width = weight.size(1);
    p.silu = silu;
    p.x = x.data_ptr();
    p.x_batch_stride = x.stride(0); p.x_seq_stride = x.stride(2);
    p.weight = weight.data_ptr();
    p.bias = b.defined() ? b.data_ptr() : nullptr;
    return {weight, b};
}

// An empty [B, C, L] tensor laid out as [B, L, C] (x's options)
static torch::Tensor conv_out_like(const torch::Tensor& x) {
    return torch::empty({x.size(0), x.size(2), x.size(1)}, x.options()).transpose(1, 2);
}

// 16-byte loads: C a multiple of the vector, every operand's start and batch / token strides aligned to it
static bool conv_vec(const ConvParams& p, std::initializer_list<const torch::Tensor*> ts) {
    const int n = 16 / ts.begin()[0]->element_size();
    if (p.channels % n) return false;
    for (const torch::Tensor* t : ts) {
        if (reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 || t->stride(0) % n || t->stride(2) % n) return false;
    }
    return true;
}

torch::Tensor causal_conv1d_forward(torch::Tensor x, torch::Tensor weight, c10::optional<torch::Tensor> bias,
                                    bool silu) {
    ConvParams p = {};
    const auto held = conv_inputs(x, weight, bias, silu, p);
    torch::Tensor y = conv_out_like(x);
    p.y = y.data_ptr();
    p.y_batch_stride = y.stride(0); p.y_seq_stride = y.stride(2);
    const at::cuda::CUDAGuard device_guard(x.device());
    causal_conv1d_launch(p, conv_vec(p, {&x, &y}), false, at::cuda::getCurrentCUDAStream());
    FLASH_ATTN_CHECK_CUDA(cudaGetLastError());
    return y;
}

// (dx, dweight, dbias); dweight / dbias empty (0 elements) unless need_dw (dbias also without a bias)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> causal_conv1d_backward(
    torch::Tensor dy, torch::Tensor x, torch::Tensor weight, c10::optional<torch::Tensor> bias, bool silu,
    bool need_dw) {
    ConvParams p = {};
    const auto held = conv_inputs(x, weight, bias, silu, p);
    TORCH_CHECK(dy.sizes() == x.sizes() && dy.dtype() == x.dtype() && dy.device() == x.device(),
                "dy must match x [B, C, L]");
    if (dy.stride(1) != 1 && dy.size(1) != 1) dy = dy.transpose(1, 2).contiguous().transpose(1, 2);
    torch::Tensor dx = conv_out_like(x);
    torch::Tensor dw = torch::empty({0}, held.first.options()), db = torch::empty({0}, held.first.options());
    torch::Tensor partial;
    p.dy = dy.data_ptr();
    p.dy_batch_stride = dy.stride(0); p.dy_seq_stride = dy.stride(2);
    p.dx = dx.data_ptr();
    p.y_batch_stride = dx.stride(0); p.y_seq_stride = dx.stride(2);
    p.runs = causal_conv1d_runs(p.seqlen);
    if (need_dw && p.seqlen > 0 && p.batch > 0) {
        partial = torch::empty({int64_t(p.batch) * p.runs, p.channels, p.width + 1},
                               x.options().dtype(torch::kFloat32));
        dw = torch::empty_like(held.first);
        p.partial = partial.data_ptr<float>();
        p.dweight = dw.data_ptr();
        if (held.second.defined()) {
            db = torch::empty_like(held.second);
            p.dbias = db.data_ptr();
        }
    } else if (need_dw) {
        dw = torch::zeros_like(held.first);
        if (held.second.defined()) db = torch::zeros_like(held.second);
    }
    const at::cuda::CUDAGuard device_guard(x.device());
    causal_conv1d_launch(p, conv_vec(p, {&x, &dy, &dx}), true, at::cuda::getCurrentCUDAStream());
    FLASH_ATTN_CHECK_CUDA(cudaGetLastError());
    return {dx, dw, db};
}

// One decode step: x [B, C, S] the new tokens, conv_state [B, C, state_len] (state_len >= W - 1, any dtype of the
// three, any strides) updated in place (transformers' causal_conv1d_update); returns y [B, C, S] channel-last
torch::Tensor causal_conv1d_update(torch::Tensor x, const torch::Tensor& conv_state, torch::Tensor weight,
                                   c10::optional<torch::Tensor> bias, bool silu) {
    ConvParams p = {};
    const auto held = conv_inputs(x, weight, bias, silu, p);
    TORCH_CHECK(conv_state.device() == x.device() && conv_state.dim() == 3 && conv_state.size(0) == x.size(0) &&
                conv_state.size(1) == x.size(1) && conv_state.size(2) >= p.width - 1 && conv_state.size(2) <= INT32_MAX,
                "conv_state must be [B, C, state_len >= W - 1] on x's device, got ", conv_state.sizes());
    p.state_dtype = dtype_code(conv_state, "conv_state");
    p.state = conv_state.data_ptr();
    p.state_batch_stride = conv_state.stride(0); p.state_channel_stride = conv_state.stride(1);
    p.state_seq_stride = conv_state.stride(2);
    p.state_len = conv_state.size(2);
    torch::Tensor y = conv_out_like(x);
    p.y = y.data_ptr();
    p.y_batch_stride = y.stride(0); p.y_seq_stride = y.stride(2);
    const at::cuda::CUDAGuard device_guard(x.device());
    causal_conv1d_update_launch(p, at::cuda::getCurrentCUDAStream());
    FLASH_ATTN_CHECK_CUDA(cudaGetLastError());
    return y;
}

// ---------------------------------------------------------------------------
// The sequential scan of the chunk gated delta rule (gdn_scan.cu, GdnScanParams;
// docs/PLAN.md P6.10 c): w, kt [B, H, NC, 64, K], u [B, H, NC, 64, V],
// d [B, H, NC], initial_state [B, H, K, V], fp32 (the reference's intra-chunk
// phase computes them; gated_delta.py). The forward returns (h [B, H, NC, K, V],
// the state at each chunk's start; v_new [B, H, NC, 64, V]; the final state
// [B, H, K, V]); the backward, from the gradients of h / v_new / the final state,
// (G [B, H, NC, K, V], the gradient of the state after each chunk; dv^ [B, H, NC,
// 64, V], all of v_new's gradient; the initial state's gradient). Autograd is
// registered in gated_delta.py.
// ---------------------------------------------------------------------------
// v_dim: V (the backward's dv_new may be undefined); u: u, or dv_new (backward, optional)
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> gdn_scan_run(
    torch::Tensor w, torch::Tensor kt, torch::Tensor d, c10::optional<torch::Tensor> u_opt, int64_t v_dim,
    c10::optional<torch::Tensor> state, c10::optional<torch::Tensor> dh, bool backward
) {
    TORCH_CHECK(w.is_cuda() && w.dim() == 5 && w.size(3) == 64, "w must be a CUDA tensor [B, H, NC, 64, K]");
    const int64_t B = w.size(0), H = w.size(1), NC = w.size(2), K = w.size(4), V = v_dim;
    torch::Tensor u;
    if (u_opt.has_value() && u_opt->defined()) {
        u = *u_opt;
        TORCH_CHECK(u.dim() == 5 && u.size(0) == B && u.size(1) == H && u.size(2) == NC && u.size(3) == 64 &&
                    u.size(4) == V, "u / dv_new must be [B, H, NC, 64, V]");
    }
    TORCH_CHECK(kt.sizes() == w.sizes() && d.dim() == 3 && d.size(0) == B && d.size(1) == H && d.size(2) == NC,
                "kt must be w's shape and d [B, H, NC]");
    TORCH_CHECK(K == 64 || K == 128, "K must be 64 or 128, got ", K);
    TORCH_CHECK(V % 16 == 0 && V > 0, "V must be a positive multiple of 16, got ", V);
    TORCH_CHECK(B <= 65535 && H <= 65535 && NC <= INT32_MAX, "too many rows / heads / chunks");
    for (const torch::Tensor* t : {&w, &u, &kt, &d}) {
        TORCH_CHECK(!t->defined() || (t->dtype() == torch::kFloat32 && t->device() == w.device()),
                    "inputs must be float32 on w's device");
    }
    w = contiguous_aligned(w); kt = contiguous_aligned(kt);  // 16-byte loads of w / kt rows
    d = d.contiguous();
    if (u.defined()) u = u.contiguous();
    torch::Tensor s0, dhc;
    if (state.has_value() && state->defined()) {
        TORCH_CHECK(state->dtype() == torch::kFloat32 && state->device() == w.device() && state->dim() == 4 &&
                    state->size(0) == B && state->size(1) == H && state->size(2) == K && state->size(3) == V,
                    "the state must be float32 [B, H, K, V] on w's device");
        s0 = state->contiguous();
    }
    if (backward && dh.has_value() && dh->defined()) {
        TORCH_CHECK(dh->dtype() == torch::kFloat32 && dh->device() == w.device() && dh->dim() == 5 &&
                    dh->size(0) == B && dh->size(1) == H && dh->size(2) == NC && dh->size(3) == K && dh->size(4) == V,
                    "dh must be float32 [B, H, NC, K, V] on w's device");
        dhc = dh->contiguous();
    }
    torch::Tensor h = torch::empty({B, H, NC, K, V}, w.options());
    torch::Tensor v_new = torch::empty({B, H, NC, 64, V}, w.options());
    torch::Tensor s_final = torch::empty({B, H, K, V}, w.options());
    if (NC == 0) {
        if (s0.defined()) s_final.copy_(s0); else s_final.zero_();
        return {h, v_new, s_final};
    }
    GdnScanParams p = {};
    p.w = w.data_ptr<float>(); p.kt = kt.data_ptr<float>(); p.d = d.data_ptr<float>();
    p.u = u.defined() ? u.data_ptr<float>() : nullptr;
    p.s0 = s0.defined() ? s0.data_ptr<float>() : nullptr;
    p.dh = dhc.defined() ? dhc.data_ptr<float>() : nullptr;
    p.h = h.data_ptr<float>(); p.v_new = v_new.data_ptr<float>(); p.s_final = s_final.data_ptr<float>();
    p.batch = B; p.heads = H; p.chunks = NC; p.k_dim = K; p.v_dim = V;
    const at::cuda::CUDAGuard device_guard(w.device());
    int sms = 0;
    FLASH_ATTN_CHECK_CUDA(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, w.get_device()));
    // 32 state columns per block, 16 below two blocks per SM (a block of 256 threads at 64-96 registers: two or three
    // per SM; shared memory 32 KB at K = 128, VS = 32, 20 KB at VS = 16)
    const int vs = (V % 32 == 0 && B * H * (V / 32) >= 2 * sms) ? 32 : 16;
    gdn_scan_launch(p, vs, backward, at::cuda::getCurrentCUDAStream());
    FLASH_ATTN_CHECK_CUDA(cudaGetLastError());
    return {h, v_new, s_final};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> gdn_chunk_scan(
    torch::Tensor w, torch::Tensor u, torch::Tensor kt, torch::Tensor d, c10::optional<torch::Tensor> initial_state
) {
    TORCH_CHECK(u.dim() == 5, "u must be [B, H, NC, 64, V]");
    return gdn_scan_run(w, kt, d, u, u.size(4), initial_state, c10::nullopt, false);
}

// (G, dv^, the initial state's gradient) from w, kt, d and the gradients of h, v_new and the final state (each may
// be None: zeros); v_dim: V
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> gdn_chunk_scan_backward(
    torch::Tensor w, torch::Tensor kt, torch::Tensor d, c10::optional<torch::Tensor> dh,
    c10::optional<torch::Tensor> dv_new, c10::optional<torch::Tensor> d_final_state, int64_t v_dim
) {
    return gdn_scan_run(w, kt, d, dv_new, v_dim, d_final_state, dh, true);
}

// ---------------------------------------------------------------------------
// The intra-chunk phase of the chunk gated delta rule (gdn_prep.cu, GdnPrepParams;
// docs/PLAN.md P6.10 c2): q, k [B, L, H, K] and v [B, L, H, V] in one dtype
// (float16 / bfloat16 / float32, any strides with the last dimension contiguous),
// g and beta [B, L, H] (cast to float32 here). Returns, fp32, NC = ceil(L / 64):
// (q~, k~, w [B, H, NC, 64, K], u [B, H, NC, 64, V], d [B, H, NC], A, T
// [B, H, NC, 64, 64]). The analytic backward is registered in gated_delta.py.
// ---------------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
gdn_chunk_prep(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta, double scale) {
    TORCH_CHECK(q.is_cuda() && q.dim() == 4 && k.sizes() == q.sizes(), "q and k must be CUDA tensors [B, L, H, K]");
    TORCH_CHECK(v.dim() == 4 && v.size(0) == q.size(0) && v.size(1) == q.size(1) && v.size(2) == q.size(2),
                "v must be [B, L, H, V]");
    const int64_t B = q.size(0), L = q.size(1), H = q.size(2), K = q.size(3), V = v.size(3);
    TORCH_CHECK(K % 16 == 0 && K > 0 && V % 16 == 0 && V > 0, "K and V must be positive multiples of 16, got ", K,
                " and ", V);
    TORCH_CHECK(g.dim() == 3 && g.size(0) == B && g.size(1) == L && g.size(2) == H && beta.sizes() == g.sizes(),
                "g and beta must be [B, L, H]");
    TORCH_CHECK(B <= 65535 && H <= 65535 && L <= INT32_MAX - 64, "too many rows / heads / tokens");
    const int dtype = dtype_code(q, "q");
    TORCH_CHECK(k.dtype() == q.dtype() && v.dtype() == q.dtype(), "q, k and v must have one dtype");
    for (const torch::Tensor* t : {&k, &v, &g, &beta}) {
        TORCH_CHECK(t->device() == q.device(), "inputs must be on q's device");
    }
    TORCH_CHECK(g.is_floating_point() && beta.is_floating_point(), "g and beta must be floating point");
    if (q.stride(3) != 1) q = q.contiguous();
    if (k.stride(3) != 1) k = k.contiguous();
    if (v.stride(3) != 1) v = v.contiguous();
    g = g.to(torch::kFloat32).contiguous();
    beta = beta.to(torch::kFloat32).contiguous();
    const int64_t NC = (L + 63) / 64;
    const auto f32 = q.options().dtype(torch::kFloat32);
    torch::Tensor qt = torch::empty({B, H, NC, 64, K}, f32), kt = torch::empty({B, H, NC, 64, K}, f32);
    torch::Tensor w = torch::empty({B, H, NC, 64, K}, f32), u = torch::empty({B, H, NC, 64, V}, f32);
    torch::Tensor d = torch::empty({B, H, NC}, f32);
    torch::Tensor A = torch::empty({B, H, NC, 64, 64}, f32), T = torch::empty({B, H, NC, 64, 64}, f32);
    GdnPrepParams p = {};
    p.q = q.data_ptr(); p.k = k.data_ptr(); p.v = v.data_ptr();
    p.g = g.data_ptr<float>(); p.beta = beta.data_ptr<float>();
    p.qt = qt.data_ptr<float>(); p.kt = kt.data_ptr<float>(); p.w = w.data_ptr<float>(); p.u = u.data_ptr<float>();
    p.d = d.data_ptr<float>(); p.A = A.data_ptr<float>(); p.T = T.data_ptr<float>();
    p.q_batch_stride = q.stride(0); p.q_seq_stride = q.stride(1); p.q_head_stride = q.stride(2);
    p.k_batch_stride = k.stride(0); p.k_seq_stride = k.stride(1); p.k_head_stride = k.stride(2);
    p.v_batch_stride = v.stride(0); p.v_seq_stride = v.stride(1); p.v_head_stride = v.stride(2);
    p.scale = static_cast<float>(scale);
    p.batch = B; p.seqlen = L; p.heads = H; p.chunks = NC; p.k_dim = K; p.v_dim = V; p.dtype = dtype;
    const at::cuda::CUDAGuard device_guard(q.device());
    gdn_prep_launch(p, at::cuda::getCurrentCUDAStream());
    FLASH_ATTN_CHECK_CUDA(cudaGetLastError());
    return {qt, kt, w, u, d, A, T};
}

}  // namespace flash_attn_legacy

// The kernels are dispatcher operators (torch.ops.flash_attn_legacy.*): torch.compile
// and CUDA graphs see them as opaque ops (fake implementations in flash_attn.py), at
// the cost of one dispatcher call. Default arguments as in the Python wrappers.
TORCH_LIBRARY(flash_attn_legacy, m) {
    m.def("forward(Tensor Q, Tensor K, Tensor V, float softmax_scale, bool is_causal=False, int window_left=-1, "
          "int window_right=-1, float softcap=0., Tensor? sinks=None, Tensor? score_scale=None) -> (Tensor, Tensor)");
    m.def("backward(Tensor dO, Tensor Q, Tensor K, Tensor V, Tensor O, Tensor L, float softmax_scale, "
          "bool is_causal=False, int window_left=-1, int window_right=-1, float softcap=0., Tensor? sinks=None) "
          "-> (Tensor, Tensor, Tensor, Tensor)");
    m.def("forward_varlen(Tensor Q, Tensor K, Tensor V, Tensor cu_seqlens_q, Tensor cu_seqlens_k, int max_seqlen_q, "
          "int max_seqlen_k, float softmax_scale, bool is_causal=False, int window_left=-1, int window_right=-1, "
          "float softcap=0., Tensor? sinks=None, Tensor? score_scale=None, bool validate=True) -> (Tensor, Tensor)");
    m.def("backward_varlen(Tensor dO, Tensor Q, Tensor K, Tensor V, Tensor O, Tensor L, Tensor cu_seqlens_q, "
          "Tensor cu_seqlens_k, int max_seqlen_q, int max_seqlen_k, float softmax_scale, bool is_causal=False, "
          "int window_left=-1, int window_right=-1, float softcap=0., Tensor? sinks=None, bool validate=True) "
          "-> (Tensor, Tensor, Tensor, Tensor)");
    m.def("with_kvcache(Tensor q, Tensor k_cache, Tensor v_cache, Tensor? cache_seqlens, float softmax_scale, "
          "bool is_causal=False, int window_left=-1, int window_right=-1, float softcap=0., Tensor? sinks=None, "
          "int num_splits=0, Tensor? cache_leftpad=None, Tensor? block_table=None, Tensor? key_mask=None, "
          "int uniform_seqlen=-1) -> (Tensor, Tensor)");
    m.def("kvcache_append(Tensor(a!) k_cache, Tensor(b!) v_cache, Tensor k, Tensor v, Tensor? cache_seqlens, "
          "int uniform_seqlen=-1, Tensor? block_table=None) -> ()");
    m.def("linear(Tensor x, Tensor W, Tensor? bias=None, bool w_kn=False) -> Tensor");
    m.def("causal_conv1d(Tensor x, Tensor weight, Tensor? bias=None, bool silu=False) -> Tensor");
    m.def("causal_conv1d_backward(Tensor dy, Tensor x, Tensor weight, Tensor? bias, bool silu, bool need_dw) "
          "-> (Tensor, Tensor, Tensor)");
    m.def("causal_conv1d_update(Tensor x, Tensor(a!) conv_state, Tensor weight, Tensor? bias=None, "
          "bool silu=False) -> Tensor");
    m.def("gdn_chunk_scan(Tensor w, Tensor u, Tensor kt, Tensor d, Tensor? initial_state=None) "
          "-> (Tensor, Tensor, Tensor)");
    m.def("gdn_chunk_scan_backward(Tensor w, Tensor kt, Tensor d, Tensor? dh, Tensor? dv_new, "
          "Tensor? d_final_state, int v_dim) -> (Tensor, Tensor, Tensor)");
    m.def("gdn_chunk_prep(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, float scale) "
          "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(flash_attn_legacy, CUDA, m) {
    m.impl("forward", &flash_attn_legacy::flash_attn_forward);
    m.impl("backward", &flash_attn_legacy::flash_attn_backward);
    m.impl("forward_varlen", &flash_attn_legacy::flash_attn_forward_varlen);
    m.impl("backward_varlen", &flash_attn_legacy::flash_attn_backward_varlen);
    m.impl("with_kvcache", &flash_attn_legacy::flash_attn_with_kvcache);
    m.impl("kvcache_append", &flash_attn_legacy::flash_attn_kvcache_append);
    m.impl("linear", &flash_attn_legacy::linear_forward);
    m.impl("causal_conv1d", &flash_attn_legacy::causal_conv1d_forward);
    m.impl("causal_conv1d_backward", &flash_attn_legacy::causal_conv1d_backward);
    m.impl("causal_conv1d_update", &flash_attn_legacy::causal_conv1d_update);
    m.impl("gdn_chunk_scan", &flash_attn_legacy::gdn_chunk_scan);
    m.impl("gdn_chunk_scan_backward", &flash_attn_legacy::gdn_chunk_scan_backward);
    m.impl("gdn_chunk_prep", &flash_attn_legacy::gdn_chunk_prep);
}

// Helpers (host-side queries, not traced)
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compiled_archs", &flash_attn_legacy::compiled_archs,
          "CUDA architectures this build contains, e.g. ['6.0', '7.0', '7.5+PTX']");
    m.def("decode_max_rows", &flash_attn_legacy::flash_attn_decode_max_rows,
          "Query rows per KV head ((H_q / H_kv) * q_len) of flash_attn_with_kvcache's decode kernel "
          "(more run on the forward kernels with key splits)");
    m.def("num_splits", &flash_attn_legacy::flash_attn_num_splits,
          "Automatic key splits of flash_attn_with_kvcache for (blocks, SMs, resident blocks per SM, max_len, "
          "query rows per block)", py::arg("blocks"), py::arg("sms"), py::arg("resident"), py::arg("max_len"),
          py::arg("rows"));
    m.def("linear_plan", &flash_attn_legacy::linear_plan,
          "(tiles, first stream-K tile, stream-K blocks, resident blocks) of the linear op for x [M, K], W [N, K] "
          "on the current device (compute capability 6.0)", py::arg("M"), py::arg("N"), py::arg("K"));
}
