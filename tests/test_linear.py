"""
Tests for the FP16 linear layer of the P100 (flash_attn_legacy.linear, docs/PLAN.md P8): the HFMA2 kernel with
chunked accumulation (data-parallel + stream-K), its operator, autograd, fallbacks and patch_linear.

Kernel tests run on compute capability 6.0 only; on other GPUs only the no-op / fallback tests run.

Run:  pytest tests/test_linear.py -v
"""

import contextlib
import copy
import itertools
import os
import time

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

flash_attn_legacy = pytest.importorskip("flash_attn_legacy")

from flash_attn_legacy import fallback_stats, fused_linear, patch_linear, reset_fallback_stats  # noqa: E402
from flash_attn_legacy import linear as linear_mod  # noqa: E402


def _is_sm60():
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (6, 0)


sm60_only = pytest.mark.skipif(not _is_sm60(), reason="the HFMA2 linear kernel runs on compute capability 6.0 only")
compile_ready = pytest.mark.skipif(not hasattr(torch.library, "opcheck"), reason="torch >= 2.4 required")

DEV = "cuda"
# Relative Frobenius error vs FP64 of the chunked kernel: 8.7e-4 on real layer shapes with CHUNK = 64 (1.2e-3 with
# 128; cuBLAS COMPUTE_32F: ~2e-4; a chunked cuBLAS COMPUTE_16F emulation: up to 1.4e-3).
CHUNK = 64  # kChunk in csrc/kernels/linear_pascal.cu
TOL = 2.5e-3
_STREAM_K_ENV = "FLASH_ATTN_LEGACY_LINEAR_STREAM_K"


@pytest.fixture(autouse=True)
def _clean_fallback_stats():
    reset_fallback_stats()
    yield
    reset_fallback_stats()


def ops():
    return torch.ops.flash_attn_legacy


def plan(M, N, K):
    """(tiles, first stream-K tile, stream-K blocks, resident blocks) of the op on the current device."""
    import flash_attn_legacy_cuda
    return tuple(flash_attn_legacy_cuda.linear_plan(M, N, K))


@contextlib.contextmanager
def data_parallel_only():
    """The op without stream-K (every tile one block, the whole K): the reference of the stream-K tiles."""
    old = os.environ.get(_STREAM_K_ENV)
    os.environ[_STREAM_K_ENV] = "0"
    try:
        yield
    finally:
        if old is None:
            del os.environ[_STREAM_K_ENV]
        else:
            os.environ[_STREAM_K_ENV] = old


def make(M, N, K, seed, bias=True):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(M, K, device=DEV, dtype=torch.float16, generator=g)
    W = (torch.randn(N, K, device=DEV, dtype=torch.float16, generator=g) * K ** -0.5).half()
    b = torch.randn(N, device=DEV, dtype=torch.float16, generator=g) if bias else None
    return x, W, b


def ref64(x, W, b=None):
    y = x.double() @ W.double().t()
    return y if b is None else y + b.double()


def relf(y, ref):
    return ((y.double() - ref).norm() / ref.norm()).item()


def _sms():
    return torch.cuda.get_device_properties(0).multi_processor_count


# ---------------------------------------------------------------------------
# The kernel (operator torch.ops.flash_attn_legacy.linear)
# ---------------------------------------------------------------------------

_EDGE_M = (1, 3, 17, 129, 511, 512, 1000)
_EDGE_NK = ((64, 8), (100, 72), (257, 136), (896, 896), (1000, 1000), (130, 4864), (1153, 2056))


@sm60_only
class TestKernel:

    @pytest.mark.parametrize("N,K", _EDGE_NK, ids=[f"N{n}-K{k}" for n, k in _EDGE_NK])
    def test_edge_shapes(self, N, K):
        """Tails in M, N, K, with and without bias: finite, within TOL of FP64, data-parallel and stream-K both
        (a stream-K split may cut an fp16 chain earlier: never less accurate)."""
        torch.manual_seed(1234)
        for M, has_bias in itertools.product(_EDGE_M, (False, True)):
            x, W, b = make(M, N, K, seed=M * 7 + has_bias, bias=has_bias)
            ref = ref64(x, W, b)
            y = ops().linear(x, W, b)
            with data_parallel_only():
                y_dp = ops().linear(x, W, b)
            for name, out in (("default", y), ("data-parallel", y_dp)):
                assert out.shape == (M, N) and out.dtype == torch.float16
                assert torch.isfinite(out).all(), f"M={M} bias={has_bias} {name}: non-finite output"
                err = relf(out, ref)
                assert err < TOL, f"M={M} bias={has_bias} {name}: relF vs fp64 {err:.2e}"

    @pytest.mark.parametrize("M,N,K", [(1024, 1792, 896), (1024, 1792, 4864), (1024, 1920, 896),
                                       (1024, 1920, 4864), (2048, 3712, 896), (2048, 3712, 1536),
                                       (1100, 1920, 2056)])
    def test_hybrid(self, M, N, K):
        """Full waves data-parallel + stream-K over the rest: the data-parallel tiles bitwise equal to the
        data-parallel-only run, the stream-K tiles within TOL of FP64, the result reproducible bitwise (the
        reduction order does not depend on the arrival order); tiles filling whole waves of resident blocks:
        no stream-K."""
        x, W, b = make(M, N, K, seed=M + N + K)
        ref = ref64(x, W, b)
        tiles, tile0, blocks, resident = plan(M, N, K)
        y = ops().linear(x, W, b)
        with data_parallel_only():
            assert plan(M, N, K)[:3] == (tiles, tiles, 0)
            y_dp = ops().linear(x, W, b)
        assert all(torch.equal(y, ops().linear(x, W, b)) for _ in range(5)), "stream-K not reproducible"
        if tiles % resident == 0:
            assert tile0 == tiles and blocks == 0 and torch.equal(y, y_dp)
            return
        assert 0 <= tile0 < tiles and blocks > 0
        tiles_n = (N + 127) // 128
        dp = torch.zeros(M, N, dtype=torch.bool, device=DEV)
        for t in range(tile0):
            m0, n0 = (t // tiles_n) * 128, (t % tiles_n) * 128
            dp[m0:m0 + 128, n0:n0 + 128] = True
        assert torch.equal(y[dp], y_dp[dp]), "data-parallel tiles differ from the data-parallel-only run"
        sk = ~dp
        err_sk = ((y.double() - ref)[sk].norm() / ref[sk].norm()).item()
        assert torch.isfinite(y).all() and err_sk < TOL and relf(y, ref) < TOL

    def test_concurrent_streams(self):
        """Two streams launching stream-K shapes at once with different inputs: every output bitwise equal to the
        serial one (per-stream counters, deterministic reduction)."""
        shapes = [(1024, 1920, 896), (128, 896, 4864), (2048, 3712, 1536), (512, 1152, 896)]
        for s in shapes:
            assert plan(*s)[2] > 0, f"{s} must use stream-K"
        iters, n_inputs = 50, 4
        for sa, sb in ((shapes[0], shapes[1]), (shapes[2], shapes[3]), (shapes[0], shapes[0])):
            ins = {s: [make(*shape, seed=100 * s + i) for i in range(n_inputs)] for s, shape in ((0, sa), (1, sb))}
            expected = {s: [ops().linear(*ins[s][i]) for i in range(n_inputs)] for s in (0, 1)}
            torch.cuda.synchronize()
            streams = [torch.cuda.Stream(), torch.cuda.Stream()]
            bad = [torch.zeros((), dtype=torch.int64, device=DEV) for _ in range(2)]
            for s in streams:
                s.wait_stream(torch.cuda.current_stream())
            for it in range(iters):
                for s in (0, 1):
                    with torch.cuda.stream(streams[s]):
                        i = (it * 7 + s) % n_inputs
                        bad[s] += (ops().linear(*ins[s][i]) != expected[s][i]).any().long()
            torch.cuda.synchronize()
            assert int(bad[0]) == 0 and int(bad[1]) == 0, f"{sa} || {sb}: mismatching outputs {bad}"

    def test_cuda_graph(self):
        """Capture of two chained fused_linear calls (stream-K shapes: per-call counters zeroed by a captured
        memset), replays with new inputs bitwise equal to eager; two graphs replayed on two streams while eager
        calls of the same shapes run on a third."""
        M, K, N1, N2 = 1024, 896, 1920, 1152  # y1 = x W1^T + b1 (data-parallel wave + stream-K), y2 = y1 W2^T
        x, W1, b1 = make(M, N1, K, 1)
        _, W2, _ = make(M, N2, N1, 2, bias=False)
        assert plan(M, N1, K)[2] > 0 and plan(M, N2, N1)[2] > 0

        def chain(inp):
            return fused_linear(fused_linear(inp, W1, b1), W2)

        static_x = x.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                chain(static_x)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_y = chain(static_x)
        for r in range(3):
            xr = make(M, N1, K, 10 + r)[0]
            static_x.copy_(xr)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(static_y, chain(xr)), f"replay {r} differs from eager"
            ref = ref64(ref64(xr, W1, b1).half(), W2)
            assert relf(static_y, ref) < 3e-3
        assert fallback_stats() == {}

        xa, xb = make(M, N1, K, 21)[0], make(M, N1, K, 22)[0]
        ga, gb = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        sxa, sxb = xa.clone(), xb.clone()
        with torch.cuda.graph(ga):
            ya = chain(sxa)
        with torch.cuda.graph(gb):
            yb = chain(sxb)
        ea, eb = chain(xa), chain(xb)
        torch.cuda.synchronize()
        sa, sb, se = torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream()
        for s in (sa, sb, se):
            s.wait_stream(torch.cuda.current_stream())
        bad = torch.zeros(3, dtype=torch.int64, device=DEV)
        for _ in range(20):
            with torch.cuda.stream(sa):
                ga.replay()
                bad[0] += (ya != ea).any().long()
            with torch.cuda.stream(sb):
                gb.replay()
                bad[1] += (yb != eb).any().long()
            with torch.cuda.stream(se):
                bad[2] += (chain(xa) != ea).any().long()
        torch.cuda.synchronize()
        assert int(bad.sum()) == 0, f"mismatches (graph a, graph b, eager) {bad.tolist()}"

    # The hog's blocks count themselves in mapped pinned host memory once resident: the host waits for all of them
    # before launching the linear (no fixed sleep)
    _HOG_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
static int* g_started = nullptr;  // mapped pinned host memory
__global__ void hog_kernel(long long cycles, int* started, int* out) {
    extern __shared__ int sm[];
    if (threadIdx.x == 0) {
        atomicAdd_system(started, 1);
        __threadfence_system();
    }
    const long long t0 = clock64();
    int v = 0;
    while (clock64() - t0 < cycles) v += sm[threadIdx.x & 31];
    if (v == 12345) out[0] = v;
}
void hog(int64_t blocks, int64_t cycles, torch::Tensor out) {
    if (g_started == nullptr) {
        TORCH_CHECK(cudaHostAlloc((void**)&g_started, sizeof(int), cudaHostAllocMapped) == cudaSuccess,
                    "cudaHostAlloc failed");
    }
    *(volatile int*)g_started = 0;
    int* started = nullptr;
    TORCH_CHECK(cudaHostGetDevicePointer((void**)&started, g_started, 0) == cudaSuccess,
                "cudaHostGetDevicePointer failed");
    TORCH_CHECK(cudaFuncSetAttribute(hog_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 48 * 1024) ==
                cudaSuccess, "cudaFuncSetAttribute failed");
    hog_kernel<<<(int)blocks, 32, 48 * 1024, at::cuda::getCurrentCUDAStream()>>>(cycles, started,
                                                                                 out.data_ptr<int>());
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "hog launch failed");
}
int64_t hog_started() { return g_started == nullptr ? 0 : *(volatile int*)g_started; }
"""

    def test_busy_sms(self):
        """A kernel holds every SM but one (48 KB of shared memory per block: no linear block fits beside it)
        for ~4 s on another stream: a stream-K linear launched once all its blocks are resident must complete
        before it ends (no block waits for another: forward progress on one SM). Polled with deadlines, never
        a hang of the test."""
        from torch.utils.cpp_extension import load_inline
        hog = load_inline("flash_attn_legacy_test_hog",
                          "void hog(int64_t blocks, int64_t cycles, torch::Tensor out); int64_t hog_started();",
                          cuda_sources=self._HOG_SRC, functions=["hog", "hog_started"],
                          extra_cuda_cflags=["-gencode=arch=compute_60,code=sm_60"])
        sms, hog_s = _sms(), 4.0
        out = torch.zeros(1, dtype=torch.int32, device=DEV)
        for M, N, K in ((128, 896, 4864), (1024, 1920, 896)):
            assert plan(M, N, K)[2] > 0
            x, W, b = make(M, N, K, 5)
            expected = ops().linear(x, W, b)
            torch.cuda.synchronize()
            hog_stream, gemm_stream = torch.cuda.Stream(), torch.cuda.Stream()
            hog_done, gemm_done = torch.cuda.Event(), torch.cuda.Event()
            with torch.cuda.stream(hog_stream):
                hog.hog(sms - 1, int(hog_s * 1.48e9), out)  # P100 boost clock: ~4 s or more
                hog_done.record()
            deadline = time.time() + 10
            while hog.hog_started() < sms - 1 and time.time() < deadline:
                time.sleep(0.001)
            assert hog.hog_started() == sms - 1, f"only {hog.hog_started()} of {sms - 1} hog blocks resident"
            with torch.cuda.stream(gemm_stream):
                y = ops().linear(x, W, b)
                gemm_done.record()
            deadline = time.time() + hog_s + 30
            while not gemm_done.query() and time.time() < deadline:
                time.sleep(0.001)
            finished, hog_running = gemm_done.query(), not hog_done.query()
            torch.cuda.synchronize()
            assert finished and hog_running, (f"M={M} N={N} K={K}: linear finished={finished}, "
                                              f"hog still running at that point={hog_running}")
            assert torch.equal(y, expected)

    @pytest.mark.parametrize("stream_k", [True, False], ids=["default", "data-parallel"])
    def test_lane_overflow(self, stream_k):
        """Each fp16 lane sums ~40000 per chunk (the two lanes' fp16 sum would overflow, the fp32 fold must not);
        every second chunk cancels 80% of the one before: y ~ 32000. Four chunks, so that the default plan splits K
        (stream-K) on the P100."""
        C = CHUNK
        wv = 40000.0 / (C // 2) / 25.0
        M, N, K = 256, 256, 4 * C
        if stream_k:
            assert plan(M, N, K)[2] > 0, "must use stream-K"
        x = torch.full((M, K), 25.0, device=DEV, dtype=torch.float16)
        W = torch.empty(N, K, device=DEV, dtype=torch.float16)
        for c in range(4):
            W[:, c * C:(c + 1) * C] = wv if c % 2 == 0 else -0.8 * wv
        with contextlib.ExitStack() as stack:
            if not stream_k:
                stack.enter_context(data_parallel_only())
            y = ops().linear(x, W)
        ref = ref64(x, W)
        scale = ref64(x.abs(), W.abs())  # sum |x_k w_k|: the chunk pairs cancel 80%
        assert torch.isfinite(y).all()
        assert ((y.double() - ref).abs() / scale).max().item() < 5e-3

    def test_empty_and_zero_k(self):
        b = torch.randn(64, device=DEV, dtype=torch.float16)
        x0, W0 = (torch.empty(n, 0, device=DEV, dtype=torch.float16) for n in (300, 64))
        assert torch.equal(ops().linear(x0, W0, b), b.expand(300, 64))
        assert torch.equal(ops().linear(x0, W0), torch.zeros(300, 64, device=DEV, dtype=torch.float16))
        x, W, b = make(512, 1152, 896, 3)
        assert ops().linear(x[:0], W, b).shape == (0, 1152)

    def test_misaligned_and_strided_inputs_copied(self):
        """The op itself copies non-contiguous / misaligned inputs (what reaches it under torch.compile)."""
        x, W, b = make(512, 1152, 896, 3)
        expected = ops().linear(x, W, b)
        big = torch.empty(512 * 896 + 8, device=DEV, dtype=torch.float16)
        x_odd = big[1:1 + 512 * 896].view(512, 896)
        x_odd.copy_(x)
        assert x_odd.data_ptr() % 16 != 0
        assert torch.equal(ops().linear(x_odd, W, b), expected)
        W_t = W.t().contiguous().t()
        assert torch.equal(ops().linear(x, W_t, b), expected)

    def test_rejects(self):
        x = torch.randn(4, 12, device=DEV, dtype=torch.float16)
        with pytest.raises(RuntimeError, match="multiple of 8"):
            ops().linear(x, x)
        xi = torch.randn(65536, 8, device=DEV, dtype=torch.float16)
        Wi = torch.randn(32768, 8, device=DEV, dtype=torch.float16)  # M * N = 2^31
        with pytest.raises(RuntimeError, match="int32"):
            ops().linear(xi, Wi)
        assert linear_mod._fallback_reason(xi, Wi, None) == "linear_int32"
        with pytest.raises(RuntimeError, match="float16"):
            ops().linear(xi.float(), Wi.float())
        if torch.cuda.device_count() > 1:
            x, W, b = make(512, 1152, 896, 3)
            with pytest.raises(RuntimeError, match="same CUDA device"):
                ops().linear(x, W.to("cuda:1"), b)
            with pytest.raises(RuntimeError, match="device of x"):
                ops().linear(x, W, b.to("cuda:1"))

    @compile_ready
    def test_opcheck(self):
        x, W, b = make(300, 1152, 896, 4)
        for args in ((x, W, b), (x, W, None), (x[:0], W, b)):
            torch.library.opcheck(ops().linear.default, args, test_utils=("test_schema", "test_faketensor"))
        # needs inputs that require gradients
        torch.library.opcheck(ops().linear.default, (x.clone().requires_grad_(), W.clone().requires_grad_(), b),
                              test_utils=("test_autograd_registration",))


# ---------------------------------------------------------------------------
# Autograd (register_autograd on the op: 2-D in and out, reshapes outside)
# ---------------------------------------------------------------------------

@sm60_only
@pytest.mark.skipif(not hasattr(torch.library, "register_autograd"), reason="torch >= 2.4 required")
class TestAutograd:

    @staticmethod
    def _run(fn, x0, W0, b0, gy, dtype, req_x=True, inplace=False, mask=None):
        x = x0.detach().clone().to(dtype).requires_grad_(req_x)
        W = W0.detach().clone().to(dtype).requires_grad_(True)
        b = b0.detach().clone().to(dtype).requires_grad_(True) if b0 is not None else None
        y = fn(x, W, b)
        if mask is not None:
            y = y * mask
        elif inplace:
            y.relu_()
        y.backward(gy.to(dtype))
        return y, x.grad, W.grad, (b.grad if b is not None else None)

    @pytest.mark.parametrize("x_shape,N,has_bias,req_x,kernel_calls", [
        ((4, 300, 896), 1152, True, True, 2),   # every gradient, dX on the kernel
        ((2, 600, 1536), 1004, True, True, 1),  # N % 8 != 0: stock dX
        ((1, 64, 896), 1152, True, True, 0),    # M < 128: F.linear
        ((3, 256, 896), 896, False, False, 1),  # no bias, W only (x saved, W not)
    ], ids=["kernel-dX", "stock-dX", "small-M", "W-only"])
    @pytest.mark.parametrize("inplace", [False, True], ids=["plain", "relu_"])
    def test_grads_vs_fp32(self, monkeypatch, x_shape, N, has_bias, req_x, kernel_calls, inplace):
        calls = []

        class CountingOps:  # the op calls of fused_linear and of its backward (dX)
            @staticmethod
            def linear(*args):
                calls.append(args[0].shape)
                return ops().linear(*args)

        monkeypatch.setattr(linear_mod, "_ops", CountingOps)
        torch.manual_seed(0)
        K = x_shape[-1]
        x0 = torch.randn(*x_shape, device=DEV, dtype=torch.float16)
        W0 = (torch.randn(N, K, device=DEV, dtype=torch.float16) * K ** -0.5).half()
        b0 = torch.randn(N, device=DEV, dtype=torch.float16) if has_bias else None
        gy = torch.randn(*x_shape[:-1], N, device=DEV, dtype=torch.float16)
        got = self._run(fused_linear, x0, W0, b0, gy, torch.float16, req_x, inplace)
        # relu_ on the reference: the fp16 output's mask, since the outputs near 0 flip sign between fp16 and fp32
        # (stock F.linear in fp16 is 7e-3 off a relu'd fp32 reference, 2e-4 off one with the same mask)
        mask = (got[0] > 0).float() if inplace else None
        ref = self._run(F.linear, x0, W0, b0, gy, torch.float32, req_x, inplace, mask)
        assert len(calls) == kernel_calls, calls
        for name, g, r in zip(("y", "dX", "dW", "db"), got, ref):
            if r is None:
                assert g is None, name
                continue
            assert g.dtype == torch.float16 and torch.isfinite(g).all(), name
            err = ((g.float() - r).norm() / r.norm()).item()
            assert err < 3e-3, f"{name}: relF vs fp32 {err:.2e}"

    @pytest.mark.parametrize("mode", ["no_grad", "inference_mode"])
    def test_no_grad_modes(self, mode):
        x, W, b = make(300, 1152, 896, 6)
        W.requires_grad_(True)
        expected = ops().linear(x, W.detach(), b)
        with getattr(torch, mode)():
            y = fused_linear(x.view(3, 100, 896), W, b)
        assert y.grad_fn is None and y.shape == (3, 100, 1152)
        assert torch.equal(y.reshape(300, 1152), expected)

    def test_double_backward_raises(self):
        x, W, b = make(300, 1152, 896, 7)
        x.requires_grad_(True)
        y = fused_linear(x, W, b)
        with pytest.raises(RuntimeError, match="double backward"):
            torch.autograd.grad(y.float().square().sum(), x, create_graph=True)


# ---------------------------------------------------------------------------
# torch.compile (fullgraph: no graph break on the kernel path nor on the fallbacks)
# ---------------------------------------------------------------------------

@compile_ready
class TestCompile:

    @staticmethod
    def _compiled(fn):
        torch._dynamo.reset()
        linear_mod._SM60.clear()  # the compute capability read while tracing (a folded constant)
        return torch.compile(fn, backend="aot_eager", fullgraph=True)

    def test_fwd_bwd(self):
        """Kernel path (M = 600, in-place on the output) and decode-size fallback (M = 5) in one compiled graph,
        forward and backward equal to eager. On GPUs other than SM 6.0 both are fallbacks."""
        torch.manual_seed(0)
        x = torch.randn(2, 300, 896, device=DEV, dtype=torch.float16, requires_grad=True)
        xs = torch.randn(1, 5, 896, device=DEV, dtype=torch.float16, requires_grad=True)
        W = (torch.randn(1152, 896, device=DEV, dtype=torch.float16) * 896 ** -0.5).requires_grad_(True)
        b = torch.randn(1152, device=DEV, dtype=torch.float16, requires_grad=True)

        def step(x, xs, W, b):
            y = fused_linear(x, W, b)
            y.relu_()
            return y.float().square().sum() + fused_linear(xs, W, b).float().sum()

        results = []
        for fn in (step, self._compiled(step)):
            loss = fn(x, xs, W, b)
            results.append((loss, *torch.autograd.grad(loss, (x, xs, W, b))))
        for a, b_ in zip(*results):
            torch.testing.assert_close(a, b_, atol=1e-3, rtol=1e-3)

    def test_inference(self):
        x, W, b = (torch.randn(s, device=DEV, dtype=torch.float16) for s in ((4, 64, 256), (512, 256), (512,)))
        with torch.inference_mode():
            expected = fused_linear(x, W, b)
            got = self._compiled(lambda x, W, b: fused_linear(x, W, b))(x, W, b)
        torch.testing.assert_close(expected, got, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Fallbacks: bitwise F.linear, the reason in fallback_stats()
# ---------------------------------------------------------------------------

class _Sub(torch.Tensor):
    pass


def _base():
    x, W, b = make(512, 1152, 896, 3)
    return x, W, b


def _misaligned():
    x, W, b = _base()
    big = torch.randn(512 * 896 + 8, device=DEV, dtype=torch.float16)
    return big[1:1 + 512 * 896].view(512, 896), W, b


# name -> (inputs, reason, decided before the compute-capability check)
_FALLBACKS = {
    "K%8": (lambda: tuple(t[..., :12].contiguous() if t.dim() == 2 else t for t in _base()), "linear_shape", False),
    "K==0": (lambda: (torch.empty(300, 0, device=DEV, dtype=torch.float16),
                      torch.empty(64, 0, device=DEV, dtype=torch.float16),
                      torch.randn(64, device=DEV, dtype=torch.float16)), "linear_shape", False),
    "small-M": (lambda: (_base()[0][:64], *_base()[1:]), "linear_small_m", True),
    "1-D x": (lambda: (_base()[0][0], *_base()[1:]), "linear_small_m", True),
    "x-non-contiguous": (lambda: (torch.randn(512, 1024, device=DEV, dtype=torch.float16)[:, :896], *_base()[1:]),
                         "linear_layout", False),
    "W-non-contiguous": (lambda: (_base()[0], torch.randn(896, 1152, device=DEV, dtype=torch.float16).t(),
                                  _base()[2]), "linear_layout", False),
    "x-misaligned": (_misaligned, "linear_layout", False),
    "subclass": (lambda: (_base()[0].as_subclass(_Sub), *_base()[1:]), "linear_layout", True),
    "fp32": (lambda: tuple(t.float() for t in _base()), "linear_dtype", True),
    "cpu": (lambda: tuple(t.cpu().float() for t in _base()), "linear_device", True),
}


@pytest.mark.parametrize("name", list(_FALLBACKS))
def test_fallback(name):
    make_inputs, reason, before_arch = _FALLBACKS[name]
    x, W, b = make_inputs()
    want = reason if (_is_sm60() or before_arch) else "linear_arch"
    assert linear_mod._fallback_reason(x, W, b) == want
    y = fused_linear(x, W, b)
    assert type(y) is type(F.linear(x, W, b)) and torch.equal(y, F.linear(x, W, b))
    assert fallback_stats() == ({} if want == "linear_small_m" else {want: 1})  # decode: by design, not counted


def test_fallback_autocast_other_dtype():
    x, W, b = _base()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if not torch.is_autocast_enabled():
            pytest.skip("bf16 autocast unavailable on this GPU")
        y, expected = fused_linear(x, W, b), F.linear(x, W, b)
    assert y.dtype == expected.dtype == torch.bfloat16 and torch.equal(y, expected)
    assert fallback_stats() == {"linear_dtype": 1}


def test_eligible_input():
    """Control: the kernel runs (no fallback, a result other than cuBLAS's) on SM 6.0 only, also under fp16
    autocast; elsewhere the fallback is the compute capability."""
    x, W, b = _base()
    y = fused_linear(x, W, b)
    with torch.autocast("cuda", dtype=torch.float16):
        y_autocast = fused_linear(x, W, b)
    if _is_sm60():
        assert fallback_stats() == {} and not torch.equal(y, F.linear(x, W, b)) and torch.equal(y, y_autocast)
        assert relf(y, ref64(x, W, b)) < TOL
    else:
        assert fallback_stats() == {"linear_arch": 2} and torch.equal(y, F.linear(x, W, b))


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_fallback_device_mismatch():
    x, W, b = _base()
    for args in ((x, W.to("cuda:1"), b), (x, W, b.to("cuda:1"))):
        assert linear_mod._fallback_reason(*args) == "linear_device"
    assert fallback_stats() == {}


# ---------------------------------------------------------------------------
# patch_linear and the FLASH_ATTN_LEGACY_LINEAR flag
# ---------------------------------------------------------------------------

class _Doubled(nn.Linear):
    def forward(self, x):
        return super().forward(x) * 2


def _tiny_model(dtype=torch.float16):
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 256, bias=False),
                         _Doubled(256, 256)).to(DEV, dtype)


def _patched(m):
    return type(m) is linear_mod._FusedLinear


def test_patch_linear():
    model = _tiny_model()
    x = torch.randn(2, 200, 256, device=DEV, dtype=torch.float16)
    with torch.no_grad():
        stock = model(x)
        n = 2 if _is_sm60() else 0  # _Doubled overrides forward: left alone
        assert patch_linear(model) == n and patch_linear(model) == n  # idempotent
        assert [_patched(m) for m in model] == [_is_sm60(), False, _is_sm60(), False]
        assert all("forward" not in m.__dict__ for m in model)
        patched = model(x)
        assert fallback_stats() == {}
        if _is_sm60():
            assert relf(patched, stock.double()) < 5e-3
            clone = copy.deepcopy(model)
            assert _patched(clone[0]) and torch.equal(clone(x), patched)
        assert patch_linear(model, enable=False) == 0
        assert all(type(m) is not linear_mod._FusedLinear for m in model) and type(model[0]) is nn.Linear
        assert torch.equal(model(x), stock)
    assert patch_linear(_tiny_model(torch.float32)) == 0


def test_patch_linear_skips():
    """Left alone: a Linear with an instance forward (accelerate's hooks), a subclass, a weight that is not a
    plain nn.Parameter."""
    hooked, sub, odd = (nn.Linear(256, 256), _Doubled(256, 256), nn.Linear(256, 256))
    model = nn.Sequential(hooked, sub, odd).to(DEV, torch.float16)
    original = hooked.forward
    hooked.forward = lambda x: original(x)  # what accelerate's add_hook_to_module does, in short
    odd.weight = nn.Parameter(odd.weight.detach().as_subclass(_Sub), requires_grad=False)
    assert patch_linear(model) == 0
    assert type(hooked) is nn.Linear and type(sub) is _Doubled and type(odd) is nn.Linear


@compile_ready
def test_patched_model_compiled():
    """A patched model under torch.compile(fullgraph=True): the class-swapped forward traces like any module
    (on GPUs other than SM 6.0 nothing is patched: the stock model)."""
    model = _tiny_model()
    patch_linear(model)
    x = torch.randn(2, 200, 256, device=DEV, dtype=torch.float16)
    with torch.no_grad():
        expected = model(x)
        torch._dynamo.reset()
        got = torch.compile(model, backend="aot_eager", fullgraph=True)(x)
    torch.testing.assert_close(got, expected, atol=0, rtol=0)


def test_env_flag(monkeypatch):
    """FLASH_ATTN_LEGACY_LINEAR=1: an HF model built with attn_implementation="flash_attn_legacy" gets
    patch_linear at its first forward (once moved to the GPU in fp16); not without the flag."""
    pytest.importorskip("transformers")
    from .hf_helpers import tiny_llama

    def linears(model):
        mods = [m for m in model.modules() if type(m) in (nn.Linear, linear_mod._FusedLinear)]
        return len(mods), sum(_patched(m) for m in mods)

    ids = torch.randint(0, 512, (1, 200), device=DEV)
    for flag in ("1", None):
        if flag is None:
            monkeypatch.delenv("FLASH_ATTN_LEGACY_LINEAR", raising=False)
        else:
            monkeypatch.setenv("FLASH_ATTN_LEGACY_LINEAR", flag)
        model = tiny_llama(attn_implementation="flash_attn_legacy").to(DEV, torch.float16)
        assert model.config._attn_implementation == "flash_attn_legacy"
        assert linears(model)[1] == 0  # nothing before the first forward
        with torch.no_grad():
            model(input_ids=ids)
            n, patched = linears(model)
            assert patched == (n if flag and _is_sm60() else 0)
            assert not any(h is flash_attn_legacy.integrations._linear_pre_hook
                           for h in model._forward_pre_hooks.values())
            model(input_ids=ids)  # the hook is gone: no second patch, no error
