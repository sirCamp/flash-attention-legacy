"""
Tests for the depthwise causal conv1d of Gated DeltaNet layers (csrc/kernels/causal_conv1d.cu,
torch.ops.flash_attn_legacy.causal_conv1d*, flash_attn_legacy.gated_delta.patch_conv1d; docs/PLAN.md P6.10 b):
forward, backward and the decode update against fp64, layouts and dtypes, the ops' registration, and the Qwen3.5
integration. No architecture-specific code: every GPU runs every test.

Run:  pytest tests/test_causal_conv1d.py -v
"""

import sys
import types

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

flash_attn_legacy = pytest.importorskip("flash_attn_legacy")
mq = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
hub_kernels = pytest.importorskip("transformers.integrations.hub_kernels")

from flash_attn_legacy import fallback_stats, gated_delta, reset_fallback_stats  # noqa: E402

DEV = "cuda"
compile_ready = pytest.mark.skipif(not hasattr(torch.library, "opcheck"), reason="torch >= 2.4 required")
_EPS = {torch.float16: 2.0 ** -11, torch.bfloat16: 2.0 ** -8, torch.float32: 2.0 ** -24}


@pytest.fixture(autouse=True)
def _clean_fallback_stats():
    reset_fallback_stats()
    yield
    reset_fallback_stats()


def ops():
    return torch.ops.flash_attn_legacy


def channel_last(B, C, L, dtype, seed, pad=0, offset=0):
    """x [B, C, L] as transformers passes it (mixed_qkv [B, L, C] transposed); pad: a slice of a wider [B, L, C +
    pad] projection; offset: a view that starts `offset` elements into its storage (misaligned)."""
    g = torch.Generator(DEV).manual_seed(seed)
    base = torch.randn(B * L * (C + pad) + offset, device=DEV, generator=g).to(dtype)
    return base[offset:].view(B, L, C + pad)[..., :C].transpose(1, 2)


def params(C, W, dtype, seed, bias=True):
    g = torch.Generator(DEV).manual_seed(seed + 1)
    w = (torch.randn(C, W, device=DEV, generator=g) * 0.5).to(dtype)
    b = (torch.randn(C, device=DEV, generator=g) * 0.1).to(dtype) if bias else None
    return w, b


def ref64(x, w, b, silu):
    """fp64 reference: transformers' causal_conv1d_fn (F.conv1d, padding W-1, the first L outputs)."""
    L, W = x.shape[2], w.shape[1]
    y = F.conv1d(x.double(), w.double().unsqueeze(1), None if b is None else b.double(), padding=W - 1,
                 groups=x.shape[1])[..., :L]
    return F.silu(y) if silu else y


def ref_impl(x, w, b, silu):
    """transformers' reference in the inputs' dtype (two roundings: the conv, then SiLU)."""
    L, W = x.shape[2], w.shape[1]
    y = F.conv1d(x.to(w.dtype), w.unsqueeze(1), b, padding=W - 1, groups=x.shape[1])[..., :L]
    return (F.silu(y) if silu else y).to(x.dtype)


def relf(a, ref):
    return ((a.double() - ref).norm() / ref.norm().clamp_min(1e-30)).item()


def one_rounding_ok(y, ref, dtype):
    """Within one rounding of the fp64 value, plus fp32 accumulation (a few fp32 ulps, absolute near 0)."""
    err = (y.double() - ref).abs()
    return bool((err <= _EPS[dtype] * ref.abs() + 1e-6 + 4 * 2.0 ** -24 * ref.abs()).all())


_SHAPES = [(1, 8, 1), (3, 8, 2), (1, 64, 3), (2, 64, 4), (1, 72, 5), (2, 136, 127), (1, 6144, 300), (2, 6150, 70)]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=["fp16", "bf16", "fp32"])
@pytest.mark.parametrize("W", [2, 3, 4])
@pytest.mark.parametrize("w_fp32", [False, True], ids=["w-x-dtype", "w-fp32"])
def test_forward(dtype, W, w_fp32):
    """Every shape (L < W included, C not a multiple of 8: the scalar path), with and without bias and SiLU: within
    one rounding of fp64 and no worse than transformers' reference; the output channel-last."""
    for i, (B, C, L) in enumerate(_SHAPES):
        for bias, silu in ((True, True), (False, False), (True, False)):
            x = channel_last(B, C, L, dtype, seed=i)
            w, b = params(C, W, torch.float32 if w_fp32 else dtype, seed=i, bias=bias)
            y = ops().causal_conv1d(x, w, b, silu)
            ref = ref64(x, w, b, silu)
            label = f"B={B} C={C} L={L} bias={bias} silu={silu}"
            assert y.shape == x.shape and y.dtype == dtype and y.stride(1) == 1, label
            assert one_rounding_ok(y, ref, dtype), f"{label}: max err {(y.double() - ref).abs().max().item():.2e}"
            assert relf(y, ref) <= relf(ref_impl(x, w, b, silu), ref) * 1.05 + 1e-7, label


@pytest.mark.parametrize("layout", ["slice", "misaligned", "contiguous-copy"])
def test_layouts(layout):
    """A slice of a wider projection (token stride > C), a misaligned view (scalar loads), and a contiguous [B, C, L]
    tensor's channel-last copy: the same values as the plain channel-last input."""
    B, C, L = 2, 64, 50
    plain = channel_last(B, C, L, torch.float16, seed=3)
    if layout == "slice":
        x = channel_last(B, C, L, torch.float16, seed=9, pad=16)
    elif layout == "misaligned":
        x = channel_last(B, C, L, torch.float16, seed=9, offset=1)
    else:
        x = plain.contiguous().transpose(1, 2).contiguous().transpose(1, 2)
    x.copy_(plain)
    w, b = params(C, 4, torch.float16, seed=3)
    assert torch.equal(ops().causal_conv1d(x, w, b, True), ops().causal_conv1d(plain, w, b, True))


def test_rejects():
    x = channel_last(1, 8, 4, torch.float16, seed=0)
    w, b = params(8, 4, torch.float16, seed=0)
    with pytest.raises(RuntimeError, match="channel-last"):
        ops().causal_conv1d(x.contiguous(), w, b, False)
    with pytest.raises(RuntimeError, match="W = 2..4"):
        ops().causal_conv1d(x, torch.randn(8, 5, device=DEV, dtype=torch.float16), None, False)
    with pytest.raises(RuntimeError, match="x's dtype or float32"):
        ops().causal_conv1d(x, w.bfloat16(), None, False)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=["fp16", "bf16", "fp32"])
@pytest.mark.parametrize("W", [2, 3, 4])
@pytest.mark.parametrize("w_fp32", [False, True], ids=["w-x-dtype", "w-fp32"])
def test_backward(dtype, W, w_fp32):
    """dX, dW, db against fp64 autograd, no worse than transformers' reference's (with a small slack: dW / db sum
    over B * L in fp32 in another order); dW / db in the weight's dtype (fp32 weights: the AMP case), bitwise
    reproducible."""
    wdt = torch.float32 if w_fp32 else dtype
    for i, (B, C, L) in enumerate([(2, 64, 3), (1, 136, 127), (2, 6144, 200), (1, 70, 65)]):
        for silu in (True, False):
            x0 = channel_last(B, C, L, dtype, seed=10 + i)
            w0, b0 = params(C, W, wdt, seed=10 + i)
            gy = torch.randn(B, L, C, device=DEV, dtype=dtype, generator=torch.Generator(DEV).manual_seed(i)
                             ).transpose(1, 2)

            def grads(fn, cast=None):
                x = (x0 if cast is None else x0.to(cast)).detach().requires_grad_(True)
                w = (w0 if cast is None else w0.to(cast)).detach().requires_grad_(True)
                b = (b0 if cast is None else b0.to(cast)).detach().requires_grad_(True)
                fn(x, w, b).backward(gy if cast is None else gy.to(cast))
                return x.grad, w.grad, b.grad

            got = grads(lambda x, w, b: ops().causal_conv1d(x, w, b, silu))
            ref = grads(lambda x, w, b: ref64(x, w, b, silu), torch.float64)
            stock = grads(lambda x, w, b: ref_impl(x, w, b, silu))
            for name, g, r, s in zip(("dx", "dW", "db"), got, ref, stock):
                assert g.dtype == (dtype if name == "dx" else wdt) and torch.isfinite(g).all(), name
                assert relf(g, r) <= relf(s, r) * 1.1 + 4 * _EPS[dtype], f"{name} B={B} C={C} L={L} silu={silu}"
            again = grads(lambda x, w, b: ops().causal_conv1d(x, w, b, silu))
            assert torch.equal(got[1], again[1]) and torch.equal(got[2], again[2]), "dW / db not reproducible"


@pytest.mark.parametrize("trainable", ["x", "bias", "weight"])
def test_backward_asks_for_dw_only_when_needed(monkeypatch, trainable):
    """The dW / db pass runs only when the weight or the bias needs a gradient (a frozen conv under LoRA: dX only)."""
    flags = []
    real = gated_delta._ops

    class Recorder:
        causal_conv1d = real.causal_conv1d

        @staticmethod
        def causal_conv1d_backward(*args):
            flags.append(args[-1])
            return real.causal_conv1d_backward(*args)

    monkeypatch.setattr(gated_delta, "_ops", Recorder)
    x = channel_last(1, 64, 20, torch.float16, seed=0).requires_grad_(True)
    w, b = params(64, 4, torch.float16, seed=0)
    w.requires_grad_(trainable == "weight")
    b.requires_grad_(trainable == "bias")
    ops().causal_conv1d(x, w, b, True).float().sum().backward()
    assert flags == [trainable != "x"]
    assert x.grad is not None and (w.grad is not None) == (trainable == "weight")
    assert (b.grad is not None) == (trainable == "bias")


@pytest.mark.parametrize("state_dtype", [torch.float16, torch.bfloat16, torch.float32],
                         ids=["state-fp16", "state-bf16", "state-fp32"])
@pytest.mark.parametrize("S,state_len", [(1, 4), (3, 4), (1, 3), (2, 6)])
def test_update(state_dtype, S, state_len):
    """A decode step on a conv state: the outputs those of the forward on the state then the new tokens (bitwise in
    fp32), the state shifted in place: bitwise transformers' causal_conv1d_update when the state has x's dtype, else
    the exact shift (the reference passes the old state through the weight's dtype: an fp32 state of an fp16 model
    is rounded to fp16 there, not here)."""
    reference = gated_delta._decorator_cells(mq.causal_conv1d_update)["torch_function"]
    for dtype, wdt, bias in ((torch.float16, torch.float16, True), (torch.float32, torch.float32, True),
                             (torch.bfloat16, torch.float32, False)):
        B, C, W = 2, 136, 4
        x = channel_last(B, C, S, dtype, seed=20)
        w, b = params(C, W, wdt, seed=20, bias=bias)
        state = torch.randn(B, C, state_len, device=DEV, generator=torch.Generator(DEV).manual_seed(21)
                            ).to(state_dtype)
        old, state_ref, ptr = state.clone(), state.clone(), state.data_ptr()
        y = ops().causal_conv1d_update(x, state, w, b, True)
        y_ref = reference(x, state_ref, w, b, "silu")
        assert y.dtype == dtype and y.shape == x.shape
        shifted = torch.cat([old, x.to(state_dtype)], -1)[..., -state_len:]
        assert state.data_ptr() == ptr and torch.equal(state, shifted), f"{dtype}: state not the exact shift"
        if state_dtype == dtype == wdt:
            assert torch.equal(state, state_ref), f"{dtype}: state differs from transformers'"
        joined = torch.cat([old.to(dtype), x], -1).transpose(1, 2).contiguous().transpose(1, 2)
        expected = ops().causal_conv1d(joined, w, b, True)[..., -S:]
        if dtype == torch.float32 and state_dtype == torch.float32:
            assert torch.equal(y, expected), "fp32: not the forward's bits"
        else:
            assert relf(y, expected.double()) < 4 * _EPS[dtype]
        assert relf(y, y_ref.double()) < 8 * _EPS[dtype] + 1e-6


def test_update_cuda_graph():
    """A captured decode step replays with new tokens: the state and the outputs as eager."""
    B, C, W = 2, 64, 4
    w, b = params(C, W, torch.float16, seed=30)
    static_x = channel_last(B, C, 1, torch.float16, seed=31)
    state = torch.zeros(B, C, 4, device=DEV, dtype=torch.float16)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        ops().causal_conv1d_update(static_x, state.clone(), w, b, True)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_y = ops().causal_conv1d_update(static_x, state, w, b, True)
    eager_state = torch.zeros_like(state)
    state.zero_()
    for r in range(3):
        xr = channel_last(B, C, 1, torch.float16, seed=40 + r)
        static_x.copy_(xr)
        graph.replay()
        expected = ops().causal_conv1d_update(xr, eager_state, w, b, True)
        torch.cuda.synchronize()
        assert torch.equal(static_y, expected) and torch.equal(state, eager_state), f"replay {r}"


@compile_ready
def test_opcheck():
    x = channel_last(2, 64, 20, torch.float16, seed=50)
    w, b = params(64, 4, torch.float16, seed=50)
    for args in ((x, w, b, True), (x, w, None, False), (x, w.float(), b.float(), True)):
        torch.library.opcheck(ops().causal_conv1d.default, args, test_utils=("test_schema", "test_faketensor"))
    torch.library.opcheck(ops().causal_conv1d.default,
                          (x.detach().clone().requires_grad_(), w.clone().requires_grad_(), b, True),
                          test_utils=("test_autograd_registration",))
    gy = torch.randn_like(x)
    for need_dw in (True, False):
        torch.library.opcheck(ops().causal_conv1d_backward.default, (gy, x, w, b, True, need_dw),
                              test_utils=("test_schema", "test_faketensor"))
    state = torch.randn(2, 64, 4, device=DEV, dtype=torch.float16)
    torch.library.opcheck(ops().causal_conv1d_update.default, (x[..., :1], state, w, b, True),
                          test_utils=("test_schema", "test_faketensor"))


@compile_ready
def test_compiled():
    x = channel_last(2, 64, 20, torch.float16, seed=60).requires_grad_(True)
    w, b = params(64, 4, torch.float16, seed=60)
    w.requires_grad_(True)

    def f(x, w, b):
        return ops().causal_conv1d(x, w, b, True).float().square().sum()

    results = []
    for fn in (f, torch.compile(f, backend="aot_eager", fullgraph=True)):
        loss = fn(x, w, b)
        results.append((loss, *torch.autograd.grad(loss, (x, w))))
    for a, e in zip(*results):
        torch.testing.assert_close(a, e, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Qwen3.5 (tiny random model, no download)
# ---------------------------------------------------------------------------

def _restore_globals(monkeypatch, module):
    """The module's conv / delta-rule globals restored at teardown (the patch is process-wide)."""
    for n in ("causal_conv1d_fn", "causal_conv1d_update", "torch_chunk_gated_delta_rule"):
        monkeypatch.setattr(module, n, getattr(module, n))
    return module


def _qwen35_module(monkeypatch):
    return _restore_globals(monkeypatch, mq)


def test_qwen35_patched(monkeypatch):
    """A model on our attention gets the kernels: prefill logits within fp16 noise of the reference path, greedy
    decode the same until a near-tie, a training step's gradients close; FLASH_ATTN_LEGACY_CONV1D=0 opts out."""
    from .hf_helpers import tiny_qwen35
    _qwen35_module(monkeypatch)
    originals = (mq.causal_conv1d_fn, mq.causal_conv1d_update)
    model = tiny_qwen35(attn_implementation="flash_attn_legacy").to(DEV, torch.float16)
    assert getattr(mq.causal_conv1d_fn, gated_delta._GUARDED, False)
    ids = torch.randint(0, 512, (2, 40), device=DEV, generator=torch.Generator(DEV).manual_seed(2))

    def run():
        with torch.inference_mode():
            logits = model(input_ids=ids).logits.float()
            gen = model.generate(ids, max_new_tokens=64, min_new_tokens=64, do_sample=False, output_logits=True,
                                 return_dict_in_generate=True)
        model.zero_grad(set_to_none=True)
        model(input_ids=ids, labels=ids).loss.backward()
        grads = {n: p.grad.float().clone() for n, p in model.named_parameters() if "conv1d" in n or "in_proj_qkv" in n}
        return logits, gen, grads

    ours = run()
    monkeypatch.setattr(mq, "causal_conv1d_fn", originals[0])
    monkeypatch.setattr(mq, "causal_conv1d_update", originals[1])
    stock = run()
    assert fallback_stats().get("conv1d_input", 0) == 0
    assert (ours[0] - stock[0]).abs().max().item() < 2e-2
    live = torch.ones(ids.shape[0], dtype=torch.bool, device=DEV)
    for t, (a, b) in enumerate(zip(ours[1].logits, stock[1].logits)):
        assert ((a.float() - b.float()).abs().amax(-1)[live] < 2e-2).all(), f"decode step {t}"
        parted = live & (ours[1].sequences[:, ids.shape[1] + t] != stock[1].sequences[:, ids.shape[1] + t])
        top2 = b.float().topk(2, -1).values
        assert ((top2[:, 0] - top2[:, 1])[parted] < 0.05).all(), f"step {t}: tokens part away from a near-tie"
        live &= ~parted
    for n, g in ours[2].items():
        assert relf(g, stock[2][n].double()) < 2e-2, n

    monkeypatch.setenv("FLASH_ATTN_LEGACY_CONV1D", "0")
    assert not gated_delta.patch_conv1d(tiny_qwen35(attn_implementation="flash_attn_legacy"))
    assert mq.causal_conv1d_fn is originals[0]


def test_qwen35_fallback_counted(monkeypatch):
    """An input the kernel does not take (a CPU tensor here) goes to transformers' reference, counted."""
    _qwen35_module(monkeypatch)
    from .hf_helpers import tiny_qwen35
    tiny_qwen35(attn_implementation="flash_attn_legacy")
    x = torch.randn(1, 8, 5)
    w = torch.randn(8, 4)
    y = mq.causal_conv1d_fn(x, w, None, activation="silu")
    assert y.shape == x.shape and fallback_stats() == {"conv1d_input": 1}


def test_packed_sequences_raise(monkeypatch):
    from .hf_helpers import tiny_qwen35
    _qwen35_module(monkeypatch)
    tiny_qwen35(attn_implementation="flash_attn_legacy")
    x = channel_last(1, 64, 8, torch.float16, seed=0)
    w, b = params(64, 4, torch.float16, seed=0)
    with pytest.raises(ValueError, match="packed sequences"):
        mq.causal_conv1d_fn(x, w, b, activation="silu", seq_idx=torch.zeros(1, 8, dtype=torch.int32, device=DEV))


def test_gradient_checkpointing_and_padding(monkeypatch):
    """A padded batch (left padding, the mask zeroes the pad tokens before the conv) and gradient checkpointing:
    the same as the reference conv within fp16 noise."""
    from .hf_helpers import tiny_qwen35
    _qwen35_module(monkeypatch)
    originals = (mq.causal_conv1d_fn, mq.causal_conv1d_update)
    model = tiny_qwen35(attn_implementation="flash_attn_legacy").to(DEV, torch.float16)
    model.gradient_checkpointing_enable()
    model.train()
    ids = torch.randint(0, 512, (2, 32), device=DEV, generator=torch.Generator(DEV).manual_seed(5))
    mask = torch.ones_like(ids)
    mask[1, :7] = 0

    def run():
        model.zero_grad(set_to_none=True)
        out = model(input_ids=ids, attention_mask=mask, labels=ids)
        out.loss.backward()
        return out.logits.float()[mask.bool()], {n: p.grad.float().clone() for n, p in model.named_parameters()
                                                 if p.grad is not None and "conv1d" in n}

    ours = run()
    monkeypatch.setattr(mq, "causal_conv1d_fn", originals[0])
    monkeypatch.setattr(mq, "causal_conv1d_update", originals[1])
    stock = run()
    assert ours[1] and (ours[0] - stock[0]).abs().max().item() < 2e-2
    for n, g in ours[1].items():
        assert relf(g, stock[1][n].double()) < 2e-2, n


@pytest.mark.parametrize("module_name", ["qwen3_next", "olmo_hybrid", "qwen3_5_moe"])
def test_other_hybrid_models(monkeypatch, module_name):
    """Qwen3-Next, OLMo-hybrid and Qwen3.5-MoE call the same globals (positional, or weight= / bias= / activation=
    keywords): patched through a model class of theirs, the wrappers take both call styles and match transformers'
    reference."""
    module = pytest.importorskip(f"transformers.models.{module_name}.modeling_{module_name}")
    _restore_globals(monkeypatch, module)
    model = type("HybridModel", (torch.nn.Module,), {"__module__": module.__name__})()
    assert gated_delta.patch_conv1d(model)
    x = channel_last(2, 64, 9, torch.float16, seed=7)
    w, b = params(64, 4, torch.float16, seed=7)
    y_pos = module.causal_conv1d_fn(x, w, b, activation="silu")
    y_kw = module.causal_conv1d_fn(x, weight=w, bias=b, activation="silu")
    assert torch.equal(y_pos, y_kw) and torch.equal(y_pos, ops().causal_conv1d(x, w, b, True))
    state = torch.randn(2, 64, 4, device=DEV, dtype=torch.float16)
    s2 = state.clone()
    u_pos = module.causal_conv1d_update(x[..., :1], state, w, b, "silu")
    u_kw = module.causal_conv1d_update(x[..., :1], s2, weight=w, bias=b, activation="silu")
    assert torch.equal(u_pos, u_kw) and torch.equal(state, s2)
    assert fallback_stats() == {}


def test_package_kept_where_it_has_code(monkeypatch):
    """A causal-conv1d bound by transformers stays on SM 7.5 and newer (its wheels' archs); below, our kernels run."""
    calls = []

    def package_fn(x, weight, bias=None, activation=None, **kwargs):
        calls.append("package")
        return x

    pkg = types.ModuleType("_fal_stub_conv_pkg")
    pkg.stub_conv = package_fn
    sys.modules[pkg.__name__] = pkg
    try:
        bound = hub_kernels.use_kernel_func_from_hub_with_fallback("stub_conv", pkg.__name__)(
            gated_delta._decorator_cells(mq.causal_conv1d_fn)["torch_function"])
    finally:
        del sys.modules[pkg.__name__]
    module = types.ModuleType("_fal_stub_conv_modeling")
    module.causal_conv1d_fn, module.causal_conv1d_update = bound, mq.causal_conv1d_update
    monkeypatch.setitem(sys.modules, module.__name__, module)
    model = type("StubHybrid", (torch.nn.Module,), {"__module__": module.__name__})()
    assert gated_delta.patch_conv1d(model)
    x = channel_last(1, 64, 8, torch.float16, seed=0)
    w, b = params(64, 4, torch.float16, seed=0)
    for cc, expected in (((7, 5), ["package"]), ((7, 0), [])):
        calls.clear()
        monkeypatch.setattr(gated_delta, "device_capability", lambda index, cc=cc: cc)
        y = module.causal_conv1d_fn(x, w, b, activation="silu")
        assert calls == expected, cc
        if not expected:
            assert torch.equal(y, ops().causal_conv1d(x, w, b, True))
