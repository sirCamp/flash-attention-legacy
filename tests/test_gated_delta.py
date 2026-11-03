"""
Tests for the Gated DeltaNet guard (flash_attn_legacy.gated_delta, docs/PLAN.md P6.10 a): with `fla` installed,
transformers binds the chunk delta rule of Qwen3.5-class models to fla's Triton kernel, which fails below compute
capability 8.0; models on our attention get the PyTorch reference there instead.

The unit tests need no `fla`: a stub package bound through transformers' own decorator (its module named fla.*, as
the guard requires). The integration tests run only where `fla` imports (an isolated directory on PYTHONPATH on the
test boxes).

Run:  pytest tests/test_gated_delta.py -v
"""

import functools
import importlib
import sys
import types

import pytest
import torch

transformers = pytest.importorskip("transformers")
hub_kernels = pytest.importorskip("transformers.integrations.hub_kernels")
flash_attn_legacy = pytest.importorskip("flash_attn_legacy")

from flash_attn_legacy import fallback_stats, gated_delta, reset_fallback_stats, utils  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
_PKG, _MOD = "_fal_stub_gdn_pkg", "_fal_stub_gdn_modeling"


@pytest.fixture(autouse=True)
def _clean_fallback_stats():
    reset_fallback_stats()
    yield
    reset_fallback_stats()


@pytest.fixture(autouse=True)
def _restore_qwen35_globals(monkeypatch):
    """Building a model on our attention patches the Qwen3.5 modeling module process-wide: each test starts and ends
    with transformers' own bindings."""
    try:
        mq = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    except ImportError:
        return
    for n in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn",
              "causal_conv1d_update"):
        monkeypatch.setattr(mq, n, getattr(mq, n))


def _reference(query, key, value, g, beta, initial_state=None, output_final_state=False,
               use_qk_l2norm_in_kernel=False, **kwargs):
    """Stands for transformers' torch reference: records its call."""
    return "reference", sorted(kwargs)


def _package_chunk(q, k, v, g, beta, **kwargs):
    return "package", sorted(kwargs)


def _bind(package_module_name):
    """transformers' decorator around _reference with a stub package importable at decoration time only (removed
    from sys.modules right after: nothing leaks into other tests)."""
    impl = types.FunctionType(_package_chunk.__code__, _package_chunk.__globals__, "stub_chunk_rule")
    impl.__module__ = package_module_name
    pkg = types.ModuleType(_PKG)
    pkg.stub_chunk_rule = impl
    sys.modules[_PKG] = pkg
    try:
        return hub_kernels.use_kernel_func_from_hub_with_fallback("stub_chunk_rule", _PKG)(
            functools.wraps(_reference)(lambda *a, **k: _reference(*a, **k)))
    finally:
        del sys.modules[_PKG]


@pytest.fixture
def stub_module(monkeypatch):
    """A modeling module whose torch_chunk_gated_delta_rule transformers bound to a stub `fla` package, and an
    instance of a model class defined in it."""
    module = types.ModuleType(_MOD)
    module.torch_chunk_gated_delta_rule = _bind("fla.ops.gated_delta_rule.chunk")
    monkeypatch.setitem(sys.modules, _MOD, module)
    model_cls = type("StubModel", (torch.nn.Module,), {"__module__": _MOD})
    return module, model_cls()


def _call(module, device=DEV, by_keyword=False, **kwargs):
    q = torch.zeros(1, 4, 2, 8, device=device)
    if by_keyword:
        return module.torch_chunk_gated_delta_rule(query=q, key=q, value=q, g=q[..., 0], beta=q[..., 0], **kwargs)
    return module.torch_chunk_gated_delta_rule(q, q, q, g=q[..., 0], beta=q[..., 0], **kwargs)


def _with_capability(monkeypatch, cc):
    monkeypatch.setattr(gated_delta, "device_capability", lambda index: cc)


@cuda_only
@pytest.mark.parametrize("cc,expected", [((6, 0), "reference"), ((7, 5), "reference"), ((8, 0), "package")])
def test_routes_by_capability(stub_module, monkeypatch, cc, expected):
    module, model = stub_module
    _with_capability(monkeypatch, cc)
    assert _call(module)[0] == "package"  # transformers bound the stub package
    assert gated_delta.guard_gated_delta(model)
    assert _call(module, cu_seqlens=None, position_ids=None)[0] == expected
    if expected == "reference":  # a keyword call (transformers' wrapper renames nothing for the package: positional)
        assert _call(module, by_keyword=True)[0] == "reference"
    assert fallback_stats() == ({"gated_delta_chunk_arch": 2} if expected == "reference" else {})


@cuda_only
def test_keywords(stub_module, monkeypatch):
    """Below SM 8.0 the reference receives what transformers' own wrapper would pass it, its named parameters only;
    from SM 8.0 the bound function receives the call unchanged (cu_seqlens included): the same result as unguarded."""
    module, model = stub_module
    cu = torch.tensor([0, 2, 4], device=DEV)
    unguarded = _call(module, cu_seqlens=cu, position_ids=None)
    gated_delta.guard_gated_delta(model)
    _with_capability(monkeypatch, (7, 0))
    assert _call(module, cu_seqlens=None, position_ids=None, output_final_state=True) == ("reference", [])
    _with_capability(monkeypatch, (8, 6))
    assert _call(module, cu_seqlens=cu, position_ids=None) == unguarded


@cuda_only
@pytest.mark.parametrize("packed", ["cu_seqlens", "seq_idx"])
def test_packed_sequences_raise(stub_module, monkeypatch, packed):
    module, model = stub_module
    _with_capability(monkeypatch, (6, 1))
    gated_delta.guard_gated_delta(model)
    with pytest.raises(ValueError, match="packed sequences"):
        _call(module, **{packed: torch.tensor([0, 2, 4], device=DEV)})


def test_cpu_and_non_tensor_calls_unchanged(stub_module):
    """Not CUDA tensors: the bound function, as without the guard."""
    module, model = stub_module
    gated_delta.guard_gated_delta(model)
    assert _call(module, device="cpu")[0] == "package"
    assert module.torch_chunk_gated_delta_rule(None, None, None, g=None, beta=None)[0] == "package"


def test_idempotent(stub_module):
    module, model = stub_module
    assert gated_delta.guard_gated_delta(model)
    guarded = module.torch_chunk_gated_delta_rule
    assert gated_delta.guard_gated_delta(model) and module.torch_chunk_gated_delta_rule is guarded


def test_kernels_module_layer(stub_module):
    """The `kernels` package turns transformers' wrapper into an nn.Module whose forward closes over it (`func`):
    found through it."""
    module, model = stub_module
    inner = module.torch_chunk_gated_delta_rule

    def make(func):
        class Func(torch.nn.Module):
            def forward(self, *args, **kwargs):
                return func(*args, **kwargs)
        return Func()

    if not isinstance(inner, torch.nn.Module):  # with `kernels` installed the stub already has its real layer
        module.torch_chunk_gated_delta_rule = make(inner)
    assert gated_delta._decorator_cells(module.torch_chunk_gated_delta_rule) is not None
    assert gated_delta.guard_gated_delta(model)
    assert getattr(module.torch_chunk_gated_delta_rule, gated_delta._GUARDED, False)


def test_third_party_wrapper_left_alone(stub_module):
    """A functools.wraps wrapper around transformers' function (another library's patch): not bypassed, a warning."""
    module, model = stub_module
    inner = module.torch_chunk_gated_delta_rule
    module.torch_chunk_gated_delta_rule = patched = functools.wraps(inner)(lambda *a, **k: inner(*a, **k))
    assert not gated_delta.guard_gated_delta(model) and module.torch_chunk_gated_delta_rule is patched


def test_other_package_left_alone(stub_module):
    module, model = stub_module
    module.torch_chunk_gated_delta_rule = original = _bind("other_kernels.chunk")
    assert not gated_delta.guard_gated_delta(model) and module.torch_chunk_gated_delta_rule is original


def test_subclass_in_another_module(stub_module):
    """The modeling module is found through the MRO (a user subclass defined elsewhere)."""
    module, model = stub_module
    sub = type("UserModel", (type(model),), {"__module__": __name__})()
    assert gated_delta.guard_gated_delta(sub)
    assert getattr(module.torch_chunk_gated_delta_rule, gated_delta._GUARDED, False)


def test_env_opt_out(stub_module, monkeypatch):
    module, model = stub_module
    monkeypatch.setenv("FLASH_ATTN_LEGACY_FLA_CHUNK", "fla")
    original = module.torch_chunk_gated_delta_rule
    assert not gated_delta.guard_gated_delta(model) and module.torch_chunk_gated_delta_rule is original


def test_reference_binding(monkeypatch):
    """Without fla, transformers binds its reference: the real Qwen3.5 modeling module gets the wrapper (for the scan
    kernel), unless the scan is turned off (then it is left as it is)."""
    from .hf_helpers import tiny_qwen35
    mq = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    cells = gated_delta._decorator_cells(mq.torch_chunk_gated_delta_rule)
    assert cells is not None, "transformers' kernel wrapper not recognised: the guard would never apply"
    if cells["is_new_implementation"]:
        pytest.skip("fla is installed here: see the integration tests")
    monkeypatch.setattr(mq, "torch_chunk_gated_delta_rule", mq.torch_chunk_gated_delta_rule)
    original = mq.torch_chunk_gated_delta_rule
    monkeypatch.setenv("FLASH_ATTN_LEGACY_GDN_SCAN", "0")
    assert not gated_delta.guard_gated_delta(tiny_qwen35()) and mq.torch_chunk_gated_delta_rule is original
    monkeypatch.delenv("FLASH_ATTN_LEGACY_GDN_SCAN")
    if not gated_delta._HAS_SCAN_OP:
        pytest.skip("an extension without the scan op")
    assert gated_delta.guard_gated_delta(tiny_qwen35(attn_implementation="flash_attn_legacy"))
    assert getattr(mq.torch_chunk_gated_delta_rule, gated_delta._GUARDED, False)


@pytest.mark.parametrize("impl", ["flash_attn_legacy", "sdpa"])
def test_register_hook(monkeypatch, impl):
    """Building a model with our attention runs the guard (for the outer model and its inner text model: idempotent);
    with another implementation it does not."""
    from .hf_helpers import tiny_qwen35
    seen = []
    monkeypatch.setattr(gated_delta, "guard_gated_delta", lambda model: seen.append(type(model).__name__))
    tiny_qwen35(attn_implementation=impl)
    assert ("Qwen3_5ForCausalLM" in seen) == (impl == "flash_attn_legacy") and (impl == "flash_attn_legacy" or not seen)


@cuda_only
@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile required")
@pytest.mark.parametrize("cc", [(7, 0), (8, 0)])
def test_compiled_call(stub_module, monkeypatch, cc):
    """The guarded function traces without a graph break through the capability lookup, both sides (the fallback
    count skipped under tracing): below SM 8.0 the reference; from SM 8.0 what the unguarded function gives compiled
    (transformers' wrapper itself takes its reference under tracing)."""
    module, model = stub_module
    q = torch.zeros(4, device=DEV)

    def f(q):
        out = module.torch_chunk_gated_delta_rule(q, q, q, g=q, beta=q)
        return q + (1.0 if out[0] == "reference" else 2.0)

    torch._dynamo.reset()
    unguarded = torch.compile(f, backend="eager", fullgraph=True)(q)
    gated_delta.guard_gated_delta(model)
    _with_capability(monkeypatch, cc)
    utils._CAPABILITY.clear()
    torch._dynamo.reset()
    got = torch.compile(f, backend="eager", fullgraph=True)(q)
    assert torch.equal(got, q + 1.0 if cc < gated_delta.FLA_CHUNK_MIN_CC else unguarded)


# ---------------------------------------------------------------------------
# With fla installed (skipped otherwise)
# ---------------------------------------------------------------------------

def _fla_bound():
    try:
        mq = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    except ImportError:
        return False
    cells = gated_delta._decorator_cells(mq.torch_recurrent_gated_delta_rule)
    return bool(cells and cells["is_new_implementation"])


fla_only = pytest.mark.skipif(not (torch.cuda.is_available() and _fla_bound()),
                              reason="needs CUDA and fla (transformers bound its kernels)")


@fla_only
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("K,V", [(128, 128), (64, 128)])
def test_reference_chunk_then_fla_recurrent(dtype, K, V):
    """The pairing the guard creates below SM 8.0: transformers' reference chunk on T tokens, then N decode steps of
    fla's recurrent kernel from its state (kept in the model dtype between steps, as the cache does). As close to
    the reference chunk over all T + N tokens as the reference recurrent is; the state transposed is not (the test
    sees a layout mix-up)."""
    mq = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    chunk_ref = gated_delta._decorator_cells(mq.torch_chunk_gated_delta_rule)["torch_function"]
    rec_ref = gated_delta._decorator_cells(mq.torch_recurrent_gated_delta_rule)["torch_function"]
    rec_fla = mq.torch_recurrent_gated_delta_rule
    B, T, N, H = 2, 64, 8, 4
    gen = torch.Generator(DEV).manual_seed(0)
    q, k = (torch.randn(B, T + N, H, K, device=DEV, generator=gen).to(dtype) for _ in range(2))
    v = torch.randn(B, T + N, H, V, device=DEV, generator=gen).to(dtype)
    g = -torch.rand(B, T + N, H, device=DEV, generator=gen) * 0.5
    beta = torch.rand(B, T + N, H, device=DEV, generator=gen).to(dtype)
    kw = dict(use_qk_l2norm_in_kernel=True)
    full, _ = chunk_ref(q, k, v, g=g, beta=beta, **kw)
    _, state = chunk_ref(q[:, :T], k[:, :T], v[:, :T], g=g[:, :T], beta=beta[:, :T], output_final_state=True, **kw)

    def decode(fn, s):
        outs = []
        for t in range(T, T + N):
            o, s = fn(q[:, t:t + 1], k[:, t:t + 1], v[:, t:t + 1], g=g[:, t:t + 1], beta=beta[:, t:t + 1],
                      initial_state=s, output_final_state=True, **kw)
            s = s.to(dtype)
            outs.append(o)
        out = torch.cat(outs, 1).float()
        return ((out - full[:, T:].float()).norm() / full[:, T:].float().norm()).item()

    err_ref = decode(rec_ref, state.to(dtype))
    err_fla = decode(rec_fla, state.to(dtype))
    assert err_fla <= 1.2 * err_ref + 1e-6, f"fla {err_fla:.2e} vs reference {err_ref:.2e}"
    assert decode(rec_fla, state.to(dtype).transpose(-1, -2).contiguous()) > 0.1  # negative control


@fla_only
def test_tiny_qwen35_with_fla(monkeypatch):
    """A model on our attention with fla bound: guarded; prefill bitwise the reference's (the same functions run);
    greedy decode (fla's recurrent kernel) close to the all-reference run until the tokens part at a near-tie."""
    from .hf_helpers import tiny_qwen35
    monkeypatch.setenv("FLASH_ATTN_LEGACY_GDN_SCAN", "0")  # the reference's own chunk: the bitwise prefill below
    mq = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    names = ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule")
    refs = {n: gated_delta._decorator_cells(getattr(mq, n))["torch_function"] for n in names}
    for n in names:  # restored at teardown: the guard is process-wide
        monkeypatch.setattr(mq, n, getattr(mq, n))
    model = tiny_qwen35(attn_implementation="flash_attn_legacy").to(DEV, torch.float16)
    if gated_delta.device_capability(torch.cuda.current_device()) >= gated_delta.FLA_CHUNK_MIN_CC:
        pytest.skip("SM 8.0 or newer: fla's chunk kernel runs, nothing to guard")
    assert getattr(mq.torch_chunk_gated_delta_rule, gated_delta._GUARDED, False)
    ids = torch.randint(0, 512, (2, 48), device=DEV, generator=torch.Generator(DEV).manual_seed(1))

    def run():
        with torch.inference_mode():
            return model(input_ids=ids).logits, model.generate(
                ids, max_new_tokens=16, do_sample=False, output_logits=True, return_dict_in_generate=True)

    guarded_logits, guarded_gen = run()
    for n, fn in refs.items():
        monkeypatch.setattr(mq, n, fn)
    ref_logits, ref_gen = run()
    assert fallback_stats().get("gated_delta_chunk_arch", 0) > 0
    assert torch.equal(guarded_logits, ref_logits)
    # Per row, the decode logits until the greedy tokens part (fla's recurrent kernel rounds differently: a random
    # tiny model has near-ties, after which the two runs see different contexts); a parting only at a near-tie
    live = torch.ones(ids.shape[0], dtype=torch.bool, device=DEV)
    for t, (a, b) in enumerate(zip(guarded_gen.logits, ref_gen.logits)):
        diff = (a.float() - b.float()).abs().amax(-1)
        assert (diff[live] < 2e-2).all(), f"step {t}: decode logits differ by {diff[live].max().item():.2e}"
        parted = live & (guarded_gen.sequences[:, ids.shape[1] + t] != ref_gen.sequences[:, ids.shape[1] + t])
        top2 = b.float().topk(2, -1).values
        assert ((top2[:, 0] - top2[:, 1])[parted] < 0.05).all(), f"step {t}: tokens part away from a near-tie"
        live &= ~parted


# ---------------------------------------------------------------------------
# The scan kernel (P6.10 c, c1): our copy of the reference with the sequential scan on gdn_chunk_scan
# ---------------------------------------------------------------------------

scan_only = pytest.mark.skipif(not (torch.cuda.is_available() and gated_delta._HAS_SCAN_OP),
                               reason="needs CUDA and the scan op")


def _reference_chunk():
    mq = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    reference = gated_delta._decorator_cells(mq.torch_chunk_gated_delta_rule)
    reference = reference["torch_function"] if reference else None
    if reference is None or not gated_delta._scan_matches(reference):
        pytest.skip("not the transformers reference the scan path was written from")
    return reference


def _gdn_inputs(B, L, H, K, V, dtype, regime, seed=0):  # noqa: C901
    """q / k / v / g / beta as the layer passes them ([B, L, H, .]); regime: "fast" (decays forget the state within a
    chunk), "slow" (g ~ -1e-3, beta near 1: the state carries across chunks), "flat" (g = 0, beta = 1), "parallel"
    (nearly parallel keys: an ill-conditioned UT system)."""
    gen = torch.Generator("cuda").manual_seed(seed)
    q = torch.randn(B, L, H, K, device="cuda", generator=gen)
    k = torch.randn(B, L, H, K, device="cuda", generator=gen)
    if regime == "parallel":
        k = k[:, :1] + 0.05 * k
    v = torch.randn(B, L, H, V, device="cuda", generator=gen)
    if regime == "fast":
        g = -torch.rand(B, L, H, device="cuda", generator=gen) * 0.5
        beta = torch.rand(B, L, H, device="cuda", generator=gen)
    elif regime == "flat":
        g, beta = torch.zeros(B, L, H, device="cuda"), torch.ones(B, L, H, device="cuda")
    else:
        g = -torch.rand(B, L, H, device="cuda", generator=gen) * 1e-3
        beta = 0.9 + 0.1 * torch.rand(B, L, H, device="cuda", generator=gen)
    return [t.to(dtype) for t in (q, k, v)] + [g, beta.to(dtype)]


@scan_only
@pytest.mark.parametrize("L,K,state", [(1, 128, False), (65, 128, True), (300, 64, True), (128, 128, False)])
def test_scan_copy_bitwise(L, K, state):
    """Our copy of the reference, with the reference's own loop, is bitwise the installed reference (fp16 inputs,
    padding, an initial state): only the scan differs on the kernel path."""
    reference = _reference_chunk()
    q, k, v, g, beta = _gdn_inputs(2, L, 4, K, 128, torch.float16, "slow")
    s0 = torch.randn(2, 4, K, 128, device="cuda") * 0.1 if state else None
    for final in (True, False):
        kw = dict(initial_state=s0, output_final_state=final, use_qk_l2norm_in_kernel=True)
        want = reference(q, k, v, g=g, beta=beta, **kw)
        got = gated_delta._chunk_scan(reference, q, k, v, g, beta, kernel=False, **kw)
        assert torch.equal(got[0], want[0]) and (got[1] is None) == (want[1] is None)
        if final:
            assert torch.equal(got[1], want[1])


@scan_only
@pytest.mark.parametrize("regime", ["fast", "slow", "flat", "parallel"])
@pytest.mark.parametrize("B,L,H,K", [(1, 1, 1, 128), (2, 63, 16, 128), (1, 64, 16, 64), (2, 65, 1, 128),
                                     (1, 1000, 16, 128), (2, 300, 4, 64), (2, 70, 32, 128), (1, 4096, 16, 128)],
                         ids=["1tok", "63", "64-K64", "65", "1000", "300-K64", "VS32-H32", "4096"])
def test_scan_kernel(regime, B, L, H, K):
    """The scan on the kernel against the reference in fp32 (inputs fp32: no output rounding in between): every
    head's output and final state within 1e-5 relative (fp32 reassociation over the chunks)."""
    reference = _reference_chunk()
    q, k, v, g, beta = _gdn_inputs(B, L, H, K, 128, torch.float32, regime)
    s0 = torch.randn(B, H, K, 128, device="cuda") * 0.1
    kw = dict(initial_state=s0, output_final_state=True, use_qk_l2norm_in_kernel=True)
    want = reference(q, k, v, g=g, beta=beta, **kw)
    got = gated_delta._chunk_scan(reference, q, k, v, g, beta, **kw)

    def rel(a, b, dims):  # per head: relative Frobenius over the other dims
        err = (a.double() - b.double()).pow(2).sum(dims).sqrt()
        return (err / b.double().pow(2).sum(dims).sqrt().clamp_min(1e-30)).max().item()

    assert rel(got[0], want[0], (0, 1, 3)) < 1e-5, "output"
    assert rel(got[1], want[1], (0, 2, 3)) < 1e-5, "final state"
    assert torch.equal(gated_delta._chunk_scan(reference, q, k, v, g, beta, **kw)[0], got[0]), "not deterministic"


@scan_only
def test_scan_routing(monkeypatch):
    """The wrapper takes our scan only without gradients and for the reference's supported arguments; the reference
    otherwise (bitwise)."""
    from .hf_helpers import tiny_qwen35
    mq = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    if gated_delta._decorator_cells(mq.torch_chunk_gated_delta_rule)["is_new_implementation"]:
        pytest.skip("fla is installed here")
    reference = _reference_chunk()
    monkeypatch.setattr(mq, "torch_chunk_gated_delta_rule", mq.torch_chunk_gated_delta_rule)
    assert gated_delta.guard_gated_delta(tiny_qwen35(attn_implementation="flash_attn_legacy"))
    calls = []
    real = gated_delta._ops

    class Recorder:
        @staticmethod
        def gdn_chunk_scan(*args):
            calls.append(1)
            return real.gdn_chunk_scan(*args)

    monkeypatch.setattr(gated_delta, "_ops", Recorder)
    q, k, v, g, beta = _gdn_inputs(1, 70, 2, 128, 128, torch.float16, "slow")
    run = mq.torch_chunk_gated_delta_rule

    def used(**kw):
        calls.clear()
        out = run(q, k, v, g=g, beta=beta, **{"use_qk_l2norm_in_kernel": True, **kw})
        return bool(calls), out

    with torch.inference_mode():
        assert used()[0]
    assert used()[0]  # grad enabled, nothing requires grad (a frozen prefix): the kernel
    assert not used(use_qk_l2norm_in_kernel=False)[0]
    assert not used(chunk_size=32)[0]
    qg = q.clone().requires_grad_(True)
    calls.clear()
    out = run(qg, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
    assert not calls and torch.equal(out[0], reference(qg, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)[0])
    monkeypatch.setenv("FLASH_ATTN_LEGACY_GDN_SCAN", "0")
    assert not used()[0]


@scan_only
@pytest.mark.skipif(not hasattr(torch.library, "opcheck"), reason="torch >= 2.4 required")
def test_scan_opcheck():
    w = torch.randn(1, 2, 3, 64, 128, device="cuda")
    u = torch.randn(1, 2, 3, 64, 128, device="cuda")
    d = torch.rand(1, 2, 3, device="cuda")
    for s0 in (None, torch.randn(1, 2, 128, 128, device="cuda")):
        torch.library.opcheck(torch.ops.flash_attn_legacy.gdn_chunk_scan.default, (w, u, w.clone(), d, s0),
                              test_utils=("test_schema", "test_faketensor"))


@scan_only
def test_tiny_qwen35_scan(monkeypatch):
    """A tiny Qwen3.5 (K != V) on our attention, a left-padded batch: prefill logits within fp16 noise of the
    reference's chunk; greedy decode the same until the tokens part at a near-tie."""
    from .hf_helpers import tiny_qwen35
    mq = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    if gated_delta._decorator_cells(mq.torch_chunk_gated_delta_rule)["is_new_implementation"]:
        pytest.skip("fla is installed here")
    monkeypatch.setattr(mq, "torch_chunk_gated_delta_rule", mq.torch_chunk_gated_delta_rule)
    model = tiny_qwen35(attn_implementation="flash_attn_legacy", key_dim=64).to("cuda", torch.float16)
    ids = torch.randint(0, 512, (2, 150), device="cuda", generator=torch.Generator("cuda").manual_seed(4))
    mask = torch.ones_like(ids)
    mask[1, :20] = 0  # left padding

    def run():
        with torch.inference_mode():
            return model(input_ids=ids, attention_mask=mask).logits.float()[mask.bool()], model.generate(
                ids, attention_mask=mask, max_new_tokens=32, min_new_tokens=32, do_sample=False, output_logits=True,
                return_dict_in_generate=True)

    ours = run()
    monkeypatch.setenv("FLASH_ATTN_LEGACY_GDN_SCAN", "0")
    stock = run()
    assert (ours[0] - stock[0]).abs().max().item() < 2e-2
    live = torch.ones(ids.shape[0], dtype=torch.bool, device="cuda")
    for t, (a, b) in enumerate(zip(ours[1].logits, stock[1].logits)):
        assert ((a.float() - b.float()).abs().amax(-1)[live] < 2e-2).all(), f"decode step {t}"
        parted = live & (ours[1].sequences[:, ids.shape[1] + t] != stock[1].sequences[:, ids.shape[1] + t])
        top2 = b.float().topk(2, -1).values
        assert ((top2[:, 0] - top2[:, 1])[parted] < 0.05).all(), f"step {t}: tokens part away from a near-tie"
        live &= ~parted


@scan_only
def test_scan_v48_and_dtypes():
    """The whole function against the reference: V = 48 (16-column slices), fp16 / bf16 / fp32 inputs, non-contiguous
    q / k / v (views of a wider projection, as the layer's split gives): within one rounding of the output dtype."""
    reference = _reference_chunk()
    eps = {torch.float16: 2.0 ** -10, torch.bfloat16: 2.0 ** -7, torch.float32: 1e-6}
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        for V in (48, 128):
            q, k, v, g, beta = _gdn_inputs(2, 130, 4, 128, V, dtype, "slow")
            wide = torch.cat([q, k, v], -1)  # [B, L, H, 256 + V]: q / k / v as slices
            q, k, v = wide[..., :128], wide[..., 128:256], wide[..., 256:]
            kw = dict(output_final_state=True, use_qk_l2norm_in_kernel=True)
            want = reference(q, k, v, g=g, beta=beta, **kw)
            got = gated_delta._chunk_scan(reference, q, k, v, g, beta, **kw)
            err = ((got[0].double() - want[0].double()).norm() / want[0].double().norm()).item()
            assert got[0].dtype == dtype and err < eps[dtype], f"{dtype} V={V}: {err:.1e}"
            assert ((got[1] - want[1]).norm() / want[1].norm()).item() < 1e-5


@scan_only
def test_scan_peak_memory():
    """The kernel path's peak memory (the per-chunk states included) is no higher than the reference's."""
    reference = _reference_chunk()
    q, k, v, g, beta = _gdn_inputs(1, 2048, 16, 128, 128, torch.float16, "slow")
    peaks = []
    for fn in (lambda: reference(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True),
               lambda: gated_delta._chunk_scan(reference, q, k, v, g, beta, use_qk_l2norm_in_kernel=True)):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        fn()
        torch.cuda.synchronize()
        peaks.append(torch.cuda.max_memory_allocated() - base)
    assert peaks[1] <= peaks[0], f"reference {peaks[0] / 2**20:.0f} MiB, scan {peaks[1] / 2**20:.0f} MiB"


@scan_only
def test_scan_fla_bound_route(monkeypatch):
    """With fla bound (the module's chunk -> fla), below SM 8.0 our scan serves the call (not counted as a fallback:
    the reference does not run); from SM 8.0, fla."""
    reference = _reference_chunk()
    calls = []

    def fla_chunk(*args, **kwargs):
        calls.append("fla")
        return "fla"

    cells = {"implementation": fla_chunk, "torch_function": reference, "is_new_implementation": True}
    wrapper = gated_delta._chunk_wrapper(fla_chunk, cells)
    real = gated_delta._ops

    class Recorder:
        @staticmethod
        def gdn_chunk_scan(*args):
            calls.append("scan")
            return real.gdn_chunk_scan(*args)

    monkeypatch.setattr(gated_delta, "_ops", Recorder)
    q, k, v, g, beta = _gdn_inputs(1, 70, 2, 128, 128, torch.float16, "slow")
    for cc, expected in (((7, 0), ["scan"]), ((8, 0), ["fla"])):
        calls.clear()
        monkeypatch.setattr(gated_delta, "device_capability", lambda index, cc=cc: cc)
        wrapper(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True, cu_seqlens=None)
        assert calls == expected, cc
    assert fallback_stats() == {}


@scan_only
def test_scan_routing_more(monkeypatch):
    """Gradients through g or the initial state, and CPU tensors: the reference; an unsupported inference call is
    counted."""
    from .hf_helpers import tiny_qwen35
    mq = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    if gated_delta._decorator_cells(mq.torch_chunk_gated_delta_rule)["is_new_implementation"]:
        pytest.skip("fla is installed here")
    _reference_chunk()
    assert gated_delta.guard_gated_delta(tiny_qwen35(attn_implementation="flash_attn_legacy"))
    calls = []
    real = gated_delta._ops

    class Recorder:
        @staticmethod
        def gdn_chunk_scan(*args):
            calls.append(1)
            return real.gdn_chunk_scan(*args)

    monkeypatch.setattr(gated_delta, "_ops", Recorder)
    q, k, v, g, beta = _gdn_inputs(1, 70, 2, 128, 128, torch.float16, "slow")
    s0 = torch.randn(1, 2, 128, 128, device="cuda")
    run = mq.torch_chunk_gated_delta_rule
    run(q, k, v, g=g.clone().requires_grad_(True), beta=beta, use_qk_l2norm_in_kernel=True)
    run(q, k, v, g=g, beta=beta, initial_state=s0.clone().requires_grad_(True), use_qk_l2norm_in_kernel=True)
    run(*(t.cpu() for t in (q.float(), k.float(), v.float())), g=g.cpu(), beta=beta.float().cpu(),
        use_qk_l2norm_in_kernel=True)
    assert not calls and fallback_stats() == {}
    with torch.no_grad():
        run(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True, chunk_size=32)
    assert not calls and fallback_stats() == {"gated_delta_scan_input": 1}
