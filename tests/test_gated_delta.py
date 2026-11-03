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


def test_reference_binding_untouched():
    """Without fla, transformers binds its reference: the real Qwen3.5 modeling module is left as it is."""
    from .hf_helpers import tiny_qwen35
    mq = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    cells = gated_delta._decorator_cells(mq.torch_chunk_gated_delta_rule)
    assert cells is not None, "transformers' kernel wrapper not recognised: the guard would never apply"
    if cells["is_new_implementation"]:
        pytest.skip("fla is installed here: see the integration tests")
    original = mq.torch_chunk_gated_delta_rule
    model = tiny_qwen35(attn_implementation="flash_attn_legacy")
    assert not gated_delta.guard_gated_delta(model) and mq.torch_chunk_gated_delta_rule is original


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
