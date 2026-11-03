# Gated DeltaNet with `fla` installed, below SM 8.0 (P6.10 a)

Status: done (2026-11-03): planned, reviewed, implemented in `flash_attn_legacy/gated_delta.py`, measured. Section 4
records what changed after the review and the results. First of the order decided on
2026-11-03: (a) this guard, (b) the causal conv1d (`qwen35_conv1d.md`), (c) a delta-rule kernel.

## 1. The failure

transformers 5.20.2 binds the Gated DeltaNet functions of Qwen3.5, Qwen3.5-MoE, Qwen3-Next and OLMo-hybrid at import
time (`use_kernel_func_from_hub_with_fallback`, see `qwen35_conv1d.md` section 1):
- `torch_chunk_gated_delta_rule` (prefill, training) is bound to `fla.ops.gated_delta_rule.chunk_gated_delta_rule`
  when `fla` imports;
- `torch_recurrent_gated_delta_rule` (decode) is bound to `fla`'s `fused_recurrent_gated_delta_rule`;
- otherwise both are bound to the PyTorch reference.

The decorated global keeps the reference as `__wrapped__` and records the choice in its closure
(`is_new_implementation`).

**Measured** (fla 0.6.0, Triton 3.3.1, torch 2.7.1; `fla` on `PYTHONPATH` from an isolated directory, never in the
test environments). Qwen3.5-2B fp16 with our attention, prefill of 4096 tokens and generate 2048 + 64.

On all five GPUs, both prefill and generate fail at the first chunk call: `RuntimeError: PassManager::run failed`
(Triton cannot lower `chunk_fwd.py:40` for the device: `ConvertTritonGPUToLLVM`). The GPUs are P100 (sm_60), 1080 Ti
(sm_61), V100 (sm_70), 2080 Ti and T4 (sm_75). So today, installing `fla` breaks Qwen3.5 on every GPU this project
targets. The 2026-10-24 survey had also measured `fla`'s chunk in fp32 on the P100: it runs, 1.9x slower than the
reference.

**The guard prototype** (`guard_proto.py`, scratch) sets the chunk global back to its `__wrapped__` reference and
keeps `fla`'s recurrent kernel. The wall time is from `qwen35_conv_prof.py`, one run each:

| GPU | prefill 4096, reference / guard | generate, reference / guard | recurrent calls, reference / guard |
|---|---|---|---|
| P100 | 1920 / 1920 ms | 2651 / 2622 ms (1.01x) | 319 / 162 ms |
| 1080 Ti | 3418 / 3446 ms | 4469 / 4034 ms (1.11x) | 639 / 267 ms |
| V100 | 371 / 370 ms | 2741 / 2433 ms (1.13x) | 520 / 217 ms |
| 2080 Ti | 519 / 520 ms | 3521 / 3097 ms (1.14x) | 667 / 238 ms |
| T4 | 1019 / 1043 ms | 3828 / 3520 ms (1.09x) | 621 / 285 ms |

- Prefill is unchanged: the same reference runs.
- Decode gains 9-14% on the vast boxes, through `fla`'s recurrent kernel.
- On the P100 (mew's faster host) the recurrent calls' event time halves, but the wall time barely moves: that step
  is bound elsewhere.
- With mew's sm_60 build of `causal-conv1d` also on the path, generate is 2449 ms (1.08x).

## 2. Plan

**Where.** In `register()`'s `_patched_check`: when a model is built with `attn_implementation="flash_attn_legacy"`,
the modeling module of its class (`sys.modules[type(model).__module__]`) is guarded once. This is the same hook as
`FLASH_ATTN_LEGACY_LINEAR`. A new module `flash_attn_legacy/gated_delta.py` holds the guard, and later P6.10 (b)'s
conv hook.

**What.** The guard applies when the module defines `torch_chunk_gated_delta_rule` and its closure says a package
implementation was bound. In that case the global is replaced by a wrapper with this behaviour:
- **CUDA inputs with compute capability below 8.0.** It calls `__wrapped__`, the reference, the same function
  transformers runs without `fla`. A warning is logged once per process.
- **Packed sequences** (`cu_seqlens` given) on such a device raise a `ValueError`. It says that packed sequences need
  `fla`'s chunk kernel, which does not run below SM 8.0. The reference would silently mix the sequences, which is why
  transformers raises without `fla`.
- **Otherwise** (SM 8.0 or newer, CPU) it calls the original global unchanged.

**What it does not touch.**
- The recurrent global: `fla`'s recurrent kernel ran on all five GPUs (the generate rows above).
- The conv globals: they belong to P6.10 (b), whose own guard probes a `causal-conv1d` build for the device.
- Models loaded with other attention implementations.

**Scope of the change.**
- It is process-wide for that modeling module, so other models of the same class in the process also get it. On these
  GPUs that is what they need, because without it they crash.
- It is idempotent: a wrapped global is recognised by an attribute and never wrapped twice.
- It is a no-op when the closure cannot be read (another transformers version): nothing is replaced.
- Under `torch.compile`, the capability check is a constant folded by `torch.cuda.get_device_capability`, as in
  `linear.py`.

## 3. Acceptance

1. **Unit tests, without `fla` installed** (all boxes). They use a fake modeling module whose global is decorated by
   transformers' real `use_kernel_func_from_hub_with_fallback`, with a stub package implementation:
   - below SM 8.0 the reference runs (the capability is monkeypatched to test both sides);
   - `cu_seqlens` raises below SM 8.0;
   - the wrapper is idempotent;
   - a module whose global is bound to the reference is left untouched;
   - other attention implementations leave it untouched.
2. **Integration, with `fla` on `PYTHONPATH`** (a test that skips when `fla` does not import). It runs on all five
   boxes with the isolated `fla` directory:
   - Qwen3.5 prefill and generate run (they crash today);
   - the prefill logits are bitwise those of the run without `fla`, since the same reference runs;
   - generate gives the same 32 greedy tokens as without `fla`, or diverges first at a near-tie (top-2 gap < 0.05 in
     the reference run). `fla`'s recurrent kernel differs numerically.

   Model: a tiny random-init Qwen3.5 config, plus Qwen3.5-0.8B where it is cached.
3. **Speed.** Decode with the guard against decode without `fla`, A/B alternated (2 rounds per arm), on all five
   GPUs. The expected range comes from section 1. Prefill must not slow down beyond noise.
4. **Suite.** The full suite stays green on all five boxes, without `fla`.
5. **Docs.**
   - README: a line in the Qwen3.5 notes, saying that installing `fla` is supported (decode faster), packed training
     is not, and why.
   - PLAN P6.10: (a) done.

## 4. As implemented, and the results

**Review points.**
- Verified on transformers 5.20.2: the four modeling files call the chunk and recurrent functions as module globals
  (qwen3_5: lines 612 / 625). Their gated RMSNorm is decorated only for hub kernels, not bound to `fla`, so with the
  guard the prefill runs exactly the functions it runs without `fla`.
- The guard recognises transformers' wrapper as the global itself or, with the `kernels` package installed (pinned
  by `transformers[kernels]`), inside the `nn.Module` that `kernels.use_kernel_forward_from_hub` makes of it (its
  forward's `func` cell; second review, verified with kernels 0.17.2). It reads the reference from the wrapper's
  `torch_function` cell.
  - It guards only a binding to `fla` (the implementation's module), and it leaves alone any other wrapper around
    transformers' function: a third-party patch, or our future delta-rule kernel.
  - An unrecognised global (empty cell, unknown layout) is left as it is, with a one-time warning.
- It passes the reference only its named keyword parameters, as transformers' wrapper does.
- Only CUDA tensors below SM 8.0 are routed. On CPU, fla's recurrent kernel would fail in decode anyway, so a CPU route
  for the chunk alone was dropped.
- `cu_seqlens` / `seq_idx` raise, naming both requirements: `fla`'s chunk kernel and `causal-conv1d`.
- Under `torch.compile`, transformers' wrapper itself takes its reference, so compiled calls are unchanged from SM 8.0;
  the test compares against the unguarded compiled call.
- The fallback is counted in the one registry: `gated_delta_chunk_arch`, action "reference". Counting is skipped under
  `torch.compile`.
- The module is found through the model class's MRO.
- `FLASH_ATTN_LEGACY_FLA_CHUNK=fla` opts out.
- The compute capability comes from one cached helper, `utils.device_capability`, which `linear.py` now shares. It
  reads `get_device_properties`, which compile folds.
- The threshold stays static (SM 8.0), not probed. A Triton compile failure costs seconds and prints a large MLIR dump,
  and it is measured on fla 0.6.0 / Triton 3.3.1.

**The pairing gate** (review C1; `tests/test_gated_delta.py::test_reference_chunk_then_fla_recurrent`, which needs
`fla`).
- Setup: the reference chunk on 64 tokens, then 8 steps of `fla`'s recurrent kernel from its state, kept in the model
  dtype between steps, compared with the reference chunk over all 72 tokens.
- On all five GPUs and in fp16, bf16 and fp32, with K = V = 128 and with K = 64, V = 128, `fla`'s error equals the
  reference recurrent's: fp16 3.1-3.5e-4, bf16 2.6-2.7e-3, fp32 2.1-2.7e-7.
- The negative control is rejected (0.64-0.73): the state transposed.
- So the state layout and dtype pass between the two implementations, and bf16 works too.
- Real model, V100: Qwen3.5-2B fp16 gives the same 32 greedy tokens with and without the guard, and the same top-2
  gaps within 0.02, near-ties of 0.03-0.05 included.

**Tests.**
- `tests/test_gated_delta.py`: 18 unit tests without `fla`. They use a stub `fla` package bound through transformers'
  real decorator, imported at decoration time only, and cover routing, keywords, packed sequences, the `kernels`
  layer, third-party wrappers, other packages, the MRO, the env opt-out, the `register()` hook and compile.
- 7 integration tests with `fla`, including a tiny random Qwen3.5 (K ≠ V):
  - its prefill logits are bitwise the all-reference run's;
  - its decode logits match within 2e-2 until the greedy tokens part, which happens only at a near-tie.
- With `fla` on `PYTHONPATH`, 24 passed on all five GPUs (on the P100 also with `kernels` 0.17.2). Without `fla`, the
  full suite is green on all five GPUs.

**Speed** (`gdn_ab.py`, a one-off script). Qwen3.5-2B fp16 on our attention, one process per run, A = without `fla`,
B = with `fla` and the guard, alternated A B B A A B B A (4 rounds per arm), medians. On mew the sm_60 `causal-conv1d`
build is kept off the path, so this measures the guard alone.

| GPU | generate 2048 + 64, A / B | speedup | prefill 4096, A / B |
|---|---|---|---|
| P100 | 2540 / 2410 ms | 1.054x | 1920 / 1920 ms |
| 1080 Ti | 4262 / 3982 ms | 1.070x | 3513 / 3495 ms |
| V100 | 2607 / 2375 ms | 1.098x | 369 / 370 ms |
| 2080 Ti | 3407 / 3009 ms | 1.132x | 524 / 526 ms |
| T4 | 3564 / 3352 ms | 1.063x | 1184 / 1166 ms |

The ranges of the two arms do not overlap on any GPU. Without the guard, B does not run at all: `fla`'s chunk kernel
crashes.
