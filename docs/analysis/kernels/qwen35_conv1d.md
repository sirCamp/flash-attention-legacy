# Qwen3.5's causal conv1d on pre-Ampere GPUs (P6.10 b)

Status: done (2026-11-03): analysed, reviewed, implemented (`csrc/kernels/causal_conv1d.cu`,
`flash_attn_legacy/gated_delta.py`), reviewed again, measured. Section 6 records how it was built and the results.
Order decided 2026-11-03: P6.10 (a) the `fla` guard, then this conv (b), then the delta-rule kernel (c).

Qwen3.5 interleaves Gated DeltaNet layers (3 in 4) with full-attention layers. Each Gated DeltaNet layer runs a
depthwise causal conv1d (kernel 4, SiLU) on its fused q / k / v projection before the delta rule. This note measures
what transformers' fallback for that conv costs on our GPUs and proposes a replacement.

## 1. Today

**How transformers 5.20.2 binds the conv** (`transformers/models/qwen3_5/modeling_qwen3_5.py`; Qwen3-Next and
OLMo-hybrid have the same pattern):
- Two module-level functions, `causal_conv1d_update` (line 235) and `causal_conv1d_fn` (line 255), each decorated with
  `use_kernel_func_from_hub_with_fallback(name, "causal_conv1d")`.
- At import time the decorator picks one implementation, in this order:
  1. a hub kernel, when the `kernels` package is installed and asked for;
  2. the `causal_conv1d` package, when it imports;
  3. the PyTorch reference.
- `Qwen3_5GatedDeltaNet.forward` calls the module globals by name on every call (lines 563 and 576), so replacing a
  global takes effect for already-built models too. `use_kernelized_func` (line 487) is a no-op without the `kernels`
  package.
- With the reference, a call with `cu_seqlens` or `seq_idx` raises ("can't keep packed sequences apart without
  causal-conv1d"). The chunk gated delta rule raises the same way without `fla`, so packed training needs both.

**The reference.**
- **`causal_conv1d_fn(hidden_states, weight, bias=None, activation=None)`.**
  - `hidden_states` is `mixed_qkv.transpose(1, 2)`: [B, C, L] logically, channel-last in memory.
  - It runs `F.conv1d(groups=C, padding=K-1)` and slices the first L outputs; SiLU is optional. The profile shows torch's
    `conv_depthwise2d_forward_kernel_generic`, which reads x through a contiguous copy.
  - The output is rounded to fp16 twice: once after the conv, once after SiLU.
- **`causal_conv1d_update(hidden_states, conv_state, weight, bias=None, activation=None)`.**
  - It concatenates the state [B, C, K] with the new tokens, copies the last K columns back into the state in place,
    runs `F.conv1d` without padding, keeps the last `seq_len` outputs, and applies the activation.
  - That is about six launches per decode call, one per Gated DeltaNet layer (18 in Qwen3.5-2B).
- In prefill the model writes the conv state itself (`cache_params.update_conv_state`), before `causal_conv1d_fn`: the
  conv op returns no state.

**Packages on our GPUs.**
- `causal-conv1d`'s wheels and default build cover sm_75 and newer, so the P100, 1080 Ti and V100 have no kernel. A
  source build for sm_60 works (survey 2026-10-24: 5-11x on the conv).
- The T4 and 2080 Ti (sm_75) are covered when a wheel matches torch / CUDA, else they need a source build. None of our
  boxes has the package.
- An installed package (or hub kernel) without code for the device fails at the first call ("no kernel image"). This
  is the same failure P6.10 records for `fla` on Pascal.

Sizes: C = 6144 (Qwen3.5-0.8B / 2B, 18 Gated DeltaNet layers of 24), 8192 (4B, 24 of 32); kernel 4; no bias.

**Alone** (`qwen35_conv_micro.py`: the reference on the model's layout, CUDA events, median). The floor is one read of
x and one write of y at the nominal bandwidth.

| GPU | fwd, B = 1, L = 4096 | vs floor | fwd + bwd (dX), L = 4096 | update, per call |
|---|---|---|---|---|
| P100 | 3.02 ms | 22x | 8.36 ms | 60 µs |
| 1080 Ti | 6.07 ms | 29x | 16.1 ms | 143 µs |
| V100 | 0.98 ms | 8.7x | 3.10 ms | 127 µs |
| 2080 Ti | 1.29 ms | 7.9x | 3.79 ms | 166 µs |
| T4 | 2.70 ms | 8.6x | 8.88 ms | 121 µs |

- Across L = 1k-16k and B = 1-4 the forward is 7-34x above the floor.
- The update's event time equals its host time per call. It is bound by launch overhead, and the vast boxes' hosts
  are 2-3x slower than mew's.

**In the model** (`qwen35_conv_prof.py`: CUDA events around the module functions; Qwen3.5-2B fp16, our attention,
stock `F.linear`):

| GPU, step | wall | conv | chunk delta rule | recurrent delta rule |
|---|---|---|---|---|
| P100, prefill 4096 | 1920 ms | 54.4 ms (2.8%) | 248.7 ms (13.0%) | - |
| P100, LoRA train 2048 (checkpointing) | 4795 ms | 54.9 ms fwd + ~53 ms bwd kernels (~2.2%) | 254.6 ms (5.3%) | - |
| P100, generate 2048 + 64 | 2660 ms | 27.4 ms + update 30.0 ms (2.2%) | 127.4 ms (4.8%) | 317.3 ms (11.9%) |
| V100, prefill 4096 | 370 ms | 15.7 ms (4.3%) | 146.1 ms (39.5%) | - |
| V100, generate 2048 + 64 | 2853 ms | 7.6 ms + update 191.3 ms (7.0%) | 82.8 ms (2.9%) | 547.0 ms (19.2%) |
| T4, prefill 4096 | 1040 ms | 50.8 ms (4.9%) | 448.9 ms (43.2%) | - |
| T4, generate 2048 + 64 | 4000 ms | 23.7 ms + update 266.3 ms (7.3%) | 207.9 ms (5.2%) | 657.2 ms (16.4%) |

The LoRA run did not start on the vast boxes: `peft` is not installed there.

The per-call update times in the model disagree with the micro-benchmark:
- P100: ~26 µs in the model against 60 µs in the micro-benchmark;
- V100: ~166 µs against 127 µs;
- T4: ~231 µs against 121 µs.

Both are event times of a launch-bound call, so they track the host, which differs between a model step and a tight
loop. The decode gain below is therefore an upper bound, to be measured with a prototype op.

## 2. What a replacement can gain

The conv is memory-bound: 4 multiply-adds per element read. A fused kernel (channel-last read, conv and SiLU in fp32,
one rounding, one write) at 60-70% of the bandwidth would run the forward at ~1.5x the floor, 5-20x faster than the
reference.

The update becomes one launch instead of about six. A dispatcher op still costs a few µs on the host (P6.5 measured
~6 µs per op call), so the update does not drop to zero.

Expected on Qwen3.5-2B:
- **Prefill.** -2.5% on the P100 (more with `patch_linear`, which shrinks the rest); -4 to -5% on V100 / T4.
- **LoRA.** About -2% on the P100.
- **Decode.** At most the update's share minus the op's host cost: about -1.5% on the P100 and up to about -5% on V100 /
  T4. The figure is to be measured with a prototype before it is written into a README.
  - If HF's compiled decode applies to this model (`_can_compile_fullgraph = True` on the Qwen3.5 classes; the hybrid
    cache under CUDA graphs is not verified here), the launch overhead disappears and with it this part of the gain.

This is a small, bounded gain. The larger costs of the same layers are the delta rule's PyTorch reference: 5-13% on the
P100, 40-43% of the prefill on V100 / T4, and 12-19% of decode for the recurrent step.

The decode lever is the delta rule, not the conv:
- `fla`'s recurrent kernel already runs on Pascal, 3x faster (P6.10 survey). Its guard, P6.10 (a), is cheaper than this
  proposal and saves more decode time.
- A fused decode op (conv update + recurrent step) is the larger decode win. The update op here must not block it:
  same state layout, no hidden copies.

Proposed order: P6.10 (a) guard, then this conv, then (c) the delta-rule kernel.

## 3. Options

**(a') A documented source build of `causal-conv1d`** for our architectures, e.g.
`TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5" pip install causal-conv1d --no-build-isolation`, plus the capability guard of
section 4 so that a package without code for the device is never used.
- Pros:
  - no code in our repo;
  - transformers picks the package up by itself, `seq_idx` included.
- Cons:
  - every user needs a CUDA toolchain and a 10-20 minute build;
  - a hub kernel (`kernels` package) can shadow the local build;
  - we cannot tune its Pascal path. The survey measured 5-11x, against the 5-20x the floor allows; that gap is not
    measured.

**(a) Vendor `causal-conv1d`** (Tri Dao, BSD-3) in our extension.
- Pros: its semantics exactly.
- Cons: several thousand lines of templated CUDA with its own dispatch, CUB and build flags, a second code style next
  to ours, and more build time.

**(b) Our own kernel**, in the existing extension and dispatch. A depthwise causal conv of width 2-4, channel-last, SiLU
optional, about 500 lines plus tests, with no arch-specific code.

**Recommendation: (b), with (a') documented as the alternative** for users who want `seq_idx` before the delta-rule
kernel exists. The kernel is small and memory-bound, so the gain is all in the layout and the fusion, which we
control, and it follows the project rule of one way and one place. (a) brings more code than the feature needs.

**Kernel design (b).**
- **dtypes.** fp16, bf16 and fp32 are converted on load, accumulated in fp32 and rounded once to the input dtype. bf16
  on Pascal uses software conversion, as the decode kernel does. The weight is fp32 or the input's dtype.
- **Forward.** Each thread owns 8 channels (one 16-byte load per token when C % 8 == 0 and the rows are aligned,
  otherwise scalar loads) and a run of tokens.
  - It slides a K-token window in registers.
  - The K-1 tokens before its run are a halo re-read from global memory (L2), not a dependency between threads.
  - The window is zero before the first token of each batch row.
  - Template on K ∈ {2, 3, 4}; with L < K the window is partly zero.
  - The output has x's strides (channel-last in, channel-last out): transformers transposes it back to [B, L, C] and
    splits it, so a contiguous [B, C, L] output would cost a copy.
- **Backward.**
  - g = dy · silu'(z), with z recomputed in the forward's accumulation order and silu'(z) = s(z)(1 + z(1 - s(z))) in
    fp32.
  - dX is the anti-causal conv of g. Its halo is two-sided: x[t-K+1 .. t+K-1] for z, and g beyond L-1 is zero.
  - dW and db are computed only when `needs_input_grad` asks for them (the LoRA case freezes the conv). They come from
    per-block fp32 partials [blocks, C, K], reduced in a fixed order. The partition does not depend on the SM count,
    so dW is bitwise reproducible across GPUs.
- **Update.** One thread per 8 channels.
  - It reads all state columns before writing.
  - It writes the shifted state in place: the op schema marks it `Tensor(a!)`.
  - It handles a state dtype that differs from x's: the reference casts to the weight's dtype.
  - It takes `seq_len` ≥ 1 and a state width ≥ K-1 (HF's is K).

## 4. Integration

- **Ops.** `torch.ops.flash_attn_legacy.causal_conv1d` and `causal_conv1d_update`, with fake implementations and
  `register_autograd`, as for the attention and linear ops.
- **Hook.**
  - When `register()` sees a model built with `attn_implementation="flash_attn_legacy"` (the check that the linear
    flag already uses), it replaces the module globals `causal_conv1d_fn` / `causal_conv1d_update` of the model's
    modeling module with wrappers. It applies only to modules that define both names: Qwen3.5, Qwen3.5-MoE, Qwen3-Next
    and OLMo-hybrid in 5.20.
  - The replacement is process-wide. After it, other models of the same class in the process also use our conv, as
    they would after installing `causal-conv1d`. This is documented. A model loaded with SDPA keeps it.
  - The wrappers keep the reference's signatures (`hidden_states`, `conv_state`, `weight`, `bias`, `activation`, plus
    `**kwargs`). Inputs ours does not take go to the original function, which keeps transformers' behaviour, including
    its error for packed inputs. Those inputs are: `seq_idx` / `cu_seqlens`, width above 4, a channel stride ≠ 1, and
    a non-CUDA device.
  - The fallbacks are counted under new `conv1d_*` reasons in the one registry (`FALLBACK_REASONS`, a new "conv1d"
    action in `_record_fallback`); address checks are skipped under torch.compile, as `_decode_layout_ok` does.
- **Capability guard.** If the decorator picked `causal-conv1d` or a hub kernel, ours replaces it only on devices for
  which that package has no code (below sm_75, or when a one-time probe call per device fails). There it would crash.
  The guard is shared with P6.10 (a)'s `fla` guard: one place.
- **Autocast.** Under fp16 autocast with fp32 weights (the AMP recipe), `F.conv1d` is on autocast's fp16 list. The
  wrapper casts the same way, as `linear.py` does, so AMP training does not silently fall back.
- **Default.** On, with `FLASH_ATTN_LEGACY_CONV1D=0` to opt out.
  - The change removes one rounding and changes nothing else.
  - This differs from the opt-in `FLASH_ATTN_LEGACY_LINEAR`, which changes the accumulation scheme.

## 5. Acceptance

Each item has a test or a measurement:

1. **Forward.**
   - Within the output rounding of an fp64 reference, with relative error ≤ the reference implementation's.
   - Shapes and inputs covered: B ∈ {1, 3}, L ∈ {1, 2, 3, 4, 5, 127, 4096}, C ∈ {8, 6144, 6150 (not a multiple of
     8)}, K ∈ {2, 3, 4}, with and without bias and SiLU.
   - Layouts: channel-last, contiguous, a strided slice, and a misaligned view.
   - dtypes: fp16, bf16 and fp32.
   - The output strides equal x's.
2. **Backward.**
   - dX, dW and db are within tolerance of fp64 gradients, with error ≤ the reference's.
   - dW is bitwise reproducible run to run.
   - A frozen weight computes no dW (checked by `needs_input_grad`).
3. **Update.**
   - The output equals the forward's last token on the concatenated sequence, bitwise when both run in fp32.
   - The state is updated in place; opcheck passes on the mutating schema.
   - It works for `seq_len` = 1 and 3, and for state dtype ≠ x dtype.
   - A raw `torch.cuda.graph` capture of a decode step works, and so does HF's compiled decode if it applies (verify).
4. **Ops.** opcheck (schema, fake, autograd registration) and `torch.compile(fullgraph=True)` pass.
5. **Integration**, on tiny random-init Qwen3.5 and Qwen3-Next configs (no download) and on Qwen3.5-0.8B / 2B in fp16:
   - prefill logits within the fp16 noise of the reference path;
   - greedy generation identical for 64 tokens, or its first divergence at a near-tie (top-2 gap < 0.05 in an fp32
     reference);
   - LoRA loss over 20 steps (fixed seed) within the reference's own run-to-run spread;
   - gradient checkpointing works;
   - padded batches (transformers zeroes the pad tokens before the conv) give the same results as unpadded ones;
   - fallbacks are counted;
   - nothing changes for models loaded with SDPA in a fresh process, or with `FLASH_ATTN_LEGACY_CONV1D=0`;
   - with an importable `causal-conv1d` that has code for the device, nothing is replaced.
6. **Speed.**
   - The forward is ≤ 2x the bandwidth floor and forward + backward ≤ 3x (dW pass included), on P100, V100 and T4 at
     L = 4096.
   - The update is ≤ 2 launches per call, and its host time per call is measured.
   - Prefill, decode and LoRA of Qwen3.5-2B are not slower than the reference path on any box, A/B alternated with at
     least 4 rounds per arm.
   - The prefill gain is reported with its spread.
7. **Other GPUs.** The full suite stays green on all five boxes.

## 6. As implemented, and the results

**Ops.**
- `causal_conv1d(x, weight, bias, silu)`, `causal_conv1d_backward(dy, x, weight, bias, silu, need_dw)` and
  `causal_conv1d_update(x, conv_state!, weight, bias, silu)`, with fake implementations; autograd is registered on the
  forward op.
- The kernels read 16-byte vectors along the channels (8 fp16 / bf16 or 4 fp32 channels per thread) over runs of 32
  tokens, with a W-1 token halo re-read per run.
  - Run lengths measured on the P100: 64 tokens gave 2.01x the floor at 4k, 32 gave 1.82x, 16 gave 1.83x with a slower
    dW pass.
  - Unaligned inputs or C % 8 != 0 take a scalar path.
- The backward recomputes z in the forward's order.
  - dW / db come from per-run fp32 partials, reduced in run order by a second kernel. The summation order does not
    depend on the GPU; the SFU's exp in SiLU' may.
  - dW / db are computed only when the weight or bias needs a gradient: a frozen conv under LoRA gets dX only.
- The decode update uses one thread per channel and supports runtime dtypes. It writes the state shift in place, as
  exact values, while the reference rounds an fp32 state of an fp16 model through the weight's dtype. With the state
  in x's dtype the result is bitwise the reference's.
- Registers: at most 168, no spills on sm_60 / 61 / 70 / 75.

**Deviations from the plan, after the reviews.**
- An fp16 x with fp32 weights (the AMP recipe) runs directly in the fp32-weight instance. That is more accurate than
  the reference's cast to fp16, and dW comes back in fp32. There is no autocast cast.
- The output is compact channel-last ([B, L, C] memory), not x's strides when x is a slice of a wider tensor.
- A package bound by transformers is kept on SM 7.5 and newer by capability, not probed.
- Torch below 2.4 has no `register_autograd`, but transformers 5.x requires torch >= 2.6, so the hook never meets it.
- The wrappers take both call styles of Qwen3.5 / Qwen3.5-MoE / Qwen3-Next (positional) and OLMo-hybrid (`weight=`,
  `bias=`, `activation=` keywords), verified on the 5.20.2 call sites.
- HF's compiled decode with this model was not exercised (a raw CUDA-graph capture of the update was).
- The backward's fp32 partials take B·⌈L/32⌉·C·(W+1)·4 bytes: 0.31x an fp16 x at W = 4.

**The conv alone** (`qwen35_conv_ab.py`: C = 6144, W = 4, SiLU, fp16, the model's layout; ours vs transformers'
reference; the floor is one read of x and one write of y at the nominal bandwidth):

| GPU | forward, L = 4096 | ours vs floor (L = 1k-16k) | fwd + bwd (dX), L = 4096 | update, per call |
|---|---|---|---|---|
| P100 | 3.02 -> 0.25 ms (12.1x) | 1.55-2.16x | 9.6 -> 0.93 ms (10.4x) | 60 -> 8 µs |
| 1080 Ti | 6.26 -> 0.31 ms (20.1x) | 1.48-1.95x | 17.8 -> 2.07 ms (8.6x) | 144 -> 21 µs |
| V100 | 0.98 -> 0.16 ms (6.3x) | 1.28-1.83x | 3.69 -> 0.62 ms (6.0x) | 121 -> 14 µs |
| 2080 Ti | 1.27 -> 0.21 ms (5.9x) | 1.24-1.47x | 4.70 -> 0.80 ms (5.9x) | 163 -> 19 µs |
| T4 | 3.13 -> 0.48 ms (6.6x) | 1.47-1.63x | 11.9 -> 1.76 ms (6.8x) | 129 -> 17 µs |

The forward is at most 2x the floor at 4k on every GPU (1.31-1.81x); at 1k it exceeds that only on the P100 (2.16x).

Forward + backward against its floor (5 passes: the forward's read and write, then dy and x read and dx written):
- At 4k it is 1.95-2.7x on four GPUs without dW, 3.3x on the P100 with dW.
- The 1080 Ti is at 4.0x without dW and 4.4x with it: its backward is slower than the bandwidth suggests.
- So the plan's "forward + backward ≤ 3x" holds without dW except on the 1080 Ti.

**In the model** (`qwen35_conv_model_ab.py`: Qwen3.5-2B fp16, our attention; prefill 4096, generate 2048 + 64, a LoRA
forward + backward at 1024; one process per run, A = `FLASH_ATTN_LEGACY_CONV1D=0`, B = our conv, alternated A B B A
A B B A, medians):

| GPU | prefill | generate | LoRA step |
|---|---|---|---|
| P100 | 1920 -> 1867 ms (1.029x) | 2534 -> 2438 ms (1.039x) | 2300 -> 2252 ms (1.022x) |
| V100 | 369 -> 356 ms (1.038x) | 2619 -> 2472 ms (1.060x) | 1113 -> 1110 ms (1.004x, within noise) |
| 2080 Ti | 529 -> 508 ms (1.041x) | 3356 -> 3186 ms (1.053x) | 1452 -> 1425 ms (1.019x) |
| T4 | 1186 -> 1139 ms (1.041x) | 3507 -> 3359 ms (1.044x) | 1736 -> 1686 ms (1.030x) |

No fallback was counted in either arm. Prefill and generate ranges do not overlap on any of the four GPUs.

The 1080 Ti could not be measured. Its GPU was shared with another tenant's process (4.4 GiB, outside our
container) during these runs: the LoRA step ran out of memory, and the prefill / generate times varied by 40-80%
between runs, far more than the effect.

**Tests** (`tests/test_causal_conv1d.py`).
- Forward: every dtype, W = 2..4, fp32 and x-dtype weights, L < W, C not a multiple of 8, and layouts (a slice, a
  misaligned view). Within one rounding of fp64 and no worse than the reference.
- Backward: against fp64 and the reference, fp32 weights included, reproducible. The dW pass runs only when needed.
- Update: against the forward (bitwise in fp32) and transformers' function (the state bitwise when it has x's dtype),
  in place, under CUDA-graph capture.
- opcheck and a compiled forward + backward.
- Qwen3.5 tiny model: prefill, 64 greedy tokens until a near-tie, and gradients against the reference path; padding
  with gradient checkpointing; the opt-out; the fallback count; packed sequences raising.
- The other hybrid models' call styles, and a bound package kept on SM 7.5 and newer.
