# torch.compile and CUDA graphs (P6.5)

Status: design, 2026-10-22. The code becomes the reference once implemented.

## Where we start (T4, torch 2.7.1, transformers 5.20.2)
Probe: tiny Llama, `generate(cache_implementation="static")` 32 new tokens, B=2, with and without HF's compile.

| backend | eager | compiled |
|---|---|---|
| sdpa | 138 ms | 986 ms |
| flash_attn_legacy | 150 ms | 5108 ms |

(The tiny model is launch bound: the compiled numbers are dominated by guards / recompilation, not by the kernels.)
Graph breaks, most frequent first: the pybind call `decode_max_rows` (16), `data_ptr() % 16` in `_decode_layout_ok` (8),
the pybind kernels `forward` (7), `locals()` in `flash_attn_with_kvcache` (5), `bool(tensor)` (4: the mask builder's
`pad.all()`, `_decode_padding`), plus `int(q_offset)` of the static cache in the mask builder (data dependent: a new
`n_valid`, i.e. a new shape, every step). A compiled training step works, with breaks around every attention call.

## Decisions
1. **Ops registered natively** with `TORCH_LIBRARY(flash_attn_legacy, m)` + `TORCH_LIBRARY_IMPL(..., CUDA, m)` in the
   extension: `forward`, `backward`, `forward_varlen`, `backward_varlen`, `with_kvcache`. Python calls `torch.ops.flash_attn_legacy.*`;
   fake (meta) implementations in Python with `torch.library.register_fake` (shapes only). Measured on T4: a Python
   `torch.library.custom_op` adds ~32 us of host time per call (11.7 -> 43.7 us for a decode call), the dispatcher entry of a
   C++ op ~1-2 us. The pybind entries of the kernels are removed (one way to call them); pybind keeps the helpers
   (`compiled_archs`, `decode_max_rows`, `decode_num_splits`).
   - Schema types: float -> double, int -> int64_t; `bwd` always returns 4 tensors (dsinks `[0]` fp32 without sinks: no
     undefined tensors in outputs, which fake tensors / functionalization reject).
   - Autograd stays in the `torch.autograd.Function`s (dynamo traces them; their forward / backward call the ops).
   - torch < 2.4 (no `register_fake`): the ops are still registered, compile is unsupported there.
2. **Python wrappers traceable**: no `locals()`; `decode_max_rows` read once at import. The HF decode layout check reads
   the address (`data_ptr()`) in eager only: under tracing neither `data_ptr()` nor `storage_offset()` is traceable (both
   break the graph), so there only the op checks the alignment and raises on a misaligned cache instead of falling back
   (HF caches are allocations or prefix slices of them: aligned).
   - Measured on T4: the dispatcher call costs ~6 us of host time more than the pybind one (17.9 vs 11.7 us per decode
     call; the Python wrapper adds ~28 us on top in both cases).
   - `max_seqlen_q / k` stay `int` in the schema (unused by the kernels): a compiled varlen call specialises on them.
3. **Decode key mask in the kernel** (`key_mask` [B, max_len] bool, keyword-only extension of `flash_attn_with_kvcache`):
   each block shrinks its key split to the first / last kept key (a scan of its own keys) and skips masked keys (no K / V
   read). The HF decode path passes its 2D / 4D mask as it is; with a 4D mask (static cache: empty slots after the query)
   cache_seqlens = last kept key + 1 is computed on the device, so causal / window stay aligned to the query -> no host
   sync, no "padding inside a row" fallback, no per-mask caching (`_decode_padding`, `_DECODE_ATTR` removed).
   - Measured on T4 (GQA 32/8): an all-True mask costs 0% at B=1 16k d=128, +9% at B=8 4k d=128 (the per-key mask
     load), -5% at d=64 R=8; masks are passed only with padding (all-True masks become None before the kernel).
     Tried, no gain, not kept: one row scan per block for global bounds (B=1 +13%), K / V loads not waiting for the mask
     byte (V of masked keys zeroed).
4. **Mask builder without host reads when tracing or with a tensor q_offset** (static caches): `[B, kv_length]` bool key
   mask built on the device (`arange < q_offset + q_length`, & padding), never `int(q_offset)` / `bool(pad.all())`; the
   cache is never sliced (static shapes: no recompilation per step). Decode -> kernel with the mask; q_len > 1 (prefill of a
   static cache, chunked prefill) -> the unpad plan (host sync / graph break: data-dependent shapes, as upstream FA2's
   unpad), acceptable since HF compiles the decode steps only.

Out of scope: the varlen `validate=True` sync (callers that build cu_seqlens pass `validate=False`, as HF does); the
unpad plan of padded prefills; `_mask_window`'s value read when its tag is lost (models pass `sliding_window`).

## Steps (one change each: tests on all GPUs + review)
a. Native op registration + fake implementations + traceable wrappers. Tests: `torch.library.opcheck` on every op;
   `torch.compile(fullgraph=True)` of flash_attention (fwd+bwd), flash_attn_varlen_func (validate=False) and
   flash_attn_with_kvcache, equal to eager; host overhead of a decode call unchanged.
b. Decode key mask in the kernel; HF decode path without host syncs.
c. Mask builder without host reads under tracing / static caches.
d. Measure (probe above, end-to-end generate on a real model on P100 / T4), docs.

## Acceptance
- No graph break from flash_attn_legacy in a compiled decode step with a static cache (`torch._dynamo` counters), and a
  compiled static-cache generate not slower than the eager one on the probe model and on a real model.
- Compiled training step: no graph break from the attention calls (without padding).
- Eager host time per call: +6 us per kernel call (dispatcher), accepted; all suites green on the 4 GPUs.
