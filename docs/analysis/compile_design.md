# torch.compile and CUDA graphs (P6.5)

Status: done (steps a-d), 2026-10-22. The code is the reference.

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
   extension: `forward`, `backward`, `forward_varlen`, `backward_varlen`, `with_kvcache` (at P6.5 time; `kvcache_append`
   added in P6.9). Python calls `torch.ops.flash_attn_legacy.*`;
   fake (meta) implementations in Python with `torch.library.register_fake` (shapes only). Measured on T4: a Python
   `torch.library.custom_op` adds ~32 us of host time per call (11.7 -> 43.7 us for a decode call), the dispatcher entry of a
   C++ op ~1-2 us. The pybind entries of the kernels are removed (one way to call them); pybind keeps the helpers
   (`compiled_archs`, `decode_max_rows`, `num_splits`).
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
4. **Mask builder without host reads for compiled decode steps.** transformers 5.x already disables the mask skip for
   compileable caches (StaticCache) at q_length == 1, in eager and compiled mode: the builder then returns SDPA's 4D
   boolean mask (causal, padding, window, built without host reads), the cache is not sliced (static shapes), and
   decision 3 runs it on the decode kernel (key mask, cache_seqlens from the last kept key). What broke the graph there
   was the builder's `from transformers import masking_utils` (an import of transformers' lazy module inside traced
   code): it is imported in `register()`. For the other compiled decode steps (DynamicCache, no cache) the builder
   returns SDPA's 4D mask too under tracing, instead of reading `bool(pad.all())`. Prefill keeps the slicing of a static
   cache (one `int(q_offset)` read); q_len > 1 under tracing (compiled training with an attention_mask, chunked prefill)
   still reads the mask on the host (graph break), as upstream FA2's unpad.

Out of scope: the varlen `validate=True` sync (callers that build cu_seqlens pass `validate=False`, as HF does); the
unpad plan of padded prefills; `_mask_window`'s value read when its tag is lost (models pass `sliding_window`).

## Steps (one change each: tests on all GPUs + review)
a. Native op registration + fake implementations + traceable wrappers. Tests: `torch.library.opcheck` on every op;
   `torch.compile(fullgraph=True)` of flash_attention (fwd+bwd), flash_attn_varlen_func (validate=False) and
   flash_attn_with_kvcache, equal to eager; host overhead of a decode call unchanged.
b. Decode key mask in the kernel; HF decode path without host syncs.
c. Mask builder without host reads under tracing / static caches.
d. Measure (probe above, end-to-end generate on real models: results on V100 / T4 / 2080 Ti, inductor does not run on
   Pascal), docs.

## Results (step c, T4, tiny Llama, B=2, 64 new tokens, one StaticCache reused across calls)
| | eager | compiled (HF generate, inductor) |
|---|---|---|
| sdpa | 254 ms | 102 ms |
| flash_attn_legacy | 259 ms | 103 ms |

No graph break in a compiled decode step (as SDPA), logits within fp16 noise of eager. The gain comes from step b (the
static cache's 4D decode mask on the kernel, no host sync) and the import moved out of the builder. Note: `generate` with
`cache_implementation="static"` allocates a new cache per call and recompiles on every call (guards on the cache tensors'
identity, SDPA alike): pass a StaticCache as `past_key_values` and reuse it. Inductor needs Triton (SM >= 7.0): on Pascal
compile with `backend="aot_eager"` or run eager.

## Results (step d, real models: `benchmarks/models.py`, `docs/bench/models.md`)
Decode tokens/s with a reused StaticCache and HF's compiled decode step (inductor), 128 new tokens:

| | V100 ours | V100 SDPA | V100 eager | T4 ours | T4 SDPA | T4 eager |
|---|---|---|---|---|---|---|
| TinyLlama 1.1B, B=1, 2k | 165.3 | 86.8 | 139.6 | 91.2 | 60.4 | 68.5 |
| TinyLlama 1.1B, B=8, 2k | 973.5 | 338.5 | 495.9 | 547.4 | 171.2 | OOM |
| Qwen2.5-3B, B=8, 2k | 471.7 | 263.1 | 273.0 | 224.6 | 106.1 | 103.3 |
| Qwen2.5-3B, B=1, 8k | 70.8 | 25.1 | 46.4 | 32.5 | OOM | OOM |
| Gemma 2 2B, B=1, 8k | 70.5 | 27.1 | 54.3 | - | - | - |

Every compiled step is one graph with no break and no fallback (dynamo counters and fallback statistics in the JSON lines).
Compiled vs eager steps with our kernel (V100): TinyLlama B=1 2k 52.0 -> 165.3 tok/s, Gemma 2 8k 31.9 -> 70.5. On the RTX 2080 Ti (11 GB) the compiled
steps run 1.04-1.31x where SDPA / eager fit, and at B=8 only ours fits (TinyLlama 2k 1,080 tok/s, Qwen2.5-3B 2k 477). The gain over
SDPA / eager grows with the context and the batch: the decode kernel splits the keys of a row over blocks and reads the cache
once per KV head (GQA rows share it), while SDPA's memory-efficient / math kernels on these GPUs and eager's two matmuls with
the 4D mask run a single query row with little parallelism over the keys and read K / V once per query head.

## Acceptance
- No graph break from flash_attn_legacy in a compiled decode step with a static cache (`torch._dynamo` counters), and a
  compiled static-cache generate not slower than the eager one on the probe model and on a real model.
- Compiled training step: no graph break from the attention calls (without padding).
- Eager host time per call: +6 us per kernel call (dispatcher), accepted; all suites green on the 5 GPUs (V100, T4, P100, 1080 Ti, 2080 Ti).
