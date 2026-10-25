# flash_attn_legacy as a vLLM attention backend: feasibility study

Date: 2026-10-23. Our repo: branch `p2-turing` at `68aa3dd`.
vLLM reference: release **v0.32.0** (tag published 2026-10-18; main was at `e879223b` on 2026-10-23). Every vLLM
source link below is pinned to the tag:
`https://github.com/vllm-project/vllm/blob/v0.32.0/<path>#L<n>`. Nothing in the repo was modified and nothing was installed or run on a GPU.

---

## 0. Verdict

- **Feasible, without forking vLLM, but only on Turing (sm_75: T4, 2080 Ti).** v0.32 has an official out-of-tree hook:
  a `vllm.general_plugins` entry point calls `register_backend(AttentionBackendEnum.CUSTOM, "<our class path>")`, and the
  user selects it with `--attention-backend CUSTOM`. Our decode kernel already reads a paged cache with a block table, and
  it handles vLLM's strided K and V views.
- **Probably not worth a full integration now.** Upstream vLLM does not run on three of our four architectures:
  - Volta (sm_70) was dropped in v0.20.0.
  - Pascal (sm_6x) was dropped long before. The V1 engine also depends on Triton, which does not run there.
  - PyTorch 2.15 drops sm_50, sm_60 and sm_70 from its wheels.

  Our unique value is Pascal and Volta, so in upstream vLLM only the T4/2080 Ti case is left. There we would compete with
  vLLM's tuned `TRITON_ATTN` backend. Nobody has measured that comparison, so the benefit is unknown.
- **Recommendation:** run a 1–2 day kernel-level benchmark first (Phase 0) against vLLM's `unified_attention` Triton
  kernel on T4. Build the plugin (Phase 1, roughly 1 week) only if we beat it clearly, say by at least 1.3x on decode or on
  chunked prefill. Do not pursue a Pascal/Volta vLLM fork: that is a separate product and its maintenance cost is very large.

---

## 1. GPUs vLLM supports today, and the attention backend it picks

### 1.1 Minimum compute capability

| Fact | Source |
|---|---|
| Installation docs: "GPU: compute capability **7.5** or higher (e.g., T4, RTX20xx, A100, ...)" | [docs/getting_started/installation/gpu.cuda.inc.md#L9](https://github.com/vllm-project/vllm/blob/v0.32.0/docs/getting_started/installation/gpu.cuda.inc.md#L9) |
| The same file said "compute capability **7.0** or higher (e.g., V100, ...)" up to v0.19.0 and changed to 7.5 in **v0.20.0** (checked by diffing the raw file at tags v0.15.0, v0.18.0, v0.19.0 and v0.20.0) | [v0.19.0](https://github.com/vllm-project/vllm/blob/v0.19.0/docs/getting_started/installation/gpu.cuda.inc.md), [v0.20.0](https://github.com/vllm-project/vllm/blob/v0.20.0/docs/getting_started/installation/gpu.cuda.inc.md) |
| CMake `CUDA_SUPPORTED_ARCHS`: with CUDA ≥ 12.8 the list starts at `7.5`. Only the fallback for CUDA < 12.8 still lists `7.0`. **No 6.x anywhere.** | [CMakeLists.txt#L114-L132](https://github.com/vllm-project/vllm/blob/v0.32.0/CMakeLists.txt#L114-L132) |
| Official wheels are built with CUDA 12.9 by default; cu128 and cu130 variants also exist. None of these builds contains sm_70. | [gpu.cuda.inc.md#L53](https://github.com/vllm-project/vllm/blob/v0.32.0/docs/getting_started/installation/gpu.cuda.inc.md#L53) |
| v0.32 pins `torch==2.13.0` | [requirements/cuda.txt](https://github.com/vllm-project/vllm/blob/v0.32.0/requirements/cuda.txt) |
| Pre-Ampere dtypes: fp16 and fp32 only, no bf16 ("Pascal, Volta and Turing ... BF16 is not supported") | [vllm/platforms/cuda.py#L256-L262](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/platforms/cuda.py#L256-L262) |
| A user forum thread reports that vLLM 0.20 does not install or run on V100 and suggests pinning ≤ 0.18 (a community answer, not an official one) | [discuss.vllm.ai/t/2605](https://discuss.vllm.ai/t/support-for-v100-sm-70-on-vllm-0-20/2605) |

The wider stack also excludes these GPUs:
- **Triton:** its README says "NVIDIA GPUs (Compute Capability **8.0+**)"
  ([triton README#L294-L300](https://github.com/triton-lang/triton/blob/main/README.md), main at `3b0d4f80`, 2026-10-23).
  In practice Triton runs on sm_75 (vLLM ships it there), and PyTorch Inductor refuses devices below 7.0. Our own docs
  note "inductor needs Triton", i.e. SM ≥ 7.0 (`docs/bench/models.md`, "generate_static_compiled").
- **PyTorch:** a notice dated 2026-09-04 says PyTorch 2.15 stops publishing CUDA 12.6 wheels, so it drops Maxwell
  sm_50, Pascal sm_60 and Volta sm_70. 2.14 is the last release with CUDA 12.x wheels. "CUDA 13.x does not support compute
  capabilities below Turing (sm_75)"
  ([dev-discuss.pytorch.org/t/3432](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432)).
  The cu128/cu129 wheels had already dropped sm_50 and sm_60 in PyTorch 2.8.

So **upstream vLLM v0.32 supports only T4 and 2080 Ti (sm_75) among our targets.** V100 needs vLLM ≤ 0.19 or a fork.
Pascal needs a fork.

### 1.2 The backend vLLM picks on T4 and V100

- The CUDA non-MLA priority list (outside SM100) is `FLASH_ATTN, FLASHINFER, TRITON_ATTN, FLEX_ATTENTION, TURBOQUANT`
  ([cuda.py#L171-L182](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/platforms/cuda.py#L171-L182)). The first
  backend that passes `validate_configuration` wins
  ([cuda.py#L375-L410](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/platforms/cuda.py#L375-L410)).
- `FLASH_ATTN` requires `capability >= 8.0`
  ([backends/flash_attn.py#L453-L454](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/attention/backends/flash_attn.py#L453-L454)).
- `FLASHINFER` is "supports SM75+, but is currently broken on SM75", so its floor is temporarily raised to 8.0
  ([backends/flashinfer.py#L571-L577](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/attention/backends/flashinfer.py#L571-L577)).
- **Result: on T4 and 2080 Ti vLLM selects `TRITON_ATTN`.** Its `supports_compute_capability` always returns True
  ([backends/triton_attn.py#L381-L382](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/attention/backends/triton_attn.py#L381-L382)).
  Its builder declares `AttentionCGSupport.ALWAYS`, which means full CUDA graphs even for mixed prefill+decode batches
  ([triton_attn.py#L92](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/attention/backends/triton_attn.py#L92)).
  It also supports sinks, sliding window, non-causal attention, fp8 KV, fp32, and head sizes ≥ 32
  ([triton_attn.py#L297-L382](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/attention/backends/triton_attn.py#L297-L382)).
  `FLEX_ATTENTION` is the next fallback and is also Triton-based (`torch.compile(flex_attention)`,
  [flex_attention.py#L55-L58](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/attention/backends/flex_attention.py#L55-L58)).
- **Removed backends:** XFORMERS no longer exists in `AttentionBackendEnum`. `TORCH_SDPA` is kept "only used for ViT"
  ([backends/registry.py#L34-L158](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/attention/backends/registry.py#L34-L158), L70).
- **V100** (vLLM ≤ 0.19 or a fork): the only option was the Triton or Flex path. The active V100 fork 1Cat-vLLM ships its
  own "Flash-V100" attention extension (see §4).

---

## 2. vLLM V1 attention backend interface (v0.32) and how to plug in a third-party backend

### 2.1 Classes

| Piece | What it must provide | Source |
|---|---|---|
| `AttentionBackend` (ABC) | `get_name`, `get_impl_cls`, `get_builder_cls`; capability class methods: `supported_dtypes`, `supported_kv_cache_dtypes`, `get_supported_kernel_block_sizes` (e.g. `[MultipleOf(16)]`), `supports_head_size`, `supports_sink`, `supports_sliding_window`, `supports_non_causal`, `supports_compute_capability`, `supports_attn_type`, `supported_kv_cache_layouts`; and `forward_includes_kv_cache_update` (False means a separate `do_kv_cache_update` call) | [vllm/v1/attention/backend.py#L59-L377](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/attention/backend.py#L59-L377) |
| `CommonAttentionMetadata` | `query_start_loc` (= cu_seqlens_q, [B+1]), `seq_lens` ([B], context + new tokens = seqused_k), `max_query_len`, `max_seq_len`, `block_table_tensor` [B, max_blocks], `slot_mapping`, `causal` | [backend.py#L387-L415](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/attention/backend.py#L387-L415) |
| `AttentionMetadataBuilder` | `build(...)` -> per-layer metadata; `_cudagraph_support` (`NEVER` / `UNIFORM_SINGLE_TOKEN_DECODE` / `UNIFORM_BATCH` / `ALWAYS`); optional `reorder_batch_threshold` (decodes first) | [backend.py#L554-L760](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/attention/backend.py#L554-L760) |
| `AttentionImpl` | `__init__(num_heads, head_size, scale, num_kv_heads, alibi_slopes, sliding_window, kv_cache_dtype, logits_soft_cap, attn_type, ...)`; `forward(layer, query [T,H,d], key, value, kv_cache, attn_metadata, output, ...)` writes into a **preallocated `output`**; `do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)` | [backend.py#L887-L1020](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/attention/backend.py#L887-L1020) |
| Batch split helpers | `split_decodes_and_prefills`, `reorder_batch_to_split_decodes_and_prefills` (batch ordered decode -> short extend -> long extend -> prefill) | [backends/utils.py#L803-L984](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/attention/backends/utils.py#L803-L984) |

**KV cache layout (changed from the old `[2, num_blocks, block_size, H, d]`).** The per-layer cache is logically
`[num_blocks, num_kv_heads, block_size, 2 * head_size]`, with K and V side by side in the last dimension. A
`KVCacheLayout` stride permutation chooses the physical order (`LBNHC` = NHD, `LBHNC` = HND, ...)
([vllm/v1/kv_cache_layout.py#L12-L28](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/kv_cache_layout.py#L12-L28)).
Backends take views with `kv_cache.transpose(1, 2).split(head_size, dim=-1)`, which gives K and V as
`[num_blocks, block_size, H_kv, d]` strided views
([triton_attn.py#L569, L625-L627, L796](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/attention/backends/triton_attn.py#L625-L627);
the same in [flash_attn.py#L1252, L1443](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/attention/backends/flash_attn.py#L1443)).

**Cache write.** vLLM's own ops write the cache:
- `reshape_and_cache_flash` (CUDA, [csrc/cache_kernels.cu#L819](https://github.com/vllm-project/vllm/blob/v0.32.0/csrc/cache_kernels.cu#L819), used by FLASH_ATTN at [flash_attn.py#L1452](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/attention/backends/flash_attn.py#L1452));
- `triton_reshape_and_cache_flash` (used by TRITON_ATTN at [triton_attn.py#L800](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/attention/backends/triton_attn.py#L800)).

A third-party backend can reuse either op: **we do not need our own KV-write kernel.**

**Default block size:** `CacheConfig.DEFAULT_BLOCK_SIZE = 16`
([vllm/config/cache.py#L72](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/config/cache.py#L72)).

**Sliding window as passed to backends:** `(sliding_window - 1, 0)` for decoders, the same convention as ours
([triton_attn.py#L479-L484](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/attention/backends/triton_attn.py#L479-L484)).

**How a mixed batch reaches the backend:** chunked prefill and prefix caching are on by default, so one step can mix
decodes, chunked prefills and fresh prefills. Prefills generally attend to context that is **already in the paged cache**.
The flat queries are `[num_tokens, H, d]`, with `cu_seqlens_q = query_start_loc` and `seqused_k = seq_lens`, and
causal attention is bottom-right aligned per request. TRITON_ATTN handles the whole batch in one
`unified_attention(q, k_cache, v_cache, cu_seqlens_q, seqused_k, block_table, window_size, softcap, sinks, ...)` call
([triton_attn.py#L651-L689](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/attention/backends/triton_attn.py#L651-L689)).

### 2.2 torch.compile and CUDA graphs

- Attention runs inside the custom op `vllm::unified_attention_with_output`, which mutates `output`
  ([model_executor/layers/attention/attention.py#L767-L800](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/model_executor/layers/attention/attention.py#L767-L800)).
  The cache write is a separate op, `unified_kv_cache_update`, used when `forward_includes_kv_cache_update` is False
  ([L724-L760](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/model_executor/layers/attention/attention.py#L724-L760)).
- The attention op is a **splitting op**. In piecewise mode the model graph is cut around it, so attention itself runs
  eagerly and its Python overhead counts
  ([config/compilation.py#L768-L769, L1155-L1200](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/config/compilation.py#L768)).
  The comment at [triton_attn.py#L591-L597](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/attention/backends/triton_attn.py#L591-L597)
  warns that even `view` and slicing cost time there.
- The default `cudagraph_mode` is `FULL_AND_PIECEWISE`: full graphs for decode batches, piecewise for the rest
  ([compilation.py#L610-L640](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/config/compilation.py#L610-L640)).
  For a full decode graph the attention op must be capturable:
  - no host syncs;
  - shape-stable allocations;
  - metadata in persistent buffers.

  A builder with `UNIFORM_SINGLE_TOKEN_DECODE` gets full graphs for pure decode batches only.

### 2.3 Plugging in without a fork (what exists in v0.32)

| Mechanism | Status in v0.32 | Source |
|---|---|---|
| `register_backend(AttentionBackendEnum.CUSTOM, "pkg.mod.Class")`, or `@register_backend("NAME")` for a new dynamically created enum member. It can also **override** an existing member (e.g. `TRITON_ATTN`). | exists | [backends/registry.py#L156-L190, L336-L412](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/attention/backends/registry.py#L336-L412) |
| Selection: `--attention-backend CUSTOM`, `--attention-config.backend`, `LLM(attention_backend="CUSTOM")` / `AttentionConfig(backend=...)`, per-KV-group overrides `backend_per_kind` | exists | [docs/design/attention_backends.md#L8-L57](https://github.com/vllm-project/vllm/blob/v0.32.0/docs/design/attention_backends.md), [config/attention.py#L35-L56, L183-L195](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/config/attention.py#L183-L195), [engine/arg_utils.py#L1086](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/engine/arg_utils.py#L1086) |
| `VLLM_ATTENTION_BACKEND` env var | **no longer exists** in `vllm/envs.py` v0.32 (replaced by the options above) | [vllm/envs.py](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/envs.py) (no match) |
| General plugins: entry point group `vllm.general_plugins`, "loaded in all processes (process0, engine core process and worker processes)". This is the right place to call `register_backend`. | exists | [vllm/plugins/__init__.py#L16-L18, L77-L90](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/plugins/__init__.py#L16-L18) |
| Platform plugins: `vllm.platform_plugins`, for an out-of-tree **device** (`PlatformEnum.OOT`) that overrides `get_attn_backend_cls`. Overkill here: we stay on the CUDA platform. | exists | [docs/design/plugin_system.md#L50-L130](https://github.com/vllm-project/vllm/blob/v0.32.0/docs/design/plugin_system.md) |
| Legacy `_Backend` in `vllm.attention`: removed in v0.13.0, replaced by `register_backend` | removed | [plugin_system.md#L157-L158](https://github.com/vllm-project/vllm/blob/v0.32.0/docs/design/plugin_system.md) |

`CUSTOM` is not in the automatic priority list (§1.2), so users must select it explicitly. The only way to make it
automatic is to override `TRITON_ATTN` itself, which is too intrusive to recommend.

---

## 3. Gap analysis: what vLLM needs vs what we have

**What we already have:**
- `flash_attn_with_kvcache`: decode on a contiguous or paged cache.
  - Wrapper: `flash_attn_with_kvcache` in `flash_attn_legacy/flash_attn.py`.
  - Op: `flash_attn_with_kvcache` in `csrc/flash_attn_legacy.cpp`.
  - Registered as `torch.ops.flash_attn_legacy.with_kvcache` with a fake implementation (`TORCH_LIBRARY` block in `cpp`, `_register_fakes` in `flash_attn.py`).
- `flash_attn_varlen_func`: packed varlen forward and backward on **contiguous** K/V (`flash_attn_varlen_func` in `flash_attn.py`, `flash_attn_forward_varlen` in `cpp`).
- The HF integration (`register()` in `flash_attn_legacy/integrations.py`) is HF-only. There is no vLLM code in the repo.
  `docs/analysis/kernels/decode_design.md` ("Why") already lists a vLLM backend as a separate PLAN item.

| # | vLLM requirement | Ours | Evidence |
|---|---|---|---|
| 1 | **Decode on the paged cache** (q_len 1, block_table [B, max_blocks], seqused_k per request, block size 16) | **Have.** Paged `[num_blocks, block_size, H_kv, d]`, int32 block_table, block_size any multiple of 16, any strides with a contiguous last dim and 16-byte rows. A tensor `cache_seqlens` is clamped on the GPU, with no sync. | `check_paged_cache` in `cpp` (paged checks, block size `% 16`), `check_decode_layout` in `cpp` (stride rules), `decode_kernel` in `csrc/kernels/flash_attn_decode.cu` (stride-generic paged addressing, `key_offsets`), `flash_attn_with_kvcache` docstring in `flash_attn.py` |
| 2 | Read vLLM v0.32's **K and V as `transpose(1,2).split(d, -1)` views** of `[nb, H, bs, 2d]` | **Partial.** The stride-generic addressing should accept them (token stride 2d or H·2d, both multiples of 8 for d % 8 == 0, V pointer offset 2d bytes, 16-byte aligned). The tests only cover NHD, HND and the **old** `[2, nb, ...]` vLLM layout. | `tests/test_flash_attn.py`: `TestKVCache._paged` (`layout="vllm"` = `[2, n_blocks, bs, H, d]`), `test_paged_layouts`, `test_forward_rows_paged` |
| 3 | Causal **bottom-right** per request, sliding window `(W-1, 0)`, softcap, sinks `[H_q]`, GQA/MQA | **Have**, in both decode and varlen | `flash_attn_with_kvcache` in `flash_attn.py`; `KeyWindow` in `csrc/include/flash_attn_common.h`; varlen per-sequence lengths in `fwd_varlen_kernel` (`varlen_seq_of_block`, `cu_seqlens_k`) in `csrc/include/flash_attn_fwd_band.h` |
| 4 | **Chunked/prefix prefill: varlen Q against K/V read from the paged cache** (cu_seqlens_q + seqused_k + block_table) | **Missing.** `forward_varlen` takes packed contiguous K/V `[total_k, H_kv, d]` (`contiguous_aligned`) and cu_seqlens_k. A paged-prefill kernel does not exist. Stopgap: gather K/V from the cache into a contiguous buffer (torch indexing with block_table) and call `forward_varlen(validate=False)`. That is correct but costs an extra read and write of the whole context per prefill step and a variable-size allocation. | `flash_attn_forward_varlen`, `contiguous_aligned`, `check_varlen_inputs` in `cpp`; `flash_attn_varlen_func` in `flash_attn.py` |
| 5 | **Mixed prefill+decode in one batch** | **Partial.** No unified kernel. The workaround is to split with vLLM's `split_decodes_and_prefills` (decode -> `with_kvcache`, rest -> item 4), so 2+ launches per layer. | — |
| 6 | **Speculative or multi-token decode, large GQA groups** | **Have (P6.8).** At the time: **Partial**, the decode kernel took a uniform `q_len` with `(H_q/H_kv)·q_len ≤ 8` rows per KV head and raised NotImplemented above that. Since P6.8, rows > 8 (`DECODE_MAX_ROWS`) go to the forward kernels with key splits and packed GQA rows on the same cache (paged too), behind the same `flash_attn_with_kvcache`. | `flash_attn_with_kvcache` in `cpp` (`on_forward`), `flash_attn_decode_max_rows` |
| 7 | **Write into vLLM's preallocated `output`** | **Missing (minor).** The op allocates `out` (`torch::empty_like(q)`), so the backend must `output.copy_()` (an extra small copy). An `out=` argument is easy to add. | `flash_attn_with_kvcache` in `cpp` |
| 8 | **KV cache write** (`slot_mapping`) | **Not needed.** Reuse vLLM's `reshape_and_cache_flash` or `triton_reshape_and_cache_flash` (§2.1). Since P6.9 step a our own `kvcache_append` op also exists (`flash_attn_with_kvcache`'s `k` / `v`: new keys written at `cache_seqlens`, paged too; indexed by `cache_seqlens`, not by `slot_mapping`). | `flash_attn_kvcache_append` in `cpp` |
| 9 | **CUDA graph capture of decode** | **Partial / likely OK.** Our design avoids host syncs with tensor `cache_seqlens` and automatic `num_splits` (`flash_attn_num_splits`, computed from shapes); outputs and split partials are plain allocations. Graph capture was proven only through HF's compiled static-cache decode at the time (`docs/analysis/compile_design.md`, README "Benchmarks", compiled static-cache row); since P6.9 the paged decode (with the `kvcache_append` write) also runs inside transformers' continuous-batching CUDA graphs (`docs/bench/serving.md`). Never in vLLM. With item 4's gather, prefill is **not** capturable, so declare `UNIFORM_SINGLE_TOKEN_DECODE` (vs Triton's `ALWAYS`). | `compile_design.md` "Decisions" 1-3 |
| 10 | **Host overhead in piecewise mode** | **Partial.** The Python wrapper adds about 28 µs per call and the dispatcher op about 18 µs per decode call on T4 (`compile_design.md` decision 2). The backend should call `torch.ops.flash_attn_legacy.*` directly and skip the wrapper's validation. | `docs/analysis/compile_design.md` ("~28 us on top") |
| 11 | dtypes: fp16 (bf16 does not exist below SM80 in vLLM; fp32 models allowed) | **Have fp16.** Decode reads fp16 or bf16 natively (`flash_attn_with_kvcache` in `cpp`). The varlen op is fp16 only (`check_varlen_inputs` in `cpp`); bf16/fp32 go through the Python scaling (`_to_fp16` / `_scaled_half` in `flash_attn.py`). **fp32 models are missing** (Triton supports them). | — |
| 12 | Head sizes | **Partial.** 32, 64, 72, 80, 96, 128, 256 only (TRITON_ATTN: any ≥ 32). That covers Llama, Qwen, Mistral, Gemma and gpt-oss. | `check_head_dim` in `cpp`, `SUPPORTED_HEAD_DIMS` in `flash_attn_legacy/utils.py` |
| 13 | fp8 KV cache (`fp8_e4m3/e5m2`) | **Missing.** Optional: T4/V100 have no fp8 hardware, but fp8 KV halves decode bandwidth and Triton supports it. | — |
| 14 | ALiBi | **Missing** (`alibi_slopes` raises NotImplementedError). Only for old models (BLOOM, MPT). | `flash_attn_with_kvcache` in `flash_attn.py` (unsupported arguments) |
| 15 | Encoder / encoder-only / cross attention (no cache) | **Have** through `forward_varlen` (non-causal, contiguous Q/K/V) | `flash_attn_forward_varlen` in `cpp` |
| 16 | Cascade attention, DCP/PCP, MM-prefix, R-SWA, per-token-head quant KV | **Missing.** All optional: declare unsupported (`supports_dcp` etc. default False). | — |
| 17 | `cache_leftpad` + paged | Not needed by vLLM (it is rejected by us, as by upstream FA2) | `flash_attn_with_kvcache` in `cpp` |

**Key gaps, in order of importance:**
1. A paged varlen prefill kernel (item 4).
2. A unified mixed batch, i.e. full CUDA graphs beyond pure decode (items 5 and 9).
3. ~~The decode row limit for spec decode and large GQA groups (item 6).~~ Resolved by P6.8.
4. Tests for vLLM v0.32's interleaved K|V cache views plus an `out=` argument (items 2 and 7).
5. fp32 and fp8 KV, if parity with TRITON_ATTN is wanted (items 11 and 13).

---

## 4. Value

**Pascal (P100, 1080 Ti), our strongest differentiator (README kernel-level table; SDPA on sm_61 is 10-15x slower):**
upstream vLLM cannot run here at all.
- No 6.x in CMake (§1.1).
- V1 needs Triton for sampling and other ops, for `triton_reshape_and_cache_flash`, and for Inductor; Triton does not support sm_6x.
- PyTorch 2.15 drops sm_60.

Community forks:
- **sasha0552/pascal-pkgs-ci** built Pascal vLLM up to **v0.10.0** with a patched Triton. Its last push was 2025-08-23,
  and it says "Support for new GPUs has been disabled (v0.7.0+/main)". The repo is stale
  ([README](https://github.com/sasha0552/pascal-pkgs-ci); repo API `pushed_at` 2025-08-23).
- The older `sasha0552/vllm-ci` is archived.
- A Docker image `johnnykings/vllm-pascal` exists ([hub](https://hub.docker.com/r/johnnykings/vllm-pascal)), but its
  version is unknown.

A Pascal vLLM would mean maintaining a fork with:
- CMake arch patches;
- Triton-free paths everywhere (CUDA `reshape_and_cache_flash` exists, but the sampler, MoE and other ops are Triton);
- `enforce_eager`;
- pinning to PyTorch ≤ 2.14.

That is an **XL ongoing effort** for a shrinking install base. Not recommended.

**Volta (V100):** upstream support ended at v0.19 (§1.1). The active fork **1Cat-vLLM** ("V100 / SM70-focused vLLM
engineering fork", 1,378 stars, pushed 2026-10-23) already ships a dedicated "Flash-V100" attention extension:
- fp8 E4M3 KV;
- CUDA graphs;
- long-context causal attention at about 60-71 TFLOP/s per V100;
- release v1.5.1 (2026-10-03) on CUDA 12.8 / torch 2.10.

([README](https://github.com/1CatAI/1Cat-vLLM/blob/main/README.md)). Our V100 forward is 0.66-1.06x of SDPA mem-efficient
(README kernel-level table), so we would not beat a specialized V100 serving kernel. At most we could offer our kernels to that fork.

**Turing (T4, 2080 Ti), the only target upstream vLLM supports:** the competitor is TRITON_ATTN's `unified_attention`, a
tuned kernel with full CUDA graphs for mixed batches, fp8 KV and spec decode.
- Our numbers so far compare against PyTorch SDPA or HF, not vLLM.
  - T4 kernel forward is 0.54-0.59x the time of SDPA mem-efficient (README kernel-level table).
  - T4 decode reaches about 200 GB/s of 320 (PLAN P6.1).
  - Compiled static-cache decode is 2.1-3.2x over HF SDPA (README "Benchmarks" table).
- **No public T4 or Turing measurement of vLLM's Triton backend** turned up (vLLM's Triton deep-dive blog,
  [vllm.ai/blog/vllm-triton-backend-deep-dive](https://vllm.ai/blog/vllm-triton-backend-deep-dive), and
  [arXiv 2511.11581](https://arxiv.org/pdf/2511.11581) cover newer GPUs).
- Triton's official floor is CC 8.0 (§1.1), so its sm_75 code generation is less tuned. That is a plausible but
  **unverified** opening for us.
- Decode is bandwidth-bound, and a well-written Triton kernel may already be near our roughly 60% of peak. Prefill on
  T4 (mma.sync m16n8k8) is where a hand-written FA2 is most likely to win.
- End to end, attention is only part of the step time (weights dominate decode at short contexts), so even a 1.5x
  kernel win would give a smaller serving gain, mostly at long contexts and large batches.

**Conclusion on value:** the integration is low-to-moderate value and unproven on T4, and has no value for Pascal and
Volta in upstream vLLM, which is where our library differentiates. Measure before building.

---

## 5. Effort and minimal plan

| Phase | Content | Size | Gate |
|---|---|---|---|
| **0. Measure (do first)** | On T4 and 2080 Ti with vLLM v0.32 installed in a venv: microbench `vllm.v1.attention.ops.triton_unified_attention.unified_attention` vs our `with_kvcache` (pure decode, B 1-64, ctx 1k-32k, GQA 32/8, 28/4, 64/8, d 64/128, block 16) and vs gather + `forward_varlen` (chunked prefill 512-2048 new tokens on 0-16k context). Same paged tensors built with v0.32's `[nb, H, bs, 2d]` views (this also closes gap #2 for decode). | S (1-2 d) | Continue only if ≥ 1.3x on decode, or on prefill including the gather. |
| **1. Out-of-tree plugin, decode-first** | Package `flash_attn_legacy.vllm` (optional extra) with these parts. **Validation:** fp16; head sizes; `MultipleOf(16)`; kv dtype `auto`; `supports_sink`/`supports_sliding_window` True; cc ≥ 7.5 (6.x/7.0 unreachable upstream anyway). **Entry point:** `vllm.general_plugins` -> `register_backend(AttentionBackendEnum.CUSTOM, ...)`. **Builder:** `reorder_batch_threshold=1`, `split_decodes_and_prefills`, `UNIFORM_SINGLE_TOKEN_DECODE`, persistent seq_lens/block_table buffers. **Impl:** `do_kv_cache_update` = vLLM `reshape_and_cache_flash`; decode -> `torch.ops.flash_attn_legacy.with_kvcache` on `q.view(B,1,H,d)`; prefill/extend -> gather + `forward_varlen`. Add an `out=` argument to the op (gap #7). Correctness tests against TRITON_ATTN on small Llama/Qwen/Gemma 2/gpt-oss, plus `lm_eval` spot checks. | M (~1 wk) | Same tokens as TRITON_ATTN (greedy); throughput ≥ TRITON_ATTN on `vllm bench serve`. |
| **2. Native paged prefill** | A block_table path in the three forward kernels' K/V tile loaders (Pascal, Volta, Turing), with Q varlen (cu_seqlens_q) and K length = seqused_k. This removes the gather and makes prefill capturable. Then optionally one unified launch for decode + prefill -> `AttentionCGSupport.ALWAYS`. | L (2-3 wk incl. tests on all boxes) | Prefill gain over the Phase 1 gather. |
| **3. Parity extras** | ~~Decode rows > 8 (spec decode, GQA 16)~~ (done: P6.8), fp32 models, fp8 KV dequant in the decode kernel, ALiBi | M each | Only on user demand |
| **(Not recommended) Pascal/V100 fork** | Fork vLLM ≤ 0.19 (V100) or patch CMake + remove Triton paths (Pascal) on PyTorch ≤ 2.14 | XL + continuous | — |

**Risks:**
- **Interface churn:** vLLM's attention API changes often. v0.13 removed `_Backend`; the KV cache moved to `[nb, H, bs, 2d]`
  with `KVCacheLayout` permutations; `VLLM_ATTENTION_BACKEND` is gone. A plugin needs pinning to a vLLM range and a CI job
  per release.
- **Small target market:** only sm_75 in upstream. PyTorch 2.15 and CUDA 13 make sm_75 the oldest supported architecture,
  so it is next in line to be dropped.
- **Losing on T4:** we may not beat Triton there (unmeasured). Our kernels stand out most where Triton does not run at all.
- **Features and defaults:** prefix caching and chunked prefill are the defaults, so the gather stopgap could be slow at
  long contexts (extra HBM traffic on a 320 GB/s T4). Phase 2 is needed for a fair prefill comparison.
- **CUDA graphs:** with `UNIFORM_SINGLE_TOKEN_DECODE` only, mixed batches run piecewise. Python overhead per layer
  (dispatcher about 18 µs) can dominate small models.
- **Coverage:** no fp8 KV, ALiBi, fp32, cascade or DCP. These must be declared unsupported so vLLM refuses cleanly
  instead of producing wrong output.
- **Testing:** needs a T4 and a 2080 Ti box with a vLLM install, on top of the current 4-GPU matrix.
