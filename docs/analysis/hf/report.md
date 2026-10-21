# flash_attn_legacy vs HF transformers 5.20.2: how attention is dispatched, and where we fall short

Paths are relative to `/workspace/venv/lib/python3.12/site-packages/transformers/`. Line numbers are from 5.20.2.
Evidence comes from two sources: reading the code, and a CPU-only probe in `/workspace/analysis/hf/` (`probe.py`, `scenarios.py`, `scenarios2.py`, `scenarios3.py`).
- The probe reproduces the decision logic of `integrations.py` (mask builder + attention function). A pure-torch stand-in replaces the CUDA kernel: plain bottom-right causal or full attention with GQA, ignoring every extra kwarg, which is what the kernel path does today.
- Each result is compared with `eager` on tiny random models.
- Anything marked UNVERIFIED was not checked.

## 1. The attention-function contract

- **Registry:** `AttentionInterface` (modeling_utils.py:5121-5136). Models look up `config._attn_implementation` and fall back to their own eager function.
- **Signature** (it matches `flash_attention_forward`, integrations/flash_attention.py:27-41):
  `fn(module, query, key, value, attention_mask, dropout=0.0, scaling=None, sliding_window=None, softcap=None, is_causal=None, s_aux=None, **kwargs) -> (out, weights|None)`
- **Inputs:** q/k/v are `[B,H,N,d]`. K/V can have fewer heads than q (GQA).
- **Return:** `out` is `[B,N_q,H,d]`. The model then calls `reshape(*input_shape,-1)` (llama:268; sdpa_attention.py:180-182).
- **`attention_mask`** is whatever the mask builder registered under the same name returned:
  - None
  - a 4D bool mask (sdpa)
  - a 4D float mask (eager)
  - a 2D mask (flash)
  - a BlockMask (flex)
  - a user-supplied 4D mask, passed through unchanged (masking_utils.py:797-798)
  - If no mask builder is registered under the name, HF builds no mask at all (masking_utils.py:805-806).
- **`dropout`** is already 0 when the module is not training (llama:263, gemma2:271).
- **`scaling`** can be None (llama4:809-810).
- **Kwargs:**
  - Model-specific: `sliding_window`, `softcap`, `s_aux`, `is_causal`, `deterministic`.
  - Everything in TransformersKwargs (utils/generic.py:832-871): `position_ids`, `cu_seq_lens_q/k`, `max_length_q/k`, `is_causal`, `seq_idx`, `output_attentions`, and so on.
  - The decoder layer forwards `position_ids` (llama:297-304), so decoder LLMs always pass it.
  - Continuous batching passes `cache` and `block_table` (flash_attention.py:59-65). Out of scope.

## 2. Model family x feature

| family | head_dim | GQA | causal | sliding window | softcap | sinks | extra kwargs | masks |
|---|---|---|---|---|---|---|---|---|
| llama | hidden/heads (128) | y | y (llama:217) | - | - | - | dropout, scaling (llama:257-266) | causal (llama:380) |
| qwen2 | 128 | y | y | if use_sliding_window (qwen2:185) | - | - | +sliding_window (qwen2:220) | causal+sliding |
| qwen3 / Qwen3-Embedding | 128 (cfg:68) | y | y | optional (qwen3:231) | - | - | +sliding_window | causal+sliding (qwen3:395-399) |
| mistral | 128 | y | y | 4096 (cfg:80) | - | - | +sliding_window (mistral:171) | causal/sliding |
| gemma2 | 256, scale=qpas^-0.5 (gemma2:221) | y | not use_bidirectional (gemma2:223) | 4096 on alternate layers (gemma2:238) | **50.0** (cfg:91; passed gemma2:274) | - | +sliding_window, softcap | causal+sliding (gemma2:431-432) |
| gemma3 | 256 | y | not use_bidirectional (gemma3:309) | 4096 by default, many ckpts 512/1024 | None, not passed (gemma3:358-367) | - | +sliding_window | +block_sequence_ids for images (gemma3:720) |
| phi3 | 96 (mini) | y | y | cfg (default None) | - | - | +sliding_window (phi3:246) | causal/sliding |
| gpt_oss | 64 | y | y | 128 on alternate layers | - | **s_aux=self.sinks** (gpt_oss:321) | +sliding_window, s_aux | causal+sliding; _supports_sdpa=False (gpt_oss:379); _compatible_flash_implementations (gpt_oss:390) |
| modernbert | 64 | n | **n** (modernbert:246) | local_attention//2+1 (modernbert:242) on 2 of 3 layers | - | - | +sliding_window, deterministic; no position_ids | bidirectional(+sliding) (modernbert:449-450) |
| bert/roberta | 64 | n | n (bert:161) | - | - | - | dropout = **0.1** in training (bert:198) | bidirectional/causal (bert:697-711) |
| Qwen2-VL/2.5-VL/3-VL/3.5 vision, Pixtral, GLM4V... (41 files) | 80 for Qwen2.5-VL ViT; others UNVERIFIED | n | n | Qwen2.5-VL windows via cu_window_seqlens (qwen2_5_vl:442-473) | - | - | **if "flash" in name:** mask=None + cu_seq_lens_q/k + max_length_q/k (qwen2_vl:406-422); else per-chunk loop (qwen2_vl:424-445) | - |
| llama4 | 128 | y | y | chunked | - | - | vision passes None and is_causal=False (llama4:803-811) | chunked mask **raises for "flash" names** (masking_utils.py:1376-1380) |

FA2 pads V when `v_head_dim != qk_head_dim` (flash_attention.py:72-76). We do not. UNVERIFIED whether our C++ code checks for it.

## 3. The FA2 machinery, and whether we can reuse it

### Mask builder: `flash_attention_mask` (masking_utils.py:600-640)
- It ignores `mask_function`.
- It returns the 2D mask `[:, -kv_length:]`, or None when that mask is all ones and has full length.
- With a StaticCache the mask is shorter than kv_length. That shorter mask is returned on purpose: it tells the attention function to slice k/v (modeling_flash_attention_utils.py:446-449).

### `_flash_attention_forward` (modeling_flash_attention_utils.py:736-858)
It picks one of these paths:
- **Paged** (:815-825). Out of scope.
- **(a) Padding** (:828-835).
  - `_upad_input` (:403-479) calls `_get_unpad_data` (:375-400), which does a `.item()` at :394.
  - q_len==kv_len: q reuses the k indices.
  - q_len==1: `cu_q = arange(B+1)`. This is **(c) decode with a KV cache**.
  - Otherwise: q is unpadded with `mask[:, -q_len:]`, which **assumes left padding** (:468-470).
  - Then `varlen` -> `_pad_input` (:356-372).
- **Explicit varlen kwargs** (:838-843).
- **(b) Packed through position_ids, B==1** (:846-854).
  - `_is_packed_sequence` (:560-576) costs a device sync.
  - `prepare_fa_kwargs_from_position_ids` (:482-519) treats every position equal to `position_ids.min()` as a sequence start.
  - This path also runs when an all-ones attention_mask was passed.
- **Dense** (:857-858). DynamicCache decode lands here.

### Optional features (d) sliding window, (e) softcap, (f) sinks
They are forwarded only if the kernel's signature has them (`_lazy_define_process_function` :245-266; `_process_flash_attention_kwargs` :618-733):
- `window_size=(sw-1, sw-1)`, only when `key_len > sw` (:683-688)
- `softcap` (:695-696)
- `s_aux` or `learnable_sink` (:698-702)
- `deterministic` (:690-693)
- `dropout_p` (:680-681)

**Unsupported features are dropped silently.**

### Packing on the sdpa side
- `find_packed_sequence_indices` (masking_utils.py:714-743) runs only when there is no attention mask and no cache (:840-845).
- When it finds packing, `create_causal_mask` sets `allow_is_causal_skip=False` (:953-955), so sdpa gets a dense 4D block-diagonal mask.
- Encoders never detect packing: `position_ids=None` (:1026-1028).

### Reuse option A: the hub `kernels` mechanism
- `attn_implementation="org/repo[@rev][:fn]"` matches `is_kernel` (hub_kernels.py:811-816).
- The chain is `lazy_import_flash_attention` (modeling_utils.py:1801-1804) -> `load_and_register_attn_kernel` (hub_kernels.py:824-900).
- The module must expose:
  - **`flash_attn_varlen_func(q,k,v,cu_seqlens_q,cu_seqlens_k,max_seqlen_q,max_seqlen_k,softmax_scale,causal,[dropout_p,window_size,softcap,deterministic,s_aux])`**, with layout `[total,H,d]`. Required (modeling_flash_attention_utils.py:224-228).
  - **`flash_attn_func(q,k,v,softmax_scale,causal,...)`**, with layout **`[B,N,H,d]`**. Ours takes `[B,H,N,d]`, so it needs a wrapper.
  - `flash_attn_with_kvcache`: optional (:235-240).
- HF then registers `flash_attention_forward` together with the FA2 mask builder (hub_kernels.py:876-898). No monkey-patch is needed.
- Costs:
  - `kernels>=0.17.2,<0.18` must be installed (utils/import_utils.py:169-170). It is not installed on the remote box.
  - We need a Hub repo with kernel-builder binaries.
  - Repos outside `kernels-community` need `allow_all_kernels=True` (modeling_utils.py:1427-1432).
  - Only `get_kernel_hub` is used (hub_kernels.py:973-987). There is no local-path loading for attention kernels. `kernels`' own local-override env vars: UNVERIFIED.
  - `lazy_import_flash_attention` holds one process-global implementation (:146-151, :284-289).

### Reuse option B (recommended): in-process reuse of HF's helpers
- Our own attention function calls:
  - `_upad_input`, `_pad_input`, `_unpad_input`
  - `_is_packed_sequence`, `prepare_fa_kwargs_from_position_ids`
  - `fa_peft_integration_check`
- Use `flash_attention_mask` as the base mask builder.
- These helpers are private, so pin the transformers version or vendor ~150 lines (Apache-2.0).
- Do not call `_flash_attention_forward` directly. It always calls `lazy_import_flash_attention(attn_implementation)` (:780-783).

### Dropping the monkey-patch: rename without "flash"
- `is_flash_attention_requested` is just a substring check, `"flash" in name` (utils/generic.py:269-305).
- `get_correct_attn_implementation` accepts any registered key (modeling_utils.py:1846).
- The probe ran `legacy_attn` **with no patch**.
- The "flash" substring currently has these side effects:
  1. a preload attempt (modeling_utils.py:1826-1827), which is the reason the patch exists
  2. the gpt_oss auto-fallback (:1751-1772), which our patch bypasses
  3. a Llama4 chunked-attention raise (masking_utils.py:1376-1380)
  4. 41 modeling files switch to FA-only inputs (risk R3; gpt_bigcode:452-456)
  5. a `.item()` in `get_max_seqlen` (utils/generic.py:327-330)
  6. generation forces `fullgraph=False` (generation/utils.py:2475-2482)
- Downside: third-party tools that gate on "flash" (TRL padding_free) would refuse. UNVERIFIED: TRL is not installed.

## 4. Correctness risks in the current mask builder / attention function

Root cause: when our builder hands off to `sdpa_mask`, a returned `None` carries SDPA-specific meaning. With kv>q, SDPA's None means top-left alignment, implemented by slicing k/v to q_len (sdpa_attention.py:157-159). Our kernel reads None as bottom-right causal over all of k/v. Separately, our function ignores `softcap`, `s_aux`, `cu_seq_lens_*` and `position_bias`.

| # | Case | Mechanism | Severity | Probe result |
|---|---|---|---|---|
| **R1** | **StaticCache prefill** (`generate(cache_implementation="static")`) | kv_length=max_cache_len (cache_utils.py:480-484). Our offset check hands off to sdpa, and sdpa returns None because q_offset==0 (masking_utils.py:276-279). The kernel then runs bottom-right causal over q=12, kv=32, including 20 empty slots. The cache of later layers is corrupted too. | **CRITICAL**, silent | prefill max\|d\|=0.91; generate(static) tokens differ; generate(dynamic) matches |
| **R2** | Gemma2 softcap (gemma2:274) | Ignored on both paths. HF sdpa ignores it too; FA2 and eager honor it. | HIGH for Gemma2 | softcap=2 with scaled q/k: 0.89 (same as HF sdpa) |
| **R3** | VLM vision encoders under a "flash" name | `mask=None` plus `cu_seq_lens` (qwen2_vl:406-422). We attend across all images, and over the whole image where Qwen2.5-VL expects windows (qwen2_5_vl:442-473). | **CRITICAL** for VLMs | WRONG under `flash_attn_legacy`; exact (0) under `legacy_attn` |
| **R4** | gpt_oss sinks (gpt_oss:321) | Ignored on both paths. `_supports_sdpa=False`, and our patch bypasses the compatible-fallback. | **CRITICAL** for gpt-oss | 0.75 |
| R5 | Packing through `cu_seq_lens` only (collator `return_flash_attn_kwargs=True, return_position_ids=False`) | The builder returns None and `cu_seq_lens` is ignored, so attention crosses sequences. Same as eager/sdpa; FA2 handles it. | MEDIUM (FA2 parity gap) | 0.86 |
| R6 | Packed `position_ids` plus an all-ones mask | Packing detection is skipped when a mask is present (masking_utils.py:840). Same as eager/sdpa; FA2 handles it. | MEDIUM (FA2 parity gap) | 0.86 (eager: 0.86) |
| R7 | Packed, no mask (default collator) | Correct via a 4D mask and SDPA, with O(N^2) memory. | perf | ok |
| R8 | Padding (causal or bidirectional) | Correct via SDPA. | perf | ok |
| R9 | Sliding window (mistral/qwen3/gemma/modernbert) | Correct. Kernel only when kv < window. | ok | ok |
| R10 | DynamicCache chunked prefill / decode | Correct. | ok | ok |
| R11 | dtype | bf16/fp32 are cast to fp16 (integrations.py:80-91). bf16 models such as Gemma can overflow fp16. | MEDIUM | UNVERIFIED on GPU |
| R12 | v_head_dim != d (MLA) | No padding of V. | LOW-MED | UNVERIFIED |
| R13 | `position_bias` (sdpa_attention.py:89, 164-168) | Ignored. | LOW | which models use it: UNVERIFIED |
| R14 | `or_mask_function`, `and_mask_function`, `block_sequence_ids` | Correct: they force a 4D mask and the SDPA fallback. | ok | - |

## 5. Recommendations

1. **Fix R1.** After handing off, if sdpa returned None while `q_length > 1 and q_length != kv_length`, call it again with `allow_is_causal_skip=False`. Add StaticCache and static-generate tests.
2. **Fix R2/R4/R12/R13.**
   - If softcap, s_aux or position_bias is present, or V's head dim differs from Q's, use an eager-style fallback that builds the mask itself (bottom-right causal plus padding).
   - Later, add softcap and sinks to the kernel.
3. **Rename** to a name without "flash", and keep `flash_attn_legacy` as a deprecated alias. This drops the monkey-patch, fixes R3 and the Llama4 raise, and stops FA-only inputs. The head_dim check is already done per call.
4. **Native padding/packing (option B).**
   - **Mask builder:**
     - None when the mask is plain, fully valid and end-aligned.
     - A 2D `(B,kv)` mask when the only deviation is padding. Require: plain `mask_function`, int offsets, `kv_offset==0`, no active window, and mask length equal to kv_length.
     - Otherwise a 4D sdpa mask with `allow_is_causal_skip=False`.
   - **Attention function:**
     - 2D mask -> `_upad_input` -> our varlen -> `_pad_input`.
     - `cu_seq_lens_*` present -> varlen.
     - `_is_packed_sequence` -> varlen.
     - Otherwise dense.
     - This fixes R5-R7.
   - **Kernel:**
     - per-sequence bottom-right causal when q_len != kv_len, including q=1
     - GQA
     - **native varlen backward** (no `.item()` loop)
     - optional `window_size`
5. **Training.**
   - Gradient checkpointing only forces `use_cache=False` and `past=None` (modeling_layers.py:82-111). The kwargs are unchanged, and the attention function is called twice (forward and recompute).
   - Attention dropout defaults to 0.0 for llama, qwen2/3, mistral, gemma2/3, phi3, gpt_oss and modernbert. It is **0.1** for bert, roberta, xlm-roberta and gpt2, so those currently train through the SDPA fallback.
   - The collator defaults to `return_position_ids=True, return_flash_attn_kwargs=False` (data/data_collator.py:1298-1302), so packed training hits R7.
   - Trainer has no flash logic. TRL: UNVERIFIED.
6. **Tests.** Turn every R-row into an eager-equivalence test. The probe scripts in this directory can serve as templates.
