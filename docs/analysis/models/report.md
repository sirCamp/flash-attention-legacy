# flash_attn_legacy x HF transformers 5.20.2: model-family survey (Tesla V100-SXM2-32GB, SM 7.0)

> **Snapshot before P0** (transformers 5.20.2 at the time). BUG-1/2/3 and the missing features M1-M7 are resolved, see `docs/PLAN.md` (§1, §3); each row carries its resolving entry. The present tense below describes the code at the time.

Environment: torch 2.7.1+cu126, transformers 5.20.2, all models loaded in fp16. The library code under /workspace/fa was not modified.
Everything lived in `/workspace/analysis/models/` on the test box; the scripts (`*.py`) and raw results (`results/*.json`) are tracked under `docs/analysis/models/`, the logs (`logs/*.log`) stayed on the test box and are not committed.
`python show.py results/<tag>.json` prints a compact view of one survey result.

## How correctness was judged

- **Path detection.** A spy (`common.py: Spy`) re-registers a wrapper around `flash_attn_legacy_forward` in `ALL_ATTENTION_FUNCTIONS` and wraps `flash_attn.flash_attention`. For every call it logs the layer, the q/k shapes, the mask (None or shape/dtype), all kwargs, and whether the kernel ran.
- **Reference ("gold").** For models of 4B parameters or fewer, the gold reference is the same weights in fp32 with eager attention. Eager honours softcap and sinks, while HF `sdpa` does not honour softcap. For the 7B models the reference is fp16 sdpa, with fp16 eager-vs-sdpa as the noise floor.
- **Logit metric.** "1-cos" is the mean over positions of (1 - cosine) of the logit vectors, and top1 is argmax agreement. Every flash number is reported next to the fp16-sdpa and fp16-eager numbers on the same input.
- **Per-call shadow check (`shadow.py`).** On every attention call this recomputes attention in fp32 with every feature the model passed (softcap, sinks, sliding window, mask). It reports the relative error of what flash_attn_legacy returned, alongside fp16 SDPA's error on the same q/k/v as a per-call noise floor.

## Summary table

Abbreviations:
- **K** = flash kernel ran.
- **FB** = SDPA fallback inside `flash_attn_legacy_forward`.
- **x/y** = calls of that kind out of total calls.
- Decode (q_len=1) is always FB by design, for every model, so it is not repeated per row.

| Model | head_dim, Hq/Hkv, features | Unpadded prefill 1000 | Left-pad batch of 2 (fwd + generate) | Chunked prefill 300+100 (DynamicCache) | Greedy generate 64 | Long context past the window | Correct? |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen2.5-0.5B-Instruct | 64, 14/2, no SWA | K 24/24 | FB 24/24 (mask 2x1x200x200) | K 48/48 (both chunks) | prefill K 24/24, 64/64 tokens = sdpa = fp32 gold | n/a | **Yes**. vs gold: flash 4.89e-6, sdpa 5.01e-6, eager16 1.08e-5 |
| Qwen2.5-0.5B, SWA forced (window 256, layers 12-23 sliding; config override) | 64, 14/2, SWA 256 | K on 12 full layers, FB on 12 sliding layers | FB | 2nd chunk: K on 12 full, FB on 12 sliding | 64/64 | N=1500: vs gold flash 4.29e-6, sdpa 4.41e-6. pos>=256: 3.93e-6 vs 3.93e-6. Long gen 32/32 | **Yes** |
| Qwen/Qwen3-0.6B | 128, 16/8, q/k-norm | K 28/28 | FB 28/28 | K 56/56 | 64/64 vs sdpa and gold | n/a | **Yes**. vs gold: flash 1.85e-6, sdpa 1.79e-6, eager16 2.74e-6 |
| mistralai/Mistral-7B-Instruct-v0.3 | 128, 32/8, no SWA | K 32/32 | FB 32/32 | K 64/64 | 64/64 | n/a | **Yes**. flash-vs-sdpa 6.56e-7, eager-vs-sdpa 7.75e-7. Chunked 1.55e-6 (eager 1.73e-6) |
| mistralai/Mistral-7B-v0.1 | 128, 32/8, **SWA 4096** | K 32/32 | FB 32/32 | **2nd chunk FB 32/32** (kv=400 < window) | 64/64 | N=5000: **FB 32/32** (bit-identical to sdpa; eager-vs-sdpa 1.31e-6). 256-token chunk after 4744: FB. Gen 32/32 | **Yes** (no kernel past the window) |
| microsoft/Phi-3-mini-4k-instruct | **96**, 32/32, **SWA 2047** on all layers | K 32/32 | FB | **2nd chunk FB 32/32** | 64/64 vs sdpa and gold | N=3500: **FB 32/32**. vs gold flash = sdpa = 4.77e-6 | **Yes**. 1000-token prefill vs gold: flash 5.01e-5, sdpa 7.01e-5, eager16 6.85e-5 |
| google/gemma-3-1b-it | **256**, 4/1, **SWA 512** (22 sliding + 4 full layers) | K 4/26 (full layers 5, 11, 17, 23), FB 22/26 | FB 26/26 | 1st chunk K 26/26; **2nd chunk K 4/26** | prefill K 26/26 (26 tokens < 512). Diverges at token 5 from sdpa, eager and fp32 gold; this is an fp16 near-tie (top-2 margin 0.0078 logit), not a bug | N=5000: K 4/26. vs gold flash 7.15e-7, sdpa 7.75e-7. pos>=512: 6.56e-7 vs 6.56e-7. Chunk after 4744: 8.34e-7. Gen 32/32 | **Yes**. 1000-token prefill vs gold: flash 1.31e-6, sdpa 1.37e-6, eager16 1.67e-6 |
| google/gemma-2-2b-it | **256**, 8/4, SWA 4096 alternating, **attn softcap 50**, final softcap 30 | **K 26/26**, softcap=50.0 passed and ignored | FB 26/26 (FB also ignores softcap) | 2nd chunk K 13/26 | 64/64 vs sdpa, but **diverges from fp32 gold at token 29** (eager16 stays 64/64 vs gold) | N=5000: K 13/26, FB 13/26. vs gold **flash 3.90e-5, sdpa 4.01e-5, eager16 4.17e-7**. max\|dlogit\| 12.7 (eager16 0.71) | **NO** (BUG-1: softcap ignored) |
| google-bert/bert-base-uncased | 64, 12/12, bidirectional | 512 tokens: K 12/12, causal=False (same with or without an all-ones mask) | padded batch of 4: FB 12/12 | n/a | n/a | n/a | **Yes**. vs gold flash 2.56e-6, sdpa 1.85e-6. Pooled cos 0.9999983. One-position outlier explained below |
| answerdotai/ModernBERT-base | 64, 12/12, bidirectional, local 128 (`sliding_window` kwarg = 65), global every 3rd layer | K 8/22 (global layers 0, 3, ..., 21, causal=False), FB 14/22 (local layers; also at 64 tokens) | FB 22/22 | n/a | n/a | n/a | **Yes**. vs gold flash 9.89e-6, sdpa 3.02e-5, eager16 3.74e-5 |
| Qwen/Qwen3-Embedding-0.6B (AutoModel, last-token pooling) | 128, 16/8, causal | K 28/28 | padded batch of 4: FB 28/28. Padded-vs-alone pooled cos >= 0.999997 | n/a | n/a | n/a | **Yes**. vs gold flash 3.64e-6, sdpa 3.70e-6. Pooled cos 0.9999983 |
| openai/gpt-oss-20b | 64, 64/8, SWA 128 alternating, **sinks** | **not tested (size)**: about 21B params, roughly 42 GB in fp16, which does not fit in 32 GB | - | - | - | - | Replaced by a tiny random GptOss with the same attention (2 layers, head_dim 64, window 128, sinks ~ N(0,2)): **NO** (BUG-2) |

Notes on the table:
- **Left padding.** In every causal model the padded batch produced logits bit-identical to sdpa (1-cos 0, max\|d\| 0). The short padded prompt matched the same prompt run alone, and generated tokens matched sdpa 24/24 for both sequences.
- **Chunked prefill.** "2nd chunk FB" on sliding-window models happens even when kv_len < window. The mask builder delegates to `sdpa_mask`, and that never skips the mask when q_offset > 0 and q_len != kv_len. Results are still correct; this is a missing fast path (see M3).
- **Existing test.** `tests/test_hf_equivalence.py` on 7 models (log `logs/pytest_hf_equivalence.log`, on the test box, not committed): **35 passed, 7 failed**. None of the failures is a numerical error in the kernel:
  - gemma-3 N=16, top1 0.9375: an fp16 near-tie. Flash 1-cos 9.5e-7 is below the eager floor of 1.43e-6.
  - gemma-3 N=1000 and the chunked tests on gemma-3, gemma-2, Phi-3 and Mistral-v0.1: these are expected sliding-window fallbacks. The test asserts that every layer ran the kernel.
  - gemma-3 greedy generation: the same near-tie as in the table (first divergence at token 5).
  - **gemma-2 passes everything except chunked**, because the test's reference is HF sdpa, which also ignores softcap. The test cannot see BUG-1.

## Confirmed bugs (measured)

### BUG-1 (resolved: P0.3, P4.2): `softcap` is silently ignored on both the kernel path and the fallback path. Affects Gemma-2.

What the code receives: Gemma-2 passes `softcap=50.0` (plus `sliding_window=4096`) to the attention function on every layer. `flash_attn_legacy_forward` ignores it in both branches, with no warning.

Minimal repro: `/workspace/analysis/models/repro_ignored_kwargs.py` (random q/k/v, head_dim 256, 8/4 heads, N=256):

```
[kernel (mask=None)]   no feature : rel err vs ref = 2.14e-04 (noise floor)
[kernel (mask=None)]   softcap=20 : rel err vs softcapped ref = 1.54e-01; identical to no-softcap output: True
[fallback (bool mask)] softcap=20 : rel err vs softcapped ref = 1.54e-01; identical to no-softcap output: True
warnings raised: []
```

Model-level evidence, gemma-2-2b-it (`results/gemma2_2b.json`, `results/shadow_gemma-2-2b-it.json`):

| Comparison vs fp32 eager gold | flash (all 26 layers K) | sdpa fp16 | eager fp16 (honours softcap) |
|---|---|---|---|
| prefill N=1000, 1-cos | 4.29e-6 | 4.23e-6 | 2.98e-7 |
| long N=5000, 1-cos | 3.90e-5 | 4.01e-5 | 4.17e-7 |
| long N=5000, max \|dlogit\| | 12.7 | 13.4 | 0.71 |
| greedy 64 tokens, first divergence from gold | token 29 | token 29 | 64 (none) |

Per-layer shadow check (fp32 reference with softcap; N=1000): the attention-output relative error of flash_attn_legacy is between 1.7e-3 and **1.59e-2** (worst is layer 14). The noise floor on non-softcap models is about 2.4e-4. On the 26-token prompt the worst is 7.3e-3.

Because HF's own `sdpa` also drops softcap, flash matches sdpa (1-cos 2.98e-7). Any test that uses sdpa as the reference therefore passes.

### BUG-2 (resolved: P0.3, P4.3): attention sinks (`s_aux`, gpt-oss) are silently ignored on both paths.

Minimal repro: `repro_ignored_kwargs.py`. The output with `s_aux` is bit-identical to the output without it, with rel err 6.57e-2 vs the sink-aware reference on both the kernel and fallback paths.

Model-level repro: `/workspace/analysis/models/sinks_probe.py`. It builds a tiny random GptOss (real `GptOssAttention`: 2 layers sliding+full, head_dim 64, window 128, sinks drawn from N(0,2)). Results are in `results/sinks_probe.json`. The spy shows `s_aux` arriving as a `tensor(4,):float16`.

| N | path | flash vs fp32 gold: 1-cos / top1 | eager fp16 vs gold: 1-cos / top1 |
|---|---|---|---|
| 64 | K 2/2 | **1.96e-2 / 0.719** | 5.96e-7 / 1.000 |
| 100 | K 2/2 | **1.52e-2 / 0.860** | 2.98e-7 / 1.000 |
| 512 | K 1/2 (sliding layer FB, which still ignores sinks) | **3.88e-3 / 0.941** | 4.93e-4 / 0.992 |

Note: GptOss declares `_supports_sdpa=False`. `flash_attn_legacy` is nevertheless accepted, and its "safe" SDPA fallback has no sink support either.

### BUG-3 (resolved: P0.4, P3.2): packed sequences without a packed mask run the kernel as one causal sequence; `position_ids` and `cu_seq_lens_*` are ignored.

Minimal repro: `/workspace/analysis/models/repro_packed.py`. Full probe: `train_probe2.py`, results in `results/train_probe2_qwen25.json`.

Setup: Qwen2.5-0.5B, `DataCollatorWithFlattening` output (input_ids concatenated, position_ids restarting at 0, no attention_mask), in `model.train()`.
- **Default `use_cache` (True in the config, as in a custom training loop):** transformers creates a DynamicCache, so `_preprocess_mask_arguments` skips packed-sequence detection and the mask builder returns None. Result: **K 24/24, one causal sequence.**
  - In repro_packed.py, the 2nd sequence vs the same sequence alone: 1-cos **0.346**, top1 **0.548**.
  - In train_probe2 (3 sequences of 300/150/62 tokens): seq1 1-cos 0.222 / top1 0.467, seq2 0.240 / top1 0.452. seq0 is unaffected (5.7e-6).
  - Loss is 2.3252, against 1.8872 when each sequence is run alone (token-weighted).
- **`return_flash_attn_kwargs=True`:** `cu_seq_lens_q/k` (int32, shape (4,)) and `max_length_q/k=300` do reach the attention function, but they are ignored. The results are identical to the run without them, so contamination is identical.
- **`use_cache=False`** (this is the Trainer default: `TrainingArguments.use_cache=False` sets `config.use_cache`): a packed mask (1,1,512,512) is built, so FB 24/24. Loss 1.887566 = sdpa, and contamination is at most 6e-6. **Correct, but the kernel never runs.**
- **Shared with sdpa/eager:** with use_cache=True they contaminate in the same way. That part is upstream transformers behaviour.
- **Why it is still a bug here:** HF `flash_attention_2` derives varlen boundaries from `position_ids` (`_is_packed_sequence`) or `cu_seq_lens_*` regardless of the cache. A backend named "flash" is expected to honour that contract. The library already ships `flash_attn_varlen_func`, but the integration does not use it.

## Checked and found NOT to be bugs (measured)

- **Forward kernel sweep** (`kernel_probe.py`, `results/kernel_probe.json`): 570 configurations.
  - Swept: d in {32, 64, 96, 128, 256}; GQA 4:1, 8:8 and 14:2; N in {2, 5, 16, 17, 26, 31, 33, 63, 64, 65, 100, 127, 129, 257, 1000}; q<kv cases (26/30, 5/40, 100/355, 64/200); causal and non-causal.
  - **0 bad.** Max abs error vs fp32 is 1.305e-3, identical to fp16 SDPA's 1.305e-3.
- **Backward kernel sweep** (`bwd_probe.py`): 72 configurations (B 1-3, d 64/128/256, GQA 14:2 and 8:8, N 77/512, causal and non-causal), measured per batch element.
  - 0 bad. Max rel err: dq 3.41e-4, dk 3.43e-4, dv 2.99e-4. fp16 SDPA: 4.03e-4, 4.58e-4, 3.70e-4.
  - On real captured Qwen2.5 layer-0 q/k/v (|q| up to 80.6, |k| up to 130.5): flash dq 4.91e-3 vs SDPA16 5.12e-3.
- **Per-call shadow error** on Qwen2.5, Gemma-3, BERT and ModernBERT: kernel-layer attention rel err is 2.2e-4 to 2.5e-4, the same as fp16 SDPA on the same inputs. The per-layer max-error ratio kernel/SDPA16 is 1.00 to 1.06.
- **Gemma-3 greedy divergence at token 5:** the sdpa top-2 logit margin at that step is 0.0078 (`results/shadow_gemma-3-1b-it.json`, `gen_steps`). Per-layer error is at the noise floor. The 1000-token prefill and 5000-token long-context runs match gold at sdpa level.
- **BERT one-position outlier** (`bert_rows.py`, `results/bert_rows.json`): at N=512 only position 88 (token ".") has 1-cos > 1e-5 (1.6e-4). For every layer, the kernel's per-row error at that position equals SDPA16's. This is fp16 noise amplified through depth, not a kernel error, and the pooled embedding is unaffected.
- **Sliding window in general:** never silently wrong in any measured case.
  - When kv_len >= window, transformers' sliding mask function is not "plain causal". `_flash_attn_legacy_mask` then delegates to `sdpa_mask`, which materialises a bool mask, so the call goes to FB.
  - When kv_len < window, `sdpa_mask` returns None and the kernel runs, which is correct because the window cannot bind.
  - Checked past the window: gemma-3 (5000 > 512), Mistral-v0.1 (5000 > 4096), Phi-3 (3500 > 2047), Qwen2-SWA (1500 > 256). Chunked prefill across the window and decode past the window both match.

## Training probe: Qwen2.5-0.5B, fp16, B=2x512 (`train_probe.py`, `results/train_probe_qwen25.json`)

Gold is fp32 eager. Gradient norms are listed as flash / sdpa / gold.

| Case | Path | Loss: flash / sdpa / eager16 / gold | Gradient notes |
|---|---|---|---|
| (a) no padding, mask all ones (same result with no mask) | **K 24/24** | 1.981796 / 1.981871 / 1.981817 / 1.981559 | emb 7.470 / 7.445 / 7.452; L0 q_proj 0.0993 / 0.0905 / 0.0899; L0 k_proj 0.1521 / 0.1339 / 0.1326; L12 v_proj 1.1677 / 1.1712 / 1.1677; L23 down_proj 2.809 / 2.810 / 2.740 |
| (b) right padding, labels -100 | FB 24/24 (mask 2x1x512x512) | 1.921613 / 1.921613 / 1.921516 / 1.921167 | flash grads identical to sdpa |
| (c) packed, DataCollatorWithFlattening | see BUG-3 | default use_cache: 2.325166 / 2.324738 / 2.324986 / 2.324502 (**all contaminated**). use_cache=False: 1.887566 = sdpa (correct) | - |

On (a), the gradient direction of the layer-0 q/k projections deviates from gold:
- grad cos vs gold: **flash 0.384 / 0.823**, eager16 0.359 / 0.877, HF sdpa 0.999 / 0.999.
- The isolation test `e2e_grad.py` runs an fp32 model with only the attention in fp16. There, flash gets 0.879 / 0.887, plain fp16 SDPA (mem-efficient, repeat_interleave) 0.866 / 0.875, and fp16 math 0.829 / 0.888.

So the kernel's contribution matches fp16 SDPA. The deviation is fp16-attention precision on Qwen2.5 layer 0, where q/k reach |q| 80 and |k| 130. HF sdpa does better likely because, with GQA (`enable_gqa`), torch on SM70 falls back to the math backend, which upcasts to fp32. That explanation is suspected; the backend choice was not instrumented. Loss and most gradient norms agree within 1%. The layer-0 q/k gradient norms are 10-15% higher than gold, the same as fp16 eager.

## Timing (rough; `timing.py`, `results/timing_*.json`)

Prefill is a full forward with `logits_to_keep=1`, median of 5 runs. Generate is 128 greedy tokens.

| Model | N | flash ms | sdpa ms | speedup | peak GB flash / sdpa |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | 512 / 1024 / 2048 / 4096 | 23.5 / 44.9 / 110.5 / 310.5 | 27.7 / 49.2 / 146.1 / 481.1 | 1.18 / 1.10 / 1.32 / **1.55** | 4096: 1.12 / 3.09 |
| Qwen3-0.6B | 512 / 1024 / 2048 / 4096 | 39.2 / 86.5 / 212.6 / 611.2 | 39.7 / 82.7 / 238.3 / 811.4 | 1.01 / 0.96 / 1.12 / **1.33** | 4096: 1.66 / 4.07 |
| Mistral-7B-v0.3 | 512 / 1024 / 2048 / 4096 / 8192 | 143.4 / 294.9 / 694.5 / 1856.5 / 5693 | 141.1 / 316.5 / 817.5 / 2397.4 / **OOM** | 0.98 / 1.07 / 1.18 / 1.29 / - | 4096: 14.47 / 18.98; 8192: 15.42 / OOM |

Generate throughput, in tok/s (flash vs sdpa):
- Qwen2.5: 43.9 vs 38.0 (32-token prompt), 43.6 vs 37.9 (1000-token prompt).
- Qwen3: 30.2 vs 27.4, 30.8 vs 27.7.
- Mistral: 32.4 vs 29.6, 29.1 vs 26.7.

Decode never runs the kernel. The roughly 10% decode gain comes from the integration's own SDPA call (`repeat_interleave`, no mask) being cheaper than HF's `sdpa_attention_forward`. Sliding-window models get no kernel speedup beyond the window (all of Phi-3 and Mistral-v0.1; 22/26 layers of Gemma-3).

## Missing features (and which models need them)

| # | Feature | Today | Models affected |
|---|---|---|---|
| M1 (resolved: P4.2) | **Logit softcapping** (tanh) in the kernel, or at least a softcap-aware fallback, or a hard error | silently wrong (BUG-1) | Gemma-2 (all layers); any model that passes `softcap` |
| M2 (resolved: P4.3) | **Attention sinks** (`s_aux`) | silently wrong (BUG-2) | gpt-oss-20b/120b |
| M3 (resolved: P4.1) | **Sliding-window / local attention in the kernel**, including chunked prefill with q_offset>0 while kv < window (today FB even when the window cannot bind) | correct via SDPA FB, but no speed or memory benefit | Gemma-3 (22/26 layers beyond 512 tokens), Gemma-2 (13/26 beyond 4096), Phi-3-mini-4k (all layers beyond 2047), Mistral-7B-v0.1 (all beyond 4096), Qwen2 with use_sliding_window, gpt-oss (window 128), ModernBERT local layers (14/22 at any length, window 128 bidirectional) |
| M4 (resolved: P3.1) | **Varlen / unpadding for padded batches** (the library already has `flash_attn_varlen_func`; the integration does not use it) | FB on every padded batch: batched generation, right-padded SFT, encoder batches, Qwen3-Embedding batches | all models |
| M5 (resolved: P3.2) | **Packed sequences** via `position_ids` / `cu_seq_lens_*` (FA2 contract) | wrong when no packed mask is built (BUG-3); FB when it is | all causal LMs trained with DataCollatorWithFlattening |
| M6 (resolved: P6.1) | Decode kernel (q_len=1) | always FB, by design | all |
| M7 (resolved: P0.1; d=72/80 in P2.7) | Head dims outside {32, 64, 96, 128, 256}: `_patched_check` silently switches the whole model to "sdpa" (since P0.1 the head_dim check is per call, with a recorded fallback reason) | not measured (code reading only) | e.g. head_dim 80 models |
| M8 (open: P6.4) | Attention dropout in training | FB (from code; not measured) | BERT-style training with attention_probs_dropout 0.1 |

## Unexpected or noteworthy

1. **HF `sdpa` itself ignores Gemma-2 softcap.** sdpa and flash agree with each other (1-cos 2.98e-7) and are both wrong vs gold. The repo's `test_hf_equivalence.py` uses sdpa as the reference, so it cannot detect BUG-1; the softcap check needs an eager or fp32 reference.
2. **Packed contamination with default `use_cache=True` also affects sdpa and eager** in transformers 5.20.2. The packed mask is only built when `past_key_values is None`. The Trainer avoids this because `TrainingArguments.use_cache` defaults to False.
3. **Eager fp16 returned NaN** on the heavily left-padded short sequence for Qwen2.5-0.5B (`results/qwen25_05b.json`, seq1 eager_vs_sdpa non-finite). This is in eager, not flash; flash/sdpa were finite.
4. **gpt-oss accepts `attn_implementation="flash_attn_legacy"`** even though it declares `_supports_sdpa=False`, and the backend then falls back to sink-less SDPA.
5. **ModernBERT local layers never use the kernel**, even for 64-token inputs below the window. The bidirectional sliding mask is always materialised.
6. Gemma models (fp16) produced finite outputs everywhere tested (up to 5000 tokens); no fp16 overflow was observed.
7. Mistral-7B prefill at 8192 tokens fits with flash_attn_legacy (15.4 GB peak) but OOMs with HF sdpa on 32 GB.

## Repro and script index (all under `/workspace/analysis/models/`, tracked under `docs/analysis/models/`)

- `repro_ignored_kwargs.py`: BUG-1 and BUG-2 at the attention-function level (kernel and fallback).
- `sinks_probe.py`: BUG-2 on a tiny real GptOss model.
- `repro_packed.py`: BUG-3. `train_probe2.py` is the full packed / use_cache matrix plus the backward probes.
- `survey.py MODEL [--gold] [--long N] [--override JSON]`: the per-model scenarios (prefill, left-pad, chunked, generate, long) with the spy and fp32 gold.
- `encoders.py`: BERT, ModernBERT and Qwen3-Embedding.
- `shadow.py [--encoder]`: per-call full-feature fp32 check.
- `kernel_probe.py` and `bwd_probe.py`: forward and backward kernel sweeps.
- `train_probe.py` and `e2e_grad.py`: the training probe and the gradient isolation test.
- `timing.py`: the timing runs.
- `logs/pytest_hf_equivalence.log`: the existing test run on 7 models (on the test box, not committed).
