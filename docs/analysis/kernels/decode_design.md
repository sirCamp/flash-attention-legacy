# Decode with a KV cache: split-KV kernel (P6.1)

Status: design, reviewed 2026-10-22 (review notes folded in). The code becomes the reference once implemented.

## Why
- Today every decode step (q_len == 1) leaves the kernel: SDPA, or the fp32 eager fallback when the layer has a softcap
  (Gemma 2) or sinks (gpt-oss), the last users of `_eager_attention`.
- Decode is memory bound: each step reads the whole K/V cache once. With batch 1 and few KV heads one block per (batch, head)
  leaves most SMs idle on a long context; splitting the keys over blocks (FlashDecoding) fills the GPU, a small kernel merges.
- Llama-3-8B (32 layers, H_kv 8, d 128), 8k context, B=1: ~1.07 GB of K/V per step -> ~1.5 ms V100, 1.8 P100, 2.8 1080 Ti,
  4.3 T4 at ~80% of the bandwidth. That is the target.
- A block table (paged KV cache) is cheap when designed in now (only the key address changes) and is what serving engines call
  (upstream `flash_attn_with_kvcache(..., block_table=...)`). A vLLM backend is a separate PLAN item, after checking which vLLM
  versions still run on SM 6.x / 7.x.

## API: upstream `flash_attn_with_kvcache`, same positional order
```
flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, rotary_cos=None, rotary_sin=None, cache_seqlens=None,
                        cache_batch_idx=None, cache_leftpad=None, block_table=None, softmax_scale=None, causal=False,
                        window_size=(-1, -1), softcap=0.0, rotary_interleaved=True, alibi_slopes=None, num_splits=0,
                        return_softmax_lse=False, *, sinks=None)
```
- Unsupported upstream arguments (k, v: in-kernel cache append; rotary; cache_batch_idx; alibi) raise NotImplementedError when
  not None. `sinks` is keyword-only (an extension; vLLM's fork calls it `s_aux`).
- q [B, q_len, H_q, d] (BSHD). Output [B, q_len, H_q, d] in q's dtype; with return_softmax_lse also lse [B, H_q, q_len] fp32.
- Contiguous cache k_cache / v_cache [B, max_len, H_kv, d], any strides with stride(-1) == 1 and 16-byte aligned rows (HF's
  BHSD [B, H_kv, N, d] passes as a transposed view, no copy). Paged: [num_blocks, block_size, H_kv, d] with block_table
  [B, max_blocks] int32, block_size % 16 == 0 (upstream FA2 needs % 256; a superset). Paged with cache_leftpad: rejected (as
  upstream).
- cache_seqlens: int or [B] int32, the END index of the valid keys (default max_len); cache_leftpad [B] int32 the first one.
  Valid keys of row b: [leftpad_b, seqlens_b). Query i (of q_len) sees key j iff leftpad <= j <= seqlens - q_len + i (causal;
  bottom-right) and, with a window, j >= seqlens - q_len + i - left. q_len == 1 makes causal irrelevant.
- A future cache_batch_idx keeps indexing cache_seqlens / cache_leftpad by the batch row b (as upstream), not by the
  cache index. Rotary (later) follows upstream: cos / sin rows by absolute cache slot.
- Rows with no visible key: O = 0, lse = -inf (this repo's convention; upstream FA2 may return +inf there: documented).
- dtypes: q / cache fp16 or bf16 (fp32 optional), read natively and computed in fp32 (bf16 -> fp32 is a shift): no Python
  conversion, no range check, no copy of the cache on the decode path. Inference only: raises if grad is required.

## Kernel (one CUDA-core implementation, every arch)
Arithmetic intensity is ~R FLOP/byte with R = rep x q_len query rows per KV head (rep = H_q / H_kv). Effective fp32 balance
points (~half of peak / bandwidth) are ~8-12 on these GPUs: for R <= 8 (q_len 1 with GQA <= 8, the common case) decode is
bandwidth bound everywhere and one CUDA-core kernel serves SM 6.0-7.5 (Pascal included, fp32 math). R > 8 (speculative decoding
x GQA) on SM 7.x would want the mma path (q_len x group packed into M, as upstream): a later, measured item; not looping over
q_len (that re-reads K/V).

- Grid: (num_splits, B * H_kv); 4 warps per block; a block computes all R rows of one KV head (K/V read once for the group).
- Lane mapping: each lane loads exactly one 16-byte uint4 of a key row; G = next_pow2(d / 8) lanes per key, 32 / G keys per warp
  step (d=32: G 4; 64: 8; 72/80/96/128: 16, with 9/10/12/16 active lanes; 256: 32). A group reads one contiguous row (full 32 B
  sectors in every layout). The group that scores key k also reads V row k; lane owns output dims [8 * lane_in_group, +8).
- Per lane registers: q 8R fp32 (pre-multiplied by the ScoreMap factor), O 8R, (m, l) 2R, K/V buffers 8U (U loads in flight,
  2-4), ~20 addressing: R=1 ~60, R=2 ~80, R=4 ~110; R=8 keeps q in shared memory (broadcast reads) ~140. Buckets R = 1, 2, 4, 8.
- Scores: G-lane xor reduction per row (later: reduce-scatter butterfly, ~2R shuffles per key step instead of R log2 G, and each
  lane exps only its share). Softcap is a runtime branch (one tanh per score is free when memory bound).
- Online softmax state (m, l, O) per group over its keys; one decode `merge(m, l, O)` combines groups (shuffles), warps (shared
  memory) and splits (combine kernel): one function, three call sites. The sink enters exactly once: the state of group 0 / warp 0
  / split 0 starts as (m = sink, l = 1), every other at (-inf, 0); split 0 never exits early when sinks are present.
- Keys: by range. The split ranges cover the rows' visible range (intersection of [leftpad, seqlens) with the union of the R rows'
  KeyWindow), chunk = ceil(range / splits) rounded to the warp key step, computed on the device; per-element masks only at the
  range edges. Paged: key j at block_table[b][j / block_size] * block_size + j % block_size (`kPaged`), else the strided address.
- num_splits on the host without syncs (CUDA graph friendly): from B * H_kv, the SM count and the cache capacity (or a max_seqlen
  hint): target ~2 waves, chunk >= 256 keys, cap 64-128; 1 for short contexts. num_splits == 1 writes O / lse directly; otherwise
  normalized fp32 O_s and lse_s go to a workspace and the combine computes lse = M + log sum exp(lse_s - M) (M == -inf: O = 0,
  lse = -inf; empty splits always write lse = -inf, O = 0).
- Templates: d, R bucket, dtype, paged. Softcap, window, causal runtime.

## HF integration
- Decode steps (q_len == 1, short q_len with a cache) go to the kernel; output BSHD returned as it is (no transpose copy).
- Dynamic cache: K/V contiguous BHSD [B, H_kv, past + 1, d], mask None: seqlens = N.
- Static cache: pass the unsliced cache; cache_seqlens on the device from the valid prefix (no `int(q_offset)` sync).
- Left padding: leftpad from the first 1 of the 2D mask, computed once per step on the shared mask object (one sync, as
  `_cached_segments`), never through `_unpad_plan` (its gathers copy the cache). Holes inside a row keep falling back. Masks
  built per layer type (sliding / full) give their own leftpad.
- Sliding caches: the dynamic sliding layer passes up to W - 1 + q_len keys (window satisfied); the padding scrolls out (leftpad
  -> 0): test decoding past W with and without left padding.
- Beam search: reordered caches are plain tensors; low risk. Cross-attention decode (non-causal against encoder K/V): covered
  with causal=False except T5's position_bias (stays on SDPA).
- Grad enabled: the HF path falls back instead of raising.

## Tests
- vs the FP32 reference: every d; q_len 1/2/4; rep 1/4/7/8 (7: Qwen2); BSHD and BHSD (transposed) caches; unaligned / odd
  strides rejected; cache_seqlens ragged (0, 1, step edges, max) as int and tensor; leftpad; causal q_len > 1; window (empty
  window too); softcap; sinks (also with an empty key range: lse = sink, O = 0); num_splits 1 / forced > 1 / auto, empty splits;
  bf16 cache with values beyond the fp16 range; paged with shuffled block tables, block sizes 16/64/256; return_softmax_lse;
  FORCE_ARCH=pascal on V100/T4.
- HF: generate (greedy and beam, dynamic and static, left padding, past the sliding window) for Llama, Gemma 2 (softcap),
  gpt-oss (sinks), Mistral (window), Qwen2 (rep 7) within the noise of FP32 eager; fallback_stats() without "decode".
- Benchmark: decode step latency vs SDPA and vs the prefill kernel at q_len = 1, B = 1/8, contexts 1k-32k, d = 64/128, and the
  end-to-end step time (mask builder included).

## Steps (one change each: tests on all GPUs + review)
a. Contiguous cache, num_splits = 1, R <= 4, fp16 / bf16, API with upstream order and return_softmax_lse.
b. Splits + combine (num_splits auto / forced), sinks once.
c. R = 8 (q in shared memory).
d. cache_leftpad.
e. Done: paged (block_table), without leftpad; kPaged template (the contiguous path keeps its addressing), shift / mask for
   power-of-2 block sizes. T4 GQA 32/8 d=128: B=1 16k 0.297 ms contiguous vs 0.310 paged; B=8 4k 0.606 vs 0.71-0.73 (the
   table load precedes the K / V loads: prefetching the next block indices is the open item).
f. HF routing of decode steps.
g. Tuning, measured on T4 (decode kernel only, B=8, 16k context): R <= 4 runs at 200-236 GB/s (d=128: ~65-70% of the
   peak; near the bound). R = 8 (d=64, gpt-oss) at ~100 GB/s, issue bound: ablations (wrong results, timing only) put
   the per-key update (exp2, P V FMAs, V conversion) at ~33% and the score shuffles at ~14%; 135 registers -> 3 blocks
   of 4 warps per SM. Tried, no measurable gain, not kept: U = 2 / 4 keys per group and step with one block merge
   (R = 4 got slower), q in registers at R = 8, a one-key specialised update (one exp2 when the max holds). Still
   5-10x faster than SDPA. Open: a reduce-scatter of the scores (~2R shuffles instead of R log2 G, one exp2 per row
   per group), or the mma path for R > 4 on SM 7.x; num_splits sized by capacity (HF static caches).
