# Short queries against a long cache: split-KV and packed GQA rows in the forward (P6.8)

Status: design, 2026-10-23, reviewed (notes folded in). The code becomes the reference once implemented.

## Why (measured)
The decode kernel (P6.1) handles R = (H_q / H_kv) * q_len <= 8 query rows per KV head. Everything else that HF sends with a
cache takes another path today:
- q_len > 1 (verification steps of assisted / speculative decoding, prompt lookup, chunked prefill tails): the forward kernel,
  one block per (Br query rows, query head, batch): at q_len 2-16 that is B * H_q blocks, each walking the whole cache alone,
  and each query head re-reads its KV head's cache (rep times the traffic).
- q_len == 1 with H_q / H_kv > 8 (MQA, GQA 12:1 such as StarCoder2-3B 24/2): SDPA (`integrations.py`, "decode" fallback).

Attention of one layer, d=128, us (`spec_bench.py`: median of CUDA events, host launch hidden), B=1:

| GPU | H_q/H_kv | cache | q_len | HF path today | decode kernel (R <= 8) | q_len one-token decodes | SDPA (as HF) |
|---|---|---|---|---|---|---|---|
| P100 | 32/8 | 4k | 2 | 1925 (forward) | 106 | 137 | 1134 |
| P100 | 32/8 | 4k | 16 | 1926 (forward) | - | 1098 | 1156 |
| P100 | 32/8 | 16k | 2 | 7673 (forward) | 293 | 379 | 4482 |
| P100 | 24/2 | 4k | 1 | SDPA | - | - | 1129 |
| T4 | 32/8 | 16k | 2 | 594 (forward) | 310 | 545 | 1139 |
| T4 | 32/8 | 16k | 16 | 593 (forward) | - | 4365 | 1196 |
| T4 | 24/2 | 4k | 1 | SDPA | - | - | 283 |

The cache read alone (the bandwidth floor of a step) is about the one-token decode time (P100 64 us at 4k, T4 268 at 16k): on
the P100 a verification step costs ~30x that, on the T4 ~2x.

End to end, HF `generate` of 128 greedy tokens after a 1.5k prompt, Qwen2.5-3B (GQA 16/2) with Qwen2.5-0.5B as draft, tok/s
(`spec_e2e.py`; every output identical to plain greedy, no fallback):

| | plain | assisted (draft) | prompt lookup |
|---|---|---|---|
| P100 ours / SDPA / eager | 26.8 / 20.7 / 23.2 | 21.4 / 19.7 / 22.0 | 15.7 / 15.9 / 17.6 |
| T4 ours / SDPA / eager | 26.0 / 20.1 / 23.4 | 26.5 / 22.2 / 23.0 | 32.0 / 27.5 / 28.5 |

On the P100 speculation makes our backend slower than plain decoding (prompt lookup -41%) and below eager; on the T4 it helps
but the kernel numbers leave ~2x on long caches. The gap grows with the context.

## Options
1. Widen the decode kernel to R = 16..64. It keeps per-lane O / q state for all R rows (8R floats each): beyond R = 8 the rows
   must be split over warps that re-read the same keys, i.e. a CUDA-core forward with split-KV. On SM 7.x that gives up the
   tensor cores where R * d FMAs per key make the step compute bound (T4 R=64 d=128: ~0.5 ms at 16k, no better than today).
2. Split-KV and packed GQA rows in the forward kernels (chosen; upstream FA2 does the same: `num_splits` in the forward,
   `seqlenq_ngroups_swapped` for q_len == 1, FA3 `PackGQA` for q_len > 1). One implementation per family as today (WMMA
   sm_70, mma.sync sm_75, fp32 FMA Pascal), the bands' tensor cores on SM 7.x, and K/V read once per KV head.

The decode kernel stays for R <= 8 (bandwidth bound, measured near the floor); the forward with splits takes R > 8.

## Design (reviewed 2026-10-23; review notes folded in)
- **Entry.** The decode-style entry `flash_attn_with_kvcache` (cache read in place through strided views, no
  `contiguous_aligned` copy, no fp16 conversion pass) runs the forward for R > 8; the HF integration routes q_len > 1 with a cache
  to it (today `flash_attention` copies sliced static caches, `integrations.py`, and converts bf16 caches: separate costs, measured
  apart). For R > 8 it accepts fp16 only, per-batch `cache_seqlens` (int or tensor, clamped on the device as the decode kernel
  does), causal / window / softcap / sinks; bf16, `cache_leftpad`, `key_mask` and paged caches raise NotImplementedError (the
  forward bodies are fp16; padded HF batches with q_len > 1 keep the unpad -> varlen route, the 4D mask at q_len > 1 its fallback).
- **Split-KV.** Grid gets a split axis (num_splits, 1 = today, bit-identical); split s clips `fwd_band_loop`'s kv_begin / kv_end
  to its chunk of Bc blocks (an empty chunk skips the loop but still writes its partials). With num_splits > 1 a row writes the
  decode partial format: O normalised (O * inv, fp32) and its lse in log2, from `BandSoftmax::finalize`; rows with no visible key
  give inv = 0, lse = -inf, which the merge already handles. One combine for both paths: `decode_merge`, the combine kernel and
  the element types move from `flash_attn_decode.cu` to a header with a small combine-params struct used by decode and forward.
  Sinks enter split 0 only. Grid order puts the query heads of one (KV head, split) next to each other (L2 reuse of the K/V
  tiles). num_splits: the decode rule generalised (one function) with the resident blocks per SM and the rows of the tile as
  parameters, capped by the partials' traffic (2 * S * R / N of the cache read: S = 56, R = 64, N = 4k would cost 1.75x the
  read itself).
- **Packed GQA rows (after the splits are measured).** A block covers rows of one KV head: absolute row r = q_row_start + i is
  token r / rep, query head kv * rep + r % rep (token-major: causality is per token, bands straddling tokens are fine because
  first_key / last_key are monotone in the token). One `FwdBlockArgs` row-offset helper serves every write site (O stores and L
  writes of the three families, the fp32 partial stores) and the Q loader gets a row-offset functor (one loader, not a second
  one); rows_per_token = 1 keeps today's addressing. `KeyWindow` is built over q_len tokens (its offset is seq_len_k - seq_len_q
  in tokens); each lane's rows are fixed over the K/V loop, so their tokens are computed once before it, and `fwd_band_loop`'s
  q_first / q_last / band bounds become tokens. Sinks per row (sinks[kv * rep + r % rep]): `BandSoftmax::init` takes one value
  per row. L / lse stay in the natural [B, H_q, q_len] order. Its value is mostly band compute on Pascal (q_len 2, rep 4: four
  bands become one) and K/V read once per KV head: measured against (a) before it is merged.
- **Not in scope:** backward with splits or packing (inference only, as decode); paged K/V in the forward (P6.9); a packed varlen
  batch of mixed q_len (P6.9); bf16 / leftpad / key_mask with R > 8 (later, if measured useful).

## Steps (one change each: tests on all GPUs + review)
a. Shared combine header (decode unchanged, bit-identical). Then split-KV in the forward bodies (batched, inference) behind
   `flash_attn_with_kvcache` for R > 8 without packing (rep query heads as today's head axis) and the HF routing of q_len > 1.
   Tests: every family (forced Pascal too) against SDPA math with num_splits 1..16 forced and more splits than Bc blocks,
   causal / window / softcap / sinks, q_len 1..64 against N up to 32k, per-batch cache_seqlens tensors, d = 72 / 80 padded
   columns, CUDA-graph capture; num_splits = 1 bit-identical and the existing forward's speed unchanged.
b. Packed GQA rows. Tests as (a) with rep 1..16, including 6, 7, 12 (Br % rep != 0), MQA, and sinks with packing.
c. HF: q_len == 1 with rep > 8 to the packed path; the "decode (N query heads per KV head)" fallback reason disappears. Tests: HF
   assisted generate / prompt lookup / a rep-12 model identical to plain greedy and to SDPA within noise.
d. Measure (`spec_bench.py`, `spec_e2e.py` into `benchmarks/`), all GPUs; README / PLAN.

## Acceptance
- A verification step (q_len 2-16, cache 4-16k, GQA 4 and 12) within ~1.5x of its floor: the one-token decode time (the cache
  read) on SM 7.x; on Pascal the larger of that and the fp32 compute floor (4 d R N H_kv FLOP: q_len 16, GQA 4, 4k, d 128,
  H_kv 8 is 1.07 GFLOP, ~115 us at the P100's peak against today's 1926), and faster than SDPA everywhere.
- HF assisted generation and prompt lookup on the P100 faster than plain decoding with our backend (today slower), and not
  slower than SDPA / eager on any GPU; outputs identical to plain greedy.
- No new fallback; tests green on V100, T4, P100, 1080 Ti (and the 2080 Ti while available).
