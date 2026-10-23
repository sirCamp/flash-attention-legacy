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
  q_first / q_last / band bounds become tokens. Sinks (one per query head) would need one value per row in
  `BandSoftmax::init`: calls with sinks keep the unpacked layout (a block per query head; gpt-oss at q_len 1 is within the
  decode kernel's 8 rows). L / lse stay in the natural [B, H_q, q_len] order. Its value is mostly band compute on Pascal (q_len 2, rep 4: four
  bands become one) and K/V read once per KV head: measured against (a) before it is merged.
- **Not in scope:** backward with splits or packing (inference only, as decode); paged K/V in the forward (P6.9); a packed varlen
  batch of mixed q_len (P6.9); bf16 / leftpad / key_mask with R > 8 (later, if measured useful).

## Steps (one change each: tests on all GPUs + review)
a. Shared combine header (decode unchanged, bit-identical). Then split-KV in the forward bodies (batched, inference) behind
   `flash_attn_with_kvcache` for R > 8 without packing (rep query heads as today's head axis) and the HF routing of q_len > 1.
   Tests: every family (forced Pascal too) against SDPA math with num_splits 1..16 forced and more splits than Bc blocks,
   causal / window / softcap / sinks, q_len 1..64 against N up to 32k, per-batch cache_seqlens tensors, d = 72 / 80 padded
   columns, CUDA-graph capture; num_splits = 1 bit-identical and the existing forward's speed unchanged.
b. Packed GQA rows (no sinks). Tests as (a) with rep 1..16, including 3, 7, 12 (Br % rep != 0), MQA; sinks run unpacked.
c. HF: q_len == 1 with rep > 8 to the packed path; the "decode (N query heads per KV head)" fallback reason disappears. Tests: HF
   assisted generate / prompt lookup / a rep-12 model identical to plain greedy and to SDPA within noise.
d. Measure (`spec_bench.py`, `spec_e2e.py` into `benchmarks/`), all GPUs; README / PLAN.

## Step (a), done
- `flash_attn_split.h`: element types, `decode_merge`, `split_combine_kernel` (`SplitCombineParams` in `flash_attn_common.h`, host
  entry `flash_attn_split_combine`), shared by decode and forward; decode bit-identical (50 cases: fp16 / bf16, splits 1 / 3 / 16 /
  auto, paged, sinks, softcap). One split-count rule `flash_attn_num_splits(blocks, sms, resident, max_len, rows)`, unchanged for
  decode (rows <= 8).
- Forward: `fwd_batched_kernel<Tile, Body, kCache>`. kCache (selected by `O_part`) is the variant of `flash_attn_with_kvcache`:
  per-batch key lengths, split s of the visible K/V blocks, fp32 partials + log2 lse through `fwd_store_o` / `fwd_store_lse`
  (also one split), sink in split 0. A template parameter because the plain forward must carry none of it: a runtime branch in
  the epilogue made the Pascal d=128 forward 23% slower on the P100 (36 -> 44 ms at N=4096; 39 with the partial store out of
  line), and the per-batch length load alone made d=64 9% slower on the 1080 Ti (15.7 -> 17.2 ms); with kCache the plain
  forward measures as HEAD on every GPU (alternated A/B against a HEAD build on the same box).
- `flash_attn_with_kvcache` with R > 8: the kCache forward on the cache views, fp16 and a contiguous cache (bf16, paged,
  cache_leftpad, key_mask raise NotImplementedError).
- Attention of one layer, d=128, B=1, us (HF's path today -> `flash_attn_with_kvcache`):

| GPU | GQA 32/8, 16k, q_len 16 | GQA 32/8, 4k, q_len 4 | GQA 24/2, 4k, q_len 1 (SDPA today) |
|---|---|---|---|
| P100 | 7565 -> 3306 | 1878 -> 850 | 1130 -> 666 |
| 1080 Ti | 5371 -> 2632 | 1335 -> 671 | 7218 -> 444 |
| 2080 Ti | 723 -> 447 | 185 -> 122 | 223 -> 91 |
| V100 | 1214 -> 628 | 311 -> 170 | 326 -> 157 |
| T4 | 597 -> 602 | 154 -> 168 | 283 -> 123 |

  The splits fill the grid where it was starved (Pascal, the V100's 80 and the 2080 Ti's 68 SMs); on the T4 the forward was already bandwidth
  bound and each query head re-reads its KV head's cache (rep x the traffic): that is step (b).

- HF: q_len 2..SHORT_Q_MAX (64) new tokens against a longer cache go to `flash_attn_with_kvcache` (the cache read in place: the
  decode kernel up to 8 rows per KV head, the kCache forward beyond; inference, no padding; bf16 beyond 8 rows stays on the
  forward). End to end, HF generate as in "Why" (Qwen2.5-3B, 1.5k prompt, 128 tokens, outputs identical to plain greedy, no
  fallback), tok/s ours (before -> after) / SDPA / eager:

| GPU | plain | assisted (draft) | prompt lookup |
|---|---|---|---|
| P100 | 27.0 / 20.6 / 23.0 | 21.4 -> 23.4 / 19.3 / 21.4 | 15.7 -> 23.1 / 19.1 / 21.1 |
| T4 | 25.8 / 20.2 / 23.0 | 26.5 -> 26.1 / 21.7 / 22.4 | 32.0 -> 39.5 / 33.3 / 34.5 |
| 1080 Ti | 13.3 / 11.5 / 14.5 | 12.6 / 8.5 / 13.2 | 16.1 / 9.5 / 15.5 |
| 2080 Ti | 37.3 / 32.0 / 33.7 | 36.0 / 32.1 / 31.1 | 56.4 / 49.5 / 48.1 |
| V100 | 31.1 / 27.1 / 28.7 | - | 45.8 / 42.8 / 42.9 |

  Speculation now helps (or matches) with our backend everywhere but plain decoding stays ahead on the P100 here (a 3B model on
  a 1.5k prompt; the draft / lookup acceptance decides). V100: a first run (ours 24.2 / 36.9 tok/s, behind SDPA / eager) was the
  shared box's load (load average 16): measured again alternated, the row above (and the campaign's decode, 32.8 vs 28.6 / 29.8).
  Host time per attention call of a decode step on the V100 (Qwen2.5-3B shapes): ours 61 us, HF's SDPA 175, eager 147.

## Step (b), done
- Packed GQA rows in the cache kernel (`FlashAttnParams::pack_gqa`, `gqa_rows_per_token`: one rule for the host, the launcher
  and the kernel): a block covers rows (token, query head of the group) of one KV head, token-major; `fwd_row` is the one place
  rows split into (token, head offset): the Q loader (`load_tile_g2s_rows`, the tile loader with a row-address function; the
  one-stride loader is its wrapper), the KeyWindow masks and the band / block key bounds (`fwd_token<kCache>`), the partial rows
  (`fwd_part_row`). Sinks: one per row (`BandSoftmax::init_rows`, see step c).
- Split count, measured on packed shapes (`split_sweep.py`): the cap of step (a), partials within 1/4 of the K / V read, kept
  too few splits once packing shrank the grid (GQA 16/2, 1.5k cache, q_len 11: 2 KV heads x 1-3 blocks; P100 351 us at the
  automatic count against 68 with 16 splits). Now at least max(128, rows) keys per split (partials within ~2x of the read) and
  4 resident blocks per SM on Pascal (compute bound, small tiles), 1 on SM 7.x: that shape 351 -> 82 us (P100), 59 -> 26 (T4),
  109 -> 33 (V100), 257 -> 86 (1080 Ti); decode unchanged (rows <= 8).
- One layer, d=128, B=1, us: HF's path before P6.8 -> step (a) -> step (b) (one-token decode x q_len for reference):

| GPU | GQA 32/8, 16k, q_len 16 | GQA 32/8, 4k, q_len 4 | GQA 24/2, 4k, q_len 1 (SDPA before) | GQA 24/2, 4k, q_len 8 |
|---|---|---|---|---|
| V100 | 1202 -> 628 -> 181 (1819) | 307 -> 170 -> 58 (193) | 326 -> 157 -> 23 | 306 -> 161 -> 47 |
| T4 | 592 -> 602 -> 296 (4359) | 154 -> 168 -> 92 (346) | 283 -> 123 -> 34 | 154 -> 130 -> 70 |
| 2080 Ti | 721 -> 447 -> 165 (2192) | 186 -> 122 -> 56 (178) | 225 -> 91 -> 17 | 186 -> 96 -> 39 |
| P100 | 7547 -> 3306 -> 1184 (3039) | 1894 -> 850 -> 195 (275) | 1130 -> 666 -> 70 | 1896 -> 684 -> 160 |
| 1080 Ti | 5435 -> 2632 -> 1296 (3670) | 1364 -> 671 -> 198 (360) | 7218 -> 444 -> 68 | 1386 -> 455 -> 197 |

  A verification step of 16 tokens now costs 1.1-1.6x the one-token decode on SM 7.x (the cache read once per KV head); on
  Pascal it is compute bound (fp32 FMA: q_len 16, GQA 4, 16k is ~4.3 GFLOP, ~0.46 ms at the P100's peak, 1.18 ms measured).
- Plain forward against a HEAD build, alternated on each box: within noise on V100, T4, 1080 Ti, 2080 Ti; P100 d=64 +1.4%, d=256
  +1.1%, d=128 -0.7% (the generalised tile loader's inlining, the only change on that path).
- End to end (as above, 1.5k prompt) ours / SDPA / eager, prompt lookup tok/s: V100 46.6 / 42.8 / 42.6, T4 39.8 / 33.2 / 34.5,
  2080 Ti 56.2 / 49.7 / 50.2, P100 23.3 / 19.1 / 21.1, 1080 Ti 16.2 / 9.5 / 14.8: ours fastest everywhere, outputs identical to
  plain greedy, no fallback (at 1.5k the attention is a small part of the step; the kernel gains grow with the cache).

## Step (c), done
- HF decode (q_len 1) with more than 8 query heads per KV head (MQA, GQA 12:1 / 16:1) goes to `flash_attn_with_kvcache` (the
  packed forward) in FP16 without a padding mask; BF16 or a key mask keep the SDPA fallback (`decode` reason, recorded). Test:
  an MQA 16:1 model's generate on the kernel, every step within the noise of FP32 eager; BF16 recorded as fallback.
- `key_mask` beyond 8 rows done: the cache kernel masks the keys the mask drops in the masked step (every block takes it with a
  mask) and its K / V loaders read them as zeros (`kNullRows`: never read, so a NaN there cannot reach O through P = 0 times
  V); HF short queries with padding go there with the padding as key mask (no unpad plan, no host sync), and MQA / GQA > 8:1
  decode with padding too. Possible refinement: the mask byte is read per score element and per 16-byte vector, and blocks or
  splits whose keys are all dropped still run (the decode kernel shrinks its splits to the kept keys): a per-block bitmask in
  shared memory and split bounds from the mask.
- Paged caches beyond 8 rows done: the cache kernel's K / V row of key j comes from `paged_row_offset` (one rule with the
  decode kernel: block_table[j / block_size], slot j % block_size); the per-row loader is taken only with a key mask or a
  paged cache (a uniform branch per tile): on contiguous unmasked caches the one-stride loader, as fast as before (the per-row
  function on every 16-byte vector cost up to 22% on the P100). Equal bit for bit to the same cache contiguous (NHD, HND,
  vLLM layouts; block sizes 16 / 32 / 48 / 64).
- Still open (beyond 8 rows): BF16. Sinks with packed rows done: `fwd_softmax_init`
  gives each row of the cache kernel its query head's sink (split 0; the plain forward keeps the head's one). `cache_leftpad` done: the cache
  kernel starts K / V of row b at leftpad[b] and ends at cache_seqlens[b] (causal / window aligned to the end, as decode).

## Step (d), done
- `benchmarks/speculative.py` (kernel: one verification call per path; generate: plain / prompt lookup / assisted end to end),
  results of the five GPUs in `docs/bench/speculative.md`; `occupancy_table.md` regenerated (`*_cache` rows: the cache kernel).
- Measured there: on short caches (1k) the T4's cache path is a little slower than the plain forward at q_len 8-16 (fp32
  partials and the combine also for one split): a possible refinement (write O directly when one split is chosen).

## Acceptance
- A verification step (q_len 2-16, cache 4-16k, GQA 4 and 12) within ~1.5x of its floor: the one-token decode time (the cache
  read) on SM 7.x; on Pascal the larger of that and the fp32 compute floor (4 d R N H_kv FLOP: q_len 16, GQA 4, 4k, d 128,
  H_kv 8 is 1.07 GFLOP, ~115 us at the P100's peak against today's 1926), and faster than SDPA everywhere.
- HF assisted generation and prompt lookup on the P100 faster than plain decoding with our backend (today slower), and not
  slower than SDPA / eager on any GPU; outputs identical to plain greedy.
- No new fallback; tests green on V100, T4, P100, 1080 Ti (and the 2080 Ti while available).
