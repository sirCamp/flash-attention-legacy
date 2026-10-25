# Speculative / assisted decoding: short queries against a long cache

`python benchmarks/speculative.py kernel` and `... generate --model Qwen/Qwen2.5-3B --draft Qwen/Qwen2.5-0.5B` (design and
measurements per step: `docs/analysis/kernels/spec_decode_design.md`, PLAN P6.8). Kernels with the cache kernel's per-row K / V
as a template parameter (`kRows`, 585b76c); torch 2.7.1, transformers 5.20.2.

- kernel: one attention call of a verification step, d=128, B=1, us. `forward` = the plain forward kernel on the cache (HF's
  path before P6.8); `kvcache` = flash_attn_with_kvcache (the decode kernel up to 8 query rows per KV head, the forward with key
  splits and packed GQA rows beyond: HF's path now for short queries (q_len 2-64) against a longer cache and for MQA /
  GQA > 8:1 decode); `1tok x q_len` = what plain decoding pays for the same tokens (at q_len 1 the same call as `kvcache`,
  timed separately); `sdpa` = HF's SDPA (KV repeated, causal mask).
- generate: HF greedy generate of 128 tokens after a ~1.5k-token prompt (code to rewrite: prompt lookup finds candidates);
  plain, prompt lookup (10 tokens) and assisted (Qwen2.5-0.5B draft, same implementation); "output = plain greedy" checks the
  speculative modes return the same tokens. The prompt is the start of `flash_attn_legacy/integrations.py` (1491 tokens).
  "first 35 tokens": the output equals plain greedy for 35 tokens, then takes the other side of an fp16 near-tie (a
  verification step, q_len > 1, rounds a logit differently than one-token decode); it happens with every implementation
  (ours, SDPA, eager), which one depends on the GPU, always at the same token: not a kernel error.

Reading the numbers: on short caches (1k) the T4's `kvcache` is a little slower than the plain forward at q_len 8-16 (it always
writes fp32 partials and runs the combine, also for one split); from 4k on and on every other GPU it is faster, and the
verification of 16 tokens costs about one to three one-token decodes on SM 7.x (the cache read once per KV head). Pascal is
compute bound (fp32 FMA); SDPA on the 1080 Ti runs fp16 at 1/64 rate (sm_61), hence its ~10x slower cells. At 1.5k
tokens the attention is a small part of a decode step: the end-to-end gain is bounded there and grows with the context.

#### Tesla V100-SXM2-16GB: one layer, d=128, B=1, us

| H_q / H_kv | N | q_len | rows / KV head | forward | kvcache | 1tok x q_len | sdpa |
|---|---|---|---|---|---|---|---|
| 32/8 | 1024 | 1 | 4 | - | 18 | 18 | 92 |
| 32/8 | 1024 | 2 | 8 | 94 | 32 | 43 | 92 |
| 32/8 | 1024 | 4 | 16 | 93 | 27 | 86 | 93 |
| 32/8 | 1024 | 8 | 32 | 94 | 30 | 172 | 93 |
| 32/8 | 1024 | 16 | 64 | 94 | 37 | 344 | 95 |
| 32/8 | 4096 | 1 | 4 | - | 43 | 43 | 322 |
| 32/8 | 4096 | 2 | 8 | 350 | 62 | 94 | 323 |
| 32/8 | 4096 | 4 | 16 | 350 | 59 | 188 | 324 |
| 32/8 | 4096 | 8 | 32 | 350 | 67 | 377 | 326 |
| 32/8 | 4096 | 16 | 64 | 350 | 83 | 754 | 330 |
| 32/8 | 16384 | 1 | 4 | - | 104 | 104 | 1238 |
| 32/8 | 16384 | 2 | 8 | 1376 | 142 | 215 | 1240 |
| 32/8 | 16384 | 4 | 16 | 1376 | 159 | 426 | 1244 |
| 32/8 | 16384 | 8 | 32 | 1376 | 165 | 860 | 1252 |
| 32/8 | 16384 | 16 | 64 | 1376 | 186 | 1704 | 1269 |
| 24/2 | 1024 | 1 | 12 | - | 24 | - | 92 |
| 24/2 | 1024 | 2 | 24 | 93 | 26 | - | 92 |
| 24/2 | 1024 | 4 | 48 | 93 | 28 | - | 92 |
| 24/2 | 1024 | 8 | 96 | 93 | 36 | - | 93 |
| 24/2 | 1024 | 16 | 192 | 93 | 54 | - | 94 |
| 24/2 | 4096 | 1 | 12 | - | 24 | - | 321 |
| 24/2 | 4096 | 2 | 24 | 348 | 30 | - | 321 |
| 24/2 | 4096 | 4 | 48 | 348 | 34 | - | 322 |
| 24/2 | 4096 | 8 | 96 | 348 | 47 | - | 324 |
| 24/2 | 4096 | 16 | 192 | 349 | 72 | - | 328 |
| 24/2 | 16384 | 1 | 12 | - | 56 | - | 1235 |
| 24/2 | 16384 | 2 | 24 | 1369 | 63 | - | 1236 |
| 24/2 | 16384 | 4 | 48 | 1369 | 73 | - | 1240 |
| 24/2 | 16384 | 8 | 96 | 1369 | 105 | - | 1248 |
| 24/2 | 16384 | 16 | 192 | 1369 | 180 | - | 1263 |

#### Tesla V100-SXM2-16GB: Qwen/Qwen2.5-3B, fp16, 128 new tokens (tok/s, prefill included)

| implementation | prompt tokens | mode | tok/s | output = plain greedy |
|---|---|---|---|---|
| flash_attn_legacy | 1491 | plain | 31.9 | yes |
| flash_attn_legacy | 1491 | prompt lookup | 141.4 | yes |
| flash_attn_legacy | 1491 | assisted | 37.5 | first 35 tokens |
| sdpa | 1491 | plain | 28.4 | yes |
| sdpa | 1491 | prompt lookup | 118.4 | yes |
| sdpa | 1491 | assisted | 32.4 | yes |
| eager | 1491 | plain | 28.9 | yes |
| eager | 1491 | prompt lookup | 119.6 | first 35 tokens |
| eager | 1491 | assisted | 32.9 | yes |

#### Tesla T4: one layer, d=128, B=1, us

| H_q / H_kv | N | q_len | rows / KV head | forward | kvcache | 1tok x q_len | sdpa |
|---|---|---|---|---|---|---|---|
| 32/8 | 1024 | 1 | 4 | - | 20 | 20 | 90 |
| 32/8 | 1024 | 2 | 8 | 44 | 41 | 51 | 90 |
| 32/8 | 1024 | 4 | 16 | 45 | 42 | 101 | 91 |
| 32/8 | 1024 | 8 | 32 | 45 | 48 | 197 | 92 |
| 32/8 | 1024 | 16 | 64 | 45 | 58 | 381 | 94 |
| 32/8 | 4096 | 1 | 4 | - | 81 | 81 | 301 |
| 32/8 | 4096 | 2 | 8 | 155 | 132 | 173 | 302 |
| 32/8 | 4096 | 4 | 16 | 154 | 92 | 346 | 304 |
| 32/8 | 4096 | 8 | 32 | 155 | 100 | 691 | 308 |
| 32/8 | 4096 | 16 | 64 | 155 | 117 | 1381 | 317 |
| 32/8 | 16384 | 1 | 4 | - | 268 | 268 | 1134 |
| 32/8 | 16384 | 2 | 8 | 592 | 309 | 545 | 1139 |
| 32/8 | 16384 | 4 | 16 | 593 | 272 | 1090 | 1142 |
| 32/8 | 16384 | 8 | 32 | 594 | 281 | 2181 | 1159 |
| 32/8 | 16384 | 16 | 64 | 594 | 297 | 4361 | 1196 |
| 24/2 | 1024 | 1 | 12 | - | 16 | - | 82 |
| 24/2 | 1024 | 2 | 24 | 43 | 19 | - | 83 |
| 24/2 | 1024 | 4 | 48 | 43 | 21 | - | 84 |
| 24/2 | 1024 | 8 | 96 | 43 | 27 | - | 86 |
| 24/2 | 1024 | 16 | 192 | 44 | 36 | - | 88 |
| 24/2 | 4096 | 1 | 12 | - | 34 | - | 283 |
| 24/2 | 4096 | 2 | 24 | 152 | 45 | - | 284 |
| 24/2 | 4096 | 4 | 48 | 153 | 53 | - | 285 |
| 24/2 | 4096 | 8 | 96 | 153 | 71 | - | 290 |
| 24/2 | 4096 | 16 | 192 | 154 | 86 | - | 299 |
| 24/2 | 16384 | 1 | 12 | - | 87 | - | 1079 |
| 24/2 | 16384 | 2 | 24 | 591 | 96 | - | 1083 |
| 24/2 | 16384 | 4 | 48 | 591 | 107 | - | 1090 |
| 24/2 | 16384 | 8 | 96 | 592 | 129 | - | 1112 |
| 24/2 | 16384 | 16 | 192 | 591 | 173 | - | 1151 |

#### Tesla T4: Qwen/Qwen2.5-3B, fp16, 128 new tokens (tok/s, prefill included)

| implementation | prompt tokens | mode | tok/s | output = plain greedy |
|---|---|---|---|---|
| flash_attn_legacy | 1491 | plain | 25.9 | yes |
| flash_attn_legacy | 1491 | prompt lookup | 96.3 | yes |
| flash_attn_legacy | 1491 | assisted | 30.5 | first 35 tokens |
| sdpa | 1491 | plain | 20.1 | yes |
| sdpa | 1491 | prompt lookup | 76.2 | first 35 tokens |
| sdpa | 1491 | assisted | 24.6 | first 35 tokens |
| eager | 1491 | plain | 22.8 | yes |
| eager | 1491 | prompt lookup | 76.4 | first 35 tokens |
| eager | 1491 | assisted | 25.9 | first 35 tokens |

#### Tesla P100-PCIE-16GB: one layer, d=128, B=1, us

| H_q / H_kv | N | q_len | rows / KV head | forward | kvcache | 1tok x q_len | sdpa |
|---|---|---|---|---|---|---|---|
| 32/8 | 1024 | 1 | 4 | - | 34 | 34 | 302 |
| 32/8 | 1024 | 2 | 8 | 480 | 55 | 71 | 298 |
| 32/8 | 1024 | 4 | 16 | 479 | 76 | 141 | 298 |
| 32/8 | 1024 | 8 | 32 | 478 | 86 | 285 | 303 |
| 32/8 | 1024 | 16 | 64 | 479 | 142 | 563 | 304 |
| 32/8 | 4096 | 1 | 4 | - | 65 | 65 | 1131 |
| 32/8 | 4096 | 2 | 8 | 1883 | 105 | 140 | 1133 |
| 32/8 | 4096 | 4 | 16 | 1884 | 192 | 276 | 1138 |
| 32/8 | 4096 | 8 | 32 | 1884 | 225 | 562 | 1145 |
| 32/8 | 4096 | 16 | 64 | 1884 | 361 | 1100 | 1160 |
| 32/8 | 16384 | 1 | 4 | - | 182 | 184 | 4484 |
| 32/8 | 16384 | 2 | 8 | 7507 | 280 | 378 | 4486 |
| 32/8 | 16384 | 4 | 16 | 7506 | 548 | 756 | 4504 |
| 32/8 | 16384 | 8 | 32 | 7507 | 616 | 1516 | 4544 |
| 32/8 | 16384 | 16 | 64 | 7506 | 1164 | 3015 | 4577 |
| 24/2 | 1024 | 1 | 12 | - | 71 | - | 295 |
| 24/2 | 1024 | 2 | 24 | 479 | 74 | - | 296 |
| 24/2 | 1024 | 4 | 48 | 479 | 75 | - | 298 |
| 24/2 | 1024 | 8 | 96 | 479 | 75 | - | 300 |
| 24/2 | 1024 | 16 | 192 | 479 | 120 | - | 304 |
| 24/2 | 4096 | 1 | 12 | - | 69 | - | 1128 |
| 24/2 | 4096 | 2 | 24 | 1885 | 83 | - | 1131 |
| 24/2 | 4096 | 4 | 48 | 1886 | 141 | - | 1133 |
| 24/2 | 4096 | 8 | 96 | 1886 | 160 | - | 1144 |
| 24/2 | 4096 | 16 | 192 | 1885 | 327 | - | 1152 |
| 24/2 | 16384 | 1 | 12 | - | 179 | - | 4464 |
| 24/2 | 16384 | 2 | 24 | 7509 | 211 | - | 4470 |
| 24/2 | 16384 | 4 | 48 | 7509 | 336 | - | 4487 |
| 24/2 | 16384 | 8 | 96 | 7510 | 488 | - | 4520 |
| 24/2 | 16384 | 16 | 192 | 7510 | 1015 | - | 4557 |

#### Tesla P100-PCIE-16GB: Qwen/Qwen2.5-3B, fp16, 128 new tokens (tok/s, prefill included)

| implementation | prompt tokens | mode | tok/s | output = plain greedy |
|---|---|---|---|---|
| flash_attn_legacy | 1491 | plain | 26.8 | yes |
| flash_attn_legacy | 1491 | prompt lookup | 46.0 | yes |
| flash_attn_legacy | 1491 | assisted | 29.7 | yes |
| sdpa | 1491 | plain | 20.7 | yes |
| sdpa | 1491 | prompt lookup | 39.2 | yes |
| sdpa | 1491 | assisted | 25.3 | first 35 tokens |
| eager | 1491 | plain | 23.0 | yes |
| eager | 1491 | prompt lookup | 41.9 | first 35 tokens |
| eager | 1491 | assisted | 26.9 | yes |

#### NVIDIA GeForce GTX 1080 Ti: one layer, d=128, B=1, us

| H_q / H_kv | N | q_len | rows / KV head | forward | kvcache | 1tok x q_len | sdpa |
|---|---|---|---|---|---|---|---|
| 32/8 | 1024 | 1 | 4 | - | 43 | 41 | 3531 |
| 32/8 | 1024 | 2 | 8 | 341 | 85 | 100 | 3519 |
| 32/8 | 1024 | 4 | 16 | 341 | 79 | 201 | 3529 |
| 32/8 | 1024 | 8 | 32 | 341 | 96 | 401 | 3546 |
| 32/8 | 1024 | 16 | 64 | 341 | 155 | 803 | 3548 |
| 32/8 | 4096 | 1 | 4 | - | 79 | 79 | 14015 |
| 32/8 | 4096 | 2 | 8 | 1346 | 187 | 180 | 14028 |
| 32/8 | 4096 | 4 | 16 | 1347 | 202 | 360 | 14062 |
| 32/8 | 4096 | 8 | 32 | 1346 | 229 | 721 | 14128 |
| 32/8 | 4096 | 16 | 64 | 1346 | 374 | 1442 | 14136 |
| 32/8 | 16384 | 1 | 4 | - | 216 | 216 | 55982 |
| 32/8 | 16384 | 2 | 8 | 5363 | 358 | 457 | 56047 |
| 32/8 | 16384 | 4 | 16 | 5365 | 572 | 913 | 56191 |
| 32/8 | 16384 | 8 | 32 | 5365 | 690 | 1835 | 56477 |
| 32/8 | 16384 | 16 | 64 | 5364 | 1290 | 3654 | 56501 |
| 24/2 | 1024 | 1 | 12 | - | 54 | - | 1814 |
| 24/2 | 1024 | 2 | 24 | 346 | 62 | - | 1817 |
| 24/2 | 1024 | 4 | 48 | 346 | 63 | - | 1823 |
| 24/2 | 1024 | 8 | 96 | 346 | 63 | - | 1831 |
| 24/2 | 1024 | 16 | 192 | 346 | 112 | - | 1832 |
| 24/2 | 4096 | 1 | 12 | - | 67 | - | 7192 |
| 24/2 | 4096 | 2 | 24 | 1366 | 96 | - | 7201 |
| 24/2 | 4096 | 4 | 48 | 1366 | 143 | - | 7219 |
| 24/2 | 4096 | 8 | 96 | 1366 | 199 | - | 7254 |
| 24/2 | 4096 | 16 | 192 | 1366 | 326 | - | 7258 |
| 24/2 | 16384 | 1 | 12 | - | 185 | - | 28698 |
| 24/2 | 16384 | 2 | 24 | 5448 | 215 | - | 28738 |
| 24/2 | 16384 | 4 | 48 | 5448 | 373 | - | 28802 |
| 24/2 | 16384 | 8 | 96 | 5447 | 572 | - | 28943 |
| 24/2 | 16384 | 16 | 192 | 5448 | 1069 | - | 28963 |

#### NVIDIA GeForce GTX 1080 Ti: Qwen/Qwen2.5-3B, fp16, 128 new tokens (tok/s, prefill included)

| implementation | prompt tokens | mode | tok/s | output = plain greedy |
|---|---|---|---|---|
| flash_attn_legacy | 1491 | plain | 14.8 | yes |
| flash_attn_legacy | 1491 | prompt lookup | 32.8 | yes |
| flash_attn_legacy | 1491 | assisted | 16.2 | yes |
| sdpa | 1491 | plain | 13.0 | yes |
| sdpa | 1491 | prompt lookup | 20.6 | yes |
| sdpa | 1491 | assisted | 12.6 | yes |
| eager | 1491 | plain | 13.2 | yes |
| eager | 1491 | prompt lookup | 29.3 | yes |
| eager | 1491 | assisted | 14.7 | yes |

#### NVIDIA GeForce RTX 2080 Ti: one layer, d=128, B=1, us

| H_q / H_kv | N | q_len | rows / KV head | forward | kvcache | 1tok x q_len | sdpa |
|---|---|---|---|---|---|---|---|
| 32/8 | 1024 | 1 | 4 | - | 24 | 24 | 109 |
| 32/8 | 1024 | 2 | 8 | 92 | 46 | 57 | 110 |
| 32/8 | 1024 | 4 | 16 | 93 | 35 | 114 | 110 |
| 32/8 | 1024 | 8 | 32 | 93 | 37 | 227 | 111 |
| 32/8 | 1024 | 16 | 64 | 94 | 46 | 459 | 114 |
| 32/8 | 4096 | 1 | 4 | - | 51 | 51 | 382 |
| 32/8 | 4096 | 2 | 8 | 337 | 98 | 111 | 383 |
| 32/8 | 4096 | 4 | 16 | 337 | 87 | 224 | 385 |
| 32/8 | 4096 | 8 | 32 | 337 | 95 | 444 | 390 |
| 32/8 | 4096 | 16 | 64 | 338 | 113 | 888 | 400 |
| 32/8 | 16384 | 1 | 4 | - | 147 | 148 | 1479 |
| 32/8 | 16384 | 2 | 8 | 1310 | 267 | 306 | 1483 |
| 32/8 | 16384 | 4 | 16 | 1310 | 201 | 613 | 1489 |
| 32/8 | 16384 | 8 | 32 | 1310 | 209 | 1224 | 1507 |
| 32/8 | 16384 | 16 | 64 | 1311 | 228 | 2448 | 1538 |
| 24/2 | 1024 | 1 | 12 | - | 27 | - | 109 |
| 24/2 | 1024 | 2 | 24 | 93 | 32 | - | 109 |
| 24/2 | 1024 | 4 | 48 | 93 | 34 | - | 110 |
| 24/2 | 1024 | 8 | 96 | 93 | 45 | - | 112 |
| 24/2 | 1024 | 16 | 192 | 94 | 63 | - | 114 |
| 24/2 | 4096 | 1 | 12 | - | 30 | - | 384 |
| 24/2 | 4096 | 2 | 24 | 336 | 37 | - | 385 |
| 24/2 | 4096 | 4 | 48 | 336 | 43 | - | 385 |
| 24/2 | 4096 | 8 | 96 | 336 | 61 | - | 390 |
| 24/2 | 4096 | 16 | 192 | 337 | 86 | - | 399 |
| 24/2 | 16384 | 1 | 12 | - | 77 | - | 1475 |
| 24/2 | 16384 | 2 | 24 | 1313 | 88 | - | 1486 |
| 24/2 | 16384 | 4 | 48 | 1312 | 97 | - | 1491 |
| 24/2 | 16384 | 8 | 96 | 1311 | 147 | - | 1502 |
| 24/2 | 16384 | 16 | 192 | 1312 | 224 | - | 1538 |

#### NVIDIA GeForce RTX 2080 Ti: Qwen/Qwen2.5-3B, fp16, 128 new tokens (tok/s, prefill included)

| implementation | prompt tokens | mode | tok/s | output = plain greedy |
|---|---|---|---|---|
| flash_attn_legacy | 1491 | plain | 35.0 | yes |
| flash_attn_legacy | 1491 | prompt lookup | 128.0 | yes |
| flash_attn_legacy | 1491 | assisted | 40.5 | first 35 tokens |
| sdpa | 1491 | plain | 30.5 | yes |
| sdpa | 1491 | prompt lookup | 102.7 | yes |
| sdpa | 1491 | assisted | 33.6 | yes |
| eager | 1491 | plain | 30.8 | yes |
| eager | 1491 | prompt lookup | 104.8 | first 35 tokens |
| eager | 1491 | assisted | 34.3 | yes |
