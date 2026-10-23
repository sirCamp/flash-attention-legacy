# Speculative / assisted decoding: short queries against a long cache

`python benchmarks/speculative.py kernel` and `... generate --model Qwen/Qwen2.5-3B --draft Qwen/Qwen2.5-0.5B` (design and
measurements per step: `docs/analysis/kernels/spec_decode_design.md`, PLAN P6.8). Kernels at the commit of P6.8 step c.

- kernel: one attention call of a verification step, d=128, B=1, us. `forward` = the plain forward kernel on the cache (HF's
  path before P6.8); `kvcache` = flash_attn_with_kvcache (the decode kernel up to 8 query rows per KV head, the forward with key
  splits and packed GQA rows beyond: HF's path now for q_len 2-64 after a longer cache and for MQA / GQA > 8:1 decode);
  `1tok x q_len` = what plain decoding pays for the same tokens; `sdpa` = HF's SDPA (KV repeated, causal mask).
- generate: HF greedy generate of 128 tokens after a ~1.5k-token prompt (code to rewrite: prompt lookup finds candidates);
  plain, prompt lookup (10 tokens) and assisted (Qwen2.5-0.5B draft, same implementation); "output = plain greedy" checks the
  speculative modes return the same tokens.

Reading the numbers: on short caches (1k) the T4's `kvcache` is a little slower than the plain forward at q_len 8-16 (it always
writes fp32 partials and runs the combine, also for one split); from 4k on and on every other GPU it is faster, and the
verification of 16 tokens costs about one to two one-token decodes on SM 7.x (the cache read once per KV head). Pascal is
compute bound (fp32 FMA). At 1.5k tokens the attention is a small part of a decode step: the end-to-end gain is bounded there
and grows with the context.


#### Tesla V100-SXM2-32GB: one layer, d=128, B=1, us

| H_q / H_kv | N | q_len | rows / KV head | forward | kvcache | 1tok x q_len | sdpa |
|---|---|---|---|---|---|---|---|
| 32/8 | 1024 | 1 | 4 | - | 22 | 22 | 99 |
| 32/8 | 1024 | 2 | 8 | 84 | 31 | 43 | 94 |
| 32/8 | 1024 | 4 | 16 | 84 | 27 | 86 | 95 |
| 32/8 | 1024 | 8 | 32 | 84 | 30 | 172 | 95 |
| 32/8 | 1024 | 16 | 64 | 84 | 37 | 344 | 97 |
| 32/8 | 4096 | 1 | 4 | - | 46 | 46 | 330 |
| 32/8 | 4096 | 2 | 8 | 308 | 65 | 100 | 330 |
| 32/8 | 4096 | 4 | 16 | 308 | 58 | 197 | 331 |
| 32/8 | 4096 | 8 | 32 | 308 | 66 | 406 | 333 |
| 32/8 | 4096 | 16 | 64 | 308 | 85 | 786 | 337 |
| 32/8 | 16384 | 1 | 4 | - | 110 | 110 | 1261 |
| 32/8 | 16384 | 2 | 8 | 1205 | 153 | 227 | 1262 |
| 32/8 | 16384 | 4 | 16 | 1205 | 150 | 446 | 1266 |
| 32/8 | 16384 | 8 | 32 | 1205 | 157 | 901 | 1275 |
| 32/8 | 16384 | 16 | 64 | 1206 | 182 | 1819 | 1293 |
| 24/2 | 1024 | 1 | 12 | - | 23 | - | 93 |
| 24/2 | 1024 | 2 | 24 | 83 | 26 | - | 93 |
| 24/2 | 1024 | 4 | 48 | 83 | 28 | - | 93 |
| 24/2 | 1024 | 8 | 96 | 83 | 36 | - | 94 |
| 24/2 | 1024 | 16 | 192 | 84 | 54 | - | 95 |
| 24/2 | 4096 | 1 | 12 | - | 24 | - | 327 |
| 24/2 | 4096 | 2 | 24 | 308 | 29 | - | 328 |
| 24/2 | 4096 | 4 | 48 | 308 | 33 | - | 328 |
| 24/2 | 4096 | 8 | 96 | 307 | 47 | - | 330 |
| 24/2 | 4096 | 16 | 192 | 307 | 71 | - | 334 |
| 24/2 | 16384 | 1 | 12 | - | 54 | - | 1253 |
| 24/2 | 16384 | 2 | 24 | 1203 | 61 | - | 1253 |
| 24/2 | 16384 | 4 | 48 | 1203 | 72 | - | 1257 |
| 24/2 | 16384 | 8 | 96 | 1203 | 108 | - | 1269 |
| 24/2 | 16384 | 16 | 192 | 1203 | 180 | - | 1286 |

#### Tesla V100-SXM2-32GB: Qwen/Qwen2.5-3B, fp16, 128 new tokens (decode tok/s)
| implementation | prompt tokens | mode | tok/s | output = plain greedy |
|---|---|---|---|---|
| flash_attn_legacy | 1490 | plain | 31.6 | yes |
| flash_attn_legacy | 1490 | prompt lookup | 47.3 | yes |
| flash_attn_legacy | 1490 | assisted | 31.0 | yes |
| sdpa | 1490 | plain | 26.8 | yes |
| sdpa | 1490 | prompt lookup | 42.3 | yes |
| sdpa | 1490 | assisted | 27.2 | yes |
| eager | 1490 | plain | 28.6 | yes |
| eager | 1490 | prompt lookup | 42.9 | yes |
| eager | 1490 | assisted | 28.1 | yes |

#### Tesla T4: one layer, d=128, B=1, us

| H_q / H_kv | N | q_len | rows / KV head | forward | kvcache | 1tok x q_len | sdpa |
|---|---|---|---|---|---|---|---|
| 32/8 | 1024 | 1 | 4 | - | 20 | 20 | 90 |
| 32/8 | 1024 | 2 | 8 | 44 | 41 | 51 | 90 |
| 32/8 | 1024 | 4 | 16 | 44 | 43 | 103 | 92 |
| 32/8 | 1024 | 8 | 32 | 45 | 49 | 197 | 92 |
| 32/8 | 1024 | 16 | 64 | 46 | 57 | 378 | 94 |
| 32/8 | 4096 | 1 | 4 | - | 81 | 81 | 301 |
| 32/8 | 4096 | 2 | 8 | 155 | 132 | 173 | 302 |
| 32/8 | 4096 | 4 | 16 | 155 | 92 | 345 | 304 |
| 32/8 | 4096 | 8 | 32 | 155 | 100 | 693 | 309 |
| 32/8 | 4096 | 16 | 64 | 155 | 117 | 1380 | 319 |
| 32/8 | 16384 | 1 | 4 | - | 268 | 268 | 1136 |
| 32/8 | 16384 | 2 | 8 | 593 | 310 | 545 | 1139 |
| 32/8 | 16384 | 4 | 16 | 592 | 272 | 1090 | 1144 |
| 32/8 | 16384 | 8 | 32 | 592 | 280 | 2182 | 1161 |
| 32/8 | 16384 | 16 | 64 | 592 | 296 | 4358 | 1197 |
| 24/2 | 1024 | 1 | 12 | - | 16 | - | 82 |
| 24/2 | 1024 | 2 | 24 | 43 | 19 | - | 83 |
| 24/2 | 1024 | 4 | 48 | 43 | 21 | - | 85 |
| 24/2 | 1024 | 8 | 96 | 44 | 27 | - | 86 |
| 24/2 | 1024 | 16 | 192 | 44 | 35 | - | 89 |
| 24/2 | 4096 | 1 | 12 | - | 34 | - | 283 |
| 24/2 | 4096 | 2 | 24 | 154 | 46 | - | 283 |
| 24/2 | 4096 | 4 | 48 | 153 | 53 | - | 285 |
| 24/2 | 4096 | 8 | 96 | 153 | 71 | - | 291 |
| 24/2 | 4096 | 16 | 192 | 154 | 86 | - | 299 |
| 24/2 | 16384 | 1 | 12 | - | 86 | - | 1080 |
| 24/2 | 16384 | 2 | 24 | 592 | 96 | - | 1083 |
| 24/2 | 16384 | 4 | 48 | 591 | 107 | - | 1090 |
| 24/2 | 16384 | 8 | 96 | 592 | 129 | - | 1110 |
| 24/2 | 16384 | 16 | 192 | 594 | 170 | - | 1149 |

| flash_attn_legacy | 1490 | plain | 26.1 | yes |
| flash_attn_legacy | 1490 | prompt lookup | 39.9 | yes |
| flash_attn_legacy | 1490 | assisted | 26.3 | yes |
| sdpa | 1490 | plain | 20.2 | yes |
| sdpa | 1490 | prompt lookup | 33.3 | yes |
| sdpa | 1490 | assisted | 22.1 | yes |
| eager | 1490 | plain | 22.9 | yes |
| eager | 1490 | prompt lookup | 34.4 | yes |
| eager | 1490 | assisted | 22.2 | yes |
#### Tesla T4: Qwen/Qwen2.5-3B, fp16, 128 new tokens (decode tok/s)
| implementation | prompt tokens | mode | tok/s | output = plain greedy |
|---|---|---|---|---|

#### Tesla P100-PCIE-16GB: one layer, d=128, B=1, us

| H_q / H_kv | N | q_len | rows / KV head | forward | kvcache | 1tok x q_len | sdpa |
|---|---|---|---|---|---|---|---|
| 32/8 | 1024 | 1 | 4 | - | 34 | 34 | 302 |
| 32/8 | 1024 | 2 | 8 | 482 | 55 | 71 | 298 |
| 32/8 | 1024 | 4 | 16 | 478 | 76 | 142 | 298 |
| 32/8 | 1024 | 8 | 32 | 478 | 86 | 285 | 302 |
| 32/8 | 1024 | 16 | 64 | 478 | 142 | 561 | 305 |
| 32/8 | 4096 | 1 | 4 | - | 64 | 64 | 1132 |
| 32/8 | 4096 | 2 | 8 | 1884 | 105 | 137 | 1135 |
| 32/8 | 4096 | 4 | 16 | 1884 | 192 | 274 | 1138 |
| 32/8 | 4096 | 8 | 32 | 1884 | 225 | 550 | 1149 |
| 32/8 | 4096 | 16 | 64 | 1884 | 360 | 1097 | 1159 |
| 32/8 | 16384 | 1 | 4 | - | 183 | 183 | 4480 |
| 32/8 | 16384 | 2 | 8 | 7505 | 286 | 374 | 4487 |
| 32/8 | 16384 | 4 | 16 | 7506 | 549 | 756 | 4508 |
| 32/8 | 16384 | 8 | 32 | 7507 | 613 | 1519 | 4545 |
| 32/8 | 16384 | 16 | 64 | 7507 | 1162 | 3038 | 4576 |
| 24/2 | 1024 | 1 | 12 | - | 71 | - | 296 |
| 24/2 | 1024 | 2 | 24 | 479 | 74 | - | 296 |
| 24/2 | 1024 | 4 | 48 | 479 | 75 | - | 299 |
| 24/2 | 1024 | 8 | 96 | 479 | 75 | - | 301 |
| 24/2 | 1024 | 16 | 192 | 479 | 120 | - | 303 |
| 24/2 | 4096 | 1 | 12 | - | 69 | - | 1130 |
| 24/2 | 4096 | 2 | 24 | 1885 | 83 | - | 1132 |
| 24/2 | 4096 | 4 | 48 | 1886 | 141 | - | 1135 |
| 24/2 | 4096 | 8 | 96 | 1886 | 159 | - | 1144 |
| 24/2 | 4096 | 16 | 192 | 1886 | 328 | - | 1156 |
| 24/2 | 16384 | 1 | 12 | - | 180 | - | 4463 |
| 24/2 | 16384 | 2 | 24 | 7509 | 212 | - | 4472 |
| 24/2 | 16384 | 4 | 48 | 7510 | 336 | - | 4489 |
| 24/2 | 16384 | 8 | 96 | 7510 | 487 | - | 4525 |
| 24/2 | 16384 | 16 | 192 | 7510 | 1021 | - | 4563 |

| flash_attn_legacy | 1490 | plain | 27.0 | yes |
| flash_attn_legacy | 1490 | prompt lookup | 23.3 | yes |
| flash_attn_legacy | 1490 | assisted | 23.5 | yes |
| sdpa | 1490 | plain | 20.7 | yes |
| sdpa | 1490 | prompt lookup | 19.1 | yes |
| sdpa | 1490 | assisted | 19.4 | yes |
| eager | 1490 | plain | 23.0 | yes |
| eager | 1490 | prompt lookup | 21.1 | yes |
| eager | 1490 | assisted | 21.4 | yes |
#### Tesla P100-PCIE-16GB: Qwen/Qwen2.5-3B, fp16, 128 new tokens (decode tok/s)
| implementation | prompt tokens | mode | tok/s | output = plain greedy |
|---|---|---|---|---|

#### NVIDIA GeForce GTX 1080 Ti: one layer, d=128, B=1, us

| H_q / H_kv | N | q_len | rows / KV head | forward | kvcache | 1tok x q_len | sdpa |
|---|---|---|---|---|---|---|---|
| 32/8 | 1024 | 1 | 4 | - | 43 | 43 | 3913 |
| 32/8 | 1024 | 2 | 8 | 342 | 84 | 100 | 3486 |
| 32/8 | 1024 | 4 | 16 | 338 | 78 | 197 | 3495 |
| 32/8 | 1024 | 8 | 32 | 338 | 93 | 401 | 3511 |
| 32/8 | 1024 | 16 | 64 | 338 | 150 | 803 | 3514 |
| 32/8 | 4096 | 1 | 4 | - | 79 | 79 | 13875 |
| 32/8 | 4096 | 2 | 8 | 1332 | 186 | 180 | 13890 |
| 32/8 | 4096 | 4 | 16 | 1332 | 190 | 360 | 13927 |
| 32/8 | 4096 | 8 | 32 | 1332 | 222 | 721 | 13989 |
| 32/8 | 4096 | 16 | 64 | 1332 | 366 | 1437 | 13996 |
| 32/8 | 16384 | 1 | 4 | - | 216 | 215 | 55447 |
| 32/8 | 16384 | 2 | 8 | 5309 | 356 | 457 | 55512 |
| 32/8 | 16384 | 4 | 16 | 5311 | 562 | 913 | 55639 |
| 32/8 | 16384 | 8 | 32 | 5311 | 677 | 1827 | 56430 |
| 32/8 | 16384 | 16 | 64 | 5363 | 1280 | 3654 | 56465 |
| 24/2 | 1024 | 1 | 12 | - | 53 | - | 1814 |
| 24/2 | 1024 | 2 | 24 | 346 | 61 | - | 1817 |
| 24/2 | 1024 | 4 | 48 | 346 | 62 | - | 1823 |
| 24/2 | 1024 | 8 | 96 | 346 | 63 | - | 1831 |
| 24/2 | 1024 | 16 | 192 | 346 | 111 | - | 1832 |
| 24/2 | 4096 | 1 | 12 | - | 61 | - | 7188 |
| 24/2 | 4096 | 2 | 24 | 1366 | 93 | - | 7200 |
| 24/2 | 4096 | 4 | 48 | 1366 | 139 | - | 7216 |
| 24/2 | 4096 | 8 | 96 | 1366 | 194 | - | 7251 |
| 24/2 | 4096 | 16 | 192 | 1366 | 321 | - | 7255 |
| 24/2 | 16384 | 1 | 12 | - | 173 | - | 28679 |
| 24/2 | 16384 | 2 | 24 | 5441 | 212 | - | 28716 |
| 24/2 | 16384 | 4 | 48 | 5443 | 366 | - | 28795 |
| 24/2 | 16384 | 8 | 96 | 5443 | 563 | - | 28923 |
| 24/2 | 16384 | 16 | 192 | 5442 | 1058 | - | 28944 |

#### NVIDIA GeForce GTX 1080 Ti: Qwen/Qwen2.5-3B, fp16, 128 new tokens (decode tok/s)
| implementation | prompt tokens | mode | tok/s | output = plain greedy |
|---|---|---|---|---|
| flash_attn_legacy | 1490 | plain | 15.5 | yes |
| flash_attn_legacy | 1490 | prompt lookup | 16.9 | yes |
| flash_attn_legacy | 1490 | assisted | 14.5 | yes |
| sdpa | 1490 | plain | 14.2 | yes |
| sdpa | 1490 | prompt lookup | 9.8 | yes |
| sdpa | 1490 | assisted | 9.2 | yes |
| eager | 1490 | plain | 14.0 | yes |
| eager | 1490 | prompt lookup | 15.4 | yes |
| eager | 1490 | assisted | 12.8 | yes |

#### NVIDIA GeForce RTX 2080 Ti: one layer, d=128, B=1, us

| H_q / H_kv | N | q_len | rows / KV head | forward | kvcache | 1tok x q_len | sdpa |
|---|---|---|---|---|---|---|---|
| 32/8 | 1024 | 1 | 4 | - | 20 | 19 | 65 |
| 32/8 | 1024 | 2 | 8 | 51 | 25 | 31 | 65 |
| 32/8 | 1024 | 4 | 16 | 51 | 19 | 62 | 65 |
| 32/8 | 1024 | 8 | 32 | 51 | 21 | 122 | 66 |
| 32/8 | 1024 | 16 | 64 | 51 | 29 | 248 | 68 |
| 32/8 | 4096 | 1 | 4 | - | 41 | 41 | 223 |
| 32/8 | 4096 | 2 | 8 | 184 | 70 | 89 | 223 |
| 32/8 | 4096 | 4 | 16 | 183 | 55 | 175 | 224 |
| 32/8 | 4096 | 8 | 32 | 183 | 62 | 356 | 227 |
| 32/8 | 4096 | 16 | 64 | 184 | 74 | 710 | 233 |
| 32/8 | 16384 | 1 | 4 | - | 133 | 133 | 858 |
| 32/8 | 16384 | 2 | 8 | 713 | 160 | 273 | 868 |
| 32/8 | 16384 | 4 | 16 | 717 | 144 | 547 | 868 |
| 32/8 | 16384 | 8 | 32 | 715 | 151 | 1097 | 877 |
| 32/8 | 16384 | 16 | 64 | 716 | 164 | 2191 | 901 |
| 24/2 | 1024 | 1 | 12 | - | 14 | - | 64 |
| 24/2 | 1024 | 2 | 24 | 51 | 18 | - | 64 |
| 24/2 | 1024 | 4 | 48 | 51 | 19 | - | 64 |
| 24/2 | 1024 | 8 | 96 | 51 | 25 | - | 65 |
| 24/2 | 1024 | 16 | 192 | 51 | 34 | - | 67 |
| 24/2 | 4096 | 1 | 12 | - | 17 | - | 223 |
| 24/2 | 4096 | 2 | 24 | 184 | 21 | - | 224 |
| 24/2 | 4096 | 4 | 48 | 184 | 25 | - | 225 |
| 24/2 | 4096 | 8 | 96 | 184 | 38 | - | 227 |
| 24/2 | 4096 | 16 | 192 | 184 | 50 | - | 233 |
| 24/2 | 16384 | 1 | 12 | - | 51 | - | 858 |
| 24/2 | 16384 | 2 | 24 | 716 | 59 | - | 858 |
| 24/2 | 16384 | 4 | 48 | 715 | 67 | - | 861 |
| 24/2 | 16384 | 8 | 96 | 715 | 89 | - | 872 |
| 24/2 | 16384 | 16 | 192 | 715 | 127 | - | 893 |

#### NVIDIA GeForce RTX 2080 Ti: Qwen/Qwen2.5-3B, fp16, 128 new tokens (decode tok/s)
| implementation | prompt tokens | mode | tok/s | output = plain greedy |
|---|---|---|---|---|
| flash_attn_legacy | 1490 | plain | 37.6 | yes |
| flash_attn_legacy | 1490 | prompt lookup | 57.0 | yes |
| flash_attn_legacy | 1490 | assisted | 37.2 | yes |
| sdpa | 1490 | plain | 30.9 | yes |
| sdpa | 1490 | prompt lookup | 49.5 | yes |
| sdpa | 1490 | assisted | 31.3 | yes |
| eager | 1490 | plain | 33.5 | yes |
| eager | 1490 | prompt lookup | 49.4 | yes |
| eager | 1490 | assisted | 33.1 | yes |

[exited with code 0]
