| kernel | tile | warps/block | arch | regs | stack B | spill st/ld B | dyn smem B (KB) | blocks/SM | warps/SM (max) | blocks/SM limits |
|---|---|---|---|---|---|---|---|---|---|---|
| bwd_simt | BwdTileSimt&lt;16, 16, 128, 4> | 4 | sm_60 | 112 | 0 | 0/0 | 19712 (19.2) | 3 | 12/64 | by regs=4, by smem=3 |
| bwd_simt | BwdTileSimt&lt;16, 16, 128, 4> | 4 | sm_61 | 124 | 0 | 0/0 | 19712 (19.2) | 4 | 16/64 | by regs=4, by smem=4 |
| bwd_simt | BwdTileSimt&lt;16, 16, 128, 4> | 4 | sm_70 | 128 | 0 | 0/0 | 19712 (19.2) | 4 | 16/64 | by regs=4, by smem=4 |
| bwd_simt | BwdTileSimt&lt;16, 16, 128, 4> | 4 | sm_75 | 80 | 16 | 12/12 | 19712 (19.2) | 3 | 12/32 | by regs=6, by smem=3 |
| bwd_simt | BwdTileSimt&lt;16, 16, 256, 8> | 8 | sm_60 | 115 | 0 | 0/0 | 36096 (35.2) | 1 | 8/64 | by regs=2, by smem=1 |
| bwd_simt | BwdTileSimt&lt;16, 16, 256, 8> | 8 | sm_61 | 124 | 0 | 0/0 | 36096 (35.2) | 2 | 16/64 | by regs=2, by smem=2 |
| bwd_simt | BwdTileSimt&lt;16, 16, 256, 8> | 8 | sm_70 | 128 | 0 | 0/0 | 36096 (35.2) | 2 | 16/64 | by regs=2, by smem=2 |
| bwd_simt | BwdTileSimt&lt;16, 16, 256, 8> | 8 | sm_75 | 80 | 16 | 12/12 | 36096 (35.2) | 1 | 8/32 | by regs=3, by smem=1 |
| bwd_simt | BwdTileSimt&lt;32, 16, 96, 4> | 4 | sm_60 | 122 | 0 | 0/0 | 24576 (24.0) | 2 | 8/64 | by regs=4, by smem=2 |
| bwd_simt | BwdTileSimt&lt;32, 16, 96, 4> | 4 | sm_61 | 127 | 0 | 0/0 | 24576 (24.0) | 4 | 16/64 | by regs=4, by smem=4 |
| bwd_simt | BwdTileSimt&lt;32, 16, 96, 4> | 4 | sm_70 | 128 | 0 | 0/0 | 24576 (24.0) | 4 | 16/64 | by regs=4, by smem=4 |
| bwd_simt | BwdTileSimt&lt;32, 16, 96, 4> | 4 | sm_75 | 96 | 0 | 0/0 | 24576 (24.0) | 2 | 8/32 | by regs=5, by smem=2 |
| bwd_simt | BwdTileSimt&lt;32, 32, 32, 4> | 4 | sm_60 | 72 | 0 | 0/0 | 18944 (18.5) | 3 | 12/64 | by regs=7, by smem=3 |
| bwd_simt | BwdTileSimt&lt;32, 32, 32, 4> | 4 | sm_61 | 64 | 8 | 8/8 | 18944 (18.5) | 5 | 20/64 | by regs=8, by smem=5 |
| bwd_simt | BwdTileSimt&lt;32, 32, 32, 4> | 4 | sm_70 | 79 | 0 | 0/0 | 18944 (18.5) | 5 | 20/64 | by regs=6, by smem=5 |
| bwd_simt | BwdTileSimt&lt;32, 32, 32, 4> | 4 | sm_75 | 71 | 0 | 0/0 | 18944 (18.5) | 3 | 12/32 | by regs=7, by smem=3 |
| bwd_simt | BwdTileSimt&lt;32, 32, 64, 4> | 4 | sm_60 | 110 | 0 | 0/0 | 27136 (26.5) | 2 | 8/64 | by regs=4, by smem=2 |
| bwd_simt | BwdTileSimt&lt;32, 32, 64, 4> | 4 | sm_61 | 125 | 0 | 0/0 | 27136 (26.5) | 3 | 12/64 | by regs=4, by smem=3 |
| bwd_simt | BwdTileSimt&lt;32, 32, 64, 4> | 4 | sm_70 | 128 | 0 | 0/0 | 27136 (26.5) | 3 | 12/64 | by regs=4, by smem=3 |
| bwd_simt | BwdTileSimt&lt;32, 32, 64, 4> | 4 | sm_75 | 89 | 0 | 0/0 | 27136 (26.5) | 2 | 8/32 | by regs=5, by smem=2 |
| bwd_simt_varlen | BwdTileSimt&lt;16, 16, 128, 4> | 4 | sm_60 | 111 | 0 | 0/0 | 19712 (19.2) | 3 | 12/64 | by regs=4, by smem=3 |
| bwd_simt_varlen | BwdTileSimt&lt;16, 16, 128, 4> | 4 | sm_61 | 126 | 0 | 0/0 | 19712 (19.2) | 4 | 16/64 | by regs=4, by smem=4 |
| bwd_simt_varlen | BwdTileSimt&lt;16, 16, 128, 4> | 4 | sm_70 | 126 | 0 | 0/0 | 19712 (19.2) | 4 | 16/64 | by regs=4, by smem=4 |
| bwd_simt_varlen | BwdTileSimt&lt;16, 16, 128, 4> | 4 | sm_75 | 96 | 0 | 0/0 | 19712 (19.2) | 3 | 12/32 | by regs=5, by smem=3 |
| bwd_simt_varlen | BwdTileSimt&lt;16, 16, 256, 8> | 8 | sm_60 | 114 | 0 | 0/0 | 36096 (35.2) | 1 | 8/64 | by regs=2, by smem=1 |
| bwd_simt_varlen | BwdTileSimt&lt;16, 16, 256, 8> | 8 | sm_61 | 128 | 0 | 0/0 | 36096 (35.2) | 2 | 16/64 | by regs=2, by smem=2 |
| bwd_simt_varlen | BwdTileSimt&lt;16, 16, 256, 8> | 8 | sm_70 | 126 | 0 | 0/0 | 36096 (35.2) | 2 | 16/64 | by regs=2, by smem=2 |
| bwd_simt_varlen | BwdTileSimt&lt;16, 16, 256, 8> | 8 | sm_75 | 108 | 0 | 0/0 | 36096 (35.2) | 1 | 8/32 | by regs=2, by smem=1 |
| bwd_simt_varlen | BwdTileSimt&lt;32, 16, 96, 4> | 4 | sm_60 | 124 | 0 | 0/0 | 24576 (24.0) | 2 | 8/64 | by regs=4, by smem=2 |
| bwd_simt_varlen | BwdTileSimt&lt;32, 16, 96, 4> | 4 | sm_61 | 128 | 0 | 0/0 | 24576 (24.0) | 4 | 16/64 | by regs=4, by smem=4 |
| bwd_simt_varlen | BwdTileSimt&lt;32, 16, 96, 4> | 4 | sm_70 | 128 | 0 | 0/0 | 24576 (24.0) | 4 | 16/64 | by regs=4, by smem=4 |
| bwd_simt_varlen | BwdTileSimt&lt;32, 16, 96, 4> | 4 | sm_75 | 96 | 0 | 0/0 | 24576 (24.0) | 2 | 8/32 | by regs=5, by smem=2 |
| bwd_simt_varlen | BwdTileSimt&lt;32, 32, 32, 4> | 4 | sm_60 | 76 | 0 | 0/0 | 18944 (18.5) | 3 | 12/64 | by regs=6, by smem=3 |
| bwd_simt_varlen | BwdTileSimt&lt;32, 32, 32, 4> | 4 | sm_61 | 80 | 0 | 0/0 | 18944 (18.5) | 5 | 20/64 | by regs=6, by smem=5 |
| bwd_simt_varlen | BwdTileSimt&lt;32, 32, 32, 4> | 4 | sm_70 | 80 | 8 | 8/8 | 18944 (18.5) | 5 | 20/64 | by regs=6, by smem=5 |
| bwd_simt_varlen | BwdTileSimt&lt;32, 32, 32, 4> | 4 | sm_75 | 72 | 0 | 0/0 | 18944 (18.5) | 3 | 12/32 | by regs=7, by smem=3 |
| bwd_simt_varlen | BwdTileSimt&lt;32, 32, 64, 4> | 4 | sm_60 | 111 | 0 | 0/0 | 27136 (26.5) | 2 | 8/64 | by regs=4, by smem=2 |
| bwd_simt_varlen | BwdTileSimt&lt;32, 32, 64, 4> | 4 | sm_61 | 126 | 0 | 0/0 | 27136 (26.5) | 3 | 12/64 | by regs=4, by smem=3 |
| bwd_simt_varlen | BwdTileSimt&lt;32, 32, 64, 4> | 4 | sm_70 | 126 | 0 | 0/0 | 27136 (26.5) | 3 | 12/64 | by regs=4, by smem=3 |
| bwd_simt_varlen | BwdTileSimt&lt;32, 32, 64, 4> | 4 | sm_75 | 96 | 0 | 0/0 | 27136 (26.5) | 2 | 8/32 | by regs=5, by smem=2 |
| bwd_wmma | BwdTileWmma&lt;32, 16, 256, 4> | 4 | sm_70 | 244 | 0 | 0/0 | 57344 (56.0) | 1 | 4/64 | by regs=2, by smem=1 |
| bwd_wmma | BwdTileWmma&lt;32, 16, 256, 4> | 4 | sm_75 | 222 | 0 | 0/0 | 57344 (56.0) | 1 | 4/32 | by regs=2, by smem=1 |
| bwd_wmma | BwdTileWmma&lt;32, 32, 96, 2> | 4 | sm_70 | 216 | 0 | 0/0 | 34560 (33.8) | 2 | 8/64 | by regs=2, by smem=2 |
| bwd_wmma | BwdTileWmma&lt;32, 32, 96, 2> | 4 | sm_75 | 178 | 0 | 0/0 | 34560 (33.8) | 1 | 4/32 | by regs=2, by smem=1 |
| bwd_wmma | BwdTileWmma&lt;32, 32, 128, 2> | 4 | sm_70 | 244 | 0 | 0/0 | 42752 (41.8) | 2 | 8/64 | by regs=2, by smem=2 |
| bwd_wmma | BwdTileWmma&lt;32, 32, 128, 2> | 4 | sm_75 | 238 | 0 | 0/0 | 42752 (41.8) | 1 | 4/32 | by regs=2, by smem=1 |
| bwd_wmma | BwdTileWmma&lt;32, 32, 256, 4> | 8 | sm_70 | 244 | 0 | 0/0 | 80640 (78.8) | 1 | 8/64 | by regs=1, by smem=1 |
| bwd_wmma | BwdTileWmma&lt;32, 32, 256, 4> | 8 | sm_75 | 236 | 0 | 0/0 | 80640 (78.8) | - | - | FAIL(smem>64KB) |
| bwd_wmma | BwdTileWmma&lt;32, 64, 80, 1> | 4 | sm_70 | 245 | 0 | 0/0 | 44288 (43.2) | 2 | 8/64 | by regs=2, by smem=2 |
| bwd_wmma | BwdTileWmma&lt;32, 64, 80, 1> | 4 | sm_75 | 247 | 0 | 0/0 | 44288 (43.2) | 1 | 4/32 | by regs=2, by smem=1 |
| bwd_wmma | BwdTileWmma&lt;32, 64, 96, 2> | 8 | sm_70 | 213 | 0 | 0/0 | 55552 (54.2) | 1 | 8/64 | by regs=1, by smem=1 |
| bwd_wmma | BwdTileWmma&lt;32, 64, 96, 2> | 8 | sm_75 | 202 | 0 | 0/0 | 55552 (54.2) | 1 | 8/32 | by regs=1, by smem=1 |
| bwd_wmma | BwdTileWmma&lt;32, 64, 128, 2> | 8 | sm_70 | 240 | 0 | 0/0 | 67840 (66.2) | 1 | 8/64 | by regs=1, by smem=1 |
| bwd_wmma | BwdTileWmma&lt;32, 64, 128, 2> | 8 | sm_75 | 228 | 0 | 0/0 | 67840 (66.2) | - | - | FAIL(smem>64KB) |
| bwd_wmma | BwdTileWmma&lt;64, 64, 32, 1> | 4 | sm_70 | 240 | 0 | 0/0 | 39424 (38.5) | 2 | 8/64 | by regs=2, by smem=2 |
| bwd_wmma | BwdTileWmma&lt;64, 64, 32, 1> | 4 | sm_75 | 245 | 0 | 0/0 | 39424 (38.5) | 1 | 4/32 | by regs=2, by smem=1 |
| bwd_wmma | BwdTileWmma&lt;64, 64, 64, 1> | 4 | sm_70 | 255 | 16 | 16/16 | 55808 (54.5) | 1 | 4/64 | by regs=2, by smem=1 |
| bwd_wmma | BwdTileWmma&lt;64, 64, 64, 1> | 4 | sm_75 | 255 | 0 | 0/0 | 55808 (54.5) | 1 | 4/32 | by regs=2, by smem=1 |
| bwd_wmma_varlen | BwdTileWmma&lt;32, 16, 256, 4> | 4 | sm_70 | 243 | 0 | 0/0 | 57344 (56.0) | 1 | 4/64 | by regs=2, by smem=1 |
| bwd_wmma_varlen | BwdTileWmma&lt;32, 16, 256, 4> | 4 | sm_75 | 234 | 0 | 0/0 | 57344 (56.0) | 1 | 4/32 | by regs=2, by smem=1 |
| bwd_wmma_varlen | BwdTileWmma&lt;32, 32, 96, 2> | 4 | sm_70 | 213 | 0 | 0/0 | 34560 (33.8) | 2 | 8/64 | by regs=2, by smem=2 |
| bwd_wmma_varlen | BwdTileWmma&lt;32, 32, 96, 2> | 4 | sm_75 | 202 | 0 | 0/0 | 34560 (33.8) | 1 | 4/32 | by regs=2, by smem=1 |
| bwd_wmma_varlen | BwdTileWmma&lt;32, 32, 128, 2> | 4 | sm_70 | 239 | 0 | 0/0 | 42752 (41.8) | 2 | 8/64 | by regs=2, by smem=2 |
| bwd_wmma_varlen | BwdTileWmma&lt;32, 32, 128, 2> | 4 | sm_75 | 228 | 0 | 0/0 | 42752 (41.8) | 1 | 4/32 | by regs=2, by smem=1 |
| bwd_wmma_varlen | BwdTileWmma&lt;32, 32, 256, 4> | 8 | sm_70 | 237 | 0 | 0/0 | 80640 (78.8) | 1 | 8/64 | by regs=1, by smem=1 |
| bwd_wmma_varlen | BwdTileWmma&lt;32, 32, 256, 4> | 8 | sm_75 | 226 | 0 | 0/0 | 80640 (78.8) | - | - | FAIL(smem>64KB) |
| bwd_wmma_varlen | BwdTileWmma&lt;32, 64, 80, 1> | 4 | sm_70 | 241 | 0 | 0/0 | 44288 (43.2) | 2 | 8/64 | by regs=2, by smem=2 |
| bwd_wmma_varlen | BwdTileWmma&lt;32, 64, 80, 1> | 4 | sm_75 | 242 | 0 | 0/0 | 44288 (43.2) | 1 | 4/32 | by regs=2, by smem=1 |
| bwd_wmma_varlen | BwdTileWmma&lt;32, 64, 96, 2> | 8 | sm_70 | 210 | 0 | 0/0 | 55552 (54.2) | 1 | 8/64 | by regs=1, by smem=1 |
| bwd_wmma_varlen | BwdTileWmma&lt;32, 64, 96, 2> | 8 | sm_75 | 200 | 0 | 0/0 | 55552 (54.2) | 1 | 8/32 | by regs=1, by smem=1 |
| bwd_wmma_varlen | BwdTileWmma&lt;32, 64, 128, 2> | 8 | sm_70 | 231 | 0 | 0/0 | 67840 (66.2) | 1 | 8/64 | by regs=1, by smem=1 |
| bwd_wmma_varlen | BwdTileWmma&lt;32, 64, 128, 2> | 8 | sm_75 | 220 | 0 | 0/0 | 67840 (66.2) | - | - | FAIL(smem>64KB) |
| bwd_wmma_varlen | BwdTileWmma&lt;64, 64, 32, 1> | 4 | sm_70 | 237 | 0 | 0/0 | 39424 (38.5) | 2 | 8/64 | by regs=2, by smem=2 |
| bwd_wmma_varlen | BwdTileWmma&lt;64, 64, 32, 1> | 4 | sm_75 | 245 | 0 | 0/0 | 39424 (38.5) | 1 | 4/32 | by regs=2, by smem=1 |
| bwd_wmma_varlen | BwdTileWmma&lt;64, 64, 64, 1> | 4 | sm_70 | 255 | 16 | 12/12 | 55808 (54.5) | 1 | 4/64 | by regs=2, by smem=1 |
| bwd_wmma_varlen | BwdTileWmma&lt;64, 64, 64, 1> | 4 | sm_75 | 255 | 40 | 40/56 | 55808 (54.5) | 1 | 4/32 | by regs=2, by smem=1 |
| fwd_pascal | FwdTileSimt&lt;16, 8, 256, 8> | 8 | sm_60 | 121 | 0 | 0/0 | 42496 (41.5) | 1 | 8/64 | by regs=2, by smem=1 |
| fwd_pascal | FwdTileSimt&lt;16, 8, 256, 8> | 8 | sm_61 | 101 | 0 | 0/0 | 42496 (41.5) | 2 | 16/64 | by regs=2, by smem=2 |
| fwd_pascal | FwdTileSimt&lt;16, 8, 256, 8> | 8 | sm_70 | 101 | 0 | 0/0 | 42496 (41.5) | 2 | 16/64 | by regs=2, by smem=2 |
| fwd_pascal | FwdTileSimt&lt;16, 8, 256, 8> | 8 | sm_75 | 101 | 0 | 0/0 | 42496 (41.5) | 1 | 8/32 | by regs=2, by smem=1 |
| fwd_pascal | FwdTileSimt&lt;32, 8, 128, 1> | 2 | sm_60 | 173 | 0 | 0/0 | 26624 (26.0) | 2 | 4/64 | by regs=5, by smem=2 |
| fwd_pascal | FwdTileSimt&lt;32, 8, 128, 1> | 2 | sm_61 | 173 | 0 | 0/0 | 26624 (26.0) | 3 | 6/64 | by regs=4, by smem=3 |
| fwd_pascal | FwdTileSimt&lt;32, 8, 128, 1> | 2 | sm_70 | 176 | 0 | 0/0 | 26624 (26.0) | 3 | 6/64 | by regs=4, by smem=3 |
| fwd_pascal | FwdTileSimt&lt;32, 8, 128, 1> | 2 | sm_75 | 176 | 0 | 0/0 | 26624 (26.0) | 2 | 4/32 | by regs=4, by smem=2 |
| fwd_pascal | FwdTileSimt&lt;64, 16, 64, 1> | 4 | sm_60 | 128 | 0 | 0/0 | 31232 (30.5) | 2 | 8/64 | by regs=4, by smem=2 |
| fwd_pascal | FwdTileSimt&lt;64, 16, 64, 1> | 4 | sm_61 | 128 | 0 | 0/0 | 31232 (30.5) | 3 | 12/64 | by regs=4, by smem=3 |
| fwd_pascal | FwdTileSimt&lt;64, 16, 64, 1> | 4 | sm_70 | 136 | 0 | 0/0 | 31232 (30.5) | 3 | 12/64 | by regs=3, by smem=3 |
| fwd_pascal | FwdTileSimt&lt;64, 16, 64, 1> | 4 | sm_75 | 136 | 0 | 0/0 | 31232 (30.5) | 2 | 8/32 | by regs=3, by smem=2 |
| fwd_pascal | FwdTileSimt&lt;64, 16, 96, 1> | 4 | sm_60 | 167 | 0 | 0/0 | 43520 (42.5) | 1 | 4/64 | by regs=3, by smem=1 |
| fwd_pascal | FwdTileSimt&lt;64, 16, 96, 1> | 4 | sm_61 | 167 | 0 | 0/0 | 43520 (42.5) | 2 | 8/64 | by regs=3, by smem=2 |
| fwd_pascal | FwdTileSimt&lt;64, 16, 96, 1> | 4 | sm_70 | 168 | 8 | 8/8 | 43520 (42.5) | 2 | 8/64 | by regs=3, by smem=2 |
| fwd_pascal | FwdTileSimt&lt;64, 16, 96, 1> | 4 | sm_75 | 168 | 8 | 8/8 | 43520 (42.5) | 1 | 4/32 | by regs=3, by smem=1 |
| fwd_pascal | FwdTileSimt&lt;64, 32, 32, 1> | 4 | sm_60 | 119 | 0 | 0/0 | 28672 (28.0) | 2 | 8/64 | by regs=4, by smem=2 |
| fwd_pascal | FwdTileSimt&lt;64, 32, 32, 1> | 4 | sm_61 | 128 | 0 | 0/0 | 28672 (28.0) | 3 | 12/64 | by regs=4, by smem=3 |
| fwd_pascal | FwdTileSimt&lt;64, 32, 32, 1> | 4 | sm_70 | 146 | 0 | 0/0 | 28672 (28.0) | 3 | 12/64 | by regs=3, by smem=3 |
| fwd_pascal | FwdTileSimt&lt;64, 32, 32, 1> | 4 | sm_75 | 136 | 0 | 0/0 | 28672 (28.0) | 2 | 8/32 | by regs=3, by smem=2 |
| fwd_pascal_varlen | FwdTileSimt&lt;16, 8, 256, 8> | 8 | sm_60 | 92 | 0 | 0/0 | 42496 (41.5) | 1 | 8/64 | by regs=2, by smem=1 |
| fwd_pascal_varlen | FwdTileSimt&lt;16, 8, 256, 8> | 8 | sm_61 | 102 | 0 | 0/0 | 42496 (41.5) | 2 | 16/64 | by regs=2, by smem=2 |
| fwd_pascal_varlen | FwdTileSimt&lt;16, 8, 256, 8> | 8 | sm_70 | 105 | 0 | 0/0 | 42496 (41.5) | 2 | 16/64 | by regs=2, by smem=2 |
| fwd_pascal_varlen | FwdTileSimt&lt;16, 8, 256, 8> | 8 | sm_75 | 105 | 0 | 0/0 | 42496 (41.5) | 1 | 8/32 | by regs=2, by smem=1 |
| fwd_pascal_varlen | FwdTileSimt&lt;32, 8, 128, 1> | 2 | sm_60 | 167 | 0 | 0/0 | 26624 (26.0) | 2 | 4/64 | by regs=6, by smem=2 |
| fwd_pascal_varlen | FwdTileSimt&lt;32, 8, 128, 1> | 2 | sm_61 | 167 | 0 | 0/0 | 26624 (26.0) | 3 | 6/64 | by regs=6, by smem=3 |
| fwd_pascal_varlen | FwdTileSimt&lt;32, 8, 128, 1> | 2 | sm_70 | 172 | 0 | 0/0 | 26624 (26.0) | 3 | 6/64 | by regs=4, by smem=3 |
| fwd_pascal_varlen | FwdTileSimt&lt;32, 8, 128, 1> | 2 | sm_75 | 172 | 0 | 0/0 | 26624 (26.0) | 2 | 4/32 | by regs=4, by smem=2 |
| fwd_pascal_varlen | FwdTileSimt&lt;64, 16, 64, 1> | 4 | sm_60 | 128 | 0 | 0/0 | 31232 (30.5) | 2 | 8/64 | by regs=4, by smem=2 |
| fwd_pascal_varlen | FwdTileSimt&lt;64, 16, 64, 1> | 4 | sm_61 | 128 | 0 | 0/0 | 31232 (30.5) | 3 | 12/64 | by regs=4, by smem=3 |
| fwd_pascal_varlen | FwdTileSimt&lt;64, 16, 64, 1> | 4 | sm_70 | 134 | 0 | 0/0 | 31232 (30.5) | 3 | 12/64 | by regs=3, by smem=3 |
| fwd_pascal_varlen | FwdTileSimt&lt;64, 16, 64, 1> | 4 | sm_75 | 134 | 0 | 0/0 | 31232 (30.5) | 2 | 8/32 | by regs=3, by smem=2 |
| fwd_pascal_varlen | FwdTileSimt&lt;64, 16, 96, 1> | 4 | sm_60 | 160 | 0 | 0/0 | 43520 (42.5) | 1 | 4/64 | by regs=3, by smem=1 |
| fwd_pascal_varlen | FwdTileSimt&lt;64, 16, 96, 1> | 4 | sm_61 | 161 | 0 | 0/0 | 43520 (42.5) | 2 | 8/64 | by regs=3, by smem=2 |
| fwd_pascal_varlen | FwdTileSimt&lt;64, 16, 96, 1> | 4 | sm_70 | 168 | 0 | 0/0 | 43520 (42.5) | 2 | 8/64 | by regs=3, by smem=2 |
| fwd_pascal_varlen | FwdTileSimt&lt;64, 16, 96, 1> | 4 | sm_75 | 168 | 0 | 0/0 | 43520 (42.5) | 1 | 4/32 | by regs=3, by smem=1 |
| fwd_pascal_varlen | FwdTileSimt&lt;64, 32, 32, 1> | 4 | sm_60 | 124 | 0 | 0/0 | 28672 (28.0) | 2 | 8/64 | by regs=4, by smem=2 |
| fwd_pascal_varlen | FwdTileSimt&lt;64, 32, 32, 1> | 4 | sm_61 | 124 | 0 | 0/0 | 28672 (28.0) | 3 | 12/64 | by regs=4, by smem=3 |
| fwd_pascal_varlen | FwdTileSimt&lt;64, 32, 32, 1> | 4 | sm_70 | 142 | 0 | 0/0 | 28672 (28.0) | 3 | 12/64 | by regs=3, by smem=3 |
| fwd_pascal_varlen | FwdTileSimt&lt;64, 32, 32, 1> | 4 | sm_75 | 142 | 0 | 0/0 | 28672 (28.0) | 2 | 8/32 | by regs=3, by smem=2 |
| fwd_volta | FwdTileWmma&lt;64, 16, 256> | 4 | sm_70 | 255 | 24 | 26/24 | 57088 (55.8) | 1 | 4/64 | by regs=2, by smem=1 |
| fwd_volta | FwdTileWmma&lt;64, 16, 256> | 4 | sm_75 | 255 | 88 | 98/220 | 57088 (55.8) | 1 | 4/32 | by regs=2, by smem=1 |
| fwd_volta | FwdTileWmma&lt;128, 16, 80> | 8 | sm_70 | 128 | 8 | 2/4 | 40960 (40.0) | 2 | 16/64 | by regs=2, by smem=2 |
| fwd_volta | FwdTileWmma&lt;128, 16, 80> | 8 | sm_75 | 122 | 0 | 0/0 | 40960 (40.0) | 1 | 8/32 | by regs=2, by smem=1 |
| fwd_volta | FwdTileWmma&lt;128, 16, 96> | 8 | sm_70 | 128 | 0 | 0/0 | 46080 (45.0) | 2 | 16/64 | by regs=2, by smem=2 |
| fwd_volta | FwdTileWmma&lt;128, 16, 96> | 8 | sm_75 | 135 | 0 | 0/0 | 46080 (45.0) | 1 | 8/32 | by regs=1, by smem=1 |
| fwd_volta | FwdTileWmma&lt;128, 16, 128> | 8 | sm_70 | 163 | 0 | 0/0 | 56320 (55.0) | 1 | 8/64 | by regs=1, by smem=1 |
| fwd_volta | FwdTileWmma&lt;128, 16, 128> | 8 | sm_75 | 152 | 0 | 0/0 | 56320 (55.0) | 1 | 8/32 | by regs=1, by smem=1 |
| fwd_volta | FwdTileWmma&lt;128, 16, 256> | 8 | sm_70 | 252 | 0 | 0/0 | 97280 (95.0) | 1 | 8/64 | by regs=1, by smem=1 |
| fwd_volta | FwdTileWmma&lt;128, 16, 256> | 8 | sm_75 | 249 | 0 | 0/0 | 97280 (95.0) | - | - | FAIL(smem>64KB) |
| fwd_volta | FwdTileWmma&lt;128, 32, 32> | 8 | sm_70 | 125 | 0 | 0/0 | 36352 (35.5) | 2 | 16/64 | by regs=2, by smem=2 |
| fwd_volta | FwdTileWmma&lt;128, 32, 32> | 8 | sm_75 | 105 | 0 | 0/0 | 36352 (35.5) | 1 | 8/32 | by regs=2, by smem=1 |
| fwd_volta | FwdTileWmma&lt;128, 32, 64> | 8 | sm_70 | 128 | 0 | 0/0 | 48640 (47.5) | 2 | 16/64 | by regs=2, by smem=2 |
| fwd_volta | FwdTileWmma&lt;128, 32, 64> | 8 | sm_75 | 124 | 0 | 0/0 | 48640 (47.5) | 1 | 8/32 | by regs=2, by smem=1 |
| fwd_volta | FwdTileWmma&lt;128, 32, 128> | 8 | sm_70 | 172 | 0 | 0/0 | 73216 (71.5) | 1 | 8/64 | by regs=1, by smem=1 |
| fwd_volta | FwdTileWmma&lt;128, 32, 128> | 8 | sm_75 | 184 | 0 | 0/0 | 73216 (71.5) | - | - | FAIL(smem>64KB) |
| fwd_volta_varlen | FwdTileWmma&lt;64, 16, 256> | 4 | sm_70 | 255 | 0 | 0/0 | 57088 (55.8) | 1 | 4/64 | by regs=2, by smem=1 |
| fwd_volta_varlen | FwdTileWmma&lt;64, 16, 256> | 4 | sm_75 | 255 | 88 | 98/220 | 57088 (55.8) | 1 | 4/32 | by regs=2, by smem=1 |
| fwd_volta_varlen | FwdTileWmma&lt;128, 16, 80> | 8 | sm_70 | 128 | 0 | 0/0 | 40960 (40.0) | 2 | 16/64 | by regs=2, by smem=2 |
| fwd_volta_varlen | FwdTileWmma&lt;128, 16, 80> | 8 | sm_75 | 119 | 0 | 0/0 | 40960 (40.0) | 1 | 8/32 | by regs=2, by smem=1 |
| fwd_volta_varlen | FwdTileWmma&lt;128, 16, 96> | 8 | sm_70 | 128 | 0 | 0/0 | 46080 (45.0) | 2 | 16/64 | by regs=2, by smem=2 |
| fwd_volta_varlen | FwdTileWmma&lt;128, 16, 96> | 8 | sm_75 | 128 | 0 | 0/0 | 46080 (45.0) | 1 | 8/32 | by regs=2, by smem=1 |
| fwd_volta_varlen | FwdTileWmma&lt;128, 16, 128> | 8 | sm_70 | 164 | 0 | 0/0 | 56320 (55.0) | 1 | 8/64 | by regs=1, by smem=1 |
| fwd_volta_varlen | FwdTileWmma&lt;128, 16, 128> | 8 | sm_75 | 146 | 0 | 0/0 | 56320 (55.0) | 1 | 8/32 | by regs=1, by smem=1 |
| fwd_volta_varlen | FwdTileWmma&lt;128, 16, 256> | 8 | sm_70 | 249 | 0 | 0/0 | 97280 (95.0) | 1 | 8/64 | by regs=1, by smem=1 |
| fwd_volta_varlen | FwdTileWmma&lt;128, 16, 256> | 8 | sm_75 | 242 | 0 | 0/0 | 97280 (95.0) | - | - | FAIL(smem>64KB) |
| fwd_volta_varlen | FwdTileWmma&lt;128, 32, 32> | 8 | sm_70 | 128 | 0 | 0/0 | 36352 (35.5) | 2 | 16/64 | by regs=2, by smem=2 |
| fwd_volta_varlen | FwdTileWmma&lt;128, 32, 32> | 8 | sm_75 | 121 | 0 | 0/0 | 36352 (35.5) | 1 | 8/32 | by regs=2, by smem=1 |
| fwd_volta_varlen | FwdTileWmma&lt;128, 32, 64> | 8 | sm_70 | 128 | 0 | 0/0 | 48640 (47.5) | 2 | 16/64 | by regs=2, by smem=2 |
| fwd_volta_varlen | FwdTileWmma&lt;128, 32, 64> | 8 | sm_75 | 125 | 0 | 0/0 | 48640 (47.5) | 1 | 8/32 | by regs=2, by smem=1 |
| fwd_volta_varlen | FwdTileWmma&lt;128, 32, 128> | 8 | sm_70 | 170 | 0 | 0/0 | 73216 (71.5) | 1 | 8/64 | by regs=1, by smem=1 |
| fwd_volta_varlen | FwdTileWmma&lt;128, 32, 128> | 8 | sm_75 | 184 | 0 | 0/0 | 73216 (71.5) | - | - | FAIL(smem>64KB) |
| precompute_D | - | 8 | sm_60 | 32 | 0 | 0/0 | 0 (0.0) | 8 | 64/64 | by regs=8, by smem=inf |
| precompute_D | - | 8 | sm_61 | 32 | 0 | 0/0 | 0 (0.0) | 8 | 64/64 | by regs=8, by smem=inf |
| precompute_D | - | 8 | sm_70 | 40 | 0 | 0/0 | 0 (0.0) | 6 | 48/64 | by regs=6, by smem=inf |
| precompute_D | - | 8 | sm_75 | 63 | 0 | 0/0 | 0 (0.0) | 4 | 32/32 | by regs=4, by smem=inf |
