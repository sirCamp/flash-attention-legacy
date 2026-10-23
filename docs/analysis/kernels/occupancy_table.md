| kernel | tile | warps/block | arch | regs | stack B | spill st/ld B | dyn smem B (KB) | blocks/SM | warps/SM (max) | blocks/SM limits |
|---|---|---|---|---|---|---|---|---|---|---|
| bwd_mma | BwdTileMma&lt;32, 16, 256, 4> | 4 | sm_75 | 178 | 0 | 0/0 | 53504 (52.2) | 1 | 4/32 | by regs=2, by smem=1 |
| bwd_mma | BwdTileMma&lt;32, 64, 80, 1> | 4 | sm_75 | 189 | 0 | 0/0 | 39168 (38.2) | 1 | 4/32 | by regs=2, by smem=1 |
| bwd_mma | BwdTileMma&lt;32, 64, 96, 2> | 8 | sm_75 | 130 | 0 | 0/0 | 50432 (49.2) | 1 | 8/32 | by regs=1, by smem=1 |
| bwd_mma | BwdTileMma&lt;32, 64, 128, 2> | 8 | sm_75 | 174 | 0 | 0/0 | 62720 (61.2) | 1 | 8/32 | by regs=1, by smem=1 |
| bwd_mma | BwdTileMma&lt;64, 64, 32, 1> | 4 | sm_75 | 168 | 16 | 16/16 | 30208 (29.5) | 2 | 8/32 | by regs=3, by smem=2 |
| bwd_mma | BwdTileMma&lt;64, 64, 64, 2> | 8 | sm_75 | 128 | 8 | 4/4 | 55808 (54.5) | 1 | 8/32 | by regs=2, by smem=1 |
| bwd_mma_varlen | BwdTileMma&lt;32, 16, 256, 4> | 4 | sm_75 | 180 | 0 | 0/0 | 53504 (52.2) | 1 | 4/32 | by regs=2, by smem=1 |
| bwd_mma_varlen | BwdTileMma&lt;32, 64, 80, 1> | 4 | sm_75 | 228 | 0 | 0/0 | 39168 (38.2) | 1 | 4/32 | by regs=2, by smem=1 |
| bwd_mma_varlen | BwdTileMma&lt;32, 64, 96, 2> | 8 | sm_75 | 130 | 0 | 0/0 | 50432 (49.2) | 1 | 8/32 | by regs=1, by smem=1 |
| bwd_mma_varlen | BwdTileMma&lt;32, 64, 128, 2> | 8 | sm_75 | 182 | 0 | 0/0 | 62720 (61.2) | 1 | 8/32 | by regs=1, by smem=1 |
| bwd_mma_varlen | BwdTileMma&lt;64, 64, 32, 1> | 4 | sm_75 | 184 | 0 | 0/0 | 30208 (29.5) | 2 | 8/32 | by regs=2, by smem=2 |
| bwd_mma_varlen | BwdTileMma&lt;64, 64, 64, 2> | 8 | sm_75 | 170 | 0 | 0/0 | 55808 (54.5) | 1 | 8/32 | by regs=1, by smem=1 |
| bwd_simt | BwdTileSimt&lt;16, 16, 128, 4> | 4 | sm_60 | 113 | 0 | 0/0 | 19712 (19.2) | 3 | 12/64 | by regs=4, by smem=3 |
| bwd_simt | BwdTileSimt&lt;16, 16, 128, 4> | 4 | sm_61 | 126 | 0 | 0/0 | 19712 (19.2) | 4 | 16/64 | by regs=4, by smem=4 |
| bwd_simt | BwdTileSimt&lt;16, 16, 128, 4> | 4 | sm_70 | 126 | 0 | 0/0 | 19712 (19.2) | 4 | 16/64 | by regs=4, by smem=4 |
| bwd_simt | BwdTileSimt&lt;16, 16, 128, 4> | 4 | sm_75 | 80 | 0 | 0/0 | 19712 (19.2) | 3 | 12/32 | by regs=6, by smem=3 |
| bwd_simt | BwdTileSimt&lt;16, 16, 256, 8> | 8 | sm_60 | 111 | 0 | 0/0 | 36096 (35.2) | 1 | 8/64 | by regs=2, by smem=1 |
| bwd_simt | BwdTileSimt&lt;16, 16, 256, 8> | 8 | sm_61 | 127 | 0 | 0/0 | 36096 (35.2) | 2 | 16/64 | by regs=2, by smem=2 |
| bwd_simt | BwdTileSimt&lt;16, 16, 256, 8> | 8 | sm_70 | 128 | 0 | 0/0 | 36096 (35.2) | 2 | 16/64 | by regs=2, by smem=2 |
| bwd_simt | BwdTileSimt&lt;16, 16, 256, 8> | 8 | sm_75 | 80 | 16 | 12/12 | 36096 (35.2) | 1 | 8/32 | by regs=3, by smem=1 |
| bwd_simt | BwdTileSimt&lt;32, 16, 96, 4> | 4 | sm_60 | 128 | 0 | 0/0 | 24576 (24.0) | 2 | 8/64 | by regs=4, by smem=2 |
| bwd_simt | BwdTileSimt&lt;32, 16, 96, 4> | 4 | sm_61 | 128 | 0 | 0/0 | 24576 (24.0) | 4 | 16/64 | by regs=4, by smem=4 |
| bwd_simt | BwdTileSimt&lt;32, 16, 96, 4> | 4 | sm_70 | 128 | 0 | 0/0 | 24576 (24.0) | 4 | 16/64 | by regs=4, by smem=4 |
| bwd_simt | BwdTileSimt&lt;32, 16, 96, 4> | 4 | sm_75 | 96 | 0 | 0/0 | 24576 (24.0) | 2 | 8/32 | by regs=5, by smem=2 |
| bwd_simt | BwdTileSimt&lt;32, 32, 32, 4> | 4 | sm_60 | 75 | 0 | 0/0 | 18944 (18.5) | 3 | 12/64 | by regs=6, by smem=3 |
| bwd_simt | BwdTileSimt&lt;32, 32, 32, 4> | 4 | sm_61 | 79 | 0 | 0/0 | 18944 (18.5) | 5 | 20/64 | by regs=6, by smem=5 |
| bwd_simt | BwdTileSimt&lt;32, 32, 32, 4> | 4 | sm_70 | 80 | 8 | 8/8 | 18944 (18.5) | 5 | 20/64 | by regs=6, by smem=5 |
| bwd_simt | BwdTileSimt&lt;32, 32, 32, 4> | 4 | sm_75 | 72 | 16 | 12/12 | 18944 (18.5) | 3 | 12/32 | by regs=7, by smem=3 |
| bwd_simt | BwdTileSimt&lt;32, 32, 64, 4> | 4 | sm_60 | 111 | 0 | 0/0 | 27136 (26.5) | 2 | 8/64 | by regs=4, by smem=2 |
| bwd_simt | BwdTileSimt&lt;32, 32, 64, 4> | 4 | sm_61 | 127 | 0 | 0/0 | 27136 (26.5) | 3 | 12/64 | by regs=4, by smem=3 |
| bwd_simt | BwdTileSimt&lt;32, 32, 64, 4> | 4 | sm_70 | 128 | 0 | 0/0 | 27136 (26.5) | 3 | 12/64 | by regs=4, by smem=3 |
| bwd_simt | BwdTileSimt&lt;32, 32, 64, 4> | 4 | sm_75 | 96 | 0 | 0/0 | 27136 (26.5) | 2 | 8/32 | by regs=5, by smem=2 |
| bwd_simt_varlen | BwdTileSimt&lt;16, 16, 128, 4> | 4 | sm_60 | 96 | 8 | 8/8 | 19712 (19.2) | 3 | 12/64 | by regs=5, by smem=3 |
| bwd_simt_varlen | BwdTileSimt&lt;16, 16, 128, 4> | 4 | sm_61 | 103 | 0 | 0/0 | 19712 (19.2) | 4 | 16/64 | by regs=4, by smem=4 |
| bwd_simt_varlen | BwdTileSimt&lt;16, 16, 128, 4> | 4 | sm_70 | 126 | 0 | 0/0 | 19712 (19.2) | 4 | 16/64 | by regs=4, by smem=4 |
| bwd_simt_varlen | BwdTileSimt&lt;16, 16, 128, 4> | 4 | sm_75 | 80 | 0 | 0/0 | 19712 (19.2) | 3 | 12/32 | by regs=6, by smem=3 |
| bwd_simt_varlen | BwdTileSimt&lt;16, 16, 256, 8> | 8 | sm_60 | 111 | 0 | 0/0 | 36096 (35.2) | 1 | 8/64 | by regs=2, by smem=1 |
| bwd_simt_varlen | BwdTileSimt&lt;16, 16, 256, 8> | 8 | sm_61 | 125 | 0 | 0/0 | 36096 (35.2) | 2 | 16/64 | by regs=2, by smem=2 |
| bwd_simt_varlen | BwdTileSimt&lt;16, 16, 256, 8> | 8 | sm_70 | 128 | 0 | 0/0 | 36096 (35.2) | 2 | 16/64 | by regs=2, by smem=2 |
| bwd_simt_varlen | BwdTileSimt&lt;16, 16, 256, 8> | 8 | sm_75 | 128 | 0 | 0/0 | 36096 (35.2) | 1 | 8/32 | by regs=2, by smem=1 |
| bwd_simt_varlen | BwdTileSimt&lt;32, 16, 96, 4> | 4 | sm_60 | 118 | 0 | 0/0 | 24576 (24.0) | 2 | 8/64 | by regs=4, by smem=2 |
| bwd_simt_varlen | BwdTileSimt&lt;32, 16, 96, 4> | 4 | sm_61 | 128 | 0 | 0/0 | 24576 (24.0) | 4 | 16/64 | by regs=4, by smem=4 |
| bwd_simt_varlen | BwdTileSimt&lt;32, 16, 96, 4> | 4 | sm_70 | 128 | 0 | 0/0 | 24576 (24.0) | 4 | 16/64 | by regs=4, by smem=4 |
| bwd_simt_varlen | BwdTileSimt&lt;32, 16, 96, 4> | 4 | sm_75 | 128 | 0 | 0/0 | 24576 (24.0) | 2 | 8/32 | by regs=4, by smem=2 |
| bwd_simt_varlen | BwdTileSimt&lt;32, 32, 32, 4> | 4 | sm_60 | 77 | 0 | 0/0 | 18944 (18.5) | 3 | 12/64 | by regs=6, by smem=3 |
| bwd_simt_varlen | BwdTileSimt&lt;32, 32, 32, 4> | 4 | sm_61 | 77 | 0 | 0/0 | 18944 (18.5) | 5 | 20/64 | by regs=6, by smem=5 |
| bwd_simt_varlen | BwdTileSimt&lt;32, 32, 32, 4> | 4 | sm_70 | 80 | 16 | 12/12 | 18944 (18.5) | 5 | 20/64 | by regs=6, by smem=5 |
| bwd_simt_varlen | BwdTileSimt&lt;32, 32, 32, 4> | 4 | sm_75 | 69 | 0 | 0/0 | 18944 (18.5) | 3 | 12/32 | by regs=7, by smem=3 |
| bwd_simt_varlen | BwdTileSimt&lt;32, 32, 64, 4> | 4 | sm_60 | 111 | 0 | 0/0 | 27136 (26.5) | 2 | 8/64 | by regs=4, by smem=2 |
| bwd_simt_varlen | BwdTileSimt&lt;32, 32, 64, 4> | 4 | sm_61 | 125 | 0 | 0/0 | 27136 (26.5) | 3 | 12/64 | by regs=4, by smem=3 |
| bwd_simt_varlen | BwdTileSimt&lt;32, 32, 64, 4> | 4 | sm_70 | 128 | 0 | 0/0 | 27136 (26.5) | 3 | 12/64 | by regs=4, by smem=3 |
| bwd_simt_varlen | BwdTileSimt&lt;32, 32, 64, 4> | 4 | sm_75 | 96 | 0 | 0/0 | 27136 (26.5) | 2 | 8/32 | by regs=5, by smem=2 |
| bwd_wmma | BwdTileWmma&lt;32, 16, 256, 4> | 4 | sm_70 | 240 | 0 | 0/0 | 57344 (56.0) | 1 | 4/64 | by regs=2, by smem=1 |
| bwd_wmma | BwdTileWmma&lt;32, 16, 256, 4> | 4 | sm_75 | 234 | 0 | 0/0 | 57344 (56.0) | 1 | 4/32 | by regs=2, by smem=1 |
| bwd_wmma | BwdTileWmma&lt;32, 32, 96, 2> | 4 | sm_70 | 198 | 0 | 0/0 | 34560 (33.8) | 2 | 8/64 | by regs=2, by smem=2 |
| bwd_wmma | BwdTileWmma&lt;32, 32, 96, 2> | 4 | sm_75 | 134 | 0 | 0/0 | 34560 (33.8) | 1 | 4/32 | by regs=3, by smem=1 |
| bwd_wmma | BwdTileWmma&lt;32, 32, 128, 2> | 4 | sm_70 | 221 | 0 | 0/0 | 42752 (41.8) | 2 | 8/64 | by regs=2, by smem=2 |
| bwd_wmma | BwdTileWmma&lt;32, 32, 128, 2> | 4 | sm_75 | 206 | 0 | 0/0 | 42752 (41.8) | 1 | 4/32 | by regs=2, by smem=1 |
| bwd_wmma | BwdTileWmma&lt;32, 32, 256, 4> | 8 | sm_70 | 238 | 0 | 0/0 | 80640 (78.8) | 1 | 8/64 | by regs=1, by smem=1 |
| bwd_wmma | BwdTileWmma&lt;32, 32, 256, 4> | 8 | sm_75 | 228 | 0 | 0/0 | 80640 (78.8) | - | - | FAIL(smem>64KB) |
| bwd_wmma | BwdTileWmma&lt;32, 64, 80, 1> | 4 | sm_70 | 245 | 0 | 0/0 | 44288 (43.2) | 2 | 8/64 | by regs=2, by smem=2 |
| bwd_wmma | BwdTileWmma&lt;32, 64, 80, 1> | 4 | sm_75 | 242 | 0 | 0/0 | 44288 (43.2) | 1 | 4/32 | by regs=2, by smem=1 |
| bwd_wmma | BwdTileWmma&lt;32, 64, 96, 2> | 8 | sm_70 | 196 | 0 | 0/0 | 55552 (54.2) | 1 | 8/64 | by regs=1, by smem=1 |
| bwd_wmma | BwdTileWmma&lt;32, 64, 96, 2> | 8 | sm_75 | 166 | 0 | 0/0 | 55552 (54.2) | 1 | 8/32 | by regs=1, by smem=1 |
| bwd_wmma | BwdTileWmma&lt;32, 64, 128, 2> | 8 | sm_70 | 213 | 0 | 0/0 | 67840 (66.2) | 1 | 8/64 | by regs=1, by smem=1 |
| bwd_wmma | BwdTileWmma&lt;32, 64, 128, 2> | 8 | sm_75 | 176 | 0 | 0/0 | 67840 (66.2) | - | - | FAIL(smem>64KB) |
| bwd_wmma | BwdTileWmma&lt;64, 64, 32, 1> | 4 | sm_70 | 239 | 0 | 0/0 | 39424 (38.5) | 2 | 8/64 | by regs=2, by smem=2 |
| bwd_wmma | BwdTileWmma&lt;64, 64, 32, 1> | 4 | sm_75 | 246 | 0 | 0/0 | 39424 (38.5) | 1 | 4/32 | by regs=2, by smem=1 |
| bwd_wmma | BwdTileWmma&lt;64, 64, 64, 2> | 8 | sm_70 | 250 | 0 | 0/0 | 65024 (63.5) | 1 | 8/64 | by regs=1, by smem=1 |
| bwd_wmma | BwdTileWmma&lt;64, 64, 64, 2> | 8 | sm_75 | 245 | 0 | 0/0 | 65024 (63.5) | 1 | 8/32 | by regs=1, by smem=1 |
| bwd_wmma_varlen | BwdTileWmma&lt;32, 16, 256, 4> | 4 | sm_70 | 241 | 0 | 0/0 | 57344 (56.0) | 1 | 4/64 | by regs=2, by smem=1 |
| bwd_wmma_varlen | BwdTileWmma&lt;32, 16, 256, 4> | 4 | sm_75 | 236 | 0 | 0/0 | 57344 (56.0) | 1 | 4/32 | by regs=2, by smem=1 |
| bwd_wmma_varlen | BwdTileWmma&lt;32, 32, 96, 2> | 4 | sm_70 | 199 | 0 | 0/0 | 34560 (33.8) | 2 | 8/64 | by regs=2, by smem=2 |
| bwd_wmma_varlen | BwdTileWmma&lt;32, 32, 96, 2> | 4 | sm_75 | 158 | 0 | 0/0 | 34560 (33.8) | 1 | 4/32 | by regs=3, by smem=1 |
| bwd_wmma_varlen | BwdTileWmma&lt;32, 32, 128, 2> | 4 | sm_70 | 222 | 0 | 0/0 | 42752 (41.8) | 2 | 8/64 | by regs=2, by smem=2 |
| bwd_wmma_varlen | BwdTileWmma&lt;32, 32, 128, 2> | 4 | sm_75 | 206 | 0 | 0/0 | 42752 (41.8) | 1 | 4/32 | by regs=2, by smem=1 |
| bwd_wmma_varlen | BwdTileWmma&lt;32, 32, 256, 4> | 8 | sm_70 | 239 | 0 | 0/0 | 80640 (78.8) | 1 | 8/64 | by regs=1, by smem=1 |
| bwd_wmma_varlen | BwdTileWmma&lt;32, 32, 256, 4> | 8 | sm_75 | 228 | 0 | 0/0 | 80640 (78.8) | - | - | FAIL(smem>64KB) |
| bwd_wmma_varlen | BwdTileWmma&lt;32, 64, 80, 1> | 4 | sm_70 | 242 | 0 | 0/0 | 44288 (43.2) | 2 | 8/64 | by regs=2, by smem=2 |
| bwd_wmma_varlen | BwdTileWmma&lt;32, 64, 80, 1> | 4 | sm_75 | 239 | 0 | 0/0 | 44288 (43.2) | 1 | 4/32 | by regs=2, by smem=1 |
| bwd_wmma_varlen | BwdTileWmma&lt;32, 64, 96, 2> | 8 | sm_70 | 197 | 0 | 0/0 | 55552 (54.2) | 1 | 8/64 | by regs=1, by smem=1 |
| bwd_wmma_varlen | BwdTileWmma&lt;32, 64, 96, 2> | 8 | sm_75 | 169 | 0 | 0/0 | 55552 (54.2) | 1 | 8/32 | by regs=1, by smem=1 |
| bwd_wmma_varlen | BwdTileWmma&lt;32, 64, 128, 2> | 8 | sm_70 | 214 | 0 | 0/0 | 67840 (66.2) | 1 | 8/64 | by regs=1, by smem=1 |
| bwd_wmma_varlen | BwdTileWmma&lt;32, 64, 128, 2> | 8 | sm_75 | 178 | 0 | 0/0 | 67840 (66.2) | - | - | FAIL(smem>64KB) |
| bwd_wmma_varlen | BwdTileWmma&lt;64, 64, 32, 1> | 4 | sm_70 | 242 | 0 | 0/0 | 39424 (38.5) | 2 | 8/64 | by regs=2, by smem=2 |
| bwd_wmma_varlen | BwdTileWmma&lt;64, 64, 32, 1> | 4 | sm_75 | 248 | 0 | 0/0 | 39424 (38.5) | 1 | 4/32 | by regs=2, by smem=1 |
| bwd_wmma_varlen | BwdTileWmma&lt;64, 64, 64, 2> | 8 | sm_70 | 255 | 0 | 0/0 | 65024 (63.5) | 1 | 8/64 | by regs=1, by smem=1 |
| bwd_wmma_varlen | BwdTileWmma&lt;64, 64, 64, 2> | 8 | sm_75 | 246 | 0 | 0/0 | 65024 (63.5) | 1 | 8/32 | by regs=1, by smem=1 |
| decode | d=32, R=1 | 4 | sm_60 | 64 | 0 | 0/0 | 0 (0.0) | 8 | 32/64 | by regs=8, by smem=85 |
| decode | d=32, R=1, paged | 4 | sm_60 | 70 | 0 | 0/0 | 0 (0.0) | 7 | 28/64 | by regs=7, by smem=85 |
| decode | d=32, R=1 | 4 | sm_61 | 64 | 0 | 0/0 | 0 (0.0) | 8 | 32/64 | by regs=8, by smem=128 |
| decode | d=32, R=1, paged | 4 | sm_61 | 69 | 0 | 0/0 | 0 (0.0) | 7 | 28/64 | by regs=7, by smem=128 |
| decode | d=32, R=1 | 4 | sm_70 | 63 | 0 | 0/0 | 0 (0.0) | 8 | 32/64 | by regs=8, by smem=128 |
| decode | d=32, R=1, paged | 4 | sm_70 | 72 | 0 | 0/0 | 0 (0.0) | 7 | 28/64 | by regs=7, by smem=128 |
| decode | d=32, R=1 | 4 | sm_75 | 64 | 0 | 0/0 | 0 (0.0) | 8 | 32/32 | by regs=8, by smem=85 |
| decode | d=32, R=1, paged | 4 | sm_75 | 69 | 0 | 0/0 | 0 (0.0) | 7 | 28/32 | by regs=7, by smem=85 |
| decode | d=32, R=2 | 4 | sm_60 | 87 | 0 | 0/0 | 0 (0.0) | 5 | 20/64 | by regs=5, by smem=51 |
| decode | d=32, R=2, paged | 4 | sm_60 | 92 | 0 | 0/0 | 0 (0.0) | 5 | 20/64 | by regs=5, by smem=51 |
| decode | d=32, R=2 | 4 | sm_61 | 88 | 0 | 0/0 | 0 (0.0) | 5 | 20/64 | by regs=5, by smem=76 |
| decode | d=32, R=2, paged | 4 | sm_61 | 92 | 0 | 0/0 | 0 (0.0) | 5 | 20/64 | by regs=5, by smem=76 |
| decode | d=32, R=2 | 4 | sm_70 | 95 | 0 | 0/0 | 0 (0.0) | 5 | 20/64 | by regs=5, by smem=76 |
| decode | d=32, R=2, paged | 4 | sm_70 | 94 | 0 | 0/0 | 0 (0.0) | 5 | 20/64 | by regs=5, by smem=76 |
| decode | d=32, R=2 | 4 | sm_75 | 73 | 0 | 0/0 | 0 (0.0) | 6 | 24/32 | by regs=6, by smem=51 |
| decode | d=32, R=2, paged | 4 | sm_75 | 79 | 0 | 0/0 | 0 (0.0) | 6 | 24/32 | by regs=6, by smem=51 |
| decode | d=32, R=4 | 4 | sm_60 | 137 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=28 |
| decode | d=32, R=4, paged | 4 | sm_60 | 135 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=28 |
| decode | d=32, R=4 | 4 | sm_61 | 151 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=42 |
| decode | d=32, R=4, paged | 4 | sm_61 | 151 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=42 |
| decode | d=32, R=4 | 4 | sm_70 | 150 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=42 |
| decode | d=32, R=4, paged | 4 | sm_70 | 154 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=42 |
| decode | d=32, R=4 | 4 | sm_75 | 123 | 0 | 0/0 | 0 (0.0) | 4 | 16/32 | by regs=4, by smem=28 |
| decode | d=32, R=4, paged | 4 | sm_75 | 124 | 0 | 0/0 | 0 (0.0) | 4 | 16/32 | by regs=4, by smem=28 |
| decode | d=32, R=8 | 4 | sm_60 | 168 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=11 |
| decode | d=32, R=8, paged | 4 | sm_60 | 168 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=11 |
| decode | d=32, R=8 | 4 | sm_61 | 171 | 0 | 0/0 | 0 (0.0) | 2 | 8/64 | by regs=2, by smem=17 |
| decode | d=32, R=8, paged | 4 | sm_61 | 171 | 0 | 0/0 | 0 (0.0) | 2 | 8/64 | by regs=2, by smem=17 |
| decode | d=32, R=8 | 4 | sm_70 | 168 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=17 |
| decode | d=32, R=8, paged | 4 | sm_70 | 168 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=17 |
| decode | d=32, R=8 | 4 | sm_75 | 168 | 0 | 0/0 | 0 (0.0) | 3 | 12/32 | by regs=3, by smem=11 |
| decode | d=32, R=8, paged | 4 | sm_75 | 168 | 0 | 0/0 | 0 (0.0) | 3 | 12/32 | by regs=3, by smem=11 |
| decode | d=64, R=1 | 4 | sm_60 | 64 | 0 | 0/0 | 0 (0.0) | 8 | 32/64 | by regs=8, by smem=51 |
| decode | d=64, R=1, paged | 4 | sm_60 | 70 | 0 | 0/0 | 0 (0.0) | 7 | 28/64 | by regs=7, by smem=51 |
| decode | d=64, R=1 | 4 | sm_61 | 64 | 0 | 0/0 | 0 (0.0) | 8 | 32/64 | by regs=8, by smem=76 |
| decode | d=64, R=1, paged | 4 | sm_61 | 69 | 0 | 0/0 | 0 (0.0) | 7 | 28/64 | by regs=7, by smem=76 |
| decode | d=64, R=1 | 4 | sm_70 | 63 | 0 | 0/0 | 0 (0.0) | 8 | 32/64 | by regs=8, by smem=76 |
| decode | d=64, R=1, paged | 4 | sm_70 | 72 | 0 | 0/0 | 0 (0.0) | 7 | 28/64 | by regs=7, by smem=76 |
| decode | d=64, R=1 | 4 | sm_75 | 63 | 0 | 0/0 | 0 (0.0) | 8 | 32/32 | by regs=8, by smem=51 |
| decode | d=64, R=1, paged | 4 | sm_75 | 69 | 0 | 0/0 | 0 (0.0) | 7 | 28/32 | by regs=7, by smem=51 |
| decode | d=64, R=2 | 4 | sm_60 | 88 | 0 | 0/0 | 0 (0.0) | 5 | 20/64 | by regs=5, by smem=28 |
| decode | d=64, R=2, paged | 4 | sm_60 | 92 | 0 | 0/0 | 0 (0.0) | 5 | 20/64 | by regs=5, by smem=28 |
| decode | d=64, R=2 | 4 | sm_61 | 88 | 0 | 0/0 | 0 (0.0) | 5 | 20/64 | by regs=5, by smem=42 |
| decode | d=64, R=2, paged | 4 | sm_61 | 92 | 0 | 0/0 | 0 (0.0) | 5 | 20/64 | by regs=5, by smem=42 |
| decode | d=64, R=2 | 4 | sm_70 | 95 | 0 | 0/0 | 0 (0.0) | 5 | 20/64 | by regs=5, by smem=42 |
| decode | d=64, R=2, paged | 4 | sm_70 | 96 | 0 | 0/0 | 0 (0.0) | 5 | 20/64 | by regs=5, by smem=42 |
| decode | d=64, R=2 | 4 | sm_75 | 73 | 0 | 0/0 | 0 (0.0) | 6 | 24/32 | by regs=6, by smem=28 |
| decode | d=64, R=2, paged | 4 | sm_75 | 80 | 0 | 0/0 | 0 (0.0) | 6 | 24/32 | by regs=6, by smem=28 |
| decode | d=64, R=4 | 4 | sm_60 | 134 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=15 |
| decode | d=64, R=4, paged | 4 | sm_60 | 136 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=15 |
| decode | d=64, R=4 | 4 | sm_61 | 150 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=22 |
| decode | d=64, R=4, paged | 4 | sm_61 | 154 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=22 |
| decode | d=64, R=4 | 4 | sm_70 | 150 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=22 |
| decode | d=64, R=4, paged | 4 | sm_70 | 154 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=22 |
| decode | d=64, R=4 | 4 | sm_75 | 123 | 0 | 0/0 | 0 (0.0) | 4 | 16/32 | by regs=4, by smem=15 |
| decode | d=64, R=4, paged | 4 | sm_75 | 126 | 0 | 0/0 | 0 (0.0) | 4 | 16/32 | by regs=4, by smem=15 |
| decode | d=64, R=8 | 4 | sm_60 | 168 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=6 |
| decode | d=64, R=8, paged | 4 | sm_60 | 168 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=6 |
| decode | d=64, R=8 | 4 | sm_61 | 171 | 0 | 0/0 | 0 (0.0) | 2 | 8/64 | by regs=2, by smem=9 |
| decode | d=64, R=8, paged | 4 | sm_61 | 171 | 0 | 0/0 | 0 (0.0) | 2 | 8/64 | by regs=2, by smem=9 |
| decode | d=64, R=8 | 4 | sm_70 | 167 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=9 |
| decode | d=64, R=8, paged | 4 | sm_70 | 168 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=9 |
| decode | d=64, R=8 | 4 | sm_75 | 165 | 0 | 0/0 | 0 (0.0) | 3 | 12/32 | by regs=3, by smem=6 |
| decode | d=64, R=8, paged | 4 | sm_75 | 168 | 8 | 4/4 | 0 (0.0) | 3 | 12/32 | by regs=3, by smem=6 |
| decode | d=80, R=1 | 4 | sm_60 | 64 | 0 | 0/0 | 0 (0.0) | 8 | 32/64 | by regs=8, by smem=28 |
| decode | d=80, R=1, paged | 4 | sm_60 | 70 | 0 | 0/0 | 0 (0.0) | 7 | 28/64 | by regs=7, by smem=28 |
| decode | d=80, R=1 | 4 | sm_61 | 64 | 0 | 0/0 | 0 (0.0) | 8 | 32/64 | by regs=8, by smem=42 |
| decode | d=80, R=1, paged | 4 | sm_61 | 69 | 0 | 0/0 | 0 (0.0) | 7 | 28/64 | by regs=7, by smem=42 |
| decode | d=80, R=1 | 4 | sm_70 | 64 | 0 | 0/0 | 0 (0.0) | 8 | 32/64 | by regs=8, by smem=42 |
| decode | d=80, R=1, paged | 4 | sm_70 | 72 | 0 | 0/0 | 0 (0.0) | 7 | 28/64 | by regs=7, by smem=42 |
| decode | d=80, R=1 | 4 | sm_75 | 65 | 0 | 0/0 | 0 (0.0) | 7 | 28/32 | by regs=7, by smem=28 |
| decode | d=80, R=1, paged | 4 | sm_75 | 64 | 0 | 0/0 | 0 (0.0) | 8 | 32/32 | by regs=8, by smem=28 |
| decode | d=80, R=2 | 4 | sm_60 | 87 | 0 | 0/0 | 0 (0.0) | 5 | 20/64 | by regs=5, by smem=15 |
| decode | d=80, R=2, paged | 4 | sm_60 | 92 | 0 | 0/0 | 0 (0.0) | 5 | 20/64 | by regs=5, by smem=15 |
| decode | d=80, R=2 | 4 | sm_61 | 87 | 0 | 0/0 | 0 (0.0) | 5 | 20/64 | by regs=5, by smem=22 |
| decode | d=80, R=2, paged | 4 | sm_61 | 92 | 0 | 0/0 | 0 (0.0) | 5 | 20/64 | by regs=5, by smem=22 |
| decode | d=80, R=2 | 4 | sm_70 | 95 | 0 | 0/0 | 0 (0.0) | 5 | 20/64 | by regs=5, by smem=22 |
| decode | d=80, R=2, paged | 4 | sm_70 | 96 | 0 | 0/0 | 0 (0.0) | 5 | 20/64 | by regs=5, by smem=22 |
| decode | d=80, R=2 | 4 | sm_75 | 73 | 0 | 0/0 | 0 (0.0) | 6 | 24/32 | by regs=6, by smem=15 |
| decode | d=80, R=2, paged | 4 | sm_75 | 79 | 0 | 0/0 | 0 (0.0) | 6 | 24/32 | by regs=6, by smem=15 |
| decode | d=80, R=4 | 4 | sm_60 | 134 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=7 |
| decode | d=80, R=4, paged | 4 | sm_60 | 137 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=7 |
| decode | d=80, R=4 | 4 | sm_61 | 152 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=11 |
| decode | d=80, R=4, paged | 4 | sm_61 | 152 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=11 |
| decode | d=80, R=4 | 4 | sm_70 | 150 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=11 |
| decode | d=80, R=4, paged | 4 | sm_70 | 154 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=11 |
| decode | d=80, R=4 | 4 | sm_75 | 124 | 0 | 0/0 | 0 (0.0) | 4 | 16/32 | by regs=4, by smem=7 |
| decode | d=80, R=4, paged | 4 | sm_75 | 125 | 0 | 0/0 | 0 (0.0) | 4 | 16/32 | by regs=4, by smem=7 |
| decode | d=80, R=8 | 4 | sm_60 | 167 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=3 |
| decode | d=80, R=8, paged | 4 | sm_60 | 168 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=3 |
| decode | d=80, R=8 | 4 | sm_61 | 168 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=4 |
| decode | d=80, R=8, paged | 4 | sm_61 | 169 | 0 | 0/0 | 0 (0.0) | 2 | 8/64 | by regs=2, by smem=4 |
| decode | d=80, R=8 | 4 | sm_70 | 168 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=4 |
| decode | d=80, R=8, paged | 4 | sm_70 | 168 | 8 | 4/4 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=4 |
| decode | d=80, R=8 | 4 | sm_75 | 168 | 0 | 0/0 | 0 (0.0) | 3 | 12/32 | by regs=3, by smem=3 |
| decode | d=80, R=8, paged | 4 | sm_75 | 168 | 8 | 4/4 | 0 (0.0) | 3 | 12/32 | by regs=3, by smem=3 |
| decode | d=96, R=1 | 4 | sm_60 | 64 | 0 | 0/0 | 0 (0.0) | 8 | 32/64 | by regs=8, by smem=28 |
| decode | d=96, R=1, paged | 4 | sm_60 | 70 | 0 | 0/0 | 0 (0.0) | 7 | 28/64 | by regs=7, by smem=28 |
| decode | d=96, R=1 | 4 | sm_61 | 64 | 0 | 0/0 | 0 (0.0) | 8 | 32/64 | by regs=8, by smem=42 |
| decode | d=96, R=1, paged | 4 | sm_61 | 69 | 0 | 0/0 | 0 (0.0) | 7 | 28/64 | by regs=7, by smem=42 |
| decode | d=96, R=1 | 4 | sm_70 | 64 | 0 | 0/0 | 0 (0.0) | 8 | 32/64 | by regs=8, by smem=42 |
| decode | d=96, R=1, paged | 4 | sm_70 | 72 | 0 | 0/0 | 0 (0.0) | 7 | 28/64 | by regs=7, by smem=42 |
| decode | d=96, R=1 | 4 | sm_75 | 65 | 0 | 0/0 | 0 (0.0) | 7 | 28/32 | by regs=7, by smem=28 |
| decode | d=96, R=1, paged | 4 | sm_75 | 64 | 0 | 0/0 | 0 (0.0) | 8 | 32/32 | by regs=8, by smem=28 |
| decode | d=96, R=2 | 4 | sm_60 | 87 | 0 | 0/0 | 0 (0.0) | 5 | 20/64 | by regs=5, by smem=15 |
| decode | d=96, R=2, paged | 4 | sm_60 | 92 | 0 | 0/0 | 0 (0.0) | 5 | 20/64 | by regs=5, by smem=15 |
| decode | d=96, R=2 | 4 | sm_61 | 87 | 0 | 0/0 | 0 (0.0) | 5 | 20/64 | by regs=5, by smem=22 |
| decode | d=96, R=2, paged | 4 | sm_61 | 92 | 0 | 0/0 | 0 (0.0) | 5 | 20/64 | by regs=5, by smem=22 |
| decode | d=96, R=2 | 4 | sm_70 | 95 | 0 | 0/0 | 0 (0.0) | 5 | 20/64 | by regs=5, by smem=22 |
| decode | d=96, R=2, paged | 4 | sm_70 | 96 | 0 | 0/0 | 0 (0.0) | 5 | 20/64 | by regs=5, by smem=22 |
| decode | d=96, R=2 | 4 | sm_75 | 73 | 0 | 0/0 | 0 (0.0) | 6 | 24/32 | by regs=6, by smem=15 |
| decode | d=96, R=2, paged | 4 | sm_75 | 79 | 0 | 0/0 | 0 (0.0) | 6 | 24/32 | by regs=6, by smem=15 |
| decode | d=96, R=4 | 4 | sm_60 | 134 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=7 |
| decode | d=96, R=4, paged | 4 | sm_60 | 137 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=7 |
| decode | d=96, R=4 | 4 | sm_61 | 152 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=11 |
| decode | d=96, R=4, paged | 4 | sm_61 | 152 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=11 |
| decode | d=96, R=4 | 4 | sm_70 | 150 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=11 |
| decode | d=96, R=4, paged | 4 | sm_70 | 154 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=11 |
| decode | d=96, R=4 | 4 | sm_75 | 124 | 0 | 0/0 | 0 (0.0) | 4 | 16/32 | by regs=4, by smem=7 |
| decode | d=96, R=4, paged | 4 | sm_75 | 125 | 0 | 0/0 | 0 (0.0) | 4 | 16/32 | by regs=4, by smem=7 |
| decode | d=96, R=8 | 4 | sm_60 | 167 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=3 |
| decode | d=96, R=8, paged | 4 | sm_60 | 168 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=3 |
| decode | d=96, R=8 | 4 | sm_61 | 168 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=4 |
| decode | d=96, R=8, paged | 4 | sm_61 | 169 | 0 | 0/0 | 0 (0.0) | 2 | 8/64 | by regs=2, by smem=4 |
| decode | d=96, R=8 | 4 | sm_70 | 168 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=4 |
| decode | d=96, R=8, paged | 4 | sm_70 | 168 | 8 | 4/4 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=4 |
| decode | d=96, R=8 | 4 | sm_75 | 168 | 0 | 0/0 | 0 (0.0) | 3 | 12/32 | by regs=3, by smem=3 |
| decode | d=96, R=8, paged | 4 | sm_75 | 168 | 8 | 4/4 | 0 (0.0) | 3 | 12/32 | by regs=3, by smem=3 |
| decode | d=128, R=1 | 4 | sm_60 | 64 | 0 | 0/0 | 0 (0.0) | 8 | 32/64 | by regs=8, by smem=28 |
| decode | d=128, R=1, paged | 4 | sm_60 | 70 | 0 | 0/0 | 0 (0.0) | 7 | 28/64 | by regs=7, by smem=28 |
| decode | d=128, R=1 | 4 | sm_61 | 64 | 0 | 0/0 | 0 (0.0) | 8 | 32/64 | by regs=8, by smem=42 |
| decode | d=128, R=1, paged | 4 | sm_61 | 69 | 0 | 0/0 | 0 (0.0) | 7 | 28/64 | by regs=7, by smem=42 |
| decode | d=128, R=1 | 4 | sm_70 | 64 | 0 | 0/0 | 0 (0.0) | 8 | 32/64 | by regs=8, by smem=42 |
| decode | d=128, R=1, paged | 4 | sm_70 | 72 | 0 | 0/0 | 0 (0.0) | 7 | 28/64 | by regs=7, by smem=42 |
| decode | d=128, R=1 | 4 | sm_75 | 65 | 0 | 0/0 | 0 (0.0) | 7 | 28/32 | by regs=7, by smem=28 |
| decode | d=128, R=1, paged | 4 | sm_75 | 64 | 0 | 0/0 | 0 (0.0) | 8 | 32/32 | by regs=8, by smem=28 |
| decode | d=128, R=2 | 4 | sm_60 | 87 | 0 | 0/0 | 0 (0.0) | 5 | 20/64 | by regs=5, by smem=15 |
| decode | d=128, R=2, paged | 4 | sm_60 | 92 | 0 | 0/0 | 0 (0.0) | 5 | 20/64 | by regs=5, by smem=15 |
| decode | d=128, R=2 | 4 | sm_61 | 87 | 0 | 0/0 | 0 (0.0) | 5 | 20/64 | by regs=5, by smem=22 |
| decode | d=128, R=2, paged | 4 | sm_61 | 92 | 0 | 0/0 | 0 (0.0) | 5 | 20/64 | by regs=5, by smem=22 |
| decode | d=128, R=2 | 4 | sm_70 | 95 | 0 | 0/0 | 0 (0.0) | 5 | 20/64 | by regs=5, by smem=22 |
| decode | d=128, R=2, paged | 4 | sm_70 | 96 | 0 | 0/0 | 0 (0.0) | 5 | 20/64 | by regs=5, by smem=22 |
| decode | d=128, R=2 | 4 | sm_75 | 73 | 0 | 0/0 | 0 (0.0) | 6 | 24/32 | by regs=6, by smem=15 |
| decode | d=128, R=2, paged | 4 | sm_75 | 79 | 0 | 0/0 | 0 (0.0) | 6 | 24/32 | by regs=6, by smem=15 |
| decode | d=128, R=4 | 4 | sm_60 | 134 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=7 |
| decode | d=128, R=4, paged | 4 | sm_60 | 137 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=7 |
| decode | d=128, R=4 | 4 | sm_61 | 152 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=11 |
| decode | d=128, R=4, paged | 4 | sm_61 | 152 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=11 |
| decode | d=128, R=4 | 4 | sm_70 | 150 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=11 |
| decode | d=128, R=4, paged | 4 | sm_70 | 154 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=11 |
| decode | d=128, R=4 | 4 | sm_75 | 124 | 0 | 0/0 | 0 (0.0) | 4 | 16/32 | by regs=4, by smem=7 |
| decode | d=128, R=4, paged | 4 | sm_75 | 125 | 0 | 0/0 | 0 (0.0) | 4 | 16/32 | by regs=4, by smem=7 |
| decode | d=128, R=8 | 4 | sm_60 | 167 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=3 |
| decode | d=128, R=8, paged | 4 | sm_60 | 168 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=3 |
| decode | d=128, R=8 | 4 | sm_61 | 168 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=4 |
| decode | d=128, R=8, paged | 4 | sm_61 | 169 | 0 | 0/0 | 0 (0.0) | 2 | 8/64 | by regs=2, by smem=4 |
| decode | d=128, R=8 | 4 | sm_70 | 168 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=4 |
| decode | d=128, R=8, paged | 4 | sm_70 | 168 | 8 | 4/4 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=4 |
| decode | d=128, R=8 | 4 | sm_75 | 168 | 0 | 0/0 | 0 (0.0) | 3 | 12/32 | by regs=3, by smem=3 |
| decode | d=128, R=8, paged | 4 | sm_75 | 168 | 8 | 4/4 | 0 (0.0) | 3 | 12/32 | by regs=3, by smem=3 |
| decode | d=256, R=1 | 4 | sm_60 | 56 | 0 | 0/0 | 0 (0.0) | 9 | 36/64 | by regs=9, by smem=15 |
| decode | d=256, R=1, paged | 4 | sm_60 | 70 | 0 | 0/0 | 0 (0.0) | 7 | 28/64 | by regs=7, by smem=15 |
| decode | d=256, R=1 | 4 | sm_61 | 60 | 0 | 0/0 | 0 (0.0) | 8 | 32/64 | by regs=8, by smem=22 |
| decode | d=256, R=1, paged | 4 | sm_61 | 69 | 0 | 0/0 | 0 (0.0) | 7 | 28/64 | by regs=7, by smem=22 |
| decode | d=256, R=1 | 4 | sm_70 | 64 | 0 | 0/0 | 0 (0.0) | 8 | 32/64 | by regs=8, by smem=22 |
| decode | d=256, R=1, paged | 4 | sm_70 | 71 | 0 | 0/0 | 0 (0.0) | 7 | 28/64 | by regs=7, by smem=22 |
| decode | d=256, R=1 | 4 | sm_75 | 64 | 8 | 4/4 | 0 (0.0) | 8 | 32/32 | by regs=8, by smem=15 |
| decode | d=256, R=1, paged | 4 | sm_75 | 64 | 0 | 0/0 | 0 (0.0) | 8 | 32/32 | by regs=8, by smem=15 |
| decode | d=256, R=2 | 4 | sm_60 | 88 | 0 | 0/0 | 0 (0.0) | 5 | 20/64 | by regs=5, by smem=7 |
| decode | d=256, R=2, paged | 4 | sm_60 | 92 | 0 | 0/0 | 0 (0.0) | 5 | 20/64 | by regs=5, by smem=7 |
| decode | d=256, R=2 | 4 | sm_61 | 90 | 0 | 0/0 | 0 (0.0) | 5 | 20/64 | by regs=5, by smem=11 |
| decode | d=256, R=2, paged | 4 | sm_61 | 92 | 0 | 0/0 | 0 (0.0) | 5 | 20/64 | by regs=5, by smem=11 |
| decode | d=256, R=2 | 4 | sm_70 | 93 | 0 | 0/0 | 0 (0.0) | 5 | 20/64 | by regs=5, by smem=11 |
| decode | d=256, R=2, paged | 4 | sm_70 | 95 | 0 | 0/0 | 0 (0.0) | 5 | 20/64 | by regs=5, by smem=11 |
| decode | d=256, R=2 | 4 | sm_75 | 87 | 0 | 0/0 | 0 (0.0) | 5 | 20/32 | by regs=5, by smem=7 |
| decode | d=256, R=2, paged | 4 | sm_75 | 79 | 0 | 0/0 | 0 (0.0) | 6 | 24/32 | by regs=6, by smem=7 |
| decode | d=256, R=4 | 4 | sm_60 | 153 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=3 |
| decode | d=256, R=4, paged | 4 | sm_60 | 156 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=3 |
| decode | d=256, R=4 | 4 | sm_61 | 152 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=5 |
| decode | d=256, R=4, paged | 4 | sm_61 | 150 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=5 |
| decode | d=256, R=4 | 4 | sm_70 | 150 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=5 |
| decode | d=256, R=4, paged | 4 | sm_70 | 154 | 0 | 0/0 | 0 (0.0) | 3 | 12/64 | by regs=3, by smem=5 |
| decode | d=256, R=4 | 4 | sm_75 | 148 | 0 | 0/0 | 0 (0.0) | 3 | 12/32 | by regs=3, by smem=3 |
| decode | d=256, R=4, paged | 4 | sm_75 | 152 | 0 | 0/0 | 0 (0.0) | 3 | 12/32 | by regs=3, by smem=3 |
| decode | d=256, R=8 | 4 | sm_60 | 214 | 0 | 0/0 | 0 (0.0) | 1 | 4/64 | by regs=2, by smem=1 |
| decode | d=256, R=8, paged | 4 | sm_60 | 215 | 0 | 0/0 | 0 (0.0) | 1 | 4/64 | by regs=2, by smem=1 |
| decode | d=256, R=8 | 4 | sm_61 | 217 | 0 | 0/0 | 0 (0.0) | 2 | 8/64 | by regs=2, by smem=2 |
| decode | d=256, R=8, paged | 4 | sm_61 | 215 | 0 | 0/0 | 0 (0.0) | 2 | 8/64 | by regs=2, by smem=2 |
| decode | d=256, R=8 | 4 | sm_70 | 209 | 0 | 0/0 | 0 (0.0) | 2 | 8/64 | by regs=2, by smem=2 |
| decode | d=256, R=8, paged | 4 | sm_70 | 211 | 0 | 0/0 | 0 (0.0) | 2 | 8/64 | by regs=2, by smem=2 |
| decode | d=256, R=8 | 4 | sm_75 | 207 | 0 | 0/0 | 0 (0.0) | 1 | 4/32 | by regs=2, by smem=1 |
| decode | d=256, R=8, paged | 4 | sm_75 | 209 | 0 | 0/0 | 0 (0.0) | 1 | 4/32 | by regs=2, by smem=1 |
| fwd_pascal | FwdTileSimt&lt;16, 8, 256, 8> | 8 | sm_60 | 103 | 0 | 0/0 | 42496 (41.5) | 1 | 8/64 | by regs=2, by smem=1 |
| fwd_pascal | FwdTileSimt&lt;16, 8, 256, 8> | 8 | sm_61 | 103 | 0 | 0/0 | 42496 (41.5) | 2 | 16/64 | by regs=2, by smem=2 |
| fwd_pascal | FwdTileSimt&lt;16, 8, 256, 8> | 8 | sm_70 | 104 | 0 | 0/0 | 42496 (41.5) | 2 | 16/64 | by regs=2, by smem=2 |
| fwd_pascal | FwdTileSimt&lt;16, 8, 256, 8> | 8 | sm_75 | 104 | 0 | 0/0 | 42496 (41.5) | 1 | 8/32 | by regs=2, by smem=1 |
| fwd_pascal | FwdTileSimt&lt;32, 8, 128, 1> | 2 | sm_60 | 173 | 0 | 0/0 | 26624 (26.0) | 2 | 4/64 | by regs=5, by smem=2 |
| fwd_pascal | FwdTileSimt&lt;32, 8, 128, 1> | 2 | sm_61 | 173 | 0 | 0/0 | 26624 (26.0) | 3 | 6/64 | by regs=4, by smem=3 |
| fwd_pascal | FwdTileSimt&lt;32, 8, 128, 1> | 2 | sm_70 | 175 | 0 | 0/0 | 26624 (26.0) | 3 | 6/64 | by regs=4, by smem=3 |
| fwd_pascal | FwdTileSimt&lt;32, 8, 128, 1> | 2 | sm_75 | 175 | 0 | 0/0 | 26624 (26.0) | 2 | 4/32 | by regs=4, by smem=2 |
| fwd_pascal | FwdTileSimt&lt;64, 16, 64, 1> | 4 | sm_60 | 128 | 8 | 8/8 | 31232 (30.5) | 2 | 8/64 | by regs=4, by smem=2 |
| fwd_pascal | FwdTileSimt&lt;64, 16, 64, 1> | 4 | sm_61 | 128 | 8 | 8/8 | 31232 (30.5) | 3 | 12/64 | by regs=4, by smem=3 |
| fwd_pascal | FwdTileSimt&lt;64, 16, 64, 1> | 4 | sm_70 | 136 | 0 | 0/0 | 31232 (30.5) | 3 | 12/64 | by regs=3, by smem=3 |
| fwd_pascal | FwdTileSimt&lt;64, 16, 64, 1> | 4 | sm_75 | 136 | 0 | 0/0 | 31232 (30.5) | 2 | 8/32 | by regs=3, by smem=2 |
| fwd_pascal | FwdTileSimt&lt;64, 16, 96, 1> | 4 | sm_60 | 167 | 0 | 0/0 | 43520 (42.5) | 1 | 4/64 | by regs=3, by smem=1 |
| fwd_pascal | FwdTileSimt&lt;64, 16, 96, 1> | 4 | sm_61 | 167 | 0 | 0/0 | 43520 (42.5) | 2 | 8/64 | by regs=3, by smem=2 |
| fwd_pascal | FwdTileSimt&lt;64, 16, 96, 1> | 4 | sm_70 | 168 | 16 | 12/12 | 43520 (42.5) | 2 | 8/64 | by regs=3, by smem=2 |
| fwd_pascal | FwdTileSimt&lt;64, 16, 96, 1> | 4 | sm_75 | 168 | 8 | 8/8 | 43520 (42.5) | 1 | 4/32 | by regs=3, by smem=1 |
| fwd_pascal | FwdTileSimt&lt;64, 32, 32, 1> | 4 | sm_60 | 128 | 0 | 0/0 | 28672 (28.0) | 2 | 8/64 | by regs=4, by smem=2 |
| fwd_pascal | FwdTileSimt&lt;64, 32, 32, 1> | 4 | sm_61 | 128 | 0 | 0/0 | 28672 (28.0) | 3 | 12/64 | by regs=4, by smem=3 |
| fwd_pascal | FwdTileSimt&lt;64, 32, 32, 1> | 4 | sm_70 | 146 | 0 | 0/0 | 28672 (28.0) | 3 | 12/64 | by regs=3, by smem=3 |
| fwd_pascal | FwdTileSimt&lt;64, 32, 32, 1> | 4 | sm_75 | 136 | 0 | 0/0 | 28672 (28.0) | 2 | 8/32 | by regs=3, by smem=2 |
| fwd_pascal_cache | FwdTileSimt&lt;16, 8, 256, 8> | 8 | sm_60 | 101 | 0 | 0/0 | 42496 (41.5) | 1 | 8/64 | by regs=2, by smem=1 |
| fwd_pascal_cache | FwdTileSimt&lt;16, 8, 256, 8> | 8 | sm_61 | 102 | 0 | 0/0 | 42496 (41.5) | 2 | 16/64 | by regs=2, by smem=2 |
| fwd_pascal_cache | FwdTileSimt&lt;16, 8, 256, 8> | 8 | sm_70 | 103 | 0 | 0/0 | 42496 (41.5) | 2 | 16/64 | by regs=2, by smem=2 |
| fwd_pascal_cache | FwdTileSimt&lt;16, 8, 256, 8> | 8 | sm_75 | 103 | 0 | 0/0 | 42496 (41.5) | 1 | 8/32 | by regs=2, by smem=1 |
| fwd_pascal_cache | FwdTileSimt&lt;32, 8, 128, 1> | 2 | sm_60 | 171 | 0 | 0/0 | 26624 (26.0) | 2 | 4/64 | by regs=5, by smem=2 |
| fwd_pascal_cache | FwdTileSimt&lt;32, 8, 128, 1> | 2 | sm_61 | 172 | 0 | 0/0 | 26624 (26.0) | 3 | 6/64 | by regs=4, by smem=3 |
| fwd_pascal_cache | FwdTileSimt&lt;32, 8, 128, 1> | 2 | sm_70 | 174 | 0 | 0/0 | 26624 (26.0) | 3 | 6/64 | by regs=4, by smem=3 |
| fwd_pascal_cache | FwdTileSimt&lt;32, 8, 128, 1> | 2 | sm_75 | 174 | 0 | 0/0 | 26624 (26.0) | 2 | 4/32 | by regs=4, by smem=2 |
| fwd_pascal_cache | FwdTileSimt&lt;64, 16, 64, 1> | 4 | sm_60 | 133 | 0 | 0/0 | 31232 (30.5) | 2 | 8/64 | by regs=3, by smem=2 |
| fwd_pascal_cache | FwdTileSimt&lt;64, 16, 64, 1> | 4 | sm_61 | 133 | 0 | 0/0 | 31232 (30.5) | 3 | 12/64 | by regs=3, by smem=3 |
| fwd_pascal_cache | FwdTileSimt&lt;64, 16, 64, 1> | 4 | sm_70 | 139 | 0 | 0/0 | 31232 (30.5) | 3 | 12/64 | by regs=3, by smem=3 |
| fwd_pascal_cache | FwdTileSimt&lt;64, 16, 64, 1> | 4 | sm_75 | 139 | 0 | 0/0 | 31232 (30.5) | 2 | 8/32 | by regs=3, by smem=2 |
| fwd_pascal_cache | FwdTileSimt&lt;64, 16, 96, 1> | 4 | sm_60 | 168 | 0 | 0/0 | 43520 (42.5) | 1 | 4/64 | by regs=3, by smem=1 |
| fwd_pascal_cache | FwdTileSimt&lt;64, 16, 96, 1> | 4 | sm_61 | 168 | 0 | 0/0 | 43520 (42.5) | 2 | 8/64 | by regs=3, by smem=2 |
| fwd_pascal_cache | FwdTileSimt&lt;64, 16, 96, 1> | 4 | sm_70 | 181 | 0 | 0/0 | 43520 (42.5) | 2 | 8/64 | by regs=2, by smem=2 |
| fwd_pascal_cache | FwdTileSimt&lt;64, 16, 96, 1> | 4 | sm_75 | 181 | 0 | 0/0 | 43520 (42.5) | 1 | 4/32 | by regs=2, by smem=1 |
| fwd_pascal_cache | FwdTileSimt&lt;64, 32, 32, 1> | 4 | sm_60 | 122 | 0 | 0/0 | 28672 (28.0) | 2 | 8/64 | by regs=4, by smem=2 |
| fwd_pascal_cache | FwdTileSimt&lt;64, 32, 32, 1> | 4 | sm_61 | 128 | 0 | 0/0 | 28672 (28.0) | 3 | 12/64 | by regs=4, by smem=3 |
| fwd_pascal_cache | FwdTileSimt&lt;64, 32, 32, 1> | 4 | sm_70 | 142 | 0 | 0/0 | 28672 (28.0) | 3 | 12/64 | by regs=3, by smem=3 |
| fwd_pascal_cache | FwdTileSimt&lt;64, 32, 32, 1> | 4 | sm_75 | 144 | 0 | 0/0 | 28672 (28.0) | 2 | 8/32 | by regs=3, by smem=2 |
| fwd_pascal_varlen | FwdTileSimt&lt;16, 8, 256, 8> | 8 | sm_60 | 94 | 0 | 0/0 | 42496 (41.5) | 1 | 8/64 | by regs=2, by smem=1 |
| fwd_pascal_varlen | FwdTileSimt&lt;16, 8, 256, 8> | 8 | sm_61 | 99 | 0 | 0/0 | 42496 (41.5) | 2 | 16/64 | by regs=2, by smem=2 |
| fwd_pascal_varlen | FwdTileSimt&lt;16, 8, 256, 8> | 8 | sm_70 | 101 | 0 | 0/0 | 42496 (41.5) | 2 | 16/64 | by regs=2, by smem=2 |
| fwd_pascal_varlen | FwdTileSimt&lt;16, 8, 256, 8> | 8 | sm_75 | 101 | 0 | 0/0 | 42496 (41.5) | 1 | 8/32 | by regs=2, by smem=1 |
| fwd_pascal_varlen | FwdTileSimt&lt;32, 8, 128, 1> | 2 | sm_60 | 168 | 0 | 0/0 | 26624 (26.0) | 2 | 4/64 | by regs=6, by smem=2 |
| fwd_pascal_varlen | FwdTileSimt&lt;32, 8, 128, 1> | 2 | sm_61 | 168 | 0 | 0/0 | 26624 (26.0) | 3 | 6/64 | by regs=6, by smem=3 |
| fwd_pascal_varlen | FwdTileSimt&lt;32, 8, 128, 1> | 2 | sm_70 | 173 | 0 | 0/0 | 26624 (26.0) | 3 | 6/64 | by regs=4, by smem=3 |
| fwd_pascal_varlen | FwdTileSimt&lt;32, 8, 128, 1> | 2 | sm_75 | 173 | 0 | 0/0 | 26624 (26.0) | 2 | 4/32 | by regs=4, by smem=2 |
| fwd_pascal_varlen | FwdTileSimt&lt;64, 16, 64, 1> | 4 | sm_60 | 128 | 0 | 0/0 | 31232 (30.5) | 2 | 8/64 | by regs=4, by smem=2 |
| fwd_pascal_varlen | FwdTileSimt&lt;64, 16, 64, 1> | 4 | sm_61 | 128 | 0 | 0/0 | 31232 (30.5) | 3 | 12/64 | by regs=4, by smem=3 |
| fwd_pascal_varlen | FwdTileSimt&lt;64, 16, 64, 1> | 4 | sm_70 | 136 | 0 | 0/0 | 31232 (30.5) | 3 | 12/64 | by regs=3, by smem=3 |
| fwd_pascal_varlen | FwdTileSimt&lt;64, 16, 64, 1> | 4 | sm_75 | 136 | 0 | 0/0 | 31232 (30.5) | 2 | 8/32 | by regs=3, by smem=2 |
| fwd_pascal_varlen | FwdTileSimt&lt;64, 16, 96, 1> | 4 | sm_60 | 161 | 0 | 0/0 | 43520 (42.5) | 1 | 4/64 | by regs=3, by smem=1 |
| fwd_pascal_varlen | FwdTileSimt&lt;64, 16, 96, 1> | 4 | sm_61 | 161 | 0 | 0/0 | 43520 (42.5) | 2 | 8/64 | by regs=3, by smem=2 |
| fwd_pascal_varlen | FwdTileSimt&lt;64, 16, 96, 1> | 4 | sm_70 | 168 | 0 | 0/0 | 43520 (42.5) | 2 | 8/64 | by regs=3, by smem=2 |
| fwd_pascal_varlen | FwdTileSimt&lt;64, 16, 96, 1> | 4 | sm_75 | 168 | 0 | 0/0 | 43520 (42.5) | 1 | 4/32 | by regs=3, by smem=1 |
| fwd_pascal_varlen | FwdTileSimt&lt;64, 32, 32, 1> | 4 | sm_60 | 124 | 0 | 0/0 | 28672 (28.0) | 2 | 8/64 | by regs=4, by smem=2 |
| fwd_pascal_varlen | FwdTileSimt&lt;64, 32, 32, 1> | 4 | sm_61 | 126 | 0 | 0/0 | 28672 (28.0) | 3 | 12/64 | by regs=4, by smem=3 |
| fwd_pascal_varlen | FwdTileSimt&lt;64, 32, 32, 1> | 4 | sm_70 | 140 | 0 | 0/0 | 28672 (28.0) | 3 | 12/64 | by regs=3, by smem=3 |
| fwd_pascal_varlen | FwdTileSimt&lt;64, 32, 32, 1> | 4 | sm_75 | 140 | 0 | 0/0 | 28672 (28.0) | 2 | 8/32 | by regs=3, by smem=2 |
| fwd_turing | FwdTileMma&lt;64, 16, 256> | 4 | sm_75 | 255 | 56 | 68/56 | 50688 (49.5) | 1 | 4/32 | by regs=2, by smem=1 |
| fwd_turing | FwdTileMma&lt;128, 48, 128> | 8 | sm_75 | 255 | 0 | 0/0 | 60928 (59.5) | 1 | 8/32 | by regs=1, by smem=1 |
| fwd_turing | FwdTileMma&lt;128, 64, 32> | 8 | sm_75 | 127 | 0 | 0/0 | 20480 (20.0) | 2 | 16/32 | by regs=2, by smem=3 |
| fwd_turing | FwdTileMma&lt;128, 64, 64> | 8 | sm_75 | 178 | 0 | 0/0 | 36864 (36.0) | 1 | 8/32 | by regs=1, by smem=1 |
| fwd_turing | FwdTileMma&lt;128, 64, 80> | 8 | sm_75 | 209 | 0 | 0/0 | 45056 (44.0) | 1 | 8/32 | by regs=1, by smem=1 |
| fwd_turing | FwdTileMma&lt;128, 64, 96> | 8 | sm_75 | 221 | 0 | 0/0 | 53248 (52.0) | 1 | 8/32 | by regs=1, by smem=1 |
| fwd_turing_cache | FwdTileMma&lt;64, 16, 256> | 4 | sm_75 | 255 | 56 | 60/84 | 50688 (49.5) | 1 | 4/32 | by regs=2, by smem=1 |
| fwd_turing_cache | FwdTileMma&lt;128, 48, 128> | 8 | sm_75 | 255 | 8 | 4/8 | 60928 (59.5) | 1 | 8/32 | by regs=1, by smem=1 |
| fwd_turing_cache | FwdTileMma&lt;128, 64, 32> | 8 | sm_75 | 128 | 0 | 0/0 | 20480 (20.0) | 2 | 16/32 | by regs=2, by smem=3 |
| fwd_turing_cache | FwdTileMma&lt;128, 64, 64> | 8 | sm_75 | 181 | 0 | 0/0 | 36864 (36.0) | 1 | 8/32 | by regs=1, by smem=1 |
| fwd_turing_cache | FwdTileMma&lt;128, 64, 80> | 8 | sm_75 | 212 | 0 | 0/0 | 45056 (44.0) | 1 | 8/32 | by regs=1, by smem=1 |
| fwd_turing_cache | FwdTileMma&lt;128, 64, 96> | 8 | sm_75 | 226 | 0 | 0/0 | 53248 (52.0) | 1 | 8/32 | by regs=1, by smem=1 |
| fwd_turing_varlen | FwdTileMma&lt;64, 16, 256> | 4 | sm_75 | 255 | 56 | 60/72 | 50688 (49.5) | 1 | 4/32 | by regs=2, by smem=1 |
| fwd_turing_varlen | FwdTileMma&lt;128, 48, 128> | 8 | sm_75 | 255 | 0 | 0/0 | 60928 (59.5) | 1 | 8/32 | by regs=1, by smem=1 |
| fwd_turing_varlen | FwdTileMma&lt;128, 64, 32> | 8 | sm_75 | 128 | 0 | 0/0 | 20480 (20.0) | 2 | 16/32 | by regs=2, by smem=3 |
| fwd_turing_varlen | FwdTileMma&lt;128, 64, 64> | 8 | sm_75 | 192 | 0 | 0/0 | 36864 (36.0) | 1 | 8/32 | by regs=1, by smem=1 |
| fwd_turing_varlen | FwdTileMma&lt;128, 64, 80> | 8 | sm_75 | 227 | 0 | 0/0 | 45056 (44.0) | 1 | 8/32 | by regs=1, by smem=1 |
| fwd_turing_varlen | FwdTileMma&lt;128, 64, 96> | 8 | sm_75 | 233 | 0 | 0/0 | 53248 (52.0) | 1 | 8/32 | by regs=1, by smem=1 |
| fwd_volta | FwdTileWmma&lt;64, 16, 256> | 4 | sm_70 | 255 | 16 | 10/12 | 57088 (55.8) | 1 | 4/64 | by regs=2, by smem=1 |
| fwd_volta | FwdTileWmma&lt;64, 16, 256> | 4 | sm_75 | 255 | 88 | 96/228 | 57088 (55.8) | 1 | 4/32 | by regs=2, by smem=1 |
| fwd_volta | FwdTileWmma&lt;128, 16, 80> | 8 | sm_70 | 128 | 0 | 0/0 | 40960 (40.0) | 2 | 16/64 | by regs=2, by smem=2 |
| fwd_volta | FwdTileWmma&lt;128, 16, 80> | 8 | sm_75 | 124 | 0 | 0/0 | 40960 (40.0) | 1 | 8/32 | by regs=2, by smem=1 |
| fwd_volta | FwdTileWmma&lt;128, 16, 96> | 8 | sm_70 | 162 | 0 | 0/0 | 46080 (45.0) | 1 | 8/64 | by regs=1, by smem=2 |
| fwd_volta | FwdTileWmma&lt;128, 16, 96> | 8 | sm_75 | 123 | 0 | 0/0 | 46080 (45.0) | 1 | 8/32 | by regs=2, by smem=1 |
| fwd_volta | FwdTileWmma&lt;128, 16, 128> | 8 | sm_70 | 166 | 0 | 0/0 | 56320 (55.0) | 1 | 8/64 | by regs=1, by smem=1 |
| fwd_volta | FwdTileWmma&lt;128, 16, 128> | 8 | sm_75 | 156 | 0 | 0/0 | 56320 (55.0) | 1 | 8/32 | by regs=1, by smem=1 |
| fwd_volta | FwdTileWmma&lt;128, 16, 256> | 8 | sm_70 | 247 | 0 | 0/0 | 97280 (95.0) | 1 | 8/64 | by regs=1, by smem=1 |
| fwd_volta | FwdTileWmma&lt;128, 16, 256> | 8 | sm_75 | 249 | 0 | 0/0 | 97280 (95.0) | - | - | FAIL(smem>64KB) |
| fwd_volta | FwdTileWmma&lt;128, 32, 32> | 8 | sm_70 | 125 | 0 | 0/0 | 36352 (35.5) | 2 | 16/64 | by regs=2, by smem=2 |
| fwd_volta | FwdTileWmma&lt;128, 32, 32> | 8 | sm_75 | 105 | 0 | 0/0 | 36352 (35.5) | 1 | 8/32 | by regs=2, by smem=1 |
| fwd_volta | FwdTileWmma&lt;128, 32, 64> | 8 | sm_70 | 128 | 8 | 6/8 | 48640 (47.5) | 2 | 16/64 | by regs=2, by smem=2 |
| fwd_volta | FwdTileWmma&lt;128, 32, 64> | 8 | sm_75 | 128 | 0 | 0/0 | 48640 (47.5) | 1 | 8/32 | by regs=2, by smem=1 |
| fwd_volta | FwdTileWmma&lt;128, 32, 128> | 8 | sm_70 | 202 | 0 | 0/0 | 73216 (71.5) | 1 | 8/64 | by regs=1, by smem=1 |
| fwd_volta | FwdTileWmma&lt;128, 32, 128> | 8 | sm_75 | 183 | 0 | 0/0 | 73216 (71.5) | - | - | FAIL(smem>64KB) |
| fwd_volta_cache | FwdTileWmma&lt;64, 16, 256> | 4 | sm_70 | 255 | 24 | 18/20 | 57088 (55.8) | 1 | 4/64 | by regs=2, by smem=1 |
| fwd_volta_cache | FwdTileWmma&lt;64, 16, 256> | 4 | sm_75 | 255 | 64 | 62/120 | 57088 (55.8) | 1 | 4/32 | by regs=2, by smem=1 |
| fwd_volta_cache | FwdTileWmma&lt;128, 16, 80> | 8 | sm_70 | 128 | 0 | 0/0 | 40960 (40.0) | 2 | 16/64 | by regs=2, by smem=2 |
| fwd_volta_cache | FwdTileWmma&lt;128, 16, 80> | 8 | sm_75 | 135 | 0 | 0/0 | 40960 (40.0) | 1 | 8/32 | by regs=1, by smem=1 |
| fwd_volta_cache | FwdTileWmma&lt;128, 16, 96> | 8 | sm_70 | 163 | 0 | 0/0 | 46080 (45.0) | 1 | 8/64 | by regs=1, by smem=2 |
| fwd_volta_cache | FwdTileWmma&lt;128, 16, 96> | 8 | sm_75 | 128 | 0 | 0/0 | 46080 (45.0) | 1 | 8/32 | by regs=2, by smem=1 |
| fwd_volta_cache | FwdTileWmma&lt;128, 16, 128> | 8 | sm_70 | 164 | 0 | 0/0 | 56320 (55.0) | 1 | 8/64 | by regs=1, by smem=1 |
| fwd_volta_cache | FwdTileWmma&lt;128, 16, 128> | 8 | sm_75 | 158 | 0 | 0/0 | 56320 (55.0) | 1 | 8/32 | by regs=1, by smem=1 |
| fwd_volta_cache | FwdTileWmma&lt;128, 16, 256> | 8 | sm_70 | 253 | 0 | 0/0 | 97280 (95.0) | 1 | 8/64 | by regs=1, by smem=1 |
| fwd_volta_cache | FwdTileWmma&lt;128, 16, 256> | 8 | sm_75 | 251 | 0 | 0/0 | 97280 (95.0) | - | - | FAIL(smem>64KB) |
| fwd_volta_cache | FwdTileWmma&lt;128, 32, 32> | 8 | sm_70 | 128 | 0 | 0/0 | 36352 (35.5) | 2 | 16/64 | by regs=2, by smem=2 |
| fwd_volta_cache | FwdTileWmma&lt;128, 32, 32> | 8 | sm_75 | 119 | 0 | 0/0 | 36352 (35.5) | 1 | 8/32 | by regs=2, by smem=1 |
| fwd_volta_cache | FwdTileWmma&lt;128, 32, 64> | 8 | sm_70 | 128 | 8 | 4/4 | 48640 (47.5) | 2 | 16/64 | by regs=2, by smem=2 |
| fwd_volta_cache | FwdTileWmma&lt;128, 32, 64> | 8 | sm_75 | 128 | 0 | 0/0 | 48640 (47.5) | 1 | 8/32 | by regs=2, by smem=1 |
| fwd_volta_cache | FwdTileWmma&lt;128, 32, 128> | 8 | sm_70 | 202 | 0 | 0/0 | 73216 (71.5) | 1 | 8/64 | by regs=1, by smem=1 |
| fwd_volta_cache | FwdTileWmma&lt;128, 32, 128> | 8 | sm_75 | 180 | 0 | 0/0 | 73216 (71.5) | - | - | FAIL(smem>64KB) |
| fwd_volta_varlen | FwdTileWmma&lt;64, 16, 256> | 4 | sm_70 | 255 | 16 | 18/16 | 57088 (55.8) | 1 | 4/64 | by regs=2, by smem=1 |
| fwd_volta_varlen | FwdTileWmma&lt;64, 16, 256> | 4 | sm_75 | 255 | 40 | 46/168 | 57088 (55.8) | 1 | 4/32 | by regs=2, by smem=1 |
| fwd_volta_varlen | FwdTileWmma&lt;128, 16, 80> | 8 | sm_70 | 128 | 0 | 0/0 | 40960 (40.0) | 2 | 16/64 | by regs=2, by smem=2 |
| fwd_volta_varlen | FwdTileWmma&lt;128, 16, 80> | 8 | sm_75 | 123 | 0 | 0/0 | 40960 (40.0) | 1 | 8/32 | by regs=2, by smem=1 |
| fwd_volta_varlen | FwdTileWmma&lt;128, 16, 96> | 8 | sm_70 | 128 | 0 | 0/0 | 46080 (45.0) | 2 | 16/64 | by regs=2, by smem=2 |
| fwd_volta_varlen | FwdTileWmma&lt;128, 16, 96> | 8 | sm_75 | 144 | 0 | 0/0 | 46080 (45.0) | 1 | 8/32 | by regs=1, by smem=1 |
| fwd_volta_varlen | FwdTileWmma&lt;128, 16, 128> | 8 | sm_70 | 164 | 0 | 0/0 | 56320 (55.0) | 1 | 8/64 | by regs=1, by smem=1 |
| fwd_volta_varlen | FwdTileWmma&lt;128, 16, 128> | 8 | sm_75 | 150 | 0 | 0/0 | 56320 (55.0) | 1 | 8/32 | by regs=1, by smem=1 |
| fwd_volta_varlen | FwdTileWmma&lt;128, 16, 256> | 8 | sm_70 | 248 | 0 | 0/0 | 97280 (95.0) | 1 | 8/64 | by regs=1, by smem=1 |
| fwd_volta_varlen | FwdTileWmma&lt;128, 16, 256> | 8 | sm_75 | 245 | 0 | 0/0 | 97280 (95.0) | - | - | FAIL(smem>64KB) |
| fwd_volta_varlen | FwdTileWmma&lt;128, 32, 32> | 8 | sm_70 | 122 | 0 | 0/0 | 36352 (35.5) | 2 | 16/64 | by regs=2, by smem=2 |
| fwd_volta_varlen | FwdTileWmma&lt;128, 32, 32> | 8 | sm_75 | 100 | 0 | 0/0 | 36352 (35.5) | 1 | 8/32 | by regs=2, by smem=1 |
| fwd_volta_varlen | FwdTileWmma&lt;128, 32, 64> | 8 | sm_70 | 128 | 0 | 0/0 | 48640 (47.5) | 2 | 16/64 | by regs=2, by smem=2 |
| fwd_volta_varlen | FwdTileWmma&lt;128, 32, 64> | 8 | sm_75 | 126 | 0 | 0/0 | 48640 (47.5) | 1 | 8/32 | by regs=2, by smem=1 |
| fwd_volta_varlen | FwdTileWmma&lt;128, 32, 128> | 8 | sm_70 | 172 | 0 | 0/0 | 73216 (71.5) | 1 | 8/64 | by regs=1, by smem=1 |
| fwd_volta_varlen | FwdTileWmma&lt;128, 32, 128> | 8 | sm_75 | 182 | 0 | 0/0 | 73216 (71.5) | - | - | FAIL(smem>64KB) |
| precompute_D | - | 8 | sm_60 | 42 | 0 | 0/0 | 0 (0.0) | 5 | 40/64 | by regs=5, by smem=inf |
| precompute_D | - | 8 | sm_61 | 43 | 0 | 0/0 | 0 (0.0) | 5 | 40/64 | by regs=5, by smem=inf |
| precompute_D | - | 8 | sm_70 | 46 | 0 | 0/0 | 0 (0.0) | 5 | 40/64 | by regs=5, by smem=inf |
| precompute_D | - | 8 | sm_75 | 60 | 0 | 0/0 | 0 (0.0) | 4 | 32/32 | by regs=4, by smem=inf |
| split_combine | - | 8 | sm_60 | 39 | 0 | 0/0 | 0 (0.0) | 6 | 48/64 | by regs=6, by smem=inf |
| split_combine | - | 8 | sm_61 | 39 | 0 | 0/0 | 0 (0.0) | 6 | 48/64 | by regs=6, by smem=inf |
| split_combine | - | 8 | sm_70 | 40 | 0 | 0/0 | 0 (0.0) | 6 | 48/64 | by regs=6, by smem=inf |
| split_combine | - | 8 | sm_75 | 40 | 0 | 0/0 | 0 (0.0) | 4 | 32/32 | by regs=6, by smem=inf |
