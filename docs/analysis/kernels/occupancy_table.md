| kernel | tile (Br,Bc,d,warps) | arch | regs | stack B | spill st/ld B | static smem | dyn smem B (KB) | blocks/SM | warps/SM (max) | blocks/SM limits |
|---|---|---|---|---|---|---|---|---|---|---|
| fwd_volta | TileVolta_d32 (128,32,32,4) | sm_70 | 128 | 0 | 0/0 | 0 | 36864 (36.0) | 2 | 8/64 | by regs=4, by smem=2 |
| fwd_volta | TileVolta_d32 (128,32,32,4) | sm_75 | 128 | 0 | 0/0 | 0 | 36864 (36.0) | 1 | 4/32 | by regs=4, by smem=1 |
| fwd_volta | TileVolta_d64 (128,64,64,4) | sm_70 | 255 | 0 | 0/0 | 0 | 81920 (80.0) | 1 | 4/64 | by regs=2, by smem=1 |
| fwd_volta | TileVolta_d64 (128,64,64,4) | sm_75 | 255 | 0 | 0/0 | 0 | 81920 (80.0) | - | - | FAIL(smem>64KB) |
| fwd_volta | TileVolta_d96 (128,32,96,4) | sm_70 | 254 | 0 | 0/0 | 0 | 61440 (60.0) | 1 | 4/64 | by regs=2, by smem=1 |
| fwd_volta | TileVolta_d96 (128,32,96,4) | sm_75 | 254 | 0 | 0/0 | 0 | 61440 (60.0) | 1 | 4/32 | by regs=2, by smem=1 |
| fwd_volta | TileVolta_d128 (128,32,128,4) | sm_70 | 255 | 0 | 0/0 | 0 | 73728 (72.0) | 1 | 4/64 | by regs=2, by smem=1 |
| fwd_volta | TileVolta_d128 (128,32,128,4) | sm_75 | 255 | 0 | 0/0 | 0 | 73728 (72.0) | - | - | FAIL(smem>64KB) |
| fwd_volta | TileVolta_d256 (64,16,256,2) | sm_70 | 64 | 1768 | 5708/4800 | 0 | 55296 (54.0) | 1 | 2/64 | by regs=16, by smem=1 |
| fwd_volta | TileVolta_d256 (64,16,256,2) | sm_75 | 255 | 424 | 1592/1200 | 0 | 55296 (54.0) | 1 | 2/32 | by regs=4, by smem=1 |
| fwd_volta_varlen | TileVolta_d32 (128,32,32,4) | sm_70 | 128 | 0 | 0/0 | 0 | 36864 (36.0) | 2 | 8/64 | by regs=4, by smem=2 |
| fwd_volta_varlen | TileVolta_d32 (128,32,32,4) | sm_75 | 128 | 0 | 0/0 | 0 | 36864 (36.0) | 1 | 4/32 | by regs=4, by smem=1 |
| fwd_volta_varlen | TileVolta_d64 (128,64,64,4) | sm_70 | 255 | 0 | 0/0 | 0 | 81920 (80.0) | 1 | 4/64 | by regs=2, by smem=1 |
| fwd_volta_varlen | TileVolta_d64 (128,64,64,4) | sm_75 | 255 | 0 | 0/0 | 0 | 81920 (80.0) | - | - | FAIL(smem>64KB) |
| fwd_volta_varlen | TileVolta_d96 (128,32,96,4) | sm_70 | 254 | 0 | 0/0 | 0 | 61440 (60.0) | 1 | 4/64 | by regs=2, by smem=1 |
| fwd_volta_varlen | TileVolta_d96 (128,32,96,4) | sm_75 | 254 | 0 | 0/0 | 0 | 61440 (60.0) | 1 | 4/32 | by regs=2, by smem=1 |
| fwd_volta_varlen | TileVolta_d128 (128,32,128,4) | sm_70 | 255 | 0 | 0/0 | 0 | 73728 (72.0) | 1 | 4/64 | by regs=2, by smem=1 |
| fwd_volta_varlen | TileVolta_d128 (128,32,128,4) | sm_75 | 255 | 0 | 0/0 | 0 | 73728 (72.0) | - | - | FAIL(smem>64KB) |
| fwd_volta_varlen | TileVolta_d256 (64,16,256,2) | sm_70 | 64 | 1760 | 5692/4780 | 0 | 55296 (54.0) | 1 | 2/64 | by regs=16, by smem=1 |
| fwd_volta_varlen | TileVolta_d256 (64,16,256,2) | sm_75 | 255 | 416 | 1580/1188 | 0 | 55296 (54.0) | 1 | 2/32 | by regs=4, by smem=1 |
| fwd_pascal | TilePascal_d32 (128,32,32,4) | sm_60 | 119 | 0 | 0/0 | 0 | 28672 (28.0) | 2 | 8/64 | by regs=4, by smem=2 |
| fwd_pascal | TilePascal_d32 (128,32,32,4) | sm_61 | 125 | 0 | 0/0 | 0 | 28672 (28.0) | 3 | 12/64 | by regs=4, by smem=3 |
| fwd_pascal | TilePascal_d32 (128,32,32,4) | sm_70 | 127 | 0 | 0/0 | 0 | 28672 (28.0) | 3 | 12/64 | by regs=4, by smem=3 |
| fwd_pascal | TilePascal_d32 (128,32,32,4) | sm_75 | 127 | 0 | 0/0 | 0 | 28672 (28.0) | 2 | 8/32 | by regs=4, by smem=2 |
| fwd_pascal | TilePascal_d64 (128,32,64,4) | sm_60 | 150 | 0 | 0/0 | 0 | 40960 (40.0) | 1 | 4/64 | by regs=3, by smem=1 |
| fwd_pascal | TilePascal_d64 (128,32,64,4) | sm_61 | 159 | 0 | 0/0 | 0 | 40960 (40.0) | 2 | 8/64 | by regs=3, by smem=2 |
| fwd_pascal | TilePascal_d64 (128,32,64,4) | sm_70 | 168 | 0 | 0/0 | 0 | 40960 (40.0) | 2 | 8/64 | by regs=3, by smem=2 |
| fwd_pascal | TilePascal_d64 (128,32,64,4) | sm_75 | 168 | 0 | 0/0 | 0 | 40960 (40.0) | 1 | 4/32 | by regs=3, by smem=1 |
| fwd_pascal | TilePascal_d96 (64,16,96,2) | sm_60 | 168 | 0 | 0/0 | 0 | 22528 (22.0) | 2 | 4/64 | by regs=6, by smem=2 |
| fwd_pascal | TilePascal_d96 (64,16,96,2) | sm_61 | 168 | 0 | 0/0 | 0 | 22528 (22.0) | 4 | 8/64 | by regs=6, by smem=4 |
| fwd_pascal | TilePascal_d96 (64,16,96,2) | sm_70 | 168 | 0 | 0/0 | 0 | 22528 (22.0) | 4 | 8/64 | by regs=6, by smem=4 |
| fwd_pascal | TilePascal_d96 (64,16,96,2) | sm_75 | 168 | 0 | 0/0 | 0 | 22528 (22.0) | 2 | 4/32 | by regs=6, by smem=2 |
| fwd_pascal | TilePascal_d128 (64,16,128,2) | sm_60 | 231 | 0 | 0/0 | 0 | 28672 (28.0) | 2 | 4/64 | by regs=4, by smem=2 |
| fwd_pascal | TilePascal_d128 (64,16,128,2) | sm_61 | 235 | 0 | 0/0 | 0 | 28672 (28.0) | 3 | 6/64 | by regs=4, by smem=3 |
| fwd_pascal | TilePascal_d128 (64,16,128,2) | sm_70 | 238 | 0 | 0/0 | 0 | 28672 (28.0) | 3 | 6/64 | by regs=4, by smem=3 |
| fwd_pascal | TilePascal_d128 (64,16,128,2) | sm_75 | 238 | 0 | 0/0 | 0 | 28672 (28.0) | 2 | 4/32 | by regs=4, by smem=2 |
| fwd_pascal | TilePascal_d256 (32,16,256,1) | sm_60 | 255 | 256 | 2220/1476 | 0 | 34816 (34.0) | 1 | 1/64 | by regs=8, by smem=1 |
| fwd_pascal | TilePascal_d256 (32,16,256,1) | sm_61 | 255 | 256 | 2220/1476 | 0 | 34816 (34.0) | 2 | 2/64 | by regs=8, by smem=2 |
| fwd_pascal | TilePascal_d256 (32,16,256,1) | sm_70 | 255 | 360 | 2612/1860 | 0 | 34816 (34.0) | 2 | 2/64 | by regs=8, by smem=2 |
| fwd_pascal | TilePascal_d256 (32,16,256,1) | sm_75 | 255 | 344 | 2444/1816 | 0 | 34816 (34.0) | 1 | 1/32 | by regs=8, by smem=1 |
| fwd_pascal_varlen | TilePascal_d32 (128,32,32,4) | sm_60 | 120 | 0 | 0/0 | 0 | 28672 (28.0) | 2 | 8/64 | by regs=4, by smem=2 |
| fwd_pascal_varlen | TilePascal_d32 (128,32,32,4) | sm_61 | 128 | 0 | 0/0 | 0 | 28672 (28.0) | 3 | 12/64 | by regs=4, by smem=3 |
| fwd_pascal_varlen | TilePascal_d32 (128,32,32,4) | sm_70 | 127 | 0 | 0/0 | 0 | 28672 (28.0) | 3 | 12/64 | by regs=4, by smem=3 |
| fwd_pascal_varlen | TilePascal_d32 (128,32,32,4) | sm_75 | 96 | 0 | 0/0 | 0 | 28672 (28.0) | 2 | 8/32 | by regs=5, by smem=2 |
| fwd_pascal_varlen | TilePascal_d64 (128,32,64,4) | sm_60 | 150 | 0 | 0/0 | 0 | 40960 (40.0) | 1 | 4/64 | by regs=3, by smem=1 |
| fwd_pascal_varlen | TilePascal_d64 (128,32,64,4) | sm_61 | 163 | 0 | 0/0 | 0 | 40960 (40.0) | 2 | 8/64 | by regs=3, by smem=2 |
| fwd_pascal_varlen | TilePascal_d64 (128,32,64,4) | sm_70 | 168 | 0 | 0/0 | 0 | 40960 (40.0) | 2 | 8/64 | by regs=3, by smem=2 |
| fwd_pascal_varlen | TilePascal_d64 (128,32,64,4) | sm_75 | 168 | 0 | 0/0 | 0 | 40960 (40.0) | 1 | 4/32 | by regs=3, by smem=1 |
| fwd_pascal_varlen | TilePascal_d96 (64,16,96,2) | sm_60 | 167 | 0 | 0/0 | 0 | 22528 (22.0) | 2 | 4/64 | by regs=6, by smem=2 |
| fwd_pascal_varlen | TilePascal_d96 (64,16,96,2) | sm_61 | 167 | 0 | 0/0 | 0 | 22528 (22.0) | 4 | 8/64 | by regs=6, by smem=4 |
| fwd_pascal_varlen | TilePascal_d96 (64,16,96,2) | sm_70 | 168 | 0 | 0/0 | 0 | 22528 (22.0) | 4 | 8/64 | by regs=6, by smem=4 |
| fwd_pascal_varlen | TilePascal_d96 (64,16,96,2) | sm_75 | 168 | 0 | 0/0 | 0 | 22528 (22.0) | 2 | 4/32 | by regs=6, by smem=2 |
| fwd_pascal_varlen | TilePascal_d128 (64,16,128,2) | sm_60 | 234 | 0 | 0/0 | 0 | 28672 (28.0) | 2 | 4/64 | by regs=4, by smem=2 |
| fwd_pascal_varlen | TilePascal_d128 (64,16,128,2) | sm_61 | 242 | 0 | 0/0 | 0 | 28672 (28.0) | 3 | 6/64 | by regs=4, by smem=3 |
| fwd_pascal_varlen | TilePascal_d128 (64,16,128,2) | sm_70 | 239 | 0 | 0/0 | 0 | 28672 (28.0) | 3 | 6/64 | by regs=4, by smem=3 |
| fwd_pascal_varlen | TilePascal_d128 (64,16,128,2) | sm_75 | 254 | 0 | 0/0 | 0 | 28672 (28.0) | 2 | 4/32 | by regs=4, by smem=2 |
| fwd_pascal_varlen | TilePascal_d256 (32,16,256,1) | sm_60 | 255 | 256 | 2228/1564 | 0 | 34816 (34.0) | 1 | 1/64 | by regs=8, by smem=1 |
| fwd_pascal_varlen | TilePascal_d256 (32,16,256,1) | sm_61 | 255 | 256 | 2228/1564 | 0 | 34816 (34.0) | 2 | 2/64 | by regs=8, by smem=2 |
| fwd_pascal_varlen | TilePascal_d256 (32,16,256,1) | sm_70 | 255 | 352 | 2604/1924 | 0 | 34816 (34.0) | 2 | 2/64 | by regs=8, by smem=2 |
| fwd_pascal_varlen | TilePascal_d256 (32,16,256,1) | sm_75 | 255 | 336 | 2432/1820 | 0 | 34816 (34.0) | 1 | 1/32 | by regs=8, by smem=1 |
| bwd_volta | TileBwd_d32 (32,32,32,4) | sm_70 | 128 | 0 | 0/0 | 0 | 26624 (26.0) | 3 | 12/64 | by regs=4, by smem=3 |
| bwd_volta | TileBwd_d32 (32,32,32,4) | sm_75 | 79 | 0 | 0/0 | 0 | 26624 (26.0) | 2 | 8/32 | by regs=6, by smem=2 |
| bwd_volta | TileBwd_d64 (32,32,64,4) | sm_70 | 219 | 0 | 0/0 | 0 | 43008 (42.0) | 2 | 8/64 | by regs=2, by smem=2 |
| bwd_volta | TileBwd_d64 (32,32,64,4) | sm_75 | 162 | 0 | 0/0 | 0 | 43008 (42.0) | 1 | 4/32 | by regs=3, by smem=1 |
| bwd_volta | TileBwd_d96 (32,16,96,4) | sm_70 | 168 | 8 | 4/4 | 0 | 35840 (35.0) | 2 | 8/64 | by regs=3, by smem=2 |
| bwd_volta | TileBwd_d96 (32,16,96,4) | sm_75 | 150 | 0 | 0/0 | 0 | 35840 (35.0) | 1 | 4/32 | by regs=3, by smem=1 |
| bwd_volta | TileBwd_d128 (16,16,128,4) | sm_70 | 168 | 8 | 8/12 | 0 | 35328 (34.5) | 2 | 8/64 | by regs=3, by smem=2 |
| bwd_volta | TileBwd_d128 (16,16,128,4) | sm_75 | 128 | 16 | 12/12 | 0 | 35328 (34.5) | 1 | 4/32 | by regs=4, by smem=1 |
| bwd_volta | TileBwd_d256 (16,16,256,4) | sm_70 | 145 | 0 | 0/0 | 0 | 68096 (66.5) | 1 | 4/64 | by regs=3, by smem=1 |
| bwd_volta | TileBwd_d256 (16,16,256,4) | sm_75 | 145 | 0 | 0/0 | 0 | 68096 (66.5) | - | - | FAIL(smem>64KB) |
| bwd_cudacore | TileBwd_d32 (32,32,32,4) | sm_60 | 197 | 0 | 0/0 | 0 | 8192 (8.0) | 2 | 8/64 | by regs=2, by smem=8 |
| bwd_cudacore | TileBwd_d32 (32,32,32,4) | sm_61 | 206 | 0 | 0/0 | 0 | 8192 (8.0) | 2 | 8/64 | by regs=2, by smem=12 |
| bwd_cudacore | TileBwd_d32 (32,32,32,4) | sm_70 | 204 | 0 | 0/0 | 0 | 8192 (8.0) | 2 | 8/64 | by regs=2, by smem=12 |
| bwd_cudacore | TileBwd_d32 (32,32,32,4) | sm_75 | 204 | 0 | 0/0 | 0 | 8192 (8.0) | 2 | 8/32 | by regs=2, by smem=8 |
| bwd_cudacore | TileBwd_d64 (32,32,64,4) | sm_60 | 255 | 184 | 324/384 | 0 | 16384 (16.0) | 2 | 8/64 | by regs=2, by smem=4 |
| bwd_cudacore | TileBwd_d64 (32,32,64,4) | sm_61 | 255 | 184 | 332/392 | 0 | 16384 (16.0) | 2 | 8/64 | by regs=2, by smem=6 |
| bwd_cudacore | TileBwd_d64 (32,32,64,4) | sm_70 | 255 | 192 | 328/532 | 0 | 16384 (16.0) | 2 | 8/64 | by regs=2, by smem=6 |
| bwd_cudacore | TileBwd_d64 (32,32,64,4) | sm_75 | 255 | 144 | 260/288 | 0 | 16384 (16.0) | 2 | 8/32 | by regs=2, by smem=4 |
| bwd_cudacore | TileBwd_d96 (32,16,96,4) | sm_60 | 56 | 2232 | 5456/7296 | 0 | 18432 (18.0) | 3 | 12/64 | by regs=9, by smem=3 |
| bwd_cudacore | TileBwd_d96 (32,16,96,4) | sm_61 | 40 | 2472 | 6048/8452 | 0 | 18432 (18.0) | 5 | 20/64 | by regs=12, by smem=5 |
| bwd_cudacore | TileBwd_d96 (32,16,96,4) | sm_70 | 56 | 2024 | 5816/6260 | 0 | 18432 (18.0) | 5 | 20/64 | by regs=9, by smem=5 |
| bwd_cudacore | TileBwd_d96 (32,16,96,4) | sm_75 | 64 | 1888 | 5488/5748 | 0 | 18432 (18.0) | 3 | 12/32 | by regs=8, by smem=3 |
| bwd_cudacore | TileBwd_d128 (16,16,128,4) | sm_60 | 32 | 2368 | 6868/7092 | 0 | 16384 (16.0) | 4 | 16/64 | by regs=16, by smem=4 |
| bwd_cudacore | TileBwd_d128 (16,16,128,4) | sm_61 | 40 | 2248 | 6492/6716 | 0 | 16384 (16.0) | 6 | 24/64 | by regs=12, by smem=6 |
| bwd_cudacore | TileBwd_d128 (16,16,128,4) | sm_70 | 56 | 2384 | 7244/7516 | 0 | 16384 (16.0) | 6 | 24/64 | by regs=9, by smem=6 |
| bwd_cudacore | TileBwd_d128 (16,16,128,4) | sm_75 | 64 | 2312 | 7052/7076 | 0 | 16384 (16.0) | 4 | 16/32 | by regs=8, by smem=4 |
| bwd_cudacore | TileBwd_d256 (16,16,256,4) | sm_60 | 32 | 4696 | 13892/13980 | 0 | 32768 (32.0) | 2 | 8/64 | by regs=16, by smem=2 |
| bwd_cudacore | TileBwd_d256 (16,16,256,4) | sm_61 | 48 | 4360 | 13052/13140 | 0 | 32768 (32.0) | 3 | 12/64 | by regs=10, by smem=3 |
| bwd_cudacore | TileBwd_d256 (16,16,256,4) | sm_70 | 56 | 4608 | 15096/15204 | 0 | 32768 (32.0) | 3 | 12/64 | by regs=9, by smem=3 |
| bwd_cudacore | TileBwd_d256 (16,16,256,4) | sm_75 | 64 | 4456 | 14760/14824 | 0 | 32768 (32.0) | 2 | 8/32 | by regs=8, by smem=2 |
| precompute_D | - | sm_60 | 31 | 0 | 0/0 | 0 | 0 (0.0) | 8 | 64/64 | by regs=8, by smem=inf |
| precompute_D | - | sm_61 | 32 | 0 | 0/0 | 0 | 0 (0.0) | 8 | 64/64 | by regs=8, by smem=inf |
| precompute_D | - | sm_70 | 32 | 0 | 0/0 | 0 | 0 (0.0) | 8 | 64/64 | by regs=8, by smem=inf |
| precompute_D | - | sm_75 | 57 | 0 | 0/0 | 0 | 0 (0.0) | 4 | 32/32 | by regs=4, by smem=inf |
