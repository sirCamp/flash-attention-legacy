#!/bin/bash
# Compile each kernel TU per arch with ptxas -v (CPU only, no GPU use).
# usage: build_ptxas.sh <repo> <out dir>   (writes <out dir>/raw/ptxas_<TU>_sm<arch>.txt)
# Flags = setup.py + torch's -D__CUDA_NO_HALF* defines. Then: occupancy.py <repo> <out dir>.
set -e
REPO=$(cd "$1" && pwd); OUT=$2
SRC=$REPO/csrc
mkdir -p "$OUT/cubin" "$OUT/raw"
FLAGS="-D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -O3 --use_fast_math --expt-relaxed-constexpr --expt-extended-lambda -std=c++17 -Xptxas -dlcm=ca -I$SRC -I$SRC/include"
for f in flash_attn_fwd_volta flash_attn_fwd_turing flash_attn_fwd_pascal flash_attn_bwd flash_attn_decode; do
  for a in 60 61 70 75; do
    nvcc $FLAGS -gencode arch=compute_$a,code=sm_$a -cubin -Xptxas -v "$SRC/kernels/$f.cu" \
      -o "$OUT/cubin/${f}_sm$a.cubin" > "$OUT/raw/ptxas_${f}_sm$a.txt" 2>&1
  done
done
