#!/bin/bash
# Compile each kernel TU per arch with ptxas -v (CPU only, no GPU use)
source /workspace/env.sh >/dev/null 2>&1
cd /workspace/analysis/kernels
SRC=src/csrc
mkdir -p cubin raw
FLAGS="-D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -O3 --use_fast_math --expt-relaxed-constexpr --expt-extended-lambda -std=c++17 -Xptxas -dlcm=ca -I$SRC -I$SRC/include"
: > raw/compile_times.txt
for f in flash_attn_fwd_volta flash_attn_fwd_pascal flash_attn_bwd; do
  for a in 60 61 70 75; do
    s=$(date +%s.%N)
    nvcc $FLAGS -gencode arch=compute_$a,code=sm_$a -cubin -Xptxas -v $SRC/kernels/$f.cu -o cubin/${f}_sm$a.cubin > raw/ptxas_${f}_sm$a.txt 2>&1
    e=$(date +%s.%N)
    echo "$f sm_$a $(echo "$e - $s" | bc) s rc=$?" >> raw/compile_times.txt
  done
done
cat raw/compile_times.txt
