# FP16 linear layers on the P100: `fused_linear` / `patch_linear`

PLAN P8. Every FP16 `nn.Linear` on a compute capability 6.0 GPU (Tesla P100) can run on a fused GEMM kernel
(`csrc/kernels/linear_pascal.cu`) instead of torch's cuBLAS call. **Off by default**: `flash_attn_legacy.patch_linear(model)`,
or `FLASH_ATTN_LEGACY_LINEAR=1` for models loaded with `attn_implementation="flash_attn_legacy"`; a no-op on every other GPU.

Why: torch runs FP16 GEMMs on sm_60 with cuBLAS `COMPUTE_32F` (4.6-8.7 TFLOPS on the shapes below, under the card's FP32 rate), while its
HFMA2 peak is ~19 TFLOPS. The kernel multiplies with HFMA2 (`half2` pairs along K) and folds each lane into FP32 every 64
K-elements (chunked accumulation, IBM arXiv 1812.08011; DeepSeek-V3's FP32 promotion every 128; the interval is chosen in
[Fold interval](#fold-interval)), so the error does not
grow with K as in a pure FP16 accumulation (cuBLAS `COMPUTE_16F`, llama.cpp's P100 path). Calls with fewer than 128 rows
(decode) go to `F.linear` unchanged.

Measured on mew (Tesla P100-PCIE-16GB, passive cooling, GPU cooled below 50-60 C before each run), torch 2.7.1+cu126,
transformers 5.20.2, CUDA 12.8 build, 2026-11-02 (decode alone: 2026-10-31, a path that does not use the kernel).

## GEMM shapes

`python benchmarks/linear.py --ms 128 512 2048 8192`: y = x W^T on the layer shapes of four models (qkv and gate+up fused
as one GEMM), 5 interleaved rounds, median. Error = relative Frobenius error against an FP32 product of the same FP16
inputs (first 2048 rows).

| model | layer | M | N | K | stock TFLOPS | fused TFLOPS | speedup | error stock | error fused |
|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b | qkv | 128 | 1152 | 896 | 4.75 | 5.16 | 1.09x | 2.1e-04 | 8.2e-04 |
| qwen2.5-0.5b | qkv | 512 | 1152 | 896 | 5.81 | 9.58 | 1.65x | 2.1e-04 | 8.7e-04 |
| qwen2.5-0.5b | qkv | 2048 | 1152 | 896 | 6.99 | 12.28 | 1.76x | 2.1e-04 | 8.7e-04 |
| qwen2.5-0.5b | qkv | 8192 | 1152 | 896 | 7.94 | 12.99 | 1.64x | 2.1e-04 | 8.7e-04 |
| qwen2.5-0.5b | o | 128 | 896 | 896 | 5.46 | 5.03 | 0.92x | 2.1e-04 | 8.7e-04 |
| qwen2.5-0.5b | o | 512 | 896 | 896 | 6.25 | 10.24 | 1.64x | 2.1e-04 | 8.7e-04 |
| qwen2.5-0.5b | o | 2048 | 896 | 896 | 7.00 | 13.05 | 1.86x | 2.1e-04 | 8.7e-04 |
| qwen2.5-0.5b | o | 8192 | 896 | 896 | 8.55 | 13.17 | 1.54x | 2.1e-04 | 8.7e-04 |
| qwen2.5-0.5b | gateup | 128 | 9728 | 896 | 6.87 | 11.95 | 1.74x | 2.1e-04 | 8.7e-04 |
| qwen2.5-0.5b | gateup | 512 | 9728 | 896 | 7.12 | 12.79 | 1.80x | 2.1e-04 | 8.7e-04 |
| qwen2.5-0.5b | gateup | 2048 | 9728 | 896 | 8.37 | 13.07 | 1.56x | 2.1e-04 | 8.7e-04 |
| qwen2.5-0.5b | gateup | 8192 | 9728 | 896 | 7.45 | 13.22 | 1.77x | 2.1e-04 | 8.7e-04 |
| qwen2.5-0.5b | down | 128 | 896 | 4864 | 6.51 | 10.24 | 1.57x | 2.1e-04 | 8.6e-04 |
| qwen2.5-0.5b | down | 512 | 896 | 4864 | 6.72 | 12.91 | 1.92x | 2.1e-04 | 8.7e-04 |
| qwen2.5-0.5b | down | 2048 | 896 | 4864 | 7.45 | 13.52 | 1.82x | 2.1e-04 | 8.7e-04 |
| qwen2.5-0.5b | down | 8192 | 896 | 4864 | 8.73 | 13.53 | 1.55x | 2.1e-04 | 8.7e-04 |
| qwen2.5-0.5b | lm_head | 128 | 151936 | 896 | 8.08 | 13.04 | 1.61x | 2.1e-04 | 8.7e-04 |
| qwen2.5-0.5b | lm_head | 512 | 151936 | 896 | 7.44 | 13.18 | 1.77x | 2.1e-04 | 8.7e-04 |
| qwen2.5-0.5b | lm_head | 2048 | 151936 | 896 | 7.51 | 13.37 | 1.78x | 2.1e-04 | 8.7e-04 |
| qwen2.5-0.5b | lm_head | 8192 | 151936 | 896 | 7.52 | 13.39 | 1.78x | 2.1e-04 | 8.7e-04 |
| qwen2.5-1.5b | qkv | 128 | 2048 | 1536 | 6.00 | 9.67 | 1.61x | 2.1e-04 | 8.5e-04 |
| qwen2.5-1.5b | qkv | 512 | 2048 | 1536 | 6.94 | 12.61 | 1.82x | 2.1e-04 | 8.6e-04 |
| qwen2.5-1.5b | qkv | 2048 | 2048 | 1536 | 7.31 | 13.10 | 1.79x | 2.1e-04 | 8.7e-04 |
| qwen2.5-1.5b | qkv | 8192 | 2048 | 1536 | 8.26 | 13.30 | 1.61x | 2.1e-04 | 8.7e-04 |
| qwen2.5-1.5b | o | 128 | 1536 | 1536 | 5.22 | 8.32 | 1.59x | 2.1e-04 | 8.5e-04 |
| qwen2.5-1.5b | o | 512 | 1536 | 1536 | 6.93 | 12.35 | 1.78x | 2.1e-04 | 8.6e-04 |
| qwen2.5-1.5b | o | 2048 | 1536 | 1536 | 7.25 | 13.04 | 1.80x | 2.1e-04 | 8.7e-04 |
| qwen2.5-1.5b | o | 8192 | 1536 | 1536 | 8.48 | 13.29 | 1.57x | 2.1e-04 | 8.7e-04 |
| qwen2.5-1.5b | gateup | 128 | 17920 | 1536 | 7.19 | 12.98 | 1.81x | 2.1e-04 | 8.7e-04 |
| qwen2.5-1.5b | gateup | 512 | 17920 | 1536 | 8.53 | 13.33 | 1.56x | 2.1e-04 | 8.7e-04 |
| qwen2.5-1.5b | gateup | 2048 | 17920 | 1536 | 7.49 | 13.24 | 1.77x | 2.1e-04 | 8.7e-04 |
| qwen2.5-1.5b | gateup | 8192 | 17920 | 1536 | 7.60 | 13.53 | 1.78x | 2.1e-04 | 8.7e-04 |
| qwen2.5-1.5b | down | 128 | 1536 | 8960 | 5.28 | 12.38 | 2.34x | 2.1e-04 | 8.7e-04 |
| qwen2.5-1.5b | down | 512 | 1536 | 8960 | 4.92 | 13.32 | 2.70x | 2.1e-04 | 8.7e-04 |
| qwen2.5-1.5b | down | 2048 | 1536 | 8960 | 5.02 | 13.48 | 2.69x | 2.1e-04 | 8.7e-04 |
| qwen2.5-1.5b | down | 8192 | 1536 | 8960 | 8.58 | 13.58 | 1.58x | 2.1e-04 | 8.7e-04 |
| qwen2.5-1.5b | lm_head | 128 | 151936 | 1536 | 8.13 | 13.27 | 1.63x | 2.1e-04 | 8.7e-04 |
| qwen2.5-1.5b | lm_head | 512 | 151936 | 1536 | 7.56 | 13.39 | 1.77x | 2.1e-04 | 8.7e-04 |
| qwen2.5-1.5b | lm_head | 2048 | 151936 | 1536 | 7.61 | 13.55 | 1.78x | 2.1e-04 | 8.7e-04 |
| qwen2.5-1.5b | lm_head | 8192 | 151936 | 1536 | 7.60 | 13.57 | 1.78x | 2.1e-04 | 8.7e-04 |
| qwen2.5-3b | qkv | 128 | 2560 | 2048 | 6.75 | 10.97 | 1.62x | 2.1e-04 | 8.6e-04 |
| qwen2.5-3b | qkv | 512 | 2560 | 2048 | 7.21 | 12.79 | 1.77x | 2.1e-04 | 8.7e-04 |
| qwen2.5-3b | qkv | 2048 | 2560 | 2048 | 7.42 | 13.25 | 1.79x | 2.1e-04 | 8.7e-04 |
| qwen2.5-3b | qkv | 8192 | 2560 | 2048 | 8.57 | 13.38 | 1.56x | 2.1e-04 | 8.7e-04 |
| qwen2.5-3b | o | 128 | 2048 | 2048 | 6.04 | 10.50 | 1.74x | 2.1e-04 | 8.5e-04 |
| qwen2.5-3b | o | 512 | 2048 | 2048 | 7.07 | 12.91 | 1.83x | 2.1e-04 | 8.6e-04 |
| qwen2.5-3b | o | 2048 | 2048 | 2048 | 7.40 | 13.19 | 1.78x | 2.1e-04 | 8.7e-04 |
| qwen2.5-3b | o | 8192 | 2048 | 2048 | 8.30 | 13.35 | 1.61x | 2.1e-04 | 8.7e-04 |
| qwen2.5-3b | gateup | 128 | 22016 | 2048 | 7.33 | 13.13 | 1.79x | 2.1e-04 | 8.7e-04 |
| qwen2.5-3b | gateup | 512 | 22016 | 2048 | 8.05 | 13.33 | 1.66x | 2.1e-04 | 8.7e-04 |
| qwen2.5-3b | gateup | 2048 | 22016 | 2048 | 7.59 | 13.40 | 1.77x | 2.1e-04 | 8.7e-04 |
| qwen2.5-3b | gateup | 8192 | 22016 | 2048 | 7.63 | 13.58 | 1.78x | 2.1e-04 | 8.7e-04 |
| qwen2.5-3b | down | 128 | 2048 | 11008 | 6.01 | 12.80 | 2.13x | 2.1e-04 | 8.6e-04 |
| qwen2.5-3b | down | 512 | 2048 | 11008 | 4.85 | 13.40 | 2.76x | 2.1e-04 | 8.7e-04 |
| qwen2.5-3b | down | 2048 | 2048 | 11008 | 4.98 | 13.50 | 2.71x | 2.1e-04 | 8.7e-04 |
| qwen2.5-3b | down | 8192 | 2048 | 11008 | 8.34 | 13.59 | 1.63x | 2.1e-04 | 8.7e-04 |
| qwen2.5-3b | lm_head | 128 | 151936 | 2048 | 8.20 | 13.34 | 1.63x | 2.1e-04 | 8.7e-04 |
| qwen2.5-3b | lm_head | 512 | 151936 | 2048 | 7.61 | 13.52 | 1.78x | 2.1e-04 | 8.7e-04 |
| qwen2.5-3b | lm_head | 2048 | 151936 | 2048 | 7.61 | 13.60 | 1.79x | 2.1e-04 | 8.7e-04 |
| qwen2.5-3b | lm_head | 8192 | 151936 | 2048 | 7.56 | 13.62 | 1.80x | 2.1e-04 | 8.7e-04 |
| llama3.1-8b | qkv | 128 | 6144 | 4096 | 6.26 | 11.73 | 1.88x | 2.1e-04 | 8.7e-04 |
| llama3.1-8b | qkv | 512 | 6144 | 4096 | 7.32 | 13.17 | 1.80x | 2.1e-04 | 8.7e-04 |
| llama3.1-8b | qkv | 2048 | 6144 | 4096 | 8.48 | 13.48 | 1.59x | 2.1e-04 | 8.7e-04 |
| llama3.1-8b | qkv | 8192 | 6144 | 4096 | 7.63 | 13.40 | 1.76x | 2.1e-04 | 8.7e-04 |
| llama3.1-8b | o | 128 | 4096 | 4096 | 6.39 | 12.79 | 2.00x | 2.1e-04 | 8.7e-04 |
| llama3.1-8b | o | 512 | 4096 | 4096 | 7.34 | 13.26 | 1.81x | 2.1e-04 | 8.7e-04 |
| llama3.1-8b | o | 2048 | 4096 | 4096 | 7.94 | 13.45 | 1.69x | 2.1e-04 | 8.7e-04 |
| llama3.1-8b | o | 8192 | 4096 | 4096 | 7.61 | 13.49 | 1.77x | 2.1e-04 | 8.7e-04 |
| llama3.1-8b | gateup | 128 | 28672 | 4096 | 7.45 | 13.45 | 1.81x | 2.1e-04 | 8.7e-04 |
| llama3.1-8b | gateup | 512 | 28672 | 4096 | 8.64 | 13.54 | 1.57x | 2.1e-04 | 8.7e-04 |
| llama3.1-8b | gateup | 2048 | 28672 | 4096 | 7.58 | 13.43 | 1.77x | 2.1e-04 | 8.7e-04 |
| llama3.1-8b | gateup | 8192 | 28672 | 4096 | 7.60 | 13.62 | 1.79x | 2.1e-04 | 8.7e-04 |
| llama3.1-8b | down | 128 | 4096 | 14336 | 4.61 | 13.30 | 2.88x | 2.1e-04 | 8.7e-04 |
| llama3.1-8b | down | 512 | 4096 | 14336 | 4.99 | 13.49 | 2.70x | 2.1e-04 | 8.7e-04 |
| llama3.1-8b | down | 2048 | 4096 | 14336 | 7.95 | 13.61 | 1.71x | 2.1e-04 | 8.7e-04 |
| llama3.1-8b | down | 8192 | 4096 | 14336 | 7.17 | 13.38 | 1.87x | 2.1e-04 | 8.7e-04 |
| llama3.1-8b | lm_head | 128 | 128256 | 4096 | 8.50 | 13.50 | 1.59x | 2.1e-04 | 8.7e-04 |
| llama3.1-8b | lm_head | 512 | 128256 | 4096 | 7.57 | 13.44 | 1.78x | 2.1e-04 | 8.7e-04 |
| llama3.1-8b | lm_head | 2048 | 128256 | 4096 | 7.57 | 13.63 | 1.80x | 2.1e-04 | 8.7e-04 |
| llama3.1-8b | lm_head | 8192 | 128256 | 4096 | 7.56 | 13.70 | 1.81x | 2.1e-04 | 8.7e-04 |

Geomean 1.77x over the 80 shapes, 1.54-2.76x from M = 512. From M = 512 on the kernel reaches 12.3-13.7 TFLOPS (64-72%
of the HFMA2 peak), except the two smallest shapes at M = 512 (Qwen2.5-0.5B qkv / o: 9.6 / 10.2); stock 4.6-8.7. Its
element-wise error is about 4x stock's (8.7e-4 vs 2.1e-4: products and each lane's chain of up to 32 terms per chunk in FP16), and
does not depend on K. The one case below 1 is the smallest shape (M = 128, N = K = 896: 7 output tiles for 56 SMs,
~40 us): 0.92x. The package's build (at C = 128) vs the standalone prototype it was taken from: bitwise equal outputs on
45 shapes, 0.996x its speed.

## End to end

`python benchmarks/models.py --models <id> --impls flash_attn_legacy --modes prefill generate train --prefill-lens 1024
4096 8192 --prompt-lens 2048 --gen-batch 1 8 --new-tokens 64 --train-lens 1024 2048 4096 --lora 16 --checkpointing`,
without and with `--linear`, alternated A B A B (2 rounds; mean, and the spread between the rounds). Prefill and
LoRA train in tokens/s, decode in tokens/s after a 2048-token prompt. LoRA train: rank-16 adapters (fp32) on the attention
projections, the frozen fp16 base layers on the kernel, gradient checkpointing, fp16 autocast. Patched layers: TinyLlama
155, Qwen2.5-1.5B 197, Qwen3.5-2B 187, Qwen3.5-4B 249, Llama-3.1-8B 225. Cases out of memory on 16 GB with both variants
are left out (Llama-3.1-8B beyond a 1k prefill, Qwen3.5-4B LoRA beyond 1k, Qwen3.5-2B LoRA at 4k).

| model | mode | B | N | stock | fused linear | speedup | spread stock / fused | loss stock / fused |
|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | prefill | 1 | 1024 | 3,256 | 5,188 | 1.59x | 0.0% / 0.1% | - |
| TinyLlama-1.1B-Chat-v1.0 | prefill | 1 | 4096 | 3,082 | 4,330 | 1.41x | 0.0% / 0.0% | - |
| TinyLlama-1.1B-Chat-v1.0 | prefill | 1 | 8192 | 2,536 | 3,414 | 1.35x | 0.1% / 0.0% | - |
| TinyLlama-1.1B-Chat-v1.0 | decode | 1 | 2048 | 79 | 78 | 0.99x | 0.9% / 0.0% | - |
| TinyLlama-1.1B-Chat-v1.0 | decode | 8 | 2048 | 290 | 282 | 0.97x | 0.1% / 0.0% | - |
| TinyLlama-1.1B-Chat-v1.0 | LoRA train | 1 | 1024 | 960 | 1,246 | 1.30x | 0.1% / 0.1% | 12.2554 / 12.2553 |
| TinyLlama-1.1B-Chat-v1.0 | LoRA train | 1 | 2048 | 912 | 1,161 | 1.27x | 0.0% / 0.0% | 12.1925 / 12.1918 |
| TinyLlama-1.1B-Chat-v1.0 | LoRA train | 1 | 4096 | 807 | 982 | 1.22x | 0.0% / 0.0% | 11.9632 / 11.9625 |
| Qwen2.5-1.5B | prefill | 1 | 1024 | 2,226 | 3,861 | 1.73x | 0.0% / 0.0% | - |
| Qwen2.5-1.5B | prefill | 1 | 4096 | 1,896 | 3,188 | 1.68x | 0.1% / 0.0% | - |
| Qwen2.5-1.5B | prefill | 1 | 8192 | 1,860 | 2,518 | 1.35x | 0.0% / 0.0% | - |
| Qwen2.5-1.5B | decode | 1 | 2048 | 57 | 57 | 1.00x | 0.4% / 0.5% | - |
| Qwen2.5-1.5B | decode | 8 | 2048 | 190 | 186 | 0.98x | 0.0% / 0.0% | - |
| Qwen2.5-1.5B | LoRA train | 1 | 1024 | 705 | 896 | 1.27x | 0.0% / 0.1% | 12.2771 / 12.2774 |
| Qwen2.5-1.5B | LoRA train | 1 | 2048 | 684 | 889 | 1.30x | 0.0% / 0.0% | 12.2105 / 12.2106 |
| Qwen2.5-1.5B | LoRA train | 1 | 4096 | 572 | 761 | 1.33x | 0.0% / 0.0% | 12.1451 / 12.1454 |
| Qwen3.5-2B | prefill | 1 | 1024 | 2,034 | 3,033 | 1.49x | 0.0% / 0.0% | - |
| Qwen3.5-2B | prefill | 1 | 4096 | 2,134 | 2,928 | 1.37x | 0.0% / 0.0% | - |
| Qwen3.5-2B | prefill | 1 | 8192 | 1,912 | 2,657 | 1.39x | 0.0% / 0.0% | - |
| Qwen3.5-2B | decode | 1 | 2048 | 40 | 39 | 1.00x | 1.3% / 0.3% | - |
| Qwen3.5-2B | decode | 8 | 2048 | 173 | 169 | 0.98x | 0.5% / 0.0% | - |
| Qwen3.5-2B | LoRA train | 1 | 1024 | 477 | 596 | 1.25x | 0.0% / 0.0% | 12.3397 / 12.3395 |
| Qwen3.5-2B | LoRA train | 1 | 2048 | 448 | 544 | 1.22x | 0.2% / 0.2% | 12.2297 / 12.2292 |
| Qwen3.5-4B | prefill | 1 | 1024 | 766 | 1,272 | 1.66x | 0.0% / 0.1% | - |
| Qwen3.5-4B | prefill | 1 | 4096 | 819 | 1,191 | 1.45x | 0.0% / 0.0% | - |
| Qwen3.5-4B | prefill | 1 | 8192 | 752 | 1,070 | 1.42x | 0.1% / 0.0% | - |
| Qwen3.5-4B | decode | 1 | 2048 | 25 | 25 | 1.00x | 0.0% / 0.4% | - |
| Qwen3.5-4B | decode | 8 | 2048 | 74 | 71 | 0.97x | 0.1% / 0.0% | - |
| Qwen3.5-4B | LoRA train | 1 | 1024 | 210 | 266 | 1.27x | 0.0% / 0.0% | 12.0765 / 12.0762 |
| Llama-3.1-8B-Instruct | prefill | 1 | 1024 | 467 | 848 | 1.82x | 0.0% / 0.0% | - |
| Llama-3.1-8B-Instruct | decode | 1 | 2048 | 20 | 20 | 1.00x | 0.0% / 0.0% | - |

Prefill 1.35-1.82x (geomean 1.51x), LoRA train 1.22-1.33x (1.27x; the losses agree to 3-4 digits). Decode does not use
the kernel: in `models.py` its rate is the generate time minus a separately timed prefill, which the kernel shortens, so
the 0.97-1.00x above mixes the two. Timed alone (`python benchmarks/linear.py --decode <id>`: a cache filled by a 2048-token
prefill, 64 one-token steps, 5 alternated rounds, median ms per step):

| model | B | stock | fused | speed |
|---|---|---|---|---|
| TinyLlama-1.1B | 1 | 12.04 | 12.11 | 0.994x |
| TinyLlama-1.1B | 8 | 27.73 | 27.71 | 1.001x |
| Qwen2.5-1.5B | 1 | 15.83 | 15.99 | 0.990x |
| Qwen2.5-1.5B | 8 | 42.24 | 42.22 | 1.001x |

At B = 1 a decode step of these small models is bound by the host: the patched module's Python forward (one row-count
check before `F.linear`) costs about 0.5-0.8 us per layer, 0.6-1.0% of the step; at B = 8 nothing measurable.

## Accuracy

`python benchmarks/linear.py --accuracy <id> ...`: wikitext-2 validation, 16 sequences x 1024 tokens (16,368
predictions), top-1 agreement with the same model in FP32
(FP32 weights, SDPA) and perplexity. Noise floor: what stock FP16 already loses against FP32 (`fp16 sdpa`, `fp16 stock`
= stock `F.linear` with SDPA / with our attention kernel).

| model | fp32 ppl | top-1 vs fp32, fp16 sdpa | ppl | fp16 stock | ppl | fp16 fused | ppl | fused vs stock |
|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | 6.9970 | 99.884% | 6.9970 | 99.817% | 6.9972 | 99.786% | 6.9969 | 99.823% |
| Qwen2.5-0.5B | 10.4876 | 99.664% | 10.4871 | 99.652% | 10.4879 | 99.578% | 10.4880 | 99.572% |
| Qwen2.5-1.5B | 7.7974 | 99.664% | 7.7985 | 99.688% | 7.7986 | 99.670% | 7.7978 | 99.695% |
| Qwen3-1.7B | 13.4748 | 99.762% | 13.4760 | 99.786% | 13.4766 | 99.725% | 13.4756 | 99.731% |

Perplexity: within 0.0010 of stock FP16 and 0.0008 of FP32 (fused is as close to FP32 as stock). Top-1: the kernel flips
0.21-0.42% of the predictions against FP32, stock FP16 0.18-0.35%: 1.06-1.29x stock's, within the range by which the two stock
FP16 variants differ (`fp16 stock` vs `fp16 sdpa`: 0.90-1.58x); the gap is 5-12 of 16,368 predictions per model. (Qwen3.5 is left out: its tokenizer returned no tokens on this box, cause not investigated; its prefill and LoRA
rows above use random token ids.)

## Fold interval

C, the K-elements per FP16 chain before the FP32 fold (`kChunk`), trades speed for accuracy: the fold costs ~256
instructions per thread once per C elements, against 1024 HFMA2 per 32-element stage; the error grows about as sqrt(C). Three builds differing
only in C (2026-11-02; GEMM: the 60 shapes with M >= 512 above, 2 rounds alternated; accuracy: the table above's run, once per build):

| C | GEMM geomean over stock | element error | top-1 flips vs FP32 (stock FP16: 0.18-0.35%) | registers (data-parallel / stream-K) |
|---|---|---|---|---|
| 128 | 1.87x | 1.2e-3 | 0.26-0.59% | 221 / 251 |
| **64** | 1.80x (median 0.958x of 128) | 8.7e-4 | 0.21-0.42% | 221 / 251 |
| 32 | 1.72x (median 0.919x of 128) | 6.4e-4 | 0.22-0.39% | 182 / 208 |

64 is shipped (the first version of P8 used 128): its top-1 flips are at the FP16 noise floor for 4% of the GEMM speed; 32 is
slower and not measurably better on top-1. Perplexity within 0.0033 of stock FP16 in all three (0.0010 at C = 64). All three pass `tests/test_linear.py`
with 0 spills.

## The backward: dX

dX = dY W (P8.6). Before: the kernel on a transposed copy of W (`W.t().contiguous()`, 2·N·K bytes moved per call and an
N x K transient). Now: the op reads W [N, K] in place as a [K, N] operand (`w_kn`), the same shared-memory words, so dX
is bitwise the copy path's. One process, A (copy) and B (in place) alternated, CUDA events, median of 6 rounds x 20
calls (`p86_ab.py kernel`, a one-off script); M rows, N and K of the forward layer:

| M | N | K | A copy + kernel | A kernel alone | B | B vs A kernel | B vs A |
|---|---|---|---|---|---|---|---|
| 2048 | 1536 | 1536 | 0.831 ms | 0.749 ms | 0.752 ms | 0.995x | 1.104x |
| 2048 | 256 | 1536 | 0.173 ms | 0.155 ms | 0.156 ms | 0.995x | 1.113x |
| 2048 | 8960 | 1536 | 4.626 ms | 4.162 ms | 4.193 ms | 0.992x | 1.103x |
| 2048 | 1536 | 8960 | 4.691 ms | 4.214 ms | 4.232 ms | 0.996x | 1.108x |
| 2048 | 896 | 4864 | 1.521 ms | 1.367 ms | 1.376 ms | 0.994x | 1.106x |
| 2048 | 4864 | 896 | 1.459 ms | 1.317 ms | 1.324 ms | 0.995x | 1.102x |
| 2048 | 151936 | 1536 | 113.182 ms | 69.980 ms | 70.553 ms | 0.992x | 1.604x |
| 4096 | 1536 | 1536 | 1.542 ms | 1.461 ms | 1.470 ms | 0.993x | 1.049x |
| 512 | 8960 | 1536 | 1.525 ms | 1.050 ms | 1.063 ms | 0.988x | 1.435x |

The in-place staging costs 0.4-1.2% of the GEMM (one `__byte_perm` per shared-memory word); the copy was 5-38% of
the whole dX. Peak memory of one dX of Qwen2.5-1.5B's lm_head (dY [2048, 151936], W [151936, 1536]), peak reset before
the call: 459 MiB above the inputs with the copy, 13 MiB in place (dX 6 MiB plus the stream-K workspace). LoRA step
(`p86_ab.py lora`), Qwen2.5-1.5B at 2048 tokens (rank 16 on every projection, gradient checkpointing, `patch_linear`,
AdamW), blocks of 5 steps alternated A B / B A, 40 steps each: 2590.9 -> 2507.9 ms (1.033x; steps within ±2 ms); the
peak of the step is unchanged (6.76 GiB).

## LoRA fine-tuning

A one-off run (not a repo script): Qwen2.5-0.5B loaded with `attn_implementation="flash_attn_legacy"`, peft rank-16 LoRA on every projection, 300 AdamW steps of 4 x 512
wikitext-2 tokens (lr 2e-4, warmup 20, clip 1.0, fp16 autocast + GradScaler), then the validation loss (64 x 512 tokens)
evaluated with stock `F.linear` for both variants. Two seeds (data order and LoRA init) as the noise floor.

| seed | val loss stock | val loss fused | train time stock | train time fused |
|---|---|---|---|---|
| 0 | 2.42137 | 2.42151 | 239.8 s | 183.0 s |
| 1 | 2.42031 | 2.42029 | 239.8 s | 182.9 s |

Fused vs stock: |Δ val loss| ≤ 1.4e-4, the per-step training loss 2.3-2.6e-4 apart on average; the two seeds differ by 1e-3.
Training 1.31x faster (169 layers patched; the fp32 adapters stay stock).

## Full fine-tuning: FP32 weights under FP16 autocast

The usual mixed-precision recipe (FP32 weights, `torch.autocast(dtype=torch.float16)`, `GradScaler`; HF Trainer
`fp16=True`) reached the kernel in neither direction before P8.8: `patch_linear` took FP16 weights only. Now an FP32
layer under FP16 autocast runs the kernel on the FP16 casts of x, W and bias (the casts autocast makes for
`F.linear`), dX on the kernel, dW stock (bitwise autocast's); the gradients flow back through the casts in FP32.
Layers narrower than 128 (LoRA adapters) stay stock. The casts are not cached across calls, unlike autocast's.

Before (stock, `prof_split.py full32`, CUDA time of one step): cuBLAS 66.0% of Qwen2.5-0.5B at 2048 tokens, 78.6% of
Qwen2.5-1.5B at 1024.

Step, A (stock) and B (`patch_linear`) alternated in blocks of 4 steps, 32 steps each (`p88_ab.py`, a one-off script;
gradient checkpointing, GradScaler):

| model | tokens | optimizer | stock | patched | speedup |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | 2048 | AdamW | 1486.0 ms | 1246.9 ms | 1.192x |
| Qwen2.5-1.5B | 1024 | none (FP32 AdamW states do not fit 16 GB) | 1874.0 ms | 1474.5 ms | 1.271x |

A 300-step full fine-tune (`ft_full.py`, a one-off script: Qwen2.5-0.5B, wikitext-2 train, 4 x 512 tokens, AdamW lr
2e-5, warmup 20, clip 1.0), then the validation loss (64 x 512 tokens) with stock `F.linear` for both:

| seed | val loss stock | val loss fused | skipped steps stock / fused | train time stock | train time fused |
|---|---|---|---|---|---|
| 0 | 2.52579 | 2.52515 | 3 / 3 | 377.9 s | 306.0 s |
| 1 | 2.53598 | 2.53600 | 2 / 2 | 378.3 s | 306.2 s |

Fused vs stock: |Δ val loss| ≤ 6.4e-4, the two seeds 1.0e-2 apart; the same GradScaler skips; 1.24x faster.

Loss scaling matters more than for stock. The kernel multiplies in FP16, so unscaled gradients of ~1e-5 underflow in
dX's products, where cuBLAS multiplies in FP32. On a toy MLP without a loss scale, dX was 14x stock's error against
FP32; with a scale of 2^10 or 2^16 it was 2x, the kernel's usual ratio. `GradScaler`, which FP16 autocast training uses
anyway, provides the scale.

## The flag path

`FLASH_ATTN_LEGACY_LINEAR=1`, Qwen2.5-1.5B loaded with `attn_implementation="flash_attn_legacy"` (transformers 5.20.2): 0
layers patched before the first forward, 197 after it, the pre-hook removed; prefill of 2048 tokens 945 -> 563 ms (1.68x),
top-1 99.56% against the run without the flag (2048 random tokens).
