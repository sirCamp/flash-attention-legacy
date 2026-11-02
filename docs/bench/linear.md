# FP16 linear layers on the P100: `fused_linear` / `patch_linear`

PLAN P8. Every FP16 `nn.Linear` on a compute capability 6.0 GPU (Tesla P100) can run on a fused GEMM kernel
(`csrc/kernels/linear_pascal.cu`) instead of torch's cuBLAS call. **Off by default**: `flash_attn_legacy.patch_linear(model)`,
or `FLASH_ATTN_LEGACY_LINEAR=1` for models loaded with `attn_implementation="flash_attn_legacy"`; a no-op on every other GPU.

Why: torch runs FP16 GEMMs on sm_60 with cuBLAS `COMPUTE_32F` (7-8.5 TFLOPS, below the card's FP32 rate), while its
HFMA2 peak is ~19 TFLOPS. The kernel multiplies with HFMA2 (`half2` pairs along K) and folds each lane into FP32 every 128
K-elements (chunked accumulation, IBM arXiv 1812.08011; DeepSeek-V3's FP32 promotion every 128), so the error does not
grow with K as in a pure FP16 accumulation (cuBLAS `COMPUTE_16F`, llama.cpp's P100 path). Calls with fewer than 128 rows
(decode) go to `F.linear` unchanged.

Measured on mew (Tesla P100-PCIE-16GB, passive cooling, GPU cooled below 50-60 C before each run), torch 2.7.1+cu126,
transformers 5.20.2, CUDA 12.8 build, 2026-10-31.

## GEMM shapes

`python benchmarks/linear.py --ms 128 512 2048 8192`: y = x W^T on the layer shapes of four models (qkv and gate+up fused
as one GEMM), 5 interleaved rounds, median. Error = relative Frobenius error against an FP32 product of the same FP16
inputs (first 2048 rows).

| model | layer | M | N | K | stock TFLOPS | fused TFLOPS | speedup | error stock | error fused |
|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b | qkv | 128 | 1152 | 896 | 4.74 | 5.17 | 1.09x | 2.1e-04 | 1.0e-03 |
| qwen2.5-0.5b | qkv | 512 | 1152 | 896 | 5.86 | 10.17 | 1.74x | 2.1e-04 | 1.2e-03 |
| qwen2.5-0.5b | qkv | 2048 | 1152 | 896 | 7.04 | 12.79 | 1.82x | 2.1e-04 | 1.2e-03 |
| qwen2.5-0.5b | qkv | 8192 | 1152 | 896 | 7.93 | 13.52 | 1.70x | 2.1e-04 | 1.2e-03 |
| qwen2.5-0.5b | o | 128 | 896 | 896 | 5.37 | 2.58 | 0.48x | 2.1e-04 | 1.2e-03 |
| qwen2.5-0.5b | o | 512 | 896 | 896 | 6.24 | 10.87 | 1.74x | 2.1e-04 | 1.2e-03 |
| qwen2.5-0.5b | o | 2048 | 896 | 896 | 7.01 | 13.61 | 1.94x | 2.1e-04 | 1.2e-03 |
| qwen2.5-0.5b | o | 8192 | 896 | 896 | 8.54 | 13.71 | 1.61x | 2.1e-04 | 1.2e-03 |
| qwen2.5-0.5b | gateup | 128 | 9728 | 896 | 6.87 | 12.31 | 1.79x | 2.1e-04 | 1.2e-03 |
| qwen2.5-0.5b | gateup | 512 | 9728 | 896 | 7.12 | 13.28 | 1.86x | 2.1e-04 | 1.2e-03 |
| qwen2.5-0.5b | gateup | 2048 | 9728 | 896 | 8.37 | 13.60 | 1.63x | 2.1e-04 | 1.2e-03 |
| qwen2.5-0.5b | gateup | 8192 | 9728 | 896 | 7.20 | 13.42 | 1.86x | 2.1e-04 | 1.2e-03 |
| qwen2.5-0.5b | down | 128 | 896 | 4864 | 6.46 | 10.55 | 1.63x | 2.1e-04 | 1.2e-03 |
| qwen2.5-0.5b | down | 512 | 896 | 4864 | 6.72 | 13.38 | 1.99x | 2.1e-04 | 1.2e-03 |
| qwen2.5-0.5b | down | 2048 | 896 | 4864 | 7.45 | 14.09 | 1.89x | 2.1e-04 | 1.2e-03 |
| qwen2.5-0.5b | down | 8192 | 896 | 4864 | 8.71 | 14.10 | 1.62x | 2.1e-04 | 1.2e-03 |
| qwen2.5-0.5b | lm_head | 128 | 151936 | 896 | 8.08 | 13.58 | 1.68x | 2.1e-04 | 1.2e-03 |
| qwen2.5-0.5b | lm_head | 512 | 151936 | 896 | 7.39 | 13.71 | 1.85x | 2.1e-04 | 1.2e-03 |
| qwen2.5-0.5b | lm_head | 2048 | 151936 | 896 | 7.41 | 13.84 | 1.87x | 2.1e-04 | 1.2e-03 |
| qwen2.5-0.5b | lm_head | 8192 | 151936 | 896 | 7.41 | 13.91 | 1.88x | 2.1e-04 | 1.2e-03 |
| qwen2.5-1.5b | qkv | 128 | 2048 | 1536 | 5.81 | 9.73 | 1.67x | 2.1e-04 | 1.1e-03 |
| qwen2.5-1.5b | qkv | 512 | 2048 | 1536 | 6.93 | 12.99 | 1.87x | 2.1e-04 | 1.2e-03 |
| qwen2.5-1.5b | qkv | 2048 | 2048 | 1536 | 7.30 | 13.61 | 1.86x | 2.1e-04 | 1.2e-03 |
| qwen2.5-1.5b | qkv | 8192 | 2048 | 1536 | 8.26 | 13.84 | 1.68x | 2.1e-04 | 1.2e-03 |
| qwen2.5-1.5b | o | 128 | 1536 | 1536 | 5.22 | 8.48 | 1.62x | 2.1e-04 | 1.1e-03 |
| qwen2.5-1.5b | o | 512 | 1536 | 1536 | 6.87 | 12.71 | 1.85x | 2.1e-04 | 1.2e-03 |
| qwen2.5-1.5b | o | 2048 | 1536 | 1536 | 7.25 | 13.53 | 1.87x | 2.1e-04 | 1.2e-03 |
| qwen2.5-1.5b | o | 8192 | 1536 | 1536 | 8.48 | 13.83 | 1.63x | 2.1e-04 | 1.2e-03 |
| qwen2.5-1.5b | gateup | 128 | 17920 | 1536 | 7.18 | 13.46 | 1.88x | 2.1e-04 | 1.2e-03 |
| qwen2.5-1.5b | gateup | 512 | 17920 | 1536 | 8.52 | 13.88 | 1.63x | 2.1e-04 | 1.2e-03 |
| qwen2.5-1.5b | gateup | 2048 | 17920 | 1536 | 7.45 | 13.77 | 1.85x | 2.1e-04 | 1.2e-03 |
| qwen2.5-1.5b | gateup | 8192 | 17920 | 1536 | 7.48 | 13.92 | 1.86x | 2.1e-04 | 1.2e-03 |
| qwen2.5-1.5b | down | 128 | 1536 | 8960 | 5.25 | 12.74 | 2.43x | 2.1e-04 | 1.2e-03 |
| qwen2.5-1.5b | down | 512 | 1536 | 8960 | 4.92 | 13.80 | 2.80x | 2.1e-04 | 1.2e-03 |
| qwen2.5-1.5b | down | 2048 | 1536 | 8960 | 5.02 | 14.02 | 2.79x | 2.1e-04 | 1.2e-03 |
| qwen2.5-1.5b | down | 8192 | 1536 | 8960 | 8.57 | 14.15 | 1.65x | 2.1e-04 | 1.2e-03 |
| qwen2.5-1.5b | lm_head | 128 | 151936 | 1536 | 8.12 | 13.82 | 1.70x | 2.1e-04 | 1.2e-03 |
| qwen2.5-1.5b | lm_head | 512 | 151936 | 1536 | 7.43 | 13.80 | 1.86x | 2.1e-04 | 1.2e-03 |
| qwen2.5-1.5b | lm_head | 2048 | 151936 | 1536 | 7.45 | 13.91 | 1.87x | 2.1e-04 | 1.2e-03 |
| qwen2.5-1.5b | lm_head | 8192 | 151936 | 1536 | 7.46 | 14.10 | 1.89x | 2.1e-04 | 1.2e-03 |
| qwen2.5-3b | qkv | 128 | 2560 | 2048 | 5.94 | 10.19 | 1.71x | 2.1e-04 | 1.2e-03 |
| qwen2.5-3b | qkv | 512 | 2560 | 2048 | 6.36 | 13.00 | 2.04x | 2.1e-04 | 1.2e-03 |
| qwen2.5-3b | qkv | 2048 | 2560 | 2048 | 7.41 | 13.73 | 1.85x | 2.1e-04 | 1.2e-03 |
| qwen2.5-3b | qkv | 8192 | 2560 | 2048 | 8.56 | 13.93 | 1.63x | 2.1e-04 | 1.2e-03 |
| qwen2.5-3b | o | 128 | 2048 | 2048 | 6.09 | 10.74 | 1.76x | 2.1e-04 | 1.2e-03 |
| qwen2.5-3b | o | 512 | 2048 | 2048 | 7.07 | 13.34 | 1.89x | 2.1e-04 | 1.2e-03 |
| qwen2.5-3b | o | 2048 | 2048 | 2048 | 7.40 | 13.71 | 1.85x | 2.1e-04 | 1.2e-03 |
| qwen2.5-3b | o | 8192 | 2048 | 2048 | 8.29 | 13.91 | 1.68x | 2.1e-04 | 1.2e-03 |
| qwen2.5-3b | gateup | 128 | 22016 | 2048 | 7.33 | 13.64 | 1.86x | 2.1e-04 | 1.2e-03 |
| qwen2.5-3b | gateup | 512 | 22016 | 2048 | 8.04 | 13.87 | 1.72x | 2.1e-04 | 1.2e-03 |
| qwen2.5-3b | gateup | 2048 | 22016 | 2048 | 7.50 | 13.80 | 1.84x | 2.1e-04 | 1.2e-03 |
| qwen2.5-3b | gateup | 8192 | 22016 | 2048 | 7.49 | 14.00 | 1.87x | 2.1e-04 | 1.2e-03 |
| qwen2.5-3b | down | 128 | 2048 | 11008 | 5.96 | 13.25 | 2.22x | 2.1e-04 | 1.2e-03 |
| qwen2.5-3b | down | 512 | 2048 | 11008 | 4.85 | 13.90 | 2.87x | 2.1e-04 | 1.2e-03 |
| qwen2.5-3b | down | 2048 | 2048 | 11008 | 4.98 | 14.06 | 2.82x | 2.1e-04 | 1.2e-03 |
| qwen2.5-3b | down | 8192 | 2048 | 11008 | 8.34 | 14.18 | 1.70x | 2.1e-04 | 1.2e-03 |
| qwen2.5-3b | lm_head | 128 | 151936 | 2048 | 8.21 | 13.92 | 1.70x | 2.1e-04 | 1.2e-03 |
| qwen2.5-3b | lm_head | 512 | 151936 | 2048 | 7.51 | 13.88 | 1.85x | 2.1e-04 | 1.2e-03 |
| qwen2.5-3b | lm_head | 2048 | 151936 | 2048 | 7.49 | 13.99 | 1.87x | 2.1e-04 | 1.2e-03 |
| qwen2.5-3b | lm_head | 8192 | 151936 | 2048 | 7.50 | 14.21 | 1.89x | 2.1e-04 | 1.2e-03 |
| llama3.1-8b | qkv | 128 | 6144 | 4096 | 6.99 | 13.45 | 1.92x | 2.1e-04 | 1.2e-03 |
| llama3.1-8b | qkv | 512 | 6144 | 4096 | 7.44 | 13.89 | 1.87x | 2.1e-04 | 1.2e-03 |
| llama3.1-8b | qkv | 2048 | 6144 | 4096 | 8.47 | 14.04 | 1.66x | 2.1e-04 | 1.2e-03 |
| llama3.1-8b | qkv | 8192 | 6144 | 4096 | 7.54 | 13.98 | 1.85x | 2.1e-04 | 1.2e-03 |
| llama3.1-8b | o | 128 | 4096 | 4096 | 6.44 | 13.24 | 2.06x | 2.1e-04 | 1.2e-03 |
| llama3.1-8b | o | 512 | 4096 | 4096 | 7.36 | 13.77 | 1.87x | 2.1e-04 | 1.2e-03 |
| llama3.1-8b | o | 2048 | 4096 | 4096 | 7.94 | 14.03 | 1.77x | 2.1e-04 | 1.2e-03 |
| llama3.1-8b | o | 8192 | 4096 | 4096 | 7.55 | 13.82 | 1.83x | 2.1e-04 | 1.2e-03 |
| llama3.1-8b | gateup | 128 | 28672 | 4096 | 7.45 | 14.04 | 1.89x | 2.1e-04 | 1.2e-03 |
| llama3.1-8b | gateup | 512 | 28672 | 4096 | 8.63 | 14.11 | 1.63x | 2.1e-04 | 1.2e-03 |
| llama3.1-8b | gateup | 2048 | 28672 | 4096 | 7.53 | 14.03 | 1.86x | 2.1e-04 | 1.2e-03 |
| llama3.1-8b | gateup | 8192 | 28672 | 4096 | 7.55 | 14.20 | 1.88x | 2.1e-04 | 1.2e-03 |
| llama3.1-8b | down | 128 | 4096 | 14336 | 4.62 | 13.79 | 2.99x | 2.1e-04 | 1.2e-03 |
| llama3.1-8b | down | 512 | 4096 | 14336 | 4.99 | 14.02 | 2.81x | 2.1e-04 | 1.2e-03 |
| llama3.1-8b | down | 2048 | 4096 | 14336 | 7.94 | 14.17 | 1.78x | 2.1e-04 | 1.2e-03 |
| llama3.1-8b | down | 8192 | 4096 | 14336 | 7.14 | 13.86 | 1.94x | 2.1e-04 | 1.2e-03 |
| llama3.1-8b | lm_head | 128 | 128256 | 4096 | 8.49 | 14.06 | 1.66x | 2.1e-04 | 1.2e-03 |
| llama3.1-8b | lm_head | 512 | 128256 | 4096 | 7.51 | 13.98 | 1.86x | 2.1e-04 | 1.2e-03 |
| llama3.1-8b | lm_head | 2048 | 128256 | 4096 | 7.55 | 14.21 | 1.88x | 2.1e-04 | 1.2e-03 |
| llama3.1-8b | lm_head | 8192 | 128256 | 4096 | 7.49 | 14.33 | 1.91x | 2.1e-04 | 1.2e-03 |

The kernel reaches 12.7-14.4 TFLOPS from M = 512 on (67-76% of the HFMA2 peak; 10-11 on the smallest Qwen2.5-0.5B qkv / o shapes at M = 512), stock 4.7-8.7. Its element-wise error is about 6x
stock's (1.2e-3 vs 2.1e-4: products and the first 64 partial sums per lane in FP16), and does not depend on K. The one
case below 1 is the smallest shape (M = 128, N = K = 896: 7 output tiles for 56 SMs, ~40 us): 0.48x in this run, 0.94x
re-measured in a sweep over M = 128-512, N = 896-2048, K = 896 / 1536, where every other shape is 1.09-1.89x
(1.36x from M = 256). The package's build vs the standalone prototype it was taken from: bitwise equal outputs on 45 shapes, 0.996x its speed.

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
| TinyLlama-1.1B-Chat-v1.0 | prefill | 1 | 1024 | 3,254 | 5,332 | 1.64x | 0.1% / 0.0% | - |
| TinyLlama-1.1B-Chat-v1.0 | prefill | 1 | 4096 | 3,080 | 4,434 | 1.44x | 0.1% / 0.0% | - |
| TinyLlama-1.1B-Chat-v1.0 | prefill | 1 | 8192 | 2,536 | 3,482 | 1.37x | 0.1% / 0.1% | - |
| TinyLlama-1.1B-Chat-v1.0 | decode | 1 | 2048 | 77 | 76 | 0.99x | 2.7% / 0.9% | - |
| TinyLlama-1.1B-Chat-v1.0 | decode | 8 | 2048 | 289 | 282 | 0.98x | 0.4% / 0.1% | - |
| TinyLlama-1.1B-Chat-v1.0 | LoRA train | 1 | 1024 | 961 | 1,273 | 1.32x | 0.0% / 0.0% | 12.2555 / 12.2552 |
| TinyLlama-1.1B-Chat-v1.0 | LoRA train | 1 | 2048 | 912 | 1,184 | 1.30x | 0.0% / 0.0% | 12.1919 / 12.1922 |
| TinyLlama-1.1B-Chat-v1.0 | LoRA train | 1 | 4096 | 808 | 999 | 1.24x | 0.1% / 0.0% | 11.9626 / 11.9633 |
| Qwen2.5-1.5B | prefill | 1 | 1024 | 2,226 | 3,972 | 1.78x | 0.1% / 0.0% | - |
| Qwen2.5-1.5B | prefill | 1 | 4096 | 1,898 | 3,274 | 1.73x | 0.0% / 0.0% | - |
| Qwen2.5-1.5B | prefill | 1 | 8192 | 1,861 | 2,580 | 1.39x | 0.0% / 0.0% | - |
| Qwen2.5-1.5B | decode | 1 | 2048 | 57 | 57 | 0.99x | 1.2% / 1.6% | - |
| Qwen2.5-1.5B | decode | 8 | 2048 | 188 | 186 | 0.99x | 0.1% / 0.2% | - |
| Qwen2.5-1.5B | LoRA train | 1 | 1024 | 705 | 916 | 1.30x | 0.0% / 0.0% | 12.2769 / 12.2772 |
| Qwen2.5-1.5B | LoRA train | 1 | 2048 | 685 | 908 | 1.33x | 0.0% / 0.0% | 12.2107 / 12.2105 |
| Qwen2.5-1.5B | LoRA train | 1 | 4096 | 573 | 775 | 1.35x | 0.0% / 0.0% | 12.1454 / 12.1453 |
| Qwen3.5-2B | prefill | 1 | 1024 | 2,033 | 3,107 | 1.53x | 0.0% / 0.0% | - |
| Qwen3.5-2B | prefill | 1 | 4096 | 2,134 | 3,002 | 1.41x | 0.0% / 0.0% | - |
| Qwen3.5-2B | prefill | 1 | 8192 | 1,912 | 2,722 | 1.42x | 0.1% / 0.0% | - |
| Qwen3.5-2B | decode | 1 | 2048 | 40 | 39 | 0.97x | 0.3% / 0.8% | - |
| Qwen3.5-2B | decode | 8 | 2048 | 172 | 169 | 0.99x | 0.2% / 0.1% | - |
| Qwen3.5-2B | LoRA train | 1 | 1024 | 478 | 607 | 1.27x | 0.0% / 0.0% | 12.3396 / 12.3395 |
| Qwen3.5-2B | LoRA train | 1 | 2048 | 448 | 554 | 1.24x | 0.0% / 0.2% | 12.2298 / 12.229 |
| Qwen3.5-4B | prefill | 1 | 1024 | 767 | 1,306 | 1.70x | 0.3% / 0.0% | - |
| Qwen3.5-4B | prefill | 1 | 4096 | 819 | 1,223 | 1.49x | 0.0% / 0.0% | - |
| Qwen3.5-4B | prefill | 1 | 8192 | 753 | 1,098 | 1.46x | 0.0% / 0.0% | - |
| Qwen3.5-4B | decode | 1 | 2048 | 25 | 25 | 1.00x | 0.0% / 0.4% | - |
| Qwen3.5-4B | decode | 8 | 2048 | 72 | 71 | 0.98x | 0.3% / 0.0% | - |
| Qwen3.5-4B | LoRA train | 1 | 1024 | 210 | 271 | 1.29x | 0.5% / 0.0% | 12.0765 / 12.0762 |
| Llama-3.1-8B-Instruct | prefill | 1 | 1024 | 468 | 877 | 1.88x | 0.2% / 0.0% | - |
| Llama-3.1-8B-Instruct | decode | 1 | 2048 | 20 | 20 | 1.00x | 0.0% / 0.0% | - |

Prefill 1.37-1.88x (geomean 1.55x), LoRA train 1.24-1.35x (1.29x; the losses agree to 3-4 digits). Decode does not use
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
| TinyLlama-1.1B-Chat-v1.0 | 6.9970 | 99.884% | 6.9970 | 99.817% | 6.9972 | 99.743% | 6.9974 | 99.774% |
| Qwen2.5-0.5B | 10.4876 | 99.664% | 10.4871 | 99.652% | 10.4879 | 99.407% | 10.4867 | 99.407% |
| Qwen2.5-1.5B | 7.7974 | 99.664% | 7.7985 | 99.688% | 7.7986 | 99.487% | 7.7976 | 99.572% |
| Qwen3-1.7B | 13.4748 | 99.762% | 13.4760 | 99.786% | 13.4766 | 99.713% | 13.4753 | 99.701% |

Perplexity: within 0.0013 of stock FP16 and of FP32 (fused is as close to FP32 as stock). Top-1: the kernel flips
0.26-0.59% of the predictions against FP32, stock FP16 0.18-0.35%: 1.3-1.7x more near-ties land on the other side, the
element-wise error above. (Qwen3.5 is left out: its tokenizer returned no tokens on this box, cause not investigated; its prefill and LoRA
rows above use random token ids.)

## LoRA fine-tuning

A one-off run (not a repo script): Qwen2.5-0.5B loaded with `attn_implementation="flash_attn_legacy"`, peft rank-16 LoRA on every projection, 300 AdamW steps of 4 x 512
wikitext-2 tokens (lr 2e-4, warmup 20, clip 1.0, fp16 autocast + GradScaler), then the validation loss (64 x 512 tokens)
evaluated with stock `F.linear` for both variants. Two seeds (data order and LoRA init) as the noise floor.

| seed | val loss stock | val loss fused | train time stock | train time fused |
|---|---|---|---|---|
| 0 | 2.42132 | 2.42147 | 239.5 s | 179.4 s |
| 1 | 2.42028 | 2.42019 | 239.9 s | 179.4 s |

Fused vs stock: |Δ val loss| ≤ 1.5e-4, the per-step training loss 3e-4 apart on average; the two seeds differ by 1e-3.
Training 1.34x faster (169 layers patched; the fp32 adapters stay stock).

## The flag path

`FLASH_ATTN_LEGACY_LINEAR=1`, Qwen2.5-1.5B loaded with `attn_implementation="flash_attn_legacy"` (transformers 5.20.2): 0
layers patched before the first forward, 197 after it, the pre-hook removed; prefill of 2048 tokens 944 -> 548 ms (1.72x),
top-1 99.37% against the run without the flag (2048 random tokens).
