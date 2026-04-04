# Flash Attention Legacy

**Flash Attention v2 for Pascal and Volta GPUs**

CUDA implementation of [Flash Attention v2](https://arxiv.org/abs/2307.09288) for older NVIDIA GPUs not supported by the [official flash-attn package](https://github.com/Dao-AILab/flash-attention).

## Why?

This project started as a fun experiment to avoid throwing away a couple of old Pascal GPUs. It turned into something useful for a specific, real problem.

### The problem

Many HuggingFace models and libraries explicitly require `attn_implementation="flash_attention_2"` — and the official [flash-attn](https://github.com/Dao-AILab/flash-attention) package only supports Ampere and newer (SM 8.0+). If you try to use it on a **V100, P100, or GTX 1080**, you get an error:

```
ValueError: FlashAttention2 is not supported on this GPU architecture (sm_70).
```

This leaves perfectly functional GPUs locked out — not because the hardware can't handle attention efficiently, but because the software won't run on it.

### What about SDPA?

PyTorch's `scaled_dot_product_attention` (SDPA) already provides memory-efficient attention on these GPUs via its `memory_efficient` backend. If you can use `attn_implementation="sdpa"`, **you should** — it works out of the box with no extra dependencies.

But not all code paths go through SDPA. Some models, training frameworks, and research codebases explicitly require `flash_attention_2` and won't fall back gracefully. That's where this package comes in.

### What this project actually is

1. **A drop-in replacement** for `flash_attention_2` on Pascal/Volta GPUs. If a model requires it and your GPU is rejected, `pip install` this and use `attn_implementation="flash_attn_legacy"` instead.

2. **A from-scratch CUDA implementation** of Flash Attention v2 — hand-written kernels with WMMA tensor cores (Volta) and half2 packed math (Pascal). No cuDNN, no ATen, no black boxes. If you want to understand how flash attention works at the CUDA level, this is a readable reference.

3. **Variable-length sequence support** (`flash_attn_varlen_func`) for packed batching without padding waste — useful for training with heterogeneous sequence lengths.

## Supported Hardware

| Architecture | GPUs | SM | Tensor Cores | Kernel |
|---|---|---|---|---|
| **Volta** | V100, Titan V | 7.0 | 1st gen | WMMA 16x16x16 |
| **Pascal** | P100, GTX 1080/Ti | 6.0/6.1 | None | half2 packed FP16 |

Head dimensions: **d=64, d=128**. FP16 only (Pascal/Volta lack native BF16).

Supports **MHA**, **GQA** (grouped-query), and **MQA** (multi-query) attention.

## Installation

```bash
git clone https://github.com/sirCamp/flash-attention-legacy.git
cd flash-attention-legacy
pip install -e . --no-build-isolation

# With HuggingFace integration
pip install -e ".[transformers]" --no-build-isolation

# With tests and benchmarks
pip install -e ".[test,bench]" --no-build-isolation
```

> **Why `--no-build-isolation`?** The CUDA extension needs to link against your installed PyTorch at build time. With build isolation, pip creates a temporary environment where PyTorch may not be available, causing the build to fail.

**Requirements**: CUDA toolkit >= 11.0, PyTorch >= 1.12 (PyTorch 2.4 for Pascal/SM 6.0), a Pascal or Volta GPU.

> **Beta**: This project is under active development. It has been tested with the following configuration:
> - PyTorch 2.4 (last version supporting SM 6.0 / Pascal)
> - Transformers 5.4
> - CUDA 12.4 / 12.8 (NVIDIA driver 580)
> - Tesla P100 16GB, Tesla V100-SXM2-32GB
>
> Other combinations may work but are not yet validated. Bug reports and contributions are welcome.

## Usage with HuggingFace Models

### Native integration (transformers >= 5.0)

Load any HuggingFace model with flash attention — no code changes needed:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    attn_implementation="flash_attn_legacy",
    torch_dtype=torch.float16,
    device_map="cuda",
)
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

inputs = tokenizer("The meaning of life is", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

The integration automatically uses our flash kernel for **prefill** (where memory savings matter) and falls back to standard attention for **decode** (q_len=1, no O(N^2) issue).

### Monkey-patch (transformers < 5.0)

```python
from transformers import AutoModelForCausalLM
from examples.hf_llama_patch import patch_llama_attention

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="cuda",
)
patch_llama_attention(model)
```

### Direct API

```python
import torch
from flash_attn_legacy import flash_attention

B, H, N, d = 2, 8, 2048, 64
q = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16)
k = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16)
v = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16)

out = flash_attention(q, k, v, is_causal=True)
```

### GQA (grouped-query attention)

```python
# 8 query heads, 2 KV heads (4:1 ratio, like Llama 2/3)
q = torch.randn(2, 8, 1024, 64, device='cuda', dtype=torch.float16)
k = torch.randn(2, 2, 1024, 64, device='cuda', dtype=torch.float16)
v = torch.randn(2, 2, 1024, 64, device='cuda', dtype=torch.float16)

out = flash_attention(q, k, v, is_causal=True)
```

### Variable-length (packed sequences)

```python
from flash_attn_legacy import flash_attn_varlen_func

# Batch of 3 sequences with different lengths (no padding waste)
seq_lens = [128, 256, 64]
total = sum(seq_lens)  # 448

q = torch.randn(total, 8, 64, device='cuda', dtype=torch.float16)
k = torch.randn(total, 8, 64, device='cuda', dtype=torch.float16)
v = torch.randn(total, 8, 64, device='cuda', dtype=torch.float16)

# Cumulative sequence lengths: [0, 128, 384, 448]
cu_seqlens = torch.tensor([0, 128, 384, 448], dtype=torch.int32, device='cuda')

out = flash_attn_varlen_func(
    q, k, v, cu_seqlens, cu_seqlens,
    max_seqlen_q=256, max_seqlen_k=256,
    causal=True,
)  # [448, 8, 64]
```

### Multi-Head Attention module

```python
from flash_attn_legacy import FlashMultiHeadAttention

attn = FlashMultiHeadAttention(embed_dim=512, num_heads=8, causal=True).cuda().half()
x = torch.randn(2, 1024, 512, device='cuda', dtype=torch.float16)
out = attn(x)  # [2, 1024, 512]
```

### GPU compatibility check

```python
from flash_attn_legacy import check_gpu_compatibility
check_gpu_compatibility()
```

## Benchmarks

> **Note**: All benchmarks below compare against **eager attention** (the naive O(N^2) implementation) to show the algorithmic difference. PyTorch's SDPA provides similar memory savings via its built-in `memory_efficient` backend — these numbers are about our kernel's characteristics, not a claim that eager is the only alternative.

### Prefill with TinyLlama 1.1B

Extra memory above model weights during a single forward pass (prefill only, no generation).

**V100 (Tesla V100-SXM2-32GB)** — WMMA tensor cores for both matmuls:

| N | eager (MB) | flash (MB) | Memory saved | Speedup |
|---|---|---|---|---|
| 256 | 27.7 | 22.1 | 20% | 1.24x |
| 512 | 87.6 | 45.0 | 49% | 0.86x |
| 1024 | 304.3 | 88.5 | 71% | 0.87x |
| 2048 | 1124.5 | 178.0 | 84% | 1.13x |
| 4096 | 4313.0 | 354.0 | 92% | 1.12x |
| 8192 | 16882.1 | 708.0 | **96%** | **1.15x** |

On V100, flash attention is **faster than eager for N >= 2048** thanks to WMMA tensor cores for both matmuls.

**P100 (Tesla P100-PCIE-16GB)** — half2 packed FP16, no tensor cores:

| N | eager (MB) | flash (MB) | Memory saved | Speedup |
|---|---|---|---|---|
| 256 | 27.7 | 22.1 | 20% | 0.91x |
| 512 | 87.6 | 45.0 | 49% | 0.91x |
| 1024 | 304.3 | 88.5 | 71% | 0.90x |
| 2048 | 1124.5 | 178.0 | 84% | 0.94x |
| 4096 | 4313.0 | 354.0 | 92% | 0.92x |
| 8192 | OOM | 708.0 | **eager OOM** | -- |

On P100, flash is ~6-10% slower (no tensor cores) but the memory savings are identical. At N=8192, **eager OOMs on 16GB while flash fits comfortably**.

Memory grows linearly (O(N)) instead of quadratically (O(N^2)) on both architectures.

### V100 — Prefill with Llama 3.1 8B Instruct

Same setup, larger model (head_dim=128, GQA 32Q/8KV heads, ~16GB FP16):

| N | eager (MB) | flash (MB) | Memory saved | Speedup |
|---|---|---|---|---|
| 256 | 96.6 | 96.6 | 0% | 0.91x |
| 512 | 194.0 | 194.0 | 0% | 0.86x |
| 1024 | 434.5 | 386.5 | 11% | 0.86x |
| 2048 | 1385.0 | 774.0 | 44% | 0.92x |
| 4096 | 4834.0 | 1546.0 | **68%** | 0.87x |

With head_dim=128, the attention score matrix is smaller per-element (fewer heads after GQA), so memory savings start later but are still substantial. At N=4096, flash saves **3.2 GB** of attention memory.

### V100 — Kernel-level comparison

Raw kernel (B=2, H=8, d=64):

| N | eager (MB) | flash (MB) | Memory saved | Speedup |
|---|---|---|---|---|
| 128 | 3.2 | 0.3 | 91% | 1.67x |
| 512 | 38.0 | 1.1 | 97% | 0.87x |
| 2048 | 557.8 | 4.3 | 99% | 1.07x |
| 4096 | 2197.8 | 8.7 | 100% | 1.26x |

### Pascal (P100) — Kernel-level comparison

Raw kernel on **Tesla P100-PCIE-16GB** (B=2, H=8, d=64):

| N | eager (ms) | flash (ms) | Speedup | eager (MB) | flash (MB) | Memory saved |
|---|---|---|---|---|---|---|
| 128 | 0.19 | 0.27 | 0.70x | 3.2 | 0.3 | 91% |
| 256 | 0.23 | 0.51 | 0.45x | 10.6 | 0.5 | 95% |
| 512 | 0.57 | 1.20 | 0.48x | 38.0 | 1.1 | 97% |
| 1024 | 1.95 | 3.35 | 0.58x | 143.7 | 2.2 | 98% |
| 2048 | 8.08 | 10.42 | 0.78x | 557.8 | 4.3 | 99% |
| 4096 | 28.16 | 36.03 | 0.78x | 2197.8 | 8.7 | 100% |

On Pascal, the trade-off is pure compute for memory: no tensor cores means slower matmuls, but the O(N) memory footprint is the same as on Volta. The speed gap narrows at longer sequences (0.78x at N=4096 vs 0.45x at N=256).

### Variable-length (packed sequences)

Compares padded batching (all sequences padded to max length) vs packed batching (`flash_attn_varlen_func`, single kernel launch). H=8, d=64, FP16, causal.

**V100 (Tesla V100-SXM2-32GB):**

| Scenario | Seqs | Max len | Padding waste | Memory saved | Speedup |
|---|---|---|---|---|---|
| uniform_short | 16 | 64 | 0% | 0% | 0.81x |
| uniform_medium | 8 | 256 | 0% | 0% | 0.90x |
| mixed_short | 8 | 128 | 52% | **52%** | 0.74x |
| mixed_heavy | 8 | 512 | 61% | **61%** | **1.88x** |
| one_long + rest_short | 16 | 1024 | 91% | **91%** | **6.78x** |
| all_long | 4 | 1024 | 0% | 0% | 0.97x |

**P100 (Tesla P100-PCIE-16GB):**

| Scenario | Seqs | Max len | Padding waste | Memory saved | Speedup |
|---|---|---|---|---|---|
| uniform_short | 16 | 64 | 0% | 0% | 0.92x |
| uniform_medium | 8 | 256 | 0% | 0% | 0.97x |
| mixed_short | 8 | 128 | 52% | **52%** | **1.51x** |
| mixed_heavy | 8 | 512 | 61% | **61%** | **2.17x** |
| one_long + rest_short | 16 | 1024 | 91% | **91%** | **8.39x** |
| all_long | 4 | 1024 | 0% | 0% | 0.99x |

When sequences have very different lengths (common in training), packing avoids wasting compute and memory on padding tokens. The `one_long_rest_short` scenario (1x1024 + 15x32) shows **6.78-8.39x speedup** and **91% memory savings** — padded would compute attention on 16x1024 = 16K tokens, packed only processes 1504 actual tokens.

### V100 — Training with TinyLlama 1.1B

End-to-end training comparison (forward + backward) on **Tesla V100-SXM2-32GB**. TinyLlama 1.1B, FP32 weights + FP16 autocast + GradScaler, B=2, 20 steps, gradient checkpointing enabled.

| N | eager (ms/step) | flash (ms/step) | Speedup | eager loss | flash loss | Note |
|---|---|---|---|---|---|---|
| 2048 | 1369.5 | 1179.9 | **1.16x** | 9.24 | 9.31 | Flash faster thanks to WMMA backward |
| 4096 | OOM | 2975.6 | -- | OOM | 8.24 | **Eager cannot train at this length** |

At N=2048, flash is **16% faster** than eager in full training (forward + backward) thanks to WMMA tensor cores in both passes. Both backends produce coherent loss curves (< 1% relative difference).

At N=4096, eager runs out of memory while flash trains successfully. Note: SDPA (`attn_implementation="sdpa"`) would also handle this — these numbers demonstrate our kernel's correctness and performance characteristics vs the naive baseline.

### Embeddings with Qwen3-Embedding-0.6B

Sentence-transformers encoding benchmark with [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B). Batch=16, 5 runs.

**V100 (Tesla V100-SXM2-32GB) — FP32:**

| seq_len | eager (ms) | sdpa (ms) | flash (ms) | flash vs eager | flash vs sdpa | eager (MB) | sdpa (MB) | flash (MB) |
|---|---|---|---|---|---|---|---|---|
| 256 | 236.6 | 236.3 | 252.5 | 0.94x | 0.94x | 306 | 298 | 281 |
| 512 | 467.3 | 457.3 | 488.3 | 0.96x | 0.94x | 692 | 609 | 557 |
| 1024 | 973.3 | 937.1 | 977.4 | 1.00x | 0.96x | 1698 | 1349 | 1109 |
| 2048 | 2473.8 | 2323.4 | 2123.5 | **1.16x** | **1.09x** | 4687 | 3225 | **2215** |
| 4096 | 6423.7 | 5800.9 | 5102.3 | **1.26x** | **1.14x** | 14554 | 8590 | **4424** |

In FP32 on V100, flash is **faster than both eager and SDPA** at long sequences — 1.26x vs eager and 1.14x vs SDPA at seq_len=4096 — while using **half the memory of SDPA** (4.4 GB vs 8.6 GB). The WMMA tensor cores give a real speed advantage when FP32 accumulation dominates. With the larger Qwen3-Embedding-4B, the same pattern holds: at seq_len=4096, flash is 1.08x faster than SDPA while using **half the memory** (7.1 GB vs 14.5 GB).

**V100 (Tesla V100-SXM2-32GB) — FP16:**

| seq_len | eager (ms) | flash (ms) | Speedup | eager (MB) | flash (MB) | Memory saved |
|---|---|---|---|---|---|---|
| 256 | 77.6 | 98.3 | 0.79x | 298 | 280 | 6% |
| 512 | 147.6 | 189.9 | 0.78x | 658 | 557 | 15% |
| 1024 | 320.1 | 409.7 | 0.78x | 1571 | 1109 | 29% |
| 2048 | 998.6 | 992.3 | **1.01x** | 4173 | 2213 | **47%** |
| 4096 | 2677.0 | 2747.2 | 0.97x | 12502 | 4422 | **65%** |

In FP16, flash matches eager speed at long sequences (1.01x at 2048) while saving up to **8 GB** of memory at seq_len=4096.

**P100 (Tesla P100-PCIE-16GB) — FP16:**

| seq_len | eager (ms) | flash (ms) | Speedup | eager (MB) | flash (MB) | Memory saved |
|---|---|---|---|---|---|---|
| 256 | 361.3 | 409.5 | 0.88x | 298 | 280 | 6% |
| 512 | 1438.7 | 1776.7 | 0.81x | 1571 | 1109 | 29% |
| 2048 | 3976.0 | 4446.1 | 0.89x | 4173 | 2213 | 47% |
| 4096 | 10146.6 | 12368.8 | 0.82x | 12502 | 4422 | **65%** |

On P100, flash is ~10-20% slower (no tensor cores), but saves up to **8 GB** at seq_len=4096.

**P100 (Tesla P100-PCIE-16GB) — FP32 (sentence-transformers default):**

| seq_len | eager | sdpa (ms) | flash (ms) | flash vs sdpa | sdpa (MB) | flash (MB) | Memory saved vs sdpa |
|---|---|---|---|---|---|---|---|
| 4096 | OOM | 11248.0 | 13141.3 | 0.86x | 8590 | 4424 | **49%** |
| 8192 | OOM | OOM | 39048.9 | -- | OOM | 8845 | **only flash fits** |

In FP32 (the default for sentence-transformers), flash uses **half the memory of SDPA** at seq_len=4096. At seq_len=8192, both eager and SDPA run out of memory on 16GB — flash is the only backend that fits.

## How It Works

### Algorithm

Flash Attention v2 tiles Q into blocks of Br rows and iterates over K/V blocks of Bc columns, using the [online softmax trick](https://arxiv.org/abs/1805.02867) to maintain running max/sum statistics. The full N x N attention score matrix is never materialized in HBM — only small tile-sized buffers live in shared memory.

### Forward pass

**Volta (SM 7.0):**
- S = Q @ K^T via `wmma::mma_sync` (16x16x16 tensor core fragments)
- P stored in shared memory as float, converted to half for WMMA
- O += P @ V via `wmma::mma_sync` (P converted to half, output in float registers)
- Tiles: Br=128, Bc=64 (d=64) / Br=128, Bc=32 (d=128)

**Pascal (SM 6.0/6.1):**
- S = Q @ K^T via `half2` packed multiply-add (2x throughput vs scalar)
- Same shared-memory P buffer strategy as Volta
- Tiles: Br=128, Bc=32 (d=64) / Br=64, Bc=16 (d=128)
- Vectorized `half2` loads/stores throughout

### Backward pass

**Volta (SM 7.0):**
- **D = rowsum(dO * O)** precomputed in a separate kernel
- All 5 matmuls use `wmma::mma_sync` (16x16x16 tensor core fragments):
  - S = Q @ K^T (recompute), dP = dO @ V^T, dV += P^T @ dO, dK += dS^T @ Q, dQ += dS @ K
- **dK, dV** accumulated in shared memory FP32 buffers with WMMA in-place accumulation
- **dQ** accumulated in global FP32 buffer with `atomicAdd`

**Pascal (SM 6.0/6.1):**
- Same algorithm with `half2` packed multiply-add (no tensor cores)
- **dQ** accumulated in FP32 with `atomicAdd`
- **dK, dV** accumulated in per-thread FP32 registers, converted to FP16 on writeback
- Multi-thread per KV row with warp shuffle reduction

### Register budget

| Config | Regs/thread | Shared mem | Thread util |
|---|---|---|---|
| Pascal d=64, Br=128 | 256 bytes | 40 KB | 100% |
| Pascal d=128, Br=64 | 512 bytes | 28 KB | 100% |
| Volta d=64, Br=128 | 256 bytes | 64 KB | 100% |
| Volta d=128, Br=128 | 512 bytes | 64 KB | 100% |

## Testing

```bash
pip install -e ".[test]" --no-build-isolation
pytest tests/ -v

# Performance comparison table
pytest tests/test_performance.py::TestComparisonTable -v -s

# Prefill memory benchmark with real models
pytest tests/test_generation_benchmark.py -v -s -k "prefill"
```

## Limitations

1. **Head dimensions**: only d=64 and d=128
2. **FP16 only**: no BF16 (Pascal/Volta lack native BF16)
3. **No dropout** in attention kernel
4. **Prefill only**: decode (q_len=1) falls back to standard attention (no O(N^2) issue there)

## Roadmap

- [x] GQA/MQA support
- [x] HuggingFace Transformers native integration (`attn_implementation="flash_attn_legacy"`)
- [x] WMMA tensor cores for both Q@K^T and P@V on Volta
- [x] WMMA-based backward on Volta
- [x] Variable-length sequence support (`flash_attn_varlen_func`)
- [ ] Turing (SM 7.5) optimized path

## References

- Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) (2022)
- Dao, [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.09288) (2023)
- Milakov & Gimelshein, [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) (2018)

## Citation

If you use Flash Attention Legacy in your research, please cite:

```bibtex
@software{campese2025flashattentionlegacy,
  author       = {Campese, Stefano},
  title        = {Flash Attention Legacy: Flash Attention v2 for Pascal and Volta GPUs},
  year         = {2025},
  url          = {https://github.com/sirCamp/flash-attention-legacy},
  license      = {MIT}
}
```

## License

MIT
