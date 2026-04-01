# Flash Attention Legacy

**Flash Attention v2 for Pascal and Volta GPUs**

CUDA implementation of [Flash Attention v2](https://arxiv.org/abs/2307.09288) for older NVIDIA GPUs not supported by the [official flash-attn package](https://github.com/Dao-AILab/flash-attention).

## Why?

This project started as a fun experiment to avoid throwing away a couple of old Pascal GPUs. Turns out, there's a real use case: plenty of V100s and P100s are still running in clusters, labs, and home setups — perfectly functional hardware that gets left behind because modern LLM tooling assumes Ampere or newer.

Many modern LLMs are designed around flash attention — they expect O(N) memory attention and won't fit on your GPU without it. The official `flash-attn` package requires Ampere (SM 8.0+), leaving **V100s, P100s, GTX 1080s** without support.

Without flash attention on these GPUs:
- Standard attention allocates the full **N x N score matrix** per head per layer
- A 3B model with 1024 context uses **~300 MB** just for attention scores
- At 4096 context, that's **~5 GB** — often more than the model weights themselves
- You hit OOM long before reaching the model's actual context limit

If you have older GPUs collecting dust, this project gives them a second life.

## The Solution

Flash Attention Legacy brings the same O(N) memory algorithm to Pascal and Volta GPUs. It trades slightly more compute for dramatically less memory:

| | Standard (eager) | Flash Attention Legacy |
|---|---|---|
| **Attention memory** | O(N^2) — full score matrix | O(N) — tiled, never materialized |
| **Prefill speed** | Faster on Pascal (no tensor cores) | ~10-20% slower |
| **What it enables** | Limited by GPU memory | Longer contexts, larger models, bigger batches |

**The key insight**: on memory-constrained GPUs, the bottleneck is not compute — it's memory. Flash attention lets you run models that would otherwise OOM.

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
pip install -e .

# With HuggingFace integration
pip install -e ".[transformers]"

# With tests and benchmarks
pip install -e ".[test,bench]"
```

**Requirements**: CUDA toolkit >= 11.0, PyTorch >= 1.12 (PyTorch 2.4 for Pascal/SM 6.0), a Pascal or Volta GPU.

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

> Benchmarks below are preliminary — measured on a P100 without active cooling (thermal throttling at 84C). Speed numbers will improve with proper cooling. Memory numbers are unaffected by thermals.

### Prefill memory: eager vs flash_attn_legacy

Measured on **Tesla P100 16GB** with TinyLlama 1.1B. Extra memory above model weights during a single forward pass (prefill):

| Context (N) | eager (MB) | flash (MB) | Saved |
|---|---|---|---|
| 256 | 27.7 | 22.1 | 20% |
| 512 | 87.6 | 45.0 | 49% |
| 1024 | 304.3 | 88.5 | 71% |
| 2048 | OOM | 104.5 | eager OOM |

Memory grows **quadratically** with eager (O(N^2) score matrix) but **linearly** with flash. At N=2048, eager needs more memory than the GPU has — flash fits easily.

### Kernel-level performance

Raw kernel comparison (B=2, H=8, d=64) on P100:

| N | eager (MB) | flash (MB) | Memory saved | Speed |
|---|---|---|---|---|
| 512 | 34 | 1 | 97% | 0.5x |
| 1024 | 134 | 2 | 99% | 0.5x |
| 2048 | 537 | 4 | 99% | 0.6x |
| 4096 | 2147 | 7 | 100% | 0.7x |

Flash is ~2x slower on Pascal (no tensor cores), but uses **97-100% less memory**. The speed gap narrows on Volta (V100) which has 1st-gen tensor cores.

### Why is flash slower on Pascal?

Flash Attention trades compute for memory: it recomputes partial softmax values instead of storing the full N x N matrix. On modern GPUs (A100, H100), tensor cores make this recomputation nearly free. On Pascal, which only has standard CUDA cores, the extra arithmetic costs ~10-20% more time.

**This is the right trade-off on memory-constrained GPUs**: being 10% slower but fitting a model that would otherwise OOM is strictly better than not running at all.

## How It Works

### Algorithm

Flash Attention v2 tiles Q into blocks of Br rows and iterates over K/V blocks of Bc columns, using the [online softmax trick](https://arxiv.org/abs/1805.02867) to maintain running max/sum statistics. The full N x N attention score matrix is never materialized in HBM — only small tile-sized buffers live in shared memory.

### Forward pass

**Volta (SM 7.0):**
- S = Q @ K^T via `wmma::mma_sync` (16x16x16 tensor core fragments)
- P stored in shared memory as float
- O += P @ V accumulated in registers per thread
- Tiles: Br=128, Bc=64 (d=64) / Br=128, Bc=32 (d=128)

**Pascal (SM 6.0/6.1):**
- S = Q @ K^T via `half2` packed multiply-add (2x throughput vs scalar)
- Same shared-memory P buffer strategy as Volta
- Tiles: Br=128, Bc=32 (d=64) / Br=64, Bc=16 (d=128)
- Vectorized `half2` loads/stores throughout

### Backward pass

Both architectures share a unified backward kernel:
- **D = rowsum(dO * O)** precomputed in a separate kernel
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
pip install -e ".[test]"
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
4. **No variable-length** (packed batching) support
5. **Prefill only**: decode (q_len=1) falls back to standard attention (no O(N^2) issue there)

## Roadmap

- [x] GQA/MQA support
- [x] HuggingFace Transformers native integration (`attn_implementation="flash_attn_legacy"`)
- [ ] WMMA-based P@V in Volta forward (currently scalar)
- [ ] WMMA-based backward on Volta
- [ ] Variable-length sequence support
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
