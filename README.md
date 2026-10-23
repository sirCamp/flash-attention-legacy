# Flash Attention Legacy

**Flash Attention v2 for Pascal and Volta GPUs**

CUDA implementation of [Flash Attention v2](https://arxiv.org/abs/2307.08691) for older NVIDIA GPUs not supported by the [official flash-attn package](https://github.com/Dao-AILab/flash-attention).

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
| **Pascal** | P100, GTX 1080/Ti | 6.0/6.1 | None | CUDA cores (fp16 storage, fp32 math) |
| **Turing** | T4, RTX 20xx | 7.5 | 2nd gen | `mma.sync` m16n8k8 + `ldmatrix` (FA2 register layout; tiles sized for its 64 KB of shared memory per block) |

Head dimensions: **d = 32, 64, 72, 80, 96, 128, 256** (72 and 80, as in SigLIP / Qwen2-VL vision towers, run on wider tiles with zero-padded columns).

The kernels compute in **FP16**.
- **Mixed precision** (`torch.autocast("cuda", dtype=torch.float16)` with a `GradScaler`): BF16/FP32 inputs are cast to FP16 exactly as autocast does for SDPA (no range check), training included.
- **Outside autocast**, BF16 and FP32 inputs (e.g. a checkpoint loaded in bf16 as it is) run on the kernels for inference: q, k, v are scaled by powers of 2 into the FP16 range on the GPU (exact, no host sync, any magnitude) and the output comes back in the input dtype; the decode kernel reads BF16 caches natively. Training with BF16/FP32 inputs outside fp16 autocast is not supported by the kernels (FP16 gradients without loss scaling may underflow; transformers' Trainer itself requires fp16 AMP before Ampere); through HuggingFace it falls back to SDPA.

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

The integration uses our flash kernel for unmasked **prefill** (where memory savings matter) and falls back to PyTorch SDPA for decode (q_len=1) and padded/masked batches.

### What runs on the kernel

The integration decides **per attention call** whether our kernel can run, and never returns a silently wrong result: anything the kernel cannot do runs on a correct fallback (PyTorch SDPA, or an FP32 eager implementation for features SDPA ignores), or raises an explicit error.

| Case | Path |
|---|---|
| Prefill, chunked prefill with a KV cache, static/offloaded caches | kernel |
| Sliding-window layers (Mistral, Gemma 3, Phi-3, ...), also with padding or packing | kernel (`window_size`, blocks outside the window skipped) |
| Packed sequences (`position_ids` restarting at 0, or `cu_seq_lens`), VLM vision encoders (Qwen2/2.5/3-VL, Pixtral, ...) | kernel (varlen) |
| Padded batches (left or right padding, training and generation prefill), encoders and cross-attention with padded inputs | kernel (varlen, unpadded; causal: padded query rows output 0, as FlashAttention-2) |
| BF16 / FP32 inference | kernel (FP16 compute, inputs scaled into range on the GPU) |
| Mixed precision training / inference under FP16 `torch.autocast` | kernel (FP16, as SDPA under autocast) |
| Decode (`q_len == 1`): dynamic / static caches, left padding, sliding window, softcap, sinks, FP16 or BF16 caches | decode kernel (`flash_attn_with_kvcache`, split-KV, the cache read in place) |
| Padding with holes inside a row (prefill; decode handles it on the kernel), unsupported head_dim, attention dropout, BF16/FP32 training outside FP16 autocast, FP32 decode, more than 8 query heads per KV head in decode | SDPA fallback |
| Logit softcapping (Gemma 2) | kernel (`softcap`) |
| Attention sinks (gpt-oss `s_aux`) | kernel (`sinks`, trainable) |
| Additive position bias (T5 family: T5, mT5, LongT5, UDOP, Switch Transformers, ...) | SDPA fallback (the bias in SDPA's mask, as HF's SDPA) |

Check what happened in your run:

```python
import flash_attn_legacy
flash_attn_legacy.reset_fallback_stats()
model.generate(**inputs, max_new_tokens=32)
print(flash_attn_legacy.fallback_stats())   # {} -> prefill and every decode step ran on the kernels
```

Run inference under `torch.inference_mode()` (or `torch.no_grad()`): with BF16/FP32 weights and gradients enabled outside FP16 autocast, calls fall back to SDPA.

### Examples

- `examples/generate.py`: generation (batched, left padding), tokens per second and peak memory, `--compare` with SDPA and eager.
- `examples/finetune.py`: mixed-precision fine-tuning (fp16 autocast + GradScaler) of the attention projections on a local text, `--compare` with SDPA. P100, TinyLlama-1.1B, batch 2 x 1024 tokens: 1,275 tok/s and 8.1 GiB vs SDPA 1,158 tok/s and 14.2 GiB (same loss); at 2048 tokens SDPA runs out of memory (11.8 GiB here).

### Direct API

```python
import torch
from flash_attn_legacy import flash_attention

B, H, N, d = 2, 8, 2048, 64
q = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16)
k = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16)
v = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16)

out = flash_attention(q, k, v, causal=True)
```

### GQA (grouped-query attention)

```python
# 8 query heads, 2 KV heads (4:1 ratio, like Llama 2/3)
q = torch.randn(2, 8, 1024, 64, device='cuda', dtype=torch.float16)
k = torch.randn(2, 2, 1024, 64, device='cuda', dtype=torch.float16)
v = torch.randn(2, 2, 1024, 64, device='cuda', dtype=torch.float16)

out = flash_attention(q, k, v, causal=True)
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

End to end on HuggingFace models (transformers 5.20.2, fp16), against PyTorch SDPA and eager attention on the same inputs:
`python benchmarks/models.py`. Every table (prefill, generate, train, LoRA, padded batches; TinyLlama, Qwen2.5 3B / 14B, Gemma 2,
Mistral 7B, Llama 3.1 8B, Qwen3.5 2B / 4B / 9B), with time ratios and peak memory, is in
[`docs/bench/models.md`](docs/bench/models.md). Each cell: speedup of ours over the faster of SDPA and eager, and the peak memory
saved against the smaller of their peaks; "only ours" = SDPA and eager run out of memory.

| | V100 32 GB | T4 16 GB | P100 16 GB | 1080 Ti 11 GB | 2080 Ti 11 GB |
|---|---|---|---|---|---|
| Prefill, TinyLlama 1.1B, 4k tokens | **3.42x**, -3.9 GiB | **3.89x**, -3.9 GiB | **1.72x**, -3.9 GiB | **1.71x**, -3.9 GiB | **3.28x**, -3.9 GiB |
| Prefill, Qwen2.5-3B, 8k tokens | **2.30x**, -7.7 GiB | only ours | only ours | only ours | only ours |
| Decode, Qwen2.5-3B, B=8, 2k prompt | 1.17x, -3.2 GiB | **2.20x**, -3.2 GiB | **1.75x**, -3.2 GiB | only ours | only ours |
| Decode, compiled static cache, TinyLlama, B=8, 2k | **1.96x**, -8.2 GiB | **3.20x**, -9.0 GiB | - | - | only ours |
| Training step, TinyLlama, 1k tokens | **1.34x**, -3.0 GiB | **1.45x**, -3.0 GiB | 1.07x, -3.0 GiB | 1.01x, -3.0 GiB | **1.70x**, -3.0 GiB |
| Training, gradient checkpointing, TinyLlama, 4k | **2.72x**, -5.9 GiB | **3.22x**, -5.9 GiB | **1.47x**, -5.9 GiB | only ours | only ours |
| Training, checkpointing, padded batch B=4, TinyLlama, 2k | **1.38x** | **1.40x** | **1.39x** | **5.67x** | **1.33x** |
| LoRA r=16, checkpointing, Qwen2.5-3B, 2k | **1.34x** | **1.41x** | 1.03x | 1.02x | only ours |

Longest sequence that fits (ours / the better of SDPA and eager; ">=" = no longer length measured):

| | V100 32 GB | T4 16 GB | P100 16 GB | 1080 Ti 11 GB | 2080 Ti 11 GB |
|---|---|---|---|---|---|
| Prefill, TinyLlama | >= 128k / 8k | >= 128k / 4k | >= 128k / 4k | 64k / 4k | 64k / 4k |
| Prefill, Qwen2.5-3B | >= 128k / 8k | 32k / 4k | 64k / 4k | 32k / 4k | 16k / 4k |
| Generate, Qwen2.5-3B, B=1 (prompt) | >= 64k / >= 8k | 32k / 2k | >= 64k / 2k | 32k / 2k | 32k / 2k |
| Training, checkpointing, TinyLlama | 32k / 4k | 16k / 4k | 16k / 4k | 8k / 2k | 8k / 2k |

Larger models: Llama 3.1 8B on the V100 prefills at least 16k tokens (the longest measured; SDPA / eager 4k; 1.66x and -3.8 GiB at 4k), decodes 8 x 2k prompts
compiled at 1.70x with -7.5 GiB, and LoRA-trains at 8k (others 4k; 1.48x at 4k); on the P100 (16 GB, weights 15 GiB) it prefills
2k tokens and generates from a 2k prompt where SDPA and eager stop at 1k (prefill) / 512 (prompt). Qwen2.5-14B on the V100 prefills 8k (others 2k),
Qwen3.5-9B at least 32k (others 8k).

The attention memory is O(N) in the sequence instead of O(N^2), so contexts up to 32x longer fit (TinyLlama prefill on the T4 /
P100; the same length where the weights or the KV cache, not the attention, set the limit), and where all fit the kernel is
faster or at parity: the rows below 1x in the full tables (0.93-0.99x) are the Qwen3.5 models (host bound), short prefills of
7-8B models on the P100 (matmul bound) and decode with eager steps, within the noise of the shared cloud boxes (see
`docs/bench/models.md`). The training loss agrees with SDPA and eager in every case where more than one fits (largest
difference 0.0026). With gradient checkpointing on the large-vocabulary models (Qwen2.5, Llama 3.1: vocabularies of 128k-152k) the
full-vocabulary fp32 logits of HF's loss take most of the memory, so the length gain shrinks (LoRA on the V100: Qwen2.5-3B 8k
for all, Llama 3.1 8B 8k against 4k) while the speedup stays (LoRA Qwen2.5-3B 4k: 1.53x V100, 2.69x T4); on TinyLlama (32k
vocabulary) checkpointing with our kernel trains 4-8x longer sequences. On the 1080 Ti SDPA with a padding mask runs its fp16
kernel, which sm_61 executes at 1/64 rate: hence 5.67x. Models whose time is not in attention (Qwen3.5: 3 linear-attention
layers in 4, here on transformers' PyTorch reference) run at parity or a little faster (compiled decode 1.02-1.31x), with the memory saving. Decode reads the cache in place with
a split-KV kernel (`flash_attn_with_kvcache`, paged caches too); HF's compiled decode steps (`generate` with a StaticCache,
SM >= 7.0) run without graph breaks.

Task accuracy is unchanged (`python benchmarks/tasks.py`, [`docs/bench/tasks.md`](docs/bench/tasks.md)): Llama 3.1 8B
Instruct on the V100 with lm-evaluation-harness, every document scored with each implementation:

| | ours | SDPA | eager |
|---|---|---|---|
| ARC-Easy, acc (2376) | 0.8220 | 0.8215 | 0.8224 |
| ARC-Challenge, acc_norm (1172) | 0.5589 | 0.5589 | 0.5589 |
| HellaSwag, acc_norm (10042) | 0.7950 | 0.7953 | 0.7953 |
| GSM8K 5-shot, exact match (1319) | 0.7733 | 0.7741 | 0.7756 |
| Passkey retrieval at 4k / 8k / 16k / 32k tokens | 1.00 / 1.00 / 1.00 / 1.00 | 1.00 / OOM / OOM / OOM | 1.00 / OOM / OOM / OOM |

Ours changes as many answers relative to eager as SDPA does (2-21 per task: fp16 near-ties), all within a fraction of the
standard error.

Kernel level, against SDPA's memory-efficient backend on every supported GPU and head dimension: `python benchmarks/benchmark.py`,
results in `docs/bench/<gpu>.json`. Causal, B=2, H=16, N=4096, time relative to SDPA (< 1 = faster):

| | fwd d=64 | fwd d=128 | fwd d=256 | fwd+bwd d=64 | fwd+bwd d=128 | fwd+bwd d=256 |
|---|---|---|---|---|---|---|
| V100 | 0.92 | 1.12 | 0.66 | 0.95 | 1.04 | 1.01 |
| T4 | 0.55 | 0.53 | 0.57 | 0.80 | 0.77 | 0.90 |
| 2080 Ti | 0.68 | 0.68 | 0.61 | 0.83 | 0.77 | 0.97 |
| P100 | 0.79 | 1.25 | 1.79 | 0.66 | 0.70 | 0.94 |
| 1080 Ti | 0.06 | 0.08 | 0.12 | 0.08 | 0.09 | 0.10 |

Above 1: the V100 at d=128 (forward 1.12, fwd+bwd 1.04) and d=256 fwd+bwd (1.01), and the P100 forward at d=128 / 256 (no
tensor cores on Pascal: fp32 FMAs); the
1080 Ti gap comes from the arithmetic: SDPA computes in fp16, which sm_61 runs at 1/64 of the fp32 rate, the kernel in fp32.

## How It Works

### Algorithm

Flash Attention v2 tiles Q into blocks of Br rows and iterates over K/V blocks of Bc columns, using the [online softmax trick](https://arxiv.org/abs/1805.02867) to maintain running max/sum statistics. The full N x N attention score matrix is never materialized in HBM — only small tile-sized buffers live in shared memory.

### Forward pass

All families share one band forward (`csrc/include/flash_attn_fwd_band.h`): each warp owns 16 query rows of the block, the online softmax runs in the log2 domain with warp-shuffle reductions, K/V blocks are prefetched into registers while the current one is computed, masking is applied only to the blocks that need it (sequence tail, causal diagonal) and causal blocks a warp cannot see are skipped.

**Turing (SM 7.5):** `mma.sync` m16n8k8 with `ldmatrix`, as FlashAttention-2: S and O stay in accumulator registers with a known layout, Q in registers (d <= 128), the online softmax runs on the S registers and P packed to half2 is the A operand of P @ V as it is (no shared-memory staging).

**Volta (SM 7.0):** S = Q @ K^T and O += P @ V with `wmma::mma_sync` (16x16x16 tensor cores); S goes through a small per-warp staging tile for the softmax, P is written back as half, O stays in accumulator fragments (registers). Tiles of 128 query rows (8 warps); d=128/256 use larger tiles where the GPU has 96 KB of shared memory per block and 64 KB ones otherwise.

**Pascal (SM 6.0 / 6.1):** fp32 register micro-tiles on CUDA cores: each lane holds 4 rows x Bc/8 keys and d/8 output columns; Q, K, V are converted to fp32 once, when stored to shared memory, so the inner products have no conversions and no fp16 products (no overflow; fp16 math runs at 1/64 rate on SM 6.1).

Tiles per head_dim are chosen by measurement (comments next to `TileTuring_*` / `TileVolta_*` / `TilePascal_*` in `csrc/include/flash_attn_common.h`).

### Backward pass

**D = rowsum(dO * O)** is precomputed in a separate kernel; each block owns Bc keys and loops over the query blocks (and the Q heads of its KV head, GQA; on small grids the heads of a KV head are split over several blocks, whose dK / dV partials are summed after the kernel).

**Turing / Volta:** each warp owns a band of 16 keys and a slice of the d columns of dK / dV; the warps of a band split the queries of S^T and dP^T (no work repeated) and share P^T and dS^T through shared memory; the element-wise steps run on the accumulators; dK, dV accumulate in registers; dS^T is shared so that all warps compute dQ = dS @ K, added to an FP32 buffer with `atomicAdd` (one per element and key block). Turing uses `mma.sync` m16n8k8 + `ldmatrix` (P^T / dS^T are register operands when a band has one warp), Volta WMMA fragments.

**Pascal:** the [Br, Bc] tile of S / P / dP / dS is computed cooperatively in FP32 (128-bit loads), dK, dV accumulate in registers, dQ with one `atomicAdd` per element and key block.

## Testing

```bash
pip install -e ".[test]" --no-build-isolation
pytest tests/ -v

# Performance comparison table
pytest tests/test_performance.py::TestComparisonTable -v -s

# Prefill memory benchmark with real models
pytest tests/test_generation_benchmark.py -v -s -k "prefill"

# HuggingFace integration on tiny random models (no download), vs an FP32 eager reference
pytest tests/test_hf_integration.py -v

# Real models (downloads), vs SDPA / FP32 eager
FA_HF_MODELS="TinyLlama/TinyLlama-1.1B-Chat-v1.0,Qwen/Qwen3-0.6B" pytest tests/test_hf_equivalence.py -v -s

# Run the Pascal (CUDA-core) kernels on any GPU (correctness only, not performance)
FLASH_ATTN_LEGACY_FORCE_ARCH=pascal pytest tests/test_flash_attn.py -v
```

## Limitations

1. **Head dimensions**: d = 32, 64, 72, 80, 96, 128, 256; other sizes run on the SDPA fallback
2. **FP16 compute**: BF16/FP32 for inference only (see above); no native BF16
3. **Kernel features**: causal (bottom-right aligned) or no mask, sliding window (`window_size=(left, right)`, as upstream flash-attn), logit softcapping (`softcap`), attention sinks (`sinks`, one logit per query head), packed segments. No dropout or additive position bias in the kernel yet: those layers run on the fallbacks
4. **Causal mask is bottom-right aligned** when `N_q != N_kv` (query i sees keys `j <= i + N_kv - N_q`, as in upstream flash-attn). Rows with no visible key (`N_kv < N_q`) output zeros
5. **Non-deterministic dQ** (and sink gradient): the backward accumulates them with FP32 `atomicAdd`
6. Known integration corner cases:
   - Pascal + `generate(cache_implementation="static")`: transformers compiles the decoding step with `torch.compile`, and Triton needs SM >= 7.0. Pass `disable_compile=True`
   - decoders with bidirectional attention (`is_causal=False`) and a static cache attend to the empty cache slots, exactly as HF's SDPA does
   - padded batches and static caches get FlashAttention-2's 2D `[batch, keys]` mask; models that index the attention mask as 4D (e.g. Doge) fail with it
   - Llama 4 (chunked attention) is not supported yet: transformers refuses chunked attention for backends whose name contains "flash"

## Roadmap

- [x] GQA/MQA support
- [x] HuggingFace Transformers native integration (`attn_implementation="flash_attn_legacy"`)
- [x] WMMA tensor cores for both Q@K^T and P@V on Volta
- [x] WMMA-based backward on Volta
- [x] Variable-length sequence support (`flash_attn_varlen_func`), forward and backward in one launch each
- [x] Turing (SM 7.5)
- [x] FlashAttention-2 style kernels (warp bands, register accumulators): at or above PyTorch SDPA memory-efficient speed (`docs/bench/`)
- [x] head_dim 72 / 80
- [x] Sliding window in the kernel (forward and backward, out-of-window blocks skipped)
- [x] Logit softcapping in the kernel (forward and backward)
- [x] Attention sinks in the kernel (forward and backward, sink gradient)
- [x] Decode kernel (`flash_attn_with_kvcache`: split-KV, GQA, paged caches, key masks) and HF decode steps on it
- [x] `torch.compile` (dispatcher ops with fake implementations; compiled HF decode steps without graph breaks)
- [x] BF16 / FP32 inference
- [x] End-to-end benchmarks on real models (prefill, decode, training, padded batches, LoRA; up to 14B; 5 GPUs)
- [ ] Decode kernel beyond 8 query rows per KV head: MQA / GQA > 8:1 models and assisted / speculative decoding verification steps
- [ ] Serving: varlen prefill straight from a paged cache, mixed prefill + decode batches (continuous batching in transformers;
  a vLLM backend plugin only on Turing, see `docs/analysis/vllm_feasibility.md`)
- [ ] Dropout and deterministic dQ in the kernel

## References

- Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) (2022)
- Dao, [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) (2023)
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
