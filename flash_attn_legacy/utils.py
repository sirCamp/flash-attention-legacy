"""
GPU detection, compatibility checking, and benchmarking utilities.
"""

import torch
from typing import Dict, Optional

_SM_TO_ARCH = {
    60: "Pascal (GP100)", 61: "Pascal (GP10x)", 62: "Pascal (Jetson)",
    70: "Volta", 72: "Volta (Jetson)",
    75: "Turing", 80: "Ampere", 86: "Ampere",
    89: "Ada Lovelace", 90: "Hopper",
}


def get_device_info(device: Optional[int] = None) -> Dict:
    """Get detailed GPU information."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    if device is None:
        device = torch.cuda.current_device()

    props = torch.cuda.get_device_properties(device)
    sm = props.major * 10 + props.minor
    arch = _SM_TO_ARCH.get(sm, f"Unknown (SM {sm})")

    if sm >= 70:
        kernel_info = "Volta path: wmma tensor cores"
    elif sm >= 60:
        kernel_info = "Pascal path: half2 CUDA cores"
    else:
        kernel_info = "Unsupported (SM >= 6.0 required)"

    return {
        "name": props.name,
        "sm_version": sm,
        "arch": arch,
        "total_memory_gb": round(props.total_memory / (1024**3), 2),
        "shared_mem_per_block": getattr(props, 'max_shared_memory_per_block',
                                        getattr(props, 'shared_memory_per_block', 0)),
        "multiprocessor_count": props.multi_processor_count,
        "has_tensor_cores": sm >= 70,
        "kernel_path": kernel_info,
        "supported": sm >= 60,
    }


def check_gpu_compatibility(device: Optional[int] = None, verbose: bool = True) -> bool:
    """Check if current GPU supports flash_attn_legacy."""
    if not torch.cuda.is_available():
        if verbose:
            print("CUDA is not available")
        return False

    info = get_device_info(device)
    if verbose:
        print(f"GPU: {info['name']}")
        print(f"Architecture: {info['arch']} (SM {info['sm_version']})")
        print(f"Memory: {info['total_memory_gb']} GB")
        print(f"SMs: {info['multiprocessor_count']}")
        print(f"Tensor Cores: {'yes' if info['has_tensor_cores'] else 'no'}")
        print(f"Shared Mem/Block: {info['shared_mem_per_block']} bytes")
        print(f"Kernel path: {info['kernel_path']}")
        if info['supported']:
            print(f"\n[OK] {info['name']} is supported")
        else:
            print(f"\n[FAIL] SM {info['sm_version']} not supported (need >= 6.0)")
    return info['supported']


def benchmark_flash_vs_standard(
    batch_size: int = 2, num_heads: int = 8, seq_len: int = 2048,
    head_dim: int = 64, is_causal: bool = False,
    num_warmup: int = 10, num_iters: int = 100
) -> Dict:
    """Benchmark Flash Attention vs PyTorch standard attention."""
    from .flash_attn import flash_attention

    device = torch.device("cuda")
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    def std_attn(q, k, v):
        scale = 1.0 / (head_dim ** 0.5)
        s = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        if is_causal:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
            s.masked_fill_(mask, float('-inf'))
        return torch.matmul(torch.softmax(s, dim=-1).half(), v)

    # Warmup
    for _ in range(num_warmup):
        flash_attention(q, k, v, is_causal=is_causal)
        std_attn(q, k, v)
    torch.cuda.synchronize()

    # Flash
    torch.cuda.reset_peak_memory_stats()
    s_ev, e_ev = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s_ev.record()
    for _ in range(num_iters):
        flash_attention(q, k, v, is_causal=is_causal)
    e_ev.record()
    torch.cuda.synchronize()
    flash_ms = s_ev.elapsed_time(e_ev) / num_iters
    flash_mem = torch.cuda.max_memory_allocated() / 1e6

    # Standard
    torch.cuda.reset_peak_memory_stats()
    s_ev.record()
    for _ in range(num_iters):
        std_attn(q, k, v)
    e_ev.record()
    torch.cuda.synchronize()
    std_ms = s_ev.elapsed_time(e_ev) / num_iters
    std_mem = torch.cuda.max_memory_allocated() / 1e6

    return {
        "config": f"B={batch_size} H={num_heads} N={seq_len} d={head_dim}",
        "flash_ms": round(flash_ms, 3), "std_ms": round(std_ms, 3),
        "speedup": round(std_ms / flash_ms, 2) if flash_ms > 0 else 0,
        "flash_mem_mb": round(flash_mem, 1), "std_mem_mb": round(std_mem, 1),
        "mem_savings_pct": round((1 - flash_mem / std_mem) * 100, 1) if std_mem > 0 else 0,
    }
