"""
Flash Attention Legacy — Flash Attention v2 for Pascal & Volta GPUs
===================================================================

Supports MHA, GQA (grouped-query), and MQA (multi-query) attention.

Architectures:
    - Pascal: Tesla P100, GTX 1080/Ti (SM 6.0, 6.1)
    - Volta:  Tesla V100, Titan V      (SM 7.0)

Head dimensions: 32, 64, 72, 80, 96, 128, 256

Quick start:
    >>> import torch
    >>> from flash_attn_legacy import flash_attention
    >>>
    >>> # Standard MHA
    >>> q = torch.randn(2, 8, 1024, 64, device='cuda', dtype=torch.float16)
    >>> k = torch.randn(2, 8, 1024, 64, device='cuda', dtype=torch.float16)
    >>> v = torch.randn(2, 8, 1024, 64, device='cuda', dtype=torch.float16)
    >>> out = flash_attention(q, k, v, causal=True)
    >>>
    >>> # GQA (8 Q heads, 2 KV heads)
    >>> k_gqa = torch.randn(2, 2, 1024, 64, device='cuda', dtype=torch.float16)
    >>> v_gqa = torch.randn(2, 2, 1024, 64, device='cuda', dtype=torch.float16)
    >>> out_gqa = flash_attention(q, k_gqa, v_gqa, causal=True)

FP16 linear layers on the P100 (SM 6.0 only, off by default): fused_linear(x, W, bias)
and patch_linear(model), or FLASH_ATTN_LEGACY_LINEAR=1 for HF models loaded with
attn_implementation="flash_attn_legacy".
"""

__version__ = "1.0.0"

from .flash_attn import (
    flash_attention,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    FlashAttention,
    FlashMultiHeadAttention,
)

from .utils import (
    check_gpu_compatibility,
    get_device_info,
)

from .integrations import register, fallback_stats, reset_fallback_stats
from .linear import fused_linear, patch_linear
from . import gated_delta as _gated_delta  # noqa: F401 (registers the conv op's autograd)

__all__ = [
    "flash_attention",
    "flash_attn_varlen_func",
    "flash_attn_with_kvcache",
    "FlashAttention",
    "FlashMultiHeadAttention",
    "check_gpu_compatibility",
    "get_device_info",
    "register",
    "fallback_stats",
    "reset_fallback_stats",
    "fused_linear",
    "patch_linear",
]

# Auto-register with HuggingFace transformers if available
register()
