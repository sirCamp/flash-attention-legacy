"""
Flash Attention Legacy — Flash Attention v2 for Pascal & Volta GPUs
===================================================================

Supports MHA, GQA (grouped-query), and MQA (multi-query) attention.

Architectures:
    - Pascal: Tesla P100, GTX 1080/Ti (SM 6.0, 6.1)
    - Volta:  Tesla V100, Titan V      (SM 7.0)

Head dimensions: 32, 64, 96, 128, 256

Quick start:
    >>> import torch
    >>> from flash_attn_legacy import flash_attention
    >>>
    >>> # Standard MHA
    >>> q = torch.randn(2, 8, 1024, 64, device='cuda', dtype=torch.float16)
    >>> k = torch.randn(2, 8, 1024, 64, device='cuda', dtype=torch.float16)
    >>> v = torch.randn(2, 8, 1024, 64, device='cuda', dtype=torch.float16)
    >>> out = flash_attention(q, k, v, is_causal=True)
    >>>
    >>> # GQA (8 Q heads, 2 KV heads)
    >>> k_gqa = torch.randn(2, 2, 1024, 64, device='cuda', dtype=torch.float16)
    >>> v_gqa = torch.randn(2, 2, 1024, 64, device='cuda', dtype=torch.float16)
    >>> out_gqa = flash_attention(q, k_gqa, v_gqa, is_causal=True)
"""

__version__ = "0.5.0"

from .flash_attn import (
    flash_attention,
    flash_attn_varlen_func,
    FlashAttention,
    FlashMultiHeadAttention,
)

from .utils import (
    check_gpu_compatibility,
    get_device_info,
)

from .integrations import register

__all__ = [
    "flash_attention",
    "flash_attn_varlen_func",
    "FlashAttention",
    "FlashMultiHeadAttention",
    "check_gpu_compatibility",
    "get_device_info",
    "register",
]

# Auto-register with HuggingFace transformers if available
register()
