"""
Flash Attention Legacy — PyTorch autograd interface
Supports MHA, GQA, and MQA on Pascal (SM 6.x) and Volta (SM 7.0) GPUs
"""

import math
import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Optional

try:
    import flash_attn_legacy_cuda
except ImportError as e:
    raise ImportError(
        "flash_attn_legacy CUDA extension not found. "
        "Install with: pip install -e .\n"
        f"Original error: {e}"
    )


class FlashAttentionFunc(Function):
    """
    Autograd function for Flash Attention v2.
    Supports GQA: Q [B, H_q, N, d], K/V [B, H_kv, N, d] where H_q % H_kv == 0.
    """

    @staticmethod
    def forward(ctx, q, k, v, softmax_scale, is_causal):
        assert q.dtype == torch.float16, f"Q must be float16, got {q.dtype}"
        assert k.dtype == torch.float16 and v.dtype == torch.float16
        assert q.is_cuda and q.dim() == 4
        d = q.shape[-1]
        assert d in (64, 128), f"head_dim must be 64 or 128, got {d}"

        H_q, H_kv = q.shape[1], k.shape[1]
        assert H_q % H_kv == 0, f"H_q ({H_q}) must be divisible by H_kv ({H_kv})"

        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        o, lse = flash_attn_legacy_cuda.forward(q, k, v, softmax_scale, is_causal)

        ctx.save_for_backward(q, k, v, o, lse)
        ctx.softmax_scale = softmax_scale
        ctx.is_causal = is_causal
        return o

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, o, lse = ctx.saved_tensors
        grad_output = grad_output.contiguous()

        dq, dk, dv = flash_attn_legacy_cuda.backward(
            grad_output, q, k, v, o, lse,
            ctx.softmax_scale, ctx.is_causal,
        )
        return dq, dk, dv, None, None


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Flash Attention v2 for Pascal/Volta GPUs.

    Supports MHA, GQA, and MQA:
      - MHA: Q [B, H, N, d], K [B, H, N, d], V [B, H, N, d]
      - GQA: Q [B, H_q, N, d], K [B, H_kv, N, d], V [B, H_kv, N, d]
             where H_q is divisible by H_kv
      - MQA: same as GQA with H_kv = 1

    Args:
        q: Query [B, H_q, N, d] float16
        k: Key   [B, H_kv, N, d] float16
        v: Value [B, H_kv, N, d] float16
        softmax_scale: default 1/sqrt(d)
        is_causal: causal mask

    Returns:
        [B, H_q, N, d] float16
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    return FlashAttentionFunc.apply(q, k, v, softmax_scale, is_causal)


class FlashAttention(nn.Module):
    """Drop-in Flash Attention module. Supports GQA/MQA."""

    def __init__(self, head_dim: int = 64, causal: bool = False,
                 softmax_scale: Optional[float] = None):
        super().__init__()
        assert head_dim in (64, 128)
        self.head_dim = head_dim
        self.causal = causal
        self.softmax_scale = softmax_scale or (1.0 / math.sqrt(head_dim))

    def forward(self, q, k, v):
        return flash_attention(q, k, v, self.softmax_scale, self.causal)


class FlashMultiHeadAttention(nn.Module):
    """
    Multi-head attention with Flash Attention backend.
    Supports standard MHA, GQA, and MQA.

    Args:
        embed_dim: Total embedding dimension
        num_heads: Number of query heads (H_q)
        num_kv_heads: Number of KV heads (H_kv). Default = num_heads (standard MHA).
                      Set to 1 for MQA, or any divisor of num_heads for GQA.
        causal: Apply causal mask
        bias: Use bias in linear projections
        dropout: Dropout rate (applied post-attention)

    Example:
        # Standard MHA
        mha = FlashMultiHeadAttention(512, num_heads=8)
        # GQA (LLaMA-style, 8 Q heads, 2 KV heads)
        gqa = FlashMultiHeadAttention(512, num_heads=8, num_kv_heads=2)
        # MQA
        mqa = FlashMultiHeadAttention(512, num_heads=8, num_kv_heads=1)
    """

    def __init__(self, embed_dim: int, num_heads: int, num_kv_heads: Optional[int] = None,
                 causal: bool = False, bias: bool = True, dropout: float = 0.0):
        super().__init__()
        if num_kv_heads is None:
            num_kv_heads = num_heads

        assert embed_dim % num_heads == 0
        assert num_heads % num_kv_heads == 0, (
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim in (64, 128), f"head_dim={self.head_dim}, must be 64 or 128"

        self.causal = causal
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, key_value: Optional[torch.Tensor] = None):
        """
        Args:
            x: [B, N, embed_dim]
            key_value: Optional [B, N_kv, embed_dim] for cross-attention
        Returns:
            [B, N, embed_dim]
        """
        B, N, _ = x.shape
        kv = key_value if key_value is not None else x

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(B, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)

        orig_dtype = q.dtype
        if q.dtype != torch.float16:
            q, k, v = q.half(), k.half(), v.half()

        out = flash_attention(q, k, v, self.scale, self.causal)

        if orig_dtype != torch.float16:
            out = out.to(orig_dtype)

        out = out.transpose(1, 2).contiguous().view(B, N, self.embed_dim)
        return self.dropout(self.out_proj(out))
