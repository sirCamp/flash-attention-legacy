"""
Flash Attention Legacy — PyTorch autograd interface
Supports MHA, GQA, and MQA on Pascal (SM 6.x) and Volta (SM 7.0) GPUs
"""

import math
import warnings
import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Optional

from .utils import check_head_dim

try:
    import flash_attn_legacy_cuda
except ImportError as e:
    raise ImportError(
        "flash_attn_legacy CUDA extension not found. "
        "Install with: pip install -e .\n"
        f"Original error: {e}"
    )


# The kernels compute in fp16. bf16 / fp32 inputs are converted (exact within the
# fp16 range up to fp16 precision) and the output is converted back.
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_FP16_MAX = 65504.0


class FP16RangeError(OverflowError):
    """bf16/fp32 inputs hold values beyond the fp16 range the kernels compute in."""


def _check_fp16_range(*tensors):
    """Raise FP16RangeError unless every value fits the fp16 range (one host sync;
    inf-norm reduction, no full-size temporaries). NaN also raises: it would hide
    an overflow in the same tensor; callers fall back to their original dtype."""
    amax = torch.stack([torch.linalg.vector_norm(t.detach(), float("inf")).float()
                        for t in tensors]).max()
    if not bool(amax <= _FP16_MAX):
        raise FP16RangeError(f"flash_attn_legacy: {tensors[0].dtype} inputs exceed the fp16 "
                             f"range or contain NaN (max |x| = {amax.item():.4g})")


def _to_fp16(q, k, v):
    """Convert non-fp16 inputs to fp16, range-checked. Gradients through the
    conversion are not supported: without loss scaling, bf16/fp32 training
    gradients may underflow in fp16."""
    if q.dtype == torch.float16:
        return q, k, v
    if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad):
        raise NotImplementedError(
            f"flash_attn_legacy: training with {q.dtype} inputs is not supported (the kernels "
            "compute in fp16 and gradients may underflow without loss scaling); use fp16 "
            "with a GradScaler, or run inference under torch.inference_mode()")
    _check_fp16_range(q, k, v)
    return q.half(), k.half(), v.half()


def _check_inputs(q, k, v, ndim):
    """Validate dtype/device/rank of q, k, v (raises, so it survives python -O)."""
    for name, t in (("q", q), ("k", k), ("v", v)):
        if t.dtype not in _SUPPORTED_DTYPES:
            raise TypeError(f"{name} must be one of {_SUPPORTED_DTYPES}, got {t.dtype}")
        if t.dtype != q.dtype:
            raise TypeError(f"q, k, v must have the same dtype, got {q.dtype} and {t.dtype}")
        if not t.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if t.dim() != ndim:
            raise ValueError(f"{name} must be {ndim}-D, got shape {tuple(t.shape)}")
    if not (q.device == k.device == v.device):
        raise ValueError("q, k, v must be on the same device")


class FlashAttentionFunc(Function):
    """
    Autograd function for Flash Attention v2.
    Supports GQA: Q [B, H_q, N, d], K/V [B, H_kv, N, d] where H_q % H_kv == 0.
    """

    @staticmethod
    def forward(ctx, q, k, v, softmax_scale, causal):
        _check_inputs(q, k, v, ndim=4)
        d = q.shape[-1]
        check_head_dim(d)

        H_q, H_kv = q.shape[1], k.shape[1]
        if H_q % H_kv != 0:
            raise ValueError(f"H_q ({H_q}) must be divisible by H_kv ({H_kv})")
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        o, lse = flash_attn_legacy_cuda.forward(q, k, v, softmax_scale, causal)

        ctx.save_for_backward(q, k, v, o, lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, o, lse = ctx.saved_tensors
        grad_output = grad_output.contiguous()

        dq, dk, dv = flash_attn_legacy_cuda.backward(
            grad_output, q, k, v, o, lse,
            ctx.softmax_scale, ctx.causal,
        )
        return dq, dk, dv, None, None


def _resolve_causal(causal: bool, is_causal: Optional[bool]) -> bool:
    """Support the deprecated `is_causal` keyword alongside `causal`."""
    if is_causal is None:
        return causal
    warnings.warn(
        "`is_causal` is deprecated, use `causal` (same name as flash_attn_varlen_func "
        "and the upstream flash-attn API)",
        DeprecationWarning,
        stacklevel=3,
    )
    return is_causal


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    *,
    is_causal: Optional[bool] = None,
) -> torch.Tensor:
    """
    Flash Attention v2 for Pascal/Volta GPUs.

    Supports MHA, GQA, and MQA:
      - MHA: Q [B, H, N, d], K [B, H, N, d], V [B, H, N, d]
      - GQA: Q [B, H_q, N, d], K [B, H_kv, N, d], V [B, H_kv, N, d]
             where H_q is divisible by H_kv
      - MQA: same as GQA with H_kv = 1

    Args:
        q: Query [B, H_q, N, d] float16 (bfloat16 / float32 accepted for inference:
           converted to fp16, range-checked, FP16RangeError if out of range)
        k: Key   [B, H_kv, N, d], same dtype as q
        v: Value [B, H_kv, N, d], same dtype as q
        softmax_scale: default 1/sqrt(d)
        causal: apply a causal mask
        is_causal: deprecated alias of `causal`

    Returns:
        [B, H_q, N, d] in the dtype of q
    """
    causal = _resolve_causal(causal, is_causal)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    _check_inputs(q, k, v, ndim=4)
    out = FlashAttentionFunc.apply(*_to_fp16(q, k, v), softmax_scale, causal)
    return out if q.dtype == torch.float16 else out.to(q.dtype)


class FlashAttnVarlenFunc(Function):
    """
    Autograd function for variable-length (packed) Flash Attention.
    Forward and backward use native CUDA kernels (one launch for all sequences).
    """

    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, softmax_scale, causal):
        _check_inputs(q, k, v, ndim=3)
        check_head_dim(q.shape[-1])

        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        cu_seqlens_q = cu_seqlens_q.contiguous().to(torch.int32)
        cu_seqlens_k = cu_seqlens_k.contiguous().to(torch.int32)

        o, lse = flash_attn_legacy_cuda.forward_varlen(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, softmax_scale, causal,
        )

        ctx.save_for_backward(q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.max_seqlen_q, ctx.max_seqlen_k = max_seqlen_q, max_seqlen_k
        return o

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        grad_output = grad_output.contiguous()

        dq, dk, dv = flash_attn_legacy_cuda.backward_varlen(
            grad_output, q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k,
            ctx.max_seqlen_q, ctx.max_seqlen_k, ctx.softmax_scale, ctx.causal,
        )
        return dq, dk, dv, None, None, None, None, None, None


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """
    Variable-length Flash Attention for packed sequences.

    Instead of padding all sequences to the same length, sequences are
    concatenated along the token dimension and cu_seqlens marks the
    boundaries. This avoids wasting compute/memory on padding tokens.

    Forward and backward each run as one kernel launch for all sequences.

    Args:
        q: [total_q, num_heads, head_dim] float16 (bfloat16 / float32 for inference,
           as in flash_attention)
        k: [total_k, num_heads_k, head_dim], same dtype as q
        v: [total_k, num_heads_k, head_dim], same dtype as q
        cu_seqlens_q: [batch_size + 1] int32 — cumulative sequence lengths for Q
        cu_seqlens_k: [batch_size + 1] int32 — cumulative sequence lengths for K/V
        max_seqlen_q: max sequence length in Q
        max_seqlen_k: max sequence length in K/V
        softmax_scale: default 1/sqrt(head_dim)
        causal: apply causal mask

    Returns:
        [total_q, num_heads, head_dim] in the dtype of q

    Example:
        # Batch of 3 sequences: lengths 128, 256, 64
        total = 128 + 256 + 64  # = 448
        q = torch.randn(total, 8, 64, device='cuda', dtype=torch.float16)
        k = torch.randn(total, 8, 64, device='cuda', dtype=torch.float16)
        v = torch.randn(total, 8, 64, device='cuda', dtype=torch.float16)
        cu_seqlens = torch.tensor([0, 128, 384, 448], dtype=torch.int32, device='cuda')
        out = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, 256, 256, causal=True)
    """
    d = q.shape[-1]

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)

    _check_inputs(q, k, v, ndim=3)
    out = FlashAttnVarlenFunc.apply(
        *_to_fp16(q, k, v), cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, softmax_scale, causal,
    )
    return out if q.dtype == torch.float16 else out.to(q.dtype)


class FlashAttention(nn.Module):
    """Drop-in Flash Attention module. Supports GQA/MQA."""

    def __init__(self, head_dim: int = 64, causal: bool = False,
                 softmax_scale: Optional[float] = None):
        super().__init__()
        check_head_dim(head_dim)
        self.head_dim = head_dim
        self.causal = causal
        self.softmax_scale = (
            softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
        )

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

        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        check_head_dim(self.head_dim)

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
            key_value: Optional [B, N_kv, embed_dim] for cross-attention.
                       The causal mask is bottom-right aligned when N_kv != N.
        Returns:
            [B, N, embed_dim]
        """
        B, N, _ = x.shape
        kv = key_value if key_value is not None else x

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(B, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)

        orig_dtype = q.dtype
        if orig_dtype != torch.float16 and torch.is_grad_enabled():
            # Training a non-fp16 module: kept for compatibility (fp32 modules by
            # default), but gradients are fp16 without loss scaling.
            warnings.warn("FlashMultiHeadAttention: training with non-fp16 weights computes "
                          "attention gradients in fp16 without loss scaling (they may "
                          "underflow); prefer fp16 + GradScaler", UserWarning, stacklevel=2)
            _check_fp16_range(q, k, v)
            q, k, v = q.half(), k.half(), v.half()

        out = flash_attention(q, k, v, self.scale, self.causal)  # bf16/fp32: range-checked
        out = out.to(orig_dtype)

        out = out.transpose(1, 2).contiguous().view(B, N, self.embed_dim)
        return self.dropout(self.out_proj(out))
