"""
Flash Attention Legacy — PyTorch autograd interface
Supports MHA, GQA, and MQA on Pascal (SM 6.x) and Volta (SM 7.0) GPUs
"""

import math
import numbers
import warnings
import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Optional, Tuple

from .utils import check_head_dim, fp16_autocast_active

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
    """Convert non-fp16 inputs to fp16.

    Under fp16 autocast this is autocast's own semantics (as SDPA there): a plain
    cast, gradients allowed (the GradScaler that fp16 autocast training needs
    handles underflow and inf). Otherwise the values are range-checked
    (FP16RangeError, one host sync) and gradients are not supported: without loss
    scaling, bf16/fp32 training gradients may underflow in fp16 (docs/PLAN.md P6.2)."""
    if q.dtype == torch.float16:
        return q, k, v
    if fp16_autocast_active():
        return q.half(), k.half(), v.half()
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


def _check_window(window_size):
    """(left, right) as two ints >= -1 (-1 = unbounded)."""
    if (not isinstance(window_size, (tuple, list)) or len(window_size) != 2
            or not all(isinstance(w, numbers.Integral) and not isinstance(w, bool) for w in window_size)):
        raise ValueError(f"window_size must be a pair of ints, got {window_size!r}")
    left, right = int(window_size[0]), int(window_size[1])
    if left < -1 or right < -1:
        raise ValueError(f"window_size entries must be >= -1 (-1 = unbounded), got {window_size!r}")
    return left, right


def _check_softcap(softcap):
    """Logit softcap as a float >= 0 (0 = none), within the float32 range the
    kernels compute it in (a normal float with a finite 1 / c)."""
    if not isinstance(softcap, numbers.Real) or isinstance(softcap, bool) \
            or not math.isfinite(softcap) or softcap < 0:
        raise ValueError(f"softcap must be a finite number >= 0 (0 = no softcapping), got {softcap!r}")
    f32 = torch.finfo(torch.float32)
    if softcap != 0 and not (f32.tiny <= softcap <= f32.max and 1.0 / softcap <= f32.max):
        raise ValueError(f"softcap out of the float32 range, got {softcap!r}")
    return float(softcap)


def _check_sinks(sinks, num_heads, device):
    """Attention sinks: None, or one logit per query head ([num_heads], floating
    point, on `device`)."""
    if sinks is None:
        return None
    if not isinstance(sinks, torch.Tensor) or not sinks.is_floating_point() or sinks.shape != (num_heads,) \
            or sinks.device != device:
        got = (f"{sinks.dtype} {tuple(sinks.shape)} on {sinks.device}" if isinstance(sinks, torch.Tensor)
               else type(sinks).__name__)
        raise ValueError(f"sinks must be a floating-point tensor of shape [num_heads_q] = [{num_heads}] "
                         f"on {device}, got {got}")
    return sinks


class FlashAttentionFunc(Function):
    """
    Autograd function for Flash Attention v2.
    Supports GQA: Q [B, H_q, N, d], K/V [B, H_kv, N, d] where H_q % H_kv == 0.
    """

    @staticmethod
    def forward(ctx, q, k, v, softmax_scale, causal, window_size, softcap, sinks):
        _check_inputs(q, k, v, ndim=4)
        d = q.shape[-1]
        check_head_dim(d)

        H_q, H_kv = q.shape[1], k.shape[1]
        if H_q % H_kv != 0:
            raise ValueError(f"H_q ({H_q}) must be divisible by H_kv ({H_kv})")
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        o, lse = flash_attn_legacy_cuda.forward(q, k, v, softmax_scale, causal, *window_size, softcap, sinks)

        ctx.save_for_backward(q, k, v, o, lse, sinks)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size, ctx.softcap = window_size, softcap
        return o

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, o, lse, sinks = ctx.saved_tensors
        grad_output = grad_output.contiguous()

        dq, dk, dv, dsinks = flash_attn_legacy_cuda.backward(
            grad_output, q, k, v, o, lse,
            ctx.softmax_scale, ctx.causal, *ctx.window_size, ctx.softcap,
            sinks if ctx.needs_input_grad[7] else None,  # None: no sink gradient computed
        )
        return dq, dk, dv, None, None, None, None, _sinks_grad(dsinks, sinks)


def _sinks_grad(dsinks, sinks):
    """fp32 gradient of the sinks from the kernel -> dtype of the sinks (None when
    not computed)."""
    return None if dsinks is None else dsinks.to(sinks.dtype)


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
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    sinks: Optional[torch.Tensor] = None,
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
        q: Query [B, H_q, N, d] float16. bfloat16 / float32 are converted to fp16:
           under fp16 autocast like autocast does (training allowed, no range
           check); otherwise for inference only, range-checked (FP16RangeError)
        k: Key   [B, H_kv, N, d], same dtype as q
        v: Value [B, H_kv, N, d], same dtype as q
        softmax_scale: default 1/sqrt(d)
        causal: apply a causal mask
        window_size: (left, right) local attention, as upstream flash-attn: query
            i sees keys [i + N_kv - N_q - left, i + N_kv - N_q + right]; -1 =
            unbounded. With causal, right is 0. A sliding window of W keys back
            (HF sliding_window = W) is (W - 1, 0). Blocks outside the window are
            skipped, forward and backward.
        softcap: logit softcapping c > 0, as upstream flash-attn and Gemma 2: the
            scaled scores x become c * tanh(x / c) before the softmax; 0 = none
        sinks: attention sinks [H_q] (gpt-oss, HF kwarg s_aux): one logit per
            query head added to every row's softmax, with no value (it takes
            probability away from the keys); differentiable. Finite values, or
            -inf for a head without sink.
        is_causal: deprecated alias of `causal`

    Returns:
        [B, H_q, N, d] in the dtype of q
    """
    causal = _resolve_causal(causal, is_causal)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    _check_inputs(q, k, v, ndim=4)
    window_size, softcap = _check_window(window_size), _check_softcap(softcap)
    sinks = _check_sinks(sinks, q.shape[1], q.device)
    out = FlashAttentionFunc.apply(*_to_fp16(q, k, v), softmax_scale, causal, window_size, softcap, sinks)
    return out if q.dtype == torch.float16 else out.to(q.dtype)


class FlashAttnVarlenFunc(Function):
    """
    Autograd function for variable-length (packed) Flash Attention.
    Forward and backward use native CUDA kernels (one launch for all sequences).
    """

    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, softmax_scale, causal, window_size, softcap, sinks, validate):
        _check_inputs(q, k, v, ndim=3)
        check_head_dim(q.shape[-1])

        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        cu_seqlens_q = cu_seqlens_q.contiguous().to(torch.int32)
        cu_seqlens_k = cu_seqlens_k.contiguous().to(torch.int32)

        o, lse = flash_attn_legacy_cuda.forward_varlen(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, softmax_scale, causal, *window_size, softcap, sinks, validate,
        )

        ctx.save_for_backward(q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k, sinks)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size, ctx.softcap = window_size, softcap
        ctx.max_seqlen_q, ctx.max_seqlen_k = max_seqlen_q, max_seqlen_k
        ctx.validate = validate
        return o

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k, sinks = ctx.saved_tensors
        grad_output = grad_output.contiguous()

        dq, dk, dv, dsinks = flash_attn_legacy_cuda.backward_varlen(
            grad_output, q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k,
            ctx.max_seqlen_q, ctx.max_seqlen_k, ctx.softmax_scale, ctx.causal, *ctx.window_size,
            ctx.softcap, sinks if ctx.needs_input_grad[11] else None, ctx.validate,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, _sinks_grad(dsinks, sinks), None


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
    validate: bool = True,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    sinks: Optional[torch.Tensor] = None,
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
        window_size: (left, right) local attention within each sequence, as in
            flash_attention
        softcap: logit softcapping, as in flash_attention
        sinks: attention sinks [num_heads], as in flash_attention
        validate: check cu_seqlens (start at 0, non-decreasing, end at the totals)
            on a host copy, one CPU-GPU sync per call. Pass False only for
            cu_seqlens known to be valid (no sync; e.g. the HuggingFace
            integration, which builds them): invalid ones then make the kernels
            read and write out of bounds (memory corruption, illegal address).

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
    window_size, softcap = _check_window(window_size), _check_softcap(softcap)
    sinks = _check_sinks(sinks, q.shape[1], q.device)
    out = FlashAttnVarlenFunc.apply(
        *_to_fp16(q, k, v), cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, softmax_scale, causal, window_size, softcap, sinks, validate,
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


_KVCACHE_UNSUPPORTED = ("k", "v", "rotary_cos", "rotary_sin", "cache_batch_idx", "cache_leftpad", "block_table",
                        "alibi_slopes")


def flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    cache_seqlens=None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    rotary_interleaved: bool = True,
    alibi_slopes: Optional[torch.Tensor] = None,
    num_splits: int = 0,
    return_softmax_lse: bool = False,
    *,
    sinks: Optional[torch.Tensor] = None,
):
    """
    Attention of a few new queries against a KV cache (decode), the upstream
    flash-attn ``flash_attn_with_kvcache`` with the same positional order.
    Inference only. Split-KV decode kernel on CUDA cores (every supported GPU).

    Args:
        q: [B, q_len, H_q, d] float16 or bfloat16 (q_len small: decode, speculation)
        k_cache, v_cache: [B, max_len, H_kv, d], q's dtype, read in place: any
            strides with a contiguous last dimension and 16-byte aligned rows
            (e.g. a HuggingFace cache [B, H_kv, N, d] passed as ``.transpose(1, 2)``)
        cache_seqlens: int or int32 [B]: end of the valid keys of each row
            (default max_len); keys [0, cache_seqlens) are attended. Tensor
            values are clamped to [0, max_len] on the GPU (no sync); an int
            outside that range raises
        softmax_scale: default 1/sqrt(d)
        causal: bottom-right aligned (query i of q_len sees keys
            <= cache_seqlens - q_len + i); irrelevant for q_len == 1
        window_size, softcap: as flash_attention
        num_splits: key splits per (row, KV head): 0 = automatic (enough blocks
            to fill the GPU at batch 1 and long contexts; sized by the cache
            capacity: with a large preallocated cache holding few keys pass a
            small value), >= 1 forced (capped at 128; the fp32 partials take
            num_splits * B * H_q * q_len * d * 4 bytes)
        return_softmax_lse: also return the logsumexp [B, H_q, q_len] (fp32;
            -inf for rows with no visible key nor sink)
        sinks: attention sinks [H_q] (keyword only, an extension)
        k, v, rotary_cos, rotary_sin, cache_batch_idx, cache_leftpad, block_table,
        alibi_slopes: not supported yet (NotImplementedError when given)

    Returns:
        out [B, q_len, H_q, d] in q's dtype, or (out, lse) with return_softmax_lse

    Differences from upstream: raises if autograd would need it (upstream returns
    a tensor without grad_fn); rows with no visible key give lse = -inf.
    """
    given = locals()
    unsupported = [name for name in _KVCACHE_UNSUPPORTED if given[name] is not None]
    if unsupported:
        raise NotImplementedError(f"flash_attn_with_kvcache: {', '.join(unsupported)} not supported yet")
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k_cache, v_cache)):
        raise RuntimeError("flash_attn_with_kvcache is inference only (no autograd): run under torch.no_grad()")
    if q.dim() != 4 or k_cache.dim() != 4:
        raise ValueError("q must be [B, q_len, H_q, d] and k_cache / v_cache [B, max_len, H_kv, d]")
    B, max_len = q.shape[0], k_cache.shape[1]
    if cache_seqlens is None:
        cache_seqlens = max_len
    if isinstance(cache_seqlens, numbers.Integral) and not isinstance(cache_seqlens, bool):
        if not 0 <= cache_seqlens <= max_len:
            raise ValueError(f"cache_seqlens={cache_seqlens} outside [0, max_len={max_len}]")
        cache_seqlens = torch.full((B,), int(cache_seqlens), dtype=torch.int32, device=q.device)
    else:
        cache_seqlens = cache_seqlens.to(device=q.device, dtype=torch.int32)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    if not isinstance(num_splits, numbers.Integral) or isinstance(num_splits, bool) or num_splits < 0:
        raise ValueError(f"num_splits must be an int >= 0 (0 = automatic), got {num_splits!r}")
    window_size, softcap = _check_window(window_size), _check_softcap(softcap)
    sinks = _check_sinks(sinks, q.shape[2], q.device)
    out, lse = flash_attn_legacy_cuda.with_kvcache(q, k_cache, v_cache, cache_seqlens, softmax_scale, causal,
                                                   *window_size, softcap, sinks, num_splits)
    return (out, lse) if return_softmax_lse else out
