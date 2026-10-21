# CPU-only probe: emulates the decision logic of flash_attn_legacy (mask builder + attention fn)
# with a pure-torch stand-in for the kernel, and compares against eager. Logs what HF passes.
import math, torch, warnings, sys
warnings.filterwarnings("ignore")
import transformers
from transformers import masking_utils as mu, AutoConfig
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
torch.manual_seed(0)
NAME = sys.argv[1] if len(sys.argv) > 1 else "flash_attn_legacy"
LOG = []

def kernel_ref(q, k, v, scale, causal):  # what the CUDA kernel computes: plain (bottom-right) causal / full, GQA
    rep = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(rep, 1); v = v.repeat_interleave(rep, 1)
    s = (q.float() @ k.float().transpose(-1, -2)) * scale
    if causal:
        nq, nk = q.shape[2], k.shape[2]
        m = torch.ones(nq, nk, dtype=torch.bool).tril(nk - nq)
        s = s.masked_fill(~m, float("-inf"))
    return (s.softmax(-1) @ v.float()).to(q.dtype)

def attn_fn(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kw):
    B, H, nq, d = query.shape; nk = key.shape[2]
    scaling = scaling or 1 / math.sqrt(d)
    is_causal = kw["is_causal"] if "is_causal" in kw else getattr(module, "is_causal", attention_mask is None and nq > 1)
    use_flash = nq > 1 and nq <= nk and d in (32, 64, 96, 128, 256) and attention_mask is None and not (dropout > 0 and module.training)
    LOG.append(dict(cls=type(module).__name__, mask=None if attention_mask is None else tuple(attention_mask.shape), q=nq, kv=nk,
                    is_causal=is_causal, path="KERNEL" if use_flash else "SDPA",
                    kw=sorted(k for k, v_ in kw.items() if v_ is not None)))
    if use_flash:
        out = kernel_ref(query, key, value, scaling, is_causal)
    else:
        rep = H // key.shape[1]
        key = key.repeat_interleave(rep, 1); value = value.repeat_interleave(rep, 1)
        m = attention_mask
        if m is not None: m = m[..., :nk]
        elif is_causal and nq > 1: m = torch.ones(nq, nk, dtype=torch.bool).tril(nk - nq)
        out = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=m, scale=scaling)
    return out.transpose(1, 2).contiguous(), None

def mask_fn(batch_size, q_length, kv_length, q_offset=0, kv_offset=0, mask_function=None, attention_mask=None,
            local_size=None, allow_is_causal_skip=True, **kw):
    sd = mu.ALL_MASK_ATTENTION_FUNCTIONS["sdpa"]
    mask_function = mask_function or mu.causal_mask_function
    call = dict(batch_size=batch_size, q_length=q_length, kv_length=kv_length, q_offset=q_offset, kv_offset=kv_offset,
                mask_function=mask_function, attention_mask=attention_mask, local_size=local_size,
                allow_is_causal_skip=allow_is_causal_skip, **kw)
    plain = mask_function in (mu.causal_mask_function, mu.bidirectional_mask_function)
    ints = all(isinstance(x, int) for x in (q_length, kv_length, q_offset, kv_offset))
    why = "own-None"
    if not (allow_is_causal_skip and plain and ints and q_length <= kv_length): r = sd(**call); why = "sdpa(gate)"
    elif local_size is not None and kv_length >= local_size: r = sd(**call); why = "sdpa(local)"
    elif q_offset + q_length != kv_offset + kv_length: r = sd(**call); why = "sdpa(offset)"
    elif attention_mask is not None and not bool(attention_mask[:, kv_offset:kv_offset + kv_length].all()): r = sd(**call); why = "sdpa(pad)"
    else: r = None
    LOG.append(dict(MASK=why, ret=None if r is None else tuple(r.shape), q=q_length, kv=kv_length,
                    q_off=(type(q_offset).__name__, int(q_offset)), kv_off=kv_offset, local=local_size,
                    plain=plain, skip=allow_is_causal_skip))
    return r

ALL_ATTENTION_FUNCTIONS.register(NAME, attn_fn)
mu.ALL_MASK_ATTENTION_FUNCTIONS.register(NAME, mask_fn)
if "flash" in NAME:
    orig = PreTrainedModel._check_and_adjust_attn_implementation
    def patched(self, impl, is_init_check=False, allow_all_kernels=False):
        return NAME if impl == NAME else orig(self, impl, is_init_check, allow_all_kernels)
    PreTrainedModel._check_and_adjust_attn_implementation = patched

def build(model_type, cls, **cfg):
    c = AutoConfig.for_model(model_type, **cfg)
    m_ref = cls(c).eval(); 
    c2 = AutoConfig.for_model(model_type, **cfg)
    m = cls._from_config(c2, attn_implementation=NAME).eval(); m.load_state_dict(m_ref.state_dict())
    m_ref.set_attn_implementation("eager")
    return m, m_ref

def summarize(title, out, ref, valid=None):
    a, b = out.float(), ref.float()
    if valid is not None: a, b = a[valid], b[valid]
    err = (a - b).abs().max().item()
    paths = sorted({e["path"] for e in LOG if "path" in e}); masks = sorted({e["MASK"] for e in LOG if "MASK" in e})
    print(f"\n### {title}: max|out-eager|={err:.3e} {"WRONG" if err > 1e-2 else "ok"}  attn paths={paths} mask decisions={masks}")
    seen = set()
    for e in LOG:
        key = str({k: v for k, v in e.items()})
        if key not in seen and len(seen) < 6: seen.add(key); print("   ", e)
    LOG.clear()
