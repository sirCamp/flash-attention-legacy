"""Compact view of results/<tag>.json files."""
import json
import sys


def m(d):
    if not d:
        return "-"
    return f"1-cos={d['one_minus_cos']:.2e} top1={d['top1']:.4f} max|d|={d['max_abs']:.3g}{'' if d['finite'] else ' NONFINITE'}"


def p(path):
    if not path:
        return "-"
    s = f"calls={path['calls']} kern={path['kernel_calls']}"
    for ph in ("prefill(q>1)", "decode(q=1)"):
        if ph in path:
            x = path[ph]
            s += f" | {ph}: kern {x['kernel']}/{x['calls']} kl={x['kernel_layers'][:6]}{'..' if len(x['kernel_layers'])>6 else ''} fbl={x['fallback_layers'][:6]}{'..' if len(x['fallback_layers'])>6 else ''} masks={x['masks'][:2]} qk={x['qk_shapes'][:2]}"
    return s


for f in sys.argv[1:]:
    R = json.load(open(f))
    print("=" * 100)
    print(R["tag"], R["model"], R.get("override"))
    print("config:", R.get("config"))
    print("errors:", R.get("errors"))
    S = R["scenarios"]
    if "prefill" in S:
        x = S["prefill"]
        print(f"PREFILL N={x['N']}: {p(x['path'])}")
        print("  kwargs:", x["path"].get("kwargs_seen"), "is_causal:", x["path"].get("module_is_causal"))
        print("  flash_vs_sdpa", m(x["flash_vs_sdpa"]), "\n  eager_vs_sdpa", m(x["eager_vs_sdpa"]))
    if "left_padded" in S:
        x = S["left_padded"]
        print("LEFTPAD fwd:", p(x.get("path_forward")))
        print("  gen:", p(x.get("path_generate")))
        for i in range(2):
            print(f"  seq{i} flash_vs_sdpa", m(x.get(f"seq{i}_flash_vs_sdpa")), "| eager_vs_sdpa", m(x.get(f"seq{i}_eager_vs_sdpa")))
        print("  short padded vs alone (flash)", m(x.get("short_padded_vs_alone_flash")), "gen_first_div", x.get("gen_first_div"))
    if "chunked" in S:
        x = S["chunked"]
        print("CHUNKED:", p(x["path"]))
        print("  flash", m(x["flash_chunk_vs_sdpa_full"]), "\n  eager", m(x["eager_chunk_vs_sdpa_full"]), "\n  sdpa ", m(x["sdpa_chunk_vs_sdpa_full"]))
    if "generate" in S:
        x = S["generate"]
        print("GENERATE:", p(x["path"]))
        print("  first_div flash/sdpa", x["first_div_flash_vs_sdpa"], "eager/sdpa", x["first_div_eager_vs_sdpa"], "flash/eager", x["first_div_flash_vs_eager"],
              "wall", {k: v for k, v in x.items() if k.startswith("wall")})
        print("  flash:", repr(x["text_flash"][:150]))
        print("  sdpa :", repr(x["text_sdpa"][:150]))
        print("  eager:", repr(x["text_eager"][:150]))
    if "long" in S:
        x = S["long"]
        print(f"LONG N={x['N']} W={x['window']}: {p(x['path'])}")
        for k in ("flash_vs_sdpa", "eager_vs_sdpa", "flash_vs_eager"):
            if k in x:
                for kk, vv in x[k].items():
                    print(f"  {k} [{kk}]", m(vv))
        print("  chunk path:", p(x["chunk_path"]))
        print("  chunk flash vs sdpa full", m(x["flash_chunk_vs_sdpa_full"]))
        print("  gen path:", p(x["gen_path"]))
        print("  gen first div", x["gen_first_div_flash_vs_sdpa"])
        print("  flash:", repr(x["gen_text_flash"][:120]))
        print("  sdpa :", repr(x["gen_text_sdpa"][:120]))
    G = R.get("gold_fp32_eager")
    if G:
        print("GOLD fp32 eager:")
        for k, v in G.get("prefill", {}).items():
            print(f"  prefill {k}", m(v))
        print("  chunked flash", m(G.get("chunked_flash")))
        print("  gen first div vs gold", G.get("gen_first_div_vs_gold"))
        for k, v in G.get("long", {}).items():
            for kk, vv in v.items():
                print(f"  long {k} [{kk}]", m(vv))
        if "longchunk_flash" in G:
            print("  longchunk flash", m(G["longchunk_flash"]))
            print("  longgen first div vs gold", G.get("longgen_first_div_vs_gold"))
