#!/usr/bin/env python3
"""Static SASS opcode histogram per kernel instantiation (code size + key opcodes)."""
import re, collections, glob, os

SASS = "/workspace/analysis/kernels/sass"
OUT = "/workspace/analysis/kernels/sass_opcount.txt"

INS = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(?:@!?U?P[T0-9]+\s+)?([A-Z][A-Z0-9_]*)((?:\.[A-Z0-9_]+)*)")


def split(path):
    funcs, cur = {}, None
    for line in open(path):
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            cur = m.group(1)
            funcs[cur] = collections.Counter()
            continue
        if cur:
            m = INS.search(line)
            if m:
                funcs[cur][m.group(2) + m.group(3)] += 1
    return funcs


def fam(c, *prefixes):
    return sum(v for k, v in c.items() if any(k.startswith(p) for p in prefixes))


def label(n):
    for k in ("fwd_volta_varlen", "fwd_volta", "fwd_pascal_varlen", "fwd_pascal",
              "bwd_volta", "bwd_kernel", "precompute_D"):
        if k in n:
            kind = k.replace("bwd_kernel", "bwd_cudacore")
            break
    t = re.search(r"Tile\w+?_d\d+", n)
    return kind, (t.group(0) if t else "-")


with open(OUT, "w") as out:
    print("# static SASS instruction counts (not dynamic!) per instantiation", file=out)
    print("# cols: kind arch tile | total_instr code_KB | HMUL2/HFMA2/HADD2 (fp16 math) | F2F (cvt) | "
          "FFMA | HMMA | LDS | LDL/STL (local mem) | RED/ATOM (global atomics)", file=out)
    for path in sorted(glob.glob(SASS + "/*.sass")):
        arch = "sm_" + path.split("_sm")[-1][:2]
        for n, c in split(path).items():
            kind, tile = label(n)
            if kind in ("fwd_volta", "fwd_volta_varlen", "bwd_volta") and arch in ("sm_60", "sm_61"):
                continue  # stubs
            tot = sum(c.values())
            # sm_6x: 8-byte instr (+1 control word per 3); sm_7x: 16-byte
            code_kb = tot * (8 * 4 / 3 if arch in ("sm_60", "sm_61") else 16) / 1024
            hm = fam(c, "HMMA")
            hmk = sorted({k for k in c if k.startswith("HMMA")})
            print(f"{kind:18s} {arch} {tile:16s} | instr={tot:6d} code={code_kb:7.1f}KB | "
                  f"fp16={fam(c,'HMUL2','HFMA2','HADD2'):5d} F2F={fam(c,'F2F'):5d} "
                  f"FFMA={fam(c,'FFMA'):6d} HMMA={hm:5d}{hmk if hm else ''} LDS={fam(c,'LDS'):5d} "
                  f"LDL={fam(c,'LDL'):5d} STL={fam(c,'STL'):5d} RED/ATOM={fam(c,'RED','ATOM'):5d}",
                  file=out)
print(open(OUT).read())
