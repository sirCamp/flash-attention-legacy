#!/usr/bin/env python3
"""Registers, spills, shared memory and theoretical occupancy of every kernel
instantiation, per arch.

usage: occupancy.py <repo> <out dir>   (after build_ptxas.sh <repo> <out dir>)
writes <out dir>/occupancy.json and prints the markdown table (occupancy_table.md).

Registers / spills come from ptxas -v. The tile of each kernel is read from its
demangled name, and its block size and dynamic shared memory from the headers
themselves (Tile::kNumWarps, Tile::smem_bytes(), compiled into a small host
program): no tile shape or smem formula is repeated here.
"""
import json
import math
import os
import re
import subprocess
import sys

REPO, OUT = sys.argv[1], sys.argv[2]
RAW = os.path.join(OUT, "raw")

# arch limits: smem/SM usable, smem/block max (opt-in), max warps, max blocks, regs
ARCH = {
    "sm_60": dict(name="P100", smem_sm=64 * 1024, smem_blk=48 * 1024, warps=64, blocks=32, regs=65536),
    "sm_61": dict(name="GTX1080", smem_sm=96 * 1024, smem_blk=48 * 1024, warps=64, blocks=32, regs=65536),
    "sm_70": dict(name="V100", smem_sm=96 * 1024, smem_blk=96 * 1024, warps=64, blocks=32, regs=65536),
    "sm_75": dict(name="T4", smem_sm=64 * 1024, smem_blk=64 * 1024, warps=32, blocks=16, regs=65536),
}
BODIES = {"VoltaFwd": "fwd_volta", "PascalFwd": "fwd_pascal", "BwdWmma": "bwd_wmma", "BwdSimt": "bwd_simt"}
TENSOR_CORE = ("fwd_volta", "bwd_wmma")  # stubs on sm_6x


def occ(arch, regs, smem, warps_blk):
    """Blocks per SM (cuda_occupancy.h style: registers in 256-register units per
    warp over the sub-partitions, shared memory in 256 B units)."""
    A = ARCH[arch]
    if smem > A["smem_blk"]:
        return None, "FAIL(smem>%dKB)" % (A["smem_blk"] // 1024)
    rpw = math.ceil(regs * 32 / 256) * 256
    nsub = 2 if arch == "sm_60" else 4
    b_reg = (A["regs"] // nsub // rpw) * nsub // warps_blk
    s_alloc = math.ceil(smem / 256) * 256 if smem else 0
    b_smem = A["smem_sm"] // s_alloc if s_alloc else 999
    b = min(b_reg, b_smem, A["warps"] // warps_blk, A["blocks"])
    return b, "by regs=%d, by smem=%s" % (b_reg, b_smem if b_smem < 999 else "inf")


def demangle(names):
    out = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True, check=True).stdout
    return dict(zip(names, out.splitlines()))


# 1. ptxas entries: (arch, mangled) -> regs, stack, spills, static smem
entries = []
for f in sorted(os.listdir(RAW)):
    m = re.match(r"ptxas_(\w+)_sm(\d+)\.txt", f)
    if not m:
        continue
    arch, cur = "sm_" + m.group(2), None
    for line in open(os.path.join(RAW, f)):
        mm = re.search(r"Compiling entry function '(\S+)'", line)
        if mm:
            cur = dict(arch=arch, mangled=mm.group(1))
            continue
        if cur is None:
            continue
        mm = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line)
        if mm:
            cur.update(stack=int(mm.group(1)), sst=int(mm.group(2)), sld=int(mm.group(3)))
        mm = re.search(r"Used (\d+) registers", line)
        if mm:
            ms = re.search(r"(\d+) bytes smem", line)
            cur.update(regs=int(mm.group(1)), ssmem=int(ms.group(1)) if ms else 0)
            entries.append(cur)
            cur = None
names = demangle(sorted({e["mangled"] for e in entries}))

# 2. tiles: block size and dynamic smem from the headers
tile_re = re.compile(r"flash_attn_legacy::((?:Fwd|Bwd)Tile(?:Simt|Wmma)<[^>]*>)")
tiles = sorted({tile_re.search(n).group(1) for n in names.values() if tile_re.search(n)})
src = os.path.join(OUT, "tile_report.cu")
with open(src, "w") as fh:
    fh.write('#include <cstdio>\n#include "flash_attn_common.h"\nusing namespace flash_attn_legacy;\n'
             "template <class T> void p(const char* n) { printf(\"%s|%d|%d\\n\", n, T::kNumWarps, T::smem_bytes()); }\n"
             "int main() {\n")
    for t in tiles:
        fh.write('    p<%s>("%s");\n' % (t, t))
    fh.write("    return 0;\n}\n")
exe = os.path.join(OUT, "tile_report")
subprocess.run(["nvcc", "-std=c++17", "-I" + os.path.join(REPO, "csrc", "include"), src, "-o", exe], check=True)
tile_info = {}
for line in subprocess.run([exe], capture_output=True, text=True, check=True).stdout.splitlines():
    t, w, smem = line.split("|")
    tile_info[t] = (int(w), int(smem))

# 3. rows
rows = []
for e in entries:
    name = names[e["mangled"]]
    if "precompute_D" in name:
        kind, tile, warps, dyn = "precompute_D", "-", 8, 0
    else:
        body = re.search(r"flash_attn_legacy::(\w+)>\(", name).group(1)
        kind = BODIES[body] + ("_varlen" if "varlen_kernel" in name else "")
        if BODIES[body] in TENSOR_CORE and e["arch"] in ("sm_60", "sm_61"):
            continue
        tile = tile_re.search(name).group(1)
        warps, dyn = tile_info[tile]
    b, lim = occ(e["arch"], e["regs"], dyn + e["ssmem"], warps)
    rows.append(dict(kind=kind, tile=tile, arch=e["arch"], regs=e["regs"], stack=e.get("stack", 0),
                     sst=e.get("sst", 0), sld=e.get("sld", 0), ssmem=e["ssmem"], dyn=dyn,
                     warps_blk=warps, blocks=b, warps_sm=None if b is None else b * warps, lim=lim))
rows.sort(key=lambda r: (r["kind"], [int(x) for x in re.findall(r"\d+", r["tile"])], r["arch"]))
json.dump(rows, open(os.path.join(OUT, "occupancy.json"), "w"), indent=1)

print("| kernel | tile | warps/block | arch | regs | stack B | spill st/ld B | dyn smem B (KB) | blocks/SM | warps/SM (max) | blocks/SM limits |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    ws = "-" if r["warps_sm"] is None else "%d/%d" % (r["warps_sm"], ARCH[r["arch"]]["warps"])
    print("| %s | %s | %d | %s | %d | %d | %d/%d | %d (%.1f) | %s | %s | %s |" % (
        r["kind"], r["tile"].replace("<", "&lt;"), r["warps_blk"], r["arch"], r["regs"], r["stack"], r["sst"],
        r["sld"], r["dyn"], r["dyn"] / 1024, "-" if r["blocks"] is None else r["blocks"], ws, r["lim"]))
