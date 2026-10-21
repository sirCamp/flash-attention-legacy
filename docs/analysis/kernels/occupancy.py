#!/usr/bin/env python3
"""Parse ptxas -v output and compute theoretical occupancy per kernel instantiation/arch.

STALE: TILES below predates P2.3 and the P2.8 forward; regenerate in P2.8 step 6."""
import re, glob, os, math, json
RAW = "/workspace/analysis/kernels/raw"
TILES = {  # (Br, Bc, d, warps)
 "TileVolta_d32": (128,32,32,4), "TileVolta_d64": (128,64,64,4), "TileVolta_d96": (128,32,96,4),
 "TileVolta_d128": (128,32,128,4), "TileVolta_d256": (64,16,256,2),
 "TilePascal_d32": (128,32,32,4), "TilePascal_d64": (128,32,64,4), "TilePascal_d96": (64,16,96,2),
 "TilePascal_d128": (64,16,128,2), "TilePascal_d256": (32,16,256,1),
 "TileBwd_d32": (32,32,32,4), "TileBwd_d64": (32,32,64,4), "TileBwd_d96": (32,16,96,4),
 "TileBwd_d128": (16,16,128,4), "TileBwd_d256": (16,16,256,4),
}
def dyn_smem(kind, t):
    Br,Bc,d,w = TILES[t]
    if kind in ("fwd_volta","fwd_volta_varlen"):  return (Br*d+2*Bc*d)*2 + Br*Bc*4 + Br*Bc*2
    if kind in ("fwd_pascal","fwd_pascal_varlen"): return (Br*d+2*Bc*d)*2 + Br*Bc*4
    if kind == "bwd_cudacore": return (2*Bc*d+2*Br*d)*2
    if kind == "bwd_volta": return (2*Bc*d+2*Br*d)*2 + (2*Bc*d+2*Br*Bc)*4 + Br*Bc*2
    return 0
# arch limits: smem/SM usable, smem/block max (opt-in), max warps, max blocks, regs
ARCH = {
 "sm_60": dict(name="P100",    smem_sm=64*1024, smem_blk=48*1024, warps=64, blocks=32, regs=65536),
 "sm_61": dict(name="GTX1080", smem_sm=96*1024, smem_blk=48*1024, warps=64, blocks=32, regs=65536),
 "sm_70": dict(name="V100",    smem_sm=96*1024, smem_blk=96*1024, warps=64, blocks=32, regs=65536),
 "sm_75": dict(name="T4",      smem_sm=64*1024, smem_blk=64*1024, warps=32, blocks=16, regs=65536),
}
def occ(arch, regs, smem, warps_blk):
    A = ARCH[arch]
    if smem > A["smem_blk"]: return None, "FAIL(smem>%dKB)" % (A["smem_blk"]//1024)
    # cuda_occupancy.h model: regs allocated per warp in units of 256, split over sub-partitions
    rpw = math.ceil(regs*32/256)*256
    nsub = 2 if arch == "sm_60" else 4
    warps_by_reg = (A["regs"]//nsub//rpw)*nsub
    b_reg = warps_by_reg // warps_blk
    s_alloc = math.ceil(smem/256)*256 if smem else 0
    b_smem = A["smem_sm"]//s_alloc if s_alloc else 999
    b_warp = A["warps"]//warps_blk
    b = min(b_reg, b_smem, b_warp, A["blocks"])
    lim = "by regs=%d, by smem=%s" % (b_reg, b_smem if b_smem < 999 else "inf")
    return b, lim
rows=[]
for f in sorted(glob.glob(RAW+"/ptxas_*_sm*.txt")):
    arch = "sm_" + f.split("_sm")[-1][:2]
    txt = open(f).read().split("\n")
    cur=None
    for i,l in enumerate(txt):
        m = re.search(r"Compiling entry function \x27(\S+)\x27", l)
        if m: cur = dict(mangled=m.group(1)); continue
        if cur is None: continue
        m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", l)
        if m: cur.update(stack=int(m.group(1)), sst=int(m.group(2)), sld=int(m.group(3)))
        m = re.search(r"Used (\d+) registers", l)
        if m:
            cur["regs"]=int(m.group(1))
            ms = re.search(r"(\d+) bytes smem", l); cur["ssmem"]=int(ms.group(1)) if ms else 0
            n = cur["mangled"]
            if "precompute_D" in n: kind="precompute_D"; tile="-"
            else:
                tile = re.search(r"(Tile\w+?_d\d+)E", n).group(1)
                if "fwd_volta_varlen" in n: kind="fwd_volta_varlen"
                elif "fwd_volta" in n: kind="fwd_volta"
                elif "fwd_pascal_varlen" in n: kind="fwd_pascal_varlen"
                elif "fwd_pascal" in n: kind="fwd_pascal"
                elif "bwd_volta" in n: kind="bwd_volta"
                elif "bwd_kernel" in n: kind="bwd_cudacore"
            # skip stubs (Volta kernels compiled for sm_6x)
            if kind in ("fwd_volta","fwd_volta_varlen","bwd_volta") and arch in ("sm_60","sm_61"):
                cur=None; continue
            w = TILES[tile][3] if tile!="-" else 8
            dyn = dyn_smem(kind,tile) if tile!="-" else 0
            tot = dyn + cur["ssmem"]
            b,lim = occ(arch, cur["regs"], tot, w)
            rows.append(dict(kind=kind, tile=tile, arch=arch, regs=cur["regs"], stack=cur.get("stack",0),
                 sst=cur.get("sst",0), sld=cur.get("sld",0), ssmem=cur["ssmem"], dyn=dyn, tot=tot,
                 warps_blk=w, blocks=b, warps_sm=(b*w if b is not None else None), lim=lim))
            cur=None
order={"fwd_volta":0,"fwd_volta_varlen":1,"fwd_pascal":2,"fwd_pascal_varlen":3,"bwd_volta":4,"bwd_cudacore":5,"precompute_D":6}
def dkey(t): return int(t.split("_d")[1]) if "_d" in t else 0
rows.sort(key=lambda r:(order[r["kind"]], dkey(r["tile"]), r["arch"]))
json.dump(rows, open("/workspace/analysis/kernels/occupancy.json","w"), indent=1)
print("| kernel | tile (Br,Bc,d,warps) | arch | regs | stack B | spill st/ld B | static smem | dyn smem B (KB) | blocks/SM | warps/SM (max) | blocks/SM limits |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    tl = TILES.get(r["tile"]); ts = "%s (%d,%d,%d,%d)"%((r["tile"],)+tl) if tl else "-"
    mw = ARCH[r["arch"]]["warps"]
    bs = "-" if r["blocks"] is None else str(r["blocks"])
    ws = "-" if r["warps_sm"] is None else "%d/%d"%(r["warps_sm"],mw)
    print("| %s | %s | %s | %d | %d | %d/%d | %d | %d (%.1f) | %s | %s | %s |"%(r["kind"],ts,r["arch"],r["regs"],r["stack"],r["sst"],r["sld"],r["ssmem"],r["dyn"],r["dyn"]/1024,bs,ws,r["lim"]))
