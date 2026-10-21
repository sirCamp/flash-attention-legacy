#!/usr/bin/env python3
"""Enumerate WMMA-forward tile candidates for Turing (sm_75: 64 KB smem/SM and per block,
32 warps/SM, 16 blocks/SM, 64K regs) and the Volta WMMA backward for d=256.

Current fwd layout (flash_attn_fwd_volta.cu):   Q[Br,d]h + K[Bc,d]h + V[Bc,d]h + S[Br,Bc]f + P[Br,Bc]h
Variant 'kv_alias' (K and V share one buffer; legal because K is dead after S=QK^T and V is loaded
after the softmax barrier, and the PV loop ends with __syncthreads):  Q + KV[Bc,d]h + S + P
Constraints of the current code: Br = NUM_THREADS (1 thread per Q row), Br,Bc multiples of 16,
PV_CHUNK = Bc must divide d.
Register estimate per thread = measured ptxas value of the current sm_75 kernel with the same d
(dominated by my_O[d]), so warps/SM by regs is the same as today.
"""
import math

REGS_SM75 = {32: 128, 64: 255, 96: 254, 128: 255, 256: 255}   # measured (fwd_volta, sm_75)
SMEM_SM, SMEM_BLK, MAX_WARPS, MAX_BLOCKS = 64 * 1024, 64 * 1024, 32, 16


def smem(Br, Bc, d, alias):
    kv = (1 if alias else 2) * Bc * d * 2
    return Br * d * 2 + kv + Br * Bc * 4 + Br * Bc * 2


def occ(regs, sm, warps):
    rpw = math.ceil(regs * 32 / 256) * 256
    w_reg = (65536 // 4 // rpw) * 4
    b = min(w_reg // warps, SMEM_SM // (math.ceil(sm / 256) * 256), MAX_WARPS // warps, MAX_BLOCKS)
    return b


print("d   | variant  | Br  | Bc | smem B (KB)    | blocks/SM | warps/SM | KV loads per Q row-block (rel.)")
for d in (32, 64, 96, 128, 256):
    rows = []
    for alias in (False, True):
        for Br in (32, 64, 128):
            for Bc in (16, 32, 64):
                if d % Bc:
                    continue
                s = smem(Br, Bc, d, alias)
                if s > SMEM_BLK:
                    continue
                w = Br // 32
                b = occ(REGS_SM75[d], s, w)
                rows.append((b * w, Br, Bc, -s, alias, s, b))
    rows.sort(reverse=True)
    for r in rows[:6]:
        wsm, Br, Bc, _, alias, s, b = r
        print(f"{d:3d} | {'kv_alias' if alias else 'current '} | {Br:3d} | {Bc:2d} | {s:6d} ({s/1024:5.1f}) | {b:9d} | {wsm:8d} | {128/Br:.1f}x")
    print("----")

# Volta WMMA backward (flash_attn_bwd.cu) smem: K,V[Bc,d]h + Q,dO[Br,d]h + dKacc,dVacc[Bc,d]f + S,dP[Br,Bc]f + half[Br,Bc]
def bwd(Br, Bc, d, acc_in_regs=False):
    return (2 * Bc * d + 2 * Br * d) * 2 + ((0 if acc_in_regs else 2 * Bc * d) + 2 * Br * Bc) * 4 + Br * Bc * 2

print("\nVolta-WMMA bwd smem (B): current vs dK/dV accumulators kept in WMMA register fragments")
for d, Br, Bc in ((32, 32, 32), (64, 32, 32), (96, 32, 16), (128, 16, 16), (256, 16, 16)):
    frags = 2 * (Bc // 16) * (d // 16) / 4  # accumulator fragments per warp (4 warps)
    print(f"d={d:3d} Br={Br} Bc={Bc}: current={bwd(Br,Bc,d)} ({bwd(Br,Bc,d)/1024:.1f} KB)  "
          f"acc-in-regs={bwd(Br,Bc,d,True)} ({bwd(Br,Bc,d,True)/1024:.1f} KB), "
          f"+{frags:.0f} frags/warp = +{frags*8:.0f} regs/thread")
