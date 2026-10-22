#pragma once

// mma.sync / ldmatrix helpers for Turing (sm_75), the FA2 register layout.
// Device code only exists for __CUDA_ARCH__ >= 750; the kernel bodies using it
// have trapping stubs for older architectures.
//
// mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32, lane = 4 g + t:
//   A [16 x 8]  a[0] = (row g,     k 2t, 2t+1)   a[1] = (row g + 8, k 2t, 2t+1)   (half2 as uint32)
//   B [8 x 8]   b    = (k 2t, 2t+1, col g)
//   C [16 x 8]  c[0], c[1] = (row g, cols 2t, 2t+1)   c[2], c[3] = (row g + 8, cols 2t, 2t+1)
// so an accumulator packed to half2, {c0 c1, c2 c3}, is the A operand of the same
// 16 rows with k = its 8 columns (P / dS reused as MMA inputs without shared memory).
//
// ldmatrix.m8n8 (x2 / x4): lanes 8i..8i+7 give the 16-byte aligned addresses of
// the 8 rows of matrix i; lane (g, t) receives from matrix i, as one uint32,
// elements (row g, cols 2t, 2t+1), or with .trans (rows 2t, 2t+1, col g).
// Hence, for row-major [rows][cols] storage in shared memory:
//   A operand rows (row, k consecutive)          -> ldmatrix
//   B operand from [n][k] storage (Q for S^T = K Q^T) -> ldmatrix
//   B operand from [k][n] storage (dO for dV += P^T dO) -> ldmatrix.trans
//   A operand from [k][m] storage (dS^T for dQ = dS K) -> ldmatrix.trans

#include "flash_attn_common.h"

namespace flash_attn_legacy {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 750

// c += a b (m16n8k8, fp32 accumulate)
__device__ __forceinline__ void mma_m16n8k8(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t b) {
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(b));
}

__device__ __forceinline__ unsigned smem_addr(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

// Four 8x8 matrices; `row` = this lane's row address (matrix lane / 8, row lane % 8).
// volatile + "memory": ordered with the shared-memory stores and barriers around it.
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const half* row) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_addr(row)) : "memory");
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], const half* row) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_addr(row)) : "memory");
}

__device__ __forceinline__ uint32_t pack_half2(float lo, float hi) {
    const half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&h);
}

#endif  // __CUDA_ARCH__ >= 750

}  // namespace flash_attn_legacy
