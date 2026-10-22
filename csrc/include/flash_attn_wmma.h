#pragma once

// WMMA helpers shared by the tensor-core forward and backward kernels
// (sm_70 / sm_75). Device code only exists for __CUDA_ARCH__ >= 700; the kernel
// bodies using it have trapping stubs for older architectures.

#include <mma.h>  // file scope: inside a namespace it breaks cuda_bf16.h

#include "flash_attn_common.h"

namespace flash_attn_legacy {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700

using namespace nvcuda;

// Row and column, within the 16x16 tile, of each element of this lane's float
// accumulator fragment (the layout is opaque but identical for every fragment
// of the type): store a fragment holding its own element ids and read them back.
// `scratch`: per-warp shared memory, 256 floats + 256 bytes, 32-byte aligned.
constexpr int kAccElems = 8;  // elements per lane of a 16x16x16 float accumulator
__device__ __forceinline__ void wmma_acc_layout(float* scratch, int lane, int (&row)[kAccElems],
                                                int (&col)[kAccElems]) {
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> probe;
    static_assert(decltype(probe)::num_elements == kAccElems, "unexpected WMMA accumulator size");
    #pragma unroll
    for (int i = 0; i < kAccElems; i++) probe.x[i] = static_cast<float>(lane * kAccElems + i);
    wmma::store_matrix_sync(scratch, probe, 16, wmma::mem_row_major);
    __syncwarp();
    unsigned char* table = reinterpret_cast<unsigned char*>(scratch + 256);
    #pragma unroll
    for (int k = 0; k < kAccElems; k++) {
        const int e = lane + 32 * k;  // position in the 16x16 tile
        table[static_cast<int>(scratch[e])] = static_cast<unsigned char>(e);
    }
    __syncwarp();
    #pragma unroll
    for (int i = 0; i < kAccElems; i++) {
        const int e = table[lane * kAccElems + i];
        row[i] = e / 16;
        col[i] = e % 16;
    }
    __syncwarp();
}

#endif  // __CUDA_ARCH__ >= 700

}  // namespace flash_attn_legacy
