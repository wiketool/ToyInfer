#pragma once

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "cuda_bf16.h"
#include "math_constants.h"

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        const cudaError_t err__ = (call);                                      \
        if (err__ != cudaSuccess) {                                            \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__) << " ("   \
                      << static_cast<int>(err__) << ") at " << __FILE__ << ':' \
                      << __LINE__ << std::endl;                                \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

#define FETCH_BF162_RO(addr) reinterpret_cast<const bf162*>(addr)[0]
#define FETCH_BF162(addr) reinterpret_cast<bf162*>(addr)[0]

namespace toyinfer {

__device__ __forceinline__ float reduce_sum_f32_warp(float val) {
#pragma unroll
    for (uint32_t stride = 16; stride >= 1; stride >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, stride);
    }
    return val;
}

__device__ __forceinline__ float reduce_max_f32_warp(float val) {
#pragma unroll
    for (uint32_t stride = 16; stride >= 1; stride >>= 1) {
        val = fmaxf(__shfl_xor_sync(0xffffffff, val, stride), val);
    }
    return val;
}

}  // namespace toyinfer
