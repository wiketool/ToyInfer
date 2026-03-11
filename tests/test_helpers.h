#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace kernel_test {
inline void cuda_check(cudaError_t status, const char* expr, const char* file,
                       int line) {
    if (status != cudaSuccess) {
        std::cerr << "CUDA call failed: " << expr << " at " << file << ":"
                  << line << " -> " << cudaGetErrorString(status)
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

inline bool ensure_cuda_device(bool require_bf16) {
    int device_count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&device_count);
    if (count_status != cudaSuccess || device_count == 0) {
        std::cout << "[SKIP] No CUDA device available." << std::endl;
        return false;
    }

    cudaDeviceProp props{};
    cuda_check(cudaGetDeviceProperties(&props, 0), "cudaGetDeviceProperties",
               __FILE__, __LINE__);
    if (require_bf16 && props.major < 8) {
        std::cout << "[SKIP] Device does not support bfloat16 (need sm_80+)."
                  << std::endl;
        return false;
    }
    return true;
}

inline float bf16_to_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

inline __nv_bfloat16 float_to_bf16(float value) {
    return __float2bfloat16(value);
}

inline float round_to_bf16(float value) {
    return bf16_to_float(float_to_bf16(value));
}

inline std::vector<__nv_bfloat16> to_bf16(const std::vector<float>& input) {
    std::vector<__nv_bfloat16> out(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        out[i] = float_to_bf16(input[i]);
    }
    return out;
}

inline std::vector<float> to_float(const std::vector<__nv_bfloat16>& input) {
    std::vector<float> out(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        out[i] = bf16_to_float(input[i]);
    }
    return out;
}

inline bool nearly_equal(float a, float b, float atol, float rtol = 1e-4f) {
    return std::fabs(a - b) <= atol + rtol * std::fabs(b);
}
}  // namespace kernel_test

#define TCUDA_CHECK(call) \
    ::kernel_test::cuda_check((call), #call, __FILE__, __LINE__)
