#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
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

template <typename T>
class DeviceBuffer {
   public:
    explicit DeviceBuffer(size_t size) : size_(size) {
        if (size_ > 0) {
            cuda_check(cudaMalloc(&ptr_, sizeof(T) * size_), "cudaMalloc",
                       __FILE__, __LINE__);
        }
    }

    ~DeviceBuffer() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return ptr_; }
    size_t size() const { return size_; }

    void copy_from_host(const std::vector<T>& host) {
        if (host.size() != size_) {
            std::cerr << "Host/device size mismatch: host=" << host.size()
                      << ", device=" << size_ << std::endl;
            std::exit(EXIT_FAILURE);
        }
        if (size_ > 0) {
            cuda_check(cudaMemcpy(ptr_, host.data(), sizeof(T) * size_,
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpyHostToDevice", __FILE__, __LINE__);
        }
    }

    void copy_to_host(std::vector<T>& host) const {
        host.resize(size_);
        if (size_ > 0) {
            cuda_check(cudaMemcpy(host.data(), ptr_, sizeof(T) * size_,
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpyDeviceToHost", __FILE__, __LINE__);
        }
    }

    void fill_zero() const {
        if (size_ > 0) {
            cuda_check(cudaMemset(ptr_, 0, sizeof(T) * size_), "cudaMemset",
                       __FILE__, __LINE__);
        }
    }

   private:
    T* ptr_ = nullptr;
    size_t size_ = 0;
};

template <typename T>
inline T copy_scalar_from_device(const T* ptr) {
    T value{};
    cuda_check(cudaMemcpy(&value, ptr, sizeof(T), cudaMemcpyDeviceToHost),
               "cudaMemcpyDeviceToHost", __FILE__, __LINE__);
    return value;
}

template <typename T>
inline void copy_scalar_to_device(const T& value, T* ptr) {
    cuda_check(cudaMemcpy(ptr, &value, sizeof(T), cudaMemcpyHostToDevice),
               "cudaMemcpyHostToDevice", __FILE__, __LINE__);
}

inline bool expect_vector_close(const std::vector<float>& got,
                                const std::vector<float>& expected,
                                float atol, float rtol,
                                const std::string& label) {
    if (got.size() != expected.size()) {
        std::cerr << "[FAIL] " << label << " size mismatch, expected "
                  << expected.size() << ", got " << got.size() << std::endl;
        return false;
    }

    for (size_t i = 0; i < got.size(); ++i) {
        if (!nearly_equal(got[i], expected[i], atol, rtol)) {
            std::cerr << "[FAIL] " << label << " mismatch at " << i
                      << ", expected " << expected[i] << ", got " << got[i]
                      << std::endl;
            return false;
        }
    }
    return true;
}

inline bool expect_vector_zero(const std::vector<float>& got, size_t begin_idx,
                               const std::string& label, float atol = 1e-7f) {
    for (size_t i = begin_idx; i < got.size(); ++i) {
        if (std::fabs(got[i]) > atol) {
            std::cerr << "[FAIL] " << label << " expected zero at " << i
                      << ", got " << got[i] << std::endl;
            return false;
        }
    }
    return true;
}

inline float sigmoid(float value) {
    return 1.0f / (1.0f + std::exp(-value));
}
}  // namespace kernel_test

#define TCUDA_CHECK(call) \
    ::kernel_test::cuda_check((call), #call, __FILE__, __LINE__)
