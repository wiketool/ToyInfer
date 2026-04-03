#pragma once

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <string>
#include <utility>

#if defined(__has_include)
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define TOYINFER_NVTX_ENABLED 1
#elif __has_include(<nvToolsExt.h>)
#include <nvToolsExt.h>
#define TOYINFER_NVTX_ENABLED 1
#else
#define TOYINFER_NVTX_ENABLED 0
#endif
#else
#define TOYINFER_NVTX_ENABLED 0
#endif

namespace toyinfer {

class ScopedNvtxRange {
   public:
    explicit ScopedNvtxRange(const char* name) : name_(name) {
#if TOYINFER_NVTX_ENABLED
        nvtxRangePushA(name_);
#endif
    }

    explicit ScopedNvtxRange(std::string name)
        : owned_name_(std::move(name)), name_(owned_name_.c_str()) {
#if TOYINFER_NVTX_ENABLED
        nvtxRangePushA(name_);
#endif
    }

    ~ScopedNvtxRange() {
#if TOYINFER_NVTX_ENABLED
        nvtxRangePop();
#endif
    }

    ScopedNvtxRange(const ScopedNvtxRange&) = delete;
    ScopedNvtxRange& operator=(const ScopedNvtxRange&) = delete;

   private:
    std::string owned_name_;
    const char* name_ = nullptr;
};

class ScopedCpuTimer {
   public:
    explicit ScopedCpuTimer(double& target_ms)
        : target_ms_(target_ms),
          start_(std::chrono::steady_clock::now()) {}

    ~ScopedCpuTimer() {
        const auto end = std::chrono::steady_clock::now();
        target_ms_ += elapsed_ms(start_, end);
    }

    ScopedCpuTimer(const ScopedCpuTimer&) = delete;
    ScopedCpuTimer& operator=(const ScopedCpuTimer&) = delete;

   private:
    static double elapsed_ms(
        const std::chrono::steady_clock::time_point& start,
        const std::chrono::steady_clock::time_point& end) {
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    double& target_ms_;
    std::chrono::steady_clock::time_point start_;
};

struct AverageTime {
    double total_ms = 0.0;
    uint32_t count = 0;

    void add(double ms) {
        total_ms += ms;
        count += 1;
    }

    double average_ms() const {
        if (count == 0) {
            return 0.0;
        }
        return total_ms / static_cast<double>(count);
    }
};

struct TransformerProfileStats {
    double prefill_total_ms = 0.0;
    double prefill_layer_attn_block_ms = 0.0;
    double prefill_layer_mlp_block_ms = 0.0;
    double decode_forward_total_ms = 0.0;
    double decode_layer_qkv_and_cache_ms = 0.0;
    double decode_layer_attention_ms = 0.0;
    double decode_layer_mlp_ms = 0.0;
    bool decode_layer_stage_timing_available = true;

    void reset() { *this = TransformerProfileStats{}; }
};

inline double elapsed_ms(
    const std::chrono::steady_clock::time_point& start,
    const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

}  // namespace toyinfer
