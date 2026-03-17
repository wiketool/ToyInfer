#pragma once

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "test_helpers.h"

namespace rmsnorm_test {

constexpr uint32_t kThreads = 256;

struct Case {
    std::string label;
    uint32_t size = 0;
    uint32_t buffer_size = 0;
    float eps = 1e-5f;
    std::vector<float> input;
    std::vector<float> weight;
};

inline Case make_case(const std::string& label, uint32_t size,
                      uint32_t buffer_size, float eps, float input_phase,
                      float weight_phase) {
    Case test_case;
    test_case.label = label;
    test_case.size = size;
    test_case.buffer_size = buffer_size;
    test_case.eps = eps;
    test_case.input.resize(buffer_size);
    test_case.weight.resize(buffer_size);
    for (uint32_t i = 0; i < size; ++i) {
        test_case.input[i] = 0.48f * std::sin((i + 1) * input_phase) -
                             0.22f * std::cos((i + 3) * input_phase * 0.7f) +
                             0.013f * static_cast<float>(i % 7);
        test_case.weight[i] = 0.72f + 0.18f * std::sin((i + 5) * weight_phase) -
                              0.04f * static_cast<float>((i * 3) % 5);
    }
    for (uint32_t i = size; i < buffer_size; ++i) {
        test_case.input[i] = -0.75f + 0.05f * static_cast<float>(i - size);
        test_case.weight[i] = 0.95f - 0.03f * static_cast<float>(i - size);
    }
    return test_case;
}

inline float reference_sum(const std::vector<float>& input_q, uint32_t size) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < size; ++i) {
        sum += input_q[i] * input_q[i];
    }
    return sum;
}

inline std::vector<float> reference_output(
    const std::vector<float>& input_q, const std::vector<float>& weight_q,
    const std::vector<float>& initial_out_q, float sum, float eps,
    uint32_t size) {
    std::vector<float> expected = initial_out_q;
    const float inv_rms =
        1.0f / std::sqrt(sum / static_cast<float>(size) + eps);
    for (uint32_t i = 0; i < size; ++i) {
        expected[i] =
            kernel_test::round_to_bf16(input_q[i] * inv_rms * weight_q[i]);
    }
    return expected;
}

}  // namespace rmsnorm_test
