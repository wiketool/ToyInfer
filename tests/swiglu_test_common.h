#pragma once

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "test_helpers.h"

namespace swiglu_test {

constexpr uint32_t kThreads = 256;

struct Case {
    std::string label;
    uint32_t size = 0;
    uint32_t buffer_size = 0;
    std::vector<float> gate;
    std::vector<float> up;
};

inline Case make_case(const std::string& label, uint32_t size,
                      uint32_t buffer_size, float gate_phase, float up_phase) {
    Case test_case;
    test_case.label = label;
    test_case.size = size;
    test_case.buffer_size = buffer_size;
    test_case.gate.resize(buffer_size);
    test_case.up.resize(buffer_size);
    for (uint32_t i = 0; i < size; ++i) {
        test_case.gate[i] = 0.9f * std::sin((i + 1) * gate_phase) -
                            0.7f * std::cos((i + 3) * gate_phase * 0.5f);
        test_case.up[i] = 0.42f * std::cos((i + 5) * up_phase) -
                          0.31f * std::sin((i + 1) * up_phase * 0.8f) -
                          0.02f * static_cast<float>(i % 6);
    }
    for (uint32_t i = size; i < buffer_size; ++i) {
        test_case.gate[i] = -1.2f + 0.07f * static_cast<float>(i - size);
        test_case.up[i] = 0.95f - 0.05f * static_cast<float>(i - size);
    }
    return test_case;
}

inline std::vector<float> reference_output(
    const std::vector<float>& gate_q, const std::vector<float>& up_q,
    const std::vector<float>& initial_out_q, uint32_t size) {
    std::vector<float> expected = initial_out_q;
    for (uint32_t i = 0; i < size; ++i) {
        const float silu = gate_q[i] * kernel_test::sigmoid(gate_q[i]);
        expected[i] = kernel_test::round_to_bf16(up_q[i] * silu);
    }
    return expected;
}

}  // namespace swiglu_test
