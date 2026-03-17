#pragma once

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "test_helpers.h"

namespace add_test {

constexpr uint32_t kThreads = 256;

struct AddCase {
    std::string label;
    uint32_t size = 0;
    uint32_t buffer_size = 0;
    bool run_twice = false;
    std::vector<float> residual;
    std::vector<float> hidden;
};

inline AddCase make_case(const std::string& label, uint32_t size,
                         uint32_t buffer_size, bool run_twice,
                         float residual_phase, float hidden_phase,
                         float tail_bias) {
    AddCase test_case;
    test_case.label = label;
    test_case.size = size;
    test_case.buffer_size = buffer_size;
    test_case.run_twice = run_twice;
    test_case.residual.resize(buffer_size);
    test_case.hidden.resize(buffer_size);
    for (uint32_t i = 0; i < size; ++i) {
        test_case.residual[i] =
            0.23f * std::sin((i + 1) * residual_phase) -
            0.08f * std::cos((i + 5) * residual_phase * 0.7f) +
            0.01f * static_cast<float>(i % 5);
        test_case.hidden[i] = -0.27f * std::cos((i + 3) * hidden_phase) +
                              0.19f * std::sin((i + 1) * hidden_phase * 0.6f) -
                              0.012f * static_cast<float>((i * 7) % 9);
    }
    for (uint32_t i = size; i < buffer_size; ++i) {
        test_case.residual[i] =
            tail_bias + 0.03f * static_cast<float>(i - size);
        test_case.hidden[i] = -tail_bias - 0.02f * static_cast<float>(i - size);
    }
    return test_case;
}

inline std::vector<float> apply_reference(const std::vector<float>& hidden,
                                          const std::vector<float>& residual,
                                          uint32_t size) {
    std::vector<float> out = hidden;
    for (uint32_t i = 0; i < size; ++i) {
        out[i] = kernel_test::round_to_bf16(hidden[i] + residual[i]);
    }
    return out;
}

}  // namespace add_test
