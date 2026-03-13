#pragma once

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace rope_test {

struct Case {
    std::string label;
    uint32_t num_heads = 0;
    uint32_t head_dim = 0;
    uint32_t pos = 0;
    uint32_t buffer_size = 0;
    std::vector<float> qk;
    std::vector<float> inv_freq;
};

inline Case make_case(const std::string& label, uint32_t num_heads,
                      uint32_t head_dim, uint32_t pos, uint32_t buffer_size,
                      float qk_phase, float freq_scale) {
    Case test_case;
    test_case.label = label;
    test_case.num_heads = num_heads;
    test_case.head_dim = head_dim;
    test_case.pos = pos;
    test_case.buffer_size = buffer_size;
    test_case.qk.resize(buffer_size);
    test_case.inv_freq.resize(head_dim / 2);

    const uint32_t size = num_heads * head_dim;
    for (uint32_t i = 0; i < size; ++i) {
        test_case.qk[i] =
            0.34f * std::sin((i + 1) * qk_phase) -
            0.21f * std::cos((i + 5) * qk_phase * 0.7f) +
            0.012f * static_cast<float>(i % 4);
    }
    for (uint32_t i = size; i < buffer_size; ++i) {
        test_case.qk[i] = -1.0f + 0.08f * static_cast<float>(i - size);
    }
    for (uint32_t i = 0; i < head_dim / 2; ++i) {
        test_case.inv_freq[i] = freq_scale * static_cast<float>(i + 1);
    }
    return test_case;
}

inline std::vector<float> reference_output(const Case& test_case,
                                           const std::vector<float>& qk_q) {
    std::vector<float> expected = qk_q;
    for (uint32_t h = 0; h < test_case.num_heads; ++h) {
        const uint32_t offset = h * test_case.head_dim;
        for (uint32_t i = 0; i < test_case.head_dim / 2; ++i) {
            const float real = qk_q[offset + i];
            const float imag = qk_q[offset + i + test_case.head_dim / 2];
            const float cos_v = std::cos(test_case.pos * test_case.inv_freq[i]);
            const float sin_v = std::sin(test_case.pos * test_case.inv_freq[i]);
            expected[offset + i] =
                kernel_test::round_to_bf16(cos_v * real - sin_v * imag);
            expected[offset + i + test_case.head_dim / 2] =
                kernel_test::round_to_bf16(cos_v * imag + sin_v * real);
        }
    }
    return expected;
}

}  // namespace rope_test
