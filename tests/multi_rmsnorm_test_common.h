#pragma once

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace multi_rmsnorm_test {

struct Case {
    std::string label;
    uint32_t num_heads = 0;
    uint32_t head_dim = 0;
    uint32_t buffer_size = 0;
    float eps = 1e-5f;
    bool in_place = false;
    std::vector<float> input;
    std::vector<float> weight;
};

inline Case make_case(const std::string& label, uint32_t num_heads,
                      uint32_t head_dim, uint32_t buffer_size, float eps,
                      bool in_place, float input_phase, float weight_phase) {
    Case test_case;
    test_case.label = label;
    test_case.num_heads = num_heads;
    test_case.head_dim = head_dim;
    test_case.buffer_size = buffer_size;
    test_case.eps = eps;
    test_case.in_place = in_place;
    test_case.input.resize(buffer_size);
    test_case.weight.resize(head_dim);

    const uint32_t size = num_heads * head_dim;
    for (uint32_t h = 0; h < num_heads; ++h) {
        for (uint32_t i = 0; i < head_dim; ++i) {
            const uint32_t idx = h * head_dim + i;
            test_case.input[idx] =
                0.31f * std::sin((idx + 1) * input_phase) -
                0.24f * std::cos((h + 1) * (i + 3) * input_phase * 0.7f) +
                0.01f * static_cast<float>((h + i) % 5);
        }
    }
    for (uint32_t i = size; i < buffer_size; ++i) {
        test_case.input[i] = -0.5f + 0.07f * static_cast<float>(i - size);
    }
    for (uint32_t i = 0; i < head_dim; ++i) {
        test_case.weight[i] =
            0.68f + 0.14f * std::sin((i + 5) * weight_phase) -
            0.05f * static_cast<float>(i % 3);
    }
    return test_case;
}

inline std::vector<float> reference_output(const Case& test_case,
                                           const std::vector<float>& input_q,
                                           const std::vector<float>& weight_q,
                                           const std::vector<float>& initial_out_q) {
    std::vector<float> expected = initial_out_q;
    for (uint32_t h = 0; h < test_case.num_heads; ++h) {
        const uint32_t offset = h * test_case.head_dim;
        float head_sum = 0.0f;
        for (uint32_t i = 0; i < test_case.head_dim; ++i) {
            head_sum += input_q[offset + i] * input_q[offset + i];
        }
        const float inv_rms =
            1.0f / std::sqrt(head_sum / static_cast<float>(test_case.head_dim) +
                             test_case.eps);
        for (uint32_t i = 0; i < test_case.head_dim; ++i) {
            expected[offset + i] = kernel_test::round_to_bf16(
                input_q[offset + i] * inv_rms * weight_q[i]);
        }
    }
    return expected;
}

}  // namespace multi_rmsnorm_test
