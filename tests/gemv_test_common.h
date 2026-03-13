#pragma once

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace gemv_test {

constexpr uint32_t kThreads = 256;

struct Case {
    std::string label;
    uint32_t M = 0;
    uint32_t N = 0;
    uint32_t output_buffer_size = 0;
    std::vector<float> x;
    std::vector<float> w;
};

inline Case make_case(const std::string& label, uint32_t M, uint32_t N,
                      uint32_t output_buffer_size, float x_phase,
                      float w_phase) {
    Case test_case;
    test_case.label = label;
    test_case.M = M;
    test_case.N = N;
    test_case.output_buffer_size = output_buffer_size;
    test_case.x.resize(N);
    test_case.w.resize(M * N);
    for (uint32_t i = 0; i < N; ++i) {
        test_case.x[i] =
            0.45f * std::sin((i + 1) * x_phase) -
            0.18f * std::cos((i + 3) * (x_phase * 0.7f)) +
            0.01f * static_cast<float>(i % 5);
    }
    for (uint32_t r = 0; r < M; ++r) {
        for (uint32_t c = 0; c < N; ++c) {
            test_case.w[r * N + c] =
                0.35f * std::cos((r + 1) * (c + 1) * w_phase) +
                0.22f * std::sin((r + c + 3) * (w_phase * 0.6f)) -
                0.015f * static_cast<float>((r * 7 + c * 3) % 9);
        }
    }
    return test_case;
}

inline std::vector<float> reference_sum(const Case& test_case,
                                        const std::vector<float>& x_q,
                                        const std::vector<float>& w_q) {
    std::vector<float> expected(test_case.M, 0.0f);
    for (uint32_t r = 0; r < test_case.M; ++r) {
        float sum = 0.0f;
        for (uint32_t c = 0; c < test_case.N; ++c) {
            sum += w_q[r * test_case.N + c] * x_q[c];
        }
        expected[r] = sum;
    }
    return expected;
}

}  // namespace gemv_test
