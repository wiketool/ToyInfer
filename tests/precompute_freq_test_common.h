#pragma once

#include <cmath>
#include <string>
#include <vector>

namespace precompute_freq_test {

struct Case {
    std::string label;
    int head_dim = 0;
    int buffer_size = 0;
    float theta = 10000.0f;
};

inline std::vector<float> build_expected(const Case& test_case,
                                         const std::vector<float>& initial) {
    std::vector<float> expected = initial;
    for (int i = 0; i < test_case.head_dim / 2; ++i) {
        expected[i] = 1.0f /
                      std::pow(test_case.theta,
                               (2.0f * static_cast<float>(i)) /
                                   static_cast<float>(test_case.head_dim));
    }
    return expected;
}

}  // namespace precompute_freq_test
