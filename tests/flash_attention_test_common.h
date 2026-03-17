#pragma once

#include "test_helpers.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace flash_attention_test {

struct FlashAttentionCase {
    std::string label;
    uint32_t num_q_heads = 0;
    uint32_t num_kv_heads = 0;
    uint32_t heads_dim = 0;
    uint32_t seq_len = 0;
    std::vector<float> q;
    std::vector<float> ks;
    std::vector<float> vs;
};

inline FlashAttentionCase make_case(const std::string& label,
                                    uint32_t num_q_heads,
                                    uint32_t num_kv_heads,
                                    uint32_t heads_dim, uint32_t seq_len,
                                    float q_phase, float kv_phase,
                                    float q_scale = 1.0f,
                                    float k_scale = 1.0f,
                                    float v_scale = 1.0f) {
    FlashAttentionCase test_case;
    test_case.label = label;
    test_case.num_q_heads = num_q_heads;
    test_case.num_kv_heads = num_kv_heads;
    test_case.heads_dim = heads_dim;
    test_case.seq_len = seq_len;

    const uint32_t q_dim = num_q_heads * heads_dim;
    const uint32_t kv_dim = num_kv_heads * heads_dim;
    test_case.q.resize(seq_len * q_dim);
    test_case.ks.resize(seq_len * kv_dim);
    test_case.vs.resize(seq_len * kv_dim);

    for (uint32_t t = 0; t < seq_len; ++t) {
        for (uint32_t h = 0; h < num_q_heads; ++h) {
            for (uint32_t d = 0; d < heads_dim; ++d) {
                const uint32_t idx = t * q_dim + h * heads_dim + d;
                const float t_term = static_cast<float>(static_cast<int>(t % 3) - 1);
                test_case.q[idx] =
                    q_scale *
                    (0.19f * std::sin((t + 1) * (d + 1) * q_phase) +
                     0.13f * std::cos((h + 1) * (d + 2) * q_phase * 0.7f) -
                     0.018f * static_cast<float>((t + h + d) % 5) +
                     0.011f * t_term);
            }
        }
    }

    for (uint32_t t = 0; t < seq_len; ++t) {
        for (uint32_t kv = 0; kv < num_kv_heads; ++kv) {
            for (uint32_t d = 0; d < heads_dim; ++d) {
                const uint32_t idx = t * kv_dim + kv * heads_dim + d;
                const float alt = static_cast<float>(static_cast<int>((t + d) % 4) - 2);
                test_case.ks[idx] =
                    k_scale *
                    (0.17f * std::cos((t + 1) * (d + 1) * kv_phase) +
                     0.09f *
                         std::sin((kv + 1) * (d + 3) * kv_phase * 1.1f) +
                     0.012f * static_cast<float>((t + kv * 2 + d) % 7) +
                     0.008f * alt);
                test_case.vs[idx] =
                    v_scale *
                    (0.15f * std::sin((t + 2) * (d + 1) * kv_phase * 1.3f) -
                     0.12f *
                         std::cos((kv + 2) * (d + 1) * kv_phase * 0.8f) +
                     0.01f * static_cast<float>((t * 3 + kv * 5 + d) % 11) -
                     0.014f * static_cast<float>(t % 2));
            }
        }
    }

    return test_case;
}

inline std::vector<float> reference_output(
    const FlashAttentionCase& test_case, const std::vector<float>& q_q,
    const std::vector<float>& ks_q, const std::vector<float>& vs_q) {
    const uint32_t q_dim = test_case.num_q_heads * test_case.heads_dim;
    const uint32_t kv_dim = test_case.num_kv_heads * test_case.heads_dim;
    const uint32_t q_per_kv = test_case.num_q_heads / test_case.num_kv_heads;
    const double scale =
        1.0 / std::sqrt(static_cast<double>(test_case.heads_dim));

    std::vector<float> expected(test_case.seq_len * q_dim, 0.0f);
    std::vector<double> scores(test_case.seq_len, 0.0);

    for (uint32_t q_pos = 0; q_pos < test_case.seq_len; ++q_pos) {
        for (uint32_t q_head = 0; q_head < test_case.num_q_heads; ++q_head) {
            const uint32_t q_offset = q_pos * q_dim + q_head * test_case.heads_dim;
            const uint32_t kv_head = q_head / q_per_kv;
            double row_max = -std::numeric_limits<double>::infinity();

            for (uint32_t kv_pos = 0; kv_pos <= q_pos; ++kv_pos) {
                const uint32_t kv_offset =
                    kv_pos * kv_dim + kv_head * test_case.heads_dim;
                double dot = 0.0;
                for (uint32_t d = 0; d < test_case.heads_dim; ++d) {
                    dot += static_cast<double>(q_q[q_offset + d]) *
                           static_cast<double>(ks_q[kv_offset + d]);
                }
                scores[kv_pos] = dot * scale;
                row_max = std::max(row_max, scores[kv_pos]);
            }

            double row_sum = 0.0;
            for (uint32_t kv_pos = 0; kv_pos <= q_pos; ++kv_pos) {
                row_sum += std::exp(scores[kv_pos] - row_max);
            }

            for (uint32_t kv_pos = 0; kv_pos <= q_pos; ++kv_pos) {
                const double weight =
                    std::exp(scores[kv_pos] - row_max) / row_sum;
                const uint32_t kv_offset =
                    kv_pos * kv_dim + kv_head * test_case.heads_dim;
                for (uint32_t d = 0; d < test_case.heads_dim; ++d) {
                    expected[q_offset + d] +=
                        static_cast<float>(weight) * vs_q[kv_offset + d];
                }
            }
        }
    }

    return expected;
}

inline std::vector<float> reference_convert(const std::vector<float>& input) {
    std::vector<float> expected(input.size(), 0.0f);
    for (uint32_t i = 0; i < input.size(); ++i) {
        expected[i] = kernel_test::round_to_bf16(input[i]);
    }
    return expected;
}

}  // namespace flash_attention_test
