#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace attention_test {

constexpr uint32_t kThreads = 256;

struct AttentionCase {
    std::string label;
    uint32_t num_q_heads = 0;
    uint32_t num_kv_heads = 0;
    uint32_t heads_dim = 0;
    uint32_t pos = 0;
    uint32_t max_seq_len = 0;
    std::vector<float> q;
    std::vector<float> ks;
    std::vector<float> vs;
};

struct SoftmaxCase {
    std::string label;
    uint32_t num_q_heads = 0;
    uint32_t pos = 0;
    uint32_t max_seq_len = 0;
    std::vector<float> score;
};

struct ConvertCase {
    std::string label;
    uint32_t size = 0;
    uint32_t buffer_size = 0;
    std::vector<float> input;
};

inline AttentionCase make_attention_case(const std::string& label,
                                         uint32_t num_q_heads,
                                         uint32_t num_kv_heads,
                                         uint32_t heads_dim, uint32_t pos,
                                         uint32_t max_seq_len, float q_phase,
                                         float kv_phase) {
    AttentionCase test_case;
    test_case.label = label;
    test_case.num_q_heads = num_q_heads;
    test_case.num_kv_heads = num_kv_heads;
    test_case.heads_dim = heads_dim;
    test_case.pos = pos;
    test_case.max_seq_len = max_seq_len;
    test_case.q.resize(num_q_heads * heads_dim);
    test_case.ks.resize(max_seq_len * num_kv_heads * heads_dim);
    test_case.vs.resize(max_seq_len * num_kv_heads * heads_dim);

    for (uint32_t h = 0; h < num_q_heads; ++h) {
        for (uint32_t d = 0; d < heads_dim; ++d) {
            const uint32_t idx = h * heads_dim + d;
            test_case.q[idx] =
                0.18f * std::sin((idx + 1) * q_phase) -
                0.11f * std::cos((h + 1) * (d + 1) * q_phase * 0.6f) +
                0.01f * static_cast<float>((h + d) % 5);
        }
    }

    for (uint32_t t = 0; t < max_seq_len; ++t) {
        for (uint32_t kv = 0; kv < num_kv_heads; ++kv) {
            for (uint32_t d = 0; d < heads_dim; ++d) {
                const uint32_t idx =
                    t * num_kv_heads * heads_dim + kv * heads_dim + d;
                test_case.ks[idx] =
                    0.16f * std::cos((t + 1) * (d + 1) * kv_phase) +
                    0.09f * std::sin((kv + 2) * (d + 3) * kv_phase * 0.7f) -
                    0.008f * static_cast<float>((t + kv + d) % 7);
                test_case.vs[idx] =
                    0.14f * std::sin((t + 3) * (d + 1) * kv_phase * 1.3f) -
                    0.12f * std::cos((kv + 1) * (d + 2) * kv_phase * 0.9f) +
                    0.006f * static_cast<float>((t * 5 + kv * 3 + d) % 11);
            }
        }
    }

    return test_case;
}

inline std::vector<float> reference_qk_scores(
    const AttentionCase& test_case, const std::vector<float>& q_q,
    const std::vector<float>& ks_q, const std::vector<float>& initial_score) {
    std::vector<float> expected = initial_score;
    const uint32_t q_per_kv = test_case.num_q_heads / test_case.num_kv_heads;
    const float scale =
        1.0f / std::sqrt(static_cast<float>(test_case.heads_dim));
    for (uint32_t t = 0; t <= test_case.pos; ++t) {
        for (uint32_t q_head = 0; q_head < test_case.num_q_heads; ++q_head) {
            const uint32_t kv_head = q_head / q_per_kv;
            const uint32_t kv_offset =
                t * test_case.num_kv_heads * test_case.heads_dim +
                kv_head * test_case.heads_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < test_case.heads_dim; ++d) {
                dot += q_q[q_head * test_case.heads_dim + d] *
                       ks_q[kv_offset + d];
            }
            expected[q_head * test_case.max_seq_len + t] += dot * scale;
        }
    }
    return expected;
}

inline std::vector<float> reference_softmax(
    const SoftmaxCase& test_case, const std::vector<float>& initial_score) {
    std::vector<float> expected = initial_score;
    for (uint32_t head = 0; head < test_case.num_q_heads; ++head) {
        const uint32_t offset = head * test_case.max_seq_len;
        float reg_max = -INFINITY;
        for (uint32_t i = 0; i <= test_case.pos; ++i) {
            reg_max = std::max(reg_max, initial_score[offset + i]);
        }
        float reg_sum = 0.0f;
        for (uint32_t i = 0; i <= test_case.pos; ++i) {
            reg_sum += std::exp(initial_score[offset + i] - reg_max);
        }
        for (uint32_t i = 0; i <= test_case.pos; ++i) {
            expected[offset + i] =
                std::exp(initial_score[offset + i] - reg_max) / reg_sum;
        }
    }
    return expected;
}

inline std::vector<float> reference_apply_score(
    const AttentionCase& test_case, const std::vector<float>& score,
    const std::vector<float>& vs_q, const std::vector<float>& initial_o) {
    std::vector<float> expected = initial_o;
    const uint32_t q_per_kv = test_case.num_q_heads / test_case.num_kv_heads;
    for (uint32_t q_head = 0; q_head < test_case.num_q_heads; ++q_head) {
        const uint32_t kv_head = q_head / q_per_kv;
        for (uint32_t t = 0; t <= test_case.pos; ++t) {
            const float weight = score[q_head * test_case.max_seq_len + t];
            const uint32_t kv_offset =
                t * test_case.num_kv_heads * test_case.heads_dim +
                kv_head * test_case.heads_dim;
            for (uint32_t d = 0; d < test_case.heads_dim; ++d) {
                expected[q_head * test_case.heads_dim + d] +=
                    weight * vs_q[kv_offset + d];
            }
        }
    }
    return expected;
}

inline std::vector<float> reference_attention_score(
    const AttentionCase& test_case, const std::vector<float>& q_q,
    const std::vector<float>& ks_q) {
    std::vector<float> zeros(test_case.num_q_heads * test_case.max_seq_len, 0.0f);
    SoftmaxCase softmax_case{test_case.label, test_case.num_q_heads,
                             test_case.pos, test_case.max_seq_len, {}};
    return reference_softmax(
        softmax_case, reference_qk_scores(test_case, q_q, ks_q, zeros));
}

inline std::vector<float> reference_attention_buffer(
    const AttentionCase& test_case, const std::vector<float>& score,
    const std::vector<float>& vs_q) {
    std::vector<float> zeros(test_case.num_q_heads * test_case.heads_dim, 0.0f);
    return reference_apply_score(test_case, score, vs_q, zeros);
}

inline std::vector<float> reference_convert(const std::vector<float>& input,
                                            const std::vector<float>& initial_out_q,
                                            uint32_t size) {
    std::vector<float> expected = initial_out_q;
    for (uint32_t i = 0; i < size; ++i) {
        expected[i] = kernel_test::round_to_bf16(input[i]);
    }
    return expected;
}

}  // namespace attention_test
