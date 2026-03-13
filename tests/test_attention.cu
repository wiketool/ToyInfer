#include "test_helpers.h"
#include "../src/kernel.cu"

#include <string>

namespace {

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

AttentionCase make_case(const std::string& label, uint32_t num_q_heads,
                        uint32_t num_kv_heads, uint32_t heads_dim, uint32_t pos,
                        uint32_t max_seq_len, float q_phase, float kv_phase) {
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

bool run_case(const AttentionCase& test_case) {
    const auto q_bf16 = kernel_test::to_bf16(test_case.q);
    const auto ks_bf16 = kernel_test::to_bf16(test_case.ks);
    const auto vs_bf16 = kernel_test::to_bf16(test_case.vs);
    const auto q_q = kernel_test::to_float(q_bf16);
    const auto ks_q = kernel_test::to_float(ks_bf16);
    const auto vs_q = kernel_test::to_float(vs_bf16);

    std::vector<float> expected_score(test_case.num_q_heads *
                                          test_case.max_seq_len,
                                      0.0f);
    std::vector<float> expected_o_buffer(test_case.num_q_heads *
                                             test_case.heads_dim,
                                         0.0f);
    std::vector<float> expected_o(test_case.num_q_heads * test_case.heads_dim,
                                  0.0f);
    const uint32_t q_per_kv =
        test_case.num_q_heads / test_case.num_kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(test_case.heads_dim));

    for (uint32_t q_head = 0; q_head < test_case.num_q_heads; ++q_head) {
        const uint32_t kv_head = q_head / q_per_kv;
        const uint32_t score_offset = q_head * test_case.max_seq_len;
        float reg_max = -INFINITY;
        for (uint32_t t = 0; t <= test_case.pos; ++t) {
            float dot = 0.0f;
            const uint32_t kv_offset =
                t * test_case.num_kv_heads * test_case.heads_dim +
                kv_head * test_case.heads_dim;
            for (uint32_t d = 0; d < test_case.heads_dim; ++d) {
                dot += q_q[q_head * test_case.heads_dim + d] *
                       ks_q[kv_offset + d];
            }
            expected_score[score_offset + t] = dot * scale;
            reg_max = std::max(reg_max, expected_score[score_offset + t]);
        }

        float reg_sum = 0.0f;
        for (uint32_t t = 0; t <= test_case.pos; ++t) {
            const float exp_score =
                std::exp(expected_score[score_offset + t] - reg_max);
            expected_score[score_offset + t] = exp_score;
            reg_sum += exp_score;
        }
        for (uint32_t t = 0; t <= test_case.pos; ++t) {
            expected_score[score_offset + t] /= reg_sum;
            const uint32_t kv_offset =
                t * test_case.num_kv_heads * test_case.heads_dim +
                kv_head * test_case.heads_dim;
            for (uint32_t d = 0; d < test_case.heads_dim; ++d) {
                expected_o_buffer[q_head * test_case.heads_dim + d] +=
                    expected_score[score_offset + t] * vs_q[kv_offset + d];
            }
        }

        for (uint32_t d = 0; d < test_case.heads_dim; ++d) {
            expected_o[q_head * test_case.heads_dim + d] =
                kernel_test::round_to_bf16(
                    expected_o_buffer[q_head * test_case.heads_dim + d]);
        }
    }

    kernel_test::DeviceBuffer<toyinfer::bf16> dev_q(q_bf16.size());
    kernel_test::DeviceBuffer<toyinfer::bf16> dev_ks(ks_bf16.size());
    kernel_test::DeviceBuffer<toyinfer::bf16> dev_vs(vs_bf16.size());
    kernel_test::DeviceBuffer<float> dev_score(expected_score.size());
    kernel_test::DeviceBuffer<float> dev_o_buffer(expected_o_buffer.size());
    kernel_test::DeviceBuffer<toyinfer::bf16> dev_o(expected_o.size());
    dev_q.copy_from_host(q_bf16);
    dev_ks.copy_from_host(ks_bf16);
    dev_vs.copy_from_host(vs_bf16);
    dev_score.fill_zero();
    dev_o_buffer.fill_zero();

    toyinfer::attention_bf16<256, 32>(
        dev_q.get(), dev_ks.get(), dev_vs.get(), dev_score.get(),
        dev_o_buffer.get(), dev_o.get(), test_case.num_q_heads,
        test_case.num_kv_heads, test_case.heads_dim, test_case.pos,
        test_case.max_seq_len);
    TCUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> score_out;
    std::vector<float> o_buffer_out;
    std::vector<toyinfer::bf16> o_out_bf16;
    dev_score.copy_to_host(score_out);
    dev_o_buffer.copy_to_host(o_buffer_out);
    dev_o.copy_to_host(o_out_bf16);
    const auto o_out = kernel_test::to_float(o_out_bf16);

    if (!kernel_test::expect_vector_close(score_out, expected_score, 5e-3f,
                                          5e-3f,
                                          test_case.label + " score")) {
        return false;
    }
    if (!kernel_test::expect_vector_close(
            o_buffer_out, expected_o_buffer, 5e-2f, 5e-2f,
            test_case.label + " o_buffer")) {
        return false;
    }
    return kernel_test::expect_vector_close(o_out, expected_o, 5e-2f, 5e-2f,
                                            test_case.label + " output");
}

}  // namespace

int main() {
    if (!kernel_test::ensure_cuda_device(true)) {
        return 0;
    }

    const std::vector<AttentionCase> cases = {
        make_case("attention pos0 gqa", 2, 1, 64, 0, 5, 0.07f, 0.03f),
        make_case("attention full-kv", 4, 4, 64, 6, 7, 0.05f, 0.025f),
        make_case("attention tiled-seq", 8, 2, 128, 37, 40, 0.03f, 0.017f),
    };

    for (const auto& test_case : cases) {
        if (!run_case(test_case)) {
            return 1;
        }
    }

    std::cout << "[PASS] attention_bf16" << std::endl;
    return 0;
}
