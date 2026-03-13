#include "test_helpers.h"
#include "../src/kernel.cu"
#include "attention_test_common.h"

namespace {

bool run_case(const attention_test::AttentionCase& test_case) {
    const auto q_bf16 = kernel_test::to_bf16(test_case.q);
    const auto ks_bf16 = kernel_test::to_bf16(test_case.ks);
    const auto vs_bf16 = kernel_test::to_bf16(test_case.vs);
    const auto q_q = kernel_test::to_float(q_bf16);
    const auto ks_q = kernel_test::to_float(ks_bf16);
    const auto vs_q = kernel_test::to_float(vs_bf16);

    const auto expected_score =
        attention_test::reference_attention_score(test_case, q_q, ks_q);
    const auto expected_o_buffer =
        attention_test::reference_attention_buffer(test_case, expected_score, vs_q);
    std::vector<float> initial_o(test_case.num_q_heads * test_case.heads_dim, 0.0f);
    const auto expected_o = attention_test::reference_convert(
        expected_o_buffer, initial_o, test_case.num_q_heads * test_case.heads_dim);

    std::vector<float> dirty_score(expected_score.size(), 0.0f);
    std::vector<float> dirty_o_buffer(expected_o_buffer.size(), 0.0f);
    std::vector<toyinfer::bf16> dirty_o(expected_o.size());
    for (uint32_t i = 0; i < dirty_score.size(); ++i) {
        dirty_score[i] = 10.0f - 0.1f * static_cast<float>(i % 17);
    }
    for (uint32_t i = 0; i < dirty_o_buffer.size(); ++i) {
        dirty_o_buffer[i] = -7.0f + 0.05f * static_cast<float>(i % 23);
        dirty_o[i] = kernel_test::float_to_bf16(
            3.0f - 0.08f * static_cast<float>(i % 19));
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
    dev_score.copy_from_host(dirty_score);
    dev_o_buffer.copy_from_host(dirty_o_buffer);
    dev_o.copy_from_host(dirty_o);

    toyinfer::attention_bf16<attention_test::kThreads, 32>(
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
                                          test_case.label + " wrapper score")) {
        return false;
    }
    if (!kernel_test::expect_vector_close(
            o_buffer_out, expected_o_buffer, 5e-2f, 5e-2f,
            test_case.label + " wrapper o_buffer")) {
        return false;
    }
    return kernel_test::expect_vector_close(
        o_out, expected_o, 5e-2f, 5e-2f, test_case.label + " wrapper output");
}

}  // namespace

int main() {
    if (!kernel_test::ensure_cuda_device(true)) {
        return 0;
    }

    const std::vector<attention_test::AttentionCase> cases = {
        attention_test::make_attention_case("attention wrapper pos0 gqa", 2, 1, 64,
                                            0, 5, 0.07f, 0.03f),
        attention_test::make_attention_case("attention wrapper full-kv", 4, 4, 64,
                                            6, 7, 0.05f, 0.025f),
        attention_test::make_attention_case("attention wrapper tiled-seq", 8, 2,
                                            128, 37, 40, 0.03f, 0.017f),
    };

    for (const auto& test_case : cases) {
        if (!run_case(test_case)) {
            return 1;
        }
    }

    std::cout << "[PASS] attention_bf16" << std::endl;
    return 0;
}
