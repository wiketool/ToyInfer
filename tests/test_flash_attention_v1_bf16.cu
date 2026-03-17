#include <iostream>

#include "../src/kernel.cu"
#include "flash_attention_test_common.h"
#include "test_helpers.h"

namespace {

constexpr uint32_t kOutputPadding = 0;

flash_attention_test::FlashAttentionCase make_causal_focus_case() {
    flash_attention_test::FlashAttentionCase test_case;
    test_case.label = "flash wrapper causal-focus";
    test_case.num_q_heads = 2;
    test_case.num_kv_heads = 1;
    test_case.heads_dim = 64;
    test_case.seq_len = 6;

    const uint32_t q_dim = test_case.num_q_heads * test_case.heads_dim;
    const uint32_t kv_dim = test_case.num_kv_heads * test_case.heads_dim;
    test_case.q.resize(test_case.seq_len * q_dim);
    test_case.ks.resize(test_case.seq_len * kv_dim);
    test_case.vs.resize(test_case.seq_len * kv_dim);

    for (uint32_t t = 0; t < test_case.seq_len; ++t) {
        for (uint32_t h = 0; h < test_case.num_q_heads; ++h) {
            for (uint32_t d = 0; d < test_case.heads_dim; ++d) {
                const uint32_t q_idx = t * q_dim + h * test_case.heads_dim + d;
                test_case.q[q_idx] = 0.25f + 0.004f * static_cast<float>(d) +
                                     0.015f * static_cast<float>(h) +
                                     0.01f * static_cast<float>(t % 2);
            }
        }
        for (uint32_t d = 0; d < test_case.heads_dim; ++d) {
            const uint32_t kv_idx = t * kv_dim + d;
            test_case.ks[kv_idx] = 0.22f * static_cast<float>(t + 1) +
                                   0.003f * static_cast<float>((d % 13) + 1);
            test_case.vs[kv_idx] = 0.5f * static_cast<float>(t + 1) -
                                   0.025f * static_cast<float>(d % 5) +
                                   0.01f * static_cast<float>(d % 3);
        }
    }

    return test_case;
}

template <uint32_t Bc, uint32_t Br, uint32_t HEAD_DIM>
bool run_case(const flash_attention_test::FlashAttentionCase& test_case,
              float atol = 7e-2f, float rtol = 7e-2f) {
    const auto q_bf16 = kernel_test::to_bf16(test_case.q);
    const auto ks_bf16 = kernel_test::to_bf16(test_case.ks);
    const auto vs_bf16 = kernel_test::to_bf16(test_case.vs);
    const auto q_q = kernel_test::to_float(q_bf16);
    const auto ks_q = kernel_test::to_float(ks_bf16);
    const auto vs_q = kernel_test::to_float(vs_bf16);

    const auto expected_output = flash_attention_test::reference_convert(
        flash_attention_test::reference_output(test_case, q_q, ks_q, vs_q));

    const uint32_t output_size =
        test_case.seq_len * test_case.num_q_heads * test_case.heads_dim;
    std::vector<toyinfer::bf16> initial_o(output_size + kOutputPadding);
    for (uint32_t i = 0; i < initial_o.size(); ++i) {
        initial_o[i] = kernel_test::float_to_bf16(
            -1.1f + 0.07f * static_cast<float>((i * 5) % 29));
    }

    const auto initial_o_q = kernel_test::to_float(initial_o);
    std::vector<float> expected_tail(initial_o_q.begin() + output_size,
                                     initial_o_q.end());

    kernel_test::DeviceBuffer<toyinfer::bf16> dev_q(q_bf16.size());
    kernel_test::DeviceBuffer<toyinfer::bf16> dev_ks(ks_bf16.size());
    kernel_test::DeviceBuffer<toyinfer::bf16> dev_vs(vs_bf16.size());
    kernel_test::DeviceBuffer<toyinfer::bf16> dev_o(initial_o.size());
    dev_q.copy_from_host(q_bf16);
    dev_ks.copy_from_host(ks_bf16);
    dev_vs.copy_from_host(vs_bf16);
    dev_o.copy_from_host(initial_o);

    toyinfer::flash_attention_v1_bf16<Bc, Br, HEAD_DIM>(
        dev_q.get(), dev_ks.get(), dev_vs.get(), dev_o.get(),
        test_case.num_q_heads, test_case.num_kv_heads, test_case.heads_dim,
        test_case.seq_len);
    TCUDA_CHECK(cudaDeviceSynchronize());

    std::vector<toyinfer::bf16> out_bf16;
    dev_o.copy_to_host(out_bf16);
    const auto out = kernel_test::to_float(out_bf16);

    const std::vector<float> out_valid(out.begin(), out.begin() + output_size);
    if (!kernel_test::expect_vector_close(
            out_valid, expected_output, atol, rtol,
            test_case.label + " output Bc=" + std::to_string(Bc) +
                " Br=" + std::to_string(Br))) {
        return false;
    }

    const std::vector<float> out_tail(out.begin() + output_size, out.end());
    return kernel_test::expect_vector_close(
        out_tail, expected_tail, 0.0f, 0.0f,
        test_case.label + " tail Bc=" + std::to_string(Bc) +
            " Br=" + std::to_string(Br));
}

}  // namespace

int main() {
    if (!kernel_test::ensure_cuda_device(true)) {
        return 0;
    }

    bool ok = true;
    ok = run_case<32, 2, 64>(flash_attention_test::make_case(
             "flash wrapper seq1 gqa", 2, 1, 64, 1, 0.07f, 0.03f)) &&
         ok;
    ok = run_case<32, 4, 64>(flash_attention_test::make_case(
             "flash wrapper exact-kv-tile", 4, 4, 64, 32, 0.05f, 0.021f)) &&
         ok;
    ok = run_case<32, 4, 128>(flash_attention_test::make_case(
             "flash wrapper gqa tail-tiles", 8, 2, 128, 37, 0.033f, 0.017f)) &&
         ok;
    ok = run_case<64, 4, 128>(flash_attention_test::make_case(
             "flash wrapper multi-kv-tile", 6, 3, 128, 65, 0.028f, 0.019f)) &&
         ok;
    ok = run_case<32, 4, 64>(flash_attention_test::make_case(
                                 "flash wrapper large-magnitude", 4, 2, 64, 19,
                                 0.061f, 0.027f, 12.0f, 10.0f, 1.7f),
                             1.2f, 1.2f) &&
         ok;
    ok = run_case<32, 2, 64>(make_causal_focus_case(), 9e-2f, 9e-2f) && ok;

    if (!ok) {
        return 1;
    }

    std::cout << "[PASS] flash_attention_v1_bf16" << std::endl;
    return 0;
}
