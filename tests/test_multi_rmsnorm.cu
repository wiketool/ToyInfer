#include "test_helpers.h"
#include "../src/kernel.cu"

#include <string>

namespace {

struct MultiRmsNormCase {
    std::string label;
    uint32_t num_heads = 0;
    uint32_t head_dim = 0;
    float eps = 1e-5f;
    bool in_place = false;
    std::vector<float> input;
    std::vector<float> weight;
};

bool run_case(const MultiRmsNormCase& test_case) {
    const uint32_t size = test_case.num_heads * test_case.head_dim;
    const auto input_bf16 = kernel_test::to_bf16(test_case.input);
    const auto weight_bf16 = kernel_test::to_bf16(test_case.weight);
    const auto input_q = kernel_test::to_float(input_bf16);
    const auto weight_q = kernel_test::to_float(weight_bf16);

    std::vector<float> expected(size, 0.0f);
    for (uint32_t h = 0; h < test_case.num_heads; ++h) {
        const uint32_t offset = h * test_case.head_dim;
        float head_sum = 0.0f;
        for (uint32_t i = 0; i < test_case.head_dim; ++i) {
            head_sum += input_q[offset + i] * input_q[offset + i];
        }
        const float inv_rms =
            1.0f / std::sqrt(head_sum / test_case.head_dim + test_case.eps);
        for (uint32_t i = 0; i < test_case.head_dim; ++i) {
            expected[offset + i] = kernel_test::round_to_bf16(
                input_q[offset + i] * inv_rms * weight_q[i]);
        }
    }

    kernel_test::DeviceBuffer<toyinfer::bf16> dev_input(size);
    kernel_test::DeviceBuffer<toyinfer::bf16> dev_weight(test_case.head_dim);
    dev_input.copy_from_host(input_bf16);
    dev_weight.copy_from_host(weight_bf16);

    std::vector<toyinfer::bf16> out_bf16;
    if (test_case.in_place) {
        toyinfer::multi_rmsnorm_bf16(dev_input.get(), dev_weight.get(),
                                     dev_input.get(), test_case.eps,
                                     test_case.num_heads, test_case.head_dim);
        TCUDA_CHECK(cudaDeviceSynchronize());
        dev_input.copy_to_host(out_bf16);
    } else {
        kernel_test::DeviceBuffer<toyinfer::bf16> dev_output(size);
        toyinfer::multi_rmsnorm_bf16(dev_input.get(), dev_weight.get(),
                                     dev_output.get(), test_case.eps,
                                     test_case.num_heads, test_case.head_dim);
        TCUDA_CHECK(cudaDeviceSynchronize());
        dev_output.copy_to_host(out_bf16);
    }

    const auto out = kernel_test::to_float(out_bf16);
    return kernel_test::expect_vector_close(out, expected, 2e-2f, 2e-2f,
                                            test_case.label);
}

}  // namespace

int main() {
    if (!kernel_test::ensure_cuda_device(true)) {
        return 0;
    }

    std::vector<MultiRmsNormCase> cases;

    MultiRmsNormCase single_head;
    single_head.label = "multi_rmsnorm single-head";
    single_head.num_heads = 1;
    single_head.head_dim = 64;
    single_head.eps = 1e-5f;
    single_head.in_place = false;
    single_head.input.resize(single_head.num_heads * single_head.head_dim);
    single_head.weight.resize(single_head.head_dim);
    for (uint32_t i = 0; i < single_head.head_dim; ++i) {
        single_head.input[i] =
            0.4f * std::sin((i + 1) * 0.13f) - 0.05f * static_cast<float>(i % 4);
        single_head.weight[i] =
            0.75f + 0.01f * static_cast<float>((i * 7) % 9) -
            0.03f * static_cast<float>(i % 2);
    }
    cases.push_back(single_head);

    MultiRmsNormCase multi_head_out;
    multi_head_out.label = "multi_rmsnorm shared-weight out-of-place";
    multi_head_out.num_heads = 4;
    multi_head_out.head_dim = 128;
    multi_head_out.eps = 1e-5f;
    multi_head_out.in_place = false;
    multi_head_out.input.resize(multi_head_out.num_heads * multi_head_out.head_dim);
    multi_head_out.weight.resize(multi_head_out.head_dim);
    for (uint32_t h = 0; h < multi_head_out.num_heads; ++h) {
        for (uint32_t i = 0; i < multi_head_out.head_dim; ++i) {
            const uint32_t idx = h * multi_head_out.head_dim + i;
            multi_head_out.input[idx] =
                0.35f * std::cos((idx + 3) * 0.09f) +
                0.02f * static_cast<float>((h + i) % 5) -
                0.015f * static_cast<float>(h);
        }
    }
    for (uint32_t i = 0; i < multi_head_out.head_dim; ++i) {
        multi_head_out.weight[i] =
            0.62f + 0.008f * static_cast<float>((i * 11) % 13) -
            0.01f * static_cast<float>(i % 3);
    }
    cases.push_back(multi_head_out);

    MultiRmsNormCase multi_head_in_place;
    multi_head_in_place.label = "multi_rmsnorm in-place";
    multi_head_in_place.num_heads = 5;
    multi_head_in_place.head_dim = 64;
    multi_head_in_place.eps = 1e-3f;
    multi_head_in_place.in_place = true;
    multi_head_in_place.input.resize(multi_head_in_place.num_heads *
                                     multi_head_in_place.head_dim);
    multi_head_in_place.weight.resize(multi_head_in_place.head_dim);
    for (uint32_t h = 0; h < multi_head_in_place.num_heads; ++h) {
        for (uint32_t i = 0; i < multi_head_in_place.head_dim; ++i) {
            const uint32_t idx = h * multi_head_in_place.head_dim + i;
            multi_head_in_place.input[idx] =
                0.28f * std::sin((idx + 5) * 0.17f) -
                0.19f * std::cos((h + 1) * (i + 1) * 0.03f) +
                0.01f * static_cast<float>(i % 7);
        }
    }
    for (uint32_t i = 0; i < multi_head_in_place.head_dim; ++i) {
        multi_head_in_place.weight[i] =
            -0.55f + 0.012f * static_cast<float>((i * 5) % 19);
    }
    cases.push_back(multi_head_in_place);

    for (const auto& test_case : cases) {
        if (!run_case(test_case)) {
            return 1;
        }
    }

    std::cout << "[PASS] multi_rmsnorm_bf16" << std::endl;
    return 0;
}
