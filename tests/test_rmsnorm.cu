#include "test_helpers.h"
#include "../src/kernel.cu"

#include <string>

namespace {

struct RmsNormCase {
    std::string label;
    uint32_t size = 0;
    float eps = 1e-5f;
    std::vector<float> input;
    std::vector<float> weight;
};

bool run_case(const RmsNormCase& test_case, float* dev_sum) {
    const auto input_bf16 = kernel_test::to_bf16(test_case.input);
    const auto weight_bf16 = kernel_test::to_bf16(test_case.weight);
    const auto input_q = kernel_test::to_float(input_bf16);
    const auto weight_q = kernel_test::to_float(weight_bf16);

    float ref_sum = 0.0f;
    for (uint32_t i = 0; i < test_case.size; ++i) {
        ref_sum += input_q[i] * input_q[i];
    }

    const float inv_rms =
        1.0f / std::sqrt(ref_sum / test_case.size + test_case.eps);
    std::vector<float> expected(test_case.size, 0.0f);
    for (uint32_t i = 0; i < test_case.size; ++i) {
        expected[i] = kernel_test::round_to_bf16(input_q[i] * inv_rms *
                                                 weight_q[i]);
    }

    kernel_test::DeviceBuffer<toyinfer::bf16> dev_input(test_case.size);
    kernel_test::DeviceBuffer<toyinfer::bf16> dev_weight(test_case.size);
    kernel_test::DeviceBuffer<toyinfer::bf16> dev_output(test_case.size);
    dev_input.copy_from_host(input_bf16);
    dev_weight.copy_from_host(weight_bf16);

    toyinfer::rmsnorm_bf16<256>(dev_input.get(), dev_weight.get(),
                                dev_output.get(), dev_sum, test_case.eps,
                                test_case.size);
    TCUDA_CHECK(cudaDeviceSynchronize());

    std::vector<toyinfer::bf16> out_bf16;
    dev_output.copy_to_host(out_bf16);
    const auto out = kernel_test::to_float(out_bf16);
    const float gpu_sum = kernel_test::copy_scalar_from_device(dev_sum);

    if (!kernel_test::nearly_equal(gpu_sum, ref_sum, 5e-2f, 5e-3f)) {
        std::cerr << "[FAIL] " << test_case.label << " sum mismatch, expected "
                  << ref_sum << ", got " << gpu_sum << std::endl;
        return false;
    }

    return kernel_test::expect_vector_close(out, expected, 2e-2f, 2e-2f,
                                            test_case.label);
}

}  // namespace

int main() {
    if (!kernel_test::ensure_cuda_device(true)) {
        return 0;
    }

    std::vector<RmsNormCase> cases;

    cases.push_back({"rmsnorm tiny", 2, 1e-5f, {0.0f, -1.25f}, {1.0f, -0.75f}});

    RmsNormCase single_block_partial;
    single_block_partial.label = "rmsnorm single-block partial";
    single_block_partial.size = 258;
    single_block_partial.eps = 1e-5f;
    single_block_partial.input.resize(single_block_partial.size);
    single_block_partial.weight.resize(single_block_partial.size);
    for (uint32_t i = 0; i < single_block_partial.size; ++i) {
        single_block_partial.input[i] =
            std::sin(i * 0.11f) - 0.35f * static_cast<float>(i % 3 == 0) +
            0.01f * static_cast<float>(i % 7);
        single_block_partial.weight[i] =
            (i % 2 == 0 ? 0.7f : -0.6f) +
            0.005f * static_cast<float>((i * 13) % 17);
    }
    cases.push_back(single_block_partial);

    RmsNormCase multi_block_full;
    multi_block_full.label = "rmsnorm multi-block";
    multi_block_full.size = 1024;
    multi_block_full.eps = 1e-3f;
    multi_block_full.input.resize(multi_block_full.size);
    multi_block_full.weight.resize(multi_block_full.size);
    for (uint32_t i = 0; i < multi_block_full.size; ++i) {
        multi_block_full.input[i] =
            0.45f * std::cos(i * 0.07f) + 0.03f * std::sin(i * 0.19f) -
            0.02f * static_cast<float>(i % 5);
        multi_block_full.weight[i] =
            0.8f + 0.004f * static_cast<float>((i * 5) % 31) -
            0.002f * static_cast<float>(i % 4);
    }
    cases.push_back(multi_block_full);

    RmsNormCase all_zero_input;
    all_zero_input.label = "rmsnorm zero-input resets sum";
    all_zero_input.size = 514;
    all_zero_input.eps = 1e-4f;
    all_zero_input.input.assign(all_zero_input.size, 0.0f);
    all_zero_input.weight.resize(all_zero_input.size);
    for (uint32_t i = 0; i < all_zero_input.size; ++i) {
        all_zero_input.weight[i] =
            -1.2f + 0.01f * static_cast<float>((i * 9) % 23);
    }
    cases.push_back(all_zero_input);

    kernel_test::DeviceBuffer<float> dev_sum(1);
    for (const auto& test_case : cases) {
        if (!run_case(test_case, dev_sum.get())) {
            return 1;
        }
    }

    std::cout << "[PASS] rmsnorm_bf16" << std::endl;
    return 0;
}
