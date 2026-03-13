#include "test_helpers.h"
#include "../src/kernel.cu"
#include "multi_rmsnorm_test_common.h"

#include <algorithm>

namespace {

void launch_kernel(const toyinfer::bf16* input, const toyinfer::bf16* weight,
                   toyinfer::bf16* output, float eps, uint32_t num_heads,
                   uint32_t head_dim) {
    dim3 block_dim{head_dim / 2};
    dim3 grid_dim{num_heads};
    toyinfer::multi_rmsnorm_bf16x2_kernel<<<grid_dim, block_dim>>>(
        input, weight, output, eps, head_dim);
    TCUDA_CHECK(cudaGetLastError());
}

bool run_case(const multi_rmsnorm_test::Case& test_case) {
    auto input_bf16 = kernel_test::to_bf16(test_case.input);
    const auto weight_bf16 = kernel_test::to_bf16(test_case.weight);
    const auto input_q = kernel_test::to_float(input_bf16);
    const auto weight_q = kernel_test::to_float(weight_bf16);

    kernel_test::DeviceBuffer<toyinfer::bf16> dev_input(test_case.buffer_size);
    kernel_test::DeviceBuffer<toyinfer::bf16> dev_weight(test_case.head_dim);
    dev_input.copy_from_host(input_bf16);
    dev_weight.copy_from_host(weight_bf16);

    std::vector<toyinfer::bf16> out_bf16;
    if (test_case.in_place) {
        const auto expected =
            multi_rmsnorm_test::reference_output(test_case, input_q, weight_q, input_q);
        launch_kernel(dev_input.get(), dev_weight.get(), dev_input.get(),
                      test_case.eps, test_case.num_heads, test_case.head_dim);
        TCUDA_CHECK(cudaDeviceSynchronize());
        dev_input.copy_to_host(out_bf16);
        const auto out = kernel_test::to_float(out_bf16);
        return kernel_test::expect_vector_close(out, expected, 2e-2f, 2e-2f,
                                                test_case.label + " kernel");
    }

    std::vector<float> initial_out(test_case.buffer_size, 0.0f);
    for (uint32_t i = 0; i < test_case.buffer_size; ++i) {
        initial_out[i] = 1.4f - 0.08f * static_cast<float>((i * 3) % 11);
    }
    auto initial_out_bf16 = kernel_test::to_bf16(initial_out);
    const auto initial_out_q = kernel_test::to_float(initial_out_bf16);
    const auto expected = multi_rmsnorm_test::reference_output(
        test_case, input_q, weight_q, initial_out_q);

    kernel_test::DeviceBuffer<toyinfer::bf16> dev_output(test_case.buffer_size);
    dev_output.copy_from_host(initial_out_bf16);
    launch_kernel(dev_input.get(), dev_weight.get(), dev_output.get(),
                  test_case.eps, test_case.num_heads, test_case.head_dim);
    TCUDA_CHECK(cudaDeviceSynchronize());
    dev_output.copy_to_host(out_bf16);
    const auto out = kernel_test::to_float(out_bf16);
    return kernel_test::expect_vector_close(out, expected, 2e-2f, 2e-2f,
                                            test_case.label + " kernel");
}

}  // namespace

int main() {
    if (!kernel_test::ensure_cuda_device(true)) {
        return 0;
    }

    std::vector<multi_rmsnorm_test::Case> cases;
    cases.push_back(multi_rmsnorm_test::make_case("multi_rmsnorm kernel single-head",
                                                  1, 64, 68, 1e-5f, false,
                                                  0.13f, 0.09f));
    cases.push_back(multi_rmsnorm_test::make_case(
        "multi_rmsnorm kernel out-of-place", 4, 128, 4 * 128 + 6, 1e-5f, false,
        0.09f, 0.05f));
    cases.push_back(multi_rmsnorm_test::make_case(
        "multi_rmsnorm kernel in-place", 5, 64, 5 * 64 + 4, 1e-3f, true, 0.17f,
        0.03f));
    auto zero_case = multi_rmsnorm_test::make_case("multi_rmsnorm kernel zero-input",
                                                   3, 64, 3 * 64 + 8, 1e-4f,
                                                   false, 0.07f, 0.04f);
    std::fill(zero_case.input.begin(),
              zero_case.input.begin() +
                  zero_case.num_heads * zero_case.head_dim,
              0.0f);
    cases.push_back(zero_case);

    for (const auto& test_case : cases) {
        if (!run_case(test_case)) {
            return 1;
        }
    }

    std::cout << "[PASS] multi_rmsnorm_bf16x2_kernel" << std::endl;
    return 0;
}
