#include "test_helpers.h"
#include "../src/kernel.cu"
#include "rmsnorm_test_common.h"

namespace {

template <uint32_t NUM_THREADS>
void launch_kernel(const toyinfer::bf16* input, const toyinfer::bf16* weight,
                   toyinfer::bf16* output, const float* sum, float eps,
                   uint32_t size) {
    dim3 block_dim{NUM_THREADS};
    dim3 grid_dim{(size + block_dim.x * 2 - 1) / (block_dim.x * 2)};
    toyinfer::rmsnorm_bf16x2_kernel<<<grid_dim, block_dim>>>(
        input, weight, output, sum, eps, size);
    TCUDA_CHECK(cudaGetLastError());
}

bool run_case(const rmsnorm_test::Case& test_case) {
    const auto input_bf16 = kernel_test::to_bf16(test_case.input);
    const auto weight_bf16 = kernel_test::to_bf16(test_case.weight);
    const auto input_q = kernel_test::to_float(input_bf16);
    const auto weight_q = kernel_test::to_float(weight_bf16);
    const float ref_sum = rmsnorm_test::reference_sum(input_q, test_case.size);

    std::vector<float> initial_out(test_case.buffer_size, 0.0f);
    for (uint32_t i = 0; i < test_case.buffer_size; ++i) {
        initial_out[i] = -1.5f + 0.12f * static_cast<float>(i % 11);
    }
    auto initial_out_bf16 = kernel_test::to_bf16(initial_out);
    const auto initial_out_q = kernel_test::to_float(initial_out_bf16);
    const auto expected = rmsnorm_test::reference_output(
        input_q, weight_q, initial_out_q, ref_sum, test_case.eps, test_case.size);

    kernel_test::DeviceBuffer<toyinfer::bf16> dev_input(test_case.buffer_size);
    kernel_test::DeviceBuffer<toyinfer::bf16> dev_weight(test_case.buffer_size);
    kernel_test::DeviceBuffer<toyinfer::bf16> dev_output(test_case.buffer_size);
    kernel_test::DeviceBuffer<float> dev_sum(1);
    dev_input.copy_from_host(input_bf16);
    dev_weight.copy_from_host(weight_bf16);
    dev_output.copy_from_host(initial_out_bf16);
    kernel_test::copy_scalar_to_device(ref_sum, dev_sum.get());

    launch_kernel<rmsnorm_test::kThreads>(dev_input.get(), dev_weight.get(),
                                          dev_output.get(), dev_sum.get(),
                                          test_case.eps, test_case.size);
    TCUDA_CHECK(cudaDeviceSynchronize());

    std::vector<toyinfer::bf16> out_bf16;
    dev_output.copy_to_host(out_bf16);
    const auto out = kernel_test::to_float(out_bf16);
    if (!kernel_test::expect_vector_close(out, expected, 2e-2f, 2e-2f,
                                          test_case.label + " rmsnorm kernel")) {
        return false;
    }

    const float sum_after = kernel_test::copy_scalar_from_device(dev_sum.get());
    if (!kernel_test::nearly_equal(sum_after, ref_sum, 1e-6f, 1e-6f)) {
        std::cerr << "[FAIL] " << test_case.label
                  << " rmsnorm kernel modified sum, expected " << ref_sum
                  << ", got " << sum_after << std::endl;
        return false;
    }
    return true;
}

}  // namespace

int main() {
    if (!kernel_test::ensure_cuda_device(true)) {
        return 0;
    }

    std::vector<rmsnorm_test::Case> cases;
    rmsnorm_test::Case tiny;
    tiny.label = "rmsnorm kernel tiny";
    tiny.size = 2;
    tiny.buffer_size = 6;
    tiny.eps = 1e-5f;
    tiny.input = {0.0f, -1.25f, 0.75f, -0.5f, 1.0f, -1.5f};
    tiny.weight = {1.0f, -0.75f, 0.25f, -1.25f, 0.5f, 0.875f};
    cases.push_back(tiny);
    cases.push_back(rmsnorm_test::make_case("rmsnorm kernel partial-block", 258,
                                            264, 1e-5f, 0.11f, 0.07f));
    cases.push_back(rmsnorm_test::make_case("rmsnorm kernel multi-block", 1024,
                                            1030, 1e-3f, 0.07f, 0.03f));
    rmsnorm_test::Case zero_input;
    zero_input.label = "rmsnorm kernel zero-input";
    zero_input.size = 514;
    zero_input.buffer_size = 520;
    zero_input.eps = 1e-4f;
    zero_input.input.assign(zero_input.buffer_size, 0.0f);
    zero_input.weight.resize(zero_input.buffer_size);
    for (uint32_t i = 0; i < zero_input.buffer_size; ++i) {
        zero_input.weight[i] =
            -1.2f + 0.01f * static_cast<float>((i * 9) % 23);
    }
    cases.push_back(zero_input);

    for (const auto& test_case : cases) {
        if (!run_case(test_case)) {
            return 1;
        }
    }

    std::cout << "[PASS] rmsnorm_bf16x2_kernel" << std::endl;
    return 0;
}
