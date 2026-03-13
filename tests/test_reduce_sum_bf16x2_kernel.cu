#include "test_helpers.h"
#include "../src/kernel.cu"
#include "rmsnorm_test_common.h"

namespace {

template <uint32_t NUM_THREADS>
void launch_kernel(const toyinfer::bf16* input, float* sum, uint32_t size) {
    dim3 block_dim{NUM_THREADS};
    dim3 grid_dim{(size + block_dim.x * 2 - 1) / (block_dim.x * 2)};
    toyinfer::reduce_sum_bf16x2_kernel<<<grid_dim, block_dim>>>(input, sum, size);
    TCUDA_CHECK(cudaGetLastError());
}

bool run_case(const rmsnorm_test::Case& test_case, float init_sum) {
    const auto input_bf16 = kernel_test::to_bf16(test_case.input);
    const auto input_q = kernel_test::to_float(input_bf16);
    const float expected_sum =
        init_sum + rmsnorm_test::reference_sum(input_q, test_case.size);

    kernel_test::DeviceBuffer<toyinfer::bf16> dev_input(test_case.buffer_size);
    kernel_test::DeviceBuffer<float> dev_sum(1);
    dev_input.copy_from_host(input_bf16);
    kernel_test::copy_scalar_to_device(init_sum, dev_sum.get());

    launch_kernel<rmsnorm_test::kThreads>(dev_input.get(), dev_sum.get(),
                                          test_case.size);
    TCUDA_CHECK(cudaDeviceSynchronize());

    const float gpu_sum = kernel_test::copy_scalar_from_device(dev_sum.get());
    if (!kernel_test::nearly_equal(gpu_sum, expected_sum, 5e-2f, 5e-3f)) {
        std::cerr << "[FAIL] " << test_case.label
                  << " reduce_sum expected " << expected_sum << ", got "
                  << gpu_sum << std::endl;
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
    tiny.label = "reduce_sum tiny";
    tiny.size = 2;
    tiny.buffer_size = 6;
    tiny.eps = 1e-5f;
    tiny.input = {0.0f, -1.25f, 0.75f, -0.5f, 1.0f, -1.5f};
    tiny.weight = {1.0f, -0.75f, 0.25f, -1.25f, 0.5f, 0.875f};
    cases.push_back(tiny);
    cases.push_back(rmsnorm_test::make_case("reduce_sum partial-block", 258, 264,
                                            1e-5f, 0.11f, 0.07f));
    cases.push_back(rmsnorm_test::make_case("reduce_sum multi-block", 1024, 1030,
                                            1e-3f, 0.07f, 0.03f));
    rmsnorm_test::Case zero_input;
    zero_input.label = "reduce_sum zero-input";
    zero_input.size = 514;
    zero_input.buffer_size = 520;
    zero_input.eps = 1e-4f;
    zero_input.input.assign(zero_input.buffer_size, 0.0f);
    zero_input.weight.resize(zero_input.buffer_size, 1.0f);
    cases.push_back(zero_input);

    for (const auto& test_case : cases) {
        if (!run_case(test_case, 1.75f)) {
            return 1;
        }
    }

    std::cout << "[PASS] reduce_sum_bf16x2_kernel" << std::endl;
    return 0;
}
