#include "test_helpers.h"
#include "../src/kernel.cu"
#include "add_test_common.h"

namespace {

template <uint32_t NUM_THREADS>
void launch_kernel(const toyinfer::bf16* residual, toyinfer::bf16* hidden,
                   uint32_t size) {
    dim3 block_dim{NUM_THREADS};
    dim3 grid_dim{(size + block_dim.x * 2 - 1) / (block_dim.x * 2)};
    toyinfer::add_bf16_kernel<<<grid_dim, block_dim>>>(residual, hidden, size);
    TCUDA_CHECK(cudaGetLastError());
}

bool run_case(const add_test::AddCase& test_case) {
    const auto residual_bf16 = kernel_test::to_bf16(test_case.residual);
    auto hidden_bf16 = kernel_test::to_bf16(test_case.hidden);
    const auto residual_q = kernel_test::to_float(residual_bf16);
    const auto hidden_q = kernel_test::to_float(hidden_bf16);

    kernel_test::DeviceBuffer<toyinfer::bf16> dev_residual(test_case.buffer_size);
    kernel_test::DeviceBuffer<toyinfer::bf16> dev_hidden(test_case.buffer_size);
    dev_residual.copy_from_host(residual_bf16);
    dev_hidden.copy_from_host(hidden_bf16);

    auto expected = add_test::apply_reference(hidden_q, residual_q, test_case.size);
    launch_kernel<add_test::kThreads>(dev_residual.get(), dev_hidden.get(),
                                      test_case.size);
    TCUDA_CHECK(cudaDeviceSynchronize());

    dev_hidden.copy_to_host(hidden_bf16);
    auto out = kernel_test::to_float(hidden_bf16);
    if (!kernel_test::expect_vector_close(out, expected, 2e-2f, 2e-2f,
                                          test_case.label + " kernel first")) {
        return false;
    }

    if (!test_case.run_twice) {
        return true;
    }

    expected = add_test::apply_reference(expected, residual_q, test_case.size);
    launch_kernel<add_test::kThreads>(dev_residual.get(), dev_hidden.get(),
                                      test_case.size);
    TCUDA_CHECK(cudaDeviceSynchronize());
    dev_hidden.copy_to_host(hidden_bf16);
    out = kernel_test::to_float(hidden_bf16);
    return kernel_test::expect_vector_close(out, expected, 2e-2f, 2e-2f,
                                            test_case.label + " kernel second");
}

}  // namespace

int main() {
    if (!kernel_test::ensure_cuda_device(true)) {
        return 0;
    }

    std::vector<add_test::AddCase> cases;
    add_test::AddCase tiny;
    tiny.label = "add tiny";
    tiny.size = 2;
    tiny.buffer_size = 6;
    tiny.run_twice = true;
    tiny.residual = {0.5f, -0.25f, 1.25f, -1.5f, 0.75f, -0.5f};
    tiny.hidden = {-1.0f, 0.75f, -0.125f, 0.875f, -1.75f, 1.5f};
    cases.push_back(tiny);
    cases.push_back(add_test::make_case("add partial-block", 514, 520, false,
                                        0.09f, 0.07f, 0.6f));
    cases.push_back(add_test::make_case("add multi-block", 1536, 1542, true,
                                        0.05f, 0.04f, -0.45f));

    for (const auto& test_case : cases) {
        if (!run_case(test_case)) {
            return 1;
        }
    }

    std::cout << "[PASS] add_bf16_kernel" << std::endl;
    return 0;
}
