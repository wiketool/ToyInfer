#include "test_helpers.h"
#include "../src/kernel.cu"
#include "gemv_test_common.h"

namespace {

template <uint32_t NUM_THREADS>
void launch_kernel(const toyinfer::bf16* W, const toyinfer::bf16* x,
                   toyinfer::bf16* y, uint32_t M, uint32_t N) {
    dim3 block_dim{NUM_THREADS};
    dim3 grid_dim{M};
    toyinfer::gemv_bf16_kernel<NUM_THREADS><<<grid_dim, block_dim>>>(W, x, y, M,
                                                                     N);
    TCUDA_CHECK(cudaGetLastError());
}

bool run_case(const gemv_test::Case& test_case) {
    const auto x_bf16 = kernel_test::to_bf16(test_case.x);
    const auto w_bf16 = kernel_test::to_bf16(test_case.w);
    const auto x_q = kernel_test::to_float(x_bf16);
    const auto w_q = kernel_test::to_float(w_bf16);
    const auto expected_sum = gemv_test::reference_sum(test_case, x_q, w_q);

    std::vector<float> initial_out(test_case.output_buffer_size, 0.0f);
    for (uint32_t i = 0; i < test_case.output_buffer_size; ++i) {
        initial_out[i] = -0.85f + 0.09f * static_cast<float>(i % 7);
    }
    auto initial_out_bf16 = kernel_test::to_bf16(initial_out);
    auto expected = kernel_test::to_float(initial_out_bf16);
    for (uint32_t i = 0; i < test_case.M; ++i) {
        expected[i] = kernel_test::round_to_bf16(expected_sum[i]);
    }

    kernel_test::DeviceBuffer<toyinfer::bf16> dev_w(test_case.M * test_case.N);
    kernel_test::DeviceBuffer<toyinfer::bf16> dev_x(test_case.N);
    kernel_test::DeviceBuffer<toyinfer::bf16> dev_y(test_case.output_buffer_size);
    dev_w.copy_from_host(w_bf16);
    dev_x.copy_from_host(x_bf16);
    dev_y.copy_from_host(initial_out_bf16);

    launch_kernel<gemv_test::kThreads>(dev_w.get(), dev_x.get(), dev_y.get(),
                                       test_case.M, test_case.N);
    TCUDA_CHECK(cudaDeviceSynchronize());

    std::vector<toyinfer::bf16> out_bf16;
    dev_y.copy_to_host(out_bf16);
    const auto out = kernel_test::to_float(out_bf16);
    return kernel_test::expect_vector_close(out, expected, 8e-2f, 8e-2f,
                                            test_case.label + " bf16 kernel");
}

}  // namespace

int main() {
    if (!kernel_test::ensure_cuda_device(true)) {
        return 0;
    }

    const std::vector<gemv_test::Case> cases = {
        gemv_test::make_case("gemv bf16 kernel tiny", 1, 2, 4, 0.17f, 0.01f),
        gemv_test::make_case("gemv bf16 kernel single-iteration", 9, 256, 12,
                             0.07f, 0.006f),
        gemv_test::make_case("gemv bf16 kernel multi-loop", 513, 514, 520, 0.05f,
                             0.004f),
    };

    for (const auto& test_case : cases) {
        if (!run_case(test_case)) {
            return 1;
        }
    }

    std::cout << "[PASS] gemv_bf16_kernel" << std::endl;
    return 0;
}
