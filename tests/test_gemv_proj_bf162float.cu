#include "test_helpers.h"
#include "../src/kernel.cu"
#include "gemv_test_common.h"

namespace {

bool run_case(const gemv_test::Case& test_case) {
    const auto x_bf16 = kernel_test::to_bf16(test_case.x);
    const auto w_bf16 = kernel_test::to_bf16(test_case.w);
    const auto x_q = kernel_test::to_float(x_bf16);
    const auto w_q = kernel_test::to_float(w_bf16);
    const auto expected_sum = gemv_test::reference_sum(test_case, x_q, w_q);

    std::vector<float> initial_out(test_case.output_buffer_size, 0.0f);
    for (uint32_t i = 0; i < test_case.output_buffer_size; ++i) {
        initial_out[i] = 1.5f - 0.11f * static_cast<float>((i * 5) % 9);
    }
    auto expected = initial_out;
    for (uint32_t i = 0; i < test_case.M; ++i) {
        expected[i] = expected_sum[i];
    }

    kernel_test::DeviceBuffer<toyinfer::bf16> dev_w(test_case.M * test_case.N);
    kernel_test::DeviceBuffer<toyinfer::bf16> dev_x(test_case.N);
    kernel_test::DeviceBuffer<float> dev_y(test_case.output_buffer_size);
    dev_w.copy_from_host(w_bf16);
    dev_x.copy_from_host(x_bf16);
    dev_y.copy_from_host(initial_out);

    toyinfer::gemv_proj_bf162float<gemv_test::kThreads>(
        dev_w.get(), dev_x.get(), dev_y.get(), test_case.M, test_case.N);
    TCUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> out;
    dev_y.copy_to_host(out);
    return kernel_test::expect_vector_close(out, expected, 5e-2f, 5e-2f,
                                            test_case.label + " float wrapper");
}

}  // namespace

int main() {
    if (!kernel_test::ensure_cuda_device(true)) {
        return 0;
    }

    const std::vector<gemv_test::Case> cases = {
        gemv_test::make_case("gemv proj float tiny", 3, 2, 5, 0.19f, 0.02f),
        gemv_test::make_case("gemv proj float multi-loop", 17, 514, 21, 0.09f,
                             0.005f),
        gemv_test::make_case("gemv proj float lmhead-shape-like", 257, 1024, 260,
                             0.04f, 0.003f),
    };

    for (const auto& test_case : cases) {
        if (!run_case(test_case)) {
            return 1;
        }
    }

    std::cout << "[PASS] gemv_proj_bf162float" << std::endl;
    return 0;
}
