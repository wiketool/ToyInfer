#include "test_helpers.h"
#include "../src/kernel.cu"
#include "precompute_freq_test_common.h"

namespace {

bool run_case(const precompute_freq_test::Case& test_case) {
    std::vector<float> initial(test_case.buffer_size, 0.0f);
    for (int i = 0; i < test_case.buffer_size; ++i) {
        initial[i] = 4.0f - 0.21f * static_cast<float>(i);
    }
    const auto expected = precompute_freq_test::build_expected(test_case, initial);

    kernel_test::DeviceBuffer<float> dev_out(test_case.buffer_size);
    dev_out.copy_from_host(initial);

    toyinfer::precompute_freq_f32(dev_out.get(), test_case.head_dim,
                                  test_case.theta);
    TCUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> out;
    dev_out.copy_to_host(out);
    return kernel_test::expect_vector_close(out, expected, 1e-6f, 1e-6f,
                                            test_case.label + " wrapper");
}

}  // namespace

int main() {
    if (!kernel_test::ensure_cuda_device(false)) {
        return 0;
    }

    const std::vector<precompute_freq_test::Case> cases = {
        {"precompute wrapper head_dim=2", 2, 4, 2.0f},
        {"precompute wrapper theta=1", 258, 140, 1.0f},
        {"precompute wrapper partial-grid", 130, 80, 10000.0f},
        {"precompute wrapper large uneven", 514, 300, 123.5f},
    };

    for (const auto& test_case : cases) {
        if (!run_case(test_case)) {
            return 1;
        }
    }

    std::cout << "[PASS] precompute_freq_f32" << std::endl;
    return 0;
}
