#include "test_helpers.h"
#include "../src/kernel.cu"

#include <string>

namespace {

bool run_case(const std::string& label, int head_dim, float theta) {
    kernel_test::DeviceBuffer<float> dev_out(head_dim / 2);

    toyinfer::precompute_freq_f32(dev_out.get(), head_dim, theta);
    TCUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> host_out;
    dev_out.copy_to_host(host_out);

    std::vector<float> expected(head_dim / 2, 0.0f);
    for (int i = 0; i < head_dim / 2; ++i) {
        expected[i] = 1.0f / std::pow(theta, (2.0f * i) / head_dim);
    }

    return kernel_test::expect_vector_close(host_out, expected, 1e-6f, 1e-6f,
                                            label);
}

}  // namespace

int main() {
    if (!kernel_test::ensure_cuda_device(false)) {
        return 0;
    }

    if (!run_case("precompute head_dim=2 theta=2", 2, 2.0f)) {
        return 1;
    }
    if (!run_case("precompute head_dim=128 theta=10000", 128, 10000.0f)) {
        return 1;
    }
    if (!run_case("precompute head_dim=258 theta=1", 258, 1.0f)) {
        return 1;
    }
    if (!run_case("precompute head_dim=514 theta=123.5", 514, 123.5f)) {
        return 1;
    }

    std::cout << "[PASS] precompute_freq_f32" << std::endl;
    return 0;
}
