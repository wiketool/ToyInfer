#include "test_helpers.h"
#include "../src/kernel.cu"

int main() {
    if (!kernel_test::ensure_cuda_device(false)) {
        return 0;
    }

    constexpr int head_dim = 128;
    constexpr float theta = 10000.0f;
    std::vector<float> host_out(head_dim / 2, 0.0f);

    float* dev_out = nullptr;
    TCUDA_CHECK(cudaMalloc(&dev_out, sizeof(float) * host_out.size()));

    toyinfer::precompute_freq_f32(dev_out, head_dim, theta);
    TCUDA_CHECK(cudaDeviceSynchronize());
    TCUDA_CHECK(cudaMemcpy(host_out.data(), dev_out, sizeof(float) * host_out.size(),
                           cudaMemcpyDeviceToHost));
    cudaFree(dev_out);

    for (int i = 0; i < head_dim / 2; ++i) {
        const float expected = 1.0f / std::pow(theta, (2.0f * i) / head_dim);
        if (!kernel_test::nearly_equal(host_out[i], expected, 1e-6f, 1e-6f)) {
            std::cerr << "[FAIL] precompute_freq mismatch at " << i
                      << ", expected " << expected << ", got " << host_out[i]
                      << std::endl;
            return 1;
        }
    }

    std::cout << "[PASS] precompute_freq_f32" << std::endl;
    return 0;
}
