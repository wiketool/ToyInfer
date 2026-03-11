#include "test_helpers.h"
#include "../src/kernel.cu"

int main() {
    if (!kernel_test::ensure_cuda_device(false)) {
        return 0;
    }

    constexpr int n = 1024;
    std::vector<float> host_a(n), host_b(n), host_c(n, 0.0f);
    for (int i = 0; i < n; ++i) {
        host_a[i] = std::sin(i * 0.13f);
        host_b[i] = std::cos(i * 0.07f);
    }

    float* dev_a = nullptr;
    float* dev_b = nullptr;
    float* dev_c = nullptr;
    TCUDA_CHECK(cudaMalloc(&dev_a, sizeof(float) * n));
    TCUDA_CHECK(cudaMalloc(&dev_b, sizeof(float) * n));
    TCUDA_CHECK(cudaMalloc(&dev_c, sizeof(float) * n));
    TCUDA_CHECK(cudaMemcpy(dev_a, host_a.data(), sizeof(float) * n,
                           cudaMemcpyHostToDevice));
    TCUDA_CHECK(cudaMemcpy(dev_b, host_b.data(), sizeof(float) * n,
                           cudaMemcpyHostToDevice));

    toyinfer::launch_add_kernel(dev_a, dev_b, dev_c, n);
    TCUDA_CHECK(cudaDeviceSynchronize());
    TCUDA_CHECK(cudaMemcpy(host_c.data(), dev_c, sizeof(float) * n,
                           cudaMemcpyDeviceToHost));

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    for (int i = 0; i < n; ++i) {
        const float expected = host_a[i] + host_b[i];
        if (!kernel_test::nearly_equal(host_c[i], expected, 1e-6f, 1e-6f)) {
            std::cerr << "[FAIL] add_kernel mismatch at " << i << ", expected "
                      << expected << ", got " << host_c[i] << std::endl;
            return 1;
        }
    }

    std::cout << "[PASS] add_kernel" << std::endl;
    return 0;
}
