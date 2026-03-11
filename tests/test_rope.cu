#include "test_helpers.h"
#include "../src/kernel.cu"

int main() {
    if (!kernel_test::ensure_cuda_device(true)) {
        return 0;
    }

    constexpr uint32_t nums_head = 3;
    constexpr uint32_t head_dim = 16;
    constexpr uint32_t pos = 11;
    std::vector<float> qk_f(nums_head * head_dim);
    std::vector<float> inv_freq(head_dim / 2);
    for (uint32_t i = 0; i < qk_f.size(); ++i) {
        qk_f[i] = std::sin(i * 0.03f) + 0.002f * static_cast<float>(i % 5);
    }
    for (uint32_t i = 0; i < inv_freq.size(); ++i) {
        inv_freq[i] = 0.001f * static_cast<float>(i + 1);
    }

    auto qk_bf16 = kernel_test::to_bf16(qk_f);
    const auto qk_q = kernel_test::to_float(qk_bf16);
    std::vector<float> expected(qk_q.size(), 0.0f);
    for (uint32_t h = 0; h < nums_head; ++h) {
        const uint32_t offset = h * head_dim;
        for (uint32_t i = 0; i < head_dim / 2; ++i) {
            const float real = qk_q[offset + i];
            const float imag = qk_q[offset + i + head_dim / 2];
            const float cos_v = std::cos(pos * inv_freq[i]);
            const float sin_v = std::sin(pos * inv_freq[i]);
            expected[offset + i] = kernel_test::round_to_bf16(cos_v * real - sin_v * imag);
            expected[offset + i + head_dim / 2] =
                kernel_test::round_to_bf16(cos_v * imag + sin_v * real);
        }
    }

    toyinfer::bf16* dev_qk = nullptr;
    float* dev_inv_freq = nullptr;
    TCUDA_CHECK(cudaMalloc(&dev_qk, sizeof(toyinfer::bf16) * qk_bf16.size()));
    TCUDA_CHECK(cudaMalloc(&dev_inv_freq, sizeof(float) * inv_freq.size()));
    TCUDA_CHECK(cudaMemcpy(dev_qk, qk_bf16.data(),
                           sizeof(toyinfer::bf16) * qk_bf16.size(),
                           cudaMemcpyHostToDevice));
    TCUDA_CHECK(cudaMemcpy(dev_inv_freq, inv_freq.data(),
                           sizeof(float) * inv_freq.size(),
                           cudaMemcpyHostToDevice));

    toyinfer::rope_bf16(dev_qk, dev_inv_freq, pos, nums_head, head_dim);
    TCUDA_CHECK(cudaDeviceSynchronize());
    TCUDA_CHECK(cudaMemcpy(qk_bf16.data(), dev_qk,
                           sizeof(toyinfer::bf16) * qk_bf16.size(),
                           cudaMemcpyDeviceToHost));

    cudaFree(dev_qk);
    cudaFree(dev_inv_freq);
    const auto out = kernel_test::to_float(qk_bf16);

    for (uint32_t i = 0; i < out.size(); ++i) {
        if (!kernel_test::nearly_equal(out[i], expected[i], 2e-2f, 2e-2f)) {
            std::cerr << "[FAIL] rope mismatch at " << i << ", expected "
                      << expected[i] << ", got " << out[i] << std::endl;
            return 1;
        }
    }

    std::cout << "[PASS] rope_bf16" << std::endl;
    return 0;
}
