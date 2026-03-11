#include "test_helpers.h"
#include "../src/kernel.cu"

int main() {
    if (!kernel_test::ensure_cuda_device(true)) {
        return 0;
    }

    constexpr uint32_t size = 512;
    constexpr float eps = 1e-5f;
    std::vector<float> input_f(size), weight_f(size);
    for (uint32_t i = 0; i < size; ++i) {
        input_f[i] = std::sin(i * 0.11f) + 0.005f * static_cast<float>(i % 7);
        weight_f[i] = 0.8f + 0.002f * static_cast<float>((i * 13) % 31);
    }

    const auto input_bf16 = kernel_test::to_bf16(input_f);
    const auto weight_bf16 = kernel_test::to_bf16(weight_f);
    const auto input_q = kernel_test::to_float(input_bf16);
    const auto weight_q = kernel_test::to_float(weight_bf16);
    std::vector<float> expected(size);

    float ref_sum = 0.0f;
    for (uint32_t i = 0; i < size; ++i) {
        ref_sum += input_q[i] * input_q[i];
    }
    const float inv_rms = 1.0f / std::sqrt(ref_sum / size + eps);
    for (uint32_t i = 0; i < size; ++i) {
        expected[i] = kernel_test::round_to_bf16(input_q[i] * inv_rms * weight_q[i]);
    }

    toyinfer::bf16* dev_input = nullptr;
    toyinfer::bf16* dev_weight = nullptr;
    toyinfer::bf16* dev_output = nullptr;
    float* dev_sum = nullptr;
    TCUDA_CHECK(cudaMalloc(&dev_input, sizeof(toyinfer::bf16) * size));
    TCUDA_CHECK(cudaMalloc(&dev_weight, sizeof(toyinfer::bf16) * size));
    TCUDA_CHECK(cudaMalloc(&dev_output, sizeof(toyinfer::bf16) * size));
    TCUDA_CHECK(cudaMalloc(&dev_sum, sizeof(float)));
    TCUDA_CHECK(cudaMemcpy(dev_input, input_bf16.data(),
                           sizeof(toyinfer::bf16) * size,
                           cudaMemcpyHostToDevice));
    TCUDA_CHECK(cudaMemcpy(dev_weight, weight_bf16.data(),
                           sizeof(toyinfer::bf16) * size,
                           cudaMemcpyHostToDevice));

    toyinfer::rmsnorm_bf16<256>(dev_input, dev_weight, dev_output, dev_sum, eps,
                                size);
    TCUDA_CHECK(cudaDeviceSynchronize());

    std::vector<toyinfer::bf16> out_bf16(size);
    float gpu_sum = 0.0f;
    TCUDA_CHECK(cudaMemcpy(out_bf16.data(), dev_output,
                           sizeof(toyinfer::bf16) * size,
                           cudaMemcpyDeviceToHost));
    TCUDA_CHECK(cudaMemcpy(&gpu_sum, dev_sum, sizeof(float), cudaMemcpyDeviceToHost));
    const auto out = kernel_test::to_float(out_bf16);

    cudaFree(dev_input);
    cudaFree(dev_weight);
    cudaFree(dev_output);
    cudaFree(dev_sum);

    if (!kernel_test::nearly_equal(gpu_sum, ref_sum, 5e-2f, 5e-3f)) {
        std::cerr << "[FAIL] rmsnorm sum mismatch, expected " << ref_sum
                  << ", got " << gpu_sum << std::endl;
        return 1;
    }

    for (uint32_t i = 0; i < size; ++i) {
        if (!kernel_test::nearly_equal(out[i], expected[i], 2e-2f, 2e-2f)) {
            std::cerr << "[FAIL] rmsnorm output mismatch at " << i << ", expected "
                      << expected[i] << ", got " << out[i] << std::endl;
            return 1;
        }
    }

    std::cout << "[PASS] rmsnorm_bf16" << std::endl;
    return 0;
}
