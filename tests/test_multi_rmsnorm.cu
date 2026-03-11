#include "test_helpers.h"
#include "../src/kernel.cu"

int main() {
    if (!kernel_test::ensure_cuda_device(true)) {
        return 0;
    }

    constexpr uint32_t nums_head = 4;
    constexpr uint32_t head_dim = 128;
    constexpr uint32_t size = nums_head * head_dim;
    constexpr float eps = 1e-5f;
    std::vector<float> input_f(size), weight_f(size);
    for (uint32_t h = 0; h < nums_head; ++h) {
        for (uint32_t i = 0; i < head_dim; ++i) {
            const uint32_t idx = h * head_dim + i;
            input_f[idx] =
                std::sin((idx + 1) * 0.09f) + 0.05f * static_cast<float>(h + 1);
            weight_f[idx] = 0.7f + 0.01f * static_cast<float>((idx * 5) % 11);
        }
    }

    const auto input_bf16 = kernel_test::to_bf16(input_f);
    const auto weight_bf16 = kernel_test::to_bf16(weight_f);
    const auto input_q = kernel_test::to_float(input_bf16);
    const auto weight_q = kernel_test::to_float(weight_bf16);
    std::vector<float> expected(size, 0.0f);

    for (uint32_t h = 0; h < nums_head; ++h) {
        const uint32_t offset = h * head_dim;
        float head_sum = 0.0f;
        for (uint32_t i = 0; i < head_dim; ++i) {
            head_sum += input_q[offset + i] * input_q[offset + i];
        }
        const float inv_rms = 1.0f / std::sqrt(head_sum / head_dim + eps);
        for (uint32_t i = 0; i < head_dim; ++i) {
            expected[offset + i] = kernel_test::round_to_bf16(
                input_q[offset + i] * inv_rms * weight_q[offset + i]);
        }
    }

    toyinfer::bf16* dev_input = nullptr;
    toyinfer::bf16* dev_weight = nullptr;
    toyinfer::bf16* dev_output = nullptr;
    TCUDA_CHECK(cudaMalloc(&dev_input, sizeof(toyinfer::bf16) * size));
    TCUDA_CHECK(cudaMalloc(&dev_weight, sizeof(toyinfer::bf16) * size));
    TCUDA_CHECK(cudaMalloc(&dev_output, sizeof(toyinfer::bf16) * size));
    TCUDA_CHECK(cudaMemcpy(dev_input, input_bf16.data(),
                           sizeof(toyinfer::bf16) * size,
                           cudaMemcpyHostToDevice));
    TCUDA_CHECK(cudaMemcpy(dev_weight, weight_bf16.data(),
                           sizeof(toyinfer::bf16) * size,
                           cudaMemcpyHostToDevice));

    toyinfer::multi_rmsnorm_bf16(dev_input, dev_weight, dev_output, eps, nums_head,
                                 head_dim);
    TCUDA_CHECK(cudaDeviceSynchronize());

    std::vector<toyinfer::bf16> out_bf16(size);
    TCUDA_CHECK(cudaMemcpy(out_bf16.data(), dev_output,
                           sizeof(toyinfer::bf16) * size,
                           cudaMemcpyDeviceToHost));
    const auto out = kernel_test::to_float(out_bf16);

    cudaFree(dev_input);
    cudaFree(dev_weight);
    cudaFree(dev_output);

    for (uint32_t i = 0; i < size; ++i) {
        if (!kernel_test::nearly_equal(out[i], expected[i], 2e-2f, 2e-2f)) {
            std::cerr << "[FAIL] multi_rmsnorm mismatch at " << i << ", expected "
                      << expected[i] << ", got " << out[i] << std::endl;
            return 1;
        }
    }

    std::cout << "[PASS] multi_rmsnorm_bf16" << std::endl;
    return 0;
}
