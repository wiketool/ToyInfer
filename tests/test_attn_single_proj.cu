#include "test_helpers.h"
#include "../src/kernel.cu"

int main() {
    if (!kernel_test::ensure_cuda_device(true)) {
        return 0;
    }

    constexpr uint32_t kSmallM = 9;
    constexpr uint32_t kLargeM = 70000;
    constexpr uint32_t N = 256;
    std::vector<float> host_x(N);
    for (uint32_t i = 0; i < N; ++i) {
        host_x[i] = std::sin(i * 0.17f) + 0.01f * static_cast<float>(i % 5);
    }
    const auto x_bf16 = kernel_test::to_bf16(host_x);
    const auto x_q = kernel_test::to_float(x_bf16);

    for (uint32_t M : {kSmallM, kLargeM}) {
        std::vector<float> host_w(M * N);
        for (uint32_t r = 0; r < M; ++r) {
            for (uint32_t c = 0; c < N; ++c) {
                host_w[r * N + c] =
                    std::cos((r + 1) * (c + 1) * 0.01f) +
                    0.005f * static_cast<float>((r * 7 + c * 3) % 9);
            }
        }

        const auto w_bf16 = kernel_test::to_bf16(host_w);
        const auto w_q = kernel_test::to_float(w_bf16);
        std::vector<float> expected(M, 0.0f);
        for (uint32_t r = 0; r < M; ++r) {
            float sum = 0.0f;
            for (uint32_t c = 0; c < N; ++c) {
                sum += w_q[r * N + c] * x_q[c];
            }
            expected[r] = kernel_test::round_to_bf16(sum);
        }

        toyinfer::bf16* dev_w = nullptr;
        toyinfer::bf16* dev_x = nullptr;
        toyinfer::bf16* dev_y = nullptr;
        TCUDA_CHECK(cudaMalloc(&dev_w, sizeof(toyinfer::bf16) * M * N));
        TCUDA_CHECK(cudaMalloc(&dev_x, sizeof(toyinfer::bf16) * N));
        TCUDA_CHECK(cudaMalloc(&dev_y, sizeof(toyinfer::bf16) * M));
        TCUDA_CHECK(cudaMemcpy(dev_w, w_bf16.data(),
                               sizeof(toyinfer::bf16) * M * N,
                               cudaMemcpyHostToDevice));
        TCUDA_CHECK(cudaMemcpy(dev_x, x_bf16.data(), sizeof(toyinfer::bf16) * N,
                               cudaMemcpyHostToDevice));

        toyinfer::gemv_proj_bf16<256>(dev_w, dev_x, dev_y, M, N);
        TCUDA_CHECK(cudaDeviceSynchronize());

        std::vector<toyinfer::bf16> out_bf16(M);
        TCUDA_CHECK(cudaMemcpy(out_bf16.data(), dev_y,
                               sizeof(toyinfer::bf16) * M,
                               cudaMemcpyDeviceToHost));
        const auto out = kernel_test::to_float(out_bf16);

        cudaFree(dev_w);
        cudaFree(dev_x);
        cudaFree(dev_y);

        for (uint32_t i = 0; i < M; ++i) {
            if (!kernel_test::nearly_equal(out[i], expected[i], 8e-2f, 8e-2f)) {
                std::cerr << "[FAIL] gemv_proj_bf16 mismatch at row " << i
                          << " for M=" << M << ", expected " << expected[i]
                          << ", got " << out[i] << std::endl;
                return 1;
            }
        }
    }

    std::cout << "[PASS] gemv_proj_bf16" << std::endl;
    return 0;
}
