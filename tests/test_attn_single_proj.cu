#include "test_helpers.h"
#include "../src/kernel.cu"

#include <string>

namespace {

struct GemvCase {
    std::string label;
    uint32_t M = 0;
    uint32_t N = 0;
    std::vector<float> x;
    std::vector<float> w;
};

GemvCase make_case(const std::string& label, uint32_t M, uint32_t N,
                   float x_phase, float w_phase) {
    GemvCase test_case;
    test_case.label = label;
    test_case.M = M;
    test_case.N = N;
    test_case.x.resize(N);
    test_case.w.resize(M * N);
    for (uint32_t i = 0; i < N; ++i) {
        test_case.x[i] =
            0.45f * std::sin((i + 1) * x_phase) -
            0.18f * std::cos((i + 3) * (x_phase * 0.7f)) +
            0.01f * static_cast<float>(i % 5);
    }
    for (uint32_t r = 0; r < M; ++r) {
        for (uint32_t c = 0; c < N; ++c) {
            test_case.w[r * N + c] =
                0.35f * std::cos((r + 1) * (c + 1) * w_phase) +
                0.22f * std::sin((r + c + 3) * (w_phase * 0.6f)) -
                0.015f * static_cast<float>((r * 7 + c * 3) % 9);
        }
    }
    return test_case;
}

bool run_bf16_case(const GemvCase& test_case) {
    const auto x_bf16 = kernel_test::to_bf16(test_case.x);
    const auto w_bf16 = kernel_test::to_bf16(test_case.w);
    const auto x_q = kernel_test::to_float(x_bf16);
    const auto w_q = kernel_test::to_float(w_bf16);

    std::vector<float> expected(test_case.M, 0.0f);
    for (uint32_t r = 0; r < test_case.M; ++r) {
        float sum = 0.0f;
        for (uint32_t c = 0; c < test_case.N; ++c) {
            sum += w_q[r * test_case.N + c] * x_q[c];
        }
        expected[r] = kernel_test::round_to_bf16(sum);
    }

    kernel_test::DeviceBuffer<toyinfer::bf16> dev_w(test_case.M * test_case.N);
    kernel_test::DeviceBuffer<toyinfer::bf16> dev_x(test_case.N);
    kernel_test::DeviceBuffer<toyinfer::bf16> dev_y(test_case.M);
    dev_w.copy_from_host(w_bf16);
    dev_x.copy_from_host(x_bf16);

    toyinfer::gemv_proj_bf16<256>(dev_w.get(), dev_x.get(), dev_y.get(),
                                  test_case.M, test_case.N);
    TCUDA_CHECK(cudaDeviceSynchronize());

    std::vector<toyinfer::bf16> out_bf16;
    dev_y.copy_to_host(out_bf16);
    const auto out = kernel_test::to_float(out_bf16);
    return kernel_test::expect_vector_close(out, expected, 8e-2f, 8e-2f,
                                            test_case.label);
}

bool run_float_case(const GemvCase& test_case) {
    const auto x_bf16 = kernel_test::to_bf16(test_case.x);
    const auto w_bf16 = kernel_test::to_bf16(test_case.w);
    const auto x_q = kernel_test::to_float(x_bf16);
    const auto w_q = kernel_test::to_float(w_bf16);

    std::vector<float> expected(test_case.M, 0.0f);
    for (uint32_t r = 0; r < test_case.M; ++r) {
        float sum = 0.0f;
        for (uint32_t c = 0; c < test_case.N; ++c) {
            sum += w_q[r * test_case.N + c] * x_q[c];
        }
        expected[r] = sum;
    }

    kernel_test::DeviceBuffer<toyinfer::bf16> dev_w(test_case.M * test_case.N);
    kernel_test::DeviceBuffer<toyinfer::bf16> dev_x(test_case.N);
    kernel_test::DeviceBuffer<float> dev_y(test_case.M);
    dev_w.copy_from_host(w_bf16);
    dev_x.copy_from_host(x_bf16);

    toyinfer::gemv_proj_bf162float<256>(dev_w.get(), dev_x.get(), dev_y.get(),
                                        test_case.M, test_case.N);
    TCUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> out;
    dev_y.copy_to_host(out);
    return kernel_test::expect_vector_close(out, expected, 5e-2f, 5e-2f,
                                            test_case.label);
}

}  // namespace

int main() {
    if (!kernel_test::ensure_cuda_device(true)) {
        return 0;
    }

    const std::vector<GemvCase> bf16_cases = {
        make_case("gemv_proj_bf16 tiny", 1, 2, 0.17f, 0.01f),
        make_case("gemv_proj_bf16 single-iteration", 9, 256, 0.07f, 0.006f),
        make_case("gemv_proj_bf16 multi-row-multi-loop", 513, 514, 0.05f,
                  0.004f),
    };
    for (const auto& test_case : bf16_cases) {
        if (!run_bf16_case(test_case)) {
            return 1;
        }
    }

    const std::vector<GemvCase> float_cases = {
        make_case("gemv_proj_bf162float tiny", 3, 2, 0.19f, 0.02f),
        make_case("gemv_proj_bf162float multi-loop", 17, 514, 0.09f, 0.005f),
        make_case("gemv_proj_bf162float lmhead-shape-like", 257, 1024, 0.04f,
                  0.003f),
    };
    for (const auto& test_case : float_cases) {
        if (!run_float_case(test_case)) {
            return 1;
        }
    }

    std::cout << "[PASS] gemv_proj_bf16 + gemv_proj_bf162float" << std::endl;
    return 0;
}
