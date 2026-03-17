#include <cmath>
#include <string>
#include <vector>

#include "../src/kernel.cu"
#include "gemv_test_common.h"
#include "test_helpers.h"

namespace {

struct BatchGemvCase {
    std::string label;
    uint32_t batch_size = 0;
    uint32_t M = 0;
    uint32_t N = 0;
    std::vector<float> w;
    std::vector<float> hidden;
};

BatchGemvCase make_case(const std::string& label, uint32_t batch_size,
                        uint32_t M, uint32_t N, float hidden_phase,
                        float w_phase) {
    BatchGemvCase test_case;
    test_case.label = label;
    test_case.batch_size = batch_size;
    test_case.M = M;
    test_case.N = N;
    test_case.w.resize(M * N);
    test_case.hidden.resize(batch_size * N);

    for (uint32_t r = 0; r < M; ++r) {
        for (uint32_t c = 0; c < N; ++c) {
            test_case.w[r * N + c] =
                0.33f * std::cos((r + 1) * (c + 1) * w_phase) +
                0.21f * std::sin((r + 2) * (c + 3) * (w_phase * 0.5f)) -
                0.012f * static_cast<float>((r * 5 + c * 7) % 11);
        }
    }

    for (uint32_t b = 0; b < batch_size; ++b) {
        for (uint32_t c = 0; c < N; ++c) {
            const auto idx = b * N + c;
            test_case.hidden[idx] =
                0.47f * std::sin((idx + 1) * hidden_phase) -
                0.18f * std::cos((idx + c + 3) * (hidden_phase * 0.9f)) +
                0.02f * static_cast<float>((b + c) % 6);
        }
    }

    return test_case;
}

std::vector<float> reference_output(const BatchGemvCase& test_case) {
    std::vector<float> expected(test_case.batch_size * test_case.M, 0.0f);
    for (uint32_t b = 0; b < test_case.batch_size; ++b) {
        for (uint32_t r = 0; r < test_case.M; ++r) {
            float sum = 0.0f;
            for (uint32_t c = 0; c < test_case.N; ++c) {
                sum += test_case.w[r * test_case.N + c] *
                       test_case.hidden[b * test_case.N + c];
            }
            expected[b * test_case.M + r] = kernel_test::round_to_bf16(sum);
        }
    }
    return expected;
}

bool run_case(const BatchGemvCase& test_case) {
    const auto w_bf16 = kernel_test::to_bf16(test_case.w);
    const auto hidden_bf16 = kernel_test::to_bf16(test_case.hidden);

    std::vector<float> initial_out(test_case.batch_size * test_case.M);
    for (uint32_t i = 0; i < initial_out.size(); ++i) {
        initial_out[i] = -0.92f + 0.05f * static_cast<float>((i * 3) % 13);
    }
    const auto initial_out_bf16 = kernel_test::to_bf16(initial_out);
    const auto expected = reference_output(test_case);

    kernel_test::DeviceBuffer<toyinfer::bf16> dev_w(test_case.M * test_case.N);
    kernel_test::DeviceBuffer<toyinfer::bf16> dev_hidden(test_case.batch_size *
                                                         test_case.N);
    kernel_test::DeviceBuffer<toyinfer::bf16> dev_out(test_case.batch_size *
                                                      test_case.M);

    dev_w.copy_from_host(w_bf16);
    dev_hidden.copy_from_host(hidden_bf16);
    dev_out.copy_from_host(initial_out_bf16);

    toyinfer::batch_gemv_proj_bf16<gemv_test::kThreads>(
        dev_w.get(), dev_hidden.get(), dev_out.get(), test_case.batch_size,
        test_case.M, test_case.N);
    TCUDA_CHECK(cudaDeviceSynchronize());

    std::vector<toyinfer::bf16> out_bf16;
    dev_out.copy_to_host(out_bf16);
    const auto out = kernel_test::to_float(out_bf16);
    return kernel_test::expect_vector_close(out, expected, 8e-2f, 8e-2f,
                                            test_case.label + " wrapper");
}

}  // namespace

int main() {
    if (!kernel_test::ensure_cuda_device(true)) {
        return 0;
    }

    const std::vector<BatchGemvCase> cases = {
        make_case("batch gemv proj tiny", 2, 3, 4, 0.19f, 0.12f),
        make_case("batch gemv proj multi-batch", 4, 12, 20, 0.08f, 0.06f),
    };

    for (const auto& test_case : cases) {
        if (!run_case(test_case)) {
            return 1;
        }
    }

    std::cout << "[PASS] batch_gemv_proj_bf16" << std::endl;
    return 0;
}
