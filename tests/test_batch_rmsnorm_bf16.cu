#include <cmath>
#include <string>
#include <vector>

#include "../src/kernel.cu"
#include "rmsnorm_test_common.h"
#include "test_helpers.h"

namespace {

struct BatchRmsnormCase {
    std::string label;
    uint32_t batch_size = 0;
    uint32_t size = 0;
    float eps = 1e-5f;
    std::vector<float> input;
    std::vector<float> weight;
};

BatchRmsnormCase make_case(const std::string& label, uint32_t batch_size,
                           uint32_t size, float eps, float input_phase,
                           float weight_phase) {
    BatchRmsnormCase test_case;
    test_case.label = label;
    test_case.batch_size = batch_size;
    test_case.size = size;
    test_case.eps = eps;
    test_case.input.resize(batch_size * size);
    test_case.weight.resize(size);

    for (uint32_t b = 0; b < batch_size; ++b) {
        for (uint32_t i = 0; i < size; ++i) {
            const auto idx = b * size + i;
            test_case.input[idx] =
                0.43f * std::sin((idx + 1) * input_phase) -
                0.27f * std::cos((i + 2) * input_phase * 0.4f) +
                0.02f * static_cast<float>((b + i) % 7);
        }
    }

    for (uint32_t i = 0; i < size; ++i) {
        test_case.weight[i] = 0.72f + 0.18f * std::sin((i + 5) * weight_phase) -
                              0.04f * static_cast<float>((i * 3) % 5);
    }
    return test_case;
}

std::vector<float> reference_output(const BatchRmsnormCase& test_case) {
    std::vector<float> expected(test_case.batch_size * test_case.size);
    for (uint32_t b = 0; b < test_case.batch_size; ++b) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < test_case.size; ++i) {
            const float value = test_case.input[b * test_case.size + i];
            sum += value * value;
        }
        const float inv_rms =
            1.0f /
            std::sqrt(sum / static_cast<float>(test_case.size) + test_case.eps);
        for (uint32_t i = 0; i < test_case.size; ++i) {
            const float value = test_case.input[b * test_case.size + i] *
                                inv_rms * test_case.weight[i];
            expected[b * test_case.size + i] =
                kernel_test::round_to_bf16(value);
        }
    }
    return expected;
}

bool run_case(const BatchRmsnormCase& test_case) {
    const auto input_bf16 = kernel_test::to_bf16(test_case.input);
    const auto weight_bf16 = kernel_test::to_bf16(test_case.weight);
    const auto expected = reference_output(test_case);

    kernel_test::DeviceBuffer<toyinfer::bf16> dev_input(test_case.batch_size *
                                                        test_case.size);
    kernel_test::DeviceBuffer<toyinfer::bf16> dev_weight(test_case.size);
    kernel_test::DeviceBuffer<toyinfer::bf16> dev_output(test_case.batch_size *
                                                         test_case.size);
    kernel_test::DeviceBuffer<float> dev_sums(test_case.batch_size);

    dev_input.copy_from_host(input_bf16);
    dev_weight.copy_from_host(weight_bf16);

    toyinfer::batch_rmsnorm_bf16<rmsnorm_test::kThreads>(
        dev_input.get(), dev_weight.get(), dev_output.get(), dev_sums.get(),
        test_case.eps, test_case.batch_size, test_case.size);
    TCUDA_CHECK(cudaDeviceSynchronize());

    std::vector<toyinfer::bf16> out_bf16;
    dev_output.copy_to_host(out_bf16);
    const auto out = kernel_test::to_float(out_bf16);
    return kernel_test::expect_vector_close(out, expected, 2e-2f, 2e-2f,
                                            test_case.label + " wrapper");
}

}  // namespace

int main() {
    if (!kernel_test::ensure_cuda_device(true)) {
        return 0;
    }

    const std::vector<BatchRmsnormCase> cases = {
        make_case("batch rmsnorm tiny", 1, 16, 1e-5f, 0.15f, 0.22f),
        make_case("batch rmsnorm medium", 3, 32, 1e-4f, 0.08f, 0.12f),
    };

    for (const auto& test_case : cases) {
        if (!run_case(test_case)) {
            return 1;
        }
    }

    std::cout << "[PASS] batch_rmsnorm_bf16" << std::endl;
    return 0;
}
