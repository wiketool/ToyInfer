#include <cmath>
#include <string>
#include <vector>

#include "../src/kernel.cu"
#include "multi_rmsnorm_test_common.h"
#include "test_helpers.h"

namespace {

struct BatchMultiRmsnormCase {
    std::string label;
    uint32_t batch_size = 0;
    uint32_t num_heads = 0;
    uint32_t head_dim = 0;
    float eps = 1e-5f;
    std::vector<float> input;
    std::vector<float> weight;
};

BatchMultiRmsnormCase make_case(const std::string& label, uint32_t batch_size,
                                uint32_t num_heads, uint32_t head_dim,
                                float eps, float input_phase,
                                float weight_phase) {
    BatchMultiRmsnormCase test_case;
    test_case.label = label;
    test_case.batch_size = batch_size;
    test_case.num_heads = num_heads;
    test_case.head_dim = head_dim;
    test_case.eps = eps;
    const uint32_t buffer_size = batch_size * num_heads * head_dim;
    test_case.input.resize(buffer_size);
    test_case.weight.resize(head_dim);

    for (uint32_t b = 0; b < batch_size; ++b) {
        for (uint32_t h = 0; h < num_heads; ++h) {
            const uint32_t head_offset = (b * num_heads + h) * head_dim;
            for (uint32_t i = 0; i < head_dim; ++i) {
                const uint32_t idx = head_offset + i;
                test_case.input[idx] =
                    0.31f * std::sin((idx + 1) * input_phase) -
                    0.24f * std::cos((h + 1) * (i + 3) * input_phase * 0.7f) +
                    0.01f * static_cast<float>((b + h + i) % 5);
            }
        }
    }

    for (uint32_t i = 0; i < head_dim; ++i) {
        test_case.weight[i] = 0.68f + 0.14f * std::sin((i + 5) * weight_phase) -
                              0.05f * static_cast<float>(i % 3);
    }
    return test_case;
}

std::vector<float> reference_output(const BatchMultiRmsnormCase& test_case) {
    const uint32_t buffer_size =
        test_case.batch_size * test_case.num_heads * test_case.head_dim;
    std::vector<float> expected(buffer_size);
    for (uint32_t b = 0; b < test_case.batch_size; ++b) {
        for (uint32_t h = 0; h < test_case.num_heads; ++h) {
            const uint32_t head_offset =
                (b * test_case.num_heads + h) * test_case.head_dim;
            float sum = 0.0f;
            for (uint32_t i = 0; i < test_case.head_dim; ++i) {
                const float value = test_case.input[head_offset + i];
                sum += value * value;
            }
            const float multi_val =
                1.0f / std::sqrt(sum / static_cast<float>(test_case.head_dim) +
                                 test_case.eps);
            for (uint32_t i = 0; i < test_case.head_dim; ++i) {
                const float value = test_case.input[head_offset + i] *
                                    multi_val * test_case.weight[i];
                expected[head_offset + i] = kernel_test::round_to_bf16(value);
            }
        }
    }
    return expected;
}

bool run_case(const BatchMultiRmsnormCase& test_case) {
    const auto input_bf16 = kernel_test::to_bf16(test_case.input);
    const auto weight_bf16 = kernel_test::to_bf16(test_case.weight);
    const auto expected = reference_output(test_case);

    const uint32_t buffer_size =
        test_case.batch_size * test_case.num_heads * test_case.head_dim;
    kernel_test::DeviceBuffer<toyinfer::bf16> dev_input(buffer_size);
    kernel_test::DeviceBuffer<toyinfer::bf16> dev_weight(test_case.head_dim);
    kernel_test::DeviceBuffer<toyinfer::bf16> dev_output(buffer_size);

    dev_input.copy_from_host(input_bf16);
    dev_weight.copy_from_host(weight_bf16);

    toyinfer::batch_multi_rmsnorm_bf16(
        dev_input.get(), dev_weight.get(), dev_output.get(), test_case.eps,
        test_case.batch_size, test_case.num_heads, test_case.head_dim);
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

    const std::vector<BatchMultiRmsnormCase> cases = {
        make_case("batch multi rmsnorm single-head", 2, 1, 64, 1e-5f, 0.13f,
                  0.09f),
        make_case("batch multi rmsnorm multi-head", 2, 4, 64, 1e-4f, 0.09f,
                  0.05f),
    };

    for (const auto& test_case : cases) {
        if (!run_case(test_case)) {
            return 1;
        }
    }

    std::cout << "[PASS] batch_multi_rmsnorm_bf16" << std::endl;
    return 0;
}
