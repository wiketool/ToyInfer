#include <cmath>
#include <string>
#include <vector>

#include "../src/kernel.cu"
#include "add_test_common.h"
#include "test_helpers.h"

namespace {

struct BatchResidualAddCase {
    std::string label;
    uint32_t batch_size = 0;
    uint32_t size = 0;
    std::vector<float> residual;
    std::vector<float> hidden;
};

BatchResidualAddCase make_case(const std::string& label, uint32_t batch_size,
                               uint32_t size, float residual_phase,
                               float hidden_phase) {
    BatchResidualAddCase test_case;
    test_case.label = label;
    test_case.batch_size = batch_size;
    test_case.size = size;
    test_case.residual.resize(batch_size * size);
    test_case.hidden.resize(batch_size * size);

    for (uint32_t b = 0; b < batch_size; ++b) {
        for (uint32_t i = 0; i < size; ++i) {
            const auto idx = b * size + i;
            test_case.residual[idx] =
                0.23f * std::sin((idx + 1) * residual_phase) -
                0.08f * std::cos((i + 5) * residual_phase * 0.7f) +
                0.01f * static_cast<float>((b + i) % 5);
            test_case.hidden[idx] =
                -0.27f * std::cos((idx + 3) * hidden_phase) +
                0.19f * std::sin((b + i + 1) * hidden_phase * 0.6f) -
                0.012f * static_cast<float>((i * 7) % 9);
        }
    }
    return test_case;
}

std::vector<float> reference_output(const BatchResidualAddCase& test_case) {
    std::vector<float> expected(test_case.batch_size * test_case.size);
    for (uint32_t i = 0; i < expected.size(); ++i) {
        expected[i] = kernel_test::round_to_bf16(test_case.hidden[i] +
                                                 test_case.residual[i]);
    }
    return expected;
}

bool run_case(const BatchResidualAddCase& test_case) {
    const auto residual_bf16 = kernel_test::to_bf16(test_case.residual);
    auto hidden_bf16 = kernel_test::to_bf16(test_case.hidden);
    const auto expected = reference_output(test_case);

    const uint32_t buffer_size = test_case.batch_size * test_case.size;
    kernel_test::DeviceBuffer<toyinfer::bf16> dev_residual(buffer_size);
    kernel_test::DeviceBuffer<toyinfer::bf16> dev_hidden(buffer_size);
    dev_residual.copy_from_host(residual_bf16);
    dev_hidden.copy_from_host(hidden_bf16);

    toyinfer::batch_residual_add_bf16<add_test::kThreads>(
        dev_residual.get(), dev_hidden.get(), test_case.batch_size,
        test_case.size);
    TCUDA_CHECK(cudaDeviceSynchronize());

    std::vector<toyinfer::bf16> out_bf16;
    dev_hidden.copy_to_host(out_bf16);
    const auto out = kernel_test::to_float(out_bf16);
    return kernel_test::expect_vector_close(out, expected, 2e-2f, 2e-2f,
                                            test_case.label + " wrapper");
}

}  // namespace

int main() {
    if (!kernel_test::ensure_cuda_device(true)) {
        return 0;
    }

    const std::vector<BatchResidualAddCase> cases = {
        make_case("batch residual add tiny", 1, 6, 0.09f, 0.05f),
        make_case("batch residual add medium", 3, 32, 0.07f, 0.03f),
    };

    for (const auto& test_case : cases) {
        if (!run_case(test_case)) {
            return 1;
        }
    }

    std::cout << "[PASS] batch_residual_add_bf16" << std::endl;
    return 0;
}
