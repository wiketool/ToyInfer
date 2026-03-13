#include "test_helpers.h"
#include "../src/kernel.cu"

#include <string>

namespace {

struct AddCase {
    std::string label;
    uint32_t size = 0;
    bool run_twice = false;
    std::vector<float> residual;
    std::vector<float> hidden;
};

std::vector<float> apply_add_reference(const std::vector<float>& hidden,
                                       const std::vector<float>& residual) {
    std::vector<float> out(hidden.size(), 0.0f);
    for (size_t i = 0; i < hidden.size(); ++i) {
        out[i] = kernel_test::round_to_bf16(hidden[i] + residual[i]);
    }
    return out;
}

bool run_case(const AddCase& test_case) {
    const auto residual_bf16 = kernel_test::to_bf16(test_case.residual);
    auto hidden_bf16 = kernel_test::to_bf16(test_case.hidden);
    const auto residual_q = kernel_test::to_float(residual_bf16);
    const auto hidden_q = kernel_test::to_float(hidden_bf16);

    const auto expected_once = apply_add_reference(hidden_q, residual_q);

    kernel_test::DeviceBuffer<toyinfer::bf16> dev_residual(test_case.size);
    kernel_test::DeviceBuffer<toyinfer::bf16> dev_hidden(test_case.size);
    dev_residual.copy_from_host(residual_bf16);
    dev_hidden.copy_from_host(hidden_bf16);

    toyinfer::residual_add_bf16<256>(dev_residual.get(), dev_hidden.get(),
                                     test_case.size);
    TCUDA_CHECK(cudaDeviceSynchronize());

    dev_hidden.copy_to_host(hidden_bf16);
    auto out = kernel_test::to_float(hidden_bf16);
    if (!kernel_test::expect_vector_close(out, expected_once, 2e-2f, 2e-2f,
                                          test_case.label + " first pass")) {
        return false;
    }

    if (!test_case.run_twice) {
        return true;
    }

    const auto expected_twice = apply_add_reference(expected_once, residual_q);
    toyinfer::residual_add_bf16<256>(dev_residual.get(), dev_hidden.get(),
                                     test_case.size);
    TCUDA_CHECK(cudaDeviceSynchronize());

    dev_hidden.copy_to_host(hidden_bf16);
    out = kernel_test::to_float(hidden_bf16);
    return kernel_test::expect_vector_close(out, expected_twice, 2e-2f, 2e-2f,
                                            test_case.label + " second pass");
}

}  // namespace

int main() {
    if (!kernel_test::ensure_cuda_device(true)) {
        return 0;
    }

    std::vector<AddCase> cases;
    cases.push_back({"residual_add tiny", 2, true, {0.5f, -0.25f},
                     {-1.0f, 0.75f}});

    AddCase partial_case;
    partial_case.label = "residual_add partial-block";
    partial_case.size = 514;
    partial_case.run_twice = false;
    partial_case.residual.resize(partial_case.size);
    partial_case.hidden.resize(partial_case.size);
    for (uint32_t i = 0; i < partial_case.size; ++i) {
        partial_case.residual[i] =
            0.22f * std::sin((i + 1) * 0.09f) - 0.03f * static_cast<float>(i % 4);
        partial_case.hidden[i] =
            -0.31f * std::cos((i + 3) * 0.07f) +
            0.02f * static_cast<float>((i * 5) % 7);
    }
    cases.push_back(partial_case);

    AddCase multi_block_case;
    multi_block_case.label = "residual_add multi-block";
    multi_block_case.size = 1536;
    multi_block_case.run_twice = true;
    multi_block_case.residual.resize(multi_block_case.size);
    multi_block_case.hidden.resize(multi_block_case.size);
    for (uint32_t i = 0; i < multi_block_case.size; ++i) {
        multi_block_case.residual[i] =
            0.18f * std::sin((i + 5) * 0.05f) +
            0.09f * std::cos((i + 1) * 0.03f) -
            0.01f * static_cast<float>(i % 5);
        multi_block_case.hidden[i] =
            0.27f * std::cos((i + 7) * 0.04f) -
            0.14f * std::sin((i + 1) * 0.02f) +
            0.015f * static_cast<float>((i * 11) % 9);
    }
    cases.push_back(multi_block_case);

    for (const auto& test_case : cases) {
        if (!run_case(test_case)) {
            return 1;
        }
    }

    std::cout << "[PASS] residual_add_bf16" << std::endl;
    return 0;
}
