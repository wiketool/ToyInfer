#include "test_helpers.h"
#include "../src/kernel.cu"

#include <string>

namespace {

struct SwiGluCase {
    std::string label;
    uint32_t size = 0;
    std::vector<float> gate;
    std::vector<float> up;
};

bool run_case(const SwiGluCase& test_case) {
    const auto gate_bf16 = kernel_test::to_bf16(test_case.gate);
    const auto up_bf16 = kernel_test::to_bf16(test_case.up);
    const auto gate_q = kernel_test::to_float(gate_bf16);
    const auto up_q = kernel_test::to_float(up_bf16);

    std::vector<float> expected(test_case.size, 0.0f);
    for (uint32_t i = 0; i < test_case.size; ++i) {
        const float silu = gate_q[i] * kernel_test::sigmoid(gate_q[i]);
        expected[i] = kernel_test::round_to_bf16(up_q[i] * silu);
    }

    kernel_test::DeviceBuffer<toyinfer::bf16> dev_gate(test_case.size);
    kernel_test::DeviceBuffer<toyinfer::bf16> dev_up(test_case.size);
    kernel_test::DeviceBuffer<toyinfer::bf16> dev_out(test_case.size);
    dev_gate.copy_from_host(gate_bf16);
    dev_up.copy_from_host(up_bf16);

    toyinfer::swiglu_bf16x2<256>(dev_gate.get(), dev_up.get(), dev_out.get(),
                                 test_case.size);
    TCUDA_CHECK(cudaDeviceSynchronize());

    std::vector<toyinfer::bf16> out_bf16;
    dev_out.copy_to_host(out_bf16);
    const auto out = kernel_test::to_float(out_bf16);
    return kernel_test::expect_vector_close(out, expected, 2e-2f, 2e-2f,
                                            test_case.label);
}

}  // namespace

int main() {
    if (!kernel_test::ensure_cuda_device(true)) {
        return 0;
    }

    std::vector<SwiGluCase> cases;
    cases.push_back({"swiglu tiny", 2, {-2.0f, 1.5f}, {0.75f, -0.5f}});

    SwiGluCase partial_case;
    partial_case.label = "swiglu partial-block";
    partial_case.size = 514;
    partial_case.gate.resize(partial_case.size);
    partial_case.up.resize(partial_case.size);
    for (uint32_t i = 0; i < partial_case.size; ++i) {
        partial_case.gate[i] =
            0.9f * std::sin((i + 1) * 0.08f) -
            0.7f * std::cos((i + 3) * 0.03f);
        partial_case.up[i] =
            0.35f * std::cos((i + 5) * 0.05f) -
            0.02f * static_cast<float>(i % 6);
    }
    cases.push_back(partial_case);

    SwiGluCase multi_block_case;
    multi_block_case.label = "swiglu multi-block";
    multi_block_case.size = 2048;
    multi_block_case.gate.resize(multi_block_case.size);
    multi_block_case.up.resize(multi_block_case.size);
    for (uint32_t i = 0; i < multi_block_case.size; ++i) {
        multi_block_case.gate[i] =
            1.1f * std::sin((i + 7) * 0.021f) -
            0.4f * std::cos((i + 1) * 0.013f) +
            0.01f * static_cast<float>((i * 3) % 5);
        multi_block_case.up[i] =
            0.42f * std::cos((i + 9) * 0.017f) +
            0.31f * std::sin((i + 1) * 0.011f) -
            0.015f * static_cast<float>(i % 4);
    }
    cases.push_back(multi_block_case);

    for (const auto& test_case : cases) {
        if (!run_case(test_case)) {
            return 1;
        }
    }

    std::cout << "[PASS] swiglu_bf16x2" << std::endl;
    return 0;
}
