#include <cmath>
#include <string>
#include <vector>

#include "../src/kernel.cu"
#include "swiglu_test_common.h"
#include "test_helpers.h"

namespace {

struct BatchSwigluCase {
    std::string label;
    uint32_t batch_size = 0;
    uint32_t size = 0;
    std::vector<float> gate;
    std::vector<float> up;
};

BatchSwigluCase make_case(const std::string& label, uint32_t batch_size,
                          uint32_t size, float gate_phase, float up_phase) {
    BatchSwigluCase test_case;
    test_case.label = label;
    test_case.batch_size = batch_size;
    test_case.size = size;
    test_case.gate.resize(batch_size * size);
    test_case.up.resize(batch_size * size);

    for (uint32_t b = 0; b < batch_size; ++b) {
        for (uint32_t i = 0; i < size; ++i) {
            const auto idx = b * size + i;
            test_case.gate[idx] =
                0.9f * std::sin((idx + 1) * gate_phase) -
                0.7f * std::cos((idx + 3) * gate_phase * 0.5f);
            test_case.up[idx] = 0.42f * std::cos((i + 5) * up_phase) -
                                0.31f * std::sin((idx + 1) * up_phase * 0.8f) -
                                0.02f * static_cast<float>(idx % 6);
        }
    }
    return test_case;
}

std::vector<float> reference_output(const BatchSwigluCase& test_case) {
    std::vector<float> expected(test_case.batch_size * test_case.size);
    for (uint32_t i = 0; i < expected.size(); ++i) {
        const float gate_val = test_case.gate[i];
        const float silu = gate_val * kernel_test::sigmoid(gate_val);
        expected[i] = kernel_test::round_to_bf16(test_case.up[i] * silu);
    }
    return expected;
}

bool run_case(const BatchSwigluCase& test_case) {
    const auto gate_bf16 = kernel_test::to_bf16(test_case.gate);
    const auto up_bf16 = kernel_test::to_bf16(test_case.up);
    const auto expected = reference_output(test_case);

    const uint32_t buffer_size = test_case.batch_size * test_case.size;
    kernel_test::DeviceBuffer<toyinfer::bf16> dev_gate(buffer_size);
    kernel_test::DeviceBuffer<toyinfer::bf16> dev_up(buffer_size);
    kernel_test::DeviceBuffer<toyinfer::bf16> dev_out(buffer_size);

    dev_gate.copy_from_host(gate_bf16);
    dev_up.copy_from_host(up_bf16);

    toyinfer::batch_swiglu_bf16x2<swiglu_test::kThreads>(
        dev_gate.get(), dev_up.get(), dev_out.get(), test_case.batch_size,
        test_case.size);
    TCUDA_CHECK(cudaDeviceSynchronize());

    std::vector<toyinfer::bf16> out_bf16;
    dev_out.copy_to_host(out_bf16);
    const auto out = kernel_test::to_float(out_bf16);
    return kernel_test::expect_vector_close(out, expected, 2e-2f, 2e-2f,
                                            test_case.label + " wrapper");
}

}  // namespace

int main() {
    if (!kernel_test::ensure_cuda_device(true)) {
        return 0;
    }

    const std::vector<BatchSwigluCase> cases = {
        make_case("batch swiglu tiny", 2, 8, 0.18f, 0.15f),
        make_case("batch swiglu medium", 3, 40, 0.1f, 0.07f),
    };

    for (const auto& test_case : cases) {
        if (!run_case(test_case)) {
            return 1;
        }
    }

    std::cout << "[PASS] batch_swiglu_bf16x2" << std::endl;
    return 0;
}
