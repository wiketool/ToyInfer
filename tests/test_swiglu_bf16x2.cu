#include "test_helpers.h"
#include "../src/kernel.cu"
#include "swiglu_test_common.h"

namespace {

bool run_case(const swiglu_test::Case& test_case) {
    const auto gate_bf16 = kernel_test::to_bf16(test_case.gate);
    const auto up_bf16 = kernel_test::to_bf16(test_case.up);
    const auto gate_q = kernel_test::to_float(gate_bf16);
    const auto up_q = kernel_test::to_float(up_bf16);

    std::vector<float> initial_out(test_case.buffer_size, 0.0f);
    for (uint32_t i = 0; i < test_case.buffer_size; ++i) {
        initial_out[i] = -0.7f + 0.06f * static_cast<float>((i * 5) % 9);
    }
    auto initial_out_bf16 = kernel_test::to_bf16(initial_out);
    const auto initial_out_q = kernel_test::to_float(initial_out_bf16);
    const auto expected = swiglu_test::reference_output(gate_q, up_q, initial_out_q,
                                                        test_case.size);

    kernel_test::DeviceBuffer<toyinfer::bf16> dev_gate(test_case.buffer_size);
    kernel_test::DeviceBuffer<toyinfer::bf16> dev_up(test_case.buffer_size);
    kernel_test::DeviceBuffer<toyinfer::bf16> dev_out(test_case.buffer_size);
    dev_gate.copy_from_host(gate_bf16);
    dev_up.copy_from_host(up_bf16);
    dev_out.copy_from_host(initial_out_bf16);

    toyinfer::swiglu_bf16x2<swiglu_test::kThreads>(dev_gate.get(), dev_up.get(),
                                                   dev_out.get(), test_case.size);
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

    std::vector<swiglu_test::Case> cases;
    swiglu_test::Case tiny;
    tiny.label = "swiglu wrapper tiny";
    tiny.size = 2;
    tiny.buffer_size = 6;
    tiny.gate = {-2.0f, 1.5f, 0.25f, -1.25f, 0.5f, -0.75f};
    tiny.up = {0.75f, -0.5f, 1.0f, -1.5f, 0.25f, -0.125f};
    cases.push_back(tiny);
    cases.push_back(swiglu_test::make_case("swiglu wrapper partial-block", 514,
                                           520, 0.08f, 0.05f));
    cases.push_back(swiglu_test::make_case("swiglu wrapper multi-block", 2048,
                                           2054, 0.021f, 0.017f));

    for (const auto& test_case : cases) {
        if (!run_case(test_case)) {
            return 1;
        }
    }

    std::cout << "[PASS] swiglu_bf16x2" << std::endl;
    return 0;
}
