#include "test_helpers.h"
#include "../src/kernel.cu"
#include "attention_test_common.h"

namespace {

template <uint32_t NUM_THREADS>
void launch_kernel(const float* input, toyinfer::bf16* output, uint32_t size) {
    dim3 block_dim{NUM_THREADS};
    dim3 grid_dim{(size + block_dim.x * 2 - 1) / (block_dim.x * 2)};
    toyinfer::convert_float22bfloat162<<<grid_dim, block_dim>>>(input, output,
                                                                size);
    TCUDA_CHECK(cudaGetLastError());
}

bool run_case(const attention_test::ConvertCase& test_case) {
    auto initial_out_bf16 =
        kernel_test::to_bf16(std::vector<float>(test_case.buffer_size, -0.25f));
    auto initial_out_q = kernel_test::to_float(initial_out_bf16);
    for (uint32_t i = 0; i < test_case.buffer_size; ++i) {
        initial_out_q[i] = kernel_test::round_to_bf16(
            -0.9f + 0.07f * static_cast<float>((i * 5) % 11));
        initial_out_bf16[i] = kernel_test::float_to_bf16(initial_out_q[i]);
    }
    const auto expected = attention_test::reference_convert(
        test_case.input, initial_out_q, test_case.size);

    kernel_test::DeviceBuffer<float> dev_input(test_case.input.size());
    kernel_test::DeviceBuffer<toyinfer::bf16> dev_out(test_case.buffer_size);
    dev_input.copy_from_host(test_case.input);
    dev_out.copy_from_host(initial_out_bf16);

    launch_kernel<attention_test::kThreads>(dev_input.get(), dev_out.get(),
                                            test_case.size);
    TCUDA_CHECK(cudaDeviceSynchronize());

    std::vector<toyinfer::bf16> out_bf16;
    dev_out.copy_to_host(out_bf16);
    const auto out = kernel_test::to_float(out_bf16);
    return kernel_test::expect_vector_close(out, expected, 2e-2f, 2e-2f,
                                            test_case.label + " convert kernel");
}

}  // namespace

int main() {
    if (!kernel_test::ensure_cuda_device(true)) {
        return 0;
    }

    std::vector<attention_test::ConvertCase> cases;
    cases.push_back({"convert kernel tiny", 2, 6,
                     {0.125f, -1.75f, 3.0f, -4.0f, 0.5f, -0.25f}});

    attention_test::ConvertCase large;
    large.label = "convert kernel partial-grid";
    large.size = 514;
    large.buffer_size = 520;
    large.input.resize(large.buffer_size);
    for (uint32_t i = 0; i < large.buffer_size; ++i) {
        large.input[i] =
            0.45f * std::sin((i + 1) * 0.05f) -
            0.33f * std::cos((i + 3) * 0.02f) +
            0.01f * static_cast<float>(i % 4);
    }
    cases.push_back(large);

    for (const auto& test_case : cases) {
        if (!run_case(test_case)) {
            return 1;
        }
    }

    std::cout << "[PASS] convert_float22bfloat162" << std::endl;
    return 0;
}
