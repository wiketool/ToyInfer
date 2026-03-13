#include "test_helpers.h"
#include "../src/kernel.cu"
#include "rope_test_common.h"

namespace {

bool run_case(const rope_test::Case& test_case) {
    auto qk_bf16 = kernel_test::to_bf16(test_case.qk);
    const auto qk_q = kernel_test::to_float(qk_bf16);
    const auto expected = rope_test::reference_output(test_case, qk_q);

    kernel_test::DeviceBuffer<toyinfer::bf16> dev_qk(test_case.buffer_size);
    kernel_test::DeviceBuffer<float> dev_inv_freq(test_case.inv_freq.size());
    dev_qk.copy_from_host(qk_bf16);
    dev_inv_freq.copy_from_host(test_case.inv_freq);

    toyinfer::rope_bf16(dev_qk.get(), dev_inv_freq.get(), test_case.pos,
                        test_case.num_heads, test_case.head_dim);
    TCUDA_CHECK(cudaDeviceSynchronize());

    dev_qk.copy_to_host(qk_bf16);
    const auto out = kernel_test::to_float(qk_bf16);
    return kernel_test::expect_vector_close(out, expected, 2e-2f, 2e-2f,
                                            test_case.label + " wrapper");
}

}  // namespace

int main() {
    if (!kernel_test::ensure_cuda_device(true)) {
        return 0;
    }

    std::vector<rope_test::Case> cases;
    rope_test::Case tiny;
    tiny.label = "rope wrapper pos0";
    tiny.num_heads = 1;
    tiny.head_dim = 2;
    tiny.pos = 0;
    tiny.buffer_size = 6;
    tiny.qk = {0.25f, -0.5f, 1.25f, -1.0f, 0.75f, -0.25f};
    tiny.inv_freq = {0.75f};
    cases.push_back(tiny);
    cases.push_back(rope_test::make_case("rope wrapper medium", 3, 16, 11,
                                         3 * 16 + 4, 0.03f, 0.001f));
    auto large = rope_test::make_case("rope wrapper large", 4, 128, 37,
                                      4 * 128 + 8, 0.021f, 0.0f);
    for (uint32_t i = 0; i < large.inv_freq.size(); ++i) {
        large.inv_freq[i] =
            1.0f / std::pow(10000.0f,
                            (2.0f * static_cast<float>(i)) /
                                static_cast<float>(large.head_dim));
    }
    cases.push_back(large);

    for (const auto& test_case : cases) {
        if (!run_case(test_case)) {
            return 1;
        }
    }

    std::cout << "[PASS] rope_bf16" << std::endl;
    return 0;
}
