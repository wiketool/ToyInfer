#include "test_helpers.h"
#include "../src/kernel.cu"

#include <string>

namespace {

struct RopeCase {
    std::string label;
    uint32_t num_heads = 0;
    uint32_t head_dim = 0;
    uint32_t pos = 0;
    std::vector<float> qk;
    std::vector<float> inv_freq;
};

bool run_case(const RopeCase& test_case) {
    auto qk_bf16 = kernel_test::to_bf16(test_case.qk);
    const auto qk_q = kernel_test::to_float(qk_bf16);
    std::vector<float> expected(qk_q.size(), 0.0f);

    for (uint32_t h = 0; h < test_case.num_heads; ++h) {
        const uint32_t offset = h * test_case.head_dim;
        for (uint32_t i = 0; i < test_case.head_dim / 2; ++i) {
            const float real = qk_q[offset + i];
            const float imag = qk_q[offset + i + test_case.head_dim / 2];
            const float cos_v = std::cos(test_case.pos * test_case.inv_freq[i]);
            const float sin_v = std::sin(test_case.pos * test_case.inv_freq[i]);
            expected[offset + i] =
                kernel_test::round_to_bf16(cos_v * real - sin_v * imag);
            expected[offset + i + test_case.head_dim / 2] =
                kernel_test::round_to_bf16(cos_v * imag + sin_v * real);
        }
    }

    kernel_test::DeviceBuffer<toyinfer::bf16> dev_qk(qk_bf16.size());
    kernel_test::DeviceBuffer<float> dev_inv_freq(test_case.inv_freq.size());
    dev_qk.copy_from_host(qk_bf16);
    dev_inv_freq.copy_from_host(test_case.inv_freq);

    toyinfer::rope_bf16(dev_qk.get(), dev_inv_freq.get(), test_case.pos,
                        test_case.num_heads, test_case.head_dim);
    TCUDA_CHECK(cudaDeviceSynchronize());

    dev_qk.copy_to_host(qk_bf16);
    const auto out = kernel_test::to_float(qk_bf16);
    return kernel_test::expect_vector_close(out, expected, 2e-2f, 2e-2f,
                                            test_case.label);
}

}  // namespace

int main() {
    if (!kernel_test::ensure_cuda_device(true)) {
        return 0;
    }

    std::vector<RopeCase> cases;

    cases.push_back({"rope pos=0 identity", 1, 2, 0, {0.25f, -0.5f}, {0.75f}});

    RopeCase medium_case;
    medium_case.label = "rope medium";
    medium_case.num_heads = 3;
    medium_case.head_dim = 16;
    medium_case.pos = 11;
    medium_case.qk.resize(medium_case.num_heads * medium_case.head_dim);
    medium_case.inv_freq.resize(medium_case.head_dim / 2);
    for (uint32_t i = 0; i < medium_case.qk.size(); ++i) {
        medium_case.qk[i] =
            0.45f * std::sin(i * 0.03f) + 0.07f * std::cos((i + 1) * 0.09f);
    }
    for (uint32_t i = 0; i < medium_case.inv_freq.size(); ++i) {
        medium_case.inv_freq[i] = 0.001f * static_cast<float>(i + 1);
    }
    cases.push_back(medium_case);

    RopeCase large_case;
    large_case.label = "rope large multi-head";
    large_case.num_heads = 4;
    large_case.head_dim = 128;
    large_case.pos = 37;
    large_case.qk.resize(large_case.num_heads * large_case.head_dim);
    large_case.inv_freq.resize(large_case.head_dim / 2);
    for (uint32_t i = 0; i < large_case.qk.size(); ++i) {
        large_case.qk[i] =
            0.3f * std::sin((i + 5) * 0.021f) -
            0.22f * std::cos((i + 1) * 0.047f) +
            0.01f * static_cast<float>(i % 3);
    }
    for (uint32_t i = 0; i < large_case.inv_freq.size(); ++i) {
        large_case.inv_freq[i] =
            1.0f / std::pow(10000.0f, (2.0f * static_cast<float>(i)) /
                                          static_cast<float>(large_case.head_dim));
    }
    cases.push_back(large_case);

    for (const auto& test_case : cases) {
        if (!run_case(test_case)) {
            return 1;
        }
    }

    std::cout << "[PASS] rope_bf16" << std::endl;
    return 0;
}
