#include "test_helpers.h"
#include "../src/kernel.cu"
#include "attention_test_common.h"

namespace {

template <uint32_t NUM_THREADS>
void launch_kernel(float* score, uint32_t num_q_heads, uint32_t pos,
                   uint32_t max_seq_len) {
    dim3 block_dim{NUM_THREADS};
    dim3 grid_dim{num_q_heads};
    toyinfer::softmax_f32_kernel<NUM_THREADS>
        <<<grid_dim, block_dim>>>(score, num_q_heads, pos, max_seq_len);
    TCUDA_CHECK(cudaGetLastError());
}

bool run_case(const attention_test::SoftmaxCase& test_case) {
    const auto expected =
        attention_test::reference_softmax(test_case, test_case.score);

    kernel_test::DeviceBuffer<float> dev_score(test_case.score.size());
    dev_score.copy_from_host(test_case.score);

    launch_kernel<attention_test::kThreads>(dev_score.get(), test_case.num_q_heads,
                                            test_case.pos, test_case.max_seq_len);
    TCUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> out;
    dev_score.copy_to_host(out);
    return kernel_test::expect_vector_close(out, expected, 5e-3f, 5e-3f,
                                            test_case.label + " softmax kernel");
}

}  // namespace

int main() {
    if (!kernel_test::ensure_cuda_device(true)) {
        return 0;
    }

    std::vector<attention_test::SoftmaxCase> cases;
    cases.push_back({"softmax kernel pos0", 2, 0, 5,
                     {1.25f, -9.0f, 3.0f, 4.0f, -2.0f, -0.5f, 7.0f, -1.0f, 2.0f,
                      3.0f}});

    attention_test::SoftmaxCase mixed;
    mixed.label = "softmax kernel mixed";
    mixed.num_q_heads = 3;
    mixed.pos = 5;
    mixed.max_seq_len = 8;
    mixed.score.resize(mixed.num_q_heads * mixed.max_seq_len);
    for (uint32_t i = 0; i < mixed.score.size(); ++i) {
        mixed.score[i] = -2.5f + 0.37f * std::sin((i + 1) * 0.19f) +
                         0.11f * static_cast<float>(i % 7);
    }
    cases.push_back(mixed);

    cases.push_back({"softmax kernel large magnitude", 2, 4, 7,
                     {80.0f, 81.5f, 79.25f, 90.0f, 88.0f, -3.0f, 4.0f, -45.0f,
                      -44.5f, -46.25f, -41.0f, -42.0f, 6.0f, -7.0f}});

    for (const auto& test_case : cases) {
        if (!run_case(test_case)) {
            return 1;
        }
    }

    std::cout << "[PASS] softmax_f32_kernel" << std::endl;
    return 0;
}
