#include "test_helpers.h"
#include "../src/kernel.cu"
#include "attention_test_common.h"

namespace {

template <uint32_t NUM_THREADS>
void launch_kernel(const toyinfer::bf16* Q, const toyinfer::bf16* Ks,
                   float* score, uint32_t num_q_heads, uint32_t num_kv_heads,
                   uint32_t heads_dim, uint32_t max_seq_len, uint32_t pos) {
    dim3 block_dim{NUM_THREADS};
    dim3 grid_dim{pos + 1};
    toyinfer::gqa_qk_gemv_bf16_kernel<NUM_THREADS>
        <<<grid_dim, block_dim>>>(Q, Ks, score, num_q_heads, num_kv_heads,
                                  heads_dim, max_seq_len);
    TCUDA_CHECK(cudaGetLastError());
}

bool run_case(const attention_test::AttentionCase& test_case) {
    const auto q_bf16 = kernel_test::to_bf16(test_case.q);
    const auto ks_bf16 = kernel_test::to_bf16(test_case.ks);
    const auto q_q = kernel_test::to_float(q_bf16);
    const auto ks_q = kernel_test::to_float(ks_bf16);

    std::vector<float> initial_score(test_case.num_q_heads * test_case.max_seq_len,
                                     0.0f);
    for (uint32_t i = 0; i < initial_score.size(); ++i) {
        initial_score[i] = -0.45f + 0.03f * static_cast<float>((i * 7) % 13);
    }
    const auto expected = attention_test::reference_qk_scores(
        test_case, q_q, ks_q, initial_score);

    kernel_test::DeviceBuffer<toyinfer::bf16> dev_q(q_bf16.size());
    kernel_test::DeviceBuffer<toyinfer::bf16> dev_ks(ks_bf16.size());
    kernel_test::DeviceBuffer<float> dev_score(initial_score.size());
    dev_q.copy_from_host(q_bf16);
    dev_ks.copy_from_host(ks_bf16);
    dev_score.copy_from_host(initial_score);

    launch_kernel<attention_test::kThreads>(
        dev_q.get(), dev_ks.get(), dev_score.get(), test_case.num_q_heads,
        test_case.num_kv_heads, test_case.heads_dim, test_case.max_seq_len,
        test_case.pos);
    TCUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> out;
    dev_score.copy_to_host(out);
    return kernel_test::expect_vector_close(out, expected, 1e-2f, 1e-2f,
                                            test_case.label + " qk kernel");
}

}  // namespace

int main() {
    if (!kernel_test::ensure_cuda_device(true)) {
        return 0;
    }

    const std::vector<attention_test::AttentionCase> cases = {
        attention_test::make_attention_case("qk kernel pos0 gqa", 2, 1, 64, 0, 5,
                                            0.07f, 0.03f),
        attention_test::make_attention_case("qk kernel full-kv", 4, 4, 64, 6, 8,
                                            0.05f, 0.025f),
        attention_test::make_attention_case("qk kernel tiled-seq", 8, 2, 128, 9,
                                            12, 0.03f, 0.017f),
    };

    for (const auto& test_case : cases) {
        if (!run_case(test_case)) {
            return 1;
        }
    }

    std::cout << "[PASS] gqa_qk_gemv_bf16_kernel" << std::endl;
    return 0;
}
