#include "test_helpers.h"
#include "../src/kernel.cu"
#include "attention_test_common.h"

namespace {

template <uint32_t TILE_SEQ>
void launch_kernel(float* score, float* o, const toyinfer::bf16* Vs,
                   uint32_t num_q_heads, uint32_t num_kv_heads,
                   uint32_t heads_dim, uint32_t pos, uint32_t max_seq_len) {
    dim3 block_dim{heads_dim / 2};
    dim3 grid_dim{num_q_heads, (pos + 1 + TILE_SEQ - 1) / TILE_SEQ};
    toyinfer::apply_score2v_f32_kernel<TILE_SEQ>
        <<<grid_dim, block_dim>>>(score, o, Vs, num_q_heads, num_kv_heads,
                                  heads_dim, pos, max_seq_len);
    TCUDA_CHECK(cudaGetLastError());
}

template <uint32_t TILE_SEQ>
bool run_case(const attention_test::AttentionCase& test_case,
              const std::vector<float>& score) {
    const auto vs_bf16 = kernel_test::to_bf16(test_case.vs);
    const auto vs_q = kernel_test::to_float(vs_bf16);

    std::vector<float> initial_o(test_case.num_q_heads * test_case.heads_dim, 0.0f);
    for (uint32_t i = 0; i < initial_o.size(); ++i) {
        initial_o[i] = 0.6f - 0.04f * static_cast<float>((i * 3) % 17);
    }
    const auto expected = attention_test::reference_apply_score(
        test_case, score, vs_q, initial_o);

    kernel_test::DeviceBuffer<float> dev_score(score.size());
    kernel_test::DeviceBuffer<float> dev_o(initial_o.size());
    kernel_test::DeviceBuffer<toyinfer::bf16> dev_vs(vs_bf16.size());
    dev_score.copy_from_host(score);
    dev_o.copy_from_host(initial_o);
    dev_vs.copy_from_host(vs_bf16);

    launch_kernel<TILE_SEQ>(dev_score.get(), dev_o.get(), dev_vs.get(),
                            test_case.num_q_heads, test_case.num_kv_heads,
                            test_case.heads_dim, test_case.pos,
                            test_case.max_seq_len);
    TCUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> out;
    dev_o.copy_to_host(out);
    return kernel_test::expect_vector_close(
        out, expected, 5e-2f, 5e-2f,
        test_case.label + " apply_score tile=" + std::to_string(TILE_SEQ));
}

}  // namespace

int main() {
    if (!kernel_test::ensure_cuda_device(true)) {
        return 0;
    }

    auto single_tile = attention_test::make_attention_case(
        "apply_score kernel single-tile", 4, 2, 64, 3, 8, 0.04f, 0.02f);
    std::vector<float> single_score(
        single_tile.num_q_heads * single_tile.max_seq_len, 0.0f);
    for (uint32_t h = 0; h < single_tile.num_q_heads; ++h) {
        for (uint32_t t = 0; t <= single_tile.pos; ++t) {
            single_score[h * single_tile.max_seq_len + t] =
                0.12f + 0.03f * static_cast<float>((h + t) % 5);
        }
    }
    if (!run_case<32>(single_tile, single_score)) {
        return 1;
    }

    auto multi_tile = attention_test::make_attention_case(
        "apply_score kernel multi-tile", 8, 2, 128, 5, 10, 0.03f, 0.017f);
    std::vector<float> multi_score(
        multi_tile.num_q_heads * multi_tile.max_seq_len, 0.0f);
    for (uint32_t h = 0; h < multi_tile.num_q_heads; ++h) {
        float row_sum = 0.0f;
        for (uint32_t t = 0; t <= multi_tile.pos; ++t) {
            const float value =
                0.08f + 0.01f * static_cast<float>((h * 3 + t * 5) % 9);
            multi_score[h * multi_tile.max_seq_len + t] = value;
            row_sum += value;
        }
        for (uint32_t t = 0; t <= multi_tile.pos; ++t) {
            multi_score[h * multi_tile.max_seq_len + t] /= row_sum;
        }
    }
    if (!run_case<2>(multi_tile, multi_score)) {
        return 1;
    }

    std::cout << "[PASS] apply_score2v_f32_kernel" << std::endl;
    return 0;
}
