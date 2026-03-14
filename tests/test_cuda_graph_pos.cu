#include "test_helpers.h"
#include "../src/kernel.cu"
#include "attention_test_common.h"
#include "rope_test_common.h"

namespace {

bool run_case() {
    constexpr uint32_t kThreads = 256;
    constexpr uint32_t kMaxSeqLen = 4;

    rope_test::Case base_case;
    base_case.label = "cuda graph pos replay";
    base_case.num_heads = 1;
    base_case.head_dim = 4;
    base_case.pos = 0;
    base_case.buffer_size = 4;
    base_case.qk = {0.25f, -0.5f, 1.0f, -1.5f};
    base_case.inv_freq = {0.3f, 0.7f};

    const auto input_bf16 = kernel_test::to_bf16(base_case.qk);
    const auto input_q = kernel_test::to_float(input_bf16);

    auto pos0_case = base_case;
    pos0_case.pos = 0;
    const auto expected_pos0 =
        rope_test::reference_output(pos0_case, input_q);

    auto pos2_case = base_case;
    pos2_case.pos = 2;
    const auto expected_pos2 =
        rope_test::reference_output(pos2_case, input_q);

    kernel_test::DeviceBuffer<toyinfer::bf16> input_d(input_bf16.size());
    kernel_test::DeviceBuffer<toyinfer::bf16> work_d(input_bf16.size());
    kernel_test::DeviceBuffer<float> inv_freq_d(base_case.inv_freq.size());
    kernel_test::DeviceBuffer<toyinfer::bf16> cache_d(kMaxSeqLen *
                                                      input_bf16.size());
    input_d.copy_from_host(input_bf16);
    inv_freq_d.copy_from_host(base_case.inv_freq);
    cache_d.fill_zero();

    uint32_t* pos_h = nullptr;
    uint32_t* pos_d = nullptr;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;

    TCUDA_CHECK(cudaMallocHost(&pos_h, sizeof(uint32_t)));
    TCUDA_CHECK(cudaMalloc(&pos_d, sizeof(uint32_t)));
    TCUDA_CHECK(cudaStreamCreate(&stream));

    *pos_h = 0;
    TCUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    TCUDA_CHECK(cudaMemcpyAsync(pos_d, pos_h, sizeof(uint32_t),
                                cudaMemcpyHostToDevice, stream));
    TCUDA_CHECK(cudaMemcpyAsync(work_d.get(), input_d.get(),
                                sizeof(toyinfer::bf16) * input_bf16.size(),
                                cudaMemcpyDeviceToDevice, stream));
    toyinfer::rope_bf16_graph(work_d.get(), inv_freq_d.get(), pos_d,
                              base_case.num_heads, base_case.head_dim, stream);
    toyinfer::write_kv_cache_bf16<kThreads>(work_d.get(), cache_d.get(), pos_d,
                                            input_bf16.size(), stream);
    TCUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    TCUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph));

    TCUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    TCUDA_CHECK(cudaStreamSynchronize(stream));

    *pos_h = 2;
    TCUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    TCUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<toyinfer::bf16> cache_out_bf16;
    cache_d.copy_to_host(cache_out_bf16);
    const auto cache_out = kernel_test::to_float(cache_out_bf16);

    std::vector<float> expected(cache_out.size(), 0.0f);
    for (uint32_t i = 0; i < input_bf16.size(); ++i) {
        expected[i] = expected_pos0[i];
        expected[2 * input_bf16.size() + i] = expected_pos2[i];
    }

    TCUDA_CHECK(cudaGraphExecDestroy(graph_exec));
    TCUDA_CHECK(cudaGraphDestroy(graph));
    TCUDA_CHECK(cudaStreamDestroy(stream));
    TCUDA_CHECK(cudaFree(pos_d));
    TCUDA_CHECK(cudaFreeHost(pos_h));

    return kernel_test::expect_vector_close(cache_out, expected, 2e-2f, 2e-2f,
                                            base_case.label);
}

bool run_attention_case() {
    auto pos3_case = attention_test::make_attention_case(
        "cuda graph attention replay", 2, 1, 64, 3, 5, 0.07f, 0.03f);
    auto pos0_case = pos3_case;
    pos0_case.pos = 0;

    const auto q_bf16 = kernel_test::to_bf16(pos3_case.q);
    const auto ks_bf16 = kernel_test::to_bf16(pos3_case.ks);
    const auto vs_bf16 = kernel_test::to_bf16(pos3_case.vs);
    const auto q_q = kernel_test::to_float(q_bf16);
    const auto ks_q = kernel_test::to_float(ks_bf16);
    const auto vs_q = kernel_test::to_float(vs_bf16);

    const auto expected_score0 =
        attention_test::reference_attention_score(pos0_case, q_q, ks_q);
    const auto expected_o_buffer0 =
        attention_test::reference_attention_buffer(pos0_case, expected_score0,
                                                   vs_q);
    const std::vector<float> zeros(pos0_case.num_q_heads * pos0_case.heads_dim,
                                   0.0f);
    const auto expected_o0 = attention_test::reference_convert(
        expected_o_buffer0, zeros, pos0_case.num_q_heads * pos0_case.heads_dim);

    const auto expected_score3 =
        attention_test::reference_attention_score(pos3_case, q_q, ks_q);
    const auto expected_o_buffer3 =
        attention_test::reference_attention_buffer(pos3_case, expected_score3,
                                                   vs_q);
    const auto expected_o3 = attention_test::reference_convert(
        expected_o_buffer3, zeros, pos3_case.num_q_heads * pos3_case.heads_dim);

    std::vector<float> dirty_score(expected_score3.size(), 1.0f);
    std::vector<float> dirty_o_buffer(expected_o_buffer3.size(), -2.0f);
    std::vector<toyinfer::bf16> dirty_o(expected_o3.size(),
                                        kernel_test::float_to_bf16(0.5f));

    kernel_test::DeviceBuffer<toyinfer::bf16> q_d(q_bf16.size());
    kernel_test::DeviceBuffer<toyinfer::bf16> ks_d(ks_bf16.size());
    kernel_test::DeviceBuffer<toyinfer::bf16> vs_d(vs_bf16.size());
    kernel_test::DeviceBuffer<float> score_d(expected_score3.size());
    kernel_test::DeviceBuffer<float> o_buffer_d(expected_o_buffer3.size());
    kernel_test::DeviceBuffer<toyinfer::bf16> o_d(expected_o3.size());
    q_d.copy_from_host(q_bf16);
    ks_d.copy_from_host(ks_bf16);
    vs_d.copy_from_host(vs_bf16);
    score_d.copy_from_host(dirty_score);
    o_buffer_d.copy_from_host(dirty_o_buffer);
    o_d.copy_from_host(dirty_o);

    uint32_t* pos_h = nullptr;
    uint32_t* pos_d = nullptr;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;

    TCUDA_CHECK(cudaMallocHost(&pos_h, sizeof(uint32_t)));
    TCUDA_CHECK(cudaMalloc(&pos_d, sizeof(uint32_t)));
    TCUDA_CHECK(cudaStreamCreate(&stream));

    *pos_h = pos0_case.pos;
    TCUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    TCUDA_CHECK(cudaMemcpyAsync(pos_d, pos_h, sizeof(uint32_t),
                                cudaMemcpyHostToDevice, stream));
    toyinfer::attention_bf16_graph<attention_test::kThreads, 32>(
        q_d.get(), ks_d.get(), vs_d.get(), score_d.get(), o_buffer_d.get(),
        o_d.get(), pos3_case.num_q_heads, pos3_case.num_kv_heads,
        pos3_case.heads_dim, pos_d, pos3_case.max_seq_len, stream);
    TCUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    TCUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph));

    TCUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    TCUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> score0_out;
    std::vector<float> o_buffer0_out;
    std::vector<toyinfer::bf16> o0_out_bf16;
    score_d.copy_to_host(score0_out);
    o_buffer_d.copy_to_host(o_buffer0_out);
    o_d.copy_to_host(o0_out_bf16);
    const auto o0_out = kernel_test::to_float(o0_out_bf16);

    *pos_h = pos3_case.pos;
    TCUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    TCUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> score3_out;
    std::vector<float> o_buffer3_out;
    std::vector<toyinfer::bf16> o3_out_bf16;
    score_d.copy_to_host(score3_out);
    o_buffer_d.copy_to_host(o_buffer3_out);
    o_d.copy_to_host(o3_out_bf16);
    const auto o3_out = kernel_test::to_float(o3_out_bf16);

    TCUDA_CHECK(cudaGraphExecDestroy(graph_exec));
    TCUDA_CHECK(cudaGraphDestroy(graph));
    TCUDA_CHECK(cudaStreamDestroy(stream));
    TCUDA_CHECK(cudaFree(pos_d));
    TCUDA_CHECK(cudaFreeHost(pos_h));

    if (!kernel_test::expect_vector_close(score0_out, expected_score0, 5e-3f,
                                          5e-3f,
                                          pos0_case.label + " score pos0")) {
        return false;
    }
    if (!kernel_test::expect_vector_close(
            o_buffer0_out, expected_o_buffer0, 5e-2f, 5e-2f,
            pos0_case.label + " o_buffer pos0")) {
        return false;
    }
    if (!kernel_test::expect_vector_close(o0_out, expected_o0, 5e-2f, 5e-2f,
                                          pos0_case.label + " output pos0")) {
        return false;
    }
    if (!kernel_test::expect_vector_close(score3_out, expected_score3, 5e-3f,
                                          5e-3f,
                                          pos3_case.label + " score pos3")) {
        return false;
    }
    if (!kernel_test::expect_vector_close(
            o_buffer3_out, expected_o_buffer3, 5e-2f, 5e-2f,
            pos3_case.label + " o_buffer pos3")) {
        return false;
    }
    return kernel_test::expect_vector_close(o3_out, expected_o3, 5e-2f, 5e-2f,
                                            pos3_case.label + " output pos3");
}

}  // namespace

int main() {
    if (!kernel_test::ensure_cuda_device(true)) {
        return 0;
    }

    if (!run_case()) {
        return 1;
    }
    if (!run_attention_case()) {
        return 1;
    }

    std::cout << "[PASS] cuda_graph_pos" << std::endl;
    return 0;
}
