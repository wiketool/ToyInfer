#include "transformer.h"

#include <cuda_runtime.h>

#include <cstdint>

#include "config.h"
#include "kernel_warpper.h"
#include "options.h"
#include "qwen3.h"
#include "type.h"

namespace toyinfer {

void Transformer::State::alloc(const Options& options,
                               const LLMConfig& llmconfig) {
    cudaMallocHost(&pos_h, sizeof(uint32_t));
    cudaMalloc(&pos_d, sizeof(uint32_t));
    cudaMalloc(&hidden_d, sizeof(bf16) * llmconfig.hidden_size);
    cudaMalloc(&residual_d, sizeof(bf16) * llmconfig.hidden_size);
    cudaMalloc(&x_d, sizeof(bf16) * llmconfig.hidden_size);
    cudaMalloc(&sum_d, sizeof(float));
    cudaMalloc(&inv_freq_d, sizeof(float) * llmconfig.head_dim / 2);
    cudaMalloc(&q_d, sizeof(bf16) * llmconfig.head_dim *
                         llmconfig.num_attention_heads);
    cudaMalloc(&key_d, sizeof(bf16) * llmconfig.head_dim *
                           llmconfig.num_key_value_heads);
    cudaMalloc(&val_d, sizeof(bf16) * llmconfig.head_dim *
                           llmconfig.num_key_value_heads);
    cudaMalloc(&key_cache_d, sizeof(bf16) * llmconfig.num_hidden_layers *
                                 options.max_seq_len *
                                 llmconfig.num_key_value_heads *
                                 llmconfig.head_dim);
    cudaMalloc(&val_cache_d, sizeof(bf16) * llmconfig.num_hidden_layers *
                                 options.max_seq_len *
                                 llmconfig.num_key_value_heads *
                                 llmconfig.head_dim);
    cudaMalloc(&score, sizeof(float) * llmconfig.num_attention_heads *
                           options.max_seq_len);
    cudaMalloc(&o_buffer_d, sizeof(float) * llmconfig.num_attention_heads *
                                llmconfig.head_dim);
    cudaMalloc(&o_d, sizeof(bf16) * llmconfig.num_attention_heads *
                         llmconfig.head_dim);
    cudaMalloc(&gate_d, sizeof(bf16) * llmconfig.intermediate_size);
    cudaMalloc(&up_d, sizeof(bf16) * llmconfig.intermediate_size);
    cudaMalloc(&intermedia_d, sizeof(bf16) * llmconfig.intermediate_size);
    cudaMalloc(&logits_d, sizeof(float) * llmconfig.vocab_size);

    for (uint32_t i = 0; i < 3; i++) {
        cudaStreamCreate(&stream_d[i]);
        cudaEventCreateWithFlags(&event_d[i], cudaEventDisableTiming);
    }

    // 预计算theta,只需要预计算一次就行
    precompute_freq_f32(inv_freq_d, llmconfig.head_dim, llmconfig.rope_theta);
    cudaDeviceSynchronize();
}

void Transformer::State::free() {
    if (graph_exec_d != nullptr) {
        cudaGraphExecDestroy(graph_exec_d);
    }
    if (graph_d != nullptr) {
        cudaGraphDestroy(graph_d);
    }
    cudaFreeHost(pos_h);
    cudaFree(pos_d);
    cudaFree(hidden_d);
    cudaFree(residual_d);
    cudaFree(x_d);
    cudaFree(sum_d);
    cudaFree(inv_freq_d);
    cudaFree(q_d);
    cudaFree(key_d);
    cudaFree(val_d);
    cudaFree(key_cache_d);
    cudaFree(val_cache_d);
    cudaFree(score);
    cudaFree(o_buffer_d);
    cudaFree(o_d);
    cudaFree(gate_d);
    cudaFree(up_d);
    cudaFree(intermedia_d);
    cudaFree(logits_d);
    for (uint32_t i = 0; i < 3; i++) {
        cudaStreamDestroy(stream_d[i]);
        cudaEventDestroy(event_d[i]);
    }
}

Transformer::Transformer(const Options& options, const LLMConfig& config)
    : llmconfig(config),
      options(options),
      qwen3_(options, config),
      logits_h(nullptr) {
    qwen3_.load_weights();
    state.alloc(options, config);
    cudaMallocHost(&logits_h, sizeof(float) * llmconfig.vocab_size);
}

Transformer::~Transformer() {
    state.free();
    if (logits_h != nullptr) {
        cudaFreeHost(logits_h);
    }
}

const float* Transformer::forward(uint32_t token_id, uint32_t pos) {
    const uint32_t kv_dim = llmconfig.head_dim * llmconfig.num_key_value_heads;
    const uint32_t q_dim = llmconfig.head_dim * llmconfig.num_attention_heads;
    const bf16* embedding_ptr =
        qwen3_.embed_tokens_d + llmconfig.hidden_size * token_id;
    cudaMemcpy(state.hidden_d, embedding_ptr,
               sizeof(bf16) * llmconfig.hidden_size, cudaMemcpyDeviceToDevice);
    *(state.pos_h) = pos;
    if (state.graph_d == nullptr) {
        cudaStreamBeginCapture(state.stream_d[0], cudaStreamCaptureModeGlobal);
        cudaMemcpyAsync(state.pos_d, state.pos_h, sizeof(uint32_t),
                        cudaMemcpyHostToDevice, state.stream_d[0]);
        // decoder layer
        for (int i = 0; i < llmconfig.num_hidden_layers; ++i) {
            const Qwen3::Layer& layer_ref = qwen3_.layer[i];
            bf16* layer_key_cache =
                state.key_cache_d + (i * options.max_seq_len * kv_dim);
            bf16* layer_val_cache =
                state.val_cache_d + (i * options.max_seq_len * kv_dim);
            rmsnorm_bf16<NUM_THREADS>(
                state.hidden_d, layer_ref.input_layernorm_d, state.residual_d,
                state.sum_d, llmconfig.rms_norm_eps, llmconfig.hidden_size,
                state.stream_d[0]);
            cudaEventRecord(state.event_d[0], state.stream_d[0]);

            // qkv proj
            gemv_proj_bf16<NUM_THREADS>(
                layer_ref.attention.q_proj_d, state.residual_d, state.q_d,
                q_dim, llmconfig.hidden_size, state.stream_d[0]);
            // wait for stream 1 finish input hidden norm
            cudaStreamWaitEvent(state.stream_d[1], state.event_d[0]);
            cudaStreamWaitEvent(state.stream_d[2], state.event_d[0]);
            gemv_proj_bf16<NUM_THREADS>(
                layer_ref.attention.k_proj_d, state.residual_d, state.key_d,
                kv_dim, llmconfig.hidden_size, state.stream_d[1]);
            gemv_proj_bf16<NUM_THREADS>(
                layer_ref.attention.v_proj_d, state.residual_d, state.val_d,
                kv_dim, llmconfig.hidden_size, state.stream_d[2]);

            // qk norm
            multi_rmsnorm_bf16(state.q_d, layer_ref.attention.q_norm_d,
                               state.q_d, llmconfig.rms_norm_eps,
                               llmconfig.num_attention_heads,
                               llmconfig.head_dim, state.stream_d[0]);
            multi_rmsnorm_bf16(state.key_d, layer_ref.attention.k_norm_d,
                               state.key_d, llmconfig.rms_norm_eps,
                               llmconfig.num_key_value_heads,
                               llmconfig.head_dim, state.stream_d[1]);
            // rope qk
            rope_bf16_graph(state.q_d, state.inv_freq_d, state.pos_d,
                            llmconfig.num_attention_heads, llmconfig.head_dim,
                            state.stream_d[0]);
            rope_bf16_graph(state.key_d, state.inv_freq_d, state.pos_d,
                            llmconfig.num_key_value_heads,
                            llmconfig.head_dim, state.stream_d[1]);
            write_kv_cache_bf16<NUM_THREADS>(state.key_d, layer_key_cache,
                                             state.pos_d, kv_dim,
                                             state.stream_d[1]);
            write_kv_cache_bf16<NUM_THREADS>(state.val_d, layer_val_cache,
                                             state.pos_d, kv_dim,
                                             state.stream_d[2]);
            cudaEventRecord(state.event_d[1], state.stream_d[1]);
            cudaEventRecord(state.event_d[2], state.stream_d[2]);

            // attention QKV
            cudaStreamWaitEvent(state.stream_d[0], state.event_d[1]);
            cudaStreamWaitEvent(state.stream_d[0], state.event_d[2]);
            attention_bf16_graph<NUM_THREADS, TILE_SEQ>(
                state.q_d, layer_key_cache, layer_val_cache, state.score,
                state.o_buffer_d, state.o_d,
                llmconfig.num_attention_heads, llmconfig.num_key_value_heads,
                llmconfig.head_dim, state.pos_d, options.max_seq_len,
                state.stream_d[0]);
            // o proj
            gemv_proj_bf16<NUM_THREADS>(
                layer_ref.attention.o_proj_d, state.o_d, state.residual_d,
                llmconfig.hidden_size,
                llmconfig.num_attention_heads * llmconfig.head_dim,
                state.stream_d[0]);
            // hidden = hidden + residual
            residual_add_bf16<NUM_THREADS>(state.residual_d, state.hidden_d,
                                           llmconfig.hidden_size,
                                           state.stream_d[0]);
            // residual = hidden
            cudaMemcpyAsync(state.residual_d, state.hidden_d,
                            sizeof(bf16) * llmconfig.hidden_size,
                            cudaMemcpyDeviceToDevice, state.stream_d[0]);
            //    post attention norm
            rmsnorm_bf16<NUM_THREADS>(
                state.hidden_d, layer_ref.post_attention_layernorm_d, state.x_d,
                state.sum_d, llmconfig.rms_norm_eps, llmconfig.hidden_size,
                state.stream_d[0]);
            cudaEventRecord(state.event_d[0], state.stream_d[0]);
            // MLP
            // gate + up proj
            gemv_proj_bf16<NUM_THREADS>(
                layer_ref.ffn.gate_proj_d, state.x_d, state.gate_d,
                llmconfig.intermediate_size, llmconfig.hidden_size,
                state.stream_d[0]);
            cudaStreamWaitEvent(state.stream_d[1], state.event_d[0]);
            gemv_proj_bf16<NUM_THREADS>(layer_ref.ffn.up_proj_d, state.x_d,
                                        state.up_d, llmconfig.intermediate_size,
                                        llmconfig.hidden_size,
                                        state.stream_d[1]);
            cudaEventRecord(state.event_d[1], state.stream_d[1]);
            cudaStreamWaitEvent(state.stream_d[0], state.event_d[1]);

            // swiglu(gate) * up
            swiglu_bf16x2<NUM_THREADS>(
                state.gate_d, state.up_d, state.intermedia_d,
                llmconfig.intermediate_size, state.stream_d[0]);
            // down proj
            gemv_proj_bf16<NUM_THREADS>(
                layer_ref.ffn.down_proj_d, state.intermedia_d, state.hidden_d,
                llmconfig.hidden_size, llmconfig.intermediate_size,
                state.stream_d[0]);
            residual_add_bf16<NUM_THREADS>(state.residual_d, state.hidden_d,
                                           llmconfig.hidden_size,
                                           state.stream_d[0]);
        }
        rmsnorm_bf16<NUM_THREADS>(state.hidden_d, qwen3_.norm_d, state.x_d,
                                  state.sum_d, llmconfig.rms_norm_eps,
                                  llmconfig.hidden_size, state.stream_d[0]);
        gemv_proj_bf162float<NUM_THREADS>(
            qwen3_.lmhead_d, state.x_d, state.logits_d, llmconfig.vocab_size,
            llmconfig.hidden_size, state.stream_d[0]);
        cudaStreamEndCapture(state.stream_d[0], &state.graph_d);
        cudaGraphInstantiate(&state.graph_exec_d, state.graph_d);
    }
    cudaGraphLaunch(state.graph_exec_d, state.stream_d[0]);
    cudaStreamSynchronize(state.stream_d[0]);
    cudaMemcpy(logits_h, state.logits_d, sizeof(float) * llmconfig.vocab_size,
               cudaMemcpyDeviceToHost);
    return logits_h;
}
}  // namespace toyinfer
