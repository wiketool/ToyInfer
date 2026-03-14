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
    cudaMalloc(&hidden_d, sizeof(bf16) * llmconfig.hidden_size);
    cudaMalloc(&residual_d, sizeof(bf16) * llmconfig.hidden_size);
    cudaMalloc(&x_d, sizeof(bf16) * llmconfig.hidden_size);
    cudaMalloc(&sum_d, sizeof(float));
    cudaMalloc(&inv_freq_d, sizeof(float) * llmconfig.head_dim / 2);
    cudaMalloc(&q_d, sizeof(bf16) * llmconfig.head_dim *
                         llmconfig.num_attention_heads);
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
        cudaEventCreate(&event_d[i]);
    }

    // 预计算theta,只需要预计算一次就行
    precompute_freq_f32(inv_freq_d, llmconfig.head_dim, llmconfig.rope_theta);
    cudaDeviceSynchronize();
}

void Transformer::State::free() {
    cudaFree(hidden_d);
    cudaFree(residual_d);
    cudaFree(x_d);
    cudaFree(sum_d);
    cudaFree(inv_freq_d);
    cudaFree(q_d);
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
    // decoder layer
    for (int i = 0; i < llmconfig.num_hidden_layers; ++i) {
        const Qwen3::Layer& layer_ref = qwen3_.layer[i];
        rmsnorm_bf16<NUM_THREADS>(
            state.hidden_d, layer_ref.input_layernorm_d, state.residual_d,
            state.sum_d, llmconfig.rms_norm_eps, llmconfig.hidden_size);
        // qkv proj
        gemv_proj_bf16<NUM_THREADS>(layer_ref.attention.q_proj_d,
                                    state.residual_d, state.q_d, q_dim,
                                    llmconfig.hidden_size);
        bf16* key_ptr = state.key_cache_d + (i * options.max_seq_len * kv_dim) +
                        (pos * kv_dim);
        bf16* val_ptr = state.val_cache_d + (i * options.max_seq_len * kv_dim) +
                        (pos * kv_dim);
        gemv_proj_bf16<NUM_THREADS>(layer_ref.attention.k_proj_d,
                                    state.residual_d, key_ptr, kv_dim,
                                    llmconfig.hidden_size);
        gemv_proj_bf16<NUM_THREADS>(layer_ref.attention.v_proj_d,
                                    state.residual_d, val_ptr, kv_dim,
                                    llmconfig.hidden_size);
        // qk norm
        multi_rmsnorm_bf16(state.q_d, layer_ref.attention.q_norm_d, state.q_d,
                           llmconfig.rms_norm_eps,
                           llmconfig.num_attention_heads, llmconfig.head_dim);
        multi_rmsnorm_bf16(key_ptr, layer_ref.attention.k_norm_d, key_ptr,
                           llmconfig.rms_norm_eps,
                           llmconfig.num_key_value_heads, llmconfig.head_dim);
        // rope qk
        rope_bf16(state.q_d, state.inv_freq_d, pos,
                  llmconfig.num_attention_heads, llmconfig.head_dim);
        rope_bf16(key_ptr, state.inv_freq_d, pos, llmconfig.num_key_value_heads,
                  llmconfig.head_dim);

        // attention QKV
        const bf16* Ks = state.key_cache_d + i * options.max_seq_len * kv_dim;
        const bf16* Vs = state.val_cache_d + i * options.max_seq_len * kv_dim;

        attention_bf16<NUM_THREADS, TILE_SEQ>(
            state.q_d, Ks, Vs, state.score, state.o_buffer_d, state.o_d,
            llmconfig.num_attention_heads, llmconfig.num_key_value_heads,
            llmconfig.head_dim, pos, options.max_seq_len);
        // o proj
        gemv_proj_bf16<NUM_THREADS>(
            layer_ref.attention.o_proj_d, state.o_d, state.residual_d,
            llmconfig.hidden_size,
            llmconfig.num_attention_heads * llmconfig.head_dim);
        // hidden = hidden + residual
        residual_add_bf16<NUM_THREADS>(state.residual_d, state.hidden_d,
                                       llmconfig.hidden_size);
        // residual = hidden
        cudaMemcpy(state.residual_d, state.hidden_d,
                   sizeof(bf16) * llmconfig.hidden_size,
                   cudaMemcpyDeviceToDevice);
        //    post attention norm
        rmsnorm_bf16<NUM_THREADS>(
            state.hidden_d, layer_ref.post_attention_layernorm_d, state.x_d,
            state.sum_d, llmconfig.rms_norm_eps, llmconfig.hidden_size);
        // MLP
        // gate + up proj
        gemv_proj_bf16<NUM_THREADS>(layer_ref.ffn.gate_proj_d, state.x_d,
                                    state.gate_d, llmconfig.intermediate_size,
                                    llmconfig.hidden_size);
        gemv_proj_bf16<NUM_THREADS>(layer_ref.ffn.up_proj_d, state.x_d,
                                    state.up_d, llmconfig.intermediate_size,
                                    llmconfig.hidden_size);
        // swiglu(gate) * up
        swiglu_bf16x2<NUM_THREADS>(state.gate_d, state.up_d, state.intermedia_d,
                                   llmconfig.intermediate_size);
        // down proj
        gemv_proj_bf16<NUM_THREADS>(
            layer_ref.ffn.down_proj_d, state.intermedia_d, state.hidden_d,
            llmconfig.hidden_size, llmconfig.intermediate_size);
        residual_add_bf16<NUM_THREADS>(state.residual_d, state.hidden_d,
                                       llmconfig.hidden_size);
    }
    rmsnorm_bf16<NUM_THREADS>(state.hidden_d, qwen3_.norm_d, state.x_d,
                              state.sum_d, llmconfig.rms_norm_eps,
                              llmconfig.hidden_size);
    gemv_proj_bf162float<NUM_THREADS>(qwen3_.lmhead_d, state.x_d,
                                      state.logits_d, llmconfig.vocab_size,
                                      llmconfig.hidden_size);
    cudaMemcpy(logits_h, state.logits_d, sizeof(float) * llmconfig.vocab_size,
               cudaMemcpyDeviceToHost);
    return logits_h;
}
}  // namespace toyinfer
