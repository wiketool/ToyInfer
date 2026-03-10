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
    cudaMalloc(&sum_d, sizeof(float));
    cudaMalloc(&inv_freq_d, sizeof(float) * llmconfig.head_dim / 2);
    cudaMalloc(&q_d, sizeof(bf16) * llmconfig.head_dim *
                         llmconfig.num_attention_heads);
    cudaMalloc(&key_cache_d, sizeof(bf16) * llmconfig.num_hidden_layers *
                                 options.max_seq_len *
                                 llmconfig.num_key_value_heads *
                                 llmconfig.num_attention_heads);
    cudaMalloc(&val_cache_d, sizeof(bf16) * llmconfig.num_hidden_layers *
                                 options.max_seq_len *
                                 llmconfig.num_key_value_heads *
                                 llmconfig.num_attention_heads);
}

void Transformer::State::free() {}

Transformer::Transformer(const Options& options, const LLMConfig& config)
    : qwen3_(options, config), llmconfig(config), options(options) {
    qwen3_.load_weights();
    state.alloc(options, config);
}

void Transformer::forward(uint32_t token_id, uint32_t pos,
                          std::unique_ptr<float[]> logits) {
    const uint32_t kv_dim = llmconfig.head_dim * llmconfig.num_key_value_heads;
    const uint32_t q_dim = llmconfig.head_dim * llmconfig.num_attention_heads;
    // 预计算theta
    precompute_freq_f32(state.inv_freq_d, pos, llmconfig.rope_theta);
    const bf16* embedding_ptr =
        qwen3_.embed_tokens_d + llmconfig.hidden_size * token_id;
    // decoder layer
    for (int i = 0; i < llmconfig.num_hidden_layers; ++i) {
        const Qwen3::Layer& layer_ref = qwen3_.layer[i];
        const bf16* norm_weight = layer_ref.input_layernorm_d;
        rmsnorm_bf16<NUM_THREADS>(
            embedding_ptr, norm_weight, state.hidden_d, state.sum_d,
            llmconfig.rms_norm_eps, llmconfig.hidden_size);
        // qkv proj
        attn_single_proj_bf16<NUM_THREADS>(layer_ref.attention.q_proj_d,
                                           state.hidden_d, state.q_d,
                                           q_dim, llmconfig.hidden_size);
        bf16* key_ptr = state.key_cache_d + (i * options.max_seq_len * kv_dim) +
                        (pos * kv_dim);
        bf16* val_ptr = state.val_cache_d + (i * options.max_seq_len * kv_dim) +
                        (pos * kv_dim);
        attn_single_proj_bf16<NUM_THREADS>(layer_ref.attention.k_proj_d,
                                           state.hidden_d, key_ptr,
                                           kv_dim, llmconfig.hidden_size);
        attn_single_proj_bf16<NUM_THREADS>(layer_ref.attention.v_proj_d,
                                           state.hidden_d, val_ptr,
                                           kv_dim, llmconfig.hidden_size);
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
    }
}
}  // namespace toyinfer