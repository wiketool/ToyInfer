#include "transformer.h"

#include <cuda_runtime.h>

#include <cstdint>

#include "config.h"
#include "kernel_warpper.h"
#include "options.h"
#include "qwen3.h"
#include "type.h"

namespace toyinfer {

void Transformer::State::alloc(const LLMConfig& llmconfig) {
    cudaMalloc(&hidden_d, sizeof(bf16) * llmconfig.hidden_size);
    cudaMalloc(&inv_freq_d, sizeof(float) * llmconfig.head_dim / 2);
}

void Transformer::State::free() {}

Transformer::Transformer(const Options& options, const LLMConfig& config)
    : qwen3_(options, config), llmconfig(config) {
    qwen3_.load_weights();
    state.alloc(config);
}

void Transformer::forward(uint32_t token_id, uint32_t pos,
                          std::unique_ptr<float[]> logits) {
    // 预计算theta
    launch_precompute_inv_freq_kernel(state.inv_freq_d, pos,
                                      llmconfig.rope_theta);
    state.hidden_d = qwen3_.embed_tokens_d + llmconfig.hidden_size * token_id;

    // decoder layer
    for (int i = 0; i < llmconfig.num_hidden_layers; ++i) {
        
    }
}
}  // namespace toyinfer