#pragma once

#include <cstdint>
#include <memory>

#include "config.h"
#include "options.h"
#include "qwen3.h"
namespace toyinfer {

class Transformer {
    struct State {
        bf16* hidden_d; // hidden state, [hidden_size]
        float* inv_freq_d; // rope kernel, [head_dim / 2]
        void alloc(const LLMConfig& llmconfig);
        void free();
    };

   private:
    const LLMConfig& llmconfig;
    Qwen3 qwen3_;
    State state;

   public:
    Transformer(const Options& options, const LLMConfig& config);
    void forward(uint32_t token_id ,uint32_t pos, std::unique_ptr<float[]> logits);
};
}  // namespace toyinfer