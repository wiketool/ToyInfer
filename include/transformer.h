#pragma once

#include <cstdint>
#include <memory>

#include "config.h"
#include "options.h"
#include "qwen3.h"
namespace toyinfer {

class Transformer {
    struct State {
        bf16* hidden_d;  // hidden state, [hidden_size]
        bf16* residual_d;
        bf16* x_d;

        float* sum_d;       // dim = 1
        float* inv_freq_d;  // rope kernel, [head_dim / 2]

        // for attention
        bf16* q_d;
        bf16* key_cache_d;
        bf16* val_cache_d;
        float* score;  // [num_atten_heads, MAX_SEQ_LEN]
        float* o_buffer_d;
        bf16* o_d;

        // mlp
        bf16* gate_d;
        bf16* up_d;
        bf16* intermedia_d;
        // logits
        float* logits_d;

        void alloc(const Options& options, const LLMConfig& llmconfig);
        void free();
    };

   private:
    const LLMConfig& llmconfig;
    const Options& options;
    Qwen3 qwen3_;
    State state;

   public:
    Transformer(const Options& options, const LLMConfig& config);
    void forward(uint32_t token_id, uint32_t pos,
                 std::unique_ptr<float[]> logits);
};
}  // namespace toyinfer