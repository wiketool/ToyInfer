#pragma once

#include "config.h"

namespace toyinfer {

class Sampler {
    struct TokenProb {
        uint32_t index;
        float prob;
    };

   public:
    Sampler(const LLMConfig& llmconfig, const Options& options);
    int32_t sample(const float* logits);

   private:
    const LLMConfig& llmconfig;
    const Options& options;
    std::unique_ptr<TokenProb[]> tokens_prob;

    uint32_t argmax(const float* logits);
    void quick_select(std::unique_ptr<TokenProb[]>& tokens_prob,
                      const uint32_t left, const uint32_t right,
                      const uint32_t k);
};
}  // namespace toyinfer
