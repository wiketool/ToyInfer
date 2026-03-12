#pragma once

#include "config.h"

namespace toyinfer {
class Sampler {
   public:
    Sampler(const LLMConfig& llmconfig);
    int32_t sample(std::unique_ptr<float[]>& logits);

   private:
    const LLMConfig& llmconfig;
};
}  // namespace toyinfer
