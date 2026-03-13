#pragma once

#include "config.h"

namespace toyinfer {
class Sampler {
   public:
    Sampler(const LLMConfig& llmconfig, const Options& options);
    int32_t sample(std::unique_ptr<float[]>& logits);

   private:
    const LLMConfig& llmconfig;
    const Options& options;
};
}  // namespace toyinfer
