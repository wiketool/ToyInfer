#pragma once

#include "config.h"
#include "options.h"
#include "qwen3.h"
namespace toyinfer {

class Transformer {
   private:
    Qwen3 qwen3_;

   public:
    Transformer(const Options& options, const LLMConfig& config);
    void forward(const std::vector<int>& input_ids, std::vector<float>& output);
};
}  // namespace toyinfer