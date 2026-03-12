#pragma once

#include "config.h"
#include "options.h"
#include "tokenizer.h"
#include "transformer.h"
namespace toyinfer {
class Engine {
   public:
    Engine(const Options& options);
    void chat();

   private:
    const Options& options;
    LLMConfig llm_config;
    Tokenizer tokenizer;
    Transformer transformer;
    std::unique_ptr<float[]> logits_h;
};
}  // namespace toyinfer
