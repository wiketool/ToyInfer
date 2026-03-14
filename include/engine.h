#pragma once

#include "config.h"
#include "options.h"
#include "sampler.h"
#include "tokenizer.h"
#include "transformer.h"
namespace toyinfer {
class Engine {
   public:
    Engine(Options& options);
    void chat();

   private:
    Options& options;
    LLMConfig llm_config;
    Tokenizer tokenizer;
    Transformer transformer;
    Sampler sampler;
};
}  // namespace toyinfer
