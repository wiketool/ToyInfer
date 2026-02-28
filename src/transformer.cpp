#include "transformer.h"

#include "config.h"
#include "options.h"
#include "qwen3.h"

namespace toyinfer {
Transformer::Transformer(const Options& options, const LLMConfig& config)
    : qwen3_(options, config) {}
}  // namespace toyinfer