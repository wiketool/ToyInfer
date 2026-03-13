#include "sampler.h"

#include <cfloat>
namespace toyinfer {

Sampler::Sampler(const LLMConfig& llmconfig, const Options& options)
    : llmconfig(llmconfig), options(options) {}

int32_t Sampler::sample(std::unique_ptr<float[]>& logits) {
    float max_prob = -FLT_MAX;
    uint32_t max_idx = -1;
    for (uint32_t i = 0; i < llmconfig.vocab_size; i++) {
        if (logits[i] > max_prob) {
            max_prob = logits[i];
            max_idx = i;
        }
    }
    return max_idx;
}
}  // namespace toyinfer