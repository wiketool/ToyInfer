#include "sampler.h"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <random>

#include "profiling.h"
namespace toyinfer {

Sampler::Sampler(const LLMConfig& llmconfig, const Options& options)
    : llmconfig(llmconfig),
      options(options),
      tokens_prob(std::make_unique<TokenProb[]>(llmconfig.vocab_size)) {}

// topK -> softmax ->topP -> minP
int32_t Sampler::sample(const float* logits) {
    ScopedNvtxRange sample_range("sampler.sample");
    if (fabs(options.temperature - 0.0f) < 1e-3 || options.top_k == 1) {
        return argmax(logits);
    }
    // temperature
    for (uint32_t idx = 0; idx < llmconfig.vocab_size; idx++) {
        tokens_prob[idx].prob = logits[idx] / options.temperature;
        tokens_prob[idx].index = idx;
    }
    // topK
    uint32_t k = options.top_k < llmconfig.vocab_size ? options.top_k
                                                      : llmconfig.vocab_size;
    quick_select(tokens_prob, 0, llmconfig.vocab_size - 1, k);
    float max_prob = -FLT_MAX, sum_prob = 0.0f;
    for (uint32_t i = 0; i < k; i++) {
        max_prob = fmaxf(tokens_prob[i].prob, max_prob);
    }
    for (uint32_t i = 0; i < k; i++) {
        tokens_prob[i].prob =
            expf((tokens_prob[i].prob - max_prob) / options.temperature);
        sum_prob += tokens_prob[i].prob;
    }
    for (uint32_t i = 0; i < k; i++) {
        tokens_prob[i].prob /= sum_prob;
    }
    auto prob_cmp = [](const TokenProb& a, const TokenProb& b) {
        return a.prob > b.prob;
    };
    std::sort(tokens_prob.get(), tokens_prob.get() + k, prob_cmp);
    float acc_prob = 0.0f;
    uint32_t p_idx;
    for (p_idx = 0; p_idx < k; p_idx++) {
        acc_prob += tokens_prob[p_idx].prob;
        if (acc_prob > options.top_p) {
            break;
        }
    }
    k = p_idx + 1;
    static std::mt19937 engine(
        std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    float coin = distribution(engine) * acc_prob;
    float acc = 0.0f;
    for (uint32_t i = 0; i < k; i++) {
        acc += tokens_prob[i].prob;
        if (acc > coin) {
            return tokens_prob[i].index;
        }
    }
    return tokens_prob[k - 1].index;
}

uint32_t Sampler::argmax(const float* logits) {
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

void Sampler::quick_select(std::unique_ptr<TokenProb[]>& tokens_prob,
                           const uint32_t left, const uint32_t right,
                           const uint32_t k) {
    assert(k > 0);
    uint32_t pivot = left;
    uint32_t i = left, j = right;
    while (i < j) {
        while (j > i && tokens_prob[j].prob <= tokens_prob[pivot].prob) {
            j--;
        }
        while (i < j && tokens_prob[i].prob >= tokens_prob[pivot].prob) {
            i++;
        }
        std::swap(tokens_prob[i], tokens_prob[j]);
    }
    std::swap(tokens_prob[pivot], tokens_prob[i]);
    if ((i + 1) == k) {
        return;
    } else if ((i + 1) < k) {
        quick_select(tokens_prob, i + 1, right, k);
    } else {
        quick_select(tokens_prob, left, i - 1, k);
    }
}
}  // namespace toyinfer
