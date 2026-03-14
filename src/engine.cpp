#include "engine.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "options.h"
#include "tokenizer.h"
#include "transformer.h"

namespace toyinfer {
namespace {
enum class OutputRole { UserInput, Thinking, Assistant };

struct FormatState {
    bool in_bold = false;
    bool in_thinking = false;
    bool active_role_set = false;
    OutputRole active_role = OutputRole::Assistant;
    bool active_bold = false;
    std::string pending;
};

const char* role_color(OutputRole role) {
    switch (role) {
        case OutputRole::UserInput:
            return "36";  // cyan
        case OutputRole::Thinking:
            return "33";  // yellow
        case OutputRole::Assistant:
        default:
            return "32";  // green
    }
}

void append_style(std::string& out, OutputRole role, bool bold) {
    out.append("\033[0m");
    out.append("\033[");
    if (bold) {
        out.append("1;");
    }
    out.append(role_color(role));
    out.append("m");
}

bool is_prefix_of_marker(const std::string& s) {
    static const char* kMarkers[] = {"**"};
    for (const char* marker : kMarkers) {
        const size_t marker_len = std::strlen(marker);
        if (s.size() < marker_len &&
            std::strncmp(s.c_str(), marker, s.size()) == 0) {
            return true;
        }
    }
    return false;
}

std::string format_chunk(OutputRole base_role, const char* text,
                         FormatState& state) {
    if (text == nullptr || text[0] == '\0') {
        return "";
    }

    std::string data = state.pending;
    state.pending.clear();
    data.append(text);

    std::string out{""};
    size_t i = 0;
    while (i < data.size()) {
        const size_t remaining = data.size() - i;
        const char* ptr = data.c_str() + i;

        if (remaining >= 2 && std::strncmp(ptr, "**", 2) == 0) {
            state.in_bold = !state.in_bold;
            i += 2;
            continue;
        } else if (remaining >= 7 && std::strncmp(ptr, "<think>", 7) == 0) {
            state.in_thinking = true;
            i += 7;
            continue;
        } else if (remaining >= 8 && std::strncmp(ptr, "</think>", 8) == 0) {
            state.in_thinking = false;
            i += 8;
            continue;
        } else if (remaining >= 10 &&
                   std::strncmp(ptr, "<thinking>", 10) == 0) {
            state.in_thinking = true;
            i += 10;
            continue;
        } else if (remaining >= 11 &&
                   std::strncmp(ptr, "</thinking>", 11) == 0) {
            state.in_thinking = false;
            i += 11;
            continue;
        }
        if (is_prefix_of_marker(data.substr(i))) {
            state.pending = data.substr(i);
            break;
        }
        OutputRole role = state.in_thinking ? OutputRole::Thinking : base_role;
        if (!state.active_role_set || role != state.active_role ||
            state.in_bold != state.active_bold) {
            append_style(out, role, state.in_bold);
            state.active_role = role;
            state.active_bold = state.in_bold;
            state.active_role_set = true;
        }
        out.push_back(data[i]);
        i += 1;
    }

    return out;
}

void print_formatted(OutputRole base_role, const char* text,
                     FormatState& state) {
    std::string formatted = format_chunk(base_role, text, state);
    if (!formatted.empty()) {
        std::printf("%s", formatted.c_str());
    }
}

void reset_format(FormatState& state) {
    if (state.active_role_set || state.in_bold) {
        std::printf("\033[0m");
    }
    state = FormatState{};
}

struct Beam {
    std::vector<uint32_t> tokens;
    float logprob = 0.0f;
    bool finished = false;
};

float compute_log_z(const float* logits, uint32_t vocab_size) {
    float max_logit = -std::numeric_limits<float>::infinity();
    for (uint32_t i = 0; i < vocab_size; ++i) {
        max_logit = std::max(max_logit, logits[i]);
    }
    double sum = 0.0;
    for (uint32_t i = 0; i < vocab_size; ++i) {
        sum += std::exp(static_cast<double>(logits[i] - max_logit));
    }
    return max_logit + static_cast<float>(std::log(sum));
}

std::vector<uint32_t> top_k_indices(const float* logits, uint32_t vocab_size,
                                    uint32_t k) {
    k = std::min<uint32_t>(k, vocab_size);
    using Candidate = std::pair<float, uint32_t>;
    std::priority_queue<Candidate, std::vector<Candidate>,
                        std::greater<Candidate>>
        heap;
    for (uint32_t i = 0; i < vocab_size; ++i) {
        const float logit = logits[i];
        if (heap.size() < k) {
            heap.push({logit, i});
            continue;
        }
        if (logit > heap.top().first) {
            heap.pop();
            heap.push({logit, i});
        }
    }
    std::vector<uint32_t> ids;
    ids.reserve(k);
    while (!heap.empty()) {
        ids.push_back(heap.top().second);
        heap.pop();
    }
    std::sort(ids.begin(), ids.end(),
              [&](uint32_t a, uint32_t b) { return logits[a] > logits[b]; });
    return ids;
}

float beam_score(const Beam& beam, uint32_t prompt_len, float length_penalty) {
    const uint32_t gen_len =
        beam.tokens.size() > prompt_len
            ? static_cast<uint32_t>(beam.tokens.size() - prompt_len)
            : 0U;
    // 1 / Len^penalty
    const float denom =
        std::pow(static_cast<float>(std::max(1U, gen_len)), length_penalty);
    // logprob都是负数，长度越长负的越多，惩罚大于1，会让负数变小，倾向于输出长
    return beam.logprob / denom;
}

const float* compute_logits_for_tokens(Transformer& transformer,
                                       const std::vector<uint32_t>& tokens) {
    const float* logits = nullptr;
    for (size_t pos = 0; pos < tokens.size(); ++pos) {
        logits = transformer.forward(tokens[pos], static_cast<uint32_t>(pos));
    }
    return logits;
}
}  // namespace

Engine::Engine(Options& options)
    : options(options),
      llm_config(options),
      tokenizer(options, llm_config),
      transformer(options, llm_config),
      sampler(llm_config, options) {
    if (llm_config.max_position_embeddings < options.max_seq_len) {
        options.max_seq_len = llm_config.max_position_embeddings;
    }
};

void Engine::chat() {
    while (1) {
        const char* console = "ToyInfer> ";
        char* line = linenoise(console);
        if (line == nullptr) {
            continue;
        }
        if (strcmp(line, "\\quit") == 0) {
            break;
        }
        std::unique_ptr<uint32_t[]> token_ids;
        uint32_t token_cnt;
        std::unique_ptr<char[]> prompt;
        tokenizer.render_prompt(prompt, line, nullptr);
#ifdef DEBUG
        std::cout << prompt.get() << std::endl;
#endif
        tokenizer.encode(prompt.get(), token_ids, token_cnt);
#ifdef DEBUG
        for (uint32_t i = 0; i < token_cnt; i++) {
            std::cout << token_ids[i] << " ";
        }
        std::cout << std::endl;
#endif
        FormatState format_state;
        if (options.beam_size <= 1) {
            uint32_t pos = 0;
            uint32_t token_id;
            uint32_t next_token_id = 0;
            bool assistance_end = false;
            const float* logits_h = nullptr;
            while (assistance_end == false &&
                   pos < static_cast<uint32_t>(options.max_seq_len)) {
                if (pos < token_cnt) {
                    token_id = token_ids[pos];
                } else {
                    token_id = next_token_id;
                }
                logits_h = transformer.forward(token_id, pos);
                next_token_id = sampler.sample(logits_h);
                // printf("next token id: %d\n", next_token_id);
                if ((pos + 1) < token_cnt) {
                    next_token_id = token_ids[pos + 1];
                } else {
                    if (next_token_id ==
                        static_cast<uint32_t>(llm_config.eos_token_id)) {
                        assistance_end = true;
                    } else {
                        print_formatted(OutputRole::Assistant,
                                        tokenizer.decode(next_token_id),
                                        format_state);
                        fflush(stdout);
                    }
                }
                pos++;
            }
        } else {
            const uint32_t prompt_len = token_cnt;
            if (prompt_len < static_cast<uint32_t>(options.max_seq_len)) {
                std::vector<uint32_t> prompt_tokens(token_cnt);
                for (uint32_t i = 0; i < token_cnt; ++i) {
                    prompt_tokens[i] = token_ids[i];
                }

                std::vector<Beam> beams;
                beams.reserve(static_cast<size_t>(options.beam_size));
                beams.push_back({prompt_tokens, 0.0f, false});

                const uint32_t max_gen_len =
                    static_cast<uint32_t>(options.max_seq_len) - prompt_len;
                for (uint32_t step = 0; step < max_gen_len; ++step) {
                    std::vector<Beam> candidates;
                    candidates.reserve(static_cast<size_t>(options.beam_size) *
                                       static_cast<size_t>(options.beam_size));
                    for (const auto& beam : beams) {
                        if (beam.finished) {
                            candidates.push_back(beam);
                            continue;
                        }
                        const float* logits_h =
                            compute_logits_for_tokens(transformer, beam.tokens);
                        const float log_z =
                            compute_log_z(logits_h, llm_config.vocab_size);
                        const std::vector<uint32_t> top_ids = top_k_indices(
                            logits_h, llm_config.vocab_size,
                            static_cast<uint32_t>(options.beam_size));
                        for (uint32_t token : top_ids) {
                            Beam next = beam;
                            next.tokens.push_back(token);
                            fflush(stdout);
                            // 乘法结果对数=分别对数相加，log_z=(max+sum_exp)，写下数学公式就知道了
                            next.logprob += logits_h[token] - log_z;
                            next.finished =
                                (token == static_cast<uint32_t>(
                                              llm_config.eos_token_id));
                            candidates.push_back(std::move(next));
                        }
                    }
                    std::sort(candidates.begin(), candidates.end(),
                              [&](const Beam& a, const Beam& b) {
                                  return beam_score(a, prompt_len,
                                                    options.length_penalty) >
                                         beam_score(b, prompt_len,
                                                    options.length_penalty);
                              });
                    if (candidates.size() >
                        static_cast<size_t>(options.beam_size)) {
                        candidates.resize(
                            static_cast<size_t>(options.beam_size));
                    }
                    beams = std::move(candidates);

                    bool done = true;
                    for (const auto& beam : beams) {
                        if (!beam.finished) {
                            done = false;
                            break;
                        }
                    }
                    if (done) {
                        break;
                    }
                }

                if (!beams.empty()) {
                    const Beam* best = &beams[0];
                    float best_score =
                        beam_score(*best, prompt_len, options.length_penalty);
                    for (size_t i = 1; i < beams.size(); ++i) {
                        const float score = beam_score(beams[i], prompt_len,
                                                       options.length_penalty);
                        if (score > best_score) {
                            best = &beams[i];
                            best_score = score;
                        }
                    }
                    for (size_t i = prompt_len; i < best->tokens.size(); ++i) {
                        const uint32_t token = best->tokens[i];
                        if (token ==
                            static_cast<uint32_t>(llm_config.eos_token_id)) {
                            break;
                        }
                        print_formatted(OutputRole::Assistant,
                                        tokenizer.decode(token), format_state);
                    }
                }
            }
        }
        reset_format(format_state);
        printf("\033[0m\n");
        linenoiseFree(line);
    }
}

}  // namespace toyinfer
