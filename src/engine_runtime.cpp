#include "engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "options.h"
#include "profiling.h"
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

struct Beam {
    std::vector<uint32_t> tokens;
    float logprob = 0.0f;
    bool finished = false;
};

struct InferenceStats {
    uint32_t prompt_tokens = 0;
    uint32_t generated_tokens = 0;
    double tokenizer_encode_ms = 0.0;
    double ttft_ms = 0.0;
    double tpot_ms = 0.0;
    AverageTime token_sample_time;
    AverageTime tokenizer_decode_time;
    std::chrono::steady_clock::time_point e2e_start;
    std::chrono::steady_clock::time_point inference_start;
    std::chrono::steady_clock::time_point first_token_time;
    std::chrono::steady_clock::time_point end_time;
    bool ttft_recorded = false;
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

void record_ttft(InferenceStats& stats) {
    if (stats.ttft_recorded) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    stats.first_token_time = now;
    stats.ttft_ms = elapsed_ms(stats.inference_start, now);
    stats.ttft_recorded = true;
}

void finalize_stats(InferenceStats& stats) {
    stats.end_time = std::chrono::steady_clock::now();
    if (stats.generated_tokens > 1 && stats.ttft_recorded) {
        stats.tpot_ms = elapsed_ms(stats.first_token_time, stats.end_time) /
                        static_cast<double>(stats.generated_tokens - 1);
    } else {
        stats.tpot_ms = 0.0;
    }
}

uint32_t sample_with_stats(Sampler& sampler, const float* logits,
                           InferenceStats& stats) {
    double sample_ms = 0.0;
    uint32_t token_id = 0;
    {
        ScopedCpuTimer timer(sample_ms);
        token_id = sampler.sample(logits);
    }
    stats.token_sample_time.add(sample_ms);
    return token_id;
}

const char* decode_with_stats(Tokenizer& tokenizer, uint32_t token_id,
                              InferenceStats& stats) {
    double decode_ms = 0.0;
    const char* text = nullptr;
    {
        ScopedCpuTimer timer(decode_ms);
        text = tokenizer.decode(token_id);
    }
    stats.tokenizer_decode_time.add(decode_ms);
    return text;
}

void print_inference_stats(const InferenceStats& stats,
                           const TransformerProfileStats& transformer_profile,
                           bool detail_time) {
    const uint32_t total_tokens = stats.prompt_tokens + stats.generated_tokens;
    const double inference_time_s =
        std::chrono::duration<double>(stats.end_time - stats.inference_start)
            .count();
    const double tokens_per_sec =
        inference_time_s > 0.0
            ? static_cast<double>(total_tokens) / inference_time_s
            : 0.0;

    std::printf(
        "[perf] prompt_tokens=%u, generated_tokens=%u, total_tokens=%u, "
        "inference_time=%.3fs, TTFT=%.3fms, tokens/s=%.2f\n",
        stats.prompt_tokens, stats.generated_tokens, total_tokens,
        inference_time_s, stats.ttft_ms, tokens_per_sec);

    if (!detail_time) {
        return;
    }

    const double e2e_latency_ms = elapsed_ms(stats.e2e_start, stats.end_time);
    std::printf("[detail] tokenizer_encode=%.3f ms\n",
                stats.tokenizer_encode_ms);
    std::printf("[detail] TTFT=%.3f ms\n", stats.ttft_ms);
    std::printf("[detail] TPOT=%.3f ms\n", stats.tpot_ms);
    std::printf("[detail] token_sample_avg=%.3f ms\n",
                stats.token_sample_time.average_ms());
    std::printf("[detail] tokenizer_decode_avg=%.3f ms\n",
                stats.tokenizer_decode_time.average_ms());
    std::printf("[detail] E2E_latency=%.3f ms\n", e2e_latency_ms);
    std::printf("[detail] prefill_total=%.3f ms\n",
                transformer_profile.prefill_total_ms);
    std::printf("[detail] prefill.layer.attn_block=%.3f ms\n",
                transformer_profile.prefill_layer_attn_block_ms);
    std::printf("[detail] prefill.layer.mlp_block=%.3f ms\n",
                transformer_profile.prefill_layer_mlp_block_ms);
    std::printf("[detail] decode_forward_total=%.3f ms\n",
                transformer_profile.decode_forward_total_ms);
    if (transformer_profile.decode_layer_stage_timing_available) {
        std::printf("[detail] decode.layer.qkv_and_cache=%.3f ms\n",
                    transformer_profile.decode_layer_qkv_and_cache_ms);
        std::printf("[detail] decode.layer.attention=%.3f ms\n",
                    transformer_profile.decode_layer_attention_ms);
        std::printf("[detail] decode.layer.mlp=%.3f ms\n",
                    transformer_profile.decode_layer_mlp_ms);
    } else {
        std::printf(
            "[detail] decode.layer.qkv_and_cache=N/A (CUDA graph enabled)\n");
        std::printf(
            "[detail] decode.layer.attention=N/A (CUDA graph enabled)\n");
        std::printf("[detail] decode.layer.mlp=N/A (CUDA graph enabled)\n");
    }
}

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
    const float denom =
        std::pow(static_cast<float>(std::max(1U, gen_len)), length_penalty);
    return beam.logprob / denom;
}

const float* replay_prompt_with_decode(Transformer& transformer,
                                       const uint32_t* token_ids,
                                       uint32_t token_cnt) {
    const float* logits = nullptr;
    for (uint32_t i = 0; i < token_cnt; ++i) {
        logits = transformer.forward(token_ids[i], i);
    }
    return logits;
}

const float* run_prompt_prefill(Transformer& transformer,
                                const Options& options,
                                const uint32_t* token_ids, uint32_t token_cnt) {
    if (token_ids == nullptr || token_cnt == 0) {
        return nullptr;
    }
    ScopedNvtxRange prefill_range(options.use_dedicated_prefill
                                      ? "engine.prompt_prefill.dedicated"
                                      : "engine.prompt_prefill.decode_replay");
    if (options.use_dedicated_prefill) {
        return transformer.prefill(token_ids, token_cnt);
    }
    return replay_prompt_with_decode(transformer, token_ids, token_cnt);
}

const float* compute_logits_for_tokens(Transformer& transformer,
                                       const Options& options,
                                       const std::vector<uint32_t>& tokens) {
    if (tokens.empty()) {
        return nullptr;
    }
    return run_prompt_prefill(transformer, options, tokens.data(),
                              static_cast<uint32_t>(tokens.size()));
}

static void build_profile_token_sequence(
    Tokenizer& tokenizer, uint32_t target_len,
    std::unique_ptr<uint32_t[]>& out_token_ids, uint32_t& out_token_cnt,
    const char* seed_text =
        "这是一个用于大模型推理性能分析的固定中文测试样本。"
        "我们会重复这段文本来构造稳定且可复现的输入序列，"
        "用于测量prefill、decode、TTFT以及各类CUDA算子的耗时表现。") {
    out_token_ids.reset();
    out_token_cnt = 0;

    if (target_len == 0) {
        return;
    }

    // 1) 先把 seed 文本编码成一段合法 token pattern
    std::unique_ptr<uint32_t[]> seed_token_ids;
    uint32_t seed_token_cnt = 0;
    tokenizer.encode(seed_text, seed_token_ids, seed_token_cnt);

    if (seed_token_cnt == 0) {
        throw std::runtime_error(
            "build_profile_token_sequence: seed_token_cnt == 0");
    }

    // 2) 按 pattern 循环填充到目标长度
    out_token_ids = std::make_unique<uint32_t[]>(target_len);
    for (uint32_t i = 0; i < target_len; ++i) {
        out_token_ids[i] = seed_token_ids[i % seed_token_cnt];
    }
    out_token_cnt = target_len;
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
        ScopedNvtxRange turn_range("engine.chat.turn");
        transformer.reset_profile();
        InferenceStats stats;
        stats.e2e_start = std::chrono::steady_clock::now();
        std::unique_ptr<uint32_t[]> token_ids;
        uint32_t token_cnt;
        char* line;
        if (options.bench == "") {
            const char* console = "ToyInfer> ";
            line = linenoise(console);
            if (line == nullptr) {
                continue;
            }
            if (strcmp(line, "\\quit") == 0) {
                break;
            }
            std::unique_ptr<char[]> prompt;
            tokenizer.render_prompt(prompt, line, nullptr);
#ifdef DEBUG
            std::cout << prompt.get() << std::endl;
#endif
            {
                ScopedCpuTimer timer(stats.tokenizer_encode_ms);
                tokenizer.encode(prompt.get(), token_ids, token_cnt);
            }
#ifdef DEBUG
            for (uint32_t i = 0; i < token_cnt; i++) {
                std::cout << token_ids[i] << " ";
            }
            std::cout << std::endl;
#endif
        } else if (options.bench == "short") {
            build_profile_token_sequence(tokenizer, 64, token_ids, token_cnt);
        } else if (options.bench == "long") {
            build_profile_token_sequence(tokenizer, 4096, token_ids, token_cnt);
        }
        int input_len = token_cnt;
        FormatState format_state;
        stats.prompt_tokens = token_cnt;
        stats.inference_start = std::chrono::steady_clock::now();
        if (options.beam_size <= 1) {
            ScopedNvtxRange greedy_range("engine.decode.greedy");
            if (token_cnt > 0 &&
                token_cnt < static_cast<uint32_t>(options.max_seq_len)) {
                const float* logits_h = run_prompt_prefill(
                    transformer, options, token_ids.get(), token_cnt);
                uint32_t pos = token_cnt;
                uint32_t next_token_id =
                    sample_with_stats(sampler, logits_h, stats);
                bool assistance_end = false;
                record_ttft(stats);
                while (assistance_end == false &&
                       pos < static_cast<uint32_t>(options.max_seq_len)) {
                    if (next_token_id ==
                        static_cast<uint32_t>(llm_config.eos_token_id)) {
                        assistance_end = true;
                    } else {
                        stats.generated_tokens++;
                        print_formatted(
                            OutputRole::Assistant,
                            decode_with_stats(tokenizer, next_token_id, stats),
                            format_state);
                        fflush(stdout);
                    }
                    if (assistance_end ||
                        (pos + 1) >=
                            static_cast<uint32_t>(options.max_seq_len)) {
                        break;
                    }
                    if(options.bench != "" && pos > input_len + 1024){
                        break;
                    }
                    logits_h = transformer.forward(next_token_id, pos);
                    next_token_id = sample_with_stats(sampler, logits_h, stats);
                    pos++;
                }
            }
        } else {
            ScopedNvtxRange beam_range("engine.decode.beam_search");
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
                        const float* logits_h = compute_logits_for_tokens(
                            transformer, options, beam.tokens);
                        const float log_z =
                            compute_log_z(logits_h, llm_config.vocab_size);
                        const std::vector<uint32_t> top_ids = top_k_indices(
                            logits_h, llm_config.vocab_size,
                            static_cast<uint32_t>(options.beam_size));
                        for (uint32_t token : top_ids) {
                            Beam next = beam;
                            next.tokens.push_back(token);
                            fflush(stdout);
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
                        if (!stats.ttft_recorded) {
                            record_ttft(stats);
                        }
                        stats.generated_tokens++;
                        print_formatted(
                            OutputRole::Assistant,
                            decode_with_stats(tokenizer, token, stats),
                            format_state);
                    }
                }
            }
        }
        reset_format(format_state);
        std::printf("\033[0m\n");
        finalize_stats(stats);
        print_inference_stats(stats, transformer.profile_stats(),
                              options.detail_time);
        if (options.bench == "") {
            linenoiseFree(line);
        }else{
            return;
        }
    }
}

}  // namespace toyinfer
