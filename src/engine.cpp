#include "engine.h"

#include <cstring>
#include <memory>
#include <string>

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
}  // namespace

Engine::Engine(const Options& options)
    : options(options),
      llm_config(options),
      tokenizer(options, llm_config),
      transformer(options, llm_config),
      logits_h(std::make_unique<float[]>(llm_config.vocab_size)),
      sampler(llm_config, options) {};

void Engine::chat() {
    uint32_t pos = 0;
    uint32_t token_id;
    uint32_t next_token_id;
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
        bool assistance_end = false;
        while (assistance_end == false && pos < options.max_seq_len) {
            if (pos < token_cnt) {
                token_id = token_ids[pos];
            } else {
                token_id = next_token_id;
            }
            transformer.forward(token_id, pos, logits_h);
            next_token_id = sampler.sample(logits_h);
            // printf("next token id: %d\n", next_token_id);
            if ((pos + 1) < token_cnt) {
                next_token_id = token_ids[pos + 1];
            } else {
                if (next_token_id == llm_config.eos_token_id) {
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
        reset_format(format_state);
        printf("\033[0m\n");
        pos = 0;

        linenoiseFree(line);
    }
}

}  // namespace toyinfer
