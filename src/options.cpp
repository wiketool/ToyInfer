#pragma once

#include "options.h"

#include <string>

namespace toyinfer {

void Options::options_from_cli(CLI::App& app) {
    app.add_option("--model_dir", model_dir, "Model dir")->required();
    app.add_option("--max_seq_len", max_seq_len,
                   "Maximum inference context window");
    app.add_option("--thinking", thinking,
                   "Enable thinking(Temperature=0.6, TopP=0.95, TopK=20, and "
                   "MinP=0)")
        ->default_val(true);
    app.add_option("--temperature", temperature)->default_val(0.6);
    app.add_option("--top_k", top_k)->default_val(20);
    app.add_option("--top_p", top_p)->default_val(0.95);
    app.add_option("--beam_size", beam_size)
        ->default_val(1)
        ->check(CLI::Range(1, 8))
        ->description("Beam size for beam search (1 disables beam search)");
    app.add_option("--length_penalty", length_penalty)
        ->default_val(1.0f)
        ->check(CLI::Range(0.0f, 5.0f))
        ->description(
            "Length penalty for beam search (>1 favors longer, <1 favors "
            "shorter)");
}

}  // namespace toyinfer
