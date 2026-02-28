#pragma once

#include "options.h"

#include <string>

namespace toyinfer {

void Options::options_from_cli(CLI::App& app) {
    app.add_option("--model_dir", model_dir, "Model dir")->required();
    app.add_option("--max_seq_len", max_seq_len,
                   "Maximum inference context window");
    app.add_flag("--thinking", enable_thinking,
                 "Enable thinking(Temperature=0.6, TopP=0.95, TopK=20, and "
                 "MinP=0)")
        ->default_val(false);
}

}  // namespace toyinfer
