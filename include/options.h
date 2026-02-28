#pragma once

#include "CLI/CLI.hpp"

namespace toyinfer {
struct Options {
    std::string model_dir;

    int32_t max_seq_len;
    bool enable_thinking;

    // sample options
    float temperature;
    float top_p;
    int32_t top_k;
    float min_p;

    void options_from_cli(CLI::App& app);
};
}  // namespace toyinfer
