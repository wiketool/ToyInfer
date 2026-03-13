#pragma once

#include "CLI/CLI.hpp"

namespace toyinfer {
struct Options {
    std::string model_dir;

    int32_t max_seq_len;
    bool thinking;

    // sample options
    float temperature;
    float top_p;
    int32_t top_k;

    // beam search options
    int32_t beam_size;
    float length_penalty;

    void options_from_cli(CLI::App& app);
};
}  // namespace toyinfer
