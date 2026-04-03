#pragma once

#include "CLI/CLI.hpp"

namespace toyinfer {
struct Options {
    std::string model_dir{};

    int32_t max_seq_len = 0;
    bool thinking = true;
    bool use_dedicated_prefill = true;
    bool use_multi_stream = true;
    bool enable_cuda_graph = true;
    bool detail_time = false;
    std::string bench;

    // sample options
    float temperature = 0.6f;
    float top_p = 0.95f;
    int32_t top_k = 20;

    // beam search options
    int32_t beam_size = 1;
    float length_penalty = 1.0f;

    void options_from_cli(CLI::App& app);
};
}  // namespace toyinfer
