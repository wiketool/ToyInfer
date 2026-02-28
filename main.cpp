#include "CLI/CLI.hpp"
#include "banner.h"
#include "config.h"
#include "kernel_warpper.h"
#include "linenoise.h"
#include "logger.h"
#include "options.h"
#include "qwen3.h"
#include "tokenizer.h"

int main(int argc, char** argv) {
    init_logger();
    toyinfer::Options options{};

    CLI::App app{"ToyInfer"};
    options.options_from_cli(app);
    CLI11_PARSE(app, argc, argv);

    toyinfer::LLMConfig llmConfig{options};
    toyinfer::Tokenizer tokenizer{options, llmConfig};

    toyinfer::Utils::print_banner();
    toyinfer::Qwen3 qwen3{options,llmConfig};
    qwen3.load_weights();

    while (1) {
        char* line = linenoise("ToyInfer> ");
        if (line == nullptr) {
            return 0;
        }

        linenoiseFree(line);
    }

    return 0;
}