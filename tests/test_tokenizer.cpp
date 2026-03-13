#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

#include "config.h"
#include "options.h"
#include "tokenizer.h"

namespace {
std::string expected_prompt(const std::string& user_input) {
    return "<|im_start|>user\n" + user_input +
           "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
}
}  // namespace

int main() {
    const std::filesystem::path model_dir =
        std::filesystem::path(TOYINFER_SOURCE_DIR) / "models/Qwen3-4B";
    if (!std::filesystem::exists(model_dir / "config.json")) {
        std::cout << "[SKIP] Model files not found at " << model_dir
                  << std::endl;
        return 0;
    }

    toyinfer::Options options{};
    options.model_dir = model_dir.string();
    options.enable_thinking = false;

    toyinfer::LLMConfig config(options);
    toyinfer::Tokenizer tokenizer(options, config);

    const std::string inputs[] = {"aaa", "aaaaa"};
    for (const std::string& input : inputs) {
        std::unique_ptr<char[]> prompt;
        tokenizer.render_prompt(prompt, input.c_str(), nullptr);
        if (std::string(prompt.get()) != expected_prompt(input)) {
            std::cerr << "[FAIL] Unexpected prompt rendering for input: "
                      << input << std::endl;
            return 1;
        }

        std::unique_ptr<uint32_t[]> token_ids;
        uint32_t token_count = 0;
        tokenizer.encode(prompt.get(), token_ids, token_count);
        if (token_count == 0) {
            std::cerr << "[FAIL] Prompt encoded to zero tokens for input: "
                      << input << std::endl;
            return 1;
        }
    }

    std::cout << "[PASS] tokenizer prompt rendering" << std::endl;
    return 0;
}
