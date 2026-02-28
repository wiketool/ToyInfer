#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>

#include "config.h"
#include "options.h"

namespace toyinfer {
class Tokenizer {
   public:
    Tokenizer(const Options& options, const LLMConfig& llmConfig);
    ~Tokenizer();

   private:
    std::filesystem::path model_dir;
    bool thinking_ = false;
    uint32_t vocab_size_ = 0;
    uint32_t bos_token_id_ = 0;
    uint32_t eos_token_id_ = 0;
    char** vocab_ = nullptr;        // 词表，大小为 vocab_size_
    float* merge_score_ = nullptr;  // BPE合并分数，大小为 vocab_size_
    char* system_prompt = nullptr;
    char* user_prompt = nullptr;

    void load_chat_template(char*& buffer, const char* filename);
};
}  // namespace toyinfer
