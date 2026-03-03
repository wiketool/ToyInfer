#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <unordered_map>

#include "config.h"
#include "options.h"

namespace toyinfer {
class MergeRank {
   public:
    MergeRank(const std::filesystem::path model_dir);
    ~MergeRank();
    int find_merge_rank(const uint32_t token_a_id,
                        const uint32_t token_b_idconst, uint32_t& merge_rank);
    int find_merge_token_id(const uint32_t token_a_id,
                            const uint32_t token_b_idconst,
                            uint32_t& merged_token_id);

   private:
    uint32_t merges_len;
    std::unordered_map<uint64_t, std::pair<uint32_t, uint32_t>> merge_map;
};

class Tokenizer {
   public:
    Tokenizer(const Options& options, const LLMConfig& llmConfig);
    ~Tokenizer();

    void encode(const char* text, std::unique_ptr<uint32_t[]>& token_ids,
                uint32_t& token_count);
    void decode(const uint32_t token_ids);
    void render_prompt(std::unique_ptr<char[]>& prompt, const char* user_prompt,
                       const char* system_prompt);

   private:
    struct Token {
        uint32_t id;
        char* text;
    }* token_index_ = nullptr;  // 词表索引，大小为 vocab_size_

    std::filesystem::path model_dir;
    MergeRank merge_rank;
    bool thinking_ = false;
    uint32_t vocab_size_ = 0;
    uint32_t bos_token_id_ = 0;
    uint32_t eos_token_id_ = 0;
    uint32_t max_token_len = 0;
    char** vocab_ = nullptr;  // 词表，大小为 vocab_size_
    char* system_prompt_template = nullptr;
    char* user_prompt_template = nullptr;

    void load_chat_template(char*& buffer, const char* filename);
    int32_t find_token(const char* text, uint32_t& token_id);
};
}  // namespace toyinfer
