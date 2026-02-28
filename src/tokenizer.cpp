#pragma once

#include "tokenizer.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <stdexcept>

#include "config.h"
#include "options.h"
#define DEFAULT_PROMPT_SIZE 1024

namespace toyinfer {

Tokenizer::Tokenizer(const Options& options, const LLMConfig& llmConfig)
    : model_dir(options.model_dir),
      thinking_(options.enable_thinking),
      vocab_size_(llmConfig.vocab_size),
      bos_token_id_(llmConfig.bos_token_id),
      eos_token_id_(llmConfig.eos_token_id),
      system_prompt(nullptr),
      user_prompt(nullptr) {
    // Load tokenizer from model_dir
    std::filesystem::path tokenizer_path = model_dir / "tokenizer.bin";
    std::ifstream tokenizer_file{tokenizer_path, std::ios::binary};
    if (tokenizer_file.is_open() == false) {
        throw std::runtime_error("Failed to open tokenizer file: " +
                                 tokenizer_path.string());
    }

    // 将其加载进入缓冲区
    tokenizer_file.seekg(0, std::ios::end);
    std::streamsize file_size = tokenizer_file.tellg();
    tokenizer_file.seekg(0, std::ios::beg);
    char* buffer = new char[file_size];
    tokenizer_file.read(buffer, file_size);
    tokenizer_file.close();

    // 获取tokenizer_file的token个数，检查是否小于model_config.json中的值
    uint32_t token_number;
    std::memcpy(&token_number, buffer, sizeof(uint32_t));

    if (token_number > vocab_size_) {
        throw std::runtime_error("Illegal tokenizer file: " +
                                 tokenizer_path.string());
    }

    vocab_ = new char*[vocab_size_];
    merge_score_ = new float[vocab_size_];
    uint32_t offset = sizeof(uint32_t);

    // 需要注意的是vocab_size_和实际词表大小可能不一致
    // 原因：为了提升推理和训练速度，大模型通常会将原始词表padding到最近的
    // 128 整数倍。
    for (uint32_t i = 0; i < vocab_size_; i++) {
        if (offset < file_size) {
            memcpy(merge_score_ + i, buffer + offset, sizeof(float));
            offset += sizeof(float);
            uint32_t token_len = 0;
            memcpy(&token_len, buffer + offset, sizeof(uint32_t));
            offset += sizeof(uint32_t);
            vocab_[i] = new char[token_len + 1];
            vocab_[i][token_len] = '\0';
            memcpy(vocab_[i], buffer + offset, token_len);
            offset += token_len;
        } else {
            vocab_[i] = new char[1];
            vocab_[i][0] = '\0';
        }
    }

    delete[] buffer;

    // load chat template
    if (thinking_) {
        load_chat_template(system_prompt, "template_system_thinking.txt");
        load_chat_template(user_prompt, "template_user_thinking.txt");
    } else {
        load_chat_template(system_prompt, "template_system.txt");
        load_chat_template(user_prompt, "template_user.txt");
    }
};

Tokenizer::~Tokenizer() {
    if (vocab_) {
        for (uint32_t i = 0; i < vocab_size_; ++i) {
            if (vocab_[i]) {
                delete[] vocab_[i];
            }
        }
        delete[] vocab_;
    }
    if (merge_score_) {
        delete[] merge_score_;
    }
    if (system_prompt) {
        delete[] system_prompt;
    }
    if (user_prompt) {
        delete[] user_prompt;
    }
};

void Tokenizer::load_chat_template(char*& buffer, const char* filename) {
    std::filesystem::path template_path = model_dir / filename;
    std::ifstream template_file{template_path};
    if (template_file.is_open() == false) {
        throw std::runtime_error("Open chat template file fail. File: " +
                                 std::string(filename));
    }
    template_file.seekg(0, std::ios::end);
    std::streamsize file_size = template_file.tellg();
    template_file.seekg(0, std::ios::beg);

    buffer = new char[file_size + 1];
    buffer[file_size] = '\0';
    template_file.read(buffer, file_size);
}
}  // namespace toyinfer
