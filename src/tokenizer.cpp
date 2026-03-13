
#include "tokenizer.h"

#include <sys/types.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <memory>
#include <queue>
#include <stdexcept>

#include "config.h"
#include "options.h"
#include "spdlog/spdlog.h"
#define DEFAULT_PROMPT_SIZE 1024

namespace toyinfer {
MergeRank::MergeRank(const std::filesystem::path model_dir) {
    std::filesystem::path merges_path = model_dir / "merges.bin";
    std::ifstream merges_file{merges_path, std::ios::binary};
    if (merges_file.is_open() == false) {
        throw std::runtime_error("Failed to open tokenizer file: " +
                                 merges_path.string());
    }
    merges_file.read((char*)&merges_len, sizeof(merges_len));
    merge_map.reserve(merges_len);
    for (uint32_t i = 0; i < merges_len; i++) {
        uint32_t token_a_id, token_b_id, token_merge_id;
        merges_file.read((char*)&token_a_id, sizeof((token_a_id)));
        merges_file.read((char*)&token_b_id, sizeof((token_b_id)));
        merges_file.read((char*)&token_merge_id, sizeof((token_merge_id)));
        uint64_t token_ab_id = (((uint64_t)token_a_id) << 32) | token_b_id;
        merge_map[token_ab_id] = {token_merge_id, i};
    }
}
MergeRank::~MergeRank() {}

int MergeRank::find_merge_rank(const uint32_t token_a_id,
                               const uint32_t token_b_id,
                               uint32_t& merge_rank) {
    uint64_t token_ab_id = (((uint64_t)token_a_id) << 32) | token_b_id;
    if (merge_map.find(token_ab_id) != merge_map.end()) {
        merge_rank = merge_map[token_ab_id].second;
        return 0;
    }
    return -1;
}

int MergeRank::find_merge_token_id(const uint32_t token_a_id,
                                   const uint32_t token_b_id,
                                   uint32_t& merged_token_id) {
    uint64_t token_ab_id = (((uint64_t)token_a_id) << 32) | token_b_id;
    if (merge_map.find(token_ab_id) != merge_map.end()) {
        merged_token_id = merge_map[token_ab_id].first;
        return 0;
    }
    return -1;
}

Tokenizer::Tokenizer(const Options& options, const LLMConfig& llmConfig)
    : model_dir(options.model_dir),
      merge_rank(model_dir),
      thinking_(options.enable_thinking),
      vocab_size_(llmConfig.vocab_size),
      bos_token_id_(llmConfig.bos_token_id),
      eos_token_id_(llmConfig.eos_token_id),
      max_token_len(0),
      system_prompt_template(nullptr),
      user_prompt_template(nullptr) {
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
    token_index_ = new Token[vocab_size_];
    uint32_t offset = sizeof(uint32_t);

    // 需要注意的是vocab_size_和实际词表大小可能不一致
    // 原因：为了提升推理和训练速度，大模型通常会将原始词表padding到最近的
    // 128 整数倍。
    for (uint32_t i = 0; i < vocab_size_; i++) {
        if (offset < file_size) {
            uint32_t token_len = 0;
            memcpy(&token_len, buffer + offset, sizeof(uint32_t));
            if (token_len > max_token_len) {
                max_token_len = token_len;
            }
            offset += sizeof(uint32_t);
            vocab_[i] = new char[token_len + 1];
            vocab_[i][token_len] = '\0';
            memcpy(vocab_[i], buffer + offset, token_len);
            offset += token_len;
        } else {
            vocab_[i] = new char[1];
            vocab_[i][0] = '\0';
        }
        token_index_[i] = {i, vocab_[i]};
    }

    delete[] buffer;
    qsort(token_index_, vocab_size_, sizeof(Token),
          [](const void* a, const void* b) {
              return strcmp(((Token*)a)->text, ((Token*)b)->text);
          });
    // load chat template
    if (thinking_) {
        load_chat_template(system_prompt_template,
                           "template_system_thinking.txt");
        load_chat_template(user_prompt_template, "template_user_thinking.txt");
    } else {
        load_chat_template(system_prompt_template, "template_system.txt");
        load_chat_template(user_prompt_template, "template_user.txt");
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
    if (system_prompt_template) {
        delete[] system_prompt_template;
    }
    if (user_prompt_template) {
        delete[] user_prompt_template;
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

void Tokenizer::encode(const char* text, std::unique_ptr<uint32_t[]>& token_ids,
                       uint32_t& token_count) {
    token_ids = std::make_unique<uint32_t[]>(strlen(text));
    token_count = 0;
    uint32_t token_id = 0;
    char* token_buffer = new char[max_token_len + 1];
    // scan text, find special token, if not found, split by character and find
    // token id in vocab
    for (const char* ch = text; *ch != '\0'; ch++) {
        bool found_token = false;
        if (*ch == '<') {
            for (uint32_t i = 0; i < max_token_len; i++) {
                const char* token_end = ch + i;
                if (*token_end == '\0') {
                    break;
                } else if (*token_end == '>') {
                    memcpy(token_buffer, ch, i + 1);
                    token_buffer[i + 1] = '\0';
                    if (find_token(token_buffer, token_id) != -1) {
                        found_token = true;
                        ch += i;
                    }
                    break;
                }
            }
        }
        if (!found_token) {
            token_buffer[0] = *ch;
            token_buffer[1] = '\0';
            if (find_token(token_buffer, token_id) == -1) {
                throw std::runtime_error("Unknown token: " +
                                         std::string(token_buffer));
            }
        }
        token_ids[token_count++] = token_id;
    }
    if (token_count == 0) {
        delete[] token_buffer;
        return;
    }

    // merge token
    struct Node {
        bool vaild;
        uint32_t token_id;
        int32_t prev;
        int32_t next;
    };
    struct MergeJob {
        uint32_t merge_rank;
        uint32_t idx;
        int32_t self_token_id;
        int32_t right_token_id;
        MergeJob(uint32_t merge_rank_, uint32_t idx_, uint32_t self_token_id_,
                 int32_t right_token_id_) {
            merge_rank = merge_rank_;
            idx = idx_;
            self_token_id = self_token_id_;
            right_token_id = right_token_id_;
        }
        bool operator<(const MergeJob& job_b) const {
            if (this->merge_rank == job_b.merge_rank) {
                return this->idx > job_b.idx;
            }
            return this->merge_rank > job_b.merge_rank;
        }
    };
    printf("token cnt: %d\n", token_count);

    Node* node = new Node[token_count];

    printf("token cnt: %d\n", token_count);

    for (int32_t i = 0; i < token_count; i++) {
        node[i].vaild = true;
        node[i].token_id = token_ids[i];
        node[i].prev = i > 0 ? i - 1 : -1;
        node[i].next = i < token_count - 1 ? i + 1 : -1;
    }
    std::priority_queue<MergeJob> merge_pq;
    for (uint32_t i = 0; i < token_count - 1; i++) {
        uint32_t rank;
        if (merge_rank.find_merge_rank(node[i].token_id, node[i + 1].token_id,
                                       rank) == -1) {
            continue;
        }
        merge_pq.emplace(rank, i, node[i].token_id, node[i + 1].token_id);
    }
    while (merge_pq.empty() == false) {
        MergeJob cur_job = merge_pq.top();
        merge_pq.pop();

        // 旧任务：当前节点被删除 || 尾部节点，A B C 任务队列{BC->D,AD,AB}
        if (node[cur_job.idx].vaild == false || node[cur_job.idx].next == -1) {
            continue;
        }
        Node* cur = node + cur_job.idx;
        Node* right = node + node[cur_job.idx].next;
        if (cur_job.right_token_id != right->token_id) {
            // 右侧节点执行了合并 A [B C] -> A [D]
            continue;
        }
        if( cur_job.self_token_id != node[cur_job.idx].token_id){
            continue;
        }
        // A C C -> [A C] C
        uint32_t merged_token_id;
        if (merge_rank.find_merge_token_id(cur->token_id, right->token_id,
                                           merged_token_id) == -1) {
            printf("%s %s\n", vocab_[cur->token_id], vocab_[right->token_id]);
            throw std::runtime_error("Illegal merge job!");
        }
        cur->token_id = merged_token_id;
        right->vaild = false;

        cur->next = right->next;
        token_count--;
        // 没有到最后
        if (right->next != -1) {
            node[right->next].prev = cur_job.idx;
        }

        // 检查左侧是否可以产生新的merge
        if (cur->prev != -1) {
            uint32_t rank;
            if (merge_rank.find_merge_rank(node[cur->prev].token_id,
                                           node[cur_job.idx].token_id,
                                           rank) == 0) {
                merge_pq.emplace(rank, cur->prev, node[cur->prev].token_id,
                                 node[cur_job.idx].token_id);
            }
        }
        // 检查右侧是否可以产生新的merge
        if (cur->next != -1) {
            uint32_t rank;
            if (merge_rank.find_merge_rank(node[cur_job.idx].token_id,
                                           node[cur->next].token_id,
                                           rank) == 0) {
                merge_pq.emplace(rank, cur_job.idx, node[cur_job.idx].token_id,
                                 node[cur->next].token_id);
            }
        }
    }
    uint32_t cur = 0;
    token_ids = std::make_unique<uint32_t[]>(token_count);
    cur = 0;
    for (uint32_t i = 0; i < token_count; i++) {
        token_ids[i] = node[cur].token_id;
        cur = node[cur].next;
    }

    delete[] token_buffer;
}

const char* Tokenizer::decode(const uint32_t token_ids) {
    if (token_ids < vocab_size_) {
        return vocab_[token_ids];
    }
    return nullptr;
}

void Tokenizer::render_prompt(std::unique_ptr<char[]>& prompt,
                              const char* user_prompt,
                              const char* system_prompt) {
    if (system_prompt != nullptr && user_prompt != nullptr) {
        // 返回的是不含\0的长度
        int prompt_len = snprintf(nullptr, 0, system_prompt_template,
                                  system_prompt, user_prompt);
        prompt = std::make_unique<char[]>(prompt_len + 1);
        sprintf(prompt.get(), system_prompt_template, system_prompt,
                user_prompt);
    } else if (user_prompt != nullptr) {
        int prompt_len =
            snprintf(nullptr, 0, user_prompt_template, user_prompt);
        prompt = std::make_unique<char[]>(prompt_len);
        sprintf(prompt.get(), user_prompt_template, user_prompt);
    } else {
        const char* err_msg = "Unable to render prompt";
        SPDLOG_ERROR(err_msg);
        throw std::runtime_error(err_msg);
    }
}

int32_t Tokenizer::find_token(const char* text, uint32_t& token_id) {
    Token* token;
    token =
        (Token*)bsearch(text, token_index_, vocab_size_, sizeof(Token),
                        [](const void* a, const void* b) {
                            return strcmp((const char*)a, ((Token*)b)->text);
                        });
    if (token) {
        token_id = token->id;
        return 0;
    }
    return -1;
}
}  // namespace toyinfer
