#pragma once

#include "options.h"

namespace toyinfer {

class LLMConfig {
   public:
    // 基础形状
    int vocab_size;
    int hidden_size;
    int intermediate_size;
    int num_hidden_layers;

    // Attention 相关
    int num_attention_heads;
    int num_key_value_heads;
    int head_dim;

    // 生成控制与处理
    int bos_token_id;
    int eos_token_id;
    int max_seq_len;  // 对应 max_position_embeddings

    // 算法超参
    float rms_norm_eps;
    float rope_theta;

    LLMConfig(const Options& options);
};
}  // namespace toyinfer