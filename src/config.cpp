#pragma once

#include "config.h"

#include <fstream>
#include <stdexcept>

#include "json/json.h"

namespace toyinfer {

LLMConfig::LLMConfig(const Options& options) {
    // 从 JSON 文件加载配置
    std::string config_path = std::string(options.model_dir) + "/config.json";
    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errs;
    std::ifstream config_file(config_path);
    if (config_file.is_open() == false) {
        throw std::runtime_error("无法打开配置文件: " + config_path);
    }
    if (Json::parseFromStream(builder, config_file, &root, &errs) == false) {
        throw std::runtime_error("解析配置文件失败: " + errs);
    }

    // 从 JSON 中提取参数
    vocab_size = root["vocab_size"].asInt();
    hidden_size = root["hidden_size"].asInt();
    intermediate_size = root["intermediate_size"].asInt();
    num_hidden_layers = root["num_hidden_layers"].asInt();
    num_attention_heads = root["num_attention_heads"].asInt();
    num_key_value_heads = root["num_key_value_heads"].asInt();
    head_dim = root["head_dim"].asInt();
    bos_token_id = root["bos_token_id"].asInt();
    eos_token_id = root["eos_token_id"].asInt();
    max_seq_len = root["max_position_embeddings"].asInt();
    rms_norm_eps = root["rms_norm_eps"].asFloat();
    rope_theta = root["rope_theta"].asFloat();

    if ((head_dim & 1) != 0) {
        throw std::runtime_error("HEAD_DIM 必须是偶数");
    }
};
};  // namespace toyinfer
