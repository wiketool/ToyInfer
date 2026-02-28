#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <filesystem>

#include "config.h"
#include "options.h"
#include "type.h"

namespace toyinfer {

class Qwen3 {
   public:
    Qwen3(const Options& options,const LLMConfig& config);
    ~Qwen3();
    void load_weights();

   private:
    struct TensorMeta {
        const char* name;
        uint64_t offset;
        uint64_t size;
    };

    struct FFN {
        bf16* down_proj_d;
        bf16* gate_proj_d;
        bf16* up_proj_d;
    };
    struct Attention {
        bf16* k_norm_d;
        bf16* k_proj_d;
        bf16* o_proj_d;
        bf16* q_norm_d;
        bf16* q_proj_d;
        bf16* v_proj_d;
    };
    struct Layer {
        bf16* input_layernorm_d;
        bf16* post_attention_layernorm_d;
        FFN ffn;
        Attention attention;
    };
    const static TensorMeta TENSORMETA[];
    std::filesystem::path model_dir;
    uint32_t layers;

    char* weight_d;  // 指向GPU中的参数内存起始地址
    bf16* lmhead_d;
    bf16* embed_tokens_d;
    Layer* layer;
    bf16* norm_d;
};

}  // namespace toyinfer