#pragma once

#include <cstdint>
#include <vector>

#include "config.h"
#include "options.h"
#include "profiling.h"
#include "qwen3.h"

namespace toyinfer {

class Transformer {
    struct State {
        struct PrefillState {
            uint32_t capacity = 0;
            uint32_t* token_ids_d = nullptr;
            bf16* hidden_d = nullptr;
            bf16* residual_d = nullptr;
            bf16* x_d = nullptr;
            float* sum_d = nullptr;
            bf16* q_d = nullptr;
            bf16* key_d = nullptr;
            bf16* val_d = nullptr;
            bf16* o_d = nullptr;
            bf16* gate_d = nullptr;
            bf16* up_d = nullptr;
            bf16* intermedia_d = nullptr;

            void alloc(const LLMConfig& llmconfig, uint32_t num_tokens);
            void free();
        };

        struct TimingEvents {
            bool enabled = false;

            cudaEvent_t prefill_total_start = nullptr;
            cudaEvent_t prefill_total_end = nullptr;
            std::vector<cudaEvent_t> prefill_attn_start;
            std::vector<cudaEvent_t> prefill_attn_end;
            std::vector<cudaEvent_t> prefill_mlp_start;
            std::vector<cudaEvent_t> prefill_mlp_end;

            cudaEvent_t decode_total_start = nullptr;
            cudaEvent_t decode_total_end = nullptr;
            std::vector<cudaEvent_t> decode_qkv_start;
            std::vector<cudaEvent_t> decode_qkv_end;
            std::vector<cudaEvent_t> decode_attention_start;
            std::vector<cudaEvent_t> decode_attention_end;
            std::vector<cudaEvent_t> decode_mlp_start;
            std::vector<cudaEvent_t> decode_mlp_end;

            void alloc(uint32_t layers, bool enable_timing);
            void free();
        };

        uint32_t* pos_h;  // dim = 1
        uint32_t* pos_d;  // dim = 1
        bf16* hidden_d;   // hidden state, [hidden_size]
        bf16* residual_d;
        bf16* x_d;

        float* sum_d;       // dim = 1
        float* inv_freq_d;  // rope kernel, [head_dim / 2]

        // for attention
        bf16* q_d;
        bf16* key_d;
        bf16* val_d;
        bf16* key_cache_d;
        bf16* val_cache_d;
        float* score;  // [num_atten_heads, MAX_SEQ_LEN]
        float* o_buffer_d;
        bf16* o_d;

        // mlp
        bf16* gate_d;
        bf16* up_d;
        bf16* intermedia_d;
        // logits
        float* logits_d;

        // stream and sync
        cudaStream_t stream_d[3];
        cudaEvent_t event_d[3];
        cudaGraph_t graph_d = nullptr;
        cudaGraphExec_t graph_exec_d = nullptr;
        PrefillState prefill;
        TimingEvents timing;

        void alloc(const Options& options, const LLMConfig& llmconfig);
        void free();
    };

   private:
    const LLMConfig& llmconfig;
    const Options& options;
    Qwen3 qwen3_;
    State state;
    float* logits_h = nullptr;
    TransformerProfileStats profile_stats_;

    void run_decode_body(bool record_timing_events);
    void accumulate_prefill_profile();
    void accumulate_decode_profile();

   public:
    Transformer(const Options& options, const LLMConfig& config);
    ~Transformer();
    const float* prefill(const uint32_t* token_ids, uint32_t token_cnt);
    const float* forward(uint32_t token_id, uint32_t pos);
    void reset_profile();
    const TransformerProfileStats& profile_stats() const;
};

}  // namespace toyinfer
