#include "transformer.h"

#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

#include "config.h"
#include "cuda_utils.h"
#include "kernel_warpper.h"
#include "options.h"
#include "qwen3.h"
#include "type.h"

namespace toyinfer {
namespace {

void launch_prefill_flash_attention(const bf16* __restrict__ q_d,
                                    const bf16* __restrict__ key_d,
                                    const bf16* __restrict__ val_d,
                                    bf16* __restrict__ o_d,
                                    const LLMConfig& llmconfig,
                                    uint32_t token_cnt) {
    if (llmconfig.head_dim == 64) {
        flash_attention_v1_bf16<32, 4, 64>(
            q_d, key_d, val_d, o_d, llmconfig.num_attention_heads,
            llmconfig.num_key_value_heads, llmconfig.head_dim, token_cnt);
        return;
    }
    if (llmconfig.head_dim == 128) {
        flash_attention_v1_bf16<32, 4, 128>(
            q_d, key_d, val_d, o_d, llmconfig.num_attention_heads,
            llmconfig.num_key_value_heads, llmconfig.head_dim, token_cnt);
        return;
    }
    assert(false && "unsupported head_dim for prefill flash attention");
}

void create_timing_event(cudaEvent_t& event) {
    CHECK_CUDA(cudaEventCreate(&event));
}

void destroy_timing_event(cudaEvent_t& event) {
    if (event != nullptr) {
        CHECK_CUDA(cudaEventDestroy(event));
        event = nullptr;
    }
}

void create_timing_vector(std::vector<cudaEvent_t>& events, uint32_t count) {
    events.assign(count, nullptr);
    for (auto& event : events) {
        create_timing_event(event);
    }
}

void destroy_timing_vector(std::vector<cudaEvent_t>& events) {
    for (auto& event : events) {
        destroy_timing_event(event);
    }
    events.clear();
}

double cuda_elapsed_ms(cudaEvent_t start, cudaEvent_t end) {
    float elapsed = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, end));
    return static_cast<double>(elapsed);
}

}  // namespace

void Transformer::State::TimingEvents::alloc(uint32_t layers,
                                             bool enable_timing) {
    free();
    enabled = enable_timing;
    if (!enabled) {
        return;
    }

    create_timing_event(prefill_total_start);
    create_timing_event(prefill_total_end);
    create_timing_vector(prefill_attn_start, layers);
    create_timing_vector(prefill_attn_end, layers);
    create_timing_vector(prefill_mlp_start, layers);
    create_timing_vector(prefill_mlp_end, layers);

    create_timing_event(decode_total_start);
    create_timing_event(decode_total_end);
    create_timing_vector(decode_qkv_start, layers);
    create_timing_vector(decode_qkv_end, layers);
    create_timing_vector(decode_attention_start, layers);
    create_timing_vector(decode_attention_end, layers);
    create_timing_vector(decode_mlp_start, layers);
    create_timing_vector(decode_mlp_end, layers);
}

void Transformer::State::TimingEvents::free() {
    destroy_timing_event(prefill_total_start);
    destroy_timing_event(prefill_total_end);
    destroy_timing_vector(prefill_attn_start);
    destroy_timing_vector(prefill_attn_end);
    destroy_timing_vector(prefill_mlp_start);
    destroy_timing_vector(prefill_mlp_end);

    destroy_timing_event(decode_total_start);
    destroy_timing_event(decode_total_end);
    destroy_timing_vector(decode_qkv_start);
    destroy_timing_vector(decode_qkv_end);
    destroy_timing_vector(decode_attention_start);
    destroy_timing_vector(decode_attention_end);
    destroy_timing_vector(decode_mlp_start);
    destroy_timing_vector(decode_mlp_end);
    enabled = false;
}

void Transformer::State::PrefillState::alloc(const LLMConfig& llmconfig,
                                             uint32_t num_tokens) {
    if (num_tokens <= capacity) {
        return;
    }
    free();

    const uint32_t q_dim = llmconfig.head_dim * llmconfig.num_attention_heads;
    const uint32_t kv_dim = llmconfig.head_dim * llmconfig.num_key_value_heads;

    cudaMalloc(&token_ids_d, sizeof(uint32_t) * num_tokens);
    cudaMalloc(&hidden_d, sizeof(bf16) * num_tokens * llmconfig.hidden_size);
    cudaMalloc(&residual_d, sizeof(bf16) * num_tokens * llmconfig.hidden_size);
    cudaMalloc(&x_d, sizeof(bf16) * num_tokens * llmconfig.hidden_size);
    cudaMalloc(&sum_d, sizeof(float) * num_tokens);
    cudaMalloc(&q_d, sizeof(bf16) * num_tokens * q_dim);
    cudaMalloc(&key_d, sizeof(bf16) * num_tokens * kv_dim);
    cudaMalloc(&val_d, sizeof(bf16) * num_tokens * kv_dim);
    cudaMalloc(&o_d, sizeof(bf16) * num_tokens * q_dim);
    cudaMalloc(&gate_d,
               sizeof(bf16) * num_tokens * llmconfig.intermediate_size);
    cudaMalloc(&up_d, sizeof(bf16) * num_tokens * llmconfig.intermediate_size);
    cudaMalloc(&intermedia_d,
               sizeof(bf16) * num_tokens * llmconfig.intermediate_size);
    capacity = num_tokens;
}

void Transformer::State::PrefillState::free() {
    cudaFree(token_ids_d);
    cudaFree(hidden_d);
    cudaFree(residual_d);
    cudaFree(x_d);
    cudaFree(sum_d);
    cudaFree(q_d);
    cudaFree(key_d);
    cudaFree(val_d);
    cudaFree(o_d);
    cudaFree(gate_d);
    cudaFree(up_d);
    cudaFree(intermedia_d);
    token_ids_d = nullptr;
    hidden_d = nullptr;
    residual_d = nullptr;
    x_d = nullptr;
    sum_d = nullptr;
    q_d = nullptr;
    key_d = nullptr;
    val_d = nullptr;
    o_d = nullptr;
    gate_d = nullptr;
    up_d = nullptr;
    intermedia_d = nullptr;
    capacity = 0;
}

void Transformer::State::alloc(const Options& options,
                               const LLMConfig& llmconfig) {
    cudaMallocHost(&pos_h, sizeof(uint32_t));
    cudaMalloc(&pos_d, sizeof(uint32_t));
    cudaMalloc(&hidden_d, sizeof(bf16) * llmconfig.hidden_size);
    cudaMalloc(&residual_d, sizeof(bf16) * llmconfig.hidden_size);
    cudaMalloc(&x_d, sizeof(bf16) * llmconfig.hidden_size);
    cudaMalloc(&sum_d, sizeof(float));
    cudaMalloc(&inv_freq_d, sizeof(float) * llmconfig.head_dim / 2);
    cudaMalloc(&q_d, sizeof(bf16) * llmconfig.head_dim *
                         llmconfig.num_attention_heads);
    cudaMalloc(&key_d, sizeof(bf16) * llmconfig.head_dim *
                           llmconfig.num_key_value_heads);
    cudaMalloc(&val_d, sizeof(bf16) * llmconfig.head_dim *
                           llmconfig.num_key_value_heads);
    cudaMalloc(&key_cache_d, sizeof(bf16) * llmconfig.num_hidden_layers *
                                 options.max_seq_len *
                                 llmconfig.num_key_value_heads *
                                 llmconfig.head_dim);
    cudaMalloc(&val_cache_d, sizeof(bf16) * llmconfig.num_hidden_layers *
                                 options.max_seq_len *
                                 llmconfig.num_key_value_heads *
                                 llmconfig.head_dim);
    cudaMalloc(&score, sizeof(float) * llmconfig.num_attention_heads *
                           options.max_seq_len);
    cudaMalloc(&o_buffer_d, sizeof(float) * llmconfig.num_attention_heads *
                                llmconfig.head_dim);
    cudaMalloc(&o_d, sizeof(bf16) * llmconfig.num_attention_heads *
                         llmconfig.head_dim);
    cudaMalloc(&gate_d, sizeof(bf16) * llmconfig.intermediate_size);
    cudaMalloc(&up_d, sizeof(bf16) * llmconfig.intermediate_size);
    cudaMalloc(&intermedia_d, sizeof(bf16) * llmconfig.intermediate_size);
    cudaMalloc(&logits_d, sizeof(float) * llmconfig.vocab_size);

    for (uint32_t i = 0; i < 3; i++) {
        CHECK_CUDA(cudaStreamCreate(&stream_d[i]));
        CHECK_CUDA(cudaEventCreateWithFlags(&event_d[i], cudaEventDisableTiming));
    }

    timing.alloc(llmconfig.num_hidden_layers, options.detail_time);

    precompute_freq_f32(inv_freq_d, llmconfig.head_dim, llmconfig.rope_theta);
    cudaDeviceSynchronize();
}

void Transformer::State::free() {
    if (graph_exec_d != nullptr) {
        CHECK_CUDA(cudaGraphExecDestroy(graph_exec_d));
        graph_exec_d = nullptr;
    }
    if (graph_d != nullptr) {
        CHECK_CUDA(cudaGraphDestroy(graph_d));
        graph_d = nullptr;
    }
    timing.free();

    cudaFreeHost(pos_h);
    cudaFree(pos_d);
    cudaFree(hidden_d);
    cudaFree(residual_d);
    cudaFree(x_d);
    cudaFree(sum_d);
    cudaFree(inv_freq_d);
    cudaFree(q_d);
    cudaFree(key_d);
    cudaFree(val_d);
    cudaFree(key_cache_d);
    cudaFree(val_cache_d);
    cudaFree(score);
    cudaFree(o_buffer_d);
    cudaFree(o_d);
    cudaFree(gate_d);
    cudaFree(up_d);
    cudaFree(intermedia_d);
    cudaFree(logits_d);
    prefill.free();
    for (uint32_t i = 0; i < 3; i++) {
        if (stream_d[i] != nullptr) {
            CHECK_CUDA(cudaStreamDestroy(stream_d[i]));
        }
        if (event_d[i] != nullptr) {
            CHECK_CUDA(cudaEventDestroy(event_d[i]));
        }
    }
}

Transformer::Transformer(const Options& options, const LLMConfig& config)
    : llmconfig(config),
      options(options),
      qwen3_(options, config),
      logits_h(nullptr),
      profile_stats_{} {
    qwen3_.load_weights();
    state.alloc(options, config);
    cudaMallocHost(&logits_h, sizeof(float) * llmconfig.vocab_size);
}

Transformer::~Transformer() {
    state.free();
    if (logits_h != nullptr) {
        cudaFreeHost(logits_h);
    }
}

void Transformer::reset_profile() { profile_stats_.reset(); }

const TransformerProfileStats& Transformer::profile_stats() const {
    return profile_stats_;
}

void Transformer::accumulate_prefill_profile() {
    if (!state.timing.enabled) {
        return;
    }
    profile_stats_.prefill_total_ms +=
        cuda_elapsed_ms(state.timing.prefill_total_start,
                        state.timing.prefill_total_end);
    for (size_t i = 0; i < state.timing.prefill_attn_start.size(); ++i) {
        profile_stats_.prefill_layer_attn_block_ms +=
            cuda_elapsed_ms(state.timing.prefill_attn_start[i],
                            state.timing.prefill_attn_end[i]);
        profile_stats_.prefill_layer_mlp_block_ms +=
            cuda_elapsed_ms(state.timing.prefill_mlp_start[i],
                            state.timing.prefill_mlp_end[i]);
    }
}

void Transformer::accumulate_decode_profile() {
    if (!state.timing.enabled) {
        return;
    }
    profile_stats_.decode_forward_total_ms +=
        cuda_elapsed_ms(state.timing.decode_total_start,
                        state.timing.decode_total_end);
    for (size_t i = 0; i < state.timing.decode_qkv_start.size(); ++i) {
        profile_stats_.decode_layer_qkv_and_cache_ms +=
            cuda_elapsed_ms(state.timing.decode_qkv_start[i],
                            state.timing.decode_qkv_end[i]);
        profile_stats_.decode_layer_attention_ms +=
            cuda_elapsed_ms(state.timing.decode_attention_start[i],
                            state.timing.decode_attention_end[i]);
        profile_stats_.decode_layer_mlp_ms +=
            cuda_elapsed_ms(state.timing.decode_mlp_start[i],
                            state.timing.decode_mlp_end[i]);
    }
}

const float* Transformer::prefill(const uint32_t* token_ids,
                                  uint32_t token_cnt) {
    ScopedNvtxRange prefill_range("transformer.prefill");

    assert(token_cnt > 0);
    assert(token_cnt <= static_cast<uint32_t>(options.max_seq_len));

    const uint32_t kv_dim = llmconfig.head_dim * llmconfig.num_key_value_heads;
    const uint32_t q_dim = llmconfig.head_dim * llmconfig.num_attention_heads;
    const cudaStream_t prefill_stream = nullptr;
    state.prefill.alloc(llmconfig, token_cnt);

    cudaMemcpy(state.prefill.token_ids_d, token_ids,
               sizeof(uint32_t) * token_cnt, cudaMemcpyHostToDevice);
    if (state.timing.enabled) {
        CHECK_CUDA(
            cudaEventRecord(state.timing.prefill_total_start, prefill_stream));
    }
    gather_embedding_bf16<NUM_THREADS>(
        qwen3_.embed_tokens_d, state.prefill.token_ids_d,
        state.prefill.hidden_d, token_cnt, llmconfig.hidden_size);

    for (int i = 0; i < llmconfig.num_hidden_layers; ++i) {
        ScopedNvtxRange layer_range("transformer.prefill.layer." +
                                    std::to_string(i));
        const Qwen3::Layer& layer_ref = qwen3_.layer[i];
        bf16* layer_key_cache =
            state.key_cache_d + (i * options.max_seq_len * kv_dim);
        bf16* layer_val_cache =
            state.val_cache_d + (i * options.max_seq_len * kv_dim);

        if (state.timing.enabled) {
            CHECK_CUDA(cudaEventRecord(state.timing.prefill_attn_start[i],
                                       prefill_stream));
        }
        {
            ScopedNvtxRange attn_range("transformer.prefill.layer.attn_block." +
                                       std::to_string(i));
            batch_rmsnorm_bf16<NUM_THREADS>(
                state.prefill.hidden_d, layer_ref.input_layernorm_d,
                state.prefill.residual_d, state.prefill.sum_d,
                llmconfig.rms_norm_eps, token_cnt, llmconfig.hidden_size);

            batch_gemv_proj_bf16<NUM_THREADS>(
                layer_ref.attention.q_proj_d, state.prefill.residual_d,
                state.prefill.q_d, token_cnt, q_dim, llmconfig.hidden_size);
            batch_gemv_proj_bf16<NUM_THREADS>(
                layer_ref.attention.k_proj_d, state.prefill.residual_d,
                state.prefill.key_d, token_cnt, kv_dim, llmconfig.hidden_size);
            batch_gemv_proj_bf16<NUM_THREADS>(
                layer_ref.attention.v_proj_d, state.prefill.residual_d,
                state.prefill.val_d, token_cnt, kv_dim, llmconfig.hidden_size);

            batch_multi_rmsnorm_bf16(
                state.prefill.q_d, layer_ref.attention.q_norm_d,
                state.prefill.q_d, llmconfig.rms_norm_eps, token_cnt,
                llmconfig.num_attention_heads, llmconfig.head_dim);
            batch_multi_rmsnorm_bf16(
                state.prefill.key_d, layer_ref.attention.k_norm_d,
                state.prefill.key_d, llmconfig.rms_norm_eps, token_cnt,
                llmconfig.num_key_value_heads, llmconfig.head_dim);
            batch_rope_bf16(state.prefill.q_d, state.inv_freq_d, token_cnt,
                            llmconfig.num_attention_heads, llmconfig.head_dim);
            batch_rope_bf16(state.prefill.key_d, state.inv_freq_d, token_cnt,
                            llmconfig.num_key_value_heads, llmconfig.head_dim);

            cudaMemcpy(layer_key_cache, state.prefill.key_d,
                       sizeof(bf16) * token_cnt * kv_dim,
                       cudaMemcpyDeviceToDevice);
            cudaMemcpy(layer_val_cache, state.prefill.val_d,
                       sizeof(bf16) * token_cnt * kv_dim,
                       cudaMemcpyDeviceToDevice);

            launch_prefill_flash_attention(state.prefill.q_d,
                                           state.prefill.key_d,
                                           state.prefill.val_d,
                                           state.prefill.o_d, llmconfig,
                                           token_cnt);

            batch_gemv_proj_bf16<NUM_THREADS>(
                layer_ref.attention.o_proj_d, state.prefill.o_d,
                state.prefill.residual_d, token_cnt, llmconfig.hidden_size,
                q_dim);
            batch_residual_add_bf16<NUM_THREADS>(state.prefill.residual_d,
                                                 state.prefill.hidden_d,
                                                 token_cnt,
                                                 llmconfig.hidden_size);
        }
        if (state.timing.enabled) {
            CHECK_CUDA(cudaEventRecord(state.timing.prefill_attn_end[i],
                                       prefill_stream));
        }

        cudaMemcpy(state.prefill.residual_d, state.prefill.hidden_d,
                   sizeof(bf16) * token_cnt * llmconfig.hidden_size,
                   cudaMemcpyDeviceToDevice);
        if (state.timing.enabled) {
            CHECK_CUDA(cudaEventRecord(state.timing.prefill_mlp_start[i],
                                       prefill_stream));
        }
        {
            ScopedNvtxRange mlp_range("transformer.prefill.layer.mlp_block." +
                                      std::to_string(i));
            batch_rmsnorm_bf16<NUM_THREADS>(
                state.prefill.hidden_d, layer_ref.post_attention_layernorm_d,
                state.prefill.x_d, state.prefill.sum_d,
                llmconfig.rms_norm_eps, token_cnt, llmconfig.hidden_size);
            batch_gemv_proj_bf16<NUM_THREADS>(
                layer_ref.ffn.gate_proj_d, state.prefill.x_d,
                state.prefill.gate_d, token_cnt,
                llmconfig.intermediate_size, llmconfig.hidden_size);
            batch_gemv_proj_bf16<NUM_THREADS>(
                layer_ref.ffn.up_proj_d, state.prefill.x_d, state.prefill.up_d,
                token_cnt, llmconfig.intermediate_size, llmconfig.hidden_size);
            batch_swiglu_bf16x2<NUM_THREADS>(
                state.prefill.gate_d, state.prefill.up_d,
                state.prefill.intermedia_d, token_cnt,
                llmconfig.intermediate_size);
            batch_gemv_proj_bf16<NUM_THREADS>(
                layer_ref.ffn.down_proj_d, state.prefill.intermedia_d,
                state.prefill.hidden_d, token_cnt, llmconfig.hidden_size,
                llmconfig.intermediate_size);
            batch_residual_add_bf16<NUM_THREADS>(state.prefill.residual_d,
                                                 state.prefill.hidden_d,
                                                 token_cnt,
                                                 llmconfig.hidden_size);
        }
        if (state.timing.enabled) {
            CHECK_CUDA(cudaEventRecord(state.timing.prefill_mlp_end[i],
                                       prefill_stream));
        }
    }

    batch_rmsnorm_bf16<NUM_THREADS>(state.prefill.hidden_d, qwen3_.norm_d,
                                    state.prefill.x_d, state.prefill.sum_d,
                                    llmconfig.rms_norm_eps, token_cnt,
                                    llmconfig.hidden_size);
    cudaMemcpy(state.x_d,
               state.prefill.x_d + (token_cnt - 1) * static_cast<uint32_t>(
                                                         llmconfig.hidden_size),
               sizeof(bf16) * llmconfig.hidden_size, cudaMemcpyDeviceToDevice);
    gemv_proj_bf162float<NUM_THREADS>(qwen3_.lmhead_d, state.x_d,
                                      state.logits_d, llmconfig.vocab_size,
                                      llmconfig.hidden_size);
    if (state.timing.enabled) {
        CHECK_CUDA(
            cudaEventRecord(state.timing.prefill_total_end, prefill_stream));
    }
    cudaMemcpy(logits_h, state.logits_d, sizeof(float) * llmconfig.vocab_size,
               cudaMemcpyDeviceToHost);
    if (state.timing.enabled) {
        accumulate_prefill_profile();
    }
    return logits_h;
}

void Transformer::run_decode_body() {
    const uint32_t kv_dim = llmconfig.head_dim * llmconfig.num_key_value_heads;
    const uint32_t q_dim = llmconfig.head_dim * llmconfig.num_attention_heads;
    const bool use_multi_stream = options.use_multi_stream;
    cudaStream_t stream0 = state.stream_d[0];
    cudaStream_t stream1 = use_multi_stream ? state.stream_d[1] : stream0;
    cudaStream_t stream2 = use_multi_stream ? state.stream_d[2] : stream0;

    if (state.timing.enabled) {
        CHECK_CUDA(cudaEventRecord(state.timing.decode_total_start, stream0));
    }
    CHECK_CUDA(cudaMemcpyAsync(state.pos_d, state.pos_h, sizeof(uint32_t),
                               cudaMemcpyHostToDevice, stream0));

    for (int i = 0; i < llmconfig.num_hidden_layers; ++i) {
        ScopedNvtxRange layer_range("transformer.decode.layer." +
                                    std::to_string(i));
        const Qwen3::Layer& layer_ref = qwen3_.layer[i];
        bf16* layer_key_cache =
            state.key_cache_d + (i * options.max_seq_len * kv_dim);
        bf16* layer_val_cache =
            state.val_cache_d + (i * options.max_seq_len * kv_dim);

        if (state.timing.enabled) {
            CHECK_CUDA(
                cudaEventRecord(state.timing.decode_qkv_start[i], stream0));
        }
        {
            ScopedNvtxRange qkv_range(
                "transformer.decode.layer.qkv_and_cache." + std::to_string(i));
            rmsnorm_bf16<NUM_THREADS>(
                state.hidden_d, layer_ref.input_layernorm_d, state.residual_d,
                state.sum_d, llmconfig.rms_norm_eps, llmconfig.hidden_size,
                stream0);
            if (use_multi_stream) {
                CHECK_CUDA(cudaEventRecord(state.event_d[0], stream0));
            }

            gemv_proj_bf16<NUM_THREADS>(layer_ref.attention.q_proj_d,
                                        state.residual_d, state.q_d, q_dim,
                                        llmconfig.hidden_size, stream0);
            if (use_multi_stream) {
                CHECK_CUDA(cudaStreamWaitEvent(stream1, state.event_d[0]));
                CHECK_CUDA(cudaStreamWaitEvent(stream2, state.event_d[0]));
            }
            gemv_proj_bf16<NUM_THREADS>(layer_ref.attention.k_proj_d,
                                        state.residual_d, state.key_d, kv_dim,
                                        llmconfig.hidden_size, stream1);
            gemv_proj_bf16<NUM_THREADS>(layer_ref.attention.v_proj_d,
                                        state.residual_d, state.val_d, kv_dim,
                                        llmconfig.hidden_size, stream2);

            multi_rmsnorm_bf16(state.q_d, layer_ref.attention.q_norm_d,
                               state.q_d, llmconfig.rms_norm_eps,
                               llmconfig.num_attention_heads,
                               llmconfig.head_dim, stream0);
            multi_rmsnorm_bf16(state.key_d, layer_ref.attention.k_norm_d,
                               state.key_d, llmconfig.rms_norm_eps,
                               llmconfig.num_key_value_heads,
                               llmconfig.head_dim, stream1);
            rope_bf16_graph(state.q_d, state.inv_freq_d, state.pos_d,
                            llmconfig.num_attention_heads, llmconfig.head_dim,
                            stream0);
            rope_bf16_graph(state.key_d, state.inv_freq_d, state.pos_d,
                            llmconfig.num_key_value_heads, llmconfig.head_dim,
                            stream1);
            write_kv_cache_bf16<NUM_THREADS>(state.key_d, layer_key_cache,
                                             state.pos_d, kv_dim, stream1);
            write_kv_cache_bf16<NUM_THREADS>(state.val_d, layer_val_cache,
                                             state.pos_d, kv_dim, stream2);

            if (use_multi_stream) {
                CHECK_CUDA(cudaEventRecord(state.event_d[1], stream1));
                CHECK_CUDA(cudaEventRecord(state.event_d[2], stream2));
                CHECK_CUDA(cudaStreamWaitEvent(stream0, state.event_d[1]));
                CHECK_CUDA(cudaStreamWaitEvent(stream0, state.event_d[2]));
            }
        }
        if (state.timing.enabled) {
            CHECK_CUDA(cudaEventRecord(state.timing.decode_qkv_end[i], stream0));
            CHECK_CUDA(cudaEventRecord(state.timing.decode_attention_start[i],
                                       stream0));
        }
        {
            ScopedNvtxRange attn_range("transformer.decode.layer.attention." +
                                       std::to_string(i));
            attention_bf16_graph<NUM_THREADS, TILE_SEQ>(
                state.q_d, layer_key_cache, layer_val_cache, state.score,
                state.o_buffer_d, state.o_d, llmconfig.num_attention_heads,
                llmconfig.num_key_value_heads, llmconfig.head_dim, state.pos_d,
                options.max_seq_len, stream0);
            gemv_proj_bf16<NUM_THREADS>(
                layer_ref.attention.o_proj_d, state.o_d, state.residual_d,
                llmconfig.hidden_size,
                llmconfig.num_attention_heads * llmconfig.head_dim, stream0);
            residual_add_bf16<NUM_THREADS>(state.residual_d, state.hidden_d,
                                           llmconfig.hidden_size, stream0);
            CHECK_CUDA(cudaMemcpyAsync(state.residual_d, state.hidden_d,
                                       sizeof(bf16) * llmconfig.hidden_size,
                                       cudaMemcpyDeviceToDevice, stream0));
            rmsnorm_bf16<NUM_THREADS>(
                state.hidden_d, layer_ref.post_attention_layernorm_d, state.x_d,
                state.sum_d, llmconfig.rms_norm_eps, llmconfig.hidden_size,
                stream0);
        }
        if (state.timing.enabled) {
            CHECK_CUDA(cudaEventRecord(state.timing.decode_attention_end[i],
                                       stream0));
            CHECK_CUDA(
                cudaEventRecord(state.timing.decode_mlp_start[i], stream0));
        }
        {
            ScopedNvtxRange mlp_range("transformer.decode.layer.mlp." +
                                      std::to_string(i));
            gemv_proj_bf16<NUM_THREADS>(
                layer_ref.ffn.gate_proj_d, state.x_d, state.gate_d,
                llmconfig.intermediate_size, llmconfig.hidden_size, stream0);
            if (use_multi_stream) {
                CHECK_CUDA(cudaEventRecord(state.event_d[0], stream0));
                CHECK_CUDA(cudaStreamWaitEvent(stream1, state.event_d[0]));
            }
            gemv_proj_bf16<NUM_THREADS>(layer_ref.ffn.up_proj_d, state.x_d,
                                        state.up_d, llmconfig.intermediate_size,
                                        llmconfig.hidden_size, stream1);
            if (use_multi_stream) {
                CHECK_CUDA(cudaEventRecord(state.event_d[1], stream1));
                CHECK_CUDA(cudaStreamWaitEvent(stream0, state.event_d[1]));
            }

            swiglu_bf16x2<NUM_THREADS>(
                state.gate_d, state.up_d, state.intermedia_d,
                llmconfig.intermediate_size, stream0);
            gemv_proj_bf16<NUM_THREADS>(
                layer_ref.ffn.down_proj_d, state.intermedia_d, state.hidden_d,
                llmconfig.hidden_size, llmconfig.intermediate_size, stream0);
            residual_add_bf16<NUM_THREADS>(state.residual_d, state.hidden_d,
                                           llmconfig.hidden_size, stream0);
        }
        if (state.timing.enabled) {
            CHECK_CUDA(
                cudaEventRecord(state.timing.decode_mlp_end[i], stream0));
        }
    }

    rmsnorm_bf16<NUM_THREADS>(state.hidden_d, qwen3_.norm_d, state.x_d,
                              state.sum_d, llmconfig.rms_norm_eps,
                              llmconfig.hidden_size, stream0);
    gemv_proj_bf162float<NUM_THREADS>(qwen3_.lmhead_d, state.x_d,
                                      state.logits_d, llmconfig.vocab_size,
                                      llmconfig.hidden_size, stream0);
    if (state.timing.enabled) {
        CHECK_CUDA(cudaEventRecord(state.timing.decode_total_end, stream0));
    }
}

const float* Transformer::forward(uint32_t token_id, uint32_t pos) {
    ScopedNvtxRange forward_range("transformer.forward");

    const bf16* embedding_ptr =
        qwen3_.embed_tokens_d + llmconfig.hidden_size * token_id;
    cudaMemcpy(state.hidden_d, embedding_ptr,
               sizeof(bf16) * llmconfig.hidden_size, cudaMemcpyDeviceToDevice);
    *(state.pos_h) = pos;

    if (options.enable_cuda_graph) {
        if (state.graph_d == nullptr) {
            ScopedNvtxRange capture_range("transformer.forward.graph_capture");
            CHECK_CUDA(
                cudaStreamBeginCapture(state.stream_d[0],
                                       cudaStreamCaptureModeGlobal));
            run_decode_body();
            CHECK_CUDA(cudaStreamEndCapture(state.stream_d[0], &state.graph_d));
            CHECK_CUDA(
                cudaGraphInstantiate(&state.graph_exec_d, state.graph_d));
        }
        ScopedNvtxRange launch_range("transformer.forward.graph_launch");
        CHECK_CUDA(cudaGraphLaunch(state.graph_exec_d, state.stream_d[0]));
    } else {
        run_decode_body();
    }

    CHECK_CUDA(cudaStreamSynchronize(state.stream_d[0]));
    if (state.timing.enabled) {
        accumulate_decode_profile();
    }
    cudaMemcpy(logits_h, state.logits_d, sizeof(float) * llmconfig.vocab_size,
               cudaMemcpyDeviceToHost);
    return logits_h;
}

}  // namespace toyinfer
