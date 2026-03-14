#pragma once
#include <cuda_runtime.h>

#include <cstdint>

#include "cuda_bf16.h"

namespace toyinfer {
using bf16 = __nv_bfloat16;
using bf162 = __nv_bfloat162;

template <const uint32_t NUM_THREADS>
void residual_add_bf16(const bf16* __restrict__ residual,
                       bf16* __restrict__ hidden_state, const uint32_t size,
                       cudaStream_t stream_d);

void precompute_freq_f32(float* inv_freq, int n, float base);

template <const uint32_t NUM_THREADS>
void rmsnorm_bf16(const bf16* __restrict__ input,
                  const bf16* __restrict__ weight, bf16* output, float* sum,
                  const float rms_norm_eps, const uint32_t size,
                  cudaStream_t stream_d);

// input和output是同一个地址，不能使用__restrict__
void multi_rmsnorm_bf16(const bf16* input, const bf16* __restrict__ weight,
                        bf16* output, const float rms_norm_eps,
                        const uint32_t nums_head, const uint32_t head_dim,
                        cudaStream_t stream_d);

template <const uint32_t NUM_THREADS>
void gemv_proj_bf16(const bf16* __restrict__ W,
                    const bf16* __restrict__ hidden_states,
                    bf16* __restrict__ y, const uint32_t M, const uint32_t N,
                    cudaStream_t stream_d);

template <const uint32_t NUM_THREADS>
void gemv_proj_bf162float(const bf16* __restrict__ W,
                          const bf16* __restrict__ hidden_states,
                          float* __restrict__ y, const uint32_t M,
                          const uint32_t N, cudaStream_t stream_d);

void rope_bf16(bf16* qk_ptr, const float* __restrict__ inv_freq, uint32_t pos,
               const uint32_t nums_head, const uint32_t head_dim,
               cudaStream_t stream_d);

template <const uint32_t NUM_THREADS, const uint32_t TILE_SEQ>
void attention_bf16(const bf16* __restrict__ Q, const bf16* __restrict__ Ks,
                    const bf16* __restrict__ Vs, float* __restrict__ score,
                    float* __restrict__ O_buffer, bf16* __restrict__ O,
                    const uint32_t num_q_heads, const uint32_t num_kv_heads,
                    const uint32_t heads_dim, const uint32_t pos,
                    const uint32_t max_seq_len, cudaStream_t stream_d);
template <const uint32_t NUM_THREADS>
void swiglu_bf16x2(const bf16* __restrict__ gate, const bf16* __restrict__ up,
                   bf16* __restrict__ intermedia, const uint32_t size,
                   cudaStream_t stream_d);
}  // namespace toyinfer