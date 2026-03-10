#pragma once
#include <cuda_runtime.h>

#include <cstdint>

#include "cuda_bf16.h"

namespace toyinfer {
using bf16 = __nv_bfloat16;
using bf162 = __nv_bfloat162;

void launch_add_kernel(float* a, float* b, float* c, int n);

void precompute_freq_f32(float* inv_freq, int n, float base);

template <const uint32_t NUM_THREADS>
void rmsnorm_bf16(const bf16* __restrict__ input,
                  const bf16* __restrict__ weight, bf16* output, float* sum,
                  const float rms_norm_eps, const uint32_t size);

// input和output是同一个地址，不能使用__restrict__
void multi_rmsnorm_bf16(const bf16* input, const bf16* __restrict__ weight,
                        bf16* output, const float rms_norm_eps,
                        const uint32_t nums_head, const uint32_t head_dim);

template <const uint32_t NUM_THREADS>
void attn_single_proj_bf16(const bf16* __restrict__ W,
                           const bf16* __restrict__ hidden_states,
                           bf16* __restrict__ y, const uint32_t M,
                           const uint32_t N);

void rope_bf16(bf16* qk_ptr, const float* __restrict__ inv_freq, uint32_t pos,
               const uint32_t nums_head, const uint32_t head_dim);

void attention_bf16(const bf16* __restrict__ Q, const bf16* __restrict__ Ks,
                    const bf16* __restrict__ Vs, const uint32_t num_q_heads,
                    const uint32_t num_kv_heads, const uint32_t heads_dim);
}  // namespace toyinfer