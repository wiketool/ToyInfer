#pragma once
#include <cuda_runtime.h>

#include <cstdint>

#include "cuda_bf16.h"

namespace toyinfer {
using bf16 = __nv_bfloat16;
using bf162 = __nv_bfloat162;

void launch_add_kernel(float* a, float* b, float* c, int n);

void precompute_theta_f32(float* inv_freq, int n, float base);
void rmsnorm_bf16(const bf16* __restrict__ input,
                  const bf16* __restrict__ weight, bf16* output, float* sum,
                  const float rms_norm_eps, const uint32_t size);
void launch_rope_kernel(float* emb, float* inv_freq, uint32_t pos,
                        uint32_t head_dim, uint32_t num_heads);
}  // namespace toyinfer