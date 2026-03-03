#pragma once
#include <cuda_runtime.h>

#include <cstdint>

namespace toyinfer {
void launch_add_kernel(float* a, float* b, float* c, int n);

void launch_precompute_inv_freq_kernel(float* inv_freq, int n, float base);
void launch_rope_kernel(float* emb, float* inv_freq, uint32_t pos,
                        uint32_t head_dim, uint32_t num_heads);
}  // namespace toyinfer