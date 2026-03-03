#include <cstdint>

#include "kernel_warpper.h"

namespace toyinfer {
__global__ void add_kernel(float* a, float* b, float* c, int n) {}

void launch_add_kernel(float* a, float* b, float* c, int n) {
    add_kernel<<<1, 1>>>(nullptr, nullptr, nullptr, 0);
};

__global__ void precompute_inv_freq_kernel(float* inv_freq_d, int n, float theta) {
    uint32_t tid = threadIdx.x;
    if (tid < n) {
        inv_freq_d[tid] = 1 / powf(theta, tid);
    }
}

void launch_precompute_theta_kernel(float* inv_freq_d, int n, float theta) {
    uint32_t block_dim = 32;
    uint32_t grid_dim = (n + block_dim - 1) / block_dim;
    precompute_inv_freq_kernel<<<grid_dim, block_dim>>>(inv_freq_d, n, theta);
}
}  // namespace toyinfer