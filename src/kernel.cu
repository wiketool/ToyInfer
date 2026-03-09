#include <cstdint>

#include "cuda_bf16.h"
#include "kernel_warpper.h"

namespace toyinfer {

#define FETCH_BF162_RO(addr) \
    reinterpret_cast<const bf162* __restrict__>(addr)[0]

__global__ void add_kernel(float* a, float* b, float* c, int n) {}

void launch_add_kernel(float* a, float* b, float* c, int n) {
    add_kernel<<<1, 1>>>(nullptr, nullptr, nullptr, 0);
};

__global__ void precompute_theta_kernel(float* inv_freq_d, int n, float theta) {
    uint32_t tid = threadIdx.x;
    if (tid < n) {
        inv_freq_d[tid] = 1 / powf(theta, tid);
    }
}

void precompute_theta_f32(float* inv_freq_d, int n, float theta) {
    uint32_t block_dim = 32;
    uint32_t grid_dim = (n + block_dim - 1) / block_dim;
    precompute_theta_kernel<<<grid_dim, block_dim>>>(inv_freq_d, n, theta);
}

__device__ float reduce_sum_f32_warp(float val) {
#pragma unroll
    for (uint32_t stride = 16; stride >= 1; stride >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, stride);
    }
    return val;
}

__global__ void reduce_sum_bf16x2_kernel(const bf16* __restrict__ input,
                                         float* sum, const uint32_t size) {
    __shared__ float s_warps_sum[32];
    const uint32_t offset = blockDim.x * blockIdx.x * 2;
    const uint32_t num_warps = (blockDim.x + 31) / 32;
    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t tid = threadIdx.x;

    float reg_sum = 0.0f;
    float2 tmp;
    if (offset + tid * 2 < size) {
        tmp = __bfloat1622float2(FETCH_BF162_RO(&input[offset + tid * 2]));
        reg_sum = fmaf(tmp.x, tmp.x, reg_sum);
        reg_sum = fmaf(tmp.y, tmp.y, reg_sum);
    }
    reg_sum = reduce_sum_f32_warp(reg_sum);
    if (lane_id == 0) {
        s_warps_sum[warp_id] = reg_sum;
    }
    __syncthreads();
    if (warp_id == 0) {
        reg_sum = lane_id < num_warps ? s_warps_sum[lane_id] : 0.0f;
        reg_sum = reduce_sum_f32_warp(reg_sum);
    }
    if (tid == 0) {
        atomicAdd(sum, reg_sum);
    }
}

__global__ void rmsnorm_bf16x2_kernel(const bf16* __restrict__ input,
                                      const bf16* __restrict__ weight,
                                      bf16* output, const float* sum,
                                      const float rms_norm_eps,
                                      const uint32_t size) {
    const uint32_t tid = threadIdx.x;
    const uint32_t offset = blockDim.x * blockIdx.x * 2;
    float2 input_f32x2;
    float2 weight_f32x2;
    float2 normed_f32x2;
    float val;

    if (offset + tid * 2 < size) {
        input_f32x2 =
            __bfloat1622float2(FETCH_BF162_RO(&input[offset + tid * 2]));
        weight_f32x2 =
            __bfloat1622float2(FETCH_BF162_RO(&weight[offset + tid * 2]));
        // 先load再计算，这样可以overlap
        val = fmaf(*sum, 1.0f / size, rms_norm_eps);
        val = rsqrtf(val);
        normed_f32x2.x = input_f32x2.x * val * weight_f32x2.x;
        normed_f32x2.y = input_f32x2.y * val * weight_f32x2.y;
        reinterpret_cast<bf162*>(&output[offset + tid * 2])[0] =
            __float22bfloat162_rn(normed_f32x2);
    }
}

template <const uint32_t NUM_THREADS>
void rmsnorm_bf16(const bf16* __restrict__ input,
                  const bf16* __restrict__ weight, bf16* output, float* sum,
                  const float rms_norm_eps, const uint32_t size) {
    dim3 block_dim{NUM_THREADS};
    dim3 grid_dim{(size + block_dim.x * 2 - 1) / (block_dim.x * 2)};
    cudaMemset(sum, 0, sizeof(float));
    reduce_sum_bf16x2_kernel<<<grid_dim, block_dim>>>(input, sum, size);
    rmsnorm_bf16x2_kernel<<<grid_dim, block_dim>>>(input, weight, output, sum,
                                                   rms_norm_eps, size);
}

}  // namespace toyinfer