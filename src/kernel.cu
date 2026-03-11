#include <cstdint>
#include <iostream>

#include "cuda_bf16.h"
#include "kernel_warpper.h"

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        const cudaError_t err__ = (call);                                      \
        if (err__ != cudaSuccess) {                                            \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__) << " ("   \
                      << static_cast<int>(err__) << ") at " << __FILE__ << ':' \
                      << __LINE__ << std::endl;                                \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

#define FETCH_BF162_RO(addr) reinterpret_cast<const bf162*>(addr)[0]

namespace toyinfer {

__global__ void add_kernel(float* a, float* b, float* c, int n) {}

void launch_add_kernel(float* a, float* b, float* c, int n) {
    add_kernel<<<1, 1>>>(nullptr, nullptr, nullptr, 0);
};

__global__ void precompute_freq_kernel(float* inv_freq_d, const int head_dim,
                                       const float theta) {
    uint32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < (head_dim / 2)) {
        inv_freq_d[tid] = 1 / powf(theta, (float)(2 * tid) / head_dim);
    }
}

void precompute_freq_f32(float* inv_freq_d, const int head_dim,
                         const float theta) {
    dim3 block_dim = 64;
    dim3 grid_dim = ((head_dim / 2) + block_dim.x - 1) / block_dim.x;
    precompute_freq_kernel<<<grid_dim, block_dim>>>(inv_freq_d, head_dim,
                                                    theta);
    CUDA_CHECK(cudaGetLastError());
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
    CUDA_CHECK(cudaGetLastError());
    rmsnorm_bf16x2_kernel<<<grid_dim, block_dim>>>(input, weight, output, sum,
                                                   rms_norm_eps, size);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void multi_rmsnorm_bf16x2_kernel(const bf16* input,
                                            const bf16* __restrict__ weight,
                                            bf16* output,
                                            const float rms_norm_eps,
                                            const uint32_t head_dim) {
    __shared__ float s_warps_sum[32];
    __shared__ float multi_val;
    const uint32_t tid = threadIdx.x;
    const uint32_t num_warps = (head_dim + 31) / 32;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane_id = tid % 32;
    const uint32_t head_idx = blockIdx.x;
    const uint32_t offset = head_idx * head_dim;
    const uint32_t idx = tid * 2;
    float reg_sum = 0.0f;
    float2 reg_input, reg_weight;
    if (idx < head_dim) {
        reg_input = __bfloat1622float2(FETCH_BF162_RO(&input[offset + idx]));
        reg_weight = __bfloat1622float2(FETCH_BF162_RO(&weight[offset + idx]));
        reg_sum = fmaf(reg_input.x, reg_input.x, reg_sum);
        reg_sum = fmaf(reg_input.y, reg_input.y, reg_sum);
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
        multi_val = fmaf(reg_sum, 1.0f / head_dim, rms_norm_eps);
        multi_val = rsqrtf(multi_val);
    }
    __syncthreads();
    if (idx < head_dim) {
        reg_input.x = reg_input.x * multi_val * reg_weight.x;
        reg_input.y = reg_input.y * multi_val * reg_weight.y;
        reinterpret_cast<bf162*>(&output[offset + idx])[0] =
            __float22bfloat162_rn(reg_input);
    }
}

// 一个block算一个head的norm, blockDim = head_dim/2
void multi_rmsnorm_bf16(const bf16* input, const bf16* __restrict__ weight,
                        bf16* output, const float rms_norm_eps,
                        const uint32_t nums_head, const uint32_t head_dim) {
    dim3 block_dim{head_dim / 2};
    dim3 grid_dim{nums_head};
    multi_rmsnorm_bf16x2_kernel<<<grid_dim, block_dim>>>(
        input, weight, output, rms_norm_eps, head_dim);
    CUDA_CHECK(cudaGetLastError());
}

// one block for one row, 函数假设block大小等于row，未做边界判断
template <const uint32_t NUM_THREADS>
__global__ void gemv_bf16_kernel(const bf16* __restrict__ W,
                                 const bf16* __restrict__ x,
                                 bf16* __restrict__ y, const uint32_t M,
                                 const uint32_t N) {
    __shared__ float s_warps_sum[32];
    const uint32_t tid = threadIdx.x;
    const uint32_t num_warps = (NUM_THREADS + 31) / 32;
    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t row = blockIdx.x;
    const uint32_t offset = row * N;
    float reg_sum = 0.0f;
    float2 reg_w, reg_x;

    for (uint32_t col = tid * 2; col < N; col += NUM_THREADS * 2) {
        reg_w = __bfloat1622float2(FETCH_BF162_RO(&W[offset + col]));
        reg_x = __bfloat1622float2(FETCH_BF162_RO(&x[col]));
        reg_sum = fmaf(reg_w.x, reg_x.x, reg_sum);
        reg_sum = fmaf(reg_w.y, reg_x.y, reg_sum);
    }
    reg_sum = reduce_sum_f32_warp(reg_sum);
    if (lane_id == 0) {
        s_warps_sum[warp_id] = reg_sum;
    }
    __syncthreads();
    if (warp_id == 0) {
        reg_sum = warp_id < num_warps ? s_warps_sum[lane_id] : 0.0f;
        reg_sum = reduce_sum_f32_warp(reg_sum);
    }
    if (tid == 0) {
        y[row] = __bfloat162float(reg_sum);
    }
}

template <const uint32_t NUM_THREADS>
void attn_single_proj_bf16(const bf16* __restrict__ W,
                           const bf16* __restrict__ hidden_states,
                           bf16* __restrict__ y, const uint32_t M,
                           const uint32_t N) {
    dim3 block_dim{NUM_THREADS};
    dim3 grid_dim{M};
    gemv_bf16_kernel<NUM_THREADS>
        <<<grid_dim, block_dim>>>(W, hidden_states, y, M, N);
    CUDA_CHECK(cudaGetLastError());
}

// blockDim = head_dim / 2, gridDim = nums_heads
__global__ void rope_bf16x2_kernel(bf16* qk_ptr,
                                   const float* __restrict__ inv_freq,
                                   uint32_t pos, const uint32_t head_dim) {
    const uint32_t tid = threadIdx.x;
    const uint32_t freq_idx = threadIdx.x;
    const uint32_t offset = blockIdx.x * head_dim;
    float reg_real, reg_imag;
    float reg_rotated_real, reg_rotated_imag;
    float cos_freq, sin_freq;
    float freq;
    if (tid < (head_dim / 2)) {
        freq = inv_freq[freq_idx];
        reg_real = __bfloat162float(qk_ptr[offset + tid]);
        reg_imag = __bfloat162float(qk_ptr[offset + tid + head_dim / 2]);
        cos_freq = cosf(pos * freq);
        sin_freq = sinf(pos * freq);
        reg_rotated_real = cos_freq * reg_real - sin_freq * reg_imag;
        reg_rotated_imag = cos_freq * reg_imag + sin_freq * reg_real;
        qk_ptr[offset + tid] = __float2bfloat16(reg_rotated_real);
        qk_ptr[offset + tid + head_dim / 2] =
            __float2bfloat16(reg_rotated_imag);
    }
}

void rope_bf16(bf16* qk_ptr, const float* __restrict__ inv_freq, uint32_t pos,
               const uint32_t nums_head, const uint32_t head_dim) {
    dim3 block_dim{head_dim / 2};
    dim3 grid_dim{nums_head};
    rope_bf16x2_kernel<<<grid_dim, block_dim>>>(qk_ptr, inv_freq, pos,
                                                head_dim);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace toyinfer