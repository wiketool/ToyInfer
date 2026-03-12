#include <cassert>
#include <cstdint>
#include <iostream>

#include "cuda_bf16.h"
#include "kernel_warpper.h"
#include "math_constants.h"

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

__global__ void add_bf16_kernel(const bf16* __restrict__ residual,
                                bf16* __restrict__ hidden_state,
                                const uint32_t size) {
    //
    assert(size % 2 == 0);
    //
    const uint32_t idx = blockDim.x * blockIdx.x * 2 + threadIdx.x * 2;
    if (idx < size) {
        bf162 reg_res, reg_hid;
        reg_res = FETCH_BF162_RO(&residual[idx]);
        reg_hid = reinterpret_cast<bf162*>(&hidden_state[idx])[0];
        reg_hid.x = reg_hid.x + reg_res.x;
        reg_hid.y = reg_hid.y + reg_res.y;
        reinterpret_cast<bf162*>(&hidden_state[idx])[0] = reg_hid;
    }
}

template <const uint32_t NUM_THREADS>
void residual_add_bf16(const bf16* __restrict__ residual,
                       bf16* __restrict__ hidden_state, const uint32_t size) {
    dim3 block_dim{NUM_THREADS};
    dim3 grid_dim{(size + block_dim.x * 2 - 1) / (block_dim.x * 2)};
    add_bf16_kernel<<<grid_dim, block_dim>>>(residual, hidden_state, size);
    CUDA_CHECK(cudaGetLastError());
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
    dim3 block_dim = {64};
    dim3 grid_dim = {((head_dim / 2) + block_dim.x - 1) / block_dim.x};
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

__device__ float reduce_max_f32_warp(float val) {
#pragma unroll
    for (uint32_t stride = 16; stride >= 1; stride >>= 1) {
        val = fmaxf(__shfl_xor_sync(0xffffffff, val, stride), val);
    }
    return val;
}

__global__ void convert_float22bfloat162(const float* __restrict__ x,
                                         bf16* __restrict__ y,
                                         const uint32_t size) {
    //
    assert(size % 2 == 0);
    //
    const uint32_t idx = blockDim.x * blockIdx.x * 2 + threadIdx.x * 2;
    bf162 reg_xy;
    if (idx < size) {
        reg_xy.x = __float2bfloat16(x[idx]);
        reg_xy.y = __float2bfloat16(x[idx + 1]);
        reinterpret_cast<bf162*>(&y[idx])[0] = reg_xy;
    }
}

__global__ void reduce_sum_bf16x2_kernel(const bf16* __restrict__ input,
                                         float* sum, const uint32_t size) {
    assert(size % 2 == 0);
    assert(blockDim.x % 2 == 0);
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
    assert(size % 2 == 0);
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
    CUDA_CHECK(cudaGetLastError());
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
    assert(head_dim % 2 == 0);
    assert(blockDim.x % 32 == 0);
    __shared__ float s_warps_sum[32];
    __shared__ float multi_val;
    const uint32_t tid = threadIdx.x;
    const uint32_t num_warps = (blockDim.x + 31) / 32;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane_id = tid % 32;
    const uint32_t head_idx = blockIdx.x;
    const uint32_t offset = head_idx * head_dim;
    const uint32_t idx = tid * 2;
    float reg_sum = 0.0f;
    float2 reg_input, reg_weight;
    if (idx < head_dim) {
        reg_input = __bfloat1622float2(FETCH_BF162_RO(&input[offset + idx]));
        reg_weight = __bfloat1622float2(FETCH_BF162_RO(&weight[idx]));
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

// one block for one row, 函数假设nums_block等于row，未做边界判断
template <const uint32_t NUM_THREADS>
__global__ void gemv_bf16_kernel(const bf16* __restrict__ W,
                                 const bf16* __restrict__ x,
                                 bf16* __restrict__ y, const uint32_t M,
                                 const uint32_t N) {
    assert(NUM_THREADS % 32 == 0);
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
        reg_sum = lane_id < num_warps ? s_warps_sum[lane_id] : 0.0f;
        reg_sum = reduce_sum_f32_warp(reg_sum);
    }
    if (tid == 0) {
        y[row] = __float2bfloat16(reg_sum);
    }
}

template <const uint32_t NUM_THREADS>
void gemv_proj_bf16(const bf16* __restrict__ W,
                    const bf16* __restrict__ hidden_states,
                    bf16* __restrict__ y, const uint32_t M, const uint32_t N) {
    dim3 block_dim{NUM_THREADS};
    dim3 grid_dim{M};
    gemv_bf16_kernel<NUM_THREADS>
        <<<grid_dim, block_dim>>>(W, hidden_states, y, M, N);
    CUDA_CHECK(cudaGetLastError());
}

// one block for one row, 函数假设nums_block等于row，未做边界判断
template <const uint32_t NUM_THREADS>
__global__ void gemv_bf162float_kernel(const bf16* __restrict__ W,
                                       const bf16* __restrict__ x,
                                       float* __restrict__ y, const uint32_t M,
                                       const uint32_t N) {
    assert(NUM_THREADS % 32 == 0);
    assert(N % 2 == 0);
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
        reg_sum = lane_id < num_warps ? s_warps_sum[lane_id] : 0.0f;
        reg_sum = reduce_sum_f32_warp(reg_sum);
    }
    if (tid == 0) {
        y[row] = reg_sum;
    }
}

template <const uint32_t NUM_THREADS>
void gemv_proj_bf162float(const bf16* __restrict__ W,
                          const bf16* __restrict__ hidden_states,
                          float* __restrict__ y, const uint32_t M,
                          const uint32_t N) {
    dim3 block_dim{NUM_THREADS};
    dim3 grid_dim{M};
    gemv_bf162float_kernel<NUM_THREADS>
        <<<grid_dim, block_dim>>>(W, hidden_states, y, M, N);
    CUDA_CHECK(cudaGetLastError());
}

// blockDim = head_dim / 2, gridDim = nums_heads
__global__ void rope_bf16x2_kernel(bf16* qk_ptr,
                                   const float* __restrict__ inv_freq,
                                   uint32_t pos, const uint32_t head_dim) {
    assert(head_dim % 2 == 0);
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

// blockDim = NUM_THREADS, gridDim = pos+1
// 为了保证点乘规约的正确性，head_dim必须是64的倍数
template <const uint32_t NUM_THREADS>
__global__ void gqa_qk_gemv_bf16_kernel(const bf16* __restrict__ Q,
                                        const bf16* __restrict__ Ks,
                                        float* __restrict__ score,
                                        const uint32_t num_q_heads,
                                        const uint32_t num_kv_heads,
                                        const uint32_t heads_dim,
                                        const uint32_t max_seq_len) {
    assert(heads_dim % 64 == 0);  // 一个warp一次处理64个元素
    assert(blockDim.x % 32 == 0);
    const uint32_t tid = threadIdx.x;
    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t offset = blockIdx.x * (num_kv_heads * heads_dim);
    float reg_sum;
    float2 reg_q, reg_k;
    for (uint32_t idx = tid * 2; idx < num_q_heads * heads_dim;
         idx += NUM_THREADS * 2) {
        reg_sum = 0.0f;
        const uint32_t q_head_idx = idx / heads_dim;
        const uint32_t k_head_idx = q_head_idx / (num_q_heads / num_kv_heads);
        reg_q = __bfloat1622float2(FETCH_BF162_RO(&Q[idx]));
        reg_k = __bfloat1622float2(FETCH_BF162_RO(
            &Ks[offset + k_head_idx * heads_dim + idx % heads_dim]));
        reg_sum = fmaf(reg_q.x, reg_k.x, reg_sum);
        reg_sum = fmaf(reg_q.y, reg_k.y, reg_sum);
        reg_sum = reduce_sum_f32_warp(reg_sum);
        reg_sum = reg_sum * rsqrtf(heads_dim);
        if (lane_id == 0) {
            atomicAdd(&score[q_head_idx * max_seq_len + blockIdx.x], reg_sum);
        }
    }
}

// 总共q_heads个block,一个block负责一个q_head(有pos+1个数),固定NUM_THREADS循环处理直至pos
// 要求NUM_THREADS % 32 == 0
template <const uint32_t NUM_THREADS>
__global__ void softmax_f32_kernel(float* __restrict__ score,
                                   const uint32_t num_q_heads,
                                   const uint32_t pos,
                                   const uint32_t max_seq_len) {
    assert(blockDim.x % 32 == 0);
    __shared__ float s_warps[32];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t num_warps = (NUM_THREADS + 31) / 32;
    const uint32_t head_idx = blockIdx.x;
    const uint32_t offset = head_idx * max_seq_len;

    // 求max和sum
    float reg_sum = 0.0f, reg_max = -CUDART_INF_F;
    float reg_score;
    for (uint32_t idx = tid; idx <= pos; idx += NUM_THREADS) {
        const uint32_t score_idx = offset + idx;
        reg_score = score[score_idx];
        reg_max = fmaxf(reg_score, reg_max);
    }
    reg_max = reduce_max_f32_warp(reg_max);
    if (lane_id == 0) {
        s_warps[warp_id] = reg_max;
    }
    __syncthreads();
    if (warp_id == 0) {
        reg_max = lane_id < num_warps ? s_warps[lane_id] : -CUDART_INF_F;
        reg_max = reduce_max_f32_warp(reg_max);
    }
    if (tid == 0) {
        s_warps[0] = reg_max;
    }
    __syncthreads();
    // 求exp(sum)
    reg_max = s_warps[0];
    for (uint32_t idx = tid; idx <= pos; idx += NUM_THREADS) {
        const uint32_t score_idx = offset + idx;
        reg_score = score[score_idx] - reg_max;
        reg_sum += expf(reg_score);
    }
    reg_sum = reduce_sum_f32_warp(reg_sum);
    if (lane_id == 0) {
        s_warps[warp_id] = reg_sum;
    }
    __syncthreads();
    if (warp_id == 0) {
        reg_sum = lane_id < num_warps ? s_warps[lane_id] : 0.0f;
        reg_sum = reduce_sum_f32_warp(reg_sum);
    }
    if (tid == 0) {
        s_warps[0] = reg_sum;
    }
    __syncthreads();
    for (uint32_t idx = tid; idx <= pos; idx += NUM_THREADS) {
        const uint32_t score_idx = offset + idx;
        reg_score = expf(score[score_idx] - reg_max) / reg_sum;
        score[score_idx] = reg_score;
    }
}

// 一个block负责一个Q_head，block线程数为heads_dim/2,
template <const uint32_t TILE_SEQ>
__global__ void apply_score2v_f32_kernel(
    float* __restrict__ score, float* o, const bf16* __restrict__ Vs,
    const uint32_t num_q_heads, const uint32_t num_kv_heads,
    const uint32_t heads_dim, const uint32_t pos, const uint32_t max_seq_len) {
    //
    assert(heads_dim % 2 == 0);
    assert(blockDim.x == (heads_dim / 2));
    //
    const uint32_t tid = threadIdx.x;
    const uint32_t pos_offset = blockIdx.y * TILE_SEQ;
    const uint32_t pos_end = min(pos + 1, pos_offset + TILE_SEQ);
    const uint32_t head_idx = blockIdx.x;
    const uint32_t v_head_idx = head_idx / (num_q_heads / num_kv_heads);
    const uint32_t head_offset = head_idx * heads_dim;
    const uint32_t v_head_offset = v_head_idx * heads_dim;

    float2 reg_sum{0.0f, 0.0f};
    float2 reg_v;
    float reg_score;
    for (uint32_t pos_idx = pos_offset; pos_idx < pos_end; pos_idx++) {
        reg_score = score[head_idx * max_seq_len + pos_idx];
        const uint32_t k_idx =
            pos_idx * num_kv_heads * heads_dim + v_head_offset + tid * 2;
        reg_v = __bfloat1622float2(FETCH_BF162_RO(&Vs[k_idx]));
        reg_sum.x += reg_score * reg_v.x;
        reg_sum.y += reg_score * reg_v.y;
    }
    atomicAdd(&o[head_offset + tid * 2], reg_sum.x);
    atomicAdd(&o[head_offset + tid * 2 + 1], reg_sum.y);
}

template <const uint32_t NUM_THREADS, const uint32_t TILE_SEQ>
void attention_bf16(const bf16* __restrict__ Q, const bf16* __restrict__ Ks,
                    const bf16* __restrict__ Vs, float* __restrict__ score,
                    float* __restrict__ O_buffer, bf16* __restrict__ O,
                    const uint32_t num_q_heads, const uint32_t num_kv_heads,
                    const uint32_t heads_dim, const uint32_t pos,
                    const uint32_t max_seq_len) {
    dim3 block_dim, grid_dim;
    cudaMemset(score, 0, sizeof(float) * num_q_heads * max_seq_len);
    CUDA_CHECK(cudaGetLastError());
    // q*K
    block_dim = {NUM_THREADS};
    grid_dim = {pos + 1};
    gqa_qk_gemv_bf16_kernel<NUM_THREADS><<<grid_dim, block_dim>>>(
        Q, Ks, score, num_q_heads, num_kv_heads, heads_dim, max_seq_len);
    CUDA_CHECK(cudaGetLastError());
    grid_dim = {num_q_heads};
    softmax_f32_kernel<NUM_THREADS>
        <<<grid_dim, block_dim>>>(score, num_q_heads, pos, max_seq_len);
    CUDA_CHECK(cudaGetLastError());
    cudaMemset(O_buffer, 0, sizeof(float) * num_q_heads * heads_dim);
    CUDA_CHECK(cudaGetLastError());
    block_dim = {heads_dim / 2};
    grid_dim = {num_q_heads, (pos + 1 + TILE_SEQ - 1) / TILE_SEQ};
    apply_score2v_f32_kernel<TILE_SEQ>
        <<<grid_dim, block_dim>>>(score, O_buffer, Vs, num_q_heads,
                                  num_kv_heads, heads_dim, pos, max_seq_len);
    CUDA_CHECK(cudaGetLastError());
    block_dim = {NUM_THREADS};
    grid_dim = {(num_q_heads * heads_dim + block_dim.x * 2 - 1) /
                (block_dim.x * 2)};
    convert_float22bfloat162<<<grid_dim, block_dim>>>(O_buffer, O,
                                                      num_q_heads * heads_dim);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void swiglu_bf16x2_kernel(const bf16* __restrict__ gate,
                                     const bf16* __restrict__ up,
                                     bf16* __restrict__ intermedia,
                                     const uint32_t size) {
    //
    assert(size % 2 == 0);
    //
    const uint32_t idx = blockDim.x * blockIdx.x * 2 + threadIdx.x * 2;
    float2 reg_gate, reg_up;
    if (idx < size) {
        reg_gate = __bfloat1622float2(FETCH_BF162_RO(&gate[idx]));
        reg_up = __bfloat1622float2(FETCH_BF162_RO(&up[idx]));
        reg_gate.x = reg_gate.x * (1.0f / (1.0f + expf(-reg_gate.x)));
        reg_gate.y = reg_gate.y * (1.0f / (1.0f + expf(-reg_gate.y)));
        reg_up.x = reg_up.x * reg_gate.x;
        reg_up.y = reg_up.y * reg_gate.y;
        reinterpret_cast<bf162*>(&intermedia[idx])[0] =
            __float22bfloat162_rn(reg_up);
    }
}

template <const uint32_t NUM_THREADS>
void swiglu_bf16x2(const bf16* __restrict__ gate, const bf16* __restrict__ up,
                   bf16* __restrict__ intermedia, const uint32_t size) {
    dim3 block_dim{NUM_THREADS};
    dim3 grid_dim{(size + block_dim.x * 2 - 1) / (block_dim.x * 2)};
    swiglu_bf16x2_kernel<<<grid_dim, block_dim>>>(gate, up, intermedia, size);
    CUDA_CHECK(cudaGetLastError());
}

template void residual_add_bf16<256>(const bf16* __restrict__ residual,
                                     bf16* __restrict__ hidden_state,
                                     const uint32_t size);
template void rmsnorm_bf16<256>(const bf16* __restrict__ input,
                                const bf16* __restrict__ weight, bf16* output,
                                float* sum, const float rms_norm_eps,
                                const uint32_t size);
template void gemv_proj_bf16<256>(const bf16* __restrict__ W,
                                  const bf16* __restrict__ hidden_states,
                                  bf16* __restrict__ y, const uint32_t M,
                                  const uint32_t N);

template void gemv_proj_bf162float<256>(const bf16* __restrict__ W,
                                        const bf16* __restrict__ hidden_states,
                                        float* __restrict__ y, const uint32_t M,
                                        const uint32_t N);
template void attention_bf16<256, 32>(
    const bf16* __restrict__ Q, const bf16* __restrict__ Ks,
    const bf16* __restrict__ Vs, float* __restrict__ score,
    float* __restrict__ O_buffer, bf16* __restrict__ O,
    const uint32_t num_q_heads, const uint32_t num_kv_heads,
    const uint32_t heads_dim, const uint32_t pos, const uint32_t max_seq_len);
template void swiglu_bf16x2<256>(const bf16* __restrict__ gate,
                                 const bf16* __restrict__ up,
                                 bf16* __restrict__ intermedia,
                                 const uint32_t size);

}  // namespace toyinfer
