#include "kernel_common.cuh"
#include "kernel_warpper.h"

namespace toyinfer {

template <const uint32_t NUM_THREADS>
__global__ void gather_embedding_bf16_kernel(
    const bf16* __restrict__ embedding_table,
    const uint32_t* __restrict__ token_ids, bf16* __restrict__ hidden_states,
    const uint32_t num_tokens, const uint32_t hidden_size) {
    assert(hidden_size % 2 == 0);
    const uint32_t batch_idx = blockIdx.y;
    const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (batch_idx >= num_tokens || col >= hidden_size) {
        return;
    }
    const uint32_t token_id = token_ids[batch_idx];
    const uint32_t src_idx = token_id * hidden_size + col;
    const uint32_t dst_idx = batch_idx * hidden_size + col;
    FETCH_BF162(&hidden_states[dst_idx]) =
        FETCH_BF162_RO(&embedding_table[src_idx]);
}

template <const uint32_t NUM_THREADS>
void gather_embedding_bf16(const bf16* __restrict__ embedding_table,
                           const uint32_t* __restrict__ token_ids,
                           bf16* __restrict__ hidden_states,
                           const uint32_t num_tokens,
                           const uint32_t hidden_size, cudaStream_t stream_d) {
    dim3 block_dim{NUM_THREADS};
    dim3 grid_dim{(hidden_size + block_dim.x * 2 - 1) / (block_dim.x * 2),
                  num_tokens};
    gather_embedding_bf16_kernel<NUM_THREADS>
        <<<grid_dim, block_dim, 0, stream_d>>>(
            embedding_table, token_ids, hidden_states, num_tokens, hidden_size);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void batch_reduce_sum_bf16x2_kernel(const bf16* __restrict__ input,
                                               float* sum,
                                               const uint32_t size) {
    assert(size % 2 == 0);
    assert(blockDim.x % 32 == 0);
    __shared__ float s_warps_sum[32];
    const uint32_t row = blockIdx.x;
    const uint32_t row_offset = row * size;
    const uint32_t tid = threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane_id = tid % 32;
    const uint32_t num_warps = (blockDim.x + 31) / 32;
    float reg_sum = 0.0f;
    float2 reg_input;

    for (uint32_t idx = tid * 2; idx < size; idx += blockDim.x * 2) {
        reg_input =
            __bfloat1622float2(FETCH_BF162_RO(&input[row_offset + idx]));
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
        sum[row] = reg_sum;
    }
}

__global__ void batch_rmsnorm_bf16x2_kernel(const bf16* __restrict__ input,
                                            const bf16* __restrict__ weight,
                                            bf16* __restrict__ output,
                                            const float* __restrict__ sum,
                                            const float rms_norm_eps,
                                            const uint32_t size) {
    assert(size % 2 == 0);
    const uint32_t row = blockIdx.y;
    const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (col >= size) {
        return;
    }
    const uint32_t row_offset = row * size;
    float2 reg_input =
        __bfloat1622float2(FETCH_BF162_RO(&input[row_offset + col]));
    float2 reg_weight = __bfloat1622float2(FETCH_BF162_RO(&weight[col]));
    const float val = rsqrtf(fmaf(sum[row], 1.0f / size, rms_norm_eps));
    reg_input.x = reg_input.x * val * reg_weight.x;
    reg_input.y = reg_input.y * val * reg_weight.y;
    FETCH_BF162(&output[row_offset + col]) = __float22bfloat162_rn(reg_input);
}

template <const uint32_t NUM_THREADS>
void batch_rmsnorm_bf16(const bf16* __restrict__ input,
                        const bf16* __restrict__ weight, bf16* output,
                        float* sum, const float rms_norm_eps,
                        const uint32_t num_tokens, const uint32_t size,
                        cudaStream_t stream_d) {
    dim3 block_dim{NUM_THREADS};
    dim3 grid_dim_reduce{num_tokens};
    dim3 grid_dim_norm{(size + block_dim.x * 2 - 1) / (block_dim.x * 2),
                       num_tokens};
    batch_reduce_sum_bf16x2_kernel<<<grid_dim_reduce, block_dim, 0, stream_d>>>(
        input, sum, size);
    CUDA_CHECK(cudaGetLastError());
    batch_rmsnorm_bf16x2_kernel<<<grid_dim_norm, block_dim, 0, stream_d>>>(
        input, weight, output, sum, rms_norm_eps, size);
    CUDA_CHECK(cudaGetLastError());
}

void batch_multi_rmsnorm_bf16(const bf16* input,
                              const bf16* __restrict__ weight, bf16* output,
                              const float rms_norm_eps,
                              const uint32_t num_tokens,
                              const uint32_t nums_head, const uint32_t head_dim,
                              cudaStream_t stream_d) {
    multi_rmsnorm_bf16(input, weight, output, rms_norm_eps,
                       num_tokens * nums_head, head_dim, stream_d);
}

template <const uint32_t NUM_THREADS>
__global__ void batch_gemv_bf16_kernel(const bf16* __restrict__ W,
                                       const bf16* __restrict__ hidden_states,
                                       bf16* __restrict__ y, const uint32_t M,
                                       const uint32_t N) {
    assert(NUM_THREADS % 32 == 0);
    __shared__ float s_warps_sum[32];
    const uint32_t tid = threadIdx.x;
    const uint32_t num_warps = (NUM_THREADS + 31) / 32;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane_id = tid % 32;
    const uint32_t row = blockIdx.x;
    const uint32_t batch_idx = blockIdx.y;
    const uint32_t w_offset = row * N;
    const uint32_t x_offset = batch_idx * N;
    float reg_sum = 0.0f;
    float2 reg_w, reg_x;

    for (uint32_t col = tid * 2; col < N; col += NUM_THREADS * 2) {
        reg_w = __bfloat1622float2(FETCH_BF162_RO(&W[w_offset + col]));
        reg_x =
            __bfloat1622float2(FETCH_BF162_RO(&hidden_states[x_offset + col]));
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
        y[batch_idx * M + row] = __float2bfloat16(reg_sum);
    }
}

template <const uint32_t NUM_THREADS>
void batch_gemv_proj_bf16(const bf16* __restrict__ W,
                          const bf16* __restrict__ hidden_states,
                          bf16* __restrict__ y, const uint32_t num_tokens,
                          const uint32_t M, const uint32_t N,
                          cudaStream_t stream_d) {
    dim3 block_dim{NUM_THREADS};
    dim3 grid_dim{M, num_tokens};
    batch_gemv_bf16_kernel<NUM_THREADS>
        <<<grid_dim, block_dim, 0, stream_d>>>(W, hidden_states, y, M, N);
    CUDA_CHECK(cudaGetLastError());
}

template <const uint32_t NUM_THREADS>
void batch_residual_add_bf16(const bf16* __restrict__ residual,
                             bf16* __restrict__ hidden_state,
                             const uint32_t num_tokens, const uint32_t size,
                             cudaStream_t stream_d) {
    residual_add_bf16<NUM_THREADS>(residual, hidden_state, num_tokens * size,
                                   stream_d);
}

__global__ void batch_rope_bf16x2_kernel(bf16* qk_ptr,
                                         const float* __restrict__ inv_freq,
                                         const uint32_t nums_head,
                                         const uint32_t head_dim) {
    assert(head_dim % 2 == 0);
    const uint32_t tid = threadIdx.x;
    if (tid >= (head_dim / 2)) {
        return;
    }
    const uint32_t token_idx = blockIdx.x / nums_head;
    const uint32_t head_idx = blockIdx.x % nums_head;
    const uint32_t offset =
        token_idx * nums_head * head_dim + head_idx * head_dim;
    const float freq = inv_freq[tid];
    const float reg_real = __bfloat162float(qk_ptr[offset + tid]);
    const float reg_imag =
        __bfloat162float(qk_ptr[offset + tid + head_dim / 2]);
    const float cos_freq = cosf(token_idx * freq);
    const float sin_freq = sinf(token_idx * freq);
    const float reg_rotated_real = cos_freq * reg_real - sin_freq * reg_imag;
    const float reg_rotated_imag = cos_freq * reg_imag + sin_freq * reg_real;
    qk_ptr[offset + tid] = __float2bfloat16(reg_rotated_real);
    qk_ptr[offset + tid + head_dim / 2] = __float2bfloat16(reg_rotated_imag);
}

void batch_rope_bf16(bf16* qk_ptr, const float* __restrict__ inv_freq,
                     const uint32_t num_tokens, const uint32_t nums_head,
                     const uint32_t head_dim, cudaStream_t stream_d) {
    dim3 block_dim{head_dim / 2};
    dim3 grid_dim{num_tokens * nums_head};
    batch_rope_bf16x2_kernel<<<grid_dim, block_dim, 0, stream_d>>>(
        qk_ptr, inv_freq, nums_head, head_dim);
    CUDA_CHECK(cudaGetLastError());
}

template <const uint32_t Bc, const uint32_t Br, const uint32_t HEAD_DIM>
__global__ void flash_attention_v1_bf16_kernel(
    const bf16* __restrict__ Qs, const bf16* __restrict__ Ks,
    const bf16* __restrict__ Vs, bf16* __restrict__ Os,
    const uint32_t num_q_heads, const uint32_t num_kv_heads,
    const uint32_t heads_dim, const uint32_t seq_len) {
    // warp级别归约
    assert(Bc % 32 == 0);
    assert(blockDim.x == Bc);
    assert(blockDim.y == Br);
    assert(HEAD_DIM % Bc == 0);
    // 对QKV使用原始的bf16，运算时转换，减少share memory压力
    __shared__ bf16 s_Q[Br][HEAD_DIM];
    __shared__ bf16 s_K[Bc][HEAD_DIM];
    __shared__ bf16 s_V[Bc][HEAD_DIM];
    __shared__ float s_score[Br][Bc];
    __shared__ float s_m[Br];
    __shared__ float s_l[Br];
    // shape(s_O) = (Br,head_dim)
    __shared__ float s_O[Br][HEAD_DIM];
    __shared__ float s_warp_max[Br][Bc / 32];
    __shared__ float s_warp_sum[Br][Bc / 32];
    //
    float reg_score = 0.0f;
    float reg_max = 0.0f;
    float reg_sum = 0.0f;
    float reg_delta = 0.0f;  // m(i-1) - m(i)
    float reg_d[HEAD_DIM / Bc];

    //

    const uint32_t tx = threadIdx.x;
    const uint32_t ty = threadIdx.y;
    const uint32_t num_threads = blockDim.x * blockDim.y;
    const uint32_t lane_id = tx % 32;
    const uint32_t warp_id = tx / 32;
    const uint32_t num_warps = blockDim.x / 32;
    // head dim slice分片
    const uint32_t slice_size = HEAD_DIM / Bc;
    const uint32_t dim_start = tx * slice_size;
    //
    const uint32_t head_idx = blockIdx.x;
    const uint32_t kv_head_idx = head_idx / (num_q_heads / num_kv_heads);
    const uint32_t q_dim = num_q_heads * heads_dim;
    const uint32_t kv_dim = num_kv_heads * heads_dim;

    const uint32_t q_global = blockIdx.y * Br;
    const uint32_t q_tile = ty;
    const uint32_t q_idx = q_global + q_tile;
    const uint32_t q_offset = q_idx * q_dim + head_idx * heads_dim;

    const uint32_t kv_tile = tx;
    // init m,l
    if (tx == 0) {
        s_m[q_tile] = -CUDART_INF_F;
        s_l[q_tile] = 0.0f;
    }

    // load Q,O
    for (uint32_t i = threadIdx.x; i < heads_dim; i += blockDim.x) {
        if (q_idx < seq_len) {
            s_Q[q_tile][i] = Qs[q_offset + i];  // load Q
        } else {
            s_Q[q_tile][i] = __float2bfloat16(0.0f);
        }
        s_O[q_tile][i] = 0.0f;  // set O to zero, init
    }
    // loop over K/V
    for (uint32_t i = 0; i < (seq_len + Bc - 1) / Bc; i++) {
        const uint32_t kv_idx = i * Bc + kv_tile;
        // load K and V
        const uint32_t tid = blockDim.x * ty + tx;
        for (uint32_t idx = tid; idx < Bc * heads_dim; idx += num_threads) {
            const uint32_t row = idx / heads_dim;
            const uint32_t col = idx % heads_dim;
            const uint32_t row_global = i * Bc + row;
            const uint32_t col_global = kv_head_idx * heads_dim + col;
            if (row_global < seq_len) {
                s_K[row][col] = Ks[row_global * kv_dim + col_global];
                s_V[row][col] = Vs[row_global * kv_dim + col_global];
            } else {
                s_K[row][col] = bf16{0.0f};
                s_V[row][col] = bf16{0.0f};
            }
        }
        __syncthreads();
        reg_score = 0.0f;
        // (Q*K)/sqrt(d)
        for (uint32_t j = 0; j < heads_dim; j++) {
            reg_score = fmaf(__bfloat162float(s_Q[q_tile][j]),
                             __bfloat162float(s_K[kv_tile][j]), reg_score);
        }
        reg_score *= rsqrtf(heads_dim);
        if (q_idx < kv_idx || q_idx >= seq_len || kv_idx >= seq_len) {
            reg_score = -CUDART_INF_F;
        }
        // reduce
        __syncthreads();
        reg_max = reduce_max_f32_warp(reg_score);
        reg_sum = reduce_sum_f32_warp(expf(reg_score));
        if (lane_id == 0) {
            s_warp_max[q_tile][warp_id] = reg_max;
            s_warp_sum[q_tile][warp_id] = reg_sum;
        }
        __syncthreads();
        if (warp_id == 0) {
            reg_max =
                lane_id < num_warps ? s_warp_max[q_tile][lane_id] : -CUDART_INF_F;
            reg_max = reduce_max_f32_warp(reg_max);
            reg_sum = lane_id < num_warps ? s_warp_sum[q_tile][lane_id] : 0.0f;
            reg_sum = reduce_sum_f32_warp(reg_sum);
        }
        if (tx == 0) {
            s_warp_max[q_tile][0] = reg_max;
            s_warp_sum[q_tile][0] = reg_sum;
        }
        __syncthreads();
        reg_max = s_warp_max[q_tile][0];
        reg_sum = s_warp_sum[q_tile][0];
        reg_max = fmaxf(s_m[q_tile], reg_max);
        reg_delta = s_m[q_tile] - reg_max;
        reg_score = expf(reg_score - reg_max);
        s_score[q_tile][kv_tile] = reg_score;
        reg_sum = reg_sum * expf(-reg_max);

        // 计算O
        for (uint32_t dim_slice = 0; dim_slice < slice_size; dim_slice++) {
            reg_d[dim_slice] = 0;
        }
        for (uint32_t v_tile = 0; v_tile < Bc; v_tile++) {
            reg_score = s_score[q_tile][v_tile];
            for (uint32_t dim_slice = 0; dim_slice < slice_size; dim_slice++) {
                const uint32_t dim_idx = dim_slice + dim_start;
                reg_d[dim_slice] +=
                    __bfloat162float(s_V[v_tile][dim_idx]) * reg_score;
            }
        }
        // 输出m,l,O
        if (tx == 0) {
            s_m[q_tile] = reg_max;
            s_l[q_tile] = expf(reg_delta) * s_l[q_tile] + reg_sum;
        }
        for (uint32_t dim_slice = 0; dim_slice < slice_size; dim_slice++) {
            const uint32_t dim_idx = dim_slice + dim_start;
            s_O[q_tile][dim_idx] =
                expf(reg_delta) * s_O[q_tile][dim_idx] + reg_d[dim_slice];
        }
        __syncthreads();
    }
    if (q_idx < seq_len) {
        for (uint32_t dim_slice = 0; dim_slice < slice_size; dim_slice++) {
            const uint32_t dim_idx = dim_slice + dim_start;
            Os[q_offset + dim_idx] =
                __float2bfloat16(s_O[q_tile][dim_idx] / s_l[q_tile]);
        }
    }
}

// Tile size for K/V = Bc, Tile size for Q/O = Br;
template <const uint32_t Bc, const uint32_t Br, const uint32_t HEAD_DIM>
void flash_attention_v1_bf16(const bf16* __restrict__ Qs,
                             const bf16* __restrict__ Ks,
                             const bf16* __restrict__ Vs, bf16* __restrict__ Os,
                             const uint32_t num_q_heads,
                             const uint32_t num_kv_heads,
                             const uint32_t heads_dim, const uint32_t seq_len,
                             cudaStream_t stream_d) {
    const dim3 block_dim{Bc, Br};
    const dim3 grid_dim{num_q_heads, (seq_len + Br - 1) / Br};
    flash_attention_v1_bf16_kernel<Bc, Br, HEAD_DIM>
        <<<grid_dim, block_dim, 0, stream_d>>>(
            Qs, Ks, Vs, Os, num_q_heads, num_kv_heads, heads_dim, seq_len);
}

template <const uint32_t NUM_THREADS>
void batch_swiglu_bf16x2(const bf16* __restrict__ gate,
                         const bf16* __restrict__ up,
                         bf16* __restrict__ intermedia,
                         const uint32_t num_tokens, const uint32_t size,
                         cudaStream_t stream_d) {
    swiglu_bf16x2<NUM_THREADS>(gate, up, intermedia, num_tokens * size,
                               stream_d);
}

template void gather_embedding_bf16<256>(
    const bf16* __restrict__ embedding_table,
    const uint32_t* __restrict__ token_ids, bf16* __restrict__ hidden_states,
    const uint32_t num_tokens, const uint32_t hidden_size,
    cudaStream_t stream_d);
template void batch_rmsnorm_bf16<256>(
    const bf16* __restrict__ input, const bf16* __restrict__ weight,
    bf16* output, float* sum, const float rms_norm_eps,
    const uint32_t num_tokens, const uint32_t size, cudaStream_t stream_d);
template void batch_gemv_proj_bf16<256>(const bf16* __restrict__ W,
                                        const bf16* __restrict__ hidden_states,
                                        bf16* __restrict__ y,
                                        const uint32_t num_tokens,
                                        const uint32_t M, const uint32_t N,
                                        cudaStream_t stream_d);
template void batch_residual_add_bf16<256>(const bf16* __restrict__ residual,
                                           bf16* __restrict__ hidden_state,
                                           const uint32_t num_tokens,
                                           const uint32_t size,
                                           cudaStream_t stream_d);
template void batch_swiglu_bf16x2<256>(const bf16* __restrict__ gate,
                                       const bf16* __restrict__ up,
                                       bf16* __restrict__ intermedia,
                                       const uint32_t num_tokens,
                                       const uint32_t size,
                                       cudaStream_t stream_d);
template void flash_attention_v1_bf16<32, 2, 64>(
    const bf16* __restrict__ Qs, const bf16* __restrict__ Ks,
    const bf16* __restrict__ Vs, bf16* __restrict__ Os,
    const uint32_t num_q_heads, const uint32_t num_kv_heads,
    const uint32_t heads_dim, const uint32_t seq_len, cudaStream_t stream_d);
template void flash_attention_v1_bf16<32, 4, 64>(
    const bf16* __restrict__ Qs, const bf16* __restrict__ Ks,
    const bf16* __restrict__ Vs, bf16* __restrict__ Os,
    const uint32_t num_q_heads, const uint32_t num_kv_heads,
    const uint32_t heads_dim, const uint32_t seq_len, cudaStream_t stream_d);
template void flash_attention_v1_bf16<32, 4, 128>(
    const bf16* __restrict__ Qs, const bf16* __restrict__ Ks,
    const bf16* __restrict__ Vs, bf16* __restrict__ Os,
    const uint32_t num_q_heads, const uint32_t num_kv_heads,
    const uint32_t heads_dim, const uint32_t seq_len, cudaStream_t stream_d);
template void flash_attention_v1_bf16<64, 4, 128>(
    const bf16* __restrict__ Qs, const bf16* __restrict__ Ks,
    const bf16* __restrict__ Vs, bf16* __restrict__ Os,
    const uint32_t num_q_heads, const uint32_t num_kv_heads,
    const uint32_t heads_dim, const uint32_t seq_len, cudaStream_t stream_d);

}  // namespace toyinfer
