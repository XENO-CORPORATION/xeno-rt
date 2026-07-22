#include <cuda_runtime.h>

__device__ __forceinline__ float xrt_fast_exp_cpu_order(float value) {
    float exponent = __fmul_rn(value, 1.4426950408889634f);
    exponent = fmaxf(exponent, -126.0f);
    exponent = fminf(exponent, 126.0f);
    const int rounded_exponent = __float2int_rn(exponent);
    const float rounded = __int2float_rn(rounded_exponent);
    const float remainder = __fsub_rn(exponent, rounded);
    const float two_to_integer =
        __int_as_float((rounded_exponent + 127) << 23);

    float polynomial = 0.00960083f;
    polynomial = __fmaf_rn(polynomial, remainder, 0.05550862f);
    polynomial = __fmaf_rn(polynomial, remainder, 0.24015523f);
    polynomial = __fmaf_rn(polynomial, remainder, 0.69315863f);
    polynomial = __fmaf_rn(polynomial, remainder, 1.0f);
    return __fmul_rn(two_to_integer, polynomial);
}

__device__ __forceinline__ float xrt_reduce_cpu_four_chains(
    float* chains,
    unsigned int lane) {
    if (lane < 8) {
        const float pair_01 = __fadd_rn(chains[lane], chains[lane + 8]);
        const float pair_23 = __fadd_rn(chains[lane + 16], chains[lane + 24]);
        chains[lane] = __fadd_rn(pair_01, pair_23);
    }
    __syncthreads();

    if (lane != 0) {
        return 0.0f;
    }
    const float sum_04 = __fadd_rn(chains[0], chains[4]);
    const float sum_15 = __fadd_rn(chains[1], chains[5]);
    const float sum_26 = __fadd_rn(chains[2], chains[6]);
    const float sum_37 = __fadd_rn(chains[3], chains[7]);
    const float sum_0246 = __fadd_rn(sum_04, sum_26);
    const float sum_1357 = __fadd_rn(sum_15, sum_37);
    return __fadd_rn(sum_0246, sum_1357);
}

__device__ __forceinline__ float xrt_reduce_cpu_two_chains(
    float* chains,
    unsigned int lane) {
    if (lane < 8) {
        chains[lane] = __fadd_rn(chains[lane], chains[lane + 8]);
    }
    __syncthreads();

    if (lane != 0) {
        return 0.0f;
    }
    const float sum_04 = __fadd_rn(chains[0], chains[4]);
    const float sum_15 = __fadd_rn(chains[1], chains[5]);
    const float sum_26 = __fadd_rn(chains[2], chains[6]);
    const float sum_37 = __fadd_rn(chains[3], chains[7]);
    const float sum_0246 = __fadd_rn(sum_04, sum_26);
    const float sum_1357 = __fadd_rn(sum_15, sum_37);
    return __fadd_rn(sum_0246, sum_1357);
}

__device__ __forceinline__ float xrt_reduce_cpu_four_chains_warp(
    float accumulator,
    unsigned int lane) {
    const unsigned int mask = 0xffffffffu;
    const float lane_8 =
        __shfl_sync(mask, accumulator, static_cast<int>((lane + 8) & 31));
    const float lane_16 =
        __shfl_sync(mask, accumulator, static_cast<int>((lane + 16) & 31));
    const float lane_24 =
        __shfl_sync(mask, accumulator, static_cast<int>((lane + 24) & 31));
    float combined = accumulator;
    if (lane < 8) {
        const float pair_01 = __fadd_rn(accumulator, lane_8);
        const float pair_23 = __fadd_rn(lane_16, lane_24);
        combined = __fadd_rn(pair_01, pair_23);
    }

    const float chain_1 = __shfl_sync(mask, combined, 1);
    const float chain_2 = __shfl_sync(mask, combined, 2);
    const float chain_3 = __shfl_sync(mask, combined, 3);
    const float chain_4 = __shfl_sync(mask, combined, 4);
    const float chain_5 = __shfl_sync(mask, combined, 5);
    const float chain_6 = __shfl_sync(mask, combined, 6);
    const float chain_7 = __shfl_sync(mask, combined, 7);
    if (lane != 0) {
        return 0.0f;
    }
    const float sum_04 = __fadd_rn(combined, chain_4);
    const float sum_15 = __fadd_rn(chain_1, chain_5);
    const float sum_26 = __fadd_rn(chain_2, chain_6);
    const float sum_37 = __fadd_rn(chain_3, chain_7);
    const float sum_0246 = __fadd_rn(sum_04, sum_26);
    const float sum_1357 = __fadd_rn(sum_15, sum_37);
    return __fadd_rn(sum_0246, sum_1357);
}

extern "C" __global__ void xrt_deltanet_conv1d(
    const float* current,
    const float* committed_state,
    const float* kernel,
    float* pending_state,
    float* output,
    unsigned int channels,
    unsigned int history,
    unsigned int kernel_size) {
    const unsigned int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels) {
        return;
    }

    float sum = 0.0f;
    for (unsigned int tap = 0; tap < history; ++tap) {
        const float product = __fmul_rn(
            committed_state[tap * channels + channel],
            kernel[channel * kernel_size + tap]);
        sum = __fadd_rn(sum, product);
        if (tap + 1 < history) {
            pending_state[tap * channels + channel] =
                committed_state[(tap + 1) * channels + channel];
        }
    }
    const float current_product = __fmul_rn(
        current[channel],
        kernel[channel * kernel_size + history]);
    sum = __fadd_rn(sum, current_product);
    if (history > 0) {
        pending_state[(history - 1) * channels + channel] = current[channel];
    }
    const float exp_negative = xrt_fast_exp_cpu_order(__fsub_rn(0.0f, sum));
    output[channel] = __fdiv_rn(sum, __fadd_rn(1.0f, exp_negative));
}

extern "C" __global__ void xrt_deltanet_normalize_qk(
    float* qkv,
    unsigned int state_size,
    unsigned int group_count,
    float epsilon) {
    const unsigned int group = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (group >= group_count || lane >= 32) {
        return;
    }
    const unsigned int q_offset = group * state_size;
    const unsigned int k_offset = group_count * state_size + q_offset;
    const unsigned int vectorized = (state_size / 32) * 32;
    float q_sum = 0.0f;
    float k_sum = 0.0f;
    for (unsigned int index = lane; index < vectorized; index += 32) {
        const float q = qkv[q_offset + index];
        const float k = qkv[k_offset + index];
        q_sum = __fmaf_rn(q, q, q_sum);
        k_sum = __fmaf_rn(k, k, k_sum);
    }

    __shared__ float q_chains[32];
    __shared__ float k_chains[32];
    __shared__ float inverses[2];
    q_chains[lane] = q_sum;
    k_chains[lane] = k_sum;
    __syncthreads();

    float q_total = xrt_reduce_cpu_four_chains(q_chains, lane);
    float k_total = xrt_reduce_cpu_four_chains(k_chains, lane);
    if (lane == 0) {
        for (unsigned int index = vectorized; index < state_size; ++index) {
            const float q = qkv[q_offset + index];
            const float k = qkv[k_offset + index];
            q_total = __fmaf_rn(q, q, q_total);
            k_total = __fmaf_rn(k, k, k_total);
        }
        const float q_root = __fsqrt_rn(__fadd_rn(q_total, epsilon));
        const float k_root = __fsqrt_rn(__fadd_rn(k_total, epsilon));
        inverses[0] = __fdiv_rn(1.0f, q_root);
        inverses[1] = __fdiv_rn(1.0f, k_root);
    }
    __syncthreads();

    for (unsigned int index = lane; index < state_size; index += 32) {
        qkv[q_offset + index] = __fmul_rn(qkv[q_offset + index], inverses[0]);
        qkv[k_offset + index] = __fmul_rn(qkv[k_offset + index], inverses[1]);
    }
}

extern "C" __global__ void xrt_deltanet_decay_beta(
    const float* alpha,
    const float* beta,
    const float* a,
    const float* dt_bias,
    float* decays,
    float* betas,
    unsigned int value_heads) {
    const unsigned int head = blockIdx.x * blockDim.x + threadIdx.x;
    if (head >= value_heads) {
        return;
    }
    const float alpha_biased = alpha[head] + dt_bias[head];
    const float softplus = log1pf(expf(alpha_biased));
    decays[head] = expf(softplus * a[head]);
    betas[head] = 1.0f / (1.0f + expf(-beta[head]));
}

extern "C" __global__ void xrt_deltanet_update(
    const float* qkv,
    const float* committed_state,
    const float* decays,
    const float* betas,
    float* pending_state,
    float* output,
    unsigned int state_size,
    unsigned int group_count,
    unsigned int value_heads,
    unsigned int head_value_size,
    float query_scale) {
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int output_index = blockIdx.x * (blockDim.x >> 5) + warp;
    const unsigned int inner_size = value_heads * head_value_size;
    if (output_index >= inner_size) {
        return;
    }
    const unsigned int value_head = output_index / head_value_size;
    const unsigned int value_index = output_index % head_value_size;
    const unsigned int qk_group = (value_head * group_count) / value_heads;
    const unsigned int q_offset = qk_group * state_size;
    const unsigned int k_offset = group_count * state_size + q_offset;
    const unsigned int value_offset =
        2 * group_count * state_size + value_head * head_value_size + value_index;
    const unsigned int state_offset =
        (value_head * head_value_size + value_index) * state_size;

    const unsigned int vectorized = (state_size / 32) * 32;
    float state_key = 0.0f;
    for (unsigned int index = lane; index < vectorized; index += 32) {
        state_key = __fmaf_rn(
            committed_state[state_offset + index],
            qkv[k_offset + index],
            state_key);
    }

    float reduced_state_key =
        xrt_reduce_cpu_four_chains_warp(state_key, lane);
    float delta = 0.0f;
    if (lane == 0) {
        for (unsigned int index = vectorized; index < state_size; ++index) {
            reduced_state_key = __fmaf_rn(
                committed_state[state_offset + index],
                qkv[k_offset + index],
                reduced_state_key);
        }
        reduced_state_key =
            __fmul_rn(reduced_state_key, decays[value_head]);
        delta = __fmul_rn(
            betas[value_head],
            __fsub_rn(qkv[value_offset], reduced_state_key));
    }
    delta = __shfl_sync(0xffffffffu, delta, 0);
    const float decay = decays[value_head];
    float projected = 0.0f;
    for (unsigned int index = lane; index < vectorized; index += 32) {
        const float delta_key = __fmul_rn(delta, qkv[k_offset + index]);
        const float next = __fmaf_rn(
            decay,
            committed_state[state_offset + index],
            delta_key);
        pending_state[state_offset + index] = next;
        projected = __fmaf_rn(next, qkv[q_offset + index], projected);
    }
    float reduced_projected =
        xrt_reduce_cpu_four_chains_warp(projected, lane);
    if (lane == 0) {
        float remainder = 0.0f;
        for (unsigned int index = vectorized; index < state_size; ++index) {
            const float next = __fadd_rn(
                __fmul_rn(decay, committed_state[state_offset + index]),
                __fmul_rn(delta, qkv[k_offset + index]));
            pending_state[state_offset + index] = next;
            remainder =
                __fadd_rn(remainder, __fmul_rn(next, qkv[q_offset + index]));
        }
        reduced_projected = __fadd_rn(reduced_projected, remainder);
        output[output_index] = __fmul_rn(reduced_projected, query_scale);
    }
}

extern "C" __global__ void xrt_deltanet_gated_rmsnorm(
    float* output,
    const float* gate,
    const float* norm_weight,
    unsigned int value_heads,
    unsigned int head_value_size,
    float epsilon) {
    const unsigned int head = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (head >= value_heads || lane >= 16) {
        return;
    }
    const unsigned int head_offset = head * head_value_size;
    const unsigned int vectorized = (head_value_size / 16) * 16;
    float sum = 0.0f;
    for (unsigned int index = lane; index < vectorized; index += 16) {
        const float value = output[head_offset + index];
        sum = __fmaf_rn(value, value, sum);
    }

    __shared__ float chains[16];
    __shared__ float inverse_rms;
    chains[lane] = sum;
    __syncthreads();
    float total = xrt_reduce_cpu_two_chains(chains, lane);
    if (lane == 0) {
        for (unsigned int index = vectorized; index < head_value_size; ++index) {
            const float value = output[head_offset + index];
            total = __fmaf_rn(value, value, total);
        }
        const float mean =
            __fdiv_rn(total, static_cast<float>(head_value_size));
        const float root = __fsqrt_rn(__fadd_rn(mean, epsilon));
        inverse_rms = __fdiv_rn(1.0f, root);
    }
    __syncthreads();

    for (unsigned int local = lane; local < head_value_size; local += 16) {
        const unsigned int output_index = head_offset + local;
        const float gate_value = gate[output_index];
        const float exp_negative =
            xrt_fast_exp_cpu_order(__fsub_rn(0.0f, gate_value));
        const float silu_gate =
            __fdiv_rn(gate_value, __fadd_rn(1.0f, exp_negative));
        float normalized = __fmul_rn(output[output_index], inverse_rms);
        normalized = __fmul_rn(normalized, norm_weight[local]);
        output[output_index] = __fmul_rn(normalized, silu_gate);
    }
}

extern "C" __global__ void xrt_qwen35_deinterleave_qg(
    const float* qg,
    float* query,
    float* gate,
    unsigned int head_count,
    unsigned int head_size) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total = head_count * head_size;
    if (index >= total) {
        return;
    }
    const unsigned int head = index / head_size;
    const unsigned int local = index % head_size;
    const unsigned int source = head * head_size * 2 + local;
    query[index] = qg[source];
    gate[index] = qg[source + head_size];
}

extern "C" __global__ void xrt_sigmoid_mul(
    float* values,
    const float* gate,
    unsigned int length) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= length) {
        return;
    }
    values[index] *= 1.0f / (1.0f + expf(-gate[index]));
}
