// Qwen3.6 DFlash block-drafter support kernels.
//
// These kernels intentionally own only the operations that do not map onto
// XRT's existing resident linear/RMSNorm/SwiGLU primitives: feature taps,
// ring-cache writes, per-head RMSNorm+RoPE, and bidirectional block attention.

__device__ __forceinline__ float xrt_dflash_rope_theta(
    unsigned int pair,
    unsigned int rotary_width,
    unsigned int position,
    float base) {
    const float exponent = -(2.0f * static_cast<float>(pair)) /
        static_cast<float>(rotary_width);
    return static_cast<float>(position) * exp2f(exponent * log2f(base));
}

extern "C" __global__ void xrt_dflash_capture_features(
    const float* source,
    float* destination,
    unsigned int rows,
    unsigned int hidden,
    unsigned int feature_width,
    unsigned int capture_index) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int elements = rows * hidden;
    if (index >= elements) return;
    const unsigned int row = index / hidden;
    const unsigned int column = index - row * hidden;
    // The reference target feature ring is BF16. Preserve that numerical
    // contract while retaining an F32 resident buffer for the drafter: round
    // to BF16 (ties-to-even), then widen back to F32 before feature fusion.
    unsigned int bits = __float_as_uint(source[index]);
    const unsigned int exponent = bits & 0x7f800000u;
    if (exponent != 0x7f800000u) {
        bits += 0x00007fffu + ((bits >> 16) & 1u);
    }
    bits &= 0xffff0000u;
    destination[row * feature_width + capture_index * hidden + column] = __uint_as_float(bits);
}

extern "C" __global__ void xrt_dflash_store_ring_rows(
    const float* source,
    float* destination,
    unsigned int rows,
    unsigned int width,
    unsigned int start_position,
    unsigned int capacity) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int elements = rows * width;
    if (index >= elements) return;
    const unsigned int row = index / width;
    const unsigned int column = index - row * width;
    const unsigned int slot = (start_position + row) % capacity;
    destination[slot * width + column] = source[index];
}

// DFlash evaluates a fixed block of at most 16 rows.  The generic Q8_0
// matvec path assigns a separate CUDA block to every (activation, weight-row)
// pair, which rereads the same packed weight matrix up to sixteen times.  This
// kernel assigns one block to a weight row and accumulates all activation rows
// while each packed weight is resident in a register.  The per-row reduction
// order intentionally matches the scalar Q8_0 kernel.
extern "C" __global__ void xrt_dflash_q8_0_batch16(
    const float* scales,
    const signed char* quants,
    const float* input,
    float* output,
    unsigned int rows,
    unsigned int cols,
    unsigned int batch_rows) {
    const unsigned int weight_row = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (weight_row >= rows || batch_rows == 0 || batch_rows > 16) return;

    const unsigned int blocks_per_row = cols / 32;
    const unsigned int row_block_offset = weight_row * blocks_per_row;
    float accumulators[16] = {0.0f};
    for (unsigned int column = lane; column < cols; column += blockDim.x) {
        const unsigned int block = column >> 5;
        const unsigned int within_block = column & 31;
        const unsigned int packed_block = row_block_offset + block;
        const float weight = scales[packed_block] *
            static_cast<float>(quants[packed_block * 32 + within_block]);
#pragma unroll
        for (unsigned int batch = 0; batch < 16; ++batch) {
            if (batch < batch_rows) {
                accumulators[batch] = fmaf(
                    weight, input[batch * cols + column], accumulators[batch]);
            }
        }
    }

    __shared__ float reduction[16][256];
#pragma unroll
    for (unsigned int batch = 0; batch < 16; ++batch) {
        reduction[batch][lane] = accumulators[batch];
    }
    __syncthreads();
    for (unsigned int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
#pragma unroll
            for (unsigned int batch = 0; batch < 16; ++batch) {
                reduction[batch][lane] += reduction[batch][lane + stride];
            }
        }
        __syncthreads();
    }
    if (lane == 0) {
#pragma unroll
        for (unsigned int batch = 0; batch < 16; ++batch) {
            if (batch < batch_rows) {
                output[batch * rows + weight_row] = reduction[batch][0];
            }
        }
    }
}

extern "C" __global__ void xrt_dflash_norm_rope(
    float* values,
    const float* norm_weight,
    unsigned int rows,
    unsigned int heads,
    unsigned int head_dim,
    unsigned int start_position,
    unsigned int rope_dim,
    float epsilon,
    float rope_base) {
    const unsigned int work = blockIdx.x;
    const unsigned int row = work / heads;
    const unsigned int head = work - row * heads;
    const unsigned int lane = threadIdx.x;
    if (row >= rows || head >= heads) return;

    const unsigned int offset = (row * heads + head) * head_dim;
    __shared__ float reduction[128];
    float square = 0.0f;
    if (lane < head_dim) {
        const float value = values[offset + lane];
        square = value * value;
    }
    reduction[lane] = square;
    __syncthreads();
    for (unsigned int stride = 64; stride > 0; stride >>= 1) {
        if (lane < stride) reduction[lane] += reduction[lane + stride];
        __syncthreads();
    }
    const float inverse_rms = rsqrtf(reduction[0] / static_cast<float>(head_dim) + epsilon);
    if (lane < head_dim) {
        values[offset + lane] = values[offset + lane] * inverse_rms * norm_weight[lane];
    }
    __syncthreads();

    const unsigned int rotary_width = min(rope_dim, head_dim);
    const unsigned int half_width = rotary_width / 2;
    if (lane < half_width) {
        const unsigned int first = offset + lane;
        const unsigned int second = first + half_width;
        const float x0 = values[first];
        const float x1 = values[second];
        const float theta = xrt_dflash_rope_theta(
            lane, rotary_width, start_position + row, rope_base);
        const float sine = __sinf(theta);
        const float cosine = __cosf(theta);
        values[first] = x0 * cosine - x1 * sine;
        values[second] = x0 * sine + x1 * cosine;
    }
}

extern "C" __global__ void xrt_dflash_block_attention(
    const float* query,
    const float* cached_key,
    const float* cached_value,
    const float* noise_key,
    const float* noise_value,
    float* output,
    unsigned int query_rows,
    unsigned int query_heads,
    unsigned int kv_heads,
    unsigned int head_dim,
    unsigned int context_rows,
    unsigned int context_start,
    unsigned int cache_capacity,
    unsigned int causal_noise,
    float scale) {
    const unsigned int head = blockIdx.x;
    const unsigned int query_row = blockIdx.y;
    const unsigned int lane = threadIdx.x;
    if (head >= query_heads || query_row >= query_rows) return;

    const unsigned int kv_head = head / (query_heads / kv_heads);
    const unsigned int q_width = query_heads * head_dim;
    const unsigned int kv_width = kv_heads * head_dim;
    const unsigned int q_offset = query_row * q_width + head * head_dim;
    __shared__ float reduction[128];
    __shared__ float state[4];
    float accumulator = 0.0f;
    if (lane == 0) {
        state[0] = -__int_as_float(0x7f800000);
        state[1] = 0.0f;
    }
    __syncthreads();

    // The first four Qwen3.6 DFlash layers use causal attention within the
    // 16-token noise block; the final full-attention layer sees the entire
    // block. Context features remain visible to every query in both modes.
    const unsigned int visible_noise_rows = causal_noise ? query_row + 1 : query_rows;
    const unsigned int total_rows = context_rows + visible_noise_rows;
    for (unsigned int position = 0; position < total_rows; ++position) {
        const bool cached = position < context_rows;
        const unsigned int row = cached ? position : position - context_rows;
        const unsigned int slot = (context_start + row) % cache_capacity;
        const unsigned int kv_offset =
            (cached ? slot : row) * kv_width + kv_head * head_dim;
        const float key = cached ? cached_key[kv_offset + lane] : noise_key[kv_offset + lane];
        reduction[lane] = query[q_offset + lane] * key;
        __syncthreads();
        for (unsigned int stride = 64; stride > 0; stride >>= 1) {
            if (lane < stride) reduction[lane] += reduction[lane + stride];
            __syncthreads();
        }
        if (lane == 0) {
            const float score = reduction[0] * scale;
            const float old_max = state[0];
            const float next_max = fmaxf(old_max, score);
            const float old_scale = __expf(old_max - next_max);
            const float token_scale = __expf(score - next_max);
            state[0] = next_max;
            state[1] = state[1] * old_scale + token_scale;
            state[2] = old_scale;
            state[3] = token_scale;
        }
        __syncthreads();
        const float value =
            cached ? cached_value[kv_offset + lane] : noise_value[kv_offset + lane];
        accumulator = accumulator * state[2] + value * state[3];
        __syncthreads();
    }
    output[q_offset + lane] = accumulator / state[1];
}
