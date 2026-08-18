// Batched Qwen3.5/Qwen3.6 full-attention verification kernels.
//
// Speculative verification owns a short causal suffix (currently at most 16
// rows).  Processing that suffix one row at a time turns normalization, RoPE,
// KV append, attention, and gating into hundreds of tiny launches per verify
// window.  These kernels retain the decode math and accumulation order while
// making the suffix row an explicit grid dimension.

using xrt_u64 = unsigned long long;

__device__ __forceinline__ float* xrt_f32_page(
    const xrt_u64* pages,
    unsigned int logical_page) {
    return reinterpret_cast<float*>(pages[logical_page]);
}

__device__ __forceinline__ float xrt_qwen35_rope_theta(
    unsigned int pair,
    unsigned int rotary_width,
    unsigned int position,
    float base,
    float scale) {
    const float exponent = -(2.0f * static_cast<float>(pair)) /
        static_cast<float>(rotary_width);
    const float frequency = exp2f(exponent * log2f(base));
    return static_cast<float>(position) * scale * frequency;
}

extern "C" __global__ void xrt_qwen35_verify_prepare(
    const float* qg,
    const float* key,
    const float* q_norm_weight,
    const float* k_norm_weight,
    float* query_output,
    float* gate_output,
    float* key_output,
    unsigned int batch_rows,
    unsigned int query_heads,
    unsigned int kv_heads,
    unsigned int head_dim,
    unsigned int rope_dim,
    const unsigned int* decode_params,
    float epsilon,
    float rope_base,
    float rope_scale) {
    const unsigned int start_position = decode_params[1];
    const unsigned int heads_per_row = query_heads + kv_heads;
    const unsigned int work = blockIdx.x;
    const unsigned int row = work / heads_per_row;
    const unsigned int work_head = work - row * heads_per_row;
    const unsigned int thread = threadIdx.x;
    if (row >= batch_rows || thread >= 256) {
        return;
    }

    const bool is_query = work_head < query_heads;
    const unsigned int head = is_query ? work_head : work_head - query_heads;
    const unsigned int head_count = is_query ? query_heads : kv_heads;
    const float* norm_weight = is_query ? q_norm_weight : k_norm_weight;
    const unsigned int row_width = head_count * head_dim;
    const unsigned int head_offset = row * row_width + head * head_dim;

    __shared__ float chains[8];
    __shared__ float inverse_rms;
    const unsigned int grouped_cols = (head_dim / 8) * 8;
    if (thread < 8) {
        float accumulator = 0.0f;
        for (unsigned int local = thread; local < grouped_cols; local += 8) {
            const float value = is_query
                ? qg[row * query_heads * head_dim * 2 +
                     head * head_dim * 2 + local]
                : key[head_offset + local];
            accumulator = __fmaf_rn(value, value, accumulator);
        }
        chains[thread] = accumulator;
    }
    __syncthreads();

    if (thread == 0) {
        float sum = __fadd_rn(chains[0], chains[1]);
        sum = __fadd_rn(sum, chains[2]);
        sum = __fadd_rn(sum, chains[3]);
        sum = __fadd_rn(sum, chains[4]);
        sum = __fadd_rn(sum, chains[5]);
        sum = __fadd_rn(sum, chains[6]);
        sum = __fadd_rn(sum, chains[7]);
        for (unsigned int local = grouped_cols; local < head_dim; ++local) {
            const float value = is_query
                ? qg[row * query_heads * head_dim * 2 +
                     head * head_dim * 2 + local]
                : key[head_offset + local];
            sum = __fmaf_rn(value, value, sum);
        }
        const float mean = __fdiv_rn(sum, static_cast<float>(head_dim));
        inverse_rms = __fdiv_rn(
            1.0f,
            __fsqrt_rn(__fadd_rn(mean, epsilon)));
    }
    __syncthreads();

    if (thread < head_dim) {
        if (is_query) {
            const unsigned int source = row * query_heads * head_dim * 2 +
                head * head_dim * 2 + thread;
            const float normalized = __fmul_rn(qg[source], inverse_rms);
            query_output[head_offset + thread] =
                __fmul_rn(normalized, norm_weight[thread]);
            gate_output[head_offset + thread] = qg[source + head_dim];
        } else {
            const float normalized = __fmul_rn(key[head_offset + thread], inverse_rms);
            key_output[head_offset + thread] =
                __fmul_rn(normalized, norm_weight[thread]);
        }
    }
    __syncthreads();

    const unsigned int rotary_width = rope_dim < head_dim ? rope_dim : head_dim;
    const unsigned int half_width = rotary_width / 2;
    if (thread < half_width) {
        float* output = is_query ? query_output : key_output;
        const unsigned int first = head_offset + thread;
        const unsigned int second = first + half_width;
        const float first_value = output[first];
        const float second_value = output[second];
        const float theta = xrt_qwen35_rope_theta(
            thread,
            rotary_width,
            start_position + row,
            rope_base,
            rope_scale);
        const float sine = __sinf(theta);
        const float cosine = __cosf(theta);
        output[first] = first_value * cosine - second_value * sine;
        output[second] = first_value * sine + second_value * cosine;
    }
}

// Tree verification preserves the same normalization and RoPE arithmetic as
// the linear kernel, but logical positions follow tree depth instead of the
// physical row used to stage fixed-budget KV entries.
extern "C" __global__ void xrt_qwen35_verify_prepare_tree(
    const float* qg,
    const float* key,
    const float* q_norm_weight,
    const float* k_norm_weight,
    const float* tree_depths,
    float* query_output,
    float* gate_output,
    float* key_output,
    unsigned int batch_rows,
    unsigned int query_heads,
    unsigned int kv_heads,
    unsigned int head_dim,
    unsigned int rope_dim,
    const unsigned int* decode_params,
    float epsilon,
    float rope_base,
    float rope_scale) {
    const unsigned int start_position = decode_params[1];
    const unsigned int heads_per_row = query_heads + kv_heads;
    const unsigned int work = blockIdx.x;
    const unsigned int row = work / heads_per_row;
    const unsigned int work_head = work - row * heads_per_row;
    const unsigned int thread = threadIdx.x;
    if (row >= batch_rows || thread >= 256) {
        return;
    }

    const bool is_query = work_head < query_heads;
    const unsigned int head = is_query ? work_head : work_head - query_heads;
    const unsigned int head_count = is_query ? query_heads : kv_heads;
    const float* norm_weight = is_query ? q_norm_weight : k_norm_weight;
    const unsigned int row_width = head_count * head_dim;
    const unsigned int head_offset = row * row_width + head * head_dim;

    __shared__ float chains[8];
    __shared__ float inverse_rms;
    const unsigned int grouped_cols = (head_dim / 8) * 8;
    if (thread < 8) {
        float accumulator = 0.0f;
        for (unsigned int local = thread; local < grouped_cols; local += 8) {
            const float value = is_query
                ? qg[row * query_heads * head_dim * 2 +
                     head * head_dim * 2 + local]
                : key[head_offset + local];
            accumulator = __fmaf_rn(value, value, accumulator);
        }
        chains[thread] = accumulator;
    }
    __syncthreads();

    if (thread == 0) {
        float sum = __fadd_rn(chains[0], chains[1]);
        sum = __fadd_rn(sum, chains[2]);
        sum = __fadd_rn(sum, chains[3]);
        sum = __fadd_rn(sum, chains[4]);
        sum = __fadd_rn(sum, chains[5]);
        sum = __fadd_rn(sum, chains[6]);
        sum = __fadd_rn(sum, chains[7]);
        for (unsigned int local = grouped_cols; local < head_dim; ++local) {
            const float value = is_query
                ? qg[row * query_heads * head_dim * 2 +
                     head * head_dim * 2 + local]
                : key[head_offset + local];
            sum = __fmaf_rn(value, value, sum);
        }
        const float mean = __fdiv_rn(sum, static_cast<float>(head_dim));
        inverse_rms = __fdiv_rn(
            1.0f,
            __fsqrt_rn(__fadd_rn(mean, epsilon)));
    }
    __syncthreads();

    if (thread < head_dim) {
        if (is_query) {
            const unsigned int source = row * query_heads * head_dim * 2 +
                head * head_dim * 2 + thread;
            const float normalized = __fmul_rn(qg[source], inverse_rms);
            query_output[head_offset + thread] =
                __fmul_rn(normalized, norm_weight[thread]);
            gate_output[head_offset + thread] = qg[source + head_dim];
        } else {
            const float normalized = __fmul_rn(key[head_offset + thread], inverse_rms);
            key_output[head_offset + thread] =
                __fmul_rn(normalized, norm_weight[thread]);
        }
    }
    __syncthreads();

    const unsigned int rotary_width = rope_dim < head_dim ? rope_dim : head_dim;
    const unsigned int half_width = rotary_width / 2;
    if (thread < half_width) {
        float* output = is_query ? query_output : key_output;
        const unsigned int first = head_offset + thread;
        const unsigned int second = first + half_width;
        const float first_value = output[first];
        const float second_value = output[second];
        const unsigned int logical_depth = static_cast<unsigned int>(tree_depths[row]);
        const float theta = xrt_qwen35_rope_theta(
            thread,
            rotary_width,
            start_position + logical_depth,
            rope_base,
            rope_scale);
        const float sine = __sinf(theta);
        const float cosine = __cosf(theta);
        output[first] = first_value * cosine - second_value * sine;
        output[second] = first_value * sine + second_value * cosine;
    }
}

extern "C" __global__ void xrt_qwen35_verify_append_paged_f32(
    float* keys,
    float* values,
    const unsigned int* page_table,
    const float* key,
    const float* value,
    unsigned int batch_rows,
    unsigned int width,
    unsigned int page_tokens,
    const unsigned int* decode_params) {
    const unsigned int start_position = decode_params[1];
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int element_count = batch_rows * width;
    if (index >= element_count) {
        return;
    }
    const unsigned int row = index / width;
    const unsigned int column = index - row * width;
    const unsigned int position = start_position + row;
    const unsigned int logical_page = position / page_tokens;
    const unsigned int physical_page = page_table[logical_page];
    const unsigned int page_row = position % page_tokens;
    const unsigned int destination =
        (physical_page * page_tokens + page_row) * width + column;
    keys[destination] = key[index];
    values[destination] = value[index];
}

extern "C" __global__ void xrt_qwen35_verify_append_shared_f32(
    const xrt_u64* key_pages,
    const xrt_u64* value_pages,
    const float* key,
    const float* value,
    unsigned int batch_rows,
    unsigned int width,
    unsigned int page_tokens,
    const unsigned int* decode_params) {
    const unsigned int start_position = decode_params[1];
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int element_count = batch_rows * width;
    if (index >= element_count) {
        return;
    }
    const unsigned int row = index / width;
    const unsigned int column = index - row * width;
    const unsigned int position = start_position + row;
    const unsigned int logical_page = position / page_tokens;
    const unsigned int page_row = position % page_tokens;
    const unsigned int page_offset = page_row * width + column;
    xrt_f32_page(key_pages, logical_page)[page_offset] = key[index];
    xrt_f32_page(value_pages, logical_page)[page_offset] = value[index];
}

template <bool SharedPages>
__device__ __forceinline__ float xrt_qwen35_verify_cache_value(
    const float* contiguous,
    const xrt_u64* shared,
    const unsigned int* page_table,
    unsigned int position,
    unsigned int column,
    unsigned int width,
    unsigned int page_tokens) {
    const unsigned int logical_page = position / page_tokens;
    const unsigned int page_row = position % page_tokens;
    if (SharedPages) {
        return xrt_f32_page(shared, logical_page)[page_row * width + column];
    }
    const unsigned int physical_page = page_table[logical_page];
    return contiguous[(physical_page * page_tokens + page_row) * width + column];
}

template <bool SharedPages>
__device__ void xrt_qwen35_verify_attention_impl(
    float* query_and_output,
    const float* gate,
    const float* keys,
    const float* values,
    const xrt_u64* key_pages,
    const xrt_u64* value_pages,
    const unsigned int* page_table,
    unsigned int batch_rows,
    unsigned int query_heads,
    unsigned int kv_heads,
    unsigned int head_dim,
    const unsigned int* decode_params,
    unsigned int kv_width,
    unsigned int page_tokens,
    unsigned int rows_per_block,
    const float* tree_visibility,
    float scale) {
    const unsigned int start_position = decode_params[1];
    __shared__ float reduction[512];
    __shared__ float state[4];

    const unsigned int head = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (head >= query_heads) {
        return;
    }
    const unsigned int query_heads_per_kv = query_heads / kv_heads;
    const unsigned int kv_head = head / query_heads_per_kv;
    const unsigned int query_width = query_heads * head_dim;
    const unsigned int first_row = blockIdx.y * rows_per_block;
    const unsigned int last_row = min(first_row + rows_per_block, batch_rows);
    for (unsigned int row = first_row; row < last_row; ++row) {
        const unsigned int query_offset = row * query_width + head * head_dim;
        const unsigned int cache_len = start_position + row + 1;
        float accumulator = 0.0f;

        if (lane == 0) {
            state[0] = -__int_as_float(0x7f800000);
            state[1] = 0.0f;
        }
        __syncthreads();

        for (unsigned int position = 0; position < cache_len; ++position) {
            if (tree_visibility != nullptr && position >= start_position) {
                const unsigned int local_position = position - start_position;
                const unsigned int visibility =
                    static_cast<unsigned int>(tree_visibility[row]);
                if ((visibility & (1u << local_position)) == 0) {
                    continue;
                }
            }
            float partial = 0.0f;
            const unsigned int column = kv_head * head_dim + lane;
            if (lane < head_dim) {
                partial = query_and_output[query_offset + lane] *
                    xrt_qwen35_verify_cache_value<SharedPages>(
                        keys,
                        key_pages,
                        page_table,
                        position,
                        column,
                        kv_width,
                        page_tokens);
            }
            reduction[lane] = partial;
            __syncthreads();

            for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (lane < stride) {
                    reduction[lane] += reduction[lane + stride];
                }
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

            if (lane < head_dim) {
                const unsigned int column = kv_head * head_dim + lane;
                const float token_value = xrt_qwen35_verify_cache_value<SharedPages>(
                    values,
                    value_pages,
                    page_table,
                    position,
                    column,
                    kv_width,
                    page_tokens);
                accumulator = accumulator * state[2] + token_value * state[3];
            }
            __syncthreads();
        }

        if (lane < head_dim) {
            const unsigned int output_index = query_offset + lane;
            const float attention = accumulator / state[1];
            const float sigmoid = 1.0f / (1.0f + __expf(-gate[output_index]));
            query_and_output[output_index] = attention * sigmoid;
        }
        __syncthreads();
    }
}

extern "C" __global__ void xrt_qwen35_verify_attention_paged_f32(
    float* query_and_output,
    const float* gate,
    const float* keys,
    const float* values,
    const unsigned int* page_table,
    unsigned int batch_rows,
    unsigned int query_heads,
    unsigned int kv_heads,
    unsigned int head_dim,
    const unsigned int* decode_params,
    unsigned int kv_width,
    unsigned int page_tokens,
    unsigned int rows_per_block,
    float scale) {
    xrt_qwen35_verify_attention_impl<false>(
        query_and_output,
        gate,
        keys,
        values,
        nullptr,
        nullptr,
        page_table,
        batch_rows,
        query_heads,
        kv_heads,
        head_dim,
        decode_params,
        kv_width,
        page_tokens,
        rows_per_block,
        nullptr,
        scale);
}

extern "C" __global__ void xrt_qwen35_verify_attention_shared_f32(
    float* query_and_output,
    const float* gate,
    const xrt_u64* key_pages,
    const xrt_u64* value_pages,
    unsigned int batch_rows,
    unsigned int query_heads,
    unsigned int kv_heads,
    unsigned int head_dim,
    const unsigned int* decode_params,
    unsigned int kv_width,
    unsigned int page_tokens,
    unsigned int rows_per_block,
    float scale) {
    xrt_qwen35_verify_attention_impl<true>(
        query_and_output,
        gate,
        nullptr,
        nullptr,
        key_pages,
        value_pages,
        nullptr,
        batch_rows,
        query_heads,
        kv_heads,
        head_dim,
        decode_params,
        kv_width,
        page_tokens,
        rows_per_block,
        nullptr,
        scale);
}

extern "C" __global__ void xrt_qwen35_verify_attention_tree_paged_f32(
    float* query_and_output,
    const float* gate,
    const float* keys,
    const float* values,
    const unsigned int* page_table,
    const float* tree_visibility,
    unsigned int batch_rows,
    unsigned int query_heads,
    unsigned int kv_heads,
    unsigned int head_dim,
    const unsigned int* decode_params,
    unsigned int kv_width,
    unsigned int page_tokens,
    unsigned int rows_per_block,
    float scale) {
    xrt_qwen35_verify_attention_impl<false>(
        query_and_output,
        gate,
        keys,
        values,
        nullptr,
        nullptr,
        page_table,
        batch_rows,
        query_heads,
        kv_heads,
        head_dim,
        decode_params,
        kv_width,
        page_tokens,
        rows_per_block,
        tree_visibility,
        scale);
}

extern "C" __global__ void xrt_qwen35_verify_attention_tree_shared_f32(
    float* query_and_output,
    const float* gate,
    const xrt_u64* key_pages,
    const xrt_u64* value_pages,
    const float* tree_visibility,
    unsigned int batch_rows,
    unsigned int query_heads,
    unsigned int kv_heads,
    unsigned int head_dim,
    const unsigned int* decode_params,
    unsigned int kv_width,
    unsigned int page_tokens,
    unsigned int rows_per_block,
    float scale) {
    xrt_qwen35_verify_attention_impl<true>(
        query_and_output,
        gate,
        nullptr,
        nullptr,
        key_pages,
        value_pages,
        nullptr,
        batch_rows,
        query_heads,
        kv_heads,
        head_dim,
        decode_params,
        kv_width,
        page_tokens,
        rows_per_block,
        tree_visibility,
        scale);
}
