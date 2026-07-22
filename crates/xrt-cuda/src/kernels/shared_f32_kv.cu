// Shared-page F32 KV kernels.
//
// Each page table entry is a stable CUDA device pointer to one independently
// owned page. Host-side Arc ownership permits exact prefix forks to share
// immutable pages; a session replaces only the page it is about to mutate.
// Kernels must never retain a page pointer beyond their launch lifetime.

using xrt_u64 = unsigned long long;

__device__ __forceinline__ float* xrt_f32_page(
    const xrt_u64* pages,
    unsigned int logical_page
) {
    return reinterpret_cast<float*>(pages[logical_page]);
}

extern "C" __global__ void shared_f32_kv_append_kernel(
    const xrt_u64* key_pages,
    const xrt_u64* value_pages,
    const float* key,
    const float* value,
    unsigned int slot,
    unsigned int width,
    unsigned int page_tokens,
    const unsigned int* decode_params
) {
    if (decode_params != nullptr) {
        slot = decode_params[1];
    }
    const unsigned int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= width) {
        return;
    }
    const unsigned int logical_page = slot / page_tokens;
    const unsigned int page_row = slot % page_tokens;
    const unsigned int page_offset = page_row * width + column;
    xrt_f32_page(key_pages, logical_page)[page_offset] = key[column];
    xrt_f32_page(value_pages, logical_page)[page_offset] = value[column];
}

extern "C" __global__ void shared_f32_kv_gather_kernel(
    const xrt_u64* key_pages,
    const xrt_u64* value_pages,
    float* keys,
    float* values,
    unsigned int count,
    unsigned int width,
    unsigned int page_tokens,
    unsigned int start_position
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int element_count = count * width;
    if (index >= element_count) {
        return;
    }
    const unsigned int row = index / width;
    const unsigned int column = index - row * width;
    const unsigned int position = start_position + row;
    const unsigned int logical_page = position / page_tokens;
    const unsigned int page_row = position % page_tokens;
    const unsigned int page_offset = page_row * width + column;
    keys[index] = xrt_f32_page(key_pages, logical_page)[page_offset];
    values[index] = xrt_f32_page(value_pages, logical_page)[page_offset];
}

extern "C" __global__ void shared_f32_single_query_attention_online_kernel(
    const float* query,
    const xrt_u64* key_pages,
    const xrt_u64* value_pages,
    float* output,
    unsigned int head_count,
    unsigned int kv_head_count,
    unsigned int head_dim,
    unsigned int cache_len,
    float scale,
    unsigned int page_tokens,
    unsigned int attend_start,
    const unsigned int* decode_params
) {
    __shared__ float reduction[512];
    __shared__ float state[4];

    if (decode_params != nullptr) {
        cache_len = decode_params[2];
        attend_start = decode_params[3];
    }

    const unsigned int lane = threadIdx.x;
    const unsigned int head = blockIdx.x;
    if (head >= head_count) {
        return;
    }
    const unsigned int query_heads_per_kv = head_count / kv_head_count;
    const unsigned int kv_head = head / query_heads_per_kv;
    const unsigned int kv_width = kv_head_count * head_dim;
    float accumulator = 0.0f;

    if (lane == 0) {
        state[0] = -__int_as_float(0x7f800000);
        state[1] = 0.0f;
    }
    __syncthreads();

    for (unsigned int position = attend_start; position < cache_len; ++position) {
        float partial = 0.0f;
        const unsigned int logical_page = position / page_tokens;
        const unsigned int page_row = position % page_tokens;
        const unsigned int row_offset = page_row * kv_width + kv_head * head_dim;
        if (lane < head_dim) {
            partial = query[head * head_dim + lane]
                * xrt_f32_page(key_pages, logical_page)[row_offset + lane];
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
            const float token_value =
                xrt_f32_page(value_pages, logical_page)[row_offset + lane];
            accumulator = accumulator * state[2] + token_value * state[3];
        }
        __syncthreads();
    }

    if (lane < head_dim) {
        output[head * head_dim + lane] = accumulator / state[1];
    }
}

extern "C" __global__ void shared_f32_single_query_attention_kernel(
    const float* query,
    const xrt_u64* key_pages,
    const xrt_u64* value_pages,
    float* output,
    unsigned int head_count,
    unsigned int kv_head_count,
    unsigned int head_dim,
    unsigned int cache_len,
    unsigned int output_len,
    float scale,
    unsigned int page_tokens,
    unsigned int attend_start
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= output_len) {
        return;
    }
    const unsigned int head = index / head_dim;
    const unsigned int column = index - head * head_dim;
    const unsigned int query_heads_per_kv = head_count / kv_head_count;
    const unsigned int kv_head = head / query_heads_per_kv;
    const unsigned int kv_width = kv_head_count * head_dim;

    float max_score = -__int_as_float(0x7f800000);
    for (unsigned int position = attend_start; position < cache_len; ++position) {
        const unsigned int logical_page = position / page_tokens;
        const unsigned int page_row = position % page_tokens;
        const unsigned int row_offset = page_row * kv_width + kv_head * head_dim;
        float score = 0.0f;
        for (unsigned int dim = 0; dim < head_dim; ++dim) {
            score += query[head * head_dim + dim]
                * xrt_f32_page(key_pages, logical_page)[row_offset + dim];
        }
        max_score = fmaxf(max_score, score * scale);
    }

    float denominator = 0.0f;
    float weighted_value = 0.0f;
    for (unsigned int position = attend_start; position < cache_len; ++position) {
        const unsigned int logical_page = position / page_tokens;
        const unsigned int page_row = position % page_tokens;
        const unsigned int row_offset = page_row * kv_width + kv_head * head_dim;
        float score = 0.0f;
        for (unsigned int dim = 0; dim < head_dim; ++dim) {
            score += query[head * head_dim + dim]
                * xrt_f32_page(key_pages, logical_page)[row_offset + dim];
        }
        const float weight = __expf(score * scale - max_score);
        denominator += weight;
        weighted_value +=
            weight * xrt_f32_page(value_pages, logical_page)[row_offset + column];
    }
    output[index] = weighted_value / denominator;
}
