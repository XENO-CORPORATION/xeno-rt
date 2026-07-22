// CUDA primitives shared by native image transformer adapters.
//
// These kernels intentionally use F32 activations. Quantized/BF16 component
// storage is converted or decoded by the resident matrix operators while the
// image graph keeps a single, explicit activation type for numerical audits.

extern "C" __global__ void xrt_image_bias_add(
    float* values,
    const float* bias,
    unsigned int elements,
    unsigned int width) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        values[index] += bias[index % width];
    }
}

// Scalar-per-row on purpose: it reproduces the reference's two-pass population
// variance before a parallel reduction is admitted by checkpoint tests.
extern "C" __global__ void xrt_image_layer_norm(
    const float* input,
    float* output,
    unsigned int rows,
    unsigned int width,
    float epsilon) {
    const unsigned int row = blockIdx.x;
    if (row >= rows || threadIdx.x != 0) {
        return;
    }
    const unsigned int base = row * width;
    float mean = 0.0f;
    for (unsigned int feature = 0; feature < width; ++feature) {
        mean += input[base + feature];
    }
    mean /= static_cast<float>(width);

    float variance = 0.0f;
    for (unsigned int feature = 0; feature < width; ++feature) {
        const float centered = input[base + feature] - mean;
        variance += centered * centered;
    }
    variance /= static_cast<float>(width);
    const float inverse = rsqrtf(variance + epsilon);
    for (unsigned int feature = 0; feature < width; ++feature) {
        output[base + feature] = (input[base + feature] - mean) * inverse;
    }
}

extern "C" __global__ void xrt_image_affine_rows(
    float* values,
    const float* conditioning,
    unsigned int elements,
    unsigned int sequence,
    unsigned int width,
    unsigned int conditioning_stride,
    unsigned int scale_offset,
    unsigned int shift_offset,
    float scale_bias) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const unsigned int row = index / width;
    const unsigned int feature = index - row * width;
    const unsigned int batch = row / sequence;
    const unsigned int base = batch * conditioning_stride;
    const float scale = conditioning[base + scale_offset + feature];
    const float shift = conditioning[base + shift_offset + feature];
    values[index] = values[index] * (scale_bias + scale) + shift;
}

// Edit conditioning supplies multiple modulation rows per batch (the sampled
// output timestep followed by the zero-timestep source condition). The
// selector is one byte per activation row and chooses the conditioning row
// without copying or round-tripping modulation tensors through host memory.
extern "C" __global__ void xrt_image_affine_rows_indexed(
    float* values,
    const float* conditioning,
    const unsigned char* row_selectors,
    unsigned int elements,
    unsigned int sequence,
    unsigned int width,
    unsigned int conditioning_stride,
    unsigned int conditioning_rows_per_batch,
    unsigned int scale_offset,
    unsigned int shift_offset,
    float scale_bias) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const unsigned int row = index / width;
    const unsigned int feature = index - row * width;
    const unsigned int selected = row_selectors[row];
    if (selected >= conditioning_rows_per_batch) {
        return;
    }
    const unsigned int batch = row / sequence;
    const unsigned int conditioning_row =
        batch * conditioning_rows_per_batch + selected;
    const unsigned int base = conditioning_row * conditioning_stride;
    const float scale = conditioning[base + scale_offset + feature];
    const float shift = conditioning[base + shift_offset + feature];
    values[index] = values[index] * (scale_bias + scale) + shift;
}

extern "C" __global__ void xrt_image_gated_residual(
    float* states,
    const float* update,
    const float* conditioning,
    unsigned int elements,
    unsigned int sequence,
    unsigned int width,
    unsigned int conditioning_stride,
    unsigned int gate_offset) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const unsigned int row = index / width;
    const unsigned int feature = index - row * width;
    const unsigned int batch = row / sequence;
    const float gate =
        conditioning[batch * conditioning_stride + gate_offset + feature];
    states[index] += gate * update[index];
}

extern "C" __global__ void xrt_image_gated_residual_indexed(
    float* states,
    const float* update,
    const float* conditioning,
    const unsigned char* row_selectors,
    unsigned int elements,
    unsigned int sequence,
    unsigned int width,
    unsigned int conditioning_stride,
    unsigned int conditioning_rows_per_batch,
    unsigned int gate_offset) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const unsigned int row = index / width;
    const unsigned int feature = index - row * width;
    const unsigned int selected = row_selectors[row];
    if (selected >= conditioning_rows_per_batch) {
        return;
    }
    const unsigned int batch = row / sequence;
    const unsigned int conditioning_row =
        batch * conditioning_rows_per_batch + selected;
    const float gate = conditioning[
        conditioning_row * conditioning_stride + gate_offset + feature];
    states[index] += gate * update[index];
}

extern "C" __global__ void xrt_image_gelu_tanh(
    float* values,
    unsigned int elements) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const float value = values[index];
    const float cubic = value * value * value;
    const float argument = 0.7978845608028654f * (value + 0.044715f * cubic);
    values[index] = 0.5f * value * (1.0f + tanhf(argument));
}

extern "C" __global__ void xrt_image_complex_rope(
    float* values,
    const float* cosine,
    const float* sine,
    unsigned int total_pairs,
    unsigned int sequence,
    unsigned int heads,
    unsigned int head_dim) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_pairs) {
        return;
    }
    const unsigned int pairs_per_vector = head_dim / 2;
    const unsigned int vector = index / pairs_per_vector;
    const unsigned int pair = index - vector * pairs_per_vector;
    const unsigned int sequence_index = (vector / heads) % sequence;
    const unsigned int frequency = sequence_index * pairs_per_vector + pair;
    const unsigned int value_index = vector * head_dim + pair * 2;
    const float real = values[value_index];
    const float imaginary = values[value_index + 1];
    const float cos_value = cosine[frequency];
    const float sin_value = sine[frequency];
    values[value_index] = real * cos_value - imaginary * sin_value;
    values[value_index + 1] = real * sin_value + imaginary * cos_value;
}

extern "C" __global__ void xrt_image_join_streams(
    const float* text,
    const float* image,
    float* joint,
    unsigned int elements,
    unsigned int text_sequence,
    unsigned int image_sequence,
    unsigned int width) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const unsigned int joint_sequence = text_sequence + image_sequence;
    const unsigned int row = index / width;
    const unsigned int feature = index - row * width;
    const unsigned int batch = row / joint_sequence;
    const unsigned int token = row - batch * joint_sequence;
    if (token < text_sequence) {
        joint[index] = text[(batch * text_sequence + token) * width + feature];
    } else {
        const unsigned int image_token = token - text_sequence;
        joint[index] = image[(batch * image_sequence + image_token) * width + feature];
    }
}

extern "C" __global__ void xrt_image_split_streams(
    const float* joint,
    float* text,
    float* image,
    unsigned int elements,
    unsigned int text_sequence,
    unsigned int image_sequence,
    unsigned int width) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const unsigned int joint_sequence = text_sequence + image_sequence;
    const unsigned int row = index / width;
    const unsigned int feature = index - row * width;
    const unsigned int batch = row / joint_sequence;
    const unsigned int token = row - batch * joint_sequence;
    if (token < text_sequence) {
        text[(batch * text_sequence + token) * width + feature] = joint[index];
    } else {
        const unsigned int image_token = token - text_sequence;
        image[(batch * image_sequence + image_token) * width + feature] = joint[index];
    }
}

// One block owns one (batch, query, head). Threads compute independent key
// scores in parallel while preserving the scalar feature accumulation order
// for every key. Thread zero then performs the original stable softmax scan,
// and the block computes the head output in parallel. This avoids the
// quadratic global score tensor without serializing every query-key dot on one
// thread.
extern "C" __global__ void xrt_image_attention(
    const float* query,
    const float* key,
    const float* value,
    const unsigned char* key_mask,
    float* output,
    unsigned int batch_size,
    unsigned int query_sequence,
    unsigned int key_sequence,
    unsigned int heads,
    unsigned int head_dim,
    float scale) {
    extern __shared__ float scores[];
    // The stable-softmax scan stays on lane zero in key order so its
    // arithmetic is unchanged. Normalizing each independent score can then
    // use the whole block without changing the later value-accumulation order.
    __shared__ float softmax_maximum;
    __shared__ float softmax_denominator;
    const unsigned int head = blockIdx.x;
    const unsigned int query_index = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    const unsigned int lane = threadIdx.x;
    if (batch >= batch_size || query_index >= query_sequence || head >= heads) {
        return;
    }

    const unsigned int query_base =
        ((batch * query_sequence + query_index) * heads + head) * head_dim;
    for (unsigned int key_index = lane;
         key_index < key_sequence;
         key_index += blockDim.x) {
        if (key_mask[batch * key_sequence + key_index] == 0) {
            scores[key_index] = -3.402823466e+38f;
            continue;
        }
        const unsigned int key_base =
            ((batch * key_sequence + key_index) * heads + head) * head_dim;
        float score = 0.0f;
        for (unsigned int feature = 0; feature < head_dim; ++feature) {
            score += query[query_base + feature] * key[key_base + feature];
        }
        scores[key_index] = score * scale;
    }
    __syncthreads();

    if (lane == 0) {
        float maximum = -3.402823466e+38f;
        float denominator = 0.0f;
        for (unsigned int key_index = 0; key_index < key_sequence; ++key_index) {
            const float score = scores[key_index];
            if (key_mask[batch * key_sequence + key_index] == 0) {
                continue;
            }
            if (score > maximum) {
                denominator = denominator * __expf(maximum - score) + 1.0f;
                maximum = score;
            } else {
                denominator += __expf(score - maximum);
            }
        }
        softmax_maximum = maximum;
        softmax_denominator = denominator;
    }
    __syncthreads();

    for (unsigned int key_index = lane;
         key_index < key_sequence;
         key_index += blockDim.x) {
        scores[key_index] =
            __expf(scores[key_index] - softmax_maximum) / softmax_denominator;
    }
    __syncthreads();

    const unsigned int output_base =
        ((batch * query_sequence + query_index) * heads + head) * head_dim;
    for (unsigned int feature = lane; feature < head_dim; feature += blockDim.x) {
        float sum = 0.0f;
        for (unsigned int key_index = 0; key_index < key_sequence; ++key_index) {
            const unsigned int value_index =
                ((batch * key_sequence + key_index) * heads + head) * head_dim + feature;
            sum += scores[key_index] * value[value_index];
        }
        output[output_base + feature] = sum;
    }
}

// Exact-order fallback for joint sequences whose complete score row does not
// fit the portable shared-memory budget. One block still owns one
// (batch, query, head), but scores are recomputed in block-sized tiles. The
// stable-softmax scan and every output-feature value accumulation visit keys in
// the same ascending order as xrt_image_attention.
extern "C" __global__ void xrt_image_attention_tiled(
    const float* query,
    const float* key,
    const float* value,
    const unsigned char* key_mask,
    float* output,
    unsigned int batch_size,
    unsigned int query_sequence,
    unsigned int key_sequence,
    unsigned int heads,
    unsigned int head_dim,
    float scale) {
    extern __shared__ float scores[];
    __shared__ float softmax_maximum;
    __shared__ float softmax_denominator;
    const unsigned int head = blockIdx.x;
    const unsigned int query_index = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    const unsigned int lane = threadIdx.x;
    if (batch >= batch_size || query_index >= query_sequence || head >= heads) {
        return;
    }

    const unsigned int query_base =
        ((batch * query_sequence + query_index) * heads + head) * head_dim;
    float maximum = -3.402823466e+38f;
    float denominator = 0.0f;
    for (unsigned int tile_start = 0;
         tile_start < key_sequence;
         tile_start += blockDim.x) {
        const unsigned int key_index = tile_start + lane;
        if (key_index < key_sequence &&
            key_mask[batch * key_sequence + key_index] != 0) {
            const unsigned int key_base =
                ((batch * key_sequence + key_index) * heads + head) * head_dim;
            float score = 0.0f;
            for (unsigned int feature = 0; feature < head_dim; ++feature) {
                score += query[query_base + feature] * key[key_base + feature];
            }
            scores[lane] = score * scale;
        } else {
            scores[lane] = -3.402823466e+38f;
        }
        __syncthreads();

        if (lane == 0) {
            const unsigned int remaining = key_sequence - tile_start;
            const unsigned int tile_count =
                remaining < blockDim.x ? remaining : blockDim.x;
            for (unsigned int tile_index = 0;
                 tile_index < tile_count;
                 ++tile_index) {
                const unsigned int global_key = tile_start + tile_index;
                if (key_mask[batch * key_sequence + global_key] == 0) {
                    continue;
                }
                const float score = scores[tile_index];
                if (score > maximum) {
                    denominator = denominator * __expf(maximum - score) + 1.0f;
                    maximum = score;
                } else {
                    denominator += __expf(score - maximum);
                }
            }
        }
        __syncthreads();
    }
    if (lane == 0) {
        softmax_maximum = maximum;
        softmax_denominator = denominator;
    }
    __syncthreads();

    const unsigned int output_base =
        ((batch * query_sequence + query_index) * heads + head) * head_dim;
    for (unsigned int feature_base = 0;
         feature_base < head_dim;
         feature_base += blockDim.x) {
        const unsigned int feature = feature_base + lane;
        float sum = 0.0f;
        for (unsigned int tile_start = 0;
             tile_start < key_sequence;
             tile_start += blockDim.x) {
            const unsigned int key_index = tile_start + lane;
            if (key_index < key_sequence &&
                key_mask[batch * key_sequence + key_index] != 0) {
                const unsigned int key_base =
                    ((batch * key_sequence + key_index) * heads + head) * head_dim;
                float score = 0.0f;
                for (unsigned int dot_feature = 0;
                     dot_feature < head_dim;
                     ++dot_feature) {
                    score += query[query_base + dot_feature] *
                             key[key_base + dot_feature];
                }
                scores[lane] =
                    __expf(score * scale - softmax_maximum) /
                    softmax_denominator;
            } else {
                scores[lane] = 0.0f;
            }
            __syncthreads();

            if (feature < head_dim) {
                const unsigned int remaining = key_sequence - tile_start;
                const unsigned int tile_count =
                    remaining < blockDim.x ? remaining : blockDim.x;
                for (unsigned int tile_index = 0;
                     tile_index < tile_count;
                     ++tile_index) {
                    const unsigned int global_key = tile_start + tile_index;
                    const unsigned int value_index =
                        ((batch * key_sequence + global_key) * heads + head) *
                        head_dim + feature;
                    sum += scores[tile_index] * value[value_index];
                }
            }
            __syncthreads();
        }
        if (feature < head_dim) {
            output[output_base + feature] = sum;
        }
    }
}
