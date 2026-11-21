#include <cuda_fp16.h>

// Returns the index selected by Rust's `Iterator::max_by(f32::total_cmp)`.
// Keeping the reduction on-device avoids copying an entire vocabulary-sized
// logits row to the host for each greedy MTP draft token. The winning index is
// encoded as f32; every supported vocabulary index is exactly representable.
extern "C" __global__ void argmax_total_f32_kernel(
    const float* input,
    float* output,
    unsigned int len) {
    const unsigned int thread = threadIdx.x;
    if (blockIdx.x != 0 || thread >= 256) {
        return;
    }

    int best_key = 0;
    unsigned int best_index = 0;
    unsigned int has_value = 0;
    for (unsigned int index = thread; index < len; index += 256) {
        int key = __float_as_int(input[index]);
        const unsigned int sign_mask =
            static_cast<unsigned int>(key >> 31) >> 1;
        key ^= static_cast<int>(sign_mask);
        if (!has_value || key > best_key ||
            (key == best_key && index > best_index)) {
            best_key = key;
            best_index = index;
            has_value = 1;
        }
    }

    __shared__ int keys[256];
    __shared__ unsigned int indices[256];
    __shared__ unsigned int valid[256];
    keys[thread] = best_key;
    indices[thread] = best_index;
    valid[thread] = has_value;
    __syncthreads();

    for (unsigned int stride = 128; stride != 0; stride >>= 1) {
        if (thread < stride && valid[thread + stride]) {
            const int right_key = keys[thread + stride];
            const unsigned int right_index = indices[thread + stride];
            if (!valid[thread] || right_key > keys[thread] ||
                (right_key == keys[thread] && right_index > indices[thread])) {
                keys[thread] = right_key;
                indices[thread] = right_index;
                valid[thread] = 1;
            }
        }
        __syncthreads();
    }

    if (thread == 0) {
        output[0] = static_cast<float>(indices[0]);
    }
}

__device__ __forceinline__ bool ranked_after(
    int candidate_key,
    unsigned int candidate_index,
    int current_key,
    unsigned int current_index) {
    return candidate_key > current_key ||
        (candidate_key == current_key && candidate_index > current_index);
}

__device__ __forceinline__ void insert_ranked_candidate(
    int candidate_key,
    unsigned int candidate_index,
    unsigned int candidate_valid,
    int& best_key,
    unsigned int& best_index,
    unsigned int& best_valid,
    int& second_key,
    unsigned int& second_index,
    unsigned int& second_valid) {
    if (!candidate_valid) {
        return;
    }
    if (!best_valid || ranked_after(
            candidate_key, candidate_index, best_key, best_index)) {
        if (best_valid) {
            second_key = best_key;
            second_index = best_index;
            second_valid = 1;
        }
        best_key = candidate_key;
        best_index = candidate_index;
        best_valid = 1;
    } else if (candidate_index != best_index &&
               (!second_valid || ranked_after(
                   candidate_key, candidate_index, second_key, second_index))) {
        second_key = candidate_key;
        second_index = candidate_index;
        second_valid = 1;
    }
}

// Returns the total-order argmax index, its softmax probability, and the
// top-one/top-two logit gap. The compact result lets MTP stop an uncertain
// suffix without copying a vocabulary-sized logits row to the host.
extern "C" __global__ void argmax_total_f32_confidence_kernel(
    const float* input,
    float* output,
    unsigned int len) {
    const unsigned int thread = threadIdx.x;
    if (blockIdx.x != 0 || thread >= 256) {
        return;
    }

    int best_key = 0;
    unsigned int best_index = 0;
    unsigned int best_valid = 0;
    int second_key = 0;
    unsigned int second_index = 0;
    unsigned int second_valid = 0;
    for (unsigned int index = thread; index < len; index += 256) {
        int key = __float_as_int(input[index]);
        const unsigned int sign_mask =
            static_cast<unsigned int>(key >> 31) >> 1;
        key ^= static_cast<int>(sign_mask);
        insert_ranked_candidate(
            key,
            index,
            1,
            best_key,
            best_index,
            best_valid,
            second_key,
            second_index,
            second_valid);
    }

    __shared__ int best_keys[256];
    __shared__ unsigned int best_indices[256];
    __shared__ unsigned int best_validity[256];
    __shared__ int second_keys[256];
    __shared__ unsigned int second_indices[256];
    __shared__ unsigned int second_validity[256];
    best_keys[thread] = best_key;
    best_indices[thread] = best_index;
    best_validity[thread] = best_valid;
    second_keys[thread] = second_key;
    second_indices[thread] = second_index;
    second_validity[thread] = second_valid;
    __syncthreads();

    for (unsigned int stride = 128; stride != 0; stride >>= 1) {
        if (thread < stride) {
            insert_ranked_candidate(
                best_keys[thread + stride],
                best_indices[thread + stride],
                best_validity[thread + stride],
                best_keys[thread],
                best_indices[thread],
                best_validity[thread],
                second_keys[thread],
                second_indices[thread],
                second_validity[thread]);
            insert_ranked_candidate(
                second_keys[thread + stride],
                second_indices[thread + stride],
                second_validity[thread + stride],
                best_keys[thread],
                best_indices[thread],
                best_validity[thread],
                second_keys[thread],
                second_indices[thread],
                second_validity[thread]);
        }
        __syncthreads();
    }

    __shared__ float best_value_shared;
    if (thread == 0) {
        best_value_shared = input[best_indices[0]];
        output[0] = static_cast<float>(best_indices[0]);
        output[2] = second_validity[0]
            ? best_value_shared - input[second_indices[0]]
            : __int_as_float(0x7f800000);
    }
    if (thread == 0) {
        float denominator = 0.0f;
        if (isfinite(best_value_shared)) {
            for (unsigned int index = 0; index < len; ++index) {
                const float value = input[index];
                if (isfinite(value)) {
                    denominator += expf(value - best_value_shared);
                }
            }
        }
        output[1] = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    }
}

// Selects the first maximum in every finite logits row using the same
// comparison rule as the runtime's unpenalized greedy sampler. One block owns
// one row, so a complete MTP verify window returns only O(rows) token IDs
// instead of copying O(rows * vocabulary) logits to the host.
extern "C" __global__ void argmax_first_f32_rows_kernel(
    const float* input,
    float* output,
    unsigned int rows,
    unsigned int columns) {
    const unsigned int row = blockIdx.x;
    const unsigned int thread = threadIdx.x;
    if (row >= rows || thread >= 256) {
        return;
    }

    const float* row_input = input + static_cast<unsigned long long>(row) * columns;
    float best_value = -__int_as_float(0x7f800000);
    unsigned int best_index = 0;
    unsigned int has_value = 0;
    for (unsigned int index = thread; index < columns; index += 256) {
        const float value = row_input[index];
        if (isnan(value)) {
            continue;
        }
        if (!has_value || value > best_value ||
            (value == best_value && index < best_index)) {
            best_value = value;
            best_index = index;
            has_value = 1;
        }
    }

    __shared__ float values[256];
    __shared__ unsigned int indices[256];
    __shared__ unsigned int valid[256];
    values[thread] = best_value;
    indices[thread] = best_index;
    valid[thread] = has_value;
    __syncthreads();

    for (unsigned int stride = 128; stride != 0; stride >>= 1) {
        if (thread < stride && valid[thread + stride]) {
            const float right_value = values[thread + stride];
            const unsigned int right_index = indices[thread + stride];
            if (!valid[thread] || right_value > values[thread] ||
                (right_value == values[thread] && right_index < indices[thread])) {
                values[thread] = right_value;
                indices[thread] = right_index;
                valid[thread] = 1;
            }
        }
        __syncthreads();
    }

    if (thread == 0) {
        output[row] = static_cast<float>(valid[0] ? indices[0] : 0);
    }
}

// Selects the first maximum directly from Marlin's F16 epilogue. Marlin's
// retained Q6_K path previously expanded this complete logits matrix to F32
// before applying the same comparison. Converting each candidate in registers
// preserves that decision while avoiding the global F32 write/read round trip.
extern "C" __global__ void argmax_first_f16_rows_kernel(
    const __half* input,
    float* output,
    unsigned int rows,
    unsigned int columns) {
    const unsigned int row = blockIdx.x;
    const unsigned int thread = threadIdx.x;
    if (row >= rows || thread >= 256) {
        return;
    }

    const __half* row_input = input + static_cast<unsigned long long>(row) * columns;
    float best_value = -__int_as_float(0x7f800000);
    unsigned int best_index = 0;
    unsigned int has_value = 0;
    for (unsigned int index = thread; index < columns; index += 256) {
        const float value = __half2float(row_input[index]);
        if (isnan(value)) {
            continue;
        }
        if (!has_value || value > best_value ||
            (value == best_value && index < best_index)) {
            best_value = value;
            best_index = index;
            has_value = 1;
        }
    }

    __shared__ float values[256];
    __shared__ unsigned int indices[256];
    __shared__ unsigned int valid[256];
    values[thread] = best_value;
    indices[thread] = best_index;
    valid[thread] = has_value;
    __syncthreads();

    for (unsigned int stride = 128; stride != 0; stride >>= 1) {
        if (thread < stride && valid[thread + stride]) {
            const float right_value = values[thread + stride];
            const unsigned int right_index = indices[thread + stride];
            if (!valid[thread] || right_value > values[thread] ||
                (right_value == values[thread] && right_index < indices[thread])) {
                values[thread] = right_value;
                indices[thread] = right_index;
                valid[thread] = 1;
            }
        }
        __syncthreads();
    }

    if (thread == 0) {
        output[row] = static_cast<float>(valid[0] ? indices[0] : 0);
    }
}

// Maps one device-resident, exactly encoded f32 index through an f32 lookup
// table. DSpark uses this after reduced-vocabulary argmax so its next Markov
// embedding still receives a target-model token ID without a host round trip.
extern "C" __global__ void lookup_f32_table_device_index_kernel(
    const float* table,
    const float* encoded_index,
    float* output,
    unsigned int table_len) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    const float encoded = encoded_index[0];
    const unsigned int index = static_cast<unsigned int>(encoded);
    output[0] = index < table_len ? table[index] : 0.0f;
}

__device__ __forceinline__ void insert_first_top4(
    float candidate_value,
    unsigned int candidate_index,
    float* values,
    unsigned int* indices) {
    if (isnan(candidate_value)) {
        return;
    }
    unsigned int position = 4;
#pragma unroll
    for (unsigned int slot = 0; slot < 4; ++slot) {
        if (candidate_value > values[slot] ||
            (candidate_value == values[slot] && candidate_index < indices[slot])) {
            position = slot;
            break;
        }
    }
    if (position == 4) {
        return;
    }
#pragma unroll
    for (unsigned int slot = 3; slot > position; --slot) {
        values[slot] = values[slot - 1];
        indices[slot] = indices[slot - 1];
    }
    values[position] = candidate_value;
    indices[position] = candidate_index;
}

// Returns four token IDs and their full-vocabulary log-softmax scores for
// every row.  The fixed, compact result is used to evaluate and construct
// bounded DFlash prefix trees without downloading the logits matrix.
extern "C" __global__ void top4_first_f32_rows_kernel(
    const float* input,
    float* output,
    unsigned int rows,
    unsigned int columns) {
    const unsigned int row = blockIdx.x;
    const unsigned int thread = threadIdx.x;
    if (row >= rows || thread >= 256) {
        return;
    }

    const float* row_input = input + static_cast<unsigned long long>(row) * columns;
    float local_values[4] = {
        -__int_as_float(0x7f800000),
        -__int_as_float(0x7f800000),
        -__int_as_float(0x7f800000),
        -__int_as_float(0x7f800000)};
    unsigned int local_indices[4] = {0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu};
    for (unsigned int index = thread; index < columns; index += 256) {
        insert_first_top4(row_input[index], index, local_values, local_indices);
    }

    __shared__ float values[256 * 4];
    __shared__ unsigned int indices[256 * 4];
#pragma unroll
    for (unsigned int slot = 0; slot < 4; ++slot) {
        values[thread * 4 + slot] = local_values[slot];
        indices[thread * 4 + slot] = local_indices[slot];
    }
    __syncthreads();

    for (unsigned int stride = 128; stride != 0; stride >>= 1) {
        if (thread < stride) {
#pragma unroll
            for (unsigned int slot = 0; slot < 4; ++slot) {
                const unsigned int right = (thread + stride) * 4 + slot;
                insert_first_top4(
                    values[right], indices[right], local_values, local_indices);
            }
#pragma unroll
            for (unsigned int slot = 0; slot < 4; ++slot) {
                values[thread * 4 + slot] = local_values[slot];
                indices[thread * 4 + slot] = local_indices[slot];
            }
        }
        __syncthreads();
    }

    const float maximum = values[0];
    float local_sum = 0.0f;
    if (isfinite(maximum)) {
        for (unsigned int index = thread; index < columns; index += 256) {
            const float value = row_input[index];
            if (isfinite(value)) {
                local_sum += __expf(value - maximum);
            }
        }
    }
    __shared__ float sums[256];
    sums[thread] = local_sum;
    __syncthreads();
    for (unsigned int stride = 128; stride != 0; stride >>= 1) {
        if (thread < stride) {
            sums[thread] += sums[thread + stride];
        }
        __syncthreads();
    }

    if (thread == 0) {
        const float log_denominator = sums[0] > 0.0f
            ? maximum + __logf(sums[0])
            : __int_as_float(0x7f800000);
#pragma unroll
        for (unsigned int slot = 0; slot < 4; ++slot) {
            const unsigned int output_base = row * 8 + slot * 2;
            const unsigned int index = indices[slot] == 0xffffffffu ? 0 : indices[slot];
            output[output_base] = static_cast<float>(index);
            output[output_base + 1] = values[slot] - log_denominator;
        }
    }
}
