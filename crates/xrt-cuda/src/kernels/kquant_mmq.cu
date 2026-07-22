// Native packed K-quant matrix multiplication for image-sized activation
// batches. Activations are quantized to signed Q8 in 32-value blocks, then
// integer dot products use CUDA DP4A. The established F32-activation kernels
// remain the caller's correctness fallback.

constexpr unsigned int XRT_MMQ_WARPS = 8;
constexpr unsigned int XRT_MMQ_M_TILE = 8;

__device__ __forceinline__ int xrt_pack_i8(
    int value0,
    int value1,
    int value2,
    int value3) {
    return (value0 & 0xff) |
        ((value1 & 0xff) << 8) |
        ((value2 & 0xff) << 16) |
        ((value3 & 0xff) << 24);
}

__device__ __forceinline__ void xrt_q4_k_scale_min_mmq(
    const unsigned char* packed,
    unsigned int index,
    unsigned int* scale,
    unsigned int* minimum) {
    if (index < 4) {
        *scale = packed[index] & 0x3f;
        *minimum = packed[index + 4] & 0x3f;
        return;
    }
    *scale =
        ((packed[index + 4] & 0x0f) | ((packed[index - 4] >> 6) << 4)) & 0x3f;
    *minimum =
        ((packed[index + 4] >> 4) | ((packed[index] >> 6) << 4)) & 0x3f;
}

// Workspace layout:
//   [activation_rows * cols signed Q8 bytes]
//   [activation_rows * cols / 32 F32 scales]
//   [activation_rows * cols / 32 F32 source sums]
// The source sum is retained in F32 so the constant/minimum term of Q4_K and
// Q5_K remains exact even though the multiplicative term uses Q8 activations.
extern "C" __global__ void xrt_quantize_q8_mmq(
    const float* input,
    unsigned char* workspace,
    unsigned int quant_count,
    unsigned int quant_block_count) {
    const unsigned int quant_block = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (quant_block >= quant_block_count || lane >= 32) {
        return;
    }

    const unsigned int index = quant_block * 32 + lane;
    const float value = input[index];
    float maximum = fabsf(value);
    float sum = value;
#pragma unroll
    for (unsigned int offset = 16; offset > 0; offset >>= 1) {
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, offset));
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    maximum = __shfl_sync(0xffffffffu, maximum, 0);

    const float scale = maximum > 0.0f ? maximum / 127.0f : 0.0f;
    int quantized = maximum > 0.0f
        ? static_cast<int>(roundf(value * (127.0f / maximum)))
        : 0;
    quantized = quantized < -128 ? -128 : (quantized > 127 ? 127 : quantized);
    reinterpret_cast<signed char*>(workspace)[index] =
        static_cast<signed char>(quantized);

    if (lane == 0) {
        float* metadata = reinterpret_cast<float*>(workspace + quant_count);
        metadata[quant_block] = scale;
        metadata[quant_block_count + quant_block] = sum;
    }
}

extern "C" __global__ void xrt_q4_k_q8_mmq(
    const float* d,
    const float* dmin,
    const unsigned char* scales,
    const unsigned char* quants,
    const unsigned char* workspace,
    float* output,
    unsigned int rows,
    unsigned int cols,
    unsigned int activation_rows,
    unsigned int quant_count,
    unsigned int quant_block_count) {
    const unsigned int lane = threadIdx.x;
    const unsigned int warp = threadIdx.y;
    const unsigned int row = blockIdx.x * XRT_MMQ_WARPS + warp;
    const unsigned int activation_start = blockIdx.y * XRT_MMQ_M_TILE;
    if (lane >= 32 || warp >= XRT_MMQ_WARPS || row >= rows) {
        return;
    }

    const unsigned int subgroup = lane / 8;
    const unsigned int sublane = lane % 8;
    const unsigned int blocks_per_row = cols / 256;
    const unsigned int input_blocks_per_row = cols / 32;
    const signed char* input_quants =
        reinterpret_cast<const signed char*>(workspace);
    const float* metadata = reinterpret_cast<const float*>(workspace + quant_count);
    const float* input_scales = metadata;
    const float* input_sums = metadata + quant_block_count;
    float accumulators[XRT_MMQ_M_TILE] = {};

    for (unsigned int block = 0; block < blocks_per_row; ++block) {
        const unsigned int block_index = row * blocks_per_row + block;
        const float block_d = d[block_index];
        const float block_dmin = dmin[block_index];
        const unsigned char* block_scales = scales + block_index * 12;
        const unsigned char* block_quants = quants + block_index * 128;
        const unsigned int group = subgroup;
        const unsigned int packed_offset = group * 32 + sublane * 4;
        const unsigned int packed = *reinterpret_cast<const unsigned int*>(
            block_quants + packed_offset);

#pragma unroll
        for (unsigned int half = 0; half < 2; ++half) {
            const unsigned int segment = group * 2 + half;
            const int weight_word = half == 0
                ? static_cast<int>(packed & 0x0f0f0f0fu)
                : static_cast<int>((packed >> 4) & 0x0f0f0f0fu);
            unsigned int weight_scale = 0;
            unsigned int weight_minimum = 0;
            if (sublane == 0) {
                xrt_q4_k_scale_min_mmq(
                    block_scales,
                    segment,
                    &weight_scale,
                    &weight_minimum);
            }
            weight_scale = __shfl_sync(
                0xffffffffu,
                weight_scale,
                subgroup * 8,
                32);
            weight_minimum = __shfl_sync(
                0xffffffffu,
                weight_minimum,
                subgroup * 8,
                32);

#pragma unroll
            for (unsigned int activation = 0;
                 activation < XRT_MMQ_M_TILE;
                 ++activation) {
                const unsigned int activation_row = activation_start + activation;
                if (activation_row >= activation_rows) {
                    continue;
                }
                const unsigned int input_offset =
                    activation_row * cols + block * 256 + segment * 32 + sublane * 4;
                const int input_word = *reinterpret_cast<const int*>(
                    input_quants + input_offset);
                int dot = __dp4a(weight_word, input_word, 0);
#pragma unroll
                for (unsigned int offset = 4; offset > 0; offset >>= 1) {
                    dot += __shfl_down_sync(0xffffffffu, dot, offset, 8);
                }
                if (sublane == 0) {
                    const unsigned int input_block =
                        activation_row * input_blocks_per_row + block * 8 + segment;
                    const float multiplicative_scale =
                        block_d * static_cast<float>(weight_scale) *
                        input_scales[input_block];
                    accumulators[activation] = fmaf(
                        multiplicative_scale,
                        static_cast<float>(dot),
                        accumulators[activation]);
                    accumulators[activation] = fmaf(
                        -(block_dmin * static_cast<float>(weight_minimum)),
                        input_sums[input_block],
                        accumulators[activation]);
                }
            }
        }
    }

#pragma unroll
    for (unsigned int activation = 0; activation < XRT_MMQ_M_TILE; ++activation) {
        const float part0 = __shfl_sync(0xffffffffu, accumulators[activation], 0);
        const float part1 = __shfl_sync(0xffffffffu, accumulators[activation], 8);
        const float part2 = __shfl_sync(0xffffffffu, accumulators[activation], 16);
        const float part3 = __shfl_sync(0xffffffffu, accumulators[activation], 24);
        const unsigned int activation_row = activation_start + activation;
        if (lane == 0 && activation_row < activation_rows) {
            output[activation_row * rows + row] =
                ((part0 + part1) + (part2 + part3));
        }
    }
}

extern "C" __global__ void xrt_q5_k_q8_mmq(
    const float* d,
    const float* dmin,
    const unsigned char* scales,
    const unsigned char* high_bits,
    const unsigned char* quants,
    const unsigned char* workspace,
    float* output,
    unsigned int rows,
    unsigned int cols,
    unsigned int activation_rows,
    unsigned int quant_count,
    unsigned int quant_block_count) {
    const unsigned int lane = threadIdx.x;
    const unsigned int warp = threadIdx.y;
    const unsigned int row = blockIdx.x * XRT_MMQ_WARPS + warp;
    const unsigned int activation_start = blockIdx.y * XRT_MMQ_M_TILE;
    if (lane >= 32 || warp >= XRT_MMQ_WARPS || row >= rows) {
        return;
    }

    const unsigned int subgroup = lane / 8;
    const unsigned int sublane = lane % 8;
    const unsigned int blocks_per_row = cols / 256;
    const unsigned int input_blocks_per_row = cols / 32;
    const signed char* input_quants =
        reinterpret_cast<const signed char*>(workspace);
    const float* metadata = reinterpret_cast<const float*>(workspace + quant_count);
    const float* input_scales = metadata;
    const float* input_sums = metadata + quant_block_count;
    float accumulators[XRT_MMQ_M_TILE] = {};

    for (unsigned int block = 0; block < blocks_per_row; ++block) {
        const unsigned int block_index = row * blocks_per_row + block;
        const float block_d = d[block_index];
        const float block_dmin = dmin[block_index];
        const unsigned char* block_scales = scales + block_index * 12;
        const unsigned char* block_high_bits = high_bits + block_index * 32;
        const unsigned char* block_quants = quants + block_index * 128;
        const unsigned int group = subgroup;
        const unsigned int packed_offset = group * 32 + sublane * 4;
        const unsigned int packed = *reinterpret_cast<const unsigned int*>(
            block_quants + packed_offset);
        const unsigned int high = *reinterpret_cast<const unsigned int*>(
            block_high_bits + sublane * 4);

#pragma unroll
        for (unsigned int half = 0; half < 2; ++half) {
            const unsigned int segment = group * 2 + half;
            const unsigned int low = half == 0
                ? packed & 0x0f0f0f0fu
                : (packed >> 4) & 0x0f0f0f0fu;
            const unsigned int extra =
                (((high >> segment) & 0x01010101u) << 4);
            const int weight_word = static_cast<int>(low | extra);
            unsigned int weight_scale = 0;
            unsigned int weight_minimum = 0;
            if (sublane == 0) {
                xrt_q4_k_scale_min_mmq(
                    block_scales,
                    segment,
                    &weight_scale,
                    &weight_minimum);
            }
            weight_scale = __shfl_sync(
                0xffffffffu,
                weight_scale,
                subgroup * 8,
                32);
            weight_minimum = __shfl_sync(
                0xffffffffu,
                weight_minimum,
                subgroup * 8,
                32);

#pragma unroll
            for (unsigned int activation = 0;
                 activation < XRT_MMQ_M_TILE;
                 ++activation) {
                const unsigned int activation_row = activation_start + activation;
                if (activation_row >= activation_rows) {
                    continue;
                }
                const unsigned int input_offset =
                    activation_row * cols + block * 256 + segment * 32 + sublane * 4;
                const int input_word = *reinterpret_cast<const int*>(
                    input_quants + input_offset);
                int dot = __dp4a(weight_word, input_word, 0);
#pragma unroll
                for (unsigned int offset = 4; offset > 0; offset >>= 1) {
                    dot += __shfl_down_sync(0xffffffffu, dot, offset, 8);
                }
                if (sublane == 0) {
                    const unsigned int input_block =
                        activation_row * input_blocks_per_row + block * 8 + segment;
                    const float multiplicative_scale =
                        block_d * static_cast<float>(weight_scale) *
                        input_scales[input_block];
                    accumulators[activation] = fmaf(
                        multiplicative_scale,
                        static_cast<float>(dot),
                        accumulators[activation]);
                    accumulators[activation] = fmaf(
                        -(block_dmin * static_cast<float>(weight_minimum)),
                        input_sums[input_block],
                        accumulators[activation]);
                }
            }
        }
    }

#pragma unroll
    for (unsigned int activation = 0; activation < XRT_MMQ_M_TILE; ++activation) {
        const float part0 = __shfl_sync(0xffffffffu, accumulators[activation], 0);
        const float part1 = __shfl_sync(0xffffffffu, accumulators[activation], 8);
        const float part2 = __shfl_sync(0xffffffffu, accumulators[activation], 16);
        const float part3 = __shfl_sync(0xffffffffu, accumulators[activation], 24);
        const unsigned int activation_row = activation_start + activation;
        if (lane == 0 && activation_row < activation_rows) {
            output[activation_row * rows + row] =
                ((part0 + part1) + (part2 + part3));
        }
    }
}

__device__ __forceinline__ int xrt_q6_k_quant(
    const unsigned char* encoded,
    unsigned int segment,
    unsigned int segment_index) {
    const unsigned int group = segment / 8;
    const unsigned int group_segment = segment % 8;
    const unsigned int kind = group_segment / 2;
    const unsigned int half = group_segment % 2;
    const unsigned int lane = half * 16 + segment_index;
    const unsigned char* ql = encoded;
    const unsigned char* qh = encoded + 128;
    const unsigned char ql_low = ql[group * 64 + lane];
    const unsigned char ql_high = ql[group * 64 + 32 + lane];
    const unsigned char qh_value = qh[group * 32 + lane];
    unsigned int quantized = 0;
    if (kind == 0) {
        quantized = (ql_low & 0x0f) | ((qh_value & 0x03) << 4);
    } else if (kind == 1) {
        quantized = (ql_high & 0x0f) | (((qh_value >> 2) & 0x03) << 4);
    } else if (kind == 2) {
        quantized = (ql_low >> 4) | (((qh_value >> 4) & 0x03) << 4);
    } else {
        quantized = (ql_high >> 4) | (((qh_value >> 6) & 0x03) << 4);
    }
    return static_cast<int>(quantized) - 32;
}

extern "C" __global__ void xrt_q6_k_q8_mmq(
    const float* d,
    const unsigned char* blocks,
    const unsigned char* workspace,
    float* output,
    unsigned int rows,
    unsigned int cols,
    unsigned int activation_rows,
    unsigned int quant_count,
    unsigned int quant_block_count) {
    const unsigned int lane = threadIdx.x;
    const unsigned int warp = threadIdx.y;
    const unsigned int row = blockIdx.x * XRT_MMQ_WARPS + warp;
    const unsigned int activation_start = blockIdx.y * XRT_MMQ_M_TILE;
    if (lane >= 32 || warp >= XRT_MMQ_WARPS || row >= rows) {
        return;
    }

    const unsigned int subgroup = lane / 4;
    const unsigned int sublane = lane % 4;
    const unsigned int blocks_per_row = cols / 256;
    const unsigned int input_blocks_per_row = cols / 32;
    const signed char* input_quants =
        reinterpret_cast<const signed char*>(workspace);
    const float* metadata = reinterpret_cast<const float*>(workspace + quant_count);
    const float* input_scales = metadata;
    float accumulators[XRT_MMQ_M_TILE] = {};

    for (unsigned int block = 0; block < blocks_per_row; ++block) {
        const unsigned int block_index = row * blocks_per_row + block;
        const float block_d = d[block_index];
        const unsigned char* encoded = blocks + block_index * 210;
        const signed char* weight_scales =
            reinterpret_cast<const signed char*>(encoded + 192);

#pragma unroll
        for (unsigned int pass = 0; pass < 2; ++pass) {
            const unsigned int segment = subgroup + pass * 8;
            const unsigned int segment_index = sublane * 4;
            const int weight_word = xrt_pack_i8(
                xrt_q6_k_quant(encoded, segment, segment_index),
                xrt_q6_k_quant(encoded, segment, segment_index + 1),
                xrt_q6_k_quant(encoded, segment, segment_index + 2),
                xrt_q6_k_quant(encoded, segment, segment_index + 3));
            const float weight_scale =
                block_d * static_cast<float>(weight_scales[segment]);

#pragma unroll
            for (unsigned int activation = 0;
                 activation < XRT_MMQ_M_TILE;
                 ++activation) {
                const unsigned int activation_row = activation_start + activation;
                if (activation_row >= activation_rows) {
                    continue;
                }
                const unsigned int input_offset =
                    activation_row * cols + block * 256 + segment * 16 + sublane * 4;
                const int input_word = *reinterpret_cast<const int*>(
                    input_quants + input_offset);
                int dot = __dp4a(weight_word, input_word, 0);
#pragma unroll
                for (unsigned int offset = 2; offset > 0; offset >>= 1) {
                    dot += __shfl_down_sync(0xffffffffu, dot, offset, 4);
                }
                if (sublane == 0) {
                    const unsigned int input_block =
                        activation_row * input_blocks_per_row +
                        block * 8 + segment / 2;
                    accumulators[activation] = fmaf(
                        weight_scale * input_scales[input_block],
                        static_cast<float>(dot),
                        accumulators[activation]);
                }
            }
        }
    }

#pragma unroll
    for (unsigned int activation = 0; activation < XRT_MMQ_M_TILE; ++activation) {
        float value = 0.0f;
#pragma unroll
        for (unsigned int source = 0; source < 32; source += 4) {
            value += __shfl_sync(0xffffffffu, accumulators[activation], source);
        }
        const unsigned int activation_row = activation_start + activation;
        if (lane == 0 && activation_row < activation_rows) {
            output[activation_row * rows + row] = value;
        }
    }
}
