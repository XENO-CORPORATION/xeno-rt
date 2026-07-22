__device__ __forceinline__ void xrt_q4_k_scale_min(
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

// Reproduce the four independent eight-lane accumulation chains used by
// xrt-kernels' AVX2 F32-activation Q4_K reference. Qwen3.5 persists the QKV
// projection in recurrent state, so a deterministic reduction order prevents
// small per-token projection differences from accumulating across requests.
extern "C" __global__ void xrt_q4_k_recurrent_matvec(
    const float* d,
    const float* dmin,
    const unsigned char* scales,
    const unsigned char* quants,
    const float* input,
    float* output,
    unsigned int rows,
    unsigned int cols) {
    const unsigned int row = blockIdx.x;
    const unsigned int activation_row = blockIdx.y;
    const unsigned int lane = threadIdx.x;
    if (row >= rows || lane >= 32) {
        return;
    }

    const unsigned int blocks_per_row = cols / 256;
    float accumulator = 0.0f;
    for (unsigned int block = 0; block < blocks_per_row; ++block) {
        const unsigned int block_index = row * blocks_per_row + block;
        const float block_d = d[block_index];
        const float block_dmin = dmin[block_index];
        const unsigned char* block_scales = scales + block_index * 12;
        const unsigned char* block_quants = quants + block_index * 128;
        const unsigned int input_base = block * 256;

        for (unsigned int group = 0; group < 4; ++group) {
            unsigned int scale_low;
            unsigned int minimum_low;
            unsigned int scale_high;
            unsigned int minimum_high;
            xrt_q4_k_scale_min(
                block_scales,
                group * 2,
                &scale_low,
                &minimum_low);
            xrt_q4_k_scale_min(
                block_scales,
                group * 2 + 1,
                &scale_high,
                &minimum_high);

            const unsigned char packed = block_quants[group * 32 + lane];
            const float d_low = __fmul_rn(block_d, static_cast<float>(scale_low));
            const float min_low =
                __fmul_rn(block_dmin, static_cast<float>(minimum_low));
            const float weight_low = __fmaf_rn(
                d_low,
                static_cast<float>(packed & 0x0f),
                -min_low);
            accumulator = __fmaf_rn(
                weight_low,
                input[activation_row * cols + input_base + group * 64 + lane],
                accumulator);

            const float d_high = __fmul_rn(block_d, static_cast<float>(scale_high));
            const float min_high =
                __fmul_rn(block_dmin, static_cast<float>(minimum_high));
            const float weight_high = __fmaf_rn(
                d_high,
                static_cast<float>(packed >> 4),
                -min_high);
            accumulator = __fmaf_rn(
                weight_high,
                input[activation_row * cols + input_base + group * 64 + 32 + lane],
                accumulator);
        }
    }

    __shared__ float chains[32];
    chains[lane] = accumulator;
    __syncthreads();

    if (lane < 8) {
        const float pair_01 = __fadd_rn(chains[lane], chains[lane + 8]);
        const float pair_23 = __fadd_rn(chains[lane + 16], chains[lane + 24]);
        chains[lane] = __fadd_rn(pair_01, pair_23);
    }
    __syncthreads();

    if (lane == 0) {
        const float sum_04 = __fadd_rn(chains[0], chains[4]);
        const float sum_15 = __fadd_rn(chains[1], chains[5]);
        const float sum_26 = __fadd_rn(chains[2], chains[6]);
        const float sum_37 = __fadd_rn(chains[3], chains[7]);
        const float sum_0246 = __fadd_rn(sum_04, sum_26);
        const float sum_1357 = __fadd_rn(sum_15, sum_37);
        output[activation_row * rows + row] = __fadd_rn(sum_0246, sum_1357);
    }
}

// Q5_K uses the same four eight-lane AVX2/FMA chains as Q4_K. Keeping
// Qwen3.5's SSM output projections in this reduction order prevents the
// recurrent hidden state from inheriting cuBLAS reduction drift.
extern "C" __global__ void xrt_q5_k_cpu_order_matvec(
    const float* d,
    const float* dmin,
    const unsigned char* scales,
    const unsigned char* high_bits,
    const unsigned char* quants,
    const float* input,
    float* output,
    unsigned int rows,
    unsigned int cols) {
    const unsigned int row = blockIdx.x;
    const unsigned int activation_row = blockIdx.y;
    const unsigned int lane = threadIdx.x;
    if (row >= rows || lane >= 32) {
        return;
    }

    const unsigned int blocks_per_row = cols / 256;
    float accumulator = 0.0f;
    for (unsigned int block = 0; block < blocks_per_row; ++block) {
        const unsigned int block_index = row * blocks_per_row + block;
        const float block_d = d[block_index];
        const float block_dmin = dmin[block_index];
        const unsigned char* block_scales = scales + block_index * 12;
        const unsigned char* block_high_bits = high_bits + block_index * 32;
        const unsigned char* block_quants = quants + block_index * 128;
        const unsigned int input_base = block * 256;
        const unsigned char high = block_high_bits[lane];

        for (unsigned int group = 0; group < 4; ++group) {
            unsigned int scale_low;
            unsigned int minimum_low;
            unsigned int scale_high;
            unsigned int minimum_high;
            xrt_q4_k_scale_min(
                block_scales,
                group * 2,
                &scale_low,
                &minimum_low);
            xrt_q4_k_scale_min(
                block_scales,
                group * 2 + 1,
                &scale_high,
                &minimum_high);

            const unsigned char packed = block_quants[group * 32 + lane];
            const unsigned int quant_low =
                static_cast<unsigned int>(packed & 0x0f) +
                ((high & (1u << (group * 2))) != 0 ? 16u : 0u);
            const unsigned int quant_high =
                static_cast<unsigned int>(packed >> 4) +
                ((high & (1u << (group * 2 + 1))) != 0 ? 16u : 0u);

            const float d_low = __fmul_rn(block_d, static_cast<float>(scale_low));
            const float min_low =
                __fmul_rn(block_dmin, static_cast<float>(minimum_low));
            const float weight_low = __fmaf_rn(
                d_low,
                static_cast<float>(quant_low),
                -min_low);
            accumulator = __fmaf_rn(
                weight_low,
                input[activation_row * cols + input_base + group * 64 + lane],
                accumulator);

            const float d_high = __fmul_rn(block_d, static_cast<float>(scale_high));
            const float min_high =
                __fmul_rn(block_dmin, static_cast<float>(minimum_high));
            const float weight_high = __fmaf_rn(
                d_high,
                static_cast<float>(quant_high),
                -min_high);
            accumulator = __fmaf_rn(
                weight_high,
                input[activation_row * cols + input_base + group * 64 + 32 + lane],
                accumulator);
        }
    }

    __shared__ float chains[32];
    chains[lane] = accumulator;
    __syncthreads();

    if (lane < 8) {
        const float pair_01 = __fadd_rn(chains[lane], chains[lane + 8]);
        const float pair_23 = __fadd_rn(chains[lane + 16], chains[lane + 24]);
        chains[lane] = __fadd_rn(pair_01, pair_23);
    }
    __syncthreads();

    if (lane == 0) {
        const float sum_04 = __fadd_rn(chains[0], chains[4]);
        const float sum_15 = __fadd_rn(chains[1], chains[5]);
        const float sum_26 = __fadd_rn(chains[2], chains[6]);
        const float sum_37 = __fadd_rn(chains[3], chains[7]);
        const float sum_0246 = __fadd_rn(sum_04, sum_26);
        const float sum_1357 = __fadd_rn(sum_15, sum_37);
        output[activation_row * rows + row] = __fadd_rn(sum_0246, sum_1357);
    }
}

// Image diffusion projects hundreds or thousands of token rows through the
// same packed matrix. A matvec-per-token launch rereads every weight for every
// token, so each warp below retains one decoded weight row while accumulating
// a 16-row activation tile. Eight warps cover eight output features per block.
// The small-batch path continues to use the CPU-order kernels above.
constexpr unsigned int XRT_KQUANT_M_TILE = 16;
constexpr unsigned int XRT_KQUANT_WARPS = 8;
constexpr unsigned int XRT_KQUANT_K_TILE = 256;
constexpr unsigned int XRT_Q4_K_TILED_WARPS = 16;

__device__ __forceinline__ void xrt_warp_write_tile(
    float* accumulators,
    float* output,
    unsigned int output_row,
    unsigned int output_rows,
    unsigned int activation_start,
    unsigned int activation_rows,
    unsigned int lane) {
#pragma unroll
    for (unsigned int activation = 0; activation < XRT_KQUANT_M_TILE; ++activation) {
        float value = accumulators[activation];
#pragma unroll
        for (unsigned int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
        const unsigned int activation_row = activation_start + activation;
        if (lane == 0 && activation_row < activation_rows) {
            output[activation_row * output_rows + output_row] = value;
        }
    }
}

extern "C" __global__ void xrt_q4_k_tiled_matmul(
    const float* d,
    const float* dmin,
    const unsigned char* scales,
    const unsigned char* quants,
    const float* input,
    float* output,
    unsigned int rows,
    unsigned int cols,
    unsigned int activation_rows) {
    const unsigned int lane = threadIdx.x;
    const unsigned int warp = threadIdx.y;
    const unsigned int row = blockIdx.x * XRT_Q4_K_TILED_WARPS + warp;
    const unsigned int activation_start = blockIdx.y * XRT_KQUANT_M_TILE;
    if (lane >= 32 || warp >= XRT_Q4_K_TILED_WARPS) {
        return;
    }

    // Every warp projects the same activation tile through a different packed
    // weight row. Load that tile once per block instead of relying on eight
    // independent global-memory streams to converge in cache. The arithmetic
    // below is unchanged: every lane still visits groups, low/high nibbles,
    // and K blocks in the original order.
    extern __shared__ float activation_tile[];
    const unsigned int thread = warp * 32 + lane;
    const bool active_row = row < rows;
    float accumulators[XRT_KQUANT_M_TILE] = {};
    const unsigned int blocks_per_row = cols / 256;
    for (unsigned int block = 0; block < blocks_per_row; ++block) {
        const unsigned int input_base = block * 256;

#pragma unroll
        for (unsigned int index = thread;
             index < XRT_KQUANT_M_TILE * XRT_KQUANT_K_TILE;
             index += XRT_Q4_K_TILED_WARPS * 32) {
            const unsigned int activation = index / XRT_KQUANT_K_TILE;
            const unsigned int feature = index % XRT_KQUANT_K_TILE;
            const unsigned int activation_row = activation_start + activation;
            activation_tile[index] = activation_row < activation_rows
                ? input[activation_row * cols + input_base + feature]
                : 0.0f;
        }
        __syncthreads();

        if (active_row) {
            const unsigned int block_index = row * blocks_per_row + block;
            const float block_d = d[block_index];
            const float block_dmin = dmin[block_index];
            const unsigned char* block_scales = scales + block_index * 12;
            const unsigned char* block_quants = quants + block_index * 128;

#pragma unroll
            for (unsigned int group = 0; group < 4; ++group) {
                unsigned int scale_low;
                unsigned int minimum_low;
                unsigned int scale_high;
                unsigned int minimum_high;
                xrt_q4_k_scale_min(
                    block_scales,
                    group * 2,
                    &scale_low,
                    &minimum_low);
                xrt_q4_k_scale_min(
                    block_scales,
                    group * 2 + 1,
                    &scale_high,
                    &minimum_high);
                const unsigned char packed = block_quants[group * 32 + lane];
                const float weight_low = fmaf(
                    block_d * static_cast<float>(scale_low),
                    static_cast<float>(packed & 0x0f),
                    -(block_dmin * static_cast<float>(minimum_low)));
                const float weight_high = fmaf(
                    block_d * static_cast<float>(scale_high),
                    static_cast<float>(packed >> 4),
                    -(block_dmin * static_cast<float>(minimum_high)));
                const unsigned int low_feature = group * 64 + lane;
                const unsigned int high_feature = low_feature + 32;
#pragma unroll
                for (unsigned int activation = 0;
                     activation < XRT_KQUANT_M_TILE;
                     ++activation) {
                    const float* input_row =
                        activation_tile + activation * XRT_KQUANT_K_TILE;
                    accumulators[activation] = fmaf(
                        weight_low,
                        input_row[low_feature],
                        accumulators[activation]);
                    accumulators[activation] = fmaf(
                        weight_high,
                        input_row[high_feature],
                        accumulators[activation]);
                }
            }
        }
        __syncthreads();
    }
    if (active_row) {
        xrt_warp_write_tile(
            accumulators,
            output,
            row,
            rows,
            activation_start,
            activation_rows,
            lane);
    }
}

extern "C" __global__ void xrt_q5_k_tiled_matmul(
    const float* d,
    const float* dmin,
    const unsigned char* scales,
    const unsigned char* high_bits,
    const unsigned char* quants,
    const float* input,
    float* output,
    unsigned int rows,
    unsigned int cols,
    unsigned int activation_rows) {
    const unsigned int lane = threadIdx.x;
    const unsigned int warp = threadIdx.y;
    const unsigned int row = blockIdx.x * XRT_KQUANT_WARPS + warp;
    const unsigned int activation_start = blockIdx.y * XRT_KQUANT_M_TILE;
    if (lane >= 32 || warp >= XRT_KQUANT_WARPS || row >= rows) {
        return;
    }

    float accumulators[XRT_KQUANT_M_TILE] = {};
    const unsigned int blocks_per_row = cols / 256;
    for (unsigned int block = 0; block < blocks_per_row; ++block) {
        const unsigned int block_index = row * blocks_per_row + block;
        const float block_d = d[block_index];
        const float block_dmin = dmin[block_index];
        const unsigned char* block_scales = scales + block_index * 12;
        const unsigned char* block_high_bits = high_bits + block_index * 32;
        const unsigned char* block_quants = quants + block_index * 128;
        const unsigned int input_base = block * 256;
        const unsigned char high = block_high_bits[lane];

#pragma unroll
        for (unsigned int group = 0; group < 4; ++group) {
            unsigned int scale_low;
            unsigned int minimum_low;
            unsigned int scale_high;
            unsigned int minimum_high;
            xrt_q4_k_scale_min(block_scales, group * 2, &scale_low, &minimum_low);
            xrt_q4_k_scale_min(
                block_scales,
                group * 2 + 1,
                &scale_high,
                &minimum_high);
            const unsigned char packed = block_quants[group * 32 + lane];
            const unsigned int quant_low =
                static_cast<unsigned int>(packed & 0x0f) +
                ((high & (1u << (group * 2))) != 0 ? 16u : 0u);
            const unsigned int quant_high =
                static_cast<unsigned int>(packed >> 4) +
                ((high & (1u << (group * 2 + 1))) != 0 ? 16u : 0u);
            const float weight_low = fmaf(
                block_d * static_cast<float>(scale_low),
                static_cast<float>(quant_low),
                -(block_dmin * static_cast<float>(minimum_low)));
            const float weight_high = fmaf(
                block_d * static_cast<float>(scale_high),
                static_cast<float>(quant_high),
                -(block_dmin * static_cast<float>(minimum_high)));
            const unsigned int low_feature = input_base + group * 64 + lane;
            const unsigned int high_feature = low_feature + 32;
#pragma unroll
            for (unsigned int activation = 0; activation < XRT_KQUANT_M_TILE; ++activation) {
                const unsigned int activation_row = activation_start + activation;
                if (activation_row < activation_rows) {
                    const float* input_row = input + activation_row * cols;
                    accumulators[activation] = fmaf(
                        weight_low,
                        input_row[low_feature],
                        accumulators[activation]);
                    accumulators[activation] = fmaf(
                        weight_high,
                        input_row[high_feature],
                        accumulators[activation]);
                }
            }
        }
    }
    xrt_warp_write_tile(
        accumulators,
        output,
        row,
        rows,
        activation_start,
        activation_rows,
        lane);
}

extern "C" __global__ void xrt_q6_k_tiled_matmul(
    const float* d,
    const unsigned char* blocks,
    const float* input,
    float* output,
    unsigned int rows,
    unsigned int cols,
    unsigned int activation_rows) {
    const unsigned int lane = threadIdx.x;
    const unsigned int warp = threadIdx.y;
    const unsigned int row = blockIdx.x * XRT_KQUANT_WARPS + warp;
    const unsigned int activation_start = blockIdx.y * XRT_KQUANT_M_TILE;
    if (lane >= 32 || warp >= XRT_KQUANT_WARPS || row >= rows) {
        return;
    }

    float accumulators[XRT_KQUANT_M_TILE] = {};
    const unsigned int blocks_per_row = cols / 256;
    for (unsigned int block = 0; block < blocks_per_row; ++block) {
        const unsigned int block_index = row * blocks_per_row + block;
        const float block_d = d[block_index];
        const unsigned char* encoded = blocks + block_index * 210;
        const unsigned char* ql = encoded;
        const unsigned char* qh = encoded + 128;
        const signed char* scales = reinterpret_cast<const signed char*>(encoded + 192);
        const unsigned int input_base = block * 256;

#pragma unroll
        for (unsigned int group = 0; group < 2; ++group) {
            const unsigned int ql_base = group * 64;
            const unsigned int qh_base = group * 32;
            const unsigned int scale_base = group * 8;
            const unsigned int feature_base = input_base + group * 128;
            const unsigned char qh_value = qh[qh_base + lane];
            const unsigned char ql_low = ql[ql_base + lane];
            const unsigned char ql_high = ql[ql_base + 32 + lane];
            const int quantized[4] = {
                static_cast<int>((ql_low & 0x0f) | ((qh_value & 0x03) << 4)) - 32,
                static_cast<int>((ql_high & 0x0f) | (((qh_value >> 2) & 0x03) << 4)) - 32,
                static_cast<int>((ql_low >> 4) | (((qh_value >> 4) & 0x03) << 4)) - 32,
                static_cast<int>((ql_high >> 4) | (((qh_value >> 6) & 0x03) << 4)) - 32,
            };
#pragma unroll
            for (unsigned int subgroup = 0; subgroup < 4; ++subgroup) {
                const unsigned int scale_index =
                    scale_base + subgroup * 2 + lane / 16;
                const float weight =
                    block_d * static_cast<float>(scales[scale_index]) *
                    static_cast<float>(quantized[subgroup]);
                const unsigned int feature = feature_base + subgroup * 32 + lane;
#pragma unroll
                for (unsigned int activation = 0; activation < XRT_KQUANT_M_TILE; ++activation) {
                    const unsigned int activation_row = activation_start + activation;
                    if (activation_row < activation_rows) {
                        accumulators[activation] = fmaf(
                            weight,
                            input[activation_row * cols + feature],
                            accumulators[activation]);
                    }
                }
            }
        }
    }
    xrt_warp_write_tile(
        accumulators,
        output,
        row,
        rows,
        activation_start,
        activation_rows,
        lane);
}
