#include <cuda_fp16.h>
#include <mma.h>

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

__device__ __forceinline__ float xrt_cpu_order_warp_sum(float accumulator) {
    constexpr unsigned int mask = 0xffffffffu;
    float value = __fadd_rn(accumulator, __shfl_xor_sync(mask, accumulator, 8));
    value = __fadd_rn(value, __shfl_xor_sync(mask, value, 16));
    value = __fadd_rn(value, __shfl_xor_sync(mask, value, 4));
    value = __fadd_rn(value, __shfl_xor_sync(mask, value, 2));
    return __fadd_rn(value, __shfl_xor_sync(mask, value, 1));
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
    const unsigned int warp = threadIdx.y;
    const unsigned int row = blockIdx.x * blockDim.y + warp;
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

        unsigned int lane_scale = 0;
        unsigned int lane_minimum = 0;
        if (lane < 8) {
            xrt_q4_k_scale_min(block_scales, lane, &lane_scale, &lane_minimum);
        }
        const float lane_d = __fmul_rn(block_d, static_cast<float>(lane_scale));
        const float lane_min = __fmul_rn(block_dmin, static_cast<float>(lane_minimum));

        for (unsigned int group = 0; group < 4; ++group) {
            const float d_low = __shfl_sync(0xffffffffu, lane_d, group * 2);
            const float min_low = __shfl_sync(0xffffffffu, lane_min, group * 2);
            const float d_high = __shfl_sync(0xffffffffu, lane_d, group * 2 + 1);
            const float min_high = __shfl_sync(0xffffffffu, lane_min, group * 2 + 1);

            const unsigned char packed = block_quants[group * 32 + lane];
            const float weight_low = __fmaf_rn(
                d_low,
                static_cast<float>(packed & 0x0f),
                -min_low);
            accumulator = __fmaf_rn(
                weight_low,
                input[activation_row * cols + input_base + group * 64 + lane],
                accumulator);

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

    const float sum = xrt_cpu_order_warp_sum(accumulator);
    if (lane == 0) {
        output[activation_row * rows + row] = sum;
    }
}

// same weights. Independent matvec launches reread each packed row once per
// token. This kernel decodes each Q4_K row once and accumulates the activation
// rows in registers while preserving the exact per-row FMA and final reduction
// order of xrt_q4_k_recurrent_matvec.
// Keep register-heavy 16-row verification blocks at 256 threads so Ada can
// schedule more blocks per SM. The activation tile is small relative to the
// packed weight rows, so duplicating that tile across additional blocks is
// cheaper than constraining occupancy with 512-thread blocks.
constexpr unsigned int XRT_Q4_K_VERIFY_WARPS = 8;

template <unsigned int VERIFY_ROWS>
__device__ __forceinline__ void xrt_q4_k_verify_matmul_impl(
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
    const unsigned int row = blockIdx.x * XRT_Q4_K_VERIFY_WARPS + warp;
    if (lane >= 32 || warp >= XRT_Q4_K_VERIFY_WARPS) {
        return;
    }

    extern __shared__ float verify_activation_tile[];
    const unsigned int thread = warp * 32 + lane;
    const bool active_row = row < rows;
    float accumulators[VERIFY_ROWS] = {};
    const unsigned int blocks_per_row = cols / 256;

    for (unsigned int block = 0; block < blocks_per_row; ++block) {
        const unsigned int input_base = block * 256;
        for (unsigned int index = thread;
             index < activation_rows * 256;
             index += XRT_Q4_K_VERIFY_WARPS * 32) {
            const unsigned int activation = index / 256;
            const unsigned int feature = index % 256;
            verify_activation_tile[index] =
                input[activation * cols + input_base + feature];
        }
        __syncthreads();

        if (active_row) {
            const unsigned int block_index = row * blocks_per_row + block;
            const float block_d = d[block_index];
            const float block_dmin = dmin[block_index];
            const unsigned char* block_scales = scales + block_index * 12;
            const unsigned char* block_quants = quants + block_index * 128;

            unsigned int lane_scale = 0;
            unsigned int lane_minimum = 0;
            if (lane < 8) {
                xrt_q4_k_scale_min(
                    block_scales,
                    lane,
                    &lane_scale,
                    &lane_minimum);
            }
            const float lane_d =
                __fmul_rn(block_d, static_cast<float>(lane_scale));
            const float lane_min =
                __fmul_rn(block_dmin, static_cast<float>(lane_minimum));

#pragma unroll
            for (unsigned int group = 0; group < 4; ++group) {
                const float d_low =
                    __shfl_sync(0xffffffffu, lane_d, group * 2);
                const float min_low =
                    __shfl_sync(0xffffffffu, lane_min, group * 2);
                const float d_high =
                    __shfl_sync(0xffffffffu, lane_d, group * 2 + 1);
                const float min_high =
                    __shfl_sync(0xffffffffu, lane_min, group * 2 + 1);
                const unsigned char packed = block_quants[group * 32 + lane];
                const float weight_low = __fmaf_rn(
                    d_low,
                    static_cast<float>(packed & 0x0f),
                    -min_low);
                const float weight_high = __fmaf_rn(
                    d_high,
                    static_cast<float>(packed >> 4),
                    -min_high);
                const unsigned int low_feature = group * 64 + lane;
                const unsigned int high_feature = low_feature + 32;
                for (unsigned int activation = 0;
                     activation < activation_rows;
                     ++activation) {
                    const float* input_row =
                        verify_activation_tile + activation * 256;
                    accumulators[activation] = __fmaf_rn(
                        weight_low,
                        input_row[low_feature],
                        accumulators[activation]);
                    accumulators[activation] = __fmaf_rn(
                        weight_high,
                        input_row[high_feature],
                        accumulators[activation]);
                }
            }
        }
        __syncthreads();
    }

    for (unsigned int activation = 0;
         activation < activation_rows;
         ++activation) {
        const float sum = xrt_cpu_order_warp_sum(accumulators[activation]);
        if (lane == 0 && active_row) {
            output[activation * rows + row] = sum;
        }
    }
}

#define XRT_Q4_K_VERIFY_WRAPPER(ROWS)                                      \
extern "C" __global__ void xrt_q4_k_verify_matmul_##ROWS(                 \
    const float* d,                                                         \
    const float* dmin,                                                      \
    const unsigned char* scales,                                            \
    const unsigned char* quants,                                            \
    const float* input,                                                     \
    float* output,                                                          \
    unsigned int rows,                                                      \
    unsigned int cols,                                                      \
    unsigned int activation_rows) {                                         \
    xrt_q4_k_verify_matmul_impl<ROWS>(                                      \
        d, dmin, scales, quants, input, output, rows, cols, activation_rows); \
}

XRT_Q4_K_VERIFY_WRAPPER(4)
XRT_Q4_K_VERIFY_WRAPPER(8)
XRT_Q4_K_VERIFY_WRAPPER(16)

#undef XRT_Q4_K_VERIFY_WRAPPER

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
    const unsigned int warp = threadIdx.y;
    const unsigned int row = blockIdx.x * blockDim.y + warp;
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

        unsigned int lane_scale = 0;
        unsigned int lane_minimum = 0;
        if (lane < 8) {
            xrt_q4_k_scale_min(block_scales, lane, &lane_scale, &lane_minimum);
        }
        const float lane_d = __fmul_rn(block_d, static_cast<float>(lane_scale));
        const float lane_min = __fmul_rn(block_dmin, static_cast<float>(lane_minimum));

        for (unsigned int group = 0; group < 4; ++group) {
            const float d_low = __shfl_sync(0xffffffffu, lane_d, group * 2);
            const float min_low = __shfl_sync(0xffffffffu, lane_min, group * 2);
            const float d_high = __shfl_sync(0xffffffffu, lane_d, group * 2 + 1);
            const float min_high = __shfl_sync(0xffffffffu, lane_min, group * 2 + 1);

            const unsigned char packed = block_quants[group * 32 + lane];
            const unsigned int quant_low =
                static_cast<unsigned int>(packed & 0x0f) +
                ((high & (1u << (group * 2))) != 0 ? 16u : 0u);
            const unsigned int quant_high =
                static_cast<unsigned int>(packed >> 4) +
                ((high & (1u << (group * 2 + 1))) != 0 ? 16u : 0u);

            const float weight_low = __fmaf_rn(
                d_low,
                static_cast<float>(quant_low),
                -min_low);
            accumulator = __fmaf_rn(
                weight_low,
                input[activation_row * cols + input_base + group * 64 + lane],
                accumulator);

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

    const float sum = xrt_cpu_order_warp_sum(accumulator);
    if (lane == 0) {
        output[activation_row * rows + row] = sum;
    }
}

template <unsigned int VERIFY_ROWS>
__device__ __forceinline__ void xrt_q5_k_verify_matmul_impl(
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
    const unsigned int row = blockIdx.x * XRT_Q4_K_VERIFY_WARPS + warp;
    if (lane >= 32 || warp >= XRT_Q4_K_VERIFY_WARPS) {
        return;
    }

    extern __shared__ float verify_activation_tile[];
    const unsigned int thread = warp * 32 + lane;
    const bool active_row = row < rows;
    float accumulators[VERIFY_ROWS] = {};
    const unsigned int blocks_per_row = cols / 256;

    for (unsigned int block = 0; block < blocks_per_row; ++block) {
        const unsigned int input_base = block * 256;
        for (unsigned int index = thread;
             index < activation_rows * 256;
             index += XRT_Q4_K_VERIFY_WARPS * 32) {
            const unsigned int activation = index / 256;
            const unsigned int feature = index % 256;
            verify_activation_tile[index] =
                input[activation * cols + input_base + feature];
        }
        __syncthreads();

        if (active_row) {
            const unsigned int block_index = row * blocks_per_row + block;
            const float block_d = d[block_index];
            const float block_dmin = dmin[block_index];
            const unsigned char* block_scales = scales + block_index * 12;
            const unsigned char* block_high_bits = high_bits + block_index * 32;
            const unsigned char* block_quants = quants + block_index * 128;
            const unsigned char high = block_high_bits[lane];

            unsigned int lane_scale = 0;
            unsigned int lane_minimum = 0;
            if (lane < 8) {
                xrt_q4_k_scale_min(
                    block_scales,
                    lane,
                    &lane_scale,
                    &lane_minimum);
            }
            const float lane_d =
                __fmul_rn(block_d, static_cast<float>(lane_scale));
            const float lane_min =
                __fmul_rn(block_dmin, static_cast<float>(lane_minimum));

#pragma unroll
            for (unsigned int group = 0; group < 4; ++group) {
                const float d_low =
                    __shfl_sync(0xffffffffu, lane_d, group * 2);
                const float min_low =
                    __shfl_sync(0xffffffffu, lane_min, group * 2);
                const float d_high =
                    __shfl_sync(0xffffffffu, lane_d, group * 2 + 1);
                const float min_high =
                    __shfl_sync(0xffffffffu, lane_min, group * 2 + 1);
                const unsigned char packed = block_quants[group * 32 + lane];
                const unsigned int quant_low =
                    static_cast<unsigned int>(packed & 0x0f) +
                    ((high & (1u << (group * 2))) != 0 ? 16u : 0u);
                const unsigned int quant_high =
                    static_cast<unsigned int>(packed >> 4) +
                    ((high & (1u << (group * 2 + 1))) != 0 ? 16u : 0u);
                const float weight_low = __fmaf_rn(
                    d_low,
                    static_cast<float>(quant_low),
                    -min_low);
                const float weight_high = __fmaf_rn(
                    d_high,
                    static_cast<float>(quant_high),
                    -min_high);
                const unsigned int low_feature = group * 64 + lane;
                const unsigned int high_feature = low_feature + 32;
                for (unsigned int activation = 0;
                     activation < activation_rows;
                     ++activation) {
                    const float* input_row =
                        verify_activation_tile + activation * 256;
                    accumulators[activation] = __fmaf_rn(
                        weight_low,
                        input_row[low_feature],
                        accumulators[activation]);
                    accumulators[activation] = __fmaf_rn(
                        weight_high,
                        input_row[high_feature],
                        accumulators[activation]);
                }
            }
        }
        __syncthreads();
    }

    for (unsigned int activation = 0;
         activation < activation_rows;
         ++activation) {
        const float sum = xrt_cpu_order_warp_sum(accumulators[activation]);
        if (lane == 0 && active_row) {
            output[activation * rows + row] = sum;
        }
    }
}

#define XRT_Q5_K_VERIFY_WRAPPER(ROWS)                                      \
extern "C" __global__ void xrt_q5_k_verify_matmul_##ROWS(                 \
    const float* d,                                                         \
    const float* dmin,                                                      \
    const unsigned char* scales,                                            \
    const unsigned char* high_bits,                                         \
    const unsigned char* quants,                                            \
    const float* input,                                                     \
    float* output,                                                          \
    unsigned int rows,                                                      \
    unsigned int cols,                                                      \
    unsigned int activation_rows) {                                         \
    xrt_q5_k_verify_matmul_impl<ROWS>(                                      \
        d, dmin, scales, high_bits, quants, input, output, rows, cols, activation_rows); \
}

XRT_Q5_K_VERIFY_WRAPPER(4)
XRT_Q5_K_VERIFY_WRAPPER(8)
XRT_Q5_K_VERIFY_WRAPPER(16)

#undef XRT_Q5_K_VERIFY_WRAPPER

// Experimental small-M verifier for Ada-class GPUs. One warp computes a
// 16x16 output tile with FP16 tensor-core inputs and FP32 accumulation. Packed
// Q4_K/Q5_K weights are decoded directly into the shared-memory B tile, so the
// persistent model remains GGUF-quantized and no expanded weight copy is kept.
// This path intentionally has its own runtime opt-in because FP16 conversion
// does not preserve the bit-exact CPU-order verifier contract.
using namespace nvcuda;

extern "C" __global__ void xrt_f32_to_f16_verify(
    const float* input,
    unsigned char* output,
    unsigned int elements) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        reinterpret_cast<__half*>(output)[index] = __float2half_rn(input[index]);
    }
}

extern "C" __global__ void xrt_f16_to_f32_verify(
    const unsigned char* input,
    float* output,
    unsigned int elements) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        output[index] = __half2float(reinterpret_cast<const __half*>(input)[index]);
    }
}

// Marlin writes F16 accumulators.  Keep those accumulators device-local and
// fold the conversion into the residual epilogue instead of materializing an
// intermediate F32 projection and launching a second add kernel.
extern "C" __global__ void xrt_f16_f32_residual_add_verify(
    const unsigned char* projected,
    const float* residual,
    float* output,
    unsigned int elements) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        const float value = __half2float(
            reinterpret_cast<const __half*>(projected)[index]);
        output[index] = __fadd_rn(residual[index], value);
    }
}

__device__ __forceinline__ float xrt_silu_rn(float value) {
    float exponential;
    float denominator;
    float result;
    // Match the existing hand-written SiLU PTX exactly: exp2.approx followed
    // by a round-to-nearest division.  This keeps the fused epilogue on the
    // same numerical contract as F16->F32, SiLU, then multiply.
    asm("mul.rn.f32 %0, %1, 0f3FB8AA3B;" : "=f"(exponential) : "f"(-value));
    asm("ex2.approx.f32 %0, %1;" : "=f"(exponential) : "f"(exponential));
    denominator = __fadd_rn(exponential, 1.0f);
    asm("div.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(value), "f"(denominator));
    return result;
}

// Gate and up are independent Marlin projections.  Their F16 results need no
// standalone F32 conversions: one pass performs both conversions and the
// complete SwiGLU epilogue.
extern "C" __global__ void xrt_f16_swiglu_f32_verify(
    const unsigned char* gate,
    const unsigned char* up,
    float* output,
    unsigned int elements) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        const float gate_value = __half2float(
            reinterpret_cast<const __half*>(gate)[index]);
        const float up_value = __half2float(
            reinterpret_cast<const __half*>(up)[index]);
        output[index] = __fmul_rn(xrt_silu_rn(gate_value), up_value);
    }
}

extern "C" __global__ void xrt_q4_k_tensor_core_verify(
    const float* d,
    const float* dmin,
    const unsigned char* scales,
    const unsigned char* quants,
    const void* input_data,
    float* output,
    unsigned int rows,
    unsigned int cols,
    unsigned int activation_rows,
    unsigned int input_is_f16) {
    const unsigned int lane = threadIdx.x;
    const unsigned int warp = threadIdx.y;
    const unsigned int thread = warp * 32 + lane;
    const unsigned int output_start = blockIdx.x * 16;
    const unsigned int blocks_per_row = cols / 256;
    __shared__ __align__(32) __half activation_tile[16 * 256];
    __shared__ __align__(32) __half weight_tile[16 * 256];
    __shared__ __align__(32) float output_tile[16 * 16];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);
    const float* input_f32 = reinterpret_cast<const float*>(input_data);
    const __half* input_f16 = reinterpret_cast<const __half*>(input_data);

    for (unsigned int block = 0; block < blocks_per_row; ++block) {
        const unsigned int k_start = block * 256;
        for (unsigned int index = thread; index < 16 * 256; index += 16 * 32) {
            const unsigned int activation = index / 256;
            const unsigned int k_offset = index % 256;
            activation_tile[index] = activation < activation_rows
                ? (input_is_f16 != 0
                    ? input_f16[activation * cols + k_start + k_offset]
                    : __float2half_rn(input_f32[activation * cols + k_start + k_offset]))
                : __float2half_rn(0.0f);
        }

        const unsigned int row = output_start + warp;
        if (row < rows) {
            const unsigned int block_index = row * blocks_per_row + block;
            const float block_d = d[block_index];
            const float block_dmin = dmin[block_index];
            const unsigned char* block_scales = scales + block_index * 12;
            const unsigned char* block_quants = quants + block_index * 128;
            unsigned int lane_scale = 0;
            unsigned int lane_minimum = 0;
            if (lane < 8) {
                xrt_q4_k_scale_min(
                    block_scales, lane, &lane_scale, &lane_minimum);
            }
#pragma unroll
            for (unsigned int group = 0; group < 4; ++group) {
                const float d_low = __shfl_sync(
                    0xffffffffu, __fmul_rn(block_d, static_cast<float>(lane_scale)), group * 2);
                const float min_low = __shfl_sync(
                    0xffffffffu, __fmul_rn(block_dmin, static_cast<float>(lane_minimum)), group * 2);
                const float d_high = __shfl_sync(
                    0xffffffffu, __fmul_rn(block_d, static_cast<float>(lane_scale)), group * 2 + 1);
                const float min_high = __shfl_sync(
                    0xffffffffu, __fmul_rn(block_dmin, static_cast<float>(lane_minimum)), group * 2 + 1);
                const unsigned char packed = block_quants[group * 32 + lane];
                weight_tile[warp * 256 + group * 64 + lane] = __float2half_rn(
                    __fmaf_rn(d_low, static_cast<float>(packed & 0x0f), -min_low));
                weight_tile[warp * 256 + group * 64 + 32 + lane] = __float2half_rn(
                    __fmaf_rn(d_high, static_cast<float>(packed >> 4), -min_high));
            }
        }
        __syncthreads();
        if (warp == 0) {
#pragma unroll
            for (unsigned int k_tile = 0; k_tile < 16; ++k_tile) {
                wmma::load_matrix_sync(a, activation_tile + k_tile * 16, 256);
                wmma::load_matrix_sync(b, weight_tile + k_tile * 16, 256);
                wmma::mma_sync(accumulator, a, b, accumulator);
            }
        }
        __syncthreads();
    }

    if (warp == 0) {
        wmma::store_matrix_sync(output_tile, accumulator, 16, wmma::mem_row_major);
        __syncwarp();
        for (unsigned int index = lane; index < 16 * 16; index += 32) {
            const unsigned int activation = index / 16;
            const unsigned int output_column = index % 16;
            const unsigned int row = output_start + output_column;
            if (activation < activation_rows && row < rows) {
                output[activation * rows + row] = output_tile[index];
            }
        }
    }
}

// Four-output-tile verifier for small speculative windows. A CTA decodes 64
// output rows while loading the activation tile only once, then four warps
// execute independent 16x16 WMMA tiles. Compared with four invocations of the
// original 16-row kernel this keeps the same Q4_K arithmetic and weight traffic
// while cutting activation traffic and exposing four times as many compute
// warps per resident CTA.
extern "C" __global__ __launch_bounds__(512, 1)
void xrt_q4_k_tensor_core_verify_n64(
    const float* d,
    const float* dmin,
    const unsigned char* scales,
    const unsigned char* quants,
    const void* input_data,
    float* output,
    unsigned int rows,
    unsigned int cols,
    unsigned int activation_rows,
    unsigned int input_is_f16) {
    constexpr unsigned int OUTPUT_ROWS = 64;
    constexpr unsigned int OUTPUT_TILES = 4;
    const unsigned int lane = threadIdx.x;
    const unsigned int warp = threadIdx.y;
    const unsigned int thread = warp * 32 + lane;
    const unsigned int output_start = blockIdx.x * OUTPUT_ROWS;
    const unsigned int blocks_per_row = cols / 256;
    __shared__ __align__(32) __half activation_tile[16 * 256];
    __shared__ __align__(32) __half weight_tile[OUTPUT_ROWS * 256];
    __shared__ __align__(32) float output_tile[OUTPUT_TILES][16 * 16];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);
    const float* input_f32 = reinterpret_cast<const float*>(input_data);
    const __half* input_f16 = reinterpret_cast<const __half*>(input_data);

    for (unsigned int block = 0; block < blocks_per_row; ++block) {
        const unsigned int k_start = block * 256;
        for (unsigned int index = thread; index < 16 * 256;
             index += blockDim.y * 32) {
            const unsigned int activation = index / 256;
            const unsigned int k_offset = index % 256;
            activation_tile[index] = activation < activation_rows
                ? (input_is_f16 != 0
                    ? input_f16[activation * cols + k_start + k_offset]
                    : __float2half_rn(
                        input_f32[activation * cols + k_start + k_offset]))
                : __float2half_rn(0.0f);
        }

#pragma unroll
        for (unsigned int row_group = 0; row_group < OUTPUT_TILES; ++row_group) {
            const unsigned int output_offset = warp + row_group * 16;
            const unsigned int row = output_start + output_offset;
            if (row < rows) {
                const unsigned int block_index = row * blocks_per_row + block;
                const float block_d = d[block_index];
                const float block_dmin = dmin[block_index];
                const unsigned char* block_scales = scales + block_index * 12;
                const unsigned char* block_quants = quants + block_index * 128;
                unsigned int lane_scale = 0;
                unsigned int lane_minimum = 0;
                if (lane < 8) {
                    xrt_q4_k_scale_min(
                        block_scales, lane, &lane_scale, &lane_minimum);
                }
#pragma unroll
                for (unsigned int group = 0; group < 4; ++group) {
                    const float d_low = __shfl_sync(
                        0xffffffffu,
                        __fmul_rn(block_d, static_cast<float>(lane_scale)),
                        group * 2);
                    const float min_low = __shfl_sync(
                        0xffffffffu,
                        __fmul_rn(block_dmin, static_cast<float>(lane_minimum)),
                        group * 2);
                    const float d_high = __shfl_sync(
                        0xffffffffu,
                        __fmul_rn(block_d, static_cast<float>(lane_scale)),
                        group * 2 + 1);
                    const float min_high = __shfl_sync(
                        0xffffffffu,
                        __fmul_rn(block_dmin, static_cast<float>(lane_minimum)),
                        group * 2 + 1);
                    const unsigned char packed = block_quants[group * 32 + lane];
                    const unsigned int tile_base = output_offset * 256;
                    weight_tile[tile_base + group * 64 + lane] =
                        __float2half_rn(__fmaf_rn(
                            d_low, static_cast<float>(packed & 0x0f), -min_low));
                    weight_tile[tile_base + group * 64 + 32 + lane] =
                        __float2half_rn(__fmaf_rn(
                            d_high, static_cast<float>(packed >> 4), -min_high));
                }
            }
        }
        __syncthreads();
        if (warp < OUTPUT_TILES) {
            const __half* warp_weights = weight_tile + warp * 16 * 256;
#pragma unroll
            for (unsigned int k_tile = 0; k_tile < 16; ++k_tile) {
                wmma::load_matrix_sync(a, activation_tile + k_tile * 16, 256);
                wmma::load_matrix_sync(b, warp_weights + k_tile * 16, 256);
                wmma::mma_sync(accumulator, a, b, accumulator);
            }
        }
        __syncthreads();
    }

    if (warp < OUTPUT_TILES) {
        wmma::store_matrix_sync(
            output_tile[warp], accumulator, 16, wmma::mem_row_major);
        __syncwarp();
        for (unsigned int index = lane; index < 16 * 16; index += 32) {
            const unsigned int activation = index / 16;
            const unsigned int output_column = index % 16;
            const unsigned int row = output_start + warp * 16 + output_column;
            if (activation < activation_rows && row < rows) {
                output[activation * rows + row] = output_tile[warp][index];
            }
        }
    }
}

// Two-way K-parallel verifier. A CTA stages two independent 256-column
// blocks and assigns one tensor-core warp to each block. Partial FP32 tiles are
// reduced after all K groups, replacing four decode/compute barrier rounds
// with one while keeping packed Q4_K weights resident.
extern "C" __global__ __launch_bounds__(512, 1)
void xrt_q4_k_tensor_core_verify_k2(
    const float* d,
    const float* dmin,
    const unsigned char* scales,
    const unsigned char* quants,
    const void* input_data,
    float* output,
    unsigned int rows,
    unsigned int cols,
    unsigned int activation_rows,
    unsigned int input_is_f16) {
    constexpr unsigned int K_SLOTS = 2;
    const unsigned int lane = threadIdx.x;
    const unsigned int warp = threadIdx.y;
    const unsigned int thread = warp * 32 + lane;
    const unsigned int output_start = blockIdx.x * 16;
    const unsigned int blocks_per_row = cols / 256;
    const float* input_f32 = reinterpret_cast<const float*>(input_data);
    const __half* input_f16 = reinterpret_cast<const __half*>(input_data);
    __shared__ __align__(32) __half activation_tiles[K_SLOTS][16 * 256];
    __shared__ __align__(32) __half weight_tiles[K_SLOTS][16 * 256];
    __shared__ __align__(32) float partial_tiles[K_SLOTS][16 * 16];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);

    for (unsigned int block_base = 0; block_base < blocks_per_row;
         block_base += K_SLOTS) {
#pragma unroll
        for (unsigned int slot = 0; slot < K_SLOTS; ++slot) {
            const unsigned int block = block_base + slot;
            const unsigned int k_start = block * 256;
            for (unsigned int index = thread; index < 16 * 256;
                 index += 16 * 32) {
                const unsigned int activation = index / 256;
                const unsigned int k_offset = index % 256;
                activation_tiles[slot][index] =
                    block < blocks_per_row && activation < activation_rows
                    ? (input_is_f16 != 0
                        ? input_f16[activation * cols + k_start + k_offset]
                        : __float2half_rn(
                            input_f32[activation * cols + k_start + k_offset]))
                    : __float2half_rn(0.0f);
            }

            const unsigned int row = output_start + warp;
            if (block < blocks_per_row && row < rows) {
                const unsigned int block_index = row * blocks_per_row + block;
                const float block_d = d[block_index];
                const float block_dmin = dmin[block_index];
                const unsigned char* block_scales = scales + block_index * 12;
                const unsigned char* block_quants = quants + block_index * 128;
                unsigned int lane_scale = 0;
                unsigned int lane_minimum = 0;
                if (lane < 8) {
                    xrt_q4_k_scale_min(
                        block_scales, lane, &lane_scale, &lane_minimum);
                }
#pragma unroll
                for (unsigned int group = 0; group < 4; ++group) {
                    const float d_low = __shfl_sync(
                        0xffffffffu,
                        __fmul_rn(block_d, static_cast<float>(lane_scale)),
                        group * 2);
                    const float min_low = __shfl_sync(
                        0xffffffffu,
                        __fmul_rn(block_dmin, static_cast<float>(lane_minimum)),
                        group * 2);
                    const float d_high = __shfl_sync(
                        0xffffffffu,
                        __fmul_rn(block_d, static_cast<float>(lane_scale)),
                        group * 2 + 1);
                    const float min_high = __shfl_sync(
                        0xffffffffu,
                        __fmul_rn(block_dmin, static_cast<float>(lane_minimum)),
                        group * 2 + 1);
                    const unsigned char packed = block_quants[group * 32 + lane];
                    weight_tiles[slot][warp * 256 + group * 64 + lane] =
                        __float2half_rn(__fmaf_rn(
                            d_low, static_cast<float>(packed & 0x0f), -min_low));
                    weight_tiles[slot][warp * 256 + group * 64 + 32 + lane] =
                        __float2half_rn(__fmaf_rn(
                            d_high, static_cast<float>(packed >> 4), -min_high));
                }
            } else {
                for (unsigned int index = lane; index < 256; index += 32) {
                    weight_tiles[slot][warp * 256 + index] = __float2half_rn(0.0f);
                }
            }
        }
        __syncthreads();
        if (warp < K_SLOTS && block_base + warp < blocks_per_row) {
#pragma unroll
            for (unsigned int k_tile = 0; k_tile < 16; ++k_tile) {
                wmma::load_matrix_sync(
                    a, activation_tiles[warp] + k_tile * 16, 256);
                wmma::load_matrix_sync(
                    b, weight_tiles[warp] + k_tile * 16, 256);
                wmma::mma_sync(accumulator, a, b, accumulator);
            }
        }
        __syncthreads();
    }

    if (warp < K_SLOTS) {
        wmma::store_matrix_sync(
            partial_tiles[warp], accumulator, 16, wmma::mem_row_major);
    }
    __syncthreads();
    if (thread < 16 * 16) {
        const unsigned int activation = thread / 16;
        const unsigned int output_offset = thread % 16;
        const unsigned int row = output_start + output_offset;
        if (activation < activation_rows && row < rows) {
            output[activation * rows + row] =
                partial_tiles[0][thread] + partial_tiles[1][thread];
        }
    }
}

// Q8-activation / raw-Q4 tensor-core verifier for Ampere-class and newer
// devices. Each CTA expands one 16-row weight tile to signed INT8 once per
// 256-column K block. The compute warp performs segment-local integer MMA so
// every Q4_K scale/min pair can be applied in F32 without expanding weights to
// FP16 or changing their persistent GGUF representation.
extern "C" __global__ void xrt_q4_k_int8_tensor_core_verify(
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
    const unsigned int thread = warp * 32 + lane;
    const unsigned int output_start = blockIdx.x * 16;
    const unsigned int blocks_per_row = cols / 256;
    const unsigned int input_blocks_per_row = cols / 32;
    const signed char* input_quants =
        reinterpret_cast<const signed char*>(workspace);
    const float* metadata = reinterpret_cast<const float*>(workspace + quant_count);
    const float* input_scales = metadata;
    const float* input_sums = metadata + quant_block_count;
    __shared__ __align__(32) signed char activation_tile[16 * 256];
    __shared__ __align__(32) signed char weight_tile[16 * 256];
    __shared__ float weight_scales[16 * 8];
    __shared__ float weight_minimums[16 * 8];
    __shared__ int dot_tiles[8][16 * 16];
    __shared__ float output_accumulators[16 * 16];

    if (thread < 16 * 16) {
        output_accumulators[thread] = 0.0f;
    }
    __syncthreads();
    for (unsigned int block = 0; block < blocks_per_row; ++block) {
        const unsigned int k_start = block * 256;
        if (thread < 16 * 8) {
            const unsigned int output_offset = thread / 8;
            const unsigned int segment = thread % 8;
            const unsigned int row = output_start + output_offset;
            if (row < rows) {
                const unsigned int block_index = row * blocks_per_row + block;
                unsigned int scale = 0;
                unsigned int minimum = 0;
                xrt_q4_k_scale_min(
                    scales + block_index * 12, segment, &scale, &minimum);
                weight_scales[thread] =
                    d[block_index] * static_cast<float>(scale);
                weight_minimums[thread] =
                    dmin[block_index] * static_cast<float>(minimum);
            } else {
                weight_scales[thread] = 0.0f;
                weight_minimums[thread] = 0.0f;
            }
        }
        for (unsigned int index = thread; index < 16 * 256;
             index += blockDim.y * 32) {
            const unsigned int activation = index / 256;
            const unsigned int k_offset = index % 256;
            activation_tile[index] = activation < activation_rows
                ? input_quants[activation * cols + k_start + k_offset]
                : static_cast<signed char>(0);

            const unsigned int output_offset = index / 256;
            const unsigned int row = output_start + output_offset;
            if (row < rows) {
                const unsigned int block_index = row * blocks_per_row + block;
                const unsigned int segment = k_offset / 32;
                const unsigned int group = segment / 2;
                const unsigned int within_segment = k_offset % 32;
                const unsigned char packed =
                    quants[block_index * 128 + group * 32 + within_segment];
                weight_tile[index] = static_cast<signed char>(
                    (segment & 1u) == 0 ? (packed & 0x0f) : (packed >> 4));
            } else {
                weight_tile[index] = static_cast<signed char>(0);
            }
        }
        __syncthreads();

        if (warp == 0) {
#pragma unroll
            for (unsigned int segment = 0; segment < 8; ++segment) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16,
                    signed char, wmma::row_major> activation_fragment;
                wmma::fragment<wmma::matrix_b, 16, 16, 16,
                    signed char, wmma::col_major> weight_fragment;
                wmma::fragment<wmma::accumulator, 16, 16, 16, int> dot_fragment;
                wmma::fill_fragment(dot_fragment, 0);
                const unsigned int segment_start = segment * 32;
                wmma::load_matrix_sync(
                    activation_fragment,
                    activation_tile + segment_start,
                    256);
                wmma::load_matrix_sync(
                    weight_fragment,
                    weight_tile + segment_start,
                    256);
                wmma::mma_sync(
                    dot_fragment,
                    activation_fragment,
                    weight_fragment,
                    dot_fragment);
                wmma::load_matrix_sync(
                    activation_fragment,
                    activation_tile + segment_start + 16,
                    256);
                wmma::load_matrix_sync(
                    weight_fragment,
                    weight_tile + segment_start + 16,
                    256);
                wmma::mma_sync(
                    dot_fragment,
                    activation_fragment,
                    weight_fragment,
                    dot_fragment);
                wmma::store_matrix_sync(
                    dot_tiles[segment], dot_fragment, 16, wmma::mem_row_major);
            }
        }
        __syncthreads();
        if (thread < 16 * 16) {
            const unsigned int activation = thread / 16;
            const unsigned int output_offset = thread % 16;
            const unsigned int row = output_start + output_offset;
            if (activation < activation_rows && row < rows) {
                float value = output_accumulators[thread];
#pragma unroll
                for (unsigned int segment = 0; segment < 8; ++segment) {
                    const unsigned int input_block =
                        activation * input_blocks_per_row + block * 8 + segment;
                    value = __fmaf_rn(
                        weight_scales[output_offset * 8 + segment] *
                            input_scales[input_block],
                        static_cast<float>(dot_tiles[segment][thread]),
                        value);
                    value = __fmaf_rn(
                        -weight_minimums[output_offset * 8 + segment],
                        input_sums[input_block],
                        value);
                }
                output_accumulators[thread] = value;
            }
        }
        __syncthreads();
    }

    if (thread < 16 * 16) {
        const unsigned int activation = thread / 16;
        const unsigned int output_offset = thread % 16;
        const unsigned int row = output_start + output_offset;
        if (activation < activation_rows && row < rows) {
            output[activation * rows + row] = output_accumulators[thread];
        }
    }
}

// Double-buffered variant of the small-M Q4_K verifier. Sixteen warps decode
// the next packed weight/activation tile while a dedicated warp executes WMMA
// on the current tile. This removes one CTA-wide barrier per K block and hides
// tensor-core issue latency behind the following block's unpack work.
extern "C" __global__ __launch_bounds__(544, 2)
void xrt_q4_k_tensor_core_verify_pipelined(
    const float* d,
    const float* dmin,
    const unsigned char* scales,
    const unsigned char* quants,
    const float* input,
    float* output,
    unsigned int rows,
    unsigned int cols,
    unsigned int activation_rows,
    unsigned int) {
    constexpr unsigned int DECODER_WARPS = 16;
    constexpr unsigned int COMPUTE_WARP = 16;
    const unsigned int lane = threadIdx.x;
    const unsigned int warp = threadIdx.y;
    const unsigned int thread = warp * 32 + lane;
    const unsigned int output_start = blockIdx.x * 16;
    const unsigned int blocks_per_row = cols / 256;
    __shared__ __align__(32) __half activation_tile[2][16 * 256];
    __shared__ __align__(32) __half weight_tile[2][16 * 256];
    __shared__ __align__(32) float output_tile[16 * 16];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);

    if (blocks_per_row == 0) {
        return;
    }

    if (warp < DECODER_WARPS) {
        for (unsigned int index = thread; index < 16 * 256;
             index += DECODER_WARPS * 32) {
            const unsigned int activation = index / 256;
            const unsigned int k_offset = index % 256;
            activation_tile[0][index] = activation < activation_rows
                ? __float2half_rn(input[activation * cols + k_offset])
                : __float2half_rn(0.0f);
        }
        const unsigned int row = output_start + warp;
        if (row < rows) {
            const unsigned int block_index = row * blocks_per_row;
            const float block_d = d[block_index];
            const float block_dmin = dmin[block_index];
            const unsigned char* block_scales = scales + block_index * 12;
            const unsigned char* block_quants = quants + block_index * 128;
            unsigned int lane_scale = 0;
            unsigned int lane_minimum = 0;
            if (lane < 8) {
                xrt_q4_k_scale_min(block_scales, lane, &lane_scale, &lane_minimum);
            }
#pragma unroll
            for (unsigned int group = 0; group < 4; ++group) {
                const float d_low = __shfl_sync(
                    0xffffffffu, __fmul_rn(block_d, static_cast<float>(lane_scale)), group * 2);
                const float min_low = __shfl_sync(
                    0xffffffffu, __fmul_rn(block_dmin, static_cast<float>(lane_minimum)), group * 2);
                const float d_high = __shfl_sync(
                    0xffffffffu, __fmul_rn(block_d, static_cast<float>(lane_scale)), group * 2 + 1);
                const float min_high = __shfl_sync(
                    0xffffffffu, __fmul_rn(block_dmin, static_cast<float>(lane_minimum)), group * 2 + 1);
                const unsigned char packed = block_quants[group * 32 + lane];
                weight_tile[0][warp * 256 + group * 64 + lane] = __float2half_rn(
                    __fmaf_rn(d_low, static_cast<float>(packed & 0x0f), -min_low));
                weight_tile[0][warp * 256 + group * 64 + 32 + lane] = __float2half_rn(
                    __fmaf_rn(d_high, static_cast<float>(packed >> 4), -min_high));
            }
        }
    }
    __syncthreads();

    for (unsigned int block = 0; block < blocks_per_row; ++block) {
        const unsigned int current = block & 1u;
        const unsigned int next = current ^ 1u;
        if (warp == COMPUTE_WARP) {
#pragma unroll
            for (unsigned int k_tile = 0; k_tile < 16; ++k_tile) {
                wmma::load_matrix_sync(a, activation_tile[current] + k_tile * 16, 256);
                wmma::load_matrix_sync(b, weight_tile[current] + k_tile * 16, 256);
                wmma::mma_sync(accumulator, a, b, accumulator);
            }
        } else if (warp < DECODER_WARPS && block + 1 < blocks_per_row) {
            const unsigned int next_block = block + 1;
            const unsigned int k_start = next_block * 256;
            for (unsigned int index = thread; index < 16 * 256;
                 index += DECODER_WARPS * 32) {
                const unsigned int activation = index / 256;
                const unsigned int k_offset = index % 256;
                activation_tile[next][index] = activation < activation_rows
                    ? __float2half_rn(input[activation * cols + k_start + k_offset])
                    : __float2half_rn(0.0f);
            }
            const unsigned int row = output_start + warp;
            if (row < rows) {
                const unsigned int block_index = row * blocks_per_row + next_block;
                const float block_d = d[block_index];
                const float block_dmin = dmin[block_index];
                const unsigned char* block_scales = scales + block_index * 12;
                const unsigned char* block_quants = quants + block_index * 128;
                unsigned int lane_scale = 0;
                unsigned int lane_minimum = 0;
                if (lane < 8) {
                    xrt_q4_k_scale_min(block_scales, lane, &lane_scale, &lane_minimum);
                }
#pragma unroll
                for (unsigned int group = 0; group < 4; ++group) {
                    const float d_low = __shfl_sync(
                        0xffffffffu, __fmul_rn(block_d, static_cast<float>(lane_scale)), group * 2);
                    const float min_low = __shfl_sync(
                        0xffffffffu, __fmul_rn(block_dmin, static_cast<float>(lane_minimum)), group * 2);
                    const float d_high = __shfl_sync(
                        0xffffffffu, __fmul_rn(block_d, static_cast<float>(lane_scale)), group * 2 + 1);
                    const float min_high = __shfl_sync(
                        0xffffffffu, __fmul_rn(block_dmin, static_cast<float>(lane_minimum)), group * 2 + 1);
                    const unsigned char packed = block_quants[group * 32 + lane];
                    weight_tile[next][warp * 256 + group * 64 + lane] = __float2half_rn(
                        __fmaf_rn(d_low, static_cast<float>(packed & 0x0f), -min_low));
                    weight_tile[next][warp * 256 + group * 64 + 32 + lane] = __float2half_rn(
                        __fmaf_rn(d_high, static_cast<float>(packed >> 4), -min_high));
                }
            }
        }
        __syncthreads();
    }

    if (warp == COMPUTE_WARP) {
        wmma::store_matrix_sync(output_tile, accumulator, 16, wmma::mem_row_major);
        __syncwarp();
        for (unsigned int index = lane; index < 16 * 16; index += 32) {
            const unsigned int activation = index / 16;
            const unsigned int output_column = index % 16;
            const unsigned int row = output_start + output_column;
            if (activation < activation_rows && row < rows) {
                output[activation * rows + row] = output_tile[index];
            }
        }
    }
}

// Qwen-style SwiGLU verifier for paired Q4_K gate/up matrices. Both
// projections consume the same activation tile, so keeping their tensor-core
// accumulators in one block avoids loading that tile twice and lets the gate
// activation and multiplication happen before either intermediate is written
// to global memory.
extern "C" __global__ void xrt_q4_k_tensor_core_swiglu_verify(
    const float* gate_d,
    const float* gate_dmin,
    const unsigned char* gate_scales,
    const unsigned char* gate_quants,
    const float* up_d,
    const float* up_dmin,
    const unsigned char* up_scales,
    const unsigned char* up_quants,
    const float* input,
    float* output,
    unsigned int rows,
    unsigned int cols,
    unsigned int activation_rows) {
    const unsigned int lane = threadIdx.x;
    const unsigned int warp = threadIdx.y;
    const unsigned int thread = warp * 32 + lane;
    const unsigned int output_start = blockIdx.x * 16;
    const unsigned int blocks_per_row = cols / 256;
    __shared__ __align__(32) __half activation_tile[16 * 256];
    __shared__ __align__(32) __half gate_weight_tile[16 * 256];
    __shared__ __align__(32) __half up_weight_tile[16 * 256];
    __shared__ __align__(32) float gate_output_tile[16 * 16];
    __shared__ __align__(32) float up_output_tile[16 * 16];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> gate_accumulator;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> up_accumulator;
    wmma::fill_fragment(gate_accumulator, 0.0f);
    wmma::fill_fragment(up_accumulator, 0.0f);

    for (unsigned int block = 0; block < blocks_per_row; ++block) {
        const unsigned int k_start = block * 256;
        for (unsigned int index = thread; index < 16 * 256; index += blockDim.y * 32) {
            const unsigned int activation = index / 256;
            const unsigned int k_offset = index % 256;
            activation_tile[index] = activation < activation_rows
                ? __float2half_rn(input[activation * cols + k_start + k_offset])
                : __float2half_rn(0.0f);
        }

        const unsigned int row = output_start + warp;
        if (row < rows) {
            const unsigned int block_index = row * blocks_per_row + block;
            const unsigned char* gate_block_scales = gate_scales + block_index * 12;
            const unsigned char* gate_block_quants = gate_quants + block_index * 128;
            const unsigned char* up_block_scales = up_scales + block_index * 12;
            const unsigned char* up_block_quants = up_quants + block_index * 128;
            unsigned int gate_lane_scale = 0;
            unsigned int gate_lane_minimum = 0;
            unsigned int up_lane_scale = 0;
            unsigned int up_lane_minimum = 0;
            if (lane < 8) {
                xrt_q4_k_scale_min(gate_block_scales, lane, &gate_lane_scale, &gate_lane_minimum);
                xrt_q4_k_scale_min(up_block_scales, lane, &up_lane_scale, &up_lane_minimum);
            }
#pragma unroll
            for (unsigned int group = 0; group < 4; ++group) {
                const unsigned int low_index = group * 2;
                const unsigned int high_index = low_index + 1;
                const float gate_d_low = __shfl_sync(
                    0xffffffffu,
                    __fmul_rn(gate_d[block_index], static_cast<float>(gate_lane_scale)),
                    low_index);
                const float gate_min_low = __shfl_sync(
                    0xffffffffu,
                    __fmul_rn(gate_dmin[block_index], static_cast<float>(gate_lane_minimum)),
                    low_index);
                const float gate_d_high = __shfl_sync(
                    0xffffffffu,
                    __fmul_rn(gate_d[block_index], static_cast<float>(gate_lane_scale)),
                    high_index);
                const float gate_min_high = __shfl_sync(
                    0xffffffffu,
                    __fmul_rn(gate_dmin[block_index], static_cast<float>(gate_lane_minimum)),
                    high_index);
                const float up_d_low = __shfl_sync(
                    0xffffffffu,
                    __fmul_rn(up_d[block_index], static_cast<float>(up_lane_scale)),
                    low_index);
                const float up_min_low = __shfl_sync(
                    0xffffffffu,
                    __fmul_rn(up_dmin[block_index], static_cast<float>(up_lane_minimum)),
                    low_index);
                const float up_d_high = __shfl_sync(
                    0xffffffffu,
                    __fmul_rn(up_d[block_index], static_cast<float>(up_lane_scale)),
                    high_index);
                const float up_min_high = __shfl_sync(
                    0xffffffffu,
                    __fmul_rn(up_dmin[block_index], static_cast<float>(up_lane_minimum)),
                    high_index);
                const unsigned char gate_packed = gate_block_quants[group * 32 + lane];
                const unsigned char up_packed = up_block_quants[group * 32 + lane];
                const unsigned int tile_index = warp * 256 + group * 64 + lane;
                gate_weight_tile[tile_index] = __float2half_rn(__fmaf_rn(
                    gate_d_low, static_cast<float>(gate_packed & 0x0f), -gate_min_low));
                gate_weight_tile[tile_index + 32] = __float2half_rn(__fmaf_rn(
                    gate_d_high, static_cast<float>(gate_packed >> 4), -gate_min_high));
                up_weight_tile[tile_index] = __float2half_rn(__fmaf_rn(
                    up_d_low, static_cast<float>(up_packed & 0x0f), -up_min_low));
                up_weight_tile[tile_index + 32] = __float2half_rn(__fmaf_rn(
                    up_d_high, static_cast<float>(up_packed >> 4), -up_min_high));
            }
        }
        __syncthreads();
        if (warp == 0) {
#pragma unroll
            for (unsigned int k_tile = 0; k_tile < 16; ++k_tile) {
                wmma::load_matrix_sync(a, activation_tile + k_tile * 16, 256);
                wmma::load_matrix_sync(b, gate_weight_tile + k_tile * 16, 256);
                wmma::mma_sync(gate_accumulator, a, b, gate_accumulator);
                wmma::load_matrix_sync(b, up_weight_tile + k_tile * 16, 256);
                wmma::mma_sync(up_accumulator, a, b, up_accumulator);
            }
        }
        __syncthreads();
    }

    if (warp == 0) {
        wmma::store_matrix_sync(
            gate_output_tile, gate_accumulator, 16, wmma::mem_row_major);
        wmma::store_matrix_sync(
            up_output_tile, up_accumulator, 16, wmma::mem_row_major);
    }
    __syncthreads();
    if (warp == 0) {
        for (unsigned int index = lane; index < 16 * 16; index += 32) {
            const unsigned int activation = index / 16;
            const unsigned int output_column = index % 16;
            const unsigned int row = output_start + output_column;
            if (activation < activation_rows && row < rows) {
                const float gate = gate_output_tile[index];
                const float exponent_input = __fmul_rn(
                    __fsub_rn(0.0f, gate), 1.4426950408889634f);
                float exp_negative;
                asm volatile("ex2.approx.f32 %0, %1;" : "=f"(exp_negative) : "f"(exponent_input));
                const float silu = __fdiv_rn(gate, __fadd_rn(1.0f, exp_negative));
                output[activation * rows + row] =
                    __fmul_rn(silu, up_output_tile[index]);
            }
        }
    }
}

extern "C" __global__ void xrt_q5_k_tensor_core_verify(
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
    const unsigned int thread = warp * 32 + lane;
    const unsigned int output_start = blockIdx.x * 16;
    const unsigned int blocks_per_row = cols / 256;
    __shared__ __align__(32) __half activation_tile[16 * 256];
    __shared__ __align__(32) __half weight_tile[16 * 256];
    __shared__ __align__(32) float output_tile[16 * 16];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);

    for (unsigned int block = 0; block < blocks_per_row; ++block) {
        const unsigned int k_start = block * 256;
        for (unsigned int index = thread; index < 16 * 256; index += 16 * 32) {
            const unsigned int activation = index / 256;
            const unsigned int k_offset = index % 256;
            activation_tile[index] = activation < activation_rows
                ? __float2half_rn(input[activation * cols + k_start + k_offset])
                : __float2half_rn(0.0f);
        }

        const unsigned int row = output_start + warp;
        if (row < rows) {
            const unsigned int block_index = row * blocks_per_row + block;
            const float block_d = d[block_index];
            const float block_dmin = dmin[block_index];
            const unsigned char* block_scales = scales + block_index * 12;
            const unsigned char* block_quants = quants + block_index * 128;
            const unsigned char high_word = high_bits[block_index * 32 + lane];
            unsigned int lane_scale = 0;
            unsigned int lane_minimum = 0;
            if (lane < 8) {
                xrt_q4_k_scale_min(
                    block_scales, lane, &lane_scale, &lane_minimum);
            }
#pragma unroll
            for (unsigned int group = 0; group < 4; ++group) {
                const unsigned int low_index = group * 2;
                const unsigned int high_index = low_index + 1;
                const float d_low = __shfl_sync(
                    0xffffffffu, __fmul_rn(block_d, static_cast<float>(lane_scale)), low_index);
                const float min_low = __shfl_sync(
                    0xffffffffu, __fmul_rn(block_dmin, static_cast<float>(lane_minimum)), low_index);
                const float d_high = __shfl_sync(
                    0xffffffffu, __fmul_rn(block_d, static_cast<float>(lane_scale)), high_index);
                const float min_high = __shfl_sync(
                    0xffffffffu, __fmul_rn(block_dmin, static_cast<float>(lane_minimum)), high_index);
                const unsigned char packed = block_quants[group * 32 + lane];
                const unsigned int quant_low = (packed & 0x0f) +
                    ((high_word & (1u << low_index)) != 0 ? 16u : 0u);
                const unsigned int quant_high = (packed >> 4) +
                    ((high_word & (1u << high_index)) != 0 ? 16u : 0u);
                weight_tile[warp * 256 + group * 64 + lane] = __float2half_rn(
                    __fmaf_rn(d_low, static_cast<float>(quant_low), -min_low));
                weight_tile[warp * 256 + group * 64 + 32 + lane] = __float2half_rn(
                    __fmaf_rn(d_high, static_cast<float>(quant_high), -min_high));
            }
        }
        __syncthreads();
        if (warp == 0) {
#pragma unroll
            for (unsigned int k_tile = 0; k_tile < 16; ++k_tile) {
                wmma::load_matrix_sync(a, activation_tile + k_tile * 16, 256);
                wmma::load_matrix_sync(b, weight_tile + k_tile * 16, 256);
                wmma::mma_sync(accumulator, a, b, accumulator);
            }
        }
        __syncthreads();
    }

    if (warp == 0) {
        wmma::store_matrix_sync(output_tile, accumulator, 16, wmma::mem_row_major);
        __syncwarp();
        for (unsigned int index = lane; index < 16 * 16; index += 32) {
            const unsigned int activation = index / 16;
            const unsigned int output_column = index % 16;
            const unsigned int row = output_start + output_column;
            if (activation < activation_rows && row < rows) {
                output[activation * rows + row] = output_tile[index];
            }
        }
    }
}

// Verification path for Q5_K matrices expanded once at upload time to the
// exact F16 values consumed by xrt_q5_k_tensor_core_verify. Eight warps own
// independent 16-column tiles. Keeping the matrix in output-major order also
// makes it a column-major KxN operand for WMMA, so the kernel can stream it
// directly without rebuilding a shared-memory weight tile for every request.
extern "C" __global__ __launch_bounds__(256, 2) void
xrt_q5_k_f16_weight_tensor_core_verify(
    const __half* weights,
    const __half* input,
    float* output,
    unsigned int rows,
    unsigned int cols,
    unsigned int activation_rows) {
    constexpr unsigned int OUTPUT_TILES = 8;
    const unsigned int lane = threadIdx.x;
    const unsigned int warp = threadIdx.y;
    const unsigned int output_start = blockIdx.x * (OUTPUT_TILES * 16) + warp * 16;
    if (lane >= 32 || warp >= OUTPUT_TILES || output_start >= rows) {
        return;
    }

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);

    for (unsigned int k_start = 0; k_start < cols; k_start += 16) {
        wmma::load_matrix_sync(a, input + k_start, cols);
        wmma::load_matrix_sync(b, weights + output_start * cols + k_start, cols);
        wmma::mma_sync(accumulator, a, b, accumulator);
    }

    __shared__ __align__(32) float output_tiles[OUTPUT_TILES][16 * 16];
    wmma::store_matrix_sync(
        output_tiles[warp], accumulator, 16, wmma::mem_row_major);
    __syncwarp();
    for (unsigned int index = lane; index < 16 * 16; index += 32) {
        const unsigned int activation = index / 16;
        const unsigned int output_column = index % 16;
        const unsigned int row = output_start + output_column;
        if (activation < activation_rows && row < rows) {
            output[activation * rows + row] = output_tiles[warp][index];
        }
    }
}

// Q5_K verifier consuming a caller-preconverted F16 activation matrix. The
// conversion is identical to the original kernel's __float2half_rn operation,
// but is performed once per projection instead of once per output CTA.
extern "C" __global__ void xrt_q5_k_tensor_core_verify_f16_input(
    const float* d,
    const float* dmin,
    const unsigned char* scales,
    const unsigned char* high_bits,
    const unsigned char* quants,
    const __half* input,
    float* output,
    unsigned int rows,
    unsigned int cols,
    unsigned int activation_rows) {
    const unsigned int lane = threadIdx.x;
    const unsigned int warp = threadIdx.y;
    const unsigned int thread = warp * 32 + lane;
    const unsigned int output_start = blockIdx.x * 16;
    const unsigned int blocks_per_row = cols / 256;
    __shared__ __align__(32) __half activation_tile[16 * 256];
    __shared__ __align__(32) __half weight_tile[16 * 256];
    __shared__ __align__(32) float output_tile[16 * 16];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);

    for (unsigned int block = 0; block < blocks_per_row; ++block) {
        const unsigned int k_start = block * 256;
        for (unsigned int index = thread; index < 16 * 256; index += 16 * 32) {
            const unsigned int activation = index / 256;
            const unsigned int k_offset = index % 256;
            activation_tile[index] = activation < activation_rows
                ? input[activation * cols + k_start + k_offset]
                : __float2half_rn(0.0f);
        }

        const unsigned int row = output_start + warp;
        if (row < rows) {
            const unsigned int block_index = row * blocks_per_row + block;
            const float block_d = d[block_index];
            const float block_dmin = dmin[block_index];
            const unsigned char* block_scales = scales + block_index * 12;
            const unsigned char* block_quants = quants + block_index * 128;
            const unsigned char high_word = high_bits[block_index * 32 + lane];
            unsigned int lane_scale = 0;
            unsigned int lane_minimum = 0;
            if (lane < 8) {
                xrt_q4_k_scale_min(
                    block_scales, lane, &lane_scale, &lane_minimum);
            }
#pragma unroll
            for (unsigned int group = 0; group < 4; ++group) {
                const unsigned int low_index = group * 2;
                const unsigned int high_index = low_index + 1;
                const float d_low = __shfl_sync(
                    0xffffffffu,
                    __fmul_rn(block_d, static_cast<float>(lane_scale)),
                    low_index);
                const float min_low = __shfl_sync(
                    0xffffffffu,
                    __fmul_rn(block_dmin, static_cast<float>(lane_minimum)),
                    low_index);
                const float d_high = __shfl_sync(
                    0xffffffffu,
                    __fmul_rn(block_d, static_cast<float>(lane_scale)),
                    high_index);
                const float min_high = __shfl_sync(
                    0xffffffffu,
                    __fmul_rn(block_dmin, static_cast<float>(lane_minimum)),
                    high_index);
                const unsigned char packed = block_quants[group * 32 + lane];
                const unsigned int quant_low = (packed & 0x0f) +
                    ((high_word & (1u << low_index)) != 0 ? 16u : 0u);
                const unsigned int quant_high = (packed >> 4) +
                    ((high_word & (1u << high_index)) != 0 ? 16u : 0u);
                weight_tile[warp * 256 + group * 64 + lane] = __float2half_rn(
                    __fmaf_rn(d_low, static_cast<float>(quant_low), -min_low));
                weight_tile[warp * 256 + group * 64 + 32 + lane] =
                    __float2half_rn(__fmaf_rn(
                        d_high, static_cast<float>(quant_high), -min_high));
            }
        }
        __syncthreads();
        if (warp == 0) {
#pragma unroll
            for (unsigned int k_tile = 0; k_tile < 16; ++k_tile) {
                wmma::load_matrix_sync(a, activation_tile + k_tile * 16, 256);
                wmma::load_matrix_sync(b, weight_tile + k_tile * 16, 256);
                wmma::mma_sync(accumulator, a, b, accumulator);
            }
        }
        __syncthreads();
    }

    if (warp == 0) {
        wmma::store_matrix_sync(output_tile, accumulator, 16, wmma::mem_row_major);
        __syncwarp();
        for (unsigned int index = lane; index < 16 * 16; index += 32) {
            const unsigned int activation = index / 16;
            const unsigned int output_column = index % 16;
            const unsigned int row = output_start + output_column;
            if (activation < activation_rows && row < rows) {
                output[activation * rows + row] = output_tile[index];
            }
        }
    }
}

// Two-output-tile Q5_K verifier for speculative windows. This preserves the
// original verifier's per-output WMMA accumulation order while amortizing the
// activation load and CTA barriers across 32 output rows. Two warps execute
// independent 16x16 output tiles after the existing sixteen warps decode the
// corresponding Q5_K rows.
extern "C" __global__ __launch_bounds__(512, 2)
void xrt_q5_k_tensor_core_verify_n32(
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
    constexpr unsigned int OUTPUT_ROWS = 32;
    constexpr unsigned int OUTPUT_TILES = 2;
    const unsigned int lane = threadIdx.x;
    const unsigned int warp = threadIdx.y;
    const unsigned int thread = warp * 32 + lane;
    const unsigned int output_start = blockIdx.x * OUTPUT_ROWS;
    const unsigned int blocks_per_row = cols / 256;
    __shared__ __align__(32) __half activation_tile[16 * 256];
    __shared__ __align__(32) __half weight_tile[OUTPUT_ROWS * 256];
    __shared__ __align__(32) float output_tile[OUTPUT_TILES][16 * 16];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);

    for (unsigned int block = 0; block < blocks_per_row; ++block) {
        const unsigned int k_start = block * 256;
        for (unsigned int index = thread; index < 16 * 256;
             index += blockDim.y * 32) {
            const unsigned int activation = index / 256;
            const unsigned int k_offset = index % 256;
            activation_tile[index] = activation < activation_rows
                ? __float2half_rn(input[activation * cols + k_start + k_offset])
                : __float2half_rn(0.0f);
        }

#pragma unroll
        for (unsigned int row_group = 0; row_group < OUTPUT_TILES; ++row_group) {
            const unsigned int output_offset = warp + row_group * 16;
            const unsigned int row = output_start + output_offset;
            const unsigned int tile_base = output_offset * 256;
            if (row < rows) {
                const unsigned int block_index = row * blocks_per_row + block;
                const float block_d = d[block_index];
                const float block_dmin = dmin[block_index];
                const unsigned char* block_scales = scales + block_index * 12;
                const unsigned char* block_quants = quants + block_index * 128;
                const unsigned char high_word = high_bits[block_index * 32 + lane];
                unsigned int lane_scale = 0;
                unsigned int lane_minimum = 0;
                if (lane < 8) {
                    xrt_q4_k_scale_min(
                        block_scales, lane, &lane_scale, &lane_minimum);
                }
#pragma unroll
                for (unsigned int group = 0; group < 4; ++group) {
                    const unsigned int low_index = group * 2;
                    const unsigned int high_index = low_index + 1;
                    const float d_low = __shfl_sync(
                        0xffffffffu,
                        __fmul_rn(block_d, static_cast<float>(lane_scale)),
                        low_index);
                    const float min_low = __shfl_sync(
                        0xffffffffu,
                        __fmul_rn(block_dmin, static_cast<float>(lane_minimum)),
                        low_index);
                    const float d_high = __shfl_sync(
                        0xffffffffu,
                        __fmul_rn(block_d, static_cast<float>(lane_scale)),
                        high_index);
                    const float min_high = __shfl_sync(
                        0xffffffffu,
                        __fmul_rn(block_dmin, static_cast<float>(lane_minimum)),
                        high_index);
                    const unsigned char packed = block_quants[group * 32 + lane];
                    const unsigned int quant_low = (packed & 0x0f) +
                        ((high_word & (1u << low_index)) != 0 ? 16u : 0u);
                    const unsigned int quant_high = (packed >> 4) +
                        ((high_word & (1u << high_index)) != 0 ? 16u : 0u);
                    weight_tile[tile_base + group * 64 + lane] = __float2half_rn(
                        __fmaf_rn(d_low, static_cast<float>(quant_low), -min_low));
                    weight_tile[tile_base + group * 64 + 32 + lane] =
                        __float2half_rn(__fmaf_rn(
                            d_high, static_cast<float>(quant_high), -min_high));
                }
            } else {
                for (unsigned int index = lane; index < 256; index += 32) {
                    weight_tile[tile_base + index] = __float2half_rn(0.0f);
                }
            }
        }
        __syncthreads();
        if (warp < OUTPUT_TILES) {
            const __half* warp_weights = weight_tile + warp * 16 * 256;
#pragma unroll
            for (unsigned int k_tile = 0; k_tile < 16; ++k_tile) {
                wmma::load_matrix_sync(a, activation_tile + k_tile * 16, 256);
                wmma::load_matrix_sync(b, warp_weights + k_tile * 16, 256);
                wmma::mma_sync(accumulator, a, b, accumulator);
            }
        }
        __syncthreads();
    }

    if (warp < OUTPUT_TILES) {
        wmma::store_matrix_sync(
            output_tile[warp], accumulator, 16, wmma::mem_row_major);
        __syncwarp();
        for (unsigned int index = lane; index < 16 * 16; index += 32) {
            const unsigned int activation = index / 16;
            const unsigned int output_column = index % 16;
            const unsigned int row = output_start + warp * 16 + output_column;
            if (activation < activation_rows && row < rows) {
                output[activation * rows + row] = output_tile[warp][index];
            }
        }
    }
}

// Double-buffered Q5_K verifier. Sixteen decoder warps prepare the next
// packed 256-column block while a dedicated warp executes WMMA on the current
// block. The compute warp visits blocks and K tiles in exactly the same order
// as xrt_q5_k_tensor_core_verify, preserving its accumulator ordering.
extern "C" __global__ __launch_bounds__(544, 2)
void xrt_q5_k_tensor_core_verify_pipelined(
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
    constexpr unsigned int DECODER_WARPS = 16;
    constexpr unsigned int COMPUTE_WARP = 16;
    const unsigned int lane = threadIdx.x;
    const unsigned int warp = threadIdx.y;
    const unsigned int thread = warp * 32 + lane;
    const unsigned int output_start = blockIdx.x * 16;
    const unsigned int blocks_per_row = cols / 256;
    __shared__ __align__(32) __half activation_tile[2][16 * 256];
    __shared__ __align__(32) __half weight_tile[2][16 * 256];
    __shared__ __align__(32) float output_tile[16 * 16];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);

    if (blocks_per_row == 0) {
        return;
    }

    if (warp < DECODER_WARPS) {
        for (unsigned int index = thread; index < 16 * 256;
             index += DECODER_WARPS * 32) {
            const unsigned int activation = index / 256;
            const unsigned int k_offset = index % 256;
            activation_tile[0][index] = activation < activation_rows
                ? __float2half_rn(input[activation * cols + k_offset])
                : __float2half_rn(0.0f);
        }
        const unsigned int row = output_start + warp;
        if (row < rows) {
            const unsigned int block_index = row * blocks_per_row;
            const float block_d = d[block_index];
            const float block_dmin = dmin[block_index];
            const unsigned char* block_scales = scales + block_index * 12;
            const unsigned char* block_quants = quants + block_index * 128;
            const unsigned char high_word = high_bits[block_index * 32 + lane];
            unsigned int lane_scale = 0;
            unsigned int lane_minimum = 0;
            if (lane < 8) {
                xrt_q4_k_scale_min(block_scales, lane, &lane_scale, &lane_minimum);
            }
#pragma unroll
            for (unsigned int group = 0; group < 4; ++group) {
                const unsigned int low_index = group * 2;
                const unsigned int high_index = low_index + 1;
                const float d_low = __shfl_sync(
                    0xffffffffu, __fmul_rn(block_d, static_cast<float>(lane_scale)), low_index);
                const float min_low = __shfl_sync(
                    0xffffffffu, __fmul_rn(block_dmin, static_cast<float>(lane_minimum)), low_index);
                const float d_high = __shfl_sync(
                    0xffffffffu, __fmul_rn(block_d, static_cast<float>(lane_scale)), high_index);
                const float min_high = __shfl_sync(
                    0xffffffffu, __fmul_rn(block_dmin, static_cast<float>(lane_minimum)), high_index);
                const unsigned char packed = block_quants[group * 32 + lane];
                const unsigned int quant_low = (packed & 0x0f) +
                    ((high_word & (1u << low_index)) != 0 ? 16u : 0u);
                const unsigned int quant_high = (packed >> 4) +
                    ((high_word & (1u << high_index)) != 0 ? 16u : 0u);
                weight_tile[0][warp * 256 + group * 64 + lane] = __float2half_rn(
                    __fmaf_rn(d_low, static_cast<float>(quant_low), -min_low));
                weight_tile[0][warp * 256 + group * 64 + 32 + lane] = __float2half_rn(
                    __fmaf_rn(d_high, static_cast<float>(quant_high), -min_high));
            }
        }
    }
    __syncthreads();

    for (unsigned int block = 0; block < blocks_per_row; ++block) {
        const unsigned int current = block & 1u;
        const unsigned int next = current ^ 1u;
        if (warp == COMPUTE_WARP) {
#pragma unroll
            for (unsigned int k_tile = 0; k_tile < 16; ++k_tile) {
                wmma::load_matrix_sync(a, activation_tile[current] + k_tile * 16, 256);
                wmma::load_matrix_sync(b, weight_tile[current] + k_tile * 16, 256);
                wmma::mma_sync(accumulator, a, b, accumulator);
            }
        } else if (warp < DECODER_WARPS && block + 1 < blocks_per_row) {
            const unsigned int next_block = block + 1;
            const unsigned int k_start = next_block * 256;
            for (unsigned int index = thread; index < 16 * 256;
                 index += DECODER_WARPS * 32) {
                const unsigned int activation = index / 256;
                const unsigned int k_offset = index % 256;
                activation_tile[next][index] = activation < activation_rows
                    ? __float2half_rn(input[activation * cols + k_start + k_offset])
                    : __float2half_rn(0.0f);
            }
            const unsigned int row = output_start + warp;
            if (row < rows) {
                const unsigned int block_index = row * blocks_per_row + next_block;
                const float block_d = d[block_index];
                const float block_dmin = dmin[block_index];
                const unsigned char* block_scales = scales + block_index * 12;
                const unsigned char* block_quants = quants + block_index * 128;
                const unsigned char high_word = high_bits[block_index * 32 + lane];
                unsigned int lane_scale = 0;
                unsigned int lane_minimum = 0;
                if (lane < 8) {
                    xrt_q4_k_scale_min(block_scales, lane, &lane_scale, &lane_minimum);
                }
#pragma unroll
                for (unsigned int group = 0; group < 4; ++group) {
                    const unsigned int low_index = group * 2;
                    const unsigned int high_index = low_index + 1;
                    const float d_low = __shfl_sync(
                        0xffffffffu, __fmul_rn(block_d, static_cast<float>(lane_scale)), low_index);
                    const float min_low = __shfl_sync(
                        0xffffffffu, __fmul_rn(block_dmin, static_cast<float>(lane_minimum)), low_index);
                    const float d_high = __shfl_sync(
                        0xffffffffu, __fmul_rn(block_d, static_cast<float>(lane_scale)), high_index);
                    const float min_high = __shfl_sync(
                        0xffffffffu, __fmul_rn(block_dmin, static_cast<float>(lane_minimum)), high_index);
                    const unsigned char packed = block_quants[group * 32 + lane];
                    const unsigned int quant_low = (packed & 0x0f) +
                        ((high_word & (1u << low_index)) != 0 ? 16u : 0u);
                    const unsigned int quant_high = (packed >> 4) +
                        ((high_word & (1u << high_index)) != 0 ? 16u : 0u);
                    weight_tile[next][warp * 256 + group * 64 + lane] = __float2half_rn(
                        __fmaf_rn(d_low, static_cast<float>(quant_low), -min_low));
                    weight_tile[next][warp * 256 + group * 64 + 32 + lane] = __float2half_rn(
                        __fmaf_rn(d_high, static_cast<float>(quant_high), -min_high));
                }
            }
        }
        __syncthreads();
    }

    if (warp == COMPUTE_WARP) {
        wmma::store_matrix_sync(output_tile, accumulator, 16, wmma::mem_row_major);
        __syncwarp();
        for (unsigned int index = lane; index < 16 * 16; index += 32) {
            const unsigned int activation = index / 16;
            const unsigned int output_column = index % 16;
            const unsigned int row = output_start + output_column;
            if (activation < activation_rows && row < rows) {
                output[activation * rows + row] = output_tile[index];
            }
        }
    }
}

extern "C" __global__ void xrt_q6_k_tensor_core_verify(
    const float* d,
    const unsigned char* blocks,
    const float* input,
    float* output,
    unsigned int rows,
    unsigned int cols,
    unsigned int activation_rows) {
    const unsigned int lane = threadIdx.x;
    const unsigned int warp = threadIdx.y;
    const unsigned int thread = warp * 32 + lane;
    const unsigned int output_start = blockIdx.x * 16;
    const unsigned int blocks_per_row = cols / 256;
    __shared__ __align__(32) __half activation_tile[16 * 256];
    __shared__ __align__(32) __half weight_tile[16 * 256];
    __shared__ __align__(32) float output_tile[16 * 16];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);

    for (unsigned int block = 0; block < blocks_per_row; ++block) {
        const unsigned int k_start = block * 256;
        for (unsigned int index = thread; index < 16 * 256; index += 16 * 32) {
            const unsigned int activation = index / 256;
            const unsigned int k_offset = index % 256;
            activation_tile[index] = activation < activation_rows
                ? __float2half_rn(input[activation * cols + k_start + k_offset])
                : __float2half_rn(0.0f);
        }

        const unsigned int row = output_start + warp;
        if (row < rows) {
            const unsigned int block_index = row * blocks_per_row + block;
            const float block_d = d[block_index];
            const unsigned char* encoded = blocks + block_index * 210;
            const unsigned char* ql = encoded;
            const unsigned char* qh = encoded + 128;
            const signed char* block_scales =
                reinterpret_cast<const signed char*>(encoded + 192);
#pragma unroll
            for (unsigned int group = 0; group < 2; ++group) {
                const unsigned int ql_base = group * 64;
                const unsigned int qh_base = group * 32;
                const unsigned int scale_base = group * 8;
                const unsigned int feature_base = group * 128;
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
                        block_d * static_cast<float>(block_scales[scale_index]) *
                        static_cast<float>(quantized[subgroup]);
                    const unsigned int feature = feature_base + subgroup * 32 + lane;
                    weight_tile[warp * 256 + feature] = __float2half_rn(weight);
                }
            }
        }
        __syncthreads();
        if (warp == 0) {
#pragma unroll
            for (unsigned int k_tile = 0; k_tile < 16; ++k_tile) {
                wmma::load_matrix_sync(a, activation_tile + k_tile * 16, 256);
                wmma::load_matrix_sync(b, weight_tile + k_tile * 16, 256);
                wmma::mma_sync(accumulator, a, b, accumulator);
            }
        }
        __syncthreads();
    }

    if (warp == 0) {
        wmma::store_matrix_sync(output_tile, accumulator, 16, wmma::mem_row_major);
        __syncwarp();
        for (unsigned int index = lane; index < 16 * 16; index += 32) {
            const unsigned int activation = index / 16;
            const unsigned int output_column = index % 16;
            const unsigned int row = output_start + output_column;
            if (activation < activation_rows && row < rows) {
                output[activation * rows + row] = output_tile[index];
            }
        }
    }
}

// Q6_K verifier consuming a once-preconverted F16 activation matrix. This is
// particularly useful for the vocabulary head, where thousands of output CTAs
// otherwise repeat the same F32-to-F16 activation conversion.
extern "C" __global__ void xrt_q6_k_tensor_core_verify_f16_input(
    const float* d,
    const unsigned char* blocks,
    const __half* input,
    float* output,
    unsigned int rows,
    unsigned int cols,
    unsigned int activation_rows) {
    const unsigned int lane = threadIdx.x;
    const unsigned int warp = threadIdx.y;
    const unsigned int thread = warp * 32 + lane;
    const unsigned int output_start = blockIdx.x * 16;
    const unsigned int blocks_per_row = cols / 256;
    __shared__ __align__(32) __half activation_tile[16 * 256];
    __shared__ __align__(32) __half weight_tile[16 * 256];
    __shared__ __align__(32) float output_tile[16 * 16];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);

    for (unsigned int block = 0; block < blocks_per_row; ++block) {
        const unsigned int k_start = block * 256;
        for (unsigned int index = thread; index < 16 * 256; index += 16 * 32) {
            const unsigned int activation = index / 256;
            const unsigned int k_offset = index % 256;
            activation_tile[index] = activation < activation_rows
                ? input[activation * cols + k_start + k_offset]
                : __float2half_rn(0.0f);
        }

        const unsigned int row = output_start + warp;
        if (row < rows) {
            const unsigned int block_index = row * blocks_per_row + block;
            const float block_d = d[block_index];
            const unsigned char* encoded = blocks + block_index * 210;
            const unsigned char* ql = encoded;
            const unsigned char* qh = encoded + 128;
            const signed char* block_scales =
                reinterpret_cast<const signed char*>(encoded + 192);
#pragma unroll
            for (unsigned int group = 0; group < 2; ++group) {
                const unsigned int ql_base = group * 64;
                const unsigned int qh_base = group * 32;
                const unsigned int scale_base = group * 8;
                const unsigned int feature_base = group * 128;
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
                        block_d * static_cast<float>(block_scales[scale_index]) *
                        static_cast<float>(quantized[subgroup]);
                    const unsigned int feature = feature_base + subgroup * 32 + lane;
                    weight_tile[warp * 256 + feature] = __float2half_rn(weight);
                }
            }
        }
        __syncthreads();
        if (warp == 0) {
#pragma unroll
            for (unsigned int k_tile = 0; k_tile < 16; ++k_tile) {
                wmma::load_matrix_sync(a, activation_tile + k_tile * 16, 256);
                wmma::load_matrix_sync(b, weight_tile + k_tile * 16, 256);
                wmma::mma_sync(accumulator, a, b, accumulator);
            }
        }
        __syncthreads();
    }

    if (warp == 0) {
        wmma::store_matrix_sync(output_tile, accumulator, 16, wmma::mem_row_major);
        __syncwarp();
        for (unsigned int index = lane; index < 16 * 16; index += 32) {
            const unsigned int activation = index / 16;
            const unsigned int output_column = index % 16;
            const unsigned int row = output_start + output_column;
            if (activation < activation_rows && row < rows) {
                output[activation * rows + row] = output_tile[index];
            }
        }
    }
}

// Four-output-tile Q6_K verifier. It retains the original per-output WMMA
// order while sharing activation loads and CTA barriers across 64 vocabulary
// rows, which is especially useful for the large Q6_K language-model head.
extern "C" __global__ __launch_bounds__(512, 1)
void xrt_q6_k_tensor_core_verify_n64(
    const float* d,
    const unsigned char* blocks,
    const float* input,
    float* output,
    unsigned int rows,
    unsigned int cols,
    unsigned int activation_rows) {
    constexpr unsigned int OUTPUT_ROWS = 64;
    constexpr unsigned int OUTPUT_TILES = 4;
    const unsigned int lane = threadIdx.x;
    const unsigned int warp = threadIdx.y;
    const unsigned int thread = warp * 32 + lane;
    const unsigned int output_start = blockIdx.x * OUTPUT_ROWS;
    const unsigned int blocks_per_row = cols / 256;
    __shared__ __align__(32) __half activation_tile[16 * 256];
    __shared__ __align__(32) __half weight_tile[OUTPUT_ROWS * 256];
    __shared__ __align__(32) float output_tile[OUTPUT_TILES][16 * 16];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);

    for (unsigned int block = 0; block < blocks_per_row; ++block) {
        const unsigned int k_start = block * 256;
        for (unsigned int index = thread; index < 16 * 256;
             index += blockDim.y * 32) {
            const unsigned int activation = index / 256;
            const unsigned int k_offset = index % 256;
            activation_tile[index] = activation < activation_rows
                ? __float2half_rn(input[activation * cols + k_start + k_offset])
                : __float2half_rn(0.0f);
        }

#pragma unroll
        for (unsigned int row_group = 0; row_group < OUTPUT_TILES; ++row_group) {
            const unsigned int output_offset = warp + row_group * 16;
            const unsigned int row = output_start + output_offset;
            const unsigned int tile_base = output_offset * 256;
            if (row < rows) {
                const unsigned int block_index = row * blocks_per_row + block;
                const float block_d = d[block_index];
                const unsigned char* encoded = blocks + block_index * 210;
                const unsigned char* ql = encoded;
                const unsigned char* qh = encoded + 128;
                const signed char* block_scales =
                    reinterpret_cast<const signed char*>(encoded + 192);
#pragma unroll
                for (unsigned int group = 0; group < 2; ++group) {
                    const unsigned int ql_base = group * 64;
                    const unsigned int qh_base = group * 32;
                    const unsigned int scale_base = group * 8;
                    const unsigned int feature_base = group * 128;
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
                            block_d * static_cast<float>(block_scales[scale_index]) *
                            static_cast<float>(quantized[subgroup]);
                        const unsigned int feature =
                            feature_base + subgroup * 32 + lane;
                        weight_tile[tile_base + feature] = __float2half_rn(weight);
                    }
                }
            } else {
                for (unsigned int index = lane; index < 256; index += 32) {
                    weight_tile[tile_base + index] = __float2half_rn(0.0f);
                }
            }
        }
        __syncthreads();
        if (warp < OUTPUT_TILES) {
            const __half* warp_weights = weight_tile + warp * 16 * 256;
#pragma unroll
            for (unsigned int k_tile = 0; k_tile < 16; ++k_tile) {
                wmma::load_matrix_sync(a, activation_tile + k_tile * 16, 256);
                wmma::load_matrix_sync(b, warp_weights + k_tile * 16, 256);
                wmma::mma_sync(accumulator, a, b, accumulator);
            }
        }
        __syncthreads();
    }

    if (warp < OUTPUT_TILES) {
        wmma::store_matrix_sync(
            output_tile[warp], accumulator, 16, wmma::mem_row_major);
        __syncwarp();
        for (unsigned int index = lane; index < 16 * 16; index += 32) {
            const unsigned int activation = index / 16;
            const unsigned int output_column = index % 16;
            const unsigned int row = output_start + warp * 16 + output_column;
            if (activation < activation_rows && row < rows) {
                output[activation * rows + row] = output_tile[warp][index];
            }
        }
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
