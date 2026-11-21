// Decode-specialized dense F32 kernels. These preserve the established scalar
// accumulation orders while exposing independent chains/outputs to the GPU.

extern "C" __global__ void rmsnorm_kernel(
    const float* input,
    const float* weight,
    float* output,
    unsigned int rows,
    unsigned int cols,
    float epsilon) {
    const unsigned int row = blockIdx.x;
    const unsigned int thread = threadIdx.x;
    if (row >= rows || thread >= 256) {
        return;
    }

    __shared__ float chains[8];
    __shared__ float inverse_rms;
    const unsigned int row_base = row * cols;
    const unsigned int grouped_cols = (cols / 8) * 8;

    if (thread < 8) {
        float accumulator = 0.0f;
        for (unsigned int col = thread; col < grouped_cols; col += 8) {
            const float value = input[row_base + col];
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
        for (unsigned int col = grouped_cols; col < cols; ++col) {
            const float value = input[row_base + col];
            sum = __fmaf_rn(value, value, sum);
        }
        const float mean = __fdiv_rn(sum, static_cast<float>(cols));
        const float root = __fsqrt_rn(__fadd_rn(mean, epsilon));
        inverse_rms = __fdiv_rn(1.0f, root);
    }
    __syncthreads();

    for (unsigned int col = thread; col < cols; col += blockDim.x) {
        const float normalized = __fmul_rn(input[row_base + col], inverse_rms);
        output[row_base + col] = __fmul_rn(normalized, weight[col]);
    }
}

extern "C" __global__ void rmsnorm_unweighted_kernel(
    const float* input,
    float* output,
    unsigned int rows,
    unsigned int cols,
    float epsilon) {
    const unsigned int row = blockIdx.x;
    const unsigned int thread = threadIdx.x;
    if (row >= rows || thread >= 256) {
        return;
    }

    __shared__ float inverse_rms;
    const unsigned int row_base = row * cols;
    if (thread == 0) {
        float sum = 0.0f;
        for (unsigned int col = 0; col < cols; ++col) {
            const float value = input[row_base + col];
            sum = __fmaf_rn(value, value, sum);
        }
        const float mean = __fdiv_rn(sum, static_cast<float>(cols));
        const float root = __fsqrt_rn(__fadd_rn(mean, epsilon));
        inverse_rms = __fdiv_rn(1.0f, root);
    }
    __syncthreads();
    for (unsigned int col = thread; col < cols; col += blockDim.x) {
        output[row_base + col] = __fmul_rn(input[row_base + col], inverse_rms);
    }
}

// One block owns one output. The single active thread retains the original
// depth-order FMA chain, while independent outputs run concurrently instead
// of occupying only a few warps in a 16x16 general-matmul launch.
extern "C" __global__ void matvec_serial_kernel(
    const float* input,
    const float* matrix,
    float* output,
    unsigned int rows,
    unsigned int depth,
    unsigned int cols) {
    const unsigned int col = blockIdx.x;
    if (rows != 1 || col >= cols || threadIdx.x != 0) {
        return;
    }
    float accumulator = 0.0f;
    for (unsigned int index = 0; index < depth; ++index) {
        accumulator = __fmaf_rn(
            input[index],
            matrix[index * cols + col],
            accumulator);
    }
    output[col] = accumulator;
}

// Eight independent depth-order chains hide FMA latency while retaining a
// deterministic, fixed reduction order. This is the same accumulation shape
// XRT already uses for its CPU-vectorized dense dot products.
extern "C" __global__ void matvec_eight_chain_kernel(
    const float* input,
    const float* matrix,
    float* output,
    unsigned int rows,
    unsigned int depth,
    unsigned int cols) {
    const unsigned int col = blockIdx.x;
    const unsigned int chain = threadIdx.x;
    if (rows != 1 || col >= cols || chain >= 8) {
        return;
    }
    float accumulator = 0.0f;
    for (unsigned int index = chain; index < depth; index += 8) {
        accumulator = __fmaf_rn(
            input[index],
            matrix[index * cols + col],
            accumulator);
    }
    __shared__ float chains[8];
    chains[chain] = accumulator;
    __syncthreads();
    if (chain == 0) {
        float sum = __fadd_rn(chains[0], chains[1]);
        sum = __fadd_rn(sum, chains[2]);
        sum = __fadd_rn(sum, chains[3]);
        sum = __fadd_rn(sum, chains[4]);
        sum = __fadd_rn(sum, chains[5]);
        sum = __fadd_rn(sum, chains[6]);
        output[col] = __fadd_rn(sum, chains[7]);
    }
}

extern "C" __global__ void matmul_eight_chain_kernel(
    const float* left,
    const float* right,
    float* output,
    unsigned int rows,
    unsigned int depth,
    unsigned int cols) {
    const unsigned int col = blockIdx.x;
    const unsigned int row = blockIdx.y;
    const unsigned int chain = threadIdx.x;
    if (row >= rows || col >= cols || chain >= 8) {
        return;
    }
    float accumulator = 0.0f;
    for (unsigned int index = chain; index < depth; index += 8) {
        accumulator = __fmaf_rn(
            left[row * depth + index],
            right[index * cols + col],
            accumulator);
    }
    __shared__ float chains[8];
    chains[chain] = accumulator;
    __syncthreads();
    if (chain == 0) {
        float sum = __fadd_rn(chains[0], chains[1]);
        sum = __fadd_rn(sum, chains[2]);
        sum = __fadd_rn(sum, chains[3]);
        sum = __fadd_rn(sum, chains[4]);
        sum = __fadd_rn(sum, chains[5]);
        sum = __fadd_rn(sum, chains[6]);
        output[row * cols + col] = __fadd_rn(sum, chains[7]);
    }
}

// Compact column tiles keep enough independent blocks resident for the narrow
// DeltaNet alpha/beta projections while coalescing RHS reads. Each output
// retains the exact eight-chain FMA and reduction order used above.
#define XRT_DEFINE_EIGHT_CHAIN_COALESCED(NAME, TILE)                       \
extern "C" __global__ void NAME(                                         \
    const float* left,                                                     \
    const float* right,                                                    \
    float* output,                                                         \
    unsigned int rows,                                                     \
    unsigned int depth,                                                    \
    unsigned int cols) {                                                   \
    const unsigned int lane = threadIdx.x;                                 \
    const unsigned int chain = threadIdx.y;                                \
    const unsigned int col = blockIdx.x * TILE + lane;                     \
    const unsigned int row = blockIdx.y;                                   \
    float accumulator = 0.0f;                                              \
    if (row < rows && col < cols && lane < TILE && chain < 8) {            \
        for (unsigned int index = chain; index < depth; index += 8) {       \
            accumulator = __fmaf_rn(                                       \
                left[row * depth + index],                                 \
                right[index * cols + col],                                 \
                accumulator);                                              \
        }                                                                  \
    }                                                                      \
    __shared__ float chains[8][TILE];                                      \
    chains[chain][lane] = accumulator;                                     \
    __syncthreads();                                                       \
    if (chain == 0 && row < rows && col < cols) {                          \
        float sum = __fadd_rn(chains[0][lane], chains[1][lane]);           \
        sum = __fadd_rn(sum, chains[2][lane]);                             \
        sum = __fadd_rn(sum, chains[3][lane]);                             \
        sum = __fadd_rn(sum, chains[4][lane]);                             \
        sum = __fadd_rn(sum, chains[5][lane]);                             \
        sum = __fadd_rn(sum, chains[6][lane]);                             \
        output[row * cols + col] = __fadd_rn(sum, chains[7][lane]);        \
    }                                                                      \
}

XRT_DEFINE_EIGHT_CHAIN_COALESCED(matmul_eight_chain_coalesced4_kernel, 4)
XRT_DEFINE_EIGHT_CHAIN_COALESCED(matmul_eight_chain_coalesced8_kernel, 8)
XRT_DEFINE_EIGHT_CHAIN_COALESCED(matmul_eight_chain_coalesced16_kernel, 16)

// Preserve the established eight-chain FMA/reduction order while processing
// two activation rows per CTA. Four adjacent output columns keep one warp per
// block; each RHS value is reused for both rows and the two independent FMA
// chains hide dependency latency without changing either row's arithmetic.
extern "C" __global__ void matmul_eight_chain_coalesced4_rows2_kernel(
    const float* left,
    const float* right,
    float* output,
    unsigned int rows,
    unsigned int depth,
    unsigned int cols) {
    const unsigned int lane = threadIdx.x;
    const unsigned int chain = threadIdx.y;
    const unsigned int col = blockIdx.x * 4 + lane;
    const unsigned int row0 = blockIdx.y * 2;
    const unsigned int row1 = row0 + 1;
    float accumulator0 = 0.0f;
    float accumulator1 = 0.0f;
    if (row0 < rows && col < cols && lane < 4 && chain < 8) {
        for (unsigned int index = chain; index < depth; index += 8) {
            const float rhs = right[index * cols + col];
            accumulator0 = __fmaf_rn(
                left[row0 * depth + index], rhs, accumulator0);
            if (row1 < rows) {
                accumulator1 = __fmaf_rn(
                    left[row1 * depth + index], rhs, accumulator1);
            }
        }
    }
    __shared__ float chains[2][8][4];
    chains[0][chain][lane] = accumulator0;
    chains[1][chain][lane] = accumulator1;
    __syncthreads();
    if (chain == 0 && col < cols) {
#pragma unroll
        for (unsigned int local_row = 0; local_row < 2; ++local_row) {
            const unsigned int row = row0 + local_row;
            if (row >= rows) {
                continue;
            }
            float sum = __fadd_rn(chains[local_row][0][lane], chains[local_row][1][lane]);
            sum = __fadd_rn(sum, chains[local_row][2][lane]);
            sum = __fadd_rn(sum, chains[local_row][3][lane]);
            sum = __fadd_rn(sum, chains[local_row][4][lane]);
            sum = __fadd_rn(sum, chains[local_row][5][lane]);
            sum = __fadd_rn(sum, chains[local_row][6][lane]);
            output[row * cols + col] = __fadd_rn(sum, chains[local_row][7][lane]);
        }
    }
}

// Verification-only storage specialization for the narrow DeltaNet alpha/beta
// projections. The source GGUF tensor remains authoritative F32; upload may
// retain an additional F16 RHS so concurrent verifier streams move half as
// many weight bytes. Accumulation and the eight-chain reduction stay F32.
extern "C" __global__ void matmul_eight_chain_coalesced8_f16_rhs_kernel(
    const float* left,
    const unsigned short* right,
    float* output,
    unsigned int rows,
    unsigned int depth,
    unsigned int cols) {
    const unsigned int lane = threadIdx.x;
    const unsigned int chain = threadIdx.y;
    const unsigned int col = blockIdx.x * 8 + lane;
    const unsigned int row = blockIdx.y;
    float accumulator = 0.0f;
    if (row < rows && col < cols && lane < 8 && chain < 8) {
        for (unsigned int index = chain; index < depth; index += 8) {
            float rhs;
            asm("cvt.f32.f16 %0, %1;" : "=f"(rhs) : "h"(right[index * cols + col]));
            accumulator = __fmaf_rn(
                left[row * depth + index],
                rhs,
                accumulator);
        }
    }
    __shared__ float chains[8][8];
    chains[chain][lane] = accumulator;
    __syncthreads();
    if (chain == 0 && row < rows && col < cols) {
        float sum = __fadd_rn(chains[0][lane], chains[1][lane]);
        sum = __fadd_rn(sum, chains[2][lane]);
        sum = __fadd_rn(sum, chains[3][lane]);
        sum = __fadd_rn(sum, chains[4][lane]);
        sum = __fadd_rn(sum, chains[5][lane]);
        sum = __fadd_rn(sum, chains[6][lane]);
        output[row * cols + col] = __fadd_rn(sum, chains[7][lane]);
    }
}

// A warp owns 32 adjacent output columns for one row and each of the eight
// warps owns one deterministic depth chain. This preserves the exact
// eight-chain FMA and reduction order above while turning the right-hand
// matrix reads for every depth into contiguous warp transactions.
extern "C" __global__ void matmul_eight_chain_tiled_kernel(
    const float* left,
    const float* right,
    float* output,
    unsigned int rows,
    unsigned int depth,
    unsigned int cols) {
    const unsigned int lane = threadIdx.x;
    const unsigned int chain = threadIdx.y;
    const unsigned int col = blockIdx.x * 32 + lane;
    const unsigned int row = blockIdx.y;
    float accumulator = 0.0f;
    if (row < rows && col < cols && chain < 8) {
        for (unsigned int index = chain; index < depth; index += 8) {
            accumulator = __fmaf_rn(
                left[row * depth + index],
                right[index * cols + col],
                accumulator);
        }
    }
    __shared__ float chains[8][32];
    chains[chain][lane] = accumulator;
    __syncthreads();
    if (chain == 0 && row < rows && col < cols) {
        float sum = __fadd_rn(chains[0][lane], chains[1][lane]);
        sum = __fadd_rn(sum, chains[2][lane]);
        sum = __fadd_rn(sum, chains[3][lane]);
        sum = __fadd_rn(sum, chains[4][lane]);
        sum = __fadd_rn(sum, chains[5][lane]);
        sum = __fadd_rn(sum, chains[6][lane]);
        output[row * cols + col] = __fadd_rn(sum, chains[7][lane]);
    }
}

extern "C" __global__ void matmul_kernel(
    const float* left,
    const float* right,
    float* output,
    unsigned int rows,
    unsigned int depth,
    unsigned int cols) {
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || col >= cols) {
        return;
    }
    float accumulator = 0.0f;
    for (unsigned int index = 0; index < depth; ++index) {
        accumulator = __fmaf_rn(
            left[row * depth + index],
            right[index * cols + col],
            accumulator);
    }
    output[row * cols + col] = accumulator;
}
