#include <stdint.h>
#include <math.h>

extern "C" __global__ void awq_gemm4_matvec_kernel(
    const uint32_t* __restrict__ qweight,
    const uint32_t* __restrict__ qzeros,
    const float* __restrict__ scales,
    const float* __restrict__ input,
    float* __restrict__ output,
    uint32_t rows,
    uint32_t cols,
    uint32_t group_size) {
    __shared__ float reduction[256];

    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (row >= rows) {
        return;
    }

    const uint32_t packed_rows = rows >> 3;
    const uint32_t packed_row = row >> 3;
    const uint32_t lane = row & 7u;
    // AutoAWQ GEMM stores row lanes in [0, 2, 4, 6, 1, 3, 5, 7] order.
    const uint32_t packed_lane = (lane >> 1) + ((lane & 1u) << 2);
    const uint32_t shift = packed_lane << 2;
    float sum = 0.0f;

    for (uint32_t col = tid; col < cols; col += blockDim.x) {
        const uint32_t group = col / group_size;
        const uint32_t weight_word = qweight[col * packed_rows + packed_row];
        const uint32_t zero_word = qzeros[group * packed_rows + packed_row];
        const int quant = static_cast<int>((weight_word >> shift) & 0x0fu);
        const int zero = static_cast<int>((zero_word >> shift) & 0x0fu);
        const float weight = static_cast<float>(quant - zero) * scales[group * rows + row];
        sum = fmaf(input[col], weight, sum);
    }

    reduction[tid] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduction[tid] += reduction[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[row] = reduction[0];
    }
}
