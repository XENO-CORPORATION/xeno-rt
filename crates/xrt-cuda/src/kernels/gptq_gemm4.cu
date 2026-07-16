#include <stdint.h>
#include <math.h>

extern "C" __global__ void gptq_gemm4_matvec_kernel(
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
    const uint32_t zero_shift = (row & 7u) << 2;
    float sum = 0.0f;

    for (uint32_t col = tid; col < cols; col += blockDim.x) {
        const uint32_t group = col / group_size;
        const uint32_t packed_col = col >> 3;
        const uint32_t weight_shift = (col & 7u) << 2;
        const uint32_t weight_word = qweight[packed_col * rows + row];
        const uint32_t zero_word = qzeros[group * packed_rows + packed_row];
        const int quant = static_cast<int>((weight_word >> weight_shift) & 0x0fu);
        // AutoGPTQ stores each zero point minus one and restores it modulo 16.
        const int zero = static_cast<int>((((zero_word >> zero_shift) & 0x0fu) + 1u) & 0x0fu);
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
