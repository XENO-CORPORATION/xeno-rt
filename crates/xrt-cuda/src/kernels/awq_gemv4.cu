#include <stdint.h>
#include <math.h>

extern "C" __global__ void awq_gemv4_matvec_kernel(
    const uint32_t* __restrict__ qweight,
    const uint32_t* __restrict__ qzeros,
    const float* __restrict__ scales,
    const float* __restrict__ input,
    float* __restrict__ output,
    uint32_t rows,
    uint32_t cols,
    uint32_t group_size,
    uint32_t zero_words_per_row,
    uint32_t scale_stride) {
    __shared__ float reduction[256];

    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (row >= rows) {
        return;
    }

    const uint32_t packed_cols = cols >> 3;
    float sum = 0.0f;

    for (uint32_t col = tid; col < cols; col += blockDim.x) {
        const uint32_t group = col / group_size;
        const uint32_t weight_word = qweight[row * packed_cols + (col >> 3)];
        const uint32_t zero_word =
            qzeros[row * zero_words_per_row + (group >> 3)];
        const uint32_t quant_shift = (col & 7u) << 2;
        const uint32_t zero_shift = (group & 7u) << 2;
        const int quant = static_cast<int>((weight_word >> quant_shift) & 0x0fu);
        const int zero = static_cast<int>((zero_word >> zero_shift) & 0x0fu);
        const float scale = scales[row * scale_stride + group];
        sum = fmaf(input[col], static_cast<float>(quant - zero) * scale, sum);
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
