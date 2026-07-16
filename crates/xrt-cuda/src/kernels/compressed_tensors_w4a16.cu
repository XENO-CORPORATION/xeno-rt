#include <stdint.h>
#include <math.h>

extern "C" __global__ void compressed_tensors_w4a16_matvec_kernel(
    const uint32_t* __restrict__ weight_packed,
    const float* __restrict__ scales,
    const int32_t* __restrict__ group_indices,
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

    const uint32_t packed_cols = cols >> 3;
    const uint32_t groups = cols / group_size;
    float sum = 0.0f;

    for (uint32_t col = tid; col < cols; col += blockDim.x) {
        const uint32_t packed_col = col >> 3;
        const uint32_t shift = (col & 7u) << 2;
        const uint32_t word = weight_packed[row * packed_cols + packed_col];
        const int quant = static_cast<int>((word >> shift) & 0x0fu) - 8;
        const uint32_t group = static_cast<uint32_t>(group_indices[col]);
        const float scale = scales[row * groups + group];
        sum = fmaf(input[col], static_cast<float>(quant) * scale, sum);
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
