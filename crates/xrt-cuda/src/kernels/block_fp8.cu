__device__ float e4m3fn(unsigned char b) {
    unsigned int e = (b & 127) >> 3, m = b & 7;
    float v = e == 0 ? m / 512.0f : (8 + m) * (1u << e) / 1024.0f;
    return b & 128 ? -v : v;
}
__device__ float native_bf16(float value) {
    unsigned int bits = __float_as_uint(value);
    if ((bits & 0x7f800000u) == 0x7f800000u) return value;
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return __uint_as_float(bits & 0xffff0000u);
}
__device__ float native_fp8_rne(float value) {
    if (!isfinite(value)) return __uint_as_float(0x7fc00000u);
    float magnitude = fabsf(value);
    unsigned int low = 0, high = 126;
    while (low < high) {
        unsigned int middle = (low + high) / 2;
        if (e4m3fn((unsigned char)middle) < magnitude) low = middle + 1;
        else high = middle;
    }
    unsigned int chosen = low;
    if (low > 0) {
        float previous = e4m3fn((unsigned char)(low - 1));
        float next = e4m3fn((unsigned char)low);
        float below = magnitude - previous, above = next - magnitude;
        if (below < above || (below == above && ((low - 1) & 1u) == 0)) chosen = low - 1;
    }
    float quantized = e4m3fn((unsigned char)chosen);
    return __float_as_uint(value) & 0x80000000u ? -quantized : quantized;
}
// One block per 128-element group. Quantize in-place only after every lane has
// read its original value. Each vector has independently owned group scales.
extern "C" __global__ void fp8_w8a8_quantize_groups(float* x, float* scales,
    unsigned int width, unsigned int vectors, unsigned int round_input) {
    unsigned int group = blockIdx.x, vector = blockIdx.y, lane = threadIdx.x;
    if (vector >= vectors) return;
    unsigned int groups = width / 128 + (width % 128 != 0);
    unsigned long long column = (unsigned long long)group * 128 + lane;
    float value = column < width ? x[(unsigned long long)vector * width + column] : 0.0f;
    if (round_input) value = native_bf16(value);
    __shared__ float maximum[128];
    maximum[lane] = fabsf(value); __syncthreads();
    for (unsigned int stride = 64; stride; stride /= 2) {
        if (lane < stride) maximum[lane] = fmaxf(maximum[lane], maximum[lane + stride]);
        __syncthreads();
    }
    float scale = fmaxf(maximum[0], 1e-10f) / 448.0f;
    if (lane == 0) scales[vector * groups + group] = scale;
    if (column < width) {
        float normalized = value / scale;
        x[(unsigned long long)vector * width + column] = isfinite(normalized)
            ? native_fp8_rne(fminf(448.0f, fmaxf(-448.0f, normalized)))
            : __uint_as_float(0x7fc00000u);
    }
}
extern "C" __global__ void fp8_moe_w8a8_gate_up(const unsigned long long* p,
    const float* x, const float* activation_scale, float* h,
    unsigned int experts, unsigned int hidden, unsigned int middle) {
    unsigned int row = blockIdx.x, expert = blockIdx.y, lane = threadIdx.x;
    if (row >= middle || expert >= experts) return;
    const unsigned char* gate = (const unsigned char*)p[expert];
    const unsigned char* up = (const unsigned char*)p[experts + expert];
    const float* gate_scale = (const float*)p[3 * experts + expert];
    const float* up_scale = (const float*)p[4 * experts + expert];
    unsigned int groups = hidden / 128 + (hidden % 128 != 0);
    __shared__ float ga[128], ua[128];
    float gate_total = 0.0f, up_total = 0.0f;
    for (unsigned int group = 0; group < groups; ++group) {
        unsigned long long c = (unsigned long long)group * 128 + lane;
        unsigned long long offset = (unsigned long long)row * hidden + c;
        ga[lane] = c < hidden ? e4m3fn(gate[offset]) * x[c] : 0.0f;
        ua[lane] = c < hidden ? e4m3fn(up[offset]) * x[c] : 0.0f;
        __syncthreads();
        for (unsigned int stride = 64; stride; stride /= 2) {
            if (lane < stride) { ga[lane] += ga[lane + stride]; ua[lane] += ua[lane + stride]; }
            __syncthreads();
        }
        if (lane == 0) {
            unsigned int scale = (row / 128) * groups + group;
            gate_total += (ga[0] * activation_scale[group]) * gate_scale[scale];
            up_total += (ua[0] * activation_scale[group]) * up_scale[scale];
        }
        __syncthreads();
    }
    if (lane == 0) {
        float g = native_bf16(gate_total), u = native_bf16(up_total);
        h[(unsigned long long)expert * middle + row] = (g / (1.0f + expf(-g))) * u;
    }
}
extern "C" __global__ void fp8_moe_w8a8_down_sum(const unsigned long long* p,
    const float* h, const float* activation_scale, const float* probability,
    float* y, unsigned int experts, unsigned int hidden, unsigned int middle, float denominator) {
    unsigned int row = blockIdx.x, lane = threadIdx.x;
    if (row >= hidden) return;
    unsigned int groups = middle / 128 + (middle % 128 != 0);
    __shared__ float partial[128];
    float total = 0.0f;
    for (unsigned int expert = 0; expert < experts; ++expert) {
        const unsigned char* down = (const unsigned char*)p[2 * experts + expert];
        const float* scale = (const float*)p[5 * experts + expert];
        float projected = 0.0f;
        for (unsigned int group = 0; group < groups; ++group) {
            unsigned long long c = (unsigned long long)group * 128 + lane;
            partial[lane] = c < middle ? e4m3fn(down[(unsigned long long)row * middle + c]) * h[(unsigned long long)expert * middle + c] : 0.0f;
            __syncthreads();
            for (unsigned int stride = 64; stride; stride /= 2) {
                if (lane < stride) partial[lane] += partial[lane + stride];
                __syncthreads();
            }
            if (lane == 0) projected += (partial[0] * activation_scale[expert * groups + group]) * scale[(row / 128) * groups + group];
            __syncthreads();
        }
        if (lane == 0) total += projected * native_bf16(probability[expert] / denominator);
        __syncthreads();
    }
    if (lane == 0) y[row] = total;
}
extern "C" __global__ void fp8_moe_gate_up(const unsigned long long* p, const float* x,
    float* h, unsigned int experts, unsigned int hidden, unsigned int middle) {
    unsigned int row = blockIdx.x, expert = blockIdx.y, lane = threadIdx.x;
    if (row >= middle || expert >= experts) return;
    const unsigned char* gate = (const unsigned char*)p[expert];
    const unsigned char* up = (const unsigned char*)p[experts + expert];
    const float* gate_scale = (const float*)p[3 * experts + expert];
    const float* up_scale = (const float*)p[4 * experts + expert];
    unsigned int scale_cols = (hidden + 127) / 128;
    float a = 0.0f, b = 0.0f;
    for (unsigned int c = lane; c < hidden; c += 128) {
        unsigned int scale = (row / 128) * scale_cols + c / 128;
        unsigned long long offset = (unsigned long long)row * hidden + c;
        a += (e4m3fn(gate[offset]) * gate_scale[scale]) * x[c];
        b += (e4m3fn(up[offset]) * up_scale[scale]) * x[c];
    }
    __shared__ float ga[128], ua[128];
    ga[lane] = a; ua[lane] = b; __syncthreads();
    for (unsigned int stride = 64; stride; stride /= 2) {
        if (lane < stride) { ga[lane] += ga[lane + stride]; ua[lane] += ua[lane + stride]; }
        __syncthreads();
    }
    if (lane == 0) {
        float exponential;
        float exponent = -ga[0] * 1.4426950408889634f;
        asm("ex2.approx.f32 %0, %1;" : "=f"(exponential) : "f"(exponent));
        h[expert * middle + row] = (ga[0] / (1.0f + exponential)) * ua[0];
    }
}
extern "C" __global__ void fp8_moe_down_sum(const unsigned long long* p, const float* h,
    const float* probability, float* y, unsigned int experts, unsigned int hidden,
    unsigned int middle, float denominator) {
    unsigned int row = blockIdx.x, lane = threadIdx.x;
    if (row >= hidden) return;
    unsigned int scale_cols = (middle + 127) / 128;
    __shared__ float partial[128];
    float total = 0.0f;
    for (unsigned int expert = 0; expert < experts; ++expert) {
        const unsigned char* down = (const unsigned char*)p[2 * experts + expert];
        const float* scale = (const float*)p[5 * experts + expert];
        float sum = 0.0f;
        for (unsigned int c = lane; c < middle; c += 128)
            sum += (e4m3fn(down[(unsigned long long)row * middle + c]) * scale[(row / 128) * scale_cols + c / 128]) * h[expert * middle + c];
        partial[lane] = sum; __syncthreads();
        for (unsigned int stride = 64; stride; stride /= 2) {
            if (lane < stride) partial[lane] += partial[lane + stride];
            __syncthreads();
        }
        if (lane == 0) total += (partial[0] * probability[expert]) / denominator;
        __syncthreads();
    }
    if (lane == 0) y[row] = total;
}
extern "C" __global__ void dense_f32_matvec(const float* w, const float* x, float* y,
    unsigned int rows, unsigned int cols) {
    unsigned int row = blockIdx.x, lane = threadIdx.x;
    if (row >= rows) return;
    float sum = 0.0f;
    for (unsigned int c = lane; c < cols; c += 128)
        sum += w[(unsigned long long)row * cols + c] * x[c];
    __shared__ float partial[128];
    partial[lane] = sum; __syncthreads();
    for (unsigned int stride = 64; stride; stride /= 2) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    if (lane == 0) y[row] = partial[0];
}
extern "C" __global__ void dense_bf16_matvec(const unsigned short* w, const float* x, float* y,
    unsigned int rows, unsigned int cols) {
    unsigned int row = blockIdx.x, lane = threadIdx.x;
    if (row >= rows) return;
    float sum = 0.0f;
    for (unsigned int c = lane; c < cols; c += 128) {
        float weight = __uint_as_float((unsigned int)w[(unsigned long long)row * cols + c] << 16);
        sum += weight * x[c];
    }
    __shared__ float partial[128];
    partial[lane] = sum; __syncthreads();
    for (unsigned int stride = 64; stride; stride /= 2) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    if (lane == 0) y[row] = partial[0];
}
extern "C" __global__ void block_fp8_matvec(const unsigned char* w, const float* s,
    const float* x, const float* a, float* y, unsigned int rows, unsigned int cols,
    unsigned int br, unsigned int bc, unsigned int quantized_activation) {
    unsigned int row = blockIdx.x, lane = threadIdx.x;
    if (row >= rows) return;
    unsigned int scale_cols = cols / bc + (cols % bc != 0);
    __shared__ float partial[128];
    if (quantized_activation) {
        // Preserve the W8A8 arithmetic boundary: dot the encoded values for
        // one K-group, then apply its activation and weight scales once.
        float total = 0.0f;
        for (unsigned int group = 0; group < scale_cols; ++group) {
            float dot = 0.0f;
            unsigned long long begin = (unsigned long long)group * bc;
            unsigned long long end = ((unsigned long long)group + 1) * bc;
            if (end > cols) end = cols;
            for (unsigned long long c = begin + lane; c < end; c += 128)
                dot += e4m3fn(w[(unsigned long long)row * cols + c]) * x[c];
            partial[lane] = dot; __syncthreads();
            for (unsigned int stride = 64; stride; stride /= 2) {
                if (lane < stride) partial[lane] += partial[lane + stride];
                __syncthreads();
            }
            if (lane == 0) total += (partial[0] * a[group]) * s[(row / br) * scale_cols + group];
            __syncthreads();
        }
        if (lane == 0) y[row] = total;
        return;
    }
    float sum = 0.0f;
    for (unsigned long long c = lane; c < cols; c += blockDim.x) {
        float scale = s[(row / br) * scale_cols + c / bc];
        float value = e4m3fn(w[(unsigned long long)row * cols + c]);
        sum += (value * scale) * x[c];
    }
    partial[lane] = sum;
    __syncthreads();
    for (unsigned int stride = 64; stride; stride /= 2) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    if (lane == 0) y[row] = partial[0];
}
