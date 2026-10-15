__device__ float e4m3fn(unsigned char b) {
    unsigned int e = (b & 127) >> 3, m = b & 7;
    float v = e == 0 ? m / 512.0f : (8 + m) * (1u << e) / 1024.0f;
    return b & 128 ? -v : v;
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
    unsigned int scale_cols = (cols + bc - 1) / bc;
    float sum = 0.0f;
    for (unsigned int c = lane; c < cols; c += blockDim.x) {
        float scale = s[(row / br) * scale_cols + c / bc];
        float value = e4m3fn(w[(unsigned long long)row * cols + c]);
        sum += quantized_activation ? (value * x[c]) * scale * a[c / bc] : (value * scale) * x[c];
    }
    __shared__ float partial[128];
    partial[lane] = sum;
    __syncthreads();
    for (unsigned int stride = 64; stride; stride /= 2) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    if (lane == 0) y[row] = partial[0];
}
