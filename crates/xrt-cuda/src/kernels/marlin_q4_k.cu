// XRT instantiation of the Apache-2.0 vLLM Marlin W4A16 kernel.
// Build this translation unit with vLLM's csrc and Marlin include directories.
#define MARLIN_NAMESPACE_NAME xrt_marlin
#include "kernel.h"
#include "marlin_template.h"

namespace xrt_marlin {
template __global__ void Marlin<
    vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(),
    vllm::kFloat16.id(), 256, 1, 8, 8, false, 3, 2, true>(
    MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<
    vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(),
    vllm::kFloat16.id(), 256, 1, 8, 8, true, 3, 2, true>(
    MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<
    vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(),
    vllm::kFloat16.id(), 128, 1, 4, 8, false, 3, 2, true>(
    MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<
    vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(),
    vllm::kFloat16.id(), 128, 1, 4, 8, true, 3, 2, true>(
    MARLIN_KERNEL_PARAMS);

// GGUF Q8_0 uses signed int8 values with one F16 scale per 32 weights.
// Re-biasing the bytes by 128 lets Marlin's U8 path represent the exact same
// quantized values with group_blocks=2 and a floating zero point of 128*d.
template __global__ void Marlin<
    vllm::kFloat16.id(), vllm::kU8.id(), vllm::kFloat16.id(),
    vllm::kFloat16.id(), 256, 1, 8, 8, false, 3, 2, true>(
    MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<
    vllm::kFloat16.id(), vllm::kU8.id(), vllm::kFloat16.id(),
    vllm::kFloat16.id(), 256, 1, 8, 8, true, 3, 2, true>(
    MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<
    vllm::kFloat16.id(), vllm::kU8.id(), vllm::kFloat16.id(),
    vllm::kFloat16.id(), 128, 1, 4, 8, false, 3, 2, true>(
    MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<
    vllm::kFloat16.id(), vllm::kU8.id(), vllm::kFloat16.id(),
    vllm::kFloat16.id(), 128, 1, 4, 8, true, 3, 2, true>(
    MARLIN_KERNEL_PARAMS);
// GGUF Q6_K stores unsigned six-bit values with one signed F16-scaled
// multiplier per 16 weights. Keeping the values biased by 32 maps its
// `(q - 32) * scale` equation onto Marlin's U8 plus floating-zero-point path.
template __global__ void Marlin<
    vllm::kFloat16.id(), vllm::kU8.id(), vllm::kFloat16.id(),
    vllm::kFloat16.id(), 256, 1, 8, 8, false, 3, 1, true>(
    MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<
    vllm::kFloat16.id(), vllm::kU8.id(), vllm::kFloat16.id(),
    vllm::kFloat16.id(), 256, 1, 8, 8, true, 3, 1, true>(
    MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<
    vllm::kFloat16.id(), vllm::kU8.id(), vllm::kFloat16.id(),
    vllm::kFloat16.id(), 128, 1, 4, 8, false, 3, 1, true>(
    MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<
    vllm::kFloat16.id(), vllm::kU8.id(), vllm::kFloat16.id(),
    vllm::kFloat16.id(), 128, 1, 4, 8, true, 3, 1, true>(
    MARLIN_KERNEL_PARAMS);
}
