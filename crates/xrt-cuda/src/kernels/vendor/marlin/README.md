# Vendored Marlin CUDA headers

These Apache-2.0 headers are derived from the official `vllm-project/vllm`
Marlin implementation at commit
`d6941300fcb9d4a8bbea19f8b610c2aff9fc5cc3`. See `LICENSE.vllm`.

XRT's only semantic change is documented beside the affected code in
`marlin_template.h`: the float zero-point specialization consumes Q4_K's
already-scaled affine minimum directly. This avoids a lossy divide/multiply
round trip while preserving GGUF Q4_K's `q * scale - minimum` equation.

The checked-in Q4_K wrapper uses three pipeline stages for both admitted
small-batch tile shapes. That requires 27 KiB of dynamic shared memory for the
64-column tile and 42 KiB for the 128-column tile; the Rust loader opts every
Marlin specialization into the 42 KiB maximum once when it loads the module.

Rebuild the checked-in PTX on a CUDA 12.4 host from the repository root:

```sh
nvcc --ptx --std=c++17 --expt-relaxed-constexpr \
  --gpu-architecture=compute_80 --use_fast_math -O3 \
  -Icrates/xrt-cuda/src/kernels/vendor/marlin \
  crates/xrt-cuda/src/kernels/marlin_q4_k.cu \
  -o crates/xrt-cuda/src/kernels/generated/marlin_q4_k.ptx
```
