# Qwen3.6 Marlin W4A8 rejection screen (RTX 4090, 2026-08-11)

This record preserves the result of an SM89-only Marlin W4A8 experiment against
the frozen `qwen36-greedy-admission-v1` corpus. The candidate quantized F32/F16
activations to FP8 E4M3, used a W4A8 tensor-core Marlin kernel, and applied the
GGUF Q4_K affine residual after MMA.

The physical CUDA oracle passed at 2.5085% relative L2 error, but the real-model
admission screen failed both required gates:

- mean decode throughput fell from 127.3386 to 118.3621 tok/s (-7.05%);
- 3 of 12 frozen cases changed token IDs (`counting_csv`, `rust_code`, and
  `strict_json`);
- verifier time rose from 4559.146 to 4937.889 ms in aggregate;
- draft time rose from 1446.073 to 1573.395 ms in aggregate; and
- rebase time rose from 184.490 to 195.282 ms in aggregate.

The candidate is rejected and was removed from the runtime. This is negative
benchmark evidence, not an advertised backend or quantization capability.

See `aggregate.json` for the machine-readable comparison.
