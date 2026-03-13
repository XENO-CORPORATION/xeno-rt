# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-03-13

### Added

- **xrt-core**: Core types — `DType` (F32, F16, BF16, Q8_0, Q4_0, Q4_K, Q5_K, Q6_K), `Device`, `TensorView`, error types
- **xrt-gguf**: Full GGUF binary parser with mmap, metadata extraction, tensor validation, and defensive parsing
- **xrt-tokenizer**: BPE tokenizer loaded from GGUF metadata with special token handling, encode/decode
- **xrt-kernels**: CPU compute kernels — RMSNorm, RoPE, softmax, SiLU, tiled matmul (64-wide, rayon parallel, 8-lane unroll)
- **xrt-kernels**: Quantized dequantization — Q8_0, Q4_0, Q4_K, Q5_K, Q6_K matching ggml spec
- **xrt-cuda**: CUDA GPU backend with 7 PTX kernels (matmul, rmsnorm, rope, softmax, silu, add, embed) via cudarc
- **xrt-models**: Llama-family forward pass with grouped query attention and KV cache support
- **xrt-runtime**: Paged KV cache, sampling strategies (temperature, top-k, top-p, repetition penalty), session management
- **xrt-hub**: HuggingFace model hub integration — download, cache, and verify GGUF models
- **xrt-server**: OpenAI-compatible HTTP server with `/v1/chat/completions` and `/v1/completions` endpoints, SSE streaming
- **xrt-cli**: Command-line interface with `generate` and `download` commands, HuggingFace auto-download
- **xtask**: Developer tooling for model management and cache operations
- Criterion benchmarks for kernels (RMSNorm, RoPE, softmax, SiLU, matmul, dequantization) and tokenizer
- 22 integration tests covering GGUF parsing, CPU kernels, KV cache, sampling, tokenizer, and end-to-end inference
- Apache-2.0 license with CLA for contributors
- CI pipeline (check, test, fmt, clippy, bench compilation)
- Automated CLA bot via GitHub Actions
- Security policy with vulnerability disclosure process
- Dependabot for automated dependency updates
- Release automation with cross-platform binary builds

[Unreleased]: https://github.com/XENO-CORPORATION/xeno-rt/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/XENO-CORPORATION/xeno-rt/releases/tag/v0.1.0
