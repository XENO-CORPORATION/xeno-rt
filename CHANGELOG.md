# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-03-13

### Added

- Initial 12-package Rust workspace covering core tensor types, GGUF parsing,
  tokenizer support, CPU kernels, CUDA integration, model execution, runtime
  orchestration, CLI, server, hub integration, developer tooling, and the root
  workspace test harness.
- GGUF-first model loading with memory-mapped tensor access, metadata parsing,
  and validation for zero-copy startup paths.
- Tokenizer support loaded from GGUF metadata with BPE encode/decode and
  special-token handling.
- CPU kernels for RMSNorm, rotary position embeddings, softmax, SiLU, tiled
  matrix multiplication, and quantized dequantization routines.
- Quantization support for F32, F16, BF16, Q8_0, Q4_0, Q4_K, Q5_K, and Q6_K
  tensor formats.
- Feature-gated CUDA backend scaffolding with cudarc integration and PTX kernel
  coverage for core inference operators.
- Llama-family inference support with grouped-query attention, paged KV cache,
  configurable sampling, and session management.
- Hugging Face model download and cache management through `xrt-hub`, `xrt-cli`,
  and `xtask`.
- Command-line generation and model download flows through the `xrt-cli` binary.
- OpenAI-compatible HTTP serving through `xrt-server`, including completion,
  chat completion, and streaming response endpoints.
- Integration tests and Criterion benchmarks covering GGUF parsing, kernels,
  tokenizer behavior, sampler behavior, KV cache logic, and end-to-end smoke
  paths.
