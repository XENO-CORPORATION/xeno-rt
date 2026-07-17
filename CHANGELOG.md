# Changelog

All notable changes to xeno-rt are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- Harden repository documentation, portable build defaults, hosted validation,
  release evidence, and supply-chain policy for the v0.2.0 checkpoint.

## [0.2.0] - Unreleased

### Added

- Native CUDA model execution for compatible GGUF models with resident weights,
  KV state, scratch buffers, quantized matvec kernels, and explicit GPU memory
  preflight.
- CUDA GGUF weight support for F32, F16, BF16, Q8_0, Q4_0, Q4_K, Q5_K, and
  Q6_K standard dense paths, plus the Gemma4-specific execution path.
- CUDA-only Hugging Face SafeTensors loading for Qwen2/Qwen3 standard dense,
  AutoAWQ GEMM/GEMV, GPTQ v1/v2, and compressed-tensors W4A16 layouts.
- GPU-resident F32, Q8, KQ4/VQ8, and policy-adaptive KV cache modes with paged
  storage and bounded resource accounting.
- CUDA Graph capture/replay for supported decode shapes, execution streams,
  memory-pool telemetry, transfer counters, allocation counters, and guarded
  GPU status reporting.
- Prefix caching, prompt-span cache policies, chunked prefill, bounded request
  scheduling, and decode batching.
- CPU/CUDA/external-backend benchmark comparison with structured JSON output.
- External OpenAI-compatible backend with loopback-only default targeting.
- SafeTensors, OpenAI-client, vision, C API, and Python workspace crates.
- Qwen3, Qwen3.5/Qwen3-Next family, and Gemma4 model-geometry support with
  explicit unsupported multimodal-family errors.
- mmproj-style vision loading and an experimental ONNX background-removal HTTP
  route.
- `/v1/models`, runtime load/unload/status routes, usage accounting, Hugging
  Face model download, and chat tool fields.

### Changed

- Optimized CPU execution with pre-resolved tensor metadata, fused projections,
  zero-allocation decode scratch, online attention, specialized AVX2 K-quant
  kernels, a spin-based dispatch pool, and prompt-lookup speculative decoding.
- Made default builds CPU-portable; machine-specific `target-cpu=native` is now
  an explicit local benchmark opt-in.
- Expanded runtime status with active backend, cache, scheduler, GPU allocation,
  memory-pool, graph, and transfer observations.
- Standardized the binary workspace on a committed lockfile and locked hosted
  build/test/release commands.

### Security

- Added explicit CUDA upload budgets and dimension/layout validation before
  kernel execution.
- Restricted external OpenAI-compatible targets to loopback unless remote use
  is explicitly enabled.
- Upgraded the experimental Python binding to PyO3 0.29, resolving
  RUSTSEC-2026-0176 and RUSTSEC-2026-0177; the binding now requires Rust 1.83
  or newer while the core runtime retains its Rust 1.76 MSRV.
- Added hardened release provenance, checksums, SBOM requirements, dependency
  policy, and immutable GitHub Actions references.

### Known Limitations

- SafeTensors decode is CUDA-only.
- CUDA support is limited to the model/source combinations in
  `docs/SUPPORTED_MODELS.md`.
- The HTTP server does not provide built-in inbound authentication or TLS.
- C and Python bindings and image-task routes remain experimental and are not
  included as supported v0.2.0 binary packages.
- The XENO product catalog currently expects a desktop installer, while this
  repository produces CLI/server archives. R2/Hub publication remains blocked
  until that delivery contract is resolved.

## [0.1.0] - 2026-03-13

### Added

- Initial Rust workspace with core tensor types, GGUF parsing, tokenization,
  CPU kernels, model execution, runtime orchestration, CLI, server, model hub,
  developer tooling, integration tests, and benchmarks.
- GGUF memory-mapped tensor access and metadata validation.
- F32, F16, BF16, Q8_0, Q4_0, Q4_K, Q5_K, and Q6_K CPU weight execution.
- Llama-family grouped-query attention, paged KV cache, sampling, command-line
  generation, Hugging Face GGUF download, and OpenAI-compatible completion/chat
  endpoints.

[Unreleased]: https://github.com/XENO-CORPORATION/xeno-rt/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/XENO-CORPORATION/xeno-rt/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/XENO-CORPORATION/xeno-rt/releases/tag/v0.1.0
