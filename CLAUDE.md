# CLAUDE.md

## Project Purpose

xeno-rt is a Rust LLM inference runtime designed as a high-performance,
memory-safe alternative to llama.cpp. The project is GGUF-first, optimized for
local and server-side inference, and structured so CPU and CUDA backends share
the same runtime and model-loading pipeline.

## Workspace Layout

The workspace currently spans 12 Rust packages:

1. `xrt-workspace-tests` - root package used as the workspace-level test and
   benchmark harness
2. `xrt-core` - core tensor, dtype, device, and error types
3. `xrt-gguf` - GGUF parsing, metadata decoding, and mmap-backed tensor access
4. `xrt-hub` - Hugging Face download and local cache management
5. `xrt-tokenizer` - tokenizer loading and encode/decode support
6. `xrt-kernels` - CPU kernels and quantization helpers
7. `xrt-cuda` - feature-gated CUDA backend integration
8. `xrt-models` - model definitions and forward passes
9. `xrt-runtime` - session management, sampling, scheduling, and KV cache logic
10. `xrt-cli` - local inference and model management CLI
11. `xrt-server` - OpenAI-compatible HTTP API server
12. `xtask` - developer automation and maintenance commands

## Standard Build Commands

- `cargo build`
- `cargo test --workspace`
- `cargo bench`

Use `cargo fmt --all` before submitting formatting-sensitive changes. Use
`cargo clippy --workspace --all-targets -- -D warnings` before large refactors
or API changes.

## Architecture Priorities

### GGUF-first

Treat GGUF as the primary model interchange format. Prefer preserving exact
metadata semantics and tensor-layout fidelity over introducing generic format
abstractions that weaken GGUF support.

### Memory-mapped weights

The runtime is designed around `mmap`-backed model loading. Avoid eager weight
copies or buffering layers unless a backend explicitly requires them for
correctness.

### Paged KV cache

KV cache management is page-based to control growth and reduce allocation churn.
Changes in `xrt-runtime` should preserve bounded cache growth behavior and avoid
turning append paths into full-buffer reallocations.

### Operator-level backend abstraction

Backends are selected at the operator layer, not through a large graph runtime.
Keep higher-level model code backend-agnostic and push device-specific logic
into kernels or backend adapters.

## Coding Conventions

- Use `f32` accumulation for numerically sensitive math unless there is a
  benchmarked and reviewed reason to do otherwise.
- Prefer `rayon` for CPU parallelism in kernel and batch-oriented paths instead
  of bespoke threading.
- Use `thiserror` for typed error definitions and propagation.
- Use `tracing` for structured logging rather than `println!` in library code.
- Favor explicit shapes, strides, and layout assumptions in hot paths; hidden
  allocations are usually a bug.
- Add regression tests when touching GGUF parsing, quantization, sampling, or KV
  cache behavior.
- Benchmark any change in `xrt-kernels`, `xrt-cuda`, `xrt-models`, or
  `xrt-runtime` that could affect throughput or memory use.
