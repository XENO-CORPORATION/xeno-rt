# xeno-rt — AI Assistant Context

## Project Overview

xeno-rt (xrt) is a high-performance LLM inference runtime written in Rust, designed as an alternative to llama.cpp. Built by XENO Corporation.

## Build Commands

```bash
cargo build                          # Build all crates
cargo build --release                # Release build
cargo build --features cuda          # Build with CUDA support
cargo test --workspace               # Run all tests
cargo test --workspace -- --include-ignored  # Include E2E smoke tests
cargo bench                          # Run benchmarks
cargo fmt --check                    # Check formatting
cargo clippy --workspace -- -D warnings     # Lint
```

## Workspace Structure (12 crates)

| Crate | Path | Purpose |
|---|---|---|
| xrt-core | crates/xrt-core/ | DType, Device, TensorView, error types |
| xrt-gguf | crates/xrt-gguf/ | GGUF binary parser, mmap, metadata, validation |
| xrt-tokenizer | crates/xrt-tokenizer/ | BPE tokenizer from GGUF metadata |
| xrt-kernels | crates/xrt-kernels/ | CPU kernels (RMSNorm, RoPE, softmax, SiLU, matmul, dequant) |
| xrt-cuda | crates/xrt-cuda/ | CUDA GPU backend, PTX kernels via cudarc |
| xrt-models | crates/xrt-models/ | Model forward pass (Llama with GQA) |
| xrt-runtime | crates/xrt-runtime/ | Paged KV cache, sampling, session management |
| xrt-hub | crates/xrt-hub/ | HuggingFace model downloads and caching |
| xrt-cli | crates/xrt-cli/ | CLI (generate, download commands) |
| xrt-server | crates/xrt-server/ | OpenAI-compatible API server (axum) |
| xtask | xtask/ | Developer tooling |

## Key Architecture Decisions

- **GGUF-first**: GGUF is the primary model format. Parser is in xrt-gguf with defensive validation.
- **mmap for weights**: Uses memmap2. Never eagerly copy model tensors.
- **Operator-level backend abstraction**: NOT a dynamic Tensor interface. Models call typed ops from xrt-kernels.
- **Paged KV cache from day one**: Block-level paging in xrt-runtime/src/kv_cache.rs.
- **f32 accumulation**: Always use f32 for numerically sensitive ops, even with f16/bf16 storage.
- **Quantized dequant into kernel-local tiles**: Not full tensors.

## Coding Conventions

- Use `thiserror` for error types, `tracing` for logging
- Use `rayon` for CPU parallelism in kernels
- CUDA code is feature-gated behind `--features cuda`
- Tests use synthetic GGUF fixtures built in tests/common/mod.rs
- Benchmarks use criterion in benches/

## Supported Quantizations

F32, F16, BF16, Q8_0, Q4_0, Q4_K, Q5_K, Q6_K

## License

Apache-2.0 with CLA required for contributions.
