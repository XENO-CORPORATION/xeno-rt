# Contributing to xeno-rt

We welcome contributions to xeno-rt! This document explains how to get started.

## Contributor License Agreement (CLA)

Before your first pull request can be merged, you must agree to our [Contributor License Agreement](CLA.md). This is a one-time requirement that protects both you and XENO Corporation.

**How it works:**
1. Open a pull request
2. The CLA bot will comment asking you to agree
3. Reply with a comment containing `I have read the CLA Document and I hereby sign the CLA`
4. The bot records your agreement — you won't be asked again for future PRs

**Why we require a CLA:** XENO Corporation uses xeno-rt in both open-source and commercial products. The CLA ensures we can continue to do so while protecting your rights as a contributor. This is the same approach used by Apache Foundation, Google, Microsoft, and Meta for their open-source projects.

## Development Setup

### Prerequisites

- Rust 1.76+ (install via [rustup](https://rustup.rs/))
- Git

### Building

```bash
git clone https://github.com/XENO-CORPORATION/xeno-rt.git
cd xeno-rt
cargo build
```

### Running Tests

```bash
# All tests
cargo test --workspace

# Specific test suite
cargo test --test kernels_test
cargo test --test gguf_parse_test

# Include long-running smoke tests
cargo test --workspace -- --include-ignored
```

### Running Benchmarks

```bash
cargo bench
```

### Code Quality

Before submitting a PR, ensure:

```bash
cargo fmt --check          # Code is formatted
cargo clippy --workspace   # No warnings
cargo test --workspace     # All tests pass
```

## How to Contribute

### Reporting Bugs

Open an issue on GitHub with:
- Steps to reproduce
- Expected vs actual behavior
- System info (OS, Rust version, GPU if applicable)

### Suggesting Features

Open an issue with the `enhancement` label. Describe:
- The problem you're trying to solve
- Your proposed solution
- Alternatives you've considered

### Submitting Code

1. Fork the repository
2. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/my-feature
   ```
3. Make your changes
4. Add or update tests for your changes
5. Ensure all checks pass (`fmt`, `clippy`, `test`)
6. Commit with a clear message:
   ```
   Add Q2_K dequantization kernel

   Implements the Q2_K quantization format matching ggml spec.
   Includes CPU dequantization and integration with the matmul path.
   ```
7. Push and open a pull request against `main`

### PR Guidelines

- Keep PRs focused — one feature or fix per PR
- Include tests for new functionality
- Update documentation if behavior changes
- Benchmark performance-sensitive changes (`cargo bench`)
- Reference related issues in the PR description

## Architecture Guidelines

### Where to put code

| What | Where |
|---|---|
| New data type or trait | `crates/xrt-core/` |
| GGUF format changes | `crates/xrt-gguf/` |
| New CPU kernel | `crates/xrt-kernels/src/cpu/` |
| New CUDA kernel | `crates/xrt-cuda/src/` |
| New model architecture | `crates/xrt-models/src/` |
| Runtime features (batching, caching) | `crates/xrt-runtime/` |
| CLI commands | `crates/xrt-cli/` |
| API endpoints | `crates/xrt-server/` |

### Code Style

- Use `cargo fmt` defaults (no custom rustfmt config)
- Prefer explicit types over inference in public APIs
- Use `thiserror` for error types
- Use `tracing` for logging (not `println!`)
- Keep `unsafe` to an absolute minimum — document safety invariants
- Use `f32` accumulation for quantized operations

## License

By contributing to xeno-rt, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE), subject to the terms of the [CLA](CLA.md).
