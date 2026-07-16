# Development Guide

## Toolchain

The workspace declares Rust 1.76 as its minimum supported version and tracks
stable through `rust-toolchain.toml`. Hosted CI verifies the actual contract.
Use rustup with `rustfmt` and `clippy` installed.

## Checkout

```bash
git clone https://github.com/XENO-CORPORATION/xeno-rt.git
cd xeno-rt
cargo fetch --locked
```

## Standard Gates

```bash
cargo fmt --all -- --check
cargo check --workspace --all-targets --locked
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked
cargo bench --workspace --no-run --locked
```

CUDA compile coverage:

```bash
cargo check --workspace --all-targets --features cuda --locked
```

CUDA execution tests require a controlled NVIDIA runner. Do not run ignored
real-model tests on a shared workstation merely to satisfy a pull request.

## Change Boundaries

| Change | Primary location |
|---|---|
| Shared dtype/error/cache contract | `crates/xrt-core` |
| GGUF parser/validation | `crates/xrt-gguf` |
| Hugging Face/SafeTensors loading | `crates/xrt-safetensors` |
| Tokenizer/chat template | `crates/xrt-tokenizer` |
| CPU kernel | `crates/xrt-kernels` |
| CUDA primitive/storage/graph | `crates/xrt-cuda` |
| Architecture/model execution | `crates/xrt-models` |
| Backend/session/cache/scheduler | `crates/xrt-runtime` |
| OpenAI-compatible fields/routes | `crates/xrt-server` |

Keep changes scoped. Do not combine kernel optimization, public API changes,
model-format changes, and broad source movement in one pull request.

## Required Compatibility Review

Before changing runtime behavior, answer all four questions in the PR:

1. Does the documented OpenAI-compatible request/response surface change?
2. Does any supported GGUF model or tensor encoding stop working?
3. Does the CPU-only build and fallback still work?
4. What benchmark and correctness evidence covers the change?

## Tests

- Unit tests live beside narrow implementation units.
- Root `tests/` contains cross-crate and synthetic-model integration coverage.
- Real-model tests are ignored and consume paths from `XRT_REAL_*` variables.
- New model formats require malformed-input rejection tests in addition to a
  successful fixture.
- CUDA kernels require scalar/CPU parity and a guarded device test.

## Dependencies and Lockfile

This is an application workspace, so `Cargo.lock` is committed. After a
deliberate dependency change:

```bash
cargo update -p dependency-name --precise VERSION
cargo check --workspace --all-targets --locked
```

Review the lockfile diff, licenses, advisories, and transitive source changes.
Internal crates are marked `publish = false`; crates.io publishing requires a
separate policy and package audit.

## Commit and Pull Request Policy

- Branch from current `main`.
- Use focused, imperative commit subjects such as `fix: validate GGUF tensor
  bounds` or `perf(cuda): reuse decode scratch`.
- Do not commit model files, secrets, build output, benchmark output, or local
  caches.
- Update docs and changelog for user-visible behavior.
- Use the pull request template and sign the CLA.
- Repository integration prefers squash merge; public history is not rewritten
  after merge.

## Releases

Read every file in `release-guide/` in order before release work. The portable
XENO publisher runs from `xeno-platform`, while this repository owns its
GitHub source/archive release workflow. Always run a dry-run and obtain explicit
human approval before a tag, upload, or deployment.
