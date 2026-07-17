# Contributing to xeno-rt

Contributions are welcome when they preserve the runtime's compatibility,
correctness, and performance contracts.

## Before You Start

- Search existing issues and pull requests.
- Discuss broad architecture, new backends, new model formats, and breaking API
  changes before implementation.
- Read [Architecture](docs/ARCHITECTURE.md),
  [Development](docs/DEVELOPMENT.md), and the relevant support matrix.
- Never publish a suspected vulnerability; follow [Security](SECURITY.md).

## Contributor License Agreement

Before a first pull request can merge, sign [CLA.md](CLA.md) through the CLA bot
using the exact confirmation it requests. Contributions remain yours and are
licensed under Apache-2.0 subject to the CLA.

## Development Setup

```bash
git clone https://github.com/XENO-CORPORATION/xeno-rt.git
cd xeno-rt
cargo fetch --locked
```

The repository uses `rust-toolchain.toml` for stable tooling and declares an
MSRV in `Cargo.toml`. Hosted CI is authoritative when toolchain behavior differs
locally.

## Required Checks

```bash
cargo fmt --all -- --check
cargo check --workspace --all-targets --locked
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked
cargo bench --workspace --no-run --locked
```

CUDA-affecting changes also require CUDA-feature compilation and a guarded
self-hosted validation plan. Real-model GPU tests must not run on untrusted pull
request code or an already-busy shared GPU.

## Pull Request Requirements

- Keep one coherent change per PR.
- Add success, rejection, and regression tests proportional to risk.
- Update docs and changelog for user-visible changes.
- Preserve documented HTTP request/response compatibility.
- Preserve GGUF support and the CPU-only build/fallback.
- Include before/after benchmark evidence for performance-sensitive code.
- Document `unsafe` invariants and CUDA bounds/resource assumptions.
- Do not add model binaries, secrets, generated build output, or local paths.
- Complete the PR template and link the issue/spec where applicable.

## Performance Evidence

Use [Benchmarking](docs/BENCHMARKING.md). A result without model identity,
hardware, command, build flags, commit, seed, and raw measurements is not
reviewable evidence. Correctness parity must pass before throughput is compared.

## Code Style

- Use standard `rustfmt`.
- Use structured errors and `thiserror` where appropriate.
- Use `tracing` for runtime logs.
- Keep unsafe code narrow and explain the safety contract.
- Validate external sizes, offsets, tensor geometry, and resource budgets.
- Prefer existing crate ownership and abstractions over parallel frameworks.
- Avoid unrelated refactoring in fixes and performance changes.

## Commit and History Policy

Use clear imperative subjects, preferably Conventional Commit prefixes such as
`feat:`, `fix:`, `perf:`, `docs:`, `test:`, `ci:`, or `chore:`. Pull requests
normally squash into one reviewed commit. Published branch and tag history is
not rewritten to improve appearance.

## Review

CODEOWNERS identifies the responsible maintainers. Review depth scales with
the blast radius: parsers, unsafe kernels, public APIs, and release workflows
receive stricter review than isolated documentation changes.

By participating, you agree to [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) and the
project [Governance](GOVERNANCE.md).
