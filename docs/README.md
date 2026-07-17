# xeno-rt Documentation

This directory contains the maintained technical documentation and design
records for xeno-rt. The root [README](../README.md) is the shortest path to a
working CPU or CUDA build.

## User and Operator Guides

| Document | Use it for |
|---|---|
| [API](API.md) | HTTP routes, request fields, streaming, and security boundary |
| [Configuration](CONFIGURATION.md) | Backend, memory, cache, scheduler, and proxy settings |
| [Supported Models](SUPPORTED_MODELS.md) | Architecture, source format, quantization, and backend matrix |
| [Benchmarking](BENCHMARKING.md) | Reproducible CPU/CUDA measurements and reporting rules |
| [Roadmap](ROADMAP.md) | Milestones, maintainability program, and v1.0 arrival criteria |

## Contributor Guides

| Document | Use it for |
|---|---|
| [Architecture](ARCHITECTURE.md) | Runtime layers, ownership, data flow, and invariants |
| [Development](DEVELOPMENT.md) | Tooling, checks, tests, dependency updates, and PR workflow |
| [Repository Hardening Spec](REPOSITORY_HARDENING_SPEC.md) | v0.2.0 checkpoint requirements and progress |

Root-level project policy is in [CONTRIBUTING](../CONTRIBUTING.md),
[GOVERNANCE](../GOVERNANCE.md), [SUPPORT](../SUPPORT.md),
[SECURITY](../SECURITY.md), and [RELEASE](../RELEASE.md).

## Design and Progress Records

- [GPU Runtime Acceleration Spec](GPU_RUNTIME_ACCELERATION_SPEC.md) is the
  detailed CUDA implementation record and evidence log.
- [Agent-Adaptive KV Roadmap](AGENT_ADAPTIVE_KV_ROADMAP.md) describes the
  policy-aware cache direction.
- [TurboQuant KV Cache Plan](turboquant-kv-cache-plan.md) tracks exploratory KV
  compression work.
- [ONNX Integration Plan](ONNX_INTEGRATION_PLAN.md) covers image-task runtime
  integration.

Specifications may contain future work. The maintained support contract is the
combination of the current source, release notes, and Supported Models matrix.
