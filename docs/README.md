# xeno-rt Documentation

This directory contains the maintained technical documentation and design
records for xeno-rt. The root [README](../README.md) is the shortest path to a
working CPU or CUDA build.

Start architecture work with [Runtime Domains](RUNTIME_DOMAINS.md). XENO RT is
one local inference runtime with implemented `xrt-text`, experimental
`xrt-image`, and planned `xrt-video`/`xrt-audio` capability boundaries.
Consumer applications own editing/workflow experiences.

## User and Operator Guides

| Document | Use it for |
|---|---|
| [API](API.md) | HTTP routes, request fields, streaming, and security boundary |
| [Configuration](CONFIGURATION.md) | Backend, memory, cache, scheduler, and proxy settings |
| [Supported Models](SUPPORTED_MODELS.md) | Architecture, source format, quantization, and backend matrix |
| [Benchmarking](BENCHMARKING.md) | Reproducible CPU/CUDA measurements and reporting rules |
| [Roadmap](ROADMAP.md) | Milestones, maintainability program, and v1.0 arrival criteria |
| [Runtime Domains](RUNTIME_DOMAINS.md) | Product scope, ownership, shared architecture, and admission policy |

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

### `xrt-text`

- [Gemma4 Support Spec](GEMMA4_SUPPORT_SPEC.md)
- [KTransformers-inspired exact hybrid MoE](ktransformers-inspired-hybrid-moe-acceleration-spec.md)
- [GPU Runtime Acceleration Spec](GPU_RUNTIME_ACCELERATION_SPEC.md)
- [Agent-Adaptive KV Roadmap](AGENT_ADAPTIVE_KV_ROADMAP.md)
- [TurboQuant KV Cache Plan](turboquant-kv-cache-plan.md)

The working text implementation remains in `xrt-runtime` and `xrt-models`;
`xrt-text` is the capability name, not an empty facade crate.

### `xrt-image`

- [Qwen Image inference specification](xrt-image-qwen-image-inference-spec.md)
- [Qwen Image reference harness](../reference/image/qwen/README.md)
- [Qwen Image quality admission](../reference/image/qwen/QUALITY_ADMISSION.md)
- [Phase 0 evidence](../benchmark-results/image/phase0-2026-07-21/README.md)

The native crate exists, but no model/backend/quantization tuple is
production-advertised until the specification's remaining gates pass.

### Evidence and historical handoffs

- [Hybrid/MoE benchmark protocol](../benchmark-results/hybrid-moe/README.md)
- [2026-07-17 session state](SESSION-STATE-2026-07-17.md) is a historical
  `xrt-text` CUDA handoff, not the current whole-product definition.

Specifications may contain future work. The maintained support contract is the
combination of the current source, release notes, and Supported Models matrix.
