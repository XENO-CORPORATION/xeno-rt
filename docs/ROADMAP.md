# Roadmap

This roadmap defines the arrival points for xeno-rt. It is capability-driven,
not a promise of dates. A feature is complete only when its correctness,
fallback, observability, documentation, and release evidence are complete.

## v0.2: Native CPU/CUDA Checkpoint

Goal: preserve the proven CPU runtime while making the native CUDA backend a
real, reviewable, and releasable execution path.

- [x] GGUF-native CPU inference and quantized kernels
- [x] Backend and session abstractions with explicit active-backend status
- [x] GPU-resident compatible weights, KV state, and decode scratch
- [x] Native CUDA decode for the documented dense GGUF matrix
- [x] F32 and quantized CUDA KV modes
- [x] CUDA Graph support for compatible stable decode shapes
- [x] Prefix cache, bounded scheduling, and decode batching
- [x] Selected CUDA-only SafeTensors adapters
- [x] Structured CPU/CUDA benchmark and resource telemetry
- [x] Guarded self-hosted GPU workflow
- [ ] Green release-hardening CI from a committed lockfile
- [ ] Inspected Linux/Windows release dry-run with checksums and SBOMs
- [ ] Tag build provenance and SBOM attestations verified
- [ ] Reviewed and approved `v0.2.0` tag

The detailed implementation/evidence log is
[GPU_RUNTIME_ACCELERATION_SPEC.md](GPU_RUNTIME_ACCELERATION_SPEC.md). Repository
checkpoint progress is in
[REPOSITORY_HARDENING_SPEC.md](REPOSITORY_HARDENING_SPEC.md).

## v0.3: Compatibility and Correctness Depth

Goal: expand supported combinations without weakening explicit validation.

- Broaden real-model fixtures across documented architecture/quantization
  combinations.
- Add conformance fixtures for OpenAI-compatible non-streaming and SSE shapes.
- Ratchet Clippy debt to a warning-free enforced baseline.
- Add fuzz/property coverage for GGUF metadata, tensor bounds, tokenizer input,
  and HTTP request limits.
- Define and enforce numerical parity tolerances per dtype/backend.
- Stabilize C/Python packaging or keep it explicitly experimental.
- Add authenticated deployment guidance and optional middleware design without
  changing the default loopback safety boundary.

## v0.4: Performance and Operational Scale

Goal: improve measured throughput/latency under reproducible correctness gates.

- Profile and optimize CUDA kernels by model shape and quantization.
- Expand CUDA Graph and continuous-batch coverage where topology is stable.
- Reduce final-logit and scheduler overhead with evidence-backed changes.
- Improve CPU architecture dispatch without sacrificing portable binaries.
- Add long-running soak, fragmentation, cancellation, and overload tests.
- Publish controlled hardware/model benchmark baselines and regression budgets.

## Source Maintainability Program

After v0.2.0, decompose large implementation units without mixing movement and
behavior changes:

1. Split CUDA kernel resources, buffers, matrices, streams/graphs, KV storage,
   telemetry, and device lifecycle behind unchanged re-exports.
2. Split runtime backend contracts and CPU/CUDA/external implementations.
3. Split server types, handlers, router, lifecycle, and image routes.
4. Split CLI command arguments/reporting from execution.
5. Add semver/API checks before publishing any Rust crates.

Each step must independently pass hosted CPU/CUDA compile gates and existing
behavior tests.

## v1.0 Definition of Ready

xeno-rt is ready for a stable 1.0 claim when all of these are true:

- Supported model/source/backend combinations have maintained conformance and
  real-model evidence.
- CPU fallback and documented GGUF support are release-blocking gates.
- Documented OpenAI-compatible routes have automated compatibility tests.
- Resource limits, cancellation, overload, and shutdown behavior are tested.
- Release artifacts are portable, reproducible, checksummed, SBOM-attested,
  provenance-attested, and supported on declared platforms.
- Security reporting, dependency policy, code scanning, branch protection, and
  review ownership are active.
- Performance claims are published with reproducible raw evidence and
  correctness parity.
- Large modules have clear ownership boundaries and maintainable test surfaces.
- The XENO product catalog and actual distribution format agree.

Features outside that support contract can remain experimental after 1.0, but
they must be labeled and isolated from stable paths.
