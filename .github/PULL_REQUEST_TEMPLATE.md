## Summary

<!-- Explain the user-visible result in 1-3 bullets. -->

-

## Motivation and Scope

<!-- Link the issue/spec. State what is intentionally out of scope. -->

## Change Type

- [ ] Correctness or security fix
- [ ] Backward-compatible feature
- [ ] Performance or memory improvement
- [ ] Model/format/backend support
- [ ] Public CLI or HTTP API change
- [ ] Internal refactor with no behavior change
- [ ] Documentation, CI, or release infrastructure
- [ ] Breaking change (requires migration and release approval)

## Compatibility Review

| Contract | Impact and evidence |
|---|---|
| OpenAI-compatible HTTP fields/routes | |
| GGUF loading and supported formats | |
| CPU-only build and fallback | |
| CUDA behavior and GPU safety | |
| Release/package behavior | |

Use `No change` only after checking the affected path.

## Test Evidence

<!-- List exact commands, hosted run URLs, fixtures, and expected failures. -->

```text

```

## Performance and Resource Impact

<!-- Required for xrt-kernels, xrt-cuda, xrt-models, or xrt-runtime. -->
<!-- Include base/head raw JSON and the metadata required by docs/BENCHMARKING.md. -->

## Security and Safety

<!-- Cover untrusted input, bounds, secrets, network exposure, RAM/VRAM, and cleanup. -->

## Checklist

- [ ] Change is focused and linked to an issue/spec when required
- [ ] New behavior has success, rejection, and regression coverage
- [ ] `cargo fmt --all -- --check`
- [ ] `cargo check --workspace --all-targets --locked`
- [ ] `cargo test --workspace --locked`
- [ ] Clippy and benchmark-compile policy pass in hosted CI
- [ ] CUDA changes have guarded feature/parity evidence where applicable
- [ ] Performance-sensitive changes include comparable base/head evidence
- [ ] Documentation and changelog are updated
- [ ] No secrets, model files, caches, local paths, or build output are committed
- [ ] I have signed the [Contributor License Agreement](../CLA.md)
