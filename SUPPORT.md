# Support

## Supported Release Line

During the v0.2.0 beta checkpoint, support targets the latest `0.2.x` release
and current `main`. Security support is defined separately in `SECURITY.md`.

Release artifacts target Linux x86-64 and Windows x86-64. Other source-build
platforms are best effort unless they are added to hosted CI and the release
matrix.

## Where to Ask

- Usage question or design discussion: GitHub Discussions.
- Reproducible bug: GitHub Issues using the bug-report form.
- Feature proposal: GitHub Issues using the feature-request form.
- Security vulnerability: private GitHub Security Advisory or the email in
  `SECURITY.md`.

Do not publish secrets, private models, API keys, proprietary prompts, or
sensitive model output in an issue.

## Bug Report Checklist

Include:

- xeno-rt version/commit and whether the tree is dirty;
- exact command and relevant `XRT_*` variables with secrets removed;
- OS, CPU, RAM, Rust version, and build profile;
- for CUDA: GPU, driver, VRAM, and CUDA-feature status;
- model architecture, source format, quantization, and public model identifier
  when possible;
- expected behavior, actual behavior, and the smallest reproduction;
- logs with credentials and private content redacted.

For crashes or memory exhaustion, include process RAM, device memory before the
run, configured memory budgets, token count, cache mode, and concurrency.

## Scope

Maintainers prioritize:

- incorrect output caused by runtime defects;
- supported model load/decode regressions;
- CPU fallback failures;
- documented OpenAI-compatible API regressions;
- crashes, resource leaks, and release/installation failures;
- reproducible performance regressions with comparable evidence.

Model quality, prompt engineering, unsupported architectures, custom model
conversion, and application-specific deployment design may receive community
guidance but are not guaranteed support services.

## Response Expectations

Community support has no guaranteed SLA. Maintainers triage based on severity,
reproducibility, supported scope, and available capacity. Security report
timelines are listed in `SECURITY.md`.

Commercial support or private deployment arrangements are outside this public
repository's support contract.
