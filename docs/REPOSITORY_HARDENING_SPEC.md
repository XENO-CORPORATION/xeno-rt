# Repository Hardening and v0.2.0 Release Specification

Status: Hosted validation in progress
Owner: XENO Corporation
Created: 2026-07-16
Last updated: 2026-07-17

## 1. Executive Summary

This specification turns the merged native CPU/CUDA runtime into a durable,
public v0.2.0 checkpoint. The work covers repository presentation, contributor
experience, release reproducibility, supply-chain controls, hosted validation,
repository policy, and an auditable release process.

The checkpoint deliberately separates stabilization from broad source
reorganization. The current runtime has passed hosted CPU CI and guarded RTX
4090 validation. Moving tens of thousands of lines between modules before the
first CUDA release would add review risk without changing user behavior.
Behavior-preserving module decomposition remains a follow-up program after the
v0.2.0 baseline is tagged.

## 2. Background and Problem Statement

PR #15 merged native CUDA inference into `main` as one squash commit. The code
now has real CPU and CUDA execution, but the repository still presents the
pre-CUDA project in several places and is not ready for a reproducible binary
release.

The release blockers found in the 2026-07-16 audit are:

- `Cargo.lock` is ignored and absent, while the release workflow invokes Cargo
  with `--locked`.
- `.cargo/config.toml` globally enables `target-cpu=native`, so binaries built on
  hosted runners are not guaranteed to be portable to older x86-64 CPUs.
- `README.md`, `SECURITY.md`, and `CHANGELOG.md` still describe the v0.1 state.
- The release workflow emits archives but no checksums, provenance, SBOM, or
  tag-to-manifest version check.
- GitHub Actions dependencies use movable tags instead of immutable commit
  SHAs.
- `main` has no branch protection or repository ruleset.
- Repository description, homepage, and topics are empty.
- Dependabot security updates and GitHub secret scanning are disabled.
- Accidental public artifacts are tracked: a generated Cargo cache marker, an
  empty `stock_video` file, and two unrelated internal infrastructure prompt
  documents.
- The largest implementation units are difficult to navigate
  (`xrt-cuda/src/lib.rs` is over 27,000 lines and
  `xrt-runtime/src/backend.rs` is over 9,000 lines). They need staged
  decomposition after the release checkpoint.

## 3. Goals

1. Publish an accurate, navigable public repository surface for v0.2.0.
2. Preserve OpenAI API compatibility, GGUF support, CPU fallback, and validated
   CUDA behavior.
3. Make CPU release artifacts portable, reproducible from a committed lockfile,
   checksummed, and attributable to a specific GitHub workflow and commit.
4. Run all release gates in hosted CI and keep GPU execution manual and guarded.
5. Establish maintainable contribution, security, support, benchmark, and
   release contracts.
6. Protect `main` from force pushes and unvalidated direct changes.
7. Produce a clean v0.2.0 checkpoint without rewriting already-published Git
   history.

## 4. Non-Goals

- Rewriting or force-pushing public `main` history.
- Publishing private crates to crates.io.
- Claiming parity with llama.cpp, vLLM, SGLang, or ExLlama without a controlled
  benchmark.
- Re-running real-model GPU workloads while the shared RTX workstation exceeds
  the documented 4,096 MiB initial-use safety threshold.
- Refactoring the full CUDA backend, runtime backend, CLI, and server in the same
  release-hardening change.
- Publishing an R2 desktop release before a real installer exists or the XENO
  product catalog delivery contract is changed.
- Uploading artifacts, creating a tag, or deploying website content without the
  explicit confirmation required by `release-guide/`.

## 5. Repository Findings

### 5.1 Runtime and compatibility

- Workspace version: `0.2.0`.
- CPU is the always-available backend.
- CUDA is optional and feature-gated.
- Standard dense GGUF CUDA decode supports F32, F16, BF16, Q8_0, Q4_0, Q4_K,
  Q5_K, and Q6_K paths for the documented architecture set.
- The OpenAI-compatible server exposes completions, chat completions, models,
  runtime lifecycle/status, and background-removal routes.
- SafeTensors CUDA execution supports documented Qwen2/Qwen3 dense and packed
  formats; SafeTensors CPU decode is intentionally unsupported.

### 5.2 Existing public project infrastructure

The repository already includes Apache-2.0 licensing, a CLA, DCO, contributor
guide, code of conduct, security policy, issue forms, PR template, CODEOWNERS,
Dependabot, CI, dependency audit, release automation, and a portable XENO
release guide. These should be repaired and connected rather than replaced.

### 5.3 History

PR #15 was squash-merged as one verified commit. Its development commits are
visible in the PR but are not first-parent commits on `main`. The eight July 9
runner-setup commits are noisy but legitimate operational history. Rewriting
them would invalidate hashes, signatures, PR references, and existing clones
for no runtime benefit.

Future history should use protected pull requests and squash merges. Formatting
only commits may be added to `.git-blame-ignore-revs` after they exist.

### 5.4 Distribution mismatch

The XENO platform catalog declares product slug `rt` as `delivery: desktop`, but
this repository currently produces portable CLI/server archives, not a desktop
installer. A GitHub v0.2.0 release can be completed after validation. Publishing
the XENO R2 desktop feed requires either:

1. a real signed/packaged installer, or
2. a deliberate catalog delivery change coordinated in `xeno-platform`.

Neither decision will be guessed inside this repository.

## 6. Research and Standards

- GitHub recommends pinning Actions dependencies to full commit SHAs:
  <https://docs.github.com/en/actions/reference/security/secure-use>
- GitHub artifact attestations bind released binaries to repository, workflow,
  commit, and build identity:
  <https://docs.github.com/en/actions/concepts/security/artifact-attestations>
- GitHub rulesets can require pull requests and status checks while blocking
  force pushes:
  <https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/about-rulesets>
- Cargo's `rust-version` declares the supported compiler and should be verified
  in automation:
  <https://doc.rust-lang.org/stable/cargo/reference/rust-version.html>
- Cargo package metadata should identify repository, license, documentation,
  and supported toolchain:
  <https://doc.rust-lang.org/cargo/reference/manifest.html>
- OpenSSF Scorecard provides automated security-health checks for public open
  source repositories:
  <https://github.com/ossf/scorecard-action>

## 7. Functional Requirements

### 7.1 Repository surface

- Rewrite the README around the actual v0.2.0 capabilities and limitations.
- Add a documentation index and focused architecture, API, configuration,
  model-support, benchmark, and development references.
- Add explicit governance and support policies, including the supported public
  interfaces and the boundary between community support and security reports.
- Keep the detailed GPU implementation log discoverable without making it the
  onboarding path.
- Update changelog and security support tables to v0.2.0.
- Remove unrelated and generated root artifacts from tracking.
- Add package metadata and make the intended publication policy explicit.

### 7.2 Reproducible builds

- Commit `Cargo.lock` for the binary workspace.
- All hosted build, test, benchmark-compile, audit, and release commands use
  `--locked` where supported.
- Remove globally forced native CPU flags from normal builds.
- Document an explicit opt-in native build for local benchmark work.
- Release archives contain `xrt-cli`, `xrt-server`, README, license, notice,
  changelog, and release documentation.

### 7.3 CI and security

- Set minimal workflow permissions, concurrency, and timeouts.
- Pin third-party Actions to immutable full-length SHAs with version comments.
- Test format, check, workspace tests, Clippy, benchmark compilation, and CUDA
  feature compilation on hosted runners.
- Verify the declared MSRV or update the declaration to the minimum version
  actually proven by CI.
- Run dependency audit and license/source policy checks.
- Add dependency-review, CodeQL, and OpenSSF Scorecard workflows where GitHub
  plan support permits them.
- Keep self-hosted CUDA execution manual, serial, guarded, and unavailable to
  untrusted pull-request code.

### 7.4 Release integrity

- A release tag must exactly match the workspace version.
- Release artifacts are produced for Linux x86-64 and Windows x86-64.
- Each archive has a SHA-256 checksum.
- The release includes a combined checksum manifest.
- The release includes an SPDX SBOM for each platform archive.
- GitHub build provenance attestations cover the downloadable archives.
- A manually dispatched release dry-run builds and uploads workflow artifacts
  without creating a GitHub release.
- Tag-triggered publication requires all build jobs to succeed.

### 7.5 Repository policy

- Configure description, homepage, and relevant topics.
- Enable vulnerability alerts, Dependabot security updates, secret scanning,
  and push protection where available.
- Protect `main` with pull-request and required-status-check rules and block
  force pushes/deletions.
- Preserve squash merge as the preferred integration strategy and delete merged
  branches automatically.

## 8. Non-Functional Requirements

- No CPU throughput claim changes without a benchmark.
- No CUDA behavior changes in repository-hardening commits.
- No network-facing API incompatibility.
- No model format removal.
- Documentation examples must correspond to current CLI/server source.
- Workflows must not expose secrets to pull-request code.
- Release artifacts must run on the documented portable CPU baseline rather
  than the hosted runner's exact CPU.

## 9. Proposed Repository Layout

```text
xeno-rt/
|-- .github/                 # issue forms, ownership, CI and release policy
|-- benches/                 # Criterion and runtime benchmark entry points
|-- crates/                  # production crates with narrow ownership
|-- docs/
|   |-- README.md            # documentation map
|   |-- ARCHITECTURE.md
|   |-- API.md
|   |-- BENCHMARKING.md
|   |-- CONFIGURATION.md
|   |-- DEVELOPMENT.md
|   |-- SUPPORTED_MODELS.md
|   `-- *_SPEC.md            # implementation specs and progress evidence
|-- release-guide/           # portable XENO release playbook
|-- scripts/                 # guarded operator and fixture scripts
|-- tests/                   # cross-crate integration tests and fixtures
|-- xtask/                   # repository development commands
|-- CHANGELOG.md
|-- CONTRIBUTING.md
|-- GOVERNANCE.md
|-- RELEASE.md
|-- SECURITY.md
|-- SUPPORT.md
`-- README.md
```

The layout preserves current paths used by tests, workflows, and prior PRs.
Large source files will be decomposed in follow-up PRs behind unchanged public
crate APIs.

## 10. Implementation Plan

### Phase A: clean public baseline

1. Remove accidental tracked artifacts and expand ignore rules.
2. Commit a lockfile generated by hosted Cargo 1.76 so its format remains
   readable by the declared MSRV.
3. Repair workspace/package metadata and portable build defaults.
4. Rewrite README and focused reference documentation.
5. Update changelog, security support, contribution, and release contracts.

### Phase B: automation and supply chain

1. Pin workflow dependencies.
2. Harden standard CI and add hosted CUDA compile coverage.
3. Add MSRV, dependency/license, dependency-review, CodeQL, and Scorecard gates.
4. Harden release packaging, checksums, dry-run behavior, and attestations.
5. Generate SPDX SBOMs for release archives and retain them with the release
   evidence.

### Phase C: hosted verification

1. Push the hardening branch.
2. Run standard hosted CI and repair every failure.
3. Run release workflow in dry-run mode and inspect both archives/checksums.
4. Do not run real-model GPU validation unless runtime code changed.

### Phase D: repository checkpoint

1. Open one curated hardening PR.
2. Configure repository metadata and security features.
3. Apply a `main` ruleset after required checks have reported successfully.
4. Merge with squash after review.
5. Cut `release/0.2`, then propose `v0.2.0` only after explicit approval.
6. Run the GitHub release workflow and verify artifacts and attestations.
7. Resolve the XENO platform `rt` desktop-delivery mismatch before any R2
   product-feed publication.

## 11. Testing Strategy

- Static checks: formatting, YAML parsing, Markdown link/path checks, manifest
  consistency, release tag/version validation.
- Hosted Rust checks: locked workspace check, tests, Clippy, benchmark compile,
  all-feature/CUDA compile, and MSRV check.
- Security checks: cargo audit, cargo deny, dependency review, CodeQL, secret
  scan, and OpenSSF Scorecard.
- Release checks: manual dry-run matrix, archive inventory, SHA-256 manifest,
  SPDX SBOM validation, and attestation creation on tag publication.
- GPU checks: retain the already-green guarded RTX evidence. Re-run only if
  runtime/CUDA code changes.

## 12. Rollout and Rollback

- All changes land through a release-hardening branch and PR.
- Documentation and workflow changes can be reverted independently.
- Portable build flag changes are validated by hosted binary builds before
  merge.
- No tag is moved or overwritten. A bad release candidate is superseded by a
  new RC tag; a bad stable artifact requires a patch version.
- Public `main` history is never force-rewritten as part of this work.

## 13. Acceptance Criteria

- [x] README and focused docs accurately describe v0.2.0 CPU/CUDA behavior.
- [x] Accidental root/build artifacts are no longer tracked.
- [x] `Cargo.lock` is committed and all release gates use it.
- [x] Default release binaries do not use `target-cpu=native`.
- [x] Standard hosted CI is green from a clean checkout.
- [x] CUDA feature compilation is covered without executing a real model.
- [ ] Dependency, license/source, CodeQL, and workflow security checks are green
      or documented as unavailable due to plan permissions.
- [ ] Release dry-run produces Linux and Windows archives plus SHA-256 files.
- [ ] Release dry-run produces an SPDX SBOM for each platform archive.
- [ ] Tag/workspace version mismatch fails before publication.
- [ ] Release archives receive GitHub provenance attestations on tag builds.
- [ ] Repository metadata and security settings are configured.
- [ ] `main` blocks force pushes and requires pull requests plus green checks.
- [x] No public history rewrite is performed.
- [ ] v0.2.0 tag/publication occurs only after explicit human approval.
- [ ] The first checkpoint is treated as a GitHub release candidate until the
      XENO platform delivery decision is resolved.
- [ ] R2 publication remains blocked until the `rt` desktop installer contract
      is satisfied or deliberately changed.

## 14. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Repository polish changes runtime behavior | Keep runtime source out of the hardening PR unless a release blocker is proven |
| Portable binaries lose benchmark speed | Keep native optimization opt-in and benchmark both profiles separately |
| Workflow pin becomes stale | Dependabot tracks GitHub Actions updates; version comments keep review readable |
| New required checks deadlock merges | Run each check first, then enable the ruleset using observed check names |
| Release workflow publishes accidentally | Manual runs are dry-run only; publication remains tag-gated |
| R2 feed advertises a nonexistent installer | Do not publish until delivery contract is resolved |
| Source decomposition destabilizes v0.2.0 | Defer it to staged, API-preserving follow-up PRs |

## 15. Follow-Up Engineering Program

After v0.2.0 is tagged, decompose large modules in behavior-preserving slices:

1. Move embedded PTX sources from `xrt-cuda/src/lib.rs` to dedicated kernel
   resources loaded with `include_str!`.
2. Split CUDA buffer, graph, memory-pool, quantized-matrix, and KV-cache types
   into private modules with unchanged re-exports.
3. Split `xrt-runtime/src/backend.rs` by CPU, CUDA, external, and shared backend
   contracts.
4. Split server request types, handlers, state, and router construction.
5. Split CLI argument/report types from command execution.
6. Add public API docs and semver checks before publishing any Rust libraries.

Each decomposition PR must compile and test independently and must not combine
with kernel optimization or API behavior changes.

## 16. Open Decisions

1. Should XENO RT remain a `desktop` product with a future installer, or should
   the platform catalog describe the current archive/CLI distribution?
2. Is v0.2.0 intended as a GitHub-only beta checkpoint or an R2/Hub release?
3. Which organization team should be the second required reviewer when that
   team exists?

The first two decisions block R2 publication but do not block repository
hardening, hosted validation, or a GitHub v0.2.0 release candidate.

## 17. Progress Record

Completed locally on 2026-07-16:

- Replaced the stale public README and added focused architecture, API,
  configuration, model-support, benchmark, development, support, governance,
  roadmap, and release-note documentation.
- Removed accidental tracked artifacts and made package publication policy and
  portable CPU build defaults explicit.
- Added immutable Action pins, least-privilege permissions, concurrency,
  timeouts, hosted CUDA-feature compilation, MSRV validation, dependency and
  source policy, dependency review, CodeQL, and OpenSSF Scorecard workflows.
- Added a release version contract, portable Linux and Windows archives,
  checksums, SPDX SBOMs, provenance attestations, and a non-publishing manual
  dry-run path.
- Added executable repository-policy and release-metadata checks.
- Kept runtime, model, API, test, and benchmark source unchanged.
- Passed local static validation for workflow YAML, Python and PowerShell
  syntax, repository links and policy, release metadata, and
  `git diff --check`.

Hosted progress on 2026-07-17:

- Pushed `chore/release-hardening-20260716` after explicit approval.
- Workflow run `29564577197` generated `Cargo.lock` with Cargo 1.76 and stopped
  at the intentional uncommitted-lockfile gate.
- Imported the generated lockfile as format version 3 with SHA-256
  `f05a9ab9466118174ce15dc09f1c039e7a6817894b7cd4322d2b3591caee4afe`.
- Workflow run `29565020677` passed locked check, test, format, Clippy,
  benchmark compilation, CUDA feature compilation, Rust 1.76 MSRV, and
  repository-policy jobs from one exact commit.
- Dependency-policy run `29565144724` passed license, source, and advisory
  policy evaluation far enough to identify two blockers: internal path
  dependencies were rejected by the wildcard policy, and PyO3 0.25.1 was
  affected by RUSTSEC-2026-0176 and RUSTSEC-2026-0177.
- Kept wildcard version requirements denied while allowing private workspace
  path dependencies, upgraded the experimental Python binding to PyO3 0.29,
  and split its Rust 1.83 MSRV from the core runtime's Rust 1.76 contract.
- Hardened the lockfile gate to regenerate and expose an artifact when the
  committed lockfile is stale as well as when it is absent.
- Workflow run `29565729968` proved that `cargo metadata --no-deps` does not
  validate locked dependency resolution: the gate passed, while every locked
  Cargo job rejected the stale lockfile. The gate now resolves full metadata.
- Corrected bootstrap run `29565837767` stopped at the intentional stale-lock
  gate and uploaded a Cargo 1.76-generated format-version-3 lockfile containing
  PyO3 0.29.0. The artifact SHA-256 is
  `6fb520ca7471eb6f41843c18982ed3f128c5d61c491abc9dfeb5ccfcfdd9aa80`.

Remaining remote gates:

1. Rerun locked hosted CI and dependency policy, repair only evidence-backed
   failures, and complete a release dry-run.
2. Configure repository metadata, security features, and a `main` ruleset
   after the new check names have reported.
3. Review and squash-merge the hardening PR.
4. Obtain separate explicit approval before creating `v0.2.0` or publishing
   any artifact.
5. Resolve the XENO platform desktop-delivery mismatch before R2 or Hub
   publication.
