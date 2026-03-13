# Release Process

## Purpose

This document defines how xeno-rt is versioned, stabilized, and shipped. The
goal is a predictable release train that preserves performance, correctness, and
operational stability for downstream users embedding the runtime in production
systems.

## Versioning Policy

xeno-rt follows [Semantic Versioning 2.0.0](https://semver.org/).

- `MAJOR` releases are reserved for breaking API, CLI, model-format, or
  behavior changes that require downstream migration.
- `MINOR` releases add backwards-compatible functionality, backend support,
  performance improvements, or new developer tooling.
- `PATCH` releases are limited to backwards-compatible fixes for correctness,
  security, packaging, or severe regressions.

Pre-release candidates use the form `vX.Y.Z-rc.N`.

## Release Cadence

xeno-rt ships on a monthly cadence. The default release train is:

- Week 1: feature landing and stabilization on `main`
- Week 2: feature freeze and release branch cut
- Week 3: release candidate validation and documentation review
- Week 4: stable release

The cadence may be adjusted for holidays, security fixes, or major integration
work, but monthly releases remain the default expectation.

## Branching Strategy

The repository uses a simple two-track model:

- `main` is the integration branch for active development. All feature work
  lands here first.
- `release/x.y` branches are cut from `main` for each minor release line and
  remain the source of truth for release candidates, stable tags, and patch
  releases in that line.

Rules:

- Every change intended for a release branch must exist on `main` first unless a
  security incident requires an emergency exception.
- Release-only commits are limited to version bumps, changelog updates,
  packaging fixes, and release documentation.
- After a stable release, any release-only commits must be merged back to
  `main` immediately.

## Standard Monthly Release Flow

### 1. Feature Freeze

At feature freeze, the release manager:

- Confirms all scoped work is merged into `main`
- Defers incomplete or risky changes to the next cycle
- Cuts `release/x.y` from `main`
- Updates `CHANGELOG.md` and crate versions if needed

Only stabilization work may land on the release branch after the freeze.

### 2. Release Candidate Tag

Once the branch is in a releasable state:

- Tag the branch as `vX.Y.Z-rc.1`
- Publish pre-release artifacts through the release workflow
- Announce the candidate to maintainers and downstream testers

Additional candidates (`-rc.2`, `-rc.3`, and so on) are created only when
release-blocking issues are fixed during the RC window.

### 3. Testing Period

The release candidate remains open for validation for at least five calendar
days unless an urgent security fix shortens the cycle. The testing window must
cover:

- `cargo test --workspace`
- `cargo bench` spot checks for kernels and runtime hot paths
- CLI smoke tests using representative GGUF models
- OpenAI-compatible server smoke tests for completion and streaming endpoints
- Linux x86_64 and Windows x86_64 artifact verification

Any regression in correctness, API behavior, or release packaging blocks the
stable tag until resolved.

### 4. Stable Release

If the RC period closes without release blockers:

- Update final release notes and changelog entries
- Tag the release branch as `vX.Y.Z`
- Publish the GitHub Release and signed artifacts
- Announce the release in repository channels

Stable releases must always be cut from the corresponding `release/x.y` branch,
never directly from `main`.

## Patch Releases

Patch releases are exceptional maintenance releases on an existing
`release/x.y` branch. A change qualifies for cherry-pick into a patch release
only when it meets one or more of the following criteria:

- Fixes a security vulnerability or supply-chain exposure
- Fixes incorrect numerical results, data corruption, or model-loading failures
- Restores a broken build, release artifact, or installation path
- Resolves a documented regression in a supported API, CLI behavior, or runtime
  contract
- Unblocks a supported platform or hardware target that previously worked

The following do not qualify on their own:

- New features
- Large refactors
- Broad dependency upgrades without a user-facing fix
- Performance work without a correctness or stability justification

Cherry-pick rules:

- The change must be small, isolated, and low risk.
- The originating fix must be reviewed and merged on `main` first whenever
  practical.
- The cherry-pick commit must use `git cherry-pick -x` to preserve traceability.
- Every cherry-picked fix must include or update a regression test unless the
  issue is impossible to exercise automatically.

Patch releases may use a shortened RC period, but they still require at least
one validation pass on the affected platform or subsystem.

## Release Checklist

Before shipping any stable or patch release, confirm:

- The changelog accurately reflects user-visible changes
- CI is green on the release branch
- `cargo fmt`, `cargo test --workspace`, and targeted smoke tests pass
- Release artifacts are present for Linux x86_64 and Windows x86_64
- Security disclosures, if any, have coordinated publication timing
- Release notes call out breaking changes, new capabilities, and known issues

## Ownership

The release manager for a cycle is responsible for driving the checklist,
cutting tags, and coordinating go/no-go decisions. Performance-sensitive or
security-sensitive changes should not ship without sign-off from the relevant
code owner.
