# Release Process

This document defines the xeno-rt repository checkpoint and GitHub archive
release. The broader XENO R2/Hub release is governed by `release-guide/` and is
executed from `xeno-platform`.

## Non-Negotiable Rules

- Read every file in `release-guide/` in order before release work.
- Run a dry-run before any publication.
- Obtain explicit human approval before a tag, push, upload, or deployment.
- Never move or overwrite a published tag/version.
- Never rewrite public `main` history as release cleanup.
- Release only from a clean, reviewed, protected commit with green checks.
- Preserve CPU fallback, GGUF support, and documented HTTP compatibility.

## Versioning

xeno-rt follows Semantic Versioning for documented user-facing behavior:

- `MAJOR`: incompatible CLI, HTTP, model-support, or artifact contract change.
- `MINOR`: backward-compatible feature/backend/model support.
- `PATCH`: backward-compatible correctness, security, or packaging fix.

Release candidates use `vX.Y.Z-rc.N`. The tag version must exactly match
`workspace.package.version` in `Cargo.toml`.

Internal Rust crates are not independently published and do not yet have a
separate public semver contract.

## Release Surfaces

### GitHub source and binary release

This repository owns portable Linux x86-64 and Windows x86-64 CLI/server
archives, checksums, SBOMs, provenance attestations, notes, and source tags.

### XENO R2 and Hub release

The canonical publisher runs from `xeno-platform` using product slug `rt`.
The current catalog declares desktop delivery, while xeno-rt currently creates
CLI/server archives rather than an installer. Do not publish an R2 desktop feed
until a real installer exists or the catalog contract is deliberately changed.

## Stabilization Flow

1. Freeze the scoped release changes on `main`.
2. Update `CHANGELOG.md`, support matrices, and known limitations.
3. Confirm dependency/license/security review and a committed `Cargo.lock`.
4. Run hosted CPU checks, CUDA-feature compilation, and security workflows.
5. Re-run guarded real-model GPU validation only when runtime/CUDA behavior
   changed or existing evidence no longer applies.
6. Cut `release/X.Y` after approval.
7. Run the release workflow manually in dry-run mode.
8. Download and inspect every archive, checksum, and SBOM.
9. Create an RC tag after explicit approval.
10. Hold the RC for downstream validation and resolve blockers with a new RC.
11. Create the stable tag only after a documented go/no-go decision.

## Required Gates

- Format, check, tests, Clippy policy, benchmark compile, and docs validation.
- CUDA-feature compile on hosted infrastructure.
- MSRV verification or an honestly updated MSRV declaration.
- Dependency advisory and license/source policy checks.
- CodeQL, dependency review, and workflow security checks where supported.
- Clean release build from `Cargo.lock` using `--locked`.
- Linux and Windows archive inventory validation.
- SHA-256 per artifact and combined checksum manifest.
- SPDX SBOM per platform archive.
- GitHub build provenance on tag publication.
- Changelog, release notes, security status, and known-limitations review.

## Dry-Run Acceptance

A dry-run must not create a GitHub release or tag. It must upload workflow
artifacts that can be inspected for:

- expected executable names;
- README, LICENSE, NOTICE, CHANGELOG, and release documentation;
- matching version/commit metadata;
- portable CPU baseline;
- valid checksums and SBOMs;
- no secrets, model files, caches, or developer-machine paths.

## Stable Publication

After explicit approval, push the immutable tag. The tag-triggered workflow
must rebuild from source rather than promote unverified local files. Verify the
GitHub release page, archive downloads, checksum manifest, SBOMs, and
attestations from a clean environment.

Do not start the XENO R2/Hub path until its delivery contract is satisfied.

## Failure and Rollback

- A failed dry-run is fixed on the branch and rerun.
- A bad RC is superseded by `rc.N+1`; it is not moved.
- A bad stable release is documented and fixed with a patch version.
- Compromised artifacts are removed from distribution, disclosed, and replaced
  by a new version according to the security process.

The release manager records commands, workflow URLs, artifact hashes, review
approval, and the final go/no-go decision in the release issue or PR.
