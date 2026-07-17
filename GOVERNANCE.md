# Governance

## Project Model

xeno-rt is a maintainer-led open source project stewarded by XENO Corporation.
Maintainers are identified by repository access and `CODEOWNERS`. The project
welcomes external design discussion, bug reports, benchmarks, and pull
requests, while maintainers retain final responsibility for compatibility,
security, release scope, and project direction.

## Decision Process

- Small fixes and documentation changes are decided through pull request
  review.
- Public API, model-format, backend, security, or architecture changes require
  an issue or implementation specification before code is merged.
- Decisions favor correctness, explicit failure, CPU fallback, GGUF support,
  and reproducible evidence over short-term feature count.
- When consensus is not reached, the responsible maintainer records the
  decision and rationale in the issue, PR, or design document.

## Maintainer Responsibilities

Maintainers are expected to:

- enforce the compatibility and safety invariants;
- review code within their ownership area;
- require tests and benchmark evidence proportional to risk;
- disclose conflicts of interest;
- coordinate security reports privately;
- keep release notes, support matrices, and automation accurate;
- avoid rewriting published history or moving published tags.

## Contribution Lifecycle

1. Discuss broad or breaking work before implementation.
2. Open a focused pull request against `main`.
3. Pass required checks, CLA policy, and code-owner review.
4. Resolve review findings and document behavior changes.
5. Squash merge unless preserving separate commits is technically necessary.

Direct changes to protected branches, force pushes, and tag replacement are not
part of the normal governance process.

## Compatibility Authority

The supported public contract is defined by released documentation, release
notes, CLI behavior, documented HTTP routes/fields, GGUF support, and binary
artifacts. Internal Rust crate APIs are not yet a stable semver contract because
the crates are not published independently.

## Releases

The release manager owns the release checklist and go/no-go recommendation.
Stable publication requires green hosted checks, reviewed artifacts, a dry-run,
and explicit human approval. Security-sensitive releases also require the
private disclosure process in `SECURITY.md`.

## Conduct and Escalation

Participation is governed by `CODE_OF_CONDUCT.md`. Conduct concerns should use
the contact path in that document. Security vulnerabilities must use the
private process in `SECURITY.md`, not a public issue.

Governance changes are proposed and reviewed through a pull request to this
file.
