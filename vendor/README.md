# Pinned first-party primitives

`xenosystem-primitives` is the exact Rust crate from the canonical shared
repository and commit recorded in `xenosystem-primitives.source.json`.
This uses the shared directory resolver without an application-local fork.
The Rust crate has no package registry release yet. Including its original
source lets a clean runtime build work without a second private Git credential.

Do not edit the snapshot. Update upstream first, then import a complete pinned
crate and regenerate its source inventory. The inventory check runs in CI:
`python scripts/check-vendored-primitives.py`.

The crate retains its original `LicenseRef-Proprietary` package declaration;
the runtime workspace's Apache-2.0 declaration does not relicense this source.
