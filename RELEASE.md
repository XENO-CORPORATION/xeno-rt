# Release Process

## Versioning

xeno-rt follows [Semantic Versioning](https://semver.org/):

- **MAJOR** (x.0.0) — Breaking API changes, major architecture shifts
- **MINOR** (0.x.0) — New features, model support, non-breaking enhancements
- **PATCH** (0.0.x) — Bug fixes, security patches, documentation fixes

## Release Cadence

- **Minor releases**: Monthly (first week of each month)
- **Patch releases**: As needed for bug fixes and security issues
- **Major releases**: As needed for breaking changes

## Release Process

### 1. Feature Freeze

One week before the target release date:

- Create a release branch: `release/vX.Y.Z`
- No new features merged to the release branch — only bug fixes
- Update `CHANGELOG.md` with all changes since the last release
- Update version numbers in all `Cargo.toml` files

### 2. Release Candidate

- Tag the release candidate: `vX.Y.Z-rc1`
- CI builds release artifacts automatically
- Community testing period (minimum 3 days for minor, 7 days for major)
- Fix any issues found, tag `rc2` if needed

### 3. Stable Release

- Tag the stable release: `vX.Y.Z`
- CI creates GitHub Release with binaries and changelog
- Publish crates to crates.io (when ready for public consumption)
- Announce on GitHub Discussions

### 4. Post-Release

- Merge release branch back to `main`
- Bump version on `main` to next development version

## Patch Release Criteria

A patch release is warranted for:

- Security vulnerabilities (any severity)
- Data corruption or loss bugs
- Crashes or panics during normal operation
- Incorrect inference output (wrong tokens generated)

A patch release is NOT warranted for:

- Performance regressions (unless severe)
- New feature requests
- Documentation-only changes
- CI/tooling improvements

## Cherry-Pick Process

For patch releases from a release branch:

1. Fix the issue on `main` first
2. Cherry-pick the commit(s) to the release branch
3. Verify with `cargo test --workspace` on the release branch
4. Tag the patch release

## Branching Strategy

```
main ─────────────────────────────────────────────────►
       \                    \
        release/v0.1.0       release/v0.2.0
        ├── v0.1.0-rc1       ├── v0.2.0-rc1
        ├── v0.1.0            ├── v0.2.0
        └── v0.1.1            └── ...
```

## Build Matrix

Release binaries are built for:

| Platform | Target | Format |
|---|---|---|
| Linux x86_64 | `x86_64-unknown-linux-gnu` | `.tar.gz` |
| Windows x86_64 | `x86_64-pc-windows-msvc` | `.zip` |

Future targets (planned):
- Linux aarch64 (`aarch64-unknown-linux-gnu`)
- macOS Apple Silicon (`aarch64-apple-darwin`)
- macOS Intel (`x86_64-apple-darwin`)
