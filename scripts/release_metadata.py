#!/usr/bin/env python3
"""Validate a release ref against Cargo workspace metadata."""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSION = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", required=True, help="tag or version to validate")
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()

    expected = args.expected.removeprefix("v")
    if not VERSION.fullmatch(expected):
        raise SystemExit(f"invalid release version: {args.expected}")

    manifest = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    actual = manifest["workspace"]["package"]["version"]
    if expected != actual:
        raise SystemExit(
            f"release version {expected} does not match workspace version {actual}"
        )
    if not (ROOT / "Cargo.lock").is_file():
        raise SystemExit("Cargo.lock is required for a release")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    if f"## [{actual}]" not in changelog:
        raise SystemExit(f"CHANGELOG.md has no {actual} section")
    release_notes = ROOT / "docs" / "releases" / f"{actual}.md"
    if not release_notes.is_file():
        raise SystemExit(f"missing release notes: {release_notes.relative_to(ROOT)}")

    print(f"validated release version {actual}")
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as output:
            output.write(f"version={actual}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
