#!/usr/bin/env python3
"""Validate repository policy without compiling the Rust workspace."""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
HEX_SHA = re.compile(r"^[0-9a-f]{40}$")
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
ACTION_USE = re.compile(r"^\s*-?\s*uses:\s*([^\s#]+)", re.MULTILINE)
CONTAINER_IMAGE = re.compile(r"^\s*image:\s*([^\s#]+)", re.MULTILINE)


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def check_required_files(errors: list[str]) -> None:
    required = [
        "Cargo.lock",
        "README.md",
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "GOVERNANCE.md",
        "LICENSE",
        "NOTICE",
        "RELEASE.md",
        "SECURITY.md",
        "SUPPORT.md",
        "docs/README.md",
        "docs/API.md",
        "docs/ARCHITECTURE.md",
        "docs/BENCHMARKING.md",
        "docs/CONFIGURATION.md",
        "docs/DEVELOPMENT.md",
        "docs/ROADMAP.md",
        "docs/SUPPORTED_MODELS.md",
    ]
    for relative in required:
        if not (ROOT / relative).is_file():
            fail(errors, f"missing required file: {relative}")


def check_manifest_and_build_policy(errors: list[str]) -> None:
    manifest = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    workspace_package = manifest.get("workspace", {}).get("package", {})
    version = workspace_package.get("version")
    if not isinstance(version, str):
        fail(errors, "workspace.package.version is missing")
        return

    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    if f"## [{version}]" not in changelog:
        fail(errors, f"CHANGELOG.md has no {version} section")
    if not (ROOT / "docs" / "releases" / f"{version}.md").is_file():
        fail(errors, f"missing release notes for workspace version {version}")

    if workspace_package.get("publish") is not False:
        fail(errors, "workspace.package.publish must remain false until publication policy exists")

    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    ignored = {line.strip().lstrip("/") for line in gitignore if line.strip() and not line.startswith("#")}
    if "Cargo.lock" in ignored:
        fail(errors, "Cargo.lock must not be ignored")

    cargo_config = (ROOT / ".cargo" / "config.toml").read_text(encoding="utf-8")
    if "target-cpu=native" in cargo_config or "target-feature=+" in cargo_config:
        fail(errors, "default Cargo configuration must not force host-specific CPU features")


def check_markdown_links(errors: list[str]) -> None:
    markdown_files = sorted(ROOT.glob("*.md")) + sorted((ROOT / "docs").rglob("*.md"))
    for document in markdown_files:
        text = document.read_text(encoding="utf-8")
        for raw_target in MARKDOWN_LINK.findall(text):
            target = raw_target.strip().strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            path_part = unquote(target.split("#", 1)[0])
            if not path_part or any(marker in path_part for marker in ("${{", "*", "<", ">")):
                continue
            resolved = (ROOT / path_part.lstrip("/")) if path_part.startswith("/") else (document.parent / path_part)
            if not resolved.resolve().exists():
                relative_document = document.relative_to(ROOT)
                fail(errors, f"broken relative link in {relative_document}: {target}")


def check_workflow_pins(errors: list[str]) -> None:
    workflow_dir = ROOT / ".github" / "workflows"
    for workflow in sorted(workflow_dir.glob("*.y*ml")):
        text = workflow.read_text(encoding="utf-8")
        for use in ACTION_USE.findall(text):
            if use.startswith(("./", "docker://")):
                continue
            if "@" not in use:
                fail(errors, f"unversioned action in {workflow.relative_to(ROOT)}: {use}")
                continue
            _, revision = use.rsplit("@", 1)
            if not HEX_SHA.fullmatch(revision):
                fail(errors, f"action is not pinned to a full commit SHA in {workflow.relative_to(ROOT)}: {use}")

        for image in CONTAINER_IMAGE.findall(text):
            if image.startswith("${{"):
                continue
            if "@sha256:" not in image:
                fail(errors, f"container image is not digest-pinned in {workflow.relative_to(ROOT)}: {image}")


def check_accidental_artifacts(errors: list[str]) -> None:
    forbidden = [
        "stock_video",
        "xeno-backend-live-closeout-prompts.md",
        "xeno-desktop-backend-agent-prompts.md",
    ]
    for relative in forbidden:
        if (ROOT / relative).exists():
            fail(errors, f"accidental root artifact is present: {relative}")

    if any(ROOT.glob(".target-local*/CACHEDIR.TAG")):
        fail(errors, "local Cargo target cache marker is present")


def main() -> int:
    errors: list[str] = []
    check_required_files(errors)
    check_manifest_and_build_policy(errors)
    check_markdown_links(errors)
    check_workflow_pins(errors)
    check_accidental_artifacts(errors)

    if errors:
        print("repository policy check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("repository policy check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
