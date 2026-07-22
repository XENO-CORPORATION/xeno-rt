#!/usr/bin/env python3
"""Download Phase 0 reference artifacts into a verified local cache.

The production runtime never imports this file. Downloads are content-addressed,
resumable, bounded, and verified against the checked-in immutable lock/manifest.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Iterator


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
LOCK_PATH = HERE / "phase0-lock.json"
MANIFEST_DIR = HERE / "manifests"
USER_AGENT = "xeno-rt-qwen-image-reference/1"
CHUNK_BYTES = 8 * 1024 * 1024


def cache_root() -> Path:
    configured = os.environ.get("XRT_IMAGE_REFERENCE_CACHE")
    return Path(configured).expanduser().resolve() if configured else REPO_ROOT / ".codex-tmp" / "image-reference"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def checked_metadata(metadata: dict[str, Any], label: str) -> tuple[int, str, str]:
    size = metadata.get("size_bytes")
    digest = metadata.get("sha256")
    source = metadata.get("source") or metadata.get("url")
    if not isinstance(size, int) or size <= 0:
        raise RuntimeError(f"{label}: invalid size")
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError(f"{label}: invalid SHA-256")
    if not isinstance(source, str) or not source.startswith("https://"):
        raise RuntimeError(f"{label}: immutable HTTPS source is required")
    if "resolve/main/" in source or "?" in source:
        raise RuntimeError(f"{label}: mutable or credential-bearing source rejected")
    return size, digest, source


@contextlib.contextmanager
def cache_lock(root: Path, timeout_seconds: int = 120) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".download.lock"
    handle = path.open("a+b")
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    deadline = time.monotonic() + timeout_seconds
    acquired = False
    try:
        while not acquired:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for reference-cache lock: {path}")
                time.sleep(0.25)
        yield
    finally:
        if acquired:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def verified(path: Path, expected_size: int, expected_sha256: str) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == expected_size
        and sha256_file(path) == expected_sha256
    )


def stream_once(url: str, partial: Path, expected_size: int) -> None:
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        raise RuntimeError(f"partial file exceeds expected size: {partial}")
    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        status = getattr(response, "status", response.getcode())
        if offset and status != 206:
            offset = 0
        mode = "ab" if offset else "wb"
        with partial.open(mode) as handle:
            while chunk := response.read(CHUNK_BYTES):
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())


def hugging_face_coordinates(source: str) -> tuple[str, str, str] | None:
    parsed = urllib.parse.urlparse(source)
    if parsed.hostname != "huggingface.co":
        return None
    parts = parsed.path.strip("/").split("/")
    if len(parts) < 5 or parts[2] != "resolve":
        return None
    repo_id = "/".join(parts[:2])
    revision = urllib.parse.unquote(parts[3])
    filename = urllib.parse.unquote("/".join(parts[4:]))
    return repo_id, revision, filename


def acquire_hugging_face_blob(
    root: Path,
    blob: Path,
    source: str,
    expected_size: int,
    expected_sha256: str,
) -> bool:
    coordinates = hugging_face_coordinates(source)
    if coordinates is None:
        return False
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return False
    repo_id, revision, filename = coordinates
    cached = Path(
        hf_hub_download(
            repo_id=repo_id,
            revision=revision,
            filename=filename,
            cache_dir=root / "huggingface-hub",
        )
    )
    if not verified(cached, expected_size, expected_sha256):
        raise RuntimeError(f"Hugging Face cache verification failed: {repo_id}/{filename}")
    temporary = blob.with_suffix(f".{os.getpid()}.tmp")
    try:
        os.link(cached, temporary)
    except OSError:
        shutil.copyfile(cached, temporary)
    os.replace(temporary, blob)
    return True


def acquire_blob(root: Path, metadata: dict[str, Any], label: str, verify_only: bool) -> Path:
    expected_size, digest, source = checked_metadata(metadata, label)
    blob_dir = root / "blobs" / "sha256"
    blob_dir.mkdir(parents=True, exist_ok=True)
    blob = blob_dir / digest
    if verified(blob, expected_size, digest):
        return blob
    if blob.exists():
        raise RuntimeError(f"cached blob failed verification; quarantine it before retrying: {blob}")
    if verify_only:
        raise RuntimeError(f"required verified blob is not cached: {label}")
    partial = blob.with_suffix(".partial")
    if not partial.exists() and acquire_hugging_face_blob(
        root, blob, source, expected_size, digest
    ):
        return blob
    for attempt in range(5):
        try:
            stream_once(source, partial, expected_size)
            break
        except (OSError, urllib.error.URLError) as error:
            if attempt == 4:
                raise RuntimeError(f"bounded download failed for {label}") from error
            time.sleep(min(2**attempt, 16))
    if partial.stat().st_size != expected_size:
        raise RuntimeError(
            f"downloaded size mismatch for {label}: {partial.stat().st_size} != {expected_size}"
        )
    observed = sha256_file(partial)
    if observed != digest:
        raise RuntimeError(f"downloaded SHA-256 mismatch for {label}: {observed} != {digest}")
    os.replace(partial, blob)
    return blob


def safe_target(root: Path, relative_text: str) -> Path:
    relative = Path(relative_text)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"unsafe artifact path: {relative_text}")
    target = (root / relative).resolve()
    if root.resolve() not in target.parents:
        raise RuntimeError(f"artifact path escapes cache root: {relative_text}")
    return target


def materialize(blob: Path, target: Path, expected_size: int, digest: str) -> None:
    if verified(target, expected_size, digest):
        return
    if target.exists():
        raise RuntimeError(f"materialized artifact failed verification: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
    try:
        os.link(blob, temporary)
    except OSError:
        shutil.copyfile(blob, temporary)
    os.replace(temporary, target)


def install_bundle(root: Path, manifest_path: Path, verify_only: bool) -> dict[str, Any]:
    encoded = manifest_path.read_bytes()
    manifest = json.loads(encoded)
    manifest_digest = hashlib.sha256(encoded).hexdigest()
    bundle_root = root / "bundles" / f"{manifest['id']}-{manifest_digest[:16]}"
    installed = 0
    for component in manifest["components"]:
        for file_record in component["files"]:
            label = f"{manifest['id']}:{component['role']}:{file_record['path']}"
            size, digest, _ = checked_metadata(file_record, label)
            blob = acquire_blob(root, file_record, label, verify_only)
            target = safe_target(bundle_root, file_record["path"])
            materialize(blob, target, size, digest)
            installed += 1
    manifest_target = bundle_root / "xrt.bundle.json"
    if manifest_target.exists() and manifest_target.read_bytes() != encoded:
        raise RuntimeError(f"bundle manifest drift in cache: {manifest_target}")
    if not manifest_target.exists():
        temporary = manifest_target.with_suffix(".tmp")
        temporary.write_bytes(encoded)
        os.replace(temporary, manifest_target)
    return {
        "kind": "bundle",
        "id": manifest["id"],
        "manifest_sha256": manifest_digest,
        "files": installed,
        "path": str(bundle_root),
    }


def install_comparator_components(root: Path, lock: dict[str, Any], verify_only: bool) -> dict[str, Any]:
    component = lock["native_comparator_components"]
    target_root = root / "comparator-components" / component["revision"]
    for record in component["files"]:
        label = f"native-comparator:{record['path']}"
        size, digest, _ = checked_metadata(record, label)
        blob = acquire_blob(root, record, label, verify_only)
        materialize(blob, safe_target(target_root, record["path"]), size, digest)
    return {
        "kind": "comparator-components",
        "revision": component["revision"],
        "files": len(component["files"]),
        "path": str(target_root),
    }


def extract_zip(blob: Path, destination: Path) -> None:
    with zipfile.ZipFile(blob) as archive:
        for member in archive.infolist():
            relative = Path(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"unsafe path in comparator archive: {member.filename}")
            unix_mode = member.external_attr >> 16
            if (unix_mode & 0o170000) == 0o120000:
                raise RuntimeError(f"symlink rejected in comparator archive: {member.filename}")
            target = safe_target(destination, member.filename)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, CHUNK_BYTES)


def install_comparator_tool(root: Path, lock: dict[str, Any], verify_only: bool) -> dict[str, Any]:
    tool = lock["stable_diffusion_cpp"]
    target = root / "tools" / "stable-diffusion.cpp" / tool["release"]
    stamp_path = target / "xeno-reference-install.json"
    expected_stamp = {
        "schema_version": 1,
        "release": tool["release"],
        "commit": tool["commit"],
        "archives": {
            tool["asset"]: tool["sha256"],
            tool["cuda_runtime_asset"]["asset"]: tool["cuda_runtime_asset"]["sha256"],
        },
        "build": tool["build"],
    }
    expected_bytes = (json.dumps(expected_stamp, indent=2, sort_keys=True) + "\n").encode()
    if stamp_path.is_file() and stamp_path.read_bytes() == expected_bytes:
        executable = next(target.rglob("sd-cli.exe"), None)
        if executable is not None:
            return {"kind": "comparator-tool", "path": str(target), "executable": str(executable)}
    if target.exists():
        raise RuntimeError(f"comparator installation is incomplete or drifted: {target}")
    archives = []
    for metadata in [tool, tool["cuda_runtime_asset"]]:
        archives.append(acquire_blob(root, metadata, metadata["asset"], verify_only))
    staging_parent = target.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=target.name + ".", suffix=".staging", dir=staging_parent))
    try:
        for archive in archives:
            extract_zip(archive, staging)
        executable = next(staging.rglob("sd-cli.exe"), None)
        if executable is None:
            raise RuntimeError("stable-diffusion.cpp archive does not contain sd-cli.exe")
        (staging / "xeno-reference-install.json").write_bytes(expected_bytes)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    executable = next(target.rglob("sd-cli.exe"))
    return {"kind": "comparator-tool", "path": str(target), "executable": str(executable)}


def manifest_path(value: str) -> Path:
    candidate = Path(value)
    if not candidate.suffix:
        candidate = MANIFEST_DIR / f"{value}.json"
    elif not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    if not candidate.is_file():
        raise argparse.ArgumentTypeError(f"manifest not found: {candidate}")
    return candidate


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", action="append", type=manifest_path, default=[])
    parser.add_argument("--comparator-components", action="store_true")
    parser.add_argument("--comparator-tool", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if not (args.bundle or args.comparator_components or args.comparator_tool):
        parser.error("select --bundle, --comparator-components, and/or --comparator-tool")
    return args


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    root = cache_root()
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    results = []
    with cache_lock(root):
        for path in args.bundle:
            results.append(install_bundle(root, path, args.verify_only))
        if args.comparator_components:
            results.append(install_comparator_components(root, lock, args.verify_only))
        if args.comparator_tool:
            results.append(install_comparator_tool(root, lock, args.verify_only))
    print(json.dumps({"status": "ok", "cache": str(root), "artifacts": results}, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1:])
