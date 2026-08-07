#!/usr/bin/env python3
"""Verify and execute the pinned, non-production image-quality environment.

The metadata-only ``inspect`` command is safe on development machines. Model
downloads, CUDA initialization, and the complete PaddleOCR-VL pipeline are
restricted to explicit commands intended for the dedicated quality runner.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_LOCK_PATH = HERE / "quality-environment-lock.json"


class QualityEnvironmentError(RuntimeError):
    """Raised when the evaluator environment does not match its lock."""


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_lock(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QualityEnvironmentError(f"cannot read quality environment lock: {error}") from error
    if payload.get("schema_version") != 1:
        raise QualityEnvironmentError("unsupported quality environment lock schema")
    if payload.get("object") != "xeno.image.quality_environment_lock":
        raise QualityEnvironmentError("unexpected quality environment lock object")
    if payload.get("production_support") is not False:
        raise QualityEnvironmentError("quality environment must not claim production support")
    return payload


def installed_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def validate_versions(expected: dict[str, str], actual: dict[str, str | None]) -> None:
    mismatches = [
        f"{name}: expected {version}, got {actual.get(name) or 'not installed'}"
        for name, version in sorted(expected.items())
        if actual.get(name) != version
    ]
    if mismatches:
        raise QualityEnvironmentError("package lock mismatch; " + "; ".join(mismatches))


def validate_host(lock: dict[str, Any]) -> None:
    runner = lock["runner"]
    if platform.system() != runner["operating_system"]:
        raise QualityEnvironmentError(
            f"quality execution requires {runner['operating_system']}, got {platform.system()}"
        )
    if platform.machine().upper() != runner["architecture"]:
        raise QualityEnvironmentError(
            f"quality execution requires {runner['architecture']}, got {platform.machine()}"
        )
    expected_python = tuple(int(part) for part in runner["python"].split("."))
    if sys.version_info[:2] != expected_python:
        raise QualityEnvironmentError(
            f"quality execution requires Python {runner['python']}, "
            f"got {sys.version_info.major}.{sys.version_info.minor}"
        )


def validate_suite(lock: dict[str, Any]) -> dict[str, Any]:
    suite_record = lock["suite"]
    suite_path = (REPO_ROOT / suite_record["path"]).resolve()
    if REPO_ROOT not in suite_path.parents:
        raise QualityEnvironmentError("quality suite path escapes the repository")
    if sha256_file(suite_path) != suite_record["sha256"]:
        raise QualityEnvironmentError("quality suite digest does not match environment lock")
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    if suite.get("suite_version") != suite_record["version"]:
        raise QualityEnvironmentError("quality suite version does not match environment lock")
    for role, pin in lock["models"].items():
        suite_role = "structural_identity" if role == "structural_identity" else role
        evaluator = suite["evaluators"][suite_role]
        if evaluator.get("model") != pin["repository"] or evaluator.get("revision") != pin["revision"]:
            raise QualityEnvironmentError(f"quality suite evaluator pin drift for {role}")
    return suite


def environment_evidence(lock: dict[str, Any], *, require_host: bool) -> dict[str, Any]:
    suite = validate_suite(lock)
    actual = installed_versions(lock["packages"])
    if require_host:
        validate_host(lock)
        validate_versions(lock["packages"], actual)
    return {
        "schema_version": 1,
        "object": "xeno.image.quality_environment_evidence",
        "status": "passed" if require_host else "metadata_valid",
        "production_support": False,
        "host": {
            "architecture": platform.machine(),
            "operating_system": platform.system(),
            "python": platform.python_version(),
        },
        "packages": actual,
        "suite": {
            "version": suite["suite_version"],
            "sha256": lock["suite"]["sha256"],
        },
        "models": lock["models"],
        "ocr_pipeline": lock["ocr_pipeline"],
    }


def snapshot_manifest(path: Path) -> dict[str, Any]:
    records = []
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        records.append(
            {
                "path": file_path.relative_to(path).as_posix(),
                "size": file_path.stat().st_size,
                "sha256": sha256_file(file_path),
            }
        )
    if not records:
        raise QualityEnvironmentError(f"evaluator snapshot is empty: {path}")
    return {
        "files": records,
        "manifest_sha256": hashlib.sha256(canonical_json_bytes(records)).hexdigest(),
    }


def materialize_models(
    lock: dict[str, Any],
    cache_dir: Path,
    allow_download: bool,
    roles: set[str] | None = None,
) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    available_roles = set(lock["models"])
    selected_roles = roles or available_roles
    unknown_roles = selected_roles - available_roles
    if unknown_roles:
        raise QualityEnvironmentError(
            "unknown evaluator roles: " + ", ".join(sorted(unknown_roles))
        )
    evidence: dict[str, Any] = {}
    for role, pin in sorted(lock["models"].items()):
        if role not in selected_roles:
            continue
        try:
            snapshot = Path(
                snapshot_download(
                    repo_id=pin["repository"],
                    revision=pin["revision"],
                    cache_dir=cache_dir,
                    local_files_only=not allow_download,
                )
            ).resolve()
        except Exception as error:
            mode = "download" if allow_download else "offline cache lookup"
            raise QualityEnvironmentError(f"{mode} failed for {role}: {error}") from error
        manifest = snapshot_manifest(snapshot)
        evidence[role] = {
            "repository": pin["repository"],
            "revision": pin["revision"],
            "snapshot": str(snapshot),
            "files": len(manifest["files"]),
            "manifest_sha256": manifest["manifest_sha256"],
            "file_manifest": manifest["files"],
        }
    return evidence


def serialize_ocr_result(result: Any) -> Any:
    value = getattr(result, "json", None)
    if callable(value):
        value = value()
    if value is None:
        converter = getattr(result, "to_dict", None)
        if callable(converter):
            value = converter()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
    if value is not None:
        try:
            json.dumps(value, ensure_ascii=False)
            return value
        except TypeError:
            pass
    return {"type": type(result).__name__, "repr": repr(result)[:2000]}


def execute_ocr_smoke(
    lock: dict[str, Any], output_dir: Path, cache_dir: Path
) -> dict[str, Any]:
    import paddle
    from huggingface_hub import snapshot_download
    from PIL import Image, ImageDraw, ImageFont
    from paddleocr import PaddleOCRVL

    pipeline_lock = lock["ocr_pipeline"]
    if pipeline_lock["device"].startswith("gpu"):
        if not paddle.is_compiled_with_cuda():
            raise QualityEnvironmentError("paddlepaddle-gpu is not CUDA-enabled")
        if paddle.device.cuda.device_count() < 1:
            raise QualityEnvironmentError("no CUDA device is visible to PaddlePaddle")
    os.environ["PADDLE_PDX_MODEL_SOURCE"] = pipeline_lock["model_source"]
    output_dir.mkdir(parents=True, exist_ok=True)
    smoke_image = output_dir / "ocr-smoke-input.png"
    image = Image.new("RGB", (1024, 1024), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default(size=96)
    except TypeError:
        font = ImageFont.load_default()
    draw.text((96, 430), pipeline_lock["smoke_text"], fill="black", font=font)
    image.save(smoke_image, format="PNG")

    ocr_pin = lock["models"]["ocr"]
    try:
        ocr_snapshot = Path(
            snapshot_download(
                repo_id=ocr_pin["repository"],
                revision=ocr_pin["revision"],
                cache_dir=cache_dir,
                local_files_only=True,
            )
        ).resolve()
    except Exception as error:
        raise QualityEnvironmentError(f"pinned OCR snapshot is not available offline: {error}") from error

    try:
        pipeline = PaddleOCRVL(
            pipeline_version=pipeline_lock["pipeline_version"],
            device=pipeline_lock["device"],
            use_layout_detection=pipeline_lock["use_layout_detection"],
            vl_rec_model_dir=str(ocr_snapshot),
        )
        results = list(pipeline.predict(str(smoke_image)))
    except Exception as error:
        raise QualityEnvironmentError(f"complete PaddleOCR-VL smoke failed: {error}") from error
    if not results:
        raise QualityEnvironmentError("complete PaddleOCR-VL smoke returned no results")
    return {
        "executed": True,
        "input": {
            "path": str(smoke_image),
            "sha256": sha256_file(smoke_image),
            "expected_text": pipeline_lock["smoke_text"],
        },
        "pipeline": pipeline_lock,
        "results": [serialize_ocr_result(result) for result in results],
    }


def write_output(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_json_bytes(payload)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="validate checked-in pins without imports or downloads")
    inspect_parser.add_argument("--output", type=Path, required=True)

    materialize_parser = subparsers.add_parser(
        "materialize", help="resolve exact evaluator revisions and export file manifests"
    )
    materialize_parser.add_argument("--cache-dir", type=Path, required=True)
    materialize_parser.add_argument("--allow-download", action="store_true")
    materialize_parser.add_argument("--roles", nargs="+", default=None)
    materialize_parser.add_argument("--output", type=Path, required=True)

    smoke_parser = subparsers.add_parser(
        "ocr-smoke", help="execute the complete pinned PaddleOCR-VL pipeline on CUDA"
    )
    smoke_parser.add_argument("--cache-dir", type=Path, required=True)
    smoke_parser.add_argument("--output-dir", type=Path, required=True)
    smoke_parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        lock = load_lock(args.lock.resolve())
        if args.command == "inspect":
            payload = environment_evidence(lock, require_host=False)
        elif args.command == "materialize":
            payload = environment_evidence(lock, require_host=True)
            payload["model_snapshots"] = materialize_models(
                lock,
                args.cache_dir.resolve(),
                args.allow_download,
                set(args.roles) if args.roles else None,
            )
            payload["status"] = "passed"
        else:
            payload = environment_evidence(lock, require_host=True)
            payload["ocr_smoke"] = execute_ocr_smoke(
                lock, args.output_dir.resolve(), args.cache_dir.resolve()
            )
            payload["status"] = "passed"
        write_output(args.output.resolve(), payload)
    except (QualityEnvironmentError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"quality environment error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
