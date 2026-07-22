#!/usr/bin/env python3
"""Prove comparator SafeTensors repacks preserve official BF16 tensor bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
LOCK_PATH = HERE / "phase0-lock.json"
MANIFEST_PATH = HERE / "manifests" / "qwen-image-2512-bf16.json"
OUTPUT_PATH = REPO_ROOT / "benchmark-results" / "image" / "phase0-2026-07-21" / "comparator-component-equivalence.json"
CHUNK_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class TensorRef:
    name: str
    dtype: str
    shape: tuple[int, ...]
    path: Path
    offset: int
    size_bytes: int


def cache_root() -> Path:
    configured = os.environ.get("XRT_IMAGE_REFERENCE_CACHE")
    return Path(configured).expanduser().resolve() if configured else REPO_ROOT / ".codex-tmp" / "image-reference"


def canonical_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def atomic_write(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def safetensor_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise RuntimeError(f"invalid SafeTensors prefix: {path.name}")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size <= 2 or header_size > 256 * 1024 * 1024:
            raise RuntimeError(f"unsafe SafeTensors header size in {path.name}: {header_size}")
        header = handle.read(header_size)
        if len(header) != header_size:
            raise RuntimeError(f"truncated SafeTensors header: {path.name}")
    payload = json.loads(header)
    payload["__xeno_data_start__"] = 8 + header_size
    return payload


def tensor_index(paths: list[Path]) -> dict[str, TensorRef]:
    tensors: dict[str, TensorRef] = {}
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"missing SafeTensors artifact: {path}")
        header = safetensor_header(path)
        data_start = header.pop("__xeno_data_start__")
        header.pop("__metadata__", None)
        for name, metadata in header.items():
            if name in tensors:
                raise RuntimeError(f"duplicate tensor across shards: {name}")
            offsets = metadata.get("data_offsets")
            if not isinstance(offsets, list) or len(offsets) != 2:
                raise RuntimeError(f"invalid data offsets for {name}")
            start, end = offsets
            if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
                raise RuntimeError(f"invalid tensor byte range for {name}")
            absolute = data_start + start
            if data_start + end > path.stat().st_size:
                raise RuntimeError(f"tensor range escapes SafeTensors file for {name}")
            tensors[name] = TensorRef(
                name=name,
                dtype=metadata["dtype"],
                shape=tuple(metadata["shape"]),
                path=path,
                offset=absolute,
                size_bytes=end - start,
            )
    return tensors


def normalized_names(name: str) -> set[str]:
    names = {name}
    prefixes = (
        "model.",
        "text_encoder.",
        "vae.",
        "first_stage_model.",
        "diffusion_model.",
    )
    changed = True
    while changed:
        changed = False
        for candidate in list(names):
            for prefix in prefixes:
                if candidate.startswith(prefix):
                    stripped = candidate[len(prefix) :]
                    if stripped not in names:
                        names.add(stripped)
                        changed = True
    return names


def _translate_vae_residual_tail(tail: str) -> str:
    translations = (
        ("norm1", "residual.0"),
        ("conv1", "residual.2"),
        ("norm2", "residual.3"),
        ("conv2", "residual.6"),
        ("conv_shortcut", "shortcut"),
    )
    for official, comparator in translations:
        if tail == official or tail.startswith(official + "."):
            return comparator + tail[len(official) :]
    return tail


def vae_comparator_name(name: str) -> str | None:
    """Translate official Diffusers VAE names to the pinned comparator layout."""
    direct = {
        "quant_conv.weight": "conv1.weight",
        "quant_conv.bias": "conv1.bias",
        "post_quant_conv.weight": "conv2.weight",
        "post_quant_conv.bias": "conv2.bias",
        "encoder.conv_in.weight": "encoder.conv1.weight",
        "encoder.conv_in.bias": "encoder.conv1.bias",
        "encoder.norm_out.gamma": "encoder.head.0.gamma",
        "encoder.conv_out.weight": "encoder.head.2.weight",
        "encoder.conv_out.bias": "encoder.head.2.bias",
        "decoder.conv_in.weight": "decoder.conv1.weight",
        "decoder.conv_in.bias": "decoder.conv1.bias",
        "decoder.norm_out.gamma": "decoder.head.0.gamma",
        "decoder.conv_out.weight": "decoder.head.2.weight",
        "decoder.conv_out.bias": "decoder.head.2.bias",
    }
    if name in direct:
        return direct[name]

    parts = name.split(".")
    if len(parts) >= 4 and parts[0:2] == ["encoder", "down_blocks"]:
        index = parts[2]
        tail = _translate_vae_residual_tail(".".join(parts[3:]))
        return f"encoder.downsamples.{index}.{tail}"

    if len(parts) >= 6 and parts[0:2] == ["decoder", "up_blocks"]:
        block = int(parts[2])
        if parts[3] == "resnets":
            flattened = block * 4 + int(parts[4])
            tail = _translate_vae_residual_tail(".".join(parts[5:]))
            return f"decoder.upsamples.{flattened}.{tail}"
        if parts[3:5] == ["upsamplers", "0"]:
            flattened = block * 4 + 3
            return f"decoder.upsamples.{flattened}." + ".".join(parts[5:])

    if len(parts) >= 6 and parts[1:3] == ["mid_block", "resnets"]:
        side = parts[0]
        middle = 0 if parts[3] == "0" else 2
        tail = _translate_vae_residual_tail(".".join(parts[4:]))
        return f"{side}.middle.{middle}.{tail}"

    if len(parts) >= 6 and parts[1:4] == ["mid_block", "attentions", "0"]:
        side = parts[0]
        return f"{side}.middle.1." + ".".join(parts[4:])
    return None


def map_tensors(
    role: str, official: dict[str, TensorRef], comparator: dict[str, TensorRef]
) -> list[tuple[TensorRef, TensorRef, str]]:
    mappings: list[tuple[TensorRef, TensorRef, str]] = []
    used_comparator: set[str] = set()
    unresolved = []
    normalized_comparator: dict[str, list[str]] = {}
    for name in comparator:
        for normalized in normalized_names(name):
            normalized_comparator.setdefault(normalized, []).append(name)
    for official_name, official_ref in sorted(official.items()):
        if official_name in comparator:
            candidate_names = [official_name]
            rule = "exact_name"
        else:
            candidates = {
                candidate
                for normalized in normalized_names(official_name)
                for candidate in normalized_comparator.get(normalized, [])
            }
            candidate_names = sorted(candidates)
            rule = "unique_known_prefix_normalization"
            if role == "vae":
                translated = {
                    translated
                    for normalized in normalized_names(official_name)
                    if (translated := vae_comparator_name(normalized)) is not None
                }
                translated_candidates = {
                    candidate
                    for normalized in translated
                    for candidate in normalized_comparator.get(normalized, [])
                }
                if translated_candidates:
                    candidate_names = sorted(set(candidate_names) | translated_candidates)
                    rule = "explicit_vae_layout_translation"
        compatible = [
            name
            for name in candidate_names
            if name not in used_comparator
            and comparator[name].dtype == official_ref.dtype
            and comparator[name].shape == official_ref.shape
            and comparator[name].size_bytes == official_ref.size_bytes
        ]
        if len(compatible) != 1:
            unresolved.append(
                {
                    "official": official_name,
                    "candidate_count": len(compatible),
                    "candidate_names": compatible[:10],
                }
            )
            continue
        comparator_ref = comparator[compatible[0]]
        used_comparator.add(comparator_ref.name)
        mappings.append((official_ref, comparator_ref, rule))
    extra = sorted(set(comparator) - used_comparator)
    if unresolved or extra:
        raise RuntimeError(
            "SafeTensors tensor-name mapping is incomplete: "
            + json.dumps({"unresolved": unresolved[:20], "extra": extra[:20]}, sort_keys=True)
        )
    return mappings


def hash_region(handle: BinaryIO, offset: int, size: int) -> str:
    handle.seek(offset)
    digest = hashlib.sha256()
    remaining = size
    while remaining:
        chunk = handle.read(min(CHUNK_BYTES, remaining))
        if not chunk:
            raise RuntimeError("unexpected EOF while hashing tensor data")
        digest.update(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


def compare_component(
    role: str, official_paths: list[Path], comparator_path: Path
) -> dict[str, Any]:
    official = tensor_index(official_paths)
    comparator = tensor_index([comparator_path])
    mappings = map_tensors(role, official, comparator)
    handles: dict[Path, BinaryIO] = {}
    verified = []
    started = time.monotonic()
    try:
        for official_ref, comparator_ref, rule in mappings:
            if official_ref.path not in handles:
                handles[official_ref.path] = official_ref.path.open("rb")
            if comparator_ref.path not in handles:
                handles[comparator_ref.path] = comparator_ref.path.open("rb")
            official_handle = handles[official_ref.path]
            comparator_handle = handles[comparator_ref.path]
            official_hash = hash_region(official_handle, official_ref.offset, official_ref.size_bytes)
            comparator_hash = hash_region(
                comparator_handle, comparator_ref.offset, comparator_ref.size_bytes
            )
            if official_hash != comparator_hash:
                raise RuntimeError(
                    f"{role} tensor data mismatch: {official_ref.name} != {comparator_ref.name}"
                )
            verified.append(
                {
                    "official_name": official_ref.name,
                    "comparator_name": comparator_ref.name,
                    "dtype": official_ref.dtype,
                    "shape": list(official_ref.shape),
                    "size_bytes": official_ref.size_bytes,
                    "sha256": official_hash,
                    "mapping_rule": rule,
                }
            )
    finally:
        for handle in handles.values():
            handle.close()
    aggregate = hashlib.sha256(canonical_bytes(verified)).hexdigest()
    sample_indices = sorted(
        {0, len(verified) // 4, len(verified) // 2, (3 * len(verified)) // 4, len(verified) - 1}
    )
    return {
        "role": role,
        "status": "byte_exact",
        "verified_tensors": len(verified),
        "verified_tensor_bytes": sum(item["size_bytes"] for item in verified),
        "mapping_rules": sorted({item["mapping_rule"] for item in verified}),
        "aggregate_tensor_manifest_sha256": aggregate,
        "representative_tensors": [verified[index] for index in sample_indices],
        "elapsed_seconds": time.monotonic() - started,
    }


def find_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    manifest_bytes = MANIFEST_PATH.read_bytes()
    manifest = json.loads(manifest_bytes)
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    bundle = cache_root() / "bundles" / f"{manifest['id']}-{digest[:16]}"
    comparator_lock = lock["native_comparator_components"]
    comparator_root = cache_root() / "comparator-components" / comparator_lock["revision"]
    official: dict[str, list[Path]] = {}
    for component in manifest["components"]:
        if component["role"] not in {"text_encoder", "vae"}:
            continue
        official[component["role"]] = [
            bundle / record["path"]
            for record in component["files"]
            if record["path"].endswith(".safetensors")
        ]
    comparator = {}
    for record in comparator_lock["files"]:
        path = comparator_root / record["path"]
        if "/text_encoders/" in record["path"]:
            comparator["text_encoder"] = path
        elif "/vae/" in record["path"]:
            comparator["vae"] = path
    if set(official) != {"text_encoder", "vae"} or set(comparator) != set(official):
        raise RuntimeError("component equivalence inputs are incomplete")
    return {
        "lock": lock,
        "manifest": manifest,
        "manifest_sha256": digest,
        "official": official,
        "comparator": comparator,
    }, comparator_lock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    inputs, comparator_lock = find_inputs()
    started = time.monotonic()
    components = [
        compare_component(role, inputs["official"][role], inputs["comparator"][role])
        for role in ("text_encoder", "vae")
    ]
    report = {
        "schema_version": 1,
        "status": "byte_exact",
        "policy": (
            "Every SafeTensors tensor is matched by exact name, an explicit unique prefix normalization, "
            "or the pinned Diffusers-to-comparator VAE layout translation; "
            "dtype, shape, byte length, and the full raw tensor payload SHA-256 must match."
        ),
        "official": {
            "repository": "Qwen/Qwen-Image-2512",
            "revision": inputs["manifest"]["source_revisions"]["Qwen/Qwen-Image-2512"],
            "manifest_sha256": inputs["manifest_sha256"],
        },
        "comparator": {
            "repository": comparator_lock["repository"],
            "revision": comparator_lock["revision"],
            "files": [
                {
                    "path": record["path"],
                    "size_bytes": record["size_bytes"],
                    "sha256": record["sha256"],
                }
                for record in comparator_lock["files"]
            ],
        },
        "components": components,
        "verified_tensors": sum(component["verified_tensors"] for component in components),
        "verified_tensor_bytes": sum(component["verified_tensor_bytes"] for component in components),
        "elapsed_seconds": time.monotonic() - started,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    body = canonical_bytes(report)
    if args.write:
        atomic_write(OUTPUT_PATH, body)
    print(
        json.dumps(
            {
                "status": "ok",
                "equivalence": report["status"],
                "tensors": report["verified_tensors"],
                "bytes": report["verified_tensor_bytes"],
                "output": str(OUTPUT_PATH.relative_to(REPO_ROOT)) if args.write else None,
                "sha256": hashlib.sha256(body).hexdigest(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
