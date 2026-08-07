#!/usr/bin/env python3
"""Verify and materialize immutable Phase 0 Qwen Image metadata.

This script intentionally uses only the Python standard library. It is safe to
run before installing the heavyweight reference environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
MANIFEST_DIR = HERE / "manifests"
LOCK_PATH = HERE / "phase0-lock.json"
QUALITY_SUITE_PATH = HERE.parents[2] / "tests" / "common" / "image-quality-suite.json"
REPO_ROOT = HERE.parents[2]
OPENAI_FIXTURE_MANIFEST_PATH = (
    REPO_ROOT / "tests" / "fixtures" / "openai" / "images" / "fixture-manifest.json"
)
USER_AGENT = "xeno-rt-qwen-image-phase0/1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
BLOB_SHA256_CACHE: dict[str, str] = {}


@dataclass(frozen=True)
class RepoPin:
    repo: str
    revision: str
    license: str
    capability: str


GENERATION = RepoPin(
    repo="Qwen/Qwen-Image-2512",
    revision="25468b98e3276ca6700de15c6628e51b7de54a26",
    license="Apache-2.0",
    capability="image.generate",
)
EDIT = RepoPin(
    repo="Qwen/Qwen-Image-Edit-2511",
    revision="6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9",
    license="Apache-2.0",
    capability="image.edit",
)
GENERATION_GGUF = RepoPin(
    repo="unsloth/Qwen-Image-2512-GGUF",
    revision="1626d7531f84b4d2ea1cd6d2e69f41ec027dd354",
    license="Apache-2.0",
    capability="image.generate",
)
EDIT_GGUF = RepoPin(
    repo="unsloth/Qwen-Image-Edit-2511-GGUF",
    revision="0d33d9692b4b26212297240d87b0d4719aa4fd06",
    license="Apache-2.0",
    capability="image.edit",
)
LIGHTNING = RepoPin(
    repo="lightx2v/Qwen-Image-2512-Lightning",
    revision="a52649c9d0f6e1a248bff13f0df33bb8a2abdb52",
    license="Apache-2.0",
    capability="image.generate",
)
LIGHTNING_FILE = "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors"
LIGHTNING_SIZE = 849_608_296
LIGHTNING_SHA256 = "de0d236e54ecf2c43b32447d13478c6eae0d361b1fed48c69675b084fa240d87"
COMPARATOR_COMPONENTS = RepoPin(
    repo="Comfy-Org/Qwen-Image_ComfyUI",
    revision="46839d338df81ce625d5fae27d7e370314c0fbc9",
    license="Apache-2.0 upstream components",
    capability="benchmark.native_comparator",
)
COMPARATOR_COMPONENT_PATHS = {
    "split_files/text_encoders/qwen_2.5_vl_7b.safetensors": {
        "size_bytes": 16_584_415_576,
        "sha256": "cfafd739459bc86257397259f612a9aee88e5b98e85b5c0d0d1717e898b3463a",
    },
    "split_files/vae/qwen_image_vae.safetensors": {
        "size_bytes": 253_806_246,
        "sha256": "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f",
    },
}

GGUF_TIERS = {
    "Q8_0": {
        "generation": (
            "qwen-image-2512-Q8_0.gguf",
            21_761_817_120,
            "e285a0692582acf09bb4086d9b120eb0e357c4386565d169a033cb968c6fa9a5",
        ),
        "edit": (
            "qwen-image-edit-2511-Q8_0.gguf",
            21_761_817_184,
            "ab4f0622fb002fccaaa679a2ecce6fd1b3190d8ea28a5b7b2b17b8669bc24afa",
        ),
    },
    "Q6_K": {
        "generation": (
            "qwen-image-2512-Q6_K.gguf",
            16_824_990_240,
            "58c7fcea3d29eaee4dbe77041764a7de3ffe449f9758c8bc459f0b22a09536cf",
        ),
        "edit": (
            "qwen-image-edit-2511-Q6_K.gguf",
            16_852_417_120,
            "fdc28e5b8f7d9cfe0399fd1700c375f25f000fc4159bbdb0d4a809ae898eb759",
        ),
    },
    "Q5_K_M": {
        "generation": (
            "qwen-image-2512-Q5_K_M.gguf",
            15_000_074_784,
            "e9ea2c513cf25645829fcccbd0882e821b858f4fabfb48ae1b5f103da68ecd0f",
        ),
        "edit": (
            "qwen-image-edit-2511-Q5_K_M.gguf",
            15_027_501_664,
            "c257def934d25562e1dcb7e8710eb5457a23c1ae7218f7a5ea3174e773421d29",
        ),
    },
    "Q4_K_M": {
        "generation": (
            "qwen-image-2512-Q4_K_M.gguf",
            13_244_758_560,
            "b2a5f6249eb58ee10c9e2ce8cb1114b89897db23de2fdf7dc49140800aa928fc",
        ),
        "edit": (
            "qwen-image-edit-2511-Q4_K_M.gguf",
            13_244_758_624,
            "8677bac90627adbbc11efab87b1870e701c4eb3689ee865a3de8ab81b705a723",
        ),
    },
}

PYTHON_PACKAGES = {
    "accelerate": "1.14.0",
    "diffusers": "0.39.0",
    "ftfy": "6.3.1",
    "huggingface-hub": "1.24.0",
    "numpy": "2.3.5",
    "openai": "2.46.0",
    "pillow": "12.3.0",
    "protobuf": "7.35.1",
    "safetensors": "0.8.0",
    "sentencepiece": "0.2.2",
    "torch": "2.13.0",
    "torchvision": "0.28.0",
    "transformers": "5.14.1",
}

EVALUATOR_PACKAGES = {
    "open-clip-torch": "3.3.0",
    "paddleocr": "3.7.0",
    "scikit-image": "0.26.0",
    "scipy": "1.17.1",
}

EVALUATOR_MODELS = {
    "prompt_alignment": RepoPin(
        repo="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        revision="1c2b8495b28150b8a4922ee1c8edee224c284c0c",
        license="MIT",
        capability="quality.prompt_alignment",
    ),
    "ocr": RepoPin(
        repo="PaddlePaddle/PaddleOCR-VL-1.6",
        revision="66317acc4c9fc17bd154591ce650735cd2855f3e",
        license="Apache-2.0",
        capability="quality.ocr",
    ),
    "structural_identity": RepoPin(
        repo="facebook/dinov2-large",
        revision="47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
        license="Apache-2.0",
        capability="quality.structural_identity",
    ),
}

GIT_PINS = {
    "diffusers": {
        "url": "https://github.com/huggingface/diffusers.git",
        "commit": "2919c50962c375e32b9fa40ae5fad50cd3251332",
    },
    "transformers": {
        "url": "https://github.com/huggingface/transformers.git",
        "commit": "9ed46fb37cf4c7f885677ad194d2797265e89186",
    },
    "safetensors": {
        "url": "https://github.com/huggingface/safetensors.git",
        "commit": "6eb4dc9a28ebce297606e0f4836bbf28839cacef",
    },
    "openai-python": {
        "url": "https://github.com/openai/openai-python.git",
        "commit": "d4dceb221b9a92c55c232d5b330ae89beb539415",
    },
    "openai-node": {
        "url": "https://github.com/openai/openai-node.git",
        "commit": "39a15b412fc129df15339ebd6e3e6547854aa81f",
    },
}

OPENAI_CLIENTS = {"python": "2.46.0", "node": "6.48.0"}

OPENAI_SERVER_OPENAPI = {
    "version": "2.3.0",
    "observed_at": "2026-07-22",
    "generation_endpoint": "https://api.openai.com/v1/images/generations",
    "edit_endpoint": "https://api.openai.com/v1/images/edits",
}

OPENAI_SCHEMA_SOURCES = {
    "python_image_generate_params": {
        "url": "https://raw.githubusercontent.com/openai/openai-python/d4dceb221b9a92c55c232d5b330ae89beb539415/src/openai/types/image_generate_params.py",
        "sha256": "961e9ed304b17f987d39b1b56f1d2291cfe3668a4cf0e7ded05d666d220ca2a7",
    },
    "python_image_edit_params": {
        "url": "https://raw.githubusercontent.com/openai/openai-python/d4dceb221b9a92c55c232d5b330ae89beb539415/src/openai/types/image_edit_params.py",
        "sha256": "99ecccc97a1f156849ed4f8f0090ddc24a80d50c5be9ffc853e3944627bdfa22",
    },
    "python_image_generation_partial": {
        "url": "https://raw.githubusercontent.com/openai/openai-python/d4dceb221b9a92c55c232d5b330ae89beb539415/src/openai/types/image_gen_partial_image_event.py",
        "sha256": "e55261c537fa6608d515b427ef688e9289481c23006b82c4c4c85786adf9351c",
    },
    "python_image_generation_completed": {
        "url": "https://raw.githubusercontent.com/openai/openai-python/d4dceb221b9a92c55c232d5b330ae89beb539415/src/openai/types/image_gen_completed_event.py",
        "sha256": "ae98e7a1c250e6299846b1c7c44cf9c83cd6a698bd5bab31c47617d5d29a9908",
    },
    "python_image_edit_partial": {
        "url": "https://raw.githubusercontent.com/openai/openai-python/d4dceb221b9a92c55c232d5b330ae89beb539415/src/openai/types/image_edit_partial_image_event.py",
        "sha256": "b49479f7e960dd041fa0336c22d06849c00c852f8874e1c3a27eb9e6f473d020",
    },
    "python_image_edit_completed": {
        "url": "https://raw.githubusercontent.com/openai/openai-python/d4dceb221b9a92c55c232d5b330ae89beb539415/src/openai/types/image_edit_completed_event.py",
        "sha256": "bbc463f5e5b80bb86d3b48e5e7d38fd2ee0228484ed137132f576a2c63762505",
    },
    "python_images_response": {
        "url": "https://raw.githubusercontent.com/openai/openai-python/d4dceb221b9a92c55c232d5b330ae89beb539415/src/openai/types/images_response.py",
        "sha256": "1b2512a716945c97ebd2a1dd69b5dd7989f5e4c5efd6f0f64b388848ed3541d8",
    },
    "python_image_resource": {
        "url": "https://raw.githubusercontent.com/openai/openai-python/d4dceb221b9a92c55c232d5b330ae89beb539415/src/openai/resources/images.py",
        "sha256": "f4d2e79b4de0f62137d38c4171603efc23fd1ca02e70483314028c7f7146544d",
    },
    "node_image_resource": {
        "url": "https://raw.githubusercontent.com/openai/openai-node/39a15b412fc129df15339ebd6e3e6547854aa81f/src/resources/images.ts",
        "sha256": "baaceca31eaf67c46113766fdc3d3fb4996b8bf7b979ba31a236effcbf4a4aac",
    },
}

STABLE_DIFFUSION_CPP = {
    "repository": "https://github.com/leejet/stable-diffusion.cpp",
    "release": "master-782-b290693",
    "commit": "b2906939774dc73453467215c80390404d0a2701",
    "asset": "sd-master-b290693-bin-win-cuda12-x64.zip",
    "size_bytes": 361_929_826,
    "sha256": "bc7aa2f6d471b324bfbc76f108a5cb3b76de29cb892d12609647e11b041f8a02",
    "url": (
        "https://github.com/leejet/stable-diffusion.cpp/releases/download/"
        "master-782-b290693/sd-master-b290693-bin-win-cuda12-x64.zip"
    ),
    "cuda_runtime_asset": {
        "asset": "cudart-sd-bin-win-cu12-x64.zip",
        "size_bytes": 563_452_046,
        "sha256": "fe20366827d357c00797eebb58244dddab7fd9a348d70090c3871004c320f38d",
        "url": (
            "https://github.com/leejet/stable-diffusion.cpp/releases/download/"
            "master-782-b290693/cudart-sd-bin-win-cu12-x64.zip"
        ),
    },
    "build": {
        "runner": "windows-2022",
        "cuda_toolkit": "12.8.1",
        "generator": "Ninja",
        "configuration": "Release",
        "cmake_defines": [
            "-DCMAKE_CXX_FLAGS=/bigobj",
            "-DCMAKE_C_COMPILER=cl.exe",
            "-DCMAKE_CXX_COMPILER=cl.exe",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DSD_CUDA=ON",
            "-DSD_BUILD_SHARED_LIBS=ON",
            "-DCMAKE_CUDA_ARCHITECTURES=61;70;75;80;86;89;90;100;120",
            "-DCMAKE_CUDA_FLAGS=-Xcudafe --diag_suppress=177 -Xcudafe --diag_suppress=550",
            "-DGGML_NATIVE=OFF",
            "-DSD_BUILD_SHARED_GGML_LIB=ON",
            "-DGGML_BACKEND_DL=ON",
            "-DGGML_CPU_ALL_VARIANTS=ON",
        ],
        "workflow": (
            "https://raw.githubusercontent.com/leejet/stable-diffusion.cpp/"
            "b2906939774dc73453467215c80390404d0a2701/.github/workflows/build.yml"
        ),
        "workflow_sha256": "edf451a935b073df0c7ec86a136da687ca76020963913dd5830a56adcfa19c58",
    },
}


def request_bytes(url: str) -> bytes:
    for attempt in range(5):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == 4:
                raise
            retry_after = error.headers.get("Retry-After")
            delay = int(retry_after) if retry_after and retry_after.isdigit() else 2 ** attempt
            time.sleep(min(max(delay, 1), 30))
    raise AssertionError("bounded HTTP retry loop exited unexpectedly")


def request_json(url: str) -> Any:
    return json.loads(request_bytes(url))


def hf_api(pin: RepoPin) -> dict[str, Any]:
    repo = urllib.parse.quote(pin.repo, safe="/")
    revision = urllib.parse.quote(pin.revision, safe="")
    url = f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true"
    payload = request_json(url)
    if payload.get("sha") != pin.revision:
        raise RuntimeError(
            f"{pin.repo}: expected revision {pin.revision}, got {payload.get('sha')}"
        )
    return payload


def immutable_hf_url(repo: str, revision: str, path: str) -> str:
    encoded_path = urllib.parse.quote(path, safe="/")
    return f"https://huggingface.co/{repo}/resolve/{revision}/{encoded_path}"


def hf_files(
    pin: RepoPin,
    *,
    include_paths: set[str] | None = None,
    exclude_paths: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    payload = hf_api(pin)
    files: dict[str, dict[str, Any]] = {}
    for sibling in payload.get("siblings", []):
        path = sibling["rfilename"]
        if include_paths is not None and path not in include_paths:
            continue
        if exclude_paths is not None and path in exclude_paths:
            continue
        lfs = sibling.get("lfs") or {}
        digest = lfs.get("sha256")
        if digest is None:
            blob_id = sibling.get("blobId")
            digest = BLOB_SHA256_CACHE.get(blob_id) if blob_id else None
            if digest is None:
                body = request_bytes(immutable_hf_url(pin.repo, pin.revision, path))
                digest = hashlib.sha256(body).hexdigest()
                if len(body) != sibling["size"]:
                    raise RuntimeError(
                        f"{pin.repo}/{path}: API size {sibling['size']} != body {len(body)}"
                    )
                if blob_id:
                    BLOB_SHA256_CACHE[blob_id] = digest
        if not SHA256_RE.fullmatch(digest):
            raise RuntimeError(f"{pin.repo}/{path}: invalid SHA-256 {digest!r}")
        files[path] = {
            "size_bytes": sibling["size"],
            "sha256": digest,
            "source": immutable_hf_url(pin.repo, pin.revision, path),
        }
    return files


def component_for_prefix(
    role: str,
    fmt: str,
    files: dict[str, dict[str, Any]],
    prefix: str,
    *,
    optional: bool = False,
) -> dict[str, Any]:
    selected = []
    for source_path, metadata in sorted(files.items()):
        if prefix and not source_path.startswith(prefix + "/"):
            continue
        if not prefix and source_path != "model_index.json":
            continue
        selected.append({"path": source_path, **metadata})
    if not selected:
        raise RuntimeError(f"no files found for component {role!r} prefix {prefix!r}")
    component: dict[str, Any] = {"role": role, "format": fmt, "files": selected}
    if optional:
        component["optional"] = True
    return component


def official_manifest(pin: RepoPin, files: dict[str, dict[str, Any]]) -> dict[str, Any]:
    is_edit = pin is EDIT
    components = [
        component_for_prefix("pipeline", "json", files, "", optional=True),
        component_for_prefix("scheduler", "json", files, "scheduler"),
        component_for_prefix("text_encoder", "safetensors", files, "text_encoder"),
        component_for_prefix("tokenizer", "huggingface-json", files, "tokenizer"),
    ]
    if is_edit:
        components.append(
            component_for_prefix("processor", "huggingface-json", files, "processor")
        )
    components.extend(
        [
            component_for_prefix("transformer", "safetensors", files, "transformer"),
            component_for_prefix("vae", "safetensors", files, "vae"),
        ]
    )
    return base_manifest(
        model_id="qwen-image-edit-2511-bf16" if is_edit else "qwen-image-2512-bf16",
        family="qwen-image-edit" if is_edit else "qwen-image",
        revision=pin.revision,
        capability=pin.capability,
        quantization="BF16",
        license_pin=pin,
        source_revisions={pin.repo: pin.revision},
        components=components,
    )


def gguf_manifest(
    *,
    tier: str,
    kind: str,
    official_pin: RepoPin,
    gguf_pin: RepoPin,
    official_files: dict[str, dict[str, Any]],
    gguf_files: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    filename, expected_size, expected_hash = GGUF_TIERS[tier][kind]
    observed = gguf_files.get(filename)
    if observed is None:
        raise RuntimeError(f"{gguf_pin.repo}: missing {filename}")
    if observed["size_bytes"] != expected_size or observed["sha256"] != expected_hash:
        raise RuntimeError(
            f"{gguf_pin.repo}/{filename}: pinned size/hash no longer matches metadata"
        )

    transformer_config = official_files["transformer/config.json"]
    transformer = {
        "role": "transformer",
        "format": "gguf",
        "files": [
            {"path": f"transformer/{filename}", **observed},
            {"path": "transformer/config.json", **transformer_config},
        ],
    }
    components = [
        component_for_prefix("pipeline", "json", official_files, "", optional=True),
        component_for_prefix("scheduler", "json", official_files, "scheduler"),
        component_for_prefix("text_encoder", "safetensors", official_files, "text_encoder"),
        component_for_prefix("tokenizer", "huggingface-json", official_files, "tokenizer"),
    ]
    if kind == "edit":
        components.append(
            component_for_prefix(
                "processor", "huggingface-json", official_files, "processor"
            )
        )
    components.extend(
        [transformer, component_for_prefix("vae", "safetensors", official_files, "vae")]
    )
    suffix = tier.lower()
    return base_manifest(
        model_id=(
            f"qwen-image-edit-2511-{suffix}"
            if kind == "edit"
            else f"qwen-image-2512-{suffix}"
        ),
        family="qwen-image-edit" if kind == "edit" else "qwen-image",
        revision=gguf_pin.revision,
        capability=official_pin.capability,
        quantization=tier,
        license_pin=official_pin,
        source_revisions={
            official_pin.repo: official_pin.revision,
            gguf_pin.repo: gguf_pin.revision,
        },
        components=components,
    )


def base_manifest(
    *,
    model_id: str,
    family: str,
    revision: str,
    capability: str,
    quantization: str,
    license_pin: RepoPin,
    source_revisions: dict[str, str],
    components: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "id": model_id,
        "family": family,
        "revision": revision,
        "capabilities": [capability],
        "license": {
            "spdx": license_pin.license,
            "evidence": (
                f"https://huggingface.co/{license_pin.repo}/blob/"
                f"{license_pin.revision}/README.md"
            ),
            "files": [],
        },
        "quantization": quantization,
        "source_revisions": dict(sorted(source_revisions.items())),
        "components": sorted(components, key=lambda item: item["role"]),
        "limits": {
            "max_sequence_length": 512,
            "max_width": 4096,
            "max_height": 4096,
            "max_pixels": 16_777_216,
        },
    }


def pypi_has_release(package: str, version: str) -> None:
    payload = request_json(f"https://pypi.org/pypi/{package}/json")
    if version not in payload.get("releases", {}):
        raise RuntimeError(f"PyPI no longer reports {package}=={version}")


def git_remote_contains(url: str, commit: str) -> None:
    result = subprocess.run(
        ["git", "ls-remote", url],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if commit not in result.stdout:
        # Git hosts do not promise every reachable commit in ls-remote. Verify the
        # immutable commit URL through GitHub's API when it is not a ref tip.
        parsed = urllib.parse.urlparse(url)
        parts = parsed.path.removesuffix(".git").strip("/").split("/")
        if parsed.hostname != "github.com" or len(parts) != 2:
            raise RuntimeError(f"cannot verify pinned commit {commit} for {url}")
        request_json(f"https://api.github.com/repos/{parts[0]}/{parts[1]}/commits/{commit}")


def verify_comparator() -> None:
    tag = STABLE_DIFFUSION_CPP["release"]
    payload = request_json(
        f"https://api.github.com/repos/leejet/stable-diffusion.cpp/releases/tags/{tag}"
    )
    if payload.get("target_commitish") != STABLE_DIFFUSION_CPP["commit"]:
        raise RuntimeError("stable-diffusion.cpp release commit drift")
    for expected in [STABLE_DIFFUSION_CPP, STABLE_DIFFUSION_CPP["cuda_runtime_asset"]]:
        asset = next(
            (item for item in payload.get("assets", []) if item["name"] == expected["asset"]),
            None,
        )
        if asset is None:
            raise RuntimeError(f"stable-diffusion.cpp asset is missing: {expected['asset']}")
        if asset.get("size") != expected["size_bytes"]:
            raise RuntimeError(f"stable-diffusion.cpp asset size drift: {expected['asset']}")
        if asset.get("digest") != "sha256:" + expected["sha256"]:
            raise RuntimeError(f"stable-diffusion.cpp asset digest drift: {expected['asset']}")
    workflow = request_bytes(STABLE_DIFFUSION_CPP["build"]["workflow"])
    if hashlib.sha256(workflow).hexdigest() != STABLE_DIFFUSION_CPP["build"]["workflow_sha256"]:
        raise RuntimeError("stable-diffusion.cpp build workflow drift")


def comparator_component_lock() -> dict[str, Any]:
    observed = hf_files(COMPARATOR_COMPONENTS, include_paths=set(COMPARATOR_COMPONENT_PATHS))
    for path, expected in COMPARATOR_COMPONENT_PATHS.items():
        actual = observed.get(path)
        if actual is None:
            raise RuntimeError(f"native comparator component is missing: {path}")
        if actual["size_bytes"] != expected["size_bytes"] or actual["sha256"] != expected["sha256"]:
            raise RuntimeError(f"native comparator component metadata drift: {path}")
    return {
        "repository": COMPARATOR_COMPONENTS.repo,
        "revision": COMPARATOR_COMPONENTS.revision,
        "license_basis": COMPARATOR_COMPONENTS.license,
        "files": [
            {"path": path, **metadata}
            for path, metadata in sorted(observed.items())
        ],
        "equivalence_policy": (
            "Compare tensor names, dtypes, shapes, and deterministic sampled values against the "
            "official sharded BF16 components before admitting comparator results."
        ),
    }


def qwen_image_3_audit() -> dict[str, Any]:
    query = "https://huggingface.co/api/models?author=Qwen&search=Qwen-Image-3&limit=100&full=true"
    matches = request_json(query)
    return {
        "audited_at": "2026-07-22",
        "official_hugging_face_query": query,
        "official_hugging_face_matches": [item.get("id") for item in matches],
        "announcement": "https://qwen.ai/blog?id=qwen-image-3.0",
        "local_support_admitted": False,
        "reason": (
            "No official Qwen Hugging Face checkpoint matched at audit time; "
            "recheck before implementation or support claims."
        ),
    }


def quality_suite_lock() -> dict[str, Any]:
    try:
        encoded = QUALITY_SUITE_PATH.read_bytes()
        suite = json.loads(encoded)
    except FileNotFoundError as error:
        raise RuntimeError(
            "missing quality suite; run reference/image/qwen/build_quality_suite.py"
        ) from error

    if suite.get("schema_version") != 1 or suite.get("status") != "frozen":
        raise RuntimeError("quality suite must be schema version 1 and frozen")
    expected_counts = {
        "generation_general": 100,
        "generation_typography": 40,
        "generation_faces_hands_detail": 30,
        "generation_style_color": 30,
        "edit_single_image": 30,
        "edit_multi_image": 20,
        "conditional_inpaint": 20,
    }
    if suite.get("category_counts") != expected_counts:
        raise RuntimeError("quality suite category counts do not match the admission policy")
    if len(suite.get("identity_preservation_pairs", [])) < 50:
        raise RuntimeError("quality suite has fewer than 50 identity-preservation pairs")

    case_ids = [
        case.get("id")
        for cases in suite.get("categories", {}).values()
        for case in cases
    ]
    if len(case_ids) != sum(expected_counts.values()) or len(case_ids) != len(set(case_ids)):
        raise RuntimeError("quality suite case IDs are missing or duplicated")

    evaluator_expectations = {
        "prompt_alignment": EVALUATOR_MODELS["prompt_alignment"],
        "ocr": EVALUATOR_MODELS["ocr"],
        "structural_identity": EVALUATOR_MODELS["structural_identity"],
        "face_identity": EVALUATOR_MODELS["structural_identity"],
    }
    evaluators = suite.get("evaluators", {})
    for role, pin in evaluator_expectations.items():
        evaluator = evaluators.get(role, {})
        if evaluator.get("model") != pin.repo or evaluator.get("revision") != pin.revision:
            raise RuntimeError(f"quality evaluator pin drift for {role}")
        if not evaluator.get("preprocessing"):
            raise RuntimeError(f"quality evaluator preprocessing is not pinned for {role}")
    if not evaluators.get("mask_leakage", {}).get("preprocessing"):
        raise RuntimeError("mask leakage preprocessing is not pinned")
    if set(suite.get("absolute_quality_floors", {})) != set(expected_counts):
        raise RuntimeError("quality suite lacks a category-specific absolute floor")

    fixture_records = suite.get("fixtures", {})
    for fixture_id, fixture in fixture_records.items():
        relative = Path(fixture.get("path", ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe quality fixture path for {fixture_id}")
        path = (REPO_ROOT / relative).resolve()
        if REPO_ROOT.resolve() not in path.parents:
            raise RuntimeError(f"quality fixture escapes repository for {fixture_id}")
        try:
            observed = hashlib.sha256(path.read_bytes()).hexdigest()
        except FileNotFoundError as error:
            raise RuntimeError(f"missing quality fixture {fixture_id}: {relative}") from error
        if observed != fixture.get("sha256"):
            raise RuntimeError(f"quality fixture hash drift for {fixture_id}")

    return {
        "path": str(QUALITY_SUITE_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "suite_version": suite["suite_version"],
        "cases": sum(expected_counts.values()),
        "identity_preservation_pairs": len(suite["identity_preservation_pairs"]),
        "fixtures": len(fixture_records),
    }


def openai_fixture_lock() -> dict[str, Any]:
    try:
        encoded = OPENAI_FIXTURE_MANIFEST_PATH.read_bytes()
        manifest = json.loads(encoded)
    except FileNotFoundError as error:
        raise RuntimeError(
            "missing OpenAI image fixtures; run generate_openai_fixtures.py in the pinned environment"
        ) from error
    if manifest.get("status") != "frozen" or manifest.get("schema_version") != 2:
        raise RuntimeError("OpenAI image fixture manifest must be schema version 2 and frozen")
    if manifest.get("openai_python") != OPENAI_CLIENTS["python"]:
        raise RuntimeError("OpenAI Python fixture version drift")
    if manifest.get("openai_node") != OPENAI_CLIENTS["node"]:
        raise RuntimeError("OpenAI Node fixture version drift")
    if manifest.get("server_openapi") != OPENAI_SERVER_OPENAPI:
        raise RuntimeError("OpenAI server OpenAPI observation drift")
    if manifest.get("generation_transport") != "application/json":
        raise RuntimeError("OpenAI image generation transport drift")
    if manifest.get("edit_transports") != ["application/json", "multipart/form-data"]:
        raise RuntimeError("OpenAI image edit transport drift")
    fixture_root = OPENAI_FIXTURE_MANIFEST_PATH.parent
    records = manifest.get("files", {})
    if not records:
        raise RuntimeError("OpenAI fixture manifest has no files")
    for name, metadata in records.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe OpenAI fixture path: {name}")
        path = fixture_root / relative
        try:
            body = path.read_bytes()
        except FileNotFoundError as error:
            raise RuntimeError(f"missing OpenAI fixture: {name}") from error
        if len(body) != metadata.get("size_bytes"):
            raise RuntimeError(f"OpenAI fixture size drift: {name}")
        if hashlib.sha256(body).hexdigest() != metadata.get("sha256"):
            raise RuntimeError(f"OpenAI fixture hash drift: {name}")
    return {
        "path": str(OPENAI_FIXTURE_MANIFEST_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "files": len(records),
        "schema_sources": OPENAI_SCHEMA_SOURCES,
        "server_openapi": OPENAI_SERVER_OPENAPI,
    }


def build_lock(manifest_hashes: dict[str, str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "audited_at": "2026-08-08",
        "official_models": {
            "generation": {"repository": GENERATION.repo, "revision": GENERATION.revision},
            "edit": {"repository": EDIT.repo, "revision": EDIT.revision},
        },
        "community_transformers": {
            "generation": {
                "repository": GENERATION_GGUF.repo,
                "revision": GENERATION_GGUF.revision,
            },
            "edit": {"repository": EDIT_GGUF.repo, "revision": EDIT_GGUF.revision},
            "tiers": GGUF_TIERS,
        },
        "distilled_adapters": {
            "qwen-image-2512-lightning-4step": {
                "repository": LIGHTNING.repo,
                "revision": LIGHTNING.revision,
                "license": LIGHTNING.license,
                "file": LIGHTNING_FILE,
                "size_bytes": LIGHTNING_SIZE,
                "sha256": LIGHTNING_SHA256,
            }
        },
        "python_packages": PYTHON_PACKAGES,
        "evaluator_packages": EVALUATOR_PACKAGES,
        "evaluator_models": {
            role: {
                "repository": pin.repo,
                "revision": pin.revision,
                "license": pin.license,
            }
            for role, pin in sorted(EVALUATOR_MODELS.items())
        },
        "git_pins": GIT_PINS,
        "openai_clients": OPENAI_CLIENTS,
        "openai_image_fixtures": openai_fixture_lock(),
        "stable_diffusion_cpp": STABLE_DIFFUSION_CPP,
        "native_comparator_components": comparator_component_lock(),
        "qwen_image_3": qwen_image_3_audit(),
        "quality_suite": quality_suite_lock(),
        "reference_hardware_target": {
            "gpu": "NVIDIA GeForce RTX 4090",
            "gpu_memory_mib_observed": 24_564,
            "gpu_driver_observed": "610.74",
            "cpu": "AMD Ryzen 9 9950X 16-Core Processor",
            "logical_processors": 32,
            "host_memory_gib_observed": 125.61,
            "observed_at": "2026-07-21",
        },
        "generated_manifest_sha256": dict(sorted(manifest_hashes.items())),
    }


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def expected_outputs() -> tuple[dict[str, bytes], dict[str, Any]]:
    non_runtime_paths = {".gitattributes", "README.md"}
    official_generation_files = hf_files(GENERATION, exclude_paths=non_runtime_paths)
    official_edit_files = hf_files(EDIT, exclude_paths=non_runtime_paths)
    generation_gguf_paths = {values["generation"][0] for values in GGUF_TIERS.values()}
    edit_gguf_paths = {values["edit"][0] for values in GGUF_TIERS.values()}
    generation_gguf_files = hf_files(GENERATION_GGUF, include_paths=generation_gguf_paths)
    edit_gguf_files = hf_files(EDIT_GGUF, include_paths=edit_gguf_paths)
    lightning_files = hf_files(LIGHTNING, include_paths={LIGHTNING_FILE})
    lightning_file = lightning_files.get(LIGHTNING_FILE)
    if lightning_file is None:
        raise RuntimeError(f"{LIGHTNING.repo}: missing {LIGHTNING_FILE}")
    if (
        lightning_file["size_bytes"] != LIGHTNING_SIZE
        or lightning_file["sha256"] != LIGHTNING_SHA256
    ):
        raise RuntimeError(f"{LIGHTNING.repo}/{LIGHTNING_FILE}: pinned size/hash drift")

    manifests: dict[str, dict[str, Any]] = {
        "qwen-image-2512-bf16.json": official_manifest(GENERATION, official_generation_files),
        "qwen-image-edit-2511-bf16.json": official_manifest(EDIT, official_edit_files),
    }
    for tier in GGUF_TIERS:
        manifests[f"qwen-image-2512-{tier.lower()}.json"] = gguf_manifest(
            tier=tier,
            kind="generation",
            official_pin=GENERATION,
            gguf_pin=GENERATION_GGUF,
            official_files=official_generation_files,
            gguf_files=generation_gguf_files,
        )
        manifests[f"qwen-image-edit-2511-{tier.lower()}.json"] = gguf_manifest(
            tier=tier,
            kind="edit",
            official_pin=EDIT,
            gguf_pin=EDIT_GGUF,
            official_files=official_edit_files,
            gguf_files=edit_gguf_files,
        )

    lightning_manifest = gguf_manifest(
        tier="Q4_K_M",
        kind="generation",
        official_pin=GENERATION,
        gguf_pin=GENERATION_GGUF,
        official_files=official_generation_files,
        gguf_files=generation_gguf_files,
    )
    lightning_manifest["id"] = "qwen-image-2512-lightning-4step-q4_k_m"
    lightning_manifest["revision"] = LIGHTNING.revision
    lightning_manifest["source_revisions"][LIGHTNING.repo] = LIGHTNING.revision
    lightning_manifest["source_revisions"] = dict(
        sorted(lightning_manifest["source_revisions"].items())
    )
    lightning_manifest["components"].append(
        {
            "role": "transformer_adapter",
            "format": "safetensors",
            "files": [
                {
                    "path": f"transformer_adapter/{LIGHTNING_FILE}",
                    **lightning_file,
                }
            ],
        }
    )
    lightning_manifest["components"] = sorted(
        lightning_manifest["components"], key=lambda item: item["role"]
    )
    manifests["qwen-image-2512-lightning-4step-q4_k_m.json"] = lightning_manifest

    encoded = {name: canonical_json_bytes(payload) for name, payload in manifests.items()}
    hashes = {name: hashlib.sha256(body).hexdigest() for name, body in encoded.items()}
    lock = build_lock(hashes)
    return encoded, lock


def verify_external_pins() -> None:
    for package, version in PYTHON_PACKAGES.items():
        pypi_has_release(package, version)
    for package, version in EVALUATOR_PACKAGES.items():
        pypi_has_release(package, version)
    for pin in EVALUATOR_MODELS.values():
        hf_api(pin)
    for pin in GIT_PINS.values():
        git_remote_contains(pin["url"], pin["commit"])
    for name, source in OPENAI_SCHEMA_SOURCES.items():
        observed = hashlib.sha256(request_bytes(source["url"])).hexdigest()
        if observed != source["sha256"]:
            raise RuntimeError(f"OpenAI schema source drift: {name}")
    verify_comparator()


def compare_file(path: Path, expected: bytes) -> None:
    try:
        observed = path.read_bytes()
    except FileNotFoundError as error:
        raise RuntimeError(f"missing generated artifact: {path}") from error
    if observed != expected:
        raise RuntimeError(f"generated artifact drift: {path}; run with --write after review")


def run(*, write: bool, verify: bool) -> None:
    verify_external_pins()
    manifests, lock = expected_outputs()
    lock_bytes = canonical_json_bytes(lock)
    if write:
        for name, body in manifests.items():
            atomic_write(MANIFEST_DIR / name, body)
        atomic_write(LOCK_PATH, lock_bytes)
    if verify:
        for name, body in manifests.items():
            compare_file(MANIFEST_DIR / name, body)
        compare_file(LOCK_PATH, lock_bytes)
    print(
        json.dumps(
            {
                "status": "ok",
                "mode": "write+verify" if write and verify else "write" if write else "verify",
                "manifests": len(manifests),
                "qwen_image_3_matches": lock["qwen_image_3"]["official_hugging_face_matches"],
            },
            sort_keys=True,
        )
    )


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="atomically refresh generated files")
    parser.add_argument("--verify", action="store_true", help="compare checked-in files with live pinned metadata")
    args = parser.parse_args(list(argv))
    if not args.write and not args.verify:
        parser.error("select --write and/or --verify")
    return args


if __name__ == "__main__":
    arguments = parse_args(sys.argv[1:])
    run(write=arguments.write, verify=arguments.verify)
