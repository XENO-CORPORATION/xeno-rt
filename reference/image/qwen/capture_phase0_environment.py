#!/usr/bin/env python3
"""Capture the target-machine portion of the Qwen Image Phase 0 evidence."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
LOCK_PATH = HERE / "phase0-lock.json"
MANIFEST_PATH = HERE / "manifests" / "qwen-image-2512-q4_k_m.json"
OUTPUT_PATH = REPO_ROOT / "benchmark-results" / "image" / "phase0-2026-07-21" / "environment.json"
MIB = 1024**2
GIB = 1024**3


def run(command: list[str], timeout: int = 60) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    return {
        "command": command,
        "exit_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def require(command: list[str], timeout: int = 60) -> dict[str, Any]:
    result = run(command, timeout)
    if result["exit_code"] != 0:
        raise RuntimeError(
            f"command failed ({result['exit_code']}): {command!r}\n{result['stderr']}"
        )
    return result


def windows_memory() -> dict[str, Any]:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("memory_load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("available_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended_virtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    return {
        "total_bytes": status.total_physical,
        "available_bytes": status.available_physical,
        "load_percent": status.memory_load,
    }


def parse_nvidia_row(text: str) -> dict[str, Any]:
    fields = [part.strip() for part in text.split(",")]
    if len(fields) != 9:
        raise RuntimeError(f"unexpected nvidia-smi row: {text!r}")
    return {
        "name": fields[0],
        "driver_version": fields[1],
        "memory_total_mib": int(fields[2]),
        "memory_used_mib": int(fields[3]),
        "memory_free_mib": int(fields[4]),
        "performance_state": fields[5],
        "power_limit_w": float(fields[6]),
        "power_draw_w": float(fields[7]),
        "temperature_c": int(fields[8]),
    }


def gpu_snapshot() -> tuple[dict[str, Any], dict[str, Any]]:
    summary = require(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,pstate,power.limit,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    gpu = parse_nvidia_row(summary["stdout"].splitlines()[0])
    applications = require(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = [line.strip() for line in applications["stdout"].splitlines() if line.strip()]
    reference_names = []
    for row in rows:
        lowered = row.lower()
        for name in ("sd-cli.exe", "xrt-server.exe", "xrt-cli.exe"):
            if name in lowered:
                reference_names.append(name)
    return gpu, {
        "reported_rows": len(rows),
        "rows_with_numeric_memory": sum(
            1 for row in rows if row.rsplit(",", 1)[-1].strip().isdigit()
        ),
        "xeno_or_comparator_process_names": sorted(set(reference_names)),
        "non_reference_gpu_processes_present": len(rows) > len(reference_names),
        "privacy_note": "Unrelated process IDs and paths are intentionally not persisted.",
    }


def cache_root() -> Path:
    configured = os.environ.get("XRT_IMAGE_REFERENCE_CACHE")
    return Path(configured).expanduser().resolve() if configured else REPO_ROOT / ".codex-tmp" / "image-reference"


def comparator_tool(lock: dict[str, Any]) -> Path:
    root = cache_root() / "tools" / "stable-diffusion.cpp" / lock["stable_diffusion_cpp"]["release"]
    matches = list(root.rglob("sd-cli.exe"))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one installed sd-cli.exe under {root}")
    return matches[0]


def component_sizes(manifest: dict[str, Any]) -> dict[str, Any]:
    roles: dict[str, int] = {}
    files = 0
    for component in manifest["components"]:
        size = sum(record["size_bytes"] for record in component["files"])
        roles[component["role"]] = size
        files += len(component["files"])
    return {
        "files": files,
        "total_bytes": sum(roles.values()),
        "by_role_bytes": dict(sorted(roles.items())),
    }


def placement_plan(gpu: dict[str, Any], sizes: dict[str, Any]) -> dict[str, Any]:
    memory_fraction = float(os.environ.get("XRT_GPU_MEMORY_FRACTION", "0.90"))
    reserved_mib = int(os.environ.get("XRT_GPU_RESERVED_MB", "1024"))
    total = gpu["memory_total_mib"] * MIB
    free = gpu["memory_free_mib"] * MIB
    baseline = gpu["memory_used_mib"] * MIB
    configured_budget = max(0, min(free, int(total * memory_fraction)) - reserved_mib * MIB)
    reserve_aware = max(0, total - baseline - 2 * GIB)
    image_cap = min(22 * GIB, configured_budget, reserve_aware)
    transformer = sizes["by_role_bytes"]["transformer"]
    return {
        "status": "feasible_on_artifact_bytes_pending_measured_scratch",
        "policy": "sequential_component_residency_v1",
        "effective_memory_fraction": memory_fraction,
        "effective_reserved_mib": reserved_mib,
        "observed_non_xeno_baseline_bytes": baseline,
        "configured_upload_budget_bytes": configured_budget,
        "reserve_aware_budget_bytes": reserve_aware,
        "image_owned_cap_bytes": image_cap,
        "transformer_artifact_bytes": transformer,
        "artifact_headroom_after_transformer_bytes": max(0, image_cap - transformer),
        "phases": [
            {"phase": "prompt", "text_encoder": "cuda_if_admitted_else_cpu", "transformer": "not_resident", "vae": "not_resident"},
            {"phase": "denoise", "text_encoder": "evicted", "transformer": "cuda_resident", "vae": "not_resident"},
            {"phase": "decode", "text_encoder": "evicted", "transformer": "evicted_if_required", "vae": "cuda_tiled_or_cpu"},
        ],
        "caveat": (
            "Artifact bytes do not include dequantization metadata, graph work buffers, latents, "
            "embeddings, or VAE scratch. Production admission remains pending measured peaks."
        ),
    }


def torch_snapshot() -> dict[str, Any]:
    import diffusers
    import openai
    import safetensors
    import torch
    import transformers

    cuda = torch.cuda.is_available()
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": cuda,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "safetensors": safetensors.__version__,
        "openai": openai.__version__,
    }
    if cuda:
        free, total = torch.cuda.mem_get_info(0)
        result.update(
            {
                "device": torch.cuda.get_device_name(0),
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "memory_free_bytes_after_torch_init": free,
                "memory_total_bytes": total,
            }
        )
    return result


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


def build_report() -> dict[str, Any]:
    lock_bytes = LOCK_PATH.read_bytes()
    lock = json.loads(lock_bytes)
    manifest_bytes = MANIFEST_PATH.read_bytes()
    manifest = json.loads(manifest_bytes)
    gpu, applications = gpu_snapshot()
    tool = comparator_tool(lock)
    tool_version = require([str(tool), "--version"])
    tool_devices = require([str(tool), "--list-devices"])
    sizes = component_sizes(manifest)
    cpu = require(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name).Trim()",
        ]
    )["stdout"]
    head = require(["git", "rev-parse", "HEAD"])["stdout"]
    origin_main = require(["git", "ls-remote", "origin", "refs/heads/main"])["stdout"].split()[0]
    disk = shutil.disk_usage(REPO_ROOT)
    return {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "status": "phase0_environment_captured_reference_runs_pending",
        "repository": {
            "path": ".",
            "head": head,
            "origin_main": origin_main,
        },
        "host": {
            "os": platform.platform(),
            "cpu": cpu,
            "logical_processors": os.cpu_count(),
            "memory": windows_memory(),
            "workspace_disk": {"total_bytes": disk.total, "free_bytes": disk.free},
        },
        "gpu_before_reference_process": gpu,
        "gpu_compute_applications": applications,
        "reference_environment": torch_snapshot(),
        "native_comparator": {
            "executable": str(tool.relative_to(REPO_ROOT)).replace("\\", "/"),
            "version": tool_version,
            "devices": tool_devices,
            "lock": lock["stable_diffusion_cpp"],
        },
        "phase0_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "quality_suite": lock["quality_suite"],
        "q4_bundle": {
            "manifest": str(MANIFEST_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "model_revision": manifest["revision"],
            "quantization": manifest["quantization"],
            "components": sizes,
        },
        "provisional_4090_placement": placement_plan(gpu, sizes),
        "required_followup": [
            "capture a quiet stable non-XENO VRAM baseline before timed runs",
            "measure Diffusers BF16 component hashes and image output",
            "prove comparator BF16 component conversion equivalence",
            "measure stable-diffusion.cpp Q4 peak RAM/VRAM and timing",
            "replace provisional artifact headroom with measured XENO scratch and allocation peaks",
        ],
    }


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    report = build_report()
    body = canonical_bytes(report)
    if args.write:
        atomic_write(OUTPUT_PATH, body)
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(OUTPUT_PATH.relative_to(REPO_ROOT)) if args.write else None,
                "sha256": hashlib.sha256(body).hexdigest(),
                "image_owned_cap_bytes": report["provisional_4090_placement"]["image_owned_cap_bytes"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main(sys.argv[1:])
