#!/usr/bin/env python3
"""Run the pinned stable-diffusion.cpp Qwen Image Q4 comparator."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import diffusers
import numpy as np
from diffusers import FlowMatchEulerDiscreteScheduler
from PIL import Image


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
LOCK_PATH = HERE / "phase0-lock.json"
MANIFEST_PATH = HERE / "manifests" / "qwen-image-2512-q4_k_m.json"
EVIDENCE_ROOT = REPO_ROOT / "benchmark-results" / "image" / "phase0-2026-07-21" / "stable-diffusion-cpp"
CHUNK_BYTES = 8 * 1024 * 1024


def canonical_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def cache_root() -> Path:
    configured = os.environ.get("XRT_IMAGE_REFERENCE_CACHE")
    return Path(configured).expanduser().resolve() if configured else REPO_ROOT / ".codex-tmp" / "image-reference"


def resolve_inputs(rehash: bool) -> dict[str, Any]:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    manifest_bytes = MANIFEST_PATH.read_bytes()
    manifest = json.loads(manifest_bytes)
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    bundle = cache_root() / "bundles" / f"{manifest['id']}-{digest[:16]}"
    comparator = lock["native_comparator_components"]
    components = cache_root() / "comparator-components" / comparator["revision"]
    tool_root = cache_root() / "tools" / "stable-diffusion.cpp" / lock["stable_diffusion_cpp"]["release"]
    tools = list(tool_root.rglob("sd-cli.exe"))
    if len(tools) != 1:
        raise RuntimeError(f"expected one installed sd-cli.exe under {tool_root}")

    records: list[tuple[str, Path, dict[str, Any]]] = []
    transformer_record = None
    for component in manifest["components"]:
        for record in component["files"]:
            path = bundle / record["path"]
            records.append((f"bundle:{record['path']}", path, record))
            if component["role"] == "transformer" and record["path"].endswith(".gguf"):
                transformer_record = (path, record)
    comparator_paths = {}
    for record in comparator["files"]:
        path = components / record["path"]
        records.append((f"comparator:{record['path']}", path, record))
        if "/text_encoders/" in record["path"]:
            comparator_paths["llm"] = path
        elif "/vae/" in record["path"]:
            comparator_paths["vae"] = path
    if transformer_record is None or set(comparator_paths) != {"llm", "vae"}:
        raise RuntimeError("incomplete native-comparator component mapping")
    verified_bytes = 0
    for label, path, record in records:
        if not path.is_file() or path.stat().st_size != record["size_bytes"]:
            raise RuntimeError(f"missing or wrong-sized artifact: {label}")
        if rehash and sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"artifact hash mismatch: {label}")
        verified_bytes += record["size_bytes"]
    return {
        "lock": lock,
        "manifest": manifest,
        "manifest_sha256": digest,
        "bundle": bundle,
        "tool": tools[0],
        "transformer": transformer_record[0],
        "llm": comparator_paths["llm"],
        "vae": comparator_paths["vae"],
        "verification": {"files": len(records), "bytes": verified_bytes, "sha256_rechecked": rehash},
    }


def process_working_set_bytes(pid: int) -> int | None:
    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    ctypes.windll.kernel32.OpenProcess.restype = ctypes.c_void_p
    process = ctypes.windll.kernel32.OpenProcess(0x0400 | 0x0010, False, pid)
    if not process:
        return None
    try:
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not ctypes.windll.psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
            return None
        return int(counters.WorkingSetSize)
    finally:
        ctypes.windll.kernel32.CloseHandle(process)


def device_used_mib() -> int | None:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if result.returncode:
        return None
    try:
        return int(result.stdout.splitlines()[0].strip())
    except (IndexError, ValueError):
        return None


class ChildSampler:
    def __init__(self, pid: int, interval: float = 0.25) -> None:
        self.pid = pid
        self.interval = interval
        self.samples: list[dict[str, Any]] = []
        self.stop = threading.Event()
        self.started = time.monotonic()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self.stop.is_set():
            self.samples.append(
                {
                    "elapsed_seconds": time.monotonic() - self.started,
                    "working_set_bytes": process_working_set_bytes(self.pid),
                    "device_used_mib": device_used_mib(),
                }
            )
            self.stop.wait(self.interval)

    def __enter__(self) -> "ChildSampler":
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop.set()
        self.thread.join(timeout=20)

    def summary(self) -> dict[str, Any]:
        working = [sample["working_set_bytes"] for sample in self.samples if sample["working_set_bytes"] is not None]
        device = [sample["device_used_mib"] for sample in self.samples if sample["device_used_mib"] is not None]
        return {
            "samples": len(self.samples),
            "interval_seconds": self.interval,
            "process_working_set_peak_bytes": max(working) if working else None,
            "device_used_initial_mib": device[0] if device else None,
            "device_used_peak_mib": max(device) if device else None,
            "device_used_peak_delta_mib": max(device) - device[0] if device else None,
        }


def assert_no_comparator_process() -> None:
    result = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq sd-cli.exe", "/FO", "CSV", "/NH"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode == 0 and '"sd-cli.exe"' in result.stdout.lower():
        raise RuntimeError("a pre-existing sd-cli.exe process is running")


def kill_process_tree(pid: int) -> None:
    subprocess.run(
        ["taskkill", "/T", "/F", "/PID", str(pid)],
        capture_output=True,
        timeout=60,
        check=False,
    )


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return "$XRT_IMAGE_REFERENCE_CACHE/" + str(path.relative_to(cache_root())).replace("\\", "/")


def sanitize(text: str) -> str:
    return text.replace(str(REPO_ROOT), ".").replace(str(cache_root()), "$XRT_IMAGE_REFERENCE_CACHE")


def diffusers_schedule(bundle: Path, width: int, height: int, steps: int) -> dict[str, Any]:
    """Build the exact Qwen-Image schedule used by pinned Diffusers."""
    if width % 16 or height % 16:
        raise ValueError("Qwen Image comparator dimensions must be divisible by 16")
    config_path = bundle / "scheduler" / "scheduler_config.json"
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(config)

    # Qwen Image downsamples by 8 and then packs each 2x2 latent patch into one token.
    image_seq_len = (height // 16) * (width // 16)
    base_seq_len = int(scheduler.config.get("base_image_seq_len", 256))
    max_seq_len = int(scheduler.config.get("max_image_seq_len", 4096))
    base_shift = float(scheduler.config.get("base_shift", 0.5))
    max_shift = float(scheduler.config.get("max_shift", 1.15))
    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    intercept = base_shift - slope * base_seq_len
    mu = image_seq_len * slope + intercept
    input_sigmas = np.linspace(1.0, 1.0 / steps, steps).tolist()
    scheduler.set_timesteps(num_inference_steps=steps, sigmas=input_sigmas, mu=mu)
    final_sigmas = [float(value) for value in scheduler.sigmas.tolist()]
    if not diffusers.__version__:
        raise RuntimeError("unable to identify the pinned Diffusers version")
    payload = {
        "implementation": "diffusers.FlowMatchEulerDiscreteScheduler",
        "diffusers_version": diffusers.__version__,
        "model_config_diffusers_version": config.get("_diffusers_version"),
        "config_path": display_path(config_path),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "image_seq_len": image_seq_len,
        "latent_downsample_factor": 8,
        "latent_patch_size": 2,
        "mu": mu,
        "input_sigmas": input_sigmas,
        "final_sigmas": final_sigmas,
        "timesteps": [float(value) for value in scheduler.timesteps.tolist()],
    }
    payload["sha256"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    return payload


def cleanup_staging(run_id: str) -> None:
    root = EVIDENCE_ROOT.resolve()
    if not EVIDENCE_ROOT.exists():
        return
    for candidate in EVIDENCE_ROOT.glob(f"{run_id}.*.staging"):
        resolved = candidate.resolve()
        if resolved.parent != root or not resolved.is_dir():
            raise RuntimeError(f"refusing to clean unexpected staging path: {candidate}")
        shutil.rmtree(resolved)


def run(args: argparse.Namespace) -> dict[str, Any]:
    assert_no_comparator_process()
    inputs = resolve_inputs(args.rehash_artifacts)
    schedule = diffusers_schedule(inputs["bundle"], args.width, args.height, args.steps)
    sigma_argument = ",".join(format(value, ".9g") for value in schedule["final_sigmas"])
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    final_dir = EVIDENCE_ROOT / args.run_id
    if final_dir.exists():
        raise RuntimeError(f"comparator run ID already exists: {args.run_id}")
    staging = Path(tempfile.mkdtemp(prefix=args.run_id + ".", suffix=".staging", dir=EVIDENCE_ROOT))
    output = staging / "image.png"
    command = [
        str(inputs["tool"]),
        "--mode",
        "img_gen",
        "--diffusion-model",
        str(inputs["transformer"]),
        "--llm",
        str(inputs["llm"]),
        "--vae",
        str(inputs["vae"]),
        "--prompt",
        args.prompt,
        "--negative-prompt",
        args.negative_prompt,
        "--cfg-scale",
        str(args.cfg_scale),
        "--sampling-method",
        "euler",
        "--sigmas",
        sigma_argument,
        "--steps",
        str(args.steps),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--seed",
        str(args.seed),
        "--batch-count",
        "1",
        "--threads",
        str(args.threads),
        "--backend",
        "diffusion=cuda0,te=cpu,vae=cuda0",
        "--params-backend",
        "diffusion=cuda0,te=cpu,vae=cpu",
        "--max-vram",
        f"cuda0={args.max_vram_gib}",
        "--diffusion-fa",
        "--vae-tiling",
        "--mmap",
        "--disable-image-metadata",
        "--output",
        str(output),
        "--verbose",
    ]
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=staging,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    timed_out = False
    with ChildSampler(process.pid) as sampler:
        try:
            stdout, stderr = process.communicate(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            kill_process_tree(process.pid)
            stdout, stderr = process.communicate(timeout=60)
    wall_seconds = time.monotonic() - started
    assert_no_comparator_process()
    stdout = sanitize(stdout)
    stderr = sanitize(stderr)
    atomic_write(staging / "stdout.log", stdout.encode("utf-8"))
    atomic_write(staging / "stderr.log", stderr.encode("utf-8"))
    if timed_out:
        raise RuntimeError(f"stable-diffusion.cpp timed out after {args.timeout_seconds} seconds")
    if process.returncode != 0:
        raise RuntimeError(f"stable-diffusion.cpp exited with code {process.returncode}")
    if not output.is_file():
        raise RuntimeError("stable-diffusion.cpp did not produce its output image")
    with Image.open(output) as loaded:
        image = loaded.convert("RGB")
        pixel_bytes = image.tobytes()
    image_bytes = output.read_bytes()
    version = subprocess.run(
        [str(inputs["tool"]), "--version"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    report = {
        "schema_version": 1,
        "status": "passed",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "oracle": "stable-diffusion.cpp",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_id": args.run_id,
        "profile": args.profile,
        "comparator": {
            "release": inputs["lock"]["stable_diffusion_cpp"]["release"],
            "commit": inputs["lock"]["stable_diffusion_cpp"]["commit"],
            "version_output": sanitize((version.stdout + version.stderr).strip()),
            "executable": display_path(inputs["tool"]),
            "build": inputs["lock"]["stable_diffusion_cpp"]["build"],
        },
        "model": {
            "id": inputs["manifest"]["id"],
            "revision": inputs["manifest"]["revision"],
            "manifest_sha256": inputs["manifest_sha256"],
            "artifact_verification": inputs["verification"],
            "transformer": display_path(inputs["transformer"]),
            "text_encoder": display_path(inputs["llm"]),
            "vae": display_path(inputs["vae"]),
        },
        "request": {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "seed": args.seed,
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "cfg_scale": args.cfg_scale,
            "sampling_method": "euler",
            "scheduler": schedule,
            "outputs": 1,
        },
        "placement": {
            "runtime_backend": "diffusion=cuda0,te=cpu,vae=cuda0",
            "parameter_backend": "diffusion=cuda0,te=cpu,vae=cpu",
            "max_vram_gib": args.max_vram_gib,
            "diffusion_flash_attention": True,
            "vae_tiling": True,
            "mmap": True,
        },
        "command": [display_path(Path(value)) if index in {0, 4, 6, 8, len(command) - 2} else value for index, value in enumerate(command)],
        "timings": {"wall_seconds": wall_seconds},
        "resources": sampler.summary(),
        "image": {
            "path": "image.png",
            "mode": "RGB",
            "width": image.width,
            "height": image.height,
            "pixel_sha256": hashlib.sha256(pixel_bytes).hexdigest(),
            "pixel_size_bytes": len(pixel_bytes),
            "png_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "png_size_bytes": len(image_bytes),
        },
        "logs": {
            "stdout": {"path": "stdout.log", "sha256": hashlib.sha256(stdout.encode()).hexdigest()},
            "stderr": {"path": "stderr.log", "sha256": hashlib.sha256(stderr.encode()).hexdigest()},
        },
        "admission_note": (
            "This establishes a scheduler-matched native baseline only. It is not an XENO performance pass; "
            "Diffusers and stable-diffusion.cpp still use different kernels and RNG implementations."
        ),
    }
    atomic_write(staging / "result.json", canonical_bytes(report))
    os.replace(staging, final_dir)
    return {
        "result": str((final_dir / "result.json").relative_to(REPO_ROOT)).replace("\\", "/"),
        "pixel_sha256": report["image"]["pixel_sha256"],
        "wall_seconds": wall_seconds,
        "device_peak_delta_mib": report["resources"]["device_used_peak_delta_mib"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["smoke", "release"], default="smoke")
    parser.add_argument("--run-id")
    parser.add_argument("--prompt", default="A cobalt mechanical keyboard on a walnut desk, precise product photograph.")
    parser.add_argument("--negative-prompt", default=" ")
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--max-vram-gib", type=float, default=17.0)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=int)
    parser.add_argument("--no-rehash-artifacts", action="store_false", dest="rehash_artifacts")
    parser.set_defaults(rehash_artifacts=True)
    args = parser.parse_args()
    if args.profile == "smoke":
        args.width, args.height, args.steps = 512, 512, 4
        args.timeout_seconds = args.timeout_seconds or 7200
    else:
        args.width, args.height, args.steps = 1024, 1024, 50
        args.timeout_seconds = args.timeout_seconds or 21600
    if args.run_id is None:
        args.run_id = f"q4-{args.profile}-{args.width}x{args.height}-s{args.steps}-seed{args.seed}"
    if not all(character.isalnum() or character in "-_" for character in args.run_id):
        parser.error("--run-id may contain only letters, digits, hyphens, and underscores")
    if not 1 <= args.threads <= 32 or not 1 <= args.timeout_seconds <= 21600:
        parser.error("threads or timeout is outside the bounded reference range")
    return args


def main() -> None:
    args = parse_args()
    cleanup_staging(args.run_id)
    try:
        result = run(args)
    except BaseException as error:
        cleanup_staging(args.run_id)
        failure = {
            "schema_version": 1,
            "status": "failed",
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "run_id": args.run_id,
            "error_type": type(error).__name__,
            "error": sanitize(str(error))[:4000],
        }
        atomic_write(EVIDENCE_ROOT / f"{args.run_id}-failure.json", canonical_bytes(failure))
        raise
    print(json.dumps({"status": "ok", **result}, sort_keys=True))


if __name__ == "__main__":
    main()
