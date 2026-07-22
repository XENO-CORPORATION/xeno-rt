#!/usr/bin/env python3
"""Run the pinned official Qwen Image Diffusers oracle and emit fixtures."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
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


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
MANIFEST_PATH = HERE / "manifests" / "qwen-image-2512-bf16.json"
EVIDENCE_ROOT = REPO_ROOT / "benchmark-results" / "image" / "phase0-2026-07-21"
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


def resolve_bundle(manifest_bytes: bytes, manifest: dict[str, Any], override: str | None) -> Path:
    if override:
        bundle = Path(override).expanduser().resolve()
    else:
        digest = hashlib.sha256(manifest_bytes).hexdigest()
        bundle = cache_root() / "bundles" / f"{manifest['id']}-{digest[:16]}"
    if not bundle.is_dir():
        raise RuntimeError(
            f"verified BF16 bundle is missing: {bundle}; run download_reference_artifacts.py --bundle qwen-image-2512-bf16"
        )
    return bundle


def verify_bundle(bundle: Path, manifest: dict[str, Any], rehash: bool) -> dict[str, Any]:
    verified_files = 0
    verified_bytes = 0
    for component in manifest["components"]:
        for record in component["files"]:
            relative = Path(record["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"unsafe manifest path: {record['path']}")
            path = (bundle / relative).resolve()
            if bundle.resolve() not in path.parents:
                raise RuntimeError(f"manifest path escapes bundle: {record['path']}")
            if not path.is_file() or path.stat().st_size != record["size_bytes"]:
                raise RuntimeError(f"bundle file missing or has wrong size: {record['path']}")
            if rehash and sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"bundle file hash mismatch: {record['path']}")
            verified_files += 1
            verified_bytes += record["size_bytes"]
    return {"files": verified_files, "bytes": verified_bytes, "sha256_rechecked": rehash}


def process_working_set_bytes() -> int:
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

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    process = ctypes.windll.kernel32.GetCurrentProcess()
    if not ctypes.windll.psapi.GetProcessMemoryInfo(
        process, ctypes.byref(counters), counters.cb
    ):
        raise ctypes.WinError()
    return int(counters.WorkingSetSize)


def device_used_mib() -> int | None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.splitlines()[0].strip())
    except (IndexError, ValueError):
        return None


class ResourceSampler:
    def __init__(self, interval_seconds: float = 0.25) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="phase0-resource-sampler", daemon=True)

    def _sample(self) -> None:
        try:
            working_set = process_working_set_bytes()
        except OSError:
            working_set = None
        self.samples.append(
            {
                "elapsed_seconds": time.monotonic() - self.started_at,
                "process_working_set_bytes": working_set,
                "device_used_mib": device_used_mib(),
            }
        )

    def _run(self) -> None:
        while not self.stop_event.is_set():
            self._sample()
            self.stop_event.wait(self.interval_seconds)

    def __enter__(self) -> "ResourceSampler":
        self.started_at = time.monotonic()
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop_event.set()
        self.thread.join(timeout=20)
        self._sample()

    def summary(self) -> dict[str, Any]:
        working = [sample["process_working_set_bytes"] for sample in self.samples if sample["process_working_set_bytes"] is not None]
        device = [sample["device_used_mib"] for sample in self.samples if sample["device_used_mib"] is not None]
        return {
            "samples": len(self.samples),
            "interval_seconds": self.interval_seconds,
            "process_working_set_initial_bytes": working[0] if working else None,
            "process_working_set_peak_bytes": max(working) if working else None,
            "device_used_initial_mib": device[0] if device else None,
            "device_used_peak_mib": max(device) if device else None,
            "device_used_peak_delta_mib": max(device) - device[0] if device else None,
        }


class TensorFixtures:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self.bodies: dict[str, bytes] = {}

    def capture(self, name: str, tensor: Any) -> None:
        import torch

        if tensor is None or name in self.records:
            return
        value = tensor.detach().to(device="cpu").contiguous()
        body = value.view(torch.uint8).numpy().tobytes()
        filename = f"{name}.bin"
        sample = value.reshape(-1)[:16]
        if value.is_floating_point():
            finite = bool(torch.isfinite(value.float()).all())
            sample_values: list[Any] = [float(item) for item in sample.float().tolist()]
        else:
            finite = True
            sample_values = [int(item) for item in sample.tolist()]
        self.records[name] = {
            "path": filename,
            "sha256": hashlib.sha256(body).hexdigest(),
            "size_bytes": len(body),
            "shape": list(value.shape),
            "dtype": str(value.dtype).removeprefix("torch."),
            "byte_order": sys.byteorder,
            "finite": finite,
            "first_values_as_f32_or_i64": sample_values,
        }
        self.bodies[filename] = body


def sanitized_error(error: BaseException) -> str:
    text = str(error).replace(str(REPO_ROOT), ".").replace(str(cache_root()), "$XRT_IMAGE_REFERENCE_CACHE")
    return text[:4000]


def run_reference(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["DIFFUSERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import diffusers
    import numpy
    import PIL
    import safetensors
    import torch
    import transformers
    from diffusers import QwenImagePipeline

    if not torch.cuda.is_available():
        raise RuntimeError("the pinned Diffusers reference requires a CUDA device")
    cuda_device = torch.device("cuda:0")
    # PyTorch 2.13 on Windows does not initialize the primary CUDA context
    # merely by evaluating is_available(). Memory-stat APIs reject even a
    # valid device until it has been selected explicitly.
    torch.cuda.set_device(cuda_device)
    manifest_bytes = MANIFEST_PATH.read_bytes()
    manifest = json.loads(manifest_bytes)
    bundle = resolve_bundle(manifest_bytes, manifest, args.bundle_root)
    bundle_verification = verify_bundle(bundle, manifest, args.rehash_bundle)
    dimensions = {"width": args.width, "height": args.height}
    if args.width % 16 or args.height % 16:
        raise RuntimeError("reference width and height must be divisible by 16")

    tensors = TensorFixtures()
    timings: dict[str, float] = {}
    first_step_at: float | None = None
    transformer_calls = 0
    encode_calls = 0
    started = time.monotonic()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(cuda_device)

    with ResourceSampler() as sampler:
        load_started = time.monotonic()
        pipeline = QwenImagePipeline.from_pretrained(
            bundle,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        timings["load_seconds"] = time.monotonic() - load_started
        pipeline.set_progress_bar_config(disable=True)
        pipeline.vae.enable_tiling()
        offload_started = time.monotonic()
        pipeline.enable_group_offload(
            onload_device=cuda_device,
            offload_device=torch.device("cpu"),
            offload_type="leaf_level",
            use_stream=True,
            record_stream=False,
            low_cpu_mem_usage=True,
        )
        timings["offload_hook_seconds"] = time.monotonic() - offload_started

        # Capture the positive prompt's full-sequence decoder states so native
        # implementations can distinguish graph errors from expected
        # cross-kernel BF16 drift. TensorFixtures is first-write-wins, so the
        # later negative-prompt pass cannot overwrite these checkpoints.
        text_hook_handles = []
        language_model = pipeline.text_encoder.model.language_model
        for layer_index, layer in enumerate(language_model.layers):
            def capture_text_layer(_module: Any, _inputs: Any, output: Any, *, index: int = layer_index) -> None:
                value = output[0] if isinstance(output, tuple) else output
                tensors.capture(f"text_encoder_layer_{index:02d}", value)

            text_hook_handles.append(layer.register_forward_hook(capture_text_layer))

        def capture_text_final_norm(_module: Any, _inputs: Any, output: Any) -> None:
            tensors.capture("text_encoder_final_norm", output)

        text_hook_handles.append(language_model.norm.register_forward_hook(capture_text_final_norm))

        original_encode = pipeline.encode_prompt

        def recording_encode(*call_args: Any, **call_kwargs: Any) -> Any:
            nonlocal encode_calls
            result = original_encode(*call_args, **call_kwargs)
            prefix = "prompt" if encode_calls == 0 else "negative_prompt"
            tensors.capture(f"{prefix}_embeds", result[0])
            tensors.capture(f"{prefix}_attention_mask", result[1])
            encode_calls += 1
            return result

        pipeline.encode_prompt = recording_encode
        original_prepare = pipeline.prepare_latents

        def recording_prepare(*call_args: Any, **call_kwargs: Any) -> Any:
            result = original_prepare(*call_args, **call_kwargs)
            tensors.capture("initial_latents_packed", result)
            return result

        pipeline.prepare_latents = recording_prepare
        original_transformer_forward = pipeline.transformer.forward

        def recording_transformer_forward(*call_args: Any, **call_kwargs: Any) -> Any:
            nonlocal transformer_calls
            result = original_transformer_forward(*call_args, **call_kwargs)
            if transformer_calls < 2:
                role = "conditional" if transformer_calls == 0 else "unconditional"
                output = result[0] if isinstance(result, tuple) else result.sample
                tensors.capture(f"transformer_step0_{role}_noise", output)
            transformer_calls += 1
            return result

        pipeline.transformer.forward = recording_transformer_forward
        original_decode = pipeline.vae.decode

        def recording_decode(*call_args: Any, **call_kwargs: Any) -> Any:
            if call_args:
                tensors.capture("vae_decode_input", call_args[0])
            elif "z" in call_kwargs:
                tensors.capture("vae_decode_input", call_kwargs["z"])
            result = original_decode(*call_args, **call_kwargs)
            output = result[0] if isinstance(result, tuple) else result.sample
            tensors.capture("vae_decode_output", output)
            return result

        pipeline.vae.decode = recording_decode

        capture_steps = {0, args.steps // 2, args.steps - 1}
        inference_started = time.monotonic()

        def callback(_pipeline: Any, step: int, _timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
            nonlocal first_step_at
            if first_step_at is None:
                first_step_at = time.monotonic()
            if step in capture_steps:
                tensors.capture(f"latents_after_step_{step:03d}", callback_kwargs["latents"])
            return callback_kwargs

        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        result = pipeline(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            true_cfg_scale=args.true_cfg_scale,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            generator=generator,
            output_type="pil",
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents"],
            max_sequence_length=512,
        )
        torch.cuda.synchronize(cuda_device)
        for handle in text_hook_handles:
            handle.remove()
        timings["inference_seconds"] = time.monotonic() - inference_started
        timings["time_to_first_completed_step_seconds"] = (
            first_step_at - inference_started if first_step_at is not None else math.nan
        )
        image = result.images[0].convert("RGB")
        tensors.capture("scheduler_timesteps", pipeline.scheduler.timesteps)
        tensors.capture("scheduler_sigmas", pipeline.scheduler.sigmas)

    timings["total_seconds"] = time.monotonic() - started
    pixel_bytes = image.tobytes()
    report = {
        "schema_version": 1,
        "status": "passed",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "oracle": "official_diffusers_bf16",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_id": args.run_id,
        "profile": args.profile,
        "model": {
            "id": manifest["id"],
            "revision": manifest["revision"],
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "bundle_verification": bundle_verification,
        },
        "request": {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "seed": args.seed,
            **dimensions,
            "steps": args.steps,
            "true_cfg_scale": args.true_cfg_scale,
            "outputs": 1,
        },
        "execution": {
            "backend": "cuda:0",
            "dtype": "bfloat16",
            "offload": "pipeline_leaf_level_group_offload_streamed_low_cpu_mem_usage",
            "vae_tiling": True,
            "transformer_forward_calls": transformer_calls,
            "prompt_encode_calls": encode_calls,
            "offline_after_install": True,
        },
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "diffusers": diffusers.__version__,
            "transformers": transformers.__version__,
            "safetensors": safetensors.__version__,
            "numpy": numpy.__version__,
            "pillow": PIL.__version__,
        },
        "device": {
            "name": torch.cuda.get_device_name(cuda_device),
            "compute_capability": list(torch.cuda.get_device_capability(cuda_device)),
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(cuda_device),
            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(cuda_device),
            "sampled_resources": sampler.summary(),
        },
        "timings": timings,
        "image": {
            "mode": "RGB",
            "width": image.width,
            "height": image.height,
            "pixel_sha256": hashlib.sha256(pixel_bytes).hexdigest(),
            "pixel_size_bytes": len(pixel_bytes),
        },
        "tensors": tensors.records,
    }
    if not all(record["finite"] for record in tensors.records.values()):
        raise RuntimeError("reference tensor capture contains a non-finite value")

    output_parent = EVIDENCE_ROOT / "diffusers"
    output_parent.mkdir(parents=True, exist_ok=True)
    final_dir = output_parent / args.run_id
    if final_dir.exists():
        raise RuntimeError(f"reference run ID already exists: {args.run_id}")
    staging = Path(tempfile.mkdtemp(prefix=args.run_id + ".", suffix=".staging", dir=output_parent))
    try:
        tensor_dir = staging / "tensors"
        tensor_dir.mkdir()
        for filename, body in tensors.bodies.items():
            atomic_write(tensor_dir / filename, body)
        image_path = staging / "image.png"
        image.save(image_path, format="PNG", compress_level=9, optimize=False)
        png_bytes = image_path.read_bytes()
        report["image"].update(
            {
                "path": "image.png",
                "png_sha256": hashlib.sha256(png_bytes).hexdigest(),
                "png_size_bytes": len(png_bytes),
                "encoder_profile": "Pillow pinned PNG RGB compress_level=9 optimize=false",
            }
        )
        for record in report["tensors"].values():
            record["path"] = "tensors/" + record["path"]
        report_bytes = canonical_bytes(report)
        atomic_write(staging / "result.json", report_bytes)
        os.replace(staging, final_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "result": str((final_dir / "result.json").relative_to(REPO_ROOT)).replace("\\", "/"),
        "pixel_sha256": report["image"]["pixel_sha256"],
        "inference_seconds": timings["inference_seconds"],
        "torch_peak_reserved_bytes": report["device"]["torch_peak_reserved_bytes"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["smoke", "release"], default="smoke")
    parser.add_argument("--run-id")
    parser.add_argument("--bundle-root")
    parser.add_argument("--no-rehash-bundle", action="store_false", dest="rehash_bundle")
    parser.add_argument("--prompt", default="A cobalt mechanical keyboard on a walnut desk, precise product photograph.")
    parser.add_argument("--negative-prompt", default=" ")
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.set_defaults(rehash_bundle=True)
    args = parser.parse_args()
    if args.profile == "smoke":
        args.width, args.height, args.steps = 512, 512, 4
    else:
        args.width, args.height, args.steps = 1024, 1024, 50
    if args.run_id is None:
        args.run_id = f"bf16-{args.profile}-{args.width}x{args.height}-s{args.steps}-seed{args.seed}"
    if not all(character.isalnum() or character in "-_" for character in args.run_id):
        parser.error("--run-id may contain only letters, digits, hyphens, and underscores")
    return args


def main() -> None:
    args = parse_args()
    try:
        result = run_reference(args)
    except BaseException as error:
        failure = {
            "schema_version": 1,
            "status": "failed",
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "run_id": args.run_id,
            "error_type": type(error).__name__,
            "error": sanitized_error(error),
        }
        atomic_write(EVIDENCE_ROOT / "diffusers" / f"{args.run_id}-failure.json", canonical_bytes(failure))
        raise
    print(json.dumps({"status": "ok", **result}, sort_keys=True))


if __name__ == "__main__":
    main()
