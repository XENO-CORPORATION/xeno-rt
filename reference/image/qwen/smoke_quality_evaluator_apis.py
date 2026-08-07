#!/usr/bin/env python3
"""Execute one bounded offline CLIP and DINO evaluator API smoke."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

from PIL import Image

import prepare_quality_review


class SmokeError(RuntimeError):
    pass


def snapshot(models: dict[str, Any], role: str) -> Path:
    record = models["model_snapshots"][role]
    path = Path(record["snapshot"])
    if not path.is_dir() or path.name != record["revision"]:
        raise SmokeError(f"immutable {role} snapshot is unavailable")
    return path


def run(args: argparse.Namespace) -> dict[str, Any]:
    import open_clip
    import torch
    import transformers
    from transformers import AutoImageProcessor, AutoModel

    models = prepare_quality_review.read_json(args.models)
    with Image.open(args.image) as loaded:
        image = loaded.convert("RGB").copy()

    clip_path = snapshot(models, "prompt_alignment")
    clip_identifier = f"local-dir:{clip_path}"
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        clip_identifier,
        device="cpu",
        precision="fp32",
        require_pretrained=True,
    )
    tokenizer = open_clip.get_tokenizer(clip_identifier)
    clip_model.eval()
    with torch.inference_mode():
        image_features = torch.nn.functional.normalize(
            clip_model.encode_image(preprocess(image).unsqueeze(0)), dim=-1
        )
        text_features = torch.nn.functional.normalize(
            clip_model.encode_text(tokenizer([args.prompt])), dim=-1
        )
        prompt_alignment = float((image_features * text_features).sum().cpu())
    del clip_model, image_features, text_features
    gc.collect()

    dino_path = snapshot(models, "structural_identity")
    processor = AutoImageProcessor.from_pretrained(dino_path, local_files_only=True)
    dino_model = AutoModel.from_pretrained(dino_path, local_files_only=True).eval()
    inputs = processor(images=[image, image], return_tensors="pt")
    with torch.inference_mode():
        vectors = torch.nn.functional.normalize(
            dino_model(**inputs).last_hidden_state[:, 0, :], dim=-1
        )
        self_identity = float((vectors[0] @ vectors[1]).cpu())
    if not math.isfinite(prompt_alignment) or not math.isfinite(self_identity):
        raise SmokeError("evaluator API smoke returned a non-finite metric")
    return {
        "schema_version": 1,
        "object": "xeno.image.quality_evaluator_api_smoke",
        "status": "passed",
        "production_support": False,
        "host": {
            "operating_system": platform.system(),
            "architecture": platform.machine(),
            "python": sys.version.split()[0],
            "device": "cpu",
        },
        "packages": {
            "open_clip": importlib.metadata.version("open-clip-torch"),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "input": {
            "sha256": prepare_quality_review.sha256_file(args.image),
            "prompt": args.prompt,
        },
        "models": {
            role: {
                "repository": models["model_snapshots"][role]["repository"],
                "revision": models["model_snapshots"][role]["revision"],
                "manifest_sha256": models["model_snapshots"][role]["manifest_sha256"],
            }
            for role in ("prompt_alignment", "structural_identity")
        },
        "metrics": {
            "prompt_alignment": prompt_alignment,
            "dinov2_self_identity": self_identity,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", default="A clean poster reading XENO STUDIO")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        evidence = run(args)
        prepare_quality_review.atomic_write(args.output, prepare_quality_review.canonical_bytes(evidence))
    except (OSError, ValueError, KeyError, TypeError, SmokeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}))
        return 1
    print(json.dumps({"status": "ok", "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
