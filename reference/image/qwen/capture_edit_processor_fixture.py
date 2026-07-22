#!/usr/bin/env python3
"""Capture pinned Qwen-Image-Edit-2511 image-preprocessor samples."""

from __future__ import annotations

import json
import math

import numpy as np
import torch
import transformers
from diffusers import __version__ as diffusers_version
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
from transformers import Qwen2VLImageProcessorFast


def calculate_dimensions(target_area: int, ratio: float) -> tuple[int, int]:
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    return round(width / 32) * 32, round(height / 32) * 32


def source_image() -> Image.Image:
    pixels = np.empty((9, 16, 3), dtype=np.uint8)
    for y in range(9):
        for x in range(16):
            pixels[y, x] = [
                (x * 17 + y * 13) % 256,
                (x * 7 + y * 29 + 31) % 256,
                (x * 3 + y * 5 + 127) % 256,
            ]
    return Image.fromarray(pixels, mode="RGB")


def samples(values: torch.Tensor, indices: list[int]) -> list[float]:
    flat = values.detach().cpu().float().reshape(-1)
    return [float(flat[index]) for index in indices]


def main() -> None:
    torch.set_num_threads(1)
    source = source_image()
    vae_processor = VaeImageProcessor(vae_scale_factor=16)
    condition_width, condition_height = calculate_dimensions(384 * 384, source.width / source.height)
    condition = vae_processor.resize(source, condition_height, condition_width)
    processor = Qwen2VLImageProcessorFast(
        size={"shortest_edge": 3136, "longest_edge": 12845056},
        do_resize=True,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
        do_convert_rgb=True,
        patch_size=14,
        temporal_patch_size=2,
        merge_size=2,
        resample=3,
    )
    vision = processor(images=condition, return_tensors="pt")
    vae_width, vae_height = calculate_dimensions(1024 * 1024, source.width / source.height)
    vae = vae_processor.preprocess(source, vae_height, vae_width).unsqueeze(2)

    vision_indices = [
        0,
        1,
        14,
        1175,
        1176,
        1176 * 17 + 113,
        vision["pixel_values"].numel() // 2,
        vision["pixel_values"].numel() - 1,
    ]
    vae_indices = [
        0,
        1,
        vae_width - 1,
        vae_width * vae_height,
        2 * vae_width * vae_height,
        vae.numel() - 1,
    ]
    fixture = {
        "schema_version": 1,
        "versions": {
            "diffusers": diffusers_version,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
        },
        "source": {
            "width": source.width,
            "height": source.height,
            "formula": [
                "(x * 17 + y * 13) % 256",
                "(x * 7 + y * 29 + 31) % 256",
                "(x * 3 + y * 5 + 127) % 256",
            ],
        },
        "condition_first_size": [condition_width, condition_height],
        "vision_grid": vision["image_grid_thw"].reshape(-1).tolist(),
        "vision_shape": list(vision["pixel_values"].shape),
        "vision_sample_indices": vision_indices,
        "vision_samples": samples(vision["pixel_values"], vision_indices),
        "vae_size": [vae_width, vae_height],
        "vae_shape": list(vae.shape),
        "vae_sample_indices": vae_indices,
        "vae_samples": samples(vae, vae_indices),
    }
    print(json.dumps(fixture, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
