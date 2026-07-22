#!/usr/bin/env python3
"""Capture deterministic tiny Qwen Image VAE encode/decode fixtures.

The production runtime never imports Python. This script is a pinned test
oracle for operation order and tensor layout in Diffusers 0.39.0.
"""

from __future__ import annotations

import json

import diffusers
import torch
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage


def flattened(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().cpu().float().reshape(-1).tolist()


def sampled(tensor: torch.Tensor) -> dict[str, object]:
    values = flattened(tensor)
    indices = sorted(
        {0, 1, len(values) // 4, len(values) // 2, (3 * len(values)) // 4, len(values) - 2, len(values) - 1}
    )
    return {
        "shape": list(tensor.shape),
        "sample_indices": indices,
        "samples": [values[index] for index in indices],
    }


def main() -> None:
    torch.manual_seed(0)
    model = AutoencoderKLQwenImage(
        base_dim=4,
        z_dim=2,
        dim_mult=[1, 2],
        num_res_blocks=1,
        attn_scales=[],
        temperal_downsample=[False],
        dropout=0.0,
        input_channels=3,
        latents_mean=[0.0, 0.0],
        latents_std=[1.0, 1.0],
    ).eval().float()

    selected = sorted(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("decoder.") or name.startswith("post_quant_conv.")
    )
    with torch.no_grad():
        for parameter_index, (name, parameter) in enumerate(selected):
            flat = torch.arange(parameter.numel(), dtype=torch.float32)
            if name.endswith(".gamma"):
                generated = 1.0 + ((flat % 7) - 3) * 0.01 + (parameter_index + 1) * 0.0001
            elif name.endswith(".bias"):
                generated = ((flat % 11) - 5) * 0.003 + (parameter_index + 1) * 0.0001
            else:
                generated = ((flat % 17) - 8) * 0.004 + (parameter_index + 1) * 0.00005
            parameter.copy_(generated.reshape(parameter.shape))

    checkpoints: dict[str, torch.Tensor] = {}

    def record(name: str):
        def hook(_module, _inputs, output):
            checkpoints[name] = output.detach().cpu().float()

        return hook

    handles = [
        model.post_quant_conv.register_forward_hook(record("post_quant_conv")),
        model.decoder.conv_in.register_forward_hook(record("decoder_conv_in")),
        model.decoder.mid_block.register_forward_hook(record("decoder_mid_block")),
        model.decoder.conv_out.register_forward_hook(record("decoder_conv_out")),
    ]
    for index, block in enumerate(model.decoder.up_blocks):
        handles.append(block.register_forward_hook(record(f"decoder_up_block_{index}")))

    latents = ((torch.arange(8, dtype=torch.float32) % 9) - 4).reshape(1, 2, 1, 2, 2) * 0.1
    with torch.no_grad():
        output = model.decode(latents).sample
    for handle in handles:
        handle.remove()

    tiled_latents = ((torch.arange(18, dtype=torch.float32) % 9) - 4).reshape(1, 2, 1, 3, 3) * 0.1
    model.enable_tiling(
        tile_sample_min_height=4,
        tile_sample_min_width=4,
        tile_sample_stride_height=2,
        tile_sample_stride_width=2,
    )
    with torch.no_grad():
        tiled_output = model.decode(tiled_latents).sample

    model.disable_tiling()
    encoder_selected = sorted(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("encoder.") or name.startswith("quant_conv.")
    )
    with torch.no_grad():
        for parameter_index, (name, parameter) in enumerate(encoder_selected):
            flat = torch.arange(parameter.numel(), dtype=torch.float32)
            if name.endswith(".gamma"):
                generated = 1.0 + ((flat % 7) - 3) * 0.01 + (parameter_index + 1) * 0.0001
            elif name.endswith(".bias"):
                generated = ((flat % 11) - 5) * 0.003 + (parameter_index + 1) * 0.0001
            else:
                generated = ((flat % 17) - 8) * 0.004 + (parameter_index + 1) * 0.00005
            parameter.copy_(generated.reshape(parameter.shape))
    source = ((torch.arange(192, dtype=torch.float32) % 23) - 11).reshape(1, 3, 1, 8, 8) * 0.04
    with torch.no_grad():
        encoded = model.encode(source).latent_dist.mode()

    fixture = {
        "schema_version": 1,
        "versions": {"diffusers": diffusers.__version__, "torch": torch.__version__},
        "config": {
            "_class_name": "AutoencoderKLQwenImage",
            "attn_scales": [],
            "base_dim": 4,
            "dim_mult": [1, 2],
            "dropout": 0.0,
            "input_channels": 3,
            "latents_mean": [0.0, 0.0],
            "latents_std": [1.0, 1.0],
            "num_res_blocks": 1,
            "temperal_downsample": [False],
            "z_dim": 2,
        },
        "parameter_formulas": {
            "gamma": "1 + ((flat_index % 7) - 3) * 0.01 + (sorted_parameter_index + 1) * 0.0001",
            "bias": "((flat_index % 11) - 5) * 0.003 + (sorted_parameter_index + 1) * 0.0001",
            "weight": "((flat_index % 17) - 8) * 0.004 + (sorted_parameter_index + 1) * 0.00005",
        },
        "parameter_selection": ["decoder.*", "post_quant_conv.*"],
        "latent_formula": "((flat_index % 9) - 4) * 0.1",
        "latent_shape": list(latents.shape),
        "latent": flattened(latents),
        "checkpoints": {
            name: sampled(tensor)
            for name, tensor in sorted(checkpoints.items())
        },
        "output_shape": list(output.shape),
        "output": flattened(output),
        "tiled": {
            "latent_formula": "((flat_index % 9) - 4) * 0.1",
            "latent_shape": list(tiled_latents.shape),
            "tiling": {
                "tile_latent_height": 2,
                "tile_latent_width": 2,
                "stride_latent_height": 1,
                "stride_latent_width": 1,
            },
            "output": sampled(tiled_output),
        },
        "encoder": {
            "parameter_selection": ["encoder.*", "quant_conv.*"],
            "parameter_count": len(encoder_selected),
            "source_formula": "((flat_index % 23) - 11) * 0.04",
            "source_shape": list(source.shape),
            "output_shape": list(encoded.shape),
            "output": flattened(encoded),
        },
    }
    print(json.dumps(fixture, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
