#!/usr/bin/env python3
"""Capture a deterministic tiny Qwen2.5-VL visual-tower fixture."""

from __future__ import annotations

import json

import torch
import transformers
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLVisionConfig,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
)


def main() -> None:
    torch.set_num_threads(1)
    config = Qwen2_5_VLVisionConfig(
        depth=2,
        hidden_size=8,
        hidden_act="silu",
        intermediate_size=16,
        num_heads=2,
        in_channels=3,
        patch_size=2,
        spatial_merge_size=2,
        temporal_patch_size=2,
        window_size=8,
        out_hidden_size=6,
        fullatt_block_indexes=[1],
    )
    config._attn_implementation = "eager"
    model = Qwen2_5_VisionTransformerPretrainedModel(config).eval().float()
    with torch.no_grad():
        for parameter_index, (_name, parameter) in enumerate(sorted(model.named_parameters())):
            generated = ((torch.arange(parameter.numel(), dtype=torch.float32) % 23) - 11) * 0.003
            generated += (parameter_index + 1) * 0.0001
            parameter.copy_(generated.to(torch.bfloat16).float().reshape(parameter.shape))

    grids = torch.tensor([[1, 4, 8], [1, 8, 4]], dtype=torch.long)
    patch_features = 3 * 2 * 2 * 2
    patch_rows = int(torch.prod(grids, dim=1).sum().item())
    pixels = ((torch.arange(patch_rows * patch_features, dtype=torch.float32) % 29) - 14) * 0.01
    pixels = pixels.reshape(patch_rows, patch_features)
    with torch.no_grad():
        result = model(hidden_states=pixels, grid_thw=grids)

    fixture = {
        "schema_version": 1,
        "versions": {"transformers": transformers.__version__, "torch": torch.__version__},
        "config": {
            "depth": 2,
            "fullatt_block_indexes": [1],
            "hidden_act": "silu",
            "hidden_size": 8,
            "in_channels": 3,
            "intermediate_size": 16,
            "num_heads": 2,
            "out_hidden_size": 6,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "window_size": 8,
        },
        "parameter_count": len(list(model.named_parameters())),
        "parameter_formula": "bf16(((flat_index % 23) - 11) * 0.003 + (sorted_parameter_index + 1) * 0.0001)",
        "pixel_formula": "((flat_index % 29) - 14) * 0.01",
        "grids": grids.tolist(),
        "pixel_shape": list(pixels.shape),
        "output_shape": list(result.pooler_output.shape),
        "output": result.pooler_output.detach().cpu().float().reshape(-1).tolist(),
    }
    print(json.dumps(fixture, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
