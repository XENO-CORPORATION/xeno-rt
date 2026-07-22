#!/usr/bin/env python3
"""Capture a tiny pinned Qwen2.5-VL multimodal language fixture."""

from __future__ import annotations

import json

import torch
import transformers
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLTextConfig,
    Qwen2_5_VLVisionConfig,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLModel,
    Qwen2_5_VLTextModel,
)


def main() -> None:
    torch.set_num_threads(1)
    text_config = Qwen2_5_VLTextConfig(
        vocab_size=16,
        hidden_size=12,
        intermediate_size=20,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        rope_scaling={"type": "default", "mrope_section": [1, 1, 1]},
        use_cache=False,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = Qwen2_5_VLTextModel(text_config).eval().float()
    with torch.no_grad():
        for parameter_index, (_name, parameter) in enumerate(sorted(model.named_parameters())):
            generated = ((torch.arange(parameter.numel(), dtype=torch.float32) % 19) - 9) * 0.004
            generated += (parameter_index + 1) * 0.0001
            parameter.copy_(generated.to(torch.bfloat16).float().reshape(parameter.shape))

    vision_config = Qwen2_5_VLVisionConfig(
        depth=1,
        hidden_size=12,
        hidden_act="silu",
        intermediate_size=20,
        num_heads=2,
        in_channels=3,
        patch_size=2,
        spatial_merge_size=2,
        temporal_patch_size=2,
        window_size=8,
        out_hidden_size=12,
        fullatt_block_indexes=[0],
    )
    combined_config = Qwen2_5_VLConfig(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=15,
        video_token_id=12,
        vision_start_token_id=14,
        vision_end_token_id=13,
    )
    position_model = Qwen2_5_VLModel(combined_config).eval()
    token_ids = torch.tensor([[1, 4, 15, 15, 5, 2]], dtype=torch.long)
    token_types = torch.tensor([[0, 0, 1, 1, 0, 0]], dtype=torch.int32)
    attention_mask = torch.ones_like(token_ids)
    image_grid = torch.tensor([[1, 2, 4]], dtype=torch.long)
    position_ids, _ = position_model.get_rope_index(
        token_ids,
        mm_token_type_ids=token_types,
        image_grid_thw=image_grid,
        attention_mask=attention_mask,
    )
    vision_values = ((torch.arange(24, dtype=torch.float32) % 11) - 5) * 0.02
    with torch.no_grad():
        embeddings = model.embed_tokens(token_ids).clone()
        embeddings[0, 2:4] = vision_values.reshape(2, 12)
        output = model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
        ).last_hidden_state

    fixture = {
        "schema_version": 1,
        "versions": {"transformers": transformers.__version__, "torch": torch.__version__},
        "config": {
            "hidden_size": 12,
            "intermediate_size": 20,
            "max_position_embeddings": 128,
            "mrope_section": [1, 1, 1],
            "num_attention_heads": 2,
            "num_hidden_layers": 2,
            "num_key_value_heads": 1,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10_000.0,
            "vocab_size": 16,
        },
        "parameter_count": len(list(model.named_parameters())),
        "parameter_formula": "bf16(((flat_index % 19) - 9) * 0.004 + (sorted_parameter_index + 1) * 0.0001)",
        "token_ids": token_ids.reshape(-1).tolist(),
        "image_grid": image_grid.reshape(-1).tolist(),
        "image_token_counts": [2],
        "vision_formula": "((flat_index % 11) - 5) * 0.02",
        "position_ids": position_ids[:, 0, :].tolist(),
        "output_shape": list(output.shape),
        "output": output.detach().cpu().float().reshape(-1).tolist(),
    }
    print(json.dumps(fixture, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
