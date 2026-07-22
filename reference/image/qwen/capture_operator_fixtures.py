#!/usr/bin/env python3
"""Capture small Qwen Image operator fixtures from the pinned Diffusers env."""

from __future__ import annotations

import argparse
import json

import diffusers
import torch
from diffusers.models.transformers.transformer_qwenimage import (
    QwenEmbedRope,
    QwenImageTransformerBlock,
    QwenImageTransformer2DModel,
    get_timestep_embedding,
)
from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline


def values(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().cpu().reshape(-1).float().tolist()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture deterministic Qwen Image operator fixtures from pinned Diffusers."
    )
    parser.add_argument(
        "--section",
        choices=("all", "block", "full_transformer", "edit_transformer"),
        default="all",
        help="Print the complete fixture or only the compact transformer block fixture.",
    )
    args = parser.parse_args()

    latents = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    packed = QwenImagePipeline._pack_latents(latents, 1, 1, 4, 4)

    timestep_input = torch.tensor([0.0, 0.125, 1.0], dtype=torch.float32)
    timestep = get_timestep_embedding(
        timestep_input,
        embedding_dim=256,
        flip_sin_to_cos=True,
        downscale_freq_shift=0,
        scale=1000,
    )
    timestep_sample_indices = [0, 1, 63, 127, 128, 129, 191, 255]

    rope = QwenEmbedRope(theta=10_000, axes_dim=[2, 2, 2], scale_rope=True)
    image_rope, text_rope = rope((1, 2, 2), device=torch.device("cpu"), max_txt_seq_len=2)

    block = QwenImageTransformerBlock(dim=12, num_attention_heads=2, attention_head_dim=6).eval().float()
    parameter_specs = []
    with torch.no_grad():
        for parameter_index, (name, parameter) in enumerate(sorted(block.named_parameters())):
            generated = ((torch.arange(parameter.numel(), dtype=torch.float32) % 23) - 11) * 0.003
            generated += (parameter_index + 1) * 0.0002
            parameter.copy_(generated.reshape(parameter.shape))
            parameter_specs.append({"name": name, "shape": list(parameter.shape)})
    block_image_input = ((torch.arange(48, dtype=torch.float32) % 13) - 6).reshape(1, 4, 12) * 0.05
    block_text_input = ((torch.arange(24, dtype=torch.float32) % 11) - 5).reshape(1, 2, 12) * 0.04
    block_temb = ((torch.arange(12, dtype=torch.float32) % 7) - 3).reshape(1, 12) * 0.03
    block_mask = torch.tensor([[True, False]], dtype=torch.bool)
    block_text_output, block_image_output = block(
        hidden_states=block_image_input,
        encoder_hidden_states=block_text_input,
        encoder_hidden_states_mask=block_mask,
        temb=block_temb,
        image_rotary_emb=(image_rope, text_rope),
    )
    block_image_sample_indices = [0, 1, 5, 11, 12, 23, 24, 35, 36, 47]
    block_text_sample_indices = [0, 1, 5, 11, 12, 17, 23]

    full_transformer = QwenImageTransformer2DModel(
        patch_size=2,
        in_channels=4,
        out_channels=1,
        num_layers=2,
        attention_head_dim=6,
        num_attention_heads=2,
        joint_attention_dim=8,
        guidance_embeds=False,
        axes_dims_rope=(2, 2, 2),
    ).eval().float()
    with torch.no_grad():
        for parameter_index, (_name, parameter) in enumerate(sorted(full_transformer.named_parameters())):
            generated = ((torch.arange(parameter.numel(), dtype=torch.float32) % 19) - 9) * 0.004
            generated += (parameter_index + 1) * 0.0001
            parameter.copy_(generated.to(torch.bfloat16).float().reshape(parameter.shape))
    full_image_input = ((torch.arange(16, dtype=torch.float32) % 9) - 4).reshape(1, 4, 4) * 0.07
    full_text_input = ((torch.arange(16, dtype=torch.float32) % 7) - 3).reshape(1, 2, 8) * 0.05
    full_text_mask = torch.tensor([[True, False]], dtype=torch.bool)
    full_timestep = torch.tensor([0.125], dtype=torch.float32)
    with torch.no_grad():
        full_output = full_transformer(
            hidden_states=full_image_input,
            encoder_hidden_states=full_text_input,
            encoder_hidden_states_mask=full_text_mask,
            timestep=full_timestep,
            img_shapes=[(1, 2, 2)],
        ).sample

    edit_transformer = QwenImageTransformer2DModel(
        patch_size=2,
        in_channels=4,
        out_channels=1,
        num_layers=2,
        attention_head_dim=6,
        num_attention_heads=2,
        joint_attention_dim=8,
        guidance_embeds=False,
        axes_dims_rope=(2, 2, 2),
        zero_cond_t=True,
    ).eval().float()
    with torch.no_grad():
        for parameter_index, (_name, parameter) in enumerate(sorted(edit_transformer.named_parameters())):
            generated = ((torch.arange(parameter.numel(), dtype=torch.float32) % 19) - 9) * 0.004
            generated += (parameter_index + 1) * 0.0001
            parameter.copy_(generated.to(torch.bfloat16).float().reshape(parameter.shape))
    edit_image_input = ((torch.arange(24, dtype=torch.float32) % 9) - 4).reshape(1, 6, 4) * 0.07
    with torch.no_grad():
        edit_output = edit_transformer(
            hidden_states=edit_image_input,
            encoder_hidden_states=full_text_input,
            encoder_hidden_states_mask=full_text_mask,
            timestep=full_timestep,
            img_shapes=[[(1, 2, 2), (1, 1, 2)]],
        ).sample

    fixture = {
        "schema_version": 1,
        "versions": {"diffusers": diffusers.__version__, "torch": torch.__version__},
        "pack": {
            "input_shape": list(latents.shape),
            "input": values(latents),
            "output_shape": list(packed.shape),
            "output": values(packed),
        },
        "timestep": {
            "input": values(timestep_input),
            "shape": list(timestep.shape),
            "sample_indices": timestep_sample_indices,
            "sample_rows": [
                [float(timestep[row, index]) for index in timestep_sample_indices]
                for row in range(timestep.shape[0])
            ],
        },
        "rope": {
            "axes_dims": [2, 2, 2],
            "image_shape": [1, 2, 2],
            "text_sequence_length": 2,
            "image_shape_complex": list(image_rope.shape),
            "image_real": values(image_rope.real),
            "image_imag": values(image_rope.imag),
            "text_shape_complex": list(text_rope.shape),
            "text_real": values(text_rope.real),
            "text_imag": values(text_rope.imag),
        },
        "block": {
            "batch": 1,
            "image_sequence": 4,
            "text_sequence": 2,
            "heads": 2,
            "head_dim": 6,
            "parameter_formula": "((flat_index % 23) - 11) * 0.003 + (sorted_parameter_index + 1) * 0.0002",
            "parameters": parameter_specs,
            "image_input_formula": "((flat_index % 13) - 6) * 0.05",
            "text_input_formula": "((flat_index % 11) - 5) * 0.04",
            "text_mask": [int(value) for value in block_mask.reshape(-1).tolist()],
            "timestep_embedding_formula": "((flat_index % 7) - 3) * 0.03",
            "image_output_sample_indices": block_image_sample_indices,
            "image_output_samples": [float(block_image_output.reshape(-1)[index]) for index in block_image_sample_indices],
            "text_output_sample_indices": block_text_sample_indices,
            "text_output_samples": [float(block_text_output.reshape(-1)[index]) for index in block_text_sample_indices],
        },
        "full_transformer": {
            "config": {
                "_class_name": "QwenImageTransformer2DModel",
                "attention_head_dim": 6,
                "axes_dims_rope": [2, 2, 2],
                "guidance_embeds": False,
                "in_channels": 4,
                "joint_attention_dim": 8,
                "num_attention_heads": 2,
                "num_layers": 2,
                "out_channels": 1,
                "patch_size": 2,
                "zero_cond_t": False,
                "use_additional_t_cond": False,
                "use_layer3d_rope": False,
            },
            "parameter_count": len(list(full_transformer.named_parameters())),
            "parameter_formula": "bf16(((flat_index % 19) - 9) * 0.004 + (sorted_parameter_index + 1) * 0.0001)",
            "image_input_formula": "((flat_index % 9) - 4) * 0.07",
            "text_input_formula": "((flat_index % 7) - 3) * 0.05",
            "text_mask": [1, 0],
            "timestep": [0.125],
            "image_shape": [1, 2, 2],
            "output_shape": list(full_output.shape),
            "output": values(full_output),
        },
        "edit_transformer": {
            "config": {
                "_class_name": "QwenImageTransformer2DModel",
                "attention_head_dim": 6,
                "axes_dims_rope": [2, 2, 2],
                "guidance_embeds": False,
                "in_channels": 4,
                "joint_attention_dim": 8,
                "num_attention_heads": 2,
                "num_layers": 2,
                "out_channels": 1,
                "patch_size": 2,
                "zero_cond_t": True,
                "use_additional_t_cond": False,
                "use_layer3d_rope": False,
            },
            "parameter_count": len(list(edit_transformer.named_parameters())),
            "parameter_formula": "bf16(((flat_index % 19) - 9) * 0.004 + (sorted_parameter_index + 1) * 0.0001)",
            "image_input_formula": "((flat_index % 9) - 4) * 0.07",
            "text_input_formula": "((flat_index % 7) - 3) * 0.05",
            "text_mask": [1, 0],
            "timestep": [0.125],
            "image_shapes": [[1, 2, 2], [1, 1, 2]],
            "output_shape": list(edit_output.shape),
            "output": values(edit_output),
        },
    }
    output = fixture if args.section == "all" else fixture[args.section]
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
