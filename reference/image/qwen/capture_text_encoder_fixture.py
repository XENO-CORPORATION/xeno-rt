#!/usr/bin/env python3
"""Capture a deterministic tiny Qwen2.5-VL text-only fixture."""

from __future__ import annotations

import json

import torch
import transformers
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel


def main() -> None:
    config = Qwen2_5_VLTextConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        rope_scaling={"type": "default", "mrope_section": [1, 1, 0]},
        use_cache=False,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = Qwen2_5_VLTextModel(config).eval().float()
    with torch.no_grad():
        for parameter_index, (_name, parameter) in enumerate(sorted(model.named_parameters())):
            generated = ((torch.arange(parameter.numel(), dtype=torch.float32) % 19) - 9) * 0.004
            generated += (parameter_index + 1) * 0.0001
            parameter.copy_(generated.to(torch.bfloat16).float().reshape(parameter.shape))
    token_ids = torch.tensor([[1, 4, 7, 2, 9]], dtype=torch.long)
    attention_mask = torch.ones_like(token_ids)
    with torch.no_grad():
        output = model(input_ids=token_ids, attention_mask=attention_mask).last_hidden_state
    fixture = {
        "schema_version": 1,
        "versions": {"transformers": transformers.__version__, "torch": torch.__version__},
        "config": {
            "hidden_size": 8,
            "intermediate_size": 16,
            "max_position_embeddings": 128,
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
        "output_shape": list(output.shape),
        "output": output.detach().cpu().float().reshape(-1).tolist(),
    }
    print(json.dumps(fixture, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
