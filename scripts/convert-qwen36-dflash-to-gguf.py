#!/usr/bin/env python3
"""Convert the official Qwen3.6-27B DFlash checkpoint to XRT's GGUF contract.

The official checkpoint stores BF16 transformer weights in SafeTensors. XRT's
admitted CUDA drafter keeps normalization vectors in F32 and supports either
Q8_0 or F16 linears. The resulting artifact is auxiliary to the ordinary
Qwen3.6 target GGUF and does not contain tokenizer or output-head weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gguf
import numpy as np
from safetensors import safe_open


ARCHITECTURE = "qwen35-dflash-draft"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Official model.safetensors path")
    parser.add_argument("config", type=Path, help="Official config.json path")
    parser.add_argument("output", type=Path, help="Destination GGUF path")
    parser.add_argument(
        "--linear-type",
        choices=("q8_0", "f16"),
        default="q8_0",
        help="Storage type for linear weights (default: q8_0)",
    )
    return parser.parse_args()


def tensor_names(layer_count: int) -> dict[str, str]:
    names = {
        "fc.weight": "dflash.fc.weight",
        "hidden_norm.weight": "dflash.hidden_norm.weight",
        "norm.weight": "output_norm.weight",
    }
    for layer in range(layer_count):
        source = f"layers.{layer}"
        target = f"blk.{layer}"
        names.update(
            {
                f"{source}.input_layernorm.weight": f"{target}.attn_norm.weight",
                f"{source}.post_attention_layernorm.weight": f"{target}.ffn_norm.weight",
                f"{source}.self_attn.q_norm.weight": f"{target}.attn_q_norm.weight",
                f"{source}.self_attn.k_norm.weight": f"{target}.attn_k_norm.weight",
                f"{source}.self_attn.q_proj.weight": f"{target}.attn_q.weight",
                f"{source}.self_attn.k_proj.weight": f"{target}.attn_k.weight",
                f"{source}.self_attn.v_proj.weight": f"{target}.attn_v.weight",
                f"{source}.self_attn.o_proj.weight": f"{target}.attn_output.weight",
                f"{source}.mlp.gate_proj.weight": f"{target}.ffn_gate.weight",
                f"{source}.mlp.up_proj.weight": f"{target}.ffn_up.weight",
                f"{source}.mlp.down_proj.weight": f"{target}.ffn_down.weight",
            }
        )
    return names


def add_metadata(
    writer: gguf.GGUFWriter, config: dict[str, object], linear_type: str
) -> None:
    dflash = config["dflash_config"]
    assert isinstance(dflash, dict)
    prefix = ARCHITECTURE
    writer.add_name(f"Qwen3.6-27B DFlash {linear_type.upper()}")
    file_type = (
        gguf.LlamaFileType.MOSTLY_Q8_0
        if linear_type == "q8_0"
        else gguf.LlamaFileType.MOSTLY_F16
    )
    writer.add_file_type(file_type)
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    writer.add_uint32(f"{prefix}.embedding_length", int(config["hidden_size"]))
    writer.add_uint32(f"{prefix}.block_count", int(config["num_hidden_layers"]))
    writer.add_uint32(f"{prefix}.feed_forward_length", int(config["intermediate_size"]))
    writer.add_uint32(f"{prefix}.attention.head_count", int(config["num_attention_heads"]))
    writer.add_uint32(
        f"{prefix}.attention.head_count_kv", int(config["num_key_value_heads"])
    )
    writer.add_uint32(f"{prefix}.attention.key_length", int(config["head_dim"]))
    writer.add_uint32(f"{prefix}.vocab_size", int(config["vocab_size"]))
    writer.add_float32(
        f"{prefix}.attention.layer_norm_rms_epsilon", float(config["rms_norm_eps"])
    )
    writer.add_float32(f"{prefix}.rope.freq_base", float(config["rope_theta"]))
    writer.add_uint32(f"{prefix}.dflash.block_size", int(config["block_size"]))
    writer.add_uint32(f"{prefix}.dflash.mask_token_id", int(dflash["mask_token_id"]))
    target_layers = [int(layer) for layer in dflash["target_layer_ids"]]
    writer.add_uint32(f"{prefix}.dflash.n_target_layers", len(target_layers))
    writer.add_array(f"{prefix}.dflash.target_layer_ids", target_layers)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    mappings = tensor_names(int(config["num_hidden_layers"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(args.output, ARCHITECTURE, use_temp_file=True)
    add_metadata(writer, config, args.linear_type)

    with safe_open(args.checkpoint, framework="pt", device="cpu") as source:
        available = set(source.keys())
        missing = sorted(set(mappings) - available)
        extra = sorted(available - set(mappings))
        if missing or extra:
            raise RuntimeError(f"unexpected checkpoint tensors: missing={missing}, extra={extra}")
        for source_name, target_name in mappings.items():
            tensor = source.get_tensor(source_name).float().numpy()
            if tensor.ndim == 1:
                writer.add_tensor(target_name, np.ascontiguousarray(tensor, dtype=np.float32))
                print(f"F32  {source_name} -> {target_name} {tuple(tensor.shape)}", flush=True)
                continue
            if args.linear_type == "f16":
                writer.add_tensor(target_name, np.ascontiguousarray(tensor, dtype=np.float16))
                print(
                    f"F16  {source_name} -> {target_name} {tuple(tensor.shape)}",
                    flush=True,
                )
            else:
                quantized = gguf.quantize(
                    np.ascontiguousarray(tensor), gguf.GGMLQuantizationType.Q8_0
                )
                writer.add_tensor(
                    target_name,
                    quantized,
                    raw_dtype=gguf.GGMLQuantizationType.Q8_0,
                )
                print(
                    f"Q8_0 {source_name} -> {target_name} {tuple(tensor.shape)}",
                    flush=True,
                )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
