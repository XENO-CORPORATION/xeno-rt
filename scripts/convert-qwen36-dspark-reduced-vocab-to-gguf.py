#!/usr/bin/env python3
"""Convert a reduced-vocabulary Qwen3.6 DSpark checkpoint for XRT.

The source SafeTensors artifact keeps target-space embeddings but predicts in
a smaller draft vocabulary. XRT reuses the target model's token embeddings, so
the GGUF intentionally omits ``embed_tokens.weight`` and retains the trained
``lm_head.weight``, ``markov_w2.weight``, and exact draft-to-target ``d2t`` map.
The two reduced-vocabulary projections are zero-padded to a 64-row boundary so
the resident Q8_0 Marlin path can execute them; argmax remains bounded to the
logical draft vocabulary recorded by ``d2t``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gguf
import numpy as np
from safetensors import safe_open


ARCHITECTURE = "dflash"
MARLIN_OUTPUT_TILE = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Source model.safetensors")
    parser.add_argument("config", type=Path, help="Source config.json")
    parser.add_argument("output", type=Path, help="Destination GGUF")
    parser.add_argument(
        "--linear-type",
        choices=("q8_0", "f16"),
        default="q8_0",
        help="Storage type for two-dimensional linears (default: q8_0)",
    )
    return parser.parse_args()


def tensor_names(layer_count: int) -> dict[str, str | None]:
    names: dict[str, str | None] = {
        "d2t": "d2t",
        "embed_tokens.weight": None,
        "fc.weight": "fc.weight",
        "hidden_norm.weight": "enc.output_norm.weight",
        "lm_head.weight": "lm_head.weight",
        "markov_head.markov_w1.weight": "markov_w1.weight",
        "markov_head.markov_w2.weight": "markov_w2.weight",
        "confidence_head.proj.weight": "conf_proj.weight",
        "confidence_head.proj.bias": "conf_proj.bias",
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
    rope_parameters = config["rope_parameters"]
    assert isinstance(rope_parameters, dict)
    target_layers = [int(layer) + 1 for layer in config["target_layer_ids"]]
    writer.add_name(f"Qwen3.6-27B DSpark {int(config['draft_vocab_size']) // 1000}k")
    writer.add_file_type(
        gguf.LlamaFileType.MOSTLY_Q8_0
        if linear_type == "q8_0"
        else gguf.LlamaFileType.MOSTLY_F16
    )
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    writer.add_uint32("dflash.embedding_length", int(config["hidden_size"]))
    writer.add_uint32("dflash.block_count", int(config["num_hidden_layers"]))
    writer.add_uint32("dflash.feed_forward_length", int(config["intermediate_size"]))
    writer.add_uint32("dflash.attention.head_count", int(config["num_attention_heads"]))
    writer.add_uint32(
        "dflash.attention.head_count_kv", int(config["num_key_value_heads"])
    )
    writer.add_uint32("dflash.attention.key_length", int(config["head_dim"]))
    writer.add_float32(
        "dflash.attention.layer_norm_rms_epsilon", float(config["rms_norm_eps"])
    )
    writer.add_float32("dflash.rope.freq_base", float(rope_parameters["rope_theta"]))
    writer.add_uint32("dflash.block_size", int(config["block_size"]))
    writer.add_array("dflash.target_layers", target_layers)
    writer.add_uint32("tokenizer.ggml.mask_token_id", int(config["mask_token_id"]))
    writer.add_uint32("dflash.draft_vocab_size", int(config["draft_vocab_size"]))


def padded_projection(tensor: np.ndarray, source_name: str) -> np.ndarray:
    if source_name not in {"lm_head.weight", "markov_head.markov_w2.weight"}:
        return tensor
    padded_rows = (
        (tensor.shape[0] + MARLIN_OUTPUT_TILE - 1) // MARLIN_OUTPUT_TILE
    ) * MARLIN_OUTPUT_TILE
    if padded_rows == tensor.shape[0]:
        return tensor
    result = np.zeros((padded_rows, tensor.shape[1]), dtype=tensor.dtype)
    result[: tensor.shape[0]] = tensor
    print(
        f"PAD  {source_name} rows {tensor.shape[0]} -> {padded_rows}",
        flush=True,
    )
    return result


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    draft_vocab_size = int(config["draft_vocab_size"])
    target_vocab_size = int(config["vocab_size"])
    if not 0 < draft_vocab_size < target_vocab_size:
        raise ValueError(
            f"expected reduced draft vocabulary, found draft={draft_vocab_size} "
            f"target={target_vocab_size}"
        )
    if bool(config.get("mtp_use_dedicated_embeddings", True)):
        raise ValueError("XRT reduced-vocabulary conversion requires shared target embeddings")

    mappings = tensor_names(int(config["num_hidden_layers"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(args.output, ARCHITECTURE, use_temp_file=True)
    add_metadata(writer, config, args.linear_type)

    with safe_open(args.checkpoint, framework="pt", device="cpu") as source:
        available = set(source.keys())
        missing = sorted(set(mappings) - available)
        extra = sorted(available - set(mappings))
        if missing or extra:
            raise RuntimeError(
                f"unexpected checkpoint tensors: missing={missing}, extra={extra}"
            )
        d2t_offsets = source.get_tensor("d2t").numpy()
        if d2t_offsets.shape != (draft_vocab_size,):
            raise ValueError(
                f"d2t shape {d2t_offsets.shape} does not match draft vocab {draft_vocab_size}"
            )
        # The published checkpoint stores an offset map, not absolute IDs:
        # target_id = draft_id + d2t[draft_id]. Convert it once so XRT's hot
        # device path needs only one lookup and no integer addition kernel.
        absolute_d2t = d2t_offsets.astype(np.int64) + np.arange(
            draft_vocab_size, dtype=np.int64
        )
        if np.any(absolute_d2t < 0) or np.any(absolute_d2t >= target_vocab_size):
            raise ValueError("d2t resolves a token outside the target vocabulary")
        if np.unique(absolute_d2t).size != absolute_d2t.size:
            raise ValueError("d2t resolves duplicate target token IDs")

        for source_name, target_name in mappings.items():
            if target_name is None:
                print(f"SKIP {source_name} (shared target embeddings)", flush=True)
                continue
            if source_name == "d2t":
                tensor = np.ascontiguousarray(absolute_d2t, dtype=np.float32)
                writer.add_tensor(target_name, tensor)
                print(f"F32  {source_name} -> {target_name} {tuple(tensor.shape)}", flush=True)
                continue
            tensor = source.get_tensor(source_name).float().numpy()
            if tensor.ndim == 1:
                tensor = np.ascontiguousarray(tensor, dtype=np.float32)
                writer.add_tensor(target_name, tensor)
                print(f"F32  {source_name} -> {target_name} {tuple(tensor.shape)}", flush=True)
                continue

            tensor = padded_projection(np.ascontiguousarray(tensor), source_name)
            if args.linear_type == "f16":
                tensor = np.ascontiguousarray(tensor, dtype=np.float16)
                writer.add_tensor(target_name, tensor)
                print(f"F16  {source_name} -> {target_name} {tuple(tensor.shape)}", flush=True)
            else:
                quantized = gguf.quantize(
                    np.ascontiguousarray(tensor), gguf.GGMLQuantizationType.Q8_0
                )
                writer.add_tensor(
                    target_name,
                    quantized,
                    raw_dtype=gguf.GGMLQuantizationType.Q8_0,
                )
                print(f"Q8_0 {source_name} -> {target_name} {tuple(tensor.shape)}", flush=True)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
