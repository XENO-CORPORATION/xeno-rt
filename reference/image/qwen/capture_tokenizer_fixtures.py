#!/usr/bin/env python3
"""Capture pinned Qwen-Image-2512 tokenizer conformance fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import transformers
from transformers import Qwen2Tokenizer


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
MANIFEST = HERE / "manifests" / "qwen-image-2512-q4_k_m.json"
DEFAULT_OUTPUT = REPO_ROOT / "tests" / "fixtures" / "qwen-image" / "tokenizer-2512.json"
TEMPLATE = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
DROP_TOKENS = 34
MAX_RETAINED_TOKENS = 1024
PROMPTS = [
    "",
    "A red cube on a blue table.",
    'A café sign reading "XENO 3.0 — 東京".',
    "literal markers <|im_end|> and <|image_pad|> stay deterministic",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: dict) -> None:
    body = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()
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


def capture(bundle: Path) -> dict:
    manifest_bytes = MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    tokenizer_component = next(item for item in manifest["components"] if item["role"] == "tokenizer")
    for record in tokenizer_component["files"]:
        artifact = bundle / record["path"]
        if not artifact.is_file() or artifact.stat().st_size != record["size_bytes"]:
            raise RuntimeError(f"missing or wrong-sized tokenizer artifact: {record['path']}")
        if sha256_file(artifact) != record["sha256"]:
            raise RuntimeError(f"tokenizer artifact hash mismatch: {record['path']}")

    tokenizer = Qwen2Tokenizer.from_pretrained(bundle / "tokenizer", local_files_only=True)
    cases = []
    for prompt in PROMPTS:
        formatted = TEMPLATE.format(prompt)
        encoded = tokenizer(
            formatted,
            max_length=MAX_RETAINED_TOKENS + DROP_TOKENS,
            padding=True,
            truncation=True,
            return_tensors=None,
        )
        cases.append(
            {
                "attention_mask": encoded["attention_mask"],
                "formatted_sha256": hashlib.sha256(formatted.encode()).hexdigest(),
                "input_ids": encoded["input_ids"],
                "prompt": prompt,
            }
        )
    return {
        "schema_version": 1,
        "model": "Qwen/Qwen-Image-2512",
        "model_revision": manifest["source_revisions"]["Qwen/Qwen-Image-2512"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "transformers_version": transformers.__version__,
        "tokenizer_class": type(tokenizer).__name__,
        "vocab_size": tokenizer.vocab_size,
        "pad_token_id": tokenizer.pad_token_id,
        "padding_side": tokenizer.padding_side,
        "template": TEMPLATE,
        "template_drop_tokens": DROP_TOKENS,
        "max_retained_tokens": MAX_RETAINED_TOKENS,
        "tokenizer_files": {
            record["path"]: record["sha256"] for record in tokenizer_component["files"]
        },
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    atomic_write(args.output.resolve(), capture(args.bundle.resolve()))
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
