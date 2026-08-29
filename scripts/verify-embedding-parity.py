#!/usr/bin/env python3
"""Compare xrt-server embeddings with an independent ONNX/tokenizer execution."""

from __future__ import annotations

import argparse
import json
import math
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
OUTPUT_DIMENSIONS = 512
PREFIXES = {
    "query": "search_query: ",
    "document": "search_document: ",
}
FIXTURES = {
    "query": "a red bicycle beside a tree",
    "document": "a red bicycle beside a tree",
}


def direct_embeddings(bundle_dir: Path, task: str, texts: list[str]) -> np.ndarray:
    tokenizer = Tokenizer.from_file(str(bundle_dir / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=8192, direction="right")
    encodings = tokenizer.encode_batch(
        [PREFIXES[task] + text for text in texts], add_special_tokens=True
    )
    sequence_length = max(len(encoding.ids) for encoding in encodings)
    input_ids = np.zeros((len(encodings), sequence_length), dtype=np.int64)
    token_type_ids = np.zeros_like(input_ids)
    attention_mask = np.zeros_like(input_ids)
    for row, encoding in enumerate(encodings):
        width = len(encoding.ids)
        input_ids[row, :width] = encoding.ids
        token_type_ids[row, :width] = encoding.type_ids
        attention_mask[row, :width] = 1

    session = ort.InferenceSession(
        str(bundle_dir / "model_quantized.onnx"),
        providers=["CPUExecutionProvider"],
    )
    hidden = session.run(
        ["last_hidden_state"],
        {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
        },
    )[0]
    mask = attention_mask.astype(np.float32)[..., None]
    pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
    mean = pooled.mean(axis=1, keepdims=True)
    variance = ((pooled - mean) ** 2).mean(axis=1, keepdims=True)
    projected = ((pooled - mean) / np.sqrt(variance + 1e-5))[:, :OUTPUT_DIMENSIONS]
    return projected / np.linalg.norm(projected, axis=1, keepdims=True)


def server_embedding(base_url: str, api_key: str, task: str, text: str) -> np.ndarray:
    body = json.dumps(
        {
            "model": MODEL_ID,
            "input": text,
            "task": task,
            "dimensions": OUTPUT_DIMENSIONS,
            "encoding_format": "float",
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/embeddings",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(
            f"embedding endpoint returned HTTP {error.code}: {error.read().decode('utf-8')}"
        ) from error
    contract = payload.get("xeno_contract", {})
    if payload.get("model") != MODEL_ID or contract.get("output_dimensions") != 512:
        raise RuntimeError(f"server returned the wrong embedding contract: {contract!r}")
    return np.asarray(payload["data"][0]["embedding"], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    args = parser.parse_args()

    results: dict[str, dict[str, float]] = {}
    server_vectors: dict[str, np.ndarray] = {}
    for task, text in FIXTURES.items():
        expected = direct_embeddings(args.bundle_dir, task, [text])[0]
        actual = server_embedding(args.base_url, args.api_key, task, text)
        if actual.shape != (OUTPUT_DIMENSIONS,):
            raise RuntimeError(f"{task} returned shape {actual.shape}, expected (512,)")
        maximum_absolute_error = float(np.max(np.abs(expected - actual)))
        cosine = float(np.dot(expected, actual))
        norm = float(np.linalg.norm(actual))
        if maximum_absolute_error > 5e-5 or cosine < 0.999999 or abs(norm - 1.0) > 1e-5:
            raise RuntimeError(
                f"{task} parity failed: max_abs={maximum_absolute_error:.9g}, "
                f"cosine={cosine:.9g}, norm={norm:.9g}"
            )
        results[task] = {
            "max_abs": maximum_absolute_error,
            "cosine": cosine,
            "norm": norm,
        }
        server_vectors[task] = actual

    task_cosine = float(np.dot(server_vectors["query"], server_vectors["document"]))
    if not math.isfinite(task_cosine) or task_cosine >= 0.999:
        raise RuntimeError(
            f"task-prefix fixture did not distinguish query/document vectors: {task_cosine}"
        )
    print(json.dumps({"parity": results, "query_document_cosine": task_cosine}, indent=2))


if __name__ == "__main__":
    main()
