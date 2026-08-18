#!/usr/bin/env python3
"""Build an offline DSpark prefix-scheduler profile from measured XRT runs.

The runtime profile contains the median full-block draft latency and the median
target verification latency for proposal prefix lengths 0..15. Length zero is
the ordinary target-only decode step. Every other point comes from an exact
token-parity candidate arm with that fixed proposal depth.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def measured_rows(path: Path, warmups: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        row
        for row in payload.get("results", [])
        if int(row.get("repetition") or 0) > warmups and not row.get("error")
    ]
    if not rows:
        raise ValueError(f"{path} has no successful post-warmup rows")
    return rows


def median_target_step_micros(rows: list[dict[str, Any]]) -> float:
    values = []
    for row in rows:
        tokens = int(row.get("decode_tokens") or 0)
        millis = float(row.get("decode_ms") or 0.0)
        if tokens > 0 and millis > 0:
            values.append(millis * 1000.0 / tokens)
    if not values:
        raise ValueError("target-only rows do not contain positive decode timings")
    return statistics.median(values)


def median_speculative_micros(
    rows: list[dict[str, Any]], field: str, source: Path
) -> float:
    values = []
    for row in rows:
        speculative = row.get("speculative_decode") or {}
        cycles = int(speculative.get("verification_batches") or 0)
        micros = float(speculative.get(field) or 0.0)
        if cycles > 0 and micros > 0:
            values.append(micros / cycles)
    if not values:
        raise ValueError(f"{source} does not contain positive {field} cycle timings")
    return statistics.median(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--warmup-repetitions", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=15)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmup_repetitions < 0:
        parser.error("--warmup-repetitions cannot be negative")
    if not 1 <= args.max_depth <= 15:
        parser.error("--max-depth must be between 1 and 15")

    target_path = args.directory / "target-only.json"
    fixed_path = args.directory / "candidate-fixed.json"
    target_rows = measured_rows(target_path, args.warmup_repetitions)
    fixed_rows = measured_rows(fixed_path, args.warmup_repetitions)
    verify_micros = [median_target_step_micros(target_rows)]
    sources: list[dict[str, Any]] = [
        {
            "prefix_length": 0,
            "path": str(target_path),
            "samples": len(target_rows),
            "median_verify_micros": verify_micros[0],
        }
    ]

    for depth in range(1, args.max_depth + 1):
        path = fixed_path if depth == args.max_depth else args.directory / f"candidate-depth-{depth}.json"
        rows = measured_rows(path, args.warmup_repetitions)
        value = median_speculative_micros(rows, "verify_micros", path)
        verify_micros.append(value)
        sources.append(
            {
                "prefix_length": depth,
                "path": str(path),
                "samples": len(rows),
                "median_verify_micros": value,
            }
        )

    draft_micros = median_speculative_micros(fixed_rows, "draft_micros", fixed_path)
    verify_csv = ",".join(f"{value:.6f}" for value in verify_micros)
    report = {
        "object": "xrt.qwen36_dspark_hardware_profile",
        "method": "median post-warmup microseconds per verification cycle",
        "max_draft_tokens": args.max_depth,
        "draft_profile_micros": draft_micros,
        "verify_profile_micros": verify_micros,
        "runtime_environment": {
            "XRT_QWEN_DSPARK_DRAFT_PROFILE_US": f"{draft_micros:.6f}",
            "XRT_QWEN_DSPARK_VERIFY_PROFILE_US": verify_csv,
        },
        "sources": sources,
    }
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8", newline="\n")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
