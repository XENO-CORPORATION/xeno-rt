#!/usr/bin/env python3
"""Validate that every measured XRT CLI concurrency repetition completed."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--expected-concurrency", type=int, required=True)
    parser.add_argument("--expected-repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.report.read_text(encoding="utf-8"))
    results = payload.get("results")
    failures: list[dict[str, Any]] = []
    if not isinstance(results, list):
        failures.append({"reason": "results is not an array"})
        results = []
    if len(results) != args.expected_repetitions:
        failures.append(
            {
                "reason": "unexpected repetition count",
                "expected": args.expected_repetitions,
                "actual": len(results),
            }
        )

    successful: list[dict[str, Any]] = []
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            failures.append({"index": index, "reason": "result is not an object"})
            continue
        if result.get("concurrency") != args.expected_concurrency:
            failures.append(
                {
                    "index": index,
                    "reason": "unexpected concurrency",
                    "expected": args.expected_concurrency,
                    "actual": result.get("concurrency"),
                }
            )
        if result.get("error") is not None:
            failures.append(
                {"index": index, "reason": "generation error", "detail": result.get("error")}
            )
            continue
        if not isinstance(result.get("output_tokens"), int) or result["output_tokens"] <= 0:
            failures.append({"index": index, "reason": "no output tokens"})
            continue
        if not isinstance(result.get("decode_tok_s"), (int, float)) or result["decode_tok_s"] <= 0:
            failures.append({"index": index, "reason": "invalid decode throughput"})
            continue
        successful.append(result)

    throughput = [float(result["decode_tok_s"]) for result in successful]
    end_to_end = [float(result["end_to_end_ms"]) for result in successful]
    validation = {
        "object": "xrt.concurrency_validation",
        "report": str(args.report),
        "expected_concurrency": args.expected_concurrency,
        "expected_repetitions": args.expected_repetitions,
        "passed": not failures,
        "successful_repetitions": len(successful),
        "mean_decode_tok_s": statistics.fmean(throughput) if throughput else None,
        "median_decode_tok_s": statistics.median(throughput) if throughput else None,
        "mean_end_to_end_ms": statistics.fmean(end_to_end) if end_to_end else None,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2))
    return 0 if validation["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
