#!/usr/bin/env python3
"""Validate a bounded XRT bench report that must complete successfully."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--expected-backend", required=True)
    parser.add_argument("--expected-repetitions", type=int, required=True)
    parser.add_argument("--required-text")
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

    successful = 0
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            failures.append({"index": index, "reason": "result is not an object"})
            continue
        if result.get("active_backend") != args.expected_backend:
            failures.append(
                {
                    "index": index,
                    "reason": "unexpected active backend",
                    "expected": args.expected_backend,
                    "actual": result.get("active_backend"),
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
        if args.required_text and args.required_text not in str(result.get("preview", "")):
            failures.append(
                {
                    "index": index,
                    "reason": "required text missing",
                    "required_text": args.required_text,
                    "preview": result.get("preview"),
                }
            )
            continue
        successful += 1

    validation = {
        "object": "xrt.bench_success_validation",
        "report": str(args.report),
        "expected_backend": args.expected_backend,
        "expected_repetitions": args.expected_repetitions,
        "passed": not failures,
        "successful_repetitions": successful,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2))
    return 0 if validation["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
