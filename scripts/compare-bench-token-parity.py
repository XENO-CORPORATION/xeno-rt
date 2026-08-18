#!/usr/bin/env python3
"""Compare exact generated token traces from two xrt bench suite reports."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any


def load_report(path: Path) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load {path}: {error}") from error
    if report.get("object") != "xrt.bench":
        raise ValueError(f"{path} is not an xrt.bench report")
    if not report.get("suite_id"):
        raise ValueError(f"{path} does not identify a prompt suite")
    return report


def traces(report: dict[str, Any], label: str) -> dict[str, tuple[int, ...]]:
    by_case: dict[str, set[tuple[int, ...]]] = {}
    for index, result in enumerate(report.get("results", [])):
        if result.get("error") is not None:
            raise ValueError(f"{label} result {index} failed: {result['error']}")
        case_id = result.get("case_id")
        token_ids = result.get("output_token_ids")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"{label} result {index} has no case_id")
        if not isinstance(token_ids, list) or not all(isinstance(token, int) for token in token_ids):
            raise ValueError(f"{label} result {index} has no exact output_token_ids trace")
        if result.get("output_tokens") != len(token_ids):
            raise ValueError(
                f"{label} result {index} reports {result.get('output_tokens')} output tokens "
                f"but records {len(token_ids)} token IDs"
            )
        by_case.setdefault(case_id, set()).add(tuple(token_ids))

    if not by_case:
        raise ValueError(f"{label} report contains no cases")
    unstable = sorted(case_id for case_id, values in by_case.items() if len(values) != 1)
    if unstable:
        raise ValueError(f"{label} is not deterministic for cases: {', '.join(unstable)}")
    return {case_id: next(iter(values)) for case_id, values in by_case.items()}


def performance(report: dict[str, Any]) -> dict[str, float | int | None]:
    speeds = [
        float(result["decode_tok_s"])
        for result in report.get("results", [])
        if result.get("error") is None and result.get("decode_tok_s") is not None
    ]
    speculative = [
        result["speculative_decode"]
        for result in report.get("results", [])
        if isinstance(result.get("speculative_decode"), dict)
    ]
    drafted = sum(int(stats.get("drafted_tokens", 0)) for stats in speculative)
    accepted = sum(int(stats.get("accepted_tokens", 0)) for stats in speculative)
    return {
        "samples": len(speeds),
        "mean_decode_tok_s": statistics.fmean(speeds) if speeds else None,
        "median_decode_tok_s": statistics.median(speeds) if speeds else None,
        "minimum_decode_tok_s": min(speeds) if speeds else None,
        "maximum_decode_tok_s": max(speeds) if speeds else None,
        "drafted_tokens": drafted,
        "accepted_tokens": accepted,
        "acceptance_rate": accepted / drafted if drafted else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        baseline = load_report(args.baseline)
        candidate = load_report(args.candidate)
        if baseline["suite_id"] != candidate["suite_id"]:
            raise ValueError(
                f"suite mismatch: {baseline['suite_id']} != {candidate['suite_id']}"
            )
        baseline_traces = traces(baseline, "baseline")
        candidate_traces = traces(candidate, "candidate")
        if baseline_traces.keys() != candidate_traces.keys():
            missing = sorted(baseline_traces.keys() - candidate_traces.keys())
            extra = sorted(candidate_traces.keys() - baseline_traces.keys())
            raise ValueError(f"case mismatch; missing={missing}, extra={extra}")
        mismatches = [
            case_id
            for case_id in sorted(baseline_traces)
            if baseline_traces[case_id] != candidate_traces[case_id]
        ]
        summary = {
            "object": "xrt.bench.token_parity",
            "suite_id": baseline["suite_id"],
            "parity": not mismatches,
            "case_count": len(baseline_traces),
            "mismatched_cases": mismatches,
            "baseline": performance(baseline),
            "candidate": performance(candidate),
        }
        rendered = json.dumps(summary, indent=2) + "\n"
        if args.output:
            args.output.write_text(rendered, encoding="utf-8")
        sys.stdout.write(rendered)
        return 0 if not mismatches else 1
    except ValueError as error:
        print(f"parity comparison failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
