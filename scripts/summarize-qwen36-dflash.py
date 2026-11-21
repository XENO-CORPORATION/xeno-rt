#!/usr/bin/env python3
"""Summarize measured (post-warmup) Qwen3.6 target/DFlash decode samples."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def sample_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(row.get("case_id") or "prompt"),
        int(row.get("repetition") or 0),
        str(row.get("cache_mode") or ""),
    )


def measured_rows(report: dict[str, Any], warmups: int) -> list[dict[str, Any]]:
    return [
        row
        for row in report.get("results", [])
        if int(row.get("repetition") or 0) > warmups
    ]


def arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if not row.get("error")]
    values = [float(row["decode_tok_s"]) for row in successful]
    sample_stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    total_decode_tokens = sum(int(row.get("decode_tokens") or 0) for row in successful)
    total_decode_ms = sum(float(row.get("decode_ms") or 0.0) for row in successful)
    values_by_case: dict[str, list[float]] = {}
    for row in successful:
        values_by_case.setdefault(str(row.get("case_id") or "prompt"), []).append(
            float(row["decode_tok_s"])
        )
    drafted = 0
    accepted = 0
    for row in successful:
        speculative = row.get("speculative_decode") or {}
        drafted += int(speculative.get("drafted_tokens") or 0)
        accepted += int(speculative.get("accepted_tokens") or 0)
    return {
        "samples": len(rows),
        "successful_samples": len(successful),
        "errors": [
            {"key": sample_key(row), "error": row.get("error")}
            for row in rows
            if row.get("error")
        ],
        "decode_tok_s": {
            "mean": statistics.fmean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "sample_stdev": sample_stdev,
            "normal_95pct_ci_half_width": (
                1.96 * sample_stdev / math.sqrt(len(values)) if values else None
            ),
            "min": min(values) if values else None,
            "p05": percentile(values, 0.05),
            "p95": percentile(values, 0.95),
            "max": max(values) if values else None,
        },
        "aggregate_decode_tok_s": (
            total_decode_tokens / (total_decode_ms / 1000.0) if total_decode_ms > 0 else None
        ),
        "total_decode_tokens": total_decode_tokens,
        "total_decode_ms": total_decode_ms,
        "mean_decode_tok_s_by_case": {
            case_id: statistics.fmean(case_values)
            for case_id, case_values in sorted(values_by_case.items())
        },
        "drafted_tokens": drafted,
        "accepted_tokens": accepted,
        "acceptance_rate": accepted / drafted if drafted else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--warmup-repetitions", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmup_repetitions < 0:
        parser.error("--warmup-repetitions cannot be negative")

    target = json.loads(args.target.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    target_rows = measured_rows(target, args.warmup_repetitions)
    candidate_rows = measured_rows(candidate, args.warmup_repetitions)
    target_by_key = {sample_key(row): row for row in target_rows}
    candidate_by_key = {sample_key(row): row for row in candidate_rows}
    common_keys = sorted(target_by_key.keys() & candidate_by_key.keys())
    paired_speedups = []
    for key in common_keys:
        target_row = target_by_key[key]
        candidate_row = candidate_by_key[key]
        if target_row.get("error") or candidate_row.get("error"):
            continue
        target_rate = float(target_row["decode_tok_s"])
        candidate_rate = float(candidate_row["decode_tok_s"])
        if target_rate > 0:
            paired_speedups.append(candidate_rate / target_rate)

    target_summary = arm_summary(target_rows)
    candidate_summary = arm_summary(candidate_rows)
    target_mean = target_summary["decode_tok_s"]["mean"]
    candidate_mean = candidate_summary["decode_tok_s"]["mean"]
    summary = {
        "object": "xrt.qwen36_dflash_summary",
        "warmup_repetitions_excluded_per_case": args.warmup_repetitions,
        "target": target_summary,
        "candidate": candidate_summary,
        "paired_samples": len(paired_speedups),
        "mean_of_paired_speedups": (
            statistics.fmean(paired_speedups) if paired_speedups else None
        ),
        "ratio_of_arm_means": (
            candidate_mean / target_mean
            if target_mean is not None and candidate_mean is not None and target_mean > 0
            else None
        ),
        "missing_target_keys": [key for key in sorted(candidate_by_key.keys() - target_by_key.keys())],
        "missing_candidate_keys": [key for key in sorted(target_by_key.keys() - candidate_by_key.keys())],
    }
    payload = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8", newline="\n")
    print(payload, end="")
    passed = (
        not target_summary["errors"]
        and not candidate_summary["errors"]
        and not summary["missing_target_keys"]
        and not summary["missing_candidate_keys"]
        and bool(paired_speedups)
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
