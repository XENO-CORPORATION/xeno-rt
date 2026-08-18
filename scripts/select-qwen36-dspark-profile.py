#!/usr/bin/env python3
"""Rank exact-output DSpark benchmark arms after warmup exclusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--goal", type=float, default=177.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for summary_path in sorted(args.directory.glob("*-summary.json")):
        label = summary_path.name.removesuffix("-summary.json")
        parity_path = args.directory / f"{label}-parity.json"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            parity = json.loads(parity_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            rejected.append({"label": label, "reason": str(error)})
            continue
        candidate = summary.get("candidate") or {}
        errors = candidate.get("errors") or []
        exact = parity.get("parity") is True
        mean = (candidate.get("decode_tok_s") or {}).get("mean")
        if errors or not exact or not isinstance(mean, (int, float)):
            rejected.append(
                {
                    "label": label,
                    "reason": "candidate errors, missing measurements, or token-parity failure",
                    "errors": errors,
                    "parity": parity.get("parity"),
                }
            )
            continue
        rates = candidate["decode_tok_s"]
        rows.append(
            {
                "label": label,
                "mean_decode_tok_s": mean,
                "median_decode_tok_s": rates.get("median"),
                "aggregate_decode_tok_s": candidate.get("aggregate_decode_tok_s"),
                "minimum_decode_tok_s": rates.get("min"),
                "p05_decode_tok_s": rates.get("p05"),
                "p95_decode_tok_s": rates.get("p95"),
                "maximum_decode_tok_s": rates.get("max"),
                "samples": candidate.get("successful_samples"),
                "drafted_tokens": candidate.get("drafted_tokens"),
                "accepted_tokens": candidate.get("accepted_tokens"),
                "acceptance_rate": candidate.get("acceptance_rate"),
                "exact_token_parity": True,
            }
        )

    rows.sort(key=lambda row: float(row["mean_decode_tok_s"]), reverse=True)
    best = rows[0] if rows else None
    report = {
        "object": "xrt.qwen36_dspark_profile_selection",
        "goal_mean_decode_tok_s": args.goal,
        "goal_met": best is not None and float(best["mean_decode_tok_s"]) >= args.goal,
        "best": best,
        "ranking": rows,
        "rejected": rejected,
    }
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8", newline="\n")
    print(payload, end="")
    return 0 if best is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
