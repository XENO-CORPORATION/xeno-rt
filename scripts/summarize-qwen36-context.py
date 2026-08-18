#!/usr/bin/env python3
"""Summarize per-size long-context admission from an XRT bench report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("expected", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    expected = json.loads(args.expected.read_text(encoding="utf-8"))
    expected_by_case = {case["case_id"]: case for case in expected["cases"]}
    rows = []
    for result in report.get("results", []):
        case_id = result.get("case_id")
        rule = expected_by_case.get(case_id, {})
        preview = result.get("preview") or ""
        required = rule.get("required_text") or ""
        error = result.get("error")
        passed = error is None and required.casefold() in preview.casefold()
        gpu = result.get("gpu_resource") or {}
        rows.append(
            {
                "case_id": case_id,
                "cache_mode": result.get("cache_mode"),
                "approximate_context_tokens": rule.get("approximate_context_tokens"),
                "prompt_tokens": result.get("prompt_tokens"),
                "output_tokens": result.get("output_tokens"),
                "prefill_ms": result.get("prefill_ms"),
                "decode_tok_s": result.get("decode_tok_s"),
                "preview": preview,
                "kv_allocated_bytes": gpu.get("kv_allocated_bytes"),
                "tracked_resident_vram_bytes": result.get("tracked_resident_vram_bytes"),
                "passed": passed,
                "error": error,
            }
        )

    failures = [row for row in rows if not row["passed"]]
    admitted = [row for row in rows if row["passed"]]
    summary = {
        "object": "xrt.context_admission_summary",
        "suite_id": report.get("suite_id"),
        "cache_mode": rows[0].get("cache_mode") if rows else None,
        "passed": not failures,
        "admitted_cases": len(admitted),
        "maximum_admitted_prompt_tokens": max(
            (row["prompt_tokens"] or 0 for row in admitted), default=0
        ),
        "first_failed_case": failures[0]["case_id"] if failures else None,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
