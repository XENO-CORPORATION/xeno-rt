#!/usr/bin/env python3
"""Validate task success and answer-mode hygiene in an XRT bench report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_FORBIDDEN_PREFIXES = ("Thinking Process", "Here's a thinking process", "<think>")


def answer_text(result: dict) -> str:
    """Return the assistant answer, excluding a Qwen thinking prefix."""
    if result.get("answer_text") is not None:
        return result["answer_text"]
    text = result.get("output_text") or result.get("preview") or ""
    if "</think>" in text:
        return text.split("</think>", 1)[1].lstrip()
    if text.lstrip().startswith("<think>"):
        return ""
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("expected", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    expected = json.loads(args.expected.read_text(encoding="utf-8"))
    expected_by_case = {case["case_id"]: case for case in expected["cases"]}
    results_by_case: dict[str, list[dict]] = {}
    for result in report.get("results", []):
        case_id = result.get("case_id")
        if case_id:
            results_by_case.setdefault(case_id, []).append(result)

    failures: list[dict[str, object]] = []
    checked = 0
    for case_id, rule in expected_by_case.items():
        case_results = results_by_case.get(case_id, [])
        if not case_results:
            failures.append({"case_id": case_id, "reason": "missing result"})
            continue
        for result in case_results:
            checked += 1
            if result.get("error"):
                failures.append(
                    {"case_id": case_id, "reason": "generation error", "detail": result["error"]}
                )
                continue
            preview = result.get("preview") or ""
            answer = answer_text(result)
            required = rule["required_text"]
            if required.casefold() not in answer.casefold():
                failures.append(
                    {
                        "case_id": case_id,
                        "reason": "required text missing from answer",
                        "required_text": required,
                        "preview": preview,
                        "answer": answer,
                    }
                )
            for prefix in rule.get("forbidden_prefixes", DEFAULT_FORBIDDEN_PREFIXES):
                if answer.lstrip().startswith(prefix):
                    failures.append(
                        {
                            "case_id": case_id,
                            "reason": "thinking text leaked into answer",
                            "prefix": prefix,
                            "preview": preview,
                            "answer": answer,
                        }
                    )
                    break

    summary = {
        "object": "xrt.production_validation",
        "suite_id": expected.get("suite_id"),
        "report": str(args.report),
        "enable_thinking": report.get("enable_thinking"),
        "checked_results": checked,
        "passed": not failures,
        "failures": failures,
    }
    payload = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8", newline="\n")
    print(payload, end="")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
