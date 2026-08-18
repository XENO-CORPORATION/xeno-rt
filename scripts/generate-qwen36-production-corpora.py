#!/usr/bin/env python3
"""Generate deterministic Qwen production-admission prompt corpora."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA_VERSION = 1
APPROXIMATE_CONTEXT_TOKENS = (512, 768, 1024, 2048, 4096, 8192, 16384, 24576, 32768)


def write_json(path: Path, value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8", newline="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def record(index: int) -> str:
    owner = ("amber", "birch", "cobalt", "delta")[index % 4]
    state = ("nominal", "stable", "verified", "archived")[index % 4]
    checksum = (index * 2654435761) & 0xFFFFFFFF
    return (
        f"Record {index:06d}: owner={owner}; state={state}; "
        f"checksum={checksum:08x}; region=local."
    )


def long_context_case(approximate_tokens: int, ordinal: int) -> tuple[dict, dict]:
    # This varied, deterministic text is intentionally less compressible than
    # repeating one sentence. The runtime records the exact token count.
    # Calibrated against this model's embedded tokenizer: the deliberately
    # varied record format averages about 35 tokens after chat templating.
    # Exact prompt-token counts remain authoritative in benchmark reports.
    estimated_tokens_per_record = 35
    count = max(8, approximate_tokens // estimated_tokens_per_record)
    positions = (0.12, 0.50, 0.86)
    needle_index = min(count - 1, int(count * positions[ordinal % len(positions)]))
    marker = f"XRT-{approximate_tokens:05d}-{ordinal:02d}-PASS"
    lines = [record(index) for index in range(count)]
    lines.insert(
        needle_index,
        f"Priority record: the authorization marker is {marker}. Preserve it exactly.",
    )
    content = "\n".join(lines)
    case_id = f"context_{approximate_tokens:05d}_needle_{needle_index:05d}"
    case = {
        "id": case_id,
        "messages": [
            {
                "role": "system",
                "content": "Answer retrieval questions with only the requested marker and no explanation.",
            },
            {
                "role": "user",
                "content": content
                + "\n\nWhat is the authorization marker in the priority record? Reply with only that marker.",
            },
        ],
        "max_tokens": 32,
    }
    expectation = {
        "case_id": case_id,
        "required_text": marker,
        "forbidden_prefixes": ["Thinking Process", "Here's a thinking process", "<think>"],
        "approximate_context_tokens": approximate_tokens,
        "needle_record_index": needle_index,
    }
    return case, expectation


def quality_corpus(suite_prefix: str) -> tuple[dict, dict]:
    cases = [
        {
            "id": "exact_sentinel",
            "messages": [
                {
                    "role": "system",
                    "content": "Follow the requested output format exactly. Do not explain.",
                },
                {"role": "user", "content": "Reply with exactly XRT_READY and nothing else."},
            ],
            "max_tokens": 16,
        },
        {
            "id": "arithmetic_result",
            "messages": [
                {"role": "system", "content": "Return only the numeric answer."},
                {
                    "role": "user",
                    "content": "A service receives 240 requests per minute, rises by 35 percent, then caching removes 20 percent. How many backend requests remain?",
                },
            ],
            "max_tokens": 24,
        },
        {
            "id": "multiturn_recall",
            "messages": [
                {"role": "user", "content": "Remember that the deployment codename is Ember."},
                {"role": "assistant", "content": "Understood."},
                {"role": "user", "content": "What is the deployment codename? Reply with one word."},
            ],
            "max_tokens": 16,
        },
        {
            "id": "unicode_recall",
            "messages": [
                {"role": "user", "content": "The exact release label is München-東京-مرحبا."},
                {
                    "role": "user",
                    "content": "Repeat only the exact release label, preserving Unicode and punctuation.",
                },
            ],
            "max_tokens": 24,
        },
    ]
    expected = {
        "schema_version": 1,
        "suite_id": f"{suite_prefix}-production-quality-v1",
        "cases": [
            {"case_id": "exact_sentinel", "required_text": "XRT_READY"},
            {"case_id": "arithmetic_result", "required_text": "259.2"},
            {"case_id": "multiturn_recall", "required_text": "Ember"},
            {"case_id": "unicode_recall", "required_text": "München-東京-مرحبا"},
        ],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "suite_id": expected["suite_id"],
        "cases": cases,
    }, expected


def shared_multiturn_corpus(suite_prefix: str) -> tuple[dict, dict]:
    # Stay inside the production DFlash/F32 envelope while still exercising
    # repeated multi-message history and shared-prefix behavior.
    shared_context = "\n".join(record(index) for index in range(16))
    shared_messages = [
        {
            "role": "system",
            "content": "You are a concise deployment assistant. Answer with only the requested value.",
        },
        {"role": "user", "content": shared_context},
        {"role": "assistant", "content": "Context loaded."},
        {"role": "user", "content": "The active channel is cobalt."},
        {"role": "assistant", "content": "Channel recorded."},
    ]
    prompts = (
        ("shared_history_channel", "What is the active channel?", "cobalt"),
        ("shared_history_owner", "What is the owner in Record 000003?", "delta"),
        ("shared_history_state", "What is the state in Record 000002?", "verified"),
    )
    cases = []
    expected_cases = []
    for case_id, question, answer in prompts:
        cases.append(
            {
                "id": case_id,
                "messages": shared_messages + [{"role": "user", "content": question}],
                "max_tokens": 24,
            }
        )
        expected_cases.append({"case_id": case_id, "required_text": answer})
    suite_id = f"{suite_prefix}-shared-multiturn-v1"
    return {
        "schema_version": SCHEMA_VERSION,
        "suite_id": suite_id,
        "cases": cases,
    }, {"schema_version": 1, "suite_id": suite_id, "cases": expected_cases}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark-corpora/text/qwen36-production-v1"),
    )
    parser.add_argument(
        "--suite-prefix",
        default="qwen36",
        help="suite identifier prefix, for example qwen36 or qwen38",
    )
    args = parser.parse_args()
    output_dir: Path = args.output_dir
    manifest: dict[str, object] = {"schema_version": 1, "files": {}}

    suite_prefix = args.suite_prefix
    quality, quality_expected = quality_corpus(suite_prefix)
    multiturn, multiturn_expected = shared_multiturn_corpus(suite_prefix)
    artifacts = {
        "quality.json": quality,
        "quality.expected.json": quality_expected,
        "shared-multiturn.json": multiturn,
        "shared-multiturn.expected.json": multiturn_expected,
    }
    for name, value in artifacts.items():
        manifest["files"][name] = write_json(output_dir / name, value)

    all_context_cases = []
    all_context_expectations = []
    for ordinal, approximate_tokens in enumerate(APPROXIMATE_CONTEXT_TOKENS):
        case, expectation = long_context_case(approximate_tokens, ordinal)
        all_context_cases.append(case)
        all_context_expectations.append(expectation)
        suite = {
            "schema_version": SCHEMA_VERSION,
            "suite_id": f"{suite_prefix}-long-context-v1-{approximate_tokens}",
            "cases": [case],
        }
        expected = {
            "schema_version": 1,
            "suite_id": suite["suite_id"],
            "cases": [expectation],
        }
        name = f"context-{approximate_tokens:05d}.json"
        expected_name = f"context-{approximate_tokens:05d}.expected.json"
        manifest["files"][name] = write_json(output_dir / name, suite)
        manifest["files"][expected_name] = write_json(output_dir / expected_name, expected)

    all_context_suite = {
        "schema_version": SCHEMA_VERSION,
        "suite_id": f"{suite_prefix}-long-context-v1-all",
        "cases": all_context_cases,
    }
    all_context_expected = {
        "schema_version": SCHEMA_VERSION,
        "suite_id": all_context_suite["suite_id"],
        "cases": all_context_expectations,
    }
    manifest["files"]["context-all.json"] = write_json(
        output_dir / "context-all.json", all_context_suite
    )
    manifest["files"]["context-all.expected.json"] = write_json(
        output_dir / "context-all.expected.json", all_context_expected
    )

    for maximum in (8192, 16384):
        selected = [
            (case, expectation)
            for case, expectation in zip(
                all_context_cases, all_context_expectations, strict=True
            )
            if expectation["approximate_context_tokens"] <= maximum
        ]
        suite_id = f"{suite_prefix}-long-context-v1-through-{maximum}"
        subset_suite = {
            "schema_version": SCHEMA_VERSION,
            "suite_id": suite_id,
            "cases": [case for case, _ in selected],
        }
        subset_expected = {
            "schema_version": SCHEMA_VERSION,
            "suite_id": suite_id,
            "cases": [expectation for _, expectation in selected],
        }
        subset_name = f"context-through-{maximum:05d}.json"
        subset_expected_name = f"context-through-{maximum:05d}.expected.json"
        manifest["files"][subset_name] = write_json(output_dir / subset_name, subset_suite)
        manifest["files"][subset_expected_name] = write_json(
            output_dir / subset_expected_name, subset_expected
        )

    manifest_path = output_dir / "manifest.json"
    manifest["files"][manifest_path.name] = "self"
    write_json(manifest_path, manifest)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
