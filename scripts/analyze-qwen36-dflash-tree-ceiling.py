#!/usr/bin/env python3
"""Estimate a bounded DDTree acceptance ceiling from XRT DFlash diagnostics.

The diagnostic run emits one compact top-k distribution per speculative
boundary.  This script rebuilds the official DDTree best-first prefix tree and
checks it against the already generated greedy target continuation.  It is an
acceptance/throughput ceiling study, not a product benchmark: a tree decoder
changes later verification boundaries and adds tree-scan/compaction work.
"""

from __future__ import annotations

import argparse
import heapq
import json
from dataclasses import dataclass
from pathlib import Path


PREFIX = "XRT_DFLASH_TOP4_DIAGNOSTIC "


@dataclass
class Tree:
    child_maps: list[dict[int, int]]


def build_tree(rows: list[list[list[float]]], budget: int, depth_bonus: float) -> Tree:
    if budget <= 0 or not rows:
        return Tree([{}])
    topk = min(budget, min(len(row) for row in rows))
    first_logw = float(rows[0][0][1])
    first_score = first_logw + depth_bonus
    heap: list[tuple[float, tuple[int, ...], int, int, int, float]] = [
        (-first_score, (0,), 0, 1, 0, first_logw)
    ]
    child_maps: list[dict[int, int]] = [{}]
    node_count = 0
    while heap and node_count < budget:
        _, ranks, parent_index, depth, rank, logw = heapq.heappop(heap)
        token_id = int(rows[depth - 1][rank][0])
        current_index = node_count + 1
        child_maps.append({})
        child_maps[parent_index][token_id] = current_index
        node_count += 1

        if rank + 1 < topk:
            sibling_ranks = ranks[:-1] + (rank + 1,)
            sibling_logw = (
                logw
                - float(rows[depth - 1][rank][1])
                + float(rows[depth - 1][rank + 1][1])
            )
            heapq.heappush(
                heap,
                (
                    -(sibling_logw + depth_bonus * depth),
                    sibling_ranks,
                    parent_index,
                    depth,
                    rank + 1,
                    sibling_logw,
                ),
            )
        if depth < len(rows):
            child_ranks = ranks + (0,)
            child_logw = logw + float(rows[depth][0][1])
            heapq.heappush(
                heap,
                (
                    -(child_logw + depth_bonus * (depth + 1)),
                    child_ranks,
                    current_index,
                    depth + 1,
                    0,
                    child_logw,
                ),
            )
    return Tree(child_maps)


def accepted_prefix(tree: Tree, continuation: list[int]) -> int:
    current = 0
    accepted = 0
    for token in continuation:
        child = tree.child_maps[current].get(token)
        if child is None:
            break
        accepted += 1
        current = child
    return accepted


def load_diagnostics(path: Path) -> list[dict]:
    diagnostics = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        marker = line.find(PREFIX)
        if marker >= 0:
            diagnostics.append(json.loads(line[marker + len(PREFIX) :]))
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument(
        "--throughput-reference",
        type=Path,
        help="matched benchmark used only for decode tok/s (diagnostic logging is slow)",
    )
    parser.add_argument("--budget", type=int, default=15)
    parser.add_argument(
        "--depth-bonus",
        type=float,
        default=0.0,
        help="add this score bonus per candidate depth (runtime default: 0.3)",
    )
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    results = benchmark["results"]
    if args.throughput_reference:
        reference = json.loads(args.throughput_reference.read_text(encoding="utf-8"))
        reference_results = [
            row for row in reference["results"] if int(row.get("repetition", 1)) == 1
        ]
        reference_by_case = {row["case_id"]: row for row in reference_results}
    else:
        reference_by_case = {row["case_id"]: row for row in results}
    diagnostics = load_diagnostics(args.log)
    cursor = 0
    report = []
    total_linear = 0
    total_tree = 0
    total_cycles = 0

    for result in results:
        expected_cycles = int(result["speculative_decode"]["verification_batches"])
        case_diagnostics = diagnostics[cursor : cursor + expected_cycles]
        cursor += expected_cycles
        if len(case_diagnostics) != expected_cycles:
            raise SystemExit(
                f"diagnostic log ended inside {result['case_id']}: "
                f"expected {expected_cycles}, found {len(case_diagnostics)}"
            )
        prompt_tokens = int(result["prompt_tokens"])
        output = [int(token) for token in result["output_token_ids"]]
        linear_accepted = 0
        tree_accepted = 0
        for diagnostic in case_diagnostics:
            root_index = int(diagnostic["context_len"]) - prompt_tokens
            continuation = output[root_index + 1 : root_index + 16]
            rows = diagnostic["rows"]
            linear = 0
            for depth, token in enumerate(continuation):
                if depth >= len(rows) or int(rows[depth][0][0]) != token:
                    break
                linear += 1
            linear_accepted += linear
            tree_accepted += accepted_prefix(
                build_tree(rows, args.budget, args.depth_bonus), continuation
            )

        recorded_linear = int(result["speculative_decode"]["accepted_tokens"])
        if linear_accepted != recorded_linear:
            raise SystemExit(
                f"{result['case_id']}: diagnostic top-1 accepted {linear_accepted}, "
                f"benchmark recorded {recorded_linear}; boundary mapping is invalid"
            )
        cycles = expected_cycles
        linear_advance = 1.0 + linear_accepted / cycles
        tree_advance = 1.0 + tree_accepted / cycles
        measured_tps = float(reference_by_case[result["case_id"]]["decode_tok_s"])
        ceiling_tps = measured_tps * tree_advance / linear_advance
        report.append(
            {
                "case_id": result["case_id"],
                "cycles": cycles,
                "linear_accepted": linear_accepted,
                "tree_accepted": tree_accepted,
                "linear_accepted_per_cycle": linear_accepted / cycles,
                "tree_accepted_per_cycle": tree_accepted / cycles,
                "measured_decode_tok_s": measured_tps,
                "zero_overhead_tree_ceiling_tok_s": ceiling_tps,
            }
        )
        total_linear += linear_accepted
        total_tree += tree_accepted
        total_cycles += cycles

    if cursor != len(diagnostics):
        raise SystemExit(
            f"diagnostic count mismatch: consumed {cursor}, log contains {len(diagnostics)}"
        )
    print(
        json.dumps(
            {
                "budget": args.budget,
                "depth_bonus": args.depth_bonus,
                "topk_available": 4,
                "cases": len(report),
                "cycles": total_cycles,
                "linear_accepted": total_linear,
                "tree_accepted": total_tree,
                "linear_accepted_per_cycle": total_linear / total_cycles,
                "tree_accepted_per_cycle": total_tree / total_cycles,
                "mean_measured_decode_tok_s": sum(
                    row["measured_decode_tok_s"] for row in report
                )
                / len(report),
                "mean_zero_overhead_tree_ceiling_tok_s": sum(
                    row["zero_overhead_tree_ceiling_tok_s"] for row in report
                )
                / len(report),
                "results": report,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
