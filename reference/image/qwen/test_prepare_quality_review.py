#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import evaluate_quality_suite
import prepare_quality_review


HERE = Path(__file__).resolve().parent
SUITE_PATH = HERE.parents[2] / "tests" / "common" / "image-quality-suite.json"


class PrepareQualityReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        suite_bytes = SUITE_PATH.read_bytes()
        cls.suite = json.loads(suite_bytes)
        cls.plan = evaluate_quality_suite.build_plan(
            cls.suite,
            evaluate_quality_suite.sha256_bytes(suite_bytes),
            "Q4_K_M",
            False,
        )
        cls.plan_sha256 = prepare_quality_review.canonical_digest(cls.plan)

    def test_mapping_is_stratified_balanced_and_deterministic(self) -> None:
        first = prepare_quality_review.build_mapping(
            self.plan,
            self.plan_sha256,
            prepare_quality_review.DEFAULT_RANDOMIZATION_SEED,
        )
        second = prepare_quality_review.build_mapping(
            self.plan,
            self.plan_sha256,
            prepare_quality_review.DEFAULT_RANDOMIZATION_SEED,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first["pairs"]), 200)
        self.assertEqual(Counter(pair["candidate_slot"] for pair in first["pairs"]), {"A": 100, "B": 100})
        counts = Counter(pair["category"] for pair in first["pairs"])
        self.assertEqual(counts["identity_preservation"], 50)
        self.assertEqual(
            counts,
            {
                "generation_general": 60,
                "generation_typography": 24,
                "generation_faces_hands_detail": 18,
                "generation_style_color": 18,
                "edit_single_image": 18,
                "edit_multi_image": 12,
                "identity_preservation": 50,
            },
        )
        self.assertEqual(len({pair["pair_token"] for pair in first["pairs"]}), 200)

    def test_compile_emits_exact_three_rater_candidate_records(self) -> None:
        mapping = prepare_quality_review.build_mapping(
            self.plan,
            self.plan_sha256,
            prepare_quality_review.DEFAULT_RANDOMIZATION_SEED,
        )
        mapping["rubric_sha256"] = prepare_quality_review.canonical_digest(
            self.suite["human_severe_defect_rubric"]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private = root / "package" / "private"
            private.mkdir(parents=True)
            (private / "mapping.json").write_bytes(prepare_quality_review.canonical_bytes(mapping))
            responses = []
            for rater in prepare_quality_review.RATER_IDS:
                path = root / f"{rater}.json"
                ratings = []
                for pair in mapping["pairs"]:
                    identity = pair["identity_pair"]
                    ratings.append(
                        {
                            "pair_token": pair["pair_token"],
                            "a_severe_defect": True,
                            "b_severe_defect": False,
                            "a_identity_failure": True if identity else None,
                            "b_identity_failure": False if identity else None,
                        }
                    )
                path.write_text(
                    json.dumps({"schema_version": 1, "rater_id": rater, "ratings": ratings}),
                    encoding="utf-8",
                )
                responses.append(path)
            output = root / "compiled.json"
            prepare_quality_review.compile_responses(
                argparse.Namespace(package=root / "package", responses=responses, output=output)
            )
            compiled = json.loads(output.read_text(encoding="utf-8"))
            reviews = compiled["human_reviews"]
            self.assertEqual(len(reviews), 600)
            self.assertEqual(len({(item["pair_id"], item["rater_id"]) for item in reviews}), 600)
            self.assertEqual(sum(item["severe_defect"] for item in reviews), 300)
            self.assertEqual(
                sum(item["identity_failure"] is True for item in reviews),
                3
                * sum(
                    pair["identity_pair"] and pair["candidate_slot"] == "A"
                    for pair in mapping["pairs"]
                ),
            )


if __name__ == "__main__":
    unittest.main()
