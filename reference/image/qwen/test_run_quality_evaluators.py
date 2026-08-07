#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import prepare_quality_review
import run_quality_evaluators as evaluators


class QualityEvaluatorTests(unittest.TestCase):
    def test_ocr_normalization_and_error_rates_follow_frozen_policy(self) -> None:
        self.assertEqual(evaluators.normalize_ocr_text("  XÉNO—STUDIO! "), "XÉNO STUDIO")
        cer, wer, best = evaluators.error_rates(
            "XENO STUDIO",
            ["XENO STUO", "XENO STUDIO", "irrelevant"],
        )
        self.assertEqual((cer, wer, best), (0.0, 0.0, "XENO STUDIO"))
        cer, wer, _ = evaluators.error_rates("ONE TWO", ["ONE TOO"])
        self.assertGreater(cer, 0.0)
        self.assertEqual(wer, 0.5)

    def test_ocr_candidates_preserve_block_order_and_joined_reading_order(self) -> None:
        value = {
            "res": {
                "parsing_res_list": [
                    {"block_order": 2, "block_content": "STUDIO"},
                    {"block_order": 1, "block_content": "XENO"},
                ]
            }
        }
        self.assertEqual(evaluators.ocr_candidates(value), ["XENO", "STUDIO", "XENO STUDIO"])

    def test_metric_checkpoint_resume_is_bound_to_image_and_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metric.json"
            path.write_bytes(
                prepare_quality_review.canonical_bytes(
                    {
                        "schema_version": 1,
                        "metric_schema": evaluators.METRIC_SCHEMA,
                        "evaluator": "ocr",
                        "plan_sha256": "a" * 64,
                        "id": "gen-type-001",
                        "category": "generation_typography",
                        "side": "bf16",
                        "image_sha256": "b" * 64,
                        "metrics": {"character_error_rate": 0.0, "word_error_rate": 0.0},
                    }
                )
            )
            record = evaluators.reuse_checkpoint(
                path, "ocr", "a" * 64, "gen-type-001", "bf16", "b" * 64
            )
            self.assertIsNotNone(record)
            with self.assertRaisesRegex(evaluators.EvaluatorError, "drift"):
                evaluators.reuse_checkpoint(
                    path, "ocr", "a" * 64, "gen-type-001", "bf16", "c" * 64
                )

    def test_assemble_preserves_metrics_and_adds_only_review_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = root / "metrics.json"
            reviews = root / "reviews.json"
            output = root / "results.json"
            metrics.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "object": "xeno.image.quality_metric_export",
                        "suite": {"version": "fixture", "sha256": "a" * 64},
                        "case_results": [],
                    }
                ),
                encoding="utf-8",
            )
            reviews.write_text(
                json.dumps(
                    {
                        "human_review_protocol": {"blinded": True},
                        "human_reviews": [{"pair_id": "fixture"}],
                    }
                ),
                encoding="utf-8",
            )
            evaluators.assemble(
                argparse.Namespace(metrics=metrics, human_reviews=reviews, output=output)
            )
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["object"], "xeno.image.quality_results")
            self.assertEqual(result["human_reviews"], [{"pair_id": "fixture"}])
            self.assertEqual(result["case_results"], [])


if __name__ == "__main__":
    unittest.main()
