#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from PIL import Image

import evaluate_quality_suite as quality


class QualityAdmissionTests(unittest.TestCase):
    def test_checked_in_plan_has_exact_active_coverage(self) -> None:
        suite, digest = quality.load_suite(quality.DEFAULT_SUITE_PATH)
        plan = quality.build_plan(suite, digest, "Q4_K_M", include_inpaint=False)
        self.assertEqual(plan["suite"]["sha256"], digest)
        self.assertEqual(len(plan["cases"]), 250)
        self.assertEqual(len(plan["identity_preservation_pairs"]), 50)
        self.assertNotIn(
            "conditional_inpaint",
            {case["category"] for case in plan["cases"]},
        )

        inpaint_plan = quality.build_plan(suite, digest, "Q4_K_M", include_inpaint=True)
        self.assertEqual(len(inpaint_plan["cases"]), 270)
        self.assertIn(
            "conditional_inpaint",
            {case["category"] for case in inpaint_plan["cases"]},
        )

    def test_bootstrap_is_deterministic_and_one_sided(self) -> None:
        values = [0.0, 0.01, 0.02, 0.03, 0.04]
        first = quality.bootstrap_mean_upper(values, 10_000, 1480937837, 0.95)
        second = quality.bootstrap_mean_upper(values, 10_000, 1480937837, 0.95)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first[0], 0.02)
        self.assertGreaterEqual(first[1], first[0])

    def test_one_sided_wilson_zero_defect_bounds(self) -> None:
        self.assertLess(quality.wilson_upper(0, 200, 0.95), 0.02)
        self.assertLess(quality.wilson_upper(0, 50, 0.95), 0.10)
        self.assertGreater(quality.wilson_upper(1, 50, 0.95), 0.02)

    def test_model_pair_rejects_logical_revision_mismatch(self) -> None:
        digest = "a" * 64
        pairs = {
            role: {
                "bf16": {
                    "model_id": f"{role}-bf16",
                    "bundle_digest": digest,
                    "logical_model_revision": "official-a",
                    "artifact_revision": "artifact-a",
                    "quantization": "BF16",
                },
                "candidate": {
                    "model_id": f"{role}-q4",
                    "bundle_digest": "b" * 64,
                    "logical_model_revision": "official-b" if role == "edit" else "official-a",
                    "artifact_revision": "artifact-b",
                    "quantization": "Q4_K_M",
                },
            }
            for role in ("generation", "edit")
        }
        with self.assertRaisesRegex(quality.QualityAdmissionError, "logical model revision"):
            quality.validate_model_pairs(pairs, "Q4_K_M")

    def test_artifact_validation_rejects_strictly_uniform_png(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "uniform.png"
            Image.new("RGB", (8, 8), (10, 20, 30)).save(path)
            record = {
                "artifact_path": "uniform.png",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "width": 8,
                "height": 8,
                "pipeline_finite": True,
                "blank_detected": False,
                "metrics": {"prompt_alignment": 0.5},
            }
            with self.assertRaisesRegex(quality.QualityAdmissionError, "strictly uniform"):
                quality.validate_output(
                    record,
                    "candidate",
                    "generation_general",
                    root,
                    (8, 8),
                    set(),
                )

    def test_human_review_requires_distinct_raters(self) -> None:
        suite = {
            "human_severe_defect_rubric": ["blank output"],
            "statistics": {
                "human_raters_per_pair": 3,
                "human_pairs_per_tier": 2,
                "confidence": 0.95,
                "identity_failure_upper_bound": 1.0,
            },
            "relative_admission_thresholds": {
                "Q4_K_M": {"human_severe_defect_rate": 1.0}
            },
        }
        protocol = {
            "blinded": True,
            "randomized": True,
            "rater_ids_pseudonymous": True,
            "rubric_sha256": quality.canonical_json_sha256(
                suite["human_severe_defect_rubric"]
            ),
        }
        cases = {"case-1": ("generation_general", {})}
        identities = {"identity-1": {}}
        reviews = []
        for pair_id in ("case-1", "identity-1"):
            for index, rater in enumerate(("rater-1", "rater-1", "rater-3")):
                reviews.append(
                    {
                        "pair_id": pair_id,
                        "rater_id": rater,
                        "candidate_slot": "A" if index % 2 == 0 else "B",
                        "severe_defect": False,
                        "identity_failure": False if pair_id.startswith("identity") else None,
                    }
                )
        with self.assertRaisesRegex(quality.QualityAdmissionError, "distinct pseudonymous"):
            quality.validate_human_reviews(
                suite,
                "Q4_K_M",
                protocol,
                reviews,
                cases,
                identities,
            )

    def test_complete_synthetic_report_passes_without_claiming_production(self) -> None:
        category_metrics = {
            "generation_general": {"prompt_alignment": 0.5},
            "generation_typography": {
                "prompt_alignment": 0.5,
                "character_error_rate": 0.1,
                "word_error_rate": 0.1,
            },
            "generation_faces_hands_detail": {"prompt_alignment": 0.5},
            "generation_style_color": {"prompt_alignment": 0.5},
            "edit_single_image": {
                "structural_identity": 0.8,
                "face_identity": 0.8,
            },
            "edit_multi_image": {
                "structural_identity": 0.8,
                "face_identity": 0.8,
            },
            "conditional_inpaint": {"protected_pixel_leakage": 0.0},
        }
        categories = {
            name: [{"id": f"{name}-1", "prompt": "synthetic", "seed": index + 1}]
            for index, name in enumerate(category_metrics)
        }
        evaluator_identity = {
            "prompt_alignment": {"bf16_absolute_floor": 0.2},
            "ocr": {"implementation": "synthetic"},
            "structural_identity": {"bf16_absolute_floor": 0.5},
            "face_identity": {"bf16_absolute_floor": 0.5},
            "mask_leakage": {"implementation": "synthetic"},
        }
        suite = {
            "suite_version": "synthetic-v1",
            "categories": categories,
            "identity_preservation_pairs": [
                {"id": "identity-1", "prompt": "preserve", "seed": 100}
            ],
            "execution": {
                "default_size": "8x8",
                "default_steps": 50,
                "true_cfg_scale": 4.0,
            },
            "evaluators": evaluator_identity,
            "absolute_quality_floors": {
                "generation_general": {"prompt_alignment_min": 0.2},
                "generation_typography": {
                    "prompt_alignment_min": 0.2,
                    "character_error_rate_max": 0.25,
                    "word_error_rate_max": 0.4,
                },
                "generation_faces_hands_detail": {"prompt_alignment_min": 0.2},
                "generation_style_color": {"prompt_alignment_min": 0.2},
                "edit_single_image": {
                    "structural_identity_min": 0.5,
                    "face_identity_min": 0.5,
                },
                "edit_multi_image": {
                    "structural_identity_min": 0.5,
                    "face_identity_min": 0.5,
                },
                "conditional_inpaint": {
                    "protected_pixel_leakage_upper_bound_max": 0.02
                },
            },
            "relative_admission_thresholds": {
                "Q4_K_M": {
                    "prompt_alignment_decline_relative": 0.05,
                    "ocr_cer_increase_points": 0.06,
                    "structural_identity_decline_absolute": 0.03,
                    "human_severe_defect_rate": 1.0,
                }
            },
            "statistics": {
                "paired_resamples": 10_000,
                "bootstrap_seed": 42,
                "confidence": 0.95,
                "human_pairs_per_tier": 7,
                "human_raters_per_pair": 3,
                "identity_failure_upper_bound": 1.0,
            },
            "human_severe_defect_rubric": ["blank output"],
        }
        suite_sha = "c" * 64

        def model(quantization: str, digest: str) -> dict[str, str]:
            return {
                "model_id": f"synthetic-{quantization.lower()}",
                "bundle_digest": digest,
                "logical_model_revision": "logical-v1",
                "artifact_revision": f"artifact-{quantization.lower()}",
                "quantization": quantization,
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def output_record(
                path_name: str, metrics: dict[str, float], color: int
            ) -> dict[str, object]:
                path = root / path_name
                path.parent.mkdir(parents=True, exist_ok=True)
                image = Image.new("RGB", (8, 8), (color % 255, 20, 30))
                image.putpixel((0, 0), ((color + 1) % 255, 21, 31))
                image.save(path)
                return {
                    "artifact_path": path_name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "width": 8,
                    "height": 8,
                    "pipeline_finite": True,
                    "blank_detected": False,
                    "metrics": metrics,
                }

            case_results = []
            human_pair_ids = []
            color = 1
            for category, metrics in category_metrics.items():
                if category == "conditional_inpaint":
                    continue
                case_id = f"{category}-1"
                human_pair_ids.append(case_id)
                case_results.append(
                    {
                        "id": case_id,
                        "bf16": output_record(
                            f"bf16/{case_id}.png", dict(metrics), color
                        ),
                        "candidate": output_record(
                            f"candidate/{case_id}.png", dict(metrics), color + 1
                        ),
                    }
                )
                color += 2
            identity_metrics = {"structural_identity": 0.8, "face_identity": 0.8}
            identity_results = [
                {
                    "id": "identity-1",
                    "bf16": output_record(
                        "bf16/identity-1.png", dict(identity_metrics), color
                    ),
                    "candidate": output_record(
                        "candidate/identity-1.png", dict(identity_metrics), color + 1
                    ),
                }
            ]
            human_pair_ids.append("identity-1")
            human_reviews = []
            for pair_index, pair_id in enumerate(human_pair_ids):
                for rater_index in range(3):
                    human_reviews.append(
                        {
                            "pair_id": pair_id,
                            "rater_id": f"rater-{rater_index + 1}",
                            "candidate_slot": "A"
                            if (pair_index + rater_index) % 2 == 0
                            else "B",
                            "severe_defect": False,
                            "identity_failure": False
                            if pair_id == "identity-1"
                            else None,
                        }
                    )
            result_input = {
                "schema_version": 1,
                "object": "xeno.image.quality_results",
                "suite": {"version": "synthetic-v1", "sha256": suite_sha},
                "tier": "Q4_K_M",
                "execution": {
                    "paired_bf16_reference": True,
                    "identical_xeno_initial_latent": True,
                    "size": "8x8",
                    "steps": 50,
                    "true_cfg_scale": 4.0,
                    "rng_schema": quality.RNG_SCHEMA,
                    "conditional_inpaint_admitted": False,
                },
                "model_pairs": {
                    role: {
                        "bf16": model("BF16", "a" * 64),
                        "candidate": model("Q4_K_M", "b" * 64),
                    }
                    for role in ("generation", "edit")
                },
                "evaluator_identity": evaluator_identity,
                "case_results": case_results,
                "identity_results": identity_results,
                "human_review_protocol": {
                    "blinded": True,
                    "randomized": True,
                    "rater_ids_pseudonymous": True,
                    "rubric_sha256": quality.canonical_json_sha256(
                        suite["human_severe_defect_rubric"]
                    ),
                },
                "human_reviews": human_reviews,
            }
            report = quality.compile_report(
                suite,
                suite_sha,
                result_input,
                "d" * 64,
                root,
            )
            self.assertEqual(report["status"], "passed")
            self.assertFalse(report["production_support"])
            self.assertFalse(report["admission"]["production_claim_permitted"])
            self.assertEqual(report["coverage"]["case_results"], 6)
            self.assertEqual(report["coverage"]["artifacts_verified"], 14)


if __name__ == "__main__":
    unittest.main()
