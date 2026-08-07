#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import verify_quality_environment as environment


class QualityEnvironmentTests(unittest.TestCase):
    def test_checked_in_lock_matches_frozen_suite(self) -> None:
        lock = environment.load_lock(environment.DEFAULT_LOCK_PATH)
        suite = environment.validate_suite(lock)
        self.assertEqual(suite["suite_version"], "qwen-image-release-v1")
        self.assertFalse(lock["production_support"])

        cpu_lock = environment.load_lock(
            environment.HERE / "quality-environment-cpu-lock.json"
        )
        environment.validate_suite(cpu_lock)
        self.assertEqual(cpu_lock["ocr_pipeline"]["device"], "cpu")

    def test_package_validation_is_fail_closed(self) -> None:
        with self.assertRaisesRegex(environment.QualityEnvironmentError, "paddleocr"):
            environment.validate_versions(
                {"paddleocr": "3.7.0", "paddlepaddle-gpu": "3.3.0"},
                {"paddleocr": "3.6.0", "paddlepaddle-gpu": None},
            )

    def test_inspect_does_not_import_or_execute_evaluators(self) -> None:
        lock = environment.load_lock(environment.DEFAULT_LOCK_PATH)
        with mock.patch.object(environment, "installed_versions", return_value={}) as versions:
            evidence = environment.environment_evidence(lock, require_host=False)
        versions.assert_called_once()
        self.assertEqual(evidence["status"], "metadata_valid")
        self.assertNotIn("ocr_smoke", evidence)

    def test_snapshot_manifest_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b.bin").write_bytes(b"b")
            (root / "a.bin").write_bytes(b"a")
            first = environment.snapshot_manifest(root)
            second = environment.snapshot_manifest(root)
        self.assertEqual(first, second)
        self.assertEqual([item["path"] for item in first["files"]], ["a.bin", "b.bin"])

    def test_ocr_result_json_string_is_normalized(self) -> None:
        class Result:
            json = json.dumps({"text": "XENO STUDIO"})

        self.assertEqual(
            environment.serialize_ocr_result(Result()),
            {"text": "XENO STUDIO"},
        )


if __name__ == "__main__":
    unittest.main()
