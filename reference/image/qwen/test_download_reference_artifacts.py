#!/usr/bin/env python3

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import download_reference_artifacts


class DownloadReferenceArtifactsTests(unittest.TestCase):
    def make_symlink_or_skip(self, link: Path, target: Path) -> None:
        try:
            link.symlink_to(os.path.relpath(target, link.parent))
        except OSError as error:
            if os.name == "nt" and getattr(error, "winerror", None) == 1314:
                self.skipTest("Windows symlink creation privilege is unavailable")
            raise

    def test_contained_regular_hugging_face_cache_file_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blob = root / "huggingface-hub" / "blobs" / "payload"
            blob.parent.mkdir(parents=True)
            blob.write_bytes(b"pinned payload")

            resolved = download_reference_artifacts.resolved_hugging_face_cache_file(root, blob)

            self.assertEqual(resolved, blob.resolve())

    def test_hugging_face_cache_symlink_resolves_to_contained_regular_blob(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blob = root / "huggingface-hub" / "blobs" / "payload"
            blob.parent.mkdir(parents=True)
            blob.write_bytes(b"pinned payload")
            cached = root / "huggingface-hub" / "snapshots" / "revision" / "artifact.bin"
            cached.parent.mkdir(parents=True)
            self.make_symlink_or_skip(cached, blob)

            resolved = download_reference_artifacts.resolved_hugging_face_cache_file(root, cached)

            self.assertEqual(resolved, blob.resolve())
            self.assertTrue(resolved.is_file())
            self.assertFalse(resolved.is_symlink())

    def test_hugging_face_cache_symlink_cannot_escape_cache_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / "outside.bin"
            outside.write_bytes(b"untrusted payload")

            with self.assertRaisesRegex(RuntimeError, "escapes cache root"):
                download_reference_artifacts.resolved_hugging_face_cache_file(root, outside)


if __name__ == "__main__":
    unittest.main()
