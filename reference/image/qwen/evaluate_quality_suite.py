#!/usr/bin/env python3
"""Plan and compile the frozen Qwen Image quantization quality gate.

This reference-only tool does not run inside xeno-rt. It turns the frozen suite
into a deterministic execution plan and compiles already-evaluated BF16 versus
quantized artifacts into an auditable admission report. It deliberately fails
closed on incomplete cases, drifted evaluator identities, missing artifacts,
or insufficient blinded human review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable

import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_SUITE_PATH = REPO_ROOT / "tests" / "common" / "image-quality-suite.json"
MAX_JSON_BYTES = 128 * 1024 * 1024
MAX_IMAGE_BYTES = 64 * 1024 * 1024
RNG_SCHEMA = "xrt-normal-v1-splitmix64-marsaglia-f32le"
HEX_64 = re.compile(r"^[0-9a-f]{64}$")
RATER_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


class QualityAdmissionError(RuntimeError):
    """Raised when an input cannot support a quality admission decision."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(encoded)


def read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        size = path.stat().st_size
    except FileNotFoundError as error:
        raise QualityAdmissionError(f"missing JSON input: {path}") from error
    if size <= 0 or size > MAX_JSON_BYTES:
        raise QualityAdmissionError(
            f"JSON input must be between 1 and {MAX_JSON_BYTES} bytes: {path}"
        )
    encoded = path.read_bytes()
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualityAdmissionError(f"invalid UTF-8 JSON input {path}: {error}") from error
    if not isinstance(value, dict):
        raise QualityAdmissionError(f"JSON input must be an object: {path}")
    return value, encoded


def atomic_write(path: Path, data: bytes, overwrite: bool) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise QualityAdmissionError(f"output already exists (use --overwrite): {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise QualityAdmissionError(f"output appeared during write: {path}") from error
            except OSError as error:
                raise QualityAdmissionError(
                    f"could not atomically create admission output {path}: {error}"
                ) from error
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)


def require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise QualityAdmissionError(f"{label} must be an object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise QualityAdmissionError(f"{label} must be an array")
    return value


def require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise QualityAdmissionError(f"{label} must be a boolean")
    return value


def require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise QualityAdmissionError(f"{label} must be a non-empty string")
    return value


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QualityAdmissionError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise QualityAdmissionError(f"{label} must be finite")
    return number


def relative_path(root: Path, raw: Any, label: str) -> Path:
    text = require_string(raw, label)
    candidate = Path(text)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise QualityAdmissionError(f"{label} must be a contained relative path")
    resolved_root = root.resolve()
    lexical = resolved_root / candidate
    if lexical.is_symlink():
        raise QualityAdmissionError(f"{label} must not name a symlink")
    resolved = lexical.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise QualityAdmissionError(f"{label} escapes the selected artifact root")
    return resolved


def load_suite(path: Path) -> tuple[dict[str, Any], str]:
    suite, encoded = read_json(path)
    if suite.get("schema_version") != 1 or suite.get("status") != "frozen":
        raise QualityAdmissionError("quality suite must be frozen schema version 1")
    require_string(suite.get("suite_version"), "quality suite version")
    categories = require_object(suite.get("categories"), "quality suite categories")
    counts = require_object(suite.get("category_counts"), "quality suite category_counts")
    observed: dict[str, int] = {}
    seen: set[str] = set()
    for category, raw_cases in categories.items():
        cases = require_list(raw_cases, f"quality suite category {category}")
        observed[category] = len(cases)
        for raw_case in cases:
            case = require_object(raw_case, f"quality suite case in {category}")
            case_id = require_string(case.get("id"), f"quality suite case ID in {category}")
            if case_id in seen:
                raise QualityAdmissionError(f"duplicate quality suite case ID: {case_id}")
            seen.add(case_id)
    if observed != counts:
        raise QualityAdmissionError(
            f"quality suite category count drift: expected {counts}, observed {observed}"
        )
    identities = require_list(
        suite.get("identity_preservation_pairs"),
        "quality suite identity_preservation_pairs",
    )
    identity_ids = [
        require_string(require_object(item, "identity pair").get("id"), "identity pair ID")
        for item in identities
    ]
    if len(identity_ids) < 50 or len(identity_ids) != len(set(identity_ids)):
        raise QualityAdmissionError(
            "quality suite requires at least 50 uniquely identified identity pairs"
        )
    statistics = require_object(suite.get("statistics"), "quality suite statistics")
    if statistics.get("bootstrap_rng") != "PCG64":
        raise QualityAdmissionError("quality suite bootstrap RNG must remain PCG64")
    if int(statistics.get("paired_resamples", 0)) < 10_000:
        raise QualityAdmissionError("quality suite requires at least 10,000 resamples")
    if statistics.get("degradation_bound") != "one_sided_bootstrap_upper":
        raise QualityAdmissionError("quality suite degradation bound drift")
    if statistics.get("human_defect_bound") != "one_sided_wilson_upper":
        raise QualityAdmissionError("quality suite human-review bound drift")
    fixtures = require_object(suite.get("fixtures"), "quality suite fixtures")
    repo = REPO_ROOT.resolve()
    for fixture_id, raw_fixture in fixtures.items():
        fixture = require_object(raw_fixture, f"fixture {fixture_id}")
        fixture_path = relative_path(
            repo,
            fixture.get("path"),
            f"fixture {fixture_id} path",
        )
        if not fixture_path.is_file() or fixture_path.is_symlink():
            raise QualityAdmissionError(f"fixture is missing or is a symlink: {fixture_id}")
        expected = require_string(fixture.get("sha256"), f"fixture {fixture_id} sha256")
        if not HEX_64.fullmatch(expected) or sha256_file(fixture_path) != expected:
            raise QualityAdmissionError(f"fixture digest drift: {fixture_id}")
    return suite, sha256_bytes(encoded)


def active_categories(suite: dict[str, Any], include_inpaint: bool) -> list[str]:
    categories = list(require_object(suite["categories"], "quality suite categories"))
    if not include_inpaint:
        categories.remove("conditional_inpaint")
    return categories


def case_index(
    suite: dict[str, Any], include_inpaint: bool
) -> tuple[dict[str, tuple[str, dict[str, Any]]], dict[str, dict[str, Any]]]:
    cases: dict[str, tuple[str, dict[str, Any]]] = {}
    for category in active_categories(suite, include_inpaint):
        for raw_case in suite["categories"][category]:
            case = require_object(raw_case, f"case in {category}")
            cases[case["id"]] = (category, case)
    identities = {
        item["id"]: require_object(item, "identity pair")
        for item in suite["identity_preservation_pairs"]
    }
    return cases, identities


def build_plan(
    suite: dict[str, Any], suite_sha256: str, tier: str, include_inpaint: bool
) -> dict[str, Any]:
    thresholds = require_object(
        suite.get("relative_admission_thresholds"),
        "quality suite relative_admission_thresholds",
    )
    if tier not in thresholds:
        raise QualityAdmissionError(f"unsupported quality tier: {tier}")
    cases, identities = case_index(suite, include_inpaint)
    planned_cases = []
    for case_id, (category, case) in cases.items():
        planned_cases.append(
            {
                "id": case_id,
                "category": category,
                "request": case,
                "artifacts": {
                    "bf16": f"bf16/{category}/{case_id}.png",
                    "candidate": f"candidate/{category}/{case_id}.png",
                },
            }
        )
    planned_identities = []
    for identity_id, case in identities.items():
        planned_identities.append(
            {
                "id": identity_id,
                "category": "identity_preservation",
                "request": case,
                "artifacts": {
                    "bf16": f"bf16/identity_preservation/{identity_id}.png",
                    "candidate": f"candidate/identity_preservation/{identity_id}.png",
                },
            }
        )
    return {
        "schema_version": 1,
        "object": "xeno.image.quality_plan",
        "suite": {
            "version": suite["suite_version"],
            "sha256": suite_sha256,
        },
        "tier": tier,
        "scope": "quantization_quality_only",
        "production_support": False,
        "execution": {
            "paired_bf16_reference": True,
            "identical_xeno_initial_latent": True,
            "size": suite["execution"]["default_size"],
            "steps": suite["execution"]["default_steps"],
            "true_cfg_scale": suite["execution"]["true_cfg_scale"],
            "rng_schema": RNG_SCHEMA,
            "conditional_inpaint_admitted": include_inpaint,
        },
        "required_model_roles": ["generation", "edit"],
        "evaluator_identity": suite["evaluators"],
        "relative_thresholds": thresholds[tier],
        "absolute_quality_floors": {
            name: suite["absolute_quality_floors"][name]
            for name in active_categories(suite, include_inpaint)
        },
        "human_review": {
            "minimum_pairs": suite["statistics"]["human_pairs_per_tier"],
            "raters_per_pair": suite["statistics"]["human_raters_per_pair"],
            "blinded": True,
            "randomized": True,
            "all_identity_pairs_required": True,
            "rubric_sha256": canonical_json_sha256(suite["human_severe_defect_rubric"]),
        },
        "cases": planned_cases,
        "identity_preservation_pairs": planned_identities,
    }


def validate_model_record(record: Any, label: str, quantization: str) -> dict[str, Any]:
    model = require_object(record, label)
    require_string(model.get("model_id"), f"{label}.model_id")
    digest = require_string(model.get("bundle_digest"), f"{label}.bundle_digest")
    if not HEX_64.fullmatch(digest):
        raise QualityAdmissionError(f"{label}.bundle_digest must be lowercase SHA-256")
    require_string(model.get("logical_model_revision"), f"{label}.logical_model_revision")
    require_string(model.get("artifact_revision"), f"{label}.artifact_revision")
    if model.get("quantization") != quantization:
        raise QualityAdmissionError(
            f"{label}.quantization must be {quantization}, got {model.get('quantization')!r}"
        )
    return model


def validate_model_pairs(value: Any, tier: str) -> dict[str, Any]:
    pairs = require_object(value, "model_pairs")
    if set(pairs) != {"generation", "edit"}:
        raise QualityAdmissionError("model_pairs must contain exactly generation and edit")
    for role in ("generation", "edit"):
        pair = require_object(pairs[role], f"model_pairs.{role}")
        if set(pair) != {"bf16", "candidate"}:
            raise QualityAdmissionError(
                f"model_pairs.{role} must contain exactly bf16 and candidate"
            )
        bf16 = validate_model_record(pair["bf16"], f"model_pairs.{role}.bf16", "BF16")
        candidate = validate_model_record(
            pair["candidate"], f"model_pairs.{role}.candidate", tier
        )
        if bf16["logical_model_revision"] != candidate["logical_model_revision"]:
            raise QualityAdmissionError(
                f"model_pairs.{role} does not share one logical model revision"
            )
    return pairs


METRICS_BY_CATEGORY: dict[str, tuple[str, ...]] = {
    "generation_general": ("prompt_alignment",),
    "generation_typography": (
        "prompt_alignment",
        "character_error_rate",
        "word_error_rate",
    ),
    "generation_faces_hands_detail": ("prompt_alignment",),
    "generation_style_color": ("prompt_alignment",),
    "edit_single_image": ("structural_identity", "face_identity"),
    "edit_multi_image": ("structural_identity", "face_identity"),
    "conditional_inpaint": ("protected_pixel_leakage",),
    "identity_preservation": ("structural_identity", "face_identity"),
}


def validate_metric(name: str, value: Any, label: str) -> float:
    number = finite_number(value, label)
    if name in {"prompt_alignment", "structural_identity", "face_identity"}:
        if not -1.0 <= number <= 1.0:
            raise QualityAdmissionError(f"{label} must be in [-1, 1]")
    elif name in {"character_error_rate", "word_error_rate"}:
        if number < 0.0:
            raise QualityAdmissionError(f"{label} must be non-negative")
    elif name == "protected_pixel_leakage":
        if not 0.0 <= number <= 1.0:
            raise QualityAdmissionError(f"{label} must be in [0, 1]")
    return number


def validate_output(
    raw: Any,
    label: str,
    category: str,
    artifact_root: Path,
    expected_size: tuple[int, int],
    seen_paths: set[Path],
) -> dict[str, Any]:
    output = require_object(raw, label)
    expected_keys = {
        "artifact_path",
        "sha256",
        "width",
        "height",
        "pipeline_finite",
        "blank_detected",
        "metrics",
    }
    if set(output) != expected_keys:
        raise QualityAdmissionError(
            f"{label} fields must be exactly {sorted(expected_keys)}"
        )
    path = relative_path(artifact_root, output["artifact_path"], f"{label}.artifact_path")
    if path in seen_paths:
        raise QualityAdmissionError(f"artifact path is reused: {output['artifact_path']}")
    seen_paths.add(path)
    if not path.is_file() or path.is_symlink():
        raise QualityAdmissionError(f"artifact is missing or is a symlink: {path}")
    if path.stat().st_size <= 0 or path.stat().st_size > MAX_IMAGE_BYTES:
        raise QualityAdmissionError(f"artifact byte size is outside the admission bound: {path}")
    expected_digest = require_string(output["sha256"], f"{label}.sha256")
    if not HEX_64.fullmatch(expected_digest) or sha256_file(path) != expected_digest:
        raise QualityAdmissionError(f"artifact digest mismatch: {path}")
    if require_bool(output["pipeline_finite"], f"{label}.pipeline_finite") is not True:
        raise QualityAdmissionError(f"{label} reports non-finite pipeline state")
    if require_bool(output["blank_detected"], f"{label}.blank_detected") is not False:
        raise QualityAdmissionError(f"{label} reports a blank output")
    width = output["width"]
    height = output["height"]
    if isinstance(width, bool) or not isinstance(width, int):
        raise QualityAdmissionError(f"{label}.width must be an integer")
    if isinstance(height, bool) or not isinstance(height, int):
        raise QualityAdmissionError(f"{label}.height must be an integer")
    if (width, height) != expected_size:
        raise QualityAdmissionError(
            f"{label} dimensions {(width, height)} do not match {expected_size}"
        )
    try:
        with Image.open(path) as image:
            if image.format != "PNG" or getattr(image, "n_frames", 1) != 1:
                raise QualityAdmissionError(f"{label} must be one non-animated PNG")
            image.load()
            if image.size != expected_size:
                raise QualityAdmissionError(
                    f"{label} decoded dimensions {image.size} do not match {expected_size}"
                )
            extrema = image.convert("RGB").getextrema()
            if all(low == high for low, high in extrema):
                raise QualityAdmissionError(f"{label} is a strictly uniform image")
    except QualityAdmissionError:
        raise
    except Exception as error:
        raise QualityAdmissionError(f"failed to decode {label}: {error}") from error
    metrics = require_object(output["metrics"], f"{label}.metrics")
    expected_metrics = set(METRICS_BY_CATEGORY[category])
    if set(metrics) != expected_metrics:
        raise QualityAdmissionError(
            f"{label}.metrics must be exactly {sorted(expected_metrics)}"
        )
    normalized_metrics = {
        name: validate_metric(name, metrics[name], f"{label}.metrics.{name}")
        for name in sorted(expected_metrics)
    }
    return {
        "sha256": expected_digest,
        "width": width,
        "height": height,
        "metrics": normalized_metrics,
    }


def validate_result_pairs(
    raw_results: Any,
    expected: dict[str, Any],
    label: str,
    artifact_root: Path,
    expected_size: tuple[int, int],
    seen_paths: set[Path],
) -> dict[str, dict[str, Any]]:
    results = require_list(raw_results, label)
    by_id: dict[str, dict[str, Any]] = {}
    for raw_result in results:
        result = require_object(raw_result, f"{label} entry")
        if set(result) != {"id", "bf16", "candidate"}:
            raise QualityAdmissionError(
                f"{label} entries must contain exactly id, bf16, and candidate"
            )
        result_id = require_string(result["id"], f"{label} entry ID")
        if result_id in by_id:
            raise QualityAdmissionError(f"duplicate {label} ID: {result_id}")
        if result_id not in expected:
            raise QualityAdmissionError(f"unexpected {label} ID: {result_id}")
        expected_record = expected[result_id]
        category = (
            expected_record[0]
            if isinstance(expected_record, tuple)
            else "identity_preservation"
        )
        by_id[result_id] = {
            "category": category,
            "bf16": validate_output(
                result["bf16"],
                f"{label}[{result_id}].bf16",
                category,
                artifact_root,
                expected_size,
                seen_paths,
            ),
            "candidate": validate_output(
                result["candidate"],
                f"{label}[{result_id}].candidate",
                category,
                artifact_root,
                expected_size,
                seen_paths,
            ),
        }
    missing = sorted(set(expected) - set(by_id))
    if missing:
        preview = ", ".join(missing[:8])
        raise QualityAdmissionError(
            f"{label} is incomplete; missing {len(missing)} IDs (first: {preview})"
        )
    return by_id


def bootstrap_mean_upper(
    values: Iterable[float], resamples: int, seed: int, confidence: float
) -> tuple[float, float]:
    samples = np.asarray(list(values), dtype=np.float64)
    if samples.ndim != 1 or samples.size == 0 or not np.isfinite(samples).all():
        raise QualityAdmissionError("bootstrap input must be a non-empty finite vector")
    if resamples < 10_000:
        raise QualityAdmissionError("bootstrap requires at least 10,000 resamples")
    if not 0.5 < confidence < 1.0:
        raise QualityAdmissionError("bootstrap confidence must be between 0.5 and 1")
    generator = np.random.Generator(np.random.PCG64(seed))
    means = np.empty(resamples, dtype=np.float64)
    batch_size = 1_000
    for offset in range(0, resamples, batch_size):
        count = min(batch_size, resamples - offset)
        indices = generator.integers(0, samples.size, size=(count, samples.size))
        means[offset : offset + count] = samples[indices].mean(axis=1)
    upper = float(np.quantile(means, confidence, method="higher"))
    return float(samples.mean()), upper


def wilson_upper(successes: int, total: int, confidence: float) -> float:
    if total <= 0 or not 0 <= successes <= total:
        raise QualityAdmissionError("Wilson interval counts are invalid")
    if not 0.5 < confidence < 1.0:
        raise QualityAdmissionError("Wilson confidence must be between 0.5 and 1")
    z = NormalDist().inv_cdf(confidence)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = proportion + z * z / (2.0 * total)
    spread = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    )
    return min(1.0, (center + spread) / denominator)


def gate(
    name: str,
    observed: float,
    threshold: float,
    comparison: str,
    sample_count: int,
    point_estimate: float | None = None,
) -> dict[str, Any]:
    if comparison == "lte":
        passed = observed <= threshold
    elif comparison == "gte":
        passed = observed >= threshold
    else:
        raise AssertionError(f"unsupported comparison: {comparison}")
    result: dict[str, Any] = {
        "name": name,
        "observed": observed,
        "threshold": threshold,
        "comparison": comparison,
        "sample_count": sample_count,
        "passed": passed,
    }
    if point_estimate is not None:
        result["point_estimate"] = point_estimate
    return result


def category_records(
    pairs: dict[str, dict[str, Any]], category: str
) -> list[dict[str, Any]]:
    return [record for record in pairs.values() if record["category"] == category]


def mean_metric(records: list[dict[str, Any]], side: str, metric: str) -> float:
    values = [record[side]["metrics"][metric] for record in records]
    if not values:
        raise QualityAdmissionError(f"no values for {side}.{metric}")
    return math.fsum(values) / len(values)


def absolute_quality_gates(
    suite: dict[str, Any],
    case_pairs: dict[str, dict[str, Any]],
    identity_pairs: dict[str, dict[str, Any]],
    include_inpaint: bool,
) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    floors = suite["absolute_quality_floors"]
    generation_categories = (
        "generation_general",
        "generation_typography",
        "generation_faces_hands_detail",
        "generation_style_color",
    )
    for category in generation_categories:
        records = category_records(case_pairs, category)
        threshold = float(floors[category]["prompt_alignment_min"])
        for side in ("bf16", "candidate"):
            gates.append(
                gate(
                    f"absolute.{category}.{side}.prompt_alignment_mean",
                    mean_metric(records, side, "prompt_alignment"),
                    threshold,
                    "gte",
                    len(records),
                )
            )
    typography = category_records(case_pairs, "generation_typography")
    for metric, threshold_name in (
        ("character_error_rate", "character_error_rate_max"),
        ("word_error_rate", "word_error_rate_max"),
    ):
        threshold = float(floors["generation_typography"][threshold_name])
        for side in ("bf16", "candidate"):
            gates.append(
                gate(
                    f"absolute.generation_typography.{side}.{metric}_mean",
                    mean_metric(typography, side, metric),
                    threshold,
                    "lte",
                    len(typography),
                )
            )
    for category in ("edit_single_image", "edit_multi_image"):
        records = category_records(case_pairs, category)
        for metric, threshold_name in (
            ("structural_identity", "structural_identity_min"),
            ("face_identity", "face_identity_min"),
        ):
            threshold = float(floors[category][threshold_name])
            for side in ("bf16", "candidate"):
                gates.append(
                    gate(
                        f"absolute.{category}.{side}.{metric}_mean",
                        mean_metric(records, side, metric),
                        threshold,
                        "gte",
                        len(records),
                    )
                )
    identity_records = list(identity_pairs.values())
    for metric in ("structural_identity", "face_identity"):
        threshold = float(suite["evaluators"][metric]["bf16_absolute_floor"])
        for side in ("bf16", "candidate"):
            gates.append(
                gate(
                    f"absolute.identity_preservation.{side}.{metric}_mean",
                    mean_metric(identity_records, side, metric),
                    threshold,
                    "gte",
                    len(identity_records),
                )
            )
    if include_inpaint:
        records = category_records(case_pairs, "conditional_inpaint")
        threshold = float(
            floors["conditional_inpaint"]["protected_pixel_leakage_upper_bound_max"]
        )
        statistics = suite["statistics"]
        for side in ("bf16", "candidate"):
            point, upper = bootstrap_mean_upper(
                [record[side]["metrics"]["protected_pixel_leakage"] for record in records],
                int(statistics["paired_resamples"]),
                int(statistics["bootstrap_seed"]),
                float(statistics["confidence"]),
            )
            gates.append(
                gate(
                    f"absolute.conditional_inpaint.{side}.leakage_bootstrap_upper",
                    upper,
                    threshold,
                    "lte",
                    len(records),
                    point,
                )
            )
    return gates


def relative_quality_gates(
    suite: dict[str, Any],
    tier: str,
    case_pairs: dict[str, dict[str, Any]],
    identity_pairs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    statistics = suite["statistics"]
    resamples = int(statistics["paired_resamples"])
    seed = int(statistics["bootstrap_seed"])
    confidence = float(statistics["confidence"])
    thresholds = suite["relative_admission_thresholds"][tier]
    generation = [
        record
        for record in case_pairs.values()
        if record["category"].startswith("generation_")
    ]
    prompt_declines = []
    for record in generation:
        bf16 = record["bf16"]["metrics"]["prompt_alignment"]
        candidate = record["candidate"]["metrics"]["prompt_alignment"]
        prompt_declines.append((bf16 - candidate) / max(abs(bf16), 1e-12))
    prompt_point, prompt_upper = bootstrap_mean_upper(
        prompt_declines, resamples, seed, confidence
    )
    typography = category_records(case_pairs, "generation_typography")
    cer_increases = [
        record["candidate"]["metrics"]["character_error_rate"]
        - record["bf16"]["metrics"]["character_error_rate"]
        for record in typography
    ]
    cer_point, cer_upper = bootstrap_mean_upper(cer_increases, resamples, seed, confidence)
    identity_records = [
        record
        for record in case_pairs.values()
        if record["category"] in {"edit_single_image", "edit_multi_image"}
    ] + list(identity_pairs.values())
    identity_declines = []
    for record in identity_records:
        identity_declines.append(
            math.fsum(
                record["bf16"]["metrics"][metric]
                - record["candidate"]["metrics"][metric]
                for metric in ("structural_identity", "face_identity")
            )
            / 2.0
        )
    identity_point, identity_upper = bootstrap_mean_upper(
        identity_declines, resamples, seed, confidence
    )
    return [
        gate(
            "relative.prompt_alignment_decline_bootstrap_upper",
            prompt_upper,
            float(thresholds["prompt_alignment_decline_relative"]),
            "lte",
            len(prompt_declines),
            prompt_point,
        ),
        gate(
            "relative.ocr_cer_increase_bootstrap_upper",
            cer_upper,
            float(thresholds["ocr_cer_increase_points"]),
            "lte",
            len(cer_increases),
            cer_point,
        ),
        gate(
            "relative.structural_identity_decline_bootstrap_upper",
            identity_upper,
            float(thresholds["structural_identity_decline_absolute"]),
            "lte",
            len(identity_declines),
            identity_point,
        ),
    ]


def validate_human_reviews(
    suite: dict[str, Any],
    tier: str,
    raw_protocol: Any,
    raw_reviews: Any,
    cases: dict[str, tuple[str, dict[str, Any]]],
    identities: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    protocol = require_object(raw_protocol, "human_review_protocol")
    required_protocol = {
        "blinded",
        "randomized",
        "rater_ids_pseudonymous",
        "rubric_sha256",
    }
    if set(protocol) != required_protocol:
        raise QualityAdmissionError(
            f"human_review_protocol fields must be exactly {sorted(required_protocol)}"
        )
    for key in ("blinded", "randomized", "rater_ids_pseudonymous"):
        if require_bool(protocol[key], f"human_review_protocol.{key}") is not True:
            raise QualityAdmissionError(f"human_review_protocol.{key} must be true")
    expected_rubric = canonical_json_sha256(suite["human_severe_defect_rubric"])
    if protocol["rubric_sha256"] != expected_rubric:
        raise QualityAdmissionError("human severe-defect rubric digest drift")
    reviews = require_list(raw_reviews, "human_reviews")
    allowed = set(cases) | set(identities)
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    slot_counts = {"A": 0, "B": 0}
    for index, raw_review in enumerate(reviews):
        review = require_object(raw_review, f"human_reviews[{index}]")
        expected_keys = {
            "pair_id",
            "rater_id",
            "candidate_slot",
            "severe_defect",
            "identity_failure",
        }
        if set(review) != expected_keys:
            raise QualityAdmissionError(
                f"human_reviews[{index}] fields must be exactly {sorted(expected_keys)}"
            )
        pair_id = require_string(review["pair_id"], f"human_reviews[{index}].pair_id")
        if pair_id not in allowed:
            raise QualityAdmissionError(f"unknown human-review pair: {pair_id}")
        rater = require_string(review["rater_id"], f"human_reviews[{index}].rater_id")
        if not RATER_ID.fullmatch(rater):
            raise QualityAdmissionError(
                f"human_reviews[{index}].rater_id must be a bounded pseudonymous token"
            )
        slot = review["candidate_slot"]
        if slot not in {"A", "B"}:
            raise QualityAdmissionError(
                f"human_reviews[{index}].candidate_slot must be A or B"
            )
        slot_counts[slot] += 1
        require_bool(review["severe_defect"], f"human_reviews[{index}].severe_defect")
        if pair_id in identities:
            require_bool(
                review["identity_failure"],
                f"human_reviews[{index}].identity_failure",
            )
        elif review["identity_failure"] is not None:
            raise QualityAdmissionError(
                f"human_reviews[{index}].identity_failure must be null outside identity pairs"
            )
        by_pair[pair_id].append(review)
    statistics = suite["statistics"]
    raters_per_pair = int(statistics["human_raters_per_pair"])
    for pair_id, pair_reviews in by_pair.items():
        if len(pair_reviews) != raters_per_pair:
            raise QualityAdmissionError(
                f"human-review pair {pair_id} requires exactly {raters_per_pair} ratings"
            )
        raters = {review["rater_id"] for review in pair_reviews}
        if len(raters) != raters_per_pair:
            raise QualityAdmissionError(
                f"human-review pair {pair_id} requires distinct pseudonymous raters"
            )
    required_pairs = int(statistics["human_pairs_per_tier"])
    if len(by_pair) < required_pairs:
        raise QualityAdmissionError(
            f"human review covers {len(by_pair)} pairs; at least {required_pairs} are required"
        )
    missing_identity = sorted(set(identities) - set(by_pair))
    if missing_identity:
        raise QualityAdmissionError(
            f"human review is missing {len(missing_identity)} designated identity pairs"
        )
    reviewed_categories = {
        cases[pair_id][0]
        for pair_id in by_pair
        if pair_id in cases
    }
    missing_categories = sorted(set(category for category, _ in cases.values()) - reviewed_categories)
    if missing_categories:
        raise QualityAdmissionError(
            "human-review stratification omits active categories: "
            + ", ".join(missing_categories)
        )
    if slot_counts["A"] == 0 or slot_counts["B"] == 0:
        raise QualityAdmissionError("randomized human review must exercise both candidate slots")
    majority_defects = 0
    disagreement_pairs = 0
    for pair_reviews in by_pair.values():
        votes = [bool(review["severe_defect"]) for review in pair_reviews]
        majority_defects += sum(votes) > len(votes) // 2
        disagreement_pairs += len(set(votes)) > 1
    identity_failures = 0
    identity_disagreements = 0
    for identity_id in identities:
        votes = [bool(review["identity_failure"]) for review in by_pair[identity_id]]
        identity_failures += sum(votes) > len(votes) // 2
        identity_disagreements += len(set(votes)) > 1
    confidence = float(statistics["confidence"])
    defect_upper = wilson_upper(majority_defects, len(by_pair), confidence)
    identity_upper = wilson_upper(identity_failures, len(identities), confidence)
    thresholds = suite["relative_admission_thresholds"][tier]
    gates = [
        gate(
            "human.severe_defect_wilson_upper",
            defect_upper,
            float(thresholds["human_severe_defect_rate"]),
            "lte",
            len(by_pair),
            majority_defects / len(by_pair),
        ),
        gate(
            "human.identity_failure_wilson_upper",
            identity_upper,
            float(statistics["identity_failure_upper_bound"]),
            "lte",
            len(identities),
            identity_failures / len(identities),
        ),
    ]
    summary = {
        "reviewed_pairs": len(by_pair),
        "ratings": len(reviews),
        "majority_severe_defects": majority_defects,
        "severe_defect_disagreement_pairs": disagreement_pairs,
        "identity_pairs": len(identities),
        "majority_identity_failures": identity_failures,
        "identity_disagreement_pairs": identity_disagreements,
        "candidate_slot_ratings": slot_counts,
        "category_pair_counts": {
            category: sum(
                1
                for pair_id in by_pair
                if pair_id in cases and cases[pair_id][0] == category
            )
            for category in sorted({category for category, _ in cases.values()})
        },
    }
    return gates, summary


def validate_execution(
    raw: Any, suite: dict[str, Any]
) -> tuple[dict[str, Any], bool, tuple[int, int]]:
    execution = require_object(raw, "execution")
    expected_keys = {
        "paired_bf16_reference",
        "identical_xeno_initial_latent",
        "size",
        "steps",
        "true_cfg_scale",
        "rng_schema",
        "conditional_inpaint_admitted",
    }
    if set(execution) != expected_keys:
        raise QualityAdmissionError(f"execution fields must be exactly {sorted(expected_keys)}")
    if require_bool(execution["paired_bf16_reference"], "execution.paired_bf16_reference") is not True:
        raise QualityAdmissionError("execution must use a paired BF16 reference")
    if require_bool(
        execution["identical_xeno_initial_latent"],
        "execution.identical_xeno_initial_latent",
    ) is not True:
        raise QualityAdmissionError("execution must use the identical XENO initial latent")
    if execution["size"] != suite["execution"]["default_size"]:
        raise QualityAdmissionError("execution.size does not match the frozen suite")
    if execution["steps"] != suite["execution"]["default_steps"]:
        raise QualityAdmissionError("execution.steps does not match the frozen suite")
    if finite_number(execution["true_cfg_scale"], "execution.true_cfg_scale") != float(
        suite["execution"]["true_cfg_scale"]
    ):
        raise QualityAdmissionError("execution.true_cfg_scale does not match the frozen suite")
    if execution["rng_schema"] != RNG_SCHEMA:
        raise QualityAdmissionError("execution.rng_schema does not match the frozen XENO schema")
    include_inpaint = require_bool(
        execution["conditional_inpaint_admitted"],
        "execution.conditional_inpaint_admitted",
    )
    width_text, separator, height_text = execution["size"].partition("x")
    if separator != "x" or not width_text.isdigit() or not height_text.isdigit():
        raise QualityAdmissionError("execution.size is invalid")
    expected_size = (int(width_text), int(height_text))
    return execution, include_inpaint, expected_size


def compile_report(
    suite: dict[str, Any],
    suite_sha256: str,
    result_input: dict[str, Any],
    input_sha256: str,
    artifact_root: Path,
) -> dict[str, Any]:
    if result_input.get("schema_version") != 1:
        raise QualityAdmissionError("quality result input must be schema version 1")
    if result_input.get("object") != "xeno.image.quality_results":
        raise QualityAdmissionError("quality result input object must be xeno.image.quality_results")
    identity = require_object(result_input.get("suite"), "suite identity")
    if identity != {"version": suite["suite_version"], "sha256": suite_sha256}:
        raise QualityAdmissionError("quality result input targets a drifted suite identity")
    tier = require_string(result_input.get("tier"), "tier")
    if tier not in suite["relative_admission_thresholds"]:
        raise QualityAdmissionError(f"unsupported quality tier: {tier}")
    execution, include_inpaint, expected_size = validate_execution(
        result_input.get("execution"), suite
    )
    model_pairs = validate_model_pairs(result_input.get("model_pairs"), tier)
    if result_input.get("evaluator_identity") != suite["evaluators"]:
        raise QualityAdmissionError("quality evaluator identity or preprocessing drift")
    cases, identities = case_index(suite, include_inpaint)
    seen_paths: set[Path] = set()
    case_pairs = validate_result_pairs(
        result_input.get("case_results"),
        cases,
        "case_results",
        artifact_root,
        expected_size,
        seen_paths,
    )
    identity_pairs = validate_result_pairs(
        result_input.get("identity_results"),
        identities,
        "identity_results",
        artifact_root,
        expected_size,
        seen_paths,
    )
    gates = absolute_quality_gates(
        suite, case_pairs, identity_pairs, include_inpaint
    )
    gates.extend(relative_quality_gates(suite, tier, case_pairs, identity_pairs))
    human_gates, human_summary = validate_human_reviews(
        suite,
        tier,
        result_input.get("human_review_protocol"),
        result_input.get("human_reviews"),
        cases,
        identities,
    )
    gates.extend(human_gates)
    failed = [item["name"] for item in gates if not item["passed"]]
    statistics = suite["statistics"]
    return {
        "schema_version": 1,
        "object": "xeno.image.quality_admission_report",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not failed else "failed",
        "scope": "quantization_quality_only",
        "production_support": False,
        "tier": tier,
        "suite": identity,
        "input_sha256": input_sha256,
        "model_pairs": model_pairs,
        "execution": execution,
        "evaluator_identity": result_input["evaluator_identity"],
        "coverage": {
            "case_results": len(case_pairs),
            "identity_results": len(identity_pairs),
            "artifacts_verified": len(seen_paths),
            "active_category_counts": {
                category: sum(
                    1 for record in case_pairs.values() if record["category"] == category
                )
                for category in active_categories(suite, include_inpaint)
            },
        },
        "statistics": {
            "bootstrap_rng": "PCG64",
            "bootstrap_seed": int(statistics["bootstrap_seed"]),
            "paired_resamples": int(statistics["paired_resamples"]),
            "confidence": float(statistics["confidence"]),
            "bootstrap_statistic": "mean paired degradation",
            "bootstrap_quantile_method": "higher",
            "rng_reset_per_ordered_metric_vector": True,
            "human_bound": "one-sided Wilson upper",
        },
        "gates": gates,
        "human_review": human_summary,
        "failed_gates": failed,
        "admission": {
            "quantization_quality_gate_passed": not failed,
            "production_claim_permitted": False,
            "reason": (
                "This report covers only the frozen quantization quality policy; "
                "performance, memory, API, CPU fallback, and release gates remain independent."
            ),
        },
    }


def encode_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        type=Path,
        default=DEFAULT_SUITE_PATH,
        help="frozen quality-suite JSON (defaults to the checked-in suite)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="emit a deterministic execution plan")
    plan.add_argument("--tier", required=True)
    plan.add_argument("--include-inpaint", action="store_true")
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--overwrite", action="store_true")

    admit = subparsers.add_parser("admit", help="compile a strict admission report")
    admit.add_argument("--results", type=Path, required=True)
    admit.add_argument("--artifact-root", type=Path, required=True)
    admit.add_argument("--output", type=Path, required=True)
    admit.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        suite, suite_sha256 = load_suite(args.suite.resolve())
        if args.command == "plan":
            plan = build_plan(suite, suite_sha256, args.tier, args.include_inpaint)
            atomic_write(args.output, encode_json(plan), args.overwrite)
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "output": str(args.output),
                        "suite_sha256": suite_sha256,
                        "tier": args.tier,
                        "cases": len(plan["cases"]),
                        "identity_pairs": len(plan["identity_preservation_pairs"]),
                        "conditional_inpaint_admitted": args.include_inpaint,
                    },
                    sort_keys=True,
                )
            )
            return 0
        result_input, input_bytes = read_json(args.results.resolve())
        report = compile_report(
            suite,
            suite_sha256,
            result_input,
            sha256_bytes(input_bytes),
            args.artifact_root.resolve(),
        )
        atomic_write(args.output, encode_json(report), args.overwrite)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "output": str(args.output),
                    "tier": report["tier"],
                    "failed_gates": report["failed_gates"],
                    "production_support": False,
                },
                sort_keys=True,
            )
        )
        return 0 if report["status"] == "passed" else 2
    except QualityAdmissionError as error:
        print(f"quality admission error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
