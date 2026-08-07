#!/usr/bin/env python3
"""Run and checkpoint the frozen CLIP, DINOv2, and PaddleOCR-VL metrics."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Callable

from PIL import Image

import prepare_quality_review
import evaluate_quality_suite
import verify_quality_environment


HERE = Path(__file__).resolve().parent
SUITE_PATH = HERE.parents[2] / "tests" / "common" / "image-quality-suite.json"
METRIC_SCHEMA = "xeno-image-quality-metrics-v1"
SIDES = ("bf16", "candidate")


class EvaluatorError(RuntimeError):
    pass


def load_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    plan_sha256 = prepare_quality_review.validate_plan(plan, plan_bytes)
    suite, suite_sha256 = evaluate_quality_suite.load_suite(SUITE_PATH)
    if suite_sha256 != prepare_quality_review.SUITE_SHA256:
        raise EvaluatorError("frozen suite digest drift")
    models = prepare_quality_review.read_json(args.models)
    snapshots = models.get("model_snapshots")
    if not isinstance(snapshots, dict):
        raise EvaluatorError("models evidence is missing model_snapshots")
    for role, expected in suite["evaluators"].items():
        if role == "mask_leakage":
            continue
        snapshot_role = {
            "ocr": "ocr",
            "prompt_alignment": "prompt_alignment",
            "structural_identity": "structural_identity",
            "face_identity": "structural_identity",
        }[role]
        record = snapshots.get(snapshot_role)
        if not isinstance(record, dict):
            raise EvaluatorError(f"models evidence is missing {snapshot_role}")
        if record.get("repository") != expected["model"] or record.get("revision") != expected["revision"]:
            raise EvaluatorError(f"models evidence drift for {snapshot_role}")
        snapshot = Path(record.get("snapshot", ""))
        if not snapshot.is_dir() or snapshot.name != expected["revision"]:
            raise EvaluatorError(f"immutable snapshot is unavailable for {snapshot_role}")
    return plan, plan_sha256, suite, models


def output_identity(
    case: dict[str, Any], side: str, artifact_root: Path, plan_sha256: str
) -> tuple[Path, dict[str, Any]]:
    relative = case["artifacts"][side]
    expected = f"{side}/{case['category']}/{case['id']}.png"
    if relative != expected:
        raise EvaluatorError(f"artifact path drift for {case['id']} {side}")
    path = artifact_root / relative
    checkpoint = path.with_suffix(".png.xrt.json")
    if not path.is_file() or path.is_symlink() or not checkpoint.is_file() or checkpoint.is_symlink():
        raise EvaluatorError(f"missing corpus output/checkpoint: {relative}")
    record = prepare_quality_review.read_json(checkpoint)
    digest = prepare_quality_review.sha256_file(path)
    if (
        record.get("schema_version") != 1
        or record.get("plan_sha256") != plan_sha256
        or record.get("id") != case["id"]
        or record.get("category") != case["category"]
        or record.get("side") != side
        or record.get("artifact_path") != relative
        or record.get("output_sha256") != digest
        or record.get("width") != 1024
        or record.get("height") != 1024
    ):
        raise EvaluatorError(f"corpus identity drift for {relative}")
    return path, record


def checkpoint_path(root: Path, evaluator: str, side: str, case_id: str) -> Path:
    return root / evaluator / side / f"{case_id}.json"


def reuse_checkpoint(
    path: Path, evaluator: str, plan_sha256: str, case_id: str, side: str, image_sha256: str
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    record = prepare_quality_review.read_json(path)
    if (
        record.get("schema_version") != 1
        or record.get("metric_schema") != METRIC_SCHEMA
        or record.get("evaluator") != evaluator
        or record.get("plan_sha256") != plan_sha256
        or record.get("id") != case_id
        or record.get("side") != side
        or record.get("image_sha256") != image_sha256
    ):
        raise EvaluatorError(f"metric checkpoint drift: {path}")
    values = record.get("metrics")
    if not isinstance(values, dict) or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in values.values()
    ):
        raise EvaluatorError(f"metric checkpoint contains non-finite values: {path}")
    return record


def write_checkpoint(
    root: Path,
    evaluator: str,
    plan_sha256: str,
    case: dict[str, Any],
    side: str,
    image_sha256: str,
    metrics: dict[str, float],
    observation: Any = None,
) -> None:
    path = checkpoint_path(root, evaluator, side, case["id"])
    if path.exists():
        raise EvaluatorError(f"refusing to overwrite metric checkpoint without --resume: {path}")
    record = {
        "schema_version": 1,
        "metric_schema": METRIC_SCHEMA,
        "evaluator": evaluator,
        "plan_sha256": plan_sha256,
        "id": case["id"],
        "category": case["category"],
        "side": side,
        "image_sha256": image_sha256,
        "metrics": metrics,
    }
    if observation is not None:
        record["observation"] = observation
    prepare_quality_review.atomic_write(path, prepare_quality_review.canonical_bytes(record))


def generation_cases(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [case for case in plan["cases"] if case["category"].startswith("generation_")]


def identity_cases(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        *[case for case in plan["cases"] if case["category"].startswith("edit_")],
        *plan["identity_preservation_pairs"],
    ]


def snapshot_path(models: dict[str, Any], role: str) -> Path:
    return Path(models["model_snapshots"][role]["snapshot"])


def chunks(items: list[Any], size: int) -> list[list[Any]]:
    if size < 1:
        raise EvaluatorError("batch size must be positive")
    return [items[index : index + size] for index in range(0, len(items), size)]


def run_prompt_alignment(args: argparse.Namespace) -> None:
    plan, plan_sha256, _suite, models = load_inputs(args)
    artifact_root = args.artifact_root.resolve(strict=True)
    pending = []
    for case in generation_cases(plan):
        for side in SIDES:
            path, output = output_identity(case, side, artifact_root, plan_sha256)
            existing = reuse_checkpoint(
                checkpoint_path(args.checkpoint_root, "prompt_alignment", side, case["id"]),
                "prompt_alignment",
                plan_sha256,
                case["id"],
                side,
                output["output_sha256"],
            )
            if existing is None or not args.resume:
                if existing is not None:
                    raise EvaluatorError("existing prompt checkpoint requires --resume")
                pending.append((case, side, path, output["output_sha256"]))
    if not pending:
        return
    import open_clip
    import torch

    snapshot = snapshot_path(models, "prompt_alignment")
    identifier = f"local-dir:{snapshot}"
    model, _, preprocess = open_clip.create_model_and_transforms(
        identifier,
        device=args.device,
        precision="fp32",
        require_pretrained=True,
    )
    tokenizer = open_clip.get_tokenizer(identifier)
    model.eval()
    with torch.inference_mode():
        for batch in chunks(pending, args.batch_size):
            images = []
            for _, _, path, _ in batch:
                with Image.open(path) as loaded:
                    images.append(preprocess(loaded.convert("RGB")))
            image_input = torch.stack(images).to(args.device)
            text_input = tokenizer([item[0]["request"]["prompt"] for item in batch]).to(args.device)
            image_features = torch.nn.functional.normalize(model.encode_image(image_input), dim=-1)
            text_features = torch.nn.functional.normalize(model.encode_text(text_input), dim=-1)
            scores = (image_features * text_features).sum(dim=-1).detach().cpu().tolist()
            for item, score in zip(batch, scores, strict=True):
                case, side, _, digest = item
                write_checkpoint(
                    args.checkpoint_root,
                    "prompt_alignment",
                    plan_sha256,
                    case,
                    side,
                    digest,
                    {"prompt_alignment": float(score)},
                )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def fixture_ids(request: dict[str, Any]) -> list[str]:
    plural = request.get("source_fixtures") or []
    singular = request.get("source_fixture")
    if plural and singular:
        raise EvaluatorError("edit case has ambiguous source fixtures")
    values = plural or ([singular] if singular else [])
    if not 1 <= len(values) <= 3 or any(not re.fullmatch(r"[A-Za-z0-9_-]+", value) for value in values):
        raise EvaluatorError("edit case has invalid source fixtures")
    return values


def face_crop(image: Image.Image) -> Image.Image:
    width, height = image.size
    return image.crop((round(width * 0.25), round(height * 0.16), round(width * 0.75), round(height * 0.68)))


def run_identity(args: argparse.Namespace) -> None:
    plan, plan_sha256, _suite, models = load_inputs(args)
    artifact_root = args.artifact_root.resolve(strict=True)
    fixture_root = args.fixture_root.resolve(strict=True)
    pending = []
    for case in identity_cases(plan):
        for side in SIDES:
            path, output = output_identity(case, side, artifact_root, plan_sha256)
            existing = reuse_checkpoint(
                checkpoint_path(args.checkpoint_root, "identity", side, case["id"]),
                "identity",
                plan_sha256,
                case["id"],
                side,
                output["output_sha256"],
            )
            if existing is None or not args.resume:
                if existing is not None:
                    raise EvaluatorError("existing identity checkpoint requires --resume")
                pending.append((case, side, path, output["output_sha256"]))
    if not pending:
        return
    import torch
    from transformers import AutoImageProcessor, AutoModel

    snapshot = snapshot_path(models, "structural_identity")
    processor = AutoImageProcessor.from_pretrained(snapshot, local_files_only=True)
    model = AutoModel.from_pretrained(snapshot, local_files_only=True).to(args.device).eval()

    def embed(images: list[Image.Image]) -> Any:
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(args.device) for key, value in inputs.items()}
        with torch.inference_mode():
            vectors = model(**inputs).last_hidden_state[:, 0, :]
            return torch.nn.functional.normalize(vectors, dim=-1)

    for case, side, output_path, digest in pending:
        sources = []
        for fixture in fixture_ids(case["request"]):
            path = fixture_root / f"{fixture}.png"
            if not path.is_file() or path.is_symlink():
                raise EvaluatorError(f"missing source fixture: {fixture}")
            with Image.open(path) as loaded:
                sources.append(loaded.convert("RGB").copy())
        with Image.open(output_path) as loaded:
            output = loaded.convert("RGB").copy()
        full = embed([*sources, output])
        structural = float((full[:-1] @ full[-1]).mean().detach().cpu())
        face = embed([*[face_crop(image) for image in sources], face_crop(output)])
        face_identity = float((face[:-1] @ face[-1]).mean().detach().cpu())
        write_checkpoint(
            args.checkpoint_root,
            "identity",
            plan_sha256,
            case,
            side,
            digest,
            {"structural_identity": structural, "face_identity": face_identity},
        )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def normalize_ocr_text(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value)
    return " ".join("".join(character if character.isalnum() else " " for character in normalized).split())


def edit_distance(left: list[Any], right: list[Any]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, 1):
        current = [left_index]
        for right_index, right_value in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def error_rates(expected: str, candidates: list[str]) -> tuple[float, float, str]:
    reference = normalize_ocr_text(expected)
    normalized = [normalize_ocr_text(value) for value in candidates]
    normalized = [value for value in normalized if value]
    if not normalized:
        normalized = [""]
    cer = lambda value: edit_distance(list(reference), list(value)) / max(1, len(reference))
    wer = lambda value: edit_distance(reference.split(), value.split()) / max(1, len(reference.split()))
    best = min(normalized, key=lambda value: (cer(value), wer(value), value))
    return cer(best), wer(best), best


def ocr_candidates(value: Any) -> list[str]:
    candidates: list[tuple[int, str]] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for text_key in ("block_content", "rec_text"):
                if isinstance(item.get(text_key), str):
                    order = item.get("block_order")
                    candidates.append(
                        (order if isinstance(order, int) else len(candidates), item[text_key])
                    )
            for nested in item.values():
                visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)

    visit(value)
    ordered = [text for _, text in sorted(candidates)]
    result = ordered.copy()
    if ordered:
        result.append(" ".join(ordered))
    return list(dict.fromkeys(result))


def run_ocr(args: argparse.Namespace) -> None:
    plan, plan_sha256, _suite, models = load_inputs(args)
    artifact_root = args.artifact_root.resolve(strict=True)
    cases = [case for case in generation_cases(plan) if case["category"] == "generation_typography"]
    pending = []
    for case in cases:
        for side in SIDES:
            path, output = output_identity(case, side, artifact_root, plan_sha256)
            existing = reuse_checkpoint(
                checkpoint_path(args.checkpoint_root, "ocr", side, case["id"]),
                "ocr",
                plan_sha256,
                case["id"],
                side,
                output["output_sha256"],
            )
            if existing is None or not args.resume:
                if existing is not None:
                    raise EvaluatorError("existing OCR checkpoint requires --resume")
                pending.append((case, side, path, output["output_sha256"]))
    if not pending:
        return
    from paddleocr import PaddleOCRVL

    os.environ["PADDLE_PDX_MODEL_SOURCE"] = "HuggingFace"
    snapshot = snapshot_path(models, "ocr")
    pipeline = PaddleOCRVL(
        pipeline_version="v1.6",
        device=args.device,
        use_layout_detection=False,
        vl_rec_model_dir=str(snapshot),
    )
    for case, side, path, digest in pending:
        results = [verify_quality_environment.serialize_ocr_result(item) for item in pipeline.predict(str(path))]
        if not results:
            raise EvaluatorError(f"OCR returned no result for {case['id']} {side}")
        candidates = ocr_candidates(results)
        cer, wer, recognized = error_rates(case["request"]["expected_text"], candidates)
        write_checkpoint(
            args.checkpoint_root,
            "ocr",
            plan_sha256,
            case,
            side,
            digest,
            {"character_error_rate": cer, "word_error_rate": wer},
            {"recognized_text": recognized, "candidate_count": len(candidates)},
        )


def metric_values(
    root: Path,
    evaluator: str,
    case: dict[str, Any],
    side: str,
    plan_sha256: str,
    image_sha256: str,
) -> dict[str, float]:
    record = reuse_checkpoint(
        checkpoint_path(root, evaluator, side, case["id"]),
        evaluator,
        plan_sha256,
        case["id"],
        side,
        image_sha256,
    )
    if record is None:
        raise EvaluatorError(f"missing {evaluator} checkpoint for {case['id']} {side}")
    return {key: float(value) for key, value in record["metrics"].items()}


def model_record(record: dict[str, Any]) -> dict[str, Any]:
    official = [
        revision
        for source, revision in record["source_revisions"].items()
        if source.startswith("Qwen/Qwen-Image")
    ]
    if len(official) != 1:
        raise EvaluatorError("corpus model checkpoint does not identify one official logical revision")
    return {
        "model_id": record["model_id"],
        "bundle_digest": record["bundle_digest"],
        "logical_model_revision": official[0],
        "artifact_revision": record["artifact_revision"],
        "quantization": record["quantization"],
    }


def output_entry(
    case: dict[str, Any],
    side: str,
    path: Path,
    record: dict[str, Any],
    metrics: dict[str, float],
) -> dict[str, Any]:
    with Image.open(path) as image:
        image.load()
        extrema = image.convert("RGB").getextrema()
    return {
        "artifact_path": case["artifacts"][side],
        "sha256": record["output_sha256"],
        "width": record["width"],
        "height": record["height"],
        "pipeline_finite": True,
        "blank_detected": all(low == high for low, high in extrema),
        "metrics": metrics,
    }


def export_metrics(args: argparse.Namespace) -> None:
    plan, plan_sha256, suite, models = load_inputs(args)
    artifact_root = args.artifact_root.resolve(strict=True)
    fixture_root = args.fixture_root.resolve(strict=True)
    prepare_quality_review.verify_corpus(plan, plan_sha256, artifact_root)
    role_records: dict[str, dict[str, dict[str, Any]]] = {
        "generation": {},
        "edit": {},
    }
    case_results = []
    for case in plan["cases"]:
        pair = {"id": case["id"]}
        role = "generation" if case["category"].startswith("generation_") else "edit"
        for side in SIDES:
            path, record = output_identity(case, side, artifact_root, plan_sha256)
            previous = role_records[role].setdefault(side, record)
            if model_record(previous) != model_record(record):
                raise EvaluatorError(f"model identity drift within {role} {side}")
            if role == "generation":
                metrics = metric_values(
                    args.checkpoint_root,
                    "prompt_alignment",
                    case,
                    side,
                    plan_sha256,
                    record["output_sha256"],
                )
                if case["category"] == "generation_typography":
                    metrics.update(
                        metric_values(
                            args.checkpoint_root,
                            "ocr",
                            case,
                            side,
                            plan_sha256,
                            record["output_sha256"],
                        )
                    )
            else:
                metrics = metric_values(
                    args.checkpoint_root,
                    "identity",
                    case,
                    side,
                    plan_sha256,
                    record["output_sha256"],
                )
            pair[side] = output_entry(case, side, path, record, metrics)
        case_results.append(pair)
    identity_results = []
    for case in plan["identity_preservation_pairs"]:
        pair = {"id": case["id"]}
        for side in SIDES:
            path, record = output_identity(case, side, artifact_root, plan_sha256)
            previous = role_records["edit"].setdefault(side, record)
            if model_record(previous) != model_record(record):
                raise EvaluatorError(f"model identity drift within edit {side}")
            metrics = metric_values(
                args.checkpoint_root,
                "identity",
                case,
                side,
                plan_sha256,
                record["output_sha256"],
            )
            pair[side] = output_entry(case, side, path, record, metrics)
        identity_results.append(pair)
    output = {
        "schema_version": 1,
        "object": "xeno.image.quality_metric_export",
        "suite": plan["suite"],
        "tier": plan["tier"],
        "execution": plan["execution"],
        "model_pairs": {
            role: {side: model_record(role_records[role][side]) for side in SIDES}
            for role in ("generation", "edit")
        },
        "evaluator_identity": suite["evaluators"],
        "evaluator_run": {
            "metric_schema": METRIC_SCHEMA,
            "models_evidence_sha256": prepare_quality_review.sha256_file(args.models),
            "model_manifest_sha256": {
                role: models["model_snapshots"][role]["manifest_sha256"]
                for role in ("prompt_alignment", "structural_identity", "ocr")
            },
            "fixture_root_sha256": canonical_fixture_digest(fixture_root),
        },
        "case_results": case_results,
        "identity_results": identity_results,
    }
    prepare_quality_review.atomic_write(args.output, prepare_quality_review.canonical_bytes(output))


def canonical_fixture_digest(root: Path) -> str:
    records = []
    for path in sorted(root.glob("*.png")):
        if path.is_symlink() or not path.is_file():
            raise EvaluatorError("fixture root contains a non-regular PNG")
        records.append({"path": path.name, "sha256": prepare_quality_review.sha256_file(path)})
    return prepare_quality_review.canonical_digest(records)


def assemble(args: argparse.Namespace) -> None:
    metrics = prepare_quality_review.read_json(args.metrics)
    reviews = prepare_quality_review.read_json(args.human_reviews)
    if metrics.get("object") != "xeno.image.quality_metric_export":
        raise EvaluatorError("metrics input is not a quality metric export")
    if set(reviews) != {"human_review_protocol", "human_reviews"}:
        raise EvaluatorError("human review input has unexpected fields")
    output = dict(metrics)
    output["object"] = "xeno.image.quality_results"
    output.update(reviews)
    prepare_quality_review.atomic_write(args.output, prepare_quality_review.canonical_bytes(output))


def add_common(parser: argparse.ArgumentParser, *, fixtures: bool = False) -> None:
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    if fixtures:
        parser.add_argument("--fixture-root", type=Path, required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prompt = commands.add_parser("prompt-alignment")
    add_common(prompt)
    prompt.add_argument("--device", default="cuda")
    prompt.add_argument("--batch-size", type=int, default=4)
    prompt.add_argument("--resume", action="store_true")
    identity = commands.add_parser("identity")
    add_common(identity, fixtures=True)
    identity.add_argument("--device", default="cuda")
    identity.add_argument("--resume", action="store_true")
    ocr = commands.add_parser("ocr")
    add_common(ocr)
    ocr.add_argument("--device", default="gpu:0")
    ocr.add_argument("--resume", action="store_true")
    export = commands.add_parser("export")
    add_common(export, fixtures=True)
    export.add_argument("--output", type=Path, required=True)
    combine = commands.add_parser("assemble")
    combine.add_argument("--metrics", type=Path, required=True)
    combine.add_argument("--human-reviews", type=Path, required=True)
    combine.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    actions: dict[str, Callable[[argparse.Namespace], None]] = {
        "prompt-alignment": run_prompt_alignment,
        "identity": run_identity,
        "ocr": run_ocr,
        "export": export_metrics,
        "assemble": assemble,
    }
    try:
        actions[args.command](args)
    except (OSError, ValueError, KeyError, TypeError, EvaluatorError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}))
        return 1
    print(json.dumps({"status": "ok", "command": args.command}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
