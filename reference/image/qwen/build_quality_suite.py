#!/usr/bin/env python3
"""Build the license-clean, deterministic Qwen Image quality suite."""

from __future__ import annotations

import binascii
import hashlib
import json
import os
import struct
import tempfile
import zlib
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "image-quality"
SUITE_PATH = REPO_ROOT / "tests" / "common" / "image-quality-suite.json"
WIDTH = 512
HEIGHT = 512


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", binascii.crc32(kind + data) & 0xFFFFFFFF)
    )


def encode_png(width: int, height: int, pixels: bytearray) -> bytes:
    stride = width * 3
    rows = b"".join(b"\x00" + pixels[y * stride : (y + 1) * stride] for y in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + png_chunk(b"IDAT", zlib.compress(rows, 9))
        + png_chunk(b"IEND", b"")
    )


def canvas(color: tuple[int, int, int]) -> bytearray:
    return bytearray(color * (WIDTH * HEIGHT))


def put_pixel(pixels: bytearray, x: int, y: int, color: tuple[int, int, int]) -> None:
    if 0 <= x < WIDTH and 0 <= y < HEIGHT:
        offset = (y * WIDTH + x) * 3
        pixels[offset : offset + 3] = bytes(color)


def rectangle(
    pixels: bytearray,
    left: int,
    top: int,
    right: int,
    bottom: int,
    color: tuple[int, int, int],
) -> None:
    left, top = max(left, 0), max(top, 0)
    right, bottom = min(right, WIDTH), min(bottom, HEIGHT)
    row = bytes(color) * max(right - left, 0)
    for y in range(top, bottom):
        offset = (y * WIDTH + left) * 3
        pixels[offset : offset + len(row)] = row


def circle(
    pixels: bytearray,
    center_x: int,
    center_y: int,
    radius: int,
    color: tuple[int, int, int],
) -> None:
    radius_squared = radius * radius
    for y in range(max(0, center_y - radius), min(HEIGHT, center_y + radius + 1)):
        dy = y - center_y
        horizontal = int((radius_squared - dy * dy) ** 0.5)
        rectangle(pixels, center_x - horizontal, y, center_x + horizontal + 1, y + 1, color)


def line(
    pixels: bytearray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
    thickness: int = 3,
) -> None:
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        rectangle(
            pixels,
            x0 - thickness // 2,
            y0 - thickness // 2,
            x0 + thickness // 2 + 1,
            y0 + thickness // 2 + 1,
            color,
        )
        if x0 == x1 and y0 == y1:
            break
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += sx
        if doubled <= dx:
            error += dx
            y0 += sy


def avatar_png(index: int) -> bytes:
    palettes = [
        ((36, 50, 74), (233, 188, 142), (31, 27, 25), (52, 152, 219)),
        ((76, 36, 60), (116, 76, 52), (19, 15, 13), (241, 196, 15)),
        ((28, 79, 68), (244, 205, 170), (104, 67, 42), (231, 76, 60)),
        ((66, 45, 91), (183, 128, 91), (42, 29, 20), (46, 204, 113)),
        ((91, 58, 31), (249, 214, 185), (205, 97, 51), (155, 89, 182)),
        ((25, 62, 92), (91, 62, 46), (12, 11, 10), (230, 126, 34)),
        ((73, 84, 37), (224, 172, 126), (78, 50, 29), (52, 73, 94)),
        ((82, 42, 42), (198, 139, 94), (28, 20, 17), (26, 188, 156)),
        ((35, 68, 82), (241, 194, 158), (219, 154, 67), (142, 68, 173)),
        ((60, 48, 82), (139, 91, 66), (36, 24, 18), (22, 160, 133)),
    ]
    background, skin, hair, shirt = palettes[index]
    pixels = canvas(background)
    rectangle(pixels, 0, 400, WIDTH, HEIGHT, tuple(max(value - 12, 0) for value in background))
    rectangle(pixels, 152, 328, 360, 512, shirt)
    circle(pixels, 256, 218, 118, hair)
    circle(pixels, 256, 230, 98, skin)
    # Vary hairline, eye spacing, mouth, and accessory to give each identity a
    # stable, machine-generated signature without importing third-party art.
    hairline = 178 + (index % 4) * 8
    rectangle(pixels, 164, 126, 348, hairline, hair)
    eye_offset = 38 + (index % 3) * 5
    eye_y = 222 + (index % 2) * 5
    circle(pixels, 256 - eye_offset, eye_y, 8, (24, 24, 28))
    circle(pixels, 256 + eye_offset, eye_y, 8, (24, 24, 28))
    circle(pixels, 254, 258, 5, tuple(max(value - 35, 0) for value in skin))
    line(pixels, 224, 285 + index % 4, 288, 285 - index % 4, (126, 45, 55), 5)
    if index % 2 == 0:
        rectangle(pixels, 190, eye_y - 17, 246, eye_y + 18, (45, 55, 65))
        rectangle(pixels, 266, eye_y - 17, 322, eye_y + 18, (45, 55, 65))
        rectangle(pixels, 246, eye_y - 2, 266, eye_y + 3, (45, 55, 65))
        rectangle(pixels, 195, eye_y - 12, 241, eye_y + 13, skin)
        rectangle(pixels, 271, eye_y - 12, 317, eye_y + 13, skin)
        circle(pixels, 256 - eye_offset, eye_y, 7, (24, 24, 28))
        circle(pixels, 256 + eye_offset, eye_y, 7, (24, 24, 28))
    else:
        accent = tuple(min(value + 45, 255) for value in shirt)
        rectangle(pixels, 168, 116, 344, 135, accent)
        rectangle(pixels, 198, 88, 314, 120, accent)
    return encode_png(WIDTH, HEIGHT, pixels)


def mask_png(index: int) -> bytes:
    pixels = canvas((0, 0, 0))
    if index % 2 == 0:
        circle(pixels, 256, 118 + (index % 5) * 32, 55 + index * 2, (255, 255, 255))
    else:
        left = 38 + index * 21
        rectangle(pixels, left, 55, min(left + 130, WIDTH), 455, (255, 255, 255))
    return encode_png(WIDTH, HEIGHT, pixels)


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def build_fixtures() -> dict[str, dict[str, Any]]:
    fixtures: dict[str, dict[str, Any]] = {}
    for index in range(10):
        avatar_name = f"avatar-{index + 1:02d}.png"
        avatar = avatar_png(index)
        atomic_write(FIXTURE_DIR / avatar_name, avatar)
        fixtures[f"avatar-{index + 1:02d}"] = {
            "path": f"tests/fixtures/image-quality/{avatar_name}",
            "sha256": hashlib.sha256(avatar).hexdigest(),
            "width": WIDTH,
            "height": HEIGHT,
            "kind": "procedural_identity_portrait",
        }

        mask_name = f"mask-{index + 1:02d}.png"
        mask = mask_png(index)
        atomic_write(FIXTURE_DIR / mask_name, mask)
        fixtures[f"mask-{index + 1:02d}"] = {
            "path": f"tests/fixtures/image-quality/{mask_name}",
            "sha256": hashlib.sha256(mask).hexdigest(),
            "width": WIDTH,
            "height": HEIGHT,
            "kind": "white_repaint_mask",
        }
    return fixtures


def generation_general() -> list[dict[str, Any]]:
    subjects = [
        "a cobalt mechanical keyboard on a walnut desk",
        "a solar-powered research station on a frozen plateau",
        "a ceramic teapot shaped like a quiet moon",
        "a neighborhood bakery opening before sunrise",
        "an electric motorcycle parked beneath rain-soaked neon",
        "a botanist cataloging luminous alpine flowers",
        "a modular timber library beside a city canal",
        "a rescue robot navigating a collapsed tunnel",
        "a glass perfume bottle with an angular brass cap",
        "a family picnic beneath old olive trees",
        "a compact satellite unfolding above Earth",
        "a chef plating a geometric citrus dessert",
        "a red fox crossing a misty volcanic field",
        "an underwater archaeology team mapping a stone arch",
        "a handmade fountain pen beside folded correspondence",
        "a night market assembled from colorful fabric canopies",
        "a wind-powered cargo vessel entering a northern harbor",
        "a child-sized reading nook inside a curved bookshelf",
        "a scientific cutaway of a reusable water purifier",
        "a tiny observatory built into a desert cliff",
    ]
    treatments = [
        "wide establishing composition, natural perspective, soft morning light, realistic materials",
        "close product composition, controlled studio lighting, precise edges, restrained reflections",
        "editorial documentary composition, candid motion, believable ambient light, layered depth",
        "isometric explanatory composition, clean spatial hierarchy, neutral background, fine detail",
        "cinematic evening composition, motivated practical lights, atmospheric depth, balanced color",
    ]
    cases = []
    for subject_index, subject in enumerate(subjects):
        for treatment_index, treatment in enumerate(treatments):
            number = subject_index * len(treatments) + treatment_index + 1
            cases.append(
                {
                    "id": f"gen-general-{number:03d}",
                    "prompt": f"Create {subject}; {treatment}.",
                    "seed": 10_000 + number,
                    "size": "1024x1024",
                    "tags": ["prompt_adherence", "composition"],
                }
            )
    return cases


def typography_cases() -> list[dict[str, Any]]:
    phrases = [
        ("en", "XENO STUDIO"),
        ("de", "ZUKUNFT GESTALTEN"),
        ("fr", "CRÉER DEMAIN"),
        ("es", "IMAGINA SIN LÍMITES"),
        ("zh", "星河工作室"),
        ("ja", "未来を描く"),
        ("ko", "새로운 세계"),
        ("ar", "اصنع المستقبل"),
        ("hi", "नई दुनिया"),
        ("el", "ΝΕΑ ΕΠΟΧΗ"),
    ]
    layouts = [
        "a centered museum poster with one exact headline",
        "a storefront sign photographed straight-on with one exact line",
        "a minimal book cover with the exact title and no other words",
        "a transit-card advertisement with the exact phrase in one horizontal line",
    ]
    cases = []
    for phrase_index, (language, phrase) in enumerate(phrases):
        for layout_index, layout in enumerate(layouts):
            number = phrase_index * len(layouts) + layout_index + 1
            cases.append(
                {
                    "id": f"gen-type-{number:03d}",
                    "prompt": (
                        f'Create {layout}. Render exactly "{phrase}" with correct spelling, '
                        "clear glyphs, high contrast, and no extra text."
                    ),
                    "expected_text": phrase,
                    "language": language,
                    "seed": 20_000 + number,
                    "size": "1024x1024",
                    "tags": ["typography", "ocr"],
                }
            )
    return cases


def detail_cases() -> list[dict[str, Any]]:
    subjects = [
        "an elderly clockmaker examining a brass escapement",
        "a violinist adjusting a fine-string bridge backstage",
        "a ceramic artist shaping a delicate handle by hand",
        "a field medic fastening a compact instrument pouch",
        "a jeweler setting a small emerald into a silver ring",
        "a marine biologist holding a translucent specimen jar",
        "a tailor pinning a patterned sleeve on a mannequin",
        "a barista drawing a detailed fern in a porcelain cup",
        "a mechanic routing colored wires through a small control panel",
        "a gardener tying a young bonsai branch with copper wire",
    ]
    views = [
        "natural portrait framing with anatomically coherent face and hands",
        "tight hand-focused framing with five clear fingers per visible hand and realistic joints",
        "macro detail framing with crisp material texture, legible small geometry, and controlled depth of field",
    ]
    cases = []
    for subject_index, subject in enumerate(subjects):
        for view_index, view in enumerate(views):
            number = subject_index * len(views) + view_index + 1
            cases.append(
                {
                    "id": f"gen-detail-{number:03d}",
                    "prompt": f"Photograph {subject}; {view}, neutral documentary color.",
                    "seed": 30_000 + number,
                    "size": "1024x1024",
                    "tags": ["faces", "hands", "fine_detail"],
                }
            )
    return cases


def style_cases() -> list[dict[str, Any]]:
    palettes = [
        "cobalt blue, warm ivory, and brushed brass",
        "forest green, clay red, and pale sand",
        "deep violet, electric cyan, and charcoal",
        "coral, sea-glass teal, and clean white",
        "burgundy, parchment, and muted gold",
        "slate gray, safety orange, and ice blue",
        "indigo, lavender, and soft peach",
        "black, vermilion, and rice-paper cream",
        "moss, ochre, and weathered cedar",
        "ultramarine, lemon yellow, and neutral concrete",
    ]
    forms = [
        "a screen-printed travel poster of a mountain railway, using flat geometric shapes",
        "a tactile editorial still life of folded paper, glass, and woven cloth",
        "an abstract architectural study with layered arches, shadows, and negative space",
    ]
    cases = []
    for palette_index, palette in enumerate(palettes):
        for form_index, form in enumerate(forms):
            number = palette_index * len(forms) + form_index + 1
            cases.append(
                {
                    "id": f"gen-style-{number:03d}",
                    "prompt": f"Create {form}. Restrict the dominant palette to {palette}.",
                    "seed": 40_000 + number,
                    "size": "1024x1024",
                    "tags": ["style", "color_control"],
                }
            )
    return cases


def single_edit_cases() -> list[dict[str, Any]]:
    instructions = [
        "Replace only the background with a softly lit modern library; preserve the person's face, hair, glasses or hat, pose, and shirt silhouette.",
        "Change the shirt to a tailored charcoal jacket with a small cobalt pin; preserve identity, facial geometry, hair, accessory, pose, and background.",
        "Relight the portrait as a warm sunset window portrait; preserve identity, face shape, eye spacing, hair, accessory, clothing geometry, and composition.",
    ]
    cases = []
    for source_index in range(10):
        for instruction_index, instruction in enumerate(instructions):
            number = source_index * len(instructions) + instruction_index + 1
            cases.append(
                {
                    "id": f"edit-single-{number:03d}",
                    "source_fixture": f"avatar-{source_index + 1:02d}",
                    "prompt": instruction,
                    "seed": 50_000 + number,
                    "tags": ["single_image_edit", "identity_preservation"],
                }
            )
    return cases


def multi_edit_cases() -> list[dict[str, Any]]:
    cases = []
    for pair_index in range(10):
        first = pair_index + 1
        second = ((pair_index + 3) % 10) + 1
        prompts = [
            "Place both people side by side in a bright design studio, retaining the first image on the left and the second on the right; preserve both distinct identities and accessories.",
            "Create a two-panel editorial portrait with the first person in the upper panel and the second in the lower panel; preserve ordering, identity, shirt colors, and facial geometry.",
        ]
        for variant, prompt in enumerate(prompts):
            number = pair_index * len(prompts) + variant + 1
            cases.append(
                {
                    "id": f"edit-multi-{number:03d}",
                    "source_fixtures": [f"avatar-{first:02d}", f"avatar-{second:02d}"],
                    "prompt": prompt,
                    "seed": 60_000 + number,
                    "tags": ["multi_image_edit", "ordering", "identity_preservation"],
                }
            )
    return cases


def conditional_inpaint_cases() -> list[dict[str, Any]]:
    cases = []
    for source_index in range(10):
        prompts = [
            "Within the white repaint region only, add a small cobalt paper airplane; preserve every black-mask pixel.",
            "Within the white repaint region only, add subtle warm window light and no new text; preserve every black-mask pixel.",
        ]
        for variant, prompt in enumerate(prompts):
            number = source_index * len(prompts) + variant + 1
            cases.append(
                {
                    "id": f"inpaint-{number:03d}",
                    "source_fixture": f"avatar-{source_index + 1:02d}",
                    "mask_fixture": f"mask-{source_index + 1:02d}",
                    "prompt": prompt,
                    "seed": 70_000 + number,
                    "tags": ["conditional_inpaint", "mask_leakage"],
                }
            )
    return cases


def identity_pairs() -> list[dict[str, Any]]:
    transformations = [
        "soft library background",
        "charcoal jacket with cobalt pin",
        "warm sunset relighting",
        "clean monochrome editorial treatment",
        "subtle depth-of-field increase without geometric changes",
    ]
    pairs = []
    for source_index in range(10):
        for transform_index, transformation in enumerate(transformations):
            number = source_index * len(transformations) + transform_index + 1
            pairs.append(
                {
                    "id": f"identity-{number:03d}",
                    "source_fixture": f"avatar-{source_index + 1:02d}",
                    "prompt": (
                        f"Apply {transformation}; preserve facial geometry, eye spacing, hair shape, "
                        "accessory, pose, and silhouette."
                    ),
                    "seed": 80_000 + number,
                }
            )
    return pairs


def build_suite(fixtures: dict[str, dict[str, Any]]) -> dict[str, Any]:
    categories = {
        "generation_general": generation_general(),
        "generation_typography": typography_cases(),
        "generation_faces_hands_detail": detail_cases(),
        "generation_style_color": style_cases(),
        "edit_single_image": single_edit_cases(),
        "edit_multi_image": multi_edit_cases(),
        "conditional_inpaint": conditional_inpaint_cases(),
    }
    expected_counts = {
        "generation_general": 100,
        "generation_typography": 40,
        "generation_faces_hands_detail": 30,
        "generation_style_color": 30,
        "edit_single_image": 30,
        "edit_multi_image": 20,
        "conditional_inpaint": 20,
    }
    observed_counts = {name: len(cases) for name, cases in categories.items()}
    if observed_counts != expected_counts:
        raise AssertionError(f"quality category count drift: {observed_counts}")
    identities = identity_pairs()
    if len(identities) < 50:
        raise AssertionError("identity-preservation suite must contain at least 50 pairs")

    return {
        "schema_version": 1,
        "suite_version": "qwen-image-release-v1",
        "status": "frozen",
        "frozen_at": "2026-07-21",
        "provenance": {
            "license": "Apache-2.0",
            "description": (
                "Prompts and raster fixtures were authored or procedurally generated in this repository; "
                "no third-party text or image asset is embedded."
            ),
            "generator": "reference/image/qwen/build_quality_suite.py",
        },
        "execution": {
            "paired_bf16_reference": True,
            "identical_xeno_initial_latent": True,
            "default_size": "1024x1024",
            "default_steps": 50,
            "true_cfg_scale": 4.0,
            "conditional_inpaint_status": "inactive_until_image.inpaint_is_admitted",
        },
        "evaluators": {
            "prompt_alignment": {
                "model": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                "revision": "1c2b8495b28150b8a4922ee1c8edee224c284c0c",
                "implementation": "open-clip-torch==3.3.0",
                "preprocessing": {
                    "schema": "open-clip-eval-v1",
                    "image": "model-native resize/crop/normalize from pinned open-clip",
                    "text": "model-native tokenizer from the immutable model revision",
                    "aggregation": "mean cosine similarity over each frozen category",
                },
                "direction": "higher_is_better",
                "bf16_absolute_floor": 0.24,
            },
            "ocr": {
                "model": "PaddlePaddle/PaddleOCR-VL-1.6",
                "revision": "66317acc4c9fc17bd154591ce650735cd2855f3e",
                "implementation": "paddleocr==3.7.0",
                "preprocessing": {
                    "schema": "xeno-ocr-eval-v1",
                    "image": "RGB image at native output size; no enhancement or thresholding",
                    "text": "Unicode NFC, case-sensitive, preserve letters/digits, collapse whitespace",
                    "matching": "minimum normalized edit distance across OCR reading-order candidates",
                },
                "metrics": ["character_error_rate", "word_error_rate"],
                "direction": "lower_is_better",
                "bf16_character_error_rate_ceiling": 0.25,
            },
            "structural_identity": {
                "model": "facebook/dinov2-large",
                "revision": "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
                "implementation": "transformers==5.14.1",
                "preprocessing": {
                    "schema": "xeno-dinov2-identity-v1",
                    "image": "model-native RGB resize/crop/normalize from pinned Transformers",
                    "embedding": "L2-normalized CLS token",
                    "aggregation": "mean paired cosine similarity",
                },
                "direction": "higher_is_better",
                "bf16_absolute_floor": 0.55,
            },
            "face_identity": {
                "model": "facebook/dinov2-large",
                "revision": "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
                "implementation": "transformers==5.14.1",
                "scope": (
                    "Procedural identity fixtures only; this is a face-region identity proxy, "
                    "not a biometric identity system."
                ),
                "preprocessing": {
                    "schema": "xeno-procedural-face-identity-v1",
                    "crop": "fixed normalized box x=0.25..0.75,y=0.16..0.68 from source and candidate",
                    "image": "model-native RGB resize/crop/normalize from pinned Transformers",
                    "embedding": "L2-normalized CLS token",
                    "aggregation": "mean paired cosine similarity",
                },
                "direction": "higher_is_better",
                "bf16_absolute_floor": 0.60,
            },
            "mask_leakage": {
                "implementation": "xeno-protected-pixel-delta-v1",
                "preprocessing": {
                    "schema": "xeno-protected-pixel-delta-v1",
                    "image": "decode source and candidate to sRGB float32 in [0,1] at identical dimensions",
                    "mask": "nearest-neighbor resize, white pixels are editable, black pixels are protected",
                    "changed_pixel": "maximum absolute RGB channel delta greater than 0.02",
                },
                "direction": "lower_is_better",
                "protected_pixel_delta": 0.02,
                "upper_bound_ceiling": 0.02,
            },
        },
        "absolute_quality_floors": {
            "generation_general": {"prompt_alignment_min": 0.24},
            "generation_typography": {
                "prompt_alignment_min": 0.20,
                "character_error_rate_max": 0.25,
                "word_error_rate_max": 0.40,
            },
            "generation_faces_hands_detail": {"prompt_alignment_min": 0.22},
            "generation_style_color": {"prompt_alignment_min": 0.23},
            "edit_single_image": {
                "structural_identity_min": 0.55,
                "face_identity_min": 0.60,
            },
            "edit_multi_image": {
                "structural_identity_min": 0.50,
                "face_identity_min": 0.55,
            },
            "conditional_inpaint": {
                "protected_pixel_leakage_upper_bound_max": 0.02,
            },
        },
        "statistics": {
            "paired_resamples": 10_000,
            "bootstrap_rng": "PCG64",
            "bootstrap_seed": 1480937837,
            "confidence": 0.95,
            "degradation_bound": "one_sided_bootstrap_upper",
            "human_defect_bound": "one_sided_wilson_upper",
            "human_pairs_per_tier": 200,
            "human_raters_per_pair": 3,
            "human_blinded": True,
            "human_majority_vote": True,
            "identity_failure_upper_bound": 0.10,
        },
        "relative_admission_thresholds": {
            "Q8_0": {
                "prompt_alignment_decline_relative": 0.01,
                "ocr_cer_increase_points": 0.02,
                "structural_identity_decline_absolute": 0.01,
                "human_severe_defect_rate": 0.02,
            },
            "Q6_K": {
                "prompt_alignment_decline_relative": 0.02,
                "ocr_cer_increase_points": 0.03,
                "structural_identity_decline_absolute": 0.015,
                "human_severe_defect_rate": 0.03,
            },
            "Q5_K_M": {
                "prompt_alignment_decline_relative": 0.03,
                "ocr_cer_increase_points": 0.04,
                "structural_identity_decline_absolute": 0.02,
                "human_severe_defect_rate": 0.05,
            },
            "Q4_K_M": {
                "prompt_alignment_decline_relative": 0.05,
                "ocr_cer_increase_points": 0.06,
                "structural_identity_decline_absolute": 0.03,
                "human_severe_defect_rate": 0.08,
            },
        },
        "human_severe_defect_rubric": [
            "blank or substantially uniform output",
            "repeatable NaN or non-finite pipeline state",
            "grossly corrupted anatomy or object topology central to the prompt",
            "required text is unreadable or materially misspelled",
            "source identity is replaced rather than edited",
            "multi-image subjects are omitted, merged, or attributed in the wrong order",
            "protected inpaint content changes beyond the pinned perceptual delta",
        ],
        "fixtures": dict(sorted(fixtures.items())),
        "category_counts": observed_counts,
        "categories": categories,
        "identity_preservation_pairs": identities,
    }


def main() -> None:
    fixtures = build_fixtures()
    suite = build_suite(fixtures)
    encoded = (json.dumps(suite, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    atomic_write(SUITE_PATH, encoded)
    print(
        json.dumps(
            {
                "status": "ok",
                "suite": str(SUITE_PATH.relative_to(REPO_ROOT)),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "cases": sum(suite["category_counts"].values()),
                "identity_pairs": len(suite["identity_preservation_pairs"]),
                "fixtures": len(fixtures),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
