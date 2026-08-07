#!/usr/bin/env python3
"""Build and compile the frozen blinded XRT image quality review."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SUITE_PATH = REPO_ROOT / "tests" / "common" / "image-quality-suite.json"
SUITE_VERSION = "qwen-image-release-v1"
SUITE_SHA256 = "eab7ceca3f39705c3f4e8829376c23f554f85fec99de08160414839b79544c88"
DEFAULT_RANDOMIZATION_SEED = 24081703
RATER_IDS = ("rater-01", "rater-02", "rater-03")
RATER_ID = re.compile(r"^[A-Za-z0-9_-]{3,64}$")


class ReviewError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def canonical_digest(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(body).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReviewError(f"{path} must contain one JSON object")
    return value


def atomic_write(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def validate_plan(plan: dict[str, Any], plan_bytes: bytes) -> str:
    if plan.get("schema_version") != 1 or plan.get("object") != "xeno.image.quality_plan":
        raise ReviewError("input is not an XRT image quality plan")
    if plan.get("suite") != {"version": SUITE_VERSION, "sha256": SUITE_SHA256}:
        raise ReviewError("quality plan does not target the frozen release suite")
    execution = plan.get("execution")
    expected = {
        "paired_bf16_reference": True,
        "identical_xeno_initial_latent": True,
        "size": "1024x1024",
        "steps": 50,
        "true_cfg_scale": 4.0,
        "conditional_inpaint_admitted": False,
    }
    if not isinstance(execution, dict) or any(execution.get(key) != value for key, value in expected.items()):
        raise ReviewError("quality plan execution drifts from the frozen non-inpaint protocol")
    cases = plan.get("cases")
    identities = plan.get("identity_preservation_pairs")
    if not isinstance(cases, list) or len(cases) != 250:
        raise ReviewError("quality plan must contain 250 active cases")
    if not isinstance(identities, list) or len(identities) != 50:
        raise ReviewError("quality plan must contain all 50 identity pairs")
    return hashlib.sha256(plan_bytes).hexdigest()


def hash_rank(seed: int, namespace: str, value: str) -> bytes:
    return hashlib.sha256(f"{seed}|{namespace}|{value}".encode()).digest()


def stratified_selection(plan: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    cases = plan["cases"]
    by_category: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        by_category.setdefault(case["category"], []).append(case)
    target = int(plan["human_review"]["minimum_pairs"]) - len(plan["identity_preservation_pairs"])
    if target <= 0 or target > len(cases):
        raise ReviewError("invalid frozen human-review pair count")
    total = len(cases)
    exact = {category: target * len(items) / total for category, items in by_category.items()}
    quotas = {category: int(value) for category, value in exact.items()}
    remaining = target - sum(quotas.values())
    order = sorted(by_category, key=lambda category: (-(exact[category] - quotas[category]), category))
    for category in order[:remaining]:
        quotas[category] += 1
    selected: list[dict[str, Any]] = []
    for category in sorted(by_category):
        ranked = sorted(
            by_category[category],
            key=lambda case: hash_rank(seed, f"select:{category}", case["id"]),
        )
        selected.extend(ranked[: quotas[category]])
    selected.extend(plan["identity_preservation_pairs"])
    if len(selected) != int(plan["human_review"]["minimum_pairs"]):
        raise ReviewError("stratified review selection did not reach the frozen pair count")
    return selected


def build_mapping(plan: dict[str, Any], plan_sha256: str, seed: int) -> dict[str, Any]:
    selected = stratified_selection(plan, seed)
    candidate_a = {
        case["id"]
        for case in sorted(selected, key=lambda case: hash_rank(seed, "slot", case["id"]))[
            : len(selected) // 2
        ]
    }
    pairs = []
    for case in selected:
        token = "pair-" + hashlib.sha256(f"{seed}|token|{case['id']}".encode()).hexdigest()[:20]
        pairs.append(
            {
                "pair_token": token,
                "pair_id": case["id"],
                "category": case["category"],
                "candidate_slot": "A" if case["id"] in candidate_a else "B",
                "identity_pair": case["category"] == "identity_preservation",
                "request": case["request"],
                "artifacts": case["artifacts"],
            }
        )
    return {
        "schema_version": 1,
        "object": "xeno.image.blinded_review_mapping",
        "plan_sha256": plan_sha256,
        "tier": plan["tier"],
        "randomization_seed": seed,
        "rater_ids": list(RATER_IDS),
        "pairs": pairs,
    }


def verify_corpus(plan: dict[str, Any], plan_sha256: str, artifact_root: Path) -> None:
    expected_size = tuple(int(value) for value in plan["execution"]["size"].split("x"))
    for case in [*plan["cases"], *plan["identity_preservation_pairs"]]:
        for side in ("bf16", "candidate"):
            relative = case["artifacts"][side]
            expected = f"{side}/{case['category']}/{case['id']}.png"
            if relative != expected:
                raise ReviewError(f"artifact path drift for {case['id']} {side}")
            path = artifact_root / relative
            checkpoint = path.with_suffix(".png.xrt.json")
            if not path.is_file() or path.is_symlink() or not checkpoint.is_file() or checkpoint.is_symlink():
                raise ReviewError(f"missing regular corpus output/checkpoint: {relative}")
            record = read_json(checkpoint)
            if (
                record.get("plan_sha256") != plan_sha256
                or record.get("id") != case["id"]
                or record.get("side") != side
                or record.get("artifact_path") != relative
                or record.get("output_sha256") != sha256_file(path)
            ):
                raise ReviewError(f"corpus checkpoint drift for {relative}")
            with Image.open(path) as image:
                image.load()
                if image.format != "PNG" or image.size != expected_size or getattr(image, "n_frames", 1) != 1:
                    raise ReviewError(f"invalid corpus PNG: {relative}")


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def public_pair(mapping: dict[str, Any], artifact_root: Path, fixture_root: Path, public: Path) -> dict[str, Any]:
    token = mapping["pair_token"]
    slot = mapping["candidate_slot"]
    a_side, b_side = ("candidate", "bf16") if slot == "A" else ("bf16", "candidate")
    assets = public / "assets" / token
    link_or_copy(artifact_root / mapping["artifacts"][a_side], assets / "a.png")
    link_or_copy(artifact_root / mapping["artifacts"][b_side], assets / "b.png")
    request = mapping["request"]
    sources = request.get("source_fixtures") or ([request["source_fixture"]] if request.get("source_fixture") else [])
    for index, fixture in enumerate(sources, 1):
        source = fixture_root / f"{fixture}.png"
        if not source.is_file() or source.is_symlink():
            raise ReviewError(f"missing regular source fixture: {fixture}")
        link_or_copy(source, assets / f"source-{index}.png")
    return {
        "pair_token": token,
        "category": mapping["category"],
        "prompt": request["prompt"],
        "identity_pair": mapping["identity_pair"],
        "a_image": f"assets/{token}/a.png",
        "b_image": f"assets/{token}/b.png",
        "sources": [f"assets/{token}/source-{index}.png" for index in range(1, len(sources) + 1)],
    }


def review_html(rater_id: str, pairs: list[dict[str, Any]], rubric: list[str]) -> bytes:
    data = json.dumps({"rater_id": rater_id, "pairs": pairs, "rubric": rubric}, ensure_ascii=False).replace("</", "<\\/")
    title = html.escape(f"XRT Image Quality Review - {rater_id}")
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
body{{font:16px system-ui;margin:0;background:#111;color:#eee}}main{{max-width:1500px;margin:auto;padding:24px}}
.rubric,.pair{{background:#1c1c1c;border:1px solid #444;border-radius:10px;padding:18px;margin:18px 0}}
.images{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}img{{max-width:100%;height:auto;background:#222}}
.sources img{{max-width:220px;margin:8px}}label{{display:inline-block;margin:8px 18px 8px 0}}button{{font-size:18px;padding:12px 20px}}
@media(max-width:800px){{.images{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>{title}</h1><div class="rubric" id="rubric"></div><div id="pairs"></div>
<button id="export">Export completed JSON</button></main><script>
const DATA={data};
document.getElementById('rubric').innerHTML='<h2>Severe-defect rubric</h2><ul>'+DATA.rubric.map(x=>'<li>'+x+'</li>').join('')+'</ul><p>Judge A and B independently. Do not try to identify the model.</p>';
const root=document.getElementById('pairs');
for(const [i,p] of DATA.pairs.entries()){{const d=document.createElement('section');d.className='pair';d.dataset.token=p.pair_token;
const sources=p.sources.length?'<div class="sources"><b>Source reference(s)</b><br>'+p.sources.map(x=>`<img src="${{x}}">`).join('')+'</div>':'';
const identity=p.identity_pair?'<label><input type="checkbox" data-field="a_identity_failure"> A identity failure</label><label><input type="checkbox" data-field="b_identity_failure"> B identity failure</label>':'';
d.innerHTML=`<h2>${{i+1}} / ${{DATA.pairs.length}} - ${{p.category}}</h2><p>${{p.prompt}}</p>${{sources}}<div class="images"><div><h3>A</h3><img src="${{p.a_image}}"><label><input type="checkbox" data-field="a_severe_defect"> A severe defect</label></div><div><h3>B</h3><img src="${{p.b_image}}"><label><input type="checkbox" data-field="b_severe_defect"> B severe defect</label></div></div>${{identity}}`;root.appendChild(d)}}
document.getElementById('export').onclick=()=>{{const ratings=[...document.querySelectorAll('.pair')].map(d=>{{const get=f=>d.querySelector(`[data-field="${{f}}"]`);const identity=!!get('a_identity_failure');return{{pair_token:d.dataset.token,a_severe_defect:get('a_severe_defect').checked,b_severe_defect:get('b_severe_defect').checked,a_identity_failure:identity?get('a_identity_failure').checked:null,b_identity_failure:identity?get('b_identity_failure').checked:null}}}});const out={{schema_version:1,rater_id:DATA.rater_id,ratings}};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,2)+'\\n'],{{type:'application/json'}}));a.download=DATA.rater_id+'-responses.json';a.click()}};
</script></body></html>"""
    return body.encode()


def prepare(args: argparse.Namespace) -> None:
    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    plan_sha256 = validate_plan(plan, plan_bytes)
    artifact_root = args.artifact_root.resolve(strict=True)
    fixture_root = args.fixture_root.resolve(strict=True)
    verify_corpus(plan, plan_sha256, artifact_root)
    mapping = build_mapping(plan, plan_sha256, args.seed)
    suite = read_json(SUITE_PATH)
    rubric = suite["human_severe_defect_rubric"]
    if args.output.exists():
        raise ReviewError(f"review package already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=args.output.name + ".", suffix=".staging", dir=args.output.parent))
    try:
        public = staging / "public"
        private = staging / "private"
        public.mkdir()
        private.mkdir()
        pairs = [public_pair(pair, artifact_root, fixture_root, public) for pair in mapping["pairs"]]
        for rater_id in RATER_IDS:
            ordered = sorted(pairs, key=lambda pair: hash_rank(args.seed, rater_id, pair["pair_token"]))
            atomic_write(public / f"{rater_id}.html", review_html(rater_id, ordered, rubric))
        mapping["rubric_sha256"] = canonical_digest(rubric)
        atomic_write(private / "mapping.json", canonical_bytes(mapping))
        atomic_write(
            staging / "README.txt",
            b"Give each reviewer only public/rater-NN.html plus public/assets. Keep private/mapping.json hidden.\n",
        )
        os.replace(staging, args.output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"status": "ok", "pairs": len(mapping["pairs"]), "output": str(args.output)}))


def compile_responses(args: argparse.Namespace) -> None:
    mapping = read_json(args.package / "private" / "mapping.json")
    pairs = {pair["pair_token"]: pair for pair in mapping["pairs"]}
    expected_raters = set(mapping["rater_ids"])
    seen_raters: set[str] = set()
    reviews = []
    for path in args.responses:
        response = read_json(path)
        rater = response.get("rater_id")
        if rater not in expected_raters or not RATER_ID.fullmatch(rater) or rater in seen_raters:
            raise ReviewError(f"unexpected or duplicate pseudonymous rater: {rater!r}")
        seen_raters.add(rater)
        ratings = response.get("ratings")
        if not isinstance(ratings, list) or len(ratings) != len(pairs):
            raise ReviewError(f"{rater} must rate exactly {len(pairs)} pairs")
        by_token = {rating.get("pair_token"): rating for rating in ratings if isinstance(rating, dict)}
        if set(by_token) != set(pairs):
            raise ReviewError(f"{rater} response tokens are incomplete or duplicated")
        for token, pair in pairs.items():
            rating = by_token[token]
            expected_keys = {
                "pair_token",
                "a_severe_defect",
                "b_severe_defect",
                "a_identity_failure",
                "b_identity_failure",
            }
            if set(rating) != expected_keys:
                raise ReviewError(f"invalid rating fields for {rater} {token}")
            for field in ("a_severe_defect", "b_severe_defect"):
                if not isinstance(rating[field], bool):
                    raise ReviewError(f"{rater} {token} {field} must be boolean")
            identity = pair["identity_pair"]
            for field in ("a_identity_failure", "b_identity_failure"):
                if identity and not isinstance(rating[field], bool):
                    raise ReviewError(f"{rater} {token} {field} must be boolean")
                if not identity and rating[field] is not None:
                    raise ReviewError(f"{rater} {token} {field} must be null")
            slot = pair["candidate_slot"].lower()
            reviews.append(
                {
                    "pair_id": pair["pair_id"],
                    "rater_id": rater,
                    "candidate_slot": pair["candidate_slot"],
                    "severe_defect": rating[f"{slot}_severe_defect"],
                    "identity_failure": rating[f"{slot}_identity_failure"] if identity else None,
                }
            )
    if seen_raters != expected_raters:
        raise ReviewError("responses must contain each of the three assigned raters exactly once")
    output = {
        "human_review_protocol": {
            "blinded": True,
            "randomized": True,
            "rater_ids_pseudonymous": True,
            "rubric_sha256": mapping["rubric_sha256"],
        },
        "human_reviews": reviews,
    }
    atomic_write(args.output, canonical_bytes(output))
    print(json.dumps({"status": "ok", "ratings": len(reviews), "output": str(args.output)}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare")
    build.add_argument("--plan", type=Path, required=True)
    build.add_argument("--artifact-root", type=Path, required=True)
    build.add_argument("--fixture-root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--seed", type=int, default=DEFAULT_RANDOMIZATION_SEED)
    compile_parser = commands.add_parser("compile")
    compile_parser.add_argument("--package", type=Path, required=True)
    compile_parser.add_argument("--responses", type=Path, nargs=3, required=True)
    compile_parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        prepare(args) if args.command == "prepare" else compile_responses(args)
    except (OSError, ValueError, KeyError, TypeError, ReviewError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
