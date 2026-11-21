#!/usr/bin/env bash
set -euo pipefail

model_path="${1:-/workspace/model/Qwen3.6-27B-Q4_K_S.gguf}"
artifact_dir="${2:-/workspace/dspark-model}"
suite_path="${3:-/workspace/profiles/qwen36-greedy-production-256-v1.json}"
output_root="${4:-/workspace/profiles/qwen36-dspark-artifact-screen}"
repetitions="${5:-1}"

artifacts=(
  "all-q8:Qwen3.6-27B-DSpark-Q8_0.gguf"
  "w2-f16:Qwen3.6-27B-DSpark-Q8_0-MarkovW2F16.gguf"
  "markov-f16:Qwen3.6-27B-DSpark-Q8_0-MarkovF16.gguf"
)

mkdir -p "$output_root"
target_json=""
for spec in "${artifacts[@]}"; do
  label="${spec%%:*}"
  artifact="${spec#*:}"
  artifact_path="$artifact_dir/$artifact"
  if [[ ! -f "$artifact_path" ]]; then
    printf 'missing DSpark artifact for %s: %s\n' "$label" "$artifact_path" >&2
    exit 2
  fi
  output_dir="$output_root/$label"
  if [[ -n "$target_json" ]]; then
    XRT_BENCH_TARGET_JSON="$target_json" \
      bash scripts/benchmark-qwen36-dspark.sh \
        "$model_path" "$artifact_path" "$suite_path" "$output_dir" "$repetitions"
  else
    bash scripts/benchmark-qwen36-dspark.sh \
      "$model_path" "$artifact_path" "$suite_path" "$output_dir" "$repetitions"
    target_json="$output_dir/target-only.json"
  fi
done

python3 - "$output_root" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for selection in sorted(root.glob("*/profile-selection.json")):
    report = json.loads(selection.read_text(encoding="utf-8"))
    if report.get("best"):
        rows.append({"artifact": selection.parent.name, **report["best"]})
rows.sort(key=lambda row: float(row["mean_decode_tok_s"]), reverse=True)
report = {
    "object": "xrt.qwen36_dspark_artifact_selection",
    "goal_mean_decode_tok_s": 177.0,
    "goal_met": bool(rows and float(rows[0]["mean_decode_tok_s"]) >= 177.0),
    "best": rows[0] if rows else None,
    "ranking": rows,
}
payload = json.dumps(report, indent=2) + "\n"
(root / "artifact-selection.json").write_text(payload, encoding="utf-8", newline="\n")
print(payload, end="")
raise SystemExit(0 if rows else 1)
PY

