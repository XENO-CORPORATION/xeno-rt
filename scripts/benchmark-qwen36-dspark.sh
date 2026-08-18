#!/usr/bin/env bash
set -euo pipefail

model_path="${1:-/workspace/model/Qwen3.6-27B-Q4_K_S.gguf}"
draft_path="${2:-/workspace/dspark-model/Qwen3.6-27B-DSpark-Q8_0.gguf}"
suite_path="${3:-/workspace/profiles/qwen36-greedy-production-256-v1.json}"
output_dir="${4:-/workspace/profiles/qwen36-dspark-production-256}"
repetitions="${5:-1}"
warmup_repetitions="${XRT_BENCH_WARMUP_REPETITIONS:-1}"
confidence_sweep="${XRT_DSPARK_CONFIDENCE_SWEEP:-0.15,0.25,0.35,0.45,0.55,0.65}"
depth_sweep="${XRT_DSPARK_DEPTH_SWEEP:-1,2,3,4,5,6,7,8,9,10,11,12,13,14}"
verify_grid_sweep="${XRT_DSPARK_VERIFY_GRID_SWEEP:-64,96}"
serial_swiglu_screen="${XRT_DSPARK_SERIAL_SWIGLU_SCREEN:-1}"
total_repetitions="$((repetitions + warmup_repetitions))"
xrt_cli_bin="${XRT_CLI_BIN:-target/release/xrt-cli}"
target_json="${XRT_BENCH_TARGET_JSON:-}"

mkdir -p "$output_dir"
if compgen -G "$output_dir/*-summary.json" > /dev/null && \
   [[ "${XRT_BENCH_ALLOW_EXISTING:-0}" != "1" ]]; then
  printf 'refusing to mix DSpark measurements with existing summaries in %s\n' "$output_dir" >&2
  printf 'select a new output directory or set XRT_BENCH_ALLOW_EXISTING=1 explicitly\n' >&2
  exit 2
fi

common_env=(
  XRT_BACKEND=cuda
  XRT_PREFIX_CACHE=0
  XRT_NGRAM_SPECULATION=0
  XRT_QWEN_MTP_ADAPTIVE_FALLBACK=0
  XRT_QWEN_MTP_VOCAB_ROWS=65536
  XRT_QWEN_MTP_BATCHED_REBASE=1
  XRT_CUDA_Q4_K_MARLIN=1
  XRT_CUDA_KQUANT_TENSOR_CORE_VERIFY=1
  XRT_CUDA_PARALLEL_VERIFY_PROJECTIONS=1
  XRT_CUDA_MARLIN_FUSED_EPILOGUES=1
  XRT_QWEN_MTP_VERIFY_GRAPH=1
  XRT_GPU_MEMORY_FRACTION=0.94
  XRT_GPU_RESERVED_MB=1024
  XRT_GPU_KV_FRACTION=0.55
)

candidate_env=(
  XRT_QWEN_MTP=1
  XRT_QWEN_MTP_MAX_DRAFT_TOKENS=15
  XRT_QWEN_MTP_Q6_TENSOR_CORE_HEAD=0
  XRT_QWEN_DFLASH_DRAFT_MODEL="$draft_path"
  XRT_CUDA_DFLASH_Q8_0_MARLIN=1
  XRT_CUDA_DFLASH_PARALLEL_PROJECTIONS=0
)

arguments=(
  bench
  --model "$model_path"
  --prompt-suite "$suite_path"
  --cache-modes f32
  --backends cuda-resident
  --cache-policy default_chat
  --repetitions "$total_repetitions"
  --concurrency 1
  --temperature 0
  --top-k 1
  --top-p 1
  --repetition-penalty 1
  --presence-penalty 0
  --frequency-penalty 0
  --seed 424242
  --enable-thinking false
  --json
)

run_candidate() {
  local label="$1"
  shift
  env -u XRT_QWEN_DSPARK_CONFIDENCE_MIN \
    -u XRT_QWEN_DSPARK_DRAFT_PROFILE_US \
    -u XRT_QWEN_DSPARK_VERIFY_PROFILE_US \
    -u XRT_QWEN_DSPARK_CONFIDENCE_TEMPERATURES \
    -u XRT_CUDA_SERIAL_SWIGLU_PROJECTIONS \
    -u XRT_CUDA_MARLIN_PARALLEL_GRID_BLOCKS \
    -u XRT_CUDA_MARLIN_SECONDARY_GRID_BLOCKS \
    -u XRT_QWEN_MTP_PREFER_NGRAM \
    -u XRT_QWEN_MTP_NGRAM_ORDER \
    "${common_env[@]}" "${candidate_env[@]}" "$@" \
    "$xrt_cli_bin" "${arguments[@]}" > "$output_dir/candidate-${label}.json"
  python3 scripts/compare-bench-token-parity.py \
    "$output_dir/target-only.json" "$output_dir/candidate-${label}.json" \
    --output "$output_dir/${label}-parity.json"
  python3 scripts/summarize-qwen36-dflash.py \
    "$output_dir/target-only.json" "$output_dir/candidate-${label}.json" \
    --warmup-repetitions "$warmup_repetitions" \
    --output "$output_dir/${label}-summary.json"
}

if [[ -n "$target_json" ]]; then
  if [[ ! -f "$target_json" ]]; then
    printf 'missing reusable target benchmark: %s\n' "$target_json" >&2
    exit 2
  fi
  cp "$target_json" "$output_dir/target-only.json"
else
  env -u XRT_QWEN_DSPARK_CONFIDENCE_MIN \
    -u XRT_QWEN_DSPARK_DRAFT_PROFILE_US \
    -u XRT_QWEN_DSPARK_VERIFY_PROFILE_US \
    -u XRT_QWEN_DSPARK_CONFIDENCE_TEMPERATURES \
    -u XRT_CUDA_SERIAL_SWIGLU_PROJECTIONS \
    -u XRT_CUDA_MARLIN_PARALLEL_GRID_BLOCKS \
    -u XRT_CUDA_MARLIN_SECONDARY_GRID_BLOCKS \
    -u XRT_QWEN_MTP_PREFER_NGRAM \
    -u XRT_QWEN_MTP_NGRAM_ORDER \
    "${common_env[@]}" XRT_QWEN_MTP=0 XRT_QWEN_MTP_Q6_TENSOR_CORE_HEAD=0 \
    "$xrt_cli_bin" "${arguments[@]}" > "$output_dir/target-only.json"
fi

run_candidate fixed

IFS=',' read -r -a depths <<< "$depth_sweep"
for depth in "${depths[@]}"; do
  depth="${depth//[[:space:]]/}"
  [[ -n "$depth" ]] || continue
  run_candidate "depth-${depth}" XRT_QWEN_MTP_MAX_DRAFT_TOKENS="$depth"
done

python3 scripts/select-qwen36-dspark-profile.py "$output_dir" \
  --goal 177 \
  --output "$output_dir/depth-selection.json" > /dev/null
best_depth="$(python3 -c '
import json, sys
label = json.load(open(sys.argv[1], encoding="utf-8"))["best"]["label"]
print(15 if label == "fixed" else int(label.removeprefix("depth-")))
' "$output_dir/depth-selection.json")"

python3 scripts/build-qwen36-dspark-hardware-profile.py "$output_dir" \
  --warmup-repetitions "$warmup_repetitions" \
  --output "$output_dir/hardware-profile.json" > /dev/null
hardware_draft_us="$(python3 -c '
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["runtime_environment"]["XRT_QWEN_DSPARK_DRAFT_PROFILE_US"])
' "$output_dir/hardware-profile.json")"
hardware_verify_us="$(python3 -c '
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["runtime_environment"]["XRT_QWEN_DSPARK_VERIFY_PROFILE_US"])
' "$output_dir/hardware-profile.json")"
run_candidate hardware-profile \
  XRT_QWEN_MTP_MAX_DRAFT_TOKENS=15 \
  XRT_QWEN_MTP_VERIFY_GRAPH=0 \
  XRT_QWEN_DSPARK_DRAFT_PROFILE_US="$hardware_draft_us" \
  XRT_QWEN_DSPARK_VERIFY_PROFILE_US="$hardware_verify_us"

# Product sessions can take a zero-model-cost prompt-lookup continuation when
# the recent token history contains the same trigram. Measure that composition
# separately: it preserves complete target verification but is intentionally
# not mixed into the pure DSpark depth and hardware-cost curves above.
run_candidate prefer-ngram \
  XRT_QWEN_MTP_MAX_DRAFT_TOKENS=15 \
  XRT_QWEN_MTP_PREFER_NGRAM=1 \
  XRT_QWEN_MTP_NGRAM_ORDER=8

# The 4090 can lose verifier throughput when both wide FFN projections each
# occupy the full Marlin grid. Screen serialization and capped per-stream grids
# under the same exact-token gate instead of assuming concurrency is faster.
if [[ "$serial_swiglu_screen" == "1" ]]; then
  run_candidate serial-swiglu \
    XRT_QWEN_MTP_MAX_DRAFT_TOKENS=15 \
    XRT_CUDA_SERIAL_SWIGLU_PROJECTIONS=1
fi

IFS=',' read -r -a verify_grids <<< "$verify_grid_sweep"
for grid_blocks in "${verify_grids[@]}"; do
  grid_blocks="${grid_blocks//[[:space:]]/}"
  [[ -n "$grid_blocks" ]] || continue
  if ! [[ "$grid_blocks" =~ ^[1-9][0-9]*$ ]] || (( grid_blocks > 512 )); then
    printf 'invalid Marlin verify grid block count: %s\n' "$grid_blocks" >&2
    exit 2
  fi
  run_candidate "parallel-grid-${grid_blocks}" \
    XRT_QWEN_MTP_MAX_DRAFT_TOKENS=15 \
    XRT_CUDA_MARLIN_PARALLEL_GRID_BLOCKS="$grid_blocks"
done

IFS=',' read -r -a thresholds <<< "$confidence_sweep"
for threshold in "${thresholds[@]}"; do
  threshold="${threshold//[[:space:]]/}"
  [[ -n "$threshold" ]] || continue
  label="confidence-${threshold//./p}-depth-${best_depth}"
  run_candidate "$label" \
    XRT_QWEN_MTP_MAX_DRAFT_TOKENS="$best_depth" \
    XRT_QWEN_MTP_VERIFY_GRAPH=0 \
    XRT_QWEN_DSPARK_CONFIDENCE_MIN="$threshold"
done

sha256sum "$model_path" "$draft_path" "$suite_path" "$xrt_cli_bin" \
  > "$output_dir/artifact-sha256.txt"
{
  printf 'warmup_repetitions_per_case=%s\n' "$warmup_repetitions"
  printf 'measured_repetitions_per_case=%s\n' "$repetitions"
  printf 'total_repetitions_per_case=%s\n' "$total_repetitions"
  printf 'confidence_sweep=%s\n' "$confidence_sweep"
  printf 'depth_sweep=%s\n' "$depth_sweep"
  printf 'serial_swiglu_screen=%s\n' "$serial_swiglu_screen"
  printf 'verify_grid_sweep=%s\n' "$verify_grid_sweep"
  printf 'confidence_base_depth=%s\n' "$best_depth"
  printf 'hardware_draft_profile_us=%s\n' "$hardware_draft_us"
  printf 'hardware_verify_profile_us=%s\n' "$hardware_verify_us"
  printf 'gpu_memory_fraction=0.94\n'
  printf 'gpu_reserved_mb=1024\n'
  printf 'gpu_kv_fraction=0.55\n'
} > "$output_dir/measurement-policy.txt"

python3 scripts/select-qwen36-dspark-profile.py "$output_dir" \
  --goal 177 \
  --output "$output_dir/profile-selection.json"
