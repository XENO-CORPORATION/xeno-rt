#!/usr/bin/env bash
set -euo pipefail

model_path="${1:-/workspace/model/Qwen3.6-27B-Q4_K_S.gguf}"
draft_path="${2:-/workspace/dflash-model/dflash-draft-3.6-q8_0-rope10m.gguf}"
suite_path="${3:-/workspace/profiles/qwen36-greedy-production-256-v1.json}"
output_dir="${4:-/workspace/profiles/qwen36-dflash-strategy-screen}"
repetitions="${5:-1}"
warmup_repetitions="${XRT_BENCH_WARMUP_REPETITIONS:-1}"
total_repetitions="$((repetitions + warmup_repetitions))"
xrt_cli_bin="${XRT_CLI_BIN:-target/release/xrt-cli}"
target_json="${XRT_BENCH_TARGET_JSON:-}"
read -r -a depths <<< "${XRT_DFLASH_SCREEN_DEPTHS:-2 4 8 15}"
read -r -a ngram_orders <<< "${XRT_DFLASH_SCREEN_NGRAM_ORDERS:-3 4 6 8}"
read -r -a consensus_orders <<< "${XRT_DFLASH_SCREEN_CONSENSUS_ORDERS:-}"
read -r -a secondary_grids <<< "${XRT_DFLASH_SCREEN_SECONDARY_GRIDS:-}"
consensus_depth="${XRT_DFLASH_SCREEN_CONSENSUS_DEPTH:-15}"
if ! [[ "$consensus_depth" =~ ^[1-9][0-9]*$ ]] || (( consensus_depth > 15 )); then
  printf 'invalid DFlash consensus depth: %s\n' "$consensus_depth" >&2
  exit 2
fi
if [[ "${XRT_DFLASH_SCREEN_ONLY_CONSENSUS:-0}" == "1" ]]; then
  depths=()
  ngram_orders=()
fi

for required in "$model_path" "$draft_path" "$suite_path" "$xrt_cli_bin"; do
  if [[ ! -f "$required" ]]; then
    printf 'missing DFlash strategy-screen input: %s\n' "$required" >&2
    exit 2
  fi
done

if command -v strings >/dev/null 2>&1 &&
   strings "$xrt_cli_bin" | grep -Fq 'CUDA backend requested but xrt-runtime was built without'; then
  printf 'DFlash strategy screen requires a CUDA-enabled CLI; rebuild with: cargo build --release -p xrt-cli --features cuda\n' >&2
  exit 2
fi

mkdir -p "$output_dir"
if compgen -G "$output_dir/*-summary.json" >/dev/null &&
   [[ "${XRT_BENCH_ALLOW_EXISTING:-0}" != "1" ]]; then
  printf 'refusing to mix DFlash strategy measurements in %s\n' "$output_dir" >&2
  exit 2
fi

common_env=(
  XRT_BACKEND=cuda
  XRT_PREFIX_CACHE=0
  XRT_NGRAM_SPECULATION=0
  XRT_QWEN_MTP_ADAPTIVE_FALLBACK=0
  XRT_QWEN_MTP_VOCAB_ROWS=65536
  XRT_QWEN_MTP_BATCHED_REBASE=1
  XRT_QWEN_MTP_Q6_TENSOR_CORE_HEAD=0
  XRT_QWEN_DFLASH_DRAFT_MODEL="$draft_path"
  XRT_CUDA_Q4_K_MARLIN=1
  XRT_CUDA_KQUANT_TENSOR_CORE_VERIFY=1
  XRT_CUDA_PARALLEL_VERIFY_PROJECTIONS=1
  XRT_CUDA_MARLIN_FUSED_EPILOGUES=1
  XRT_QWEN_MTP_VERIFY_GRAPH=1
  XRT_CUDA_DFLASH_Q8_0_MARLIN=1
  XRT_CUDA_DFLASH_PARALLEL_PROJECTIONS=0
  XRT_GPU_MEMORY_FRACTION=0.94
  XRT_GPU_RESERVED_MB=1024
  XRT_GPU_KV_FRACTION=0.55
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

clean_env=(
  -u XRT_QWEN_DSPARK_CONFIDENCE_MIN
  -u XRT_QWEN_DSPARK_DRAFT_PROFILE_US
  -u XRT_QWEN_DSPARK_VERIFY_PROFILE_US
  -u XRT_QWEN_DSPARK_CONFIDENCE_TEMPERATURES
  -u XRT_QWEN_MTP_PREFER_NGRAM
  -u XRT_QWEN_MTP_NGRAM_ORDER
  -u XRT_QWEN_MTP_NGRAM_CONSENSUS
  -u XRT_QWEN_MTP_NGRAM_MIN_HITS
  -u XRT_QWEN_MTP_NGRAM_MIN_PERCENT
  -u XRT_QWEN_MTP_NGRAM_LOOKBACK
  -u XRT_QWEN_MTP_REUSE_DFLASH_SUFFIX
  -u XRT_CUDA_MARLIN_SECONDARY_GRID_BLOCKS
)

run_candidate() {
  local label="$1"
  local depth="$2"
  shift 2
  env "${clean_env[@]}" "${common_env[@]}" \
    XRT_QWEN_MTP=1 \
    XRT_QWEN_MTP_MAX_DRAFT_TOKENS="$depth" \
    "$@" "$xrt_cli_bin" "${arguments[@]}" \
    > "$output_dir/candidate-${label}.json"
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
  env "${clean_env[@]}" "${common_env[@]}" XRT_QWEN_MTP=0 \
    "$xrt_cli_bin" "${arguments[@]}" > "$output_dir/target-only.json"
fi

for depth in "${depths[@]}"; do
  run_candidate "depth-${depth}" "$depth"
done

for order in "${ngram_orders[@]}"; do
  run_candidate "depth-15-ngram-${order}" 15 \
    XRT_QWEN_MTP_PREFER_NGRAM=1 XRT_QWEN_MTP_NGRAM_ORDER="$order"
done

for order in "${consensus_orders[@]}"; do
  run_candidate "depth-${consensus_depth}-consensus-${order}" "$consensus_depth" \
    XRT_QWEN_MTP_PREFER_NGRAM=1 \
    XRT_QWEN_MTP_NGRAM_ORDER="$order" \
    XRT_QWEN_MTP_NGRAM_CONSENSUS=1
done

for grid_blocks in "${secondary_grids[@]}"; do
  if ! [[ "$grid_blocks" =~ ^[1-9][0-9]*$ ]] || (( grid_blocks > 512 )); then
    printf 'invalid secondary Marlin grid block count: %s\n' "$grid_blocks" >&2
    exit 2
  fi
  run_candidate "depth-15-consensus-3-secondary-grid-${grid_blocks}" 15 \
    XRT_QWEN_MTP_PREFER_NGRAM=1 \
    XRT_QWEN_MTP_NGRAM_ORDER=3 \
    XRT_QWEN_MTP_NGRAM_CONSENSUS=1 \
    XRT_CUDA_MARLIN_SECONDARY_GRID_BLOCKS="$grid_blocks"
done

sha256sum "$model_path" "$draft_path" "$suite_path" "$xrt_cli_bin" \
  > "$output_dir/artifact-sha256.txt"
{
  printf 'warmup_repetitions_per_case=%s\n' "$warmup_repetitions"
  printf 'measured_repetitions_per_case=%s\n' "$repetitions"
  printf 'depths=%s\n' "${depths[*]}"
  printf 'ngram_orders=%s\n' "${ngram_orders[*]}"
  printf 'consensus_orders=%s\n' "${consensus_orders[*]}"
  printf 'consensus_depth=%s\n' "$consensus_depth"
  printf 'secondary_grids=%s\n' "${secondary_grids[*]}"
  printf 'ngram_policy=fixed-width history proposal, neural fallback\n'
} > "$output_dir/measurement-policy.txt"

python3 scripts/select-qwen36-dspark-profile.py "$output_dir" \
  --goal 177 \
  --output "$output_dir/strategy-selection.json"
