#!/usr/bin/env bash
set -euo pipefail

model_path="${1:?usage: benchmark-qwen36-mtp-depth-sweep.sh MODEL [OUTPUT_DIR] [REPETITIONS] [DEPTH ...]}"
output_dir="${2:-/workspace/profiles/qwen36-mtp-depth-sweep}"
repetitions="${3:-1}"
if (( $# > 3 )); then
  depths=("${@:4}")
else
  depths=(2 4 6 8 10 12 15)
fi
xrt_cli_bin="${XRT_CLI_BIN:-target/release/xrt-cli}"
suite_path="${XRT_PROMPT_SUITE:-benchmark-corpora/text/qwen36-greedy-admission-v1.json}"
baseline_path="${XRT_TARGET_BASELINE:-}"

mkdir -p "$output_dir"

common_env=(
  XRT_BACKEND=cuda
  XRT_PREFIX_CACHE=0
  XRT_NGRAM_SPECULATION=0
  XRT_QWEN_MTP=1
  XRT_QWEN_MTP_ADAPTIVE_FALLBACK=0
  XRT_QWEN_MTP_VOCAB_ROWS="${XRT_QWEN_MTP_VOCAB_ROWS:-65536}"
  XRT_QWEN_MTP_BATCHED_REBASE=1
  XRT_QWEN_MTP_Q6_TENSOR_CORE_HEAD=1
  XRT_CUDA_DENSE_SMALL_MATMUL_TILED=1
  XRT_CUDA_Q4_K_MARLIN=1
  XRT_CUDA_KQUANT_TENSOR_CORE_VERIFY=1
  XRT_CUDA_PARALLEL_VERIFY_PROJECTIONS=1
  XRT_QWEN_MTP_VERIFY_GRAPH=1
  XRT_QWEN_MTP_DRAFT_GRAPH=1
)

arguments=(
  bench
  --model "$model_path"
  --prompt-suite "$suite_path"
  --cache-modes f32
  --backends cuda-resident
  --cache-policy default_chat
  --repetitions "$repetitions"
  --concurrency 1
  --temperature 0
  --top-k 1
  --top-p 1
  --repetition-penalty 1
  --presence-penalty 0
  --frequency-penalty 0
  --seed 424242
  --json
)

for depth in "${depths[@]}"; do
  report="$output_dir/depth-${depth}.json"
  env "${common_env[@]}" XRT_QWEN_MTP_MAX_DRAFT_TOKENS="$depth" \
    "$xrt_cli_bin" "${arguments[@]}" > "$report"
  if [[ -n "$baseline_path" ]]; then
    python3 scripts/compare-bench-token-parity.py \
      "$baseline_path" "$report" \
      --output "$output_dir/depth-${depth}-parity.json"
  fi
done
