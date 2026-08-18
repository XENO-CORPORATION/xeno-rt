#!/usr/bin/env bash
set -euo pipefail

model_path="${1:-/workspace/model/Qwen3.6-27B-Q4_K_S.gguf}"
draft_path="${2:-/workspace/dflash-model/dflash-draft-3.6-q8_0-rope10m.gguf}"
output_dir="${3:-/workspace/profiles/qwen36-dflash}"
repetitions="${4:-1}"
enable_thinking="${5:-}"
warmup_repetitions="${XRT_BENCH_WARMUP_REPETITIONS:-1}"
total_repetitions="$((repetitions + warmup_repetitions))"
xrt_cli_bin="${XRT_CLI_BIN:-target/release/xrt-cli}"
suite_path="${XRT_PROMPT_SUITE:-benchmark-corpora/text/qwen36-greedy-admission-v1.json}"

mkdir -p "$output_dir"

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
  --json
)

if [[ -n "$enable_thinking" ]]; then
  arguments+=(--enable-thinking "$enable_thinking")
fi

env "${common_env[@]}" \
  XRT_QWEN_MTP=0 \
  XRT_QWEN_MTP_Q6_TENSOR_CORE_HEAD=0 \
  "$xrt_cli_bin" "${arguments[@]}" > "$output_dir/target-only.json"

env "${common_env[@]}" \
  XRT_QWEN_MTP=1 \
  XRT_QWEN_MTP_MAX_DRAFT_TOKENS=15 \
  XRT_QWEN_MTP_Q6_TENSOR_CORE_HEAD=0 \
  XRT_QWEN_DFLASH_DRAFT_MODEL="$draft_path" \
  XRT_CUDA_DFLASH_Q8_0_MARLIN=1 \
  "$xrt_cli_bin" "${arguments[@]}" > "$output_dir/depth-15-serial.json"

env "${common_env[@]}" \
  XRT_QWEN_MTP=1 \
  XRT_QWEN_MTP_MAX_DRAFT_TOKENS=15 \
  XRT_QWEN_MTP_Q6_TENSOR_CORE_HEAD=0 \
  XRT_QWEN_DFLASH_DRAFT_MODEL="$draft_path" \
  XRT_CUDA_DFLASH_Q8_0_MARLIN=1 \
  XRT_CUDA_DFLASH_PARALLEL_PROJECTIONS=1 \
  "$xrt_cli_bin" "${arguments[@]}" > "$output_dir/depth-15-parallel.json"

python3 scripts/compare-bench-token-parity.py \
  "$output_dir/target-only.json" "$output_dir/depth-15-serial.json" \
  --output "$output_dir/serial-parity.json"
python3 scripts/compare-bench-token-parity.py \
  "$output_dir/target-only.json" "$output_dir/depth-15-parallel.json" \
  --output "$output_dir/parallel-parity.json"
python3 scripts/summarize-qwen36-dflash.py \
  "$output_dir/target-only.json" "$output_dir/depth-15-serial.json" \
  --warmup-repetitions "$warmup_repetitions" \
  --output "$output_dir/serial-summary.json"
python3 scripts/summarize-qwen36-dflash.py \
  "$output_dir/target-only.json" "$output_dir/depth-15-parallel.json" \
  --warmup-repetitions "$warmup_repetitions" \
  --output "$output_dir/parallel-summary.json"

{
  printf 'warmup_repetitions_per_case=%s\n' "$warmup_repetitions"
  printf 'measured_repetitions_per_case=%s\n' "$repetitions"
  printf 'total_repetitions_per_case=%s\n' "$total_repetitions"
  printf 'reported_statistics_must_exclude_repetitions_le=%s\n' "$warmup_repetitions"
} > "$output_dir/measurement-policy.txt"
