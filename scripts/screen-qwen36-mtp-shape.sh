#!/usr/bin/env bash
set -euo pipefail

model_path="${1:-/workspace/model/Qwen3.6-27B-Q4_K_S.gguf}"
output_dir="${2:-/workspace/profiles/xrt-mtp-shape-screen}"
repetitions="${3:-6}"
benchmark_script="${XRT_BENCHMARK_SCRIPT:-scripts/benchmark-qwen36-mtp.sh}"
depth_vocab_rows="${XRT_QWEN_MTP_SCREEN_DEPTH_VOCAB_ROWS:-65536}"
vocab_depth="${XRT_QWEN_MTP_SCREEN_VOCAB_DEPTH:-8}"
depths="${XRT_QWEN_MTP_SCREEN_DEPTHS:-4 6 8 10 12}"
vocab_rows="${XRT_QWEN_MTP_SCREEN_VOCAB_ROWS:-32768 49152 65536 81920 98304}"

mkdir -p "$output_dir"

for depth in $depths; do
  output="$output_dir/depth-${depth}-vocab-${depth_vocab_rows}.json"
  XRT_QWEN_MTP_MAX_DRAFT_TOKENS="$depth" \
    XRT_QWEN_MTP_VOCAB_ROWS="$depth_vocab_rows" \
    XRT_QWEN_MTP_VERIFY_GRAPH=1 \
    "$benchmark_script" "$model_path" "$repetitions" "$output"
done

for rows in $vocab_rows; do
  output="$output_dir/depth-${vocab_depth}-vocab-${rows}.json"
  if [[ -e "$output" ]]; then
    continue
  fi
  XRT_QWEN_MTP_MAX_DRAFT_TOKENS="$vocab_depth" \
    XRT_QWEN_MTP_VOCAB_ROWS="$rows" \
    XRT_QWEN_MTP_VERIFY_GRAPH=1 \
    "$benchmark_script" "$model_path" "$repetitions" "$output"
done

