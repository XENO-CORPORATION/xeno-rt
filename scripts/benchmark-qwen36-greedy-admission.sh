#!/usr/bin/env bash
set -euo pipefail

model_path="${1:-/workspace/model/Qwen3.6-27B-Q4_K_S.gguf}"
output_dir="${2:-/workspace/profiles/qwen36-greedy-admission}"
repetitions="${3:-1}"
xrt_cli_bin="${XRT_CLI_BIN:-target/release/xrt-cli}"
suite_path="${XRT_PROMPT_SUITE:-benchmark-corpora/text/qwen36-greedy-admission-v1.json}"

mkdir -p "$output_dir"

common_env=(
  XRT_BACKEND=cuda
  XRT_PREFIX_CACHE=0
  XRT_NGRAM_SPECULATION=0
  XRT_QWEN_MTP_MAX_DRAFT_TOKENS=8
  XRT_QWEN_MTP_ADAPTIVE_FALLBACK=0
  XRT_QWEN_MTP_VOCAB_ROWS=65536
  XRT_QWEN_MTP_BATCHED_REBASE=1
  XRT_CUDA_Q4_K_MARLIN=1
  XRT_CUDA_KQUANT_TENSOR_CORE_VERIFY=1
  XRT_CUDA_PARALLEL_VERIFY_PROJECTIONS=1
  XRT_QWEN_MTP_VERIFY_GRAPH=1
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

env "${common_env[@]}" XRT_QWEN_MTP=0 XRT_QWEN_MTP_Q6_TENSOR_CORE_HEAD=0 \
  "$xrt_cli_bin" "${arguments[@]}" > "$output_dir/target-only.json"

env "${common_env[@]}" XRT_QWEN_MTP=1 XRT_QWEN_MTP_Q6_TENSOR_CORE_HEAD=0 \
  "$xrt_cli_bin" "${arguments[@]}" > "$output_dir/mtp-scalar.json"

env "${common_env[@]}" XRT_QWEN_MTP=1 XRT_QWEN_MTP_Q6_TENSOR_CORE_HEAD=1 \
  "$xrt_cli_bin" "${arguments[@]}" > "$output_dir/mtp-tensor-core.json"

env "${common_env[@]}" \
  XRT_QWEN_MTP=1 \
  XRT_QWEN_MTP_MAX_DRAFT_TOKENS=10 \
  XRT_QWEN_MTP_Q6_TENSOR_CORE_HEAD=1 \
  XRT_CUDA_DENSE_SMALL_MATMUL_TILED=1 \
  "$xrt_cli_bin" "${arguments[@]}" > "$output_dir/mtp-tiled-depth10.json"

python3 scripts/compare-bench-token-parity.py \
  "$output_dir/target-only.json" "$output_dir/mtp-scalar.json" \
  --output "$output_dir/target-vs-mtp-scalar.json"
python3 scripts/compare-bench-token-parity.py \
  "$output_dir/target-only.json" "$output_dir/mtp-tensor-core.json" \
  --output "$output_dir/target-vs-mtp-tensor-core.json"
python3 scripts/compare-bench-token-parity.py \
  "$output_dir/target-only.json" "$output_dir/mtp-tiled-depth10.json" \
  --output "$output_dir/target-vs-mtp-tiled-depth10.json"
