#!/usr/bin/env bash
set -euo pipefail

model_path="${1:-Qwen3.8-27B-Q4_K_M.gguf}"
draft_path="${2:-mtp-Qwen3.8-27B-Q8_0.gguf}"
output_dir="${3:-qwen38-mtp-benchmark}"
xrt_cli_bin="${XRT_CLI_BIN:-target/release/xrt-cli}"
suite_path="${XRT_QWEN38_SUITE:-benchmark-corpora/text/qwen38-greedy-admission-v1.json}"
repetitions="${XRT_QWEN38_REPETITIONS:-1}"
draft_depth="${XRT_QWEN38_MTP_DEPTH:-4}"
adaptive_fallback="${XRT_QWEN38_ADAPTIVE_FALLBACK:-0}"
prefer_ngram="${XRT_QWEN38_PREFER_NGRAM:-0}"

mkdir -p "$output_dir/metadata"

common_env=(
  XRT_BACKEND=cuda
  XRT_PREFIX_CACHE=0
  XRT_NGRAM_SPECULATION=0
  XRT_CUDA_Q4_K_MARLIN=1
  XRT_CUDA_Q5_K_MARLIN=1
  XRT_CUDA_Q5_K_MARLIN_SHAPES=all
  XRT_CUDA_Q5_K_PACKED_MARLIN=1
  XRT_CUDA_Q6_K_MARLIN=1
  XRT_CUDA_KQUANT_TENSOR_CORE_VERIFY=1
  XRT_CUDA_PARALLEL_VERIFY_PROJECTIONS=1
  XRT_CUDA_MARLIN_FUSED_EPILOGUES=1
  XRT_CUDA_DENSE_SMALL_MATMUL_COALESCED_COLUMNS=16
  XRT_GPU_MEMORY_FRACTION=0.97
  XRT_GPU_RESERVED_MB=512
  XRT_GPU_KV_FRACTION=0.45
)

arguments=(
  bench
  --model "$model_path"
  --prompt-suite "$suite_path"
  --enable-thinking false
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

sha256sum "$model_path" "$draft_path" "$suite_path" "$xrt_cli_bin" \
  > "$output_dir/metadata/sha256.txt"
nvidia-smi -q > "$output_dir/metadata/nvidia-smi-q.txt"
{
  printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'suite=%s\n' "$suite_path"
  printf 'repetitions=%s\n' "$repetitions"
  printf 'mtp_depth=%s\n' "$draft_depth"
  printf 'adaptive_fallback=%s\n' "$adaptive_fallback"
  printf 'prefer_ngram=%s\n' "$prefer_ngram"
  printf 'rustc=%s\n' "$(rustc --version 2>/dev/null || printf unavailable)"
  printf 'cargo=%s\n' "$(cargo --version 2>/dev/null || printf unavailable)"
  printf 'nvcc=%s\n' "$(nvcc --version 2>/dev/null | tail -1 || printf unavailable)"
  uname -a
} > "$output_dir/metadata/environment.txt"

env -u XRT_QWEN_MTP_DRAFT_MODEL -u XRT_QWEN_MTP_VOCAB_ROWS \
  "${common_env[@]}" XRT_QWEN_MTP=0 \
  "$xrt_cli_bin" "${arguments[@]}" > "$output_dir/target.json"

# The reusable verify graph is not admitted for Qwen3.8 yet: its captured
# recurrent-state path can diverge after the graph transitions from warmup.
# The ordinary batched verifier is exact and remains the qualified default.
env "${common_env[@]}" \
  XRT_QWEN_MTP=1 \
  XRT_QWEN_MTP_DRAFT_MODEL="$draft_path" \
  XRT_QWEN_MTP_MAX_DRAFT_TOKENS="$draft_depth" \
  XRT_QWEN_MTP_ADAPTIVE_FALLBACK="$adaptive_fallback" \
  XRT_QWEN_MTP_PREFER_NGRAM="$prefer_ngram" \
  XRT_QWEN_MTP_VOCAB_ROWS=65536 \
  XRT_QWEN_MTP_BATCHED_REBASE=1 \
  XRT_QWEN_MTP_VERIFY_GRAPH=0 \
  XRT_QWEN_MTP_DRAFT_GRAPH=1 \
  "$xrt_cli_bin" "${arguments[@]}" > "$output_dir/mtp.json"

python3 scripts/compare-bench-token-parity.py \
  "$output_dir/target.json" "$output_dir/mtp.json" \
  --output "$output_dir/target-vs-mtp-parity.json"

sha256sum "$output_dir/target.json" "$output_dir/mtp.json" \
  "$output_dir/target-vs-mtp-parity.json" \
  > "$output_dir/metadata/result-sha256.txt"
