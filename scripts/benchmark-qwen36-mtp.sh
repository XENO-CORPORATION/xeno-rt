#!/usr/bin/env bash
set -euo pipefail

model_path="${1:-/workspace/model/Qwen3.6-27B-Q4_K_S.gguf}"
repetitions="${2:-20}"
output_path="${3:-}"
xrt_cli_bin="${XRT_CLI_BIN:-target/release/xrt-cli}"
nsys_output="${XRT_NSYS_OUTPUT:-}"

export XRT_BACKEND="${XRT_BACKEND:-cuda}"
export XRT_PREFIX_CACHE="${XRT_PREFIX_CACHE:-0}"
export XRT_NGRAM_SPECULATION="${XRT_NGRAM_SPECULATION:-0}"
export XRT_QWEN_MTP="${XRT_QWEN_MTP:-1}"
export XRT_QWEN_MTP_MAX_DRAFT_TOKENS="${XRT_QWEN_MTP_MAX_DRAFT_TOKENS:-8}"
export XRT_QWEN_MTP_ADAPTIVE_FALLBACK="${XRT_QWEN_MTP_ADAPTIVE_FALLBACK:-0}"
export XRT_QWEN_MTP_VOCAB_ROWS="${XRT_QWEN_MTP_VOCAB_ROWS:-65536}"
export XRT_QWEN_MTP_BATCHED_REBASE="${XRT_QWEN_MTP_BATCHED_REBASE:-1}"
export XRT_CUDA_Q4_K_MARLIN="${XRT_CUDA_Q4_K_MARLIN:-1}"
export XRT_CUDA_KQUANT_TENSOR_CORE_VERIFY="${XRT_CUDA_KQUANT_TENSOR_CORE_VERIFY:-1}"
export XRT_CUDA_PARALLEL_VERIFY_PROJECTIONS="${XRT_CUDA_PARALLEL_VERIFY_PROJECTIONS:-1}"
export XRT_QWEN_MTP_VERIFY_GRAPH="${XRT_QWEN_MTP_VERIFY_GRAPH:-0}"

arguments=(
  bench
  --model "$model_path"
  --prompt "Write the numbers from 1 to 100 in order, separated by commas, and do not stop early."
  --cache-modes f32
  --backends cuda-resident
  --cache-policy default_chat
  --max-tokens 64
  --repetitions "$repetitions"
  --temperature 0
  --top-k 1
  --top-p 1
  --repetition-penalty 1
  --seed 424242
  --json
)

command=("$xrt_cli_bin")
if [[ -n "$nsys_output" ]]; then
  nsys_bin="${XRT_NSYS_BIN:-nsys}"
  if [[ -z "${XRT_QWEN_MTP_PROFILE_DRAFT_WINDOW:-}" && -z "${XRT_QWEN_MTP_PROFILE_VERIFY_WINDOW:-}" ]]; then
    export XRT_QWEN_MTP_PROFILE_VERIFY_WINDOW=4
  fi
  command=(
    "$nsys_bin"
    profile
    --trace=cuda
    --sample=none
    --cpuctxsw=none
    --capture-range=cudaProfilerApi
    --capture-range-end=stop
    --cuda-graph-trace=node
    --force-overwrite=true
    --output="$nsys_output"
    "$xrt_cli_bin"
  )
fi

if [[ -n "$output_path" ]]; then
  "${command[@]}" "${arguments[@]}" > "$output_path"
else
  exec "${command[@]}" "${arguments[@]}"
fi
