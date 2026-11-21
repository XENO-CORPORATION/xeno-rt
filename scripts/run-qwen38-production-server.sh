#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
  printf 'usage: run-qwen38-production-server.sh TARGET_Q4_K_M.gguf MTP_Q8_0.gguf [PORT]\n' >&2
  exit 2
fi

target_path="$1"
draft_path="$2"
port="${3:-${XRT_QWEN38_PORT:-3000}}"
xrt_server_bin="${XRT_SERVER_BIN:-target/release/xrt-server}"
host="${XRT_QWEN38_HOST:-127.0.0.1}"

for artifact in "$target_path" "$draft_path" "$xrt_server_bin"; do
  if [[ ! -f "$artifact" ]]; then
    printf 'required artifact not found: %s\n' "$artifact" >&2
    exit 2
  fi
done

# This is the admitted 24 GB RTX 4090 profile. Environment values supplied by
# the operator remain authoritative for scheduler limits and bind address; the
# acceleration and memory values are pinned so an invocation cannot silently
# drift away from the benchmarked tuple. The hybrid Qwen3.8 reservation
# estimator needs just over 2.0 GiB for the admitted 8K request. A 0.75 KV
# fraction leaves roughly one quarter of the post-weight RTX 4090 headroom
# outside the KV arena while admitting that measured case.
exec env \
  XRT_BACKEND=cuda \
  XRT_PREFIX_CACHE=1 \
  XRT_NGRAM_SPECULATION=0 \
  XRT_QWEN_MTP=1 \
  XRT_QWEN_MTP_DRAFT_MODEL="$draft_path" \
  XRT_QWEN_MTP_MAX_DRAFT_TOKENS=4 \
  XRT_QWEN_MTP_VOCAB_ROWS=65536 \
  XRT_QWEN_MTP_PREFER_NGRAM=0 \
  XRT_QWEN_MTP_ADAPTIVE_FALLBACK=0 \
  XRT_QWEN_MTP_BATCHED_REBASE=1 \
  XRT_QWEN_MTP_VERIFY_GRAPH=0 \
  XRT_QWEN_MTP_DRAFT_GRAPH=1 \
  XRT_QWEN_BATCHED_PREFILL=1 \
  XRT_QWEN_BATCHED_PREFILL_MAX_ROWS=5 \
  XRT_CUDA_Q4_K_MARLIN=1 \
  XRT_CUDA_Q5_K_MARLIN=1 \
  XRT_CUDA_Q5_K_MARLIN_SHAPES=all \
  XRT_CUDA_Q5_K_PACKED_MARLIN=1 \
  XRT_CUDA_Q6_K_MARLIN=1 \
  XRT_CUDA_KQUANT_TENSOR_CORE_VERIFY=1 \
  XRT_CUDA_PARALLEL_VERIFY_PROJECTIONS=1 \
  XRT_CUDA_MARLIN_FUSED_EPILOGUES=1 \
  XRT_CUDA_DENSE_SMALL_MATMUL_COALESCED_COLUMNS=16 \
  XRT_GPU_MEMORY_FRACTION=0.97 \
  XRT_GPU_RESERVED_MB=512 \
  XRT_GPU_KV_FRACTION=0.75 \
  XRT_MAX_ACTIVE_SEQUENCES="${XRT_MAX_ACTIVE_SEQUENCES:-1}" \
  XRT_MAX_QUEUED_SEQUENCES="${XRT_MAX_QUEUED_SEQUENCES:-2}" \
  "$xrt_server_bin" \
    --model "$target_path" \
    --backend cuda-resident \
    --host "$host" \
    --port "$port"
