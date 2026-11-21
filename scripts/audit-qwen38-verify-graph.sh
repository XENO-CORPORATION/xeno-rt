#!/usr/bin/env bash
# Reproduce the Qwen3.8 verify-graph phase shift under a failing audit.
#
# The production and benchmark scripts force XRT_QWEN_MTP_VERIFY_GRAPH=0 because
# the reusable verify graph diverges after it transitions out of warmup. This
# script deliberately turns it ON and runs the audit in strict mode, so the run
# stops at the FIRST diverging window and names it, instead of producing a
# quietly wrong completion.
#
# Expected outcome while the defect is present: a nonzero exit carrying
# start_position, rows, recurrent_generation and verify_graph_enabled.
set -euo pipefail

model_path="${1:?usage: audit-qwen38-verify-graph.sh TARGET.gguf COMPANION.gguf [OUT_DIR]}"
draft_path="${2:?missing NextN companion path}"
output_dir="${3:-qwen38-verify-graph-audit}"
xrt_cli_bin="${XRT_CLI_BIN:-target/release/xrt-cli}"
suite_path="${XRT_QWEN38_SUITE:-benchmark-corpora/text/qwen38-mtp-repetition-audit-v1.json}"
draft_depth="${XRT_QWEN38_MTP_DEPTH:-4}"
# The audit reroutes greedy verification onto the all-logits window, so it needs
# more headroom than the compact greedy path. Keep KV small: the repetition case
# is a short prompt and the defect appears within 64 generated tokens.
kv_fraction="${XRT_QWEN38_KV_FRACTION:-0.20}"

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
)

arguments=(
  bench
  --model "$model_path"
  --prompt-suite "$suite_path"
  --enable-thinking false
  --cache-modes f32
  --backends cuda-resident
  --cache-policy default_chat
  --repetitions 1
  --concurrency 1
  --temperature 0 --top-k 1 --top-p 1
  --repetition-penalty 1 --presence-penalty 0 --frequency-penalty 0
  --seed 424242
  --json
)

mtp_env=(
  XRT_QWEN_MTP=1
  XRT_QWEN_MTP_DRAFT_MODEL="$draft_path"
  XRT_QWEN_MTP_MAX_DRAFT_TOKENS="$draft_depth"
  XRT_QWEN_MTP_ADAPTIVE_FALLBACK=0
  XRT_QWEN_MTP_PREFER_NGRAM=0
  XRT_QWEN_MTP_VOCAB_ROWS=65536
  XRT_QWEN_MTP_BATCHED_REBASE=1
  XRT_QWEN_MTP_DRAFT_GRAPH=1
  XRT_GPU_KV_FRACTION="$kv_fraction"
)

{
  printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'suite=%s\n' "$suite_path"
  printf 'mtp_depth=%s\n' "$draft_depth"
  printf 'kv_fraction=%s\n' "$kv_fraction"
  printf 'host=%s\n' "$(uname -a)"
} > "$output_dir/metadata/environment.txt"
nvidia-smi -q > "$output_dir/metadata/nvidia-smi-q.txt" 2>/dev/null || true

status=0

# Control: verify graphs OFF, audit strict. This is the qualified configuration
# and must PASS. If it fails, the defect is not graph-specific and every
# conclusion below changes.
printf '== control: verify graph OFF, audit strict ==\n'
env "${common_env[@]}" "${mtp_env[@]}" \
  XRT_QWEN_MTP_VERIFY_GRAPH=0 \
  XRT_QWEN_MTP_VERIFY_AUDIT=strict \
  "$xrt_cli_bin" "${arguments[@]}" > "$output_dir/control-graph-off.json" \
  2> "$output_dir/control-graph-off.log" || status=$?
printf 'control exit: %s\n' "$status"

# Candidate: verify graphs ON, audit strict. Expected to FAIL at the first
# diverging window while the defect is present.
candidate_status=0
printf '== candidate: verify graph ON, audit strict ==\n'
env "${common_env[@]}" "${mtp_env[@]}" \
  XRT_QWEN_MTP_VERIFY_GRAPH=1 \
  XRT_QWEN_MTP_VERIFY_AUDIT=strict \
  "$xrt_cli_bin" "${arguments[@]}" > "$output_dir/candidate-graph-on.json" \
  2> "$output_dir/candidate-graph-on.log" || candidate_status=$?
printf 'candidate exit: %s\n' "$candidate_status"

printf '\n== first divergence reported ==\n'
grep -m1 -o 'diverged from serial target execution[^"]*' \
  "$output_dir/candidate-graph-on.log" || printf '(no divergence line)\n'

printf '\ncontrol_exit=%s candidate_exit=%s\n' "$status" "$candidate_status" \
  | tee "$output_dir/metadata/exit-codes.txt"
