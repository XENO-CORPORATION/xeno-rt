#!/usr/bin/env bash
set -euo pipefail

model_path="${1:-/root/models/qwen38/Qwen3.8-27B-Q4_K_M.gguf}"
draft_path="${2:-/root/models/qwen38/mtp-Qwen3.8-27B-Q8_0.gguf}"
output_dir="${3:-qwen38-production-benchmark}"
phases="${4:-quality,multiturn,context}"
xrt_cli_bin="${XRT_CLI_BIN:-target/release/xrt-cli}"
corpus_dir="${XRT_QWEN38_PRODUCTION_CORPUS_DIR:-benchmark-corpora/text/qwen38-production-v1}"
repetitions="${XRT_QWEN38_PRODUCTION_REPETITIONS:-1}"
draft_depth="${XRT_QWEN38_MTP_DEPTH:-4}"

mkdir -p "$output_dir/metadata"
overall_status=0

has_phase() {
  [[ ",$phases," == *",$1,"* ]]
}

common_env=(
  XRT_BACKEND=cuda
  XRT_PREFIX_CACHE=0
  XRT_NGRAM_SPECULATION=0
  XRT_QWEN_MTP_PREFER_NGRAM=0
  XRT_QWEN_MTP_ADAPTIVE_FALLBACK=0
  XRT_QWEN_BATCHED_PREFILL=1
  # Five rows bounds DeltaNet's recurrent rebase journal to four snapshots.
  # This leaves enough 24 GB headroom for the resident official MTP companion
  # and a later request with a larger KV reservation.
  XRT_QWEN_BATCHED_PREFILL_MAX_ROWS=5
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

candidate_env=(
  XRT_QWEN_MTP=1
  XRT_QWEN_MTP_DRAFT_MODEL="$draft_path"
  XRT_QWEN_MTP_MAX_DRAFT_TOKENS="$draft_depth"
  XRT_QWEN_MTP_VOCAB_ROWS=65536
  XRT_QWEN_MTP_BATCHED_REBASE=1
  XRT_QWEN_MTP_VERIFY_GRAPH=0
  XRT_QWEN_MTP_DRAFT_GRAPH=1
)

performance_arguments=(
  --model "$model_path"
  --enable-thinking false
  --cache-modes f32
  --backends cuda-resident
  --cache-policy default_chat
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

quality_arguments=(
  --model "$model_path"
  --enable-thinking true
  --cache-modes f32
  --backends cuda-resident
  --cache-policy default_chat
  --concurrency 1
  --temperature 1
  --top-k 20
  --top-p 0.95
  --repetition-penalty 1
  --presence-penalty 0
  --frequency-penalty 0
  --seed 424242
  --json
)

record_metadata() {
  sha256sum "$model_path" "$draft_path" "$xrt_cli_bin" \
    > "$output_dir/metadata/artifact-sha256.txt"
  sha256sum "$corpus_dir"/*.json > "$output_dir/metadata/corpus-sha256.txt"
  nvidia-smi -q > "$output_dir/metadata/nvidia-smi-q.txt"
  {
    printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'phases=%s\n' "$phases"
    printf 'repetitions=%s\n' "$repetitions"
    printf 'mtp_depth=%s\n' "$draft_depth"
    printf 'target_source=%s\n' 'https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF'
    printf 'target_source_revision=%s\n' '0669b98607d47046c7c2b3f801011d54a08cfccf'
    printf 'mtp_source=%s\n' 'https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF'
    printf 'mtp_source_revision=%s\n' '0669b98607d47046c7c2b3f801011d54a08cfccf'
    printf 'base_model_source=%s\n' 'https://huggingface.co/Qwen/Qwen3.8-27B'
    printf 'base_model_revision=%s\n' '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
    printf 'base_model_license=%s\n' 'Apache-2.0'
    printf 'license_provenance=%s\n' 'https://github.com/QwenLM/Qwen3#license-agreement'
    printf 'git_head=%s\n' "$(git rev-parse HEAD 2>/dev/null || printf unavailable)"
    printf 'git_status_entries=%s\n' "$(git status --porcelain 2>/dev/null | wc -l || true)"
    printf 'rustc=%s\n' "$(rustc --version 2>/dev/null || printf unavailable)"
    printf 'cargo=%s\n' "$(cargo --version 2>/dev/null || printf unavailable)"
    printf 'nvcc=%s\n' "$(nvcc --version 2>/dev/null | tail -1 || printf unavailable)"
    uname -a
  } > "$output_dir/metadata/environment.txt"
}

run_pair() {
  local name="$1"
  local suite="$2"
  local expected="$3"
  local runs="$4"
  local profile="${5:-performance}"
  local phase_dir="$output_dir/$name"
  local status=0
  local -a selected_arguments=("${performance_arguments[@]}")
  if [[ "$profile" == quality ]]; then
    selected_arguments=("${quality_arguments[@]}")
  fi
  mkdir -p "$phase_dir"

  env -u XRT_QWEN_MTP_DRAFT_MODEL -u XRT_QWEN_MTP_VOCAB_ROWS \
    "${common_env[@]}" XRT_QWEN_MTP=0 \
    "$xrt_cli_bin" bench --prompt-suite "$suite" --repetitions "$runs" \
    "${selected_arguments[@]}" > "$phase_dir/target.json"

  env "${common_env[@]}" "${candidate_env[@]}" \
    "$xrt_cli_bin" bench --prompt-suite "$suite" --repetitions "$runs" \
    "${selected_arguments[@]}" > "$phase_dir/candidate.json"

  python3 scripts/compare-bench-token-parity.py \
    "$phase_dir/target.json" "$phase_dir/candidate.json" \
    --output "$phase_dir/parity.json" || status=1
  python3 scripts/validate-qwen36-production.py \
    "$phase_dir/target.json" "$expected" \
    --output "$phase_dir/target-validation.json" || status=1
  python3 scripts/validate-qwen36-production.py \
    "$phase_dir/candidate.json" "$expected" \
    --output "$phase_dir/candidate-validation.json" || status=1
  sha256sum "$phase_dir"/*.json > "$phase_dir/result-sha256.txt"
  return "$status"
}

record_metadata

if has_phase quality; then
  run_pair \
    quality \
    "$corpus_dir/quality.json" \
    "$corpus_dir/quality.expected.json" \
    "$repetitions" \
    quality || overall_status=1
fi

if has_phase multiturn; then
  run_pair \
    multiturn \
    "$corpus_dir/shared-multiturn.json" \
    "$corpus_dir/shared-multiturn.expected.json" \
    "$repetitions" || overall_status=1
fi

if has_phase context; then
  run_pair \
    context-through-08192 \
    "$corpus_dir/context-through-08192.json" \
    "$corpus_dir/context-through-08192.expected.json" \
    1 || overall_status=1
fi

sha256sum "$output_dir/metadata"/* > "$output_dir/metadata/result-sha256.txt"
exit "$overall_status"
