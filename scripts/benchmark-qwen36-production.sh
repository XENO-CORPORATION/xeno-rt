#!/usr/bin/env bash
set -euo pipefail

model_path="${1:-/workspace/model/Qwen3.6-27B-Q4_K_S.gguf}"
draft_path="${2:-/workspace/dflash-model/dflash-draft-3.6-q8_0-rope10m.gguf}"
output_dir="${3:-/workspace/profiles/qwen36-production}"
phases="${4:-quality,context,tuned-context,quantized-context,multiturn,sampling,concurrency,cpu}"
xrt_cli_bin="${XRT_CLI_BIN:-target/release/xrt-cli}"
corpus_dir="${XRT_PRODUCTION_CORPUS_DIR:-benchmark-corpora/text/qwen36-production-v1}"
repetitions="${XRT_PRODUCTION_REPETITIONS:-3}"
tuned_max_context="${XRT_TUNED_MAX_CONTEXT:-16384}"
quantized_max_context="${XRT_QUANTIZED_MAX_CONTEXT:-8192}"
production_draft_depth="${XRT_PRODUCTION_DRAFT_DEPTH:-15}"
production_confidence_min="${XRT_PRODUCTION_DSPARK_CONFIDENCE_MIN:-}"
production_draft_profile_us="${XRT_PRODUCTION_DSPARK_DRAFT_PROFILE_US:-}"
production_verify_profile_us="${XRT_PRODUCTION_DSPARK_VERIFY_PROFILE_US:-}"
production_confidence_temperatures="${XRT_PRODUCTION_DSPARK_CONFIDENCE_TEMPERATURES:-}"

if [[ -n "$production_confidence_min" && \
      ( -n "$production_draft_profile_us" || -n "$production_verify_profile_us" ) ]]; then
  printf 'static DSpark confidence and hardware-aware DSpark scheduling are mutually exclusive\n' >&2
  exit 2
fi
if [[ -n "$production_draft_profile_us" && -z "$production_verify_profile_us" ]] || \
   [[ -z "$production_draft_profile_us" && -n "$production_verify_profile_us" ]]; then
  printf 'both production DSpark draft and verify profiles are required\n' >&2
  exit 2
fi
if [[ -n "$production_confidence_temperatures" && \
      -z "$production_confidence_min" && -z "$production_draft_profile_us" ]]; then
  printf 'DSpark confidence temperatures require a static or hardware-aware confidence scheduler\n' >&2
  exit 2
fi

mkdir -p "$output_dir" "$output_dir/metadata"
overall_status=0

has_phase() {
  [[ ",$phases," == *",$1,"* ]]
}

common_env=(
  XRT_BACKEND=cuda
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

candidate_env=(
  XRT_QWEN_MTP=1
  XRT_QWEN_MTP_MAX_DRAFT_TOKENS="$production_draft_depth"
  XRT_QWEN_MTP_Q6_TENSOR_CORE_HEAD=0
  XRT_QWEN_DFLASH_DRAFT_MODEL="$draft_path"
  XRT_CUDA_DFLASH_Q8_0_MARLIN=1
  XRT_CUDA_DFLASH_PARALLEL_PROJECTIONS=0
)
if [[ -n "$production_confidence_min" ]]; then
  candidate_env+=(XRT_QWEN_DSPARK_CONFIDENCE_MIN="$production_confidence_min")
fi
if [[ -n "$production_draft_profile_us" ]]; then
  candidate_env+=(
    XRT_QWEN_DSPARK_DRAFT_PROFILE_US="$production_draft_profile_us"
    XRT_QWEN_DSPARK_VERIFY_PROFILE_US="$production_verify_profile_us"
  )
fi
if [[ -n "$production_confidence_temperatures" ]]; then
  candidate_env+=(XRT_QWEN_DSPARK_CONFIDENCE_TEMPERATURES="$production_confidence_temperatures")
fi

production_memory_env=(
  XRT_GPU_MEMORY_FRACTION=0.94
  XRT_GPU_RESERVED_MB=1024
  XRT_GPU_KV_FRACTION=0.55
)

base_arguments=(
  --model "$model_path"
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
  --enable-thinking false
  --json
)

record_metadata() {
  local rustc_bin cargo_bin
  rustc_bin="$(command -v rustc || true)"
  cargo_bin="$(command -v cargo || true)"
  if [[ -z "$rustc_bin" && -x "${HOME:-}/.cargo/bin/rustc" ]]; then
    rustc_bin="${HOME}/.cargo/bin/rustc"
  fi
  if [[ -z "$cargo_bin" && -x "${HOME:-}/.cargo/bin/cargo" ]]; then
    cargo_bin="${HOME}/.cargo/bin/cargo"
  fi
  sha256sum "$model_path" "$draft_path" > "$output_dir/metadata/model-sha256.txt"
  sha256sum "$xrt_cli_bin" > "$output_dir/metadata/binary-sha256.txt"
  sha256sum "$corpus_dir"/*.json > "$output_dir/metadata/corpus-sha256.txt"
  {
    printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'phases=%s\n' "$phases"
    printf 'repetitions=%s\n' "$repetitions"
    printf 'tuned_max_context=%s\n' "$tuned_max_context"
    printf 'quantized_max_context=%s\n' "$quantized_max_context"
    printf 'production_draft_depth=%s\n' "$production_draft_depth"
    printf 'production_dspark_confidence_min=%s\n' "${production_confidence_min:-disabled}"
    printf 'production_dspark_draft_profile_us=%s\n' "${production_draft_profile_us:-disabled}"
    printf 'production_dspark_verify_profile_us=%s\n' "${production_verify_profile_us:-disabled}"
    printf 'production_dspark_confidence_temperatures=%s\n' "${production_confidence_temperatures:-disabled}"
    printf 'git_head=%s\n' "$(git rev-parse HEAD 2>/dev/null || printf unavailable)"
    printf 'git_status=%s\n' "$(git status --porcelain 2>/dev/null | wc -l || true)"
    printf 'rustc=%s\n' "$(${rustc_bin:-false} --version 2>/dev/null || printf unavailable)"
    printf 'cargo=%s\n' "$(${cargo_bin:-false} --version 2>/dev/null || printf unavailable)"
    printf 'nvcc=%s\n' "$(/usr/local/cuda/bin/nvcc --version | tail -1)"
    uname -a
  } > "$output_dir/metadata/environment.txt"
  nvidia-smi -q > "$output_dir/metadata/nvidia-smi-q.txt"
  lscpu > "$output_dir/metadata/lscpu.txt"
  find Cargo.toml Cargo.lock rust-toolchain.toml .cargo crates src xtask tests \
    benches examples scripts -type f -not -path '*/__pycache__/*' -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    > "$output_dir/metadata/source-files-sha256.txt"
  sha256sum "$output_dir/metadata/source-files-sha256.txt" \
    > "$output_dir/metadata/source-tree-sha256.txt"

  model_sha="$(sha256sum "$model_path" | cut -d' ' -f1)"
  draft_sha="$(sha256sum "$draft_path" | cut -d' ' -f1)"
  {
    printf 'target_sha256=%s\n' "$model_sha"
    printf 'target_bytes=%s\n' "$(stat -c %s "$model_path")"
    if [[ "$model_sha" == "a5ef62184c1729c38c9565b502303ac88e2fad3b1c3c6aa430d9e273bdd7f917" ]]; then
      printf 'target_source=%s\n' 'https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF'
      printf 'target_source_revision=%s\n' '5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace'
      printf 'target_source_license=apache-2.0\n'
    else
      printf 'target_source=unverified-for-this-hash\n'
    fi
    printf 'draft_sha256=%s\n' "$draft_sha"
    printf 'draft_bytes=%s\n' "$(stat -c %s "$draft_path")"
    if [[ "$draft_sha" == "3612295e4928167eb84512b8b78983ab6ade6efb18bf122d5199c769556e1a1a" ]]; then
      printf 'draft_source=%s\n' 'https://huggingface.co/z-lab/Qwen3.6-27B-DFlash'
      printf 'draft_source_revision=%s\n' '0919688658996800f86b895034249700e9481106'
      printf 'draft_source_license=mit\n'
      printf 'draft_source_safetensors_sha256=%s\n' \
        'e0c050b34798d32728a164d2c3f1681746ff85c11945701b0205b654e2f1fdbe'
      printf 'draft_conversion_script_sha256=%s\n' \
        'ad5cacd88e5f380975fcc57d13e97a49ea984d1bbf7912248364536db4f86e3d'
      printf 'draft_conversion=Q8_0 with rope theta 10000000\n'
    else
      printf 'draft_source=unverified-for-this-hash\n'
    fi
  } > "$output_dir/metadata/artifact-provenance.txt"
}

run_target_suite() {
  local suite="$1"
  local output="$2"
  local runs="$3"
  env "${common_env[@]}" XRT_PREFIX_CACHE=0 XRT_QWEN_MTP=0 \
    "$xrt_cli_bin" bench --prompt-suite "$suite" --repetitions "$runs" \
    "${base_arguments[@]}" > "$output"
}

run_candidate_suite() {
  local suite="$1"
  local output="$2"
  local runs="$3"
  env "${common_env[@]}" "${candidate_env[@]}" XRT_PREFIX_CACHE=0 \
    "$xrt_cli_bin" bench --prompt-suite "$suite" --repetitions "$runs" \
    "${base_arguments[@]}" > "$output"
}

compare_and_validate() {
  local target="$1"
  local candidate="$2"
  local expected="$3"
  local prefix="$4"
  local status=0
  python3 scripts/compare-bench-token-parity.py "$target" "$candidate" \
    --output "${prefix}-parity.json" || status=1
  python3 scripts/validate-qwen36-production.py "$target" "$expected" \
    --output "${prefix}-target-validation.json" || status=1
  python3 scripts/validate-qwen36-production.py "$candidate" "$expected" \
    --output "${prefix}-candidate-validation.json" || status=1
  return "$status"
}

record_metadata

if has_phase quality; then
  mkdir -p "$output_dir/quality"
  run_target_suite "$corpus_dir/quality.json" "$output_dir/quality/target.json" "$repetitions"
  run_candidate_suite "$corpus_dir/quality.json" "$output_dir/quality/candidate.json" "$repetitions"
  compare_and_validate \
    "$output_dir/quality/target.json" \
    "$output_dir/quality/candidate.json" \
    "$corpus_dir/quality.expected.json" \
    "$output_dir/quality/result" || overall_status=1
fi

if has_phase context; then
  mkdir -p "$output_dir/context"
  : > "$output_dir/context/admitted-sizes.txt"
  for suite in "$corpus_dir"/context-[0-9][0-9][0-9][0-9][0-9].json; do
    name="$(basename "$suite" .json)"
    expected="$corpus_dir/${name}.expected.json"
    run_target_suite "$suite" "$output_dir/context/${name}-target.json" 1
    run_candidate_suite "$suite" "$output_dir/context/${name}-candidate.json" 1
    if compare_and_validate \
      "$output_dir/context/${name}-target.json" \
      "$output_dir/context/${name}-candidate.json" \
      "$expected" \
      "$output_dir/context/${name}"; then
      printf '%s\n' "$name" >> "$output_dir/context/admitted-sizes.txt"
    else
      printf '%s\n' "$name" > "$output_dir/context/first-failed-size.txt"
      overall_status=1
      break
    fi
  done
fi

if has_phase tuned-context; then
  mkdir -p "$output_dir/tuned-context"
  tuned_context_label="$(printf '%05d' "$tuned_max_context")"
  suite="$corpus_dir/context-through-${tuned_context_label}.json"
  expected="$corpus_dir/context-through-${tuned_context_label}.expected.json"
  if [[ ! -f "$suite" || ! -f "$expected" ]]; then
    printf 'missing bounded tuned-context corpus for %s tokens\n' \
      "$tuned_max_context" >&2
    exit 2
  fi
  env "${common_env[@]}" "${production_memory_env[@]}" XRT_PREFIX_CACHE=0 XRT_QWEN_MTP=0 \
    "$xrt_cli_bin" bench --prompt-suite "$suite" --repetitions 1 \
    "${base_arguments[@]}" > "$output_dir/tuned-context/target.json"
  env "${common_env[@]}" "${candidate_env[@]}" "${production_memory_env[@]}" \
    XRT_PREFIX_CACHE=0 \
    "$xrt_cli_bin" bench --prompt-suite "$suite" --repetitions 1 \
    "${base_arguments[@]}" > "$output_dir/tuned-context/candidate.json"
  python3 scripts/compare-bench-token-parity.py \
    "$output_dir/tuned-context/target.json" \
    "$output_dir/tuned-context/candidate.json" \
    --output "$output_dir/tuned-context/parity.json" || overall_status=1
  for profile in target candidate; do
    python3 scripts/validate-qwen36-production.py \
      "$output_dir/tuned-context/${profile}.json" "$expected" \
      --output "$output_dir/tuned-context/${profile}-validation.json" \
      || overall_status=1
    python3 scripts/summarize-qwen36-context.py \
      "$output_dir/tuned-context/${profile}.json" "$expected" \
      --output "$output_dir/tuned-context/${profile}-summary.json" \
      || overall_status=1
  done
fi

if has_phase quantized-context; then
  mkdir -p "$output_dir/quantized-context"
  quantized_context_label="$(printf '%05d' "$quantized_max_context")"
  suite="$corpus_dir/context-through-${quantized_context_label}.json"
  expected="$corpus_dir/context-through-${quantized_context_label}.expected.json"
  if [[ ! -f "$suite" || ! -f "$expected" ]]; then
    printf 'missing bounded quantized-context corpus for %s tokens\n' \
      "$quantized_max_context" >&2
    exit 2
  fi
  for cache_mode in q8 kq4_vq8; do
    mode_dir="$output_dir/quantized-context/$cache_mode"
    mkdir -p "$mode_dir"
    env "${common_env[@]}" "${production_memory_env[@]}" \
      XRT_PREFIX_CACHE=0 XRT_QWEN_MTP=0 \
      "$xrt_cli_bin" bench \
      --model "$model_path" \
      --prompt-suite "$suite" \
      --cache-modes "$cache_mode" \
      --backends cuda-resident \
      --cache-policy default_chat \
      --repetitions 1 \
      --concurrency 1 \
      --temperature 0 \
      --top-k 1 \
      --top-p 1 \
      --repetition-penalty 1 \
      --presence-penalty 0 \
      --frequency-penalty 0 \
      --seed 424242 \
      --enable-thinking false \
      --json > "$mode_dir/target.json"
    env "${common_env[@]}" "${candidate_env[@]}" "${production_memory_env[@]}" \
      XRT_PREFIX_CACHE=0 \
      "$xrt_cli_bin" bench \
      --model "$model_path" \
      --prompt-suite "$suite" \
      --cache-modes "$cache_mode" \
      --backends cuda-resident \
      --cache-policy default_chat \
      --repetitions 1 \
      --concurrency 1 \
      --temperature 0 \
      --top-k 1 \
      --top-p 1 \
      --repetition-penalty 1 \
      --presence-penalty 0 \
      --frequency-penalty 0 \
      --seed 424242 \
      --enable-thinking false \
      --json > "$mode_dir/candidate.json"
    python3 scripts/compare-bench-token-parity.py \
      "$mode_dir/target.json" "$mode_dir/candidate.json" \
      --output "$mode_dir/parity.json" || overall_status=1
    for profile in target candidate; do
      python3 scripts/validate-qwen36-production.py \
        "$mode_dir/${profile}.json" "$expected" \
        --output "$mode_dir/${profile}-validation.json" || overall_status=1
      python3 scripts/summarize-qwen36-context.py \
        "$mode_dir/${profile}.json" "$expected" \
        --output "$mode_dir/${profile}-summary.json" || overall_status=1
    done
  done
fi

if has_phase multiturn; then
  mkdir -p "$output_dir/multiturn"
  suite="$corpus_dir/shared-multiturn.json"
  expected="$corpus_dir/shared-multiturn.expected.json"
  env "${common_env[@]}" XRT_PREFIX_CACHE=1 XRT_QWEN_MTP=0 \
    "$xrt_cli_bin" bench --prompt-suite "$suite" --repetitions 2 \
    "${base_arguments[@]}" > "$output_dir/multiturn/target.json"
  env "${common_env[@]}" "${candidate_env[@]}" XRT_PREFIX_CACHE=1 \
    "$xrt_cli_bin" bench --prompt-suite "$suite" --repetitions 2 \
    "${base_arguments[@]}" > "$output_dir/multiturn/candidate.json"
  compare_and_validate \
    "$output_dir/multiturn/target.json" \
    "$output_dir/multiturn/candidate.json" \
    "$expected" \
    "$output_dir/multiturn/result" || overall_status=1
fi

if has_phase sampling; then
  mkdir -p "$output_dir/sampling"
  sampling_args=(
    --model "$model_path"
    --prompt-suite "$corpus_dir/quality.json"
    --cache-modes f32
    --backends cuda-resident
    --repetitions 2
    --concurrency 1
    --temperature 0.7
    --top-k 20
    --top-p 0.8
    --repetition-penalty 1
    --presence-penalty 0
    --frequency-penalty 0
    --seed 8675309
    --enable-thinking false
    --json
  )
  env "${common_env[@]}" XRT_PREFIX_CACHE=0 XRT_QWEN_MTP=0 \
    "$xrt_cli_bin" bench "${sampling_args[@]}" > "$output_dir/sampling/target.json"
  env "${common_env[@]}" "${candidate_env[@]}" XRT_PREFIX_CACHE=0 \
    "$xrt_cli_bin" bench "${sampling_args[@]}" > "$output_dir/sampling/candidate.json"
  python3 scripts/compare-bench-token-parity.py \
    "$output_dir/sampling/target.json" "$output_dir/sampling/candidate.json" \
    --output "$output_dir/sampling/parity.json" || overall_status=1
fi

if has_phase concurrency; then
  mkdir -p "$output_dir/concurrency"
  concurrency_status=0
  for concurrency in 1 2; do
    env "${common_env[@]}" "${candidate_env[@]}" "${production_memory_env[@]}" \
      XRT_PREFIX_CACHE=0 \
      "$xrt_cli_bin" bench \
      --model "$model_path" \
      --prompt "Return exactly eight lowercase hexadecimal groups separated by hyphens." \
      --cache-modes f32 \
      --backends cuda-resident \
      --max-tokens 64 \
      --repetitions 3 \
      --concurrency "$concurrency" \
      --temperature 0 \
      --top-k 1 \
      --top-p 1 \
      --repetition-penalty 1 \
      --seed 424242 \
      --enable-thinking false \
      --json > "$output_dir/concurrency/candidate-c${concurrency}.json"
    python3 scripts/validate-qwen36-concurrency.py \
      "$output_dir/concurrency/candidate-c${concurrency}.json" \
      --expected-concurrency "$concurrency" \
      --expected-repetitions 3 \
      --output "$output_dir/concurrency/candidate-c${concurrency}-validation.json" \
      || concurrency_status=1
  done
  if [[ "$concurrency_status" -ne 0 ]]; then
    overall_status=1
  fi
fi

if has_phase cpu; then
  mkdir -p "$output_dir/cpu"
  env XRT_QWEN_MTP=0 XRT_NGRAM_SPECULATION=0 XRT_PREFIX_CACHE=0 \
    "$xrt_cli_bin" bench \
    --model "$model_path" \
    --prompt "Reply with OK." \
    --cache-modes f32 \
    --backends cpu \
    --max-tokens 1 \
    --repetitions 1 \
    --concurrency 1 \
    --temperature 0 \
    --top-k 1 \
    --top-p 1 \
    --repetition-penalty 1 \
    --seed 424242 \
     --enable-thinking false \
     --json > "$output_dir/cpu/one-token.json"
  python3 scripts/validate-xrt-bench-success.py \
    "$output_dir/cpu/one-token.json" \
    --expected-backend cpu \
    --expected-repetitions 1 \
    --required-text OK \
    --output "$output_dir/cpu/validation.json" || overall_status=1
fi

printf 'production benchmark phases completed: %s\n' "$phases"
exit "$overall_status"
