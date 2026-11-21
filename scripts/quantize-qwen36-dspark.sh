#!/usr/bin/env bash
set -euo pipefail

quantize_bin="${LLAMA_QUANTIZE_BIN:-/workspace/research/llama-dspark/build-quant/bin/llama-quantize}"
input_path="${1:-/workspace/dspark-model/Qwen3.6-27B-DSpark.gguf}"
output_dir="${2:-/workspace/dspark-model}"
threads="${XRT_DSPARK_QUANT_THREADS:-$(nproc)}"

mkdir -p "$output_dir"

for required in "$quantize_bin" "$input_path"; do
  if [[ ! -f "$required" ]]; then
    printf 'missing required DSpark quantization input: %s\n' "$required" >&2
    exit 2
  fi
done

run_quant() {
  local output="$1"
  shift
  if [[ -e "$output" ]]; then
    printf 'refusing to overwrite existing quantized artifact: %s\n' "$output" >&2
    exit 2
  fi
  "$quantize_bin" "$@" "$input_path" "$output" Q8_0 "$threads"
}

# Control: quantize every eligible 2D draft tensor to Q8_0.
run_quant "$output_dir/Qwen3.6-27B-DSpark-Q8_0.gguf" --pure

# Accuracy-oriented candidate: keep both Markov maps and the confidence
# projection at F16 while the five-layer DSpark backbone remains Q8_0. XRT
# executes all of these tensors resident on the GPU, so this does not restore
# the former host-token round trip.
run_quant "$output_dir/Qwen3.6-27B-DSpark-Q8_0-MarkovF16.gguf" \
  --pure \
  --tensor-type '^markov_w1\.weight$=f16' \
  --tensor-type '^markov_w2\.weight$=f16' \
  --tensor-type '^conf_proj\.weight$=f16'

# Throughput/quality midpoint: W1 stays compact Q8_0 for the lookup while W2,
# which directly changes the logits, retains F16 precision and tensor-core GEMM.
run_quant "$output_dir/Qwen3.6-27B-DSpark-Q8_0-MarkovW2F16.gguf" \
  --pure \
  --tensor-type '^markov_w2\.weight$=f16' \
  --tensor-type '^conf_proj\.weight$=f16'

sha256sum \
  "$input_path" \
  "$output_dir/Qwen3.6-27B-DSpark-Q8_0.gguf" \
  "$output_dir/Qwen3.6-27B-DSpark-Q8_0-MarkovF16.gguf" \
  "$output_dir/Qwen3.6-27B-DSpark-Q8_0-MarkovW2F16.gguf" \
  > "$output_dir/dspark-quantized-sha256.txt"

