#!/usr/bin/env bash
# Verify one local model actually runs on this machine, and emit the facts a
# catalog entry needs. Written because a catalog entry is a promise: listing a
# model the runtime has never executed makes the download the user's problem.
#
#   scripts/verify-local-model.sh MODEL.gguf OUT_DIR [extra env assignments...]
#
# Exits nonzero when the model fails to load, generates nothing, or any case
# reports an error. The CLI's own bench gate enforces the last two.
set -euo pipefail

model_path="${1:?usage: verify-local-model.sh MODEL.gguf OUT_DIR [ENV=VAL ...]}"
output_dir="${2:?missing output directory}"
shift 2

xrt_cli_bin="${XRT_CLI_BIN:-target/release/xrt-cli}"
suite_path="${XRT_VERIFY_SUITE:-benchmark-corpora/text/qwen38-greedy-admission-v1.json}"
repetitions="${XRT_VERIFY_REPETITIONS:-2}"

for artifact in "$model_path" "$xrt_cli_bin" "$suite_path"; do
  [[ -f "$artifact" ]] || { printf 'missing: %s\n' "$artifact" >&2; exit 2; }
done
mkdir -p "$output_dir"

# Free VRAM before load, so the reported floor is a measurement and not a guess.
free_before=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)

common=(
  XRT_BACKEND=cuda
  XRT_PREFIX_CACHE=0
  XRT_NGRAM_SPECULATION=0
  XRT_CUDA_Q4_K_MARLIN=1
  XRT_CUDA_Q5_K_MARLIN=1
  XRT_CUDA_Q6_K_MARLIN=1
)

printf 'verifying %s\n' "$model_path"
env "${common[@]}" "$@" \
  "$xrt_cli_bin" bench \
    --model "$model_path" \
    --prompt-suite "$suite_path" \
    --enable-thinking false \
    --cache-modes f32 \
    --backends cuda-resident \
    --cache-policy default_chat \
    --repetitions "$repetitions" \
    --concurrency 1 \
    --temperature 0 --top-k 1 --top-p 1 \
    --repetition-penalty 1 --presence-penalty 0 --frequency-penalty 0 \
    --seed 424242 --json \
  > "$output_dir/bench.json" 2> "$output_dir/bench.log"
bench_status=$?

sha256sum "$model_path" > "$output_dir/model-sha256.txt"
stat -c '%s' "$model_path" > "$output_dir/model-bytes.txt"
nvidia-smi -q > "$output_dir/nvidia-smi-q.txt" 2>/dev/null || true

python3 - "$output_dir" "$model_path" "$free_before" "$bench_status" <<'PY'
import json, os, statistics as st, sys
out_dir, model_path, free_before, status = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
report = json.load(open(os.path.join(out_dir, "bench.json")))
results = report["results"]
rates = [r["decode_tok_s"] for r in results if r.get("decode_tok_s")]
vram = [r.get("tracked_resident_vram_bytes") for r in results]
vram = [v for v in vram if v]
device_used = [(r.get("gpu_resource") or {}).get("device_used_vram_bytes") for r in results]
device_used = [v for v in device_used if v]
summary = {
    "object": "xrt.local_model_verification",
    "model_file": os.path.basename(model_path),
    "size_bytes": int(open(os.path.join(out_dir, "model-bytes.txt")).read().strip()),
    "sha256": open(os.path.join(out_dir, "model-sha256.txt")).read().split()[0],
    "architecture": sorted({r.get("model_architecture") for r in results if r.get("model_architecture")}),
    "backend": sorted({r.get("active_backend") for r in results if r.get("active_backend")}),
    "quantization": report.get("quantization"),
    "cases": len(results),
    "cases_with_output": sum(1 for r in results if (r.get("output_tokens") or 0) > 0),
    "errors": [r.get("error") for r in results if r.get("error")],
    "decode_tok_s": {
        "mean": st.mean(rates) if rates else 0.0,
        "median": st.median(rates) if rates else 0.0,
        "min": min(rates) if rates else 0.0,
        "max": max(rates) if rates else 0.0,
    },
    "tracked_resident_vram_bytes_peak": max(vram) if vram else None,
    "device_used_vram_bytes_peak": max(device_used) if device_used else None,
    "host_free_vram_mib_before_load": free_before,
    "bench_exit_code": status,
}
summary["verified"] = (
    status == 0
    and not summary["errors"]
    and summary["cases_with_output"] == summary["cases"]
    and summary["cases"] > 0
)
json.dump(summary, open(os.path.join(out_dir, "verification.json"), "w"), indent=2)
print(json.dumps({k: summary[k] for k in
                  ("model_file","architecture","quantization","cases","cases_with_output",
                   "decode_tok_s","tracked_resident_vram_bytes_peak","device_used_vram_bytes_peak","verified")}, indent=2))
if not summary["verified"]:
    print("VERIFICATION FAILED", file=sys.stderr)
    raise SystemExit(1)
PY
