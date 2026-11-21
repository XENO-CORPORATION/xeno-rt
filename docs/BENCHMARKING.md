# Benchmarking

Performance claims are accepted only with reproducible commands and enough
environment metadata to explain the result. Benchmarking is not a substitute
for correctness parity.

## Safety First

Real-model CPU and CUDA runs can consume most system RAM or VRAM. Prefer hosted
CI and the guarded self-hosted GPU workflow. Before a local GPU run:

1. confirm the intended model size and quantization;
2. inspect device-wide memory use with `nvidia-smi`;
3. stop if existing GPU use exceeds the operator threshold;
4. set conservative GPU budget and token limits;
5. run one process and one sequence first;
6. verify cleanup before increasing load.

The repository's guarded GPU runner uses a 4096 MiB initial-use threshold for
shared-machine validation. Do not weaken that guard to make a benchmark start.

## Compile Benchmarks Without Running

```bash
cargo bench --workspace --no-run --locked
```

This is the normal CI gate. It validates benchmark code without starting a
model workload.

## Runtime Benchmark Command

```bash
cargo run --release --locked -p xrt-cli -- bench \
  --model ./models/model.gguf \
  --prompt "Benchmark prompt" \
  --backends cpu \
  --cache-modes f32,q8,agent_adaptive \
  --max-tokens 64 \
  --repetitions 3 \
  --seed 1 \
  --json
```

For CUDA, build the CLI with the feature and start with one token:

```bash
cargo run --release --locked -p xrt-cli --features cuda -- bench \
  --model ./models/model.gguf \
  --prompt "Hello" \
  --backends cuda \
  --cache-modes f32 \
  --max-tokens 1 \
  --repetitions 1 \
  --concurrency 1 \
  --seed 1 \
  --json
```

The JSON report records requested and active backend, model/architecture,
prompt/output tokens, load/prefill/total times, throughput, process memory, GPU
resource status, cache/scheduler status, transfer deltas, allocation deltas,
and errors.

## Frozen Prompt-Suite Parity

Use `--prompt-suite` for a versioned local admission corpus. The runtime loads
the model once per requested backend, then executes every case in the suite.
Suite runs require concurrency one and add `suite_id`, `case_id`, and the exact
`output_token_ids` to the JSON report. Exact token IDs are deliberately limited
to local single-sequence runs; an external OpenAI-compatible endpoint does not
expose a portable token-ID contract.

```bash
target/release/xrt-cli bench \
  --model /workspace/model/Qwen3.6-27B-Q4_K_S.gguf \
  --prompt-suite benchmark-corpora/text/qwen36-greedy-admission-v1.json \
  --backends cuda-resident \
  --cache-modes f32 \
  --temperature 0 \
  --top-k 1 \
  --top-p 1 \
  --repetition-penalty 1 \
  --seed 424242 \
  --json
```

The Qwen3.6 admission helper runs target-only, scalar-head MTP, tensor-head
MTP, and the retained depth-ten tiled-verifier arm with the same frozen cases.
It then fails unless every repetition is internally deterministic and every
candidate token trace exactly matches the target-only trace:

```bash
scripts/benchmark-qwen36-greedy-admission.sh \
  /workspace/model/Qwen3.6-27B-Q4_K_S.gguf \
  /workspace/profiles/qwen36-greedy-admission \
  1
```

The parity comparator is also independently reusable:

```bash
python3 scripts/compare-bench-token-parity.py baseline.json candidate.json
```

Passing this corpus is evidence for deterministic greedy parity on its pinned
cases only. It does not admit non-greedy speculative sampling, long context,
concurrency, cancellation, recovery, another model hash, or another GPU.

## Qwen3.8 Official MTP Admission

The Qwen3.8 helper compares the official Q4_K_M target with its separate
official Q8_0 NextN companion. It defaults to the retained depth-four policy,
disables the unadmitted target verify graph, records artifact and environment
hashes, and rejects any token mismatch:

```bash
XRT_QWEN38_REPETITIONS=3 \
  scripts/benchmark-qwen38-mtp.sh \
  /models/Qwen3.8-27B-Q4_K_M.gguf \
  /models/mtp-Qwen3.8-27B-Q8_0.gguf \
  /profiles/qwen38-greedy
```

Generate the Qwen3.8 quality, multi-turn, and context corpora and run their
separate production gates with:

```bash
python3 scripts/generate-qwen36-production-corpora.py \
  --suite-prefix qwen38 \
  --output-dir benchmark-corpora/text/qwen38-production-v1
scripts/benchmark-qwen38-production.sh \
  /models/Qwen3.8-27B-Q4_K_M.gguf \
  /models/mtp-Qwen3.8-27B-Q8_0.gguf \
  /profiles/qwen38-production \
  quality,multiturn,context
```

See [Qwen3.8-27B Official NextN/MTP Admission](QWEN38_MTP_ADMISSION.md)
for the pinned tuple, retained configuration, evidence, and claim boundary.

## Qwen3.6 Production-Admission Matrix

Generate the deterministic quality, shared-history, and long-context corpora,
then run the production matrix one phase at a time so a failed gate does not
erase later evidence:

```bash
python3 scripts/generate-qwen36-production-corpora.py
scripts/benchmark-qwen36-production.sh \
  /workspace/model/Qwen3.6-27B-Q4_K_S.gguf \
  /workspace/dflash-model/dflash-draft-3.6-q8_0-rope10m.gguf \
  /workspace/profiles/qwen36-production \
  quality,context,tuned-context,quantized-context,multiturn,sampling,concurrency,cpu
```

The matrix records immutable model, draft, corpus, binary, source-tree, GPU,
CPU, CUDA, and toolchain metadata. The quality and context phases validate
required output text as well as generation success. Candidate greedy runs also
require exact target-token parity. `tuned-context` defaults to the complete
through-16K ladder under its documented memory policy, which includes at least
one point beyond the admitted DFlash profile; set `XRT_TUNED_MAX_CONTEXT` only
to another generated bounded corpus. Because long-prompt causal attention is
expensive, `quantized-context` defaults to the complete through-8K ladder; set
`XRT_QUANTIZED_MAX_CONTEXT=16384` only when the additional runtime and cost are
explicitly intended. Every quantized-cache mode runs both target-only and
DFlash arms, validates the required answer, and records exact greedy token
parity. A capacity failure is not extrapolated into a successful larger
context, and a later size is never needed to establish a smaller candidate's
admission boundary.

This matrix uses `--enable-thinking false` for the fast non-thinking profile.
Thinking-enabled quality must be reported separately; a non-thinking parity
pass proves implementation parity, not task correctness or equivalent model
quality.

Local JSON results report `prompt_tokenize_ms` separately from model execution.
`prefill_ms`, `total_ms`, and `tok_s` remain inference-only for comparison with
older XRT records; `end_to_end_ms` and `end_to_end_tok_s` add one prompt encode.
Use the live OpenAI service harness for authoritative client-observed latency,
because it also includes HTTP, queueing, template rendering, and response work.

The DFlash A/B helper treats its repetition argument as measured repetitions
and prepends one warmup repetition per case by default. Raw JSON retains every
sample for auditability; reported statistics must exclude repetition 1 as
recorded in `measurement-policy.txt`. Set
`XRT_BENCH_WARMUP_REPETITIONS` only when the policy change is documented.

The experimental Qwen3.6 DSpark screen runs the full fixed-depth ladder before
testing confidence scheduling. It derives a hardware profile from the median
post-warmup draft and target-verification microseconds per cycle:

```bash
bash scripts/benchmark-qwen36-dspark.sh \
  /workspace/model/Qwen3.6-27B-Q4_K_S.gguf \
  /workspace/dspark-model/Qwen3.6-27B-DSpark-Q8_0.gguf \
  /workspace/profiles/qwen36-greedy-production-256-v1.json \
  /workspace/profiles/qwen36-dspark-production-256
```

`hardware-profile.json` records the measured curve. The runtime consumes its
`XRT_QWEN_DSPARK_DRAFT_PROFILE_US` scalar and
`XRT_QWEN_DSPARK_VERIFY_PROFILE_US` comma-separated values for prefix lengths
0 through 15. For each draft cycle it computes cumulative prefix-survival
probabilities from the DSpark conditional confidence head and selects the
prefix with the highest expected tokens per profiled cycle. An optional
`XRT_QWEN_DSPARK_CONFIDENCE_TEMPERATURES` list applies position-wise sequential
temperature calibration. The hardware-aware profile and the older static
`XRT_QWEN_DSPARK_CONFIDENCE_MIN` threshold are mutually exclusive.

The same screen also compares the default parallel FFN verifier against a
serialized gate/up projection arm and capped parallel Marlin grids. The
default grid caps are 64 and 96 blocks and can be replaced with
`XRT_DSPARK_VERIFY_GRID_SWEEP`; set `XRT_DSPARK_SERIAL_SWIGLU_SCREEN=0` to omit
the serialized arm. These are candidate measurements, not portable defaults,
and remain subject to the same frozen-suite exact-token-parity gate.

The `prefer-ngram` arm measures the runtime's composed product policy: a
history-matched eight-token continuation is proposed without running the neural
drafter, and DSpark remains the fallback when no history match exists. This arm
is kept out of the fixed-depth hardware curve because its proposal cost and
availability depend on request history, but it is still fully target-verified
and must pass the same exact greedy token-parity gate.

To compare that product composition against both retained neural drafters with
one reusable target baseline, run:

```bash
bash scripts/benchmark-qwen36-draft-strategies.sh \
  /workspace/model/Qwen3.6-27B-Q4_K_S.gguf \
  /workspace/dflash-model/dflash-draft-3.6-q8_0-rope10m.gguf \
  /workspace/dspark-model/Qwen3.6-27B-DSpark-Q8_0.gguf \
  /workspace/profiles/qwen36-greedy-production-256-v1.json \
  /workspace/profiles/qwen36-draft-strategies
```

The four exact-parity arms are DFlash, DFlash with history-first n-grams,
DSpark, and DSpark with history-first n-grams. The screen pins order eight with
`XRT_QWEN_MTP_NGRAM_ORDER=8`; the runtime default remains order three unless a
profile selects another bounded order. Results are ranked by equally weighted
mean decode throughput and retain aggregate throughput separately.

This scheduling changes only how many proposals are sent to the target. The
selected prefix is still verified by the ordinary target runtime, so exact
greedy token parity remains mandatory. Profiles are hardware, model, runtime,
context, and concurrency specific; never reuse a curve as a portable default
or a throughput guarantee.

Run the service gate against loopback with an intentionally bounded scheduler:

```bash
env \
  XRT_NGRAM_SPECULATION=0 \
  XRT_QWEN_MTP=1 XRT_QWEN_MTP_MAX_DRAFT_TOKENS=15 \
  XRT_QWEN_MTP_ADAPTIVE_FALLBACK=0 XRT_QWEN_MTP_VOCAB_ROWS=65536 \
  XRT_QWEN_MTP_BATCHED_REBASE=1 XRT_QWEN_MTP_Q6_TENSOR_CORE_HEAD=0 \
  XRT_QWEN_DFLASH_DRAFT_MODEL=/workspace/dflash-model/dflash-draft-3.6-q8_0-rope10m.gguf \
  XRT_CUDA_Q4_K_MARLIN=1 XRT_CUDA_KQUANT_TENSOR_CORE_VERIFY=1 \
  XRT_CUDA_PARALLEL_VERIFY_PROJECTIONS=1 XRT_CUDA_MARLIN_FUSED_EPILOGUES=1 \
  XRT_QWEN_MTP_VERIFY_GRAPH=1 XRT_CUDA_DFLASH_Q8_0_MARLIN=1 \
  XRT_CUDA_DFLASH_PARALLEL_PROJECTIONS=0 \
  XRT_GPU_MEMORY_FRACTION=0.94 XRT_GPU_RESERVED_MB=1024 XRT_GPU_KV_FRACTION=0.55 \
  target/release/xrt-server \
  --model /workspace/model/Qwen3.6-27B-Q4_K_S.gguf \
  --backend cuda-resident \
  --host 127.0.0.1 --port 18080 \
  --max-active-sequences 1 --max-queued-sequences 4 \
  > /workspace/profiles/qwen36-production/api/server.log 2>&1 &
server_pid=$!

python3 scripts/benchmark-xrt-openai-service.py \
  --base-url http://127.0.0.1:18080 \
  --server-pid "$server_pid" \
  --model-path /workspace/model/Qwen3.6-27B-Q4_K_S.gguf \
  --long-context-suite benchmark-corpora/text/qwen36-production-v1/context-08192.json \
  --long-context-expected benchmark-corpora/text/qwen36-production-v1/context-08192.expected.json \
  --timeout 1200 \
  --soak-requests 100 \
  --output /workspace/profiles/qwen36-production/api/service.json

kill "$server_pid"
wait "$server_pid" || true
```

This verifies both completion routes, streaming terminal reasons, direct and
nested thinking controls, thinking-enabled arithmetic, multi-turn recall,
bounded concurrency, queue-overload backpressure, streaming cancellation,
invalid input, scheduler cleanup, soak latency and memory growth, plus unload,
unavailable behavior, reload, and post-reload inference.

For the pinned Qwen3.8-27B RTX 4090 tuple, start the service through the same
profile used by admission instead of reconstructing its environment manually:

```bash
XRT_SERVER_BIN=target/release/xrt-server \
  bash scripts/run-qwen38-production-server.sh \
  /models/Qwen3.8-27B-Q4_K_M.gguf \
  /models/mtp-Qwen3.8-27B-Q8_0.gguf \
  18080
```

The launcher binds to loopback by default, pins the admitted depth-four MTP,
Marlin, batched-prefill, memory, and scheduler profile, and leaves the standard
OpenAI-compatible endpoints unchanged. The Qwen3.8 service gate reuses the
same harness with `--request-model Qwen3.8-27B`. The CLI suite retains all six
context sizes for exact target/MTP parity; an API qualification may select the
maximum case with `--long-context-case context_08192_needle_00201` after the
complete CLI suite passes, avoiding duplicate lower-context work without
extrapolating the maximum result.

## Portable vs Native CPU Builds

Release and CI builds are portable and do not set `target-cpu=native`. Local
microbenchmarks may opt in explicitly.

PowerShell:

```powershell
$env:RUSTFLAGS = '-C target-cpu=native'
cargo bench --workspace --locked
Remove-Item Env:RUSTFLAGS
```

Bash:

```bash
RUSTFLAGS='-C target-cpu=native' cargo bench --workspace --locked
```

Never compare a native local build to a portable release build without labeling
the difference.

## Required Result Metadata

Every published result must include:

- xeno-rt commit and dirty-tree state;
- exact command and environment overrides;
- model repository/file/hash, architecture, and quantization;
- CPU model, core/thread policy, RAM, OS, and power mode;
- GPU model, driver, visible VRAM, and CUDA path when applicable;
- build profile and `RUSTFLAGS`;
- prompt tokens, generated tokens, seed, cache mode, and concurrency;
- warmup/repetition count and raw JSON output;
- correctness/parity gate used before timing.

Report median and distribution, not only the fastest sample. Keep load time,
prefill latency, time to first token, decode throughput, and peak memory as
separate measurements.

## Regression Rule

A performance-sensitive pull request must compare the same command on the base
and head commits. Correctness, API compatibility, CPU fallback, and memory
safety take precedence over a throughput improvement.
