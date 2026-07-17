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
