# Qwen3.6 MTP draft-head profile and tensor-core screen

This record profiles the stable-graph MTP draft phase and evaluates two
existing XRT Q6_K projection strategies against the scalar packed-Q6 control.
It retains a Q6_K tensor-core draft-head candidate and rejects single-row Q8
MMQ reuse.

## Registered tuple

- GPU: NVIDIA GeForce RTX 4090, 24 GB, driver 570.158.01, 450 W limit
- remote worker: RunPod pod `b3xo3ohcu4uw1b`
- image: `runpod/pytorch:0.7.0-cu1241-torch260-ubuntu2204`
- model: `Qwen3.6-27B-Q4_K_S.gguf`, 16,121,357,440 bytes
- model SHA-256: `a5ef62184c1729c38c9565b502303ac88e2fad3b1c3c6aa430d9e273bdd7f917`
- source base: `77911ff37ee8a8e94c11815726a4008dd949f1e0` plus the recorded workspace candidate
- decode: the registered 64-token greedy prompt, depth eight, 65,536 draft
  rows, F32 KV, stable verifier graph, one stream
- statistics: 18 repetitions per final arm, first discarded, 17 retained

## Bounded draft profile

Nsight Systems captured only draft window four via
`XRT_QWEN_MTP_PROFILE_DRAFT_WINDOW=4`. The eight-token window contained
9.163 ms of summed GPU kernel time:

| Kernel family | Total | Share of kernel time |
|---|---:|---:|
| packed Q6_K draft output head | 4.885 ms | 53.3% |
| Marlin Q4_K predictor projections | 2.022 ms | 22.1% |
| Q8_0 predictor projections | 0.729 ms | 8.0% |
| RMSNorm | 0.560 ms | 6.1% |
| single-query attention | 0.460 ms | 5.0% |
| remaining kernels | 0.508 ms | 5.5% |

There were eight device-to-host token readbacks. Their CUDA API calls occupied
7.673 ms because every next predictor step depends on the previous argmax.
This identifies two distinct limits: Q6 output-head bandwidth and a serial
host-visible token boundary between predictor steps.

Canonical profiler artifacts are `xrt-draft-window4.nsys-rep`, its exported
SQLite database, and `xrt-draft-window4-stats.csv`.

## Final A/B result

| Path | Retained n | Mean tok/s | SD | Median | Min-max | Mean draft |
|---|---:|---:|---:|---:|---:|---:|
| scalar packed-Q6 control | 17 | 149.2613 | 0.3301 | 149.2696 | 148.5525-149.7596 | 83.409 ms |
| **Q6_K tensor-core draft head** | **17** | **151.1889** | **0.3410** | **151.2126** | **150.3680-151.6712** | **78.967 ms** |

The candidate improves end-to-end decode by **1.9276 tok/s (1.291%)** and
reduces draft time by **4.443 ms (5.326%)**. A Welch comparison gives
`t=16.746`, approximately 32 degrees of freedom, with an approximate 95%
difference interval of `+1.693` to `+2.162 tok/s`.

Both arms accepted 55 of 68 proposals in nine verifier batches, produced one
identical deterministic preview across retained runs, and reported zero
errors. Target verification remains complete and authoritative; the
tensor-core path changes only speculative draft logits.

The earlier seven-sample confirmation files are retained alongside the final
arms. Their same-host means were 150.891 tok/s for tensor core and 149.581
tok/s for control.

## Rejected MMQ result

Reusing the batched Q6xQ8 DP4A MMQ path for one draft row preserved 55/68
acceptance and the same preview but regressed warm mean throughput to 142.649
tok/s and raised draft time to 103.296 ms. Per-token Q8 activation setup is
not amortized at batch one. This path was removed from product code.

## Reproduction

```bash
# Control
XRT_CLI_BIN=/workspace/xeno-rt/target/release/xrt-cli \
XRT_QWEN_MTP_VERIFY_GRAPH=1 \
scripts/benchmark-qwen36-mtp.sh \
  /workspace/model/Qwen3.6-27B-Q4_K_S.gguf 18 control.json

# Candidate
XRT_CLI_BIN=/workspace/xeno-rt/target/release/xrt-cli \
XRT_QWEN_MTP_VERIFY_GRAPH=1 \
XRT_QWEN_MTP_Q6_TENSOR_CORE_HEAD=1 \
scripts/benchmark-qwen36-mtp.sh \
  /workspace/model/Qwen3.6-27B-Q4_K_S.gguf 18 tensor-core.json
```

This is a narrow experimental performance admission. The candidate remains
opt-in until multi-prompt acceptance/performance, long-context, concurrency,
failure recovery, and clean-environment reproduction gates pass.
