# Qwen3.6 recurrent-journal A/B — RTX 4090 — 2026-08-09

This record tests XRT's direct recurrent rollback journal for fused Qwen3.6
MTP verification. It is controlled performance evidence, not a general
production claim.

## Result

The experiment used a candidate-control-candidate ordering. Each invocation
ran three warmups followed by ten retained measurements.

| Path | Retained samples | Decode tok/s, mean +/- sample SD | 95% CI half-width |
| --- | ---: | ---: | ---: |
| Direct persistent journal | 20 | **73.212 +/- 0.114** | 0.050 |
| Legacy-copy workload control | 10 | 72.360 +/- 0.090 | 0.055 |

Direct journaling improved decode throughput by **1.18%** on this workload.
Both paths produced one identical preview and exactly 55 accepted of 68
drafted tokens in nine verification batches on every retained repetition.

## Mechanism

The production candidate lets each fused DeltaNet verifier write all rollback
boundaries directly into one persistent per-layer journal. The control uses
the same production state layout and kernel, then reintroduces the removed
per-boundary device-copy workload after state publication. This isolates the
copy/launch cost without changing the generated state or output.

Per request, the candidate removed:

- 5,664 device-to-device calls (33.40%);
- 9,256,797,856 device-to-device bytes (64.37%); and
- 10,205 microseconds of mean verifier time (1.37%).

The control does not recreate the former vector-of-buffers allocation layout,
so the A/B measures the eliminated copy workload only. The production change
also removes those per-boundary journal allocations, but no allocation-speed
claim is made from this control.

## Workload

- GPU: secure-cloud NVIDIA GeForce RTX 4090, UUID
  `GPU-8495639f-ad2f-49b2-d9ea-65a065e3ab43`, driver `570.195.03`;
- model: `Qwen3.6-27B-Q4_K_S.gguf`, 16,121,357,440 bytes, SHA-256
  `a5ef62184c1729c38c9565b502303ac88e2fad3b1c3c6aa430d9e273bdd7f917`;
- prompt: `Write the numbers from 1 to 100 in order, separated by commas, and do not stop early.`;
- greedy generation, 64 tokens, seed 424242, F32 KV, one stream;
- MTP depth eight, adaptive fallback off, 65,536 draft-vocabulary rows; and
- packed-GGUF tensor-core verification enabled.

The raw JSON files are unmodified CLI output. `summary.json` contains the
aggregates and `control-method.md` records the benchmark-only control.

## Verification and cleanup

- local CUDA-feature runtime tests: 100 passed, 28 hardware-gated;
- physical RTX 4090 recurrent-state tests: 5 passed;
- release candidate and control builds: passed;
- post-run RunPod inventory: zero pods and zero endpoints.

The secure RTX 4090 was quoted at $0.74/hour and was terminated immediately
after evidence collection. Billing settlement lagged the deletion, so no exact
final charge is claimed.
