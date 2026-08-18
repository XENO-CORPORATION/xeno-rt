# XRT Qwen3.6-27B MTP Benchmark Report — 2026-08-09

- **Runtime domain:** `xrt-text`
- **Report status:** Controlled performance evidence; not production admission
- **Candidate status:** Dirty development tree based on commit `77911ff`
- **Machine-readable evidence:** [`qwen36-adaptive-mtp-rtx4090-2026-08-09.json`](../benchmark-results/text/qwen36-adaptive-mtp-rtx4090-2026-08-09.json)
- **Admission record:** [Qwen3.6 NextN/MTP admission](QWEN36_MTP_ADMISSION.md)

## 1. Result

The retained XRT-native MTP configuration passed the narrow 50 tokens/second
workload objective in all five measured repetitions.

| Path | n | Mean tok/s | Median tok/s | Sample SD | Range tok/s |
|---|---:|---:|---:|---:|---:|
| Clean target-only | 3 | 30.723067 | 30.722300 | 0.004400 | 30.719100-30.727800 |
| Retained MTP | 5 | 50.242600 | 50.268606 | 0.074422 | 50.110936-50.294142 |

The MTP mean was **63.53%** above clean target-only decode. The sample
coefficient of variation was approximately 0.15% for MTP and 0.014% for the
target-only control.

This result supports one claim only: the retained XRT configuration exceeded
50 tokens/second on the pinned workload and environment. It does not establish
a general runtime average.

## 2. Environment identity

| Field | Value |
|---|---|
| Provider | RunPod secure cloud |
| Region | EU-RO-1 |
| GPU | NVIDIA GeForce RTX 4090 |
| Visible VRAM | 24,564 MiB |
| Driver | 570.195.03 |
| CUDA toolkit | 12.8.1 |
| PTX target | Portable `compute_70` |
| Candidate base | `77911ff` |
| Working tree | Dirty |

The checked-in `q4_k_recurrent.ptx` used for the retained candidate had SHA-256
`94c1f656a8a6ccb5d0034bb24041b2a73c057d95db24233de801e90848fbfdec`
in both the local and remote verification environments.

## 3. Model identity

| Field | Value |
|---|---|
| Artifact | `Qwen3.6-27B-Q4_K_S.gguf` |
| Size | 16,121,357,440 bytes |
| SHA-256 | `a5ef62184c1729c38c9565b502303ac88e2fad3b1c3c6aa430d9e273bdd7f917` |
| Weight quantization | Q4_K_S artifact with Q4_K/Q5_K/Q6_K packed matrices |
| Predictor | Integrated one-layer NextN/MTP path |

Model weights and GGUF are third-party ecosystem artifacts. The benchmarked
runtime implementation is XRT-native.

## 4. Workload controls

| Control | Value |
|---|---|
| Prompt | `Write the numbers from 1 to 100 in order, separated by commas, and do not stop early.` |
| Prompt tokens | 34 |
| Maximum output tokens | 64 |
| Actual output tokens | 28 in every retained run |
| Temperature | 0.0 |
| Top-k | 1 |
| Top-p | 1.0 |
| Repetition penalty | 1.0 |
| Seed | 424242 |
| Concurrency | Single stream |
| KV cache | F32 |
| Prefix cache | Off |
| N-gram speculation | Off |
| MTP depth | 8 |
| Draft vocabulary rows | 74,752 |

The short 28-token completion is an important limitation. It reduces the
amount of behavior exercised and prevents this sample from representing long
responses, long context, agent turns, or sustained server load.

## 5. Raw results

### 5.1 Clean target-only

| Repetition | Decode tok/s |
|---:|---:|
| 1 | 30.727800 |
| 2 | 30.722300 |
| 3 | 30.719100 |

### 5.2 Retained MTP

| Repetition | Decode tok/s | Drafted | Accepted | Rejected | Verify batches |
|---:|---:|---:|---:|---:|---:|
| 1 | 50.110936 | 32 | 24 | 8 | 4 |
| 2 | 50.294142 | 32 | 24 | 8 | 4 |
| 3 | 50.272950 | 32 | 24 | 8 | 4 |
| 4 | 50.268606 | 32 | 24 | 8 | 4 |
| 5 | 50.266367 | 32 | 24 | 8 | 4 |

The proposal-token acceptance rate was 75%. Output-token count, preview, draft
count, acceptance count, rejection count, and verification-batch count were
stable across the five repetitions.

## 6. Performance mechanism

The retained candidate combined:

- device-side draft argmax;
- a 74,752-row draft projection prefix;
- bit-exact Q4_K verification with a warp-shuffle reduction;
- dedicated bit-exact 2-16 row Q5_K verification;
- packed-weight tiled Q6_K verification for small windows;
- accepted-boundary rebase that retains the already-correct first predictor KV
  row; and
- adaptive fallback after at least six proposals when cumulative acceptance is
  below 25%.

### 6.1 Instrumented phase timings

| Phase | Observed range per retained run |
|---|---:|
| Draft | 42.239-42.323 ms |
| Verify | 478.299-480.170 ms |
| Rebase | 11.019-11.041 ms |

Verification represented about 90% of the sum of these three instrumented
phases. This is not a percentage of every end-to-end operation, but it clearly
identifies verification as the next dominant optimization target.

### 6.2 Post-optimization profile

| Kernel | Total time | Instances |
|---|---:|---:|
| `xrt_q4_k_verify_matmul_8` | 310.066 ms | 1,700 |
| `xrt_q5_k_verify_matmul_8` | 38.368 ms | 300 |
| `xrt_q6_k_tiled_matmul` | 28.816 ms | 5 |

The Q4 verifier is the largest named verify cost and the launch count is high.
Any fusion, persistence, or graph proposal must beat this retained profile
while preserving exactness.

## 7. Correctness and regression controls

The retained evidence records:

- stable greedy output for the measured target/MTP prompt;
- bit-exact Q4_K verifier testing;
- bit-exact Q5_K verifier testing;
- portable PTX reproduction; and
- an adaptive low-acceptance code control.

For the code control:

| Path | Decode tok/s | Output tokens |
|---|---:|---:|
| Target-only | 31.881383 | 128 |
| MTP with adaptive fallback | 32.072054 | 128 |

MTP drafted eight tokens, accepted one, triggered one fallback, and preserved
the 128-token output preview. The control demonstrates that the retained
fallback avoids sustained low-acceptance drafting in this case. It is not a
substitute for the multi-prompt quality suite.

## 8. Rejected experiments

Negative results are part of the benchmark record and must not be silently
reintroduced:

| Experiment | Result | Decision |
|---|---|---|
| Q4 Q8/DP4A verifier | 35.78 tok/s | Rejected; slower than exact verifier |
| Q4 verifier at 4/8 warps | Slower than retained 16 warps | Rejected |
| Q4 direct activation load | Neutral | Not retained as an improvement |
| Q4 seven-row specialization | 22.95 tok/s | Rejected |
| Q6 eight-row specialization | 44.31 tok/s | Rejected |
| Generic tiled Q5 verifier | 44.72 tok/s | Rejected |
| Ada-native `compute_89` PTX | Neutral | Portable `compute_70` retained |

Earlier vocabulary-prefix screens also showed that a smaller prefix can reduce
acceptance enough to lose performance. Draft cost and acceptance must be tuned
together.

## 9. llama.cpp comparison boundary

An earlier same-GPU record contains a llama.cpp commit `18f7ad7` `tg128`
reference of **51.26 +/- 0.18 tokens/second** using the same artifact. It was
useful evidence that the hardware/model combination could cross 50
tokens/second.

It is not a direct head-to-head result for this report:

- llama.cpp used its `tg128` benchmark shape;
- the retained XRT MTP workload generated 28 tokens from a specific prompt;
- command flags, cache behavior, warmup policy, and token path are not identical;
  and
- no common response-quality comparison is attached to the `tg128` result.

That comparison has now been rerun with pinned binaries, the exact model hash,
the same prompt and sampling controls, cache reuse disabled, three warmups, ten
measurements, and one shared OpenAI streaming client. The raw evidence and
aggregate manifest are retained in
[`qwen36-xrt-llama-paired-rtx4090-2026-08-09`](../benchmark-results/text/qwen36-xrt-llama-paired-rtx4090-2026-08-09/README.md).

| Runtime path | Native decode tok/s, mean +/- sample SD | Output tokens | MTP acceptance |
|---|---:|---:|---:|
| XRT target-only | 28.718 +/- 0.152 | 28 | - |
| XRT MTP | 45.480 +/- 0.218 | 28 | 24/32 (75.0%) |
| llama.cpp target-only | 50.476 +/- 0.087 | 64 | - |
| llama.cpp MTP | 143.132 +/- 0.481 | 64 | 54/67 (80.6%) |

This establishes that the earlier approximately 51 tok/s llama.cpp figure was
target-only-class performance, not llama.cpp MTP. On the paired request, XRT
MTP reached 90.10% of llama.cpp target-only native decode and 31.78% of current
llama.cpp MTP native decode.

The comparison is still not token-stream parity: XRT stopped after 28 output
tokens while llama.cpp reached the 64-token cap, and the previews differed.
Those facts are correctness and workload-duration blockers. The result may be
used to prioritize performance work, but not to claim response-quality parity
or a production-ready MTP path.

## 10. Reproducibility assessment

### Present in the retained artifact

- model filename, byte size, and SHA-256;
- base commit and dirty-tree flag;
- GPU, VRAM, driver, toolkit, provider, and region;
- prompt and generation controls;
- raw throughput samples;
- acceptance and verification counts;
- retained and rejected optimization descriptions; and
- correctness/admission gate states.

### Missing for a publication-grade rerun

- a clean candidate commit hash containing the exact code and generated PTX;
- the exact benchmark command and all environment variables;
- operating system/container image identity;
- compiler and Rust toolchain identity and complete build flags;
- CPU, host RAM, power/clock policy, and competing GPU-process state;
- explicit warmup count and measurement ordering;
- load, prefill, TTFT, end-to-end latency, transfer, and peak-memory samples for
  the final configuration; and
- retained complete token output or a privacy-safe digest tied to each run.

The current JSON is sufficient for the internal workload performance gate but
not for a third party to reproduce the result byte-for-byte. The next admitted
run must close these metadata gaps.

## 11. Gate disposition

| Gate | Status |
|---|---|
| 50 tok/s pinned-workload objective | Passed, five of five samples |
| Paired current-session 50 tok/s XRT objective | Failed; 45.480 tok/s mean |
| Same-request llama.cpp target comparison | Completed; XRT MTP is 9.90% lower |
| Same-request llama.cpp MTP comparison | Completed; XRT MTP is 68.23% lower |
| Cross-runtime token-stream parity | Failed; 28 versus 64 output tokens |
| Q4 verifier exactness | Passed |
| Q5 verifier exactness | Passed |
| Adaptive low-acceptance control | Passed |
| Portable PTX | Passed |
| Multi-prompt quality admission | Pending |
| Exact non-greedy rejection sampling | Pending |
| Clean-checkout reproducibility | Pending |
| Long-context/concurrency/reliability matrix | Pending |
| Production admission | Pending |

MTP remains experimental and disabled by default. Human quality review begins
only after the automated multi-prompt suite and non-greedy correctness work
produce an admissible candidate.

## 12. Allowed claims

### DeltaNet correction update

The later Q/K head-map correction supersedes the paired response-length and
acceptance figures above. On the same request shape, corrected XRT target and
MTP both reached the 64-token cap; MTP accepted 55/68 proposals. Corrected XRT
measured 30.785 +/- 0.052 tok/s target-only, 45.736 +/- 0.129 tok/s with
full-vocabulary MTP, and 50.418 +/- 0.255 tok/s with a target-verified 65,536-row
draft projection. The fresh pinned llama.cpp controls measured 50.627 +/- 0.027
tok/s target-only and 144.073 +/- 1.614 tok/s with MTP. See the
[`DeltaNet correction evidence`](../benchmark-results/text/qwen36-deltanet-qk-map-rtx4090-2026-08-09/README.md).

This update removes the 28-versus-64 output-length blocker for the retained XRT
target/MTP run. It does not establish complete cross-runtime token-id parity or
production admission, and it does not close the llama.cpp MTP throughput gap.

Allowed:

> On the pinned Qwen3.6-27B Q4_K_S greedy RTX 4090 workload, XRT's native
> experimental MTP path averaged 50.243 tok/s across five runs, 63.53% above
> its clean target-only mean.

Not allowed from this evidence:

- “XRT runs Qwen3.6-27B at 50 tok/s on every prompt.”
- “XRT is faster than llama.cpp.”
- “MTP is production-supported.”
- “Non-greedy sampling is lossless.”
- “The result is reproducible from the current public commit.”
