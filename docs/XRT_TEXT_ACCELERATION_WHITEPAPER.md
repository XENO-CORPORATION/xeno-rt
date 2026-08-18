# XRT Native Stable-Graph MTP Inference

- **Runtime domain:** `xrt-text`
- **Document status:** Engineering whitepaper candidate
- **Evidence date:** 2026-08-10
- **Claim maturity:** Registered workload breakthrough; production admission pending
- **Canonical architecture:** [Stable-Graph MTP Architecture](XRT_TEXT_MTP_STABLE_GRAPH_ARCHITECTURE.md)
- **Canonical evidence:** [RTX 4090 150 tok/s record](../benchmark-results/text/qwen36-stable-graph-150tps-rtx4090-2026-08-10/README.md)
- **Draft-head follow-up:** [RTX 4090 tensor-core screen](../benchmark-results/text/qwen36-mtp-draft-head-screen-rtx4090-2026-08-10/README.md)
- **Product boundary:** [XENO RT Runtime Domains](RUNTIME_DOMAINS.md)

## Abstract

XENO RT is an Apache-2.0 local AI runtime implemented in Rust and CUDA. It does
not embed or launch llama.cpp for model execution. This paper describes XRT's
native Multi-Token Prediction (MTP) path for the integrated NextN predictor in
a pinned Qwen3.6-27B Q4_K_S GGUF artifact.

On one 24 GB NVIDIA GeForce RTX 4090, XRT's retained 17-sample result averaged
**150.1710 decode tokens/second** with a 150.3018 median. The same-host
pre-change XRT control averaged 116.0165 tok/s, making the final path 29.44%
faster. A pinned llama.cpp control using the same model artifact, prompt,
greedy sampler, output cap, and depth-eight MTP averaged 144.0091 tok/s after
warmup. XRT was 4.28% faster on this registered tuple.

The result came from an architectural change, not a favorable counter or a
single kernel toggle: stable dual-topology CUDA graphs, device-resident dynamic
parameters, graph-compatible recurrent transactions, shared-page batched
attention, heterogeneous projection streams, graph-captured target decisions,
and batched host metadata publication.

A subsequent replacement-host profile found that the predictor's packed Q6_K
output head consumed 53.3% of draft GPU time. Reusing XRT's Q6_K tensor-core
verifier for the speculative single-row head reduced draft time 5.33% and
improved a 17-versus-17 same-host comparison from 149.2613 to **151.1889
tok/s** (+1.291%). This follow-up is opt-in and does not replace the original
same-host XRT/llama.cpp comparison.

This is a meaningful runtime-engineering breakthrough for XRT. It is not a
claim of a new speculative-decoding algorithm, a universal 150 tok/s product
guarantee, or general superiority over llama.cpp. The confidence interval
crosses 150 tok/s, six of 17 retained samples were below 150, and the result
covers one short greedy workload. MTP and verifier graphs remain experimental
and disabled by default.

## 1. Claim boundary

The strongest statement supported by the evidence is:

> XRT's native experimental stable-graph MTP path averaged 150.1710 decode
> tok/s on a pinned Qwen3.6-27B Q4_K_S greedy workload on one RTX 4090. It was
> 29.44% faster than the same-host pre-change XRT verifier and 4.28% faster than
> a same-host llama.cpp control on the registered request tuple.

The following statements are not yet supported:

- every RTX 4090 request sustains at least 150 tok/s;
- XRT is faster than llama.cpp across models, prompts, contexts, or samplers;
- MTP preserves the exact target distribution for every non-greedy sampler;
- the graph path is production-admitted or enabled by default; or
- the implementation constitutes a novel or patentable algorithm.

XRT's implementation is XENO-authored and XRT-native. The repository is
Apache-2.0 rather than proprietary in the exclusive-source sense. Qwen model
weights, GGUF, CUDA, Rust dependencies, and public speculative-decoding ideas
retain their own ownership and licenses.

## 2. Why MTP can accelerate batch-one decoding

Ordinary autoregressive decode evaluates the target model once per output
token. At batch size one, a 27B quantized model repeatedly streams a large
weight set to produce one token. The GPU is frequently limited by weight
traffic and launch dependencies rather than peak arithmetic throughput.

MTP changes the execution shape:

1. an integrated smaller predictor drafts several candidate tokens;
2. the target evaluates those candidates together as multiple activation rows;
3. a verified prefix is committed; and
4. state is rebased at the accepted boundary.

One target pass can therefore commit several tokens while each target weight
tile is resident. The speedup is real only when saved target passes exceed the
cost of drafting, batched verification, decision readback, and state rebase.
Acceptance rate and kernel shape are as important as raw target speed.

## 3. XRT architecture

```text
OpenAI-compatible API / CLI / language bindings
                         |
                    xrt-runtime
       admission | sampler | session | MTP transaction
          |             |                 |
       GGUF loader   CPU fallback    CUDA resident session
          |                               |
     xrt-models                   stable verify graph pair
                                          |
                   +----------------------+--------------------+
                   |                      |                    |
              DeltaNet state       shared-page F32 KV    packed weights
              transaction pair      pointer tables       Q4/Q5/Q6/F32
                   |                      |                    |
                   +---------- heterogeneous streams ---------+
                                          |
                              device argmax + bounded readback
```

The public API remains additive and OpenAI-compatible. GGUF loading and CPU
fallback remain intact. The acceleration lives behind model, backend, cache,
sampler, and environment eligibility checks.

### 3.1 Predictor and target ownership

The loader identifies the appended NextN predictor separately from the target
decoder blocks. The target and predictor share admitted embeddings and output
weights, but own distinct execution state. Predictor tokens are proposals;
only target-verified tokens become externally committed output.

The retained configuration drafts at most eight tokens and uses the first
65,536 output rows for the predictor's fast vocabulary projection. Target
verification always uses the complete 248,320-row output head. A restricted
draft can reduce acceptance, but cannot authorize an unverified target token.

### 3.2 Transactional hybrid state

Qwen3.6 combines full-attention and Gated DeltaNet layers. Verification must
advance both KV state and recurrent state through several speculative rows
without exposing a rejected suffix.

XRT holds paired committed and pending DeltaNet buffers. A verify window writes
device state into a stable pointer orientation. Only after the graph completes
successfully does the host swap the committed/pending handles. The parity of
the row count determines the resulting buffer generation.

That generation is part of the graph identity. XRT retains at most two graph
executables: one for generation zero and one for generation one. A graph is
never replayed against the opposite pointer orientation.

### 3.3 Stable device parameters

CUDA graph replay requires stable virtual addresses. Values that change each
window—start position, token metadata, and cache range—are uploaded into a
small persistent `CudaDecodeParams` allocation. Captured RoPE, KV append, and
attention kernels read those values on-device.

Q4 embeddings write directly into persistent verifier scratch. The graph does
not capture temporary host allocations or per-window replacement buffers.

### 3.4 Shared-page batched full attention

The session's actual F32 KV cache uses shared copy-on-write pages. The first
batched verifier path handled only the contiguous cache topology and silently
lost the intended batched execution when used with shared pages.

The retained architecture gives shared pages stable device pointer tables and
implements batched append plus causal attention over that topology. A physical
RTX test compares the batched shared-page result, output, keys, and values with
the serial pipeline.

### 3.5 Heterogeneous projection schedule

Qwen's layers contain independent projections whose resource profiles differ.
Executing all of them serially left portions of the Ada GPU idle. XRT now
schedules compatible Q4_K, Q5_K, and dense projections across the main stream
and dedicated projection streams, joining them with device events only at the
first true dependency.

The scheduler deliberately keeps gate and up projections separate. A grouped
gate/up experiment reduced launch count but lost useful overlap and was slower
end-to-end.

### 3.6 Graph-captured target decision

For the admitted unpenalized greedy path, the final output head and row-wise
first-argmax kernel are part of the verify graph. The host downloads only the
small vector of predicted token indices, compares it with the draft, and
commits the accepted boundary.

Sampling modes that need complete logits, penalties, logprobs, or unsupported
transforms retain the ordinary full-logit path. The optimization does not
weaken their contract.

### 3.7 Deferred and batched publication

Device work and host metadata have different lifetimes. Publishing host state
inside capture made the graph topology unstable and repeated work for every
row and layer.

The retained flow performs device mutations in the graph, then validates and
publishes recurrent handles after launch. Full-attention cache lengths are
committed once per affected layer for the whole row range. This reduced host
work from proportional to rows times layers to proportional to layers.

## 4. What produced the speedup

The controlled same-host progression was:

| Configuration | Retained mean | Increment |
|---|---:|---:|
| Pre-change eager verifier | 116.0165 tok/s | control |
| Stable dual-topology verifier graphs | 137.2741 tok/s | +18.32% |
| Recurrent heterogeneous projections | 144.7952 tok/s | +5.48% |
| Recurrent and attention heterogeneous projections | 148.4991 tok/s | +2.56% |
| Final 17-sample retained candidate | 150.1710 tok/s | +1.13% |
| Replacement-host tensor-core draft head | 151.1889 tok/s | +1.291% vs matched replacement-host control |

The final increment includes direct stable embeddings, graph-captured argmax,
immediate capture for the full row bucket, and batched cache publication.

The earlier path tried to update or recapture a graph containing thousands of
nodes for every window. Capture overhead erased the launch savings. Stable
allocation and state topology—not graph use by itself—was the central insight.

## 5. Evidence and statistics

The final run used 20 repetitions and discarded the first three as warmups.

| Statistic | XRT result |
|---|---:|
| Retained samples | 17 |
| Mean | 150.1710 tok/s |
| Median | 150.3018 tok/s |
| Sample standard deviation | 0.6554 tok/s |
| Normal 95% confidence interval | 149.8595-150.4826 tok/s |
| Minimum / maximum | 148.6076 / 151.1824 tok/s |
| Samples at or above 150 tok/s | 11 / 17 |

Every retained XRT run:

- produced 64 output tokens with no error;
- accepted 55 of 68 drafted tokens;
- used nine verification batches; and
- retained the same deterministic preview and phase shape.

The artifact was `Qwen3.6-27B-Q4_K_S.gguf`, 16,121,357,440 bytes, SHA-256
`a5ef62184c1729c38c9565b502303ac88e2fad3b1c3c6aa430d9e273bdd7f917`.

## 6. Same-host llama.cpp control

The llama.cpp control used the same RTX 4090, model file, prompt, greedy
sampling, 64-token limit, seed, and depth-eight MTP. Five requests retained
after warmup averaged 144.0091 tok/s. XRT's registered mean was 4.28% higher.

This paired result is useful because it proves that XRT is no longer merely
approaching an upstream feasibility number on this workload. It remains a
single-workload comparison. llama.cpp has a broader backend/model matrix, a
larger optimization community, and multiple speculative methods; no general
runtime ranking follows from this measurement.

The XRT code does not link to, invoke, or redistribute llama.cpp. Upstream
source and behavior were used as a diagnostic and benchmark reference under
their applicable license.

## 7. Correctness and failure containment

The acceleration is guarded by these invariants:

1. only a compatible dense hybrid Qwen tuple enters this batched verifier;
2. the cache must be F32 and graph-compatible;
3. every recurrent layer must support the fused transactional path;
4. graph identity includes row bucket and recurrent buffer generation;
5. changing values live in stable device allocations;
6. host state is published only after a successful launch;
7. graph capture or replay failure moves the session to eager fallback;
8. partial final windows can execute eagerly; and
9. unsupported sampling routes keep their full-logit target path.

Physical GPU tests establish that shared-page batched attention matches the
serial pipeline and that fused DeltaNet verification is bit-exact with its
serial kernels. The full workspace test suite, CUDA-feature build, benchmark
output, generated PTX hashes, and raw Nsight traces are retained with the
evidence record.

## 8. Current performance model

Across the 17 retained runs, the 63 timed decode tokens averaged 419.529 ms.

| Phase | Mean time | Decode share |
|---|---:|---:|
| Draft | 85.010 ms | 20.26% |
| Target verify | 319.495 ms | 76.16% |
| Rebase | 11.332 ms | 2.70% |
| Other measured overhead | 3.692 ms | 0.88% |

The retained structural trace shows a representative full verify window at
35.261 ms, with 32.520 ms of kernel-union time and 2.741 ms without an active
kernel in the profiled span. Marlin Q4_K kernels contributed 23.533 ms of
summed kernel time and remain the dominant target component.

A separate bounded draft trace attributes an eight-token predictor window as
follows: packed Q6_K output head 4.885 ms (53.3%), Marlin Q4_K projections
2.022 ms (22.1%), Q8_0 projections 0.729 ms (8.0%), RMSNorm 0.560 ms (6.1%),
and attention 0.460 ms (5.0%). Eight token readbacks occupied 7.673 ms of CUDA
API time because every next draft step depends on the previous argmax. The
trace is retained with the [draft-head evidence](../benchmark-results/text/qwen36-mtp-draft-head-screen-rtx4090-2026-08-10/README.md).

Two useful upper bounds follow:

- eliminating all draft and rebase time while leaving verification unchanged
  would cap this request near 197 tok/s; and
- eliminating every measured intra-window idle gap alone would put the request
  near 160 tok/s.

These are diagnostic ceilings, not forecasts. They show that another material
step requires target verification improvements, better proposal efficiency,
or both.

## 9. How XRT can go further

### 9.1 Retained shape and tensor-core draft head

Stable graphs and heterogeneous scheduling changed the cost curve. Depths and
draft vocabulary sizes were re-screened on the final architecture. Depth eight
with 65,536 draft rows remained best: smaller prefixes lost enough acceptance
to add target windows, larger prefixes added projection work, and depths 6,
10, and 12 were slower. The complete negative screen is
[registered here](../benchmark-results/text/qwen36-stable-graph-shape-screen-rtx4090-2026-08-10/README.md).

The Q6_K tensor-core draft-head candidate retained the same 55/68 acceptance
and target output while raising the 17-sample same-host mean from 149.2613 to
151.1889 tok/s. It remains opt-in through
`XRT_QWEN_MTP_Q6_TENSOR_CORE_HEAD=1` until the multi-prompt performance gate
is complete. Single-row Q8 MMQ reuse was measured and removed after regressing
to 142.6491 tok/s.

### 9.2 Frequency-ranked draft vocabulary

The current fast draft head uses the first N vocabulary rows. FR-Spec shows
that a frequency-ranked mapped subset can retain better draft coverage for the
same projection size. XRT can add a validated token-id map and gathered Q6_K
output rows while preserving complete target verification. This attacks part
of the 85 ms draft budget without changing committed target tokens.

See [FR-Spec](https://aclanthology.org/2025.acl-long.198/).

### 9.3 Q4 verifier roofline and SASS work

The next kernel effort must start with Nsight Compute counters: achieved memory
bandwidth, tensor-core utilization, occupancy, register pressure, stalls, and
L2 behavior for the exact Marlin row shapes. Candidate changes include offline
weight-layout specialization, deeper asynchronous copies, and epilogues that
remove material activation traffic. Each candidate needs a physical scalar or
serial oracle before the end-to-end screen.

### 9.4 Layer-local fusion without losing overlap

The trace still contains small normalization, conversion, activation, residual,
and event boundaries. Fusion is useful only where the saved traffic exceeds
the overlap it destroys. Gate/up grouping is already a negative control.

### 9.5 Device-resident transaction control

CUDA conditional graph nodes can represent device-selected branches and loops.
A future row-bucket graph could keep acceptance, boundary selection, and parts
of rebase on-device. This is a research path because cancellation, errors,
dynamic output limits, and state publication must remain observable and safe.

NVIDIA's current [CUDA Graph documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#conditional-graph-nodes)
defines the relevant constraints. Programmatic Dependent Launch is not an RTX
4090 optimization path because NVIDIA requires compute capability 9.0, while
the RTX 4090 is Ada compute capability 8.9.

### 9.6 Better drafters are a separate track

DFlash and tree or feature-level drafters can increase accepted tokens or
parallelize drafting, but require extra model artifacts, formats, memory, and
admission work. They should be evaluated as additive predictor backends, not
silently substituted into the integrated NextN result.

See the [DFlash paper](https://arxiv.org/abs/2602.06036) and the original
[multi-token prediction paper](https://arxiv.org/abs/2404.19737).

## 10. Product admission remains separate

Before MTP or stable verifier graphs become official production support, XRT
still needs:

- frozen multi-prompt greedy target/MTP token parity;
- exact target-distribution-preserving non-greedy speculation;
- long-context and cache-boundary coverage;
- cancellation and graph-failure recovery tests;
- concurrent request latency and throughput curves;
- memory-pressure and allocation-lifetime evidence;
- clean pinned-container reproduction on a second machine;
- cross-GPU and driver/toolkit coverage;
- API, security, packaging, installation, and upgrade verification; and
- human quality review after the automated gates pass.

Performance work must not weaken GGUF support, CPU fallback, OpenAI-compatible
contracts, or explicit unsupported-route behavior.

## 11. Conclusion

The 150.1710 tok/s registered comparison, followed by the 151.1889 tok/s
replacement-host draft-head candidate, is important because it demonstrates that XRT can
own the complete performance-critical path—from model state and packed kernels
through graph lifecycle and target decisions—and exceed a strong upstream
control on a carefully matched workload.

The breakthrough was architectural: make speculative verification a stable,
replayable device transaction rather than a sequence of temporary operations.
The next advance will require the same discipline. XRT should optimize the
measured 319 ms verification budget and roughly 79 ms optimized drafting budget, preserve exact
state and target behavior, register negative results, and expand the evidence
matrix before turning a benchmark milestone into a product promise.
