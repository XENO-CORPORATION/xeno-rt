# XRT Text Acceleration Roadmap

- **Runtime domain:** `xrt-text`
- **Status:** Internal engineering roadmap
- **Updated:** 2026-08-10
- **Starting evidence:** [Qwen3.6-27B MTP benchmark report](XRT_TEXT_MTP_BENCHMARK_REPORT_2026-08-09.md)
- **Product admission rules:** [XENO RT Runtime Domains](RUNTIME_DOMAINS.md#support-and-admission)

## 1. Objective

Turn the retained 150.1710 tokens/second single-workload result into a correct,
reproducible, and product-admissible XRT acceleration system.

The roadmap does not set “50 tok/s” as a universal model promise. It defines
separate performance objectives for named model, quantization, GPU, prompt,
context, sampling, and concurrency tuples.

Priority order is fixed:

1. target-distribution correctness;
2. reproducibility and observability;
3. single-stream decode performance;
4. persistent-agent latency and cache reuse;
5. server throughput and concurrency;
6. broader model and hardware coverage; and
7. production admission and default enablement.

No performance improvement may weaken GGUF support, OpenAI-compatible
contracts, CPU fallback, cancellation, memory safety, or explicit unsupported
behavior.

## 2. Current measured baseline

The controlled progression is:

| Configuration | Mean decode |
|---|---:|
| Clean target-only | 30.723 tok/s |
| Depth-eight MTP, 74,752 draft rows | 50.243 tok/s |
| Corrected compact verifier | 114.2518 tok/s |
| Batched rebase plus Marlin occupancy | 118.8529 tok/s |
| Stable verifier graph plus heterogeneous projections | 150.1710 tok/s |
| Replacement-host tensor-core draft-head candidate | 151.1889 tok/s |

A later replacement-host A/B retained correctly provisioned three-stage
Marlin over the two-stage control (111.0078 versus 110.8178 tok/s, +0.1715%).
Because absolute host performance differed, this does not replace or
extrapolate the 118.8529 tok/s record. It only admits the stage-depth change.
The scheduler evidence is
[`qwen36-marlin-stage-depth-screen-rtx4090-2026-08-10`](../benchmark-results/text/qwen36-marlin-stage-depth-screen-rtx4090-2026-08-10/README.md).

The current baseline uses an RTX 4090, a pinned Qwen3.6-27B Q4_K_S artifact,
greedy sampling, one short prompt, one stream, F32 KV, no prefix cache, and
portable PTX. Stable dual-topology verifier graphs, shared-page batched
attention, heterogeneous projection streams, graph-captured argmax, deferred
host state publication, and batched layer cache commits raised the same-host
control from 116.0165 to 150.1710 tok/s. This passes the narrow 150 tok/s mean
objective, but its confidence interval crosses 150 and it does not pass product
admission. The retained evidence is
[`qwen36-stable-graph-150tps-rtx4090-2026-08-10`](../benchmark-results/text/qwen36-stable-graph-150tps-rtx4090-2026-08-10/README.md).

The follow-up draft trace found the packed Q6_K output head at 53.3% of draft
GPU time. A matched replacement-host experiment reused the Q6_K WMMA verifier
for speculative single-row output and improved 149.2613 to 151.1889 tok/s
across 17 retained samples per arm. It remains opt-in pending multi-prompt
coverage; see the [draft-head evidence](../benchmark-results/text/qwen36-mtp-draft-head-screen-rtx4090-2026-08-10/README.md).

## 3. Phase A — close correctness and evidence gaps

### A1. Multi-prompt greedy parity harness

Build a frozen corpus containing at minimum:

- structured counting and formatting;
- code generation and code continuation;
- open-ended prose;
- tool-call JSON and grammar-constrained output;
- repetition-prone prompts;
- low-acceptance prompts;
- short and long system prompts;
- multi-turn tool-schema reuse; and
- context-length buckets from short prompts through the admitted maximum.

For every case, compare target-only and MTP token IDs, stopping behavior,
sampler transformations, final text, and target/predictor state boundaries.

**Exit gate:** all supported greedy settings are token-identical, every mismatch
has a resolved cause, and the corpus plus runner is versioned.

### A2. Exact non-greedy speculative sampling

Implement target-distribution-preserving accept/reject behavior for temperature,
top-k, top-p, repetition, presence, and frequency penalties. Keep unsupported
combinations on target-only decode until their exact contract is proven.

Tests must include:

- deterministic seeded replay;
- forced acceptance and forced rejection cases;
- EOS inside a proposal window;
- rollback after partial acceptance;
- cancellation during draft and verify;
- target/predictor KV and recurrent-state boundary checks; and
- statistical comparison with target-only sampling over a frozen corpus.

**Exit gate:** no unsupported sampler enters MTP, exact unit/integration cases
pass, and the frozen statistical gate shows no unexplained distribution drift.

### A3. Clean benchmark candidate

Commit the exact source and generated PTX, then rerun from a clean checkout in
a pinned container. Record the exact command, environment variables, model
hash, toolchains, hardware, clocks/power policy, competing utilization,
warmups, repetitions, raw outputs, TTFT, prefill, decode, end-to-end latency,
transfers, and peak RAM/VRAM.

Use interleaved target/MTP blocks to reduce drift. Report median, p10/p90 or
confidence interval, every raw sample, and failures rather than only the best
run.

**Exit gate:** a second clean environment reproduces the registered result
within the documented variance budget.

## 4. Phase B — reduce exact verification cost

The depth-eight profile attributes roughly 478-480 ms to verify versus about
42 ms to draft and 11 ms to rebase. Q4 verification accounts for 310.066 ms
across 1,700 profiled kernel instances. Verification is the first performance
target.

### B1. Establish verify microbenchmarks

Add exact per-layer and end-to-end microbenchmarks for Q4_K, Q5_K, and Q6_K at
proposal rows 2-16. Record:

- effective weight bandwidth;
- activation rows per weight read;
- kernel launches and launch gaps;
- occupancy, registers, shared memory, and spills;
- device/host transfers;
- arithmetic exactness; and
- accepted tokens per unit of verify time.

**Exit gate:** the profile can attribute at least 95% of verification GPU time
to named kernels or synchronization boundaries.

### B2. Exact kernel fusion

Screen, one at a time:

- fused normalization plus projection where it removes material traffic;
- fused gate/up activation paths;
- projection epilogues that retain logits or hidden rows device-side;
- layer-local fusion that reduces launches without changing arithmetic order;
- reusable activation tiles across Q4/Q5/Q6 projections; and
- persistent or grouped verify execution for repeated layer shapes.

Every candidate needs a bit-exact or admitted-tolerance test before the
end-to-end benchmark. Retain negative results with compiler/PTX identity.

The 2026-08-10 scheduler screen exhausted the local Marlin parameter class:
two, three, and four stages; every upstream small-batch K/N tile; exact shared
memory; and concatenated gate/up. Only three stages produced a repeatable gain,
and it was 0.1715%. Further tile or stage sweeps are not a credible route to
150 tok/s. The next experiment must cross a layer or projection boundary and
remove measurable weight traffic or launches; a mere repacking or larger
single projection is insufficient because it loses existing stream overlap.
The subsequent retained heterogeneous schedule preserved that overlap across
recurrent and full-attention projections and contributed the final step from
137.2741 tok/s with stable graphs alone to 150.1710 tok/s.

**Exit gate:** at least a 10% reduction in verification time on the registered
workload with unchanged token output, no target-only regression above policy,
and no VRAM-budget violation.

### B3. CUDA graph and launch strategy

Measure graph capture/replay for stable verify row buckets and the complete
draft-verify-rebase transaction. Do not assume a graph helps; compare total
launch gaps, capture constraints, pointer stability, cancellation behavior,
and fallback cost.

**Exit gate:** retain only a graph path that improves p50 and does not worsen
p95, cancellation, dynamic-depth fallback, or memory ownership.

**2026-08-10 result:** the performance half of this gate is met for the
registered greedy workload. Two stable graphs keyed by recurrent buffer
generation replaced per-window recapture, while partial windows use eager
execution. The controlled screen improved 116.0165 to 137.2741 tok/s before
projection scheduling and the final combined path reached 150.1710 tok/s.
Cancellation, dynamic-depth, long-context, concurrency, and memory-pressure
coverage remain required before this gate is product-complete.

### B4. Device-resident verification decisions

Keep target sampling transforms, proposal comparison, accepted-length
reduction, EOS detection, and next-state selection on device where doing so
preserves exact behavior. Return bounded summaries rather than full logits
when the API does not request them.

**Exit gate:** lower transfer/synchronization cost with exact sampler parity and
no loss of logprobs behavior on advertised routes.

## 5. Phase C — adaptive speculation policy

One fixed depth cannot be optimal for prose, code, tool JSON, long context, and
concurrent serving.

### C1. Online cost model

Estimate whether another draft token is profitable using:

- rolling acceptance by depth;
- measured predictor cost;
- measured target verify cost by row bucket;
- remaining output budget;
- prompt/task signals that do not inspect private text in telemetry;
- active batch/concurrency; and
- current memory pressure.

The policy chooses depth zero through the admitted maximum. Depth zero means
ordinary target decode, not an error.

**Exit gate:** the adaptive policy is never more than 2% slower than the better
of registered target-only and static-depth choices at p50 across the frozen
workload suite, with explicit exceptions reviewed at p95.

### C2. Safe predictor vocabulary policy

Evaluate vocabulary-prefix sizes by draft cost and acceptance, not projection
cost alone. Target verification remains complete. If a request's accepted
tokens repeatedly fall outside the prefix, expand or disable the prefix within
a bounded policy.

**Exit gate:** the selected policy improves geometric-mean decode without
changing any committed target token.

### C3. Concurrency-aware MTP

As batch size grows, verifying `batch * depth` rows can become more expensive
than normal decode. Reduce depth or disable MTP based on registered throughput
and latency curves.

**Exit gate:** MTP does not reduce aggregate throughput or violate per-request
p95 latency budgets at admitted concurrency levels.

## 6. Phase D — persistent-agent acceleration

Single-turn decode speed is only one XENO product metric. Persistent agents
repeatedly reuse system prompts, tool schemas, plans, and recent tool results.

### D1. Predictor-aware prefix cache

Extend prefix snapshots to include synchronized predictor attention and any
recurrent state required to resume MTP exactly. Reject incompatible snapshots
by model, quantization, backend, cache mode, predictor identity, and topology.

**Exit gate:** warm tool-call turns enter MTP without re-prefilling the shared
prefix and match a cold target-only reference.

### D2. Session cache persistence

Design a versioned, integrity-checked cache format for restart-safe session
state. Separate RAM and SSD budgets, use atomic writes, bind state to immutable
model/build identities, encrypt or avoid sensitive prompt-derived state as the
security policy requires, and clean expired entries.

**Exit gate:** a long registered session restores materially faster than
prefill, with exact continuation, bounded disk use, corruption recovery, and
explicit user/operator controls.

### D3. Agent-adaptive KV policy

Coordinate MTP with the existing agent-adaptive direction:

- pin system/developer instructions and active tool schemas;
- protect recent tool results;
- compress lower-priority cold context only through admitted KV modes;
- make rollback work across hot and compressed tiers; and
- report quality, memory, TTFT, and decode separately.

**Exit gate:** lower long-session memory or restore latency without reducing
tool-call correctness on the registered agent suite.

## 7. Phase E — model, hardware, and server matrix

### E1. Model/quantization tuples

Admit each tuple independently. Initial candidates:

- the pinned Qwen3.6-27B Q4_K_S artifact;
- compatible Q4_K_M/Q5_K_M variants when memory permits;
- smaller dense Qwen models for consumer hardware; and
- hybrid/MoE models only after their recurrent and routing state contracts
  pass the same rollback gates.

Do not infer MTP support from tensor names or file presence.

### E2. Hardware tiers

Maintain a portable PTX baseline and add device-specific kernels only when
they beat it reproducibly. The first matrix should cover:

- RTX 4090, 24 GB;
- one prior-generation 24 GB NVIDIA GPU;
- one datacenter NVIDIA GPU used by CI or RunPod; and
- CPU target-only fallback regression coverage.

AMD, Apple, or other accelerators require real backend work and independent
admission; CUDA results do not imply support.

### E3. Server behavior

Measure:

- single-stream TTFT and inter-token latency;
- aggregate throughput at admitted concurrency;
- continuous batching interactions;
- prefix-hit and prefix-miss paths;
- cancellation and client disconnect during each phase;
- queue saturation and overload behavior;
- load/unload and fragmentation soaks; and
- mixed text/image resource admission once `xrt-image` is ready for that gate.

**Exit gate:** the OpenAI-compatible routes retain schemas, usage, streaming,
errors, tools, and logprobs behavior while MTP is enabled internally.

## 8. Performance gates

All thresholds are tuple-specific and must be registered before measurement.
For the pinned RTX 4090/Qwen3.6-27B Q4_K_S tuple, proposed gates are:

| Gate | Proposed requirement |
|---|---|
| Greedy correctness | Token-identical target/MTP output across frozen suite |
| Non-greedy correctness | Exact algorithm plus seeded and statistical gates |
| Single-stream decode | Median at least 50 tok/s on the registered long-output suite, not only one prompt |
| Regression floor | No registered workload more than 2% slower than target-only after adaptive fallback, unless explicitly reviewed |
| Stability | At least 10 measured runs per workload after warmup; publish distribution and failures |
| Memory | Stay within the reserve-aware 24 GB profile with no leak or OOM |
| TTFT | No unexplained p95 regression above the registered budget |
| Concurrency | No aggregate-throughput loss at admitted request counts |
| Reliability | Cancellation, disconnect, unload, and long soak pass |
| Reproducibility | Clean pinned build reproduced in a second environment |

A higher stretch target, such as 60+ tok/s, is pursued only after the 50 tok/s
long-output suite and quality gates pass. It is an optimization target, not a
release promise.

## 9. Comparison protocol

Compare XRT with llama.cpp, vLLM, SGLang, or TensorRT-LLM only when the
comparison is meaningful for the runtime's supported artifact path.

Every report must pin:

- upstream commit/container and dependency versions;
- identical model bytes or a clearly disclosed format/quantization difference;
- identical prompt corpus and tokenizer behavior;
- context and output token counts;
- sampler, penalties, seed, stop conditions, and chat template;
- cache, flash-attention, offload, graph, and speculative settings;
- warmup, repetition, concurrency, and measurement boundaries;
- hardware, clocks/power policy, drivers, and system load; and
- correctness/quality evidence.

Report TTFT, prefill, decode, end-to-end latency, memory, power when available,
acceptance, and aggregate throughput separately. Never compare a batched
aggregate number with a single-stream number under one “tok/s” label.

## 10. Human review gate

Human review is useful only after automation creates a stable candidate.
Request human review when all of the following are ready:

1. the frozen multi-prompt suite passes target/MTP correctness;
2. non-greedy speculative sampling passes its exactness gates;
3. complete outputs are packaged blindly with target-only controls;
4. regressions, repetitions, and suspicious outputs are automatically flagged;
5. the reviewer rubric covers factuality, instruction following, code validity,
   tool-call correctness, repetition, truncation, and style; and
6. the candidate build and model identities are immutable.

Before that point, human review may find examples but cannot close admission.

## 11. Documentation and release gates

Before enabling MTP by default:

- update [Supported Models](SUPPORTED_MODELS.md) with exact admitted tuples;
- document configuration and fallback behavior;
- document benchmark commands and expected resource budgets;
- add API conformance and packaging evidence;
- update the canonical [Roadmap](ROADMAP.md), changelog, and release notes;
- rerun CPU and CUDA gates from a clean checkout; and
- follow `release-guide/` through the XENO platform release process.

No roadmap checkbox, benchmark file, or successful smoke test alone changes
the support contract.

## 12. Immediate execution queue

1. Freeze the multi-prompt greedy corpus and output-digest format.
2. Implement and test exact non-greedy rejection sampling.
3. Create a clean reproducible benchmark wrapper that records missing metadata.
4. Add a frequency-ranked mapped draft vocabulary behind an audit flag.
5. Prototype device-resident draft argmax-to-embedding handoff.
6. Add verify row-bucket roofline microbenchmarks and complete launch attribution.
7. Prototype a persistent cross-layer Q4 verifier with an explicit 10% verify-time exit gate.
8. Implement an acceptance-and-cost-based adaptive depth policy.
9. Add predictor-aware prefix snapshots and tool-turn benchmarks.
10. Package automated outputs for human review, then run reliability, server,
    security, packaging, and release admission.
