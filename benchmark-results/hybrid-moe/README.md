# Hybrid and MoE benchmark protocol

**Runtime domain:** `xrt-text`
**Canonical architecture:** [../../docs/RUNTIME_DOMAINS.md](../../docs/RUNTIME_DOMAINS.md)

This directory is the durable evidence boundary for
`docs/ktransformers-inspired-hybrid-moe-acceleration-spec.md`. Generated
Criterion HTML and `target/` data are intentionally not committed.

Retained evidence:

- `phase0-baseline-2026-07-20.json` records the pre-change repository and host
  baseline.
- `phase2-cpu-diagnostic-2026-07-20.json` records the post-worker-idling
  synthetic CPU result. It is useful engineering evidence, but it is not a
  real-model default-admission gate.
- `phase3-hybrid-diagnostic-2026-07-20.json` records forced CPU/GPU split,
  full-residency, pinned-staging, transfer-bound, and live RTX 4090 parity
  evidence. Its tiny-model timing is an overhead diagnostic and explicitly
  does not admit `auto`.
- `phase4-qwen35-hybrid-moe-diagnostic-2026-07-20.json` records the combined
  Qwen3.5 DeltaNet/full-attention plus exact hybrid-MoE CUDA path, including
  CPU parity, recurrent-state parity, expert-graph replay, and injected
  mid-token rollback. Adaptive placement and layerwise prefill remain rejected
  for this combined architecture pending dedicated gates.
- `phase5-qwen35-graph-diagnostic-2026-07-20.json` records alternating
  recurrent-buffer graph keys, device-local checkpoint parity, concurrent
  capture safety, accepted/rejected/cancelled speculation, synchronized
  checkpoint/rollback cost, and the live RTX 4090 eager/graph diagnostic. The
  synthetic graph speedup passes its shape-local gate but does not admit real
  Qwen3.5 models to `auto`.
- `phase5-moe-graph-diagnostic-2026-07-20.json` records exact resident-expert
  graph capture, placement/scratch epoch rejection, retained expert-slot
  lifetimes, central graph accounting, and the RTX 4090 eager/graph diagnostic.
  Full-GPU clears its synthetic gate; hybrid remains explicit-only after a
  5.01% median slowdown.
- `phase6-hybrid-prefix-diagnostic-2026-07-20.json` records paired KV/recurrent
  prefix identity, concurrent forks, eviction lifetime, split host/device
  accounting, F32 page-granular device COW, and the synthetic RTX 4090 TTFT
  diagnostic. Quantized/adaptive KV layouts do not claim page-level COW.
- `phase7-placement-layerwise-diagnostic-2026-07-20.json` records exact
  profiled-manifest validation, bounded adaptive update cost/churn, and
  double-buffered layerwise prefill. Layerwise clears its synthetic opt-in
  gate; adaptive regresses the recorded workload and therefore remains
  excluded from `auto`.
- `real-model-parity-2026-07-20.json` records the two-token licensed Qwen3 and
  Qwen3.5 CPU/CUDA smoke. Both fixtures preserve greedy argmax and clear the
  registered logit gates; Qwen3.5 also clears the recurrent-state tolerances.
  This is not the full quality gate.
- `qwen3-quality-harness-smoke-2026-07-21.json` records the executable
  CPU/hybrid-CUDA prompt and logical-route smoke. One short, one multi-turn, and
  one 174-token long-context case preserved all 10 greedy outputs, cleared the
  logit gates, and matched all 11,856 per-token/per-layer logical expert-route
  records without overflow.
- `qwen3-quality-harness-full-2026-07-21.json` records the passing full
  20/10/5 prompt profile at a 256-token long-context admission minimum, five
  256-token generations, 1,310 logit comparisons, and 171,264 route entries.
  Greedy outputs matched, mean/worst cosine was `1.0`, worst normalized RMS was
  `1.2629e-5`, and no route divergence occurred. The same artifact records the
  rejected 384/512/4096-token extended soaks.
- `qwen3-gsm8k-task-2026-07-21.json` records the paired, SHA-pinned 16-case
  GSM8K gate at the adjudicated 512-token output cap. Optimized CPU and exact
  hybrid CUDA each scored 15/16; every paired correctness outcome matched, so
  the 10,000-resample 95% interval for the score difference was `[0, 0]`.
- `qwen3-perplexity-gate-2026-07-21.json` records the passing SHA-pinned
  WikiText gate. With the registered F32 canonical expert-activation mode,
  CPU perplexity was `29.316061962` and hybrid CUDA was `29.316141694`, a
  relative change of `0.000002720` against the `0.001` limit. The separate
  production-mode diagnostic changed by `-2.70%` and remains visible rather
  than being substituted for the passing canonical result.
- `qwen3-perplexity-harness-open-2026-07-21.json` preserves the earlier timeout
  diagnosis. It is superseded by the passing artifact after the evaluator
  stopped tokenizing the complete 1.29 MB corpus before applying its token cap.
- `qwen3-xrt-llamacpp-comparison-2026-07-20.json` retains all measured samples,
  build/runtime flags, memory observations, bootstrap inputs, and limitations
  for the same-hardware external comparison.
- `qwen3-internal-admission-2026-07-20.json` records the required symmetric
  same-hardware comparison between the final exact hybrid path, its immediately
  preceding build, and XENO RT's optimized CPU path. The packed canonical merge
  is a measured improvement, but hybrid fails both registered `auto` bounds and
  therefore remains explicit-only.
- `qwen3-residual-q8-rejected-2026-07-20.json` records a later Q4_K
  residual-Q8 activation experiment, its bounded 16-token quality pass, two
  symmetric performance comparisons, and the rejected Q6_K variants. The raw
  repeat bounded candidate/previous median throughput at `0.972..0.998`, so
  the kernel, API, and runtime route were removed rather than shipping a
  measured regression.
- `qwen3-fused-silu-mul-rejected-2026-07-20.json` records an exact fused
  SiLU-times-up launch evaluated for resident MoE experts. Its device result was
  bit-identical to the retained two-launch path, but the real-Qwen3 `A-B-B-A`
  point estimate was `0.991x` baseline throughput and its `0.980..1.005`
  interval did not establish no-regression, so the fused kernel was removed.
- `qwen3-whole-layer-placement-rejected-2026-07-20.json` records an exact
  placement experiment that made eight complete layers GPU-resident under the
  same 4 GiB budget. Transfers fell, but its controlled throughput interval was
  `0.956..0.985x` uniform placement, so the policy and runtime path were removed.
- `qwen3-fragmented-layer-fallback-rejected-2026-07-21.json` records an exact
  attempt to reroute a layer's lone selected GPU-resident expert through the
  canonical CPU executor. Two-token parity passed, but the controlled
  throughput interval was `0.974..1.013x`, the p95-latency interval exceeded
  the registered ceiling, and H2D traffic rose by 3.98 MB per request. The
  threshold and runtime branch were removed.
- `qwen3-parallel-expert-graphs-rejected-2026-07-21.json` records isolated
  per-expert scratch and concurrent CUDA-stream replay of already captured
  resident-expert graphs. The controlled `A-B-B-A` throughput point was only
  `1.002x` with a `0.986..1.025` interval, while the p95-latency interval ended
  at `1.055` and scratch grew by 786,432 bytes. Parallel replay, larger/parent
  graph caches, and the temporary whole-layer branch were removed.
- `qwen3-q4k-cpu-order-kernel-2026-07-21.json` records the retained reuse of
  the existing 32-thread CPU-order Q4_K CUDA matvec for Qwen3 MoE resident
  matrices. Sixteen-token parity passed, and the controlled `A-B-B-A` result
  cleared the throughput and p95 no-regression bounds. Its `0.987..1.026`
  throughput interval does not establish superiority or admit `auto`.
- `qwen3-final-source-control-2026-07-21.json` records the final rebuilt CLI
  against the preserved pre-canonical-boundary control in a fresh `A-B-B-A`
  run. The current/control throughput interval was `0.984..1.001` and the
  p95-latency ratio interval was `0.960..1.023`, clearing the registered
  no-regression bounds without establishing a speedup or changing `auto`.
- `qwen3-aggregate-decode-graph-lifetime-2026-07-20.json` records the additive
  TTFT/decode benchmark fields, the concurrency-8 CUDA graph invalid-free
  reproducer, its session-lifetime repair, a final three-repetition live-GPU
  soak, and the registered single-stream `A-B-B-A` no-regression comparison.
  The repaired run reaches only 10.16 aggregate decode tok/s, so it explicitly
  rejects a hundreds-TPS claim for this model/hardware/path.
- `dense-protection-2026-07-20.json` records the same-hardware Qwen2.5 Q4_0
  baseline/candidate gate for dense CPU and CUDA decode. Both paths clear the
  pre-registered throughput and p95 latency no-regression intervals.
- `validation-2026-07-20.json` records default/CUDA workspace validation,
  the complete bounded safe-CUDA run, quality-harness build/smoke evidence,
  byte-identical PTX reproduction, and the pre-existing repository-wide
  formatting/Clippy blockers that remain open.

## Real Qwen3 result

The RTX 4090 comparison used the same pinned Qwen3-30B-A3B Q4_K_M file.
XENO RT ran exact uniform expert-slot residency with a 4 GiB expert budget.
llama.cpp ran three reference placements. Five measured runs were discarded
after each harness warmup and the following ten were retained; XENO was sampled
in three bracketing blocks.

| Runtime/configuration | Total-device reading | Median prefill | Median single-stream decode |
|---|---:|---:|---:|
| llama.cpp full GPU | 22,576 MiB | 637.38 tok/s | 185.52 tok/s |
| XENO RT uniform hybrid | 9,723 MiB | 9.82 tok/s | 11.32 tok/s |
| llama.cpp `-ncmoe 38` | 10,002 MiB | 63.89 tok/s | 8.02 tok/s |
| llama.cpp `-ncmoe 48` | 6,495 MiB | 41.38 tok/s | 6.48 tok/s |

At approximately matched total device use, XENO RT's median decode was 41.26%
higher than the pinned llama.cpp layer-offload point. The independent
10,000-resample bootstrap interval for the median ratio was `1.360` to `1.557`.
The result is diagnostic: the run order was bracketing `A-B-A`, not the
preferred `ABBA`, XENO used a real prompt while `llama-bench` used synthetic
tokens, and the offload strategies differ.

The comparison also identifies the primary performance debt. llama.cpp's
full-GPU reference was 18.70 times the bracketed XENO hybrid median, and even
the VRAM-matched llama.cpp prefill microbenchmark was 6.51 times faster. XENO
does not claim hundreds-TPS from this result and remains excluded from
real-model `auto` admission.

## Internal Qwen3 admission result

The final admission run used a symmetric previous/final/CPU/CPU/final/previous
order on the same Qwen3 fixture and machine. Each block had five discarded
warmups and ten retained samples. Prefix reuse and speculation were disabled;
hybrid used exact uniform placement, CUDA graphs, and a 4 GiB expert budget.

| XENO path | Median single-stream decode | P95 decode latency |
|---|---:|---:|
| Previous grouped-row hybrid | 17.30 tok/s | 420.72 ms |
| Final packed-merge hybrid | 17.61 tok/s | 426.70 ms |
| Optimized CPU | 18.02 tok/s | 408.33 ms |

The final merge is 1.81% faster by median than the previous hybrid build; its
paired-bootstrap ratio interval is `1.003` to `1.034`. Against optimized CPU,
however, hybrid reaches only `0.977x` median throughput (95% interval `0.944`
to `0.988`). Its p95 latency ratio is `1.045` with an interval of `0.989` to
`1.106`. The registered `auto` gate requires a throughput lower bound of at
least `1.15` and a p95 upper bound no greater than `1.05`, so both conditions
fail. The result closes the missing ordered comparison; it does not enable a
default.

A 30-repetition adaptive-placement follow-up also failed its own gate. With
one-token evaluation intervals and bounded four-expert updates, repetitions
26-30 reached a median 17.12 tok/s. The final request reduced host-to-device
traffic from 76,129,152 to 69,985,152 bytes, but remained slower than both the
17.61 tok/s fixed-uniform result and the 18.02 tok/s optimized CPU result.
Adaptive placement therefore remains experimental for this workload; aggregate
expert counts were not misused to fabricate the layer-specific profiled
manifest required by the runtime.

## Aggregate decode and graph-lifetime result

The benchmark JSON now retains the old `prefill_ms` and `tok_s` fields while
adding explicit `ttft_ms`, `decode_tokens`, `decode_ms`, and `decode_tok_s`.
Concurrent local and external workers use one published start epoch, so their
durations are comparable. Aggregate decode is defined as all post-first tokens
divided by the window from the earliest first token to the latest completion;
it is not relabeled single-stream throughput.

A Qwen3 graph-on run with concurrency 8 and 32 output tokens per request exposed
a real lifetime bug: one worker could destroy session buffers with
`cuMemFreeAsync` while another worker captured the shared CUDA stream. Graph-off
completed, while graph-on failed with `CUDA_ERROR_INVALID_VALUE`. The retained
repair gates the full standard-MoE graph execution window and holds both graph
gates through actual session field destruction, not only through a preceding
synchronization call.

The final rebuilt binary completed three repetitions (768 output tokens) with
no errors or panics. Aggregate decode measured 10.11, 10.16, and 10.17 tok/s.
The matching single-stream `A-B-B-A` check measured 17.87 tok/s for the repair
versus 17.99 tok/s for the retained baseline; the paired-bootstrap ratio was
`0.993` with a 95% interval of `0.983..1.010`, clearing the 2% throughput
no-regression floor. The p95 point ratio was `1.004`, but its `0.926..1.051`
interval is marginally inconclusive against the separate 5% auto-admission
latency limit. Hybrid remains explicit-only. Most importantly, 10.16 aggregate
decode tok/s is only about one tenth of the `>=100` stretch objective.

## Rejected residual-Q8 follow-up

A two-pass residual-Q8 activation representation was evaluated for Q4_K gate
and up projections. It preserved all 16 Qwen3 greedy tokens and cleared the
bounded logit gates with worst cosine `0.999999999` and normalized RMS error
`0.000040775`. The corresponding one-pass representation failed quality, and
both raw and repacked Q6_K extensions were slower in preliminary diagnostics.

Quality was not enough to admit the two-pass candidate. In the retained raw
symmetric repeat, the existing packed-F32 hybrid path reached 18.19 tok/s,
the candidate reached 17.90 tok/s, and optimized CPU reached 18.62 tok/s. The
paired-bootstrap candidate/previous throughput ratio was `0.984`, with a 95%
interval of `0.972` to `0.998`; candidate/CPU was `0.961`, with an interval of
`0.934` to `0.990`. The experiment was therefore removed from the runtime and
CI surface. It remains only as durable negative evidence, and `auto` remains
disabled.

## Rejected fused SiLU-times-up follow-up

The next bounded experiment fused the resident expert's in-place SiLU and
elementwise multiply into one CUDA launch without changing the F32 instruction
sequence. A dedicated device test confirmed bit-identical output. The pinned
Qwen3 `A-B-B-A` comparison nevertheless measured 17.88 tok/s for the candidate
versus 18.04 tok/s for the retained path. The candidate/baseline median
throughput ratio was `0.991`, with a 95% paired-bootstrap interval of `0.980` to
`1.005`; the p95 latency interval also crossed parity. The kernel, CUDA API, and
runtime route were removed, and the exact two-launch implementation remains.

## Rejected whole-layer placement follow-up

An explicit whole-layer layout used the 4 GiB expert budget for all 128 experts
in layers 0, 6, 12, 18, 24, 30, 36, and 42. The other 40 layers ran all selected
experts through the grouped CPU path. Two-token CPU/CUDA parity passed, and the
layout reduced per-request H2D traffic from 76.13 MB to 73.40 MB and D2H traffic
from 16.56 MB to 14.73 MB.

The lower transfer volume did not translate into faster decode. In the retained
same-binary `A-B-B-A` comparison, uniform placement reached 17.97 tok/s and the
whole-layer layout reached 17.44 tok/s. The paired-bootstrap throughput ratio
was `0.970`, with a 95% interval of `0.956` to `0.985`; p95 latency rose from
419.61 ms to 428.43 ms. The policy was removed. This result indicates that the
current selected-expert GPU kernels must improve before concentrating all eight
routed experts onto a GPU layer can pay off.

## Rejected fragmented-layer CPU fallback follow-up

The next exact experiment kept uniform placement but routed a layer's lone
selected GPU-resident expert back through the canonical CPU executor. This was
intended to avoid a CUDA expert launch that could not amortize heterogeneous
coordination. Real Qwen3 two-token CPU/CUDA parity passed.

The candidate did not remove the per-layer activation download and added 3.98
MB of CPU-result upload traffic per request. In the same-binary `A-B-B-A` gate,
the retained path reached 17.75 tok/s and the fallback reached 17.64 tok/s. The
candidate/retained median-throughput ratio was `0.994`, with a 95% interval of
`0.974..1.013`; its p95-latency ratio was `1.026`, with an interval of
`0.992..1.055`. Both no-regression gates failed, so the environment knob,
threshold, and runtime branch were removed. Fragmentation must be attacked by
removing transfer/launch overhead or batching selected-expert work, not by this
CPU reroute.

## Retained CPU-order Q4_K kernel follow-up

The existing CUDA source already contained a 32-thread Q4_K matvec that mirrors
the CPU AVX2/FMA accumulation chains for recurrent Qwen3.5 projections. Reusing
that exact kernel for Qwen3 MoE resident matrices avoided adding another CUDA
implementation. Dense Qwen3, Qwen2, and Llama remain on their previous Q4_K
dispatch; a unit test locks this architecture boundary.

The real-Qwen3 `A-B-B-A` comparison measured 18.03 tok/s for the candidate and
17.87 tok/s for the prior path. The candidate/baseline throughput ratio was
`1.009`, with a 95% paired-bootstrap interval of `0.987..1.026`; the p95
latency-ratio interval was `0.939..1.037`. Both registered no-regression bounds
pass, while decode superiority is not established because the throughput
interval crosses parity. Median prefill improved from 1387.09 ms to 1375.98 ms,
with a ratio interval of `0.986..0.998`.

A scoped 16-token CPU/CUDA run preserved every greedy token with worst
normalized RMS logit error `0.000002999`. The final hash-pinned source also
passed the two-token parity test, and its last ten confirmation samples reached
18.38 tok/s median. That unpaired confirmation is not used as a speedup claim.
The controlled candidate is approximately at the 18.02 tok/s optimized-CPU
reference and nowhere near the registered `1.15x` lower-confidence `auto` gate,
so hybrid remains explicit-only.

## Dense protection result

The licensed dense fixture is official Qwen2.5-0.5B-Instruct Q4_0 at pinned
revision `872f8a96064a1242ac3a3359cad77c3042548405`. Baseline and candidate
binaries used identical external dependency versions. Each backend/version had
five warmups and ten retained samples in `A-B-B-A` blocks. Prefix reuse and
speculation were disabled.

| Backend | Baseline median decode | Candidate median decode | Candidate/baseline 95% CI | P95 latency ratio 95% CI | Gate |
|---|---:|---:|---:|---:|---:|
| CPU, 256 output tokens | 53.30 tok/s | 83.06 tok/s | 1.162 to 1.807 | 0.572 to 0.684 | Pass |
| CUDA resident, 128 output tokens | 91.31 tok/s | 105.02 tok/s | 1.137 to 1.165 | 0.878 to 0.919 | Pass |

This closes the representative dense CPU/CUDA protection criterion. It does not
admit an MoE acceleration mode to `auto`.

## Pinned comparisons

- XENO RT source baseline: `e1bb2e67fa4a2cf6ac399a8bbaee34e9d20de2e2`.
- llama.cpp: `178a6c44937154dc4c4eff0d166f4a044c4fceba`.
- KTransformers: `d1a3ed8a308cf45a2bdf8dc0ec18ea0cf782486c`.
- KTransformers' audited SGLang integration:
  `1e098a77ba395dc1a5f2dcbdf57bdb188e84bcee`.
- lm-evaluation-harness:
  `f4d4b3de3ee6741a7151a9fe74945ee515262f4c`.
- Dense protection fixture: Qwen/Qwen2.5-0.5B-Instruct-GGUF revision
  `872f8a96064a1242ac3a3359cad77c3042548405`.

The KTransformers/SGLang comparison is explicitly omitted from the Phase 0
Windows baseline. The pinned stack requires a separately provisioned Linux
environment and an original-format checkpoint; an upstream support label is
not treated as a successful XENO smoke test.

## Reproduction

Run correctness first:

```powershell
cargo test -p xrt-kernels --lib
cargo test -p xrt-models --lib
cargo test -p xrt-runtime --lib
cargo test -p xrt-workspace-tests --test hybrid_session_state
cargo test -p xrt-workspace-tests --test moe_execution
cargo test -p xrt-workspace-tests --test moe_allocation
```

Run microbenchmarks with at least five warmups and ten samples for a retained
result. The short Phase 0 development samples are marked diagnostic and cannot
admit a default path:

```powershell
cargo bench -p xrt-workspace-tests --bench hybrid_state_bench -- --noplot
cargo bench -p xrt-workspace-tests --bench moe_bench -- --noplot
cargo bench -p xrt-workspace-tests --bench inference_bench -- --noplot
cargo bench -p xrt-workspace-tests --features cuda --bench qwen35_cuda_bench -- --noplot
```

Real-model runs must alternate legacy and candidate modes in ABBA order on the
same boot and power profile. Use a seeded paired bootstrap with 10,000
resamples for the 95% confidence interval. Populate `result-template.json`;
do not report prefill or aggregate throughput as single-stream decode.

The llama.cpp comparison is built from the pinned source:

```powershell
cmake -S . -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
cmake --build build --config Release -j
build\bin\Release\llama-bench.exe -m <fixture.gguf> -p 512 -n 128 -ngl 0
build\bin\Release\llama-bench.exe -m <fixture.gguf> -p 512 -n 128 -ngl 999
```

Record compiler, CUDA toolkit, all runtime flags, tokenizer identity, context,
batch, concurrency, RSS, VRAM, and the model revision/SHA from `fixtures.json`.
