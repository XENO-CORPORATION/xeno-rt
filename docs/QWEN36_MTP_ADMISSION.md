# Qwen3.6 NextN/MTP admission

Status: experimental, disabled by default.

## Production qualification result (2026-08-11)

The pinned Qwen3.6-27B Q4_K_S target plus the Qwen3.6 DFlash Q8_0 draft was
exercised on one RTX 4090 with the final serial-projection production profile.
The candidate is **not production-admitted**. The complete qualification
report and raw evidence are in
[`qwen36-production-rtx4090-2026-08-11/admission-v2-final`](../benchmark-results/text/qwen36-production-rtx4090-2026-08-11/admission-v2-final/README.md).

The deterministic throughput corpus passed exact target/candidate token parity
for all 12 cases and 36 measured samples per arm. Target-only decode averaged
32.5407 tok/s; DFlash averaged 126.0533 tok/s when cases were equally weighted,
with a 105.6976 median and 93.4740 tok/s token-weighted aggregate. The very wide
38.6308--356.2281 tok/s range is workload dependent and does not support a
per-request throughput guarantee.

F32 KV-cache testing passed exact retrieval and exact target/candidate token
parity through 7,776 actual prompt tokens. Multi-turn correctness, deterministic
sampling fallback, concurrency-one and concurrency-two runs, real CPU fallback,
physical CUDA kernel checks, and the OpenAI-compatible service suite also
passed. The service suite covered streaming and non-streaming requests,
cancellation and drain, bounded overload with 429 responses, a 100-request
soak, unload/reload, and one 7,773-token API request. Prefix caching was enabled,
but that service run recorded no cache hits; cache effectiveness is therefore
not admitted by this result.

Admission remains blocked by the following evidence:

- The final non-thinking quality suite failed its arithmetic case in both the
  target and DFlash arms, returning `218.4` instead of `259.2` in all three
  repetitions. This preserves target parity but fails product quality. The
  separate thinking-enabled API case returned `259.2` and stopped cleanly.
- Qwen3.6 hybrid CUDA execution rejected both `q8` and `kq4_vq8` KV-cache
  modes. Q4_K_S *model weights* are supported; quantized KV-cache support must
  not be inferred from that fact.
- Strict Clippy is not clean, and workspace-wide Rust 1.76 verification is
  blocked by `xrt-python` declaring Rust 1.83 even though the text runtime,
  CLI, and server packages pass their Rust 1.76 checks.
- This is one model tuple on one RTX 4090, not a cross-hardware reliability,
  security, packaging, or throughput-SLA admission.

The benchmark harness now propagates any failed quality, context, quantized-KV,
concurrency, sampling, or CPU gate as a nonzero process exit. Earlier staged
output that printed a completed phase despite a quantized-KV validation failure
is retained as historical evidence and is not treated as a pass.

## Frozen greedy admission harness (2026-08-10)

The CLI now accepts a versioned local `--prompt-suite`, loads the model once per
backend, and records exact generated token IDs for every single-sequence case.
The initial `qwen36-greedy-admission-v1` corpus spans structured JSON, code,
technical prose, arithmetic, SQL, multilingual text, multi-turn constraints,
repetition, summarization, creative structure, and random-looking formatting.
An automated comparator rejects missing cases, generation errors, token-count
inconsistency, nondeterminism across repetitions, or any target/candidate token
mismatch.

This closes the tooling gap that made prior single-preview comparisons
insufficient. On the pinned Qwen3.6-27B Q4_K_S artifact and RTX 4090, the
target-only arm averaged 33.1115 tok/s across 12 cases. The retained
depth-ten MTP candidate with the tensor-core draft head and bit-exact tiled F32
verifier kernel produced the exact same generated token IDs in all 36
post-load samples (three repetitions of all 12 cases). It averaged **133.5585
tok/s**, with a 127.2545 median, 84.7750 minimum, 207.9894 maximum, and
1,917/3,483 accepted draft tokens. This is a 4.033x mean speedup over the
target-only control.

The new tiled kernel assigns one warp to each deterministic accumulation chain
and 32 adjacent output columns to its lanes. It retains the existing FMA and
reduction order while coalescing the right-hand matrix reads. A physical RTX
4090 oracle matched the established eight-chain kernel bit-for-bit, including
an 11x128x257 tail shape. In bounded Nsight captures, the affected dense-kernel
time fell from 10.273 ms to 9.011 ms (-12.28%); end-to-end mean throughput rose
from 131.506 to 133.872 tok/s in the matched one-repetition screen.

The complete corpus, raw JSON, exact-token comparator outputs, depth screen,
negative experiments, profiler captures, hashes, and reproduction commands
are registered in
[`qwen36-greedy-admission-rtx4090-2026-08-10`](../benchmark-results/text/qwen36-greedy-admission-rtx4090-2026-08-10/README.md).

This completes only the pinned deterministic-greedy multi-prompt gate. The
207.9894 tok/s maximum is a prompt-specific sample, not a 200 tok/s average or
service guarantee. Non-greedy sampling, long context, concurrency,
cancellation, memory pressure, recovery, clean packaging, and wider hardware
remain separate admission gates.

## Draft-head profile and tensor-core candidate (2026-08-10)

A bounded Nsight capture attributes 53.3% of the integrated predictor's GPU
time to its 65,536-row packed Q6_K output head. The runtime now has an opt-in
`XRT_QWEN_MTP_Q6_TENSOR_CORE_HEAD=1` path that reuses the existing Q6_K WMMA
verifier for the speculative single-row projection. The complete target head
and verifier remain unchanged and authoritative.

On a replacement RTX 4090 host, matched 17-sample arms measured 151.1889 +/-
0.3410 tok/s for the tensor-core head and 149.2613 +/- 0.3301 tok/s for the
scalar packed-Q6 control. The +1.9276 tok/s (+1.291%) improvement reduced mean
draft time 5.326%, retained 55/68 proposal acceptance and one deterministic
preview, and produced no errors. A single-row Q8 MMQ reuse experiment
regressed to 142.6491 tok/s and was removed from runtime code.

The stable-graph depth/vocabulary re-screen also retained depth eight with
65,536 rows. Depths 6, 10, and 12 and prefixes 57,344, 73,728, and 81,920 were
slower. Raw JSON, profiler exports, statistics, reproduction commands, and
limitations are in the
[`draft-head screen`](../benchmark-results/text/qwen36-mtp-draft-head-screen-rtx4090-2026-08-10/README.md)
and [`shape screen`](../benchmark-results/text/qwen36-stable-graph-shape-screen-rtx4090-2026-08-10/README.md).

This is a narrow experimental performance admission, not default enablement.
Multi-prompt acceptance/performance, long-context, concurrency, recovery, and
clean-environment reproduction remain open.

## Stable verifier graph and projection schedule (2026-08-10)

The retained same-host RTX 4090 result now averages **150.1710 tok/s** across
17 post-warmup samples: 150.3018 median, 0.6554 sample standard deviation,
148.6076 minimum, and 151.1824 maximum. Eleven of 17 samples reached at least
150 tok/s. The normal 95% confidence interval crosses 150 tok/s, so this admits
the measured mean only; it is not a per-request or cross-machine guarantee.

The candidate replaces per-window graph recapture with two stable verifier
graphs keyed by recurrent state-buffer generation. Stable device decode
parameters carry changing positions, Q4 embeddings write into persistent
scratch, shared-page F32 attention uses stable page-pointer tables, recurrent
and attention projections overlap on heterogeneous CUDA streams, host state is
published after launch, and layer cache lengths are committed in one validated
batch. The supported greedy verifier's device argmax is part of graph capture.
Partial final windows retain the eager fallback.

All retained samples produced 64 tokens with identical 55/68 MTP acceptance
in nine target verification batches and no errors. The same-host pre-change
XRT control averaged 116.0165 tok/s; the final result is 29.44% faster. A
same-host llama.cpp control with the same model artifact, prompt, greedy
settings, output cap, and MTP depth averaged 144.0091 tok/s after warmup. XRT
is 4.28% faster on this narrow tuple, not broadly admitted as faster.

Physical GPU tests prove that shared-page batched Qwen attention matches the
serial pipeline and that fused DeltaNet verification is bit-exact with the
serial kernels. Raw samples, profiler traces, hashes, reproduction command,
negative screens, and limitations are registered in
[`qwen36-stable-graph-150tps-rtx4090-2026-08-10`](../benchmark-results/text/qwen36-stable-graph-150tps-rtx4090-2026-08-10/README.md).

This closes the narrow 150 tok/s mean performance objective. It does not close
the production admission gates: MTP and verifier graphs remain experimental,
opt-in, and disabled by default pending multi-prompt parity, exact non-greedy
sampling, long-context, concurrency, cancellation, memory-pressure,
reliability, security, clean-environment reproduction, packaging, and wider
hardware validation.

## Marlin stage-depth follow-up (2026-08-10)

The verifier now uses three correctly provisioned Marlin pipeline stages on
the admitted Q4_K execution path: 27 KiB dynamic shared memory for the
64-column tile and 42 KiB for the 128-column tile. CUDA function opt-in is
configured once at module load. A new physical-GPU regression test also proves
that concatenating two Q4_K matrices by output row preserves the separate
Marlin results; the model-level grouped gate/up design was measured and
rejected because it removed useful two-stream projection overlap.

On one replacement RTX 4090 host, a same-host 23-run comparison retained 20
samples after three warmups. Stage two averaged 110.8178 +/- 0.0229 tok/s
(95% confidence half-width); stage three averaged 111.0078 +/- 0.0268 tok/s.
The 0.1715% gain reduced mean target verification from 465.576 ms to 464.718
ms with identical deterministic previews and identical 55/68 MTP acceptance.
The complete accepted and rejected scheduler screens are in
[`qwen36-marlin-stage-depth-screen-rtx4090-2026-08-10`](../benchmark-results/text/qwen36-marlin-stage-depth-screen-rtx4090-2026-08-10/README.md).

The different host was slower in absolute terms, so these figures do not
replace or extrapolate the earlier 118.8529 tok/s record. They admit only the
same-host stage-depth change. Four stages, both remaining upstream tile shapes,
minimum shared-memory reservations, and grouped gate/up were rejected. The
At that stage, the 150 tok/s target remained open and required a persistent or
otherwise cross-layer verifier design that eliminated material matrix traffic
or launch boundaries, not another Marlin launch-parameter sweep. The stable
graph result above subsequently supplied that design.

## Batched rebase and Marlin occupancy result (2026-08-10)

The latest retained RTX 4090 record averages **118.8529 tok/s** across 17
post-warmup samples (0.0394 tok/s sample standard deviation, 118.8063 minimum,
118.9189 maximum). It combines shared verifier F16 activation conversion,
preallocated batched NextN rebase for full proposal windows, and the upstream
Marlin small-output occupancy choice. The complete record is in
[`qwen36-batched-rebase-marlin-n64-rtx4090-2026-08-10`](../benchmark-results/text/qwen36-batched-rebase-marlin-n64-rtx4090-2026-08-10/README.md).

Five real-model audit windows retained identical optimized and serial argmax
vectors. Serial and batched rebase also produced byte-identical complete
greedy output and identical 55/68 acceptance, while rebase time fell from
25.389 ms to 10.037 ms in the paired screen. The final physical Marlin affine
test passed on the RTX 4090.

That candidate did not reach the 150 tok/s objective. Its decode window was about
530 ms for 63 timed tokens, versus the 420 ms required for 150 tok/s. Target
verification still consumes about 434 ms, so the next material optimization
must reduce verifier matrix work or launch/traffic cost rather than tune
rebase. MTP and Marlin remain experimental and disabled by default; the wider
admission gates below remain open.

## Compact greedy target verification (2026-08-10)

The CUDA target verifier now selects the first argmax for every MTP logit row
on-device. For unpenalized greedy requests without an EOS draft, the runtime
returns only the accepted-prefix length and boundary token instead of copying
the complete `rows x 248,320` logit matrix to the host. Temperature sampling,
repetition/presence/frequency penalties, EOS drafts, unsupported backends, and
audit execution retain the full-logit fallback.

On the pinned Qwen3.6-27B Q4_K_S artifact and RTX 4090, the final 20-run record
retained 17 samples after three warmups: **114.2518 tok/s mean**, 114.6814
median, 108.9885 minimum, and 114.7769 maximum. All 17 remained above 100
tok/s. Mean explicit device-to-host traffic fell from 77,476,112 bytes in the
same-node pre-change screen to 993,860 bytes, a 98.72% reduction. The complete
record and negative screens are in
[`qwen36-compact-greedy-rtx4090-2026-08-10`](../benchmark-results/text/qwen36-compact-greedy-rtx4090-2026-08-10/README.md).

Physical row-argmax coverage passed. The real-model verifier audit also now
suspends predictor tracking without destroying its cache while running the
serial target reference; three audited windows had identical optimized and
serial argmax vectors. Whole-verifier CUDA Graph update, proposal depths seven
and twelve, and batched-attention tile 16 were measured and rejected.

This is a narrow experimental performance admission, not production MTP
support. The multi-prompt parity, exact non-greedy sampling, long-context,
concurrency, reliability, security, packaging, and clean-environment gates
below remain open.

## Reusable verification scratch and Ada occupancy (2026-08-09)

The corrected depth-eight verifier now reuses its CUDA projection destinations
instead of allocating replacement buffers inside every layer. On one retained
64-token workload this removed 73.35% of allocation calls and 43.51% of
allocated bytes. The exact Q4_K/Q5_K verification kernels now launch eight
output-row warps per block instead of 16; the smaller register-heavy blocks
improved scheduling on the RTX 4090 without changing the arithmetic order.

Across ten measurements after three warmups, the same-GPU corrected baseline
averaged 49.312 +/- 0.386 tok/s, scratch reuse averaged 49.699 +/- 0.169 tok/s,
and the retained eight-warp candidate averaged **51.867 +/- 0.169 tok/s**.
All retained runs produced 64 tokens with 55/68 accepted drafts. Focused Q4_K
and Q5_K CUDA tests matched serial recurrent matvec bits; Q5_K coverage now
uses distinct activation rows rather than repeated copies.

A nine-row specialization and four-warp blocks were measured and rejected.
The former doubled verification time; the latter collapsed real-model
acceptance and throughput. Raw evidence and machine-readable aggregates are in
[`qwen36-mtp-scratch-reuse-rtx4090-2026-08-09`](../benchmark-results/text/qwen36-mtp-scratch-reuse-rtx4090-2026-08-09/README.md).

This clears the narrow 50 tok/s workload objective, not MTP admission or
llama.cpp MTP parity. The previously pinned llama.cpp MTP result remains about
144 tok/s. The next architectural target is a fused causal verification path
for the currently row-serialized DeltaNet and full-attention work, followed by
multi-prompt, non-greedy, long-context, concurrency, reliability, security,
and packaging gates.

## DeltaNet Q/K head-map correction (2026-08-09)

The paired comparison exposed a target-correctness defect in XRT's Gated
DeltaNet broadcast. XRT expanded each Q/K group into adjacent value-head
buckets; Qwen3.6 and the pinned llama.cpp reference tile Q/K groups over the
value heads. Qwen3.6-27B therefore requires `value_head % 16` across its 48
value heads. The CPU/reference and CUDA paths now use that mapping, reject
non-divisible geometry, and have explicit 32:16 and 48:16 tests.

The correction changed the retained prompt from an early 28-token completion
with 24/32 MTP acceptance to a 64-token-cap completion with 55/68 acceptance.
On a fresh secure RTX 4090, the corrected target-only path averaged 30.785 +/-
0.052 tok/s and full-vocabulary MTP averaged 45.736 +/- 0.129 tok/s. A
target-verified 65,536-row draft projection preserved the same output preview,
length, and 55/68 acceptance while averaging 50.418 +/- 0.255 tok/s across ten
measurements after three warmups. The exact same-host llama.cpp commit measured
50.627 +/- 0.027 tok/s target-only and 144.073 +/- 1.614 tok/s with MTP.

The retained evidence is
[`qwen36-deltanet-qk-map-rtx4090-2026-08-09`](../benchmark-results/text/qwen36-deltanet-qk-map-rtx4090-2026-08-09/README.md).
This resolves the earlier response-length mismatch and meets the narrow 50
tok/s mean objective with an opt-in draft prefix, but it does not admit MTP:
strict cross-runtime token-id parity, multi-prompt quality/performance,
non-greedy rejection sampling, clean reproducibility, and the remaining
runtime-domain gates are still pending.

## Adaptive exact verification result (2026-08-09)

The current portable `compute_70` candidate passes the requested workload-level
50 tokens/second gate on the pinned 24 GB RTX 4090. The complete record is
`benchmark-results/text/qwen36-adaptive-mtp-rtx4090-2026-08-09.json`.

- Clean target-only decode averaged 30.723 tokens/second with prompt lookup,
  prefix caching, and MTP disabled.
- Exact depth-eight MTP with a 74,752-row draft prefix measured 50.111, 50.294,
  50.273, 50.269, and 50.266 tokens/second, averaging 50.243. That is a 63.53%
  improvement over the clean target and passed the threshold in all five runs.
- Every run produced the same 28-token output and accepted 24 of 32 proposals
  in four target verification windows.
- A new Q5_K verifier reuses each packed weight row across 2-16 activations
  while remaining bit-exact to the recurrent matvec. The Q4_K verifier now uses
  the same exact warp reduction without a shared-memory reduction tail, and
  small Q6_K verification windows use the existing reusable-weight tile.
- Accepted-boundary rebase preserves the first predictor KV row because it was
  already computed from the same target-hidden checkpoint, then replays only
  later accepted rows. Phase telemetry reports draft, verify, and rebase time
  separately.
- The low-acceptance code control triggered adaptive fallback after one
  eight-token probe, preserved the 128-token output preview, and measured
  32.072 tokens/second versus 31.881 target-only.

This passes a narrow performance objective, not production admission. The
retained output samples themselves still expose model/runtime quality problems,
the automated multi-prompt quality suite is incomplete, and exact non-greedy
speculative rejection sampling is not implemented. MTP therefore stays
experimental and disabled by default; human quality review should wait for
those automated gates.

## FastMTP and llama-style kernel screen (2026-08-08)

The follow-up RTX 4090 screen is recorded in
`benchmark-results/text/qwen36-fastmtp-kernel-screen-rtx4090-2026-08-08.json`.
It adds two exact speculative optimizations and one experimental target kernel:

- device-side `f32::total_cmp` argmax for MTP draft logits, with the previous
  host scan retained as a fallback;
- an opt-in `XRT_QWEN_MTP_VOCAB_ROWS` draft projection prefix. The complete
  target projection still verifies every committed token, so an omitted draft
  token can only reduce acceptance, not bypass target verification; and
- an adaptive exact fallback that stops MTP after the first verification window
  once at least 6 cumulative proposals have been observed and
  cumulative acceptance is below 25 percent, then continues through the
  ordinary target decoder. It is enabled by default for MTP requests and can
  be disabled for controlled experiments with
  `XRT_QWEN_MTP_ADAPTIVE_FALLBACK=off`; and
- an opt-in `XRT_CUDA_Q4_K_FAST_MMVQ` Q8/DP4A batch-one Q4_K kernel using the
  activation scale and reconstructed-sum semantics used by llama.cpp's CUDA
  path.

On one retained raw prompt, device argmax reduced device-to-host traffic from
65,556,480 to 35,758,200 bytes without changing the generated output or the
23/30 acceptance count. A 77,824-row draft prefix then raised the measured MTP
decode rate from 33.882 to 36.066 tokens/second. The commonly cited 32K
FastMTP prefix regressed to 30.115 tokens/second on this artifact because draft
acceptance fell, so no prefix is a product default.

The Q8/DP4A kernel passed its scalar-reference CUDA test and made an isolated
5,120 by 5,120 Q4_K matvec 1.64x faster. It is not admitted: its short chat
screen changed one output into a repetition loop, and combining it with MTP
accepted only 1 of 24 drafts and fell to 6.867 tokens/second.

A CUDA 12.4 PTX rebuild did not reproduce the earlier greedy output: the target
selected EOS immediately while the CPU control generated 16 tokens. That was
the wrong compiler for this repository's CUDA 12.8.1 PTX pin. A follow-up on
driver 570.195.03 and CUDA 12.8.1 reproduced the established Q4 and dense PTX
byte-for-byte; the new Q8/MMVQ and argmax modules were regenerated there and
also passed byte freshness and their CUDA unit tests.

The pinned-toolchain production-chat screen still blocks admission. Target-only
decode averaged 31.127 tokens/second, while static depth-six MTP with the 77,824
row prefix accepted only 11 of 309 drafts and fell to 9.362 tokens/second. The
adaptive fallback unit test passes and now prevents sustained acceptance below
25 percent from drafting for the rest of the request, but its GPU benchmark was
interrupted when RunPod stopped the pod and rejected restart with a 402 balance
error. All new paths therefore remain opt-in or disabled, the earlier 30-37
tokens/second figures remain screening evidence only, and the 50 tokens/second
objective remains unmet. Do not request human quality review until the adaptive
fallback GPU run and the automated multi-prompt quality suite pass.

XENO RT recognizes integrated Qwen3.5-compatible GGUF artifacts whose physical
block count includes appended `nextn_predict_layers`. The target decoder trunk
and appended predictor blocks are tracked separately so the predictor can never
be executed as an ordinary target layer.

The first execution lane targets the one-layer Qwen3.6 NextN layout:

- `qwen35.nextn_predict_layers = 1`;
- one full-attention predictor block appended after the target trunk;
- `nextn.enorm`, `nextn.hnorm`, `nextn.eh_proj`, and
  `nextn.shared_head_norm` tensors; and
- shared token embeddings and output projection.

Set `XRT_QWEN_MTP=on` to opt into CUDA greedy drafting. Draft depth defaults to
one token and can be bounded from one through fifteen with
`XRT_QWEN_MTP_MAX_DRAFT_TOKENS`. The complete target model verifies a 2-16 row
proposal window in one layerwise pass through specialized 4-, 8-, and 16-row
Q4_K CUDA kernels. DeltaNet state advances causally inside each recurrent
layer, and the retained recurrent boundary is donated from an on-demand device
journal without replay. The checkpoint remains available for errors and
streaming cancellation.
Non-greedy requests remain on target-only or prompt-lookup decoding until exact
speculative rejection sampling is implemented.

MTP requests currently bypass shared prefix snapshots because those snapshots
do not yet include the predictor's synchronized attention lane. Requests with
image-conditioned embedding overrides also remain target-only until the MTP
head can consume the exact override embedding.

Full greedy parity requires verification through the configured target sampler,
including repetition penalty and EOS handling. Raw-logit argmax is not an
equivalent verifier when request-time sampling transforms are active.

This lane must remain disabled by default until a pinned real artifact passes:

1. target-only parity with the same artifact;
2. deterministic MTP-on/off output parity for greedy decoding;
3. accepted-boundary KV and recurrent-state rollback tests;
4. no OOM within the documented 24 GB RTX 4090 profile;
5. repeated throughput measurements showing a material decode improvement; and
6. the ordinary text runtime correctness, compatibility, and packaging gates in
   `RUNTIME_DOMAINS.md`.

## Current RTX 4090 MTP screening (2026-08-08)

The dirty candidate based on `77911ff` was exercised against the pinned
Qwen3.6-27B Q4_K_S artifact. The complete record is
`benchmark-results/text/qwen36-mtp-rtx4090-q4_k_s-2026-08-08.json`.

- Pure target decode averaged 29.800 tokens/second with prompt lookup disabled,
  up 227.61% from the 9.096 tokens/second scalar-kernel baseline.
- MTP depth 6 averaged 37.123 tokens/second: 1.246x the optimized target, or a
  24.57% improvement. Each repetition drafted 30 tokens, accepted 23 (76.67%),
  used five verification batches, and performed zero rollback replays.
- The prior depth-7 winner averaged 34.366 after this kernel change. Neighboring
  one-run screens measured 34.487 at depth 5 and 26.389 at depth 15. Draft depth
  therefore remains workload- and hardware-dependent.
- Nsight Systems attributed 41.7% of the original GPU time to scalar RMSNorm
  and 33.4% to scalar small F32 projections. The retained CUDA kernels execute
  RMSNorm's established eight chains concurrently, parallelize its output, and
  use the same deterministic eight-chain reduction for 1-16 row dense
  projections. The checked-in PTX is regenerated and freshness-checked in CI.
- A same-host exact-kernel A/B packed eight independent Q4_K output-row warps
  into each CUDA block and replaced the shared-memory reduction with an
  AVX2-order warp shuffle. Target decode increased from 28.210 to 30.166
  tokens/second (+6.93%); a 4/8/16-warp sweep selected eight. A final NVCC/PTX
  confirmation averaged 29.800 tokens/second. MTP remained flat at about 37.1.
- A fresh target profile still attributes 64.4% of GPU kernel time to the exact
  Q4_K projection (8,840 launches averaging 54.9 microseconds), so the next
  large gain requires a better exact packed-weight decode strategy or launch
  fusion. Generic Q4 dispatch and two rows per warp were measured and rejected.
- Target and MTP raw-generation samples were byte-identical, and the optimized
  target verifier was bit-exact against serial target execution.
- The depth-6 session used 17,886,446,416 tracked resident bytes on the 24 GB
  RTX 4090. Recurrent verify journals grow only when the selected depth needs
  them; target-only sessions retain the three-copy transactional footprint.
- A current llama.cpp `tg128` reference on the identical GPU and artifact
  measured 51.26 +/- 0.18 tokens/second. This proves the 50 tokens/second
  hardware objective is feasible, while isolating XRT's remaining gap to its
  target CUDA kernel stack.

This is a substantial MTP implementation result, not production admission.
XRT remains at 37.123 rather than 50 tokens/second, the retained throughput
sample is too narrow, and exact non-greedy rejection sampling is not yet
implemented. Human quality review should wait for the automated multi-prompt
parity/performance suite and the target-kernel gap to close.

## Historical RTX 4090 result (2026-08-08)

Commit `4ec4f4a` was exercised on a 24 GB RTX 4090 with the pinned
`Qwen3.6-27B-Q4_K_S.gguf` artifact recorded in
`benchmark-results/text/qwen36-mtp-rtx4090-q4_k_s-2026-08-08.json`.

- The 16-token CLI A/B produced byte-identical output and the three-run
  benchmark produced the same eight-token output in every target and MTP run.
- The CUDA transaction test passed with the default `1.1` repetition penalty.
- The real model loaded without OOM and peaked at 21,157,251,872 tracked bytes
  with MTP enabled.
- The acceptance sample accepted 5 of 10 drafts (50%).
- Warm decode fell from 7.276 to 3.544 tokens/second, a 51.29% regression.

The performance admission gate therefore failed. MTP remains experimental and
disabled by default. Before rerunning admission, remove enough predictor and
verifier overhead to demonstrate a material speedup, then run a multi-prompt
quality/parity suite and the remaining text release gates. Human quality review
is not useful until those automated performance and coverage gates pass.

## A40 Turbo screening result (2026-08-08)

The follow-up working-tree candidate based on `77911ff` is recorded in
`benchmark-results/text/qwen36-turbo-a40-q4_k_s-2026-08-08.json`.

- The 248,320 x 5,120 Q6_K output matrix remains packed instead of expanding
  past 5 GiB in F32. Tracked warm residency fell by 4,022,784,000 bytes.
- The recurrent Q4_K kernel decodes each scale/minimum pair once per warp while
  preserving the exact CPU-order accumulation and reduction contract.
- Matched target-only warm decode increased from 5.165 to 5.703 tokens/second
  on an NVIDIA A40, a 10.40% improvement.
- Accepted-boundary incremental verification increased the depth-two MTP lane
  from 3.206 to 5.431 warm tokens/second and eliminated ordinary rejection
  rollbacks, but it remained 4.76% slower than the improved target-only lane.
- Baseline, candidate target-only, and candidate MTP produced byte-identical
  32-token greedy output for the retained parity prompt.

This is screening evidence, not admission. The remaining performance-critical
change is a real verify-shaped target execution path: project the two- or
three-row speculative window layer by layer, use verify-specific quantized
matrix kernels that reuse each weight row, advance DeltaNet state sequentially
inside the verify operation, run causal full attention over the same window,
and publish only the accepted KV/recurrent boundary. The existing
`forward_batch_all_logits` loop is not that implementation. MTP stays disabled
by default until this path beats target-only decode across structured and
open-ended workloads on the pinned RTX 4090 target.

## RTX 4090 Turbo follow-up (2026-08-08)

The same dirty candidate was re-run on the required 24 GB RTX 4090 target. The
full record is
`benchmark-results/text/qwen36-turbo-rtx4090-q4_k_s-2026-08-08.json`.

- Matched target-only warm decode increased from 8.551 to 9.506 tokens/second,
  an 11.17% improvement, while tracked resident VRAM fell by 4,022,784,000
  bytes.
- Depth-two MTP reached 9.082 warm tokens/second and remained 4.46% slower than
  the improved target-only lane.
- A final-step regression exposed a retained two-token MTP cache paired with a
  one-token decode-parameter buffer when only one output token remained. MTP
  parameter capacity now follows the retained cache allocation. The exact
  16-token reproduction completes with zero rollbacks and byte-identical
  target/MTP output.
- OpenAI-compatible `presence_penalty` and `frequency_penalty` now flow through
  the server, CLI, Python binding, request model, and both greedy and
  temperature samplers. Their zero defaults preserve existing output.

This is still not Turbo/MTP admission: 9.506 tokens/second is far below the
50-token/second objective, and serialized target verification cannot turn high
draft acceptance into a throughput gain. A compiled multi-row verifier and a
substantially faster CUDA quantized-matvec baseline remain required.
