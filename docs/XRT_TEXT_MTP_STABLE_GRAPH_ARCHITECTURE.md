# XRT Text Stable-Graph MTP Architecture

- **Scope:** Native Qwen3.6 dense-hybrid MTP on CUDA
- **Status:** Experimental, opt-in, disabled by default
- **Decision date:** 2026-08-10
- **Performance record:** [150 tok/s RTX 4090 evidence](../benchmark-results/text/qwen36-stable-graph-150tps-rtx4090-2026-08-10/README.md)
- **Draft-head candidate:** [151 tok/s RTX 4090 follow-up](../benchmark-results/text/qwen36-mtp-draft-head-screen-rtx4090-2026-08-10/README.md)
- **Multi-prompt admission:** [134 tok/s mean, 208 tok/s peak RTX 4090 evidence](../benchmark-results/text/qwen36-greedy-admission-rtx4090-2026-08-10/README.md)
- **Public admission:** [Qwen3.6 MTP admission](QWEN36_MTP_ADMISSION.md)

## 1. Purpose

This document records the implementation architecture behind XRT's retained
150.1710 tok/s Qwen3.6-27B result. It is the code-facing complement to the
[acceleration whitepaper](XRT_TEXT_ACCELERATION_WHITEPAPER.md).

The design goal is to evaluate a fixed-size MTP proposal window as one stable
CUDA transaction while preserving XRT's target-token, cache, rollback, error,
and CPU-fallback contracts.

## 2. Non-goals

The stable graph path does not:

- add a new public API or change an OpenAI-compatible response;
- replace GGUF, CPU fallback, or ordinary CUDA decode;
- production-admit MTP, arbitrary Qwen models, or arbitrary KV modes;
- implement exact non-greedy speculative sampling;
- execute partial final windows through the full-row graph; or
- depend on llama.cpp at build time or runtime.

## 3. Component map

| Concern | Owner | Primary implementation |
|---|---|---|
| MTP request policy and token transaction | `xrt-runtime` | `crates/xrt-runtime/src/session.rs` |
| CUDA model/session admission | `xrt-runtime` | `crates/xrt-runtime/src/backend.rs` |
| Stable graph cache and execution | `xrt-runtime` | `CudaQwen35VerifyGraphState` in `backend.rs` |
| DeltaNet transactional state | `xrt-runtime` | `crates/xrt-runtime/src/recurrent_state.rs` |
| CUDA buffers, streams, graphs, and kernels | `xrt-cuda` | `crates/xrt-cuda/src/lib.rs` |
| Batched Qwen attention | `xrt-cuda` | `qwen35_verify_attention.cu` |
| Q4/Q5/Q6 verifier kernels | `xrt-cuda` | `q4_k_recurrent.cu`, `kquant_mmq.cu`, `marlin_q4_k.cu` |
| Device target decision | `xrt-cuda` | `argmax_f32.cu` |
| Controlled benchmark | repository script | `scripts/benchmark-qwen36-mtp.sh` |
| Depth/vocabulary screen | repository script | `scripts/screen-qwen36-mtp-shape.sh` |
| Bounded draft/verify profiling | `xrt-runtime` | `XRT_QWEN_MTP_PROFILE_DRAFT_WINDOW`, `XRT_QWEN_MTP_PROFILE_VERIFY_WINDOW` |

## 4. Eligibility

A request enters the graph verifier only when all required properties are
true:

```text
CUDA resident backend
  AND compatible dense hybrid Qwen model
  AND proposal rows in the admitted verifier range
  AND F32 session KV cache
  AND fused DeltaNet verify enabled and eligible
  AND fixed verifier scratch allocated
  AND shared-page cache pointer topology stable
  AND XRT_QWEN_MTP_VERIFY_GRAPH=1
  AND profiler capture is not forcing eager execution
```

Failure to meet an eligibility condition selects the existing path. It is not
silently reported as graph execution.

## 5. State model

### 5.1 Graph key

The current verifier graph identity is:

```text
GraphKey = (proposal row bucket, recurrent buffer generation)
```

The graph state retains at most two entries because recurrent buffers have two
pointer orientations. Changing the row bucket resets the entries. This is
deliberately smaller and stricter than a general graph cache.

### 5.2 Recurrent generation

Each DeltaNet layer has committed and pending device buffers. The host-side
generation identifies which orientation is committed:

```text
generation 0: A = committed, B = pending
generation 1: B = committed, A = pending
```

The device graph writes the appropriate orientation but does not swap Rust
handles during capture. After a successful graph launch,
`commit_fused_verify_graph_layers()` performs the host-only publication and
advances the generation when row parity requires it.

### 5.3 Full-attention state

Full-attention layers use F32 shared pages with stable key and value pointer
tables. Kernels append speculative rows using device parameters. After launch,
`commit_layer_kv_graph_append_batch()` validates the expected start and end and
publishes the new logical length once per layer.

### 5.4 Scratch ownership

The verifier owns persistent buffers for:

- alternating layer inputs;
- normalized activations and F16 conversion;
- Q, K, V, attention, gate, up, and hidden intermediates;
- full target logits and compact argmax indices;
- stable decode parameters; and
- projection streams and synchronization events.

No captured kernel may depend on a per-window allocation whose address can
change before replay.

## 6. Transaction sequence

```text
Host/session                Stable device memory             CUDA graph
     |                               |                           |
     | validate model/cache/state    |                           |
     |------------------------------>|                           |
     | upload token rows directly    |                           |
     |------------------------------>| layer_input_a             |
     | update position metadata      |                           |
     |------------------------------>| CudaDecodeParams          |
     |                               |                           |
     | launch graph for generation --+-------------------------->|
     |                               |   layer 0..N               |
     |                               |   recurrent/full attention |
     |                               |   FFN/output projection    |
     |                               |   row argmax               |
     |                               |<---------------------------|
     | download bounded argmax ids   |                           |
     |<------------------------------|                           |
     | compute accepted boundary     |                           |
     | publish recurrent handles     |                           |
     | publish KV lengths by layer   |                           |
     | rebase predictor to boundary  |                           |
     | commit verified output tokens |                           |
```

The host never publishes a speculative suffix before target verification.

## 7. Graph lifecycle

```text
disabled ----------------------------------------------> eager path
   |
enabled, unseen row bucket
   v
not captured -- prepare stable scratch --> warm/cold-capture decision
   |                                          |
   | eager warmup                             | full bucket may capture cold
   v                                          v
warmed ----------------------------------> capture generation G
                                              |
                                              v
                                  graph G captured and replayable
                                              |
                         opposite generation observed and captured
                                              |
                                              v
                                two stable graph executables

Any capture/replay error --> clear entries --> eager fallback for session
```

The retained full proposal bucket captures immediately because all buffers and
topology are already prepared. A partial final bucket uses eager execution and
does not evict the useful full-bucket graphs.

## 8. Layer execution

### 8.1 DeltaNet layers

Fused batched DeltaNet verification advances convolution and recurrent state
on-device across the row window. State copies are device operations inside the
graph; host handle swaps are deferred until successful completion.

### 8.2 Full-attention layers

The batched kernel performs:

1. Q/gate deinterleave;
2. per-head Q/K normalization;
3. position-dependent RoPE using `CudaDecodeParams`;
4. shared-page K/V append;
5. causal grouped-query attention for every verify row; and
6. sigmoid gate multiplication.

The shared pointer-table implementation is required because this is the cache
topology used by real sessions.

### 8.3 Projection schedule

Independent projections run on multiple streams:

- Q4_K Marlin projections use F16 activation scratch;
- Q5_K verifier projections use their exact packed path;
- F32 projections use dense kernels; and
- completion events join each group at its true consumer.

The stream schedule is part of capture. Pointers, streams, and event topology
must remain stable for graph replay.

### 8.4 Output and decision

The final RMSNorm and complete target vocabulary projection always execute.
For supported greedy requests, row argmax executes on-device and only one token
index per row is downloaded. Full logits remain available to routes whose
sampling or API contract requires them.

### 8.5 Speculative Q6_K draft head

The predictor normally projects the admitted 65,536-row vocabulary prefix
with the packed F32-activation Q6_K row kernel. An opt-in candidate reuses the
existing Q6_K WMMA verifier with one live activation row and 15 zero-padded
rows. Its F16 staging can change draft proposals, but cannot publish a token:
the complete target head still verifies every accepted token.

On the registered replacement-host workload, this reduced mean draft time
from 83.409 to 78.967 ms and raised end-to-end decode from 149.2613 to 151.1889
tok/s across 17 retained samples per arm. It is not enabled by default until
multi-prompt acceptance and performance are registered.

### 8.6 Tiled exact F32 verifier projection

Small F32 verifier projections with two through sixteen activation rows can
opt into `matmul_eight_chain_tiled_kernel`. One block owns 32 adjacent output
columns for one activation row, and its eight warps retain the established
eight independent depth chains. Each warp therefore reads adjacent matrix
columns at a fixed depth while preserving the exact per-chain FMA order and
ordered reduction used by the control kernel.

The physical 11x128x257 oracle is bit-exact, including the non-multiple-of-32
tail. On the frozen 12-prompt suite, depth ten plus the 65,536-row tensor-core
draft head is the retained multi-prompt tuple: 133.5585 tok/s across 36 samples
with exact target-token parity. This newer multi-prompt record supplements the
earlier single-prompt depth-eight result; it does not replace that experiment's
historical accounting or production-admit either tuple.

## 9. Correctness invariants

| Invariant | Enforcement |
|---|---|
| Every committed token is selected by the complete target head | Full target output projection in every verify window |
| Graph pointers match recurrent state orientation | Graph entry keyed by buffer generation |
| Rejected suffix cannot leak into host-visible state | Publication occurs after successful launch and accepted-boundary processing |
| Shared-page attention matches serial semantics | Physical GPU oracle compares outputs and gathered K/V |
| DeltaNet graph state matches serial kernels | Physical bit-exact fused-window oracle |
| Unsupported sampler behavior is unchanged | Compact readback eligibility and full-logit fallback |
| Capture failure is recoverable | Session-local eager fallback and retained ordinary path |
| Partial output windows are safe | Eager execution for unmatched row bucket |

## 10. Performance accounting

The retained request averages 419.529 ms for 63 timed tokens:

```text
Draft       85.010 ms  ####################                 20.3%
Verify     319.495 ms  ############################################################################ 76.2%
Rebase      11.332 ms  ###                                    2.7%
Other        3.692 ms  #                                      0.9%
```

The architecture moved the same-host control through these stages:

```text
116.0165  eager verifier
    |
    +-- stable dual graph topology ----------------> 137.2741
    +-- recurrent projection overlap --------------> 144.7952
    +-- full-attention projection overlap ----------> 148.4991
    +-- direct input + graph argmax + batch commit -> 150.1710 tok/s
```

The dominant remaining target kernel family is Q4_K Marlin. Drafting is now
large enough to deserve independent optimization. Its bounded trace identifies
the Q6_K output head and per-token readback dependency as the dominant draft
limits.

## 11. Configuration surface

The benchmark helpers control two recorded experimental tuples. Values below
are shown as `single-prompt / multi-prompt` where the screens selected
different settings:

| Variable | Recorded value | Meaning |
|---|---:|---|
| `XRT_QWEN_MTP` | `1` | Enable integrated Qwen MTP |
| `XRT_QWEN_MTP_MAX_DRAFT_TOKENS` | `8 / 10` | Proposal ceiling |
| `XRT_QWEN_MTP_VOCAB_ROWS` | `65536` | Fast predictor output prefix |
| `XRT_QWEN_MTP_BATCHED_REBASE` | `1` | Use batched predictor rebase |
| `XRT_QWEN_MTP_VERIFY_GRAPH` | `1` | Enable stable verifier graphs |
| `XRT_QWEN_MTP_Q6_TENSOR_CORE_HEAD` | `1` | Opt into the measured WMMA speculative output head |
| `XRT_CUDA_DENSE_SMALL_MATMUL_TILED` | `0 / 1` | Coalesce exact small-row F32 verifier projections |
| `XRT_CUDA_PARALLEL_VERIFY_PROJECTIONS` | `1` | Enable heterogeneous projection streams |
| `XRT_CUDA_Q4_K_MARLIN` | `1` | Use admitted Q4_K Marlin path |
| `XRT_CUDA_KQUANT_TENSOR_CORE_VERIFY` | `1` | Enable admitted verifier kernels |

These variables are experimental controls, not a production configuration
contract.

## 12. Observability

The benchmark JSON records:

- requested and active backend;
- model architecture and quantization;
- output and decode timing;
- MTP draft, acceptance, rejection, verify-batch, rollback, and phase time;
- explicit transfer deltas;
- allocator baseline, final, and peak values; and
- GPU resource and host memory snapshots.

`XRT_QWEN_MTP_PROFILE_VERIFY_WINDOW` starts a bounded CUDA profiler capture for
one numbered verify window. The benchmark helper can wrap execution in Nsight
Systems using `XRT_NSYS_OUTPUT`.

`XRT_QWEN_MTP_PROFILE_DRAFT_WINDOW` performs the equivalent bounded capture
around one complete predictor window. If either bounded selector is supplied,
the benchmark helper does not silently activate the other selector.

The runtime-level GPU status snapshot does not currently identify the graph
owned by an individual live session. Session-scoped graph state should be
added before production admission so observability does not rely on profiler
traces or capture log messages.

## 13. Rejected designs retained as evidence

- recapturing or updating the full verifier graph every window;
- row-serial attention under the shared-page cache wrapper;
- grouped gate/up projection that removed stream overlap;
- Q5_K N32 and N64 experimental tile candidates;
- deeper or alternative Marlin stage/tile settings without a same-host gain;
- graph paths that included host handle mutation during capture; and
- full-logit host readback for the eligible compact greedy path;
- depth 6, 10, and 12 and draft prefixes 57,344, 73,728, and 81,920 on the
  final stable-graph topology; and
- single-row Q8/DP4A MMQ for the Q6_K draft head, which increased setup cost
  and reduced throughput to 142.6491 tok/s.
- a variable-row adaptive depth controller, whose extra graph shapes reduced
  the 12-case mean to 107.6387 tok/s;
- reuse of the generic coalesced dense matmul, which reduced the mean to
  102.0161 tok/s;
- a 64-column exact tiled kernel, which reduced the mean to 129.4825 tok/s;
- contiguous expansion of the draft vocabulary to 77,824 rows, which recovered
  only one additional accepted token and reduced the mean to 131.8414 tok/s;
  and
- n-gram-only speculation on this corpus, which averaged 39.0168 tok/s.

Negative results are part of the architecture record because they prevent a
future launch-count optimization from being reintroduced without measuring the
overlap it destroys.

## 14. Next architecture experiments

Ordered by current evidence:

1. extend the frozen corpus to non-greedy sampling and longer contexts;
2. implement frequency-ranked mapped draft rows behind an audit flag;
3. prototype a device-resident token/embedding handoff across the draft
   steps so the host does not serialize every argmax;
4. collect Nsight Compute roofline/SASS metrics for dominant Q4_K row shapes;
5. fuse only the small layer boundaries proven to be memory or launch limited;
6. prototype device-resident acceptance and conditional-graph control;
7. add session-scoped graph observability and cancellation tests; and
8. rerun the complete candidate from a clean pinned container and second GPU.

Every experiment must retain identical target output for the greedy corpus,
report acceptance and phase timing, and preserve an eager fallback.

## 15. Review checklist

Before changing this architecture, reviewers should confirm:

- graph keys cover every pointer topology changed by the patch;
- no host handle or vector mutation occurs inside capture;
- device buffers outlive every graph executable that references them;
- event dependencies are sufficient across projection streams;
- cache publication validates the exact expected range;
- failure before publication leaves recoverable state;
- fallback behavior is explicit for unsupported sampling and KV modes;
- physical GPU oracles cover new kernels/topologies; and
- benchmark evidence includes the control, candidate, raw samples, and hashes.
