# KTransformers-Inspired Exact Hybrid MoE and Qwen3.5 Acceleration

**Status:** Implemented for explicit opt-in; overall completion and `auto` admission gates remain open

**Owner:** XENO RT team

**Created:** 2026-07-20

**Last updated:** 2026-07-21

**Target repository:** `xeno-rt`

**Related plan:** `docs/GPU_RUNTIME_ACCELERATION_SPEC.md`

**Runtime domain:** `xrt-text`

**Canonical architecture:** [RUNTIME_DOMAINS.md](RUNTIME_DOMAINS.md)

This specification covers exact sparse/hybrid conversational-model execution.
It does not define the whole XENO RT product or impose MoE/session abstractions
on image, video, or audio runtimes.

## 1. Executive summary

XENO RT already has a strong dense-model CUDA path, a pure-Rust CPU fallback, GGUF loading, an OpenAI-compatible server, paged and quantized KV caches, CUDA graphs for supported dense decode shapes, and initial CPU implementations of Qwen3.5-style DeltaNet and mixture-of-experts (MoE) layers. The largest remaining opportunity is not another isolated kernel. It is an execution architecture that can run the active parts of a sparse or hybrid model concurrently across CPU and GPU while keeping model semantics unchanged.

This specification adapts the most transferable ideas demonstrated by KTransformers and its current SGLang integration:

- keep hot experts resident on the GPU and execute cold experts on the CPU;
- overlap CPU expert work with GPU expert work instead of serially offloading whole layers;
- make expert placement explicit, measurable, and eventually adaptive;
- use topology-aware CPU worker pools for expert computation;
- give every request its own recurrent state so Qwen3.5 hybrid models can be scheduled concurrently;
- use stable buffers and placement generations so fixed-shape work can later participate in CUDA graphs;
- add layerwise prefill as a later, exact memory-saving mode; and
- treat recurrent state as part of prefix-cache and speculative-decoding correctness.

The production default in this specification is **exact execution**. Every router-selected expert is executed, the same routing weights are applied, and results are merged in a deterministic logical order. Expert dropping and expert deferral are deliberately excluded because they change model behavior, even when their average benchmark impact is small.

The work is staged behind internal capability gates. OpenAI request and response schemas do not change. GGUF remains a required, first-class model input for this work, while existing SafeTensors, multimodal-chat, and ONNX task/vision paths remain unchanged. CPU-only builds continue to compile and run. Unsupported explicit CUDA requests fail clearly, while `auto` selects a validated exact path during runtime/model loading, before request state can be mutated.

“Hundreds of tokens per second” is a useful optimization direction, not a portable acceptance claim. Hundreds of prefill tokens per second or aggregate decode tokens per second can be realistic for suitable active-parameter counts, batch sizes, memory bandwidth, CPUs, and GPUs. Hundreds of single-stream decode tokens per second on a large model cannot be promised by an architecture document. XENO RT will report single-stream decode, aggregate decode, and prefill separately and enable a new path by default only when same-hardware measurements show a real benefit.

## 2. Problem statement

The current runtime has three related limitations.

First, MoE execution is CPU-only. Routing and selected-expert evaluation are implemented in `crates/xrt-models/src/llama.rs`, but selected experts run serially and multiple forward paths contain similar MoE loops. Routing helpers allocate per token, the global CPU pool is not NUMA-aware, and no runtime object describes where an expert resides or how CPU and GPU expert results should be joined.

Second, Qwen3.5 DeltaNet state is owned by `LlamaModel` through a shared `RwLock<Option<DeltaNetState>>`. `BackendSession` owns the KV cache but does not own recurrent state. Consequently:

- hybrid sessions require an exclusive scheduler turn;
- prefix sharing is bypassed for hybrid models;
- the model-level snapshot API clones shared recurrent state; active hybrid n-gram drafting is currently disabled, while any future rollback path still needs a session-specific snapshot boundary;
- two sessions cannot safely advance independent recurrent histories through one model instance; and
- a future CUDA DeltaNet implementation would have no correct per-session allocation boundary.

Third, the CUDA resident backend rejects MoE and Qwen3.5 hybrid models. The dense resident path, GPU resource accounting, scratch buffers, graph cache, scheduler, and quantized KV mechanisms exist, but there is no resident expert-slot abstraction, heterogeneous execution coordinator, or CUDA recurrent-state pool.

Optimizing only the current CPU loop would leave most of the opportunity unused. Offloading an entire sparse model to a constrained GPU would either fail to fit or waste transfers on inactive experts. Copying selected expert weights every decode token would usually be dominated by PCIe traffic. The runtime instead needs to keep a bounded hot set on the GPU, retain all experts in host-accessible storage, transfer only activations during decode, and execute both tiers concurrently.

## 3. Goals

### 3.1 Functional goals

1. Run supported GGUF MoE models through an exact optimized CPU path.
2. Run supported GGUF MoE models through an exact heterogeneous CPU/GPU path when only a subset of experts fits in VRAM.
3. Support static GPU expert placement first and safe adaptive placement later.
4. Move all DeltaNet recurrent state into the owning `BackendSession`.
5. Add CUDA Qwen3.5 DeltaNet decode and extend/prefill execution without changing model semantics.
6. Permit multiple hybrid sessions to make progress without sharing recurrent state.
7. Extend prefix caching and speculative rollback to include recurrent state at exact token boundaries.
8. Preserve the current dense-model behavior and performance.
9. Expose enough telemetry to explain whether CPU work, GPU work, transfers, synchronization, routing locality, or recurrent state is limiting throughput.

### 3.2 Compatibility goals

1. Preserve `/v1/chat/completions`, `/v1/completions`, and `/v1/models` compatibility.
2. Preserve GGUF loading, existing supported tensor encodings, and existing SafeTensors paths.
3. Preserve a working CPU-only build with no CUDA Toolkit or CUDA driver dependency.
4. Preserve explicit backend semantics:
   - `cpu` must remain CPU-only;
   - explicit `cuda`/`cuda-resident` must either run a supported CUDA path or return an actionable unsupported error;
   - `auto` may fall back to CPU during runtime/model loading, before request state is mutated.
5. Keep CUDA kernel compilation reproducible through the repository’s pinned CUDA container and checked-in PTX workflow.
6. Keep the new local GGUF paths offline. Existing explicitly requested external-proxy, model-download, and image-URL behavior remains unchanged. KTransformers is a design and differential-performance reference, not a runtime dependency.
7. Preserve existing multimodal chat inputs and `xrt-vision`/ONNX task endpoints, including `/v1/images/remove-background`.

### 3.3 Performance goals

1. Remove avoidable per-token allocation and duplicate routing work from the MoE hot path.
2. Overlap cold-expert CPU computation with hot-expert GPU computation.
3. Improve CPU socket and memory locality for expert weights.
4. Bound VRAM use by a declared expert budget and report the effective allocation.
5. Enable fixed-placement CUDA graph reuse once the eager exact path is correct.
6. Increase aggregate throughput through safe batching after recurrent state becomes session-owned.
7. Establish and pursue a documented `>=100 aggregate decode tokens/s` stretch result on at least one suitable MoE model and hardware configuration. This is a benchmark objective, not a universal release gate.

## 4. Non-goals

1. Replacing GGUF with a KTransformers-specific or Hugging Face-only model format.
2. Embedding Python, SGLang, KTransformers, `llama.cpp`, or an external inference process in XENO RT.
3. Breaking the OpenAI-compatible HTTP surface to expose backend internals.
4. Requiring CUDA for model loading or inference.
5. Promising a specific absolute token rate across models or machines.
6. Expert dropping, expert skipping, or expert deferral in the production exact path.
7. Silent mid-request fallback after recurrent or KV state has been mutated.
8. Implementing AMD ROCm, Apple Metal, or Intel GPU acceleration in this project. The design must not prevent those backends later.
9. Replacing all existing dense CUDA kernels or KV-cache work.
10. Copying KTransformers’ archived Python runtime or vendoring its third-party tree.
11. Cutting a release. Release work remains governed by `release-guide/`.

## 5. Definitions

- **Logical expert:** The expert index encoded by the model and selected by the router.
- **Physical expert slot:** A bounded CPU or GPU storage/execution location that currently contains one logical expert.
- **Hot expert:** An expert assigned to a GPU slot for a placement generation.
- **Cold expert:** An expert executed from host-resident weights.
- **Exact execution:** All selected experts run, the normalized routing weights are preserved, and no approximation intentionally changes the model graph.
- **Placement generation:** A monotonically increasing identifier for one immutable logical-to-physical expert mapping.
- **Execution plan:** The per-layer, per-token or per-batch routing result partitioned into CPU and GPU work under one placement generation.
- **Hybrid model:** A model containing both full-attention layers and recurrent/linear-attention layers, such as supported Qwen3.5 layouts.
- **Recurrent state:** DeltaNet convolution and state-matrix data that evolves for one session.
- **Durable snapshot:** A backend-independent host representation suitable for prefix ownership or correctness checks.
- **Fast checkpoint:** A backend-local rollback mechanism, such as a CUDA journal or ping-pong device state, used within a request.
- **Decode throughput:** Generated tokens per second after prefill.
- **Aggregate decode throughput:** Sum of decoded tokens across concurrent requests divided by wall time.
- **Prefill throughput:** Prompt tokens processed per second.

## 6. Current repository findings

This section records the implementation baseline as of 2026-07-20. The audited workspace was local HEAD `e1bb2e67fa4a2cf6ac399a8bbaee34e9d20de2e2`; the audited `origin/main` and live remote `main` were both `9bdc8d17dd618a38513df1176bff1eb8be52792a`. Implementers must re-run the cited inspections before changing interfaces.

### 6.1 Existing model support

`crates/xrt-models/src/llama.rs` already:

- recognizes Llama, Qwen2, Qwen3, Qwen3.5-like, and Gemma4 families;
- accepts Qwen3.5-related GGUF architecture aliases;
- reads `expert_count` and `expert_used_count`;
- loads router and per-expert gate, up, and down tensors;
- implements exact selected-expert CPU evaluation;
- implements Qwen3.5 DeltaNet on CPU; and
- can save and restore DeltaNet convolution and recurrent matrices.

The implementation is currently concentrated in a large file. MoE logic is duplicated across single-token and batched forward paths. The `top_k_indices` helper builds and sorts a fresh index vector, and selected-expert temporary values are allocated in several paths.

The current Qwen3.5 layer schedule is inferred by an architecture-specific rule in `LlamaConfig::is_recurrent`. That rule must be validated against every accepted real checkpoint. A resolved layer plan must eventually distinguish recurrent, full-attention, dense FFN, routed MoE, and any shared-expert contribution without guessing from a model name alone.

### 6.2 State ownership

`LlamaModel` owns `deltanet_state: RwLock<Option<DeltaNetState>>`. The runtime trait exposes model-level `clear_state`, `save_state`, and `restore_state` operations without a session parameter.

By contrast, `crates/xrt-runtime/src/backend.rs` gives each `BackendSession` a KV cache and CUDA resources. `crates/xrt-runtime/src/session.rs` therefore grants hybrid generation an exclusive execution turn, disables normal cooperative behavior, and bypasses prefix-cache reuse. This shared-state boundary is the first issue to fix.

Moving recurrent state is necessary but not sufficient to permit parallel model forwards. `LlamaModel` also owns `ForwardScratch` and `BatchScratch` behind write locks that are held across forward execution. The exclusive hybrid scheduler guard must remain until every mutable hybrid hot-path scratch object is session-owned or obtained through a bounded runtime lease.

### 6.3 Existing CUDA foundation

The resident backend already has:

- model-resident dense and Gemma weights;
- persistent CUDA buffers and tracked resource accounting;
- multiple GGUF and packed SafeTensors matrix implementations;
- F32, Q8, mixed KQ4/VQ8, and adaptive GPU KV modes;
- continuous batching for supported dense shapes;
- fixed-shape CUDA graph infrastructure;
- exact prefix snapshots for supported full-attention models;
- an additive runtime-status surface; and
- safe compile-only and opt-in real-GPU validation scripts.

The current support predicates intentionally reject MoE and Qwen3.5 hybrid models. Multi-sequence decode and CUDA graph eligibility also exclude hybrid models.

### 6.4 Existing CPU foundation

`xrt-kernels` already includes substantial AVX2, FMA, AVX-512, VNNI, and quantized matvec work. This proposal must reuse those kernels. The missing piece is an MoE-oriented execution layer with bounded task queues, stable scratch, expert grouping, and topology awareness.

The current global spin pool is intentionally simple, caps its worker count, and does not model sockets or NUMA nodes. Adding a second unconstrained pool would risk oversubscription; the new expert executor needs an explicit ownership and thread-budget policy.

### 6.5 Benchmark and CI foundation

The repository already provides:

- Criterion inference and matvec benchmarks;
- `xrt bench` JSON reports with latency, throughput, memory, backend, and GPU status;
- external OpenAI service comparison;
- standard `check`, `test`, `fmt`, `clippy`, and benchmark-build CI;
- a pinned CUDA 12.8.1 PTX generation/reproducibility job; and
- an opt-in RTX self-hosted workflow for real model validation.

No current benchmark isolates router cost, expert dispatch, placement hit rate, CPU/GPU overlap, recurrent-state copying, or end-to-end hybrid-MoE behavior.

The server also exposes `/v1/completions`, `/v1/runtime/status`, `/v1/runtime/load`, `/v1/runtime/unload`, and `/v1/images/remove-background`. It has no inbound authentication middleware, and its bind host can be configured beyond loopback. This specification therefore must not add placement manifests, expert budgets, or placement controls as new HTTP fields.

## 7. External research and transfer boundary

### 7.1 Adopted ideas

The following ideas are architecture references, not code dependencies:

1. **Concurrent heterogeneous MoE.** KTransformers’ current CPU backend and SGLang wrapper submit CPU experts while GPU-resident experts execute, then join both results.
2. **Hot/cold expert placement.** Logical expert IDs are remapped to a bounded GPU-resident set, with unplaced experts handled by CPU.
3. **Stable mapping storage.** Placement maps can be updated in place while their device addresses remain stable, which is useful for graph capture.
4. **Topology-aware expert workers.** KTransformers binds and organizes CPU workers with NUMA locality and work stealing.
5. **Stream-integrated CPU submission.** KTransformers uses CUDA host callbacks to submit and synchronize CPU work inside a stream schedule.
6. **Frequency-informed placement.** Routing statistics can improve which experts occupy scarce GPU slots.
7. **Layerwise prefill.** Expert weights can be staged layer by layer with double buffering when the full active layer set does not fit in VRAM.
8. **Request-indexed recurrent state.** SGLang’s Qwen3.5 linear-attention backend indexes convolution and state-matrix storage by request and distinguishes decode, extend, and speculative verification.
9. **State-aware radix reuse.** Hybrid prefix reuse must associate full-attention KV with recurrent state and support fork, copy-on-write, protection, and eviction.

The audited KTransformers support matrix marks Qwen3, Qwen3.5, and Qwen3-Coder-Next as “Needs smoke.” KTransformers/SGLang is therefore a design and diagnostic-performance reference for these targets, not a correctness oracle or merge gate; the exact pinned model/server combination must pass its own smoke test before comparison.

### 7.2 Ideas not adopted in the exact path

KTransformers’ research also describes expert deferral. Deferral deliberately uses an expert result at a later point in the network and can change task scores. It is not exact execution and is outside this specification.

If XENO RT later investigates approximate execution:

- it requires a separate specification;
- it must be explicit opt-in;
- API/status output must disclose approximation;
- quality gates must be model- and task-specific; and
- exact mode must remain the default and fallback.

### 7.3 Portability boundary

KTransformers’ current optimized packages and topology implementation focus on Linux, x86-64, and NVIDIA CUDA. XENO RT supports a broader CPU-only portability contract. Therefore:

- Linux NUMA affinity may be the first optimized implementation;
- Windows and macOS must retain a correct single-node fallback;
- no unconditional `libnuma`, `hwloc`, CUDA, or Python dependency may enter default builds; and
- architecture-specific unsafe code must sit behind a safe trait with documented invariants and scalar tests.

The currently pinned `cudarc` source exposes the raw driver binding for `cuLaunchHostFunc`, but no audited safe wrapper was found. The event/coordinator design is therefore the required initial path; a callback remains an isolated later safety experiment.

### 7.4 Licensing and provenance

KTransformers is Apache-2.0, as is XENO RT, but compatible top-level licenses do not remove per-file provenance obligations. The preferred implementation is a conceptual reimplementation against XENO RT’s types and tests.

Any copied or closely adapted source requires:

1. review of the exact upstream file and commit;
2. preservation of applicable copyright and license notices;
3. an entry in the repository’s NOTICE/provenance record if required;
4. confirmation that the source is not inherited from an incompatible third party; and
5. maintainer approval before merge.

Do not copy from KTransformers’ `archive/` or bundled third-party sources.

## 8. User and system scenarios

### 8.1 CPU-only sparse model

A user starts a supported MoE GGUF on a machine without CUDA. XENO RT loads every expert from GGUF, creates a bounded expert worker pool, routes tokens exactly, groups work by expert, and uses the existing quantized CPU kernels. The OpenAI response is unchanged.

### 8.2 VRAM-constrained hybrid MoE

A user loads a model whose complete expert set does not fit in VRAM. `auto` computes the dense/recurrent/KV/scratch reservation first, uses the remaining configured budget for GPU expert slots, selects a validated static placement, and keeps all other experts host-resident. Each layer runs CPU and GPU experts concurrently and merges every selected result.

### 8.3 Multi-session Qwen3.5

Two clients send unrelated conversations through one loaded Qwen3.5 model. Each `BackendSession` has independent KV and recurrent state. After mutable model scratch is also session-owned or bounded-leased, continuous scheduling can interleave or batch them without serializing the whole model or contaminating one conversation with the other.

### 8.4 Reused hybrid prefix

Two requests share an exact token prefix. A state-aware prefix entry owns the full-attention KV pages and recurrent state produced at the matched boundary. Forked sessions share immutable pages and receive copy-on-write state before either advances.

### 8.5 Unsupported explicit CUDA request

A user explicitly requests CUDA for a MoE tensor layout or recurrent geometry not yet supported. Model loading fails before generation with the unsupported architecture, tensor, layer, and fallback options. XENO RT does not silently switch to a different backend.

## 9. Requirements

### 9.1 Must requirements

| ID | Requirement |
|---|---|
| R1 | Recurrent state is owned by `BackendSession`, never by the shared model object. |
| R2 | Exact mode executes every selected expert and applies every selected routing weight. |
| R3 | CPU-only builds load and run all previously supported GGUF models. |
| R4 | Dense CPU/CUDA, multimodal-chat, and `xrt-vision`/ONNX task paths remain behaviorally unchanged. |
| R5 | The router produces a deterministic ordered selection for equal finite logits. |
| R6 | CPU and GPU completion order cannot change merge order. |
| R7 | Placement changes are published atomically between execution epochs. |
| R8 | A token or batch observes exactly one placement generation per MoE layer. |
| R9 | GPU memory for expert slots, staging, scratch, recurrent state, and graphs is tracked by the existing resource-accounting boundary. |
| R10 | Explicit CUDA never silently falls back after runtime/model loading selects the backend. |
| R11 | `auto` falls back only during runtime/model loading, before request state is mutated. |
| R12 | Cancellation, panic, forward error, and CUDA failure cannot publish partially updated recurrent state; a session is poisoned/reset if rollback cannot be proven. |
| R13 | The OpenAI completion and model schemas do not change. |
| R14 | No new local GGUF inference path introduced here requires network access; existing explicitly selected external-proxy and model-download behavior is unchanged. |
| R15 | Every performance change has a same-hardware benchmark result and correctness evidence. |

### 9.2 Should requirements

| ID | Requirement |
|---|---|
| R16 | Decode transfers activations and outputs, not whole cold-expert weights. |
| R17 | MoE routing and dispatch perform no steady-state heap allocation. |
| R18 | The CPU expert executor honors one runtime-wide thread budget. |
| R19 | Static placement is available before adaptive placement. |
| R20 | Fixed placement and stable buffers can be captured by the existing graph infrastructure. |
| R21 | Recurrent snapshots support a fast same-backend rollback and a durable host form. |
| R22 | Benchmarks distinguish single-stream decode, aggregate decode, and prefill. |
| R23 | Runtime status explains the selected path and why an acceleration mode was rejected. |

### 9.3 May requirements

| ID | Requirement |
|---|---|
| R24 | Linux builds may use optional NUMA discovery and affinity. |
| R25 | A future backend may use a safe CUDA host callback if the driver wrapper supports its lifetime requirements. |
| R26 | A later exact prefill path may stream and repack expert weights layer by layer. |

## 10. Proposed architecture

### 10.1 Architectural rule

Model objects own immutable configuration, tensor metadata, and weights. Sessions own all mutable inference state. Backends own shared execution resources such as devices, worker pools, immutable resident weights, placement managers, and graph caches.

```text
LlamaModel / model descriptor
  immutable layer plan, router weights, logical experts
                  |
                  v
CausalLmBackend ---------------- shared execution resources
  CPU kernels, expert pool, CUDA streams, GPU slots, placement manager
                  |
                  v
BackendSession ----------------- request-owned mutable state
  KV cache, recurrent state, scratch lease, placement snapshot, rollback
```

This ownership rule is a prerequisite for hybrid concurrency, cancellation safety, and state-aware prefix reuse.

### 10.2 Resolved layer plan

At load time, construct an immutable `ModelLayerPlan`:

```rust
enum DecoderLayerKind {
    FullAttention,
    DeltaNet(DeltaNetGeometry),
}

enum FeedForwardKind {
    Dense,
    RoutedMoe(MoeGeometry),
    RoutedMoeWithSharedExpert(MoeGeometry, SharedExpertGeometry),
}

struct LayerPlan {
    decoder: DecoderLayerKind,
    feed_forward: FeedForwardKind,
}
```

The exact names may change during implementation, but the following properties are required:

- layer kind is resolved once, not recomputed from scattered conditionals;
- all dimensions are checked with overflow-safe arithmetic;
- `expert_count > 0`;
- `1 <= expert_used_count <= expert_count`;
- every required logical expert tensor exists and matches the declared geometry;
- shared-expert tensors are either explicitly supported or rejected;
- the Qwen3.5 recurrent/full-attention schedule is verified against checkpoint metadata and tensor layout; and
- unsupported ambiguity produces a load-time error instead of a guessed schedule.

No GGUF extension is required. If existing GGUF metadata is insufficient for a checkpoint family, add a narrowly scoped, documented architecture rule only after validating an exact model revision and fixture.

### 10.3 Session-owned recurrent state

Move `DeltaNetState` out of `LlamaModel`. The model forward boundary must accept mutable state from the caller:

```rust
struct ModelExecutionState<'a> {
    kv: &'a mut dyn KvState,
    recurrent: Option<&'a mut DeltaNetState>,
}
```

This is illustrative rather than a mandatory trait-object design. The implementation may use concrete CPU and CUDA paths to avoid dynamic dispatch.

`BackendSession` gains a recurrent-state member in both CPU and CUDA variants. It may remain uninitialized until the first fallible generation preparation:

```rust
enum SessionRecurrentState {
    None,
    Uninitialized(DeltaNetStateDescriptor),
    Cpu(DeltaNetState),
    #[cfg(feature = "cuda")]
    Cuda(CudaDeltaNetState),
}
```

The model no longer exposes global clear/save/restore behavior. Backend state methods become session-specific:

```rust
fn prepare_session_state(
    &self,
    session: &mut BackendSession,
) -> Result<()>;

fn reset_session_state(&self, session: &mut BackendSession);

fn snapshot_session_state(
    &self,
    session: &BackendSession,
) -> Result<BackendStateSnapshot>;

fn restore_session_state(
    &self,
    session: &mut BackendSession,
    snapshot: &BackendStateSnapshot,
) -> Result<()>;
```

The current public `Runtime::new_session() -> Session`, `Runtime::new_session_with_cache_mode(...) -> Session`, and `Session::reset() -> ()` contracts remain unchanged. Checked recurrent geometry and worst-case per-session bytes are computed at model load. State/resource admission and allocation happen in `prepare_session_state` before token 0, where failure can still be returned without mutating KV or recurrent state. CPU reset zeroes owned buffers. CUDA reset must be an infallible logical discard/reinitialize operation; it must not hide a fallible device synchronization or allocation inside the public `reset()` call.

The durable host form replaces the current untyped `Vec<Option<(Vec<f32>, Vec<f32>)>>` alias with a validated snapshot equivalent to:

```rust
struct BackendStateSnapshot {
    version: u32,
    model_geometry_fingerprint: [u8; 32],
    position: u64,
    layers: Box<[Option<DeltaNetLayerSnapshot>]>,
}

struct DeltaNetLayerSnapshot {
    conv_state_f32: Box<[f32]>,
    recurrent_state_f32: Box<[f32]>,
}
```

The exact fingerprint representation may differ, but `position`, format version, layer count/presence, and geometry identity are mandatory. Conversion between durable `u64` position and the current in-memory `usize` is checked. Restore first validates all metadata and checked payload lengths, then mutates the destination; malformed input returns an error without panicking or partially restoring. Restore also verifies that the paired KV boundary and accepted position agree, or performs both changes inside one transaction. The current per-layer F32 vectors may be reused as payload storage, but the current alias cannot be retained unchanged because it omits `position` and cannot validate geometry safely.

A later CUDA implementation separates:

- a durable host snapshot for prefix ownership and differential tests; and
- a fast device-local checkpoint/journal for speculative verification.

After first successful preparation, buffers are reused for the session. No operation acquires a model-global recurrent-state lock.

`ForwardScratch` and `BatchScratch` are a separate concurrency gate. Phase 1 may retain their current model locks together with exclusive hybrid scheduling. Before two hybrid sessions may enter model forward concurrently, all mutable scratch touched by those paths must move into the owning session or a bounded backend pool whose lease spans the complete forward call. Removing only `deltanet_state` does not justify removing the exclusive guard.

### 10.4 Canonical exact router

Create one routing implementation shared by all CPU forward paths and used as the semantic reference for CUDA:

1. compute router logits in F32;
2. reject a row with no finite candidate;
3. select the highest `expert_used_count` logits to establish the exact top-k
   boundary `b`;
4. treat finite candidates in the closed interval
   `[b - 1e-5, b + 1e-5]` as one numerical boundary tie, retain candidates
   clearly above that interval, and fill the remaining slots by descending
   logical expert ID;
5. keep clearly separated selections ordered by descending logit and the
   boundary-tie selections ordered by descending logical expert ID;
6. apply max-subtracted F32 softmax over selected logits only; and
7. emit fixed-capacity IDs and weights into caller-provided scratch.

The implementation must characterize the current router on real fixtures before adopting the tie rule. If the current path differs only in undefined tie or NaN behavior, update all paths together and add a compatibility test. Normal finite inputs outside the registered boundary band must remain identical. The `1e-5` band and descending-ID rule are explicit compatibility semantics selected from the pinned full-profile evidence; changing either requires the same evidence and spec-revision process as any other numerical threshold. Wider `1e-4` and `2e-4` bands were tested and rejected because they changed upstream state enough to create new high-impact route divergences.

Avoid a heap-allocated all-expert index vector. For small `top_k`, use a fixed-capacity insertion selection or bounded heap. For batched prefill, produce a contiguous `[token][top_k]` result and a reusable expert-to-token grouping index.

### 10.5 MoE descriptors and execution plan

Introduce immutable and per-execution structures equivalent to:

```rust
struct MoeLayerDescriptor {
    layer_index: usize,
    expert_count: usize,
    selected_per_token: usize,
    hidden_size: usize,
    intermediate_size: usize,
}

struct ExpertPlacementSnapshot {
    generation: u64,
    logical_to_gpu_slot: Box<[Option<u16>]>,
    gpu_slot_to_logical: Box<[u16]>,
}

struct MoeRoutingRow {
    logical_ids: SmallFixedIds,
    weights: SmallFixedWeights,
}

struct MoeExecutionPlan<'a> {
    layer: &'a MoeLayerDescriptor,
    placement: &'a ExpertPlacementSnapshot,
    routes: &'a [MoeRoutingRow],
    cpu_work: WorkSpan,
    gpu_work: WorkSpan,
}
```

Requirements for these structures:

- no per-token ownership of expert weights;
- no allocation in steady-state planning;
- logical expert IDs remain authoritative;
- a physical slot is never mistaken for a logical ID;
- a placement snapshot is immutable while referenced;
- all GPU remap arrays use a checked integer width; and
- an execution plan cannot outlive its placement snapshot or scratch lease.

### 10.6 Exact heterogeneous MoE flow

For a CUDA-resident non-expert layer state, one MoE layer executes as follows:

```text
GPU router
    |
    +--> deterministic top-k and placement remap
    |
    +--> hot expert kernels on GPU stream ------------------+
    |                                                       |
    +--> hidden activation D2H into pinned staging           |
           -> bounded CPU expert jobs -> pinned output H2D --+
                                                            |
                             ordered weighted merge <--------+
                                      |
                                  residual path
```

Detailed sequence:

1. Acquire a stable scratch/staging lease and placement snapshot.
2. Compute router logits.
3. Produce canonical selected logical IDs and routing weights.
4. Partition selected `(token, expert)` pairs by physical tier.
5. For any CPU work, copy each required normalized hidden row once into pinned host staging.
6. Submit CPU work only after the copy-complete event is observable.
7. Launch GPU-resident selected experts on an independent stream where dependencies permit.
8. CPU workers group rows by expert, reuse packed weights and scratch, and write one result per logical selection.
9. Upload CPU results or partials to a stable device buffer.
10. Wait for both tiers.
11. Merge in canonical top-k order, never completion order.
12. Release the scratch lease and placement snapshot.

The initial exact path retains one contribution per logical selection so CPU/GPU partitioning cannot silently change the reduction sequence. A later fused or pre-aggregated partial is eligible only after it proves the same canonical logical merge semantics and passes the numerical gates for every supported path.

The initial eager implementation may use explicit CUDA events plus a backend coordinator thread. It must not block correctness on `cudaLaunchHostFunc` support. A host callback can be added only after proving:

- a narrow wrapper over the raw driver binding has been implemented and audited for the required operation;
- callback captures outlive asynchronous execution;
- callbacks do not call forbidden CUDA APIs;
- cancellation cannot free captured memory; and
- panic cannot unwind across FFI.

If all selected experts are CPU-resident or all are GPU-resident, the same plan degenerates to one tier without special semantic behavior.

### 10.7 CPU expert executor

Add a runtime-owned `ExpertWorkerPool` rather than spawning work per token.

The pool must:

- use a bounded queue;
- obtain its worker budget from the same runtime-wide CPU budget used by other kernels;
- prevent nested oversubscription with Rayon or the existing spin pool;
- keep reusable per-worker activation and accumulator scratch;
- group batched tokens by expert to reuse weight/cache locality;
- expose synchronous and submit/join interfaces;
- catch worker failure and convert it to a request error;
- support clean shutdown; and
- work as a single topology node on every platform.

Introduce a `CpuTopology` abstraction:

```rust
struct CpuTopology {
    nodes: Vec<CpuNode>,
    logical_cpus: usize,
    affinity_supported: bool,
}
```

The first portable implementation may report one node. An optional Linux implementation can discover NUMA nodes and bind workers. Windows can later use processor-group and NUMA APIs. If discovery or affinity fails in `auto`, log the reason and continue with the portable pool. `strict` mode may fail for benchmark diagnosis.

Expert-weight placement policies:

- preserve the GGUF mapping as the canonical storage;
- optionally create packed CPU expert weights once at load;
- allocate/touch a packed expert on its assigned NUMA node;
- prefer local workers;
- allow work stealing only after a measured local queue threshold; and
- record remote steals and bytes in benchmark telemetry.

Reuse existing SIMD/quantized matvec kernels. Add MoE-specific fused gate/up projection and down-projection entry points only when a benchmark proves they reduce memory traffic. AMX is a later optional specialization, not an MVP dependency.

### 10.8 GPU expert slots

Add a resident MoE resource layer built on the existing resident matrix abstractions:

```rust
struct ResidentExpertSlot {
    logical_expert: Option<u16>,
    gate: ResidentQuantMatrix,
    up: ResidentQuantMatrix,
    down: ResidentQuantMatrix,
    bytes: u64,
}

struct ResidentMoeLayer {
    router: ResidentQuantMatrix,
    slots: Vec<ResidentExpertSlot>,
    placement_device_map: CudaBytes,
}
```

`ResidentQuantMatrix` is the existing private resident-matrix enum in `xrt-runtime/src/backend.rs`; implementation should extract it to a focused runtime module or make it `pub(crate)` rather than inventing a parallel matrix hierarchy. `CudaBytes` is the existing CUDA byte-buffer boundary. Its allocation must remain alive and unmoved for a graph epoch while map contents are updated through checked copies.

Implementation requirements:

- the dense, attention, recurrent, KV, graph, and scratch reservations are accounted before expert budget calculation;
- slots are allocated through the central GPU allocation/resource boundary planned in `GPU_RUNTIME_ACCELERATION_SPEC.md`;
- every supported expert matrix format uses an already validated resident matrix implementation or adds format-specific parity tests;
- the loader never silently dequantizes a model into an unbounded VRAM representation;
- partial slot construction is rolled back on error;
- placement maps and work buffers have stable addresses for one graph epoch;
- a slot swap uploads all three projections and validates completion before publication; and
- old placement snapshots remain alive until no execution references them.

The initial exact hybrid decode path does not transfer expert weights per token. A cold expert runs on CPU.

### 10.9 GPU expert kernels

Implement the smallest correct kernel surface first:

1. router matvec/GEMM through an existing resident matrix path;
2. canonical top-k reference on CPU for parity, followed by a CUDA top-k/remap kernel when profiling justifies it;
3. selected-expert gate/up projection;
4. SiLU and gate/up product;
5. selected-expert down projection; and
6. ordered weighted merge.

The eager MVP may reuse existing matvec calls with stable buffers. Fused selected-expert kernels follow only after end-to-end parity.

For batched prefill/decode:

- group `(token, expert)` pairs by physical GPU slot;
- retain each token’s logical top-k order for final merge;
- use GEMM above a measured row threshold and GEMV below it;
- bound all index buffers by `batch_tokens * selected_per_token`; and
- reject arithmetic overflow before allocation or launch.

New `.cu` sources must be compiled in `nvidia/cuda:12.8.1-devel-ubuntu22.04`, committed as generated PTX, and checked byte-for-byte in `.github/workflows/cuda.yml`. Runtime NVRTC remains unnecessary.

### 10.10 Qwen3.5 CUDA recurrent state

Represent each recurrent layer with stable session-owned device buffers:

```rust
struct CudaDeltaNetLayerState {
    conv_state: CudaF32Buffer,
    recurrent_state: CudaF32Buffer,
    geometry: DeltaNetGeometry,
}

struct CudaDeltaNetState {
    layers: Vec<Option<CudaDeltaNetLayerState>>,
    position: u64,
    transaction: Option<CudaStateTransaction>,
}
```

Phase 4 uses F32 recurrent state as the required parity baseline. Lower-precision recurrent state is a separate, later quality-gated optimization, not an implementation choice left open by this specification.

Implement in this order:

1. single-session, single-token decode;
2. prompt extend/prefill;
3. multiple request-indexed states in the scheduler;
4. target verification with rollback; and
5. state-aware prefix fork.

Decode and extend are separate kernel contracts. Do not disguise extend as repeated decode without measuring it. The state update must be transactional:

- write into a next-state buffer or journal;
- publish the new state only after all layer work for the accepted token succeeds;
- discard or restore it on cancellation/error; and
- never leave a session at an unreported intermediate token boundary.

Speculative target verification remains disabled for CUDA hybrid sessions until a fast checkpoint passes exact rollback tests. The current full host clone may remain a correctness fallback for small fixtures but must not be the default performance path for large states.

### 10.11 Static and adaptive placement

Placement evolves through three levels:

1. **Uniform static:** spread the GPU slot budget across MoE layers.
2. **Profiled static:** load a placement manifest produced by an offline routing trace for the exact model revision and workload.
3. **Adaptive:** maintain per-layer routing-frequency statistics and update between safe execution epochs.

Adaptive placement is not part of the initial default. When implemented:

- collect aggregate logical-expert counts or an EWMA;
- never log prompt text, token IDs, or activations by default;
- make decisions only at a request/prefill boundary initially;
- use hysteresis, minimum residency duration, and a maximum moves-per-update budget;
- upload replacement weights on a staging stream;
- update logical/physical maps in stable buffers;
- publish a new immutable placement snapshot only after upload completion;
- increment the graph/placement epoch;
- drain or invalidate graph executions tied to an old epoch; and
- expose update count, bytes, duration, and churn.

Frequency is not universally optimal. Every placement benchmark must record model revision, quantization, prompt set, batch/concurrency, CPU topology, GPU, slot budget, and placement policy.

### 10.12 CUDA graphs and stream coordination

Correct eager execution is the prerequisite. Graph work proceeds in this order:

1. capture all-GPU fixed-placement MoE decode;
2. capture the GPU portion of fixed-placement hybrid decode with explicit event joins;
3. investigate host-callback submission/synchronization; and
4. add batch-shape graph variants.

A graph cache key must include:

- model identity;
- device;
- batch/shape bucket;
- quantization/kernel path;
- placement generation or graph epoch;
- recurrent layer geometry;
- KV mode; and
- scratch/resource generation.

Replaying a graph with a stale placement map, freed expert slot, changed state pointer, or changed scratch address is a correctness error. The runtime must eagerly fall back and record a reason rather than replaying it.

### 10.13 State-aware prefix cache

Hybrid prefix reuse associates:

- token-prefix identity;
- full-attention KV pages;
- recurrent convolution state;
- recurrent state matrices;
- accepted token position;
- model/config/tokenizer identity; and
- state geometry/version.

The first state-aware implementation may materialize recurrent state per entry. Copy-on-write follows after correctness. The final design should:

- share immutable KV pages;
- fork recurrent state using a backend-specific COW or copy operation;
- protect in-use entries from eviction;
- account host and device state bytes;
- reject a geometry/version mismatch;
- clear both KV and recurrent state on reset; and
- never match a prefix ending inside an uncommitted speculative transaction.

The existing hybrid prefix bypass remains until all these invariants pass tests.

### 10.14 Layerwise prefill

Layerwise prefill is a late exact optimization for models whose expert weights do not fit in VRAM.

For one layer:

1. group prompt rows by selected expert;
2. stage the next expert or expert tile from memory-mapped/pinned host storage;
3. transfer into a bounded staging slot;
4. repack only if the resident kernel requires it;
5. execute the current expert group;
6. overlap transfer/repack for the next group through double buffering; and
7. release/reuse the slot at the layer boundary.

Requirements:

- decode hot/cold placement remains independent;
- all selected experts still run;
- transfer and repack bytes are reported separately;
- staging obeys the central VRAM budget;
- admission budgets the documented worst case, including one complete MoE layer plus temporary/repack and double-buffer storage when the selected kernel requires that working set;
- failed transfer cannot corrupt a resident decode placement; and
- the mode enables only when it improves measured TTFT/prefill or allows a model to run within the declared memory limit.

## 11. Component and file plan

Exact file boundaries may be adjusted to preserve crate layering, but responsibilities must not move back into one monolithic backend file.

### 11.1 `xrt-models`

Create or extract:

- `crates/xrt-models/src/moe.rs`
  - MoE geometry and immutable tensor descriptors;
  - canonical routing semantics and scalar reference;
  - CPU selected-expert semantic reference.
- `crates/xrt-models/src/hybrid_state.rs`
  - `DeltaNetGeometry`;
  - CPU `DeltaNetState`;
  - durable state snapshot representation;
  - reset/snapshot/restore validation.

Modify:

- `crates/xrt-models/src/llama.rs`
  - remove model-owned `deltanet_state`;
  - resolve and store `ModelLayerPlan`;
  - pass caller-owned recurrent state through forward paths;
  - retain exclusive scheduling until `ForwardScratch` and `BatchScratch` are moved to session ownership or a bounded lease boundary;
  - route every MoE path through one helper;
  - keep tensor naming and GGUF validation centralized.
- `crates/xrt-models/src/lib.rs`
  - export only the types required by `xrt-runtime`.

Do not expose execution-placement policy from `xrt-models`.

### 11.2 `xrt-kernels`

Create:

- `crates/xrt-kernels/src/cpu/moe.rs`
  - allocation-free routing selection where appropriate;
  - expert-group execution over existing quantized kernels;
  - fused MoE kernels justified by benchmarks.
- `crates/xrt-kernels/src/cpu/topology.rs`
  - portable topology model;
  - optional platform discovery and affinity adapter.
- `crates/xrt-kernels/src/cpu/expert_pool.rs`
  - bounded pool, scratch, submit/join, failure propagation.

Modify the existing thread-pool boundary so dense and expert work share an explicit budget rather than nesting unconstrained pools.

### 11.3 `xrt-runtime`

Create:

- `crates/xrt-runtime/src/moe.rs`
  - placement snapshots;
  - execution-plan construction;
  - heterogeneous coordinator;
  - metrics.
- `crates/xrt-runtime/src/expert_placement.rs`
  - uniform/profiled/adaptive policies;
  - placement publication and generation lifecycle.
- `crates/xrt-runtime/src/recurrent_state.rs`
  - backend session state wrapper;
  - transaction/checkpoint interface;
  - prefix integration.

Modify:

- `crates/xrt-runtime/src/backend.rs`
  - session-owned state APIs;
  - resident expert resources;
  - exact hybrid capability checks;
  - clear errors and resource accounting.
- `crates/xrt-runtime/src/resident_tensor.rs`
  - extract or expose the existing `ResidentQuantMatrix` as a focused `pub(crate)` resident-weight boundary shared by dense and expert resources.
- `crates/xrt-runtime/src/session.rs`
  - remove model-global state calls;
  - lift exclusive hybrid scheduling only after isolation tests;
  - transactional cancellation/reset.
- scheduler/batch modules
  - batch request-indexed recurrent states;
  - include hybrid support only after shape/state gates.
- prefix-cache modules
  - state-aware key, fork, COW, eviction, accounting.
- GPU resource/scratch modules
  - central arena allocations for expert slots, staging, state, and graphs.

### 11.4 `xrt-cuda`

Create or extend:

- resident expert-slot storage built from current resident matrices;
- pinned double-buffer staging primitives;
- stream/event primitives needed by the coordinator;
- selected-expert and ordered-merge CUDA kernels;
- DeltaNet decode and extend kernels;
- recurrent-state device allocation/copy/checkpoint helpers;
- generated PTX and CUDA-disabled stubs.

Every public CUDA type must have a default-feature stub or remain entirely feature-gated so a CPU build does not link CUDA.

### 11.5 CLI, server, benchmark, and CI

Modify:

- `xrt bench` JSON schema with additive MoE and recurrent-state fields;
- `/v1/runtime/status` with additive capability/effective-mode fields;
- Criterion benches for router, expert dispatch, and state checkpointing;
- `.github/workflows/moe-validation.yml` with opt-in real MoE/Qwen3.5 parity
  and prompt/route quality profiles;
- `scripts/safe-cuda-check.ps1` for compile-only kernel and state tests;
- safe smoke scripts with explicit model, expert budget, mode, token bound, timeout, and confirmation; and
- integration tests for OpenAI schema invariance.

## 12. Configuration and API compatibility

### 12.1 Internal configuration

Add one structured runtime configuration, parsed once:

```rust
struct MoeRuntimeConfig {
    acceleration: MoeAcceleration,
    gpu_expert_budget_bytes: Option<u64>,
    placement: MoePlacementPolicy,
    placement_manifest: Option<PathBuf>,
    placement_update_tokens: u64,
    layerwise_prefill: bool,
    numa: NumaPolicy,
}
```

Proposed environment mappings:

| Variable | Values | Default during rollout |
|---|---|---|
| `XRT_MOE_ACCELERATION` | `legacy`, `auto`, `cpu`, `hybrid`, `gpu` | `legacy`, then gated `auto` |
| `XRT_MOE_GPU_EXPERT_BUDGET_BYTES` | positive integer | unset; required for explicit `hybrid`/`gpu`, derived from the bounded residual budget in opt-in `auto` |
| `XRT_MOE_PLACEMENT` | `uniform`, `profiled`, `adaptive` | `uniform` |
| `XRT_MOE_PLACEMENT_MANIFEST` | local path | unset; required for `profiled` |
| `XRT_MOE_PLACEMENT_UPDATE_TOKENS` | positive integer | implementation default |
| `XRT_MOE_LAYERWISE_PREFILL` | `on`/`off`, `true`/`false`, `enabled`/`disabled`, or `1`/`0` | `off`; exact explicit-hybrid opt-in |
| `XRT_MOE_NUMA` | `auto`, `off`, `strict` | `auto` |
| `XRT_CUDA_GRAPH` | `auto`, `on`, `off` and equivalent booleans | `auto`; validated full-GPU MoE shapes are eligible, while hybrid MoE subgraphs require explicit `on` until their performance gate passes |

All modes in this specification are exact. Do not add an “approximate” value.

`BackendKind` remains the existing top-level selector (`auto`, `cpu`, `cuda-resident`, or `external-openai`); hybrid execution is not a new `BackendKind`. `MoeAcceleration` is a subordinate local-execution policy resolved once during runtime/model loading:

Configuration-source precedence follows the existing entrypoint contract: an explicit programmatic or CLI value overrides its environment variable, which overrides the documented default. After resolving each field, the backend/MoE pair is validated by the table below; contradictory resolved values return an actionable load/startup error.

| `BackendKind` | Allowed/effective MoE behavior |
|---|---|
| `cpu` | `legacy`, `cpu`, or `auto`; `auto` chooses a validated CPU implementation. Explicit `hybrid` or `gpu` is a configuration error. |
| `cuda-resident` | `auto`, `hybrid`, or `gpu` for a supported model. `legacy`/`cpu` on a MoE model conflicts with explicit CUDA and errors instead of falling back. |
| `auto` | `legacy`/`cpu` forces the corresponding CPU path; explicit `hybrid`/`gpu` requires that exact CUDA mode and errors if unavailable; `auto` may select CUDA, hybrid, optimized CPU, or legacy CPU from the validated capability table. |
| `external-openai` | Local MoE knobs do not apply. Supplying any non-default local MoE knob is rejected at startup rather than silently ignored. |

For a non-MoE model, default/`auto` MoE policy has no effect; an explicit `hybrid` or `gpu` MoE policy is rejected as unsupported. `legacy` is the rollback switch for the existing CPU MoE implementation. `cpu` means the optimized CPU executor. `hybrid` requires CUDA plus at least one CPU and one GPU physical expert assignment. `gpu` requires every logical expert in every routed layer to be resident and GPU-executable for the placement epoch, not merely the experts selected by observed requests. The selected backend and MoE plan are fixed before `Runtime` construction returns; later lazy per-session state admission may fail before token 0 but cannot change that plan.

An explicit `hybrid`/`gpu` request requires an explicit positive expert budget. In `auto`, the resource manager may derive the expert budget only after reserving dense/recurrent/KV/graph/scratch requirements and applying its safety margin. A non-default knob that is inapplicable to the resolved mode is rejected rather than silently ignored.

`profiled` requires a versioned manifest bound to the exact model identity. Parsing must validate schema version, model/config hash, quantization, layer count, logical layer and expert IDs, uniqueness, integer bounds, and the configured slot/byte budget before any upload. Unknown fields follow an explicit schema-version policy; malformed or mismatched manifests fail load without partial placement.

CLI equivalents may be added to `xrt bench` first. User-facing serve/generate flags should be added only when naming and status behavior are stable.

Because the current server has no inbound authentication boundary, this specification adds no placement, manifest-path, expert-budget, or acceleration controls to chat/completion requests or `/v1/runtime/load`. For this work they are limited to process-start configuration and local benchmark inputs. Remotely mutable operator controls require a separate authenticated control-plane design.

### 12.2 OpenAI surface

No fields are added to or removed from:

- chat completion requests;
- chat completion chunks/responses;
- legacy completion requests and responses;
- usage objects; or
- model list entries required for compatibility.

Existing multimodal `image_url` chat handling and the XENO-specific `/v1/images/remove-background` request/response contract are also unchanged.

Backend telemetry belongs in the existing XENO-specific runtime-status and benchmark surfaces. Unknown internal configuration must never leak into OpenAI response bodies.

### 12.3 Additive runtime status

Example conceptual fields:

```json
{
  "moe": {
    "supported": true,
    "effective_mode": "hybrid",
    "exact": true,
    "placement": "uniform",
    "placement_generation": 3,
    "gpu_expert_slots": 24,
    "gpu_expert_bytes": 8589934592,
    "gpu_hit_rate": 0.71
  },
  "hybrid_state": {
    "owner": "session",
    "backend": "cuda",
    "bytes_per_session": 19922944,
    "prefix_cache_supported": false,
    "speculative_rollback_supported": false
  }
}
```

These values are illustrative. Do not hard-code the shown byte count.

Status must distinguish requested and effective mode and provide a bounded rejection/fallback reason.

## 13. Correctness and numerical semantics

“Exact” means graph-level equivalence, not guaranteed bitwise equality between scalar CPU, vector CPU, and CUDA floating-point instructions.

The following semantics are fixed:

1. the same checkpoint and quantized weights are used;
2. router logits are evaluated over all logical experts;
3. top-k count and IDs follow the canonical router;
4. every selected expert executes;
5. routing weights are normalized over the selected set;
6. shared experts, when supported, always execute according to the model graph;
7. merge order follows canonical selected order;
8. residual, normalization, and state-update ordering match the CPU reference; and
9. recurrent state commits only for accepted tokens.

Cross-device validation uses layered tolerances:

- router selection IDs must match exactly;
- F32 CPU optimized router logits: maximum absolute error `<= 1e-5` against scalar reference;
- CUDA router logits: maximum absolute error `<= 2e-4` for the same router
  input and identical selected IDs outside the canonical `1e-5` boundary rule;
- selected routing weights: absolute error `<= 2e-5`;
- kernel-level output tolerance is declared per dtype/quantization fixture;
- end-to-end greedy tokens must match on the required fixture corpus; and
- model-level logit drift and perplexity gates in Section 18 must pass.

Phase 0 must run the measurement harness against the unchanged reference path, record estimator noise, and confirm that these numerical thresholds and the Section 18 corpus gates are measurable for each pinned dtype/quantization fixture. Any threshold change requires an explicit spec revision with evidence before implementation comparison begins; implementers must not relax a gate after seeing an accelerated-path result.

If a real checkpoint exposes near-tied router logits that select different experts under allowed arithmetic drift, the CUDA router must use a higher-precision or deterministic fallback. The implementation may not waive expert-ID parity as “close enough.”

Execution clarification: independent long CPU/CUDA forward passes are not
required to remain bit-identical after accumulated hidden-state drift. Section
18.2 registers bounded route-agreement and end-to-end output gates for that
case. Route drift must remain instrumented and may not be silently omitted.

## 14. Error handling and fallback

### 14.1 Load-time and pre-token preparation errors

Return structured, actionable errors for:

- missing or inconsistent expert metadata;
- invalid expert/top-k counts;
- absent or malformed router/expert tensors;
- unsupported shared-expert layout;
- ambiguous recurrent layer schedule;
- unsupported quantization for the requested tier;
- integer overflow in state, slot, scratch, or staging geometry;
- GPU expert budget below the minimum viable reservation;
- pinned-memory allocation failure;
- recurrent state larger than configured per-session limits; and
- incompatible placement manifest identity.

Configuration, tensor geometry, checked byte requirements, and backend/MoE mode compatibility are validated during runtime/model loading. Concrete per-session recurrent buffers, pinned staging, and other fallible session resources may be admitted and allocated lazily by `prepare_session_state`, but that preparation must complete before token 0 and before KV or recurrent state is mutated.

### 14.2 Runtime errors

Handle:

- bounded queue saturation;
- worker panic/failure;
- CUDA launch, event, stream, or copy failure;
- placement-generation mismatch;
- graph epoch mismatch;
- recurrent state transaction failure;
- cancellation during CPU/GPU overlap;
- prefix state geometry mismatch; and
- session reset during outstanding work.

If a forward error occurs after mutation and the runtime cannot prove restoration to the pre-token KV and recurrent boundary, it marks the session poisoned and rejects further generation until reset. Error injection tests must cover every state-commit stage.

### 14.3 Fallback rules

1. `auto` may choose legacy CPU, optimized CPU, full GPU, or hybrid only during runtime/model loading.
2. Explicit `cpu`, `hybrid`, or `gpu` returns an error if unsupported.
3. A placement miss is normal hybrid behavior and runs the logical expert on CPU.
4. A GPU execution failure is not a placement miss.
5. After a token begins mutating KV or recurrent state, the runtime must either roll back both successfully or poison the session and require reset.
6. Do not silently re-run a failed GPU expert on CPU unless the whole token has a proven pre-token checkpoint and deterministic replay path.
7. Error messages may include layer, tensor role, dtype, shape, and backend, but not prompt contents.

## 15. Concurrency, cancellation, and resource lifetime

### 15.1 Lock ordering

Document and test one lock order:

1. session execution permit;
2. standard-MoE graph execution gate, when applicable;
3. placement snapshot reference;
4. architecture-specific CUDA capture gate, when applicable;
5. scratch/staging lease;
6. recurrent transaction;
7. backend queue submission.

Never hold a placement-manager write lock while waiting on CPU work or CUDA events.

### 15.2 Cancellation

Cancellation behavior:

- stop submitting new layer work;
- allow already submitted CPU/CUDA operations to reach a safe join;
- discard the pending recurrent transaction;
- restore KV length and recurrent state to the last committed token;
- release staging, graph, and placement references; and
- return the existing cancellation behavior to the API layer.

Today cancellation is observed between model invocations. That behavior remains the Phase 1 baseline. Intra-forward cancellation may be added only with the transaction and safe-join behavior above; otherwise submitted work completes to a token boundary and its result is discarded.

### 15.3 Session destruction

A session cannot free device state while a graph, stream callback, coordinator task, or worker references it. Use owning leases/fences rather than timing assumptions. Backend shutdown drains queues before device and pool destruction.

The 2026-07-20 concurrency-8 Qwen3 soak proved that synchronization before
`Session::drop` is insufficient if the guard is released before Rust destroys
the session-owned CUDA fields. Safe destruction therefore consumes the backend
session and holds its graph gates through synchronization and actual field
destruction. Standard-MoE graph-enabled execution holds a shared backend mutex
for its complete allocation/capture window, preventing another session's
`cuMemFreeAsync` from entering the captured cudarc stream.

### 15.4 Batching

Batching hybrid sessions becomes eligible only after:

- recurrent state is request-indexed;
- per-request position and layer state are explicit;
- route buffers retain token/session association;
- one failed row cannot partially commit other sessions without a defined policy; and
- scheduler fairness and cancellation tests pass.

The first batch implementation may use an all-or-nothing batch transaction. Per-session partial acceptance is a later optimization.

## 16. Observability

Add low-cardinality counters and benchmark fields:

### 16.1 MoE

- routed tokens and selected expert calls;
- CPU and GPU expert calls;
- GPU placement hits/misses and hit rate;
- placement generation and update count;
- placement upload bytes/time and churn;
- router, dispatch, CPU expert, GPU expert, join, merge, and layer durations;
- activation D2H and result H2D bytes/time;
- CPU queue depth/wait and worker utilization;
- NUMA local/remote calls and steals when available;
- GPU slot bytes and staging bytes;
- eager/graph execution and graph fallback reason.

### 16.2 Recurrent state

- state bytes per session and total;
- reset, durable snapshot, fast checkpoint, commit, and rollback counts;
- snapshot/rollback bytes and time;
- hybrid sessions active/queued;
- prefix hits/misses rejected for recurrent reasons; and
- speculative decoding disabled reason.

### 16.3 Privacy and cardinality

Do not record:

- prompt text;
- generated text;
- token IDs;
- activation values; or
- unbounded per-expert metric labels in production.

Per-expert histograms may appear in an explicitly requested benchmark artifact, keyed by logical integer ID and bounded by model expert count.

Production hot-path observability is limited to low-overhead bounded counters and coarse totals. Per-layer timing, transfer timing, queue tracing, and routing histograms are benchmark/profile opt-ins so measurement instrumentation does not become the decode bottleneck.

## 17. Benchmark plan

### 17.1 Measurement rules

Every result records:

- XENO RT commit and dirty status;
- model URI/path label, upstream revision, file SHA-256, architecture, quantization, and size;
- tokenizer identity;
- CPU model, logical/physical cores, ISA, socket/NUMA topology, and memory;
- GPU model, driver, VRAM, and CUDA feature state;
- OS and power mode;
- backend and effective MoE/recurrent configuration;
- expert slot budget and placement generation/policy;
- prompt/output lengths, batch, and concurrency;
- warmup count, sample count, median, p95, and variability;
- host RSS and tracked/driver VRAM;
- transfer bytes and CPU/GPU utilization; and
- quality/parity result.

Use at least five warmups for stable decode paths and ten measured runs unless run time makes that impractical. A benchmark note must justify fewer samples.

Every pre/post comparison uses an alternating `ABBA` order or a seeded randomized order on the same machine to reduce thermal and drift bias. Report the estimator and a 95% confidence interval (paired bootstrap by default). A no-regression gate passes only when the interval's upper regression bound is within the allowed limit; an improvement gate passes only when its lower bound reaches the claimed improvement. Otherwise the result is inconclusive and the path remains opt-in.

### 17.2 Benchmark layers

1. **Microbenchmarks**
   - top-k at representative expert counts and `top_k`;
   - CPU expert gate/up/down by quantization;
   - token grouping by expert;
   - pinned D2H/H2D for representative hidden sizes;
   - GPU selected-expert kernels;
   - state reset/snapshot/checkpoint/rollback;
   - placement map update.
2. **Synthetic integration**
   - tiny deterministic MoE;
   - tiny four-layer Qwen3.5-style full/recurrent schedule;
   - tiny hybrid-MoE with forced CPU/GPU split.
3. **Real model**
   - one redistributable Qwen3 MoE GGUF;
   - one redistributable Qwen3.5 hybrid or hybrid-MoE GGUF;
   - a larger capacity/low-active-parameter model on the self-hosted runner when RAM permits.
4. **API**
   - streaming and non-streaming chat;
   - cancellation;
   - concurrent sessions;
   - status and error behavior.

### 17.3 Throughput matrix

For each real fixture report:

| Workload | Prompt | Output | Concurrency |
|---|---:|---:|---:|
| Short interactive | 32 | 128 | 1, 2, 4, 8 |
| Normal chat | 512 | 256 | 1, 2, 4 |
| Long context | 4096 or model-safe bound | 128 | 1, 2 |
| Prefill-only | 128, 512, 2048, 8192 where supported | 1 | 1, batched |

Report separately:

- TTFT;
- prefill tokens/s;
- single-stream decode tokens/s;
- aggregate decode tokens/s;
- per-request p50/p95 latency; and
- peak RSS/VRAM.

Never present aggregate or prefill throughput as single-stream decode throughput.

### 17.4 Baselines

Compare on the same machine and model where possible:

1. XENO RT legacy CPU MoE;
2. XENO RT optimized CPU MoE;
3. XENO RT exact static hybrid;
4. XENO RT all-GPU when the same model fits;
5. XENO RT adaptive/layerwise modes when implemented;
6. a Phase 0-pinned `llama.cpp` commit/container, build flags, runtime flags, and compatible GGUF; and
7. pinned KTransformers/SGLang as an external OpenAI endpoint for a compatible checkpoint.

External systems are diagnostic references, not merge gates. Record tokenizer, quantization, serving defaults, and unsupported differences. Use the existing external-OpenAI benchmark path where its usage accounting is comparable. Run and record a smoke test of the exact pinned KTransformers/SGLang/model combination before benchmarking; the upstream “Needs smoke” support status cannot substitute for XENO RT correctness evidence.

### 17.5 Performance gates

- Session-state refactor: no more than `2%` median regression on legacy hybrid/MoE CPU decode.
- Dense protection: no more than `1%` median regression and no more than `3%` p95 regression on representative dense CPU and CUDA decode.
- Optimized CPU path: no more than `2%` regression on any required MoE fixture; it becomes default only with a statistically stable end-to-end gain or at least a `10%` reduction in the constrained peak memory metric (RSS for CPU pressure or tracked/driver VRAM for GPU pressure), while reporting both RSS and VRAM and meeting latency/quality gates.
- Exact static hybrid: enable in `auto` for a hardware/model class only when it beats the best viable XENO RT non-hybrid baseline by at least `15%` median decode throughput under the same VRAM limit, with no more than `5%` p95 latency regression.
- CUDA graph path: must beat eager on its eligible shape and never regress ineligible shapes.
- Layerwise prefill: enable only when it either permits execution under the declared VRAM budget or improves median prefill throughput/TTFT by at least `10%`.
- Adaptive placement: must improve the target workload over uniform static placement after including update cost, and must not degrade a required workload by more than `2%`.

If a path fails its performance gate, keep it opt-in or remove it. Correctness alone does not justify a slower default.

### 17.6 Hundreds-TPS objective

Track three explicit stretch lines:

1. `>=100 prefill tokens/s`;
2. `>=100 aggregate decode tokens/s` at documented concurrency; and
3. single-stream decode as an unconstrained measured value, with no fabricated target.

The benchmark artifact must name the model’s total and active parameter counts and the exact hardware. A result achieved by batching or concurrency must say so in the title and table.

## 18. Quality validation

### 18.1 Kernel and layer parity

For every supported dtype/quantization:

- compare canonical router output to scalar reference;
- compare each expert projection;
- compare SiLU/product and selected-expert result;
- compare ordered merge;
- compare DeltaNet decode state and output for at least 128 steps;
- compare extend output/state to repeated reference decode where mathematically applicable; and
- test zero, maximum finite, denormal, NaN/Inf rejection, and near-tied routing inputs.

Tolerance tables live beside tests and identify dtype, quantization, shape, absolute tolerance, relative tolerance, and rationale. Do not use one permissive tolerance for every kernel.

### 18.2 End-to-end parity

Required real-fixture gates:

1. greedy token identity for at least:
   - 20 short prompts;
   - 10 multi-turn prompts;
   - 5 long-context prompts; and
   - 256 generated tokens on at least 5 prompts;
2. router-set validation for every instrumented token/layer:
   - short and multi-turn cases require exact selected-set identity, except a
     one-for-one numerical-boundary substitution whose paired gap is within
     `4e-4`; such substitutions must remain below `0.01%` of all traced routes;
   - long-context cases require at least `99%` route-entry agreement and report
     the divergence count and maximum selected-set symmetric difference; and
   - the passing full admission profile uses five prompts of at least 256
     tokens, while `XRT_REAL_MOE_QUALITY_LONG_TOKENS` reproduces longer 384,
     512, and 4096-token diagnostic soaks;
3. mean final-logit cosine similarity `>= 0.99999`;
4. normalized RMS final-logit error `<= 1e-3`;
5. relative perplexity change `<= 0.1%` on a pinned corpus; and
6. no material regression outside the confidence interval on one pinned task evaluation appropriate to the model.

Phase 0 pins the prompt and evaluation corpora by revision/hash, verifies redistribution/use terms, records preprocessing and the exact command, fixes all seeds and decoding parameters, and pre-registers the task metric, non-inferiority margin, and confidence-interval method. Private production prompts are not collected for this purpose.

Execution evidence through 2026-07-21 closes gates 1-6. The full
prompt/logit/route profile passed at the 256-token long-context admission
minimum. The pinned 16-case GSM8K comparison then scored 15/16 for both
optimized CPU and exact hybrid CUDA; every paired correctness result matched,
yielding a 10,000-resample 95% score-difference interval of `[0, 0]`. On the
SHA-pinned WikiText profile, the registered F32 canonical-activation run scored
CPU perplexity `29.316061962` and hybrid CUDA `29.316141694`; the relative
change was `0.000002720`, below the `0.001` limit. A separate production-mode
diagnostic changed by `-0.026975528` because CPU and hybrid execute different
proportions of experts through the existing quantized CPU-activation path. That
diagnostic remains recorded and is not substituted for the canonical gate.

If greedy identity fails because two final logits are nearly tied, record both logits, but do not waive the gate without an explicit maintainer decision and a narrower numerical fix.

### 18.3 State isolation

Tests must prove:

- session A output/state is unchanged when interleaved with session B;
- resetting A does not change B;
- canceling A does not change B;
- prefix-forked sessions agree through the shared prefix and diverge only after distinct tokens;
- a rollback restores both recurrent state and KV length; and
- the same session produces the same greedy result under sequential and cooperative scheduling.

## 19. Test strategy

### 19.1 Unit tests

- metadata and layer-plan validation;
- router tie, NaN, top-k, and allocation behavior;
- placement bijection and generation lifecycle;
- slot budget arithmetic and rollback;
- CPU task grouping and deterministic merge;
- topology discovery fallback;
- recurrent state construction/reset and position-preserving snapshot/restore, including malformed layer count/presence/vector lengths;
- transaction commit/rollback;
- CUDA-disabled stubs;
- graph-key epoch behavior; and
- status/config parsing.

### 19.2 Property and fuzz tests

- random router logits versus a scalar full-sort oracle;
- random expert/slot maps preserve logical IDs;
- random cancellation points restore committed state;
- malformed GGUF metadata/tensor geometry never panics or overflows;
- malformed recurrent snapshots never panic or partially mutate a session;
- placement updates never expose partial mappings; and
- randomized two-session interleaving equals isolated execution.

### 19.3 Integration tests

- tiny CPU MoE generation;
- tiny forced hybrid generation;
- forced zero/all/some GPU slot configurations;
- CPU/CUDA output parity;
- two concurrent Qwen3.5 sessions;
- reset, cancel, prefix, and speculative rollback;
- explicit CUDA unsupported error;
- `auto` runtime/model-load fallback;
- streaming/non-streaming chat, legacy completion, and model-list schema snapshots;
- multimodal chat and `/v1/images/remove-background` regression tests; and
- runtime status additive compatibility.

### 19.4 CI commands

Required normal CI:

```powershell
cargo check --workspace
cargo test --workspace
cargo fmt --all --check
cargo clippy --workspace -- -D warnings
cargo bench --no-run
```

Required CUDA compile/reproducibility validation:

```powershell
.\scripts\safe-cuda-check.ps1
```

Real model/GPU execution remains opt-in through the safe smoke script and self-hosted workflow. Add dedicated inputs for:

- MoE model path/identity;
- Qwen3.5 model path/identity;
- expert budget;
- placement mode;
- maximum prompt/output tokens;
- parity mode; and
- benchmark sample count;
- quality profile (`smoke` or `full`); and
- a hard quality-run timeout.

No real model GPU run is added to default developer commands.

## 20. Implementation phases

### Phase 0 — provenance, fixtures, and baselines

Deliver:

- pin the external source commits in Section 27;
- select model fixtures with license and SHA-256;
- pin the quality corpora, evaluation commands, seeds, metric margins, and confidence method;
- pin the `llama.cpp` comparison commit/container and build/runtime flags;
- add tiny deterministic MoE and hybrid-state test builders;
- add router/expert/state microbenchmarks;
- capture legacy CPU and current unsupported-CUDA behavior;
- capture same-hardware dense protection baselines; and
- write a benchmark-result template.

Exit criteria:

- fixtures are reproducible;
- baseline JSON artifacts include required environment fields;
- the pinned KTransformers/SGLang/model combination is either smoke-validated or explicitly omitted from comparison;
- performance and quality thresholds are pre-registered against measured harness noise;
- no production execution behavior changes.

### Phase 1 — session-owned recurrent state

Deliver:

- move `DeltaNetState` ownership into `BackendSession`;
- change model/backend forward and snapshot interfaces;
- remove model-global state locking;
- preserve the existing infallible public session-construction/reset signatures through checked load-time geometry and fallible pre-token preparation;
- replace the incomplete state alias with a versioned snapshot that includes position and validates geometry/layer payloads before mutation;
- make reset/cancel session-specific and poison the session when a pre-token boundary cannot be restored;
- keep exclusive hybrid scheduling initially;
- add two-session isolation, malformed-snapshot, rollback, and injected-forward-failure tests.

Exit criteria:

- no mutable recurrent state remains on `LlamaModel`;
- snapshot/restore preserves position and rejects wrong layer count/presence or vector length without panic or partial mutation;
- sequential CPU outputs match the Phase 0 baseline;
- two interleaved sessions equal isolated runs;
- CPU-only workspace CI passes;
- performance gate passes.

This is the first production implementation slice.

### Phase 2 — canonical router and optimized CPU MoE

Deliver:

- one canonical router;
- allocation-free fixed-capacity top-k;
- centralized single/batch MoE execution;
- bounded expert worker pool and shared thread budget;
- portable single-node topology;
- batched token grouping;
- optional Linux NUMA discovery/affinity;
- CPU telemetry and benchmarks.

Exit criteria:

- router/expert/end-to-end quality gates pass;
- no steady-state allocation in the instrumented route/dispatch path;
- CPU fallback works on non-x86 and no-NUMA configurations through scalar/portable paths;
- performance gate passes before default selection changes.

### Phase 3 — exact eager static hybrid MoE

Prerequisite: the central GPU allocation/scratch arena from `GPU_RUNTIME_ACCELERATION_SPEC.md` is available or this phase lands through that same abstraction.

Deliver:

- GPU expert slots and resource accounting;
- uniform static placement;
- pinned activation/output staging;
- event-based heterogeneous coordinator;
- minimal selected-expert CUDA kernels or reuse of resident primitives;
- exact ordered merge;
- explicit `hybrid` benchmark mode;
- failure/cancellation cleanup.

Exit criteria:

- forced CPU/GPU split passes all parity tests;
- no expert weights transfer per decode token;
- explicit and `auto` fallback rules pass;
- VRAM stays within budget;
- hybrid performance gate passes on the self-hosted target before `auto` eligibility.

### Phase 4 — CUDA Qwen3.5 recurrent execution

Deliver:

- device recurrent-state allocation per session;
- DeltaNet decode kernels;
- extend/prefill kernels;
- transactional state commit;
- request-indexed scheduler integration;
- session-owned or bounded leased `ForwardScratch`/`BatchScratch` for every concurrently entered hybrid forward path;
- multiple hybrid sessions;
- durable snapshot and a correctness-first rollback.

Exit criteria:

- 128-step state/output parity passes;
- two concurrent sessions pass isolation tests;
- no concurrently eligible hybrid path acquires model-global mutable scratch;
- cancellation restores exact state;
- exclusive scheduling is removed only for validated paths;
- unsupported shapes remain clear errors or runtime/model-load `auto` CPU fallback.

Implementation evidence:
`benchmark-results/hybrid-moe/phase4-qwen35-hybrid-moe-diagnostic-2026-07-20.json`
records the exact combined Qwen3.5 recurrent/full-attention and fixed-placement
hybrid-MoE CUDA path. The synthetic RTX 4090 gate matches CPU logits and
recurrent state, replays resident-expert subgraphs, and proves that an injected
mid-token failure does not publish pending recurrent state. Adaptive placement
and layerwise prefill remain explicit load-time errors for this combined
architecture until their separate gates pass; no real-model `auto` admission is
claimed.

### Phase 5 — graph capture and fast state transactions

Deliver:

- fixed-placement graph keys/epochs;
- all-GPU MoE graph path;
- hybrid GPU-subgraph capture;
- safe callback experiment or documented event-based decision;
- device-local recurrent checkpoint/journal;
- speculative target verification.

Exit criteria:

- stale placement/state/scratch cannot replay;
- graph mode beats eager on eligible shapes;
- rollback cost is bounded and measured;
- speculative accepted/rejected sequences match the reference.

Implementation evidence: `benchmark-results/hybrid-moe/phase5-moe-graph-diagnostic-2026-07-20.json`
records exact fixed-placement expert subgraphs, retained expert-slot lifetimes,
placement/scratch epoch rejection, central accounting, and a 15.32% median
synthetic full-GPU decode improvement. The exact hybrid GPU subgraph is
available with `XRT_CUDA_GRAPH=on`, but remains excluded from `auto` after its
same-hardware synthetic result was 5.01% slower by median.

### Phase 6 — state-aware prefix/radix reuse

Prerequisite: shared GPU page allocation and COW infrastructure from the main GPU plan.

Deliver:

- hybrid prefix identity and durable state;
- fork/COW/protect/evict;
- device and host accounting;
- concurrent prefix reuse;
- scheduler integration.

Exit criteria:

- prefix results equal no-cache execution;
- eviction cannot free in-use state;
- prefix sharing reduces measured TTFT or memory for a required workload;
- hybrid bypass remains for any unsupported backend/state form.

### Phase 7 — adaptive placement and layerwise prefill

Deliver:

- profiled static manifests;
- bounded adaptive placement;
- in-place stable map updates with placement epochs;
- placement upload telemetry;
- exact layerwise prefill with double buffering;
- expanded external comparison suite.

Exit criteria:

- update cost and churn are included in performance;
- adaptive gate passes before default use;
- layerwise prefill gate passes;
- quality remains identical under placement changes;
- `auto` policy is documented by hardware/model capability class.

## 21. Rollout, migration, and rollback

### 21.1 Rollout

1. Land Phases 0–1 with no default backend change.
2. Land optimized CPU MoE behind `XRT_MOE_ACCELERATION=cpu`.
3. Retain `legacy` during at least one full validation cycle.
4. Land static hybrid behind explicit `hybrid`.
5. Admit exact hardware/model combinations to `auto` only through a capability table backed by CI artifacts.
6. Add Qwen3.5 CUDA and batching per validated geometry.
7. Keep adaptive placement, hybrid graph-host integration, state-aware prefix, and layerwise prefill opt-in until their individual gates pass. Full-GPU MoE expert graphs may participate in graph `auto` only inside an already explicit `gpu` MoE mode; this does not admit MoE acceleration itself to `auto`.

### 21.2 Data migration

There is no persistent user-data or model-format migration. Existing GGUF files remain valid. Prefix-cache entries are runtime-local and must include a state-format version; incompatible entries are discarded.

### 21.3 Rollback

Operational rollback:

- set `XRT_MOE_ACCELERATION=legacy` for MoE;
- select `cpu` for hybrid models;
- return placement to `uniform`, set `XRT_MOE_LAYERWISE_PREFILL=off`, set `XRT_CUDA_GRAPH=off`, and disable prefix/adaptive behavior through the documented kill switch delivered with each new feature; no such switch is assumed to exist before that feature lands;
- restart sessions so no state crosses implementations.

Code rollback is safe because no OpenAI or GGUF contract changes. Do not attempt to downgrade a live session containing a CUDA recurrent state; terminate/reset it explicitly.

## 22. Security and privacy

1. New local GGUF execution remains offline and must not contact upstream KTransformers or model services. Existing explicit external-proxy and model-download modes are unchanged.
2. Placement-manifest paths are process-start or local-benchmark inputs only. Manifest reads are size-bounded, parsed without executing content, and validated as described in Section 12.1.
3. All dimension and byte calculations use checked arithmetic.
4. Queues, batches, selected pairs, snapshots, and placement histories are bounded.
5. Unsafe SIMD, affinity, pinned-memory, callback, and device-pointer code documents ownership, alignment, thread, and lifetime invariants.
6. CPU callbacks never unwind across FFI.
7. Routing telemetry excludes content by default.
8. Errors and traces do not include prompt/token/activation data.
9. Host recurrent buffers and pinned activation/output staging are explicitly overwritten before reset completion or cross-session reuse. A shared scratch lease is not returned to the pool until its request-derived region is cleared.
10. CUDA recurrent/staging allocations are zeroed on their owning stream before cross-session reuse. If zeroing or fencing fails, the allocation is retired with the failed session and is not returned to a shared pool. Destruction alone is not claimed as secure zeroization.
11. Placement manifests and benchmark routing profiles contain only bounded aggregate logical-expert counts, never prompts, token IDs, or activations. Runtime status exposes a manifest identity/hash, not its filesystem path or frequency table.
12. A client cannot directly choose arbitrary physical expert slots or device pointers, and this specification adds no remote placement controls to the unauthenticated server.
13. Resource admission includes recurrent state and staging so concurrent requests cannot overcommit memory silently.

## 23. Compatibility matrix

| Capability | CPU-only | CUDA dense | CUDA MoE | CUDA Qwen3.5 hybrid |
|---|---:|---:|---:|---:|
| GGUF load | Required | Required | Required for supported layout | Required for supported layout |
| Existing SafeTensors load | Existing limitations unchanged | Unchanged | Not expanded by this spec | Not expanded by this spec |
| OpenAI API | Unchanged | Unchanged | Unchanged | Unchanged |
| Legacy inference | Required | Existing | Required fallback | Required fallback |
| Optimized CPU expert pool | Yes | N/A | Yes for cold experts | Yes where MoE |
| GPU expert slots | N/A | N/A | Phase 3 | Phase 3 where MoE |
| Session recurrent state | CPU | N/A | If hybrid | Phase 1 CPU / Phase 4 CUDA |
| Continuous batching | Existing eligible paths | Existing | After Phase 3 validation | After Phase 4 validation |
| CUDA graph | N/A | Existing eligible paths | Phase 5 | Phase 5 eligible subgraphs |
| Prefix reuse | Existing full-attention path | Existing | Full-attention rules | Phase 6 |
| Speculative rollback | Existing eligible paths | Existing eligible paths | Validate per path | Phase 5 |

Unsupported combinations remain explicit. The presence of `expert_count` or a Qwen alias alone does not imply CUDA support.

## 24. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| PCIe activation traffic exceeds saved GPU work | Hybrid slower than CPU or GPU | Measure bytes/time; overlap tiers; gate `auto`; group batch rows |
| CPU and GPU finish at very different times | Join stalls | Placement tuning, per-layer telemetry, balanced slot policy |
| CPU pool oversubscribes dense kernels | Dense and MoE regress | One runtime thread budget; no nested unrestricted pools |
| NUMA binding hurts portable machines | Regression or launch failure | Optional topology adapter; portable fallback; strict only for diagnosis |
| Dynamic placement thrashes | Transfer overhead and graph invalidation | Hysteresis, minimum residency, bounded moves, safe boundaries |
| Logical/physical ID confusion | Wrong expert output | Typed maps, immutable snapshots, property tests |
| Asynchronous completion changes sum order | Numerical/model drift | Canonical ordered merge |
| Recurrent state remains partly global | Cross-session corruption | Phase 1 ownership audit and two-session tests |
| Snapshot omits position or accepts wrong geometry | Incorrect rollback/prefix state or panic | Versioned complete snapshot; validate all fields before mutation |
| Model-global scratch remains after state migration | Hidden serialization or cross-session corruption | Keep exclusive guard until session-owned/bounded scratch leases pass concurrency tests |
| State memory grows per session | OOM under concurrency | Admission accounting, quotas, compact dtype only after quality spec |
| Cancellation commits partial state | Corrupt conversation | Token transaction over KV and recurrent state |
| Qwen3.5 layer rule is wrong | Incorrect model execution | Resolved layer plan, real fixture, reject ambiguity |
| Shared-expert variant is unhandled | Missing model contribution | Explicit descriptor/support or load-time rejection |
| Graph captures stale placement pointers | Wrong weights or use-after-free | Generation in key, stable buffers, leases, eager fallback |
| Host callback lifetime is unsafe | Crash/deadlock | Event coordinator MVP; callback requires isolated proof |
| Upstream KTransformers changes | Stale assumptions | Pin commits; re-audit before each borrowed technique |
| Benchmark headline is misleading | False performance expectations | Separate prefill/single/aggregate; publish hardware and active parameters |
| Quality threshold hides route changes | Silent degradation | Exact selected-ID instrumentation and greedy parity |
| New placement knobs enter unauthenticated HTTP control plane | Resource abuse or local-path disclosure | Startup/bench-only configuration in this spec; separate authenticated operator design |

## 25. Dependencies and sequencing

This specification complements rather than replaces `docs/GPU_RUNTIME_ACCELERATION_SPEC.md`.

Relationship to the current main GPU TODO list:

| Main GPU TODO | Relationship to this specification |
|---|---|
| Central GPU scratch/allocation arena | Required before resident expert slots and pinned/stable hybrid staging |
| Broader continuous batching | Unblocked for Qwen3.5 by Phase 1 session-owned recurrent state; enabled after Phase 4 isolation gates |
| Shared GPU page allocator and COW | Required by Phase 6 state-aware hybrid prefix reuse |
| Kernel optimization | Extended here with selected-expert, ordered-merge, and DeltaNet kernels |
| Broader architecture support | Qwen3.5 hybrid recurrent layers and MoE are the primary scope |
| Advanced format breadth | Reuses validated resident matrix formats; new formats remain separately gated |
| Radix/prefix work | Extended here so recurrent state and full-attention KV share one exact prefix boundary |
| Comparative benchmarking | Adds pinned KTransformers/SGLang and MoE-specific same-hardware measurements |
| Reliability | Adds multi-session isolation, transactional cancellation, placement epochs, and failure cleanup |

Dependencies:

1. Phase 1 session ownership can start immediately.
2. Phase 2 CPU work can proceed without CUDA.
3. Phase 3 must use the planned central GPU allocation/scratch arena to avoid a second untracked allocator.
4. Phase 5 graph work depends on stable Phase 3 resources.
5. Phase 6 state-aware prefix COW depends on the shared GPU page allocator/COW work in the main GPU plan.
6. Quantized recurrent state or TurboQuant is a separate quality-gated optimization and is not required here.
7. Dynamic placement depends on exact static hybrid behavior and reliable transfer/resource telemetry.

The main GPU TODO order remains authoritative where these projects overlap. Do not implement a private expert allocator or private page cache merely to start later phases early.

## 26. Acceptance criteria

### Execution status — 2026-07-21

The exact implementation is complete through the opt-in Phase 7 surface. It
includes session-owned recurrent state, one canonical router, grouped CPU MoE,
resident expert slots, exact heterogeneous coordination, CUDA Qwen3.5
DeltaNet, fixed-placement graph replay, paired KV/recurrent prefix COW,
profiled/adaptive placement machinery, layerwise prefill, recycled per-layer
device buffers, batched CPU-result upload, parallel/grouped cold-expert
execution, and an ordered packed CPU-row merge. Unsupported or unadmitted
combinations remain explicit errors or opt-in modes; this execution does not
change the OpenAI schema or admit real MoE models to `auto`.

Durable evidence:

- `benchmark-results/hybrid-moe/real-model-parity-2026-07-20.json` records the
  licensed Qwen3 and Qwen3.5 two-token CPU/CUDA smoke, including Qwen3.5
  recurrent-state parity.
- `benchmark-results/hybrid-moe/qwen3-quality-harness-smoke-2026-07-21.json`
  records the executable, feature-gated prompt/logit/route harness. Its bounded
  profile ran one short, one multi-turn, and one 174-token long-context case;
  all 10 greedy comparisons and all 11,856 per-token/per-layer logical expert
  routes matched, with mean/worst cosine `1.0` and worst normalized RMS error
  `0.000003415`. The trace records only layer/logical-expert IDs, is bounded,
  and is compiled out unless `moe-route-trace` is enabled.
- `benchmark-results/hybrid-moe/qwen3-quality-harness-full-2026-07-21.json`
  records the passing 20/10/5 prompt profile at the 256-token long-context
  admission minimum, five 256-token generations, 1,310 logit comparisons, and
  171,264 route entries. Greedy outputs matched, mean/worst cosine was `1.0`,
  worst normalized RMS was `1.2629e-5`, and no route divergence occurred. The
  artifact also records the rejected 384/512/4096-token extended soaks.
- `benchmark-results/hybrid-moe/qwen3-gsm8k-task-2026-07-21.json` records the
  SHA-pinned 16-case task gate at its adjudicated 512-token output cap.
  Optimized CPU and exact hybrid CUDA each scored 15/16, with identical paired
  correctness and a paired-bootstrap 95% score-difference interval of `[0, 0]`.
- `benchmark-results/hybrid-moe/qwen3-perplexity-gate-2026-07-21.json` records
  the passing pinned WikiText gate. Canonical CPU and hybrid CUDA perplexity
  were `29.316061962` and `29.316141694`, respectively, for a relative change
  of `0.000002720`. It also retains the separate `-2.70%` production-mode
  diagnostic and the line-bounded tokenization repair. The earlier
  `qwen3-perplexity-harness-open-2026-07-21.json` remains as historical timeout
  evidence and links to the resolution.
- `benchmark-results/hybrid-moe/qwen3-xrt-llamacpp-comparison-2026-07-20.json`
  records three bracketing XENO blocks and three pinned llama.cpp placements
  with raw samples. At approximately matched total device use, XENO's median
  decode was 41.26% higher (independent-bootstrap median-ratio interval
  `1.360..1.557`), while llama.cpp full-GPU was 18.70 times the bracketed XENO
  median and XENO prefill remained substantially slower.
- `benchmark-results/hybrid-moe/qwen3-internal-admission-2026-07-20.json`
  records the required symmetric same-hardware comparison. The final packed
  merge improves median decode over the prior grouped-row build by 1.81%
  (paired-bootstrap ratio interval `1.003..1.034`), but reaches only `0.977x`
  optimized-CPU throughput (`0.944..0.988`) and has a p95 latency-ratio upper
  bound of `1.106`. It therefore fails both registered `auto` criteria and
  remains explicit-only. A post-validation rebuild reproduced the controlled
  final median within `0.01%`. Its 30-repetition adaptive-placement follow-up
  reduced H2D traffic but ended at 17.12 tok/s for repetitions 26-30, below
  both fixed uniform placement and optimized CPU, so adaptive placement also
  remains experimental for this workload.
- `benchmark-results/hybrid-moe/qwen3-residual-q8-rejected-2026-07-20.json`
  records a subsequent two-pass residual-Q8 activation experiment for Q4_K
  expert projections. The candidate passed a bounded 16-token Qwen3 quality
  smoke, but the raw symmetric repeat measured a candidate/previous median
  throughput ratio of `0.984` with a paired-bootstrap interval of
  `0.972..0.998`. Raw and repacked Q6_K extensions were also slower. All
  residual-Q8 source, PTX, API, workflow, and runtime routing changes were
  removed; the artifact is retained as negative evidence.
- `benchmark-results/hybrid-moe/qwen3-fused-silu-mul-rejected-2026-07-20.json`
  records a later exact launch-fusion experiment for GPU-resident experts. The
  fused SiLU-times-up kernel was bit-identical to the retained two-launch F32
  path, but the real-Qwen3 `A-B-B-A` throughput point estimate was `0.991x`
  baseline with a paired-bootstrap interval of `0.980..1.005`; the latency
  interval also crossed parity. The kernel, API, and runtime route were removed
  because the evidence did not establish no-regression.
- `benchmark-results/hybrid-moe/qwen3-whole-layer-placement-rejected-2026-07-20.json`
  records an exact whole-layer placement experiment. Eight complete Qwen3 MoE
  layers fit under the same 4 GiB budget and reduced activation transfers, but
  the same-binary `A-B-B-A` comparison measured a candidate/uniform throughput
  ratio of `0.970` with a paired-bootstrap interval of `0.956..0.985`. The
  policy and runtime path were removed; selected-expert GPU execution must
  improve before concentrating every routed expert onto resident GPU layers.
- `benchmark-results/hybrid-moe/qwen3-fragmented-layer-fallback-rejected-2026-07-21.json`
  records an exact fallback that routed a layer's lone selected GPU-resident
  expert through the canonical CPU executor. Real Qwen3 two-token parity passed,
  but the same-binary `A-B-B-A` throughput ratio was `0.994` with a
  `0.974..1.013` interval; the p95-latency ratio interval was `0.992..1.055`,
  and H2D traffic increased by 3.98 MB per request. The threshold, environment
  knob, and runtime branch were removed.
- `benchmark-results/hybrid-moe/qwen3-parallel-expert-graphs-rejected-2026-07-21.json`
  records a KTransformers-inspired attempt to overlap selected resident experts:
  each logical expert received isolated gate/up scratch and already captured
  child graphs replayed on independent CUDA streams before the unchanged
  canonical merge. The controlled `A-B-B-A` point estimate was only `1.002x`
  retained throughput with a `0.986..1.025` interval; the p95-latency ratio
  interval was `0.956..1.055`, and scratch grew by 786,432 bytes. Larger child
  caches, parent-graph composition, and whole-layer placement also failed their
  direction finders. All candidate source paths were removed; this evidence
  rules out graph-level expert concurrency as a sufficient optimization on the
  pinned model/hardware, not lower-level grouped/fused expert kernels.
- `benchmark-results/hybrid-moe/qwen3-q4k-cpu-order-kernel-2026-07-21.json`
  records the retained reuse of the existing 32-thread CPU-order Q4_K CUDA
  matvec for Qwen3 MoE resident matrices. The scoped 16-token parity run
  preserved every greedy token with worst normalized RMS error
  `0.000002999`. In the controlled `A-B-B-A` comparison, candidate/baseline
  throughput was `1.009x` with a `0.987..1.026` interval and the p95-latency
  ratio interval was `0.939..1.037`, clearing both no-regression bounds but not
  establishing decode superiority. Dense Qwen3, Qwen2, and Llama remain on the
  prior Q4_K dispatch, and `auto` remains disabled.
- `benchmark-results/hybrid-moe/qwen3-final-source-control-2026-07-21.json`
  records a fresh final-binary `A-B-B-A` comparison against the preserved
  pre-canonical-boundary control. The current/control throughput interval was
  `0.984..1.001` and the p95-latency ratio interval was `0.960..1.023`, so the
  final source clears both no-regression bounds without establishing a speedup.
- `benchmark-results/hybrid-moe/qwen3-aggregate-decode-graph-lifetime-2026-07-20.json`
  records additive, backward-compatible TTFT/decode benchmark fields, the
  concurrency-8 graph invalid-free reproducer, its session-lifetime repair, a
  final three-repetition live-GPU soak, and an `A-B-B-A` single-stream
  no-regression comparison. The repaired path measured a median 10.16
  aggregate decode tok/s at concurrency 8, so this fixture reaches only about
  one tenth of the `>=100` stretch objective. The single-stream throughput
  ratio interval was `0.983..1.010`; the p95 ratio point was `1.004`, while its
  `0.926..1.051` interval remains marginally inconclusive against the separate
  auto-admission latency bound. No default was enabled.
- `benchmark-results/hybrid-moe/validation-2026-07-20.json` records default and
  CUDA builds/tests, the complete bounded safe-CUDA suite, workflow/script
  parsing, and byte-identical NVCC reproduction for the three new PTX files.
- `benchmark-results/hybrid-moe/dense-protection-2026-07-20.json` records
  paired-bootstrap Qwen2.5 Q4_0 CPU and CUDA baseline/candidate gates. Both
  backends clear the dense throughput and p95 latency no-regression bounds.
- The phase-specific JSON files in `benchmark-results/hybrid-moe/` retain the
  synthetic performance, placement, graph, rollback, prefix, and layerwise
  diagnostics.

The specification's overall completion gate remains open for four reasons:

1. all registered Section 18.2 prompt/logit/route, perplexity, and task gates
   pass at the declared admission profile, but the 384/512/4096-token diagnostic
   profiles remain failed/open and are not claimed as broader context coverage;
2. the ordered real-model comparison is complete, but exact hybrid fails the
   Section 17.5 throughput and p95-latency bounds against optimized CPU; the
   subsequent parallel-expert graph experiment was removed after establishing
   neither superiority nor the registered p95 no-regression bound;
3. repository-wide `cargo fmt --all --check` and strict Clippy are blocked by
   pre-existing unrelated debt; all 36 Rust files touched by this work pass
   direct `rustfmt --check`, and the newly added code has no reported strict
   Clippy diagnostic; and
4. the dedicated self-hosted real-MoE workflow has been parsed locally but has
   not been dispatched and approved on GitHub.

The overall specification is complete when:

- [x] A reproducible tiny MoE and tiny Qwen3.5 hybrid fixture exist.
- [x] At least one real licensed MoE GGUF and one real licensed Qwen3.5 hybrid fixture are pinned by revision and SHA-256.
- [x] `LlamaModel` contains no mutable per-session recurrent state.
- [x] Public `Runtime::new_session*` and `Session::reset()` signatures remain compatible; fallible recurrent allocation occurs before token 0.
- [x] Durable recurrent snapshots preserve the paired KV/accepted position and reject malformed layer geometry/payloads without panic or partial restore.
- [x] Two hybrid sessions run without state contamination.
- [x] CPU-only builds and all existing supported models continue to work.
- [x] Existing multimodal chat and `xrt-vision`/ONNX task endpoint regression tests continue to pass.
- [x] One canonical allocation-free routing path serves all forward modes.
- [x] Exact CPU expert execution passes kernel, layer, greedy-token, logit, and perplexity gates.
- [x] Exact static hybrid executes hot and cold selected experts concurrently.
- [x] Decode does not transfer cold-expert weights per token.
- [x] GPU expert slots and recurrent state are included in resource admission/status.
- [x] Explicit backend errors and `auto` fallback follow Section 14.
- [x] `XRT_BACKEND`/MoE-policy precedence, full-residency `gpu` semantics, manifest validation, and non-MoE conflict cases have configuration tests.
- [x] CUDA Qwen3.5 decode and extend pass state/output parity.
- [x] Cancellation rolls back KV and recurrent state together.
- [x] Hybrid multi-session scheduling passes fairness and isolation tests.
- [x] No concurrently eligible hybrid forward holds model-global mutable `ForwardScratch` or `BatchScratch`.
- [x] Fixed-placement graph replay rejects stale generations.
- [x] State-aware prefix reuse equals uncached generation.
- [x] Dense CPU/CUDA regression gates pass.
- [ ] Each enabled acceleration path passes its same-hardware performance gate.
- [x] Benchmark reports clearly separate TTFT, prefill, single-stream decode, and aggregate decode.
- [x] `/v1/chat/completions`, `/v1/completions`, and `/v1/models` schema snapshot tests show no compatibility change.
- [x] No new MoE placement/budget/manifest control is exposed through HTTP, and request-derived recurrent/staging/scratch buffers pass cross-session clearing tests.
- [ ] Normal CI, safe CUDA checks, PTX reproducibility, and approved real-GPU workflows pass.
- [x] Provenance review is complete for any adapted source.

## 27. References and pinned research baseline

Research was verified on 2026-07-20. Re-verify upstream before implementation because support and code paths change quickly.

### 27.1 XENO RT

- `../XENO CORPORATION - Full Ecosystem Report.md`
- `Cargo.toml`
- `Cargo.lock`
- `docs/GPU_RUNTIME_ACCELERATION_SPEC.md`
- `docs/AGENT_ADAPTIVE_KV_ROADMAP.md`
- `docs/turboquant-kv-cache-plan.md`
- `docs/SESSION-STATE-2026-07-17.md`
- `crates/xrt-models/src/llama.rs`
- `crates/xrt-runtime/src/lib.rs`
- `crates/xrt-runtime/src/backend.rs`
- `crates/xrt-runtime/src/session.rs`
- `crates/xrt-runtime/src/resident_tensor.rs`
- `crates/xrt-cuda/src/lib.rs`
- `crates/xrt-server/src/main.rs`
- `crates/xrt-kernels/src/cpu/thread_pool.rs`
- `.github/workflows/ci.yml`
- `.github/workflows/cuda.yml`
- `.github/workflows/moe-validation.yml`

### 27.2 KTransformers

- Repository: https://github.com/kvcache-ai/ktransformers
- Audited KTransformers commit: https://github.com/kvcache-ai/ktransformers/commit/d1a3ed8a308cf45a2bdf8dc0ec18ea0cf782486c
- Current inference architecture: https://ktransformers.net/en/docs/inference
- Support matrix: https://ktransformers.net/en/docs/support-matrix
- Expert placement: https://ktransformers.net/en/docs/optimization-techniques/expert-placement
- Layerwise prefill: https://ktransformers.net/en/docs/optimization-techniques/layerwise-prefill
- CPU inference queue/stream integration:
  https://github.com/kvcache-ai/ktransformers/blob/d1a3ed8a308cf45a2bdf8dc0ec18ea0cf782486c/kt-kernel/cpu_backend/cpuinfer.h
- Topology-aware worker pool:
  https://github.com/kvcache-ai/ktransformers/blob/d1a3ed8a308cf45a2bdf8dc0ec18ea0cf782486c/kt-kernel/cpu_backend/worker_pool.cpp
- SOSP 2025 paper:
  https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf

### 27.3 Pinned SGLang integration used by the audited KTransformers tree

The audited KTransformers tree pins SGLang commit `1e098a77ba395dc1a5f2dcbdf57bdb188e84bcee`.

- KTransformers expert-parallel wrapper:
  https://github.com/kvcache-ai/sglang/blob/1e098a77ba395dc1a5f2dcbdf57bdb188e84bcee/python/sglang/srt/layers/moe/kt_ep_wrapper.py
- Qwen3.5 model/layer selection:
  https://github.com/kvcache-ai/sglang/blob/1e098a77ba395dc1a5f2dcbdf57bdb188e84bcee/python/sglang/srt/models/qwen3_5.py
- Gated DeltaNet request-indexed state:
  https://github.com/kvcache-ai/sglang/blob/1e098a77ba395dc1a5f2dcbdf57bdb188e84bcee/python/sglang/srt/layers/attention/linear/gdn_backend.py
- Hybrid recurrent radix cache:
  https://github.com/kvcache-ai/sglang/blob/1e098a77ba395dc1a5f2dcbdf57bdb188e84bcee/python/sglang/srt/mem_cache/mamba_radix_cache.py

## 28. Assumptions

1. KTransformers is used only as a reference and optional differential benchmark.
2. NVIDIA CUDA remains the first GPU implementation.
3. The central GPU allocation/scratch work in the main GPU plan lands before resident expert slots.
4. Existing GGUF expert tensor naming is sufficient for at least one target Qwen3/Qwen3.5 fixture.
5. CPU expert weights can remain memory-mapped or be packed once without changing GGUF compatibility.
6. Static placement is sufficient to prove the hybrid architecture before adaptation.
7. Exact execution with the same quantized checkpoint should have no measurable task-quality loss after numerical parity gates.
8. No release or hosted deployment is part of this work.

## 29. Resolved decisions and open questions

Resolved during execution through 2026-07-21:

1. The real fixtures are Qwen/Qwen3-30B-A3B-GGUF
   `Qwen3-30B-A3B-Q4_K_M.gguf` at revision
   `e4d4bafdfb96a411a163846265362aceb0b9c63a` and
   janhq/Qwen3.5-35B-A3B-GGUF `Qwen3.5-35B-A3B-Q4_K_S.gguf` at revision
   `b1dc3970f5de842277dda379f57867b8595b23bf`. `fixtures.json` records their
   exact SHA-256 digests and licenses. These are runner inputs, not repository
   blobs.
2. The accepted Qwen3.5 family rule is three recurrent layers followed by one
   full-attention layer. The current GGUFs do not supply a per-layer schedule
   that XENO RT treats as authoritative, so load-time resolution applies the
   family rule and then validates every expected recurrent or full-attention
   tensor and its geometry. A mismatch fails model loading instead of silently
   guessing a different graph. The pinned 40-layer real fixture passed this
   validation.
3. The first supported shared-expert variant is the Qwen3.5 four-tensor
   sigmoid-gated layout: `ffn_gate_inp_shexp`, `ffn_gate_shexp`,
   `ffn_up_shexp`, and `ffn_down_shexp`. All four tensors and the declared
   shared intermediate size are required and shape-validated. The contribution
   runs exactly on CPU and in layerwise hybrid execution; the pinned mixed-quant
   Qwen3.5 fixture passed the two-token CPU/CUDA parity smoke.
4. The canonical router uses a narrow `1e-5` top-k boundary band and descending
   logical IDs inside that band. The full Qwen3 investigation rejected wider
   `1e-4` and `2e-4` bands after they changed upstream state and created new,
   high-impact route divergences. The narrow descending-ID rule aligned all four
   reproduced short-prompt boundary seams and the final admission profile passed
   171,264 route entries with zero substitutions. Trace comparison canonicalizes
   each selected set so harmless rank swaps do not masquerade as membership
   divergence. Independent long-context backends instead use the registered
   `99%` route-agreement plus token/logit gates because their hidden states are
   not bit-identical indefinitely.
5. The executable full prompt profile uses a 256-token minimum for each of five
   long prompts. It passed 35 cases, 1,310 logit comparisons, five 256-token
   generations, mean/worst cosine `1.0`, and worst normalized RMS `1.2629e-5`.
   Extended 384- and 512-token `long-03` soaks failed the unchanged normalized
   RMS gate, while the 4096-token `long-01` soak exposed accumulated route drift.
   Those lengths remain reproducible with `XRT_REAL_MOE_QUALITY_LONG_TOKENS` and
   are not claimed as passing admission coverage.
6. The pinned generic task gate uses the first 16 GSM8K `main/test` rows in
   Parquet order at Hugging Face revision
   `740312add88f781978c0658806c59bc2815b9866`. The SHA-pinned UTF-8 JSONL
   projection, greedy chat-template prompt with `/no_think`, and adjudicated
   512-token output cap produced 15/16 exact match for both optimized CPU and
   exact hybrid CUDA. All paired outcomes matched and the 95% score-difference
   interval was `[0, 0]`, so the registered zero-point non-inferiority gate
   passes.
7. The perplexity corpus is WikiText-2 raw test at revision
   `b08601e04326c79dfdd32d625aee71d232d685c3`, with source and derived UTF-8
   hashes recorded in the quality plan. A repository-native, SHA-verifying
   CPU/CUDA evaluator and an opt-in zero-epsilon router control now exist. The
   first evaluator incorrectly tokenized the complete 1.29 MB corpus before
   truncation; line-bounded prefix tokenization removed that operational stall
   without weakening the full-file hash. The registered F32 canonical run
   produced CPU/CUDA perplexities `29.316061962` and `29.316141694`, a relative
   change of `0.000002720`, and passes the `<=0.1%` gate. The separate
   production-mode comparison changed by `-2.70%`; it remains a visible failed
   diagnostic and is not used to relax or replace the canonical gate.

Still open before `auto` admission:

1. Which additional execution change can make exact hybrid clear the registered
   `1.15x` optimized-CPU throughput lower bound without exceeding the `1.05x`
   p95-latency upper bound? The current packed-merge result clears neither
   bound. A 30-repetition adaptive-placement follow-up reduced per-request H2D
   bytes but produced only 17.12 tok/s in repetitions 26-30, below the 17.61
   tok/s fixed-uniform result and the 18.02 tok/s optimized CPU result. More
   placement tuning is therefore not accepted as the missing execution change;
   the next investigation must reduce fragmented selected-expert GPU work or
   avoid the heterogeneous split when its measured cost exceeds grouped CPU
   execution. A later residual-Q8 DP4A experiment is also rejected: despite
   passing a bounded 16-token logit gate, its controlled candidate/previous
   throughput interval was `0.972..0.998`, and Q6_K variants were slower still.
   An exact fused SiLU-times-up launch was likewise rejected after its `0.991x`
   point estimate and `0.980..1.005` interval failed to prove no-regression. A
   whole-layer placement experiment reduced transfers but regressed throughput
   to `0.970x` uniform with a `0.956..0.985` interval, so placement reshaping
   alone is also not the missing execution change. Finally, routing a layer's
   lone selected GPU-resident expert back through the canonical CPU executor
   produced a `0.994x` throughput point estimate with a `0.974..1.013`
   interval, raised the p95-latency interval to `0.992..1.055`, and added 3.98
   MB of H2D traffic per request. That fallback was removed as well. Reusing
   the existing CPU-order Q4_K CUDA matvec was retained after clearing the
   no-regression bounds: its `1.009x` decode point estimate had a
   `0.987..1.026` interval, while prefill improved with a `0.986..0.998` ratio
   interval. This closes one low-risk selected-expert kernel opportunity but
   does not establish decode superiority or approach the `1.15x` lower-bound
   gate. The next candidate must eliminate more transfer/launch overhead,
   deliver a stronger selected-expert kernel gain, or batch selected-expert
   work across sequences.
2. What RAM, CPU ISA/topology, and VRAM tiers define the initial `auto`
   capability table?
3. Is Linux NUMA optimization sufficient for the first performance milestone
   while Windows uses a correct single-node pool, or is Windows NUMA required
   before default enablement?
4. What maximum recurrent-state bytes per session should resource admission
   allow by default?
5. Which public, licensed prompt corpus best represents XENO Agent workloads
   without collecting private user content? WikiText-2 and GSM8K are pinned for
   the current generic quality gate, but their workload representativeness is
   not established.
6. What documented model/hardware/concurrency pair is credible for the
   `>=100 aggregate decode tokens/s` stretch objective? The first honest
   shared-epoch Qwen3-30B-A3B Q4_K_M measurement on the RTX 4090 reached only
   10.16 aggregate decode tok/s at concurrency 8. Scheduler telemetry reported
   no fused MoE decode batches. This configuration is therefore not credible
   for the stretch line; the next candidate must add true multi-sequence MoE
   batching or a substantially faster selected-expert path and must be measured
   under the full Section 17.3 matrix.

## 30. First implementation handoff

Historical handoff — completed on 2026-07-20 and retained for sequencing
provenance.

The first pull request should implement only Phase 0 fixtures/baselines and Phase 1 session ownership. It must not add CUDA MoE kernels.

Required change sequence:

1. Add a tiny deterministic hybrid fixture with at least two independent sessions.
2. Make `DeltaNetState` constructible from validated model geometry outside `LlamaModel`.
3. Remove `deltanet_state` from `LlamaModel`.
4. Add an uninitialized/CPU recurrent-state descriptor to `BackendSession`; validate geometry at model load and allocate fallibly before token 0.
5. Pass mutable session state through CPU forward/decode paths.
6. Replace global clear/save/restore trait calls with session-specific operations.
7. Introduce the versioned snapshot with position and full geometry/payload validation; restore validates everything before copying.
8. Make reset and any future speculative rollback operate on the owning session without changing the public infallible `new_session`/`reset` signatures.
9. Mark a session poisoned if an injected forward failure cannot restore the pre-token KV/recurrent boundary.
10. Keep the existing exclusive hybrid scheduler guard and model scratch locks.
11. Add isolated, interleaved, reset, cancellation, position-preserving snapshot/restore, malformed-snapshot, and failure-injection tests.
12. Run the normal CI commands and record paired pre/post CPU hybrid performance.

Do not remove the exclusive hybrid guard until the isolation suite passes, all call paths use session-owned recurrent state, and mutable forward/batch scratch is session-owned or bounded-leased. Do not begin resident GPU expert allocation until it can use the central GPU resource boundary.
