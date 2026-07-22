# Session State - 2026-07-17

> **Historical `xrt-text` handoff.** This checkpoint records one native CUDA
> effort and intentionally preserves its original facts. It does not define the
> current whole-product scope. See [RUNTIME_DOMAINS.md](RUNTIME_DOMAINS.md) for
> the canonical text/image/video/audio runtime architecture.

## Why this checkpoint exists

The Codex rollout file for session `019ed92e-e82c-78c3-bfe5-751a3eae5070` is missing from
`~/.codex/sessions`. This file preserves the implementation state needed to continue the work
without that conversation. Treat this document and `docs/GPU_RUNTIME_ACCELERATION_SPEC.md` as the
handoff for the current native CUDA effort.

This checkpoint was intentionally created without running Cargo, CUDA kernels, model inference, or
the server. The local PC has crashed during prior heavy work. Verification should run through the
bounded scripts and GitHub Actions/self-hosted GPU runner described below.

## Repository snapshot

- Repository: `xeno-rt`
- Working directory: `X:\code\xeno-corporation\xeno-rt`
- Current branch: `feat/v0.2.0-vision-tasks-remove-background`
- Pre-checkpoint HEAD: `daf7054` (`ci: add cuda runner workflow`)
- Tracking branch: `origin/feat/v0.2.0-vision-tasks-remove-background`
- Important branch warning: the branch name describes an older vision task, but the working tree
  contains the cumulative native CUDA/runtime implementation. Do not infer scope from the branch
  name.
- Related release state: GitHub-only `v0.2.0` was published separately from protected `main` at
  commit `9bdc8d17dd618a38513df1176bff1eb8be52792a`. It was not published to R2 or XENO Hub. This
  dirty feature checkout was not the source of that release.

## Overall goal

Build xeno-rt into a production-grade, native, GPU-resident LLM inference runtime that is a credible
Rust/GGUF alternative to llama.cpp, vLLM, SGLang, and ExLlama while preserving the existing CPU
runtime and the API contracts consumed by the XENO ecosystem.

The final arrival point is:

1. `XRT_BACKEND=cuda` and `--backend cuda` execute supported GGUF models end to end on NVIDIA GPUs.
2. The initial dense CUDA support matrix includes F32, F16, BF16, Q8_0, Q4_0, Q4_K, Q5_K, and Q6_K
   tensors, with Q4_K/Q6_K/Q8_0 as core production targets.
3. Model weights, KV cache, and reusable decode scratch remain GPU resident across tokens.
4. Decode uses fused attention and stable destination buffers, with CUDA Graph replay where the
   model/session shape is eligible.
5. Multiple requests use bounded scheduling, chunked prefill, continuous batching, cancellation,
   and fair KV admission.
6. Prefix reuse is safe, bounded, observable, and eventually supports page-granular copy-on-write
   and longest-prefix matching.
7. CPU/GGUF behavior remains a first-class fallback and does not regress.
8. `/v1/chat/completions`, streaming, `/v1/models`, CLI behavior, and external OpenAI proxy behavior
   remain compatible with existing xeno-agent-sdk consumers.
9. Explicit CUDA requests never silently run on CPU. Unsupported models/configurations fail with
   precise errors; `auto` may choose CPU when CUDA is unavailable or unsupported.
10. Performance and semantic correctness are demonstrated on the same models/hardware against the
    CPU backend and external runtimes, with reproducible JSON evidence.

## Non-negotiable constraints

- Never break OpenAI API compatibility.
- Never remove GGUF support.
- CPU fallback must always build and work without CUDA installed.
- CUDA is optional and feature-gated.
- Never claim CUDA execution while silently falling back to CPU.
- Benchmark and parity-check performance-sensitive changes.
- Preserve bounded memory, cancellation, and error behavior under concurrency.
- Avoid unbounded local GPU/model runs because this PC has become unstable under heavy workloads.

## Phase plan followed in this work

### Phase 0 - Measurement and acceptance harness

Establish backend-comparison benchmarks, JSON output, token accounting, prefill/decode timing,
memory and transfer telemetry, deterministic seeds, real-model semantic checks, and safe bounded
runner scripts. This phase provides the evidence required before optimization claims are accepted.

### Phase 1 - Runtime/backend boundary

Separate the model-facing runtime from CPU implementation details through `BackendKind`,
`CausalLmBackend`, and `BackendSession`. Preserve CPU parity, expose backend/resource status, and
make backend selection explicit through CLI/environment/server configuration.

### Phase 2 - Resident weights and native single-token CUDA decode

Add persistent device buffers, GGUF tensor upload, format-aware resident matrices, reusable decode
scratch, native embedding/norm/projection/FFN/attention kernels, and a full single-token layer
executor. Bring up correctness format by format and architecture by architecture.

### Phase 3 - GPU-resident paged KV cache

Move key/value state to device memory, add direct page-table addressing, enforce KV budgets, support
F32 and quantized KV modes, expose allocation status, and preserve session isolation and overflow
checks.

### Phase 4 - Fused decode attention

Fuse score calculation, masking, online softmax, and value accumulation for batch-1 decode. Support
standard GQA and Gemma geometry, sliding attention, large head widths, and quantized KV modes while
keeping eager fallback paths.

### Phase 5 - CUDA Graph replay

Capture stable batch-1 decode work, reuse fixed device pointers, update token/position inputs, replay
when eligible, and always fall back to eager CUDA without changing output semantics.

### Phase 6 - Chunked prefill and continuous batching

Bound prefill chunks, add request/KV admission, rendezvous compatible decode requests, compose child
graphs under a parent batch execution path, preserve fairness, cancellation, streaming backpressure,
and resource cleanup.

### Phase 7 - Prefix reuse

Create immutable structural-prefix snapshots, bounded LRU admission, CPU page sharing/COW, CUDA
snapshot restore, observability, and correctness tests. Evolve exact-key reuse into page-granular
GPU COW and radix longest-prefix matching.

### Phase 8 - Advanced model and weight formats

Add bounded SafeTensors/Hugging Face loading and native CUDA execution for AutoAWQ, GPTQ, and
compressed-tensors W4A16. Validate metadata/layout variants independently and retain strict
unsupported errors for ambiguous formats.

### Production hardening and release

Complete central allocation/scratch residency, broaden graphs/batching/model support, run comparative
and soak matrices on the GPU runner, cleanly integrate the checkpoint onto protected `main`, then cut
future releases using `release-guide/` from `xeno-platform`.

## DONE - implementation state

The implementation is well beyond scaffolding. The initial dense native CUDA target has been built
and exercised on the RTX runner. The remaining work is production breadth, allocator/scratch
completion, optimization, and integration hardening.

### Runtime and native CUDA capabilities completed

- Backend and backend-session abstractions exist with CPU, native CUDA, and external OpenAI modes.
- Explicit CUDA feature/build selection and fast, precise non-CUDA errors exist.
- Persistent device buffers and resident tensor/matrix wrappers exist.
- Native dense CUDA decode supports F32, F16, BF16, Q8_0, Q4_0, Q4_K, Q5_K, and Q6_K paths.
- Real VibeThinker 3B and Gemma4 12B Q4_K_M runs produced multi-position semantic parity and bounded
  multi-token generation evidence on the RTX runner.
- GPU KV supports F32, Q8, KQ4-VQ8, and agent-adaptive hot/cold page tables with direct addressing.
- Fused online-softmax decode attention exists for standard and Gemma geometries through head width
  512 and across the implemented KV modes.
- Batch-1 CUDA Graph replay exists for the standard dense F32-KV path.
- Continuous batching and child-graph parent composition exist for compatible standard dense F32-KV
  sessions.
- Exact immutable prefix snapshots, bounded LRU admission, and CPU page COW are implemented and
  validated.
- SafeTensors/Hugging Face loading and native AutoAWQ GEMM/GEMV, GPTQ v1/v2 including act-order,
  and compressed-tensors W4A16 execution paths are implemented and fixture-tested.
- External OpenAI proxy/benchmark mode is implemented at the HTTP boundary.
- Persistent GPU allocation, transfer counters, process memory, and resource/scheduler telemetry are
  exposed through runtime/server/benchmark status.
- Safe local scripts and self-hosted GitHub GPU workflows exist for serial, bounded validation.

## DONE - complete file inventory in this checkpoint

Every currently modified or untracked file is listed here. These files form the worktree being
checkpointed; some changes accumulated before the recovered conversation and are included because
the user explicitly requested committing the entire tree.

### Modified tracked files

- `.github/workflows/cuda.yml` - Manual self-hosted NVIDIA workflow with PTX reproducibility and
  serial synthetic/real CUDA gates, fixture controls, and scheduler/graph/prefix smoke options.
- `Cargo.toml` - Adds workspace crates/dependencies and forwards the root optional CUDA feature.
- `crates/xrt-capi/src/lib.rs` - Adapts the C API to generation returning a generated-token count.
- `crates/xrt-cli/Cargo.toml` - Adds JSON/serialization and shared external OpenAI client dependencies.
- `crates/xrt-cli/src/main.rs` - Adds backend selection, local/external benchmark modes, concurrency,
  scheduler/prefix/GPU/memory/transfer JSON, token accounting, and external SSE handling.
- `crates/xrt-cuda/Cargo.toml` - Adds GGUF/kernel dependencies needed by resident quantized tensors
  and reference validation.
- `crates/xrt-cuda/src/lib.rs` - Implements the native CUDA driver backend: persistent buffers,
  resident dense/quant matrices, AWQ/GPTQ/compressed formats, paged KV modes, fused attention,
  graphs/batching, transfer telemetry, feature-disabled stubs, and CUDA tests.
- `crates/xrt-gguf/src/lib.rs` - Adds boolean metadata-array extraction required by per-layer Gemma4
  configuration.
- `crates/xrt-hub/src/lib.rs` - Adds environment-configurable local model aliases for real
  VibeThinker/Gemma fixtures and tests.
- `crates/xrt-kernels/src/cpu/mod.rs` - Exports CPU GELU and GEGLU helpers.
- `crates/xrt-kernels/src/cpu/silu.rs` - Adds PyTorch-compatible tanh GELU/GEGLU for Gemma4.
- `crates/xrt-kernels/src/lib.rs` - Re-exports the new CPU activation kernels.
- `crates/xrt-models/Cargo.toml` - Adds SafeTensors/Hugging Face metadata support.
- `crates/xrt-models/src/lib.rs` - Exports Gemma4 trace types used by parity diagnostics.
- `crates/xrt-models/src/llama.rs` - Adds Qwen2/Qwen2.5 and Gemma4 config/CPU decode, per-layer
  geometry, sliding attention, GELU/post norms, trace diagnostics, HF conversion, and quant metadata.
- `crates/xrt-runtime/Cargo.toml` - Wires optional CUDA and runtime dependencies for CUDA kernels,
  SafeTensors, scheduling, and tests.
- `crates/xrt-runtime/src/grammar.rs` - Removes an unused rule-name field to keep builds warning-clean.
- `crates/xrt-runtime/src/kv_cache.rs` - Adds Arc page sharing/COW for CPU F32/Q8/KQ4-VQ8 prefix
  snapshots, geometry/accounting, cache-mode hashing/aliases, and tests.
- `crates/xrt-runtime/src/lib.rs` - Exports backend/GPU/prefix/scheduler APIs; loads GGUF/SafeTensors;
  selects CPU/CUDA; and manages session/resource/prefix lifecycle.
- `crates/xrt-runtime/src/policy.rs` - Makes policy/span types hashable and equatable for deterministic
  structural-prefix keys.
- `crates/xrt-runtime/src/session.rs` - Routes backend sessions; implements chunked/scheduled
  prefill/decode, continuous batching, prefix attach/store, graph capacity, cancellation, overflow
  checks, resource status, and generated-token accounting.
- `crates/xrt-server/Cargo.toml` - Adds the shared external OpenAI client dependency.
- `crates/xrt-server/src/main.rs` - Adds backend/external loading and status, scheduler/stream
  backpressure/cancellation, OpenAI proxy endpoints, GPU/prefix telemetry, image safety, and the
  existing remove-background endpoint.
- `crates/xrt-tokenizer/Cargo.toml` - Adds JSON and tempfile support for Hugging Face tokenizer tests.
- `crates/xrt-tokenizer/src/chat_template.rs` - Supports Python/Jinja mapping `.get()` through a
  MiniJinja filter and tests it.
- `crates/xrt-tokenizer/src/lib.rs` - Adds a bounded Hugging Face BPE/tokenizer/config/added-token/chat
  template loader with parity and rejection tests.
- `tests/common/mod.rs` - Adds synthetic GGUF builders for dense/quantized types and Gemma4 fixtures.
- `tests/gguf_parse_test.rs` - Validates the added GGUF boolean metadata-array parsing.
- `tests/kv_cache_test.rs` - Covers KV cache-mode aliases.
- `tests/model_architecture_test.rs` - Covers Qwen2/Gemma4 load/decode and unsupported multimodal
  architecture behavior.
- `tests/smoke_e2e.rs` - Adds broad CPU/CUDA parity, KV, graph, batch, prefix, Gemma4, real GGUF,
  SafeTensors, AWQ, GPTQ, compressed-format, semantic, and diagnostic tests.

### New untracked files and directories

- `XENO-MONETIZATION-AND-ACCOUNT.md` - Separate product monetization/account architecture notes;
  unrelated to CUDA but included because the full worktree must be preserved.
- `crates/xrt-cli/src/process_memory.rs` - Windows/Linux current and lifetime-peak process resident
  memory sampling for benchmark telemetry.
- `crates/xrt-cuda/src/kernels/awq_gemm4.cu` - AutoAWQ GEMM-layout packed 4-bit matvec source.
- `crates/xrt-cuda/src/kernels/awq_gemv4.cu` - AutoAWQ GEMV row-major padded 4-bit matvec source.
- `crates/xrt-cuda/src/kernels/compressed_tensors_w4a16.cu` - Signed-offset compressed-tensors W4A16
  group-index matvec source.
- `crates/xrt-cuda/src/kernels/gptq_gemm4.cu` - GPTQ v1 standard-group packed 4-bit matvec source.
- `crates/xrt-cuda/src/kernels/gptq_explicit_gemm4.cu` - GPTQ act-order/v2 explicit group-index and
  configurable zero-encoding matvec source.
- `crates/xrt-cuda/src/kernels/generated/awq_gemm4.ptx` - Checked-in reproducible PTX for AWQ GEMM.
- `crates/xrt-cuda/src/kernels/generated/awq_gemv4.ptx` - Checked-in reproducible PTX for AWQ GEMV.
- `crates/xrt-cuda/src/kernels/generated/compressed_tensors_w4a16.ptx` - Checked-in reproducible PTX
  for compressed-tensors W4A16.
- `crates/xrt-cuda/src/kernels/generated/gptq_gemm4.ptx` - Checked-in reproducible PTX for GPTQ v1.
- `crates/xrt-cuda/src/kernels/generated/gptq_explicit_gemm4.ptx` - Checked-in reproducible PTX for
  GPTQ act-order/v2.
- `crates/xrt-openai/Cargo.toml` - Manifest for the shared external OpenAI HTTP client crate.
- `crates/xrt-openai/src/lib.rs` - Pooled bounded HTTP client with auth, loopback/remote policy,
  redirect/timeouts, response caps, and redacted errors.
- `crates/xrt-runtime/src/backend.rs` - Defines backend kinds/traits/sessions and CPU/native CUDA
  implementations, including resident weights/scratch, KV modes, graph/batch state, decode,
  budgets, and telemetry.
- `crates/xrt-runtime/src/gpu_resource.rs` - Defines graph/environment configuration, GPU allocation
  and transfer status, and the runtime resource manager.
- `crates/xrt-runtime/src/prefix_cache.rs` - Implements exact structural-prefix keys, immutable
  snapshots, bounded LRU behavior, and status.
- `crates/xrt-runtime/src/resident_tensor.rs` - Normalizes and validates GGUF/HF tensors plus
  AWQ/GPTQ/compressed metadata into a format-neutral resident mapping.
- `crates/xrt-runtime/src/scheduler.rs` - Implements request/KV admission, prefill/decode fairness,
  rendezvous batching, status, and RAII permits.
- `crates/xrt-safetensors/Cargo.toml` - Manifest for bounded SafeTensors/Hugging Face loading.
- `crates/xrt-safetensors/src/lib.rs` - Loads bounded single/sharded bundles, typed HF config and
  quant metadata, with path-containment and geometry validation.
- `crates/xrt-server/src/external_openai.rs` - Implements buffered/SSE upstream proxying while
  preserving status/body bytes and enforcing caps/backpressure/error mapping.
- `docs/GEMMA4_SUPPORT_SPEC.md` - Gemma4 design, implementation status, constraints, and test plan.
- `docs/GPU_RUNTIME_ACCELERATION_SPEC.md` - Authoritative native CUDA final goal, phase plan,
  implementation log, runner evidence, known limitations, and definition of done.
- `scripts/prepare-real-awq-fixture.ps1` - Atomically acquires size/SHA/revision-pinned AWQ GEMM/GEMV
  fixtures and GGUF references.
- `scripts/prepare-real-compressed-tensors-fixture.ps1` - Acquires pinned W4A16 and BF16 reference
  fixtures.
- `scripts/prepare-real-gptq-fixture.ps1` - Acquires a pinned GPTQ v1 fixture and GGUF reference.
- `scripts/prepare-real-gptq-variants-fixture.ps1` - Acquires pinned act-order/dense fixtures and
  performs deterministic GPTQ v1-to-v2 qzero conversion.
- `scripts/safe-cuda-check.ps1` - Serial bounded compile/test/PTX/optional real-parity gate with
  timeout, process-leak guards, and optional soak behavior.
- `scripts/safe-cuda-smoke.ps1` - Explicitly opted-in bounded real-model benchmark with cache,
  graph, prefix, and concurrency controls; requires `-ConfirmGpuRun`.
- `scripts/safe-cuda-server-smoke.ps1` - Explicitly opted-in bounded concurrent OpenAI SSE server
  smoke with scheduler/resource cleanup checks.
- `docs/SESSION-STATE-2026-07-17.md` - This recovery checkpoint and zero-context continuation guide.

## IN PROGRESS at the moment of preservation

The active engineering seam is **complete scratch residency and a central GPU allocation arena**.
The foundational native CUDA decode, resident weights, paged KV, fused attention, initial graphs,
batching, prefix snapshots, and initial advanced formats are already implemented. Do not restart
those phases as if only scaffolding exists.

Files involved:

- `crates/xrt-runtime/src/backend.rs`
- `crates/xrt-runtime/src/gpu_resource.rs`
- `crates/xrt-cuda/src/lib.rs`
- `crates/xrt-runtime/src/session.rs` where session lifecycle/accounting changes are required

The next concrete edit was going to:

1. Extend session-owned `CudaDecodeScratch` with stable, preallocated device destinations for token
   embedding output, single-query attention output, FFN down-projection output, and the
   post-residual output that still use transient per-layer/per-token allocations.
2. Add or extend destination-buffer variants of the corresponding CUDA kernel APIs so these kernels
   write into those stable slices instead of allocating and returning new buffers.
3. Route those allocations through one central GPU arena/high-water tracker in the runtime resource
   manager.
4. Report exact transient allocation and peak VRAM telemetry rather than only persistent tracked
   allocations plus device-wide samples.
5. Preserve eager execution and current output semantics before extending graph capture to use the
   new stable destinations.

Acceptance for that edit: no per-token allocation remains in the supported standard dense decode
loop for these stages; CPU behavior is unchanged; CUDA eager parity stays within established
tolerances; graph-ineligible configurations still work; resource counters return to baseline after
session drop/cancellation.

## NOT DONE - ordered continuation TODO

1. **Finish the central GPU scratch/allocation arena.** Remove remaining embedding,
   attention-output, FFN-down, and post-residual transient device allocations; make ownership and
   lifetimes session-safe; add exact current/peak transient accounting and cleanup tests.
2. **Broaden CUDA Graph eligibility.** Extend capture/replay beyond standard dense batch-1 F32 KV to
   Gemma4 variable-width layers, Q8/KQ4/adaptive KV, and larger compatible batch sizes. Every graph
   path must retain a correct eager fallback.
3. **Broaden continuous batching and graph composition.** Generalize parent/child graph execution
   beyond standard dense F32 KV. Hybrid/recurrent models need explicit session-owned recurrent state
   before they can join this path safely.
4. **Implement a shared GPU page allocator and page-granular COW.** Current CUDA prefix restore can
   copy a whole session allocation on first write; adaptive KV reserves full hot and cold stores.
   Replace this with central physical pages, refcounts, bounded eviction, and page-level duplication.
5. **Optimize correctness-first kernels.** Profile and tune quantized GEMV/GEMM, F32 paths, packed
   Q6, embedding, and output projection with warp-level reductions or validated vendor kernels.
   Never reintroduce unvalidated handwritten PTX solely for apparent speed.
6. **Broaden architecture support.** Add Qwen3.5 DeltaNet/hybrid recurrent execution, MoE routing,
   additional Gemma/Qwen layouts, attention heads wider than 512 where needed, and more sliding-window
   variants with explicit support checks.
7. **Complete advanced-format breadth.** Validate additional AWQ layouts, broaden independent GPTQ
   v2 coverage, evaluate EXL3, and decide whether SafeTensors CPU decode is a product requirement.
   SafeTensors model directories are currently native-CUDA-only by design.
8. **Upgrade prefix reuse to a radix cache.** Add longest-prefix matching, shared GPU pages, partial
   prefix attachment, multimodal prefix state, and eviction/admission behavior under concurrent load.
9. **Add optional F16 KV and later TurboQuant.** Do this only after allocator/fused-path maturity.
   Agent-adaptive hot/cold KV is the foundation but is not yet fully memory-saving because both stores
   reserve capacity.
10. **Run the full comparative benchmark matrix.** Compare xeno-rt with llama.cpp, vLLM, SGLang, and
    ExLlama on the same GPU/model/context and record TTFT, decode tok/s, memory, transfer volume,
    concurrency scaling, and semantic output.
11. **Complete production reliability gates.** Run bounded soak, OOM admission, cancellation,
    disconnect/backpressure, session churn, prefix eviction, graph fallback, and multi-session cleanup
    tests on hosted/self-hosted runners rather than the unstable local machine.
12. **Integrate this checkpoint onto current protected `main`.** The current branch is stale and
    mixes an old vision branch name with CUDA work. Rebase or cherry-pick deliberately, resolve the
    separately released `v0.2.0` state, open a focused PR, obtain green CPU and GPU checks, and avoid
    rewriting already-published release history.

## Important decisions, constraints, and gotchas

### Backend and compatibility rules

- Explicit CUDA must never silently execute CPU code. `auto` can fall back; `cuda` must report why a
  model/build/device is unsupported.
- The CPU/GGUF path and OpenAI-compatible API are hard compatibility boundaries.
- CUDA remains optional through the root `cuda` feature and forwarded crate features such as
  `xrt-runtime/cuda` and `xrt-cuda/cuda`.
- A binary built without CUDA fails fast when CUDA is explicitly selected, before expensive model
  loading.
- SafeTensors/Hugging Face directory loading is currently CUDA-only; CPU returns a clear unsupported
  error instead of pretending parity.

### CUDA module and build lessons

- CUDA kernel modules are loaded lazily. A malformed/unsupported kernel should fail at its call site,
  not prevent CUDA device initialization.
- Driver JIT error logs are captured in CUDA errors because `CUDA_ERROR_INVALID_PTX` alone was not
  actionable.
- An early Q8_0 kernel used NVRTC. The local machine lacked `nvrtc64*.dll`, and cudarc panicked while
  loading NVRTC. The panic was converted to `XrtError`; production kernels then moved to checked-in,
  build-validated PTX. Do not add a runtime NVRTC requirement back without packaging it explicitly.
- Existing legacy RMSNorm PTX once failed `CUDA_ERROR_INVALID_PTX` during eager device initialization;
  lazy loading limited the failure to the primitive and allowed independent kernel validation.
- Generated AWQ/GPTQ/compressed PTX must remain reproducible against its checked-in CUDA source; the
  workflow has a dedicated verification gate.

### Performance experiments that failed or regressed

- F32 `m == 1` atomics/grouped fast paths and the first packed Q6 prototypes caused regressions or
  runs longer than ten minutes and were reverted. Profile first and use build-validated PTX/CUBIN or
  cuBLAS where appropriate.
- Packed Q4_K embedding had an approximately 60-second first-use stall. Embeddings switched to a
  bounded expanded row-major/transposed F32 representation while packed matrix paths remained for
  projections.
- The early real VibeThinker CUDA path took roughly 411.8 seconds per token before shared-memory
  reduction and packed-kernel work. Subsequent Q4_K and graph/fused improvements brought bounded
  generation into seconds, but performance evidence must still be measured per current commit.
- Q4_K/Q6_K expanded embedding allocations are capped at 4 GiB; larger tables use the packed path.
  Q5_K embedding is still expanded.

### Correctness and acceptance rules

- CPU and CUDA full logit vectors for K-quants can differ partly because the CPU fused path quantizes
  activations to Q8. Acceptance therefore combines float-reference kernel parity, bounded tolerance,
  greedy winner parity, and generated-text parity rather than requiring bitwise full-vector identity.
- Every unsupported tensor layout, architecture, head width, quant metadata variant, or graph shape
  should be rejected explicitly before unsafe execution.
- CUDA Graph capture/replay is an optimization only. Capture failure or ineligibility must always use
  eager CUDA and must never break generation.
- Gemma4 uses per-layer variable geometry, sliding/full attention patterns, post norms, and different
  activation behavior. It cannot be treated as a fixed Llama layer loop.

### KV, prefix, batching, and telemetry limits

- Current exact prefix snapshots are validated, but they are not yet a full radix longest-prefix
  cache.
- CUDA COW may currently duplicate a whole session allocation on first write; a shared physical-page
  allocator is still required.
- Agent-adaptive KV uses full-capacity hot and cold backing stores, so its policy works but its final
  memory-saving allocator design is unfinished.
- Device-used VRAM is a device-wide sample, while tracked bytes represent xeno-rt persistent
  allocations. Do not present the former as precise per-process ownership.
- Process peak memory is the lifetime peak working set. Transfer counters are relaxed observational
  telemetry and do not include every implicit driver migration.
- External OpenAI mode is an HTTP-boundary proxy/benchmark adapter, not a token-level native backend.
  It is loopback-only by default, requires explicit remote opt-in, disables redirects, caps bodies at
  16 MiB, bounds SSE processing, and must never expose API keys in errors/logs.

### Machine and workflow safety

- The local PC has crashed during heavy compilation/GPU/model runs. Routine validation should use
  GitHub-hosted CPU jobs and the registered self-hosted NVIDIA runner.
- Local safety scripts force serial behavior with `CARGO_BUILD_JOBS=1` and `RUST_TEST_THREADS=1`, use
  bounded timeouts and process guards, and require `-ConfirmGpuRun` before real GPU work.
- Do not run full workspace compilation, multiple Cargo commands in parallel, large model loading,
  server soak, or ignored CUDA tests locally unless the user explicitly accepts the risk.

### Git and release context

- This branch is stale relative to protected `main`; it contains cumulative work not represented by
  its old vision-task name.
- The user explicitly authorized committing all tracked and untracked changes in this checkout,
  including unrelated files, because preservation takes priority over commit scope.
- GitHub release `v0.2.0` was already published from main in a separate clean worktree. No R2 or XENO
  Hub publication was authorized or performed. Do not move/recreate that tag from this branch.
- Future releases must follow every file in `release-guide/` and run from `xeno-platform`; do not
  improvise release commands.

## Validation and evidence state

Prior work recorded successful default and CUDA-feature compilation, CPU/runtime/workspace tests,
safe benchmark smoke behavior, and multiple self-hosted RTX runs. Exact run IDs and dated progress
entries through 2026-07-12 are recorded in `docs/GPU_RUNTIME_ACCELERATION_SPEC.md` (including RTX
workflow runs through `29195704139`). That specification is the authoritative evidence ledger.

No validation command was rerun while producing this preservation checkpoint because the immediate
goal is data safety and the local machine has shown instability. The next agent should begin with
static review, then use `.github/workflows/cuda.yml` and the bounded `scripts/safe-cuda-*.ps1` gates.

## Recorded working tree before checkpoint

The following commands were run immediately before creating this file.

### `git status --short --branch`

```text
## feat/v0.2.0-vision-tasks-remove-background...origin/feat/v0.2.0-vision-tasks-remove-background
 M .github/workflows/cuda.yml
 M Cargo.toml
 M crates/xrt-capi/src/lib.rs
 M crates/xrt-cli/Cargo.toml
 M crates/xrt-cli/src/main.rs
 M crates/xrt-cuda/Cargo.toml
 M crates/xrt-cuda/src/lib.rs
 M crates/xrt-gguf/src/lib.rs
 M crates/xrt-hub/src/lib.rs
 M crates/xrt-kernels/src/cpu/mod.rs
 M crates/xrt-kernels/src/cpu/silu.rs
 M crates/xrt-kernels/src/lib.rs
 M crates/xrt-models/Cargo.toml
 M crates/xrt-models/src/lib.rs
 M crates/xrt-models/src/llama.rs
 M crates/xrt-runtime/Cargo.toml
 M crates/xrt-runtime/src/grammar.rs
 M crates/xrt-runtime/src/kv_cache.rs
 M crates/xrt-runtime/src/lib.rs
 M crates/xrt-runtime/src/policy.rs
 M crates/xrt-runtime/src/session.rs
 M crates/xrt-server/Cargo.toml
 M crates/xrt-server/src/main.rs
 M crates/xrt-tokenizer/Cargo.toml
 M crates/xrt-tokenizer/src/chat_template.rs
 M crates/xrt-tokenizer/src/lib.rs
 M tests/common/mod.rs
 M tests/gguf_parse_test.rs
 M tests/kv_cache_test.rs
 M tests/model_architecture_test.rs
 M tests/smoke_e2e.rs
?? XENO-MONETIZATION-AND-ACCOUNT.md
?? crates/xrt-cli/src/process_memory.rs
?? crates/xrt-cuda/src/kernels/
?? crates/xrt-openai/
?? crates/xrt-runtime/src/backend.rs
?? crates/xrt-runtime/src/gpu_resource.rs
?? crates/xrt-runtime/src/prefix_cache.rs
?? crates/xrt-runtime/src/resident_tensor.rs
?? crates/xrt-runtime/src/scheduler.rs
?? crates/xrt-safetensors/
?? crates/xrt-server/src/external_openai.rs
?? docs/GEMMA4_SUPPORT_SPEC.md
?? docs/GPU_RUNTIME_ACCELERATION_SPEC.md
?? scripts/
```

### `git diff --stat`

```text
 .github/workflows/cuda.yml                |   514 +-
 Cargo.toml                                |     7 +
 crates/xrt-capi/src/lib.rs                |     2 +-
 crates/xrt-cli/Cargo.toml                 |     3 +
 crates/xrt-cli/src/main.rs                |  1410 ++-
 crates/xrt-cuda/Cargo.toml                |     2 +
 crates/xrt-cuda/src/lib.rs                | 16127 ++++++++++++++++++++++++++--
 crates/xrt-gguf/src/lib.rs                |     4 +
 crates/xrt-hub/src/lib.rs                 |    76 +
 crates/xrt-kernels/src/cpu/mod.rs         |     2 +-
 crates/xrt-kernels/src/cpu/silu.rs        |    13 +
 crates/xrt-kernels/src/lib.rs             |     7 +-
 crates/xrt-models/Cargo.toml              |     1 +
 crates/xrt-models/src/lib.rs              |     2 +-
 crates/xrt-models/src/llama.rs            |  1463 ++-
 crates/xrt-runtime/Cargo.toml             |    10 +-
 crates/xrt-runtime/src/grammar.rs         |     2 -
 crates/xrt-runtime/src/kv_cache.rs        |   195 +-
 crates/xrt-runtime/src/lib.rs             |   276 +-
 crates/xrt-runtime/src/policy.rs          |     8 +-
 crates/xrt-runtime/src/session.rs         |   611 +-
 crates/xrt-server/Cargo.toml              |     1 +
 crates/xrt-server/src/main.rs             |   595 +-
 crates/xrt-tokenizer/Cargo.toml           |     4 +
 crates/xrt-tokenizer/src/chat_template.rs |    27 +-
 crates/xrt-tokenizer/src/lib.rs           |   472 +-
 tests/common/mod.rs                       |   821 +-
 tests/gguf_parse_test.rs                  |    32 +-
 tests/kv_cache_test.rs                    |     4 +
 tests/model_architecture_test.rs          |    45 +
 tests/smoke_e2e.rs                        |  1906 +++-
 31 files changed, 23578 insertions(+), 1064 deletions(-)
```

`git diff --stat` only reports tracked-file changes. The untracked inventory is recorded separately
above and will be included by `git add -A` in the preservation commit.

## Resume procedure for a new agent

1. Read this file completely.
2. Read `docs/GPU_RUNTIME_ACCELERATION_SPEC.md` completely, especially the dated progress log,
   acceptance gates, and remaining limitations.
3. Read `docs/GEMMA4_SUPPORT_SPEC.md` before changing Gemma behavior.
4. Inspect the preservation commit and compare it with current `main`; do not rewrite `v0.2.0`.
5. Start at the scratch-residency/central-arena task described under **IN PROGRESS**.
6. Keep CPU and non-CUDA checks green through hosted CI.
7. Use the self-hosted GPU workflow for PTX, synthetic parity, real-model, graph, batching, prefix,
   and advanced-format gates.
8. Record every material result and runner ID back into
   `docs/GPU_RUNTIME_ACCELERATION_SPEC.md` before proceeding to the next milestone.
