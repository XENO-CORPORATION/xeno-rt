# XENO RT Roadmap

- **Status:** Canonical repository roadmap
- **Last reviewed:** 2026-07-23
- **Scope:** Shared runtime, `xrt-text`, `xrt-image`, `xrt-vision`,
  `xrt-video`, and `xrt-audio`

XENO RT is one native Rust runtime for hosting, exposing, and running local AI
models. This roadmap records what has shipped, what exists only on unreleased
`main`, and what evidence is still required before a capability becomes a
supported product claim.

The roadmap is capability-driven, not a promise of dates. The authoritative
support contract is the combination of the current release notes,
[Supported Models](SUPPORTED_MODELS.md), API documentation, and checked-in
evidence. A checked box means the repository has accepted the implementation
and its required evidence; source-file presence or a smoke test is not enough.

## Status vocabulary

| State | Meaning |
|---|---|
| **Shipped** | Published in an immutable GitHub release with release evidence. |
| **Supported** | Part of the maintained public contract for the listed model, format, backend, and workload. |
| **Beta** | Usable and tested, but its support matrix or operational evidence is intentionally narrower than the stable contract. |
| **Experimental** | Implementation exists and may execute real workloads, but production admission remains open. |
| **Planned** | Product boundary or design direction only; no support claim and no placeholder crate required. |

## Current product ledger

| Area | Current state | What exists now | What remains |
|---|---|---|---|
| `v0.2.0` | **Shipped** | Text-focused Linux/Windows CLI and server archives, checksums, SPDX SBOMs, and a successful tag workflow. | XENO Hub/R2 distribution still needs an agreed CLI/server versus desktop-installer contract. |
| `xrt-text` GGUF CPU | **Supported** | Local generation for the architecture, dtype, and quantization combinations in [Supported Models](SUPPORTED_MODELS.md); CPU is the default fallback. | Broader real-model conformance, fuzzing, soak coverage, and maintained regression budgets. |
| `xrt-text` GGUF CUDA | **Beta** | Native resident CUDA execution, paged/quantized KV modes, CUDA Graphs, prefix reuse, scheduling, and telemetry. | Per-tuple production evidence, broader hardware coverage, long-running reliability, and promotion criteria. |
| `xrt-text` SafeTensors | **Beta, CUDA only** | Selected dense Qwen2/Qwen3, AutoAWQ, GPTQ, and compressed-tensors layouts. | Decide whether CPU SafeTensors belongs in the stable contract; expand real-model fixtures without weakening strict validation. |
| OpenAI-compatible text API | **Beta** | Completions, chat completions, SSE streaming, model discovery, usage fields, tool request fields, and lifecycle/status endpoints. | Broader conformance fixtures, shared inbound authentication, overload/soak evidence, and a supported remote deployment profile. |
| `xrt-text` multimodal chat/mmproj | **Experimental** | The text runtime can load a separate compatible mmproj GGUF for image-text input. | Admit each language-model/projection pair with preprocessing, parity, memory, API, and real-model evidence. |
| Hybrid MoE/Qwen3.5 acceleration | **Experimental, opt-in** | Exact CPU/GPU expert placement, recurrent-state ownership, graph/prefix handling, parity, and retained benchmark evidence. | Same-hardware performance admission and approved real-MoE hosted validation; `auto` remains disabled. |
| `xrt-image` | **Experimental, unreleased** | Native Qwen-Image-2512 generation and Qwen-Image-Edit-2511 execution foundations, component bundles, CLI/API surfaces, CPU-safe builds, optional CUDA, and deterministic development evidence. | Full-resolution quality, memory, performance, reliability, transport, and packaging admission. No image tier is production-advertised. |
| `xrt-vision` | **Experimental** | Self-contained ONNX background-removal task inference. | Admit task families individually with bundles, parity, CPU fallback, resource accounting, and mixed-workload evidence. |
| `xrt-video` | **Planned** | Runtime-domain and ownership boundary only. | Select and specify the first open model, implement a real adapter, and define temporal quality/performance gates. |
| `xrt-audio` | **Planned** | Runtime-domain and ownership boundary only. | Define speech/music/audio profiles, select the first open model, and implement streaming-aware execution and admission. |
| Native bindings | **Experimental** | C and Python workspace crates. | Package, version, test, and support them independently or keep them outside the stable contract. |

## Delivered checkpoints

### v0.1.0: native CPU text foundation

- [x] Rust workspace with GGUF parsing, tokenizer loading, model execution,
  sampling, CLI, server, tests, and benchmarks.
- [x] Memory-mapped GGUF loading with metadata, shape, offset, and range
  validation.
- [x] CPU execution for F32, F16, BF16, Q8_0, Q4_0, Q4_K, Q5_K, and Q6_K
  weights.
- [x] Llama-family generation and OpenAI-compatible completion/chat endpoints.

### v0.2.0: native CPU/CUDA release checkpoint

- [x] Preserve portable GGUF CPU inference and fallback.
- [x] Add native resident CUDA execution for the documented dense matrix.
- [x] Add F32 and quantized paged KV modes, prefix caching, bounded scheduling,
  decode batching, and compatible CUDA Graph replay.
- [x] Add selected CUDA-only SafeTensors adapters with strict metadata and
  tensor-layout validation.
- [x] Add structured CPU/CUDA benchmark, memory, transfer, allocation,
  scheduler, cache, and graph telemetry.
- [x] Add locked hosted CI, security/dependency policy, MSRV checks, CUDA
  feature compilation, guarded GPU validation, release archives, checksums,
  SBOMs, and tag publication.
- [x] Publish `v0.2.0` as a text-focused GitHub release while retaining CUDA,
  bindings, and task inference at their documented maturity levels.

Detailed historical evidence remains in
[GPU Runtime Acceleration](GPU_RUNTIME_ACCELERATION_SPEC.md),
[Repository Hardening](REPOSITORY_HARDENING_SPEC.md), and the
[v0.2.0 release notes](releases/0.2.0.md).

### Unreleased `main`: multimodal and hybrid foundations

- [x] Define `xrt-text`, `xrt-image`, `xrt-video`, and `xrt-audio` as one
  shared-runtime product boundary.
- [x] Add the real feature-gated `xrt-image` crate rather than wrapping
  Diffusers, ComfyUI, or stable-diffusion.cpp.
- [x] Add Qwen-Image generation/edit component loading, native execution
  foundations, deterministic fixtures, CLI commands, and additive HTTP
  surfaces.
- [x] Add exact opt-in hybrid MoE and Qwen3.5 foundations inspired by
  KTransformers architecture while preserving native Rust ownership.
- [x] Keep image support and automatic hybrid placement experimental until
  their evidence gates pass.

## Execution order

Work should be prioritized in this order unless a release blocker changes it:

1. Preserve and harden the released `xrt-text` contract.
2. Admit the first production `xrt-image` generation and edit tiers.
3. Mature shared multi-domain hosting and `xrt-vision` task inference.
4. Add the first real `xrt-video` adapter.
5. Add the first real `xrt-audio` adapter.

Video and audio planning must not interrupt the text support contract or bypass
image admission. Empty facade crates do not count as progress.

## Track 1: `xrt-text` production completeness

**Goal:** make XENO RT a production-grade local LLM host for its explicitly
supported model/source/backend matrix, without claiming universal model
compatibility.

### Compatibility and correctness

- [ ] Maintain pinned real-model fixtures for every advertised architecture,
  source format, quantization, and backend tuple.
- [ ] Add broader non-streaming and raw-SSE OpenAI conformance fixtures,
  including errors, usage accounting, chat templates, and tool request fields.
- [ ] Define numerical tolerances per model family, dtype, backend, prefill
  path, and decode path.
- [ ] Add fuzz/property coverage for GGUF metadata, tensor bounds, tokenizer
  inputs, template parsing, and HTTP limits.
- [ ] Decide whether CPU SafeTensors is in scope; either implement and admit it
  or keep the unsupported boundary explicit.
- [ ] Admit compatible multimodal-chat language-model/mmproj pairs separately
  with image preprocessing, parity, memory, API, and real-model evidence.
- [ ] Expand architecture coverage only through strict loader validation,
  real-model evidence, and an updated support matrix.

### Operations and hosting

- [ ] Add one shared inbound authentication policy for every enabled modality
  while preserving loopback-first local use.
- [ ] Publish a supported remote-hosting profile covering authentication, TLS
  termination, network isolation, secrets, request limits, and observability.
- [ ] Pass long-running load/unload, cancellation, disconnect, queue
  saturation, fragmentation, overload, and graceful-shutdown soaks.
- [ ] Define multi-model residency and eviction policy without allowing one
  request or modality to silently overcommit the device.
- [ ] Resolve the XENO Hub/R2 delivery contract for CLI/server archives versus
  a future installer and model catalog.
- [ ] Either ship versioned C/Python packages with compatibility tests or keep
  them explicitly experimental and outside release artifacts.

### Performance and accelerator promotion

- [ ] Publish controlled CPU and CUDA baselines for the maintained model
  matrix with throughput, TTFT, prefill, decode, memory, and latency budgets.
- [ ] Promote CUDA tuples from beta only after real-model parity, memory,
  reliability, and same-hardware regression gates pass.
- [ ] Complete the hybrid-MoE same-hardware performance gate and approved
  hosted real-MoE workflow.
- [ ] Enable automatic hybrid placement only when it beats the registered
  baseline without violating p95 latency, quality, or memory bounds.
- [ ] Evaluate agent-adaptive KV policies on pinned Qwen3.5 agent/tool
  workloads before changing the default from F32.

**Track exit:** the supported matrix has repeatable correctness, API,
security, packaging, operational, and benchmark evidence. Unsupported
architectures and formats still fail explicitly.

## Track 2: `xrt-image` production admission

**Goal:** ship native local Qwen image generation and editing with an honest,
evidence-backed support matrix. The first target is Qwen-Image-2512 and
Qwen-Image-Edit-2511; the first practical RTX 4090 candidate is Q4_K_M.

### Already implemented

- [x] CPU-safe `xrt-image` crate with native Rust pipeline ownership.
- [x] Immutable multi-component bundles with revision, size, SHA-256, role,
  license-evidence, atomic install, and offline cache contracts.
- [x] Qwen-Image generation/edit graph foundations, FlowMatch scheduling,
  deterministic RNG, bounded image codecs, VAE paths, and quantized execution.
- [x] Feature-gated CLI and synchronous OpenAI-compatible generation/edit
  routes with stable unsupported-capability behavior.
- [x] Shared image allocation accounting and guarded CPU/CUDA validation
  foundations.
- [x] Bounded deterministic development evidence for BF16, Q8_0, Q6_K,
  Q5_K_M, and Q4_K_M paths; this evidence is not production admission.

### Reference and quality gates

- [x] Pin and execute a reproducible reference evaluator/export environment,
  including the complete OCR pipeline used by the frozen quality suite.
- [x] Add a resumable, hash-checkpointed, load-once corpus producer guarded for
  dedicated remote CUDA runners; pinned resumable CLIP, DINOv2, and direct
  PaddleOCR-VL metric exports; plus deterministic blinded-review packaging and
  three-rater response compilation.
- [ ] Produce the real 1024-pixel BF16 and quantized candidate corpus, metric
  exports, and blinded human ratings.
- [ ] Pass text rendering, prompt adherence, perceptual quality, faces/detail,
  determinism, and absolute-quality floors for every advertised tier.
- [ ] Admit quantizations independently. Q3/Q2 remain experimental unless
  separate evidence proves acceptable quality.

### RTX 4090 generation gates

- [ ] Capture a quiet and repeatable non-XENO VRAM baseline.
- [ ] Run the pinned full-resolution workload with exact peak allocation,
  transfer, load-time, denoising, VAE, and output telemetry.
- [ ] Complete the registered quiet 30-run native versus
  stable-diffusion.cpp comparison and satisfy its confidence-bound threshold.
- [ ] Keep the advertised Q4_K_M workload under the reserve-aware device cap,
  with no per-step weight transfers, leaks, or OOMs.
- [ ] Admit Q5_K_M, Q6_K, Q8_0, BF16, or staged/offloaded variants only when
  their complete-bundle evidence passes independently.

### Edit and conditioning gates

- [ ] Pass full-resolution single-, two-, and three-image Edit-2511 component,
  quality, identity, source-order attribution, determinism, and memory gates.
- [ ] Complete a bounded real CPU Q4 edit/generation fallback on a high-RAM
  runner or document a narrower CPU support contract before advertising it.
- [ ] Keep masked inpainting unsupported until the Qwen-Image-2512 inpaint
  profile independently passes mask normalization, leakage, quality, and API
  gates.

### API, scheduling, and release gates

- [ ] Pass mixed text/image admission, cancellation, unload, replacement,
  disconnect, queue, shutdown, and resource-cleanup soaks.
- [ ] Add measured image streaming usage and raw-SSE conformance before
  accepting `stream=true`; until then return the stable unsupported error.
- [ ] Either implement SSRF-safe HTTPS and operator-backed `file_id` resolvers
  or explicitly ship the multipart plus bounded local-data-URL edit subset.
- [ ] Treat asynchronous jobs and URL output storage as optional capabilities;
  if enabled, require ownership, quota, deadline, idempotency, TTL, and startup
  cleanup tests.
- [ ] Package reviewed manifests, model-install documentation, support matrix,
  API/CLI guidance, rollback behavior, and reproducible CUDA PTX evidence.
- [ ] Re-run every CPU, hosted CI, and approved RTX gate from a clean checkout
  before enabling image support by default.

**Track exit:** at least one complete Qwen-Image-2512 generation tier and one
complete Qwen-Image-Edit-2511 tier are production-advertised with quality,
memory, performance, API, security, and packaging evidence.

Qwen-Image-3.0 was announced as a hosted product, but as of the last review no
official local checkpoint/component tree was observed. It receives a separate
adapter only after official weights, configuration, tensor names, license,
processor assets, and a reproducible reference pipeline are public. It is
never aliased to the 2512 implementation. Ideogram work remains behind a
separate licensing decision.

The full gate definitions and retained evidence are in
[Qwen Image Inference](xrt-image-qwen-image-inference-spec.md).

## Track 3: `xrt-vision` task inference

**Goal:** run task-oriented AI models through XENO RT while keeping
deterministic media processing in `xeno-lib` and creative workflows in consumer
applications.

- [x] Establish an auxiliary task-inference crate and CPU fallback pattern
  through ONNX background removal.
- [ ] Add immutable task-model registry and bundle support without weakening
  one-file GGUF cache behavior.
- [ ] Give task sessions the shared per-device resource manager, explicit
  residency classes, eviction policy, and mixed-workload telemetry.
- [ ] Migrate task families one at a time with source parity, CPU fallback,
  bounded inputs, license/provenance, model hashes, and workload benchmarks.
- [ ] Prioritize the concrete families in
  [ONNX Integration](ONNX_INTEGRATION_PLAN.md): upscale, background removal,
  inpainting model execution, face restoration, depth, OCR, pose,
  transcription, stem separation, denoise, style transfer, frame
  interpolation, face detection/analysis, and colorization.
- [ ] Add segmentation and color-transfer artifacts before claiming the full
  historical task catalog.
- [ ] Never count a catalog entry, manifest, or migrated UI action as runtime
  support without an admitted model/backend tuple.

**Track exit:** each advertised task has immutable artifacts, parity, CPU
fallback where promised, resource accounting, API/CLI tests, and release
packaging.

## Track 4: shared runtime and product operations

**Goal:** let text, image, task, and future video/audio models share one host
without sharing incorrect execution semantics.

- [x] Share format, tensor, kernel, cache, telemetry, and resource primitives
  where concrete consumers prove the abstraction.
- [x] Keep causal text sessions separate from denoising image sessions.
- [ ] Complete one server-owned per-device resource manager for every enabled
  runtime and allocation class.
- [ ] Stabilize capability discovery, multi-model load/unload, draining,
  replacement, pinning, queueing, cancellation, and eviction contracts.
- [ ] Unify immutable model-bundle discovery, download, verification, offline
  cache, rollback, and catalog generation.
- [ ] Add common authentication, principal isolation, quotas, audit events,
  request identities, and safe non-loopback deployment controls.
- [ ] Publish modality-specific telemetry and regression budgets through one
  stable runtime-status surface.
- [ ] Decompose large CUDA, backend, server, and CLI modules in
  behavior-preserving PRs with unchanged public contracts.
- [ ] Add semver/API checks before publishing any Rust crate.

## Track 5: `xrt-video`

**Goal:** add native model-level video generation/transformation inference
after the shared image/runtime foundations are production-proven.

- [ ] Select the first open-weight target through an architecture, license,
  artifact-size, hardware, quality, and reference-implementation review.
- [ ] Create an implementation-ready adapter specification with immutable
  fixtures before creating a crate.
- [ ] Define frame/latent/temporal execution, scheduler, conditioning,
  cancellation, preview, and deterministic identity contracts.
- [ ] Use `xeno-lib` for decode/encode and deterministic media processing;
  keep model execution and scheduling in XENO RT.
- [ ] Add temporal memory placement, offloading, shared-device admission, and
  long-running cleanup behavior.
- [ ] Define seconds per clip, frames per second, step throughput, time to first
  preview, peak RAM/VRAM, temporal consistency, motion quality, and prompt
  adherence gates.
- [ ] Add additive APIs and CLI surfaces without changing existing text/image
  contracts.
- [ ] Admit the first model/backend/resolution/duration tuple only after
  correctness, quality, performance, security, and packaging pass.

No empty `xrt-video` crate is created before a real tested adapter exists.

## Track 6: `xrt-audio`

**Goal:** support local speech, music, and audio model inference through
profile-specific contracts rather than one ambiguous audio endpoint.

- [ ] Define initial profiles for speech recognition, speech synthesis,
  audio/music generation, enhancement, and separation.
- [ ] Select the first open-weight target and pin its license, artifacts,
  processor, reference implementation, and reproducible fixtures.
- [ ] Create a real adapter specification covering sample rate, channels,
  chunking, streaming, conditioning, deterministic identity, and cancellation.
- [ ] Use `xeno-lib` for codecs and deterministic DSP; keep neural execution in
  XENO RT.
- [ ] Add streaming-aware buffers, backpressure, latency, memory admission,
  mixed-workload scheduling, and cleanup.
- [ ] Define real-time factor, time to first audio, samples/seconds processed,
  peak RAM/VRAM, intelligibility, speaker/identity, music quality, and artifact
  gates per profile.
- [ ] Add stable APIs and CLI surfaces, then admit model/backend/profile tuples
  independently.

No empty `xrt-audio` crate is created before a real tested adapter exists.

## Release checkpoints

Versions are assigned only when their required evidence is achievable.
Experimental features may be present behind feature flags without becoming a
support promise.

| Checkpoint | Required outcome |
|---|---|
| **v0.2.0** | Shipped text-focused CPU baseline, beta native CUDA/SafeTensors, hardened repository, and portable GitHub archives. |
| **v0.3.0 target** | Text compatibility and operational-depth release: conformance, security/deployment policy, reliability soaks, regression baselines, and resolved distribution scope. Image may remain experimental. |
| **First image-support release** | At least one production-admitted Qwen generation tier and one Edit tier, with the Track 2 evidence and packaging complete. Version is chosen when the admission gate passes. |
| **Multimodal host checkpoint** | Stable shared resource manager, capability discovery, multi-model lifecycle, task registry, and mixed text/image/task operations. |
| **First video/audio releases** | One independently admitted adapter/profile tuple per domain; neither domain is bundled by implication. |

## v1.0 definition of ready

XENO RT is ready for a stable 1.0 claim when all of these are true:

- The maintained text model/source/quantization/backend matrix has real-model
  conformance and regression evidence.
- GGUF support and CPU fallback remain release-blocking invariants.
- The documented OpenAI-compatible text and image contracts have automated
  schema, raw-stream, error, and usage-accounting compatibility tests.
- At least one image generation tier and one image edit tier are
  production-admitted; planned video/audio domains may remain explicitly
  experimental or planned.
- Shared model lifecycle, resource limits, authentication, cancellation,
  overload, isolation, and shutdown behavior pass production soaks.
- Release artifacts are portable, reproducible, checksummed, SBOM-attested,
  provenance-attested, installable, rollback-safe, and supported on the
  declared platforms.
- Performance claims include reproducible raw evidence, correctness/quality
  parity, hardware/model/build identities, and maintained regression budgets.
- Large modules have clear ownership boundaries and maintainable tests.
- XENO Hub/platform catalog entries, model bundles, and actual distribution
  formats agree.

v1.0 does not mean every future model or modality is implemented. It means
every capability advertised as supported satisfies one enforceable production
contract.

## GitHub tracking policy

- This file is the canonical public roadmap in the repository.
- [Supported Models](SUPPORTED_MODELS.md) and release notes determine what
  users may rely on today; specifications and unchecked roadmap items do not.
- Implementation issues should link to one roadmap item and one acceptance
  gate rather than restating an entire specification.
- When implementation issues are opened, create and use domain labels such as
  `xrt-text`, `xrt-image`, `xrt-vision`, `xrt-video`, and `xrt-audio`, plus
  cross-cutting labels such as `runtime`, `security`, `performance`, and
  `release`.
- A PR that changes a capability's maturity must update this roadmap,
  Supported Models, relevant API docs, changelog, and evidence in the same
  review.
- Dependabot PRs are maintenance input, not the product roadmap.

## Out of scope

XENO RT does not own:

- ComfyUI-style workflow graphs or node editors;
- canvases, layers, masks, timelines, tracks, undo/redo, or project state;
- product-specific inpainting/outpainting UX;
- general image/video/audio codecs or deterministic media processing; or
- unsupported model claims based only on parsable files.

Consumer applications such as XENO Pixel/Edit/Motion/Sound own workflows and
call XENO RT for model execution. `xeno-lib` owns codecs and deterministic
media processing. See [Runtime Domains](RUNTIME_DOMAINS.md) for the canonical
ownership boundary.
