# XENO RT Runtime Domains

- **Status:** Canonical product and architecture boundary
- **Last updated:** 2026-07-22
- **Applies to:** Every runtime, model adapter, API, benchmark, and technical document in this repository

## Product definition

XENO RT is XENO's unified, headless runtime for hosting, exposing, and running
local AI models. It provides one shared Rust-native systems layer for model
loading, tensor execution, quantization, CPU/CUDA placement, scheduling, memory
accounting, model caching, telemetry, and network APIs.

XENO RT is not an editor, timeline, canvas, workflow builder, or media codec
suite. XENO applications create those product experiences and call XENO RT for
AI inference. `xeno-lib` owns non-AI media processing and format I/O.

## Public capability domains

| Domain | Responsibility | Current repository status |
|---|---|---|
| `xrt-text` | Language and conversational model inference | Implemented by the existing `xrt-runtime` and `xrt-models` paths. The public facade name is reserved; there is no separate `xrt-text` crate yet. |
| `xrt-image` | Image generation and model-level image conditioning/edit inference | A real feature-gated crate exists. Qwen Image generation and Edit execution are experimental and not production-admitted. |
| `xrt-video` | Video generation and generative transformation inference | Planned capability boundary. No crate or production model adapter exists yet. |
| `xrt-audio` | Speech, music, and audio model inference | Planned capability boundary. Existing task-model audio paths remain where they are until a tested facade is designed. |

These are public capability boundaries, not four unrelated inference engines.
They share XENO RT's formats, tensor types, kernels, device management, bundle
cache, scheduler, telemetry, security rules, and server.

`xrt-vision` remains an auxiliary task-inference domain for discriminative or
deterministic image operations such as segmentation, background removal, depth,
OCR, and upscaling. A model belongs to a domain according to its advertised
product capability and primary output, not every internal input type. For
example, Qwen Image Edit consumes images and text internally but belongs to
`xrt-image` because it produces a generated image.

## Ownership boundary

### XENO RT owns

- validated model and component loading;
- GGUF, SafeTensors, and explicitly supported task-model formats;
- model-family graph assembly and tensor mapping;
- quantized and full-precision execution;
- CPU fallback and optional accelerator backends;
- device placement, offloading, scheduling, batching, and memory accounting;
- immutable model bundles, cache integrity, and offline execution;
- capability discovery, load/unload, inference, progress, and cancellation APIs;
- OpenAI-compatible contracts where an applicable standard contract exists;
- additive XENO endpoints for capabilities the standard API does not express;
- deterministic execution identities, observability, security, and benchmarks.

### Consumer applications own

- canvases, layers, masks, timelines, tracks, and project state;
- creative workflows, undo/redo, presets, and product-specific defaults;
- asset libraries, user interaction, and result presentation;
- orchestration of multiple inference calls and non-AI processing steps;
- deciding which declared model capability to invoke.

For example, XENO Edit or XENO Pixel may extend a canvas and construct an
outpainting mask. `xrt-image` executes a compatible image model with the image,
mask, and prompt. The runtime owns the inference contract; the application owns
the editing experience.

### `xeno-lib` owns

- video and audio decode/encode;
- deterministic image/audio/video processing;
- screen capture and format conversion; and
- other non-AI media primitives.

AI model execution never moves into `xeno-lib`. Conversely, XENO RT does not
absorb general media editing simply because a consumer uses inference results.

## Shared architecture

```text
XENO applications and agents
        |
        v
xrt-cli / xrt-server / native bindings
        |
        +-- xrt-text   -> current text Runtime and model adapters
        +-- xrt-image  -> ImageRuntime and generative image adapters
        +-- xrt-video  -> future tested video runtime/adapters
        +-- xrt-audio  -> future tested audio runtime/adapters
        +-- xrt-vision -> task-oriented image inference
        |
        v
shared formats, bundles, tensors, kernels, scheduler, resource manager,
CPU/CUDA backends, cache, telemetry, and security
```

The server may host multiple domains simultaneously. Runtimes on the same
device must use the same resource manager so one modality cannot silently
overcommit memory or evict another modality's state without policy.

## API and hosting contract

XENO RT exposes a stable server rather than requiring applications to link a
model implementation directly.

- Existing text endpoints such as `/v1/chat/completions`, `/v1/completions`, and
  `/v1/models` remain compatible.
- Image generation and edit use the applicable OpenAI-compatible image
  contracts when enabled and admitted.
- `/v1/runtime/models` exposes richer XENO capability, backend, quantization,
  and lifecycle state without changing the standard `/v1/models` object.
- Load, unload, queueing, cancellation, progress, and future video/audio
  operations use additive XENO contracts when no suitable standard exists.
- No modality may pretend an unsupported option executed successfully. It
  returns a stable unsupported-capability error.

## Support and admission

File parsing or a successful smoke test does not make a model supported. A
model/bundle/backend/quantization tuple is advertised only after it passes:

1. immutable provenance, format, tensor, and license validation;
2. end-to-end reference correctness at the advertised workload;
3. same-backend determinism or a documented reproducibility contract;
4. model-output quality gates appropriate to the modality;
5. bounded CPU RAM and accelerator-memory admission;
6. controlled latency/throughput benchmarks with no accepted regression;
7. CPU fallback for every capability the product advertises as CPU-capable;
8. API, cancellation, concurrency, security, and cleanup tests; and
9. clean-checkout CI, packaging, installation, rollback, and documentation.

Metrics are modality-specific. Text uses tokens per second and time to first
token; image uses seconds per image, denoising steps per second, and time to
first preview; video and audio will define frame-, sample-, duration-, and
streaming-aware gates before their first adapters are admitted.

## Crate policy

- Do not rename or move the working text implementation merely to create an
  `xrt-text` directory.
- Do not create empty `xrt-video` or `xrt-audio` crates.
- Add a domain crate or facade only when it owns a real runtime or stable public
  facade with tests.
- Share format-neutral infrastructure only after a concrete second consumer
  proves the abstraction. Do not force image, video, or audio pipelines through
  causal-language-model session APIs.
- Model-specific code remains behind a domain adapter; shared crates stay
  modality-neutral.

## Documentation policy

Every new technical plan or specification declares one scope near its title:

- `shared runtime`;
- `xrt-text`;
- `xrt-image`;
- `xrt-video`;
- `xrt-audio`; or
- `xrt-vision` task inference.

Domain-specific plans do not redefine XENO RT as a whole. Historical handoffs
retain their original facts but carry a note identifying their domain. Product
UI and workflow specifications belong in the consuming application repository,
not here.
