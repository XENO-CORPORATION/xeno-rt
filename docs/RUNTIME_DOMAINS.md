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
| `xrt-audio` | Speech, music, and audio model inference | Crate exists and owns the tested **signal frontend** (STFT, Slaney mel filterbank, Whisper log-mel, downmix, resample). **No model adapter, no endpoint, not admitted** — see below. |

These are public capability boundaries, not four unrelated inference engines.
They share XENO RT's formats, tensor types, kernels, device management, bundle
cache, scheduler, telemetry, security rules, and server.

`xrt-vision` remains an auxiliary task-inference domain for discriminative or
deterministic image operations such as segmentation, background removal, depth,
OCR, and upscaling. A model belongs to a domain according to its advertised
product capability and primary output, not every internal input type. For
example, Qwen Image Edit consumes images and text internally but belongs to
`xrt-image` because it produces a generated image.

### `xrt-audio` status — measured 2026-08-27

The crate was created under the "real public facade with tests" clause of the
crate policy below, not as a placeholder. It is deliberately **not** advertised:
no endpoint is registered, no capability is declared, and nothing in the server
references it.

| | |
|---|---|
| **Implemented and tested** | `stft` (Hann window, reflect padding, centred STFT), `mel` (Slaney filterbank, Whisper log-mel, `pad_or_trim`), `to_mono`, `resample_linear`. 24 tests, all four mutation-checked. |
| **Whisper adapter (Rust)** | ✅ `whisper.rs` — `ort` encoder/decoder sessions, KV-cache greedy decode, `xrt-tokenizer` detokenisation, and long-form 30-second windowing. Model dimensions are read from `config.json` and special token ids resolved from the vocabulary BY NAME, so base/small/medium load without a table to maintain. |
| **Verified against real weights** | ✅ Transcribes the standard JFK sample to the exact reference text, from Rust, using artifacts **downloaded back from the CDN**: 11 s in **647 ms** (~17× realtime, CPU). Long-form proven separately — a 44 s file yields two windows, the second correctly bounded at 44.0 s rather than the padded 60. Gate: `tests/whisper_e2e.rs`, **mutation-checked 2/2** (never signalling the cache branch, and never carrying the decoder cache forward, both fail it). |
| **Not implemented** | Demucs separation. Whisper **timestamp tokens** (segments are per-window, not per-phrase), **language detection** (English is assumed, and `Transcript::language` reports `None` rather than a guess), and every `/v1/audio/*` route. |

⚠️ **Read the verification row precisely: it says the adapter works, not that the
product does.** There is still no HTTP route and no capability, so nothing
outside this crate can reach it — deliberately, per `CLAUDE.md` rule 6, until
this domain defines its admission gates.

⚠️ **The chunking is the SIMPLE strategy and its seam is visible.** Fixed 30-second
cuts can land mid-phrase; OpenAI's reference implementation uses the model's own
timestamp tokens to end a window on a phrase boundary. Timestamp-guided
windowing and VAD are refinements of the same loop, not replacements for it.

*Re-derive:* `cargo test -p xrt-audio`, and
`grep -rn "xrt-audio" crates/xrt-server/` (expects no match while unadmitted).

**Two blockers stand between this and an admitted adapter, and only one is code.**

1. **Task-model weights — mostly still unpublished. Whisper now is.**

   *Measured 2026-08-27, earlier the same day:* `models/manifest.json` had
   advertised ~33 ONNX models since March 2026 and **not one had ever been
   uploaded** — the bucket held only `manifest.json`,
   `local-model-catalog.json` and seven GGUF language models, with **zero
   `.onnx` objects**. Every task model 404'd, including
   `realesrgan_x4plus.onnx`, which `xrt-vision`'s upscaler has working code
   for.

   *Corrected 2026-08-27, later the same day:* **whisper-base is published and
   verified.** `whisper-base-encoder.onnx`, `whisper-base-decoder.onnx` and
   `whisper-base-tokenizer.json` resolve with real sizes, real `sha256`, and
   Apache-2.0 provenance (`openai/whisper-base`), via
   `xeno-platform/scripts/publish-onnx-models.mjs`. The other 33 entries are
   untouched and **still 404, still with an empty `sha256`**.

   ⚠️ The pre-existing `whisper-base` manifest entry names a single
   `whisper-base.onnx` of 74,000,000 bytes. That is architecturally impossible
   — ONNX Whisper is **two** graphs plus a tokenizer — and 74,000,000 is a
   round number like every other size in that file. Those entries were written
   without reference to any real export, which is why the new ones carry their
   own names rather than repairing that one in a publish.

   *Re-derive:* `curl -sI https://updates.xenostudio.ai/models/whisper-base-encoder.onnx`
   (expect 200) and `rclone ls r2:xeno-hub-releases/models`.

2. **The admission metrics for this domain are undefined.** "Support and
   admission" below requires audio to define sample-, duration- and
   streaming-aware gates *before* its first adapter is admitted. That definition
   does not exist yet and is a prerequisite, not paperwork to follow the code.

**What was deliberately NOT reused.** `xeno-lib` carries `transcribe/` and
`audio_separate/` under `src/ai_deprecated/`. They are not a starting point: its
`audio_to_mel` contains no FFT (per-frame RMS multiplied by a fixed sine curve,
so any two equal-power signals produce identical output) and its `decode_tokens`
has no vocabulary, emitting `"[50364]"` token ids as transcript text. Both were
covered by tests asserting tensor *shape*, so both passed. The frontend here is
mutation-checked against exactly that defect: reinstating the fabricated mel
fails `mel::tests::tone_frequency_selects_the_mel_band` and nothing else.

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
