# ONNX Integration Plan

**Date:** 2026-03-20
**Status:** Audit and proposal
**Runtime domain:** Shared task-model infrastructure, currently consumed primarily by `xrt-vision` and task routes
**Canonical architecture:** [RUNTIME_DOMAINS.md](RUNTIME_DOMAINS.md)
**Scope:** Add task-specific ONNX inference to XENO RT while preserving GGUF and every existing OpenAI-compatible text contract.

This plan predates the native `xrt-image` implementation. Its historical
current-state sections remain useful evidence, but it does not define the whole
product or route generative image/video/audio models through ONNX by default.
Generative adapters follow their public runtime domain; reusable bundle,
resource-manager, and task-model ideas may be shared where proven.

## Goals and Guardrails

- Keep `GET /v1/models`, `POST /v1/completions`, and `POST /v1/chat/completions` behavior compatible with the current OpenAI-style surface.
- Keep GGUF as the LLM model format. ONNX is an addition, not a replacement.
- Keep CPU fallback available for every ONNX task.
- Benchmark every change. ONNX integration must not regress existing GGUF load time, token throughput, or latency.

## 1. Current xeno-rt Architecture

### 1.1 Workspace Layout

| Crate | Current role |
|---|---|
| `xrt-server` | Axum HTTP server exposing `/v1/models`, `/v1/completions`, and `/v1/chat/completions` |
| `xrt-runtime` | High-level `Runtime` and per-request `Session` for autoregressive generation |
| `xrt-models` | `LlamaModel` and GGUF-based `VisionEncoder` |
| `xrt-gguf` | GGUF parser and memory-mapped tensor access |
| `xrt-tokenizer` | GGUF tokenizer loading plus chat-template rendering |
| `xrt-kernels` | CPU hot-path kernels and thread-pool implementation |
| `xrt-cuda` | CUDA/PTX kernel crate exists, but is not wired into the active runtime/server execution path |
| `xrt-hub` | GGUF download, cache, and Hugging Face resolution |
| `xrt-cli`, `xtask` | CLI entry points for generation and GGUF cache management |
| `xrt-capi`, `xrt-python` | FFI and Python bindings for the current LLM runtime |

### 1.2 How LLM Inference Works Today

1. `xrt-server` starts with one loaded model. `AppState` holds a single `Arc<Runtime>`.
2. The model is either opened from a local GGUF path or resolved through `xrt-hub`.
3. `xrt-hub` is GGUF-specific today. It validates `.gguf`, downloads with a `.part` file, and caches under `~/.cache/xrt/models`.
4. `Runtime::load()` opens the GGUF with `xrt-gguf`, creates a `Tokenizer`, and builds a `LlamaModel`.
5. `xrt-gguf` memory-maps the file and exposes tensor metadata and raw slices. Weights are not eagerly reserialized into a second model format.
6. `xrt-models::LlamaModel` resolves tensor names once, then reads quantized weights directly from the mapped GGUF. The hot path is CPU-first and uses `xrt-kernels`.
7. `xrt-runtime::Session` owns the `PagedKvCache`, `Sampler`, and token history for one generation session.
8. Chat requests are formatted with the tokenizer's chat template when available; otherwise the server falls back to a simple role-prefixed prompt.
9. `Session::generate_stream()` tokenizes the prompt, performs batch prefill with `forward_batch`, then runs token-by-token decode using the KV cache and sampler.
10. Streaming responses are sent with SSE and terminated by `[DONE]`.

### 1.3 Important Current Capabilities

- GGUF LLM loading is already modular and production-oriented.
- The runtime supports more than plain LLaMA: the current code handles `llama`, `qwen3`, and `qwen35`, including hybrid attention/recurrent behavior.
- A GGUF-based `VisionEncoder` already exists for multimodal mmproj-style models, but it is still part of the GGUF path, not an ONNX path.
- The tokenizer/chat-template stack is already where OpenAI-compatible chat formatting happens today.

### 1.4 Current Gaps Relative to ONNX Inference

- The server only knows about one loaded runtime. There is no task router, model registry, or multi-model scheduler.
- `xrt-runtime` is built around autoregressive generation, not generic tensor inference.
- `xrt-hub` only understands GGUF artifacts and one-file model resolution.
- `xrt-cli`, `xtask`, `xrt-capi`, and `xrt-python` only expose the LLM path.
- `xrt-cuda` exists, but there is no shared GPU device manager or VRAM policy in the active runtime. From an ONNX-integration perspective, GPU coordination does not exist yet.

## 2. ONNX Models to Add From xeno-lib

### 2.1 Source-of-Truth Note

`xeno-lib` currently has three overlapping inventories:

- `README.md` advertises **17 task-specific AI capabilities**.
- `docs/AI_MIGRATION.md` describes **15 concrete task modules** plus **4 generative stubs** that are out of scope for this document.
- `models/manifest.json` contains downloadable artifacts for most, but not all, of the task families.

For this plan, the migration contract is the **17-task surface promised by xeno-lib**, with each task marked by current implementation status.

### 2.2 The 17-Task Migration Contract

| # | Task family | Current xeno-lib backing | Representative ONNX artifacts | Migration status |
|---|---|---|---|---|
| 1 | Upscale | `src/upscale/` | `realesrgan_x2.onnx`, `realesrgan_x4plus.onnx`, `realesrgan_x4plus_anime.onnx`, `realesrgan_x8.onnx` | Concrete |
| 2 | Background removal | `src/background/` | `birefnet-general.onnx` | Concrete |
| 3 | Inpainting | `src/inpaint/` | `lama.onnx` | Concrete |
| 4 | Face restoration | `src/face_restore/` | `gfpgan.onnx`, `codeformer.onnx`, `restoreformer.onnx` | Concrete |
| 5 | Depth estimation | `src/depth/` | `depth_anything.onnx`, `midas_v31_large.onnx`, `midas_v31_small.onnx` | Concrete |
| 6 | OCR | `src/ocr/` | `paddle_det.onnx`, `paddle_rec.onnx` | Concrete |
| 7 | Pose estimation | `src/pose/` | `movenet_lightning.onnx`, `movenet_thunder.onnx`, `movenet_multipose.onnx` | Concrete |
| 8 | Transcription | `src/transcribe/` | `whisper-tiny.onnx`, `whisper-base.onnx`, `whisper-small.onnx`, `whisper-medium.onnx`, `whisper-large.onnx` | Concrete |
| 9 | Stem separation | `src/audio_separate/` | `demucs_hybrid.onnx`, `demucs_mdx.onnx`, `uvr_mdx.onnx` | Concrete |
| 10 | Noise reduction / denoise | `src/noise_reduce/` | `rnnoise.onnx`, `dtln.onnx`, `deepfilternet.onnx` | Concrete module, mixed artifact maturity |
| 11 | Style transfer | `src/style_transfer/` | `style_mosaic.onnx`, `style_candy.onnx`, `style_rain_princess.onnx`, `style_udnie.onnx`, `style_pointillism.onnx`, `style_starry_night.onnx`, `style_kandinsky.onnx` | Concrete |
| 12 | Segmentation | Publicly promised in `README.md` as SAM2-style interactive segmentation | No canonical `src/` module or manifest entry yet | Planned, model choice still needed |
| 13 | Frame interpolation | `src/frame_interpolate/` | `rife-v4.6.onnx`, `rife-v4-hd.onnx`, `film.onnx` | Concrete |
| 14 | Face detection | `src/face_detect/` | `scrfd_10g.onnx`, `retinaface.onnx`, `yunet.onnx` | Concrete |
| 15 | Face analysis | `src/face_analysis/` | `age_estimation.onnx`, `gender_classification.onnx`, `emotion_recognition.onnx` | Concrete |
| 16 | Colorization | `src/colorize/` | `ddcolor.onnx`, `deoldify.onnx` | Concrete |
| 17 | Color transfer | Publicly promised in `README.md` | No canonical `src/` module or manifest entry yet | Planned, model choice still needed |

### 2.3 Reconciliation Notes

- The public 17-task `xeno-lib` list is the right migration target for `xeno-rt`.
- The current code and manifest already back **15** of those task families with concrete modules or filenames.
- **Segmentation** and **color transfer** are product-level commitments, but they still need canonical artifacts and manifest entries before the migration is fully complete.
- `docs/AI_MIGRATION.md` also names `text_to_3d`, `voice_clone`, `music_gen`, and `video_gen`, but those are a separate generative expansion and should not be mixed into the first ONNX task-inference rollout for `xeno-rt`.

## 3. API Exposure Plan

### 3.1 API Principles

- Do not overload `/v1/chat/completions` or `/v1/completions` with non-LLM tasks.
- Reuse existing OpenAI-compatible endpoints only where the semantic match is clean.
- Add new task endpoints for image/audio/video operations that do not map cleanly to existing OpenAI APIs.

### 3.2 Recommendation

Use a **hybrid surface**:

- Keep existing LLM endpoints unchanged:
  - `GET /v1/models`
  - `POST /v1/completions`
  - `POST /v1/chat/completions`
- Add the OpenAI-compatible transcription endpoint for Whisper:
  - `POST /v1/audio/transcriptions`
- Add a new XENO task namespace for everything else:
  - `GET /v1/tasks/models`
  - `POST /v1/tasks/{task}`

This keeps OpenAI compatibility where it already exists, without forcing every non-LLM task into an OpenAI-shaped request that does not fit the workload.

### 3.3 Suggested Task Routing

| Task | Endpoint | Notes |
|---|---|---|
| Transcription | `POST /v1/audio/transcriptions` | Use OpenAI-compatible request/response structure |
| Upscale | `POST /v1/tasks/upscale` | Image in, image out |
| Background removal | `POST /v1/tasks/background-removal` | Image in, mask or cutout out |
| Inpainting | `POST /v1/tasks/inpaint` | Image + mask in, image out |
| Face restoration | `POST /v1/tasks/face-restore` | Image in, image out |
| Depth estimation | `POST /v1/tasks/depth` | Image in, depth map out |
| OCR | `POST /v1/tasks/ocr` | Image in, structured text blocks out |
| Pose estimation | `POST /v1/tasks/pose` | Image in, keypoints out |
| Face detection | `POST /v1/tasks/face-detect` | Image in, boxes + landmarks out |
| Face analysis | `POST /v1/tasks/face-analysis` | Image or face crop in, structured attributes out |
| Colorization | `POST /v1/tasks/colorize` | Image in, image out |
| Color transfer | `POST /v1/tasks/color-transfer` | Source + reference image in, image out |
| Style transfer | `POST /v1/tasks/style-transfer` | Image in, image out |
| Segmentation | `POST /v1/tasks/segment` | Image + optional clicks/prompts in, mask out |
| Stem separation | `POST /v1/tasks/stem-separation` | Audio in, multiple stems out |
| Noise reduction | `POST /v1/tasks/noise-reduction` | Audio in, cleaned audio out |
| Frame interpolation | `POST /v1/tasks/frame-interpolation` | Two frames in, interpolated frame out |

### 3.4 Request/Response Shape

Recommended common shape for `POST /v1/tasks/{task}`:

```json
{
  "model": "realesrgan-x4",
  "input": {
    "image_base64": "..."
  },
  "parameters": {
    "scale": 4
  },
  "output_format": "png",
  "stream": false
}
```

Recommended response envelope:

```json
{
  "id": "task_123",
  "object": "xrt.task.result",
  "created": 1774041600,
  "task": "upscale",
  "model": "realesrgan-x4",
  "data": {
    "b64_json": "..."
  }
}
```

Notes:

- Image, mask, and depth outputs should use base64 payloads in JSON, similar to other OpenAI media responses.
- Structured outputs such as OCR, pose, face detection, and face analysis should return typed JSON in `data`.
- Long-running tasks can follow the existing `stream: true` pattern and use SSE, but the default should be synchronous JSON.

### 3.5 Discovery

Do not overload the existing `/v1/models` response with task-specific schema details.

Instead:

- Keep `/v1/models` safe for OpenAI-style model listing.
- Add `GET /v1/tasks/models` for richer task metadata:
  - task name
  - model ids
  - input media types
  - required parameters
  - optional parameters
  - estimated RAM/VRAM footprint

## 4. Model Download and Caching Strategy

### 4.1 Extend xrt-hub Into a Multi-Artifact Registry

`xrt-hub` should become the shared model-distribution layer for both GGUF and ONNX.

This is more than removing a `.gguf` filename check. The hub needs to understand:

- artifact kind: `gguf`, `onnx`, and sidecar files
- model family and version
- one-file and multi-file bundles
- preferred download sources
- checksum verification
- local cache reuse

### 4.2 Registry Format

Start from the existing `xeno-lib/models/manifest.json` source, but evolve it into a richer `xrt-hub` registry with these fields per model:

- `id`
- `task`
- `family`
- `version`
- `artifacts[]`
- `size_bytes`
- `sha256`
- `estimated_ram_mb`
- `estimated_vram_mb`
- `execution_providers`
- `default_parameters`
- `input_schema`
- `output_schema`

`artifacts[]` is required because several tasks are not single-file models:

- OCR uses detection and recognition models
- face analysis is a three-model bundle
- style transfer is a family of per-style models
- future segmentation models may require encoder/decoder bundles

### 4.3 Cache Layout

Recommended layout:

```text
~/.cache/xrt/models/
  gguf/
    qwen/
      qwen3-0.6b/
        qwen3-0.6b-q4_k_m.gguf
  onnx/
    upscale/
      realesrgan-x4/
        v1/
          realesrgan_x4plus.onnx
    ocr/
      paddleocr/
        v1/
          paddle_det.onnx
          paddle_rec.onnx
  manifests/
    tasks.json
```

Key points:

- Keep GGUF and ONNX in separate top-level cache namespaces.
- Version directories are important so upgraded artifacts do not silently replace older ones.
- Bundle-oriented layout is required for OCR, face analysis, and similar multi-file tasks.

### 4.4 Download Behavior

Recommended flow:

1. Resolve the requested model id in the manifest.
2. Check the `xrt` cache.
3. If missing, check the legacy `~/.xeno-lib/models` cache and import or hard-link when possible.
4. If still missing, download to a temporary `.part` file.
5. Verify file size and SHA-256.
6. Atomically rename into place.
7. Use a per-artifact lock so two requests do not download the same file twice.

### 4.5 Hosting and Offline Behavior

- Keep the current manifest host pattern from `xeno-lib`: `https://updates.xenostudio.ai/models/...`.
- Retain Hugging Face fallback support where appropriate.
- Embed a pinned manifest snapshot in `xeno-rt` so the runtime still works offline after installation.
- After a model is cached, no network should be required to use it.

### 4.6 CLI and Admin Surface

Extend the existing cache-management tooling rather than inventing a second system.

- `xrt-cli download` should support both GGUF and ONNX ids.
- `xtask list-cached` and `xtask clean-cache` should become artifact-kind aware.
- Cache inspection should show model id, task, version, size, and on-disk path.

## 5. GPU Memory Sharing Between LLMs and ONNX Models

### 5.1 Current Reality

Today there is no active shared GPU-memory story in `xeno-rt`:

- `xrt-cuda` exists as a crate, but the current runtime/server path is still centered on CPU GGUF inference.
- There is no device manager in `xrt-runtime`.
- There is no admission control or shared VRAM accounting between workloads.

That means ONNX integration cannot rely on an existing GPU scheduler. It has to introduce one.

### 5.2 Recommended Ownership Model

Add one shared `GpuResourceManager` per CUDA device at the runtime/server layer.

It should own:

- device discovery and selection
- total/free VRAM accounting
- reserved headroom policy
- stream pool or execution queue policy
- per-model VRAM estimates
- admission control
- eviction decisions for cached ONNX sessions

The key rule is that future GPU-backed GGUF inference and ONNX inference must both report into the same manager, even if they use different backends internally.

### 5.3 Memory Classes

| Class | Examples | Policy |
|---|---|---|
| Reserved | OS/display headroom, safety margin | Never allocated by workloads |
| High-priority persistent | Future GPU-resident GGUF weights, active KV cache | Pin while active |
| Reclaimable resident | Idle ONNX sessions cached on GPU | LRU-evictable |
| Transient workspace | Input tensors, scratch buffers, output staging | Reserve per request, free immediately |

### 5.4 Scheduling Policy

Recommended policy:

1. Reserve a fixed GPU safety margin first.
2. Reserve LLM budget before ONNX budget.
3. Admit an ONNX task to GPU only if it fits within the remaining budget.
4. If it does not fit, evict idle ONNX sessions from the GPU cache.
5. If it still does not fit, fall back to CPU execution instead of disturbing the active LLM workload.

This preserves the current product hierarchy: LLM serving is the always-on workload, and task inference must not destabilize it.

### 5.5 Practical First Phase

Because the current LLM path is still CPU-centric, the first ONNX implementation can start with:

- shared device discovery
- ONNX-side GPU session caching
- estimated VRAM accounting
- CPU fallback when over budget

But the abstraction must be placed where a future GPU-backed GGUF runtime can plug into the same manager. Do not build an ONNX-only GPU island that has to be replaced later.

### 5.6 Session Caching Strategy

Recommended ONNX residency states:

- **Cold:** only on disk
- **Warm:** loaded in CPU/host memory or CPU execution provider
- **Hot:** loaded with CUDA execution provider and counted against the shared GPU budget

For most workloads:

- keep only a few small, frequently used ONNX sessions hot on GPU
- allow fast CPU fallback for less common tasks
- prefer evicting ONNX sessions over touching an active LLM reservation

## 6. Recommended Delivery Order

### 6.1 Phase 1: Foundation

- Add an ONNX-focused crate, for example `xrt-onnx`, that wraps ONNX Runtime and normalizes execution-provider handling.
- Extend `xrt-hub` to understand ONNX artifacts and model bundles.
- Add task-model registry support.
- Introduce the shared GPU resource manager abstraction.
- Add task endpoints to `xrt-server` without touching the existing LLM endpoints.

### 6.2 Phase 2: Migrate the 15 Concrete Task Families

Migrate the families that already have concrete `xeno-lib` source and/or manifest artifacts:

- upscale
- background removal
- inpainting
- face restoration
- depth estimation
- OCR
- pose estimation
- transcription
- stem separation
- noise reduction
- style transfer
- frame interpolation
- face detection
- face analysis
- colorization

### 6.3 Phase 3: Close the Two Contract Gaps

Before calling the migration complete, add canonical artifacts and manifest entries for:

- segmentation
- color transfer

Those two are already part of the public `xeno-lib` promise and should have stable task ids from day one, even if they are implemented last.

## 7. Benchmark and Validation Requirements

Every integration step should be gated by benchmarks and parity checks.

Minimum benchmark set:

- GGUF cold load time before and after ONNX integration
- GGUF prompt prefill throughput before and after ONNX integration
- GGUF decode latency before and after ONNX integration
- ONNX cold model load time per task family
- ONNX warm inference latency on CPU and CUDA
- mixed-workload test: active chat generation plus ONNX task inference
- RAM and VRAM usage under mixed workload

Minimum correctness checks:

- output parity against current `xeno-lib` task results
- CPU fallback for every ONNX task
- no behavior changes in existing OpenAI-compatible LLM endpoints

## 8. Bottom Line

At the original 2026-03 audit, XENO RT had strong model-loading, runtime,
streaming, and cache foundations but was still organized around one GGUF text
runtime. That audit established the following additive task-model direction:

- a second model family under the same runtime umbrella
- a richer `xrt-hub` model registry and cache
- a task-specific API namespace that leaves OpenAI LLM compatibility untouched
- a shared GPU resource manager that gives LLM workloads first claim on VRAM

Those foundations now coexist with native `xrt-image` work. They remain the
task-inference direction without weakening GGUF or redefining the generative
runtime domains.
