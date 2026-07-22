# Native Image Inference and Qwen Image Support

**Status:** Implemented as an experimental integration candidate; not production-admitted or released

**Owner:** XENO RT team

**Created:** 2026-07-21

**Last updated:** 2026-07-22

**Target repository:** `xeno-rt`

**Primary deliverable:** `xrt-image` with Qwen-Image-2512 generation, followed by Qwen-Image-Edit-2511 editing

**Runtime domain:** `xrt-image`

**Related documents:** `docs/RUNTIME_DOMAINS.md`, `docs/SESSION-STATE-2026-07-17.md`, `docs/GPU_RUNTIME_ACCELERATION_SPEC.md`, `docs/ONNX_INTEGRATION_PLAN.md`

## 1. Executive summary

XENO RT should expose four long-term generative capability domains:

- `xrt-text` for language and multimodal conversational generation;
- `xrt-image` for native image generation and generative image editing;
- `xrt-video` for future video generation and transformation; and
- `xrt-audio` for future speech, music, and generative audio.

These are public capability boundaries, not four independent inference engines. They must share XENO RT's tensor types, model formats, CPU/CUDA kernels, model cache, resource accounting, scheduling policy, telemetry, and OpenAI-compatible server. The existing `xrt-vision` crate remains separate: it owns discriminative and task-oriented media operations such as background removal, segmentation, depth, OCR, and upscaling. A model that consumes an image internally may still live in `xrt-image` when its product capability is generative image output.

This specification creates only the first useful new domain, `xrt-image`. It does not rename the working text implementation or create empty `xrt-video` and `xrt-audio` crates. The existing `xrt-runtime`, `xrt-models`, and server paths continue to provide text inference while a later, separately gated refactor introduces an `xrt-text` facade without changing behavior.

The first supported model is Qwen-Image-2512 for text-to-image generation. The second is Qwen-Image-Edit-2511 for image editing. Both have public Apache-2.0 weights and official Diffusers reference pipelines. Qwen-Image-3.0 was announced on 2026-07-21, but a 2026-07-22 recheck found no official public checkpoint, configuration, or local inference implementation in the official catalog. XENO RT will prepare an adapter boundary and recheck that status at implementation kickoff, but it will not claim or fabricate local 3.0 support before public artifacts exist.

The target admission matrix is BF16 SafeTensors plus complete, validated Q8_0, Q6_K, Q5_K_M, and Q4_K_M bundles. It is not the current supported matrix: each bundle/backend pair becomes supported only after its execution, quality, memory, and performance gates pass. Q3 and Q2 remain experimental until image-quality gates pass. A GGUF diffusion transformer file alone is not a runnable model: a supported bundle must also identify the text encoder, tokenizer/processor, VAE, scheduler configuration, and editing projection components where required.

The product target is a pure-Rust, headless image runtime with CPU fallback, optional CUDA, GGUF and SafeTensors components, deterministic same-backend execution, explicit memory planning, and OpenAI-compatible image generation and edit endpoints. Python, PyTorch, ComfyUI, stable-diffusion.cpp, and Diffusers are reference or test oracles only; none is a production runtime dependency.

## 2. Decision record

### 2.1 Model priority

| Target | Public local weights on 2026-07-22 | License position | Initial 24 GiB target | Decision |
|---|---:|---|---:|---|
| Qwen-Image-2512 | Yes | Apache-2.0 | Q4_K_M with staged component placement | Implement first |
| Qwen-Image-Edit-2511 | Yes | Apache-2.0 | Q4_K_M with staged component placement | Implement second |
| Original Qwen-Image and Qwen-Image-Edit | Yes | Apache-2.0 | Compatibility fixtures only | Support where mappings are identical; not the quality target |
| Qwen-Image-3.0 | No official checkpoint found during this audit | Unknown until weights are published | Unknown | Track and prepare an adapter gate |
| Ideogram 4 | Gated public weights exist | Public quantized weights are restricted; customer-facing use needs separate terms | Official NF4 targets 24 GiB | Defer to adapter two after legal approval |

### 2.2 Product boundary

XENO RT is the inference engine. It should compete with local inference engines on correctness, portability, memory use, and speed. It is not a ComfyUI clone in this milestone. Node graphs, canvas workflows, plugin ecosystems, and XENO Pixel integration are outside scope. Later frontends may call the same HTTP and Rust APIs.

### 2.3 Runtime boundary

`xrt-image` gets an image-specific `ImageRuntime` and pipeline traits. It must not be forced through `CausalLmBackend`, `BackendSession`, token sampling, or KV-cache interfaces. The server owns text and image runtimes side by side and, after the additive injection refactor specified below, passes runtimes on the same CUDA device the same `Arc<GpuResourceManager>`. This avoids a dependency cycle and limits regression risk. A fully modality-neutral runtime registry can be extracted later after two real modality runtimes establish the common contract.

### 2.4 Format boundary

GGUF remains required and first-class. SafeTensors remains the full-precision reference path. The runtime does not introduce another weight serialization format. It adds a small, checksummed bundle manifest that describes how existing component files form one executable pipeline.

## 3. Problem statement

The current repository is optimized around autoregressive text inference:

- `xrt-runtime::Runtime` owns a causal language-model backend and tokenizer;
- `CausalLmBackend` and `BackendSession` assume prefill, token decode, KV state, and logits;
- `xrt-hub` resolves and downloads one GGUF file at a time;
- the SafeTensors loader requires causal-LM Hugging Face configuration fields;
- `xrt-server::AppState` holds one active text runtime; and
- the only image route, `/v1/images/remove-background`, delegates a bounded ONNX task to `xrt-vision`.

Native diffusion or flow-matching image generation needs a different graph and lifecycle. It loads multiple components, encodes a prompt once, iteratively denoises a latent tensor, decodes that latent through a VAE, and may encode one or more source images for editing. The dominant allocation, scheduling, and performance metrics differ from token generation.

Adding Qwen Image directly to the text `Runtime` would couple unrelated state machines, make cancellation and memory accounting ambiguous, and turn `CausalLmBackend` into an unsafe generic abstraction. Adding a standalone Python service would violate the pure-Rust, offline, CPU-fallback, and unified-resource goals. Adding only a transformer GGUF loader would appear to support a model while leaving the majority of the pipeline external.

XENO RT therefore needs a native image domain with a complete component-bundle contract, reusable denoising primitives, its own request scheduler, and controlled integration with the existing server and GPU resource manager.

## 4. Goals

### 4.1 Functional goals

1. Generate images end to end from Qwen-Image-2512 without Python or an external inference process.
2. Perform semantic and multi-image edits end to end with Qwen-Image-Edit-2511; add masked inpainting only after the separate Qwen Image inpaint reference path passes its own compatibility gate.
3. Load official BF16 SafeTensors component trees and validated mixed-format bundles containing GGUF components.
4. Support Q8_0, Q6_K, Q5_K_M, and Q4_K_M as named, bundle-level compatibility tiers.
5. Run every admitted model bundle on CPU-only builds when sufficient host RAM is available.
6. Run admitted bundles on CUDA with explicit full-residency or component-offload plans.
7. Preserve deterministic output for a fixed model bundle, backend, build, seed, dimensions, scheduler, and parameters.
8. Expose OpenAI-compatible image generation and edit endpoints.
9. Provide a Rust API and CLI that do not depend on the HTTP server.
10. Report load, prompt encoding, denoising, VAE, transfer, queue, RAM, and VRAM measurements.
11. Keep the component and adapter boundary capable of accepting Qwen-Image-3.0 after official artifacts are published and audited.

### 4.2 Compatibility goals

1. Preserve current `/v1/models`, `/v1/completions`, and `/v1/chat/completions` response schemas.
2. Preserve all existing GGUF text paths and `xrt-vision` task behavior.
3. Preserve current `xrt generate`, `xrt chat`, `xrt bench`, and `xrt download` commands.
4. Keep default builds functional without the CUDA Toolkit or CUDA driver.
5. Keep explicit backend semantics: `cpu` is CPU-only, `cuda` either uses a supported CUDA/offload plan or fails clearly, and `auto` may select CPU after admission planning.
6. Keep cached models usable offline and never require runtime network access after bundle installation.

### 4.3 Performance goals

1. Fit the validated Qwen-Image-2512 Q4_K_M bundle on an RTX 4090 without exhausting the entire 24 GiB device.
2. Avoid host/device weight transfers inside a denoising step loop.
3. Reuse stable latent, attention, and projection scratch allocations across steps.
4. Support VAE slicing and tiling so decode peak memory is bounded independently of full output size.
5. Reach competitive warm-generation latency against a pinned stable-diffusion.cpp build on the same model, settings, and hardware before CUDA support is declared production-ready.
6. Keep existing text CPU and CUDA benchmarks within their no-regression gates.

## 5. Non-goals

1. Implementing Qwen-Image-3.0 before official weights, configuration, license, and reference inference are public.
2. Promising support for every community quantization filename or tensor mapping.
3. Declaring Q2 or Q3 production-ready without measured quality evidence.
4. Implementing Ideogram 4 before a licensing decision permits the intended XENO distribution and customer-facing use.
5. Building a ComfyUI-compatible graph editor, plugin system, or desktop UI.
6. Integrating XENO Pixel in this milestone.
7. Implementing video or audio generation now.
8. Renaming `xrt-runtime` or moving the current LLM implementation merely to create an `xrt-text` directory.
9. Running Diffusers, PyTorch, Python, stable-diffusion.cpp, or ComfyUI in the production request path.
10. Replacing GGUF, removing CPU inference, or silently falling back from an explicit backend.
11. Training, fine-tuning, LoRA training, or model conversion in the server.
12. Claiming cross-backend bit-for-bit equality when CPU and CUDA arithmetic legitimately differ.
13. Cutting or publishing a release. Release work remains governed by `release-guide/`.

## 6. Definitions and modality taxonomy

### 6.1 Public capability domains

| Domain | Owns | Does not own in this milestone |
|---|---|---|
| `xrt-text` | Future stable facade for text generation, embeddings, and conversational multimodal models | No immediate crate move; current code stays in `xrt-runtime` and `xrt-models` |
| `xrt-image` | Text-to-image, image-to-image, inpainting through a generative pipeline, and multi-image generative editing | OCR, segmentation, background removal, deterministic filters |
| `xrt-video` | Future video generation, extension, and generative transformation | Empty placeholder code today |
| `xrt-audio` | Future TTS, voice generation, music generation, and generative audio transformation | Existing ONNX transcription or denoise tasks unless later promoted behind a shared facade |
| `xrt-vision` | Perception and task inference: background removal, segmentation, depth, OCR, upscale, restoration, and similar ONNX tasks | Diffusion/flow image generation |

The domain is selected by the user-visible capability, not every internal input type. Qwen-Image-Edit uses a vision-language encoder internally but remains an `xrt-image` adapter because it returns a generated image.

The word "inpaint" exists in both domains with different contracts. Existing LaMa-style deterministic/task inpainting remains an `xrt-vision` capability. Prompt-conditioned flow/diffusion inpainting belongs to `xrt-image` and advertises `image.inpaint`; callers and manifests must use the capability ID rather than route by the word alone.

### 6.2 Terms

- **Image bundle:** A versioned manifest plus every component needed to execute one model capability.
- **Component:** A transformer/DiT, text encoder, tokenizer or processor, VAE, scheduler configuration, vision projection, or other independently stored part of a bundle.
- **Adapter:** Model-family-specific configuration, tensor-name mapping, shape validation, and graph assembly in `xrt-image`.
- **Reference path:** The official BF16 pipeline used to establish numerical fixtures; it is not necessarily the deployment path.
- **Admission:** Validation that component formats, dimensions, capabilities, memory, backend, and configured limits permit a request before mutable execution begins.
- **Placement plan:** An immutable per-request/component decision describing CPU residency, CUDA residency, staged upload, or eviction.
- **Denoising step:** One scheduler/transformer update of the latent state. It is not an LLM token.
- **Same-backend determinism:** Repeated execution on the same supported backend, device architecture, driver/compute-runtime versions, and build produces identical normalized, uncompressed pixel bytes for a fixed request and bundle. That pixel hash is the normative gate. Encoded-file bytes are additionally required only for a pinned deterministic encoder profile; lossy codec bytes are not the cross-version identity.
- **Compatibility fixture:** An older or community model used to detect mapping regressions but not advertised as the primary quality tier.

## 7. Audited repository baseline and implementation checkpoint

This audit used local branch `feat/v0.2.0-vision-tasks-remove-background` at commit `e1bb2e67fa4a2cf6ac399a8bbaee34e9d20de2e2` plus its existing local working-tree changes. Live GitHub `origin/main` resolved to `9bdc8d17dd618a38513df1176bff1eb8be52792a` on 2026-07-22. Implementers must repeat the audit before editing because the local hybrid-MoE work is ahead of public `main` and must not be overwritten.

Sections 7.1 and 7.2 preserve the pre-implementation baseline used to derive the design. They are not a statement that those gaps still exist. Section 7.4 is the authoritative implementation checkpoint for the experimental integration candidate; production-advertised support remains empty until the later admission gates pass.

### 7.1 Pre-implementation reusable foundations

- `xrt-core` provides device and dtype identifiers plus basic tensor views.
- `xrt-gguf` memory-maps GGUF artifacts and exposes tensor metadata without reserialization.
- `xrt-safetensors` validates single-file and sharded SafeTensors bundles and already resolves several quantization metadata schemes.
- `xrt-tokenizer` supports GGUF and Hugging Face tokenizer assets used by current Qwen text models.
- `xrt-kernels` and `xrt-cuda` contain native linear, quantized, normalization, attention, and allocation primitives.
- `xrt-runtime::GpuResourceManager` tracks persistent weights, KV, scratch, budgets, and explicit transfers within one `Runtime`; current text load paths construct a new manager internally rather than accepting a server-owned shared manager.
- `xrt-hub` performs atomic single-file downloads into `~/.cache/xrt/models`.
- `xrt-server` preserves OpenAI response schemas and has tests that reject leaked acceleration metadata.
- `xrt-vision` establishes a separate image-task crate and a CPU fallback pattern for ONNX background removal.

### 7.2 Pre-implementation gaps that required explicit work

1. `xrt-runtime::Runtime` and `CausalLmBackend` are text-specific and cannot represent denoising pipelines cleanly.
2. `Runtime::load_with_backend_configs` ultimately constructs its own `Arc<GpuResourceManager>`. The server therefore cannot yet give text and image runtimes the same per-device manager; an additive injection path is required while existing constructors retain their behavior.
3. `GpuAllocationClass` and status output describe model/KV/scratch concepts but not image components, latents, prompt embeddings, or VAE tiles.
4. `xrt-safetensors::HfModelConfig` requires causal-LM fields. Its shard mapping and tensor validation are reusable, but the config layer must remain intact while a generic component config is composed beside it.
5. `xrt-hub::ModelHub` accepts one `.gguf` target, downloads from `resolve/main`, and validates size rather than SHA-256; Qwen Image needs a revision-pinned, atomic multi-artifact transaction. The hub exposes `with_cache_dir`, but no `XRT_CACHE_DIR` environment setting exists today.
6. `xrt-server::AppState` stores one text runtime and one loaded model identity rather than multiple capability runtimes. Its `external-openai` transition clears the local text runtime and must remain text-scoped when image state is added.
7. HTTP image generation and edit request/response types do not exist. The current bodyless `POST /v1/runtime/unload` has no modality field, while `POST /v1/runtime/load` already accepts legacy text `model_path`, Hugging Face, and external-provider fields.
8. The current remove-background input helper accepts HTTP, local file, and data URLs. The generation/edit implementation must not reuse that permissive fetch boundary.
9. Current CUDA kernels are optimized for token-oriented matrix/vector shapes. Image attention, batched matrix multiplication, convolution, patchification, and VAE operations need separate profiling and kernels.
10. Existing text schedulers, prefix cache, token sampler, and KV cache are not reusable image-session state.
11. No repository benchmark records seconds per image, denoising steps per second, time to first preview, or image-quality regressions.

### 7.3 Existing plan reconciliation

`docs/ONNX_INTEGRATION_PLAN.md` already calls for multi-artifact manifests and one GPU resource manager across LLM and task models. `xrt-image` extends that direction rather than introducing a second registry. `docs/GPU_RUNTIME_ACCELERATION_SPEC.md` defines explicit backend behavior, central allocation telemetry, safe CUDA validation, and no-regression rules. Those rules remain authoritative.

### 7.4 Implementation checkpoint on 2026-07-22

This checkpoint describes uncommitted local work at `HEAD e1bb2e67fa4a2cf6ac399a8bbaee34e9d20de2e2`; it is evidence of progress, not a support announcement.

| Area | Verified local state | Remaining before admission |
|---|---|---|
| Phase 0 provenance | `reference/image/qwen/phase0-lock.json` pins official models, community conversions, reference engines, SDKs, evaluators, and fixture hashes. Ten generation/edit BF16/Q8/Q6/Q5/Q4 development manifests, Diffusers oracles, stable-diffusion.cpp baselines, and the frozen quality suite exist. The reference-only quality tool now emits a deterministic 250-active-case plus 50-identity-pair plan, verifies complete paired artifacts and evaluator identity, and compiles the frozen 10,000-resample PCG64 and one-sided Wilson gates without permitting a production claim. Two plan emissions matched SHA-256 `36d8f3f021474015ebbd9c4b7e4ad87df611bbc6179db80b9cd9a91e1a7d0a8d`; seven focused tool tests pass, including a complete synthetic admission report. The refreshed schema-v2 OpenAI fixture manifest pins server OpenAPI 2.3.0, both edit transports, and named raw generation/edit SSE frames whose typed completed event is terminal without `[DONE]`. Three additional serialized Q4 comparator runs were pixel/PNG deterministic with a 258 MiB sampled peak-delta range, but their summary correctly marks the quiet-baseline gate failed because interactive GPU workloads remained resident. | Capture the quiet, repeatable non-XENO VRAM baseline after interactive GPU workloads are closed. No real BF16/candidate evaluator export or human-review set has been run through the new compiler, so every quantized quality gate remains open. Later 30-run native comparisons remain a separate admission gate. |
| Foundation and bundles | `xrt-image`, reusable `SafeTensorStore`, injected/shared `GpuResourceManager`, image allocation classes, `XRT_CACHE_DIR`, atomic hash-verified bundle install/recovery, deterministic RNG/scheduler, bounded codecs, cancellation, and synthetic generation/edit tests are present. The 2026-07-22 audit passed `cargo test --workspace --no-default-features`. | The integration candidate still requires protected-branch CI and merge review. Reviewed manifests currently live in the development reference tree; release packaging of the executable/cache catalog remains to be proven. |
| Qwen-Image-2512 generation | Native Rust BF16, Q8_0, Q6_K, Q5_K_M, and Q4_K_M CPU bounded real-model runs have deterministic checked-in output evidence. The pinned Q4_K_M, Q6_K, and Q5_K_M native CUDA pipelines also completed bounded 16x16, two-step generation with candidate-plus-locked or previously locked exact PNG hashes. Their measured resident/peak arena bytes are Q4 `13,649,426,688 / 13,651,491,072`, Q6 `17,224,349,952 / 17,226,414,336`, and Q5 `15,399,434,496 / 15,401,498,880`. Q4_K_M additionally completed sixteen retained-path native 512x512, four-step repetitions of the pinned comparator workload across the VAE, ordered-BF16, softmax-normalization, shared-activation-tile, spatial VAE scheduling, and feature-major BF16 linear optimizations with the same PNG SHA-256 `16d53f008029550757b257e2b40db234a7b913d26615573b698c4a77d015ade9` and `13,957,019,904`-byte tracked peak. Manual inspection found the expected blue mechanical keyboard on a wood desk, but no automated quality claim follows from that observation. | BF16 CUDA is not admitted on the 24 GiB tier. The 512x512 smoke is non-quiet, four-step development evidence only. Release-resolution component parity, the frozen quality suite, quiet-host memory measurement, and release workloads remain open. |
| Quantized breadth | The CPU and CUDA executors recognize Q8_0, Q6_K, Q5_K_M, and Q4_K_M, and focused packed-kernel parity exists. Complete pinned bundles for all four tiers are locally hash-verified. Q8_0 has bounded deterministic CPU execution; Q6_K and Q5_K_M each have four serial manual CPU tests, two matching CPU CLI outputs, an unpinned CUDA candidate, and a matching locked CUDA repeat. The exact Q8 matrix map is 6 BF16 plus 840 Q8_0; Q6 is 6 BF16 plus 840 Q6_K; Q5 is 6 BF16, 560 Q5_K, and 280 Q6_K. | Every tier still needs its applicable full-resolution quality gate. Q8_0 has no CUDA evidence because its 21,761,817,120-byte transformer artifact alone exceeds the audited 4090 cap after non-XENO use and reserve; supporting it on this device requires an independently designed and measured layer/staged-offload path. None of the bounded smokes constitutes production tier admission. |
| CUDA performance | The retained packed Q4 tile, exact-order CUDA attention, CPU convolution, and ordered-BF16 changes have reduced the matched 512x512/four-step pipeline from a 105.777-second mean to 51.201 seconds without changing its locked PNG hash or `13,957,019,904`-byte peak. The latest retained Q4 kernel cooperatively caches each 16x256 activation tile across sixteen output warps: four transformer forwards averaged 3.614 seconds, and two full pipelines averaged 28.311 seconds of denoising versus 34.379 seconds before shared tiling. Spatial VAE tile scheduling preserved per-pixel arithmetic and reduced VAE decode from 13.618 to 12.782 seconds. Feature-major BF16 linear scheduling reuses each mapped weight row across prompt tokens while calling the same ordered dot routine; it passed bit-exact multi-row/single-row tests and the real Diffusers prompt oracle, reduced two prompt encodes from 20.850 to 9.920 seconds (2.102x), and reduced wall time from 61.999 to 51.201 seconds (1.211x). All sixteen retained-path outputs match exactly. Q8-activation/DP4A regressed and remains opt-in; a 32-warp block gained only 0.69% end to end and was rejected for portability margin; a width-scaled VAE tile regressed decode 3.97% and was rejected. Real Edit attention separately improved 3.690x at the denoiser while preserving its hash/peak, but remains slow. The pinned stable-diffusion.cpp capture took 29.563 seconds, so the current non-quiet native ratio is 1.732x. | These active-workstation timings are not an admission baseline. No quiet 30-run matched comparison or release-resolution memory trace exists, and the native path remains materially slower than the comparator, so IMG-P09 and Section 20.5 fail and CUDA remains experimental. |
| Rust/OpenAI/CLI surfaces | `ImageRuntime`; generation/edit request, result, and error types; `xrt image generate|edit|bench`; manifest-backed and exact-audited raw Diffusers `xrt image import`; `xrt download --bundle`; `/v1/images/generations`; content-type-dispatched multipart and JSON `/v1/images/edits`; `/v1/runtime/models`; modality-aware load/unload; a shared server-owned GPU manager; bounded queue/auth/lifecycle paths; and generation/edit SSE framing foundations exist. The raw importer recognizes the pinned Qwen generation/edit pipeline classes, rejects remote-code fields, requires every official BF16 artifact byte to match the audited revision, emits a local-only reviewable candidate, and never mutates or downloads into the source. The generation importer passed a real 28-file, 46.9 GB manifest-less smoke. The Edit importer now also passed a complete 33-file, 57,720,454,694-byte manifest-less smoke at revision `6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9`; it produced digest `469e53720300e7728c01a42c4ea989642103997822b87269aeaa3035bb0e19d6` in 28.694 seconds while leaving file count, byte count, and manifest absence unchanged. JSON edits preserve ordered local PNG/JPEG/WebP data URLs, validate exactly one reference kind, recognize `moderation`, and fail closed for unconfigured HTTPS/`file_id` resolvers and all local-file paths. Pinned raw fixtures and internal encoders preserve both image stream event shapes, while public generation/edit requests now reject `stream=true` until compliant usage metering exists. Image-specific CUDA compile/parity and real-smoke wrappers now require explicit confirmation, serialize processes, enforce time and memory preflight bounds, clean new process trees, validate the exact pinned output, and preserve evidence. `xrt image bench --retain-first-output` atomically preserves a first measured PNG without overwriting, and the guarded wrapper now exposes both the locked 16x16 correctness workload and the pinned 512x512 comparator workload. CLI benchmark telemetry reports the active image CUDA backend instead of the generic manager's unprobed default. | Arbitrary or modified Diffusers/fine-tune manifest synthesis is intentionally unsupported rather than inferred from filenames. HTTPS and `file_id` resolution, the optional asynchronous job API, URL output store, and `output_compression` execution remain unimplemented. Quality presets are still server constants rather than bundle-versioned policy. Response-specific enums now normalize default/`auto` and legacy `standard`/`hd` quality values, resolve `background=auto` to `opaque`, and omit non-representable synchronous sizes; focused tests cover every request quality value and local dimensions. Public generation/edit streaming now returns a stable `unsupported_parameter` error until versioned usage metering can populate the required completed-event accounting. Until configured resolvers exist, the honest edit compatibility claim is multipart plus bounded local data URLs rather than every executable reference kind. |
| Qwen-Image-Edit-2511 | The native graph includes the edit processor, independently resized semantic and VAE source branches, VAE encode/decode, vision conditioning, multimodal prompt conditioning, ordered multi-image execution, and `zero_cond_t`. EXIF orientation is applied before bounds validation. Embedded ICC profiles are rejected with a stable sRGB-conversion requirement instead of being silently stripped, and edit preprocessing explicitly retains stored RGB bytes while discarding alpha to match the pinned PIL `convert("RGB")` behavior. A complete pinned Q4_K_M bundle is now locally hash-verified: manifest SHA-256 `24a59c225877206ac1d2b7e6f87db1cd4895b22a98970c0ddf5829bbe79d0bdf`, bundle digest `3e446b89bb0f1ebc1b5d33481cb5fa4adb9c8e2fa07d6c93b1dce785274de978`, and transformer SHA-256 `8677bac90627adbbc11efab87b1870e701c4eb3689ee865a3de8ab81b705a723`. The loader accepts the upstream zero-byte `__index_timestep_zero__` compatibility marker only through an explicit Edit policy and only as the final rank-one `[0]` F32 tensor at the Qwen Image data-section end; strict GGUF loading and malformed marker cases still reject it. Cache-tiled CPU VAE convolution reduced comparable real 1024-pixel source-encode checkpoints by about 3.2-3.5x and completed the full VAE encode in 80.547 seconds. The retained contiguous-row fast path then reduced the two real Edit CUDA source encodes from an 80.064-second mean to 34.722 seconds (2.306x). Ordered BF16 AVX2 further reduced source encoding to 30.555 seconds and prompt encoding from 58.617 to 45.836 seconds (1.279x), without changing the output hash or peak. Query-row CPU attention now uses all available Rayon workers while preserving per-output arithmetic order; five exact serial-oracle release samples at `[1,256,256,8,64]` measured a median 5.053x kernel speedup. Explicit CUDA/sequential edit execution produced the same locked 16x16 two-step PNG SHA-256 `a7210827c5a229ff94b1b0c15752eec65bd937d24fe31a0dbaa77bd7ccb3230f` in all twelve full repetitions. The first two baseline executions took 523.523 and 518.957 seconds; key-parallel CUDA attention reduced the next pair to 234.283 and 237.847 seconds, the shared convolution-row optimization reduced the following pair to 190.361 and 194.655 seconds, ordered BF16 AVX2 reduced the next pair to 176.607 and 174.651 seconds, and parallel score normalization reduced the following pair to 169.745 and 170.214 seconds. The current shared-Q4, spatial-VAE, and feature-major-BF16 paths reduced the final pair to 141.611 and 143.492 seconds; their mean source encoding is 28.678 seconds, prompt encoding is 29.989 seconds, and denoising is 83.842 seconds, with the same `14,921,677,056`-byte tracked peak. Separate ordered two-source and three-source candidate/locked pairs also matched exactly. Two-source used hash `5dde8efa3c6f2c3dc6a159956082e5677a88d4e1307e279f653fc6bdf822e7d3`, peak `16,191,568,128` bytes, and 564.178/562.694-second executions. Three-source required the exact-order tiled long-sequence fallback, used hash `b592ba1d170944f7ca9b41979c1df7d44f0bad4a8a6ee02c1e5d640d128ab31f`, peak `17,460,869,376` bytes, and 1,811.921/1,807.733-second executions. | The full CPU run timed out without an image before the convolution optimization. A 180-second probe and a later 600-second post-attention probe both reached denoising and cancelled cleanly; the later run completed source encoding at 79.676 seconds and prompt encoding at 137.774 seconds but completed zero denoising steps, so large quantized projections and full joint attention remain impractical and no complete CPU output exists. The optimized CUDA results are bounded, active-workstation, experimental correctness evidence only. The configured one-to-three source range now has 16x16 two-step execution evidence, but single/two/three-source means are about 2.4, 9.4, and 30.2 minutes and do not satisfy quiet performance, full-resolution quality, edit identity, source-order attribution, or component parity. `Auto` therefore remains CPU, no Edit tier is advertised, and masked inpainting remains a separate unsupported capability. |
| Quality admission tooling | `reference/image/qwen/evaluate_quality_suite.py` now validates the frozen suite and fixtures, emits deterministic per-tier plans, requires exact evaluator identity and logical model pairing, hash-checks every unique 1024px PNG, rejects incomplete/non-finite/blank/uniform evidence, checks BF16 and candidate absolute floors, computes the three paired one-sided bootstrap gates with the frozen PCG64 policy, and enforces blinded three-rater Wilson gates over at least 200 stratified pairs plus all 50 identity pairs. Invalid input exits without a report; a failed but complete run writes evidence and exits nonzero. `xrt image bench --json` now embeds the frozen suite version and SHA-256. | The compiler has passed synthetic and fail-closed validation only; it consumes precomputed metrics and does not run evaluators. No reproducible evaluator/export producer exists, and the current optional Python dependencies do not by themselves define a runnable full PaddleOCR-VL pipeline or inference engine. No actual 1024x1024 BF16/candidate corpus, pinned evaluator export, or human ratings have been supplied, so this closes the reporting seam but no tier's quality gate. |
| Production status | No image model or quantization is production-advertised. No release was requested or performed. | Every applicable definition-of-done item in Section 24.1 must pass before changing this status. |

Focused verification on 2026-07-22 passed 60 non-default `xrt-image` unit tests (one manual unit smoke ignored), 30 image-enabled server tests, 12 `xrt-openai` tests, 18 image-and-CUDA-enabled CLI tests, nine hub tests, the full no-default-feature workspace suite, 61 CUDA-feature `xrt-image` unit tests (one manual unit smoke ignored), and exactly three selected CUDA parity tests through `safe-image-cuda-check.ps1`: tiny generation, tiny zero-conditioned Edit, and a 12,289-key tiled-attention case beyond the old portable shared-memory gate. The latest scoped rerun additionally passed 16 focused CPU image-kernel tests (two manual benchmarks ignored), seven non-device CUDA-feature library tests (31 device tests ignored), the same 60/61 CPU/CUDA `xrt-image` unit counts, all three guarded GPU tests, 19 image-generation CLI tests, and seven quality-admission-tool tests. A separate ignored CUDA kernel test exercises partial output and activation tiles (`9x512` weights and 17 activation rows), passes scalar-reference tolerance, and repeats bit-exactly through the retained shared-activation Q4 path. CPU kernel tests now also prove bit-exact single-thread versus multi-thread spatial convolution and bit-exact batched feature-major versus single-row BF16 linear scheduling; the real prompt-oracle test passes its pinned Diffusers tolerances. The non-default and CUDA test invocations also left their separate real-model integration suites ignored/manual, so none were counted as ordinary test passes. Six GGUF parser tests prove strict rejection plus the exact Edit-marker compatibility boundary. The CPU image kernel suite includes a bit-exact parallel-attention comparison against the prior serial multi-batch/multi-head arithmetic order; an ignored release benchmark recorded five 32-thread samples with a median 5.053x improvement.

Four explicitly selected manual real-model tests per Q8_0, Q6_K, and Q5_K_M generation tier passed serially in release mode: complete bundle validation, GGUF executor construction, exact dtype-map validation, and a hash-locked CPU generation smoke. Q6_K and Q5_K_M then passed unpinned-candidate plus locked-repeat native CUDA runs through the generalized `safe-image-cuda-smoke.ps1`; both pairs have identical per-tier output hashes and allocation peaks. The complete Edit-2511 Q4_K_M bundle passed real CPU load/plan, real CUDA load/plan, targeted vision/VAE/phase probes, twelve full single-image explicit-CUDA bounded repetitions with the same locked output hash, and separate two-image and three-image candidate-plus-locked pairs. The regenerated OpenAI fixture set (15 hashed wire files plus its manifest) and all ten Phase 0 component manifests also verify against their live immutable pins. Release-mode raw-import smokes verified all 28 files and 46,926,328,525 bytes of the pinned Qwen-Image-2512 BF16 tree and all 33 files and 57,720,454,694 bytes of the pinned Edit-2511 BF16 tree without source manifests; their warm validation checkpoints were 30.516 and 28.694 seconds respectively, and neither source was mutated.

The bounded full native Q4 generation CUDA smoke also passed through `safe-image-cuda-smoke.ps1` with the exact expected output; two preceding successful-child wrapper-validation failures are retained separately and make no inference claim. The new 512x512 comparator-workload path retained a valid PNG and raw child report on its first successful model run, then exposed a PowerShell-5.1-only post-processing incompatibility; that run is retained as a non-admission wrapper-failure record. After replacing the unsupported path API, the locked repeat passed, independently matched the first PNG hash, and wrote the authoritative evidence plus retained PNG. The eight-warp shared-activation-tile pair is recorded in `benchmark-results/image/native/qwen-image-2512-q4_k_m-cuda-comparator-512x512-s4-seed424242-post-q4-shared-activation-tile-2026-07-22.json` and its `-repeat2-` companion. The retained sixteen-warp pair adds `-warps16-` to those names; all four have retained PNGs and exact hash/allocation validation. The real single-image Edit smoke used its ignored release test with an internal 900-second cooperative-cancellation bound; it is recorded in `benchmark-results/image/native/qwen-image-edit-2511-q4_k_m-cuda-smoke-16x16-s2-2026-07-22.json`. The separate two-image diagnostic used a 1,800-second bound and is recorded in `benchmark-results/image/native/qwen-image-edit-2511-q4_k_m-cuda-two-image-smoke-16x16-s2-2026-07-22.json`; the maximum-source diagnostic used a 2,100-second bound and is recorded in `benchmark-results/image/native/qwen-image-edit-2511-q4_k_m-cuda-three-image-smoke-16x16-s2-2026-07-22.json`. None is a quiet-performance claim.

Strict warnings-denied Clippy passes for `xrt-openai`, `xrt-image`, and the changed image-enabled CLI/server surfaces when scoped to those crates (`--no-deps`) or when unrelated pre-existing lint debt is allowed; repository-wide warnings-denied Clippy remains blocked by existing hybrid-worktree lint debt outside this work.

The debug-spec audit reproduced the `cuda-kernel-ptx` workflow command inside its pinned `nvidia/cuda:12.8.1-devel-ubuntu22.04` container and initially found stale checked-in `q4_k_recurrent.ptx` and `kquant_mmq.ptx` artifacts. The integration regenerated both with the workflow's exact CUDA 12.8.1 flags. `q4_k_recurrent.ptx` now has SHA-256 `7b853b0c36f59b6a00c483a8a6e6695b04b0f8d92d5df3dd67ad5dbc3e756127` and 163,555 bytes; `kquant_mmq.ptx` has SHA-256 `b1ab73a81af8a25cf2e2ff5cbeacfbe2ff43878c36bbcfa8c12ca6d52828fff9` and 106,311 bytes. A clean pinned-container replay byte-compared all nine generated PTX files successfully. This closes PTX reproducibility for the retained sources and flags, but it does not by itself admit image or hybrid CUDA performance.

## 8. External research baseline

All availability claims in this section were refreshed on 2026-07-22 and must be refreshed again before production admission.

### 8.1 Qwen targets

The official [Qwen-Image-2512 model](https://huggingface.co/Qwen/Qwen-Image-2512) is an Apache-2.0, 20B-class image model distributed as an official Diffusers bundle. The revision observed in this audit is `25468b98e3276ca6700de15c6628e51b7de54a26`. Its `model_index.json` identifies `QwenImagePipeline`, `QwenImageTransformer2DModel`, `Qwen2_5_VLForConditionalGeneration`, `Qwen2Tokenizer`, `AutoencoderKLQwenImage`, and `FlowMatchEulerDiscreteScheduler`. The public transformer configuration currently describes 60 layers, 24 attention heads, head dimension 128, 64 input channels, 16 output channels, patch size 2, joint-attention dimension 3584, and three-axis rotary dimensions `[16, 56, 56]`. These values are compatibility fixtures, not constants to embed in kernels.

The official [Qwen-Image-Edit-2511 model](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) is also Apache-2.0; the revision observed in this audit is `6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9`. It uses `QwenImageEditPlusPipeline`, adds a `Qwen2VLProcessor`, accepts multiple source images, and uses an editing transformer configuration with `zero_cond_t`. The [Qwen Image paper](https://arxiv.org/abs/2508.02324) describes the 20B MMDiT and frozen Qwen2.5-VL foundation and the dual semantic/reconstruction conditioning used for editing.

The [official Diffusers Qwen Image documentation](https://huggingface.co/docs/diffusers/api/pipelines/qwenimage) is the behavioral reference for prompt encoding, FlowMatch Euler scheduling, true classifier-free guidance, generation, editing, and VAE behavior. For the current family, `true_cfg_scale` is the meaningful CFG control; `guidance_scale` must not be silently treated as the same parameter.

The official [Qwen-Image-3.0 announcement](https://qwen.ai/blog?id=qwen-image-3.0) was published on 2026-07-21. During this audit, the announcement directed users to Qwen Chat, and no official checkpoint, Hugging Face model tree, architecture configuration, license, or local inference repository was found in the reviewed official catalog. This is an observation, not a permanent assumption.

### 8.2 Quantized interoperability targets

Community repositories provide useful compatibility fixtures:

- [Unsloth Qwen-Image-2512 GGUF](https://huggingface.co/unsloth/Qwen-Image-2512-GGUF) lists transformer artifacts around 13.2 GB for Q4_K_M, 15 GB for Q5_K_M, 16.8 GB for Q6_K, and 21.8 GB for Q8_0.
- [Unsloth Qwen-Image-Edit-2511 GGUF](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF) publishes a similar transformer quantization set.
- [city96 Qwen-Image GGUF](https://huggingface.co/city96/Qwen-Image-gguf) explicitly separates the diffusion transformer from the Qwen2.5-VL text encoder and VAE.
- [QuantStack Qwen-Image-Edit GGUF](https://huggingface.co/QuantStack/Qwen-Image-Edit-GGUF) demonstrates that edit execution may require a transformer GGUF, text-encoder GGUF, projection component, and SafeTensors VAE.

These are not first-party compatibility guarantees. Phase 0 must pin exact revisions, enumerate every component, record SHA-256 hashes, and verify tensor layouts. XENO RT must never infer complete support from a transformer filename alone.

### 8.3 Reference engines

[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) supports Qwen Image-family generation/editing, GGUF and SafeTensors, CPU/GPU placement, VAE tiling, Flash Attention, and multiple hardware backends. It is the closest external native inference and performance comparator. Its component mapping and memory behavior may be studied, but XENO RT must implement against its own abstractions and preserve source provenance.

Diffusers is the official numerical oracle. ComfyUI and [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) are interoperability references, not the target product architecture.

### 8.4 Ideogram 4 deferral

The [Ideogram 4 reference repository](https://github.com/ideogram-oss/ideogram4) describes a 9.3B single-stream flow-matching DiT, official quantized paths, and a 24 GiB target. However, [Ideogram's weights licensing](https://ideogram.ai/licensing/) distinguishes the public non-commercial weights from commercial and customer-facing use. XENO RT may benchmark the public code internally only within its terms. Product adapter work, bundling, or customer-facing support requires recorded legal approval.

## 9. Functional and compatibility requirements

### 9.1 Functional requirements

| ID | Requirement |
|---|---|
| IMG-F01 | `xrt-image` shall generate a PNG from a Qwen-Image-2512 prompt without any Python or external process. |
| IMG-F02 | `xrt-image` shall edit one to three ordered source images with Qwen-Image-Edit-2511. |
| IMG-F03 | The OpenAI edit schema shall parse an optional mask, but a runtime shall accept it only when the selected model advertises `image.inpaint`; Qwen-Image-Edit-2511 alone shall return `unsupported_parameter` because its official Edit Plus call has no mask input. |
| IMG-F04 | The loader shall validate all component roles, formats, hashes, architectures, tensor shapes, and cross-component dimensions before allocating accelerator memory. |
| IMG-F05 | The scheduler implementation shall be driven by the bundle configuration and match the official FlowMatch Euler reference at recorded checkpoints. |
| IMG-F06 | The pipeline shall expose seed, dimensions, step count, negative prompt, true CFG scale, and output format. |
| IMG-F07 | Cancellation shall be checked between components and at every denoising step; cancelled jobs shall release leases and temporary files. |
| IMG-F08 | CPU-only builds shall execute every advertised bundle tier when sufficient host RAM exists. |
| IMG-F09 | CUDA builds shall support explicit full-residency and staged component-offload plans. |
| IMG-F10 | VAE decode shall support slicing and tiling with overlap blending and seam regression tests. |
| IMG-F11 | The runtime shall support one or more progress subscribers without making preview generation mandatory. |
| IMG-F12 | A request shall receive an immutable placement plan before accelerator allocations or component-residency changes begin. Queue/job bookkeeping may exist before that point, but a failed admission shall not mutate loaded runtime state. |
| IMG-F13 | Installed bundles shall execute with network access disabled. |
| IMG-F14 | The loader shall reject unsupported GGUF tensor types or mappings; it shall never reinterpret an unknown layout as a supported quantization. |
| IMG-F15 | The server shall expose bounded model load, unload, generation, and edit operations; if the XENO asynchronous extension is enabled, it shall also expose bounded status, result, event, and cancellation operations. |
| IMG-F16 | A generation or edit request with `n > 1` shall return an ordered batch result with one recorded derived seed per output. |
| IMG-F17 | Every asynchronous job shall have an authorized owner, bounded queue/run lifetime, retrievable result or terminal error, and defined unload/shutdown behavior. |

### 9.2 Compatibility requirements

| ID | Requirement |
|---|---|
| IMG-C01 | Existing OpenAI text endpoint schema snapshots shall remain byte-for-byte structurally compatible. |
| IMG-C02 | Existing `/v1/images/remove-background` behavior shall remain in `xrt-vision` and shall not be routed through a generative pipeline. |
| IMG-C03 | `POST /v1/images/generations` and `POST /v1/images/edits` shall preserve standard OpenAI field names and error envelopes. |
| IMG-C04 | XENO-only controls shall live under `x_xeno` or separate `/v1/xeno/*` endpoints, never as required changes to a standard request. |
| IMG-C05 | `/v1/models` shall continue returning the existing basic model-object schema. Rich capability data belongs on `/v1/runtime/models`. |
| IMG-C06 | `POST /v1/runtime/load` shall default a missing `modality` field to `text`; bodyless `POST /v1/runtime/unload` shall remain valid and retain its current text-default behavior. |
| IMG-C07 | Existing text commands and flags shall not change. Image commands shall use a new nested `xrt image` command. |
| IMG-C08 | GGUF remains supported for text and image components. SafeTensors is additive. |
| IMG-C09 | A default non-CUDA build shall compile and pass synthetic image-pipeline tests. |
| IMG-C10 | No image feature may select an external inference service implicitly. |
| IMG-C11 | `/v1/images/edits` shall dispatch by content type and recognize both pinned OpenAI request forms: multipart uploads and JSON `images`/optional `mask` references. Each JSON reference shall contain exactly one `image_url` or `file_id`; a bounded base64 `data:` URL is an `image_url`, not a XENO-only request form. Multipart-only support shall be labeled a compatibility subset, not full current edit-transport compatibility. |
| IMG-C12 | Request-side compatibility enums and flexible local dimensions shall not be reused blindly as response types. Synchronous and streaming responses shall serialize only values allowed by their separately pinned official schemas: resolve `auto` and legacy quality/background choices to effective response values, omit optional synchronous fields that cannot represent a local value, and reject streaming combinations whose required event fields cannot be represented. |
| IMG-C13 | Required streaming `usage` shall be derived from versioned runtime metering and remain internally consistent. A fabricated all-zero placeholder is forbidden; a runtime without the required accounting shall reject `stream=true` until it can produce compliant usage. |

### 9.3 Performance and reliability requirements

| ID | Requirement |
|---|---|
| IMG-P01 | The 4090 Q4_K_M release bundle shall keep exact xeno-owned peak device allocation at or below the smaller of 22 GiB and the admission budget after observed non-XENO use plus configured reserve are deducted. |
| IMG-P02 | Model weights shall not cross PCIe inside the denoising-step loop after the denoiser phase begins. |
| IMG-P03 | The runtime shall preallocate or reuse denoiser scratch after a shape is admitted. |
| IMG-P04 | Default image concurrency is one per CUDA device until measured aggregate throughput and peak memory justify more. |
| IMG-P05 | Queue length, request body size, decoded pixels, output count, dimensions, and steps shall all have bounded defaults. |
| IMG-P06 | An OOM or component error shall fail one job without corrupting a loaded text or image runtime. |
| IMG-P07 | Image work shall not evict or unload an active text runtime unless an operator explicitly selects an image-exclusive policy. |
| IMG-P08 | Existing controlled text benchmarks shall regress by no more than 2% median after three matched runs; noise above 2% requires five-run confirmation. |
| IMG-P09 | Production CUDA admission requires the one-sided 95% upper confidence bound for the warm median-time ratio to be at most 1.15 versus pinned stable-diffusion.cpp on the same RTX 4090 workload. Earlier support remains labeled experimental. |
| IMG-P10 | Every benchmark report shall include model component hashes, build commit, backend, quantization, dimensions, steps, seed, hardware, driver, and memory policy. |
| IMG-P11 | Per-principal queue depth, job deadline, progress-channel capacity, encoded-result bytes, result-store bytes, and result TTL shall have finite operator-configurable limits. |
| IMG-P12 | Model loading, CPU kernels, codecs, hashing, and blocking filesystem work shall run on a bounded image worker/blocking pool, never on Axum/Tokio core executor threads. |

## 10. Target architecture

### 10.1 Domain layout

```text
                         xrt-server / xrt-cli / bindings
                                      |
              +-----------------------+-----------------------+
              |             |             |         |        |
       current text     xrt-image     xrt-vision   future   future
       Runtime          ImageRuntime   tasks       xrt-video xrt-audio
              |             |             |         |        |
              +-------------+-------------+---------+--------+
                                      |
        xrt-runtime resource admission, xrt-hub bundles, xrt-core
          xrt-gguf, xrt-safetensors, xrt-tokenizer, xrt-kernels
                              xrt-cuda (optional)
```

The diagram is a dependency concept, not a mandate to make `xrt-runtime` depend on every modality crate. During this project:

1. `xrt-image` depends on the shared foundation and the public resource-management subset of `xrt-runtime`.
2. `xrt-runtime` does not depend on `xrt-image`.
3. `xrt-server` owns both `Arc<Runtime>` for text and `Arc<ImageRuntime>` for image.
4. A small server-side `LoadedRuntimeSet` maps `(modality, model_id)` to runtime state.
5. If the shared surface grows beyond resource admission and status types, extract a later `xrt-engine-core`; do not create that crate preemptively.

The server shall construct a registry of one `Arc<GpuResourceManager>` per CUDA device at startup from the existing GPU budget controls. `xrt-runtime` gains additive load entry points that accept that `Arc`; all current public text constructors remain and delegate through the new path with a newly constructed manager so library compatibility is preserved. Server-managed text and image loads must use the injected path. A runtime must reject a resource manager for the wrong device ordinal or incompatible budget configuration. Multi-GPU support uses the device ordinal as the registry key rather than sharing one arena across devices.

`external-openai` remains a text-routing mode. Activating or unloading it may transition the local text runtime exactly as today, but it must neither clear a loaded image runtime nor cause `/v1/images/*` to proxy externally. A future external image adapter requires a separate specification and explicit operator selection.

### 10.2 Shared versus domain-specific state

| Shared | Text-specific | Image-specific |
|---|---|---|
| Device discovery and handles | KV cache | Latent tensor |
| GPU budget and allocation leases | Token sampler | Scheduler timesteps/sigmas |
| Tensor formats and checksums | Token history | Prompt/image conditioning |
| Linear/attention/normalization kernels | Prefix cache | Denoising scratch |
| Model cache and artifact locks | Causal session | VAE tiles and decode scratch |
| Cancellation and bounded queues | Decode batching | Image job progress/previews |
| Telemetry primitives | Tokens per second | Steps per second and seconds per image |

Image state must never be stored in a text `BackendSession`, and image jobs must not appear as fake token sequences.

### 10.3 Capability identifiers

The shared model registry uses stable capability strings:

- `text.generate`
- `image.generate`
- `image.edit`
- `image.inpaint`
- `vision.background_removal`
- reserved future prefixes `video.*` and `audio.*`

Routing is based on capability plus model ID, not filename extension.

## 11. `xrt-image` crate and runtime contracts

### 11.1 Crate features

`crates/xrt-image/Cargo.toml` defines:

- default CPU support with no CUDA dependency;
- optional `cuda` forwarding to `xrt-cuda` and `xrt-runtime/cuda`;
- image codecs limited to explicitly supported PNG, JPEG, and WebP crates; and
- no Python, libtorch, ONNX Runtime, or external-process dependency for Qwen Image.

`xrt-cli` and `xrt-server` each add an optional `xrt-image` dependency plus an `image-generation` feature; their `cuda` features forward to `xrt-image/cuda` when image generation is enabled. Cargo features are package-scoped, so the root test package does not pretend to enable sibling-package features. The workspace root adds the new member and may add an integration-test-only optional dependency/feature if tests require it. Image generation remains off in default CLI/server packaging until the production gates pass, while `xrt-image` itself always has a CPU-compilable default.

### 11.2 Proposed public Rust API

The following names define the implementation contract. Minor naming changes are acceptable only if the same ownership and invariants remain visible.

```rust
pub struct ImageRuntime;

impl ImageRuntime {
    pub fn load(
        bundle: ImageModelBundle,
        backend: ImageBackendKind,
        resources: Arc<GpuResourceManager>,
    ) -> Result<Self, ImageError>;

    pub fn generate(
        &self,
        request: ImageGenerationRequest,
        cancellation: ImageCancellation,
        progress: Option<Arc<dyn ImageProgressSink>>,
    ) -> Result<ImageBatchResult, ImageError>;

    pub fn edit(
        &self,
        request: ImageEditRequest,
        cancellation: ImageCancellation,
        progress: Option<Arc<dyn ImageProgressSink>>,
    ) -> Result<ImageBatchResult, ImageError>;
}
```

`ImageRuntime` is immutable after load except for bounded caches, metrics, and scheduler state behind explicit synchronization. Each request owns its latent, scheduler cursor, RNG, conditioning references, scratch lease, and cancellation state.

### 11.3 Internal pipeline trait

```rust
trait ImagePipeline: Send + Sync {
    fn capabilities(&self) -> &[ImageCapability];
    fn plan(&self, request: &ImageRequest) -> Result<ImageExecutionPlan, ImageError>;
    fn execute(
        &self,
        plan: ImageExecutionPlan,
        cancellation: &ImageCancellation,
        progress: Option<&dyn ImageProgressSink>,
    ) -> Result<ImageBatchResult, ImageError>;
}
```

Only adapters implement this trait. The initial implementation is `QwenImagePipeline`. Generation and editing share lower-level components but have separate request validation and conditioning builders.

### 11.4 Request and result types

`ImageGenerationRequest` contains:

- model ID;
- prompt and optional negative prompt;
- width and height;
- output count;
- step count;
- true CFG scale;
- one deterministic `u64` base seed, with per-output seeds derived during admission;
- output format and quality;
- backend/offload policy; and
- optional preview interval.

`ImageEditRequest` adds:

- one to three decoded and ordered input images;
- an optional normalized mask, admitted only for a pipeline with `image.inpaint` capability;
- edit-strength/model-specific conditioning controls; and
- preprocessing policy recorded in result metadata.

`ImageBatchResult` contains an ordered, non-empty `Vec<ImageResult>`, batch-level timings, and versioned metering sufficient to construct any advertised response usage: prompt text tokens, input image tokens for edits, output image tokens, and their required details/totals. Each `ImageResult` contains encoded image bytes, MIME type, dimensions, its derived seed, model ID, bundle digest, backend, quantization tier, timings, and optional safe-to-return metadata. If an adapter cannot measure usage under the pinned accounting schema, its streaming API remains unsupported; it never substitutes fabricated zeros. Raw prompts, input images, local paths, and component URLs are never included in default logs or status output. Progress events always include the zero-based output index so multi-output requests cannot interleave ambiguously.

### 11.5 Error contract

`ImageError` must distinguish:

- invalid request;
- unsupported capability, quantization, tensor, shape, or backend;
- missing/corrupt component;
- checksum or manifest failure;
- admission/insufficient memory;
- cancellation;
- codec/input limit failure;
- execution/numerical failure; and
- internal invariant failure.

Errors become stable OpenAI-style HTTP errors at the server boundary. Internal paths and secrets are redacted.

## 12. Model bundle and artifact contract

### 12.1 Bundle manifest

Every supported installed model has an `xrt.bundle.json` with schema version 1. Reviewed development manifests currently live under `reference/image/qwen/manifests/`. Release packaging shall copy approved catalog entries to `catalog/image/` beside the executable or to `$XRT_CACHE_DIR/catalog/image/`, matching the current CLI resolver; `xrt-hub` owns installation and cache integrity, not a nonexistent `crates/xrt-hub/manifests/` source tree. An installed cache directory contains the exact resolved manifest beside the artifacts. Schema version 1 does not claim signature verification. Adding signatures later requires a separate trust-root, key-rotation, revocation, and rollback contract rather than an unused `signature` field.

Illustrative schema:

```json
{
  "schema_version": 1,
  "id": "qwen-image-2512-q4_k_m",
  "family": "qwen-image",
  "revision": "xeno-catalog-entry-revision",
  "capabilities": ["image.generate"],
  "license": {
    "spdx": "Apache-2.0",
    "evidence": "https://huggingface.co/Qwen/Qwen-Image-2512/blob/25468b98e3276ca6700de15c6628e51b7de54a26/README.md",
    "files": []
  },
  "quantization": "Q4_K_M",
  "components": [
    {
      "role": "transformer",
      "format": "gguf",
      "files": [
        {
          "path": "transformer/model-q4_k_m.gguf",
          "size_bytes": 0,
          "sha256": "64-hex-required",
          "source": "https://huggingface.co/<repo>/resolve/<immutable-revision>/<file>"
        }
      ]
    },
    {
      "role": "text_encoder",
      "format": "gguf",
      "files": [
        {
          "path": "text_encoder/model.gguf",
          "size_bytes": 0,
          "sha256": "64-hex-required",
          "source": "https://huggingface.co/<repo>/resolve/<immutable-revision>/<file>"
        }
      ]
    },
    {
      "role": "tokenizer",
      "format": "huggingface-json",
      "files": [
        {
          "path": "tokenizer/tokenizer_config.json",
          "size_bytes": 0,
          "sha256": "64-hex-required",
          "source": "https://huggingface.co/Qwen/Qwen-Image-2512/resolve/25468b98e3276ca6700de15c6628e51b7de54a26/tokenizer/tokenizer_config.json"
        }
      ]
    },
    {
      "role": "vae",
      "format": "safetensors",
      "files": [
        {
          "path": "vae/diffusion_pytorch_model.safetensors",
          "size_bytes": 0,
          "sha256": "64-hex-required",
          "source": "https://huggingface.co/Qwen/Qwen-Image-2512/resolve/25468b98e3276ca6700de15c6628e51b7de54a26/vae/diffusion_pytorch_model.safetensors"
        }
      ]
    },
    {
      "role": "scheduler",
      "format": "json",
      "files": [
        {
          "path": "scheduler/scheduler_config.json",
          "size_bytes": 0,
          "sha256": "64-hex-required",
          "source": "https://huggingface.co/Qwen/Qwen-Image-2512/resolve/25468b98e3276ca6700de15c6628e51b7de54a26/scheduler/scheduler_config.json"
        }
      ]
    }
  ],
  "limits": {
    "max_sequence_length": 512,
    "max_width": 4096,
    "max_height": 4096,
    "max_pixels": 16777216
  }
}
```

The arrays above are abbreviated to show the schema shape; a production component lists every file needed for that role, including configuration and tokenizer assets. Directories are never hashed as opaque filesystem objects. Each catalog file has a normalized relative path, positive byte length, SHA-256, and credential-free source URL pinned to an immutable revision. The validator rejects zero sizes, placeholders, mutable `resolve/main` URLs, duplicate `(role, path)` pairs, absolute paths, `..`, symlinks, and omitted required files. Phase 0 replaces every placeholder and checks the complete list; no implementer may invent or omit a hash to make a test pass.

License evidence is mandatory. License files are listed and preserved when the upstream revision contains them; they are not fabricated when the upstream model card records the SPDX license without a standalone `LICENSE` or `NOTICE` artifact. The evidence URL itself is revision-pinned.

### 12.2 Required component roles

| Capability | Required roles |
|---|---|
| Qwen generation | transformer, text encoder, tokenizer, VAE, scheduler |
| Qwen editing | transformer, text encoder, tokenizer, processor, VAE, scheduler, and every projection/vision component named by the pinned reference |
| Qwen inpainting, if admitted | the generation roles plus a reference-validated inpaint graph and mask preprocessor; it is not implied by the edit bundle |

Optional roles include preview decoder, LoRA adapters in a future specification, and metadata/license files. Unknown required roles fail load. Unknown optional roles are retained in metadata but ignored only when the manifest marks them optional.

### 12.3 Import behavior

The target loader supports two explicit entry paths:

1. **Catalog install:** `xrt-hub` resolves a known model ID to a pinned XENO manifest, acquires a cancellation-aware per-bundle-digest lock with a finite wait, and downloads immutable-revision URLs into a randomly named staging directory on the same filesystem as the cache. It applies bounded retries only to transient transport failures, verifies every size and SHA-256, fsyncs files and the staging directory where supported, and atomically renames the whole directory into its digest-qualified final location. Only after that rename may an atomic local index update make the bundle discoverable. Hash/config failures delete or quarantine staging and are never retried as a different artifact; startup recovery removes stale staging and verifies then re-indexes (or prunes after a grace period) a complete digest directory orphaned between rename and index update. Existing one-file GGUF cache behavior remains compatible and separate.
2. **Local import:** the implemented path accepts either a local directory containing `xrt.bundle.json` and every declared artifact or an exact audited raw Diffusers tree for Qwen-Image-2512 BF16 or Qwen-Image-Edit-2511 BF16. Raw discovery reads only the bounded `model_index.json`, rejects custom/remote-code fields and unknown pipeline classes, converts the matching immutable catalog manifest to `source_kind=local`, and then verifies every declared size and SHA-256 before validation or installation. Without `--install`, it prints the reviewable candidate manifest; with `--install`, `xrt-hub` atomically imports the same verified plan. It never executes model code, follows symlinks, downloads implicitly, records the operator's absolute source path, or guesses support for a modified/fine-tuned tree. A recognized class with any byte drift fails as an exact-artifact mismatch.

`trust_remote_code` is never supported. Symlinks and canonical paths must remain under the selected bundle root unless the operator explicitly uses an already-resolved absolute local component path at CLI load time. Catalog manifests and public image request bodies cannot contain credentials, signed query strings, or local paths. In the private manifest variant created only by explicit local import, each file records `"source_kind": "local"` instead of `source`; the canonical bundle digest includes that marker and the content digests. The validator rejects this variant in catalog/export mode, and neither the manifest nor cache index copies the operator's absolute source path into public metadata.

Catalog transport accepts HTTPS sources only, enforces declared-size and total-install byte caps while streaming, and has finite connect/read/total deadlines. Redirect count is bounded; every redirect is revalidated against the catalog's reviewed origin/CDN host set, HTTPS downgrade and private/loopback targets are rejected, and authorization headers are never forwarded across origins. Transient CDN redirect query parameters may exist only in the in-memory transport flow and are never written back into the manifest or logs.

### 12.4 Cache layout

The existing cache root remains canonical to avoid migration churn:

```text
~/.cache/xrt/models/
  bundles/
    qwen-image-2512-q4_k_m/
      <full-bundle-digest>/
        xrt.bundle.json
        transformer/
        text_encoder/
        tokenizer/
        vae/
        scheduler/
  .staging/
  manifests/
```

`XRT_CACHE_DIR`, `--cache-dir`, and `ModelHub::with_cache_dir` are implemented in the integration candidate. Precedence is explicit CLI/library path, then `XRT_CACHE_DIR`, then the existing `~/.cache/xrt/models` default. A managed `~/.xeno/models` root is read only when explicitly configured and is never silently moved or rewritten. Locks are per full bundle digest, so concurrent installers do not duplicate downloads while unrelated bundles may proceed.

### 12.5 Bundle identity

The stable bundle digest is `SHA256("xrt-bundle-v1\0" || canonical_manifest_bytes)`. Canonical bytes use UTF-8, lexicographically sorted object keys, no insignificant whitespace, preserved JSON array order, and no `bundle_digest` field; component/file arrays are emitted in lexicographic `(role, path)` order before canonicalization. Every file digest and size is therefore already covered. Golden-byte fixtures pin canonicalization across platforms. The digest keys prompt-embedding caches, result metadata, benchmark records, and compatibility fixtures. Filenames and repository names alone are not identities.

## 13. Qwen-Image-2512 generation pipeline

### 13.1 Component graph

The adapter shall load configuration rather than hardcode public dimensions. The pinned 2512 fixture is expected to resolve:

1. Qwen2.5-VL text encoder and Qwen2 tokenizer;
2. Qwen Image MMDiT transformer;
3. FlowMatch Euler scheduler with dynamic shifting from its config; and
4. `AutoencoderKLQwenImage` VAE.

The adapter rejects a bundle when text hidden size, joint-attention size, latent channels, patch geometry, RoPE axes, VAE scale factors, or transformer input/output channels are inconsistent.

### 13.2 Execution sequence

For each output image:

1. Validate dimensions, pixel count, step count, prompt length, output count, backend, and memory plan.
2. Canonicalize the request and derive each output seed with `base_seed.checked_add(output_index as u64)`. Reject the request before execution if the addition overflows; wrapping is forbidden. Output zero therefore retains the caller's seed and the ordered derivation is pinned by tests.
3. Tokenize and encode positive and, when required, negative prompts exactly as the pinned official pipeline does.
4. A bounded generation-only prompt cache may retain immutable embeddings under bundle digest, tokenizer/processor digest, prompt-template/cache-key schema version, maximum sequence length, encoder precision, principal namespace, and a process-keyed HMAC-SHA-256 of the canonical positive and negative prompt bytes. Raw prompt bytes and unkeyed prompt hashes are never retained as keys or emitted in metrics. Cross-principal reuse is disabled by default, the key secret is regenerated on process start, entries have byte/count/TTL limits, and edit conditioning is not cached in version 1. The server supplies the opaque namespace outside the public image schema; direct Rust callers default to one private runtime-local namespace or may disable the cache.
5. Create the initial latent with XENO's specified normal RNG algorithm. The algorithm, normal transform, counter/stream assignment, dtype conversion, and byte order are versioned so determinism does not depend on Rust's default RNG implementation. Version 1 does not claim that a XENO seed produces the same noise as PyTorch CPU/CUDA generators; official downstream parity supplies the identical recorded latent to both implementations.
6. Build timesteps and sigmas through the configured FlowMatch Euler scheduler, including its sequence-length-dependent shift.
7. For every step, assemble latent, timestep, prompt embeddings, attention masks, image IDs, and rotary embeddings; invoke the MMDiT; apply true classifier-free guidance when requested; and advance the latent through the scheduler.
8. Check cancellation and finite values after every step. A NaN/Inf aborts with step and component metadata but no prompt content.
9. Denormalize the final latent using the VAE config's mean/std contract.
10. Decode through the VAE, optionally by slices or overlap-blended tiles.
11. Clamp and convert pixels using the pinned reference convention, encode the requested format, and atomically return or write the result.

### 13.3 Numerical conformance checkpoints

Phase 2 records reference artifacts for:

- token IDs and attention masks;
- positive and negative prompt embeddings;
- rotary/image position IDs;
- XENO RNG golden vectors and initial noise for fixed seeds;
- scheduler timesteps, sigmas, and dynamic shift;
- transformer output at steps 0, midpoint, and final step;
- latent state after the same steps;
- VAE pre/post-normalization values; and
- final decoded pixels before image compression.

Small tensors are stored directly in test fixtures. Large tensors are represented by shape, dtype, finite-value statistics, deterministic sampled values, and a cryptographic digest of the full reference artifact kept in the benchmark bundle.

For transformer/scheduler/VAE parity, the reference harness passes the XENO-generated initial latent through the official pipeline's supported `latents` input rather than conflating model correctness with PyTorch RNG compatibility. A future PyTorch-compatible RNG profile, if useful, is a separately named and tested policy; it cannot silently replace the version-1 seed contract.

### 13.4 Prompt and scheduler semantics

- `true_cfg_scale` maps to the official Qwen CFG behavior.
- A standard OpenAI `quality` preset may choose steps and default CFG, but explicit `x_xeno.steps` and `x_xeno.true_cfg_scale` take precedence.
- An unsupported standard `guidance_scale` alias is rejected or documented as ignored; it is never silently substituted for true CFG.
- Scheduler configuration is part of bundle identity. Changing shift rules or step spacing produces a different result contract.
- Width and height must satisfy model patch/VAE divisibility. The API returns a validation error rather than silently rounding unless `x_xeno.resize_policy` explicitly requests deterministic rounding.

## 14. Qwen-Image-Edit-2511 editing pipeline

Editing is production-admitted only after generation, text encoding, VAE decode, scheduler, and quantized transformer execution have passed their gates. Implementation may proceed ahead of admission, as the integration candidate does, but it reuses those components and must still pass independent conditioning tests.

The official `QwenImageEditPlusPipeline.__call__` accepts source image(s), prompt conditioning, steps, guidance, and latents, but it does not accept `mask_image`. Diffusers documents masked Qwen Image inpainting through the separate `QwenImageInpaintPipeline`. XENO RT must preserve that distinction: Qwen-Image-Edit-2511 provides semantic and multi-image editing, while masked editing is advertised only if the Qwen-Image-2512 components pass a separate inpaint conformance gate. A mask is never silently ignored or implemented as unreported final-image compositing.

### 14.1 Input normalization

1. Decode PNG, JPEG, or WebP with total-byte, dimension, and decoded-pixel limits.
2. Apply EXIF orientation before hashing the normalized input.
3. Normalize decoded pixels to RGBA8. Reject any embedded ICC profile with a stable error requiring the caller to convert the input to sRGB; never strip the profile while claiming an unmanaged conversion. For Qwen edit conditioning, match the pinned PIL `convert("RGB")` behavior by retaining the stored RGB channels and discarding alpha without compositing; keep this policy covered by transparent and partially transparent pixel fixtures.
4. Preserve the caller's image order for multi-image conditioning.
5. Process each source independently, matching the pinned Edit Plus processor: preserve aspect ratio, create the semantic/vision branch around the configured 384-squared target area with processor smart-resize/alignment rules, and create the reconstruction/VAE branch around the configured 1024-squared target area with its required alignment. Do not crop, pad, stretch, or force different sources to one common size unless a separately named XENO-only preprocessing mode is requested and quality-tested.
6. For `size=auto`, derive output aspect and aligned dimensions from the last ordered source image, matching the pinned official pipeline. An explicit output size uses the generation request's `reject` or `round_down` alignment policy; that policy does not resize source-conditioning images.

### 14.2 Conditioning sequence

The adapter shall match the pinned `QwenImageEditPlusPipeline`:

- process textual and image inputs through the Qwen2-VL processor/text encoder path;
- VAE-encode source images for reconstruction conditioning;
- construct every model-specific image embedding, attention mask, latent, and position identifier in reference order;
- honor the pinned `zero_cond_t` configuration;
- preserve multi-image order and count in prompt/cache identity; and
- use the same FlowMatch Euler update contract as the official edit pipeline.

No projection file is guessed from a common name. Every required vision/projection tensor is declared by role and validated against the transformer config.

### 14.3 Conditional masked-inpaint gate

Before accepting `mask` for any Qwen bundle:

1. Pin and execute the official `QwenImageInpaintPipeline` reference with the exact Qwen-Image-2512 revision.
2. Prove that its component configs and transformer channel contract are compatible; do not assume compatibility because generation loads.
3. Add a distinct manifest capability and model profile, such as `qwen-image-2512-inpaint-q4_k_m`, only after BF16 and quantized component checkpoints pass.
4. Convert OpenAI alpha masks and grayscale masks to an internal F32 mask where `1.0` means repaint and `0.0` means preserve, matching the official white-repaint/black-preserve convention.
5. Reject an all-zero mask as a no-op unless `x_xeno.allow_noop` is true.
6. Route by the caller-selected model/capability. Do not transparently replace an Edit-2511 request with a different model.
7. Until this gate passes, return `unsupported_parameter` when `mask` is supplied to the Qwen-Image-Edit-2511 profile.

Simple post-generation compositing may later be exposed as an explicitly named image-processing option, but it does not qualify as native masked inpainting and is not part of this specification.

### 14.4 Edit-specific correctness gates

Fixtures must cover:

- single-image semantic edit;
- two- and three-image conditioning order;
- text rendering replacement;
- identity-preserving portrait edit;
- differing source dimensions, independent semantic/VAE branch sizes, and explicit output-alignment policies; and
- cancellation during source encoding and denoising.

Source identity, text correctness, and multi-image attribution are measured separately from generation quality. If `image.inpaint` is admitted, add localized replacement, transparent-input mask compatibility, and mask-leakage fixtures to that profile's independent gate.

## 15. Format, quantization, and support policy

### 15.1 Advertised tiers

| Tier | Component format | CPU | CUDA | Initial status |
|---|---|---:|---:|---|
| BF16 | Official SafeTensors bundle | Required when RAM is sufficient | Required on a sufficiently large device or through explicit staging | Numerical reference |
| Q8_0 | Validated GGUF components plus declared sidecars | Required | Required | First quantized correctness tier |
| Q6_K | Validated GGUF components plus declared sidecars | Required | Required | Quality tier |
| Q5_K_M | Mixed per-tensor GGUF policy | Required | Required | 24 GiB high-quality candidate |
| Q4_K_M | Mixed per-tensor GGUF policy | Required | Required | Initial RTX 4090 product tier |
| Q3/Q2 | Community/experimental | Best effort only after a separate manifest is added | Best effort | Not advertised until quality admission |

`Q5_K_M` and `Q4_K_M` are bundle labels. Individual tensors may intentionally remain BF16/F16, Q8_0, Q6_K, Q5_K, or Q4_K according to the pinned quantization map. The loader trusts each tensor's encoded type and verifies the manifest's expected map; it does not force every tensor into the label's nominal type.

### 15.2 Complete-bundle support rule

XENO RT may state "Qwen-Image-2512 Q4_K_M supported" only when one pinned bundle passes end to end. Loading the transformer while delegating the text encoder or VAE to Python does not qualify. Support declarations are keyed by:

- model family and exact revision;
- component digests;
- capability;
- quantization tier;
- CPU/CUDA backend;
- minimum XENO bundle-schema and runtime versions; and
- tested placement policy.

### 15.3 SafeTensors changes

Refactor the existing `SafeTensorShard`, `SafeTensorInfo`, index validation, and mmap ownership behind a reusable public `SafeTensorStore`; do not implement a second parser or tensor map. `HfModelBundle` continues to compose that store with the unchanged causal-LM `HfModelConfig`, while image components compose it with a new `HfComponentConfig`. The generic store shall:

- open the exact single-file or index/shard filenames declared by the component manifest, including both `model.safetensors(.index.json)` and `diffusion_pytorch_model.safetensors(.index.json)`, rather than hardcoding the causal-LM constants or guessing among directory files;
- validate sharded indices and duplicate names;
- expose arbitrary-rank shape metadata and checked derived contiguous strides (SafeTensors does not encode arbitrary external strides);
- preserve BF16/F16/F32 without conversion until an execution backend chooses a layout;
- reject overlapping, out-of-range, or inconsistent tensor slices; and
- parse component-local `config.json` without requiring LLM-only fields.

Existing text SafeTensors behavior and errors remain unchanged.

### 15.4 GGUF changes

The GGUF path shall add model-family adapters and arbitrary component roles without weakening validation. It must test:

- tensor-name mapping for official/community conversion conventions;
- matrix orientation and rank;
- quantized block alignment;
- mixed precision first/last or sensitive layers;
- metadata identifying architecture and source revision; and
- exact dequantization samples against the converter/reference implementation.

Unknown architecture names, missing metadata required for a mapping, or unsupported quantized tensor roles fail before execution.

## 16. Device placement, memory, and scheduling

### 16.1 RTX 4090 placement target

The initial 24 GiB plan is sequential component residency:

1. Map all artifacts into host memory without eagerly duplicating them.
2. Admit and run the text encoder on CUDA if it fits; otherwise run it on CPU according to the immutable plan.
3. Retain only the resulting prompt embeddings and reusable tokenizer state, then release or evict text-encoder device allocations.
4. Upload and keep the quantized MMDiT resident for the complete denoising loop.
5. Keep latents, embeddings, rotary data, and stable scratch on device.
6. After the final step, release denoiser scratch and, when necessary, transformer residency.
7. Load/run the VAE decoder with slicing or tiling and encode the image on CPU.

For the 4090 gate, the xeno-owned allocation cap is the minimum of 22 GiB, the normal `XRT_GPU_MEMORY_FRACTION`/`XRT_GPU_RESERVED_MB` budget, and `device_total - stable_observed_non_xeno_baseline - 2 GiB`. Admission fails if that calculation underflows or the immutable plan exceeds it. Exact XENO allocation peaks are paired with device-wide telemetry and zero-OOM evidence; the document does not pretend CUDA/display allocations are owned by the XENO arena.

### 16.2 Allocation classes

Extend resource telemetry with image-specific classes:

- image component weights by role;
- prompt embeddings;
- input/source encodings;
- latent state;
- denoiser persistent scratch;
- denoiser transient scratch;
- VAE tile/slice scratch;
- preview scratch; and
- output staging.

Every device allocation must be leased through the central manager or wrapped by a tracked arena whose owner reports into it. Exact transient peak tracking replaces sampled estimates for the production gate.

Adding allocation classes requires updating `GpuAllocationClass::COUNT`, index/name conversion, fixed-size accounting arrays, exhaustive matches, serialization, and status tests together. New class detail appears only on XENO runtime/benchmark surfaces; it must not change `/v1/models`, chat, or completion schemas.

### 16.3 Backend semantics

| Backend request | Required behavior |
|---|---|
| `cpu` | Use CPU for every component. Return an admission error when estimated host RAM exceeds the configured budget. |
| `cuda` | Use CUDA for supported compute and the declared offload policy. Fail before execution if the plan cannot fit or a required CUDA kernel is unsupported. Never silently switch the whole job to CPU. |
| `auto` | Select a supported CUDA plan when it fits; otherwise choose an explicit CPU plan and report the decision before execution. |

`offload=none|sequential|balanced|cpu` is a XENO extension. `none` requires full residency. `sequential` uses the component phases above. `balanced` may place specified layers/components on CPU only after transfer measurements prove it does not move weights inside each step. `cpu` is equivalent to the CPU backend.

### 16.4 Mixed text and image workloads

Default policy is `text-priority`:

- active text weights/KV are never evicted by an image admission;
- an image job waits, uses remaining GPU budget, or uses CPU under `auto`;
- explicit `cuda` returns a bounded insufficient-memory error instead of destabilizing text;
- the queue is FIFO within priority and has configurable maximum depth; and
- one image job runs per device by default.

An operator may select `fair` or `image-exclusive` through startup configuration, not a public request field. `image-exclusive` may unload idle text weights only through the existing controlled runtime transition and must never interrupt an active request.

The server queue dispatches admitted work to a bounded image worker pool. Model loading, CPU execution, image codecs, large hashes, and blocking filesystem operations never run directly in an Axum handler or on Tokio core executor threads. The async boundary bridges disconnect/deadline cancellation into `ImageCancellation`, and worker completion returns through bounded channels. CPU worker concurrency and Rayon thread use are coordinated so nested pools cannot oversubscribe the machine without an explicit operator setting.

### 16.5 Hot-path requirements

- No component weight transfer occurs between denoising steps.
- Prompt embeddings remain resident or move once before step 0.
- Rotary tables and scheduler scalars are cached by admitted shape.
- Stable scratch buffers are reused across steps.
- CUDA graphs may be added only after eager correctness and stable pointers are proven; graph failure falls back to eager CUDA, not CPU.
- Flash Attention or fused attention is admitted only by same-shape numerical and performance tests.
- Preview decode is rate-limited and disabled by default because it can dominate VAE and transfer cost.

## 17. HTTP and CLI interface

### 17.1 OpenAI-compatible generation

The field and event lists below were audited against OpenAI OpenAPI version 2.3.0, the official API reference, and generated SDK types on 2026-07-22. Phase 0 pins the server OpenAPI contract, exact SDK commits, and raw transport fixtures separately because SDK convenience types can lag a server request form and this external contract can change.

Add `POST /v1/images/generations`. The implementation shall recognize the current standard fields even when a local Qwen profile cannot honor every model-specific option:

| Field | XENO behavior |
|---|---|
| `prompt` | Required non-empty string, bounded by the selected bundle's processor limit |
| `model` | Required unless exactly one default `image.generate` model is configured |
| `n` | Defaults to 1; local safety cap is 4 even though the public API permits higher values for some models |
| `size` | Accept `auto` or model-valid `WIDTHxHEIGHT`; validate divisibility, aspect, edge, and pixel limits without silent rounding |
| `quality` | Recognize `auto`, `low`, `medium`, `high`, legacy `standard`, and legacy `hd`; map them through versioned manifest presets |
| `output_format` | Support `png`, `jpeg`, and `webp` |
| `output_compression` | Support 0-100 for JPEG/WebP; reject it for PNG |
| `background` | Accept `auto` and `opaque`; return `unsupported_value` for `transparent` until a validated alpha-producing path exists |
| `response_format` | Accept legacy `b64_json`; support `url` only with a configured output store |
| `stream` | Synchronous `false` is implemented first; production streaming follows the official SSE event schema in Phase 5 |
| `partial_images` | Accept 0-3 only with `stream=true`; produce deterministic preview decodes at recorded step boundaries |
| `moderation` | Recognize but return `unsupported_parameter` unless an operator has configured a local moderation implementation; never pretend moderation ran |
| `style` | Recognize but return `unsupported_parameter` for Qwen profiles because it is a DALL-E-specific control |
| `user` | Accept for compatibility, keep out of logs/metrics, and do not persist by default |

The first functional slice is synchronous and non-streaming. Before the endpoint is described as an OpenAI-compatible implementation for an advertised local Qwen profile, `stream=true` shall emit the official `image_generation.partial_image` and `image_generation.completed` event shapes pinned in Phase 0. Compatibility means the pinned transport, recognized fields, response/error envelope, and documented model-specific unsupported errors; it does not claim that Qwen implements every behavior of every OpenAI-hosted image model. A client that requests streaming before that gate receives a clear `unsupported_parameter` error; the server never returns a JSON body under an SSE content type. Raw-wire fixtures, rather than a hand-built event array, define framing and termination.

Request and response schemas are distinct compatibility surfaces. For the pinned synchronous `ImagesResponse`, XENO resolves `quality` to `low`, `medium`, or `high` and `background` to `opaque` or `transparent`; it never echoes request-only `auto`, `standard`, or `hd`. It includes optional `size` only when the effective dimensions are representable by the pinned response enum and otherwise omits it without changing the generated dimensions. Because the pinned completed SSE event requires these fields, `stream=true` is rejected for a quality, background, or size combination that cannot be represented exactly. Default/`auto`, legacy `standard`/`hd`, and nonstandard-size fixtures must be parsed by the pinned official SDK, not merely compared as untyped JSON.

Generation and edit completed events also require honest usage. XENO derives prompt text tokens, edit input-image tokens, output-image tokens, details, and totals from the versioned accounting recorded by `ImageBatchResult`; non-empty work must not be represented by an all-zero object, and `total_tokens` must equal `input_tokens + output_tokens`. Until this metering exists for a profile, streaming remains explicitly unsupported for that profile even if its event framing tests pass.

`quality` presets are stored in the bundle manifest so changing them changes the bundle/reproducibility identity. The initial Qwen-Image-2512 generation proposal is `low=20`, `medium/standard=35`, and `high/hd/auto=50` denoising steps, subject to the Phase 0/quality audit. The Edit-2511 profile pins its own tested presets instead of inheriting generation values. An explicit `x_xeno.steps` overrides the preset and is echoed only on the XENO metadata surface.

Example:

```json
{
  "model": "qwen-image-2512-q4_k_m",
  "prompt": "A product photograph of a cobalt mechanical keyboard",
  "n": 1,
  "size": "1024x1024",
  "quality": "high",
  "response_format": "b64_json",
  "x_xeno": {
    "seed": 42,
    "steps": 50,
    "true_cfg_scale": 4.0,
    "offload": "sequential"
  }
}
```

The response preserves the OpenAI image envelope:

```json
{
  "created": 1784650000,
  "output_format": "png",
  "quality": "high",
  "size": "1024x1024",
  "data": [
    { "b64_json": "..." }
  ]
}
```

`response_format=url` is available only when an operator configures a bounded output store and expiry policy. Otherwise it returns an OpenAI-style `unsupported_value` error rather than exposing a local file path. The compatibility default TTL is 60 minutes. The store writes beneath one canonical operator-selected root using create-new plus atomic rename, assigns an independent 256-bit random capability token, stores only a keyed digest of that token, and returns an HTTP(S) content URL containing no filename, prompt, model path, or sequential ID. Reads require either the originating principal or constant-time validation of the unexpired capability token, set the exact MIME type plus `X-Content-Type-Options: nosniff` and private/no-store caching, and enforce byte/range limits. Per-principal/global byte quotas, TTL deletion, cancellation cleanup, startup cleanup, and path-containment tests are mandatory; callers never choose an output path.

### 17.2 OpenAI-compatible edits

Add `POST /v1/images/edits` with content-type dispatch for both request contracts in the current pinned OpenAI server schema. The generated Python/JavaScript convenience APIs may remain file-oriented, so their multipart output is one fixture source, not evidence that the JSON server contract does not exist.

- Multipart accepts one or more ordered image file parts and an optional `mask` file part. It shall accept both repeated `image` and `image[]` array encodings because the audited OpenAPI description names `image` while official curl/client examples emit `image[]`; either encoding preserves wire order and mixed spellings are covered by a deterministic duplicate/order test.
- JSON accepts an ordered, non-empty `images` array and an optional `mask` reference. Every reference is a closed object containing exactly one of `image_url` or `file_id`. Bounded base64 `data:` URLs may be decoded locally. HTTPS references and `file_id` values execute only through separately configured, bounded resolvers; without the applicable resolver they return a stable `unsupported_parameter` naming the reference field. `file://`, arbitrary local paths, and implicit network fetches are always rejected.
- Parse only fields admitted by the selected pinned transport schema. The current JSON form includes `prompt`, `model`, `n`, `size`, `quality`, `output_format`, `output_compression`, `background`, `input_fidelity`, `partial_images`, `stream`, `moderation`, and `user`; the multipart form retains its binary `image`/`mask` fields and legacy `response_format`. `moderation` follows the same honest unsupported behavior as generation wherever the pinned schema exposes it.
- For the multipart legacy field, `response_format=b64_json` is accepted; URL output requires the configured output store. GPT Image-style JSON responses are base64 by default and do not require a `response_format` request property.
- Multipart may include an optional JSON text field `x_xeno`; JSON may include the same strict object property. Malformed JSON or unknown XENO members fail validation.
- File parts and decoded data URLs are streamed or decoded through per-item and aggregate byte counters into bounded memory or restricted temporary files. Field count, header bytes, JSON depth/item count, and duplicate scalar fields are bounded; temporary files follow the cleanup rules in Section 18.

The selected profile caps ordered source images at three even though some OpenAI-hosted models accept more. `input_fidelity` is recognized but remains `unsupported_parameter` until a Qwen-specific, quality-tested mapping exists. `mask` is accepted only when the selected profile advertises `image.inpaint`; Edit-2511 returns `unsupported_parameter` as defined in Section 14.3. Transparent background follows the generation behavior.

Streaming edits use the official `image_edit.partial_image` and `image_edit.completed` SSE event shapes after the Phase 5 conformance gate. At the 2026-07-22 audited contract, the typed `completed` event is terminal for both generation and edit streams; XENO shall not append `data: [DONE]` unless a later pinned official raw-wire fixture requires it. An implementation that accepts only multipart may be released only as an explicitly documented multipart compatibility subset, not as full compatibility with the current edit transport.

### 17.3 XENO extensions

The optional `x_xeno` object may contain:

- `seed`;
- `negative_prompt`;
- `steps`;
- `true_cfg_scale`;
- `offload`;
- `resize_policy`;
- `preview_interval_steps`; and
- `allow_noop` for edits.

Unknown `x_xeno` fields return a validation error so misspellings do not silently alter expensive jobs. Internal allocation budgets, model paths, arbitrary component URLs, and process-wide scheduling policy are never request fields.

### 17.4 Asynchronous job extension

After synchronous compatibility passes, add:

- `POST /v1/xeno/image/jobs/generations` with the standard generation JSON fields except `stream`;
- `POST /v1/xeno/image/jobs/edits` with the standard edit multipart fields except `stream`;
- `GET /v1/xeno/image/jobs/{id}`;
- `GET /v1/xeno/image/jobs/{id}/events` for bounded SSE progress;
- `GET /v1/xeno/image/jobs/{id}/result` for the authorized terminal image envelope; and
- `DELETE /v1/xeno/image/jobs/{id}` for cancellation.

Job IDs contain at least 128 bits from the operating-system CSPRNG, are never derived from prompts, and expire. Status includes state, queue position, component phase, output index, step/total steps, percentage, elapsed time, and safe metrics. It excludes prompts, input digests, and image bytes. The result route returns the normal image envelope only after success and is subject to the same aggregate response/output-store limits as the standard endpoint. Unknown and non-owner IDs both return indistinguishable 404 responses; an authorized owner may receive 410 while an expiry tombstone remains, without exposing prior content.

The authenticated API-key identity is the job principal; the OpenAI `user` request field is metadata and never grants access. Loopback-only unauthenticated mode uses one explicit local principal. Submit, status, event, result, and cancellation handlers must enforce the same principal (or a separately configured administrative principal) with constant-time credential comparison. Stored job ownership uses a keyed credential identifier, never the API key itself.

Both job-creation routes accept an optional bounded ASCII `Idempotency-Key` (maximum 255 bytes). For the configured TTL, the server stores `(principal, HMAC(key), keyed canonical-request digest) -> job_id`, never the raw key: the same key and request returns the same job, while the same key with a different request returns 409. Generation canonicalization uses normalized scalar/JSON fields; edit canonicalization uses normalized scalar fields plus ordered keyed digests of uploaded bytes, independent of multipart boundaries. This map is process-local in version 1, so no cross-restart idempotency is promised. The server does not automatically retry a job after model execution begins. Queue depth and queued/running counts are bounded globally and per principal.

States are `queued`, `admitting`, `running`, `succeeded`, `failed`, `cancelled`, and `expired`. Startup configuration supplies finite maximum queue wait, execution time, progress-buffer capacity, result bytes, and retention TTL. Cancellation is checked at the boundaries in IMG-F07; queued cancellation is immediate, running cancellation is cooperative, repeated cancellation is idempotent, and terminal success/failure is not rewritten. Slow SSE consumers have intermediate previews dropped or are disconnected according to a recorded policy; they cannot block denoising or grow an unbounded channel, and a still-connected stream never drops its terminal event.

The standard OpenAI endpoints may use the same internal queue while holding the HTTP request open; they do not return the XENO job schema. Client disconnect cancels that internal job by default. Server shutdown stops admission, cancels queued work, gives running work a bounded grace period, then signals cancellation and waits for resource leases/temp files to release before process exit.

### 17.5 Runtime discovery and loading

`POST /v1/runtime/load` retains all current text fields (`model_path`, `hf_repo`, `hf_file`, `mmproj_path`, external-provider fields, and `backend`) and gains additive `modality` and catalog `model` fields:

```json
{
  "modality": "image",
  "model": "qwen-image-2512-q4_k_m",
  "backend": "auto"
}
```

Omitted `modality` remains `text`; existing load payloads therefore retain their routing. For `modality=image`, the HTTP form accepts an already installed catalog model ID and never a local component path or arbitrary repository URL; a missing bundle returns `model_not_installed` and does not trigger network I/O. Operators install through `xrt download --bundle`, or use `xrt image ... --model-path`/local-import library APIs for unmanaged local image bundles. `external-openai` is rejected for `modality=image` and its text transition leaves image entries untouched.

The current bodyless `POST /v1/runtime/unload` remains valid and means text unload. The route additionally accepts an optional JSON body `{ "modality": "image", "model": "...", "force": false }`; missing `modality` still means text. For `modality=image`, `model` is required unless exactly one image model is loaded, and ambiguity returns 400 without unloading anything. Each admitted job pins an `Arc` to a specific runtime generation and bundle digest. Image unload first removes that generation from new-request routing and returns a draining state; `force=false` lets pinned work finish, while administrative `force=true` signals cancellation and releases the runtime only after all leases exit. Reload atomically publishes a new generation while the old generation drains. These image lifecycle rules do not change the current bodyless text-unload response contract.

`/v1/models` lists loaded image model IDs using its existing basic schema. Add `GET /v1/runtime/models` for capability, component digest, runtime generation, quantization, backend, estimated/actual memory, and `loading|ready|draining|failed` state. No acceleration metadata is added to the OpenAI model object.

When `external-openai` text routing is active, `/v1/models` must merge the validated upstream basic model list with locally loaded image model objects instead of returning early from the current text proxy. Preserve upstream order and append local image objects in stable ID order; endpoint modality disambiguates an unlikely duplicate ID, and `/v1/runtime/models` remains the authoritative capability view. An upstream model-list failure retains the current proxy failure behavior; locally loaded image runtimes remain discoverable through `/v1/runtime/models`.

### 17.6 Error envelope

Failures from the standard image endpoints and new `/v1/xeno/image/*` endpoints use:

```json
{
  "error": {
    "message": "human-readable and redacted",
    "type": "invalid_request_error",
    "param": "size",
    "code": "image_dimensions_unsupported"
  }
}
```

Admission conflicts use HTTP 409, queue saturation uses 429, request/input validation uses 400 or 413, missing models use 404, and internal execution failures use 500. Cancellation maps to 499 when the server can emit it or to a closed connection; job status records `cancelled`.

Existing non-image `/v1/runtime/*`, completion, chat, and proxy error bodies retain their current contracts. Adding image modality dispatch to the shared runtime load/unload routes must not convert legacy text errors or the bodyless text-unload response into a new envelope. New image-modality errors on those XENO runtime routes follow the existing runtime-route contract; OpenAI envelopes are required only at the image compatibility/API boundary described above.

### 17.7 CLI

Add nested commands without changing existing text commands:

```text
xrt image generate --model qwen-image-2512-q4_k_m --prompt "..." --size 1024x1024 --steps 50 --seed 42 --output out.png
xrt image edit --model qwen-image-edit-2511-q4_k_m --image source.png --prompt "..." --output edited.png
xrt image edit --model qwen-image-2512-inpaint-q4_k_m --image source.png --mask mask.png --prompt "..." --output inpainted.png
xrt image bench --model qwen-image-2512-q4_k_m --suite qwen-image-release --backend cuda --json
xrt image import --path ./local-diffusers-or-bundle --install
xrt download --bundle qwen-image-2512-q4_k_m
```

Output files use create-new or explicit overwrite semantics and an atomic temporary file in the destination directory. `--metadata sidecar.json` writes safe reproducibility metadata. Default console output reports progress and metrics but not base64 image data.

## 18. Security, privacy, and licensing

### 18.1 Input security

- Default maximum request body: 128 MiB.
- Default maximum encoded input image: 32 MiB each.
- Default maximum decoded inputs: three images, 4096 pixels per side, and 16,777,216 pixels per image.
- Decode codecs in bounded mode and reject decompression bombs before full allocation.
- Apply a manifest-specific output dimension/area limit no higher than the server policy.
- Default maximum output count: four; default maximum steps: 100.
- Encoded bytes per output, aggregate encoded bytes, aggregate base64 response bytes, multipart field/header count, and temporary-disk bytes have finite defaults checked during admission and encoding.
- Reject NaN, infinity, negative dimensions, overflow, and inconsistent MIME signatures.
- Standard image generation/edit requests and XENO image-job requests never accept a local model path, output path, `file://` URL, or arbitrary component URL. The existing XENO administrative text `POST /v1/runtime/load` path fields remain a separate compatibility surface and are governed by the authenticated/canonical-root rules below.
- The JSON edit form may decode bounded base64 `data:` image URLs locally. Remote HTTPS image resolution is out of scope for the initial Qwen profile, and `file_id` resolution is disabled unless a bounded operator-selected object/file store exists. Disabled reference kinds are recognized and rejected explicitly rather than being fetched implicitly or misparsed as multipart.
- If remote HTTPS image resolution is later added, it requires an HTTPS-only allowlist or equally restrictive egress policy, post-resolution public-IP checks for every DNS result, redirect revalidation, response/type/byte/time limits, decompression limits, no ambient credentials, and SSRF tests covering DNS rebinding, IPv4/IPv6 private ranges, redirects, and proxy behavior.

### 18.2 Model supply chain

- Pin source revision, file size, and SHA-256 for every component.
- Install through the whole-directory staging/verification/atomic-publish transaction in Section 12.3; a collection of individually renamed files is not an atomic bundle.
- Never execute model repository code.
- Validate SafeTensors/GGUF bounds before mapping or uploading.
- Preserve and surface upstream license/NOTICE files when present, plus the pinned license-evidence record; never fabricate a missing NOTICE file.
- Treat community conversions as untrusted input until their tensor maps and numerical fixtures pass.
- Record converter name/version/commit in the manifest when known.

### 18.3 Privacy

- Do not log prompts, negative prompts, input image bytes, masks, base64, output bytes, or user identifiers by default.
- Redact bearer credentials, idempotency keys, job ownership identifiers, and output capability tokens from application and access logs.
- Metrics labels use model/bundle/capability, never prompt-derived text.
- Job records are memory-resident and expire in version 1. Persisting job metadata or request content requires a later encryption/retention specification; only the explicitly configured output store may persist result bytes for its bounded TTL.
- Prompt-cache keys follow Section 13.2: no raw or unkeyed prompt-derived key is retained, cross-principal reuse is disabled by default, and cache contents disappear at process exit.
- After the Section 14.1 color-profile conversion-or-rejection decision, encoded outputs strip source EXIF/ICC/comment metadata; removing an ICC tag from unconverted non-sRGB pixels is prohibited. Prompts and local paths are never embedded. Reproducibility metadata is opt-in through the CLI sidecar or XENO metadata response, not hidden in image files.
- Temporary decoded or encoded files are avoided. When required, they use restricted permissions and are removed on success, error, cancellation, and startup cleanup.
- No telemetry leaves the machine unless the operator configures an exporter.
- Native Qwen execution has no implicit content-moderation service. Documentation and runtime capability status shall say so plainly; an omitted `moderation` field must never be reported as if a safety classifier ran.

### 18.4 Network exposure

Image generation is a high-cost denial-of-service surface. Loopback remains the safe default. `XRT_API_KEY` and `XRT_ALLOW_UNAUTHENTICATED_IMAGE_API` are new settings introduced by this project. When bound to a non-loopback address, startup with image routes enabled fails unless `XRT_API_KEY` is set or the operator explicitly sets `XRT_ALLOW_UNAUTHENTICATED_IMAGE_API=1`. Bearer credentials are compared in constant time, excluded from logs/errors, and never accepted through query strings. This image-route restriction is additive and does not silently change current OpenAI text endpoint authentication behavior.

With a configured key, all `/v1/xeno/image/*`, `/v1/images/generations`, `/v1/images/edits`, non-capability output reads, and image runtime-management operations share the bearer gate. On non-loopback binds, the unauthenticated acknowledgement enables only the synchronous standard generation/edit routes; asynchronous job/result routes and runtime-management routes remain disabled until a key exists because they require a stable owner/administrator principal. After this feature is enabled, both mutating `/v1/runtime/load` and `/v1/runtime/unload` operations are administrative and require the API key on non-loopback binds, including their legacy text forms. Legacy local text model paths are canonicalized and must remain beneath configured model roots; image HTTP load accepts only catalog IDs as specified in Section 17.5. The acknowledgement does not authorize arbitrary local paths. Migration documentation must call out the new non-loopback administrative protection.

### 18.5 License gates

- Qwen official bundles may be cataloged under Apache-2.0 with pinned license evidence and every upstream notice file that is actually present.
- Community conversions require source and redistribution review before XENO hosts copies; users may still import local artifacts.
- Ideogram 4 product support remains blocked until a repository decision records terms for redistribution, commercial runtime use, and customer-facing/API use.
- Qwen-Image-3.0 support remains blocked until its own public license and weights are audited; family branding does not imply the 2512 license carries forward.

## 19. Observability

### 19.1 Per-job timings

Record monotonic durations for:

- queue wait and admission;
- bundle/component load;
- tokenizer/processor;
- text encoding;
- source image/VAE encoding;
- denoiser initialization;
- each denoising step and aggregate steps per second;
- preview generation;
- VAE decode;
- image encode; and
- total seconds per image.

### 19.2 Memory and transfer metrics

Report:

- current and peak tracked host/device bytes by image allocation class;
- sampled device-wide used/free/total bytes with its existing caveat;
- component residency and eviction counts;
- explicit H2D, D2H, and D2D calls/bytes by phase;
- scratch-arena reserved/high-water bytes;
- VAE tile size/count/overlap; and
- OOM/admission rejections.

### 19.3 Scheduler metrics

Report queued, active, completed, failed, and cancelled image jobs; queue wait percentiles; device occupancy; selected workload policy; and maximum observed concurrency. Do not report prompts or input digests that could enable cross-user correlation.

### 19.4 Benchmark output

`xrt image bench --json` writes schema-versioned results under `benchmark-results/image/<date>-<model>-<backend>/`. It includes exact bundle/component digests, git commit and dirty flag, compiler profile, CPU/GPU/driver/CUDA versions, settings, repetition-level timings, percentiles, memory, transfers, output hashes, and quality-suite version.

Image metrics are seconds per image, denoising steps per second, peak RAM/VRAM, cold/warm load time, and time to first preview. They are never reported as tokens per second.

## 20. Validation and benchmark plan

### 20.1 Reference environment

Phase 0 pins:

- exact official model revisions;
- exact Diffusers, Transformers, Safetensors, PyTorch, and CUDA versions used to produce fixtures;
- exact stable-diffusion.cpp commit/build flags;
- exact community GGUF revisions and converter metadata; and
- test hardware/driver identifiers.

Reference scripts live under `reference/image/qwen/` and are never imported by production crates. They emit fixture manifests and hashes. Large model artifacts remain in the model cache or CI secrets-backed runner, not Git.

### 20.2 Test layers

1. **Unit:** manifest canonicalization/golden bytes, hashes, config parsing, shape validation, scheduler math, checked multi-output seed derivation, keyed/partitioned prompt-cache keys, conditional inpaint-mask conversion, EXIF/color-profile conversion-or-rejection, image limits, tiling coordinates, overlap blending, cancellation, and error mapping.
2. **Kernel:** CPU scalar reference versus SIMD/CUDA for linear, quantized linear, convolution, normalization, RoPE, attention, patchify/unpatchify, interpolation, and VAE activations.
3. **Component:** tokenizer/text encoder, transformer block, full transformer sampled outputs, VAE encode/decode, and scheduler checkpoints versus pinned reference tensors.
4. **Synthetic pipeline:** a tiny generated MMDiT/text encoder/VAE fixture completes on every default CPU CI run and exercises generation, edit, cancellation, and OOM cleanup.
5. **Real-model CPU:** bounded low-resolution Q4 and BF16 component smokes on a high-RAM runner.
6. **Real-model CUDA:** Q8 first, then Q6_K, Q5_K_M, and Q4_K_M on an opt-in RTX workflow using the repository's safe, serialized GPU-run pattern.
7. **HTTP/CLI:** pinned OpenAI server-schema and raw-wire fixtures, official-client multipart fixtures, JSON edit reference validation/resolver behavior, multipart ordering/limits, raw SSE termination, errors, b64 and authorized URL output, legacy text load/bodyless unload, modality-aware load/unload, job ownership/idempotency/result/cancel/expiry, and atomic file behavior. Response tests cover omitted defaults, `auto`, legacy `standard`/`hd`, every standard size, nonstandard local sizes, and SDK parsing of both synchronous and streaming outputs; streaming assertions reject fabricated all-zero usage and verify detail/total consistency.
8. **Bundle crash/recovery:** concurrent install, interruption before and after fsync/rename/index update, hash mismatch, stale staging cleanup, offline restart, and proof that no partial bundle becomes discoverable.
9. **Mixed workload:** active text streaming plus queued/running image generation under each workload policy, with one injected per-device manager, runtime replacement/drain, allocation cleanup, and text no-regression assertions.
10. **Shutdown:** queued cancellation, bounded running-job grace, forced cancellation, result/temp cleanup, and zero leaked resource leases across restart.

### 20.3 Determinism gates

- Same model digest, backend, device architecture, driver/compute-runtime versions, build profile, seed, dimensions, steps, and policy must produce the same uncompressed pixel hash across three cold process starts.
- CPU and CUDA need not have identical pixel hashes. Their component outputs must meet recorded tolerances, and their end results must pass the quality non-inferiority suite.
- Changing scheduler config, manifest digest, quantization, resize policy, or RNG schema version must change the reproducibility identity exposed in metadata.

### 20.4 Quantization quality suite

Create a versioned, license-clean suite containing at least:

- 100 general prompt-adherence/composition prompts;
- 40 typography prompts across the languages supported by the target model and available OCR evaluators;
- 30 face/hands/fine-detail prompts;
- 30 style/color prompts;
- 30 single-image edits;
- 20 masked inpaint cases when an `image.inpaint` profile is under admission; and
- 20 multi-image edits; and
- at least 50 designated identity-preservation output pairs, which may reuse edit prompts across pinned seeds.

Use the BF16 official path as the paired reference, feeding BF16 and quantized candidates the identical XENO seed-derived initial latent described in Section 13.3. Record CLIP-style prompt alignment, OCR character/word error rate, DINO-style structural similarity, face identity similarity where applicable, conditional inpaint mask leakage, and blinded human review. Higher prompt-alignment/structural/identity scores are better; lower OCR error, leakage, and severe-defect rates are better. Every evaluator, preprocessing rule, score direction, and category-specific absolute quality floor is version-pinned before quantized results are inspected so a weak BF16 fixture cannot make a broken quant pass by relative comparison.

The admission compiler is a consumer of metric exports, not the evaluator. Phase 0 shall add a reference-only evaluator runner, or exact reproducible containerized commands, that produce the raw export from the corpus without modifying production crates. It pins model artifacts, package versions, preprocessing, device/backend, and an offline execution procedure. The PaddleOCR-VL path must use the full official pipeline rather than the VLM component alone and pin either a dedicated container image by digest or an exact compatible PaddlePaddle plus `paddleocr[doc-parser]` environment and named inference engine. Synthetic compiler tests and a dependency extra alone do not count as an evaluator run.

Admission thresholds relative to BF16 on the same seeds:

| Tier | Prompt-alignment decline | OCR CER increase | Mean structural/identity decline | Human severe-defect rate |
|---|---:|---:|---:|---:|
| Q8_0 | <= 1% relative | <= 2 percentage points | <= 0.01 absolute | <= 2% |
| Q6_K | <= 2% relative | <= 3 percentage points | <= 0.015 absolute | <= 3% |
| Q5_K_M | <= 3% relative | <= 4 percentage points | <= 0.02 absolute | <= 5% |
| Q4_K_M | <= 5% relative | <= 6 percentage points | <= 0.03 absolute | <= 8% |

For the first three metric columns, compute the per-case degradation from its BF16 pair and a one-sided 95% bootstrap upper confidence bound with at least 10,000 deterministic resamples; that upper bound, not only the point estimate, must be within the table limit. Human review uses at least 200 stratified, randomized pairs per tier, three blinded raters, and a prewritten severe-defect rubric; majority vote defines an absolute severe defect in the quantized output, and the one-sided 95% Wilson upper bound must be within the table limit. The paired BF16 image supplies blinded context and must also meet its absolute category floors. Rater disagreement and category counts are retained in the evidence.

No tier passes if any fixed fixture produces repeatable NaN/Inf or a blank image. Gross anatomy corruption, unreadable required text, and edit-identity failures are counted under the pinned rubric; the one-sided 95% upper bound for identity failure must be at most 10% on designated fixtures. An advertised inpaint profile also fails when the pinned leakage metric's one-sided 95% upper bound exceeds 2% of protected pixels at the defined perceptual delta. These thresholds are proposed XENO release policy, not claims from Qwen or another upstream. Maintainers approve and freeze the suite in Phase 0; changing a threshold/evaluator increments the suite version and requires every compared tier, including BF16, to rerun.

### 20.5 Performance workload

The RTX 4090 release comparison uses:

- one pinned Qwen-Image-2512 Q4_K_M complete bundle;
- 1024x1024 output;
- 50 denoising steps;
- true CFG scale 4.0;
- fixed prompt and seed suite;
- concurrency 1;
- previews off;
- warm component cache; and
- 30 measured repetitions per engine after two warmups, with engine order alternated in matched blocks.

Production CUDA support requires:

- zero OOMs;
- exact xeno-owned peak VRAM stays within the IMG-P01 admission cap and the configured reserve is not consumed;
- no weight transfer within the denoising loop;
- the one-sided 95% bootstrap upper confidence bound for the median-time ratio is <= 1.15;
- the one-sided 95% bootstrap upper confidence bound for the P95-time ratio is <= 1.25;
- identical requested output dimensions and step count; and
- Q4_K_M quality gates passing.

Diffusers provides correctness, not the native performance threshold. The comparator must use the same logical model revision, tensor quantization map, dimensions, sampler/scheduler parameters, seed set, and output count; Phase 0 records any unavoidable artifact-layout conversion and proves sampled tensor equivalence. All comparisons run serially on the same machine after checking for leftover GPU processes, fixed power/clock policy where controllable, and thermal steady state. A failed or hung real-model run is stopped and investigated; it is not retried indefinitely or excluded without being reported.

### 20.6 Existing-runtime no-regression gates

- Full existing default CPU tests pass.
- CUDA compile/reproducibility checks pass through the repository's safe scripts.
- OpenAI text schema snapshots pass unchanged.
- Existing controlled text generation benchmarks stay within IMG-P08.
- Background removal output/endpoint tests pass unchanged.
- Model cache operations for existing GGUF aliases remain compatible.

## 21. File-level implementation map

### 21.1 Current crate layout

```text
crates/xrt-image/
  Cargo.toml
  src/lib.rs
  src/backend.rs
  src/bundle.rs
  src/cancellation.rs
  src/error.rs
  src/image_io.rs
  src/memory.rs
  src/metrics.rs
  src/pipeline.rs
  src/request.rs
  src/rng.rs
  src/runtime.rs
  src/synthetic.rs
  src/scheduler/mod.rs
  src/scheduler/flow_match_euler.rs
  src/models/mod.rs
  src/models/qwen_image/mod.rs
  src/models/qwen_image/config.rs
  src/models/qwen_image/edit.rs
  src/models/qwen_image/edit_processor.rs
  src/models/qwen_image/pipeline.rs
  src/models/qwen_image/prompt.rs
  src/models/qwen_image/tensors.rs
  src/models/qwen_image/text_encoder.rs
  src/models/qwen_image/text_encoder_cpu.rs
  src/models/qwen_image/transformer.rs
  src/models/qwen_image/transformer_executor.rs
  src/models/qwen_image/transformer_gguf.rs
  src/models/qwen_image/transformer_safetensors.rs
  src/models/qwen_image/transformer_cuda.rs   # optional CUDA feature
  src/models/qwen_image/vae.rs
  src/models/qwen_image/vae_decoder.rs
  src/models/qwen_image/vision_encoder_cpu.rs
```

Do not create `xrt-text`, `xrt-video`, or `xrt-audio` directories in this project unless they contain a real facade or runtime with tests. Reserve their names in architecture docs and capability identifiers.

### 21.2 Shared crate changes

The rows below describe the required end state and retained ownership boundaries; Section 7.4, not the imperative wording in this table, records what is currently implemented and admitted.

| Area | Required change or retained responsibility |
|---|---|
| Root `Cargo.toml` | Add the `xrt-image` workspace member and only the root-package integration-test wiring actually required |
| `xrt-core` | Reuse the existing shape/stride-aware `TensorView`; add only format-neutral owned-storage/layout helpers that a concrete image operator proves are missing |
| `xrt-safetensors` | Extract the current validated shard/index/mmap machinery behind `SafeTensorStore`, then compose generic component config without changing causal-LM loader behavior |
| `xrt-gguf` | Add image architecture metadata/tensor mapping validation as required by pinned fixtures |
| `xrt-tokenizer` | Add the exact Qwen2.5-VL processor/tokenizer surface required by reference fixtures |
| `xrt-kernels` | Add scalar-first image ops, then measured SIMD paths |
| `xrt-cuda` | Add batched GEMM/attention, image RoPE, convolution/VAE, stable scratch, and optional fused kernels |
| `xrt-runtime` | Add resource-manager injection constructors while preserving existing loaders; expose shared leases/allocation classes and image-safe status aggregation; do not add image state to `Session` |
| `xrt-hub` | Add immutable-revision bundle manifests, whole-directory staging/atomic publish, SHA-256, locks, cleanup, and offline resolution without changing legacy one-file aliases |
| `xrt-openai` | Add reusable image request/response/error schema types while retaining external proxy behavior |
| `xrt-server` | Add an optional `xrt-image` dependency with package-local `image-generation`/CUDA forwarding, `LoadedRuntimeSet`, per-device manager registry, bounded multipart parsing, image routes/job/result queue, ownership/auth/limits, runtime-generation drain semantics, and disconnect/shutdown cancellation |
| `xrt-cli` | Add an optional `xrt-image` dependency with package-local `image-generation`/CUDA forwarding, nested `image` commands, and bundle download support |
| `xrt-capi`/`xrt-python` | Add only after Rust/HTTP APIs stabilize; never make Python binding a runtime dependency |

### 21.3 Tests and evidence

```text
crates/xrt-server/src/image_api.rs             # current inline HTTP/queue/lifecycle tests
crates/xrt-image/tests/cuda_qwen_edit.rs       # current ignored tiny CUDA parity tests
crates/xrt-image/tests/real_qwen_bundle.rs     # current ignored/manual real-model harness
tests/image_bundle.rs                          # current cross-crate bundle coverage
tests/image_pipeline_synthetic.rs              # current cross-crate synthetic coverage
tests/image_resource_admission.rs              # current cross-crate resource coverage
tests/image_qwen_reference.rs                  # planned ignored/manual real model
tests/image_edit_reference.rs                  # planned ignored/manual real model
tests/common/image-quality-suite.json
tests/fixtures/openai/images/
reference/image/qwen/
reference/image/qwen/evaluate_quality_suite.py # current strict quality plan/report compiler
reference/image/qwen/run_quality_evaluators.py # planned reference-only metric-export producer
reference/image/qwen/QUALITY_ADMISSION.md      # current operator contract
scripts/safe-image-reference.ps1               # current bounded reference workflow
scripts/safe-image-cuda-check.ps1              # current bounded compile/tiny parity workflow
scripts/safe-image-cuda-smoke.ps1              # current pinned Q4 bounded real smoke workflow
benchmark-results/image/
```

The list distinguishes current files from planned acceptance-test paths. Cross-crate bundle, synthetic pipeline, and resource-admission tests exist at the workspace root. Server image API tests currently live inline in `crates/xrt-server/src/image_api.rs`; the real-model harness and tiny CUDA parity tests live under `crates/xrt-image/tests/`. The two root-level real-reference test files remain planned. `scripts/safe-image-reference.ps1` provides the bounded Phase 0 oracle/comparator workflow; the image CUDA check and smoke wrappers provide the serialized compile/parity and exact pinned low-resolution real-model workflows. Neither wrapper turns a smoke into production or performance admission.

Every future safe smoke script requires an explicit confirmation flag, bounds runtime, runs only one Cargo/GPU process at a time on Windows, and checks for leftover processes after interruption, matching existing repository practice.

## 22. Phased delivery and rollout

These phases define acceptance order, not inferred completion from file presence. The current evidence-backed state is recorded in Section 7.4; a schema, command, kernel, or ignored test existing does not satisfy an exit gate by itself.

### Phase 0: Provenance, pins, and baselines

1. Pin official model/config revisions and reference library versions.
2. Pin one complete community bundle per quantization tier without redistributing it.
3. Generate reference tensors/images and record hashes.
4. Benchmark Diffusers and stable-diffusion.cpp on the target hardware.
5. Record component sizes, peak CPU RAM/VRAM, and a feasible 4090 placement plan.
6. Recheck the official Qwen-Image-3.0 catalog and update only the availability note.
7. Pin the current OpenAI server request schemas for generation plus multipart and JSON edits, exact official SDK multipart fixtures, and raw generation/edit SSE bytes including terminal behavior. Fixture generators shall not synthesize an undocumented sentinel.
8. Approve and freeze the quality-suite prompts, evaluator versions/directions, absolute floors, statistical rules, and thresholds before running quantized candidates.
9. Pin and execute the reference-only metric-export producer, including a complete runnable OCR pipeline/engine, then prove that its export is accepted by the separately tested admission compiler.

**Exit gate:** reproducible fixture manifest, license record, reference scripts, runnable evaluator/export procedure, and baseline report exist. No model executor is merged before this gate.

### Phase 1: Domain foundation and synthetic pipeline

1. Add `xrt-image` with CPU-safe features and exact request/error contracts.
2. Add generic SafeTensors component access and bundle-manifest validation.
3. Extend `xrt-hub` for atomic bundles.
4. Implement FlowMatch Euler, versioned RNG, image IO limits, and a tiny synthetic pipeline.
5. Add the injected text-runtime resource-manager path and per-device server registry, then extend central resource leases without modifying text `Session` semantics.

**Exit gate:** default CPU CI compiles and runs synthetic generation/edit/cancellation; all existing tests pass.

### Phase 2: Qwen-Image-2512 BF16 reference execution

1. Implement tokenizer/text-encoder parity.
2. Implement MMDiT config/tensor mapping and scalar CPU operators.
3. Implement VAE decode, slicing, and tiling.
4. Complete BF16 CPU reference and CUDA correctness paths.
5. Compare every numerical checkpoint to official Diffusers.

**Exit gate:** bounded real generation completes; component tolerances and same-backend determinism pass. Performance may remain experimental.

### Phase 3: GGUF quantized execution

1. Validate Q8_0 mappings and dequantization first.
2. Add Q6_K, Q5_K_M, and Q4_K_M in that order.
3. Validate complete mixed-format bundles, not isolated transformers.
4. Run the quantization quality suite and reject tiers that fail.

**Exit gate:** all four admitted tiers execute end to end on CPU and CUDA; Q4_K_M passes quality gates.

### Phase 4: RTX 4090 memory and performance

1. Implement immutable sequential component placement.
2. Keep the MMDiT resident during denoising.
3. Add stable scratch arenas, optimized batched kernels, VAE tiling, and measured attention fusion.
4. Add CUDA graph replay only for proven stable shapes.
5. Record exact transient peak device use and transfers.

**Exit gate:** the Section 20.5 performance and memory gates pass. Until then the CUDA route remains `experimental` and is never described as competitive.

### Phase 5: HTTP, CLI, and operations

1. Add OpenAI-compatible generation schemas and execution. Dispatch edits by content type, add bounded multipart uploads and JSON `images`/`mask` references, and return honest reference-resolver or model-capability errors; successful Qwen-Image-Edit-2511 execution remains gated on Phase 6.
2. Add runtime model discovery/load/unload without changing text defaults.
3. Add nested CLI commands and benchmark JSON.
4. Add bounded synchronous queue, disconnect cancellation, API key gate, and schema tests.
5. Add official image-generation SSE event conformance for `stream=true` and bounded partial previews. Verify raw framing and terminal behavior without assuming a text-chat `[DONE]` sentinel; separately verify response-domain normalization for defaults, legacy qualities, and local sizes plus real, internally consistent usage metering. The edit SSE encoder may be fixture-tested with the synthetic pipeline, but it is not advertised for Qwen-Image-Edit-2511 until Phase 6 succeeds.
6. Add the optional XENO async job/progress/result endpoints with ownership, idempotency, deadlines, quotas, unload/shutdown semantics, and TTL cleanup.
7. Add the optional bounded output store before advertising `response_format=url`.

**Exit gate:** generation request/response/raw-SSE conformance; edit multipart and JSON parsing, reference-resolution behavior, raw-SSE fixtures, and unsupported-capability errors; queue/cancel tests; security limits; CLI tests; and mixed text/image tests pass. If HTTPS/`file_id` resolvers are deferred, the release is explicitly documented as a multipart-plus-local-data-URL subset rather than full current edit-transport compatibility. A successful real edit response is not a Phase 5 claim.

### Phase 6: Qwen-Image-Edit-2511

1. Add processor, source VAE encode, vision/projection conditioning, and `zero_cond_t` behavior.
2. Add ordered multi-image conditioning and semantic edit preprocessing.
3. Run edit-specific component, quality, identity, and attribution gates.
4. Independently test Qwen-Image-2512 with the official inpaint pipeline; add mask normalization and mask-leakage gates only if the profile passes.
5. Repeat 4090 admission/performance tests for each complete advertised edit or inpaint bundle.
6. Enable successful Qwen-Image-Edit-2511 execution through the existing edit route for every advertised request transport and pass the pinned edit response/raw-SSE conformance fixtures against the real runtime.

**Exit gate:** single and multi-image Edit-2511 behavior passes every advertised backend/tier gate. Masked inpainting is either independently admitted with evidence or explicitly unsupported; it does not block honest Edit-2511 support.

### Phase 7: Production admission and documentation

1. Keep `image-generation` gated until all required evidence is checked in.
2. Document model installation, supported bundle matrix, API, CLI, memory policies, and limitations.
3. Add upgrade/rollback behavior for bundle schema version 1.
4. Re-run every default and RTX gate from a clean checkout.
5. Regenerate every checked-in CUDA PTX file with the exact pinned workflow image and command, require byte-for-byte equality, and retain the command/toolchain hashes in evidence.

**Exit gate:** maintainers approve the support matrix and default-on policy. Release remains a separate, explicitly requested workflow.

### Phase 8: Later adapters

Qwen-Image-3.0 work starts only when all are public and pinned:

1. official weights and complete component tree;
2. architecture/configuration and tensor names;
3. license and redistribution terms;
4. official reference inference;
5. tokenizer/processor assets; and
6. a reproducible oracle output.

The adapter audit then classifies differences from the current `QwenImagePipeline`; no aliasing 3.0 to 2512 is permitted.

Ideogram 4 starts only after legal approval. It should be a separate adapter using shared flow, text-encoder, VAE, and bundle primitives where configurations truly match.

### Rollback

- The server feature flag can remove image routes without affecting text routes.
- Loaded image runtimes and their queues are independent of text `Runtime` ownership.
- Bundle schema additions are additive; existing one-file GGUF cache resolution remains.
- A failing optimized CUDA kernel falls back to a validated eager CUDA implementation for explicit CUDA, not to CPU.
- A failing new adapter is removed from the catalog/support matrix without deleting user artifacts.

## 23. Risks, alternatives, and open questions

### 23.1 Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Current kernels are token-shape optimized | Image path is correct but very slow | Profile batched shapes; add scalar reference then cuBLAS/custom kernels behind parity gates |
| A complete Qwen bundle exceeds 24 GiB | OOM on the primary target | Sequential component residency, quantized text encoder, VAE tiling, 22 GiB admission cap |
| Community GGUF layouts drift | Wrong images or unsafe reads | Pin revisions/hashes, validate metadata/shapes, component parity, reject unknown mappings |
| Quantization damages text/faces/details | Low-quality product output | Per-tier quality gates; Q3/Q2 remain experimental; sensitive tensors may stay higher precision |
| Diffusers changes reference behavior | Fixtures become ambiguous | Pin exact versions and keep scheduler/config in bundle digest |
| CPU fallback is prohibitively slow | Poor usability without CUDA | Maintain correctness, expose estimates clearly, optimize measured CPU hot paths, never hide the cost |
| Image work destabilizes chat | Broken platform consumers | One resource manager, text-priority default, no implicit text eviction, mixed-workload tests |
| API extensions leak into OpenAI schemas | SDK incompatibility | `x_xeno` namespace, separate runtime endpoints, existing schema snapshots |
| Image inputs create SSRF or decompression attacks | Host compromise/DoS | No URL/file fetch, bounded codecs, pixel/body/queue limits, auth gate off loopback |
| Async status/results or prompt caches cross tenant boundaries | Prompt/output disclosure | Principal-scoped authorization, keyed cache/request digests, finite TTL, no `user`-field trust |
| Multi-file install is interrupted | Partial or mixed-revision execution | Same-filesystem staging, per-digest lock, verify/fsync, whole-directory rename, atomic index, recovery tests |
| Qwen-Image-3.0 architecture differs materially | Adapter assumptions fail | Configuration-driven components and a formal publication gate; no premature support claim |
| Ideogram terms conflict with product use | Legal/product risk | Explicit legal gate and no default bundling |
| Four modality crates duplicate infrastructure | Maintenance drift | Capability boundaries share core/runtime/hub/kernels; create crates only with real code |

### 23.2 Alternatives rejected

1. **Wrap ComfyUI or Diffusers:** fastest prototype, but violates native Rust/offline dependency goals and duplicates scheduling/resource ownership.
2. **Shell out to stable-diffusion.cpp:** useful as a benchmark bridge, but it would make XENO RT a wrapper rather than its own engine.
3. **Put image generation in `xrt-vision`:** conflates discriminative task inference with stateful generative pipelines and makes future modality routing unclear.
4. **Generalize `CausalLmBackend` into an untyped `ModelBackend`:** erases useful state contracts and risks text regressions. Share resources, not fake semantics.
5. **Create `xrt-text`, `xrt-video`, and `xrt-audio` immediately:** produces churn and empty abstractions before real shared interfaces are known.
6. **Support only Q4_K_M:** reaches a 4090 quickly but removes the BF16/Q8 numerical ladder needed to debug correctness.
7. **Claim all GGUF quants:** filename-level support cannot guarantee component completeness or acceptable image quality.

### 23.3 Open questions requiring owner decisions

1. Which exact community conversion revisions may XENO document, mirror, or redistribute? Local import can proceed before mirroring is approved.
2. Which high-memory runner will host the BF16 CPU and large-GPU numerical oracle jobs?
3. Should the XENO managed-model catalog live entirely in `xrt-hub` or be generated from the platform's central model catalog? The runtime bundle schema remains the same either way.
4. Does the first public image release require asynchronous jobs, or may synchronous OpenAI endpoints ship first with disconnect cancellation?
5. Should non-loopback inbound authentication become a shared server feature for all modalities rather than the image-only enablement gate proposed here?
6. Which output-object store, if any, should back `response_format=url`? Until selected, base64 is the supported local response.
7. Can the Qwen2.5-VL encoder share a format-neutral block executor with current text models without exposing causal-session state? The default is to share kernels and tensor resolution, not `Runtime`.
8. What quality-suite content can be redistributed in the public repository? Prompts can be checked in; reference images may need generated, license-clean fixtures or digest-only storage.
9. When the first stable image runtime exists, should the follow-up `xrt-text` facade be a crate or only a public module/re-export? This is intentionally not a blocker here.
10. Does the first public edit release include the SSRF-safe HTTPS and operator-backed `file_id` resolvers required for full current JSON edit-transport compatibility, or explicitly advertise only multipart uploads plus bounded local data URLs?
11. Will quality evaluation use a digest-pinned PaddleOCR-VL container or a separately pinned local PaddlePaddle/`paddleocr[doc-parser]` environment? The choice must be made before real metric exports are treated as reproducible evidence.

## 24. Acceptance criteria and references

### 24.1 Definition of done for this project

The native image project is complete only when all statements below are true:

1. `xrt-image` exists as a CPU-safe Rust crate and owns a real image pipeline, not a wrapper process.
2. Qwen-Image-2512 generates valid images end to end from complete pinned BF16, Q8_0, Q6_K, Q5_K_M, and Q4_K_M bundles on every backend advertised for that bundle.
3. Qwen-Image-Edit-2511 performs single-image and multi-image semantic edits for the same advertised tiers; any advertised Qwen-Image-2512 inpaint profile independently passes native mask gates.
4. CPU fallback works without compiling or loading CUDA and passes a real bounded Q4 smoke on a high-RAM runner.
5. The RTX 4090 Q4_K_M generation workload stays within the IMG-P01 reserve-aware cap, performs no per-step weight transfers, records zero OOMs, and passes the native comparator threshold.
6. Numerical checkpoints match the pinned official reference within recorded tolerances.
7. Every advertised quantization passes its quality threshold; Q3/Q2 are not advertised by implication.
8. Same-backend deterministic runs produce identical uncompressed pixel hashes under the defined reproducibility identity.
9. OpenAI-compatible generation JSON plus both current edit request forms, responses, errors, and raw SSE streams pass pinned server-schema and official-client fixtures. Request-only enum values are normalized or omitted according to the separately pinned response schema, nonstandard local sizes have an explicit synchronous/streaming policy, and every streamed usage object is measured and internally consistent rather than fabricated. Enabled JSON reference kinds resolve within their security contract; disabled resolver kinds return their documented stable error. Multipart-only support is not counted as full current edit-transport compatibility, and no undocumented `[DONE]` sentinel is emitted.
10. Existing text OpenAI schemas, GGUF loading, CPU fallback, CUDA paths, CLI commands, and `xrt-vision` background removal pass unchanged; every checked-in CUDA PTX file is byte-reproducible from its retained source with the exact pinned workflow toolchain and flags.
11. Server-managed text and image runtimes on one CUDA device use the same injected GPU manager; it accounts for every image allocation, and mixed workloads neither leak nor corrupt resources.
12. Model bundles are immutable-revision/hash/license-evidence pinned, published by an atomic whole-directory transaction, recover cleanly from interruption, and are usable offline.
13. Security tests cover input bombs, size overflow, path/URL rejection, prompt-cache isolation, job ownership/idempotency, output-store containment/expiry, queue exhaustion, cancellation/shutdown cleanup, administrative local-path controls, and non-loopback enablement.
14. Benchmark and quality evidence is checked into `benchmark-results/image/` with reproducible commands.
15. Documentation clearly states that `xrt-text`, `xrt-image`, `xrt-video`, and `xrt-audio` are the long-term domains, while only `xrt-image` is newly implemented by this project and `xrt-vision` remains task inference.
16. Qwen-Image-3.0 and Ideogram 4 remain unadvertised until their separate publication/license gates pass.
17. Runtime replacement/unload pins in-flight jobs to one bundle generation, blocks new routing to draining generations, and preserves the current bodyless text-unload behavior.
18. If asynchronous jobs or URL outputs are enabled, authorized result retrieval, finite deadlines/quotas/TTLs, slow-consumer behavior, and startup cleanup pass their lifecycle tests before those capabilities are advertised.

### 24.2 Assumptions

- Official Qwen-Image-2512 and Qwen-Image-Edit-2511 artifacts remain available under Apache-2.0 at the pinned revisions.
- The primary optimization machine is an RTX 4090 with 24 GiB VRAM, but support is not hardcoded to that GPU.
- A high-RAM CPU or larger-GPU runner is available for BF16 reference validation.
- Community GGUF artifacts are interoperability inputs, not proof of correctness or permission to redistribute.
- The existing local hybrid-MoE changes are user-owned and will be preserved while implementing this specification.
- No Qwen-Image-3.0 local support can be scheduled until the Phase 8 prerequisites are observed directly.

### 24.3 Authoritative references

- [Qwen-Image-2512 model](https://huggingface.co/Qwen/Qwen-Image-2512)
- [Qwen-Image-2512 model index at audited revision](https://huggingface.co/Qwen/Qwen-Image-2512/blob/25468b98e3276ca6700de15c6628e51b7de54a26/model_index.json)
- [Qwen-Image-2512 transformer configuration at audited revision](https://huggingface.co/Qwen/Qwen-Image-2512/blob/25468b98e3276ca6700de15c6628e51b7de54a26/transformer/config.json)
- [Qwen-Image-2512 scheduler configuration at audited revision](https://huggingface.co/Qwen/Qwen-Image-2512/blob/25468b98e3276ca6700de15c6628e51b7de54a26/scheduler/scheduler_config.json)
- [Qwen-Image-2512 VAE configuration at audited revision](https://huggingface.co/Qwen/Qwen-Image-2512/blob/25468b98e3276ca6700de15c6628e51b7de54a26/vae/config.json)
- [Qwen-Image-Edit-2511 model](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
- [Qwen-Image-Edit-2511 model index at audited revision](https://huggingface.co/Qwen/Qwen-Image-Edit-2511/blob/6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9/model_index.json)
- [Diffusers Qwen Image pipelines](https://huggingface.co/docs/diffusers/api/pipelines/qwenimage)
- [Qwen Image paper](https://arxiv.org/abs/2508.02324)
- [Qwen-Image-3.0 announcement](https://qwen.ai/blog?id=qwen-image-3.0)
- [Official Qwen Image model collection](https://huggingface.co/collections/Qwen/qwen-image)
- [OpenAI image generation API](https://developers.openai.com/api/reference/resources/images/methods/generate)
- [OpenAI image edit API](https://developers.openai.com/api/reference/resources/images/methods/edit)
- [OpenAI image generation streaming events](https://developers.openai.com/api/reference/resources/images/generation-streaming-events)
- [OpenAI image edit streaming events](https://developers.openai.com/api/reference/resources/images/edit-streaming-events)
- [OpenAI Python image edit request type](https://github.com/openai/openai-python/blob/main/src/openai/types/image_edit_params.py)
- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
- [stable-diffusion.cpp Qwen Image guide](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/qwen_image.md)
- [Unsloth Qwen-Image-2512 GGUF](https://huggingface.co/unsloth/Qwen-Image-2512-GGUF)
- [Unsloth Qwen-Image-Edit-2511 GGUF](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF)
- [Unsloth Edit-2511 `index_timestep_zero` compatibility-marker discussion](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF/discussions/4)
- [ComfyUI Qwen Image Edit compatibility detection at the audited source revision](https://github.com/Comfy-Org/ComfyUI/blob/e4c61d75555036fa28b6bb34e5fd67b007c9f391/comfy/model_detection.py#L622)
- [city96 Qwen-Image GGUF](https://huggingface.co/city96/Qwen-Image-gguf)
- [QuantStack Qwen-Image-Edit GGUF](https://huggingface.co/QuantStack/Qwen-Image-Edit-GGUF)
- [Ideogram 4 reference implementation](https://github.com/ideogram-oss/ideogram4)
- [Ideogram 4 technical details](https://ideogram.ai/blog/ideogram-4.0/)
- [Ideogram weights licensing](https://ideogram.ai/licensing/)
