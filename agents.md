# agents.md — XENO RT (for Codex CLI and AI agents)

## Identity

You are working on **xeno-rt**, XENO's unified local AI inference runtime. It hosts, exposes, and runs text, image, future video, and future audio models through shared Rust infrastructure, CPU fallback, optional CUDA, model bundles, and stable APIs.

Read `docs/RUNTIME_DOMAINS.md` before changing architecture or public scope. The public capability boundaries are:

- `xrt-text` — language and conversational model inference;
- `xrt-image` — image generation and model-level image conditioning/edit inference;
- `xrt-video` — future video generation/transformation inference; and
- `xrt-audio` — future speech, music, and audio inference.

These are shared-runtime domains, not four unrelated engines. Current text code remains in `xrt-runtime`/`xrt-models`; `xrt-image` is real but experimental; do not create empty video/audio crates.

## Ecosystem

Read `../XENO CORPORATION - Full Ecosystem Report.md`. XENO RT enables the platform to run AI models locally and offline. Consumer apps own canvases, timelines, editing workflows, and project UX; XENO RT owns model execution, hosting, scheduling, hardware use, and inference APIs. `xeno-lib` owns non-AI media processing and format I/O.

## Safety

1. **NEVER break an existing OpenAI-compatible contract.** The Agent SDK and creative apps switch between local and cloud providers. XENO-only capabilities must be additive.
2. **NEVER remove GGUF format support.** This is how users get models.
3. **CPU fallback must always work for every advertised CPU capability.** CUDA is optional.
4. **Benchmark every change with modality-appropriate metrics.** Never admit a regression without explicit evidence and policy.
5. **Never move product UI or editing workflows into XENO RT.** Expose model capabilities; let consumer apps orchestrate them.
6. **Never claim a domain, model, backend, or quantization is supported from file presence or a smoke test alone.** Use the admission gates in `docs/RUNTIME_DOMAINS.md`.

## Stack: Rust 1.76+, cudarc (CUDA), rayon, GGUF, SafeTensors, ONNX task integration, tokenizers, image codecs
## API: OpenAI-compatible text/image surfaces plus additive `/v1/runtime/*` capability and lifecycle APIs
## Consumers: xeno-agent-sdk, xeno-agent-cli, XENO creative apps, automation, and future local-model consumers
## Releasing — read `release-guide/` in full before any release

This repo ships the portable `release-guide/` playbook (canonical copy lives in `xeno-platform`). Before cutting ANY release — a new version (installer or CLI) OR a landing/docs change — read every file in `release-guide/` in order, starting with `release-guide/README.md`. Releases run from the **xeno-platform** repo. Do not improvise release commands — or just say "release <product>" to invoke the `xeno-product-release` skill (installed globally).
