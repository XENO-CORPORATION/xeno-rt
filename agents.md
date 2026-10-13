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

## 🏗️ The platform hierarchy & naming — build under the lock (LOCKED 2026-09-17)

Two ladders, five rungs, meeting at the App — `../XENO FULL-STACK HIERARCHY.md` (master),
`../XENO FRONT-END HIERARCHY.md` (Elements → Components → Blocks → Templates → Apps — rung 3 is BLOCKS; a panel is the slot a block sits in),
`../XENO BACK-END HIERARCHY.md` (Primitives → Capabilities → Nodes → Blueprints → Apps).
One naming rule on every rung — `../XENO PACKAGE NAMING - STANDARD.md`:
`@xenosystem/<rung>/<family>` → one named export per unit. Gate: `node ../scripts/check-package-naming.mjs`.

**This repo is:** **BACK rung 1 — a Primitive.** A platform service or runtime the back ladder is built on (inference, processing, device actions, the OIDC origin and ledger, hosted runs, the registry). It exposes capabilities (rung 2) through ONE code path; nothing on the front ladder imports it, and proprietary consumers reach AGPL primitives out of process (root CLAUDE.md §5b).

🔴 **Never create a temporary name, package, path or layer "for now"** (root `CLAUDE.md`
§BUILD UNDER THE LOCK FROM DAY ONE). Concretely:
- depend DOWN only — never on a rung above, never on `xeno-apps`;
- never re-implement a lower rung here — extract DOWN to its repo and mount it;
- never publish a per-unit package on a ladder rung (`@xenosystem/block-<x>`/`panel-<x>`, `component-<x>`, `node-<x>`) — a unit is a named export in a family subpath;
- never commit a `file:` dependency to a `.tgz` in a Temp directory or an absolute path — publish, wait for npm's read replica, depend on the range;
- seen before used — it renders or runs standalone in `xeno-apps` before this repo relies on it.
