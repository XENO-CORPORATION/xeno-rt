# CLAUDE.md — XENO RT Engineering Standards

## ⚠️ Before You Debug ANYTHING — Read This First

**`../docs/engineering-learnings.md`** is the canonical cross-ecosystem bug log. When a user reports a weird symptom, **`grep -i "<keyword>" ../docs/engineering-learnings.md` BEFORE you investigate**. We share React 19, Electron, Zustand, and canvas patterns across every xeno-* repo, so a bug fixed in one is usually latent in all of them. Re-discovering a documented fix costs hours; grepping costs seconds. When you fix a NEW bug whose symptom isn't there, append it. That file is how we stop repeating mistakes across agents.


## You Are Working On

**xeno-rt** — XENO's unified local AI inference runtime. It hosts, exposes, and runs text, image, future video, and future audio models through shared Rust infrastructure, CPU fallback, optional CUDA, model bundles, and stable APIs.

Read [`docs/RUNTIME_DOMAINS.md`](docs/RUNTIME_DOMAINS.md) before changing public scope. `xrt-text`, `xrt-image`, `xrt-video`, and `xrt-audio` are capability boundaries within one runtime. The working text implementation has not been renamed; `xrt-image` exists but is experimental; video/audio remain planned until real tested adapters exist.

## Critical Context

Part of a 16+ repo ecosystem. Read `../XENO CORPORATION - Full Ecosystem Report.md`.

```
YOUR REPO: xeno-rt (Layer 2 — Compute & AI)
    ↑ consumed by: xeno-agent-sdk / xeno-agent-cli (`xrt-text`)
    ↑ consumed by: creative apps (`xrt-image`, `xrt-vision`, future video/audio)
    ↑ exposes: local model capabilities through shared CLI/server/bindings
    ↑ alternative to: cloud inference for offline/privacy use
```

## WHY THIS REPO MATTERS

xeno-rt enables the XENO platform to run AI models **fully offline**. When embedded:
- Agents can reason about creative tasks without internet
- Creative apps can invoke local generative and task models through one runtime
- Users with privacy requirements never send data to the cloud
- Local inference avoids provider usage costs

Consumer apps own canvases, timelines, masks, tracks, editing workflows, and project UX. XENO RT owns model loading, inference graphs, quantization, CPU/CUDA execution, scheduling, memory, model caching, APIs, and telemetry. `xeno-lib` owns non-AI media processing and format I/O.

## ABSOLUTE RULES

1. **Existing OpenAI API compatibility is sacred.** Text and enabled image contracts must remain compatible. Add XENO-only controls under additive namespaces/endpoints.
2. **Never regress performance.** Benchmark every change against the previous version.
3. **GGUF format support must be maintained.** This is how users get models (HuggingFace).
4. **CPU fallback must always work for every advertised CPU capability.** CUDA is optional.
5. **Do not put product workflows in the runtime.** Expose inference capabilities; let XENO apps orchestrate them.
6. **Do not create empty modality crates or advertise unadmitted support.** A model/backend/tier needs correctness, quality, memory, performance, API, and reliability evidence.

## Code Quality

- Pure Rust. Minimize unsafe (document every usage).
- Criterion benchmarks for all hot paths.
- Memory safety: no buffer overflows, no use-after-free, no data races.
- CUDA code isolated behind feature flags.
- Tests and metrics must match the modality: tokens for text, images/steps/previews for image, and separately defined video/audio gates when those domains land.
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
