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

## 🔴 XENO Hub is a CONSUMER of this repo — the contract, and what it is still waiting for

**Recorded 2026-08-24 by the orchestrator, after auditing Hub's Models tab end to end.** Hub's
Models tab is a complete, correctly-wired UI (868 lines, all seven IPC handlers implemented) sitting
in front of a runtime that **is never shipped**. For every installed user:

```
localRuntimeStart() -> {"success":false,
  "error":"xrt-server was not found. Install the local runtime from the Models tab..."}
```

Hub resolved the binary from `../xeno-rt/target/release/` — a sibling Rust build tree — and its
`package.json` has zero references to `xrt-server`. So the tab works on a developer machine with
this repo cloned and built, and **nowhere else**. That is the "built, tested, unreachable" shape
this ecosystem keeps rediscovering.

### What Hub has already fixed on its side (done, on `main`)

- Binary resolution is now **platform-correct**, derived from this repo's own `release.yml`
  (`xrt-server` / `xrt-server.exe`, plus `xrt-cli`). It previously hardcoded `.exe` in all five
  candidate paths with no `process.platform` branch, so on Linux — where Hub has shipped an
  AppImage since 0.9.1 — the runtime could never resolve at all.
- **macOS is REFUSED, not searched.** `release.yml` publishes linux-x86_64 and windows-x86_64 only.
  Hunting four paths for a file that cannot exist reports "not found", which reads as a broken
  install rather than an unsupported platform. If a macOS build ever ships, Hub must be told.

### The delivery design — LOCKED by size, not by preference

| Payload | Route | Why |
|---|---|---|
| **CPU runtime (~24 MB)** | Hub's signed-package pipeline, bundled copy as floor | small enough to verify in memory; gives "always latest xeno-rt" with no Hub release |
| **CUDA payload (~870 MB)** | **this repo's own artifact downloader** | far too large for Hub's pipeline |
| **Model weights (2.7–21.7 GB)** | same | needs resume; Hub has no business transferring these |

🔴 **The constraint that decides it: Hub's signed-package pipeline downloads to MEMORY, verifies,
and only then writes to disk.** A ~900 MB CUDA package is the wrong shape for it entirely. The
split is therefore by transfer characteristics, not by kind — Hub does small verified code, this
repo does large resumable artifacts.

### The six things this repo owes, from its own audit

1. **Catalog CONSUMER, not server.** A catalog served by the runtime is unreachable exactly when it
   is needed most — before the runtime is installed. Source of truth is a static signed JSON on
   `updates.xenostudio.ai`; both Hub and this runtime read it.
   ⚠️ `/v1/runtime/models` is NOT that catalog — it is `image_api::runtime_models`, registered only
   on the image-generation feature build. A default build does not have it.
2. **Checksum + resume for GGUF.** `ModelHub` fetches from HF with progress but **no sha256 and no
   resume**; `BundleInstaller` has the integrity model but serves xrt-image. Extend the latter to
   cover GGUF from R2. Resume on a 21.7 GB Ornith download is not optional.
3. **Capability/compat endpoint, and MoE config inferred from GGUF metadata.** The loader already
   reads `expert_count` / `expert_used_count` before any config is applied — it should select
   hybrid placement itself. **Do not push `XRT_MOE_ACCELERATION=hybrid` into Hub**: model-specific
   knowledge in the host is exactly what this whole design removes.
4. **Typed error instead of the cudarc panic** on missing CUDA DLLs.
5. **Drain-and-exit contract, and a VERSIONED on-disk layout with a test.**
6. **Dry-run fit check** reusing the existing CUDA preflight.

### Two HARD ordering constraints — not backlog, gates

- 🔴 **Auto-update cannot ship before the on-disk layout is a stated contract with a test.** Models
  live under `~/.cache/xrt/models` by repo/filename and nothing versions that. "Installed by N loads
  on N+1" is currently an observation, not a promise. Auto-updating first builds a mechanism whose
  failure mode is silently orphaning a 21.7 GB download the user waited an hour for.
- 🔴 **The typed error lands before any GPU delivery path exists.** cudarc loads driver/nvrtc/cublas
  eagerly and panics if absent, so a GPU runtime installed without its ~870 MB of DLLs beside it
  gives the user a process crash with no message.

### Measured facts — do not re-derive from assumptions

- CPU build: `cargo build --release`, **no `--features cuda`**, ~12 MB per binary, two binaries.
- CUDA is a **payload, not a binary**: `cublasLt64_12.dll` 668.7 MB, `cublas64_12.dll` 102.5 MB,
  `nvrtc64_120_0.dll` 89.8 MB, `nvrtc-builtins64_129.dll` 7.2 MB. `nvcuda.dll` ships with the driver.
- **DirectML is not involved anywhere.** An earlier orchestrator note wrongly assumed it was.
- Ornith 1.5 35B A3B is **21.7 GB on disk and ran in an 11.31 GB device peak** — expert placement
  keeps cold experts in host RAM. File size is off by **2× in the direction that scares users off a
  card that fits**, which is precisely why the fit check belongs here and not in Hub's arithmetic.

### ⚠️ State of the model work as of 2026-08-24

- **Qwen 3.8 4B** and **Ornith 1.5 35B A3B** are verified end to end with retained evidence.
- **Qwen 3.8 9B is NOT releasable** — downloaded and hashed, never executed. By this repo's own rule
  it does not go in the catalog.
- 🔴 **Neither model is in R2.** `models/local-chat/` contains only the 3.5 line. Hub cannot offer
  3.8 until the GGUFs are uploaded with their sha256 — a catalog entry pointing at a missing object
  is a download button that 404s.
- 🔴 **`feat/qwen38-mtp` has NO UPSTREAM and is 30 commits ahead of `origin/main`.** Nothing is
  pushed, so `clean_checkout_ci` has never run against any of it. A runtime that only builds on the
  machine that made it is the exact shape this workspace keeps discovering after release — and 30
  commits of verified model work currently exist on one disk.
- **No performance number is publishable.** Three identical runs measured 64.19 / 69.14 / 73.25
  tok/s on one machine — ±14%. Publish capability, not speed.

### 🔴 Publishing weights to R2 — three traps, all hit on 2026-08-24

**`rclone size` reading 0 during a multipart upload is EXPECTED, not a stall.** S3 shows nothing
at the key until `CompleteMultipartUpload`; a 21.7 GB transfer reads as 0 bytes for its entire
duration. Misreading that as a hung upload is what caused a relaunch, and the relaunch is what
caused everything below. Check for a live `rclone.exe` and its command line — never infer progress
from the destination.

**`pgrep` in Git Bash cannot see Windows processes.** `pgrep -f upload-models.mjs` matches nothing
even while the process is running, so an `until ! pgrep …` wait returns instantly and reads as
"finished". Use `tasklist /FI "PID eq <pid>"`, or `Get-CimInstance Win32_Process`, and capture the
PID at launch. Two of these false "completions" produced **two concurrent rclone uploads writing
the same file to the same key**.

⚠️ That race was survivable, and the reason is worth knowing rather than relying on: each
`CompleteMultipartUpload` is atomic and last-writer-wins, so the object is whole either way — but
**which** writer won is unknowable without reading the object back. On a `models/…` key, where the
filename carries no version and R2 has no object versioning, that is exactly why the checksum must
be read back **from R2** and never restated from the local file.

**Killing strays: by PID and command line, NEVER by image name.** `node.exe` is what Claude Code
itself runs on; `Stop-Process -Name node` ends the session and every other node job on the machine.
Confirm the command line identifies your own work (`Get-CimInstance Win32_Process -Filter
"Name='rclone.exe'"` prints it) before stopping anything.

**Orphaned multiparts bill until cleaned.** Abandoned uploads leave parts that never complete.
Clean by AGE, not UploadId, so a live transfer cannot be hit by mistake — and dry-run first:
`rclone backend cleanup r2:xeno-hub-releases -o max-age=30m --dry-run`.
