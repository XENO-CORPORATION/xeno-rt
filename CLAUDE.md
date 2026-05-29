# CLAUDE.md — XENO RT Engineering Standards

## ⚠️ Before You Debug ANYTHING — Read This First

**`../docs/engineering-learnings.md`** is the canonical cross-ecosystem bug log. When a user reports a weird symptom, **`grep -i "<keyword>" ../docs/engineering-learnings.md` BEFORE you investigate**. We share React 19, Electron, Zustand, and canvas patterns across every xeno-* repo, so a bug fixed in one is usually latent in all of them. Re-discovering a documented fix costs hours; grepping costs seconds. When you fix a NEW bug whose symptom isn't there, append it. That file is how we stop repeating mistakes across agents.


## You Are Working On

**xeno-rt** — high-performance LLM inference runtime written in pure Rust. Alternative to llama.cpp. OpenAI-compatible API server.

## Critical Context

Part of a 16+ repo ecosystem. Read `../XENO CORPORATION - Full Ecosystem Report.md`.

```
YOUR REPO: xeno-rt (Layer 2 — Compute & AI)
    ↑ consumed by: xeno-agent-sdk (local LLM provider for agent reasoning)
    ↑ consumed by: xeno-agent-cli (local inference for terminal agent)
    ↑ will be consumed by: xeno-pixel, xeno-motion, xeno-sound (agent reasoning)
    ↑ alternative to: api.xenostudio.ai cloud LLM (for offline/privacy use)
```

## WHY THIS REPO MATTERS

xeno-rt enables the entire XENO platform to work **fully offline**. When embedded:
- Agents in Pixel/Motion/Sound can reason about creative tasks without internet
- Users with privacy requirements never send data to the cloud
- No per-token costs for local inference

## ABSOLUTE RULES

1. **OpenAI API compatibility is sacred.** `/v1/chat/completions` and `/v1/models` must match the OpenAI spec exactly. The agent SDK switches between xeno-rt and cloud LLM — the API must be identical.
2. **Never regress performance.** Benchmark every change against the previous version.
3. **GGUF format support must be maintained.** This is how users get models (HuggingFace).
4. **CPU fallback must always work.** CUDA is optional. The runtime must function on any x86_64/aarch64 machine.

## Code Quality

- Pure Rust. Minimize unsafe (document every usage).
- Criterion benchmarks for all hot paths.
- Memory safety: no buffer overflows, no use-after-free, no data races.
- CUDA code isolated behind feature flags.
