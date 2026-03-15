# agents.md — XENO RT (for Codex CLI and AI agents)

## Identity

You are working on **xeno-rt**, a high-performance LLM inference runtime (llama.cpp alternative). Pure Rust, GGUF format, CUDA optional, OpenAI-compatible API.

## Ecosystem

Read `../XENO CORPORATION - Full Ecosystem Report.md`. xeno-rt enables the entire platform to work offline. The agent SDK uses xeno-rt as an alternative to cloud LLM.

## Safety

1. **NEVER break OpenAI API compatibility.** Agent SDK switches between xeno-rt and cloud — API must be identical.
2. **NEVER remove GGUF format support.** This is how users get models.
3. **CPU fallback must always work.** CUDA is optional.
4. **Benchmark every change.** Never regress performance.

## Stack: Rust 1.76+, cudarc (CUDA), rayon, GGUF parser, BPE tokenizer
## API: OpenAI-compatible /v1/chat/completions, /v1/models
## Consumers: xeno-agent-sdk, xeno-agent-cli, future: all creative apps
