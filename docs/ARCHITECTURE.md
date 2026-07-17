# Architecture

## Design Goals

xeno-rt is an offline-first inference runtime with four non-negotiable
invariants:

1. The documented OpenAI-compatible HTTP surface remains compatible.
2. GGUF remains a first-class model format.
3. CPU inference remains available when CUDA is absent or unsuitable.
4. Performance changes are accompanied by benchmark evidence.

The implementation favors explicit backend contracts and validated model
metadata over silent conversion or fallback that could produce incorrect
logits.

## System Layers

```text
Clients
  |
  +-- xrt-cli
  +-- xrt-server ------------- xrt-openai (optional external proxy)
           |
       xrt-runtime
           |
     CausalLmBackend
       /       \
    CPU         CUDA
     |            |
 xrt-models   xrt-models + xrt-cuda
       \          /
        model sources
   xrt-gguf / xrt-safetensors
            |
       xrt-tokenizer
            |
         xrt-core
```

### Interface layer

- `xrt-cli` owns command parsing, interactive chat, downloads, and benchmark
  reporting.
- `xrt-server` owns HTTP request validation, OpenAI-shaped responses, SSE
  streaming, runtime lifecycle routes, and image-task routing.
- `xrt-openai` is a guarded client for forwarding to another
  OpenAI-compatible runtime. Remote hosts require an explicit opt-in.

### Runtime layer

`xrt-runtime` coordinates:

- backend selection (`auto`, `cpu`, `cuda-resident`);
- model and tokenizer lifetime;
- per-request sessions and sampling;
- F32 and quantized KV cache modes;
- policy-aware prompt spans;
- prefix-cache lookup and eviction;
- bounded request admission and decode batching;
- GPU budget, transfer, allocation, pool, and graph status.

The `CausalLmBackend` trait is the behavioral boundary between orchestration
and execution. CPU and CUDA implementations must produce the same logical
model contract even when storage and scheduling differ.

### Model layer

`xrt-models` resolves model metadata and tensor names before execution. It owns
the CPU forward pass and the architecture-specific geometry reused by CUDA.
LoRA and mmproj vision support also live here.

`xrt-gguf` validates GGUF headers, metadata, tensor dimensions, offsets, and
byte ranges before exposing memory-mapped tensor data. `xrt-safetensors`
validates supported Hugging Face directory layouts for the CUDA-only paths.

### Compute layer

- `xrt-kernels` provides CPU normalization, RoPE, attention helpers,
  activations, quantization, and matvec kernels.
- `xrt-cuda` owns persistent device buffers, quantized resident matrices,
  execution streams, CUDA Graphs, paged KV storage, kernel launch validation,
  and resource counters.

CUDA kernels use checked-in PTX. Runtime model decode does not depend on a
third-party inference process.

## Model Load Flow

### GGUF

1. `Runtime::load_with_backend` opens and validates the GGUF file.
2. The tokenizer and model configuration are derived from GGUF metadata.
3. `cpu` creates a CPU backend over memory-mapped weights.
4. `cuda` validates support and the VRAM budget before resident upload.
5. `auto` attempts compatible CUDA initialization in a CUDA build and falls
   back to CPU if the CUDA path cannot be created.

Explicit `cuda` requests fail explicitly; they do not silently use CPU.

### Hugging Face SafeTensors directory

SafeTensors directories currently require a CUDA-enabled build and an explicit
or automatic CUDA path. CPU SafeTensors decode is not implemented. The loader
validates the model type and quantization metadata before device upload.

## Request Flow

1. A CLI or HTTP request is normalized into `GenerateRequest`.
2. The scheduler admits or rejects the sequence according to configured
   active/queued limits.
3. The tokenizer applies the model chat template where applicable.
4. Prefix-cache state may satisfy a validated prompt prefix.
5. Prefill runs in configured chunks; decode turns can be continuously batched.
6. The active backend updates its KV state and returns logits.
7. The sampler chooses the next token and the interface emits text or SSE.
8. Runtime status exposes cache, scheduler, backend, and GPU observations.

## Cache Architecture

The runtime supports `f32`, `q8`, `kq4_vq8`, and `agent_adaptive` KV modes.
The selected mode is part of cache identity; incompatible snapshots are not
reused across modes or model namespaces.

Prompt policies can pin or protect system, developer, tool-schema, and
tool-result spans. Prefix-cache entries are bounded by entry count, byte size,
and minimum token count.

## Failure Policy

- Invalid model metadata fails before execution.
- Unsupported architecture/source/backend combinations return an explicit
  error.
- Explicit CUDA never silently becomes CPU.
- `auto` may fall back to CPU only for GGUF and logs the CUDA failure.
- External proxy targets are loopback-only unless remote access is explicitly
  enabled.

## Source Ownership

| Area | Primary path |
|---|---|
| Shared types and errors | `crates/xrt-core` |
| GGUF parsing | `crates/xrt-gguf` |
| SafeTensors/HF loading | `crates/xrt-safetensors` |
| Tokenization/templates | `crates/xrt-tokenizer` |
| CPU kernels | `crates/xrt-kernels` |
| CUDA primitives | `crates/xrt-cuda` |
| Model execution | `crates/xrt-models` |
| Sessions/backends/caches | `crates/xrt-runtime` |
| HTTP API | `crates/xrt-server` |

Large CUDA and backend modules will be decomposed after the v0.2.0 checkpoint
in behavior-preserving changes. The staged plan is recorded in the
[repository hardening specification](REPOSITORY_HARDENING_SPEC.md).
