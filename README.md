# xeno-rt

[![CI](https://github.com/XENO-CORPORATION/xeno-rt/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/XENO-CORPORATION/xeno-rt/actions/workflows/ci.yml)
[![CUDA validation](https://github.com/XENO-CORPORATION/xeno-rt/actions/workflows/cuda.yml/badge.svg)](https://github.com/XENO-CORPORATION/xeno-rt/actions/workflows/cuda.yml)
[![Security audit](https://github.com/XENO-CORPORATION/xeno-rt/actions/workflows/audit.yml/badge.svg?branch=main)](https://github.com/XENO-CORPORATION/xeno-rt/actions/workflows/audit.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**xeno-rt** is XENO's unified, headless Rust runtime for hosting, exposing, and
running local AI models. Its capability domains are `xrt-text`, `xrt-image`,
`xrt-video`, and `xrt-audio`, backed by shared formats, kernels, CPU/CUDA
resource management, model caching, telemetry, CLI, bindings, and stable HTTP
APIs. Read [Runtime Domains](docs/RUNTIME_DOMAINS.md) for the canonical product
and ownership boundary.

**v0.2.0** is the first stable GitHub release checkpoint. CPU execution is the
default supported path. CUDA execution is a real native runtime path and
remains beta; it is not a wrapper around another inference engine. The project
does not claim performance parity with other runtimes without a reproducible
benchmark. The released checkpoint is text-focused. Native `xrt-image` code is
present in the development tree but remains experimental and unreleased;
`xrt-video` and `xrt-audio` are planned boundaries with no crate or supported
adapter yet.

## Why xeno-rt

- **CPU always available:** the default build requires no CUDA installation.
- **Native CUDA execution:** model weights, KV cache, and decode scratch can
  remain GPU-resident in CUDA-enabled builds.
- **GGUF first:** memory-mapped GGUF loading, metadata validation, tokenizer
  loading, and quantized weight execution.
- **Multiple model sources:** GGUF on CPU or CUDA, plus selected Hugging Face
  SafeTensors layouts on CUDA.
- **OpenAI-compatible serving:** completions, chat completions, streaming,
  model discovery, and explicit runtime lifecycle endpoints.
- **Production-oriented runtime controls:** paged and quantized KV modes,
  prefix caching, bounded scheduling, decode batching, CUDA Graph support, and
  resource telemetry.
- **Observable benchmarks:** structured JSON reports include active backend,
  timing, memory, transfer, allocation, scheduler, and cache data.
- **Native image foundations:** feature-gated Qwen Image generation/edit,
  component bundles, CPU-safe execution, optional CUDA, and image APIs without
  launching Python, Diffusers, ComfyUI, or another inference process.
- **Evidence-gated support:** parsing or a smoke test does not advertise a
  model; each model/bundle/backend/quantization tuple needs correctness,
  quality, memory, performance, API, and reliability evidence.

## Status

| Surface | v0.2.0 status | Notes |
|---|---|---|
| GGUF CPU inference | Supported | Default build and fallback path |
| GGUF CUDA inference | Beta | Optional `cuda` feature; NVIDIA sm_70+ PTX baseline |
| Hugging Face SafeTensors | Beta, CUDA only | Qwen2/Qwen3 dense and selected 4-bit packed layouts |
| OpenAI-compatible HTTP API | Beta | Supported endpoint/field subset documented below |
| C and Python bindings | Experimental | Workspace crates exist; packaging is not yet released; Python requires Rust 1.83+ |
| `xrt-image` generative inference | Experimental, unreleased | Native Qwen-Image-2512 generation and Qwen-Image-Edit-2511 execution foundations; production admission remains open |
| `xrt-vision` task inference | Experimental | mmproj vision input and self-contained ONNX background removal |
| `xrt-video` / `xrt-audio` | Planned | Capability boundaries only; no empty placeholder crates or support claims |

See [Supported Models](docs/SUPPORTED_MODELS.md) for the exact architecture,
format, and backend matrix. Unsupported combinations return explicit errors;
`auto` may fall back from CUDA to CPU for GGUF models.

## Quick Start

### Requirements

- Rust toolchain with Cargo. Rust 1.76 is enforced for the core runtime, CLI,
  server, and C binding by the hosted MSRV gate. The experimental `xrt-python`
  binding requires Rust 1.83 or newer.
- A GGUF model for CPU or CUDA inference.
- For CUDA: an NVIDIA GPU supporting the PTX baseline and a compatible driver.

### Build

```bash
git clone https://github.com/XENO-CORPORATION/xeno-rt.git
cd xeno-rt

# Portable CPU binaries
cargo build --release --locked -p xrt-cli -p xrt-server

# CUDA-enabled binaries
cargo build --release --locked -p xrt-cli -p xrt-server --features cuda
```

Default release builds are portable. Machine-specific CPU flags are opt-in and
reserved for local benchmarking; see [Benchmarking](docs/BENCHMARKING.md).

### Download a GGUF model

```bash
cargo run --release --locked -p xrt-cli -- download \
  --repo Qwen/Qwen3-0.6B-GGUF \
  --quantization Q4_K_M
```

Downloaded models are cached under `~/.cache/xrt/models`.

### Generate on CPU

```bash
cargo run --release --locked -p xrt-cli -- generate \
  --model ./models/model.gguf \
  --backend cpu \
  --prompt "Explain local inference in three sentences." \
  --max-tokens 128
```

### Generate on CUDA

```bash
cargo run --release --locked -p xrt-cli --features cuda -- generate \
  --model ./models/model.gguf \
  --backend cuda \
  --prompt "Explain GPU-resident KV cache in three sentences." \
  --max-tokens 128
```

Backend values are `auto`, `cpu`, `cuda` (aliases include `gpu` and
`cuda-resident`), and `external-openai` where proxying is supported. `auto`
selects CUDA only in a CUDA-enabled build when the GGUF model is compatible and
the device initializes; otherwise it uses CPU.

## Command Line

The `xrt` CLI exposes four commands:

```text
xrt generate   One-shot text generation
xrt chat       Interactive chat
xrt bench      CPU, CUDA, or external-backend benchmark reports
xrt download   Hugging Face GGUF download and cache management
```

Build `xrt-cli` with `--features image-generation` to add the experimental
`xrt image` command group (`generate`, `edit`, `bench`, and `import`). Real-model
procedures and exact bundle manifests live in the
[Qwen Image reference guide](reference/image/qwen/README.md); their presence is
not a production support claim.

Use `--help` at each level for the authoritative options:

```bash
cargo run --locked -p xrt-cli -- --help
cargo run --locked -p xrt-cli -- bench --help
```

## OpenAI-Compatible Server

Start a CPU server bound to localhost:

```bash
cargo run --release --locked -p xrt-server -- \
  --model ./models/model.gguf \
  --backend cpu \
  --host 127.0.0.1 \
  --port 3000
```

Start the same server with CUDA enabled:

```bash
cargo run --release --locked -p xrt-server --features cuda -- \
  --model ./models/model.gguf \
  --backend cuda
```

Chat completion example:

```bash
curl http://127.0.0.1:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-model",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 64,
    "temperature": 0.7,
    "stream": false
  }'
```

Implemented routes:

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/v1/models` | OpenAI-style model list |
| `POST` | `/v1/completions` | Text completions, including SSE streaming |
| `POST` | `/v1/chat/completions` | Chat completions, including SSE streaming |
| `GET` | `/v1/runtime/status` | Backend, GPU, scheduler, and cache status |
| `POST` | `/v1/runtime/load` | Load or replace a local/external runtime |
| `POST` | `/v1/runtime/unload` | Release the active runtime |
| `POST` | `/v1/images/remove-background` | Experimental ONNX image task |

With `xrt-server` built using `--features image-generation`, the server also
adds experimental `POST /v1/images/generations`, `POST /v1/images/edits`, and
`GET /v1/runtime/models` routes. Existing text routes and schemas remain
unchanged when image inference is disabled or enabled. Image generation and
edit are synchronous for now; `stream=true` returns an explicit unsupported
error until compliant image-usage metering is implemented.

The server defaults to `127.0.0.1`. Text routes currently have no built-in
inbound authentication. Feature-gated image routes support `XRT_API_KEY` and
refuse an unauthenticated non-loopback bind unless the operator explicitly sets
`XRT_ALLOW_UNAUTHENTICATED_IMAGE_API=1`; a TLS/authenticating reverse proxy and
network-level controls are still required on untrusted networks. See
[API Reference](docs/API.md) and [Security Policy](SECURITY.md).

## Architecture

```text
XENO applications, agents, and local clients
                         |
                  xrt-cli / xrt-server
                         |
        +----------------+----------------+
        |                |                |
     xrt-text         xrt-image        xrt-vision
   implemented       experimental    task inference
        |
  xrt-runtime + xrt-models
        +----------------+----------------+
                         |
 formats + bundles + tensors + kernels + scheduler
 resource manager + CPU/CUDA + cache + telemetry

Future capability boundaries: xrt-video and xrt-audio
```

The runtime owns validated model loading, graph execution, backend selection,
scheduling, memory accounting, caching, telemetry, and inference APIs.
Consumer applications own canvases, layers, masks, timelines, tracks, project
state, and editing workflows. `xeno-lib` owns deterministic media processing
and codecs. GGUF remains the portable text-model contract and CPU remains the
required fallback for every capability advertised as CPU-capable.

### Workspace map

| Crate | Responsibility |
|---|---|
| `xrt-core` | Shared dtypes, tensor views, cache traits, and errors |
| `xrt-gguf` | Memory-mapped GGUF parsing and validation |
| `xrt-safetensors` | Hugging Face config and SafeTensors bundle loading |
| `xrt-tokenizer` | GGUF/HF tokenizer loading and chat templates |
| `xrt-kernels` | CPU kernels and quantized matvec implementations |
| `xrt-cuda` | CUDA buffers, kernels, graphs, KV pages, and telemetry |
| `xrt-models` | Text model execution, LoRA, and mmproj vision encoder |
| `xrt-runtime` | Current text runtime plus shared scheduling and resource primitives |
| `xrt-image` | Experimental native generative-image runtime and Qwen adapters |
| `xrt-openai` | OpenAI-compatible schemas and guarded external client |
| `xrt-hub` | Hugging Face discovery plus immutable bundle/download/cache handling |
| `xrt-cli` | Text commands and feature-gated image commands |
| `xrt-server` | OpenAI-compatible and additive runtime HTTP services |
| `xrt-vision` | ONNX image-task pipelines |
| `xrt-capi` | Experimental C ABI |
| `xrt-python` | Experimental PyO3 binding; Rust 1.83+; packaging not released |
| `xtask` | Repository maintenance commands |

The detailed design is in [Architecture](docs/ARCHITECTURE.md).

## Configuration

Common runtime variables:

| Variable | Default | Purpose |
|---|---:|---|
| `XRT_BACKEND` | `auto` | Backend selection |
| `XRT_KV_CACHE_MODE` | `f32` | `f32`, `q8`, `kq4_vq8`, or `agent_adaptive` |
| `XRT_CUDA_DEVICE` | `0` | CUDA device ordinal |
| `XRT_GPU_MEMORY_FRACTION` | `0.90` | Fraction of visible GPU memory available to xeno-rt |
| `XRT_GPU_RESERVED_MB` | `1024` | GPU memory kept outside the model budget |
| `XRT_GPU_KV_FRACTION` | `0.30` | Post-weight CUDA budget reserved for KV |
| `XRT_CUDA_GRAPH` | `auto` | `auto`, enabled, or disabled |
| `XRT_PREFIX_CACHE` | enabled | Prefix-cache toggle |
| `RUST_LOG` | `info` | `tracing` filter for CLI/server logs |

See [Configuration](docs/CONFIGURATION.md) for all resource, scheduler, prefix
cache, external backend, and diagnostic settings.

## Testing and Benchmarks

Normal development gates are hosted in GitHub Actions. Lightweight commands
for a development machine are:

```bash
cargo fmt --all -- --check
cargo check --workspace --all-targets --locked
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked
cargo bench --workspace --no-run --locked

# CPU-safe native image crate; real-model suites remain explicit and ignored.
cargo test -p xrt-image --no-default-features --locked
```

CUDA execution tests require an NVIDIA self-hosted runner and are never run on
untrusted pull-request code. Benchmark results must include the model, format,
backend, hardware, driver, build profile, commit, and command. See
[Benchmarking](docs/BENCHMARKING.md).

## Documentation

- [Documentation index](docs/README.md)
- [Runtime domains and product boundary](docs/RUNTIME_DOMAINS.md)
- [Architecture](docs/ARCHITECTURE.md)
- [API reference](docs/API.md)
- [Configuration](docs/CONFIGURATION.md)
- [Supported models](docs/SUPPORTED_MODELS.md)
- [Benchmarking](docs/BENCHMARKING.md)
- [Development guide](docs/DEVELOPMENT.md)
- [Roadmap](docs/ROADMAP.md)
- [GPU acceleration specification](docs/GPU_RUNTIME_ACCELERATION_SPEC.md)
- [KTransformers-inspired exact hybrid MoE specification](docs/ktransformers-inspired-hybrid-moe-acceleration-spec.md)
- [Qwen Image inference specification](docs/xrt-image-qwen-image-inference-spec.md)
- [Repository hardening specification](docs/REPOSITORY_HARDENING_SPEC.md)

## Project Policy

- [Contributing](CONTRIBUTING.md)
- [Governance](GOVERNANCE.md)
- [Support](SUPPORT.md)
- [Security](SECURITY.md)
- [Release process](RELEASE.md)
- [Changelog](CHANGELOG.md)
- [v0.2.0 release notes](docs/releases/0.2.0.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)

The Rust crates are currently internal workspace components and are not
published to crates.io. Public compatibility commitments apply to documented
CLI behavior, documented HTTP fields/routes, GGUF support, and release
artifacts unless a release note says otherwise.

## License

Licensed under the [Apache License 2.0](LICENSE). Contributions are subject to
the project [CLA](CLA.md) and contribution policy.
