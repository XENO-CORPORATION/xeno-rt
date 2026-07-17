# xeno-rt

[![CI](https://github.com/XENO-CORPORATION/xeno-rt/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/XENO-CORPORATION/xeno-rt/actions/workflows/ci.yml)
[![CUDA validation](https://github.com/XENO-CORPORATION/xeno-rt/actions/workflows/cuda.yml/badge.svg)](https://github.com/XENO-CORPORATION/xeno-rt/actions/workflows/cuda.yml)
[![Security audit](https://github.com/XENO-CORPORATION/xeno-rt/actions/workflows/audit.yml/badge.svg?branch=main)](https://github.com/XENO-CORPORATION/xeno-rt/actions/workflows/audit.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**xeno-rt** is a Rust inference runtime for running language models locally. It
provides GGUF-native CPU inference, an optional native CUDA backend, a command
line interface, and an OpenAI-compatible HTTP surface for applications that
need an offline alternative to cloud inference.

**v0.2.0** is the first stable GitHub release checkpoint. CPU execution is the
default supported path. CUDA execution is a real native runtime path and
remains beta; it is not a wrapper around another inference engine. The project
does not claim performance parity with other runtimes without a reproducible
benchmark.

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

## Status

| Surface | v0.2.0 status | Notes |
|---|---|---|
| GGUF CPU inference | Supported | Default build and fallback path |
| GGUF CUDA inference | Beta | Optional `cuda` feature; NVIDIA sm_70+ PTX baseline |
| Hugging Face SafeTensors | Beta, CUDA only | Qwen2/Qwen3 dense and selected 4-bit packed layouts |
| OpenAI-compatible HTTP API | Beta | Supported endpoint/field subset documented below |
| C and Python bindings | Experimental | Workspace crates exist; packaging is not yet released; Python requires Rust 1.83+ |
| Vision and image tasks | Experimental | mmproj vision input and ONNX background removal |

See [Supported Models](docs/SUPPORTED_MODELS.md) for the exact architecture,
format, and backend matrix. Unsupported combinations return explicit errors;
`auto` may fall back from CUDA to CPU for GGUF models.

## Quick Start

### Requirements

- Rust toolchain with Cargo. The core runtime, CLI, server, and C binding declare
  Rust 1.76 as their minimum; hosted MSRV jobs are the source of truth while
  v0.2.0 is being stabilized. The experimental `xrt-python` binding requires
  Rust 1.83 or newer.
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

The server defaults to `127.0.0.1` and currently has no built-in inbound API
authentication. Do not bind it to an untrusted network without a TLS/authenticating
reverse proxy and network-level access controls. See [API Reference](docs/API.md)
and [Security Policy](SECURITY.md).

## Architecture

```text
Applications / xrt CLI / OpenAI-compatible clients
                         |
                  xrt-cli / xrt-server
                         |
                     xrt-runtime
        +----------------+----------------+
        |                |                |
    CPU backend     CUDA backend    external-openai
        |                |
    xrt-models      xrt-cuda + xrt-models
        +----------------+
                         |
       xrt-gguf / xrt-safetensors / xrt-tokenizer
                         |
                      xrt-core
```

The runtime owns backend selection, sessions, KV cache policy, prefix caching,
scheduling, sampling, and resource status. Backend implementations must retain
the same causal-language-model contract. GGUF remains the portable model
contract and CPU remains the required fallback.

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
| `xrt-runtime` | Backend abstraction, sessions, scheduling, caches, sampling |
| `xrt-openai` | Guarded external OpenAI-compatible client |
| `xrt-hub` | Hugging Face model discovery, download, and local cache |
| `xrt-cli` | Generate, chat, download, and benchmark commands |
| `xrt-server` | HTTP API and runtime lifecycle service |
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
```

CUDA execution tests require an NVIDIA self-hosted runner and are never run on
untrusted pull-request code. Benchmark results must include the model, format,
backend, hardware, driver, build profile, commit, and command. See
[Benchmarking](docs/BENCHMARKING.md).

## Documentation

- [Documentation index](docs/README.md)
- [Architecture](docs/ARCHITECTURE.md)
- [API reference](docs/API.md)
- [Configuration](docs/CONFIGURATION.md)
- [Supported models](docs/SUPPORTED_MODELS.md)
- [Benchmarking](docs/BENCHMARKING.md)
- [Development guide](docs/DEVELOPMENT.md)
- [Roadmap](docs/ROADMAP.md)
- [GPU acceleration specification](docs/GPU_RUNTIME_ACCELERATION_SPEC.md)
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
