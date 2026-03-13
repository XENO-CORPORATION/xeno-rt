<p align="center">
  <h1 align="center">xeno-rt</h1>
  <p align="center">
    <strong>High-performance LLM inference runtime written in Rust</strong>
  </p>
  <p align="center">
    <a href="https://github.com/XENO-CORPORATION/xeno-rt/actions"><img src="https://img.shields.io/github/actions/workflow/status/XENO-CORPORATION/xeno-rt/ci.yml?branch=main&style=flat-square&logo=github&label=CI" alt="CI"></a>
    <a href="https://github.com/XENO-CORPORATION/xeno-rt/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue?style=flat-square" alt="License"></a>
    <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/rust-1.76%2B-orange?style=flat-square&logo=rust" alt="Rust"></a>
    <a href="https://github.com/XENO-CORPORATION/xeno-rt/releases"><img src="https://img.shields.io/github/v/release/XENO-CORPORATION/xeno-rt?style=flat-square&label=release" alt="Release"></a>
  </p>
</p>

---

**xeno-rt** (`xrt`) is a from-scratch Rust inference runtime for large language models, designed as a high-performance alternative to [llama.cpp](https://github.com/ggml-org/llama.cpp). It prioritizes memory safety, clean architecture, and first-class GGUF compatibility while delivering competitive inference speed through hand-tuned CPU kernels and CUDA GPU acceleration.

## Highlights

- **Pure Rust** — no C/C++ dependencies in the hot path. Memory-safe by default.
- **GGUF native** — first-class support for the GGUF model format with mmap-based zero-copy loading.
- **Quantization** — F32, F16, BF16, Q8_0, Q4_0, Q4_K, Q5_K, Q6_K with ggml-compatible dequantization.
- **CPU + CUDA** — optimized CPU kernels (AVX2/NEON auto-vectorization, rayon parallelism) with feature-gated CUDA backend.
- **Llama-family models** — Llama, Mistral, Qwen and compatible architectures out of the box.
- **OpenAI-compatible API** — drop-in replacement server with `/v1/chat/completions` and `/v1/completions` + SSE streaming.
- **Paged KV cache** — memory-efficient key-value cache with block-level paging from day one.
- **HuggingFace Hub** — download and cache GGUF models directly from HuggingFace with one command.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        xrt-cli / xrt-server                     │
│                   (CLI interface / OpenAI API)                   │
├─────────────────────────────────────────────────────────────────┤
│                          xrt-runtime                            │
│            (session management, sampling, scheduling)            │
├──────────────────────┬──────────────────────────────────────────┤
│     xrt-models       │              xrt-hub                     │
│  (Llama forward pass,│     (HuggingFace downloads,              │
│   GQA, transformer)  │      model caching)                      │
├──────────────────────┼──────────────────────────────────────────┤
│    xrt-kernels       │            xrt-cuda                      │
│  (CPU: RMSNorm, RoPE,│     (GPU: PTX kernels,                  │
│   softmax, SiLU,     │      cudarc backend)                     │
│   tiled matmul,      │                                          │
│   quantized matmul)  │                                          │
├──────────────────────┴──────────────────────────────────────────┤
│                     xrt-gguf / xrt-tokenizer                    │
│            (GGUF parser, mmap, BPE tokenizer)                   │
├─────────────────────────────────────────────────────────────────┤
│                          xrt-core                               │
│              (DType, Device, TensorView, errors)                │
└─────────────────────────────────────────────────────────────────┘
```

### Crate Overview

| Crate | Description |
|---|---|
| `xrt-core` | Core types: `DType`, `Device`, `TensorView`, error definitions |
| `xrt-gguf` | GGUF binary parser with mmap, metadata extraction, tensor validation |
| `xrt-tokenizer` | BPE tokenizer loaded from GGUF metadata, special token handling |
| `xrt-kernels` | CPU compute kernels — RMSNorm, RoPE, softmax, SiLU, tiled matmul, quantized ops |
| `xrt-cuda` | CUDA GPU backend — 7 PTX kernels via cudarc (feature-gated) |
| `xrt-models` | Model implementations — Llama forward pass with grouped query attention |
| `xrt-runtime` | Inference runtime — paged KV cache, sampling strategies, session management |
| `xrt-hub` | HuggingFace model hub integration — download, cache, verify GGUF models |
| `xrt-cli` | Command-line interface — `generate`, `download` commands |
| `xrt-server` | HTTP server — OpenAI-compatible API with SSE streaming |
| `xtask` | Developer tooling — model management, cache operations |

## Getting Started

### Prerequisites

- **Rust 1.76+** (install via [rustup](https://rustup.rs/))
- **CUDA Toolkit 12.x** (optional, for GPU acceleration)

### Build

```bash
# Clone the repository
git clone https://github.com/XENO-CORPORATION/xeno-rt.git
cd xeno-rt

# Build (CPU only)
cargo build --release

# Build with CUDA support
cargo build --release --features cuda
```

### Download a Model

```bash
# Download from HuggingFace Hub
cargo run --release -p xrt-cli -- download \
  --hf-repo TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  --hf-file tinyllama-1.1b-chat-v1.0.Q4_0.gguf

# Or use xtask for model management
cargo xtask download --repo TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  --file tinyllama-1.1b-chat-v1.0.Q4_0.gguf
```

Models are cached in `~/.cache/xrt/models/`.

### Generate Text (CLI)

```bash
# From a local GGUF file
cargo run --release -p xrt-cli -- generate \
  --model ./models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
  --prompt "Explain quantum computing in simple terms" \
  --max-tokens 256 \
  --temperature 0.7 \
  --top-p 0.9

# From HuggingFace (auto-downloads if not cached)
cargo run --release -p xrt-cli -- generate \
  --hf-repo TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  --hf-file tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
  --prompt "Hello, world!"
```

### Start the API Server

```bash
cargo run --release -p xrt-server -- \
  --model ./models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
  --host 0.0.0.0 \
  --port 8080
```

The server exposes OpenAI-compatible endpoints:

```bash
# Chat completions
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tinyllama",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 128,
    "temperature": 0.7,
    "stream": true
  }'

# Text completions
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tinyllama",
    "prompt": "The meaning of life is",
    "max_tokens": 64
  }'
```

## Supported Models

| Architecture | Status | Examples |
|---|---|---|
| Llama | Supported | Llama 2, Llama 3, CodeLlama, TinyLlama |
| Mistral | Supported | Mistral 7B, Mixtral (dense layers) |
| Qwen | Supported | Qwen 1.5, Qwen 2 |

### Supported Quantizations

| Format | Block Size | Status | Notes |
|---|---|---|---|
| F32 | — | Full support | Reference precision |
| F16 | — | Full support | Half precision |
| BF16 | — | Full support | Brain float |
| Q8_0 | 32 | Full support | 8-bit symmetric |
| Q4_0 | 32 | Full support | 4-bit symmetric |
| Q4_K | 256 | Full support | 4-bit K-quant with 6-bit scales |
| Q5_K | 256 | Full support | 5-bit K-quant with 6-bit scales |
| Q6_K | 256 | Full support | 6-bit K-quant with 8-bit scales |

## Performance

### Benchmarks

```bash
# Run all benchmarks
cargo bench

# Specific benchmark
cargo bench --bench inference_bench
cargo bench --bench tokenizer_bench
```

Benchmarks cover:
- **Kernels** — RMSNorm, RoPE, softmax, SiLU across dimensions [128, 512, 2048, 4096]
- **Matmul** — tiled f32 matmul at [(128,128,128), (512,512,512), (1024,1024,64)]
- **Quantization** — Q4_0 and Q4_K dequantization throughput
- **Tokenizer** — BPE encode/decode at varying input lengths

### CPU Optimizations

- **Tiled matmul** with 64-wide tiles and 8-lane unrolled inner loop
- **Rayon** row-parallel execution for matmul and batch operations
- **Auto-vectorization** friendly loops (chunked-8 processing) for RMSNorm and softmax
- **f32 accumulation** for numerically stable quantized operations
- **mmap** zero-copy model loading — no eager weight copies

### GPU Acceleration

CUDA backend includes PTX kernels for:
- Matrix multiplication (tiled with shared memory)
- RMSNorm, RoPE, softmax, SiLU
- Elementwise add (residual connections)
- Embedding lookup

Enable with `--features cuda` (requires CUDA Toolkit 12.x).

## Testing

```bash
# Run all tests
cargo test --workspace

# Include ignored smoke tests (runs full inference pipeline)
cargo test --workspace -- --include-ignored

# Run specific test suite
cargo test --test kernels_test
cargo test --test gguf_parse_test
cargo test --test sampler_test
cargo test --test kv_cache_test
cargo test --test tokenizer_test
cargo test --test smoke_e2e
```

### Test Coverage

| Suite | Tests | What's Covered |
|---|---|---|
| `gguf_parse_test` | 4 | Header parsing, metadata, validation, error cases |
| `kernels_test` | 7 | RMSNorm, RoPE, softmax, SiLU, matmul, Q8_0/Q4_0 roundtrips |
| `kv_cache_test` | 3 | Page allocation, read/write, growth |
| `sampler_test` | 5 | Temperature, top-k, top-p, repetition penalty, greedy |
| `tokenizer_test` | 2 | Encode/decode roundtrip, special tokens |
| `smoke_e2e` | 2 | Full pipeline with synthetic model (1-token + 8-token) |

## Project Structure

```
xeno-rt/
├── Cargo.toml              # Workspace manifest
├── rust-toolchain.toml     # Rust stable toolchain
├── .cargo/config.toml      # SIMD target flags
├── crates/
│   ├── xrt-core/           # Core types and traits
│   ├── xrt-gguf/           # GGUF binary parser
│   ├── xrt-tokenizer/      # BPE tokenizer
│   ├── xrt-kernels/        # CPU compute kernels
│   │   └── src/cpu/
│   │       ├── matmul.rs   # Tiled matrix multiplication
│   │       ├── rmsnorm.rs  # RMS normalization
│   │       ├── rope.rs     # Rotary position embeddings
│   │       ├── softmax.rs  # Numerically stable softmax
│   │       ├── silu.rs     # SiLU activation
│   │       └── quantize.rs # Dequantization kernels
│   ├── xrt-cuda/           # CUDA GPU backend
│   ├── xrt-models/         # Model implementations
│   │   └── src/llama.rs    # Llama transformer
│   ├── xrt-runtime/        # Inference runtime
│   │   └── src/
│   │       ├── session.rs  # Session management
│   │       ├── sampler.rs  # Sampling strategies
│   │       └── kv_cache.rs # Paged KV cache
│   ├── xrt-hub/            # HuggingFace integration
│   ├── xrt-cli/            # CLI application
│   └── xrt-server/         # OpenAI-compatible server
├── xtask/                  # Developer tooling
├── benches/                # Criterion benchmarks
└── tests/                  # Integration tests
```

## Roadmap

### v0.1 — Foundation (current)
- [x] GGUF parser with mmap
- [x] BPE tokenizer from GGUF metadata
- [x] CPU kernels (RMSNorm, RoPE, softmax, SiLU, matmul)
- [x] F32/F16/BF16/Q8_0/Q4_0/Q4_K/Q5_K/Q6_K quantization
- [x] Llama-family forward pass with GQA
- [x] Paged KV cache
- [x] Sampling (temperature, top-k, top-p, repetition penalty)
- [x] CLI and OpenAI-compatible server
- [x] CUDA backend (structural, 7 PTX kernels)
- [x] HuggingFace Hub integration
- [x] Criterion benchmarks

### v0.2 — Performance
- [ ] AVX2/AVX-512 intrinsic kernels
- [ ] ARM NEON kernels
- [ ] Flash Attention implementation
- [ ] Continuous batching scheduler
- [ ] CUDA kernel optimization and profiling
- [ ] Metal backend (Apple Silicon)

### v0.3 — Production
- [ ] Speculative decoding
- [ ] Grammar-constrained generation
- [ ] LoRA adapter loading and hot-swap
- [ ] Safetensors import
- [ ] Distributed inference (tensor parallelism)
- [ ] Prometheus metrics and structured logging

### v0.4 — Ecosystem
- [ ] Python bindings (PyO3)
- [ ] C API for FFI consumers
- [ ] Docker images
- [ ] Multimodal support (vision encoders)
- [ ] MoE (Mixture of Experts) routing

## Configuration

### Environment Variables

| Variable | Description | Default |
|---|---|---|
| `XRT_LOG` | Log level filter (`trace`, `debug`, `info`, `warn`, `error`) | `info` |
| `XRT_CACHE_DIR` | Model cache directory | `~/.cache/xrt/models` |
| `XRT_THREADS` | Number of CPU threads for inference | System default |
| `CUDA_VISIBLE_DEVICES` | GPU device selection (when CUDA enabled) | All visible |

### Sampling Parameters

| Parameter | CLI Flag | API Field | Default | Description |
|---|---|---|---|---|
| Temperature | `--temperature` | `temperature` | `0.7` | Controls randomness (0 = greedy) |
| Top-K | `--top-k` | `top_k` | `40` | Keep top K candidates |
| Top-P | `--top-p` | `top_p` | `0.9` | Nucleus sampling threshold |
| Repetition Penalty | `--repeat-penalty` | `repetition_penalty` | `1.1` | Penalize repeated tokens |
| Max Tokens | `--max-tokens` | `max_tokens` | `256` | Maximum tokens to generate |

## Contributing

Contributions are welcome. Please open an issue or pull request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Ensure tests pass (`cargo test --workspace`)
4. Ensure code is formatted (`cargo fmt --check`)
5. Ensure no warnings (`cargo clippy --workspace`)
6. Submit a pull request

## License

Licensed under the [Apache License 2.0](LICENSE).

---

<p align="center">
  Built by <a href="https://github.com/XENO-CORPORATION">XENO CORPORATION</a>
</p>
