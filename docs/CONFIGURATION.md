# Configuration

Command-line flags override or feed the same runtime settings where a flag is
available. Invalid values generally fall back to a documented default for
resource policies; explicit backend/model incompatibilities fail.

## Backend

| Variable | Default | Values |
|---|---:|---|
| `XRT_BACKEND` | `auto` | `auto`, `cpu`, `cuda`, `external-openai` |
| `XRT_KV_CACHE_MODE` | `f32` | `f32`, `q8`, `kq4_vq8`, `agent_adaptive` |

`auto` uses CPU in a non-CUDA build. In a CUDA build it attempts a compatible
GGUF CUDA backend and falls back to CPU if initialization fails. An explicit
`cuda` request fails rather than falling back.

## CUDA Resource Policy

| Variable | Default | Meaning |
|---|---:|---|
| `XRT_CUDA_DEVICE` | `0` | CUDA device ordinal |
| `XRT_GPU_MEMORY_FRACTION` | `0.90` | Maximum usable fraction of total device memory |
| `XRT_GPU_RESERVED_MB` | `1024` | Device memory excluded from xeno-rt's upload budget |
| `XRT_GPU_KV_FRACTION` | `0.30` | Remaining post-weight budget assigned to KV |
| `XRT_CUDA_GRAPH` | `auto` | `auto`, `1`/`enabled`, or `0`/`disabled` |
| `XRT_CUDA_POOL_RELEASE_THRESHOLD_MB` | `256` | Stream-ordered pool retention threshold, max `4096` |
| `XRT_CUDA_PROFILE` | disabled | Enables CUDA profiling diagnostics when truthy |

The backend estimates resident model bytes and checks the configured safe
budget before upload. These controls do not reserve memory from other
processes; operators must also monitor device-wide usage.

## Scheduler

| Variable | Default | Constraint |
|---|---:|---|
| `XRT_MAX_ACTIVE_SEQUENCES` | `1` | At least 1 |
| `XRT_MAX_QUEUED_SEQUENCES` | `32` | 0 disables waiting queue capacity |
| `XRT_STREAM_BUFFER_CAPACITY` | `32` | At least 1 |
| `XRT_PREFILL_CHUNK_TOKENS` | `128` | At least 1 |
| `XRT_MAX_DECODE_TURNS_BEFORE_PREFILL` | `8` | At least 1 |
| `XRT_MAX_DECODE_BATCH_SIZE` | `4` | At least 1 |
| `XRT_DECODE_BATCH_WAIT_MICROS` | `20000` | May be 0 |

Increasing concurrency or batch size can materially increase KV and scratch
memory. Change one control at a time and observe `/v1/runtime/status`.

## Prefix Cache

| Variable | Default | Meaning |
|---|---:|---|
| `XRT_PREFIX_CACHE` | enabled | Boolean toggle |
| `XRT_PREFIX_CACHE_MAX_ENTRIES` | `32` | Maximum cached prefixes |
| `XRT_PREFIX_CACHE_MAX_BYTES` | `268435456` | Maximum resident cache bytes |
| `XRT_PREFIX_CACHE_MIN_TOKENS` | `8` | Minimum prefix length admitted |

Setting max entries or max bytes to zero disables the prefix cache. Entries
are namespaced by model, architecture, geometry, tokenizer, and backend.

## Cache Policies

Requests may set `cache_policy` to:

| Policy | Recent window | Behavior |
|---|---:|---|
| `default_chat` | `256` | Standard F32/default-cache behavior |
| `agent_adaptive` | `256` | Pins system/tool schema and protects tool results |
| `long_context` | `384` | Larger recent tier and earlier protected spans |
| `memory_saver` | `192` | Smaller recent tier with selective protection |

`recent_window_tokens` overrides the selected policy default.

## External OpenAI-Compatible Backend

| Variable | Default | Meaning |
|---|---:|---|
| `XRT_EXTERNAL_BASE_URL` | required | Base URL including `/v1` |
| `XRT_EXTERNAL_API_KEY` | unset | Outbound bearer token |
| `XRT_EXTERNAL_MODEL` | unset | Model inserted when a request omits one |
| `XRT_EXTERNAL_TIMEOUT_SECONDS` | `300` | Request timeout, 1 through 3600 |
| `XRT_EXTERNAL_ALLOW_REMOTE` | disabled | Permit a non-loopback target when truthy |

Remote targets are denied by default to reduce accidental credential leakage
and SSRF exposure.

## Model Loading and Diagnostics

| Variable | Default | Meaning |
|---|---:|---|
| `XRT_LOCAL_MODEL_ROOT` | platform default | Local alias search root |
| `XRT_HEAP_WEIGHTS` | disabled | Windows-only experimental GGUF heap copy (`1`) |
| `RUST_LOG` | `info` | `tracing` filter, for example `xrt_runtime=debug` |

Variables beginning with `XRT_REAL_` are integration-test fixture locations,
not runtime configuration.

## Example: Guarded CUDA Server

```powershell
$env:XRT_BACKEND = 'cuda'
$env:XRT_CUDA_DEVICE = '0'
$env:XRT_GPU_MEMORY_FRACTION = '0.80'
$env:XRT_GPU_RESERVED_MB = '2048'
$env:XRT_KV_CACHE_MODE = 'q8'
$env:RUST_LOG = 'info'

cargo run --release --locked -p xrt-server --features cuda -- `
  --model .\models\model.gguf
```
