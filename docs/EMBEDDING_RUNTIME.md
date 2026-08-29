# Embedding Runtime Deployment

**Scope:** `xrt-embedding`

The admitted contract is `nomic-ai/nomic-embed-text-v1.5` at immutable revision
`a15734e81021ea6c92b09050d2c7085001db8f36`. Its audited manifest is
`reference/embedding/nomic-embed-text-v1.5-a15734e.json`. The manifest is the
authority for artifact sizes, SHA-256 digests, source URLs, license notice,
task prefixes, pooling, normalization, and the 512-dimensional output.

## Provision the bundle

Online provisioning downloads every file before service startup, verifies the
declared size and SHA-256, and atomically publishes the digest-addressed cache:

```powershell
cargo run --release --locked -p xrt-cli -- bundle install `
  --manifest reference/embedding/nomic-embed-text-v1.5-a15734e.json `
  --cache-dir C:\ProgramData\XENO\xrt-models
```

For an offline or release-image build, place the two declared files in one
directory and use `bundle import` with `--source-dir`. Runtime startup points at
the installed digest directory, not at Hugging Face and not at a mutable model
alias. Preserve `xrt.bundle.json` beside the model files as deployment evidence.

## Start the internal service

```powershell
$env:XRT_EMBEDDING_MODEL_DIR = '<installed digest directory>'
$env:XRT_EMBEDDING_API_KEY = '<service secret from the deployment secret store>'
$env:XRT_EMBEDDING_MAX_CONCURRENT_REQUESTS = '4'
xrt-server --host 0.0.0.0 --port 3099
```

A non-loopback bind with embeddings enabled fails at startup without the exact
service secret. Keep the service on a protected network; terminate TLS and
apply network policy at the platform ingress. The platform sends the same
secret as `Authorization: Bearer ...` through `XENO_EMBEDDING_API_KEY`.

For the hosted platform service, use the codified Docker deployment. Its
default is a no-side-effect dry run:

```powershell
node scripts/deploy-embedding-runtime.mjs
node scripts/deploy-embedding-runtime.mjs --build-only --execute
node scripts/deploy-embedding-runtime.mjs --execute
```

The target host must already provide `/etc/xeno/xrt-embedding.env` with mode
`0600` or `0640` and a strong `XRT_EMBEDDING_API_KEY`. The image pins its Rust
builder and Debian runtime by digest, verifies the immutable Nomic bundle during
the build, verifies the official ONNX Runtime 1.20.0 Linux archive against
`reference/runtime/onnxruntime-1.20.0-linux-x64.json`, and runs an isolated
authenticated candidate before any swap. It joins only the private platform
Docker network and publishes no host port.

## Readiness and negative probes

`GET /v1/runtime/status` must report `ready: true`, the exact model identity,
512 dimensions, and `embedding_auth_required: true`. Before enabling semantic
workers, prove missing and incorrect bearer tokens return `401`, then prove the
correct token returns a vector and the exact `xeno_contract` revision.

Run `scripts/verify-embedding-parity.py` against the candidate service and
installed bundle. It independently executes the tokenizer and ONNX graph,
reproduces pooling/projection/normalization, compares all 512 values for the
locked query and document fixtures, and proves the two task prefixes produce
distinct vectors.

## ONNX Runtime packaging

`xrt-embedding` and `xrt-vision` use the exact workspace ORT dependency.
Windows uses explicit dynamic loading and requires the integrity-
locked ONNX Runtime 1.20.0 companion declared in
`reference/runtime/onnxruntime-1.20.0-windows-x64.json`. After building, run
`scripts/provision-onnxruntime.ps1 -Destination target/release`; preserve the
DLL and both generated license/notice files beside `xrt-server.exe`.

Embedding startup verifies the companion size and SHA-256 before initializing
ORT from its absolute path. It never falls through to a machine-global DLL.
Qualification inspects the loaded module path and runs existing text, image,
vision, and embedding gates against the packaged binary before rollout.

## Rollback

Disable semantic backfill first. Point the service at the prior immutable
runtime and bundle, prove readiness and contract identity, then restore traffic.
Do not write embeddings while the database active contract and runtime response
disagree; lexical retrieval remains the safe fallback.
