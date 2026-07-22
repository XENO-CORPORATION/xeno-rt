# Qwen Image reference harness

**Runtime domain:** `xrt-image`
**Canonical architecture:** [../../../docs/RUNTIME_DOMAINS.md](../../../docs/RUNTIME_DOMAINS.md)

This directory contains the non-production reference tooling and immutable
metadata used to admit native Qwen Image support into XENO RT. Python,
Diffusers, and stable-diffusion.cpp are correctness or benchmark oracles only;
no production Rust crate imports or launches them.

## Pinned environment

The Python environment is project-scoped and locked by `uv.lock`:

```powershell
uv sync --project reference/image/qwen --python 3.11 --frozen
uv run --project reference/image/qwen --frozen python reference/image/qwen/audit_phase0.py --verify
```

Install the optional frozen quality evaluators only on a quality runner:

```powershell
uv sync --project reference/image/qwen --python 3.11 --extra quality --frozen
```

PyTorch is resolved from its official CUDA 13.0 wheel index. The other
packages come from PyPI. Do not update an individual package without
regenerating the lock and all affected reference artifacts.

## Metadata audit

`audit_phase0.py` has immutable expected revisions embedded in source. It:

- verifies the official Qwen-Image-2512 and Qwen-Image-Edit-2511 revisions;
- verifies the pinned Unsloth GGUF transformer revisions and the Q8_0, Q6_K,
  Q5_K_M, and Q4_K_M artifact sizes and LFS SHA-256 values;
- hashes every small Hugging Face configuration/tokenizer file through an
  immutable `resolve/<revision>/...` URL;
- emits complete BF16 and mixed GGUF bundle manifests;
- verifies pinned reference-library releases, SDK commits, and both official
  stable-diffusion.cpp CUDA release assets;
- locks the generated quality suite, OpenAI wire fixtures, native comparator
  component conversions, and the comparator's exact CI build flags; and
- repeats the official Qwen organization search for a public Qwen-Image-3.0
  checkpoint without treating absence as a permanent fact.

To reproduce checked-in metadata without modifying files:

```powershell
uv run --project reference/image/qwen python reference/image/qwen/audit_phase0.py --verify
```

To intentionally refresh generated artifacts after reviewing upstream changes:

```powershell
uv run --project reference/image/qwen python reference/image/qwen/audit_phase0.py --write
```

The write mode uses atomic replacement. Generated manifests contain no access
tokens, signed query strings, local paths, or mutable `resolve/main` URLs.

## Frozen test and API fixtures

The license-clean quality suite and its procedural PNG inputs are rebuilt and
verified with:

```powershell
python reference/image/qwen/build_quality_suite.py
python reference/image/qwen/audit_phase0.py --write --verify
```

The frozen suite is not itself an admission result. Build its deterministic
execution plan and compile complete BF16-versus-quantized evaluator exports
with the fail-closed reference tool:

```powershell
uv run --project reference/image/qwen --extra quality python `
  reference/image/qwen/evaluate_quality_suite.py plan `
  --tier Q4_K_M `
  --output .codex-tmp/image-quality/q4-plan.json

uv run --project reference/image/qwen --extra quality python `
  reference/image/qwen/evaluate_quality_suite.py admit `
  --results .codex-tmp/image-quality/q4-results.json `
  --artifact-root .codex-tmp/image-quality/artifacts `
  --output benchmark-results/image/quality/qwen-image-q4_k_m.json
```

See [QUALITY_ADMISSION.md](QUALITY_ADMISSION.md) for the strict result schema,
artifact rules, human-review quorum, and statistical formulas. A passing report
covers only the quantization quality gate and never claims production support.

The OpenAI image JSON, multipart, SSE, response, and error fixtures are emitted
by the pinned official Python SDK. Multipart boundaries and binary bodies are
normalized to ordered names, sizes, and hashes so random SDK boundaries do not
create fixture churn:

```powershell
uv run --project reference/image/qwen --frozen python `
  reference/image/qwen/generate_openai_fixtures.py --write --verify
```

## Large artifacts

Model weights and executable archives are deliberately excluded from Git by
the repository `.gitignore`. Reference commands use
`XRT_IMAGE_REFERENCE_CACHE`, defaulting to `.codex-tmp/image-reference/` in the
repository. A cached artifact is usable only after its declared size and
SHA-256 have both been verified.

Install the pinned native comparator and complete Q4 generation inputs:

```powershell
uv run --project reference/image/qwen --frozen python `
  reference/image/qwen/download_reference_artifacts.py `
  --comparator-tool --comparator-components `
  --bundle qwen-image-2512-q4_k_m
```

Install the official BF16 Diffusers bundle separately:

```powershell
uv run --project reference/image/qwen --frozen python `
  reference/image/qwen/download_reference_artifacts.py `
  --bundle qwen-image-2512-bf16
```

Install an additional pinned quantized development bundle by its audited
manifest ID. For example, Q8_0 is the numerical-debugging tier and is useful
for CPU reference execution, but its transformer alone does not satisfy the
reserve-aware 24 GiB CUDA admission cap on the audited RTX 4090 workstation:

```powershell
uv run --project reference/image/qwen --frozen python `
  reference/image/qwen/download_reference_artifacts.py `
  --bundle qwen-image-2512-q8_0
```

The same command accepts `qwen-image-2512-q6_k` and
`qwen-image-2512-q5_k_m`. Each manifest shares already verified component
blobs where possible and downloads only missing content.

Install the pinned Qwen-Image-Edit-2511 Q4_K_M development bundle separately:

```powershell
uv run --project reference/image/qwen --frozen python `
  reference/image/qwen/download_reference_artifacts.py `
  --bundle qwen-image-edit-2511-q4_k_m
```

The Edit transformer contains the upstream zero-byte
`__index_timestep_zero__` compatibility marker. XENO RT does not weaken normal
GGUF validation for it: the Edit adapter opts into a narrow policy that accepts
only the final rank-one `[0]` F32 marker at the Qwen Image data-section end.
The bundle remains usable only after the manifest and every artifact hash pass.

Install the complete official Edit-2511 BF16 tree when validating SafeTensors
or the exact raw-Diffusers importer. This is roughly 57.7 GB before cache
deduplication and is not needed for the Q4 execution smoke:

```powershell
uv run --project reference/image/qwen --frozen python `
  reference/image/qwen/download_reference_artifacts.py `
  --bundle qwen-image-edit-2511-bf16
```

`xrt image import` treats a directory containing `xrt.bundle.json` as an
already-manifested bundle. To exercise raw import, use a separate local fixture
containing exactly the 33 pinned model files and no manifest, then run without
`--install`:

```powershell
cargo run --release -p xrt-cli --features image-generation -- `
  image import --path '<exact audited raw Edit-2511 BF16 directory>'
```

Successful raw validation prints a reviewable local-only candidate manifest to
stdout. It does not write into the source tree, install the candidate, or imply
that BF16 inference has passed.

The downloader uses a content-addressed blob cache, the pinned Hugging Face
client/Xet transport for large Hub artifacts, hard links where supported, and
atomic materialization. Interrupted direct transfers retain a `.partial` file
for bounded resume; hash failures never become discoverable bundles.

After the Q4 bundle and comparator components are installed, prove that the
single-file comparator repacks preserve every official BF16 tensor payload:

```powershell
uv run --project reference/image/qwen --frozen python `
  reference/image/qwen/verify_comparator_equivalence.py --write
```

## Real reference runs

Real model runs must go through the serialized Windows safety wrapper. The
explicit confirmation flag prevents an accidental tens-of-GiB load:

```powershell
.\scripts\safe-image-reference.ps1 `
  -ConfirmLargeModelRun -Engine diffusers -Profile smoke

.\scripts\safe-image-reference.ps1 `
  -ConfirmLargeModelRun -Engine native -Profile smoke
```

The Diffusers runner captures initial latents, prompt embeddings, first-step
transformer outputs, scheduler checkpoints, VAE inputs/outputs, an uncompressed
pixel hash, and sampled RAM/VRAM. The native runner records the pinned Q4
command, output hashes, logs, wall time, and sampled RAM/VRAM. A smoke result is
reference evidence, not the 30-run Phase 4 performance admission.

Capture the machine/toolchain/placement snapshot with:

```powershell
uv run --project reference/image/qwen --frozen python `
  reference/image/qwen/capture_phase0_environment.py --write
```

The Phase 0 lock records provenance. It is not a product support declaration.
Native support remains gated by the numerical, quality, memory, API, and
performance criteria in `docs/xrt-image-qwen-image-inference-spec.md`.

## Native CUDA safety workflows

Compile the image CUDA surfaces without executing GPU kernels:

```powershell
.\scripts\safe-image-cuda-check.ps1 -CompileOnly
```

Run the three bounded generation/edit/long-sequence CUDA parity tests only with
explicit consent:

```powershell
.\scripts\safe-image-cuda-check.ps1 -ConfirmGpuRun
```

The pinned Q4_K_M low-resolution real CUDA smoke has a separate explicit
large-model confirmation and performs host-memory, free-VRAM, manifest-hash,
timeout, process-serialization, output-hash, and cleanup checks:

```powershell
.\scripts\safe-image-cuda-smoke.ps1 `
  -Tier q4_k_m `
  -BundlePath '<verified qwen-image-2512-q4_k_m bundle>' `
  -ConfirmLargeModelRun
```

The audited `q6_k` and `q5_k_m` profiles have their own locked manifest,
output, and resident-allocation expectations. A newly added tier must first be
captured with `-AcceptUnpinnedCandidate`; that result explicitly makes no
correctness claim. Its observed hash/allocation must be reviewed and pinned,
then the run is repeated without that switch before a bounded correctness
smoke can pass. Q8_0 is intentionally absent from the CUDA profile set because
its transformer alone exceeds the reserve-aware cap on the audited 24 GiB
device.

The same wrapper has a pinned 512x512 quality-review workload for Q4_K_M. It
retains the first benchmark PNG as well as the JSON evidence and verifies that
the file hash matches the in-memory report:

```powershell
.\scripts\safe-image-cuda-smoke.ps1 `
  -Tier q4_k_m `
  -Workload comparator-512x512-s4 `
  -BundlePath '<verified qwen-image-2512-q4_k_m bundle>' `
  -ConfirmLargeModelRun
```

The benchmark CLI exposes the underlying `--retain-first-output <FILE>` option
for controlled quality workflows. It refuses to overwrite an existing file.
Both wrapper workloads remain bounded development evidence. The 512x512 case
uses only four steps and is not the quiet 30-run or release-resolution
performance/quality admission workload.

On the audited workstation, the retained contiguous convolution-row fast path
kept the locked 512x512 PNG hash and allocation peak while reducing the mean
CPU VAE decode from 44.560 seconds to 13.375 seconds across two repetitions.
Mean native wall time fell from 105.777 seconds to 74.624 seconds. Ordered AVX2
BF16 decode/multiply with original-order scalar accumulation then reduced mean
prompt encoding from 25.974 to 21.050 seconds and mean wall time to 69.204
seconds, without changing the hash or peak. Parallelizing only the independent
softmax-normalization pass, while retaining the serial stable-softmax scan and
value-accumulation order, kept the same hash and peak in two more repetitions.
It reduced mean denoising from 34.877 to 34.379 seconds and mean wall time to
68.976 seconds. The retained Q4 projection kernel now also cooperatively caches
each 16x256 activation tile in shared memory while preserving packed-weight
decode, FMA, and warp-reduction order. Four warm transformer-forward samples
improved from a 4.422-second pre-change mean to 3.904 seconds (1.133x), and two
locked full-pipeline repetitions reduced mean denoising to 30.521 seconds and
mean wall time to 64.901 seconds. Expanding the same shared tile from eight to
sixteen output warps then reduced four transformer-forward samples to a
3.614-second mean and two more full-pipeline repetitions to 28.311 seconds of
denoising and 62.807 seconds wall time. All four new full-pipeline runs kept the
exact PNG hash and 13,957,019,904-byte tracked peak. The pinned comparator took
29.563 seconds. Spatial VAE row-tile scheduling then exposed concurrency when
output-channel count was below the Rayon pool width, preserving each pixel's
accumulation order and reducing two VAE decodes to a 12.782-second mean and
wall time to 61.999 seconds. The current 2.097x wall ratio remains a retained
development optimization, not a performance admission pass. Reordering BF16
linear scheduling by output feature kept every per-token dot product in the
same K order while reusing each mapped weight row across prompt tokens. Two
locked runs reduced prompt encoding from 20.850 to 9.920 seconds (2.102x) and
wall time to 51.201 seconds, improving the native/comparator ratio to 1.732x.
A 32-warp/1,024-thread probe was exact but reduced the two-run full-pipeline
mean by only 0.69%; it is
recorded as rejected because that marginal gain did not justify the weaker
cross-device occupancy and portability margin. A width-scaled 16-row VAE tile
was also exact but regressed VAE decode by 3.97%, so the fixed eight-row tile
remains retained.

## Experimental Edit-2511 evidence

Set the complete pinned bundle directory before selecting any ignored real-model
test:

```powershell
$env:XRT_QWEN_IMAGE_EDIT_BUNDLE_DIR = `
  '<verified qwen-image-edit-2511-q4_k_m bundle>'
```

The lightweight real-bundle admission-plan checks are:

```powershell
cargo test -p xrt-image --release --no-default-features `
  --test real_qwen_bundle `
  pinned_qwen_image_edit_2511_cpu_loads_and_plans `
  -- --ignored --exact --nocapture

cargo test -p xrt-image --release --features cuda `
  --test real_qwen_bundle `
  pinned_qwen_image_edit_2511_q4_cuda_loads_and_plans `
  -- --ignored --exact --nocapture
```

The full CUDA diagnostic is deliberately ignored and has an internal
cooperative timeout. It loads roughly 13.65 GB of tracked transformer weights
and must be run only on the designated large-model workstation after closing
other inference workloads:

```powershell
$env:XRT_QWEN_IMAGE_EDIT_CUDA_SMOKE_TIMEOUT_SECONDS = '900'
cargo test -p xrt-image --release --features cuda `
  --test real_qwen_bundle `
  pinned_qwen_image_edit_2511_q4_cuda_edit_smoke `
  -- --ignored --exact --nocapture
```

The ordered two-source diagnostic has a separate longer bound and locked hash:

```powershell
$env:XRT_QWEN_IMAGE_EDIT_CUDA_MULTI_IMAGE_SMOKE_TIMEOUT_SECONDS = '1800'
cargo test -p xrt-image --release --features cuda `
  --test real_qwen_bundle `
  pinned_qwen_image_edit_2511_q4_cuda_two_image_edit_smoke `
  -- --ignored --exact --nocapture --test-threads=1
```

The configured three-source maximum requires the exact-order tiled attention
fallback and has its own 35-minute bound:

```powershell
$env:XRT_QWEN_IMAGE_EDIT_CUDA_THREE_IMAGE_SMOKE_TIMEOUT_SECONDS = '2100'
cargo test -p xrt-image --release --features cuda `
  --test real_qwen_bundle `
  pinned_qwen_image_edit_2511_q4_cuda_three_image_edit_smoke `
  -- --ignored --exact --nocapture --test-threads=1
```

The 2026-07-22 baseline candidate and locked repeat produced the same
deterministic PNG hash in about 8.7 minutes. After key-parallel CUDA attention,
two more locked repetitions kept that exact hash and tracked peak allocation
while reducing execution to about 3.9 minutes (3.690x faster denoising and
2.208x faster end to end on this bounded workload). The shared contiguous-row
VAE optimization then kept the same hash and peak in two further repetitions,
reduced source encoding from an 80.064-second mean to 34.722 seconds, and
reduced execution to a 192.508-second mean (about 3.2 minutes). Ordered BF16
AVX2 kept the hash/peak through two more runs, reduced prompt encoding from a
58.617-second mean to 45.836 seconds, and reduced execution to 175.629 seconds
(about 2.9 minutes). Two final repetitions parallelizing only independent
softmax normalization kept the same hash and peak, reduced denoising from a
99.190-second mean to 94.143 seconds (1.054x), and reduced execution to a
169.980-second mean (about 2.8 minutes). The retained shared Q4 activation tile,
fixed eight-row spatial VAE scheduling, and feature-major BF16 linear path then
kept the same hash and peak in two additional repetitions. Prompt encoding fell
from a 45.509-second mean to 29.989 seconds (1.518x), denoising from 94.143 to
83.842 seconds (1.123x), and total execution from 169.980 to 142.551 seconds
(1.192x, about 2.4 minutes). This remains an active-workstation experimental
result. CPU output, quiet performance, full-resolution quality,
edit identity, attribution, and production multi-image gates remain open, so no
Edit tier is production-advertised.

The bounded ordered two-image candidate and locked repeat also produced the
same PNG SHA-256
`5dde8efa3c6f2c3dc6a159956082e5677a88d4e1307e279f653fc6bdf822e7d3`
and the same 16,191,568,128-byte tracked peak. Their executions took 564.178
and 562.694 seconds. This proves only deterministic two-source execution at
16x16 and two steps. The three-source maximum then passed its own candidate and
locked repeat with PNG SHA-256
`b592ba1d170944f7ca9b41979c1df7d44f0bad4a8a6ee02c1e5d640d128ab31f`
and an identical 17,460,869,376-byte tracked peak. Those executions took
1,811.921 and 1,807.733 seconds. Together these runs cover the configured
one-to-three source range only for bounded CUDA correctness; release-resolution
quality, source-order attribution, CPU completion, and acceptable performance
remain open.
