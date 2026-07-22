# Qwen Image Phase 0 evidence

**Runtime domain:** `xrt-image`
**Canonical architecture:** [../../../docs/RUNTIME_DOMAINS.md](../../../docs/RUNTIME_DOMAINS.md)

Status: **oracle baselines captured; native Q4 CPU correctness is experimental,
not production-admitted**.

This directory is the evidence root for Phase 0 of
`docs/xrt-image-qwen-image-inference-spec.md`. It separates verified provenance
from measurements that have actually run. An empty directory, planned command,
or upstream capability is never counted as a passing result.

## Captured

- `environment.json` records the repository revisions, target host, RTX 4090,
  pinned Diffusers CUDA environment, pinned stable-diffusion.cpp executable,
  Q4 component sizes, and a provisional sequential placement calculation.
- The immutable upstream, package, comparator, quality-suite, and OpenAI-fixture
  pins live in `reference/image/qwen/phase0-lock.json`.
- Complete BF16/Q8/Q6/Q5/Q4 bundle manifests live under
  `reference/image/qwen/manifests/`.
- The frozen 270-case quality suite and 50 identity pairs live in
  `tests/common/image-quality-suite.json`; its 20 procedural inputs live under
  `tests/fixtures/image-quality/`.
- OpenAI generation JSON, ordered edit multipart, SSE, response, and error
  fixtures live under `tests/fixtures/openai/images/`.
- The official BF16 smoke and 1024x1024/50-step release runs live under
  `diffusers/`; the full release workload completed in 539.719 seconds.
- The pinned stable-diffusion.cpp Q4_K_M smoke and matched release runs live
  under `stable-diffusion-cpp/`; the release workload completed in 120.907
  seconds with a 13,951 MiB sampled device-wide peak delta.
- Three additional serialized Q4 smoke runs are summarized in
  `stable-diffusion-cpp/q4-quiet-baseline-512x512-s4-summary.json`. They produced
  identical decoded pixels and PNG bytes; the two warm-artifact runs averaged
  13.766 seconds and the sampled device peak delta ranged from 13,336 to 13,594
  MiB. The summary deliberately does not admit the quiet-baseline gate because
  interactive GPU workloads were still resident on the workstation.
- `comparator-component-equivalence.json` proves byte-exact equivalence for
  all 923 repacked text-encoder and VAE tensors (16,838,118,374 bytes).
- `diffusers/bf16-smoke-text-checkpoints-v1/` captures all 28 text-layer
  checkpoints plus final norm for cross-backend numerical admission.
- `../native/qwen-image-2512-q4_k_m-cpu-smoke-16x16-s2.json` records four
  real native Q4 CPU executions and a three-cold-process determinism gate.
- `../native/qwen-image-2512-bf16-cpu-smoke-16x16-s2.json` records three real
  native BF16 CPU executions with identical decoded-pixel and PNG hashes.

## Still required for the Phase 0 exit gate

- A quiet, repeatable non-XENO VRAM baseline. The original measurement and the
  three-run 2026-07-22 repeatability sample were captured on an active
  workstation and are not release-performance baselines. Close interactive GPU
  workloads, verify the preflight inventory, and rerun the serialized sample.

The 30-repetition matched native performance comparison and quantized quality
runs are later admission gates. Phase 0 establishes reproducible oracles and a
feasible measurement path; it does not claim that XENO is already faster or
that any image quantization is production-supported.

## Reproduction

Use the project-scoped commands in `reference/image/qwen/README.md`. Real GPU
runs must use `scripts/safe-image-reference.ps1`, which requires explicit
confirmation, serializes the target processes, bounds execution time, and
checks for leftovers after failure or interruption.
