# Qwen Image quality admission

**Runtime domain:** `xrt-image`
**Canonical architecture:** [../../../docs/RUNTIME_DOMAINS.md](../../../docs/RUNTIME_DOMAINS.md)

`evaluate_quality_suite.py` is reference-only tooling for the frozen
`qwen-image-release-v1` suite. It does not run in production and a passing
report does not advertise a model tier. Performance, memory, CPU fallback,
API, and release gates remain independent.

Before generating a corpus, the manual `Image Quality Reference` workflow
must pass. Its hosted CPU job proves the complete PaddleOCR-VL v1.6 document
pipeline—not only its VLM component—can be reproduced without using a
developer workstation. The optional dedicated `image-quality` GPU job
captures equivalent CUDA evidence. Retained `models.json` records the frozen
evaluator revision and `ocr-smoke.json` records the pipeline execution. A
metadata-only local check is not admission evidence.

The first passing hosted execution is recorded in
`benchmark-results/image/quality/environment-2026-08-07.json`. It closes only
the evaluator-environment gate; all corpus, metric, human-review, and
production-admission gates below remain open.

## 1. Create the execution plan

Run this inside the pinned reference environment:

```powershell
uv run --project reference/image/qwen --extra quality python `
  reference/image/qwen/evaluate_quality_suite.py plan `
  --tier Q4_K_M `
  --output .codex-tmp/image-quality/q4-plan.json
```

The default plan contains all 250 currently active generation and Edit cases
plus all 50 designated identity-preservation pairs. Conditional inpaint is
excluded because the Qwen profile does not advertise `image.inpaint`. Only use
`--include-inpaint` after that capability independently passes its admission
gate; doing so adds the 20 frozen inpaint cases.

The plan fixes the suite digest, case order, prompts, seeds, artifact layout,
evaluator identities, 1024x1024 size, 50 steps, true CFG 4.0, and XENO RNG
schema. Generate the BF16 and candidate PNGs from the identical seed-derived
initial latent. Do not substitute Diffusers RNG output for either side.

## 2. Produce the results input

The admission compiler accepts one strict JSON object:

```json
{
  "schema_version": 1,
  "object": "xeno.image.quality_results",
  "suite": {
    "version": "qwen-image-release-v1",
    "sha256": "<sha256 from the plan>"
  },
  "tier": "Q4_K_M",
  "execution": {
    "paired_bf16_reference": true,
    "identical_xeno_initial_latent": true,
    "size": "1024x1024",
    "steps": 50,
    "true_cfg_scale": 4.0,
    "rng_schema": "xrt-normal-v1-splitmix64-marsaglia-f32le",
    "conditional_inpaint_admitted": false
  },
  "model_pairs": {
    "generation": {
      "bf16": {
        "model_id": "qwen-image-2512-bf16",
        "bundle_digest": "<lowercase sha256>",
        "logical_model_revision": "<official revision>",
        "artifact_revision": "<artifact revision>",
        "quantization": "BF16"
      },
      "candidate": {
        "model_id": "qwen-image-2512-q4_k_m",
        "bundle_digest": "<lowercase sha256>",
        "logical_model_revision": "<same official revision>",
        "artifact_revision": "<GGUF artifact revision>",
        "quantization": "Q4_K_M"
      }
    },
    "edit": {
      "bf16": { "...": "same fields" },
      "candidate": { "...": "same fields" }
    }
  },
  "evaluator_identity": { "...": "copy exactly from the plan" },
  "case_results": [],
  "identity_results": [],
  "human_review_protocol": {
    "blinded": true,
    "randomized": true,
    "rater_ids_pseudonymous": true,
    "rubric_sha256": "<sha256 from the plan>"
  },
  "human_reviews": []
}
```

Every planned case and identity pair must have one `bf16` and one `candidate`
record. The compiler rejects missing or extra IDs, reused paths, symlinks,
digest mismatches, non-PNG/animated/wrong-size artifacts, explicitly reported
NaN/Inf state, blank flags, and strictly uniform decoded images.

Each output record has exactly this shape:

```json
{
  "artifact_path": "candidate/generation_general/gen-general-001.png",
  "sha256": "<lowercase sha256>",
  "width": 1024,
  "height": 1024,
  "pipeline_finite": true,
  "blank_detected": false,
  "metrics": {
    "prompt_alignment": 0.0
  }
}
```

Required metrics are derived from the frozen category:

| Category | Required metrics |
|---|---|
| General, detail, style generation | `prompt_alignment` |
| Typography generation | `prompt_alignment`, `character_error_rate`, `word_error_rate` |
| Single/multi-image Edit | `structural_identity`, `face_identity` |
| Identity-preservation pair | `structural_identity`, `face_identity` |
| Admitted conditional inpaint | `protected_pixel_leakage` |

Evaluator model, revision, implementation, preprocessing, and direction must
exactly equal `evaluator_identity` from the plan. The compiler consumes those
metrics; it does not silently download an evaluator or infer provenance from a
cache directory.

Human review needs at least 200 stratified pairs, exactly three distinct
pseudonymous raters per pair, coverage of every active category, and all 50
identity pairs. Each rating is:

```json
{
  "pair_id": "identity-001",
  "rater_id": "opaque-rater-17",
  "candidate_slot": "B",
  "severe_defect": false,
  "identity_failure": false
}
```

`identity_failure` is Boolean for designated identity pairs and `null` for all
other pairs. Do not put a name, email address, or other personal identifier in
`rater_id`.

## 3. Compile the report

```powershell
uv run --project reference/image/qwen --extra quality python `
  reference/image/qwen/evaluate_quality_suite.py admit `
  --results .codex-tmp/image-quality/q4-results.json `
  --artifact-root .codex-tmp/image-quality/artifacts `
  --output benchmark-results/image/quality/qwen-image-q4_k_m.json
```

The command writes failed reports as evidence and exits `2` when a quality gate
fails. Invalid or incomplete inputs exit `1` without manufacturing an
admission report. A complete pass exits `0`.

For the first three relative gates, the compiler uses the frozen PCG64 seed and
10,000 resamples, resets the generator for each ordered metric vector, computes
the mean paired degradation, and selects the one-sided 95% quantile with
NumPy's conservative `higher` method. Human severe-defect and identity-failure
rates use majority vote and a one-sided 95% Wilson upper bound. Absolute floors
are checked for both BF16 and the candidate.
