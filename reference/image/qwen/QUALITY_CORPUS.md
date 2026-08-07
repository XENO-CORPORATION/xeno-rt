# Q4_K_M quality corpus operation

This procedure produces admission evidence; it does not enable production
support. Run it only on a dedicated remote CUDA machine. Never apply the
`image-quality` runner label or the
`XRT_DEDICATED_IMAGE_QUALITY_RUNNER=1` machine marker to a developer PC.

## Corpus

Use the manual `Image Quality Corpus Shard` workflow. Each dispatch executes
one `generation|edit` and `bf16|candidate` shard. Supply the matching exact
bundle directory already present on the runner, one persistent artifact root,
and the literal confirmation `RUN_XRT_IMAGE_Q4_CORPUS`.

The default 20-way partition requires 80 successful dispatches: 20 shards for
each of the four role/side combinations. Interrupted dispatches are safe to
repeat with identical inputs. A PNG is reused only when its adjacent
`.png.xrt.json` checkpoint still matches the frozen plan, bundle digest,
request identity, dimensions, and SHA-256.

Do not start human review until all 600 PNGs and the pinned metric export have
passed their automated checks.

## Automated metrics

Use the manual `Image Quality Metrics` workflow against the same persistent
artifact root. Its three stages are independently resumable:

- `prompt-alignment` evaluates all 400 generation outputs with the immutable
  local OpenCLIP snapshot;
- `identity` evaluates all 200 edit/identity outputs against their source
  fixtures with DINOv2 full-frame and fixed face-region embeddings; and
- `ocr` evaluates all 80 typography outputs with the immutable PaddleOCR-VL
  recognition snapshot and no separately downloaded layout detector.

Run `export` only after those checkpoints are complete, or use `all` when the
six-hour job bound is sufficient. The resulting
`metrics/q4-k-m-metrics.json` is still non-human evidence and does not by
itself admit the tier.

## Blinded review

After automated metrics pass, build the offline review package on the corpus
machine:

```powershell
uv run --project reference/image/qwen --frozen python `
  reference/image/qwen/prepare_quality_review.py prepare `
  --plan D:\xrt-image-quality\protocol\q4-k-m-plan.json `
  --artifact-root D:\xrt-image-quality `
  --fixture-root tests\fixtures\image-quality `
  --output D:\xrt-image-quality\review
```

Keep `review/private/mapping.json` from all raters. Give each reviewer the
shared `public/assets` directory and exactly one of `rater-01.html`,
`rater-02.html`, or `rater-03.html`. Each reviewer exports one JSON response.
Compile the three responses without opening the private candidate mapping:

```powershell
uv run --project reference/image/qwen --frozen python `
  reference/image/qwen/prepare_quality_review.py compile `
  --package D:\xrt-image-quality\review `
  --responses rater-01-responses.json rater-02-responses.json rater-03-responses.json `
  --output D:\xrt-image-quality\human-reviews.json
```

The compiler requires every selected pair from each distinct pseudonymous
rater and emits the exact protocol and rating fields consumed by
`evaluate_quality_suite.py admit`.
