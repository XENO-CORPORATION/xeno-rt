# XENO RT — `xrt-audio` admission gates

- **Scope:** `xrt-audio`
- **Status:** Gate definitions + first measured run
- **Last updated:** 2026-08-28
- **Applies to:** every audio model adapter before it is advertised

`RUNTIME_DOMAINS.md` § *Support and admission* lists nine requirements and then
says metrics are modality-specific: text uses tokens/second and time-to-first
token, image uses seconds/image and time-to-first-preview, and *"video and audio
will define frame-, sample-, duration- and streaming-aware gates before their
first adapters are admitted."* This file is that definition for audio.

## Why audio needs its own metrics

Neither existing set transfers, and the reason is not cosmetic. **The unit of
audio work is a DURATION.** Tokens/second says nothing about whether a
40-minute interview is practical, and seconds-per-item is meaningless when items
range from a 2-second sting to a feature film. So the throughput gate is
**duration-relative** (real-time factor), and the latency gate is the streaming
analogue of time-to-first-token: how long before a caller can show a user
anything at all.

Quality also has an established measure that neither text nor image generation
uses. Transcription is scored by **word error rate** against a reference
transcript — a metric with decades of published baselines, which is what makes
a threshold defensible instead of invented.

## The gates

| Gate | Metric | Threshold | Why this number |
|---|---|---|---|
| `reference_correctness` | WER on a labelled corpus | **≤ 10%** | OpenAI report ~5% for whisper-base on LibriSpeech test-clean using their own text normaliser. Ours is plainer (below), which inflates WER by a few points, so the bar sits above the published figure rather than at it. |
| `throughput` | real-time factor (audio secs ÷ wall secs) | **≥ 3.0×** | 1.0× is break-even — below it, a 40-minute interview takes longer than 40 minutes and the feature is unusable. 3× leaves headroom for a loaded machine. |
| `first_segment_latency` | ms to the first returned segment | **≤ 5000 ms** | The streaming analogue of time-to-first-token. Past ~5 s a caller cannot show progress and the request reads as hung. |
| `determinism` | same input, same backend, twice | **byte-identical** | The policy allows determinism *or* a documented reproducibility contract. Greedy decoding over a fixed graph should be reproducible; if it is not, cached transcripts and subtitle files become unstable for reasons a user cannot see. |
| `memory` | peak working set | **≤ 2 GB** for a base-class model | Must fit alongside a creative app that is already holding video frames. Larger model classes get their own bound; this one is not a global ceiling. |
| `cpu_fallback` | every advertised capability runs CPU-only | **required** | `RUNTIME_DOMAINS.md` ABSOLUTE RULE 4. CUDA is optional, always. |

🔴 **Thresholds are set from published baselines BEFORE measuring, never fitted
to the run.** Fitting a threshold to an observation turns a gate into a
thermometer — it will report whatever the code does and call it a pass.

## First measured run — whisper-base, 2026-08-28

CPU only, ONNX Runtime 1.26, `whisper-base` fp32 as published to
`updates.xenostudio.ai/models`. Corpus: **73 clips / 481 s** of LibriSpeech
validation (`hf-internal-testing/librispeech_asr_dummy`) with reference
transcripts.

| Gate | Measured | Verdict |
|---|---|---|
| `reference_correctness` | **9.50%** WER | PASS |
| `throughput` | **13.6×** realtime | PASS |
| `first_segment_latency` | **878 ms** | PASS |
| `determinism` | identical across 2 runs | PASS |
| `memory` | **998 MB** peak working set | PASS |
| `cpu_fallback` | exercised — every number above is CPU-only | PASS |

Reproduce:

```bash
ORT_DYLIB_PATH=<onnxruntime.dll|.so> \
cargo run --release -p xrt-audio --example admission -- <model-dir> <corpus/wav> <corpus/refs.json>
```

Peak memory is sampled by the caller rather than self-reported, because reading
peak RSS in-process needs a platform crate this benchmark does not otherwise
want. The run above was measured by polling `tasklist` once a second.

## ⚠️ Read the WER number correctly

**9.50% against a 10% threshold is a thin margin, and it is dominated by the
normaliser, not by the model.** The harness lowercases and strips everything
that is not alphanumeric. It does **not** fold contractions or spell out
numerals, so:

| reference | output | scored |
|---|---|---|
| `MISTER QUILTER` | `Mr. Quilter` | **error** |
| `DON'T` | `do not` | **error** |
| `FIVE` | `5` | **error** |

Whisper's own `EnglishTextNormalizer` handles all three, which is why published
figures are lower. The measurement here is therefore **pessimistic**, and that
is the correct direction for a gate to be wrong in — it can fail a working
model, never pass a broken one.

**Exit condition for the margin:** port an equivalent normaliser and re-measure.
Do NOT raise the threshold. A threshold moved to accommodate a result is not a
gate, and the next person to read 10% would have no way to know it had been
adjusted to fit.

## ⚠️ What this run does NOT establish

- **Corpus size.** 73 clips of clean, read, American-accented speech is enough to
  catch a broken adapter and nowhere near enough to characterise accuracy.
  Accented, noisy, overlapping and non-English speech are all unmeasured.
- **Only one model class.** whisper-base. small/medium/large-v3 each need their
  own run; `memory` in particular does not extrapolate.
- **No GPU path measured.** CUDA is optional and untested here.
- **Cancellation and long-run cleanup** — policy item 8 is now only PARTLY
  closed. Since this run the route exists, and its API shape, three error paths
  and concurrency ARE tested against the running server: two simultaneous
  requests returned 200 in 0.61 s and 1.23 s with identical bodies, which is
  serialisation rather than a race. What remains untested is a client
  disconnecting mid-inference, and memory behaviour across a long run of many
  requests rather than one batch.
- **Policy item 9** — clean-checkout CI, packaging, installation and rollback are
  untouched. Together with the above, this is why the `transcription` feature
  defaults to OFF.

## Standing rule

Every gate must REPORT. A gate that did not run exits non-zero exactly like one
that failed. This ecosystem shipped a smoke harness that broke OPEN — an unrun
gate reported `undefined`, was skipped, and the run still printed OK — and the
harness here is written so that cannot happen: gates start as `None` and any
`None` at the end fails the run.
