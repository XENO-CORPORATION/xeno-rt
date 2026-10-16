# Experimental native Drama 3 service

This is an opt-in qualification interface for the native Rust adapter. It is
**not a production model admission**. It does not add Drama 3 to `/v1/models`,
or change `/v1/audio/speech`. Word verification runs only when explicitly
requested and enabled by the operator.
The first qualification target is offline narration on a 24 GB NVIDIA GPU.
Applications own script segmentation, timelines and editing.

## Build and configure

```powershell
cargo build --release --locked -p xrt-server --features transcription,agent-capabilities,cuda
./target/release/xrt-server.exe --drama3-config ./native-audio.json --port 3000
```

Omit `cuda` and both GPU budgets for CPU execution. The default server build
does not include these routes or the native model dependencies. CUDA libraries
must be installed by the operator; this interface downloads nothing.

Example operator configuration:

```json
{
  "model_bundle": "E:/xeno-models/bundles/fish-drama3-native-dev/2c67787970d622f410f1e606ea9f6cdd1c8775934ced465ded211b78540ed3e3",
  "codec_bundle": "E:/xeno-models/drama3-codec-native-v3-20261014",
  "numerics": "fp32_reference",
  "word_check_enabled": false,
  "expert_mib": 13312,
  "dense_mib": 5120,
  "host_gib": 64,
  "codec_mib": 3072,
  "max_frames": 256
}
```

Paths are resolved relative to the configuration file. HTTP clients cannot
choose filesystem paths. Configuration rejects unknown fields and resource
limits that exceed the adapter's bounds. Weights load lazily, under shared audio
admission, on the first valid speech request. GPU execution uses the same arena
as text, image and existing audio, and checks live device headroom separately.
The model remains loaded until explicit unload or process exit.

On Windows, load and inference also check live physical RAM and available
system commit. The commit forecast includes device allocations not yet resident
in the expert cache and a 2 GiB margin. A late headroom refusal releases the
service's native owners. This observation cannot reserve memory against other
processes, and cross-platform host/cgroup admission remains unqualified.
Native immutable uploads borrow their mapped source synchronously; they do not
retain cudarc's extra host vector for each device buffer.

The checkpoint must be the exact managed bundle pinned in
[provenance](../reference/audio/drama3/provenance.json). Convert the verified DAC
checkpoint using `scripts/convert-drama3-codec.py --include-encoder` for voice
references. The runtime accepts only numerically qualified conversion hashes;
changing a checksum in adjacent metadata cannot establish provenance. The
converted codec currently remains an operator-provisioned development artifact.
Its installed-package and migration contracts are still open.

`fp32_reference` retains encoded BF16/FP8 weights with FP32 activations and
recurrent state. `bf16_w8a8` is a separate experimental precision contract:
current strict upstream layer checks fail. Intelligible output does not establish
its numerical parity or admit it for production.

## HTTP contract

| Method and route | Result |
|---|---|
| `GET /v1/xrt/audio/native/status` | Configuration/load/activity, progress stage and units, precision, frame cap; always `admitted: false` |
| `POST /v1/xrt/audio/native/speech` | One complete 44.1 kHz float WAV, or a typed error; no partial audio on frame exhaustion |
| `POST /v1/xrt/audio/native/jobs` | Submit native speech to the existing durable audio ledger; requires `Idempotency-Key` |
| `GET /v1/audio/jobs/{id}` | Read typed job status, progress stage, units, one-based attempt and persisted result metadata |
| `GET /v1/audio/jobs/{id}/result` | Read a completed native WAV, or the existing JSON result for a legacy job |
| `POST /v1/audio/jobs/{id}/cancel` | Cancel a queued or running job; result publication checks cancellation atomically |
| `DELETE /v1/audio/jobs/{id}` | Delete a terminal job and its private result |
| `POST /v1/xrt/audio/native/cancel` | Signal the active native request; does not cancel another audio adapter |
| `POST /v1/xrt/audio/native/unload` | Release native model ownership and GPU leases; refuses while any audio worker is active |
| `POST /v1/audio/drain` | Existing shared audio drain also cancels native work and refuses new requests until restart |

Speech request:

```json
{
  "input": "The evidence remains clear.",
  "seed": 42,
  "max_frames": 128,
  "temperature": 0.7,
  "top_p": 0.7,
  "top_k": 30,
  "timeout_seconds": 1800
}
```

Optional `references` contains up to two objects with `transcript` and `wav_b64`.
Only inline WAV is accepted; reference encoding requires 44.1 kHz audio of at
most 30 seconds. Reference speaker identity and multi-speaker quality remain
unadmitted. A correct transcript alone cannot establish identity or emotion.

Input is limited to 8,000 characters, references to 8 MiB of encoded WAV each,
and total JSON body to 20 MiB. The frame cap is bounded by operator admission.
Structural prompt-token injection, unknown fields and invalid sampling settings
are rejected. No automatic chunking, speech editing or speed conversion
is performed. Raw speech has one attempt. Sampling defaults match the pinned Fish SDK; the native RNG is not
Torch's same-seed RNG.

### Optional immutable-word verification

Set `word_check_enabled: true` in the operator configuration and explicitly
provision the managed `whisper-base` set before offline use. Add this request
field to keep spoken words separate from emotion/speaker tags:

```json
"word_check": {"spoken_text": "The evidence remains clear.", "language": "en", "max_attempts": 2}
```

`max_attempts` is bounded to 1–3. Each attempt starts with fresh inference state,
using the supplied seed plus its zero-based attempt index. Only incomplete/empty
generation or a lexical mismatch can retry. Cancellation, invalid requests and
resource failures return directly. Independent CPU Whisper runs through the
existing Rust audio runtime after native synthesis; the Drama 3 graph and codec
remain Rust-native. This optional checker loads ONNX Runtime. Native TTS module
audits that exclude ONNX apply to requests without word verification.

`language` defaults to `en`. Explicit checker hints are `en`, `de`, `es`, `fr`,
`zh`, `ja`, `pt`, `it`, `ko` and `ru`; status exposes `word_check_languages`.
These are checker controls, not admitted Drama 3 language tiers or detected
languages. The checker uses the matching Whisper language token, retains NFC
accents and compares CJK/Hangul characters while preserving Latin/digit runs.
Homophones, character substitutions, negation changes and lost decimal points
are rejected. A checker mismatch is not proof that the synthesis, rather than
ASR, made the error; listening remains a separate gate.

The checker requires zero normalized word errors; it does not make fuzzy name
substitutions, infer speaker identity, rate emotion or publish invented word
timings. Missing or corrupt checker state is refused before loading Drama 3.
There is no network download or repair on this path. Jobs bind the installed
checker digest at submission and reject a changed checker before execution.
Managed file verification and a retained read lease protect the loaded checker.
Exhausted attempts return `422 quality_rejected` without publishing audio.

Successful WAV responses include `Cache-Control: no-store`,
`X-Xrt-Admitted: false`, `X-Xrt-Numerics`, `X-Xrt-Frames`,
`X-Xrt-Clamped-Residual-Codes`, `X-Xrt-Generation-Seconds`, `X-Xrt-Attempts`,
`X-Xrt-Word-Check`, `X-Xrt-Word-Check-Language`, `X-Xrt-Word-Check-Digest` and
`X-Xrt-Total-Seconds`.
Generation time covers the accepted take; total time covers loading, references,
all attempts and verification, excluding HTTP transfer. Durable results retain
these metadata headers.

Errors distinguish unconfigured model (428), invalid input/reference (400),
another active worker (429), unload conflict (409), cancellation/deadline (408),
empty or incomplete speech (422), drain (503), and native worker/load failure
(500). `X-Xrt-Word-Check: verified` is required to assert lexical verification;
an ordinary HTTP 200 does not imply it. Missing checker state returns
`503 word_checker_unavailable`.
Host headroom refusal is 503 with error code `host_headroom`.

Disconnects and deadlines signal cancellation. The blocking worker retains
admission and model ownership until it actually returns; an HTTP disconnect
cannot free GPU resources underneath it. Hash verification and device loading
currently finish their synchronous phase before observing cancellation.
Cancellation latency during those phases remains an admission gap.

## Reproduce service qualification

```powershell
python scripts/qualify-drama3-native-server.py --server ./target/release/xrt-server.exe --config ./native-audio.json --out ./new-native-service-proof
```

The runner starts and stops only its own child, disables proxy egress, records
the executable/config hashes, builds from source and verifies its source digest
did not change during the build, exercises real HTTP generation and resource
cleanup, and compares waveform hashes across cancellation and process restart.
Use `--cpu-only` for a CPU build. `--artifact-sha256 <expected-digest>` explicitly
selects an existing artifact instead of rebuilding; the supplied digest must
match. `--expect-host-refusal` qualifies refusal of an intentionally oversized
host plan without treating that as a successful speech run.
The runner does not provision models or prove a signed installed package. See the
[ISG ledger](DRAMA3-NATIVE-ISG.md) for open numerical, quality, performance,
licensing and delivery gates.

## Automation contract measurement, 2026-10-15

The native HTTP surface is wired and runs unattended through the same `Engine`
that library consumers use. No UI selection, dialog, renderer or agent SDK is
in its inference path. Status reads actual owner/activity state; progress reports
`stage` and `units` (prompt rows for `prefill`, acoustic frames for `decode` and
`codec`, reference index for `encoding_reference`). `cancel_requested: true`
means a signal was issued, not that the blocking worker has exited. Unload refuses
active work and is observable through status and shared GPU accounting.

The optional `agent-capabilities` build now exposes the canonical native-audio
binding on the existing loopback listener when `--agent-capabilities` is set.
Its seven generated actions and argument/result contracts are documented in
[AGENT-CAPABILITIES.md](AGENT-CAPABILITIES.md). The process rotates a bearer token
each launch and publishes it only in the owner-restricted shared dirs discovery
file `run/platform/capability-servers/rt.json`. Browser Origin/Referer requests are
refused. Do not copy discovery tokens into logs or agent prompts.

The binding uses the same native handlers, shared audio admission and SQLite/WAL
job ledger. Completed WAVs survive restart; previously queued/running jobs become
`interrupted` and are not automatically replayed. Legacy JSON jobs remain
compatible. The ledger shares its existing capacity and storage limits across
both kinds. Capability audit rows contain action/class/input digest/status,
without spoken scripts, tokens or audio, and are bounded to 10,000 rows.

Catalogue drift, authentication, transport and job migration are tested. This
is native-audio coverage only: text, image, voice-library and provisioning
capabilities remain outside the initial registry. Local compute is unbilled;
cloud metering and whole-product parity are not implemented. No new application
UI or agent runtime is introduced in the compute layer.

To verify narration through an installed candidate, use
`scripts/qualify-drama3-verified-narration.py`; supply the executable checksum
from an independently inspected bundle inventory. The helper never builds,
downloads or publishes weights.

Built with Fish Audio. The [Fish Audio Community License](../reference/audio/drama3/LICENSE)
and pinned attribution apply; no public redistribution or hosted-service
permission is asserted by this development interface.
