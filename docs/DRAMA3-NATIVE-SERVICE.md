# Experimental native Drama 3 service

This is an opt-in qualification interface for the native Rust adapter. It is
**not a production model admission**. It does not add Drama 3 to `/v1/models`,
change `/v1/audio/speech`, or claim the existing word-check pipeline ran.
The first qualification target is offline narration on a 24 GB NVIDIA GPU.
Applications own script segmentation, timelines and editing.

## Build and configure

```powershell
cargo build --release --locked -p xrt-server --features drama3-native,cuda
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
are rejected. No automatic chunking, retries, speech editing or speed conversion
is performed. Sampling defaults match the pinned Fish SDK; the native RNG is not
Torch's same-seed RNG.

Successful WAV responses include `Cache-Control: no-store`,
`X-Xrt-Admitted: false`, `X-Xrt-Numerics`, `X-Xrt-Frames`,
`X-Xrt-Clamped-Residual-Codes` and `X-Xrt-Generation-Seconds`. Generation time
excludes lazy model loading, reference encoding and HTTP transfer.

Errors distinguish unconfigured model (428), invalid input/reference (400),
another active worker (429), unload conflict (409), cancellation/deadline (408),
empty or incomplete speech (422), drain (503), and native worker/load failure
(500). No word-check success is implied by HTTP 200.
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

These inference routes are not the canonical ecosystem capability binding.
Under the ecosystem's `XENO AGENT CAPABILITY - SPEC.md`,
that binding is **absent**, not conformant: there is no generated capability
registry/catalogue, per-launch capability authentication, or native job handle
adapter. The existing `/v1/runtime/capabilities` reports text token-count controls
and must not be described as this missing registry.

Owner: `xeno-rt`, with consumer UI in Hub/creative apps. Exit: a generated,
authenticated capability binding and catalogue, model-free transport/parity
gates, and native work connected to the shared bounded job lifecycle. The HTTP
documentation here is an API contract, not a hand-written replacement catalogue.
No new application UI or agent runtime is introduced in the compute layer.

Built with Fish Audio. The [Fish Audio Community License](../reference/audio/drama3/LICENSE)
and pinned attribution apply; no public redistribution or hosted-service
permission is asserted by this development interface.
