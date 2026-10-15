# Drama 3 ISG follow-up, 2026-10-15

**Still experimental and unadmitted.** This directory preserves failures as well
as successful scoped checks. No waveform, model weights, private signing key or
capability bearer token is committed.

| Receipt | Meaning |
|---|---|
| `backbone-bf16-current-failed.json` | Fresh CPU composed-layer comparison: 52/80 pass, maximum relative RMS 0.576%; strict gate remains failed |
| `tower-bf16-current-failed.json`, `tower-bf16-cuda-current-failed.json` | CPU 7/26 and CUDA 12/26 pass; strict gates remain failed |
| `tower-isolated-operators.json` | 806/806 captured-input operator checks; not composition or autoregressive parity |
| `gdn-layer1-isolated-operators.json` | 98/98 checks including recurrent state after explicit native value-major/upstream key-major conversion |
| `criterion-fused-experts.json` | Actual RTX 4090 warm synthetic expert comparison, with correctness and allocation cleanup guards; approximately 0.256 ms fused versus 2.620 ms unfused |
| `verified-narration/` | Eight English categories times two seeds: 16/16 jobs pass strict normalized word verification; 14 use one attempt and two use two attempts; missing-checker, wrong-script and second-attempt cancellation controls pass |
| `signed-development-candidate.json` | Real local executable bundle signed with an ephemeral TEST key, installed and rehashed; modified signature/payload refused; no production signer or publication |
| `signed-patched-candidate.json` | Separate test-signed candidate after the rustls 0.23.46 security patch |
| `patched-verified-smoke/` | Patched installed candidate passes a verified narration take and repeats the missing-checker, wrong-script and second-attempt cancellation controls |
| `patched-installed-lifecycle/` | Runs that patched installed executable: native module audit, busy/cancel/deadline/disconnect/frame-limit/drain/unload, idempotency/conflict, active/queued cancellation, persisted WAV readback and interrupted running/queued jobs after process termination |
| `production-source-continuity.json` | Production Rust/CUDA source is unchanged between the full narration candidate and the TLS-patched candidate; examples, benchmarks and dependency patch qualification remain separate |
| `checks.json` | Local workspace/native/bundle/MSRV checks; the inherited transitive paste maintenance advisory remains visible |

The full narration candidate server is SHA-256
`76fc1d834abfc64dfec22f63086da18ffd45264ec4de9a1a29b33d49604ccc2d`.
The TLS-patched candidate is
`42cd408d27e796cd1497fe625017f41f8f9eecf0212263eabaf8e1023f349486`.
Installed paths and independent expected artifact checksums are retained in the
lifecycle build/start evidence outside Git. The original lifecycle runner's
summary conservatively says "not installed package"; its `build.json` records
`source_built: false` and the expected checksum, and its input executable is the
verified bundle-cache copy. This establishes local installed-artifact behavior,
not a public download, production signing authorization or model admission.

The FP32 waveform remains
`99a3bd5b0d8d5b96c1db12672461eaf4fe976da45a3d92f003abc60db7927d30`
after cancellation and fresh-process restart, matching the earlier baseline.
Raw native requests load no Python, Torch or ONNX module; requests with optional
word verification intentionally use the existing Rust/ONNX CPU Whisper checker.

Timings come from a shared workstation. These runs do not qualify MOS, emotion,
speaker identity, multilingual or arbitrary-length narration, exclusive hardware
peaks, real-time speech or production deployment. The generic IP quick scan is
retained at `X:/tmp/drama3-isg-compliance-20261015`; it is advisory and includes
Python reference lockfile findings and a first-party proprietary-license finding.
The full Rust graph passes license/bans/source checks. Advisory scanning still
fails on inherited `paste 1.0.15` (RUSTSEC-2024-0436); four rustls vulnerabilities
were removed by the 0.23.46 update, without adding an ignore.
