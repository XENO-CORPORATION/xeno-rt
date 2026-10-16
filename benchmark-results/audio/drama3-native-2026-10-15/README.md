# Native Drama 3 qualification, 2026-10-15

**Experimental and unadmitted.** Receipts have different scopes; a green service
run does not close failed numerical, voice-quality or installed-package gates.

| Evidence | What it establishes |
|---|---|
| `backbone-fp32-upstream.json`, `tower-fp32-upstream.json` | 80 + 26 trained-layer checks pass against actual hash-pinned upstream code; two positions, with each layer given its native input |
| `*-bf16-upstream-failed.json` | Strict BF16/W8A8 layer comparisons still fail; 37/80 backbone and 1/26 tower checks pass |
| `dense-bf16-diagnostics.json` | Per-operator BF16 errors; diagnostic, not a replacement admission threshold |
| `sdk-defaults/` | FP32 control failures at corrected SDK sampling defaults, retained as failures |
| `bf16-development/`, `bf16-*-asr.json` | Intelligible plain and two-reference speech; unreferenced speaker case still exhausts frames; no voice identity/style qualification |
| `service-initial/` | First HTTP cancellation, frame-limit, unload and restart checks; before the later host-memory work |
| `service-before-host-fix-failed/` | Real allocation abort under Windows commit pressure; not a passing gate |
| `service-current/` | Expanded real HTTP deadline/disconnect/cancel/frame-limit/restart run, complete WAVs and resource release; executable hash in the start receipts |
| `service-source-built/` | Repeated expanded lifecycle pass, with the runner building and binding its source digest; same independently transcribed WAV hashes and an owned-process module audit |
| `host-headroom/` | Intentionally oversized host plan returns typed HTTP 503 and leaves no native resources allocated |
| `clean-cpu-sdk-defaults/` | Complete CPU-only speech from a clean clone, at SDK defaults; byte-identical to the CUDA WAV, zero GPU admission, no external inference engine or CUDA module loaded |
| `service-word-fidelity.json` | Both service WAVs transcribe to the intended English words, ignoring punctuation and explicitly mapping seventeen to 17 |
| `service-cuda-modules.json` | CUDA driver/NVRTC modules observed in the owned native server; no ORT/Python/Torch match |

The FP32 layer reports were produced on October 14 and retained here with the
later precision work. BF16 speech receipts precede the subsequent gated-norm
rounding correction; they cannot qualify the corrected mode. Failed reports
remain failed.

`service-current` was built before its run, but the runner did not yet enforce
source freshness itself. The updated runner builds and binds a digest of its
source inputs before producing a new proof, or checks an explicitly supplied
installed-artifact digest. A signed installed package remains unqualified.
`checks.json` also records the repeated model-free tests from the clean clone;
the Cargo registry cache was reused, while the build directory was new.

All WAVs and weights remain outside the repository under the `X:/tmp/drama3-*`
and managed model locations. Their hashes identify retained outputs. References
use this model's own earlier generated clips, not a third party's voice.

## ISG follow-up receipts

The later [follow-up evidence](isg-follow-up/README.md) records the corrected
W8A8 group scaling and stable BF16 norms, fused expert benchmark, generated
capability/job integration, a 16/16 immutable-word-verified narration matrix,
local test-signed installed candidates and actual installed lifecycle checks.
Strict BF16 fidelity still fails, the production publisher identity remains
unprovisioned, and no model weights or public catalog entry were published.

Timings were collected on a shared workstation and are not controlled product
performance numbers. Private-memory snapshots are not peaks. See the
[admission ledger](../../../docs/DRAMA3-NATIVE-ISG.md) and
[service contract](../../../docs/DRAMA3-NATIVE-SERVICE.md) for remaining work.
