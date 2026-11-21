# Qwen3.6 Marlin verifier scheduler screen - RTX 4090 - 2026-08-10

This directory records same-host scheduler experiments for the pinned
Qwen3.6-27B Q4_K_S MTP benchmark. See `docs/QWEN36_MTP_ADMISSION.md` for the
admission interpretation and exact reproduction command.

The retained candidate uses three Marlin pipeline stages with 27 KiB dynamic
shared memory for the 64-column tile and 42 KiB for the 128-column tile. The
function opt-in is configured once when the CUDA module loads.

The long paired comparison designated the first three of 23 repetitions as
warmup and retained 20 samples:

| Candidate | Mean decode | Sample SD | 95% CI half-width | Mean verifier | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Two-stage control | 110.8178 tok/s | 0.0489 | 0.0229 | 465.576 ms | Superseded for this tuple |
| Three-stage candidate | 111.0078 tok/s | 0.0572 | 0.0268 | 464.718 ms | Retained |

The candidate improved mean decode by 0.1715% and reduced mean verification
time by 0.858 ms. All retained samples generated 64 tokens, drafted 68 tokens,
accepted 55, reported no error, and emitted the same deterministic preview.
This is a small kernel improvement, not the 150 tok/s objective and not a
production-admission claim.

Primary evidence:

- `stage2-paired-23.json`, SHA-256
  `947a595832b20812e5cf03f3b78f622f2eee594ede45118fcf8eff7e94556664`;
- `stage3-paired-23.json`, SHA-256
  `816178d7f148af5900b5485af07dc1e4a7f0cb3584218ffee4bb0d9606838cdc`;
- final generated Marlin PTX, SHA-256
  `81040b0d8f4513320a7f2b4d20acfa6b9597246753ea74cd40ccb17344af9283`.

Rejected screens in this directory cover four-stage Marlin, grouped gate/up,
the K64/N128 tile, forced K128/N64 tiling, and exact-minimum shared-memory
reservations. The grouped gate/up low-level packing test passed, but its model
screen lost the existing two-stream projection overlap.
