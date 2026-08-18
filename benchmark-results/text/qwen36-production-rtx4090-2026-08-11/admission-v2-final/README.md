# Qwen3.6-27B DFlash production qualification

Date: 2026-08-11  
Status: **FAILED ADMISSION — experimental and disabled by default**

This bundle qualifies one pinned tuple: Qwen3.6-27B Q4_K_S target weights,
the Qwen3.6 DFlash Q8_0 draft, XRT CUDA resident execution, F32 KV cache, and
one NVIDIA GeForce RTX 4090. It records strong performance and broad functional
coverage, but it does not justify a production guarantee.

## Decision

Do not enable this DFlash profile by default or advertise a throughput SLA.
Deterministic target parity, long-context retrieval, API behavior, concurrency,
soak, lifecycle, CPU fallback, and physical CUDA checks passed. Production
admission is blocked by a real non-thinking quality failure, unsupported
quantized KV modes for this hybrid CUDA path, strict lint debt, the workspace
Rust-version mismatch, and absence of cross-hardware qualification.

Human review is not requested yet. Automated blockers should be repaired and
the complete suite rerun first; human output review is the next gate only after
the automated report is green.

## Pinned artifacts and runtime

| Item | Identity |
| --- | --- |
| Target | `Qwen3.6-27B-Q4_K_S.gguf`, SHA-256 `a5ef62184c1729c38c9565b502303ac88e2fad3b1c3c6aa430d9e273bdd7f917` |
| DFlash draft | `dflash-draft-3.6-q8_0-rope10m.gguf`, SHA-256 `3612295e4928167eb84512b8b78983ab6ade6efb18bf122d5199c769556e1a1a` |
| Qualified CLI | SHA-256 `a36903e8550123319bb6ef253a49a37a82257019fa869febe3d0bfd608250c83` |
| Qualified server | SHA-256 `a849082f9c3d57438df4449cf675a2743d33162f00b6385280d55e7d9b9882cb` |
| Rust/Cargo source manifest used by the binaries | SHA-256 `aa08a6cd2bf89dd699673fae6da8010a32b415ec76d25d15572f227b8c2a19d7` |
| Hardware | One NVIDIA GeForce RTX 4090, 25,250,627,584 reported bytes of VRAM |
| Memory profile | `XRT_GPU_MEMORY_FRACTION=.94`, `XRT_GPU_RESERVED_MB=1024`, `XRT_GPU_KV_FRACTION=.55` |
| DFlash schedule | Depth 15, serial projections (`XRT_CUDA_DFLASH_PARALLEL_PROJECTIONS=0`) |

Artifact origins, revisions, licenses, sizes, and conversion details are in
[`artifact-provenance.txt`](remote-complete/admission-v2-final/metadata/artifact-provenance.txt).
The source archive identity is recorded in [`final-identities.txt`](metadata/final-identities.txt).

## Evidence matrix

| Gate | Result | Evidence |
| --- | --- | --- |
| Deterministic greedy parity | PASS | 12/12 cases; exact generated-token parity; 36 measured samples per arm after one warmup repetition per case |
| Performance | PASS as a measurement, not an SLA | Target mean 32.5407 tok/s; candidate mean 126.0533, median 105.6976, token-weighted aggregate 93.4740 tok/s; mean-arm ratio 3.8737x |
| F32 long context | PASS through measured limit | 6/6 retrieval cases, exact parity, maximum 7,776 actual prompt tokens; candidate 8K-case decode 19.8595 tok/s and prefill 687,816 ms |
| Non-thinking quality | **FAIL** | Target and candidate returned `218.4` instead of required `259.2` three times; candidate remained token-identical to target |
| Thinking API quality | PASS | Returned `259.2`; 900 completion tokens; `finish_reason=stop`; required answer present at the end |
| Multi-turn CLI | PASS | 3/3 cases, 6/6 samples per arm, validators green, exact target/candidate parity |
| Seeded stochastic behavior | PASS by safe fallback | 4/4 cases and 8/8 samples exact; candidate drafted/accepted zero tokens and used target-only fallback |
| Concurrency | PASS at tested bounds | c1: 3/3, 113.2165 mean tok/s/request; c2: 3/3, 58.1682 mean tok/s/request |
| OpenAI-compatible service | PASS | 941.54-second suite: streaming/non-streaming chat and completions, multi-turn, cancellation, overload, soak, lifecycle, invalid request, discovery |
| API long context | PASS | HTTP 200, exact `XRT-08192-05-PASS`, 7,773 prompt tokens, 713.46 seconds |
| Overload and cancellation | PASS | Eight simultaneous requests produced five successes and three expected 429s; aborted stream drained to zero active sessions |
| Soak | PASS | 100/100 requests, one output hash, 0.7269 s mean, 0.7328 s p95, zero RSS growth, negative observed GPU delta |
| Lifecycle | PASS | Unload 200, unavailable probe 503, reload 200, post-reload inference 200 with `RELOAD_OK` |
| Memory ceiling | PASS for tested profile | Peak tracked arena 22,578,085,068 bytes below 22,661,848,044-byte budget |
| CPU fallback | PASS minimally | Real 27B artifact, CPU backend, 16 prompt tokens, one output token (`OK`), 7,850.9 ms total, about 16.74 GB RSS |
| Physical CUDA correctness | PASS after fixture repair | Q4/Q8 Marlin, DFlash attention, batched Qwen verification attention, fused DeltaNet, F32/Q8/KQ4-VQ8 KV kernels, graph replay, and Q4 verification checks passed |
| Quantized KV for hybrid CUDA | **FAIL / unsupported** | Both `q8` and `kq4_vq8` were rejected for all six context cases; this does not invalidate Q4_K_S weight support |
| Workspace tests | PASS | `cargo test --workspace` passed; real-bundle/GPU fixtures that require explicit hardware remain ignored by the ordinary run |
| Rust 1.76 text packages | PASS | `xrt-runtime`, `xrt-cli`, and `xrt-server` check with Rust 1.76 |
| Workspace Rust 1.76 | **FAIL** | `xrt-python` declares Rust 1.83 / uses a PyO3 dependency requiring 1.83 |
| Strict Clippy | **FAIL** | Existing warnings in the CUDA/kernel dependency path and runtime all-target checks prevent `-D warnings` admission |

## Performance interpretation

The primary result is [`serial-summary.json`](performance-final/serial-summary.json).
It uses equal weighting across the 12 prompt cases, which is why the 126.0533
tok/s arithmetic mean differs from the 93.4740 tok/s token-weighted aggregate.
The 95% normal half-width is 27.7765 tok/s and the observed range is
38.6308--356.2281 tok/s. The maximum is a short, highly accepted prompt result;
it is not an average, minimum, service capacity figure, or guarantee.

The parallel-projection arm averaged 124.8784 tok/s and had a lower 90.6121
tok/s token-weighted aggregate, so the serial schedule is the retained profile.
Both candidates accepted 1,500 of 6,888 drafted tokens (21.777%).

## Context, sessions, and service behavior

The F32 context sweep in [`tuned-context-final`](tuned-context-final/) passed
all six needle-retrieval cases at approximately 512, 768, 1K, 2K, 4K, and 8K
context sizes with exact target/candidate tokens. At the largest case the target
decoded at 6.0728 tok/s and the candidate at 19.8595 tok/s. Prefill remained the
dominant cost at roughly 685--688 seconds.

The final service record is [`service.json`](api-serial-final/service.json).
It passed 100 soak requests without RSS growth, exercised queue rejection and
cancellation drain, and survived unload/reload. Prefix-cache support was
enabled, but its multi-turn snapshot recorded zero entries and zero hits; this
run therefore proves correct multi-turn responses, not prompt-cache speedup.

## Blocking failures

1. The pinned non-thinking target itself fails the arithmetic quality case.
   DFlash is exactly target-equivalent, but target equivalence is insufficient
   when the target answer is wrong. The thinking-enabled API profile answers
   correctly, so the next work item is to define and qualify the intended
   production thinking/default-generation policy, then rerun the corpus.
2. Qwen3.6's hybrid recurrent CUDA path currently requires F32 KV. The runtime
   rejects `q8` and `kq4_vq8` explicitly. Either implement these modes for the
   hybrid path and rerun, or narrow the advertised profile to F32 only.
3. Make strict Clippy clean for the release scope and reconcile the documented
   Rust 1.76 baseline with `xrt-python`'s Rust 1.83 requirement.
4. Repeat correctness, soak, context, and memory-pressure testing on the
   intended hardware matrix and complete security/packaging admission. One
   community-cloud RTX 4090 run cannot establish a production SLA.

## Reproduction

From the repository root on a CUDA host with the two pinned artifacts:

```bash
cargo build --release -p xrt-cli -p xrt-server --features cuda
XRT_PRODUCTION_REPETITIONS=4 \
  XRT_TUNED_MAX_CONTEXT=8192 \
  XRT_QUANTIZED_MAX_CONTEXT=8192 \
  bash scripts/benchmark-qwen36-production.sh \
    /path/to/Qwen3.6-27B-Q4_K_S.gguf \
    /path/to/dflash-draft-3.6-q8_0-rope10m.gguf \
    benchmark-results/text/qwen36-production-rerun all
```

The script now exits nonzero if any requested validator fails. Exact memory,
scheduler, DFlash, and server settings are captured in the per-run metadata and
orchestrator logs. The remote evidence is preserved both unpacked under
[`remote-complete`](remote-complete/) and as the SHA-verified
[`remote-admission-v2-final.tar.gz`](remote-admission-v2-final.tar.gz).

## Evidence authority

Use the final-suffixed records for decisions:

- [`performance-final`](performance-final/) for paired throughput and parity;
- [`tuned-context-final`](tuned-context-final/) for F32 context;
- [`api-serial-final`](api-serial-final/) for the OpenAI-compatible service;
- [`production-profile-final`](production-profile-final/) for multi-turn,
  sampling fallback, concurrency, CPU, and quantized-KV results;
- [`quality-serial-final`](remote-complete/admission-v2-final/quality-serial-final/)
  for the final non-thinking quality failure; and
- [`cuda-correctness.log`](remote-complete/admission-v2-final/hardware-tests/cuda-correctness.log)
  for physical CUDA regressions and their final repaired reruns.

Earlier directories are retained for audit history and must not override these
final records. In particular, a staged production-profile run predating the
final harness hardening printed a completed phase while its quantized-KV
validators failed. The validators are authoritative, and the current harness
propagates such failures through its exit status.
