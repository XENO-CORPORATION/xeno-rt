# Qwen3.6 greedy admission and verifier-kernel screen

## Outcome

The frozen deterministic-greedy multi-prompt gate passes for the retained
experimental XRT tuple on one RTX 4090. The candidate generated exactly the
same token IDs as target-only execution for every case and repetition.

| Arm | Samples | Mean tok/s | Median | Minimum | Maximum | Accepted / drafted |
|---|---:|---:|---:|---:|---:|---:|
| Target-only | 12 | 33.1115 | 33.2304 | 32.1964 | 33.4841 | n/a |
| MTP depth 8, scalar draft head | 12 | 125.2930 | 120.0580 | 82.1069 | 169.9820 | 631 / 1,009 |
| MTP depth 8, tensor-core draft head | 12 | 129.4882 | 125.0791 | 82.9053 | 172.2343 | 631 / 1,009 |
| Retained: MTP depth 10, tiled verifier | 36 | **133.5585** | 127.2545 | 84.7750 | **207.9894** | 1,917 / 3,483 |

The retained mean is 4.033x the target-only mean. The maximum above 200 tok/s
is one prompt-specific sample; this evidence does not establish a 200 tok/s
average or production service-level guarantee.

After the retained source and PTX were finalized, a fresh release rebuild
passed the same 12-case gate again at 133.7458 tok/s mean and 208.4643 tok/s
maximum, with the same 639/1,161 acceptance and exact target-token parity.
Those final-build artifacts are `admission-r1/final-release-depth-10-tiled.json`
and `admission-r1/final-release-depth-10-tiled-parity.json`.

## Retained tuple

- Qwen3.6-27B, Q4_K_S GGUF;
- CUDA-resident F32 session cache;
- fixed MTP proposal depth ten;
- 65,536-row tensor-core Q6_K speculative draft head;
- stable verifier graph, batched rebase, parallel verifier projections, and
  admitted Q4_K Marlin/Q5-Q6 tensor-core paths; and
- `XRT_CUDA_DENSE_SMALL_MATMUL_TILED=1` for exact small-row F32 projections.

The tiled kernel maps 32 adjacent output columns to a warp and one of the eight
existing deterministic depth chains to each warp. It changes memory access,
not arithmetic order. The physical 11x128x257 CUDA oracle matched the control
kernel bit-for-bit, including the tail columns.

## Correctness evidence

`depth-screen-r1/depth-10-tiled-r3-parity.json` reports:

- suite `qwen36-greedy-admission-v1`;
- 12 cases and 36 candidate samples;
- no generation errors;
- deterministic repetitions within every case;
- no missing, extra, or mismatched cases; and
- exact candidate/target equality for every generated token ID.

The suite covers counting/CSV, Rust code, strict JSON, technical explanation,
arithmetic, SQL, multilingual Unicode, multi-turn constraints, repetition
resistance, summarization, creative structure, and random-looking formatting.

## Fixed-depth screen

Every depth below retained exact target-token parity:

| Depth | Mean tok/s | Minimum | Maximum | Accepted / drafted |
|---:|---:|---:|---:|---:|
| 2 | 78.455 | 75.198 | 83.273 | 488 / 546 |
| 4 | 106.736 | 81.765 | 122.128 | 580 / 720 |
| 6 | 116.925 | 79.489 | 143.672 | 608 / 900 |
| 8 | 129.757 | 82.905 | 172.106 | 631 / 1,009 |
| 10 | **131.506** | 83.339 | 205.567 | 639 / 1,161 |
| 12 | 124.744 | 79.342 | 189.023 | 641 / 1,360 |
| 15 | 129.099 | 74.907 | 214.531 | 649 / 1,539 |

Depth ten was the highest fixed-depth mean before the tiled optimization.
Per-prompt optimal depths would average about 137.6 tok/s, so selecting among
these fixed depths alone cannot credibly produce a 200 tok/s corpus mean.

## Profiler result

The bounded pre-change depth-ten verifier capture attributed 47.4% of kernel
time to Q4_K Marlin and 20.6% to the original F32 eight-chain dense kernel. In
the matched tiled capture, F32 dense time fell from 10.273 ms to 9.011 ms
(-12.28%). The one-repetition corpus mean rose from 131.506 to 133.872 tok/s
(+1.80%). The target verifier trunk remains the dominant optimization area.

Raw Nsight reports and exported kernel summaries are in `profiles/`.

## Rejected screens

All correctness-preserving candidates below were removed or left disabled
because they reduced the frozen-corpus mean:

| Candidate | Mean tok/s | Reason rejected |
|---|---:|---|
| Variable-shape adaptive depth | 107.6387 | Session-local graph captures erased the acceptance benefit |
| Generic coalesced dense matmul | 102.0161 | Lost the eight-chain latency-hiding structure |
| Exact 64-column tiled dense | 129.4825 | Larger block reduced end-to-end throughput |
| 77,824-row contiguous draft prefix | 131.8414 | Only one additional accepted token for more projection work |
| N-gram-only speculation | 39.0168 | Insufficient matching on the corpus |

## Environment identity

- Source base: `77911ff37ee8a8e94c11815726a4008dd949f1e0`, branch
  `feat/qwen36-mtp`, with the documented uncommitted candidate changes;
- GPU: NVIDIA GeForce RTX 4090, 24,564 MiB visible VRAM, 450 W;
- driver: 580.173.02;
- CUDA compiler: 12.8, build V12.8.93;
- Rust/Cargo: stable 1.97.1;
- Nsight Systems: 2024.6.2;
- model SHA-256:
  `a5ef62184c1729c38c9565b502303ac88e2fad3b1c3c6aa430d9e273bdd7f917`;
- corpus SHA-256:
  `196b64dcdf2c56b9d162080b28e1a8f3385b00454ef2bb32098c2cb11ff0ce25`;
- generated `dense_f32.ptx` SHA-256:
  `c869ae3ae9410aabb170296a5883d804ba0581cbd5acb4300a22754b34046a6b`.

The checkout was deliberately not clean, so this is not clean-container
reproduction evidence.

## Reproduction

Build the CUDA CLI, then execute the four-arm admission helper:

```bash
cargo build --release --locked -p xrt-cli --features cuda
scripts/benchmark-qwen36-greedy-admission.sh \
  /workspace/model/Qwen3.6-27B-Q4_K_S.gguf \
  /workspace/profiles/qwen36-greedy-admission \
  3
```

Run a tiled fixed-depth sweep against an existing target baseline:

```bash
XRT_TARGET_BASELINE=/workspace/profiles/qwen36-greedy-admission/target-only.json \
  scripts/benchmark-qwen36-mtp-depth-sweep.sh \
  /workspace/model/Qwen3.6-27B-Q4_K_S.gguf \
  /workspace/profiles/qwen36-depth-sweep \
  1 2 4 6 8 10 12 15
```

The exact-token comparator can be rerun independently:

```bash
python3 scripts/compare-bench-token-parity.py \
  admission-r1/target-only.json \
  depth-screen-r1/depth-10-tiled-r3.json
```

## Admission boundary

This evidence admits only deterministic greedy generation for the pinned
model, corpus, settings, and RTX 4090 environment. MTP and the tiled path remain
experimental, opt-in, and disabled by default. Production admission still
requires non-greedy parity, long-context behavior, concurrency, cancellation,
memory-pressure and recovery tests, security review, clean packaging, and
wider hardware coverage.
