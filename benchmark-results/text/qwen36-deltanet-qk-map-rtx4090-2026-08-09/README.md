# Qwen3.6 DeltaNet Q/K head-map correction

This directory retains the controlled 2026-08-09 RTX 4090 validation of the
Qwen3.6 Gated DeltaNet head-map correction. It is correctness and performance
evidence for an experimental path, not production admission.

## Root cause

XRT assigned value heads to Q/K groups in adjacent buckets. Qwen3.6 and the
pinned llama.cpp reference tile the Q/K groups over value heads instead:

```text
wrong:  0,0,0,1,1,1,...
right:  0,1,...,15,0,1,...,15,0,1,...,15
```

For Qwen3.6-27B, the correct map is `value_head % 16` across 48 value heads.
The fix is applied to CPU/reference and CUDA execution, rejects non-divisible
geometry, and has deterministic 32:16 and 48:16 coverage.

## Result

All values below exclude repetitions 1-3 and summarize repetitions 4-13.

| Runtime path | Decode tok/s, mean +/- sample SD | Output | Acceptance |
|---|---:|---:|---:|
| XRT target-only | 30.785 +/- 0.052 | 64 | - |
| XRT MTP, full vocabulary | 45.736 +/- 0.129 | 64 | 55/68 (80.88%) |
| XRT MTP, 65,536 draft rows | **50.418 +/- 0.255** | 64 | 55/68 (80.88%) |
| llama.cpp target-only | 50.627 +/- 0.027 | 64 | - |
| llama.cpp MTP | 144.073 +/- 1.614 | 64 | 54/67 (80.60%) |

The corrected 65,536-row XRT path is 63.77% faster than corrected XRT
target-only and is within 0.42% of llama.cpp target-only on this workload. It
does not close the llama.cpp MTP gap: XRT reaches 34.99% of llama.cpp MTP.

The target, full-vocabulary MTP, and 65,536-row MTP runs all emitted 64 tokens.
XRT target/MTP previews were identical, and the draft-prefix optimization
preserved both the preview and 55/68 acceptance. A complete cross-runtime token
digest was not retained, so strict cross-runtime token-stream parity remains an
open gate.

## Protocol

- exact model: `Qwen3.6-27B-Q4_K_S.gguf`, 16,121,357,440 bytes,
  SHA-256 `a5ef62184c1729c38c9565b502303ac88e2fad3b1c3c6aa430d9e273bdd7f917`;
- secure-cloud RTX 4090, 24,564 MiB, 450 W, driver 570.195.03, CUDA 12.8;
- exact llama.cpp commit `08659901c43b51de735740f1cf61bb82fbe0c4e4`;
- greedy sampling, seed 424242, F32 target KV, one stream;
- XRT prefix caching and N-gram speculation disabled;
- llama.cpp prompt caching disabled with `--no-cache-prompt`;
- three consecutive warmups followed by ten measurements; and
- llama.cpp native values are the server's `eval time` records, while XRT
  native values are `xrt-cli bench`'s `decode_tok_s` records.

The retained XRT MTP environment is:

```text
XRT_BACKEND=cuda
XRT_PREFIX_CACHE=0
XRT_NGRAM_SPECULATION=0
XRT_QWEN_MTP=1
XRT_QWEN_MTP_MAX_DRAFT_TOKENS=8
XRT_QWEN_MTP_ADAPTIVE_FALLBACK=0
XRT_QWEN_MTP_VOCAB_ROWS=65536
```

The common generation arguments were:

```text
--prompt "Write the numbers from 1 to 100 in order, separated by commas, and do not stop early."
--max-tokens 64 --repetitions 13 --concurrency 1
--temperature 0 --top-k 1 --top-p 1 --repetition-penalty 1 --seed 424242
```

llama.cpp target used F32 K/V cache, one parallel slot, full GPU offload, and
`--no-warmup --no-cache-prompt`. Its MTP run additionally used
`--spec-type draft-mtp --spec-draft-n-max 8 --spec-draft-n-min 0`.

## Verification

- `cargo fmt --all -- --check`;
- `git diff --check`;
- `cargo test -p xrt-models --lib` (20 passed);
- `cargo test -p xrt-runtime --lib` (108 passed, 7 ignored artifact tests);
- `cargo check -p xrt-runtime --features cuda`;
- focused CUDA geometry validation (passed); and
- ignored 128-step DeltaNet CUDA/scalar comparison on the same RTX 4090 using
  two Q/K groups tiled over four value heads (passed).

`summary.json` contains aggregate values and provenance. The remaining JSON,
logs, timing extracts, help output, and environment capture are raw evidence.

## Admission boundary

The 50 tok/s mean is a narrow workload result. MTP and draft-prefix projection
remain experimental and disabled by default. Production admission still needs
multi-prompt token parity and quality, exact non-greedy rejection sampling,
long-context/concurrency/reliability coverage, security and packaging gates,
and a clean reproducible candidate commit.
