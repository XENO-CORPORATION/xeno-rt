# Qwen3.8-27B official MTP admission on RTX 4090

Status: deterministic greedy throughput gate passed. The wider RTX 4090 product
candidate is retained under [`product-candidate-v2`](product-candidate-v2).

## Retained configuration

- Target: official `Qwen3.8-27B-Q4_K_M.gguf`
- Draft: official `mtp-Qwen3.8-27B-Q8_0.gguf`
- GPU: NVIDIA GeForce RTX 4090, 24 GB
- XRT backend: CUDA resident, F32 KV cache
- MTP depth: 4
- Verify CUDA graph: disabled
- Draft CUDA graph: enabled
- Prompt lookup and adaptive fallback: disabled
- Sampling: deterministic greedy, thinking disabled

The target, draft, corpus, and binary hashes are frozen in
[`greedy-depth4-repeat3/metadata/sha256.txt`](greedy-depth4-repeat3/metadata/sha256.txt).

## Paired result

The six-case corpus was run three times per arm after one model load per arm.
All 18 candidate samples emitted exactly the same token IDs as target-only.

| Metric | Target-only | Official MTP |
| --- | ---: | ---: |
| Samples | 18 | 18 |
| Mean decode tok/s | 30.5348 | **50.6128** |
| Median decode tok/s | 30.6411 | **55.4259** |
| Minimum decode tok/s | 30.1397 | **30.5369** |
| Maximum decode tok/s | 30.7494 | **60.9301** |

The candidate mean is 65.75% faster than the paired target-only mean. It
accepted 1,563 of 2,142 drafted tokens (72.97%). This is a corpus mean, not a
per-request or cross-hardware throughput guarantee.

| Case | Target mean | MTP mean | Draft acceptance |
| --- | ---: | ---: | ---: |
| Counting CSV | 30.6823 | 60.7561 | 306/306 |
| Rust code | 30.3171 | 57.0171 | 300/324 |
| Technical explanation | 30.6184 | 30.6847 | 231/603 |
| Strict JSON | 30.1441 | 58.5296 | 228/231 |
| Repetition resistance | 30.7054 | 54.1164 | 294/348 |
| Multilingual Unicode | 30.7414 | 42.5728 | 204/330 |

## Evidence and reproduction

- [`target.json`](greedy-depth4-repeat3/target.json): raw target-only samples
- [`mtp.json`](greedy-depth4-repeat3/mtp.json): raw MTP samples and counters
- [`target-vs-mtp-parity.json`](greedy-depth4-repeat3/target-vs-mtp-parity.json): exact-token comparison
- [`metadata`](greedy-depth4-repeat3/metadata): hashes, environment, and full GPU report

Reproduce on a prepared CUDA host:

```bash
XRT_QWEN38_REPETITIONS=3 \
  scripts/benchmark-qwen38-mtp.sh \
  /path/to/Qwen3.8-27B-Q4_K_M.gguf \
  /path/to/mtp-Qwen3.8-27B-Q8_0.gguf \
  qwen38-greedy-depth4-repeat3
```

## Product-candidate follow-up

The original [`production-v1`](production-v1) evidence identified the failed
arithmetic prompt and slow prefill. The retained
[`product-candidate-v2`](product-candidate-v2) repairs the quality profile,
adds Qwen-native thinking and sampling behavior, separates reasoning from the
final API answer, improves long-context prefill, and adds the complete live
service/lifecycle gate.

- Quality passed all four required answers with exact target/candidate parity.
- Multi-turn passed all three required answers with exact parity.
- Long-context retrieval passed all six markers through 7,776 prompt tokens.
- Batched prefill improved the longest target case from 714.83 to 473.69
  seconds and MTP from 751.59 to 504.57 seconds.
- The final 713.17-second service suite passed streaming/non-streaming API,
  reasoning separation, usage, 8K retrieval, concurrency, overload,
  cancellation, 30-request soak, unload/reload, and recovery with zero
  failures.

| Actual prompt tokens | Target prefill | MTP prefill | Target decode | MTP decode |
| ---: | ---: | ---: | ---: | ---: |
| 532 | 7.99 s | 8.48 s | 22.45 tok/s | 35.72 tok/s |
| 764 | 12.57 s | 13.42 s | 20.72 tok/s | 31.11 tok/s |
| 1,025 | 18.59 s | 19.75 s | 18.71 tok/s | 27.10 tok/s |
| 1,980 | 47.87 s | 50.90 s | 14.36 tok/s | 18.59 tok/s |
| 3,921 | 143.22 s | 152.54 s | 9.70 tok/s | 11.35 tok/s |
| 7,776 | 473.69 s | 504.57 s | 5.91 tok/s | 6.41 tok/s |

## Admission boundary

This record proves exact deterministic-greedy parity and the retained
throughput result. The product-candidate record adds sampled quality, live API,
concurrency, cancellation, memory, soak, and lifecycle evidence. Official
production admission still needs clean-checkout CI, packaged install/rollback,
and human release review. Other GPUs and CPU execution are outside this tuple,
and 8K latency remains explicitly documented. The reusable target verify graph
is excluded because it diverged during the Qwen3.8 repetition audit; the
ordinary batched verifier is both exact and faster for the retained workload.
