# Qwen3.6-27B XRT/llama.cpp paired RTX 4090 comparison

This directory retains the 2026-08-09 paired comparison that supersedes the
earlier unpaired llama.cpp `tg128` reference. It is controlled performance
evidence, not production admission.

## Result

| Runtime path | Native decode tok/s, mean +/- sample SD | Output tokens | MTP acceptance |
|---|---:|---:|---:|
| XRT target-only | 28.718 +/- 0.152 | 28 | - |
| XRT MTP | 45.480 +/- 0.218 | 28 | 24/32 (75.0%) |
| llama.cpp target-only | 50.476 +/- 0.087 | 64 | - |
| llama.cpp MTP | 143.132 +/- 0.481 | 64 | 54/67 (80.6%) |

XRT MTP was 58.37% faster than XRT target-only. It reached 90.10% of
llama.cpp target-only native decode and 31.78% of llama.cpp MTP native decode
on this request.

The old approximately 51 tok/s llama.cpp result was target-only-class
performance. Current llama.cpp MTP is a separate, much faster path.

## Protocol

- exact model: `Qwen3.6-27B-Q4_K_S.gguf`, 16,121,357,440 bytes,
  SHA-256 `a5ef62184c1729c38c9565b502303ac88e2fad3b1c3c6aa430d9e273bdd7f917`;
- one secure-cloud RTX 4090, driver 580.159.04, CUDA 12.8;
- greedy sampling, seed 424242, F32 KV, cache reuse off, one stream;
- three consecutive warmups, followed immediately by ten measurements;
- XRT native metrics from `xrt-cli bench` repetitions 4-13;
- llama.cpp native metrics from the server's ten final `eval time` records;
- the same `xrt-cli bench --external-base-url` SSE client measured both OpenAI
  server paths.

## Critical limitation

The request shape is matched, but the generated token streams are not. XRT
stopped after 28 tokens; llama.cpp reached the 64-token cap. The previews also
differ. Therefore this evidence resolves the benchmark-shape question and
identifies the performance gap, but it is not strict token-stream parity and
cannot establish response-quality parity.

`summary.json` contains the aggregate values and provenance. The other JSON
and log files are the unmodified raw evidence used to calculate them.
