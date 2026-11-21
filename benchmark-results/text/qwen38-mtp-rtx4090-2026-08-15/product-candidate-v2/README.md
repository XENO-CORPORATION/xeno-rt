# Qwen3.8-27B XRT product candidate on RTX 4090

Status: **product candidate for one pinned CUDA tuple**. The runtime, model
behavior, bounded memory profile, OpenAI-compatible service, and lifecycle
gates passed. This is not yet a product-wide production admission: the current
checkout is dirty and still requires clean-checkout CI, packaging/install/
rollback verification, and human release review.

## Qualified tuple

- GPU: NVIDIA GeForce RTX 4090, 24 GB
- Target: `Qwen3.8-27B-Q4_K_M.gguf`
  (`31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`)
- Official NextN companion: `mtp-Qwen3.8-27B-Q8_0.gguf`
  (`cbf60a0c48b431bb61f1d49b8948dc88ac29c398d6dbdbbb2e6e89ef77eacc9a`)
- Official GGUF revision: `0669b98607d47046c7c2b3f801011d54a08cfccf`
- Base-model revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- XRT server binary:
  `6ea8d6c20e0daa6edbb04be44bf53f37e4a1d024eed6f71bd7269c66570ef96a`
- XRT CLI binary:
  `7424177a70ace94237d23de8fa078dcdc569b7dc388ecd6beae509d45ac3073f`

The server launcher pins CUDA-resident execution, depth-four official MTP,
batched prefill, exact non-graph verification, bounded `1 active + 2 queued`
scheduling, and a 2.12 GiB logical KV budget. It binds to loopback by default.

## Retained results

### Model quality and parity

The final quality binary passed all four required-answer cases in both arms and
preserved exact target/candidate token parity. The retained answers are
`XRT_READY`, `259.2`, `Ember`, and `München-東京-مرحبا`.

The quality profile uses Qwen3.8's native non-greedy sampling defaults. XRT
therefore falls back to target decoding instead of incorrectly applying the
greedy-only MTP path; the candidate is exact but receives no speculative
speedup on these requests.

The three multi-turn recall cases also passed exact parity and returned
`cobalt`, `delta`, and `verified`.

### Throughput and context

The deterministic six-case, three-repetition gate remains the throughput
record: target-only averaged 30.5348 tok/s and official MTP averaged 50.6128
tok/s with 72.97% draft acceptance and exact output-token parity. It is a
corpus result, not an SLA.

Batched prefill improved the earlier long-context baseline by about 1.5x at
7,776 prompt tokens. All six context markers passed with exact parity.

| Prompt tokens | Target prefill | MTP prefill | Target decode | MTP decode |
| ---: | ---: | ---: | ---: | ---: |
| 532 | 7.99 s | 8.48 s | 22.45 tok/s | 35.72 tok/s |
| 764 | 12.57 s | 13.42 s | 20.72 tok/s | 31.11 tok/s |
| 1,025 | 18.59 s | 19.75 s | 18.71 tok/s | 27.10 tok/s |
| 1,980 | 47.87 s | 50.90 s | 14.36 tok/s | 18.59 tok/s |
| 3,921 | 143.22 s | 152.54 s | 9.70 tok/s | 11.35 tok/s |
| 7,776 | 473.69 s | 504.57 s | 5.91 tok/s | 6.41 tok/s |

The final HTTP gate independently completed the selected 8K case with 7,773
API prompt tokens, the exact required marker, correct usage accounting, and no
memory overcommit. It took 530.39 seconds. At peak it reserved 2,154,299,392
KV bytes, used 23,506 MiB of the 24 GB device, and retained 605 MiB free.
Long-context latency is therefore a documented limitation, not a hidden
throughput claim.

### OpenAI-compatible service and reliability

The 713.17-second service suite passed with zero failures:

- non-streaming and streaming chat/completions contracts;
- separate `reasoning_content` and final answer content for Qwen thinking;
- exact generated-token usage and correct `stop`/`length` finish reasons;
- three multi-turn API recalls;
- the full 8K API request and post-request KV cleanup;
- concurrency levels 1 and 2;
- bounded overload: 3 admitted and 5 HTTP 429 responses from 8 simultaneous
  requests;
- client cancellation and scheduler drain;
- invalid-request HTTP 400 handling;
- 30/30 deterministic soak requests, one output digest, 0.39 s mean and 0.40 s
  p95, +221,184 host RSS bytes and -268,435,456 GPU bytes;
- unload to 710,934,528 device bytes, an expected HTTP 503 while unloaded,
  102.63-second reload, and a successful post-reload quality probe.

### Rejected experiment

A warp-shuffle verify-attention reduction passed the direct CUDA correctness
probe and preserved the selected output, but made the 1,980-token prefill 0.4%
slower. It was reverted. The rejected binary/PTX hashes and raw screen remain
under [`rejected-warp-attention`](rejected-warp-attention); they are not part
of the product candidate.

## Evidence map

- [`quality-v4`](quality-v4): final required-answer reports, token parity,
  provenance, hashes, and GPU metadata
- [`context-v3`](context-v3): exact multi-turn and six-length context reports
- [`service-v5/service.json`](service-v5/service.json): final live HTTP,
  concurrency, cancellation, soak, memory, and lifecycle evidence
- [`service-v5/server.log`](service-v5/server.log): retained server-side log
- [`manifest.json`](manifest.json): machine-readable scope and admission state
- [`source-sha256.txt`](source-sha256.txt): retained source, binary, launcher,
  harness, and evidence hashes

## Run the candidate

```bash
XRT_SERVER_BIN=/path/to/xrt-server \
  scripts/run-qwen38-production-server.sh \
  /path/to/Qwen3.8-27B-Q4_K_M.gguf \
  /path/to/mtp-Qwen3.8-27B-Q8_0.gguf \
  3000
```

The default bind address is `127.0.0.1`. Do not expose the service publicly
without an authenticated product gateway.

## Remaining release gate

Official production admission needs a human-reviewed clean branch, CI from a
clean checkout, packaged-artifact install/rollback validation, and release
approval. Broader GPU claims require separate hardware evidence. No CPU path is
advertised for this CUDA-only tuple. The measured 8K latency must remain visible
in product documentation until a faster retained implementation replaces it.
