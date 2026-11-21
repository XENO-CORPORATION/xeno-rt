# Qwen3.8-27B official NextN/MTP admission

Status: product candidate and opt-in for the pinned RTX 4090 tuple below.
Quality, exact parity, bounded memory, long-context correctness, the live API,
concurrency/backpressure, cancellation, soak, and lifecycle recovery pass.
Official product-wide production admission still requires clean-checkout CI,
packaging/install/rollback verification, and human release review.

## Supported artifact tuple

XRT loads the official Qwen3.8 target and NextN companion as separate GGUF
artifacts. Set `XRT_QWEN_MTP_DRAFT_MODEL` to the companion path; XRT validates
its architecture and tensor geometry, then overlays only the companion NextN
layer onto the resident target model. Duplicate embeddings and output tensors
remain owned by the target.

The retained tuple is:

- `Qwen3.8-27B-Q4_K_M.gguf`, SHA-256
  `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`;
- `mtp-Qwen3.8-27B-Q8_0.gguf`, SHA-256
  `cbf60a0c48b431bb61f1d49b8948dc88ac29c398d6dbdbbb2e6e89ef77eacc9a`;
- official GGUF revision
  `0669b98607d47046c7c2b3f801011d54a08cfccf`.

Target plus companion occupy about 20.95 GB of tracked resident weights and fit
on a 24 GB RTX 4090 with the admitted F32 KV and scratch budgets.

## Retained execution policy

```text
XRT_BACKEND=cuda
XRT_QWEN_MTP=1
XRT_QWEN_MTP_DRAFT_MODEL=/path/to/mtp-Qwen3.8-27B-Q8_0.gguf
XRT_QWEN_MTP_MAX_DRAFT_TOKENS=4
XRT_QWEN_MTP_VOCAB_ROWS=65536
XRT_QWEN_MTP_BATCHED_REBASE=1
XRT_QWEN_MTP_VERIFY_GRAPH=0
XRT_QWEN_MTP_DRAFT_GRAPH=1
XRT_QWEN_BATCHED_PREFILL=1
XRT_QWEN_BATCHED_PREFILL_MAX_ROWS=5
XRT_CUDA_Q4_K_MARLIN=1
XRT_CUDA_Q5_K_MARLIN=1
XRT_CUDA_Q6_K_MARLIN=1
XRT_GPU_MEMORY_FRACTION=0.97
XRT_GPU_RESERVED_MB=512
XRT_GPU_KV_FRACTION=0.75
XRT_MAX_ACTIVE_SEQUENCES=1
XRT_MAX_QUEUED_SEQUENCES=2
```

Depth four is intentional. The official single NextN layer is applied
recursively, so errors compound at longer proposal depths. On the admission
corpus, depth eight averaged 45.14 tok/s with 49.96% acceptance; depth four
averaged 50.77 tok/s with 72.97% acceptance in the one-repetition screen.

The Qwen3.8 target verify CUDA graph is not admitted. It produced a phase shift
in the repetition case after 32 generated tokens. The non-graph batched
verifier matched serial target argmax in every audited window and is faster on
the retained case. Benchmark and production scripts therefore force verify
graphs off instead of inheriting an unsafe environment setting.

## Retained RTX 4090 result

Across six deterministic greedy cases and three repetitions per arm, all 18
MTP samples matched the target token IDs exactly. Target-only averaged 30.5348
tok/s; MTP averaged **50.6128 tok/s**, with a 55.4259 median and 30.5369--60.9301
range. The mean speedup is 65.75%, and 1,563/2,142 drafts were accepted.

Raw JSON, hashes, GPU metadata, reproduction commands, and the case breakdown
are in
[`qwen38-mtp-rtx4090-2026-08-15`](../benchmark-results/text/qwen38-mtp-rtx4090-2026-08-15/README.md).

These figures are workload-dependent measurements, not a 50 tok/s guarantee.
The technical-prose case averaged only 30.6847 tok/s while structured cases
reached 57--61 tok/s.

## Product-candidate gate result

The final quality profile uses Qwen3.8's native non-greedy sampling behavior
and passed every required answer in both arms: `XRT_READY`, `259.2`, `Ember`,
and `München-東京-مرحبا`. Target and candidate token IDs match exactly. The
greedy-only MTP path correctly falls back to target decoding for this sampled
profile instead of changing its distribution.

The three-case multi-turn suite passed exact parity and returned `cobalt`,
`delta`, and `verified`. The six-case context suite returned every marker
through 7,776 prompt tokens with exact parity. Batched prefill reduced the
7,776-token target measurement from 714.83 to 473.69 seconds and the MTP
measurement from 751.59 to 504.57 seconds. MTP decode remained workload- and
context-dependent, reaching 6.41 tok/s on the longest case.

The final live OpenAI-compatible service suite passed with zero failures. It
covered non-streaming and streaming chat/completions, separate Qwen thinking
`reasoning_content`, final-answer hygiene, usage and finish reasons, multi-turn
requests, the selected 8K request, concurrency, bounded HTTP 429 backpressure,
cancellation, invalid input, 30/30 deterministic soak requests, unload, the
expected unloaded HTTP 503, reload, and a post-reload quality probe. The 8K API
case used 7,773 prompt tokens, returned `XRT-08192-05-PASS`, took 530.39 seconds,
and cleaned its KV reservation. Peak device use was 23,506 MiB with 605 MiB
free.

The raw reports, hashes, GPU metadata, server log, and machine-readable
admission state are in
[`product-candidate-v2`](../benchmark-results/text/qwen38-mtp-rtx4090-2026-08-15/product-candidate-v2/README.md).

## Remaining release gates

The model/runtime tuple is now a product candidate, not an official
product-wide production admission. A human reviewer must place the changes on a
clean branch, run clean-checkout CI, validate packaged install and rollback,
and approve the release. Broader GPU claims require their own hardware
evidence. This CUDA-only tuple does not advertise a CPU path.

The measured 530-second 8K API latency is a supported-but-slow boundary and
must remain visible in product documentation. The 50.6128 tok/s greedy MTP
figure remains a corpus mean, not a per-request, long-context, or cross-hardware
throughput guarantee.
