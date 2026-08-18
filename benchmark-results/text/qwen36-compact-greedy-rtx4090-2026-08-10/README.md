# XRT compact greedy MTP verification - RTX 4090 - 2026-08-10

## Outcome

XRT now keeps raw greedy MTP verification decisions on the GPU. The verifier
computes the first argmax for every target-logit row, returns the accepted
prefix and boundary token, and avoids downloading the complete
`rows x 248,320` logit matrix. Requests with sampling transforms, penalties,
or an EOS draft retain the established full-logit path.

The final 20-repetition run designated the first three repetitions as warmup.
Across the 17 retained samples:

| Metric | Result |
| --- | ---: |
| Mean decode | 114.2518 tok/s |
| Median decode | 114.6814 tok/s |
| Sample standard deviation | 1.3827 tok/s |
| 95% confidence half-width | 0.6573 tok/s |
| Minimum | 108.9885 tok/s |
| Maximum | 114.7769 tok/s |
| Samples above 100 tok/s | 17/17 |
| Mean end-to-end rate, including prefill | 41.5299 tok/s |

Sixteen retained samples were between 113.7118 and 114.7769 tok/s; one host
outlier measured 108.9885 tok/s and remains included. All runs produced 64
tokens, one output preview, 55 accepted of 68 drafted tokens, 13 rejected
tokens, and nine verification batches with no reported error.

The same-node pre-change screen averaged 112.1119 tok/s over five retained
samples. The final retained mean is 1.91% higher, but the sample counts differ,
so this is supporting A/B evidence rather than a universal speed claim. The
larger architectural result is deterministic transfer reduction: mean explicit
device-to-host traffic fell from 77,476,112 to 993,860 bytes per run, a 98.72%
reduction.

## Test identity

- GPU: NVIDIA GeForce RTX 4090, 24 GiB
- Runtime backend: `cuda-resident`
- Model: `Qwen3.6-27B-Q4_K_S.gguf`
- Model size: 16,121,357,440 bytes
- Model SHA-256: `a5ef62184c1729c38c9565b502303ac88e2fad3b1c3c6aa430d9e273bdd7f917`
- Prompt tokens: 34
- Output tokens: 64 per repetition; 63 timed decode tokens
- Sampling: temperature 0, top-k 1, top-p 1, repetition penalty 1,
  seed 424242
- Cache: F32, `default_chat`, prefix cache disabled
- Maximum reported device-used VRAM: 20,041,564,160 bytes
- Final evidence: [`retained-token-only-20.json`](retained-token-only-20.json)
- Final evidence SHA-256:
  `da8752d2f4e80eb78c7304f5558b5869f0b04a4af6dab36ea2374875816c5f5f`
- Generated argmax PTX SHA-256:
  `5c466a15c2407697151420f65427a18740e5335ea9bae73678a6ad05dcf38b8b`

## Correctness and regression gates

- The physical RTX 4090 row-argmax test passed, including ties, negative
  infinity, NaNs, multiple rows, and ranged device downloads.
- The repaired real-model target-verifier audit completed three windows and
  produced identical optimized and serial argmax vectors for every row. The
  maximum observed logit absolute error was 0.083511; no winner changed.
- Runtime tests passed: 109 passed, 0 failed, 7 ignored.
- CUDA library non-hardware tests passed: 11 passed, 0 failed, 83 ignored.
- CUDA-feature compilation, formatting, and diff whitespace checks passed.
- CPU fallback and the OpenAI-compatible API contract are unchanged.

## Measured rejected paths

The negative screens are retained rather than promoted:

- Whole-verifier CUDA Graph capture/update averaged 109.6972 tok/s versus the
  same-node eager baseline of 112.1119 tok/s. Updating the 3,786-node graph for
  each changing window cost more than replay saved.
- Depth seven averaged 107.7329 tok/s and required ten verification batches.
- Depth twelve averaged 90.2607 tok/s and also required ten batches, with lower
  effective proposal acceptance.
- Batched full-attention tile 16 averaged 114.2966 tok/s versus 114.3108 tok/s
  for its compact eager comparison, a neutral result.

## Reproduction

```bash
XRT_BACKEND=cuda \
XRT_PREFIX_CACHE=0 \
XRT_NGRAM_SPECULATION=0 \
XRT_QWEN_MTP=1 \
XRT_QWEN_MTP_MAX_DRAFT_TOKENS=8 \
XRT_QWEN_MTP_ADAPTIVE_FALLBACK=0 \
XRT_QWEN_MTP_VOCAB_ROWS=65536 \
XRT_CUDA_Q4_K_MARLIN=1 \
XRT_CUDA_KQUANT_TENSOR_CORE_VERIFY=1 \
XRT_CUDA_PARALLEL_VERIFY_PROJECTIONS=1 \
XRT_QWEN_MTP_VERIFY_GRAPH=0 \
target/release/xrt-cli bench \
  --model /workspace/model/Qwen3.6-27B-Q4_K_S.gguf \
  --prompt "Write the numbers from 1 to 100 in order, separated by commas, and do not stop early." \
  --cache-modes f32 \
  --backends cuda-resident \
  --cache-policy default_chat \
  --max-tokens 64 \
  --repetitions 20 \
  --temperature 0 \
  --top-k 1 \
  --top-p 1 \
  --repetition-penalty 1 \
  --seed 424242 \
  --json
```

## Scope

This admits one experimental Qwen3.6 greedy benchmark configuration. It is not
a promise for every model, quantization, prompt, context length, sampler,
concurrency level, or GPU. MTP and Marlin remain opt-in while multi-prompt
parity, non-greedy speculative sampling, long-context, concurrency,
reliability, security, packaging, and clean-environment reproduction gates are
completed.
