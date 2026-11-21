# XRT Qwen3.6 27B Q4_K_S 100 tok/s evidence

Date: 2026-08-10

## Outcome

XRT's opt-in CUDA path exceeded the requested 100 decode-token/s threshold on an
RTX 4090. The benchmark ran 20 consecutive repetitions in one loaded process.
The first three repetitions were designated as warmup; all 17 retained samples
were above 100 decode token/s.

| Metric | Retained result |
| --- | ---: |
| Samples | 17 |
| Mean | 104.8782 tok/s |
| Median | 104.6499 tok/s |
| Standard deviation | 0.7548 tok/s |
| Minimum | 103.6761 tok/s |
| p10 | 103.9752 tok/s |
| p90 | 105.9295 tok/s |
| Maximum | 106.1316 tok/s |
| Samples above 100 tok/s | 17/17 |

The preceding stable XRT configuration measured 73.6403 tok/s mean on this GPU
and workload. The new result is 1.4242x, or 42.42%, faster.

An earlier 10-repetition screening run contained one transient 66.53 tok/s
sample. It was not discarded to form this claim; instead, the benchmark was
expanded to the independent 20-repetition run recorded here. That expanded run
had no sub-100 retained sample.

## Test identity

- GPU: NVIDIA GeForce RTX 4090, 24 GiB
- Runtime backend: `cuda-resident`
- Model: `Qwen3.6-27B-Q4_K_S.gguf`
- Model size: 16,121,357,440 bytes
- Model SHA-256: `a5ef62184c1729c38c9565b502303ac88e2fad3b1c3c6aa430d9e273bdd7f917`
- Quantization: `Q4_K_S`
- Prompt tokens: 34
- Output tokens: 64 per repetition; 63 timed decode tokens
- Sampling: temperature 0, top-k 1, top-p 1, repetition penalty 1,
  seed 424242
- Cache: F32, `default_chat`, prefix cache disabled
- Peak reported device-used VRAM: 20,041,564,160 bytes
- Evidence: [`retained-20.json`](retained-20.json)
- Evidence SHA-256: `1722f83c13abbc14c5edf01d9d68bb148b29e827d4b9f7e9bbd2559d8c72612e`

All 20 repetitions produced 64 output tokens, reported no error, and had the
same preview. Each repetition recorded 9 MTP verification batches, 68 drafted
tokens, 55 accepted tokens, and 13 rejected tokens.

## Reproduction

Build:

```bash
cargo build --release -p xrt-cli --features cuda
```

Run:

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

## What changed

- Q4_K projection weights are repacked once at model load into Marlin's tiled
  W4A16 layout rather than decoded through the earlier per-call path.
- Decode and MTP verification use specialized M1-M16 tensor-core kernels with
  persistent per-matrix input, output, lock, and temporary buffers.
- The Q4_K affine minimum is preserved directly, avoiding a lossy
  divide/re-multiply zero-point conversion.
- Existing XRT projection streams overlap independent Q/K/V, recurrent, and
  feed-forward projections during verification.
- Token embeddings retain their row-addressable GGUF-packed representation and
  do not use the projection-only Marlin layout.
- The CUDA grid is sized from the detected multiprocessor count.

The vendored Marlin headers are derived from vLLM commit
`d6941300fcb9d4a8bbea19f8b610c2aff9fc5cc3`. Rebuilding the checked-in PTX from
the vendored source produced an identical SHA-256:
`fc38ba17939f836170b6914737651439f26b7912922e217661a346e2b5dab697`.

## Verification and scope

- CUDA library tests: 11 passed, 0 failed; hardware tests remain opt-in.
- Physical RTX 4090 affine correctness test passed for M1 and M8 against the
  scalar Q4_K reference.
- CUDA-enabled workspace checks passed for `xrt-cuda`, `xrt-runtime`, and
  `xrt-cli`; CPU-only `xrt-cuda` and `xrt-runtime` checks also passed.
- Formatting and diff-whitespace checks passed.

This evidence admits one performance configuration, not every model, prompt,
GPU, context length, quantization, or concurrency level. The Marlin route
remains explicitly opt-in through `XRT_CUDA_Q4_K_MARLIN=1` while broader
correctness, compatibility, long-context, concurrency, and production admission
gates are completed. It does not change the OpenAI-compatible contract, remove
GGUF support, or remove CPU fallback.
