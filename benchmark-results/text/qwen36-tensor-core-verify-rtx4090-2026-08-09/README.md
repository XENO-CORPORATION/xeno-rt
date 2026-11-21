# Qwen3.6 tensor-core verifier — RTX 4090 — 2026-08-09

This record measures the experimental XRT packed-GGUF tensor-core verification
path added behind `XRT_CUDA_KQUANT_TENSOR_CORE_VERIFY=1`. The retained kernel
decodes Q4_K, Q5_K, and Q6_K weights directly into FP16 tiles, executes WMMA
with FP32 accumulation, and does not retain expanded model weights.

## Controlled result

The baseline and candidate used the same secure RunPod RTX 4090, model bytes,
prompt, seed, MTP policy, executable source tree, and CUDA 12.4 compatibility
build. Each record contains three warmups followed by ten retained repetitions.

| Path | Mean decode | Sample SD | 95% CI | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact fused DeltaNet baseline | 49.049 tok/s | 0.113 | 0.070 | 48.782 | 49.146 |
| Tensor-core verify | 70.243 tok/s | 0.903 | 0.559 | 68.698 | 71.257 |

The candidate is **43.21% faster**, and all thirteen candidate repetitions
produced the same preview and speculative counters: 55 accepted of 68 drafted
tokens in nine verification batches. The 100 tok/s objective is not met; the
remaining measured gap is 29.757 tok/s.

## Workload

- Model: `Qwen3.6-27B-Q4_K_S.gguf`
- SHA-256: `a5ef62184c1729c38c9565b502303ac88e2fad3b1c3c6aa430d9e273bdd7f917`
- GPU: NVIDIA GeForce RTX 4090, 23,785,373,696 bytes VRAM
- Prompt: `Write the numbers from 1 to 100 in order, separated by commas, and do not stop early.`
- Generation: 64 tokens, greedy, seed 424242
- MTP: depth 8, adaptive fallback off, 65,536 draft-vocabulary rows

The retained environment was:

```text
XRT_BACKEND=cuda
XRT_PREFIX_CACHE=0
XRT_NGRAM_SPECULATION=0
XRT_QWEN_MTP=1
XRT_QWEN_MTP_MAX_DRAFT_TOKENS=8
XRT_QWEN_MTP_ADAPTIVE_FALLBACK=0
XRT_QWEN_MTP_VOCAB_ROWS=65536
XRT_CUDA_KQUANT_TENSOR_CORE_VERIFY=1
```

## Admission boundary

This is an experimental performance result, not general production admission.
The deterministic pinned output stayed unchanged, the exact fused DeltaNet path
and rollback tests passed on the physical GPU, CUDA-feature compilation passed,
and 108 runtime unit tests passed locally. Broader prompt-quality, device,
quantization, memory, and long-context gates remain required before making the
tensor-core mode automatic.

The next measured target is a batched/fused causal full-attention verifier. The
current verifier still processes each proposed row serially through Q/K
normalization, RoPE, KV append, attention, and gating in every full-attention
layer; profiling attributes about 103 ms per 64-token run to that core, while
FFN and projection work remain the largest total share.
