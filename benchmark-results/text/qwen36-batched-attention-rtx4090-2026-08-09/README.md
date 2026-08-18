# Qwen3.6 verifier optimization cycle - RTX 4090 - 2026-08-09

This cycle tested the next XRT-native CUDA changes after the admitted packed
GGUF tensor-core verifier. The 100 tok/s objective was not reached. No measured
regression was promoted to the default execution path.

## Retained result

The final fixed-shape tensor-core path averaged **68.105 tok/s** across ten
retained repetitions (three warmups first), with a 1.641 tok/s sample standard
deviation and a 95% confidence half-width of 1.174 tok/s. One repetition was a
63.478 tok/s host outlier; the other nine averaged 68.619 tok/s. Every run kept
the same preview and the same speculative result: 55 of 68 drafted tokens
accepted in nine verification batches.

This is consistent with the existing experimental tensor-core admission record,
not a new speed claim. The earlier controlled record remains 70.243 tok/s. Run
variance on this rented node was large enough that short 71+ tok/s screens did
not survive retained reruns.

## Implemented and measured

- Added a three-launch batched causal full-attention verifier: fused Q/G
  preparation, batched KV append, and batched attention/gating. It has strict
  shape checks, F32 paged-KV support, and an exact serial fallback. A physical
  GPU equivalence test passed. The model-level result was inconclusive, so the
  path remains behind `XRT_QWEN_BATCHED_VERIFY_ATTENTION=1`.
- Added a fused paired-Q4_K SwiGLU research kernel and a physical GPU comparison
  against the separate tensor-core pipeline. Numerical equivalence passed, but
  all launch shapes regressed, so it requires the separate explicit
  `XRT_CUDA_Q4_K_VERIFY_SWIGLU=1` opt-in and is off by default.
- Tested reduced decoder occupancy, short-batch Q8/DP4A MMQ, and a reusable FP16
  activation workspace. The dynamic occupancy and automatic FP16 changes were
  removed after retained regressions. Short-batch MMQ remains confined to its
  pre-existing explicit MMQ opt-in.
- Generated checked-in PTX with CUDA 12.8.93. The Q4_K PTX SHA-256 is
  `af4f34fa8348da333c334be01454bbac658b4265a5627d40731d6de4402ef515`;
  the batched-attention PTX SHA-256 is
  `5e2ddfd2747a858a1e6565049a4ecaabb40becd4edce263a59cf956a18246166`.

## Workload

- Model: `Qwen3.6-27B-Q4_K_S.gguf`
- Model SHA-256: `a5ef62184c1729c38c9565b502303ac88e2fad3b1c3c6aa430d9e273bdd7f917`
- GPU: NVIDIA GeForce RTX 4090
- Prompt: `Write the numbers from 1 to 100 in order, separated by commas, and do not stop early.`
- Generation: 64 tokens, greedy, seed 424242
- MTP: depth 8, adaptive fallback off, draft vocabulary 65,536 rows
- Cache: F32, `default_chat`

## Verification

- Physical RTX 4090 batched-attention equivalence: passed.
- Physical RTX 4090 fused-SwiGLU equivalence: passed.
- Runtime unit tests: 108 passed, 7 ignored.
- CUDA crate CPU/stub tests: 5 passed.
- CUDA-feature compile: passed.
- `git diff --check`: passed.

## Next architecture for 100 tok/s

FFN and projection work remains dominant. The rejected FP16 screen shows that a
conversion cache must live at graph scope and be shared by all consumers of an
activation (Q/K/V and gate/up), not rebuilt inside each projection. The next
credible step is a grouped/persistent quantized verifier with shared activation
tiles and graph replay, followed by an interleaved A/B benchmark against the
fixed tensor-core path. CUDA launch reduction alone is insufficient.
