# Qwen3.6-27B MTP verifier optimization on RTX 4090

This directory retains the same-GPU A/B evidence for two exact XRT verifier
optimizations. It is controlled development evidence, not production
admission.

## Retained result

| Candidate | Decode tok/s, mean +/- sample SD | Verify time | Allocation calls | Allocated bytes |
|---|---:|---:|---:|---:|
| Corrected baseline | 49.312 +/- 0.386 | 1,147,864.5 us | 6,098 | 2,939,150,728 |
| Reusable verifier scratch | 49.699 +/- 0.169 | 1,138,866.8 us | 1,625 | 1,660,236,168 |
| Scratch + eight-warp blocks | **51.867 +/- 0.169** | **1,086,745.4 us** | 1,625 | 1,660,236,168 |

Reusable destination buffers remove 73.35% of verifier allocation calls and
43.51% of allocated bytes. Reducing the register-heavy exact Q4_K/Q5_K verify
block from 16 to eight warps lets Ada schedule more blocks per SM. Together,
the changes improve native decode by 5.18% without changing the 64-token
preview or the 55/68 MTP acceptance result.

Two alternatives were measured and rejected. A dedicated nine-row kernel was
bit-exact but doubled verification time and averaged 27.058 tok/s. A four-warp
screen averaged 8.216 tok/s and changed real-model deterministic behavior, so
it was not retained. The Q5_K CUDA test now uses nine distinct activation rows
instead of repeated copies to strengthen coverage around this boundary.

## Protocol

- exact model `Qwen3.6-27B-Q4_K_S.gguf`, 16,121,357,440 bytes, SHA-256
  `a5ef62184c1729c38c9565b502303ac88e2fad3b1c3c6aa430d9e273bdd7f917`;
- one RunPod secure-cloud RTX 4090, UUID
  `GPU-3a83f51f-710f-ac66-1d43-269bf0f13716`, 24,564 MiB, 450 W, driver
  580.159.04, CUDA 12.8;
- greedy sampling, seed 424242, F32 target KV, one stream;
- XRT prefix caching and N-gram speculation disabled;
- MTP depth eight, adaptive fallback disabled, 65,536 draft-vocabulary rows;
- three consecutive warmups followed by ten measurements; and
- native values are `xrt-cli bench` `decode_tok_s` records.

The common generation arguments were:

```text
--prompt "Write the numbers from 1 to 100 in order, separated by commas, and do not stop early."
--max-tokens 64 --repetitions 13 --concurrency 1
--temperature 0 --top-k 1 --top-p 1 --repetition-penalty 1 --seed 424242
```

The retained MTP environment was:

```text
XRT_BACKEND=cuda
XRT_PREFIX_CACHE=0
XRT_NGRAM_SPECULATION=0
XRT_QWEN_MTP=1
XRT_QWEN_MTP_MAX_DRAFT_TOKENS=8
XRT_QWEN_MTP_ADAPTIVE_FALLBACK=0
XRT_QWEN_MTP_VOCAB_ROWS=65536
```

## Verification and limits

Focused Q4_K and Q5_K CUDA tests passed against serial recurrent matvec bits,
including distinct Q5_K activation rows. Local formatting, CUDA-feature
compilation, and all 108 non-artifact `xrt-runtime` unit tests passed.

This crosses the narrow 50 tok/s objective on the retained workload. It does
not match the previously pinned llama.cpp MTP result of about 144 tok/s, and
that reference was not rerun on this physical GPU after this optimization.
The next large gain requires a fused causal verifier for the per-row DeltaNet
and full-attention work, followed by the multi-prompt and non-greedy admission
gates. `summary.json` contains machine-readable aggregates and provenance.
