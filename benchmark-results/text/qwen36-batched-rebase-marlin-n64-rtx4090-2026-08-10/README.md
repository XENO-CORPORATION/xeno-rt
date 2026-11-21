# XRT batched MTP rebase and Marlin occupancy - RTX 4090 - 2026-08-10

## Outcome

On the pinned Qwen3.6-27B Q4_K_S workload, the retained 20-repetition run
designated the first three repetitions as warmup. The remaining 17 samples
measured **118.8529 tok/s mean**. This is a verified improvement, but it does
not meet the proposed 150 tok/s objective and is not a universal product claim.

| Metric | Result |
| --- | ---: |
| Retained samples | 17 |
| Mean decode | 118.8529 tok/s |
| Median decode | 118.8349 tok/s |
| Sample standard deviation | 0.0394 tok/s |
| 95% confidence half-width | 0.0187 tok/s |
| Minimum | 118.8063 tok/s |
| Maximum | 118.9189 tok/s |
| Mean end-to-end rate, including prefill | 42.2715 tok/s |
| Samples above 100 tok/s | 17/17 |

Every retained run generated 64 tokens, accepted 55 of 68 drafted tokens, and
used nine target-verification windows. The mean phase times were 84.593 ms for
drafting, 433.690 ms for target verification, and 9.825 ms for rebase.

## Retained changes

- Convert each normalized verifier activation matrix from F32 to F16 once and
  reuse it across eligible Marlin projections.
- Rebase accepted Qwen NextN rows as one device batch for full draft windows,
  using preallocated scratch; retain the serial path as a fallback.
- Add the upstream Marlin small-output occupancy choice: a 128-thread,
  64-column tile is selected when four 128-column stripes cannot occupy the
  device. `XRT_CUDA_MARLIN_N64_MAX_COLUMNS` provides an experimental override;
  the RTX 4090 default is 4,096 columns.

The same-request A/B record measured serial rebase at 25.389 ms and batched
rebase at 10.037 ms. Both modes emitted byte-identical output with SHA-256
`0ac8697a94d71d8de3066f5e0b18b5f396fc437f9eef4ccbf796728bac39b005`
and identical 55/68 proposal acceptance.

## Correctness evidence

- The physical RTX 4090 Q4_K Marlin affine test passed after the final
  two-stage PTX was restored.
- Five real-model verifier-audit windows produced identical optimized and
  serial argmax vectors. The largest observed logit absolute error was
  0.113782; no winning token changed.
- Serial and batched rebase produced byte-identical complete `generate`
  stdout, not only matching previews.
- The final retained benchmark had no reported generation errors.

Evidence files:

- [`retained-20.json`](retained-20.json), SHA-256
  `468822e7e25b074f6fd7a31696cc6a6a4e161f62f8ae907ac4c828ba3f622652`
- [`verifier-audit.log`](verifier-audit.log)
- [`rebase-serial.json`](rebase-serial.json) and
  [`rebase-batched.json`](rebase-batched.json)
- [`generate-serial.txt`](generate-serial.txt) and
  [`generate-batched.txt`](generate-batched.txt)

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
- MTP: depth eight, adaptive fallback disabled, draft vocabulary rows 65,536
- Generated Marlin PTX SHA-256:
  `937992346d80ae1815649bab13e52d6926c34ac59e7d46699b08cbc7d836a4ba`

## Reproduction

```bash
XRT_BACKEND=cuda \
XRT_PREFIX_CACHE=0 \
XRT_NGRAM_SPECULATION=0 \
XRT_QWEN_MTP=1 \
XRT_QWEN_MTP_MAX_DRAFT_TOKENS=8 \
XRT_QWEN_MTP_ADAPTIVE_FALLBACK=0 \
XRT_QWEN_MTP_VOCAB_ROWS=65536 \
XRT_QWEN_MTP_BATCHED_REBASE=1 \
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

## Rejected screens and remaining gap

The following candidates were measured and not retained: Marlin pipeline
stage four, two blocks per SM, depth seven, depth nine, depth twelve,
whole-verifier CUDA Graph replay, a larger 8,192-column N64 threshold, batched
full attention, and concatenating gate/up Q4_K tensors. The first pipeline
stage-three screen caused an illegal device access because it launched with an
under-sized 32 KiB dynamic shared-memory reservation. A later same-host screen
allocated the required 27/42 KiB and admitted stage three as a small follow-up
improvement; see
`../qwen36-marlin-stage-depth-screen-rtx4090-2026-08-10/README.md`.

At 63 timed tokens, 150 tok/s requires the decode window to fall to 420 ms.
The retained window is about 530 ms, of which target verification alone is
about 434 ms. Rebase tuning cannot close that remaining 109 ms. The next
credible work item is a purpose-built grouped or persistent verifier kernel
that shares activation/weight traffic across compatible projections while
preserving separate GGUF matrix layouts, followed by the full admission suite.

This record admits only one experimental benchmark tuple. It does not admit
MTP, Marlin, Qwen3.6, or this quantization for production by itself.
