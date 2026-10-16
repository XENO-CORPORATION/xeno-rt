# Drama 3 BF16 fidelity correction

The previously failing bounded BF16 numerical gate now passes. **Drama 3 is
still not admitted as a production tier.** Multilingual/perceptual quality,
exclusive performance, final installed-tier/codec qualification, trust delivery
and distribution authorization remain separate requirements. No weights were
published.

## Measured results

| Check | Result |
|---|---|
| Independently composed backbone, 26 prompt rows + 8 teacher-forced frames | 1,360/1,360 layer outputs bit-exact |
| Final top-20 log probabilities | 34/34 pass at rtol/atol 0.02; all 9 generation argmaxes match |
| Historical isolated two-position backbone | 80/80 bit-exact |
| Historical two-position codebook tower | 26/26 bit-exact |
| Actual-driver components and recurrent state | 19 + 1 pass, including exact eight-step BF16 state outputs |
| Local workspace / native audio | 454 passed, 39 explicit ignores / 69 passed |
| Rust 1.83 combined server/CUDA all-targets | Pass |
| CPU FP32 compatibility | Historical 52-frame WAV SHA256 unchanged |
| Normal-build BF16 lifecycle, 4 GiB expert cache | Five cases pass; A/B/A WAV identity; cancellation/limit discard; zero admission leases after cleanup |

Layer tolerances remain rtol 0.008 / atol 0.002; no threshold was widened.
The normal build uses `drama3-native,cuda`, without diagnostics or a precision
environment override. The composed oracle propagates only reference layer
outputs. It verifies initial-embedding identity and the teacher-forced fixture
digest before comparison. Matching sampled speech for arbitrary inputs is not
claimed.

`inventory.json` binds these receipts to their original bytes. The source
inventory was captured during the final build and verified unchanged after the
checks. It also records the staged Git content fingerprint and explicitly lists
CRLF-to-LF normalizations, verified to contain no other differences.
`build-provenance.json` records executable digests and the original
reference file digest. Raw native traces, reference activation vectors, model
weights and generated WAVs stay outside Git; only metrics and hashes are retained
here. The 80/26 operator suites preceded a nonnumerical boxing/layout cleanup;
the composed comparison and lifecycle were rerun on the final executable.

## Root causes and failure controls

The fixes preserve rounded reciprocal FP8 scales, Triton division boundaries,
GPU BF16 normalization/RoPE/activations, 128-key online efficient attention,
CUDA router softmax, top-eight normalization, four-accumulator expert reduction
and vectorized recurrent-state reduction geometry. Dense BF16 uses the normal
bounded CUDA-12 cuBLAS path; FP32 keeps its prior implementation.

The retained routing negative control selects the correct experts but changes a
BF16 weight from 0.267578125 to 0.265625. The previous audit fed native routing
weights into both expert paths and missed that boundary. The audit now computes
routing independently and rejects the old trace. Other retained failed composed
comparisons preserve the intermediate 148/1,360 and 337/1,360 results.

The synthetic eight-expert reduction fixture rejects the wrong binary tree at
723 FP32 outputs. The independent attention fixture covers 16 cases, actual
backbone/tower head counts, partial prefixes and multiple 128-key blocks. A
separate 8,192-row allocation/reuse test exercises the full workspace bound.
The shared arena charges 104 MiB scratch and recurrent state separately;
attention's reusable portion has a 40 MiB limit. Combined contexts above 8,192
rows are refused before generation.

The 8 GiB reusable-engine profile was refused by Windows commit admission on
this shared workstation. Those refusals are retained. Reducing only the expert
cache to 4 GiB allowed the five-case lifecycle run; the guard was not changed.
The numerical probes use their explicitly reported budgets and are not resource
admission or service proofs.

## Performance and admission limits

Criterion receipts cover dense projection, normalization, attention, routing and
recurrent decode. They are short, synchronized component measurements on a shared
workstation. Dense BF16 measurements expose overhead versus the old
FP32-accumulation control, and repeated unchanged controls also vary. No exclusive
performance pass or complete narration speedup is claimed. Capped-warning clippy
completes with inherited warnings; it is not a warning-free pass. Existing
dependency-advisory findings remain governed by the admission ledger.

The prior 600-frame, three-attempt, ten-language matrix completed 12/20 cases and
three negative controls on an older candidate. Its retained summary is a failed
quality result, not evidence for the corrected candidate. MOS, speaker/emotion
fidelity, long narration, managed codec delivery and distributed publisher trust
remain open.

## Reproduction

Use the exact checkpoint and hash-pinned six upstream modules recorded in
`reference-provenance.json`. Reference execution requires the independently
installed pinned CUDA Torch/Triton environment. Runtime inference requires no
Python process. Replace placeholders with the locally verified files:

```powershell
cargo build --release -p xrt-audio --example drama3_backbone_probe --example drama3_tower_probe --example drama3_qualify_native --features drama3-native,cuda --locked
drama3_backbone_probe.exe <managed-model> <new-native.json> bf16-w8a8 none "The evidence remains clear." 8192 5120 <teacher-forced-codes.json>
python scripts/drama3_compare_frozen_reference.py --native <new-native.json> --reference <independently-captured-reference.json> --out <new-comparison.json>
drama3_backbone_probe.exe <managed-model> <new-synthetic.json> bf16-w8a8 none --synthetic 8192 5120
python scripts/drama3_backbone_oracle.py --source <pinned-source> --model <managed-model> --native <new-synthetic.json> --out <new-backbone-reference.json> --bf16-w8a8 --device cuda
drama3_tower_probe.exe <managed-model> <new-tower.json> bf16-w8a8 none 5120
python scripts/drama3_tower_oracle.py --source <pinned-source> --model <managed-model> --native <new-tower.json> --out <new-tower-reference.json> --bf16-w8a8 --device cuda
drama3_qualify_native.exe <managed-model> <native-codec> <cases.json> <new-directory> 4096 5120 bf16-w8a8
cargo test -p xrt-cuda --features cuda fp8_execution::tests --locked -- --include-ignored --test-threads=1
cargo test -p xrt-cuda --features cuda ordered_delta_matches_pinned --locked -- --ignored --test-threads=1
```

Create the frozen composed reference with `drama3_backbone_oracle.py --composed
--capture-reference --bf16-w8a8 --device cuda`; retain its source/fixture/input
digests. An older native trace may fail that command while its independent
reference outputs remain valid. Never treat that old native result as passing.

Primary references: [pinned PyTorch reduction](https://raw.githubusercontent.com/pytorch/pytorch/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/native/cuda/Reduce.cuh),
[pinned efficient attention](https://raw.githubusercontent.com/pytorch/pytorch/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h),
[NVIDIA PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k8).
