# Qwen3.6 stable-verifier graph result on RTX 4090

Date: 2026-08-10  
Domain: `xrt-text`  
Status: retained experimental result; not a production admission

## Result

XRT reached a retained mean of **150.1710 decode tok/s** for the pinned
Qwen3.6-27B Q4_K_S workload on one 24 GB RTX 4090. The final run contains 20
repetitions; the first three are warmups and the remaining 17 are the
registered samples.

| Statistic | Result |
|---|---:|
| Retained samples | 17 |
| Mean | 150.1710 tok/s |
| Median | 150.3018 tok/s |
| Sample standard deviation | 0.6554 tok/s |
| Normal 95% confidence half-width | 0.3116 tok/s |
| Minimum / maximum | 148.6076 / 151.1824 tok/s |
| Samples at or above 150 tok/s | 11 / 17 |

The mean clears 150 tok/s, but the confidence interval crosses 150 and six
individual samples are below it. This is therefore evidence for the measured
mean on this tuple, not a promise that every request or every RTX 4090 exceeds
150 tok/s.

All 17 retained runs produced 64 output tokens with no errors and identical
MTP telemetry: 55 accepted of 68 drafted tokens in nine verification batches.
The model SHA-256 is
`a5ef62184c1729c38c9565b502303ac88e2fad3b1c3c6aa430d9e273bdd7f917`
and its size is 16,121,357,440 bytes.

## Controlled progression

Every XRT row below was measured on the same active RTX 4090 pod, model,
prompt, sampler, and 64-token output cap. The short screens discard their
first repetition.

| Candidate | Retained mean | Effect |
|---|---:|---:|
| Pre-change eager verifier | 116.0165 tok/s | control |
| Stable dual-topology verifier graph | 137.2741 tok/s | +18.32% |
| Recurrent heterogeneous projection streams | 144.7952 tok/s | +5.48% |
| Recurrent and attention heterogeneous streams | 148.4991 tok/s | +2.56% |
| Immediate full-window capture screen | 150.9444 tok/s | short screen |
| Final 17-sample retained run | 150.1710 tok/s | +29.44% vs control |

The same-host pinned llama.cpp control used the same model artifact, prompt,
greedy settings, 64-token cap, and depth-eight MTP. After one warmup, its five
requests averaged 144.0091 tok/s. XRT's retained mean is 4.28% higher for this
narrow paired workload. This comparison does not imply broad runtime
superiority; a multi-prompt and long-context comparison remains required.

## What changed

- The verifier keeps two stable CUDA graph executables, one for each recurrent
  state-buffer generation, instead of recapturing or updating a graph every
  proposal window.
- Changing positions and row metadata live in stable device decode parameters;
  Q4 embeddings write directly into stable scratch allocations.
- Full-attention verification now uses the shared-page F32 KV topology used by
  production sessions, including stable key/value page-pointer tables.
- Q4_K, Q5_K, and dense projections are scheduled on heterogeneous CUDA streams
  in both recurrent and full-attention layers.
- DeltaNet device commits are separated from host handle publication, and F32
  KV lengths are committed once per layer after graph launch instead of once
  per row and layer.
- Device argmax is captured in the verifier graph. Only the bounded argmax
  index vector is returned for the supported unpenalized greedy path.
- Partial final windows retain an eager fallback. The optimization is opt-in
  through `XRT_QWEN_MTP_VERIFY_GRAPH=1`.

An Nsight node trace of the heterogeneous candidate reduced the fourth verify
window from a 39.404 ms pre-schedule span to 35.261 ms. The latter contained
32.520 ms of kernel-union time and 2.741 ms of launch gaps. That trace predates
the final graph-argmax and immediate-capture changes; the 20-run JSON, not the
trace, is the authoritative final throughput record.

The repeatedly recaptured graph, Q5_K N32/N64 experimental tiles, and grouped
gate/up projection candidates were not retained. They either failed to
improve end-to-end throughput or removed useful stream overlap.

## Reproduction

The benchmark helper records the complete controlled command and permits all
environment values to be overridden:

```bash
XRT_QWEN_MTP_VERIFY_GRAPH=1 \
  XRT_CLI_BIN=/workspace/target/release/xrt-cli \
  scripts/benchmark-qwen36-mtp.sh \
  /workspace/model/Qwen3.6-27B-Q4_K_S.gguf \
  20 \
  /workspace/profiles/xrt-final-retained-20c.json
```

The helper defaults to CUDA-resident execution, F32 KV, disabled prefix cache
and n-gram speculation, depth-eight Qwen MTP, a 65,536-row draft projection,
batched rebase, Marlin Q4_K, tensor-core verifier kernels, parallel projection
streams, greedy sampling, seed 424242, and the registered counting prompt.

Relevant generated PTX SHA-256 values:

- `qwen35_verify_attention.ptx`:
  `7490d9c989184dc2cb4e299daca62b124dd41d3d488ba37d806d5b52fa3b0d4a`
- `kquant_mmq.ptx`:
  `089b350641d33ccc768c26b39c26378279316ac702771e46f4bf71506c00a114`
- `q4_k_recurrent.ptx`:
  `ce7e2a9a047c0f2d60d169eb1d75ffb2e49f6f527264b5d798a5d30ff0a00041`

Artifact SHA-256 values:

- `xrt-final-retained-20c.json`:
  `cbb6ff983cd8eee2514df85c7b56bebffe310e41239f291afe0d94d3a74318eb`
- `xrt-final-node-window4.nsys-rep`:
  `4a51dadcc6d440dffb31be8a25b601b32ed13533732884e496708531e4e6a98e`
- `xrt-final-node-window4.sqlite`:
  `d1bc24a3f7caa86e703d9bdff6df58ac4da0f22b0a6c6af937a50b23bf3b47be`

## Evidence inventory

- `xrt-final-retained-20c.json`: authoritative final raw XRT run.
- `xrt-baseline-6.json`: same-host eager control.
- `xrt-stable-graph-6.json`: stable-graph screen.
- `xrt-heterogeneous-projections-6.json`: recurrent projection screen.
- `xrt-all-heterogeneous-projections-6.json`: recurrent plus attention screen.
- `xrt-immediate-graph-capture-8.json`: final short candidate screen.
- `llama-1.json` through `llama-6.json`: same-host llama.cpp control requests.
- `xrt-final-node-window4.nsys-rep` and its exported SQLite database: retained
  structural profiler evidence.
- `xrt-stable-graph-node-window4.sqlite`: pre-heterogeneous graph node trace.

## Admission boundary

This result covers one model hash, quantization, prompt, greedy sampler, short
context, one request stream, F32 KV cache, and one RTX 4090. MTP and the stable
verifier graph remain experimental and disabled by default. Multi-prompt token
parity, exact non-greedy speculative sampling, long-context correctness,
concurrency, cancellation, memory-pressure, reliability, security, clean
container reproduction, packaging, and cross-hardware gates remain open.
