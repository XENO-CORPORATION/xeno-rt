# Drama 3 native runtime support

Scope: **xrt-audio**, with shared numeric and placement primitives.
Date: 2026-10-15. Implementation in progress; full-model support is not admitted.

Objective: execute the released Drama 3 graph and neural codec within XENO RT,
with measured CPU and bounded CUDA placement, directed speech and multi-speaker
conditioning. A cloud API or an upstream Python process is not this objective.

## Benchmark and failure lessons

| Leader | Concrete approach | Limitation or failure to avoid |
|---|---|---|
| Fish Speech / SGLang-Omni | FP8 MoE backbone, codebook decoder and separately budgeted codec stages; streaming, reference conditioning and prior VQ context; a newer GPTQ INT4 serving checkpoint reduces engine memory | The original FP8 serving tier targets 48 GB NVIDIA memory. The newer INT4 tier is a separate checkpoint/backend, not evidence that this native adapter supports INT4. Optimized sampling and fused tower paths can change sampled frames; concurrency can OOM when vocoder and engine shares overlap. |
| vLLM | Architecture-specific hybrid attention state, separate recurrent/KV caches, quantization-aware projection layouts | GatedDeltaNet projections do not all support the same block quantization; tensor-parallel packing and short-prefix cache behavior need explicit contracts. |
| llama.cpp / GGML | CPU-first quantized execution, explicit GPU/CPU expert placement and shared memory ownership | Reported Qwen3.5 hybrid offload corruption and fused-expert offload failures show why byte loading and model names do not establish numerical correctness. GPU expert caching remains an evolving design, not a performance guarantee. |

Sources: [Fish serving](https://sgl-project.github.io/sglang-omni/cookbook/fishaudio_drama3.html),
[Fish hardware](https://speech.fish.audio/install/),
[vLLM gated delta](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn/),
[GGML offload report](https://github.com/ggml-org/llama.cpp/issues/20140),
[fused expert report](https://github.com/ggml-org/llama.cpp/issues/20414).
Reports are evidence of possible failure modes, not assertions that every current
release remains affected.

The standard: immutable checkpoint identity; shape/dtype/scale validation;
independent numerical fixtures; explicit precision contracts; memory budgets and
leased cache ownership; cancellation/reset isolation; model-specific prompt and
codec handling; measured latency/RTF/quality on each admitted tier.

Relevant formats: SafeTensors storage bounds, OCP E4M3FN encoding, HTTP/OpenAI
speech compatibility and the repository's runtime-domain admission contract.

## Grade and implementation boundary

| Criterion | Existing XENO evidence | Initial grade / required work |
|---|---|---|
| Shared loading | `crates/xrt-safetensors/src/lib.rs`: mmap-backed format-neutral store | Partial: FP8 dtype and paired block-scale geometry required |
| Hybrid/MoE compute | `crates/xrt-models/src/llama.rs`, CPU delta kernel and CUDA hybrid kernels | Partial: Fish head expansion differs; exact mapping and FP8 experts required |
| FP8 execution | No block-FP8 native matvec before this work | Missing: CPU reference, dynamic W8A8, CUDA execution, Tensor Core batching |
| Speech controls | `crates/xrt-audio/src/direction.rs`: bounded chunk controls | Partial: scoped directions, speaker references, model-specific prompt rows |
| Audio graph and codec | Existing Chatterbox ONNX execution | Missing: Fish backbone/tower graph and 44.1 kHz neural codec |
| Resource admission | Shared allocation accounting and bundle integrity exist | Unverified for this model: host experts, bounded device cache, codec and state peaks |
| Service and quality | Existing narration jobs, word checks and cancellation | Unverified for Fish: preserve transcript separately from control tags and events |

**Overall: NOT ISG for Drama 3 yet.** Native utterances now run, but control
fidelity, performance and installed-tier gates remain open.

## Work packages, ordered by damage if omitted

1. Pin and validate every component, including codec and license. Upgrade;
   moderate implementation, high consequence of wrong tensor/scale association.
2. Native FP8 expert execution and explicit recurrent state. Shared-kernel
   upgrade; substantial performance work, independently testable before model load.
3. Fish-specific backbone, codebook decoder and neural codec. New adapter;
   major numerical implementation. Reuse existing primitives only after checking
   layouts, normalization, head expansion and state semantics.
4. Bounded cold-expert placement and hot-expert cache. Backend upgrade;
   substantial concurrency and transfer work. Quantization/offloading on 24 GB
   is a qualification target, not a claimed supported tier.
5. Immutable script plus structured directions, speakers and VQ continuation.
   Adapter/API upgrade; moderate integration, high risk of text/tag leakage or
   cross-request voice/cache reuse.
6. Actual long-form speech, multilingual control fidelity, latency/RTF,
   memory, cancellation, restart, packaged CPU/CUDA installation and rollback.
   Qualification; no timeline or performance claim until measured.

The public model is 35B/3B-active with FP8 experts. A 3B active-token workload
does not imply a 3B resident weight footprint. The pinned checkpoint and codec
total about 41.5 GB before runtime state. Full dequantization into resident FP32
would defeat the intended memory envelope; preserve encoded weights and execute
or stage selected experts within an explicit budget.

## Current grade

| Criterion | Current evidence | Grade |
|---|---|---|
| Immutable loading and numeric layout | Entire 16-file checkpoint installed and verified; encoded FP8/scales, mapped handles and exact codec layout | Partial: executable packaging and installed backend admission still open |
| Native graph and codec | Full 35B/3B-active backbone and tower; all 80 backbone and 26 tower FP32 trained-layer checks pass against pinned upstream code; DAC decoder and reference encoder | Partial: whole-stock autoregressive equivalence and production precision still open |
| Bounded CUDA placement | 4.92 GB encoded dense weights; explicit expert budget, bounded workspaces, shared arena, Windows live commit preflight and borrowed uploads; leases return to zero | Partial: exclusive physical peaks, cross-process and cross-platform host admission still open |
| Reproducibility and isolation | Worker-thread execution, busy rejection, cancel/deadline/disconnect/frame-limit discard; identical WAV after restart; native jobs reuse the durable ledger; seven generated authenticated capabilities | Partial: current installed-candidate verification is separate from production publisher authorization and full model admission |
| Directed/reference speech | Whispered and one-reference outputs transcribe exactly; reference encoder matches every tested code | Partial: voice identity/style listening gates not established |
| Multi-speaker controls | FP32 control cases fail/omit words; one BF16/W8A8 two-reference sample transcribes both lines, but unreferenced speech still exhausts its frame cap | Below: BF16 strict layer checks fail; speaker identity/style not qualified |
| Throughput and latency | Retained staged improvements and warm/cold measurements | Below: still substantially slower than real time on RTX 4090 |

The FP32 reference uses original BF16/FP8 encoded weights. GPU routed MLP uses
an approximate CUDA exponential. BF16/W8A8 is a separately selectable experimental
mode. Historical checks after gated DeltaNet normalization rounding passed
37/80 backbone and 1/26 tower comparisons (maximum relative RMS 2.20% and 0.884%).
Later stable BF16 RMS reduction improves the tower to 12/26 on CUDA and 7/26 on
CPU, but these strict composed-layer reports still fail. The historical backbone
report predates the later group-scale order and RMS-reduction corrections and
does not qualify their final composition. A fresh CPU backbone check after both
corrections passes 52/80 (largest relative RMS 0.576%). This is improved
agreement, still a failed strict fidelity gate.
The independent CPU W8A8 oracle emulates the pinned Triton operator; it does not
execute that CUDA serving kernel. Do not infer stock serving parity from speech
intelligibility or loosen these checks to declare the mode admitted.

The [experimental service](DRAMA3-NATIVE-SERVICE.md) is opt-in at compile time and
operator configuration. It exposes native speech, progress, cancellation and
unload without changing existing OpenAI speech routes or adding an unadmitted
model to discovery. Blocking workers retain shared audio admission across HTTP
disconnects. The managed bundle read lease prevents cooperating installers and
removers from mutating mapped weights during model ownership.

## Implemented component evidence

- CPU E4M3FN storage, nearest-even conversion, dynamic W8A8 and AVX2 execution
  have independent PyTorch numerical fixtures. Encoded weights remain encoded.
- Native CUDA execution and content-bound bounded LRU pass driver/NVRTC tests on
  RTX 4090, including eviction, context ownership and allocation cleanup.
- An actual trained 512x2048 expert projection from the SHA-256-verified first
  model shard agrees with PyTorch: CPU FP32 activation maximum absolute error
  5.01e-6, CUDA 5.73e-6; W8A8 CPU 2.39e-7 and CUDA 1.44e-6.
- Recurrent state matches PyTorch across successive steps. Repeat-interleave
  head expansion is distinct from the existing tiled Qwen mapping.
- Five canonical prompt fixtures match exactly: plain, directed, speaker
  switching, reference conditioning and prior-take continuation.
- Native DAC decoding of four explicit frames produces 8,192 samples at
  44.1 kHz. An independent PyTorch operator reconstruction of the tensor graph
  gives maximum sample error 1.49e-5 and SNR 100.46 dB. This does not establish
  full text-to-speech quality or equivalence to the upstream serving binary.
- The native audio feature passes 58 unit tests; the default workspace passes 446
  tests with 39 explicit ignores. Eight native CUDA driver tests run with ignores
  explicitly enabled and pass.
  The new dependency graph checks under the declared Rust 1.83 floor after
  pinning unicode-segmentation 1.12. Existing legacy clippy warnings remain;
  strict warning-free workspace lint is not claimed.
- All 40 trained backbone layers and 13 tower layers were compared at two
  successive teacher-forced positions against hash-pinned upstream modules.
  All 106 FP32 checks pass. Each layer receives its recorded native input;
  this isolates operators and retained state, rather than proving a complete
  upstream autoregressive trajectory.
- Corrected sampling defaults are temperature 0.7 / top-p 0.7 / top-k 30.
  Historical default mismatches remain explicitly labeled in retained receipts.
- The BF16/W8A8 plain and two-reference samples transcribe exactly as
  “The evidence remains clear.” and “Hello there. Welcome back.” This is
  scoped word fidelity, not a successful strict numerical or speaker-identity gate.
- Actual HTTP narration also passes independent English ASR, including the full
  longer sentence and its negation/number. Word comparison ignores punctuation
  and explicitly equates “seventeen” with “17”. The native speech endpoint itself
  did not run the existing word-check pipeline in those historical runs. The
  optional immutable-word verification path added later is documented separately.
- A fresh local clone with a separate build directory reproduces all 446
  workspace, 58 native-audio and 38 native-server tests. Its CPU-only executable
  generates the same 52-frame WAV as CUDA at SDK defaults, with zero GPU
  admission and no CUDA/ORT/Python/Torch module loaded. This closes the scoped
  CPU narration check; it does not qualify CPU performance or signed delivery.
- An expanded run aborted under Windows commit pressure before the host fix;
  its failed receipt is retained. Native uploads no longer retain cudarc's
  device-sized host vectors. A later resident private-memory sample is
  19,695,140,864 bytes, versus an earlier 38,728,433,664-byte snapshot. These are
  observations, not controlled or exclusive peak measurements.
- HTTP cancellation, a one-second deadline, client disconnect, partial-output
  refusal, unload to zero GPU admission, and byte-identical restart are exercised
  in the retained service harness. Native module inspection found driver/NVRTC,
  with no ONNX Runtime, Python or Torch module loaded in that server.

Full native speech has been independently transcribed with the intended words:
“The evidence remains clear.” and reference-conditioned “The runtime works
offline.” The negation/number case transcribes “Do not open door number 17.”
The existing local ASR endpoint is English-only, so it cannot qualify the Chinese
sample. Repeated plain requests produce the same WAV SHA-256 after cancellation
and after a direction request. This establishes scoped behavior, not model-wide
speech quality.

Native reference encoding of 90,112 samples produces 44 frames / 440 code ids,
all identical to an independent PyTorch tensor-graph reference. Optimized DAC
waveform comparison has 104.77 dB SNR and maximum sample error 6.74e-6.

Retained historical development measurements used temperature 1.0 / top-p 0.9
(before correcting the adapter to Fish SDK defaults 0.7 / 0.7).
They are not current default-performance claims.

Retained development measurements (8 Rayon threads for later runs; earlier
thread configuration was not retained, so these are not a controlled product
benchmark or an isolated causal speedup claim):

| Execution stage | Utterance generation time | Audio duration | RTF |
|---|---:|---:|---:|
| Initial host dense / CUDA expert reference | 216.12 s | 2.043 s | 105.77 |
| Expanded dense CUDA | 130.63 s | 2.043 s | 63.93 |
| Prepared immutable sources | 121.49 s | 2.043 s | 59.46 |
| Encoded BF16 dense, reusable expert scratch and SIMD codec | 67.98 s | 2.043 s | 33.27 |
| Fused routed MLP, warm owner | 16.75 s | 2.043 s | 8.20 |
| Dense workspace reuse, warm owner | 15.83 s | 2.043 s | 7.75 |
| Full CPU-only reference | 259.98 s | 2.043 s | 127.23 |

The CPU-only and final GPU runs preserve the same WAV hash as the preceding reference
backend. The last model load took about 29 s and first generation about 43 s;
weight verification/load is reported separately from inference. The shared GPU
arena reserved 18,949,505,024 bytes and released every lease after the tested
run. That is reservation evidence, not an exclusive physical peak measurement.

Retained receipts: [development evidence](../benchmark-results/audio/drama3-native-2026-10-14/README.md).
Further receipts: [2026-10-15 qualification](../benchmark-results/audio/drama3-native-2026-10-15/README.md).

Development commands, from this checkout:

```powershell
cargo run --release -p xrt-audio --features drama3-native --example drama3_prompt_probe -- <tokenizer.json> reference/audio/drama3/prompt-golden.json
cargo run --release -p xrt-audio --features drama3-native,cuda --example drama3_expert_probe -- <trained-projection-fixture>
cargo run --release -p xrt-audio --features drama3-native --example drama3_codec_probe -- <converted-codec> <codes.json> <new-output-prefix>
cargo run --release -p xrt-audio --features drama3-native,cuda --example drama3_speech_native -- <verified-model-bundle> <converted-codec> "The evidence remains clear." <new-output.wav> 8192
```

The last command is an unadmitted development path. It requires the entire
verified checkpoint, refuses exhausted-frame output, and reports the actual
CPU/hybrid FP32 reference backend. The public speech routes remain unchanged.

## Additional qualification work, 2026-10-15

The four requested workstreams remain separately accountable:

Current local checks pass 452 workspace tests (39 explicit ignores), 67
native-audio tests including integrations, 43 native-server tests and 23 bundle
tests. Nine CUDA FP8 tests were explicitly executed on the driver. The current
server/CUDA all-targets graph checks under Rust 1.83. Hosted runner execution is
not implied by these local results.

| Workstream | Implemented and measured | Still required for admission |
|---|---|---|
| BF16 fidelity | Corrected W8A8 group-dot scaling order on CPU/CUDA; BF16 stable RMS reduction; isolated tower audit passes 806/806 operators and the worst historical GDN layer passes 98/98 operator/state checks | Strict composed-layer comparisons still fail. Isolated input-conditioned checks and intelligibility do not establish stock autoregressive equivalence. |
| Broader quality and performance | Eight English categories across two seeds; fused native W8A8 experts; explicit immutable-word verification and 1–3 bounded attempts through existing CPU Whisper | Perceptual emotion/voice identity, multilingual and genuinely long narration, controlled cold/warm latency and peak memory, exclusive comparative performance and wider hardware/platform coverage |
| Capability/job integration | Seven generated token-gated capabilities, shared dirs discovery, typed durable jobs/results, progress attempts, idempotency, cancellation, restart interruption and retained WAV metadata | Whole-runtime capability parity is outside this native-audio subset. Current-artifact lifecycle and installed narration receipts must be assessed at their stated scope. |
| Signed delivery | Domain-separated Ed25519 manifest verification, independently pinned public-key identity, tamper rejection and real local candidate installation | Protected production publisher identity/trust provisioning and distribution authorization. A test identity is not production signing. Drama 3 weights remain unpublished. |

The fused BF16 development matrix completes 14/16 takes with one attempt; every
completed take passes strict independent lexical checking. Two takes exhaust
their frame cap, so this is a failed 16-case matrix, not a 100% quality pass.
The preceding unfused run completes 10/16. Shared-workstation warm RTF observations
are roughly 4.7–5 for the fused path versus about 6.4 previously; different sampled
waveforms, shared load and unisolated runs prevent a controlled speedup claim.
The first cold take includes cache warming. No real-time tier is advertised.

The optional word-verified mode retries a failed or incomplete take under the same
audio execution lease, with fresh request state and a different deterministic
seed. It publishes no result when the immutable spoken script is rejected.
Checker provisioning is explicitly separate from inference; missing/corrupt
installed checker files fail offline. Jobs bind the checker bundle digest.
Word fidelity remains separate from MOS, speaker identity and emotion control.

The installed development candidate passes all 16 English cases with at most two
attempts, plus missing-checker refusal before native loading, wrong-script refusal
with no result and cancellation during the second attempt. This candidate's
receipt binds its executable checksum. The later TLS-only patch has a separately
hashed and test-signed candidate; its lifecycle evidence is recorded separately.

A Criterion benchmark with eight synthetic 2048/512 experts on RTX 4090 measures
the warm fused W8A8 path at 0.256 ms versus 2.620 ms unfused. The benchmark first
checks matching outputs within its operator tolerance and verifies allocation
cleanup. It measures synchronized, cached expert execution with synthetic
weights, not full narration, cold transfer, Tensor Core throughput or exclusive
machine performance. The full model remains slower than real time.

See [service and jobs](DRAMA3-NATIVE-SERVICE.md),
[generated capabilities](AGENT-CAPABILITIES.md) and
[signed pinned delivery](PINNED-ARTIFACT-SIGNING.md). Formatting uses
`python scripts/format-workspace.py --check`: Cargo's `fmt --all` also rewrites
excluded path dependencies, which would break the exact shared-crate inventory.
`python scripts/check-vendored-primitives.py` verifies the pinned 30-file snapshot.

Dependency policy now checks the entire workspace rather than only the root
package graph. The 2026-10-15 advisory scan found four newly published rustls
advisories, fixed by updating 0.23.45 to 0.23.46. The inherited transitive
`paste 1.0.15` maintenance advisory remains visible; it has no safe version
upgrade and is not waived. Source/license checks and advisory results are
reported separately. The generic quick IP scan also flags Python reference
lockfiles and the first-party proprietary primitive; those findings are retained
and do not establish that Python dependencies are in the native executable.

## Rights and provenance

**Built with Fish Audio.**

Pinned model/config/license records are in `reference/audio/drama3/`.
Fish Audio Community License v1.0 is not Apache/MIT. Commercial revenue and
hosted-service conditions, attribution, output-use and voice-consent terms remain
applicable to native implementation and derived checkpoints. No model commercial
approval, public catalog admission, or redistribution decision is asserted here.
The codec exception in that license applies to Drama 3 usage, not general S2-Pro
weight admission. Product licensing remains a separate review before shipping.
