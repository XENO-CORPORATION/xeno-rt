# GPU Runtime Acceleration Spec

Status: Draft implementation spec, Phase 6 cooperative scheduling/chunked prefill RTX validated; fused multi-sequence CUDA decode pending
Date: 2026-06-19
Primary target: NVIDIA RTX 4090-class desktop GPUs

## Objective

Turn xeno-rt from a CPU-first GGUF runtime with auxiliary CUDA kernels into a GPU-resident inference runtime that can compete with ExLlama, vLLM, and SGLang on local desktop hardware, while preserving GGUF support, CPU fallback, and OpenAI-compatible APIs.

The goal is not to replace xeno-rt with an external Python server. The goal is to port the right systems ideas into xeno-rt:

- persistent GPU weights
- persistent GPU scratch buffers
- GPU-resident paged KV cache
- fused quantized matvec/GEMM kernels
- fused decode attention
- CUDA Graph decode replay
- continuous batching and prefix cache
- optional external-runtime adapter for benchmarking and fallback

## Final Arrival Point

The final native CUDA target is:

- `XRT_BACKEND=cuda` and `--backend cuda` run real GGUF Q4_K/Q6_K/Q8_0 local models end to end.
- Model weights, KV cache, and decode scratch stay GPU-resident across tokens.
- Batch-1 decode is materially faster than CPU on RTX 4090-class hardware.
- OpenAI-compatible `/v1/chat/completions`, streaming, `/v1/models`, CLI generation, and benchmark JSON stay API-compatible.
- `XRT_BACKEND=auto` picks CUDA only when the model/runtime can actually decode on CUDA, otherwise CPU fallback works.
- Unsupported architectures or quantizations fail clearly, never silently produce CPU-labeled-as-CUDA output.
- Reproducible benchmark reports prove CPU fallback, CUDA load, CUDA decode, and server behavior.

## Hard Constraints

- OpenAI-compatible `/v1/chat/completions`, `/v1/completions`, `/v1/models`, and streaming response shapes must not break.
- GGUF remains the first-class model format.
- CPU fallback must continue to work without CUDA.
- Existing Llama, Qwen2/Qwen2.5, Qwen3, Qwen3.5, Gemma4 dense text, tokenizer, sampler, and server tests must continue passing.
- Every performance change requires a benchmark gate.
- CUDA is optional at build time and runtime.
- GPU path must fail explicitly when unsupported, then fall back or return a clear error depending on user configuration.

## Research Summary

External runtimes point to the same core lesson: speed comes from keeping the whole decode path on GPU and removing per-token overhead.

Important observations:

- ExLlamaV2 is archived; development moved to ExLlamaV3. ExLlamaV3 is the better current reference for consumer GPU inference, EXL3 quantization, continuous batching, cache quantization, speculative decoding, and TabbyAPI serving.
- vLLM's relevant ideas are PagedAttention, continuous batching, chunked prefill, prefix caching, CUDA/HIP graph execution, and broad quantization support.
- SGLang's relevant ideas are RadixAttention, prefix caching, CUDA Graph decode, piecewise CUDA Graph for dynamic prefill, and high-throughput OpenAI-compatible serving.
- FlashInfer's relevant ideas are reusable LLM serving kernels for attention, page attention, LoRA, and backend-specialized attention/GEMM execution.

Sources:

- ExLlamaV2 README: https://github.com/turboderp-org/exllamav2
- ExLlamaV3 README: https://github.com/turboderp-org/exllamav3
- vLLM docs: https://docs.vllm.ai/
- vLLM quantization docs: https://docs.vllm.ai/en/latest/features/quantization/
- SGLang docs: https://docs.sglang.io/
- SGLang CUDA Graph docs: https://sgl-project-sglang-93.mintlify.app/optimization/cuda-graph
- FlashInfer docs: https://docs.flashinfer.ai/

## Current xeno-rt State

Current strengths:

- GGUF parser and mmap loader are stable.
- Tokenizer and OpenAI-compatible server already exist.
- CPU kernels support quantized GGUF formats.
- Runtime has paged KV cache modes: F32, Q8, key-Q4/value-Q8, and agent-adaptive.
- Model path supports Llama-like, Qwen, Qwen3.5 hybrid, and Gemma4 dense text.
- `xrt-cuda` exists behind a feature flag with standalone PTX kernels.

Current gaps:

- Native CUDA decode currently supports standard dense F32/F16/BF16/Q8_0/Q4_0/Q4_K/Q5_K/Q6_K paths.
- Real VibeThinker 3B Q4_K_M and Gemma4 12B Q4_K_M models can load and run the CUDA decode path; broader multi-token correctness and throughput validation are not production-ready yet.
- Q4_0 exists as an expanded resident primitive and is wired for token embeddings plus dense projection/output matrices.
- Batch decode is sequential, not fused prefill or continuous batching.
- CUDA KV cache modes use device page tables and GPU-side logical-to-physical addressing, but there is not yet a central page allocator with eviction or prefix reuse.
- Scratch buffers are not managed by a central GPU arena yet.
- Peak VRAM telemetry and a central GPU allocation arena are not wired. Batch-1 CUDA Graph replay is wired for standard dense decode with F32 KV; Gemma4, quantized KV, and larger batch graph variants still use eager CUDA.

## Progress Log

- 2026-06-19: Added the final arrival point and explicit definition of done for native CUDA inference.
- 2026-06-19: Added standard dense Q8_0/F32 CUDA decode slice with resident KV and sequential batch support.
- 2026-06-19: Added resident Q4_0 and Q4_K matvec primitives; Q4_K now defaults to packed resident K-quant payloads.
- 2026-06-19: Wired expanded Q4_K token embeddings, projection, and output matrices into the CUDA runtime decode path for synthetic dense models.
- 2026-06-19: Added expanded Q6_K token embedding, projection, and output support to the same CUDA runtime decode path.
- 2026-06-19: Real VibeThinker 3B Q4_K_M CUDA-feature benchmark reaches `cuda-resident` without the previous unsupported decode error; output/performance validation remains pending.
- 2026-06-19: Tried CUDA F32 `m == 1` matvec fast paths for expanded Q4_K/Q6_K matrices; both per-output and grouped-atomic variants were reverted after the real VibeThinker benchmark timed out past 10 minutes.
- 2026-06-19: Added an explicit packed Q4_K prototype path (`upload_q4_k_matrix_packed`, packed matvec, packed embedding). Synthetic CUDA parity passes, but the real VibeThinker benchmark timed out past 10 minutes, so runtime Q4_K remains on the expanded F32 path by default.
- 2026-06-19: Fixed benchmark token accounting so `output_tokens` reports generated token count instead of non-empty decoded text chunks. Clean VibeThinker one-token smoke: CPU `24.0s`, CUDA-resident `411.8s`, both with `output_tokens: 1`.
- 2026-06-19: Checked the shortest throughput path. Local CUDA install does not expose cuBLAS DLLs, and an inline shared-reduction F32 GEMV PTX entry was rejected by the driver JIT and reverted. Next throughput work should use build-time validated PTX/CUBIN or install/link cuBLAS explicitly.
- 2026-06-19: Made the packed Q4_K prototype block-parallel instead of accidentally running only `tid == 0`; synthetic CUDA parity still passes. Tried making packed Q4_K the runtime default again, but the real VibeThinker one-token CUDA smoke still timed out past 10 minutes, so default Q4_K was reverted to expanded F32.
- 2026-06-19: Added CUDA driver JIT-log capture on PTX module-load failures. Future invalid PTX errors should include the driver compiler log instead of only `CUDA_ERROR_INVALID_PTX`.
- 2026-06-19: Replaced Q8_0/Q4_0 matvec global atomic accumulation with an in-block shared-memory reduction. CUDA synthetic parity passes.
- 2026-06-19: Replaced packed Q4_K global atomic accumulation with in-block shared-memory reduction and made packed Q4_K the runtime default. VibeThinker 3B Q4_K_M one-token CUDA smoke: load `23.2s`, generation `4.8s`, `output_tokens: 1`, `resident_dense_quant_decode_available: true`.
- 2026-06-19: Tried a packed Q6_K matvec prototype. Synthetic CUDA parity passed, but real VibeThinker one-token CUDA smoke regressed badly (`146.8s` generation, then timeout on rerun), so the packed Q6_K prototype was deleted and Q6_K stays on the expanded resident F32 path.
- 2026-06-19: Added `XRT_CUDA_PROFILE=1` stage timing. Profiling showed the post-Q6 rollback real-model stall was not the layer path: Q4_K token embedding first use spent about `60s` before layer 0, while layers were roughly `1-6ms`. Q4_K token embedding now temporarily CPU-dequantizes the requested row and uploads one F32 activation to avoid the packed embedding kernel first-use stall.
- 2026-06-19: Verified the Q4_K token-embedding bridge on VibeThinker 3B Q4_K_M. CUDA smoke completed with load `17.9s`, generation `3.64s`, `output_tokens: 1`, no backend error. Profiled token embedding is now about `1.03s` on the first prompt token and about `0.10ms` after warmup; warmed token total is about `127ms`, with final logits now the largest stage at about `73ms`.
- 2026-06-19: CUDA `forward_batch` now advances KV state without computing final logits for intermediate prompt tokens; it computes logits only for the final prompt token, matching what generation needs. On the 20-token VibeThinker smoke this was correctness cleanup rather than a throughput win: clean CUDA smoke remained about `3.65s` generation.
- 2026-06-26: After local machine instability, added and validated `scripts/safe-cuda-check.ps1` for serial CUDA-backend compile/unit verification with `CARGO_BUILD_JOBS=1`, `RUST_TEST_THREADS=1`, profiling disabled, timeout handling, and post-run process checks. Added `scripts/safe-cuda-smoke.ps1` as the only opt-in real-model smoke wrapper; it requires `-ConfirmGpuRun`. No real-model CUDA smoke should run by default.
- 2026-06-26: Removed one host-side full-vocabulary logits copy in the CUDA decode path by moving the downloaded logits vector into the output buffer.
- 2026-06-26: `CudaResidentBackend::forward_batch_with_embeddings` now supports per-position embedding overrides by uploading override vectors into the dense quant decode path, enabling the CUDA prefill path to accept multimodal patch embeddings instead of immediately returning unsupported.
- 2026-06-26: `CudaResidentBackend::embedding_lookup` now delegates to the already-loaded CPU model embedding lookup for compatibility-only callers; CUDA decode still uses resident token embedding paths.
- 2026-06-26: CUDA embedding overrides are preflight-validated for in-range positions and exact embedding width before prefill mutates resident KV state.
- 2026-06-26: Dense CUDA layer loading/execution now supports paired F32 Q/K head-norm tensors (`attn_q_norm.weight`, `attn_k_norm.weight`) by applying existing CUDA RMSNorm per attention head before RoPE.
- 2026-06-26: Dense CUDA layer/output loading now accepts F32 linear tensors by uploading transposed resident F32 matrices and routing them through the existing resident RHS matmul path.
- 2026-06-26: Resident CUDA float-tensor upload now accepts GGUF `F16` and `BF16` tensors by converting them once into device F32 buffers, so norms, biases, token embeddings, and F16/BF16 dense linear weights can reuse existing F32 CUDA kernels.
- 2026-06-26: Dense CUDA layer/output loading now wires GGUF `Q4_0` projection/output matrices through the existing expanded resident Q4_0 matvec path.
- 2026-06-26: CUDA token embedding loading now accepts GGUF `Q4_0` by reusing the expanded resident Q4_0/Q8_0 embedding device path.
- 2026-06-26: Dense CUDA layer/output and token embedding loading now accept GGUF `Q5_K` by dequantizing once into resident transposed F32 storage and reusing the existing F32 matmul/embedding paths. The safe CUDA wrappers now prefer the real stable Cargo binary over the PATH shim, reject pre-existing Rust/XRT processes, print command lines for leftovers, and cover the xrt-cuda Q5_K no-device conversion test.
- 2026-06-26: Expanded Q5_K/Q6_K CUDA token-embedding uploads now keep a row-major F32 copy in addition to transposed F32 matvec storage, fixing the previous embedding layout mismatch without doubling every projection/output matrix.
- 2026-06-26: `GpuResourceStatus.active_sessions` now reflects live runtime sessions instead of a hardcoded zero; the counter is shared across `load_vision` clones and saturates on unregister.
- 2026-06-26: Session-level GPU resource status now includes the active session's CUDA KV allocation, and benchmark JSON records that session snapshot instead of a runtime-level zero-allocation placeholder.
- 2026-06-26: Safe CUDA wrappers now use `Get-Process` instead of CIM/WMI for pre/post Rust process guards, avoiding slow WMI process enumeration under machine load.
- 2026-06-26: `scripts/safe-cuda-check.ps1` now includes `xrt-cli --features cuda` compile coverage so benchmark/reporting changes are checked without running a model.
- 2026-06-26: `scripts/safe-cuda-check.ps1` now also includes `xrt-server --features cuda` compile coverage so OpenAI-compatible server/status changes are checked without running a model.
- 2026-06-26: CUDA backend sessions now update layer-count/width geometry when replacing cache mode, while preserving the session context length; the safe check covers this without allocating GPU memory.
- 2026-06-26: `scripts/safe-cuda-check.ps1` now compiles the `xrt-runtime --features cuda` test binary with `--no-run`, catching feature-gated test/build regressions without executing CUDA.
- 2026-06-26: The layer-0 CUDA projection debug probe now rejects nonzero positions instead of silently running attention with an empty prefix KV cache.
- 2026-06-26: The narrow legacy `resident_q8_0_probe_available` status now requires both token embedding and output projection to be Q8_0; broad dense CUDA readiness remains reported by `resident_dense_quant_decode_available`.
- 2026-06-26: Safe CUDA wrappers keep process guards on lightweight `Get-Process` only and re-check after every cargo/build/run step, preventing validation commands from stacking on orphaned cargo/rustc children without invoking slow WMI/CIM enumeration.
- 2026-06-26: `scripts/safe-cuda-check.ps1` now builds test binaries once and runs focused test filters through the generated executables, avoiding repeated `cargo test` process stacks during crash-safe validation.
- 2026-06-26: Safe CUDA wrappers now remember pre-command Rust/XRT process IDs and stop only new leftover cargo/rustc/xrt children that fail to drain after validation.
- 2026-06-26: CUDA `forward_batch_all_logits` now checks all-logits output length with overflow handling before reserving host logits storage.
- 2026-06-26: CUDA token and batch decode now check position arithmetic overflow before cache preparation or per-token forward calls.
- 2026-06-26: Session speculation/replay now checks batch-length arithmetic and all-logits slice bounds before indexing backend logits, turning malformed backend output into runtime errors instead of panics.
- 2026-06-26: Multimodal embedding override construction now checks image patch counts and embedding slice arithmetic before feeding override vectors into CPU/CUDA prefill.
- 2026-06-26: OpenAI-compatible image preprocessing now rejects zero/overflowing image tensor sizes instead of panicking while allocating CHW tensors.
- 2026-06-26: The layer-0 CUDA projection debug probe now preserves the normal `Ok(None)` unavailable-probe behavior before applying the position-0 guard.
- 2026-06-26: CUDA backend construction now queries CUDA free/total VRAM and rejects GGUF model upload before bulk tensor allocation when raw tensor bytes exceed the configured safe budget from `XRT_GPU_MEMORY_FRACTION` and `XRT_GPU_RESERVED_MB`.
- 2026-06-26: Runtime GPU resource status now reports CUDA device name and total VRAM when the CUDA backend is active, reusing the same device telemetry queried during upload preflight.
- 2026-06-26: CUDA upload preflight and `model_weight_bytes` status now estimate dense-decode duplicate resident buffers, including expanded F32 Q5_K/Q6_K tensors, instead of checking/reporting only raw GGUF tensor bytes.
- 2026-06-26: `resident_dense_quant_decode_available` now requires a GPU-resident token embedding; models that still use the Q4_K CPU-row embedding fallback can run through the CUDA path but are no longer reported as fully resident dense CUDA decode.
- 2026-06-26: Replaced the Q4_K CPU-row token embedding bridge with an expanded resident F32 embedding upload. Q4_K token embeddings now keep transposed and row-major F32 buffers on GPU, matching the Q5_K/Q6_K embedding strategy while avoiding the packed Q4_K embedding first-use stall.
- 2026-06-26: `scripts/safe-cuda-check.ps1` now also compiles the root `xrt-workspace-tests --features cuda` integration-test binaries with `--no-run`, so ignored CUDA parity tests are checked without executing GPU/model code.
- 2026-06-26: Added a synthetic Q5_K GGUF fixture and ignored CUDA runtime parity test, closing the test-coverage gap between the advertised Q5_K dense CUDA path and the existing Q4_K/Q6_K parity tests.
- 2026-06-26: Added a synthetic Q4_0 GGUF fixture and ignored CUDA runtime parity test, so every advertised dense quantized CUDA format now has a synthetic runtime parity test compiled by the safe gate.
- 2026-06-26: Added synthetic F16 and BF16 GGUF fixtures plus ignored CUDA runtime parity tests, so the advertised dense float CUDA formats are compiled by the safe gate without running GPU/model code.
- 2026-06-26: Safe CUDA validation now compiles the default workspace test binary and runs a focused CPU fallback decode test for the synthetic F16/BF16 fixtures, proving those GGUFs load and decode without requiring CUDA hardware.
- 2026-06-26: Added `xrt-cuda::CudaQ8LayerKvCache` allocation and byte accounting as the first quantized GPU-KV building block. Runtime CUDA sessions still report/use F32 KV until Q8 append and attention kernels are wired.
- 2026-06-26: Added scalar CUDA Q8 KV append/dequantize entry points plus an ignored CUDA roundtrip parity test. This proves the API shape and test coverage for Q8 GPU-KV without switching runtime sessions away from F32 KV yet.
- 2026-06-26: Added a Q8 KV attention bridge API that dequantizes Q8 cache rows into a temporary GPU F32 cache and reuses the existing single-query attention kernel. This gives Q8 GPU-KV a correctness-first attention read path before adding a fully fused Q8 attention kernel.
- 2026-06-26: CUDA-feature runtime sessions can now select explicit `XRT_KV_CACHE_MODE=q8`, allocate `CudaQ8LayerKvCache` stores, and dispatch dense decode attention through the Q8 bridge.
- 2026-06-26: `GpuResourceStatus` now includes nullable `free_vram_bytes`, `requested_kv_cache_mode`, `kv_cache_mode`, and `kv_budget_bytes` fields. Runtime-level status keeps session modes `null`, while live session status reports both the requested mode and the effective mode (`f32`, `q8`, `kq4_vq8`, or `agent_adaptive`) used for CUDA KV allocation/accounting.
- 2026-06-26: `scripts/safe-cuda-smoke.ps1` accepts `-CacheMode`, defaulting to `f32`, so the real-GPU smoke can explicitly run `-CacheMode q8`, `-CacheMode kq4_vq8`, or `-CacheMode agent_adaptive` only when hardware execution is approved.
- 2026-06-26: `scripts/safe-cuda-check.ps1` now verifies that `safe-cuda-smoke.ps1` rejects invalid `-CacheMode` values before build or GPU confirmation.
- 2026-06-26: Safe CUDA wrappers now use a longer 30-second clean-exit soak so delayed `cargo`/`rustc` children are killed before the scripts report success.
- 2026-06-26: Safe CUDA wrappers now use a long clean-exit soak after delayed stale-`cargo` recurrences. `safe-cuda-check.ps1` also runs `gpu_resource_status_tracks_active_sessions`, covering requested/effective KV status without executing CUDA hardware.
- 2026-06-26: Direct Q8 KV attention is wired as `single_query_attention_q8_kernel` using the same supported 11-argument cudarc launch shape as the F32 attention kernel. Compile-only CUDA validation passes; real GPU parity remains behind ignored/manual tests.
- 2026-06-26: Safe CUDA wrappers now stop leftover Rust/XRT process trees with `taskkill /T /F` and perform a clean-exit soak before reporting success. This guards against orphaned `cargo`/`rustc` processes after crash-safe validation.
- 2026-06-27: `safe-cuda-check.ps1` now repeats post-command Rust/XRT process-tree cleanup until the process table is quiet, covering delayed `cargo`/`rustc` child waves seen during workspace CUDA test-binary compilation.
- 2026-06-26: Added CUDA key-Q4/value-Q8 KV cache storage, append/dequantize kernels, runtime `XRT_KV_CACHE_MODE=kq4_vq8` selection, and direct `single_query_attention_kq4_vq8_kernel` wiring. Compile-only validation passes; real GPU parity remains behind ignored/manual tests.
- 2026-06-27: CUDA `XRT_KV_CACHE_MODE=agent_adaptive` now resolves to a mixed hot-F32/cold-KQ4-VQ8 GPU cache instead of the uniform KQ4/VQ8 interim mode. Batch prefill classifies positions against the full batch length so old unpinned prompt tokens can enter cold storage immediately; runtime preparation now rebuilds compact hot/cold caches when existing rows age into a different route. The rebuild is row-by-row and correctness-first; fused in-GPU migration remains pending.
- 2026-06-26: Added ignored CUDA runtime smoke coverage for Q8_0 decode with `q8`, `kq4_vq8`, and `agent_adaptive` KV modes. The safe gate compiles this test without executing GPU hardware; manual hardware runs can now validate that quantized GPU-KV modes decode finite logits.
- 2026-06-26: CUDA sessions now retain `SessionPolicy`, prompt token count, and prompt spans so the future mixed hot-F32/cold-quantized adaptive KV router has the same policy metadata already used by CPU `AgentAdaptive` cache.
- 2026-06-26: CUDA sessions now expose a tested adaptive hot-position routing predicate and byte mask matching CPU `AgentAdaptive`: recent-window tokens stay hot, and pinned system/developer/tool spans stay hot within the configured prompt token count. The correctness bridge consumes this route; the fused mixed-attention fast path is still pending.
- 2026-06-26: `xrt-cuda` now has a correctness-first mixed hot-F32/cold-KQ4-VQ8 single-query attention bridge. It reconstructs a temporary F32 cache from compact hot and cold caches, then reuses the existing F32 attention kernel. This avoids a risky new fused kernel while giving the runtime a real mixed-cache primitive to wire next; it is not the final fast path.
- 2026-06-26: CUDA session KV allocation now checks the estimated full-context cache size against `XRT_GPU_KV_FRACTION` before allocating GPU cache buffers. Oversized F32/Q8/KQ4-VQ8 cache requests fail with a clear budget error instead of attempting a risky allocation.
- 2026-06-26: `agent` is accepted as a `KvCacheMode` alias for `agent_adaptive`, matching cache-policy parsing and the safe CUDA smoke wrapper.
- 2026-07-09: Added a serialized, manually dispatched GitHub Actions CUDA validation workflow on the repository self-hosted Windows RTX 4090 runner. It uses one Cargo build job, one Rust test thread, bounded process cleanup, a persistent isolated target directory, and opt-in real-model parity/smoke inputs.
- 2026-07-10: The RTX 4090 gate passes all low-level resident kernels and synthetic runtime decode cases for F32/F16/BF16/Q8_0/Q4_0/Q4_K/Q5_K/Q6_K plus F32/Q8/KQ4-VQ8/agent-adaptive KV. Synthetic K-quant parity uses a bounded `0.05` maximum logit delta because the CPU SIMD reference quantizes activations to Q8_0 while CUDA consumes resident F32 activations; real-model top-token agreement remains the semantic gate.
- 2026-07-10: Real VibeThinker parity exposed unsafe full-context CUDA KV preallocation (`9.66 GB` F32 KV against a `4.03 GB` budget) before token 0. Replaced that policy with page-sized, doubling grow-on-demand capacity for all CUDA KV modes. Growth preserves resident rows with device-to-device copies, enforces model context length, and checks temporary old-plus-new allocation peak against the configured KV budget. RTX 4090 revalidation passes the F32/Q8/KQ4-VQ8 resize tests and every synthetic runtime case; the real model now reaches logits but still fails strict top-token parity (`CUDA 151643`, `CPU 9707`), so correctness remains open.
- 2026-07-10: Zero-layer real-model diagnostics localized the first major parity fault before transformer layer 0. VibeThinker's tied Q6_K token embedding/output produced a `59.63` maximum logit delta because the expanded K-quant embedding PTX indexed a row-major upload as column-major. The kernel now indexes `token * hidden + column`, and the low-level CUDA gate includes an exact expanded K-quant embedding-row test. RTX 4090 run `29085693045` passes the entire serial gate and strict real-model top-token parity: zero-layer max delta `0.0000505`, layer-1 max delta `0.0468`, and full 36-layer max delta `0.4407`, with CPU and CUDA both selecting token `9707`.
- 2026-07-10: Bounded end-to-end CLI smoke run `29086022962` passes on VibeThinker 3B Q4_K_M with F32 GPU KV and `max_tokens=1`: load `18.78s`, generation `1.044s` (`0.957 tok/s`), `8.68 GB` reported resident model allocation, `2.36 MB` allocated KV, and no backend error. Same-commit RTX 4090 comparison run `29086451766` records CPU `22.397s` (`0.0446 tok/s`) versus CUDA `0.991s` (`1.009 tok/s`) for one generated token, a `22.6x` CUDA speedup with no backend errors. This satisfies the initial Phase 2 faster-than-CPU decode criterion; multi-token steady-state reports remain required for later optimization phases.
- 2026-07-10: Removed the unused second GPU copy of every raw GGUF tensor from `CudaResidentBackend`; supported dense CUDA models now retain only the layouts consumed by decode kernels. Quantized tied token-embedding/output matrices share the same resident allocation through `Arc`, including VibeThinker's expanded Q6_K transposed/row-major pair. Full parity run `29087144051` passes. Comparison run `29087358930` reduces reported resident weights from `8.684 GB` to `5.515 GB` (36.5% less) and CUDA load from `21.66s` to `16.30s`; one-token CUDA decode remains in the prior range at `1.071s` and is `21.5x` faster than the same-run CPU result (`22.995s`).
- 2026-07-10: Benchmark JSON rows now include an additive `repetition` field, `xrt-cli bench` accepts bounded `--repetitions`, and the manual RTX 4090 workflow runs two serial CPU and CUDA repetitions. This separates first-use/JIT-sensitive timing from a warm repeated sample without changing the default one-run CLI behavior.
- 2026-07-10: Repeated comparison run `29087948209` passes the serial compile gate and records two error-free one-token samples per backend. CPU takes `21.365s` and `21.938s` (`0.0468` and `0.0456 tok/s`); CUDA takes `1.056s` and `0.903s` (`0.947` and `1.107 tok/s`) after a `15.455s` model load. The two-sample means are `21.651s` CPU and `0.980s` CUDA, a `22.1x` decode speedup. Reported resident CUDA allocations remain stable at `5,515,198,464` model bytes and `2,359,296` KV bytes.
- 2026-07-10: Added reusable destination-buffer CUDA APIs for upload/copy, RMSNorm, F32 matmul, Q8_0/Q4_0/Q4_K/Q5_K/Q6_K matvec, SiLU, add, and multiply. `BackendSession` now retains a geometry-checked decode scratch arena for Q/K/V, normalization, FFN, and logits buffers and reports its allocation through `scratch_allocated_bytes`. Full RTX 4090 run `29089139201` passes low-level, synthetic, and VibeThinker real-model parity with the same full-model maximum logit delta (`0.44067883`) and CPU/CUDA top token (`9707`). The two CUDA samples improve to `0.938s` and `0.861s` (`1.067` and `1.161 tok/s`), averaging `0.899s`, while the same-run CPU mean is `22.737s`; persistent scratch is `723,456` bytes. This is an `8.2%` improvement over the preceding two-sample CUDA mean and a `25.3x` same-run CPU/CUDA decode speedup.
- 2026-07-10: Tried extending scratch residency to embedding, attention output, and ping-pong layer outputs. Correctness remained exact at the existing parity bounds in run `29090211205`, but repeated CUDA decode regressed to `1.831s`/`2.222s`, and confirmation run `29090643946` remained slow at `2.364s`/`2.326s`. An identical baseline workflow at the prior scratch commit (`29091015392`) produced `1.158s`/`0.927s`, so commit `6426b5f` was reverted. Do not retry broad output-buffer reuse without CUDA stage profiling that identifies the synchronization/allocation tradeoff; the smaller validated `723,456`-byte scratch arena remains the default.
- 2026-07-10: Four-token RTX 4090 comparisons established the first multi-token baseline. F32 CUDA completed in `1.423s`/`1.152s` versus CPU `27.496s`/`27.129s`; Q8 CUDA completed in `1.317s`/`1.150s` and reduced live KV from `2,359,296` bytes to `599,040` bytes. Both modes produced the same four-token preview as CPU in these seeded runs.
- 2026-07-10: A KQ4/VQ8 audit found that CUDA used 32-element key scale groups while the CPU cache contract uses 64, and the fused CUDA attention kernel selected a scale by KV-head rather than by the packed element group. CUDA allocation, append, dequantize, attention, and VRAM accounting now use the CPU 64-element contract. A 128-wide two-scale GPU regression test passes in run `29101965892`; controlled real-model positions 0-3 retain CPU/CUDA top-token agreement for Q8 and KQ4/VQ8.
- 2026-07-10: Corrected KQ4/VQ8 four-token run `29102410042` completed without backend errors at `1.339s`/`1.325s`, versus CPU `28.925s`/`29.735s`, with `465,408` live KV bytes and `723,456` scratch bytes. Its seeded sampled preview still differs from CPU (`.K.G` versus `  We Okay`) even though controlled greedy top-token parity passes; sampled text equality is therefore not treated as a correctness proof while CPU/CUDA logits remain non-bit-identical.
- 2026-07-10: The synthetic quantized-KV hardware test now compares CPU and CUDA logits for Q8, KQ4/VQ8, and agent-adaptive over four sequential tokens. Agent-adaptive uses a forced one-token hot window, causing prior rows to migrate into cold KQ4/VQ8 storage before attention. RTX 4090 run `29102948104` passes the `0.02` logit tolerance, covering the correctness-first mixed-cache rebuild path rather than only checking finite output.
- 2026-07-11: Added the first page-table-backed CUDA KV path for F32 and agent-adaptive hot rows. Each cache owns a device `u32` page table, fixed `page_tokens`, and page count; append and the initial attention bridge resolve logical rows through that table. The remapped-page RTX 4090 regression reverses two physical pages before append and passes scalar attention parity in run `29150446495`, proving decode does not assume logical-contiguous storage. Q8 and KQ4/VQ8 remain resident contiguous layouts for now; their paged layouts are still required before Phase 3 is complete.
- 2026-07-11: The first F32 page-table bridge gathered mapped KV rows on GPU into the old attention kernel, eliminating host KV copies but adding a gather plus synchronization per layer. Real VibeThinker 3B F32 smoke `29150647070` stayed error-free and faster than CPU (CUDA `1.216s`/`1.048s`, CPU `24.760s`/`23.173s`; `2,359,440` live KV bytes), but regressed from the prior `0.899s` CUDA mean. This bridge is correctness-only; direct paged attention must replace it before treating the cache as performance-ready.
- 2026-07-11: Replaced the F32 gather bridge with direct page-table lookup in the single-query attention kernel. The score and value passes translate every logical token through the device page table, so decode no longer allocates gathered KV buffers or forces a synchronization per layer. RTX 4090 parity run `29151941668` passes the reversed-page scalar regression and the complete serial CUDA gate. Real VibeThinker run `29152553451` remains error-free at CUDA `1.069s`/`0.889s` versus CPU `22.596s`/`23.192s`; the warm CUDA sample returns to the prior `0.899s` contiguous baseline while retaining paged addressing. Q8 and KQ4/VQ8 still need equivalent page-table kernels.
- 2026-07-11: Extended device page tables to Q8 and KQ4/VQ8 caches, including agent-adaptive cold rows. Quantized append, dequantize, and fused attention kernels translate logical positions to physical pages directly; cache growth copies physical storage and preserves existing page-table entries, and VRAM budget/status accounting includes table bytes. RTX 4090 gate `29153515034` passes reversed two-page growth/dequantize/attention parity for both formats. Four-token Q8 run `29153672903` is error-free at CUDA `1.416s`/`1.236s` versus CPU `28.410s`/`28.992s`, with identical `The user says` preview and `599,184` live KV bytes. KQ4/VQ8 run `29153924775` is error-free at CUDA `1.219s`/`1.059s` versus CPU `27.799s`/`27.229s`, with `465,552` live KV bytes; its sampled `.K.G` preview retains the previously documented sampling divergence despite passing controlled greedy/logit parity.
- 2026-07-11: Agent-adaptive real-model run `29154116231` validates the page-backed hot-F32/cold-KQ4-VQ8 combination over four generated tokens. CPU and CUDA both preview `The user says`; CUDA completes in `1.695s`/`1.430s` versus CPU `26.833s`/`26.944s`, reports `2,824,992` live KV bytes plus `723,456` scratch bytes, and returns no backend errors. Its mixed attention still rebuilds a temporary F32 view row-by-row, so a fused device route/gather path remains a performance follow-up rather than a Phase 3 correctness blocker.
- 2026-07-11: Gemma4 CUDA groundwork now covers per-layer KV widths, unweighted RMSNorm, tanh GeGLU, scalar layer scaling, logit softcap, missing-V fallback, sliding-window attention, and the exact Gemma4 residual/norm order. RTX runs `29154650076` and `29154989889` pass geometry/activation parity; synthetic two-layer Gemma4 F32-KV runtime parity passes in run `29155168267`, including a sequence that crosses the sliding-window boundary.
- 2026-07-11: Real Gemma4 bring-up exposed two bounded pre-decode blockers. Run `29155323381` failed on Jinja mapping `.get`, which is now supported by the chat-template engine. Run `29155526703` then timed out in the requested 12B CPU comparison, so the workflow gained a CUDA-only smoke mode. CUDA-only run `29155920179` reached model/CUDA initialization but retained about `32.3 GB` of host working set while expanding the tied token embedding.
- 2026-07-11: Diagnostic run `29156909849` identified the real tied tensor as Q6_K `[262144, 3840]` (`825,753,600` GGUF bytes). Its two expanded F32 copies consumed `8,053,063,680` resident bytes, pushed preflight to `19,739,820,224` model bytes, and left a `31.7 GB` working set. Oversized Q4_K/Q6_K token embeddings now select packed storage above a `4 GiB` expanded-layout cap, while smaller tables retain the faster expanded path.
- 2026-07-11: Added native packed Q6_K embedding and row-reduction matvec PTX over GGUF blocks plus F32 block scales, allowing a tied token embedding/output matrix to stay packed. Full RTX gate `29157393504` passes the packed embedding/matvec CPU-reference test and every existing synthetic CUDA runtime case.
- 2026-07-11: Real Gemma4 12B Q4_K_M CUDA-only run `29157578588` passes end to end. The tied Q6_K embedding/output uses `841,482,240` resident bytes; total reported resident model allocation falls to `12,528,238,784` bytes with a `2,873,678,433`-byte KV budget. Load is `33.892s`; one-token generation is `3.809s` (`0.2625 tok/s`) for 27 prompt tokens, preview `Hello`, `22,020,288` live F32 KV bytes, and `error: null`. This proves the first real Gemma4 native CUDA decode path; multi-token throughput remains a separate gate.
- 2026-07-11: Strengthened real Gemma4 parity run `29158155746` passes zero-layer, one-layer, full-model, and four sequential F32 positions with identical CPU/CUDA top tokens and at most `0.4561` winning-logit score delta. Full-model position 0 has `0.1717` maximum vector delta. Position 3 retains one non-winning outlier (`CPU -4.8342`, `CUDA 4.2100`, delta `9.0442`) while both select token `107`; therefore semantic top-token parity is proven, but strict full-vector sequential tolerance remains open. Multi-block packed Q6_K embedding/matvec parity passes in run `29158412130`, ruling out basic cross-block indexing in the tied output kernel.
- 2026-07-11: Layer localization runs `29159100033` and `29159541580` show exact scaled embedding input at position 3 and localize the first material CPU/CUDA divergence to layer-0 Q/K projections. Run `29159867404` proves this is expected reference-path drift, not a CUDA defect: CUDA Q, K, and V projections match a float-domain CPU row-dot reference within `0.000458`, `0.000336`, and `0.000458` respectively. The optimized CPU Q/K path quantizes activations to Q8_0, while CUDA consumes resident F32 activations. The diagnostic gate now asserts `0.001` projection parity against the float-domain reference, and final assertion run `29160175536` passes the full serial CUDA and real Gemma4 parity gate. Real-model semantic correctness remains gated by greedy top-token agreement and bounded winning-logit delta rather than strict equality with the lower-precision CPU SIMD path.
- 2026-07-11: Gemma4 variable-width Q8 and KQ4/VQ8 caches now append into per-layer paged device storage and execute direct windowed attention without rebuilding an F32 cache. The quantized attention kernels accept an explicit sliding-window start and Gemma attention scale, while standard dense callers preserve the full-prefix `1/sqrt(head_dim)` behavior. RTX 4090 run `29160670734` passes remapped-page low-level tests plus a five-token synthetic Gemma4 sequence crossing the sliding-window boundary in both quantized modes.
- 2026-07-11: Real Gemma4 quantized-KV semantic gate `29161019732` passes four sequential positions for F32, Q8, and KQ4/VQ8 with identical CPU/CUDA greedy top tokens. F32 and Q8 enforce a `1.0` winning-score bound; KQ4/VQ8 enforces `2.0` because 4-bit key-cache error compounds the already-proven optimized-CPU Q8 activation drift while retaining the same winning token.
- 2026-07-11: Bounded Gemma4 12B CUDA-only Q8 smoke `29161170100` generates four stable tokens (`Hello! How can`) twice at `0.917`/`0.936 tok/s`, with `5,517,504` live KV bytes and no backend errors. KQ4/VQ8 smoke `29161425527` generates the same four-token continuation twice at `0.855`/`0.882 tok/s`, uses `4,307,136` live KV bytes, and has no backend errors. These runs close the real-model Gemma4 Q8 and KQ4/VQ8 windowed-cache validation gap.
- 2026-07-11: Added a persistent device route table for agent-adaptive KV. Each logical token encodes hot/cold storage plus its local row; route growth, truncate, policy migration, and VRAM accounting stay session-owned. Route-writer and existing hardware parity pass in run `29161899016`.
- 2026-07-11: Added direct mixed hot-F32/cold-KQ4-VQ8 attention with independent page-table lookup, GQA grouping, Gemma sliding-window start, and explicit scale. Initial real Gemma4 run `29162696124` exposed a position-3 winner mismatch caused by rebuilding and requantizing every cold row during each policy migration. Cold-to-cold migration now copies compressed key/value bytes and scales directly on GPU, while only real hot/cold transitions quantize or dequantize.
- 2026-07-11: Full RTX gate `29163202668` passes remapped-page mixed attention, a 128-wide two-head GQA scalar-reference case, five-token synthetic Gemma4 adaptive migration, and real Gemma4 F32/Q8/KQ4/adaptive four-position parity. Adaptive position 3 restores exact CPU/CUDA top token `107`; the winning-score delta is `3.6596`, covered by an adaptive-specific `4.0` bound while exact greedy-token agreement remains mandatory.
- 2026-07-11: Gemma4 12B adaptive CUDA-only smoke `29163413111` generates `Hello! How can` in both four-token repetitions at `0.784`/`0.763 tok/s`, reports `26,333,568` live KV bytes, and returns no backend errors. The current cache reserves full hot and cold capacities, so adaptive correctness is complete but memory efficiency still requires a shared dynamic page allocator.
- 2026-07-11: Replaced the per-output, two-pass F32 decode-attention algorithm with a block-per-query-head kernel. Each token's QK dot is reduced once, online softmax updates running max/normalizer state, and each lane accumulates V directly through the existing page table. RTX gate `29164049411` passes 128-wide remapped-page scalar parity and the full synthetic runtime suite. VibeThinker four-token run `29164206051` improves mean CUDA latency from the pre-online `1.288s` baseline to `1.124s` (12.7%) while preserving `The user says` output.
- 2026-07-11: Added equivalent online kernels for Q8, KQ4/VQ8, and mixed hot-F32/cold-KQ4-VQ8 adaptive pages. The KQ4 path preserves the 64-element key-scale contract, and the mixed path resolves route plus independent hot/cold page tables inside the fused kernel. The launch uses 256 threads through 256-wide heads and 512 threads for Gemma4's actual `head_dim=512`; wider unsupported geometries retain the correctness-first legacy fallback. Low-level and synthetic gates pass in runs `29164748327`, `29165081015`, and `29165766640`.
- 2026-07-11: Final real Gemma4 gate `29165766640` exercises the online path at 512-wide heads and passes four sequential positions in F32, Q8, KQ4/VQ8, and agent-adaptive modes with exact CPU/CUDA greedy top tokens. Position 3 selects token `107` in every mode within the existing winning-score bounds.
- 2026-07-11: Final four-token Gemma4 smokes quantify Phase 4 throughput. Q8 run `29166255091` reaches `0.998`/`1.046 tok/s`, a 10.3% mean gain over `0.917`/`0.936`; KQ4/VQ8 run `29166450087` reaches `0.979`/`1.020 tok/s`, a 15.1% gain over `0.855`/`0.882`; adaptive run `29166006624` reaches `1.061`/`1.089 tok/s`, a 38.9% gain over `0.784`/`0.763`. Q8 and adaptive produce `Hello! How can`; KQ4's seeded sampled preview is `<channel|>Hello! How`, but controlled greedy parity passes and no backend reports an error.
- 2026-07-11: Added CUDA 12 graph capture/instantiate/replay ownership, `XRT_CUDA_GRAPH=0|1|auto`, runtime capture status, stable batch-1 decode buffers, and device-resident mutable token/position/cache parameters. Low-level replay and two-position RoPE/paged-KV/attention tests pass in run `29166903614`; standard-dense synthetic runtime capture/replay passes in runs `29167709398` and `29167872651`.
- 2026-07-11: The first real VibeThinker graph run `29168209289` correctly fell back because full-context F32 KV preallocation required `9.66 GB` against a `4.98 GB` KV budget. Graph ownership now keys the executable by the active KV capacity, invalidates before pointer-changing growth, reallocates only the 16-byte dynamic parameter buffer, and recaptures after growth. The forced one-token-page growth regression and full CUDA parity gate pass in run `29168647294`.
- 2026-07-11: Replaced the per-token synchronous graph-parameter upload with an ordered asynchronous upload and preallocated only the bounded request horizon (`prompt_len + max_tokens`). Ten-sample VibeThinker 3B Q4_K_M runs use one 759-node capture per session, preserve the same 64-token preview, report `captured`, and return no errors. Eager run `29169946375` averages `3.949s` for the 63 post-first-token decode calls (`15.95 tok/s`); graph run `29170169983` averages `3.740s` (`16.84 tok/s`), reducing steady-state batch-1 decode latency 5.3% and increasing throughput 5.6%. Mean end-to-end 64-token latency improves from `4.838s` to `4.586s` (5.2%).
- 2026-07-11: Added bounded async request admission, explicit queue-full rejection, bounded SSE channels, and cancellation-aware generation so a disconnected client stops before the next model invocation and releases its scheduler/GPU resources. Runtime/server compile, admission saturation, waiter cancellation, and clean-exit validation pass in run `29170831328`.
- 2026-07-11: Added FIFO cooperative execution turns with bounded decode priority, configurable prefill chunks, multimodal override remapping, hybrid-model exclusive turns, and aggregate request-horizon KV reservations against the runtime CUDA KV budget. Scheduled one-token chunks preserve unscheduled CPU output, and the full default/CUDA gate passes in runs `29171263761` and `29172049635`.
- 2026-07-11: Added synchronized concurrent CLI benchmarking. Matched three-sample VibeThinker F32 graph runs `29171619356` and `29171776234` improve mean aggregate throughput from `12.21 tok/s` at concurrency 1 to `16.19 tok/s` at concurrency 2 (32.5%). Mean per-request latency rises from `2.622s` to `3.936s` (50.1%), an explicit policy tradeoff rather than hidden queue time.
- 2026-07-11: Concurrent OpenAI SSE validation now launches a real CUDA server, verifies two simultaneous `/v1/chat/completions` streams through `[DONE]`, asserts scheduler/KV drain, and forces process-tree cleanup. Run `29172983541` passes with one short request overlapping a multi-chunk long prefill: 4 prefill turns, 14 decode turns, and a decode while the long sequence remained in prefill. Prefill/batch APIs no longer capture decode graphs; graph capture is restricted to `forward_token`.

## Design Principle

Do not start by adding model file formats. Start by fixing residency.

Supporting EXL2, EXL3, GPTQ, or AWQ before xeno-rt has an end-to-end GPU execution path would add file parsing complexity without unlocking the 4090. The first performance milestone should run existing GGUF Q4_K/Q6_K/Q8_0 models with weights and KV on GPU.

## Target Architecture

```text
xrt-cli / xrt-server
        |
        v
xrt-runtime
  Session, Scheduler, Sampler, OpenAI API compatibility
        |
        v
Backend trait
  + CpuBackend
  + CudaResidentBackend
  + ExternalOpenAiBackend, optional benchmark/fallback only
        |
        v
CudaResidentBackend
  + GpuModelWeights
  + GpuScratchArena
  + GpuPagedKvCache
  + Quantized GEMV/GEMM kernels
  + Fused decode attention
  + CUDA Graph replay
        |
        v
xrt-gguf / xrt-tokenizer / xrt-hub
```

## Backend API Sketch

Add a backend layer below `xrt-runtime` and above model execution.

```rust
pub enum BackendKind {
    Cpu,
    CudaResident,
    ExternalOpenAi,
}

pub trait CausalLmBackend: Send + Sync {
    fn model_name(&self) -> &str;
    fn config(&self) -> &LlamaConfig;
    fn clear_state(&self);
    fn prefill(&self, tokens: &[u32], start_position: usize, session: &mut BackendSession) -> Result<()>;
    fn decode_one(&self, token: u32, position: usize, session: &mut BackendSession, logits: &mut [f32]) -> Result<()>;
    fn supports_draft_layers(&self) -> bool;
}
```

Keep `LlamaModel` as the CPU implementation initially. Add `CudaResidentModel` as a separate implementation once GPU ownership is real.

Do not force CUDA into the existing `KvCache` trait. The CPU cache trait is row-copy-oriented and is not a good abstraction for GPU page attention.

## Runtime Selection

Add explicit runtime selection:

- `XRT_BACKEND=cpu`
- `XRT_BACKEND=cuda`
- `XRT_BACKEND=auto`
- `XRT_BACKEND=external-openai`

CLI:

```text
xrt generate --model qwen3-8b-q4 --backend cuda
xrt serve --model gemma-4-12b-coder-q4 --backend auto
```

Behavior:

- `cpu`: force current CPU path.
- `cuda`: require CUDA path; fail clearly if unsupported.
- `auto`: try CUDA for supported dense F32/F16/BF16/Q8_0/Q4_0/Q4_K/Q5_K/Q6_K GGUFs in CUDA builds, otherwise fall back to CPU.
- `external-openai`: proxy to a configured local OpenAI-compatible runtime, useful for TabbyAPI/vLLM/SGLang benchmark comparisons.

Implementation status as of 2026-07-10:

- `BackendKind` exists in `crates/xrt-runtime/src/backend.rs`.
- `CausalLmBackend` and `CpuBackend` wrap the existing `LlamaModel` execution surface.
- `BackendSession` exists and owns CPU or CUDA-resident session cache state.
- `Session` routes prefill, decode, speculative verification, hybrid state rollback, cache policy configuration, cache preparation, and rollback through backend/session boundaries.
- `Runtime::load_with_backend` accepts `auto`, `cpu`, `cuda-resident`, and `external-openai`.
- `auto` falls back to CPU by default and selects `cuda-resident` only for metadata-supported dense F32/F16/BF16/Q8_0/Q4_0/Q4_K/Q5_K/Q6_K GGUFs when CUDA is enabled and initializes successfully.
- `cuda-resident` exists for CUDA-feature builds on the supported dense F32/F16/BF16/Q8_0/Q4_0/Q4_K/Q5_K/Q6_K slice; `external-openai` still returns an explicit unsupported error.
- `xrt generate` and `xrt chat` accept `--backend` and `XRT_BACKEND`.
- `xrt bench` accepts comma-delimited `--backends` and reports the active backend in the benchmark table.
- `xrt bench --json` emits a structured report with model path, inferred quantization, prompt token count, OS/arch, git commit, CUDA feature flag, per-backend load time, generation metrics, GPU resource status, and unsupported-backend errors.
- `xrt-server` accepts `--backend`, `POST /v1/runtime/load` accepts `backend`, and `GET /v1/runtime/status` reports requested and active backend values.
- `GpuResourceManager` exists as a runtime-level scaffold with `XRT_CUDA_DEVICE`, `XRT_GPU_MEMORY_FRACTION`, `XRT_GPU_RESERVED_MB`, and `XRT_GPU_KV_FRACTION` parsing.
- `GET /v1/runtime/status` and `POST /v1/runtime/load` report GPU resource status fields, including resident model bytes once the CUDA backend loads.
- `tests/smoke_e2e.rs` includes a deterministic CPU parity test comparing direct `LlamaModel::forward_token` logits to `Runtime` CPU backend logits on a synthetic GGUF fixture.
- OpenAI-compatible generation endpoints remain unchanged.

## GPU Memory Manager

Introduce a shared `GpuResourceManager` at runtime/server level. This aligns with the existing ONNX plan and avoids separate GPU islands.

Responsibilities:

- track total VRAM and reserved safety margin
- allocate persistent weight residency
- allocate per-session KV pages
- allocate scratch arenas
- reserve temporary prefill buffers
- expose runtime status
- support LRU eviction for non-critical future ONNX or embedding sessions

Environment:

- `XRT_CUDA_DEVICE=0`
- `XRT_GPU_MEMORY_FRACTION=0.90`
- `XRT_GPU_RESERVED_MB=1024`
- `XRT_GPU_KV_FRACTION=0.30`

Runtime status should report:

- backend kind
- CUDA device name
- total VRAM
- reserved VRAM
- model weight bytes
- KV bytes allocated
- scratch bytes allocated
- active sessions
- graph capture status

## Phase 0: Benchmark Harness First

Before implementing CUDA acceleration, add stable benchmark commands.

Benchmarks:

- cold model load time
- warm model load time
- prefill tokens/sec
- decode tokens/sec
- first-token latency
- tokens/sec at batch size 1
- tokens/sec at batch sizes 2, 4, 8
- VRAM/host RAM usage
- KV cache bytes/token

Commands:

```text
xrt bench decode --model qwen3-8b-q4 --prompt-len 128 --tokens 256 --backend cpu
xrt bench decode --model qwen3-8b-q4 --prompt-len 128 --tokens 256 --backend cuda
xrt bench compare --model qwen3-8b-q4 --backends cpu,cuda,external-openai
```

Acceptance:

- Benchmark command emits JSON and human-readable table.
- Benchmarks include model path, quantization, backend, CPU, GPU, driver, CUDA version, and commit hash.
- CI can run tiny synthetic benchmarks without CUDA.
- Local CUDA benchmark can be run manually on the 4090.

## Phase 1: Backend Boundary

Implement the backend trait without changing behavior.

Tasks:

- Add backend enum and loader path in `xrt-runtime`.
- Wrap existing `LlamaModel` as `CpuBackend`.
- Add `BackendSession` abstraction that hides CPU KV cache or future GPU session state.
- Preserve current public APIs.
- Add `XRT_BACKEND=cpu|auto|cuda`.
- Add runtime status reporting selected backend.

Acceptance:

- Existing tests pass.
- CPU output is bit-for-bit equivalent to current output under deterministic sampling.
- `XRT_BACKEND=cuda` either runs a supported CUDA slice or returns a clear unsupported error.
- OpenAI endpoint response schema is unchanged.

Remaining Phase 1 follow-up:

- Add CI policy around the new JSON benchmark output.
- Add external-runtime proxy backend wiring if benchmarking against TabbyAPI/vLLM/SGLang should happen before native CUDA lands.

## Phase 2: GPU-Resident Weights for Single-Token Decode

First real CUDA path: load one supported architecture and quantization into GPU memory and decode one token at a time.

Implementation status as of 2026-06-19:

- `xrt-cuda` exposes persistent device buffer primitives: `CudaBytes` and `CudaF32Buffer`.
- `xrt-cuda::GpuModelWeights` can upload all GGUF tensor byte payloads to the selected CUDA device once.
- `xrt-cuda::GpuF32Tensor` can upload named GGUF F32 tensors into persistent device buffers.
- `xrt-cuda::GpuF32Tensor` can upload 2D F32 tensors transposed, which lets the existing CUDA matmul kernel consume GGUF linear weights as resident RHS matrices.
- `xrt-cuda::CudaQ8_0Matrix` can convert GGUF Q8_0 matrix bytes into a resident split CUDA layout: F32 scales plus raw i8 quant payloads.
- `xrt-cuda` can upload GGUF Q4_0 matrices by expanding packed nibbles once into the same resident scale+i8 layout used by the Q8_0 matvec kernel.
- `xrt-cuda::CudaQ4KMatrix` uploads GGUF Q4_K projection/output matrices as packed resident K-quant payloads split into `d`, `dmin`, scale/min bytes, and packed quant nibbles; token-embedding uploads use expanded transposed + row-major F32 buffers to avoid the packed embedding first-use stall.
- `xrt-cuda::CudaQ5KMatrix` can upload GGUF Q5_K matrices by dequantizing once into the same resident F32 transposed RHS layout; token-embedding uploads add one row-major F32 copy for embedding lookup.
- `xrt-cuda::CudaQ6KMatrix` can upload GGUF Q6_K matrices by dequantizing once into the same resident F32 transposed RHS layout; token-embedding uploads add one row-major F32 copy for embedding lookup.
- Existing CUDA embedding, RMSNorm, and matmul kernels now have resident-buffer entry points for F32 weights/tables: `embed_resident`, `rmsnorm_resident_weight`, and `matmul_resident_rhs`.
- Native Q8_0, expanded-Q4_0, packed-Q4_K, expanded-Q5_K, and expanded-Q6_K CUDA matvec primitives exist: `upload_q8_0_matrix`, `upload_q4_0_matrix`, `upload_q4_k_matrix`, `upload_q5_k_matrix`, `upload_q6_k_matrix`, `matvec_q8_0`, `matvec_q4_0`, `matvec_q4_k`, `matvec_q5_k`, and `matvec_q6_k`.
- `xrt-cuda` exposes device-buffer activation entry points for the probe path: `embed_resident_device`, `rmsnorm_device`, `matmul_resident_rhs_device`, `matvec_q8_0_resident_device`, `silu_device`, `mul_device`, `repeat_kv_for_gqa_device`, `add_device`, and `add_assign_device`.
- The activation APIs also expose destination-buffer variants so callers can launch into stable reusable allocations rather than allocate an output for every operation.
- `xrt-cuda::CudaLayerKvCache` provides an F32 resident per-layer KV cache plus `append_layer_kv` and `single_query_attention_device` for single-query attention over cached K/V.
- `BackendSession` can now create CUDA-resident per-layer KV cache state lazily for `cuda-resident` sessions, bounded by model context length.
- `CudaResidentBackend::forward_token` can execute a minimal standard dense F32/F16/BF16/Q8_0/Q4_0/Q4_K/Q5_K/Q6_K model path across all layers: embedding, per-layer attention/FFN, final RMSNorm, and logits. Q4_K token embedding uses expanded resident F32 buffers instead of the packed embedding kernel to avoid the earlier first-use stall. `forward_batch` and `forward_batch_all_logits` process token batches sequentially through the same path.
- The Q8_0 matvec primitive now loads driver PTX directly; it no longer depends on runtime NVRTC.
- CUDA kernel modules are loaded lazily. Opening a CUDA device no longer eagerly JIT-loads every legacy PTX module, so a broken or unsupported primitive fails at the call site instead of preventing CUDA device initialization.
- `CudaResidentBackend` detects a narrow F32 primitive probe path when `token_embd.weight`, `output_norm.weight`, and `output.weight` or tied `token_embd.weight` are all shape-compatible F32 tensors.
- `CudaResidentBackend` also detects a narrow Q8_0 projection probe when `token_embd.weight` is F32 or Q8_0, `output_norm.weight` is F32, and explicit or tied `output.weight` is shape-compatible Q8_0.
- `CudaResidentBackend` detects a standard dense layer-0 Q8_0 projection probe when `blk.0` Q/K/V/O and FFN gate/up/down tensors are shape-compatible Q8_0 and the required norms are F32.
- The F32 and Q8_0 runtime probes now keep embedding, norm, and projection activations on GPU and download only final logits.
- The layer-0 Q8_0 projection probe applies RoPE to resident Q/K buffers with an explicit token position, appends K/V to a probe-local resident KV cache, and runs single-query CUDA attention before the attention-output projection.
- The layer-0 Q8_0 projection probe now applies the attention residual before FFN norm and the FFN residual after down projection, matching the single-token block dataflow except for KV-history attention.
- `GpuResourceStatus` reports `resident_f32_probe_available`, narrow legacy `resident_q8_0_probe_available` / `resident_q8_0_layer0_probe_available` probe flags, and the broad `resident_dense_quant_decode_available` CUDA decode gate so these paths are observable without changing normal generation behavior.
- `XRT_CUDA_PROFILE=1` logs CUDA decode stage timings for token embedding, QKV, attention, attention output, FFN, per-layer total, final norm, final projection, logits download, final logits, and per-token total. Empty, `0`, `false`, and `off` keep profiling disabled.
- CUDA `forward_batch` skips final-logit projection for intermediate prompt tokens and keeps `forward_batch_all_logits` as the explicit all-logits path.
- CUDA `forward_batch_with_embeddings` can process prompt batches with per-position embedding overrides by uploading the override vector for patched positions, then running the same dense quant layer path.
- `xrt-runtime` has a `CudaResidentBackend` that opens `XRT_CUDA_DEVICE`, uploads only the specialized tensor layouts consumed by decode, and reports `model_weight_bytes` through runtime GPU resource status.
- Live `Session` objects retain reusable CUDA decode scratch and can report their own CUDA KV and scratch allocations through `Session::gpu_resource_status()`; benchmark JSON uses that session snapshot after generation.
- Default non-CUDA builds fail fast for `XRT_BACKEND=cuda` before opening the model file.
- CUDA-feature builds can generate through the standard and Gemma4 dense F32/F16/BF16/Q8_0/Q4_0/Q4_K/Q5_K/Q6_K slices, including matching token embeddings and dense projection/output matrices. VibeThinker 3B and Gemma4 12B Q4_K_M complete bounded RTX 4090 one-token smokes. Broader model shapes still return an explicit unsupported decode error.
- Benchmarks now count generated tokens separately from decoded text chunks, so special/empty decoded pieces no longer report as zero-token generations.
- `xrt-cuda` has default-build tests for the CUDA-disabled resident API stubs and ignored CUDA manual parity tests for resident F32 kernels, Q8_0 matvec, RoPE, GQA repeat, and single-query attention.
- `xrt-workspace-tests` has ignored serialized CUDA integration tests that compare multi-layer Q8_0 runtime logits, tied-output logits, sequential `forward_batch`, and `forward_batch_all_logits` against the CPU backend.

Important limitation:

- Full production decode still needs real-model correctness and text-output validation, throughput work, hybrid/MoE support, and broader architecture variants. Batch decode is sequential, not a parallel/fused prefill kernel.
- Q4_K and Q6_K token embeddings use the expanded two-copy F32 layout only while that layout is at most `4 GiB`; larger tables use packed resident kernels. Q5_K embeddings still require expanded F32 storage.
- Packed Q4_K/Q6_K matvec kernels remain one block per output row and correctness-first. The Gemma4 12B packed Q6_K output projection is functional but contributes to a `3.809s` first one-token generation and still needs profiling/optimization.
- Real Gemma4 sequential top-token parity passes through position 3. A non-winning position-3 logit retains a `9.0442` CPU/CUDA delta, but layerwise tracing proves CUDA projection kernels match a float-domain CPU reference within `0.000458`; the larger optimized-CPU delta comes from CPU Q8_0 activation quantization. Real K-quant correctness therefore uses float-reference kernel parity plus greedy top-token/winning-score checks, not strict full-vector equality against the lower-precision CPU SIMD path.
- The Q8_0/Q4_0/Q4_K matvec PTX now uses one block per row with a shared-memory reduction. It still needs warp-level reduction or a vendor GEMV path if profiling shows row reduction overhead.
- The current F32 RMSNorm and F32 matmul PTX paths are still correctness-first scalar/simple kernels. Earlier atomics-based `m == 1` and packed-Q6 prototypes regressed VibeThinker; the new bounded packed Q6_K path fixes large-model residency, but throughput work should still use profiling followed by warp-level or vendor GEMV rather than unmeasured fanout.
- PTX module-load failures now append CUDA driver JIT logs when the driver provides them.
- Do not add more unvalidated inline PTX for F32 GEMV. The next F32/packed GEMV optimization needs either cuBLAS availability or a build-time PTX/CUBIN validation step so invalid kernels fail before runtime.
- The session scratch arena removes repeated allocations for normalization, Q/K/V projections, attention-output projection, FFN gate/up activations, and final logits. Token embedding, single-query attention output, and FFN down/post-residual output still allocate during each layer/token path and are the next scratch-residency targets.
- Agent-adaptive currently reserves full-capacity hot F32 and cold KQ4/VQ8 stores for every layer. The four-token Gemma4 smoke reports `26,333,568` KV bytes versus `22,020,288` for the earlier F32 one-token baseline, so adaptive mode is a correctness/policy foundation rather than a memory-saving mode until hot/cold stores draw pages from a shared dynamic allocator.

Initial support matrix:

| Architecture | Quantization | Status Target |
|---|---|---|
| Llama/Qwen2/Qwen3 standard attention | Q4_K, Q6_K, Q8_0 | Required |
| Gemma4 dense text | Q4_K, Q6_K, Q8_0 | Required after standard attention |
| Qwen3.5 DeltaNet hybrid | Q4_K, Q6_K, Q8_0 | Later |
| MoE | Q4_K, Q6_K, Q8_0 | Later |

Tasks:

- Done: add `GpuModelWeights`.
- Done: upload resolved GGUF tensors once at model load.
- Done: preserve GGUF as source format.
- Done: keep CPU tokenizer and sampler initially.
- In progress: session-owned GPU scratch now covers normalization, Q/K/V, attention-output projection, FFN gate/up, and logits; embedding, attention output, and layer-output ping-pong still need stable destination buffers.
- In progress: CUDA quantized GEMV for Q8_0, Q4_0, Q4_K, Q5_K, and Q6_K.
- In progress: copy only final logits back to host per token. Q4_K/Q6_K token embeddings select bounded expanded or packed resident layouts; Q5_K remains expanded, and batch prefill remains sequential.
- Done for the first target: deterministic VibeThinker 3B Q4_K_M CPU/CUDA top-token parity and repeated one-token RTX 4090 throughput evidence.
- Pending: multi-token text-output validation and correctness/throughput breadth across supported GGUF architectures and quantizations.

Acceptance:

- CUDA decode works for one small synthetic model.
- CUDA decode works for one real GGUF Q4_K model.
- Logits match CPU within quantization tolerance on deterministic prompts.
- No host-device copies inside the layer loop except explicit debug mode.
- CPU fallback still works.

## Phase 3: GPU-Resident Paged KV Cache

Move KV cache to GPU and stop copying key/value rows to CPU.

Design:

- Add `GpuPagedKvCache`.
- Pages are fixed token blocks.
- Each layer has page tables.
- Cache allocation is owned by `BackendSession`, not global model weights.
- Use a separate layout for standard fixed-width models and Gemma4 variable-width layers.

Initial cache modes:

- `gpu-f16`
- `gpu-q8`
- `gpu-kq4-vq8`

Do not implement TurboQuant in this phase.

Acceptance:

- Decode attention reads KV entirely on GPU.
- Cache append happens on GPU.
- Session truncate works for speculative rollback.
- Runtime status reports GPU KV bytes.
- CPU KV tests remain unchanged.

Implementation status as of 2026-07-11:

- Complete and RTX 4090 validated for standard fixed-width dense layers in F32, Q8, KQ4/VQ8, and agent-adaptive modes.
- Every fixed-width cache owns a device page table; append, dequantize, and direct attention resolve logical positions on GPU. Growth preserves physical storage and page-table entries, and truncate retains logical-prefix semantics.
- Runtime allocation and status/budget accounting include page-table bytes.
- Gemma4 variable-width layers use per-layer page-backed F32, Q8, and KQ4/VQ8 caches with direct windowed attention. Remapped-page low-level parity, five-token synthetic parity across the sliding-window boundary, real four-position semantic parity, and bounded four-token Gemma4 12B smokes are RTX 4090 validated in runs `29160670734`, `29161019732`, `29161170100`, and `29161425527`.
- The current unquantized CUDA cache mode stores F32 rather than the initially proposed F16. A dedicated F16 cache representation remains optional future memory work, not a correctness dependency for the validated F32/Q8/KQ4 paths.
- Gemma4 agent-adaptive uses a persistent logical route table plus direct page-aware mixed F32/KQ4-VQ8 attention. Policy migration preserves already-cold compressed rows byte-for-byte, and low-level, synthetic, real semantic, and four-token smoke gates pass in runs `29161899016`, `29162917437`, `29163202668`, and `29163413111`.

## Phase 4: Fused Decode Attention

Implement decode attention kernels specialized for batch size 1 and small batch sizes.

Tasks:

- Add fused QK dot, online softmax, and V accumulation kernel.
- Support GQA/MQA head grouping.
- Support causal decode.
- Support sliding-window mask for Gemma4.
- Support per-layer head dimensions for Gemma4.
- Keep a CPU parity test for tiny models.

Acceptance:

- No per-position KV row copy.
- Attention output matches CPU within tolerance.
- Decode throughput improves over Phase 2.
- Gemma4 Q4 smoke still generates successfully.

Implementation status as of 2026-07-11:

- Complete and RTX 4090 validated for the current standard-dense and Gemma4 target geometries through 512-wide heads.
- F32, Q8, KQ4/VQ8, and agent-adaptive caches use direct page-aware block-per-head kernels that fuse QK reduction, numerically stable online softmax, and V accumulation. GQA/MQA grouping, causal prefix length, Gemma4 sliding-window start, explicit Gemma scale, and variable per-layer widths remain kernel inputs.
- No attention path copies per-position KV rows or reconstructs a temporary F32 cache. Head dimensions above 512 keep the old correctness-first kernel until a wider production model establishes a required geometry.
- Low-level 128-wide and 512-wide scalar-reference tests, synthetic runtime parity, real four-mode Gemma4 semantic parity, VibeThinker before/after comparison, and bounded Gemma4 smokes pass in runs `29164049411`, `29164206051`, `29164748327`, `29165081015`, `29165766640`, `29166006624`, `29166255091`, and `29166450087`.
- The measured VibeThinker F32 mean latency improves 12.7% over the pre-online Phase 2/3 four-token baseline. Gemma4 Q8, KQ4/VQ8, and adaptive throughput improve 10.3%, 15.1%, and 38.9% respectively over their directly comparable pre-online runs.

## Phase 5: CUDA Graph Decode Replay

Reduce per-token kernel launch overhead.

Tasks:

- Pre-allocate static decode buffers.
- Capture decode graph for fixed batch sizes.
- Start with batch size 1.
- Add graph cache by architecture, quantization, and shape.
- Fall back to eager CUDA if capture fails.

Acceptance:

- `XRT_CUDA_GRAPH=0|1|auto`.
- Graph capture status appears in runtime status.
- Decode throughput improves at batch size 1.
- Failure to capture does not break generation.

Implementation status as of 2026-07-11:

- Complete and RTX 4090 validated for the first target: standard dense batch-1 decode with F32 paged KV and resident F32/F16/BF16/Q8_0/Q4_0/Q4_K/Q5_K/Q6_K weight layouts.
- `CudaGraphExec` owns captured CUDA graph/executable handles, launches on the session device stream, and destroys both handles after rebinding the retained CUDA context.
- `CudaDecodeGraphState` is session-owned and keys the active executable by architecture, resident weight kinds, KV mode/capacity, layer count, and decode geometry. Pointer-changing KV growth resets the executable before allocation replacement; the next eligible token captures a graph for the new capacity.
- `Session::generate_stream` preallocates only the budget-checked request horizon for supported graph sessions. It never reserves the model's full context solely for graph stability, and allocation/capture/launch failures retain explicit `eager-fallback` behavior.
- Token ID, position, cache length, and attention start are updated through a 16-byte resident parameter buffer. The graph path enqueues that upload without an extra synchronization; the final logits download remains the per-token synchronization boundary.
- Runtime/session GPU status separates requested `cuda_graph_mode` from observed `graph_capture` state (`disabled`, `not-captured`, `captured`, or `eager-fallback`). The manual RTX workflow can force `0`, `1`, or `auto` for matched benchmarks.
- Low-level replay, mutable-parameter, synthetic full-runtime replay, and forced KV-growth recapture tests pass. Ten-sample real VibeThinker runs `29169946375` and `29170169983` show one 759-node capture per graph-enabled session, a 5.3% steady-state decode-latency reduction, and a 5.6% throughput gain while preserving output and bounded allocations.
- Gemma4 variable-width layers, Q8/KQ4/adaptive KV graph capture, and batch sizes above one remain eager CUDA extensions; they do not block the validated Phase 5 batch-1 standard-dense target.

## Phase 6: Chunked Prefill and Continuous Batching

Once single-user decode is fast, add serving throughput features.

Tasks:

- Add request scheduler in `xrt-runtime`.
- Separate prefill and decode queues.
- Add chunked prefill.
- Add continuous batching for decode.
- Add backpressure and max active sequences.
- Preserve streaming semantics.

Acceptance:

- Multiple concurrent OpenAI chat requests stream correctly.
- A long prefill does not block all decode tokens.
- Throughput improves with concurrent clients.
- Latency regression is bounded by configured scheduler policy.

Implementation status as of 2026-07-11:

- Request admission is bounded by `max_active_sequences` and `max_queued_sequences`; queue saturation returns HTTP 429. Defaults remain conservative at one active sequence until broader aggregate-memory data is available.
- Streaming uses bounded channels. Client disconnects cancel generation before the next model call, preventing abandoned streams from retaining an active slot or unbounded buffered chunks.
- Dense sessions split prompt work by `prefill_chunk_tokens` and acquire FIFO prefill/decode turns. Decode receives bounded priority through `max_decode_turns_before_prefill`, while same-phase FIFO tickets prevent a long prompt from immediately reacquiring every prefill turn.
- Hybrid/recurrent architectures hold an exclusive execution turn because their recurrent state is backend-global; they remain correct but do not participate in dense-session interleaving yet.
- Scheduled CUDA sessions reserve their worst-case request-horizon KV growth peak against one aggregate scheduler budget. Runtime status reports budget, live reservations, active/queued sequences, prefill/decode waiters and turns, overlap counters, admissions, and rejections.
- The CLI supports synchronized `--concurrency`, aggregate throughput, mean/max request latency, and scheduler JSON. The matched RTX runs above show a 32.5% two-sequence aggregate-throughput gain with a 50.1% mean latency increase.
- Real OpenAI-compatible concurrent streaming and long-prefill/decode overlap pass in run `29172983541`; all resources drain to zero after both streams finish.

Remaining Phase 6 work:

- Cooperative scheduling currently executes one backend turn at a time. Implement a true multi-sequence CUDA decode batch so compatible ready sequences share kernel launches and graph executables instead of only filling host/launch gaps.
- Add aggregate decode-batch metrics and matched concurrency 1/2/4 benchmarks after the fused batch path exists.

## Phase 7: Prefix Cache / Radix Cache

Implement prefix reuse for repeated system prompts, tools, memory blocks, and document-store workloads.

Tasks:

- Hash token prefixes by model, tokenizer, prompt segment, and backend.
- Store immutable prefix KV pages.
- Allow sessions to attach prefix pages copy-on-write.
- Add eviction policy.
- Integrate with agent span policy from `AGENT_ADAPTIVE_KV_ROADMAP.md`.

Acceptance:

- Repeated system prompt prefill is skipped.
- Runtime status reports prefix-cache hit rate.
- Cache invalidation is deterministic across model/tokenizer changes.

## Phase 8: Advanced Quantization Formats

Only after GPU-resident GGUF is working, evaluate extra formats.

Candidate formats:

- GPTQ
- AWQ
- EXL3
- compressed-tensors

Recommendation:

- Do not implement EXL2. ExLlamaV2 is archived.
- Consider EXL3 only if local model availability and speed justify the extra parser/kernel complexity.
- Add GPTQ/AWQ first if the goal is interoperability with vLLM/Hugging Face quantized checkpoints.
- Keep GGUF path as the default.

Acceptance:

- New formats are optional.
- GGUF path remains unaffected.
- Format support has separate tests and clear model compatibility errors.

## External Runtime Adapter

Add an optional OpenAI-compatible proxy backend for comparison and emergency fallback.

Supported external runtimes:

- TabbyAPI with ExLlamaV3
- vLLM OpenAI server
- SGLang OpenAI server

Use cases:

- benchmark xeno-rt against external runtimes on the same prompt/model family
- run a format xeno-rt cannot yet load
- provide a temporary path for high-speed local serving while native CUDA matures

Non-goal:

- Do not make xeno-rt a thin proxy by default.

Environment:

- `XRT_EXTERNAL_BASE_URL=http://127.0.0.1:8000/v1`
- `XRT_EXTERNAL_API_KEY=...`
- `XRT_BACKEND=external-openai`

Acceptance:

- `/v1/chat/completions` can proxy to a local external runtime.
- Streaming proxy preserves OpenAI-compatible SSE framing.
- Runtime status makes proxy mode obvious.

## TurboQuant Relationship

TurboQuant and agent-adaptive KV compression are downstream of GPU residency.

Required prerequisites:

- GPU-resident KV pages.
- Fused attention that can read compressed pages.
- Per-layer width support.
- Prefix/policy metadata for hot/cold spans.

Do not implement TurboQuant before Phase 3 and Phase 4 are stable.

## File/Crate Plan

Likely new files:

- `crates/xrt-runtime/src/backend.rs`
- `crates/xrt-runtime/src/gpu_resource.rs`
- `crates/xrt-models/src/backend_cpu.rs`
- `crates/xrt-models/src/backend_cuda.rs`
- `crates/xrt-cuda/src/model.rs`
- `crates/xrt-cuda/src/kv_cache.rs`
- `crates/xrt-cuda/src/kernels/`
- `crates/xrt-server/src/external_openai.rs`
- `benches/gpu_decode_bench.rs`

Likely changed files:

- `crates/xrt-runtime/src/session.rs`
- `crates/xrt-runtime/src/lib.rs`
- `crates/xrt-models/src/llama.rs`
- `crates/xrt-cuda/src/lib.rs`
- `crates/xrt-cli/src/main.rs`
- `crates/xrt-server/src/main.rs`
- `README.md`

## Testing Plan

Required tests:

- CPU backend deterministic parity with current behavior.
- Backend selection environment parsing.
- CUDA unsupported errors when feature is disabled.
- Synthetic CUDA model load behind CUDA feature.
- Real GGUF CUDA smoke tests as ignored/manual tests.
- GPU KV append/read/truncate tests.
- CUDA decode parity against CPU on tiny F32/F16 fixtures.
- Quantized GGUF parity with loose tolerances.
- OpenAI API response schema unchanged.
- External proxy streaming SSE framing.

Local verification safety:

- Prefer `.\scripts\safe-cuda-check.ps1` for routine CUDA-backend compile/profiling-toggle validation.
- Use `.\scripts\safe-cuda-smoke.ps1 -ConfirmGpuRun` only when a real CUDA model run is specifically needed. Keep `-MaxTokens 1`, profiling off, and bounded `-RunTimeoutSeconds`. Add `-CacheMode q8`, `-CacheMode kq4_vq8`, or `-CacheMode agent_adaptive` only when validating quantized GPU-KV paths.
- Run Cargo/GPU verification serially on Windows. Do not launch parallel `cargo` checks/tests against the same `target` directory.
- Bound real-model CUDA smokes with a timeout and check for leftover `cargo`, `rustc`, and `xrt-cli` processes after interrupted runs.
- Do not keep retrying a hung profiled smoke in the same turn. Stop the process, inspect the last profile line, then make one targeted change.

Manual 4090 tests:

```text
cargo run --release -p xrt-cli --features cuda -- bench decode --model qwen3-8b-q4 --backend cuda
cargo run --release -p xrt-cli --features cuda -- bench decode --model gemma-4-12b-coder-q4 --backend cuda
cargo run --release -p xrt-cli --features cuda -- serve --model qwen3-8b-q4 --backend cuda
```

## Benchmark Gates

Every phase must record before/after values for:

- prompt length
- generated token count
- model name
- quantization
- backend
- GPU name
- driver/CUDA version
- peak VRAM
- host RAM
- prefill tokens/sec
- decode tokens/sec
- first-token latency
- total latency

Initial benchmark target:

- Phase 1 must have zero throughput regression on CPU.
- Phase 2 CUDA can be slower than mature external runtimes but must beat CPU decode on 4090.
- Phase 4 must show a clear decode improvement over Phase 2.
- Phase 5 must show a clear batch-size-1 decode improvement over eager CUDA.

Do not claim ExLlama/vLLM/SGLang parity until measured on the same GPU and comparable model.

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| CUDA path becomes a second model implementation with drift | Incorrect logits | Keep CPU parity tests and shared metadata/tensor resolution |
| GGUF K-quants are hard to optimize on GPU | Poor speed | Start with Q8_0 and one K-quant; add formats incrementally |
| Host-device copies remain hidden in hot path | No speedup | Add instrumentation and debug assertions for transfer counts |
| GPU memory fragmentation | OOM during long sessions | Use page allocators and central `GpuResourceManager` |
| Gemma4 variable-width KV breaks generic kernels | Incorrect attention | Keep Gemma4-specific layer config and specialized kernel dispatch |
| CUDA Graph capture is brittle | Runtime failures | Make graph replay optional with eager fallback |
| External runtimes create API drift | Broken xeno-agent compatibility | Keep proxy mode explicit and schema-tested |

## Non-Goals

- Replacing GGUF as the default model format.
- Removing CPU inference.
- Depending on Python for the native backend.
- Implementing EXL2.
- Implementing TurboQuant before GPU KV pages and fused attention.
- Claiming 130-170 tok/s without local benchmark evidence.

## Definition of Done

The GPU acceleration project is successful when:

- `XRT_BACKEND=cuda` runs a real GGUF Q4/Q6/Q8 model on the RTX 4090.
- Weights, KV, and scratch buffers remain resident on GPU during decode.
- Decode throughput is materially faster than CPU on the same machine.
- OpenAI-compatible API behavior is unchanged.
- CPU fallback works without CUDA.
- Benchmarks are reproducible and checked into reports.
- The design can later host TurboQuant and agent-adaptive KV policies without another rewrite.
