# GPU Runtime Acceleration Spec

Status: Draft implementation spec, Phases 1-7 initial targets, Phase 8 dense/packed Qwen2-Qwen3 formats, and external OpenAI server/benchmark adapter validated
Date: 2026-06-19
Last updated: 2026-07-13
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
- NVIDIA CUDA Driver API graph management: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html

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
- Real VibeThinker 3B Q4_K_M and Gemma4 12B Q4_K_M models load and run the CUDA decode path with multi-position semantic parity and bounded multi-token RTX throughput evidence. Broader architecture/model coverage remains incremental.
- Q4_0 exists as an expanded resident primitive and is wired for token embeddings plus dense projection/output matrices.
- Standard dense F32-KV decode supports bounded continuous batching through one parallel CUDA parent graph composed from independent per-session child graphs. Gemma4, hybrid/recurrent models, and quantized-KV modes retain cooperative eager/graph fallback paths.
- CUDA KV cache modes use device page tables and GPU-side logical-to-physical addressing. Reusable prefixes are retained by a bounded runtime cache as immutable snapshots. A low-level bounded shared F32 page pool now proves physical-page reuse, partial-page copy-on-write, device-pointer append/gather, and direct pointer-table online attention, but runtime sessions still use per-layer contiguous allocations; cross-stream ordering, CUDA Graph lifetime rules, quantized pools, and runtime integration remain future memory-efficiency work.
- Model, scratch, KV, page-table, route, graph-parameter, and temporary CUDA slices allocate through the device's current stream-ordered CUDA memory pool when the device supports memory pools. Scratch ownership remains session-local; there is no category-specific user-space scratch arena or runtime-integrated shared KV page allocator yet.
- Sampled device-wide VRAM usage, process peak host residency, a clone-shared ledger for explicit xeno-owned CUDA allocations, and CUDA pool used/reserved current/peak telemetry are wired. The explicit ledger records category-owned current/peak bytes, allocation calls, and total allocated bytes, including temporary old-plus-new overlap during KV growth. Pool telemetry adds driver-level backing-allocation visibility but does not attribute bytes by xeno category. Batch-1 CUDA Graph replay is wired for standard dense decode with F32 KV; Gemma4, quantized KV, and larger batch graph variants still use eager CUDA.

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
- 2026-07-12: Added owned-session decode rendezvous with bounded `max_decode_batch_size` and `decode_batch_wait_micros`, stable sequence IDs, FIFO execution integration, and aggregate batch metrics. The scheduler transfers each `BackendSession` into the batch processor and returns it with logits, avoiding borrowed state or unsafe cross-thread pointers. The full compile/default gate passes in run `29173513689`.
- 2026-07-12: Added parallel CUDA parent graphs composed from independent per-session batch-1 child graphs. Each child retains isolated KV/scratch buffers, while the parent has no dependency edges between children and needs one graph launch per ready multi-sequence batch. NVIDIA documents that `cuGraphAddChildGraphNode` clones the child graph into the parent. Low-level child composition plus two-session CPU/CUDA runtime parity pass in run `29174858469`.
- 2026-07-12: Kept two rejected designs in the benchmark record rather than shipping them as wins. A 2 ms rendezvous never formed a real-model batch (`29173965411`, about `14.42 tok/s`). A serial 1,518-node shared graph formed size-2 batches but averaged only `13.97 tok/s` (`29174127898`). Concurrent streams reached `16.62 tok/s` (`29174561139`); the final one-launch parallel parent graph is faster and remains the primary path, with streams retained only as a composition/launch fallback.
- 2026-07-12: Matched VibeThinker 3B Q4_K_M, F32-KV, graph-auto, 32-token, three-sample runs close the Phase 6 throughput gate. Concurrency 1 run `29175121050` averages `12.41 tok/s` and `2.582s`; concurrency 2 run `29174975589` averages `17.46 tok/s`, `3.667s` wall time, and `3.621s` mean request latency with 29 fused size-2 replays per sample; concurrency 4 run `29175256908` averages `22.05 tok/s`, `5.809s` wall time, and `5.681s` mean request latency while reaching size 4. Aggregate throughput improves 40.7% at concurrency 2 and 77.6% at concurrency 4 versus concurrency 1. The final concurrency-2 parent graph is 7.8% faster than the prior cooperative-only `16.19 tok/s` baseline while mean request latency improves about 8.0%.
- 2026-07-12: Real OpenAI-compatible validation run `29175405002` passes two concurrent SSE requests with `[DONE]`, 4 chunked-prefill turns, 9 decode batches, 4 fused size-2 parent-graph replays, and a decode while the long request remained in prefill. Active/queued sequences, KV reservations, prefill registrations, and the server process all drain to zero.
- 2026-07-12: Added a bounded deterministic prefix cache keyed by runtime model/tokenizer namespace, backend, KV mode, session policy, reusable prompt tokens, and clipped semantic span metadata. CPU F32/Q8/KQ4-VQ8/adaptive pages use `Arc` plus page-level `Arc::make_mut`; CUDA sessions attach immutable snapshots through `Arc` and materialize one device-to-device mutable copy on first write. Entry/byte LRU limits, exact invalidation, structural system/developer/tool-schema selection, status counters, CLI benchmark JSON, and scheduler accounting for externally retained CUDA KV are wired. Expanded all-mode RTX gate `29176939064` passes.
- 2026-07-12: Matched VibeThinker 3B Q4_K_M, F32-KV, graph-auto, 20-prompt-token runs quantify prefix reuse. Cache-disabled run `29177251045` has warm prefill `803.159`/`797.761 ms`, total `915.433`/`908.445 ms`, and `4.370`/`4.403 tok/s`. Cache-enabled run `29177391329` records one miss/insert followed by two hits; each hit skips 19 prompt tokens, with warm prefill `125.639`/`123.379 ms`, total `241.119`/`241.228 ms`, and `16.589`/`16.582 tok/s`. Warm means improve by 84.4% for prefill, 73.6% for total latency, and 3.78x for throughput while preserving the same seeded `The user says` preview and retaining `2,359,440` prefix bytes.
- 2026-07-12: OpenAI-compatible server run `29177699687` passes two concurrent SSE requests plus two sequential repeated-prefix probes. It records 8 prefill turns, 23 decode turns, 4 fused size-2 parent-graph replays, and a decode while prefill is active. The gate requires a prefix hit and saved tokens, exact equality between scheduler external KV reservation and prefix resident bytes, zero active/queued/session KV leakage, `[DONE]` framing, and clean process shutdown.
- 2026-07-12: Added `xrt-safetensors`, a read-only mmap loader for single-file or indexed sharded SafeTensors bundles with bounded JSON parsing, path containment, exact shard/index validation, typed Qwen2 config metadata, and normalized AWQ/GPTQ/compressed-tensors declarations. Added Hugging Face BPE/tokenizer loading with exact HF-versus-GGUF token-ID parity on the real VibeThinker model. Metadata gate `29178475508` passes against the two-shard 6,171,877,376-byte BF16 bundle.
- 2026-07-12: Decoupled `CudaResidentBackend` model identity/config from its optional GGUF CPU reference and introduced a normalized `ResidentTensorSource`. GGUF now passes every resident support check, upload, optional tensor lookup, and VRAM estimate through the adapter. Full default/CUDA compile and synthetic RTX parity gates `29178707505`, `29179075954`, and `29179431144` pass without changing GGUF behavior.
- 2026-07-12: Added the first native SafeTensors execution target: dense Qwen2 F32/F16/BF16 directories map Hugging Face tensor names into the canonical resident CUDA layout and load through `Runtime` with the HF tokenizer. Real RTX run `29179748626` uploads the VibeThinker 3B BF16 bundle into 13,588,414,464 resident bytes in 43.523 seconds. Zero-layer, one-layer, and full-model comparisons against the equivalent Q4_K_M GGUF CPU reference select identical top tokens; winning-logit deltas are 0.0015, 0.0891, and 0.0602, with comparison stages taking 0.045, 0.047, and 0.324 seconds. GPTQ, compressed-tensors, non-Qwen2 architectures, rope-scaling variants, and SafeTensors CPU decode still fail explicitly.
- 2026-07-12: Strengthened the dense Qwen2 SafeTensors gate in RTX run `29179999285`: every zero-layer, one-layer, and full-model winning-score delta must remain at most `1.0`, winning token IDs must match exactly, and a normal deterministic one-token `Runtime` session must generate identical text. The real BF16 CUDA and equivalent GGUF CPU runtimes both generated `"Hello"`; the complete release-mode model test finished in 44.63 seconds.
- 2026-07-12: Added native AutoAWQ GEMM 4-bit resident matrices with strict method/version/bit/group/zero-point validation, packed qweight/qzeros residency, F16/BF16/F32 scale decoding, and host/device/device-into CUDA matvec entry points. The CUDA C source is compiled in pinned `nvidia/cuda:12.8.1-devel-ubuntu22.04`; the checked-in PTX must compare byte-for-byte in every CUDA workflow so the Windows RTX runner needs no CUDA Toolkit. Packed scalar parity passes in run `29180437742`, reproducibility in `29180665552`, and complete synthetic Qwen2 AutoAWQ decode plus malformed-geometry/GPTQ rejection in `29181216992`.
- 2026-07-12: Added atomic, size-bounded, SHA-256-pinned real fixture acquisition for `Qwen/Qwen2.5-0.5B-Instruct-AWQ` revision `cb07c13df107486a6d99bd487a819dd8905510e9` and its official GGUF reference. The repository's Q4_K_M file contains unsupported legacy Q5_0 tensors, so the gate deliberately uses official Q8_0 revision `df5bf01389a39c743ab467d734bf501681e041c5`, which is already supported by the CPU path. Existing files are rehashed, downloads use dedicated `.partial` paths, and verified files move atomically into the persistent runner cache.
- 2026-07-12: Real AutoAWQ RTX gate `29182210301` validates 627 physical tensors and 291 canonical tensors, loads 1,280,856,576 resident bytes in 3.197 seconds, and executes the normal 24-layer Qwen2 CUDA path. Against the independently quantized official Q8_0 CPU reference, zero-layer, one-layer, and full-model greedy winners match exactly at token IDs `785`, `3070`, and `2701`; top-5 overlaps are `5`, `4`, and `2`, and winning-score deltas are `0.0011`, `0.0273`, and `3.9069`. The semantic gate retains exact winner and deterministic generated-text parity while allowing the documented cross-format score drift; both runtimes generate `" Paris"`. The complete serial CUDA gate, process cleanup, and quiet soak pass.
- 2026-07-12: Added native GPTQ v1 GEMM 4-bit resident matrices for standard non-act-order checkpoints. The adapter requires bits `4`, symmetric quantization, `desc_act=false`, `exllama_config.version=1`, standard monotonic `g_idx`, and group sizes `-1`, `32`, `64`, or `128`; GPTQ v2, act-order maps, ambiguous packing, and unsupported metadata fail before upload. The checked-in CUDA 12.8.1 PTX and scalar packed parity pass in run `29182763779`; complete synthetic Qwen2 mapping, malformed-metadata rejection, kernel parity, and full CUDA decode pass in run `29183216679`.
- 2026-07-12: Added atomic, size-bounded, SHA-256-pinned acquisition for `Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4` revision `c34a4a91629f09f73a285f32dbd26106b033c654`, reusing the official Q8_0 GGUF reference revision `df5bf01389a39c743ab467d734bf501681e041c5`. The first real gate `29183570661` validated all 794 physical tensors and 290 canonical tensors but showed that a 24-layer draft from only the prompt's first BPE token is not a stable cross-quantization oracle: GPTQ and Q8 select different full-draft winners despite matching zero/one-layer winners.
- 2026-07-12: Final real GPTQ RTX gate `29183838567` validates all 168 packed linear mappings, compares real layer-0 attention-Q and layer-23 FFN-down CUDA outputs against an upstream-equivalent host dequantization, loads 1,280,856,576 resident bytes in 2.788 seconds, and retains exact zero/one-layer winner parity at token IDs `785` and `3070`. The 24-layer single-fragment comparison remains diagnostic and documents independent-quantization drift. The semantic gate uses the complete prompt and requires both the official Q8_0 CPU runtime and native GPTQ CUDA runtime to generate the exact known greedy token `" Paris"`; the full serial safe gate and cleanup pass.
- 2026-07-12: Added native compressed-tensors `pack-quantized` W4A16 resident matrices for symmetric static group quantization with `actorder=group`. The adapter validates the four-component `weight_packed`/`weight_scale`/`weight_shape`/`weight_g_idx` contract, exact physical geometry, every shape payload, and permuted group cardinality before upload. The CUDA kernel consumes row-major signed-offset INT4 words, decoded resident F32 scales, and the explicit group map. Reproducible CUDA 12.8.1 PTX and scalar parity pass in run `29184700425`; strict synthetic mapping/rejection tests plus full Qwen2 CUDA decode pass in run `29185248116`.
- 2026-07-12: Added atomic, size-bounded, SHA-256-pinned acquisition for `RedHatAI/Qwen2.5-0.5B-quantized.w4a16` revision `81f31585caa4e516d62f8e6c132a1ad4076b402d` and the exact official dense `Qwen/Qwen2.5-0.5B` BF16 reference revision `060db6499f32faf8b98477b0a26969ef7d8b9987`. The first real gate `29185680341` verified every downloaded byte and exposed that optional Hugging Face metadata may be explicitly `null`; optional typed fields now normalize absent and null values identically while required fields remain strict.
- 2026-07-12: Final compressed-tensors RTX gate `29185883493` validates 795 physical tensors, 291 canonical tensors, and all 168 W4A16 linear mappings. Real layer-0 attention-Q and layer-23 FFN-down CUDA outputs match host dequantization. The dense BF16 and W4A16 runtimes load 2,520,669,696 and 1,291,623,936 resident bytes in 5.555 and 3.758 seconds. Zero-layer logits are identical; one-layer winners match at token `3070` with a `0.260142` winning-score delta. The 24-layer first-fragment result remains diagnostic because accumulated quantization drift changes its winner. Exact tokenizer IDs and full-prompt greedy generation are mandatory; both runtimes generate the known token `" Paris"`, and the complete serial safe gate and cleanup pass.
- 2026-07-12: Added a separate native GPTQ explicit-group GEMM4 path for v1 act-order and v2 direct-zero checkpoints. The adapter requires explicit `desc_act`, validates every signed `g_idx` value and exact per-group cardinality, accepts only supported 4-bit/group/packing declarations, and keeps standard monotonic GPTQ v1 on its existing kernel. `GptqZeroEncoding` makes v1 minus-one and v2 direct packed-zero semantics explicit at upload and launch. The checked-in CUDA 12.8.1 PTX has SHA-256 `9b10a2e76086020df561fd88cb24fdcfdaaac4110e2ae9aa04de96295d034a66`; synthetic v1 act-order/v2 mapping, malformed-metadata rejection, low-level scalar parity, and full Qwen2 CUDA decode pass in run `29187433305`.
- 2026-07-12: Added atomic, byte-counted, SHA-256-pinned real GPTQ variant fixtures. The act-order gate uses `Mohaaxa/qwen2.5-1.5b-gptq-4bit-v2` revision `46e6f58dadc81c981175388a91d010f4c37fbfba` (despite its repository name, metadata declares GPTQ v1 with `desc_act=true`) and official dense reference `Qwen/Qwen2.5-1.5B-Instruct` revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. The v2 gate deterministically converts the already-pinned Qwen2.5 0.5B v1 fixture by incrementing each packed zero nibble independently and pins the derived SafeTensors SHA-256 `069d132daa79ec1f618606fa02447aa09be6554d34b2347bc9f2f5355b7c2cff`.
- 2026-07-12: Final GPTQ variant RTX gate `29188273180` validates all 1,038 physical/338 canonical act-order tensors and all 196 explicit packed matrices, then compares real layer-0 attention-Q and layer-27 FFN-down CUDA outputs with host dequantization. Official dense BF16 and act-order runtimes load 7,108,352,000 and 2,616,825,856 resident bytes in 17.038 and 5.683 seconds. Zero-layer winners match at token `785` with maximum logit delta `0.0000038`; one-layer winners match at token `90767` with maximum logit delta `0.5694`; the observed full-model first-fragment winner also matches at token `2701` but remains diagnostic. Both runtimes generate `" Paris"`. The same run validates 794 physical/290 canonical derived-v2 tensors and all 168 explicit matrices, selected real kernels against direct-zero host dequantization, exact v1/v2 zero-, one-, and full-model logit vectors, and exact `" Paris"` generation. The complete broad CUDA regression gate and 165-second leaked-process soak pass.
- 2026-07-12: Added native AutoAWQ `GEMV` 4-bit execution using the upstream row-major contract: `qweight [out, in/8]`, padded row-major `qzeros`, padded row-major scales, and natural nibble order. A dedicated CUDA kernel consumes persistent packed weights plus decoded F32 scales; checked-in CUDA 12.8.1 PTX is rebuilt and compared byte-for-byte in every CUDA workflow. Synthetic group-32 coverage forces four padded zero words per row. Low-level kernel parity and full expanded-query Qwen3 synthetic decode pass in run `29190022580`.
- 2026-07-12: Added a byte-counted, SHA-256-pinned real Qwen3 gate using `casimiir/Qwen3-0.6B-Base-awq-gemv-w4` revision `ad0963720d88c62b49f93b1bcec0db146576d1f1` and official `Qwen/Qwen3-0.6B-GGUF` Q8_0 revision `23749fefcc72300e3a2ad315e1317431b06b590a`. Initial run `29190226958` mapped all 702 physical/310 canonical tensors and passed selected real kernel-to-host parity, then exposed that Qwen3 query RMSNorm needed a `q_width` scratch buffer (`2048`) rather than the hidden-size buffer (`1024`). The runtime now owns a dedicated query-normalization scratch allocation, and the synthetic fixture uses expanded query geometry so this cannot regress silently. Final RTX gate `29190511600` validates all 196 packed matrices, loads 1,480,605,696 resident bytes in 3.518 seconds, matches zero/one-layer winners at tokens `785` and `2701`, generates the exact known token `" Paris"` on both native GEMV CUDA and official Q8_0 CPU paths, and passes the complete serial regression gate plus 165-second quiet-exit soak. Full-model first-fragment raw logits remain diagnostic across independently quantized checkpoints.
- 2026-07-12: Implemented the explicit `external-openai` server mode at the HTTP boundary. `xrt-server` can start or switch through `/v1/runtime/load` without loading a local model, forwards `/v1/models`, `/v1/completions`, and `/v1/chat/completions`, preserves arbitrary JSON request fields, upstream status/body, and raw SSE bytes including `[DONE]`, and exposes the configured base URL/model without exposing bearer credentials. External URLs are loopback-only by default, redirects are disabled, buffered bodies are capped at 16 MiB, SSE uses bounded backpressure, and remote hosts require `XRT_EXTERNAL_ALLOW_REMOTE=1`. Serial validation run `29191664582` passes default/CUDA server builds, JSON/auth forwarding, SSE framing, upstream error preservation, redacted status, process cleanup, and PTX reproducibility.
- 2026-07-12: Added `xrt-openai` as the shared pooled HTTP client/config boundary for server proxying and external benchmark comparison. `xrt bench --backends external-openai` no longer requires a local model, issues OpenAI chat SSE requests, supports synchronized concurrency, and reports upstream prompt/output tokens, first-chunk latency, total/mean/max request latency, aggregate throughput, preview, and errors while leaving local GPU/KV/scheduler fields null. Mixed local/external runs retain per-result tokenizer counts. SSE lines/events and error bodies are bounded, `[DONE]` is mandatory, and shared loopback/auth/redirect/timeout policy prevents server/CLI drift. Serial validation run `29192730833` passes shared-client policy tests, external-only CLI parsing, usage/output measurement, all server proxy regressions, default/CUDA builds, cleanup, and PTX reproducibility.
- 2026-07-12: Added live memory telemetry to runtime status and benchmark JSON. `GpuResourceStatus` now reports sampled device-wide CUDA usage plus xeno-tracked persistent model/KV/scratch bytes; the CLI reports current and process-lifetime peak host working set on Windows and Linux and carries the maximum tracked resident allocation across each measurement. Serial RTX parity run `29193409860` passes the complete gate. Bounded VibeThinker 3B Q4_K_M CPU/CUDA run `29193619302` generates one token per backend without errors and records CUDA device-used `7,133,855,744` bytes, xeno-tracked resident `5,518,314,144` bytes, and process peak resident `7,749,885,952` bytes; CPU process peak resident is `2,012,839,936` bytes. The CUDA sample is device-wide current usage at measurement time, not an exact transient high-water mark.
- 2026-07-12: Added clone-shared relaxed-atomic CUDA transfer counters for every explicit H2D, D2H, and D2D driver copy site. Runtime status exposes cumulative totals, while benchmark JSON snapshots a generation-only `transfer_delta` after model load. Full RTX parity run `29194759030` passes exact low-level copy counts and a synthetic two-token gate that requires exactly one final-logits download per token and no model-sized H2D transfer. Bounded VibeThinker 3B Q4_K_M run `29195003369` generates two tokens (`"The"`) at `1.889 tok/s`; generation performs 59 H2D calls totaling only `272` bytes, two D2H calls totaling `1,215,488` bytes (one `151,936`-logit F32 vector per token), and 792 D2D calls totaling `6,488,064` bytes, with no backend error. Initial assertion run `29195358437` exposed tracing text before the JSON object; the parser now extracts the bounded `xrt.bench` payload explicitly. Exact follow-up run `29195704139` passes the residency assertions, process cleanup, and quiet soak.
- 2026-07-12: Added clone-shared explicit CUDA allocation accounting across resident model buffers, decode scratch, F32/Q8/KQ4-VQ8 KV storage, page tables, adaptive routes, decode parameters, and bounded token-ID temporaries. Drop-safe leases decrement live bytes; replacement allocation records the old-plus-new overlap before the old buffer is released. Runtime status exposes cumulative `allocation_totals`, and benchmark JSON reports generation-interval `allocation_delta` with baseline, final, peak, allocation calls, and allocated bytes. Full RTX parity run `29196655601` passes low-level lease release, delta arithmetic, synthetic runtime parity, KV-growth peak accounting, and post-session release checks.
- 2026-07-12: The first bounded allocation-ledger model run `29196934828` exited with Windows access violation `-1073741819` before `CudaDevice::new` completed; startup telemetry showed `11,349 MiB / 24,564 MiB` already used with Unreal Editor active. It is not real-model allocation evidence. Heavy validation scripts now share `scripts/cuda-safety.ps1`, inspect the selected `XRT_CUDA_DEVICE`, and reject real GGUF/SafeTensors, benchmark, and server workloads before Cargo/model/CUDA initialization when initial device use exceeds `4,096 MiB`. Model-free serial run `29197616687` passes PTX reproducibility and the full synthetic parity gate. Guard proof run `29197847869` passes its compile gate, then stops the model smoke at `11,694 MiB / 24,564 MiB` with no GGUF open or CUDA initialization. A successful real-model `allocation_delta` remains pending until the runner GPU is below the safety threshold.
- 2026-07-12: `CudaDevice` now owns and configures the selected device's current stream-ordered CUDA memory pool. The runtime applies a bounded `256 MiB` default release threshold, enables event/opportunistic/internal-dependency reuse, resets pool high-water marks at device initialization, exposes used/reserved current and peak bytes through `GpuResourceStatus.memory_pool`, and provides an explicit trim operation. The focused hardware test proves a `64 KiB` allocation raises pool use, release returns used bytes to baseline, and trim does not increase reserved backing memory. Full model-free RTX 4090 validation and PTX reproducibility pass in run `29204963620`. Real-model pool/allocation-delta evidence remains pending because the shared runner is above the `4,096 MiB` safety threshold.
- 2026-07-12: Added the first low-level shared CUDA KV allocator slice for F32 pages. `CudaF32KvPagePool` lazily acquires bounded key/value pages from the stream-ordered CUDA allocator, retains released pages for user-space reuse, exposes live/free/acquire/reuse statistics, and can trim free pages. `CudaSharedF32LayerKvCache` owns a stable device pointer table, snapshots prefixes by sharing `Arc` page leases, and copies only a shared partial page before mutation. RTX test `shared_f32_kv_page_pool_reuses_pages_and_copies_partial_prefixes` proves full-page sharing, partial-page isolation, page return, reuse without another physical allocation, and trim in model-free run `29206337572`. This slice is not yet used by runtime decode; device-pointer attention kernels and cross-stream/CUDA Graph ordering must pass before replacing the contiguous session caches.
- 2026-07-13: Shared F32 KV pages now execute append and cross-page gather through their stable interleaved device pointer table, validated in model-free RTX run `29207272334`. A dedicated block-per-head kernel then reads the same pointer table directly for fused QK reduction, online softmax, and V accumulation without reconstructing contiguous KV. Run `29208223785` proves GQA, cross-page access, windowed attention, and partial-prefix copy-on-write isolation against the scalar reference. Runtime adoption remains gated on explicit cross-stream/event ordering and CUDA Graph pointer-lifetime rules; quantized shared page pools are still pending.

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

Implementation status as of 2026-07-12:

- `BackendKind` exists in `crates/xrt-runtime/src/backend.rs`.
- `CausalLmBackend` and `CpuBackend` wrap the existing `LlamaModel` execution surface.
- `BackendSession` exists and owns CPU or CUDA-resident session cache state.
- `Session` routes prefill, decode, speculative verification, hybrid state rollback, cache policy configuration, cache preparation, and rollback through backend/session boundaries.
- `Runtime::load_with_backend` accepts `auto`, `cpu`, `cuda-resident`, and `external-openai`.
- `auto` falls back to CPU by default and selects `cuda-resident` only for metadata-supported dense F32/F16/BF16/Q8_0/Q4_0/Q4_K/Q5_K/Q6_K GGUFs when CUDA is enabled and initializes successfully.
- `cuda-resident` exists for CUDA-feature builds on the supported dense F32/F16/BF16/Q8_0/Q4_0/Q4_K/Q5_K/Q6_K slice. `external-openai` is implemented by `xrt-server` at the HTTP boundary; direct token-level `Runtime::load_with_backend` calls reject it with a message directing callers to the server proxy mode.
- SafeTensors CUDA loading supports standard-dense Qwen2 and Qwen3 geometry. RTX-validated packed paths include AutoAWQ GEMM/GEMV, GPTQ v1/v2 including act-order, and compressed-tensors W4A16; unsupported methods and layouts fail before upload.
- `xrt generate` and `xrt chat` accept `--backend` and `XRT_BACKEND`.
- `xrt bench` accepts comma-delimited `--backends`, reports the active backend in the benchmark table, and executes `external-openai` through the shared pooled HTTP client instead of the token-level runtime. External-only runs do not require a local model; mixed runs report tokenizer-specific prompt counts per result.
- `xrt bench --json` emits a structured report with model path, inferred quantization, prompt token count, OS/arch, git commit, CUDA feature flag, per-backend load time, generation metrics, GPU resource status, and unsupported-backend errors.
- `xrt-server` accepts `--backend`, `POST /v1/runtime/load` accepts `backend`, and `GET /v1/runtime/status` reports requested and active backend values.
- `GpuResourceManager` exists as a runtime-level scaffold with `XRT_CUDA_DEVICE`, `XRT_GPU_MEMORY_FRACTION`, `XRT_GPU_RESERVED_MB`, and `XRT_GPU_KV_FRACTION` parsing.
- `GET /v1/runtime/status` and `POST /v1/runtime/load` report live CUDA free/total/device-used bytes and xeno-tracked persistent model/KV/scratch allocation once the CUDA backend loads.
- `xrt bench --json` reports current and process-lifetime peak host resident bytes on Windows and Linux. Its `tracked_resident_vram_bytes` remains the maximum sampled model/KV/scratch category sum; `allocation_delta.peak_bytes` is the exact high-water mark for explicit xeno-owned CUDA allocations during the generation interval.
- Runtime GPU status exposes cumulative explicit allocation current/peak bytes, allocation calls, and total allocated bytes. The ledger does not claim device-wide or driver-internal allocation visibility.
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
- `XRT_CUDA_POOL_RELEASE_THRESHOLD_MB=256` (bounded to `0..=4096`)

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
- cumulative explicit H2D, D2H, and D2D transfer calls and bytes
- stream-ordered memory-pool release threshold plus used/reserved current and peak bytes

Telemetry semantics:

- `device_used_vram_bytes` is sampled device-wide usage computed from live CUDA total/free memory. It can include allocations owned by other processes and is not a transient high-water mark.
- `tracked_allocated_bytes` is the current xeno-owned persistent model-weight, KV, and scratch allocation total.
- Benchmark `tracked_resident_vram_bytes` is the maximum sampled `tracked_allocated_bytes` during that measurement.
- `allocation_totals` records current and peak bytes held by explicit xeno-rt CUDA buffers plus cumulative allocation calls and allocated bytes since `CudaDevice` initialization. Drop-safe leases keep the current count balanced when buffers are released.
- Benchmark `allocation_delta` resets the explicit peak to the current baseline immediately before session work, then reports baseline, final, interval peak, calls, and allocated bytes after session release. It includes temporary explicit allocation overlap but excludes driver-internal memory and allocations from other processes.
- `memory_pool` reports the current CUDA pool's configured release threshold, live/peak bytes in use, and live/peak backing bytes reserved by the driver. It is nullable on CPU and on CUDA devices without stream-ordered pool support. Pool used bytes can exceed `allocation_totals.current_bytes` because the driver pool is broader and allocation granularity differs; use the explicit ledger for xeno category accounting.
- The stream-ordered pool is the central allocator for current cudarc `CudaSlice` allocations, but it is not the shared logical KV page allocator described in Phase 3/7. It does not yet provide per-category reuse policy, defragmentation attribution, or page-granular prefix copy-on-write.
- `host_memory.process_peak_resident_bytes` is the operating system's process-lifetime resident working-set high-water mark. Device-wide transient peak sampling remains unavailable; use the explicit allocation peak for xeno-owned buffers.
- `transfer_totals` counts successful explicit driver copies since `CudaDevice` initialization. Benchmark `transfer_delta` subtracts the snapshot taken immediately before session work, so model-load uploads are excluded from the measured generation interval.
- Transfer counters are observational relaxed atomics. They do not add a CUDA synchronization and do not include implicit driver migration outside xeno-rt's explicit copy calls.

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

Phase 1 follow-up status:

- Complete: CI exercises benchmark aggregation, JSON-producing real-model smokes, and external benchmark parsing/measurement helpers.
- Complete: the explicit external OpenAI server proxy and benchmark adapter support comparison against compatible local runtimes without changing token-level native runtime behavior.

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
- Done for the initial standard-dense targets: bounded multi-token VibeThinker and Gemma4 text-output, correctness, and throughput evidence is recorded above. Broader architecture/model/quantization coverage remains incremental.

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
- A separate low-level `CudaF32KvPagePool`/`CudaSharedF32LayerKvCache` path now validates lazy bounded page allocation, stable pointer tables, reusable page leases, prefix sharing, partial-page copy-on-write, device-pointer append/gather, and direct online attention. Model-free RTX runs `29206337572`, `29207272334`, and `29208223785` cover allocator ownership, cross-page access, GQA, attention windows, and copy-on-write isolation. Runtime decode still uses the established contiguous cache representation until cross-stream/event ordering and CUDA Graph pointer-lifetime rules are validated. Q8, KQ4/VQ8, and agent-adaptive shared pools remain pending.

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

Implementation status as of 2026-07-12: complete for the initial standard-dense F32-KV target.

- Request admission is bounded by `max_active_sequences` and `max_queued_sequences`; queue saturation returns HTTP 429. Defaults remain conservative at one active sequence until broader aggregate-memory data is available.
- Streaming uses bounded channels. Client disconnects cancel generation before the next model call, preventing abandoned streams from retaining an active slot or unbounded buffered chunks.
- Dense sessions split prompt work by `prefill_chunk_tokens` and acquire FIFO prefill/decode turns. Decode receives bounded priority through `max_decode_turns_before_prefill`, while same-phase FIFO tickets prevent a long prompt from immediately reacquiring every prefill turn.
- Hybrid/recurrent architectures hold an exclusive execution turn because their recurrent state is backend-global; they remain correct but do not participate in dense-session interleaving yet.
- Scheduled CUDA sessions reserve their worst-case request-horizon KV growth peak against one aggregate scheduler budget. Runtime status reports budget, live reservations, active/queued sequences, prefill/decode waiters and turns, overlap counters, admissions, and rejections.
- Compatible standard-dense F32-KV decode calls rendezvous for at most `max_decode_batch_size` sequences within `decode_batch_wait_micros`. Defaults are batch 4 and 20 ms, but the server's conservative `max_active_sequences=1` default keeps single-user behavior unchanged until operators opt into concurrency.
- The scheduler owns each submitted `BackendSession` while it waits and returns that same state with its logits. No request thread lends a mutable session across threads, and backend reloads cannot mix because jobs batch only when their backend `Arc` identities match.
- Every sequence first captures its normal 759-node batch-1 graph. The runtime then builds a bounded cache of parent graphs whose child graph nodes have no dependency edges, giving one parent launch with isolated per-session KV/scratch and parallel-ready child execution. Pointer-changing cache/scratch growth increments the session graph epoch and prevents stale parent reuse.
- If parent composition or launch is unavailable, dedicated nonblocking CUDA streams replay the already-captured per-session graphs concurrently. If batching is unsupported or disabled, the existing serial graph/eager path remains available.
- Runtime status reports pending/active batch size, submitted/completed batch items, completed batches, fused parent batches, and maximum observed batch size. Session graph status reports `batch-captured` after successful multi-sequence replay.
- The CLI supports synchronized `--concurrency`, aggregate throughput, mean/max request latency, and scheduler JSON. Matched final RTX runs average `12.41`, `17.46`, and `22.05 tok/s` at concurrency 1, 2, and 4. Concurrency 2 is 7.8% faster than the prior cooperative-only result while lowering mean request latency about 8.0%.
- Real OpenAI-compatible concurrent streaming, long-prefill/decode overlap, fused size-2 replay, cancellation/resource drain, and process cleanup pass in run `29175405002`.

Post-Phase6 extensions (not blockers for the validated initial target):

- Extend parent-graph batching to Gemma4 variable-width layers and Q8/KQ4/adaptive KV once those paths gain stable per-session graph capture.
- Hybrid/recurrent architectures remain exclusive because recurrent state is backend-global; they need session-owned recurrent state before interleaving or batching.
- A future kernel-level tensor batch may outperform parallel child graphs at larger batch sizes, but it must beat the recorded parent-graph benchmark before replacing this path.

## Phase 7: Prefix Cache / Radix Cache

Implement prefix reuse for repeated system prompts, tools, memory blocks, and document-store workloads.

Status: Initial exact-prefix target complete and RTX validated on 2026-07-12.

Tasks:

- Hash token prefixes by model, tokenizer, prompt segment, and backend.
- Store immutable prefix KV pages.
- Allow sessions to attach prefix pages copy-on-write.
- Add eviction policy.
- Integrate with agent span policy from `AGENT_ADAPTIVE_KV_ROADMAP.md`.

Acceptance:

- Repeated system prompt prefill is skipped. Complete: real-model hits skip 19 of 20 prompt tokens.
- Runtime status reports prefix-cache hit rate. Complete: status also reports entries, bytes, lookups, hits, misses, tokens saved, inserts, evictions, and rejections.
- Cache invalidation is deterministic across model/tokenizer changes. Complete: cache ownership is runtime-scoped and keys include the model/tokenizer namespace plus backend, KV mode, policy, tokens, and structural spans.

Implemented contract:

- `XRT_PREFIX_CACHE=0|1` controls the feature and defaults to enabled.
- `XRT_PREFIX_CACHE_MAX_ENTRIES` defaults to `32`.
- `XRT_PREFIX_CACHE_MAX_BYTES` defaults to `268435456` bytes.
- `XRT_PREFIX_CACHE_MIN_TOKENS` defaults to `8`.
- The reusable boundary includes leading system, developer, tool-schema, and policy-pinned spans. Requests without span metadata reuse all prompt tokens except the final token, which is always executed to produce correct logits.
- Images and hybrid/recurrent paths bypass prefix reuse until their state can be represented as immutable session-owned snapshots.
- CPU snapshots share immutable KV pages and copy only mutated pages.
- CUDA snapshots preserve page tables and compressed payloads. Attach is pointer-only; first mutation performs one bounded device-to-device copy, including a growth copy when required, and checks snapshot-plus-mutable peak bytes against the KV budget.
- LRU eviction drops only the manager's ownership. Attached sessions retain their snapshot through `Arc`, so eviction cannot invalidate active generation.
- Retained CUDA prefix bytes count against scheduler admission through `kv_external_reserved_bytes` and are exposed in runtime/server status without changing OpenAI completion schemas.
- Benchmark JSON includes cumulative prefix-cache status for each measurement.

Validation:

- CPU manager tests cover exact key dimensions, structural span boundaries, uncached suffix reuse, entry eviction, bounded configuration, and repeated-prompt output parity.
- CPU F32, Q8, and KQ4/VQ8 tests prove page-level copy-on-write isolation.
- CUDA clone tests preserve remapped F32/Q8/KQ4-VQ8 page tables, compressed bytes, adaptive routes, growth capacity, and source/clone independence.
- Runtime RTX tests cover repeated-prompt output parity and cache hits in F32, Q8, KQ4/VQ8, and agent-adaptive modes.
- Real-model benchmark and OpenAI SSE evidence are recorded in runs `29177251045`, `29177391329`, and `29177699687`.

Post-Phase7 extensions (not blockers for the validated initial target):

- Replace exact hashed structural-prefix entries with a radix tree when longest-prefix matching across partially shared prompts is needed.
- Integrate the validated low-level F32 page pool with runtime snapshots so first write copies only touched pages instead of the complete session allocation; extend the same ownership contract to Q8, KQ4/VQ8, and agent-adaptive storage.
- Add multimodal prefix snapshots once image-embedding identity and cache invalidation are explicit.

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

Implementation status as of 2026-07-12:

- Complete: read-only single-file and sharded SafeTensors bundle validation in `xrt-safetensors`.
- Complete: typed Hugging Face model/config metadata, including normalized AWQ, GPTQ, and compressed-tensors declarations with conflict and geometry checks.
- Complete: Hugging Face BPE/tokenizer asset loading with exact token-ID parity against the equivalent GGUF tokenizer.
- Complete: format-neutral `ResidentTensorSource` boundary for CUDA support checks, weight upload, optional tensors, tied output handling, and resident VRAM estimation. GGUF remains the default adapter and passes the full existing RTX parity gate.
- Complete for the first execution target: dense Qwen2 F32/F16/BF16 SafeTensors directories load through `Runtime` on CUDA and execute the existing resident embedding, RMSNorm, attention, FFN, paged KV, and output kernels without a second transformer implementation.
- RTX validated: VibeThinker 3B BF16 SafeTensors zero-layer, one-layer, and full-model greedy logits select the same winning tokens as the equivalent Q4_K_M GGUF CPU reference in runs `29179748626` and `29179999285`, with every observed winning-score delta below `0.1`. The final gate also proves exact one-token generated-text parity through normal `Runtime` sessions.
- Complete for the first packed execution target: AutoAWQ `GEMM`, 4 bits, asymmetric zero points, and group sizes `-1`, `32`, `64`, or `128` map into native resident CUDA matrices while dense embedding, norm, bias, and output tensors share the same transformer executor.
- RTX validated: pinned Qwen2.5 0.5B AutoAWQ zero-layer, one-layer, and full-model greedy logits select the same winners as the official Q8_0 GGUF CPU reference in run `29182210301`; deterministic one-token generation is exactly `" Paris"` on both paths.
- Complete for the second packed execution target: GPTQ v1 `GEMM`, 4 bits, symmetric zero points, standard non-act-order `g_idx`, and group sizes `-1`, `32`, `64`, or `128` map into separate native resident CUDA matrices without reusing AutoAWQ packing assumptions.
- RTX validated: pinned Qwen2.5 0.5B GPTQ v1 real matrices match host dequantization and full-prompt deterministic generation is exactly `" Paris"` on both native GPTQ CUDA and the official Q8_0 GGUF CPU reference in run `29183838567`. Zero/one-layer cross-format winners match; full-model first-fragment logits remain diagnostic because independently quantized checkpoints diverge there.
- Complete for the third packed execution target: compressed-tensors `pack-quantized` W4A16, symmetric static group weights, `actorder=group`, and group sizes `32`, `64`, or `128` map into native resident CUDA matrices with explicit permuted group-index lookup.
- RTX validated: pinned Qwen2.5 0.5B W4A16 real matrices match host dequantization and full-prompt deterministic generation is exactly `" Paris"` on both native compressed-tensors CUDA and the official dense BF16 CUDA reference in run `29185883493`.
- Complete for the fourth packed execution target: GPTQ v1 act-order and GPTQ v2 direct-zero 4-bit checkpoints use a separate resident matrix/kernel with explicit group-index lookup and typed zero encoding. Standard monotonic GPTQ v1 remains on its previously validated path.
- RTX validated: pinned Qwen2.5 1.5B GPTQ v1 act-order matrices match host dequantization and generate the same deterministic `" Paris"` token as the official dense BF16 CUDA reference. A hash-pinned derived Qwen2.5 0.5B GPTQ v2 fixture matches the source v1 runtime exactly at zero, one, and all transformer layers and generates the same token. Full evidence is in run `29188273180`.
- Complete for the fifth packed execution target: AutoAWQ `GEMV`, 4 bits, asymmetric zero points, natural nibble order, and upstream row-major padded zero/scale geometry map into a dedicated native resident CUDA matrix and kernel. Standard-dense Qwen3 adds required per-head Q/K norms and supports expanded query widths without reusing hidden-size scratch.
- RTX validated: pinned Qwen3 0.6B AutoAWQ GEMV maps all 196 packed matrices, selected real kernels match host dequantization, zero/one-layer winners match the official Q8_0 GGUF CPU reference, and both full runtimes generate the exact known token `" Paris"` in run `29190511600`.
- Explicitly pending: AWQ variants beyond validated GEMM/GEMV contracts, EXL3 evaluation, SafeTensors CPU decode, architectures beyond Qwen2/Qwen3, and broader independent GPTQ v2 checkpoint coverage beyond the derived semantic fixture.
- Safety: SafeTensors model directories require a CUDA-enabled build and `cuda` or `auto`; requesting CPU returns a clear unsupported error. Unsupported quantization methods and packing versions are rejected before upload and are never reinterpreted as AutoAWQ or GPTQ GEMM.

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
- `XRT_EXTERNAL_MODEL=...` optionally supplies a default request model.
- `XRT_EXTERNAL_TIMEOUT_SECONDS=300` bounds connect/read/write operations.
- `XRT_EXTERNAL_ALLOW_REMOTE=1` explicitly opts into non-loopback hosts.
- `XRT_BACKEND=external-openai`

Acceptance:

- `/v1/chat/completions` can proxy to a local external runtime.
- Streaming proxy preserves OpenAI-compatible SSE framing.
- Runtime status makes proxy mode obvious.

Implementation status as of 2026-07-12: server and benchmark adapter complete and validated.

- Startup flags and `POST /v1/runtime/load` can activate `external-openai` without a GGUF/SafeTensors model; load/unload transitions clear incompatible local/proxy state and scheduler GPU reservations.
- `/v1/models`, `/v1/completions`, and `/v1/chat/completions` forward bearer authorization and preserve upstream HTTP status, content type, body, arbitrary JSON fields, and default-model injection.
- Streaming responses pass through raw bytes over a bounded channel, preserving OpenAI SSE framing and `[DONE]` rather than decoding and rebuilding events.
- `GET /v1/runtime/status` reports `requested_backend`, `active_backend`, `external_base_url`, and `external_model`; API keys are never serialized or included in `Debug` output.
- The default security boundary permits loopback hosts only, rejects credentials/query/fragment in base URLs, disables redirects, caps buffered responses at 16 MiB, and requires explicit remote-host opt-in.
- Validation run `29191664582` passes the complete serial safe gate and all focused proxy compatibility tests.
- `xrt-openai` centralizes pooled HTTP transport, bearer auth, default-model injection, timeout policy, redirect rejection, and URL validation for both server and CLI.
- `xrt bench --backends external-openai` measures chat SSE first-chunk and total latency, usage-token throughput, synchronized concurrency, preview, and upstream errors. External-only runs do not require GGUF/SafeTensors; mixed backend runs retain local-model requirements for the local results.
- External benchmark JSON keeps GPU resource, prefix-cache, and scheduler fields null because those resources belong to the upstream runtime rather than xeno-rt. Per-result `prompt_tokens` avoids conflating upstream and local tokenizer counts.
- Validation run `29192730833` passes the shared-client and external benchmark tests plus every prior proxy compatibility test through the complete serial safe gate.

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
- `crates/xrt-openai/src/lib.rs`
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
- CUDA stream-ordered memory-pool allocation/release/trim telemetry tests.
- CUDA decode parity against CPU on tiny F32/F16 fixtures.
- Quantized GGUF parity with loose tolerances.
- OpenAI API response schema unchanged.
- External proxy streaming SSE framing.

Local verification safety:

- On the shared Windows/RTX workstation, run Cargo, model, server, and CUDA execution through the serialized GitHub Actions workflow rather than directly in the interactive checkout.
- `safe-cuda-smoke.ps1`, `safe-cuda-server-smoke.ps1`, and real-model/SafeTensors modes in `safe-cuda-check.ps1` source `cuda-safety.ps1`. The helper checks the selected `XRT_CUDA_DEVICE` and defaults to rejecting the workload when existing GPU use exceeds `4,096 MiB`, before Cargo or model initialization.
- Do not raise or bypass `MaxInitialGpuMemoryUsedMB` while Unreal Editor or another GPU-heavy application is active. Synthetic parity remains separately runnable because it does not load a real model.
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
| Host-device copies regress into the hot path | No speedup | Keep cumulative and per-benchmark transfer telemetry plus synthetic/real residency assertions |
| GPU memory fragmentation | OOM during long sessions | Use the configured stream-ordered CUDA pool now; add shared logical KV page allocation and workload-level eviction next |
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
