# TurboQuant KV Cache Plan

**Runtime domain:** `xrt-text`
**Canonical architecture:** [RUNTIME_DOMAINS.md](RUNTIME_DOMAINS.md)

This plan concerns autoregressive text KV storage only. Other runtime domains
reuse shared memory/resource infrastructure but define their own state and
quality contracts.

## Goal

Add KV-cache compression to `xeno-rt` so larger contexts and larger local models fit in memory with lower latency and lower memory bandwidth pressure.

This plan is specifically for KV-cache compression, not full model-weight quantization.

For the product-default roadmap that should differentiate `xeno-rt` from generic runtimes, see:

- [AGENT_ADAPTIVE_KV_ROADMAP.md](AGENT_ADAPTIVE_KV_ROADMAP.md)

## Current Status

The first foundation slice is now implemented in `xeno-rt`:

- `KvCacheMode` supports `f32` and experimental `q8`
- `SessionKvCache` can select either cache backend
- attention now reads KV rows through the cache abstraction instead of assuming only raw `&[f32]`
- `xrt-server` reports the active KV-cache mode in `/v1/runtime/status`

Current activation:

- default: `f32`
- experimental: set `XRT_KV_CACHE_MODE=q8`

This is not TurboQuant yet. It is the baseline compressed-cache plumbing needed before a TurboQuant-style implementation can land safely.

It is also the plumbing needed for the preferred XENO product path:

- request-scoped cache policies
- pinned spans
- adaptive compression for persistent tool-using agents

## Why

Current `xeno-rt` KV-cache storage is full-precision float storage:

- [kv_cache.rs](../crates/xrt-runtime/src/kv_cache.rs)
- [lib.rs](../crates/xrt-core/src/lib.rs)
- [llama.rs](../crates/xrt-models/src/llama.rs)

Today:

- keys and values are stored as `Vec<f32>`
- attention reads raw float slices from the cache
- speculative decoding rollback assumes a straightforward append/truncate model

That is simple and correct, but it does not scale well for long contexts.

## Target Outcome

Support multiple KV-cache modes:

- `f32`
- `int8` or conservative blockwise quantized mode
- `turboquant` experimental mode

The long-term goal is to make TurboQuant-style compression available where it provides real memory/context gains without unacceptable quality loss.

For XENO, that should likely ship as the backend of an agent-aware cache policy, not as a naked global mode toggle.

## Implementation Phases

### Phase 1: Architecture Refactor

1. Split the current cache implementation into explicit variants.

   Target:

   - `PagedKvCacheF32`
   - `PagedKvCacheQuantized`

2. Refactor the `KvCache` interface in [lib.rs](../crates/xrt-core/src/lib.rs).

   Current interface assumes direct float slice access:

   - `key(...) -> Option<&[f32]>`
   - `value(...) -> Option<&[f32]>`

   That is too restrictive for compressed storage.

3. Introduce a cache-read abstraction that lets attention consume either:

   - raw float rows
   - decompressed-on-read rows
   - compressed rows scored through specialized kernels

4. Keep these semantics intact:

   - append
   - append_batch
   - truncate
   - clear

### Phase 2: Conservative Quantized KV Cache

5. Implement a simple baseline quantized KV cache before TurboQuant.

   Recommended first version:

   - blockwise int8 KV
   - or 4-bit KV with explicit per-block scale metadata

6. Add a runtime flag for cache mode selection.

   Example:

   - `--kv-cache f32`
   - `--kv-cache int8`
   - `--kv-cache q4`
   - later `--kv-cache turboquant`

7. Wire the cache mode into `xrt-server` and `xrt-cli`.

### Phase 3: Attention Integration

8. Refactor attention in [llama.rs](../crates/xrt-models/src/llama.rs) so it no longer assumes raw `&[f32]` cache rows.

9. Add one of these strategies:

   - decompress per row before dot product
   - decompress batched pages into scratch buffers
   - add direct compressed-key scoring kernels where worthwhile

10. Keep hybrid-model and speculative-decoding behavior correct.

   This especially affects:

   - rollback after rejected draft tokens
   - `truncate()` semantics
   - replay after rollback

### Phase 4: Benchmark and Evaluation Harness

11. Add benchmarks for:

   - memory usage
   - first-token latency
   - decode tokens/sec
   - long-context throughput

12. Add quality evaluation for:

   - perplexity drift
   - long-context prompt retention
   - instruction-following regression
   - agent/tool-call accuracy regression

13. Compare:

   - `f32`
   - conservative quantized cache
   - TurboQuant experimental mode

### Phase 5: TurboQuant Experimental Implementation

14. Implement TurboQuant-style compression after the cache abstraction and eval harness are stable.

15. Start with keys first.

   Rationale:

   - keys are central to attention score computation
   - this is where the memory and bandwidth wins matter most

16. Evaluate whether values should use:

   - the same compression mode
   - a different mode
   - or remain less aggressively compressed

17. Keep the implementation behind an experimental feature/flag until quality and performance are proven.

## Acceptance Criteria

TurboQuant mode should not be considered production-ready until:

- no correctness regressions in cache append/truncate/replay behavior
- long-context quality stays within acceptable bounds
- memory savings are material on real `xeno-rt` workloads
- performance is neutral or positive on supported models
- agent/tool-call behavior does not regress under local runtime use

## Risks

1. `KvCache` API churn will touch core runtime and model code.
2. Quantized cache reads can become slower if decompression strategy is naive.
3. Small models may not benefit enough to justify complexity.
4. Speculative decoding rollback can become subtle with compressed page layouts.
5. Paper/blog claims do not equal drop-in implementation details.

## Recommended Order

1. Refactor cache abstractions.
2. Add conservative quantized KV mode.
3. Build benchmarks and eval harness.
4. Implement TurboQuant-style experimental path.
5. Tune rollout policy and defaults.

## Non-Goals For First Iteration

- full model weight re-quantization
- vector-search indexing integration
- GPU-specific compressed attention kernels
- making TurboQuant the default immediately
