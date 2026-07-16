# Agent-Adaptive KV Roadmap

## Goal

Make `xeno-rt` the best local runtime for persistent desktop agents, not just a generic GGUF server.

The differentiator is not plain KV-cache quantization by itself. The differentiator is an
**agent-aware KV policy system** that decides what stays high-fidelity, what gets compressed,
and what can be degraded first based on the structure of an agent session.

This is the path that can make `xeno-rt` meaningfully more novel than `llama.cpp` for the XENO use case.

## Why This Path

`llama.cpp` is already strong at:

- general GGUF inference
- OpenAI-compatible serving
- structured outputs / grammar-constrained generation
- generic runtime engineering

Competing by only adding static KV quantization would make `xeno-rt` a smaller general-purpose clone.

`xeno-rt` should instead optimize for what XENO actually runs:

- long-lived tool-using agents
- multi-turn workspace sessions
- repeated tool schemas and tool results
- retrieval-heavy prompts
- desktop memory constraints

The novelty target is:

- **agent-native memory policy**
- **request-scoped cache behavior**
- **long-session quality retention**

## Product Outcome

The desired steady state is:

- `xeno-rt` remains OpenAI-compatible
- model loading and chat behavior stay boring and reliable
- KV strategy becomes session-aware and role-aware
- older low-value context gets compressed before high-value context
- important agent context remains pinned or high-fidelity

In practice:

- recent turns stay `f32`
- system prompt stays pinned
- tool schema stays pinned
- tool results stay pinned longer than ordinary assistant chatter
- retrieval chunks can be compressed more aggressively after they become cold
- old conversational filler degrades first

## Core Idea

Introduce a **policy layer above the KV backend**.

Instead of only:

- `f32`
- `q8`
- `turboquant`

the runtime should choose behavior from:

- `default_chat`
- `agent_adaptive`
- `long_context`
- `memory_saver`

Each policy decides:

- which spans are pinned
- which spans remain high-fidelity
- when spans become compressible
- which compression mode is allowed for each span
- whether eviction is allowed at all

## Design Pillars

### 1. Request-Scoped Policy, Not Global Env Only

Today KV mode is mostly driven by `XRT_KV_CACHE_MODE`.

That is too coarse for a desktop agent runtime.

We need:

- a runtime default
- an optional per-request override
- a policy object attached to each session

Target:

- server startup default: `f32`
- per-request policy: `agent_adaptive`

### 2. Semantic Span Classification

The runtime should classify prompt segments into span types:

- `system`
- `developer`
- `tool_schema`
- `tool_result`
- `retrieval`
- `user`
- `assistant`
- `scratch`

These span types drive KV retention and compression strategy.

### 3. Pinned and Protected Context

Some spans should never be degraded first:

- system prompt
- developer instructions
- active tool schema
- current task plan
- most recent tool outputs

The point is not perfect permanence. The point is **better degradation order**.

### 4. Key-First Compression

The first serious adaptive implementation should compress keys before values.

Reason:

- keys dominate attention score behavior
- this is the most research-aligned path
- it maps cleanly onto the current TurboQuant groundwork

### 5. Safety for Speculative Decoding and Rollback

Everything must preserve:

- append
- append_batch
- truncate
- replay after rollback

If speculative decoding becomes fragile, the feature is not production-ready.

## Proposed Runtime API

### Internal Session Config

Add a session config object, conceptually like:

```rust
pub struct SessionPolicy {
    pub cache_policy: CachePolicyKind,
    pub recent_window_tokens: usize,
    pub pin_system_prompt: bool,
    pub pin_tool_schema: bool,
    pub protect_tool_results: bool,
    pub retrieval_compression_delay: usize,
}
```

### Cache Policy Enum

```rust
pub enum CachePolicyKind {
    DefaultChat,
    AgentAdaptive,
    LongContext,
    MemorySaver,
}
```

### Span Metadata

At prompt-build time, capture span metadata, conceptually:

```rust
pub struct PromptSpan {
    pub kind: PromptSpanKind,
    pub token_start: usize,
    pub token_end: usize,
    pub priority: SpanPriority,
}
```

This metadata stays in the session so the cache layer knows which token ranges matter more.

## Implementation Phases

### Phase 0: Instrumentation

Add observability before changing behavior.

Ship:

- token-count breakdown by span kind
- KV memory by span kind
- session policy name in runtime status
- debug logs for cache decisions

Acceptance:

- we can see where context memory goes in real agent sessions

### Phase 1: Session Policy Plumbing

Add session-level policy configuration across:

- `xrt-server`
- `xrt-cli`
- runtime session creation

Do not change cache behavior yet.

Acceptance:

- sessions can be created with a named policy
- policy is visible in status/debug output

### Phase 2: Prompt Span Metadata

Track prompt spans in the runtime.

Initial span extraction should support:

- system messages
- user messages
- assistant messages
- tool schema block
- tool results

Acceptance:

- session knows which token ranges belong to which semantic span type

### Phase 3: Protected Window + Pinned Spans

Implement the first adaptive rule set:

- keep recent N tokens in `f32`
- keep pinned spans in `f32`
- mark old non-pinned spans as compressible

No aggressive TurboQuant logic yet.

Acceptance:

- no quality regression on short sessions
- stable improvement in memory pressure on long sessions

### Phase 4: Adaptive Compressed-Key Backend

Use the current KV abstraction to add compressed-key behavior for cold spans.

Start with:

- `f32` recent window
- compressed keys for older non-pinned spans
- conservative value handling

Acceptance:

- long-session memory drops materially
- tool-call accuracy remains stable on `Qwen 3.5 4B` and `9B`

### Phase 5: Retrieval and Tool-Result Policies

Differentiate by span type:

- retrieval chunks become compressible sooner
- tool results stay protected longer
- stale assistant chatter degrades first

Acceptance:

- better long-session task continuity than a flat policy

### Phase 6: TurboQuant-Like Experimental Mode

Only after the adaptive policy system is stable:

- add a more advanced compressed-key mode
- evaluate PolarQuant / TurboQuant-style transforms as the backend for cold spans

Acceptance:

- measurable improvement over simple `q8`
- no unacceptable regression on agent workloads

## Evaluation Matrix

Do not judge this only on tokens/sec.

Track:

- first-token latency
- decode tokens/sec
- memory footprint
- max usable context
- long-context instruction retention
- tool-call accuracy
- tool argument correctness
- recovery after long tool chains
- speculative decoding correctness under rollback

Required model set:

- `Qwen 3.5 4B Q4_K_M`
- `Qwen 3.5 9B Q4_K_M`

Optional:

- small regression check on `0.8B`

## Default Rollout Strategy

Do not make adaptive compression global immediately.

Recommended rollout:

1. Keep runtime default as `f32`
2. Add `agent_adaptive` as opt-in
3. Evaluate on real Hub agent sessions
4. Make `agent_adaptive` the default only for:
   - long sessions
   - larger local models
   - agent/tool mode

## Non-Goals

For the first serious version, do not try to solve:

- full model-weight re-quantization
- GPU-specific compressed attention kernels
- third-party model marketplace policy tuning
- replacing every generic server use case of `llama.cpp`

The goal is narrower and more strategic:

- make `xeno-rt` the best runtime for persistent local XENO agents

## Success Criteria

This roadmap succeeds if:

- `xeno-rt` keeps chat/tool behavior stable on `Qwen 3.5 4B` and `9B`
- long agent sessions retain quality better than a flat KV policy
- memory usage is materially lower than plain `f32`
- the runtime exposes a real product-level story that `llama.cpp` does not have by default

That story is:

- **agent-aware local runtime memory management**

