# Gemma4 GGUF Support Spec

Status: Implemented for dense text local 12B GGUFs; remaining gaps tracked below
Date: 2026-06-17
Primary target: local `gemma-4-12b-coder` GGUF aliases in `X:\ai\models\llm`
**Runtime domain:** `xrt-text`
**Canonical architecture:** [RUNTIME_DOMAINS.md](RUNTIME_DOMAINS.md)

## Objective

Add native Gemma4 text-generation support to xeno-rt without breaking existing Llama, Qwen2, Qwen3, Qwen3.5, GGUF, CPU fallback, or OpenAI-compatible API behavior.

The first deliverable is text-only generation for the local 12B coder GGUFs:

- `gemma-4-12b-coder` -> Q6_K default
- `gemma-4-12b-coder-q4`
- `gemma-4-12b-coder-q8`

Multimodal image/audio/video **inputs whose advertised output remains conversational text** are follow-up `xrt-text` phases after text correctness. Generative image, video, or audio output belongs to `xrt-image`, `xrt-video`, or `xrt-audio` rather than this model plan. Assistant drafter/MTP support and TurboQuant KV compression are also follow-up phases.

## Implementation Status

Implemented in the current xeno-rt branch:

- Gemma4 GGUF architecture recognition and scalar/array metadata normalization.
- Per-layer local/global attention config, including different Q/KV widths and sliding-window masks.
- Dense text Gemma4 tensor loading, including optional missing `attn_v.weight` fallback to K projection.
- CPU single-token forward pass with Gemma4 embedding scaling, Q/K/V norms, GELU-gated FFN, post-attention/post-FFN norms, layer output scale, and final logit softcap.
- Batch/prefill fallback to token-by-token execution for correctness.
- Local aliases for `gemma-4-12b-coder`, `gemma-4-12b-coder-q4`, `gemma-4-12b-coder-q8`, and `vibethinker-3b` variants.

Still intentionally not implemented:

- Gemma4 shared-KV tail-layer reuse when `gemma4.attention.shared_kv_layers > 0`.
- Optimized batch prefill for variable-width Gemma4 layers.
- Gemma4 multimodal/audio/video token ingestion.
- Gemma4-specific tokenizer parity tests and tool/reasoning output parsing.

## Research Summary

Gemma4 is a real current target, not an alias of Gemma2/Gemma3. Google documents Gemma4 with text, audio, and image input, long context up to 256K, and official llama.cpp GGUF usage. Google also documents Gemma4 12B as a dense multimodal model with a unified encoder-free architecture and the same advanced decoder structure as Gemma4 31B Dense.

Important architecture facts:

- Gemma4 uses hybrid local/global attention. Gemma3 used a 5-local-to-1-global pattern, and Gemma4 config files expose equivalent per-layer `layer_types` / GGUF `attention.sliding_window_pattern`.
- Sliding-window and full-attention layers have different head dimensions. Public cache documentation calls this a heterogeneous KV cache problem: sliding head dim 256, full head dim 512.
- Some Gemma4 variants use cross-layer shared KV cache. GGUF metadata exposes `gemma4.attention.shared_kv_layers`.
- Gemma4 text config exposes `final_logit_softcapping = 30.0`, `hidden_activation = "gelu_pytorch_tanh"`, `num_kv_shared_layers`, `head_dim`, `global_head_dim`, `sliding_window`, and a 262144 vocabulary.
- GGUF metadata keys include Gemma4-specific fields: `attention.key_length_swa`, `attention.value_length_swa`, `attention.shared_kv_layers`, `attention.sliding_window_pattern`, `rope.dimension_count_swa`, and `rope.freq_base_swa`.
- vLLM documents Gemma4-specific function-calling/reasoning parsers and custom tool-call tokens. This is API/template work, not required for first-token text correctness.

Sources:

- Google Gemma with llama.cpp: https://ai.google.dev/gemma/docs/integrations/llamacpp
- Google Gemma4 12B developer guide: https://developers.googleblog.com/gemma-4-12b-the-developer-guide/
- Google DeepMind Gemma4 model page: https://deepmind.google/models/gemma/gemma-4/
- Hugging Face `google/gemma-4-E4B-it` config: https://huggingface.co/google/gemma-4-E4B-it/blob/main/config.json
- llama.cpp GGUF metadata constants: https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/constants.py
- Hugging Face GGUF metadata dump discussion: https://huggingface.co/google/gemma-4-E2B-it/discussions/6
- LMCache Gemma4 KV caveats: https://docs.lmcache.ai/recipes/gemma4.html
- vLLM Gemma4 guide: https://docs.vllm.ai/projects/recipes/en/stable/Google/Gemma4.html

## Local Model Facts

Observed local model family:

- `X:\ai\models\llm\gemma-4-12B-coder-fable5-composer2.5-GGUF\gemma4-coding-Q4_K_M.gguf`
- `X:\ai\models\llm\gemma-4-12B-coder-fable5-composer2.5-GGUF\gemma4-coding-Q6_K.gguf`
- `X:\ai\models\llm\gemma-4-12B-coder-fable5-composer2.5-GGUF\gemma4-coding-Q8_0.gguf`

Key metadata from local inspection:

- `general.architecture = gemma4`
- `gemma4.context_length = 262144`
- `gemma4.embedding_length = 3840`
- `gemma4.block_count = 48`
- `gemma4.feed_forward_length = 15360`
- `gemma4.attention.head_count = 16`
- `gemma4.attention.head_count_kv` is an array in at least some variants
- `gemma4.attention.key_length = 512`
- `gemma4.attention.key_length_swa = 256`
- `gemma4.attention.value_length = 512`
- `gemma4.attention.value_length_swa = 256`
- `gemma4.rope.dimension_count = 512`
- `gemma4.rope.dimension_count_swa = 256`
- `gemma4.attention.sliding_window = 1024`
- `gemma4.attention.sliding_window_pattern` is a bool array
- `tokenizer.ggml.model = gemma4`
- `tokenizer.ggml.tokens` size is 262144
- `tokenizer.ggml.add_bos_token = false`

Representative tensor differences from Llama/Qwen:

- Local/SWA layers have Q/K/V projection widths different from full-attention layers.
- Some full/shared layers may omit normal K/V tensors and reuse KV from an earlier owner layer.
- Per-layer tensors include extra norms and scaling, such as `post_attention_norm`, `post_ffw_norm`, and `layer_output_scale.weight`.
- `attn_q_norm` / `attn_k_norm` are sized to the layer's head dimension, not one global model head dimension.

## Current xeno-rt Gaps

The current model path is centered on `LlamaModel`:

- `Runtime` stores `Arc<LlamaModel>`.
- `Session` calls `LlamaModel::forward_batch`, `forward_token`, `forward_batch_all_logits`, and state save/restore directly.
- `LlamaConfig` assumes scalar metadata for most dimensions.
- `KvCache` exposes one global `width()` for every layer.
- `PagedKvCache`, `QuantizedPagedKvCache`, `KeyQ4ValueQ8PagedKvCache`, and `AgentAdaptiveKvCache` all assume fixed per-layer width.
- Attention currently assumes one global `head_dim`, one global `kv_width`, and one RoPE frequency set for standard transformer layers.
- FFN currently assumes Llama/Qwen-style SiLU-SwiGLU, not Gemma4 `gelu_pytorch_tanh` gated MLP.
- Tokenizer supports generic BPE/GPT2-BPE/Piece but does not explicitly test Gemma4 tokenizer parity or `add_space_prefix = false`.
- There is no final logit softcapping pass.

## Design Decision

Phase 1 should be implemented inside the existing `LlamaModel` path as a new `ArchitectureFamily::Gemma4` plus Gemma-specific config and layer weights.

Reasoning:

- `Runtime`, `Session`, C API, Python API, CLI, and server are all hard-wired to `LlamaModel`.
- A broad trait-object model abstraction is the correct long-term design, but it adds migration risk before Gemma correctness is known.
- A contained Gemma4 branch preserves OpenAI API compatibility and minimizes blast radius.

Phase 2 should extract a `CausalModel` trait once Gemma4 text is green:

```rust
trait CausalModel {
    fn config(&self) -> &ModelConfigView;
    fn model_name(&self) -> &str;
    fn forward_token(...);
    fn forward_batch(...);
    fn forward_batch_all_logits(...);
    fn save_state(...);
    fn restore_state(...);
    fn clear_state(...);
}
```

Do not do that refactor first.

## Implementation Phases

### Phase 0: Metadata and Fixture Foundation

Add metadata helpers:

- `MetadataArray::as_bool_vec()`
- `metadata_usize_array_any(prefixes, suffix)`
- `metadata_i32_or_usize_array_any(prefixes, suffix)`
- `metadata_bool_array_any(prefixes, suffix)`
- scalar-or-array normalization helpers, because Gemma4 fields may be scalar in one model and array in another.

Add synthetic fixture support:

- bool arrays in `tests/common::MetadataValueSpec`
- i32 arrays in `tests/common::MetadataValueSpec`
- synthetic Gemma4 metadata fixtures with:
  - mixed sliding/full layer pattern
  - per-layer `head_count_kv`
  - per-layer `feed_forward_length`
  - full and SWA RoPE metadata

Acceptance:

- Existing GGUF parse tests pass.
- New metadata-array tests cover bool, i32, u32, f32.
- No model loading behavior changes for existing Llama/Qwen tests.

### Phase 1: Gemma4 Config and Tensor Loading

Add a Gemma4 layer config, either embedded in `LlamaConfig` or behind a new internal enum:

```rust
struct Gemma4LayerConfig {
    layer_index: usize,
    attention_kind: Gemma4AttentionKind, // SlidingWindow or Full
    q_head_count: usize,
    kv_head_count: usize,
    q_head_dim: usize,
    k_head_dim: usize,
    v_head_dim: usize,
    q_width: usize,
    k_width: usize,
    v_width: usize,
    attn_output_input_width: usize,
    ffn_width: usize,
    rope_dim: usize,
    rope_freq_base: f32,
    sliding_window: Option<usize>,
    kv_owner_layer: usize,
}
```

Rules:

- Derive actual widths from tensor metadata where possible. Treat GGUF tensor dimensions as source of truth when metadata arrays are ambiguous.
- Use `sliding_window_pattern[layer]` to select SWA vs full attention.
- For SWA layers, use `key_length_swa`, `value_length_swa`, `rope.dimension_count_swa`, and `rope.freq_base_swa`.
- For full layers, use `key_length`, `value_length`, `rope.dimension_count`, and `rope.freq_base`.
- If `attention.shared_kv_layers > 0`, resolve `kv_owner_layer` for layers that omit K/V tensors.
- Do not guess missing tensors silently. Return `XrtError::Unsupported` or `XrtError::InvalidTensor` with the layer index and expected tensor.

Add weights:

```rust
enum AttnWeights {
    ...
    Gemma4 {
        attn_q: ResolvedWeight,
        attn_k: Option<ResolvedWeight>,
        attn_v: Option<ResolvedWeight>,
        attn_output: ResolvedWeight,
        attn_q_norm: String,
        attn_k_norm: String,
        post_attention_norm: Option<String>,
    }
}

enum FfnWeights {
    ...
    Gemma4 {
        gate: ResolvedWeight,
        up: ResolvedWeight,
        down: ResolvedWeight,
        post_ffw_norm: Option<String>,
        layer_output_scale: Option<String>,
    }
}
```

Acceptance:

- `gemma-4-12b-coder-q4` loads metadata and all required tensors.
- If generation is not implemented yet, load should fail after tensor validation with a deliberate "Gemma4 forward not implemented" error, not generic architecture unsupported.

### Phase 2: Gemma4 Forward Pass, CPU Correctness First

Add Gemma4 single-token forward first. Batch prefill can initially fall back to token-by-token for Gemma4 until correctness is verified.

Forward skeleton:

1. Input embedding lookup.
2. For each layer:
   - `attn_input = rmsnorm(x, attn_norm)`
   - Project Q.
   - If this layer owns KV, project K and V; otherwise copy/read KV from `kv_owner_layer`.
   - Apply per-head Q/K RMSNorm using layer-specific head dimension.
   - Apply layer-specific RoPE:
     - SWA: `rope.freq_base_swa`, `rope.dimension_count_swa`
     - Full: `rope.freq_base`, `rope.dimension_count`
   - Append owned KV to cache.
   - Compute attention:
     - Full layers attend to all cached positions.
     - SWA layers attend only `[max(0, position + 1 - sliding_window), position]`.
   - Project attention output.
   - Apply optional `post_attention_norm`.
   - Residual add.
   - `ffn_input = rmsnorm(x, ffn_norm)`
   - Apply gated MLP using `gelu_pytorch_tanh(gate) * up`, not SiLU-SwiGLU.
   - Project `down`.
   - Apply optional `post_ffw_norm`.
   - Residual add.
   - Apply optional `layer_output_scale`.
3. Final `output_norm`.
4. Output projection.
5. Apply final logit softcap:

```rust
for logit in logits {
    *logit = softcap * (*logit / softcap).tanh();
}
```

Implementation notes:

- Validate the exact order of `post_attention_norm`, `post_ffw_norm`, and `layer_output_scale` against upstream llama.cpp or Hugging Face before locking tests.
- Add `gelu_pytorch_tanh` to `xrt-kernels::cpu`.
- Current `RopeFreqs` stores one frequency set. Gemma4 needs two frequency sets or per-layer cached RoPE state.
- Current scratch buffers are sized from one global head dimension. Gemma4 needs max-sized buffers plus per-layer slices.

Acceptance:

- `cargo run -q -p xrt-cli -- generate --model gemma-4-12b-coder-q4 --prompt "Hello" --max-tokens 1 --seed 1` completes without panic.
- Existing `vibethinker-3b` generation still works.
- Existing Llama/Qwen synthetic tests pass.

### Phase 3: KV Cache Correctness

MVP option:

- Keep current fixed-width `KvCache` and allocate using the maximum Gemma4 KV width across layers.
- Append/read only the layer's active `k_width` / `v_width` slice and leave padding zeroed.
- This is acceptable for first correctness on short prompts but is not acceptable for long-context performance.

Production option:

- Add per-layer cache widths:

```rust
trait KvCache {
    fn width(&self) -> usize; // keep for existing models
    fn key_width(&self, layer: usize) -> usize { self.width() }
    fn value_width(&self, layer: usize) -> usize { self.width() }
}
```

- Add a `VariablePagedKvCache` or adapt existing page structs to store `key_widths[layer]` and `value_widths[layer]`.
- Agent-adaptive and quantized caches must support per-layer width before enabling long-context Gemma4.
- Shared-KV layers must not duplicate cache storage. They should read from the owner layer and keep lengths aligned.

Recommendation:

- Use fixed max-width padding only for Phase 2.
- Move to variable-width KV before claiming long-context or performance support.

Acceptance:

- SWA layers produce identical results whether evaluated one token at a time or via fallback prefill.
- Shared-KV layers reuse owner-layer cache and do not append duplicate KV.
- Cache truncate/rollback used by speculative decoding remains safe, or speculation is disabled for Gemma4 until variable-width/shared-KV rollback is tested.

### Phase 4: Tokenizer and Chat Template

Add explicit Gemma4 tokenizer coverage:

- Treat `tokenizer.ggml.model = gemma4` as a supported BPE tokenizer mode.
- Respect `tokenizer.ggml.add_bos_token = false`.
- Respect `tokenizer.ggml.add_space_prefix = false` if current generic BPE behavior differs.
- Ensure special pieces and tool-call tokens are recognized as indivisible tokens.
- Add fixture tests for Gemma4-like BPE merges and special token matching.

Chat/API behavior:

- Keep `/v1/chat/completions` OpenAI-compatible.
- Use GGUF `tokenizer.chat_template` for prompt formatting.
- Do not add Gemma4-specific response schemas to `/v1/models`.
- Later, add optional reasoning/tool-call parsing compatible with Gemma4 templates, but keep raw model output available.

Acceptance:

- `Tokenizer::from_gguf` accepts `tokenizer.ggml.model = gemma4`.
- A minimal Gemma4 chat template fixture formats and tokenizes without fallback to ChatML.
- Existing Qwen tool-call behavior is unchanged.

### Phase 5: Batch Prefill and Performance

Once token-by-token Gemma4 is correct:

- Implement Gemma4 batch prefill for local and full layers.
- Batch attention must support per-layer attention window start.
- Batch RoPE must use per-layer frequency tables.
- Avoid allocating `Vec` per head and per position inside attention.
- Add release-mode CPU benchmark for `gemma-4-12b-coder-q4`.

Performance gates:

- No regression over current Qwen2/VibeThinker smoke benchmarks by more than 10%.
- Gemma4 debug-mode correctness is not a performance signal.
- Release-mode CPU path should produce tokens; CUDA optimization is not required for first merge.

### Phase 6: Multimodal and Assistant Variants

Do not block text support on multimodal.

Later scope:

- Image/audio/video token ingestion for unified encoder-free Gemma4 12B.
- Gemma4 E2B/E4B external audio/vision modules if required by their GGUF/mmproj format.
- Assistant/drafter MTP models.
- Centroid/sparse logit projection for assistant checkpoints if relevant.
- Gemma4 reasoning/tool-call parser in server responses.

### Phase 7: TurboQuant / KV Compression

Gemma4 is a strong candidate for KV compression, but only after native correctness.

Requirements before TurboQuant:

- Variable-width KV cache by layer.
- Explicit SWA/full cache groups.
- Shared-KV ownership model.
- Layer-specific head dimensions.
- Quantized cache read/write parity tests.

Do not implement TurboQuant on top of the current fixed-width cache; it will lock in the wrong abstraction.

## Test Matrix

Required unit tests:

- GGUF bool/i32 array metadata parsing.
- Gemma4 scalar-or-array metadata normalization.
- Gemma4 layer config derivation for:
  - all local layers
  - 5:1 local/full pattern
  - shared-KV tail layers
  - per-layer KV head-count arrays
- `gelu_pytorch_tanh` numerical snapshots.
- final logit softcap numerical snapshots.
- SWA attention mask window boundaries.

Required integration tests:

- Synthetic tiny Gemma4 GGUF loads.
- Synthetic tiny Gemma4 single-token forward returns finite logits.
- Existing `qwen2`, `qwen3`, `qwen3_next`, omni unsupported, GLM unsupported, and Gemma4 unsupported tests updated to new behavior.
- Local smoke:

```powershell
cargo run -q -p xrt-cli -- generate --model gemma-4-12b-coder-q4 --prompt "Hello" --max-tokens 1 --seed 1
```

Regression tests:

```powershell
cargo check -p xrt-gguf -p xrt-kernels -p xrt-models -p xrt-runtime -p xrt-cli -p xrt-server
cargo test -p xrt-hub
cargo test -p xrt-workspace-tests --test model_architecture_test --test gguf_parse_test
cargo run -q -p xrt-cli -- generate --model vibethinker-3b-q4 --prompt "Hello" --max-tokens 1 --seed 1
```

Optional golden tests:

- Compare first-token top-10 logits against llama.cpp or Hugging Face Transformers for a tiny deterministic prompt.
- Tolerance should be loose for quantized GGUF, but top-1/top-5 should match on low-temperature deterministic prompts.

## Risks

- Tensor naming may differ between official GGUFs and local finetunes. Mitigation: tensor dimensions are source of truth, and errors must include exact missing tensor names.
- Tokenizer parity may be imperfect. Mitigation: add Gemma4-specific tokenizer fixtures before judging model quality.
- Fixed-width KV padding will waste memory. Mitigation: use it only for correctness MVP, then implement variable-width cache.
- Shared-KV can break cache reuse and speculative rollback. Mitigation: disable self-speculative paths for Gemma4 until rollback is tested.
- Multimodal token paths may require new request-shaping and preprocessing. Mitigation: text-only first; do not fake multimodal support.

## Acceptance Criteria For First Merge

- `gemma-4-12b-coder-q4` loads and generates at least one token on CPU.
- `gemma-4-12b-coder`, `gemma-4-12b-coder-q4`, and `gemma-4-12b-coder-q8` aliases remain valid.
- Existing Qwen2/VibeThinker generation still works.
- Existing OpenAI-compatible endpoint schemas are unchanged.
- CPU fallback works without CUDA.
- Unsupported Gemma4 multimodal/audio/assistant paths fail explicitly, not as generic architecture errors.
- The implementation has a clear follow-up path for variable-width KV cache and TurboQuant.
