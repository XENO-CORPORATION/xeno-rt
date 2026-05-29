use crate::{
    CachePolicyKind, KvCacheMode, PromptSpan, Runtime, Sampler, SamplerConfig, SessionKvCache,
    SessionPolicy,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};
use xrt_core::{KvCache, Result, XrtError};

/// N-gram order for prompt lookup decoding.
const NGRAM_ORDER: usize = 3;

/// Maximum number of draft tokens per speculation round.
const MAX_DRAFT: usize = 5;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,
    pub add_special_tokens: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_policy: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recent_window_tokens: Option<usize>,
    #[serde(default)]
    pub prompt_spans: Vec<PromptSpan>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub seed: Option<u64>,
    /// Optional image data for multimodal models.
    /// Each image is a flat f32 array in CHW layout, shape [3, image_size, image_size],
    /// with pixel values normalized to [-1, 1].
    /// The prompt must already contain the model's expanded image patch placeholder sequence.
    /// The OpenAI-compatible server builds that sequence automatically for multipart image inputs.
    #[serde(skip)]
    pub images: Vec<Vec<f32>>,
}

impl Default for GenerateRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            add_special_tokens: true,
            cache_policy: None,
            recent_window_tokens: None,
            prompt_spans: Vec::new(),
            max_tokens: 128,
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            repetition_penalty: 1.1,
            seed: None,
            images: Vec::new(),
        }
    }
}

pub struct Session {
    runtime: Arc<Runtime>,
    default_cache_mode: KvCacheMode,
    page_tokens: usize,
    cache: SessionKvCache,
    sampler: Sampler,
    tokens: Vec<u32>,
}

impl Session {
    pub(crate) fn new(runtime: Arc<Runtime>) -> Self {
        Self::new_with_cache_mode(runtime, KvCacheMode::from_env())
    }

    pub(crate) fn new_with_cache_mode(
        runtime: Arc<Runtime>,
        default_cache_mode: KvCacheMode,
    ) -> Self {
        let config = runtime.model().config();
        let block_count = config.block_count;
        let kv_width = config.kv_width();
        let page_tokens = 32;
        Self {
            runtime,
            default_cache_mode,
            page_tokens,
            cache: SessionKvCache::new(default_cache_mode, block_count, kv_width, page_tokens),
            sampler: Sampler::new(None),
            tokens: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.cache.clear();
        self.tokens.clear();
        self.runtime.model().clear_state();
    }

    pub fn generate(&mut self, request: &GenerateRequest) -> Result<String> {
        let mut output = String::new();
        self.generate_stream(request, |piece| output.push_str(piece))?;
        Ok(output)
    }

    pub fn generate_stream<F>(&mut self, request: &GenerateRequest, mut on_token: F) -> Result<()>
    where
        F: FnMut(&str),
    {
        self.reset();
        self.sampler.reseed(request.seed);

        let runtime = self.runtime.clone();
        let tokenizer = runtime.tokenizer();
        let mut prompt_tokens =
            tokenizer.encode_with_options(&request.prompt, request.add_special_tokens, true)?;
        if prompt_tokens.is_empty() {
            if let Some(bos) = tokenizer.special_tokens().bos {
                prompt_tokens.push(bos);
            } else {
                return Err(XrtError::Runtime(
                    "empty prompt and tokenizer has no BOS token".to_string(),
                ));
            }
        }
        if prompt_tokens.len() > runtime.model().config().context_length {
            return Err(XrtError::Runtime(format!(
                "prompt length {} exceeds model context length {}",
                prompt_tokens.len(),
                self.runtime.model().config().context_length
            )));
        }

        let requested_policy = request
            .cache_policy
            .as_deref()
            .and_then(CachePolicyKind::parse);
        let effective_mode =
            if requested_policy.is_some_and(CachePolicyKind::requires_adaptive_cache) {
                KvCacheMode::AgentAdaptive
            } else {
                self.default_cache_mode
            };
        self.ensure_cache_mode(effective_mode);
        let default_policy = if effective_mode == KvCacheMode::AgentAdaptive {
            CachePolicyKind::AgentAdaptive
        } else {
            CachePolicyKind::DefaultChat
        };
        let session_policy = SessionPolicy::from_request(
            request.cache_policy.as_deref(),
            request.recent_window_tokens,
            default_policy,
        );
        self.cache
            .configure_policy(session_policy, prompt_tokens.len(), &request.prompt_spans);
        self.cache.prepare_for_total_len(prompt_tokens.len())?;

        let embedding_overrides = if request.images.is_empty() {
            None
        } else {
            Some(build_image_embedding_overrides(
                &runtime,
                &prompt_tokens,
                &request.images,
            )?)
        };

        // Batch prefill: process all prompt tokens in a single forward pass.
        let mut logits = if let Some(overrides) = embedding_overrides {
            runtime.model().forward_batch_with_embeddings(
                &prompt_tokens,
                0,
                &mut self.cache,
                overrides,
            )?
        } else {
            runtime
                .model()
                .forward_batch(&prompt_tokens, 0, &mut self.cache)?
        };
        self.tokens.extend_from_slice(&prompt_tokens);

        let sampler_config = SamplerConfig {
            temperature: request.temperature,
            top_k: request.top_k,
            top_p: request.top_p,
            repetition_penalty: request.repetition_penalty,
            seed: request.seed,
        };

        let eos = tokenizer.special_tokens().eos;
        let ctx_len = runtime.model().config().context_length;
        let vocab_size = runtime.model().config().vocab_size;
        let is_hybrid = runtime.model().config().is_hybrid();
        let mut generated = 0usize;
        let mut pending_decode_tokens = Vec::new();

        let mut emit_token = |token: u32, force_flush: bool| -> Result<()> {
            pending_decode_tokens.push(token);
            match tokenizer.decode(&pending_decode_tokens, true) {
                Ok(piece) => {
                    if !piece.is_empty() {
                        on_token(&piece);
                    }
                    pending_decode_tokens.clear();
                    Ok(())
                }
                Err(XrtError::Tokenizer(message))
                    if message.contains("invalid utf8 in decode") && !force_flush =>
                {
                    Ok(())
                }
                Err(XrtError::Tokenizer(message)) if message.contains("invalid utf8 in decode") => {
                    let piece = tokenizer.decode_lossy(&pending_decode_tokens, true)?;
                    if !piece.is_empty() {
                        on_token(&piece);
                    }
                    pending_decode_tokens.clear();
                    Ok(())
                }
                Err(err) => Err(err),
            }
        };

        while generated < request.max_tokens {
            let next = self.sampler.sample(&logits, &self.tokens, sampler_config)?;
            if Some(next) == eos {
                break;
            }
            if self.tokens.len() >= ctx_len {
                break;
            }

            self.tokens.push(next);
            generated += 1;
            emit_token(next, false)?;

            let remaining = request.max_tokens - generated;
            if remaining == 0 {
                break;
            }

            // Try n-gram draft (free — no model calls, just pattern matching in token history)
            // Disabled for hybrid models: state save/restore is too expensive (~19MB per checkpoint)
            let draft = if is_hybrid {
                Vec::new()
            } else {
                self.ngram_draft(remaining)
            };

            if draft.is_empty() {
                // No speculation: standard single-token decode
                self.cache.prepare_for_total_len(self.tokens.len())?;
                runtime.model().forward_token(
                    next,
                    self.tokens.len() - 1,
                    &mut self.cache,
                    &mut logits,
                )?;
            } else if !is_hybrid {
                // Standard transformer speculation: batched forward + KV cache rollback
                let mut batch_tokens = Vec::with_capacity(1 + draft.len());
                batch_tokens.push(next);
                batch_tokens.extend_from_slice(&draft);

                let start_pos = self.tokens.len() - 1;
                self.cache
                    .prepare_for_total_len(start_pos + batch_tokens.len())?;
                let all_logits = runtime.model().forward_batch_all_logits(
                    &batch_tokens,
                    start_pos,
                    &mut self.cache,
                )?;

                // Verify draft tokens greedily (argmax)
                let mut accepted = 0;
                for i in 0..draft.len() {
                    let pos_logits = &all_logits[i * vocab_size..(i + 1) * vocab_size];
                    let predicted = argmax(pos_logits);
                    if predicted == draft[i] {
                        accepted += 1;
                        self.tokens.push(draft[i]);
                        generated += 1;
                        emit_token(draft[i], false)?;
                        if Some(draft[i]) == eos
                            || self.tokens.len() >= ctx_len
                            || generated >= request.max_tokens
                        {
                            break;
                        }
                    } else {
                        break;
                    }
                }

                // Roll back KV cache for rejected draft tokens
                self.cache.truncate(self.tokens.len());

                // Use logits from the last accepted position
                let last_logit_idx = accepted;
                logits.resize(vocab_size, 0.0);
                logits.copy_from_slice(
                    &all_logits[last_logit_idx * vocab_size..(last_logit_idx + 1) * vocab_size],
                );

                if generated >= request.max_tokens || self.tokens.len() >= ctx_len {
                    break;
                }
                if accepted > 0 && Some(self.tokens[self.tokens.len() - 1]) == eos {
                    break;
                }
            } else {
                // Hybrid model speculation: save DeltaNet state, verify, restore on rejection
                let state_snapshot = runtime.model().save_state();
                let cache_len_before = self.tokens.len() - 1; // before `next` was processed

                // Process `next` + draft tokens sequentially through the model
                // (forward_batch_all_logits already handles this for hybrid models)
                let mut batch_tokens = Vec::with_capacity(1 + draft.len());
                batch_tokens.push(next);
                batch_tokens.extend_from_slice(&draft);

                let start_pos = self.tokens.len() - 1;
                self.cache
                    .prepare_for_total_len(start_pos + batch_tokens.len())?;
                let all_logits = runtime.model().forward_batch_all_logits(
                    &batch_tokens,
                    start_pos,
                    &mut self.cache,
                )?;

                // Verify draft tokens
                let mut accepted = 0;
                for i in 0..draft.len() {
                    let pos_logits = &all_logits[i * vocab_size..(i + 1) * vocab_size];
                    let predicted = argmax(pos_logits);
                    if predicted == draft[i] {
                        accepted += 1;
                        self.tokens.push(draft[i]);
                        generated += 1;
                        emit_token(draft[i], false)?;
                        if Some(draft[i]) == eos
                            || self.tokens.len() >= ctx_len
                            || generated >= request.max_tokens
                        {
                            break;
                        }
                    } else {
                        break;
                    }
                }

                let total_processed = 1 + draft.len(); // next + all draft tokens
                let total_kept = 1 + accepted; // next + accepted draft tokens

                if total_kept < total_processed {
                    // Some tokens were rejected — roll back both DeltaNet state and KV cache
                    if let Some(ref snap) = state_snapshot {
                        runtime.model().restore_state(snap);
                    }
                    // Truncate KV cache to before any speculation started
                    self.cache.truncate(cache_len_before);
                    // Replay only the kept tokens through the model
                    let replay_tokens = &batch_tokens[..total_kept];
                    self.cache
                        .prepare_for_total_len(start_pos + replay_tokens.len())?;
                    let replay_logits = runtime.model().forward_batch_all_logits(
                        replay_tokens,
                        start_pos,
                        &mut self.cache,
                    )?;
                    let last_idx = total_kept - 1;
                    logits.resize(vocab_size, 0.0);
                    logits.copy_from_slice(
                        &replay_logits[last_idx * vocab_size..(last_idx + 1) * vocab_size],
                    );
                } else {
                    // All draft tokens accepted — state is correct as-is
                    let last_idx = total_processed - 1;
                    logits.resize(vocab_size, 0.0);
                    logits.copy_from_slice(
                        &all_logits[last_idx * vocab_size..(last_idx + 1) * vocab_size],
                    );
                }

                if generated >= request.max_tokens || self.tokens.len() >= ctx_len {
                    break;
                }
                if accepted > 0 && Some(self.tokens[self.tokens.len() - 1]) == eos {
                    break;
                }
            }
        }

        if !pending_decode_tokens.is_empty() {
            let piece = tokenizer.decode_lossy(&pending_decode_tokens, true)?;
            if !piece.is_empty() {
                on_token(&piece);
            }
        }

        Ok(())
    }

    fn ensure_cache_mode(&mut self, mode: KvCacheMode) {
        if self.cache.mode() == mode {
            return;
        }
        let config = self.runtime.model().config();
        self.cache = SessionKvCache::new(
            mode,
            config.block_count,
            config.kv_width(),
            self.page_tokens,
        );
    }

    /// Search for an n-gram match in the token history and return draft continuation tokens.
    fn ngram_draft(&self, max_tokens: usize) -> Vec<u32> {
        let n = NGRAM_ORDER;
        let tokens = &self.tokens;
        if tokens.len() < n + 1 {
            return Vec::new();
        }

        let max_draft = MAX_DRAFT.min(max_tokens);
        if max_draft == 0 {
            return Vec::new();
        }

        let needle = &tokens[tokens.len() - n..];
        let search_end = tokens.len() - n;

        for start in (0..search_end).rev() {
            if start + n > search_end {
                continue;
            }
            if tokens[start..start + n] == *needle {
                let continuation_start = start + n;
                let draft_len = max_draft.min(tokens.len() - continuation_start);
                if draft_len > 0 {
                    return tokens[continuation_start..continuation_start + draft_len].to_vec();
                }
            }
        }

        Vec::new()
    }
}

fn argmax(values: &[f32]) -> u32 {
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

fn build_image_embedding_overrides(
    runtime: &Runtime,
    prompt_tokens: &[u32],
    images: &[Vec<f32>],
) -> Result<HashMap<usize, Vec<f32>>> {
    let vision = runtime.vision().ok_or_else(|| {
        XrtError::Runtime(
            "image inputs require a loaded multimodal projection; load the model with mmproj first"
                .to_string(),
        )
    })?;
    let layout = runtime.vision_prompt_layout().ok_or_else(|| {
        XrtError::Runtime(
            "image inputs require tokenizer support for image placeholder tokens".to_string(),
        )
    })?;
    let expected_patch_tokens = images.len() * layout.patches_per_image;
    let patch_positions = prompt_tokens
        .iter()
        .enumerate()
        .filter_map(|(index, &token)| (token == layout.patch_token_id).then_some(index))
        .collect::<Vec<_>>();

    if patch_positions.len() != expected_patch_tokens {
        return Err(XrtError::Runtime(format!(
            "prompt contains {} image patch tokens, but {} image(s) require {}; ensure the prompt uses the runtime's expanded image placeholder sequence",
            patch_positions.len(),
            images.len(),
            expected_patch_tokens
        )));
    }

    let embedding_dim = runtime.model().config().embedding_length;
    if vision.config().projection_dim != embedding_dim {
        return Err(XrtError::Runtime(format!(
            "vision projection dim {} does not match model embedding dim {}",
            vision.config().projection_dim,
            embedding_dim
        )));
    }

    let mut overrides = HashMap::with_capacity(expected_patch_tokens);
    for (image_index, image) in images.iter().enumerate() {
        let embeddings = vision.encode(image)?;
        let expected_len = layout.patches_per_image * embedding_dim;
        if embeddings.len() != expected_len {
            return Err(XrtError::Runtime(format!(
                "vision encoder returned {} floats, expected {} for {} patches x {} dim",
                embeddings.len(),
                expected_len,
                layout.patches_per_image,
                embedding_dim
            )));
        }

        let patch_offset = image_index * layout.patches_per_image;
        for patch_index in 0..layout.patches_per_image {
            let src_start = patch_index * embedding_dim;
            let src_end = src_start + embedding_dim;
            overrides.insert(
                patch_positions[patch_offset + patch_index],
                embeddings[src_start..src_end].to_vec(),
            );
        }
    }

    Ok(overrides)
}

#[cfg(test)]
mod tests {
    use super::argmax;

    #[test]
    fn argmax_returns_first_maximum() {
        let values = [0.25, 1.0, 0.5, 1.0];
        assert_eq!(argmax(&values), 1);
    }
}
