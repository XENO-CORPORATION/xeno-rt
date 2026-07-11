use crate::{
    BackendSession, CachePolicyKind, KvCacheMode, PromptSpan, Runtime, Sampler, SamplerConfig,
    SessionPolicy,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};
use xrt_core::{checked_mul, Result, XrtError};

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
    backend_session: BackendSession,
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
        let page_tokens = 32;
        let backend_session = runtime
            .backend()
            .new_session(default_cache_mode, page_tokens);
        runtime.register_session();
        Self {
            runtime,
            default_cache_mode,
            page_tokens,
            backend_session,
            sampler: Sampler::new(None),
            tokens: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.backend_session.clear();
        self.tokens.clear();
        self.runtime.backend().clear_state();
    }

    pub fn gpu_resource_status(&self) -> crate::GpuResourceStatus {
        self.runtime.gpu_resource_status_with_session_allocations(
            self.backend_session.cuda_kv_allocated_bytes(),
            self.backend_session.cuda_scratch_allocated_bytes(),
            Some(self.backend_session.requested_cache_mode()),
            Some(self.backend_session.cache_mode()),
            self.backend_session.cuda_graph_capture_status(),
        )
    }

    pub fn generate(&mut self, request: &GenerateRequest) -> Result<String> {
        let mut output = String::new();
        self.generate_stream(request, |piece| output.push_str(piece))?;
        Ok(output)
    }

    pub fn generate_stream<F>(
        &mut self,
        request: &GenerateRequest,
        mut on_token: F,
    ) -> Result<usize>
    where
        F: FnMut(&str),
    {
        self.reset();
        self.sampler.reseed(request.seed);

        let runtime = self.runtime.clone();
        let tokenizer = runtime.tokenizer();
        let backend = runtime.backend();
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
        if prompt_tokens.len() > backend.config().context_length {
            return Err(XrtError::Runtime(format!(
                "prompt length {} exceeds model context length {}",
                prompt_tokens.len(),
                backend.config().context_length
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
        self.backend_session.configure_policy(
            session_policy,
            prompt_tokens.len(),
            &request.prompt_spans,
        );
        let graph_total_len = prompt_tokens
            .len()
            .checked_add(request.max_tokens)
            .ok_or_else(|| XrtError::Runtime("generation length overflow".to_string()))?
            .min(backend.config().context_length);
        let graph_capacity_prepared = backend.supports_cuda_graph_decode()
            && self
                .backend_session
                .prepare_cuda_graph_generation_capacity(graph_total_len);
        if !graph_capacity_prepared {
            self.backend_session
                .prepare_for_total_len(prompt_tokens.len())?;
        }

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
            backend.forward_batch_with_embeddings(
                &prompt_tokens,
                0,
                &mut self.backend_session,
                overrides,
            )?
        } else {
            backend.forward_batch(&prompt_tokens, 0, &mut self.backend_session)?
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
        let ctx_len = backend.config().context_length;
        let vocab_size = backend.config().vocab_size;
        let is_hybrid = backend.config().is_hybrid();
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
                self.backend_session
                    .prepare_for_total_len(self.tokens.len())?;
                backend.forward_token(
                    next,
                    self.tokens.len() - 1,
                    &mut self.backend_session,
                    &mut logits,
                )?;
            } else if !is_hybrid {
                // Standard transformer speculation: batched forward + KV cache rollback
                let mut batch_tokens = Vec::with_capacity(1 + draft.len());
                batch_tokens.push(next);
                batch_tokens.extend_from_slice(&draft);

                let start_pos = self.tokens.len() - 1;
                self.backend_session
                    .prepare_for_total_len(total_len_after_batch(start_pos, batch_tokens.len())?)?;
                let all_logits = backend.forward_batch_all_logits(
                    &batch_tokens,
                    start_pos,
                    &mut self.backend_session,
                )?;

                // Verify draft tokens greedily (argmax)
                let mut accepted = 0;
                for i in 0..draft.len() {
                    let pos_logits = logits_for_position(&all_logits, i, vocab_size)?;
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
                self.backend_session.truncate(self.tokens.len());

                // Use logits from the last accepted position
                let last_logit_idx = accepted;
                logits.resize(vocab_size, 0.0);
                logits.copy_from_slice(logits_for_position(
                    &all_logits,
                    last_logit_idx,
                    vocab_size,
                )?);

                if generated >= request.max_tokens || self.tokens.len() >= ctx_len {
                    break;
                }
                if accepted > 0 && Some(self.tokens[self.tokens.len() - 1]) == eos {
                    break;
                }
            } else {
                // Hybrid model speculation: save DeltaNet state, verify, restore on rejection
                let state_snapshot = backend.save_state();
                let cache_len_before = self.tokens.len() - 1; // before `next` was processed

                // Process `next` + draft tokens sequentially through the model
                // (forward_batch_all_logits already handles this for hybrid models)
                let mut batch_tokens = Vec::with_capacity(1 + draft.len());
                batch_tokens.push(next);
                batch_tokens.extend_from_slice(&draft);

                let start_pos = self.tokens.len() - 1;
                self.backend_session
                    .prepare_for_total_len(total_len_after_batch(start_pos, batch_tokens.len())?)?;
                let all_logits = backend.forward_batch_all_logits(
                    &batch_tokens,
                    start_pos,
                    &mut self.backend_session,
                )?;

                // Verify draft tokens
                let mut accepted = 0;
                for i in 0..draft.len() {
                    let pos_logits = logits_for_position(&all_logits, i, vocab_size)?;
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
                        backend.restore_state(snap);
                    }
                    // Truncate KV cache to before any speculation started
                    self.backend_session.truncate(cache_len_before);
                    // Replay only the kept tokens through the model
                    let replay_tokens = &batch_tokens[..total_kept];
                    self.backend_session
                        .prepare_for_total_len(total_len_after_batch(
                            start_pos,
                            replay_tokens.len(),
                        )?)?;
                    let replay_logits = backend.forward_batch_all_logits(
                        replay_tokens,
                        start_pos,
                        &mut self.backend_session,
                    )?;
                    let last_idx = total_kept - 1;
                    logits.resize(vocab_size, 0.0);
                    logits.copy_from_slice(logits_for_position(
                        &replay_logits,
                        last_idx,
                        vocab_size,
                    )?);
                } else {
                    // All draft tokens accepted — state is correct as-is
                    let last_idx = total_processed - 1;
                    logits.resize(vocab_size, 0.0);
                    logits.copy_from_slice(logits_for_position(&all_logits, last_idx, vocab_size)?);
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

        Ok(generated)
    }

    fn ensure_cache_mode(&mut self, mode: KvCacheMode) {
        if self.backend_session.requested_cache_mode() == mode {
            return;
        }
        let config = self.runtime.backend().config();
        if let Some(layer_widths) = config.gemma4_layer_kv_widths() {
            self.backend_session.replace_cache_with_layer_widths(
                mode,
                layer_widths,
                self.page_tokens,
            );
        } else {
            self.backend_session.replace_cache(
                mode,
                config.block_count,
                config.kv_width(),
                self.page_tokens,
            );
        }
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

impl Drop for Session {
    fn drop(&mut self) {
        self.runtime.unregister_session();
    }
}

fn total_len_after_batch(start_pos: usize, batch_len: usize) -> Result<usize> {
    start_pos
        .checked_add(batch_len)
        .ok_or_else(|| XrtError::Runtime("batch length overflow".to_string()))
}

fn logits_for_position(logits: &[f32], index: usize, vocab_size: usize) -> Result<&[f32]> {
    let start = index
        .checked_mul(vocab_size)
        .ok_or_else(|| XrtError::Runtime("logit offset overflow".to_string()))?;
    let end = start
        .checked_add(vocab_size)
        .ok_or_else(|| XrtError::Runtime("logit range overflow".to_string()))?;
    logits.get(start..end).ok_or_else(|| {
        XrtError::Runtime(format!(
            "backend returned {} logits, missing position {index} with vocab size {vocab_size}",
            logits.len()
        ))
    })
}

fn checked_add(lhs: usize, rhs: usize, what: &str) -> Result<usize> {
    lhs.checked_add(rhs)
        .ok_or_else(|| XrtError::Runtime(format!("{what} overflow")))
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
    let expected_patch_tokens = checked_mul(
        images.len(),
        layout.patches_per_image,
        "image patch token count",
    )?;
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

    let embedding_dim = runtime.backend().config().embedding_length;
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
        let expected_len = checked_mul(
            layout.patches_per_image,
            embedding_dim,
            "vision embedding output length",
        )?;
        if embeddings.len() != expected_len {
            return Err(XrtError::Runtime(format!(
                "vision encoder returned {} floats, expected {} for {} patches x {} dim",
                embeddings.len(),
                expected_len,
                layout.patches_per_image,
                embedding_dim
            )));
        }

        let patch_offset = checked_mul(
            image_index,
            layout.patches_per_image,
            "image patch position offset",
        )?;
        for patch_index in 0..layout.patches_per_image {
            let src_start = checked_mul(patch_index, embedding_dim, "image embedding row offset")?;
            let src_end = checked_add(src_start, embedding_dim, "image embedding row end")?;
            let dst_index = checked_add(patch_offset, patch_index, "image patch position index")?;
            overrides.insert(
                patch_positions[dst_index],
                embeddings[src_start..src_end].to_vec(),
            );
        }
    }

    Ok(overrides)
}

#[cfg(test)]
mod tests {
    use super::{argmax, checked_add, logits_for_position, total_len_after_batch};

    #[test]
    fn argmax_returns_first_maximum() {
        let values = [0.25, 1.0, 0.5, 1.0];
        assert_eq!(argmax(&values), 1);
    }

    #[test]
    fn total_len_after_batch_checks_overflow() {
        assert_eq!(total_len_after_batch(4, 3).unwrap(), 7);
        assert!(total_len_after_batch(usize::MAX, 1).is_err());
    }

    #[test]
    fn logits_for_position_checks_bounds() {
        let logits = [0.0, 1.0, 2.0, 3.0];
        assert_eq!(logits_for_position(&logits, 1, 2).unwrap(), &[2.0, 3.0]);
        assert!(logits_for_position(&logits, usize::MAX, 2).is_err());
        assert!(logits_for_position(&logits, 2, 2).is_err());
    }

    #[test]
    fn checked_add_reports_overflow() {
        assert_eq!(checked_add(2, 3, "test").unwrap(), 5);
        assert!(checked_add(usize::MAX, 1, "test").is_err());
    }
}
