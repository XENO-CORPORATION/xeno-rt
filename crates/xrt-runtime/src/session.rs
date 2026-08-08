use crate::{
    prefix_cache::PrefixCacheRequest, BackendSession, CachePolicyKind, KvCacheMode, PromptSpan,
    RequestScheduler, Runtime, Sampler, SamplerConfig, SchedulerExecutionPhase, SessionPolicy,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    env,
    ops::ControlFlow,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};
use xrt_core::{checked_mul, Result, XrtError};

/// N-gram order for prompt lookup decoding.
const NGRAM_ORDER: usize = 3;

/// Maximum number of draft tokens per speculation round.
const MAX_DRAFT: usize = 5;

static NEXT_DECODE_SEQUENCE_ID: AtomicU64 = AtomicU64::new(1);

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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeculativeDecodeStats {
    pub verification_batches: u64,
    pub drafted_tokens: u64,
    pub accepted_tokens: u64,
    pub rejected_tokens: u64,
    pub rollback_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HybridRuntimeStatus {
    pub owner: &'static str,
    pub backend: String,
    pub state_format_version: u32,
    pub recurrent_layers: usize,
    pub full_attention_layers: usize,
    pub durable_snapshot_bytes: u64,
    pub bytes_per_session: u64,
    pub prefix_cache_supported: bool,
    pub prefix_cache_enabled: bool,
    pub shared_f32_kv_page_cow_supported: bool,
    pub quantized_kv_page_cow_supported: bool,
    pub speculative_rollback_supported: bool,
    pub speculative_decoding_enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speculative_decoding_disabled_reason: Option<String>,
}

pub struct Session {
    runtime: Arc<Runtime>,
    default_cache_mode: KvCacheMode,
    page_tokens: usize,
    decode_sequence_id: u64,
    backend_session: Option<BackendSession>,
    sampler: Sampler,
    tokens: Vec<u32>,
    ngram_speculation_enabled: bool,
    mtp_speculation_enabled: bool,
    speculative_stats: SpeculativeDecodeStats,
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
            decode_sequence_id: NEXT_DECODE_SEQUENCE_ID.fetch_add(1, Ordering::Relaxed),
            backend_session: Some(backend_session),
            sampler: Sampler::new(None),
            tokens: Vec::new(),
            ngram_speculation_enabled: ngram_speculation_enabled_from_env(),
            mtp_speculation_enabled: mtp_speculation_enabled_from_env(),
            speculative_stats: SpeculativeDecodeStats::default(),
        }
    }

    fn backend_session(&self) -> &BackendSession {
        self.backend_session
            .as_ref()
            .expect("backend session is only absent during a synchronous decode rendezvous")
    }

    fn backend_session_mut(&mut self) -> &mut BackendSession {
        self.backend_session
            .as_mut()
            .expect("backend session is only absent during a synchronous decode rendezvous")
    }

    pub fn reset(&mut self) {
        self.backend_session_mut().clear();
        self.tokens.clear();
        self.speculative_stats = SpeculativeDecodeStats::default();
    }

    pub fn speculative_decode_stats(&self) -> SpeculativeDecodeStats {
        self.speculative_stats
    }

    /// Enables or disables prompt lookup (n-gram) speculation for this session.
    ///
    /// Sessions default to `XRT_NGRAM_SPECULATION` (`on` when unset). The
    /// per-session override is useful for deterministic comparisons and lets a
    /// caller stop drafting without changing the model, KV, or API contract.
    pub fn set_ngram_speculation_enabled(&mut self, enabled: bool) {
        self.ngram_speculation_enabled = enabled;
    }

    pub fn ngram_speculation_enabled(&self) -> bool {
        self.ngram_speculation_enabled
    }

    /// Enables or disables trained Qwen NextN/MTP drafting for this session.
    /// It is experimental and defaults to `XRT_QWEN_MTP=off` until the real
    /// model admission and performance gates are complete.
    pub fn set_mtp_speculation_enabled(&mut self, enabled: bool) {
        self.mtp_speculation_enabled = enabled;
    }

    pub fn mtp_speculation_enabled(&self) -> bool {
        self.mtp_speculation_enabled
    }

    /// Materializes the session's durable recurrent-state snapshot.
    ///
    /// This is primarily a correctness and checkpointing boundary. CUDA state
    /// is copied to host, so callers should not use it in the decode hot path.
    pub fn recurrent_state_snapshot(&self) -> Result<Option<crate::backend::BackendStateSnapshot>> {
        self.backend_session().recurrent_state_snapshot()
    }

    pub fn gpu_resource_status(&self) -> crate::GpuResourceStatus {
        let backend_session = self.backend_session();
        self.runtime.gpu_resource_status_with_session_allocations(
            backend_session.cuda_kv_allocated_bytes(),
            backend_session.cuda_scratch_allocated_bytes(),
            backend_session.cuda_staging_allocated_bytes(),
            Some(backend_session.requested_cache_mode()),
            Some(backend_session.cache_mode()),
            backend_session.cuda_graph_capture_status(),
        )
    }

    pub fn generate(&mut self, request: &GenerateRequest) -> Result<String> {
        let mut output = String::new();
        self.generate_stream(request, |piece| output.push_str(piece))?;
        Ok(output)
    }

    pub fn generate_scheduled(
        &mut self,
        request: &GenerateRequest,
        scheduler: &Arc<RequestScheduler>,
    ) -> Result<String> {
        let mut output = String::new();
        self.generate_stream_scheduled(request, scheduler, |piece| output.push_str(piece))?;
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
        self.generate_stream_inner(request, None, |piece| {
            on_token(piece);
            ControlFlow::Continue(())
        })
    }

    /// Generates tokens until completion or until the callback requests cancellation.
    ///
    /// Cancellation is checked after each decoded text piece and stops before the next
    /// model invocation. This lets streaming transports release runtime resources when
    /// their client disconnects without changing the existing `generate_stream` API.
    pub fn generate_stream_with_control<F>(
        &mut self,
        request: &GenerateRequest,
        on_token: F,
    ) -> Result<usize>
    where
        F: FnMut(&str) -> ControlFlow<()>,
    {
        self.generate_stream_inner(request, None, on_token)
    }

    pub fn generate_stream_scheduled<F>(
        &mut self,
        request: &GenerateRequest,
        scheduler: &Arc<RequestScheduler>,
        mut on_token: F,
    ) -> Result<usize>
    where
        F: FnMut(&str),
    {
        self.generate_stream_inner(request, Some(scheduler), |piece| {
            on_token(piece);
            ControlFlow::Continue(())
        })
    }

    pub fn generate_stream_scheduled_with_control<F>(
        &mut self,
        request: &GenerateRequest,
        scheduler: &Arc<RequestScheduler>,
        on_token: F,
    ) -> Result<usize>
    where
        F: FnMut(&str) -> ControlFlow<()>,
    {
        self.generate_stream_inner(request, Some(scheduler), on_token)
    }

    fn generate_stream_inner<F>(
        &mut self,
        request: &GenerateRequest,
        scheduler: Option<&Arc<RequestScheduler>>,
        mut on_token: F,
    ) -> Result<usize>
    where
        F: FnMut(&str) -> ControlFlow<()>,
    {
        let runtime = self.runtime.clone();
        let backend = runtime.backend_arc();
        let is_hybrid = backend.config().is_hybrid();
        let _exclusive_turn = scheduler
            .filter(|_| is_hybrid)
            .map(|scheduler| scheduler.acquire_execution_turn(SchedulerExecutionPhase::Exclusive));
        let cooperative_scheduler = scheduler.filter(|_| !is_hybrid);

        self.reset();
        backend.prepare_request()?;
        self.sampler.reseed(request.seed);

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
        self.backend_session_mut().configure_policy(
            session_policy.clone(),
            prompt_tokens.len(),
            &request.prompt_spans,
        );
        backend.prepare_session_state(self.backend_session_mut())?;
        let prefix_request = (request.images.is_empty()
            && self.backend_session().supports_prefix_cache())
        .then(|| {
            runtime.prefix_cache().request(
                runtime.active_backend(),
                self.backend_session().cache_mode(),
                &session_policy,
                &prompt_tokens,
                &request.prompt_spans,
            )
        })
        .flatten();
        let mut prefix_hit = false;
        if let Some(prefix_request) = prefix_request.as_ref() {
            if let Some(snapshot) = runtime.prefix_cache().lookup(prefix_request) {
                match self.backend_session_mut().attach_prefix_snapshot(&snapshot) {
                    Ok(attached_len) if attached_len == prefix_request.prefix_len() => {
                        self.tokens
                            .extend_from_slice(&prompt_tokens[..attached_len]);
                        prefix_hit = true;
                    }
                    Ok(attached_len) => tracing::warn!(
                        "prefix-cache snapshot attached {attached_len} tokens, expected {}; ignoring the entry",
                        prefix_request.prefix_len()
                    ),
                    Err(err) => {
                        tracing::warn!("prefix-cache snapshot attach failed; using prefill: {err}")
                    }
                }
            }
        }
        let graph_total_len = prompt_tokens
            .len()
            .checked_add(request.max_tokens)
            .ok_or_else(|| XrtError::Runtime("generation length overflow".to_string()))?
            .min(backend.config().context_length);
        let kv_reservation_bytes = self
            .backend_session()
            .kv_reservation_bytes_for_total_len(graph_total_len)?;
        if let Some(scheduler) = scheduler {
            let external_kv_bytes = if runtime.active_backend() == crate::BackendKind::CudaResident
            {
                runtime.prefix_cache_status().device_resident_bytes
            } else {
                0
            };
            scheduler.configure_external_kv_bytes(external_kv_bytes);
        }
        let _kv_reservation = scheduler
            .map(|scheduler| scheduler.reserve_kv_bytes(kv_reservation_bytes))
            .transpose()
            .map_err(|err| XrtError::Runtime(err.to_string()))?;

        let mut embedding_overrides = if request.images.is_empty() {
            HashMap::new()
        } else {
            build_image_embedding_overrides(&runtime, &prompt_tokens, &request.images)?
        };

        let prefill_chunk_tokens = cooperative_scheduler
            .map(|scheduler| scheduler.config().prefill_chunk_tokens)
            .unwrap_or(prompt_tokens.len());
        let prefill_registration =
            cooperative_scheduler.map(|scheduler| scheduler.register_prefill_sequence());
        let mut logits = Vec::new();
        let mut capacity_prepared = false;
        let mut prefix_stored = prefix_hit;
        let mut prompt_position = self.tokens.len();
        while prompt_position < prompt_tokens.len() {
            let prefix_boundary = prefix_request
                .as_ref()
                .filter(|_| !prefix_stored)
                .map(PrefixCacheRequest::prefix_len)
                .filter(|&prefix_len| prefix_len > prompt_position)
                .unwrap_or(prompt_tokens.len());
            let chunk_end = prompt_position
                .checked_add(prefill_chunk_tokens)
                .ok_or_else(|| XrtError::Runtime("prefill chunk position overflow".to_string()))?
                .min(prefix_boundary)
                .min(prompt_tokens.len());
            let chunk = &prompt_tokens[prompt_position..chunk_end];
            let start_position = prompt_position;
            let chunk_overrides =
                take_embedding_overrides(&mut embedding_overrides, start_position, chunk.len())?;
            let _turn = cooperative_scheduler.map(|scheduler| {
                scheduler.acquire_execution_turn(SchedulerExecutionPhase::Prefill)
            });
            if !capacity_prepared {
                if backend.supports_cuda_graph_decode() {
                    self.backend_session_mut()
                        .prepare_cuda_graph_generation_capacity(graph_total_len);
                }
                // Graph capacity does not make shared prefix pages writable:
                // copy only pages touched by this prompt chunk/session.
                self.backend_session_mut()
                    .prepare_for_total_len(prompt_tokens.len())?;
                capacity_prepared = true;
            }
            logits = if chunk_overrides.is_empty() {
                backend.forward_batch(chunk, start_position, self.backend_session_mut())?
            } else {
                backend.forward_batch_with_embeddings(
                    chunk,
                    start_position,
                    self.backend_session_mut(),
                    chunk_overrides,
                )?
            };
            self.tokens.extend_from_slice(chunk);
            prompt_position = chunk_end;

            if !prefix_stored
                && prefix_request
                    .as_ref()
                    .is_some_and(|request| request.prefix_len() == prompt_position)
            {
                if let Some(snapshot) = self
                    .backend_session_mut()
                    .snapshot_prefix(prompt_position)?
                {
                    runtime.prefix_cache().insert(
                        prefix_request
                            .as_ref()
                            .expect("prefix request exists at its boundary")
                            .clone(),
                        snapshot,
                    );
                    if let Some(scheduler) = scheduler {
                        let external_kv_bytes =
                            if runtime.active_backend() == crate::BackendKind::CudaResident {
                                runtime.prefix_cache_status().device_resident_bytes
                            } else {
                                0
                            };
                        scheduler.configure_external_kv_bytes(external_kv_bytes);
                    }
                    capacity_prepared = false;
                }
                prefix_stored = true;
            }
        }
        drop(prefill_registration);

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
        let mut generated = 0usize;
        let mut pending_decode_tokens = Vec::new();

        let mut emit_token = |token: u32, force_flush: bool| -> Result<bool> {
            pending_decode_tokens.push(token);
            match tokenizer.decode(&pending_decode_tokens, true) {
                Ok(piece) => {
                    let should_continue =
                        piece.is_empty() || matches!(on_token(&piece), ControlFlow::Continue(()));
                    pending_decode_tokens.clear();
                    Ok(should_continue)
                }
                Err(XrtError::Tokenizer(message))
                    if message.contains("invalid utf8 in decode") && !force_flush =>
                {
                    Ok(true)
                }
                Err(XrtError::Tokenizer(message)) if message.contains("invalid utf8 in decode") => {
                    let piece = tokenizer.decode_lossy(&pending_decode_tokens, true)?;
                    let should_continue =
                        piece.is_empty() || matches!(on_token(&piece), ControlFlow::Continue(()));
                    pending_decode_tokens.clear();
                    Ok(should_continue)
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
            if !emit_token(next, false)? {
                return Ok(generated);
            }

            let remaining = request.max_tokens - generated;
            if remaining == 0 {
                break;
            }

            // Hybrid speculation is admitted only when the backend owns a
            // device-local recurrent journal. CPU hybrid sessions retain the
            // correctness-first non-speculative path.
            let draft = if is_hybrid && !self.backend_session().supports_fast_recurrent_checkpoint()
            {
                Vec::new()
            } else {
                let mtp = if self.mtp_speculation_enabled && sampler_config.temperature <= 1e-5 {
                    backend.draft_mtp_greedy(next, remaining, self.backend_session_mut())?
                } else {
                    None
                };
                match mtp {
                    Some(draft) => draft,
                    None if self.ngram_speculation_enabled => self.ngram_draft(remaining),
                    None => Vec::new(),
                }
            };

            if draft.is_empty() {
                // No speculation: standard single-token decode
                let total_len = self.tokens.len();
                self.backend_session_mut()
                    .prepare_for_total_len(total_len)?;
                let decode_position = total_len - 1;
                let can_batch = cooperative_scheduler.is_some()
                    && backend.supports_multi_sequence_decode_batch()
                    && self.backend_session().cache_mode() == KvCacheMode::F32
                    && cooperative_scheduler
                        .is_some_and(|scheduler| scheduler.config().max_decode_batch_size > 1);
                if can_batch {
                    let scheduler = cooperative_scheduler
                        .expect("batched decode requires a cooperative scheduler");
                    let backend_session = self
                        .backend_session
                        .take()
                        .expect("backend session must be available before decode rendezvous");
                    let (backend_session, decode_result) = scheduler.forward_token_batched(
                        backend.clone(),
                        self.decode_sequence_id,
                        next,
                        decode_position,
                        backend_session,
                    );
                    self.backend_session = Some(backend_session);
                    logits = decode_result?;
                } else {
                    let _turn = cooperative_scheduler.map(|scheduler| {
                        scheduler.acquire_execution_turn(SchedulerExecutionPhase::Decode)
                    });
                    backend.forward_token(
                        next,
                        decode_position,
                        self.backend_session_mut(),
                        &mut logits,
                    )?;
                }
            } else if !is_hybrid {
                // Standard transformer speculation: batched forward + KV cache rollback
                let mut batch_tokens = Vec::with_capacity(1 + draft.len());
                batch_tokens.push(next);
                batch_tokens.extend_from_slice(&draft);

                let start_pos = self.tokens.len() - 1;
                let all_logits = {
                    let _turn = cooperative_scheduler.map(|scheduler| {
                        scheduler.acquire_execution_turn(SchedulerExecutionPhase::Decode)
                    });
                    self.backend_session_mut()
                        .prepare_for_total_len(total_len_after_batch(
                            start_pos,
                            batch_tokens.len(),
                        )?)?;
                    backend.forward_batch_all_logits(
                        &batch_tokens,
                        start_pos,
                        self.backend_session_mut(),
                    )?
                };

                // Verify draft tokens greedily (argmax)
                let mut accepted = 0;
                for i in 0..draft.len() {
                    let pos_logits = logits_for_position(&all_logits, i, vocab_size)?;
                    let predicted = argmax(pos_logits);
                    if predicted == draft[i] {
                        accepted += 1;
                        self.tokens.push(draft[i]);
                        generated += 1;
                        if !emit_token(draft[i], false)? {
                            return Ok(generated);
                        }
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
                let retained_len = self.tokens.len();
                self.backend_session_mut().truncate(retained_len)?;

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
                // CUDA hybrid speculation keeps one persistent device-local
                // DeltaNet journal. KV and recurrent state are always restored
                // to the same accepted boundary before a rejected suffix is
                // replayed.
                let cache_len_before = self.tokens.len() - 1;
                let mut batch_tokens = Vec::with_capacity(1 + draft.len());
                batch_tokens.push(next);
                batch_tokens.extend_from_slice(&draft);

                let start_pos = self.tokens.len() - 1;
                let verification_total_len = total_len_after_batch(start_pos, batch_tokens.len())?;
                self.backend_session_mut()
                    .begin_fast_recurrent_checkpoint(cache_len_before)?;
                if let Err(error) = self
                    .backend_session_mut()
                    .prepare_for_total_len(verification_total_len)
                {
                    return Err(self.hybrid_speculation_error(
                        cache_len_before,
                        "verification preparation",
                        error,
                    ));
                }
                let all_logits = match backend.forward_batch_all_logits(
                    &batch_tokens,
                    start_pos,
                    self.backend_session_mut(),
                ) {
                    Ok(logits) => logits,
                    Err(forward_error) => {
                        return Err(self.hybrid_speculation_error(
                            cache_len_before,
                            "verification",
                            forward_error,
                        ));
                    }
                };

                let mut accepted = 0;
                for i in 0..draft.len() {
                    let pos_logits = match logits_for_position(&all_logits, i, vocab_size) {
                        Ok(logits) => logits,
                        Err(error) => {
                            return Err(self.hybrid_speculation_error(
                                cache_len_before,
                                "logit verification",
                                error,
                            ));
                        }
                    };
                    let predicted = argmax(pos_logits);
                    if predicted == draft[i] {
                        accepted += 1;
                        if Some(draft[i]) == eos
                            || self.tokens.len() + accepted >= ctx_len
                            || generated + accepted >= request.max_tokens
                        {
                            break;
                        }
                    } else {
                        break;
                    }
                }
                self.speculative_stats.verification_batches = self
                    .speculative_stats
                    .verification_batches
                    .saturating_add(1);
                self.speculative_stats.drafted_tokens = self
                    .speculative_stats
                    .drafted_tokens
                    .saturating_add(u64::try_from(draft.len()).unwrap_or(u64::MAX));
                self.speculative_stats.accepted_tokens = self
                    .speculative_stats
                    .accepted_tokens
                    .saturating_add(u64::try_from(accepted).unwrap_or(u64::MAX));
                self.speculative_stats.rejected_tokens =
                    self.speculative_stats.rejected_tokens.saturating_add(
                        u64::try_from(draft.len().saturating_sub(accepted)).unwrap_or(u64::MAX),
                    );

                let total_processed = 1 + draft.len(); // next + all draft tokens
                let total_kept = 1 + accepted; // next + accepted draft tokens

                let verified_logits = if total_kept < total_processed {
                    self.rollback_hybrid_speculation(cache_len_before)?;
                    // Keep a fresh journal active across replay and callbacks so
                    // cancellation can discard an accepted-but-unemitted suffix.
                    let replay_tokens = &batch_tokens[..total_kept];
                    let replay_total_len = total_len_after_batch(start_pos, replay_tokens.len())?;
                    self.backend_session_mut()
                        .begin_fast_recurrent_checkpoint(cache_len_before)?;
                    if let Err(error) = self
                        .backend_session_mut()
                        .prepare_for_total_len(replay_total_len)
                    {
                        return Err(self.hybrid_speculation_error(
                            cache_len_before,
                            "accepted-prefix replay preparation",
                            error,
                        ));
                    }
                    let replay_logits = match backend.forward_batch_all_logits(
                        replay_tokens,
                        start_pos,
                        self.backend_session_mut(),
                    ) {
                        Ok(logits) => logits,
                        Err(error) => {
                            return Err(self.hybrid_speculation_error(
                                cache_len_before,
                                "accepted-prefix replay",
                                error,
                            ));
                        }
                    };
                    let last_idx = total_kept - 1;
                    match logits_for_position(&replay_logits, last_idx, vocab_size) {
                        Ok(logits) => logits.to_vec(),
                        Err(error) => {
                            return Err(self.hybrid_speculation_error(
                                cache_len_before,
                                "accepted-prefix replay logits",
                                error,
                            ));
                        }
                    }
                } else {
                    let last_idx = total_processed - 1;
                    match logits_for_position(&all_logits, last_idx, vocab_size) {
                        Ok(logits) => logits.to_vec(),
                        Err(error) => {
                            return Err(self.hybrid_speculation_error(
                                cache_len_before,
                                "accepted verification logits",
                                error,
                            ));
                        }
                    }
                };

                let mut emitted_accepted = 0usize;
                for &token in draft.iter().take(accepted) {
                    self.tokens.push(token);
                    generated += 1;
                    emitted_accepted += 1;
                    let should_continue = match emit_token(token, false) {
                        Ok(should_continue) => should_continue,
                        Err(error) => {
                            return Err(self.hybrid_speculation_error(
                                cache_len_before,
                                "accepted-token callback",
                                error,
                            ));
                        }
                    };
                    if !should_continue {
                        self.rollback_hybrid_speculation(cache_len_before)?;
                        // Match ordinary streaming cancellation: the token
                        // whose callback returned Break is emitted but is not
                        // forwarded. `next` plus only earlier accepted draft
                        // tokens therefore remain in backend state.
                        let retained = emitted_accepted;
                        let retained_total_len = total_len_after_batch(start_pos, retained)?;
                        self.backend_session_mut()
                            .begin_fast_recurrent_checkpoint(cache_len_before)?;
                        if let Err(error) = self
                            .backend_session_mut()
                            .prepare_for_total_len(retained_total_len)
                        {
                            return Err(self.hybrid_speculation_error(
                                cache_len_before,
                                "cancelled-prefix replay preparation",
                                error,
                            ));
                        }
                        if let Err(error) = backend.forward_batch_all_logits(
                            &batch_tokens[..retained],
                            start_pos,
                            self.backend_session_mut(),
                        ) {
                            return Err(self.hybrid_speculation_error(
                                cache_len_before,
                                "cancelled-prefix replay",
                                error,
                            ));
                        }
                        if let Err(error) = self
                            .backend_session_mut()
                            .commit_fast_recurrent_checkpoint()
                        {
                            return Err(self.hybrid_speculation_error(
                                cache_len_before,
                                "cancelled-prefix commit",
                                error,
                            ));
                        }
                        return Ok(generated);
                    }
                }
                if let Err(error) = self
                    .backend_session_mut()
                    .commit_fast_recurrent_checkpoint()
                {
                    return Err(self.hybrid_speculation_error(
                        cache_len_before,
                        "accepted-prefix commit",
                        error,
                    ));
                }
                logits.resize(vocab_size, 0.0);
                logits.copy_from_slice(&verified_logits);

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
                let _ = on_token(&piece);
            }
        }

        Ok(generated)
    }

    fn ensure_cache_mode(&mut self, mode: KvCacheMode) {
        if self.backend_session().requested_cache_mode() == mode {
            return;
        }
        let config = self.runtime.backend().config();
        let layer_widths = config.gemma4_layer_kv_widths();
        let block_count = config.block_count;
        let kv_width = config.kv_width();
        let page_tokens = self.page_tokens;
        if let Some(layer_widths) = layer_widths {
            self.backend_session_mut().replace_cache_with_layer_widths(
                mode,
                layer_widths,
                page_tokens,
            );
        } else {
            self.backend_session_mut()
                .replace_cache(mode, block_count, kv_width, page_tokens);
        }
    }

    fn rollback_hybrid_speculation(&mut self, boundary: usize) -> Result<()> {
        self.speculative_stats.rollback_count =
            self.speculative_stats.rollback_count.saturating_add(1);
        let kv_result = self.backend_session_mut().truncate(boundary);
        let recurrent_result = self
            .backend_session_mut()
            .rollback_fast_recurrent_checkpoint(boundary);
        match (kv_result, recurrent_result) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(kv_error), Ok(())) => Err(XrtError::Runtime(format!(
                "hybrid speculative KV rollback to {boundary} failed: {kv_error}"
            ))),
            (Ok(()), Err(recurrent_error)) => Err(XrtError::Runtime(format!(
                "hybrid speculative recurrent rollback to {boundary} failed: {recurrent_error}"
            ))),
            (Err(kv_error), Err(recurrent_error)) => Err(XrtError::Runtime(format!(
                "hybrid speculative rollback to {boundary} failed for KV ({kv_error}) and recurrent state ({recurrent_error})"
            ))),
        }
    }

    fn hybrid_speculation_error(
        &mut self,
        boundary: usize,
        phase: &str,
        error: XrtError,
    ) -> XrtError {
        match self.rollback_hybrid_speculation(boundary) {
            Ok(()) => error,
            Err(rollback_error) => XrtError::Runtime(format!(
                "hybrid speculative {phase} failed ({error}); {rollback_error}"
            )),
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
        let destruction_error = self
            .backend_session
            .take()
            .and_then(|session| session.destroy_safely().err());
        if let Some(error) = destruction_error {
            tracing::error!(
                "failed to synchronize backend session before destruction; retiring its CUDA allocations: {error}"
            );
        }
        self.runtime.unregister_session();
    }
}

pub(crate) fn ngram_speculation_enabled_from_env() -> bool {
    env::var("XRT_NGRAM_SPECULATION")
        .ok()
        .as_deref()
        .and_then(parse_bool)
        .unwrap_or(true)
}

pub(crate) fn mtp_speculation_enabled_from_env() -> bool {
    env::var("XRT_QWEN_MTP")
        .ok()
        .as_deref()
        .and_then(parse_bool)
        .unwrap_or(false)
}

fn parse_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "on" | "enabled" => Some(true),
        "0" | "false" | "off" | "disabled" => Some(false),
        _ => None,
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

fn take_embedding_overrides(
    overrides: &mut HashMap<usize, Vec<f32>>,
    start_position: usize,
    chunk_len: usize,
) -> Result<HashMap<usize, Vec<f32>>> {
    let mut chunk_overrides = HashMap::new();
    for local_index in 0..chunk_len {
        let global_index = checked_add(
            start_position,
            local_index,
            "prefill embedding override position",
        )?;
        if let Some(embedding) = overrides.remove(&global_index) {
            chunk_overrides.insert(local_index, embedding);
        }
    }
    Ok(chunk_overrides)
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
    use super::{
        argmax, checked_add, logits_for_position, parse_bool, take_embedding_overrides,
        total_len_after_batch,
    };
    use std::collections::HashMap;

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

    #[test]
    fn ngram_speculation_kill_switch_values_are_explicit() {
        for enabled in ["1", "true", "ON", "enabled"] {
            assert_eq!(parse_bool(enabled), Some(true));
        }
        for disabled in ["0", "false", "OFF", "disabled"] {
            assert_eq!(parse_bool(disabled), Some(false));
        }
        assert_eq!(parse_bool("auto"), None);
        assert_eq!(parse_bool(""), None);
    }

    #[test]
    fn chunk_embedding_overrides_move_and_remap_local_positions() {
        let mut overrides = HashMap::from([
            (1usize, vec![1.0f32]),
            (3usize, vec![3.0f32]),
            (5usize, vec![5.0f32]),
        ]);

        let chunk = take_embedding_overrides(&mut overrides, 2, 3).unwrap();
        assert_eq!(chunk, HashMap::from([(1usize, vec![3.0f32])]));
        assert_eq!(
            overrides,
            HashMap::from([(1usize, vec![1.0f32]), (5usize, vec![5.0f32])])
        );
    }
}
