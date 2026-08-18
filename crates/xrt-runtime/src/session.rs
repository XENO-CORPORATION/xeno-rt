use crate::{
    backend::MtpDraftProposal, prefix_cache::PrefixCacheRequest, BackendSession, CachePolicyKind,
    KvCacheMode, PromptSpan, RequestScheduler, Runtime, Sampler, SamplerConfig,
    SchedulerExecutionPhase, SessionPolicy,
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
    time::Instant,
};
use xrt_core::{checked_mul, Result, XrtError};

/// N-gram order for prompt lookup decoding.
const NGRAM_ORDER: usize = 3;

/// Maximum number of draft tokens per speculation round.
const MAX_DRAFT: usize = 5;
// One full depth-six probe is enough to identify requests where the trained
// draft head is providing no useful work. Waiting for four windows made
// low-acceptance requests measurably slower than target-only decode.
const MTP_ADAPTIVE_MIN_DRAFTED_TOKENS: u64 = 6;

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
    #[serde(default)]
    pub presence_penalty: f32,
    #[serde(default)]
    pub frequency_penalty: f32,
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
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
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
    pub adaptive_fallbacks: u64,
    pub draft_micros: u64,
    pub verify_micros: u64,
    pub rebase_micros: u64,
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
    generated_token_start: Option<usize>,
    ngram_speculation_enabled: bool,
    mtp_draft_diagnostics_enabled: bool,
    mtp_prefer_ngram_enabled: bool,
    mtp_ngram_order: usize,
    mtp_ngram_consensus_enabled: bool,
    mtp_ngram_min_hits: usize,
    mtp_ngram_min_percent: usize,
    mtp_ngram_lookback: usize,
    mtp_speculation_enabled: bool,
    mtp_max_draft_tokens: usize,
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
        let mtp_speculation_enabled = mtp_speculation_enabled_from_env();
        let mut backend_session = runtime
            .backend()
            .new_session(default_cache_mode, page_tokens);
        backend_session.set_mtp_tracking_enabled(mtp_speculation_enabled);
        runtime.register_session();
        Self {
            runtime,
            default_cache_mode,
            page_tokens,
            decode_sequence_id: NEXT_DECODE_SEQUENCE_ID.fetch_add(1, Ordering::Relaxed),
            backend_session: Some(backend_session),
            sampler: Sampler::new(None),
            tokens: Vec::new(),
            generated_token_start: None,
            ngram_speculation_enabled: ngram_speculation_enabled_from_env(),
            mtp_draft_diagnostics_enabled: mtp_draft_diagnostics_enabled_from_env(),
            mtp_prefer_ngram_enabled: mtp_prefer_ngram_enabled_from_env(),
            mtp_ngram_order: mtp_ngram_order_from_env(),
            mtp_ngram_consensus_enabled: mtp_ngram_consensus_enabled_from_env(),
            mtp_ngram_min_hits: mtp_ngram_min_hits_from_env(),
            mtp_ngram_min_percent: mtp_ngram_min_percent_from_env(),
            mtp_ngram_lookback: mtp_ngram_lookback_from_env(),
            mtp_speculation_enabled,
            mtp_max_draft_tokens: mtp_max_draft_tokens_from_env(),
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
        self.generated_token_start = None;
        self.speculative_stats = SpeculativeDecodeStats::default();
    }

    /// Returns the exact token IDs emitted by the most recent generation.
    ///
    /// `None` means generation did not reach the decode boundary (for example,
    /// prefill failed). A successful generation that emitted no tokens returns
    /// an empty slice. This additive trace surface lets admission benchmarks
    /// prove greedy parity without changing streaming or OpenAI-compatible APIs.
    pub fn generated_token_ids(&self) -> Option<&[u32]> {
        self.generated_token_start
            .and_then(|start| self.tokens.get(start..))
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
        self.backend_session_mut().set_mtp_tracking_enabled(enabled);
    }

    pub fn mtp_speculation_enabled(&self) -> bool {
        self.mtp_speculation_enabled
    }

    /// Bounds recursive Qwen NextN drafting to one through fifteen tokens.
    /// The default is one; deeper drafting must earn admission on the target
    /// model because rejection cost grows with every speculative token.
    pub fn set_mtp_max_draft_tokens(&mut self, max_draft_tokens: usize) {
        self.mtp_max_draft_tokens = max_draft_tokens.clamp(1, 15);
    }

    pub fn mtp_max_draft_tokens(&self) -> usize {
        self.mtp_max_draft_tokens
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

    /// Generates with prompt tokens already encoded by this runtime.
    ///
    /// In-process frontends use this additive path when they must tokenize for
    /// usage accounting before generation. `prompt_tokens` must encode the
    /// request prompt with the same tokenizer and `add_special_tokens` policy.
    pub fn generate_scheduled_with_prompt_tokens(
        &mut self,
        request: &GenerateRequest,
        prompt_tokens: &[u32],
        scheduler: &Arc<RequestScheduler>,
    ) -> Result<String> {
        let mut output = String::new();
        self.generate_stream_scheduled_with_prompt_tokens(
            request,
            prompt_tokens,
            scheduler,
            |piece| output.push_str(piece),
        )?;
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
        self.generate_stream_inner(request, None, None, |_, piece| {
            on_token(piece);
            ControlFlow::Continue(())
        })
    }

    pub fn generate_stream_with_prompt_tokens<F>(
        &mut self,
        request: &GenerateRequest,
        prompt_tokens: &[u32],
        mut on_token: F,
    ) -> Result<usize>
    where
        F: FnMut(&str),
    {
        self.generate_stream_inner(request, None, Some(prompt_tokens), |_, piece| {
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
        mut on_token: F,
    ) -> Result<usize>
    where
        F: FnMut(&str) -> ControlFlow<()>,
    {
        self.generate_stream_inner(request, None, None, |_, piece| {
            if piece.is_empty() {
                ControlFlow::Continue(())
            } else {
                on_token(piece)
            }
        })
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
        self.generate_stream_inner(request, Some(scheduler), None, |_, piece| {
            on_token(piece);
            ControlFlow::Continue(())
        })
    }

    pub fn generate_stream_scheduled_with_prompt_tokens<F>(
        &mut self,
        request: &GenerateRequest,
        prompt_tokens: &[u32],
        scheduler: &Arc<RequestScheduler>,
        mut on_token: F,
    ) -> Result<usize>
    where
        F: FnMut(&str),
    {
        self.generate_stream_inner(request, Some(scheduler), Some(prompt_tokens), |_, piece| {
            on_token(piece);
            ControlFlow::Continue(())
        })
    }

    pub fn generate_stream_scheduled_with_control<F>(
        &mut self,
        request: &GenerateRequest,
        scheduler: &Arc<RequestScheduler>,
        mut on_token: F,
    ) -> Result<usize>
    where
        F: FnMut(&str) -> ControlFlow<()>,
    {
        self.generate_stream_inner(request, Some(scheduler), None, |_, piece| {
            if piece.is_empty() {
                ControlFlow::Continue(())
            } else {
                on_token(piece)
            }
        })
    }

    /// Scheduled streaming variant that exposes the emitted token ID even
    /// when its decoded text is empty (for example Qwen's thinking boundary).
    pub fn generate_stream_scheduled_with_token_control<F>(
        &mut self,
        request: &GenerateRequest,
        scheduler: &Arc<RequestScheduler>,
        on_token: F,
    ) -> Result<usize>
    where
        F: FnMut(u32, &str) -> ControlFlow<()>,
    {
        self.generate_stream_inner(request, Some(scheduler), None, on_token)
    }

    fn generate_stream_inner<F>(
        &mut self,
        request: &GenerateRequest,
        scheduler: Option<&Arc<RequestScheduler>>,
        prompt_token_hint: Option<&[u32]>,
        mut on_token: F,
    ) -> Result<usize>
    where
        F: FnMut(u32, &str) -> ControlFlow<()>,
    {
        let runtime = self.runtime.clone();
        let backend = runtime.backend_arc();
        let is_hybrid = backend.config().is_hybrid();
        let _exclusive_turn = scheduler
            .filter(|_| is_hybrid)
            .map(|scheduler| scheduler.acquire_execution_turn(SchedulerExecutionPhase::Exclusive));
        let cooperative_scheduler = scheduler.filter(|_| !is_hybrid);

        self.reset();
        // Image-conditioned tokens replace the model embedding for placeholder
        // positions. The current MTP head consumes token IDs, so it cannot
        // reproduce that conditioning yet; keep those requests target-only.
        let mut mtp_request_enabled = mtp_request_eligible(
            self.mtp_speculation_enabled,
            request.images.is_empty(),
            request.temperature,
        );
        let mtp_adaptive_fallback_enabled = mtp_adaptive_fallback_enabled_from_env();
        self.backend_session_mut()
            .set_mtp_tracking_enabled(mtp_request_enabled);
        backend.prepare_request()?;
        self.sampler.reseed(request.seed);

        let tokenizer = runtime.tokenizer();
        let mut prompt_tokens = match prompt_token_hint {
            Some(tokens) => tokens.to_vec(),
            None => {
                tokenizer.encode_with_options(&request.prompt, request.add_special_tokens, true)?
            }
        };
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
        // A target-only prefix snapshot does not yet carry the synchronized
        // MTP attention lane. Reusing it would leave the draft cache behind
        // the target position, so MTP sessions prefill both lanes together.
        let prefix_request = (!mtp_request_enabled
            && request.images.is_empty()
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
        self.generated_token_start = Some(self.tokens.len());

        let sampler_config = SamplerConfig {
            temperature: request.temperature,
            top_k: request.top_k,
            top_p: request.top_p,
            repetition_penalty: request.repetition_penalty,
            presence_penalty: request.presence_penalty,
            frequency_penalty: request.frequency_penalty,
            seed: request.seed,
        };

        let eos = tokenizer.special_tokens().eos;
        let ctx_len = backend.config().context_length;
        let vocab_size = backend.config().vocab_size;
        let mut generated = 0usize;
        let mut pending_decode_tokens = Vec::new();
        let mut verified_greedy_next = None;
        let reuse_dflash_suffix = mtp_reuse_dflash_suffix_enabled_from_env();
        let mut reusable_dflash_suffix = Vec::new();

        let mut emit_token = |token: u32, force_flush: bool| -> Result<bool> {
            pending_decode_tokens.push(token);
            match tokenizer.decode(&pending_decode_tokens, true) {
                Ok(piece) => {
                    let should_continue =
                        matches!(on_token(token, &piece), ControlFlow::Continue(()));
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
                        matches!(on_token(token, &piece), ControlFlow::Continue(()));
                    pending_decode_tokens.clear();
                    Ok(should_continue)
                }
                Err(err) => Err(err),
            }
        };

        while generated < request.max_tokens {
            let next = match verified_greedy_next.take() {
                Some(token) => token,
                None => self.sampler.sample(&logits, &self.tokens, sampler_config)?,
            };
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
            if remaining == 0 || self.tokens.len() >= ctx_len {
                break;
            }

            // Hybrid speculation is admitted only when the backend owns a
            // device-local recurrent journal. CPU hybrid sessions retain the
            // correctness-first non-speculative path.
            let (draft, verify_with_target_sampler, draft_from_reused_suffix, draft_tree) =
                if is_hybrid && !self.backend_session().supports_fast_recurrent_checkpoint() {
                    (Vec::new(), false, false, None)
                } else {
                    let mut reused_suffix = if reuse_dflash_suffix {
                        std::mem::take(&mut reusable_dflash_suffix)
                    } else {
                        Vec::new()
                    };
                    if !reused_suffix.is_empty() {
                        reused_suffix.resize(self.mtp_max_draft_tokens, 0);
                        (reused_suffix, true, true, None)
                    } else {
                        let mut hybrid_ngram = if self.mtp_prefer_ngram_enabled
                            && mtp_request_enabled
                            && sampler_config.temperature <= 1e-5
                        {
                            if self.mtp_ngram_consensus_enabled {
                                ngram_consensus_draft(
                                    &self.tokens,
                                    self.mtp_ngram_order,
                                    remaining.min(self.mtp_max_draft_tokens),
                                    self.mtp_ngram_min_hits,
                                    self.mtp_ngram_min_percent,
                                    self.mtp_ngram_lookback,
                                )
                            } else {
                                self.ngram_draft_with_order(
                                    self.mtp_ngram_order,
                                    remaining,
                                    self.mtp_max_draft_tokens,
                                )
                            }
                        } else {
                            Vec::new()
                        };
                        if !hybrid_ngram.is_empty() {
                            // Keep history proposals on the same fixed verifier
                            // topology as the neural draft.  N-gram matches near
                            // the start of a sequence can be shorter than the
                            // configured proposal width; sending that transient
                            // row count through the hybrid verifier retires its
                            // pointer-bound CUDA graph for the remainder of the
                            // request.  The target still accepts only the real
                            // history tokens.  Padding is executed speculatively
                            // and discarded beyond that explicit limit.
                            let acceptance_limit = pad_hybrid_draft_to_width(
                                &mut hybrid_ngram,
                                remaining.min(self.mtp_max_draft_tokens),
                            );
                            self.backend_session_mut()
                                .set_mtp_draft_acceptance_limit(Some(acceptance_limit));
                        }
                        let mtp = if hybrid_ngram.is_empty()
                            && mtp_request_enabled
                            && sampler_config.temperature <= 1e-5
                        {
                            let started = Instant::now();
                            let draft = backend.draft_mtp_proposal(
                                next,
                                remaining.min(self.mtp_max_draft_tokens),
                                self.backend_session_mut(),
                            )?;
                            self.speculative_stats.draft_micros = self
                                .speculative_stats
                                .draft_micros
                                .saturating_add(elapsed_micros(started));
                            draft
                        } else {
                            None
                        };
                        if !hybrid_ngram.is_empty() {
                            (hybrid_ngram, true, false, None)
                        } else {
                            match mtp {
                                Some(MtpDraftProposal::Linear(draft)) => (draft, true, false, None),
                                Some(MtpDraftProposal::Tree(tree))
                                    if mtp_compact_greedy_eligible(sampler_config) =>
                                {
                                    (tree.tokens.clone(), true, false, Some(tree))
                                }
                                Some(MtpDraftProposal::Tree(tree)) => {
                                    // Tree verification is an exact greedy-only
                                    // optimization. For penalties or stochastic
                                    // sampling, retain the drafter's rank-zero
                                    // chain and use the established sampler-aware
                                    // linear verifier.
                                    let mut path = Vec::new();
                                    let mut parent = 0usize;
                                    while path.len() < remaining.min(self.mtp_max_draft_tokens) {
                                        let Some(index) = tree
                                            .parents
                                            .iter()
                                            .position(|&candidate| candidate == parent)
                                        else {
                                            break;
                                        };
                                        path.push(tree.tokens[index]);
                                        parent = index + 1;
                                    }
                                    (path, true, false, None)
                                }
                                None if self.ngram_speculation_enabled => {
                                    (self.ngram_draft(remaining), false, false, None)
                                }
                                None => (Vec::new(), false, false, None),
                            }
                        }
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
                    let predicted = if verify_with_target_sampler {
                        self.sampler
                            .sample(pos_logits, &self.tokens, sampler_config)?
                    } else {
                        argmax(pos_logits)
                    };
                    // EOS belongs to the sampler boundary: accepting and
                    // emitting it here would differ from the ordinary path,
                    // which stops before adding the token to session history.
                    if predicted == draft[i] && Some(predicted) != eos {
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
                // Hybrid MTP verifies the complete 2..16-token target window in
                // one layerwise CUDA pass. A device-local journal protects the
                // accepted boundary; rejected suffixes pay one bounded repair
                // replay while fully accepted windows commit without replay.
                let cache_len_before = self.tokens.len() - 1;
                let verify_draft_len = draft
                    .len()
                    .min(ctx_len.saturating_sub(self.tokens.len()))
                    .min(request.max_tokens.saturating_sub(generated));
                let verify_draft = &draft[..verify_draft_len];
                let draft_acceptance_limit = self
                    .backend_session()
                    .mtp_draft_acceptance_limit()
                    .unwrap_or(verify_draft_len)
                    .min(verify_draft_len);
                let diagnostic_ngram_drafts = self.mtp_draft_diagnostics_enabled.then(|| {
                    (3..=8)
                        .map(|order| (order, self.ngram_draft_with_order(order, remaining, 15)))
                        .collect::<Vec<_>>()
                });
                let mut batch_tokens = Vec::with_capacity(1 + verify_draft.len());
                batch_tokens.push(next);
                batch_tokens.extend_from_slice(verify_draft);

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
                let verify_started = Instant::now();
                let verify_result =
                    (|| -> Result<(usize, Vec<f32>, Option<u32>, Option<Vec<usize>>)> {
                        if let Some(tree) = draft_tree.as_ref() {
                            let mut compact = backend
                            .forward_mtp_verify_tree_greedy(
                                &batch_tokens,
                                tree,
                                start_pos,
                                self.backend_session_mut(),
                            )?
                            .ok_or_else(|| {
                                XrtError::Unsupported(
                                    "the active backend could not execute admitted Qwen draft-tree verification"
                                        .to_string(),
                                )
                            })?;
                            if compact.accepted_rows.first().copied() != Some(0) {
                                return Err(XrtError::Runtime(
                                    "compact MTP tree result did not start at root row zero"
                                        .to_string(),
                                ));
                            }
                            let mut parent = 0usize;
                            for &row in compact.accepted_rows.iter().skip(1) {
                                if row == 0
                                    || row > tree.tokens.len()
                                    || tree.parents[row - 1] != parent
                                {
                                    return Err(XrtError::Runtime(format!(
                                    "compact MTP tree selected row {row} outside the target-approved path from parent {parent}"
                                )));
                                }
                                parent = row;
                            }
                            if compact.boundary_token as usize >= vocab_size {
                                return Err(XrtError::Runtime(format!(
                                "compact MTP tree boundary token {} exceeds vocabulary size {vocab_size}",
                                compact.boundary_token
                            )));
                            }

                            // EOS remains a sampler boundary and is never added to
                            // session history. If the tree followed an EOS node,
                            // retain only its parent path and publish EOS as that
                            // parent's already verified next token.
                            if let Some(eos_token) = eos {
                                if let Some(path_index) = compact
                                    .accepted_rows
                                    .iter()
                                    .skip(1)
                                    .position(|&row| tree.tokens[row - 1] == eos_token)
                                {
                                    compact.accepted_rows.truncate(path_index + 1);
                                    compact.boundary_token = eos_token;
                                }
                            }
                            let accepted = compact.accepted_rows.len().saturating_sub(1);
                            return Ok((
                                accepted,
                                Vec::new(),
                                Some(compact.boundary_token),
                                Some(compact.accepted_rows),
                            ));
                        }
                        if verify_with_target_sampler
                            && mtp_compact_greedy_eligible(sampler_config)
                            && !verify_draft.iter().any(|&token| Some(token) == eos)
                        {
                            if let Some(compact) = backend.forward_mtp_verify_greedy(
                                &batch_tokens,
                                verify_draft,
                                start_pos,
                                self.backend_session_mut(),
                            )? {
                                if compact.accepted > verify_draft.len() {
                                    return Err(XrtError::Runtime(format!(
                                    "compact MTP verification accepted {} tokens from a {}-token draft",
                                    compact.accepted,
                                    verify_draft.len()
                                )));
                                }
                                if compact.boundary_token as usize >= vocab_size {
                                    return Err(XrtError::Runtime(format!(
                                    "compact MTP boundary token {} exceeds vocabulary size {vocab_size}",
                                    compact.boundary_token
                                )));
                                }
                                return Ok((
                                    compact.accepted,
                                    Vec::new(),
                                    Some(compact.boundary_token),
                                    None,
                                ));
                            }
                        }

                        let all_logits = backend.forward_mtp_verify_all_logits(
                            &batch_tokens,
                            start_pos,
                            self.backend_session_mut(),
                        )?;
                        let mut accepted = 0;
                        let mut verification_history = self.tokens.clone();
                        for (input_index, &draft_token) in
                            verify_draft.iter().take(draft_acceptance_limit).enumerate()
                        {
                            let pos_logits =
                                logits_for_position(&all_logits, input_index, vocab_size)?;
                            let predicted = if verify_with_target_sampler {
                                self.sampler.sample(
                                    pos_logits,
                                    &verification_history,
                                    sampler_config,
                                )?
                            } else {
                                argmax(pos_logits)
                            };
                            if predicted == draft_token && Some(predicted) != eos {
                                accepted += 1;
                                verification_history.push(draft_token);
                            } else {
                                break;
                            }
                        }
                        Ok((
                            accepted,
                            logits_for_position(&all_logits, accepted, vocab_size)?.to_vec(),
                            None,
                            None,
                        ))
                    })();
                self.speculative_stats.verify_micros = self
                    .speculative_stats
                    .verify_micros
                    .saturating_add(elapsed_micros(verify_started));
                let (accepted, verified_logits, verified_token, selected_tree_rows) =
                    match verify_result {
                        Ok(result) => result,
                        Err(forward_error) => {
                            return Err(self.hybrid_speculation_error(
                                cache_len_before,
                                "batched verification",
                                forward_error,
                            ));
                        }
                    };
                self.speculative_stats.verification_batches = self
                    .speculative_stats
                    .verification_batches
                    .saturating_add(1);
                self.speculative_stats.drafted_tokens = self
                    .speculative_stats
                    .drafted_tokens
                    .saturating_add(u64::try_from(draft_acceptance_limit).unwrap_or(u64::MAX));
                self.speculative_stats.accepted_tokens = self
                    .speculative_stats
                    .accepted_tokens
                    .saturating_add(u64::try_from(accepted).unwrap_or(u64::MAX));
                self.speculative_stats.rejected_tokens =
                    self.speculative_stats.rejected_tokens.saturating_add(
                        u64::try_from(draft_acceptance_limit.saturating_sub(accepted))
                            .unwrap_or(u64::MAX),
                    );
                if draft_tree.is_none()
                    && reuse_dflash_suffix
                    && !draft_from_reused_suffix
                    && accepted > 0
                    && accepted + 1 < verify_draft.len()
                {
                    reusable_dflash_suffix = verify_draft[accepted + 1..].to_vec();
                } else {
                    reusable_dflash_suffix.clear();
                }
                tracing::debug!(
                    target: "xrt_runtime::mtp",
                    start_position = start_pos,
                    drafted = draft_acceptance_limit,
                    padded_rows = verify_draft.len().saturating_sub(draft_acceptance_limit),
                    accepted,
                    retained_inputs = 1 + accepted,
                    draft_tokens = ?verify_draft,
                    diagnostic_ngram_drafts = ?diagnostic_ngram_drafts,
                    verified_boundary = ?verified_token,
                    selected_tree_rows = ?selected_tree_rows,
                    "verified Qwen MTP window"
                );

                let accepted_tokens = if let (Some(tree), Some(rows)) =
                    (draft_tree.as_ref(), selected_tree_rows.as_ref())
                {
                    rows.iter()
                        .skip(1)
                        .map(|&row| tree.tokens[row - 1])
                        .collect::<Vec<_>>()
                } else {
                    verify_draft.iter().take(accepted).copied().collect()
                };
                let mut emitted_accepted = 0usize;
                let mut callback_cancelled = false;
                for token in accepted_tokens {
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
                        callback_cancelled = true;
                        break;
                    }
                }

                // A callback that stops on an accepted token emits that token
                // but, like ordinary streaming cancellation, does not forward
                // it. Otherwise retain `next` plus every accepted draft token.
                let retained_inputs = if callback_cancelled {
                    emitted_accepted
                } else {
                    1 + accepted
                };
                let rebase_started = Instant::now();
                let retained_tree_rows = selected_tree_rows
                    .as_ref()
                    .map(|rows| rows[..retained_inputs.min(rows.len())].to_vec());
                let rebase_result = if let (Some(tree), Some(rows)) =
                    (draft_tree.as_ref(), retained_tree_rows.as_ref())
                {
                    backend.rebase_mtp_tree_after_verify(
                        &batch_tokens,
                        tree,
                        rows,
                        start_pos,
                        self.backend_session_mut(),
                    )
                } else {
                    backend.rebase_mtp_after_verify(
                        &batch_tokens,
                        start_pos,
                        retained_inputs,
                        self.backend_session_mut(),
                    )
                };
                self.speculative_stats.rebase_micros = self
                    .speculative_stats
                    .rebase_micros
                    .saturating_add(elapsed_micros(rebase_started));
                if let Err(error) = rebase_result {
                    return Err(self.hybrid_speculation_error(
                        cache_len_before,
                        "accepted-prefix MTP cache rebase",
                        error,
                    ));
                }
                if retained_inputs < batch_tokens.len() {
                    let retained_total_len = total_len_after_batch(start_pos, retained_inputs)?;
                    if let Err(error) = self.backend_session_mut().truncate(retained_total_len) {
                        return Err(self.hybrid_speculation_error(
                            cache_len_before,
                            "accepted-prefix KV rebase",
                            error,
                        ));
                    }
                    let publish = if let Some(rows) = retained_tree_rows.as_ref() {
                        self.backend_session_mut()
                            .publish_fast_recurrent_tree_boundary(
                                cache_len_before,
                                *rows.last().unwrap_or(&0),
                                retained_inputs,
                            )
                    } else {
                        self.backend_session_mut()
                            .publish_fast_recurrent_verify_boundary(
                                cache_len_before,
                                retained_inputs,
                            )
                    };
                    if let Err(error) = publish {
                        return Err(self.hybrid_speculation_error(
                            cache_len_before,
                            "accepted-prefix recurrent rebase",
                            error,
                        ));
                    }
                } else if let Err(error) = if let Some(rows) = retained_tree_rows.as_ref() {
                    self.backend_session_mut()
                        .publish_fast_recurrent_tree_boundary(
                            cache_len_before,
                            *rows.last().unwrap_or(&0),
                            retained_inputs,
                        )
                } else {
                    self.backend_session_mut()
                        .publish_fast_recurrent_verify_boundary(cache_len_before, retained_inputs)
                } {
                    return Err(self.hybrid_speculation_error(
                        cache_len_before,
                        "fully accepted recurrent publish",
                        error,
                    ));
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
                if mtp_adaptive_fallback_enabled
                    && mtp_should_adaptively_fallback(self.speculative_stats)
                {
                    mtp_request_enabled = false;
                    self.speculative_stats.adaptive_fallbacks =
                        self.speculative_stats.adaptive_fallbacks.saturating_add(1);
                    self.backend_session_mut().set_mtp_tracking_enabled(false);
                }
                if callback_cancelled {
                    return Ok(generated);
                }
                if let Some(token) = verified_token {
                    verified_greedy_next = Some(token);
                } else {
                    logits.resize(vocab_size, 0.0);
                    logits.copy_from_slice(&verified_logits);
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
                let token = *pending_decode_tokens
                    .last()
                    .expect("non-empty pending decode tokens");
                let _ = on_token(token, &piece);
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
        self.ngram_draft_with_order(NGRAM_ORDER, max_tokens, MAX_DRAFT)
    }

    fn ngram_draft_with_order(
        &self,
        n: usize,
        max_tokens: usize,
        max_draft_tokens: usize,
    ) -> Vec<u32> {
        let tokens = &self.tokens;
        if tokens.len() < n + 1 {
            return Vec::new();
        }

        let max_draft = max_draft_tokens.min(max_tokens);
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

fn mtp_draft_diagnostics_enabled_from_env() -> bool {
    env::var("XRT_MTP_DRAFT_DIAGNOSTICS")
        .ok()
        .as_deref()
        .and_then(parse_bool)
        .unwrap_or(false)
}

fn mtp_prefer_ngram_enabled_from_env() -> bool {
    env::var("XRT_QWEN_MTP_PREFER_NGRAM")
        .ok()
        .as_deref()
        .and_then(parse_bool)
        .unwrap_or(false)
}

fn mtp_ngram_order_from_value(value: Option<&str>) -> usize {
    value
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|order| (3..=32).contains(order))
        .unwrap_or(NGRAM_ORDER)
}

fn mtp_ngram_order_from_env() -> usize {
    mtp_ngram_order_from_value(env::var("XRT_QWEN_MTP_NGRAM_ORDER").ok().as_deref())
}

fn mtp_ngram_consensus_enabled_from_env() -> bool {
    env::var("XRT_QWEN_MTP_NGRAM_CONSENSUS")
        .ok()
        .as_deref()
        .and_then(parse_bool)
        .unwrap_or(false)
}

fn mtp_ngram_min_hits_from_value(value: Option<&str>) -> usize {
    value
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|hits| (1..=32).contains(hits))
        .unwrap_or(2)
}

fn mtp_ngram_min_hits_from_env() -> usize {
    mtp_ngram_min_hits_from_value(env::var("XRT_QWEN_MTP_NGRAM_MIN_HITS").ok().as_deref())
}

fn mtp_ngram_min_percent_from_value(value: Option<&str>) -> usize {
    value
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|percent| *percent <= 100)
        .unwrap_or(66)
}

fn mtp_ngram_min_percent_from_env() -> usize {
    mtp_ngram_min_percent_from_value(env::var("XRT_QWEN_MTP_NGRAM_MIN_PERCENT").ok().as_deref())
}

fn mtp_ngram_lookback_from_value(value: Option<&str>) -> usize {
    value
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|lookback| (32..=65_536).contains(lookback))
        .unwrap_or(8_192)
}

fn mtp_ngram_lookback_from_env() -> usize {
    mtp_ngram_lookback_from_value(env::var("XRT_QWEN_MTP_NGRAM_LOOKBACK").ok().as_deref())
}

fn ngram_consensus_draft(
    tokens: &[u32],
    order: usize,
    max_draft_tokens: usize,
    min_hits: usize,
    min_percent: usize,
    lookback: usize,
) -> Vec<u32> {
    if order == 0 || max_draft_tokens == 0 || tokens.len() <= order {
        return Vec::new();
    }

    let mut draft = Vec::with_capacity(max_draft_tokens);
    while draft.len() < max_draft_tokens {
        let combined_len = tokens.len().saturating_add(draft.len());
        if combined_len < order {
            break;
        }
        let suffix_start = combined_len - order;
        let mut suffix = Vec::with_capacity(order);
        for index in suffix_start..combined_len {
            suffix.push(if index < tokens.len() {
                tokens[index]
            } else {
                draft[index - tokens.len()]
            });
        }

        // Count the token following every matching suffix.  Tracking the most
        // recent occurrence makes ties deterministic and preserves the useful
        // locality of the original prompt-lookup route.
        let search_end = tokens.len().saturating_sub(order);
        let search_start = search_end.saturating_sub(lookback);
        let mut counts: HashMap<u32, (usize, usize)> = HashMap::new();
        let mut total_hits = 0usize;
        for start in search_start..search_end {
            if tokens[start..start + order] != suffix {
                continue;
            }
            let next = tokens[start + order];
            let entry = counts.entry(next).or_insert((0, start));
            entry.0 = entry.0.saturating_add(1);
            entry.1 = start;
            total_hits = total_hits.saturating_add(1);
        }
        if total_hits < min_hits {
            break;
        }
        let Some((token, (winning_hits, _))) = counts.into_iter().max_by(
            |(left_token, (left_hits, left_position)),
             (right_token, (right_hits, right_position))| {
                left_hits
                    .cmp(right_hits)
                    .then_with(|| left_position.cmp(right_position))
                    .then_with(|| right_token.cmp(left_token))
            },
        ) else {
            break;
        };
        if winning_hits.saturating_mul(100) < min_percent.saturating_mul(total_hits) {
            break;
        }
        draft.push(token);
    }
    draft
}

fn mtp_reuse_dflash_suffix_enabled_from_env() -> bool {
    env::var("XRT_QWEN_MTP_REUSE_DFLASH_SUFFIX")
        .ok()
        .as_deref()
        .and_then(parse_bool)
        .unwrap_or(false)
}

fn pad_hybrid_draft_to_width(draft: &mut Vec<u32>, width: usize) -> usize {
    let acceptance_limit = draft.len().min(width);
    draft.truncate(acceptance_limit);
    if acceptance_limit > 0 {
        // Token zero is never eligible for acceptance beyond
        // `acceptance_limit`; it only supplies a bounded verifier row.
        draft.resize(width, 0);
    }
    acceptance_limit
}

pub(crate) fn mtp_speculation_enabled_from_env() -> bool {
    env::var("XRT_QWEN_MTP")
        .ok()
        .as_deref()
        .and_then(parse_bool)
        .unwrap_or(false)
}

pub(crate) fn mtp_max_draft_tokens_from_env() -> usize {
    env::var("XRT_QWEN_MTP_MAX_DRAFT_TOKENS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(1)
        .clamp(1, 15)
}

pub(crate) fn mtp_adaptive_fallback_enabled_from_env() -> bool {
    env::var("XRT_QWEN_MTP_ADAPTIVE_FALLBACK")
        .ok()
        .as_deref()
        .and_then(parse_bool)
        .unwrap_or(true)
}

fn mtp_should_adaptively_fallback(stats: SpeculativeDecodeStats) -> bool {
    stats.drafted_tokens >= MTP_ADAPTIVE_MIN_DRAFTED_TOKENS
        && stats.accepted_tokens.saturating_mul(4) < stats.drafted_tokens
}

fn mtp_compact_greedy_eligible(config: SamplerConfig) -> bool {
    config.temperature <= 1e-5
        && config.repetition_penalty <= 1.0
        && config.presence_penalty == 0.0
        && config.frequency_penalty == 0.0
}

fn mtp_request_eligible(enabled: bool, has_no_images: bool, temperature: f32) -> bool {
    enabled && has_no_images && temperature <= 1e-5
}

fn elapsed_micros(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX)
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
        argmax, checked_add, logits_for_position, mtp_compact_greedy_eligible,
        mtp_ngram_lookback_from_value, mtp_ngram_min_hits_from_value,
        mtp_ngram_min_percent_from_value, mtp_ngram_order_from_value, mtp_request_eligible,
        mtp_should_adaptively_fallback, ngram_consensus_draft, pad_hybrid_draft_to_width,
        parse_bool, take_embedding_overrides, total_len_after_batch, SamplerConfig,
        SpeculativeDecodeStats,
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
    fn mtp_ngram_order_is_bounded_and_defaults_to_three() {
        assert_eq!(mtp_ngram_order_from_value(None), 3);
        assert_eq!(mtp_ngram_order_from_value(Some("8")), 8);
        assert_eq!(mtp_ngram_order_from_value(Some(" 16 ")), 16);
        for invalid in ["", "2", "33", "invalid"] {
            assert_eq!(mtp_ngram_order_from_value(Some(invalid)), 3);
        }
    }

    #[test]
    fn hybrid_ngram_drafts_preserve_their_acceptance_limit_at_fixed_width() {
        let mut draft = vec![11, 12, 13];
        assert_eq!(pad_hybrid_draft_to_width(&mut draft, 8), 3);
        assert_eq!(draft, vec![11, 12, 13, 0, 0, 0, 0, 0]);

        let mut truncated = vec![1, 2, 3, 4];
        assert_eq!(pad_hybrid_draft_to_width(&mut truncated, 2), 2);
        assert_eq!(truncated, vec![1, 2]);

        let mut empty = Vec::new();
        assert_eq!(pad_hybrid_draft_to_width(&mut empty, 8), 0);
        assert!(empty.is_empty());
    }

    #[test]
    fn consensus_ngram_draft_extends_only_dominant_history_patterns() {
        let repeated = vec![1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3];
        assert_eq!(
            ngram_consensus_draft(&repeated, 3, 5, 2, 66, 8_192),
            vec![4, 1, 2, 3, 4]
        );

        let ambiguous = vec![1, 2, 3, 4, 1, 2, 3, 5, 1, 2, 3];
        assert!(ngram_consensus_draft(&ambiguous, 3, 5, 2, 66, 8_192).is_empty());
        assert_eq!(
            ngram_consensus_draft(&ambiguous, 3, 1, 2, 50, 8_192),
            vec![5]
        );
    }

    #[test]
    fn consensus_ngram_configuration_is_bounded() {
        assert_eq!(mtp_ngram_min_hits_from_value(None), 2);
        assert_eq!(mtp_ngram_min_hits_from_value(Some("4")), 4);
        assert_eq!(mtp_ngram_min_hits_from_value(Some("0")), 2);
        assert_eq!(mtp_ngram_min_percent_from_value(None), 66);
        assert_eq!(mtp_ngram_min_percent_from_value(Some("75")), 75);
        assert_eq!(mtp_ngram_min_percent_from_value(Some("101")), 66);
        assert_eq!(mtp_ngram_lookback_from_value(None), 8_192);
        assert_eq!(mtp_ngram_lookback_from_value(Some("4096")), 4_096);
        assert_eq!(mtp_ngram_lookback_from_value(Some("4")), 8_192);
    }

    #[test]
    fn mtp_adaptive_fallback_requires_one_low_acceptance_probe_window() {
        let stats = |drafted_tokens, accepted_tokens| SpeculativeDecodeStats {
            drafted_tokens,
            accepted_tokens,
            ..SpeculativeDecodeStats::default()
        };
        assert!(!mtp_should_adaptively_fallback(stats(5, 0)));
        assert!(mtp_should_adaptively_fallback(stats(6, 1)));
        assert!(!mtp_should_adaptively_fallback(stats(6, 2)));
        assert!(!mtp_should_adaptively_fallback(stats(30, 23)));
    }

    #[test]
    fn compact_mtp_greedy_requires_unpenalized_argmax() {
        let eligible = SamplerConfig {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            seed: Some(7),
        };
        assert!(mtp_compact_greedy_eligible(eligible));
        assert!(!mtp_compact_greedy_eligible(SamplerConfig {
            repetition_penalty: 1.1,
            ..eligible
        }));
        assert!(!mtp_compact_greedy_eligible(SamplerConfig {
            presence_penalty: 0.25,
            ..eligible
        }));
        assert!(!mtp_compact_greedy_eligible(SamplerConfig {
            temperature: 0.8,
            ..eligible
        }));
    }

    #[test]
    fn mtp_request_eligibility_falls_back_for_sampling_and_images() {
        assert!(mtp_request_eligible(true, true, 0.0));
        assert!(!mtp_request_eligible(false, true, 0.0));
        assert!(!mtp_request_eligible(true, false, 0.0));
        assert!(!mtp_request_eligible(true, true, 0.7));
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
