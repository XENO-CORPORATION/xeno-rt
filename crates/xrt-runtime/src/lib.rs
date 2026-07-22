pub mod backend;
pub mod expert_placement;
pub mod gpu_resource;
pub mod grammar;
pub mod kv_cache;
pub mod moe;
pub mod moe_config;
mod moe_manifest;
pub mod policy;
pub mod prefix_cache;
pub mod recurrent_state;
mod resident_tensor;
pub mod sampler;
pub mod scheduler;
pub mod session;

use std::{
    path::Path,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};
use xrt_core::{Result, XrtError};
use xrt_gguf::GgufFile;
use xrt_models::{LlamaModel, VisionEncoder, DELTANET_STATE_SNAPSHOT_VERSION};
#[cfg(feature = "moe-route-trace")]
pub use xrt_models::{MoeRouteTrace, MoeRouteTraceEntry};
use xrt_safetensors::HfModelBundle;
use xrt_tokenizer::Tokenizer;

pub use backend::{
    BackendDecodeBatchExecution, BackendDecodeBatchItem, BackendKind, BackendSession,
    CausalLmBackend, CpuBackend, CudaResidentBackend,
};
pub use expert_placement::{ExpertPlacementManager, ExpertPlacementSnapshot};
pub use gpu_resource::{
    CudaGraphMode, GpuAllocationArena, GpuAllocationBreakdown, GpuAllocationClass,
    GpuAllocationDelta, GpuAllocationLease, GpuAllocationSnapshot, GpuAllocationStats,
    GpuMemoryPoolStats, GpuResourceConfig, GpuResourceManager, GpuResourceStatus, GpuTransferStats,
};
pub use grammar::Grammar;
pub use kv_cache::{
    KeyQ4ValueQ8PagedKvCache, KvCacheMode, PagedKvCache, QuantizedPagedKvCache, SessionKvCache,
};
pub use moe::{build_moe_execution_plan, MoeExecutionPlan, MoeWorkItem};
pub use moe_config::{MoeAcceleration, MoePlacementPolicy, MoeRuntimeConfig, MoeRuntimeStatus};
pub use moe_manifest::moe_config_sha256;
pub use policy::{CachePolicyKind, PromptSpan, PromptSpanKind, SessionPolicy};
pub use prefix_cache::{PrefixCacheConfig, PrefixCacheManager, PrefixCacheStatus};
pub use recurrent_state::CudaDeltaNetState;
pub use sampler::{Sampler, SamplerConfig};
pub use scheduler::{
    RequestScheduler, SchedulerAcquireError, SchedulerConfig, SchedulerExecutionPermit,
    SchedulerExecutionPhase, SchedulerKvReservation, SchedulerPermit, SchedulerPrefillRegistration,
    SchedulerStatus,
};
pub use session::{GenerateRequest, HybridRuntimeStatus, Session, SpeculativeDecodeStats};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisionPromptLayout {
    pub patch_token_piece: String,
    pub patch_token_id: u32,
    pub start_token_piece: Option<String>,
    pub start_token_id: Option<u32>,
    pub end_token_piece: Option<String>,
    pub end_token_id: Option<u32>,
    pub patches_per_image: usize,
}

impl VisionPromptLayout {
    pub fn prompt_fragment(&self) -> String {
        let mut fragment = String::new();
        if let Some(start) = &self.start_token_piece {
            fragment.push_str(start);
        }
        for _ in 0..self.patches_per_image {
            fragment.push_str(&self.patch_token_piece);
        }
        if let Some(end) = &self.end_token_piece {
            fragment.push_str(end);
        }
        fragment
    }
}

pub struct Runtime {
    requested_backend: BackendKind,
    active_backend: BackendKind,
    backend: Arc<dyn CausalLmBackend>,
    gpu_resources: Arc<GpuResourceManager>,
    model: Option<Arc<LlamaModel>>,
    tokenizer: Arc<Tokenizer>,
    vision: Option<Arc<VisionEncoder>>,
    active_sessions: Arc<AtomicUsize>,
    prefix_cache: Arc<PrefixCacheManager>,
    moe_config: MoeRuntimeConfig,
    moe_status: MoeRuntimeStatus,
}

impl Runtime {
    pub fn load(model_path: impl AsRef<Path>) -> Result<Arc<Self>> {
        Self::load_with_backend(model_path, BackendKind::from_env())
    }

    pub fn load_with_backend(
        model_path: impl AsRef<Path>,
        requested_backend: BackendKind,
    ) -> Result<Arc<Self>> {
        Self::load_with_backend_and_moe_config(
            model_path,
            requested_backend,
            MoeRuntimeConfig::from_env()?,
        )
    }

    pub fn load_with_backend_and_moe_config(
        model_path: impl AsRef<Path>,
        requested_backend: BackendKind,
        moe_config: MoeRuntimeConfig,
    ) -> Result<Arc<Self>> {
        Self::load_with_backend_configs(
            model_path,
            requested_backend,
            moe_config,
            GpuResourceConfig::from_env(),
        )
    }

    /// Load a runtime with fully resolved local execution configuration.
    ///
    /// This is the programmatic-precedence entry point used by benchmarks and
    /// embedders that must not mutate process-global environment variables.
    pub fn load_with_backend_configs(
        model_path: impl AsRef<Path>,
        requested_backend: BackendKind,
        moe_config: MoeRuntimeConfig,
        gpu_config: GpuResourceConfig,
    ) -> Result<Arc<Self>> {
        let gpu_resources = Arc::new(GpuResourceManager::new(gpu_config));
        Self::load_with_backend_configs_and_resource_manager(
            model_path,
            requested_backend,
            moe_config,
            gpu_config,
            gpu_resources,
        )
    }

    /// Load with a caller-owned resource manager shared by all runtimes on a
    /// CUDA device. Existing constructors retain their isolated-manager
    /// behavior and delegate through this additive entry point.
    pub fn load_with_backend_configs_and_resource_manager(
        model_path: impl AsRef<Path>,
        requested_backend: BackendKind,
        moe_config: MoeRuntimeConfig,
        gpu_config: GpuResourceConfig,
        gpu_resources: Arc<GpuResourceManager>,
    ) -> Result<Arc<Self>> {
        gpu_resources.validate_compatible_config(gpu_config)?;
        let active_backend = match requested_backend {
            BackendKind::Auto => BackendKind::Auto,
            other => other.resolve_active()?,
        };
        if active_backend == BackendKind::CudaResident && !cfg!(feature = "cuda") {
            return Err(XrtError::Cuda(
                "CUDA backend requested but xrt-runtime was built without the `cuda` feature"
                    .to_string(),
            ));
        }
        let model_path = model_path.as_ref();
        if model_path.is_dir() {
            if matches!(active_backend, BackendKind::Cpu) {
                return Err(XrtError::Unsupported(
                    "SafeTensors model directories currently require --backend cuda or XRT_BACKEND=cuda; CPU SafeTensors decode is not implemented"
                        .to_string(),
                ));
            }
            if !cfg!(feature = "cuda") {
                return Err(XrtError::Cuda(
                    "SafeTensors model directories currently require a CUDA-enabled xrt-runtime build"
                        .to_string(),
                ));
            }
            return Self::from_hf_with_backend(
                model_path,
                requested_backend,
                moe_config,
                gpu_resources,
            );
        }
        let gguf = Arc::new(GgufFile::open(model_path)?);
        Self::from_gguf_with_backend(
            gguf,
            requested_backend,
            active_backend,
            moe_config,
            gpu_resources,
        )
    }

    fn from_hf_with_backend(
        model_path: &Path,
        requested_backend: BackendKind,
        moe_config: MoeRuntimeConfig,
        gpu_resources: Arc<GpuResourceManager>,
    ) -> Result<Arc<Self>> {
        let bundle = HfModelBundle::open(model_path)?;
        let tokenizer = Arc::new(Tokenizer::from_hf_dir(model_path)?);
        let backend: Arc<dyn CausalLmBackend> =
            Arc::new(CudaResidentBackend::from_hf_bundle_with_resource_manager(
                &bundle,
                Arc::clone(&gpu_resources),
            )?);
        let active_backend = BackendKind::CudaResident;
        moe_config.resolve_backend(requested_backend, active_backend, false)?;
        let moe_status = moe_config.status(false, active_backend)?;
        let prefix_cache_namespace = format!(
            "{}:{}:{}:{}:{}:{}",
            backend.model_name(),
            backend.config().architecture,
            backend.config().block_count,
            backend.config().embedding_length,
            tokenizer.vocab_size(),
            active_backend.as_str(),
        );
        Ok(Arc::new(Self {
            requested_backend,
            active_backend,
            backend,
            gpu_resources,
            model: None,
            tokenizer,
            vision: None,
            active_sessions: Arc::new(AtomicUsize::new(0)),
            prefix_cache: Arc::new(PrefixCacheManager::from_env(prefix_cache_namespace)),
            moe_config,
            moe_status,
        }))
    }

    pub fn from_gguf(gguf: Arc<GgufFile>) -> Result<Arc<Self>> {
        let gpu_config = GpuResourceConfig::from_env();
        Self::from_gguf_with_backend(
            gguf,
            BackendKind::Cpu,
            BackendKind::Cpu,
            MoeRuntimeConfig::default(),
            Arc::new(GpuResourceManager::new(gpu_config)),
        )
    }

    fn from_gguf_with_backend(
        gguf: Arc<GgufFile>,
        requested_backend: BackendKind,
        active_backend: BackendKind,
        moe_config: MoeRuntimeConfig,
        gpu_resources: Arc<GpuResourceManager>,
    ) -> Result<Arc<Self>> {
        let tokenizer = Arc::new(Tokenizer::from_gguf(&gguf)?);
        let model = Arc::new(LlamaModel::from_gguf_with_moe_execution(
            gguf.clone(),
            moe_config.model_cpu_execution(),
        )?);
        let active_backend = moe_config.resolve_backend(
            requested_backend,
            active_backend,
            model.config().is_moe(),
        )?;
        if active_backend == BackendKind::CudaResident && !cfg!(feature = "cuda") {
            return Err(XrtError::Cuda(
                "CUDA MoE backend requested but xrt-runtime was built without the `cuda` feature"
                    .to_string(),
            ));
        }
        let cpu_backend = || Arc::new(CpuBackend::new(model.clone())) as Arc<dyn CausalLmBackend>;
        let (active_backend, backend): (BackendKind, Arc<dyn CausalLmBackend>) =
            match active_backend {
                BackendKind::Cpu => (BackendKind::Cpu, cpu_backend()),
                BackendKind::CudaResident => (
                    BackendKind::CudaResident,
                    if model.config().is_moe()
                        && matches!(
                            moe_config.acceleration,
                            MoeAcceleration::Hybrid | MoeAcceleration::Gpu
                        )
                    {
                        Arc::new(CudaResidentBackend::new_moe_with_resource_manager(
                            model.clone(),
                            Arc::clone(&gguf),
                            Arc::clone(&gpu_resources),
                            &moe_config,
                        )?)
                    } else {
                        Arc::new(CudaResidentBackend::new_with_resource_manager(
                            model.clone(),
                            &gguf,
                            Arc::clone(&gpu_resources),
                        )?)
                    },
                ),
                BackendKind::Auto => {
                    #[cfg(feature = "cuda")]
                    {
                        // Qwen3.5 CUDA is explicit-only until its real-model
                        // throughput/latency gate is recorded on the target
                        // hardware. Correctness parity alone does not change
                        // `auto` behavior.
                        if !model.config().is_hybrid()
                            && CudaResidentBackend::supports_dense_quant_decode(
                                &gguf,
                                model.config(),
                            )
                        {
                            match CudaResidentBackend::new_with_resource_manager(
                                model.clone(),
                                &gguf,
                                Arc::clone(&gpu_resources),
                            ) {
                                Ok(cuda_backend) => (
                                    BackendKind::CudaResident,
                                    Arc::new(cuda_backend) as Arc<dyn CausalLmBackend>,
                                ),
                                Err(err) => {
                                    tracing::warn!(
                                        "auto backend falling back to CPU after CUDA load failed: {err}"
                                    );
                                    (BackendKind::Cpu, cpu_backend())
                                }
                            }
                        } else {
                            (BackendKind::Cpu, cpu_backend())
                        }
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        (BackendKind::Cpu, cpu_backend())
                    }
                }
                BackendKind::ExternalOpenAi => {
                    unreachable!("backend selection should resolve before runtime construction")
                }
            };
        let moe_status = moe_config.status(model.config().is_moe(), active_backend)?;
        let prefix_cache_namespace = format!(
            "{}:{}:{}:{}:{}:{}",
            model.model_name(),
            model.config().architecture,
            model.config().block_count,
            model.config().embedding_length,
            tokenizer.vocab_size(),
            active_backend.as_str(),
        );
        Ok(Arc::new(Self {
            requested_backend,
            active_backend,
            backend,
            gpu_resources,
            model: Some(model),
            tokenizer,
            vision: None,
            active_sessions: Arc::new(AtomicUsize::new(0)),
            prefix_cache: Arc::new(PrefixCacheManager::from_env(prefix_cache_namespace)),
            moe_config,
            moe_status,
        }))
    }

    /// Load a multimodal projection (mmproj) GGUF for vision support.
    pub fn load_vision(self: &Arc<Self>, mmproj_path: &str) -> Result<Arc<Self>> {
        let encoder = VisionEncoder::load(mmproj_path)?;
        Ok(Arc::new(Self {
            requested_backend: self.requested_backend,
            active_backend: self.active_backend,
            backend: self.backend.clone(),
            gpu_resources: self.gpu_resources.clone(),
            model: self.model.clone(),
            tokenizer: self.tokenizer.clone(),
            vision: Some(Arc::new(encoder)),
            active_sessions: self.active_sessions.clone(),
            prefix_cache: self.prefix_cache.clone(),
            moe_config: self.moe_config.clone(),
            moe_status: self.moe_status.clone(),
        }))
    }

    pub fn model(&self) -> &LlamaModel {
        self.model.as_deref().expect(
            "Runtime::model is unavailable for a SafeTensors-backed CUDA runtime; use Runtime::backend/config metadata instead",
        )
    }

    pub fn cpu_model(&self) -> Option<&LlamaModel> {
        self.model.as_deref()
    }

    pub fn requested_backend(&self) -> BackendKind {
        self.requested_backend
    }

    pub fn active_backend(&self) -> BackendKind {
        self.active_backend
    }

    pub fn backend(&self) -> &dyn CausalLmBackend {
        self.backend.as_ref()
    }

    pub(crate) fn backend_arc(&self) -> Arc<dyn CausalLmBackend> {
        self.backend.clone()
    }

    pub fn gpu_resource_status(&self) -> GpuResourceStatus {
        self.gpu_resource_status_with_session_allocations(0, 0, 0, None, None, None)
    }

    pub fn gpu_resource_manager(&self) -> Arc<GpuResourceManager> {
        Arc::clone(&self.gpu_resources)
    }

    pub fn gpu_transfer_stats(&self) -> Option<GpuTransferStats> {
        self.backend.cuda_transfer_stats().map(Into::into)
    }

    pub fn gpu_allocation_stats(&self) -> Option<GpuAllocationStats> {
        self.backend.cuda_allocation_stats().map(Into::into)
    }

    pub fn reset_gpu_allocation_peak(&self) {
        self.backend.reset_cuda_allocation_peak();
    }

    pub fn moe_config(&self) -> &MoeRuntimeConfig {
        &self.moe_config
    }

    pub fn moe_status(&self) -> MoeRuntimeStatus {
        let mut status = self.moe_status.clone();
        status.placement_generation = self.backend.moe_placement_generation();
        status.placement_manifest_sha256 = self
            .backend
            .moe_placement_manifest_sha256()
            .map(str::to_string);
        status.gpu_expert_slots = self.backend.moe_gpu_expert_slots();
        status.gpu_expert_bytes = self.backend.moe_gpu_expert_bytes();
        let cuda_telemetry = self.backend.cuda_moe_telemetry();
        status.cpu_expert_calls = cuda_telemetry.cpu_expert_calls;
        status.gpu_expert_calls = cuda_telemetry.gpu_expert_calls;
        status.gpu_placement_hits = cuda_telemetry.gpu_placement_hits;
        status.gpu_placement_misses = cuda_telemetry.gpu_placement_misses;
        status.activation_d2h_bytes = cuda_telemetry.activation_d2h_bytes;
        status.result_h2d_bytes = cuda_telemetry.result_h2d_bytes;
        status.coordinator_failures = cuda_telemetry.coordinator_failures;
        status.graph_eager_expert_calls = cuda_telemetry.graph_eager_expert_calls;
        status.graph_captures = cuda_telemetry.graph_captures;
        status.graph_replays = cuda_telemetry.graph_replays;
        status.graph_fallbacks = cuda_telemetry.graph_fallbacks;
        status.placement_evaluations = cuda_telemetry.placement_evaluations;
        status.placement_updates = cuda_telemetry.placement_updates;
        status.placement_moves = cuda_telemetry.placement_moves;
        status.placement_upload_bytes = cuda_telemetry.placement_upload_bytes;
        status.placement_update_micros = cuda_telemetry.placement_update_micros;
        status.placement_last_update_micros = cuda_telemetry.placement_last_update_micros;
        status.layerwise_prefill_batches = cuda_telemetry.layerwise_prefill_batches;
        status.layerwise_prefill_tokens = cuda_telemetry.layerwise_prefill_tokens;
        status.layerwise_prefill_weight_upload_bytes =
            cuda_telemetry.layerwise_prefill_weight_upload_bytes;
        status.layerwise_prefill_repack_bytes = cuda_telemetry.layerwise_prefill_repack_bytes;
        status.layerwise_prefill_micros = cuda_telemetry.layerwise_prefill_micros;
        if let Some(model) = &self.model {
            let telemetry = model.moe_telemetry();
            status.routed_tokens = telemetry.routed_tokens;
            status.selected_expert_calls = telemetry.selected_expert_calls;
            status.legacy_batches = telemetry.legacy_batches;
            status.grouped_batches = telemetry.grouped_batches;
            status.grouped_tokens = telemetry.grouped_tokens;
            status.worker_failures = telemetry.worker_failures;
            status.expert_call_counts = telemetry.expert_call_counts;
        }
        status
    }

    pub fn hybrid_state_status(&self) -> Option<HybridRuntimeStatus> {
        let config = self.backend.config();
        let descriptor = config.deltanet_state_descriptor()?;
        let recurrent_layers = descriptor.layers().iter().flatten().count();
        let full_attention_layers = config.block_count.saturating_sub(recurrent_layers);
        let durable_snapshot_bytes = u64::try_from(descriptor.allocated_f32_elements().ok()?)
            .unwrap_or(u64::MAX)
            .saturating_mul(std::mem::size_of::<f32>() as u64);
        let cuda = self.active_backend == BackendKind::CudaResident;
        let speculative_rollback_supported = cuda;
        let speculation_requested = session::ngram_speculation_enabled_from_env();
        let speculative_decoding_enabled = speculation_requested && speculative_rollback_supported;
        let speculative_decoding_disabled_reason = if speculative_decoding_enabled {
            None
        } else if !speculation_requested {
            Some("disabled by XRT_NGRAM_SPECULATION".to_string())
        } else {
            Some("the active backend has no device-local recurrent checkpoint journal".to_string())
        };

        Some(HybridRuntimeStatus {
            owner: "session",
            backend: self.active_backend.as_str().to_string(),
            state_format_version: DELTANET_STATE_SNAPSHOT_VERSION,
            recurrent_layers,
            full_attention_layers,
            durable_snapshot_bytes,
            bytes_per_session: durable_snapshot_bytes.saturating_mul(if cuda { 3 } else { 2 }),
            prefix_cache_supported: true,
            prefix_cache_enabled: self.prefix_cache.status().enabled,
            shared_f32_kv_page_cow_supported: cuda,
            quantized_kv_page_cow_supported: false,
            speculative_rollback_supported,
            speculative_decoding_enabled,
            speculative_decoding_disabled_reason,
        })
    }

    pub(crate) fn gpu_resource_status_with_session_allocations(
        &self,
        kv_allocated_bytes: u64,
        scratch_allocated_bytes: u64,
        staging_allocated_bytes: u64,
        requested_kv_cache_mode: Option<KvCacheMode>,
        kv_cache_mode: Option<KvCacheMode>,
        graph_capture: Option<&'static str>,
    ) -> GpuResourceStatus {
        let mut status = self
            .gpu_resources
            .status_with_allocations_staging_and_probe(
                self.backend.model_weight_bytes(),
                kv_allocated_bytes,
                scratch_allocated_bytes,
                staging_allocated_bytes,
                self.active_sessions.load(Ordering::Relaxed),
                self.active_backend == BackendKind::CudaResident,
                self.backend.resident_f32_probe_available(),
                self.backend.resident_q8_0_probe_available(),
                self.backend.resident_q8_0_layer0_probe_available(),
                self.backend.resident_dense_quant_decode_available(),
            );
        status.requested_kv_cache_mode = requested_kv_cache_mode.map(KvCacheMode::as_str);
        status.kv_cache_mode = kv_cache_mode.map(KvCacheMode::as_str);
        status.device_name = self.backend.cuda_device_name().map(str::to_string);
        if let Some((free_vram_bytes, total_vram_bytes)) = self.backend.cuda_memory_info() {
            status.free_vram_bytes = Some(free_vram_bytes);
            status.total_vram_bytes = Some(total_vram_bytes);
            status.device_used_vram_bytes = Some(total_vram_bytes.saturating_sub(free_vram_bytes));
        }
        status.transfer_totals = self.gpu_transfer_stats();
        status.allocation_totals = self.gpu_allocation_stats();
        status.memory_pool = self.backend.cuda_memory_pool_stats().map(Into::into);
        status.kv_budget_bytes = self.backend.cuda_kv_budget_bytes();
        if let Some(graph_capture) = graph_capture {
            status.graph_capture = graph_capture;
        }
        status
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        self.tokenizer.as_ref()
    }

    pub fn model_name(&self) -> &str {
        self.backend.model_name()
    }

    pub fn model_architecture(&self) -> &str {
        &self.backend.config().architecture
    }

    pub fn vision(&self) -> Option<&VisionEncoder> {
        self.vision.as_deref()
    }

    pub fn vision_prompt_layout(&self) -> Option<VisionPromptLayout> {
        let vision = self.vision()?;
        let tokenizer = self.tokenizer();

        for (patch_piece, start_piece, end_piece) in [
            (
                "<|image_pad|>",
                Some("<|vision_start|>"),
                Some("<|vision_end|>"),
            ),
            ("<image>", None, None),
            ("<|image|>", None, None),
            ("<image_pad>", None, None),
        ] {
            let patch_token_id = match tokenizer.token_id_for_piece(patch_piece) {
                Some(id) => id,
                None => continue,
            };
            let start_token_id = start_piece.and_then(|piece| tokenizer.token_id_for_piece(piece));
            let end_token_id = end_piece.and_then(|piece| tokenizer.token_id_for_piece(piece));
            let use_wrappers = start_piece.is_some()
                && end_piece.is_some()
                && start_token_id.is_some()
                && end_token_id.is_some();

            return Some(VisionPromptLayout {
                patch_token_piece: patch_piece.to_string(),
                patch_token_id,
                start_token_piece: use_wrappers.then(|| start_piece.unwrap().to_string()),
                start_token_id: use_wrappers.then_some(start_token_id.unwrap()),
                end_token_piece: use_wrappers.then(|| end_piece.unwrap().to_string()),
                end_token_id: use_wrappers.then_some(end_token_id.unwrap()),
                patches_per_image: vision.config().patch_count,
            });
        }

        None
    }

    pub fn new_session(self: &Arc<Self>) -> Session {
        Session::new(self.clone())
    }

    pub fn new_session_with_cache_mode(self: &Arc<Self>, mode: KvCacheMode) -> Session {
        Session::new_with_cache_mode(self.clone(), mode)
    }

    pub fn prefix_cache_status(&self) -> PrefixCacheStatus {
        self.prefix_cache.status()
    }

    pub fn clear_prefix_cache(&self) {
        self.prefix_cache.clear();
    }

    pub(crate) fn prefix_cache(&self) -> &PrefixCacheManager {
        self.prefix_cache.as_ref()
    }

    pub(crate) fn register_session(&self) {
        self.active_sessions.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn unregister_session(&self) {
        let _ = self
            .active_sessions
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |count| {
                count.checked_sub(1)
            });
    }
}
