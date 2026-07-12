pub mod backend;
pub mod gpu_resource;
pub mod grammar;
pub mod kv_cache;
pub mod policy;
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
use xrt_models::{LlamaModel, VisionEncoder};
use xrt_tokenizer::Tokenizer;

pub use backend::{
    BackendDecodeBatchExecution, BackendDecodeBatchItem, BackendKind, BackendSession,
    CausalLmBackend, CpuBackend, CudaResidentBackend,
};
pub use gpu_resource::{CudaGraphMode, GpuResourceConfig, GpuResourceManager, GpuResourceStatus};
pub use grammar::Grammar;
pub use kv_cache::{
    KeyQ4ValueQ8PagedKvCache, KvCacheMode, PagedKvCache, QuantizedPagedKvCache, SessionKvCache,
};
pub use policy::{CachePolicyKind, PromptSpan, PromptSpanKind, SessionPolicy};
pub use sampler::{Sampler, SamplerConfig};
pub use scheduler::{
    RequestScheduler, SchedulerAcquireError, SchedulerConfig, SchedulerExecutionPermit,
    SchedulerExecutionPhase, SchedulerKvReservation, SchedulerPermit, SchedulerPrefillRegistration,
    SchedulerStatus,
};
pub use session::{GenerateRequest, Session};

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
    model: Arc<LlamaModel>,
    tokenizer: Arc<Tokenizer>,
    vision: Option<Arc<VisionEncoder>>,
    active_sessions: Arc<AtomicUsize>,
}

impl Runtime {
    pub fn load(model_path: impl AsRef<Path>) -> Result<Arc<Self>> {
        Self::load_with_backend(model_path, BackendKind::from_env())
    }

    pub fn load_with_backend(
        model_path: impl AsRef<Path>,
        requested_backend: BackendKind,
    ) -> Result<Arc<Self>> {
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
        let gguf = Arc::new(GgufFile::open(model_path)?);
        Self::from_gguf_with_backend(gguf, requested_backend, active_backend)
    }

    pub fn from_gguf(gguf: Arc<GgufFile>) -> Result<Arc<Self>> {
        Self::from_gguf_with_backend(gguf, BackendKind::Cpu, BackendKind::Cpu)
    }

    fn from_gguf_with_backend(
        gguf: Arc<GgufFile>,
        requested_backend: BackendKind,
        active_backend: BackendKind,
    ) -> Result<Arc<Self>> {
        let tokenizer = Arc::new(Tokenizer::from_gguf(&gguf)?);
        let model = Arc::new(LlamaModel::from_gguf(gguf.clone())?);
        let gpu_resources = Arc::new(GpuResourceManager::from_env());
        let cpu_backend = || Arc::new(CpuBackend::new(model.clone())) as Arc<dyn CausalLmBackend>;
        let (active_backend, backend): (BackendKind, Arc<dyn CausalLmBackend>) =
            match active_backend {
                BackendKind::Cpu => (BackendKind::Cpu, cpu_backend()),
                BackendKind::CudaResident => (
                    BackendKind::CudaResident,
                    Arc::new(CudaResidentBackend::new(
                        model.clone(),
                        &gguf,
                        gpu_resources.config(),
                    )?),
                ),
                BackendKind::Auto => {
                    #[cfg(feature = "cuda")]
                    {
                        if CudaResidentBackend::supports_dense_quant_decode(&gguf, model.config()) {
                            match CudaResidentBackend::new(
                                model.clone(),
                                &gguf,
                                gpu_resources.config(),
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
        Ok(Arc::new(Self {
            requested_backend,
            active_backend,
            backend,
            gpu_resources,
            model,
            tokenizer,
            vision: None,
            active_sessions: Arc::new(AtomicUsize::new(0)),
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
        }))
    }

    pub fn model(&self) -> &LlamaModel {
        self.model.as_ref()
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
        self.gpu_resource_status_with_session_allocations(0, 0, None, None, None)
    }

    pub(crate) fn gpu_resource_status_with_session_allocations(
        &self,
        kv_allocated_bytes: u64,
        scratch_allocated_bytes: u64,
        requested_kv_cache_mode: Option<KvCacheMode>,
        kv_cache_mode: Option<KvCacheMode>,
        graph_capture: Option<&'static str>,
    ) -> GpuResourceStatus {
        let mut status = self.gpu_resources.status_with_allocations_and_probe(
            self.backend.model_weight_bytes(),
            kv_allocated_bytes,
            scratch_allocated_bytes,
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
        status.free_vram_bytes = self.backend.cuda_free_vram_bytes();
        status.total_vram_bytes = self.backend.cuda_total_vram_bytes();
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
        self.model.model_name()
    }

    pub fn model_architecture(&self) -> &str {
        &self.model.config().architecture
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
