use std::{collections::HashMap, env, fmt, sync::Arc, time::Instant};

use crate::{
    gpu_resource::GpuResourceConfig,
    kv_cache::{KvCacheMode, SessionKvCache},
    policy::{PromptSpan, SessionPolicy},
};
use serde::{Deserialize, Serialize};
use tracing::info;
use xrt_core::{checked_mul, DType, KvCache, Result, XrtError};
use xrt_cuda::{
    CudaDevice, CudaF32Buffer, CudaKeyQ4ValueQ8LayerKvCache, CudaLayerKvCache, CudaQ4KMatrix,
    CudaQ4_0Matrix, CudaQ5KMatrix, CudaQ6KMatrix, CudaQ8LayerKvCache, CudaQ8_0Matrix, GpuF32Tensor,
};
use xrt_gguf::GgufFile;
use xrt_models::{LlamaConfig, LlamaModel};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BackendKind {
    Auto,
    Cpu,
    CudaResident,
    ExternalOpenAi,
}

impl BackendKind {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "cpu" => Some(Self::Cpu),
            "cuda" | "cuda-resident" | "cuda_resident" | "gpu" => Some(Self::CudaResident),
            "external-openai" | "external_openai" | "openai" | "proxy" => {
                Some(Self::ExternalOpenAi)
            }
            _ => None,
        }
    }

    pub fn from_env() -> Self {
        env::var("XRT_BACKEND")
            .ok()
            .as_deref()
            .and_then(Self::parse)
            .unwrap_or(Self::Auto)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::CudaResident => "cuda-resident",
            Self::ExternalOpenAi => "external-openai",
        }
    }

    pub(crate) fn resolve_active(self) -> Result<Self> {
        match self {
            Self::Auto | Self::Cpu => Ok(Self::Cpu),
            Self::CudaResident => Ok(Self::CudaResident),
            Self::ExternalOpenAi => Err(XrtError::Unsupported(
                "external-openai backend is not implemented in xrt-runtime yet".to_string(),
            )),
        }
    }
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

pub type BackendStateSnapshot = Vec<Option<(Vec<f32>, Vec<f32>)>>;

#[derive(Debug)]
pub enum CudaLayerKvStore {
    F32(CudaLayerKvCache),
    Q8(CudaQ8LayerKvCache),
    KeyQ4ValueQ8(CudaKeyQ4ValueQ8LayerKvCache),
    AgentAdaptive {
        hot: CudaLayerKvCache,
        cold: CudaKeyQ4ValueQ8LayerKvCache,
        hot_mask: Vec<u8>,
    },
}

impl CudaLayerKvStore {
    fn allocate(
        device: &CudaDevice,
        mode: KvCacheMode,
        capacity: usize,
        width: usize,
    ) -> Result<Self> {
        match mode {
            KvCacheMode::Q8 => device
                .alloc_q8_layer_kv_cache(capacity, width)
                .map(Self::Q8),
            KvCacheMode::KeyQ4ValueQ8 => device
                .alloc_key_q4_value_q8_layer_kv_cache(capacity, width)
                .map(Self::KeyQ4ValueQ8),
            KvCacheMode::AgentAdaptive => Ok(Self::AgentAdaptive {
                hot: device.alloc_layer_kv_cache(capacity, width)?,
                cold: device.alloc_key_q4_value_q8_layer_kv_cache(capacity, width)?,
                hot_mask: Vec::with_capacity(capacity),
            }),
            _ => device.alloc_layer_kv_cache(capacity, width).map(Self::F32),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::F32(cache) => cache.len(),
            Self::Q8(cache) => cache.len(),
            Self::KeyQ4ValueQ8(cache) => cache.len(),
            Self::AgentAdaptive { hot_mask, .. } => hot_mask.len(),
        }
    }

    fn capacity(&self) -> usize {
        match self {
            Self::F32(cache) => cache.capacity(),
            Self::Q8(cache) => cache.capacity(),
            Self::KeyQ4ValueQ8(cache) => cache.capacity(),
            Self::AgentAdaptive { hot, cold, .. } => hot.capacity().min(cold.capacity()),
        }
    }

    fn grow(&mut self, device: &CudaDevice, new_capacity: usize) -> Result<()> {
        match self {
            Self::F32(cache) => device.grow_layer_kv_cache(cache, new_capacity),
            Self::Q8(cache) => device.grow_q8_layer_kv_cache(cache, new_capacity),
            Self::KeyQ4ValueQ8(cache) => {
                device.grow_key_q4_value_q8_layer_kv_cache(cache, new_capacity)
            }
            Self::AgentAdaptive { hot, cold, .. } => {
                device.grow_key_q4_value_q8_layer_kv_cache(cold, new_capacity)?;
                device.grow_layer_kv_cache(hot, new_capacity)
            }
        }
    }

    fn clear(&mut self) {
        match self {
            Self::F32(cache) => cache.clear(),
            Self::Q8(cache) => cache.clear(),
            Self::KeyQ4ValueQ8(cache) => cache.clear(),
            Self::AgentAdaptive {
                hot,
                cold,
                hot_mask,
            } => {
                hot.clear();
                cold.clear();
                hot_mask.clear();
            }
        }
    }

    fn truncate(&mut self, new_len: usize) {
        match self {
            Self::F32(cache) => cache.truncate(new_len),
            Self::Q8(cache) => cache.truncate(new_len),
            Self::KeyQ4ValueQ8(cache) => cache.truncate(new_len),
            Self::AgentAdaptive {
                hot,
                cold,
                hot_mask,
            } => {
                let retained = new_len.min(hot_mask.len());
                let hot_len = hot_mask[..retained]
                    .iter()
                    .filter(|&&is_hot| is_hot != 0)
                    .count();
                hot.truncate(hot_len);
                cold.truncate(retained - hot_len);
                hot_mask.truncate(retained);
            }
        }
    }

    fn allocated_bytes(&self) -> u64 {
        match self {
            Self::F32(cache) => cache.allocated_bytes(),
            Self::Q8(cache) => cache.allocated_bytes(),
            Self::KeyQ4ValueQ8(cache) => cache.allocated_bytes(),
            Self::AgentAdaptive { hot, cold, .. } => {
                hot.allocated_bytes().saturating_add(cold.allocated_bytes())
            }
        }
    }

    fn migrate_agent_adaptive_route(
        &mut self,
        device: &CudaDevice,
        desired_hot_mask: &[u8],
    ) -> Result<()> {
        let Self::AgentAdaptive {
            hot,
            cold,
            hot_mask,
        } = self
        else {
            return Ok(());
        };
        if !BackendSession::cuda_adaptive_route_migration_needed(hot_mask, desired_hot_mask) {
            return Ok(());
        }
        if desired_hot_mask.len() < hot_mask.len() {
            return Err(XrtError::Runtime(format!(
                "CUDA agent_adaptive desired route length {} is shorter than cache length {}",
                desired_hot_mask.len(),
                hot_mask.len()
            )));
        }

        let capacity = hot.capacity();
        let width = hot.width();
        let current_hot_mask = hot_mask.clone();
        let mut rebuilt_hot = device.alloc_layer_kv_cache(capacity, width)?;
        let mut rebuilt_cold = device.alloc_key_q4_value_q8_layer_kv_cache(capacity, width)?;
        let mut source_hot_position = 0usize;
        let mut source_cold_position = 0usize;

        // ponytail: row-by-row rebuild is correctness-first; replace with a GPU gather kernel when it shows up in profiling.
        for (position, &was_hot) in current_hot_mask.iter().enumerate() {
            let (key, value) = if was_hot != 0 {
                let row = device.copy_layer_kv(hot, source_hot_position)?;
                source_hot_position += 1;
                row
            } else {
                let row = device.dequantize_key_q4_value_q8_layer_kv(cold, source_cold_position)?;
                source_cold_position += 1;
                row
            };

            if desired_hot_mask[position] != 0 {
                device.append_layer_kv(&mut rebuilt_hot, &key, &value)?;
            } else {
                device.append_key_q4_value_q8_layer_kv(&mut rebuilt_cold, &key, &value)?;
            }
        }

        *hot = rebuilt_hot;
        *cold = rebuilt_cold;
        *hot_mask = desired_hot_mask[..current_hot_mask.len()].to_vec();
        Ok(())
    }
}

#[derive(Debug)]
pub enum BackendSession {
    Cpu {
        cache: SessionKvCache,
    },
    Cuda {
        device: CudaDevice,
        requested_cache_mode: KvCacheMode,
        cache_mode: KvCacheMode,
        layer_caches: Vec<CudaLayerKvStore>,
        layer_count: usize,
        width: usize,
        max_len: usize,
        page_tokens: usize,
        kv_budget_bytes: Option<u64>,
        policy: SessionPolicy,
        prompt_token_count: usize,
        prompt_spans: Vec<PromptSpan>,
    },
}

impl BackendSession {
    fn cuda_adaptive_position_is_hot_for_policy(
        policy: &SessionPolicy,
        prompt_token_count: usize,
        spans: &[PromptSpan],
        position: usize,
        total_len: usize,
    ) -> bool {
        let recent_window = policy.recent_window_tokens.max(1);
        let recent_start = total_len.saturating_sub(recent_window);
        if position >= recent_start {
            return true;
        }

        spans.iter().any(|span| {
            if !policy.is_span_pinned(span.kind) {
                return false;
            }
            let end = span.token_end.min(prompt_token_count);
            let start = span.token_start.min(end);
            position >= start && position < end
        })
    }

    fn cuda_adaptive_hot_position_mask_for_policy(
        policy: &SessionPolicy,
        prompt_token_count: usize,
        spans: &[PromptSpan],
        total_len: usize,
    ) -> Vec<u8> {
        (0..total_len)
            .map(|position| {
                u8::from(Self::cuda_adaptive_position_is_hot_for_policy(
                    policy,
                    prompt_token_count,
                    spans,
                    position,
                    total_len,
                ))
            })
            .collect()
    }

    fn cuda_adaptive_route_migration_needed(
        current_hot_mask: &[u8],
        desired_hot_mask: &[u8],
    ) -> bool {
        if current_hot_mask.len() > desired_hot_mask.len() {
            return true;
        }

        current_hot_mask
            .iter()
            .zip(desired_hot_mask)
            .any(|(&current, &desired)| (current != 0) != (desired != 0))
    }

    fn cuda_cache_layout_changed(
        current_mode: KvCacheMode,
        next_mode: KvCacheMode,
        current_layer_count: usize,
        next_layer_count: usize,
        current_width: usize,
        next_width: usize,
    ) -> bool {
        current_mode != next_mode
            || current_layer_count != next_layer_count
            || current_width != next_width
    }

    fn cuda_cache_mode(cache_mode: KvCacheMode) -> KvCacheMode {
        match cache_mode {
            KvCacheMode::F32 => KvCacheMode::F32,
            KvCacheMode::Q8 if cfg!(feature = "cuda") => KvCacheMode::Q8,
            KvCacheMode::KeyQ4ValueQ8 if cfg!(feature = "cuda") => KvCacheMode::KeyQ4ValueQ8,
            KvCacheMode::AgentAdaptive if cfg!(feature = "cuda") => KvCacheMode::AgentAdaptive,
            KvCacheMode::Q8 | KvCacheMode::KeyQ4ValueQ8 | KvCacheMode::AgentAdaptive => {
                KvCacheMode::F32
            }
        }
    }

    pub fn new_cpu(
        cache_mode: KvCacheMode,
        layer_count: usize,
        width: usize,
        page_tokens: usize,
    ) -> Self {
        Self::Cpu {
            cache: SessionKvCache::new(cache_mode, layer_count, width, page_tokens),
        }
    }

    pub fn new_cuda(
        device: CudaDevice,
        cache_mode: KvCacheMode,
        layer_count: usize,
        width: usize,
        max_len: usize,
    ) -> Self {
        Self::new_cuda_with_kv_budget(device, cache_mode, layer_count, width, max_len, None)
    }

    fn new_cuda_with_kv_budget(
        device: CudaDevice,
        cache_mode: KvCacheMode,
        layer_count: usize,
        width: usize,
        max_len: usize,
        kv_budget_bytes: Option<u64>,
    ) -> Self {
        Self::new_cuda_with_kv_budget_and_page_tokens(
            device,
            cache_mode,
            layer_count,
            width,
            max_len,
            max_len.clamp(1, 32),
            kv_budget_bytes,
        )
    }

    fn new_cuda_with_kv_budget_and_page_tokens(
        device: CudaDevice,
        cache_mode: KvCacheMode,
        layer_count: usize,
        width: usize,
        max_len: usize,
        page_tokens: usize,
        kv_budget_bytes: Option<u64>,
    ) -> Self {
        Self::Cuda {
            device,
            requested_cache_mode: cache_mode,
            cache_mode: Self::cuda_cache_mode(cache_mode),
            layer_caches: Vec::new(),
            layer_count,
            width,
            max_len,
            page_tokens: page_tokens.max(1),
            kv_budget_bytes,
            policy: SessionPolicy::default(),
            prompt_token_count: 0,
            prompt_spans: Vec::new(),
        }
    }

    pub fn cache_mode(&self) -> KvCacheMode {
        match self {
            Self::Cpu { cache } => cache.mode(),
            Self::Cuda { cache_mode, .. } => *cache_mode,
        }
    }

    pub fn requested_cache_mode(&self) -> KvCacheMode {
        match self {
            Self::Cpu { cache } => cache.mode(),
            Self::Cuda {
                requested_cache_mode,
                ..
            } => *requested_cache_mode,
        }
    }

    pub fn cuda_adaptive_position_is_hot(&self, position: usize, total_len: usize) -> bool {
        match self {
            Self::Cuda {
                requested_cache_mode: KvCacheMode::AgentAdaptive,
                policy,
                prompt_token_count,
                prompt_spans,
                ..
            } => Self::cuda_adaptive_position_is_hot_for_policy(
                policy,
                *prompt_token_count,
                prompt_spans,
                position,
                total_len,
            ),
            _ => false,
        }
    }

    pub fn cuda_adaptive_hot_position_mask(&self, total_len: usize) -> Option<Vec<u8>> {
        match self {
            Self::Cuda {
                requested_cache_mode: KvCacheMode::AgentAdaptive,
                policy,
                prompt_token_count,
                prompt_spans,
                ..
            } => Some(Self::cuda_adaptive_hot_position_mask_for_policy(
                policy,
                *prompt_token_count,
                prompt_spans,
                total_len,
            )),
            _ => None,
        }
    }

    pub fn replace_cpu_cache(
        &mut self,
        cache_mode: KvCacheMode,
        layer_count: usize,
        width: usize,
        page_tokens: usize,
    ) {
        *self = Self::new_cpu(cache_mode, layer_count, width, page_tokens);
    }

    pub fn replace_cache(
        &mut self,
        cache_mode: KvCacheMode,
        layer_count: usize,
        width: usize,
        page_tokens: usize,
    ) {
        match self {
            Self::Cpu { .. } => {
                self.replace_cpu_cache(cache_mode, layer_count, width, page_tokens);
            }
            Self::Cuda {
                requested_cache_mode,
                cache_mode: current,
                layer_caches,
                layer_count: current_layer_count,
                width: current_width,
                page_tokens: current_page_tokens,
                policy,
                prompt_token_count,
                prompt_spans,
                ..
            } => {
                let next_cache_mode = Self::cuda_cache_mode(cache_mode);
                let requested_changed = *requested_cache_mode != cache_mode;
                let layout_changed = Self::cuda_cache_layout_changed(
                    *current,
                    next_cache_mode,
                    *current_layer_count,
                    layer_count,
                    *current_width,
                    width,
                );
                *requested_cache_mode = cache_mode;
                *current = next_cache_mode;
                *current_layer_count = layer_count;
                *current_width = width;
                *current_page_tokens = page_tokens.max(1);
                *policy = SessionPolicy::default();
                *prompt_token_count = 0;
                prompt_spans.clear();
                if layout_changed {
                    layer_caches.clear();
                } else if requested_changed {
                    for cache in layer_caches {
                        cache.clear();
                    }
                }
            }
        }
    }

    pub fn configure_policy(
        &mut self,
        policy: SessionPolicy,
        prompt_token_count: usize,
        spans: &[PromptSpan],
    ) {
        match self {
            Self::Cpu { cache } => cache.configure_policy(policy, prompt_token_count, spans),
            Self::Cuda {
                policy: cuda_policy,
                prompt_token_count: cuda_prompt_token_count,
                prompt_spans,
                ..
            } => {
                // ponytail: stored for the future mixed hot/cold CUDA router; current CUDA KV modes are still uniform.
                *cuda_policy = policy;
                *cuda_prompt_token_count = prompt_token_count;
                prompt_spans.clear();
                prompt_spans.extend_from_slice(spans);
            }
        }
    }

    pub fn prepare_for_total_len(&mut self, total_len: usize) -> Result<()> {
        match self {
            Self::Cpu { cache } => cache.prepare_for_total_len(total_len),
            Self::Cuda {
                device,
                cache_mode,
                layer_caches,
                layer_count,
                width,
                max_len,
                page_tokens,
                kv_budget_bytes,
                policy,
                prompt_token_count,
                prompt_spans,
                ..
            } => {
                if total_len > *max_len {
                    return Err(XrtError::Runtime(format!(
                        "CUDA KV request length {total_len} exceeds context length {max_len}"
                    )));
                }
                let current_capacity = layer_caches
                    .first()
                    .map(CudaLayerKvStore::capacity)
                    .unwrap_or(0);
                if total_len > current_capacity {
                    let target_capacity = cuda_kv_growth_capacity(
                        current_capacity,
                        total_len,
                        *page_tokens,
                        *max_len,
                    )?;
                    let required_bytes = cuda_session_kv_allocated_bytes(
                        *cache_mode,
                        *layer_count,
                        target_capacity,
                        *width,
                    )?;
                    let current_bytes = layer_caches
                        .iter()
                        .map(CudaLayerKvStore::allocated_bytes)
                        .try_fold(0u64, |total, bytes| {
                            total.checked_add(bytes).ok_or_else(|| {
                                XrtError::Runtime(
                                    "CUDA KV current allocation byte count overflow".to_string(),
                                )
                            })
                        })?;
                    let peak_bytes =
                        required_bytes.checked_add(current_bytes).ok_or_else(|| {
                            XrtError::Runtime("CUDA KV growth peak byte count overflow".to_string())
                        })?;
                    if let Some(budget_bytes) = kv_budget_bytes {
                        if peak_bytes > *budget_bytes {
                            return Err(XrtError::Cuda(format!(
                                "CUDA KV cache growth requires {peak_bytes} peak bytes for mode {} (final allocation {required_bytes} bytes), but the configured KV budget is {budget_bytes} bytes",
                                cache_mode.as_str()
                            )));
                        }
                    }
                    if layer_caches.is_empty() {
                        let mut caches = Vec::with_capacity(*layer_count);
                        for _ in 0..*layer_count {
                            caches.push(CudaLayerKvStore::allocate(
                                device,
                                *cache_mode,
                                target_capacity,
                                *width,
                            )?);
                        }
                        *layer_caches = caches;
                    } else {
                        for cache in layer_caches.iter_mut() {
                            cache.grow(device, target_capacity)?;
                        }
                    }
                }
                if *cache_mode == KvCacheMode::AgentAdaptive && !layer_caches.is_empty() {
                    let desired_hot_mask = Self::cuda_adaptive_hot_position_mask_for_policy(
                        policy,
                        *prompt_token_count,
                        prompt_spans,
                        total_len,
                    );
                    for cache in layer_caches {
                        cache.migrate_agent_adaptive_route(device, &desired_hot_mask)?;
                    }
                }
                Ok(())
            }
        }
    }

    pub fn clear(&mut self) {
        match self {
            Self::Cpu { cache } => cache.clear(),
            Self::Cuda { layer_caches, .. } => {
                for cache in layer_caches {
                    cache.clear();
                }
            }
        }
    }

    pub fn truncate(&mut self, new_len: usize) {
        match self {
            Self::Cpu { cache } => cache.truncate(new_len),
            Self::Cuda { layer_caches, .. } => {
                for cache in layer_caches {
                    cache.truncate(new_len);
                }
            }
        }
    }

    fn cpu_cache_mut(&mut self) -> Result<&mut SessionKvCache> {
        match self {
            Self::Cpu { cache } => Ok(cache),
            Self::Cuda { .. } => Err(XrtError::Runtime(
                "CPU KV cache requested from CUDA backend session".to_string(),
            )),
        }
    }

    pub fn cuda_layer_cache_mut(&mut self, layer: usize) -> Result<&mut CudaLayerKvStore> {
        match self {
            Self::Cuda { layer_caches, .. } => layer_caches.get_mut(layer).ok_or_else(|| {
                XrtError::Runtime(format!("missing CUDA KV cache for layer {layer}"))
            }),
            Self::Cpu { .. } => Err(XrtError::Runtime(
                "CUDA KV cache requested from CPU backend session".to_string(),
            )),
        }
    }

    pub fn cuda_kv_allocated_bytes(&self) -> u64 {
        match self {
            Self::Cpu { .. } => 0,
            Self::Cuda { layer_caches, .. } => layer_caches
                .iter()
                .map(CudaLayerKvStore::allocated_bytes)
                .sum(),
        }
    }
}

pub trait CausalLmBackend: Send + Sync {
    fn kind(&self) -> BackendKind;
    fn model_name(&self) -> &str;
    fn config(&self) -> &LlamaConfig;
    fn new_session(&self, cache_mode: KvCacheMode, page_tokens: usize) -> BackendSession {
        let config = self.config();
        BackendSession::new_cpu(
            cache_mode,
            config.block_count,
            config.kv_width(),
            page_tokens,
        )
    }
    fn clear_state(&self);
    fn save_state(&self) -> Option<BackendStateSnapshot>;
    fn restore_state(&self, snapshot: &[Option<(Vec<f32>, Vec<f32>)>]);
    fn forward_token(
        &self,
        token_id: u32,
        position: usize,
        session: &mut BackendSession,
        output_logits: &mut Vec<f32>,
    ) -> Result<()>;
    fn forward_draft(
        &self,
        token_id: u32,
        position: usize,
        n_layers: usize,
        session: &mut BackendSession,
        output_logits: &mut Vec<f32>,
    ) -> Result<()> {
        if n_layers == self.config().block_count {
            return self.forward_token(token_id, position, session, output_logits);
        }
        Err(XrtError::Unsupported(format!(
            "{} backend does not support a {n_layers}-layer draft forward pass",
            self.kind()
        )))
    }
    fn forward_batch(
        &self,
        token_ids: &[u32],
        start_position: usize,
        session: &mut BackendSession,
    ) -> Result<Vec<f32>>;
    fn forward_batch_with_embeddings(
        &self,
        token_ids: &[u32],
        start_position: usize,
        session: &mut BackendSession,
        embedding_overrides: HashMap<usize, Vec<f32>>,
    ) -> Result<Vec<f32>>;
    fn forward_batch_all_logits(
        &self,
        token_ids: &[u32],
        start_position: usize,
        session: &mut BackendSession,
    ) -> Result<Vec<f32>>;
    fn embedding_lookup(&self, token_id: usize) -> Result<Vec<f32>>;

    fn model_weight_bytes(&self) -> u64 {
        0
    }

    fn cuda_device_name(&self) -> Option<&str> {
        None
    }

    fn cuda_free_vram_bytes(&self) -> Option<u64> {
        None
    }

    fn cuda_total_vram_bytes(&self) -> Option<u64> {
        None
    }

    fn cuda_kv_budget_bytes(&self) -> Option<u64> {
        None
    }

    fn resident_f32_probe_available(&self) -> bool {
        false
    }

    fn resident_q8_0_probe_available(&self) -> bool {
        false
    }

    fn resident_q8_0_layer0_probe_available(&self) -> bool {
        false
    }

    fn resident_dense_quant_decode_available(&self) -> bool {
        false
    }
}

pub struct CpuBackend {
    model: Arc<LlamaModel>,
}

impl CpuBackend {
    pub fn new(model: Arc<LlamaModel>) -> Self {
        Self { model }
    }
}

impl CausalLmBackend for CpuBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Cpu
    }

    fn model_name(&self) -> &str {
        self.model.model_name()
    }

    fn config(&self) -> &LlamaConfig {
        self.model.config()
    }

    fn clear_state(&self) {
        self.model.clear_state();
    }

    fn save_state(&self) -> Option<BackendStateSnapshot> {
        self.model.save_state()
    }

    fn restore_state(&self, snapshot: &[Option<(Vec<f32>, Vec<f32>)>]) {
        self.model.restore_state(snapshot);
    }

    fn forward_token(
        &self,
        token_id: u32,
        position: usize,
        session: &mut BackendSession,
        output_logits: &mut Vec<f32>,
    ) -> Result<()> {
        let cache = session.cpu_cache_mut()?;
        self.model
            .forward_token(token_id, position, cache, output_logits)
    }

    fn forward_draft(
        &self,
        token_id: u32,
        position: usize,
        n_layers: usize,
        session: &mut BackendSession,
        output_logits: &mut Vec<f32>,
    ) -> Result<()> {
        let cache = session.cpu_cache_mut()?;
        self.model
            .forward_draft(token_id, position, n_layers, cache, output_logits)
    }

    fn forward_batch(
        &self,
        token_ids: &[u32],
        start_position: usize,
        session: &mut BackendSession,
    ) -> Result<Vec<f32>> {
        let cache = session.cpu_cache_mut()?;
        self.model.forward_batch(token_ids, start_position, cache)
    }

    fn forward_batch_with_embeddings(
        &self,
        token_ids: &[u32],
        start_position: usize,
        session: &mut BackendSession,
        embedding_overrides: HashMap<usize, Vec<f32>>,
    ) -> Result<Vec<f32>> {
        let cache = session.cpu_cache_mut()?;
        self.model.forward_batch_with_embeddings(
            token_ids,
            start_position,
            cache,
            embedding_overrides,
        )
    }

    fn forward_batch_all_logits(
        &self,
        token_ids: &[u32],
        start_position: usize,
        session: &mut BackendSession,
    ) -> Result<Vec<f32>> {
        let cache = session.cpu_cache_mut()?;
        self.model
            .forward_batch_all_logits(token_ids, start_position, cache)
    }

    fn embedding_lookup(&self, token_id: usize) -> Result<Vec<f32>> {
        self.model.embedding_lookup(token_id)
    }
}

pub struct CudaResidentBackend {
    model: Arc<LlamaModel>,
    device: CudaDevice,
    resident_model_weight_bytes: u64,
    device_name: Option<String>,
    free_vram_bytes: Option<u64>,
    total_vram_bytes: Option<u64>,
    kv_budget_bytes: u64,
    f32_probe: Option<ResidentF32ProbeWeights>,
    q8_0_probe: Option<ResidentQ8_0ProbeWeights>,
    q8_0_layer_probes: Option<Vec<ResidentQ8_0LayerWeights>>,
}

impl CudaResidentBackend {
    pub fn new(model: Arc<LlamaModel>, gguf: &GgufFile, config: GpuResourceConfig) -> Result<Self> {
        if !Self::supports_dense_quant_decode(gguf, model.config()) {
            return Err(Self::decode_unsupported());
        }
        let device = CudaDevice::new(config.device_ordinal)?;
        let (free_vram_bytes, total_vram_bytes, resident_model_weight_bytes, kv_budget_bytes) =
            Self::preflight_model_upload(gguf, model.config(), &device, config)?;
        let device_name = device.name().ok();
        let f32_probe = ResidentF32ProbeWeights::try_load(&device, gguf, model.config())?;
        let q8_0_probe = ResidentQ8_0ProbeWeights::try_load(&device, gguf, model.config())?;
        let q8_0_layer_probes =
            ResidentQ8_0LayerWeights::try_load_all(&device, gguf, model.config())?;
        Ok(Self {
            model,
            device,
            resident_model_weight_bytes,
            device_name,
            free_vram_bytes: Some(free_vram_bytes),
            total_vram_bytes: Some(total_vram_bytes),
            kv_budget_bytes,
            f32_probe,
            q8_0_probe,
            q8_0_layer_probes,
        })
    }

    pub fn supports_dense_quant_decode(gguf: &GgufFile, config: &LlamaConfig) -> bool {
        ResidentQ8_0ProbeWeights::supports(gguf, config)
            && ResidentQ8_0LayerWeights::supports_all(gguf, config)
    }

    fn preflight_model_upload(
        gguf: &GgufFile,
        model_config: &LlamaConfig,
        device: &CudaDevice,
        config: GpuResourceConfig,
    ) -> Result<(u64, u64, u64, u64)> {
        let model_weight_bytes = cuda_estimated_resident_upload_bytes(gguf, model_config)?;
        let (free_vram_bytes, total_vram_bytes) = device.memory_info()?;
        let upload_budget_bytes =
            cuda_model_upload_budget_bytes(free_vram_bytes, total_vram_bytes, config);
        if model_weight_bytes > upload_budget_bytes {
            return Err(XrtError::Cuda(format!(
                "CUDA model upload requires {model_weight_bytes} bytes, but only {upload_budget_bytes} bytes are inside the configured safe VRAM budget (free={free_vram_bytes}, total={total_vram_bytes}, memory_fraction={}, reserved_bytes={})",
                config.memory_fraction,
                config.reserved_bytes()
            )));
        }
        Ok((
            free_vram_bytes,
            total_vram_bytes,
            model_weight_bytes,
            cuda_kv_budget_bytes(upload_budget_bytes, model_weight_bytes, config),
        ))
    }

    fn decode_unsupported() -> XrtError {
        XrtError::Unsupported(
            "cuda-resident decode currently supports only standard dense F32/F16/BF16/Q8_0/Q4_0/Q4_K/Q5_K/Q6_K models; broader GGUF decode is not wired yet"
                .to_string(),
        )
    }

    fn cuda_profile_enabled() -> bool {
        env::var("XRT_CUDA_PROFILE").is_ok_and(|value| Self::cuda_profile_value_enabled(&value))
    }

    fn cuda_profile_value_enabled(value: &str) -> bool {
        let value = value.trim();
        !value.is_empty()
            && value != "0"
            && !value.eq_ignore_ascii_case("false")
            && !value.eq_ignore_ascii_case("off")
    }

    fn validate_embedding_overrides(
        token_count: usize,
        embedding_length: usize,
        embedding_overrides: &HashMap<usize, Vec<f32>>,
    ) -> Result<()> {
        for (&index, embedding) in embedding_overrides {
            if index >= token_count {
                return Err(XrtError::Runtime(format!(
                    "embedding override position {index} exceeds token batch length {token_count}"
                )));
            }
            if embedding.len() != embedding_length {
                return Err(XrtError::Runtime(format!(
                    "embedding override at position {index} has {} floats, expected {embedding_length}",
                    embedding.len()
                )));
            }
        }
        Ok(())
    }

    pub fn resident_f32_probe_logits(&self, token_id: u32) -> Result<Option<Vec<f32>>> {
        let Some(probe) = &self.f32_probe else {
            return Ok(None);
        };
        if token_id as usize >= probe.vocab_size {
            return Err(XrtError::Model(format!(
                "token id {token_id} exceeds embedding rows {}",
                probe.vocab_size
            )));
        }

        let embedding = self.device.embed_resident_device(
            probe.token_embedding.buffer(),
            probe.vocab_size,
            probe.embedding_length,
            &[token_id],
        )?;
        let normed = self.device.rmsnorm_device(
            &embedding,
            probe.output_norm.buffer(),
            1,
            probe.embedding_length,
            self.model.config().rms_norm_eps,
        )?;
        self.device
            .matmul_resident_rhs_device(
                &normed,
                1,
                probe.embedding_length,
                probe.output_transposed.buffer(),
                probe.vocab_size,
            )
            .and_then(|logits| self.device.download_f32(&logits))
            .map(Some)
    }

    pub fn resident_q8_0_probe_logits(&self, token_id: u32) -> Result<Option<Vec<f32>>> {
        let Some(probe) = &self.q8_0_probe else {
            return Ok(None);
        };
        if token_id as usize >= probe.vocab_size {
            return Err(XrtError::Model(format!(
                "token id {token_id} exceeds embedding rows {}",
                probe.vocab_size
            )));
        }

        let embedding = self.embed_q8_0_probe_token(probe, token_id)?;
        let normed = self.device.rmsnorm_device(
            &embedding,
            probe.output_norm.buffer(),
            1,
            probe.embedding_length,
            self.model.config().rms_norm_eps,
        )?;
        self.matvec_quant_resident_device(&probe.output, &normed)
            .and_then(|logits| self.device.download_f32(&logits))
            .map(Some)
    }

    pub fn resident_q8_0_layer0_projection_probe(
        &self,
        token_id: u32,
    ) -> Result<Option<ResidentQ8_0Layer0ProjectionOutput>> {
        self.resident_q8_0_layer0_projection_probe_at(token_id, 0)
    }

    pub fn resident_q8_0_layer0_projection_probe_at(
        &self,
        token_id: u32,
        position: usize,
    ) -> Result<Option<ResidentQ8_0Layer0ProjectionOutput>> {
        let (Some(output_probe), Some(layer_probes)) = (&self.q8_0_probe, &self.q8_0_layer_probes)
        else {
            return Ok(None);
        };
        let Some(probe) = layer_probes.first() else {
            return Ok(None);
        };
        Self::validate_layer0_probe_position(position)?;
        if token_id as usize >= probe.vocab_size {
            return Err(XrtError::Model(format!(
                "token id {token_id} exceeds embedding rows {}",
                probe.vocab_size
            )));
        }
        let config = self.model.config();
        let mut kv_cache =
            CudaLayerKvStore::F32(self.device.alloc_layer_kv_cache(1, config.kv_width())?);
        let embedding = self.embed_q8_0_probe_token(output_probe, token_id)?;
        let layer =
            self.run_q8_0_layer_device(0, probe, &embedding, position, true, &mut kv_cache)?;

        Ok(Some(ResidentQ8_0Layer0ProjectionOutput {
            position,
            q: self.device.download_f32(&layer.q)?,
            k: self.device.download_f32(&layer.k)?,
            v: self.device.download_f32(&layer.v)?,
            attn_output: self.device.download_f32(&layer.attn_output)?,
            post_attention: self.device.download_f32(&layer.post_attention)?,
            gate: self.device.download_f32(&layer.gate)?,
            up: self.device.download_f32(&layer.up)?,
            ffn_hidden: self.device.download_f32(&layer.ffn_hidden)?,
            down: self.device.download_f32(&layer.down)?,
            post_ffn: self.device.download_f32(&layer.post_ffn)?,
        }))
    }

    fn validate_layer0_probe_position(position: usize) -> Result<()> {
        if position == 0 {
            return Ok(());
        }
        Err(XrtError::Unsupported(
            "layer-0 CUDA projection probe supports only position 0; nonzero positions require a populated prefix KV cache"
                .to_string(),
        ))
    }

    fn run_q8_0_layer_device(
        &self,
        layer_index: usize,
        probe: &ResidentQ8_0LayerWeights,
        input: &CudaF32Buffer,
        position: usize,
        adaptive_is_hot: bool,
        kv_cache: &mut CudaLayerKvStore,
    ) -> Result<ResidentQ8_0Layer0DeviceOutput> {
        let config = self.model.config();
        let profile = Self::cuda_profile_enabled();
        let layer_start = Instant::now();
        let stage_start = Instant::now();
        let attn_normed = self.device.rmsnorm_device(
            input,
            probe.attn_norm.buffer(),
            1,
            probe.embedding_length,
            config.rms_norm_eps,
        )?;
        let mut q = self.matvec_quant_resident_device(&probe.attn_q, &attn_normed)?;
        let mut k = self.matvec_quant_resident_device(&probe.attn_k, &attn_normed)?;
        let mut v = self.matvec_quant_resident_device(&probe.attn_v, &attn_normed)?;
        if let Some(bias) = &probe.attn_q_bias {
            q = self.device.add_device(&q, bias.buffer())?;
        }
        if let Some(bias) = &probe.attn_k_bias {
            k = self.device.add_device(&k, bias.buffer())?;
        }
        if let Some(bias) = &probe.attn_v_bias {
            v = self.device.add_device(&v, bias.buffer())?;
        }
        if let Some(q_norm) = &probe.attn_q_norm {
            q = self.device.rmsnorm_device(
                &q,
                q_norm.buffer(),
                config.attention_head_count,
                config.head_dim(),
                config.rms_norm_eps,
            )?;
        }
        if let Some(k_norm) = &probe.attn_k_norm {
            k = self.device.rmsnorm_device(
                &k,
                k_norm.buffer(),
                config.attention_head_count_kv,
                config.head_dim(),
                config.rms_norm_eps,
            )?;
        }
        if profile {
            info!(
                layer_index,
                ms = stage_start.elapsed().as_secs_f64() * 1000.0,
                "cuda profile: qkv"
            );
        }

        let stage_start = Instant::now();
        self.device.rope_device(
            &mut q,
            config.attention_head_count,
            config.head_dim(),
            position,
            config.rope_dimension_count,
            config.rope_freq_base,
            config.rope_freq_scale,
        )?;
        self.device.rope_device(
            &mut k,
            config.attention_head_count_kv,
            config.head_dim(),
            position,
            config.rope_dimension_count,
            config.rope_freq_base,
            config.rope_freq_scale,
        )?;
        let attention_values = match kv_cache {
            CudaLayerKvStore::F32(cache) => {
                self.device.append_layer_kv(cache, &k, &v)?;
                self.device.single_query_attention_device(
                    &q,
                    cache,
                    config.attention_head_count,
                    config.attention_head_count_kv,
                    config.head_dim(),
                )?
            }
            CudaLayerKvStore::Q8(cache) => {
                self.device.append_q8_layer_kv(cache, &k, &v)?;
                self.device.single_query_attention_q8_device(
                    &q,
                    cache,
                    config.attention_head_count,
                    config.attention_head_count_kv,
                    config.head_dim(),
                )?
            }
            CudaLayerKvStore::KeyQ4ValueQ8(cache) => {
                self.device.append_key_q4_value_q8_layer_kv(cache, &k, &v)?;
                self.device.single_query_attention_key_q4_value_q8_device(
                    &q,
                    cache,
                    config.attention_head_count,
                    config.attention_head_count_kv,
                    config.head_dim(),
                )?
            }
            CudaLayerKvStore::AgentAdaptive {
                hot,
                cold,
                hot_mask,
            } => {
                if adaptive_is_hot {
                    self.device.append_layer_kv(hot, &k, &v)?;
                    hot_mask.push(1);
                } else {
                    self.device.append_key_q4_value_q8_layer_kv(cold, &k, &v)?;
                    hot_mask.push(0);
                }
                self.device
                    .single_query_attention_mixed_key_q4_value_q8_device(
                        &q,
                        hot,
                        cold,
                        hot_mask,
                        config.attention_head_count,
                        config.attention_head_count_kv,
                        config.head_dim(),
                    )?
            }
        };
        if profile {
            info!(
                layer_index,
                ms = stage_start.elapsed().as_secs_f64() * 1000.0,
                "cuda profile: attention"
            );
        }

        let stage_start = Instant::now();
        let attn_output =
            self.matvec_quant_resident_device(&probe.attn_output, &attention_values)?;
        let post_attention = self.device.add_device(input, &attn_output)?;
        if profile {
            info!(
                layer_index,
                ms = stage_start.elapsed().as_secs_f64() * 1000.0,
                "cuda profile: attention output"
            );
        }

        let stage_start = Instant::now();
        let ffn_normed = self.device.rmsnorm_device(
            &post_attention,
            probe.ffn_norm.buffer(),
            1,
            probe.embedding_length,
            config.rms_norm_eps,
        )?;
        let gate = self.matvec_quant_resident_device(&probe.ffn_gate, &ffn_normed)?;
        let up = self.matvec_quant_resident_device(&probe.ffn_up, &ffn_normed)?;
        let activated_gate = self.device.silu_device(&gate)?;
        let ffn_hidden = self.device.mul_device(&activated_gate, &up)?;
        let down = self.matvec_quant_resident_device(&probe.ffn_down, &ffn_hidden)?;
        let post_ffn = self.device.add_device(&post_attention, &down)?;
        if profile {
            info!(
                layer_index,
                ms = stage_start.elapsed().as_secs_f64() * 1000.0,
                "cuda profile: ffn"
            );
            info!(
                layer_index,
                ms = layer_start.elapsed().as_secs_f64() * 1000.0,
                "cuda profile: layer"
            );
        }

        Ok(ResidentQ8_0Layer0DeviceOutput {
            q,
            k,
            v,
            attn_output,
            post_attention,
            gate,
            up,
            ffn_hidden,
            down,
            post_ffn,
        })
    }

    fn try_forward_token_q8_0(
        &self,
        token_id: u32,
        position: usize,
        session: &mut BackendSession,
        output_logits: &mut Vec<f32>,
    ) -> Result<bool> {
        self.try_forward_token_q8_0_with_logits(
            token_id,
            position,
            session,
            output_logits,
            true,
            None,
            cuda_total_len_for_position(position)?,
            None,
        )
    }

    fn try_forward_token_q8_0_with_logits(
        &self,
        token_id: u32,
        position: usize,
        session: &mut BackendSession,
        output_logits: &mut Vec<f32>,
        compute_logits: bool,
        embedding_override: Option<&[f32]>,
        adaptive_total_len: usize,
        max_layers: Option<usize>,
    ) -> Result<bool> {
        let config = self.model.config();
        let profile = Self::cuda_profile_enabled();
        let token_start = Instant::now();
        let (Some(layer_probes), Some(output_probe)) = (&self.q8_0_layer_probes, &self.q8_0_probe)
        else {
            return Ok(false);
        };
        if layer_probes.len() != config.block_count {
            return Ok(false);
        }
        let layer_count = max_layers.unwrap_or(config.block_count);
        if layer_count > config.block_count {
            return Err(XrtError::Runtime(format!(
                "CUDA draft layer count {layer_count} exceeds model layer count {}",
                config.block_count
            )));
        }
        if embedding_override.is_none() && token_id as usize >= output_probe.vocab_size {
            return Err(XrtError::Model(format!(
                "token id {token_id} exceeds embedding rows {}",
                output_probe.vocab_size
            )));
        }

        let prepare_total_len = adaptive_total_len.max(cuda_total_len_for_position(position)?);
        session.prepare_for_total_len(prepare_total_len)?;
        let stage_start = Instant::now();
        let mut x = if let Some(embedding) = embedding_override {
            self.device.upload_f32(embedding)?
        } else {
            self.embed_q8_0_probe_token(output_probe, token_id)?
        };
        if profile {
            info!(
                position,
                ms = stage_start.elapsed().as_secs_f64() * 1000.0,
                "cuda profile: token embedding"
            );
        }
        for (layer_index, layer_probe) in layer_probes.iter().take(layer_count).enumerate() {
            let adaptive_is_hot =
                session.cuda_adaptive_position_is_hot(position, adaptive_total_len);
            let kv_cache = session.cuda_layer_cache_mut(layer_index)?;
            if kv_cache.len() != position {
                return Err(XrtError::Runtime(format!(
                    "CUDA KV cache length mismatch at layer {layer_index}: expected {position}, found {}",
                    kv_cache.len()
                )));
            }
            x = self
                .run_q8_0_layer_device(
                    layer_index,
                    layer_probe,
                    &x,
                    position,
                    adaptive_is_hot,
                    kv_cache,
                )?
                .post_ffn;
        }
        if !compute_logits {
            output_logits.clear();
            if profile {
                info!(
                    position,
                    ms = token_start.elapsed().as_secs_f64() * 1000.0,
                    "cuda profile: token"
                );
            }
            return Ok(true);
        }

        let final_start = Instant::now();
        let stage_start = Instant::now();
        let normed = self.device.rmsnorm_device(
            &x,
            output_probe.output_norm.buffer(),
            1,
            output_probe.embedding_length,
            config.rms_norm_eps,
        )?;
        if profile {
            info!(
                position,
                ms = stage_start.elapsed().as_secs_f64() * 1000.0,
                "cuda profile: final norm"
            );
        }

        let stage_start = Instant::now();
        let logits_dev = self.matvec_quant_resident_device(&output_probe.output, &normed)?;
        if profile {
            info!(
                position,
                ms = stage_start.elapsed().as_secs_f64() * 1000.0,
                "cuda profile: final projection"
            );
        }

        let stage_start = Instant::now();
        let logits = self.device.download_f32(&logits_dev)?;
        if profile {
            info!(
                position,
                ms = stage_start.elapsed().as_secs_f64() * 1000.0,
                "cuda profile: logits download"
            );
            info!(
                position,
                ms = final_start.elapsed().as_secs_f64() * 1000.0,
                "cuda profile: final logits"
            );
            info!(
                position,
                ms = token_start.elapsed().as_secs_f64() * 1000.0,
                "cuda profile: token"
            );
        }
        *output_logits = logits;
        Ok(true)
    }

    fn embed_q8_0_probe_token(
        &self,
        probe: &ResidentQ8_0ProbeWeights,
        token_id: u32,
    ) -> Result<CudaF32Buffer> {
        match &probe.token_embedding {
            ResidentTokenEmbedding::F32(tensor) => self.device.embed_resident_device(
                tensor.buffer(),
                probe.vocab_size,
                probe.embedding_length,
                &[token_id],
            ),
            ResidentTokenEmbedding::Q8_0(matrix) => {
                self.device.embed_q8_0_resident_device(matrix, &[token_id])
            }
            ResidentTokenEmbedding::Q4_0(matrix) => {
                self.device.embed_q8_0_resident_device(matrix, &[token_id])
            }
            ResidentTokenEmbedding::Q4K(matrix) => {
                self.device.embed_q4_k_resident_device(matrix, &[token_id])
            }
            ResidentTokenEmbedding::Q5K(matrix) => {
                self.device.embed_q5_k_resident_device(matrix, &[token_id])
            }
            ResidentTokenEmbedding::Q6K(matrix) => {
                self.device.embed_q6_k_resident_device(matrix, &[token_id])
            }
        }
    }

    fn matvec_quant_resident_device(
        &self,
        matrix: &ResidentQuantMatrix,
        input: &CudaF32Buffer,
    ) -> Result<CudaF32Buffer> {
        match matrix {
            ResidentQuantMatrix::F32(tensor) => self.device.matmul_resident_rhs_device(
                input,
                1,
                tensor.dimensions[1],
                tensor.buffer(),
                tensor.dimensions[0],
            ),
            ResidentQuantMatrix::Q8_0(matrix) => {
                self.device.matvec_q8_0_resident_device(matrix, input)
            }
            ResidentQuantMatrix::Q4_0(matrix) => {
                self.device.matvec_q4_0_resident_device(matrix, input)
            }
            ResidentQuantMatrix::Q4K(matrix) => {
                self.device.matvec_q4_k_resident_device(matrix, input)
            }
            ResidentQuantMatrix::Q5K(matrix) => {
                self.device.matvec_q5_k_resident_device(matrix, input)
            }
            ResidentQuantMatrix::Q6K(matrix) => {
                self.device.matvec_q6_k_resident_device(matrix, input)
            }
        }
    }
}

pub struct ResidentQ8_0Layer0ProjectionOutput {
    pub position: usize,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_output: Vec<f32>,
    pub post_attention: Vec<f32>,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub ffn_hidden: Vec<f32>,
    pub down: Vec<f32>,
    pub post_ffn: Vec<f32>,
}

struct ResidentQ8_0Layer0DeviceOutput {
    q: CudaF32Buffer,
    k: CudaF32Buffer,
    v: CudaF32Buffer,
    attn_output: CudaF32Buffer,
    post_attention: CudaF32Buffer,
    gate: CudaF32Buffer,
    up: CudaF32Buffer,
    ffn_hidden: CudaF32Buffer,
    down: CudaF32Buffer,
    post_ffn: CudaF32Buffer,
}

struct ResidentF32ProbeWeights {
    token_embedding: GpuF32Tensor,
    output_norm: GpuF32Tensor,
    output_transposed: GpuF32Tensor,
    vocab_size: usize,
    embedding_length: usize,
}

impl ResidentF32ProbeWeights {
    fn try_load(
        device: &CudaDevice,
        gguf: &GgufFile,
        config: &LlamaConfig,
    ) -> Result<Option<Self>> {
        let token_embedding_name = "token_embd.weight";
        let output_norm_name = "output_norm.weight";
        let output_name = if gguf.tensor_info("output.weight").is_some() {
            "output.weight"
        } else {
            token_embedding_name
        };

        let Some(token_embedding_info) = gguf.tensor_info(token_embedding_name) else {
            return Ok(None);
        };
        let Some(output_norm_info) = gguf.tensor_info(output_norm_name) else {
            return Ok(None);
        };
        let Some(output_info) = gguf.tensor_info(output_name) else {
            return Ok(None);
        };

        if !is_supported_resident_float_dtype(token_embedding_info.dtype)
            || !is_supported_resident_float_dtype(output_norm_info.dtype)
            || !is_supported_resident_float_dtype(output_info.dtype)
        {
            return Ok(None);
        }
        if token_embedding_info.row_len() != config.embedding_length
            || token_embedding_info.rows() != config.vocab_size
            || output_norm_info.numel() != config.embedding_length
            || output_info.row_len() != config.embedding_length
            || output_info.rows() != config.vocab_size
        {
            return Ok(None);
        }

        Ok(Some(Self {
            token_embedding: device.upload_f32_tensor(gguf, token_embedding_name)?,
            output_norm: device.upload_f32_tensor(gguf, output_norm_name)?,
            output_transposed: device.upload_f32_tensor_transposed_2d(gguf, output_name)?,
            vocab_size: config.vocab_size,
            embedding_length: config.embedding_length,
        }))
    }
}

struct ResidentQ8_0ProbeWeights {
    token_embedding: ResidentTokenEmbedding,
    output_norm: GpuF32Tensor,
    output: ResidentQuantMatrix,
    vocab_size: usize,
    embedding_length: usize,
}

enum ResidentTokenEmbedding {
    F32(GpuF32Tensor),
    Q8_0(Arc<CudaQ8_0Matrix>),
    Q4_0(Arc<CudaQ4_0Matrix>),
    Q4K(Arc<CudaQ4KMatrix>),
    Q5K(Arc<CudaQ5KMatrix>),
    Q6K(Arc<CudaQ6KMatrix>),
}

impl ResidentTokenEmbedding {
    fn is_q8_0(&self) -> bool {
        matches!(self, Self::Q8_0(_))
    }

    fn is_gpu_resident(&self) -> bool {
        true
    }

    fn tied_output_matrix(&self) -> Option<ResidentQuantMatrix> {
        match self {
            Self::F32(_) => None,
            Self::Q8_0(matrix) => Some(ResidentQuantMatrix::Q8_0(Arc::clone(matrix))),
            Self::Q4_0(matrix) => Some(ResidentQuantMatrix::Q4_0(Arc::clone(matrix))),
            Self::Q4K(matrix) => Some(ResidentQuantMatrix::Q4K(Arc::clone(matrix))),
            Self::Q5K(matrix) => Some(ResidentQuantMatrix::Q5K(Arc::clone(matrix))),
            Self::Q6K(matrix) => Some(ResidentQuantMatrix::Q6K(Arc::clone(matrix))),
        }
    }
}

enum ResidentQuantMatrix {
    F32(GpuF32Tensor),
    Q8_0(Arc<CudaQ8_0Matrix>),
    Q4_0(Arc<CudaQ4_0Matrix>),
    Q4K(Arc<CudaQ4KMatrix>),
    Q5K(Arc<CudaQ5KMatrix>),
    Q6K(Arc<CudaQ6KMatrix>),
}

impl ResidentQuantMatrix {
    fn upload(device: &CudaDevice, gguf: &GgufFile, name: &str) -> Result<Self> {
        let info = gguf.require_tensor(name)?;
        match info.dtype {
            DType::F32 | DType::F16 | DType::BF16 => device
                .upload_f32_tensor_transposed_2d(gguf, name)
                .map(Self::F32),
            DType::Q8_0 => device
                .upload_q8_0_tensor(gguf, name)
                .map(Arc::new)
                .map(Self::Q8_0),
            DType::Q4_0 => device
                .upload_q4_0_tensor(gguf, name)
                .map(Arc::new)
                .map(Self::Q4_0),
            DType::Q4_K => device
                .upload_q4_k_tensor(gguf, name)
                .map(Arc::new)
                .map(Self::Q4K),
            DType::Q5_K => device
                .upload_q5_k_tensor(gguf, name)
                .map(Arc::new)
                .map(Self::Q5K),
            DType::Q6_K => device
                .upload_q6_k_tensor(gguf, name)
                .map(Arc::new)
                .map(Self::Q6K),
        }
    }

    fn is_q8_0(&self) -> bool {
        matches!(self, Self::Q8_0(_))
    }
}

impl ResidentQ8_0ProbeWeights {
    fn output_name(gguf: &GgufFile) -> &'static str {
        if gguf.tensor_info("output.weight").is_some() {
            "output.weight"
        } else {
            "token_embd.weight"
        }
    }

    fn supports(gguf: &GgufFile, config: &LlamaConfig) -> bool {
        let token_embedding_name = "token_embd.weight";
        let output_norm_name = "output_norm.weight";
        let output_name = Self::output_name(gguf);

        let Some(token_embedding_info) = gguf.tensor_info(token_embedding_name) else {
            return false;
        };
        let Some(output_norm_info) = gguf.tensor_info(output_norm_name) else {
            return false;
        };
        let Some(output_info) = gguf.tensor_info(output_name) else {
            return false;
        };

        is_supported_resident_linear_dtype(token_embedding_info.dtype)
            && is_supported_resident_float_dtype(output_norm_info.dtype)
            && is_supported_resident_linear_dtype(output_info.dtype)
            && token_embedding_info.row_len() == config.embedding_length
            && token_embedding_info.rows() == config.vocab_size
            && output_norm_info.numel() == config.embedding_length
            && output_info.row_len() == config.embedding_length
            && output_info.rows() == config.vocab_size
    }

    fn try_load(
        device: &CudaDevice,
        gguf: &GgufFile,
        config: &LlamaConfig,
    ) -> Result<Option<Self>> {
        let token_embedding_name = "token_embd.weight";
        let output_norm_name = "output_norm.weight";
        let output_name = Self::output_name(gguf);

        if !Self::supports(gguf, config) {
            return Ok(None);
        }
        let token_embedding_info = gguf
            .tensor_info(token_embedding_name)
            .expect("token embedding tensor was checked above");
        let token_embedding = match token_embedding_info.dtype {
            DType::F32 | DType::F16 | DType::BF16 => {
                ResidentTokenEmbedding::F32(device.upload_f32_tensor(gguf, token_embedding_name)?)
            }
            DType::Q8_0 => ResidentTokenEmbedding::Q8_0(Arc::new(
                device.upload_q8_0_tensor(gguf, token_embedding_name)?,
            )),
            DType::Q4_0 => ResidentTokenEmbedding::Q4_0(Arc::new(
                device.upload_q4_0_tensor(gguf, token_embedding_name)?,
            )),
            DType::Q4_K => ResidentTokenEmbedding::Q4K(Arc::new(
                device.upload_q4_k_embedding_tensor(gguf, token_embedding_name)?,
            )),
            DType::Q5_K => ResidentTokenEmbedding::Q5K(Arc::new(
                device.upload_q5_k_embedding_tensor(gguf, token_embedding_name)?,
            )),
            DType::Q6_K => ResidentTokenEmbedding::Q6K(Arc::new(
                device.upload_q6_k_embedding_tensor(gguf, token_embedding_name)?,
            )),
        };
        let output = if output_name == token_embedding_name {
            if let Some(shared) = token_embedding.tied_output_matrix() {
                shared
            } else {
                ResidentQuantMatrix::upload(device, gguf, output_name)?
            }
        } else {
            ResidentQuantMatrix::upload(device, gguf, output_name)?
        };

        Ok(Some(Self {
            token_embedding,
            output_norm: device.upload_f32_tensor(gguf, output_norm_name)?,
            output,
            vocab_size: config.vocab_size,
            embedding_length: config.embedding_length,
        }))
    }

    fn is_q8_0_only(&self) -> bool {
        q8_0_probe_status_available(self.token_embedding.is_q8_0(), self.output.is_q8_0())
    }

    fn token_embedding_gpu_resident(&self) -> bool {
        self.token_embedding.is_gpu_resident()
    }
}

fn q8_0_probe_status_available(token_embedding_is_q8_0: bool, output_is_q8_0: bool) -> bool {
    token_embedding_is_q8_0 && output_is_q8_0
}

fn dense_quant_decode_status_available(
    has_output_probe: bool,
    token_embedding_gpu_resident: bool,
    loaded_layer_count: Option<usize>,
    expected_layer_count: usize,
) -> bool {
    has_output_probe
        && token_embedding_gpu_resident
        && loaded_layer_count.is_some_and(|count| count == expected_layer_count)
}

struct ResidentQ8_0LayerWeights {
    attn_norm: GpuF32Tensor,
    ffn_norm: GpuF32Tensor,
    attn_q: ResidentQuantMatrix,
    attn_k: ResidentQuantMatrix,
    attn_v: ResidentQuantMatrix,
    attn_q_norm: Option<GpuF32Tensor>,
    attn_k_norm: Option<GpuF32Tensor>,
    attn_q_bias: Option<GpuF32Tensor>,
    attn_k_bias: Option<GpuF32Tensor>,
    attn_v_bias: Option<GpuF32Tensor>,
    attn_output: ResidentQuantMatrix,
    ffn_gate: ResidentQuantMatrix,
    ffn_up: ResidentQuantMatrix,
    ffn_down: ResidentQuantMatrix,
    vocab_size: usize,
    embedding_length: usize,
}

impl ResidentQ8_0LayerWeights {
    fn supports_all(gguf: &GgufFile, config: &LlamaConfig) -> bool {
        if config.block_count == 0 || config.is_hybrid() || config.is_gemma4() || config.is_moe() {
            return false;
        }
        let q_width = config.q_width();
        let kv_width = config.kv_width();
        let dim = config.embedding_length;
        let ff_dim = config.feed_forward_length;
        let head_dim = config.head_dim();

        for layer in 0..config.block_count {
            if !matches_optional_qk_norm_pair(gguf, layer, head_dim) {
                return false;
            }
            if !matches_f32_vector(gguf, &format!("blk.{layer}.attn_norm.weight"), dim)
                || !matches_f32_vector(gguf, &format!("blk.{layer}.ffn_norm.weight"), dim)
            {
                return false;
            }
            for (name, len) in [
                (format!("blk.{layer}.attn_q.bias"), q_width),
                (format!("blk.{layer}.attn_k.bias"), kv_width),
                (format!("blk.{layer}.attn_v.bias"), kv_width),
            ] {
                if !matches_optional_f32_vector(gguf, &name, len) {
                    return false;
                }
            }
            for (name, rows, cols) in [
                (format!("blk.{layer}.attn_q.weight"), q_width, dim),
                (format!("blk.{layer}.attn_k.weight"), kv_width, dim),
                (format!("blk.{layer}.attn_v.weight"), kv_width, dim),
                (format!("blk.{layer}.attn_output.weight"), dim, q_width),
                (format!("blk.{layer}.ffn_gate.weight"), ff_dim, dim),
                (format!("blk.{layer}.ffn_up.weight"), ff_dim, dim),
                (format!("blk.{layer}.ffn_down.weight"), dim, ff_dim),
            ] {
                if !matches_supported_linear_shape(gguf, &name, rows, cols) {
                    return false;
                }
            }
        }
        true
    }

    fn try_load_all(
        device: &CudaDevice,
        gguf: &GgufFile,
        config: &LlamaConfig,
    ) -> Result<Option<Vec<Self>>> {
        if !Self::supports_all(gguf, config) {
            return Ok(None);
        }
        let dim = config.embedding_length;

        let mut layers = Vec::with_capacity(config.block_count);
        for layer in 0..config.block_count {
            layers.push(Self {
                attn_norm: device
                    .upload_f32_tensor(gguf, &format!("blk.{layer}.attn_norm.weight"))?,
                ffn_norm: device
                    .upload_f32_tensor(gguf, &format!("blk.{layer}.ffn_norm.weight"))?,
                attn_q: ResidentQuantMatrix::upload(
                    device,
                    gguf,
                    &format!("blk.{layer}.attn_q.weight"),
                )?,
                attn_k: ResidentQuantMatrix::upload(
                    device,
                    gguf,
                    &format!("blk.{layer}.attn_k.weight"),
                )?,
                attn_v: ResidentQuantMatrix::upload(
                    device,
                    gguf,
                    &format!("blk.{layer}.attn_v.weight"),
                )?,
                attn_q_norm: upload_optional_f32_tensor(
                    device,
                    gguf,
                    &format!("blk.{layer}.attn_q_norm.weight"),
                )?,
                attn_k_norm: upload_optional_f32_tensor(
                    device,
                    gguf,
                    &format!("blk.{layer}.attn_k_norm.weight"),
                )?,
                attn_q_bias: upload_optional_f32_tensor(
                    device,
                    gguf,
                    &format!("blk.{layer}.attn_q.bias"),
                )?,
                attn_k_bias: upload_optional_f32_tensor(
                    device,
                    gguf,
                    &format!("blk.{layer}.attn_k.bias"),
                )?,
                attn_v_bias: upload_optional_f32_tensor(
                    device,
                    gguf,
                    &format!("blk.{layer}.attn_v.bias"),
                )?,
                attn_output: ResidentQuantMatrix::upload(
                    device,
                    gguf,
                    &format!("blk.{layer}.attn_output.weight"),
                )?,
                ffn_gate: ResidentQuantMatrix::upload(
                    device,
                    gguf,
                    &format!("blk.{layer}.ffn_gate.weight"),
                )?,
                ffn_up: ResidentQuantMatrix::upload(
                    device,
                    gguf,
                    &format!("blk.{layer}.ffn_up.weight"),
                )?,
                ffn_down: ResidentQuantMatrix::upload(
                    device,
                    gguf,
                    &format!("blk.{layer}.ffn_down.weight"),
                )?,
                vocab_size: config.vocab_size,
                embedding_length: dim,
            });
        }
        Ok(Some(layers))
    }

    fn is_q8_0_only(&self) -> bool {
        [
            &self.attn_q,
            &self.attn_k,
            &self.attn_v,
            &self.attn_output,
            &self.ffn_gate,
            &self.ffn_up,
            &self.ffn_down,
        ]
        .into_iter()
        .all(ResidentQuantMatrix::is_q8_0)
    }
}

fn matches_f32_vector(gguf: &GgufFile, name: &str, len: usize) -> bool {
    gguf.tensor_info(name)
        .is_some_and(|info| is_supported_resident_float_dtype(info.dtype) && info.numel() == len)
}

fn matches_optional_f32_vector(gguf: &GgufFile, name: &str, len: usize) -> bool {
    match gguf.tensor_info(name) {
        Some(info) => is_supported_resident_float_dtype(info.dtype) && info.numel() == len,
        None => true,
    }
}

fn matches_optional_qk_norm_pair(gguf: &GgufFile, layer: usize, head_dim: usize) -> bool {
    let q_name = format!("blk.{layer}.attn_q_norm.weight");
    let k_name = format!("blk.{layer}.attn_k_norm.weight");
    match (gguf.tensor_info(&q_name), gguf.tensor_info(&k_name)) {
        (None, None) => true,
        (Some(q), Some(k)) => {
            is_supported_resident_float_dtype(q.dtype)
                && is_supported_resident_float_dtype(k.dtype)
                && q.numel() == head_dim
                && k.numel() == head_dim
        }
        _ => false,
    }
}

fn upload_optional_f32_tensor(
    device: &CudaDevice,
    gguf: &GgufFile,
    name: &str,
) -> Result<Option<GpuF32Tensor>> {
    if gguf.tensor_info(name).is_some() {
        device.upload_f32_tensor(gguf, name).map(Some)
    } else {
        Ok(None)
    }
}

fn matches_supported_linear_shape(gguf: &GgufFile, name: &str, rows: usize, cols: usize) -> bool {
    gguf.tensor_info(name).is_some_and(|info| {
        is_supported_resident_linear_dtype(info.dtype)
            && info.rows() == rows
            && info.row_len() == cols
    })
}

fn is_supported_resident_linear_dtype(dtype: DType) -> bool {
    is_supported_resident_float_dtype(dtype)
        || matches!(
            dtype,
            DType::Q8_0 | DType::Q4_0 | DType::Q4_K | DType::Q5_K | DType::Q6_K
        )
}

fn is_supported_resident_float_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::F16 | DType::BF16)
}

impl CausalLmBackend for CudaResidentBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::CudaResident
    }

    fn model_name(&self) -> &str {
        self.model.model_name()
    }

    fn config(&self) -> &LlamaConfig {
        self.model.config()
    }

    fn new_session(&self, cache_mode: KvCacheMode, page_tokens: usize) -> BackendSession {
        let config = self.model.config();
        BackendSession::new_cuda_with_kv_budget_and_page_tokens(
            self.device.clone(),
            cache_mode,
            config.block_count,
            config.kv_width(),
            config.context_length,
            page_tokens,
            Some(self.kv_budget_bytes),
        )
    }

    fn clear_state(&self) {
        self.model.clear_state();
    }

    fn save_state(&self) -> Option<BackendStateSnapshot> {
        self.model.save_state()
    }

    fn restore_state(&self, snapshot: &[Option<(Vec<f32>, Vec<f32>)>]) {
        self.model.restore_state(snapshot);
    }

    fn forward_token(
        &self,
        token_id: u32,
        position: usize,
        session: &mut BackendSession,
        output_logits: &mut Vec<f32>,
    ) -> Result<()> {
        if self.try_forward_token_q8_0(token_id, position, session, output_logits)? {
            return Ok(());
        }
        Err(Self::decode_unsupported())
    }

    fn forward_draft(
        &self,
        token_id: u32,
        position: usize,
        n_layers: usize,
        session: &mut BackendSession,
        output_logits: &mut Vec<f32>,
    ) -> Result<()> {
        if self.try_forward_token_q8_0_with_logits(
            token_id,
            position,
            session,
            output_logits,
            true,
            None,
            cuda_total_len_for_position(position)?,
            Some(n_layers),
        )? {
            return Ok(());
        }
        Err(Self::decode_unsupported())
    }

    fn forward_batch(
        &self,
        token_ids: &[u32],
        start_position: usize,
        session: &mut BackendSession,
    ) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(XrtError::Runtime("empty token batch".to_string()));
        }
        let mut logits = Vec::new();
        let last_index = token_ids.len() - 1;
        let batch_total_len = cuda_total_len_after_batch(start_position, token_ids.len())?;
        for (index, token_id) in token_ids.iter().copied().enumerate() {
            let position = cuda_batch_position(start_position, index)?;
            if self.try_forward_token_q8_0_with_logits(
                token_id,
                position,
                session,
                &mut logits,
                index == last_index,
                None,
                batch_total_len,
                None,
            )? {
                continue;
            }
            return Err(Self::decode_unsupported());
        }
        Ok(logits)
    }

    fn forward_batch_with_embeddings(
        &self,
        token_ids: &[u32],
        start_position: usize,
        session: &mut BackendSession,
        embedding_overrides: HashMap<usize, Vec<f32>>,
    ) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(XrtError::Runtime("empty token batch".to_string()));
        }
        Self::validate_embedding_overrides(
            token_ids.len(),
            self.model.config().embedding_length,
            &embedding_overrides,
        )?;
        let mut logits = Vec::new();
        let last_index = token_ids.len() - 1;
        let batch_total_len = cuda_total_len_after_batch(start_position, token_ids.len())?;
        for (index, token_id) in token_ids.iter().copied().enumerate() {
            let position = cuda_batch_position(start_position, index)?;
            if self.try_forward_token_q8_0_with_logits(
                token_id,
                position,
                session,
                &mut logits,
                index == last_index,
                embedding_overrides.get(&index).map(Vec::as_slice),
                batch_total_len,
                None,
            )? {
                continue;
            }
            return Err(Self::decode_unsupported());
        }
        Ok(logits)
    }

    fn forward_batch_all_logits(
        &self,
        token_ids: &[u32],
        start_position: usize,
        session: &mut BackendSession,
    ) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(XrtError::Runtime("empty token batch".to_string()));
        }
        let vocab_size = self.model.config().vocab_size;
        let output_len = cuda_all_logits_output_len(token_ids.len(), vocab_size)?;
        let mut all_logits = Vec::with_capacity(output_len);
        let mut logits = Vec::new();
        let batch_total_len = cuda_total_len_after_batch(start_position, token_ids.len())?;
        for (index, token_id) in token_ids.iter().copied().enumerate() {
            let position = cuda_batch_position(start_position, index)?;
            if !self.try_forward_token_q8_0_with_logits(
                token_id,
                position,
                session,
                &mut logits,
                true,
                None,
                batch_total_len,
                None,
            )? {
                return Err(Self::decode_unsupported());
            }
            all_logits.extend_from_slice(&logits);
        }
        Ok(all_logits)
    }

    fn embedding_lookup(&self, token_id: usize) -> Result<Vec<f32>> {
        self.model.embedding_lookup(token_id)
    }

    fn model_weight_bytes(&self) -> u64 {
        self.resident_model_weight_bytes
    }

    fn cuda_device_name(&self) -> Option<&str> {
        self.device_name.as_deref()
    }

    fn cuda_free_vram_bytes(&self) -> Option<u64> {
        self.free_vram_bytes
    }

    fn cuda_total_vram_bytes(&self) -> Option<u64> {
        self.total_vram_bytes
    }

    fn cuda_kv_budget_bytes(&self) -> Option<u64> {
        Some(self.kv_budget_bytes)
    }

    fn resident_f32_probe_available(&self) -> bool {
        self.f32_probe.is_some()
    }

    fn resident_q8_0_probe_available(&self) -> bool {
        self.q8_0_probe
            .as_ref()
            .is_some_and(ResidentQ8_0ProbeWeights::is_q8_0_only)
    }

    fn resident_q8_0_layer0_probe_available(&self) -> bool {
        self.q8_0_layer_probes.as_ref().is_some_and(|layers| {
            layers
                .first()
                .is_some_and(ResidentQ8_0LayerWeights::is_q8_0_only)
        })
    }

    fn resident_dense_quant_decode_available(&self) -> bool {
        let Some(probe) = &self.q8_0_probe else {
            return false;
        };
        dense_quant_decode_status_available(
            true,
            probe.token_embedding_gpu_resident(),
            self.q8_0_layer_probes.as_ref().map(Vec::len),
            self.model.config().block_count,
        )
    }
}

fn cuda_all_logits_output_len(token_count: usize, vocab_size: usize) -> Result<usize> {
    checked_mul(token_count, vocab_size, "CUDA all-logits batch output")
}

fn cuda_estimated_resident_upload_bytes(gguf: &GgufFile, config: &LlamaConfig) -> Result<u64> {
    if !CudaResidentBackend::supports_dense_quant_decode(gguf, config) {
        return Err(CudaResidentBackend::decode_unsupported());
    }
    let output_name = ResidentQ8_0ProbeWeights::output_name(gguf);
    gguf.tensor_infos().iter().try_fold(0u64, |total, info| {
        total
            .checked_add(cuda_extra_resident_tensor_bytes(info, output_name)?)
            .ok_or_else(|| {
                XrtError::Runtime("CUDA resident upload byte count overflow".to_string())
            })
    })
}

fn cuda_extra_resident_tensor_bytes(info: &xrt_gguf::TensorInfo, output_name: &str) -> Result<u64> {
    if info.name == "token_embd.weight" {
        let embedding_bytes = cuda_embedding_resident_tensor_bytes(info)?;
        let tied_output_bytes = if output_name == "token_embd.weight"
            && is_supported_resident_float_dtype(info.dtype)
        {
            cuda_matrix_resident_tensor_bytes(info)?
        } else {
            0
        };
        return embedding_bytes
            .checked_add(tied_output_bytes)
            .ok_or_else(|| {
                XrtError::Runtime("CUDA resident token embedding byte count overflow".to_string())
            });
    }
    cuda_matrix_resident_tensor_bytes(info)
}

fn cuda_embedding_resident_tensor_bytes(info: &xrt_gguf::TensorInfo) -> Result<u64> {
    match info.dtype {
        DType::Q4_K | DType::Q5_K | DType::Q6_K => {
            let bytes = cuda_resident_f32_tensor_bytes(info)?;
            bytes.checked_mul(2).ok_or_else(|| {
                XrtError::Runtime("CUDA resident K-quant embedding byte count overflow".to_string())
            })
        }
        _ => cuda_matrix_resident_tensor_bytes(info),
    }
}

fn cuda_resident_f32_tensor_bytes(info: &xrt_gguf::TensorInfo) -> Result<u64> {
    checked_mul(info.numel(), 4, "CUDA resident F32 tensor bytes").map(|v| v as u64)
}

fn cuda_quant_block_count(info: &xrt_gguf::TensorInfo) -> Result<u64> {
    if info.row_len() % info.dtype.block_size() != 0 {
        return Err(XrtError::InvalidTensor(format!(
            "tensor `{}` row length {} is not divisible by {:?} block size {}",
            info.name,
            info.row_len(),
            info.dtype,
            info.dtype.block_size()
        )));
    }
    let blocks_per_row = info.row_len() / info.dtype.block_size();
    checked_mul(
        info.rows(),
        blocks_per_row,
        "CUDA resident quant block count",
    )
    .map(|v| v as u64)
}

fn cuda_matrix_resident_tensor_bytes(info: &xrt_gguf::TensorInfo) -> Result<u64> {
    let f32_bytes =
        || checked_mul(info.numel(), 4, "CUDA resident F32 tensor bytes").map(|v| v as u64);
    let blocks = || cuda_quant_block_count(info);

    match info.dtype {
        DType::F32 | DType::F16 | DType::BF16 => f32_bytes(),
        DType::Q8_0 | DType::Q4_0 => blocks().and_then(|blocks| {
            blocks
                .checked_mul((4 + info.dtype.block_size()) as u64)
                .ok_or_else(|| {
                    XrtError::Runtime("CUDA resident Q4_0/Q8_0 byte count overflow".to_string())
                })
        }),
        DType::Q4_K => blocks().and_then(|blocks| {
            // split Q4_K resident layout: d + dmin + scales + quants.
            blocks
                .checked_mul((4 + 4 + 12 + 128) as u64)
                .ok_or_else(|| {
                    XrtError::Runtime("CUDA resident Q4_K byte count overflow".to_string())
                })
        }),
        DType::Q5_K | DType::Q6_K => f32_bytes(),
    }
}

fn cuda_model_upload_budget_bytes(
    free_vram_bytes: u64,
    total_vram_bytes: u64,
    config: GpuResourceConfig,
) -> u64 {
    let fraction_limited_total =
        ((total_vram_bytes as f64) * f64::from(config.memory_fraction)).floor() as u64;
    free_vram_bytes
        .min(fraction_limited_total)
        .saturating_sub(config.reserved_bytes())
}

fn cuda_kv_budget_bytes(
    upload_budget_bytes: u64,
    model_weight_bytes: u64,
    config: GpuResourceConfig,
) -> u64 {
    ((upload_budget_bytes.saturating_sub(model_weight_bytes) as f64)
        * f64::from(config.kv_fraction))
    .floor() as u64
}

fn cuda_kv_growth_capacity(
    current_capacity: usize,
    required_len: usize,
    page_tokens: usize,
    max_len: usize,
) -> Result<usize> {
    if required_len == 0 {
        return Ok(current_capacity);
    }
    if required_len > max_len || current_capacity > max_len {
        return Err(XrtError::Runtime(format!(
            "CUDA KV growth requires length {required_len} from capacity {current_capacity}, but context length is {max_len}"
        )));
    }

    let initial_capacity = page_tokens.max(1).min(max_len);
    let mut capacity = current_capacity.max(initial_capacity);
    while capacity < required_len {
        let next = capacity.saturating_mul(2).min(max_len);
        if next <= capacity {
            return Err(XrtError::Runtime(format!(
                "CUDA KV capacity cannot grow from {capacity} to required length {required_len}"
            )));
        }
        capacity = next;
    }
    Ok(capacity)
}

fn cuda_layer_kv_allocated_bytes(mode: KvCacheMode, capacity: usize, width: usize) -> Result<u64> {
    let elements = checked_mul(capacity, width, "CUDA KV cache elements")?;
    match mode {
        KvCacheMode::F32 => elements
            .checked_mul(2 * std::mem::size_of::<f32>())
            .map(|bytes| bytes as u64)
            .ok_or_else(|| XrtError::Runtime("CUDA F32 KV cache byte count overflow".to_string())),
        KvCacheMode::Q8 => {
            let scale_bytes = checked_mul(
                capacity,
                std::mem::size_of::<f32>(),
                "CUDA Q8 KV cache scale bytes",
            )?;
            elements
                .checked_mul(2)
                .and_then(|bytes| bytes.checked_add(scale_bytes.checked_mul(2)?))
                .map(|bytes| bytes as u64)
                .ok_or_else(|| {
                    XrtError::Runtime("CUDA Q8 KV cache byte count overflow".to_string())
                })
        }
        KvCacheMode::KeyQ4ValueQ8 | KvCacheMode::AgentAdaptive => {
            let key_bytes = checked_mul(capacity, width.div_ceil(2), "CUDA KQ4/VQ8 key bytes")?;
            let value_bytes = elements;
            let key_scale_count =
                checked_mul(capacity, width.div_ceil(32), "CUDA KQ4/VQ8 key scale count")?;
            let key_scale_bytes = checked_mul(
                key_scale_count,
                std::mem::size_of::<f32>(),
                "CUDA KQ4/VQ8 key scale bytes",
            )?;
            let value_scale_bytes = checked_mul(
                capacity,
                std::mem::size_of::<f32>(),
                "CUDA KQ4/VQ8 value scale bytes",
            )?;
            let quant_bytes = key_bytes
                .checked_add(value_bytes)
                .and_then(|bytes| bytes.checked_add(key_scale_bytes))
                .and_then(|bytes| bytes.checked_add(value_scale_bytes))
                .ok_or_else(|| {
                    XrtError::Runtime("CUDA KQ4/VQ8 KV cache byte count overflow".to_string())
                })?;
            if mode == KvCacheMode::AgentAdaptive {
                let f32_bytes = elements
                    .checked_mul(2 * std::mem::size_of::<f32>())
                    .ok_or_else(|| {
                        XrtError::Runtime(
                            "CUDA adaptive hot KV cache byte count overflow".to_string(),
                        )
                    })?;
                f32_bytes
                    .checked_add(quant_bytes)
                    .map(|bytes| bytes as u64)
                    .ok_or_else(|| {
                        XrtError::Runtime(
                            "CUDA adaptive mixed KV cache byte count overflow".to_string(),
                        )
                    })
            } else {
                Ok(quant_bytes as u64)
            }
        }
    }
}

fn cuda_session_kv_allocated_bytes(
    mode: KvCacheMode,
    layer_count: usize,
    capacity: usize,
    width: usize,
) -> Result<u64> {
    let layer_bytes = cuda_layer_kv_allocated_bytes(mode, capacity, width)?;
    (layer_count as u64)
        .checked_mul(layer_bytes)
        .ok_or_else(|| XrtError::Runtime("CUDA session KV cache byte count overflow".to_string()))
}

fn cuda_total_len_for_position(position: usize) -> Result<usize> {
    position
        .checked_add(1)
        .ok_or_else(|| XrtError::Runtime("CUDA token position overflow".to_string()))
}

fn cuda_batch_position(start_position: usize, index: usize) -> Result<usize> {
    start_position
        .checked_add(index)
        .ok_or_else(|| XrtError::Runtime("CUDA batch position overflow".to_string()))
}

fn cuda_total_len_after_batch(start_position: usize, batch_len: usize) -> Result<usize> {
    start_position
        .checked_add(batch_len)
        .ok_or_else(|| XrtError::Runtime("CUDA batch total length overflow".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_backend_aliases() {
        assert_eq!(BackendKind::parse("auto"), Some(BackendKind::Auto));
        assert_eq!(BackendKind::parse("CPU"), Some(BackendKind::Cpu));
        assert_eq!(BackendKind::parse("cuda"), Some(BackendKind::CudaResident));
        assert_eq!(
            BackendKind::parse("cuda_resident"),
            Some(BackendKind::CudaResident)
        );
        assert_eq!(
            BackendKind::parse("external-openai"),
            Some(BackendKind::ExternalOpenAi)
        );
        assert_eq!(BackendKind::parse("unknown"), None);
    }

    #[test]
    fn backend_selection_resolves_active_backend() {
        assert_eq!(
            BackendKind::Auto.resolve_active().unwrap(),
            BackendKind::Cpu
        );
        assert_eq!(BackendKind::Cpu.resolve_active().unwrap(), BackendKind::Cpu);
        assert_eq!(
            BackendKind::CudaResident.resolve_active().unwrap(),
            BackendKind::CudaResident
        );
    }

    #[test]
    fn cuda_profile_value_requires_truthy_value() {
        for value in ["", " ", "0", "false", "FALSE", "off", "Off"] {
            assert!(!CudaResidentBackend::cuda_profile_value_enabled(value));
        }
        for value in ["1", "true", "yes", "on", "profile"] {
            assert!(CudaResidentBackend::cuda_profile_value_enabled(value));
        }
    }

    #[test]
    fn embedding_overrides_validate_before_cuda_prefill() {
        let mut overrides = HashMap::from([(1usize, vec![0.0; 4])]);
        assert!(CudaResidentBackend::validate_embedding_overrides(2, 4, &overrides).is_ok());

        overrides.insert(2, vec![0.0; 4]);
        assert!(CudaResidentBackend::validate_embedding_overrides(2, 4, &overrides).is_err());

        overrides.clear();
        overrides.insert(0, vec![0.0; 3]);
        assert!(CudaResidentBackend::validate_embedding_overrides(2, 4, &overrides).is_err());
    }

    #[test]
    fn resident_linear_dtype_support_includes_f32_and_current_quant_formats() {
        for dtype in [
            DType::F32,
            DType::F16,
            DType::BF16,
            DType::Q8_0,
            DType::Q4_0,
            DType::Q4_K,
            DType::Q5_K,
            DType::Q6_K,
        ] {
            assert!(is_supported_resident_linear_dtype(dtype));
        }
    }

    #[test]
    fn q8_0_probe_status_requires_q8_embedding_and_output() {
        assert!(q8_0_probe_status_available(true, true));
        assert!(!q8_0_probe_status_available(false, true));
        assert!(!q8_0_probe_status_available(true, false));
    }

    #[test]
    fn dense_decode_status_requires_gpu_resident_embedding_and_all_layers() {
        assert!(dense_quant_decode_status_available(true, true, Some(2), 2));
        assert!(!dense_quant_decode_status_available(
            false,
            true,
            Some(2),
            2
        ));
        assert!(!dense_quant_decode_status_available(
            true,
            false,
            Some(2),
            2
        ));
        assert!(!dense_quant_decode_status_available(true, true, Some(1), 2));
        assert!(!dense_quant_decode_status_available(true, true, None, 2));
    }

    #[test]
    fn cuda_decode_unsupported_message_lists_current_dense_formats() {
        let message = CudaResidentBackend::decode_unsupported().to_string();
        for dtype in ["F32", "F16", "BF16", "Q8_0", "Q4_0", "Q4_K", "Q5_K", "Q6_K"] {
            assert!(message.contains(dtype), "missing {dtype} in {message}");
        }
    }

    #[test]
    fn cuda_all_logits_output_len_checks_overflow() {
        assert_eq!(cuda_all_logits_output_len(3, 5).unwrap(), 15);
        assert!(cuda_all_logits_output_len(usize::MAX, 2).is_err());
    }

    #[test]
    fn cuda_model_upload_budget_applies_fraction_free_and_reserve() {
        let config = GpuResourceConfig {
            device_ordinal: 0,
            memory_fraction: 0.5,
            reserved_mb: 1,
            kv_fraction: 0.3,
        };
        assert_eq!(
            cuda_model_upload_budget_bytes(8 * 1024 * 1024, 16 * 1024 * 1024, config),
            7 * 1024 * 1024
        );
        assert_eq!(
            cuda_model_upload_budget_bytes(4 * 1024 * 1024, 16 * 1024 * 1024, config),
            3 * 1024 * 1024
        );
        assert_eq!(
            cuda_model_upload_budget_bytes(512 * 1024, 16 * 1024 * 1024, config),
            0
        );
    }

    #[test]
    fn cuda_kv_budget_uses_remaining_safe_vram_fraction() {
        let config = GpuResourceConfig {
            device_ordinal: 0,
            memory_fraction: 1.0,
            reserved_mb: 0,
            kv_fraction: 0.25,
        };
        assert_eq!(cuda_kv_budget_bytes(1000, 200, config), 200);
        assert_eq!(cuda_kv_budget_bytes(1000, 1200, config), 0);
    }

    #[test]
    fn cuda_session_kv_byte_estimate_matches_cache_modes() {
        assert_eq!(
            cuda_session_kv_allocated_bytes(KvCacheMode::F32, 2, 8, 4).unwrap(),
            512
        );
        assert_eq!(
            cuda_session_kv_allocated_bytes(KvCacheMode::Q8, 2, 8, 4).unwrap(),
            256
        );
        assert_eq!(
            cuda_session_kv_allocated_bytes(KvCacheMode::KeyQ4ValueQ8, 2, 8, 64).unwrap(),
            1728
        );
        assert_eq!(
            cuda_session_kv_allocated_bytes(KvCacheMode::AgentAdaptive, 2, 8, 64).unwrap(),
            9920
        );
    }

    #[test]
    fn cuda_kv_capacity_grows_by_pages_then_doubles_within_context() {
        assert_eq!(cuda_kv_growth_capacity(0, 1, 32, 4096).unwrap(), 32);
        assert_eq!(cuda_kv_growth_capacity(32, 33, 32, 4096).unwrap(), 64);
        assert_eq!(cuda_kv_growth_capacity(64, 129, 32, 4096).unwrap(), 256);
        assert_eq!(cuda_kv_growth_capacity(256, 300, 32, 300).unwrap(), 300);
        assert!(cuda_kv_growth_capacity(256, 301, 32, 300).is_err());
    }

    #[test]
    fn cuda_extra_resident_tensor_bytes_accounts_for_expanded_and_tied_formats() {
        let q8 = xrt_gguf::TensorInfo {
            name: "blk.0.attn_q.weight".to_string(),
            dimensions: vec![64, 2],
            strides: vec![],
            dtype: DType::Q8_0,
            offset: 0,
            nbytes: 0,
        };
        assert_eq!(
            cuda_extra_resident_tensor_bytes(&q8, "output.weight").unwrap(),
            4 * 36
        );

        let q4_embedding = xrt_gguf::TensorInfo {
            name: "token_embd.weight".to_string(),
            dimensions: vec![256, 3],
            strides: vec![],
            dtype: DType::Q4_K,
            offset: 0,
            nbytes: 0,
        };
        assert_eq!(
            cuda_extra_resident_tensor_bytes(&q4_embedding, "output.weight").unwrap(),
            256 * 3 * 4 * 2
        );
        assert_eq!(
            cuda_extra_resident_tensor_bytes(&q4_embedding, "token_embd.weight").unwrap(),
            256 * 3 * 4 * 2
        );

        let q6_embedding = xrt_gguf::TensorInfo {
            name: "token_embd.weight".to_string(),
            dimensions: vec![256, 3],
            strides: vec![],
            dtype: DType::Q6_K,
            offset: 0,
            nbytes: 0,
        };
        assert_eq!(
            cuda_extra_resident_tensor_bytes(&q6_embedding, "output.weight").unwrap(),
            256 * 3 * 4 * 2
        );
        assert_eq!(
            cuda_extra_resident_tensor_bytes(&q6_embedding, "token_embd.weight").unwrap(),
            256 * 3 * 4 * 2
        );
    }

    #[test]
    fn cuda_position_helpers_check_overflow() {
        assert_eq!(cuda_total_len_for_position(0).unwrap(), 1);
        assert_eq!(cuda_batch_position(4, 2).unwrap(), 6);
        assert!(cuda_total_len_for_position(usize::MAX).is_err());
        assert!(cuda_batch_position(usize::MAX, 1).is_err());
    }

    #[test]
    fn layer0_projection_probe_rejects_nonzero_position() {
        assert!(CudaResidentBackend::validate_layer0_probe_position(0).is_ok());
        let err = CudaResidentBackend::validate_layer0_probe_position(1).unwrap_err();
        assert!(matches!(err, XrtError::Unsupported(message) if message.contains("position 0")));
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn cuda_session_rejects_lengths_beyond_context_before_allocating() {
        let mut session = BackendSession::new_cuda(CudaDevice, KvCacheMode::F32, 1, 4, 2);
        assert_eq!(session.cache_mode(), KvCacheMode::F32);
        assert_eq!(session.cuda_kv_allocated_bytes(), 0);
        let err = session.prepare_for_total_len(3).unwrap_err();
        assert!(
            matches!(err, XrtError::Runtime(message) if message.contains("exceeds context length"))
        );
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn cuda_session_zero_length_prepare_stays_unallocated() {
        let mut session = BackendSession::new_cuda(CudaDevice, KvCacheMode::F32, 2, 4, 8);
        session.prepare_for_total_len(0).unwrap();
        session.truncate(0);
        assert_eq!(session.cuda_kv_allocated_bytes(), 0);

        let err = session.cuda_layer_cache_mut(0).unwrap_err();
        assert!(matches!(
            err,
            XrtError::Runtime(message) if message.contains("missing CUDA KV cache for layer 0")
        ));
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn cuda_session_rejects_kv_allocation_over_budget_before_allocating() {
        let mut session = BackendSession::new_cuda_with_kv_budget(
            CudaDevice,
            KvCacheMode::F32,
            2,
            4,
            8,
            Some(511),
        );
        let err = session.prepare_for_total_len(1).unwrap_err();
        assert!(matches!(err, XrtError::Cuda(message) if message.contains("configured KV budget")));
        assert_eq!(session.cuda_kv_allocated_bytes(), 0);
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn non_cuda_session_maps_quantized_modes_to_f32() {
        let mut session = BackendSession::new_cuda(CudaDevice, KvCacheMode::AgentAdaptive, 1, 4, 8);
        assert_eq!(session.requested_cache_mode(), KvCacheMode::AgentAdaptive);
        assert_eq!(session.cache_mode(), KvCacheMode::F32);

        session.replace_cache(KvCacheMode::Q8, 1, 4, 32);
        assert_eq!(session.requested_cache_mode(), KvCacheMode::Q8);
        assert_eq!(session.cache_mode(), KvCacheMode::F32);

        session.replace_cache(KvCacheMode::KeyQ4ValueQ8, 1, 4, 32);
        assert_eq!(session.requested_cache_mode(), KvCacheMode::KeyQ4ValueQ8);
        assert_eq!(session.cache_mode(), KvCacheMode::F32);
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn cuda_session_retains_policy_metadata_for_future_adaptive_router() {
        let mut session = BackendSession::new_cuda(CudaDevice, KvCacheMode::AgentAdaptive, 1, 4, 8);
        let policy = SessionPolicy::agent_adaptive();
        let spans = [PromptSpan {
            kind: crate::policy::PromptSpanKind::System,
            token_start: 0,
            token_end: 2,
        }];

        session.configure_policy(policy.clone(), 4, &spans);
        match &session {
            BackendSession::Cuda {
                policy: stored_policy,
                prompt_token_count,
                prompt_spans,
                ..
            } => {
                assert_eq!(stored_policy, &policy);
                assert_eq!(*prompt_token_count, 4);
                assert_eq!(prompt_spans, &spans);
            }
            BackendSession::Cpu { .. } => panic!("expected CUDA session"),
        }

        session.replace_cache(KvCacheMode::Q8, 1, 4, 32);
        match &session {
            BackendSession::Cuda {
                policy: stored_policy,
                prompt_token_count,
                prompt_spans,
                ..
            } => {
                assert_eq!(stored_policy, &SessionPolicy::default());
                assert_eq!(*prompt_token_count, 0);
                assert!(prompt_spans.is_empty());
            }
            BackendSession::Cpu { .. } => panic!("expected CUDA session"),
        }
    }

    #[test]
    fn cuda_adaptive_position_routing_matches_policy() {
        let policy = SessionPolicy {
            recent_window_tokens: 2,
            ..SessionPolicy::agent_adaptive()
        };
        let spans = [
            PromptSpan {
                kind: crate::policy::PromptSpanKind::System,
                token_start: 0,
                token_end: 2,
            },
            PromptSpan {
                kind: crate::policy::PromptSpanKind::User,
                token_start: 2,
                token_end: 4,
            },
            PromptSpan {
                kind: crate::policy::PromptSpanKind::ToolResult,
                token_start: 4,
                token_end: 6,
            },
            PromptSpan {
                kind: crate::policy::PromptSpanKind::Developer,
                token_start: 8,
                token_end: 12,
            },
        ];

        assert!(BackendSession::cuda_adaptive_position_is_hot_for_policy(
            &policy, 10, &spans, 0, 10
        ));
        assert!(!BackendSession::cuda_adaptive_position_is_hot_for_policy(
            &policy, 10, &spans, 2, 10
        ));
        assert!(BackendSession::cuda_adaptive_position_is_hot_for_policy(
            &policy, 10, &spans, 5, 10
        ));
        assert!(BackendSession::cuda_adaptive_position_is_hot_for_policy(
            &policy, 10, &spans, 8, 10
        ));
        assert!(!BackendSession::cuda_adaptive_position_is_hot_for_policy(
            &policy, 10, &spans, 7, 10
        ));
        assert!(BackendSession::cuda_adaptive_position_is_hot_for_policy(
            &policy, 10, &spans, 9, 10
        ));
    }

    #[test]
    fn cuda_adaptive_route_migration_needed_detects_mask_drift() {
        assert!(!BackendSession::cuda_adaptive_route_migration_needed(
            &[1, 0, 1, 1],
            &[1, 0, 1, 1, 1]
        ));
        assert!(BackendSession::cuda_adaptive_route_migration_needed(
            &[1, 1, 1, 1],
            &[0, 0, 1, 1]
        ));
        assert!(BackendSession::cuda_adaptive_route_migration_needed(
            &[0, 0, 1, 1],
            &[1, 0, 1, 1]
        ));
        assert!(BackendSession::cuda_adaptive_route_migration_needed(
            &[1, 0, 1, 1],
            &[1, 0, 1]
        ));
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn cuda_session_adaptive_router_uses_retained_policy_metadata() {
        let mut session = BackendSession::new_cuda(CudaDevice, KvCacheMode::AgentAdaptive, 1, 4, 8);
        let policy = SessionPolicy {
            recent_window_tokens: 2,
            ..SessionPolicy::agent_adaptive()
        };
        let spans = [PromptSpan {
            kind: crate::policy::PromptSpanKind::System,
            token_start: 0,
            token_end: 2,
        }];

        session.configure_policy(policy, 4, &spans);
        assert!(session.cuda_adaptive_position_is_hot(0, 6));
        assert!(!session.cuda_adaptive_position_is_hot(2, 6));
        assert!(session.cuda_adaptive_position_is_hot(5, 6));
        assert_eq!(
            session.cuda_adaptive_hot_position_mask(6),
            Some(vec![1, 1, 0, 0, 1, 1])
        );

        session.replace_cache(KvCacheMode::Q8, 1, 4, 8);
        assert!(!session.cuda_adaptive_position_is_hot(0, 6));
        assert_eq!(session.cuda_adaptive_hot_position_mask(6), None);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_feature_session_can_select_quantized_gpu_kv() {
        assert_eq!(
            BackendSession::cuda_cache_mode(KvCacheMode::Q8),
            KvCacheMode::Q8
        );
        assert_eq!(
            BackendSession::cuda_cache_mode(KvCacheMode::KeyQ4ValueQ8),
            KvCacheMode::KeyQ4ValueQ8
        );
        assert_eq!(
            BackendSession::cuda_cache_mode(KvCacheMode::AgentAdaptive),
            KvCacheMode::AgentAdaptive
        );
    }

    #[test]
    fn cuda_cache_layout_changes_when_mode_or_shape_changes() {
        assert!(!BackendSession::cuda_cache_layout_changed(
            KvCacheMode::F32,
            KvCacheMode::F32,
            2,
            2,
            8,
            8,
        ));
        assert!(BackendSession::cuda_cache_layout_changed(
            KvCacheMode::F32,
            KvCacheMode::Q8,
            2,
            2,
            8,
            8,
        ));
        assert!(BackendSession::cuda_cache_layout_changed(
            KvCacheMode::F32,
            KvCacheMode::F32,
            2,
            3,
            8,
            8,
        ));
        assert!(BackendSession::cuda_cache_layout_changed(
            KvCacheMode::F32,
            KvCacheMode::F32,
            2,
            2,
            8,
            16,
        ));
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn cuda_replace_cache_updates_shape_without_replacing_context_len() {
        let mut session = BackendSession::new_cuda(CudaDevice, KvCacheMode::F32, 1, 4, 8);

        session.replace_cache(KvCacheMode::Q8, 3, 6, 32);

        match session {
            BackendSession::Cuda {
                cache_mode,
                layer_caches,
                layer_count,
                width,
                max_len,
                page_tokens,
                ..
            } => {
                assert_eq!(cache_mode, KvCacheMode::F32);
                assert!(layer_caches.is_empty());
                assert_eq!(layer_count, 3);
                assert_eq!(width, 6);
                assert_eq!(max_len, 8);
                assert_eq!(page_tokens, 32);
            }
            BackendSession::Cpu { .. } => panic!("expected CUDA session"),
        }
    }
}
