use std::{
    collections::{HashMap, VecDeque},
    env, fmt,
    sync::Arc,
    time::Instant,
};

use crate::{
    gpu_resource::{CudaGraphMode, GpuResourceConfig},
    kv_cache::{KvCacheMode, SessionKvCache},
    policy::{PromptSpan, SessionPolicy},
    resident_tensor::{
        GgufResidentTensorSource, HfStandardDenseResidentTensorSource, ResidentTensorInfo,
        ResidentTensorSource, ResidentTensorStorage,
    },
};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tracing::info;
use xrt_core::{checked_mul, decode_bf16, decode_f16, DType, KvCache, Result, XrtError};
use xrt_cuda::{
    CudaAdaptiveKvRoutes, CudaAllocationStats, CudaAwqGemm4Matrix, CudaAwqGemv4Matrix,
    CudaCompressedTensorsW4A16Matrix, CudaDecodeParams, CudaDevice, CudaExecutionStream,
    CudaF32Buffer, CudaF32KvPagePool, CudaGptqExplicitGemm4Matrix, CudaGptqGemm4Matrix,
    CudaGraphExec, CudaKeyQ4ValueQ8LayerKvCache, CudaLayerKvCache, CudaMemoryPoolStats,
    CudaQ4KMatrix, CudaQ4_0Matrix, CudaQ5KMatrix, CudaQ6KMatrix, CudaQ8LayerKvCache,
    CudaQ8_0Matrix, CudaSharedF32LayerKvCache, CudaTransferStats, GpuF32Tensor,
};
use xrt_gguf::GgufFile;
use xrt_models::{Gemma4LayerTrace, LlamaConfig, LlamaModel};
use xrt_safetensors::HfModelBundle;

// Keep the faster expanded path for smaller vocabularies without allowing its
// two F32 copies and upload temporaries to exhaust host memory on large models.
const CUDA_K_QUANT_EXPANDED_EMBEDDING_MAX_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const CUDA_DECODE_BATCH_GRAPH_CACHE_ENTRIES: usize = 8;
const CUDA_SHARED_KV_MAX_REPLICAS: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
                "external-openai is an HTTP proxy mode provided by xrt-server; it is not a token-level xrt-runtime backend"
                    .to_string(),
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
pub(crate) enum BackendPrefixSnapshot {
    Cpu {
        cache: SessionKvCache,
        prefix_len: usize,
        allocated_bytes: u64,
    },
    Cuda {
        layer_caches: Arc<Vec<CudaLayerKvStore>>,
        cache_mode: KvCacheMode,
        layer_widths: Vec<usize>,
        page_tokens: usize,
        prefix_len: usize,
        allocated_bytes: u64,
    },
}

impl BackendPrefixSnapshot {
    pub(crate) fn prefix_len(&self) -> usize {
        match self {
            Self::Cpu { prefix_len, .. } => *prefix_len,
            Self::Cuda { prefix_len, .. } => *prefix_len,
        }
    }

    pub(crate) fn allocated_bytes(&self) -> u64 {
        match self {
            Self::Cpu {
                allocated_bytes, ..
            } => *allocated_bytes,
            Self::Cuda {
                allocated_bytes, ..
            } => *allocated_bytes,
        }
    }
}

pub struct BackendDecodeBatchItem {
    pub(crate) sequence_id: u64,
    pub(crate) token_id: u32,
    pub(crate) position: usize,
    pub(crate) session: BackendSession,
    pub(crate) output_logits: Vec<f32>,
}

impl BackendDecodeBatchItem {
    pub fn new(sequence_id: u64, token_id: u32, position: usize, session: BackendSession) -> Self {
        Self {
            sequence_id,
            token_id,
            position,
            session,
            output_logits: Vec::new(),
        }
    }

    pub fn output_logits(&self) -> &[f32] {
        &self.output_logits
    }

    pub fn session(&self) -> &BackendSession {
        &self.session
    }

    pub fn into_parts(self) -> (u64, BackendSession, Vec<f32>) {
        (self.sequence_id, self.session, self.output_logits)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BackendDecodeBatchExecution {
    pub fused: bool,
}

#[derive(Debug)]
pub enum CudaLayerKvStore {
    F32(CudaLayerKvCache),
    SharedF32(CudaSharedF32LayerKvCache),
    Q8(CudaQ8LayerKvCache),
    KeyQ4ValueQ8(CudaKeyQ4ValueQ8LayerKvCache),
    AgentAdaptive {
        hot: CudaLayerKvCache,
        cold: CudaKeyQ4ValueQ8LayerKvCache,
        routes: CudaAdaptiveKvRoutes,
        hot_mask: Vec<u8>,
    },
}

impl CudaLayerKvStore {
    fn allocate(
        device: &CudaDevice,
        mode: KvCacheMode,
        capacity: usize,
        width: usize,
        page_tokens: usize,
    ) -> Result<Self> {
        match mode {
            KvCacheMode::Q8 => device
                .alloc_paged_q8_layer_kv_cache(capacity, width, page_tokens)
                .map(Self::Q8),
            KvCacheMode::KeyQ4ValueQ8 => device
                .alloc_paged_key_q4_value_q8_layer_kv_cache(capacity, width, page_tokens)
                .map(Self::KeyQ4ValueQ8),
            KvCacheMode::AgentAdaptive => Ok(Self::AgentAdaptive {
                hot: device.alloc_paged_layer_kv_cache(capacity, width, page_tokens)?,
                cold: device.alloc_paged_key_q4_value_q8_layer_kv_cache(
                    capacity,
                    width,
                    page_tokens,
                )?,
                routes: device.alloc_adaptive_kv_routes(capacity)?,
                hot_mask: Vec::with_capacity(capacity),
            }),
            _ => device
                .alloc_paged_layer_kv_cache(capacity, width, page_tokens)
                .map(Self::F32),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::F32(cache) => cache.len(),
            Self::SharedF32(cache) => cache.len(),
            Self::Q8(cache) => cache.len(),
            Self::KeyQ4ValueQ8(cache) => cache.len(),
            Self::AgentAdaptive { hot_mask, .. } => hot_mask.len(),
        }
    }

    fn mode(&self) -> KvCacheMode {
        match self {
            Self::F32(_) | Self::SharedF32(_) => KvCacheMode::F32,
            Self::Q8(_) => KvCacheMode::Q8,
            Self::KeyQ4ValueQ8(_) => KvCacheMode::KeyQ4ValueQ8,
            Self::AgentAdaptive { .. } => KvCacheMode::AgentAdaptive,
        }
    }

    fn capacity(&self) -> usize {
        match self {
            Self::F32(cache) => cache.capacity(),
            Self::SharedF32(cache) => cache.capacity(),
            Self::Q8(cache) => cache.capacity(),
            Self::KeyQ4ValueQ8(cache) => cache.capacity(),
            Self::AgentAdaptive {
                hot, cold, routes, ..
            } => hot.capacity().min(cold.capacity()).min(routes.capacity()),
        }
    }

    fn grow(&mut self, device: &CudaDevice, new_capacity: usize) -> Result<()> {
        match self {
            Self::F32(cache) => device.grow_layer_kv_cache(cache, new_capacity),
            Self::SharedF32(cache) if new_capacity <= cache.capacity() => Ok(()),
            Self::SharedF32(cache) => Err(XrtError::Runtime(format!(
                "CUDA shared F32 KV cache capacity {} cannot grow to {new_capacity}; its stable page table was allocated for the session context",
                cache.capacity()
            ))),
            Self::Q8(cache) => device.grow_q8_layer_kv_cache(cache, new_capacity),
            Self::KeyQ4ValueQ8(cache) => {
                device.grow_key_q4_value_q8_layer_kv_cache(cache, new_capacity)
            }
            Self::AgentAdaptive {
                hot, cold, routes, ..
            } => {
                device.grow_key_q4_value_q8_layer_kv_cache(cold, new_capacity)?;
                device.grow_layer_kv_cache(hot, new_capacity)?;
                device.grow_adaptive_kv_routes(routes, new_capacity)
            }
        }
    }

    fn deep_clone(&self, device: &CudaDevice) -> Result<Self> {
        self.deep_clone_with_capacity(device, self.capacity())
    }

    fn deep_clone_with_capacity(&self, device: &CudaDevice, capacity: usize) -> Result<Self> {
        match self {
            Self::F32(cache) => device
                .clone_layer_kv_cache_with_capacity(cache, capacity)
                .map(Self::F32),
            Self::SharedF32(cache) if capacity <= cache.capacity() => cache
                .snapshot_prefix(cache.len())
                .map(Self::SharedF32),
            Self::SharedF32(cache) => Err(XrtError::Runtime(format!(
                "CUDA shared F32 KV snapshot capacity {} cannot satisfy requested capacity {capacity}",
                cache.capacity()
            ))),
            Self::Q8(cache) => device
                .clone_q8_layer_kv_cache_with_capacity(cache, capacity)
                .map(Self::Q8),
            Self::KeyQ4ValueQ8(cache) => device
                .clone_key_q4_value_q8_layer_kv_cache_with_capacity(cache, capacity)
                .map(Self::KeyQ4ValueQ8),
            Self::AgentAdaptive {
                hot,
                cold,
                routes,
                hot_mask,
            } => Ok(Self::AgentAdaptive {
                hot: device.clone_layer_kv_cache_with_capacity(hot, capacity)?,
                cold: device.clone_key_q4_value_q8_layer_kv_cache_with_capacity(cold, capacity)?,
                routes: device.clone_adaptive_kv_routes_with_capacity(routes, capacity)?,
                hot_mask: hot_mask.clone(),
            }),
        }
    }

    fn clear(&mut self) {
        match self {
            Self::F32(cache) => cache.clear(),
            Self::SharedF32(cache) => {
                if let Err(err) = cache.clear() {
                    tracing::warn!("failed to clear CUDA shared F32 KV cache: {err}");
                }
            }
            Self::Q8(cache) => cache.clear(),
            Self::KeyQ4ValueQ8(cache) => cache.clear(),
            Self::AgentAdaptive {
                hot,
                cold,
                routes,
                hot_mask,
            } => {
                hot.clear();
                cold.clear();
                routes.clear();
                hot_mask.clear();
            }
        }
    }

    fn truncate(&mut self, new_len: usize) -> Result<()> {
        match self {
            Self::F32(cache) => cache.truncate(new_len),
            Self::SharedF32(cache) => cache.truncate(new_len)?,
            Self::Q8(cache) => cache.truncate(new_len),
            Self::KeyQ4ValueQ8(cache) => cache.truncate(new_len),
            Self::AgentAdaptive {
                hot,
                cold,
                routes,
                hot_mask,
            } => {
                let retained = new_len.min(hot_mask.len());
                let hot_len = hot_mask[..retained]
                    .iter()
                    .filter(|&&is_hot| is_hot != 0)
                    .count();
                hot.truncate(hot_len);
                cold.truncate(retained - hot_len);
                routes.truncate(retained);
                hot_mask.truncate(retained);
            }
        }
        Ok(())
    }

    fn allocated_bytes(&self) -> u64 {
        match self {
            Self::F32(cache) => cache.allocated_bytes(),
            Self::SharedF32(cache) => cache
                .page_table_bytes()
                .saturating_add(cache.referenced_page_bytes()),
            Self::Q8(cache) => cache.allocated_bytes(),
            Self::KeyQ4ValueQ8(cache) => cache.allocated_bytes(),
            Self::AgentAdaptive {
                hot, cold, routes, ..
            } => hot
                .allocated_bytes()
                .saturating_add(cold.allocated_bytes())
                .saturating_add(routes.allocated_bytes()),
        }
    }

    fn uses_shared_pages(&self) -> bool {
        matches!(self, Self::SharedF32(_))
    }

    fn snapshot_f32_prefix_into_pool(
        &self,
        device: &CudaDevice,
        pool: &CudaF32KvPagePool,
        max_tokens: usize,
        prefix_len: usize,
    ) -> Result<Self> {
        match self {
            Self::F32(source) => {
                if source.len() != prefix_len {
                    return Err(XrtError::Runtime(format!(
                        "cannot copy {prefix_len} shared F32 prefix tokens from contiguous CUDA cache length {}",
                        source.len()
                    )));
                }
                let mut snapshot = pool.allocate_cache(max_tokens)?;
                for position in 0..prefix_len {
                    let (key, value) = device.copy_layer_kv(source, position)?;
                    snapshot.append(&key, &value)?;
                }
                Ok(Self::SharedF32(snapshot))
            }
            Self::SharedF32(source) => source.snapshot_prefix(prefix_len).map(Self::SharedF32),
            other => Err(XrtError::Runtime(format!(
                "cannot create a shared F32 prefix snapshot from {} CUDA KV storage",
                other.mode().as_str()
            ))),
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
            routes,
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
        let page_tokens = hot.page_tokens();
        let current_hot_mask = hot_mask.clone();
        let mut rebuilt_hot = device.alloc_paged_layer_kv_cache(capacity, width, page_tokens)?;
        let mut rebuilt_cold =
            device.alloc_paged_key_q4_value_q8_layer_kv_cache(capacity, width, page_tokens)?;
        let mut source_hot_position = 0usize;
        let mut source_cold_position = 0usize;

        for (position, &was_hot) in current_hot_mask.iter().enumerate() {
            let should_be_hot = desired_hot_mask[position] != 0;
            match (was_hot != 0, should_be_hot) {
                (true, true) => {
                    let (key, value) = device.copy_layer_kv(hot, source_hot_position)?;
                    source_hot_position += 1;
                    device.append_layer_kv(&mut rebuilt_hot, &key, &value)?;
                }
                (true, false) => {
                    let (key, value) = device.copy_layer_kv(hot, source_hot_position)?;
                    source_hot_position += 1;
                    device.append_key_q4_value_q8_layer_kv(&mut rebuilt_cold, &key, &value)?;
                }
                (false, true) => {
                    let (key, value) =
                        device.dequantize_key_q4_value_q8_layer_kv(cold, source_cold_position)?;
                    source_cold_position += 1;
                    device.append_layer_kv(&mut rebuilt_hot, &key, &value)?;
                }
                (false, false) => {
                    device.copy_key_q4_value_q8_layer_kv_row(
                        cold,
                        source_cold_position,
                        &mut rebuilt_cold,
                    )?;
                    source_cold_position += 1;
                }
            }
        }

        *hot = rebuilt_hot;
        *cold = rebuilt_cold;
        let rebuilt_hot_mask = &desired_hot_mask[..current_hot_mask.len()];
        device.replace_adaptive_kv_routes(routes, rebuilt_hot_mask)?;
        *hot_mask = rebuilt_hot_mask.to_vec();
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CudaGraphCaptureState {
    Disabled,
    NotCaptured,
    Captured,
    EagerFallback,
}

impl CudaGraphCaptureState {
    fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::NotCaptured => "not-captured",
            Self::Captured => "captured",
            Self::EagerFallback => "eager-fallback",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CudaDecodeGraphKey {
    architecture: String,
    weight_kinds: Vec<&'static str>,
    cache_mode: KvCacheMode,
    kv_capacity: usize,
    layer_count: usize,
    embedding_length: usize,
    kv_width: usize,
    feed_forward_length: usize,
    vocab_size: usize,
    attention_head_count: usize,
    attention_head_count_kv: usize,
    head_dim: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CudaDecodeBatchGraphKey {
    sessions: Vec<(u64, u64, CudaDecodeGraphKey)>,
}

#[derive(Debug)]
enum CudaDecodeBatchGraphEntryState {
    Captured(CudaGraphExec),
    EagerFallback,
}

#[derive(Debug)]
struct CudaDecodeBatchGraphEntry {
    key: CudaDecodeBatchGraphKey,
    state: CudaDecodeBatchGraphEntryState,
}

#[derive(Debug, Default)]
struct CudaDecodeBatchGraphCache {
    entries: VecDeque<CudaDecodeBatchGraphEntry>,
}

impl CudaDecodeBatchGraphCache {
    fn entry_mut(
        &mut self,
        key: &CudaDecodeBatchGraphKey,
    ) -> Option<&mut CudaDecodeBatchGraphEntryState> {
        self.entries
            .iter_mut()
            .find(|entry| &entry.key == key)
            .map(|entry| &mut entry.state)
    }

    fn insert(&mut self, key: CudaDecodeBatchGraphKey, state: CudaDecodeBatchGraphEntryState) {
        if let Some(index) = self.entries.iter().position(|entry| entry.key == key) {
            self.entries.remove(index);
        }
        while self.entries.len() >= CUDA_DECODE_BATCH_GRAPH_CACHE_ENTRIES {
            self.entries.pop_front();
        }
        self.entries
            .push_back(CudaDecodeBatchGraphEntry { key, state });
    }
}

#[derive(Debug)]
struct CudaDecodeGraphState {
    executable: Option<CudaGraphExec>,
    key: Option<CudaDecodeGraphKey>,
    mode: CudaGraphMode,
    capture_state: CudaGraphCaptureState,
    last_error: Option<String>,
}

impl CudaDecodeGraphState {
    fn new(mode: CudaGraphMode) -> Self {
        Self {
            executable: None,
            key: None,
            mode,
            capture_state: if mode == CudaGraphMode::Disabled {
                CudaGraphCaptureState::Disabled
            } else {
                CudaGraphCaptureState::NotCaptured
            },
            last_error: None,
        }
    }

    fn reset(&mut self) {
        self.executable = None;
        self.key = None;
        self.last_error = None;
        self.capture_state = if self.mode == CudaGraphMode::Disabled {
            CudaGraphCaptureState::Disabled
        } else {
            CudaGraphCaptureState::NotCaptured
        };
    }

    fn fallback(&mut self, error: impl Into<String>) {
        if self.mode == CudaGraphMode::Disabled {
            return;
        }
        self.executable = None;
        self.key = None;
        self.last_error = Some(error.into());
        self.capture_state = CudaGraphCaptureState::EagerFallback;
    }

    fn captured(&mut self, key: CudaDecodeGraphKey, executable: CudaGraphExec) {
        self.executable = Some(executable);
        self.key = Some(key);
        self.last_error = None;
        self.capture_state = CudaGraphCaptureState::Captured;
    }

    fn is_enabled(&self) -> bool {
        self.mode != CudaGraphMode::Disabled
            && self.capture_state != CudaGraphCaptureState::EagerFallback
    }
}

#[derive(Debug)]
struct CudaDecodeScratch {
    decode_capacity: usize,
    embedding_length: usize,
    q_width: usize,
    kv_width: usize,
    feed_forward_length: usize,
    vocab_size: usize,
    decode_params: CudaDecodeParams,
    layer_input_a: CudaF32Buffer,
    layer_input_b: CudaF32Buffer,
    attention: CudaF32Buffer,
    normed_post_attention: CudaF32Buffer,
    q: CudaF32Buffer,
    q_temp: CudaF32Buffer,
    k: CudaF32Buffer,
    v: CudaF32Buffer,
    hidden_temp: CudaF32Buffer,
    kv_temp: CudaF32Buffer,
    gate: CudaF32Buffer,
    up: CudaF32Buffer,
    logits: CudaF32Buffer,
}

impl CudaDecodeScratch {
    fn allocate(
        device: &CudaDevice,
        embedding_length: usize,
        q_width: usize,
        kv_width: usize,
        feed_forward_length: usize,
        vocab_size: usize,
        decode_capacity: usize,
    ) -> Result<Self> {
        Ok(Self {
            decode_capacity,
            embedding_length,
            q_width,
            kv_width,
            feed_forward_length,
            vocab_size,
            decode_params: device.alloc_decode_params(decode_capacity, vocab_size)?,
            layer_input_a: device.zeros_f32(embedding_length)?,
            layer_input_b: device.zeros_f32(embedding_length)?,
            attention: device.zeros_f32(q_width)?,
            normed_post_attention: device.zeros_f32(embedding_length)?,
            q: device.zeros_f32(q_width)?,
            q_temp: device.zeros_f32(q_width)?,
            k: device.zeros_f32(kv_width)?,
            v: device.zeros_f32(kv_width)?,
            hidden_temp: device.zeros_f32(embedding_length)?,
            kv_temp: device.zeros_f32(kv_width)?,
            gate: device.zeros_f32(feed_forward_length)?,
            up: device.zeros_f32(feed_forward_length)?,
            logits: device.zeros_f32(vocab_size)?,
        })
    }

    fn matches_geometry(
        &self,
        embedding_length: usize,
        q_width: usize,
        kv_width: usize,
        feed_forward_length: usize,
        vocab_size: usize,
    ) -> bool {
        self.embedding_length == embedding_length
            && self.q_width == q_width
            && self.kv_width == kv_width
            && self.feed_forward_length == feed_forward_length
            && self.vocab_size == vocab_size
    }

    fn allocated_bytes(&self) -> u64 {
        [
            &self.layer_input_a,
            &self.layer_input_b,
            &self.attention,
            &self.normed_post_attention,
            &self.q,
            &self.q_temp,
            &self.k,
            &self.v,
            &self.hidden_temp,
            &self.kv_temp,
            &self.gate,
            &self.up,
            &self.logits,
        ]
        .into_iter()
        .map(|buffer| buffer.byte_len() as u64)
        .sum::<u64>()
        .saturating_add(self.decode_params.byte_len() as u64)
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
        decode_graph: CudaDecodeGraphState,
        batch_graph_epoch: u64,
        batch_graph_captured: bool,
        layer_caches: Vec<CudaLayerKvStore>,
        pending_prefix: Option<Arc<Vec<CudaLayerKvStore>>>,
        decode_scratch: Option<CudaDecodeScratch>,
        layer_count: usize,
        width: usize,
        layer_widths: Vec<usize>,
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

    fn snapshot_shared_f32_prefix(
        device: &CudaDevice,
        layer_caches: &[CudaLayerKvStore],
        layer_widths: &[usize],
        prefix_len: usize,
        max_len: usize,
        page_tokens: usize,
        kv_budget_bytes: Option<u64>,
    ) -> Result<Vec<CudaLayerKvStore>> {
        if layer_caches.len() != layer_widths.len() {
            return Err(XrtError::Runtime(format!(
                "cannot build shared F32 prefix from {} caches for {} layer widths",
                layer_caches.len(),
                layer_widths.len()
            )));
        }

        let page_tokens = page_tokens.max(1);
        let page_capacity = max_len.div_ceil(page_tokens);
        let full_session_bytes = cuda_session_kv_allocated_bytes_for_widths(
            KvCacheMode::F32,
            layer_widths,
            max_len,
            page_tokens,
        )?;
        let replica_limit = kv_budget_bytes
            .filter(|_| full_session_bytes != 0)
            .map(|budget| budget / full_session_bytes)
            .unwrap_or(2)
            .clamp(2, CUDA_SHARED_KV_MAX_REPLICAS as u64) as usize;

        let mut layers_per_width = HashMap::<usize, usize>::new();
        for &width in layer_widths {
            let count = layers_per_width.entry(width).or_default();
            *count = count.checked_add(1).ok_or_else(|| {
                XrtError::Runtime("CUDA shared F32 layer-count overflow".to_string())
            })?;
        }

        let mut pools = HashMap::<usize, CudaF32KvPagePool>::new();
        for (&width, &layer_count) in &layers_per_width {
            let base_pages = checked_mul(
                page_capacity,
                layer_count,
                "CUDA shared F32 base page count",
            )?;
            let max_pages = checked_mul(
                base_pages,
                replica_limit,
                "CUDA shared F32 bounded page count",
            )?;
            pools.insert(
                width,
                CudaF32KvPagePool::new(device, page_tokens, width, max_pages)?,
            );
        }

        layer_caches
            .iter()
            .zip(layer_widths)
            .map(|(cache, width)| {
                let pool = pools.get(width).ok_or_else(|| {
                    XrtError::Runtime(format!(
                        "missing CUDA shared F32 page pool for width {width}"
                    ))
                })?;
                cache.snapshot_f32_prefix_into_pool(device, pool, max_len, prefix_len)
            })
            .collect()
    }

    fn projected_shared_f32_bytes(
        layer_widths: &[usize],
        total_len: usize,
        max_len: usize,
        page_tokens: usize,
    ) -> Result<u64> {
        let page_tokens = page_tokens.max(1);
        let resident_pages = total_len.div_ceil(page_tokens);
        let page_capacity = max_len.div_ceil(page_tokens);
        layer_widths.iter().try_fold(0u64, |total, &width| {
            let page_elements = checked_mul(
                page_tokens,
                width,
                "CUDA shared F32 projected page elements",
            )?;
            let page_bytes = checked_mul(
                checked_mul(
                    page_elements,
                    std::mem::size_of::<f32>(),
                    "CUDA shared F32 projected component bytes",
                )?,
                2,
                "CUDA shared F32 projected page bytes",
            )?;
            let referenced_bytes = checked_mul(
                resident_pages,
                page_bytes,
                "CUDA shared F32 projected referenced bytes",
            )?;
            let pointer_entries = checked_mul(
                page_capacity,
                2,
                "CUDA shared F32 projected pointer entries",
            )?;
            let pointer_bytes = checked_mul(
                pointer_entries,
                std::mem::size_of::<u64>(),
                "CUDA shared F32 projected pointer bytes",
            )?;
            total
                .checked_add(referenced_bytes as u64)
                .and_then(|bytes| bytes.checked_add(pointer_bytes as u64))
                .ok_or_else(|| {
                    XrtError::Runtime("CUDA shared F32 projected byte count overflow".to_string())
                })
        })
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
        Self::new_cuda_with_kv_budget_page_tokens_and_layer_widths(
            device,
            cache_mode,
            vec![width; layer_count],
            max_len,
            page_tokens,
            kv_budget_bytes,
        )
    }

    fn new_cuda_with_kv_budget_page_tokens_and_layer_widths(
        device: CudaDevice,
        cache_mode: KvCacheMode,
        layer_widths: Vec<usize>,
        max_len: usize,
        page_tokens: usize,
        kv_budget_bytes: Option<u64>,
    ) -> Self {
        let layer_count = layer_widths.len();
        let width = layer_widths.iter().copied().max().unwrap_or(0);
        Self::Cuda {
            device,
            requested_cache_mode: cache_mode,
            cache_mode: Self::cuda_cache_mode(cache_mode),
            decode_graph: CudaDecodeGraphState::new(CudaGraphMode::Disabled),
            batch_graph_epoch: 0,
            batch_graph_captured: false,
            layer_caches: Vec::new(),
            pending_prefix: None,
            decode_scratch: None,
            layer_count,
            width,
            layer_widths,
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

    pub(crate) fn supports_prefix_cache(&self) -> bool {
        true
    }

    pub(crate) fn snapshot_prefix(
        &mut self,
        prefix_len: usize,
    ) -> Result<Option<BackendPrefixSnapshot>> {
        match self {
            Self::Cpu { cache } => {
                let cache = cache.snapshot_prefix(prefix_len)?;
                let allocated_bytes = cache.allocated_bytes();
                Ok(Some(BackendPrefixSnapshot::Cpu {
                    cache,
                    prefix_len,
                    allocated_bytes,
                }))
            }
            Self::Cuda {
                device,
                cache_mode,
                decode_graph,
                batch_graph_epoch,
                batch_graph_captured,
                layer_caches,
                pending_prefix,
                layer_widths,
                max_len,
                page_tokens,
                kv_budget_bytes,
                ..
            } => {
                if pending_prefix.is_some() {
                    return Err(XrtError::Runtime(
                        "cannot snapshot a CUDA prefix while another prefix is pending".to_string(),
                    ));
                }
                if layer_caches.len() != layer_widths.len()
                    || layer_caches.iter().any(|cache| cache.len() != prefix_len)
                {
                    return Err(XrtError::Runtime(format!(
                        "cannot snapshot {prefix_len} CUDA prefix tokens from {} initialized layers",
                        layer_caches.len()
                    )));
                }
                if *cache_mode == KvCacheMode::F32 {
                    let snapshot_caches = Arc::new(Self::snapshot_shared_f32_prefix(
                        device,
                        layer_caches,
                        layer_widths,
                        prefix_len,
                        *max_len,
                        *page_tokens,
                        *kv_budget_bytes,
                    )?);
                    let allocated_bytes = snapshot_caches
                        .iter()
                        .map(CudaLayerKvStore::allocated_bytes)
                        .sum();
                    return Ok(Some(BackendPrefixSnapshot::Cuda {
                        layer_caches: snapshot_caches,
                        cache_mode: *cache_mode,
                        layer_widths: layer_widths.clone(),
                        page_tokens: *page_tokens,
                        prefix_len,
                        allocated_bytes,
                    }));
                }
                let allocated_bytes = layer_caches
                    .iter()
                    .map(CudaLayerKvStore::allocated_bytes)
                    .sum();
                let layer_caches = Arc::new(std::mem::take(layer_caches));
                *pending_prefix = Some(layer_caches.clone());
                decode_graph.reset();
                *batch_graph_epoch = (*batch_graph_epoch).wrapping_add(1);
                *batch_graph_captured = false;
                Ok(Some(BackendPrefixSnapshot::Cuda {
                    layer_caches,
                    cache_mode: *cache_mode,
                    layer_widths: layer_widths.clone(),
                    page_tokens: *page_tokens,
                    prefix_len,
                    allocated_bytes,
                }))
            }
        }
    }

    pub(crate) fn attach_prefix_snapshot(
        &mut self,
        snapshot: &BackendPrefixSnapshot,
    ) -> Result<usize> {
        match (self, snapshot) {
            (
                Self::Cpu { cache },
                BackendPrefixSnapshot::Cpu {
                    cache: snapshot_cache,
                    prefix_len,
                    ..
                },
            ) if cache.geometry_matches(snapshot_cache) => {
                *cache = snapshot_cache.clone();
                Ok(*prefix_len)
            }
            (Self::Cpu { .. }, BackendPrefixSnapshot::Cpu { .. }) => Err(XrtError::Runtime(
                "prefix-cache CPU snapshot geometry does not match the target session".to_string(),
            )),
            (Self::Cuda { .. }, BackendPrefixSnapshot::Cpu { .. }) => Err(XrtError::Runtime(
                "cannot attach a CPU prefix-cache snapshot to a CUDA session".to_string(),
            )),
            (Self::Cpu { .. }, BackendPrefixSnapshot::Cuda { .. }) => Err(XrtError::Runtime(
                "cannot attach a CUDA prefix-cache snapshot to a CPU session".to_string(),
            )),
            (
                Self::Cuda {
                    cache_mode,
                    decode_graph,
                    batch_graph_epoch,
                    batch_graph_captured,
                    layer_caches,
                    pending_prefix,
                    layer_widths,
                    max_len,
                    page_tokens,
                    ..
                },
                BackendPrefixSnapshot::Cuda {
                    layer_caches: snapshot_caches,
                    cache_mode: snapshot_mode,
                    layer_widths: snapshot_widths,
                    page_tokens: snapshot_page_tokens,
                    prefix_len,
                    ..
                },
            ) if cache_mode == snapshot_mode
                && layer_widths == snapshot_widths
                && page_tokens == snapshot_page_tokens
                && *prefix_len <= *max_len =>
            {
                layer_caches.clear();
                *pending_prefix = Some(snapshot_caches.clone());
                decode_graph.reset();
                *batch_graph_epoch = (*batch_graph_epoch).wrapping_add(1);
                *batch_graph_captured = false;
                Ok(*prefix_len)
            }
            (Self::Cuda { .. }, BackendPrefixSnapshot::Cuda { .. }) => Err(XrtError::Runtime(
                "prefix-cache CUDA snapshot geometry does not match the target session".to_string(),
            )),
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

    fn configure_cuda_graph_mode(&mut self, mode: CudaGraphMode) {
        if let Self::Cuda { decode_graph, .. } = self {
            *decode_graph = CudaDecodeGraphState::new(mode);
        }
    }

    pub fn cuda_graph_capture_status(&self) -> Option<&'static str> {
        match self {
            Self::Cpu { .. } => None,
            Self::Cuda {
                decode_graph,
                batch_graph_captured,
                ..
            } => Some(if *batch_graph_captured {
                "batch-captured"
            } else {
                decode_graph.capture_state.as_str()
            }),
        }
    }

    fn cuda_batch_graph_epoch(&self) -> Option<u64> {
        match self {
            Self::Cpu { .. } => None,
            Self::Cuda {
                batch_graph_epoch, ..
            } => Some(*batch_graph_epoch),
        }
    }

    fn mark_cuda_batch_graph_captured(&mut self) {
        if let Self::Cuda {
            batch_graph_captured,
            ..
        } = self
        {
            *batch_graph_captured = true;
        }
    }

    pub fn cuda_graph_last_error(&self) -> Option<&str> {
        match self {
            Self::Cpu { .. } => None,
            Self::Cuda { decode_graph, .. } => decode_graph.last_error.as_deref(),
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
        self.replace_cache_with_layer_widths(cache_mode, vec![width; layer_count], page_tokens);
    }

    pub fn replace_cache_with_layer_widths(
        &mut self,
        cache_mode: KvCacheMode,
        next_layer_widths: Vec<usize>,
        page_tokens: usize,
    ) {
        let layer_count = next_layer_widths.len();
        let width = next_layer_widths.iter().copied().max().unwrap_or(0);
        match self {
            Self::Cpu { .. } => {
                self.replace_cpu_cache(cache_mode, layer_count, width, page_tokens);
            }
            Self::Cuda {
                requested_cache_mode,
                cache_mode: current,
                decode_graph,
                batch_graph_epoch,
                batch_graph_captured,
                layer_caches,
                pending_prefix,
                layer_count: current_layer_count,
                width: current_width,
                layer_widths,
                page_tokens: current_page_tokens,
                policy,
                prompt_token_count,
                prompt_spans,
                ..
            } => {
                let next_cache_mode = Self::cuda_cache_mode(cache_mode);
                let requested_changed = *requested_cache_mode != cache_mode;
                let layout_changed = layer_widths.as_slice() != next_layer_widths.as_slice()
                    || Self::cuda_cache_layout_changed(
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
                *layer_widths = next_layer_widths;
                *current_page_tokens = page_tokens.max(1);
                *policy = SessionPolicy::default();
                *prompt_token_count = 0;
                prompt_spans.clear();
                *pending_prefix = None;
                decode_graph.reset();
                if layout_changed {
                    *batch_graph_epoch = (*batch_graph_epoch).wrapping_add(1);
                    *batch_graph_captured = false;
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
                *cuda_policy = policy;
                *cuda_prompt_token_count = prompt_token_count;
                prompt_spans.clear();
                prompt_spans.extend_from_slice(spans);
            }
        }
    }

    pub fn kv_reservation_bytes_for_total_len(&self, total_len: usize) -> Result<u64> {
        match self {
            Self::Cpu { .. } => Ok(0),
            Self::Cuda {
                cache_mode,
                layer_widths,
                max_len,
                page_tokens,
                ..
            } => {
                if total_len > *max_len {
                    return Err(XrtError::Runtime(format!(
                        "CUDA KV request length {total_len} exceeds context length {max_len}"
                    )));
                }
                let target_capacity =
                    cuda_kv_growth_capacity(0, total_len, *page_tokens, *max_len)?;
                let final_bytes = cuda_session_kv_allocated_bytes_for_widths(
                    *cache_mode,
                    layer_widths,
                    target_capacity,
                    *page_tokens,
                )?;
                final_bytes.checked_mul(2).ok_or_else(|| {
                    XrtError::Runtime("CUDA KV reservation byte count overflow".to_string())
                })
            }
        }
    }

    pub fn prepare_for_total_len(&mut self, total_len: usize) -> Result<()> {
        match self {
            Self::Cpu { cache } => cache.prepare_for_total_len(total_len),
            Self::Cuda {
                device,
                cache_mode,
                decode_graph,
                batch_graph_epoch,
                batch_graph_captured,
                layer_caches,
                pending_prefix,
                layer_count,
                layer_widths,
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
                if let Some(snapshot_caches) = pending_prefix.take() {
                    let uses_shared_pages = snapshot_caches
                        .iter()
                        .any(CudaLayerKvStore::uses_shared_pages);
                    if uses_shared_pages
                        && snapshot_caches
                            .iter()
                            .any(|cache| !cache.uses_shared_pages())
                    {
                        *pending_prefix = Some(snapshot_caches);
                        return Err(XrtError::Runtime(
                            "CUDA prefix snapshot mixes shared and contiguous layer caches"
                                .to_string(),
                        ));
                    }
                    let source_capacity = snapshot_caches
                        .first()
                        .map(|cache| cache.capacity())
                        .unwrap_or(0);
                    let target_capacity = if total_len > source_capacity {
                        cuda_kv_growth_capacity(source_capacity, total_len, *page_tokens, *max_len)?
                    } else {
                        source_capacity
                    };
                    let required_bytes = if uses_shared_pages {
                        Self::projected_shared_f32_bytes(
                            layer_widths,
                            total_len,
                            *max_len,
                            *page_tokens,
                        )?
                    } else {
                        cuda_session_kv_allocated_bytes_for_widths(
                            *cache_mode,
                            layer_widths,
                            target_capacity,
                            *page_tokens,
                        )?
                    };
                    let snapshot_bytes = snapshot_caches
                        .iter()
                        .map(CudaLayerKvStore::allocated_bytes)
                        .try_fold(0u64, |total, bytes| {
                            total.checked_add(bytes).ok_or_else(|| {
                                XrtError::Runtime(
                                    "CUDA prefix snapshot byte count overflow".to_string(),
                                )
                            })
                        })?;
                    let peak_bytes =
                        snapshot_bytes.checked_add(required_bytes).ok_or_else(|| {
                            XrtError::Runtime(
                                "CUDA prefix attach peak byte count overflow".to_string(),
                            )
                        })?;
                    if let Some(budget_bytes) = kv_budget_bytes {
                        if peak_bytes > *budget_bytes {
                            *pending_prefix = Some(snapshot_caches);
                            return Err(XrtError::Cuda(format!(
                                "CUDA prefix attach requires {peak_bytes} peak KV bytes (snapshot {snapshot_bytes}, mutable copy {required_bytes}), but the configured KV budget is {budget_bytes} bytes"
                            )));
                        }
                    }

                    let materialized = snapshot_caches
                        .iter()
                        .map(|cache| cache.deep_clone_with_capacity(device, target_capacity))
                        .collect::<Result<Vec<_>>>();
                    match materialized {
                        Ok(caches) => {
                            let materialized_shared =
                                caches.iter().any(CudaLayerKvStore::uses_shared_pages);
                            *layer_caches = caches;
                            if materialized_shared {
                                decode_graph.fallback(
                                    "CUDA Graph decode for runtime-attached shared F32 KV pages is not wired yet",
                                );
                            } else {
                                decode_graph.reset();
                            }
                            *batch_graph_epoch = (*batch_graph_epoch).wrapping_add(1);
                            *batch_graph_captured = false;
                        }
                        Err(err) => {
                            *pending_prefix = Some(snapshot_caches);
                            return Err(err);
                        }
                    }
                }
                let uses_shared_pages =
                    layer_caches.iter().any(CudaLayerKvStore::uses_shared_pages);
                if uses_shared_pages {
                    if layer_caches.iter().any(|cache| !cache.uses_shared_pages()) {
                        return Err(XrtError::Runtime(
                            "CUDA session mixes shared and contiguous layer caches".to_string(),
                        ));
                    }
                    let required_bytes = Self::projected_shared_f32_bytes(
                        layer_widths,
                        total_len,
                        *max_len,
                        *page_tokens,
                    )?;
                    if let Some(budget_bytes) = kv_budget_bytes {
                        if required_bytes > *budget_bytes {
                            return Err(XrtError::Cuda(format!(
                                "CUDA shared F32 KV cache requires {required_bytes} bytes for {total_len} tokens, but the configured KV budget is {budget_bytes} bytes"
                            )));
                        }
                    }
                }
                let current_capacity = layer_caches
                    .first()
                    .map(CudaLayerKvStore::capacity)
                    .unwrap_or(0);
                if total_len > current_capacity {
                    decode_graph.reset();
                    let target_capacity = cuda_kv_growth_capacity(
                        current_capacity,
                        total_len,
                        *page_tokens,
                        *max_len,
                    )?;
                    let required_bytes = cuda_session_kv_allocated_bytes_for_widths(
                        *cache_mode,
                        layer_widths,
                        target_capacity,
                        *page_tokens,
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
                    *batch_graph_epoch = (*batch_graph_epoch).wrapping_add(1);
                    *batch_graph_captured = false;
                    if layer_caches.is_empty() {
                        let mut caches = Vec::with_capacity(*layer_count);
                        for &layer_width in layer_widths.iter() {
                            caches.push(CudaLayerKvStore::allocate(
                                device,
                                *cache_mode,
                                target_capacity,
                                layer_width,
                                *page_tokens,
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

    fn cuda_graph_decode_ready(&mut self) -> bool {
        match self {
            Self::Cpu { .. } => false,
            Self::Cuda {
                cache_mode,
                decode_graph,
                layer_caches,
                pending_prefix,
                ..
            } => {
                if !decode_graph.is_enabled() {
                    return false;
                }
                if *cache_mode != KvCacheMode::F32 {
                    decode_graph.fallback(format!(
                        "CUDA Graph decode currently requires f32 KV, found {}",
                        cache_mode.as_str()
                    ));
                    return false;
                }
                if layer_caches.iter().any(CudaLayerKvStore::uses_shared_pages)
                    || pending_prefix.as_ref().is_some_and(|caches| {
                        caches.iter().any(CudaLayerKvStore::uses_shared_pages)
                    })
                {
                    decode_graph.fallback(
                        "CUDA Graph decode for runtime-attached shared F32 KV pages is not wired yet",
                    );
                    return false;
                }
                true
            }
        }
    }

    pub(crate) fn prepare_cuda_graph_generation_capacity(&mut self, total_len: usize) -> bool {
        if !self.cuda_graph_decode_ready() {
            return false;
        }
        match self.prepare_for_total_len(total_len) {
            Ok(()) => true,
            Err(err) => {
                self.cuda_graph_fallback(err.to_string());
                tracing::warn!(
                    "bounded CUDA Graph KV preallocation failed; using eager CUDA: {err}"
                );
                false
            }
        }
    }

    fn cuda_kv_capacity(&self) -> Option<usize> {
        match self {
            Self::Cpu { .. } => None,
            Self::Cuda {
                layer_caches,
                pending_prefix,
                ..
            } => layer_caches
                .first()
                .or_else(|| pending_prefix.as_ref().and_then(|caches| caches.first()))
                .map(CudaLayerKvStore::capacity),
        }
    }

    fn cuda_graph_parts_mut(
        &mut self,
    ) -> Result<(
        &mut CudaDecodeGraphState,
        &mut [CudaLayerKvStore],
        &mut CudaDecodeScratch,
    )> {
        match self {
            Self::Cuda {
                decode_graph,
                layer_caches,
                decode_scratch,
                ..
            } => {
                let scratch = decode_scratch.as_mut().ok_or_else(|| {
                    XrtError::Runtime("CUDA decode scratch is not allocated".to_string())
                })?;
                Ok((decode_graph, layer_caches, scratch))
            }
            Self::Cpu { .. } => Err(XrtError::Runtime(
                "CUDA graph state requested from CPU backend session".to_string(),
            )),
        }
    }

    fn cuda_graph_executable(&self) -> Option<&CudaGraphExec> {
        match self {
            Self::Cpu { .. } => None,
            Self::Cuda { decode_graph, .. } => decode_graph.executable.as_ref(),
        }
    }

    fn cuda_graph_fallback(&mut self, error: impl Into<String>) {
        if let Self::Cuda { decode_graph, .. } = self {
            decode_graph.fallback(error);
        }
    }

    pub fn clear(&mut self) {
        match self {
            Self::Cpu { cache } => cache.clear(),
            Self::Cuda {
                layer_caches,
                pending_prefix,
                ..
            } => {
                *pending_prefix = None;
                for cache in layer_caches {
                    cache.clear();
                }
            }
        }
    }

    pub fn truncate(&mut self, new_len: usize) -> Result<()> {
        match self {
            Self::Cpu { cache } => {
                cache.truncate(new_len);
                Ok(())
            }
            Self::Cuda {
                layer_caches,
                pending_prefix,
                ..
            } => {
                if pending_prefix.is_some() {
                    return Err(XrtError::Runtime(
                        "cannot truncate a CUDA prefix before its copy-on-write materialization"
                            .to_string(),
                    ));
                }
                for cache in layer_caches {
                    cache.truncate(new_len)?;
                }
                Ok(())
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

    fn ensure_cuda_decode_scratch(
        &mut self,
        device: &CudaDevice,
        embedding_length: usize,
        q_width: usize,
        kv_width: usize,
        feed_forward_length: usize,
        vocab_size: usize,
        decode_capacity: usize,
    ) -> Result<()> {
        match self {
            Self::Cuda {
                decode_graph,
                batch_graph_epoch,
                batch_graph_captured,
                decode_scratch,
                ..
            } => {
                let needs_allocation = decode_scratch.as_ref().map_or(true, |scratch| {
                    !scratch.matches_geometry(
                        embedding_length,
                        q_width,
                        kv_width,
                        feed_forward_length,
                        vocab_size,
                    )
                });
                if needs_allocation {
                    let scratch = CudaDecodeScratch::allocate(
                        device,
                        embedding_length,
                        q_width,
                        kv_width,
                        feed_forward_length,
                        vocab_size,
                        decode_capacity,
                    )?;
                    decode_graph.reset();
                    *batch_graph_epoch = (*batch_graph_epoch).wrapping_add(1);
                    *batch_graph_captured = false;
                    *decode_scratch = Some(scratch);
                } else if decode_scratch
                    .as_ref()
                    .is_some_and(|scratch| scratch.decode_capacity != decode_capacity)
                {
                    let decode_params = device.alloc_decode_params(decode_capacity, vocab_size)?;
                    decode_graph.reset();
                    *batch_graph_epoch = (*batch_graph_epoch).wrapping_add(1);
                    *batch_graph_captured = false;
                    let scratch = decode_scratch.as_mut().ok_or_else(|| {
                        XrtError::Runtime(
                            "CUDA decode scratch disappeared during capacity update".to_string(),
                        )
                    })?;
                    scratch.decode_capacity = decode_capacity;
                    scratch.decode_params = decode_params;
                }
                Ok(())
            }
            Self::Cpu { .. } => Err(XrtError::Runtime(
                "CUDA scratch requested from CPU backend session".to_string(),
            )),
        }
    }

    fn cuda_layer_cache_and_scratch_mut(
        &mut self,
        layer: usize,
    ) -> Result<(&mut CudaLayerKvStore, &mut CudaDecodeScratch)> {
        match self {
            Self::Cuda {
                layer_caches,
                decode_scratch,
                ..
            } => {
                let cache = layer_caches.get_mut(layer).ok_or_else(|| {
                    XrtError::Runtime(format!("missing CUDA KV cache for layer {layer}"))
                })?;
                let scratch = decode_scratch.as_mut().ok_or_else(|| {
                    XrtError::Runtime("CUDA decode scratch is not allocated".to_string())
                })?;
                Ok((cache, scratch))
            }
            Self::Cpu { .. } => Err(XrtError::Runtime(
                "CUDA layer state requested from CPU backend session".to_string(),
            )),
        }
    }

    fn cuda_decode_scratch_mut(&mut self) -> Result<&mut CudaDecodeScratch> {
        match self {
            Self::Cuda { decode_scratch, .. } => decode_scratch.as_mut().ok_or_else(|| {
                XrtError::Runtime("CUDA decode scratch is not allocated".to_string())
            }),
            Self::Cpu { .. } => Err(XrtError::Runtime(
                "CUDA scratch requested from CPU backend session".to_string(),
            )),
        }
    }

    pub fn cuda_kv_allocated_bytes(&self) -> u64 {
        match self {
            Self::Cpu { .. } => 0,
            Self::Cuda {
                layer_caches,
                pending_prefix,
                ..
            } => layer_caches
                .iter()
                .chain(pending_prefix.iter().flat_map(|caches| caches.iter()))
                .map(CudaLayerKvStore::allocated_bytes)
                .sum(),
        }
    }

    pub fn cuda_scratch_allocated_bytes(&self) -> u64 {
        match self {
            Self::Cpu { .. } => 0,
            Self::Cuda { decode_scratch, .. } => decode_scratch
                .as_ref()
                .map_or(0, CudaDecodeScratch::allocated_bytes),
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

    fn supports_multi_sequence_decode_batch(&self) -> bool {
        false
    }

    fn forward_token_batch(
        &self,
        batch: &mut [BackendDecodeBatchItem],
    ) -> Result<BackendDecodeBatchExecution> {
        for item in batch {
            self.forward_token(
                item.token_id,
                item.position,
                &mut item.session,
                &mut item.output_logits,
            )?;
        }
        Ok(BackendDecodeBatchExecution { fused: false })
    }
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
    fn gemma4_layer0_trace(
        &self,
        _token_id: u32,
        _position: usize,
        _session: &mut BackendSession,
    ) -> Result<Option<Gemma4LayerTrace>> {
        Ok(None)
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

    fn cuda_memory_info(&self) -> Option<(u64, u64)> {
        self.cuda_free_vram_bytes()
            .zip(self.cuda_total_vram_bytes())
    }

    fn cuda_transfer_stats(&self) -> Option<CudaTransferStats> {
        None
    }

    fn cuda_allocation_stats(&self) -> Option<CudaAllocationStats> {
        None
    }

    fn cuda_memory_pool_stats(&self) -> Option<CudaMemoryPoolStats> {
        None
    }

    fn reset_cuda_allocation_peak(&self) {}

    fn cuda_kv_budget_bytes(&self) -> Option<u64> {
        None
    }

    fn supports_cuda_graph_decode(&self) -> bool {
        false
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

    fn gemma4_layer0_trace(
        &self,
        token_id: u32,
        position: usize,
        session: &mut BackendSession,
    ) -> Result<Option<Gemma4LayerTrace>> {
        let cache = session.cpu_cache_mut()?;
        self.model
            .gemma4_layer0_trace(token_id, position, cache)
            .map(Some)
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
    cpu_reference_model: Option<Arc<LlamaModel>>,
    model_name: String,
    config: LlamaConfig,
    device: CudaDevice,
    resident_model_weight_bytes: u64,
    device_name: Option<String>,
    kv_budget_bytes: u64,
    cuda_graph_mode: CudaGraphMode,
    decode_batch_graphs: Mutex<CudaDecodeBatchGraphCache>,
    decode_batch_streams: Mutex<Vec<CudaExecutionStream>>,
    f32_probe: Option<ResidentF32ProbeWeights>,
    q8_0_probe: Option<ResidentQ8_0ProbeWeights>,
    q8_0_layer_probes: Option<Vec<ResidentQ8_0LayerWeights>>,
    gemma4_layer_probes: Option<Vec<ResidentGemma4LayerWeights>>,
}

impl CudaResidentBackend {
    pub fn new(
        model: Arc<LlamaModel>,
        gguf: &GgufFile,
        gpu_config: GpuResourceConfig,
    ) -> Result<Self> {
        let model_name = model.model_name().to_string();
        let model_config = model.config().clone();
        let source = GgufResidentTensorSource::new(gguf);
        Self::new_with_source(Some(model), model_name, model_config, &source, gpu_config)
    }

    pub fn from_hf_bundle(bundle: &HfModelBundle, gpu_config: GpuResourceConfig) -> Result<Self> {
        let model_config = LlamaConfig::from_hf(bundle.config())?;
        let model_name = bundle
            .config()
            .model_name
            .as_deref()
            .filter(|name| !name.trim().is_empty())
            .map(ToOwned::to_owned)
            .or_else(|| {
                bundle
                    .root()
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
            })
            .unwrap_or_else(|| "safetensors-model".to_string());
        let source = HfStandardDenseResidentTensorSource::new(bundle)?;
        Self::new_with_source(None, model_name, model_config, &source, gpu_config)
    }

    fn new_with_source(
        cpu_reference_model: Option<Arc<LlamaModel>>,
        model_name: String,
        model_config: LlamaConfig,
        source: &impl ResidentTensorSource,
        gpu_config: GpuResourceConfig,
    ) -> Result<Self> {
        if !Self::supports_dense_quant_decode_source(source, &model_config) {
            return Err(Self::decode_unsupported());
        }
        let device = CudaDevice::new(gpu_config.device_ordinal)?;
        for tensor_name in [
            "token_embd.weight",
            ResidentQ8_0ProbeWeights::output_name(source),
        ] {
            if let Some(info) = source.tensor_info(tensor_name) {
                info!(
                    tensor = tensor_name,
                    dtype = ?info.dtype,
                    storage = ?info.storage,
                    rows = info.rows,
                    cols = info.cols,
                    source_bytes = info.byte_len,
                    "CUDA resident tensor plan"
                );
            }
        }
        let (free_vram_bytes, total_vram_bytes, resident_model_weight_bytes, kv_budget_bytes) =
            Self::preflight_model_upload(source, &model_config, &device, gpu_config)?;
        info!(
            resident_model_weight_bytes,
            free_vram_bytes,
            total_vram_bytes,
            kv_budget_bytes,
            "CUDA resident upload preflight passed"
        );
        let device_name = device.name().ok();
        info!("loading CUDA resident output weights");
        let f32_probe = ResidentF32ProbeWeights::try_load(&device, source, &model_config)?;
        let q8_0_probe = ResidentQ8_0ProbeWeights::try_load(&device, source, &model_config)?;
        info!("loading CUDA resident transformer layers");
        let q8_0_layer_probes =
            ResidentQ8_0LayerWeights::try_load_all(&device, source, &model_config)?;
        let gemma4_layer_probes =
            ResidentGemma4LayerWeights::try_load_all(&device, source, &model_config)?;
        info!("CUDA resident model upload complete");
        Ok(Self {
            cpu_reference_model,
            model_name,
            config: model_config,
            device,
            resident_model_weight_bytes,
            device_name,
            kv_budget_bytes,
            cuda_graph_mode: gpu_config.cuda_graph_mode,
            decode_batch_graphs: Mutex::new(CudaDecodeBatchGraphCache::default()),
            decode_batch_streams: Mutex::new(Vec::new()),
            f32_probe,
            q8_0_probe,
            q8_0_layer_probes,
            gemma4_layer_probes,
        })
    }

    pub fn supports_dense_quant_decode(gguf: &GgufFile, config: &LlamaConfig) -> bool {
        Self::supports_dense_quant_decode_source(&GgufResidentTensorSource::new(gguf), config)
    }

    fn supports_dense_quant_decode_source(
        source: &impl ResidentTensorSource,
        config: &LlamaConfig,
    ) -> bool {
        ResidentQ8_0ProbeWeights::supports(source, config)
            && if config.is_gemma4() {
                ResidentGemma4LayerWeights::supports_all(source, config)
            } else {
                ResidentQ8_0LayerWeights::supports_all(source, config)
            }
    }

    fn preflight_model_upload(
        source: &impl ResidentTensorSource,
        model_config: &LlamaConfig,
        device: &CudaDevice,
        config: GpuResourceConfig,
    ) -> Result<(u64, u64, u64, u64)> {
        let model_weight_bytes = cuda_estimated_resident_upload_bytes(source, model_config)?;
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
            "cuda-resident decode currently supports standard dense and Gemma4 GGUF F32/F16/BF16/Q8_0/Q4_0/Q4_K/Q5_K/Q6_K models plus dense Qwen2/Qwen3 SafeTensors, AutoAWQ GEMM/GEMV, GPTQ v1/v2 GEMM4, or compressed-tensors W4A16; broader model sources are not wired yet"
                .to_string(),
        )
    }

    fn require_cpu_reference_model(&self) -> Result<&LlamaModel> {
        self.cpu_reference_model.as_deref().ok_or_else(|| {
            XrtError::Unsupported(
                "CPU-reference model operations are unavailable for this CUDA model source"
                    .to_string(),
            )
        })
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
            self.config.rms_norm_eps,
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
            self.config.rms_norm_eps,
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
        let config = &self.config;
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
        let config = &self.config;
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
            CudaLayerKvStore::SharedF32(cache) => {
                cache.append(&k, &v)?;
                cache.single_query_attention_device(
                    &q,
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
                routes,
                hot_mask,
            } => {
                if adaptive_is_hot {
                    let local_position = hot.len();
                    self.device.append_layer_kv(hot, &k, &v)?;
                    self.device
                        .append_adaptive_kv_route(routes, true, local_position)?;
                    hot_mask.push(1);
                } else {
                    let local_position = cold.len();
                    self.device.append_key_q4_value_q8_layer_kv(cold, &k, &v)?;
                    self.device
                        .append_adaptive_kv_route(routes, false, local_position)?;
                    hot_mask.push(0);
                }
                self.device
                    .single_query_attention_mixed_key_q4_value_q8_device(
                        &q,
                        hot,
                        cold,
                        routes,
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

    fn run_q8_0_layer_device_with_scratch(
        &self,
        layer_index: usize,
        probe: &ResidentQ8_0LayerWeights,
        input: &CudaF32Buffer,
        position: usize,
        adaptive_is_hot: bool,
        kv_cache: &mut CudaLayerKvStore,
        scratch: &mut CudaDecodeScratch,
    ) -> Result<CudaF32Buffer> {
        let config = &self.config;
        let profile = Self::cuda_profile_enabled();
        let layer_start = Instant::now();
        let stage_start = Instant::now();
        self.device.rmsnorm_device_into(
            input,
            probe.attn_norm.buffer(),
            1,
            probe.embedding_length,
            config.rms_norm_eps,
            &mut scratch.normed_post_attention,
        )?;
        self.matvec_quant_resident_device_into(
            &probe.attn_q,
            &scratch.normed_post_attention,
            &mut scratch.q,
        )?;
        self.matvec_quant_resident_device_into(
            &probe.attn_k,
            &scratch.normed_post_attention,
            &mut scratch.k,
        )?;
        self.matvec_quant_resident_device_into(
            &probe.attn_v,
            &scratch.normed_post_attention,
            &mut scratch.v,
        )?;
        if let Some(bias) = &probe.attn_q_bias {
            self.device
                .add_assign_device(&mut scratch.q, bias.buffer())?;
        }
        if let Some(bias) = &probe.attn_k_bias {
            self.device
                .add_assign_device(&mut scratch.k, bias.buffer())?;
        }
        if let Some(bias) = &probe.attn_v_bias {
            self.device
                .add_assign_device(&mut scratch.v, bias.buffer())?;
        }
        if let Some(q_norm) = &probe.attn_q_norm {
            self.device.rmsnorm_device_into(
                &scratch.q,
                q_norm.buffer(),
                config.attention_head_count,
                config.head_dim(),
                config.rms_norm_eps,
                &mut scratch.q_temp,
            )?;
            std::mem::swap(&mut scratch.q, &mut scratch.q_temp);
        }
        if let Some(k_norm) = &probe.attn_k_norm {
            self.device.rmsnorm_device_into(
                &scratch.k,
                k_norm.buffer(),
                config.attention_head_count_kv,
                config.head_dim(),
                config.rms_norm_eps,
                &mut scratch.kv_temp,
            )?;
            std::mem::swap(&mut scratch.k, &mut scratch.kv_temp);
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
            &mut scratch.q,
            config.attention_head_count,
            config.head_dim(),
            position,
            config.rope_dimension_count,
            config.rope_freq_base,
            config.rope_freq_scale,
        )?;
        self.device.rope_device(
            &mut scratch.k,
            config.attention_head_count_kv,
            config.head_dim(),
            position,
            config.rope_dimension_count,
            config.rope_freq_base,
            config.rope_freq_scale,
        )?;
        let attention_values = match kv_cache {
            CudaLayerKvStore::F32(cache) => {
                self.device.append_layer_kv(cache, &scratch.k, &scratch.v)?;
                self.device.single_query_attention_device(
                    &scratch.q,
                    cache,
                    config.attention_head_count,
                    config.attention_head_count_kv,
                    config.head_dim(),
                )?
            }
            CudaLayerKvStore::SharedF32(cache) => {
                cache.append(&scratch.k, &scratch.v)?;
                cache.single_query_attention_device(
                    &scratch.q,
                    config.attention_head_count,
                    config.attention_head_count_kv,
                    config.head_dim(),
                )?
            }
            CudaLayerKvStore::Q8(cache) => {
                self.device
                    .append_q8_layer_kv(cache, &scratch.k, &scratch.v)?;
                self.device.single_query_attention_q8_device(
                    &scratch.q,
                    cache,
                    config.attention_head_count,
                    config.attention_head_count_kv,
                    config.head_dim(),
                )?
            }
            CudaLayerKvStore::KeyQ4ValueQ8(cache) => {
                self.device
                    .append_key_q4_value_q8_layer_kv(cache, &scratch.k, &scratch.v)?;
                self.device.single_query_attention_key_q4_value_q8_device(
                    &scratch.q,
                    cache,
                    config.attention_head_count,
                    config.attention_head_count_kv,
                    config.head_dim(),
                )?
            }
            CudaLayerKvStore::AgentAdaptive {
                hot,
                cold,
                routes,
                hot_mask,
            } => {
                if adaptive_is_hot {
                    let local_position = hot.len();
                    self.device.append_layer_kv(hot, &scratch.k, &scratch.v)?;
                    self.device
                        .append_adaptive_kv_route(routes, true, local_position)?;
                    hot_mask.push(1);
                } else {
                    let local_position = cold.len();
                    self.device
                        .append_key_q4_value_q8_layer_kv(cold, &scratch.k, &scratch.v)?;
                    self.device
                        .append_adaptive_kv_route(routes, false, local_position)?;
                    hot_mask.push(0);
                }
                self.device
                    .single_query_attention_mixed_key_q4_value_q8_device(
                        &scratch.q,
                        hot,
                        cold,
                        routes,
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
        self.matvec_quant_resident_device_into(
            &probe.attn_output,
            &attention_values,
            &mut scratch.hidden_temp,
        )?;
        self.device
            .copy_f32_device(input, &mut scratch.normed_post_attention)?;
        self.device
            .add_assign_device(&mut scratch.normed_post_attention, &scratch.hidden_temp)?;
        if profile {
            info!(
                layer_index,
                ms = stage_start.elapsed().as_secs_f64() * 1000.0,
                "cuda profile: attention output"
            );
        }

        let stage_start = Instant::now();
        self.device.rmsnorm_device_into(
            &scratch.normed_post_attention,
            probe.ffn_norm.buffer(),
            1,
            probe.embedding_length,
            config.rms_norm_eps,
            &mut scratch.hidden_temp,
        )?;
        self.matvec_quant_resident_device_into(
            &probe.ffn_gate,
            &scratch.hidden_temp,
            &mut scratch.gate,
        )?;
        self.matvec_quant_resident_device_into(
            &probe.ffn_up,
            &scratch.hidden_temp,
            &mut scratch.up,
        )?;
        self.device.silu_assign_device(&mut scratch.gate)?;
        self.device
            .mul_assign_device(&mut scratch.gate, &scratch.up)?;
        let mut post_ffn = self.matvec_quant_resident_device(&probe.ffn_down, &scratch.gate)?;
        self.device
            .add_assign_device(&mut post_ffn, &scratch.normed_post_attention)?;
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

        Ok(post_ffn)
    }

    fn run_gemma4_layer_device(
        &self,
        layer_index: usize,
        weights: &ResidentGemma4LayerWeights,
        input: &CudaF32Buffer,
        position: usize,
        adaptive_is_hot: bool,
        kv_cache: &mut CudaLayerKvStore,
    ) -> Result<CudaF32Buffer> {
        self.run_gemma4_layer_device_with_trace(
            layer_index,
            weights,
            input,
            position,
            adaptive_is_hot,
            kv_cache,
            None,
        )
    }

    fn run_gemma4_layer_device_with_trace(
        &self,
        layer_index: usize,
        weights: &ResidentGemma4LayerWeights,
        input: &CudaF32Buffer,
        position: usize,
        adaptive_is_hot: bool,
        kv_cache: &mut CudaLayerKvStore,
        mut trace: Option<&mut Gemma4LayerTrace>,
    ) -> Result<CudaF32Buffer> {
        let config = &self.config;
        let layer_config = config.gemma4_layer_config(layer_index).ok_or_else(|| {
            XrtError::Runtime(format!("missing Gemma4 config for layer {layer_index}"))
        })?;
        if kv_cache.len() != position {
            return Err(XrtError::Runtime(format!(
                "CUDA KV cache length mismatch at Gemma4 layer {layer_index}: expected {position}, found {}",
                kv_cache.len()
            )));
        }

        macro_rules! trace_stage {
            ($name:literal, $buffer:expr) => {
                if let Some(trace) = trace.as_deref_mut() {
                    let values = self.device.download_f32($buffer)?;
                    trace.record($name, &values);
                }
            };
        }

        let attn_normed = self.device.rmsnorm_device(
            input,
            weights.attn_norm.buffer(),
            1,
            weights.embedding_length,
            config.rms_norm_eps,
        )?;
        trace_stage!("attention_norm", &attn_normed);
        let q = self.matvec_quant_resident_device(&weights.attn_q, &attn_normed)?;
        let k = self.matvec_quant_resident_device(&weights.attn_k, &attn_normed)?;
        let v = if let Some(attn_v) = &weights.attn_v {
            self.matvec_quant_resident_device(attn_v, &attn_normed)?
        } else {
            let mut v = self.device.zeros_f32(k.len())?;
            self.device.copy_f32_device(&k, &mut v)?;
            v
        };
        trace_stage!("q_projection", &q);
        trace_stage!("q_projection_float_reference", &q);
        trace_stage!("k_projection", &k);
        trace_stage!("k_projection_float_reference", &k);
        trace_stage!("v_projection", &v);
        trace_stage!("v_projection_float_reference", &v);

        let mut q = self.device.rmsnorm_device(
            &q,
            weights.attn_q_norm.buffer(),
            layer_config.head_count(),
            layer_config.head_dim(),
            config.rms_norm_eps,
        )?;
        let mut k = self.device.rmsnorm_device(
            &k,
            weights.attn_k_norm.buffer(),
            layer_config.kv_head_count(),
            layer_config.head_dim(),
            config.rms_norm_eps,
        )?;
        let v = self.device.rmsnorm_unweighted_device(
            &v,
            layer_config.kv_head_count(),
            layer_config.head_dim(),
            config.rms_norm_eps,
        )?;
        trace_stage!("q_head_norm", &q);
        trace_stage!("k_head_norm", &k);
        trace_stage!("v_head_norm", &v);

        self.device.rope_device(
            &mut q,
            layer_config.head_count(),
            layer_config.head_dim(),
            position,
            layer_config.rope_dimension_count(),
            layer_config.rope_freq_base(),
            config.rope_freq_scale,
        )?;
        self.device.rope_device(
            &mut k,
            layer_config.kv_head_count(),
            layer_config.head_dim(),
            position,
            layer_config.rope_dimension_count(),
            layer_config.rope_freq_base(),
            config.rope_freq_scale,
        )?;
        trace_stage!("q_rope", &q);
        trace_stage!("k_rope", &k);

        let attention = match kv_cache {
            CudaLayerKvStore::F32(cache) => {
                self.device.append_layer_kv(cache, &k, &v)?;
                let attend_start = layer_config
                    .sliding_window()
                    .map(|window| cache.len().saturating_sub(window))
                    .unwrap_or(0);
                self.device.single_query_attention_windowed_device(
                    &q,
                    cache,
                    layer_config.head_count(),
                    layer_config.kv_head_count(),
                    layer_config.head_dim(),
                    attend_start,
                    1.0,
                )?
            }
            CudaLayerKvStore::SharedF32(cache) => {
                cache.append(&k, &v)?;
                let attend_start = layer_config
                    .sliding_window()
                    .map(|window| cache.len().saturating_sub(window))
                    .unwrap_or(0);
                cache.single_query_attention_windowed_device(
                    &q,
                    layer_config.head_count(),
                    layer_config.kv_head_count(),
                    layer_config.head_dim(),
                    attend_start,
                    1.0,
                )?
            }
            CudaLayerKvStore::Q8(cache) => {
                self.device.append_q8_layer_kv(cache, &k, &v)?;
                let attend_start = layer_config
                    .sliding_window()
                    .map(|window| cache.len().saturating_sub(window))
                    .unwrap_or(0);
                self.device.single_query_attention_q8_windowed_device(
                    &q,
                    cache,
                    layer_config.head_count(),
                    layer_config.kv_head_count(),
                    layer_config.head_dim(),
                    attend_start,
                    1.0,
                )?
            }
            CudaLayerKvStore::KeyQ4ValueQ8(cache) => {
                self.device.append_key_q4_value_q8_layer_kv(cache, &k, &v)?;
                let attend_start = layer_config
                    .sliding_window()
                    .map(|window| cache.len().saturating_sub(window))
                    .unwrap_or(0);
                self.device
                    .single_query_attention_key_q4_value_q8_windowed_device(
                        &q,
                        cache,
                        layer_config.head_count(),
                        layer_config.kv_head_count(),
                        layer_config.head_dim(),
                        attend_start,
                        1.0,
                    )?
            }
            CudaLayerKvStore::AgentAdaptive {
                hot,
                cold,
                routes,
                hot_mask,
            } => {
                if adaptive_is_hot {
                    let local_position = hot.len();
                    self.device.append_layer_kv(hot, &k, &v)?;
                    self.device
                        .append_adaptive_kv_route(routes, true, local_position)?;
                    hot_mask.push(1);
                } else {
                    let local_position = cold.len();
                    self.device.append_key_q4_value_q8_layer_kv(cold, &k, &v)?;
                    self.device
                        .append_adaptive_kv_route(routes, false, local_position)?;
                    hot_mask.push(0);
                }
                let attend_start = layer_config
                    .sliding_window()
                    .map(|window| routes.len().saturating_sub(window))
                    .unwrap_or(0);
                self.device
                    .single_query_attention_mixed_key_q4_value_q8_windowed_device(
                        &q,
                        hot,
                        cold,
                        routes,
                        layer_config.head_count(),
                        layer_config.kv_head_count(),
                        layer_config.head_dim(),
                        attend_start,
                        1.0,
                    )?
            }
        };
        trace_stage!("attention", &attention);

        let attention_projection =
            self.matvec_quant_resident_device(&weights.attn_output, &attention)?;
        trace_stage!("attention_projection", &attention_projection);
        let post_attention_normed = self.device.rmsnorm_device(
            &attention_projection,
            weights.post_attention_norm.buffer(),
            1,
            weights.embedding_length,
            config.rms_norm_eps,
        )?;
        trace_stage!("post_attention_norm", &post_attention_normed);
        let post_attention = self.device.add_device(input, &post_attention_normed)?;
        trace_stage!("post_attention", &post_attention);

        let ffn_normed = self.device.rmsnorm_device(
            &post_attention,
            weights.ffn_norm.buffer(),
            1,
            weights.embedding_length,
            config.rms_norm_eps,
        )?;
        trace_stage!("ffn_norm", &ffn_normed);
        let mut gate = self.matvec_quant_resident_device(&weights.ffn_gate, &ffn_normed)?;
        let up = self.matvec_quant_resident_device(&weights.ffn_up, &ffn_normed)?;
        trace_stage!("ffn_gate", &gate);
        trace_stage!("ffn_up", &up);
        self.device
            .geglu_pytorch_tanh_assign_device(&mut gate, &up)?;
        trace_stage!("ffn_hidden", &gate);
        let down = self.matvec_quant_resident_device(&weights.ffn_down, &gate)?;
        trace_stage!("ffn_down", &down);
        let post_ffw_normed = self.device.rmsnorm_device(
            &down,
            weights.post_ffw_norm.buffer(),
            1,
            weights.embedding_length,
            config.rms_norm_eps,
        )?;
        trace_stage!("post_ffw_norm", &post_ffw_normed);
        let mut output = self.device.add_device(&post_attention, &post_ffw_normed)?;
        if let Some(scale) = weights.layer_output_scale {
            self.device.scale_assign_device(&mut output, scale)?;
        }
        trace_stage!("output", &output);
        Ok(output)
    }

    fn trace_gemma4_layer0(
        &self,
        token_id: u32,
        position: usize,
        session: &mut BackendSession,
    ) -> Result<Option<Gemma4LayerTrace>> {
        let config = &self.config;
        let (Some(layer_weights), Some(output_weights)) =
            (&self.gemma4_layer_probes, &self.q8_0_probe)
        else {
            return Ok(None);
        };
        let Some(layer0) = layer_weights.first() else {
            return Ok(None);
        };
        if session.cache_mode() != KvCacheMode::F32 {
            return Err(XrtError::Unsupported(
                "Gemma4 CUDA layer tracing requires XRT_KV_CACHE_MODE=f32".to_string(),
            ));
        }
        if token_id as usize >= output_weights.vocab_size {
            return Err(XrtError::Model(format!(
                "token id {token_id} exceeds embedding rows {}",
                output_weights.vocab_size
            )));
        }

        session.prepare_for_total_len(cuda_total_len_for_position(position)?)?;
        let mut x = self.embed_q8_0_probe_token(output_weights, token_id)?;
        self.device
            .scale_assign_device(&mut x, (config.embedding_length as f32).sqrt())?;

        let mut trace = Gemma4LayerTrace::new(0, position);
        trace.record("input", &self.device.download_f32(&x)?);
        let kv_cache = session.cuda_layer_cache_mut(0)?;
        x = self.run_gemma4_layer_device_with_trace(
            0,
            layer0,
            &x,
            position,
            false,
            kv_cache,
            Some(&mut trace),
        )?;

        let normed = self.device.rmsnorm_device(
            &x,
            output_weights.output_norm.buffer(),
            1,
            output_weights.embedding_length,
            config.rms_norm_eps,
        )?;
        trace.record("final_norm", &self.device.download_f32(&normed)?);
        let mut logits = self.matvec_quant_resident_device(&output_weights.output, &normed)?;
        if let Some(softcap) = config.gemma4_final_logit_softcapping() {
            self.device
                .logit_softcap_assign_device(&mut logits, softcap)?;
        }
        trace.record("logits", &self.device.download_f32(&logits)?);
        Ok(Some(trace))
    }

    fn try_forward_token_gemma4_with_logits(
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
        let config = &self.config;
        let (Some(layer_weights), Some(output_weights)) =
            (&self.gemma4_layer_probes, &self.q8_0_probe)
        else {
            return Ok(false);
        };
        if layer_weights.len() != config.block_count {
            return Ok(false);
        }
        if !matches!(
            session.cache_mode(),
            KvCacheMode::F32
                | KvCacheMode::Q8
                | KvCacheMode::KeyQ4ValueQ8
                | KvCacheMode::AgentAdaptive
        ) {
            return Err(XrtError::Unsupported(
                "Gemma4 CUDA decode supports XRT_KV_CACHE_MODE=f32, q8, kq4_vq8, or agent_adaptive"
                    .to_string(),
            ));
        }

        let layer_count = max_layers.unwrap_or(config.block_count);
        if layer_count > config.block_count {
            return Err(XrtError::Runtime(format!(
                "CUDA draft layer count {layer_count} exceeds model layer count {}",
                config.block_count
            )));
        }
        if embedding_override.is_none() && token_id as usize >= output_weights.vocab_size {
            return Err(XrtError::Model(format!(
                "token id {token_id} exceeds embedding rows {}",
                output_weights.vocab_size
            )));
        }

        let prepare_total_len = adaptive_total_len.max(cuda_total_len_for_position(position)?);
        session.prepare_for_total_len(prepare_total_len)?;
        let adaptive_is_hot = session.cuda_adaptive_position_is_hot(position, prepare_total_len);
        let mut x = if let Some(embedding) = embedding_override {
            self.device.upload_f32(embedding)?
        } else {
            self.embed_q8_0_probe_token(output_weights, token_id)?
        };
        self.device
            .scale_assign_device(&mut x, (config.embedding_length as f32).sqrt())?;

        for (layer_index, weights) in layer_weights.iter().take(layer_count).enumerate() {
            let kv_cache = session.cuda_layer_cache_mut(layer_index)?;
            x = self.run_gemma4_layer_device(
                layer_index,
                weights,
                &x,
                position,
                adaptive_is_hot,
                kv_cache,
            )?;
        }
        if !compute_logits {
            output_logits.clear();
            return Ok(true);
        }

        let normed = self.device.rmsnorm_device(
            &x,
            output_weights.output_norm.buffer(),
            1,
            output_weights.embedding_length,
            config.rms_norm_eps,
        )?;
        let mut logits = self.matvec_quant_resident_device(&output_weights.output, &normed)?;
        if let Some(softcap) = config.gemma4_final_logit_softcapping() {
            self.device
                .logit_softcap_assign_device(&mut logits, softcap)?;
        }
        *output_logits = self.device.download_f32(&logits)?;
        Ok(true)
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
        allow_graph_decode: bool,
        embedding_override: Option<&[f32]>,
        adaptive_total_len: usize,
        max_layers: Option<usize>,
    ) -> Result<bool> {
        let config = &self.config;
        if config.is_gemma4() {
            session.cuda_graph_fallback(
                "CUDA Graph decode is not yet implemented for Gemma4 variable-width layers",
            );
            return self.try_forward_token_gemma4_with_logits(
                token_id,
                position,
                session,
                output_logits,
                compute_logits,
                embedding_override,
                adaptive_total_len,
                max_layers,
            );
        }
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
        let graph_capacity_ready = allow_graph_decode
            && compute_logits
            && !profile
            && embedding_override.is_none()
            && max_layers.is_none()
            && session.cuda_graph_decode_ready();
        if graph_capacity_ready {
            if let Some(logits) = self.try_standard_dense_graph_decode(
                token_id,
                position,
                session,
                output_probe,
                layer_probes,
            )? {
                *output_logits = logits;
                return Ok(true);
            }
        }

        session.ensure_cuda_decode_scratch(
            &self.device,
            config.embedding_length,
            config.q_width(),
            config.kv_width(),
            config.feed_forward_length,
            output_probe.vocab_size,
            config.context_length,
        )?;
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
            let (kv_cache, scratch) = session.cuda_layer_cache_and_scratch_mut(layer_index)?;
            if kv_cache.len() != position {
                return Err(XrtError::Runtime(format!(
                    "CUDA KV cache length mismatch at layer {layer_index}: expected {position}, found {}",
                    kv_cache.len()
                )));
            }
            x = self.run_q8_0_layer_device_with_scratch(
                layer_index,
                layer_probe,
                &x,
                position,
                adaptive_is_hot,
                kv_cache,
                scratch,
            )?;
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
        let scratch = session.cuda_decode_scratch_mut()?;
        self.device.rmsnorm_device_into(
            &x,
            output_probe.output_norm.buffer(),
            1,
            output_probe.embedding_length,
            config.rms_norm_eps,
            &mut scratch.hidden_temp,
        )?;
        if profile {
            info!(
                position,
                ms = stage_start.elapsed().as_secs_f64() * 1000.0,
                "cuda profile: final norm"
            );
        }

        let stage_start = Instant::now();
        self.matvec_quant_resident_device_into(
            &output_probe.output,
            &scratch.hidden_temp,
            &mut scratch.logits,
        )?;
        if profile {
            info!(
                position,
                ms = stage_start.elapsed().as_secs_f64() * 1000.0,
                "cuda profile: final projection"
            );
        }

        let stage_start = Instant::now();
        let logits = self.device.download_f32(&scratch.logits)?;
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
            ResidentQuantMatrix::AwqGemm4(matrix) => {
                self.device.matvec_awq_gemm4_resident_device(matrix, input)
            }
            ResidentQuantMatrix::AwqGemv4(matrix) => {
                self.device.matvec_awq_gemv4_resident_device(matrix, input)
            }
            ResidentQuantMatrix::GptqGemm4(matrix) => {
                self.device.matvec_gptq_gemm4_resident_device(matrix, input)
            }
            ResidentQuantMatrix::GptqExplicitGemm4(matrix) => self
                .device
                .matvec_gptq_explicit_gemm4_resident_device(matrix, input),
            ResidentQuantMatrix::CompressedTensorsW4A16(matrix) => self
                .device
                .matvec_compressed_tensors_w4a16_resident_device(matrix, input),
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

    fn matvec_quant_resident_device_into(
        &self,
        matrix: &ResidentQuantMatrix,
        input: &CudaF32Buffer,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        match matrix {
            ResidentQuantMatrix::F32(tensor) => self.device.matmul_resident_rhs_device_into(
                input,
                1,
                tensor.dimensions[1],
                tensor.buffer(),
                tensor.dimensions[0],
                output,
            ),
            ResidentQuantMatrix::AwqGemm4(matrix) => self
                .device
                .matvec_awq_gemm4_resident_device_into(matrix, input, output),
            ResidentQuantMatrix::AwqGemv4(matrix) => self
                .device
                .matvec_awq_gemv4_resident_device_into(matrix, input, output),
            ResidentQuantMatrix::GptqGemm4(matrix) => self
                .device
                .matvec_gptq_gemm4_resident_device_into(matrix, input, output),
            ResidentQuantMatrix::GptqExplicitGemm4(matrix) => self
                .device
                .matvec_gptq_explicit_gemm4_resident_device_into(matrix, input, output),
            ResidentQuantMatrix::CompressedTensorsW4A16(matrix) => self
                .device
                .matvec_compressed_tensors_w4a16_resident_device_into(matrix, input, output),
            ResidentQuantMatrix::Q8_0(matrix) => self
                .device
                .matvec_q8_0_resident_device_into(matrix, input, output),
            ResidentQuantMatrix::Q4_0(matrix) => self
                .device
                .matvec_q4_0_resident_device_into(matrix, input, output),
            ResidentQuantMatrix::Q4K(matrix) => self
                .device
                .matvec_q4_k_resident_device_into(matrix, input, output),
            ResidentQuantMatrix::Q5K(matrix) => self
                .device
                .matvec_q5_k_resident_device_into(matrix, input, output),
            ResidentQuantMatrix::Q6K(matrix) => self
                .device
                .matvec_q6_k_resident_device_into(matrix, input, output),
        }
    }

    fn standard_dense_graph_key(
        &self,
        output: &ResidentQ8_0ProbeWeights,
        layers: &[ResidentQ8_0LayerWeights],
        cache_mode: KvCacheMode,
        kv_capacity: usize,
    ) -> CudaDecodeGraphKey {
        let config = &self.config;
        let mut weight_kinds = Vec::with_capacity(2 + layers.len() * 10);
        weight_kinds.push(output.token_embedding.graph_kind());
        weight_kinds.push(output.output.graph_kind());
        for layer in layers {
            weight_kinds.extend([
                layer.attn_q.graph_kind(),
                layer.attn_k.graph_kind(),
                layer.attn_v.graph_kind(),
                layer.attn_output.graph_kind(),
                layer.ffn_gate.graph_kind(),
                layer.ffn_up.graph_kind(),
                layer.ffn_down.graph_kind(),
                if layer.attn_q_norm.is_some() {
                    "q-norm"
                } else {
                    "no-q-norm"
                },
                if layer.attn_k_norm.is_some() {
                    "k-norm"
                } else {
                    "no-k-norm"
                },
                if layer.attn_q_bias.is_some()
                    || layer.attn_k_bias.is_some()
                    || layer.attn_v_bias.is_some()
                {
                    "qkv-bias"
                } else {
                    "no-qkv-bias"
                },
            ]);
        }
        CudaDecodeGraphKey {
            architecture: config.architecture.clone(),
            weight_kinds,
            cache_mode,
            kv_capacity,
            layer_count: layers.len(),
            embedding_length: config.embedding_length,
            kv_width: config.kv_width(),
            feed_forward_length: config.feed_forward_length,
            vocab_size: config.vocab_size,
            attention_head_count: config.attention_head_count,
            attention_head_count_kv: config.attention_head_count_kv,
            head_dim: config.head_dim(),
        }
    }

    fn embed_probe_with_decode_params_into(
        &self,
        probe: &ResidentQ8_0ProbeWeights,
        params: &CudaDecodeParams,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        match &probe.token_embedding {
            ResidentTokenEmbedding::F32(tensor) => {
                self.device.embed_resident_with_decode_params_into(
                    tensor.buffer(),
                    probe.vocab_size,
                    probe.embedding_length,
                    params,
                    output,
                )
            }
            ResidentTokenEmbedding::Q8_0(matrix) => self
                .device
                .embed_q8_0_with_decode_params_into(matrix, params, output),
            ResidentTokenEmbedding::Q4_0(matrix) => self
                .device
                .embed_q8_0_with_decode_params_into(matrix, params, output),
            ResidentTokenEmbedding::Q4K(matrix) => self
                .device
                .embed_q4_k_with_decode_params_into(matrix, params, output),
            ResidentTokenEmbedding::Q5K(matrix) => self
                .device
                .embed_q5_k_with_decode_params_into(matrix, params, output),
            ResidentTokenEmbedding::Q6K(matrix) => self
                .device
                .embed_q6_k_with_decode_params_into(matrix, params, output),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_standard_dense_graph_layer(
        &self,
        weights: &ResidentQ8_0LayerWeights,
        input: &CudaF32Buffer,
        output: &mut CudaF32Buffer,
        params: &CudaDecodeParams,
        cache: &mut CudaLayerKvCache,
        normed_post_attention: &mut CudaF32Buffer,
        q: &mut CudaF32Buffer,
        q_temp: &mut CudaF32Buffer,
        k: &mut CudaF32Buffer,
        v: &mut CudaF32Buffer,
        hidden_temp: &mut CudaF32Buffer,
        kv_temp: &mut CudaF32Buffer,
        gate: &mut CudaF32Buffer,
        up: &mut CudaF32Buffer,
        attention: &mut CudaF32Buffer,
    ) -> Result<()> {
        let config = &self.config;
        self.device.rmsnorm_device_into(
            input,
            weights.attn_norm.buffer(),
            1,
            weights.embedding_length,
            config.rms_norm_eps,
            normed_post_attention,
        )?;
        self.matvec_quant_resident_device_into(&weights.attn_q, normed_post_attention, q)?;
        self.matvec_quant_resident_device_into(&weights.attn_k, normed_post_attention, k)?;
        self.matvec_quant_resident_device_into(&weights.attn_v, normed_post_attention, v)?;
        if let Some(bias) = &weights.attn_q_bias {
            self.device.add_assign_device(q, bias.buffer())?;
        }
        if let Some(bias) = &weights.attn_k_bias {
            self.device.add_assign_device(k, bias.buffer())?;
        }
        if let Some(bias) = &weights.attn_v_bias {
            self.device.add_assign_device(v, bias.buffer())?;
        }

        let query = if let Some(q_norm) = &weights.attn_q_norm {
            self.device.rmsnorm_device_into(
                q,
                q_norm.buffer(),
                config.attention_head_count,
                config.head_dim(),
                config.rms_norm_eps,
                q_temp,
            )?;
            self.device.rope_device_with_decode_params(
                q_temp,
                config.attention_head_count,
                config.head_dim(),
                params,
                config.rope_dimension_count,
                config.rope_freq_base,
                config.rope_freq_scale,
            )?;
            &*q_temp
        } else {
            self.device.rope_device_with_decode_params(
                q,
                config.attention_head_count,
                config.head_dim(),
                params,
                config.rope_dimension_count,
                config.rope_freq_base,
                config.rope_freq_scale,
            )?;
            &*q
        };
        let key = if let Some(k_norm) = &weights.attn_k_norm {
            self.device.rmsnorm_device_into(
                k,
                k_norm.buffer(),
                config.attention_head_count_kv,
                config.head_dim(),
                config.rms_norm_eps,
                kv_temp,
            )?;
            self.device.rope_device_with_decode_params(
                kv_temp,
                config.attention_head_count_kv,
                config.head_dim(),
                params,
                config.rope_dimension_count,
                config.rope_freq_base,
                config.rope_freq_scale,
            )?;
            &*kv_temp
        } else {
            self.device.rope_device_with_decode_params(
                k,
                config.attention_head_count_kv,
                config.head_dim(),
                params,
                config.rope_dimension_count,
                config.rope_freq_base,
                config.rope_freq_scale,
            )?;
            &*k
        };

        self.device
            .append_layer_kv_with_decode_params(cache, key, v, params)?;
        self.device.single_query_attention_with_decode_params_into(
            query,
            cache,
            params,
            config.attention_head_count,
            config.attention_head_count_kv,
            config.head_dim(),
            1.0 / (config.head_dim() as f32).sqrt(),
            attention,
        )?;
        self.matvec_quant_resident_device_into(&weights.attn_output, attention, hidden_temp)?;
        self.device.copy_f32_device(input, normed_post_attention)?;
        self.device
            .add_assign_device(normed_post_attention, hidden_temp)?;

        self.device.rmsnorm_device_into(
            normed_post_attention,
            weights.ffn_norm.buffer(),
            1,
            weights.embedding_length,
            config.rms_norm_eps,
            hidden_temp,
        )?;
        self.matvec_quant_resident_device_into(&weights.ffn_gate, hidden_temp, gate)?;
        self.matvec_quant_resident_device_into(&weights.ffn_up, hidden_temp, up)?;
        self.device.silu_assign_device(gate)?;
        self.device.mul_assign_device(gate, up)?;
        self.matvec_quant_resident_device_into(&weights.ffn_down, gate, output)?;
        self.device.add_assign_device(output, normed_post_attention)
    }

    fn run_standard_dense_graph_ops(
        &self,
        output_weights: &ResidentQ8_0ProbeWeights,
        layer_weights: &[ResidentQ8_0LayerWeights],
        layer_caches: &mut [CudaLayerKvStore],
        scratch: &mut CudaDecodeScratch,
    ) -> Result<()> {
        if layer_caches.len() != layer_weights.len() {
            return Err(XrtError::Runtime(format!(
                "CUDA graph layer cache count {} does not match weight count {}",
                layer_caches.len(),
                layer_weights.len()
            )));
        }
        let CudaDecodeScratch {
            decode_params,
            layer_input_a,
            layer_input_b,
            attention,
            normed_post_attention,
            q,
            q_temp,
            k,
            v,
            hidden_temp,
            kv_temp,
            gate,
            up,
            logits,
            ..
        } = scratch;

        self.embed_probe_with_decode_params_into(output_weights, decode_params, layer_input_a)?;
        let mut input_is_a = true;
        for (weights, cache) in layer_weights.iter().zip(layer_caches) {
            let cache = match cache {
                CudaLayerKvStore::F32(cache) => cache,
                other => {
                    return Err(XrtError::Unsupported(format!(
                        "CUDA Graph standard decode requires f32 KV, found {}",
                        other.mode().as_str()
                    )));
                }
            };
            if input_is_a {
                self.run_standard_dense_graph_layer(
                    weights,
                    layer_input_a,
                    layer_input_b,
                    decode_params,
                    cache,
                    normed_post_attention,
                    q,
                    q_temp,
                    k,
                    v,
                    hidden_temp,
                    kv_temp,
                    gate,
                    up,
                    attention,
                )?;
            } else {
                self.run_standard_dense_graph_layer(
                    weights,
                    layer_input_b,
                    layer_input_a,
                    decode_params,
                    cache,
                    normed_post_attention,
                    q,
                    q_temp,
                    k,
                    v,
                    hidden_temp,
                    kv_temp,
                    gate,
                    up,
                    attention,
                )?;
            }
            input_is_a = !input_is_a;
        }

        let final_hidden = if input_is_a {
            &*layer_input_a
        } else {
            &*layer_input_b
        };
        self.device.rmsnorm_device_into(
            final_hidden,
            output_weights.output_norm.buffer(),
            1,
            output_weights.embedding_length,
            self.config.rms_norm_eps,
            hidden_temp,
        )?;
        self.matvec_quant_resident_device_into(&output_weights.output, hidden_temp, logits)
    }

    fn validate_standard_dense_graph_caches(
        layer_caches: &[CudaLayerKvStore],
        position: usize,
        capacity: usize,
    ) -> Result<()> {
        for (layer, cache) in layer_caches.iter().enumerate() {
            let CudaLayerKvStore::F32(cache) = cache else {
                return Err(XrtError::Unsupported(
                    "CUDA Graph standard decode requires f32 KV".to_string(),
                ));
            };
            if cache.len() != position {
                return Err(XrtError::Runtime(format!(
                    "CUDA graph layer {layer} expected KV len {position}, found {}",
                    cache.len()
                )));
            }
            if cache.capacity() != capacity {
                return Err(XrtError::Runtime(format!(
                    "CUDA graph layer {layer} expected KV capacity {capacity}, found {}",
                    cache.capacity()
                )));
            }
        }
        Ok(())
    }

    fn commit_standard_dense_graph_caches(
        &self,
        layer_caches: &mut [CudaLayerKvStore],
        position: usize,
    ) -> Result<()> {
        for cache in layer_caches {
            let CudaLayerKvStore::F32(cache) = cache else {
                return Err(XrtError::Unsupported(
                    "CUDA Graph standard decode requires f32 KV".to_string(),
                ));
            };
            self.device.commit_layer_kv_graph_append(cache, position)?;
        }
        Ok(())
    }

    fn download_standard_dense_batch_graph_outputs(
        &self,
        batch: &mut [BackendDecodeBatchItem],
    ) -> Result<()> {
        for item in batch {
            {
                let (_, layer_caches, scratch) = item.session.cuda_graph_parts_mut()?;
                item.output_logits = self.device.download_f32(&scratch.logits)?;
                self.commit_standard_dense_graph_caches(layer_caches, item.position)?;
            }
            item.session.mark_cuda_batch_graph_captured();
        }
        Ok(())
    }

    fn try_concurrent_standard_dense_graph_decode(
        &self,
        batch: &mut [BackendDecodeBatchItem],
    ) -> Result<Option<BackendDecodeBatchExecution>> {
        if batch.len() < 2
            || self.cuda_graph_mode == CudaGraphMode::Disabled
            || Self::cuda_profile_enabled()
            || self.config.is_gemma4()
            || self.config.is_hybrid()
            || batch
                .iter()
                .any(|item| item.session.cache_mode() != KvCacheMode::F32)
        {
            return Ok(None);
        }
        let (Some(layer_weights), Some(output_weights)) =
            (&self.q8_0_layer_probes, &self.q8_0_probe)
        else {
            return Ok(None);
        };
        if layer_weights.len() != self.config.block_count {
            return Ok(None);
        }

        let config = &self.config;
        let mut session_keys = Vec::with_capacity(batch.len());
        for item in batch.iter_mut() {
            let total_len = cuda_total_len_for_position(item.position)?;
            item.session.prepare_for_total_len(total_len)?;
            if !item.session.cuda_graph_decode_ready() {
                return Ok(None);
            }
            let kv_capacity = item.session.cuda_kv_capacity().ok_or_else(|| {
                XrtError::Runtime(
                    "concurrent CUDA graph replay requires allocated KV caches".to_string(),
                )
            })?;
            item.session.ensure_cuda_decode_scratch(
                &self.device,
                config.embedding_length,
                config.q_width(),
                config.kv_width(),
                config.feed_forward_length,
                output_weights.vocab_size,
                kv_capacity,
            )?;
            if item.session.cuda_graph_executable().is_none() {
                return Ok(None);
            }
            let (_, layer_caches, scratch) = item.session.cuda_graph_parts_mut()?;
            Self::validate_standard_dense_graph_caches(layer_caches, item.position, kv_capacity)?;
            self.device.update_decode_params(
                &mut scratch.decode_params,
                item.token_id,
                item.position,
                total_len,
                0,
            )?;
            let epoch = item.session.cuda_batch_graph_epoch().ok_or_else(|| {
                XrtError::Runtime("CUDA batch graph received a CPU backend session".to_string())
            })?;
            session_keys.push((
                item.sequence_id,
                epoch,
                self.standard_dense_graph_key(
                    output_weights,
                    layer_weights,
                    KvCacheMode::F32,
                    kv_capacity,
                ),
            ));
        }

        let key = CudaDecodeBatchGraphKey {
            sessions: session_keys,
        };
        {
            let mut cache = self.decode_batch_graphs.lock();
            let mut parent_launched = false;
            if let Some(state) = cache.entry_mut(&key) {
                if let CudaDecodeBatchGraphEntryState::Captured(graph) = state {
                    match graph.launch() {
                        Ok(()) => parent_launched = true,
                        Err(err) => {
                            tracing::warn!(
                                "parallel CUDA decode graph launch failed; using stream replay: {err}"
                            );
                            *state = CudaDecodeBatchGraphEntryState::EagerFallback;
                        }
                    }
                }
            } else {
                let child_graphs = batch
                    .iter()
                    .map(|item| {
                        item.session.cuda_graph_executable().ok_or_else(|| {
                            XrtError::Runtime(
                                "CUDA decode graph disappeared before composition".to_string(),
                            )
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                match self.device.compose_parallel_graphs(&child_graphs) {
                    Ok(graph) => {
                        info!(
                            batch_size = batch.len(),
                            nodes = graph.node_count(),
                            "composed parallel CUDA multi-sequence decode graph"
                        );
                        match graph.launch() {
                            Ok(()) => {
                                cache.insert(
                                    key.clone(),
                                    CudaDecodeBatchGraphEntryState::Captured(graph),
                                );
                                parent_launched = true;
                            }
                            Err(err) => {
                                tracing::warn!(
                                    "parallel CUDA decode graph launch failed; using stream replay: {err}"
                                );
                                cache.insert(
                                    key.clone(),
                                    CudaDecodeBatchGraphEntryState::EagerFallback,
                                );
                            }
                        }
                    }
                    Err(err) => {
                        tracing::warn!(
                            "parallel CUDA decode graph composition failed; using stream replay: {err}"
                        );
                        cache.insert(key.clone(), CudaDecodeBatchGraphEntryState::EagerFallback);
                    }
                }
            }
            if parent_launched {
                self.download_standard_dense_batch_graph_outputs(batch)?;
                return Ok(Some(BackendDecodeBatchExecution { fused: true }));
            }
        }

        let mut streams = self.decode_batch_streams.lock();
        while streams.len() < batch.len() {
            match self.device.create_execution_stream() {
                Ok(stream) => streams.push(stream),
                Err(err) => {
                    tracing::warn!(
                        "CUDA concurrent decode stream creation failed; using serial graph replay: {err}"
                    );
                    return Ok(None);
                }
            }
        }

        let mut launched = 0usize;
        for (item, stream) in batch.iter().zip(streams.iter()) {
            let graph = item.session.cuda_graph_executable().ok_or_else(|| {
                XrtError::Runtime(
                    "CUDA decode graph disappeared before concurrent launch".to_string(),
                )
            })?;
            if let Err(err) = graph.launch_on_stream(stream) {
                for launched_stream in streams.iter().take(launched) {
                    let _ = launched_stream.synchronize();
                }
                return Err(err);
            }
            launched += 1;
        }
        for stream in streams.iter().take(batch.len()) {
            stream.synchronize()?;
        }
        drop(streams);

        self.download_standard_dense_batch_graph_outputs(batch)?;
        Ok(Some(BackendDecodeBatchExecution { fused: true }))
    }

    fn try_standard_dense_graph_decode(
        &self,
        token_id: u32,
        position: usize,
        session: &mut BackendSession,
        output_weights: &ResidentQ8_0ProbeWeights,
        layer_weights: &[ResidentQ8_0LayerWeights],
    ) -> Result<Option<Vec<f32>>> {
        if !session.cuda_graph_decode_ready() {
            return Ok(None);
        }
        let config = &self.config;
        let kv_capacity = session.cuda_kv_capacity().ok_or_else(|| {
            XrtError::Runtime("CUDA Graph decode requires allocated KV caches".to_string())
        })?;
        session.ensure_cuda_decode_scratch(
            &self.device,
            config.embedding_length,
            config.q_width(),
            config.kv_width(),
            config.feed_forward_length,
            output_weights.vocab_size,
            kv_capacity,
        )?;
        let (graph_state, layer_caches, scratch) = session.cuda_graph_parts_mut()?;
        if !graph_state.is_enabled() {
            return Ok(None);
        }
        let key = self.standard_dense_graph_key(
            output_weights,
            layer_weights,
            KvCacheMode::F32,
            kv_capacity,
        );
        if graph_state
            .key
            .as_ref()
            .is_some_and(|existing| existing != &key)
        {
            graph_state.reset();
        }
        Self::validate_standard_dense_graph_caches(layer_caches, position, kv_capacity)?;
        // Every prior graph replay is synchronized by the logits download, and the update plus
        // graph launch use the same stream, so the owned async parameter upload remains live and
        // ordered without adding another per-token synchronization.
        unsafe {
            self.device.update_decode_params_async(
                &mut scratch.decode_params,
                token_id,
                position,
                position + 1,
                0,
            )?;
        }

        if let Some(graph) = graph_state.executable.as_ref() {
            if let Err(err) = graph.launch() {
                graph_state.fallback(err.to_string());
                tracing::warn!("CUDA Graph launch failed; using eager CUDA: {err}");
                return Ok(None);
            }
            let logits = self.device.download_f32(&scratch.logits)?;
            self.commit_standard_dense_graph_caches(layer_caches, position)?;
            return Ok(Some(logits));
        }

        if let Err(err) =
            self.run_standard_dense_graph_ops(output_weights, layer_weights, layer_caches, scratch)
        {
            graph_state.fallback(err.to_string());
            tracing::warn!("CUDA Graph warm execution failed; using eager CUDA: {err}");
            return Ok(None);
        }
        let logits = self.device.download_f32(&scratch.logits)?;
        self.commit_standard_dense_graph_caches(layer_caches, position)?;

        let captured = unsafe {
            self.device.capture_graph(|| {
                self.run_standard_dense_graph_ops(
                    output_weights,
                    layer_weights,
                    layer_caches,
                    scratch,
                )
            })
        };
        match captured {
            Ok(graph) => {
                info!(
                    nodes = graph.node_count(),
                    "captured CUDA batch-1 decode graph"
                );
                graph_state.captured(key, graph);
            }
            Err(err) => {
                graph_state.fallback(err.to_string());
                tracing::warn!("CUDA Graph capture failed; using eager CUDA: {err}");
            }
        }
        Ok(Some(logits))
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

fn upload_resident_f32_tensor(
    device: &CudaDevice,
    source: &impl ResidentTensorSource,
    name: &str,
) -> Result<GpuF32Tensor> {
    let info = source.require_tensor(name)?;
    if info.storage != ResidentTensorStorage::Dense {
        return Err(XrtError::InvalidTensor(format!(
            "resident F32 tensor `{name}` uses non-dense storage"
        )));
    }
    device.upload_f32_tensor_bytes(
        name,
        &info.dimensions,
        info.dtype,
        source.tensor_data(name)?,
    )
}

fn upload_resident_f32_tensor_transposed_2d(
    device: &CudaDevice,
    source: &impl ResidentTensorSource,
    name: &str,
) -> Result<GpuF32Tensor> {
    let info = source.require_tensor(name)?;
    if info.storage != ResidentTensorStorage::Dense || info.rank != 2 {
        return Err(XrtError::Unsupported(format!(
            "resident transposed F32 tensor upload requires a 2D tensor, tensor `{name}` has dimensions {:?}",
            info.dimensions
        )));
    }
    device.upload_f32_tensor_transposed_2d_bytes(
        name,
        info.rows,
        info.cols,
        info.dtype,
        source.tensor_data(name)?,
    )
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
        source: &impl ResidentTensorSource,
        config: &LlamaConfig,
    ) -> Result<Option<Self>> {
        let token_embedding_name = "token_embd.weight";
        let output_norm_name = "output_norm.weight";
        let output_name = if source.tensor_info("output.weight").is_some() {
            "output.weight"
        } else {
            token_embedding_name
        };

        let Some(token_embedding_info) = source.tensor_info(token_embedding_name) else {
            return Ok(None);
        };
        let Some(output_norm_info) = source.tensor_info(output_norm_name) else {
            return Ok(None);
        };
        let Some(output_info) = source.tensor_info(output_name) else {
            return Ok(None);
        };

        if !is_supported_resident_float_tensor(&token_embedding_info)
            || !is_supported_resident_float_tensor(&output_norm_info)
            || !is_supported_resident_float_tensor(&output_info)
        {
            return Ok(None);
        }
        if token_embedding_info.cols != config.embedding_length
            || token_embedding_info.rows != config.vocab_size
            || output_norm_info.numel != config.embedding_length
            || output_info.cols != config.embedding_length
            || output_info.rows != config.vocab_size
        {
            return Ok(None);
        }

        Ok(Some(Self {
            token_embedding: upload_resident_f32_tensor(device, source, token_embedding_name)?,
            output_norm: upload_resident_f32_tensor(device, source, output_norm_name)?,
            output_transposed: upload_resident_f32_tensor_transposed_2d(
                device,
                source,
                output_name,
            )?,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CudaKQuantEmbeddingLayout {
    ExpandedF32,
    Packed,
}

impl CudaKQuantEmbeddingLayout {
    fn as_str(self) -> &'static str {
        match self {
            Self::ExpandedF32 => "expanded-f32",
            Self::Packed => "packed",
        }
    }
}

impl ResidentTokenEmbedding {
    fn graph_kind(&self) -> &'static str {
        match self {
            Self::F32(_) => "f32",
            Self::Q8_0(_) => "q8_0",
            Self::Q4_0(_) => "q4_0",
            Self::Q4K(_) => "q4_k",
            Self::Q5K(_) => "q5_k",
            Self::Q6K(_) => "q6_k",
        }
    }

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
    AwqGemm4(Arc<CudaAwqGemm4Matrix>),
    AwqGemv4(Arc<CudaAwqGemv4Matrix>),
    GptqGemm4(Arc<CudaGptqGemm4Matrix>),
    GptqExplicitGemm4(Arc<CudaGptqExplicitGemm4Matrix>),
    CompressedTensorsW4A16(Arc<CudaCompressedTensorsW4A16Matrix>),
    Q8_0(Arc<CudaQ8_0Matrix>),
    Q4_0(Arc<CudaQ4_0Matrix>),
    Q4K(Arc<CudaQ4KMatrix>),
    Q5K(Arc<CudaQ5KMatrix>),
    Q6K(Arc<CudaQ6KMatrix>),
}

impl ResidentQuantMatrix {
    fn graph_kind(&self) -> &'static str {
        match self {
            Self::F32(_) => "f32",
            Self::AwqGemm4(_) => "awq_gemm4",
            Self::AwqGemv4(_) => "awq_gemv4",
            Self::GptqGemm4(_) => "gptq_gemm4",
            Self::GptqExplicitGemm4(_) => "gptq_explicit_gemm4",
            Self::CompressedTensorsW4A16(_) => "compressed_tensors_w4a16",
            Self::Q8_0(_) => "q8_0",
            Self::Q4_0(_) => "q4_0",
            Self::Q4K(_) => "q4_k",
            Self::Q5K(_) => "q5_k",
            Self::Q6K(_) => "q6_k",
        }
    }

    fn upload(device: &CudaDevice, source: &impl ResidentTensorSource, name: &str) -> Result<Self> {
        let info = source.require_tensor(name)?;
        if let ResidentTensorStorage::AwqGemm4 { group_size } = info.storage {
            let data = source.awq_gemm4_data(name)?.ok_or_else(|| {
                XrtError::InvalidTensor(format!(
                    "tensor `{name}` declares AWQ GEMM4 storage without component data"
                ))
            })?;
            if data.rows != info.rows || data.cols != info.cols || data.group_size != group_size {
                return Err(XrtError::InvalidTensor(format!(
                    "tensor `{name}` AWQ metadata changed between validation and upload"
                )));
            }
            return device
                .upload_awq_gemm4_matrix(
                    data.qweight,
                    data.qzeros,
                    data.scales,
                    data.scale_dtype,
                    data.rows,
                    data.cols,
                    data.group_size,
                )
                .map(Arc::new)
                .map(Self::AwqGemm4);
        }
        if let ResidentTensorStorage::AwqGemv4 {
            group_size,
            zero_words_per_row,
        } = info.storage
        {
            let data = source.awq_gemv4_data(name)?.ok_or_else(|| {
                XrtError::InvalidTensor(format!(
                    "tensor `{name}` declares AWQ GEMV4 storage without component data"
                ))
            })?;
            if data.rows != info.rows
                || data.cols != info.cols
                || data.group_size != group_size
                || data.zero_words_per_row != zero_words_per_row
                || zero_words_per_row.checked_mul(8) != Some(data.scale_stride)
            {
                return Err(XrtError::InvalidTensor(format!(
                    "tensor `{name}` AWQ GEMV metadata changed between validation and upload"
                )));
            }
            return device
                .upload_awq_gemv4_matrix(
                    data.qweight,
                    data.qzeros,
                    data.scales,
                    data.scale_dtype,
                    data.rows,
                    data.cols,
                    data.group_size,
                )
                .map(Arc::new)
                .map(Self::AwqGemv4);
        }
        if let ResidentTensorStorage::GptqGemm4 { group_size } = info.storage {
            let data = source.gptq_gemm4_data(name)?.ok_or_else(|| {
                XrtError::InvalidTensor(format!(
                    "tensor `{name}` declares GPTQ GEMM4 storage without component data"
                ))
            })?;
            if data.rows != info.rows || data.cols != info.cols || data.group_size != group_size {
                return Err(XrtError::InvalidTensor(format!(
                    "tensor `{name}` GPTQ metadata changed between validation and upload"
                )));
            }
            return device
                .upload_gptq_gemm4_matrix(
                    data.qweight,
                    data.qzeros,
                    data.scales,
                    data.scale_dtype,
                    data.rows,
                    data.cols,
                    data.group_size,
                )
                .map(Arc::new)
                .map(Self::GptqGemm4);
        }
        if let ResidentTensorStorage::GptqExplicitGemm4 {
            group_size,
            zero_encoding,
        } = info.storage
        {
            let data = source.gptq_explicit_gemm4_data(name)?.ok_or_else(|| {
                XrtError::InvalidTensor(format!(
                    "tensor `{name}` declares explicit-group GPTQ GEMM4 storage without component data"
                ))
            })?;
            if data.rows != info.rows
                || data.cols != info.cols
                || data.group_size != group_size
                || data.zero_encoding != zero_encoding
            {
                return Err(XrtError::InvalidTensor(format!(
                    "tensor `{name}` explicit-group GPTQ metadata changed between validation and upload"
                )));
            }
            return device
                .upload_gptq_explicit_gemm4_matrix(
                    data.qweight,
                    data.qzeros,
                    data.scales,
                    data.scale_dtype,
                    data.group_indices,
                    data.rows,
                    data.cols,
                    data.group_size,
                    data.zero_encoding,
                )
                .map(Arc::new)
                .map(Self::GptqExplicitGemm4);
        }
        if let ResidentTensorStorage::CompressedTensorsW4A16 { group_size } = info.storage {
            let data = source
                .compressed_tensors_w4a16_data(name)?
                .ok_or_else(|| {
                    XrtError::InvalidTensor(format!(
                        "tensor `{name}` declares compressed-tensors W4A16 storage without component data"
                    ))
                })?;
            if data.rows != info.rows || data.cols != info.cols || data.group_size != group_size {
                return Err(XrtError::InvalidTensor(format!(
                    "tensor `{name}` compressed-tensors metadata changed between validation and upload"
                )));
            }
            return device
                .upload_compressed_tensors_w4a16_matrix(
                    data.weight_packed,
                    data.scales,
                    data.scale_dtype,
                    data.group_indices,
                    data.rows,
                    data.cols,
                    data.group_size,
                )
                .map(Arc::new)
                .map(Self::CompressedTensorsW4A16);
        }

        let data = source.tensor_data(name)?;
        match info.dtype {
            DType::F32 | DType::F16 | DType::BF16 => {
                upload_resident_f32_tensor_transposed_2d(device, source, name).map(Self::F32)
            }
            DType::Q8_0 => device
                .upload_q8_0_matrix(data, info.rows, info.cols)
                .map(Arc::new)
                .map(Self::Q8_0),
            DType::Q4_0 => device
                .upload_q4_0_matrix(data, info.rows, info.cols)
                .map(Arc::new)
                .map(Self::Q4_0),
            DType::Q4_K => device
                .upload_q4_k_matrix(data, info.rows, info.cols)
                .map(Arc::new)
                .map(Self::Q4K),
            DType::Q5_K => device
                .upload_q5_k_matrix(data, info.rows, info.cols)
                .map(Arc::new)
                .map(Self::Q5K),
            DType::Q6_K => device
                .upload_q6_k_matrix(data, info.rows, info.cols)
                .map(Arc::new)
                .map(Self::Q6K),
        }
    }

    fn is_q8_0(&self) -> bool {
        matches!(self, Self::Q8_0(_))
    }
}

impl ResidentQ8_0ProbeWeights {
    fn output_name(source: &impl ResidentTensorSource) -> &'static str {
        if source.tensor_info("output.weight").is_some() {
            "output.weight"
        } else {
            "token_embd.weight"
        }
    }

    fn supports(source: &impl ResidentTensorSource, config: &LlamaConfig) -> bool {
        let token_embedding_name = "token_embd.weight";
        let output_norm_name = "output_norm.weight";
        let output_name = Self::output_name(source);

        let Some(token_embedding_info) = source.tensor_info(token_embedding_name) else {
            return false;
        };
        let Some(output_norm_info) = source.tensor_info(output_norm_name) else {
            return false;
        };
        let Some(output_info) = source.tensor_info(output_name) else {
            return false;
        };

        token_embedding_info.storage == ResidentTensorStorage::Dense
            && is_supported_resident_linear_tensor(&token_embedding_info)
            && is_supported_resident_float_tensor(&output_norm_info)
            && is_supported_resident_linear_tensor(&output_info)
            && token_embedding_info.cols == config.embedding_length
            && token_embedding_info.rows == config.vocab_size
            && output_norm_info.numel == config.embedding_length
            && output_info.cols == config.embedding_length
            && output_info.rows == config.vocab_size
    }

    fn try_load(
        device: &CudaDevice,
        source: &impl ResidentTensorSource,
        config: &LlamaConfig,
    ) -> Result<Option<Self>> {
        let token_embedding_name = "token_embd.weight";
        let output_norm_name = "output_norm.weight";
        let output_name = Self::output_name(source);

        if !Self::supports(source, config) {
            return Ok(None);
        }
        let token_embedding_info = source
            .tensor_info(token_embedding_name)
            .expect("token embedding tensor was checked above");
        let token_embedding_data = source.tensor_data(token_embedding_name)?;
        let token_embedding = match token_embedding_info.dtype {
            DType::F32 | DType::F16 | DType::BF16 => ResidentTokenEmbedding::F32(
                upload_resident_f32_tensor(device, source, token_embedding_name)?,
            ),
            DType::Q8_0 => ResidentTokenEmbedding::Q8_0(Arc::new(device.upload_q8_0_matrix(
                token_embedding_data,
                token_embedding_info.rows,
                token_embedding_info.cols,
            )?)),
            DType::Q4_0 => ResidentTokenEmbedding::Q4_0(Arc::new(device.upload_q4_0_matrix(
                token_embedding_data,
                token_embedding_info.rows,
                token_embedding_info.cols,
            )?)),
            DType::Q4_K => {
                let layout = cuda_k_quant_embedding_layout(&token_embedding_info)?;
                let resident_bytes = cuda_embedding_resident_tensor_bytes(&token_embedding_info)?;
                info!(
                    tensor = token_embedding_name,
                    rows = token_embedding_info.rows,
                    cols = token_embedding_info.cols,
                    layout = layout.as_str(),
                    resident_bytes,
                    "selected CUDA Q4_K token embedding layout"
                );
                let matrix = match layout {
                    CudaKQuantEmbeddingLayout::ExpandedF32 => device.upload_q4_k_embedding_matrix(
                        token_embedding_data,
                        token_embedding_info.rows,
                        token_embedding_info.cols,
                    )?,
                    CudaKQuantEmbeddingLayout::Packed => device.upload_q4_k_matrix(
                        token_embedding_data,
                        token_embedding_info.rows,
                        token_embedding_info.cols,
                    )?,
                };
                ResidentTokenEmbedding::Q4K(Arc::new(matrix))
            }
            DType::Q5_K => {
                ResidentTokenEmbedding::Q5K(Arc::new(device.upload_q5_k_embedding_matrix(
                    token_embedding_data,
                    token_embedding_info.rows,
                    token_embedding_info.cols,
                )?))
            }
            DType::Q6_K => {
                let layout = cuda_k_quant_embedding_layout(&token_embedding_info)?;
                let resident_bytes = cuda_embedding_resident_tensor_bytes(&token_embedding_info)?;
                info!(
                    tensor = token_embedding_name,
                    rows = token_embedding_info.rows,
                    cols = token_embedding_info.cols,
                    layout = layout.as_str(),
                    resident_bytes,
                    "selected CUDA Q6_K token embedding layout"
                );
                let matrix = match layout {
                    CudaKQuantEmbeddingLayout::ExpandedF32 => device.upload_q6_k_embedding_matrix(
                        token_embedding_data,
                        token_embedding_info.rows,
                        token_embedding_info.cols,
                    )?,
                    CudaKQuantEmbeddingLayout::Packed => device
                        .upload_q6_k_embedding_matrix_packed(
                            token_embedding_data,
                            token_embedding_info.rows,
                            token_embedding_info.cols,
                        )?,
                };
                ResidentTokenEmbedding::Q6K(Arc::new(matrix))
            }
        };
        let output = if output_name == token_embedding_name {
            if let Some(shared) = token_embedding.tied_output_matrix() {
                shared
            } else {
                ResidentQuantMatrix::upload(device, source, output_name)?
            }
        } else {
            ResidentQuantMatrix::upload(device, source, output_name)?
        };

        Ok(Some(Self {
            token_embedding,
            output_norm: upload_resident_f32_tensor(device, source, output_norm_name)?,
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
    fn supports_all(source: &impl ResidentTensorSource, config: &LlamaConfig) -> bool {
        if config.block_count == 0 || config.is_hybrid() || config.is_gemma4() || config.is_moe() {
            return false;
        }
        let q_width = config.q_width();
        let kv_width = config.kv_width();
        let dim = config.embedding_length;
        let ff_dim = config.feed_forward_length;
        let head_dim = config.head_dim();

        for layer in 0..config.block_count {
            if !matches_optional_qk_norm_pair(source, layer, head_dim) {
                return false;
            }
            if !matches_f32_vector(source, &format!("blk.{layer}.attn_norm.weight"), dim)
                || !matches_f32_vector(source, &format!("blk.{layer}.ffn_norm.weight"), dim)
            {
                return false;
            }
            for (name, len) in [
                (format!("blk.{layer}.attn_q.bias"), q_width),
                (format!("blk.{layer}.attn_k.bias"), kv_width),
                (format!("blk.{layer}.attn_v.bias"), kv_width),
            ] {
                if !matches_optional_f32_vector(source, &name, len) {
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
                if !matches_supported_linear_shape(source, &name, rows, cols) {
                    return false;
                }
            }
        }
        true
    }

    fn try_load_all(
        device: &CudaDevice,
        source: &impl ResidentTensorSource,
        config: &LlamaConfig,
    ) -> Result<Option<Vec<Self>>> {
        if !Self::supports_all(source, config) {
            return Ok(None);
        }
        let dim = config.embedding_length;

        let mut layers = Vec::with_capacity(config.block_count);
        for layer in 0..config.block_count {
            layers.push(Self {
                attn_norm: upload_resident_f32_tensor(
                    device,
                    source,
                    &format!("blk.{layer}.attn_norm.weight"),
                )?,
                ffn_norm: upload_resident_f32_tensor(
                    device,
                    source,
                    &format!("blk.{layer}.ffn_norm.weight"),
                )?,
                attn_q: ResidentQuantMatrix::upload(
                    device,
                    source,
                    &format!("blk.{layer}.attn_q.weight"),
                )?,
                attn_k: ResidentQuantMatrix::upload(
                    device,
                    source,
                    &format!("blk.{layer}.attn_k.weight"),
                )?,
                attn_v: ResidentQuantMatrix::upload(
                    device,
                    source,
                    &format!("blk.{layer}.attn_v.weight"),
                )?,
                attn_q_norm: upload_optional_f32_tensor(
                    device,
                    source,
                    &format!("blk.{layer}.attn_q_norm.weight"),
                )?,
                attn_k_norm: upload_optional_f32_tensor(
                    device,
                    source,
                    &format!("blk.{layer}.attn_k_norm.weight"),
                )?,
                attn_q_bias: upload_optional_f32_tensor(
                    device,
                    source,
                    &format!("blk.{layer}.attn_q.bias"),
                )?,
                attn_k_bias: upload_optional_f32_tensor(
                    device,
                    source,
                    &format!("blk.{layer}.attn_k.bias"),
                )?,
                attn_v_bias: upload_optional_f32_tensor(
                    device,
                    source,
                    &format!("blk.{layer}.attn_v.bias"),
                )?,
                attn_output: ResidentQuantMatrix::upload(
                    device,
                    source,
                    &format!("blk.{layer}.attn_output.weight"),
                )?,
                ffn_gate: ResidentQuantMatrix::upload(
                    device,
                    source,
                    &format!("blk.{layer}.ffn_gate.weight"),
                )?,
                ffn_up: ResidentQuantMatrix::upload(
                    device,
                    source,
                    &format!("blk.{layer}.ffn_up.weight"),
                )?,
                ffn_down: ResidentQuantMatrix::upload(
                    device,
                    source,
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

struct ResidentGemma4LayerWeights {
    attn_norm: GpuF32Tensor,
    attn_q: ResidentQuantMatrix,
    attn_k: ResidentQuantMatrix,
    attn_v: Option<ResidentQuantMatrix>,
    attn_output: ResidentQuantMatrix,
    attn_q_norm: GpuF32Tensor,
    attn_k_norm: GpuF32Tensor,
    post_attention_norm: GpuF32Tensor,
    ffn_norm: GpuF32Tensor,
    ffn_gate: ResidentQuantMatrix,
    ffn_up: ResidentQuantMatrix,
    ffn_down: ResidentQuantMatrix,
    post_ffw_norm: GpuF32Tensor,
    layer_output_scale: Option<f32>,
    embedding_length: usize,
}

impl ResidentGemma4LayerWeights {
    fn supports_all(source: &impl ResidentTensorSource, config: &LlamaConfig) -> bool {
        if config.block_count == 0 || !config.is_gemma4() || config.is_hybrid() || config.is_moe() {
            return false;
        }

        let dim = config.embedding_length;
        let ff_dim = config.feed_forward_length;
        for layer in 0..config.block_count {
            let Some(layer_config) = config.gemma4_layer_config(layer) else {
                return false;
            };
            let head_dim = layer_config.head_dim();
            let q_width = layer_config.q_width();
            let kv_width = layer_config.kv_width();

            for (name, len) in [
                (format!("blk.{layer}.attn_norm.weight"), dim),
                (format!("blk.{layer}.attn_q_norm.weight"), head_dim),
                (format!("blk.{layer}.attn_k_norm.weight"), head_dim),
                (format!("blk.{layer}.post_attention_norm.weight"), dim),
                (format!("blk.{layer}.ffn_norm.weight"), dim),
                (format!("blk.{layer}.post_ffw_norm.weight"), dim),
            ] {
                if !matches_f32_vector(source, &name, len) {
                    return false;
                }
            }

            for (name, rows, cols) in [
                (format!("blk.{layer}.attn_q.weight"), q_width, dim),
                (format!("blk.{layer}.attn_k.weight"), kv_width, dim),
                (format!("blk.{layer}.attn_output.weight"), dim, q_width),
                (format!("blk.{layer}.ffn_gate.weight"), ff_dim, dim),
                (format!("blk.{layer}.ffn_up.weight"), ff_dim, dim),
                (format!("blk.{layer}.ffn_down.weight"), dim, ff_dim),
            ] {
                if !matches_supported_linear_shape(source, &name, rows, cols) {
                    return false;
                }
            }

            if !matches_optional_supported_linear_shape(
                source,
                &format!("blk.{layer}.attn_v.weight"),
                kv_width,
                dim,
            ) || !matches_optional_f32_vector(
                source,
                &format!("blk.{layer}.layer_output_scale.weight"),
                1,
            ) {
                return false;
            }
        }
        true
    }

    fn try_load_all(
        device: &CudaDevice,
        source: &impl ResidentTensorSource,
        config: &LlamaConfig,
    ) -> Result<Option<Vec<Self>>> {
        if !Self::supports_all(source, config) {
            return Ok(None);
        }

        let mut layers = Vec::with_capacity(config.block_count);
        for layer in 0..config.block_count {
            let v_name = format!("blk.{layer}.attn_v.weight");
            let scale_name = format!("blk.{layer}.layer_output_scale.weight");
            layers.push(Self {
                attn_norm: upload_resident_f32_tensor(
                    device,
                    source,
                    &format!("blk.{layer}.attn_norm.weight"),
                )?,
                attn_q: ResidentQuantMatrix::upload(
                    device,
                    source,
                    &format!("blk.{layer}.attn_q.weight"),
                )?,
                attn_k: ResidentQuantMatrix::upload(
                    device,
                    source,
                    &format!("blk.{layer}.attn_k.weight"),
                )?,
                attn_v: if source.tensor_info(&v_name).is_some() {
                    Some(ResidentQuantMatrix::upload(device, source, &v_name)?)
                } else {
                    None
                },
                attn_output: ResidentQuantMatrix::upload(
                    device,
                    source,
                    &format!("blk.{layer}.attn_output.weight"),
                )?,
                attn_q_norm: upload_resident_f32_tensor(
                    device,
                    source,
                    &format!("blk.{layer}.attn_q_norm.weight"),
                )?,
                attn_k_norm: upload_resident_f32_tensor(
                    device,
                    source,
                    &format!("blk.{layer}.attn_k_norm.weight"),
                )?,
                post_attention_norm: upload_resident_f32_tensor(
                    device,
                    source,
                    &format!("blk.{layer}.post_attention_norm.weight"),
                )?,
                ffn_norm: upload_resident_f32_tensor(
                    device,
                    source,
                    &format!("blk.{layer}.ffn_norm.weight"),
                )?,
                ffn_gate: ResidentQuantMatrix::upload(
                    device,
                    source,
                    &format!("blk.{layer}.ffn_gate.weight"),
                )?,
                ffn_up: ResidentQuantMatrix::upload(
                    device,
                    source,
                    &format!("blk.{layer}.ffn_up.weight"),
                )?,
                ffn_down: ResidentQuantMatrix::upload(
                    device,
                    source,
                    &format!("blk.{layer}.ffn_down.weight"),
                )?,
                post_ffw_norm: upload_resident_f32_tensor(
                    device,
                    source,
                    &format!("blk.{layer}.post_ffw_norm.weight"),
                )?,
                layer_output_scale: load_optional_resident_float_scalar(source, &scale_name)?,
                embedding_length: config.embedding_length,
            });
        }
        Ok(Some(layers))
    }
}

fn load_optional_resident_float_scalar(
    source: &impl ResidentTensorSource,
    name: &str,
) -> Result<Option<f32>> {
    let Some(info) = source.tensor_info(name) else {
        return Ok(None);
    };
    if !is_supported_resident_float_tensor(&info) || info.numel != 1 {
        return Err(XrtError::InvalidTensor(format!(
            "optional scalar tensor `{name}` must contain one F32/F16/BF16 value"
        )));
    }

    let bytes = source.tensor_data(name)?;
    let value = match info.dtype {
        DType::F32 => {
            let bytes: [u8; 4] = bytes.try_into().map_err(|_| {
                XrtError::InvalidTensor(format!(
                    "F32 scalar tensor `{name}` has {} bytes, expected 4",
                    bytes.len()
                ))
            })?;
            f32::from_le_bytes(bytes)
        }
        DType::F16 => decode_f16(bytes)?,
        DType::BF16 => decode_bf16(bytes)?,
        _ => unreachable!("scalar dtype was validated above"),
    };
    if !value.is_finite() {
        return Err(XrtError::InvalidTensor(format!(
            "optional scalar tensor `{name}` must be finite"
        )));
    }
    Ok(Some(value))
}

fn matches_f32_vector(source: &impl ResidentTensorSource, name: &str, len: usize) -> bool {
    source
        .tensor_info(name)
        .is_some_and(|info| is_supported_resident_float_tensor(&info) && info.numel == len)
}

fn matches_optional_f32_vector(source: &impl ResidentTensorSource, name: &str, len: usize) -> bool {
    match source.tensor_info(name) {
        Some(info) => is_supported_resident_float_tensor(&info) && info.numel == len,
        None => true,
    }
}

fn matches_optional_qk_norm_pair(
    source: &impl ResidentTensorSource,
    layer: usize,
    head_dim: usize,
) -> bool {
    let q_name = format!("blk.{layer}.attn_q_norm.weight");
    let k_name = format!("blk.{layer}.attn_k_norm.weight");
    match (source.tensor_info(&q_name), source.tensor_info(&k_name)) {
        (None, None) => true,
        (Some(q), Some(k)) => {
            is_supported_resident_float_tensor(&q)
                && is_supported_resident_float_tensor(&k)
                && q.numel == head_dim
                && k.numel == head_dim
        }
        _ => false,
    }
}

fn upload_optional_f32_tensor(
    device: &CudaDevice,
    source: &impl ResidentTensorSource,
    name: &str,
) -> Result<Option<GpuF32Tensor>> {
    if source.tensor_info(name).is_some() {
        upload_resident_f32_tensor(device, source, name).map(Some)
    } else {
        Ok(None)
    }
}

fn matches_supported_linear_shape(
    source: &impl ResidentTensorSource,
    name: &str,
    rows: usize,
    cols: usize,
) -> bool {
    source.tensor_info(name).is_some_and(|info| {
        is_supported_resident_linear_tensor(&info) && info.rows == rows && info.cols == cols
    })
}

fn matches_optional_supported_linear_shape(
    source: &impl ResidentTensorSource,
    name: &str,
    rows: usize,
    cols: usize,
) -> bool {
    match source.tensor_info(name) {
        Some(info) => {
            is_supported_resident_linear_tensor(&info) && info.rows == rows && info.cols == cols
        }
        None => true,
    }
}

fn is_supported_resident_linear_dtype(dtype: DType) -> bool {
    is_supported_resident_float_dtype(dtype)
        || matches!(
            dtype,
            DType::Q8_0 | DType::Q4_0 | DType::Q4_K | DType::Q5_K | DType::Q6_K
        )
}

fn is_supported_resident_linear_tensor(info: &ResidentTensorInfo) -> bool {
    match info.storage {
        ResidentTensorStorage::Dense => is_supported_resident_linear_dtype(info.dtype),
        ResidentTensorStorage::AwqGemm4 { .. }
        | ResidentTensorStorage::AwqGemv4 { .. }
        | ResidentTensorStorage::GptqGemm4 { .. }
        | ResidentTensorStorage::GptqExplicitGemm4 { .. }
        | ResidentTensorStorage::CompressedTensorsW4A16 { .. } => true,
    }
}

fn is_supported_resident_float_tensor(info: &ResidentTensorInfo) -> bool {
    info.storage == ResidentTensorStorage::Dense && is_supported_resident_float_dtype(info.dtype)
}

fn is_supported_resident_float_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::F16 | DType::BF16)
}

impl CausalLmBackend for CudaResidentBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::CudaResident
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn config(&self) -> &LlamaConfig {
        &self.config
    }

    fn new_session(&self, cache_mode: KvCacheMode, page_tokens: usize) -> BackendSession {
        let config = &self.config;
        let layer_widths = config
            .gemma4_layer_kv_widths()
            .unwrap_or_else(|| vec![config.kv_width(); config.block_count]);
        let mut session = BackendSession::new_cuda_with_kv_budget_page_tokens_and_layer_widths(
            self.device.clone(),
            cache_mode,
            layer_widths,
            config.context_length,
            page_tokens,
            Some(self.kv_budget_bytes),
        );
        session.configure_cuda_graph_mode(self.cuda_graph_mode);
        session
    }

    fn clear_state(&self) {
        if let Some(model) = &self.cpu_reference_model {
            model.clear_state();
        }
    }

    fn save_state(&self) -> Option<BackendStateSnapshot> {
        self.cpu_reference_model
            .as_ref()
            .and_then(|model| model.save_state())
    }

    fn restore_state(&self, snapshot: &[Option<(Vec<f32>, Vec<f32>)>]) {
        if let Some(model) = &self.cpu_reference_model {
            model.restore_state(snapshot);
        }
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

    fn supports_multi_sequence_decode_batch(&self) -> bool {
        let config = &self.config;
        self.cuda_graph_mode != CudaGraphMode::Disabled
            && !config.is_gemma4()
            && !config.is_hybrid()
            && self.q8_0_probe.is_some()
            && self
                .q8_0_layer_probes
                .as_ref()
                .is_some_and(|layers| layers.len() == config.block_count)
    }

    fn forward_token_batch(
        &self,
        batch: &mut [BackendDecodeBatchItem],
    ) -> Result<BackendDecodeBatchExecution> {
        batch.sort_by_key(|item| item.sequence_id);
        if let Some(execution) = self.try_concurrent_standard_dense_graph_decode(batch)? {
            return Ok(execution);
        }
        for item in batch {
            self.forward_token(
                item.token_id,
                item.position,
                &mut item.session,
                &mut item.output_logits,
            )?;
        }
        Ok(BackendDecodeBatchExecution { fused: false })
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
            false,
            None,
            cuda_total_len_for_position(position)?,
            Some(n_layers),
        )? {
            return Ok(());
        }
        Err(Self::decode_unsupported())
    }

    fn gemma4_layer0_trace(
        &self,
        token_id: u32,
        position: usize,
        session: &mut BackendSession,
    ) -> Result<Option<Gemma4LayerTrace>> {
        self.trace_gemma4_layer0(token_id, position, session)
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
                false,
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
            self.config.embedding_length,
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
                false,
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
        let vocab_size = self.config.vocab_size;
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
                false,
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
        self.require_cpu_reference_model()?
            .embedding_lookup(token_id)
    }

    fn model_weight_bytes(&self) -> u64 {
        self.resident_model_weight_bytes
    }

    fn cuda_device_name(&self) -> Option<&str> {
        self.device_name.as_deref()
    }

    fn cuda_free_vram_bytes(&self) -> Option<u64> {
        self.device.memory_info().ok().map(|(free, _)| free)
    }

    fn cuda_total_vram_bytes(&self) -> Option<u64> {
        self.device.memory_info().ok().map(|(_, total)| total)
    }

    fn cuda_memory_info(&self) -> Option<(u64, u64)> {
        self.device.memory_info().ok()
    }

    fn cuda_transfer_stats(&self) -> Option<CudaTransferStats> {
        Some(self.device.transfer_stats())
    }

    fn cuda_allocation_stats(&self) -> Option<CudaAllocationStats> {
        Some(self.device.allocation_stats())
    }

    fn cuda_memory_pool_stats(&self) -> Option<CudaMemoryPoolStats> {
        self.device.memory_pool_stats().ok().flatten()
    }

    fn reset_cuda_allocation_peak(&self) {
        self.device.reset_allocation_peak();
    }

    fn cuda_kv_budget_bytes(&self) -> Option<u64> {
        Some(self.kv_budget_bytes)
    }

    fn supports_cuda_graph_decode(&self) -> bool {
        let config = &self.config;
        !config.is_gemma4()
            && !config.is_hybrid()
            && self.q8_0_probe.is_some()
            && self
                .q8_0_layer_probes
                .as_ref()
                .is_some_and(|layers| layers.len() == config.block_count)
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
        let loaded_layer_count = if self.config.is_gemma4() {
            self.gemma4_layer_probes.as_ref().map(Vec::len)
        } else {
            self.q8_0_layer_probes.as_ref().map(Vec::len)
        };
        dense_quant_decode_status_available(
            true,
            probe.token_embedding_gpu_resident(),
            loaded_layer_count,
            self.config.block_count,
        )
    }
}

fn cuda_all_logits_output_len(token_count: usize, vocab_size: usize) -> Result<usize> {
    checked_mul(token_count, vocab_size, "CUDA all-logits batch output")
}

fn cuda_estimated_resident_upload_bytes(
    source: &impl ResidentTensorSource,
    config: &LlamaConfig,
) -> Result<u64> {
    if !CudaResidentBackend::supports_dense_quant_decode_source(source, config) {
        return Err(CudaResidentBackend::decode_unsupported());
    }
    let output_name = ResidentQ8_0ProbeWeights::output_name(source);
    source.tensor_infos().iter().try_fold(0u64, |total, info| {
        total
            .checked_add(cuda_extra_resident_tensor_bytes(info, output_name)?)
            .ok_or_else(|| {
                XrtError::Runtime("CUDA resident upload byte count overflow".to_string())
            })
    })
}

fn cuda_extra_resident_tensor_bytes(info: &ResidentTensorInfo, output_name: &str) -> Result<u64> {
    if info.name == "token_embd.weight" {
        let embedding_bytes = cuda_embedding_resident_tensor_bytes(info)?;
        let tied_output_bytes =
            if output_name == "token_embd.weight" && is_supported_resident_float_tensor(info) {
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

fn cuda_embedding_resident_tensor_bytes(info: &ResidentTensorInfo) -> Result<u64> {
    if info.storage != ResidentTensorStorage::Dense {
        return Err(XrtError::Unsupported(format!(
            "CUDA token embedding `{}` cannot use grouped packed storage",
            info.name
        )));
    }
    match info.dtype {
        DType::Q4_K | DType::Q6_K => match cuda_k_quant_embedding_layout(info)? {
            CudaKQuantEmbeddingLayout::ExpandedF32 => cuda_expanded_embedding_bytes(info),
            CudaKQuantEmbeddingLayout::Packed => cuda_packed_embedding_bytes(info),
        },
        DType::Q5_K => cuda_expanded_embedding_bytes(info),
        _ => cuda_matrix_resident_tensor_bytes(info),
    }
}

fn cuda_expanded_embedding_bytes(info: &ResidentTensorInfo) -> Result<u64> {
    let bytes = cuda_resident_f32_tensor_bytes(info)?;
    bytes.checked_mul(2).ok_or_else(|| {
        XrtError::Runtime("CUDA resident K-quant embedding byte count overflow".to_string())
    })
}

fn cuda_k_quant_embedding_layout(info: &ResidentTensorInfo) -> Result<CudaKQuantEmbeddingLayout> {
    if !matches!(info.dtype, DType::Q4_K | DType::Q6_K) {
        return Err(XrtError::InvalidTensor(format!(
            "CUDA packed embedding layout requires Q4_K or Q6_K dtype, tensor `{}` is {:?}",
            info.name, info.dtype
        )));
    }
    if cuda_expanded_embedding_bytes(info)? <= CUDA_K_QUANT_EXPANDED_EMBEDDING_MAX_BYTES {
        Ok(CudaKQuantEmbeddingLayout::ExpandedF32)
    } else {
        Ok(CudaKQuantEmbeddingLayout::Packed)
    }
}

fn cuda_packed_embedding_bytes(info: &ResidentTensorInfo) -> Result<u64> {
    match info.dtype {
        DType::Q4_K => cuda_matrix_resident_tensor_bytes(info),
        DType::Q6_K => cuda_quant_block_count(info).and_then(|blocks| {
            blocks
                .checked_mul((4 + DType::Q6_K.block_bytes()) as u64)
                .ok_or_else(|| {
                    XrtError::Runtime("CUDA resident packed Q6_K byte count overflow".to_string())
                })
        }),
        _ => Err(XrtError::InvalidTensor(format!(
            "CUDA packed embedding bytes require Q4_K or Q6_K dtype, tensor `{}` is {:?}",
            info.name, info.dtype
        ))),
    }
}

fn cuda_resident_f32_tensor_bytes(info: &ResidentTensorInfo) -> Result<u64> {
    checked_mul(info.numel, 4, "CUDA resident F32 tensor bytes").map(|v| v as u64)
}

fn cuda_quant_block_count(info: &ResidentTensorInfo) -> Result<u64> {
    if info.cols % info.dtype.block_size() != 0 {
        return Err(XrtError::InvalidTensor(format!(
            "tensor `{}` row length {} is not divisible by {:?} block size {}",
            info.name,
            info.cols,
            info.dtype,
            info.dtype.block_size()
        )));
    }
    let blocks_per_row = info.cols / info.dtype.block_size();
    checked_mul(info.rows, blocks_per_row, "CUDA resident quant block count").map(|v| v as u64)
}

fn cuda_matrix_resident_tensor_bytes(info: &ResidentTensorInfo) -> Result<u64> {
    match info.storage {
        ResidentTensorStorage::AwqGemm4 { group_size } => {
            return cuda_grouped_gemm4_resident_tensor_bytes(info, group_size, "AWQ")
        }
        ResidentTensorStorage::AwqGemv4 {
            group_size,
            zero_words_per_row,
        } => return cuda_awq_gemv4_resident_tensor_bytes(info, group_size, zero_words_per_row),
        ResidentTensorStorage::GptqGemm4 { group_size } => {
            return cuda_grouped_gemm4_resident_tensor_bytes(info, group_size, "GPTQ")
        }
        ResidentTensorStorage::GptqExplicitGemm4 { group_size, .. } => {
            let packed_bytes =
                cuda_grouped_gemm4_resident_tensor_bytes(info, group_size, "explicit-group GPTQ")?;
            let group_index_bytes =
                checked_mul(info.cols, 4, "CUDA explicit-group GPTQ group index bytes")? as u64;
            return packed_bytes.checked_add(group_index_bytes).ok_or_else(|| {
                XrtError::Runtime(
                    "CUDA explicit-group GPTQ resident byte count overflow".to_string(),
                )
            });
        }
        ResidentTensorStorage::CompressedTensorsW4A16 { group_size } => {
            return cuda_compressed_tensors_w4a16_resident_tensor_bytes(info, group_size)
        }
        ResidentTensorStorage::Dense => {}
    }

    let f32_bytes =
        || checked_mul(info.numel, 4, "CUDA resident F32 tensor bytes").map(|v| v as u64);
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

fn cuda_grouped_gemm4_resident_tensor_bytes(
    info: &ResidentTensorInfo,
    group_size: usize,
    format_name: &str,
) -> Result<u64> {
    if info.rows == 0
        || info.cols == 0
        || info.rows % 8 != 0
        || group_size == 0
        || info.cols % group_size != 0
    {
        return Err(XrtError::InvalidTensor(format!(
            "CUDA {format_name} matrix `{}` has incompatible rows={}, cols={}, group_size={group_size}",
            info.name, info.rows, info.cols
        )));
    }
    let packed_rows = info.rows / 8;
    let groups = info.cols / group_size;
    let qweight_bytes = checked_mul(
        checked_mul(
            info.cols,
            packed_rows,
            &format!("CUDA {format_name} qweight words"),
        )?,
        4,
        &format!("CUDA {format_name} qweight bytes"),
    )?;
    let qzero_bytes = checked_mul(
        checked_mul(
            groups,
            packed_rows,
            &format!("CUDA {format_name} qzero words"),
        )?,
        4,
        &format!("CUDA {format_name} qzero bytes"),
    )?;
    let scale_bytes = checked_mul(
        checked_mul(
            groups,
            info.rows,
            &format!("CUDA {format_name} scale count"),
        )?,
        4,
        &format!("CUDA {format_name} scale bytes"),
    )?;
    qweight_bytes
        .checked_add(qzero_bytes)
        .and_then(|bytes| bytes.checked_add(scale_bytes))
        .map(|bytes| bytes as u64)
        .ok_or_else(|| {
            XrtError::Runtime(format!("CUDA {format_name} resident byte count overflow"))
        })
}

fn cuda_awq_gemv4_resident_tensor_bytes(
    info: &ResidentTensorInfo,
    group_size: usize,
    zero_words_per_row: usize,
) -> Result<u64> {
    if info.rows == 0
        || info.cols == 0
        || info.rows % 8 != 0
        || info.cols % 8 != 0
        || group_size == 0
        || info.cols % group_size != 0
        || zero_words_per_row == 0
    {
        return Err(XrtError::InvalidTensor(format!(
            "CUDA AWQ GEMV matrix `{}` has incompatible rows={}, cols={}, group_size={group_size}, zero_words_per_row={zero_words_per_row}",
            info.name, info.rows, info.cols
        )));
    }
    let qweight_bytes = checked_mul(
        checked_mul(info.rows, info.cols / 8, "CUDA AWQ GEMV qweight words")?,
        4,
        "CUDA AWQ GEMV qweight bytes",
    )?;
    let qzero_bytes = checked_mul(
        checked_mul(info.rows, zero_words_per_row, "CUDA AWQ GEMV qzero words")?,
        4,
        "CUDA AWQ GEMV qzero bytes",
    )?;
    let scale_stride = checked_mul(zero_words_per_row, 8, "CUDA AWQ GEMV padded scale stride")?;
    let scale_bytes = checked_mul(
        checked_mul(info.rows, scale_stride, "CUDA AWQ GEMV scale count")?,
        4,
        "CUDA AWQ GEMV scale bytes",
    )?;
    qweight_bytes
        .checked_add(qzero_bytes)
        .and_then(|bytes| bytes.checked_add(scale_bytes))
        .map(|bytes| bytes as u64)
        .ok_or_else(|| XrtError::Runtime("CUDA AWQ GEMV resident byte count overflow".to_string()))
}

fn cuda_compressed_tensors_w4a16_resident_tensor_bytes(
    info: &ResidentTensorInfo,
    group_size: usize,
) -> Result<u64> {
    if info.rows == 0
        || info.cols == 0
        || info.cols % 8 != 0
        || group_size == 0
        || info.cols % group_size != 0
    {
        return Err(XrtError::InvalidTensor(format!(
            "CUDA compressed-tensors W4A16 matrix `{}` has incompatible rows={}, cols={}, group_size={group_size}",
            info.name, info.rows, info.cols
        )));
    }
    let packed_weight_bytes = checked_mul(
        checked_mul(
            info.rows,
            info.cols / 8,
            "CUDA compressed-tensors packed weight words",
        )?,
        4,
        "CUDA compressed-tensors packed weight bytes",
    )?;
    let scale_bytes = checked_mul(
        checked_mul(
            info.rows,
            info.cols / group_size,
            "CUDA compressed-tensors scale count",
        )?,
        4,
        "CUDA compressed-tensors scale bytes",
    )?;
    let group_index_bytes = checked_mul(info.cols, 4, "CUDA compressed-tensors group index bytes")?;
    packed_weight_bytes
        .checked_add(scale_bytes)
        .and_then(|bytes| bytes.checked_add(group_index_bytes))
        .map(|bytes| bytes as u64)
        .ok_or_else(|| {
            XrtError::Runtime(
                "CUDA compressed-tensors W4A16 resident byte count overflow".to_string(),
            )
        })
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

fn cuda_layer_kv_allocated_bytes(
    mode: KvCacheMode,
    capacity: usize,
    width: usize,
    page_tokens: usize,
) -> Result<u64> {
    let elements = checked_mul(capacity, width, "CUDA KV cache elements")?;
    let storage_bytes = match mode {
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
                checked_mul(capacity, width.div_ceil(64), "CUDA KQ4/VQ8 key scale count")?;
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
    }?;
    let page_count = capacity.div_ceil(page_tokens.max(1));
    let table_count = if mode == KvCacheMode::AgentAdaptive {
        2
    } else {
        1
    };
    let page_table_bytes = checked_mul(
        checked_mul(page_count, table_count, "CUDA KV page table count")?,
        std::mem::size_of::<u32>(),
        "CUDA KV page table bytes",
    )? as u64;
    let route_bytes = if mode == KvCacheMode::AgentAdaptive {
        checked_mul(
            capacity,
            std::mem::size_of::<u32>(),
            "CUDA adaptive KV route bytes",
        )? as u64
    } else {
        0
    };
    storage_bytes
        .checked_add(page_table_bytes)
        .and_then(|bytes| bytes.checked_add(route_bytes))
        .ok_or_else(|| XrtError::Runtime("CUDA KV page-table byte count overflow".to_string()))
}

fn cuda_session_kv_allocated_bytes(
    mode: KvCacheMode,
    layer_count: usize,
    capacity: usize,
    width: usize,
    page_tokens: usize,
) -> Result<u64> {
    let layer_bytes = cuda_layer_kv_allocated_bytes(mode, capacity, width, page_tokens)?;
    (layer_count as u64)
        .checked_mul(layer_bytes)
        .ok_or_else(|| XrtError::Runtime("CUDA session KV cache byte count overflow".to_string()))
}

fn cuda_session_kv_allocated_bytes_for_widths(
    mode: KvCacheMode,
    layer_widths: &[usize],
    capacity: usize,
    page_tokens: usize,
) -> Result<u64> {
    layer_widths.iter().try_fold(0u64, |total, &width| {
        let layer_bytes = cuda_layer_kv_allocated_bytes(mode, capacity, width, page_tokens)?;
        total.checked_add(layer_bytes).ok_or_else(|| {
            XrtError::Runtime("CUDA variable-width KV cache byte count overflow".to_string())
        })
    })
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

    fn resident_tensor_info(
        name: &str,
        dimensions: Vec<usize>,
        dtype: DType,
    ) -> ResidentTensorInfo {
        let rank = dimensions.len();
        let rows = if rank <= 1 {
            1
        } else {
            dimensions[1..].iter().copied().product()
        };
        let cols = dimensions.first().copied().unwrap_or_default();
        let numel = dimensions.iter().copied().product();
        ResidentTensorInfo {
            name: name.to_string(),
            dimensions,
            dtype,
            rank,
            rows,
            cols,
            numel,
            byte_len: 0,
            storage: ResidentTensorStorage::Dense,
        }
    }

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
        assert!(message.contains("Gemma4"), "missing Gemma4 in {message}");
        assert!(
            message.contains("AutoAWQ GEMM/GEMV"),
            "missing AutoAWQ in {message}"
        );
        assert!(
            message.contains("GPTQ v1/v2 GEMM4"),
            "missing GPTQ in {message}"
        );
        assert!(
            message.contains("compressed-tensors W4A16"),
            "missing compressed-tensors in {message}"
        );
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
            ..GpuResourceConfig::default()
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
            ..GpuResourceConfig::default()
        };
        assert_eq!(cuda_kv_budget_bytes(1000, 200, config), 200);
        assert_eq!(cuda_kv_budget_bytes(1000, 1200, config), 0);
    }

    #[test]
    fn cuda_session_kv_byte_estimate_matches_cache_modes() {
        assert_eq!(
            cuda_session_kv_allocated_bytes(KvCacheMode::F32, 2, 8, 4, 4).unwrap(),
            528
        );
        assert_eq!(
            cuda_session_kv_allocated_bytes(KvCacheMode::Q8, 2, 8, 4, 4).unwrap(),
            272
        );
        assert_eq!(
            cuda_session_kv_allocated_bytes(KvCacheMode::KeyQ4ValueQ8, 2, 8, 64, 4).unwrap(),
            1680
        );
        assert_eq!(
            cuda_session_kv_allocated_bytes(KvCacheMode::AgentAdaptive, 2, 8, 64, 4).unwrap(),
            9952
        );
        assert_eq!(
            cuda_session_kv_allocated_bytes_for_widths(KvCacheMode::F32, &[4, 8], 8, 4).unwrap(),
            784
        );
        assert!(
            cuda_session_kv_allocated_bytes_for_widths(KvCacheMode::F32, &[4, 8], 8, 4).unwrap()
                < cuda_session_kv_allocated_bytes(KvCacheMode::F32, 2, 8, 8, 4).unwrap()
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
        let q8 = resident_tensor_info("blk.0.attn_q.weight", vec![64, 2], DType::Q8_0);
        assert_eq!(
            cuda_extra_resident_tensor_bytes(&q8, "output.weight").unwrap(),
            4 * 36
        );

        let q4_embedding = resident_tensor_info("token_embd.weight", vec![256, 3], DType::Q4_K);
        assert_eq!(
            cuda_extra_resident_tensor_bytes(&q4_embedding, "output.weight").unwrap(),
            256 * 3 * 4 * 2
        );
        assert_eq!(
            cuda_extra_resident_tensor_bytes(&q4_embedding, "token_embd.weight").unwrap(),
            256 * 3 * 4 * 2
        );

        let q6_embedding = resident_tensor_info("token_embd.weight", vec![256, 3], DType::Q6_K);
        assert_eq!(
            cuda_extra_resident_tensor_bytes(&q6_embedding, "output.weight").unwrap(),
            256 * 3 * 4 * 2
        );
        assert_eq!(
            cuda_extra_resident_tensor_bytes(&q6_embedding, "token_embd.weight").unwrap(),
            256 * 3 * 4 * 2
        );

        let mut awq = resident_tensor_info("blk.0.ffn_down.weight", vec![64, 16], DType::F16);
        awq.storage = ResidentTensorStorage::AwqGemm4 { group_size: 32 };
        assert!(is_supported_resident_linear_tensor(&awq));
        assert!(!is_supported_resident_float_tensor(&awq));
        assert_eq!(cuda_matrix_resident_tensor_bytes(&awq).unwrap(), 656);

        let mut awq_gemv = resident_tensor_info("blk.0.ffn_down.weight", vec![64, 16], DType::F16);
        awq_gemv.storage = ResidentTensorStorage::AwqGemv4 {
            group_size: 32,
            zero_words_per_row: 4,
        };
        assert!(is_supported_resident_linear_tensor(&awq_gemv));
        assert!(!is_supported_resident_float_tensor(&awq_gemv));
        assert_eq!(
            cuda_matrix_resident_tensor_bytes(&awq_gemv).unwrap(),
            16 * 8 * 4 + 16 * 4 * 4 + 16 * 32 * 4
        );

        let mut gptq = resident_tensor_info("blk.0.ffn_down.weight", vec![64, 16], DType::F16);
        gptq.storage = ResidentTensorStorage::GptqGemm4 { group_size: 32 };
        assert!(is_supported_resident_linear_tensor(&gptq));
        assert!(!is_supported_resident_float_tensor(&gptq));
        assert_eq!(cuda_matrix_resident_tensor_bytes(&gptq).unwrap(), 656);

        let mut explicit_gptq =
            resident_tensor_info("blk.0.ffn_down.weight", vec![64, 16], DType::F16);
        explicit_gptq.storage = ResidentTensorStorage::GptqExplicitGemm4 {
            group_size: 32,
            zero_encoding: xrt_cuda::GptqZeroEncoding::V2Direct,
        };
        assert!(is_supported_resident_linear_tensor(&explicit_gptq));
        assert!(!is_supported_resident_float_tensor(&explicit_gptq));
        assert_eq!(
            cuda_matrix_resident_tensor_bytes(&explicit_gptq).unwrap(),
            656 + 64 * 4
        );

        let mut compressed =
            resident_tensor_info("blk.0.ffn_down.weight", vec![64, 16], DType::BF16);
        compressed.storage = ResidentTensorStorage::CompressedTensorsW4A16 { group_size: 32 };
        assert!(is_supported_resident_linear_tensor(&compressed));
        assert!(!is_supported_resident_float_tensor(&compressed));
        assert_eq!(
            cuda_matrix_resident_tensor_bytes(&compressed).unwrap(),
            16 * 8 * 4 + 16 * 2 * 4 + 64 * 4
        );
    }

    #[test]
    fn cuda_k_quant_embedding_layout_caps_expanded_residency() {
        let rows_at_limit = (CUDA_K_QUANT_EXPANDED_EMBEDDING_MAX_BYTES / (256 * 4 * 2)) as usize;
        let at_limit =
            resident_tensor_info("token_embd.weight", vec![256, rows_at_limit], DType::Q4_K);
        assert_eq!(
            cuda_k_quant_embedding_layout(&at_limit).unwrap(),
            CudaKQuantEmbeddingLayout::ExpandedF32
        );
        assert_eq!(
            cuda_embedding_resident_tensor_bytes(&at_limit).unwrap(),
            CUDA_K_QUANT_EXPANDED_EMBEDDING_MAX_BYTES
        );

        let above_limit = resident_tensor_info(
            "token_embd.weight",
            vec![256, rows_at_limit + 1],
            DType::Q4_K,
        );
        assert_eq!(
            cuda_k_quant_embedding_layout(&above_limit).unwrap(),
            CudaKQuantEmbeddingLayout::Packed
        );
        assert_eq!(
            cuda_embedding_resident_tensor_bytes(&above_limit).unwrap(),
            (rows_at_limit as u64 + 1) * (4 + 4 + 12 + 128)
        );

        let q6_above_limit = resident_tensor_info(
            "token_embd.weight",
            vec![256, rows_at_limit + 1],
            DType::Q6_K,
        );
        assert_eq!(
            cuda_k_quant_embedding_layout(&q6_above_limit).unwrap(),
            CudaKQuantEmbeddingLayout::Packed
        );
        assert_eq!(
            cuda_embedding_resident_tensor_bytes(&q6_above_limit).unwrap(),
            (rows_at_limit as u64 + 1) * (4 + DType::Q6_K.block_bytes() as u64)
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
    fn shared_f32_projected_bytes_count_live_pages_and_stable_tables() {
        assert_eq!(
            BackendSession::projected_shared_f32_bytes(&[4, 8], 3, 8, 2).unwrap(),
            512
        );
        assert_eq!(
            BackendSession::projected_shared_f32_bytes(&[4, 8], 0, 8, 2).unwrap(),
            128
        );
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
        session.truncate(0).unwrap();
        assert_eq!(session.cuda_kv_allocated_bytes(), 0);

        let err = session.cuda_layer_cache_mut(0).unwrap_err();
        assert!(matches!(
            err,
            XrtError::Runtime(message) if message.contains("missing CUDA KV cache for layer 0")
        ));
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn session_kv_reservation_estimate_covers_growth_peak() {
        let cuda = BackendSession::new_cuda(CudaDevice, KvCacheMode::F32, 2, 4, 8);
        let final_bytes = cuda_session_kv_allocated_bytes(KvCacheMode::F32, 2, 8, 4, 8).unwrap();
        assert_eq!(
            cuda.kv_reservation_bytes_for_total_len(1).unwrap(),
            final_bytes * 2
        );
        assert!(cuda.kv_reservation_bytes_for_total_len(9).is_err());

        let cpu = BackendSession::new_cpu(KvCacheMode::F32, 2, 4, 8);
        assert_eq!(cpu.kv_reservation_bytes_for_total_len(8).unwrap(), 0);
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

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA-capable device and driver"]
    fn cuda_runtime_shared_f32_prefix_attachment_copies_only_touched_page() -> Result<()> {
        let device = CudaDevice::new(0)?;
        let layer_widths = vec![128, 128];
        let mut source = BackendSession::new_cuda_with_kv_budget_page_tokens_and_layer_widths(
            device.clone(),
            KvCacheMode::F32,
            layer_widths.clone(),
            8,
            2,
            Some(64 * 1024 * 1024),
        );
        source.configure_cuda_graph_mode(CudaGraphMode::Enabled);
        source.prepare_for_total_len(3)?;

        let mut expected_last_rows = Vec::new();
        for layer in 0..layer_widths.len() {
            for position in 0..3 {
                let key = (0..layer_widths[layer])
                    .map(|index| layer as f32 * 100.0 + position as f32 * 10.0 + index as f32)
                    .collect::<Vec<_>>();
                let value = key.iter().map(|value| -*value).collect::<Vec<_>>();
                if position == 2 {
                    expected_last_rows.push((key.clone(), value.clone()));
                }
                let key = device.upload_f32(&key)?;
                let value = device.upload_f32(&value)?;
                let cache = source.cuda_layer_cache_mut(layer)?;
                let CudaLayerKvStore::F32(cache) = cache else {
                    return Err(XrtError::Runtime(
                        "source CUDA prefix cache unexpectedly used shared storage".to_string(),
                    ));
                };
                device.append_layer_kv(cache, &key, &value)?;
            }
        }

        let snapshot = source.snapshot_prefix(3)?.ok_or_else(|| {
            XrtError::Runtime("CUDA F32 prefix snapshot was unavailable".to_string())
        })?;
        let snapshot_caches = match &snapshot {
            BackendPrefixSnapshot::Cuda {
                layer_caches,
                prefix_len,
                ..
            } => {
                assert_eq!(*prefix_len, 3);
                layer_caches.clone()
            }
            BackendPrefixSnapshot::Cpu { .. } => {
                return Err(XrtError::Runtime(
                    "CUDA session produced a CPU prefix snapshot".to_string(),
                ));
            }
        };
        match &source {
            BackendSession::Cuda {
                layer_caches,
                pending_prefix,
                ..
            } => {
                assert!(pending_prefix.is_none());
                assert!(layer_caches
                    .iter()
                    .all(|cache| matches!(cache, CudaLayerKvStore::F32(_))));
            }
            BackendSession::Cpu { .. } => {
                return Err(XrtError::Runtime(
                    "source session changed backend while snapshotting".to_string(),
                ));
            }
        }
        assert!(snapshot_caches
            .iter()
            .all(|cache| matches!(cache, CudaLayerKvStore::SharedF32(_))));

        let mut attached = BackendSession::new_cuda_with_kv_budget_page_tokens_and_layer_widths(
            device.clone(),
            KvCacheMode::F32,
            layer_widths.clone(),
            8,
            2,
            Some(64 * 1024 * 1024),
        );
        attached.configure_cuda_graph_mode(CudaGraphMode::Enabled);
        assert_eq!(attached.attach_prefix_snapshot(&snapshot)?, 3);
        assert!(!attached.cuda_graph_decode_ready());
        assert_eq!(attached.cuda_graph_capture_status(), Some("eager-fallback"));
        attached.prepare_for_total_len(4)?;

        for layer in 0..layer_widths.len() {
            let key = vec![1000.0 + layer as f32; layer_widths[layer]];
            let value = vec![-1000.0 - layer as f32; layer_widths[layer]];
            let key_device = device.upload_f32(&key)?;
            let value_device = device.upload_f32(&value)?;
            let cache = attached.cuda_layer_cache_mut(layer)?;
            let CudaLayerKvStore::SharedF32(cache) = cache else {
                return Err(XrtError::Runtime(
                    "attached CUDA prefix did not materialize shared F32 storage".to_string(),
                ));
            };
            assert_eq!(cache.shared_page_count(), 2);
            cache.append(&key_device, &value_device)?;
            assert_eq!(cache.len(), 4);
            assert_eq!(cache.shared_page_count(), 1);
            assert_eq!(cache.row(2)?, expected_last_rows[layer]);
            assert_eq!(cache.row(3)?, (key, value));
        }

        for (layer, cache) in snapshot_caches.iter().enumerate() {
            let CudaLayerKvStore::SharedF32(cache) = cache else {
                return Err(XrtError::Runtime(
                    "immutable CUDA prefix lost shared F32 storage".to_string(),
                ));
            };
            assert_eq!(cache.len(), 3);
            assert_eq!(cache.row(2)?, expected_last_rows[layer]);
        }
        Ok(())
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
