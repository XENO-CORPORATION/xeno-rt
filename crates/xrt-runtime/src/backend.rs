use std::{
    collections::{BTreeSet, HashMap, VecDeque},
    env, fmt,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};

use crate::{
    expert_placement::{
        AdaptivePlacementDecision, AdaptivePlacementMove, AdaptivePlacementTracker,
        ExpertPlacementSnapshot,
    },
    gpu_resource::{
        CudaGraphMode, GpuAllocationArena, GpuAllocationClass, GpuAllocationLease,
        GpuResourceConfig, GpuResourceManager,
    },
    kv_cache::{KvCacheMode, SessionKvCache},
    moe::{build_moe_execution_plan, HeterogeneousMoeCoordinator, MoeWorkItem},
    moe_config::{MoeAcceleration, MoePlacementPolicy, MoeRuntimeConfig},
    moe_manifest::{
        load_moe_placement_manifest, moe_config_sha256, sha256_file, MoePlacementManifestContext,
    },
    policy::{PromptSpan, SessionPolicy},
    recurrent_state::CudaDeltaNetState,
    resident_tensor::{
        GgufPackedExpertTensorSource, GgufResidentTensorSource,
        HfStandardDenseResidentTensorSource, ResidentTensorInfo, ResidentTensorSource,
        ResidentTensorStorage,
    },
};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use tracing::info;
use xrt_core::{checked_mul, decode_bf16, decode_f16, DType, KvCache, Result, XrtError};
use xrt_cuda::{
    CudaAdaptiveKvRoutes, CudaAllocationStats, CudaAwqGemm4Matrix, CudaAwqGemv4Matrix,
    CudaCompressedTensorsW4A16Matrix, CudaDecodeParams, CudaDevice, CudaExecutionStream,
    CudaF32Buffer, CudaF32KvPagePool, CudaGptqExplicitGemm4Matrix, CudaGptqGemm4Matrix,
    CudaGraphExec, CudaKeyQ4ValueQ8LayerKvCache, CudaKq4Vq8KvPagePool, CudaLayerKvCache,
    CudaMemoryPoolStats, CudaPinnedF32Buffer, CudaPinnedF32Download, CudaQ4KMatrix, CudaQ4_0Matrix,
    CudaQ5KMatrix, CudaQ6KMatrix, CudaQ8KvPagePool, CudaQ8LayerKvCache, CudaQ8_0Matrix,
    CudaSharedAdaptiveGraphBinding, CudaSharedAdaptiveLayerKvCache, CudaSharedF32GraphBinding,
    CudaSharedF32LayerKvCache, CudaSharedKq4Vq8GraphBinding, CudaSharedKq4Vq8LayerKvCache,
    CudaSharedQ8GraphBinding, CudaSharedQ8LayerKvCache, CudaTransferStats, GpuF32Tensor,
};
use xrt_gguf::GgufFile;
#[cfg(feature = "moe-route-trace")]
use xrt_models::MoeRouteTrace;
use xrt_models::{
    DeltaNetState, DeltaNetStateDescriptor, DeltaNetStateSnapshot, Gemma4LayerTrace, LlamaConfig,
    LlamaModel, MoeLayerDescriptor, MoeRoutingRow, MAX_SELECTED_EXPERTS,
};

use xrt_safetensors::HfModelBundle;

// Keep the faster expanded path for smaller vocabularies without allowing its
// two F32 copies and upload temporaries to exhaust host memory on large models.
const CUDA_K_QUANT_EXPANDED_EMBEDDING_MAX_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const CUDA_DECODE_BATCH_GRAPH_CACHE_ENTRIES: usize = 8;
const CUDA_SHARED_KV_MAX_REPLICAS: usize = 64;
const CUDA_MOE_EXPERT_GRAPH_CACHE_ENTRIES: usize = 64;
const ADAPTIVE_MOE_MAX_MOVES_PER_UPDATE: usize = 4;
const ADAPTIVE_MOE_MIN_RESIDENCY_EPOCHS: u64 = 1;
const ADAPTIVE_MOE_HYSTERESIS_PERCENT: u64 = 10;

fn qwen3_moe_uses_cpu_order_q4_k_matvec(architecture: &str) -> bool {
    matches!(architecture, "qwen3moe" | "qwen3_moe")
}

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

pub type BackendStateSnapshot = DeltaNetStateSnapshot;

#[derive(Debug)]
pub enum SessionRecurrentState {
    None,
    Uninitialized(DeltaNetStateDescriptor),
    Cpu(DeltaNetState),
    Cuda(CudaDeltaNetState),
    Poisoned {
        descriptor: Option<DeltaNetStateDescriptor>,
        reason: String,
    },
}

impl SessionRecurrentState {
    fn from_descriptor(descriptor: Option<DeltaNetStateDescriptor>) -> Self {
        descriptor.map_or(Self::None, Self::Uninitialized)
    }

    fn prepare_cpu(&mut self) -> Result<()> {
        match self {
            Self::None | Self::Cpu(_) => Ok(()),
            Self::Uninitialized(descriptor) => {
                let state = DeltaNetState::try_new(descriptor.clone())?;
                *self = Self::Cpu(state);
                Ok(())
            }
            Self::Cuda(_) => Err(XrtError::Runtime(
                "CUDA recurrent state cannot be prepared by the CPU backend".to_string(),
            )),
            Self::Poisoned { reason, .. } => Err(XrtError::Runtime(format!(
                "session recurrent state is poisoned and must be reset: {reason}"
            ))),
        }
    }

    fn cpu_mut(&mut self) -> Result<Option<&mut DeltaNetState>> {
        match self {
            Self::None => Ok(None),
            Self::Cpu(state) => Ok(Some(state)),
            Self::Cuda(_) => Err(XrtError::Runtime(
                "CUDA recurrent state requested by the CPU backend".to_string(),
            )),
            Self::Uninitialized(_) => Err(XrtError::Runtime(
                "session recurrent state was not prepared before forward".to_string(),
            )),
            Self::Poisoned { reason, .. } => Err(XrtError::Runtime(format!(
                "session recurrent state is poisoned and must be reset: {reason}"
            ))),
        }
    }

    fn prepare_cuda(
        &mut self,
        device: &CudaDevice,
        allocation_arena: Option<&Arc<GpuAllocationArena>>,
    ) -> Result<()> {
        match self {
            Self::None => Ok(()),
            Self::Uninitialized(descriptor) => {
                let state = CudaDeltaNetState::try_new(
                    device.clone(),
                    descriptor.clone(),
                    allocation_arena,
                )?;
                *self = Self::Cuda(state);
                Ok(())
            }
            Self::Cuda(state) => state.prepare(),
            Self::Cpu(_) => Err(XrtError::Runtime(
                "CPU recurrent state cannot be prepared by the CUDA backend".to_string(),
            )),
            Self::Poisoned { reason, .. } => Err(XrtError::Runtime(format!(
                "session recurrent state is poisoned and must be reset: {reason}"
            ))),
        }
    }

    fn cuda_mut(&mut self) -> Result<Option<&mut CudaDeltaNetState>> {
        match self {
            Self::None => Ok(None),
            Self::Cuda(state) => Ok(Some(state)),
            Self::Uninitialized(_) => Err(XrtError::Runtime(
                "session recurrent state was not prepared before CUDA forward".to_string(),
            )),
            Self::Cpu(_) => Err(XrtError::Runtime(
                "CPU recurrent state requested by the CUDA backend".to_string(),
            )),
            Self::Poisoned { reason, .. } => Err(XrtError::Runtime(format!(
                "session recurrent state is poisoned and must be reset: {reason}"
            ))),
        }
    }

    fn snapshot(&self) -> Result<Option<BackendStateSnapshot>> {
        match self {
            Self::None => Ok(None),
            Self::Cpu(state) => state.snapshot().map(Some),
            Self::Cuda(state) => state.snapshot().map(Some),
            Self::Uninitialized(_) => Err(XrtError::Runtime(
                "cannot snapshot uninitialized session recurrent state".to_string(),
            )),
            Self::Poisoned { reason, .. } => Err(XrtError::Runtime(format!(
                "cannot snapshot poisoned session recurrent state: {reason}"
            ))),
        }
    }

    fn restore(
        &mut self,
        snapshot: Option<&BackendStateSnapshot>,
        expected_position: usize,
    ) -> Result<()> {
        match (self, snapshot) {
            (Self::None, None) => Ok(()),
            (Self::Cpu(state), Some(snapshot)) => {
                if snapshot.position()
                    != u64::try_from(expected_position).map_err(|_| {
                        XrtError::Runtime(
                            "rollback position cannot be represented in a recurrent snapshot"
                                .to_string(),
                        )
                    })?
                {
                    return Err(XrtError::Runtime(format!(
                        "recurrent rollback boundary mismatch: snapshot position {} != KV position {expected_position}",
                        snapshot.position()
                    )));
                }
                state.restore(snapshot)
            }
            (Self::Cuda(state), Some(snapshot)) => {
                if snapshot.position()
                    != u64::try_from(expected_position).map_err(|_| {
                        XrtError::Runtime(
                            "rollback position cannot be represented in a recurrent snapshot"
                                .to_string(),
                        )
                    })?
                {
                    return Err(XrtError::Runtime(format!(
                        "recurrent rollback boundary mismatch: snapshot position {} != KV position {expected_position}",
                        snapshot.position()
                    )));
                }
                state.restore(snapshot)
            }
            (Self::None, Some(_)) => Err(XrtError::Runtime(
                "cannot restore recurrent state into a non-hybrid session".to_string(),
            )),
            (Self::Cpu(_) | Self::Cuda(_), None) => Err(XrtError::Runtime(
                "missing recurrent snapshot for a hybrid session".to_string(),
            )),
            (Self::Uninitialized(_), _) => Err(XrtError::Runtime(
                "cannot restore uninitialized session recurrent state".to_string(),
            )),
            (Self::Poisoned { reason, .. }, _) => Err(XrtError::Runtime(format!(
                "cannot restore poisoned session recurrent state: {reason}"
            ))),
        }
    }

    fn clear(&mut self) {
        match self {
            Self::None => {}
            Self::Uninitialized(_) => {}
            Self::Cpu(state) => state.clear(),
            Self::Cuda(state) => state.logical_reset(),
            Self::Poisoned { descriptor, .. } => {
                *self = Self::from_descriptor(descriptor.take());
            }
        }
    }

    fn poison(&mut self, reason: String) {
        let descriptor = match self {
            Self::None => None,
            Self::Uninitialized(descriptor) => Some(descriptor.clone()),
            Self::Cpu(state) => Some(state.descriptor().clone()),
            Self::Cuda(state) => Some(state.descriptor().clone()),
            Self::Poisoned { descriptor, .. } => descriptor.clone(),
        };
        *self = Self::Poisoned { descriptor, reason };
    }

    fn allocated_bytes(&self) -> u64 {
        match self {
            Self::Cpu(state) => state.allocated_bytes(),
            Self::Cuda(state) => state.allocated_bytes(),
            Self::None | Self::Uninitialized(_) | Self::Poisoned { .. } => 0,
        }
    }

    fn layer_uses_recurrent_state(&self, layer: usize) -> bool {
        let descriptor = match self {
            Self::Uninitialized(descriptor) => Some(descriptor),
            Self::Cpu(state) => Some(state.descriptor()),
            Self::Cuda(state) => Some(state.descriptor()),
            Self::Poisoned { descriptor, .. } => descriptor.as_ref(),
            Self::None => None,
        };
        descriptor
            .and_then(|descriptor| descriptor.layers().get(layer))
            .is_some_and(Option::is_some)
    }

    fn supports_fast_checkpoint(&self) -> bool {
        matches!(self, Self::Cuda(_))
    }

    fn begin_fast_checkpoint(&mut self, expected_position: usize) -> Result<()> {
        match self {
            Self::Cuda(state) => state.begin_fast_checkpoint(expected_position),
            Self::None | Self::Cpu(_) | Self::Uninitialized(_) => Err(XrtError::Unsupported(
                "device-local recurrent checkpoints require prepared CUDA DeltaNet state"
                    .to_string(),
            )),
            Self::Poisoned { reason, .. } => Err(XrtError::Runtime(format!(
                "cannot checkpoint poisoned recurrent state: {reason}"
            ))),
        }
    }

    fn commit_fast_checkpoint(&mut self) -> Result<()> {
        match self {
            Self::Cuda(state) => state.commit_fast_checkpoint(),
            Self::None | Self::Cpu(_) | Self::Uninitialized(_) => Err(XrtError::Unsupported(
                "device-local recurrent checkpoints require prepared CUDA DeltaNet state"
                    .to_string(),
            )),
            Self::Poisoned { reason, .. } => Err(XrtError::Runtime(format!(
                "cannot commit a checkpoint for poisoned recurrent state: {reason}"
            ))),
        }
    }

    fn rollback_fast_checkpoint(&mut self, expected_position: usize) -> Result<()> {
        match self {
            Self::Cuda(state) => state.rollback_fast_checkpoint(expected_position),
            Self::None | Self::Cpu(_) | Self::Uninitialized(_) => Err(XrtError::Unsupported(
                "device-local recurrent checkpoints require prepared CUDA DeltaNet state"
                    .to_string(),
            )),
            Self::Poisoned { reason, .. } => Err(XrtError::Runtime(format!(
                "cannot roll back a checkpoint for poisoned recurrent state: {reason}"
            ))),
        }
    }

    fn committed_buffer_generation(&self) -> Option<u8> {
        match self {
            Self::Cuda(state) => Some(state.committed_buffer_generation()),
            Self::None | Self::Uninitialized(_) | Self::Cpu(_) | Self::Poisoned { .. } => None,
        }
    }

    fn validate_position(&self, expected_position: usize) -> Result<()> {
        match self {
            Self::None => Ok(()),
            Self::Cpu(state) => state.validate_position(expected_position),
            Self::Cuda(state) if state.position() == expected_position => Ok(()),
            Self::Cuda(state) => Err(XrtError::Runtime(format!(
                "CUDA DeltaNet state position mismatch: expected {expected_position}, found {}",
                state.position()
            ))),
            Self::Uninitialized(_) => Err(XrtError::Runtime(
                "session recurrent state was not prepared before position validation".to_string(),
            )),
            Self::Poisoned { reason, .. } => Err(XrtError::Runtime(format!(
                "session recurrent state is poisoned and must be reset: {reason}"
            ))),
        }
    }
}

#[derive(Debug)]
pub(crate) enum BackendPrefixSnapshot {
    Cpu {
        cache: SessionKvCache,
        recurrent: Option<BackendStateSnapshot>,
        prefix_len: usize,
        allocated_bytes: u64,
    },
    Cuda {
        layer_caches: Arc<Vec<CudaLayerKvStore>>,
        allocation: Option<GpuAllocationLease>,
        cow_allocations: Vec<GpuAllocationLease>,
        cache_mode: KvCacheMode,
        layer_widths: Vec<usize>,
        page_tokens: usize,
        recurrent: Option<BackendStateSnapshot>,
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

    pub(crate) fn device_allocated_bytes(&self) -> u64 {
        match self {
            Self::Cpu { .. } => 0,
            Self::Cuda { layer_caches, .. } => layer_caches
                .iter()
                .map(CudaLayerKvStore::allocated_bytes)
                .sum(),
        }
    }

    pub(crate) fn host_allocated_bytes(&self) -> u64 {
        match self {
            Self::Cpu {
                allocated_bytes, ..
            } => *allocated_bytes,
            Self::Cuda { recurrent, .. } => recurrent
                .as_ref()
                .map_or(0, BackendStateSnapshot::allocated_bytes),
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
    SharedQ8(CudaSharedQ8LayerKvCache),
    KeyQ4ValueQ8(CudaKeyQ4ValueQ8LayerKvCache),
    SharedKeyQ4ValueQ8(CudaSharedKq4Vq8LayerKvCache),
    AgentAdaptive {
        hot: CudaLayerKvCache,
        cold: CudaKeyQ4ValueQ8LayerKvCache,
        routes: CudaAdaptiveKvRoutes,
        hot_mask: Vec<u8>,
    },
    SharedAgentAdaptive(CudaSharedAdaptiveLayerKvCache),
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
                .alloc_shared_paged_layer_kv_cache(capacity, width, page_tokens)
                .map(Self::F32),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::F32(cache) => cache.len(),
            Self::SharedF32(cache) => cache.len(),
            Self::Q8(cache) => cache.len(),
            Self::SharedQ8(cache) => cache.len(),
            Self::KeyQ4ValueQ8(cache) => cache.len(),
            Self::SharedKeyQ4ValueQ8(cache) => cache.len(),
            Self::AgentAdaptive { hot_mask, .. } => hot_mask.len(),
            Self::SharedAgentAdaptive(cache) => cache.len(),
        }
    }

    fn mode(&self) -> KvCacheMode {
        match self {
            Self::F32(_) | Self::SharedF32(_) => KvCacheMode::F32,
            Self::Q8(_) | Self::SharedQ8(_) => KvCacheMode::Q8,
            Self::KeyQ4ValueQ8(_) | Self::SharedKeyQ4ValueQ8(_) => KvCacheMode::KeyQ4ValueQ8,
            Self::AgentAdaptive { .. } | Self::SharedAgentAdaptive(_) => KvCacheMode::AgentAdaptive,
        }
    }

    fn capacity(&self) -> usize {
        match self {
            Self::F32(cache) => cache.capacity(),
            Self::SharedF32(cache) => cache.capacity(),
            Self::Q8(cache) => cache.capacity(),
            Self::SharedQ8(cache) => cache.capacity(),
            Self::KeyQ4ValueQ8(cache) => cache.capacity(),
            Self::SharedKeyQ4ValueQ8(cache) => cache.capacity(),
            Self::AgentAdaptive {
                hot, cold, routes, ..
            } => hot.capacity().min(cold.capacity()).min(routes.capacity()),
            Self::SharedAgentAdaptive(cache) => cache.capacity(),
        }
    }

    #[allow(dead_code)]
    fn grow(&mut self, device: &CudaDevice, new_capacity: usize) -> Result<()> {
        match self {
            Self::F32(cache) => device.grow_layer_kv_cache(cache, new_capacity),
            Self::SharedF32(cache) if new_capacity <= cache.capacity() => Ok(()),
            Self::SharedF32(cache) => Err(XrtError::Runtime(format!(
                "CUDA shared F32 KV cache capacity {} cannot grow to {new_capacity}; its stable page table was allocated for the session context",
                cache.capacity()
            ))),
            Self::Q8(cache) => device.grow_q8_layer_kv_cache(cache, new_capacity),
            Self::SharedQ8(cache) if new_capacity <= cache.capacity() => Ok(()),
            Self::SharedQ8(cache) => Err(XrtError::Runtime(format!(
                "CUDA shared Q8 KV cache capacity {} cannot grow to {new_capacity}; its stable page table was allocated for the session context",
                cache.capacity()
            ))),
            Self::KeyQ4ValueQ8(cache) => {
                device.grow_key_q4_value_q8_layer_kv_cache(cache, new_capacity)
            }
            Self::SharedKeyQ4ValueQ8(cache) if new_capacity <= cache.capacity() => Ok(()),
            Self::SharedKeyQ4ValueQ8(cache) => Err(XrtError::Runtime(format!(
                "CUDA shared KQ4/VQ8 KV cache capacity {} cannot grow to {new_capacity}; its stable page table was allocated for the session context",
                cache.capacity()
            ))),
            Self::SharedAgentAdaptive(cache) if new_capacity <= cache.capacity() => Ok(()),
            Self::SharedAgentAdaptive(cache) => Err(XrtError::Runtime(format!(
                "CUDA shared adaptive KV cache capacity {} cannot grow to {new_capacity}; its stable page and route tables were allocated for the session context",
                cache.capacity()
            ))),
            Self::AgentAdaptive {
                hot, cold, routes, ..
            } => {
                device.grow_key_q4_value_q8_layer_kv_cache(cold, new_capacity)?;
                device.grow_layer_kv_cache(hot, new_capacity)?;
                device.grow_adaptive_kv_routes(routes, new_capacity)
            }
        }
    }

    #[allow(dead_code)]
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
            Self::SharedQ8(cache) if capacity <= cache.capacity() => cache
                .snapshot_prefix(cache.len())
                .map(Self::SharedQ8),
            Self::SharedQ8(cache) => Err(XrtError::Runtime(format!(
                "CUDA shared Q8 KV snapshot capacity {} cannot satisfy requested capacity {capacity}",
                cache.capacity()
            ))),
            Self::KeyQ4ValueQ8(cache) => device
                .clone_key_q4_value_q8_layer_kv_cache_with_capacity(cache, capacity)
                .map(Self::KeyQ4ValueQ8),
            Self::SharedKeyQ4ValueQ8(cache) if capacity <= cache.capacity() => cache
                .snapshot_prefix(cache.len())
                .map(Self::SharedKeyQ4ValueQ8),
            Self::SharedKeyQ4ValueQ8(cache) => Err(XrtError::Runtime(format!(
                "CUDA shared KQ4/VQ8 KV snapshot capacity {} cannot satisfy requested capacity {capacity}",
                cache.capacity()
            ))),
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
            Self::SharedAgentAdaptive(cache) if capacity <= cache.capacity() => cache
                .snapshot_prefix(cache.len())
                .map(Self::SharedAgentAdaptive),
            Self::SharedAgentAdaptive(cache) => Err(XrtError::Runtime(format!(
                "CUDA shared adaptive KV snapshot capacity {} cannot satisfy requested capacity {capacity}",
                cache.capacity()
            ))),
        }
    }

    fn shared_clone_private_bytes(&self, device: &CudaDevice, capacity: usize) -> Result<u64> {
        match self {
            Self::F32(cache) if cache.is_shared_pages() => {
                device.shared_layer_kv_clone_private_bytes(cache, capacity)
            }
            _ => Err(XrtError::Unsupported(
                "CUDA page-sharing clone is available only for shared F32 KV".to_string(),
            )),
        }
    }

    fn share_with_capacity(&self, device: &CudaDevice, capacity: usize) -> Result<Self> {
        match self {
            Self::F32(cache) if cache.is_shared_pages() => device
                .share_layer_kv_cache_with_capacity(cache, capacity)
                .map(Self::F32),
            _ => Err(XrtError::Unsupported(
                "CUDA page-sharing clone is available only for shared F32 KV".to_string(),
            )),
        }
    }

    fn cow_bytes_for_range(&self, start: usize, end: usize) -> Result<u64> {
        match self {
            Self::F32(cache) if cache.is_shared_pages() => cache.cow_bytes_for_range(start, end),
            _ => Ok(0),
        }
    }

    fn ensure_writable_range(
        &mut self,
        device: &CudaDevice,
        start: usize,
        end: usize,
    ) -> Result<u64> {
        match self {
            Self::F32(cache) if cache.is_shared_pages() => {
                device.ensure_shared_layer_kv_writable_range(cache, start, end)
            }
            _ => Ok(0),
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
            Self::SharedQ8(cache) => {
                if let Err(err) = cache.clear() {
                    tracing::warn!("failed to clear CUDA shared Q8 KV cache: {err}");
                }
            }
            Self::KeyQ4ValueQ8(cache) => cache.clear(),
            Self::SharedKeyQ4ValueQ8(cache) => {
                if let Err(err) = cache.clear() {
                    tracing::warn!("failed to clear CUDA shared KQ4/VQ8 KV cache: {err}");
                }
            }
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
            Self::SharedAgentAdaptive(cache) => {
                if let Err(err) = cache.clear() {
                    tracing::warn!("failed to clear CUDA shared adaptive KV cache: {err}");
                }
            }
        }
    }

    fn truncate(&mut self, new_len: usize) -> Result<()> {
        match self {
            Self::F32(cache) => cache.truncate(new_len),
            Self::SharedF32(cache) => cache.truncate(new_len)?,
            Self::Q8(cache) => cache.truncate(new_len),
            Self::SharedQ8(cache) => cache.truncate(new_len)?,
            Self::KeyQ4ValueQ8(cache) => cache.truncate(new_len),
            Self::SharedKeyQ4ValueQ8(cache) => cache.truncate(new_len)?,
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
            Self::SharedAgentAdaptive(cache) => cache.truncate(new_len)?,
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
            Self::SharedQ8(cache) => cache
                .page_table_bytes()
                .saturating_add(cache.referenced_page_bytes()),
            Self::KeyQ4ValueQ8(cache) => cache.allocated_bytes(),
            Self::SharedKeyQ4ValueQ8(cache) => cache
                .page_table_bytes()
                .saturating_add(cache.referenced_page_bytes()),
            Self::AgentAdaptive {
                hot, cold, routes, ..
            } => hot
                .allocated_bytes()
                .saturating_add(cold.allocated_bytes())
                .saturating_add(routes.allocated_bytes()),
            Self::SharedAgentAdaptive(cache) => cache
                .page_table_bytes()
                .saturating_add(cache.route_table_bytes())
                .saturating_add(cache.referenced_page_bytes()),
        }
    }

    fn uses_shared_pages(&self) -> bool {
        matches!(
            self,
            Self::SharedF32(_)
                | Self::SharedQ8(_)
                | Self::SharedKeyQ4ValueQ8(_)
                | Self::SharedAgentAdaptive(_)
        )
    }

    fn is_shared_f32(&self) -> bool {
        matches!(self, Self::SharedF32(_))
    }

    fn is_shared_q8(&self) -> bool {
        matches!(self, Self::SharedQ8(_))
    }

    fn is_shared_kq4_vq8(&self) -> bool {
        matches!(self, Self::SharedKeyQ4ValueQ8(_))
    }

    fn is_shared_adaptive(&self) -> bool {
        matches!(self, Self::SharedAgentAdaptive(_))
    }

    fn prepare_shared_f32_graph_capacity(&mut self, total_len: usize) -> Result<bool> {
        match self {
            Self::SharedF32(cache) => {
                let previous_epoch = cache.topology_epoch();
                cache.prepare_graph_capacity(total_len)?;
                Ok(cache.topology_epoch() != previous_epoch)
            }
            _ => Ok(false),
        }
    }

    fn prepare_shared_q8_graph_capacity(&mut self, total_len: usize) -> Result<bool> {
        match self {
            Self::SharedQ8(cache) => {
                let previous_epoch = cache.topology_epoch();
                cache.prepare_graph_capacity(total_len)?;
                Ok(cache.topology_epoch() != previous_epoch)
            }
            _ => Ok(false),
        }
    }

    fn prepare_shared_kq4_vq8_graph_capacity(&mut self, total_len: usize) -> Result<bool> {
        match self {
            Self::SharedKeyQ4ValueQ8(cache) => {
                let previous_epoch = cache.topology_epoch();
                cache.prepare_graph_capacity(total_len)?;
                Ok(cache.topology_epoch() != previous_epoch)
            }
            _ => Ok(false),
        }
    }

    fn prepare_shared_adaptive_graph_capacity(&mut self, total_len: usize) -> Result<bool> {
        match self {
            Self::SharedAgentAdaptive(cache) => {
                let previous_epoch = cache.topology_epoch();
                cache.prepare_graph_capacity(total_len)?;
                Ok(cache.topology_epoch() != previous_epoch)
            }
            _ => Ok(false),
        }
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

    fn snapshot_q8_prefix_into_pool(
        &self,
        pool: &CudaQ8KvPagePool,
        max_tokens: usize,
        prefix_len: usize,
    ) -> Result<Self> {
        match self {
            Self::Q8(source) => {
                if source.len() != prefix_len {
                    return Err(XrtError::Runtime(format!(
                        "cannot copy {prefix_len} shared Q8 prefix tokens from contiguous CUDA cache length {}",
                        source.len()
                    )));
                }
                let mut snapshot = pool.allocate_cache(max_tokens)?;
                snapshot.copy_prefix_from_paged_q8(source, prefix_len)?;
                Ok(Self::SharedQ8(snapshot))
            }
            Self::SharedQ8(source) => source.snapshot_prefix(prefix_len).map(Self::SharedQ8),
            other => Err(XrtError::Runtime(format!(
                "cannot create a shared Q8 prefix snapshot from {} CUDA KV storage",
                other.mode().as_str()
            ))),
        }
    }

    fn snapshot_kq4_vq8_prefix_into_pool(
        &self,
        pool: &CudaKq4Vq8KvPagePool,
        max_tokens: usize,
        prefix_len: usize,
    ) -> Result<Self> {
        match self {
            Self::KeyQ4ValueQ8(source) => {
                if source.len() != prefix_len {
                    return Err(XrtError::Runtime(format!(
                        "cannot copy {prefix_len} shared KQ4/VQ8 prefix tokens from contiguous CUDA cache length {}",
                        source.len()
                    )));
                }
                let mut snapshot = pool.allocate_cache(max_tokens)?;
                snapshot.copy_prefix_from_paged_kq4_vq8(source, prefix_len)?;
                Ok(Self::SharedKeyQ4ValueQ8(snapshot))
            }
            Self::SharedKeyQ4ValueQ8(source) => source
                .snapshot_prefix(prefix_len)
                .map(Self::SharedKeyQ4ValueQ8),
            other => Err(XrtError::Runtime(format!(
                "cannot create a shared KQ4/VQ8 prefix snapshot from {} CUDA KV storage",
                other.mode().as_str()
            ))),
        }
    }

    fn snapshot_adaptive_prefix_into_pools(
        &self,
        hot_pool: &CudaF32KvPagePool,
        cold_pool: &CudaKq4Vq8KvPagePool,
        max_tokens: usize,
        prefix_len: usize,
    ) -> Result<Self> {
        match self {
            Self::AgentAdaptive {
                hot,
                cold,
                hot_mask,
                ..
            } => {
                if hot_mask.len() != prefix_len {
                    return Err(XrtError::Runtime(format!(
                        "cannot copy {prefix_len} shared adaptive prefix tokens from contiguous CUDA cache length {}",
                        hot_mask.len()
                    )));
                }
                let mut snapshot =
                    CudaSharedAdaptiveLayerKvCache::new(hot_pool, cold_pool, max_tokens)?;
                snapshot.copy_prefix_from_paged_adaptive(hot, cold, hot_mask)?;
                Ok(Self::SharedAgentAdaptive(snapshot))
            }
            Self::SharedAgentAdaptive(source) => source
                .snapshot_prefix(prefix_len)
                .map(Self::SharedAgentAdaptive),
            other => Err(XrtError::Runtime(format!(
                "cannot create a shared adaptive prefix snapshot from {} CUDA KV storage",
                other.mode().as_str()
            ))),
        }
    }

    fn migrate_agent_adaptive_route(
        &mut self,
        device: &CudaDevice,
        desired_hot_mask: &[u8],
    ) -> Result<()> {
        if let Self::SharedAgentAdaptive(cache) = self {
            return cache.migrate_hot_to_cold(desired_hot_mask);
        }
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
    model_identity: String,
    architecture: String,
    device_ordinal: usize,
    weight_kinds: Vec<&'static str>,
    cache_mode: KvCacheMode,
    shared_kv_pages: bool,
    kv_capacity: usize,
    placement_generation: u64,
    scratch_generation: u64,
    recurrent_buffer_generation: Option<u8>,
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
    Captured {
        graph: CudaGraphExec,
        _allocation: Option<GpuAllocationLease>,
    },
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
    shared_f32_bindings: Vec<CudaSharedF32GraphBinding>,
    shared_q8_bindings: Vec<CudaSharedQ8GraphBinding>,
    shared_kq4_vq8_bindings: Vec<CudaSharedKq4Vq8GraphBinding>,
    shared_adaptive_bindings: Vec<CudaSharedAdaptiveGraphBinding>,
    allocation: Option<GpuAllocationLease>,
    key: Option<CudaDecodeGraphKey>,
    alternate_executable: Option<CudaGraphExec>,
    alternate_allocation: Option<GpuAllocationLease>,
    alternate_key: Option<CudaDecodeGraphKey>,
    alternate_shared_f32_bindings: Vec<CudaSharedF32GraphBinding>,
    alternate_shared_q8_bindings: Vec<CudaSharedQ8GraphBinding>,
    alternate_shared_kq4_vq8_bindings: Vec<CudaSharedKq4Vq8GraphBinding>,
    alternate_shared_adaptive_bindings: Vec<CudaSharedAdaptiveGraphBinding>,
    mode: CudaGraphMode,
    capture_state: CudaGraphCaptureState,
    last_error: Option<String>,
}

impl CudaDecodeGraphState {
    fn new(mode: CudaGraphMode) -> Self {
        Self {
            executable: None,
            shared_f32_bindings: Vec::new(),
            shared_q8_bindings: Vec::new(),
            shared_kq4_vq8_bindings: Vec::new(),
            shared_adaptive_bindings: Vec::new(),
            allocation: None,
            key: None,
            alternate_executable: None,
            alternate_allocation: None,
            alternate_key: None,
            alternate_shared_f32_bindings: Vec::new(),
            alternate_shared_q8_bindings: Vec::new(),
            alternate_shared_kq4_vq8_bindings: Vec::new(),
            alternate_shared_adaptive_bindings: Vec::new(),
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
        self.shared_f32_bindings.clear();
        self.shared_q8_bindings.clear();
        self.shared_kq4_vq8_bindings.clear();
        self.shared_adaptive_bindings.clear();
        self.allocation = None;
        self.key = None;
        self.alternate_executable = None;
        self.alternate_allocation = None;
        self.alternate_key = None;
        self.alternate_shared_f32_bindings.clear();
        self.alternate_shared_q8_bindings.clear();
        self.alternate_shared_kq4_vq8_bindings.clear();
        self.alternate_shared_adaptive_bindings.clear();
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
        self.shared_f32_bindings.clear();
        self.shared_q8_bindings.clear();
        self.shared_kq4_vq8_bindings.clear();
        self.shared_adaptive_bindings.clear();
        self.allocation = None;
        self.key = None;
        self.alternate_executable = None;
        self.alternate_allocation = None;
        self.alternate_key = None;
        self.alternate_shared_f32_bindings.clear();
        self.alternate_shared_q8_bindings.clear();
        self.alternate_shared_kq4_vq8_bindings.clear();
        self.alternate_shared_adaptive_bindings.clear();
        self.last_error = Some(error.into());
        self.capture_state = CudaGraphCaptureState::EagerFallback;
    }

    fn captured(
        &mut self,
        key: CudaDecodeGraphKey,
        executable: CudaGraphExec,
        shared_f32_bindings: Vec<CudaSharedF32GraphBinding>,
        shared_q8_bindings: Vec<CudaSharedQ8GraphBinding>,
        shared_kq4_vq8_bindings: Vec<CudaSharedKq4Vq8GraphBinding>,
        shared_adaptive_bindings: Vec<CudaSharedAdaptiveGraphBinding>,
        allocation: Option<GpuAllocationLease>,
    ) {
        if self.key.as_ref() != Some(&key) {
            self.alternate_executable = self.executable.take();
            self.alternate_allocation = self.allocation.take();
            self.alternate_key = self.key.take();
            self.alternate_shared_f32_bindings = std::mem::take(&mut self.shared_f32_bindings);
            self.alternate_shared_q8_bindings = std::mem::take(&mut self.shared_q8_bindings);
            self.alternate_shared_kq4_vq8_bindings =
                std::mem::take(&mut self.shared_kq4_vq8_bindings);
            self.alternate_shared_adaptive_bindings =
                std::mem::take(&mut self.shared_adaptive_bindings);
        }
        self.executable = Some(executable);
        self.shared_f32_bindings = shared_f32_bindings;
        self.shared_q8_bindings = shared_q8_bindings;
        self.shared_kq4_vq8_bindings = shared_kq4_vq8_bindings;
        self.shared_adaptive_bindings = shared_adaptive_bindings;
        self.allocation = allocation;
        self.key = Some(key);
        self.last_error = None;
        self.capture_state = CudaGraphCaptureState::Captured;
    }

    fn is_enabled(&self) -> bool {
        self.mode != CudaGraphMode::Disabled
            && self.capture_state != CudaGraphCaptureState::EagerFallback
    }

    fn validate_shared_f32_bindings(
        &self,
        layer_caches: &[CudaLayerKvStore],
        append_position: usize,
    ) -> Result<()> {
        let shared_cache_count = layer_caches
            .iter()
            .filter(|cache| cache.is_shared_f32())
            .count();
        if shared_cache_count == 0 {
            if self.shared_f32_bindings.is_empty() {
                return Ok(());
            }
            return Err(XrtError::Cuda(
                "CUDA decode graph retained shared F32 pages for contiguous KV caches".to_string(),
            ));
        }
        if shared_cache_count != layer_caches.len()
            || self.shared_f32_bindings.len() != layer_caches.len()
        {
            return Err(XrtError::Cuda(format!(
                "CUDA shared F32 decode graph binding count {} does not match {} layer caches",
                self.shared_f32_bindings.len(),
                layer_caches.len()
            )));
        }
        for (layer, (binding, cache)) in self
            .shared_f32_bindings
            .iter()
            .zip(layer_caches)
            .enumerate()
        {
            let CudaLayerKvStore::SharedF32(cache) = cache else {
                return Err(XrtError::Cuda(format!(
                    "CUDA shared F32 decode graph layer {layer} changed cache layout"
                )));
            };
            binding.validate_cache(cache, append_position)?;
        }
        Ok(())
    }

    fn validate_shared_q8_bindings(
        &self,
        layer_caches: &[CudaLayerKvStore],
        append_position: usize,
    ) -> Result<()> {
        let shared_cache_count = layer_caches
            .iter()
            .filter(|cache| cache.is_shared_q8())
            .count();
        if shared_cache_count == 0 {
            if self.shared_q8_bindings.is_empty() {
                return Ok(());
            }
            return Err(XrtError::Cuda(
                "CUDA decode graph retained shared Q8 pages for another KV layout".to_string(),
            ));
        }
        if shared_cache_count != layer_caches.len()
            || self.shared_q8_bindings.len() != layer_caches.len()
        {
            return Err(XrtError::Cuda(format!(
                "CUDA shared Q8 decode graph binding count {} does not match {} layer caches",
                self.shared_q8_bindings.len(),
                layer_caches.len()
            )));
        }
        for (layer, (binding, cache)) in
            self.shared_q8_bindings.iter().zip(layer_caches).enumerate()
        {
            let CudaLayerKvStore::SharedQ8(cache) = cache else {
                return Err(XrtError::Cuda(format!(
                    "CUDA shared Q8 decode graph layer {layer} changed cache layout"
                )));
            };
            binding.validate_cache(cache, append_position)?;
        }
        Ok(())
    }

    fn validate_shared_kq4_vq8_bindings(
        &self,
        layer_caches: &[CudaLayerKvStore],
        append_position: usize,
    ) -> Result<()> {
        let shared_cache_count = layer_caches
            .iter()
            .filter(|cache| cache.is_shared_kq4_vq8())
            .count();
        if shared_cache_count == 0 {
            if self.shared_kq4_vq8_bindings.is_empty() {
                return Ok(());
            }
            return Err(XrtError::Cuda(
                "CUDA decode graph retained shared KQ4/VQ8 pages for another KV layout".to_string(),
            ));
        }
        if shared_cache_count != layer_caches.len()
            || self.shared_kq4_vq8_bindings.len() != layer_caches.len()
        {
            return Err(XrtError::Cuda(format!(
                "CUDA shared KQ4/VQ8 decode graph binding count {} does not match {} layer caches",
                self.shared_kq4_vq8_bindings.len(),
                layer_caches.len()
            )));
        }
        for (layer, (binding, cache)) in self
            .shared_kq4_vq8_bindings
            .iter()
            .zip(layer_caches)
            .enumerate()
        {
            let CudaLayerKvStore::SharedKeyQ4ValueQ8(cache) = cache else {
                return Err(XrtError::Cuda(format!(
                    "CUDA shared KQ4/VQ8 decode graph layer {layer} changed cache layout"
                )));
            };
            binding.validate_cache(cache, append_position)?;
        }
        Ok(())
    }

    fn validate_shared_adaptive_bindings(
        &self,
        layer_caches: &[CudaLayerKvStore],
        append_position: usize,
    ) -> Result<()> {
        let shared_cache_count = layer_caches
            .iter()
            .filter(|cache| cache.is_shared_adaptive())
            .count();
        if shared_cache_count == 0 {
            if self.shared_adaptive_bindings.is_empty() {
                return Ok(());
            }
            return Err(XrtError::Cuda(
                "CUDA decode graph retained shared adaptive pages for another KV layout"
                    .to_string(),
            ));
        }
        if shared_cache_count != layer_caches.len()
            || self.shared_adaptive_bindings.len() != layer_caches.len()
        {
            return Err(XrtError::Cuda(format!(
                "CUDA shared adaptive decode graph binding count {} does not match {} layer caches",
                self.shared_adaptive_bindings.len(),
                layer_caches.len()
            )));
        }
        for (layer, (binding, cache)) in self
            .shared_adaptive_bindings
            .iter()
            .zip(layer_caches)
            .enumerate()
        {
            let CudaLayerKvStore::SharedAgentAdaptive(cache) = cache else {
                return Err(XrtError::Cuda(format!(
                    "CUDA shared adaptive decode graph layer {layer} changed cache layout"
                )));
            };
            binding.validate_cache(cache, append_position)?;
        }
        Ok(())
    }

    fn executable_for(&mut self, key: &CudaDecodeGraphKey) -> Option<&CudaGraphExec> {
        if self.key.as_ref() == Some(key) {
            return self.executable.as_ref();
        }
        if self.alternate_key.as_ref() == Some(key) {
            std::mem::swap(&mut self.key, &mut self.alternate_key);
            std::mem::swap(&mut self.executable, &mut self.alternate_executable);
            std::mem::swap(&mut self.allocation, &mut self.alternate_allocation);
            std::mem::swap(
                &mut self.shared_f32_bindings,
                &mut self.alternate_shared_f32_bindings,
            );
            std::mem::swap(
                &mut self.shared_q8_bindings,
                &mut self.alternate_shared_q8_bindings,
            );
            std::mem::swap(
                &mut self.shared_kq4_vq8_bindings,
                &mut self.alternate_shared_kq4_vq8_bindings,
            );
            std::mem::swap(
                &mut self.shared_adaptive_bindings,
                &mut self.alternate_shared_adaptive_bindings,
            );
            return self.executable.as_ref();
        }
        None
    }

    fn has_executable_for(&self, key: &CudaDecodeGraphKey) -> bool {
        (self.key.as_ref() == Some(key) && self.executable.is_some())
            || (self.alternate_key.as_ref() == Some(key) && self.alternate_executable.is_some())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CudaMoeExpertGraphKey {
    model_identity: String,
    architecture: String,
    device_ordinal: usize,
    cache_mode: KvCacheMode,
    placement_generation: u64,
    scratch_generation: u64,
    layer_index: usize,
    logical_expert: usize,
    gpu_slot: usize,
    embedding_length: usize,
    intermediate_size: usize,
    selected_per_token: usize,
    weight_kinds: [&'static str; 3],
}

struct CudaMoeExpertGraphEntry {
    key: CudaMoeExpertGraphKey,
    graph: CudaGraphExec,
    _allocation: Option<GpuAllocationLease>,
    // Captured kernel nodes retain raw pointers into these matrices. Keeping
    // the slot alive also makes old placement epochs safe to destroy lazily.
    _expert_slot: Arc<ResidentMoeExpertSlot>,
}

impl fmt::Debug for CudaMoeExpertGraphEntry {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaMoeExpertGraphEntry")
            .field("key", &self.key)
            .field("graph", &self.graph)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Default)]
struct CudaMoeExpertGraphCache {
    entries: VecDeque<CudaMoeExpertGraphEntry>,
}

impl CudaMoeExpertGraphCache {
    fn graph_for(&mut self, key: &CudaMoeExpertGraphKey) -> Option<&CudaGraphExec> {
        let index = self.entries.iter().position(|entry| &entry.key == key)?;
        if index + 1 != self.entries.len() {
            let entry = self
                .entries
                .remove(index)
                .expect("the located MoE graph entry must still exist");
            self.entries.push_back(entry);
        }
        self.entries.back().map(|entry| &entry.graph)
    }

    fn insert(
        &mut self,
        key: CudaMoeExpertGraphKey,
        graph: CudaGraphExec,
        allocation: Option<GpuAllocationLease>,
        expert_slot: Arc<ResidentMoeExpertSlot>,
    ) {
        if let Some(index) = self.entries.iter().position(|entry| entry.key == key) {
            self.entries.remove(index);
        }
        while self.entries.len() >= CUDA_MOE_EXPERT_GRAPH_CACHE_ENTRIES {
            self.entries.pop_front();
        }
        self.entries.push_back(CudaMoeExpertGraphEntry {
            key,
            graph,
            _allocation: allocation,
            _expert_slot: expert_slot,
        });
    }

    fn clear(&mut self) {
        self.entries.clear();
    }
}

fn reserve_cuda_graph_allocation(
    arena: Option<&Arc<GpuAllocationArena>>,
    graph: &CudaGraphExec,
) -> Result<Option<GpuAllocationLease>> {
    arena
        .map(|arena| arena.reserve(GpuAllocationClass::Graph, graph.accounting_bytes()))
        .transpose()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MoeScratchGeometry {
    expert_count: usize,
    selected_per_token: usize,
    embedding_length: usize,
    intermediate_size: usize,
    shared_intermediate_size: Option<usize>,
}

impl MoeScratchGeometry {
    fn from_config(config: &LlamaConfig) -> Result<Option<Self>> {
        if !config.is_moe() {
            return Ok(None);
        }
        let expert_count = config.expert_count.ok_or_else(|| {
            XrtError::InvalidMetadata("MoE scratch geometry is missing expert_count".to_string())
        })?;
        let selected_per_token = config.expert_used_count.ok_or_else(|| {
            XrtError::InvalidMetadata(
                "MoE scratch geometry is missing expert_used_count".to_string(),
            )
        })?;
        if selected_per_token == 0
            || selected_per_token > expert_count
            || selected_per_token > MAX_SELECTED_EXPERTS
        {
            return Err(XrtError::InvalidMetadata(format!(
                "MoE selected expert count {selected_per_token} must be in 1..={} and no greater than expert_count {expert_count}",
                MAX_SELECTED_EXPERTS
            )));
        }
        Ok(Some(Self {
            expert_count,
            selected_per_token,
            embedding_length: config.embedding_length,
            intermediate_size: config.feed_forward_length,
            shared_intermediate_size: config.expert_shared_feed_forward_length,
        }))
    }

    fn output_rows(self) -> usize {
        self.selected_per_token + usize::from(self.shared_intermediate_size.is_some())
    }

    fn device_element_count(self) -> Result<usize> {
        let auxiliary_hidden = self
            .embedding_length
            .checked_mul(1 + usize::from(self.shared_intermediate_size.is_some()))
            .ok_or_else(|| {
                XrtError::Runtime(
                    "CUDA MoE auxiliary device scratch element count overflowed".to_string(),
                )
            })?;
        let packed_cpu_outputs = self
            .output_rows()
            .checked_mul(self.embedding_length)
            .ok_or_else(|| {
                XrtError::Runtime(
                    "CUDA MoE packed CPU output scratch element count overflowed".to_string(),
                )
            })?;
        self.expert_count
            .checked_mul(self.embedding_length)
            .and_then(|elements| elements.checked_add(auxiliary_hidden))
            .and_then(|elements| elements.checked_add(packed_cpu_outputs))
            .and_then(|elements| elements.checked_add(self.expert_count))
            .ok_or_else(|| {
                XrtError::Runtime("CUDA MoE device scratch element count overflowed".to_string())
            })
    }

    fn pinned_element_count(self) -> Result<usize> {
        self.output_rows()
            .checked_mul(self.embedding_length)
            .and_then(|elements| elements.checked_add(self.embedding_length))
            .and_then(|elements| elements.checked_add(self.expert_count))
            .ok_or_else(|| {
                XrtError::Runtime("CUDA MoE pinned staging element count overflowed".to_string())
            })
    }

    fn device_bytes(self) -> Result<u64> {
        self.device_element_count()?
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| {
                XrtError::Runtime("CUDA MoE device scratch byte count overflowed".to_string())
            })
    }

    fn pinned_bytes(self) -> Result<u64> {
        self.pinned_element_count()?
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| {
                XrtError::Runtime("CUDA MoE pinned staging byte count overflowed".to_string())
            })
    }
}

#[derive(Debug)]
struct MoePinnedHostStaging {
    geometry: MoeScratchGeometry,
    router_logits: CudaPinnedF32Buffer,
    input: Option<CudaPinnedF32Buffer>,
    transfer_stream: Option<CudaExecutionStream>,
    outputs: CudaPinnedF32Buffer,
    gate: Vec<f32>,
    up: Vec<f32>,
    shared_gate: Vec<f32>,
    shared_up: Vec<f32>,
}

impl MoePinnedHostStaging {
    fn allocate(device: &CudaDevice, geometry: MoeScratchGeometry) -> Result<Self> {
        let output_elements = geometry
            .output_rows()
            .checked_mul(geometry.embedding_length)
            .ok_or_else(|| {
                XrtError::Runtime("CUDA MoE host output staging size overflowed".to_string())
            })?;
        Ok(Self {
            geometry,
            router_logits: device.alloc_pinned_f32(geometry.expert_count)?,
            input: Some(device.alloc_pinned_f32(geometry.embedding_length)?),
            transfer_stream: Some(device.create_execution_stream()?),
            outputs: device.alloc_pinned_f32(output_elements)?,
            gate: try_zeroed_f32(
                geometry
                    .selected_per_token
                    .checked_mul(geometry.intermediate_size)
                    .ok_or_else(|| {
                        XrtError::Runtime(
                            "CPU MoE pinned-stage gate scratch size overflowed".to_string(),
                        )
                    })?,
                "CPU MoE pinned-stage gate scratch",
            )?,
            up: try_zeroed_f32(
                geometry
                    .selected_per_token
                    .checked_mul(geometry.intermediate_size)
                    .ok_or_else(|| {
                        XrtError::Runtime(
                            "CPU MoE pinned-stage up scratch size overflowed".to_string(),
                        )
                    })?,
                "CPU MoE pinned-stage up scratch",
            )?,
            shared_gate: try_zeroed_f32(
                geometry.shared_intermediate_size.unwrap_or(0),
                "CPU MoE shared gate scratch",
            )?,
            shared_up: try_zeroed_f32(
                geometry.shared_intermediate_size.unwrap_or(0),
                "CPU MoE shared up scratch",
            )?,
        })
    }

    fn clear_request_data(&mut self) {
        self.router_logits.clear();
        if let Some(input) = &mut self.input {
            input.clear();
        }
        self.outputs.clear();
        self.gate.fill(0.0);
        self.up.fill(0.0);
        self.shared_gate.fill(0.0);
        self.shared_up.fill(0.0);
    }

    fn pinned_bytes(&self) -> u64 {
        let input_bytes = u64::try_from(self.geometry.embedding_length)
            .unwrap_or(u64::MAX)
            .saturating_mul(std::mem::size_of::<f32>() as u64);
        u64::try_from(self.router_logits.byte_len())
            .unwrap_or(u64::MAX)
            .saturating_add(input_bytes)
            .saturating_add(u64::try_from(self.outputs.byte_len()).unwrap_or(u64::MAX))
    }

    fn ensure_transfer_resources(&mut self, device: &CudaDevice) -> Result<()> {
        if self.input.is_none() {
            self.input = Some(device.alloc_pinned_f32(self.geometry.embedding_length)?);
        }
        if self.transfer_stream.is_none() {
            self.transfer_stream = Some(device.create_execution_stream()?);
        }
        Ok(())
    }
}

fn try_zeroed_f32(len: usize, label: &str) -> Result<Vec<f32>> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|error| {
        XrtError::Runtime(format!(
            "failed to reserve {label} ({len} f32 values): {error}"
        ))
    })?;
    values.resize(len, 0.0);
    Ok(values)
}

#[derive(Debug)]
struct CudaMoeDecodeScratch {
    geometry: MoeScratchGeometry,
    router_logits: CudaF32Buffer,
    graph_cache: CudaMoeExpertGraphCache,
    graph_mode: CudaGraphMode,
    graph_capture_state: CudaGraphCaptureState,
    graph_last_error: Option<String>,
    expert_outputs: Vec<CudaF32Buffer>,
    cpu_outputs: CudaF32Buffer,
    shared_output: Option<CudaF32Buffer>,
    accumulator: CudaF32Buffer,
    host: Arc<Mutex<MoePinnedHostStaging>>,
}

impl CudaMoeDecodeScratch {
    fn allocate(device: &CudaDevice, geometry: MoeScratchGeometry) -> Result<Self> {
        let mut expert_outputs = Vec::new();
        expert_outputs
            .try_reserve_exact(geometry.expert_count)
            .map_err(|error| {
                XrtError::Runtime(format!(
                    "failed to reserve CUDA MoE expert output handles: {error}"
                ))
            })?;
        for _ in 0..geometry.expert_count {
            expert_outputs.push(device.zeros_f32(geometry.embedding_length)?);
        }
        Ok(Self {
            geometry,
            router_logits: device.zeros_f32(geometry.expert_count)?,
            graph_cache: CudaMoeExpertGraphCache::default(),
            graph_mode: CudaGraphMode::Disabled,
            graph_capture_state: CudaGraphCaptureState::Disabled,
            graph_last_error: None,
            expert_outputs,
            cpu_outputs: device.zeros_f32(
                geometry
                    .output_rows()
                    .checked_mul(geometry.embedding_length)
                    .ok_or_else(|| {
                        XrtError::Runtime(
                            "CUDA MoE packed CPU output allocation overflowed".to_string(),
                        )
                    })?,
            )?,
            shared_output: geometry
                .shared_intermediate_size
                .map(|_| device.zeros_f32(geometry.embedding_length))
                .transpose()?,
            accumulator: device.zeros_f32(geometry.embedding_length)?,
            host: Arc::new(Mutex::new(MoePinnedHostStaging::allocate(
                device, geometry,
            )?)),
        })
    }

    fn device_bytes(&self) -> u64 {
        self.router_logits.byte_len() as u64
            + self
                .expert_outputs
                .iter()
                .map(|output| output.byte_len() as u64)
                .sum::<u64>()
            + self.cpu_outputs.byte_len() as u64
            + self
                .shared_output
                .as_ref()
                .map_or(0, |output| output.byte_len() as u64)
            + self.accumulator.byte_len() as u64
    }

    fn staging_bytes(&self) -> u64 {
        self.host.lock().pinned_bytes()
    }

    fn clear_host(&self) {
        self.host.lock().clear_request_data();
    }

    fn configure_graph_mode(&mut self, mode: CudaGraphMode) {
        if self.graph_mode == mode {
            return;
        }
        self.graph_cache.clear();
        self.graph_mode = mode;
        self.graph_capture_state = if mode == CudaGraphMode::Disabled {
            CudaGraphCaptureState::Disabled
        } else {
            CudaGraphCaptureState::NotCaptured
        };
        self.graph_last_error = None;
    }

    fn reset_graphs(&mut self) {
        self.graph_cache.clear();
        self.graph_capture_state = if self.graph_mode == CudaGraphMode::Disabled {
            CudaGraphCaptureState::Disabled
        } else {
            CudaGraphCaptureState::NotCaptured
        };
        self.graph_last_error = None;
    }

    fn graph_enabled(&self, full_gpu_residency: bool) -> bool {
        matches!(self.graph_mode, CudaGraphMode::Enabled)
            || (self.graph_mode == CudaGraphMode::Auto && full_gpu_residency)
    }

    fn graph_available(&self, full_gpu_residency: bool) -> bool {
        self.graph_enabled(full_gpu_residency)
            && self.graph_capture_state != CudaGraphCaptureState::EagerFallback
    }

    fn mark_graph_captured(&mut self) {
        self.graph_capture_state = CudaGraphCaptureState::Captured;
        self.graph_last_error = None;
    }

    fn graph_fallback(&mut self, error: impl Into<String>) {
        if self.graph_mode == CudaGraphMode::Disabled {
            return;
        }
        self.graph_capture_state = CudaGraphCaptureState::EagerFallback;
        self.graph_last_error = Some(error.into());
    }

    fn clear_graphs(&mut self) {
        self.graph_cache.clear();
    }
}

struct MoeHostStagingClearGuard {
    host: Arc<Mutex<MoePinnedHostStaging>>,
}

impl MoeHostStagingClearGuard {
    fn new(host: Arc<Mutex<MoePinnedHostStaging>>) -> Self {
        Self { host }
    }
}

impl Drop for MoeHostStagingClearGuard {
    fn drop(&mut self) {
        self.host.lock().clear_request_data();
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Qwen35ScratchGeometry {
    embedding_length: usize,
    conv_channels: usize,
    inner_size: usize,
    value_heads: usize,
    q_width: usize,
}

impl Qwen35ScratchGeometry {
    fn from_config(config: &LlamaConfig) -> Result<Option<Self>> {
        let Some(descriptor) = config.deltanet_state_descriptor() else {
            return Ok(None);
        };
        let conv_channels = descriptor
            .state_size()
            .checked_mul(descriptor.group_count())
            .and_then(|value| value.checked_mul(2))
            .and_then(|value| value.checked_add(descriptor.inner_size()))
            .ok_or_else(|| {
                XrtError::InvalidMetadata(
                    "Qwen3.5 CUDA scratch convolution width overflowed".to_string(),
                )
            })?;
        Ok(Some(Self {
            embedding_length: config.embedding_length,
            conv_channels,
            inner_size: descriptor.inner_size(),
            value_heads: descriptor.dt_rank(),
            q_width: config.q_width(),
        }))
    }

    fn device_elements(self) -> Result<usize> {
        self.conv_channels
            .checked_mul(2)
            .and_then(|value| {
                self.inner_size
                    .checked_mul(2)
                    .and_then(|extra| value.checked_add(extra))
            })
            .and_then(|value| {
                self.value_heads
                    .checked_mul(4)
                    .and_then(|extra| value.checked_add(extra))
            })
            .and_then(|value| {
                self.q_width
                    .checked_mul(3)
                    .and_then(|extra| value.checked_add(extra))
            })
            .and_then(|value| {
                self.embedding_length
                    .checked_mul(3)
                    .and_then(|extra| value.checked_add(extra))
            })
            .ok_or_else(|| {
                XrtError::Runtime("Qwen3.5 CUDA scratch element count overflowed".to_string())
            })
    }

    fn device_bytes(self) -> Result<u64> {
        self.device_elements()?
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| {
                XrtError::Runtime("Qwen3.5 CUDA scratch byte count overflowed".to_string())
            })
    }
}

#[derive(Debug)]
struct CudaQwen35DecodeScratch {
    geometry: Qwen35ScratchGeometry,
    qkv: CudaF32Buffer,
    conv_output: CudaF32Buffer,
    deltanet_gate: CudaF32Buffer,
    alpha: CudaF32Buffer,
    beta: CudaF32Buffer,
    decays: CudaF32Buffer,
    betas: CudaF32Buffer,
    deltanet_output: CudaF32Buffer,
    qg: CudaF32Buffer,
    attention_gate: CudaF32Buffer,
    mtp_hidden: CudaF32Buffer,
    mtp_concat: CudaF32Buffer,
}

impl CudaQwen35DecodeScratch {
    fn allocate(device: &CudaDevice, geometry: Qwen35ScratchGeometry) -> Result<Self> {
        Ok(Self {
            geometry,
            qkv: device.zeros_f32(geometry.conv_channels)?,
            conv_output: device.zeros_f32(geometry.conv_channels)?,
            deltanet_gate: device.zeros_f32(geometry.inner_size)?,
            alpha: device.zeros_f32(geometry.value_heads)?,
            beta: device.zeros_f32(geometry.value_heads)?,
            decays: device.zeros_f32(geometry.value_heads)?,
            betas: device.zeros_f32(geometry.value_heads)?,
            deltanet_output: device.zeros_f32(geometry.inner_size)?,
            qg: device.zeros_f32(geometry.q_width.checked_mul(2).ok_or_else(|| {
                XrtError::Runtime("Qwen3.5 interleaved Q/G scratch width overflowed".to_string())
            })?)?,
            attention_gate: device.zeros_f32(geometry.q_width)?,
            mtp_hidden: device.zeros_f32(geometry.embedding_length)?,
            mtp_concat: device.zeros_f32(geometry.embedding_length.checked_mul(2).ok_or_else(
                || XrtError::Runtime("Qwen3.5 MTP concat width overflowed".to_string()),
            )?)?,
        })
    }

    fn allocated_bytes(&self) -> u64 {
        [
            &self.qkv,
            &self.conv_output,
            &self.deltanet_gate,
            &self.alpha,
            &self.beta,
            &self.decays,
            &self.betas,
            &self.deltanet_output,
            &self.qg,
            &self.attention_gate,
            &self.mtp_hidden,
            &self.mtp_concat,
        ]
        .into_iter()
        .map(|buffer| buffer.byte_len() as u64)
        .sum()
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
    moe: Option<CudaMoeDecodeScratch>,
    qwen35: Option<CudaQwen35DecodeScratch>,
}

impl CudaDecodeScratch {
    fn estimated_allocated_bytes(
        embedding_length: usize,
        q_width: usize,
        kv_width: usize,
        feed_forward_length: usize,
        vocab_size: usize,
        moe_geometry: Option<MoeScratchGeometry>,
        qwen35_geometry: Option<Qwen35ScratchGeometry>,
    ) -> Result<u64> {
        let elements = embedding_length
            .checked_mul(4)
            .and_then(|value| q_width.checked_mul(3).and_then(|q| value.checked_add(q)))
            .and_then(|value| kv_width.checked_mul(3).and_then(|kv| value.checked_add(kv)))
            .and_then(|value| {
                feed_forward_length
                    .checked_mul(2)
                    .and_then(|ffn| value.checked_add(ffn))
            })
            .and_then(|value| value.checked_add(vocab_size))
            .ok_or_else(|| {
                XrtError::Runtime("CUDA decode scratch element count overflowed".to_string())
            })?;
        let buffer_bytes = elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                XrtError::Runtime("CUDA decode scratch byte count overflowed".to_string())
            })? as u64;
        let base_bytes = buffer_bytes
            .checked_add((4 * std::mem::size_of::<u32>()) as u64)
            .ok_or_else(|| {
                XrtError::Runtime("CUDA decode parameter byte count overflowed".to_string())
            })?;
        let with_moe = base_bytes
            .checked_add(
                moe_geometry
                    .map(MoeScratchGeometry::device_bytes)
                    .transpose()?
                    .unwrap_or(0),
            )
            .ok_or_else(|| {
                XrtError::Runtime("CUDA decode plus MoE scratch byte count overflowed".to_string())
            })?;
        with_moe
            .checked_add(
                qwen35_geometry
                    .map(Qwen35ScratchGeometry::device_bytes)
                    .transpose()?
                    .unwrap_or(0),
            )
            .ok_or_else(|| {
                XrtError::Runtime(
                    "CUDA decode plus Qwen3.5 scratch byte count overflowed".to_string(),
                )
            })
    }

    fn estimated_staging_bytes(moe_geometry: Option<MoeScratchGeometry>) -> Result<u64> {
        moe_geometry
            .map(MoeScratchGeometry::pinned_bytes)
            .transpose()
            .map(|bytes| bytes.unwrap_or(0))
    }

    fn allocate(
        device: &CudaDevice,
        embedding_length: usize,
        q_width: usize,
        kv_width: usize,
        feed_forward_length: usize,
        vocab_size: usize,
        decode_capacity: usize,
        moe_geometry: Option<MoeScratchGeometry>,
        qwen35_geometry: Option<Qwen35ScratchGeometry>,
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
            moe: moe_geometry
                .map(|geometry| CudaMoeDecodeScratch::allocate(device, geometry))
                .transpose()?,
            qwen35: qwen35_geometry
                .map(|geometry| CudaQwen35DecodeScratch::allocate(device, geometry))
                .transpose()?,
        })
    }

    fn matches_geometry(
        &self,
        embedding_length: usize,
        q_width: usize,
        kv_width: usize,
        feed_forward_length: usize,
        vocab_size: usize,
        moe_geometry: Option<MoeScratchGeometry>,
        qwen35_geometry: Option<Qwen35ScratchGeometry>,
    ) -> bool {
        self.embedding_length == embedding_length
            && self.q_width == q_width
            && self.kv_width == kv_width
            && self.feed_forward_length == feed_forward_length
            && self.vocab_size == vocab_size
            && self.moe.as_ref().map(|moe| moe.geometry) == moe_geometry
            && self.qwen35.as_ref().map(|scratch| scratch.geometry) == qwen35_geometry
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
        .saturating_add(
            self.moe
                .as_ref()
                .map_or(0, CudaMoeDecodeScratch::device_bytes),
        )
        .saturating_add(
            self.qwen35
                .as_ref()
                .map_or(0, CudaQwen35DecodeScratch::allocated_bytes),
        )
    }

    fn staging_bytes(&self) -> u64 {
        self.moe
            .as_ref()
            .map_or(0, CudaMoeDecodeScratch::staging_bytes)
    }

    fn clear_host_staging(&self) {
        if let Some(moe) = &self.moe {
            moe.clear_host();
        }
    }
}

impl Drop for CudaDecodeScratch {
    fn drop(&mut self) {
        // MoE graphs retain raw device pointers to parent and nested scratch
        // buffers. Destroy every executable before Rust drops those buffers.
        if let Some(moe) = &mut self.moe {
            moe.clear_graphs();
        }
    }
}

// Callers select a backend through this public enum, while CUDA graph and scratch
// implementation details remain intentionally opaque.
#[allow(private_interfaces)]
#[derive(Debug)]
#[allow(private_interfaces)]
pub enum BackendSession {
    Cpu {
        cache: SessionKvCache,
        recurrent: SessionRecurrentState,
    },
    Cuda {
        device: CudaDevice,
        capture_gate: Option<Arc<RwLock<()>>>,
        moe_graph_execution_gate: Option<Arc<Mutex<()>>>,
        allocation_arena: Option<Arc<GpuAllocationArena>>,
        kv_allocation: Option<GpuAllocationLease>,
        kv_cow_allocations: Vec<GpuAllocationLease>,
        scratch_allocation: Option<GpuAllocationLease>,
        staging_allocation: Option<GpuAllocationLease>,
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
        adaptive_route_horizon: Option<usize>,
        recurrent: SessionRecurrentState,
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

    fn cuda_adaptive_graph_suffix_is_hot_for_policy(
        policy: &SessionPolicy,
        prompt_token_count: usize,
        spans: &[PromptSpan],
        first_append_position: usize,
        total_len: usize,
    ) -> bool {
        first_append_position <= total_len
            && (first_append_position..total_len).all(|position| {
                Self::cuda_adaptive_position_is_hot_for_policy(
                    policy,
                    prompt_token_count,
                    spans,
                    position,
                    total_len,
                )
            })
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

    fn snapshot_shared_quantized_prefix(
        device: &CudaDevice,
        mode: KvCacheMode,
        layer_caches: &[CudaLayerKvStore],
        layer_widths: &[usize],
        prefix_len: usize,
        max_len: usize,
        page_tokens: usize,
        kv_budget_bytes: Option<u64>,
    ) -> Result<(Vec<CudaLayerKvStore>, u64)> {
        if !matches!(mode, KvCacheMode::Q8 | KvCacheMode::KeyQ4ValueQ8) {
            return Err(XrtError::Runtime(format!(
                "CUDA shared quantized prefix does not support mode {}",
                mode.as_str()
            )));
        }
        if layer_caches.len() != layer_widths.len() {
            return Err(XrtError::Runtime(format!(
                "cannot build shared {} prefix from {} caches for {} layer widths",
                mode.as_str(),
                layer_caches.len(),
                layer_widths.len()
            )));
        }

        let page_tokens = page_tokens.max(1);
        let page_capacity = max_len.div_ceil(page_tokens);
        let pages_per_layer = page_capacity.checked_add(1).ok_or_else(|| {
            XrtError::Runtime("CUDA shared quantized page capacity overflow".to_string())
        })?;
        let mut layers_per_width = HashMap::<usize, usize>::new();
        for &width in layer_widths {
            let count = layers_per_width.entry(width).or_default();
            *count = count.checked_add(1).ok_or_else(|| {
                XrtError::Runtime("CUDA shared quantized layer-count overflow".to_string())
            })?;
        }

        let mut reserved_bytes = 0u64;
        for (&width, &layer_count) in &layers_per_width {
            let max_pages = checked_mul(
                pages_per_layer,
                layer_count,
                "CUDA shared quantized bounded page count",
            )?;
            let page_bytes = cuda_layer_kv_allocated_bytes(mode, page_tokens, width, page_tokens)?
                .checked_sub(std::mem::size_of::<u32>() as u64)
                .ok_or_else(|| {
                    XrtError::Runtime("CUDA shared quantized page byte count underflow".to_string())
                })?;
            let arena_bytes = page_bytes.checked_mul(max_pages as u64).ok_or_else(|| {
                XrtError::Runtime("CUDA shared quantized arena byte count overflow".to_string())
            })?;
            let table_entries = checked_mul(
                page_capacity,
                layer_count,
                "CUDA shared quantized page-table entries",
            )?;
            let table_bytes = checked_mul(
                table_entries,
                std::mem::size_of::<u32>(),
                "CUDA shared quantized page-table bytes",
            )? as u64;
            reserved_bytes = reserved_bytes
                .checked_add(arena_bytes)
                .and_then(|bytes| bytes.checked_add(table_bytes))
                .ok_or_else(|| {
                    XrtError::Runtime(
                        "CUDA shared quantized reserved byte count overflow".to_string(),
                    )
                })?;
        }
        let source_bytes = layer_caches.iter().try_fold(0u64, |total, cache| {
            total.checked_add(cache.allocated_bytes()).ok_or_else(|| {
                XrtError::Runtime("CUDA quantized prefix source byte count overflow".to_string())
            })
        })?;
        let peak_bytes = source_bytes.checked_add(reserved_bytes).ok_or_else(|| {
            XrtError::Runtime("CUDA quantized prefix peak byte count overflow".to_string())
        })?;
        if let Some(budget_bytes) = kv_budget_bytes {
            if peak_bytes > budget_bytes {
                return Err(XrtError::Cuda(format!(
                    "CUDA shared {} prefix conversion requires {peak_bytes} peak KV bytes (source {source_bytes}, shared arena {reserved_bytes}), but the configured KV budget is {budget_bytes} bytes",
                    mode.as_str()
                )));
            }
        }

        let caches = match mode {
            KvCacheMode::Q8 => {
                let mut pools = HashMap::<usize, CudaQ8KvPagePool>::new();
                for (&width, &layer_count) in &layers_per_width {
                    let max_pages = checked_mul(
                        pages_per_layer,
                        layer_count,
                        "CUDA shared Q8 bounded page count",
                    )?;
                    pools.insert(
                        width,
                        CudaQ8KvPagePool::new(device, page_tokens, width, max_pages)?,
                    );
                }
                layer_caches
                    .iter()
                    .zip(layer_widths)
                    .map(|(cache, width)| {
                        let pool = pools.get(width).ok_or_else(|| {
                            XrtError::Runtime(format!(
                                "missing CUDA shared Q8 page pool for width {width}"
                            ))
                        })?;
                        cache.snapshot_q8_prefix_into_pool(pool, max_len, prefix_len)
                    })
                    .collect::<Result<Vec<_>>>()?
            }
            KvCacheMode::KeyQ4ValueQ8 => {
                let mut pools = HashMap::<usize, CudaKq4Vq8KvPagePool>::new();
                for (&width, &layer_count) in &layers_per_width {
                    let max_pages = checked_mul(
                        pages_per_layer,
                        layer_count,
                        "CUDA shared KQ4/VQ8 bounded page count",
                    )?;
                    pools.insert(
                        width,
                        CudaKq4Vq8KvPagePool::new(device, page_tokens, width, max_pages)?,
                    );
                }
                layer_caches
                    .iter()
                    .zip(layer_widths)
                    .map(|(cache, width)| {
                        let pool = pools.get(width).ok_or_else(|| {
                            XrtError::Runtime(format!(
                                "missing CUDA shared KQ4/VQ8 page pool for width {width}"
                            ))
                        })?;
                        cache.snapshot_kq4_vq8_prefix_into_pool(pool, max_len, prefix_len)
                    })
                    .collect::<Result<Vec<_>>>()?
            }
            _ => unreachable!("quantized prefix mode was validated above"),
        };
        Ok((caches, reserved_bytes))
    }

    fn shared_adaptive_reserved_bytes(
        layer_widths: &[usize],
        max_len: usize,
        page_tokens: usize,
    ) -> Result<u64> {
        if max_len == 0 {
            return Err(XrtError::Runtime(
                "CUDA shared adaptive KV requires a nonzero context length".to_string(),
            ));
        }
        let page_tokens = page_tokens.max(1);
        let page_capacity = max_len.div_ceil(page_tokens);
        let hot_pages_per_layer =
            checked_mul(page_capacity, 2, "CUDA shared adaptive hot pages per layer")?;
        let cold_pages_per_layer = page_capacity.checked_add(1).ok_or_else(|| {
            XrtError::Runtime("CUDA shared adaptive cold page capacity overflow".to_string())
        })?;

        layer_widths.iter().try_fold(0u64, |total, &width| {
            let hot_page_bytes =
                cuda_layer_kv_allocated_bytes(KvCacheMode::F32, page_tokens, width, page_tokens)?
                    .checked_sub((2 * std::mem::size_of::<u64>()) as u64)
                    .ok_or_else(|| {
                        XrtError::Runtime(
                            "CUDA shared adaptive hot page byte count underflow".to_string(),
                        )
                    })?;
            let cold_page_bytes = cuda_layer_kv_allocated_bytes(
                KvCacheMode::KeyQ4ValueQ8,
                page_tokens,
                width,
                page_tokens,
            )?
            .checked_sub(std::mem::size_of::<u32>() as u64)
            .ok_or_else(|| {
                XrtError::Runtime("CUDA shared adaptive cold page byte count underflow".to_string())
            })?;
            let hot_arena_bytes = hot_page_bytes
                .checked_mul(hot_pages_per_layer as u64)
                .ok_or_else(|| {
                    XrtError::Runtime(
                        "CUDA shared adaptive hot arena byte count overflow".to_string(),
                    )
                })?;
            let cold_arena_bytes = cold_page_bytes
                .checked_mul(cold_pages_per_layer as u64)
                .ok_or_else(|| {
                    XrtError::Runtime(
                        "CUDA shared adaptive cold arena byte count overflow".to_string(),
                    )
                })?;
            let hot_table_bytes = checked_mul(
                checked_mul(page_capacity, 2, "CUDA shared adaptive hot pointer entries")?,
                std::mem::size_of::<u64>(),
                "CUDA shared adaptive hot pointer bytes",
            )? as u64;
            let cold_table_bytes = checked_mul(
                page_capacity,
                std::mem::size_of::<u32>(),
                "CUDA shared adaptive cold page-table bytes",
            )? as u64;
            let route_bytes = checked_mul(
                max_len,
                std::mem::size_of::<u32>(),
                "CUDA shared adaptive route-table bytes",
            )? as u64;
            total
                .checked_add(hot_arena_bytes)
                .and_then(|bytes| bytes.checked_add(cold_arena_bytes))
                .and_then(|bytes| bytes.checked_add(hot_table_bytes))
                .and_then(|bytes| bytes.checked_add(cold_table_bytes))
                .and_then(|bytes| bytes.checked_add(route_bytes))
                .ok_or_else(|| {
                    XrtError::Runtime(
                        "CUDA shared adaptive reserved byte count overflow".to_string(),
                    )
                })
        })
    }

    fn snapshot_shared_adaptive_prefix(
        device: &CudaDevice,
        layer_caches: &[CudaLayerKvStore],
        layer_widths: &[usize],
        prefix_len: usize,
        max_len: usize,
        page_tokens: usize,
        kv_budget_bytes: Option<u64>,
    ) -> Result<(Vec<CudaLayerKvStore>, u64)> {
        if layer_caches.len() != layer_widths.len() {
            return Err(XrtError::Runtime(format!(
                "cannot build shared adaptive prefix from {} caches for {} layer widths",
                layer_caches.len(),
                layer_widths.len()
            )));
        }

        let page_tokens = page_tokens.max(1);
        let page_capacity = max_len.div_ceil(page_tokens);
        let hot_pages_per_layer =
            checked_mul(page_capacity, 2, "CUDA shared adaptive hot pages per layer")?;
        let cold_pages_per_layer = page_capacity.checked_add(1).ok_or_else(|| {
            XrtError::Runtime("CUDA shared adaptive cold page capacity overflow".to_string())
        })?;
        let reserved_bytes =
            Self::shared_adaptive_reserved_bytes(layer_widths, max_len, page_tokens)?;
        let source_bytes = layer_caches.iter().try_fold(0u64, |total, cache| {
            total.checked_add(cache.allocated_bytes()).ok_or_else(|| {
                XrtError::Runtime("CUDA adaptive prefix source byte count overflow".to_string())
            })
        })?;
        let peak_bytes = source_bytes.checked_add(reserved_bytes).ok_or_else(|| {
            XrtError::Runtime("CUDA adaptive prefix peak byte count overflow".to_string())
        })?;
        if let Some(budget_bytes) = kv_budget_bytes {
            if peak_bytes > budget_bytes {
                return Err(XrtError::Cuda(format!(
                    "CUDA shared agent-adaptive prefix conversion requires {peak_bytes} peak KV bytes (source {source_bytes}, shared arenas {reserved_bytes}), but the configured KV budget is {budget_bytes} bytes"
                )));
            }
        }

        let mut layers_per_width = HashMap::<usize, usize>::new();
        for &width in layer_widths {
            let count = layers_per_width.entry(width).or_default();
            *count = count.checked_add(1).ok_or_else(|| {
                XrtError::Runtime("CUDA shared adaptive layer-count overflow".to_string())
            })?;
        }
        let mut hot_pools = HashMap::<usize, CudaF32KvPagePool>::new();
        let mut cold_pools = HashMap::<usize, CudaKq4Vq8KvPagePool>::new();
        for (&width, &layer_count) in &layers_per_width {
            let hot_max_pages = checked_mul(
                hot_pages_per_layer,
                layer_count,
                "CUDA shared adaptive bounded hot page count",
            )?;
            let cold_max_pages = checked_mul(
                cold_pages_per_layer,
                layer_count,
                "CUDA shared adaptive bounded cold page count",
            )?;
            hot_pools.insert(
                width,
                CudaF32KvPagePool::new(device, page_tokens, width, hot_max_pages)?,
            );
            cold_pools.insert(
                width,
                CudaKq4Vq8KvPagePool::new(device, page_tokens, width, cold_max_pages)?,
            );
        }

        let caches = layer_caches
            .iter()
            .zip(layer_widths)
            .map(|(cache, width)| {
                let hot_pool = hot_pools.get(width).ok_or_else(|| {
                    XrtError::Runtime(format!(
                        "missing CUDA shared adaptive hot page pool for width {width}"
                    ))
                })?;
                let cold_pool = cold_pools.get(width).ok_or_else(|| {
                    XrtError::Runtime(format!(
                        "missing CUDA shared adaptive cold page pool for width {width}"
                    ))
                })?;
                cache.snapshot_adaptive_prefix_into_pools(hot_pool, cold_pool, max_len, prefix_len)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((caches, reserved_bytes))
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

    fn projected_shared_quantized_bytes(
        mode: KvCacheMode,
        layer_widths: &[usize],
        total_len: usize,
        max_len: usize,
        page_tokens: usize,
    ) -> Result<u64> {
        if !matches!(mode, KvCacheMode::Q8 | KvCacheMode::KeyQ4ValueQ8) {
            return Err(XrtError::Runtime(format!(
                "CUDA shared quantized projection does not support mode {}",
                mode.as_str()
            )));
        }
        let page_tokens = page_tokens.max(1);
        let resident_pages = total_len.div_ceil(page_tokens);
        let page_capacity = max_len.div_ceil(page_tokens);
        layer_widths.iter().try_fold(0u64, |total, &width| {
            let page_bytes = cuda_layer_kv_allocated_bytes(mode, page_tokens, width, page_tokens)?
                .checked_sub(std::mem::size_of::<u32>() as u64)
                .ok_or_else(|| {
                    XrtError::Runtime(
                        "CUDA shared quantized projected page bytes underflow".to_string(),
                    )
                })?;
            let referenced_bytes =
                page_bytes
                    .checked_mul(resident_pages as u64)
                    .ok_or_else(|| {
                        XrtError::Runtime(
                            "CUDA shared quantized referenced byte count overflow".to_string(),
                        )
                    })?;
            let table_bytes = checked_mul(
                page_capacity,
                std::mem::size_of::<u32>(),
                "CUDA shared quantized projected page-table bytes",
            )? as u64;
            total
                .checked_add(referenced_bytes)
                .and_then(|bytes| bytes.checked_add(table_bytes))
                .ok_or_else(|| {
                    XrtError::Runtime(
                        "CUDA shared quantized projected byte count overflow".to_string(),
                    )
                })
        })
    }

    fn projected_shared_adaptive_bytes(
        layer_widths: &[usize],
        total_len: usize,
        max_len: usize,
        page_tokens: usize,
    ) -> Result<u64> {
        let page_tokens = page_tokens.max(1);
        let resident_pages = total_len.div_ceil(page_tokens);
        let page_capacity = max_len.div_ceil(page_tokens);
        layer_widths.iter().try_fold(0u64, |total, &width| {
            let hot_page_bytes =
                cuda_layer_kv_allocated_bytes(KvCacheMode::F32, page_tokens, width, page_tokens)?
                    .checked_sub((2 * std::mem::size_of::<u64>()) as u64)
                    .ok_or_else(|| {
                        XrtError::Runtime(
                            "CUDA shared adaptive projected hot page bytes underflow".to_string(),
                        )
                    })?;
            let cold_page_bytes = cuda_layer_kv_allocated_bytes(
                KvCacheMode::KeyQ4ValueQ8,
                page_tokens,
                width,
                page_tokens,
            )?
            .checked_sub(std::mem::size_of::<u32>() as u64)
            .ok_or_else(|| {
                XrtError::Runtime(
                    "CUDA shared adaptive projected cold page bytes underflow".to_string(),
                )
            })?;
            let hot_referenced = hot_page_bytes
                .checked_mul(resident_pages as u64)
                .ok_or_else(|| {
                    XrtError::Runtime(
                        "CUDA shared adaptive projected hot bytes overflow".to_string(),
                    )
                })?;
            let cold_referenced = cold_page_bytes
                .checked_mul(resident_pages as u64)
                .ok_or_else(|| {
                    XrtError::Runtime(
                        "CUDA shared adaptive projected cold bytes overflow".to_string(),
                    )
                })?;
            let hot_table_bytes = checked_mul(
                checked_mul(
                    page_capacity,
                    2,
                    "CUDA shared adaptive projected hot pointer entries",
                )?,
                std::mem::size_of::<u64>(),
                "CUDA shared adaptive projected hot pointer bytes",
            )? as u64;
            let cold_table_bytes = checked_mul(
                page_capacity,
                std::mem::size_of::<u32>(),
                "CUDA shared adaptive projected cold page-table bytes",
            )? as u64;
            let route_bytes = checked_mul(
                max_len,
                std::mem::size_of::<u32>(),
                "CUDA shared adaptive projected route-table bytes",
            )? as u64;
            total
                .checked_add(hot_referenced)
                .and_then(|bytes| bytes.checked_add(cold_referenced))
                .and_then(|bytes| bytes.checked_add(hot_table_bytes))
                .and_then(|bytes| bytes.checked_add(cold_table_bytes))
                .and_then(|bytes| bytes.checked_add(route_bytes))
                .ok_or_else(|| {
                    XrtError::Runtime(
                        "CUDA shared adaptive projected byte count overflow".to_string(),
                    )
                })
        })
    }

    fn projected_shared_kv_bytes(
        mode: KvCacheMode,
        layer_widths: &[usize],
        total_len: usize,
        max_len: usize,
        page_tokens: usize,
    ) -> Result<u64> {
        match mode {
            KvCacheMode::F32 => {
                Self::projected_shared_f32_bytes(layer_widths, total_len, max_len, page_tokens)
            }
            KvCacheMode::Q8 | KvCacheMode::KeyQ4ValueQ8 => Self::projected_shared_quantized_bytes(
                mode,
                layer_widths,
                total_len,
                max_len,
                page_tokens,
            ),
            KvCacheMode::AgentAdaptive => {
                Self::projected_shared_adaptive_bytes(layer_widths, total_len, max_len, page_tokens)
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
            recurrent: SessionRecurrentState::None,
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
            capture_gate: None,
            moe_graph_execution_gate: None,
            allocation_arena: None,
            kv_allocation: None,
            kv_cow_allocations: Vec::new(),
            scratch_allocation: None,
            staging_allocation: None,
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
            adaptive_route_horizon: None,
            recurrent: SessionRecurrentState::None,
        }
    }

    fn set_initial_recurrent_state(&mut self, descriptor: Option<DeltaNetStateDescriptor>) {
        let recurrent = match self {
            Self::Cpu { recurrent, .. } | Self::Cuda { recurrent, .. } => recurrent,
        };
        *recurrent = SessionRecurrentState::from_descriptor(descriptor);
    }

    fn attach_gpu_allocation_arena(&mut self, arena: Arc<GpuAllocationArena>) -> Result<()> {
        match self {
            Self::Cpu { .. } => Err(XrtError::Runtime(
                "cannot attach a GPU allocation arena to a CPU session".to_string(),
            )),
            Self::Cuda {
                allocation_arena,
                decode_scratch,
                layer_caches,
                pending_prefix,
                ..
            } => {
                if decode_scratch.is_some() || !layer_caches.is_empty() || pending_prefix.is_some()
                {
                    return Err(XrtError::Runtime(
                        "GPU allocation arena must be attached before session CUDA allocations"
                            .to_string(),
                    ));
                }
                *allocation_arena = Some(arena);
                Ok(())
            }
        }
    }

    fn attach_cuda_capture_gate(&mut self, gate: Arc<RwLock<()>>) {
        if let Self::Cuda { capture_gate, .. } = self {
            *capture_gate = Some(gate);
        }
    }

    fn attach_moe_graph_execution_gate(&mut self, gate: Arc<Mutex<()>>) {
        if let Self::Cuda {
            moe_graph_execution_gate,
            ..
        } = self
        {
            *moe_graph_execution_gate = Some(gate);
        }
    }

    pub(crate) fn prepare_recurrent_state(&mut self) -> Result<()> {
        match self {
            Self::Cpu { recurrent, .. } => recurrent.prepare_cpu(),
            Self::Cuda {
                device,
                allocation_arena,
                recurrent,
                ..
            } => recurrent.prepare_cuda(device, allocation_arena.as_ref()),
        }
    }

    fn cuda_recurrent_state_mut(&mut self) -> Result<Option<&mut CudaDeltaNetState>> {
        match self {
            Self::Cuda { recurrent, .. } => recurrent.cuda_mut(),
            Self::Cpu { .. } => Err(XrtError::Runtime(
                "CUDA recurrent state requested from a CPU session".to_string(),
            )),
        }
    }

    pub(crate) fn recurrent_state_snapshot(&self) -> Result<Option<BackendStateSnapshot>> {
        let gate = self.cuda_capture_gate();
        let _capture_guard = gate.as_ref().map(|gate| gate.read());
        match self {
            Self::Cpu { recurrent, .. } | Self::Cuda { recurrent, .. } => recurrent.snapshot(),
        }
    }

    pub(crate) fn restore_recurrent_state(
        &mut self,
        snapshot: Option<&BackendStateSnapshot>,
        expected_position: usize,
    ) -> Result<()> {
        let gate = self.cuda_capture_gate();
        let _capture_guard = gate.as_ref().map(|gate| gate.read());
        match self {
            Self::Cpu { recurrent, .. } | Self::Cuda { recurrent, .. } => {
                recurrent.restore(snapshot, expected_position)
            }
        }
    }

    pub fn recurrent_state_allocated_bytes(&self) -> u64 {
        match self {
            Self::Cpu { recurrent, .. } | Self::Cuda { recurrent, .. } => {
                recurrent.allocated_bytes()
            }
        }
    }

    /// Returns whether this backend session has a device-local recurrent
    /// checkpoint journal suitable for speculative verification.
    pub fn supports_fast_recurrent_checkpoint(&self) -> bool {
        match self {
            Self::Cpu { recurrent, .. } | Self::Cuda { recurrent, .. } => {
                recurrent.supports_fast_checkpoint()
            }
        }
    }

    /// Copies the accepted recurrent boundary into the backend-local journal.
    pub fn begin_fast_recurrent_checkpoint(&mut self, expected_position: usize) -> Result<()> {
        let gate = self.cuda_capture_gate();
        let _capture_guard = gate.as_ref().map(|gate| gate.read());
        match self {
            Self::Cpu { recurrent, .. } | Self::Cuda { recurrent, .. } => {
                recurrent.begin_fast_checkpoint(expected_position)
            }
        }
    }

    /// Discards an active recurrent journal while retaining current state.
    pub fn commit_fast_recurrent_checkpoint(&mut self) -> Result<()> {
        match self {
            Self::Cpu { recurrent, .. } | Self::Cuda { recurrent, .. } => {
                recurrent.commit_fast_checkpoint()
            }
        }
    }

    /// Restores an active recurrent journal to `expected_position`.
    pub fn rollback_fast_recurrent_checkpoint(&mut self, expected_position: usize) -> Result<()> {
        let gate = self.cuda_capture_gate();
        let _capture_guard = gate.as_ref().map(|gate| gate.read());
        let result = match self {
            Self::Cpu { recurrent, .. } | Self::Cuda { recurrent, .. } => {
                recurrent.rollback_fast_checkpoint(expected_position)
            }
        };
        if let Err(error) = &result {
            self.poison_recurrent_state(format!(
                "device-local recurrent checkpoint rollback to {expected_position} failed: {error}"
            ));
        }
        result
    }

    fn recurrent_buffer_generation(&self) -> Option<u8> {
        match self {
            Self::Cpu { recurrent, .. } | Self::Cuda { recurrent, .. } => {
                recurrent.committed_buffer_generation()
            }
        }
    }

    fn cuda_capture_gate(&self) -> Option<Arc<RwLock<()>>> {
        match self {
            Self::Cpu { .. } => None,
            Self::Cuda { capture_gate, .. } => capture_gate.clone(),
        }
    }

    fn cuda_moe_graph_execution_gate(&self) -> Option<Arc<Mutex<()>>> {
        match self {
            Self::Cpu { .. } => None,
            Self::Cuda {
                moe_graph_execution_gate,
                ..
            } => moe_graph_execution_gate.clone(),
        }
    }

    /// Waits for all CUDA work owned by this session to finish.
    ///
    /// CPU sessions complete immediately.
    pub fn synchronize_cuda(&self) -> Result<()> {
        let gate = self.cuda_capture_gate();
        let _capture_guard = gate.as_ref().map(|gate| gate.read());
        match self {
            Self::Cpu { .. } => Ok(()),
            Self::Cuda { device, .. } => device.synchronize(),
        }
    }

    pub(crate) fn destroy_safely(self) -> Result<()> {
        // CUDA graph capture and async deallocation both use the device's
        // shared stream. Hold both backend gates until every session-owned
        // allocation has actually been destroyed, not merely until the
        // preceding work has synchronized.
        let moe_graph_execution_gate = self.cuda_moe_graph_execution_gate();
        let capture_gate = self.cuda_capture_gate();
        let _moe_graph_execution_guard = moe_graph_execution_gate.as_ref().map(|gate| gate.lock());
        let _capture_guard = capture_gate.as_ref().map(|gate| gate.read());
        let synchronized = match &self {
            Self::Cpu { .. } => Ok(()),
            Self::Cuda { device, .. } => device.synchronize(),
        };
        if let Err(error) = synchronized {
            // A failed synchronization means the driver may still reference
            // these allocations. Retiring them is safer than freeing memory
            // whose completion state is unknown.
            std::mem::forget(self);
            return Err(error);
        }
        drop(self);
        Ok(())
    }

    fn validate_recurrent_position(&self, expected_position: usize) -> Result<()> {
        match self {
            Self::Cpu { recurrent, .. } | Self::Cuda { recurrent, .. } => {
                recurrent.validate_position(expected_position)
            }
        }
    }

    pub(crate) fn poison_recurrent_state(&mut self, reason: String) {
        match self {
            Self::Cpu { recurrent, .. } | Self::Cuda { recurrent, .. } => recurrent.poison(reason),
        }
    }

    pub fn cache_mode(&self) -> KvCacheMode {
        match self {
            Self::Cpu { cache, .. } => cache.mode(),
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
        let gate = self.cuda_capture_gate();
        let _capture_guard = gate.as_ref().map(|gate| gate.read());
        let prefix_position = u64::try_from(prefix_len).map_err(|_| {
            XrtError::Runtime(format!(
                "prefix length {prefix_len} cannot be represented as a recurrent-state position"
            ))
        })?;
        match self {
            Self::Cpu { cache, recurrent } => {
                let recurrent_snapshot = recurrent.snapshot()?;
                if recurrent_snapshot
                    .as_ref()
                    .is_some_and(|snapshot| snapshot.position() != prefix_position)
                {
                    return Err(XrtError::Runtime(format!(
                        "cannot snapshot CPU prefix at {prefix_len} with recurrent state at {}",
                        recurrent_snapshot
                            .as_ref()
                            .map_or(0, BackendStateSnapshot::position)
                    )));
                }
                let cache = if let Some(snapshot) = recurrent_snapshot.as_ref() {
                    let empty_layers = snapshot
                        .descriptor()
                        .layers()
                        .iter()
                        .map(Option::is_some)
                        .collect::<Vec<_>>();
                    cache.snapshot_prefix_with_empty_layers(prefix_len, &empty_layers)?
                } else {
                    cache.snapshot_prefix(prefix_len)?
                };
                let allocated_bytes = cache.allocated_bytes().saturating_add(
                    recurrent_snapshot
                        .as_ref()
                        .map_or(0, BackendStateSnapshot::allocated_bytes),
                );
                Ok(Some(BackendPrefixSnapshot::Cpu {
                    cache,
                    recurrent: recurrent_snapshot,
                    prefix_len,
                    allocated_bytes,
                }))
            }
            Self::Cuda {
                device,
                allocation_arena,
                cache_mode,
                decode_graph,
                batch_graph_epoch,
                batch_graph_captured,
                layer_caches,
                kv_allocation,
                kv_cow_allocations,
                pending_prefix,
                layer_widths,
                max_len,
                page_tokens,
                kv_budget_bytes,
                recurrent,
                ..
            } => {
                if pending_prefix.is_some() {
                    return Err(XrtError::Runtime(
                        "cannot snapshot a CUDA prefix while another prefix is pending".to_string(),
                    ));
                }
                let recurrent_snapshot = recurrent.snapshot()?;
                if recurrent_snapshot
                    .as_ref()
                    .is_some_and(|snapshot| snapshot.position() != prefix_position)
                {
                    return Err(XrtError::Runtime(format!(
                        "cannot snapshot CUDA prefix at {prefix_len} with recurrent state at {}",
                        recurrent_snapshot
                            .as_ref()
                            .map_or(0, BackendStateSnapshot::position)
                    )));
                }
                let cache_lengths_match = layer_caches.len() == layer_widths.len()
                    && layer_caches.iter().enumerate().all(|(layer, cache)| {
                        let expected = recurrent_snapshot
                            .as_ref()
                            .and_then(|snapshot| snapshot.descriptor().layers().get(layer))
                            .is_some_and(Option::is_some)
                            .then_some(0)
                            .unwrap_or(prefix_len);
                        cache.len() == expected
                    });
                if !cache_lengths_match {
                    return Err(XrtError::Runtime(format!(
                        "cannot snapshot {prefix_len} CUDA prefix tokens from {} initialized layers",
                        layer_caches.len()
                    )));
                }
                if *cache_mode == KvCacheMode::F32 {
                    if layer_caches.iter().any(CudaLayerKvStore::is_shared_f32) {
                        decode_graph.reset();
                        *batch_graph_epoch = (*batch_graph_epoch).wrapping_add(1);
                        *batch_graph_captured = false;
                    }
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
                        .sum::<u64>();
                    let allocation = allocation_arena
                        .as_ref()
                        .filter(|_| allocated_bytes != 0)
                        .map(|arena| arena.reserve(GpuAllocationClass::KvCache, allocated_bytes))
                        .transpose()?;
                    let total_allocated_bytes = allocated_bytes.saturating_add(
                        recurrent_snapshot
                            .as_ref()
                            .map_or(0, BackendStateSnapshot::allocated_bytes),
                    );
                    return Ok(Some(BackendPrefixSnapshot::Cuda {
                        layer_caches: snapshot_caches,
                        allocation,
                        cow_allocations: Vec::new(),
                        cache_mode: *cache_mode,
                        layer_widths: layer_widths.clone(),
                        page_tokens: *page_tokens,
                        recurrent: recurrent_snapshot,
                        prefix_len,
                        allocated_bytes: total_allocated_bytes,
                    }));
                }
                if matches!(*cache_mode, KvCacheMode::Q8 | KvCacheMode::KeyQ4ValueQ8) {
                    if (*cache_mode == KvCacheMode::Q8
                        && layer_caches.iter().any(CudaLayerKvStore::is_shared_q8))
                        || (*cache_mode == KvCacheMode::KeyQ4ValueQ8
                            && layer_caches.iter().any(CudaLayerKvStore::is_shared_kq4_vq8))
                    {
                        decode_graph.reset();
                        *batch_graph_epoch = (*batch_graph_epoch).wrapping_add(1);
                        *batch_graph_captured = false;
                    }
                    let (snapshot_caches, allocated_bytes) =
                        Self::snapshot_shared_quantized_prefix(
                            device,
                            *cache_mode,
                            layer_caches,
                            layer_widths,
                            prefix_len,
                            *max_len,
                            *page_tokens,
                            *kv_budget_bytes,
                        )?;
                    let allocation = allocation_arena
                        .as_ref()
                        .filter(|_| allocated_bytes != 0)
                        .map(|arena| arena.reserve(GpuAllocationClass::KvCache, allocated_bytes))
                        .transpose()?;
                    let total_allocated_bytes = allocated_bytes.saturating_add(
                        recurrent_snapshot
                            .as_ref()
                            .map_or(0, BackendStateSnapshot::allocated_bytes),
                    );
                    return Ok(Some(BackendPrefixSnapshot::Cuda {
                        layer_caches: Arc::new(snapshot_caches),
                        allocation,
                        cow_allocations: Vec::new(),
                        cache_mode: *cache_mode,
                        layer_widths: layer_widths.clone(),
                        page_tokens: *page_tokens,
                        recurrent: recurrent_snapshot,
                        prefix_len,
                        allocated_bytes: total_allocated_bytes,
                    }));
                }
                if *cache_mode == KvCacheMode::AgentAdaptive {
                    if layer_caches
                        .iter()
                        .any(CudaLayerKvStore::is_shared_adaptive)
                    {
                        decode_graph.reset();
                        *batch_graph_epoch = (*batch_graph_epoch).wrapping_add(1);
                        *batch_graph_captured = false;
                    }
                    let (snapshot_caches, allocated_bytes) = Self::snapshot_shared_adaptive_prefix(
                        device,
                        layer_caches,
                        layer_widths,
                        prefix_len,
                        *max_len,
                        *page_tokens,
                        *kv_budget_bytes,
                    )?;
                    let allocation = allocation_arena
                        .as_ref()
                        .filter(|_| allocated_bytes != 0)
                        .map(|arena| arena.reserve(GpuAllocationClass::KvCache, allocated_bytes))
                        .transpose()?;
                    let total_allocated_bytes = allocated_bytes.saturating_add(
                        recurrent_snapshot
                            .as_ref()
                            .map_or(0, BackendStateSnapshot::allocated_bytes),
                    );
                    return Ok(Some(BackendPrefixSnapshot::Cuda {
                        layer_caches: Arc::new(snapshot_caches),
                        allocation,
                        cow_allocations: Vec::new(),
                        cache_mode: *cache_mode,
                        layer_widths: layer_widths.clone(),
                        page_tokens: *page_tokens,
                        recurrent: recurrent_snapshot,
                        prefix_len,
                        allocated_bytes: total_allocated_bytes,
                    }));
                }
                let allocated_bytes = layer_caches
                    .iter()
                    .map(CudaLayerKvStore::allocated_bytes)
                    .sum::<u64>()
                    .saturating_add(
                        recurrent_snapshot
                            .as_ref()
                            .map_or(0, BackendStateSnapshot::allocated_bytes),
                    );
                let layer_caches = Arc::new(std::mem::take(layer_caches));
                *pending_prefix = Some(layer_caches.clone());
                decode_graph.reset();
                *batch_graph_epoch = (*batch_graph_epoch).wrapping_add(1);
                *batch_graph_captured = false;
                Ok(Some(BackendPrefixSnapshot::Cuda {
                    layer_caches,
                    allocation: kv_allocation.clone(),
                    cow_allocations: kv_cow_allocations.clone(),
                    cache_mode: *cache_mode,
                    layer_widths: layer_widths.clone(),
                    page_tokens: *page_tokens,
                    recurrent: recurrent_snapshot,
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
        let gate = self.cuda_capture_gate();
        let _capture_guard = gate.as_ref().map(|gate| gate.read());
        match (self, snapshot) {
            (
                Self::Cpu { cache, recurrent },
                BackendPrefixSnapshot::Cpu {
                    cache: snapshot_cache,
                    recurrent: snapshot_recurrent,
                    prefix_len,
                    ..
                },
            ) if cache.geometry_matches(snapshot_cache) => {
                recurrent.prepare_cpu()?;
                recurrent.restore(snapshot_recurrent.as_ref(), *prefix_len)?;
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
                    kv_allocation,
                    kv_cow_allocations,
                    pending_prefix,
                    layer_widths,
                    max_len,
                    page_tokens,
                    device,
                    allocation_arena,
                    recurrent,
                    ..
                },
                BackendPrefixSnapshot::Cuda {
                    layer_caches: snapshot_caches,
                    allocation: snapshot_allocation,
                    cow_allocations: snapshot_cow_allocations,
                    cache_mode: snapshot_mode,
                    layer_widths: snapshot_widths,
                    page_tokens: snapshot_page_tokens,
                    recurrent: snapshot_recurrent,
                    prefix_len,
                    ..
                },
            ) if cache_mode == snapshot_mode
                && layer_widths == snapshot_widths
                && page_tokens == snapshot_page_tokens
                && *prefix_len <= *max_len =>
            {
                recurrent.prepare_cuda(device, allocation_arena.as_ref())?;
                recurrent.restore(snapshot_recurrent.as_ref(), *prefix_len)?;
                layer_caches.clear();
                *pending_prefix = Some(snapshot_caches.clone());
                *kv_allocation = snapshot_allocation.clone();
                *kv_cow_allocations = snapshot_cow_allocations.clone();
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
            Self::Cpu { cache, .. } => cache.mode(),
            Self::Cuda {
                requested_cache_mode,
                ..
            } => *requested_cache_mode,
        }
    }

    fn configure_cuda_graph_mode(&mut self, mode: CudaGraphMode) {
        if let Self::Cuda {
            decode_graph,
            decode_scratch,
            ..
        } = self
        {
            *decode_graph = CudaDecodeGraphState::new(mode);
            if let Some(moe) = decode_scratch
                .as_mut()
                .and_then(|scratch| scratch.moe.as_mut())
            {
                moe.configure_graph_mode(mode);
            }
        }
    }

    pub fn cuda_graph_capture_status(&self) -> Option<&'static str> {
        match self {
            Self::Cpu { .. } => None,
            Self::Cuda {
                decode_graph,
                batch_graph_captured,
                decode_scratch,
                ..
            } => Some(if *batch_graph_captured {
                "batch-captured"
            } else if let Some(moe) = decode_scratch
                .as_ref()
                .and_then(|scratch| scratch.moe.as_ref())
            {
                moe.graph_capture_state.as_str()
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
            Self::Cuda {
                decode_graph,
                decode_scratch,
                ..
            } => decode_scratch
                .as_ref()
                .and_then(|scratch| scratch.moe.as_ref())
                .and_then(|moe| moe.graph_last_error.as_deref())
                .or(decode_graph.last_error.as_deref()),
        }
    }

    pub fn cuda_adaptive_position_is_hot(&self, position: usize, total_len: usize) -> bool {
        match self {
            Self::Cuda {
                requested_cache_mode: KvCacheMode::AgentAdaptive,
                policy,
                prompt_token_count,
                prompt_spans,
                adaptive_route_horizon,
                ..
            } => {
                let routing_total_len =
                    adaptive_route_horizon.map_or(total_len, |horizon| total_len.max(horizon));
                Self::cuda_adaptive_position_is_hot_for_policy(
                    policy,
                    *prompt_token_count,
                    prompt_spans,
                    position,
                    routing_total_len,
                )
            }
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
                adaptive_route_horizon,
                ..
            } => {
                let routing_total_len =
                    adaptive_route_horizon.map_or(total_len, |horizon| total_len.max(horizon));
                Some(Self::cuda_adaptive_hot_position_mask_for_policy(
                    policy,
                    *prompt_token_count,
                    prompt_spans,
                    routing_total_len,
                ))
            }
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
                kv_allocation,
                kv_cow_allocations,
                pending_prefix,
                layer_count: current_layer_count,
                width: current_width,
                layer_widths,
                page_tokens: current_page_tokens,
                policy,
                prompt_token_count,
                prompt_spans,
                adaptive_route_horizon,
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
                *adaptive_route_horizon = None;
                *pending_prefix = None;
                if layer_caches.is_empty() {
                    *kv_allocation = None;
                    kv_cow_allocations.clear();
                }
                decode_graph.reset();
                if layout_changed {
                    *batch_graph_epoch = (*batch_graph_epoch).wrapping_add(1);
                    *batch_graph_captured = false;
                    layer_caches.clear();
                    *kv_allocation = None;
                    kv_cow_allocations.clear();
                } else if requested_changed {
                    for cache in layer_caches.iter_mut() {
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
            Self::Cpu { cache, .. } => cache.configure_policy(policy, prompt_token_count, spans),
            Self::Cuda {
                requested_cache_mode,
                decode_graph,
                policy: cuda_policy,
                prompt_token_count: cuda_prompt_token_count,
                prompt_spans,
                adaptive_route_horizon,
                ..
            } => {
                *cuda_policy = policy;
                *cuda_prompt_token_count = prompt_token_count;
                prompt_spans.clear();
                prompt_spans.extend_from_slice(spans);
                *adaptive_route_horizon = None;
                if *requested_cache_mode == KvCacheMode::AgentAdaptive {
                    decode_graph.reset();
                }
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
        let gate = self.cuda_capture_gate();
        let _capture_guard = gate.as_ref().map(|gate| gate.read());
        self.prepare_for_total_len_inner(total_len, true)
    }

    fn prepare_for_total_len_inner(
        &mut self,
        total_len: usize,
        make_append_range_writable: bool,
    ) -> Result<()> {
        match self {
            Self::Cpu { cache, .. } => cache.prepare_for_total_len(total_len),
            Self::Cuda {
                device,
                allocation_arena,
                cache_mode,
                decode_graph,
                batch_graph_epoch,
                batch_graph_captured,
                layer_caches,
                kv_allocation,
                kv_cow_allocations,
                pending_prefix,
                layer_widths,
                max_len,
                page_tokens,
                kv_budget_bytes,
                policy,
                prompt_token_count,
                prompt_spans,
                adaptive_route_horizon,
                recurrent,
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
                    let share_f32_pages = *cache_mode == KvCacheMode::F32
                        && snapshot_caches.iter().all(
                            |cache| matches!(cache, CudaLayerKvStore::F32(cache) if cache.is_shared_pages()),
                        );
                    if share_f32_pages {
                        let private_bytes =
                            snapshot_caches.iter().try_fold(0u64, |total, cache| {
                                total
                                    .checked_add(
                                        cache
                                            .shared_clone_private_bytes(device, target_capacity)?,
                                    )
                                    .ok_or_else(|| {
                                        XrtError::Runtime(
                                            "CUDA shared prefix private byte count overflow"
                                                .to_string(),
                                        )
                                    })
                            })?;
                        let peak_bytes =
                            snapshot_bytes.checked_add(private_bytes).ok_or_else(|| {
                                XrtError::Runtime(
                                    "CUDA shared prefix attach peak byte count overflow"
                                        .to_string(),
                                )
                            })?;
                        if kv_budget_bytes.is_some_and(|budget| peak_bytes > budget) {
                            *pending_prefix = Some(snapshot_caches);
                            return Err(XrtError::Cuda(format!(
                                "CUDA shared prefix attach requires {peak_bytes} peak KV bytes (snapshot {snapshot_bytes}, private pointer/new-page storage {private_bytes}), but the configured KV budget is {} bytes",
                                kv_budget_bytes.unwrap_or_default()
                            )));
                        }
                        let private_lease_result = allocation_arena
                            .as_ref()
                            .filter(|_| private_bytes != 0)
                            .map(|arena| arena.reserve(GpuAllocationClass::KvCache, private_bytes))
                            .transpose();
                        let private_lease = match private_lease_result {
                            Ok(lease) => lease,
                            Err(err) => {
                                *pending_prefix = Some(snapshot_caches);
                                return Err(err);
                            }
                        };
                        let materialized = snapshot_caches
                            .iter()
                            .map(|cache| cache.share_with_capacity(device, target_capacity))
                            .collect::<Result<Vec<_>>>();
                        match materialized {
                            Ok(caches) => {
                                *layer_caches = caches;
                                if let Some(lease) = private_lease {
                                    kv_cow_allocations.push(lease);
                                }
                                decode_graph.reset();
                                *batch_graph_epoch = (*batch_graph_epoch).wrapping_add(1);
                                *batch_graph_captured = false;
                            }
                            Err(err) => {
                                *pending_prefix = Some(snapshot_caches);
                                return Err(err);
                            }
                        }
                    } else {
                        let required_bytes = if uses_shared_pages {
                            Self::projected_shared_kv_bytes(
                                *cache_mode,
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
                        let peak_bytes =
                            snapshot_bytes.checked_add(required_bytes).ok_or_else(|| {
                                XrtError::Runtime(
                                    "CUDA prefix attach peak byte count overflow".to_string(),
                                )
                            })?;
                        if kv_budget_bytes.is_some_and(|budget| peak_bytes > budget) {
                            *pending_prefix = Some(snapshot_caches);
                            return Err(XrtError::Cuda(format!(
                                "CUDA prefix attach requires {peak_bytes} peak KV bytes (snapshot {snapshot_bytes}, mutable copy {required_bytes}), but the configured KV budget is {} bytes",
                                kv_budget_bytes.unwrap_or_default()
                            )));
                        }
                        let replacement_lease_result = allocation_arena
                            .as_ref()
                            .map(|arena| arena.reserve(GpuAllocationClass::KvCache, required_bytes))
                            .transpose();
                        let replacement_lease = match replacement_lease_result {
                            Ok(lease) => lease,
                            Err(err) => {
                                *pending_prefix = Some(snapshot_caches);
                                return Err(err);
                            }
                        };

                        let materialized = snapshot_caches
                            .iter()
                            .map(|cache| cache.deep_clone_with_capacity(device, target_capacity))
                            .collect::<Result<Vec<_>>>();
                        match materialized {
                            Ok(caches) => {
                                let materialized_shared =
                                    caches.iter().any(CudaLayerKvStore::uses_shared_pages);
                                let shared_graph_eligible = (*cache_mode == KvCacheMode::F32
                                    && caches.iter().all(CudaLayerKvStore::is_shared_f32))
                                    || (*cache_mode == KvCacheMode::Q8
                                        && caches.iter().all(CudaLayerKvStore::is_shared_q8))
                                    || (*cache_mode == KvCacheMode::KeyQ4ValueQ8
                                        && caches.iter().all(CudaLayerKvStore::is_shared_kq4_vq8))
                                    || (*cache_mode == KvCacheMode::AgentAdaptive
                                        && caches.iter().all(CudaLayerKvStore::is_shared_adaptive));
                                *layer_caches = caches;
                                *kv_allocation = replacement_lease;
                                kv_cow_allocations.clear();
                                if materialized_shared && !shared_graph_eligible {
                                    decode_graph.fallback(
                                        "CUDA Graph decode for this runtime-attached shared KV layout is not wired yet",
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
                }
                let uses_shared_pages =
                    layer_caches.iter().any(CudaLayerKvStore::uses_shared_pages);
                if uses_shared_pages {
                    if layer_caches.iter().any(|cache| !cache.uses_shared_pages()) {
                        return Err(XrtError::Runtime(
                            "CUDA session mixes shared and contiguous layer caches".to_string(),
                        ));
                    }
                    let required_bytes = Self::projected_shared_kv_bytes(
                        *cache_mode,
                        layer_widths,
                        total_len,
                        *max_len,
                        *page_tokens,
                    )?;
                    if let Some(budget_bytes) = kv_budget_bytes {
                        if required_bytes > *budget_bytes {
                            return Err(XrtError::Cuda(format!(
                                "CUDA shared {} KV cache requires {required_bytes} bytes for {total_len} tokens, but the configured KV budget is {budget_bytes} bytes",
                                cache_mode.as_str()
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
                    let replacement_lease = allocation_arena
                        .as_ref()
                        .map(|arena| arena.reserve(GpuAllocationClass::KvCache, required_bytes))
                        .transpose()?;
                    let replacement_caches = if layer_caches.is_empty() {
                        layer_widths
                            .iter()
                            .map(|&layer_width| {
                                CudaLayerKvStore::allocate(
                                    device,
                                    *cache_mode,
                                    target_capacity,
                                    layer_width,
                                    *page_tokens,
                                )
                            })
                            .collect::<Result<Vec<_>>>()
                    } else {
                        layer_caches
                            .iter()
                            .map(|cache| cache.deep_clone_with_capacity(device, target_capacity))
                            .collect::<Result<Vec<_>>>()
                    }?;
                    *batch_graph_epoch = (*batch_graph_epoch).wrapping_add(1);
                    *batch_graph_captured = false;
                    *layer_caches = replacement_caches;
                    *kv_allocation = replacement_lease;
                    kv_cow_allocations.clear();
                }
                if *cache_mode == KvCacheMode::AgentAdaptive && !layer_caches.is_empty() {
                    let routing_total_len =
                        adaptive_route_horizon.map_or(total_len, |horizon| total_len.max(horizon));
                    let desired_hot_mask = Self::cuda_adaptive_hot_position_mask_for_policy(
                        policy,
                        *prompt_token_count,
                        prompt_spans,
                        routing_total_len,
                    );
                    for cache in layer_caches.iter_mut() {
                        cache.migrate_agent_adaptive_route(device, &desired_hot_mask)?;
                    }
                }
                if make_append_range_writable && *cache_mode == KvCacheMode::F32 {
                    let cow_bytes = layer_caches.iter().enumerate().try_fold(
                        0u64,
                        |total, (layer, cache)| {
                            if recurrent.layer_uses_recurrent_state(layer) {
                                return Ok(total);
                            }
                            total
                                .checked_add(cache.cow_bytes_for_range(cache.len(), total_len)?)
                                .ok_or_else(|| {
                                    XrtError::Runtime(
                                        "CUDA shared prefix COW byte count overflow".to_string(),
                                    )
                                })
                        },
                    )?;
                    if cow_bytes != 0 {
                        let accounted_bytes = kv_allocation
                            .as_ref()
                            .map_or(0, GpuAllocationLease::bytes)
                            .checked_add(kv_cow_allocations.iter().try_fold(
                                0u64,
                                |total, lease| {
                                    total.checked_add(lease.bytes()).ok_or_else(|| {
                                        XrtError::Runtime(
                                            "CUDA shared prefix lease byte count overflow"
                                                .to_string(),
                                        )
                                    })
                                },
                            )?)
                            .and_then(|bytes| bytes.checked_add(cow_bytes))
                            .ok_or_else(|| {
                                XrtError::Runtime(
                                    "CUDA shared prefix accounted byte count overflow".to_string(),
                                )
                            })?;
                        if kv_budget_bytes
                            .is_some_and(|budget_bytes| accounted_bytes > budget_bytes)
                        {
                            return Err(XrtError::Cuda(format!(
                                "CUDA shared prefix COW requires {cow_bytes} additional bytes ({accounted_bytes} accounted), exceeding the configured {}-byte KV budget",
                                kv_budget_bytes.unwrap_or_default()
                            )));
                        }
                        let cow_lease = allocation_arena
                            .as_ref()
                            .map(|arena| arena.reserve(GpuAllocationClass::KvCache, cow_bytes))
                            .transpose()?;
                        if let Some(lease) = cow_lease {
                            kv_cow_allocations.push(lease);
                        }
                        for (layer, cache) in layer_caches.iter_mut().enumerate() {
                            if recurrent.layer_uses_recurrent_state(layer) {
                                continue;
                            }
                            cache.ensure_writable_range(device, cache.len(), total_len)?;
                        }
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
                if matches!(*cache_mode, KvCacheMode::Q8 | KvCacheMode::KeyQ4ValueQ8) {
                    let caches = pending_prefix
                        .as_ref()
                        .map(|caches| caches.as_slice())
                        .unwrap_or(layer_caches.as_slice());
                    let homogeneous_shared = match *cache_mode {
                        KvCacheMode::Q8 => caches.iter().all(CudaLayerKvStore::is_shared_q8),
                        KvCacheMode::KeyQ4ValueQ8 => {
                            caches.iter().all(CudaLayerKvStore::is_shared_kq4_vq8)
                        }
                        _ => unreachable!("quantized graph readiness mode changed"),
                    };
                    if caches.is_empty() || !homogeneous_shared {
                        decode_graph.fallback(format!(
                            "CUDA Graph {} decode currently requires homogeneous runtime-shared {} KV pages",
                            cache_mode.as_str(),
                            cache_mode.as_str()
                        ));
                        return false;
                    }
                    return true;
                }
                if *cache_mode == KvCacheMode::AgentAdaptive {
                    let caches = pending_prefix
                        .as_ref()
                        .map(|caches| caches.as_slice())
                        .unwrap_or(layer_caches.as_slice());
                    if caches.is_empty() || !caches.iter().all(CudaLayerKvStore::is_shared_adaptive)
                    {
                        decode_graph.fallback(
                            "CUDA Graph agent_adaptive decode requires homogeneous runtime-shared adaptive KV pages",
                        );
                        return false;
                    }
                    return true;
                }
                if *cache_mode != KvCacheMode::F32 {
                    decode_graph.fallback(format!(
                        "CUDA Graph decode currently requires f32 or runtime-shared q8/kq4-vq8/adaptive KV, found {}",
                        cache_mode.as_str()
                    ));
                    return false;
                }
                if let Some(caches) = pending_prefix {
                    if caches.iter().any(CudaLayerKvStore::uses_shared_pages)
                        && caches.iter().any(|cache| !cache.uses_shared_pages())
                    {
                        decode_graph.fallback(
                            "CUDA Graph decode cannot attach a mixed shared/contiguous prefix",
                        );
                        return false;
                    }
                    if caches.iter().any(CudaLayerKvStore::uses_shared_pages)
                        && caches.iter().any(|cache| !cache.is_shared_f32())
                    {
                        decode_graph.fallback(
                            "CUDA Graph decode for pending quantized or adaptive shared KV pages is not wired yet",
                        );
                        return false;
                    }
                    return true;
                }
                let shared_f32_layers = layer_caches
                    .iter()
                    .filter(|cache| cache.is_shared_f32())
                    .count();
                if shared_f32_layers != 0 && shared_f32_layers != layer_caches.len() {
                    decode_graph.fallback(
                        "CUDA Graph decode cannot mix shared and contiguous F32 KV layers",
                    );
                    return false;
                }
                true
            }
        }
    }

    fn prepare_cuda_adaptive_graph_horizon(&mut self, total_len: usize) -> Result<()> {
        let Self::Cuda {
            cache_mode,
            layer_caches,
            pending_prefix,
            policy,
            prompt_token_count,
            prompt_spans,
            adaptive_route_horizon,
            ..
        } = self
        else {
            return Ok(());
        };
        if *cache_mode != KvCacheMode::AgentAdaptive {
            return Ok(());
        }

        let caches = pending_prefix
            .as_ref()
            .map(|caches| caches.as_slice())
            .unwrap_or(layer_caches.as_slice());
        let first_append_position = caches.first().map(CudaLayerKvStore::len).ok_or_else(|| {
            XrtError::Unsupported(
                "CUDA Graph agent_adaptive decode requires an attached shared prefix".to_string(),
            )
        })?;
        if caches
            .iter()
            .any(|cache| !cache.is_shared_adaptive() || cache.len() != first_append_position)
        {
            return Err(XrtError::Unsupported(
                "CUDA Graph agent_adaptive decode requires homogeneous shared layers with equal lengths"
                    .to_string(),
            ));
        }
        if !Self::cuda_adaptive_graph_suffix_is_hot_for_policy(
            policy,
            *prompt_token_count,
            prompt_spans,
            first_append_position,
            total_len,
        ) {
            return Err(XrtError::Unsupported(format!(
                "CUDA Graph agent_adaptive decode requires every position {first_append_position}..{total_len} to remain in the final recent-window hot tier"
            )));
        }
        *adaptive_route_horizon = Some(total_len);
        Ok(())
    }

    pub(crate) fn prepare_cuda_graph_generation_capacity(&mut self, total_len: usize) -> bool {
        if !self.cuda_graph_decode_ready() {
            return false;
        }
        if let Err(err) = self.prepare_cuda_adaptive_graph_horizon(total_len) {
            self.cuda_graph_fallback(err.to_string());
            tracing::warn!(
                "bounded CUDA Graph adaptive route preparation failed; using eager CUDA: {err}"
            );
            return false;
        }
        let gate = self.cuda_capture_gate();
        let _capture_guard = gate.as_ref().map(|gate| gate.read());
        let prepared =
            self.prepare_for_total_len_inner(total_len, false)
                .and_then(|()| match self {
                    Self::Cpu { .. } => Ok(()),
                    Self::Cuda {
                        decode_graph,
                        batch_graph_epoch,
                        batch_graph_captured,
                        layer_caches,
                        ..
                    } => {
                        let mut topology_changed = false;
                        for cache in layer_caches.iter_mut() {
                            topology_changed |=
                                cache.prepare_shared_f32_graph_capacity(total_len)?;
                            topology_changed |=
                                cache.prepare_shared_q8_graph_capacity(total_len)?;
                            topology_changed |=
                                cache.prepare_shared_kq4_vq8_graph_capacity(total_len)?;
                            topology_changed |=
                                cache.prepare_shared_adaptive_graph_capacity(total_len)?;
                        }
                        if topology_changed {
                            decode_graph.reset();
                            *batch_graph_epoch = (*batch_graph_epoch).wrapping_add(1);
                            *batch_graph_captured = false;
                        }
                        Ok(())
                    }
                });
        match prepared {
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

    fn prepare_cuda_graph_append_position(&mut self, position: usize) -> Result<()> {
        let total_len = cuda_total_len_for_position(position)?;
        match self {
            Self::Cpu { .. } => Ok(()),
            Self::Cuda {
                decode_graph,
                batch_graph_epoch,
                batch_graph_captured,
                layer_caches,
                ..
            } => {
                if decode_graph.executable.is_some() {
                    return Ok(());
                }
                let mut topology_changed = false;
                for cache in layer_caches.iter_mut() {
                    topology_changed |= cache.prepare_shared_f32_graph_capacity(total_len)?;
                    topology_changed |= cache.prepare_shared_q8_graph_capacity(total_len)?;
                    topology_changed |= cache.prepare_shared_kq4_vq8_graph_capacity(total_len)?;
                    topology_changed |= cache.prepare_shared_adaptive_graph_capacity(total_len)?;
                }
                if topology_changed {
                    decode_graph.reset();
                    *batch_graph_epoch = (*batch_graph_epoch).wrapping_add(1);
                    *batch_graph_captured = false;
                }
                Ok(())
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

    fn cuda_graph_uses_shared_f32(&self) -> bool {
        match self {
            Self::Cpu { .. } => false,
            Self::Cuda {
                layer_caches,
                pending_prefix,
                ..
            } => {
                layer_caches.iter().any(CudaLayerKvStore::is_shared_f32)
                    || pending_prefix
                        .as_ref()
                        .is_some_and(|caches| caches.iter().any(CudaLayerKvStore::is_shared_f32))
            }
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

    fn cuda_allocation_arena(&self) -> Option<Arc<GpuAllocationArena>> {
        match self {
            Self::Cpu { .. } => None,
            Self::Cuda {
                allocation_arena, ..
            } => allocation_arena.clone(),
        }
    }

    fn cuda_graph_executable(&self) -> Option<&CudaGraphExec> {
        match self {
            Self::Cpu { .. } => None,
            Self::Cuda { decode_graph, .. } => decode_graph.executable.as_ref(),
        }
    }

    fn cuda_graph_has_executable_for(&self, key: &CudaDecodeGraphKey) -> bool {
        match self {
            Self::Cpu { .. } => false,
            Self::Cuda { decode_graph, .. } => decode_graph.has_executable_for(key),
        }
    }

    fn cuda_graph_fallback(&mut self, error: impl Into<String>) {
        if let Self::Cuda { decode_graph, .. } = self {
            decode_graph.fallback(error);
        }
    }

    pub fn clear(&mut self) {
        match self {
            Self::Cpu { cache, recurrent } => {
                cache.clear();
                recurrent.clear();
            }
            Self::Cuda {
                decode_graph,
                batch_graph_epoch,
                batch_graph_captured,
                layer_caches,
                kv_allocation,
                kv_cow_allocations,
                pending_prefix,
                adaptive_route_horizon,
                decode_scratch,
                recurrent,
                ..
            } => {
                let invalidates_shared_graph =
                    layer_caches.iter().any(CudaLayerKvStore::uses_shared_pages)
                        || pending_prefix.as_ref().is_some_and(|caches| {
                            caches.iter().any(CudaLayerKvStore::uses_shared_pages)
                        });
                if invalidates_shared_graph {
                    decode_graph.reset();
                    *batch_graph_epoch = (*batch_graph_epoch).wrapping_add(1);
                    *batch_graph_captured = false;
                }
                if let Some(scratch) = decode_scratch {
                    scratch.clear_host_staging();
                }
                *pending_prefix = None;
                *adaptive_route_horizon = None;
                if layer_caches.is_empty() {
                    *kv_allocation = None;
                    kv_cow_allocations.clear();
                }
                for cache in layer_caches {
                    cache.clear();
                }
                recurrent.clear();
            }
        }
    }

    pub fn truncate(&mut self, new_len: usize) -> Result<()> {
        match self {
            Self::Cpu { cache, .. } => {
                cache.truncate(new_len);
                Ok(())
            }
            Self::Cuda {
                decode_graph,
                batch_graph_epoch,
                batch_graph_captured,
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
                if layer_caches
                    .iter()
                    .any(|cache| cache.uses_shared_pages() && new_len < cache.len())
                {
                    decode_graph.reset();
                    *batch_graph_epoch = (*batch_graph_epoch).wrapping_add(1);
                    *batch_graph_captured = false;
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
            Self::Cpu { cache, .. } => Ok(cache),
            Self::Cuda { .. } => Err(XrtError::Runtime(
                "CPU KV cache requested from CUDA backend session".to_string(),
            )),
        }
    }

    fn cpu_cache_and_recurrent_mut(
        &mut self,
    ) -> Result<(&mut SessionKvCache, Option<&mut DeltaNetState>)> {
        match self {
            Self::Cpu { cache, recurrent } => Ok((cache, recurrent.cpu_mut()?)),
            Self::Cuda { .. } => Err(XrtError::Runtime(
                "CPU state requested from CUDA backend session".to_string(),
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
        moe_geometry: Option<MoeScratchGeometry>,
        qwen35_geometry: Option<Qwen35ScratchGeometry>,
    ) -> Result<()> {
        match self {
            Self::Cuda {
                allocation_arena,
                scratch_allocation,
                staging_allocation,
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
                        moe_geometry,
                        qwen35_geometry,
                    )
                });
                if needs_allocation {
                    let required_bytes = CudaDecodeScratch::estimated_allocated_bytes(
                        embedding_length,
                        q_width,
                        kv_width,
                        feed_forward_length,
                        vocab_size,
                        moe_geometry,
                        qwen35_geometry,
                    )?;
                    let required_staging_bytes =
                        CudaDecodeScratch::estimated_staging_bytes(moe_geometry)?;
                    let replacement_lease = allocation_arena
                        .as_ref()
                        .map(|arena| arena.reserve(GpuAllocationClass::Scratch, required_bytes))
                        .transpose()?;
                    let replacement_staging_lease = allocation_arena
                        .as_ref()
                        .map(|arena| {
                            arena.reserve(GpuAllocationClass::Staging, required_staging_bytes)
                        })
                        .transpose()?;
                    let mut scratch = CudaDecodeScratch::allocate(
                        device,
                        embedding_length,
                        q_width,
                        kv_width,
                        feed_forward_length,
                        vocab_size,
                        decode_capacity,
                        moe_geometry,
                        qwen35_geometry,
                    )?;
                    if let Some(moe) = &mut scratch.moe {
                        moe.configure_graph_mode(decode_graph.mode);
                    }
                    decode_graph.reset();
                    *batch_graph_epoch = (*batch_graph_epoch).wrapping_add(1);
                    *batch_graph_captured = false;
                    *decode_scratch = Some(scratch);
                    *scratch_allocation = replacement_lease;
                    *staging_allocation = replacement_staging_lease;
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
                    if let Some(moe) = &mut scratch.moe {
                        moe.reset_graphs();
                    }
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

    fn cuda_qwen35_parts_mut(
        &mut self,
    ) -> Result<(
        &mut [CudaLayerKvStore],
        &mut CudaDecodeScratch,
        &mut CudaDeltaNetState,
    )> {
        match self {
            Self::Cuda {
                layer_caches,
                decode_scratch,
                recurrent,
                ..
            } => {
                let scratch = decode_scratch.as_mut().ok_or_else(|| {
                    XrtError::Runtime("Qwen3.5 CUDA decode scratch is not allocated".to_string())
                })?;
                let recurrent = recurrent.cuda_mut()?.ok_or_else(|| {
                    XrtError::Runtime(
                        "Qwen3.5 CUDA execution requires session recurrent state".to_string(),
                    )
                })?;
                let trunk_layers = recurrent.descriptor().layers().len();
                let cache_count = layer_caches.len();
                let trunk_caches = layer_caches.get_mut(..trunk_layers).ok_or_else(|| {
                    XrtError::Runtime(format!(
                        "Qwen3.5 CUDA session has {} layer caches for a {trunk_layers}-layer trunk",
                        cache_count
                    ))
                })?;
                Ok((trunk_caches, scratch, recurrent))
            }
            Self::Cpu { .. } => Err(XrtError::Runtime(
                "Qwen3.5 CUDA state requested from a CPU session".to_string(),
            )),
        }
    }

    fn cuda_qwen35_graph_parts_mut(
        &mut self,
    ) -> Result<(
        &mut CudaDecodeGraphState,
        &mut [CudaLayerKvStore],
        &mut CudaDecodeScratch,
        &mut CudaDeltaNetState,
    )> {
        match self {
            Self::Cuda {
                decode_graph,
                layer_caches,
                decode_scratch,
                recurrent,
                ..
            } => {
                let scratch = decode_scratch.as_mut().ok_or_else(|| {
                    XrtError::Runtime("Qwen3.5 CUDA decode scratch is not allocated".to_string())
                })?;
                let recurrent = recurrent.cuda_mut()?.ok_or_else(|| {
                    XrtError::Runtime(
                        "Qwen3.5 CUDA execution requires session recurrent state".to_string(),
                    )
                })?;
                let trunk_layers = recurrent.descriptor().layers().len();
                let cache_count = layer_caches.len();
                let trunk_caches = layer_caches.get_mut(..trunk_layers).ok_or_else(|| {
                    XrtError::Runtime(format!(
                        "Qwen3.5 CUDA session has {} layer caches for a {trunk_layers}-layer trunk",
                        cache_count
                    ))
                })?;
                Ok((decode_graph, trunk_caches, scratch, recurrent))
            }
            Self::Cpu { .. } => Err(XrtError::Runtime(
                "Qwen3.5 CUDA graph state requested from a CPU session".to_string(),
            )),
        }
    }

    fn cuda_qwen35_mtp_parts_mut(
        &mut self,
    ) -> Result<(
        &mut CudaLayerKvStore,
        &mut CudaDecodeScratch,
        &mut CudaDeltaNetState,
    )> {
        match self {
            Self::Cuda {
                layer_caches,
                decode_scratch,
                recurrent,
                ..
            } => {
                let scratch = decode_scratch.as_mut().ok_or_else(|| {
                    XrtError::Runtime("Qwen3.5 CUDA decode scratch is not allocated".to_string())
                })?;
                let recurrent = recurrent.cuda_mut()?.ok_or_else(|| {
                    XrtError::Runtime(
                        "Qwen3.5 CUDA execution requires session recurrent state".to_string(),
                    )
                })?;
                let mtp_index = recurrent.descriptor().layers().len();
                let mtp_cache = layer_caches.get_mut(mtp_index).ok_or_else(|| {
                    XrtError::Runtime(format!(
                        "Qwen MTP cache is missing at appended layer {mtp_index}"
                    ))
                })?;
                Ok((mtp_cache, scratch, recurrent))
            }
            Self::Cpu { .. } => Err(XrtError::Runtime(
                "Qwen3.5 CUDA MTP state requested from a CPU session".to_string(),
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

    pub fn cuda_staging_allocated_bytes(&self) -> u64 {
        match self {
            Self::Cpu { .. } => 0,
            Self::Cuda { decode_scratch, .. } => decode_scratch
                .as_ref()
                .map_or(0, CudaDecodeScratch::staging_bytes),
        }
    }
}

pub trait CausalLmBackend: Send + Sync {
    fn kind(&self) -> BackendKind;
    fn model_name(&self) -> &str;
    fn config(&self) -> &LlamaConfig;
    #[cfg(feature = "moe-route-trace")]
    fn start_moe_route_trace(&self, _max_entries: usize) -> Result<()> {
        Err(XrtError::Unsupported(
            "this backend does not expose MoE route tracing".to_string(),
        ))
    }
    #[cfg(feature = "moe-route-trace")]
    fn take_moe_route_trace(&self) -> Result<Option<MoeRouteTrace>> {
        Ok(None)
    }
    fn new_session(&self, cache_mode: KvCacheMode, page_tokens: usize) -> BackendSession {
        let config = self.config();
        let mut session = BackendSession::new_cpu(
            cache_mode,
            config.block_count,
            config.kv_width(),
            page_tokens,
        );
        session.set_initial_recurrent_state(config.deltanet_state_descriptor().cloned());
        session
    }
    fn prepare_session_state(&self, session: &mut BackendSession) -> Result<()> {
        session.prepare_recurrent_state()
    }
    /// Safe request-boundary hook used by global, epoch-based runtime policy.
    ///
    /// Implementations must finish before token zero and may not mutate the
    /// caller's session.
    fn prepare_request(&self) -> Result<()> {
        Ok(())
    }
    fn save_state(&self, session: &BackendSession) -> Result<Option<BackendStateSnapshot>> {
        session.recurrent_state_snapshot()
    }
    fn restore_state(
        &self,
        session: &mut BackendSession,
        snapshot: Option<&BackendStateSnapshot>,
        expected_position: usize,
    ) -> Result<()> {
        session.restore_recurrent_state(snapshot, expected_position)
    }
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
    /// Returns trained model proposals when the backend has an admitted MTP
    /// head. The default keeps all existing backends on their current path.
    fn draft_mtp_greedy(
        &self,
        _next_token_id: u32,
        _max_draft_tokens: usize,
        _session: &mut BackendSession,
    ) -> Result<Option<Vec<u32>>> {
        Ok(None)
    }
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

    fn moe_gpu_expert_slots(&self) -> usize {
        0
    }

    fn moe_gpu_expert_bytes(&self) -> u64 {
        0
    }

    fn moe_placement_generation(&self) -> u64 {
        0
    }

    fn moe_placement_manifest_sha256(&self) -> Option<&str> {
        None
    }

    fn cuda_moe_telemetry(&self) -> CudaMoeTelemetrySnapshot {
        CudaMoeTelemetrySnapshot::default()
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

    fn run_token_transaction<T>(
        &self,
        session: &mut BackendSession,
        start_position: usize,
        forward: impl FnOnce(&LlamaModel, &mut SessionKvCache, Option<&mut DeltaNetState>) -> Result<T>,
    ) -> Result<T> {
        session.prepare_recurrent_state()?;
        let result = {
            let (cache, recurrent) = session.cpu_cache_and_recurrent_mut()?;
            forward(&self.model, cache, recurrent)
        };
        match result {
            Ok(value) => Ok(value),
            Err(forward_error) => {
                let rollback = session
                    .truncate(start_position)
                    .and_then(|_| session.validate_recurrent_position(start_position));
                match rollback {
                    Ok(()) => Err(forward_error),
                    Err(rollback_error) => {
                        let reason = format!(
                            "CPU forward failed ({forward_error}); rollback to token boundary {start_position} failed ({rollback_error})"
                        );
                        session.poison_recurrent_state(reason.clone());
                        Err(XrtError::Runtime(format!(
                            "{reason}; session is poisoned and must be reset"
                        )))
                    }
                }
            }
        }
    }

    fn run_batch_transaction<T>(
        &self,
        session: &mut BackendSession,
        start_position: usize,
        forward: impl FnOnce(&LlamaModel, &mut SessionKvCache, Option<&mut DeltaNetState>) -> Result<T>,
    ) -> Result<T> {
        session.prepare_recurrent_state()?;
        let recurrent_snapshot = session.recurrent_state_snapshot()?;
        let result = {
            let (cache, recurrent) = session.cpu_cache_and_recurrent_mut()?;
            forward(&self.model, cache, recurrent)
        };
        match result {
            Ok(value) => Ok(value),
            Err(forward_error) => {
                let rollback = session.truncate(start_position).and_then(|_| {
                    session.restore_recurrent_state(recurrent_snapshot.as_ref(), start_position)
                });
                match rollback {
                    Ok(()) => Err(forward_error),
                    Err(rollback_error) => {
                        let reason = format!(
                            "CPU batch forward failed ({forward_error}); rollback to token boundary {start_position} failed ({rollback_error})"
                        );
                        session.poison_recurrent_state(reason.clone());
                        Err(XrtError::Runtime(format!(
                            "{reason}; session is poisoned and must be reset"
                        )))
                    }
                }
            }
        }
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

    #[cfg(feature = "moe-route-trace")]
    fn start_moe_route_trace(&self, max_entries: usize) -> Result<()> {
        self.model.start_moe_route_trace(max_entries)
    }

    #[cfg(feature = "moe-route-trace")]
    fn take_moe_route_trace(&self) -> Result<Option<MoeRouteTrace>> {
        Ok(self.model.take_moe_route_trace())
    }

    fn forward_token(
        &self,
        token_id: u32,
        position: usize,
        session: &mut BackendSession,
        output_logits: &mut Vec<f32>,
    ) -> Result<()> {
        self.run_token_transaction(session, position, |model, cache, recurrent| {
            model.forward_token_with_state(token_id, position, recurrent, cache, output_logits)
        })
    }

    fn forward_draft(
        &self,
        token_id: u32,
        position: usize,
        n_layers: usize,
        session: &mut BackendSession,
        output_logits: &mut Vec<f32>,
    ) -> Result<()> {
        self.run_token_transaction(session, position, |model, cache, recurrent| {
            model.forward_draft_with_state(
                token_id,
                position,
                n_layers,
                recurrent,
                cache,
                output_logits,
            )
        })
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
        self.run_batch_transaction(session, start_position, |model, cache, recurrent| {
            model.forward_batch_with_state(token_ids, start_position, recurrent, cache)
        })
    }

    fn forward_batch_with_embeddings(
        &self,
        token_ids: &[u32],
        start_position: usize,
        session: &mut BackendSession,
        embedding_overrides: HashMap<usize, Vec<f32>>,
    ) -> Result<Vec<f32>> {
        self.run_batch_transaction(session, start_position, |model, cache, recurrent| {
            model.forward_batch_with_embeddings_and_state(
                token_ids,
                start_position,
                recurrent,
                cache,
                embedding_overrides,
            )
        })
    }

    fn forward_batch_all_logits(
        &self,
        token_ids: &[u32],
        start_position: usize,
        session: &mut BackendSession,
    ) -> Result<Vec<f32>> {
        self.run_batch_transaction(session, start_position, |model, cache, recurrent| {
            model.forward_batch_all_logits_with_state(token_ids, start_position, recurrent, cache)
        })
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
    device_ordinal: usize,
    resident_model_weight_bytes: u64,
    _model_allocation: Option<GpuAllocationLease>,
    _expert_allocation: Option<GpuAllocationLease>,
    allocation_arena: Option<Arc<GpuAllocationArena>>,
    device_name: Option<String>,
    kv_budget_bytes: u64,
    cuda_graph_mode: CudaGraphMode,
    cpu_order_q4_k_matvec: bool,
    qwen35_capture_gate: Arc<RwLock<()>>,
    moe_graph_execution_gate: Arc<Mutex<()>>,
    decode_batch_graphs: Mutex<CudaDecodeBatchGraphCache>,
    decode_batch_streams: Mutex<Vec<CudaExecutionStream>>,
    f32_probe: Option<ResidentF32ProbeWeights>,
    q8_0_probe: Option<ResidentQ8_0ProbeWeights>,
    q8_0_layer_probes: Option<Vec<ResidentQ8_0LayerWeights>>,
    gemma4_layer_probes: Option<Vec<ResidentGemma4LayerWeights>>,
    qwen35_layer_probes: Option<Vec<ResidentQwen35LayerWeights>>,
    qwen35_mtp_probe: Option<ResidentQwen35MtpWeights>,
    qwen35_moe_layer_probes: Option<Vec<ResidentQwen35MoeLayerWeights>>,
    moe_layer_probes: Option<Vec<ResidentMoeLayerWeights>>,
    moe_coordinator: Option<HeterogeneousMoeCoordinator>,
    moe_placement_gate: Arc<RwLock<()>>,
    adaptive_moe: Option<MoeAdaptiveRuntime>,
    layerwise_moe_prefill: Option<MoeLayerwisePrefillRuntime>,
    gpu_expert_slots: usize,
    gpu_expert_bytes: u64,
    placement_generation: AtomicU64,
    placement_manifest_sha256: Option<String>,
    moe_telemetry: CudaMoeTelemetry,
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
        Self::new_with_source(
            Some(model),
            model_name,
            model_config,
            &source,
            gpu_config,
            None,
            GpuAllocationClass::ModelWeights,
        )
    }

    pub fn new_with_resource_manager(
        model: Arc<LlamaModel>,
        gguf: &GgufFile,
        resources: Arc<GpuResourceManager>,
    ) -> Result<Self> {
        let model_name = model.model_name().to_string();
        let model_config = model.config().clone();
        let source = GgufResidentTensorSource::new(gguf);
        Self::new_with_source(
            Some(model),
            model_name,
            model_config,
            &source,
            resources.config(),
            Some(resources.allocation_arena()),
            GpuAllocationClass::ModelWeights,
        )
    }

    pub fn new_moe_with_resource_manager(
        model: Arc<LlamaModel>,
        gguf: Arc<GgufFile>,
        resources: Arc<GpuResourceManager>,
        moe_runtime: &MoeRuntimeConfig,
    ) -> Result<Self> {
        let model_config = model.config().clone();
        if !model_config.is_moe() {
            return Err(XrtError::Unsupported(
                "CUDA MoE constructor requires an MoE model".to_string(),
            ));
        }
        if model_config.is_hybrid() && moe_runtime.placement == MoePlacementPolicy::Adaptive {
            return Err(XrtError::Unsupported(
                "Qwen3.5 hybrid-MoE adaptive placement is not yet enabled; use uniform or profiled fixed placement"
                    .to_string(),
            ));
        }
        if model_config.is_hybrid() && moe_runtime.layerwise_prefill {
            return Err(XrtError::Unsupported(
                "Qwen3.5 hybrid-MoE layerwise prefill is not yet enabled; disable XRT_MOE_LAYERWISE_PREFILL"
                    .to_string(),
            ));
        }
        if !matches!(
            moe_runtime.acceleration,
            MoeAcceleration::Hybrid | MoeAcceleration::Gpu
        ) {
            return Err(XrtError::Runtime(format!(
                "CUDA MoE constructor cannot serve acceleration mode {}",
                moe_runtime.acceleration.as_str()
            )));
        }

        let gpu_config = resources.config();
        let device = CudaDevice::new(gpu_config.device_ordinal)?;
        let source = GgufResidentTensorSource::new(&gguf);
        if !ResidentQ8_0ProbeWeights::supports(&source, &model_config) {
            return Err(XrtError::Unsupported(
                "CUDA MoE requires a supported resident token embedding, output norm, and output projection"
                    .to_string(),
            ));
        }
        let plan = MoeResidentUploadPlan::build(&gguf, &source, &model_config, moe_runtime)?;
        if model_config.is_hybrid() {
            ResidentQwen35MoeLayerWeights::validate_source(
                &source,
                &model_config,
                &plan.placements,
            )?;
        } else {
            ResidentMoeLayerWeights::validate_source(&source, &model_config, &plan.placements)?;
        }

        let (free_vram_bytes, total_vram_bytes) = device.memory_info()?;
        let upload_budget_bytes =
            cuda_model_upload_budget_bytes(free_vram_bytes, total_vram_bytes, gpu_config);
        let resident_model_weight_bytes = plan
            .non_expert_bytes
            .checked_add(plan.expert_bytes)
            .ok_or_else(|| {
                XrtError::Runtime("CUDA MoE resident model byte count overflowed".to_string())
            })?;
        if resident_model_weight_bytes > upload_budget_bytes {
            return Err(XrtError::Cuda(format!(
                "CUDA MoE upload requires {resident_model_weight_bytes} bytes (non-expert={}, experts={}), exceeding the configured safe {upload_budget_bytes}-byte VRAM budget",
                plan.non_expert_bytes, plan.expert_bytes
            )));
        }

        let allocation_arena = resources.allocation_arena();
        allocation_arena.configure_budget(upload_budget_bytes)?;
        let model_allocation =
            allocation_arena.reserve(GpuAllocationClass::ModelWeights, plan.non_expert_bytes)?;
        let expert_allocation =
            allocation_arena.reserve(GpuAllocationClass::ExpertWeights, plan.expert_bytes)?;

        info!(
            non_expert_bytes = plan.non_expert_bytes,
            expert_bytes = plan.expert_bytes,
            expert_slots = plan.expert_slots,
            free_vram_bytes,
            total_vram_bytes,
            "CUDA MoE resident upload preflight passed"
        );
        let output_probe = ResidentQ8_0ProbeWeights::try_load(&device, &source, &model_config)?
            .ok_or_else(|| {
                XrtError::Unsupported(
                    "CUDA MoE output weights changed after capability validation".to_string(),
                )
            })?;
        let (moe_layer_probes, qwen35_moe_layer_probes) = if model_config.is_hybrid() {
            (
                None,
                Some(ResidentQwen35MoeLayerWeights::try_load_all(
                    &device,
                    &gguf,
                    &source,
                    &model_config,
                    &plan.placements,
                )?),
            )
        } else {
            (
                Some(ResidentMoeLayerWeights::try_load_all(
                    &device,
                    &gguf,
                    &source,
                    &model_config,
                    &plan.placements,
                )?),
                None,
            )
        };
        let adaptive_moe = if moe_runtime.placement == MoePlacementPolicy::Adaptive {
            Some(MoeAdaptiveRuntime::new(
                Arc::clone(&gguf),
                &model_config,
                moe_runtime,
                plan.expert_costs.clone(),
                gpu_config.device_ordinal,
            )?)
        } else {
            None
        };
        let layerwise_moe_prefill = if moe_runtime.layerwise_prefill {
            Some(MoeLayerwisePrefillRuntime::new(
                Arc::clone(&gguf),
                &model_config,
                plan.expert_costs.clone(),
                gpu_config.device_ordinal,
            )?)
        } else {
            None
        };
        let placement_generation = plan
            .placements
            .iter()
            .map(|placement| placement.generation())
            .max()
            .unwrap_or(0);
        let model_name = model.model_name().to_string();
        let device_name = device.name().ok();
        let kv_budget_bytes =
            cuda_kv_budget_bytes(upload_budget_bytes, resident_model_weight_bytes, gpu_config);
        let cpu_order_q4_k_matvec = model_config.is_qwen35_family()
            || qwen3_moe_uses_cpu_order_q4_k_matvec(&model_config.architecture);

        Ok(Self {
            cpu_reference_model: Some(model),
            model_name,
            config: model_config,
            device,
            device_ordinal: gpu_config.device_ordinal,
            resident_model_weight_bytes,
            _model_allocation: Some(model_allocation),
            _expert_allocation: Some(expert_allocation),
            allocation_arena: Some(allocation_arena),
            device_name,
            kv_budget_bytes,
            cuda_graph_mode: gpu_config.cuda_graph_mode,
            cpu_order_q4_k_matvec,
            qwen35_capture_gate: Arc::new(RwLock::new(())),
            moe_graph_execution_gate: Arc::new(Mutex::new(())),
            decode_batch_graphs: Mutex::new(CudaDecodeBatchGraphCache::default()),
            decode_batch_streams: Mutex::new(Vec::new()),
            f32_probe: None,
            q8_0_probe: Some(output_probe),
            q8_0_layer_probes: None,
            gemma4_layer_probes: None,
            qwen35_layer_probes: None,
            qwen35_mtp_probe: None,
            qwen35_moe_layer_probes,
            moe_layer_probes,
            moe_coordinator: Some(HeterogeneousMoeCoordinator::new()?),
            moe_placement_gate: Arc::new(RwLock::new(())),
            adaptive_moe,
            layerwise_moe_prefill,
            gpu_expert_slots: plan.expert_slots,
            gpu_expert_bytes: plan.expert_bytes,
            placement_generation: AtomicU64::new(placement_generation),
            placement_manifest_sha256: plan.manifest_sha256,
            moe_telemetry: CudaMoeTelemetry::default(),
        })
    }

    fn record_adaptive_moe_route(&self, layer_index: usize, route: &MoeRoutingRow) -> Result<()> {
        let Some(adaptive) = &self.adaptive_moe else {
            return Ok(());
        };
        adaptive.tracker.lock().record_route(
            layer_index,
            route.logical_ids(),
            layer_index + 1 == self.config.block_count,
        )
    }

    fn prepare_adaptive_moe_request(&self) -> Result<()> {
        let Some(adaptive) = &self.adaptive_moe else {
            return Ok(());
        };
        let layers = self.moe_layer_probes.as_ref().ok_or_else(|| {
            XrtError::Runtime("adaptive MoE runtime has no resident layer set".to_string())
        })?;

        // This is the only publication boundary. It drains in-flight MoE
        // forwards, and every later step either fails before publication or is
        // an infallible replacement of prevalidated handles.
        let _epoch_guard = self.moe_placement_gate.write();
        let current = layers
            .iter()
            .map(|layer| Arc::clone(&layer.resident.read().snapshot))
            .collect::<Vec<_>>();
        let Some(decision) = adaptive.tracker.lock().propose(&current)? else {
            return Ok(());
        };
        if decision.moves().is_empty() {
            adaptive.tracker.lock().commit_evaluation(&decision)?;
            self.moe_telemetry.record_placement_evaluation(0, 0, 0);
            return Ok(());
        }

        let target_snapshots = decision
            .target_gpu_experts()
            .iter()
            .enumerate()
            .map(|(layer_index, experts)| {
                ExpertPlacementSnapshot::from_gpu_experts(
                    layer_index,
                    self.config.expert_count.unwrap_or_default(),
                    decision.placement_generation(),
                    experts,
                )
                .map(Arc::new)
            })
            .collect::<Result<Vec<_>>>()?;
        if target_snapshots.len() != layers.len() {
            return Err(XrtError::Runtime(
                "adaptive MoE decision changed the layer count".to_string(),
            ));
        }
        for movement in decision.moves() {
            let layer = layers.get(movement.layer_index()).ok_or_else(|| {
                XrtError::Runtime("adaptive MoE move references a missing layer".to_string())
            })?;
            let resident = layer.resident.read();
            let slot = resident.slots.get(movement.gpu_slot()).ok_or_else(|| {
                XrtError::Runtime("adaptive MoE move references a missing GPU slot".to_string())
            })?;
            if slot.logical_expert() != movement.outgoing_expert()
                || target_snapshots[movement.layer_index()]
                    .logical_expert_for(u16::try_from(movement.gpu_slot()).map_err(|_| {
                        XrtError::Runtime("adaptive MoE GPU slot does not fit u16".to_string())
                    })?)
                    .map(usize::from)
                    != Some(movement.incoming_expert())
            {
                return Err(XrtError::Runtime(
                    "adaptive MoE decision no longer matches the resident slot map".to_string(),
                ));
            }
        }

        let upload_bytes = adaptive.incoming_bytes(&decision)?;
        let _staging_allocation = self
            .allocation_arena
            .as_ref()
            .map(|arena| arena.reserve(GpuAllocationClass::Staging, upload_bytes))
            .transpose()?;
        self.device.synchronize()?;
        let started = Instant::now();
        let source = GgufResidentTensorSource::new(&adaptive.gguf);
        let mut uploaded = Vec::with_capacity(decision.moves().len());
        for &movement in decision.moves() {
            uploaded.push(AdaptiveUploadedSlot {
                movement,
                slot: Some(ResidentMoeExpertSlot::upload(
                    &adaptive.staging_device,
                    &adaptive.gguf,
                    &source,
                    &self.config,
                    movement.layer_index(),
                    movement.incoming_expert(),
                )?),
            });
        }
        // Every replacement allocation/copy uses a distinct CUDA device-stream
        // owner for the same primary context. Nothing is published until that
        // dedicated staging stream proves all transfers complete.
        adaptive.staging_device.synchronize()?;

        adaptive.tracker.lock().commit_evaluation(&decision)?;
        for (layer_index, (layer, snapshot)) in layers.iter().zip(target_snapshots).enumerate() {
            let mut resident = layer.resident.write();
            for replacement in uploaded
                .iter_mut()
                .filter(|replacement| replacement.movement.layer_index() == layer_index)
            {
                resident.slots[replacement.movement.gpu_slot()] = Arc::new(
                    replacement
                        .slot
                        .take()
                        .expect("adaptive upload was prevalidated and is published once"),
                );
            }
            resident.snapshot = snapshot;
        }
        self.placement_generation
            .store(decision.placement_generation(), Ordering::Release);
        self.decode_batch_graphs.lock().entries.clear();
        let elapsed_micros = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX);
        self.moe_telemetry.record_placement_evaluation(
            decision.moves().len(),
            upload_bytes,
            elapsed_micros,
        );
        info!(
            placement_generation = decision.placement_generation(),
            moves = decision.moves().len(),
            upload_bytes,
            elapsed_micros,
            "published adaptive MoE placement epoch"
        );
        Ok(())
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
        Self::new_with_source(
            None,
            model_name,
            model_config,
            &source,
            gpu_config,
            None,
            GpuAllocationClass::ModelWeights,
        )
    }

    pub fn from_hf_bundle_with_resource_manager(
        bundle: &HfModelBundle,
        resources: Arc<GpuResourceManager>,
    ) -> Result<Self> {
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
        Self::new_with_source(
            None,
            model_name,
            model_config,
            &source,
            resources.config(),
            Some(resources.allocation_arena()),
            GpuAllocationClass::ModelWeights,
        )
    }

    /// Load a dense SafeTensors language backbone owned by an image pipeline.
    /// The execution path is identical to [`Self::from_hf_bundle_with_resource_manager`],
    /// but the shared arena reports the resident weights as image-component
    /// memory rather than text-model memory.
    pub fn from_hf_bundle_as_image_component(
        bundle: &HfModelBundle,
        resources: Arc<GpuResourceManager>,
    ) -> Result<Self> {
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
            .unwrap_or_else(|| "safetensors-image-text-encoder".to_string());
        let source = HfStandardDenseResidentTensorSource::new(bundle)?;
        Self::new_with_source(
            None,
            model_name,
            model_config,
            &source,
            resources.config(),
            Some(resources.allocation_arena()),
            GpuAllocationClass::ImageComponentWeights,
        )
    }

    /// Encode a non-empty token sequence and return the final, normalized
    /// hidden state for every position in row-major `[sequence, hidden]`
    /// order.
    ///
    /// This intentionally exposes only the standard dense CUDA path. It is
    /// used by component runtimes such as Qwen Image that consume a causal
    /// language model as a text encoder without computing vocabulary logits.
    pub fn encode_standard_dense_hidden_states(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(XrtError::Runtime(
                "hidden-state encoding requires at least one token".to_string(),
            ));
        }
        if token_ids.len() > self.config.context_length {
            return Err(XrtError::Runtime(format!(
                "hidden-state sequence length {} exceeds model context length {}",
                token_ids.len(),
                self.config.context_length
            )));
        }
        if self.config.is_hybrid() || self.config.is_moe() || self.config.is_gemma4() {
            return Err(XrtError::Unsupported(
                "hidden-state encoding currently requires a standard dense CUDA model".to_string(),
            ));
        }

        let output_len = token_ids
            .len()
            .checked_mul(self.config.embedding_length)
            .ok_or_else(|| {
                XrtError::Runtime("hidden-state output length overflowed usize".to_string())
            })?;
        let total_len = cuda_total_len_after_batch(0, token_ids.len())?;
        let mut session = <Self as CausalLmBackend>::new_session(self, KvCacheMode::F32, 32);
        let mut output = Vec::with_capacity(output_len);
        let mut logits = Vec::new();
        let mut hidden = Vec::new();
        for (position, token_id) in token_ids.iter().copied().enumerate() {
            if !self.try_forward_token_q8_0_with_logits(
                token_id,
                position,
                &mut session,
                &mut logits,
                false,
                false,
                None,
                total_len,
                None,
                Some(&mut hidden),
            )? {
                return Err(Self::decode_unsupported());
            }
            if hidden.len() != self.config.embedding_length {
                return Err(XrtError::Runtime(format!(
                    "CUDA hidden-state width mismatch at position {position}: expected {}, found {}",
                    self.config.embedding_length,
                    hidden.len()
                )));
            }
            if hidden.iter().any(|value| !value.is_finite()) {
                return Err(XrtError::Runtime(format!(
                    "CUDA hidden state contains non-finite values at position {position}"
                )));
            }
            output.extend_from_slice(&hidden);
        }
        debug_assert_eq!(output.len(), output_len);
        Ok(output)
    }

    fn new_with_source(
        cpu_reference_model: Option<Arc<LlamaModel>>,
        model_name: String,
        model_config: LlamaConfig,
        source: &impl ResidentTensorSource,
        gpu_config: GpuResourceConfig,
        allocation_arena: Option<Arc<GpuAllocationArena>>,
        allocation_class: GpuAllocationClass,
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
        let (
            free_vram_bytes,
            total_vram_bytes,
            upload_budget_bytes,
            resident_model_weight_bytes,
            kv_budget_bytes,
        ) = Self::preflight_model_upload(source, &model_config, &device, gpu_config)?;
        let model_allocation = if let Some(arena) = allocation_arena.as_ref() {
            arena.configure_budget(upload_budget_bytes)?;
            Some(arena.reserve(allocation_class, resident_model_weight_bytes)?)
        } else {
            None
        };
        info!(
            resident_model_weight_bytes,
            free_vram_bytes,
            total_vram_bytes,
            kv_budget_bytes,
            "CUDA resident upload preflight passed"
        );
        let cpu_order_q4_k_matvec = model_config.is_qwen35_family()
            || qwen3_moe_uses_cpu_order_q4_k_matvec(&model_config.architecture);
        let device_name = device.name().ok();
        info!("loading CUDA resident output weights");
        let f32_probe = ResidentF32ProbeWeights::try_load(&device, source, &model_config)?;
        let q8_0_probe = ResidentQ8_0ProbeWeights::try_load(&device, source, &model_config)?;
        info!("loading CUDA resident transformer layers");
        let q8_0_layer_probes =
            ResidentQ8_0LayerWeights::try_load_all(&device, source, &model_config)?;
        let gemma4_layer_probes =
            ResidentGemma4LayerWeights::try_load_all(&device, source, &model_config)?;
        let qwen35_layer_probes =
            ResidentQwen35LayerWeights::try_load_all(&device, source, &model_config)?;
        let qwen35_mtp_probe = ResidentQwen35MtpWeights::try_load(&device, source, &model_config)?;
        info!("CUDA resident model upload complete");
        Ok(Self {
            cpu_reference_model,
            model_name,
            config: model_config,
            device,
            device_ordinal: gpu_config.device_ordinal,
            resident_model_weight_bytes,
            _model_allocation: model_allocation,
            _expert_allocation: None,
            allocation_arena,
            device_name,
            kv_budget_bytes,
            cuda_graph_mode: gpu_config.cuda_graph_mode,
            cpu_order_q4_k_matvec,
            qwen35_capture_gate: Arc::new(RwLock::new(())),
            moe_graph_execution_gate: Arc::new(Mutex::new(())),
            decode_batch_graphs: Mutex::new(CudaDecodeBatchGraphCache::default()),
            decode_batch_streams: Mutex::new(Vec::new()),
            f32_probe,
            q8_0_probe,
            q8_0_layer_probes,
            gemma4_layer_probes,
            qwen35_layer_probes,
            qwen35_mtp_probe,
            qwen35_moe_layer_probes: None,
            moe_layer_probes: None,
            moe_coordinator: None,
            moe_placement_gate: Arc::new(RwLock::new(())),
            adaptive_moe: None,
            layerwise_moe_prefill: None,
            gpu_expert_slots: 0,
            gpu_expert_bytes: 0,
            placement_generation: AtomicU64::new(0),
            placement_manifest_sha256: None,
            moe_telemetry: CudaMoeTelemetry::default(),
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
            && if config.is_hybrid() {
                ResidentQwen35LayerWeights::supports_all(source, config)
                    && (!config.has_nextn_predictor()
                        || ResidentQwen35MtpWeights::supports(source, config))
            } else if config.is_gemma4() {
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
    ) -> Result<(u64, u64, u64, u64, u64)> {
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
            upload_budget_bytes,
            model_weight_bytes,
            cuda_kv_budget_bytes(upload_budget_bytes, model_weight_bytes, config),
        ))
    }

    fn decode_unsupported() -> XrtError {
        XrtError::Unsupported(
            "cuda-resident decode currently supports standard dense, dense Qwen3.5 hybrid, and Gemma4 GGUF F32/F16/BF16/Q8_0/Q4_0/Q4_K/Q5_K/Q6_K models plus dense Qwen2/Qwen3 SafeTensors, AutoAWQ GEMM/GEMV, GPTQ v1/v2 GEMM4, or compressed-tensors W4A16; unsupported Qwen3.5 recurrent geometry and hybrid MoE layouts require CPU or explicit hybrid-MoE support"
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
        let mut kv_cache = CudaLayerKvStore::F32(self.device.alloc_shared_paged_layer_kv_cache(
            1,
            config.kv_width(),
            1,
        )?);
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
            CudaLayerKvStore::SharedQ8(cache) => {
                cache.append(&k, &v)?;
                cache.single_query_attention_device(
                    &q,
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
            CudaLayerKvStore::SharedKeyQ4ValueQ8(cache) => {
                cache.append(&k, &v)?;
                cache.single_query_attention_device(
                    &q,
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
            CudaLayerKvStore::SharedAgentAdaptive(cache) => {
                cache.append(adaptive_is_hot, &k, &v)?;
                cache.single_query_attention_device(
                    &q,
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
            CudaLayerKvStore::SharedQ8(cache) => {
                cache.append(&scratch.k, &scratch.v)?;
                cache.single_query_attention_device(
                    &scratch.q,
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
            CudaLayerKvStore::SharedKeyQ4ValueQ8(cache) => {
                cache.append(&scratch.k, &scratch.v)?;
                cache.single_query_attention_device(
                    &scratch.q,
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
            CudaLayerKvStore::SharedAgentAdaptive(cache) => {
                cache.append(adaptive_is_hot, &scratch.k, &scratch.v)?;
                cache.single_query_attention_device(
                    &scratch.q,
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

    #[allow(clippy::too_many_arguments)]
    fn run_moe_attention_and_norm_with_scratch(
        &self,
        layer_index: usize,
        weights: &ResidentMoeLayerWeights,
        input: &CudaF32Buffer,
        position: usize,
        adaptive_is_hot: bool,
        kv_cache: &mut CudaLayerKvStore,
        scratch: &mut CudaDecodeScratch,
    ) -> Result<()> {
        let config = &self.config;
        self.device.rmsnorm_device_into(
            input,
            weights.attn_norm.buffer(),
            1,
            weights.embedding_length,
            config.rms_norm_eps,
            &mut scratch.normed_post_attention,
        )?;
        self.matvec_quant_resident_device_into(
            &weights.attn_q,
            &scratch.normed_post_attention,
            &mut scratch.q,
        )?;
        self.matvec_quant_resident_device_into(
            &weights.attn_k,
            &scratch.normed_post_attention,
            &mut scratch.k,
        )?;
        self.matvec_quant_resident_device_into(
            &weights.attn_v,
            &scratch.normed_post_attention,
            &mut scratch.v,
        )?;
        if let Some(bias) = &weights.attn_q_bias {
            self.device
                .add_assign_device(&mut scratch.q, bias.buffer())?;
        }
        if let Some(bias) = &weights.attn_k_bias {
            self.device
                .add_assign_device(&mut scratch.k, bias.buffer())?;
        }
        if let Some(bias) = &weights.attn_v_bias {
            self.device
                .add_assign_device(&mut scratch.v, bias.buffer())?;
        }
        if let Some(q_norm) = &weights.attn_q_norm {
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
        if let Some(k_norm) = &weights.attn_k_norm {
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
            CudaLayerKvStore::SharedQ8(cache) => {
                cache.append(&scratch.k, &scratch.v)?;
                cache.single_query_attention_device(
                    &scratch.q,
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
            CudaLayerKvStore::SharedKeyQ4ValueQ8(cache) => {
                cache.append(&scratch.k, &scratch.v)?;
                cache.single_query_attention_device(
                    &scratch.q,
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
            CudaLayerKvStore::SharedAgentAdaptive(cache) => {
                cache.append(adaptive_is_hot, &scratch.k, &scratch.v)?;
                cache.single_query_attention_device(
                    &scratch.q,
                    config.attention_head_count,
                    config.attention_head_count_kv,
                    config.head_dim(),
                )?
            }
        };
        self.matvec_quant_resident_device_into(
            &weights.attn_output,
            &attention_values,
            &mut scratch.hidden_temp,
        )?;
        self.device
            .copy_f32_device(input, &mut scratch.normed_post_attention)?;
        self.device
            .add_assign_device(&mut scratch.normed_post_attention, &scratch.hidden_temp)?;
        self.device.rmsnorm_device_into(
            &scratch.normed_post_attention,
            weights.ffn_norm.buffer(),
            1,
            weights.embedding_length,
            config.rms_norm_eps,
            &mut scratch.hidden_temp,
        )?;
        if Self::cuda_profile_enabled() {
            info!(
                layer_index,
                "cuda profile: MoE attention and post-attention norm complete"
            );
        }
        Ok(())
    }

    fn run_resident_moe_expert_into(
        &self,
        slot: &ResidentMoeExpertSlot,
        input: &CudaF32Buffer,
        gate: &mut CudaF32Buffer,
        up: &mut CudaF32Buffer,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        self.matvec_quant_resident_device_into(&slot.gate, input, gate)?;
        self.matvec_quant_resident_device_into(&slot.up, input, up)?;
        self.device.silu_assign_device(gate)?;
        self.device.mul_assign_device(gate, up)?;
        self.matvec_quant_resident_device_into(&slot.down, gate, output)
    }

    fn execute_cpu_moe_layer(
        model: Arc<LlamaModel>,
        layer_index: usize,
        descriptor: MoeLayerDescriptor,
        host: Arc<Mutex<MoePinnedHostStaging>>,
        input_transfer: CudaPinnedF32Download,
        cpu_work: [MoeWorkItem; MAX_SELECTED_EXPERTS],
        cpu_work_len: usize,
        run_shared: bool,
    ) -> Result<CpuMoeLayerOutput> {
        if cpu_work_len > cpu_work.len() {
            return Err(XrtError::Runtime(format!(
                "CPU MoE work length {cpu_work_len} exceeds fixed capacity {}",
                cpu_work.len()
            )));
        }
        let (input_after_event, transfer_stream) = input_transfer.wait()?;
        let mut output_rows = [None; MAX_SELECTED_EXPERTS];
        let mut staging = host.lock();
        if staging.input.is_some() || staging.transfer_stream.is_some() {
            return Err(XrtError::Runtime(
                "CUDA MoE transfer resources were returned twice".to_string(),
            ));
        }
        staging.input = Some(input_after_event);
        staging.transfer_stream = Some(transfer_stream);
        let MoePinnedHostStaging {
            geometry,
            input,
            outputs,
            gate,
            up,
            shared_gate,
            shared_up,
            ..
        } = &mut *staging;
        if *geometry
            != (MoeScratchGeometry {
                expert_count: descriptor.expert_count(),
                selected_per_token: descriptor.selected_per_token(),
                embedding_length: descriptor.hidden_size(),
                intermediate_size: descriptor.intermediate_size(),
                shared_intermediate_size: model.moe_shared_intermediate_size(layer_index),
            })
        {
            return Err(XrtError::Runtime(format!(
                "CPU MoE staging geometry changed for layer {layer_index}"
            )));
        }
        let input = input.as_ref().ok_or_else(|| {
            XrtError::Runtime("CUDA MoE input staging disappeared after transfer".to_string())
        })?;
        let input = input.as_slice();
        let mut logical_experts = [0usize; MAX_SELECTED_EXPERTS];
        let mut routing_weights = [0.0f32; MAX_SELECTED_EXPERTS];
        for (output_row, item) in cpu_work.into_iter().take(cpu_work_len).enumerate() {
            if item.token_index() != 0 {
                return Err(XrtError::Runtime(format!(
                    "batch-1 CUDA MoE coordinator received token row {}",
                    item.token_index()
                )));
            }
            let canonical_index = item.canonical_index();
            if canonical_index >= descriptor.selected_per_token() {
                return Err(XrtError::Runtime(format!(
                    "CPU MoE canonical index {canonical_index} exceeds top-k {}",
                    descriptor.selected_per_token()
                )));
            }
            logical_experts[output_row] = item.logical_expert();
            routing_weights[output_row] = item.routing_weight();
            output_rows[canonical_index] = Some(output_row);
        }
        let expert_scratch_len = cpu_work_len
            .checked_mul(descriptor.intermediate_size())
            .ok_or_else(|| {
                XrtError::Runtime("CPU MoE expert scratch size overflowed".to_string())
            })?;
        let expert_output_len = cpu_work_len
            .checked_mul(descriptor.hidden_size())
            .ok_or_else(|| {
                XrtError::Runtime("CPU MoE expert output size overflowed".to_string())
            })?;
        model.execute_moe_experts_parallel_into(
            layer_index,
            &logical_experts[..cpu_work_len],
            input,
            &mut gate[..expert_scratch_len],
            &mut up[..expert_scratch_len],
            &mut outputs.as_mut_slice()[..expert_output_len],
        )?;
        for (output_row, &weight) in routing_weights.iter().take(cpu_work_len).enumerate() {
            let start = output_row
                .checked_mul(descriptor.hidden_size())
                .ok_or_else(|| XrtError::Runtime("CPU MoE output offset overflowed".to_string()))?;
            let end = start
                .checked_add(descriptor.hidden_size())
                .ok_or_else(|| XrtError::Runtime("CPU MoE output end overflowed".to_string()))?;
            for value in &mut outputs.as_mut_slice()[start..end] {
                *value *= weight;
            }
        }

        let shared_result = if run_shared {
            let intermediate_size =
                model
                    .moe_shared_intermediate_size(layer_index)
                    .ok_or_else(|| {
                        XrtError::Runtime(format!(
                            "MoE layer {layer_index} lost its shared-expert descriptor"
                        ))
                    })?;
            if shared_gate.len() != intermediate_size || shared_up.len() != intermediate_size {
                return Err(XrtError::Runtime(format!(
                    "MoE shared scratch length changed for layer {layer_index}"
                )));
            }
            let output_row = cpu_work_len;
            let start = output_row
                .checked_mul(descriptor.hidden_size())
                .ok_or_else(|| {
                    XrtError::Runtime("CPU shared MoE output offset overflowed".to_string())
                })?;
            let end = start.checked_add(descriptor.hidden_size()).ok_or_else(|| {
                XrtError::Runtime("CPU shared MoE output end overflowed".to_string())
            })?;
            let weight = model.execute_shared_moe_expert_into(
                layer_index,
                input,
                shared_gate,
                shared_up,
                &mut outputs.as_mut_slice()[start..end],
            )?;
            for value in &mut outputs.as_mut_slice()[start..end] {
                *value *= weight;
            }
            Some(output_row)
        } else {
            None
        };
        Ok(CpuMoeLayerOutput {
            output_rows,
            output_row_count: cpu_work_len + usize::from(shared_result.is_some()),
            shared_result,
        })
    }

    fn run_moe_ffn_with_scratch<W: ResidentMoeFfnLayer>(
        &self,
        weights: &W,
        mut recycled_input: CudaF32Buffer,
        scratch: &mut CudaDecodeScratch,
        allow_graph_decode: bool,
        cache_mode: KvCacheMode,
        scratch_generation: u64,
    ) -> Result<CudaF32Buffer> {
        let descriptor = weights.moe_descriptor();
        let embedding_length = weights.moe_embedding_length();
        if recycled_input.len() != embedding_length {
            return Err(XrtError::Runtime(format!(
                "CUDA MoE recycled input length {} does not match embedding length {embedding_length}",
                recycled_input.len()
            )));
        }
        let cpu_model = self.require_cpu_reference_model()?;
        let coordinator = self
            .moe_coordinator
            .as_ref()
            .ok_or_else(|| XrtError::Runtime("CUDA MoE coordinator is unavailable".to_string()))?;
        let CudaDecodeScratch {
            hidden_temp,
            normed_post_attention,
            gate,
            up,
            moe,
            ..
        } = scratch;
        let moe = moe.as_mut().ok_or_else(|| {
            XrtError::Runtime("CUDA MoE scratch was not prepared before execution".to_string())
        })?;
        if moe.geometry.expert_count != descriptor.expert_count()
            || moe.geometry.selected_per_token != descriptor.selected_per_token()
            || moe.geometry.embedding_length != descriptor.hidden_size()
            || moe.geometry.intermediate_size != descriptor.intermediate_size()
        {
            return Err(XrtError::Runtime(format!(
                "CUDA MoE scratch geometry does not match layer {}",
                descriptor.layer_index()
            )));
        }
        let host = Arc::clone(&moe.host);
        let _clear_host = MoeHostStagingClearGuard::new(Arc::clone(&host));

        self.matvec_quant_resident_device_into(
            weights.moe_router(),
            hidden_temp,
            &mut moe.router_logits,
        )?;
        let mut route = MoeRoutingRow::default();
        {
            let mut staging = host.lock();
            self.device
                .download_f32_into_pinned(&moe.router_logits, &mut staging.router_logits)?;
            cpu_model.route_moe_logits(
                descriptor.layer_index(),
                staging.router_logits.as_slice(),
                &mut route,
            )?;
        }
        self.record_adaptive_moe_route(descriptor.layer_index(), &route)?;

        let routes = [route];
        let mut cpu_plan_scratch = [MoeWorkItem::default(); MAX_SELECTED_EXPERTS];
        let mut gpu_plan_scratch = [MoeWorkItem::default(); MAX_SELECTED_EXPERTS];
        let resident = weights.moe_resident().read();
        let plan = build_moe_execution_plan(
            descriptor,
            &resident.snapshot,
            &routes,
            &mut cpu_plan_scratch,
            &mut gpu_plan_scratch,
        )?;
        let mut cpu_work = [MoeWorkItem::default(); MAX_SELECTED_EXPERTS];
        let cpu_work_len = plan.cpu_work().len();
        cpu_work[..cpu_work_len].copy_from_slice(plan.cpu_work());
        let mut gpu_work = [MoeWorkItem::default(); MAX_SELECTED_EXPERTS];
        let gpu_work_len = plan.gpu_work().len();
        gpu_work[..gpu_work_len].copy_from_slice(plan.gpu_work());
        drop(plan);
        self.moe_telemetry.record_plan(cpu_work_len, gpu_work_len);

        let run_shared = cpu_model.moe_layer_has_shared_expert(descriptor.layer_index());
        let cpu_join = if cpu_work_len == 0 && !run_shared {
            None
        } else {
            let input_transfer = {
                let mut staging = host.lock();
                staging.ensure_transfer_resources(&self.device)?;
                let input = staging.input.take().ok_or_else(|| {
                    XrtError::Runtime("CUDA MoE input staging is unavailable".to_string())
                })?;
                let stream = staging.transfer_stream.take().ok_or_else(|| {
                    XrtError::Runtime("CUDA MoE transfer stream is unavailable".to_string())
                })?;
                // SAFETY: `hidden_temp` is stable session-owned device scratch
                // and remains alive and read-only until the CPU join below has
                // waited the transfer event.
                unsafe {
                    self.device
                        .download_f32_into_pinned_async(hidden_temp, input, stream)?
                }
            };
            self.moe_telemetry
                .record_activation_d2h(hidden_temp.byte_len());
            let model = Arc::clone(
                self.cpu_reference_model
                    .as_ref()
                    .expect("CPU MoE reference model was checked above"),
            );
            let descriptor = descriptor.clone();
            let layer_index = descriptor.layer_index();
            let worker_host = Arc::clone(&host);
            Some(coordinator.submit(move || {
                Self::execute_cpu_moe_layer(
                    model,
                    layer_index,
                    descriptor,
                    worker_host,
                    input_transfer,
                    cpu_work,
                    cpu_work_len,
                    run_shared,
                )
            })?)
        };

        let selected = descriptor.selected_per_token();
        let graph_allowed = allow_graph_decode && !Self::cuda_profile_enabled();
        let full_gpu_residency = resident.snapshot.gpu_slot_count() == descriptor.expert_count();
        let mut gpu_completed = [false; MAX_SELECTED_EXPERTS];
        for item in gpu_work.into_iter().take(gpu_work_len) {
            let gpu_slot = item.gpu_slot().ok_or_else(|| {
                XrtError::Runtime("GPU MoE work item has no physical slot".to_string())
            })?;
            let slot = resident.slot(gpu_slot, descriptor.layer_index())?;
            if slot.logical_expert() != item.logical_expert() {
                return Err(XrtError::Runtime(format!(
                    "MoE logical expert {} was remapped to slot {gpu_slot} containing expert {}",
                    item.logical_expert(),
                    slot.logical_expert()
                )));
            }
            let canonical_index = item.canonical_index();
            let logical_expert = item.logical_expert();
            let graph_key = (graph_allowed && moe.graph_available(full_gpu_residency)).then(|| {
                self.moe_expert_graph_key(
                    descriptor,
                    slot.as_ref(),
                    usize::from(gpu_slot),
                    cache_mode,
                    scratch_generation,
                )
            });
            let graph_launch = graph_key
                .as_ref()
                .and_then(|key| moe.graph_cache.graph_for(key))
                .map(CudaGraphExec::launch);
            match graph_launch {
                Some(Ok(())) => {
                    self.moe_telemetry.record_graph_replay();
                }
                Some(Err(error)) => {
                    let reason = format!(
                        "CUDA MoE expert graph launch failed for layer {} expert {logical_expert}: {error}",
                        descriptor.layer_index()
                    );
                    moe.graph_fallback(reason.clone());
                    self.moe_telemetry.record_graph_fallback();
                    self.device.synchronize().map_err(|sync_error| {
                        XrtError::Cuda(format!(
                            "{reason}; CUDA stream recovery also failed: {sync_error}"
                        ))
                    })?;
                    let destination =
                        moe.expert_outputs
                            .get_mut(logical_expert)
                            .ok_or_else(|| {
                                XrtError::Runtime(format!(
                                    "MoE logical expert output {logical_expert} exceeds expert count {}",
                                    descriptor.expert_count()
                                ))
                            })?;
                    self.run_resident_moe_expert_into(
                        slot.as_ref(),
                        hidden_temp,
                        gate,
                        up,
                        destination,
                    )?;
                    self.moe_telemetry.record_graph_eager_call();
                }
                None => {
                    let captured = {
                        let destination =
                            moe.expert_outputs
                                .get_mut(logical_expert)
                                .ok_or_else(|| {
                                    XrtError::Runtime(format!(
                                        "MoE logical expert output {logical_expert} exceeds expert count {}",
                                        descriptor.expert_count()
                                    ))
                                })?;
                        self.run_resident_moe_expert_into(
                            slot.as_ref(),
                            hidden_temp,
                            gate,
                            up,
                            destination,
                        )?;
                        if graph_key.is_some() {
                            Some(unsafe {
                                // SAFETY: the expert weights are backend-owned for the
                                // placement epoch; input, gate/up scratch, and the
                                // logical-expert output are session-owned and stable
                                // until this cache is destroyed by `CudaDecodeScratch`.
                                self.device.capture_graph(|| {
                                    self.run_resident_moe_expert_into(
                                        slot.as_ref(),
                                        hidden_temp,
                                        gate,
                                        up,
                                        destination,
                                    )
                                })
                            })
                        } else {
                            None
                        }
                    };
                    self.moe_telemetry.record_graph_eager_call();
                    if let (Some(key), Some(captured)) = (graph_key, captured) {
                        match captured {
                            Ok(graph) => {
                                match reserve_cuda_graph_allocation(
                                    self.allocation_arena.as_ref(),
                                    &graph,
                                ) {
                                    Ok(allocation) => {
                                        let nodes = graph.node_count();
                                        let accounting_bytes = graph.accounting_bytes();
                                        moe.graph_cache.insert(
                                            key,
                                            graph,
                                            allocation,
                                            Arc::clone(&slot),
                                        );
                                        moe.mark_graph_captured();
                                        self.moe_telemetry.record_graph_capture();
                                        info!(
                                            layer = descriptor.layer_index(),
                                            logical_expert,
                                            gpu_slot,
                                            nodes,
                                            accounting_bytes,
                                            "captured CUDA MoE resident-expert subgraph"
                                        );
                                    }
                                    Err(error) => {
                                        moe.graph_fallback(error.to_string());
                                        self.moe_telemetry.record_graph_fallback();
                                    }
                                }
                            }
                            Err(error) => {
                                moe.graph_fallback(error.to_string());
                                self.moe_telemetry.record_graph_fallback();
                            }
                        }
                    }
                }
            }
            gpu_completed[canonical_index] = true;
        }

        let cpu_output = match cpu_join {
            Some(join) => match join.join() {
                Ok(output) => output,
                Err(error) => {
                    self.moe_telemetry.record_coordinator_failure();
                    return Err(error);
                }
            },
            None => CpuMoeLayerOutput {
                output_rows: [None; MAX_SELECTED_EXPERTS],
                output_row_count: 0,
                shared_result: None,
            },
        };

        self.device.zero_f32(&mut moe.accumulator)?;
        let staging = host.lock();
        let cpu_output_elements = cpu_output
            .output_row_count
            .checked_mul(embedding_length)
            .ok_or_else(|| XrtError::Runtime("CPU MoE upload size overflowed".to_string()))?;
        if cpu_output_elements != 0 {
            self.device.upload_f32_prefix_from_pinned(
                &staging.outputs,
                cpu_output_elements,
                &mut moe.cpu_outputs,
            )?;
            self.moe_telemetry
                .record_result_h2d(cpu_output_elements * std::mem::size_of::<f32>());
        }
        let mut canonical_index = 0usize;
        while canonical_index < selected {
            if let Some(output_row) = cpu_output.output_rows[canonical_index] {
                let mut row_count = 1usize;
                while canonical_index + row_count < selected
                    && cpu_output.output_rows[canonical_index + row_count]
                        == output_row.checked_add(row_count)
                {
                    row_count += 1;
                }
                self.device.packed_rows_add_assign_device(
                    &mut moe.accumulator,
                    &moe.cpu_outputs,
                    output_row,
                    row_count,
                )?;
                canonical_index += row_count;
                continue;
            }

            if gpu_completed[canonical_index] {
                let logical_expert = usize::try_from(route.logical_ids()[canonical_index])
                    .map_err(|_| {
                        XrtError::Runtime(format!(
                            "MoE logical expert {} does not fit the host index",
                            route.logical_ids()[canonical_index]
                        ))
                    })?;
                let contribution = moe.expert_outputs.get(logical_expert).ok_or_else(|| {
                    XrtError::Runtime(format!(
                        "MoE logical expert output {logical_expert} exceeds expert count {}",
                        descriptor.expert_count()
                    ))
                })?;
                self.device.scaled_row_add_assign_device(
                    &mut moe.accumulator,
                    contribution,
                    0,
                    route.weights()[canonical_index],
                )?;
            } else {
                return Err(XrtError::Runtime(format!(
                    "MoE canonical selection {canonical_index} produced no CPU or GPU output"
                )));
            }
            canonical_index += 1;
        }
        if let Some(output_row) = cpu_output.shared_result {
            if moe.shared_output.is_none() {
                return Err(XrtError::Runtime(
                    "CPU shared expert completed without a CUDA shared output slot".to_string(),
                ));
            }
            self.device.packed_rows_add_assign_device(
                &mut moe.accumulator,
                &moe.cpu_outputs,
                output_row,
                1,
            )?;
        }
        drop(staging);
        self.device
            .add_assign_device(&mut moe.accumulator, normed_post_attention)?;
        // The current layer no longer needs its input after the residual has
        // been formed. Recycle that allocation as the next layer's MoE
        // accumulator and return the completed accumulator as the new hidden
        // state. Captured expert graphs do not reference the accumulator.
        std::mem::swap(&mut moe.accumulator, &mut recycled_input);
        Ok(recycled_input)
    }

    fn run_layerwise_expert_group(
        &self,
        slot: &ResidentMoeExpertSlot,
        logical_expert: usize,
        ffn_inputs: &[CudaF32Buffer],
        routes: &[MoeRoutingRow],
        expert_outputs: &mut [Vec<CudaF32Buffer>],
        scratch: &mut CudaDecodeScratch,
    ) -> Result<usize> {
        if ffn_inputs.len() != routes.len() || routes.len() != expert_outputs.len() {
            return Err(XrtError::Runtime(
                "layerwise MoE expert group has inconsistent token dimensions".to_string(),
            ));
        }
        let CudaDecodeScratch { gate, up, .. } = scratch;
        let mut calls = 0usize;
        for (token_index, route) in routes.iter().enumerate() {
            for (route_slot, &selected_expert) in route.logical_ids().iter().enumerate() {
                if selected_expert as usize != logical_expert {
                    continue;
                }
                let destination = expert_outputs
                    .get_mut(token_index)
                    .and_then(|outputs| outputs.get_mut(route_slot))
                    .ok_or_else(|| {
                        XrtError::Runtime(
                            "layerwise MoE canonical output geometry changed".to_string(),
                        )
                    })?;
                self.run_resident_moe_expert_into(
                    slot,
                    &ffn_inputs[token_index],
                    gate,
                    up,
                    destination,
                )?;
                calls = calls.saturating_add(1);
            }
        }
        Ok(calls)
    }

    fn run_layerwise_shared_experts(
        &self,
        layer_index: usize,
        ffn_inputs: &[CudaF32Buffer],
    ) -> Result<Vec<Option<(CudaF32Buffer, f32)>>> {
        let model = self.require_cpu_reference_model()?;
        let Some(intermediate_size) = model.moe_shared_intermediate_size(layer_index) else {
            return Ok(std::iter::repeat_with(|| None)
                .take(ffn_inputs.len())
                .collect());
        };
        let mut outputs = Vec::with_capacity(ffn_inputs.len());
        let mut gate = try_zeroed_f32(
            intermediate_size,
            "layerwise CPU shared-expert gate scratch",
        )?;
        let mut up = try_zeroed_f32(intermediate_size, "layerwise CPU shared-expert up scratch")?;
        let mut output = try_zeroed_f32(
            self.config.embedding_length,
            "layerwise CPU shared-expert output",
        )?;
        for input in ffn_inputs {
            let host_input = self.device.download_f32(input)?;
            self.moe_telemetry.record_activation_d2h(input.byte_len());
            let weight = model.execute_shared_moe_expert_into(
                layer_index,
                &host_input,
                &mut gate,
                &mut up,
                &mut output,
            )?;
            let uploaded = self.device.upload_f32(&output)?;
            self.moe_telemetry.record_result_h2d(uploaded.byte_len());
            outputs.push(Some((uploaded, weight)));
        }
        Ok(outputs)
    }

    fn try_forward_batch_moe_layerwise(
        &self,
        token_ids: &[u32],
        start_position: usize,
        session: &mut BackendSession,
        embedding_overrides: &HashMap<usize, Vec<f32>>,
        all_logits: bool,
    ) -> Result<Option<Vec<f32>>> {
        let Some(layerwise) = &self.layerwise_moe_prefill else {
            return Ok(None);
        };
        if token_ids.len() < 2 {
            return Ok(None);
        }
        let (Some(layers), Some(output_weights)) = (&self.moe_layer_probes, &self.q8_0_probe)
        else {
            return Ok(None);
        };
        if layers.len() != self.config.block_count {
            return Ok(None);
        }
        Self::validate_embedding_overrides(
            token_ids.len(),
            self.config.embedding_length,
            embedding_overrides,
        )?;
        for &token_id in token_ids {
            if token_id as usize >= output_weights.vocab_size {
                return Err(XrtError::Model(format!(
                    "token id {token_id} exceeds embedding rows {}",
                    output_weights.vocab_size
                )));
            }
        }

        let total_len = cuda_total_len_after_batch(start_position, token_ids.len())?;
        let selected = self.config.expert_used_count.ok_or_else(|| {
            XrtError::InvalidMetadata("layerwise MoE plan is missing expert_used_count".to_string())
        })?;
        let working_buffer_count = selected.checked_add(3).ok_or_else(|| {
            XrtError::Runtime("layerwise MoE buffer count overflowed".to_string())
        })?;
        let working_elements = token_ids
            .len()
            .checked_mul(self.config.embedding_length)
            .and_then(|elements| elements.checked_mul(working_buffer_count))
            .ok_or_else(|| {
                XrtError::Runtime("layerwise MoE working element count overflowed".to_string())
            })?;
        let working_bytes = working_elements
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| {
                XrtError::Runtime("layerwise MoE working byte count overflowed".to_string())
            })?;
        let _working_allocation = self
            .allocation_arena
            .as_ref()
            .map(|arena| arena.reserve(GpuAllocationClass::Scratch, working_bytes))
            .transpose()?;
        let _staging_allocation = self
            .allocation_arena
            .as_ref()
            .map(|arena| {
                arena.reserve(
                    GpuAllocationClass::Staging,
                    layerwise.worst_case_staging_bytes,
                )
            })
            .transpose()?;

        session.cuda_graph_fallback(
            "layerwise MoE prefill uses exact eager grouped execution with bounded staging",
        );
        session.prepare_for_total_len(total_len)?;
        session.ensure_cuda_decode_scratch(
            &self.device,
            self.config.embedding_length,
            self.config.q_width(),
            self.config.kv_width(),
            self.config.feed_forward_length,
            output_weights.vocab_size,
            self.config.context_length,
            MoeScratchGeometry::from_config(&self.config)?,
            None,
        )?;

        let mut hidden = Vec::with_capacity(token_ids.len());
        for (token_index, &token_id) in token_ids.iter().enumerate() {
            hidden.push(
                if let Some(embedding) = embedding_overrides.get(&token_index) {
                    self.device.upload_f32(embedding)?
                } else {
                    self.embed_q8_0_probe_token(output_weights, token_id)?
                },
            );
        }
        let mut ffn_inputs = (0..token_ids.len())
            .map(|_| self.device.zeros_f32(self.config.embedding_length))
            .collect::<Result<Vec<_>>>()?;
        let mut expert_outputs = (0..token_ids.len())
            .map(|_| {
                (0..selected)
                    .map(|_| self.device.zeros_f32(self.config.embedding_length))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        let _placement_epoch = self.moe_placement_gate.read();
        let started = Instant::now();
        let mut weight_upload_bytes = 0u64;
        let mut repack_bytes = 0u64;

        let execution = (|| -> Result<Vec<f32>> {
            for (layer_index, weights) in layers.iter().enumerate() {
                for token_index in 0..token_ids.len() {
                    let position = cuda_batch_position(start_position, token_index)?;
                    let adaptive_is_hot =
                        session.cuda_adaptive_position_is_hot(position, total_len);
                    let (kv_cache, scratch) =
                        session.cuda_layer_cache_and_scratch_mut(layer_index)?;
                    if kv_cache.len() != position {
                        return Err(XrtError::Runtime(format!(
                            "layerwise MoE KV cache length mismatch at layer {layer_index}: expected {position}, found {}",
                            kv_cache.len()
                        )));
                    }
                    self.run_moe_attention_and_norm_with_scratch(
                        layer_index,
                        weights,
                        &hidden[token_index],
                        position,
                        adaptive_is_hot,
                        kv_cache,
                        scratch,
                    )?;
                    self.device.copy_f32_device(
                        &scratch.normed_post_attention,
                        &mut hidden[token_index],
                    )?;
                    self.device
                        .copy_f32_device(&scratch.hidden_temp, &mut ffn_inputs[token_index])?;
                }

                let mut routes = Vec::with_capacity(token_ids.len());
                for input in &ffn_inputs {
                    let logits = {
                        let scratch = session.cuda_decode_scratch_mut()?;
                        let moe = scratch.moe.as_mut().ok_or_else(|| {
                            XrtError::Runtime(
                                "layerwise MoE router scratch is unavailable".to_string(),
                            )
                        })?;
                        self.matvec_quant_resident_device_into(
                            &weights.router,
                            input,
                            &mut moe.router_logits,
                        )?;
                        self.device.download_f32(&moe.router_logits)?
                    };
                    let mut route = MoeRoutingRow::default();
                    self.require_cpu_reference_model()?.route_moe_logits(
                        layer_index,
                        &logits,
                        &mut route,
                    )?;
                    self.record_adaptive_moe_route(layer_index, &route)?;
                    routes.push(route);
                }

                let resident = weights.resident.read();
                let mut cold_experts = BTreeSet::new();
                let mut resident_calls = 0usize;
                let mut staged_calls = 0usize;
                for logical_expert in 0..weights.descriptor.expert_count() {
                    if let Some(gpu_slot) = resident.snapshot.gpu_slot_for(logical_expert) {
                        let slot = resident.slot(gpu_slot, layer_index)?;
                        resident_calls =
                            resident_calls.saturating_add(self.run_layerwise_expert_group(
                                slot.as_ref(),
                                logical_expert,
                                &ffn_inputs,
                                &routes,
                                &mut expert_outputs,
                                session.cuda_decode_scratch_mut()?,
                            )?);
                    } else if routes
                        .iter()
                        .any(|route| route.logical_ids().contains(&(logical_expert as u32)))
                    {
                        cold_experts.insert(logical_expert);
                    }
                }

                let cold_experts = cold_experts.into_iter().collect::<Vec<_>>();
                let source = GgufResidentTensorSource::new(&layerwise.gguf);
                let mut current = cold_experts
                    .first()
                    .copied()
                    .map(|expert| {
                        ResidentMoeExpertSlot::upload(
                            &layerwise.staging_devices[0],
                            &layerwise.gguf,
                            &source,
                            &self.config,
                            layer_index,
                            expert,
                        )
                        .map(|slot| (expert, 0usize, slot))
                    })
                    .transpose()?;
                for next_index in 1..=cold_experts.len() {
                    let next = cold_experts
                        .get(next_index)
                        .copied()
                        .map(|expert| {
                            let stream_index = next_index % 2;
                            ResidentMoeExpertSlot::upload(
                                &layerwise.staging_devices[stream_index],
                                &layerwise.gguf,
                                &source,
                                &self.config,
                                layer_index,
                                expert,
                            )
                            .map(|slot| (expert, stream_index, slot))
                        })
                        .transpose()?;
                    if let Some((expert, stream_index, slot)) = current.take() {
                        layerwise.staging_devices[stream_index].synchronize()?;
                        staged_calls =
                            staged_calls.saturating_add(self.run_layerwise_expert_group(
                                &slot,
                                expert,
                                &ffn_inputs,
                                &routes,
                                &mut expert_outputs,
                                session.cuda_decode_scratch_mut()?,
                            )?);
                        self.device.synchronize()?;
                        let bytes = layerwise.expert_bytes(layer_index, expert)?;
                        weight_upload_bytes =
                            weight_upload_bytes.checked_add(bytes).ok_or_else(|| {
                                XrtError::Runtime(
                                    "layerwise MoE upload telemetry overflowed".to_string(),
                                )
                            })?;
                        // All currently supported resident expert formats are
                        // transposed, decoded, or split before kernel use.
                        repack_bytes = repack_bytes.checked_add(bytes).ok_or_else(|| {
                            XrtError::Runtime(
                                "layerwise MoE repack telemetry overflowed".to_string(),
                            )
                        })?;
                    }
                    current = next;
                }
                self.moe_telemetry
                    .record_layerwise_plan(resident_calls, staged_calls);

                let mut shared_outputs =
                    self.run_layerwise_shared_experts(layer_index, &ffn_inputs)?;
                for token_index in 0..token_ids.len() {
                    let scratch = session.cuda_decode_scratch_mut()?;
                    let moe = scratch.moe.as_mut().ok_or_else(|| {
                        XrtError::Runtime(
                            "layerwise MoE accumulator scratch is unavailable".to_string(),
                        )
                    })?;
                    self.device.zero_f32(&mut moe.accumulator)?;
                    for route_slot in 0..selected {
                        let contribution = &mut expert_outputs[token_index][route_slot];
                        self.device.scale_assign_device(
                            contribution,
                            routes[token_index].weights()[route_slot],
                        )?;
                        self.device
                            .add_assign_device(&mut moe.accumulator, contribution)?;
                    }
                    if let Some((shared, weight)) = shared_outputs[token_index].as_mut() {
                        self.device.scale_assign_device(shared, *weight)?;
                        self.device
                            .add_assign_device(&mut moe.accumulator, shared)?;
                    }
                    self.device
                        .add_assign_device(&mut moe.accumulator, &hidden[token_index])?;
                    self.device
                        .copy_f32_device(&moe.accumulator, &mut hidden[token_index])?;
                }
            }

            let expected_output_len = if all_logits {
                cuda_all_logits_output_len(token_ids.len(), output_weights.vocab_size)?
            } else {
                output_weights.vocab_size
            };
            let mut output = Vec::with_capacity(expected_output_len);
            for (token_index, hidden) in hidden.iter().enumerate() {
                if !all_logits && token_index + 1 != token_ids.len() {
                    continue;
                }
                let scratch = session.cuda_decode_scratch_mut()?;
                self.device.rmsnorm_device_into(
                    hidden,
                    output_weights.output_norm.buffer(),
                    1,
                    output_weights.embedding_length,
                    self.config.rms_norm_eps,
                    &mut scratch.hidden_temp,
                )?;
                self.matvec_quant_resident_device_into(
                    &output_weights.output,
                    &scratch.hidden_temp,
                    &mut scratch.logits,
                )?;
                output.extend(self.device.download_f32(&scratch.logits)?);
            }
            Ok(output)
        })();

        match execution {
            Ok(output) => {
                let elapsed_micros =
                    u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX);
                self.moe_telemetry.record_layerwise_prefill(
                    token_ids.len(),
                    weight_upload_bytes,
                    repack_bytes,
                    elapsed_micros,
                );
                Ok(Some(output))
            }
            Err(error) => {
                let synchronized = self.device.synchronize();
                let rollback = session.truncate(start_position);
                match (synchronized, rollback) {
                    (Ok(()), Ok(())) => Err(error),
                    (sync, rollback) => Err(XrtError::Runtime(format!(
                        "layerwise MoE prefill failed ({error}); CUDA synchronization result={sync:?}; KV rollback result={rollback:?}"
                    ))),
                }
            }
        }
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
            CudaLayerKvStore::SharedQ8(cache) => {
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
            CudaLayerKvStore::SharedKeyQ4ValueQ8(cache) => {
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
            CudaLayerKvStore::SharedAgentAdaptive(cache) => {
                cache.append(adaptive_is_hot, &k, &v)?;
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

    #[allow(clippy::too_many_arguments)]
    fn run_qwen35_layer_with_scratch(
        &self,
        layer_index: usize,
        weights: &ResidentQwen35LayerWeights,
        input: &CudaF32Buffer,
        output: &mut CudaF32Buffer,
        _position: usize,
        params: &CudaDecodeParams,
        kv_cache: &mut CudaLayerKvStore,
        recurrent: &mut CudaDeltaNetState,
        attention: &mut CudaF32Buffer,
        normed_post_attention: &mut CudaF32Buffer,
        q: &mut CudaF32Buffer,
        q_temp: &mut CudaF32Buffer,
        k: &mut CudaF32Buffer,
        v: &mut CudaF32Buffer,
        hidden_temp: &mut CudaF32Buffer,
        kv_temp: &mut CudaF32Buffer,
        gate: &mut CudaF32Buffer,
        up: &mut CudaF32Buffer,
        qwen35: &mut CudaQwen35DecodeScratch,
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

        match &weights.attention {
            ResidentQwen35AttentionWeights::DeltaNet {
                attn_qkv,
                attn_gate,
                ssm_alpha,
                ssm_beta,
                ssm_a,
                ssm_dt_bias,
                ssm_norm,
                ssm_out,
                conv1d,
            } => {
                self.matvec_recurrent_qkv_resident_device_into(
                    attn_qkv,
                    normed_post_attention,
                    &mut qwen35.qkv,
                )?;
                self.matvec_quant_resident_device_into(
                    attn_gate,
                    normed_post_attention,
                    &mut qwen35.deltanet_gate,
                )?;
                self.matvec_quant_resident_device_into(
                    ssm_alpha,
                    normed_post_attention,
                    &mut qwen35.alpha,
                )?;
                self.matvec_quant_resident_device_into(
                    ssm_beta,
                    normed_post_attention,
                    &mut qwen35.beta,
                )?;

                let (
                    committed_conv,
                    pending_conv,
                    committed_recurrent,
                    pending_recurrent,
                    geometry,
                ) = recurrent.layer_buffers_mut(layer_index)?;
                self.device.deltanet_conv1d_device(
                    &qwen35.qkv,
                    committed_conv,
                    conv1d.buffer(),
                    pending_conv,
                    &mut qwen35.conv_output,
                    geometry,
                )?;
                self.device.deltanet_normalize_qk_device(
                    &mut qwen35.conv_output,
                    geometry,
                    config.rms_norm_eps,
                )?;
                self.device.deltanet_decay_beta_device(
                    &qwen35.alpha,
                    &qwen35.beta,
                    ssm_a.buffer(),
                    ssm_dt_bias.buffer(),
                    &mut qwen35.decays,
                    &mut qwen35.betas,
                    geometry,
                )?;
                self.device.deltanet_update_device(
                    &qwen35.conv_output,
                    committed_recurrent,
                    &qwen35.decays,
                    &qwen35.betas,
                    pending_recurrent,
                    &mut qwen35.deltanet_output,
                    geometry,
                )?;
                self.device.deltanet_gated_rmsnorm_device(
                    &mut qwen35.deltanet_output,
                    &qwen35.deltanet_gate,
                    ssm_norm.buffer(),
                    geometry,
                    config.rms_norm_eps,
                )?;
                self.matvec_quant_resident_device_into(
                    ssm_out,
                    &qwen35.deltanet_output,
                    hidden_temp,
                )?;
                self.device.add_device_into(input, hidden_temp, output)?;
            }
            ResidentQwen35AttentionWeights::Full {
                attn_qg,
                attn_k,
                attn_v,
                attn_output,
                attn_q_norm,
                attn_k_norm,
            } => {
                self.matvec_quant_resident_device_into(
                    attn_qg,
                    normed_post_attention,
                    &mut qwen35.qg,
                )?;
                self.matvec_quant_resident_device_into(attn_k, normed_post_attention, k)?;
                self.matvec_quant_resident_device_into(attn_v, normed_post_attention, v)?;
                self.device.qwen35_deinterleave_qg_device(
                    &qwen35.qg,
                    q,
                    &mut qwen35.attention_gate,
                    config.attention_head_count,
                    config.head_dim(),
                )?;
                self.device.rmsnorm_device_into(
                    q,
                    attn_q_norm.buffer(),
                    config.attention_head_count,
                    config.head_dim(),
                    config.rms_norm_eps,
                    q_temp,
                )?;
                self.device.rmsnorm_device_into(
                    k,
                    attn_k_norm.buffer(),
                    config.attention_head_count_kv,
                    config.head_dim(),
                    config.rms_norm_eps,
                    kv_temp,
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
                self.device.rope_device_with_decode_params(
                    kv_temp,
                    config.attention_head_count_kv,
                    config.head_dim(),
                    params,
                    config.rope_dimension_count,
                    config.rope_freq_base,
                    config.rope_freq_scale,
                )?;
                let cache = match kv_cache {
                    CudaLayerKvStore::F32(cache) => cache,
                    other => {
                        return Err(XrtError::Unsupported(format!(
                            "Qwen3.5 CUDA recurrent execution currently requires f32 KV for full-attention layers, found {}",
                            other.mode().as_str()
                        )));
                    }
                };
                self.device
                    .append_layer_kv_with_decode_params(cache, kv_temp, v, params)?;
                self.device.single_query_attention_with_decode_params_into(
                    q_temp,
                    cache,
                    params,
                    config.attention_head_count,
                    config.attention_head_count_kv,
                    config.head_dim(),
                    1.0 / (config.head_dim() as f32).sqrt(),
                    attention,
                )?;
                self.device
                    .sigmoid_mul_assign_device(attention, &qwen35.attention_gate)?;
                self.matvec_quant_resident_device_into(attn_output, attention, hidden_temp)?;
                self.device.add_device_into(input, hidden_temp, output)?;
            }
        }

        self.device.rmsnorm_device_into(
            output,
            weights.ffn_norm.buffer(),
            1,
            weights.embedding_length,
            config.rms_norm_eps,
            normed_post_attention,
        )?;
        self.matvec_quant_resident_device_into(&weights.ffn_gate, normed_post_attention, gate)?;
        self.matvec_quant_resident_device_into(&weights.ffn_up, normed_post_attention, up)?;
        self.device.silu_assign_device(gate)?;
        self.device.mul_assign_device(gate, up)?;
        self.matvec_quant_resident_device_into(&weights.ffn_down, gate, hidden_temp)?;
        self.device.add_assign_device(output, hidden_temp)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_qwen35_moe_attention_and_norm_with_scratch(
        &self,
        layer_index: usize,
        weights: &ResidentQwen35MoeLayerWeights,
        input: &CudaF32Buffer,
        kv_cache: &mut CudaLayerKvStore,
        recurrent: &mut CudaDeltaNetState,
        scratch: &mut CudaDecodeScratch,
    ) -> Result<()> {
        let config = &self.config;
        let CudaDecodeScratch {
            decode_params,
            layer_input_b,
            attention,
            normed_post_attention,
            q,
            q_temp,
            k,
            v,
            hidden_temp,
            kv_temp,
            qwen35,
            ..
        } = scratch;
        let qwen35 = qwen35.as_mut().ok_or_else(|| {
            XrtError::Runtime("Qwen3.5 hybrid-MoE CUDA scratch geometry is missing".to_string())
        })?;

        self.device.rmsnorm_device_into(
            input,
            weights.attn_norm.buffer(),
            1,
            weights.embedding_length,
            config.rms_norm_eps,
            normed_post_attention,
        )?;

        match &weights.attention {
            ResidentQwen35AttentionWeights::DeltaNet {
                attn_qkv,
                attn_gate,
                ssm_alpha,
                ssm_beta,
                ssm_a,
                ssm_dt_bias,
                ssm_norm,
                ssm_out,
                conv1d,
            } => {
                self.matvec_recurrent_qkv_resident_device_into(
                    attn_qkv,
                    normed_post_attention,
                    &mut qwen35.qkv,
                )?;
                self.matvec_quant_resident_device_into(
                    attn_gate,
                    normed_post_attention,
                    &mut qwen35.deltanet_gate,
                )?;
                self.matvec_quant_resident_device_into(
                    ssm_alpha,
                    normed_post_attention,
                    &mut qwen35.alpha,
                )?;
                self.matvec_quant_resident_device_into(
                    ssm_beta,
                    normed_post_attention,
                    &mut qwen35.beta,
                )?;

                let (
                    committed_conv,
                    pending_conv,
                    committed_recurrent,
                    pending_recurrent,
                    geometry,
                ) = recurrent.layer_buffers_mut(layer_index)?;
                self.device.deltanet_conv1d_device(
                    &qwen35.qkv,
                    committed_conv,
                    conv1d.buffer(),
                    pending_conv,
                    &mut qwen35.conv_output,
                    geometry,
                )?;
                self.device.deltanet_normalize_qk_device(
                    &mut qwen35.conv_output,
                    geometry,
                    config.rms_norm_eps,
                )?;
                self.device.deltanet_decay_beta_device(
                    &qwen35.alpha,
                    &qwen35.beta,
                    ssm_a.buffer(),
                    ssm_dt_bias.buffer(),
                    &mut qwen35.decays,
                    &mut qwen35.betas,
                    geometry,
                )?;
                self.device.deltanet_update_device(
                    &qwen35.conv_output,
                    committed_recurrent,
                    &qwen35.decays,
                    &qwen35.betas,
                    pending_recurrent,
                    &mut qwen35.deltanet_output,
                    geometry,
                )?;
                self.device.deltanet_gated_rmsnorm_device(
                    &mut qwen35.deltanet_output,
                    &qwen35.deltanet_gate,
                    ssm_norm.buffer(),
                    geometry,
                    config.rms_norm_eps,
                )?;
                self.matvec_quant_resident_device_into(
                    ssm_out,
                    &qwen35.deltanet_output,
                    hidden_temp,
                )?;
                self.device
                    .add_device_into(input, hidden_temp, layer_input_b)?;
            }
            ResidentQwen35AttentionWeights::Full {
                attn_qg,
                attn_k,
                attn_v,
                attn_output,
                attn_q_norm,
                attn_k_norm,
            } => {
                self.matvec_quant_resident_device_into(
                    attn_qg,
                    normed_post_attention,
                    &mut qwen35.qg,
                )?;
                self.matvec_quant_resident_device_into(attn_k, normed_post_attention, k)?;
                self.matvec_quant_resident_device_into(attn_v, normed_post_attention, v)?;
                self.device.qwen35_deinterleave_qg_device(
                    &qwen35.qg,
                    q,
                    &mut qwen35.attention_gate,
                    config.attention_head_count,
                    config.head_dim(),
                )?;
                self.device.rmsnorm_device_into(
                    q,
                    attn_q_norm.buffer(),
                    config.attention_head_count,
                    config.head_dim(),
                    config.rms_norm_eps,
                    q_temp,
                )?;
                self.device.rmsnorm_device_into(
                    k,
                    attn_k_norm.buffer(),
                    config.attention_head_count_kv,
                    config.head_dim(),
                    config.rms_norm_eps,
                    kv_temp,
                )?;
                self.device.rope_device_with_decode_params(
                    q_temp,
                    config.attention_head_count,
                    config.head_dim(),
                    decode_params,
                    config.rope_dimension_count,
                    config.rope_freq_base,
                    config.rope_freq_scale,
                )?;
                self.device.rope_device_with_decode_params(
                    kv_temp,
                    config.attention_head_count_kv,
                    config.head_dim(),
                    decode_params,
                    config.rope_dimension_count,
                    config.rope_freq_base,
                    config.rope_freq_scale,
                )?;
                let cache = match kv_cache {
                    CudaLayerKvStore::F32(cache) => cache,
                    other => {
                        return Err(XrtError::Unsupported(format!(
                            "Qwen3.5 hybrid-MoE CUDA execution requires f32 KV for full-attention layers, found {}",
                            other.mode().as_str()
                        )));
                    }
                };
                self.device
                    .append_layer_kv_with_decode_params(cache, kv_temp, v, decode_params)?;
                self.device.single_query_attention_with_decode_params_into(
                    q_temp,
                    cache,
                    decode_params,
                    config.attention_head_count,
                    config.attention_head_count_kv,
                    config.head_dim(),
                    1.0 / (config.head_dim() as f32).sqrt(),
                    attention,
                )?;
                self.device
                    .sigmoid_mul_assign_device(attention, &qwen35.attention_gate)?;
                self.matvec_quant_resident_device_into(attn_output, attention, hidden_temp)?;
                self.device
                    .add_device_into(input, hidden_temp, layer_input_b)?;
            }
        }

        self.device.rmsnorm_device_into(
            layer_input_b,
            weights.ffn_norm.buffer(),
            1,
            weights.embedding_length,
            config.rms_norm_eps,
            hidden_temp,
        )?;
        self.device
            .copy_f32_device(layer_input_b, normed_post_attention)
    }

    fn run_qwen35_token_ops(
        &self,
        output_weights: &ResidentQ8_0ProbeWeights,
        layer_weights: &[ResidentQwen35LayerWeights],
        layer_caches: &mut [CudaLayerKvStore],
        scratch: &mut CudaDecodeScratch,
        recurrent: &mut CudaDeltaNetState,
        position: usize,
        compute_logits: bool,
    ) -> Result<()> {
        if layer_caches.len() != layer_weights.len() {
            return Err(XrtError::Runtime(format!(
                "Qwen3.5 CUDA layer cache count {} does not match weight count {}",
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
            qwen35,
            ..
        } = scratch;
        let qwen35 = qwen35.as_mut().ok_or_else(|| {
            XrtError::Runtime("Qwen3.5 CUDA scratch geometry is missing".to_string())
        })?;

        let mut input_is_a = true;
        for (layer_index, (weights, cache)) in layer_weights.iter().zip(layer_caches).enumerate() {
            if input_is_a {
                self.run_qwen35_layer_with_scratch(
                    layer_index,
                    weights,
                    layer_input_a,
                    layer_input_b,
                    position,
                    decode_params,
                    cache,
                    recurrent,
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
                    qwen35,
                )?;
            } else {
                self.run_qwen35_layer_with_scratch(
                    layer_index,
                    weights,
                    layer_input_b,
                    layer_input_a,
                    position,
                    decode_params,
                    cache,
                    recurrent,
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
                    qwen35,
                )?;
            }
            input_is_a = !input_is_a;
        }

        if compute_logits {
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
            self.matvec_quant_resident_device_into(&output_weights.output, hidden_temp, logits)?;
        }
        Ok(())
    }

    fn draft_qwen35_mtp_greedy(
        &self,
        next_token_id: u32,
        max_draft_tokens: usize,
        session: &mut BackendSession,
    ) -> Result<Option<Vec<u32>>> {
        let (Some(mtp), Some(output_weights), Some(trunk_layers)) = (
            &self.qwen35_mtp_probe,
            &self.q8_0_probe,
            &self.qwen35_layer_probes,
        ) else {
            return Ok(None);
        };
        if max_draft_tokens == 0 {
            return Ok(Some(Vec::new()));
        }
        if next_token_id as usize >= output_weights.vocab_size {
            return Err(XrtError::Model(format!(
                "MTP input token {next_token_id} exceeds embedding rows {}",
                output_weights.vocab_size
            )));
        }
        if session.cache_mode() != KvCacheMode::F32 {
            return Ok(None);
        }
        let kv_capacity = session.cuda_kv_capacity().ok_or_else(|| {
            XrtError::Runtime("Qwen MTP requires allocated CUDA KV capacity".to_string())
        })?;
        session.ensure_cuda_decode_scratch(
            &self.device,
            self.config.embedding_length,
            self.config.q_width(),
            self.config.kv_width(),
            self.config.feed_forward_length,
            output_weights.vocab_size,
            kv_capacity,
            None,
            Qwen35ScratchGeometry::from_config(&self.config)?,
        )?;

        let execution = (|| -> Result<Vec<u32>> {
            let (mtp_cache, scratch, recurrent) = session.cuda_qwen35_mtp_parts_mut()?;
            mtp_cache.truncate(0)?;
            let final_hidden_is_a = trunk_layers.len() % 2 == 0;
            {
                let qwen35 = scratch.qwen35.as_mut().ok_or_else(|| {
                    XrtError::Runtime("Qwen MTP scratch geometry is missing".to_string())
                })?;
                if final_hidden_is_a {
                    self.device
                        .copy_f32_device(&scratch.layer_input_a, &mut qwen35.mtp_hidden)?;
                } else {
                    self.device
                        .copy_f32_device(&scratch.layer_input_b, &mut qwen35.mtp_hidden)?;
                }
            }

            let mut token = next_token_id;
            let mut draft = Vec::with_capacity(max_draft_tokens);
            for depth in 0..max_draft_tokens {
                self.device.update_decode_params(
                    &mut scratch.decode_params,
                    token,
                    depth,
                    depth + 1,
                    0,
                )?;
                self.embed_probe_with_decode_params_into(
                    output_weights,
                    &scratch.decode_params,
                    &mut scratch.layer_input_a,
                )?;
                self.device.rmsnorm_device_into(
                    &scratch.layer_input_a,
                    mtp.enorm.buffer(),
                    1,
                    self.config.embedding_length,
                    self.config.rms_norm_eps,
                    &mut scratch.normed_post_attention,
                )?;
                {
                    let qwen35 = scratch.qwen35.as_mut().ok_or_else(|| {
                        XrtError::Runtime("Qwen MTP scratch geometry is missing".to_string())
                    })?;
                    self.device.rmsnorm_device_into(
                        &qwen35.mtp_hidden,
                        mtp.hnorm.buffer(),
                        1,
                        self.config.embedding_length,
                        self.config.rms_norm_eps,
                        &mut scratch.layer_input_b,
                    )?;
                    self.device.copy_f32_device_into_range(
                        &scratch.normed_post_attention,
                        &mut qwen35.mtp_concat,
                        0,
                    )?;
                    self.device.copy_f32_device_into_range(
                        &scratch.layer_input_b,
                        &mut qwen35.mtp_concat,
                        self.config.embedding_length,
                    )?;
                    self.matvec_quant_resident_device_into(
                        &mtp.eh_proj,
                        &qwen35.mtp_concat,
                        &mut scratch.layer_input_b,
                    )?;
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
                    qwen35,
                    ..
                } = scratch;
                let qwen35 = qwen35.as_mut().ok_or_else(|| {
                    XrtError::Runtime("Qwen MTP scratch geometry is missing".to_string())
                })?;
                self.run_qwen35_layer_with_scratch(
                    0,
                    &mtp.layer,
                    layer_input_b,
                    layer_input_a,
                    depth,
                    decode_params,
                    mtp_cache,
                    recurrent,
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
                    qwen35,
                )?;
                self.device
                    .copy_f32_device(layer_input_a, &mut qwen35.mtp_hidden)?;
                self.device.rmsnorm_device_into(
                    &qwen35.mtp_hidden,
                    mtp.shared_head_norm.buffer(),
                    1,
                    self.config.embedding_length,
                    self.config.rms_norm_eps,
                    hidden_temp,
                )?;
                self.matvec_quant_resident_device_into(
                    &output_weights.output,
                    hidden_temp,
                    logits,
                )?;
                let host_logits = self.device.download_f32(logits)?;
                token = host_logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, left), (_, right)| left.total_cmp(right))
                    .map(|(index, _)| index as u32)
                    .ok_or_else(|| XrtError::Runtime("MTP produced empty logits".to_string()))?;
                draft.push(token);
                self.commit_qwen35_graph_caches(
                    std::slice::from_ref(&mtp.layer),
                    std::slice::from_mut(mtp_cache),
                    depth,
                )?;
            }
            Ok(draft)
        })();

        let cleanup = session
            .cuda_layer_cache_mut(self.config.block_count)
            .and_then(|cache| cache.truncate(0));
        match (execution, cleanup) {
            (Ok(draft), Ok(())) => Ok(Some(draft)),
            (Err(error), Ok(())) => Err(error),
            (Ok(_), Err(error)) => Err(error),
            (Err(error), Err(cleanup_error)) => Err(XrtError::Runtime(format!(
                "Qwen MTP draft failed ({error}); cache cleanup also failed ({cleanup_error})"
            ))),
        }
    }

    fn validate_qwen35_graph_caches(
        layer_weights: &[ResidentQwen35LayerWeights],
        layer_caches: &[CudaLayerKvStore],
        position: usize,
        capacity: usize,
    ) -> Result<()> {
        if layer_caches.len() != layer_weights.len() {
            return Err(XrtError::Runtime(format!(
                "Qwen3.5 CUDA layer cache count {} does not match weight count {}",
                layer_caches.len(),
                layer_weights.len()
            )));
        }
        for (layer, (weights, cache)) in layer_weights.iter().zip(layer_caches).enumerate() {
            let CudaLayerKvStore::F32(cache) = cache else {
                return Err(XrtError::Unsupported(format!(
                    "Qwen3.5 CUDA graph layer {layer} requires f32 KV"
                )));
            };
            let expected_len = match weights.attention {
                ResidentQwen35AttentionWeights::DeltaNet { .. } => 0,
                ResidentQwen35AttentionWeights::Full { .. } => position,
            };
            if cache.len() != expected_len {
                return Err(XrtError::Runtime(format!(
                    "Qwen3.5 CUDA KV cache length mismatch at layer {layer}: expected {expected_len}, found {}",
                    cache.len()
                )));
            }
            if cache.capacity() != capacity {
                return Err(XrtError::Runtime(format!(
                    "Qwen3.5 CUDA KV cache capacity mismatch at layer {layer}: expected {capacity}, found {}",
                    cache.capacity()
                )));
            }
        }
        Ok(())
    }

    fn commit_qwen35_graph_caches(
        &self,
        layer_weights: &[ResidentQwen35LayerWeights],
        layer_caches: &mut [CudaLayerKvStore],
        position: usize,
    ) -> Result<()> {
        for (layer, (weights, cache)) in layer_weights
            .iter()
            .zip(layer_caches.iter_mut())
            .enumerate()
        {
            if matches!(
                weights.attention,
                ResidentQwen35AttentionWeights::Full { .. }
            ) {
                let CudaLayerKvStore::F32(cache) = cache else {
                    return Err(XrtError::Unsupported(format!(
                        "Qwen3.5 CUDA graph layer {layer} requires f32 KV"
                    )));
                };
                self.device.commit_layer_kv_graph_append(cache, position)?;
            }
        }
        Ok(())
    }

    fn commit_qwen35_moe_caches(
        &self,
        layer_weights: &[ResidentQwen35MoeLayerWeights],
        layer_caches: &mut [CudaLayerKvStore],
        position: usize,
    ) -> Result<()> {
        if layer_caches.len() != layer_weights.len() {
            return Err(XrtError::Runtime(format!(
                "Qwen3.5 hybrid-MoE CUDA layer cache count {} does not match weight count {}",
                layer_caches.len(),
                layer_weights.len()
            )));
        }
        for (layer, (weights, cache)) in layer_weights
            .iter()
            .zip(layer_caches.iter_mut())
            .enumerate()
        {
            if matches!(
                weights.attention,
                ResidentQwen35AttentionWeights::Full { .. }
            ) {
                let CudaLayerKvStore::F32(cache) = cache else {
                    return Err(XrtError::Unsupported(format!(
                        "Qwen3.5 hybrid-MoE CUDA full-attention layer {layer} requires f32 KV"
                    )));
                };
                self.device.commit_layer_kv_graph_append(cache, position)?;
            }
        }
        Ok(())
    }

    fn run_qwen35_graph_ops(
        &self,
        output_weights: &ResidentQ8_0ProbeWeights,
        layer_weights: &[ResidentQwen35LayerWeights],
        layer_caches: &mut [CudaLayerKvStore],
        scratch: &mut CudaDecodeScratch,
        recurrent: &mut CudaDeltaNetState,
        position: usize,
    ) -> Result<()> {
        self.embed_probe_with_decode_params_into(
            output_weights,
            &scratch.decode_params,
            &mut scratch.layer_input_a,
        )?;
        self.run_qwen35_token_ops(
            output_weights,
            layer_weights,
            layer_caches,
            scratch,
            recurrent,
            position,
            true,
        )
    }

    fn try_qwen35_graph_decode(
        &self,
        token_id: u32,
        position: usize,
        session: &mut BackendSession,
        output_weights: &ResidentQ8_0ProbeWeights,
        layer_weights: &[ResidentQwen35LayerWeights],
    ) -> Result<Option<Vec<f32>>> {
        if !session.cuda_graph_decode_ready() {
            return Ok(None);
        }
        let kv_capacity = session.cuda_kv_capacity().ok_or_else(|| {
            XrtError::Runtime("Qwen3.5 CUDA Graph decode requires allocated KV caches".to_string())
        })?;
        let recurrent_generation = session.recurrent_buffer_generation().ok_or_else(|| {
            XrtError::Runtime(
                "Qwen3.5 CUDA Graph decode requires prepared recurrent state".to_string(),
            )
        })?;
        let scratch_generation = session.cuda_batch_graph_epoch().ok_or_else(|| {
            XrtError::Runtime("Qwen3.5 CUDA Graph received a CPU backend session".to_string())
        })?;
        let key = self.qwen35_graph_key(
            output_weights,
            layer_weights,
            KvCacheMode::F32,
            kv_capacity,
            scratch_generation,
            recurrent_generation,
        );
        if session.cuda_graph_has_executable_for(&key) {
            let _capture_guard = self.qwen35_capture_gate.read();
            return self.try_qwen35_graph_decode_locked(
                token_id,
                position,
                session,
                output_weights,
                layer_weights,
                kv_capacity,
                recurrent_generation,
                key,
                false,
            );
        }

        // CUDA stream capture forbids unrelated submissions to the captured
        // stream. Only the warm/capture window is exclusive; established graph
        // launches use a shared read guard above.
        let _capture_guard = self.qwen35_capture_gate.write();
        self.try_qwen35_graph_decode_locked(
            token_id,
            position,
            session,
            output_weights,
            layer_weights,
            kv_capacity,
            recurrent_generation,
            key,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn try_qwen35_graph_decode_locked(
        &self,
        token_id: u32,
        position: usize,
        session: &mut BackendSession,
        output_weights: &ResidentQ8_0ProbeWeights,
        layer_weights: &[ResidentQwen35LayerWeights],
        kv_capacity: usize,
        recurrent_generation: u8,
        key: CudaDecodeGraphKey,
        allow_capture: bool,
    ) -> Result<Option<Vec<f32>>> {
        let graph_allocation_arena = session.cuda_allocation_arena();
        let (graph_state, layer_caches, scratch, recurrent) =
            session.cuda_qwen35_graph_parts_mut()?;
        if !graph_state.is_enabled() {
            return Ok(None);
        }
        if !allow_capture && !graph_state.has_executable_for(&key) {
            return Ok(None);
        }
        Self::validate_qwen35_graph_caches(layer_weights, layer_caches, position, kv_capacity)?;
        // Parameter upload and graph launch share the owning stream. The
        // logits download synchronizes replay before host-side handle swaps.
        unsafe {
            self.device.update_decode_params_async(
                &mut scratch.decode_params,
                token_id,
                position,
                cuda_total_len_for_position(position)?,
                0,
            )?;
        }
        recurrent.begin_token(position)?;

        if let Some(launch_result) = graph_state.executable_for(&key).map(CudaGraphExec::launch) {
            launch_result.map_err(|error| {
                graph_state.fallback(error.to_string());
                XrtError::Cuda(format!(
                    "Qwen3.5 CUDA Graph launch failed before token commit: {error}"
                ))
            })?;
            let logits = self.device.download_f32(&scratch.logits)?;
            self.commit_qwen35_graph_caches(layer_weights, layer_caches, position)?;
            recurrent.commit_token(position)?;
            return Ok(Some(logits));
        }

        if !allow_capture {
            return Ok(None);
        }
        self.run_qwen35_graph_ops(
            output_weights,
            layer_weights,
            layer_caches,
            scratch,
            recurrent,
            position,
        )?;
        let logits = self.device.download_f32(&scratch.logits)?;
        let captured = unsafe {
            self.device.capture_graph(|| {
                self.run_qwen35_graph_ops(
                    output_weights,
                    layer_weights,
                    layer_caches,
                    scratch,
                    recurrent,
                    position,
                )
            })
        };
        match captured {
            Ok(graph) => {
                match reserve_cuda_graph_allocation(graph_allocation_arena.as_ref(), &graph) {
                    Ok(allocation) => {
                        info!(
                            nodes = graph.node_count(),
                            accounting_bytes = graph.accounting_bytes(),
                            recurrent_generation,
                            "captured Qwen3.5 CUDA recurrent decode graph"
                        );
                        graph_state.captured(
                            key,
                            graph,
                            Vec::new(),
                            Vec::new(),
                            Vec::new(),
                            Vec::new(),
                            allocation,
                        );
                    }
                    Err(error) => {
                        graph_state.fallback(error.to_string());
                        tracing::warn!(
                            "Qwen3.5 CUDA Graph admission failed; retaining transactional eager execution: {error}"
                        );
                    }
                }
            }
            Err(error) => {
                graph_state.fallback(error.to_string());
                tracing::warn!(
                    "Qwen3.5 CUDA Graph capture failed; retaining transactional eager execution: {error}"
                );
            }
        }
        self.commit_qwen35_graph_caches(layer_weights, layer_caches, position)?;
        recurrent.commit_token(position)?;
        Ok(Some(logits))
    }

    #[allow(clippy::too_many_arguments)]
    fn try_forward_token_qwen35_with_logits(
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
        let (Some(layer_weights), Some(output_weights)) =
            (&self.qwen35_layer_probes, &self.q8_0_probe)
        else {
            return Ok(false);
        };
        if max_layers.is_some_and(|layers| layers != self.config.block_count) {
            return Err(XrtError::Unsupported(
                "Qwen3.5 CUDA draft execution requires the complete recurrent/full-attention layer schedule"
                    .to_string(),
            ));
        }
        if session.cache_mode() != KvCacheMode::F32 {
            return Err(XrtError::Unsupported(format!(
                "Qwen3.5 CUDA recurrent execution currently requires XRT_KV_CACHE_MODE=f32, found {}",
                session.cache_mode().as_str()
            )));
        }
        if embedding_override.is_none() && token_id as usize >= output_weights.vocab_size {
            return Err(XrtError::Model(format!(
                "token id {token_id} exceeds embedding rows {}",
                output_weights.vocab_size
            )));
        }
        let execution = (|| -> Result<Option<Vec<f32>>> {
            let total_len = adaptive_total_len.max(cuda_total_len_for_position(position)?);
            session.prepare_for_total_len(total_len)?;
            {
                let _capture_guard = self.qwen35_capture_gate.read();
                session.prepare_recurrent_state()?;
                let kv_capacity = session.cuda_kv_capacity().ok_or_else(|| {
                    XrtError::Runtime(
                        "Qwen3.5 CUDA execution requires allocated KV capacity".to_string(),
                    )
                })?;
                session.ensure_cuda_decode_scratch(
                    &self.device,
                    self.config.embedding_length,
                    self.config.q_width(),
                    self.config.kv_width(),
                    self.config.feed_forward_length,
                    output_weights.vocab_size,
                    kv_capacity,
                    None,
                    Qwen35ScratchGeometry::from_config(&self.config)?,
                )?;
            }

            if allow_graph_decode && compute_logits && embedding_override.is_none() {
                if let Some(logits) = self.try_qwen35_graph_decode(
                    token_id,
                    position,
                    session,
                    output_weights,
                    layer_weights,
                )? {
                    return Ok(Some(logits));
                }
            }

            let _capture_guard = self.qwen35_capture_gate.read();
            {
                let (layer_caches, scratch, recurrent) = session.cuda_qwen35_parts_mut()?;
                self.device.update_decode_params(
                    &mut scratch.decode_params,
                    if embedding_override.is_some() {
                        0
                    } else {
                        token_id
                    },
                    position,
                    cuda_total_len_for_position(position)?,
                    0,
                )?;
                if let Some(embedding) = embedding_override {
                    self.device
                        .upload_f32_into(embedding, &mut scratch.layer_input_a)?;
                } else {
                    self.embed_probe_with_decode_params_into(
                        output_weights,
                        &scratch.decode_params,
                        &mut scratch.layer_input_a,
                    )?;
                }
                recurrent.begin_token(position)?;
                self.run_qwen35_token_ops(
                    output_weights,
                    layer_weights,
                    layer_caches,
                    scratch,
                    recurrent,
                    position,
                    compute_logits,
                )?;
            }

            let logits = if compute_logits {
                Some(
                    self.device
                        .download_f32(&session.cuda_decode_scratch_mut()?.logits)?,
                )
            } else {
                None
            };
            {
                let (layer_caches, _, _) = session.cuda_qwen35_parts_mut()?;
                self.commit_qwen35_graph_caches(layer_weights, layer_caches, position)?;
            }
            session
                .cuda_recurrent_state_mut()?
                .ok_or_else(|| {
                    XrtError::Runtime(
                        "Qwen3.5 CUDA recurrent state disappeared before commit".to_string(),
                    )
                })?
                .commit_token(position)?;
            Ok(logits)
        })();

        match execution {
            Ok(logits) => {
                if let Some(logits) = logits {
                    *output_logits = logits;
                } else {
                    output_logits.clear();
                }
                Ok(true)
            }
            Err(error) => {
                if let Ok(Some(recurrent)) = session.cuda_recurrent_state_mut() {
                    recurrent.abort_token();
                }
                let rollback = session
                    .truncate(position)
                    .and_then(|_| session.validate_recurrent_position(position));
                match rollback {
                    Ok(()) => Err(error),
                    Err(rollback_error) => {
                        let reason = format!(
                            "Qwen3.5 CUDA forward failed ({error}); rollback to token boundary {position} failed ({rollback_error})"
                        );
                        session.poison_recurrent_state(reason.clone());
                        Err(XrtError::Runtime(format!(
                            "{reason}; session is poisoned and must be reset"
                        )))
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn try_forward_token_qwen35_moe_with_logits(
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
        let (Some(layer_weights), Some(output_weights)) =
            (&self.qwen35_moe_layer_probes, &self.q8_0_probe)
        else {
            return Ok(false);
        };
        if layer_weights.len() != self.config.block_count {
            return Ok(false);
        }
        if max_layers.is_some_and(|layers| layers != self.config.block_count) {
            return Err(XrtError::Unsupported(
                "Qwen3.5 hybrid-MoE CUDA draft execution requires the complete recurrent/full-attention layer schedule"
                    .to_string(),
            ));
        }
        if session.cache_mode() != KvCacheMode::F32 {
            return Err(XrtError::Unsupported(format!(
                "Qwen3.5 hybrid-MoE CUDA recurrent execution currently requires XRT_KV_CACHE_MODE=f32, found {}",
                session.cache_mode().as_str()
            )));
        }
        if embedding_override.is_none() && token_id as usize >= output_weights.vocab_size {
            return Err(XrtError::Model(format!(
                "token id {token_id} exceeds embedding rows {}",
                output_weights.vocab_size
            )));
        }

        let _placement_epoch = self.moe_placement_gate.read();
        let execution = (|| -> Result<Option<Vec<f32>>> {
            let total_len = adaptive_total_len.max(cuda_total_len_for_position(position)?);
            session.prepare_for_total_len(total_len)?;
            {
                let _capture_guard = self.qwen35_capture_gate.read();
                session.prepare_recurrent_state()?;
                let kv_capacity = session.cuda_kv_capacity().ok_or_else(|| {
                    XrtError::Runtime(
                        "Qwen3.5 hybrid-MoE CUDA execution requires allocated KV capacity"
                            .to_string(),
                    )
                })?;
                session.ensure_cuda_decode_scratch(
                    &self.device,
                    self.config.embedding_length,
                    self.config.q_width(),
                    self.config.kv_width(),
                    self.config.feed_forward_length,
                    output_weights.vocab_size,
                    kv_capacity,
                    MoeScratchGeometry::from_config(&self.config)?,
                    Qwen35ScratchGeometry::from_config(&self.config)?,
                )?;
            }
            let scratch_generation = session.cuda_batch_graph_epoch().ok_or_else(|| {
                XrtError::Runtime(
                    "Qwen3.5 hybrid-MoE graph received a CPU backend session".to_string(),
                )
            })?;

            let _capture_guard = self.qwen35_capture_gate.read();
            let mut hidden = if let Some(embedding) = embedding_override {
                self.device.upload_f32(embedding)?
            } else {
                self.embed_q8_0_probe_token(output_weights, token_id)?
            };
            {
                let (layer_caches, scratch, recurrent) = session.cuda_qwen35_parts_mut()?;
                self.device.update_decode_params(
                    &mut scratch.decode_params,
                    if embedding_override.is_some() {
                        0
                    } else {
                        token_id
                    },
                    position,
                    cuda_total_len_for_position(position)?,
                    0,
                )?;
                recurrent.begin_token(position)?;
                for (layer_index, (weights, cache)) in
                    layer_weights.iter().zip(layer_caches).enumerate()
                {
                    self.run_qwen35_moe_attention_and_norm_with_scratch(
                        layer_index,
                        weights,
                        &hidden,
                        cache,
                        recurrent,
                        scratch,
                    )?;
                    hidden = self.run_moe_ffn_with_scratch(
                        weights,
                        hidden,
                        scratch,
                        allow_graph_decode,
                        KvCacheMode::F32,
                        scratch_generation,
                    )?;
                }
                if compute_logits {
                    self.device.rmsnorm_device_into(
                        &hidden,
                        output_weights.output_norm.buffer(),
                        1,
                        output_weights.embedding_length,
                        self.config.rms_norm_eps,
                        &mut scratch.hidden_temp,
                    )?;
                    self.matvec_quant_resident_device_into(
                        &output_weights.output,
                        &scratch.hidden_temp,
                        &mut scratch.logits,
                    )?;
                }
            }

            let logits = if compute_logits {
                Some(
                    self.device
                        .download_f32(&session.cuda_decode_scratch_mut()?.logits)?,
                )
            } else {
                None
            };
            {
                let (layer_caches, _, _) = session.cuda_qwen35_parts_mut()?;
                self.commit_qwen35_moe_caches(layer_weights, layer_caches, position)?;
            }
            session
                .cuda_recurrent_state_mut()?
                .ok_or_else(|| {
                    XrtError::Runtime(
                        "Qwen3.5 hybrid-MoE CUDA recurrent state disappeared before commit"
                            .to_string(),
                    )
                })?
                .commit_token(position)?;
            Ok(logits)
        })();

        match execution {
            Ok(logits) => {
                if let Some(logits) = logits {
                    *output_logits = logits;
                } else {
                    output_logits.clear();
                }
                Ok(true)
            }
            Err(error) => {
                if let Ok(Some(recurrent)) = session.cuda_recurrent_state_mut() {
                    recurrent.abort_token();
                }
                let rollback = session
                    .truncate(position)
                    .and_then(|_| session.validate_recurrent_position(position));
                match rollback {
                    Ok(()) => Err(error),
                    Err(rollback_error) => {
                        let reason = format!(
                            "Qwen3.5 hybrid-MoE CUDA forward failed ({error}); rollback to token boundary {position} failed ({rollback_error})"
                        );
                        session.poison_recurrent_state(reason.clone());
                        Err(XrtError::Runtime(format!(
                            "{reason}; session is poisoned and must be reset"
                        )))
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn try_forward_token_moe_with_logits(
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
        let (Some(layer_weights), Some(output_weights)) =
            (&self.moe_layer_probes, &self.q8_0_probe)
        else {
            return Ok(false);
        };
        let config = &self.config;
        if layer_weights.len() != config.block_count {
            return Ok(false);
        }
        let layer_count = max_layers.unwrap_or(config.block_count);
        if layer_count > config.block_count {
            return Err(XrtError::Runtime(format!(
                "CUDA MoE draft layer count {layer_count} exceeds model layer count {}",
                config.block_count
            )));
        }
        if embedding_override.is_none() && token_id as usize >= output_weights.vocab_size {
            return Err(XrtError::Model(format!(
                "token id {token_id} exceeds embedding rows {}",
                output_weights.vocab_size
            )));
        }
        // Per-expert graph capture uses the device-wide cudarc stream. Keep a
        // standard-MoE decode exclusive for the full allocation/capture
        // window so another request cannot prepare or destroy session buffers
        // while this stream is being captured.
        let _moe_graph_execution_guard = (allow_graph_decode
            && self.cuda_graph_mode != CudaGraphMode::Disabled)
            .then(|| self.moe_graph_execution_gate.lock());
        let _placement_epoch = self.moe_placement_gate.read();
        let execution = (|| -> Result<bool> {
            let prepare_total_len = adaptive_total_len.max(cuda_total_len_for_position(position)?);
            session.prepare_for_total_len(prepare_total_len)?;
            session.ensure_cuda_decode_scratch(
                &self.device,
                config.embedding_length,
                config.q_width(),
                config.kv_width(),
                config.feed_forward_length,
                output_weights.vocab_size,
                config.context_length,
                MoeScratchGeometry::from_config(config)?,
                None,
            )?;
            let cache_mode = session.cache_mode();
            let scratch_generation = session.cuda_batch_graph_epoch().ok_or_else(|| {
                XrtError::Runtime("CUDA MoE graph received a CPU backend session".to_string())
            })?;
            let mut hidden = if let Some(embedding) = embedding_override {
                self.device.upload_f32(embedding)?
            } else {
                self.embed_q8_0_probe_token(output_weights, token_id)?
            };

            for (layer_index, weights) in layer_weights.iter().take(layer_count).enumerate() {
                let adaptive_is_hot =
                    session.cuda_adaptive_position_is_hot(position, adaptive_total_len);
                let (kv_cache, scratch) = session.cuda_layer_cache_and_scratch_mut(layer_index)?;
                if kv_cache.len() != position {
                    return Err(XrtError::Runtime(format!(
                        "CUDA MoE KV cache length mismatch at layer {layer_index}: expected {position}, found {}",
                        kv_cache.len()
                    )));
                }
                self.run_moe_attention_and_norm_with_scratch(
                    layer_index,
                    weights,
                    &hidden,
                    position,
                    adaptive_is_hot,
                    kv_cache,
                    scratch,
                )?;
                hidden = self.run_moe_ffn_with_scratch(
                    weights,
                    hidden,
                    scratch,
                    allow_graph_decode,
                    cache_mode,
                    scratch_generation,
                )?;
            }

            if !compute_logits {
                output_logits.clear();
                return Ok(true);
            }
            let scratch = session.cuda_decode_scratch_mut()?;
            self.device.rmsnorm_device_into(
                &hidden,
                output_weights.output_norm.buffer(),
                1,
                output_weights.embedding_length,
                config.rms_norm_eps,
                &mut scratch.hidden_temp,
            )?;
            self.matvec_quant_resident_device_into(
                &output_weights.output,
                &scratch.hidden_temp,
                &mut scratch.logits,
            )?;
            *output_logits = self.device.download_f32(&scratch.logits)?;
            Ok(true)
        })();

        match execution {
            Ok(executed) => Ok(executed),
            Err(error) => {
                if let Err(rollback_error) = session.truncate(position) {
                    return Err(XrtError::Runtime(format!(
                        "CUDA MoE execution failed: {error}; KV rollback to {position} also failed: {rollback_error}"
                    )));
                }
                Err(error)
            }
        }
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
        mut output_hidden: Option<&mut Vec<f32>>,
    ) -> Result<bool> {
        let config = &self.config;
        if output_hidden.is_some() && (config.is_hybrid() || config.is_moe() || config.is_gemma4())
        {
            return Err(XrtError::Unsupported(
                "hidden-state extraction currently requires a standard dense CUDA model"
                    .to_string(),
            ));
        }
        if config.is_hybrid() {
            if config.is_moe() {
                return self.try_forward_token_qwen35_moe_with_logits(
                    token_id,
                    position,
                    session,
                    output_logits,
                    compute_logits,
                    allow_graph_decode,
                    embedding_override,
                    adaptive_total_len,
                    max_layers,
                );
            }
            return self.try_forward_token_qwen35_with_logits(
                token_id,
                position,
                session,
                output_logits,
                compute_logits,
                allow_graph_decode,
                embedding_override,
                adaptive_total_len,
                max_layers,
            );
        }
        if config.is_moe() {
            return self.try_forward_token_moe_with_logits(
                token_id,
                position,
                session,
                output_logits,
                compute_logits,
                allow_graph_decode,
                embedding_override,
                adaptive_total_len,
                max_layers,
            );
        }
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
            None,
            None,
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
        if !compute_logits && output_hidden.is_none() {
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
        if let Some(hidden) = output_hidden.as_deref_mut() {
            *hidden = self.device.download_f32(&scratch.hidden_temp)?;
        }
        if profile {
            info!(
                position,
                ms = stage_start.elapsed().as_secs_f64() * 1000.0,
                "cuda profile: final norm"
            );
        }

        if !compute_logits {
            output_logits.clear();
            return Ok(true);
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
            ResidentTokenEmbedding::MXFP4(matrix) => {
                self.device.embed_q8_0_resident_device(matrix, &[token_id])
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
            ResidentQuantMatrix::Q4K(matrix) if self.cpu_order_q4_k_matvec => self
                .device
                .matvec_q4_k_recurrent_resident_device(matrix, input),
            ResidentQuantMatrix::Q4K(matrix) => {
                self.device.matvec_q4_k_resident_device(matrix, input)
            }
            ResidentQuantMatrix::Q5K(matrix) => {
                self.device.matvec_q5_k_resident_device(matrix, input)
            }
            ResidentQuantMatrix::Q6K(matrix) => {
                self.device.matvec_q6_k_resident_device(matrix, input)
            }
            ResidentQuantMatrix::MXFP4(matrix) => {
                self.device.matvec_mxfp4_resident_device(matrix, input)
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
            ResidentQuantMatrix::Q4K(matrix) if self.cpu_order_q4_k_matvec => self
                .device
                .matvec_q4_k_recurrent_resident_device_into(matrix, input, output),
            ResidentQuantMatrix::Q4K(matrix) => self
                .device
                .matvec_q4_k_resident_device_into(matrix, input, output),
            ResidentQuantMatrix::Q5K(matrix) => self
                .device
                .matvec_q5_k_resident_device_into(matrix, input, output),
            ResidentQuantMatrix::Q6K(matrix) => self
                .device
                .matvec_q6_k_resident_device_into(matrix, input, output),
            ResidentQuantMatrix::MXFP4(matrix) => self
                .device
                .matvec_mxfp4_resident_device_into(matrix, input, output),
        }
    }

    fn matvec_recurrent_qkv_resident_device_into(
        &self,
        matrix: &ResidentQuantMatrix,
        input: &CudaF32Buffer,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        match matrix {
            ResidentQuantMatrix::Q4K(matrix) => self
                .device
                .matvec_q4_k_recurrent_resident_device_into(matrix, input, output),
            _ => self.matvec_quant_resident_device_into(matrix, input, output),
        }
    }

    fn moe_expert_graph_key(
        &self,
        descriptor: &MoeLayerDescriptor,
        slot: &ResidentMoeExpertSlot,
        gpu_slot: usize,
        cache_mode: KvCacheMode,
        scratch_generation: u64,
    ) -> CudaMoeExpertGraphKey {
        CudaMoeExpertGraphKey {
            model_identity: self.model_name.clone(),
            architecture: self.config.architecture.clone(),
            device_ordinal: self.device_ordinal,
            cache_mode,
            placement_generation: self.placement_generation.load(Ordering::Acquire),
            scratch_generation,
            layer_index: descriptor.layer_index(),
            logical_expert: slot.logical_expert(),
            gpu_slot,
            embedding_length: descriptor.hidden_size(),
            intermediate_size: descriptor.intermediate_size(),
            selected_per_token: descriptor.selected_per_token(),
            weight_kinds: [
                slot.gate.graph_kind(),
                slot.up.graph_kind(),
                slot.down.graph_kind(),
            ],
        }
    }

    fn standard_dense_graph_key(
        &self,
        output: &ResidentQ8_0ProbeWeights,
        layers: &[ResidentQ8_0LayerWeights],
        cache_mode: KvCacheMode,
        kv_capacity: usize,
        shared_kv_pages: bool,
        scratch_generation: u64,
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
            model_identity: self.model_name.clone(),
            architecture: config.architecture.clone(),
            device_ordinal: self.device_ordinal,
            weight_kinds,
            cache_mode,
            shared_kv_pages,
            kv_capacity,
            placement_generation: self.placement_generation.load(Ordering::Acquire),
            scratch_generation,
            recurrent_buffer_generation: None,
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

    fn qwen35_graph_key(
        &self,
        output: &ResidentQ8_0ProbeWeights,
        layers: &[ResidentQwen35LayerWeights],
        cache_mode: KvCacheMode,
        kv_capacity: usize,
        scratch_generation: u64,
        recurrent_buffer_generation: u8,
    ) -> CudaDecodeGraphKey {
        let config = &self.config;
        let mut weight_kinds = Vec::with_capacity(2 + layers.len() * 10);
        weight_kinds.push(output.token_embedding.graph_kind());
        weight_kinds.push(output.output.graph_kind());
        for layer in layers {
            match &layer.attention {
                ResidentQwen35AttentionWeights::DeltaNet {
                    attn_qkv,
                    attn_gate,
                    ssm_alpha,
                    ssm_beta,
                    ssm_out,
                    ..
                } => weight_kinds.extend([
                    "deltanet",
                    attn_qkv.graph_kind(),
                    attn_gate.graph_kind(),
                    ssm_alpha.graph_kind(),
                    ssm_beta.graph_kind(),
                    ssm_out.graph_kind(),
                ]),
                ResidentQwen35AttentionWeights::Full {
                    attn_qg,
                    attn_k,
                    attn_v,
                    attn_output,
                    ..
                } => weight_kinds.extend([
                    "full-attention",
                    attn_qg.graph_kind(),
                    attn_k.graph_kind(),
                    attn_v.graph_kind(),
                    attn_output.graph_kind(),
                    "qk-norm",
                ]),
            }
            weight_kinds.extend([
                layer.ffn_gate.graph_kind(),
                layer.ffn_up.graph_kind(),
                layer.ffn_down.graph_kind(),
            ]);
        }
        CudaDecodeGraphKey {
            model_identity: self.model_name.clone(),
            architecture: config.architecture.clone(),
            device_ordinal: self.device_ordinal,
            weight_kinds,
            cache_mode,
            shared_kv_pages: false,
            kv_capacity,
            placement_generation: self.placement_generation.load(Ordering::Acquire),
            scratch_generation,
            recurrent_buffer_generation: Some(recurrent_buffer_generation),
            layer_count: layers.len(),
            embedding_length: config.embedding_length,
            kv_width: config.kv_width(),
            feed_forward_length: config.feed_forward_length,
            vocab_size: output.vocab_size,
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
            ResidentTokenEmbedding::MXFP4(matrix) => self
                .device
                .embed_q8_0_with_decode_params_into(matrix, params, output),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_standard_dense_graph_layer(
        &self,
        weights: &ResidentQ8_0LayerWeights,
        input: &CudaF32Buffer,
        output: &mut CudaF32Buffer,
        params: &CudaDecodeParams,
        cache: &mut CudaLayerKvStore,
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

        match cache {
            CudaLayerKvStore::F32(cache) => {
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
            }
            CudaLayerKvStore::SharedF32(cache) => {
                cache.append_with_decode_params(key, v, params)?;
                cache.single_query_attention_with_decode_params_into(
                    query,
                    params,
                    config.attention_head_count,
                    config.attention_head_count_kv,
                    config.head_dim(),
                    1.0 / (config.head_dim() as f32).sqrt(),
                    attention,
                )?;
            }
            CudaLayerKvStore::SharedQ8(cache) => {
                cache.append_with_decode_params(key, v, params)?;
                cache.single_query_attention_with_decode_params_into(
                    query,
                    params,
                    config.attention_head_count,
                    config.attention_head_count_kv,
                    config.head_dim(),
                    1.0 / (config.head_dim() as f32).sqrt(),
                    attention,
                )?;
            }
            CudaLayerKvStore::SharedKeyQ4ValueQ8(cache) => {
                cache.append_with_decode_params(key, v, params)?;
                cache.single_query_attention_with_decode_params_into(
                    query,
                    params,
                    config.attention_head_count,
                    config.attention_head_count_kv,
                    config.head_dim(),
                    1.0 / (config.head_dim() as f32).sqrt(),
                    attention,
                )?;
            }
            CudaLayerKvStore::SharedAgentAdaptive(cache) => {
                cache.append_hot_with_decode_params(key, v, params)?;
                cache.single_query_attention_with_decode_params_into(
                    query,
                    params,
                    config.attention_head_count,
                    config.attention_head_count_kv,
                    config.head_dim(),
                    1.0 / (config.head_dim() as f32).sqrt(),
                    attention,
                )?;
            }
            other => {
                return Err(XrtError::Unsupported(format!(
                    "CUDA Graph standard decode requires contiguous/shared f32 or shared q8/kq4-vq8/adaptive KV, found {}",
                    other.mode().as_str()
                )));
            }
        }
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
        let shared = layer_caches
            .first()
            .is_some_and(CudaLayerKvStore::is_shared_f32);
        let shared_q8 = layer_caches
            .first()
            .is_some_and(CudaLayerKvStore::is_shared_q8);
        let shared_kq4_vq8 = layer_caches
            .first()
            .is_some_and(CudaLayerKvStore::is_shared_kq4_vq8);
        let shared_adaptive = layer_caches
            .first()
            .is_some_and(CudaLayerKvStore::is_shared_adaptive);
        for (layer, cache) in layer_caches.iter().enumerate() {
            let (cache_len, cache_capacity) = match (
                shared,
                shared_q8,
                shared_kq4_vq8,
                shared_adaptive,
                cache,
            ) {
                (false, false, false, false, CudaLayerKvStore::F32(cache)) => {
                    (cache.len(), cache.capacity())
                }
                (true, false, false, false, CudaLayerKvStore::SharedF32(cache)) => {
                    (cache.len(), cache.capacity())
                }
                (false, true, false, false, CudaLayerKvStore::SharedQ8(cache)) => {
                    (cache.len(), cache.capacity())
                }
                (false, false, true, false, CudaLayerKvStore::SharedKeyQ4ValueQ8(cache)) => {
                    (cache.len(), cache.capacity())
                }
                (false, false, false, true, CudaLayerKvStore::SharedAgentAdaptive(cache)) => {
                    (cache.len(), cache.capacity())
                }
                _ => {
                    return Err(XrtError::Unsupported(
                        "CUDA Graph standard decode requires homogeneous contiguous/shared f32 or shared q8/kq4-vq8/adaptive KV"
                            .to_string(),
                    ));
                }
            };
            if cache_len != position {
                return Err(XrtError::Runtime(format!(
                    "CUDA graph layer {layer} expected KV len {position}, found {cache_len}",
                )));
            }
            if cache_capacity != capacity {
                return Err(XrtError::Runtime(format!(
                    "CUDA graph layer {layer} expected KV capacity {capacity}, found {cache_capacity}",
                )));
            }
        }
        Ok(())
    }

    fn capture_shared_f32_graph_bindings(
        layer_caches: &[CudaLayerKvStore],
        first_append_position: usize,
    ) -> Result<Vec<CudaSharedF32GraphBinding>> {
        if layer_caches
            .iter()
            .all(|cache| matches!(cache, CudaLayerKvStore::F32(_)))
        {
            return Ok(Vec::new());
        }
        layer_caches
            .iter()
            .enumerate()
            .map(|(layer, cache)| match cache {
                CudaLayerKvStore::SharedF32(cache) => cache.graph_binding(first_append_position),
                _ => Err(XrtError::Unsupported(format!(
                    "CUDA shared F32 graph binding layer {layer} has an incompatible cache layout"
                ))),
            })
            .collect()
    }

    fn capture_shared_q8_graph_bindings(
        layer_caches: &[CudaLayerKvStore],
        first_append_position: usize,
    ) -> Result<Vec<CudaSharedQ8GraphBinding>> {
        if layer_caches
            .iter()
            .all(|cache| !matches!(cache, CudaLayerKvStore::SharedQ8(_)))
        {
            return Ok(Vec::new());
        }
        layer_caches
            .iter()
            .enumerate()
            .map(|(layer, cache)| match cache {
                CudaLayerKvStore::SharedQ8(cache) => cache.graph_binding(first_append_position),
                _ => Err(XrtError::Unsupported(format!(
                    "CUDA shared Q8 graph binding layer {layer} has an incompatible cache layout"
                ))),
            })
            .collect()
    }

    fn capture_shared_kq4_vq8_graph_bindings(
        layer_caches: &[CudaLayerKvStore],
        first_append_position: usize,
    ) -> Result<Vec<CudaSharedKq4Vq8GraphBinding>> {
        if layer_caches
            .iter()
            .all(|cache| !matches!(cache, CudaLayerKvStore::SharedKeyQ4ValueQ8(_)))
        {
            return Ok(Vec::new());
        }
        layer_caches
            .iter()
            .enumerate()
            .map(|(layer, cache)| match cache {
                CudaLayerKvStore::SharedKeyQ4ValueQ8(cache) => {
                    cache.graph_binding(first_append_position)
                }
                _ => Err(XrtError::Unsupported(format!(
                    "CUDA shared KQ4/VQ8 graph binding layer {layer} has an incompatible cache layout"
                ))),
            })
            .collect()
    }

    fn capture_shared_adaptive_graph_bindings(
        layer_caches: &[CudaLayerKvStore],
        first_append_position: usize,
    ) -> Result<Vec<CudaSharedAdaptiveGraphBinding>> {
        if layer_caches
            .iter()
            .all(|cache| !matches!(cache, CudaLayerKvStore::SharedAgentAdaptive(_)))
        {
            return Ok(Vec::new());
        }
        layer_caches
            .iter()
            .enumerate()
            .map(|(layer, cache)| match cache {
                CudaLayerKvStore::SharedAgentAdaptive(cache) => {
                    cache.graph_binding(first_append_position)
                }
                _ => Err(XrtError::Unsupported(format!(
                    "CUDA shared adaptive graph binding layer {layer} has an incompatible cache layout"
                ))),
            })
            .collect()
    }

    fn commit_standard_dense_graph_caches(
        &self,
        layer_caches: &mut [CudaLayerKvStore],
        position: usize,
    ) -> Result<()> {
        for cache in layer_caches {
            match cache {
                CudaLayerKvStore::F32(cache) => {
                    self.device.commit_layer_kv_graph_append(cache, position)?;
                }
                CudaLayerKvStore::SharedF32(cache) => {
                    cache.commit_graph_append(position)?;
                }
                CudaLayerKvStore::SharedQ8(cache) => {
                    cache.commit_graph_append(position)?;
                }
                CudaLayerKvStore::SharedKeyQ4ValueQ8(cache) => {
                    cache.commit_graph_append(position)?;
                }
                CudaLayerKvStore::SharedAgentAdaptive(cache) => {
                    cache.commit_graph_hot_append(position)?;
                }
                _ => {
                    return Err(XrtError::Unsupported(
                        "CUDA Graph standard decode requires contiguous/shared f32 or shared q8/kq4-vq8/adaptive KV"
                            .to_string(),
                    ));
                }
            }
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
            || batch
                .iter()
                .any(|item| item.session.cuda_graph_uses_shared_f32())
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
                None,
                None,
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
                    false,
                    epoch,
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
                if let CudaDecodeBatchGraphEntryState::Captured { graph, .. } = state {
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
                        match reserve_cuda_graph_allocation(self.allocation_arena.as_ref(), &graph)
                        {
                            Ok(allocation) => {
                                info!(
                                    batch_size = batch.len(),
                                    nodes = graph.node_count(),
                                    accounting_bytes = graph.accounting_bytes(),
                                    "composed parallel CUDA multi-sequence decode graph"
                                );
                                match graph.launch() {
                                    Ok(()) => {
                                        cache.insert(
                                            key.clone(),
                                            CudaDecodeBatchGraphEntryState::Captured {
                                                graph,
                                                _allocation: allocation,
                                            },
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
                                    "parallel CUDA decode graph admission failed; using stream replay: {err}"
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
        session.prepare_cuda_graph_append_position(position)?;
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
            None,
            None,
        )?;
        let cache_mode = session.cache_mode();
        let graph_allocation_arena = session.cuda_allocation_arena();
        let scratch_generation = session.cuda_batch_graph_epoch().ok_or_else(|| {
            XrtError::Runtime("CUDA Graph received a CPU backend session".to_string())
        })?;
        let (graph_state, layer_caches, scratch) = session.cuda_graph_parts_mut()?;
        if !graph_state.is_enabled() {
            return Ok(None);
        }
        let shared_kv_pages = layer_caches
            .first()
            .is_some_and(CudaLayerKvStore::uses_shared_pages);
        let key = self.standard_dense_graph_key(
            output_weights,
            layer_weights,
            cache_mode,
            kv_capacity,
            shared_kv_pages,
            scratch_generation,
        );
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

        if graph_state.has_executable_for(&key) {
            let _ = graph_state.executable_for(&key);
            if let Err(err) = graph_state.validate_shared_f32_bindings(layer_caches, position) {
                graph_state.fallback(err.to_string());
                tracing::warn!(
                    "CUDA shared F32 graph binding validation failed; using eager CUDA: {err}"
                );
                return Ok(None);
            }
            if let Err(err) = graph_state.validate_shared_q8_bindings(layer_caches, position) {
                graph_state.fallback(err.to_string());
                tracing::warn!(
                    "CUDA shared Q8 graph binding validation failed; using eager CUDA: {err}"
                );
                return Ok(None);
            }
            if let Err(err) = graph_state.validate_shared_kq4_vq8_bindings(layer_caches, position) {
                graph_state.fallback(err.to_string());
                tracing::warn!(
                    "CUDA shared KQ4/VQ8 graph binding validation failed; using eager CUDA: {err}"
                );
                return Ok(None);
            }
            if let Err(err) = graph_state.validate_shared_adaptive_bindings(layer_caches, position)
            {
                graph_state.fallback(err.to_string());
                tracing::warn!(
                    "CUDA shared adaptive graph binding validation failed; using eager CUDA: {err}"
                );
                return Ok(None);
            }
            let launch_result = graph_state
                .executable
                .as_ref()
                .expect("matching CUDA graph key must retain an executable")
                .launch();
            if let Err(err) = launch_result {
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
                let bindings = match Self::capture_shared_f32_graph_bindings(layer_caches, position)
                {
                    Ok(bindings) => bindings,
                    Err(err) => {
                        graph_state.fallback(err.to_string());
                        tracing::warn!(
                            "CUDA shared F32 graph binding failed; using eager CUDA: {err}"
                        );
                        return Ok(Some(logits));
                    }
                };
                let q8_bindings =
                    match Self::capture_shared_q8_graph_bindings(layer_caches, position) {
                        Ok(bindings) => bindings,
                        Err(err) => {
                            graph_state.fallback(err.to_string());
                            tracing::warn!(
                                "CUDA shared Q8 graph binding failed; using eager CUDA: {err}"
                            );
                            return Ok(Some(logits));
                        }
                    };
                let kq4_vq8_bindings =
                    match Self::capture_shared_kq4_vq8_graph_bindings(layer_caches, position) {
                        Ok(bindings) => bindings,
                        Err(err) => {
                            graph_state.fallback(err.to_string());
                            tracing::warn!(
                                "CUDA shared KQ4/VQ8 graph binding failed; using eager CUDA: {err}"
                            );
                            return Ok(Some(logits));
                        }
                    };
                let adaptive_bindings =
                    match Self::capture_shared_adaptive_graph_bindings(layer_caches, position) {
                        Ok(bindings) => bindings,
                        Err(err) => {
                            graph_state.fallback(err.to_string());
                            tracing::warn!(
                                "CUDA shared adaptive graph binding failed; using eager CUDA: {err}"
                            );
                            return Ok(Some(logits));
                        }
                    };
                match reserve_cuda_graph_allocation(graph_allocation_arena.as_ref(), &graph) {
                    Ok(allocation) => {
                        info!(
                            nodes = graph.node_count(),
                            accounting_bytes = graph.accounting_bytes(),
                            "captured CUDA batch-1 decode graph"
                        );
                        graph_state.captured(
                            key,
                            graph,
                            bindings,
                            q8_bindings,
                            kq4_vq8_bindings,
                            adaptive_bindings,
                            allocation,
                        );
                    }
                    Err(err) => {
                        graph_state.fallback(err.to_string());
                        tracing::warn!("CUDA Graph admission failed; using eager CUDA: {err}");
                    }
                }
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
    MXFP4(Arc<CudaQ8_0Matrix>),
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
            Self::MXFP4(_) => "mxfp4",
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
            Self::MXFP4(matrix) => Some(ResidentQuantMatrix::MXFP4(Arc::clone(matrix))),
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
    MXFP4(Arc<CudaQ8_0Matrix>),
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
            Self::MXFP4(_) => "mxfp4",
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
            DType::MXFP4 => device
                .upload_mxfp4_matrix(data, info.rows, info.cols)
                .map(Arc::new)
                .map(Self::MXFP4),
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
            DType::MXFP4 => ResidentTokenEmbedding::MXFP4(Arc::new(device.upload_mxfp4_matrix(
                token_embedding_data,
                token_embedding_info.rows,
                token_embedding_info.cols,
            )?)),
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

struct ResidentMoeExpertSlot {
    logical_expert: u16,
    gate: ResidentQuantMatrix,
    up: ResidentQuantMatrix,
    down: ResidentQuantMatrix,
}

struct ResidentMoePlacement {
    snapshot: Arc<ExpertPlacementSnapshot>,
    slots: Vec<Arc<ResidentMoeExpertSlot>>,
}

impl ResidentMoePlacement {
    fn slot(&self, gpu_slot: u16, layer_index: usize) -> Result<Arc<ResidentMoeExpertSlot>> {
        let slot = self.slots.get(usize::from(gpu_slot)).ok_or_else(|| {
            XrtError::Runtime(format!(
                "MoE layer {layer_index} placement references missing GPU slot {gpu_slot}"
            ))
        })?;
        let expected_logical = self.snapshot.logical_expert_for(gpu_slot).ok_or_else(|| {
            XrtError::Runtime(format!(
                "MoE layer {layer_index} placement has no logical expert for GPU slot {gpu_slot}"
            ))
        })?;
        if slot.logical_expert != expected_logical {
            return Err(XrtError::Runtime(format!(
                "MoE layer {layer_index} GPU slot {gpu_slot} contains logical expert {}, expected {expected_logical}",
                slot.logical_expert
            )));
        }
        Ok(Arc::clone(slot))
    }
}

struct CpuMoeLayerOutput {
    output_rows: [Option<usize>; MAX_SELECTED_EXPERTS],
    output_row_count: usize,
    shared_result: Option<usize>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaMoeTelemetrySnapshot {
    pub cpu_expert_calls: u64,
    pub gpu_expert_calls: u64,
    pub gpu_placement_hits: u64,
    pub gpu_placement_misses: u64,
    pub activation_d2h_bytes: u64,
    pub result_h2d_bytes: u64,
    pub coordinator_failures: u64,
    pub graph_eager_expert_calls: u64,
    pub graph_captures: u64,
    pub graph_replays: u64,
    pub graph_fallbacks: u64,
    pub placement_evaluations: u64,
    pub placement_updates: u64,
    pub placement_moves: u64,
    pub placement_upload_bytes: u64,
    pub placement_update_micros: u64,
    pub placement_last_update_micros: u64,
    pub layerwise_prefill_batches: u64,
    pub layerwise_prefill_tokens: u64,
    pub layerwise_prefill_weight_upload_bytes: u64,
    pub layerwise_prefill_repack_bytes: u64,
    pub layerwise_prefill_micros: u64,
}

#[derive(Debug, Default)]
struct CudaMoeTelemetry {
    cpu_expert_calls: AtomicU64,
    gpu_expert_calls: AtomicU64,
    gpu_placement_hits: AtomicU64,
    gpu_placement_misses: AtomicU64,
    activation_d2h_bytes: AtomicU64,
    result_h2d_bytes: AtomicU64,
    coordinator_failures: AtomicU64,
    graph_eager_expert_calls: AtomicU64,
    graph_captures: AtomicU64,
    graph_replays: AtomicU64,
    graph_fallbacks: AtomicU64,
    placement_evaluations: AtomicU64,
    placement_updates: AtomicU64,
    placement_moves: AtomicU64,
    placement_upload_bytes: AtomicU64,
    placement_update_micros: AtomicU64,
    placement_last_update_micros: AtomicU64,
    layerwise_prefill_batches: AtomicU64,
    layerwise_prefill_tokens: AtomicU64,
    layerwise_prefill_weight_upload_bytes: AtomicU64,
    layerwise_prefill_repack_bytes: AtomicU64,
    layerwise_prefill_micros: AtomicU64,
}

impl CudaMoeTelemetry {
    fn add(counter: &AtomicU64, value: u64) {
        let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            Some(current.saturating_add(value))
        });
    }

    fn record_plan(&self, cpu_calls: usize, gpu_calls: usize) {
        let cpu_calls = u64::try_from(cpu_calls).unwrap_or(u64::MAX);
        let gpu_calls = u64::try_from(gpu_calls).unwrap_or(u64::MAX);
        Self::add(&self.cpu_expert_calls, cpu_calls);
        Self::add(&self.gpu_expert_calls, gpu_calls);
        Self::add(&self.gpu_placement_misses, cpu_calls);
        Self::add(&self.gpu_placement_hits, gpu_calls);
    }

    fn record_layerwise_plan(&self, resident_calls: usize, staged_calls: usize) {
        let resident_calls = u64::try_from(resident_calls).unwrap_or(u64::MAX);
        let staged_calls = u64::try_from(staged_calls).unwrap_or(u64::MAX);
        Self::add(
            &self.gpu_expert_calls,
            resident_calls.saturating_add(staged_calls),
        );
        Self::add(&self.gpu_placement_hits, resident_calls);
        Self::add(&self.gpu_placement_misses, staged_calls);
    }

    fn record_activation_d2h(&self, bytes: usize) {
        Self::add(
            &self.activation_d2h_bytes,
            u64::try_from(bytes).unwrap_or(u64::MAX),
        );
    }

    fn record_result_h2d(&self, bytes: usize) {
        Self::add(
            &self.result_h2d_bytes,
            u64::try_from(bytes).unwrap_or(u64::MAX),
        );
    }

    fn record_coordinator_failure(&self) {
        Self::add(&self.coordinator_failures, 1);
    }

    fn record_graph_eager_call(&self) {
        Self::add(&self.graph_eager_expert_calls, 1);
    }

    fn record_graph_capture(&self) {
        Self::add(&self.graph_captures, 1);
    }

    fn record_graph_replay(&self) {
        Self::add(&self.graph_replays, 1);
    }

    fn record_graph_fallback(&self) {
        Self::add(&self.graph_fallbacks, 1);
    }

    fn record_placement_evaluation(&self, moves: usize, upload_bytes: u64, elapsed_micros: u64) {
        Self::add(&self.placement_evaluations, 1);
        if moves == 0 {
            return;
        }
        Self::add(&self.placement_updates, 1);
        Self::add(
            &self.placement_moves,
            u64::try_from(moves).unwrap_or(u64::MAX),
        );
        Self::add(&self.placement_upload_bytes, upload_bytes);
        Self::add(&self.placement_update_micros, elapsed_micros);
        self.placement_last_update_micros
            .store(elapsed_micros, Ordering::Relaxed);
    }

    fn record_layerwise_prefill(
        &self,
        tokens: usize,
        upload_bytes: u64,
        repack_bytes: u64,
        elapsed_micros: u64,
    ) {
        Self::add(&self.layerwise_prefill_batches, 1);
        Self::add(
            &self.layerwise_prefill_tokens,
            u64::try_from(tokens).unwrap_or(u64::MAX),
        );
        Self::add(&self.layerwise_prefill_weight_upload_bytes, upload_bytes);
        Self::add(&self.layerwise_prefill_repack_bytes, repack_bytes);
        Self::add(&self.layerwise_prefill_micros, elapsed_micros);
    }

    fn snapshot(&self) -> CudaMoeTelemetrySnapshot {
        CudaMoeTelemetrySnapshot {
            cpu_expert_calls: self.cpu_expert_calls.load(Ordering::Relaxed),
            gpu_expert_calls: self.gpu_expert_calls.load(Ordering::Relaxed),
            gpu_placement_hits: self.gpu_placement_hits.load(Ordering::Relaxed),
            gpu_placement_misses: self.gpu_placement_misses.load(Ordering::Relaxed),
            activation_d2h_bytes: self.activation_d2h_bytes.load(Ordering::Relaxed),
            result_h2d_bytes: self.result_h2d_bytes.load(Ordering::Relaxed),
            coordinator_failures: self.coordinator_failures.load(Ordering::Relaxed),
            graph_eager_expert_calls: self.graph_eager_expert_calls.load(Ordering::Relaxed),
            graph_captures: self.graph_captures.load(Ordering::Relaxed),
            graph_replays: self.graph_replays.load(Ordering::Relaxed),
            graph_fallbacks: self.graph_fallbacks.load(Ordering::Relaxed),
            placement_evaluations: self.placement_evaluations.load(Ordering::Relaxed),
            placement_updates: self.placement_updates.load(Ordering::Relaxed),
            placement_moves: self.placement_moves.load(Ordering::Relaxed),
            placement_upload_bytes: self.placement_upload_bytes.load(Ordering::Relaxed),
            placement_update_micros: self.placement_update_micros.load(Ordering::Relaxed),
            placement_last_update_micros: self.placement_last_update_micros.load(Ordering::Relaxed),
            layerwise_prefill_batches: self.layerwise_prefill_batches.load(Ordering::Relaxed),
            layerwise_prefill_tokens: self.layerwise_prefill_tokens.load(Ordering::Relaxed),
            layerwise_prefill_weight_upload_bytes: self
                .layerwise_prefill_weight_upload_bytes
                .load(Ordering::Relaxed),
            layerwise_prefill_repack_bytes: self
                .layerwise_prefill_repack_bytes
                .load(Ordering::Relaxed),
            layerwise_prefill_micros: self.layerwise_prefill_micros.load(Ordering::Relaxed),
        }
    }
}

impl ResidentMoeExpertSlot {
    fn logical_expert(&self) -> usize {
        usize::from(self.logical_expert)
    }

    fn upload(
        device: &CudaDevice,
        gguf: &GgufFile,
        source: &GgufResidentTensorSource<'_>,
        config: &LlamaConfig,
        layer: usize,
        logical_expert: usize,
    ) -> Result<Self> {
        let expert_count = config.expert_count.ok_or_else(|| {
            XrtError::InvalidMetadata("MoE CUDA plan is missing expert_count".to_string())
        })?;
        Ok(Self {
            logical_expert: u16::try_from(logical_expert).map_err(|_| {
                XrtError::Unsupported(format!(
                    "MoE logical expert {logical_expert} exceeds the u16 slot identity"
                ))
            })?,
            gate: upload_moe_expert_projection(
                device,
                gguf,
                source,
                layer,
                logical_expert,
                expert_count,
                "gate",
                config.feed_forward_length,
                config.embedding_length,
            )?,
            up: upload_moe_expert_projection(
                device,
                gguf,
                source,
                layer,
                logical_expert,
                expert_count,
                "up",
                config.feed_forward_length,
                config.embedding_length,
            )?,
            down: upload_moe_expert_projection(
                device,
                gguf,
                source,
                layer,
                logical_expert,
                expert_count,
                "down",
                config.embedding_length,
                config.feed_forward_length,
            )?,
        })
    }
}

struct MoeAdaptiveRuntime {
    gguf: Arc<GgufFile>,
    staging_device: CudaDevice,
    tracker: Mutex<AdaptivePlacementTracker>,
    expert_costs: Vec<Vec<u64>>,
}

struct AdaptiveUploadedSlot {
    movement: AdaptivePlacementMove,
    slot: Option<ResidentMoeExpertSlot>,
}

struct MoeLayerwisePrefillRuntime {
    gguf: Arc<GgufFile>,
    staging_devices: [CudaDevice; 2],
    expert_costs: Vec<Vec<u64>>,
    worst_case_staging_bytes: u64,
}

impl MoeLayerwisePrefillRuntime {
    fn new(
        gguf: Arc<GgufFile>,
        config: &LlamaConfig,
        expert_costs: Vec<Vec<u64>>,
        device_ordinal: usize,
    ) -> Result<Self> {
        let expert_count = config.expert_count.ok_or_else(|| {
            XrtError::InvalidMetadata("layerwise MoE plan is missing expert_count".to_string())
        })?;
        if expert_costs.len() != config.block_count
            || expert_costs.iter().any(|layer| layer.len() != expert_count)
        {
            return Err(XrtError::Runtime(
                "layerwise MoE expert-cost geometry is inconsistent".to_string(),
            ));
        }
        let complete_layer_bytes = expert_costs
            .iter()
            .map(|layer| {
                layer.iter().try_fold(0u64, |total, &bytes| {
                    total.checked_add(bytes).ok_or_else(|| {
                        XrtError::Runtime(
                            "layerwise MoE complete-layer byte count overflowed".to_string(),
                        )
                    })
                })
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max()
            .unwrap_or(0);
        let largest_expert_bytes = expert_costs.iter().flatten().copied().max().unwrap_or(0);
        let worst_case_staging_bytes = largest_expert_bytes
            .checked_mul(2)
            .and_then(|double_buffer| complete_layer_bytes.checked_add(double_buffer))
            .ok_or_else(|| {
                XrtError::Runtime("layerwise MoE staging admission bytes overflowed".to_string())
            })?;
        Ok(Self {
            gguf,
            staging_devices: [
                CudaDevice::new(device_ordinal)?,
                CudaDevice::new(device_ordinal)?,
            ],
            expert_costs,
            worst_case_staging_bytes,
        })
    }

    fn expert_bytes(&self, layer: usize, logical_expert: usize) -> Result<u64> {
        self.expert_costs
            .get(layer)
            .and_then(|costs| costs.get(logical_expert))
            .copied()
            .ok_or_else(|| {
                XrtError::Runtime(format!(
                    "layerwise MoE byte cost is missing layer {layer} expert {logical_expert}"
                ))
            })
    }
}

impl MoeAdaptiveRuntime {
    fn new(
        gguf: Arc<GgufFile>,
        config: &LlamaConfig,
        runtime: &MoeRuntimeConfig,
        expert_costs: Vec<Vec<u64>>,
        device_ordinal: usize,
    ) -> Result<Self> {
        let expert_count = config.expert_count.ok_or_else(|| {
            XrtError::InvalidMetadata("adaptive MoE plan is missing expert_count".to_string())
        })?;
        if expert_costs.len() != config.block_count
            || expert_costs.iter().any(|layer| layer.len() != expert_count)
        {
            return Err(XrtError::Runtime(
                "adaptive MoE expert-cost geometry is inconsistent".to_string(),
            ));
        }
        for (layer_index, layer) in expert_costs.iter().enumerate() {
            if layer.windows(2).any(|pair| pair[0] != pair[1]) {
                return Err(XrtError::Unsupported(format!(
                    "adaptive MoE layer {layer_index} has variable-size expert slots; use profiled or uniform placement"
                )));
            }
        }
        Ok(Self {
            gguf,
            staging_device: CudaDevice::new(device_ordinal)?,
            tracker: Mutex::new(AdaptivePlacementTracker::new(
                config.block_count,
                expert_count,
                runtime.placement_update_tokens,
                ADAPTIVE_MOE_MAX_MOVES_PER_UPDATE,
                ADAPTIVE_MOE_MIN_RESIDENCY_EPOCHS,
                ADAPTIVE_MOE_HYSTERESIS_PERCENT,
            )?),
            expert_costs,
        })
    }

    fn incoming_bytes(&self, decision: &AdaptivePlacementDecision) -> Result<u64> {
        decision.moves().iter().try_fold(0u64, |total, movement| {
            let bytes = self
                .expert_costs
                .get(movement.layer_index())
                .and_then(|layer| layer.get(movement.incoming_expert()))
                .copied()
                .ok_or_else(|| {
                    XrtError::Runtime(
                        "adaptive MoE move has no corresponding expert byte cost".to_string(),
                    )
                })?;
            total.checked_add(bytes).ok_or_else(|| {
                XrtError::Runtime("adaptive MoE upload byte count overflowed".to_string())
            })
        })
    }
}

trait ResidentMoeFfnLayer {
    fn moe_descriptor(&self) -> &MoeLayerDescriptor;
    fn moe_resident(&self) -> &RwLock<ResidentMoePlacement>;
    fn moe_router(&self) -> &ResidentQuantMatrix;
    fn moe_embedding_length(&self) -> usize;
}

struct ResidentMoeLayerWeights {
    descriptor: MoeLayerDescriptor,
    resident: RwLock<ResidentMoePlacement>,
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
    router: ResidentQuantMatrix,
    embedding_length: usize,
}

impl ResidentMoeFfnLayer for ResidentMoeLayerWeights {
    fn moe_descriptor(&self) -> &MoeLayerDescriptor {
        &self.descriptor
    }

    fn moe_resident(&self) -> &RwLock<ResidentMoePlacement> {
        &self.resident
    }

    fn moe_router(&self) -> &ResidentQuantMatrix {
        &self.router
    }

    fn moe_embedding_length(&self) -> usize {
        self.embedding_length
    }
}

impl ResidentMoeLayerWeights {
    fn validate_source(
        source: &impl ResidentTensorSource,
        config: &LlamaConfig,
        placements: &[Arc<ExpertPlacementSnapshot>],
    ) -> Result<()> {
        let expert_count = config.expert_count.ok_or_else(|| {
            XrtError::InvalidMetadata("MoE CUDA plan is missing expert_count".to_string())
        })?;
        let selected_per_token = config.expert_used_count.ok_or_else(|| {
            XrtError::InvalidMetadata("MoE CUDA plan is missing expert_used_count".to_string())
        })?;
        if placements.len() != config.block_count {
            return Err(XrtError::Runtime(format!(
                "MoE CUDA plan has {} placements for {} layers",
                placements.len(),
                config.block_count
            )));
        }
        let dim = config.embedding_length;
        let q_width = config.q_width();
        let kv_width = config.kv_width();
        for (layer, placement) in placements.iter().enumerate() {
            let descriptor = MoeLayerDescriptor::new(
                layer,
                expert_count,
                selected_per_token,
                dim,
                config.feed_forward_length,
            )?;
            if placement.layer_index() != layer
                || placement.expert_count() != descriptor.expert_count()
            {
                return Err(XrtError::Runtime(format!(
                    "MoE CUDA placement geometry does not match layer {layer}"
                )));
            }
            if !matches_optional_qk_norm_pair(source, layer, config.head_dim()) {
                return Err(XrtError::InvalidTensor(format!(
                    "MoE CUDA layer {layer} has an incomplete or malformed Q/K norm pair"
                )));
            }
            for (name, len) in [
                (format!("blk.{layer}.attn_norm.weight"), dim),
                (format!("blk.{layer}.ffn_norm.weight"), dim),
            ] {
                if !matches_f32_vector(source, &name, len) {
                    return Err(XrtError::InvalidTensor(format!(
                        "MoE CUDA tensor `{name}` must be a supported float vector of length {len}"
                    )));
                }
            }
            for (name, len) in [
                (format!("blk.{layer}.attn_q.bias"), q_width),
                (format!("blk.{layer}.attn_k.bias"), kv_width),
                (format!("blk.{layer}.attn_v.bias"), kv_width),
            ] {
                if !matches_optional_f32_vector(source, &name, len) {
                    return Err(XrtError::InvalidTensor(format!(
                        "MoE CUDA optional tensor `{name}` has invalid geometry"
                    )));
                }
            }
            for (name, rows, cols) in [
                (format!("blk.{layer}.attn_q.weight"), q_width, dim),
                (format!("blk.{layer}.attn_k.weight"), kv_width, dim),
                (format!("blk.{layer}.attn_v.weight"), kv_width, dim),
                (format!("blk.{layer}.attn_output.weight"), dim, q_width),
                (
                    format!("blk.{layer}.ffn_gate_inp.weight"),
                    expert_count,
                    dim,
                ),
            ] {
                if !matches_supported_linear_shape(source, &name, rows, cols) {
                    return Err(XrtError::InvalidTensor(format!(
                        "MoE CUDA tensor `{name}` must be a supported {rows}x{cols} matrix"
                    )));
                }
            }
        }
        Ok(())
    }

    fn try_load_all(
        device: &CudaDevice,
        gguf: &GgufFile,
        source: &GgufResidentTensorSource<'_>,
        config: &LlamaConfig,
        placements: &[Arc<ExpertPlacementSnapshot>],
    ) -> Result<Vec<Self>> {
        Self::validate_source(source, config, placements)?;
        let expert_count = config
            .expert_count
            .expect("MoE expert count was validated above");
        let selected_per_token = config
            .expert_used_count
            .expect("MoE selected expert count was validated above");
        let mut layers = Vec::with_capacity(config.block_count);
        for (layer, placement) in placements.iter().enumerate() {
            let mut slots = Vec::with_capacity(placement.gpu_slot_count());
            for &logical_expert in placement.gpu_slots_to_logical() {
                let logical_expert_usize = usize::from(logical_expert);
                slots.push(Arc::new(ResidentMoeExpertSlot::upload(
                    device,
                    gguf,
                    source,
                    config,
                    layer,
                    logical_expert_usize,
                )?));
            }
            layers.push(Self {
                descriptor: MoeLayerDescriptor::new(
                    layer,
                    expert_count,
                    selected_per_token,
                    config.embedding_length,
                    config.feed_forward_length,
                )?,
                resident: RwLock::new(ResidentMoePlacement {
                    snapshot: Arc::clone(placement),
                    slots,
                }),
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
                router: ResidentQuantMatrix::upload(
                    device,
                    source,
                    &format!("blk.{layer}.ffn_gate_inp.weight"),
                )?,
                embedding_length: config.embedding_length,
            });
        }
        Ok(layers)
    }
}

fn upload_moe_expert_projection(
    device: &CudaDevice,
    gguf: &GgufFile,
    source: &GgufResidentTensorSource<'_>,
    layer: usize,
    logical_expert: usize,
    expert_count: usize,
    projection: &str,
    rows: usize,
    cols: usize,
) -> Result<ResidentQuantMatrix> {
    let packed_name = format!("blk.{layer}.ffn_{projection}_exps.weight");
    if source.tensor_info(&packed_name).is_some() {
        let logical_source = GgufPackedExpertTensorSource::new(
            gguf,
            &packed_name,
            logical_expert,
            expert_count,
            rows,
            cols,
        )?;
        ResidentQuantMatrix::upload(device, &logical_source, &packed_name)
    } else {
        ResidentQuantMatrix::upload(
            device,
            source,
            &format!("blk.{layer}.ffn_{projection}.{logical_expert}.weight"),
        )
    }
}

fn moe_expert_projection_info(
    gguf: &GgufFile,
    source: &GgufResidentTensorSource<'_>,
    layer: usize,
    logical_expert: usize,
    expert_count: usize,
    projection: &str,
    rows: usize,
    cols: usize,
) -> Result<ResidentTensorInfo> {
    let packed_name = format!("blk.{layer}.ffn_{projection}_exps.weight");
    if source.tensor_info(&packed_name).is_some() {
        let logical_source = GgufPackedExpertTensorSource::new(
            gguf,
            &packed_name,
            logical_expert,
            expert_count,
            rows,
            cols,
        )?;
        logical_source.require_tensor(&packed_name)
    } else {
        source.require_tensor(&format!(
            "blk.{layer}.ffn_{projection}.{logical_expert}.weight"
        ))
    }
}

struct MoeResidentUploadPlan {
    placements: Vec<Arc<ExpertPlacementSnapshot>>,
    expert_costs: Vec<Vec<u64>>,
    non_expert_bytes: u64,
    expert_bytes: u64,
    expert_slots: usize,
    manifest_sha256: Option<String>,
}

impl MoeResidentUploadPlan {
    fn build(
        gguf: &GgufFile,
        source: &GgufResidentTensorSource<'_>,
        config: &LlamaConfig,
        runtime: &MoeRuntimeConfig,
    ) -> Result<Self> {
        let expert_count = config.expert_count.ok_or_else(|| {
            XrtError::InvalidMetadata("MoE CUDA plan is missing expert_count".to_string())
        })?;
        let budget = runtime.gpu_expert_budget_bytes.ok_or_else(|| {
            XrtError::Runtime("explicit CUDA MoE requires a GPU expert budget".to_string())
        })?;
        let source_output_name = ResidentQ8_0ProbeWeights::output_name(source);
        let mut non_expert_names = BTreeSet::new();
        non_expert_names.insert("token_embd.weight".to_string());
        non_expert_names.insert("output_norm.weight".to_string());
        non_expert_names.insert(source_output_name.to_string());
        if config.is_hybrid() {
            let recurrent = config.deltanet_state_descriptor().ok_or_else(|| {
                XrtError::InvalidMetadata(
                    "Qwen3.5 hybrid-MoE upload planning requires a DeltaNet descriptor".to_string(),
                )
            })?;
            for (layer, recurrent_layer) in recurrent.layers().iter().enumerate() {
                for name in [
                    format!("blk.{layer}.attn_norm.weight"),
                    format!("blk.{layer}.post_attention_norm.weight"),
                    format!("blk.{layer}.ffn_gate_inp.weight"),
                ] {
                    non_expert_names.insert(name);
                }
                if recurrent_layer.is_some() {
                    for name in [
                        format!("blk.{layer}.attn_qkv.weight"),
                        format!("blk.{layer}.attn_gate.weight"),
                        format!("blk.{layer}.ssm_alpha.weight"),
                        format!("blk.{layer}.ssm_beta.weight"),
                        format!("blk.{layer}.ssm_a"),
                        format!("blk.{layer}.ssm_dt.bias"),
                        format!("blk.{layer}.ssm_norm.weight"),
                        format!("blk.{layer}.ssm_out.weight"),
                        format!("blk.{layer}.ssm_conv1d.weight"),
                    ] {
                        non_expert_names.insert(name);
                    }
                } else {
                    for name in [
                        format!("blk.{layer}.attn_q.weight"),
                        format!("blk.{layer}.attn_k.weight"),
                        format!("blk.{layer}.attn_v.weight"),
                        format!("blk.{layer}.attn_output.weight"),
                        format!("blk.{layer}.attn_q_norm.weight"),
                        format!("blk.{layer}.attn_k_norm.weight"),
                    ] {
                        non_expert_names.insert(name);
                    }
                }
            }
        } else {
            for layer in 0..config.block_count {
                for name in [
                    format!("blk.{layer}.attn_norm.weight"),
                    format!("blk.{layer}.ffn_norm.weight"),
                    format!("blk.{layer}.attn_q.weight"),
                    format!("blk.{layer}.attn_k.weight"),
                    format!("blk.{layer}.attn_v.weight"),
                    format!("blk.{layer}.attn_output.weight"),
                    format!("blk.{layer}.ffn_gate_inp.weight"),
                ] {
                    non_expert_names.insert(name);
                }
                for name in [
                    format!("blk.{layer}.attn_q_norm.weight"),
                    format!("blk.{layer}.attn_k_norm.weight"),
                    format!("blk.{layer}.attn_q.bias"),
                    format!("blk.{layer}.attn_k.bias"),
                    format!("blk.{layer}.attn_v.bias"),
                ] {
                    if source.tensor_info(&name).is_some() {
                        non_expert_names.insert(name);
                    }
                }
            }
        }
        let non_expert_bytes = non_expert_names.iter().try_fold(0u64, |total, name| {
            let info = source.require_tensor(name)?;
            let bytes = if name == "token_embd.weight" {
                cuda_extra_resident_tensor_bytes(&info, source_output_name)?
            } else {
                cuda_matrix_resident_tensor_bytes(&info)?
            };
            total.checked_add(bytes).ok_or_else(|| {
                XrtError::Runtime("CUDA MoE non-expert resident byte count overflowed".to_string())
            })
        })?;

        let mut expert_costs = Vec::with_capacity(config.block_count);
        let mut expert_quantization_kinds = BTreeSet::new();
        for layer in 0..config.block_count {
            let mut layer_costs = Vec::with_capacity(expert_count);
            for logical_expert in 0..expert_count {
                let mut bytes = 0u64;
                for (projection, rows, cols) in [
                    ("gate", config.feed_forward_length, config.embedding_length),
                    ("up", config.feed_forward_length, config.embedding_length),
                    ("down", config.embedding_length, config.feed_forward_length),
                ] {
                    let info = moe_expert_projection_info(
                        gguf,
                        source,
                        layer,
                        logical_expert,
                        expert_count,
                        projection,
                        rows,
                        cols,
                    )?;
                    expert_quantization_kinds
                        .insert(format!("{:?}", info.dtype).to_ascii_lowercase());
                    bytes = bytes
                        .checked_add(cuda_matrix_resident_tensor_bytes(&info)?)
                        .ok_or_else(|| {
                            XrtError::Runtime(
                                "CUDA MoE expert resident byte count overflowed".to_string(),
                            )
                        })?;
                }
                layer_costs.push(bytes);
            }
            expert_costs.push(layer_costs);
        }

        let max_slots_per_layer = match runtime.acceleration {
            MoeAcceleration::Hybrid => expert_count.saturating_sub(1),
            MoeAcceleration::Gpu => expert_count,
            other => {
                return Err(XrtError::Runtime(format!(
                    "CUDA MoE upload plan cannot be built for acceleration mode {}",
                    other.as_str()
                )))
            }
        };
        if max_slots_per_layer == 0 {
            return Err(XrtError::Unsupported(
                "hybrid MoE placement requires at least two logical experts".to_string(),
            ));
        }

        let (placements, selected_bytes, manifest_sha256) = match runtime.placement {
            MoePlacementPolicy::Uniform | MoePlacementPolicy::Adaptive => {
                let mut selected_placements = None;
                let mut selected_bytes = 0u64;
                for slot_count in 1..=max_slots_per_layer {
                    let placements = (0..config.block_count)
                        .map(|layer| {
                            ExpertPlacementSnapshot::uniform(layer, expert_count, slot_count, 1)
                                .map(Arc::new)
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let bytes = placements.iter().try_fold(0u64, |total, placement| {
                        placement.gpu_slots_to_logical().iter().try_fold(
                            total,
                            |subtotal, &logical_expert| {
                                subtotal
                                    .checked_add(
                                        expert_costs[placement.layer_index()]
                                            [usize::from(logical_expert)],
                                    )
                                    .ok_or_else(|| {
                                        XrtError::Runtime(
                                            "CUDA MoE placement byte count overflowed".to_string(),
                                        )
                                    })
                            },
                        )
                    })?;
                    if bytes > budget {
                        continue;
                    }
                    selected_bytes = bytes;
                    selected_placements = Some(placements);
                }
                let placements = selected_placements.ok_or_else(|| {
                    XrtError::Cuda(format!(
                        "GPU expert budget {budget} bytes cannot place one expert in each of {} MoE layers",
                        config.block_count
                    ))
                })?;
                (placements, selected_bytes, None)
            }
            MoePlacementPolicy::Profiled => {
                let manifest_path = runtime.placement_manifest.as_deref().ok_or_else(|| {
                    XrtError::Runtime("profiled MoE placement requires a manifest path".to_string())
                })?;
                let model_sha256 = sha256_file(gguf.path())?;
                let config_sha256 = moe_config_sha256(config);
                let quantization = expert_quantization_kinds
                    .iter()
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("+");
                let validated = load_moe_placement_manifest(
                    manifest_path,
                    &MoePlacementManifestContext {
                        model_sha256: &model_sha256,
                        config_sha256: &config_sha256,
                        architecture: &config.architecture,
                        quantization: &quantization,
                        layer_count: config.block_count,
                        expert_count,
                        gpu_expert_budget_bytes: budget,
                        acceleration: runtime.acceleration,
                        expert_costs: &expert_costs,
                    },
                )?;
                info!(
                    manifest_sha256 = validated.manifest_sha256,
                    expert_slots = validated.expert_slots,
                    expert_bytes = validated.expert_bytes,
                    "validated profiled MoE placement manifest"
                );
                (
                    validated.placements,
                    validated.expert_bytes,
                    Some(validated.manifest_sha256),
                )
            }
        };
        if runtime.acceleration == MoeAcceleration::Gpu
            && placements
                .iter()
                .any(|placement| placement.gpu_slot_count() != expert_count)
        {
            let full_bytes = expert_costs
                .iter()
                .flatten()
                .try_fold(0u64, |total, &bytes| {
                    total.checked_add(bytes).ok_or_else(|| {
                        XrtError::Runtime("CUDA full-expert byte count overflowed".to_string())
                    })
                })?;
            return Err(XrtError::Cuda(format!(
                "GPU MoE mode requires every expert resident ({full_bytes} bytes), exceeding the configured {budget}-byte expert budget"
            )));
        }
        let expert_slots = placements
            .iter()
            .map(|placement| placement.gpu_slot_count())
            .sum();
        Ok(Self {
            placements,
            expert_costs,
            non_expert_bytes,
            expert_bytes: selected_bytes,
            expert_slots,
            manifest_sha256,
        })
    }
}

enum ResidentQwen35AttentionWeights {
    DeltaNet {
        attn_qkv: ResidentQuantMatrix,
        attn_gate: ResidentQuantMatrix,
        ssm_alpha: ResidentQuantMatrix,
        ssm_beta: ResidentQuantMatrix,
        ssm_a: GpuF32Tensor,
        ssm_dt_bias: GpuF32Tensor,
        ssm_norm: GpuF32Tensor,
        ssm_out: ResidentQuantMatrix,
        conv1d: GpuF32Tensor,
    },
    Full {
        attn_qg: ResidentQuantMatrix,
        attn_k: ResidentQuantMatrix,
        attn_v: ResidentQuantMatrix,
        attn_output: ResidentQuantMatrix,
        attn_q_norm: GpuF32Tensor,
        attn_k_norm: GpuF32Tensor,
    },
}

struct ResidentQwen35LayerWeights {
    attn_norm: GpuF32Tensor,
    attention: ResidentQwen35AttentionWeights,
    ffn_norm: GpuF32Tensor,
    ffn_gate: ResidentQuantMatrix,
    ffn_up: ResidentQuantMatrix,
    ffn_down: ResidentQuantMatrix,
    embedding_length: usize,
}

impl ResidentQwen35LayerWeights {
    fn supports_all(source: &impl ResidentTensorSource, config: &LlamaConfig) -> bool {
        if config.block_count == 0 || !config.is_hybrid() || config.is_gemma4() || config.is_moe() {
            return false;
        }
        let Some(descriptor) = config.deltanet_state_descriptor() else {
            return false;
        };
        if descriptor.layers().len() != config.block_count {
            return false;
        }
        let dim = config.embedding_length;
        let q_width = config.q_width();
        let qg_width = match q_width.checked_mul(2) {
            Some(value) => value,
            None => return false,
        };
        let kv_width = config.kv_width();
        let head_dim = config.head_dim();
        let ff_dim = config.feed_forward_length;
        let conv_channels = match descriptor
            .state_size()
            .checked_mul(descriptor.group_count())
            .and_then(|value| value.checked_mul(2))
            .and_then(|value| value.checked_add(descriptor.inner_size()))
        {
            Some(value) => value,
            None => return false,
        };
        let conv_elements = match conv_channels.checked_mul(descriptor.conv_kernel()) {
            Some(value) => value,
            None => return false,
        };
        let head_value_size = descriptor.inner_size() / descriptor.dt_rank();

        for (layer, recurrent) in descriptor.layers().iter().enumerate() {
            if !matches_f32_vector(source, &format!("blk.{layer}.attn_norm.weight"), dim)
                || !matches_f32_vector(
                    source,
                    &format!("blk.{layer}.post_attention_norm.weight"),
                    dim,
                )
            {
                return false;
            }
            for (name, rows, cols) in [
                (format!("blk.{layer}.ffn_gate.weight"), ff_dim, dim),
                (format!("blk.{layer}.ffn_up.weight"), ff_dim, dim),
                (format!("blk.{layer}.ffn_down.weight"), dim, ff_dim),
            ] {
                if !matches_supported_linear_shape(source, &name, rows, cols) {
                    return false;
                }
            }

            if recurrent.is_some() {
                for (name, rows, cols) in [
                    (format!("blk.{layer}.attn_qkv.weight"), conv_channels, dim),
                    (
                        format!("blk.{layer}.attn_gate.weight"),
                        descriptor.inner_size(),
                        dim,
                    ),
                    (
                        format!("blk.{layer}.ssm_alpha.weight"),
                        descriptor.dt_rank(),
                        dim,
                    ),
                    (
                        format!("blk.{layer}.ssm_beta.weight"),
                        descriptor.dt_rank(),
                        dim,
                    ),
                    (
                        format!("blk.{layer}.ssm_out.weight"),
                        dim,
                        descriptor.inner_size(),
                    ),
                ] {
                    if !matches_supported_linear_shape(source, &name, rows, cols) {
                        return false;
                    }
                }
                for (name, len) in [
                    (format!("blk.{layer}.ssm_a"), descriptor.dt_rank()),
                    (format!("blk.{layer}.ssm_dt.bias"), descriptor.dt_rank()),
                    (format!("blk.{layer}.ssm_norm.weight"), head_value_size),
                    (format!("blk.{layer}.ssm_conv1d.weight"), conv_elements),
                ] {
                    if !matches_f32_vector(source, &name, len) {
                        return false;
                    }
                }
            } else {
                for (name, rows, cols) in [
                    (format!("blk.{layer}.attn_q.weight"), qg_width, dim),
                    (format!("blk.{layer}.attn_k.weight"), kv_width, dim),
                    (format!("blk.{layer}.attn_v.weight"), kv_width, dim),
                    (format!("blk.{layer}.attn_output.weight"), dim, q_width),
                ] {
                    if !matches_supported_linear_shape(source, &name, rows, cols) {
                        return false;
                    }
                }
                if !matches_f32_vector(source, &format!("blk.{layer}.attn_q_norm.weight"), head_dim)
                    || !matches_f32_vector(
                        source,
                        &format!("blk.{layer}.attn_k_norm.weight"),
                        head_dim,
                    )
                {
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
        let descriptor = config
            .deltanet_state_descriptor()
            .expect("Qwen3.5 resident support validated the descriptor");
        let mut layers = Vec::with_capacity(config.block_count);
        for (layer, recurrent) in descriptor.layers().iter().enumerate() {
            let attention = if recurrent.is_some() {
                ResidentQwen35AttentionWeights::DeltaNet {
                    attn_qkv: ResidentQuantMatrix::upload(
                        device,
                        source,
                        &format!("blk.{layer}.attn_qkv.weight"),
                    )?,
                    attn_gate: ResidentQuantMatrix::upload(
                        device,
                        source,
                        &format!("blk.{layer}.attn_gate.weight"),
                    )?,
                    ssm_alpha: ResidentQuantMatrix::upload(
                        device,
                        source,
                        &format!("blk.{layer}.ssm_alpha.weight"),
                    )?,
                    ssm_beta: ResidentQuantMatrix::upload(
                        device,
                        source,
                        &format!("blk.{layer}.ssm_beta.weight"),
                    )?,
                    ssm_a: upload_resident_f32_tensor(
                        device,
                        source,
                        &format!("blk.{layer}.ssm_a"),
                    )?,
                    ssm_dt_bias: upload_resident_f32_tensor(
                        device,
                        source,
                        &format!("blk.{layer}.ssm_dt.bias"),
                    )?,
                    ssm_norm: upload_resident_f32_tensor(
                        device,
                        source,
                        &format!("blk.{layer}.ssm_norm.weight"),
                    )?,
                    ssm_out: ResidentQuantMatrix::upload(
                        device,
                        source,
                        &format!("blk.{layer}.ssm_out.weight"),
                    )?,
                    conv1d: upload_resident_f32_tensor(
                        device,
                        source,
                        &format!("blk.{layer}.ssm_conv1d.weight"),
                    )?,
                }
            } else {
                ResidentQwen35AttentionWeights::Full {
                    attn_qg: ResidentQuantMatrix::upload(
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
                }
            };
            layers.push(Self {
                attn_norm: upload_resident_f32_tensor(
                    device,
                    source,
                    &format!("blk.{layer}.attn_norm.weight"),
                )?,
                attention,
                ffn_norm: upload_resident_f32_tensor(
                    device,
                    source,
                    &format!("blk.{layer}.post_attention_norm.weight"),
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
                embedding_length: config.embedding_length,
            });
        }
        Ok(Some(layers))
    }
}

struct ResidentQwen35MtpWeights {
    layer: ResidentQwen35LayerWeights,
    enorm: GpuF32Tensor,
    hnorm: GpuF32Tensor,
    eh_proj: ResidentQuantMatrix,
    shared_head_norm: GpuF32Tensor,
}

impl ResidentQwen35MtpWeights {
    fn supports(source: &impl ResidentTensorSource, config: &LlamaConfig) -> bool {
        if !config.is_qwen35_family()
            || !config.is_hybrid()
            || config.is_moe()
            || config.nextn_predict_layers != 1
        {
            return false;
        }
        let layer = config.block_count;
        let dim = config.embedding_length;
        let q_width = config.q_width();
        let Some(qg_width) = q_width.checked_mul(2) else {
            return false;
        };
        let Some(mtp_input_width) = dim.checked_mul(2) else {
            return false;
        };
        let kv_width = config.kv_width();
        let head_dim = config.head_dim();
        let ff_dim = config.feed_forward_length;
        for (name, len) in [
            (format!("blk.{layer}.attn_norm.weight"), dim),
            (format!("blk.{layer}.post_attention_norm.weight"), dim),
            (format!("blk.{layer}.attn_q_norm.weight"), head_dim),
            (format!("blk.{layer}.attn_k_norm.weight"), head_dim),
            (format!("blk.{layer}.nextn.enorm.weight"), dim),
            (format!("blk.{layer}.nextn.hnorm.weight"), dim),
            (format!("blk.{layer}.nextn.shared_head_norm.weight"), dim),
        ] {
            if !matches_f32_vector(source, &name, len) {
                return false;
            }
        }
        for (name, rows, cols) in [
            (format!("blk.{layer}.attn_q.weight"), qg_width, dim),
            (format!("blk.{layer}.attn_k.weight"), kv_width, dim),
            (format!("blk.{layer}.attn_v.weight"), kv_width, dim),
            (format!("blk.{layer}.attn_output.weight"), dim, q_width),
            (format!("blk.{layer}.ffn_gate.weight"), ff_dim, dim),
            (format!("blk.{layer}.ffn_up.weight"), ff_dim, dim),
            (format!("blk.{layer}.ffn_down.weight"), dim, ff_dim),
            (
                format!("blk.{layer}.nextn.eh_proj.weight"),
                dim,
                mtp_input_width,
            ),
        ] {
            if !matches_supported_linear_shape(source, &name, rows, cols) {
                return false;
            }
        }
        true
    }

    fn try_load(
        device: &CudaDevice,
        source: &impl ResidentTensorSource,
        config: &LlamaConfig,
    ) -> Result<Option<Self>> {
        if !config.has_nextn_predictor() {
            return Ok(None);
        }
        if !Self::supports(source, config) {
            return Err(XrtError::Unsupported(
                "Qwen NextN metadata is present, but the appended MTP block does not match the admitted one-layer Qwen3.6 tensor contract"
                    .to_string(),
            ));
        }
        let index = config.block_count;
        let layer = ResidentQwen35LayerWeights {
            attn_norm: upload_resident_f32_tensor(
                device,
                source,
                &format!("blk.{index}.attn_norm.weight"),
            )?,
            attention: ResidentQwen35AttentionWeights::Full {
                attn_qg: ResidentQuantMatrix::upload(
                    device,
                    source,
                    &format!("blk.{index}.attn_q.weight"),
                )?,
                attn_k: ResidentQuantMatrix::upload(
                    device,
                    source,
                    &format!("blk.{index}.attn_k.weight"),
                )?,
                attn_v: ResidentQuantMatrix::upload(
                    device,
                    source,
                    &format!("blk.{index}.attn_v.weight"),
                )?,
                attn_output: ResidentQuantMatrix::upload(
                    device,
                    source,
                    &format!("blk.{index}.attn_output.weight"),
                )?,
                attn_q_norm: upload_resident_f32_tensor(
                    device,
                    source,
                    &format!("blk.{index}.attn_q_norm.weight"),
                )?,
                attn_k_norm: upload_resident_f32_tensor(
                    device,
                    source,
                    &format!("blk.{index}.attn_k_norm.weight"),
                )?,
            },
            ffn_norm: upload_resident_f32_tensor(
                device,
                source,
                &format!("blk.{index}.post_attention_norm.weight"),
            )?,
            ffn_gate: ResidentQuantMatrix::upload(
                device,
                source,
                &format!("blk.{index}.ffn_gate.weight"),
            )?,
            ffn_up: ResidentQuantMatrix::upload(
                device,
                source,
                &format!("blk.{index}.ffn_up.weight"),
            )?,
            ffn_down: ResidentQuantMatrix::upload(
                device,
                source,
                &format!("blk.{index}.ffn_down.weight"),
            )?,
            embedding_length: config.embedding_length,
        };
        Ok(Some(Self {
            layer,
            enorm: upload_resident_f32_tensor(
                device,
                source,
                &format!("blk.{index}.nextn.enorm.weight"),
            )?,
            hnorm: upload_resident_f32_tensor(
                device,
                source,
                &format!("blk.{index}.nextn.hnorm.weight"),
            )?,
            eh_proj: ResidentQuantMatrix::upload(
                device,
                source,
                &format!("blk.{index}.nextn.eh_proj.weight"),
            )?,
            shared_head_norm: upload_resident_f32_tensor(
                device,
                source,
                &format!("blk.{index}.nextn.shared_head_norm.weight"),
            )?,
        }))
    }
}

struct ResidentQwen35MoeLayerWeights {
    descriptor: MoeLayerDescriptor,
    resident: RwLock<ResidentMoePlacement>,
    attn_norm: GpuF32Tensor,
    attention: ResidentQwen35AttentionWeights,
    ffn_norm: GpuF32Tensor,
    router: ResidentQuantMatrix,
    embedding_length: usize,
}

impl ResidentMoeFfnLayer for ResidentQwen35MoeLayerWeights {
    fn moe_descriptor(&self) -> &MoeLayerDescriptor {
        &self.descriptor
    }

    fn moe_resident(&self) -> &RwLock<ResidentMoePlacement> {
        &self.resident
    }

    fn moe_router(&self) -> &ResidentQuantMatrix {
        &self.router
    }

    fn moe_embedding_length(&self) -> usize {
        self.embedding_length
    }
}

impl ResidentQwen35MoeLayerWeights {
    fn validate_source(
        source: &impl ResidentTensorSource,
        config: &LlamaConfig,
        placements: &[Arc<ExpertPlacementSnapshot>],
    ) -> Result<()> {
        if config.block_count == 0 || !config.is_hybrid() || !config.is_moe() {
            return Err(XrtError::Unsupported(
                "Qwen3.5 hybrid-MoE CUDA weights require a non-empty hybrid MoE model".to_string(),
            ));
        }
        let recurrent = config.deltanet_state_descriptor().ok_or_else(|| {
            XrtError::InvalidMetadata(
                "Qwen3.5 hybrid-MoE CUDA weights require a validated DeltaNet descriptor"
                    .to_string(),
            )
        })?;
        if recurrent.layers().len() != config.block_count {
            return Err(XrtError::InvalidMetadata(format!(
                "Qwen3.5 hybrid-MoE recurrent schedule has {} layers, expected {}",
                recurrent.layers().len(),
                config.block_count
            )));
        }
        if placements.len() != config.block_count {
            return Err(XrtError::Runtime(format!(
                "Qwen3.5 hybrid-MoE CUDA plan has {} placements for {} layers",
                placements.len(),
                config.block_count
            )));
        }
        let expert_count = config.expert_count.ok_or_else(|| {
            XrtError::InvalidMetadata(
                "Qwen3.5 hybrid-MoE CUDA plan is missing expert_count".to_string(),
            )
        })?;
        let selected_per_token = config.expert_used_count.ok_or_else(|| {
            XrtError::InvalidMetadata(
                "Qwen3.5 hybrid-MoE CUDA plan is missing expert_used_count".to_string(),
            )
        })?;
        let dim = config.embedding_length;
        let q_width = config.q_width();
        let qg_width = q_width.checked_mul(2).ok_or_else(|| {
            XrtError::InvalidMetadata("Qwen3.5 hybrid-MoE query/gate width overflowed".to_string())
        })?;
        let kv_width = config.kv_width();
        let head_dim = config.head_dim();
        let conv_channels = recurrent
            .state_size()
            .checked_mul(recurrent.group_count())
            .and_then(|value| value.checked_mul(2))
            .and_then(|value| value.checked_add(recurrent.inner_size()))
            .ok_or_else(|| {
                XrtError::InvalidMetadata(
                    "Qwen3.5 hybrid-MoE convolution channel count overflowed".to_string(),
                )
            })?;
        let conv_elements = conv_channels
            .checked_mul(recurrent.conv_kernel())
            .ok_or_else(|| {
                XrtError::InvalidMetadata(
                    "Qwen3.5 hybrid-MoE convolution weight size overflowed".to_string(),
                )
            })?;
        let head_value_size = recurrent.inner_size() / recurrent.dt_rank();

        for (layer, recurrent_layer) in recurrent.layers().iter().enumerate() {
            let descriptor = MoeLayerDescriptor::new(
                layer,
                expert_count,
                selected_per_token,
                dim,
                config.feed_forward_length,
            )?;
            let placement = &placements[layer];
            if placement.layer_index() != layer
                || placement.expert_count() != descriptor.expert_count()
            {
                return Err(XrtError::Runtime(format!(
                    "Qwen3.5 hybrid-MoE CUDA placement geometry does not match layer {layer}"
                )));
            }
            for (name, len) in [
                (format!("blk.{layer}.attn_norm.weight"), dim),
                (format!("blk.{layer}.post_attention_norm.weight"), dim),
            ] {
                if !matches_f32_vector(source, &name, len) {
                    return Err(XrtError::InvalidTensor(format!(
                        "Qwen3.5 hybrid-MoE tensor `{name}` must be a supported float vector of length {len}"
                    )));
                }
            }
            let router_name = format!("blk.{layer}.ffn_gate_inp.weight");
            if !matches_supported_linear_shape(source, &router_name, expert_count, dim) {
                return Err(XrtError::InvalidTensor(format!(
                    "Qwen3.5 hybrid-MoE router `{router_name}` must be a supported {expert_count}x{dim} matrix"
                )));
            }

            if recurrent_layer.is_some() {
                for (name, rows, cols) in [
                    (format!("blk.{layer}.attn_qkv.weight"), conv_channels, dim),
                    (
                        format!("blk.{layer}.attn_gate.weight"),
                        recurrent.inner_size(),
                        dim,
                    ),
                    (
                        format!("blk.{layer}.ssm_alpha.weight"),
                        recurrent.dt_rank(),
                        dim,
                    ),
                    (
                        format!("blk.{layer}.ssm_beta.weight"),
                        recurrent.dt_rank(),
                        dim,
                    ),
                    (
                        format!("blk.{layer}.ssm_out.weight"),
                        dim,
                        recurrent.inner_size(),
                    ),
                ] {
                    if !matches_supported_linear_shape(source, &name, rows, cols) {
                        return Err(XrtError::InvalidTensor(format!(
                            "Qwen3.5 hybrid-MoE tensor `{name}` must be a supported {rows}x{cols} matrix"
                        )));
                    }
                }
                for (name, len) in [
                    (format!("blk.{layer}.ssm_a"), recurrent.dt_rank()),
                    (format!("blk.{layer}.ssm_dt.bias"), recurrent.dt_rank()),
                    (format!("blk.{layer}.ssm_norm.weight"), head_value_size),
                    (format!("blk.{layer}.ssm_conv1d.weight"), conv_elements),
                ] {
                    if !matches_f32_vector(source, &name, len) {
                        return Err(XrtError::InvalidTensor(format!(
                            "Qwen3.5 hybrid-MoE tensor `{name}` must be a supported float vector of length {len}"
                        )));
                    }
                }
            } else {
                for (name, rows, cols) in [
                    (format!("blk.{layer}.attn_q.weight"), qg_width, dim),
                    (format!("blk.{layer}.attn_k.weight"), kv_width, dim),
                    (format!("blk.{layer}.attn_v.weight"), kv_width, dim),
                    (format!("blk.{layer}.attn_output.weight"), dim, q_width),
                ] {
                    if !matches_supported_linear_shape(source, &name, rows, cols) {
                        return Err(XrtError::InvalidTensor(format!(
                            "Qwen3.5 hybrid-MoE tensor `{name}` must be a supported {rows}x{cols} matrix"
                        )));
                    }
                }
                for (name, len) in [
                    (format!("blk.{layer}.attn_q_norm.weight"), head_dim),
                    (format!("blk.{layer}.attn_k_norm.weight"), head_dim),
                ] {
                    if !matches_f32_vector(source, &name, len) {
                        return Err(XrtError::InvalidTensor(format!(
                            "Qwen3.5 hybrid-MoE tensor `{name}` must be a supported float vector of length {len}"
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    fn try_load_all(
        device: &CudaDevice,
        gguf: &GgufFile,
        source: &GgufResidentTensorSource<'_>,
        config: &LlamaConfig,
        placements: &[Arc<ExpertPlacementSnapshot>],
    ) -> Result<Vec<Self>> {
        Self::validate_source(source, config, placements)?;
        let recurrent = config
            .deltanet_state_descriptor()
            .expect("Qwen3.5 hybrid-MoE resident support validated the descriptor");
        let expert_count = config
            .expert_count
            .expect("Qwen3.5 hybrid-MoE expert count was validated");
        let selected_per_token = config
            .expert_used_count
            .expect("Qwen3.5 hybrid-MoE selected expert count was validated");
        let mut layers = Vec::with_capacity(config.block_count);
        for (layer, recurrent_layer) in recurrent.layers().iter().enumerate() {
            let placement = &placements[layer];
            let mut slots = Vec::with_capacity(placement.gpu_slot_count());
            for &logical_expert in placement.gpu_slots_to_logical() {
                slots.push(Arc::new(ResidentMoeExpertSlot::upload(
                    device,
                    gguf,
                    source,
                    config,
                    layer,
                    usize::from(logical_expert),
                )?));
            }
            let attention = if recurrent_layer.is_some() {
                ResidentQwen35AttentionWeights::DeltaNet {
                    attn_qkv: ResidentQuantMatrix::upload(
                        device,
                        source,
                        &format!("blk.{layer}.attn_qkv.weight"),
                    )?,
                    attn_gate: ResidentQuantMatrix::upload(
                        device,
                        source,
                        &format!("blk.{layer}.attn_gate.weight"),
                    )?,
                    ssm_alpha: ResidentQuantMatrix::upload(
                        device,
                        source,
                        &format!("blk.{layer}.ssm_alpha.weight"),
                    )?,
                    ssm_beta: ResidentQuantMatrix::upload(
                        device,
                        source,
                        &format!("blk.{layer}.ssm_beta.weight"),
                    )?,
                    ssm_a: upload_resident_f32_tensor(
                        device,
                        source,
                        &format!("blk.{layer}.ssm_a"),
                    )?,
                    ssm_dt_bias: upload_resident_f32_tensor(
                        device,
                        source,
                        &format!("blk.{layer}.ssm_dt.bias"),
                    )?,
                    ssm_norm: upload_resident_f32_tensor(
                        device,
                        source,
                        &format!("blk.{layer}.ssm_norm.weight"),
                    )?,
                    ssm_out: ResidentQuantMatrix::upload(
                        device,
                        source,
                        &format!("blk.{layer}.ssm_out.weight"),
                    )?,
                    conv1d: upload_resident_f32_tensor(
                        device,
                        source,
                        &format!("blk.{layer}.ssm_conv1d.weight"),
                    )?,
                }
            } else {
                ResidentQwen35AttentionWeights::Full {
                    attn_qg: ResidentQuantMatrix::upload(
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
                }
            };
            layers.push(Self {
                descriptor: MoeLayerDescriptor::new(
                    layer,
                    expert_count,
                    selected_per_token,
                    config.embedding_length,
                    config.feed_forward_length,
                )?,
                resident: RwLock::new(ResidentMoePlacement {
                    snapshot: Arc::clone(placement),
                    slots,
                }),
                attn_norm: upload_resident_f32_tensor(
                    device,
                    source,
                    &format!("blk.{layer}.attn_norm.weight"),
                )?,
                attention,
                ffn_norm: upload_resident_f32_tensor(
                    device,
                    source,
                    &format!("blk.{layer}.post_attention_norm.weight"),
                )?,
                router: ResidentQuantMatrix::upload(
                    device,
                    source,
                    &format!("blk.{layer}.ffn_gate_inp.weight"),
                )?,
                embedding_length: config.embedding_length,
            });
        }
        Ok(layers)
    }
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
            DType::Q8_0 | DType::Q4_0 | DType::Q4_K | DType::Q5_K | DType::Q6_K | DType::MXFP4
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

    #[cfg(feature = "moe-route-trace")]
    fn start_moe_route_trace(&self, max_entries: usize) -> Result<()> {
        self.require_cpu_reference_model()?
            .start_moe_route_trace(max_entries)
    }

    #[cfg(feature = "moe-route-trace")]
    fn take_moe_route_trace(&self) -> Result<Option<MoeRouteTrace>> {
        Ok(self.require_cpu_reference_model()?.take_moe_route_trace())
    }

    fn new_session(&self, cache_mode: KvCacheMode, page_tokens: usize) -> BackendSession {
        let config = &self.config;
        let layer_widths = config
            .gemma4_layer_kv_widths()
            .unwrap_or_else(|| vec![config.kv_width(); config.total_block_count]);
        let mut session = BackendSession::new_cuda_with_kv_budget_page_tokens_and_layer_widths(
            self.device.clone(),
            cache_mode,
            layer_widths,
            config.context_length,
            page_tokens,
            Some(self.kv_budget_bytes),
        );
        if let Some(arena) = &self.allocation_arena {
            session
                .attach_gpu_allocation_arena(Arc::clone(arena))
                .expect("new CUDA session must not allocate before arena attachment");
        }
        session.attach_cuda_capture_gate(Arc::clone(&self.qwen35_capture_gate));
        session.attach_moe_graph_execution_gate(Arc::clone(&self.moe_graph_execution_gate));
        session.configure_cuda_graph_mode(self.cuda_graph_mode);
        session.set_initial_recurrent_state(config.deltanet_state_descriptor().cloned());
        session
    }

    fn prepare_session_state(&self, session: &mut BackendSession) -> Result<()> {
        let _capture_guard = self.qwen35_capture_gate.read();
        session.prepare_recurrent_state()
    }

    fn prepare_request(&self) -> Result<()> {
        self.prepare_adaptive_moe_request()
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
            None,
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
        if let Some(logits) = self.try_forward_batch_moe_layerwise(
            token_ids,
            start_position,
            session,
            &HashMap::new(),
            false,
        )? {
            return Ok(logits);
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
        if let Some(logits) = self.try_forward_batch_moe_layerwise(
            token_ids,
            start_position,
            session,
            &embedding_overrides,
            false,
        )? {
            return Ok(logits);
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
                embedding_overrides.get(&index).map(Vec::as_slice),
                batch_total_len,
                None,
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
        if let Some(logits) = self.try_forward_batch_moe_layerwise(
            token_ids,
            start_position,
            session,
            &HashMap::new(),
            true,
        )? {
            return Ok(logits);
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
                true,
                None,
                batch_total_len,
                None,
                None,
            )? {
                return Err(Self::decode_unsupported());
            }
            all_logits.extend_from_slice(&logits);
        }
        Ok(all_logits)
    }

    fn draft_mtp_greedy(
        &self,
        next_token_id: u32,
        max_draft_tokens: usize,
        session: &mut BackendSession,
    ) -> Result<Option<Vec<u32>>> {
        self.draft_qwen35_mtp_greedy(next_token_id, max_draft_tokens.min(3), session)
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

    fn moe_gpu_expert_slots(&self) -> usize {
        self.gpu_expert_slots
    }

    fn moe_gpu_expert_bytes(&self) -> u64 {
        self.gpu_expert_bytes
    }

    fn moe_placement_generation(&self) -> u64 {
        self.placement_generation.load(Ordering::Acquire)
    }

    fn moe_placement_manifest_sha256(&self) -> Option<&str> {
        self.placement_manifest_sha256.as_deref()
    }

    fn cuda_moe_telemetry(&self) -> CudaMoeTelemetrySnapshot {
        self.moe_telemetry.snapshot()
    }

    fn supports_cuda_graph_decode(&self) -> bool {
        let config = &self.config;
        if config.is_gemma4() || self.q8_0_probe.is_none() {
            return false;
        }
        if config.is_hybrid() {
            return if config.is_moe() {
                self.qwen35_moe_layer_probes
                    .as_ref()
                    .is_some_and(|layers| layers.len() == config.block_count)
            } else {
                self.qwen35_layer_probes
                    .as_ref()
                    .is_some_and(|layers| layers.len() == config.block_count)
            };
        }
        if config.is_moe() {
            return self
                .moe_layer_probes
                .as_ref()
                .is_some_and(|layers| layers.len() == config.block_count);
        }
        self.q8_0_layer_probes
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
        let loaded_layer_count = if self.config.is_hybrid() {
            if self.config.is_moe() {
                self.qwen35_moe_layer_probes.as_ref().map(Vec::len)
            } else {
                self.qwen35_layer_probes.as_ref().map(Vec::len)
            }
        } else if self.config.is_gemma4() {
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
        DType::Q8_0 | DType::Q4_0 | DType::MXFP4 => blocks().and_then(|blocks| {
            blocks
                .checked_mul((4 + info.dtype.block_size()) as u64)
                .ok_or_else(|| {
                    XrtError::Runtime(
                        "CUDA resident Q4_0/Q8_0/MXFP4 byte count overflow".to_string(),
                    )
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
    let page_table_bytes = match mode {
        KvCacheMode::F32 => checked_mul(
            checked_mul(page_count, 2, "CUDA shared F32 KV pointer table count")?,
            std::mem::size_of::<u64>(),
            "CUDA shared F32 KV pointer table bytes",
        )?,
        KvCacheMode::AgentAdaptive => checked_mul(
            checked_mul(page_count, 2, "CUDA adaptive KV page table count")?,
            std::mem::size_of::<u32>(),
            "CUDA adaptive KV page table bytes",
        )?,
        KvCacheMode::Q8 | KvCacheMode::KeyQ4ValueQ8 => checked_mul(
            page_count,
            std::mem::size_of::<u32>(),
            "CUDA KV page table bytes",
        )?,
    } as u64;
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

#[cfg(test)]
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

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn graph_cache_rejects_stale_identity_device_and_epochs_and_releases_its_lease() {
        let key = |placement_generation| CudaDecodeGraphKey {
            model_identity: "synthetic-model".to_string(),
            architecture: "synthetic-moe".to_string(),
            device_ordinal: 0,
            weight_kinds: vec!["q8_0"],
            cache_mode: KvCacheMode::F32,
            shared_kv_pages: false,
            kv_capacity: 32,
            placement_generation,
            scratch_generation: 3,
            recurrent_buffer_generation: None,
            layer_count: 2,
            embedding_length: 8,
            kv_width: 4,
            feed_forward_length: 16,
            vocab_size: 32,
            attention_head_count: 2,
            attention_head_count_kv: 1,
            head_dim: 4,
        };
        let current = key(7);
        let stale = key(6);
        let mut wrong_model = current.clone();
        wrong_model.model_identity = "different-model".to_string();
        let mut wrong_device = current.clone();
        wrong_device.device_ordinal = 1;
        let mut stale_scratch = current.clone();
        stale_scratch.scratch_generation = current.scratch_generation - 1;
        let arena = Arc::new(GpuAllocationArena::default());
        arena.configure_budget(1024 * 1024).unwrap();
        let graph = CudaGraphExec::default();
        let accounted_bytes = graph.accounting_bytes();
        let allocation = reserve_cuda_graph_allocation(Some(&arena), &graph).unwrap();
        let mut state = CudaDecodeGraphState::new(CudaGraphMode::Enabled);
        state.captured(
            current.clone(),
            graph,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            allocation,
        );

        assert!(state.has_executable_for(&current));
        assert!(!state.has_executable_for(&stale));
        assert!(!state.has_executable_for(&wrong_model));
        assert!(!state.has_executable_for(&wrong_device));
        assert!(!state.has_executable_for(&stale_scratch));
        assert!(state.executable_for(&stale).is_none());
        assert_eq!(arena.snapshot().by_class.graph_bytes, accounted_bytes);

        state.reset();
        assert_eq!(arena.snapshot().by_class.graph_bytes, 0);
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
    fn cpu_order_q4_k_matvec_is_scoped_to_qwen3_moe() {
        for architecture in ["qwen3moe", "qwen3_moe"] {
            assert!(qwen3_moe_uses_cpu_order_q4_k_matvec(architecture));
        }
        for architecture in ["qwen3", "qwen2", "llama"] {
            assert!(!qwen3_moe_uses_cpu_order_q4_k_matvec(architecture));
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
            576
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
            832
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
    fn shared_quantized_projected_bytes_count_live_pages_and_stable_tables() {
        assert_eq!(
            BackendSession::projected_shared_quantized_bytes(KvCacheMode::Q8, &[64, 128], 3, 8, 2,)
                .unwrap(),
            1632
        );
        assert_eq!(
            BackendSession::projected_shared_quantized_bytes(
                KvCacheMode::KeyQ4ValueQ8,
                &[64, 128],
                3,
                8,
                2,
            )
            .unwrap(),
            1264
        );
        assert!(
            BackendSession::projected_shared_quantized_bytes(KvCacheMode::F32, &[64], 1, 8, 2,)
                .is_err()
        );
    }

    #[test]
    fn shared_adaptive_bytes_cover_both_tiers_routes_and_hot_rebuild_headroom() {
        assert_eq!(
            BackendSession::projected_shared_adaptive_bytes(&[64, 128], 3, 8, 2).unwrap(),
            7600
        );
        assert_eq!(
            BackendSession::shared_adaptive_reserved_bytes(&[64, 128], 8, 2).unwrap(),
            27880
        );
        assert!(BackendSession::shared_adaptive_reserved_bytes(&[128], 0, 2).is_err());
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
    fn cuda_adaptive_graph_requires_entire_suffix_in_final_hot_window() {
        let policy = SessionPolicy {
            recent_window_tokens: 2,
            ..SessionPolicy::agent_adaptive()
        };
        assert!(
            BackendSession::cuda_adaptive_graph_suffix_is_hot_for_policy(&policy, 3, &[], 3, 5,)
        );
        assert!(
            !BackendSession::cuda_adaptive_graph_suffix_is_hot_for_policy(&policy, 3, &[], 2, 5,)
        );
        assert!(
            !BackendSession::cuda_adaptive_graph_suffix_is_hot_for_policy(&policy, 3, &[], 6, 5,)
        );
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
        assert!(attached.cuda_graph_decode_ready());
        assert_eq!(attached.cuda_graph_capture_status(), Some("not-captured"));
        assert!(attached.prepare_cuda_graph_generation_capacity(4));

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
            assert_eq!(cache.resident_page_count(), 2);
            assert_eq!(cache.shared_page_count(), 1);
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

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA-capable device and driver"]
    fn cuda_runtime_shared_quantized_prefix_attachment_preserves_rows_and_cow() -> Result<()> {
        fn run_mode(device: &CudaDevice, mode: KvCacheMode) -> Result<()> {
            let layer_widths = vec![128];
            let mut source = BackendSession::new_cuda_with_kv_budget_page_tokens_and_layer_widths(
                device.clone(),
                mode,
                layer_widths.clone(),
                8,
                2,
                Some(64 * 1024 * 1024),
            );
            source.configure_cuda_graph_mode(CudaGraphMode::Enabled);
            source.prepare_for_total_len(3)?;

            for position in 0..3 {
                let key = (0..128)
                    .map(|index| match index {
                        0..=31 => position as f32 + 0.25,
                        32..=63 => position as f32 + 8.0,
                        64..=95 => -(position as f32) - 0.25,
                        _ => -(position as f32) - 8.0,
                    })
                    .collect::<Vec<_>>();
                let value = (0..128)
                    .map(|index| (index as f32 - 63.5) / (position as f32 + 32.0))
                    .collect::<Vec<_>>();
                let key = device.upload_f32(&key)?;
                let value = device.upload_f32(&value)?;
                match source.cuda_layer_cache_mut(0)? {
                    CudaLayerKvStore::Q8(cache) if mode == KvCacheMode::Q8 => {
                        device.append_q8_layer_kv(cache, &key, &value)?;
                    }
                    CudaLayerKvStore::KeyQ4ValueQ8(cache) if mode == KvCacheMode::KeyQ4ValueQ8 => {
                        device.append_key_q4_value_q8_layer_kv(cache, &key, &value)?;
                    }
                    other => {
                        return Err(XrtError::Runtime(format!(
                            "source CUDA {} prefix used {} storage",
                            mode.as_str(),
                            other.mode().as_str()
                        )));
                    }
                }
            }

            let expected_last = match source.cuda_layer_cache_mut(0)? {
                CudaLayerKvStore::Q8(cache) if mode == KvCacheMode::Q8 => {
                    let (key, value) = device.dequantize_q8_layer_kv(cache, 2)?;
                    (device.download_f32(&key)?, device.download_f32(&value)?)
                }
                CudaLayerKvStore::KeyQ4ValueQ8(cache) if mode == KvCacheMode::KeyQ4ValueQ8 => {
                    let (key, value) = device.dequantize_key_q4_value_q8_layer_kv(cache, 2)?;
                    (device.download_f32(&key)?, device.download_f32(&value)?)
                }
                other => {
                    return Err(XrtError::Runtime(format!(
                        "source CUDA {} prefix used {} storage",
                        mode.as_str(),
                        other.mode().as_str()
                    )));
                }
            };

            let snapshot = source.snapshot_prefix(3)?.ok_or_else(|| {
                XrtError::Runtime(format!(
                    "CUDA {} prefix snapshot was unavailable",
                    mode.as_str()
                ))
            })?;
            let snapshot_caches = match &snapshot {
                BackendPrefixSnapshot::Cuda {
                    layer_caches,
                    prefix_len,
                    allocated_bytes,
                    ..
                } => {
                    assert_eq!(*prefix_len, 3);
                    assert!(*allocated_bytes > 0);
                    layer_caches.clone()
                }
                BackendPrefixSnapshot::Cpu { .. } => {
                    return Err(XrtError::Runtime(
                        "CUDA quantized session produced a CPU prefix snapshot".to_string(),
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
                    assert!(matches!(
                        (&layer_caches[0], mode),
                        (CudaLayerKvStore::Q8(_), KvCacheMode::Q8)
                            | (CudaLayerKvStore::KeyQ4ValueQ8(_), KvCacheMode::KeyQ4ValueQ8)
                    ));
                }
                BackendSession::Cpu { .. } => {
                    return Err(XrtError::Runtime(
                        "source quantized session changed backend while snapshotting".to_string(),
                    ));
                }
            }
            assert!(matches!(
                (&snapshot_caches[0], mode),
                (CudaLayerKvStore::SharedQ8(_), KvCacheMode::Q8)
                    | (
                        CudaLayerKvStore::SharedKeyQ4ValueQ8(_),
                        KvCacheMode::KeyQ4ValueQ8
                    )
            ));

            let mut attached = BackendSession::new_cuda_with_kv_budget_page_tokens_and_layer_widths(
                device.clone(),
                mode,
                layer_widths,
                8,
                2,
                Some(64 * 1024 * 1024),
            );
            attached.configure_cuda_graph_mode(CudaGraphMode::Enabled);
            assert_eq!(attached.attach_prefix_snapshot(&snapshot)?, 3);
            attached.prepare_for_total_len(4)?;
            assert_eq!(attached.cuda_graph_capture_status(), Some("not-captured"));
            assert!(attached.cuda_graph_decode_ready());
            assert!(attached.prepare_cuda_graph_generation_capacity(4));
            assert_eq!(attached.cuda_graph_capture_status(), Some("not-captured"));

            let replacement_key = vec![11.0; 128];
            let replacement_value = vec![-7.0; 128];
            let replacement_key_device = device.upload_f32(&replacement_key)?;
            let replacement_value_device = device.upload_f32(&replacement_value)?;
            match attached.cuda_layer_cache_mut(0)? {
                CudaLayerKvStore::SharedQ8(cache) if mode == KvCacheMode::Q8 => {
                    assert_eq!(cache.shared_page_count(), 1);
                    cache.append(&replacement_key_device, &replacement_value_device)?;
                    assert_eq!(cache.shared_page_count(), 1);
                    assert_eq!(cache.row(2)?, expected_last);
                }
                CudaLayerKvStore::SharedKeyQ4ValueQ8(cache)
                    if mode == KvCacheMode::KeyQ4ValueQ8 =>
                {
                    assert_eq!(cache.shared_page_count(), 1);
                    cache.append(&replacement_key_device, &replacement_value_device)?;
                    assert_eq!(cache.shared_page_count(), 1);
                    assert_eq!(cache.row(2)?, expected_last);
                }
                other => {
                    return Err(XrtError::Runtime(format!(
                        "attached CUDA {} prefix used {} storage",
                        mode.as_str(),
                        other.mode().as_str()
                    )));
                }
            }

            match &snapshot_caches[0] {
                CudaLayerKvStore::SharedQ8(cache) if mode == KvCacheMode::Q8 => {
                    assert_eq!(cache.len(), 3);
                    assert_eq!(cache.row(2)?, expected_last);
                }
                CudaLayerKvStore::SharedKeyQ4ValueQ8(cache)
                    if mode == KvCacheMode::KeyQ4ValueQ8 =>
                {
                    assert_eq!(cache.len(), 3);
                    assert_eq!(cache.row(2)?, expected_last);
                }
                other => {
                    return Err(XrtError::Runtime(format!(
                        "immutable CUDA {} prefix used {} storage",
                        mode.as_str(),
                        other.mode().as_str()
                    )));
                }
            }
            Ok(())
        }

        let device = CudaDevice::new(0)?;
        run_mode(&device, KvCacheMode::Q8)?;
        run_mode(&device, KvCacheMode::KeyQ4ValueQ8)
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA-capable device and driver"]
    fn cuda_runtime_shared_adaptive_prefix_migrates_aged_rows_and_preserves_snapshot() -> Result<()>
    {
        let device = CudaDevice::new(0)?;
        let layer_widths = vec![128];
        let policy = SessionPolicy {
            recent_window_tokens: 2,
            ..SessionPolicy::agent_adaptive()
        };
        let spans = vec![PromptSpan {
            kind: crate::policy::PromptSpanKind::System,
            token_start: 1,
            token_end: 2,
        }];
        let mut source = BackendSession::new_cuda_with_kv_budget_page_tokens_and_layer_widths(
            device.clone(),
            KvCacheMode::AgentAdaptive,
            layer_widths.clone(),
            8,
            2,
            Some(64 * 1024 * 1024),
        );
        source.configure_policy(policy.clone(), 3, &spans);
        source.configure_cuda_graph_mode(CudaGraphMode::Enabled);
        source.prepare_for_total_len(3)?;
        let source_mask = source.cuda_adaptive_hot_position_mask(3).ok_or_else(|| {
            XrtError::Runtime("CUDA adaptive source mask was unavailable".to_string())
        })?;
        assert_eq!(source_mask, vec![0, 1, 1]);

        let keys = (0..3)
            .map(|position| {
                (0..128)
                    .map(|index| match index {
                        0..=31 => position as f32 + 0.25,
                        32..=63 => position as f32 + 8.0,
                        64..=95 => -(position as f32) - 0.25,
                        _ => -(position as f32) - 8.0,
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let values = (0..3)
            .map(|position| {
                (0..128)
                    .map(|index| (index as f32 - 63.5) / (position as f32 + 32.0))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for position in 0..3 {
            let key = device.upload_f32(&keys[position])?;
            let value = device.upload_f32(&values[position])?;
            let CudaLayerKvStore::AgentAdaptive {
                hot,
                cold,
                routes,
                hot_mask,
            } = source.cuda_layer_cache_mut(0)?
            else {
                return Err(XrtError::Runtime(
                    "adaptive source unexpectedly used shared storage".to_string(),
                ));
            };
            if source_mask[position] != 0 {
                let local_position = hot.len();
                device.append_layer_kv(hot, &key, &value)?;
                device.append_adaptive_kv_route(routes, true, local_position)?;
                hot_mask.push(1);
            } else {
                let local_position = cold.len();
                device.append_key_q4_value_q8_layer_kv(cold, &key, &value)?;
                device.append_adaptive_kv_route(routes, false, local_position)?;
                hot_mask.push(0);
            }
        }
        let expected_cold = match source.cuda_layer_cache_mut(0)? {
            CudaLayerKvStore::AgentAdaptive { cold, .. } => {
                let (key, value) = device.dequantize_key_q4_value_q8_layer_kv(cold, 0)?;
                (device.download_f32(&key)?, device.download_f32(&value)?)
            }
            _ => unreachable!("adaptive source storage was checked above"),
        };
        let expected_migrated = {
            let mut reference = device.alloc_paged_key_q4_value_q8_layer_kv_cache(1, 128, 2)?;
            let key = device.upload_f32(&keys[2])?;
            let value = device.upload_f32(&values[2])?;
            device.append_key_q4_value_q8_layer_kv(&mut reference, &key, &value)?;
            let (key, value) = device.dequantize_key_q4_value_q8_layer_kv(&reference, 0)?;
            (device.download_f32(&key)?, device.download_f32(&value)?)
        };

        let snapshot = source.snapshot_prefix(3)?.ok_or_else(|| {
            XrtError::Runtime("CUDA adaptive prefix snapshot was unavailable".to_string())
        })?;
        let snapshot_caches = match &snapshot {
            BackendPrefixSnapshot::Cuda {
                layer_caches,
                allocated_bytes,
                ..
            } => {
                assert!(*allocated_bytes > 0);
                layer_caches.clone()
            }
            BackendPrefixSnapshot::Cpu { .. } => {
                return Err(XrtError::Runtime(
                    "CUDA adaptive session produced a CPU prefix snapshot".to_string(),
                ));
            }
        };
        let CudaLayerKvStore::SharedAgentAdaptive(retained) = &snapshot_caches[0] else {
            return Err(XrtError::Runtime(
                "CUDA adaptive prefix did not convert to shared storage".to_string(),
            ));
        };
        assert_eq!(retained.hot_len(), 2);
        assert_eq!(retained.cold_len(), 1);
        assert_eq!(retained.row(0)?, expected_cold);
        assert_eq!(retained.row(2)?, (keys[2].clone(), values[2].clone()));

        let mut attached = BackendSession::new_cuda_with_kv_budget_page_tokens_and_layer_widths(
            device.clone(),
            KvCacheMode::AgentAdaptive,
            layer_widths,
            8,
            2,
            Some(64 * 1024 * 1024),
        );
        attached.configure_policy(policy, 3, &spans);
        attached.configure_cuda_graph_mode(CudaGraphMode::Enabled);
        assert_eq!(attached.attach_prefix_snapshot(&snapshot)?, 3);
        assert!(attached.prepare_cuda_graph_generation_capacity(5));
        assert_eq!(attached.cuda_graph_capture_status(), Some("not-captured"));
        attached.prepare_for_total_len(4)?;
        assert_eq!(
            attached.cuda_adaptive_hot_position_mask(4),
            Some(vec![0, 1, 0, 1, 1])
        );

        let replacement_key = vec![11.0f32; 128];
        let replacement_value = vec![-7.0f32; 128];
        let replacement_key_device = device.upload_f32(&replacement_key)?;
        let replacement_value_device = device.upload_f32(&replacement_value)?;
        let CudaLayerKvStore::SharedAgentAdaptive(cache) = attached.cuda_layer_cache_mut(0)? else {
            return Err(XrtError::Runtime(
                "attached adaptive prefix did not materialize shared storage".to_string(),
            ));
        };
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.hot_len(), 1);
        assert_eq!(cache.cold_len(), 2);
        assert_eq!(cache.row(0)?, expected_cold);
        assert_eq!(cache.row(1)?, (keys[1].clone(), values[1].clone()));
        assert_eq!(cache.row(2)?, expected_migrated);
        cache.append(true, &replacement_key_device, &replacement_value_device)?;
        assert_eq!(cache.row(3)?, (replacement_key, replacement_value));

        assert_eq!(retained.len(), 3);
        assert_eq!(retained.hot_len(), 2);
        assert_eq!(retained.cold_len(), 1);
        assert_eq!(retained.row(0)?, expected_cold);
        assert_eq!(retained.row(2)?, (keys[2].clone(), values[2].clone()));
        Ok(())
    }

    #[test]
    fn cuda_decode_scratch_estimate_matches_declared_buffer_geometry() {
        let expected_elements = 4 * 2 + 3 * 3 + 3 * 4 + 2 * 5 + 6;
        assert_eq!(
            CudaDecodeScratch::estimated_allocated_bytes(2, 3, 4, 5, 6, None, None).unwrap(),
            (expected_elements * std::mem::size_of::<f32>() + 4 * std::mem::size_of::<u32>())
                as u64
        );
        assert!(CudaDecodeScratch::estimated_allocated_bytes(
            usize::MAX,
            usize::MAX,
            usize::MAX,
            usize::MAX,
            usize::MAX,
            None,
            None,
        )
        .is_err());
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA-capable device and driver"]
    fn cuda_allocation_arena_tracks_kv_scratch_and_prefix_lifetimes() -> Result<()> {
        let device = CudaDevice::new(0)?;
        let arena = Arc::new(GpuAllocationArena::default());
        arena.configure_budget(64 * 1024 * 1024)?;
        let mut session = BackendSession::new_cuda_with_kv_budget_and_page_tokens(
            device.clone(),
            KvCacheMode::F32,
            2,
            64,
            32,
            4,
            Some(32 * 1024 * 1024),
        );
        session.attach_gpu_allocation_arena(Arc::clone(&arena))?;
        session.prepare_for_total_len(4)?;
        session.ensure_cuda_decode_scratch(&device, 64, 64, 64, 128, 256, 32, None, None)?;

        let kv_bytes = session.cuda_kv_allocated_bytes();
        let scratch_bytes = session.cuda_scratch_allocated_bytes();
        let live = arena.snapshot();
        assert_eq!(live.by_class.kv_cache_bytes, kv_bytes);
        assert_eq!(live.by_class.scratch_bytes, scratch_bytes);

        let prefix = session
            .snapshot_prefix(0)?
            .expect("CUDA prefix snapshot should exist");
        session.clear();
        assert_eq!(arena.snapshot().by_class.kv_cache_bytes, kv_bytes);
        drop(prefix);
        assert_eq!(arena.snapshot().by_class.kv_cache_bytes, 0);
        drop(session);
        assert_eq!(arena.snapshot().allocated_bytes, 0);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA-capable device and driver"]
    fn cuda_prefix_fork_accounts_pointer_tables_and_only_copied_pages() -> Result<()> {
        let device = CudaDevice::new(0)?;
        let arena = Arc::new(GpuAllocationArena::default());
        arena.configure_budget(64 * 1024 * 1024)?;
        let layer_count = 2;
        let width = 64;
        let max_len = 32;
        let page_tokens = 4;
        let mut source = BackendSession::new_cuda_with_kv_budget_and_page_tokens(
            device.clone(),
            KvCacheMode::F32,
            layer_count,
            width,
            max_len,
            page_tokens,
            Some(32 * 1024 * 1024),
        );
        source.attach_gpu_allocation_arena(Arc::clone(&arena))?;
        source.prepare_for_total_len(1)?;
        let capacity = source
            .cuda_kv_capacity()
            .expect("source KV capacity should be allocated");
        let base_bytes = arena.snapshot().by_class.kv_cache_bytes;
        let prefix = source
            .snapshot_prefix(0)?
            .expect("empty CUDA prefix snapshot should exist");

        let mut fork = BackendSession::new_cuda_with_kv_budget_and_page_tokens(
            device.clone(),
            KvCacheMode::F32,
            layer_count,
            width,
            max_len,
            page_tokens,
            Some(32 * 1024 * 1024),
        );
        fork.attach_gpu_allocation_arena(Arc::clone(&arena))?;
        assert_eq!(fork.attach_prefix_snapshot(&prefix)?, 0);
        let transfers_before = device.transfer_stats();
        fork.prepare_for_total_len(1)?;
        let transfers = device.transfer_stats().saturating_sub(transfers_before);

        let page_count = capacity.div_ceil(page_tokens);
        let private_pointer_bytes =
            (layer_count * page_count * 2 * std::mem::size_of::<u64>()) as u64;
        let copied_page_bytes =
            (layer_count * page_tokens * width * 2 * std::mem::size_of::<f32>()) as u64;
        assert_eq!(transfers.device_to_device_bytes, copied_page_bytes);
        assert_eq!(
            arena.snapshot().by_class.kv_cache_bytes,
            base_bytes + private_pointer_bytes + copied_page_bytes
        );

        device.synchronize()?;
        drop(fork);
        assert_eq!(arena.snapshot().by_class.kv_cache_bytes, base_bytes);
        drop(prefix);
        drop(source);
        assert_eq!(arena.snapshot().allocated_bytes, 0);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA-capable device and driver"]
    fn cuda_moe_staging_is_leased_and_cleared_on_session_reset() -> Result<()> {
        let device = CudaDevice::new(0)?;
        let arena = Arc::new(GpuAllocationArena::default());
        arena.configure_budget(64 * 1024 * 1024)?;
        let mut session = BackendSession::new_cuda_with_kv_budget(
            device.clone(),
            KvCacheMode::F32,
            1,
            8,
            32,
            Some(32 * 1024 * 1024),
        );
        session.attach_gpu_allocation_arena(Arc::clone(&arena))?;
        let geometry = MoeScratchGeometry {
            expert_count: 4,
            selected_per_token: 2,
            embedding_length: 8,
            intermediate_size: 16,
            shared_intermediate_size: Some(12),
        };
        session.ensure_cuda_decode_scratch(&device, 8, 8, 8, 16, 32, 32, Some(geometry), None)?;
        let staging_bytes = geometry.pinned_bytes()?;
        assert_eq!(session.cuda_staging_allocated_bytes(), staging_bytes);
        assert_eq!(arena.snapshot().by_class.staging_bytes, staging_bytes);

        if let BackendSession::Cuda {
            decode_scratch: Some(scratch),
            ..
        } = &mut session
        {
            let host = &scratch.moe.as_ref().expect("MoE scratch should exist").host;
            let mut staging = host.lock();
            staging.router_logits.as_mut_slice().fill(1.0);
            staging
                .input
                .as_mut()
                .expect("input staging should exist")
                .as_mut_slice()
                .fill(2.0);
            staging.outputs.as_mut_slice().fill(3.0);
            staging.gate.fill(4.0);
            staging.up.fill(5.0);
            staging.shared_gate.fill(6.0);
            staging.shared_up.fill(7.0);
        } else {
            panic!("expected allocated CUDA MoE scratch");
        }

        session.clear();
        if let BackendSession::Cuda {
            decode_scratch: Some(scratch),
            ..
        } = &session
        {
            let staging = scratch
                .moe
                .as_ref()
                .expect("MoE scratch should exist")
                .host
                .lock();
            assert!(staging
                .router_logits
                .as_slice()
                .iter()
                .all(|&value| value == 0.0));
            assert!(staging
                .input
                .as_ref()
                .expect("input staging should exist")
                .as_slice()
                .iter()
                .all(|&value| value == 0.0));
            assert!(staging.outputs.as_slice().iter().all(|&value| value == 0.0));
            assert!(staging.gate.iter().all(|&value| value == 0.0));
            assert!(staging.up.iter().all(|&value| value == 0.0));
            assert!(staging.shared_gate.iter().all(|&value| value == 0.0));
            assert!(staging.shared_up.iter().all(|&value| value == 0.0));
        } else {
            panic!("expected retained CUDA MoE scratch after reset");
        }

        drop(session);
        assert_eq!(arena.snapshot().allocated_bytes, 0);
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
