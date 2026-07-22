use std::{env, sync::Arc};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use xrt_core::{Result, XrtError};
use xrt_cuda::{CudaAllocationStats, CudaMemoryPoolStats, CudaTransferStats};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CudaGraphMode {
    Disabled,
    Enabled,
    Auto,
}

impl Default for CudaGraphMode {
    fn default() -> Self {
        Self::Auto
    }
}

impl CudaGraphMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Enabled => "enabled",
            Self::Auto => "auto",
        }
    }

    fn parse(value: Option<&str>) -> Option<Self> {
        match value?.trim().to_ascii_lowercase().as_str() {
            "0" | "false" | "off" | "disabled" => Some(Self::Disabled),
            "1" | "true" | "on" | "enabled" => Some(Self::Enabled),
            "auto" => Some(Self::Auto),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GpuResourceConfig {
    pub device_ordinal: usize,
    pub memory_fraction: f32,
    pub reserved_mb: u64,
    pub kv_fraction: f32,
    pub cuda_graph_mode: CudaGraphMode,
}

impl Default for GpuResourceConfig {
    fn default() -> Self {
        Self {
            device_ordinal: 0,
            memory_fraction: 0.90,
            reserved_mb: 1024,
            kv_fraction: 0.30,
            cuda_graph_mode: CudaGraphMode::Auto,
        }
    }
}

impl GpuResourceConfig {
    pub fn from_env() -> Self {
        Self::from_values(
            env::var("XRT_CUDA_DEVICE").ok().as_deref(),
            env::var("XRT_GPU_MEMORY_FRACTION").ok().as_deref(),
            env::var("XRT_GPU_RESERVED_MB").ok().as_deref(),
            env::var("XRT_GPU_KV_FRACTION").ok().as_deref(),
            env::var("XRT_CUDA_GRAPH").ok().as_deref(),
        )
    }

    fn from_values(
        device_ordinal: Option<&str>,
        memory_fraction: Option<&str>,
        reserved_mb: Option<&str>,
        kv_fraction: Option<&str>,
        cuda_graph_mode: Option<&str>,
    ) -> Self {
        let default = Self::default();
        Self {
            device_ordinal: parse_usize(device_ordinal).unwrap_or(default.device_ordinal),
            memory_fraction: parse_fraction(memory_fraction).unwrap_or(default.memory_fraction),
            reserved_mb: parse_u64(reserved_mb).unwrap_or(default.reserved_mb),
            kv_fraction: parse_fraction(kv_fraction).unwrap_or(default.kv_fraction),
            cuda_graph_mode: CudaGraphMode::parse(cuda_graph_mode)
                .unwrap_or(default.cuda_graph_mode),
        }
    }

    pub fn reserved_bytes(&self) -> u64 {
        self.reserved_mb.saturating_mul(1024 * 1024)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuTransferStats {
    pub host_to_device_calls: u64,
    pub host_to_device_bytes: u64,
    pub device_to_host_calls: u64,
    pub device_to_host_bytes: u64,
    pub device_to_device_calls: u64,
    pub device_to_device_bytes: u64,
}

const GPU_ALLOCATION_CLASS_COUNT: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuAllocationClass {
    ModelWeights,
    ExpertWeights,
    KvCache,
    Scratch,
    Staging,
    RecurrentState,
    Graph,
    ImageComponentWeights,
    PromptEmbeddings,
    ImageConditioning,
    LatentState,
    DenoiserPersistentScratch,
    DenoiserTransientScratch,
    VaeScratch,
    PreviewScratch,
    OutputStaging,
}

impl GpuAllocationClass {
    fn index(self) -> usize {
        match self {
            Self::ModelWeights => 0,
            Self::ExpertWeights => 1,
            Self::KvCache => 2,
            Self::Scratch => 3,
            Self::Staging => 4,
            Self::RecurrentState => 5,
            Self::Graph => 6,
            Self::ImageComponentWeights => 7,
            Self::PromptEmbeddings => 8,
            Self::ImageConditioning => 9,
            Self::LatentState => 10,
            Self::DenoiserPersistentScratch => 11,
            Self::DenoiserTransientScratch => 12,
            Self::VaeScratch => 13,
            Self::PreviewScratch => 14,
            Self::OutputStaging => 15,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuAllocationBreakdown {
    pub model_weight_bytes: u64,
    pub expert_weight_bytes: u64,
    pub kv_cache_bytes: u64,
    pub scratch_bytes: u64,
    pub staging_bytes: u64,
    pub recurrent_state_bytes: u64,
    pub graph_bytes: u64,
    pub image_component_weight_bytes: u64,
    pub prompt_embedding_bytes: u64,
    pub image_conditioning_bytes: u64,
    pub latent_state_bytes: u64,
    pub denoiser_persistent_scratch_bytes: u64,
    pub denoiser_transient_scratch_bytes: u64,
    pub vae_scratch_bytes: u64,
    pub preview_scratch_bytes: u64,
    pub output_staging_bytes: u64,
}

impl GpuAllocationBreakdown {
    fn from_counts(counts: [u64; GPU_ALLOCATION_CLASS_COUNT]) -> Self {
        Self {
            model_weight_bytes: counts[GpuAllocationClass::ModelWeights.index()],
            expert_weight_bytes: counts[GpuAllocationClass::ExpertWeights.index()],
            kv_cache_bytes: counts[GpuAllocationClass::KvCache.index()],
            scratch_bytes: counts[GpuAllocationClass::Scratch.index()],
            staging_bytes: counts[GpuAllocationClass::Staging.index()],
            recurrent_state_bytes: counts[GpuAllocationClass::RecurrentState.index()],
            graph_bytes: counts[GpuAllocationClass::Graph.index()],
            image_component_weight_bytes: counts[GpuAllocationClass::ImageComponentWeights.index()],
            prompt_embedding_bytes: counts[GpuAllocationClass::PromptEmbeddings.index()],
            image_conditioning_bytes: counts[GpuAllocationClass::ImageConditioning.index()],
            latent_state_bytes: counts[GpuAllocationClass::LatentState.index()],
            denoiser_persistent_scratch_bytes: counts
                [GpuAllocationClass::DenoiserPersistentScratch.index()],
            denoiser_transient_scratch_bytes: counts
                [GpuAllocationClass::DenoiserTransientScratch.index()],
            vae_scratch_bytes: counts[GpuAllocationClass::VaeScratch.index()],
            preview_scratch_bytes: counts[GpuAllocationClass::PreviewScratch.index()],
            output_staging_bytes: counts[GpuAllocationClass::OutputStaging.index()],
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuAllocationSnapshot {
    pub budget_bytes: Option<u64>,
    pub allocated_bytes: u64,
    pub peak_allocated_bytes: u64,
    pub by_class: GpuAllocationBreakdown,
}

#[derive(Debug, Default)]
struct GpuAllocationState {
    budget_bytes: Option<u64>,
    allocated_bytes: u64,
    peak_allocated_bytes: u64,
    by_class: [u64; GPU_ALLOCATION_CLASS_COUNT],
}

/// Central, fallible GPU allocation-admission boundary.
///
/// A lease is acquired before the corresponding CUDA allocation and is held
/// for exactly as long as that allocation. If CUDA construction fails, normal
/// Rust drop releases the reservation, preventing partial-resource leaks.
#[derive(Debug, Default)]
pub struct GpuAllocationArena {
    state: Arc<Mutex<GpuAllocationState>>,
}

impl GpuAllocationArena {
    pub fn configure_budget(&self, budget_bytes: u64) -> Result<()> {
        if budget_bytes == 0 {
            return Err(XrtError::Cuda(
                "GPU allocation arena budget must be greater than zero".to_string(),
            ));
        }
        let mut state = self.state.lock();
        match state.budget_bytes {
            None => {
                state.budget_bytes = Some(budget_bytes);
                Ok(())
            }
            Some(existing) if existing == budget_bytes => Ok(()),
            Some(existing) => Err(XrtError::Cuda(format!(
                "GPU allocation arena is already configured for {existing} bytes and cannot be reconfigured to {budget_bytes} bytes"
            ))),
        }
    }

    pub fn reserve(&self, class: GpuAllocationClass, bytes: u64) -> Result<GpuAllocationLease> {
        let mut state = self.state.lock();
        let budget_bytes = state.budget_bytes.ok_or_else(|| {
            XrtError::Cuda(
                "GPU allocation arena must be configured before reserving memory".to_string(),
            )
        })?;
        let next_total = state.allocated_bytes.checked_add(bytes).ok_or_else(|| {
            XrtError::Cuda("GPU allocation reservation byte count overflowed".to_string())
        })?;
        if next_total > budget_bytes {
            return Err(XrtError::Cuda(format!(
                "GPU allocation reservation requires {bytes} bytes ({next_total} total), exceeding the configured {budget_bytes}-byte budget"
            )));
        }
        let class_index = class.index();
        let next_class = state.by_class[class_index]
            .checked_add(bytes)
            .ok_or_else(|| {
                XrtError::Cuda("GPU allocation class byte count overflowed".to_string())
            })?;
        state.allocated_bytes = next_total;
        state.peak_allocated_bytes = state.peak_allocated_bytes.max(next_total);
        state.by_class[class_index] = next_class;
        drop(state);

        Ok(GpuAllocationLease {
            reservation: Arc::new(GpuAllocationReservation {
                state: Arc::clone(&self.state),
                class,
                bytes,
            }),
        })
    }

    pub fn snapshot(&self) -> GpuAllocationSnapshot {
        let state = self.state.lock();
        GpuAllocationSnapshot {
            budget_bytes: state.budget_bytes,
            allocated_bytes: state.allocated_bytes,
            peak_allocated_bytes: state.peak_allocated_bytes,
            by_class: GpuAllocationBreakdown::from_counts(state.by_class),
        }
    }
}

#[derive(Clone, Debug)]
pub struct GpuAllocationLease {
    reservation: Arc<GpuAllocationReservation>,
}

#[derive(Debug)]
struct GpuAllocationReservation {
    state: Arc<Mutex<GpuAllocationState>>,
    class: GpuAllocationClass,
    bytes: u64,
}

impl GpuAllocationLease {
    pub fn bytes(&self) -> u64 {
        self.reservation.bytes
    }

    pub fn class(&self) -> GpuAllocationClass {
        self.reservation.class
    }

    pub fn release(self) {
        drop(self);
    }
}

impl Drop for GpuAllocationReservation {
    fn drop(&mut self) {
        let mut state = self.state.lock();
        let class_bytes = &mut state.by_class[self.class.index()];
        debug_assert!(*class_bytes >= self.bytes);
        *class_bytes = class_bytes.saturating_sub(self.bytes);
        state.allocated_bytes = state.allocated_bytes.saturating_sub(self.bytes);
    }
}

impl GpuTransferStats {
    pub fn saturating_sub(&self, earlier: &Self) -> Self {
        Self {
            host_to_device_calls: self
                .host_to_device_calls
                .saturating_sub(earlier.host_to_device_calls),
            host_to_device_bytes: self
                .host_to_device_bytes
                .saturating_sub(earlier.host_to_device_bytes),
            device_to_host_calls: self
                .device_to_host_calls
                .saturating_sub(earlier.device_to_host_calls),
            device_to_host_bytes: self
                .device_to_host_bytes
                .saturating_sub(earlier.device_to_host_bytes),
            device_to_device_calls: self
                .device_to_device_calls
                .saturating_sub(earlier.device_to_device_calls),
            device_to_device_bytes: self
                .device_to_device_bytes
                .saturating_sub(earlier.device_to_device_bytes),
        }
    }
}

impl From<CudaTransferStats> for GpuTransferStats {
    fn from(stats: CudaTransferStats) -> Self {
        Self {
            host_to_device_calls: stats.host_to_device_calls,
            host_to_device_bytes: stats.host_to_device_bytes,
            device_to_host_calls: stats.device_to_host_calls,
            device_to_host_bytes: stats.device_to_host_bytes,
            device_to_device_calls: stats.device_to_device_calls,
            device_to_device_bytes: stats.device_to_device_bytes,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuAllocationStats {
    pub current_bytes: u64,
    pub peak_bytes: u64,
    pub allocation_calls: u64,
    pub total_allocated_bytes: u64,
}

impl From<CudaAllocationStats> for GpuAllocationStats {
    fn from(stats: CudaAllocationStats) -> Self {
        Self {
            current_bytes: stats.current_bytes,
            peak_bytes: stats.peak_bytes,
            allocation_calls: stats.allocation_calls,
            total_allocated_bytes: stats.total_allocated_bytes,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuMemoryPoolStats {
    pub release_threshold_bytes: u64,
    pub reserved_current_bytes: u64,
    pub reserved_peak_bytes: u64,
    pub used_current_bytes: u64,
    pub used_peak_bytes: u64,
}

impl From<CudaMemoryPoolStats> for GpuMemoryPoolStats {
    fn from(stats: CudaMemoryPoolStats) -> Self {
        Self {
            release_threshold_bytes: stats.release_threshold_bytes,
            reserved_current_bytes: stats.reserved_current_bytes,
            reserved_peak_bytes: stats.reserved_peak_bytes,
            used_current_bytes: stats.used_current_bytes,
            used_peak_bytes: stats.used_peak_bytes,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuAllocationDelta {
    pub baseline_bytes: u64,
    pub final_bytes: u64,
    pub peak_bytes: u64,
    pub allocation_calls: u64,
    pub allocated_bytes: u64,
}

impl GpuAllocationDelta {
    pub fn between(before: &GpuAllocationStats, after: &GpuAllocationStats) -> Self {
        Self {
            baseline_bytes: before.current_bytes,
            final_bytes: after.current_bytes,
            peak_bytes: after.peak_bytes.max(before.current_bytes),
            allocation_calls: after
                .allocation_calls
                .saturating_sub(before.allocation_calls),
            allocated_bytes: after
                .total_allocated_bytes
                .saturating_sub(before.total_allocated_bytes),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuResourceStatus {
    pub cuda_feature_enabled: bool,
    pub cuda_available: bool,
    pub device_ordinal: usize,
    pub device_name: Option<String>,
    pub free_vram_bytes: Option<u64>,
    pub total_vram_bytes: Option<u64>,
    pub device_used_vram_bytes: Option<u64>,
    pub memory_fraction: f32,
    pub reserved_vram_bytes: u64,
    pub kv_fraction: f32,
    pub kv_budget_bytes: Option<u64>,
    pub model_weight_bytes: u64,
    pub kv_allocated_bytes: u64,
    pub requested_kv_cache_mode: Option<&'static str>,
    pub kv_cache_mode: Option<&'static str>,
    pub scratch_allocated_bytes: u64,
    pub staging_allocated_bytes: u64,
    pub tracked_allocated_bytes: u64,
    pub arena_budget_bytes: Option<u64>,
    pub arena_allocated_bytes: u64,
    pub arena_peak_allocated_bytes: u64,
    pub arena_allocations: GpuAllocationBreakdown,
    pub transfer_totals: Option<GpuTransferStats>,
    pub allocation_totals: Option<GpuAllocationStats>,
    pub memory_pool: Option<GpuMemoryPoolStats>,
    pub active_sessions: usize,
    pub cuda_graph_mode: &'static str,
    pub graph_capture: &'static str,
    pub resident_f32_probe_available: bool,
    pub resident_q8_0_probe_available: bool,
    pub resident_q8_0_layer0_probe_available: bool,
    pub resident_dense_quant_decode_available: bool,
    pub note: &'static str,
}

#[derive(Debug, Clone)]
pub struct GpuResourceManager {
    config: GpuResourceConfig,
    allocation_arena: Arc<GpuAllocationArena>,
}

impl GpuResourceManager {
    pub fn new(config: GpuResourceConfig) -> Self {
        Self {
            config,
            allocation_arena: Arc::new(GpuAllocationArena::default()),
        }
    }

    pub fn from_env() -> Self {
        Self::new(GpuResourceConfig::from_env())
    }

    pub fn config(&self) -> GpuResourceConfig {
        self.config
    }

    pub fn validate_compatible_config(&self, expected: GpuResourceConfig) -> Result<()> {
        if self.config != expected {
            return Err(XrtError::Cuda(format!(
                "shared GPU resource manager configuration mismatch: manager={:?}, requested={expected:?}",
                self.config
            )));
        }
        Ok(())
    }

    pub fn allocation_arena(&self) -> Arc<GpuAllocationArena> {
        Arc::clone(&self.allocation_arena)
    }

    pub fn status(&self) -> GpuResourceStatus {
        self.status_with_allocations(0, 0, 0, 0, false)
    }

    pub fn status_with_allocations(
        &self,
        model_weight_bytes: u64,
        kv_allocated_bytes: u64,
        scratch_allocated_bytes: u64,
        active_sessions: usize,
        cuda_available: bool,
    ) -> GpuResourceStatus {
        self.status_with_allocations_and_probe(
            model_weight_bytes,
            kv_allocated_bytes,
            scratch_allocated_bytes,
            active_sessions,
            cuda_available,
            false,
            false,
            false,
            false,
        )
    }

    pub fn status_with_allocations_and_probe(
        &self,
        model_weight_bytes: u64,
        kv_allocated_bytes: u64,
        scratch_allocated_bytes: u64,
        active_sessions: usize,
        cuda_available: bool,
        resident_f32_probe_available: bool,
        resident_q8_0_probe_available: bool,
        resident_q8_0_layer0_probe_available: bool,
        resident_dense_quant_decode_available: bool,
    ) -> GpuResourceStatus {
        self.status_with_allocations_staging_and_probe(
            model_weight_bytes,
            kv_allocated_bytes,
            scratch_allocated_bytes,
            0,
            active_sessions,
            cuda_available,
            resident_f32_probe_available,
            resident_q8_0_probe_available,
            resident_q8_0_layer0_probe_available,
            resident_dense_quant_decode_available,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn status_with_allocations_staging_and_probe(
        &self,
        model_weight_bytes: u64,
        kv_allocated_bytes: u64,
        scratch_allocated_bytes: u64,
        staging_allocated_bytes: u64,
        active_sessions: usize,
        cuda_available: bool,
        resident_f32_probe_available: bool,
        resident_q8_0_probe_available: bool,
        resident_q8_0_layer0_probe_available: bool,
        resident_dense_quant_decode_available: bool,
    ) -> GpuResourceStatus {
        let graph_capture = match self.config.cuda_graph_mode {
            CudaGraphMode::Disabled => "disabled",
            CudaGraphMode::Enabled | CudaGraphMode::Auto if cuda_available => "not-captured",
            CudaGraphMode::Enabled | CudaGraphMode::Auto => "inactive",
        };
        let explicit_allocated_bytes = model_weight_bytes
            .saturating_add(kv_allocated_bytes)
            .saturating_add(scratch_allocated_bytes)
            .saturating_add(staging_allocated_bytes);
        let arena = self.allocation_arena.snapshot();
        GpuResourceStatus {
            cuda_feature_enabled: cfg!(feature = "cuda"),
            cuda_available,
            device_ordinal: self.config.device_ordinal,
            device_name: None,
            free_vram_bytes: None,
            total_vram_bytes: None,
            device_used_vram_bytes: None,
            memory_fraction: self.config.memory_fraction,
            reserved_vram_bytes: self.config.reserved_bytes(),
            kv_fraction: self.config.kv_fraction,
            kv_budget_bytes: None,
            model_weight_bytes,
            kv_allocated_bytes,
            requested_kv_cache_mode: None,
            kv_cache_mode: None,
            scratch_allocated_bytes,
            staging_allocated_bytes,
            tracked_allocated_bytes: explicit_allocated_bytes.max(arena.allocated_bytes),
            arena_budget_bytes: arena.budget_bytes,
            arena_allocated_bytes: arena.allocated_bytes,
            arena_peak_allocated_bytes: arena.peak_allocated_bytes,
            arena_allocations: arena.by_class,
            transfer_totals: None,
            allocation_totals: None,
            memory_pool: None,
            active_sessions,
            cuda_graph_mode: self.config.cuda_graph_mode.as_str(),
            graph_capture,
            resident_f32_probe_available,
            resident_q8_0_probe_available,
            resident_q8_0_layer0_probe_available,
            resident_dense_quant_decode_available,
            note: "GPU resource status separates requested CUDA Graph mode from observed capture state; not-captured and inactive do not claim graph replay",
        }
    }
}

fn parse_usize(value: Option<&str>) -> Option<usize> {
    value?.trim().parse().ok()
}

fn parse_u64(value: Option<&str>) -> Option<u64> {
    value?.trim().parse().ok()
}

fn parse_fraction(value: Option<&str>) -> Option<f32> {
    let parsed = value?.trim().parse::<f32>().ok()?;
    parsed.is_finite().then(|| parsed.clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::{
        CudaGraphMode, GpuAllocationArena, GpuAllocationClass, GpuAllocationDelta,
        GpuAllocationStats, GpuResourceConfig, GpuResourceManager, GpuTransferStats,
    };
    use std::sync::Arc;

    #[test]
    fn uses_safe_defaults_for_missing_values() {
        assert_eq!(
            GpuResourceConfig::from_values(None, None, None, None, None),
            GpuResourceConfig::default()
        );
    }

    #[test]
    fn parses_and_clamps_fraction_values() {
        let config = GpuResourceConfig::from_values(
            Some("2"),
            Some("1.5"),
            Some("2048"),
            Some("-0.1"),
            Some("1"),
        );
        assert_eq!(config.device_ordinal, 2);
        assert_eq!(config.memory_fraction, 1.0);
        assert_eq!(config.reserved_mb, 2048);
        assert_eq!(config.kv_fraction, 0.0);
        assert_eq!(config.cuda_graph_mode, CudaGraphMode::Enabled);
    }

    #[test]
    fn invalid_values_fall_back_to_safe_defaults() {
        let default = GpuResourceConfig::default();
        let config = GpuResourceConfig::from_values(
            Some("bad"),
            Some("NaN"),
            Some("nope"),
            Some("inf"),
            Some("sometimes"),
        );
        assert_eq!(config, default);
    }

    #[test]
    fn parses_all_cuda_graph_modes() {
        for (value, expected) in [
            ("0", CudaGraphMode::Disabled),
            ("off", CudaGraphMode::Disabled),
            ("1", CudaGraphMode::Enabled),
            ("on", CudaGraphMode::Enabled),
            ("auto", CudaGraphMode::Auto),
        ] {
            let config = GpuResourceConfig::from_values(None, None, None, None, Some(value));
            assert_eq!(config.cuda_graph_mode, expected);
        }
    }

    #[test]
    fn runtime_level_status_has_no_session_cache_mode() {
        let manager = GpuResourceManager {
            config: GpuResourceConfig::default(),
            allocation_arena: Arc::new(GpuAllocationArena::default()),
        };
        let status = manager.status();
        assert_eq!(status.requested_kv_cache_mode, None);
        assert_eq!(status.kv_cache_mode, None);
        assert_eq!(status.kv_budget_bytes, None);
        assert_eq!(status.free_vram_bytes, None);
        assert_eq!(status.device_used_vram_bytes, None);
        assert_eq!(status.tracked_allocated_bytes, 0);
        assert_eq!(status.arena_budget_bytes, None);
        assert_eq!(status.arena_allocated_bytes, 0);
        assert_eq!(status.transfer_totals, None);
        assert_eq!(status.allocation_totals, None);
        assert_eq!(status.cuda_graph_mode, "auto");
        assert_eq!(status.graph_capture, "inactive");

        let cuda_status = manager.status_with_allocations(0, 0, 0, 0, true);
        assert_eq!(cuda_status.graph_capture, "not-captured");

        let allocated_status = manager.status_with_allocations(10, 20, 30, 1, true);
        assert_eq!(allocated_status.tracked_allocated_bytes, 60);
    }

    #[test]
    fn central_allocation_leases_enforce_budget_and_roll_back_on_drop() {
        let arena = GpuAllocationArena::default();
        arena.configure_budget(100).unwrap();
        arena.configure_budget(100).unwrap();
        assert!(arena.configure_budget(101).is_err());

        let model = arena.reserve(GpuAllocationClass::ModelWeights, 60).unwrap();
        let expert = arena
            .reserve(GpuAllocationClass::ExpertWeights, 40)
            .unwrap();
        assert!(arena.reserve(GpuAllocationClass::Scratch, 1).is_err());
        let full = arena.snapshot();
        assert_eq!(full.allocated_bytes, 100);
        assert_eq!(full.peak_allocated_bytes, 100);
        assert_eq!(full.by_class.model_weight_bytes, 60);
        assert_eq!(full.by_class.expert_weight_bytes, 40);

        drop(expert);
        let after_drop = arena.snapshot();
        assert_eq!(after_drop.allocated_bytes, 60);
        assert_eq!(after_drop.peak_allocated_bytes, 100);
        assert_eq!(after_drop.by_class.expert_weight_bytes, 0);
        drop(model);
        assert_eq!(arena.snapshot().allocated_bytes, 0);
    }

    #[test]
    fn image_allocation_classes_are_independently_accounted() {
        let arena = GpuAllocationArena::default();
        arena.configure_budget(1_000).unwrap();
        let weights = arena
            .reserve(GpuAllocationClass::ImageComponentWeights, 400)
            .unwrap();
        let latent = arena.reserve(GpuAllocationClass::LatentState, 100).unwrap();
        let scratch = arena
            .reserve(GpuAllocationClass::DenoiserPersistentScratch, 200)
            .unwrap();
        let snapshot = arena.snapshot();
        assert_eq!(snapshot.allocated_bytes, 700);
        assert_eq!(snapshot.by_class.image_component_weight_bytes, 400);
        assert_eq!(snapshot.by_class.latent_state_bytes, 100);
        assert_eq!(snapshot.by_class.denoiser_persistent_scratch_bytes, 200);
        drop((weights, latent, scratch));
        assert_eq!(arena.snapshot().allocated_bytes, 0);
    }

    #[test]
    fn shared_manager_rejects_incompatible_device_or_budget_config() {
        let manager = GpuResourceManager::new(GpuResourceConfig::default());
        assert!(manager
            .validate_compatible_config(GpuResourceConfig::default())
            .is_ok());
        let mut wrong_device = GpuResourceConfig::default();
        wrong_device.device_ordinal = 1;
        assert!(manager.validate_compatible_config(wrong_device).is_err());
        let mut wrong_budget = GpuResourceConfig::default();
        wrong_budget.reserved_mb += 1;
        assert!(manager.validate_compatible_config(wrong_budget).is_err());
    }

    #[test]
    fn zero_byte_reservations_are_tracked_without_underflow() {
        let arena = GpuAllocationArena::default();
        arena.configure_budget(1).unwrap();
        let lease = arena.reserve(GpuAllocationClass::Graph, 0).unwrap();
        assert_eq!(lease.bytes(), 0);
        drop(lease);
        assert_eq!(arena.snapshot().allocated_bytes, 0);
    }

    #[test]
    fn transfer_stats_delta_is_componentwise_and_saturating() {
        let before = GpuTransferStats {
            host_to_device_calls: 2,
            host_to_device_bytes: 20,
            device_to_host_calls: 3,
            device_to_host_bytes: 30,
            device_to_device_calls: 4,
            device_to_device_bytes: 40,
        };
        let after = GpuTransferStats {
            host_to_device_calls: 7,
            host_to_device_bytes: 50,
            device_to_host_calls: 1,
            device_to_host_bytes: 35,
            device_to_device_calls: 6,
            device_to_device_bytes: 10,
        };

        assert_eq!(
            after.saturating_sub(&before),
            GpuTransferStats {
                host_to_device_calls: 5,
                host_to_device_bytes: 30,
                device_to_host_calls: 0,
                device_to_host_bytes: 5,
                device_to_device_calls: 2,
                device_to_device_bytes: 0,
            }
        );
    }

    #[test]
    fn allocation_delta_preserves_baseline_final_and_interval_peak() {
        let before = GpuAllocationStats {
            current_bytes: 100,
            peak_bytes: 150,
            allocation_calls: 4,
            total_allocated_bytes: 300,
        };
        let after = GpuAllocationStats {
            current_bytes: 100,
            peak_bytes: 180,
            allocation_calls: 7,
            total_allocated_bytes: 380,
        };

        assert_eq!(
            GpuAllocationDelta::between(&before, &after),
            GpuAllocationDelta {
                baseline_bytes: 100,
                final_bytes: 100,
                peak_bytes: 180,
                allocation_calls: 3,
                allocated_bytes: 80,
            }
        );
    }
}
