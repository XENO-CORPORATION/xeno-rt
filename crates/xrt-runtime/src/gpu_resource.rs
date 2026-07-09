use std::env;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GpuResourceConfig {
    pub device_ordinal: usize,
    pub memory_fraction: f32,
    pub reserved_mb: u64,
    pub kv_fraction: f32,
}

impl Default for GpuResourceConfig {
    fn default() -> Self {
        Self {
            device_ordinal: 0,
            memory_fraction: 0.90,
            reserved_mb: 1024,
            kv_fraction: 0.30,
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
        )
    }

    fn from_values(
        device_ordinal: Option<&str>,
        memory_fraction: Option<&str>,
        reserved_mb: Option<&str>,
        kv_fraction: Option<&str>,
    ) -> Self {
        let default = Self::default();
        Self {
            device_ordinal: parse_usize(device_ordinal).unwrap_or(default.device_ordinal),
            memory_fraction: parse_fraction(memory_fraction).unwrap_or(default.memory_fraction),
            reserved_mb: parse_u64(reserved_mb).unwrap_or(default.reserved_mb),
            kv_fraction: parse_fraction(kv_fraction).unwrap_or(default.kv_fraction),
        }
    }

    pub fn reserved_bytes(&self) -> u64 {
        self.reserved_mb.saturating_mul(1024 * 1024)
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
    pub memory_fraction: f32,
    pub reserved_vram_bytes: u64,
    pub kv_fraction: f32,
    pub kv_budget_bytes: Option<u64>,
    pub model_weight_bytes: u64,
    pub kv_allocated_bytes: u64,
    pub requested_kv_cache_mode: Option<&'static str>,
    pub kv_cache_mode: Option<&'static str>,
    pub scratch_allocated_bytes: u64,
    pub active_sessions: usize,
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
}

impl GpuResourceManager {
    pub fn from_env() -> Self {
        Self {
            config: GpuResourceConfig::from_env(),
        }
    }

    pub fn config(&self) -> GpuResourceConfig {
        self.config
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
        GpuResourceStatus {
            cuda_feature_enabled: cfg!(feature = "cuda"),
            cuda_available,
            device_ordinal: self.config.device_ordinal,
            device_name: None,
            free_vram_bytes: None,
            total_vram_bytes: None,
            memory_fraction: self.config.memory_fraction,
            reserved_vram_bytes: self.config.reserved_bytes(),
            kv_fraction: self.config.kv_fraction,
            kv_budget_bytes: None,
            model_weight_bytes,
            kv_allocated_bytes,
            requested_kv_cache_mode: None,
            kv_cache_mode: None,
            scratch_allocated_bytes,
            active_sessions,
            graph_capture: "unavailable",
            resident_f32_probe_available,
            resident_q8_0_probe_available,
            resident_q8_0_layer0_probe_available,
            resident_dense_quant_decode_available,
            note: "GPU resource status reports configured and allocated runtime resources; CUDA device telemetry is populated when a CUDA backend is active; graph capture is not wired yet",
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
    use super::{GpuResourceConfig, GpuResourceManager};

    #[test]
    fn uses_safe_defaults_for_missing_values() {
        assert_eq!(
            GpuResourceConfig::from_values(None, None, None, None),
            GpuResourceConfig::default()
        );
    }

    #[test]
    fn parses_and_clamps_fraction_values() {
        let config =
            GpuResourceConfig::from_values(Some("2"), Some("1.5"), Some("2048"), Some("-0.1"));
        assert_eq!(config.device_ordinal, 2);
        assert_eq!(config.memory_fraction, 1.0);
        assert_eq!(config.reserved_mb, 2048);
        assert_eq!(config.kv_fraction, 0.0);
    }

    #[test]
    fn invalid_values_fall_back_to_safe_defaults() {
        let default = GpuResourceConfig::default();
        let config =
            GpuResourceConfig::from_values(Some("bad"), Some("NaN"), Some("nope"), Some("inf"));
        assert_eq!(config, default);
    }

    #[test]
    fn runtime_level_status_has_no_session_cache_mode() {
        let status = GpuResourceManager::from_env().status();
        assert_eq!(status.requested_kv_cache_mode, None);
        assert_eq!(status.kv_cache_mode, None);
        assert_eq!(status.kv_budget_bytes, None);
        assert_eq!(status.free_vram_bytes, None);
    }
}
