use std::env;

use serde::{Deserialize, Serialize};

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
    pub tracked_allocated_bytes: u64,
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
        let graph_capture = match self.config.cuda_graph_mode {
            CudaGraphMode::Disabled => "disabled",
            CudaGraphMode::Enabled | CudaGraphMode::Auto if cuda_available => "not-captured",
            CudaGraphMode::Enabled | CudaGraphMode::Auto => "inactive",
        };
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
            tracked_allocated_bytes: model_weight_bytes
                .saturating_add(kv_allocated_bytes)
                .saturating_add(scratch_allocated_bytes),
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
    use super::{CudaGraphMode, GpuResourceConfig, GpuResourceManager};

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
        };
        let status = manager.status();
        assert_eq!(status.requested_kv_cache_mode, None);
        assert_eq!(status.kv_cache_mode, None);
        assert_eq!(status.kv_budget_bytes, None);
        assert_eq!(status.free_vram_bytes, None);
        assert_eq!(status.device_used_vram_bytes, None);
        assert_eq!(status.tracked_allocated_bytes, 0);
        assert_eq!(status.cuda_graph_mode, "auto");
        assert_eq!(status.graph_capture, "inactive");

        let cuda_status = manager.status_with_allocations(0, 0, 0, 0, true);
        assert_eq!(cuda_status.graph_capture, "not-captured");

        let allocated_status = manager.status_with_allocations(10, 20, 30, 1, true);
        assert_eq!(allocated_status.tracked_allocated_bytes, 60);
    }
}
