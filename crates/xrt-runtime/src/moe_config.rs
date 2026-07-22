use std::env;
use std::path::PathBuf;

use serde::Serialize;
use xrt_core::{Result, XrtError};
use xrt_kernels::{CpuThreadBudget, CpuTopology, NumaPolicy};
use xrt_models::MoeCpuExecution;

use crate::BackendKind;

const DEFAULT_PLACEMENT_UPDATE_TOKENS: u64 = 1024;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MoeAcceleration {
    #[default]
    Legacy,
    Auto,
    Cpu,
    Hybrid,
    Gpu,
}

impl MoeAcceleration {
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "legacy" => Ok(Self::Legacy),
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "hybrid" => Ok(Self::Hybrid),
            "gpu" => Ok(Self::Gpu),
            other => Err(XrtError::Runtime(format!(
                "invalid XRT_MOE_ACCELERATION value {other:?}; expected legacy, auto, cpu, hybrid, or gpu"
            ))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Hybrid => "hybrid",
            Self::Gpu => "gpu",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MoePlacementPolicy {
    #[default]
    Uniform,
    Profiled,
    Adaptive,
}

impl MoePlacementPolicy {
    fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "uniform" => Ok(Self::Uniform),
            "profiled" => Ok(Self::Profiled),
            "adaptive" => Ok(Self::Adaptive),
            other => Err(XrtError::Runtime(format!(
                "invalid XRT_MOE_PLACEMENT value {other:?}; expected uniform, profiled, or adaptive"
            ))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            Self::Profiled => "profiled",
            Self::Adaptive => "adaptive",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MoeRuntimeConfig {
    pub acceleration: MoeAcceleration,
    pub gpu_expert_budget_bytes: Option<u64>,
    pub placement: MoePlacementPolicy,
    pub placement_manifest: Option<PathBuf>,
    pub placement_update_tokens: u64,
    pub layerwise_prefill: bool,
    pub numa: NumaPolicy,
}

impl Default for MoeRuntimeConfig {
    fn default() -> Self {
        Self {
            acceleration: MoeAcceleration::Legacy,
            gpu_expert_budget_bytes: None,
            placement: MoePlacementPolicy::Uniform,
            placement_manifest: None,
            placement_update_tokens: DEFAULT_PLACEMENT_UPDATE_TOKENS,
            layerwise_prefill: false,
            numa: NumaPolicy::Auto,
        }
    }
}

impl MoeRuntimeConfig {
    pub fn from_env() -> Result<Self> {
        let mut config = Self::default();
        if let Ok(value) = env::var("XRT_MOE_ACCELERATION") {
            config.acceleration = MoeAcceleration::parse(&value)?;
        }
        if let Ok(value) = env::var("XRT_MOE_GPU_EXPERT_BUDGET_BYTES") {
            let budget = value.parse::<u64>().map_err(|_| {
                XrtError::Runtime(format!(
                    "XRT_MOE_GPU_EXPERT_BUDGET_BYTES must be a positive integer, received {value:?}"
                ))
            })?;
            if budget == 0 {
                return Err(XrtError::Runtime(
                    "XRT_MOE_GPU_EXPERT_BUDGET_BYTES must be greater than zero".to_string(),
                ));
            }
            config.gpu_expert_budget_bytes = Some(budget);
        }
        if let Ok(value) = env::var("XRT_MOE_PLACEMENT") {
            config.placement = MoePlacementPolicy::parse(&value)?;
        }
        if let Ok(value) = env::var("XRT_MOE_PLACEMENT_MANIFEST") {
            if value.trim().is_empty() {
                return Err(XrtError::Runtime(
                    "XRT_MOE_PLACEMENT_MANIFEST must not be empty".to_string(),
                ));
            }
            config.placement_manifest = Some(PathBuf::from(value));
        }
        if let Ok(value) = env::var("XRT_MOE_PLACEMENT_UPDATE_TOKENS") {
            let tokens = value.parse::<u64>().map_err(|_| {
                XrtError::Runtime(format!(
                    "XRT_MOE_PLACEMENT_UPDATE_TOKENS must be a positive integer, received {value:?}"
                ))
            })?;
            if tokens == 0 {
                return Err(XrtError::Runtime(
                    "XRT_MOE_PLACEMENT_UPDATE_TOKENS must be greater than zero".to_string(),
                ));
            }
            config.placement_update_tokens = tokens;
        }
        if let Ok(value) = env::var("XRT_MOE_LAYERWISE_PREFILL") {
            config.layerwise_prefill = parse_bool_flag("XRT_MOE_LAYERWISE_PREFILL", &value)?;
        }
        if let Ok(value) = env::var("XRT_MOE_NUMA") {
            config.numa = value.parse()?;
        }
        config.validate_shape()?;
        Ok(config)
    }

    pub fn optimized_cpu() -> Self {
        Self {
            acceleration: MoeAcceleration::Cpu,
            ..Self::default()
        }
    }

    pub(crate) fn model_cpu_execution(&self) -> MoeCpuExecution {
        match self.acceleration {
            MoeAcceleration::Cpu | MoeAcceleration::Hybrid => MoeCpuExecution::Optimized,
            _ => MoeCpuExecution::Legacy,
        }
    }

    pub(crate) fn resolve_backend(
        &self,
        requested: BackendKind,
        initially_active: BackendKind,
        is_moe: bool,
    ) -> Result<BackendKind> {
        self.validate_shape()?;
        if !is_moe {
            if matches!(
                self.acceleration,
                MoeAcceleration::Hybrid | MoeAcceleration::Gpu
            ) {
                return Err(XrtError::Unsupported(format!(
                    "MoE acceleration mode {} was requested for a non-MoE model",
                    self.acceleration.as_str()
                )));
            }
            return Ok(initially_active);
        }

        match requested {
            BackendKind::Cpu => match self.acceleration {
                MoeAcceleration::Legacy | MoeAcceleration::Auto | MoeAcceleration::Cpu => {
                    Ok(BackendKind::Cpu)
                }
                MoeAcceleration::Hybrid | MoeAcceleration::Gpu => {
                    Err(XrtError::Unsupported(format!(
                        "MoE acceleration mode {} requires CUDA and conflicts with the explicit CPU backend",
                        self.acceleration.as_str()
                    )))
                }
            },
            BackendKind::CudaResident => match self.acceleration {
                MoeAcceleration::Legacy | MoeAcceleration::Cpu => Err(XrtError::Unsupported(
                    format!(
                        "MoE acceleration mode {} conflicts with the explicit CUDA-resident backend",
                        self.acceleration.as_str()
                    ),
                )),
                MoeAcceleration::Auto => Err(XrtError::Unsupported(
                    "explicit CUDA MoE requires XRT_MOE_ACCELERATION=hybrid or gpu plus a GPU expert budget; auto is not eligible until the real-model performance gate passes"
                        .to_string(),
                )),
                MoeAcceleration::Hybrid | MoeAcceleration::Gpu => Ok(BackendKind::CudaResident),
            },
            BackendKind::Auto => match self.acceleration {
                MoeAcceleration::Legacy | MoeAcceleration::Cpu => Ok(BackendKind::Cpu),
                MoeAcceleration::Auto => Ok(initially_active),
                MoeAcceleration::Hybrid | MoeAcceleration::Gpu => Ok(BackendKind::CudaResident),
            },
            BackendKind::ExternalOpenAi => Err(XrtError::Unsupported(
                "local MoE configuration does not apply to external-openai".to_string(),
            )),
        }
    }

    pub(crate) fn status(
        &self,
        is_moe: bool,
        active_backend: BackendKind,
    ) -> Result<MoeRuntimeStatus> {
        let thread_budget = CpuThreadBudget::resolve_from_environment()?;
        let topology = CpuTopology::discover(self.numa)?;
        let (effective_mode, fallback_reason) = if !is_moe {
            ("not-applicable", None)
        } else {
            match (active_backend, self.acceleration) {
                (BackendKind::Cpu, MoeAcceleration::Cpu) => ("cpu", None),
                (BackendKind::Cpu, MoeAcceleration::Auto) => (
                    "legacy",
                    Some(
                        "auto remains on the legacy CPU path until hardware/model performance gates pass"
                            .to_string(),
                    ),
                ),
                (BackendKind::Cpu, _) => ("legacy", None),
                (BackendKind::CudaResident, MoeAcceleration::Auto) => (
                    "unavailable",
                    Some(
                        "CUDA MoE execution is not admitted by the current capability table"
                        .to_string(),
                    ),
                ),
                (BackendKind::CudaResident, MoeAcceleration::Hybrid) => ("hybrid", None),
                (BackendKind::CudaResident, MoeAcceleration::Gpu) => ("gpu", None),
                _ => ("unavailable", None),
            }
        };
        Ok(MoeRuntimeStatus {
            supported: is_moe,
            requested_mode: self.acceleration.as_str().to_string(),
            effective_mode: effective_mode.to_string(),
            exact: true,
            placement: self.placement.as_str().to_string(),
            placement_generation: 0,
            placement_manifest_sha256: None,
            gpu_expert_slots: 0,
            gpu_expert_bytes: 0,
            cpu_expert_calls: 0,
            gpu_expert_calls: 0,
            gpu_placement_hits: 0,
            gpu_placement_misses: 0,
            activation_d2h_bytes: 0,
            result_h2d_bytes: 0,
            coordinator_failures: 0,
            graph_eager_expert_calls: 0,
            graph_captures: 0,
            graph_replays: 0,
            graph_fallbacks: 0,
            placement_evaluations: 0,
            placement_updates: 0,
            placement_moves: 0,
            placement_upload_bytes: 0,
            placement_update_micros: 0,
            placement_last_update_micros: 0,
            layerwise_prefill_enabled: is_moe
                && active_backend == BackendKind::CudaResident
                && self.layerwise_prefill,
            layerwise_prefill_batches: 0,
            layerwise_prefill_tokens: 0,
            layerwise_prefill_weight_upload_bytes: 0,
            layerwise_prefill_repack_bytes: 0,
            layerwise_prefill_micros: 0,
            topology_nodes: topology.nodes().len(),
            logical_cpus: topology.logical_cpus(),
            affinity_supported: topology.affinity_supported(),
            thread_budget: thread_budget.total_threads(),
            thread_budget_source: thread_budget.source().to_string(),
            numa: numa_as_str(self.numa).to_string(),
            fallback_reason,
            routed_tokens: 0,
            selected_expert_calls: 0,
            legacy_batches: 0,
            grouped_batches: 0,
            grouped_tokens: 0,
            worker_failures: 0,
            expert_call_counts: Vec::new(),
        })
    }

    fn validate_shape(&self) -> Result<()> {
        let gpu_mode = matches!(
            self.acceleration,
            MoeAcceleration::Hybrid | MoeAcceleration::Gpu
        );
        if gpu_mode && self.gpu_expert_budget_bytes.is_none() {
            return Err(XrtError::Runtime(format!(
                "XRT_MOE_GPU_EXPERT_BUDGET_BYTES is required for explicit {} mode",
                self.acceleration.as_str()
            )));
        }
        if !gpu_mode && self.gpu_expert_budget_bytes.is_some() {
            return Err(XrtError::Runtime(format!(
                "a GPU expert budget is inapplicable to MoE acceleration mode {}",
                self.acceleration.as_str()
            )));
        }
        if self.placement == MoePlacementPolicy::Profiled && self.placement_manifest.is_none() {
            return Err(XrtError::Runtime(
                "profiled MoE placement requires XRT_MOE_PLACEMENT_MANIFEST".to_string(),
            ));
        }
        if self.placement != MoePlacementPolicy::Profiled && self.placement_manifest.is_some() {
            return Err(XrtError::Runtime(
                "a MoE placement manifest is valid only with profiled placement".to_string(),
            ));
        }
        if self.placement == MoePlacementPolicy::Adaptive
            && self.acceleration != MoeAcceleration::Hybrid
        {
            return Err(XrtError::Runtime(
                "adaptive MoE placement requires hybrid acceleration; full GPU mode already keeps every expert resident"
                .to_string(),
            ));
        }
        if self.layerwise_prefill && self.acceleration != MoeAcceleration::Hybrid {
            return Err(XrtError::Runtime(
                "layerwise MoE prefill requires hybrid acceleration and remains explicit opt-in"
                    .to_string(),
            ));
        }
        if !gpu_mode && self.placement != MoePlacementPolicy::Uniform {
            return Err(XrtError::Runtime(format!(
                "MoE placement policy {} is inapplicable to acceleration mode {}",
                self.placement.as_str(),
                self.acceleration.as_str()
            )));
        }
        if self.placement_update_tokens == 0 {
            return Err(XrtError::Runtime(
                "MoE placement update interval must be greater than zero".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct MoeRuntimeStatus {
    pub supported: bool,
    pub requested_mode: String,
    pub effective_mode: String,
    pub exact: bool,
    pub placement: String,
    pub placement_generation: u64,
    /// SHA-256 of the validated local placement manifest. Its path and routing
    /// frequency table are deliberately never exposed through runtime status.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub placement_manifest_sha256: Option<String>,
    pub gpu_expert_slots: usize,
    pub gpu_expert_bytes: u64,
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
    /// Total logical-expert replacements; this is the bounded churn counter.
    pub placement_moves: u64,
    pub placement_upload_bytes: u64,
    pub placement_update_micros: u64,
    pub placement_last_update_micros: u64,
    pub layerwise_prefill_enabled: bool,
    pub layerwise_prefill_batches: u64,
    pub layerwise_prefill_tokens: u64,
    pub layerwise_prefill_weight_upload_bytes: u64,
    pub layerwise_prefill_repack_bytes: u64,
    pub layerwise_prefill_micros: u64,
    pub topology_nodes: usize,
    pub logical_cpus: usize,
    pub affinity_supported: bool,
    pub thread_budget: usize,
    pub thread_budget_source: String,
    pub numa: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback_reason: Option<String>,
    pub routed_tokens: u64,
    pub selected_expert_calls: u64,
    pub legacy_batches: u64,
    pub grouped_batches: u64,
    pub grouped_tokens: u64,
    pub worker_failures: u64,
    /// Bounded by the loaded model's validated logical expert count.
    pub expert_call_counts: Vec<u64>,
}

fn numa_as_str(policy: NumaPolicy) -> &'static str {
    match policy {
        NumaPolicy::Auto => "auto",
        NumaPolicy::Off => "off",
        NumaPolicy::Strict => "strict",
    }
}

fn parse_bool_flag(name: &str, value: &str) -> Result<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "on" | "enabled" => Ok(true),
        "0" | "false" | "off" | "disabled" => Ok(false),
        _ => Err(XrtError::Runtime(format!(
            "{name} must be on/off, true/false, enabled/disabled, or 1/0; received {value:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_gpu_modes_require_a_positive_budget() {
        let mut config = MoeRuntimeConfig::default();
        config.acceleration = MoeAcceleration::Hybrid;
        assert!(config.validate_shape().is_err());
        config.gpu_expert_budget_bytes = Some(1);
        assert!(config.validate_shape().is_ok());
    }

    #[test]
    fn cpu_and_backend_conflicts_are_rejected_without_fallback() {
        let config = MoeRuntimeConfig::optimized_cpu();
        assert!(config
            .resolve_backend(BackendKind::CudaResident, BackendKind::CudaResident, true)
            .is_err());
        assert_eq!(
            config
                .resolve_backend(BackendKind::Auto, BackendKind::Auto, true)
                .unwrap(),
            BackendKind::Cpu
        );
    }

    #[test]
    fn explicit_gpu_mode_is_rejected_for_dense_models() {
        let config = MoeRuntimeConfig {
            acceleration: MoeAcceleration::Gpu,
            gpu_expert_budget_bytes: Some(1024),
            ..MoeRuntimeConfig::default()
        };
        assert!(config
            .resolve_backend(BackendKind::Auto, BackendKind::Auto, false)
            .is_err());
    }

    #[test]
    fn adaptive_placement_is_hybrid_only() {
        let adaptive_gpu = MoeRuntimeConfig {
            acceleration: MoeAcceleration::Gpu,
            gpu_expert_budget_bytes: Some(1024),
            placement: MoePlacementPolicy::Adaptive,
            ..MoeRuntimeConfig::default()
        };
        assert!(adaptive_gpu.validate_shape().is_err());

        let adaptive_hybrid = MoeRuntimeConfig {
            acceleration: MoeAcceleration::Hybrid,
            gpu_expert_budget_bytes: Some(1024),
            placement: MoePlacementPolicy::Adaptive,
            ..MoeRuntimeConfig::default()
        };
        assert!(adaptive_hybrid.validate_shape().is_ok());
    }

    #[test]
    fn layerwise_prefill_flag_accepts_documented_boolean_values() {
        for value in ["1", "true", "TRUE", "on", " enabled "] {
            assert!(
                parse_bool_flag("XRT_MOE_LAYERWISE_PREFILL", value).unwrap(),
                "{value:?} should enable layerwise prefill"
            );
        }
        for value in ["0", "false", "FALSE", "off", " disabled "] {
            assert!(
                !parse_bool_flag("XRT_MOE_LAYERWISE_PREFILL", value).unwrap(),
                "{value:?} should disable layerwise prefill"
            );
        }
        let error = parse_bool_flag("XRT_MOE_LAYERWISE_PREFILL", "sometimes")
            .expect_err("unknown boolean spellings must be rejected")
            .to_string();
        assert!(error.contains("XRT_MOE_LAYERWISE_PREFILL"));
        assert!(error.contains("on/off"));
    }

    #[test]
    fn layerwise_prefill_is_explicit_hybrid_only() {
        for acceleration in [
            MoeAcceleration::Legacy,
            MoeAcceleration::Auto,
            MoeAcceleration::Cpu,
            MoeAcceleration::Gpu,
        ] {
            let config = MoeRuntimeConfig {
                acceleration,
                gpu_expert_budget_bytes: (acceleration == MoeAcceleration::Gpu).then_some(1024),
                layerwise_prefill: true,
                ..MoeRuntimeConfig::default()
            };
            assert!(
                config.validate_shape().is_err(),
                "{} must reject layerwise prefill",
                acceleration.as_str()
            );
        }

        let hybrid = MoeRuntimeConfig {
            acceleration: MoeAcceleration::Hybrid,
            gpu_expert_budget_bytes: Some(1024),
            layerwise_prefill: true,
            ..MoeRuntimeConfig::default()
        };
        assert!(hybrid.validate_shape().is_ok());
    }

    #[test]
    fn explicit_cuda_moe_requires_an_explicit_gpu_tier_mode() {
        let auto = MoeRuntimeConfig {
            acceleration: MoeAcceleration::Auto,
            ..MoeRuntimeConfig::default()
        };
        assert!(auto
            .resolve_backend(BackendKind::CudaResident, BackendKind::CudaResident, true)
            .is_err());

        for acceleration in [MoeAcceleration::Hybrid, MoeAcceleration::Gpu] {
            let config = MoeRuntimeConfig {
                acceleration,
                gpu_expert_budget_bytes: Some(1024),
                ..MoeRuntimeConfig::default()
            };
            assert_eq!(
                config
                    .resolve_backend(BackendKind::CudaResident, BackendKind::CudaResident, true)
                    .unwrap(),
                BackendKind::CudaResident
            );
            assert_eq!(
                config
                    .resolve_backend(BackendKind::Auto, BackendKind::Cpu, true)
                    .unwrap(),
                BackendKind::CudaResident
            );
        }
    }
}
