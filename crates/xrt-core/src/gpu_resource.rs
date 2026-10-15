//! Shared allocation admission for every runtime modality.
use crate::{Result, XrtError};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

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

    /// Establish a process/device budget once, never recompute it from the
    /// remaining free memory after another modality already allocated. The
    /// initial conservative budget remains the ceiling for the server lifetime;
    /// every caller must ALSO preflight its new allocation against live VRAM.
    /// Atomic under the arena lock, unlike snapshot()+configure_budget().
    pub fn initialize_budget(&self, available_budget: u64) -> Result<u64> {
        let mut state = self.state.lock();
        if let Some(existing) = state.budget_bytes {
            return Ok(existing);
        }
        if available_budget == 0 {
            return Err(XrtError::Cuda("no device budget available".into()));
        }
        state.budget_bytes = Some(available_budget);
        Ok(available_budget)
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
