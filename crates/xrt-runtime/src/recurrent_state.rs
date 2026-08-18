use std::sync::Arc;

use xrt_core::{Result, XrtError};
use xrt_cuda::{CudaDeltaNetGeometry, CudaDevice, CudaF32Buffer};
use xrt_models::{DeltaNetStateDescriptor, DeltaNetStateSnapshot};

use crate::gpu_resource::{GpuAllocationArena, GpuAllocationClass, GpuAllocationLease};

#[derive(Debug)]
struct CudaDeltaNetLayerState {
    committed_conv: CudaF32Buffer,
    pending_conv: CudaF32Buffer,
    checkpoint_conv: CudaF32Buffer,
    verify_rebase_conv: CudaF32Buffer,
    committed_recurrent: CudaF32Buffer,
    pending_recurrent: CudaF32Buffer,
    checkpoint_recurrent: CudaF32Buffer,
    verify_rebase_recurrent: CudaF32Buffer,
    verify_rebase_capacity: usize,
    geometry: CudaDeltaNetGeometry,
}

#[derive(Debug)]
struct CudaVerifyWindowTransaction {
    rows: usize,
    tree_layout: bool,
    layer_advances: Vec<usize>,
}

/// Session-owned, transactional F32 DeltaNet state.
///
/// Kernels write only the pending buffers. The complete token becomes visible
/// through a handle swap after every recurrent/full-attention/FFN layer and
/// the final output projection have succeeded.
#[derive(Debug)]
pub struct CudaDeltaNetState {
    device: CudaDevice,
    descriptor: DeltaNetStateDescriptor,
    allocation_arena: Option<Arc<GpuAllocationArena>>,
    layers: Vec<Option<CudaDeltaNetLayerState>>,
    position: usize,
    transaction_active: bool,
    verify_window: Option<CudaVerifyWindowTransaction>,
    last_verify_window: Option<(usize, usize, bool)>,
    checkpoint_position: Option<usize>,
    committed_buffer_generation: u8,
    needs_zero: bool,
    allocated_bytes: u64,
    _allocation: Option<GpuAllocationLease>,
    _verify_rebase_allocations: Vec<GpuAllocationLease>,
}

impl CudaDeltaNetState {
    pub fn try_new(
        device: CudaDevice,
        descriptor: DeltaNetStateDescriptor,
        allocation_arena: Option<&Arc<GpuAllocationArena>>,
    ) -> Result<Self> {
        let geometry = CudaDeltaNetGeometry::new(
            descriptor.state_size(),
            descriptor.group_count(),
            descriptor.inner_size(),
            descriptor.dt_rank(),
            descriptor.conv_kernel(),
        )?;
        for (index, layer) in descriptor.layers().iter().enumerate() {
            if let Some(layer) = layer {
                if layer.conv_state_len() != geometry.conv_state_len()?
                    || layer.recurrent_state_len() != geometry.recurrent_state_len()?
                {
                    return Err(XrtError::Shape(format!(
                        "CUDA DeltaNet layer {index} state geometry does not match its descriptor"
                    )));
                }
            }
        }

        let allocated_elements = descriptor
            .allocated_f32_elements()?
            .checked_mul(3)
            .ok_or_else(|| {
                XrtError::Cuda(
                    "CUDA DeltaNet transactional state element count overflowed".to_string(),
                )
            })?;
        let allocated_bytes_usize = allocated_elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                XrtError::Cuda("CUDA DeltaNet state byte count overflowed".to_string())
            })?;
        let allocated_bytes = u64::try_from(allocated_bytes_usize).map_err(|_| {
            XrtError::Cuda(
                "CUDA DeltaNet state byte count cannot be represented as u64".to_string(),
            )
        })?;
        let allocation = allocation_arena
            .map(|arena| arena.reserve(GpuAllocationClass::RecurrentState, allocated_bytes))
            .transpose()?;

        let mut layers = Vec::new();
        layers
            .try_reserve_exact(descriptor.layers().len())
            .map_err(|error| {
                XrtError::Runtime(format!(
                    "failed to reserve CUDA DeltaNet layer descriptors: {error}"
                ))
            })?;
        for layer in descriptor.layers() {
            layers.push(match layer {
                Some(layer) => Some(CudaDeltaNetLayerState {
                    committed_conv: device.zeros_f32(layer.conv_state_len())?,
                    pending_conv: device.zeros_f32(layer.conv_state_len())?,
                    checkpoint_conv: device.zeros_f32(layer.conv_state_len())?,
                    verify_rebase_conv: device.zeros_f32(0)?,
                    committed_recurrent: device.zeros_f32(layer.recurrent_state_len())?,
                    pending_recurrent: device.zeros_f32(layer.recurrent_state_len())?,
                    checkpoint_recurrent: device.zeros_f32(layer.recurrent_state_len())?,
                    verify_rebase_recurrent: device.zeros_f32(0)?,
                    verify_rebase_capacity: 0,
                    geometry,
                }),
                None => None,
            });
        }

        Ok(Self {
            device,
            descriptor,
            allocation_arena: allocation_arena.cloned(),
            layers,
            position: 0,
            transaction_active: false,
            verify_window: None,
            last_verify_window: None,
            checkpoint_position: None,
            committed_buffer_generation: 0,
            needs_zero: false,
            allocated_bytes,
            _allocation: allocation,
            _verify_rebase_allocations: Vec::new(),
        })
    }

    fn ensure_verify_rebase_capacity(&mut self, capacity: usize) -> Result<()> {
        let current = self
            .layers
            .iter()
            .flatten()
            .map(|layer| layer.verify_rebase_capacity)
            .min()
            .unwrap_or(capacity);
        if current >= capacity {
            return Ok(());
        }
        let additional = capacity - current;
        let additional_elements = self
            .descriptor
            .allocated_f32_elements()?
            .checked_mul(additional)
            .ok_or_else(|| {
                XrtError::Cuda("CUDA DeltaNet verify-rebase element count overflowed".to_string())
            })?;
        let additional_bytes_usize = additional_elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                XrtError::Cuda("CUDA DeltaNet verify-rebase byte count overflowed".to_string())
            })?;
        let additional_bytes = u64::try_from(additional_bytes_usize).map_err(|_| {
            XrtError::Cuda(
                "CUDA DeltaNet verify-rebase byte count cannot be represented as u64".to_string(),
            )
        })?;
        let allocation = self
            .allocation_arena
            .as_ref()
            .map(|arena| arena.reserve(GpuAllocationClass::RecurrentState, additional_bytes))
            .transpose()?;

        let mut staged = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            staged.push(match layer {
                Some(layer) => Some((
                    self.device.zeros_f32(
                        capacity
                            .checked_mul(layer.committed_conv.len())
                            .ok_or_else(|| {
                                XrtError::Cuda(
                                    "CUDA DeltaNet verify convolution journal overflowed"
                                        .to_string(),
                                )
                            })?,
                    )?,
                    self.device.zeros_f32(
                        capacity
                            .checked_mul(layer.committed_recurrent.len())
                            .ok_or_else(|| {
                                XrtError::Cuda(
                                    "CUDA DeltaNet verify recurrent journal overflowed".to_string(),
                                )
                            })?,
                    )?,
                )),
                None => None,
            });
        }
        for (layer, buffers) in self.layers.iter_mut().zip(staged) {
            if let (Some(layer), Some((conv, recurrent))) = (layer.as_mut(), buffers) {
                layer.verify_rebase_conv = conv;
                layer.verify_rebase_recurrent = recurrent;
                layer.verify_rebase_capacity = capacity;
            }
        }
        self.allocated_bytes = self
            .allocated_bytes
            .checked_add(additional_bytes)
            .ok_or_else(|| {
                XrtError::Cuda("CUDA DeltaNet allocated byte count overflowed".to_string())
            })?;
        if let Some(allocation) = allocation {
            self._verify_rebase_allocations.push(allocation);
        }
        Ok(())
    }

    /// Reserves recurrent rollback journals before an adaptive verifier starts.
    ///
    /// Adaptive row counts may otherwise grow a partially populated journal by
    /// staging a complete replacement alongside it. On memory-constrained GPUs
    /// that transient double allocation can fail even though the final maximum
    /// journal fits.
    pub fn prepare_verify_rebase_capacity(&mut self, capacity: usize) -> Result<()> {
        if capacity > 15 {
            return Err(XrtError::Shape(format!(
                "CUDA DeltaNet verify-rebase capacity must not exceed 15, found {capacity}"
            )));
        }
        if self.transaction_active {
            return Err(XrtError::Runtime(
                "cannot reserve CUDA DeltaNet verify-rebase storage during an active transaction"
                    .to_string(),
            ));
        }
        self.ensure_verify_rebase_capacity(capacity)
    }

    pub fn descriptor(&self) -> &DeltaNetStateDescriptor {
        &self.descriptor
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn allocated_bytes(&self) -> u64 {
        self.allocated_bytes
    }

    /// Identifies which half of the transactional pair is currently committed.
    ///
    /// CUDA graph keys include this value so a graph captured against one
    /// committed/pending pointer orientation cannot replay after a handle swap.
    pub fn committed_buffer_generation(&self) -> u8 {
        self.committed_buffer_generation
    }

    /// Stable verifier graphs require every recurrent layer to use the fused
    /// path. Row-serial fallback mutates host-side buffer handles while the
    /// graph is being recorded and therefore cannot be safely replayed.
    pub fn fused_verify_graph_eligible(&self) -> bool {
        self.layers
            .iter()
            .flatten()
            .all(|layer| layer.geometry.state_size() == 128 && layer.geometry.history() <= 8)
    }

    pub fn fast_checkpoint_position(&self) -> Option<usize> {
        self.checkpoint_position
    }

    pub fn prepare(&mut self) -> Result<()> {
        if !self.needs_zero {
            return Ok(());
        }
        if self.transaction_active {
            return Err(XrtError::Runtime(
                "cannot prepare CUDA DeltaNet state during an active transaction".to_string(),
            ));
        }
        for layer in self.layers.iter_mut().flatten() {
            self.device.zero_f32(&mut layer.committed_conv)?;
            self.device.zero_f32(&mut layer.pending_conv)?;
            self.device.zero_f32(&mut layer.checkpoint_conv)?;
            self.device.zero_f32(&mut layer.verify_rebase_conv)?;
            self.device.zero_f32(&mut layer.committed_recurrent)?;
            self.device.zero_f32(&mut layer.pending_recurrent)?;
            self.device.zero_f32(&mut layer.checkpoint_recurrent)?;
            self.device.zero_f32(&mut layer.verify_rebase_recurrent)?;
        }
        self.needs_zero = false;
        Ok(())
    }

    /// Infallible public-reset half: no allocation or synchronization occurs.
    /// The next fallible pre-token preparation securely zeroes the retained
    /// buffers on their owning CUDA stream before reuse.
    pub fn logical_reset(&mut self) {
        self.position = 0;
        self.transaction_active = false;
        self.verify_window = None;
        self.last_verify_window = None;
        self.checkpoint_position = None;
        self.needs_zero = true;
    }

    /// Copies the accepted recurrent boundary into a persistent device-local
    /// journal. The journal may span multiple committed token transactions.
    pub fn begin_fast_checkpoint(&mut self, expected_position: usize) -> Result<()> {
        self.prepare()?;
        if self.transaction_active {
            return Err(XrtError::Runtime(
                "cannot checkpoint CUDA DeltaNet state during an active token transaction"
                    .to_string(),
            ));
        }
        if self.last_verify_window.is_some() {
            return Err(XrtError::Runtime(
                "cannot checkpoint CUDA DeltaNet state before publishing the prior verify boundary"
                    .to_string(),
            ));
        }
        if let Some(position) = self.checkpoint_position {
            return Err(XrtError::Runtime(format!(
                "CUDA DeltaNet fast checkpoint is already active at position {position}"
            )));
        }
        if self.position != expected_position {
            return Err(XrtError::Runtime(format!(
                "CUDA DeltaNet checkpoint position mismatch: expected {expected_position}, found {}",
                self.position
            )));
        }
        for layer in self.layers.iter_mut().flatten() {
            self.device
                .copy_f32_device(&layer.committed_conv, &mut layer.checkpoint_conv)?;
            self.device
                .copy_f32_device(&layer.committed_recurrent, &mut layer.checkpoint_recurrent)?;
        }
        self.checkpoint_position = Some(expected_position);
        Ok(())
    }

    /// Discards the device-local journal while retaining the newly committed
    /// state after a fully accepted speculative sequence.
    pub fn commit_fast_checkpoint(&mut self) -> Result<()> {
        if self.transaction_active {
            return Err(XrtError::Runtime(
                "cannot commit a CUDA DeltaNet checkpoint during an active token transaction"
                    .to_string(),
            ));
        }
        if self.last_verify_window.is_some() {
            return Err(XrtError::Runtime(
                "cannot commit a CUDA DeltaNet checkpoint before publishing its verify boundary"
                    .to_string(),
            ));
        }
        if self.checkpoint_position.take().is_none() {
            return Err(XrtError::Runtime(
                "CUDA DeltaNet fast checkpoint is not active".to_string(),
            ));
        }
        Ok(())
    }

    /// Restores the device-local journal into the currently committed handles.
    /// All copies are ordered on the owning device stream before later kernels.
    pub fn rollback_fast_checkpoint(&mut self, expected_position: usize) -> Result<()> {
        if self.transaction_active {
            return Err(XrtError::Runtime(
                "cannot roll back a CUDA DeltaNet checkpoint during an active token transaction"
                    .to_string(),
            ));
        }
        let checkpoint_position = self.checkpoint_position.ok_or_else(|| {
            XrtError::Runtime("CUDA DeltaNet fast checkpoint is not active".to_string())
        })?;
        if checkpoint_position != expected_position {
            return Err(XrtError::Runtime(format!(
                "CUDA DeltaNet checkpoint rollback boundary mismatch: expected {expected_position}, found {checkpoint_position}"
            )));
        }
        for layer in self.layers.iter_mut().flatten() {
            self.device
                .copy_f32_device(&layer.checkpoint_conv, &mut layer.committed_conv)?;
            self.device
                .copy_f32_device(&layer.checkpoint_recurrent, &mut layer.committed_recurrent)?;
        }
        self.position = checkpoint_position;
        self.checkpoint_position = None;
        self.last_verify_window = None;
        self.needs_zero = false;
        Ok(())
    }

    pub fn begin_token(&mut self, expected_position: usize) -> Result<()> {
        self.prepare()?;
        if self.position != expected_position {
            return Err(XrtError::Runtime(format!(
                "CUDA DeltaNet state position mismatch: expected {expected_position}, found {}",
                self.position
            )));
        }
        if self.transaction_active {
            return Err(XrtError::Runtime(
                "CUDA DeltaNet token transaction is already active".to_string(),
            ));
        }
        if self.last_verify_window.is_some() {
            return Err(XrtError::Runtime(
                "cannot begin a CUDA DeltaNet token before publishing the verify boundary"
                    .to_string(),
            ));
        }
        self.transaction_active = true;
        Ok(())
    }

    /// Begins a layerwise speculative verification transaction.
    ///
    /// Each recurrent layer advances through every row before the next layer
    /// runs. This preserves causal DeltaNet dependencies while allowing the
    /// expensive projections around them to execute as one small matrix.
    pub fn begin_verify_window(&mut self, expected_position: usize, rows: usize) -> Result<()> {
        self.begin_verify_window_inner(expected_position, rows, false)
    }

    pub fn begin_tree_verify_window(
        &mut self,
        expected_position: usize,
        rows: usize,
    ) -> Result<()> {
        self.begin_verify_window_inner(expected_position, rows, true)
    }

    fn begin_verify_window_inner(
        &mut self,
        expected_position: usize,
        rows: usize,
        tree_layout: bool,
    ) -> Result<()> {
        self.prepare()?;
        if !(2..=16).contains(&rows) {
            return Err(XrtError::Runtime(format!(
                "CUDA DeltaNet verify window requires 2..=16 rows, found {rows}"
            )));
        }
        if self.position != expected_position {
            return Err(XrtError::Runtime(format!(
                "CUDA DeltaNet verify position mismatch: expected {expected_position}, found {}",
                self.position
            )));
        }
        if self.checkpoint_position != Some(expected_position) {
            return Err(XrtError::Runtime(format!(
                "CUDA DeltaNet verify window requires a fast checkpoint at position {expected_position}"
            )));
        }
        if self.transaction_active {
            return Err(XrtError::Runtime(
                "CUDA DeltaNet transaction is already active".to_string(),
            ));
        }
        // Keep every non-final boundary in one contiguous allocation. Besides
        // making rollback indexing constant-time, this lets the fused verifier
        // publish its snapshot tensor with one bulk D2D operation per state
        // kind instead of one launch per row.
        self.ensure_verify_rebase_capacity(rows.saturating_sub(1))?;
        self.transaction_active = true;
        self.verify_window = Some(CudaVerifyWindowTransaction {
            rows,
            tree_layout,
            layer_advances: vec![0; self.layers.len()],
        });
        Ok(())
    }

    pub fn layer_buffers_mut(
        &mut self,
        layer: usize,
    ) -> Result<(
        &CudaF32Buffer,
        &mut CudaF32Buffer,
        &CudaF32Buffer,
        &mut CudaF32Buffer,
        CudaDeltaNetGeometry,
    )> {
        if !self.transaction_active {
            return Err(XrtError::Runtime(
                "CUDA DeltaNet layer access requires an active token transaction".to_string(),
            ));
        }
        let layer = self
            .layers
            .get_mut(layer)
            .and_then(Option::as_mut)
            .ok_or_else(|| {
                XrtError::Runtime(format!(
                    "layer {layer} is not a CUDA DeltaNet recurrent layer"
                ))
            })?;
        Ok((
            &layer.committed_conv,
            &mut layer.pending_conv,
            &layer.committed_recurrent,
            &mut layer.pending_recurrent,
            layer.geometry,
        ))
    }

    /// Returns the committed inputs for a fused speculative layer pass.
    pub fn verify_layer_committed_buffers(
        &self,
        layer: usize,
    ) -> Result<(&CudaF32Buffer, &CudaF32Buffer, CudaDeltaNetGeometry)> {
        if !self.transaction_active || self.verify_window.is_none() {
            return Err(XrtError::Runtime(
                "CUDA DeltaNet fused layer access requires an active verify window".to_string(),
            ));
        }
        let state = self
            .layers
            .get(layer)
            .and_then(Option::as_ref)
            .ok_or_else(|| {
                XrtError::Runtime(format!(
                    "layer {layer} is not a CUDA DeltaNet recurrent layer"
                ))
            })?;
        Ok((
            &state.committed_conv,
            &state.committed_recurrent,
            state.geometry,
        ))
    }

    /// Returns stable graph inputs and persistent rollback outputs for one
    /// fused speculative layer. The verifier can write snapshots directly to
    /// this journal instead of staging them through request scratch.
    pub fn verify_layer_fused_buffers_mut(
        &mut self,
        layer: usize,
    ) -> Result<(
        &CudaF32Buffer,
        &CudaF32Buffer,
        &mut CudaF32Buffer,
        &mut CudaF32Buffer,
        CudaDeltaNetGeometry,
    )> {
        if !self.transaction_active || self.verify_window.is_none() {
            return Err(XrtError::Runtime(
                "CUDA DeltaNet fused layer access requires an active verify window".to_string(),
            ));
        }
        let state = self
            .layers
            .get_mut(layer)
            .and_then(Option::as_mut)
            .ok_or_else(|| {
                XrtError::Runtime(format!(
                    "layer {layer} is not a CUDA DeltaNet recurrent layer"
                ))
            })?;
        Ok((
            &state.committed_conv,
            &state.committed_recurrent,
            &mut state.verify_rebase_conv,
            &mut state.verify_rebase_recurrent,
            state.geometry,
        ))
    }

    /// Installs all recurrent boundaries emitted by one fused verify kernel.
    ///
    /// The final and penultimate states are placed in the same committed and
    /// pending handles produced by row-at-a-time swaps. Earlier boundaries are
    /// copied into the persistent rollback journal used by prefix acceptance.
    pub fn complete_fused_verify_layer(
        &mut self,
        layer: usize,
        final_conv: &CudaF32Buffer,
        final_recurrent: &CudaF32Buffer,
        conv_snapshots: &CudaF32Buffer,
        recurrent_snapshots: &CudaF32Buffer,
    ) -> Result<()> {
        self.complete_fused_verify_layer_inner(
            layer,
            final_conv,
            final_recurrent,
            Some((conv_snapshots, recurrent_snapshots)),
        )
    }

    /// Publishes final handles after a fused kernel wrote its snapshots
    /// directly into the persistent rollback journal.
    pub fn complete_fused_verify_layer_from_journal(
        &mut self,
        layer: usize,
        final_conv: &CudaF32Buffer,
        final_recurrent: &CudaF32Buffer,
    ) -> Result<()> {
        self.complete_fused_verify_layer_inner(layer, final_conv, final_recurrent, None)
    }

    /// Enqueues only the device copies that publish a fused verifier's final
    /// and penultimate states. Host handle swaps are deliberately deferred so
    /// this operation can be captured once and replayed by a CUDA Graph.
    pub fn enqueue_fused_verify_layer_from_journal(
        &mut self,
        layer: usize,
        final_conv: &CudaF32Buffer,
        final_recurrent: &CudaF32Buffer,
    ) -> Result<()> {
        if !self.transaction_active {
            return Err(XrtError::Runtime(
                "CUDA DeltaNet fused layer completion requires an active transaction".to_string(),
            ));
        }
        let transaction = self.verify_window.as_ref().ok_or_else(|| {
            XrtError::Runtime(
                "CUDA DeltaNet fused layer completion requires a verify window".to_string(),
            )
        })?;
        let advances = transaction.layer_advances.get(layer).ok_or_else(|| {
            XrtError::Runtime(format!(
                "CUDA DeltaNet verify layer {layer} is out of range"
            ))
        })?;
        if *advances != 0 {
            return Err(XrtError::Runtime(format!(
                "CUDA DeltaNet fused layer {layer} already advanced {advances} rows"
            )));
        }
        let rows = transaction.rows;
        let state = self
            .layers
            .get_mut(layer)
            .and_then(Option::as_mut)
            .ok_or_else(|| {
                XrtError::Runtime(format!(
                    "layer {layer} is not a CUDA DeltaNet recurrent layer"
                ))
            })?;
        let conv_len = state.committed_conv.len();
        let recurrent_len = state.committed_recurrent.len();
        if final_conv.len() != conv_len || final_recurrent.len() != recurrent_len {
            return Err(XrtError::Shape(format!(
                "CUDA DeltaNet fused final-state length mismatch for layer {layer}"
            )));
        }
        let penultimate = rows - 2;
        if rows % 2 == 1 {
            self.device
                .copy_f32_device(final_conv, &mut state.pending_conv)?;
            self.device
                .copy_f32_device(final_recurrent, &mut state.pending_recurrent)?;
            self.device.copy_f32_device_range(
                &state.verify_rebase_conv,
                penultimate.saturating_mul(conv_len),
                &mut state.committed_conv,
            )?;
            self.device.copy_f32_device_range(
                &state.verify_rebase_recurrent,
                penultimate.saturating_mul(recurrent_len),
                &mut state.committed_recurrent,
            )?;
        } else {
            self.device
                .copy_f32_device(final_conv, &mut state.committed_conv)?;
            self.device
                .copy_f32_device(final_recurrent, &mut state.committed_recurrent)?;
            self.device.copy_f32_device_range(
                &state.verify_rebase_conv,
                penultimate.saturating_mul(conv_len),
                &mut state.pending_conv,
            )?;
            self.device.copy_f32_device_range(
                &state.verify_rebase_recurrent,
                penultimate.saturating_mul(recurrent_len),
                &mut state.pending_recurrent,
            )?;
        }
        Ok(())
    }

    /// Publishes the last physical tree node into the pending handles while
    /// leaving the committed root state untouched. After target path
    /// selection, `publish_tree_verify_boundary` copies either this pending
    /// state or an earlier node snapshot into the committed handles.
    pub fn enqueue_fused_tree_verify_layer_from_journal(
        &mut self,
        layer: usize,
        final_conv: &CudaF32Buffer,
        final_recurrent: &CudaF32Buffer,
    ) -> Result<()> {
        if !self.transaction_active {
            return Err(XrtError::Runtime(
                "CUDA DeltaNet tree completion requires an active transaction".to_string(),
            ));
        }
        let transaction = self.verify_window.as_ref().ok_or_else(|| {
            XrtError::Runtime("CUDA DeltaNet tree completion requires a verify window".to_string())
        })?;
        if !transaction.tree_layout {
            return Err(XrtError::Runtime(
                "CUDA DeltaNet tree completion received a linear verify window".to_string(),
            ));
        }
        let advances = transaction.layer_advances.get(layer).ok_or_else(|| {
            XrtError::Runtime(format!(
                "CUDA DeltaNet verify layer {layer} is out of range"
            ))
        })?;
        if *advances != 0 {
            return Err(XrtError::Runtime(format!(
                "CUDA DeltaNet tree layer {layer} already advanced {advances} rows"
            )));
        }
        let state = self
            .layers
            .get_mut(layer)
            .and_then(Option::as_mut)
            .ok_or_else(|| {
                XrtError::Runtime(format!(
                    "layer {layer} is not a CUDA DeltaNet recurrent layer"
                ))
            })?;
        if final_conv.len() != state.pending_conv.len()
            || final_recurrent.len() != state.pending_recurrent.len()
        {
            return Err(XrtError::Shape(format!(
                "CUDA DeltaNet tree final-state length mismatch for layer {layer}"
            )));
        }
        self.device
            .copy_f32_device(final_conv, &mut state.pending_conv)?;
        self.device
            .copy_f32_device(final_recurrent, &mut state.pending_recurrent)
    }

    pub fn complete_fused_tree_verify_layer_from_journal(
        &mut self,
        layer: usize,
        final_conv: &CudaF32Buffer,
        final_recurrent: &CudaF32Buffer,
    ) -> Result<()> {
        self.enqueue_fused_tree_verify_layer_from_journal(layer, final_conv, final_recurrent)?;
        let rows = self
            .verify_window
            .as_ref()
            .expect("validated tree verify window must exist")
            .rows;
        self.verify_window
            .as_mut()
            .expect("validated tree verify window must exist")
            .layer_advances[layer] = rows;
        Ok(())
    }

    /// Applies the host-only half of graph-compatible fused verification after
    /// the graph launch has completed successfully.
    pub fn commit_fused_verify_graph_layers(&mut self) -> Result<()> {
        let rows = self
            .verify_window
            .as_ref()
            .ok_or_else(|| {
                XrtError::Runtime(
                    "CUDA DeltaNet graph completion requires a verify window".to_string(),
                )
            })?
            .rows;
        for layer in 0..self.layers.len() {
            if self.layers[layer].is_none() {
                continue;
            }
            let advances = self
                .verify_window
                .as_ref()
                .and_then(|window| window.layer_advances.get(layer))
                .copied()
                .ok_or_else(|| {
                    XrtError::Runtime(format!(
                        "CUDA DeltaNet verify layer {layer} is out of range"
                    ))
                })?;
            if advances != 0 {
                return Err(XrtError::Runtime(format!(
                    "CUDA DeltaNet fused layer {layer} already advanced {advances} rows"
                )));
            }
            let tree_layout = self
                .verify_window
                .as_ref()
                .expect("checked verify window must exist")
                .tree_layout;
            if !tree_layout && rows % 2 == 1 {
                let state = self.layers[layer]
                    .as_mut()
                    .expect("checked recurrent layer must exist");
                std::mem::swap(&mut state.committed_conv, &mut state.pending_conv);
                std::mem::swap(&mut state.committed_recurrent, &mut state.pending_recurrent);
            }
            self.verify_window
                .as_mut()
                .expect("checked verify window must exist")
                .layer_advances[layer] = rows;
        }
        Ok(())
    }

    fn complete_fused_verify_layer_inner(
        &mut self,
        layer: usize,
        final_conv: &CudaF32Buffer,
        final_recurrent: &CudaF32Buffer,
        staged_snapshots: Option<(&CudaF32Buffer, &CudaF32Buffer)>,
    ) -> Result<()> {
        if !self.transaction_active {
            return Err(XrtError::Runtime(
                "CUDA DeltaNet fused layer completion requires an active transaction".to_string(),
            ));
        }
        let transaction = self.verify_window.as_mut().ok_or_else(|| {
            XrtError::Runtime(
                "CUDA DeltaNet fused layer completion requires a verify window".to_string(),
            )
        })?;
        let advances = transaction.layer_advances.get_mut(layer).ok_or_else(|| {
            XrtError::Runtime(format!(
                "CUDA DeltaNet verify layer {layer} is out of range"
            ))
        })?;
        if *advances != 0 {
            return Err(XrtError::Runtime(format!(
                "CUDA DeltaNet fused layer {layer} already advanced {advances} rows"
            )));
        }
        let rows = transaction.rows;
        let state = self
            .layers
            .get_mut(layer)
            .and_then(Option::as_mut)
            .ok_or_else(|| {
                XrtError::Runtime(format!(
                    "layer {layer} is not a CUDA DeltaNet recurrent layer"
                ))
            })?;
        let conv_len = state.committed_conv.len();
        let recurrent_len = state.committed_recurrent.len();
        if final_conv.len() != conv_len || final_recurrent.len() != recurrent_len {
            return Err(XrtError::Shape(format!(
                "CUDA DeltaNet fused final-state length mismatch for layer {layer}"
            )));
        }
        let snapshot_rows = rows - 1;
        let expected_conv_snapshots = snapshot_rows.saturating_mul(conv_len);
        let expected_recurrent_snapshots = snapshot_rows.saturating_mul(recurrent_len);
        if state.verify_rebase_conv.len() < expected_conv_snapshots
            || state.verify_rebase_recurrent.len() < expected_recurrent_snapshots
        {
            return Err(XrtError::Shape(format!(
                "CUDA DeltaNet fused journal capacity is insufficient for layer {layer}"
            )));
        }

        if let Some((conv_snapshots, recurrent_snapshots)) = staged_snapshots {
            if conv_snapshots.len() != expected_conv_snapshots
                || recurrent_snapshots.len() != expected_recurrent_snapshots
            {
                return Err(XrtError::Shape(format!(
                    "CUDA DeltaNet fused snapshot length mismatch for layer {layer}"
                )));
            }
            self.device.copy_f32_device_into_range(
                conv_snapshots,
                &mut state.verify_rebase_conv,
                0,
            )?;
            self.device.copy_f32_device_into_range(
                recurrent_snapshots,
                &mut state.verify_rebase_recurrent,
                0,
            )?;
        }

        let _ = state;
        let _ = advances;
        self.enqueue_fused_verify_layer_from_journal(layer, final_conv, final_recurrent)?;
        if rows % 2 == 1 {
            let state = self.layers[layer]
                .as_mut()
                .expect("validated recurrent layer must exist");
            std::mem::swap(&mut state.committed_conv, &mut state.pending_conv);
            std::mem::swap(&mut state.committed_recurrent, &mut state.pending_recurrent);
        }
        self.verify_window
            .as_mut()
            .expect("validated verify window must exist")
            .layer_advances[layer] = rows;
        Ok(())
    }

    /// Publishes one row for one recurrent layer inside a verify window.
    pub fn advance_verify_layer(&mut self, layer: usize) -> Result<()> {
        if !self.transaction_active {
            return Err(XrtError::Runtime(
                "CUDA DeltaNet verify layer advance requires an active transaction".to_string(),
            ));
        }
        let transaction = self.verify_window.as_mut().ok_or_else(|| {
            XrtError::Runtime(
                "CUDA DeltaNet verify layer advance requires a verify window".to_string(),
            )
        })?;
        let advances = transaction.layer_advances.get_mut(layer).ok_or_else(|| {
            XrtError::Runtime(format!(
                "CUDA DeltaNet verify layer {layer} is out of range"
            ))
        })?;
        if *advances >= transaction.rows {
            return Err(XrtError::Runtime(format!(
                "CUDA DeltaNet verify layer {layer} already advanced {} rows",
                transaction.rows
            )));
        }
        let state = self
            .layers
            .get_mut(layer)
            .and_then(Option::as_mut)
            .ok_or_else(|| {
                XrtError::Runtime(format!(
                    "layer {layer} is not a CUDA DeltaNet recurrent layer"
                ))
            })?;
        let row = *advances;
        if row < transaction.rows.saturating_sub(1) {
            self.device.copy_f32_device_into_range(
                &state.pending_conv,
                &mut state.verify_rebase_conv,
                row.saturating_mul(state.pending_conv.len()),
            )?;
            self.device.copy_f32_device_into_range(
                &state.pending_recurrent,
                &mut state.verify_rebase_recurrent,
                row.saturating_mul(state.pending_recurrent.len()),
            )?;
        }
        std::mem::swap(&mut state.committed_conv, &mut state.pending_conv);
        std::mem::swap(&mut state.committed_recurrent, &mut state.pending_recurrent);
        *advances += 1;
        Ok(())
    }

    pub fn commit_verify_window(&mut self, expected_position: usize) -> Result<()> {
        if !self.transaction_active {
            return Err(XrtError::Runtime(
                "CUDA DeltaNet verify transaction is not active".to_string(),
            ));
        }
        if self.position != expected_position {
            return Err(XrtError::Runtime(format!(
                "CUDA DeltaNet verify commit position mismatch: expected {expected_position}, found {}",
                self.position
            )));
        }
        let transaction = self.verify_window.as_ref().ok_or_else(|| {
            XrtError::Runtime("CUDA DeltaNet verify window is not active".to_string())
        })?;
        for (layer, (state, advances)) in self
            .layers
            .iter()
            .zip(&transaction.layer_advances)
            .enumerate()
        {
            if state.is_some() && *advances != transaction.rows {
                return Err(XrtError::Runtime(format!(
                    "CUDA DeltaNet verify layer {layer} advanced {advances} of {} rows",
                    transaction.rows
                )));
            }
        }
        let transaction = self
            .verify_window
            .take()
            .expect("verified CUDA DeltaNet window must still be present");
        if !transaction.tree_layout {
            self.position = self.position.checked_add(transaction.rows).ok_or_else(|| {
                XrtError::Runtime("CUDA DeltaNet position overflowed".to_string())
            })?;
            if transaction.rows % 2 == 1 {
                self.committed_buffer_generation ^= 1;
            }
        }
        self.transaction_active = false;
        self.last_verify_window =
            Some((expected_position, transaction.rows, transaction.tree_layout));
        Ok(())
    }

    /// Publishes an accepted prefix of the last verify window without replay.
    /// Early boundaries are donated from persistent per-layer rebase buffers;
    /// the penultimate boundary is already retained in the pending handles.
    pub fn publish_verify_boundary(
        &mut self,
        expected_position: usize,
        retained_rows: usize,
    ) -> Result<()> {
        if self.transaction_active {
            return Err(XrtError::Runtime(
                "cannot publish a CUDA DeltaNet verify boundary during an active transaction"
                    .to_string(),
            ));
        }
        let (start, rows, tree_layout) = self.last_verify_window.ok_or_else(|| {
            XrtError::Runtime("CUDA DeltaNet has no completed verify window".to_string())
        })?;
        if tree_layout {
            return Err(XrtError::Runtime(
                "CUDA DeltaNet tree verification requires a selected node boundary".to_string(),
            ));
        }
        if start != expected_position {
            return Err(XrtError::Runtime(format!(
                "CUDA DeltaNet verify boundary mismatch: expected {expected_position}, found {start}"
            )));
        }
        if self.checkpoint_position != Some(expected_position) {
            return Err(XrtError::Runtime(format!(
                "CUDA DeltaNet verify boundary requires a checkpoint at {expected_position}"
            )));
        }
        if retained_rows == 0 || retained_rows > rows {
            return Err(XrtError::Runtime(format!(
                "CUDA DeltaNet verify boundary must retain 1..={rows} rows, found {retained_rows}"
            )));
        }
        if retained_rows < rows {
            for layer in self.layers.iter_mut().flatten() {
                let rebase = retained_rows - 1;
                self.device.copy_f32_device_range(
                    &layer.verify_rebase_conv,
                    rebase.saturating_mul(layer.committed_conv.len()),
                    &mut layer.committed_conv,
                )?;
                self.device.copy_f32_device_range(
                    &layer.verify_rebase_recurrent,
                    rebase.saturating_mul(layer.committed_recurrent.len()),
                    &mut layer.committed_recurrent,
                )?;
            }
            self.position = expected_position
                .checked_add(retained_rows)
                .ok_or_else(|| {
                    XrtError::Runtime("CUDA DeltaNet verify boundary overflowed".to_string())
                })?;
        }
        self.last_verify_window = None;
        Ok(())
    }

    /// Publishes one selected root-to-node tree path without replaying rejected
    /// siblings. `selected_row` is the physical verifier row containing the
    /// last retained input; `retained_inputs` is its logical depth plus one.
    pub fn publish_tree_verify_boundary(
        &mut self,
        expected_position: usize,
        selected_row: usize,
        retained_inputs: usize,
    ) -> Result<()> {
        if self.transaction_active {
            return Err(XrtError::Runtime(
                "cannot publish a CUDA DeltaNet tree boundary during an active transaction"
                    .to_string(),
            ));
        }
        let (start, rows, tree_layout) = self.last_verify_window.ok_or_else(|| {
            XrtError::Runtime("CUDA DeltaNet has no completed tree verify window".to_string())
        })?;
        if !tree_layout || start != expected_position {
            return Err(XrtError::Runtime(format!(
                "CUDA DeltaNet tree boundary mismatch at position {expected_position}"
            )));
        }
        if self.checkpoint_position != Some(expected_position) {
            return Err(XrtError::Runtime(format!(
                "CUDA DeltaNet tree boundary requires a checkpoint at {expected_position}"
            )));
        }
        if selected_row >= rows || retained_inputs == 0 || retained_inputs > rows {
            return Err(XrtError::Runtime(format!(
                "CUDA DeltaNet tree boundary row {selected_row} / retained {retained_inputs} exceeds {rows} verifier rows"
            )));
        }
        for layer in self.layers.iter_mut().flatten() {
            if selected_row + 1 == rows {
                self.device
                    .copy_f32_device(&layer.pending_conv, &mut layer.committed_conv)?;
                self.device
                    .copy_f32_device(&layer.pending_recurrent, &mut layer.committed_recurrent)?;
            } else {
                self.device.copy_f32_device_range(
                    &layer.verify_rebase_conv,
                    selected_row.saturating_mul(layer.committed_conv.len()),
                    &mut layer.committed_conv,
                )?;
                self.device.copy_f32_device_range(
                    &layer.verify_rebase_recurrent,
                    selected_row.saturating_mul(layer.committed_recurrent.len()),
                    &mut layer.committed_recurrent,
                )?;
            }
        }
        self.position = expected_position
            .checked_add(retained_inputs)
            .ok_or_else(|| {
                XrtError::Runtime("CUDA DeltaNet tree boundary overflowed".to_string())
            })?;
        self.last_verify_window = None;
        Ok(())
    }

    pub fn commit_token(&mut self, expected_position: usize) -> Result<()> {
        if !self.transaction_active {
            return Err(XrtError::Runtime(
                "CUDA DeltaNet token transaction is not active".to_string(),
            ));
        }
        if self.verify_window.is_some() {
            return Err(XrtError::Runtime(
                "cannot commit a single token during a CUDA DeltaNet verify window".to_string(),
            ));
        }
        if self.position != expected_position {
            return Err(XrtError::Runtime(format!(
                "CUDA DeltaNet commit position mismatch: expected {expected_position}, found {}",
                self.position
            )));
        }
        let next_position = self.position.checked_add(1).ok_or_else(|| {
            XrtError::Runtime("CUDA DeltaNet state position overflowed".to_string())
        })?;
        for layer in self.layers.iter_mut().flatten() {
            std::mem::swap(&mut layer.committed_conv, &mut layer.pending_conv);
            std::mem::swap(&mut layer.committed_recurrent, &mut layer.pending_recurrent);
        }
        self.committed_buffer_generation ^= 1;
        self.position = next_position;
        self.transaction_active = false;
        Ok(())
    }

    pub fn abort_token(&mut self) {
        self.transaction_active = false;
        self.verify_window = None;
        self.last_verify_window = None;
    }

    pub fn snapshot(&self) -> Result<DeltaNetStateSnapshot> {
        if self.transaction_active {
            return Err(XrtError::Runtime(
                "cannot snapshot CUDA DeltaNet state during an active token transaction"
                    .to_string(),
            ));
        }
        if let Some(position) = self.checkpoint_position {
            return Err(XrtError::Runtime(format!(
                "cannot create a durable CUDA DeltaNet snapshot while a fast checkpoint is active at position {position}"
            )));
        }
        let layers = if self.needs_zero {
            self.descriptor
                .layers()
                .iter()
                .map(|geometry| {
                    geometry.as_ref().map(|geometry| {
                        (
                            vec![0.0; geometry.conv_state_len()].into_boxed_slice(),
                            vec![0.0; geometry.recurrent_state_len()].into_boxed_slice(),
                        )
                    })
                })
                .collect()
        } else {
            let mut payloads = Vec::with_capacity(self.layers.len());
            for layer in &self.layers {
                payloads.push(match layer {
                    Some(layer) => Some((
                        self.device
                            .download_f32(&layer.committed_conv)?
                            .into_boxed_slice(),
                        self.device
                            .download_f32(&layer.committed_recurrent)?
                            .into_boxed_slice(),
                    )),
                    None => None,
                });
            }
            payloads
        };
        DeltaNetStateSnapshot::try_from_parts(self.descriptor.clone(), self.position, layers)
    }

    pub fn restore(&mut self, snapshot: &DeltaNetStateSnapshot) -> Result<()> {
        if self.transaction_active {
            return Err(XrtError::Runtime(
                "cannot restore CUDA DeltaNet state during an active token transaction".to_string(),
            ));
        }
        if let Some(position) = self.checkpoint_position {
            return Err(XrtError::Runtime(format!(
                "cannot restore CUDA DeltaNet state while a fast checkpoint is active at position {position}"
            )));
        }
        let position = snapshot.validate_for_descriptor(&self.descriptor)?;

        // Stage every payload into the non-visible buffers first. A failed H2D
        // copy cannot partially publish the restored boundary.
        for (layer, payload) in self.layers.iter_mut().zip(snapshot.layers()) {
            match (layer.as_mut(), payload.as_ref()) {
                (Some(layer), Some(payload)) => {
                    self.device
                        .upload_f32_into(payload.conv_state_f32(), &mut layer.pending_conv)?;
                    self.device.upload_f32_into(
                        payload.recurrent_state_f32(),
                        &mut layer.pending_recurrent,
                    )?;
                }
                (None, None) => {}
                _ => unreachable!("snapshot presence was validated before CUDA mutation"),
            }
        }
        for layer in self.layers.iter_mut().flatten() {
            std::mem::swap(&mut layer.committed_conv, &mut layer.pending_conv);
            std::mem::swap(&mut layer.committed_recurrent, &mut layer.pending_recurrent);
        }
        self.committed_buffer_generation ^= 1;
        self.position = position;
        self.needs_zero = false;
        Ok(())
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a CUDA-capable device and driver"]
    fn fast_checkpoint_restores_device_state_without_changing_buffer_generation() -> Result<()> {
        let device = CudaDevice::new(0)?;
        let descriptor = DeltaNetStateDescriptor::from_geometry(
            "qwen3next",
            2,
            2,
            1,
            4,
            2,
            &[true, false, true],
        )?;
        let expected_bytes = descriptor
            .allocated_f32_elements()?
            .checked_mul(3)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .expect("test geometry should fit");
        let arena = Arc::new(GpuAllocationArena::default());
        arena.configure_budget(expected_bytes)?;
        let mut state = CudaDeltaNetState::try_new(device.clone(), descriptor, Some(&arena))?;
        assert_eq!(state.allocated_bytes(), expected_bytes);
        assert_eq!(
            arena.snapshot().by_class.recurrent_state_bytes,
            expected_bytes
        );

        state.begin_token(0)?;
        {
            let (_, pending_conv, _, pending_recurrent, _) = state.layer_buffers_mut(0)?;
            device.upload_f32_into(&[1.0; 8], pending_conv)?;
            device.upload_f32_into(&[2.0; 8], pending_recurrent)?;
        }
        state.commit_token(0)?;
        let accepted = state.snapshot()?;
        let generation_before_checkpoint = state.committed_buffer_generation();

        state.begin_fast_checkpoint(1)?;
        assert_eq!(state.fast_checkpoint_position(), Some(1));
        assert!(state.snapshot().is_err());
        state.begin_token(1)?;
        {
            let (_, pending_conv, _, pending_recurrent, _) = state.layer_buffers_mut(0)?;
            device.upload_f32_into(&[9.0; 8], pending_conv)?;
            device.upload_f32_into(&[10.0; 8], pending_recurrent)?;
        }
        state.commit_token(1)?;
        assert_ne!(
            state.committed_buffer_generation(),
            generation_before_checkpoint
        );
        state.rollback_fast_checkpoint(1)?;
        assert_eq!(state.position(), 1);
        assert_eq!(state.fast_checkpoint_position(), None);
        // Rollback copies into the currently committed handles and therefore
        // does not lie about the pointer generation used by graph keys.
        assert_ne!(
            state.committed_buffer_generation(),
            generation_before_checkpoint
        );
        assert_eq!(state.snapshot()?, accepted);

        state.begin_fast_checkpoint(1)?;
        state.commit_fast_checkpoint()?;
        assert_eq!(state.snapshot()?, accepted);
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA-capable device and driver"]
    fn verify_window_advances_each_layer_causally_and_rolls_back() -> Result<()> {
        let device = CudaDevice::new(0)?;
        let descriptor = DeltaNetStateDescriptor::from_geometry(
            "qwen3next",
            2,
            2,
            1,
            4,
            2,
            &[true, false, true],
        )?;
        let mut state = CudaDeltaNetState::try_new(device.clone(), descriptor, None)?;
        state.begin_token(0)?;
        for layer in [0, 2] {
            let (_, pending_conv, _, pending_recurrent, _) = state.layer_buffers_mut(layer)?;
            device.upload_f32_into(&[1.0; 8], pending_conv)?;
            device.upload_f32_into(&[2.0; 8], pending_recurrent)?;
        }
        state.commit_token(0)?;
        let accepted = state.snapshot()?;
        let generation = state.committed_buffer_generation();

        state.begin_fast_checkpoint(1)?;
        state.begin_verify_window(1, 3)?;
        for layer in [0, 2] {
            for row in 0..3 {
                let (_, pending_conv, _, pending_recurrent, _) = state.layer_buffers_mut(layer)?;
                device.upload_f32_into(&[10.0 + row as f32; 8], pending_conv)?;
                device.upload_f32_into(&[20.0 + row as f32; 8], pending_recurrent)?;
                state.advance_verify_layer(layer)?;
            }
        }
        state.commit_verify_window(1)?;
        assert_eq!(state.position(), 4);
        assert_ne!(state.committed_buffer_generation(), generation);
        state.rollback_fast_checkpoint(1)?;
        assert_eq!(state.position(), 1);
        assert_eq!(state.snapshot()?, accepted);
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA-capable device and driver"]
    fn verify_window_publishes_an_early_boundary_without_replay() -> Result<()> {
        let device = CudaDevice::new(0)?;
        let descriptor =
            DeltaNetStateDescriptor::from_geometry("qwen3next", 2, 2, 1, 4, 2, &[true])?;
        let mut state = CudaDeltaNetState::try_new(device.clone(), descriptor, None)?;
        state.begin_fast_checkpoint(0)?;
        state.begin_verify_window(0, 4)?;
        for row in 0..4 {
            let (_, pending_conv, _, pending_recurrent, _) = state.layer_buffers_mut(0)?;
            device.upload_f32_into(&[10.0 + row as f32; 8], pending_conv)?;
            device.upload_f32_into(&[20.0 + row as f32; 8], pending_recurrent)?;
            state.advance_verify_layer(0)?;
        }
        state.commit_verify_window(0)?;
        state.publish_verify_boundary(0, 1)?;
        state.commit_fast_checkpoint()?;
        let snapshot = state.snapshot()?;
        assert_eq!(snapshot.position(), 1);
        let layer = snapshot.layers()[0].as_ref().expect("recurrent layer");
        assert_eq!(layer.conv_state_f32(), &[10.0; 8]);
        assert_eq!(layer.recurrent_state_f32(), &[20.0; 8]);
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA-capable device and driver"]
    fn fused_verify_completion_preserves_every_transaction_boundary() -> Result<()> {
        let device = CudaDevice::new(0)?;
        let descriptor =
            DeltaNetStateDescriptor::from_geometry("qwen3next", 2, 2, 1, 4, 2, &[true])?;

        for (rows, retained_rows) in [(4, 1), (4, 3), (4, 4), (3, 2), (3, 3)] {
            let mut state = CudaDeltaNetState::try_new(device.clone(), descriptor.clone(), None)?;
            state.begin_fast_checkpoint(0)?;
            state.begin_verify_window(0, rows)?;
            let final_conv = device.upload_f32(&vec![9.0 + rows as f32; 8])?;
            let final_recurrent = device.upload_f32(&vec![19.0 + rows as f32; 8])?;
            let conv_snapshots = device.upload_f32(
                &(0..rows - 1)
                    .flat_map(|row| vec![10.0 + row as f32; 8])
                    .collect::<Vec<_>>(),
            )?;
            let recurrent_snapshots = device.upload_f32(
                &(0..rows - 1)
                    .flat_map(|row| vec![20.0 + row as f32; 8])
                    .collect::<Vec<_>>(),
            )?;
            state.complete_fused_verify_layer(
                0,
                &final_conv,
                &final_recurrent,
                &conv_snapshots,
                &recurrent_snapshots,
            )?;
            state.commit_verify_window(0)?;
            state.publish_verify_boundary(0, retained_rows)?;
            state.commit_fast_checkpoint()?;

            let snapshot = state.snapshot()?;
            assert_eq!(snapshot.position(), retained_rows as u64);
            let layer = snapshot.layers()[0].as_ref().expect("recurrent layer");
            assert_eq!(layer.conv_state_f32(), &[9.0 + retained_rows as f32; 8]);
            assert_eq!(
                layer.recurrent_state_f32(),
                &[19.0 + retained_rows as f32; 8]
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA-capable device and driver"]
    fn verify_window_grows_rebase_storage_on_demand() -> Result<()> {
        let device = CudaDevice::new(0)?;
        let descriptor = DeltaNetStateDescriptor::from_geometry(
            "qwen3next",
            2,
            2,
            1,
            4,
            2,
            &[true, false, true],
        )?;
        let base_bytes = descriptor
            .allocated_f32_elements()?
            .checked_mul(3)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .expect("test geometry should fit");
        let rebase_bytes = descriptor
            .allocated_f32_elements()?
            .checked_mul(15)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .expect("test geometry should fit");
        let arena = Arc::new(GpuAllocationArena::default());
        arena.configure_budget(base_bytes + rebase_bytes)?;
        let mut state = CudaDeltaNetState::try_new(device.clone(), descriptor, Some(&arena))?;
        assert_eq!(state.allocated_bytes(), base_bytes);

        state.begin_fast_checkpoint(0)?;
        state.begin_verify_window(0, 16)?;
        assert_eq!(state.allocated_bytes(), base_bytes + rebase_bytes);
        assert_eq!(
            arena.snapshot().by_class.recurrent_state_bytes,
            base_bytes + rebase_bytes
        );
        for layer in [0, 2] {
            for row in 0..16 {
                let (_, pending_conv, _, pending_recurrent, _) = state.layer_buffers_mut(layer)?;
                device.upload_f32_into(&[10.0 + row as f32; 8], pending_conv)?;
                device.upload_f32_into(&[20.0 + row as f32; 8], pending_recurrent)?;
                state.advance_verify_layer(layer)?;
            }
        }
        state.commit_verify_window(0)?;
        state.publish_verify_boundary(0, 7)?;
        state.commit_fast_checkpoint()?;
        assert_eq!(state.position(), 7);
        let snapshot = state.snapshot()?;
        let layer = snapshot.layers()[0].as_ref().expect("recurrent layer");
        assert_eq!(layer.conv_state_f32(), &[16.0; 8]);
        assert_eq!(layer.recurrent_state_f32(), &[26.0; 8]);
        Ok(())
    }
}
