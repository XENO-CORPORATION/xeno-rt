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
    committed_recurrent: CudaF32Buffer,
    pending_recurrent: CudaF32Buffer,
    checkpoint_recurrent: CudaF32Buffer,
    geometry: CudaDeltaNetGeometry,
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
    layers: Vec<Option<CudaDeltaNetLayerState>>,
    position: usize,
    transaction_active: bool,
    checkpoint_position: Option<usize>,
    committed_buffer_generation: u8,
    needs_zero: bool,
    allocated_bytes: u64,
    _allocation: Option<GpuAllocationLease>,
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
                XrtError::Cuda(
                    "CUDA DeltaNet double-buffered state byte count overflowed".to_string(),
                )
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
                    committed_recurrent: device.zeros_f32(layer.recurrent_state_len())?,
                    pending_recurrent: device.zeros_f32(layer.recurrent_state_len())?,
                    checkpoint_recurrent: device.zeros_f32(layer.recurrent_state_len())?,
                    geometry,
                }),
                None => None,
            });
        }

        Ok(Self {
            device,
            descriptor,
            layers,
            position: 0,
            transaction_active: false,
            checkpoint_position: None,
            committed_buffer_generation: 0,
            needs_zero: false,
            allocated_bytes,
            _allocation: allocation,
        })
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
            self.device.zero_f32(&mut layer.committed_recurrent)?;
            self.device.zero_f32(&mut layer.pending_recurrent)?;
            self.device.zero_f32(&mut layer.checkpoint_recurrent)?;
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
        self.transaction_active = true;
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

    pub fn commit_token(&mut self, expected_position: usize) -> Result<()> {
        if !self.transaction_active {
            return Err(XrtError::Runtime(
                "CUDA DeltaNet token transaction is not active".to_string(),
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
}
