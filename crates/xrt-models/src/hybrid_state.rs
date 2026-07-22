use crate::llama::LlamaConfig;
use xrt_core::{Result, XrtError};

pub const DELTANET_STATE_SNAPSHOT_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeltaNetLayerGeometry {
    conv_state_len: usize,
    recurrent_state_len: usize,
}

impl DeltaNetLayerGeometry {
    pub fn conv_state_len(&self) -> usize {
        self.conv_state_len
    }

    pub fn recurrent_state_len(&self) -> usize {
        self.recurrent_state_len
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeltaNetStateDescriptor {
    architecture: String,
    conv_kernel: usize,
    state_size: usize,
    group_count: usize,
    inner_size: usize,
    dt_rank: usize,
    layers: Box<[Option<DeltaNetLayerGeometry>]>,
}

impl DeltaNetStateDescriptor {
    pub fn from_config(config: &LlamaConfig) -> Result<Option<Self>> {
        if !config.is_hybrid() {
            return Ok(None);
        }

        let recurrent_layers = (0..config.block_count)
            .map(|layer| config.is_recurrent(layer))
            .collect::<Vec<_>>();
        Self::from_geometry(
            config.architecture.clone(),
            config.ssm_conv_kernel.unwrap_or(4),
            config.ssm_state_size.unwrap_or(128),
            config.ssm_group_count.unwrap_or(16),
            config.ssm_inner_size.unwrap_or(2048),
            config.ssm_dt_rank.unwrap_or(16),
            &recurrent_layers,
        )
        .map(Some)
    }

    pub fn from_geometry(
        architecture: impl Into<String>,
        conv_kernel: usize,
        state_size: usize,
        group_count: usize,
        inner_size: usize,
        dt_rank: usize,
        recurrent_layers: &[bool],
    ) -> Result<Self> {
        if conv_kernel == 0
            || state_size == 0
            || group_count == 0
            || inner_size == 0
            || dt_rank == 0
        {
            return Err(XrtError::InvalidMetadata(
                "DeltaNet state geometry dimensions must be non-zero".to_string(),
            ));
        }
        if recurrent_layers.is_empty() || !recurrent_layers.iter().any(|&layer| layer) {
            return Err(XrtError::InvalidMetadata(
                "DeltaNet state geometry must contain at least one recurrent layer".to_string(),
            ));
        }
        if inner_size % dt_rank != 0 {
            return Err(XrtError::InvalidMetadata(format!(
                "DeltaNet inner size {inner_size} is not divisible by time-step rank {dt_rank}"
            )));
        }
        if dt_rank > 64 {
            return Err(XrtError::InvalidMetadata(format!(
                "DeltaNet time-step rank {dt_rank} exceeds the current CPU executor limit of 64"
            )));
        }

        let qk_channels = state_size
            .checked_mul(group_count)
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                XrtError::InvalidMetadata(
                    "DeltaNet Q/K convolution channel count overflows usize".to_string(),
                )
            })?;
        let conv_channels = qk_channels.checked_add(inner_size).ok_or_else(|| {
            XrtError::InvalidMetadata(
                "DeltaNet convolution channel count overflows usize".to_string(),
            )
        })?;
        let history = conv_kernel.checked_sub(1).ok_or_else(|| {
            XrtError::InvalidMetadata("DeltaNet convolution kernel must be non-zero".to_string())
        })?;
        let conv_state_len = history.checked_mul(conv_channels).ok_or_else(|| {
            XrtError::InvalidMetadata("DeltaNet convolution state size overflows usize".to_string())
        })?;
        let head_v_dim = inner_size / dt_rank;
        let recurrent_state_len = dt_rank
            .checked_mul(head_v_dim)
            .and_then(|value| value.checked_mul(state_size))
            .ok_or_else(|| {
                XrtError::InvalidMetadata(
                    "DeltaNet recurrent state size overflows usize".to_string(),
                )
            })?;
        let recurrent_geometry = DeltaNetLayerGeometry {
            conv_state_len,
            recurrent_state_len,
        };
        let layers = recurrent_layers
            .iter()
            .map(|&is_recurrent| is_recurrent.then(|| recurrent_geometry.clone()))
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Ok(Self {
            architecture: architecture.into(),
            conv_kernel,
            state_size,
            group_count,
            inner_size,
            dt_rank,
            layers,
        })
    }

    pub fn layers(&self) -> &[Option<DeltaNetLayerGeometry>] {
        &self.layers
    }

    pub fn architecture(&self) -> &str {
        &self.architecture
    }

    pub fn conv_kernel(&self) -> usize {
        self.conv_kernel
    }

    pub fn state_size(&self) -> usize {
        self.state_size
    }

    pub fn group_count(&self) -> usize {
        self.group_count
    }

    pub fn inner_size(&self) -> usize {
        self.inner_size
    }

    pub fn dt_rank(&self) -> usize {
        self.dt_rank
    }

    pub fn allocated_f32_elements(&self) -> Result<usize> {
        self.layers.iter().try_fold(0usize, |total, layer| {
            let layer_elements = layer.as_ref().map_or(Ok(0), |geometry| {
                geometry
                    .conv_state_len
                    .checked_add(geometry.recurrent_state_len)
                    .ok_or_else(|| {
                        XrtError::Runtime(
                            "DeltaNet layer state element count overflows usize".to_string(),
                        )
                    })
            })?;
            total.checked_add(layer_elements).ok_or_else(|| {
                XrtError::Runtime("DeltaNet state element count overflows usize".to_string())
            })
        })
    }
}

#[derive(Debug)]
pub struct DeltaNetLayerState {
    pub(crate) conv_state: Vec<f32>,
    pub(crate) recurrent_state: Vec<f32>,
    pending_conv_state: Vec<f32>,
    pending_recurrent_state: Vec<f32>,
}

#[derive(Debug)]
pub struct DeltaNetState {
    descriptor: DeltaNetStateDescriptor,
    layers: Vec<Option<DeltaNetLayerState>>,
    position: usize,
    transaction_active: bool,
}

impl DeltaNetState {
    pub fn try_new(descriptor: DeltaNetStateDescriptor) -> Result<Self> {
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(descriptor.layers.len())
            .map_err(|err| {
                XrtError::Runtime(format!(
                    "failed to reserve DeltaNet layer state descriptors: {err}"
                ))
            })?;
        for geometry in descriptor.layers.iter() {
            layers.push(match geometry {
                Some(geometry) => Some(DeltaNetLayerState {
                    conv_state: try_zeroed_f32(geometry.conv_state_len, "convolution")?,
                    recurrent_state: try_zeroed_f32(geometry.recurrent_state_len, "recurrent")?,
                    pending_conv_state: try_zeroed_f32(
                        geometry.conv_state_len,
                        "pending convolution",
                    )?,
                    pending_recurrent_state: try_zeroed_f32(
                        geometry.recurrent_state_len,
                        "pending recurrent",
                    )?,
                }),
                None => None,
            });
        }
        Ok(Self {
            descriptor,
            layers,
            position: 0,
            transaction_active: false,
        })
    }

    pub fn descriptor(&self) -> &DeltaNetStateDescriptor {
        &self.descriptor
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn allocated_bytes(&self) -> u64 {
        self.layers
            .iter()
            .flatten()
            .map(|layer| {
                layer
                    .conv_state
                    .len()
                    .saturating_add(layer.recurrent_state.len())
                    .saturating_add(layer.pending_conv_state.len())
                    .saturating_add(layer.pending_recurrent_state.len())
                    .saturating_mul(std::mem::size_of::<f32>())
            })
            .map(|bytes| u64::try_from(bytes).unwrap_or(u64::MAX))
            .fold(0u64, u64::saturating_add)
    }

    pub fn clear(&mut self) {
        for layer in self.layers.iter_mut().flatten() {
            layer.conv_state.fill(0.0);
            layer.recurrent_state.fill(0.0);
            layer.pending_conv_state.fill(0.0);
            layer.pending_recurrent_state.fill(0.0);
        }
        self.position = 0;
        self.transaction_active = false;
    }

    pub fn validate_position(&self, expected: usize) -> Result<()> {
        if self.position != expected {
            return Err(XrtError::Runtime(format!(
                "DeltaNet state position mismatch: expected {expected}, found {}",
                self.position
            )));
        }
        Ok(())
    }

    pub fn begin_token(
        &mut self,
        expected_position: usize,
    ) -> Result<DeltaNetTokenTransaction<'_>> {
        self.validate_position(expected_position)?;
        if self.transaction_active {
            return Err(XrtError::Runtime(
                "DeltaNet token transaction is already active".to_string(),
            ));
        }
        self.transaction_active = true;
        Ok(DeltaNetTokenTransaction {
            state: self,
            expected_position,
        })
    }

    pub fn snapshot(&self) -> Result<DeltaNetStateSnapshot> {
        if self.transaction_active {
            return Err(XrtError::Runtime(
                "cannot snapshot DeltaNet state during an active token transaction".to_string(),
            ));
        }
        Ok(DeltaNetStateSnapshot {
            version: DELTANET_STATE_SNAPSHOT_VERSION,
            descriptor: self.descriptor.clone(),
            position: u64::try_from(self.position).map_err(|_| {
                XrtError::Runtime(
                    "DeltaNet state position cannot be represented in a durable snapshot"
                        .to_string(),
                )
            })?,
            layers: self
                .layers
                .iter()
                .map(|layer| {
                    layer.as_ref().map(|layer| DeltaNetLayerSnapshot {
                        conv_state_f32: layer.conv_state.clone().into_boxed_slice(),
                        recurrent_state_f32: layer.recurrent_state.clone().into_boxed_slice(),
                    })
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        })
    }

    pub fn restore(&mut self, snapshot: &DeltaNetStateSnapshot) -> Result<()> {
        if self.transaction_active {
            return Err(XrtError::Runtime(
                "cannot restore DeltaNet state during an active token transaction".to_string(),
            ));
        }
        let position = self.validate_snapshot(snapshot)?;
        for (state, payload) in self.layers.iter_mut().zip(snapshot.layers.iter()) {
            match (state.as_mut(), payload.as_ref()) {
                (Some(state), Some(payload)) => {
                    state.conv_state.copy_from_slice(&payload.conv_state_f32);
                    state
                        .recurrent_state
                        .copy_from_slice(&payload.recurrent_state_f32);
                }
                (None, None) => {}
                _ => unreachable!("snapshot presence was validated before mutation"),
            }
        }
        self.position = position;
        Ok(())
    }

    fn validate_snapshot(&self, snapshot: &DeltaNetStateSnapshot) -> Result<usize> {
        snapshot.validate_for_descriptor(&self.descriptor)
    }
}

pub struct DeltaNetTokenTransaction<'a> {
    state: &'a mut DeltaNetState,
    expected_position: usize,
}

impl DeltaNetTokenTransaction<'_> {
    pub(crate) fn layer_buffers_mut(
        &mut self,
        layer: usize,
    ) -> Result<(&[f32], &mut [f32], &[f32], &mut [f32])> {
        let layer = self
            .state
            .layers
            .get_mut(layer)
            .and_then(Option::as_mut)
            .ok_or_else(|| {
                XrtError::Runtime(format!("layer {layer} is not a DeltaNet recurrent layer"))
            })?;
        Ok((
            &layer.conv_state,
            &mut layer.pending_conv_state,
            &layer.recurrent_state,
            &mut layer.pending_recurrent_state,
        ))
    }

    pub fn commit(self) -> Result<()> {
        self.state.validate_position(self.expected_position)?;
        let next_position = self.state.position.checked_add(1).ok_or_else(|| {
            XrtError::Runtime("DeltaNet state position overflows usize".to_string())
        })?;
        for layer in self.state.layers.iter_mut().flatten() {
            std::mem::swap(&mut layer.conv_state, &mut layer.pending_conv_state);
            std::mem::swap(
                &mut layer.recurrent_state,
                &mut layer.pending_recurrent_state,
            );
        }
        self.state.position = next_position;
        self.state.transaction_active = false;
        Ok(())
    }
}

impl Drop for DeltaNetTokenTransaction<'_> {
    fn drop(&mut self) {
        self.state.transaction_active = false;
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeltaNetLayerSnapshot {
    conv_state_f32: Box<[f32]>,
    recurrent_state_f32: Box<[f32]>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeltaNetStateSnapshot {
    version: u32,
    descriptor: DeltaNetStateDescriptor,
    position: u64,
    layers: Box<[Option<DeltaNetLayerSnapshot>]>,
}

impl DeltaNetStateSnapshot {
    pub fn try_from_parts(
        descriptor: DeltaNetStateDescriptor,
        position: usize,
        layers: Vec<Option<(Box<[f32]>, Box<[f32]>)>>,
    ) -> Result<Self> {
        let snapshot = Self {
            version: DELTANET_STATE_SNAPSHOT_VERSION,
            descriptor,
            position: u64::try_from(position).map_err(|_| {
                XrtError::Runtime(
                    "DeltaNet state position cannot be represented in a durable snapshot"
                        .to_string(),
                )
            })?,
            layers: layers
                .into_iter()
                .map(|layer| {
                    layer.map(
                        |(conv_state_f32, recurrent_state_f32)| DeltaNetLayerSnapshot {
                            conv_state_f32,
                            recurrent_state_f32,
                        },
                    )
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        };
        snapshot.validate_for_descriptor(&snapshot.descriptor)?;
        Ok(snapshot)
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn position(&self) -> u64 {
        self.position
    }

    pub fn descriptor(&self) -> &DeltaNetStateDescriptor {
        &self.descriptor
    }

    pub fn layers(&self) -> &[Option<DeltaNetLayerSnapshot>] {
        &self.layers
    }

    pub fn allocated_bytes(&self) -> u64 {
        self.layers
            .iter()
            .flatten()
            .map(|layer| {
                layer
                    .conv_state_f32
                    .len()
                    .saturating_add(layer.recurrent_state_f32.len())
                    .saturating_mul(std::mem::size_of::<f32>())
            })
            .map(|bytes| u64::try_from(bytes).unwrap_or(u64::MAX))
            .fold(0u64, u64::saturating_add)
    }

    pub fn validate_for_descriptor(&self, descriptor: &DeltaNetStateDescriptor) -> Result<usize> {
        if self.version != DELTANET_STATE_SNAPSHOT_VERSION {
            return Err(XrtError::Runtime(format!(
                "unsupported DeltaNet state snapshot version {}, expected {}",
                self.version, DELTANET_STATE_SNAPSHOT_VERSION
            )));
        }
        if &self.descriptor != descriptor {
            return Err(XrtError::Runtime(
                "DeltaNet state snapshot geometry does not match the session".to_string(),
            ));
        }
        if self.layers.len() != descriptor.layers.len() {
            return Err(XrtError::Runtime(format!(
                "DeltaNet snapshot has {} layers, expected {}",
                self.layers.len(),
                descriptor.layers.len()
            )));
        }
        let position = usize::try_from(self.position).map_err(|_| {
            XrtError::Runtime(format!(
                "DeltaNet snapshot position {} exceeds this platform's usize",
                self.position
            ))
        })?;
        for (index, (geometry, payload)) in
            descriptor.layers.iter().zip(self.layers.iter()).enumerate()
        {
            match (geometry.as_ref(), payload.as_ref()) {
                (None, None) => {}
                (Some(geometry), Some(payload))
                    if payload.conv_state_f32.len() == geometry.conv_state_len
                        && payload.recurrent_state_f32.len() == geometry.recurrent_state_len => {}
                (Some(geometry), Some(payload)) => {
                    return Err(XrtError::Runtime(format!(
                        "DeltaNet snapshot layer {index} payload mismatch: conv {} != {}, recurrent {} != {}",
                        payload.conv_state_f32.len(),
                        geometry.conv_state_len,
                        payload.recurrent_state_f32.len(),
                        geometry.recurrent_state_len
                    )));
                }
                _ => {
                    return Err(XrtError::Runtime(format!(
                        "DeltaNet snapshot layer {index} presence does not match session geometry"
                    )));
                }
            }
        }
        Ok(position)
    }
}

impl DeltaNetLayerSnapshot {
    pub fn conv_state_f32(&self) -> &[f32] {
        &self.conv_state_f32
    }

    pub fn recurrent_state_f32(&self) -> &[f32] {
        &self.recurrent_state_f32
    }
}

fn try_zeroed_f32(len: usize, role: &str) -> Result<Vec<f32>> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|err| {
        XrtError::Runtime(format!(
            "failed to reserve {len} F32 values for DeltaNet {role} state: {err}"
        ))
    })?;
    values.resize(len, 0.0);
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::{
        DeltaNetLayerGeometry, DeltaNetState, DeltaNetStateDescriptor,
        DELTANET_STATE_SNAPSHOT_VERSION,
    };

    fn descriptor() -> DeltaNetStateDescriptor {
        DeltaNetStateDescriptor {
            architecture: "qwen3_5".to_string(),
            conv_kernel: 4,
            state_size: 2,
            group_count: 1,
            inner_size: 4,
            dt_rank: 2,
            layers: vec![
                Some(DeltaNetLayerGeometry {
                    conv_state_len: 12,
                    recurrent_state_len: 8,
                }),
                None,
                Some(DeltaNetLayerGeometry {
                    conv_state_len: 12,
                    recurrent_state_len: 8,
                }),
            ]
            .into_boxed_slice(),
        }
    }

    #[test]
    fn snapshot_round_trip_preserves_position_and_payloads() {
        let mut state = DeltaNetState::try_new(descriptor()).unwrap();
        state.begin_token(0).unwrap().commit().unwrap();
        state.begin_token(1).unwrap().commit().unwrap();
        state.layers[0].as_mut().unwrap().conv_state[2] = 3.5;
        state.layers[2].as_mut().unwrap().recurrent_state[5] = -2.25;
        let snapshot = state.snapshot().unwrap();

        state.clear();
        state.restore(&snapshot).unwrap();

        assert_eq!(state.position(), 2);
        assert_eq!(state.layers[0].as_ref().unwrap().conv_state[2], 3.5);
        assert_eq!(state.layers[2].as_ref().unwrap().recurrent_state[5], -2.25);
    }

    #[test]
    fn malformed_snapshot_is_rejected_before_mutation() {
        let mut state = DeltaNetState::try_new(descriptor()).unwrap();
        state.begin_token(0).unwrap().commit().unwrap();
        state.layers[0].as_mut().unwrap().conv_state[0] = 9.0;
        let before = state.snapshot().unwrap();

        let mut malformed = before.clone();
        malformed.layers[0].as_mut().unwrap().conv_state_f32 = vec![0.0; 1].into_boxed_slice();
        assert!(state.restore(&malformed).is_err());

        assert_eq!(state.snapshot().unwrap(), before);
    }

    #[test]
    fn wrong_version_and_layer_presence_are_rejected() {
        let mut state = DeltaNetState::try_new(descriptor()).unwrap();
        let before = state.snapshot().unwrap();

        let mut wrong_version = before.clone();
        wrong_version.version = DELTANET_STATE_SNAPSHOT_VERSION + 1;
        assert!(state.restore(&wrong_version).is_err());

        let mut wrong_presence = before.clone();
        wrong_presence.layers[1] = wrong_presence.layers[0].clone();
        assert!(state.restore(&wrong_presence).is_err());

        assert_eq!(state.snapshot().unwrap(), before);
    }
}
