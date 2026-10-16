use super::*;
use xrt_kernels::cpu::delta_state::HeadExpansion;

/// Request-owned, key-major FP32 state for an eager BF16 serving contract.
/// The existing text/CPU-reference DeltaNet implementation stays separate.
pub struct GpuOrderedDeltaState {
    device: CudaDevice,
    state: CudaF32Buffer,
    input: CudaF32Buffer,
    normalized: CudaF32Buffer,
    output: CudaF32Buffer,
    key_heads: usize,
    value_heads: usize,
    tiled: bool,
    position: usize,
    poisoned: bool,
}
pub struct OrderedDeltaOutput {
    pub core: Vec<f32>,
    pub beta: Vec<f32>,
    pub log_decay: Vec<f32>,
}
impl GpuOrderedDeltaState {
    pub fn planned_bytes(key_heads: usize, value_heads: usize) -> Result<u64> {
        if key_heads == 0
            || value_heads == 0
            || key_heads > 64
            || value_heads > 64
            || value_heads % key_heads != 0
        {
            return Err(XrtError::Shape("ordered delta head geometry".into()));
        }
        let elements = value_heads * 128 * 128
            + 2 * key_heads * 128
            + value_heads * 128
            + 4 * value_heads
            + 2 * key_heads * 128
            + 2 * value_heads
            + value_heads * 128
            + 2 * value_heads;
        Ok(elements as u64 * 4)
    }
    pub fn new(
        device: CudaDevice,
        key_heads: usize,
        value_heads: usize,
        expansion: HeadExpansion,
        budget: u64,
    ) -> Result<Self> {
        let bytes = Self::planned_bytes(key_heads, value_heads)?;
        if bytes > budget || bytes > device.memory_info()?.0 {
            return Err(XrtError::Cuda(
                "ordered delta state exceeds memory budget/headroom".into(),
            ));
        }
        let module = "xrt_ordered_delta";
        if device
            .device
            .get_func(module, "ordered_delta_step")
            .is_none()
        {
            let _library = super::native_libraries::nvrtc()?;
            let ptx = cudarc::nvrtc::compile_ptx(include_str!("kernels/ordered_delta.cu"))
                .map_err(|e| XrtError::Cuda(format!("compile ordered delta: {e}")))?;
            device
                .device
                .load_ptx(
                    ptx,
                    module,
                    &["ordered_delta_prepare", "ordered_delta_step"],
                )
                .map_err(|e| XrtError::Cuda(format!("load ordered delta: {e}")))?;
        }
        Ok(Self {
            state: device.zeros_f32(value_heads * 128 * 128)?,
            input: device.zeros_f32(2 * key_heads * 128 + value_heads * 128 + 4 * value_heads)?,
            normalized: device.zeros_f32(2 * key_heads * 128 + 2 * value_heads)?,
            output: device.zeros_f32(value_heads * 128 + 2 * value_heads)?,
            device,
            key_heads,
            value_heads,
            tiled: expansion == HeadExpansion::Tile,
            position: 0,
            poisoned: false,
        })
    }
    pub fn reset(&mut self) -> Result<()> {
        self.poisoned = true;
        self.device
            .upload_f32_into(&vec![0.0; self.value_heads * 128 * 128], &mut self.state)?;
        self.position = 0;
        self.poisoned = false;
        Ok(())
    }
    pub fn snapshot_value_major(&self) -> Result<Vec<f32>> {
        let state = self.device.download_f32(&self.state)?;
        let mut result = vec![0.0; state.len()];
        for h in 0..self.value_heads {
            for k in 0..128 {
                for v in 0..128 {
                    result[(h * 128 + v) * 128 + k] = state[(h * 128 + k) * 128 + v];
                }
            }
        }
        Ok(result)
    }
    pub fn step(
        &mut self,
        qkv: &[f32],
        a: &[f32],
        b: &[f32],
        bias: &[f32],
        log_a: &[f32],
    ) -> Result<OrderedDeltaOutput> {
        if self.poisoned {
            return Err(XrtError::Runtime(
                "ordered delta state is poisoned; reset required".into(),
            ));
        }
        if qkv.len() != 2 * self.key_heads * 128 + self.value_heads * 128
            || [a, b, bias, log_a]
                .iter()
                .any(|v| v.len() != self.value_heads)
            || qkv
                .iter()
                .chain(a)
                .chain(b)
                .chain(bias)
                .chain(log_a)
                .any(|v| !v.is_finite())
        {
            return Err(XrtError::InvalidTensor("ordered delta inputs".into()));
        }
        let next = self
            .position
            .checked_add(1)
            .ok_or_else(|| XrtError::Shape("ordered delta position overflow".into()))?;
        let result = (|| {
            self.device
                .upload_f32_into(&[qkv, a, b, bias, log_a].concat(), &mut self.input)?;
            let prepare = self
                .device
                .device
                .get_func("xrt_ordered_delta", "ordered_delta_prepare")
                .ok_or_else(|| XrtError::Cuda("ordered delta prepare absent".into()))?;
            let step = self
                .device
                .device
                .get_func("xrt_ordered_delta", "ordered_delta_step")
                .ok_or_else(|| XrtError::Cuda("ordered delta step absent".into()))?;
            // SAFETY: all owners share this retained context. Shapes are fixed
            // at 128x128 and bounded to 64 heads; disjoint heads own their state.
            // The final download synchronizes both launches before host use.
            unsafe {
                prepare
                    .launch(
                        LaunchConfig {
                            grid_dim: (self.key_heads as u32, 1, 1),
                            block_dim: (32, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            &self.input.data,
                            &mut self.normalized.data,
                            self.key_heads as u32,
                            self.value_heads as u32,
                        ),
                    )
                    .map_err(|e| XrtError::Cuda(format!("prepare ordered delta: {e}")))?;
                step.launch(
                    LaunchConfig {
                        grid_dim: (self.value_heads as u32, 1, 1),
                        block_dim: (32, 4, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &mut self.state.data,
                        &self.input.data,
                        &self.normalized.data,
                        &mut self.output.data,
                        self.key_heads as u32,
                        self.value_heads as u32,
                        u32::from(self.tiled),
                        u32::from(self.position == 0),
                    ),
                )
                .map_err(|e| XrtError::Cuda(format!("advance ordered delta: {e}")))?;
            }
            let mut values = self.device.download_f32(&self.output)?;
            if values.iter().any(|v| !v.is_finite()) {
                return Err(XrtError::InvalidTensor("ordered delta overflow".into()));
            }
            let log_decay = values.split_off(self.value_heads * 128 + self.value_heads);
            let beta = values.split_off(self.value_heads * 128);
            Ok(OrderedDeltaOutput {
                core: values,
                beta,
                log_decay,
            })
        })();
        if result.is_ok() {
            self.position = next;
        } else {
            self.poisoned = true;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn floats(v: &serde_json::Value) -> Vec<f32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect()
    }
    fn close(actual: &[f32], expected: &[f32], rtol: f32, atol: f32) {
        assert_eq!(actual.len(), expected.len());
        for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - b).abs() <= atol + rtol * b.abs(),
                "index {i}: {a} vs {b}"
            );
        }
    }
    #[test]
    #[ignore = "requires installed NVIDIA driver and NVRTC; run explicitly"]
    fn ordered_delta_matches_pinned_cuda_sequence_mapping_reset_and_cleanup() {
        let f: serde_json::Value = serde_json::from_str(include_str!(
            "../../../reference/audio/drama3/bf16-delta-fixture.json"
        ))
        .unwrap();
        let device = CudaDevice::new(0).unwrap();
        let baseline = device.allocation_stats().current_bytes;
        let input: Vec<_> = f["input"].as_array().unwrap().iter().map(floats).collect();
        let a: Vec<_> = f["a"].as_array().unwrap().iter().map(floats).collect();
        let b: Vec<_> = f["b"].as_array().unwrap().iter().map(floats).collect();
        let bias = floats(&f["bias"]);
        let log_a = floats(&f["log_a"]);
        assert!(GpuOrderedDeltaState::planned_bytes(usize::MAX, 4).is_err());
        assert!(GpuOrderedDeltaState::new(
            device.clone(),
            2,
            4,
            HeadExpansion::RepeatInterleave,
            1
        )
        .is_err());
        for case in f["cases"].as_array().unwrap() {
            let expansion = if case["tiled"].as_bool().unwrap() {
                HeadExpansion::Tile
            } else {
                HeadExpansion::RepeatInterleave
            };
            let mut state =
                GpuOrderedDeltaState::new(device.clone(), 2, 4, expansion, 2 * 1024 * 1024)
                    .unwrap();
            assert_eq!(
                device.allocation_stats().current_bytes - baseline,
                GpuOrderedDeltaState::planned_bytes(2, 4).unwrap()
            );
            let mut first = None;
            for i in 0..input.len() {
                let result = state.step(&input[i], &a[i], &b[i], &bias, &log_a).unwrap();
                let expected_core = floats(&case["outputs"][i]["core"]);
                assert_eq!(
                    result
                        .core
                        .iter()
                        .zip(&expected_core)
                        .filter(|(a, b)| a != b)
                        .count(),
                    0,
                    "BF16 recurrent core at position {i}"
                );
                close(
                    &result.core,
                    &floats(&case["outputs"][i]["core"]),
                    0.008,
                    0.002,
                );
                close(
                    &result.beta,
                    &floats(&case["outputs"][i]["beta"]),
                    2e-6,
                    1e-6,
                );
                close(
                    &result.log_decay,
                    &floats(&case["outputs"][i]["log_decay"]),
                    2e-6,
                    1e-6,
                );
                if i == 0 {
                    first = Some(result.core);
                }
            }
            close(
                &state.snapshot_value_major().unwrap(),
                &floats(&case["final_state_value_major"]),
                2e-4,
                2e-4,
            );
            let snapshot = state.snapshot_value_major().unwrap();
            assert!(state
                .step(&[f32::NAN; 1024], &a[0], &b[0], &bias, &log_a)
                .is_err());
            assert_eq!(state.snapshot_value_major().unwrap(), snapshot);
            let raw = input[0].clone();
            let aa = a[0].clone();
            let bb = b[0].clone();
            let bias = bias.clone();
            let log_a = log_a.clone();
            state = std::thread::spawn(move || {
                state.reset().unwrap();
                assert!(state
                    .snapshot_value_major()
                    .unwrap()
                    .iter()
                    .all(|v| *v == 0.0));
                assert_eq!(
                    state.step(&raw, &aa, &bb, &bias, &log_a).unwrap().core,
                    first.unwrap()
                );
                state
            })
            .join()
            .unwrap();
            state.reset().unwrap();
            drop(state);
            device.synchronize().unwrap();
            assert_eq!(device.allocation_stats().current_bytes, baseline);
        }
    }
}
