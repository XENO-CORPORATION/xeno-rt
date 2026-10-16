// Included in the projection workspace module so every activation buffer uses
// its existing byte budget, owner context and clearing policy.
struct AttentionScratch {
    rows: usize,
    packed: CudaF32Buffer,
    scores: CudaF32Buffer,
    probabilities: CudaF32Buffer,
    stats: CudaF32Buffer,
    output: CudaF32Buffer,
}
impl AttentionScratch {
    fn bytes(&self) -> u64 {
        [
            &self.packed,
            &self.scores,
            &self.probabilities,
            &self.stats,
            &self.output,
        ]
        .iter()
        .map(|v| v.byte_len() as u64)
        .sum()
    }
}
impl GpuDenseWorkspace {
    /// One decode row with explicit grouped-query geometry. Keys/values are the
    /// complete causal prefix. BF16 operands and online 128-key accumulation
    /// follow the qualified efficient-attention serving precision profile.
    pub fn attention_bf16(
        &mut self,
        query: &[f32],
        keys: &[f32],
        values: &[f32],
        heads: usize,
        kv: usize,
        dim: usize,
    ) -> Result<Vec<f32>> {
        if !(1..=64).contains(&heads)
            || kv == 0
            || kv > heads
            || heads % kv != 0
            || heads / kv > 16
            || ![128, 256].contains(&dim)
            || query.len() != heads * dim
            || keys.is_empty()
            || keys.len() % (kv * dim) != 0
            || values.len() != keys.len()
        {
            return Err(XrtError::Shape("BF16 attention geometry".into()));
        }
        let rows = keys.len() / (kv * dim);
        if rows > 8192
            || query
                .iter()
                .chain(keys)
                .chain(values)
                .any(|v| !v.is_finite())
        {
            return Err(XrtError::InvalidTensor(
                "BF16 attention input/context bounds".into(),
            ));
        }
        let capacity = rows.next_power_of_two();
        let key = (heads, kv, dim);
        let qwidth = heads * dim;
        let input_capacity = qwidth + 2 * capacity * kv * dim;
        let score_capacity = heads * capacity;
        let stat_capacity = heads * capacity.div_ceil(128) * 2;
        let needed = (input_capacity + 2 * score_capacity + stat_capacity + qwidth) as u64 * 4;
        if needed > self.budget {
            return Err(XrtError::Cuda(
                "BF16 attention exceeds workspace budget".into(),
            ));
        }
        self.device
            .device
            .bind_to_thread()
            .map_err(|e| cuda_error("BF16 attention context", e))?;
        let module = "xrt_bf16_attention";
        if self
            .device
            .device
            .get_func(module, "native_attention_pv")
            .is_none()
        {
            let _library = super::native_libraries::nvrtc()?;
            let ptx = cudarc::nvrtc::compile_ptx_with_opts(
                include_str!("kernels/bf16_attention.cu"),
                cudarc::nvrtc::CompileOptions {
                    arch: Some("compute_80"),
                    ..Default::default()
                },
            )
            .map_err(|e| cuda_error("compile BF16 attention", e))?;
            self.device
                .device
                .load_ptx(
                    ptx,
                    module,
                    &[
                        "native_attention_qk",
                        "native_attention_softmax",
                        "native_attention_pv",
                    ],
                )
                .map_err(|e| cuda_error("load BF16 attention", e))?;
        }
        let grow = self.attention.get(&key).map_or(true, |v| v.rows < rows);
        if grow {
            if let Some(old) = self.attention.remove(&key) {
                self.bytes -= old.bytes();
                drop(old);
            }
            if self.bytes + needed > self.budget || self.attention.len() >= 4 {
                self.clear();
            }
            let scratch = AttentionScratch {
                rows: capacity,
                packed: self.device.zeros_f32(input_capacity)?,
                scores: self.device.zeros_f32(score_capacity)?,
                probabilities: self.device.zeros_f32(score_capacity)?,
                stats: self.device.zeros_f32(stat_capacity)?,
                output: self.device.zeros_f32(qwidth)?,
            };
            self.bytes += scratch.bytes();
            self.peak = self.peak.max(self.bytes);
            self.attention.insert(key, scratch);
        }
        let scratch = self
            .attention
            .get_mut(&key)
            .expect("attention scratch inserted");
        let packed = [query, keys, values].concat();
        {
            let mut view = scratch.packed.data.slice_mut(..packed.len());
            self.device
                .device
                .htod_sync_copy_into(&packed, &mut view)
                .map_err(|e| cuda_error("BF16 attention upload", e))?;
            self.device
                .transfer_counters
                .record_host_to_device(packed.len() * 4);
        }
        let qk = self
            .device
            .device
            .get_func(module, "native_attention_qk")
            .ok_or_else(|| XrtError::Cuda("attention QK absent".into()))?;
        let softmax = self
            .device
            .device
            .get_func(module, "native_attention_softmax")
            .ok_or_else(|| XrtError::Cuda("attention softmax absent".into()))?;
        let pv = self
            .device
            .device
            .get_func(module, "native_attention_pv")
            .ok_or_else(|| XrtError::Cuda("attention PV absent".into()))?;
        let groups = (heads / kv).div_ceil(16) as u32;
        let h = heads as u32;
        let k = kv as u32;
        let d = dim as u32;
        let r = rows as u32;
        let blocks = r.div_ceil(128);
        // SAFETY: all packed, score, statistic and output extents were checked
        // against owned capacities. Each tile/head writes disjoint elements;
        // three ordered launches share one retained stream. Readback completes
        // all use before another operation may reuse or clear this workspace.
        unsafe {
            qk.launch(
                LaunchConfig {
                    grid_dim: (r.div_ceil(8), k, groups),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&scratch.packed.data, &mut scratch.scores.data, h, k, d, r),
            )
            .map_err(|e| cuda_error("BF16 attention QK", e))?;
            softmax
                .launch(
                    LaunchConfig {
                        grid_dim: (h, 1, 1),
                        block_dim: (16, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &scratch.scores.data,
                        &mut scratch.probabilities.data,
                        &mut scratch.stats.data,
                        r,
                        blocks,
                    ),
                )
                .map_err(|e| cuda_error("BF16 attention softmax", e))?;
            pv.launch(
                LaunchConfig {
                    grid_dim: (d.div_ceil(8), k, groups),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &scratch.packed.data,
                    &scratch.probabilities.data,
                    &scratch.stats.data,
                    &mut scratch.output.data,
                    h,
                    k,
                    d,
                    r,
                    blocks,
                ),
            )
            .map_err(|e| cuda_error("BF16 attention PV", e))?;
        }
        let out = self.device.download_f32(&scratch.output)?;
        if out.iter().any(|v| !v.is_finite()) {
            return Err(XrtError::InvalidTensor("BF16 attention overflow".into()));
        }
        Ok(out)
    }
}
