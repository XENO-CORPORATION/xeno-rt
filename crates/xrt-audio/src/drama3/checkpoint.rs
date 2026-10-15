use super::config::Config;
#[cfg(feature = "cuda")]
use std::collections::HashMap;
use std::path::Path;
use std::{collections::BTreeMap, sync::Mutex, time::Instant};
use xrt_core::{decode_bf16, decode_f16, Result, XrtError};
use xrt_kernels::cpu::{fp8::BlockFp8Matrix, linear_bf16, linear_f16, linear_f32_bytes};
use xrt_safetensors::{SafeTensorDType, SafeTensorStore};

pub struct Checkpoint {
    pub config: Config,
    pub tensors: SafeTensorStore,
    pub numerics: super::numerics::Numerics,
    #[cfg(feature = "cuda")]
    gpu_cache: Option<Mutex<ExpertCache>>,
    #[cfg(feature = "cuda")]
    dense_gpu: Option<DenseGpu>,
    profile: Mutex<BTreeMap<&'static str, ProjectionStats>>,
    #[cfg(feature = "drama3-diagnostics")]
    recorder: Mutex<Option<super::diagnostics::Recorder>>,
}
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct ProjectionStats {
    pub calls: u64,
    pub seconds: f64,
}
#[cfg(feature = "cuda")]
struct ExpertCache {
    device: xrt_cuda::GpuFp8Cache,
    sources: HashMap<String, xrt_cuda::PreparedFp8Matrix<xrt_safetensors::SafeTensorHandle>>,
}
#[cfg(feature = "cuda")]
impl ExpertCache {
    fn prepare(&mut self, tensors: &SafeTensorStore, name: &str) -> Result<()> {
        if self.sources.contains_key(name) {
            return Ok(());
        }
        let tensor = tensors.require_tensor(name)?;
        if tensor.info.dtype != SafeTensorDType::F8E4M3 || tensor.info.shape.len() != 2 {
            return Err(XrtError::InvalidTensor(
                "native expert source dtype/rank".into(),
            ));
        }
        let rows = tensor.info.shape[0];
        let cols = tensor.info.shape[1];
        let scale_name = name
            .strip_suffix(".weight")
            .ok_or_else(|| XrtError::InvalidTensor("native expert weight suffix".into()))?
            .to_owned()
            + ".weight_scale_inv";
        let scales = tensors.require_tensor(&scale_name)?;
        if scales.info.dtype != SafeTensorDType::F32
            || scales.info.shape != [rows.div_ceil(128), cols.div_ceil(128)]
        {
            return Err(XrtError::InvalidTensor(
                "native expert scale geometry".into(),
            ));
        }
        let source = xrt_cuda::PreparedFp8Matrix::new(
            tensors.tensor_handle(name)?,
            decode(scales.data, &scales.info.dtype)?,
            rows,
            cols,
            128,
            128,
        )?;
        self.sources.insert(name.to_owned(), source);
        Ok(())
    }
}
#[cfg(feature = "cuda")]
struct DenseGpu {
    matrices: HashMap<String, DenseMatrix>,
    workspace: Mutex<xrt_cuda::GpuDenseWorkspace>,
    bytes: u64,
}
#[cfg(feature = "cuda")]
enum DenseMatrix {
    Bf16(xrt_cuda::GpuDenseBf16Matrix),
    F32(xrt_cuda::GpuDenseF32Matrix),
}
impl Checkpoint {
    #[cfg(feature = "drama3-diagnostics")]
    pub fn begin_operator_trace(&self, bytes: usize) -> Result<()> {
        *self
            .recorder
            .lock()
            .map_err(|_| XrtError::Runtime("operator recorder lock poisoned".into()))? =
            Some(super::diagnostics::Recorder::new(bytes)?);
        Ok(())
    }
    #[cfg(feature = "drama3-diagnostics")]
    pub fn take_operator_trace(&self) -> Result<Vec<super::diagnostics::Operator>> {
        Ok(self
            .recorder
            .lock()
            .map_err(|_| XrtError::Runtime("operator recorder lock poisoned".into()))?
            .take()
            .map_or_else(Vec::new, |v| v.operators))
    }
    #[cfg(feature = "drama3-diagnostics")]
    pub(super) fn record(
        &self,
        kind: &'static str,
        name: &str,
        input: &[f32],
        auxiliary: &[f32],
        output: &[f32],
        parameters: serde_json::Value,
    ) -> Result<()> {
        if let Some(recorder) = self
            .recorder
            .lock()
            .map_err(|_| XrtError::Runtime("operator recorder lock poisoned".into()))?
            .as_mut()
        {
            recorder.push(kind, name, input, auxiliary, output, parameters)?;
        }
        Ok(())
    }
    pub fn norm(
        &self,
        input: &[f32],
        weight: &[f32],
        epsilon: f32,
        zero_centered: bool,
    ) -> Result<Vec<f32>> {
        let mut out = if self.numerics == super::numerics::Numerics::Bf16W8a8 {
            super::backbone::rmsnorm_stable(input, weight, epsilon, zero_centered)?
        } else {
            super::backbone::rmsnorm(input, weight, epsilon, zero_centered)?
        };
        self.numerics.round_checked(&mut out)?;
        #[cfg(feature = "drama3-diagnostics")]
        self.record(
            "norm",
            "",
            input,
            weight,
            &out,
            serde_json::json!({"epsilon":epsilon,"zero_centered":zero_centered}),
        )?;
        Ok(out)
    }
    pub fn mlp_hidden(&self, gate: &[f32], up: &[f32], routed: bool) -> Result<Vec<f32>> {
        if gate.len() != up.len() {
            return Err(XrtError::Shape("native MLP widths".into()));
        }
        let out = gate
            .iter()
            .zip(up)
            .map(|(g, u)| {
                let silu = *g / (1.0 + (-*g).exp());
                if routed && self.numerics == super::numerics::Numerics::Bf16W8a8 {
                    silu * u
                } else {
                    self.numerics.round(self.numerics.round(silu) * u)
                }
            })
            .collect::<Vec<_>>();
        #[cfg(feature = "drama3-diagnostics")]
        self.record(
            "silu_mul",
            "",
            gate,
            up,
            &out,
            serde_json::json!({"routed":routed}),
        )?;
        Ok(out)
    }
    pub fn attention_gate(&self, attended: &mut [f32], gate: &[f32]) -> Result<()> {
        #[cfg(feature = "drama3-diagnostics")]
        let before = attended.to_vec();
        if attended.len() != gate.len() {
            return Err(XrtError::Shape("native attention gate dimensions".into()));
        }
        for (x, g) in attended.iter_mut().zip(gate) {
            *x = self
                .numerics
                .round(self.numerics.round(*x) * self.numerics.round(1.0 / (1.0 + (-*g).exp())));
        }
        #[cfg(feature = "drama3-diagnostics")]
        self.record(
            "attention_gate",
            "",
            &before,
            gate,
            attended,
            serde_json::json!({}),
        )?;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub fn routed_experts(
        &self,
        prefix: &str,
        ids: &[usize],
        input: &[f32],
        probabilities: &[f32],
        denominator: f32,
    ) -> Result<Option<Vec<f32>>> {
        let Some(cache) = &self.gpu_cache else {
            return Ok(None);
        };
        let mut cache = cache
            .lock()
            .map_err(|_| XrtError::Runtime("FP8 cache lock poisoned".into()))?;
        let cfg = &self.config.text_config;
        let projection = cfg
            .hidden_size
            .checked_mul(cfg.moe_intermediate_size)
            .and_then(|n| {
                n.checked_add(
                    cfg.hidden_size.div_ceil(128) * cfg.moe_intermediate_size.div_ceil(128) * 4,
                )
            })
            .and_then(|n| n.checked_mul(3))
            .and_then(|n| n.checked_mul(ids.len()))
            .ok_or_else(|| XrtError::Shape("native selected expert bytes overflow".into()))?
            as u64;
        // Small caches retain the existing individual-projection path.
        if projection > cache.device.budget_bytes() {
            return Ok(None);
        }
        if ids.is_empty()
            || ids.len() > 8
            || ids.iter().any(|e| *e >= cfg.num_experts)
            || input.len() != cfg.hidden_size
        {
            return Err(XrtError::Shape("native routed expert model bounds".into()));
        }
        self.timed("fp8_moe_cuda", || {
            let mut names = Vec::with_capacity(ids.len());
            for id in ids {
                let triple = [
                    format!("{prefix}.experts.{id}.gate_proj.weight"),
                    format!("{prefix}.experts.{id}.up_proj.weight"),
                    format!("{prefix}.experts.{id}.down_proj.weight"),
                ];
                for name in &triple {
                    cache.prepare(&self.tensors, name)?;
                }
                names.push(triple);
            }
            let ExpertCache { device, sources } = &mut *cache;
            let selected: Vec<_> = names
                .iter()
                .map(|n| [&sources[&n[0]], &sources[&n[1]], &sources[&n[2]]])
                .collect();
            if self.numerics == super::numerics::Numerics::Bf16W8a8 {
                device
                    .routed_mlp_w8a8_prepared(&selected, input, probabilities, denominator)
                    .map(Some)
            } else {
                device
                    .routed_mlp_prepared(&selected, input, probabilities, denominator)
                    .map(Some)
            }
        })
    }
    pub fn values(&self, name: &str) -> Result<Vec<f32>> {
        let view = self.tensors.require_tensor(name)?;
        if view.data.len() > 64 * 1024 * 1024 {
            return Err(XrtError::Runtime(
                "small tensor expansion exceeds 64 MiB".into(),
            ));
        }
        decode(view.data, &view.info.dtype)
    }
    pub fn open(root: &Path) -> Result<Self> {
        let config_path = root.join("config.json");
        if std::fs::metadata(&config_path)?.len() > 1024 * 1024 {
            return Err(XrtError::InvalidMetadata(
                "Fish config exceeds bounds".into(),
            ));
        }
        let config = Config::parse(&std::fs::read(config_path)?)?;
        let tensors = SafeTensorStore::open(root)?;
        let model = Self {
            config,
            tensors,
            numerics: Default::default(),
            #[cfg(feature = "cuda")]
            gpu_cache: None,
            #[cfg(feature = "cuda")]
            dense_gpu: None,
            profile: Mutex::new(BTreeMap::new()),
            #[cfg(feature = "drama3-diagnostics")]
            recorder: Mutex::new(None),
        };
        model.validate_experts()?;
        Ok(model)
    }
    #[cfg(feature = "cuda")]
    pub fn enable_expert_cache(&mut self, device: usize, budget_bytes: u64) -> Result<()> {
        if budget_bytes == 0 {
            return Err(XrtError::Runtime(
                "FP8 cache budget must be positive".into(),
            ));
        }
        let device = xrt_cuda::CudaDevice::new(device)?;
        if budget_bytes > device.memory_info()?.0 {
            return Err(XrtError::Cuda(
                "FP8 cache budget exceeds currently free VRAM".into(),
            ));
        }
        self.gpu_cache = Some(Mutex::new(ExpertCache {
            device: xrt_cuda::GpuFp8Cache::new(device, budget_bytes),
            sources: HashMap::new(),
        }));
        Ok(())
    }
    pub fn projection_profile(&self) -> Result<BTreeMap<&'static str, ProjectionStats>> {
        self.profile
            .lock()
            .map(|v| v.clone())
            .map_err(|_| XrtError::Runtime("projection profile lock poisoned".into()))
    }
    pub fn reset_projection_profile(&self) -> Result<()> {
        self.profile
            .lock()
            .map_err(|_| XrtError::Runtime("projection profile lock poisoned".into()))?
            .clear();
        Ok(())
    }
    fn timed<T>(&self, stage: &'static str, work: impl FnOnce() -> Result<T>) -> Result<T> {
        let start = Instant::now();
        let result = work();
        let mut profile = self
            .profile
            .lock()
            .map_err(|_| XrtError::Runtime("projection profile lock poisoned".into()))?;
        let stats = profile.entry(stage).or_default();
        stats.calls += 1;
        stats.seconds += start.elapsed().as_secs_f64();
        result
    }
    #[cfg(feature = "cuda")]
    pub fn dense_cuda_plan(&self) -> Result<(Vec<String>, u64)> {
        let mut names = Vec::new();
        let mut required = 0u64;
        for name in self.tensors.tensor_names() {
            let info = self.tensors.tensor_info(name).expect("name from store");
            if info.shape.len() == 2
                && matches!(
                    info.dtype,
                    SafeTensorDType::Bf16 | SafeTensorDType::F16 | SafeTensorDType::F32
                )
                && name.ends_with(".weight")
                && !name.contains("embedding")
                && !name.contains("embed_tokens")
                && name != "lm_head.weight"
            {
                let bytes = (info.numel()? as u64)
                    .checked_mul(if info.dtype == SafeTensorDType::Bf16 {
                        2
                    } else {
                        4
                    })
                    .ok_or_else(|| XrtError::Runtime("dense CUDA byte size overflow".into()))?;
                required = required
                    .checked_add(bytes)
                    .ok_or_else(|| XrtError::Runtime("dense CUDA total overflow".into()))?;
                names.push(name.to_owned());
            }
        }
        Ok((names, required))
    }
    #[cfg(feature = "cuda")]
    pub fn enable_dense_cuda(&mut self, ordinal: usize, budget_bytes: u64) -> Result<u64> {
        let (names, required) = self.dense_cuda_plan()?;
        if required > budget_bytes {
            return Err(XrtError::Runtime(format!(
                "dense CUDA weights need {required} bytes, budget {budget_bytes}"
            )));
        }
        let device = xrt_cuda::CudaDevice::new(ordinal)?;
        if required > device.memory_info()?.0 {
            return Err(XrtError::Cuda(
                "dense CUDA residency exceeds free memory".into(),
            ));
        }
        let mut matrices = std::collections::HashMap::new();
        for name in names {
            let tensor = self.tensors.require_tensor(&name)?;
            let dtype = match tensor.info.dtype {
                SafeTensorDType::Bf16 => xrt_core::DType::BF16,
                SafeTensorDType::F16 => xrt_core::DType::F16,
                SafeTensorDType::F32 => xrt_core::DType::F32,
                _ => unreachable!(),
            };
            let matrix = if tensor.info.dtype == SafeTensorDType::Bf16 {
                DenseMatrix::Bf16(device.upload_dense_bf16_matrix(
                    tensor.data,
                    tensor.info.shape[0],
                    tensor.info.shape[1],
                    budget_bytes,
                )?)
            } else {
                DenseMatrix::F32(device.upload_dense_f32_matrix(
                    tensor.data,
                    dtype,
                    tensor.info.shape[0],
                    tensor.info.shape[1],
                    budget_bytes,
                )?)
            };
            matrices.insert(name, matrix);
        }
        self.dense_gpu = Some(DenseGpu {
            workspace: Mutex::new(xrt_cuda::GpuDenseWorkspace::new(
                device.clone(),
                4 * 1024 * 1024,
            )),
            matrices,
            bytes: required,
        });
        Ok(required)
    }
    #[cfg(feature = "cuda")]
    pub fn dense_resident_bytes(&self) -> u64 {
        self.dense_gpu.as_ref().map_or(0, |g| g.bytes)
    }
    pub fn linear_rows(
        &self,
        name: &str,
        input: &[f32],
        first: usize,
        count: usize,
    ) -> Result<Vec<f32>> {
        self.timed("selected_cpu", || {
            let cooked = if self.numerics == super::numerics::Numerics::Bf16W8a8 {
                Some(
                    input
                        .iter()
                        .map(|v| self.numerics.round(*v))
                        .collect::<Vec<_>>(),
                )
            } else {
                None
            };
            let mut out =
                self.linear_rows_untracked(name, cooked.as_deref().unwrap_or(input), first, count)?;
            self.numerics.round_checked(&mut out)?;
            #[cfg(feature = "drama3-diagnostics")]
            self.record(
                "linear_rows",
                name,
                input,
                &[],
                &out,
                serde_json::json!({"first":first,"count":count}),
            )?;
            Ok(out)
        })
    }
    fn linear_rows_untracked(
        &self,
        name: &str,
        input: &[f32],
        first: usize,
        count: usize,
    ) -> Result<Vec<f32>> {
        let tensor = self.tensors.require_tensor(name)?;
        let shape = &tensor.info.shape;
        let end = first
            .checked_add(count)
            .ok_or_else(|| XrtError::Shape("selected projection row overflow".into()))?;
        if shape.len() != 2 || shape[1] != input.len() || count == 0 || end > shape[0] {
            return Err(XrtError::Shape("selected projection row dimensions".into()));
        }
        let size = element_bytes(&tensor.info.dtype)?;
        let bytes = &tensor.data[first * shape[1] * size..end * shape[1] * size];
        let mut out = vec![0.0; count];
        match tensor.info.dtype {
            SafeTensorDType::Bf16 => {
                linear_bf16(input, 1, input.len(), bytes, count, None, &mut out)?
            }
            SafeTensorDType::F16 => {
                linear_f16(input, 1, input.len(), bytes, count, None, &mut out)?
            }
            SafeTensorDType::F32 => {
                linear_f32_bytes(input, 1, input.len(), bytes, count, None, &mut out)?
            }
            _ => return Err(XrtError::Unsupported("selected projection dtype".into())),
        };
        if out.iter().any(|x| !x.is_finite()) {
            return Err(XrtError::InvalidTensor(
                "non-finite selected projection".into(),
            ));
        }
        Ok(out)
    }
    #[cfg(feature = "cuda")]
    pub fn cache_stats(&self) -> Result<Option<xrt_cuda::Fp8CacheStats>> {
        self.gpu_cache
            .as_ref()
            .map(|c| {
                c.lock()
                    .map(|v| v.device.stats())
                    .map_err(|_| XrtError::Runtime("FP8 cache lock poisoned".into()))
            })
            .transpose()
    }
    pub fn vector(&self, name: &str) -> Result<Vec<f32>> {
        let view = self.tensors.require_tensor(name)?;
        if view.info.shape.len() != 1 {
            return Err(XrtError::Shape(format!("{name} is not a vector")));
        }
        self.values(name)
    }
    pub fn row(&self, name: &str, row: usize) -> Result<Vec<f32>> {
        let v = self.tensors.require_tensor(name)?;
        if v.info.shape.len() != 2 || row >= v.info.shape[0] {
            return Err(XrtError::Shape(format!("invalid embedding row for {name}")));
        }
        let cols = v.info.shape[1];
        let size = element_bytes(&v.info.dtype)?;
        decode(
            &v.data[row * cols * size..(row + 1) * cols * size],
            &v.info.dtype,
        )
    }
    pub fn linear(&self, name: &str, input: &[f32]) -> Result<Vec<f32>> {
        let stage = if self
            .tensors
            .tensor_info(name)
            .is_some_and(|v| v.dtype == SafeTensorDType::F8E4M3)
        {
            "fp8_cpu"
        } else {
            "dense_cpu"
        };
        #[cfg(feature = "cuda")]
        let stage = {
            let mut stage = stage;
            if stage == "fp8_cpu" && self.gpu_cache.is_some() {
                stage = "fp8_cuda";
            }
            if self
                .dense_gpu
                .as_ref()
                .is_some_and(|g| g.matrices.contains_key(name))
            {
                stage = "dense_cuda";
            }
            stage
        };
        self.timed(stage, || {
            let expert_down = name.contains(".mlp.experts.") && name.ends_with(".down_proj.weight");
            let cooked = if self.numerics == super::numerics::Numerics::Bf16W8a8 && !expert_down {
                Some(
                    input
                        .iter()
                        .map(|v| self.numerics.round(*v))
                        .collect::<Vec<_>>(),
                )
            } else {
                None
            };
            let mut result = self.linear_untracked(name, cooked.as_deref().unwrap_or(input))?;
            if !expert_down {
                self.numerics.round_checked(&mut result)?;
            }
            #[cfg(feature = "drama3-diagnostics")]
            self.record(
                "linear",
                name,
                input,
                &[],
                &result,
                serde_json::json!({"expert_down":expert_down}),
            )?;
            Ok(result)
        })
    }
    fn linear_untracked(&self, name: &str, input: &[f32]) -> Result<Vec<f32>> {
        #[cfg(feature = "cuda")]
        if let Some(gpu) = &self.dense_gpu {
            if let Some(matrix) = gpu.matrices.get(name) {
                let mut workspace = gpu
                    .workspace
                    .lock()
                    .map_err(|_| XrtError::Runtime("dense CUDA workspace lock poisoned".into()))?;
                return match matrix {
                    DenseMatrix::Bf16(m) => {
                        if cfg!(feature = "drama3-diagnostics")
                            && self.numerics == super::numerics::Numerics::Bf16W8a8
                            && std::env::var_os("XRT_DRAMA3_DIAGNOSTIC_CUBLAS")
                                .is_some_and(|v| v == "1")
                        {
                            workspace.matvec_bf16_activation(m, input)
                        } else {
                            workspace.matvec_bf16(m, input)
                        }
                    }
                    DenseMatrix::F32(m) => workspace.matvec_f32(m, input),
                };
            }
        }
        let v = self.tensors.require_tensor(name)?;
        if v.info.shape.len() != 2 || v.info.shape[1] != input.len() {
            return Err(XrtError::Shape(format!("linear dimensions for {name}")));
        }
        let rows = v.info.shape[0];
        let mut result = vec![0.0; rows];
        match &v.info.dtype {
            SafeTensorDType::Bf16 => {
                linear_bf16(input, 1, input.len(), v.data, rows, None, &mut result)?
            }
            SafeTensorDType::F16 => {
                linear_f16(input, 1, input.len(), v.data, rows, None, &mut result)?
            }
            SafeTensorDType::F32 => {
                linear_f32_bytes(input, 1, input.len(), v.data, rows, None, &mut result)?
            }
            SafeTensorDType::F8E4M3 => {
                let scale_name = name
                    .strip_suffix(".weight")
                    .ok_or_else(|| XrtError::InvalidTensor("FP8 weight suffix".into()))?
                    .to_owned()
                    + ".weight_scale_inv";
                let s = self.tensors.require_tensor(&scale_name)?;
                if s.info.dtype != SafeTensorDType::F32
                    || s.info.shape != [rows.div_ceil(128), input.len().div_ceil(128)]
                {
                    return Err(XrtError::Shape(format!("FP8 scale geometry for {name}")));
                }
                #[cfg(feature = "cuda")]
                if let Some(cache) = &self.gpu_cache {
                    let mut cache = cache
                        .lock()
                        .map_err(|_| XrtError::Runtime("FP8 cache lock poisoned".into()))?;
                    cache.prepare(&self.tensors, name)?;
                    let ExpertCache { device, sources } = &mut *cache;
                    return device.matvec_prepared(
                        &sources[name],
                        input,
                        self.numerics == super::numerics::Numerics::Bf16W8a8,
                    );
                }
                let scales = decode(s.data, &s.info.dtype)?;
                let matrix = BlockFp8Matrix::new(v.data, &scales, rows, input.len(), 128, 128)?;
                if self.numerics == super::numerics::Numerics::Bf16W8a8 {
                    matrix.matvec_w8a8(input, &mut result)?;
                } else {
                    matrix.matvec(input, &mut result)?;
                }
            }
            other => {
                return Err(XrtError::Unsupported(format!(
                    "Fish linear dtype {other:?}"
                )))
            }
        }
        if result.iter().any(|v| !v.is_finite()) {
            return Err(XrtError::InvalidTensor(format!(
                "non-finite output for {name}"
            )));
        }
        Ok(result)
    }
    fn validate_experts(&self) -> Result<()> {
        let t = &self.config.text_config;
        for layer in 0..t.num_hidden_layers {
            for expert in 0..t.num_experts {
                let prefix = format!("model.language_model.layers.{layer}.mlp.experts.{expert}");
                for (projection, rows, cols) in [
                    ("gate_proj", t.moe_intermediate_size, t.hidden_size),
                    ("up_proj", t.moe_intermediate_size, t.hidden_size),
                    ("down_proj", t.hidden_size, t.moe_intermediate_size),
                ] {
                    let name = format!("{prefix}.{projection}.weight");
                    let info = self
                        .tensors
                        .tensor_info(&name)
                        .ok_or_else(|| XrtError::InvalidTensor(format!("missing {name}")))?;
                    if info.shape != [rows, cols] || info.dtype != SafeTensorDType::F8E4M3 {
                        return Err(XrtError::Shape(format!("invalid FP8 expert {name}")));
                    }
                    let scale = format!("{prefix}.{projection}.weight_scale_inv");
                    let s = self
                        .tensors
                        .tensor_info(&scale)
                        .ok_or_else(|| XrtError::InvalidTensor(format!("missing {scale}")))?;
                    if s.dtype != SafeTensorDType::F32
                        || s.shape != [rows.div_ceil(128), cols.div_ceil(128)]
                    {
                        return Err(XrtError::Shape(format!("invalid FP8 expert scale {scale}")));
                    }
                }
            }
        }
        Ok(())
    }
}
fn element_bytes(dtype: &SafeTensorDType) -> Result<usize> {
    match dtype {
        SafeTensorDType::F16 | SafeTensorDType::Bf16 => Ok(2),
        SafeTensorDType::F32 => Ok(4),
        d => Err(XrtError::Unsupported(format!("vector dtype {d:?}"))),
    }
}
fn decode(data: &[u8], dtype: &SafeTensorDType) -> Result<Vec<f32>> {
    let size = element_bytes(dtype)?;
    if data.len() % size != 0 {
        return Err(XrtError::InvalidTensor("truncated floating tensor".into()));
    }
    let out: Result<Vec<_>> = data
        .chunks_exact(size)
        .map(|b| match dtype {
            SafeTensorDType::Bf16 => decode_bf16(b),
            SafeTensorDType::F16 => decode_f16(b),
            SafeTensorDType::F32 => Ok(f32::from_le_bytes(b.try_into().unwrap())),
            _ => unreachable!(),
        })
        .collect();
    let out = out?;
    if out.iter().any(|x| !x.is_finite()) {
        return Err(XrtError::InvalidTensor("non-finite tensor".into()));
    }
    Ok(out)
}
