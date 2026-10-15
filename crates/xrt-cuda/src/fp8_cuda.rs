use super::*;
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap};
use xrt_kernels::cpu::fp8::{quantize_e4m3fn, BlockFp8Matrix, E4M3FN};

impl CudaDevice {
    // cudarc htod_copy(Vec) retains that host Vec for the entire CudaSlice
    // lifetime. Immutable mapped weights already have an owner: copy borrowed
    // bytes synchronously instead of retaining another device-sized host copy.
    fn upload_native_bytes(&self, values: &[u8]) -> Result<CudaBytes> {
        let data = self
            .device
            .htod_sync_copy(values)
            .map_err(|e| cuda_error("native borrowed byte upload", e))?;
        self.transfer_counters.record_host_to_device(values.len());
        let allocation = self.track_allocation(values.len());
        Ok(CudaBytes {
            data,
            len: values.len(),
            _allocation: allocation,
        })
    }
    fn upload_native_f32(&self, values: &[f32]) -> Result<CudaF32Buffer> {
        let data = self
            .device
            .htod_sync_copy(values)
            .map_err(|e| cuda_error("native borrowed f32 upload", e))?;
        let bytes = std::mem::size_of_val(values);
        self.transfer_counters.record_host_to_device(bytes);
        let allocation = self.track_allocation(bytes);
        Ok(CudaF32Buffer {
            data,
            len: values.len(),
            _allocation: allocation,
        })
    }
}

#[derive(Debug)]
pub struct GpuDenseF32Matrix {
    owner: Arc<DriverCudaDevice>,
    values: CudaF32Buffer,
    rows: usize,
    cols: usize,
}
impl GpuDenseF32Matrix {
    pub fn resident_bytes(&self) -> u64 {
        self.values.byte_len() as u64
    }
}
#[derive(Debug)]
pub struct GpuDenseBf16Matrix {
    owner: Arc<DriverCudaDevice>,
    weights: CudaBytes,
    rows: usize,
    cols: usize,
}
impl GpuDenseBf16Matrix {
    pub fn resident_bytes(&self) -> u64 {
        self.weights.byte_len() as u64
    }
}
struct DenseScratch {
    input: CudaF32Buffer,
    output: CudaF32Buffer,
}
/// Exclusive, bounded reusable projection buffers. Dense weights stay owned
/// by their matrices; these buffers contain request activations only.
pub struct GpuDenseWorkspace {
    device: CudaDevice,
    budget: u64,
    bytes: u64,
    peak: u64,
    scratch: HashMap<(usize, usize), DenseScratch>,
}
impl GpuDenseWorkspace {
    pub fn new(device: CudaDevice, budget: u64) -> Self {
        Self {
            device,
            budget,
            bytes: 0,
            peak: 0,
            scratch: HashMap::new(),
        }
    }
    pub fn resident_bytes(&self) -> u64 {
        self.bytes
    }
    pub fn peak_bytes(&self) -> u64 {
        self.peak
    }
    pub fn clear(&mut self) {
        self.scratch.clear();
        self.bytes = 0;
    }
    fn prepare(&mut self, rows: usize, cols: usize) -> Result<()> {
        let bytes = rows
            .checked_add(cols)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| XrtError::Shape("dense workspace bytes overflow".into()))?
            as u64;
        if bytes > self.budget {
            return Err(XrtError::Cuda("dense workspace exceeds budget".into()));
        }
        if !self.scratch.contains_key(&(rows, cols)) {
            if self.scratch.len() == 32 || self.bytes > self.budget - bytes {
                self.clear();
            }
            let scratch = self.device.dense_scratch(rows, cols)?;
            self.bytes += bytes;
            self.peak = self.peak.max(self.bytes);
            self.scratch.insert((rows, cols), scratch);
        }
        Ok(())
    }
    pub fn matvec_bf16(&mut self, matrix: &GpuDenseBf16Matrix, input: &[f32]) -> Result<Vec<f32>> {
        self.device
            .validate_dense_input(&matrix.owner, matrix.cols, input)?;
        self.prepare(matrix.rows, matrix.cols)?;
        self.device.matvec_dense_bf16_scratch(
            matrix,
            input,
            self.scratch
                .get_mut(&(matrix.rows, matrix.cols))
                .expect("dense workspace inserted"),
        )
    }
    pub fn matvec_f32(&mut self, matrix: &GpuDenseF32Matrix, input: &[f32]) -> Result<Vec<f32>> {
        self.device
            .validate_dense_input(&matrix.owner, matrix.cols, input)?;
        self.prepare(matrix.rows, matrix.cols)?;
        self.device.matvec_dense_f32_scratch(
            matrix,
            input,
            self.scratch
                .get_mut(&(matrix.rows, matrix.cols))
                .expect("dense workspace inserted"),
        )
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Fp8CacheStats {
    pub resident_bytes: u64,
    pub peak_resident_bytes: u64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub workspace_bytes: u64,
    pub peak_workspace_bytes: u64,
    pub upload_seconds: f64,
    pub eviction_seconds: f64,
    pub execution_seconds: f64,
    pub reused_allocations: u64,
}
const FP8_WORKSPACE_LIMIT: u64 = 4 * 1024 * 1024;
struct Fp8Scratch {
    input: CudaF32Buffer,
    scales: CudaF32Buffer,
    output: CudaF32Buffer,
}
struct MoeScratch {
    pointers: CudaBytes,
    input: CudaF32Buffer,
    hidden: CudaF32Buffer,
    probabilities: CudaF32Buffer,
    output: CudaF32Buffer,
}
impl MoeScratch {
    fn bytes(&self) -> u64 {
        (self.pointers.byte_len()
            + self.input.byte_len()
            + self.hidden.byte_len()
            + self.probabilities.byte_len()
            + self.output.byte_len()) as u64
    }
}
impl Fp8Scratch {
    fn bytes(&self) -> u64 {
        (self.input.byte_len() + self.scales.byte_len() + self.output.byte_len()) as u64
    }
}
mod source_seal {
    pub trait Storage {}
    impl Storage for Vec<u8> {}
    impl Storage for Box<[u8]> {}
    impl Storage for std::sync::Arc<[u8]> {}
    impl Storage for &[u8] {}
    impl Storage for xrt_safetensors::SafeTensorHandle {}
}
/// Sealed to stable, immutable byte storage. An arbitrary AsRef implementation
/// with interior switching/mutation cannot bypass the content identity.
pub trait Fp8SourceBytes: source_seal::Storage + AsRef<[u8]> {}
impl<T: source_seal::Storage + AsRef<[u8]>> Fp8SourceBytes for T {}
/// A validated, content-bound source retained across decode steps. Storage must
/// remain immutable for the handle's lifetime (including mapped backing files).
/// The constructor scans/hashes once; cache hits only inspect the private key.
pub struct PreparedFp8Matrix<S: Fp8SourceBytes> {
    storage: S,
    scales: Vec<f32>,
    rows: usize,
    cols: usize,
    br: usize,
    bc: usize,
    key: [u8; 32],
    bytes: u64,
}
impl<S: Fp8SourceBytes> PreparedFp8Matrix<S> {
    pub fn new(
        storage: S,
        scales: Vec<f32>,
        rows: usize,
        cols: usize,
        br: usize,
        bc: usize,
    ) -> Result<Self> {
        let matrix = BlockFp8Matrix::new(storage.as_ref(), &scales, rows, cols, br, bc)?;
        let bytes = matrix.resident_bytes() as u64;
        let mut hash = Sha256::new();
        for value in [rows, cols, br, bc] {
            hash.update((value as u64).to_le_bytes());
        }
        hash.update(storage.as_ref());
        for s in &scales {
            hash.update(s.to_bits().to_le_bytes());
        }
        Ok(Self {
            storage,
            scales,
            rows,
            cols,
            br,
            bc,
            key: hash.finalize().into(),
            bytes,
        })
    }
}
pub struct GpuFp8Cache {
    device: CudaDevice,
    budget: u64,
    tick: u64,
    stats: Fp8CacheStats,
    entries: HashMap<[u8; 32], (GpuBlockFp8Matrix, u64)>,
    ages: BTreeSet<(u64, [u8; 32])>,
    scratch: HashMap<(usize, usize, usize), Fp8Scratch>,
    moe_scratch: Option<(usize, usize, usize, MoeScratch)>,
}
impl GpuFp8Cache {
    pub fn new(device: CudaDevice, budget: u64) -> Self {
        Self {
            device,
            budget,
            tick: 0,
            stats: Default::default(),
            entries: HashMap::new(),
            ages: BTreeSet::new(),
            scratch: HashMap::new(),
            moe_scratch: None,
        }
    }
    pub fn stats(&self) -> Fp8CacheStats {
        self.stats
    }
    pub fn budget_bytes(&self) -> u64 {
        self.budget
    }
    pub fn clear(&mut self) {
        self.entries.clear();
        self.ages.clear();
        self.stats.resident_bytes = 0;
        self.scratch.clear();
        self.moe_scratch = None;
        self.stats.workspace_bytes = 0;
    }
    /// Content-bound LRU. A caller cannot alias changed weights under a reused
    /// tensor name. The exclusive borrow pins the active matrix through launch.
    pub fn matvec(
        &mut self,
        bytes: &[u8],
        scales: &[f32],
        rows: usize,
        cols: usize,
        br: usize,
        bc: usize,
        input: &[f32],
        w8a8: bool,
    ) -> Result<Vec<f32>> {
        let source = PreparedFp8Matrix::new(bytes, scales.to_vec(), rows, cols, br, bc)?;
        self.matvec_prepared(&source, input, w8a8)
    }
    fn ensure_resident<S: Fp8SourceBytes>(&mut self, source: &PreparedFp8Matrix<S>) -> Result<()> {
        let size = source.bytes;
        if size > self.budget {
            return Err(XrtError::Cuda(format!(
                "expert needs {size} bytes, cache budget {}",
                self.budget
            )));
        }
        let key = source.key;
        self.tick = self
            .tick
            .checked_add(1)
            .ok_or_else(|| XrtError::Runtime("FP8 cache age overflow".into()))?;
        if !self.entries.contains_key(&key) {
            self.stats.misses += 1;
            let start = std::time::Instant::now();
            let mut reusable = None;
            while self.stats.resident_bytes > self.budget - size {
                let (_, oldest) = self
                    .ages
                    .pop_first()
                    .ok_or_else(|| XrtError::Runtime("FP8 cache accounting inconsistent".into()))?;
                let (victim, _) = self
                    .entries
                    .remove(&oldest)
                    .expect("key selected from cache");
                self.stats.resident_bytes -= victim.resident_bytes();
                if reusable.is_none()
                    && victim.weights.len() == source.storage.as_ref().len()
                    && victim.scales.len() == source.scales.len()
                {
                    reusable = Some(victim);
                } else {
                    drop(victim);
                }
                self.stats.evictions += 1;
            }
            self.stats.eviction_seconds += start.elapsed().as_secs_f64();
            let start = std::time::Instant::now();
            let gpu = if let Some(mut gpu) = reusable {
                self.device
                    .device
                    .htod_sync_copy_into(source.storage.as_ref(), &mut gpu.weights.data)
                    .map_err(|e| cuda_error("restage native FP8 weight allocation", e))?;
                self.device
                    .transfer_counters
                    .record_host_to_device(source.storage.as_ref().len());
                self.device
                    .upload_f32_into(&source.scales, &mut gpu.scales)?;
                gpu.rows = source.rows;
                gpu.cols = source.cols;
                gpu.block_rows = source.br;
                gpu.block_cols = source.bc;
                self.stats.reused_allocations += 1;
                gpu
            } else {
                // Prepared sources already checked every encoded byte, scale
                // and dimension. A cache miss must not rescan immutable data.
                if source.bytes > self.device.memory_info()?.0 {
                    return Err(XrtError::Cuda(
                        "insufficient native FP8 device memory".into(),
                    ));
                }
                GpuBlockFp8Matrix {
                    owner: self.device.device.clone(),
                    weights: self.device.upload_native_bytes(source.storage.as_ref())?,
                    scales: self.device.upload_native_f32(&source.scales)?,
                    rows: source.rows,
                    cols: source.cols,
                    block_rows: source.br,
                    block_cols: source.bc,
                }
            };
            self.stats.resident_bytes += gpu.resident_bytes();
            self.stats.peak_resident_bytes = self
                .stats
                .peak_resident_bytes
                .max(self.stats.resident_bytes);
            self.entries.insert(key, (gpu, self.tick));
            self.stats.upload_seconds += start.elapsed().as_secs_f64();
        } else {
            self.stats.hits += 1;
        }
        let (_, age) = self.entries.get_mut(&key).expect("entry inserted or hit");
        self.ages.remove(&(*age, key));
        *age = self.tick;
        self.ages.insert((self.tick, key));
        Ok(())
    }
    /// Two native launches for a routed FP8 gated MLP. The exclusive cache
    /// borrow and whole-selected-set budget keep every referenced allocation
    /// pinned through the synchronized output. Only FP32 activation precision
    /// is implemented here; SiLU uses CUDA's approximate exponential.
    pub fn routed_mlp_prepared<S: Fp8SourceBytes>(
        &mut self,
        experts: &[[&PreparedFp8Matrix<S>; 3]],
        input: &[f32],
        probabilities: &[f32],
        denominator: f32,
    ) -> Result<Vec<f32>> {
        let count = experts.len();
        if count == 0
            || count > 8
            || probabilities.len() != count
            || input.is_empty()
            || input.iter().any(|x| !x.is_finite())
            || probabilities.iter().any(|p| !p.is_finite() || *p < 0.0)
            || !denominator.is_finite()
            || denominator <= 0.0
        {
            return Err(XrtError::Shape(
                "native routed MLP input/routing bounds".into(),
            ));
        }
        let hidden = input.len();
        let middle = experts[0][0].rows;
        let mut selected = HashMap::new();
        for triple in experts {
            for (i, source) in triple.iter().enumerate() {
                let shape = if i == 2 {
                    (hidden, middle)
                } else {
                    (middle, hidden)
                };
                if (source.rows, source.cols) != shape || source.br != 128 || source.bc != 128 {
                    return Err(XrtError::Shape(
                        "native routed MLP projection layout".into(),
                    ));
                }
                selected.insert(source.key, source.bytes);
            }
        }
        let selected_bytes = selected.values().try_fold(0u64, |n, b| {
            n.checked_add(*b)
                .ok_or_else(|| XrtError::Runtime("routed MLP resident sum overflow".into()))
        })?;
        if selected_bytes > self.budget {
            return Err(XrtError::Cuda(
                "selected expert set exceeds native cache budget".into(),
            ));
        }
        // Touching all selected keys places them ahead of every unselected key.
        // The selected-set check guarantees staging cannot evict one of them.
        for triple in experts {
            for source in triple {
                self.ensure_resident(source)?;
            }
        }
        let mut pointers = Vec::with_capacity(count * 6);
        for kind in 0..6 {
            for triple in experts {
                let entry = &self.entries[&triple[kind % 3].key].0;
                pointers.push(if kind < 3 {
                    *entry.weights.data.device_ptr()
                } else {
                    *entry.scales.data.device_ptr()
                });
            }
        }
        let needed = hidden
            .checked_mul(8)
            .and_then(|n| {
                middle
                    .checked_mul(count)
                    .and_then(|m| m.checked_mul(4))
                    .and_then(|m| n.checked_add(m))
            })
            .and_then(|n| n.checked_add(count * 52))
            .ok_or_else(|| XrtError::Shape("native MLP workspace size overflow".into()))?
            as u64;
        if needed > FP8_WORKSPACE_LIMIT {
            return Err(XrtError::Cuda(
                "native MLP workspace exceeds 4 MiB limit".into(),
            ));
        }
        let same = self
            .moe_scratch
            .as_ref()
            .is_some_and(|(n, h, m, _)| (*n, *h, *m) == (count, hidden, middle));
        if !same {
            self.scratch.clear();
            self.moe_scratch = None;
            self.stats.workspace_bytes = 0;
            let scratch = MoeScratch {
                pointers: self.device.upload_native_bytes(&vec![0u8; count * 6 * 8])?,
                input: self.device.zeros_f32(hidden)?,
                hidden: self.device.zeros_f32(count * middle)?,
                probabilities: self.device.zeros_f32(count)?,
                output: self.device.zeros_f32(hidden)?,
            };
            self.stats.workspace_bytes = scratch.bytes();
            self.stats.peak_workspace_bytes = self.stats.peak_workspace_bytes.max(scratch.bytes());
            self.moe_scratch = Some((count, hidden, middle, scratch));
        }
        let start = std::time::Instant::now();
        let scratch = &mut self
            .moe_scratch
            .as_mut()
            .expect("MLP workspace initialized")
            .3;
        let bytes: Vec<u8> = pointers.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.device
            .device
            .htod_sync_copy_into(&bytes, &mut scratch.pointers.data)
            .map_err(|e| cuda_error("stage routed MLP pointers", e))?;
        self.device
            .transfer_counters
            .record_host_to_device(bytes.len());
        self.device.upload_f32_into(input, &mut scratch.input)?;
        self.device
            .upload_f32_into(probabilities, &mut scratch.probabilities)?;
        self.device.load_native_matvec_kernels()?;
        let n = to_u32(count, "MLP experts")?;
        let h = to_u32(hidden, "MLP hidden")?;
        let m = to_u32(middle, "MLP intermediate")?;
        let gate = self
            .device
            .device
            .get_func("xrt_block_fp8", "fp8_moe_gate_up")
            .ok_or_else(|| XrtError::Cuda("native MLP gate kernel absent".into()))?;
        let down = self
            .device
            .device
            .get_func("xrt_block_fp8", "fp8_moe_down_sum")
            .ok_or_else(|| XrtError::Cuda("native MLP down kernel absent".into()))?;
        // SAFETY: all pointers come from dimension/context-checked live entries
        // pinned by this exclusive borrow. Launches share a stream; download
        // synchronizes before workspace or selected entries can be changed.
        unsafe {
            gate.launch(
                LaunchConfig {
                    grid_dim: (m, n, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &scratch.pointers.data,
                    &scratch.input.data,
                    &mut scratch.hidden.data,
                    n,
                    h,
                    m,
                ),
            )
            .map_err(|e| cuda_error("native routed MLP gate/up", e))?;
            down.launch(
                LaunchConfig {
                    grid_dim: (h, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &scratch.pointers.data,
                    &scratch.hidden.data,
                    &scratch.probabilities.data,
                    &mut scratch.output.data,
                    n,
                    h,
                    m,
                    denominator,
                ),
            )
            .map_err(|e| cuda_error("native routed MLP down/sum", e))?;
        }
        let result = self.device.download_f32(&scratch.output)?;
        self.stats.execution_seconds += start.elapsed().as_secs_f64();
        if result.iter().any(|x| !x.is_finite()) {
            return Err(XrtError::InvalidTensor(
                "native routed MLP result non-finite".into(),
            ));
        }
        Ok(result)
    }
    pub fn matvec_prepared<S: Fp8SourceBytes>(
        &mut self,
        source: &PreparedFp8Matrix<S>,
        input: &[f32],
        w8a8: bool,
    ) -> Result<Vec<f32>> {
        if input.len() != source.cols || input.iter().any(|x| !x.is_finite()) {
            return Err(XrtError::Shape("FP8 cache activation dimensions".into()));
        }
        self.ensure_resident(source)?;
        if let Some((_, _, _, scratch)) = self.moe_scratch.take() {
            self.stats.workspace_bytes -= scratch.bytes();
        }
        let (entry, _) = self
            .entries
            .get(&source.key)
            .expect("prepared expert is resident");
        let scratch_key = (source.rows, source.cols, source.cols.div_ceil(source.bc));
        if !self.scratch.contains_key(&scratch_key) {
            let bytes = (source.rows as u64)
                .checked_add(source.cols as u64)
                .and_then(|n| n.checked_add(scratch_key.2 as u64))
                .and_then(|n| n.checked_mul(4))
                .ok_or_else(|| XrtError::Shape("FP8 workspace byte overflow".into()))?;
            if bytes > FP8_WORKSPACE_LIMIT {
                return Err(XrtError::Cuda("FP8 workspace exceeds 4 MiB limit".into()));
            }
            if self.scratch.len() == 8 || self.stats.workspace_bytes > FP8_WORKSPACE_LIMIT - bytes {
                self.scratch.clear();
                self.stats.workspace_bytes = 0;
            }
            let scratch = self
                .device
                .fp8_scratch(source.rows, source.cols, scratch_key.2)?;
            self.stats.workspace_bytes += scratch.bytes();
            self.stats.peak_workspace_bytes = self
                .stats
                .peak_workspace_bytes
                .max(self.stats.workspace_bytes);
            self.scratch.insert(scratch_key, scratch);
        }
        let start = std::time::Instant::now();
        let result = self.device.matvec_block_fp8_scratch(
            entry,
            input,
            w8a8,
            self.scratch
                .get_mut(&scratch_key)
                .expect("workspace inserted"),
        );
        self.stats.execution_seconds += start.elapsed().as_secs_f64();
        result
    }
}

#[derive(Debug)]
pub struct GpuBlockFp8Matrix {
    owner: Arc<DriverCudaDevice>,
    weights: CudaBytes,
    scales: CudaF32Buffer,
    rows: usize,
    cols: usize,
    block_rows: usize,
    block_cols: usize,
}
impl GpuBlockFp8Matrix {
    pub fn resident_bytes(&self) -> u64 {
        (self.weights.byte_len() + self.scales.byte_len()) as u64
    }
}

impl CudaDevice {
    fn validate_dense_input(
        &self,
        owner: &Arc<DriverCudaDevice>,
        cols: usize,
        input: &[f32],
    ) -> Result<()> {
        if !Arc::ptr_eq(&self.device, owner)
            || input.len() != cols
            || input.iter().any(|x| !x.is_finite())
        {
            return Err(XrtError::Shape("dense CUDA context/input mismatch".into()));
        }
        Ok(())
    }
    fn dense_scratch(&self, rows: usize, cols: usize) -> Result<DenseScratch> {
        Ok(DenseScratch {
            input: self.zeros_f32(cols)?,
            output: self.zeros_f32(rows)?,
        })
    }
    fn load_native_matvec_kernels(&self) -> Result<()> {
        let module = "xrt_block_fp8";
        if self.device.get_func(module, "dense_f32_matvec").is_none() {
            let _library = super::native_libraries::nvrtc()?;
            let ptx = cudarc::nvrtc::compile_ptx(include_str!("kernels/block_fp8.cu"))
                .map_err(|e| cuda_error("compile native matvec kernels", e))?;
            self.device
                .load_ptx(
                    ptx,
                    module,
                    &[
                        "block_fp8_matvec",
                        "dense_f32_matvec",
                        "dense_bf16_matvec",
                        "fp8_moe_gate_up",
                        "fp8_moe_down_sum",
                    ],
                )
                .map_err(|e| cuda_error("load native matvec kernels", e))?;
        }
        Ok(())
    }
    pub fn upload_dense_bf16_matrix(
        &self,
        bytes: &[u8],
        rows: usize,
        cols: usize,
        budget: u64,
    ) -> Result<GpuDenseBf16Matrix> {
        let elements = checked_mul(rows, cols, "BF16 CUDA matrix elements")?;
        let size = checked_mul(elements, 2, "BF16 CUDA matrix bytes")?;
        if elements == 0 || bytes.len() != size || size as u64 > budget {
            return Err(XrtError::Shape(
                "BF16 CUDA matrix storage/budget mismatch".into(),
            ));
        }
        if bytes
            .chunks_exact(2)
            .any(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7f80 == 0x7f80)
        {
            return Err(XrtError::InvalidTensor(
                "non-finite BF16 CUDA matrix".into(),
            ));
        }
        if size as u64 > self.memory_info()?.0 {
            return Err(XrtError::Cuda(
                "BF16 dense matrix exceeds free memory".into(),
            ));
        }
        Ok(GpuDenseBf16Matrix {
            owner: self.device.clone(),
            weights: self.upload_native_bytes(bytes)?,
            rows,
            cols,
        })
    }
    pub fn matvec_dense_bf16(
        &self,
        matrix: &GpuDenseBf16Matrix,
        input: &[f32],
    ) -> Result<Vec<f32>> {
        self.validate_dense_input(&matrix.owner, matrix.cols, input)?;
        let mut scratch = self.dense_scratch(matrix.rows, matrix.cols)?;
        self.matvec_dense_bf16_scratch(matrix, input, &mut scratch)
    }
    fn matvec_dense_bf16_scratch(
        &self,
        matrix: &GpuDenseBf16Matrix,
        input: &[f32],
        scratch: &mut DenseScratch,
    ) -> Result<Vec<f32>> {
        if !Arc::ptr_eq(&self.device, &matrix.owner)
            || input.len() != matrix.cols
            || input.iter().any(|x| !x.is_finite())
        {
            return Err(XrtError::Shape(
                "BF16 CUDA matrix context/input mismatch".into(),
            ));
        }
        self.load_native_matvec_kernels()?;
        self.upload_f32_into(input, &mut scratch.input)?;
        let rows = to_u32(matrix.rows, "BF16 CUDA rows")?;
        let cols = to_u32(matrix.cols, "BF16 CUDA cols")?;
        let function = self
            .device
            .get_func("xrt_block_fp8", "dense_bf16_matvec")
            .ok_or_else(|| XrtError::Cuda("BF16 CUDA kernel absent".into()))?;
        // SAFETY: the encoded two-byte storage is dimension/context checked and
        // allocations remain alive through synchronized output download.
        unsafe {
            function.launch(
                LaunchConfig {
                    grid_dim: (rows, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &matrix.weights.data,
                    &scratch.input.data,
                    &mut scratch.output.data,
                    rows,
                    cols,
                ),
            )
        }
        .map_err(|e| cuda_error("BF16 CUDA matvec", e))?;
        let out = self.download_f32(&scratch.output)?;
        if out.iter().any(|x| !x.is_finite()) {
            return Err(XrtError::InvalidTensor(
                "BF16 CUDA output non-finite".into(),
            ));
        }
        Ok(out)
    }
    pub fn upload_dense_f32_matrix(
        &self,
        bytes: &[u8],
        dtype: DType,
        rows: usize,
        cols: usize,
        budget: u64,
    ) -> Result<GpuDenseF32Matrix> {
        if rows == 0 || cols == 0 {
            return Err(XrtError::Shape(
                "dense CUDA matrix dimensions must be positive".into(),
            ));
        }
        let elements = checked_mul(rows, cols, "dense CUDA matrix elements")?;
        let size = checked_mul(elements, 4, "dense CUDA matrix bytes")? as u64;
        if size > budget || size > self.memory_info()?.0 {
            return Err(XrtError::Cuda(
                "dense CUDA matrix exceeds resident budget/free memory".into(),
            ));
        }
        let values = decode_float_tensor_bytes(bytes, "native dense matrix", dtype, elements)?;
        if values.iter().any(|x| !x.is_finite()) {
            return Err(XrtError::InvalidTensor(
                "non-finite dense CUDA weights".into(),
            ));
        }
        Ok(GpuDenseF32Matrix {
            owner: self.device.clone(),
            values: self.upload_native_f32(&values)?,
            rows,
            cols,
        })
    }
    pub fn matvec_dense_f32(&self, matrix: &GpuDenseF32Matrix, input: &[f32]) -> Result<Vec<f32>> {
        self.validate_dense_input(&matrix.owner, matrix.cols, input)?;
        let mut scratch = self.dense_scratch(matrix.rows, matrix.cols)?;
        self.matvec_dense_f32_scratch(matrix, input, &mut scratch)
    }
    fn matvec_dense_f32_scratch(
        &self,
        matrix: &GpuDenseF32Matrix,
        input: &[f32],
        scratch: &mut DenseScratch,
    ) -> Result<Vec<f32>> {
        if !Arc::ptr_eq(&self.device, &matrix.owner)
            || input.len() != matrix.cols
            || input.iter().any(|x| !x.is_finite())
        {
            return Err(XrtError::Shape(
                "dense CUDA matrix context/input mismatch".into(),
            ));
        }
        self.load_native_matvec_kernels()?;
        self.upload_f32_into(input, &mut scratch.input)?;
        let rows = to_u32(matrix.rows, "dense CUDA rows")?;
        let cols = to_u32(matrix.cols, "dense CUDA columns")?;
        let function = self
            .device
            .get_func("xrt_block_fp8", "dense_f32_matvec")
            .ok_or_else(|| XrtError::Cuda("dense CUDA kernel absent".into()))?;
        // SAFETY: owner, dimensions and retained buffers are validated above;
        // synchronized download settles the operation before temporaries drop.
        unsafe {
            function.launch(
                LaunchConfig {
                    grid_dim: (rows, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &matrix.values.data,
                    &scratch.input.data,
                    &mut scratch.output.data,
                    rows,
                    cols,
                ),
            )
        }
        .map_err(|e| cuda_error("native dense CUDA matvec", e))?;
        let out = self.download_f32(&scratch.output)?;
        if out.iter().any(|x| !x.is_finite()) {
            return Err(XrtError::InvalidTensor(
                "dense CUDA result is non-finite".into(),
            ));
        }
        Ok(out)
    }
    pub fn upload_block_fp8(
        &self,
        bytes: &[u8],
        scales: &[f32],
        rows: usize,
        cols: usize,
        block_rows: usize,
        block_cols: usize,
        budget_bytes: u64,
    ) -> Result<GpuBlockFp8Matrix> {
        let checked = BlockFp8Matrix::new(bytes, scales, rows, cols, block_rows, block_cols)?;
        let resident = checked.resident_bytes() as u64;
        if resident > budget_bytes {
            return Err(XrtError::Cuda(format!(
                "FP8 resident bytes {resident} exceed budget {budget_bytes}"
            )));
        }
        if resident > self.memory_info()?.0 {
            return Err(XrtError::Cuda(
                "insufficient CUDA memory for FP8 matrix".into(),
            ));
        }
        Ok(GpuBlockFp8Matrix {
            owner: self.device.clone(),
            weights: self.upload_native_bytes(bytes)?,
            scales: self.upload_native_f32(scales)?,
            rows,
            cols,
            block_rows,
            block_cols,
        })
    }

    /// Decode validation kernel. Kept separate from future Tensor Core GEMM;
    /// it establishes native execution and a measurable numerical reference.
    pub fn matvec_block_fp8(
        &self,
        matrix: &GpuBlockFp8Matrix,
        input: &[f32],
        w8a8: bool,
    ) -> Result<Vec<f32>> {
        let mut scratch = self.fp8_scratch(
            matrix.rows,
            matrix.cols,
            matrix.cols.div_ceil(matrix.block_cols),
        )?;
        self.matvec_block_fp8_scratch(matrix, input, w8a8, &mut scratch)
    }
    fn fp8_scratch(&self, rows: usize, cols: usize, scales: usize) -> Result<Fp8Scratch> {
        Ok(Fp8Scratch {
            input: self.zeros_f32(cols)?,
            scales: self.zeros_f32(scales)?,
            output: self.zeros_f32(rows)?,
        })
    }
    fn matvec_block_fp8_scratch(
        &self,
        matrix: &GpuBlockFp8Matrix,
        input: &[f32],
        w8a8: bool,
        scratch: &mut Fp8Scratch,
    ) -> Result<Vec<f32>> {
        if !Arc::ptr_eq(&self.device, &matrix.owner) {
            return Err(XrtError::Cuda(
                "FP8 matrix belongs to another CUDA context".into(),
            ));
        }
        if input.len() != matrix.cols || input.iter().any(|x| !x.is_finite()) {
            return Err(XrtError::Shape("invalid CUDA FP8 activation vector".into()));
        }
        let module = "xrt_block_fp8";
        let function_name = "block_fp8_matvec";
        if self.device.get_func(module, function_name).is_none() {
            let _nvrtc_library = super::native_libraries::nvrtc()?;
            let ptx = cudarc::nvrtc::compile_ptx(include_str!("kernels/block_fp8.cu"))
                .map_err(|e| cuda_error("failed to compile block-FP8 kernel", e))?;
            self.device
                .load_ptx(
                    ptx,
                    module,
                    &[
                        function_name,
                        "dense_f32_matvec",
                        "dense_bf16_matvec",
                        "fp8_moe_gate_up",
                        "fp8_moe_down_sum",
                    ],
                )
                .map_err(|e| cuda_error("failed to load block-FP8 kernel", e))?;
        }
        let (values, scales) = if w8a8 {
            let (bytes, scales) = quantize_e4m3fn(input, matrix.block_cols)?;
            (
                bytes
                    .iter()
                    .map(|b| E4M3FN[*b as usize])
                    .collect::<Vec<_>>(),
                scales,
            )
        } else {
            (
                input.to_vec(),
                vec![1.0; matrix.cols.div_ceil(matrix.block_cols)],
            )
        };
        self.upload_f32_into(&values, &mut scratch.input)?;
        self.upload_f32_into(&scales, &mut scratch.scales)?;
        let rows = to_u32(matrix.rows, "FP8 output rows")?;
        let cols = to_u32(matrix.cols, "FP8 input columns")?;
        let block_rows = to_u32(matrix.block_rows, "FP8 block rows")?;
        let block_cols = to_u32(matrix.block_cols, "FP8 block columns")?;
        let function = self
            .device
            .get_func(module, function_name)
            .ok_or_else(|| XrtError::Cuda("FP8 kernel absent after load".into()))?;
        // SAFETY: owners remain alive through synchronized download; checked
        // geometry exactly bounds every buffer access in the kernel.
        unsafe {
            function.launch(
                LaunchConfig {
                    grid_dim: (rows, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &matrix.weights.data,
                    &matrix.scales.data,
                    &scratch.input.data,
                    &scratch.scales.data,
                    &mut scratch.output.data,
                    rows,
                    cols,
                    block_rows,
                    block_cols,
                    w8a8 as u32,
                ),
            )
        }
        .map_err(|e| cuda_error("failed to launch block-FP8 matvec", e))?;
        let result = self.download_f32(&scratch.output)?;
        if result.iter().any(|x| !x.is_finite()) {
            return Err(XrtError::InvalidTensor(
                "CUDA FP8 output is non-finite".into(),
            ));
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires installed NVIDIA driver and NVRTC; run explicitly"]
    fn dense_workspace_reuses_buffers_without_activation_leakage() {
        let device = CudaDevice::new(0).unwrap();
        let baseline = device.allocation_stats().current_bytes;
        {
            let bits: Vec<u16> = (0..17 * 259)
                .map(|i| ((i as f32 * 0.03).sin().to_bits() >> 16) as u16)
                .collect();
            let bytes: Vec<u8> = bits.iter().flat_map(|v| v.to_le_bytes()).collect();
            let matrix = device
                .upload_dense_bf16_matrix(&bytes, 17, 259, bytes.len() as u64)
                .unwrap();
            let input: Vec<f32> = (0..259).map(|i| (i as f32 * 0.07).cos()).collect();
            let expected = device.matvec_dense_bf16(&matrix, &input).unwrap();
            let mut workspace = GpuDenseWorkspace::new(device.clone(), 4096);
            assert!(workspace.matvec_bf16(&matrix, &[0.0]).is_err());
            assert_eq!(workspace.resident_bytes(), 0);
            assert_eq!(workspace.matvec_bf16(&matrix, &input).unwrap(), expected);
            let allocation = device.allocation_stats().current_bytes;
            assert_eq!(
                workspace.matvec_bf16(&matrix, &vec![0.0; 259]).unwrap(),
                [0.0; 17]
            );
            assert_eq!(workspace.matvec_bf16(&matrix, &input).unwrap(), expected);
            assert_eq!(device.allocation_stats().current_bytes, allocation);
            assert_eq!(workspace.resident_bytes(), (17 + 259) * 4);
            assert!(workspace.peak_bytes() <= 4096);
        }
        device.synchronize().unwrap();
        assert_eq!(device.allocation_stats().current_bytes, baseline);
    }
    #[test]
    #[ignore = "requires installed NVIDIA driver and NVRTC; run explicitly"]
    fn routed_fp8_mlp_matches_independent_cpu_graph_and_releases() {
        let device = CudaDevice::new(0).unwrap();
        let baseline = device.allocation_stats().current_bytes;
        let input: Vec<f32> = (0..19).map(|i| (i as f32 * 0.31).sin()).collect();
        let mut sources = Vec::new();
        for expert in 0..2 {
            for (p, rows, cols) in [(0, 8, 19), (1, 8, 19), (2, 19, 8)] {
                let bytes: Vec<u8> = (0..rows * cols)
                    .map(|i| {
                        ((i * 17 + expert * 11 + p * 3) % 120) as u8
                            | if i % 3 == 0 { 128 } else { 0 }
                    })
                    .collect();
                sources.push(
                    PreparedFp8Matrix::new(bytes, vec![0.003], rows, cols, 128, 128).unwrap(),
                );
            }
        }
        let probabilities = [0.35, 0.6];
        let denominator = 0.95;
        let mut expected = vec![0.0; 19];
        for expert in 0..2 {
            let triple = &sources[expert * 3..][..3];
            let mut gate = vec![0.0; 8];
            let mut up = gate.clone();
            BlockFp8Matrix::new(
                triple[0].storage.as_ref(),
                &triple[0].scales,
                8,
                19,
                128,
                128,
            )
            .unwrap()
            .matvec(&input, &mut gate)
            .unwrap();
            BlockFp8Matrix::new(
                triple[1].storage.as_ref(),
                &triple[1].scales,
                8,
                19,
                128,
                128,
            )
            .unwrap()
            .matvec(&input, &mut up)
            .unwrap();
            for (g, u) in gate.iter_mut().zip(up) {
                *g = *g / (1.0 + (-*g).exp()) * u;
            }
            let mut down = vec![0.0; 19];
            BlockFp8Matrix::new(
                triple[2].storage.as_ref(),
                &triple[2].scales,
                19,
                8,
                128,
                128,
            )
            .unwrap()
            .matvec(&gate, &mut down)
            .unwrap();
            for (out, x) in expected.iter_mut().zip(down) {
                *out += x * probabilities[expert] / denominator;
            }
        }
        let routed = [
            [&sources[0], &sources[1], &sources[2]],
            [&sources[3], &sources[4], &sources[5]],
        ];
        let mut cache = GpuFp8Cache::new(device.clone(), 2048);
        assert!(cache
            .routed_mlp_prepared(&routed, &input, &probabilities, 0.0)
            .is_err());
        for _ in 0..2 {
            let actual = cache
                .routed_mlp_prepared(&routed, &input, &probabilities, denominator)
                .unwrap();
            for (a, b) in actual.iter().zip(&expected) {
                assert!((a - b).abs() < 3e-5 * b.abs().max(1.0), "{a} != {b}");
            }
        }
        assert_eq!(cache.stats().hits, 6);
        assert!(cache.stats().peak_workspace_bytes <= FP8_WORKSPACE_LIMIT);
        cache.matvec_prepared(&sources[0], &input, false).unwrap();
        assert!(cache.moe_scratch.is_none());
        cache.clear();
        device.synchronize().unwrap();
        assert_eq!(device.allocation_stats().current_bytes, baseline);
    }
    #[test]
    #[ignore = "requires installed NVIDIA driver and NVRTC; run explicitly"]
    fn fp8_cache_moves_to_worker_and_reuses_owner_context() {
        let device = CudaDevice::new(0).unwrap();
        let baseline = device.allocation_stats().current_bytes;
        let mut cache = GpuFp8Cache::new(device.clone(), 1024);
        let source = PreparedFp8Matrix::new(vec![56u8; 8], vec![1.0], 2, 4, 128, 128).unwrap();
        cache = std::thread::spawn(move || {
            assert_eq!(
                cache.matvec_prepared(&source, &[1.0; 4], false).unwrap(),
                [4.0; 2]
            );
            cache
        })
        .join()
        .unwrap();
        let source = PreparedFp8Matrix::new(vec![56u8; 8], vec![1.0], 2, 4, 128, 128).unwrap();
        assert_eq!(
            cache.matvec_prepared(&source, &[2.0; 4], false).unwrap(),
            [8.0; 2]
        );
        assert_eq!(cache.stats().hits, 1);
        drop(cache);
        device.synchronize().unwrap();
        assert_eq!(device.allocation_stats().current_bytes, baseline);
    }
    #[test]
    #[ignore = "requires installed NVIDIA driver and NVRTC; run explicitly"]
    fn dense_bf16_preserves_encoded_weights_and_fp32_reference() {
        let device = CudaDevice::new(0).unwrap();
        let baseline = device.allocation_stats().current_bytes;
        let bits: Vec<u16> = (0..17 * 259)
            .map(|i| ((i as f32 * 0.03).sin().to_bits() >> 16) as u16)
            .collect();
        let bytes: Vec<u8> = bits.iter().flat_map(|x| x.to_le_bytes()).collect();
        let expanded: Vec<f32> = bits
            .iter()
            .map(|b| f32::from_bits((*b as u32) << 16))
            .collect();
        let input: Vec<f32> = (0..259).map(|i| (i as f32 * 0.07).cos()).collect();
        assert!(device
            .upload_dense_bf16_matrix(&[128, 127], 1, 1, 2)
            .is_err());
        assert_eq!(device.allocation_stats().current_bytes, baseline);
        {
            let bf16 = device
                .upload_dense_bf16_matrix(&bytes, 17, 259, bytes.len() as u64)
                .unwrap();
            let expanded_bytes: Vec<u8> = expanded.iter().flat_map(|x| x.to_le_bytes()).collect();
            let f32 = device
                .upload_dense_f32_matrix(
                    &expanded_bytes,
                    DType::F32,
                    17,
                    259,
                    expanded_bytes.len() as u64,
                )
                .unwrap();
            assert_eq!(bf16.resident_bytes() * 2, f32.resident_bytes());
            let actual = device.matvec_dense_bf16(&bf16, &input).unwrap();
            let expected = device.matvec_dense_f32(&f32, &input).unwrap();
            assert_eq!(actual, expected);
        }
        device.synchronize().unwrap();
        assert_eq!(device.allocation_stats().current_bytes, baseline);
    }
    #[test]
    fn prepared_fp8_rejects_invalid_storage_and_binds_scales_and_geometry() {
        assert!(PreparedFp8Matrix::new(vec![127; 8], vec![1.0], 2, 4, 128, 128).is_err());
        assert!(PreparedFp8Matrix::new(vec![56; 8], vec![0.0], 2, 4, 128, 128).is_err());
        let a = PreparedFp8Matrix::new(vec![56; 8], vec![1.0], 2, 4, 128, 128).unwrap();
        let b = PreparedFp8Matrix::new(vec![56; 8], vec![2.0], 2, 4, 128, 128).unwrap();
        let c = PreparedFp8Matrix::new(vec![56; 8], vec![1.0], 4, 2, 128, 128).unwrap();
        assert_ne!(a.key, b.key);
        assert_ne!(a.key, c.key);
    }
    #[test]
    #[ignore = "requires installed NVIDIA driver and NVRTC; run explicitly"]
    fn dense_cuda_matvec_matches_reference_and_releases() {
        let device = CudaDevice::new(0).unwrap();
        let baseline = device.allocation_stats().current_bytes;
        let weights: Vec<_> = (0..17 * 259).map(|i| (i as f32 * 0.03).sin()).collect();
        let bytes: Vec<_> = weights.iter().flat_map(|x| x.to_le_bytes()).collect();
        let input: Vec<_> = (0..259).map(|i| (i as f32 * 0.07).cos()).collect();
        {
            let matrix = device
                .upload_dense_f32_matrix(&bytes, DType::F32, 17, 259, 1_000_000)
                .unwrap();
            let actual = device.matvec_dense_f32(&matrix, &input).unwrap();
            for (row, value) in actual.iter().enumerate() {
                let expected: f32 = weights[row * 259..][..259]
                    .iter()
                    .zip(&input)
                    .map(|(w, x)| w * x)
                    .sum();
                assert!((value - expected).abs() < 1e-4 * expected.abs().max(1.0));
            }
        }
        device.synchronize().unwrap();
        assert_eq!(device.allocation_stats().current_bytes, baseline);
    }
    #[test]
    #[ignore = "requires an installed NVIDIA driver and NVRTC; run explicitly"]
    fn fp8_cuda_matches_cpu_and_releases_allocation() {
        let device = CudaDevice::new(0).unwrap();
        let baseline = device.allocation_stats().current_bytes;
        let bytes: Vec<_> = (0..259 * 261)
            .map(|i| ((i * 13) % 126) as u8 | if i % 3 == 0 { 128 } else { 0 })
            .collect();
        let scales: Vec<_> = (0..9).map(|i| (i + 1) as f32 * 0.005).collect();
        let input: Vec<_> = (0..261).map(|i| (i as f32 * 0.1).sin()).collect();
        let cpu = BlockFp8Matrix::new(&bytes, &scales, 259, 261, 128, 128).unwrap();
        assert!(device
            .upload_block_fp8(&bytes, &scales, 259, 261, 128, 128, 1)
            .is_err());
        assert_eq!(device.allocation_stats().current_bytes, baseline);
        {
            let gpu = device
                .upload_block_fp8(&bytes, &scales, 259, 261, 128, 128, 1_000_000)
                .unwrap();
            assert_eq!(
                device.allocation_stats().current_bytes - baseline,
                gpu.resident_bytes()
            );
            for w8a8 in [false, true] {
                let mut expected = vec![0.0; 259];
                if w8a8 {
                    cpu.matvec_w8a8(&input, &mut expected).unwrap();
                } else {
                    cpu.matvec(&input, &mut expected).unwrap();
                }
                let actual = device.matvec_block_fp8(&gpu, &input, w8a8).unwrap();
                for (a, b) in actual.iter().zip(expected) {
                    assert!((a - b).abs() < 3e-4 * b.abs().max(1.0), "{a} != {b}");
                }
            }
        }
        device.synchronize().unwrap();
        assert_eq!(device.allocation_stats().current_bytes, baseline);
    }
    #[test]
    #[ignore = "requires installed NVIDIA driver and NVRTC; run explicitly"]
    fn fp8_cache_evicts_within_budget_and_binds_content() {
        let device = CudaDevice::new(0).unwrap();
        let baseline = device.allocation_stats().current_bytes;
        let mut cache = GpuFp8Cache::new(device.clone(), 16);
        let input = [1.0; 4];
        let a = [56u8; 8];
        let b = [64u8; 8];
        let first = cache
            .matvec(&a, &[1.0], 2, 4, 128, 128, &input, false)
            .unwrap();
        assert_eq!(first, [4.0, 4.0]);
        cache
            .matvec(&a, &[1.0], 2, 4, 128, 128, &input, false)
            .unwrap();
        assert_eq!(cache.stats().hits, 1);
        let changed = cache
            .matvec(&b, &[1.0], 2, 4, 128, 128, &input, false)
            .unwrap();
        assert_eq!(changed, [8.0, 8.0]);
        assert_eq!(cache.stats().evictions, 1);
        assert!(cache.stats().peak_resident_bytes <= 16);
        let source = PreparedFp8Matrix::new(a.to_vec(), vec![1.0], 2, 4, 128, 128).unwrap();
        assert_eq!(
            cache.matvec_prepared(&source, &input, false).unwrap(),
            first
        );
        let hits = cache.stats().hits;
        assert_eq!(
            cache.matvec_prepared(&source, &input, false).unwrap(),
            first
        );
        assert_eq!(cache.stats().hits, hits + 1);
        assert_eq!(cache.stats().reused_allocations, 2);
        cache.clear();
        cache.device.synchronize().unwrap();
        assert_eq!(cache.device.allocation_stats().current_bytes, baseline);
    }
}
