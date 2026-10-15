use super::*;
use cudarc::{
    cublas::{result as blas_result, sys as blas_sys},
    driver::DevicePtrMut,
};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap};
use xrt_kernels::cpu::fp8::{quantize_e4m3fn, BlockFp8Matrix, E4M3FN};

const BF16_BLAS_WORKSPACE_BYTES: usize = 1024 * 1024;
struct Bf16Blas {
    handle: blas_sys::cublasHandle_t,
    workspace: CudaBytes,
    owner: Arc<DriverCudaDevice>,
    _library: libloading::Library,
}
// SAFETY: the owning projection workspace is exclusive. Every operation binds
// its retained CUDA context to the calling worker before touching the handle.
unsafe impl Send for Bf16Blas {}
impl Bf16Blas {
    fn new(device: &CudaDevice) -> Result<Self> {
        let library = super::native_libraries::cublas()?;
        device
            .device
            .bind_to_thread()
            .map_err(|e| cuda_error("cuBLAS context", e))?;
        let workspace = device.upload_native_bytes(&vec![0; BF16_BLAS_WORKSPACE_BYTES])?;
        let handle = blas_result::create_handle().map_err(|e| cuda_error("cuBLAS handle", e))?;
        let mut result = Self {
            handle,
            workspace,
            owner: device.device.clone(),
            _library: library,
        };
        // SAFETY: handle/context/stream share ownership. The explicit workspace
        // is alive until after handle destruction and is allocation-accounted.
        unsafe {
            blas_result::set_stream(result.handle, *result.owner.cu_stream() as *mut _)
                .map_err(|e| cuda_error("cuBLAS stream", e))?;
            blas_sys::lib()
                .cublasSetWorkspace_v2(
                    result.handle,
                    *result.workspace.data.device_ptr_mut() as *mut c_void,
                    BF16_BLAS_WORKSPACE_BYTES,
                )
                .result()
                .map_err(|e| cuda_error("cuBLAS bounded workspace", e))?;
        }
        Ok(result)
    }
}
impl Drop for Bf16Blas {
    fn drop(&mut self) {
        if let Err(e) = self.owner.bind_to_thread() {
            tracing::warn!(error=?e,"cuBLAS cleanup context unavailable");
            return;
        }
        // SAFETY: this handle has one owner and is destroyed before its buffers.
        if let Err(e) = unsafe { blas_result::destroy_handle(self.handle) } {
            tracing::warn!(error=?e,"cuBLAS handle cleanup failed");
        }
    }
}
struct Bf16Scratch {
    input: CudaBytes,
    output: CudaBytes,
}

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
    bf16_scratch: HashMap<(usize, usize), Bf16Scratch>,
    blas: Option<Bf16Blas>,
}
impl GpuDenseWorkspace {
    pub fn new(device: CudaDevice, budget: u64) -> Self {
        Self {
            device,
            budget,
            bytes: 0,
            peak: 0,
            scratch: HashMap::new(),
            bf16_scratch: HashMap::new(),
            blas: None,
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
        self.bf16_scratch.clear();
        self.blas.take();
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
    /// BF16 input/output and FP32 accumulation through CUDA-12 cuBLAS. The
    /// FP32-activation reference method above retains its existing arithmetic.
    pub fn matvec_bf16_activation(
        &mut self,
        matrix: &GpuDenseBf16Matrix,
        input: &[f32],
    ) -> Result<Vec<f32>> {
        self.device
            .validate_dense_input(&matrix.owner, matrix.cols, input)?;
        let rows =
            i32::try_from(matrix.rows).map_err(|_| XrtError::Shape("cuBLAS row limit".into()))?;
        let cols = i32::try_from(matrix.cols)
            .map_err(|_| XrtError::Shape("cuBLAS column limit".into()))?;
        let shape = (matrix.rows, matrix.cols);
        let buffer_bytes = matrix
            .rows
            .checked_add(matrix.cols)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| XrtError::Shape("BF16 workspace overflow".into()))?
            as u64;
        let minimum = buffer_bytes + BF16_BLAS_WORKSPACE_BYTES as u64;
        if minimum > self.budget {
            return Err(XrtError::Cuda(
                "bounded cuBLAS workspace exceeds dense budget".into(),
            ));
        }
        let extra = (if self.blas.is_none() {
            BF16_BLAS_WORKSPACE_BYTES as u64
        } else {
            0
        }) + if self.bf16_scratch.contains_key(&shape) {
            0
        } else {
            buffer_bytes
        };
        if self.bytes + extra > self.budget
            || (self.scratch.len() + self.bf16_scratch.len() >= 32
                && !self.bf16_scratch.contains_key(&shape))
        {
            self.clear();
        }
        if self.blas.is_none() {
            self.blas = Some(Bf16Blas::new(&self.device)?);
            self.bytes += BF16_BLAS_WORKSPACE_BYTES as u64;
        }
        if !self.bf16_scratch.contains_key(&shape) {
            let buffers = Bf16Scratch {
                input: self.device.upload_native_bytes(&vec![0; matrix.cols * 2])?,
                output: self.device.upload_native_bytes(&vec![0; matrix.rows * 2])?,
            };
            self.bf16_scratch.insert(shape, buffers);
            self.bytes += buffer_bytes;
        }
        self.peak = self.peak.max(self.bytes);
        let values: Vec<u16> = input
            .iter()
            .map(|v| {
                let b = v.to_bits();
                (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
            })
            .collect();
        if values.iter().any(|b| b & 0x7f80 == 0x7f80) {
            return Err(XrtError::InvalidTensor(
                "BF16 input overflows selected precision".into(),
            ));
        }
        self.device
            .device
            .bind_to_thread()
            .map_err(|e| cuda_error("BF16 cuBLAS worker context", e))?;
        let scratch = self
            .bf16_scratch
            .get_mut(&shape)
            .expect("prepared bounded buffers");
        let blas = self.blas.as_ref().expect("prepared handle");
        let alpha = 1.0f32;
        let beta = 0.0f32;
        // SAFETY: row-major W[N,K] is transposed column-major A[K,N]. B[K,1]
        // and C[N,1] have checked exact extents; all encoded buffers are BF16.
        // The exclusive handle uses the same retained context/default stream,
        // and synchronous output readback keeps every allocation alive.
        unsafe {
            driver_result::memcpy_htod_sync(*scratch.input.data.device_ptr(), &values)
                .map_err(|e| cuda_error("BF16 input upload", e))?;
            blas_result::gemm_ex(
                blas.handle,
                blas_sys::cublasOperation_t::CUBLAS_OP_T,
                blas_sys::cublasOperation_t::CUBLAS_OP_N,
                rows,
                1,
                cols,
                &alpha as *const f32 as *const c_void,
                *matrix.weights.data.device_ptr() as *const c_void,
                blas_sys::cudaDataType_t::CUDA_R_16BF,
                cols,
                *scratch.input.data.device_ptr() as *const c_void,
                blas_sys::cudaDataType_t::CUDA_R_16BF,
                cols,
                &beta as *const f32 as *const c_void,
                *scratch.output.data.device_ptr_mut() as *mut c_void,
                blas_sys::cudaDataType_t::CUDA_R_16BF,
                rows,
                blas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                blas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
            .map_err(|e| cuda_error("BF16 cuBLAS projection", e))?;
        }
        self.device.synchronize()?;
        let mut output = vec![0u16; matrix.rows];
        // SAFETY: output length exactly matches the checked BF16 device buffer.
        unsafe { driver_result::memcpy_dtoh_sync(&mut output, *scratch.output.data.device_ptr()) }
            .map_err(|e| cuda_error("BF16 output download", e))?;
        self.device
            .transfer_counters
            .record_host_to_device(values.len() * 2);
        self.device
            .transfer_counters
            .record_device_to_host(output.len() * 2);
        let output: Vec<_> = output
            .into_iter()
            .map(|b| f32::from_bits((b as u32) << 16))
            .collect();
        if output.iter().any(|v| !v.is_finite()) {
            return Err(XrtError::InvalidTensor(
                "BF16 cuBLAS output is non-finite".into(),
            ));
        }
        Ok(output)
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
    input_scales: CudaF32Buffer,
    hidden_scales: CudaF32Buffer,
    probabilities: CudaF32Buffer,
    output: CudaF32Buffer,
    partial: CudaF32Buffer,
}
impl MoeScratch {
    fn bytes(&self) -> u64 {
        (self.pointers.byte_len()
            + self.input.byte_len()
            + self.hidden.byte_len()
            + self.input_scales.byte_len()
            + self.hidden_scales.byte_len()
            + self.probabilities.byte_len()
            + self.output.byte_len()) as u64
            + self.partial.byte_len() as u64
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
    tensor_cores: bool,
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
            tensor_cores: true,
            entries: HashMap::new(),
            ages: BTreeSet::new(),
            scratch: HashMap::new(),
            moe_scratch: None,
        }
    }
    pub fn stats(&self) -> Fp8CacheStats {
        self.stats
    }
    /// Unsupported hardware retains the checked SIMT path. The policy permits
    /// controlled comparisons without advertising unsupported FP8 instructions.
    pub fn allow_tensor_cores(&mut self, enabled: bool) {
        self.tensor_cores = enabled;
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
        self.routed_mlp_precision(experts, input, probabilities, denominator, false)
    }
    /// BF16 gate/up boundaries and routing weights, FP32 SiLU/product/down,
    /// and independent dynamic FP8 groups before both expert projections.
    pub fn routed_mlp_w8a8_prepared<S: Fp8SourceBytes>(
        &mut self,
        experts: &[[&PreparedFp8Matrix<S>; 3]],
        input: &[f32],
        probabilities: &[f32],
        denominator: f32,
    ) -> Result<Vec<f32>> {
        self.routed_mlp_precision(experts, input, probabilities, denominator, true)
    }
    fn routed_mlp_precision<S: Fp8SourceBytes>(
        &mut self,
        experts: &[[&PreparedFp8Matrix<S>; 3]],
        input: &[f32],
        probabilities: &[f32],
        denominator: f32,
        w8a8: bool,
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
            .and_then(|n| n.checked_add(hidden.div_ceil(128) * 4))
            .and_then(|n| n.checked_add(middle.div_ceil(128) * count * 4))
            .and_then(|n| {
                hidden
                    .checked_mul(count)
                    .and_then(|v| v.checked_mul(4))
                    .and_then(|v| n.checked_add(v))
            })
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
                input_scales: self.device.zeros_f32(hidden.div_ceil(128))?,
                hidden_scales: self.device.zeros_f32(middle.div_ceil(128) * count)?,
                probabilities: self.device.zeros_f32(count)?,
                output: self.device.zeros_f32(hidden)?,
                partial: self.device.zeros_f32(count * hidden)?,
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
        if w8a8 {
            let mma = self.tensor_cores && self.device.load_native_mma_kernels()?;
            let quantize = self
                .device
                .device
                .get_func("xrt_block_fp8", "fp8_w8a8_quantize_groups")
                .ok_or_else(|| XrtError::Cuda("native W8A8 quantization kernel absent".into()))?;
            let gate = self
                .device
                .device
                .get_func(
                    if mma { "xrt_fp8_mma" } else { "xrt_block_fp8" },
                    if mma {
                        "fp8_mma_gate_up"
                    } else {
                        "fp8_moe_w8a8_gate_up"
                    },
                )
                .ok_or_else(|| XrtError::Cuda("native W8A8 gate/up kernel absent".into()))?;
            let down = self
                .device
                .device
                .get_func(
                    if mma { "xrt_fp8_mma" } else { "xrt_block_fp8" },
                    if mma {
                        "fp8_mma_down"
                    } else {
                        "fp8_moe_w8a8_down_sum"
                    },
                )
                .ok_or_else(|| XrtError::Cuda("native W8A8 down/sum kernel absent".into()))?;
            // SAFETY: shapes, current context, selected pointer owners and workspace
            // bounds were checked above. Four launches share a stream. Each group
            // writes distinct elements; the synchronous download completes all use.
            unsafe {
                quantize
                    .clone()
                    .launch(
                        LaunchConfig {
                            grid_dim: (h.div_ceil(128), 1, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            &mut scratch.input.data,
                            &mut scratch.input_scales.data,
                            h,
                            1u32,
                            1u32,
                        ),
                    )
                    .map_err(|e| cuda_error("native W8A8 input quantization", e))?;
                gate.launch(
                    LaunchConfig {
                        grid_dim: (if mma { m.div_ceil(8) } else { m }, n, 1),
                        block_dim: (if mma { 32 } else { 128 }, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &scratch.pointers.data,
                        &scratch.input.data,
                        &scratch.input_scales.data,
                        &mut scratch.hidden.data,
                        n,
                        h,
                        m,
                    ),
                )
                .map_err(|e| cuda_error("native W8A8 gate/up", e))?;
                quantize
                    .launch(
                        LaunchConfig {
                            grid_dim: (m.div_ceil(128), n, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            &mut scratch.hidden.data,
                            &mut scratch.hidden_scales.data,
                            m,
                            n,
                            0u32,
                        ),
                    )
                    .map_err(|e| cuda_error("native W8A8 hidden quantization", e))?;
                down.launch(
                    LaunchConfig {
                        grid_dim: (
                            if mma { h.div_ceil(8) } else { h },
                            if mma { n } else { 1 },
                            1,
                        ),
                        block_dim: (if mma { 32 } else { 128 }, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &scratch.pointers.data,
                        &scratch.hidden.data,
                        &scratch.hidden_scales.data,
                        &scratch.probabilities.data,
                        if mma {
                            &mut scratch.partial.data
                        } else {
                            &mut scratch.output.data
                        },
                        n,
                        h,
                        m,
                        denominator,
                    ),
                )
                .map_err(|e| cuda_error("native W8A8 down/sum", e))?;
                if mma {
                    let reduce = self
                        .device
                        .device
                        .get_func("xrt_fp8_mma", "fp8_mma_reduce")
                        .ok_or_else(|| {
                            XrtError::Cuda("FP8 Tensor Core reduction kernel absent".into())
                        })?;
                    reduce
                        .launch(
                            LaunchConfig {
                                grid_dim: (h.div_ceil(128), 1, 1),
                                block_dim: (128, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (&scratch.partial.data, &mut scratch.output.data, n, h),
                        )
                        .map_err(|e| cuda_error("FP8 expert reduction", e))?;
                }
            }
            let result = self.device.download_f32(&scratch.output)?;
            self.stats.execution_seconds += start.elapsed().as_secs_f64();
            if result.iter().any(|v| !v.is_finite()) {
                return Err(XrtError::InvalidTensor(
                    "native W8A8 result non-finite".into(),
                ));
            }
            return Ok(result);
        }
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
    fn load_native_mma_kernels(&self) -> Result<bool> {
        let module = "xrt_fp8_mma";
        if self.device.get_func(module, "fp8_mma_gate_up").is_some() {
            return Ok(true);
        }
        let major = self
            .device
            .attribute(sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(|e| cuda_error("FP8 MMA device capability", e))?;
        let minor = self
            .device
            .attribute(sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(|e| cuda_error("FP8 MMA device capability", e))?;
        if major * 10 + minor < 89 {
            return Ok(false);
        }
        let _library = super::native_libraries::nvrtc()?;
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            include_str!("kernels/fp8_mma.cu"),
            cudarc::nvrtc::CompileOptions {
                arch: Some("compute_89"),
                ..Default::default()
            },
        )
        .map_err(|e| cuda_error("compile SM89 FP8 Tensor Core kernels", e))?;
        self.device
            .load_ptx(
                ptx,
                module,
                &["fp8_mma_gate_up", "fp8_mma_down", "fp8_mma_reduce"],
            )
            .map_err(|e| cuda_error("load FP8 Tensor Core kernels", e))?;
        Ok(true)
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
                        "fp8_w8a8_quantize_groups",
                        "fp8_moe_w8a8_gate_up",
                        "fp8_moe_w8a8_down_sum",
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
                        "fp8_w8a8_quantize_groups",
                        "fp8_moe_w8a8_gate_up",
                        "fp8_moe_w8a8_down_sum",
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
    fn w8a8_cancellation_precedes_block_scaling_on_device() {
        let device = CudaDevice::new(0).unwrap();
        let matrix = device
            .upload_block_fp8(&[60, 56, 64], &[0.031234], 1, 3, 128, 128, 7)
            .unwrap();
        assert_eq!(
            device
                .matvec_block_fp8(&matrix, &[0.448, 0.224, -0.448], true)
                .unwrap(),
            [0.0]
        );
    }
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
    fn fused_w8a8_preserves_group_scales_bf16_boundaries_tails_and_cleanup() {
        let device = CudaDevice::new(0).unwrap();
        let baseline = device.allocation_stats().current_bytes;
        let bf16 = |value: f32| {
            let bits = value.to_bits();
            f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff0000)
        };
        let hidden: usize = 259;
        let middle: usize = 129;
        let input: Vec<_> = (0..hidden)
            .map(|i| bf16((i as f32 * 0.031).sin() * 1.7))
            .collect();
        let mut sources = Vec::new();
        for expert in 0..2 {
            for (projection, rows, cols) in [
                (0, middle, hidden),
                (1, middle, hidden),
                (2, hidden, middle),
            ] {
                let bytes: Vec<u8> = (0..rows * cols)
                    .map(|i| {
                        ((i * 17 + expert * 13 + projection * 5) % 120) as u8
                            | if i % 3 == 0 { 128 } else { 0 }
                    })
                    .collect();
                let scales: Vec<_> = (0..rows.div_ceil(128) * cols.div_ceil(128))
                    .map(|i| 0.0007 + i as f32 * 0.000031)
                    .collect();
                sources.push(PreparedFp8Matrix::new(bytes, scales, rows, cols, 128, 128).unwrap());
            }
        }
        let probabilities = [0.333, 0.679];
        let denominator = 1.012;
        let mut expected = vec![0.0; hidden];
        for expert in 0..2 {
            let project = |index: usize, input: &[f32], rows, cols| {
                let source = &sources[expert * 3 + index];
                let mut output = vec![0.0; rows];
                BlockFp8Matrix::new(
                    source.storage.as_ref(),
                    &source.scales,
                    rows,
                    cols,
                    128,
                    128,
                )
                .unwrap()
                .matvec_w8a8(input, &mut output)
                .unwrap();
                output
            };
            let gate = project(0, &input, middle, hidden);
            let up = project(1, &input, middle, hidden);
            let activated: Vec<_> = gate
                .iter()
                .zip(up)
                .map(|(g, u)| {
                    let g = bf16(*g);
                    g / (1.0 + (-g).exp()) * bf16(u)
                })
                .collect();
            let down = project(2, &activated, hidden, middle);
            for (sum, value) in expected.iter_mut().zip(down) {
                *sum += value * bf16(probabilities[expert] / denominator);
            }
        }
        let routed = [
            [&sources[0], &sources[1], &sources[2]],
            [&sources[3], &sources[4], &sources[5]],
        ];
        let mut cache = GpuFp8Cache::new(device.clone(), 512 * 1024);
        cache.allow_tensor_cores(false); // independent CPU/SIMT precision contract
        for _ in 0..2 {
            let actual = cache
                .routed_mlp_w8a8_prepared(&routed, &input, &probabilities, denominator)
                .unwrap();
            for (actual, expected) in actual.iter().zip(&expected) {
                assert!(
                    (actual - expected).abs() <= 4e-5 * expected.abs().max(1.0),
                    "{actual} != {expected}"
                );
            }
        }
        let zeros = cache
            .routed_mlp_w8a8_prepared(&routed, &vec![0.0; hidden], &probabilities, denominator)
            .unwrap();
        assert!(zeros.iter().all(|v| *v == 0.0));
        assert!(cache
            .routed_mlp_w8a8_prepared(
                &routed,
                &vec![f32::MAX; hidden],
                &probabilities,
                denominator
            )
            .is_err());
        assert!(cache
            .routed_mlp_w8a8_prepared(&routed, &input, &probabilities, 0.0)
            .is_err());
        assert!(cache.stats().peak_workspace_bytes <= FP8_WORKSPACE_LIMIT);
        cache.clear();
        device.synchronize().unwrap();
        assert_eq!(device.allocation_stats().current_bytes, baseline);
    }
    #[test]
    #[ignore = "requires installed NVIDIA driver and CUDA-12 cuBLAS; run explicitly"]
    fn cublas_bf16_projection_matches_exact_dyadic_reference_and_releases() {
        let device = CudaDevice::new(0).unwrap();
        let baseline = device.allocation_stats().current_bytes;
        for (rows, cols) in [(17usize, 259usize), (259, 129)] {
            let weights: Vec<f32> = (0..rows * cols)
                .map(|i| ((i % 31) as f32 - 15.0) / 32.0)
                .collect();
            let encoded: Vec<u8> = weights
                .iter()
                .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
                .collect();
            let input: Vec<f32> = (0..cols)
                .map(|i| (((i * 7) % 17) as f32 - 8.0) / 8.0)
                .collect();
            let expected: Vec<f32> = weights
                .chunks_exact(cols)
                .map(|w| w.iter().zip(&input).map(|(a, b)| a * b).sum())
                .collect();
            let rounded: Vec<f32> = expected
                .iter()
                .map(|v| {
                    let b = v.to_bits();
                    f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff0000)
                })
                .collect();
            let matrix = device
                .upload_dense_bf16_matrix(&encoded, rows, cols, encoded.len() as u64)
                .unwrap();
            let mut too_small = GpuDenseWorkspace::new(device.clone(), 1024);
            assert!(too_small.matvec_bf16_activation(&matrix, &input).is_err());
            assert_eq!(too_small.resident_bytes(), 0);
            let mut workspace = GpuDenseWorkspace::new(device.clone(), 4 * 1024 * 1024);
            assert_eq!(workspace.matvec_bf16(&matrix, &input).unwrap(), expected);
            assert_eq!(
                workspace.matvec_bf16_activation(&matrix, &input).unwrap(),
                rounded
            );
            let count = device.allocation_stats().allocation_calls;
            assert_eq!(
                workspace.matvec_bf16_activation(&matrix, &input).unwrap(),
                rounded
            );
            assert_eq!(device.allocation_stats().allocation_calls, count);
            assert!(workspace
                .matvec_bf16_activation(&matrix, &vec![f32::MAX; cols])
                .is_err());
            let (mut workspace, matrix) = std::thread::spawn(move || {
                assert_eq!(
                    workspace.matvec_bf16_activation(&matrix, &input).unwrap(),
                    rounded
                );
                (workspace, matrix)
            })
            .join()
            .unwrap();
            assert!(workspace
                .matvec_bf16_activation(&matrix, &vec![0.0; cols])
                .unwrap()
                .iter()
                .all(|v| *v == 0.0));
            workspace.clear();
            assert_eq!(workspace.resident_bytes(), 0);
            drop(matrix);
            device.synchronize().unwrap();
            assert_eq!(device.allocation_stats().current_bytes, baseline);
        }
    }
    #[test]
    #[ignore = "requires SM89+ NVIDIA GPU and NVRTC; run explicitly"]
    fn tensor_core_w8a8_matches_actual_pinned_triton_with_partial_groups() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../reference/audio/drama3/fp8-mma-fixture.json"
        ))
        .unwrap();
        let input: Vec<f32> = fixture["input"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let expected: Vec<f32> = fixture["output"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let (hidden, middle) = (259usize, 129usize);
        let device = CudaDevice::new(0).unwrap();
        assert!(device.load_native_mma_kernels().unwrap());
        let baseline = device.allocation_stats().current_bytes;
        let mut sources = Vec::new();
        for expert in 0..2 {
            for (projection, rows, cols) in [
                (0, middle, hidden),
                (1, middle, hidden),
                (2, hidden, middle),
            ] {
                let bytes: Vec<u8> = (0..rows * cols)
                    .map(|i| {
                        ((i * 17 + expert * 13 + projection * 5) % 120) as u8
                            | if i % 3 == 0 { 128 } else { 0 }
                    })
                    .collect();
                let scales: Vec<_> = (0..rows.div_ceil(128) * cols.div_ceil(128))
                    .map(|i| 0.0007 + i as f32 * 0.000031)
                    .collect();
                sources.push(PreparedFp8Matrix::new(bytes, scales, rows, cols, 128, 128).unwrap());
            }
        }
        let routed = [
            [&sources[0], &sources[1], &sources[2]],
            [&sources[3], &sources[4], &sources[5]],
        ];
        let mut cache = GpuFp8Cache::new(device.clone(), 512 * 1024);
        let round = |value: f32| {
            let b = value.to_bits();
            f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff0000)
        };
        for _ in 0..2 {
            let actual = cache
                .routed_mlp_w8a8_prepared(&routed, &input, &[0.333, 0.679], 1.012)
                .unwrap();
            for (i, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                assert_eq!(round(*actual), *expected, "Triton mismatch at output {i}");
            }
        }
        let zeros = cache
            .routed_mlp_w8a8_prepared(&routed, &vec![0.0; hidden], &[0.333, 0.679], 1.012)
            .unwrap();
        assert!(zeros.iter().all(|v| *v == 0.0));
        assert!(cache
            .routed_mlp_w8a8_prepared(&routed, &vec![f32::MAX; hidden], &[0.333, 0.679], 1.012)
            .is_err());
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
