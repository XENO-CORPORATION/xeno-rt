use rayon::prelude::*;
use std::sync::OnceLock;
use xrt_core::{checked_mul, DType, Result, XrtError};

use super::quantize::{dot_mxfp4, dot_q4_0, dot_q4_k, dot_q5_k, dot_q6_k, dot_q8_0};
use super::simd;
use super::thread_pool::global_pool;

const MATMUL_TILE: usize = 64;
const VECTOR_WIDTH: usize = 8;

/// Opt-in diagnostic mode that keeps quantized-weight matvecs in the
/// float-activation domain. The optimized CPU path normally quantizes
/// activations to Q8_0 before integer SIMD, which is faster but is not the
/// appropriate reference for validating a CUDA path that consumes F32
/// activations directly.
fn float_activation_reference_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("XRT_CPU_FLOAT_ACTIVATION_REFERENCE")
            .is_ok_and(|value| matches!(value.trim(), "1" | "true" | "TRUE" | "True"))
    })
}

/// A wrapper around a raw mutable pointer stored as usize for Send+Sync.
/// SAFETY: The caller must guarantee that concurrent accesses through this pointer
/// are to disjoint memory locations (no data races).
#[derive(Clone, Copy)]
struct SendPtr(usize);

impl SendPtr {
    fn new(ptr: *mut f32) -> Self {
        Self(ptr as usize)
    }

    /// # Safety
    /// The caller must ensure the offset is within the original allocation
    /// and that no other thread writes to the same index concurrently.
    unsafe fn write_at(&self, idx: usize, val: f32) {
        let ptr = self.0 as *mut f32;
        *ptr.add(idx) = val;
    }
}

unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

pub fn matvec(matrix: &[f32], rows: usize, cols: usize, vector: &[f32], output: &mut [f32]) {
    assert_eq!(matrix.len(), rows * cols);
    assert_eq!(vector.len(), cols);
    assert_eq!(output.len(), rows);

    output
        .par_iter_mut()
        .enumerate()
        .for_each(|(row_index, output)| {
            let row = &matrix[row_index * cols..(row_index + 1) * cols];
            *output = dot(row, vector);
        });
}

pub fn matmul(a: &[f32], m: usize, k: usize, b: &[f32], n: usize, output: &mut [f32]) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(output.len(), m * n);

    if n == 0 || k == 0 {
        output.fill(0.0);
        return;
    }

    output
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(row_index, output_row)| {
            let a_row = &a[row_index * k..(row_index + 1) * k];
            output_row.fill(0.0);

            for k_base in (0..k).step_by(MATMUL_TILE) {
                let k_end = (k_base + MATMUL_TILE).min(k);

                for n_base in (0..n).step_by(MATMUL_TILE) {
                    let n_end = (n_base + MATMUL_TILE).min(n);
                    let output_tile = &mut output_row[n_base..n_end];

                    for (di, &a_value) in a_row[k_base..k_end].iter().enumerate() {
                        let depth = k_base + di;
                        let b_offset = depth * n + n_base;
                        let b_row = &b[b_offset..b_offset + output_tile.len()];
                        accumulate_scaled(output_tile, b_row, a_value);
                    }
                }
            }
        });
}

pub fn quantized_row_dot(dtype: DType, row: &[u8], input: &[f32]) -> Result<f32> {
    fused_dot(dtype, row, input)
}

fn fused_dot(dtype: DType, row: &[u8], input: &[f32]) -> Result<f32> {
    match dtype {
        DType::Q8_0 => Ok(dot_q8_0(row, input)),
        DType::Q4_0 => Ok(dot_q4_0(row, input)),
        DType::Q4_K => Ok(dot_q4_k(row, input)),
        DType::Q5_K => Ok(dot_q5_k(row, input)),
        DType::Q6_K => Ok(dot_q6_k(row, input)),
        DType::MXFP4 => Ok(dot_mxfp4(row, input)),
        _ => Err(XrtError::Unsupported(format!(
            "fused dot not supported for {dtype:?}"
        ))),
    }
}

pub fn matvec_quantized(
    matrix: &[u8],
    rows: usize,
    cols: usize,
    dtype: DType,
    vector: &[f32],
    output: &mut [f32],
) -> Result<()> {
    if !dtype.is_quantized() {
        return Err(XrtError::Unsupported(format!(
            "matvec_quantized expects a quantized dtype, got {dtype:?}"
        )));
    }
    if vector.len() != cols {
        return Err(XrtError::InvalidTensor(format!(
            "input vector length {} does not match matrix column count {cols}",
            vector.len()
        )));
    }
    if output.len() != rows {
        return Err(XrtError::InvalidTensor(format!(
            "output length {} does not match matrix row count {rows}",
            output.len()
        )));
    }
    if cols % dtype.block_size() != 0 {
        return Err(XrtError::InvalidTensor(format!(
            "matrix column count {cols} is not divisible by block size {} for {dtype:?}",
            dtype.block_size()
        )));
    }

    let row_bytes = checked_mul(
        cols / dtype.block_size(),
        dtype.block_bytes(),
        "quantized matvec row bytes",
    )?;
    let expected = checked_mul(row_bytes, rows, "quantized matvec matrix bytes")?;
    if matrix.len() != expected {
        return Err(XrtError::InvalidTensor(format!(
            "quantized matrix bytes {} do not match expected size {expected}",
            matrix.len()
        )));
    }

    let output_ptr = SendPtr::new(output.as_mut_ptr());

    // AVX-512 integer-domain fast path for Q4_K (Zen5+, Ice Lake+).
    // Processes 64 bytes per maddubs vs 32 for AVX2.
    #[cfg(target_arch = "x86_64")]
    if !float_activation_reference_enabled() && dtype == DType::Q4_K && simd::has_avx512_vnni() {
        let (input_scales, input_quants, input_half_sums) =
            simd::quantize_f32_to_q8_0_with_sums(vector);
        global_pool().par_for(rows, |start_row, end_row| {
            for row_index in start_row..end_row {
                let start = row_index * row_bytes;
                let row = &matrix[start..start + row_bytes];
                let val = unsafe {
                    simd::dot_q4_k_q8_0_avx512(row, &input_scales, &input_quants, &input_half_sums)
                };
                unsafe { output_ptr.write_at(row_index, val) };
            }
        });
        return Ok(());
    }

    // Pre-quantize input to Q8_0, then use AVX2 integer SIMD for dot products.
    // Integer-domain kernels are faster than float-domain for Q8_0, Q4_0, Q4_K, and Q5_K.
    // Q6_K still uses float-domain (no integer kernel implemented).
    #[cfg(target_arch = "x86_64")]
    if !float_activation_reference_enabled()
        && matches!(dtype, DType::Q8_0 | DType::Q4_0 | DType::Q4_K | DType::Q5_K)
        && simd::has_avx2_fma()
    {
        let (input_scales, input_quants, input_half_sums) =
            simd::quantize_f32_to_q8_0_with_sums(vector);
        global_pool().par_for(rows, |start_row, end_row| {
            for row_index in start_row..end_row {
                let start = row_index * row_bytes;
                let row = &matrix[start..start + row_bytes];
                let val = unsafe {
                    match dtype {
                        DType::Q4_0 => simd::dot_q4_0_q8_0_avx2(row, &input_scales, &input_quants),
                        DType::Q8_0 => simd::dot_q8_0_q8_0_avx2(row, &input_scales, &input_quants),
                        DType::Q4_K => simd::dot_q4_k_q8_0_avx2(
                            row,
                            &input_scales,
                            &input_quants,
                            &input_half_sums,
                        ),
                        DType::Q5_K => simd::dot_q5_k_q8_0_avx2(
                            row,
                            &input_scales,
                            &input_quants,
                            &input_half_sums,
                        ),
                        _ => unreachable!(),
                    }
                };
                unsafe { output_ptr.write_at(row_index, val) };
            }
        });
        return Ok(());
    }

    // Float-domain path for K-quants and generic fallback
    let error: std::sync::Mutex<Option<XrtError>> = std::sync::Mutex::new(None);
    global_pool().par_for(rows, |start_row, end_row| {
        for row_index in start_row..end_row {
            let start = row_index * row_bytes;
            let row = &matrix[start..start + row_bytes];
            match fused_dot(dtype, row, vector) {
                Ok(val) => unsafe { output_ptr.write_at(row_index, val) },
                Err(e) => {
                    *error.lock().unwrap() = Some(e);
                    return;
                }
            }
        }
    });

    if let Some(e) = error.into_inner().unwrap() {
        return Err(e);
    }
    Ok(())
}

pub fn matvec_quantized_batch(
    matrix: &[u8],
    rows: usize,
    cols: usize,
    dtype: DType,
    inputs: &[f32],
    seq_len: usize,
    outputs: &mut [f32],
) -> Result<()> {
    if seq_len == 0 {
        return Ok(());
    }
    if seq_len == 1 {
        return matvec_quantized(matrix, rows, cols, dtype, inputs, outputs);
    }

    if !dtype.is_quantized() {
        return Err(XrtError::Unsupported(format!(
            "matvec_quantized_batch expects a quantized dtype, got {dtype:?}"
        )));
    }
    if inputs.len() != seq_len * cols {
        return Err(XrtError::InvalidTensor(format!(
            "inputs length {} does not match seq_len({seq_len}) * cols({cols}) = {}",
            inputs.len(),
            seq_len * cols
        )));
    }
    if outputs.len() != seq_len * rows {
        return Err(XrtError::InvalidTensor(format!(
            "outputs length {} does not match seq_len({seq_len}) * rows({rows}) = {}",
            outputs.len(),
            seq_len * rows
        )));
    }
    if cols % dtype.block_size() != 0 {
        return Err(XrtError::InvalidTensor(format!(
            "matrix column count {cols} is not divisible by block size {} for {dtype:?}",
            dtype.block_size()
        )));
    }

    let row_bytes = checked_mul(
        cols / dtype.block_size(),
        dtype.block_bytes(),
        "quantized matvec_batch row bytes",
    )?;
    let expected = checked_mul(row_bytes, rows, "quantized matvec_batch matrix bytes")?;
    if matrix.len() != expected {
        return Err(XrtError::InvalidTensor(format!(
            "quantized matrix bytes {} do not match expected size {expected}",
            matrix.len()
        )));
    }

    let output_ptr = SendPtr::new(outputs.as_mut_ptr());
    let output_len = outputs.len();

    // AVX-512 fast path for Q4_K batch
    #[cfg(target_arch = "x86_64")]
    if !float_activation_reference_enabled() && dtype == DType::Q4_K && simd::has_avx512_vnni() {
        let mut all_scales: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
        let mut all_quants: Vec<Vec<i8>> = Vec::with_capacity(seq_len);
        let mut all_half_sums: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let input_vec = &inputs[t * cols..(t + 1) * cols];
            let (scales, quants, half_sums) = simd::quantize_f32_to_q8_0_with_sums(input_vec);
            all_scales.push(scales);
            all_quants.push(quants);
            all_half_sums.push(half_sums);
        }
        global_pool().par_for(rows, |start_row, end_row| {
            for row_index in start_row..end_row {
                let start = row_index * row_bytes;
                let row = &matrix[start..start + row_bytes];
                for t in 0..seq_len {
                    let idx = t * rows + row_index;
                    debug_assert!(idx < output_len);
                    let val = unsafe {
                        simd::dot_q4_k_q8_0_avx512(
                            row,
                            &all_scales[t],
                            &all_quants[t],
                            &all_half_sums[t],
                        )
                    };
                    unsafe { output_ptr.write_at(idx, val) };
                }
            }
        });
        return Ok(());
    }

    // AVX2 fast path: pre-quantize all input vectors, then parallel over rows
    #[cfg(target_arch = "x86_64")]
    if !float_activation_reference_enabled()
        && matches!(dtype, DType::Q8_0 | DType::Q4_0 | DType::Q4_K | DType::Q5_K)
        && simd::has_avx2_fma()
    {
        // Pre-quantize all seq_len input vectors to Q8_0
        let mut all_scales: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
        let mut all_quants: Vec<Vec<i8>> = Vec::with_capacity(seq_len);
        let mut all_half_sums: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let input_vec = &inputs[t * cols..(t + 1) * cols];
            let (scales, quants, half_sums) = simd::quantize_f32_to_q8_0_with_sums(input_vec);
            all_scales.push(scales);
            all_quants.push(quants);
            all_half_sums.push(half_sums);
        }

        // SAFETY: each (row_index, t) pair maps to a unique index t*rows+row_index,
        // so no two parallel iterations write to the same location.
        global_pool().par_for(rows, |start_row, end_row| {
            for row_index in start_row..end_row {
                let start = row_index * row_bytes;
                let row = &matrix[start..start + row_bytes];
                for t in 0..seq_len {
                    let idx = t * rows + row_index;
                    debug_assert!(idx < output_len);
                    let val = unsafe {
                        match dtype {
                            DType::Q4_0 => {
                                simd::dot_q4_0_q8_0_avx2(row, &all_scales[t], &all_quants[t])
                            }
                            DType::Q8_0 => {
                                simd::dot_q8_0_q8_0_avx2(row, &all_scales[t], &all_quants[t])
                            }
                            DType::Q4_K => simd::dot_q4_k_q8_0_avx2(
                                row,
                                &all_scales[t],
                                &all_quants[t],
                                &all_half_sums[t],
                            ),
                            DType::Q5_K => simd::dot_q5_k_q8_0_avx2(
                                row,
                                &all_scales[t],
                                &all_quants[t],
                                &all_half_sums[t],
                            ),
                            _ => unreachable!(),
                        }
                    };
                    unsafe { output_ptr.write_at(idx, val) };
                }
            }
        });
        return Ok(());
    }

    // Fallback: use fused_dot for each (row, token) pair
    let error: std::sync::Mutex<Option<XrtError>> = std::sync::Mutex::new(None);
    // SAFETY: each (row_index, t) maps to a unique index.
    global_pool().par_for(rows, |start_row, end_row| {
        for row_index in start_row..end_row {
            let start = row_index * row_bytes;
            let row = &matrix[start..start + row_bytes];
            for t in 0..seq_len {
                let input_vec = &inputs[t * cols..(t + 1) * cols];
                match fused_dot(dtype, row, input_vec) {
                    Ok(val) => {
                        let idx = t * rows + row_index;
                        debug_assert!(idx < output_len);
                        unsafe { output_ptr.write_at(idx, val) };
                    }
                    Err(e) => {
                        *error.lock().unwrap() = Some(e);
                        return;
                    }
                }
            }
        }
    });

    if let Some(e) = error.into_inner().unwrap() {
        return Err(e);
    }
    Ok(())
}

/// Compute multiple matvecs sharing the same input vector in a single parallel dispatch.
/// Saves (N-1) thread barrier synchronizations compared to N separate `matvec_quantized` calls.
///
/// All matrices must share the same `cols` and `dtype`. Each matrix's result is written
/// to a separate output buffer. Typical use: fused QKV projection (3 matrices) or
/// fused gate+up projection (2 matrices).
pub fn matvec_quantized_fused(
    matrices: &[&[u8]],
    row_counts: &[usize],
    cols: usize,
    dtype: DType,
    input: &[f32],
    outputs: &mut [&mut [f32]],
) -> Result<()> {
    let n = matrices.len();
    if n == 0 || n != row_counts.len() || n != outputs.len() {
        return Err(XrtError::InvalidTensor(
            "fused matvec: mismatched slice lengths".into(),
        ));
    }
    if !dtype.is_quantized() {
        return Err(XrtError::Unsupported(format!(
            "matvec_quantized_fused expects a quantized dtype, got {dtype:?}"
        )));
    }
    if input.len() != cols {
        return Err(XrtError::InvalidTensor(format!(
            "input length {} != cols {cols}",
            input.len()
        )));
    }
    if cols % dtype.block_size() != 0 {
        return Err(XrtError::InvalidTensor(format!(
            "cols {cols} not divisible by block size {} for {dtype:?}",
            dtype.block_size()
        )));
    }

    let row_bytes = (cols / dtype.block_size()) * dtype.block_bytes();

    // Validate sizes and build prefix-sum offsets
    let mut offsets = Vec::with_capacity(n + 1);
    offsets.push(0usize);
    for i in 0..n {
        let rows = row_counts[i];
        let expected = row_bytes * rows;
        if matrices[i].len() != expected {
            return Err(XrtError::InvalidTensor(format!(
                "fused matvec: matrix {i} has {} bytes, expected {expected}",
                matrices[i].len()
            )));
        }
        if outputs[i].len() != rows {
            return Err(XrtError::InvalidTensor(format!(
                "fused matvec: output {i} has {} elements, expected {rows}",
                outputs[i].len()
            )));
        }
        offsets.push(offsets[i] + rows);
    }
    let total_rows = *offsets.last().unwrap();

    // Collect output pointers (SendPtr for thread safety)
    let out_ptrs: Vec<SendPtr> = outputs
        .iter_mut()
        .map(|o| SendPtr::new(o.as_mut_ptr()))
        .collect();

    // Map global row → (matrix index, local row)
    let resolve_row = |global_row: usize| -> (usize, usize) {
        for i in 0..n {
            if global_row < offsets[i + 1] {
                return (i, global_row - offsets[i]);
            }
        }
        unreachable!()
    };

    #[cfg(target_arch = "x86_64")]
    if !float_activation_reference_enabled() && dtype == DType::Q4_K && simd::has_avx512_vnni() {
        let (input_scales, input_quants, input_half_sums) =
            simd::quantize_f32_to_q8_0_with_sums(input);
        global_pool().par_for(total_rows, |start, end| {
            for global_row in start..end {
                let (mat_idx, local_row) = resolve_row(global_row);
                let start = local_row * row_bytes;
                let row = &matrices[mat_idx][start..start + row_bytes];
                let val = unsafe {
                    simd::dot_q4_k_q8_0_avx512(row, &input_scales, &input_quants, &input_half_sums)
                };
                unsafe { out_ptrs[mat_idx].write_at(local_row, val) };
            }
        });
        return Ok(());
    }

    #[cfg(target_arch = "x86_64")]
    if !float_activation_reference_enabled()
        && matches!(dtype, DType::Q8_0 | DType::Q4_0 | DType::Q4_K | DType::Q5_K)
        && simd::has_avx2_fma()
    {
        let (input_scales, input_quants, input_half_sums) =
            simd::quantize_f32_to_q8_0_with_sums(input);
        global_pool().par_for(total_rows, |start, end| {
            for global_row in start..end {
                let (mat_idx, local_row) = resolve_row(global_row);
                let start = local_row * row_bytes;
                let row = &matrices[mat_idx][start..start + row_bytes];
                let val = unsafe {
                    match dtype {
                        DType::Q4_0 => simd::dot_q4_0_q8_0_avx2(row, &input_scales, &input_quants),
                        DType::Q8_0 => simd::dot_q8_0_q8_0_avx2(row, &input_scales, &input_quants),
                        DType::Q4_K => simd::dot_q4_k_q8_0_avx2(
                            row,
                            &input_scales,
                            &input_quants,
                            &input_half_sums,
                        ),
                        DType::Q5_K => simd::dot_q5_k_q8_0_avx2(
                            row,
                            &input_scales,
                            &input_quants,
                            &input_half_sums,
                        ),
                        _ => unreachable!(),
                    }
                };
                unsafe { out_ptrs[mat_idx].write_at(local_row, val) };
            }
        });
        return Ok(());
    }

    let error: std::sync::Mutex<Option<XrtError>> = std::sync::Mutex::new(None);
    global_pool().par_for(total_rows, |start, end| {
        for global_row in start..end {
            let (mat_idx, local_row) = resolve_row(global_row);
            let start = local_row * row_bytes;
            let row = &matrices[mat_idx][start..start + row_bytes];
            match fused_dot(dtype, row, input) {
                Ok(val) => unsafe { out_ptrs[mat_idx].write_at(local_row, val) },
                Err(e) => {
                    *error.lock().unwrap() = Some(e);
                    return;
                }
            }
        }
    });

    if let Some(e) = error.into_inner().unwrap() {
        return Err(e);
    }
    Ok(())
}

/// Mixed-dtype fused matvec: multiple matrices with potentially different dtypes
/// sharing the same input vector. Pre-quantizes input once, dispatches per-row.
/// Saves redundant input quantization and barrier syncs vs separate matvec calls.
pub fn matvec_quantized_fused_mixed(
    matrices: &[&[u8]],
    row_counts: &[usize],
    cols: usize,
    dtypes: &[DType],
    input: &[f32],
    outputs: &mut [&mut [f32]],
) -> Result<()> {
    let n = matrices.len();
    if n == 0 || n != row_counts.len() || n != outputs.len() || n != dtypes.len() {
        return Err(XrtError::InvalidTensor(
            "fused mixed matvec: mismatched slice lengths".into(),
        ));
    }
    if input.len() != cols {
        return Err(XrtError::InvalidTensor(format!(
            "input length {} != cols {cols}",
            input.len()
        )));
    }

    // Check if all dtypes are the same — if so, delegate to the optimized same-dtype path
    if dtypes.windows(2).all(|w| w[0] == w[1]) {
        return matvec_quantized_fused(matrices, row_counts, cols, dtypes[0], input, outputs);
    }

    // Build row_bytes per matrix and offsets
    struct MatInfo {
        row_bytes: usize,
        dtype: DType,
    }
    let mut infos = Vec::with_capacity(n);
    let mut offsets = Vec::with_capacity(n + 1);
    offsets.push(0usize);
    for i in 0..n {
        let dtype = dtypes[i];
        if cols % dtype.block_size() != 0 {
            return Err(XrtError::InvalidTensor(format!(
                "cols {cols} not divisible by block size {} for {dtype:?}",
                dtype.block_size()
            )));
        }
        let row_bytes = (cols / dtype.block_size()) * dtype.block_bytes();
        if matrices[i].len() != row_bytes * row_counts[i] {
            return Err(XrtError::InvalidTensor(format!(
                "mixed fused matvec: matrix {i} size mismatch"
            )));
        }
        infos.push(MatInfo { row_bytes, dtype });
        offsets.push(offsets[i] + row_counts[i]);
    }
    let total_rows = *offsets.last().unwrap();

    let out_ptrs: Vec<SendPtr> = outputs
        .iter_mut()
        .map(|o| SendPtr::new(o.as_mut_ptr()))
        .collect();

    let resolve_row = |global_row: usize| -> (usize, usize) {
        for i in 0..n {
            if global_row < offsets[i + 1] {
                return (i, global_row - offsets[i]);
            }
        }
        unreachable!()
    };

    // Pre-quantize input once for all matrices
    #[cfg(target_arch = "x86_64")]
    if !float_activation_reference_enabled() && simd::has_avx2_fma() {
        let (input_scales, input_quants, input_half_sums) =
            simd::quantize_f32_to_q8_0_with_sums(input);
        global_pool().par_for(total_rows, |start, end| {
            for global_row in start..end {
                let (mat_idx, local_row) = resolve_row(global_row);
                let info = &infos[mat_idx];
                let row_start = local_row * info.row_bytes;
                let row = &matrices[mat_idx][row_start..row_start + info.row_bytes];
                let val = unsafe {
                    match info.dtype {
                        DType::Q4_0 => simd::dot_q4_0_q8_0_avx2(row, &input_scales, &input_quants),
                        DType::Q8_0 => simd::dot_q8_0_q8_0_avx2(row, &input_scales, &input_quants),
                        DType::Q4_K => {
                            if simd::has_avx512_vnni() {
                                simd::dot_q4_k_q8_0_avx512(
                                    row,
                                    &input_scales,
                                    &input_quants,
                                    &input_half_sums,
                                )
                            } else {
                                simd::dot_q4_k_q8_0_avx2(
                                    row,
                                    &input_scales,
                                    &input_quants,
                                    &input_half_sums,
                                )
                            }
                        }
                        DType::Q5_K => simd::dot_q5_k_q8_0_avx2(
                            row,
                            &input_scales,
                            &input_quants,
                            &input_half_sums,
                        ),
                        DType::Q6_K => simd::dot_q6_k_q8_0_avx2(
                            row,
                            &input_scales,
                            &input_quants,
                            &input_half_sums,
                        ),
                        _ => {
                            // Fallback to float-domain for unknown dtypes
                            match info.dtype {
                                DType::Q4_K => simd::dot_q4_k_avx2(row, input),
                                DType::Q5_K => simd::dot_q5_k_avx2(row, input),
                                DType::Q6_K => simd::dot_q6_k_avx2(row, input),
                                _ => dot_q4_0(row, input),
                            }
                        }
                    }
                };
                unsafe { out_ptrs[mat_idx].write_at(local_row, val) };
            }
        });
        return Ok(());
    }

    // Scalar fallback
    let error: std::sync::Mutex<Option<XrtError>> = std::sync::Mutex::new(None);
    global_pool().par_for(total_rows, |start, end| {
        for global_row in start..end {
            let (mat_idx, local_row) = resolve_row(global_row);
            let info = &infos[mat_idx];
            let row_start = local_row * info.row_bytes;
            let row = &matrices[mat_idx][row_start..row_start + info.row_bytes];
            match fused_dot(info.dtype, row, input) {
                Ok(val) => unsafe { out_ptrs[mat_idx].write_at(local_row, val) },
                Err(e) => {
                    *error.lock().unwrap() = Some(e);
                    return;
                }
            }
        }
    });
    if let Some(e) = error.into_inner().unwrap() {
        return Err(e);
    }
    Ok(())
}

/// Compute independent quantized matvecs in one bounded row dispatch.
///
/// Unlike [`matvec_quantized_fused`], every matrix has its own input vector.
/// This is the selected-expert down-projection shape: the matrices share
/// geometry and quantization, while each expert consumes its own activated
/// intermediate row.
pub fn matvec_quantized_independent(
    matrices: &[&[u8]],
    rows: usize,
    cols: usize,
    dtype: DType,
    inputs: &[&[f32]],
    outputs: &mut [&mut [f32]],
) -> Result<()> {
    let task_count = matrices.len();
    if task_count == 0 || task_count != inputs.len() || task_count != outputs.len() {
        return Err(XrtError::InvalidTensor(
            "independent matvec: mismatched slice lengths".into(),
        ));
    }
    if !dtype.is_quantized() {
        return Err(XrtError::Unsupported(format!(
            "independent matvec expects a quantized dtype, got {dtype:?}"
        )));
    }
    if cols % dtype.block_size() != 0 {
        return Err(XrtError::InvalidTensor(format!(
            "independent matvec cols {cols} not divisible by block size {} for {dtype:?}",
            dtype.block_size()
        )));
    }
    let row_bytes = (cols / dtype.block_size())
        .checked_mul(dtype.block_bytes())
        .ok_or_else(|| XrtError::InvalidTensor("independent matvec row size overflowed".into()))?;
    for task in 0..task_count {
        let expected = rows.checked_mul(row_bytes).ok_or_else(|| {
            XrtError::InvalidTensor("independent matvec matrix size overflowed".into())
        })?;
        if matrices[task].len() != expected
            || inputs[task].len() != cols
            || outputs[task].len() != rows
        {
            return Err(XrtError::InvalidTensor(format!(
                "independent matvec task {task} has invalid matrix/input/output geometry"
            )));
        }
    }
    let total_rows = task_count
        .checked_mul(rows)
        .ok_or_else(|| XrtError::InvalidTensor("independent matvec row count overflowed".into()))?;
    let out_ptrs: Vec<SendPtr> = outputs
        .iter_mut()
        .map(|output| SendPtr::new(output.as_mut_ptr()))
        .collect();

    #[cfg(target_arch = "x86_64")]
    if !float_activation_reference_enabled()
        && matches!(
            dtype,
            DType::Q8_0 | DType::Q4_0 | DType::Q4_K | DType::Q5_K | DType::Q6_K
        )
        && simd::has_avx2_fma()
    {
        let quantized_inputs: Vec<_> = inputs
            .iter()
            .map(|input| simd::quantize_f32_to_q8_0_with_sums(input))
            .collect();
        global_pool().par_for(total_rows, |start, end| {
            for global_row in start..end {
                let task = global_row / rows;
                let local_row = global_row % rows;
                let row_start = local_row * row_bytes;
                let row = &matrices[task][row_start..row_start + row_bytes];
                let (input_scales, input_quants, input_half_sums) = &quantized_inputs[task];
                let value = unsafe {
                    match dtype {
                        DType::Q4_0 => simd::dot_q4_0_q8_0_avx2(row, input_scales, input_quants),
                        DType::Q8_0 => simd::dot_q8_0_q8_0_avx2(row, input_scales, input_quants),
                        DType::Q4_K if simd::has_avx512_vnni() => simd::dot_q4_k_q8_0_avx512(
                            row,
                            input_scales,
                            input_quants,
                            input_half_sums,
                        ),
                        DType::Q4_K => simd::dot_q4_k_q8_0_avx2(
                            row,
                            input_scales,
                            input_quants,
                            input_half_sums,
                        ),
                        DType::Q5_K => simd::dot_q5_k_q8_0_avx2(
                            row,
                            input_scales,
                            input_quants,
                            input_half_sums,
                        ),
                        DType::Q6_K => simd::dot_q6_k_q8_0_avx2(
                            row,
                            input_scales,
                            input_quants,
                            input_half_sums,
                        ),
                        _ => unreachable!(),
                    }
                };
                unsafe { out_ptrs[task].write_at(local_row, value) };
            }
        });
        return Ok(());
    }

    let error: std::sync::Mutex<Option<XrtError>> = std::sync::Mutex::new(None);
    global_pool().par_for(total_rows, |start, end| {
        for global_row in start..end {
            let task = global_row / rows;
            let local_row = global_row % rows;
            let row_start = local_row * row_bytes;
            let row = &matrices[task][row_start..row_start + row_bytes];
            match fused_dot(dtype, row, inputs[task]) {
                Ok(value) => unsafe { out_ptrs[task].write_at(local_row, value) },
                Err(err) => {
                    *error.lock().unwrap() = Some(err);
                    return;
                }
            }
        }
    });
    if let Some(error) = error.into_inner().unwrap() {
        return Err(error);
    }
    Ok(())
}

fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;
    let mut sum4 = 0.0f32;
    let mut sum5 = 0.0f32;
    let mut sum6 = 0.0f32;
    let mut sum7 = 0.0f32;

    let mut lhs_chunks = lhs.chunks_exact(VECTOR_WIDTH);
    let mut rhs_chunks = rhs.chunks_exact(VECTOR_WIDTH);

    for (lhs_chunk, rhs_chunk) in lhs_chunks.by_ref().zip(rhs_chunks.by_ref()) {
        sum0 = lhs_chunk[0].mul_add(rhs_chunk[0], sum0);
        sum1 = lhs_chunk[1].mul_add(rhs_chunk[1], sum1);
        sum2 = lhs_chunk[2].mul_add(rhs_chunk[2], sum2);
        sum3 = lhs_chunk[3].mul_add(rhs_chunk[3], sum3);
        sum4 = lhs_chunk[4].mul_add(rhs_chunk[4], sum4);
        sum5 = lhs_chunk[5].mul_add(rhs_chunk[5], sum5);
        sum6 = lhs_chunk[6].mul_add(rhs_chunk[6], sum6);
        sum7 = lhs_chunk[7].mul_add(rhs_chunk[7], sum7);
    }

    let mut sum = sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7;
    for (&lhs, &rhs) in lhs_chunks
        .remainder()
        .iter()
        .zip(rhs_chunks.remainder().iter())
    {
        sum = lhs.mul_add(rhs, sum);
    }

    sum
}

#[inline(always)]
pub fn accumulate_scaled(output: &mut [f32], rhs: &[f32], lhs: f32) {
    debug_assert_eq!(output.len(), rhs.len());

    let mut output_chunks = output.chunks_exact_mut(VECTOR_WIDTH);
    let mut rhs_chunks = rhs.chunks_exact(VECTOR_WIDTH);

    for (output_chunk, rhs_chunk) in output_chunks.by_ref().zip(rhs_chunks.by_ref()) {
        output_chunk[0] = lhs.mul_add(rhs_chunk[0], output_chunk[0]);
        output_chunk[1] = lhs.mul_add(rhs_chunk[1], output_chunk[1]);
        output_chunk[2] = lhs.mul_add(rhs_chunk[2], output_chunk[2]);
        output_chunk[3] = lhs.mul_add(rhs_chunk[3], output_chunk[3]);
        output_chunk[4] = lhs.mul_add(rhs_chunk[4], output_chunk[4]);
        output_chunk[5] = lhs.mul_add(rhs_chunk[5], output_chunk[5]);
        output_chunk[6] = lhs.mul_add(rhs_chunk[6], output_chunk[6]);
        output_chunk[7] = lhs.mul_add(rhs_chunk[7], output_chunk[7]);
    }

    for (output, &rhs) in output_chunks
        .into_remainder()
        .iter_mut()
        .zip(rhs_chunks.remainder().iter())
    {
        *output = lhs.mul_add(rhs, *output);
    }
}

#[cfg(test)]
mod tests {
    use half::f16;

    use super::{matvec_quantized, matvec_quantized_independent};
    use xrt_core::DType;

    fn q8_0_matrix(rows: usize, cols: usize, seed: i32) -> Vec<u8> {
        assert_eq!(cols % 32, 0);
        let mut matrix = Vec::with_capacity(rows * (cols / 32) * 34);
        for row in 0..rows {
            for block in 0..(cols / 32) {
                let scale = f16::from_f32(0.25 + (row + block + 1) as f32 * 0.125);
                matrix.extend_from_slice(&scale.to_bits().to_le_bytes());
                for column in 0..32 {
                    let quant = ((seed + row as i32 * 7 + block as i32 * 5 + column as i32) % 31
                        - 15) as i8;
                    matrix.push(quant as u8);
                }
            }
        }
        matrix
    }

    #[test]
    fn independent_quantized_matvec_matches_individual_dispatches() {
        let rows = 3;
        let cols = 64;
        let matrix_a = q8_0_matrix(rows, cols, 3);
        let matrix_b = q8_0_matrix(rows, cols, 19);
        let input_a = (0..cols)
            .map(|index| (index as f32 - 23.0) / 17.0)
            .collect::<Vec<_>>();
        let input_b = (0..cols)
            .map(|index| (41.0 - index as f32) / 13.0)
            .collect::<Vec<_>>();

        let mut expected_a = vec![0.0f32; rows];
        let mut expected_b = vec![0.0f32; rows];
        matvec_quantized(
            &matrix_a,
            rows,
            cols,
            DType::Q8_0,
            &input_a,
            &mut expected_a,
        )
        .expect("first reference matvec should succeed");
        matvec_quantized(
            &matrix_b,
            rows,
            cols,
            DType::Q8_0,
            &input_b,
            &mut expected_b,
        )
        .expect("second reference matvec should succeed");

        let mut actual_a = vec![0.0f32; rows];
        let mut actual_b = vec![0.0f32; rows];
        let matrices = [&matrix_a[..], &matrix_b[..]];
        let inputs = [&input_a[..], &input_b[..]];
        let mut outputs = [&mut actual_a[..], &mut actual_b[..]];
        matvec_quantized_independent(&matrices, rows, cols, DType::Q8_0, &inputs, &mut outputs)
            .expect("independent matvec dispatch should succeed");

        assert_eq!(actual_a, expected_a);
        assert_eq!(actual_b, expected_b);
    }
}
