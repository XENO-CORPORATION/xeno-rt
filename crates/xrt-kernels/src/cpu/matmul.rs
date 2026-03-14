use rayon::prelude::*;
use xrt_core::{checked_mul, DType, Result, XrtError};

use super::quantize::{dot_q4_0, dot_q4_k, dot_q5_k, dot_q6_k, dot_q8_0};
use super::simd;
use super::thread_pool::global_pool;

const MATMUL_TILE: usize = 64;
const VECTOR_WIDTH: usize = 8;

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

    // Pre-quantize input to Q8_0, then use AVX2 integer SIMD for dot products.
    // This path benefits Q8_0 and Q4_0 where the integer maddubs+madd sequence
    // is strictly faster than float-domain (no bias correction term needed).
    // K-quants (Q4_K, Q5_K, Q6_K) use float-domain kernels because the dmin
    // correction term negates the maddubs advantage on Zen4.
    #[cfg(target_arch = "x86_64")]
    if matches!(dtype, DType::Q8_0 | DType::Q4_0) && simd::has_avx2_fma() {
        let (input_scales, input_quants) = simd::quantize_f32_to_q8_0(vector);
        global_pool().par_for(rows, |start_row, end_row| {
            for row_index in start_row..end_row {
                let start = row_index * row_bytes;
                let row = &matrix[start..start + row_bytes];
                let val = unsafe {
                    if dtype == DType::Q4_0 {
                        simd::dot_q4_0_q8_0_avx2(row, &input_scales, &input_quants)
                    } else {
                        simd::dot_q8_0_q8_0_avx2(row, &input_scales, &input_quants)
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

    // AVX2 fast path: pre-quantize all input vectors, then parallel over rows
    #[cfg(target_arch = "x86_64")]
    if (dtype == DType::Q8_0 || dtype == DType::Q4_0) && simd::has_avx2_fma() {
        // Pre-quantize all seq_len input vectors to Q8_0
        let mut all_scales: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
        let mut all_quants: Vec<Vec<i8>> = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let input_vec = &inputs[t * cols..(t + 1) * cols];
            let (scales, quants) = simd::quantize_f32_to_q8_0(input_vec);
            all_scales.push(scales);
            all_quants.push(quants);
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
                        if dtype == DType::Q4_0 {
                            simd::dot_q4_0_q8_0_avx2(row, &all_scales[t], &all_quants[t])
                        } else {
                            simd::dot_q8_0_q8_0_avx2(row, &all_scales[t], &all_quants[t])
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
