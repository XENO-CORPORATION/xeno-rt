use rayon::prelude::*;
use xrt_core::{checked_mul, DType, Result, XrtError};

use super::quantize::{
    dequantize_q4_0_row, dequantize_q4_k_row, dequantize_q5_k_row, dequantize_q6_k_row,
    dequantize_q8_0_row,
};

const MATMUL_TILE: usize = 64;
const VECTOR_WIDTH: usize = 8;

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

                    for depth in k_base..k_end {
                        let a_value = a_row[depth];
                        let b_offset = depth * n + n_base;
                        let b_row = &b[b_offset..b_offset + output_tile.len()];
                        accumulate_scaled(output_tile, b_row, a_value);
                    }
                }
            }
        });
}

pub fn quantized_row_dot(dtype: DType, row: &[u8], input: &[f32]) -> Result<f32> {
    let mut scratch = vec![0.0f32; input.len()];
    dequantize_row(dtype, row, &mut scratch)?;
    Ok(dot(&scratch, input))
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

    output
        .par_iter_mut()
        .enumerate()
        .try_for_each(|(row_index, output)| -> Result<()> {
            let mut scratch = vec![0.0f32; cols];
            let start = row_index * row_bytes;
            let row = &matrix[start..start + row_bytes];
            dequantize_row(dtype, row, &mut scratch)?;
            *output = dot(&scratch, vector);
            Ok(())
        })
}

fn dequantize_row(dtype: DType, row: &[u8], output: &mut [f32]) -> Result<()> {
    match dtype {
        DType::Q8_0 => dequantize_q8_0_row(row, output),
        DType::Q4_0 => dequantize_q4_0_row(row, output),
        DType::Q4_K => dequantize_q4_k_row(row, output),
        DType::Q5_K => dequantize_q5_k_row(row, output),
        DType::Q6_K => dequantize_q6_k_row(row, output),
        _ => Err(XrtError::Unsupported(format!(
            "unsupported quantized row dtype {dtype:?}"
        ))),
    }
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
fn accumulate_scaled(output: &mut [f32], rhs: &[f32], lhs: f32) {
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
