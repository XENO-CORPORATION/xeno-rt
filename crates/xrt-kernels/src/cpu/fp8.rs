//! Checked, block-scaled OCP E4M3FN execution without expanding resident weights.
//! The FP32 activation reference is distinct from dynamic W8A8 execution.
use rayon::prelude::*;
use xrt_core::{Result, XrtError};

pub const E4M3FN_MAX: f32 = 448.0;

pub const fn decode_e4m3fn(bits: u8) -> f32 {
    let magnitude = bits & 127;
    let exponent = (magnitude >> 3) as u32;
    let fraction = (magnitude & 7) as u32;
    let value = if magnitude == 127 {
        f32::NAN
    } else if exponent == 0 {
        fraction as f32 / 512.0
    } else {
        (8 + fraction) as f32 * (1u32 << exponent) as f32 / 1024.0
    };
    if bits & 128 != 0 {
        -value
    } else {
        value
    }
}

const fn lookup() -> [f32; 256] {
    let mut result = [0.0; 256];
    let mut i = 0;
    while i < result.len() {
        result[i] = decode_e4m3fn(i as u8);
        i += 1;
    }
    result
}
pub const E4M3FN: [f32; 256] = lookup();

fn block_dot(bytes: &[u8], input: &[f32], scale: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if super::simd::has_avx2_fma() {
        // SAFETY: feature detection and equal, dimension-checked slices.
        return unsafe { block_dot_avx2(bytes, input, scale) };
    }
    bytes
        .iter()
        .zip(input)
        .map(|(b, x)| (E4M3FN[*b as usize] * scale) * x)
        .sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn block_dot_avx2(bytes: &[u8], input: &[f32], scale: f32) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_ps();
    let multiplier = _mm256_set1_ps(scale);
    let mut offset = 0;
    while offset + 8 <= bytes.len() {
        let packed = _mm_loadl_epi64(bytes.as_ptr().add(offset).cast());
        let indices = _mm256_cvtepu8_epi32(packed);
        let values = _mm256_i32gather_ps(E4M3FN.as_ptr(), indices, 4);
        let weights = _mm256_mul_ps(values, multiplier);
        let activations = _mm256_loadu_ps(input.as_ptr().add(offset));
        acc = _mm256_fmadd_ps(weights, activations, acc);
        offset += 8;
    }
    let mut partial = [0.0; 8];
    _mm256_storeu_ps(partial.as_mut_ptr(), acc);
    let tail: f32 = bytes[offset..]
        .iter()
        .zip(&input[offset..])
        .map(|(b, x)| (E4M3FN[*b as usize] * scale) * x)
        .sum();
    partial.iter().sum::<f32>() + tail
}

/// Saturating round-to-nearest, ties-to-even. Non-finite input is refused.
pub fn encode_e4m3fn(value: f32) -> Result<u8> {
    if !value.is_finite() {
        return Err(XrtError::InvalidTensor("non-finite FP8 input".into()));
    }
    let sign = if value.is_sign_negative() { 128 } else { 0 };
    let value = value.abs().min(E4M3FN_MAX);
    let mut low = 0usize;
    let mut high = 126usize;
    while low < high {
        let mid = (low + high) / 2;
        if E4M3FN[mid] < value {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    let mut selected = low;
    if low > 0 {
        let lower = value - E4M3FN[low - 1];
        let upper = E4M3FN[low] - value;
        if lower < upper || (lower == upper && (low - 1) % 2 == 0) {
            selected -= 1;
        }
    }
    Ok(sign | selected as u8)
}

/// Dynamic activation groups, with the same positive-scale floor used by
/// block-FP8 inference engines. The final group can be shorter than `group`.
pub fn quantize_e4m3fn(input: &[f32], group: usize) -> Result<(Vec<u8>, Vec<f32>)> {
    if input.is_empty() || group == 0 || input.iter().any(|x| !x.is_finite()) {
        return Err(XrtError::InvalidTensor(
            "invalid FP8 activation group".into(),
        ));
    }
    let mut bytes = Vec::with_capacity(input.len());
    let mut scales = Vec::with_capacity(input.len().div_ceil(group));
    for part in input.chunks(group) {
        // Match the rounded reciprocal multiply of the pinned serving kernel;
        // a correctly rounded divide has different FP8 tie behavior.
        let scale = part.iter().fold(1e-10f32, |a, b| a.max(b.abs())) * (1.0 / E4M3FN_MAX);
        scales.push(scale);
        for value in part {
            bytes.push(encode_e4m3fn(
                (value / scale).clamp(-E4M3FN_MAX, E4M3FN_MAX),
            )?);
        }
    }
    Ok((bytes, scales))
}

/// Row-major FP8 bytes and row-major 2D block scales. Construction validates
/// storage once; execution borrows the bytes and allocates no dense weight copy.
#[derive(Clone, Copy)]
pub struct BlockFp8Matrix<'a> {
    bytes: &'a [u8],
    scales: &'a [f32],
    rows: usize,
    cols: usize,
    block_rows: usize,
    block_cols: usize,
    scale_cols: usize,
}
impl<'a> BlockFp8Matrix<'a> {
    pub fn new(
        bytes: &'a [u8],
        scales: &'a [f32],
        rows: usize,
        cols: usize,
        block_rows: usize,
        block_cols: usize,
    ) -> Result<Self> {
        if rows == 0 || cols == 0 || block_rows == 0 || block_cols == 0 {
            return Err(XrtError::Shape("FP8 dimensions must be positive".into()));
        }
        let count = rows
            .checked_mul(cols)
            .ok_or_else(|| XrtError::Shape("FP8 size overflow".into()))?;
        let scale_cols = cols.div_ceil(block_cols);
        let scale_count = rows
            .div_ceil(block_rows)
            .checked_mul(scale_cols)
            .ok_or_else(|| XrtError::Shape("FP8 scale size overflow".into()))?;
        if bytes.len() != count || scales.len() != scale_count {
            return Err(XrtError::Shape(
                "FP8 weight/scale storage does not match geometry".into(),
            ));
        }
        if bytes.iter().any(|b| b & 127 == 127)
            || scales.iter().any(|s| !s.is_finite() || *s <= 0.0)
        {
            return Err(XrtError::InvalidTensor(
                "FP8 NaN or invalid block scale".into(),
            ));
        }
        Ok(Self {
            bytes,
            scales,
            rows,
            cols,
            block_rows,
            block_cols,
            scale_cols,
        })
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    pub fn resident_bytes(&self) -> usize {
        self.bytes.len() + std::mem::size_of_val(self.scales)
    }

    /// FP32 activation oracle: weights are scaled before multiplication.
    pub fn matvec(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        self.execute(input, None, output)
    }
    /// Dynamic W8A8: accumulate quantized products within each input block,
    /// then apply the weight and activation scales once per block.
    pub fn matvec_w8a8(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        let (bytes, scales) = quantize_e4m3fn(input, self.block_cols)?;
        let decoded: Vec<_> = bytes.iter().map(|b| E4M3FN[*b as usize]).collect();
        self.execute(&decoded, Some(&scales), output)
    }
    fn execute(
        &self,
        input: &[f32],
        activation_scales: Option<&[f32]>,
        output: &mut [f32],
    ) -> Result<()> {
        if input.len() != self.cols
            || output.len() != self.rows
            || input.iter().any(|x| !x.is_finite())
        {
            return Err(XrtError::Shape("invalid FP8 matvec input/output".into()));
        }
        output.par_iter_mut().enumerate().for_each(|(row, out)| {
            let values = &self.bytes[row * self.cols..(row + 1) * self.cols];
            let scales =
                &self.scales[(row / self.block_rows) * self.scale_cols..][..self.scale_cols];
            let mut sum = 0.0;
            for (block, part) in values.chunks(self.block_cols).enumerate() {
                let activations = &input[block * self.block_cols..][..part.len()];
                if let Some(activation_scales) = activation_scales {
                    let dot = block_dot(part, activations, 1.0);
                    // The pinned block-scaled GEMM applies activation scale,
                    // then weight scale, after its complete K-group dot.
                    sum += (dot * activation_scales[block]) * scales[block];
                } else {
                    sum += block_dot(part, activations, scales[block]);
                }
            }
            *out = sum;
        });
        if output.iter().any(|x| !x.is_finite()) {
            return Err(XrtError::InvalidTensor(
                "FP8 matvec produced non-finite output".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn w8a8_cancellation_happens_before_block_scaling() {
        // An independently evaluated exact FP8 dot is 672 + 224 - 896 = 0.
        // Scaling each term first can introduce a nonzero cancellation residue.
        let matrix = BlockFp8Matrix::new(&[60, 56, 64], &[0.031234], 1, 3, 128, 128).unwrap();
        let mut output = [f32::NAN];
        matrix
            .matvec_w8a8(&[0.448, 0.224, -0.448], &mut output)
            .unwrap();
        assert_eq!(output, [0.0]);
    }
    #[test]
    fn exhaustive_finite_round_trip_and_boundaries() {
        for bits in 0..=255u8 {
            if bits & 127 == 127 {
                assert!(decode_e4m3fn(bits).is_nan());
            } else {
                assert_eq!(encode_e4m3fn(decode_e4m3fn(bits)).unwrap(), bits);
            }
        }
        assert_eq!(decode_e4m3fn(1), 1.0 / 512.0);
        assert_eq!(decode_e4m3fn(126), 448.0);
        assert_eq!(encode_e4m3fn(1.0625).unwrap(), 56);
        assert_eq!(encode_e4m3fn(1.1875).unwrap(), 58);
        assert_eq!(encode_e4m3fn(1000.0).unwrap(), 126);
        assert!(encode_e4m3fn(f32::NAN).is_err());
    }
    #[test]
    fn block_boundaries_and_partial_blocks_match_independent_dense_math() {
        let bytes: Vec<_> = (0..35).map(|i| 48 + (i % 14) as u8).collect();
        let scales = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let matrix = BlockFp8Matrix::new(&bytes, &scales, 5, 7, 2, 3).unwrap();
        let input = [1.0, -2.0, 3.0, 4.0, -5.0, 6.0, 7.0];
        let mut output = [0.0; 5];
        matrix.matvec(&input, &mut output).unwrap();
        for row in 0..5 {
            let expected: f32 = (0..7)
                .map(|col| {
                    decode_e4m3fn(bytes[row * 7 + col])
                        * scales[(row / 2) * 3 + col / 3]
                        * input[col]
                })
                .sum();
            assert!((output[row] - expected).abs() < 1e-5);
        }
        assert_eq!(matrix.resident_bytes(), 35 + 36);
        matrix.matvec_w8a8(&input, &mut output).unwrap();
        assert!(output.iter().all(|x| x.is_finite()));
    }
    #[test]
    fn malformed_storage_and_nonfinite_inputs_are_refused() {
        assert!(BlockFp8Matrix::new(&[127], &[1.0], 1, 1, 1, 1).is_err());
        assert!(BlockFp8Matrix::new(&[0], &[0.0], 1, 1, 1, 1).is_err());
        assert!(BlockFp8Matrix::new(&[], &[], usize::MAX, 2, 128, 128).is_err());
        let matrix = BlockFp8Matrix::new(&[56], &[1.0], 1, 1, 128, 128).unwrap();
        let mut output = [42.0];
        assert!(matrix.matvec(&[f32::NAN], &mut output).is_err());
        assert_eq!(output, [42.0]);
    }
}
