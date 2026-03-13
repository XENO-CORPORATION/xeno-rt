/// AVX2-accelerated quantized dot products using integer-only arithmetic.
///
/// Key insight: instead of converting i8→f32 for every element, quantize the
/// f32 input vector to Q8_0 once, then use integer SIMD instructions
/// (_mm256_maddubs_epi16 + _mm256_madd_epi16) for the inner dot product.
/// This matches llama.cpp's approach and avoids the int-to-float conversion bottleneck.
///
/// Safety: all functions require AVX2 + FMA support, checked via `is_x86_feature_detected!`
/// at the dispatch layer (not here).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use half::f16;

const QK8_0: usize = 32;
const QK4_0: usize = 32;

// ============================================================================
// On-the-fly quantization: f32 → Q8_0 (used to quantize the input vector once)
// ============================================================================

/// Quantize an f32 slice into Q8_0 blocks (scale as f32 + 32 x i8).
/// Returns (scales, quantized_values) where each block of 32 f32s becomes
/// one f32 scale + 32 i8 quants.
pub fn quantize_f32_to_q8_0(input: &[f32]) -> (Vec<f32>, Vec<i8>) {
    let n_blocks = input.len() / QK8_0;
    let mut scales = Vec::with_capacity(n_blocks + 1);
    let mut quants = Vec::with_capacity((n_blocks + 1) * QK8_0);

    for block in input.chunks_exact(QK8_0) {
        // Find max absolute value
        let mut amax = 0.0f32;
        for &v in block {
            let abs = v.abs();
            if abs > amax {
                amax = abs;
            }
        }

        let scale = amax / 127.0;
        scales.push(scale);

        if amax == 0.0 {
            quants.extend_from_slice(&[0i8; QK8_0]);
        } else {
            let inv_scale = 127.0 / amax;
            for &v in block {
                let q = (v * inv_scale).round() as i32;
                quants.push(q.clamp(-128, 127) as i8);
            }
        }
    }

    // Handle remainder (pad with zeros)
    let remainder = input.len() % QK8_0;
    if remainder > 0 {
        let block = &input[n_blocks * QK8_0..];
        let mut amax = 0.0f32;
        for &v in block {
            let abs = v.abs();
            if abs > amax {
                amax = abs;
            }
        }
        let scale = amax / 127.0;
        scales.push(scale);
        if amax == 0.0 {
            quants.extend(std::iter::repeat_n(0i8, QK8_0));
        } else {
            let inv_scale = 127.0 / amax;
            for &v in block {
                quants.push((v * inv_scale).round().clamp(-128.0, 127.0) as i8);
            }
            for _ in 0..(QK8_0 - remainder) {
                quants.push(0i8);
            }
        }
    }

    (scales, quants)
}

// ============================================================================
// Q8_0 × Q8_0 integer-only dot product (AVX2)
// Both sides are quantized: matrix row is Q8_0 blocks (f16 scale + 32×i8),
// input is pre-quantized Q8_0 (f32 scale + 32×i8).
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_q8_0_q8_0_avx2(
    row: &[u8],
    input_scales: &[f32],
    input_quants: &[i8],
) -> f32 {
    let block_size = 2 + QK8_0; // f16 (2 bytes) + 32 i8
    let n_blocks = row.len() / block_size;
    let ones_16 = _mm256_set1_epi16(1);
    let mut acc = _mm256_setzero_ps();

    for bi in 0..n_blocks {
        let block_ptr = row.as_ptr().add(bi * block_size);

        // Read matrix block scale (f16 → f32)
        let d_bytes = [*block_ptr, *block_ptr.add(1)];
        let matrix_scale = f16::from_le_bytes(d_bytes).to_f32();
        let input_scale = *input_scales.get_unchecked(bi);
        let combined_scale = _mm256_set1_ps(matrix_scale * input_scale);

        let qs_ptr = block_ptr.add(2);
        let iq_ptr = input_quants.as_ptr().add(bi * QK8_0) as *const __m256i;

        // Load 32 i8 from matrix and 32 i8 from quantized input
        let q_mat = _mm256_loadu_si256(qs_ptr as *const __m256i);
        let q_inp = _mm256_loadu_si256(iq_ptr);

        // Integer dot product using maddubs + madd
        // maddubs requires first arg unsigned, second signed.
        // We need to handle the sign correctly. Since both are signed i8,
        // we use the trick: compute sum_i (a_i * b_i) using:
        //   abs(a) * sign(a,b) via _mm256_sign_epi8, then maddubs
        let sign_mat = _mm256_sign_epi8(q_mat, q_mat); // abs(q_mat), but sets 0 where q_mat is -128
        let q_inp_signed = _mm256_sign_epi8(q_inp, q_mat); // q_inp * sign(q_mat)

        // maddubs: treats first as u8, second as i8, multiplies pairs and adds adjacent to i16
        let prod_16 = _mm256_maddubs_epi16(sign_mat, q_inp_signed);

        // madd: multiply i16 by 1 and add adjacent pairs to i32
        let prod_32 = _mm256_madd_epi16(prod_16, ones_16);

        // Convert i32 to f32 and multiply by combined scale
        let prod_f32 = _mm256_cvtepi32_ps(prod_32);
        acc = _mm256_fmadd_ps(combined_scale, prod_f32, acc);
    }

    hsum_f32_avx2(acc)
}

// ============================================================================
// Q4_0 × Q8_0 integer-only dot product (AVX2)
// Matrix row is Q4_0 (f16 scale + 16 packed nibble bytes),
// input is pre-quantized Q8_0 (f32 scale + 32×i8).
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_q4_0_q8_0_avx2(
    row: &[u8],
    input_scales: &[f32],
    input_quants: &[i8],
) -> f32 {
    let block_size = 2 + QK4_0 / 2; // f16 (2 bytes) + 16 packed bytes
    let n_blocks = row.len() / block_size;
    let mask_low = _mm256_set1_epi8(0x0f);
    let ones_16 = _mm256_set1_epi16(1);
    let mut acc = _mm256_setzero_ps();

    for bi in 0..n_blocks {
        let block_ptr = row.as_ptr().add(bi * block_size);

        // Read matrix block scale
        let d_bytes = [*block_ptr, *block_ptr.add(1)];
        let matrix_scale = f16::from_le_bytes(d_bytes).to_f32();
        let input_scale = *input_scales.get_unchecked(bi);
        let combined_scale = _mm256_set1_ps(matrix_scale * input_scale);

        let qs_ptr = block_ptr.add(2);
        let iq_ptr = input_quants.as_ptr().add(bi * QK4_0) as *const __m256i;

        // Load 16 packed nibble bytes
        let packed_128 = _mm_loadu_si128(qs_ptr as *const __m128i);
        let packed_256 = _mm256_castsi128_si256(packed_128);
        let packed_256 = _mm256_inserti128_si256(packed_256, packed_128, 1);

        // Unpack: low nibbles in lower lane, high nibbles in upper lane
        let low_nibbles = _mm256_and_si256(packed_256, mask_low);
        let high_nibbles = _mm256_and_si256(_mm256_srli_epi16(packed_256, 4), mask_low);

        // Interleave: we need [low0..low15, high0..high15] to match input order
        // low_nibbles lower 128 = low nibbles of bytes 0-15 (elements 0-15)
        // high_nibbles lower 128 = high nibbles of bytes 0-15 (elements 16-31)
        let low_128 = _mm256_castsi256_si128(low_nibbles);
        let high_128 = _mm256_castsi256_si128(high_nibbles);
        let quants = _mm256_set_m128i(high_128, low_128);

        // Subtract zero-point 8 to get signed values (-8..7)
        // For maddubs, we need unsigned first arg. The nibbles (0-15) are already unsigned.
        // Instead of subtracting 8 and dealing with signs, we compute:
        // dot(nibble - 8, q_inp) = dot(nibble, q_inp) - 8 * sum(q_inp)
        // This avoids the subtraction and keeps nibbles as u8 for maddubs.

        let q_inp = _mm256_loadu_si256(iq_ptr);

        // dot(nibble, q_inp) as unsigned * signed
        let prod_16 = _mm256_maddubs_epi16(quants, q_inp);
        let prod_32 = _mm256_madd_epi16(prod_16, ones_16);

        // 8 * sum(q_inp): sum the signed i8 input values in groups of 4
        // Use maddubs(all_1_u8, q_inp_i8) to sum adjacent pairs preserving sign,
        // then madd with 8 to sum pairs again and multiply by 8.
        let ones_u8 = _mm256_set1_epi8(1);
        let eight_i16 = _mm256_set1_epi16(8);
        let sum_pairs_16 = _mm256_maddubs_epi16(ones_u8, q_inp); // u8(1)*i8 → i16 pair sums
        let sum_inp_32 = _mm256_madd_epi16(sum_pairs_16, eight_i16); // i16 pair sums * 8 → i32

        // Subtract: prod - 8*sum
        let result_32 = _mm256_sub_epi32(prod_32, sum_inp_32);

        let result_f32 = _mm256_cvtepi32_ps(result_32);
        acc = _mm256_fmadd_ps(combined_scale, result_f32, acc);
    }

    hsum_f32_avx2(acc)
}

// ============================================================================
// Float-domain SIMD kernels (fallback when input is not pre-quantized)
// ============================================================================

/// Compute dot product of a Q8_0 quantized row with an f32 input vector using AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_q8_0_avx2(row: &[u8], input: &[f32]) -> f32 {
    let block_size = 2 + QK8_0;
    let n_blocks = row.len() / block_size;
    let mut acc = _mm256_setzero_ps();

    for block_idx in 0..n_blocks {
        let block_ptr = row.as_ptr().add(block_idx * block_size);
        let d_bytes = [*block_ptr, *block_ptr.add(1)];
        let scale = f16::from_le_bytes(d_bytes).to_f32();
        let scale_vec = _mm256_set1_ps(scale);
        let qs_ptr = block_ptr.add(2);
        let input_ptr = input.as_ptr().add(block_idx * QK8_0);
        let mut block_acc = _mm256_setzero_ps();

        for g in (0..32).step_by(8) {
            let q_i8 = _mm_loadl_epi64(qs_ptr.add(g) as *const __m128i);
            let q_i32 = _mm256_cvtepi8_epi32(q_i8);
            let q_f32 = _mm256_cvtepi32_ps(q_i32);
            let inp = _mm256_loadu_ps(input_ptr.add(g));
            block_acc = _mm256_fmadd_ps(q_f32, inp, block_acc);
        }

        acc = _mm256_fmadd_ps(scale_vec, block_acc, acc);
    }

    hsum_f32_avx2(acc)
}

/// Compute dot product of a Q4_0 quantized row with an f32 input vector using AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_q4_0_avx2(row: &[u8], input: &[f32]) -> f32 {
    let block_size = 2 + QK4_0 / 2;
    let n_blocks = row.len() / block_size;
    let mask_low = _mm256_set1_epi8(0x0f);
    let offset_8 = _mm256_set1_epi8(8);
    let mut acc = _mm256_setzero_ps();

    for block_idx in 0..n_blocks {
        let block_ptr = row.as_ptr().add(block_idx * block_size);
        let d_bytes = [*block_ptr, *block_ptr.add(1)];
        let scale = f16::from_le_bytes(d_bytes).to_f32();
        let scale_vec = _mm256_set1_ps(scale);
        let qs_ptr = block_ptr.add(2);
        let input_ptr = input.as_ptr().add(block_idx * QK4_0);

        let packed_128 = _mm_loadu_si128(qs_ptr as *const __m128i);
        let packed_256 = _mm256_castsi128_si256(packed_128);
        let packed_256 = _mm256_inserti128_si256(packed_256, packed_128, 1);

        let low_nibbles = _mm256_and_si256(packed_256, mask_low);
        let high_nibbles = _mm256_and_si256(_mm256_srli_epi16(packed_256, 4), mask_low);
        let low_signed = _mm256_sub_epi8(low_nibbles, offset_8);
        let high_signed = _mm256_sub_epi8(high_nibbles, offset_8);

        let low_128_0 = _mm256_castsi256_si128(low_signed);
        let q_i32 = _mm256_cvtepi8_epi32(low_128_0);
        let q_f32 = _mm256_cvtepi32_ps(q_i32);
        let inp = _mm256_loadu_ps(input_ptr);
        let mut block_acc = _mm256_mul_ps(q_f32, inp);

        let low_128_0_shifted = _mm_srli_si128(low_128_0, 8);
        let q_i32 = _mm256_cvtepi8_epi32(low_128_0_shifted);
        let q_f32 = _mm256_cvtepi32_ps(q_i32);
        let inp = _mm256_loadu_ps(input_ptr.add(8));
        block_acc = _mm256_fmadd_ps(q_f32, inp, block_acc);

        let high_128_0 = _mm256_castsi256_si128(high_signed);
        let q_i32 = _mm256_cvtepi8_epi32(high_128_0);
        let q_f32 = _mm256_cvtepi32_ps(q_i32);
        let inp = _mm256_loadu_ps(input_ptr.add(16));
        block_acc = _mm256_fmadd_ps(q_f32, inp, block_acc);

        let high_128_0_shifted = _mm_srli_si128(high_128_0, 8);
        let q_i32 = _mm256_cvtepi8_epi32(high_128_0_shifted);
        let q_f32 = _mm256_cvtepi32_ps(q_i32);
        let inp = _mm256_loadu_ps(input_ptr.add(24));
        block_acc = _mm256_fmadd_ps(q_f32, inp, block_acc);

        acc = _mm256_fmadd_ps(scale_vec, block_acc, acc);
    }

    hsum_f32_avx2(acc)
}

/// Compute dot product of two f32 slices using AVX2 + FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    let chunks = n / 32;
    for i in 0..chunks {
        let base = i * 32;
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a_ptr.add(base)), _mm256_loadu_ps(b_ptr.add(base)), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a_ptr.add(base + 8)), _mm256_loadu_ps(b_ptr.add(base + 8)), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a_ptr.add(base + 16)), _mm256_loadu_ps(b_ptr.add(base + 16)), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a_ptr.add(base + 24)), _mm256_loadu_ps(b_ptr.add(base + 24)), acc3);
    }

    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc0 = _mm256_add_ps(acc0, acc2);

    let mut sum = hsum_f32_avx2(acc0);

    for i in (chunks * 32)..n {
        sum += *a_ptr.add(i) * *b_ptr.add(i);
    }
    sum
}

// ============================================================================
// Utility
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_f32_avx2(v: __m256) -> f32 {
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo128, hi128);
    let hi64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, hi64);
    let hi32 = _mm_shuffle_ps(sum64, sum64, 0x01);
    let sum32 = _mm_add_ss(sum64, hi32);
    _mm_cvtss_f32(sum32)
}

/// Returns true if the CPU supports AVX2 + FMA.
pub fn has_avx2_fma() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::quantize::dot_q4_0;

    #[test]
    fn test_q4_0_integer_vs_scalar() {
        let scale_f16 = f16::from_f32(0.5);
        let scale_bytes = scale_f16.to_le_bytes();
        let nibble_data: [u8; 16] = [
            0x73, 0x51, 0xA2, 0xB4, 0x60, 0xF8, 0xD9, 0xE3,
            0x41, 0x25, 0x87, 0xC6, 0x3A, 0x9B, 0x0F, 0xDE,
        ];
        let mut row = Vec::with_capacity(18);
        row.extend_from_slice(&scale_bytes);
        row.extend_from_slice(&nibble_data);

        let input: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let scalar_result = dot_q4_0(&row, &input);
        let (scales, quants) = quantize_f32_to_q8_0(&input);
        let simd_result = unsafe { dot_q4_0_q8_0_avx2(&row, &scales, &quants) };

        let diff = (scalar_result - simd_result).abs();
        eprintln!("Scalar: {scalar_result}, SIMD int: {simd_result}, diff: {diff}");
        assert!(diff < 0.1, "Absolute error too large: {diff}");
    }

    #[test]
    fn test_q4_0_large_vector() {
        let n = 2048;
        let n_blocks = n / 32;
        let mut row = Vec::with_capacity(n_blocks * 18);
        for bi in 0..n_blocks {
            let scale = f16::from_f32(0.1 + bi as f32 * 0.01);
            row.extend_from_slice(&scale.to_le_bytes());
            for i in 0..16u8 {
                let low = (i + bi as u8) % 16;
                let high = (i.wrapping_mul(3) + (bi as u8).wrapping_mul(2)) % 16;
                row.push(low | (high << 4));
            }
        }

        let input: Vec<f32> = (0..n).map(|i| ((i as f32 * 7.13).sin()) * 2.0).collect();
        let scalar_result = dot_q4_0(&row, &input);
        let (scales, quants) = quantize_f32_to_q8_0(&input);
        let simd_result = unsafe { dot_q4_0_q8_0_avx2(&row, &scales, &quants) };

        let diff = (scalar_result - simd_result).abs();
        let rel_err = diff / scalar_result.abs().max(1e-6);
        eprintln!("Large: Scalar: {scalar_result}, SIMD int: {simd_result}, diff: {diff}, rel_err: {rel_err}");
        assert!(rel_err < 0.1, "Relative error too large: {rel_err}");
    }
}
