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
        // Prefetch next blocks to hide memory latency
        if bi + 2 < n_blocks {
            _mm_prefetch(row.as_ptr().add((bi + 2) * block_size) as *const i8, _MM_HINT_T0);
            _mm_prefetch(input_quants.as_ptr().add((bi + 2) * QK8_0) as *const i8, _MM_HINT_T0);
        }

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
        // Prefetch next blocks to hide memory latency
        if bi + 2 < n_blocks {
            _mm_prefetch(row.as_ptr().add((bi + 2) * block_size) as *const i8, _MM_HINT_T0);
            _mm_prefetch(input_quants.as_ptr().add((bi + 2) * QK4_0) as *const i8, _MM_HINT_T0);
        }

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
        // Prefetch next block to hide memory latency
        if block_idx + 2 < n_blocks {
            _mm_prefetch(row.as_ptr().add((block_idx + 2) * block_size) as *const i8, _MM_HINT_T0);
        }

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
        // Prefetch next block to hide memory latency
        if block_idx + 2 < n_blocks {
            _mm_prefetch(row.as_ptr().add((block_idx + 2) * block_size) as *const i8, _MM_HINT_T0);
        }

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
// Q4_K float-domain AVX2 dot product
// Block: d(f16) + dmin(f16) + scales[12] + qs[128] = 144 bytes for 256 elements
// ============================================================================

const QK_K: usize = 256;
const BLOCK_Q4_K: usize = 144;
const BLOCK_Q5_K: usize = 176;
const BLOCK_Q6_K: usize = 210;

#[inline(always)]
fn get_scale_min_k4(index: usize, packed: &[u8; 12]) -> (u8, u8) {
    if index < 4 {
        (packed[index] & 0x3f, packed[index + 4] & 0x3f)
    } else {
        (
            ((packed[index + 4] & 0x0f) | ((packed[index - 4] >> 6) << 4)) & 0x3f,
            ((packed[index + 4] >> 4) | ((packed[index] >> 6) << 4)) & 0x3f,
        )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_q4_k_avx2(row: &[u8], input: &[f32]) -> f32 {
    let n_blocks = row.len() / BLOCK_Q4_K;
    let mask_low_128 = _mm_set1_epi8(0x0f);
    let mut acc = _mm256_setzero_ps();

    for bi in 0..n_blocks {
        let block_ptr = row.as_ptr().add(bi * BLOCK_Q4_K);

        if bi + 1 < n_blocks {
            _mm_prefetch(block_ptr.add(BLOCK_Q4_K) as *const i8, _MM_HINT_T0);
        }

        let d = f16::from_le_bytes([*block_ptr, *block_ptr.add(1)]).to_f32();
        let dmin = f16::from_le_bytes([*block_ptr.add(2), *block_ptr.add(3)]).to_f32();

        let mut scales_raw = [0u8; 12];
        std::ptr::copy_nonoverlapping(block_ptr.add(4), scales_raw.as_mut_ptr(), 12);

        let qs_ptr = block_ptr.add(16);
        let inp = input.as_ptr().add(bi * QK_K);

        for group in 0..4 {
            let q_ptr = qs_ptr.add(group * 32);
            let (sc1, m1) = get_scale_min_k4(group * 2, &scales_raw);
            let (sc2, m2) = get_scale_min_k4(group * 2 + 1, &scales_raw);

            let d1 = _mm256_set1_ps(d * sc1 as f32);
            let d2 = _mm256_set1_ps(d * sc2 as f32);
            let min1 = _mm256_set1_ps(dmin * m1 as f32);
            let min2 = _mm256_set1_ps(dmin * m2 as f32);
            let base = group * 64;

            // Low nibbles: 32 elements in 4 chunks of 8
            for chunk in 0..4 {
                let off = chunk * 8;
                let raw = _mm_loadl_epi64(q_ptr.add(off) as *const __m128i);
                let nibbles = _mm_and_si128(raw, mask_low_128);
                let q_i32 = _mm256_cvtepu8_epi32(nibbles);
                let q_f32 = _mm256_cvtepi32_ps(q_i32);
                let inp_vec = _mm256_loadu_ps(inp.add(base + off));
                // (d1 * nibble - min1) * input
                let scaled = _mm256_fmsub_ps(d1, q_f32, min1);
                acc = _mm256_fmadd_ps(scaled, inp_vec, acc);
            }

            // High nibbles: 32 elements in 4 chunks of 8
            for chunk in 0..4 {
                let off = chunk * 8;
                let raw = _mm_loadl_epi64(q_ptr.add(off) as *const __m128i);
                let shifted = _mm_srli_epi16(raw, 4);
                let nibbles = _mm_and_si128(shifted, mask_low_128);
                let q_i32 = _mm256_cvtepu8_epi32(nibbles);
                let q_f32 = _mm256_cvtepi32_ps(q_i32);
                let inp_vec = _mm256_loadu_ps(inp.add(base + 32 + off));
                let scaled = _mm256_fmsub_ps(d2, q_f32, min2);
                acc = _mm256_fmadd_ps(scaled, inp_vec, acc);
            }
        }
    }

    hsum_f32_avx2(acc)
}

// ============================================================================
// Q5_K float-domain AVX2 dot product
// Block: d(f16) + dmin(f16) + scales[12] + qh[32] + qs[128] = 176 bytes for 256 elements
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_q5_k_avx2(row: &[u8], input: &[f32]) -> f32 {
    let n_blocks = row.len() / BLOCK_Q5_K;
    let mask_low_128 = _mm_set1_epi8(0x0f);
    let sixteen = _mm_set1_epi8(16);
    let mut acc = _mm256_setzero_ps();

    for bi in 0..n_blocks {
        let block_ptr = row.as_ptr().add(bi * BLOCK_Q5_K);

        if bi + 1 < n_blocks {
            _mm_prefetch(block_ptr.add(BLOCK_Q5_K) as *const i8, _MM_HINT_T0);
        }

        let d = f16::from_le_bytes([*block_ptr, *block_ptr.add(1)]).to_f32();
        let dmin = f16::from_le_bytes([*block_ptr.add(2), *block_ptr.add(3)]).to_f32();

        let mut scales_raw = [0u8; 12];
        std::ptr::copy_nonoverlapping(block_ptr.add(4), scales_raw.as_mut_ptr(), 12);

        let qh_ptr = block_ptr.add(16); // 32 bytes of high bits
        let qs_ptr = block_ptr.add(48); // 128 bytes of low nibbles
        let inp = input.as_ptr().add(bi * QK_K);

        for group in 0..4 {
            let q_ptr = qs_ptr.add(group * 32);
            let (sc1, m1) = get_scale_min_k4(group * 2, &scales_raw);
            let (sc2, m2) = get_scale_min_k4(group * 2 + 1, &scales_raw);

            let d1 = _mm256_set1_ps(d * sc1 as f32);
            let d2 = _mm256_set1_ps(d * sc2 as f32);
            let min1 = _mm256_set1_ps(dmin * m1 as f32);
            let min2 = _mm256_set1_ps(dmin * m2 as f32);
            let base = group * 64;

            let high_mask_low = 1u8 << (group * 2);
            let high_mask_high = 1u8 << (group * 2 + 1);
            let hmask_low = _mm_set1_epi8(high_mask_low as i8);
            let hmask_high = _mm_set1_epi8(high_mask_high as i8);

            // Low nibbles + 5th bit: 32 elements in 4 chunks of 8
            for chunk in 0..4 {
                let off = chunk * 8;
                let raw = _mm_loadl_epi64(q_ptr.add(off) as *const __m128i);
                let nibbles = _mm_and_si128(raw, mask_low_128);
                // Extract 5th bit from qh
                let qh_raw = _mm_loadl_epi64(qh_ptr.add(off) as *const __m128i);
                let has_bit = _mm_cmpeq_epi8(_mm_and_si128(qh_raw, hmask_low), hmask_low);
                let bit5 = _mm_and_si128(has_bit, sixteen); // 16 where bit is set, 0 otherwise
                let q5 = _mm_add_epi8(nibbles, bit5);
                let q_i32 = _mm256_cvtepu8_epi32(q5);
                let q_f32 = _mm256_cvtepi32_ps(q_i32);
                let inp_vec = _mm256_loadu_ps(inp.add(base + off));
                let scaled = _mm256_fmsub_ps(d1, q_f32, min1);
                acc = _mm256_fmadd_ps(scaled, inp_vec, acc);
            }

            // High nibbles + 5th bit: 32 elements in 4 chunks of 8
            for chunk in 0..4 {
                let off = chunk * 8;
                let raw = _mm_loadl_epi64(q_ptr.add(off) as *const __m128i);
                let shifted = _mm_srli_epi16(raw, 4);
                let nibbles = _mm_and_si128(shifted, mask_low_128);
                let qh_raw = _mm_loadl_epi64(qh_ptr.add(off) as *const __m128i);
                let has_bit = _mm_cmpeq_epi8(_mm_and_si128(qh_raw, hmask_high), hmask_high);
                let bit5 = _mm_and_si128(has_bit, sixteen);
                let q5 = _mm_add_epi8(nibbles, bit5);
                let q_i32 = _mm256_cvtepu8_epi32(q5);
                let q_f32 = _mm256_cvtepi32_ps(q_i32);
                let inp_vec = _mm256_loadu_ps(inp.add(base + 32 + off));
                let scaled = _mm256_fmsub_ps(d2, q_f32, min2);
                acc = _mm256_fmadd_ps(scaled, inp_vec, acc);
            }
        }
    }

    hsum_f32_avx2(acc)
}

// ============================================================================
// Q6_K float-domain AVX2 dot product
// Block: ql[128] + qh[64] + scales[16](i8) + d(f16) = 210 bytes for 256 elements
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_q6_k_avx2(row: &[u8], input: &[f32]) -> f32 {
    let n_blocks = row.len() / BLOCK_Q6_K;
    let mask_low_128 = _mm_set1_epi8(0x0f);
    let mask_2bit = _mm_set1_epi8(0x03);
    let offset_32 = _mm256_set1_ps(32.0);
    let mut acc = _mm256_setzero_ps();

    for bi in 0..n_blocks {
        let block_ptr = row.as_ptr().add(bi * BLOCK_Q6_K);

        if bi + 1 < n_blocks {
            _mm_prefetch(block_ptr.add(BLOCK_Q6_K) as *const i8, _MM_HINT_T0);
        }

        let ql_ptr = block_ptr;           // 128 bytes
        let qh_ptr = block_ptr.add(128);   // 64 bytes
        let sc_ptr = block_ptr.add(192);   // 16 bytes (i8 scales)
        let d = f16::from_le_bytes([*block_ptr.add(208), *block_ptr.add(209)]).to_f32();

        let inp = input.as_ptr().add(bi * QK_K);

        // 2 groups of 128 elements
        for group in 0..2 {
            let ql = ql_ptr.add(group * 64);
            let qh = qh_ptr.add(group * 32);
            let scales = sc_ptr.add(group * 8);
            let gi = inp.add(group * 128);

            // 4 sub-groups of 32 elements:
            // sub 0: ql[0..31] low nibble + qh[0..31] bits 0-1, scale[0 or 1]
            // sub 1: ql[32..63] low nibble + qh[0..31] bits 2-3, scale[2 or 3]
            // sub 2: ql[0..31] high nibble + qh[0..31] bits 4-5, scale[4 or 5]
            // sub 3: ql[32..63] high nibble + qh[0..31] bits 6-7, scale[6 or 7]

            // Sub-group 0: elements [0..31]
            {
                let scale_idx = 0;
                for chunk in 0..4 {
                    let off = chunk * 8;
                    let si = (off / 16) + scale_idx;
                    let s = *scales.add(si) as i8;
                    let ds = _mm256_set1_ps(d * s as f32);

                    let ql_raw = _mm_loadl_epi64(ql.add(off) as *const __m128i);
                    let ql_lo = _mm_and_si128(ql_raw, mask_low_128);
                    let qh_raw = _mm_loadl_epi64(qh.add(off) as *const __m128i);
                    let qh_bits = _mm_and_si128(qh_raw, mask_2bit);
                    let qh_shifted = _mm_slli_epi16(qh_bits, 4);
                    let qh_masked = _mm_and_si128(qh_shifted, _mm_set1_epi8(0x30));
                    let q6 = _mm_or_si128(ql_lo, qh_masked);

                    let q_i32 = _mm256_cvtepu8_epi32(q6);
                    let q_f32 = _mm256_cvtepi32_ps(q_i32);
                    let q_centered = _mm256_sub_ps(q_f32, offset_32);

                    let inp_vec = _mm256_loadu_ps(gi.add(off));
                    acc = _mm256_fmadd_ps(_mm256_mul_ps(ds, q_centered), inp_vec, acc);
                }
            }

            // Sub-group 1: elements [32..63]
            {
                let scale_idx = 2;
                for chunk in 0..4 {
                    let off = chunk * 8;
                    let si = (off / 16) + scale_idx;
                    let s = *scales.add(si) as i8;
                    let ds = _mm256_set1_ps(d * s as f32);

                    let ql_raw = _mm_loadl_epi64(ql.add(32 + off) as *const __m128i);
                    let ql_lo = _mm_and_si128(ql_raw, mask_low_128);
                    let qh_raw = _mm_loadl_epi64(qh.add(off) as *const __m128i);
                    let qh_shifted = _mm_srli_epi16(qh_raw, 2);
                    let qh_bits = _mm_and_si128(qh_shifted, mask_2bit);
                    let qh_hi = _mm_slli_epi16(qh_bits, 4);
                    let qh_masked = _mm_and_si128(qh_hi, _mm_set1_epi8(0x30));
                    let q6 = _mm_or_si128(ql_lo, qh_masked);

                    let q_i32 = _mm256_cvtepu8_epi32(q6);
                    let q_f32 = _mm256_cvtepi32_ps(q_i32);
                    let q_centered = _mm256_sub_ps(q_f32, offset_32);

                    let inp_vec = _mm256_loadu_ps(gi.add(32 + off));
                    acc = _mm256_fmadd_ps(_mm256_mul_ps(ds, q_centered), inp_vec, acc);
                }
            }

            // Sub-group 2: elements [64..95]
            {
                let scale_idx = 4;
                for chunk in 0..4 {
                    let off = chunk * 8;
                    let si = (off / 16) + scale_idx;
                    let s = *scales.add(si) as i8;
                    let ds = _mm256_set1_ps(d * s as f32);

                    let ql_raw = _mm_loadl_epi64(ql.add(off) as *const __m128i);
                    let ql_hi = _mm_and_si128(_mm_srli_epi16(ql_raw, 4), mask_low_128);
                    let qh_raw = _mm_loadl_epi64(qh.add(off) as *const __m128i);
                    let qh_shifted = _mm_srli_epi16(qh_raw, 4);
                    let qh_bits = _mm_and_si128(qh_shifted, mask_2bit);
                    let qh_hi = _mm_slli_epi16(qh_bits, 4);
                    let qh_masked = _mm_and_si128(qh_hi, _mm_set1_epi8(0x30));
                    let q6 = _mm_or_si128(ql_hi, qh_masked);

                    let q_i32 = _mm256_cvtepu8_epi32(q6);
                    let q_f32 = _mm256_cvtepi32_ps(q_i32);
                    let q_centered = _mm256_sub_ps(q_f32, offset_32);

                    let inp_vec = _mm256_loadu_ps(gi.add(64 + off));
                    acc = _mm256_fmadd_ps(_mm256_mul_ps(ds, q_centered), inp_vec, acc);
                }
            }

            // Sub-group 3: elements [96..127]
            {
                let scale_idx = 6;
                for chunk in 0..4 {
                    let off = chunk * 8;
                    let si = (off / 16) + scale_idx;
                    let s = *scales.add(si) as i8;
                    let ds = _mm256_set1_ps(d * s as f32);

                    let ql_raw = _mm_loadl_epi64(ql.add(32 + off) as *const __m128i);
                    let ql_hi = _mm_and_si128(_mm_srli_epi16(ql_raw, 4), mask_low_128);
                    let qh_raw = _mm_loadl_epi64(qh.add(off) as *const __m128i);
                    let qh_shifted = _mm_srli_epi16(qh_raw, 6);
                    let qh_bits = _mm_and_si128(qh_shifted, mask_2bit);
                    let qh_hi = _mm_slli_epi16(qh_bits, 4);
                    let qh_masked = _mm_and_si128(qh_hi, _mm_set1_epi8(0x30));
                    let q6 = _mm_or_si128(ql_hi, qh_masked);

                    let q_i32 = _mm256_cvtepu8_epi32(q6);
                    let q_f32 = _mm256_cvtepi32_ps(q_i32);
                    let q_centered = _mm256_sub_ps(q_f32, offset_32);

                    let inp_vec = _mm256_loadu_ps(gi.add(96 + off));
                    acc = _mm256_fmadd_ps(_mm256_mul_ps(ds, q_centered), inp_vec, acc);
                }
            }
        }
    }

    hsum_f32_avx2(acc)
}

// ============================================================================
// Fast vectorized exp (AVX2+FMA)
// Uses exp(x) = exp2(x * log2(e)) with polynomial approximation for 2^r
// Accuracy: ~0.1% relative error, sufficient for softmax and sigmoid
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn fast_exp_avx2(x: __m256) -> __m256 {
    // exp(x) = 2^(x * log2(e))
    let log2e = _mm256_set1_ps(1.4426950408889634f32);
    let t = _mm256_mul_ps(x, log2e);

    // Clamp to prevent overflow/underflow
    let t = _mm256_max_ps(t, _mm256_set1_ps(-126.0));
    let t = _mm256_min_ps(t, _mm256_set1_ps(126.0));

    // n = round(t), r = t - n (so r is in [-0.5, 0.5])
    let n = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    let r = _mm256_sub_ps(t, n);

    // 2^n via IEEE 754 exponent manipulation
    let n_i32 = _mm256_cvtps_epi32(n);
    let exp_i = _mm256_slli_epi32(_mm256_add_epi32(n_i32, _mm256_set1_epi32(127)), 23);
    let two_n = _mm256_castsi256_ps(exp_i);

    // Polynomial approximation for 2^r on [-0.5, 0.5]
    // p(r) = 1 + r*(c1 + r*(c2 + r*(c3 + r*c4)))
    let mut p = _mm256_set1_ps(0.00960083f32);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.05550862));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.24015523));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.69315863));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0));

    _mm256_mul_ps(two_n, p)
}

// ============================================================================
// AVX2 softmax
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn softmax_avx2(values: &mut [f32]) {
    let n = values.len();
    if n == 0 {
        return;
    }

    let chunks = n / 8;
    let ptr = values.as_mut_ptr();

    // Pass 1: find max
    let mut max_vec = _mm256_set1_ps(f32::NEG_INFINITY);
    for i in 0..chunks {
        let v = _mm256_loadu_ps(ptr.add(i * 8));
        max_vec = _mm256_max_ps(max_vec, v);
    }
    let mut max_val = hsum_max_avx2(max_vec);
    for i in (chunks * 8)..n {
        max_val = max_val.max(*ptr.add(i));
    }
    let max_broadcast = _mm256_set1_ps(max_val);

    // Pass 2: exp(x - max) and sum
    let mut sum_vec = _mm256_setzero_ps();
    for i in 0..chunks {
        let base = i * 8;
        let v = _mm256_loadu_ps(ptr.add(base));
        let shifted = _mm256_sub_ps(v, max_broadcast);
        let exp_val = fast_exp_avx2(shifted);
        _mm256_storeu_ps(ptr.add(base), exp_val);
        sum_vec = _mm256_add_ps(sum_vec, exp_val);
    }
    let mut sum = hsum_f32_avx2(sum_vec);
    for i in (chunks * 8)..n {
        let v = (*ptr.add(i) - max_val).exp();
        *ptr.add(i) = v;
        sum += v;
    }

    if sum == 0.0 {
        return;
    }

    // Pass 3: divide by sum
    let inv_sum = _mm256_set1_ps(1.0 / sum);
    for i in 0..chunks {
        let base = i * 8;
        let v = _mm256_loadu_ps(ptr.add(base));
        _mm256_storeu_ps(ptr.add(base), _mm256_mul_ps(v, inv_sum));
    }
    let inv_sum_scalar = 1.0 / sum;
    for i in (chunks * 8)..n {
        *ptr.add(i) *= inv_sum_scalar;
    }
}

/// Horizontal max of 8 f32 lanes
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_max_avx2(v: __m256) -> f32 {
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let max128 = _mm_max_ps(lo128, hi128);
    let hi64 = _mm_movehl_ps(max128, max128);
    let max64 = _mm_max_ps(max128, hi64);
    let hi32 = _mm_shuffle_ps(max64, max64, 0x01);
    let max32 = _mm_max_ss(max64, hi32);
    _mm_cvtss_f32(max32)
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

    /// Build a synthetic Q4_K block (144 bytes) for testing.
    fn make_q4_k_block(d: f32, dmin: f32) -> Vec<u8> {
        let mut block = Vec::with_capacity(BLOCK_Q4_K);
        block.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        block.extend_from_slice(&f16::from_f32(dmin).to_le_bytes());
        // scales: 12 bytes, use simple values (all lower 6 bits)
        let scales: [u8; 12] = [3, 5, 2, 7, 1, 4, 6, 3, 0x12, 0x34, 0x56, 0x78];
        block.extend_from_slice(&scales);
        // qs: 128 bytes of packed nibbles
        for i in 0..128u8 {
            let low = (i.wrapping_mul(3)) % 16;
            let high = (i.wrapping_mul(7).wrapping_add(5)) % 16;
            block.push(low | (high << 4));
        }
        block
    }

    /// Build a synthetic Q5_K block (176 bytes) for testing.
    fn make_q5_k_block(d: f32, dmin: f32) -> Vec<u8> {
        let mut block = Vec::with_capacity(BLOCK_Q5_K);
        block.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        block.extend_from_slice(&f16::from_f32(dmin).to_le_bytes());
        let scales: [u8; 12] = [3, 5, 2, 7, 1, 4, 6, 3, 0x12, 0x34, 0x56, 0x78];
        block.extend_from_slice(&scales);
        // qh: 32 bytes of high bits
        for i in 0..32u8 {
            block.push(i.wrapping_mul(0x55) ^ 0xAA);
        }
        // qs: 128 bytes
        for i in 0..128u8 {
            let low = (i.wrapping_mul(3)) % 16;
            let high = (i.wrapping_mul(7).wrapping_add(5)) % 16;
            block.push(low | (high << 4));
        }
        block
    }

    /// Build a synthetic Q6_K block (210 bytes) for testing.
    fn make_q6_k_block(d: f32) -> Vec<u8> {
        let mut block = Vec::with_capacity(BLOCK_Q6_K);
        // ql: 128 bytes
        for i in 0..128u8 {
            let low = (i.wrapping_mul(3)) % 16;
            let high = (i.wrapping_mul(7).wrapping_add(5)) % 16;
            block.push(low | (high << 4));
        }
        // qh: 64 bytes
        for i in 0..64u8 {
            block.push(i.wrapping_mul(0x37) ^ 0xC5);
        }
        // scales: 16 signed i8
        for i in 0..16i8 {
            block.push((i.wrapping_mul(3).wrapping_sub(7)) as u8);
        }
        // d: f16
        block.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        block
    }

    #[test]
    fn test_q4_k_avx2_vs_scalar() {
        if !has_avx2_fma() { return; }
        let row = make_q4_k_block(0.5, 0.1);
        let input: Vec<f32> = (0..QK_K).map(|i| ((i as f32 * 1.37).sin()) * 2.0).collect();

        // Disable AVX2 dispatch by calling scalar directly via the original function logic
        let scalar = scalar_dot_q4_k(&row, &input);
        let simd = unsafe { dot_q4_k_avx2(&row, &input) };

        let diff = (scalar - simd).abs();
        let rel_err = diff / scalar.abs().max(1e-6);
        eprintln!("Q4_K: scalar={scalar}, simd={simd}, diff={diff}, rel_err={rel_err}");
        assert!(rel_err < 0.01, "Q4_K relative error too large: {rel_err}");
    }

    #[test]
    fn test_q5_k_avx2_vs_scalar() {
        if !has_avx2_fma() { return; }
        let row = make_q5_k_block(0.5, 0.1);
        let input: Vec<f32> = (0..QK_K).map(|i| ((i as f32 * 1.37).sin()) * 2.0).collect();

        let scalar = scalar_dot_q5_k(&row, &input);
        let simd = unsafe { dot_q5_k_avx2(&row, &input) };

        let diff = (scalar - simd).abs();
        let rel_err = diff / scalar.abs().max(1e-6);
        eprintln!("Q5_K: scalar={scalar}, simd={simd}, diff={diff}, rel_err={rel_err}");
        assert!(rel_err < 0.01, "Q5_K relative error too large: {rel_err}");
    }

    #[test]
    fn test_q6_k_avx2_vs_scalar() {
        if !has_avx2_fma() { return; }
        let row = make_q6_k_block(0.5);
        let input: Vec<f32> = (0..QK_K).map(|i| ((i as f32 * 1.37).sin()) * 2.0).collect();

        let scalar = scalar_dot_q6_k(&row, &input);
        let simd = unsafe { dot_q6_k_avx2(&row, &input) };

        let diff = (scalar - simd).abs();
        let rel_err = diff / scalar.abs().max(1e-6);
        eprintln!("Q6_K: scalar={scalar}, simd={simd}, diff={diff}, rel_err={rel_err}");
        assert!(rel_err < 0.01, "Q6_K relative error too large: {rel_err}");
    }

    #[test]
    fn test_q4_k_large_vector() {
        if !has_avx2_fma() { return; }
        let n_blocks = 8;
        let mut row = Vec::new();
        for bi in 0..n_blocks {
            row.extend(make_q4_k_block(0.1 + bi as f32 * 0.05, 0.02 + bi as f32 * 0.01));
        }
        let n = n_blocks * QK_K;
        let input: Vec<f32> = (0..n).map(|i| ((i as f32 * 7.13).sin()) * 2.0).collect();

        let scalar = scalar_dot_q4_k(&row, &input);
        let simd = unsafe { dot_q4_k_avx2(&row, &input) };

        let diff = (scalar - simd).abs();
        let rel_err = diff / scalar.abs().max(1e-6);
        eprintln!("Q4_K large: scalar={scalar}, simd={simd}, diff={diff}, rel_err={rel_err}");
        assert!(rel_err < 0.01, "Q4_K large relative error too large: {rel_err}");
    }

    /// Scalar Q4_K dot product for testing (bypasses AVX2 dispatch)
    fn scalar_dot_q4_k(row: &[u8], input: &[f32]) -> f32 {
        use bytemuck::pod_read_unaligned;
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct BQ4K { d: f16, dmin: f16, scales: [u8; 12], qs: [u8; 128] }

        let mut sum = 0.0f32;
        for (bi, chunk) in row.chunks_exact(BLOCK_Q4_K).enumerate() {
            let block: BQ4K = pod_read_unaligned(chunk);
            let d = block.d.to_f32();
            let dmin = block.dmin.to_f32();
            let inp = &input[bi * QK_K..(bi + 1) * QK_K];
            for group in 0..4 {
                let q = &block.qs[group * 32..(group + 1) * 32];
                let (sc1, m1) = get_scale_min_k4(group * 2, &block.scales);
                let (sc2, m2) = get_scale_min_k4(group * 2 + 1, &block.scales);
                let d1 = d * sc1 as f32;
                let d2 = d * sc2 as f32;
                let min1 = dmin * m1 as f32;
                let min2 = dmin * m2 as f32;
                let base = group * 64;
                for lane in 0..32 {
                    sum += (d1 * (q[lane] & 0x0f) as f32 - min1) * inp[base + lane];
                    sum += (d2 * (q[lane] >> 4) as f32 - min2) * inp[base + 32 + lane];
                }
            }
        }
        sum
    }

    /// Scalar Q5_K dot product for testing
    fn scalar_dot_q5_k(row: &[u8], input: &[f32]) -> f32 {
        use bytemuck::pod_read_unaligned;
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct BQ5K { d: f16, dmin: f16, scales: [u8; 12], qh: [u8; 32], qs: [u8; 128] }

        let mut sum = 0.0f32;
        for (bi, chunk) in row.chunks_exact(BLOCK_Q5_K).enumerate() {
            let block: BQ5K = pod_read_unaligned(chunk);
            let d = block.d.to_f32();
            let dmin = block.dmin.to_f32();
            let inp = &input[bi * QK_K..(bi + 1) * QK_K];
            for group in 0..4 {
                let ql = &block.qs[group * 32..(group + 1) * 32];
                let (sc1, m1) = get_scale_min_k4(group * 2, &block.scales);
                let (sc2, m2) = get_scale_min_k4(group * 2 + 1, &block.scales);
                let d1 = d * sc1 as f32;
                let d2 = d * sc2 as f32;
                let min1 = dmin * m1 as f32;
                let min2 = dmin * m2 as f32;
                let hmask_low = 1u8 << (group * 2);
                let hmask_high = 1u8 << (group * 2 + 1);
                let base = group * 64;
                for lane in 0..32 {
                    let low = (ql[lane] & 0x0f) as i32
                        + if (block.qh[lane] & hmask_low) != 0 { 16 } else { 0 };
                    let high = (ql[lane] >> 4) as i32
                        + if (block.qh[lane] & hmask_high) != 0 { 16 } else { 0 };
                    sum += (d1 * low as f32 - min1) * inp[base + lane];
                    sum += (d2 * high as f32 - min2) * inp[base + 32 + lane];
                }
            }
        }
        sum
    }

    /// Scalar Q6_K dot product for testing
    fn scalar_dot_q6_k(row: &[u8], input: &[f32]) -> f32 {
        use bytemuck::pod_read_unaligned;
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct BQ6K { ql: [u8; 128], qh: [u8; 64], scales: [i8; 16], d: f16 }

        let mut sum = 0.0f32;
        for (bi, chunk) in row.chunks_exact(BLOCK_Q6_K).enumerate() {
            let block: BQ6K = pod_read_unaligned(chunk);
            let d = block.d.to_f32();
            let inp = &input[bi * QK_K..(bi + 1) * QK_K];
            for group in 0..2 {
                let ql = &block.ql[group * 64..(group + 1) * 64];
                let qh = &block.qh[group * 32..(group + 1) * 32];
                let scales = &block.scales[group * 8..(group + 1) * 8];
                let gi = &inp[group * 128..(group + 1) * 128];
                for lane in 0..32 {
                    let si = lane / 16;
                    let q1 = ((ql[lane] & 0x0f) | ((qh[lane] & 0x03) << 4)) as i32 - 32;
                    let q2 = ((ql[lane + 32] & 0x0f) | (((qh[lane] >> 2) & 0x03) << 4)) as i32 - 32;
                    let q3 = ((ql[lane] >> 4) | (((qh[lane] >> 4) & 0x03) << 4)) as i32 - 32;
                    let q4 = ((ql[lane + 32] >> 4) | (((qh[lane] >> 6) & 0x03) << 4)) as i32 - 32;
                    sum += d * scales[si] as f32 * q1 as f32 * gi[lane];
                    sum += d * scales[si + 2] as f32 * q2 as f32 * gi[lane + 32];
                    sum += d * scales[si + 4] as f32 * q3 as f32 * gi[lane + 64];
                    sum += d * scales[si + 6] as f32 * q4 as f32 * gi[lane + 96];
                }
            }
        }
        sum
    }

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
