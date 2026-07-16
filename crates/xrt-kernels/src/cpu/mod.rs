pub mod matmul;
pub mod quantize;
pub mod rmsnorm;
pub mod rope;
pub mod silu;
pub mod simd;
pub mod softmax;
pub mod thread_pool;

pub use matmul::{
    accumulate_scaled, matmul, matvec, matvec_quantized, matvec_quantized_batch,
    matvec_quantized_fused, matvec_quantized_fused_mixed, quantized_row_dot,
};
pub use quantize::{
    dequantize_q4_0, dequantize_q4_0_row, dequantize_q4_k, dequantize_q4_k_row, dequantize_q5_k,
    dequantize_q5_k_row, dequantize_q6_k, dequantize_q6_k_row, dequantize_q8_0,
    dequantize_q8_0_row, dot_q4_0, dot_q4_k, dot_q5_k, dot_q6_k, dot_q8_0,
};
pub use rmsnorm::apply_rmsnorm;
pub use rope::{apply_rotary, apply_rotary_qk, RopeFreqs};
pub use silu::{geglu_pytorch_tanh, gelu_pytorch_tanh, silu, silu_inplace, swiglu};
pub use softmax::softmax_inplace;
pub use thread_pool::{global_pool, SpinPool};

/// SiLU activation in-place, SIMD-accelerated.
pub fn silu_inplace_fast(data: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if simd::has_avx2_fma() {
            return unsafe { simd::silu_inplace_avx2(data) };
        }
    }
    silu_inplace(data);
}

/// L2-normalize a slice in-place with epsilon, SIMD-accelerated.
pub fn l2_normalize(data: &mut [f32], eps: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if simd::has_avx2_fma() {
            return unsafe { simd::l2_normalize_avx2(data, eps) };
        }
    }
    let mut norm_sq = 0.0f32;
    for &val in data.iter() {
        norm_sq += val * val;
    }
    let inv = 1.0 / (norm_sq + eps).sqrt();
    for val in data.iter_mut() {
        *val *= inv;
    }
}

/// Gated RMSNorm: out = RMSNorm(out) * silu(gate) * norm_w, SIMD-accelerated.
///
/// # Safety
/// gate and norm_w must be valid for out.len() elements.
pub unsafe fn gated_rmsnorm(out: &mut [f32], gate: *const f32, norm_w: *const f32, eps: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if simd::has_avx2_fma() {
            return simd::gated_rmsnorm_avx2(out, gate, norm_w, eps);
        }
    }
    let n = out.len();
    let mut sum_sq = 0.0f32;
    for &val in out.iter() {
        sum_sq += val * val;
    }
    let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();
    for i in 0..n {
        let gate_val = *gate.add(i);
        let silu = gate_val / (1.0 + (-gate_val).exp());
        out[i] = out[i] * inv_rms * *norm_w.add(i) * silu;
    }
}

/// Full delta rule for one group (fused decay + update + output).
/// state: v_dim × k_dim row-major. Computes in-place state update and output vector.
///
/// # Safety
/// Pointers must be valid for the given dimensions.
pub unsafe fn delta_rule_group(
    state: &mut [f32],
    k: *const f32,
    q: *const f32,
    v: *const f32,
    out: *mut f32,
    v_dim: usize,
    k_dim: usize,
    decay: f32,
    beta: f32,
    q_scale: f32,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if simd::has_avx2_fma() {
            return simd::delta_rule_group_avx2(
                state, k, q, v, out, v_dim, k_dim, decay, beta, q_scale,
            );
        }
    }
    // Scalar fallback
    for vi in 0..v_dim {
        let row = &mut state[vi * k_dim..(vi + 1) * k_dim];
        let mut sk = 0.0f32;
        for ki in 0..k_dim {
            sk += row[ki] * *k.add(ki);
        }
        sk *= decay;
        let d = beta * (*v.add(vi) - sk);
        let mut out_sum = 0.0f32;
        for ki in 0..k_dim {
            row[ki] = decay * row[ki] + d * *k.add(ki);
            out_sum += row[ki] * *q.add(ki);
        }
        *out.add(vi) = out_sum * q_scale;
    }
}

pub fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if simd::has_avx2_fma() {
            return unsafe { simd::dot_f32_avx2(lhs, rhs) };
        }
    }
    lhs.iter().zip(rhs.iter()).map(|(lhs, rhs)| lhs * rhs).sum()
}

pub fn add_inplace(dst: &mut [f32], src: &[f32]) {
    for (dst, src) in dst.iter_mut().zip(src.iter()) {
        *dst += src;
    }
}

pub fn q8_0_row_dot(row: &[u8], input: &[f32]) -> xrt_core::Result<f32> {
    quantized_row_dot(xrt_core::DType::Q8_0, row, input)
}

pub fn q4_0_row_dot(row: &[u8], input: &[f32]) -> xrt_core::Result<f32> {
    quantized_row_dot(xrt_core::DType::Q4_0, row, input)
}

pub fn q4_k_row_dot(row: &[u8], input: &[f32]) -> xrt_core::Result<f32> {
    quantized_row_dot(xrt_core::DType::Q4_K, row, input)
}

pub fn q5_k_row_dot(row: &[u8], input: &[f32]) -> xrt_core::Result<f32> {
    quantized_row_dot(xrt_core::DType::Q5_K, row, input)
}

pub fn q6_k_row_dot(row: &[u8], input: &[f32]) -> xrt_core::Result<f32> {
    quantized_row_dot(xrt_core::DType::Q6_K, row, input)
}
