pub mod matmul;
pub mod quantize;
pub mod rmsnorm;
pub mod rope;
pub mod silu;
pub mod softmax;

pub use matmul::{matmul, matvec, matvec_quantized, quantized_row_dot};
pub use quantize::{
    dequantize_q4_0, dequantize_q4_0_row, dequantize_q4_k, dequantize_q4_k_row, dequantize_q5_k,
    dequantize_q5_k_row, dequantize_q6_k, dequantize_q6_k_row, dequantize_q8_0,
    dequantize_q8_0_row, dot_q4_0, dot_q4_k, dot_q5_k, dot_q6_k, dot_q8_0,
};
pub use rmsnorm::apply_rmsnorm;
pub use rope::{apply_rotary, apply_rotary_qk, RopeFreqs};
pub use silu::{silu, silu_inplace, swiglu};
pub use softmax::softmax_inplace;

pub fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
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
