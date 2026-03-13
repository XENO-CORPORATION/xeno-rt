use half::f16;
use xrt_kernels::cpu::{dequantize_q4_0, dequantize_q8_0};
use xrt_kernels::{apply_rmsnorm, apply_rotary, matmul, silu, silu_inplace, softmax_inplace};

fn assert_close(lhs: f32, rhs: f32, tolerance: f32) {
    assert!(
        (lhs - rhs).abs() <= tolerance,
        "left={lhs}, right={rhs}, tolerance={tolerance}"
    );
}

fn assert_slice_close(lhs: &[f32], rhs: &[f32], tolerance: f32) {
    assert_eq!(lhs.len(), rhs.len());
    for (index, (lhs, rhs)) in lhs.iter().zip(rhs.iter()).enumerate() {
        assert!(
            (lhs - rhs).abs() <= tolerance,
            "index {index}: left={lhs}, right={rhs}, tolerance={tolerance}"
        );
    }
}

fn serialize_q8_0(scale: f32, quants: &[i8; 32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(34);
    bytes.extend_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
    bytes.extend(quants.iter().map(|value| *value as u8));
    bytes
}

fn serialize_q4_0(scale: f32, quants: &[i8; 32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(18);
    bytes.extend_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
    for index in 0..16 {
        let low = (quants[index] + 8) as u8;
        let high = (quants[index + 16] + 8) as u8;
        bytes.push(low | (high << 4));
    }
    bytes
}

fn requantize_q8_0(values: &[f32]) -> Vec<u8> {
    assert_eq!(values.len(), 32);
    let max_abs = values
        .iter()
        .fold(0.0f32, |acc, value| acc.max(value.abs()));
    let scale = if max_abs == 0.0 { 0.0 } else { max_abs / 127.0 };
    let mut quants = [0i8; 32];
    for (dst, value) in quants.iter_mut().zip(values.iter()) {
        *dst = if scale == 0.0 {
            0
        } else {
            (value / scale).round().clamp(-127.0, 127.0) as i8
        };
    }
    serialize_q8_0(scale, &quants)
}

fn requantize_q4_0(values: &[f32]) -> Vec<u8> {
    assert_eq!(values.len(), 32);
    let max_abs = values
        .iter()
        .fold(0.0f32, |acc, value| acc.max(value.abs()));
    let scale = if max_abs == 0.0 { 0.0 } else { max_abs / 8.0 };
    let mut quants = [0i8; 32];
    for (dst, value) in quants.iter_mut().zip(values.iter()) {
        *dst = if scale == 0.0 {
            0
        } else {
            (value / scale).round().clamp(-8.0, 7.0) as i8
        };
    }
    serialize_q4_0(scale, &quants)
}

#[test]
fn rmsnorm_matches_known_values() {
    let input = [1.0, 2.0, 3.0, 4.0];
    let weight = [1.0, 1.5, 2.0, 2.5];
    let mut output = [0.0; 4];

    apply_rmsnorm(&input, &weight, 1e-5, &mut output);

    let inv_rms = 1.0
        / ((1.0f32.powi(2) + 2.0f32.powi(2) + 3.0f32.powi(2) + 4.0f32.powi(2)) / 4.0 + 1e-5).sqrt();
    let expected = [
        1.0 * inv_rms * 1.0,
        2.0 * inv_rms * 1.5,
        3.0 * inv_rms * 2.0,
        4.0 * inv_rms * 2.5,
    ];
    assert_slice_close(&output, &expected, 1e-6);
}

#[test]
fn rope_rotation_matches_manual_computation() {
    let mut tensor = [1.0, 2.0, 3.0, 4.0];

    apply_rotary(&mut tensor, 1, 4, 2, 4, 10_000.0, 1.0);

    let angle0 = 2.0f32;
    let angle1 = 2.0f32 / 100.0;
    let expected = [
        1.0 * angle0.cos() - 3.0 * angle0.sin(),
        2.0 * angle1.cos() - 4.0 * angle1.sin(),
        1.0 * angle0.sin() + 3.0 * angle0.cos(),
        2.0 * angle1.sin() + 4.0 * angle1.cos(),
    ];
    assert_slice_close(&tensor, &expected, 1e-6);
}

#[test]
fn softmax_outputs_normalized_probabilities() {
    let mut values = [1.0, 2.0, 3.0];

    softmax_inplace(&mut values);

    let denom = (-2.0f32).exp() + (-1.0f32).exp() + 1.0;
    let expected = [
        (-2.0f32).exp() / denom,
        (-1.0f32).exp() / denom,
        1.0 / denom,
    ];
    assert_slice_close(&values, &expected, 1e-6);
    assert_close(values.iter().sum::<f32>(), 1.0, 1e-6);
}

#[test]
fn silu_matches_manual_formula() {
    let inputs = [-2.0f32, -0.5, 0.0, 1.5];
    let expected = inputs.map(|value| value * (1.0 / (1.0 + (-value).exp())));
    let scalar = inputs.map(silu);
    assert_slice_close(&scalar, &expected, 1e-6);

    let mut inplace = inputs;
    silu_inplace(&mut inplace);
    assert_slice_close(&inplace, &expected, 1e-6);
}

#[test]
fn matmul_matches_naive_reference() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut output = [0.0; 4];

    matmul(&a, 2, 3, &b, 2, &mut output);

    let expected = [
        1.0 * 7.0 + 2.0 * 9.0 + 3.0 * 11.0,
        1.0 * 8.0 + 2.0 * 10.0 + 3.0 * 12.0,
        4.0 * 7.0 + 5.0 * 9.0 + 6.0 * 11.0,
        4.0 * 8.0 + 5.0 * 10.0 + 6.0 * 12.0,
    ];
    assert_slice_close(&output, &expected, 1e-6);
}

#[test]
fn q8_0_roundtrip_preserves_quantized_values() {
    let scale = 0.25f32;
    let quants = [
        -127, -96, -64, -32, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32, 48, 63, 64, 80, 96, 112,
        120, 124, 125, 126, 127, -48, -24, 24, 12, 6,
    ];
    let original = serialize_q8_0(scale, &quants);

    let dequantized = dequantize_q8_0(&original, 32).expect("q8_0 dequantization should work");
    let requantized = requantize_q8_0(&dequantized);

    assert_eq!(requantized, original);
}

#[test]
fn q4_0_roundtrip_preserves_quantized_values() {
    let scale = 0.5f32;
    let quants = [
        -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8, 7, -7, 6, -6, 5, -5, 4, -4, 3,
        -3, 2, -2, 1, -1, 0,
    ];
    let original = serialize_q4_0(scale, &quants);

    let dequantized = dequantize_q4_0(&original, 32).expect("q4_0 dequantization should work");
    let requantized = requantize_q4_0(&dequantized);

    assert_eq!(requantized, original);
}
