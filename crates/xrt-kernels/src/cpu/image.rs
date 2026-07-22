use half::{bf16, f16};
use rayon::prelude::*;
use xrt_core::{checked_mul, Result, XrtError};

use super::{dot, simd};

// Keep one output-channel tile in L1 while reusing contiguous source rows.
// Eight 1024-wide F32 rows occupy 32 KiB, matching the common high-resolution
// Qwen Image VAE source geometry without allocating an im2col buffer.
const CONV_OUTPUT_ROW_TILE: usize = 8;

#[inline]
fn accumulate_conv_row(
    input: &[f32],
    output: &mut [f32],
    kernel_x: usize,
    stride_x: usize,
    padding_left: usize,
    weight: f32,
) {
    let output_start = padding_left.saturating_sub(kernel_x).div_ceil(stride_x);
    let output_end = padding_left
        .saturating_add(input.len())
        .saturating_sub(kernel_x)
        .div_ceil(stride_x)
        .min(output.len());
    if output_start >= output_end {
        return;
    }

    if stride_x == 1 {
        let input_start = output_start + kernel_x - padding_left;
        let input = &input[input_start..input_start + output_end - output_start];
        let output = &mut output[output_start..output_end];
        for (output, input) in output.iter_mut().zip(input) {
            *output += *input * weight;
        }
        return;
    }

    for (offset, output) in output[output_start..output_end].iter_mut().enumerate() {
        let output_x = output_start + offset;
        let input_x = output_x * stride_x + kernel_x - padding_left;
        *output += input[input_x] * weight;
    }
}

fn exact_len(dimensions: &[usize], label: &str) -> Result<usize> {
    dimensions.iter().try_fold(1usize, |length, dimension| {
        checked_mul(length, *dimension, label)
    })
}

/// Dense affine transform for row-major input `[rows, input_features]` and
/// PyTorch/SafeTensors weight `[output_features, input_features]`.
pub fn linear_f32(
    input: &[f32],
    rows: usize,
    input_features: usize,
    weight: &[f32],
    output_features: usize,
    bias: Option<&[f32]>,
    output: &mut [f32],
) -> Result<()> {
    if input.len() != exact_len(&[rows, input_features], "linear input")?
        || weight.len() != exact_len(&[output_features, input_features], "linear weight")?
        || output.len() != exact_len(&[rows, output_features], "linear output")?
        || bias.is_some_and(|values| values.len() != output_features)
    {
        return Err(XrtError::Shape(format!(
            "linear shape mismatch: input={}, weight={}, bias={:?}, output={}, rows={rows}, in={input_features}, out={output_features}",
            input.len(),
            weight.len(),
            bias.map(<[f32]>::len),
            output.len()
        )));
    }
    if input_features == 0 || output_features == 0 {
        return Err(XrtError::Shape(
            "linear feature dimensions must be positive".to_string(),
        ));
    }

    output
        .par_chunks_mut(output_features)
        .enumerate()
        .for_each(|(row_index, output_row)| {
            let input_row = &input[row_index * input_features..(row_index + 1) * input_features];
            for (feature, value) in output_row.iter_mut().enumerate() {
                let weight_row = &weight[feature * input_features..(feature + 1) * input_features];
                *value = dot(input_row, weight_row) + bias.map_or(0.0, |values| values[feature]);
            }
        });
    Ok(())
}

/// BF16-weight reference affine transform. Weight bytes retain the
/// SafeTensors/PyTorch `[output_features, input_features]` order and are
/// decoded row-by-row, avoiding an F32 copy of multi-gigabyte components.
pub fn linear_bf16(
    input: &[f32],
    rows: usize,
    input_features: usize,
    weight_bytes: &[u8],
    output_features: usize,
    bias: Option<&[f32]>,
    output: &mut [f32],
) -> Result<()> {
    let weight_elements = exact_len(&[output_features, input_features], "BF16 linear weight")?;
    let expected_weight_bytes = checked_mul(weight_elements, 2, "BF16 linear weight bytes")?;
    if input.len() != exact_len(&[rows, input_features], "BF16 linear input")?
        || weight_bytes.len() != expected_weight_bytes
        || output.len() != exact_len(&[rows, output_features], "BF16 linear output")?
        || bias.is_some_and(|values| values.len() != output_features)
    {
        return Err(XrtError::Shape(format!(
            "BF16 linear shape mismatch: input={}, weight_bytes={}, bias={:?}, output={}, rows={rows}, in={input_features}, out={output_features}",
            input.len(),
            weight_bytes.len(),
            bias.map(<[f32]>::len),
            output.len()
        )));
    }
    if input_features == 0 || output_features == 0 {
        return Err(XrtError::Shape(
            "BF16 linear feature dimensions must be positive".to_string(),
        ));
    }

    #[cfg(target_arch = "x86_64")]
    if simd::has_avx2_fma() {
        if rows > 1 {
            // Text encoders apply the same multi-megabyte BF16 matrix to many
            // token rows. Keep one mapped weight row hot while evaluating all
            // tokens, then transpose the comparatively small activation. Each
            // scalar dot still visits K in the established order, so this is
            // bit-identical to the row-major scheduling below.
            let mut feature_major = vec![0.0f32; output.len()];
            feature_major
                .par_chunks_mut(rows)
                .enumerate()
                .for_each(|(feature, output_column)| {
                    let weight_start = feature * input_features * 2;
                    let weight_row = &weight_bytes[weight_start..weight_start + input_features * 2];
                    let bias = bias.map_or(0.0, |values| values[feature]);
                    for (row_index, value) in output_column.iter_mut().enumerate() {
                        let input_row =
                            &input[row_index * input_features..(row_index + 1) * input_features];
                        *value =
                            unsafe { simd::dot_bf16_f32_ordered_avx2(input_row, weight_row, bias) };
                    }
                });
            output.par_chunks_mut(output_features).enumerate().for_each(
                |(row_index, output_row)| {
                    for (feature, value) in output_row.iter_mut().enumerate() {
                        *value = feature_major[feature * rows + row_index];
                    }
                },
            );
            return Ok(());
        }

        output
            .par_chunks_mut(output_features)
            .enumerate()
            .for_each(|(row_index, output_row)| {
                let input_row =
                    &input[row_index * input_features..(row_index + 1) * input_features];
                for (feature, value) in output_row.iter_mut().enumerate() {
                    let weight_start = feature * input_features * 2;
                    let weight_row = &weight_bytes[weight_start..weight_start + input_features * 2];
                    *value = unsafe {
                        simd::dot_bf16_f32_ordered_avx2(
                            input_row,
                            weight_row,
                            bias.map_or(0.0, |values| values[feature]),
                        )
                    };
                }
            });
        return Ok(());
    }

    output
        .par_chunks_mut(output_features)
        .enumerate()
        .for_each(|(row_index, output_row)| {
            let input_row = &input[row_index * input_features..(row_index + 1) * input_features];
            for (feature, value) in output_row.iter_mut().enumerate() {
                let weight_start = feature * input_features * 2;
                let weight_row = &weight_bytes[weight_start..weight_start + input_features * 2];
                let mut sum = bias.map_or(0.0, |values| values[feature]);
                for (input_value, encoded) in input_row.iter().zip(weight_row.chunks_exact(2)) {
                    let weight =
                        bf16::from_bits(u16::from_le_bytes([encoded[0], encoded[1]])).to_f32();
                    sum += input_value * weight;
                }
                *value = sum;
            }
        });
    Ok(())
}

/// F16-weight affine transform with the same mmap-friendly storage contract as
/// [`linear_bf16`].
pub fn linear_f16(
    input: &[f32],
    rows: usize,
    input_features: usize,
    weight_bytes: &[u8],
    output_features: usize,
    bias: Option<&[f32]>,
    output: &mut [f32],
) -> Result<()> {
    linear_16bit(
        input,
        rows,
        input_features,
        weight_bytes,
        output_features,
        bias,
        output,
        "F16",
        |bits| f16::from_bits(bits).to_f32(),
    )
}

/// F32-weight affine transform over encoded little-endian bytes. GGUF callers
/// can retain mmap-backed storage without relying on pointer alignment or
/// materializing an additional matrix copy.
pub fn linear_f32_bytes(
    input: &[f32],
    rows: usize,
    input_features: usize,
    weight_bytes: &[u8],
    output_features: usize,
    bias: Option<&[f32]>,
    output: &mut [f32],
) -> Result<()> {
    let weight_elements = exact_len(&[output_features, input_features], "F32 linear weight")?;
    let expected_weight_bytes = checked_mul(weight_elements, 4, "F32 linear weight bytes")?;
    validate_encoded_linear(
        input,
        rows,
        input_features,
        weight_bytes,
        expected_weight_bytes,
        output_features,
        bias,
        output,
        "F32",
    )?;

    output
        .par_chunks_mut(output_features)
        .enumerate()
        .for_each(|(row_index, output_row)| {
            let input_row = &input[row_index * input_features..(row_index + 1) * input_features];
            for (feature, value) in output_row.iter_mut().enumerate() {
                let weight_start = feature * input_features * 4;
                let weight_row = &weight_bytes[weight_start..weight_start + input_features * 4];
                let mut sum = bias.map_or(0.0, |values| values[feature]);
                for (input_value, encoded) in input_row.iter().zip(weight_row.chunks_exact(4)) {
                    let weight =
                        f32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]);
                    sum += input_value * weight;
                }
                *value = sum;
            }
        });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn linear_16bit<F>(
    input: &[f32],
    rows: usize,
    input_features: usize,
    weight_bytes: &[u8],
    output_features: usize,
    bias: Option<&[f32]>,
    output: &mut [f32],
    label: &str,
    decode: F,
) -> Result<()>
where
    F: Fn(u16) -> f32 + Sync,
{
    let weight_elements = exact_len(
        &[output_features, input_features],
        &format!("{label} linear weight"),
    )?;
    let expected_weight_bytes =
        checked_mul(weight_elements, 2, &format!("{label} linear weight bytes"))?;
    validate_encoded_linear(
        input,
        rows,
        input_features,
        weight_bytes,
        expected_weight_bytes,
        output_features,
        bias,
        output,
        label,
    )?;

    output
        .par_chunks_mut(output_features)
        .enumerate()
        .for_each(|(row_index, output_row)| {
            let input_row = &input[row_index * input_features..(row_index + 1) * input_features];
            for (feature, value) in output_row.iter_mut().enumerate() {
                let weight_start = feature * input_features * 2;
                let weight_row = &weight_bytes[weight_start..weight_start + input_features * 2];
                let mut sum = bias.map_or(0.0, |values| values[feature]);
                for (input_value, encoded) in input_row.iter().zip(weight_row.chunks_exact(2)) {
                    let weight = decode(u16::from_le_bytes([encoded[0], encoded[1]]));
                    sum += input_value * weight;
                }
                *value = sum;
            }
        });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_encoded_linear(
    input: &[f32],
    rows: usize,
    input_features: usize,
    weight_bytes: &[u8],
    expected_weight_bytes: usize,
    output_features: usize,
    bias: Option<&[f32]>,
    output: &[f32],
    label: &str,
) -> Result<()> {
    if input.len() != exact_len(&[rows, input_features], "encoded linear input")?
        || weight_bytes.len() != expected_weight_bytes
        || output.len() != exact_len(&[rows, output_features], "encoded linear output")?
        || bias.is_some_and(|values| values.len() != output_features)
    {
        return Err(XrtError::Shape(format!(
            "{label} linear shape mismatch: input={}, weight_bytes={}, bias={:?}, output={}, rows={rows}, in={input_features}, out={output_features}",
            input.len(),
            weight_bytes.len(),
            bias.map(<[f32]>::len),
            output.len()
        )));
    }
    if input_features == 0 || output_features == 0 {
        return Err(XrtError::Shape(
            "linear feature dimensions must be positive".to_string(),
        ));
    }
    Ok(())
}

/// PyTorch-compatible LayerNorm over the last dimension using population
/// variance (`unbiased=false`).
pub fn layer_norm_rows(values: &mut [f32], rows: usize, width: usize, eps: f32) -> Result<()> {
    validate_normalization(values, rows, width, eps, "layer norm")?;
    values.par_chunks_mut(width).for_each(|row| {
        let mean = row.iter().copied().sum::<f32>() / width as f32;
        let variance = row
            .iter()
            .map(|value| {
                let centered = *value - mean;
                centered * centered
            })
            .sum::<f32>()
            / width as f32;
        let inverse = (variance + eps).sqrt().recip();
        for value in row {
            *value = (*value - mean) * inverse;
        }
    });
    Ok(())
}

/// RMSNorm over the last dimension with an optional per-feature scale.
pub fn rms_norm_rows(
    values: &mut [f32],
    rows: usize,
    width: usize,
    weight: Option<&[f32]>,
    eps: f32,
) -> Result<()> {
    validate_normalization(values, rows, width, eps, "RMS norm")?;
    if weight.is_some_and(|weight| weight.len() != width) {
        return Err(XrtError::Shape(format!(
            "RMS norm weight length {:?} does not match width {width}",
            weight.map(<[f32]>::len)
        )));
    }
    values.par_chunks_mut(width).for_each(|row| {
        let mean_square = row.iter().map(|value| value * value).sum::<f32>() / width as f32;
        let inverse = (mean_square + eps).sqrt().recip();
        for (index, value) in row.iter_mut().enumerate() {
            *value *= inverse * weight.map_or(1.0, |weight| weight[index]);
        }
    });
    Ok(())
}

fn validate_normalization(
    values: &[f32],
    rows: usize,
    width: usize,
    eps: f32,
    label: &str,
) -> Result<()> {
    if width == 0 || !eps.is_finite() || eps <= 0.0 {
        return Err(XrtError::Shape(format!(
            "{label} requires positive width and finite positive epsilon"
        )));
    }
    let expected = exact_len(&[rows, width], label)?;
    if values.len() != expected {
        return Err(XrtError::Shape(format!(
            "{label} input length {} does not match rows({rows}) * width({width}) = {expected}",
            values.len()
        )));
    }
    Ok(())
}

/// Apply complex rotary frequencies to adjacent feature pairs in a tensor
/// laid out as `[batch, sequence, heads, head_dim]`. `cos` and `sin` use
/// `[sequence, head_dim/2]` complex-frequency layout.
pub fn apply_complex_rope(
    values: &mut [f32],
    batch: usize,
    sequence: usize,
    heads: usize,
    head_dim: usize,
    cos: &[f32],
    sin: &[f32],
) -> Result<()> {
    if batch == 0 || sequence == 0 || heads == 0 || head_dim == 0 || head_dim % 2 != 0 {
        return Err(XrtError::Shape(format!(
            "complex RoPE head dimension {head_dim} must be positive and even"
        )));
    }
    let value_len = exact_len(&[batch, sequence, heads, head_dim], "complex RoPE values")?;
    let frequency_len = exact_len(&[sequence, head_dim / 2], "complex RoPE frequencies")?;
    if values.len() != value_len || cos.len() != frequency_len || sin.len() != frequency_len {
        return Err(XrtError::Shape(format!(
            "complex RoPE shape mismatch: values={}, cos={}, sin={}, expected values={value_len}, frequencies={frequency_len}",
            values.len(), cos.len(), sin.len()
        )));
    }

    values
        .par_chunks_mut(head_dim)
        .enumerate()
        .for_each(|(vector_index, vector)| {
            let sequence_index = (vector_index / heads) % sequence;
            let frequency = sequence_index * (head_dim / 2);
            for pair in 0..head_dim / 2 {
                let real_index = pair * 2;
                let real = vector[real_index];
                let imaginary = vector[real_index + 1];
                let cosine = cos[frequency + pair];
                let sine = sin[frequency + pair];
                vector[real_index] = real * cosine - imaginary * sine;
                vector[real_index + 1] = real * sine + imaginary * cosine;
            }
        });
    Ok(())
}

/// Scalar reference scaled-dot-product attention. Query uses
/// `[batch, query_sequence, heads, head_dim]`; key/value use the same layout
/// with `key_sequence`. A false key-mask entry is excluded for every query.
#[allow(clippy::too_many_arguments)]
pub fn scaled_dot_product_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    batch: usize,
    query_sequence: usize,
    key_sequence: usize,
    heads: usize,
    head_dim: usize,
    key_mask: Option<&[u8]>,
    output: &mut [f32],
) -> Result<()> {
    if batch == 0 || query_sequence == 0 || heads == 0 || head_dim == 0 || key_sequence == 0 {
        return Err(XrtError::Shape(
            "attention heads, head_dim, and key_sequence must be positive".to_string(),
        ));
    }
    let query_len = exact_len(&[batch, query_sequence, heads, head_dim], "attention query")?;
    let key_len = exact_len(
        &[batch, key_sequence, heads, head_dim],
        "attention key/value",
    )?;
    let mask_len = exact_len(&[batch, key_sequence], "attention mask")?;
    if query.len() != query_len
        || key.len() != key_len
        || value.len() != key_len
        || output.len() != query_len
        || key_mask.is_some_and(|mask| mask.len() != mask_len)
    {
        return Err(XrtError::Shape(format!(
            "attention shape mismatch: q={}, k={}, v={}, mask={:?}, out={}, expected q={query_len}, kv={key_len}, mask={mask_len}",
            query.len(),
            key.len(),
            value.len(),
            key_mask.map(<[u8]>::len),
            output.len()
        )));
    }
    if let Some(mask) = key_mask {
        for batch_index in 0..batch {
            if !mask[batch_index * key_sequence..(batch_index + 1) * key_sequence]
                .iter()
                .any(|value| *value != 0)
            {
                return Err(XrtError::Shape(format!(
                    "attention mask batch row {batch_index} has no active keys"
                )));
            }
        }
    }

    let scale = (head_dim as f32).sqrt().recip();
    let query_stride = heads * head_dim;
    let batch_query_stride = query_sequence * query_stride;
    let batch_key_stride = key_sequence * query_stride;
    output
        .par_chunks_mut(query_stride)
        .enumerate()
        .for_each_init(
            || vec![0.0f32; key_sequence],
            |scores, (query_row_index, query_output)| {
                let batch_index = query_row_index / query_sequence;
                let query_index = query_row_index % query_sequence;
                for head in 0..heads {
                    let query_start = batch_index * batch_query_stride
                        + query_index * query_stride
                        + head * head_dim;
                    let query_row = &query[query_start..query_start + head_dim];
                    for (key_index, score) in scores.iter_mut().enumerate() {
                        let active = key_mask.map_or(true, |mask| {
                            mask[batch_index * key_sequence + key_index] != 0
                        });
                        if !active {
                            *score = f32::NEG_INFINITY;
                            continue;
                        }
                        let key_start = batch_index * batch_key_stride
                            + key_index * query_stride
                            + head * head_dim;
                        *score = dot(query_row, &key[key_start..key_start + head_dim]) * scale;
                    }
                    reference_softmax(scores);
                    let output_row = &mut query_output[head * head_dim..(head + 1) * head_dim];
                    output_row.fill(0.0);
                    for (key_index, probability) in scores.iter().copied().enumerate() {
                        if probability == 0.0 {
                            continue;
                        }
                        let value_start = batch_index * batch_key_stride
                            + key_index * query_stride
                            + head * head_dim;
                        for (destination, source) in output_row
                            .iter_mut()
                            .zip(&value[value_start..value_start + head_dim])
                        {
                            *destination += probability * source;
                        }
                    }
                }
            },
        );
    Ok(())
}

/// Causal grouped-query attention for dense text encoders. Query is
/// `[batch, sequence, query_heads, head_dim]`; key/value use
/// `[batch, sequence, key_value_heads, head_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn grouped_causal_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    batch: usize,
    sequence: usize,
    query_heads: usize,
    key_value_heads: usize,
    head_dim: usize,
    output: &mut [f32],
) -> Result<()> {
    if batch == 0
        || sequence == 0
        || query_heads == 0
        || key_value_heads == 0
        || head_dim == 0
        || query_heads % key_value_heads != 0
    {
        return Err(XrtError::Shape(format!(
            "invalid grouped causal attention geometry: batch={batch}, sequence={sequence}, q_heads={query_heads}, kv_heads={key_value_heads}, head_dim={head_dim}"
        )));
    }
    let query_len = exact_len(
        &[batch, sequence, query_heads, head_dim],
        "grouped attention query",
    )?;
    let key_len = exact_len(
        &[batch, sequence, key_value_heads, head_dim],
        "grouped attention key/value",
    )?;
    if query.len() != query_len
        || key.len() != key_len
        || value.len() != key_len
        || output.len() != query_len
    {
        return Err(XrtError::Shape(format!(
            "grouped causal attention shape mismatch: q={}, k={}, v={}, out={}, expected q={query_len}, kv={key_len}",
            query.len(), key.len(), value.len(), output.len()
        )));
    }
    let query_group = query_heads / key_value_heads;
    let scale = (head_dim as f32).sqrt().recip();
    output
        .par_chunks_mut(head_dim)
        .enumerate()
        .for_each(|(vector_index, output_row)| {
            let query_head = vector_index % query_heads;
            let token = (vector_index / query_heads) % sequence;
            let batch_index = vector_index / (sequence * query_heads);
            let key_head = query_head / query_group;
            let query_start = vector_index * head_dim;
            let query_row = &query[query_start..query_start + head_dim];
            let mut scores = vec![0.0f32; token + 1];
            for (key_token, score) in scores.iter_mut().enumerate() {
                let key_start =
                    ((batch_index * sequence + key_token) * key_value_heads + key_head) * head_dim;
                *score = dot(query_row, &key[key_start..key_start + head_dim]) * scale;
            }
            reference_softmax(&mut scores);
            output_row.fill(0.0);
            for (key_token, probability) in scores.into_iter().enumerate() {
                let value_start =
                    ((batch_index * sequence + key_token) * key_value_heads + key_head) * head_dim;
                for (destination, source) in output_row
                    .iter_mut()
                    .zip(&value[value_start..value_start + head_dim])
                {
                    *destination += probability * source;
                }
            }
        });
    Ok(())
}

fn reference_softmax(values: &mut [f32]) {
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for value in values.iter_mut() {
        *value = (*value - maximum).exp();
        sum += *value;
    }
    let inverse = sum.recip();
    for value in values {
        *value *= inverse;
    }
}

/// Qwen Image VAE RMS normalization across the channel axis of an NCTHW
/// tensor. This matches `F.normalize(..., dim=1) * sqrt(channels) * gamma`.
#[allow(clippy::too_many_arguments)]
pub fn vae_rms_norm_channels_ncthw(
    values: &mut [f32],
    batch: usize,
    channels: usize,
    depth: usize,
    height: usize,
    width: usize,
    gamma: &[f32],
    epsilon: f32,
) -> Result<()> {
    if batch == 0
        || channels == 0
        || depth == 0
        || height == 0
        || width == 0
        || !epsilon.is_finite()
        || epsilon <= 0.0
    {
        return Err(XrtError::Shape(
            "VAE RMS normalization requires positive geometry and epsilon".to_string(),
        ));
    }
    let spatial = exact_len(&[depth, height, width], "VAE RMS spatial extent")?;
    let expected = exact_len(&[batch, channels, spatial], "VAE RMS input")?;
    if values.len() != expected || gamma.len() != channels {
        return Err(XrtError::Shape(format!(
            "VAE RMS shape mismatch: values={}, gamma={}, expected values={expected}, channels={channels}",
            values.len(),
            gamma.len()
        )));
    }
    let scale = (channels as f32).sqrt();
    for batch_index in 0..batch {
        for position in 0..spatial {
            let mut squared_norm = 0.0f32;
            for channel in 0..channels {
                let value = values[(batch_index * channels + channel) * spatial + position];
                squared_norm += value * value;
            }
            let inverse = squared_norm.sqrt().max(epsilon).recip() * scale;
            for (channel, channel_scale) in gamma.iter().copied().enumerate() {
                let index = (batch_index * channels + channel) * spatial + position;
                values[index] *= inverse * channel_scale;
            }
        }
    }
    Ok(())
}

/// Spatial Conv2D applied independently to every temporal frame of an NCTHW
/// tensor. Weights use PyTorch `[out_channels, in_channels, kh, kw]` order;
/// padding is `[top, bottom, left, right]`.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_ncthw(
    input: &[f32],
    batch: usize,
    input_channels: usize,
    depth: usize,
    input_height: usize,
    input_width: usize,
    weight: &[f32],
    output_channels: usize,
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 4],
    bias: Option<&[f32]>,
    output: &mut [f32],
) -> Result<[usize; 2]> {
    if batch == 0
        || input_channels == 0
        || output_channels == 0
        || depth == 0
        || input_height == 0
        || input_width == 0
        || kernel.contains(&0)
        || stride.contains(&0)
    {
        return Err(XrtError::Shape(
            "Conv2D channels, geometry, kernel, and stride must be positive".to_string(),
        ));
    }
    let input_len = exact_len(
        &[batch, input_channels, depth, input_height, input_width],
        "Conv2D NCTHW input",
    )?;
    let weight_len = exact_len(
        &[output_channels, input_channels, kernel[0], kernel[1]],
        "Conv2D weight",
    )?;
    if input.len() != input_len
        || weight.len() != weight_len
        || bias.is_some_and(|values| values.len() != output_channels)
    {
        return Err(XrtError::Shape(format!(
            "Conv2D NCTHW input/weight/bias mismatch: input={}, weight={}, bias={:?}",
            input.len(),
            weight.len(),
            bias.map(<[f32]>::len)
        )));
    }
    let padded_height = input_height
        .checked_add(padding[0])
        .and_then(|value| value.checked_add(padding[1]))
        .ok_or_else(|| XrtError::Shape("Conv2D padded height overflow".to_string()))?;
    let padded_width = input_width
        .checked_add(padding[2])
        .and_then(|value| value.checked_add(padding[3]))
        .ok_or_else(|| XrtError::Shape("Conv2D padded width overflow".to_string()))?;
    if padded_height < kernel[0] || padded_width < kernel[1] {
        return Err(XrtError::Shape(
            "Conv2D padded input is smaller than its kernel".to_string(),
        ));
    }
    let output_shape = [
        (padded_height - kernel[0]) / stride[0] + 1,
        (padded_width - kernel[1]) / stride[1] + 1,
    ];
    let output_len = exact_len(
        &[
            batch,
            output_channels,
            depth,
            output_shape[0],
            output_shape[1],
        ],
        "Conv2D NCTHW output",
    )?;
    if output.len() != output_len {
        return Err(XrtError::Shape(format!(
            "Conv2D NCTHW output length {} does not match expected {output_len}",
            output.len()
        )));
    }

    let output_plane = depth * output_shape[0] * output_shape[1];
    let output_frame = output_shape[0] * output_shape[1];
    let output_row_tile = CONV_OUTPUT_ROW_TILE;
    let output_tile = output_row_tile * output_shape[1];
    let input_plane = depth * input_height * input_width;
    let kernel_plane = kernel[0] * kernel[1];
    let parallelize_tiles = batch.saturating_mul(output_channels) < rayon::current_num_threads()
        && depth.saturating_mul(output_shape[0].div_ceil(output_row_tile)) > 1;
    output
        .par_chunks_mut(output_plane)
        .enumerate()
        .for_each(|(batch_channel, output_channel)| {
            let batch_index = batch_channel / output_channels;
            let output_channel_index = batch_channel % output_channels;
            let bias_value = bias.map_or(0.0, |bias| bias[output_channel_index]);
            let process_tile = |temporal: usize, tile_y: usize, output_tile: &mut [f32]| {
                output_tile.fill(bias_value);
                let tile_rows = output_tile.len() / output_shape[1];
                let tile_y_end = tile_y + tile_rows;
                for input_channel in 0..input_channels {
                    let input_channel_offset = batch_index * input_channels * input_plane
                        + input_channel * input_plane
                        + temporal * input_height * input_width;
                    let weight_channel_offset =
                        output_channel_index * input_channels * kernel_plane
                            + input_channel * kernel_plane;
                    for kernel_y in 0..kernel[0] {
                        for kernel_x in 0..kernel[1] {
                            let weight_value =
                                weight[weight_channel_offset + kernel_y * kernel[1] + kernel_x];
                            for output_y in tile_y..tile_y_end {
                                let padded_y = output_y * stride[0] + kernel_y;
                                if padded_y < padding[0] {
                                    continue;
                                }
                                let input_y = padded_y - padding[0];
                                if input_y >= input_height {
                                    continue;
                                }
                                let input_row = input_channel_offset + input_y * input_width;
                                let local_output_y = output_y - tile_y;
                                let output_row = local_output_y * output_shape[1];
                                accumulate_conv_row(
                                    &input[input_row..input_row + input_width],
                                    &mut output_tile[output_row..output_row + output_shape[1]],
                                    kernel_x,
                                    stride[1],
                                    padding[2],
                                    weight_value,
                                );
                            }
                        }
                    }
                }
            };

            if parallelize_tiles {
                output_channel
                    .par_chunks_mut(output_frame)
                    .enumerate()
                    .for_each(|(temporal, output_frame)| {
                        output_frame
                            .par_chunks_mut(output_tile)
                            .enumerate()
                            .for_each(|(tile_index, output_tile)| {
                                process_tile(temporal, tile_index * output_row_tile, output_tile);
                            });
                    });
            } else {
                for temporal in 0..depth {
                    let frame_start = temporal * output_frame;
                    let output_frame = &mut output_channel[frame_start..frame_start + output_frame];
                    for (tile_index, output_tile) in
                        output_frame.chunks_mut(output_tile).enumerate()
                    {
                        process_tile(temporal, tile_index * output_row_tile, output_tile);
                    }
                }
            }
        });
    Ok(output_shape)
}

/// Nearest-exact 2x spatial upsampling for an NCTHW tensor.
#[allow(clippy::too_many_arguments)]
pub fn nearest_2x_ncthw(
    input: &[f32],
    batch: usize,
    channels: usize,
    depth: usize,
    height: usize,
    width: usize,
    output: &mut [f32],
) -> Result<()> {
    let input_len = exact_len(&[batch, channels, depth, height, width], "nearest input")?;
    let output_height = checked_mul(height, 2, "nearest output height")?;
    let output_width = checked_mul(width, 2, "nearest output width")?;
    let output_len = exact_len(
        &[batch, channels, depth, output_height, output_width],
        "nearest output",
    )?;
    if batch == 0
        || channels == 0
        || depth == 0
        || height == 0
        || width == 0
        || input.len() != input_len
        || output.len() != output_len
    {
        return Err(XrtError::Shape(format!(
            "nearest 2x NCTHW shape mismatch: input={}, output={}, expected input={input_len}, output={output_len}",
            input.len(),
            output.len()
        )));
    }
    let input_plane = depth * height * width;
    let output_plane = depth * output_height * output_width;
    output
        .par_chunks_mut(output_plane)
        .enumerate()
        .for_each(|(batch_channel, output_channel)| {
            let input_channel =
                &input[batch_channel * input_plane..(batch_channel + 1) * input_plane];
            for temporal in 0..depth {
                for output_y in 0..output_height {
                    for output_x in 0..output_width {
                        output_channel[temporal * output_height * output_width
                            + output_y * output_width
                            + output_x] = input_channel
                            [temporal * height * width + (output_y / 2) * width + output_x / 2];
                    }
                }
            }
        });
    Ok(())
}

/// PyTorch-layout 3D cross-correlation for NCTHW input and
/// `[out_channels, in_channels, kernel_t, kernel_h, kernel_w]` weights.
/// Padding order is `[front, back, top, bottom, left, right]`.
#[allow(clippy::too_many_arguments)]
pub fn causal_conv3d_ncthw(
    input: &[f32],
    batch: usize,
    input_channels: usize,
    input_depth: usize,
    input_height: usize,
    input_width: usize,
    weight: &[f32],
    output_channels: usize,
    kernel: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    bias: Option<&[f32]>,
    output: &mut [f32],
) -> Result<[usize; 3]> {
    if kernel.contains(&0)
        || stride.contains(&0)
        || batch == 0
        || input_channels == 0
        || output_channels == 0
        || input_depth == 0
        || input_height == 0
        || input_width == 0
    {
        return Err(XrtError::Shape(
            "causal conv3d channels, kernel, and stride must be positive".to_string(),
        ));
    }
    let input_len = exact_len(
        &[
            batch,
            input_channels,
            input_depth,
            input_height,
            input_width,
        ],
        "causal conv3d input",
    )?;
    let weight_len = exact_len(
        &[
            output_channels,
            input_channels,
            kernel[0],
            kernel[1],
            kernel[2],
        ],
        "causal conv3d weight",
    )?;
    if input.len() != input_len
        || weight.len() != weight_len
        || bias.is_some_and(|values| values.len() != output_channels)
    {
        return Err(XrtError::Shape(format!(
            "causal conv3d input/weight/bias mismatch: input={}, weight={}, bias={:?}",
            input.len(),
            weight.len(),
            bias.map(<[f32]>::len)
        )));
    }

    let padded_depth = input_depth
        .checked_add(2usize.checked_mul(padding[0]).ok_or_else(|| {
            XrtError::Shape("causal conv3d temporal padding overflow".to_string())
        })?)
        .ok_or_else(|| XrtError::Shape("causal conv3d depth overflow".to_string()))?;
    let padded_height =
        input_height
            .checked_add(2usize.checked_mul(padding[1]).ok_or_else(|| {
                XrtError::Shape("causal conv3d height padding overflow".to_string())
            })?)
            .ok_or_else(|| XrtError::Shape("causal conv3d height overflow".to_string()))?;
    let padded_width =
        input_width
            .checked_add(2usize.checked_mul(padding[2]).ok_or_else(|| {
                XrtError::Shape("causal conv3d width padding overflow".to_string())
            })?)
            .ok_or_else(|| XrtError::Shape("causal conv3d width overflow".to_string()))?;
    for (padded, kernel, label) in [
        (padded_depth, kernel[0], "depth"),
        (padded_height, kernel[1], "height"),
        (padded_width, kernel[2], "width"),
    ] {
        if padded < kernel {
            return Err(XrtError::Shape(format!(
                "causal conv3d padded {label} {padded} is smaller than kernel {kernel}"
            )));
        }
    }
    let output_shape = [
        (padded_depth - kernel[0]) / stride[0] + 1,
        (padded_height - kernel[1]) / stride[1] + 1,
        (padded_width - kernel[2]) / stride[2] + 1,
    ];
    let output_len = exact_len(
        &[
            batch,
            output_channels,
            output_shape[0],
            output_shape[1],
            output_shape[2],
        ],
        "causal conv3d output",
    )?;
    if output.len() != output_len {
        return Err(XrtError::Shape(format!(
            "causal conv3d output length {} does not match expected {output_len}",
            output.len()
        )));
    }

    let output_plane = output_shape[0] * output_shape[1] * output_shape[2];
    let output_frame = output_shape[1] * output_shape[2];
    let output_row_tile = CONV_OUTPUT_ROW_TILE;
    let output_tile = output_row_tile * output_shape[2];
    let input_plane = input_depth * input_height * input_width;
    let kernel_plane = kernel[0] * kernel[1] * kernel[2];
    let parallelize_tiles = batch.saturating_mul(output_channels) < rayon::current_num_threads()
        && output_shape[0].saturating_mul(output_shape[1].div_ceil(output_row_tile)) > 1;
    output
        .par_chunks_mut(output_plane)
        .enumerate()
        .for_each(|(batch_channel, output_channel)| {
            let batch_index = batch_channel / output_channels;
            let output_channel_index = batch_channel % output_channels;
            let bias_value = bias.map_or(0.0, |bias| bias[output_channel_index]);
            let process_tile = |output_t: usize, tile_y: usize, output_tile: &mut [f32]| {
                output_tile.fill(bias_value);
                let tile_rows = output_tile.len() / output_shape[2];
                let tile_y_end = tile_y + tile_rows;
                for input_channel in 0..input_channels {
                    let input_channel_offset =
                        batch_index * input_channels * input_plane + input_channel * input_plane;
                    let weight_channel_offset =
                        output_channel_index * input_channels * kernel_plane
                            + input_channel * kernel_plane;
                    for kernel_t in 0..kernel[0] {
                        let padded_t = output_t * stride[0] + kernel_t;
                        let temporal_padding = 2 * padding[0];
                        if padded_t < temporal_padding {
                            continue;
                        }
                        let input_t = padded_t - temporal_padding;
                        if input_t >= input_depth {
                            continue;
                        }
                        for kernel_y in 0..kernel[1] {
                            for kernel_x in 0..kernel[2] {
                                let weight_value = weight[weight_channel_offset
                                    + kernel_t * kernel[1] * kernel[2]
                                    + kernel_y * kernel[2]
                                    + kernel_x];
                                for output_y in tile_y..tile_y_end {
                                    let padded_y = output_y * stride[1] + kernel_y;
                                    if padded_y < padding[1] {
                                        continue;
                                    }
                                    let input_y = padded_y - padding[1];
                                    if input_y >= input_height {
                                        continue;
                                    }
                                    let input_row = input_channel_offset
                                        + input_t * input_height * input_width
                                        + input_y * input_width;
                                    let local_output_y = output_y - tile_y;
                                    let output_row = local_output_y * output_shape[2];
                                    accumulate_conv_row(
                                        &input[input_row..input_row + input_width],
                                        &mut output_tile[output_row..output_row + output_shape[2]],
                                        kernel_x,
                                        stride[2],
                                        padding[2],
                                        weight_value,
                                    );
                                }
                            }
                        }
                    }
                }
            };

            if parallelize_tiles {
                output_channel
                    .par_chunks_mut(output_frame)
                    .enumerate()
                    .for_each(|(output_t, output_frame)| {
                        output_frame
                            .par_chunks_mut(output_tile)
                            .enumerate()
                            .for_each(|(tile_index, output_tile)| {
                                process_tile(output_t, tile_index * output_row_tile, output_tile);
                            });
                    });
            } else {
                for output_t in 0..output_shape[0] {
                    let frame_start = output_t * output_frame;
                    let output_frame = &mut output_channel[frame_start..frame_start + output_frame];
                    for (tile_index, output_tile) in
                        output_frame.chunks_mut(output_tile).enumerate()
                    {
                        process_tile(output_t, tile_index * output_row_tile, output_tile);
                    }
                }
            }
        });
    Ok(output_shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "value {index}: actual={actual}, expected={expected}"
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn serial_attention_oracle(
        query: &[f32],
        key: &[f32],
        value: &[f32],
        batch: usize,
        query_sequence: usize,
        key_sequence: usize,
        heads: usize,
        head_dim: usize,
        key_mask: &[u8],
    ) -> Vec<f32> {
        let mut output = vec![0.0; query.len()];
        let scale = (head_dim as f32).sqrt().recip();
        let query_stride = heads * head_dim;
        let batch_query_stride = query_sequence * query_stride;
        let batch_key_stride = key_sequence * query_stride;
        let mut scores = vec![0.0; key_sequence];
        for batch_index in 0..batch {
            for query_index in 0..query_sequence {
                for head in 0..heads {
                    let query_start = batch_index * batch_query_stride
                        + query_index * query_stride
                        + head * head_dim;
                    for (key_index, score) in scores.iter_mut().enumerate() {
                        if key_mask[batch_index * key_sequence + key_index] == 0 {
                            *score = f32::NEG_INFINITY;
                            continue;
                        }
                        let key_start = batch_index * batch_key_stride
                            + key_index * query_stride
                            + head * head_dim;
                        *score = dot(
                            &query[query_start..query_start + head_dim],
                            &key[key_start..key_start + head_dim],
                        ) * scale;
                    }
                    reference_softmax(&mut scores);
                    for (key_index, probability) in scores.iter().copied().enumerate() {
                        if probability == 0.0 {
                            continue;
                        }
                        let value_start = batch_index * batch_key_stride
                            + key_index * query_stride
                            + head * head_dim;
                        for feature in 0..head_dim {
                            output[query_start + feature] +=
                                probability * value[value_start + feature];
                        }
                    }
                }
            }
        }
        output
    }

    #[test]
    fn linear_uses_pytorch_weight_order_and_bias() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let weight = [1.0, 0.0, 0.0, 1.0, 2.0, -1.0];
        let mut output = [0.0; 6];
        linear_f32(
            &input,
            2,
            2,
            &weight,
            3,
            Some(&[0.5, -0.5, 1.0]),
            &mut output,
        )
        .unwrap();
        assert_eq!(output, [1.5, 1.5, 1.0, 3.5, 3.5, 3.0]);
    }

    #[test]
    fn bf16_linear_decodes_without_an_expanded_weight_copy() {
        let weight = [1.0f32, 0.0, 0.0, 1.0, 2.0, -1.0]
            .into_iter()
            .flat_map(|value| bf16::from_f32(value).to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        let mut output = [0.0; 3];
        linear_bf16(
            &[1.0, 2.0],
            1,
            2,
            &weight,
            3,
            Some(&[0.5, -0.5, 1.0]),
            &mut output,
        )
        .unwrap();
        assert_eq!(output, [1.5, 1.5, 1.0]);
    }

    #[test]
    fn feature_major_bf16_linear_is_bit_exact_with_single_row_schedule() {
        const ROWS: usize = 5;
        const INPUT_FEATURES: usize = 33;
        const OUTPUT_FEATURES: usize = 17;
        let input = (0..ROWS * INPUT_FEATURES)
            .map(|index| ((index * 17 % 101) as f32 - 50.0) / 37.0)
            .collect::<Vec<_>>();
        let weight = (0..OUTPUT_FEATURES * INPUT_FEATURES)
            .flat_map(|index| {
                bf16::from_f32(((index * 13 % 89) as f32 - 44.0) / 53.0)
                    .to_bits()
                    .to_le_bytes()
            })
            .collect::<Vec<_>>();
        let bias = (0..OUTPUT_FEATURES)
            .map(|index| (index as f32 - 8.0) / 19.0)
            .collect::<Vec<_>>();
        let mut batched = vec![0.0; ROWS * OUTPUT_FEATURES];
        linear_bf16(
            &input,
            ROWS,
            INPUT_FEATURES,
            &weight,
            OUTPUT_FEATURES,
            Some(&bias),
            &mut batched,
        )
        .unwrap();

        let mut expected = Vec::with_capacity(batched.len());
        for input_row in input.chunks_exact(INPUT_FEATURES) {
            let mut output_row = vec![0.0; OUTPUT_FEATURES];
            linear_bf16(
                input_row,
                1,
                INPUT_FEATURES,
                &weight,
                OUTPUT_FEATURES,
                Some(&bias),
                &mut output_row,
            )
            .unwrap();
            expected.extend(output_row);
        }
        assert_eq!(
            batched
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn ordered_avx2_bf16_dot_is_bit_exact_with_scalar_accumulation() {
        if !simd::has_avx2_fma() {
            return;
        }
        for length in [1, 7, 8, 9, 31, 32, 33, 257] {
            let input = (0..length)
                .map(|index| ((index * 17 % 41) as f32 - 20.0) * 0.03125)
                .collect::<Vec<_>>();
            let weights = (0..length)
                .map(|index| bf16::from_f32(((index * 13 % 37) as f32 - 18.0) * 0.0625))
                .collect::<Vec<_>>();
            let encoded = weights
                .iter()
                .flat_map(|weight| weight.to_bits().to_le_bytes())
                .collect::<Vec<_>>();
            let mut expected = -0.375f32;
            for (input, weight) in input.iter().zip(&weights) {
                expected += input * weight.to_f32();
            }
            let actual = unsafe { simd::dot_bf16_f32_ordered_avx2(&input, &encoded, -0.375) };
            assert_eq!(actual.to_bits(), expected.to_bits(), "length={length}");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    #[ignore = "manual release-mode ordered BF16 dot benchmark"]
    fn ordered_avx2_bf16_dot_reports_scalar_baseline() {
        if !simd::has_avx2_fma() {
            eprintln!("ordered BF16 benchmark skipped: AVX2/FMA unavailable");
            return;
        }
        let length = 3_584;
        let repetitions = 20_000;
        let input = (0..length)
            .map(|index| ((index * 17 % 41) as f32 - 20.0) * 0.03125)
            .collect::<Vec<_>>();
        let encoded = (0..length)
            .flat_map(|index| {
                bf16::from_f32(((index * 13 % 37) as f32 - 18.0) * 0.0625)
                    .to_bits()
                    .to_le_bytes()
            })
            .collect::<Vec<_>>();

        let scalar_started = std::time::Instant::now();
        let mut scalar = 0.0f32;
        for _ in 0..repetitions {
            let mut sum = -0.375f32;
            for (input, encoded) in input.iter().zip(encoded.chunks_exact(2)) {
                let weight = bf16::from_bits(u16::from_le_bytes([encoded[0], encoded[1]])).to_f32();
                sum += input * weight;
            }
            scalar = std::hint::black_box(sum);
        }
        let scalar_seconds = scalar_started.elapsed().as_secs_f64();

        let avx_started = std::time::Instant::now();
        let mut avx = 0.0f32;
        for _ in 0..repetitions {
            avx = std::hint::black_box(unsafe {
                simd::dot_bf16_f32_ordered_avx2(&input, &encoded, -0.375)
            });
        }
        let avx_seconds = avx_started.elapsed().as_secs_f64();
        assert_eq!(avx.to_bits(), scalar.to_bits());
        eprintln!(
            "ordered BF16 dot benchmark: scalar={scalar_seconds:.6}s avx2={avx_seconds:.6}s speedup={:.3}x length={length} repetitions={repetitions}",
            scalar_seconds / avx_seconds
        );
    }

    #[test]
    fn f16_and_f32_byte_linears_preserve_pytorch_weight_order() {
        let values = [1.0f32, 0.0, 0.0, 1.0, 2.0, -1.0];
        let f16_weight = values
            .into_iter()
            .flat_map(|value| f16::from_f32(value).to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        let f32_weight = values
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let mut f16_output = [0.0; 3];
        let mut f32_output = [0.0; 3];
        linear_f16(
            &[1.0, 2.0],
            1,
            2,
            &f16_weight,
            3,
            Some(&[0.5, -0.5, 1.0]),
            &mut f16_output,
        )
        .unwrap();
        linear_f32_bytes(
            &[1.0, 2.0],
            1,
            2,
            &f32_weight,
            3,
            Some(&[0.5, -0.5, 1.0]),
            &mut f32_output,
        )
        .unwrap();
        assert_eq!(f16_output, [1.5, 1.5, 1.0]);
        assert_eq!(f32_output, f16_output);
    }

    #[test]
    fn normalization_matches_population_formulas() {
        let mut layer = [1.0, 2.0, 3.0, 4.0];
        layer_norm_rows(&mut layer, 1, 4, 1e-5).unwrap();
        assert_close(
            &layer,
            &[-1.341_635_5, -0.447_211_83, 0.447_211_83, 1.341_635_5],
            2e-6,
        );
        let mut rms = [1.0, 2.0];
        rms_norm_rows(&mut rms, 1, 2, Some(&[2.0, 0.5]), 1e-6).unwrap();
        assert_close(&rms, &[1.264_910_8, 0.632_455_4], 2e-6);
    }

    #[test]
    fn complex_rope_rotates_adjacent_pairs() {
        let mut values = [1.0, 2.0, 3.0, 4.0];
        apply_complex_rope(&mut values, 1, 1, 1, 4, &[0.0, 1.0], &[1.0, 0.0]).unwrap();
        assert_eq!(values, [-2.0, 1.0, 3.0, 4.0]);
    }

    #[test]
    fn attention_excludes_masked_keys() {
        let query = [1.0, 0.0];
        let key = [1.0, 0.0, 100.0, 0.0];
        let value = [2.0, 3.0, 50.0, 60.0];
        let mut output = [0.0; 2];
        scaled_dot_product_attention(
            &query,
            &key,
            &value,
            1,
            1,
            2,
            1,
            2,
            Some(&[1, 0]),
            &mut output,
        )
        .unwrap();
        assert_eq!(output, [2.0, 3.0]);
    }

    #[test]
    fn parallel_attention_matches_serial_multi_batch_multi_head_oracle() {
        const BATCH: usize = 2;
        const QUERY_SEQUENCE: usize = 3;
        const KEY_SEQUENCE: usize = 4;
        const HEADS: usize = 2;
        const HEAD_DIM: usize = 3;
        let query = (0..BATCH * QUERY_SEQUENCE * HEADS * HEAD_DIM)
            .map(|index| (index as f32 % 11.0 - 5.0) * 0.07)
            .collect::<Vec<_>>();
        let key = (0..BATCH * KEY_SEQUENCE * HEADS * HEAD_DIM)
            .map(|index| (index as f32 % 13.0 - 6.0) * 0.05)
            .collect::<Vec<_>>();
        let value = (0..BATCH * KEY_SEQUENCE * HEADS * HEAD_DIM)
            .map(|index| (index as f32 % 17.0 - 8.0) * 0.03)
            .collect::<Vec<_>>();
        let mask = [1, 1, 0, 1, 1, 0, 1, 1];
        let expected = serial_attention_oracle(
            &query,
            &key,
            &value,
            BATCH,
            QUERY_SEQUENCE,
            KEY_SEQUENCE,
            HEADS,
            HEAD_DIM,
            &mask,
        );

        let mut actual = vec![0.0; query.len()];
        scaled_dot_product_attention(
            &query,
            &key,
            &value,
            BATCH,
            QUERY_SEQUENCE,
            KEY_SEQUENCE,
            HEADS,
            HEAD_DIM,
            Some(&mask),
            &mut actual,
        )
        .unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    #[ignore = "manual release-mode image attention benchmark"]
    fn parallel_attention_reports_serial_baseline() {
        const BATCH: usize = 1;
        const QUERY_SEQUENCE: usize = 256;
        const KEY_SEQUENCE: usize = 256;
        const HEADS: usize = 8;
        const HEAD_DIM: usize = 64;
        let query = (0..BATCH * QUERY_SEQUENCE * HEADS * HEAD_DIM)
            .map(|index| (index as f32 % 29.0 - 14.0) * 0.01)
            .collect::<Vec<_>>();
        let key = (0..BATCH * KEY_SEQUENCE * HEADS * HEAD_DIM)
            .map(|index| (index as f32 % 31.0 - 15.0) * 0.008)
            .collect::<Vec<_>>();
        let value = (0..BATCH * KEY_SEQUENCE * HEADS * HEAD_DIM)
            .map(|index| (index as f32 % 37.0 - 18.0) * 0.006)
            .collect::<Vec<_>>();
        let mask = vec![1; BATCH * KEY_SEQUENCE];

        let serial_started = std::time::Instant::now();
        let expected = serial_attention_oracle(
            &query,
            &key,
            &value,
            BATCH,
            QUERY_SEQUENCE,
            KEY_SEQUENCE,
            HEADS,
            HEAD_DIM,
            &mask,
        );
        let serial_seconds = serial_started.elapsed().as_secs_f64();
        let mut actual = vec![0.0; query.len()];
        let parallel_started = std::time::Instant::now();
        scaled_dot_product_attention(
            &query,
            &key,
            &value,
            BATCH,
            QUERY_SEQUENCE,
            KEY_SEQUENCE,
            HEADS,
            HEAD_DIM,
            Some(&mask),
            &mut actual,
        )
        .unwrap();
        let parallel_seconds = parallel_started.elapsed().as_secs_f64();
        assert_eq!(actual, expected);
        eprintln!(
            "image attention benchmark: serial={serial_seconds:.6}s parallel={parallel_seconds:.6}s speedup={:.3}x rayon_threads={}",
            serial_seconds / parallel_seconds,
            rayon::current_num_threads()
        );
    }

    #[test]
    fn grouped_attention_is_causal_and_reuses_kv_heads() {
        // Two query heads share one KV head. The first token must not see the
        // much larger second value, while the second token may attend both.
        let query = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let key = [1.0, 0.0, 1.0, 0.0];
        let value = [2.0, 3.0, 50.0, 60.0];
        let mut output = [0.0; 8];
        grouped_causal_attention(&query, &key, &value, 1, 2, 2, 1, 2, &mut output).unwrap();
        assert_eq!(&output[..4], &[2.0, 3.0, 2.0, 3.0]);
        assert!(output[4] > 2.0 && output[6] > 2.0);
    }

    #[test]
    fn causal_conv3d_never_reads_future_frames() {
        let input = [1.0, 2.0];
        let weight = [1.0, 10.0, 100.0];
        let mut output = [0.0; 2];
        let shape = causal_conv3d_ncthw(
            &input,
            1,
            1,
            2,
            1,
            1,
            &weight,
            1,
            [3, 1, 1],
            [1, 1, 1],
            [1, 0, 0],
            None,
            &mut output,
        )
        .unwrap();
        assert_eq!(shape, [2, 1, 1]);
        assert_eq!(output, [100.0, 210.0]);
    }

    #[test]
    fn vae_rms_norm_operates_across_channels() {
        let mut values = [3.0, 0.0, 4.0, 0.0];
        vae_rms_norm_channels_ncthw(&mut values, 1, 2, 1, 1, 2, &[1.0, 0.5], 1e-12).unwrap();
        assert_close(&values, &[0.848_528_15, 0.0, 0.565_685_45, 0.0], 2e-6);
    }

    #[test]
    fn conv2d_preserves_temporal_frames_and_honors_asymmetric_padding() {
        let input = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let mut output = [0.0; 2];
        let shape = conv2d_ncthw(
            &input,
            1,
            1,
            2,
            2,
            2,
            &[1.0; 9],
            1,
            [3, 3],
            [2, 2],
            [0, 1, 0, 1],
            None,
            &mut output,
        )
        .unwrap();
        assert_eq!(shape, [1, 1]);
        assert_eq!(output, [10.0, 100.0]);
    }

    #[test]
    fn spatial_tile_convolution_parallelism_is_bit_exact() {
        const INPUT_CHANNELS: usize = 3;
        const OUTPUT_CHANNELS: usize = 2;
        const DEPTH: usize = 2;
        const HEIGHT: usize = 17;
        const WIDTH: usize = 19;

        let input = (0..INPUT_CHANNELS * DEPTH * HEIGHT * WIDTH)
            .map(|index| ((index * 13 % 251) as f32 - 125.0) / 97.0)
            .collect::<Vec<_>>();
        let bias = [0.125, -0.25];
        let conv2d_weight = (0..OUTPUT_CHANNELS * INPUT_CHANNELS * 3 * 3)
            .map(|index| ((index * 7 % 31) as f32 - 15.0) / 43.0)
            .collect::<Vec<_>>();
        let conv3d_weight = (0..OUTPUT_CHANNELS * INPUT_CHANNELS * 3 * 3 * 3)
            .map(|index| ((index * 11 % 47) as f32 - 23.0) / 59.0)
            .collect::<Vec<_>>();

        let run_conv2d = |threads| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let mut output = vec![0.0; OUTPUT_CHANNELS * DEPTH * HEIGHT * WIDTH];
            pool.install(|| {
                assert_eq!(
                    conv2d_ncthw(
                        &input,
                        1,
                        INPUT_CHANNELS,
                        DEPTH,
                        HEIGHT,
                        WIDTH,
                        &conv2d_weight,
                        OUTPUT_CHANNELS,
                        [3, 3],
                        [1, 1],
                        [1, 1, 1, 1],
                        Some(&bias),
                        &mut output,
                    )
                    .unwrap(),
                    [HEIGHT, WIDTH]
                );
            });
            output
        };
        let run_conv3d = |threads| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let mut output = vec![0.0; OUTPUT_CHANNELS * DEPTH * HEIGHT * WIDTH];
            pool.install(|| {
                assert_eq!(
                    causal_conv3d_ncthw(
                        &input,
                        1,
                        INPUT_CHANNELS,
                        DEPTH,
                        HEIGHT,
                        WIDTH,
                        &conv3d_weight,
                        OUTPUT_CHANNELS,
                        [3, 3, 3],
                        [1, 1, 1],
                        [1, 1, 1],
                        Some(&bias),
                        &mut output,
                    )
                    .unwrap(),
                    [DEPTH, HEIGHT, WIDTH]
                );
            });
            output
        };

        assert_eq!(run_conv2d(1), run_conv2d(4));
        assert_eq!(run_conv3d(1), run_conv3d(4));
    }

    #[test]
    fn convolution_row_fast_path_matches_scalar_padding_and_stride_bounds() {
        for input_width in 1..17 {
            let input = (0..input_width)
                .map(|index| index as f32 * 0.25 - 1.0)
                .collect::<Vec<_>>();
            for output_width in 1..17 {
                for stride in 1..=4 {
                    for padding_left in 0..=4 {
                        for kernel_x in 0..=5 {
                            let mut expected = vec![0.125; output_width];
                            for (output_x, output) in expected.iter_mut().enumerate() {
                                let padded_x = output_x * stride + kernel_x;
                                if padded_x >= padding_left {
                                    let input_x = padded_x - padding_left;
                                    if input_x < input.len() {
                                        *output += input[input_x] * 0.75;
                                    }
                                }
                            }

                            let mut actual = vec![0.125; output_width];
                            accumulate_conv_row(
                                &input,
                                &mut actual,
                                kernel_x,
                                stride,
                                padding_left,
                                0.75,
                            );
                            assert_eq!(actual, expected);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn nearest_2x_duplicates_each_spatial_value() {
        let mut output = [0.0; 8];
        nearest_2x_ncthw(&[1.0, 2.0], 1, 1, 1, 1, 2, &mut output).unwrap();
        assert_eq!(output, [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0]);
    }
}
