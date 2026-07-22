use xrt_core::{DType, XrtError};
use xrt_kernels::cpu::matvec_quantized_batch;
use xrt_kernels::{
    apply_complex_rope, gelu_pytorch_tanh, layer_norm_rows, linear_bf16, linear_f16, linear_f32,
    linear_f32_bytes, rms_norm_rows, scaled_dot_product_attention, silu,
};

use crate::ImageError;

const QWEN_IMAGE_ROPE_THETA: f32 = 10_000.0;
const QWEN_IMAGE_ROPE_TABLE_SIZE: usize = 4_096;

/// Real-valued form of the complex RoPE tables consumed by the image and
/// text streams. Each table uses `[sequence, head_dim / 2]` layout.
#[derive(Debug, Clone, PartialEq)]
pub struct QwenImageRotaryEmbeddings {
    pub image_cos: Vec<f32>,
    pub image_sin: Vec<f32>,
    pub text_cos: Vec<f32>,
    pub text_sin: Vec<f32>,
    pub image_sequence_length: usize,
    pub text_sequence_length: usize,
    pub head_dim: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct QwenImageLinear<'a> {
    pub weight: &'a [f32],
    pub bias: &'a [f32],
    pub input_features: usize,
    pub output_features: usize,
}

impl QwenImageLinear<'_> {
    fn forward(&self, input: &[f32], rows: usize) -> Result<Vec<f32>, ImageError> {
        let length = rows.checked_mul(self.output_features).ok_or_else(|| {
            ImageError::UnsupportedShape("linear output length overflow".to_string())
        })?;
        let mut output = vec![0.0; length];
        linear_f32(
            input,
            rows,
            self.input_features,
            self.weight,
            self.output_features,
            Some(self.bias),
            &mut output,
        )
        .map_err(map_kernel_error)?;
        Ok(output)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct QwenImageBf16Linear<'a> {
    pub weight_bytes: &'a [u8],
    pub bias: &'a [f32],
    pub input_features: usize,
    pub output_features: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct QwenImageGgufLinear<'a> {
    pub weight_bytes: &'a [u8],
    pub dtype: DType,
    pub bias: &'a [f32],
    pub input_features: usize,
    pub output_features: usize,
}

impl QwenImageGgufLinear<'_> {
    fn forward(&self, input: &[f32], rows: usize) -> Result<Vec<f32>, ImageError> {
        let length = rows.checked_mul(self.output_features).ok_or_else(|| {
            ImageError::UnsupportedShape("GGUF linear output length overflow".to_string())
        })?;
        let mut output = vec![0.0; length];
        match self.dtype {
            DType::F32 => linear_f32_bytes(
                input,
                rows,
                self.input_features,
                self.weight_bytes,
                self.output_features,
                Some(self.bias),
                &mut output,
            ),
            DType::F16 => linear_f16(
                input,
                rows,
                self.input_features,
                self.weight_bytes,
                self.output_features,
                Some(self.bias),
                &mut output,
            ),
            DType::BF16 => linear_bf16(
                input,
                rows,
                self.input_features,
                self.weight_bytes,
                self.output_features,
                Some(self.bias),
                &mut output,
            ),
            dtype if dtype.is_quantized() => {
                let result = matvec_quantized_batch(
                    self.weight_bytes,
                    self.output_features,
                    self.input_features,
                    dtype,
                    input,
                    rows,
                    &mut output,
                );
                if result.is_ok() {
                    for row in output.chunks_exact_mut(self.output_features) {
                        for (value, bias) in row.iter_mut().zip(self.bias) {
                            *value += *bias;
                        }
                    }
                }
                result
            }
            _ => Err(XrtError::Unsupported(format!(
                "Qwen Image GGUF linear does not support {:?}",
                self.dtype
            ))),
        }
        .map_err(map_kernel_error)?;
        Ok(output)
    }
}

impl QwenImageBf16Linear<'_> {
    pub(super) fn forward(&self, input: &[f32], rows: usize) -> Result<Vec<f32>, ImageError> {
        let length = rows.checked_mul(self.output_features).ok_or_else(|| {
            ImageError::UnsupportedShape("BF16 linear output length overflow".to_string())
        })?;
        let mut output = vec![0.0; length];
        linear_bf16(
            input,
            rows,
            self.input_features,
            self.weight_bytes,
            self.output_features,
            Some(self.bias),
            &mut output,
        )
        .map_err(map_kernel_error)?;
        Ok(output)
    }
}

pub(super) trait QwenImageLinearOperator {
    fn forward(&self, input: &[f32], rows: usize) -> Result<Vec<f32>, ImageError>;
}

impl QwenImageLinearOperator for QwenImageLinear<'_> {
    fn forward(&self, input: &[f32], rows: usize) -> Result<Vec<f32>, ImageError> {
        QwenImageLinear::forward(self, input, rows)
    }
}

impl QwenImageLinearOperator for QwenImageBf16Linear<'_> {
    fn forward(&self, input: &[f32], rows: usize) -> Result<Vec<f32>, ImageError> {
        QwenImageBf16Linear::forward(self, input, rows)
    }
}

impl QwenImageLinearOperator for QwenImageGgufLinear<'_> {
    fn forward(&self, input: &[f32], rows: usize) -> Result<Vec<f32>, ImageError> {
        QwenImageGgufLinear::forward(self, input, rows)
    }
}

/// One F32 reference block. Production BF16/GGUF executors use the same
/// operation order with storage-specific linear kernels.
#[derive(Debug, Clone, Copy)]
pub struct QwenImageTransformerBlockWeights<'a, L = QwenImageLinear<'a>> {
    pub image_modulation: L,
    pub text_modulation: L,
    pub image_query: L,
    pub image_key: L,
    pub image_value: L,
    pub image_attention_output: L,
    pub text_query: L,
    pub text_key: L,
    pub text_value: L,
    pub text_attention_output: L,
    pub image_query_norm: &'a [f32],
    pub image_key_norm: &'a [f32],
    pub text_query_norm: &'a [f32],
    pub text_key_norm: &'a [f32],
    pub image_mlp_in: L,
    pub image_mlp_out: L,
    pub text_mlp_in: L,
    pub text_mlp_out: L,
}

/// Execute one dual-stream Qwen Image transformer block in F32 reference
/// arithmetic. Image and text states are updated in place.
#[allow(clippy::too_many_arguments)]
pub fn qwen_image_transformer_block_f32(
    weights: &QwenImageTransformerBlockWeights<'_, QwenImageLinear<'_>>,
    image_states: &mut [f32],
    text_states: &mut [f32],
    text_mask: Option<&[u8]>,
    timestep_embedding: &[f32],
    batch: usize,
    image_sequence: usize,
    text_sequence: usize,
    heads: usize,
    head_dim: usize,
    rope: &QwenImageRotaryEmbeddings,
) -> Result<(), ImageError> {
    qwen_image_transformer_block_impl(
        weights,
        image_states,
        text_states,
        text_mask,
        timestep_embedding,
        batch,
        image_sequence,
        text_sequence,
        heads,
        head_dim,
        rope,
        None,
    )
}

/// Execute one block while retaining BF16 matrix weights in their mmap-backed
/// SafeTensors representation. Activations and small bias/norm tensors use F32.
#[allow(clippy::too_many_arguments)]
pub fn qwen_image_transformer_block_bf16(
    weights: &QwenImageTransformerBlockWeights<'_, QwenImageBf16Linear<'_>>,
    image_states: &mut [f32],
    text_states: &mut [f32],
    text_mask: Option<&[u8]>,
    timestep_embedding: &[f32],
    batch: usize,
    image_sequence: usize,
    text_sequence: usize,
    heads: usize,
    head_dim: usize,
    rope: &QwenImageRotaryEmbeddings,
) -> Result<(), ImageError> {
    qwen_image_transformer_block_impl(
        weights,
        image_states,
        text_states,
        text_mask,
        timestep_embedding,
        batch,
        image_sequence,
        text_sequence,
        heads,
        head_dim,
        rope,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn qwen_image_transformer_block_impl<L: QwenImageLinearOperator>(
    weights: &QwenImageTransformerBlockWeights<'_, L>,
    image_states: &mut [f32],
    text_states: &mut [f32],
    text_mask: Option<&[u8]>,
    timestep_embedding: &[f32],
    batch: usize,
    image_sequence: usize,
    text_sequence: usize,
    heads: usize,
    head_dim: usize,
    rope: &QwenImageRotaryEmbeddings,
    image_modulation_index: Option<&[u8]>,
) -> Result<(), ImageError> {
    if batch == 0 || image_sequence == 0 || text_sequence == 0 || heads == 0 || head_dim == 0 {
        return Err(ImageError::UnsupportedShape(
            "Qwen Image block dimensions must be positive".to_string(),
        ));
    }
    let dimension = heads.checked_mul(head_dim).ok_or_else(|| {
        ImageError::UnsupportedShape("transformer inner dimension overflow".to_string())
    })?;
    let image_rows = batch.checked_mul(image_sequence).ok_or_else(|| {
        ImageError::UnsupportedShape("image stream row count overflow".to_string())
    })?;
    let text_rows = batch.checked_mul(text_sequence).ok_or_else(|| {
        ImageError::UnsupportedShape("text stream row count overflow".to_string())
    })?;
    validate_block_inputs(
        image_states,
        text_states,
        text_mask,
        timestep_embedding,
        batch,
        image_sequence,
        text_sequence,
        dimension,
        head_dim,
        rope,
        image_modulation_index,
    )?;

    let image_modulation_rows = if image_modulation_index.is_some() {
        batch.checked_mul(2).ok_or_else(|| {
            ImageError::UnsupportedShape("edit modulation batch overflow".to_string())
        })?
    } else {
        batch
    };
    let image_modulation = modulation(
        &weights.image_modulation,
        timestep_embedding,
        image_modulation_rows,
    )?;
    let text_timestep_len = checked_product(&[batch, dimension], "text timestep embedding")?;
    let text_modulation = modulation(
        &weights.text_modulation,
        &timestep_embedding[..text_timestep_len],
        batch,
    )?;
    let image_modulated = normalized_and_modulated(
        image_states,
        &image_modulation,
        batch,
        image_sequence,
        dimension,
        0,
        image_modulation_index,
    )?;
    let text_modulated = normalized_and_modulated(
        text_states,
        &text_modulation,
        batch,
        text_sequence,
        dimension,
        0,
        None,
    )?;

    let mut image_query = weights.image_query.forward(&image_modulated, image_rows)?;
    let mut image_key = weights.image_key.forward(&image_modulated, image_rows)?;
    let image_value = weights.image_value.forward(&image_modulated, image_rows)?;
    let mut text_query = weights.text_query.forward(&text_modulated, text_rows)?;
    let mut text_key = weights.text_key.forward(&text_modulated, text_rows)?;
    let text_value = weights.text_value.forward(&text_modulated, text_rows)?;

    for (values, norm, rows) in [
        (&mut image_query, weights.image_query_norm, image_rows),
        (&mut image_key, weights.image_key_norm, image_rows),
        (&mut text_query, weights.text_query_norm, text_rows),
        (&mut text_key, weights.text_key_norm, text_rows),
    ] {
        rms_norm_rows(values, rows * heads, head_dim, Some(norm), 1e-6)
            .map_err(map_kernel_error)?;
    }
    apply_complex_rope(
        &mut image_query,
        batch,
        image_sequence,
        heads,
        head_dim,
        &rope.image_cos,
        &rope.image_sin,
    )
    .map_err(map_kernel_error)?;
    apply_complex_rope(
        &mut image_key,
        batch,
        image_sequence,
        heads,
        head_dim,
        &rope.image_cos,
        &rope.image_sin,
    )
    .map_err(map_kernel_error)?;
    apply_complex_rope(
        &mut text_query,
        batch,
        text_sequence,
        heads,
        head_dim,
        &rope.text_cos,
        &rope.text_sin,
    )
    .map_err(map_kernel_error)?;
    apply_complex_rope(
        &mut text_key,
        batch,
        text_sequence,
        heads,
        head_dim,
        &rope.text_cos,
        &rope.text_sin,
    )
    .map_err(map_kernel_error)?;

    let joint_sequence = text_sequence.checked_add(image_sequence).ok_or_else(|| {
        ImageError::UnsupportedShape("joint attention sequence overflow".to_string())
    })?;
    let joint_query = join_streams(
        &text_query,
        &image_query,
        batch,
        text_sequence,
        image_sequence,
        dimension,
    )?;
    let joint_key = join_streams(
        &text_key,
        &image_key,
        batch,
        text_sequence,
        image_sequence,
        dimension,
    )?;
    let joint_value = join_streams(
        &text_value,
        &image_value,
        batch,
        text_sequence,
        image_sequence,
        dimension,
    )?;
    let joint_mask = joint_attention_mask(text_mask, batch, text_sequence, image_sequence)?;
    let joint_len = checked_product(
        &[batch, joint_sequence, dimension],
        "joint attention output",
    )?;
    let mut joint_output = vec![0.0; joint_len];
    scaled_dot_product_attention(
        &joint_query,
        &joint_key,
        &joint_value,
        batch,
        joint_sequence,
        joint_sequence,
        heads,
        head_dim,
        Some(&joint_mask),
        &mut joint_output,
    )
    .map_err(map_kernel_error)?;
    let (text_attention, image_attention) = split_streams(
        &joint_output,
        batch,
        text_sequence,
        image_sequence,
        dimension,
    )?;
    let image_attention = weights
        .image_attention_output
        .forward(&image_attention, image_rows)?;
    let text_attention = weights
        .text_attention_output
        .forward(&text_attention, text_rows)?;
    gated_residual(
        image_states,
        &image_attention,
        &image_modulation,
        batch,
        image_sequence,
        dimension,
        2,
        image_modulation_index,
    )?;
    gated_residual(
        text_states,
        &text_attention,
        &text_modulation,
        batch,
        text_sequence,
        dimension,
        2,
        None,
    )?;

    let image_mlp_input = normalized_and_modulated(
        image_states,
        &image_modulation,
        batch,
        image_sequence,
        dimension,
        3,
        image_modulation_index,
    )?;
    let text_mlp_input = normalized_and_modulated(
        text_states,
        &text_modulation,
        batch,
        text_sequence,
        dimension,
        3,
        None,
    )?;
    let mut image_mlp = weights.image_mlp_in.forward(&image_mlp_input, image_rows)?;
    let mut text_mlp = weights.text_mlp_in.forward(&text_mlp_input, text_rows)?;
    image_mlp
        .iter_mut()
        .for_each(|value| *value = gelu_pytorch_tanh(*value));
    text_mlp
        .iter_mut()
        .for_each(|value| *value = gelu_pytorch_tanh(*value));
    let image_mlp = weights.image_mlp_out.forward(&image_mlp, image_rows)?;
    let text_mlp = weights.text_mlp_out.forward(&text_mlp, text_rows)?;
    gated_residual(
        image_states,
        &image_mlp,
        &image_modulation,
        batch,
        image_sequence,
        dimension,
        5,
        image_modulation_index,
    )?;
    gated_residual(
        text_states,
        &text_mlp,
        &text_modulation,
        batch,
        text_sequence,
        dimension,
        5,
        None,
    )?;

    if image_states
        .iter()
        .chain(text_states.iter())
        .any(|value| !value.is_finite())
    {
        return Err(ImageError::Numerical {
            component: "transformer_block",
            step: 0,
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_block_inputs(
    image_states: &[f32],
    text_states: &[f32],
    text_mask: Option<&[u8]>,
    timestep_embedding: &[f32],
    batch: usize,
    image_sequence: usize,
    text_sequence: usize,
    dimension: usize,
    head_dim: usize,
    rope: &QwenImageRotaryEmbeddings,
    image_modulation_index: Option<&[u8]>,
) -> Result<(), ImageError> {
    let image_len = checked_product(&[batch, image_sequence, dimension], "image stream")?;
    let text_len = checked_product(&[batch, text_sequence, dimension], "text stream")?;
    let mask_len = checked_product(&[batch, text_sequence], "text attention mask")?;
    let timestep_batches = if image_modulation_index.is_some() {
        batch.checked_mul(2).ok_or_else(|| {
            ImageError::UnsupportedShape("edit timestep batch overflow".to_string())
        })?
    } else {
        batch
    };
    let timestep_len = checked_product(&[timestep_batches, dimension], "timestep embedding")?;
    if image_states.len() != image_len
        || text_states.len() != text_len
        || text_mask.is_some_and(|mask| mask.len() != mask_len)
        || timestep_embedding.len() != timestep_len
        || rope.image_sequence_length != image_sequence
        || rope.text_sequence_length != text_sequence
        || rope.head_dim != head_dim
        || image_modulation_index.is_some_and(|index| {
            index.len() != batch * image_sequence || index.iter().any(|value| *value > 1)
        })
    {
        return Err(ImageError::UnsupportedShape(format!(
            "Qwen Image block input geometry mismatch: image={}, text={}, mask={:?}, temb={}, rope=[{}, {}, {}]",
            image_states.len(),
            text_states.len(),
            text_mask.map(<[u8]>::len),
            timestep_embedding.len(),
            rope.image_sequence_length,
            rope.text_sequence_length,
            rope.head_dim
        )));
    }
    Ok(())
}

fn modulation<L: QwenImageLinearOperator>(
    linear: &L,
    timestep_embedding: &[f32],
    batch: usize,
) -> Result<Vec<f32>, ImageError> {
    let mut activated = timestep_embedding.to_vec();
    activated.iter_mut().for_each(|value| *value = silu(*value));
    linear.forward(&activated, batch)
}

fn normalized_and_modulated(
    states: &[f32],
    modulation: &[f32],
    batch: usize,
    sequence: usize,
    dimension: usize,
    chunk_start: usize,
    modulation_index: Option<&[u8]>,
) -> Result<Vec<f32>, ImageError> {
    let mut output = states.to_vec();
    layer_norm_rows(&mut output, batch * sequence, dimension, 1e-6).map_err(map_kernel_error)?;
    let modulation_batches = if modulation_index.is_some() {
        batch.checked_mul(2).ok_or_else(|| {
            ImageError::UnsupportedShape("indexed modulation batch overflow".to_string())
        })?
    } else {
        batch
    };
    let expected_modulation = checked_product(&[modulation_batches, 6, dimension], "modulation")?;
    if modulation.len() != expected_modulation || chunk_start + 2 >= 6 {
        return Err(ImageError::UnsupportedShape(
            "Qwen Image modulation tensor has invalid geometry".to_string(),
        ));
    }
    for batch_index in 0..batch {
        for token in 0..sequence {
            for feature in 0..dimension {
                let state_index = (batch_index * sequence + token) * dimension + feature;
                let selected_batch = if let Some(index) = modulation_index {
                    batch_index + usize::from(index[batch_index * sequence + token]) * batch
                } else {
                    batch_index
                };
                let base = selected_batch * 6 * dimension;
                let shift = modulation[base + chunk_start * dimension + feature];
                let scale = modulation[base + (chunk_start + 1) * dimension + feature];
                output[state_index] = output[state_index] * (1.0 + scale) + shift;
            }
        }
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn gated_residual(
    states: &mut [f32],
    update: &[f32],
    modulation: &[f32],
    batch: usize,
    sequence: usize,
    dimension: usize,
    gate_chunk: usize,
    modulation_index: Option<&[u8]>,
) -> Result<(), ImageError> {
    if states.len() != update.len() || gate_chunk >= 6 {
        return Err(ImageError::UnsupportedShape(
            "Qwen Image gated residual geometry mismatch".to_string(),
        ));
    }
    for batch_index in 0..batch {
        for token in 0..sequence {
            for feature in 0..dimension {
                let index = (batch_index * sequence + token) * dimension + feature;
                let selected_batch = if let Some(modulation_index) = modulation_index {
                    batch_index
                        + usize::from(modulation_index[batch_index * sequence + token]) * batch
                } else {
                    batch_index
                };
                let gate =
                    modulation[selected_batch * 6 * dimension + gate_chunk * dimension + feature];
                states[index] += gate * update[index];
            }
        }
    }
    Ok(())
}

fn join_streams(
    text: &[f32],
    image: &[f32],
    batch: usize,
    text_sequence: usize,
    image_sequence: usize,
    dimension: usize,
) -> Result<Vec<f32>, ImageError> {
    let joint_sequence = text_sequence
        .checked_add(image_sequence)
        .ok_or_else(|| ImageError::UnsupportedShape("joint sequence overflow".to_string()))?;
    let mut output =
        vec![0.0; checked_product(&[batch, joint_sequence, dimension], "joined stream")?];
    for batch_index in 0..batch {
        let destination = batch_index * joint_sequence * dimension;
        let text_start = batch_index * text_sequence * dimension;
        let text_end = text_start + text_sequence * dimension;
        output[destination..destination + text_sequence * dimension]
            .copy_from_slice(&text[text_start..text_end]);
        let image_start = batch_index * image_sequence * dimension;
        let image_end = image_start + image_sequence * dimension;
        output[destination + text_sequence * dimension
            ..(batch_index + 1) * joint_sequence * dimension]
            .copy_from_slice(&image[image_start..image_end]);
    }
    Ok(output)
}

fn split_streams(
    joint: &[f32],
    batch: usize,
    text_sequence: usize,
    image_sequence: usize,
    dimension: usize,
) -> Result<(Vec<f32>, Vec<f32>), ImageError> {
    let joint_sequence = text_sequence
        .checked_add(image_sequence)
        .ok_or_else(|| ImageError::UnsupportedShape("joint sequence overflow".to_string()))?;
    let expected = checked_product(&[batch, joint_sequence, dimension], "split stream")?;
    if joint.len() != expected {
        return Err(ImageError::UnsupportedShape(
            "joint stream length changed before split".to_string(),
        ));
    }
    let mut text = vec![0.0; checked_product(&[batch, text_sequence, dimension], "split text")?];
    let mut image = vec![0.0; checked_product(&[batch, image_sequence, dimension], "split image")?];
    for batch_index in 0..batch {
        let source = batch_index * joint_sequence * dimension;
        let text_start = batch_index * text_sequence * dimension;
        text[text_start..text_start + text_sequence * dimension]
            .copy_from_slice(&joint[source..source + text_sequence * dimension]);
        let image_start = batch_index * image_sequence * dimension;
        image[image_start..image_start + image_sequence * dimension].copy_from_slice(
            &joint[source + text_sequence * dimension
                ..(batch_index + 1) * joint_sequence * dimension],
        );
    }
    Ok((text, image))
}

fn joint_attention_mask(
    text_mask: Option<&[u8]>,
    batch: usize,
    text_sequence: usize,
    image_sequence: usize,
) -> Result<Vec<u8>, ImageError> {
    let joint_sequence = text_sequence
        .checked_add(image_sequence)
        .ok_or_else(|| ImageError::UnsupportedShape("joint mask sequence overflow".to_string()))?;
    let mut output = vec![1u8; checked_product(&[batch, joint_sequence], "joint mask")?];
    if let Some(text_mask) = text_mask {
        for batch_index in 0..batch {
            let source = batch_index * text_sequence;
            let destination = batch_index * joint_sequence;
            output[destination..destination + text_sequence]
                .copy_from_slice(&text_mask[source..source + text_sequence]);
        }
    }
    Ok(output)
}

fn map_kernel_error(error: XrtError) -> ImageError {
    match error {
        XrtError::Shape(message) | XrtError::InvalidTensor(message) => {
            ImageError::UnsupportedShape(message)
        }
        other => ImageError::Execution(other.to_string()),
    }
}

/// Pack NCHW VAE latents into the Diffusers Qwen Image transformer layout
/// `[batch, (height/2)*(width/2), channels*4]`.
pub fn pack_latents(
    latents: &[f32],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> Result<Vec<f32>, ImageError> {
    let input_len = checked_product(&[batch, channels, height, width], "latent input")?;
    if batch == 0 || channels == 0 || height == 0 || width == 0 || height % 2 != 0 || width % 2 != 0
    {
        return Err(ImageError::UnsupportedShape(format!(
            "latent pack requires positive batch/channels and even positive height/width, found [{batch}, {channels}, {height}, {width}]"
        )));
    }
    if latents.len() != input_len {
        return Err(ImageError::UnsupportedShape(format!(
            "latent input contains {} values, expected {input_len}",
            latents.len()
        )));
    }
    let patch_height = height / 2;
    let patch_width = width / 2;
    let packed_channels = channels.checked_mul(4).ok_or_else(|| {
        ImageError::UnsupportedShape("packed latent channels overflow".to_string())
    })?;
    let output_len = checked_product(
        &[batch, patch_height, patch_width, packed_channels],
        "packed latent output",
    )?;
    let mut output = vec![0.0; output_len];
    for batch_index in 0..batch {
        for patch_y in 0..patch_height {
            for patch_x in 0..patch_width {
                let token = (batch_index * patch_height + patch_y) * patch_width + patch_x;
                for channel in 0..channels {
                    for offset_y in 0..2 {
                        for offset_x in 0..2 {
                            let source = ((batch_index * channels + channel) * height
                                + patch_y * 2
                                + offset_y)
                                * width
                                + patch_x * 2
                                + offset_x;
                            let feature = (channel * 2 + offset_y) * 2 + offset_x;
                            output[token * packed_channels + feature] = latents[source];
                        }
                    }
                }
            }
        }
    }
    Ok(output)
}

/// Inverse of [`pack_latents`], returning NCTHW with a singleton temporal
/// dimension (which is byte-identical to NCHW for one frame).
pub fn unpack_latents(
    packed: &[f32],
    batch: usize,
    packed_channels: usize,
    patch_height: usize,
    patch_width: usize,
) -> Result<Vec<f32>, ImageError> {
    if batch == 0
        || packed_channels == 0
        || packed_channels % 4 != 0
        || patch_height == 0
        || patch_width == 0
    {
        return Err(ImageError::UnsupportedShape(format!(
            "latent unpack requires positive geometry and channels divisible by four, found [{batch}, {packed_channels}, {patch_height}, {patch_width}]"
        )));
    }
    let input_len = checked_product(
        &[batch, patch_height, patch_width, packed_channels],
        "packed latent input",
    )?;
    if packed.len() != input_len {
        return Err(ImageError::UnsupportedShape(format!(
            "packed latent contains {} values, expected {input_len}",
            packed.len()
        )));
    }
    let channels = packed_channels / 4;
    let height = patch_height.checked_mul(2).ok_or_else(|| {
        ImageError::UnsupportedShape("unpacked latent height overflow".to_string())
    })?;
    let width = patch_width.checked_mul(2).ok_or_else(|| {
        ImageError::UnsupportedShape("unpacked latent width overflow".to_string())
    })?;
    let output_len = checked_product(&[batch, channels, height, width], "unpacked latent output")?;
    let mut output = vec![0.0; output_len];
    for batch_index in 0..batch {
        for patch_y in 0..patch_height {
            for patch_x in 0..patch_width {
                let token = (batch_index * patch_height + patch_y) * patch_width + patch_x;
                for channel in 0..channels {
                    for offset_y in 0..2 {
                        for offset_x in 0..2 {
                            let feature = (channel * 2 + offset_y) * 2 + offset_x;
                            let destination = ((batch_index * channels + channel) * height
                                + patch_y * 2
                                + offset_y)
                                * width
                                + patch_x * 2
                                + offset_x;
                            output[destination] = packed[token * packed_channels + feature];
                        }
                    }
                }
            }
        }
    }
    Ok(output)
}

/// Match Diffusers `Timesteps(256, flip_sin_to_cos=True,
/// downscale_freq_shift=0, scale=1000)` for transformer conditioning.
pub fn qwen_timestep_projection(timesteps: &[f32]) -> Result<Vec<f32>, ImageError> {
    if timesteps.is_empty() || timesteps.iter().any(|value| !value.is_finite()) {
        return Err(ImageError::InvalidRequest(
            "Qwen Image timesteps must be a non-empty finite vector".to_string(),
        ));
    }
    const DIMENSION: usize = 256;
    const HALF: usize = DIMENSION / 2;
    let output_len = timesteps.len().checked_mul(DIMENSION).ok_or_else(|| {
        ImageError::UnsupportedShape("timestep projection length overflow".to_string())
    })?;
    let mut output = vec![0.0f32; output_len];
    for (row, timestep) in timesteps.iter().copied().enumerate() {
        for index in 0..HALF {
            let exponent = -QWEN_IMAGE_ROPE_THETA.ln() * index as f32 / HALF as f32;
            let angle = timestep * exponent.exp() * 1_000.0;
            output[row * DIMENSION + index] = angle.cos();
            output[row * DIMENSION + HALF + index] = angle.sin();
        }
    }
    Ok(output)
}

/// Build generation-time three-axis RoPE tables matching the pinned
/// Diffusers Qwen Image implementation (`scale_rope=True`).
pub fn qwen_image_rotary_embeddings(
    frames: usize,
    patch_height: usize,
    patch_width: usize,
    text_sequence_length: usize,
    axes_dims: &[usize],
) -> Result<QwenImageRotaryEmbeddings, ImageError> {
    qwen_image_rotary_embeddings_for_shapes(
        &[[frames, patch_height, patch_width]],
        text_sequence_length,
        axes_dims,
    )
}

/// Build the concatenated image RoPE table used by Qwen Image editing. Each
/// entry is one latent sequence in reference order: generated output first,
/// followed by the ordered source-image latents. The pinned Edit-2511 config
/// uses the standard scaled RoPE implementation, where the sequence index is
/// the frame-axis offset for that sequence.
pub fn qwen_image_rotary_embeddings_for_shapes(
    image_shapes: &[[usize; 3]],
    text_sequence_length: usize,
    axes_dims: &[usize],
) -> Result<QwenImageRotaryEmbeddings, ImageError> {
    if image_shapes.is_empty()
        || image_shapes.iter().any(|shape| shape.contains(&0))
        || text_sequence_length == 0
        || axes_dims.len() != 3
        || axes_dims
            .iter()
            .any(|dimension| *dimension == 0 || dimension % 2 != 0)
    {
        return Err(ImageError::UnsupportedShape(format!(
            "invalid Qwen Image RoPE geometry: shapes={image_shapes:?}, text={text_sequence_length}, axes={axes_dims:?}"
        )));
    }
    let head_dim = axes_dims.iter().try_fold(0usize, |total, dimension| {
        total
            .checked_add(*dimension)
            .ok_or_else(|| ImageError::UnsupportedShape("RoPE head dimension overflow".to_string()))
    })?;
    let image_sequence_length = image_shapes.iter().try_fold(0usize, |total, shape| {
        let sequence = checked_product(shape, "image RoPE sequence")?;
        total.checked_add(sequence).ok_or_else(|| {
            ImageError::UnsupportedShape("concatenated image RoPE sequence overflow".to_string())
        })
    })?;
    let complex_width = head_dim / 2;
    let image_len = checked_product(&[image_sequence_length, complex_width], "image RoPE table")?;
    let text_len = checked_product(&[text_sequence_length, complex_width], "text RoPE table")?;
    let max_video_index = image_shapes.iter().fold(0usize, |largest, shape| {
        largest.max(shape[1] / 2).max(shape[2] / 2)
    });
    let largest_text_position = max_video_index
        .checked_add(text_sequence_length)
        .ok_or_else(|| ImageError::UnsupportedShape("text RoPE position overflow".to_string()))?;
    if image_shapes.iter().enumerate().any(|(index, shape)| {
        index
            .checked_add(shape[0])
            .map_or(true, |end| end > QWEN_IMAGE_ROPE_TABLE_SIZE)
            || shape[1] > QWEN_IMAGE_ROPE_TABLE_SIZE
            || shape[2] > QWEN_IMAGE_ROPE_TABLE_SIZE
    }) || largest_text_position > QWEN_IMAGE_ROPE_TABLE_SIZE
    {
        return Err(ImageError::UnsupportedShape(format!(
            "Qwen Image RoPE positions exceed the pinned {QWEN_IMAGE_ROPE_TABLE_SIZE}-entry table"
        )));
    }

    let mut image_cos = Vec::with_capacity(image_len);
    let mut image_sin = Vec::with_capacity(image_len);
    for (sequence_index, [frames, patch_height, patch_width]) in
        image_shapes.iter().copied().enumerate()
    {
        let height_positions = centered_positions(patch_height)?;
        let width_positions = centered_positions(patch_width)?;
        for frame in 0..frames {
            let frame_position = sequence_index.checked_add(frame).ok_or_else(|| {
                ImageError::UnsupportedShape("image RoPE frame position overflow".to_string())
            })?;
            for &height in &height_positions {
                for &width in &width_positions {
                    append_axis_frequencies(
                        frame_position as isize,
                        axes_dims[0],
                        &mut image_cos,
                        &mut image_sin,
                    );
                    append_axis_frequencies(height, axes_dims[1], &mut image_cos, &mut image_sin);
                    append_axis_frequencies(width, axes_dims[2], &mut image_cos, &mut image_sin);
                }
            }
        }
    }

    let mut text_cos = Vec::with_capacity(text_len);
    let mut text_sin = Vec::with_capacity(text_len);
    for token in 0..text_sequence_length {
        let position = max_video_index.checked_add(token).ok_or_else(|| {
            ImageError::UnsupportedShape("text RoPE token position overflow".to_string())
        })? as isize;
        for dimension in axes_dims {
            append_axis_frequencies(position, *dimension, &mut text_cos, &mut text_sin);
        }
    }
    debug_assert_eq!(image_cos.len(), image_len);
    debug_assert_eq!(image_sin.len(), image_len);
    debug_assert_eq!(text_cos.len(), text_len);
    debug_assert_eq!(text_sin.len(), text_len);
    Ok(QwenImageRotaryEmbeddings {
        image_cos,
        image_sin,
        text_cos,
        text_sin,
        image_sequence_length,
        text_sequence_length,
        head_dim,
    })
}

fn centered_positions(size: usize) -> Result<Vec<isize>, ImageError> {
    let negative_count = size - size / 2;
    let negative_count = isize::try_from(negative_count).map_err(|_| {
        ImageError::UnsupportedShape("RoPE position does not fit isize".to_string())
    })?;
    let positive_count = isize::try_from(size / 2).map_err(|_| {
        ImageError::UnsupportedShape("RoPE position does not fit isize".to_string())
    })?;
    Ok((-negative_count..0).chain(0..positive_count).collect())
}

fn append_axis_frequencies(
    position: isize,
    dimension: usize,
    cos: &mut Vec<f32>,
    sin: &mut Vec<f32>,
) {
    for complex_index in 0..dimension / 2 {
        let exponent = -((2 * complex_index) as f32) / dimension as f32;
        let angle = position as f32 * QWEN_IMAGE_ROPE_THETA.powf(exponent);
        cos.push(angle.cos());
        sin.push(angle.sin());
    }
}

fn checked_product(dimensions: &[usize], label: &str) -> Result<usize, ImageError> {
    dimensions.iter().try_fold(1usize, |total, dimension| {
        total
            .checked_mul(*dimension)
            .ok_or_else(|| ImageError::UnsupportedShape(format!("{label} element count overflow")))
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use half::f16;

    use super::*;

    fn fixture() -> serde_json::Value {
        serde_json::from_str(include_str!(
            "../../../../../tests/fixtures/qwen-image/operators-diffusers-0.39.json"
        ))
        .unwrap()
    }

    fn assert_close(actual: f32, expected: f32, tolerance: f32) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual}, expected={expected}"
        );
    }

    fn fixture_linear<'a>(
        parameters: &'a BTreeMap<String, Vec<f32>>,
        prefix: &str,
        input_features: usize,
        output_features: usize,
    ) -> QwenImageLinear<'a> {
        QwenImageLinear {
            weight: parameters
                .get(&format!("{prefix}.weight"))
                .unwrap()
                .as_slice(),
            bias: parameters
                .get(&format!("{prefix}.bias"))
                .unwrap()
                .as_slice(),
            input_features,
            output_features,
        }
    }

    #[test]
    fn gguf_linear_executes_batched_q8_rows_and_bias() {
        let mut weight = Vec::new();
        for quant in [1i8, 2i8] {
            weight.extend_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
            weight.extend(std::iter::repeat(quant as u8).take(32));
        }
        let linear = QwenImageGgufLinear {
            weight_bytes: &weight,
            dtype: DType::Q8_0,
            bias: &[0.5, -1.0],
            input_features: 32,
            output_features: 2,
        };
        let input = [vec![1.0f32; 32], vec![0.5f32; 32]].concat();
        let output = linear.forward(&input, 2).unwrap();
        for (actual, expected) in output.iter().zip([32.5, 63.0, 16.5, 31.0]) {
            assert_close(*actual, expected, 1e-4);
        }
    }

    #[test]
    fn latent_pack_order_matches_qwen_diffusers_permutation() {
        let fixture = fixture();
        assert_eq!(fixture["versions"]["diffusers"], "0.39.0");
        let input = fixture["pack"]["input"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect::<Vec<_>>();
        let packed = pack_latents(&input, 1, 1, 4, 4).unwrap();
        let expected = fixture["pack"]["output"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect::<Vec<_>>();
        assert_eq!(packed, expected);
        assert_eq!(unpack_latents(&packed, 1, 4, 2, 2).unwrap(), input);
    }

    #[test]
    fn timestep_projection_flips_cosine_before_sine() {
        let projection = qwen_timestep_projection(&[0.0]).unwrap();
        assert_eq!(projection.len(), 256);
        assert!(projection[..128].iter().all(|value| *value == 1.0));
        assert!(projection[128..].iter().all(|value| *value == 0.0));
    }

    #[test]
    fn generation_rope_centers_spatial_axes_and_offsets_text() {
        let fixture = fixture();
        let rope = qwen_image_rotary_embeddings(1, 2, 2, 2, &[2, 2, 2]).unwrap();
        assert_eq!(rope.image_sequence_length, 4);
        assert_eq!(rope.text_sequence_length, 2);
        assert_eq!(rope.head_dim, 6);
        for (actual, expected) in [
            (&rope.image_cos, &fixture["rope"]["image_real"]),
            (&rope.image_sin, &fixture["rope"]["image_imag"]),
            (&rope.text_cos, &fixture["rope"]["text_real"]),
            (&rope.text_sin, &fixture["rope"]["text_imag"]),
        ] {
            let expected = expected.as_array().unwrap();
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(expected) {
                assert_close(*actual, expected.as_f64().unwrap() as f32, 2e-6);
            }
        }
    }

    #[test]
    fn timestep_projection_matches_pinned_diffusers_samples() {
        let fixture = fixture();
        let input = fixture["timestep"]["input"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect::<Vec<_>>();
        let indices = fixture["timestep"]["sample_indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as usize)
            .collect::<Vec<_>>();
        let actual = qwen_timestep_projection(&input).unwrap();
        for (row, expected) in fixture["timestep"]["sample_rows"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            for (index, expected) in indices.iter().zip(expected.as_array().unwrap()) {
                assert_close(
                    actual[row * 256 + index],
                    expected.as_f64().unwrap() as f32,
                    2e-5,
                );
            }
        }
    }

    #[test]
    fn dual_stream_block_matches_pinned_diffusers_samples() {
        let fixture = fixture();
        let block = &fixture["block"];
        let parameters = block["parameters"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(parameter_index, parameter)| {
                let name = parameter["name"].as_str().unwrap().to_string();
                let length = parameter["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|dimension| dimension.as_u64().unwrap() as usize)
                    .product::<usize>();
                let values = (0..length)
                    .map(|flat_index| {
                        ((flat_index % 23) as f32 - 11.0) * 0.003
                            + (parameter_index + 1) as f32 * 0.0002
                    })
                    .collect::<Vec<_>>();
                (name, values)
            })
            .collect::<BTreeMap<_, _>>();

        let weights = QwenImageTransformerBlockWeights {
            image_modulation: fixture_linear(&parameters, "img_mod.1", 12, 72),
            text_modulation: fixture_linear(&parameters, "txt_mod.1", 12, 72),
            image_query: fixture_linear(&parameters, "attn.to_q", 12, 12),
            image_key: fixture_linear(&parameters, "attn.to_k", 12, 12),
            image_value: fixture_linear(&parameters, "attn.to_v", 12, 12),
            image_attention_output: fixture_linear(&parameters, "attn.to_out.0", 12, 12),
            text_query: fixture_linear(&parameters, "attn.add_q_proj", 12, 12),
            text_key: fixture_linear(&parameters, "attn.add_k_proj", 12, 12),
            text_value: fixture_linear(&parameters, "attn.add_v_proj", 12, 12),
            text_attention_output: fixture_linear(&parameters, "attn.to_add_out", 12, 12),
            image_query_norm: &parameters["attn.norm_q.weight"],
            image_key_norm: &parameters["attn.norm_k.weight"],
            text_query_norm: &parameters["attn.norm_added_q.weight"],
            text_key_norm: &parameters["attn.norm_added_k.weight"],
            image_mlp_in: fixture_linear(&parameters, "img_mlp.net.0.proj", 12, 48),
            image_mlp_out: fixture_linear(&parameters, "img_mlp.net.2", 48, 12),
            text_mlp_in: fixture_linear(&parameters, "txt_mlp.net.0.proj", 12, 48),
            text_mlp_out: fixture_linear(&parameters, "txt_mlp.net.2", 48, 12),
        };
        let mut image_states = (0..48)
            .map(|index| ((index % 13) as f32 - 6.0) * 0.05)
            .collect::<Vec<_>>();
        let mut text_states = (0..24)
            .map(|index| ((index % 11) as f32 - 5.0) * 0.04)
            .collect::<Vec<_>>();
        let timestep_embedding = (0..12)
            .map(|index| ((index % 7) as f32 - 3.0) * 0.03)
            .collect::<Vec<_>>();
        let text_mask = [1, 0];
        let rope = qwen_image_rotary_embeddings(1, 2, 2, 2, &[2, 2, 2]).unwrap();

        qwen_image_transformer_block_f32(
            &weights,
            &mut image_states,
            &mut text_states,
            Some(&text_mask),
            &timestep_embedding,
            1,
            4,
            2,
            2,
            6,
            &rope,
        )
        .unwrap();

        for (index, expected) in block["image_output_sample_indices"]
            .as_array()
            .unwrap()
            .iter()
            .zip(block["image_output_samples"].as_array().unwrap())
        {
            assert_close(
                image_states[index.as_u64().unwrap() as usize],
                expected.as_f64().unwrap() as f32,
                2e-5,
            );
        }
        for (index, expected) in block["text_output_sample_indices"]
            .as_array()
            .unwrap()
            .iter()
            .zip(block["text_output_samples"].as_array().unwrap())
        {
            assert_close(
                text_states[index.as_u64().unwrap() as usize],
                expected.as_f64().unwrap() as f32,
                2e-5,
            );
        }
    }
}
