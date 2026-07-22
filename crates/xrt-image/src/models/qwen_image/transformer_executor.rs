use xrt_kernels::{layer_norm_rows, rms_norm_rows, silu_inplace};

use crate::ImageError;

use super::{
    qwen_image_rotary_embeddings_for_shapes, qwen_timestep_projection,
    transformer::{qwen_image_transformer_block_impl, QwenImageLinearOperator},
    QwenImagePromptEmbeddings, QwenImageTransformerBlockWeights, QwenImageTransformerConfig,
};

/// Storage-specific access to one validated transformer tensor map. The graph
/// below is shared by SafeTensors and GGUF so operation order cannot drift
/// between precision tiers.
pub(super) trait QwenImageTransformerWeights {
    type Linear<'a>: QwenImageLinearOperator
    where
        Self: 'a;

    fn config(&self) -> &QwenImageTransformerConfig;

    fn linear(
        &self,
        prefix: &str,
        input_features: usize,
        output_features: usize,
    ) -> Result<Self::Linear<'_>, ImageError>;

    fn auxiliary(&self, name: &str) -> Result<&[f32], ImageError>;
}

#[allow(clippy::too_many_arguments)]
pub(super) fn execute_transformer<W, F>(
    weights: &W,
    packed_latents: &[f32],
    prompt: &QwenImagePromptEmbeddings,
    timestep: &[f32],
    frames: usize,
    patch_height: usize,
    patch_width: usize,
    checkpoint: F,
) -> Result<Vec<f32>, ImageError>
where
    W: QwenImageTransformerWeights,
    F: FnMut(usize) -> Result<(), ImageError>,
{
    execute_transformer_for_shapes(
        weights,
        packed_latents,
        prompt,
        timestep,
        &[[frames, patch_height, patch_width]],
        checkpoint,
    )
}

/// Execute generation or Edit-2511 conditioning using the exact ordered latent
/// sequence geometry. Edit input is `[output, source_0, source_1, ...]`; the
/// caller slices the returned prediction back to the output sequence before
/// the scheduler update.
pub(super) fn execute_transformer_for_shapes<W, F>(
    weights: &W,
    packed_latents: &[f32],
    prompt: &QwenImagePromptEmbeddings,
    timestep: &[f32],
    image_shapes: &[[usize; 3]],
    mut checkpoint: F,
) -> Result<Vec<f32>, ImageError>
where
    W: QwenImageTransformerWeights,
    F: FnMut(usize) -> Result<(), ImageError>,
{
    let config = weights.config();
    if config.use_additional_t_cond {
        return Err(ImageError::UnsupportedCapability(
            "Qwen Image additional timestep conditioning is not implemented".to_string(),
        ));
    }
    if config.use_layer3d_rope {
        return Err(ImageError::UnsupportedCapability(
            "Qwen Image Layer3D RoPE is not implemented for this checkpoint".to_string(),
        ));
    }
    if (config.zero_cond_t && image_shapes.len() < 2)
        || (!config.zero_cond_t && image_shapes.len() != 1)
    {
        return Err(ImageError::UnsupportedShape(format!(
            "transformer zero_cond_t={} is incompatible with {} ordered image sequence(s)",
            config.zero_cond_t,
            image_shapes.len()
        )));
    }
    let batch = prompt.batch_size;
    let image_sequence = image_shapes.iter().try_fold(0usize, |total, shape| {
        let sequence = checked_product(shape, "transformer image sequence")?;
        total.checked_add(sequence).ok_or_else(|| {
            ImageError::UnsupportedShape("transformer image sequence overflow".to_string())
        })
    })?;
    let expected_latents = checked_product(
        &[batch, image_sequence, config.in_channels],
        "packed transformer input",
    )?;
    let expected_prompt = checked_product(
        &[batch, prompt.sequence_length, config.joint_attention_dim],
        "transformer prompt input",
    )?;
    let expected_mask = batch.checked_mul(prompt.sequence_length).ok_or_else(|| {
        ImageError::UnsupportedShape("transformer prompt mask length overflow".to_string())
    })?;
    if batch == 0
        || packed_latents.len() != expected_latents
        || prompt.embeddings.len() != expected_prompt
        || prompt.hidden_size != config.joint_attention_dim
        || prompt.attention_mask.len() != expected_mask
        || prompt.retained_lengths.len() != batch
        || timestep.len() != batch
        || timestep.iter().any(|value| !value.is_finite())
    {
        return Err(ImageError::UnsupportedShape(format!(
            "Qwen Image transformer input mismatch: latents={}/{expected_latents}, prompt={}/{expected_prompt}, mask={}/{expected_mask}, retained={}, timestep={}/{}",
            packed_latents.len(),
            prompt.embeddings.len(),
            prompt.attention_mask.len(),
            prompt.retained_lengths.len(),
            timestep.len(),
            batch
        )));
    }

    let inner = config.inner_dim()?;
    let image_rows = checked_product(&[batch, image_sequence], "transformer image rows")?;
    let text_rows = checked_product(&[batch, prompt.sequence_length], "transformer text rows")?;
    let mut image_states = weights
        .linear("img_in", config.in_channels, inner)?
        .forward(packed_latents, image_rows)?;
    let mut normalized_prompt = prompt.embeddings.clone();
    rms_norm_rows(
        &mut normalized_prompt,
        text_rows,
        config.joint_attention_dim,
        Some(weights.auxiliary("txt_norm.weight")?),
        1e-6,
    )
    .map_err(map_kernel_error)?;
    let mut text_states = weights
        .linear("txt_in", config.joint_attention_dim, inner)?
        .forward(&normalized_prompt, text_rows)?;

    let timestep_values = if config.zero_cond_t {
        timestep
            .iter()
            .copied()
            .chain(std::iter::repeat(0.0).take(batch))
            .collect::<Vec<_>>()
    } else {
        timestep.to_vec()
    };
    let timestep_rows = timestep_values.len();
    let timestep_projection = qwen_timestep_projection(&timestep_values)?;
    let mut timestep_embedding = weights
        .linear("time_text_embed.timestep_embedder.linear_1", 256, inner)?
        .forward(&timestep_projection, timestep_rows)?;
    silu_inplace(&mut timestep_embedding);
    timestep_embedding = weights
        .linear("time_text_embed.timestep_embedder.linear_2", inner, inner)?
        .forward(&timestep_embedding, timestep_rows)?;
    let rope = qwen_image_rotary_embeddings_for_shapes(
        image_shapes,
        prompt.sequence_length,
        &config.axes_dims_rope,
    )?;
    let output_sequence = checked_product(&image_shapes[0], "output image sequence")?;
    let image_modulation_index = config.zero_cond_t.then(|| {
        let mut index = Vec::with_capacity(batch * image_sequence);
        for _ in 0..batch {
            index.extend(std::iter::repeat(0u8).take(output_sequence));
            index.extend(std::iter::repeat(1u8).take(image_sequence - output_sequence));
        }
        index
    });

    let modulation = inner.checked_mul(6).ok_or_else(|| {
        ImageError::UnsupportedShape("transformer modulation dimension overflow".to_string())
    })?;
    let feed_forward = inner.checked_mul(4).ok_or_else(|| {
        ImageError::UnsupportedShape("transformer feed-forward dimension overflow".to_string())
    })?;
    for layer in 0..config.num_layers {
        checkpoint(layer)?;
        let prefix = format!("transformer_blocks.{layer}");
        let block = QwenImageTransformerBlockWeights {
            image_modulation: weights.linear(&format!("{prefix}.img_mod.1"), inner, modulation)?,
            text_modulation: weights.linear(&format!("{prefix}.txt_mod.1"), inner, modulation)?,
            image_query: weights.linear(&format!("{prefix}.attn.to_q"), inner, inner)?,
            image_key: weights.linear(&format!("{prefix}.attn.to_k"), inner, inner)?,
            image_value: weights.linear(&format!("{prefix}.attn.to_v"), inner, inner)?,
            image_attention_output: weights.linear(
                &format!("{prefix}.attn.to_out.0"),
                inner,
                inner,
            )?,
            text_query: weights.linear(&format!("{prefix}.attn.add_q_proj"), inner, inner)?,
            text_key: weights.linear(&format!("{prefix}.attn.add_k_proj"), inner, inner)?,
            text_value: weights.linear(&format!("{prefix}.attn.add_v_proj"), inner, inner)?,
            text_attention_output: weights.linear(
                &format!("{prefix}.attn.to_add_out"),
                inner,
                inner,
            )?,
            image_query_norm: weights.auxiliary(&format!("{prefix}.attn.norm_q.weight"))?,
            image_key_norm: weights.auxiliary(&format!("{prefix}.attn.norm_k.weight"))?,
            text_query_norm: weights.auxiliary(&format!("{prefix}.attn.norm_added_q.weight"))?,
            text_key_norm: weights.auxiliary(&format!("{prefix}.attn.norm_added_k.weight"))?,
            image_mlp_in: weights.linear(
                &format!("{prefix}.img_mlp.net.0.proj"),
                inner,
                feed_forward,
            )?,
            image_mlp_out: weights.linear(
                &format!("{prefix}.img_mlp.net.2"),
                feed_forward,
                inner,
            )?,
            text_mlp_in: weights.linear(
                &format!("{prefix}.txt_mlp.net.0.proj"),
                inner,
                feed_forward,
            )?,
            text_mlp_out: weights.linear(
                &format!("{prefix}.txt_mlp.net.2"),
                feed_forward,
                inner,
            )?,
        };
        qwen_image_transformer_block_impl(
            &block,
            &mut image_states,
            &mut text_states,
            Some(&prompt.attention_mask),
            &timestep_embedding,
            batch,
            image_sequence,
            prompt.sequence_length,
            config.num_attention_heads,
            config.attention_head_dim,
            &rope,
            image_modulation_index.as_deref(),
        )?;
    }
    checkpoint(config.num_layers)?;

    let mut norm_condition = timestep_embedding[..batch * inner].to_vec();
    silu_inplace(&mut norm_condition);
    let norm_features = inner.checked_mul(2).ok_or_else(|| {
        ImageError::UnsupportedShape("transformer output modulation dimension overflow".to_string())
    })?;
    let norm_modulation = weights
        .linear("norm_out.linear", inner, norm_features)?
        .forward(&norm_condition, batch)?;
    layer_norm_rows(&mut image_states, image_rows, inner, 1e-6).map_err(map_kernel_error)?;
    for batch_index in 0..batch {
        for token in 0..image_sequence {
            for feature in 0..inner {
                let state = (batch_index * image_sequence + token) * inner + feature;
                let modulation_offset = batch_index * norm_features;
                let scale = norm_modulation[modulation_offset + feature];
                let shift = norm_modulation[modulation_offset + inner + feature];
                image_states[state] = image_states[state] * (1.0 + scale) + shift;
            }
        }
    }
    let output_features = checked_product(
        &[config.out_channels, config.patch_size, config.patch_size],
        "transformer output features",
    )?;
    let output = weights
        .linear("proj_out", inner, output_features)?
        .forward(&image_states, image_rows)?;
    if output.iter().any(|value| !value.is_finite()) {
        return Err(ImageError::Numerical {
            component: "transformer",
            step: 0,
        });
    }
    Ok(output)
}

fn checked_product(values: &[usize], label: &str) -> Result<usize, ImageError> {
    values.iter().try_fold(1usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or_else(|| ImageError::UnsupportedShape(format!("{label} overflow")))
    })
}

fn map_kernel_error(error: xrt_core::XrtError) -> ImageError {
    match error {
        xrt_core::XrtError::Shape(message) | xrt_core::XrtError::InvalidTensor(message) => {
            ImageError::UnsupportedShape(message)
        }
        other => ImageError::Execution(other.to_string()),
    }
}
