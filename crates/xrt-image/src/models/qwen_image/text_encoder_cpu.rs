use std::collections::{BTreeMap, BTreeSet};

use half::bf16;
use xrt_kernels::{grouped_causal_attention, linear_bf16, rms_norm_rows, silu};
use xrt_safetensors::{HfModelBundle, SafeTensorDType, SafeTensorStore};

use crate::{ImageError, ImageModelBundle};

use super::{
    text_encoder::{assemble_retained_embeddings, text_encoder_layout, validate_token_batch},
    QwenImageBundleConfig, QwenImagePromptEmbeddings, QwenImageTextConfig, QwenImageTokenBatch,
    QwenImageVisionEmbeddings,
};

/// Portable BF16 CPU implementation of the Qwen2.5-VL text backbone used by
/// Qwen Image. Visual tensors and the unused LM head remain mmap-only.
#[derive(Debug)]
pub struct QwenImageCpuTextEncoder {
    model: HfModelBundle,
    config: QwenImageTextConfig,
    max_sequence_length: usize,
    auxiliary: BTreeMap<String, Vec<f32>>,
    emulate_bf16_compute: bool,
}

impl QwenImageCpuTextEncoder {
    pub fn load(bundle: &ImageModelBundle) -> Result<Self, ImageError> {
        let config = QwenImageBundleConfig::load(bundle)?;
        Self::load_with_config(bundle, &config)
    }

    pub fn load_with_config(
        bundle: &ImageModelBundle,
        config: &QwenImageBundleConfig,
    ) -> Result<Self, ImageError> {
        let (component_root, layout) = text_encoder_layout(bundle)?;
        let model = HfModelBundle::open_exact(component_root, layout).map_err(|error| {
            ImageError::CorruptComponent(format!(
                "Qwen Image CPU text encoder failed exact SafeTensors validation: {error}"
            ))
        })?;
        Self::from_model_bundle(
            model,
            config.text_encoder.clone(),
            config.max_sequence_length,
        )
    }

    pub fn from_model_bundle(
        model: HfModelBundle,
        config: QwenImageTextConfig,
        max_sequence_length: usize,
    ) -> Result<Self, ImageError> {
        if max_sequence_length == 0 || max_sequence_length > config.max_position_embeddings {
            return Err(ImageError::UnsupportedShape(format!(
                "text sequence limit {max_sequence_length} exceeds model limit {}",
                config.max_position_embeddings
            )));
        }
        if model.config().hidden_size != config.hidden_size
            || model.config().intermediate_size != config.intermediate_size
            || model.config().num_hidden_layers != config.num_hidden_layers
            || model.config().num_attention_heads != config.num_attention_heads
            || model.config().num_key_value_heads != config.num_key_value_heads
            || model.config().vocab_size != config.vocab_size
        {
            return Err(ImageError::UnsupportedShape(
                "Qwen Image CPU text runtime geometry differs from the validated component config"
                    .to_string(),
            ));
        }
        validate_text_store(model.tensor_store(), &config)?;
        let mut auxiliary = BTreeMap::new();
        for tensor in expected_text_tensors(&config)?
            .into_iter()
            .filter(|tensor| tensor.shape.len() == 1)
        {
            let view = model.require_tensor(&tensor.name).map_err(|error| {
                ImageError::CorruptComponent(format!(
                    "failed to map text tensor `{}`: {error}",
                    tensor.name
                ))
            })?;
            auxiliary.insert(tensor.name.clone(), decode_bf16(view.data, &tensor.name)?);
        }
        Ok(Self {
            model,
            config,
            max_sequence_length,
            auxiliary,
            emulate_bf16_compute: true,
        })
    }

    pub fn encode_tokens(
        &self,
        tokens: &QwenImageTokenBatch,
    ) -> Result<QwenImagePromptEmbeddings, ImageError> {
        self.encode_tokens_with_control(tokens, |_, _| Ok(()))
    }

    /// Encode a prompt batch with a cooperative checkpoint before every text
    /// layer and once after the final layer of each row.
    pub fn encode_tokens_with_control<F>(
        &self,
        tokens: &QwenImageTokenBatch,
        mut checkpoint: F,
    ) -> Result<QwenImagePromptEmbeddings, ImageError>
    where
        F: FnMut(usize, usize) -> Result<(), ImageError>,
    {
        let valid_lengths = validate_token_batch(tokens, self.max_sequence_length)?;
        let mut encoded_rows = Vec::with_capacity(tokens.batch_size());
        for (row_index, (row, valid_length)) in
            tokens.input_ids.iter().zip(valid_lengths).enumerate()
        {
            encoded_rows.push(self.encode_row_with_control(&row[..valid_length], |layer| {
                checkpoint(row_index, layer)
            })?);
        }
        assemble_retained_embeddings(
            tokens,
            &encoded_rows,
            self.config.hidden_size,
            self.max_sequence_length,
        )
    }

    /// Encode one Edit Plus prompt after replacing every image placeholder
    /// with its ordered visual embedding and computing Qwen2.5-VL 3D mRoPE
    /// positions. Edit Plus currently admits one prompt per invocation.
    pub fn encode_multimodal_tokens(
        &self,
        tokens: &QwenImageTokenBatch,
        vision: &QwenImageVisionEmbeddings,
    ) -> Result<QwenImagePromptEmbeddings, ImageError> {
        self.encode_multimodal_tokens_with_control(tokens, vision, |_, _| Ok(()))
    }

    pub fn encode_multimodal_tokens_with_control<F>(
        &self,
        tokens: &QwenImageTokenBatch,
        vision: &QwenImageVisionEmbeddings,
        mut checkpoint: F,
    ) -> Result<QwenImagePromptEmbeddings, ImageError>
    where
        F: FnMut(usize, usize) -> Result<(), ImageError>,
    {
        let valid_lengths = validate_token_batch(tokens, self.max_sequence_length)?;
        if tokens.batch_size() != 1 {
            return Err(ImageError::UnsupportedCapability(
                "Qwen Image Edit multimodal encoding supports one prompt at a time".to_string(),
            ));
        }
        let valid_length = valid_lengths[0];
        let encoded = self.encode_row_with_vision_observer(
            &tokens.input_ids[0][..valid_length],
            Some(vision),
            |layer| checkpoint(0, layer),
            |_, _| Ok(()),
        )?;
        assemble_retained_embeddings(
            tokens,
            &[encoded],
            self.config.hidden_size,
            self.max_sequence_length,
        )
    }

    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    fn encode_row_with_control<F>(
        &self,
        token_ids: &[u32],
        checkpoint: F,
    ) -> Result<Vec<f32>, ImageError>
    where
        F: FnMut(usize) -> Result<(), ImageError>,
    {
        self.encode_row_with_observer(token_ids, checkpoint, |_, _| Ok(()))
    }

    fn encode_row_with_observer<F, O>(
        &self,
        token_ids: &[u32],
        checkpoint: F,
        observer: O,
    ) -> Result<Vec<f32>, ImageError>
    where
        F: FnMut(usize) -> Result<(), ImageError>,
        O: FnMut(usize, &[f32]) -> Result<(), ImageError>,
    {
        self.encode_row_with_vision_observer(token_ids, None, checkpoint, observer)
    }

    fn encode_row_with_vision_observer<F, O>(
        &self,
        token_ids: &[u32],
        vision: Option<&QwenImageVisionEmbeddings>,
        mut checkpoint: F,
        mut observer: O,
    ) -> Result<Vec<f32>, ImageError>
    where
        F: FnMut(usize) -> Result<(), ImageError>,
        O: FnMut(usize, &[f32]) -> Result<(), ImageError>,
    {
        if token_ids.is_empty() || token_ids.len() > self.config.max_position_embeddings {
            return Err(ImageError::InputLimit(format!(
                "text row length {} is outside the model context",
                token_ids.len()
            )));
        }
        let sequence = token_ids.len();
        let hidden = self.config.hidden_size;
        let head_dim = hidden / self.config.num_attention_heads;
        let kv_width = checked_product(
            &[self.config.num_key_value_heads, head_dim],
            "text KV width",
        )?;
        let embedding = self
            .model
            .require_tensor("model.embed_tokens.weight")
            .map_err(|error| {
                ImageError::CorruptComponent(format!("failed to map text embeddings: {error}"))
            })?;
        let row_bytes = checked_product(&[hidden, 2], "embedding row bytes")?;
        let mut states = vec![0.0f32; checked_product(&[sequence, hidden], "text states")?];
        for (position, token) in token_ids.iter().copied().enumerate() {
            let token = token as usize;
            if token >= self.config.vocab_size {
                return Err(ImageError::InvalidRequest(format!(
                    "token id {token} exceeds text vocabulary {}",
                    self.config.vocab_size
                )));
            }
            let start = checked_product(&[token, row_bytes], "embedding byte offset")?;
            let bytes = &embedding.data[start..start + row_bytes];
            for (feature, encoded) in bytes.chunks_exact(2).enumerate() {
                states[position * hidden + feature] =
                    bf16::from_bits(u16::from_le_bytes([encoded[0], encoded[1]])).to_f32();
            }
        }
        let position_ids = if let Some(vision) = vision {
            let vision_config = self.config.vision.as_ref().ok_or_else(|| {
                ImageError::UnsupportedCapability(
                    "text encoder does not declare a Qwen2.5-VL vision tower".to_string(),
                )
            })?;
            let image_token_id = self.config.image_token_id.ok_or_else(|| {
                ImageError::UnsupportedShape(
                    "text encoder does not declare image_token_id".to_string(),
                )
            })?;
            inject_vision_embeddings(
                &mut states,
                hidden,
                token_ids,
                image_token_id,
                vision,
                vision_config.spatial_merge_size,
            )?;
            qwen_vl_position_ids(
                token_ids,
                image_token_id,
                &vision.grids,
                vision_config.spatial_merge_size,
                self.config.max_position_embeddings,
            )?
        } else {
            (0..sequence).map(|position| [position; 3]).collect()
        };

        for layer in 0..self.config.num_hidden_layers {
            checkpoint(layer)?;
            let prefix = format!("model.layers.{layer}");
            let residual = states.clone();
            rms_norm_rows(
                &mut states,
                sequence,
                hidden,
                Some(self.auxiliary(&format!("{prefix}.input_layernorm.weight"))?),
                self.config.rms_norm_eps,
            )
            .map_err(map_kernel_error)?;
            self.round_activation(&mut states);
            let mut query = self.linear(
                &format!("{prefix}.self_attn.q_proj"),
                &states,
                sequence,
                hidden,
                hidden,
                true,
            )?;
            let mut key = self.linear(
                &format!("{prefix}.self_attn.k_proj"),
                &states,
                sequence,
                hidden,
                kv_width,
                true,
            )?;
            let value = self.linear(
                &format!("{prefix}.self_attn.v_proj"),
                &states,
                sequence,
                hidden,
                kv_width,
                true,
            )?;
            for position in 0..sequence {
                apply_multimodal_rotary_qk(
                    &mut query[position * hidden..(position + 1) * hidden],
                    &mut key[position * kv_width..(position + 1) * kv_width],
                    self.config.num_attention_heads,
                    self.config.num_key_value_heads,
                    head_dim,
                    position_ids[position],
                    self.config.mrope_section,
                    self.config.rope_theta,
                )?;
            }
            self.round_activation(&mut query);
            self.round_activation(&mut key);
            let mut attention = vec![0.0; checked_product(&[sequence, hidden], "text attention")?];
            grouped_causal_attention(
                &query,
                &key,
                &value,
                1,
                sequence,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                head_dim,
                &mut attention,
            )
            .map_err(map_kernel_error)?;
            self.round_activation(&mut attention);
            states = self.linear(
                &format!("{prefix}.self_attn.o_proj"),
                &attention,
                sequence,
                hidden,
                hidden,
                false,
            )?;
            for (state, residual) in states.iter_mut().zip(residual) {
                *state += residual;
            }
            self.round_activation(&mut states);

            let residual = states.clone();
            rms_norm_rows(
                &mut states,
                sequence,
                hidden,
                Some(self.auxiliary(&format!("{prefix}.post_attention_layernorm.weight"))?),
                self.config.rms_norm_eps,
            )
            .map_err(map_kernel_error)?;
            self.round_activation(&mut states);
            let mut gate = self.linear(
                &format!("{prefix}.mlp.gate_proj"),
                &states,
                sequence,
                hidden,
                self.config.intermediate_size,
                false,
            )?;
            let up = self.linear(
                &format!("{prefix}.mlp.up_proj"),
                &states,
                sequence,
                hidden,
                self.config.intermediate_size,
                false,
            )?;
            for (gate, up) in gate.iter_mut().zip(up) {
                let activated = self.round_scalar(silu(*gate));
                *gate = self.round_scalar(activated * up);
            }
            states = self.linear(
                &format!("{prefix}.mlp.down_proj"),
                &gate,
                sequence,
                self.config.intermediate_size,
                hidden,
                false,
            )?;
            for (state, residual) in states.iter_mut().zip(residual) {
                *state += residual;
            }
            self.round_activation(&mut states);
            observer(layer, &states)?;
        }
        checkpoint(self.config.num_hidden_layers)?;
        rms_norm_rows(
            &mut states,
            sequence,
            hidden,
            Some(self.auxiliary("model.norm.weight")?),
            self.config.rms_norm_eps,
        )
        .map_err(map_kernel_error)?;
        self.round_activation(&mut states);
        observer(self.config.num_hidden_layers, &states)?;
        if states.iter().any(|value| !value.is_finite()) {
            return Err(ImageError::Numerical {
                component: "text_encoder",
                step: self.config.num_hidden_layers,
            });
        }
        Ok(states)
    }

    #[allow(clippy::too_many_arguments)]
    fn linear(
        &self,
        prefix: &str,
        input: &[f32],
        rows: usize,
        input_features: usize,
        output_features: usize,
        has_bias: bool,
    ) -> Result<Vec<f32>, ImageError> {
        let name = format!("{prefix}.weight");
        let weight = self.model.require_tensor(&name).map_err(|error| {
            ImageError::CorruptComponent(format!("failed to map `{name}`: {error}"))
        })?;
        let bias = has_bias
            .then(|| self.auxiliary(&format!("{prefix}.bias")))
            .transpose()?;
        let mut output =
            vec![0.0; checked_product(&[rows, output_features], "text linear output")?];
        linear_bf16(
            input,
            rows,
            input_features,
            weight.data,
            output_features,
            bias,
            &mut output,
        )
        .map_err(map_kernel_error)?;
        self.round_activation(&mut output);
        Ok(output)
    }

    fn round_activation(&self, values: &mut [f32]) {
        if self.emulate_bf16_compute {
            values.iter_mut().for_each(|value| {
                *value = bf16::from_f32(*value).to_f32();
            });
        }
    }

    fn round_scalar(&self, value: f32) -> f32 {
        if self.emulate_bf16_compute {
            bf16::from_f32(value).to_f32()
        } else {
            value
        }
    }

    fn auxiliary(&self, name: &str) -> Result<&[f32], ImageError> {
        self.auxiliary
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| ImageError::UnsupportedTensor(format!("missing text tensor `{name}`")))
    }
}

fn inject_vision_embeddings(
    states: &mut [f32],
    hidden_size: usize,
    token_ids: &[u32],
    image_token_id: u32,
    vision: &QwenImageVisionEmbeddings,
    spatial_merge_size: usize,
) -> Result<(), ImageError> {
    if vision.hidden_size != hidden_size
        || vision.image_token_counts.is_empty()
        || vision.image_token_counts.len() != vision.grids.len()
        || spatial_merge_size == 0
    {
        return Err(ImageError::UnsupportedShape(
            "vision embeddings do not match the text backbone geometry".to_string(),
        ));
    }
    let merge_unit = checked_product(
        &[spatial_merge_size, spatial_merge_size],
        "vision merge unit",
    )?;
    for (index, (count, grid)) in vision
        .image_token_counts
        .iter()
        .zip(&vision.grids)
        .enumerate()
    {
        if grid.contains(&0)
            || grid[1] % spatial_merge_size != 0
            || grid[2] % spatial_merge_size != 0
        {
            return Err(ImageError::UnsupportedShape(format!(
                "vision grid {index} is invalid for merge {spatial_merge_size}: {grid:?}"
            )));
        }
        let patches = checked_product(grid, "vision grid patches")?;
        if patches / merge_unit != *count {
            return Err(ImageError::UnsupportedShape(format!(
                "vision grid {index} produces {} tokens, encoder returned {count}",
                patches / merge_unit
            )));
        }
    }
    let expected_tokens = vision
        .image_token_counts
        .iter()
        .try_fold(0usize, |total, count| total.checked_add(*count))
        .ok_or_else(|| {
            ImageError::UnsupportedShape("vision embedding token count overflow".to_string())
        })?;
    let actual_tokens = token_ids
        .iter()
        .filter(|token| **token == image_token_id)
        .count();
    if actual_tokens != expected_tokens
        || vision.values.len()
            != checked_product(&[expected_tokens, hidden_size], "vision embedding values")?
        || states.len() != checked_product(&[token_ids.len(), hidden_size], "text states")?
    {
        return Err(ImageError::InvalidRequest(format!(
            "image placeholders ({actual_tokens}) do not match visual embeddings ({expected_tokens})"
        )));
    }
    let mut source_row = 0usize;
    for (token_index, token) in token_ids.iter().copied().enumerate() {
        if token != image_token_id {
            continue;
        }
        let source = source_row * hidden_size;
        let destination = token_index * hidden_size;
        states[destination..destination + hidden_size]
            .copy_from_slice(&vision.values[source..source + hidden_size]);
        source_row += 1;
    }
    Ok(())
}

fn qwen_vl_position_ids(
    token_ids: &[u32],
    image_token_id: u32,
    grids: &[[usize; 3]],
    spatial_merge_size: usize,
    max_position_embeddings: usize,
) -> Result<Vec<[usize; 3]>, ImageError> {
    if token_ids.is_empty() || grids.is_empty() || spatial_merge_size == 0 {
        return Err(ImageError::InvalidRequest(
            "multimodal position IDs require text, grids, and a merge size".to_string(),
        ));
    }
    let mut positions = Vec::with_capacity(token_ids.len());
    let mut token_index = 0usize;
    let mut grid_index = 0usize;
    let mut current = 0usize;
    while token_index < token_ids.len() {
        let is_image = token_ids[token_index] == image_token_id;
        let group_start = token_index;
        while token_index < token_ids.len()
            && (token_ids[token_index] == image_token_id) == is_image
        {
            token_index += 1;
        }
        let group_length = token_index - group_start;
        if !is_image {
            for offset in 0..group_length {
                let position = current.checked_add(offset).ok_or_else(|| {
                    ImageError::UnsupportedShape("text position overflow".to_string())
                })?;
                positions.push([position; 3]);
            }
            current = current.checked_add(group_length).ok_or_else(|| {
                ImageError::UnsupportedShape("text position overflow".to_string())
            })?;
            continue;
        }
        let [temporal, height, width] = *grids.get(grid_index).ok_or_else(|| {
            ImageError::InvalidRequest("more image token groups than image grids".to_string())
        })?;
        if temporal == 0
            || height == 0
            || width == 0
            || height % spatial_merge_size != 0
            || width % spatial_merge_size != 0
        {
            return Err(ImageError::UnsupportedShape(format!(
                "invalid multimodal grid [{temporal}, {height}, {width}]"
            )));
        }
        let merged_height = height / spatial_merge_size;
        let merged_width = width / spatial_merge_size;
        let expected = checked_product(
            &[temporal, merged_height, merged_width],
            "multimodal image positions",
        )?;
        if group_length != expected {
            return Err(ImageError::InvalidRequest(format!(
                "image token group {grid_index} has {group_length} tokens, expected {expected}"
            )));
        }
        for time in 0..temporal {
            for row in 0..merged_height {
                for column in 0..merged_width {
                    positions.push([
                        current.checked_add(time).ok_or_else(|| {
                            ImageError::UnsupportedShape(
                                "temporal multimodal position overflow".to_string(),
                            )
                        })?,
                        current.checked_add(row).ok_or_else(|| {
                            ImageError::UnsupportedShape(
                                "height multimodal position overflow".to_string(),
                            )
                        })?,
                        current.checked_add(column).ok_or_else(|| {
                            ImageError::UnsupportedShape(
                                "width multimodal position overflow".to_string(),
                            )
                        })?,
                    ]);
                }
            }
        }
        current = current
            .checked_add(merged_height.max(merged_width))
            .ok_or_else(|| {
                ImageError::UnsupportedShape("multimodal position overflow".to_string())
            })?;
        grid_index += 1;
    }
    if grid_index != grids.len() || positions.len() != token_ids.len() {
        return Err(ImageError::InvalidRequest(format!(
            "multimodal token groups ({grid_index}) do not match image grids ({})",
            grids.len()
        )));
    }
    if positions
        .iter()
        .flat_map(|position| position.iter())
        .any(|position| *position >= max_position_embeddings)
    {
        return Err(ImageError::InputLimit(format!(
            "multimodal position exceeds text context {max_position_embeddings}"
        )));
    }
    Ok(positions)
}

#[allow(clippy::too_many_arguments)]
fn apply_multimodal_rotary_qk(
    query: &mut [f32],
    key: &mut [f32],
    query_heads: usize,
    key_value_heads: usize,
    head_dim: usize,
    positions: [usize; 3],
    sections: [usize; 3],
    base: f32,
) -> Result<(), ImageError> {
    let half = head_dim / 2;
    let section_width = sections
        .iter()
        .try_fold(0usize, |total, section| total.checked_add(*section))
        .ok_or_else(|| ImageError::UnsupportedShape("mRoPE section overflow".to_string()))?;
    if head_dim % 2 != 0
        || section_width != half
        || query.len() != checked_product(&[query_heads, head_dim], "mRoPE query")?
        || key.len() != checked_product(&[key_value_heads, head_dim], "mRoPE key")?
        || !base.is_finite()
        || base <= 0.0
    {
        return Err(ImageError::UnsupportedShape(
            "invalid Qwen2.5-VL mRoPE geometry".to_string(),
        ));
    }
    apply_multimodal_rotary(query, query_heads, head_dim, positions, sections, base);
    apply_multimodal_rotary(key, key_value_heads, head_dim, positions, sections, base);
    Ok(())
}

fn apply_multimodal_rotary(
    values: &mut [f32],
    heads: usize,
    head_dim: usize,
    positions: [usize; 3],
    sections: [usize; 3],
    base: f32,
) {
    let half = head_dim / 2;
    let temporal_end = sections[0];
    let height_end = temporal_end + sections[1];
    for head in 0..heads {
        let start = head * head_dim;
        for pair in 0..half {
            let axis = if pair < temporal_end {
                0
            } else if pair < height_end {
                1
            } else {
                2
            };
            let theta = base.powf((2.0 * pair as f32) / head_dim as f32);
            let angle = positions[axis] as f32 / theta;
            let (sin, cos) = angle.sin_cos();
            let left = values[start + pair];
            let right = values[start + pair + half];
            values[start + pair] = left * cos - right * sin;
            values[start + pair + half] = left * sin + right * cos;
        }
    }
}

#[derive(Debug)]
struct ExpectedTextTensor {
    name: String,
    shape: Vec<usize>,
}

fn expected_text_tensors(
    config: &QwenImageTextConfig,
) -> Result<Vec<ExpectedTextTensor>, ImageError> {
    if config.hidden_size % config.num_attention_heads != 0
        || config.num_attention_heads % config.num_key_value_heads != 0
    {
        return Err(ImageError::UnsupportedShape(
            "Qwen text attention heads do not divide model width".to_string(),
        ));
    }
    let head_dim = config.hidden_size / config.num_attention_heads;
    let kv_width = checked_product(
        &[config.num_key_value_heads, head_dim],
        "text KV projection width",
    )?;
    let mut tensors = vec![
        ExpectedTextTensor {
            name: "model.embed_tokens.weight".to_string(),
            shape: vec![config.vocab_size, config.hidden_size],
        },
        ExpectedTextTensor {
            name: "model.norm.weight".to_string(),
            shape: vec![config.hidden_size],
        },
    ];
    for layer in 0..config.num_hidden_layers {
        let prefix = format!("model.layers.{layer}");
        for normalization in ["input_layernorm", "post_attention_layernorm"] {
            tensors.push(ExpectedTextTensor {
                name: format!("{prefix}.{normalization}.weight"),
                shape: vec![config.hidden_size],
            });
        }
        for (projection, output) in [
            ("q_proj", config.hidden_size),
            ("k_proj", kv_width),
            ("v_proj", kv_width),
        ] {
            tensors.push(ExpectedTextTensor {
                name: format!("{prefix}.self_attn.{projection}.bias"),
                shape: vec![output],
            });
            tensors.push(ExpectedTextTensor {
                name: format!("{prefix}.self_attn.{projection}.weight"),
                shape: vec![output, config.hidden_size],
            });
        }
        tensors.push(ExpectedTextTensor {
            name: format!("{prefix}.self_attn.o_proj.weight"),
            shape: vec![config.hidden_size, config.hidden_size],
        });
        for projection in ["gate_proj", "up_proj"] {
            tensors.push(ExpectedTextTensor {
                name: format!("{prefix}.mlp.{projection}.weight"),
                shape: vec![config.intermediate_size, config.hidden_size],
            });
        }
        tensors.push(ExpectedTextTensor {
            name: format!("{prefix}.mlp.down_proj.weight"),
            shape: vec![config.hidden_size, config.intermediate_size],
        });
    }
    Ok(tensors)
}

fn validate_text_store(
    store: &SafeTensorStore,
    config: &QwenImageTextConfig,
) -> Result<(), ImageError> {
    let expected = expected_text_tensors(config)?;
    let expected_names = expected
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect::<BTreeSet<_>>();
    for tensor in &expected {
        let info = store.tensor_info(&tensor.name).ok_or_else(|| {
            ImageError::UnsupportedTensor(format!("text encoder is missing `{}`", tensor.name))
        })?;
        if info.dtype != SafeTensorDType::Bf16 || info.shape != tensor.shape {
            return Err(ImageError::UnsupportedTensor(format!(
                "text tensor `{}` is {:?} {:?}, expected BF16 {:?}",
                tensor.name, info.dtype, info.shape, tensor.shape
            )));
        }
    }
    if let Some(unknown) = store.tensor_names().find(|name| {
        name.starts_with("model.") && !expected_names.contains(name) || {
            !name.starts_with("model.") && !name.starts_with("visual.") && *name != "lm_head.weight"
        }
    }) {
        return Err(ImageError::UnsupportedTensor(format!(
            "text component contains unknown tensor `{unknown}`"
        )));
    }
    Ok(())
}

fn decode_bf16(bytes: &[u8], name: &str) -> Result<Vec<f32>, ImageError> {
    if bytes.len() % 2 != 0 {
        return Err(ImageError::CorruptComponent(format!(
            "BF16 text tensor `{name}` has odd byte length {}",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|encoded| bf16::from_bits(u16::from_le_bytes([encoded[0], encoded[1]])).to_f32())
        .collect())
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

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{Dtype, TensorView};
    use std::path::PathBuf;
    use xrt_safetensors::SafeTensorLayout;

    use crate::models::qwen_image::{
        QwenImageBundleConfig, QwenImagePromptTokenizer, QwenImageVisionConfig,
    };

    fn tiny_config() -> QwenImageTextConfig {
        QwenImageTextConfig {
            architecture: "Qwen2_5_VLForConditionalGeneration".to_string(),
            dtype: "bfloat16".to_string(),
            model_type: "qwen2_5_vl".to_string(),
            hidden_size: 8,
            intermediate_size: 16,
            max_position_embeddings: 128,
            num_attention_heads: 2,
            num_hidden_layers: 2,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            mrope_section: [1, 1, 0],
            vocab_size: 16,
            image_token_id: None,
            vision_start_token_id: None,
            vision_end_token_id: None,
            vision: None,
        }
    }

    fn tiny_multimodal_config() -> QwenImageTextConfig {
        QwenImageTextConfig {
            architecture: "Qwen2_5_VLForConditionalGeneration".to_string(),
            dtype: "bfloat16".to_string(),
            model_type: "qwen2_5_vl".to_string(),
            hidden_size: 12,
            intermediate_size: 20,
            max_position_embeddings: 128,
            num_attention_heads: 2,
            num_hidden_layers: 2,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            mrope_section: [1, 1, 1],
            vocab_size: 16,
            image_token_id: Some(15),
            vision_start_token_id: Some(14),
            vision_end_token_id: Some(13),
            vision: Some(QwenImageVisionConfig {
                depth: 1,
                fullatt_block_indexes: vec![0],
                hidden_act: "silu".to_string(),
                hidden_size: 12,
                in_channels: 3,
                intermediate_size: 20,
                num_heads: 2,
                out_hidden_size: 12,
                patch_size: 2,
                spatial_merge_size: 2,
                temporal_patch_size: 2,
                window_size: 8,
            }),
        }
    }

    #[test]
    fn tiny_text_schema_covers_every_language_parameter() {
        let tensors = expected_text_tensors(&tiny_config()).unwrap();
        assert_eq!(tensors.len(), 26);
        assert!(tensors.iter().any(|tensor| {
            tensor.name == "model.layers.1.self_attn.k_proj.weight" && tensor.shape == [4, 8]
        }));
    }

    #[test]
    fn tiny_cpu_text_encoder_matches_pinned_transformers_fixture() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../../tests/fixtures/qwen-image/text-encoder-transformers-5.14.json"
        ))
        .unwrap();
        let config = tiny_config();
        let mut specifications = expected_text_tensors(&config).unwrap();
        specifications.sort_by(|left, right| left.name.cmp(&right.name));
        let encoded = specifications
            .into_iter()
            .enumerate()
            .map(|(parameter_index, tensor)| {
                let length = tensor.shape.iter().product::<usize>();
                let bytes = (0..length)
                    .flat_map(|flat_index| {
                        let value = ((flat_index % 19) as f32 - 9.0) * 0.004
                            + (parameter_index + 1) as f32 * 0.0001;
                        bf16::from_f32(value).to_bits().to_le_bytes()
                    })
                    .collect::<Vec<_>>();
                (tensor.name, (tensor.shape, bytes))
            })
            .collect::<BTreeMap<_, _>>();
        let views = encoded
            .iter()
            .map(|(name, (shape, bytes))| {
                (
                    name.as_str(),
                    TensorView::new(Dtype::BF16, shape.clone(), bytes).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let directory = tempfile::tempdir().unwrap();
        safetensors::serialize_to_file(views, &None, &directory.path().join("model.safetensors"))
            .unwrap();
        let config_json = serde_json::json!({
            "_name_or_path": "synthetic/qwen2.5-vl-text",
            "architectures": ["Qwen2_5_VLForConditionalGeneration"],
            "model_type": "qwen2_5_vl",
            "hidden_act": "silu",
            "hidden_size": 8,
            "intermediate_size": 16,
            "max_position_embeddings": 128,
            "num_attention_heads": 2,
            "num_hidden_layers": 2,
            "num_key_value_heads": 1,
            "rms_norm_eps": 0.000001,
            "rope_theta": 10000.0,
            "rope_scaling": {"type": "default", "mrope_section": [1, 1, 0]},
            "tie_word_embeddings": false,
            "torch_dtype": "bfloat16",
            "vocab_size": 16,
            "bos_token_id": 1,
            "eos_token_id": 2
        });
        std::fs::write(
            directory.path().join("config.json"),
            serde_json::to_vec(&config_json).unwrap(),
        )
        .unwrap();
        let model = HfModelBundle::open_exact(
            directory.path(),
            SafeTensorLayout::single("model.safetensors"),
        )
        .unwrap();
        let mut encoder = QwenImageCpuTextEncoder::from_model_bundle(model, config, 64).unwrap();
        // This historical tiny oracle intentionally runs an F32 model loaded
        // with BF16-rounded weights. Real production parity is covered by the
        // ignored full-model BF16 fixture in `tests/real_qwen_bundle.rs`.
        encoder.emulate_bf16_compute = false;
        let tokens = fixture["token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as u32)
            .collect::<Vec<_>>();
        let mut checkpoints = Vec::new();
        let output = encoder
            .encode_row_with_control(&tokens, |layer| {
                checkpoints.push(layer);
                Ok(())
            })
            .unwrap();
        assert_eq!(checkpoints, [0, 1, 2]);
        let expected = fixture["output"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect::<Vec<_>>();
        assert_eq!(output.len(), expected.len());
        for (index, (actual, expected)) in output.into_iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 4e-5,
                "hidden {index}: actual={actual}, expected={expected}, error={}",
                (actual - expected).abs()
            );
        }
    }

    #[test]
    fn tiny_multimodal_text_matches_pinned_transformers_fixture() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../../tests/fixtures/qwen-image/multimodal-text-transformers-5.14.json"
        ))
        .unwrap();
        let config = tiny_multimodal_config();
        let mut specifications = expected_text_tensors(&config).unwrap();
        specifications.sort_by(|left, right| left.name.cmp(&right.name));
        let encoded = specifications
            .into_iter()
            .enumerate()
            .map(|(parameter_index, tensor)| {
                let length = tensor.shape.iter().product::<usize>();
                let bytes = (0..length)
                    .flat_map(|flat_index| {
                        let value = ((flat_index % 19) as f32 - 9.0) * 0.004
                            + (parameter_index + 1) as f32 * 0.0001;
                        bf16::from_f32(value).to_bits().to_le_bytes()
                    })
                    .collect::<Vec<_>>();
                (tensor.name, (tensor.shape, bytes))
            })
            .collect::<BTreeMap<_, _>>();
        let views = encoded
            .iter()
            .map(|(name, (shape, bytes))| {
                (
                    name.as_str(),
                    TensorView::new(Dtype::BF16, shape.clone(), bytes).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let directory = tempfile::tempdir().unwrap();
        safetensors::serialize_to_file(views, &None, &directory.path().join("model.safetensors"))
            .unwrap();
        let config_json = serde_json::json!({
            "_name_or_path": "synthetic/qwen2.5-vl-multimodal",
            "architectures": ["Qwen2_5_VLForConditionalGeneration"],
            "model_type": "qwen2_5_vl",
            "hidden_act": "silu",
            "hidden_size": 12,
            "intermediate_size": 20,
            "max_position_embeddings": 128,
            "num_attention_heads": 2,
            "num_hidden_layers": 2,
            "num_key_value_heads": 1,
            "rms_norm_eps": 0.000001,
            "rope_theta": 10000.0,
            "rope_scaling": {"type": "default", "mrope_section": [1, 1, 1]},
            "tie_word_embeddings": false,
            "torch_dtype": "bfloat16",
            "vocab_size": 16,
            "bos_token_id": 1,
            "eos_token_id": 2
        });
        std::fs::write(
            directory.path().join("config.json"),
            serde_json::to_vec(&config_json).unwrap(),
        )
        .unwrap();
        let model = HfModelBundle::open_exact(
            directory.path(),
            SafeTensorLayout::single("model.safetensors"),
        )
        .unwrap();
        let mut encoder = QwenImageCpuTextEncoder::from_model_bundle(model, config, 128).unwrap();
        encoder.emulate_bf16_compute = false;
        let tokens = fixture["token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as u32)
            .collect::<Vec<_>>();
        let vision = QwenImageVisionEmbeddings {
            values: (0..24)
                .map(|index| ((index % 11) as f32 - 5.0) * 0.02)
                .collect(),
            hidden_size: 12,
            image_token_counts: vec![2],
            grids: vec![[1, 2, 4]],
        };
        let positions = qwen_vl_position_ids(&tokens, 15, &vision.grids, 2, 128).unwrap();
        let expected_positions = fixture["position_ids"].as_array().unwrap();
        for axis in 0..3 {
            assert_eq!(
                positions
                    .iter()
                    .map(|position| position[axis])
                    .collect::<Vec<_>>(),
                expected_positions[axis]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|value| value.as_u64().unwrap() as usize)
                    .collect::<Vec<_>>()
            );
        }
        let mut checkpoints = Vec::new();
        let output = encoder
            .encode_row_with_vision_observer(
                &tokens,
                Some(&vision),
                |layer| {
                    checkpoints.push(layer);
                    Ok(())
                },
                |_, _| Ok(()),
            )
            .unwrap();
        assert_eq!(checkpoints, [0, 1, 2]);
        let expected = fixture["output"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect::<Vec<_>>();
        assert_eq!(output.len(), expected.len());
        for (index, (actual, expected)) in output.into_iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 4e-5,
                "multimodal hidden {index}: actual={actual}, expected={expected}, error={}",
                (actual - expected).abs()
            );
        }
    }

    #[test]
    #[ignore = "requires XRT_QWEN_IMAGE_BUNDLE_DIR and instrumented real Diffusers checkpoints"]
    fn real_text_layers_report_diffusers_bf16_drift() {
        const PROMPT: &str =
            "A cobalt mechanical keyboard on a walnut desk, precise product photograph.";
        let root = std::env::var_os("XRT_QWEN_IMAGE_BUNDLE_DIR")
            .map(PathBuf::from)
            .expect("XRT_QWEN_IMAGE_BUNDLE_DIR must point to the installed bundle directory");
        let bundle = ImageModelBundle::open(root).unwrap();
        let config = QwenImageBundleConfig::load(&bundle).unwrap();
        let tokenizer = QwenImagePromptTokenizer::load(
            &bundle,
            config.max_sequence_length,
            config.text_encoder.vocab_size,
        )
        .unwrap();
        let tokens = tokenizer.encode_batch(&[PROMPT]).unwrap();
        assert_eq!(tokens.input_ids[0].len(), 53);
        let encoder = QwenImageCpuTextEncoder::load_with_config(&bundle, &config).unwrap();
        let oracle_root = std::env::var_os("XRT_QWEN_IMAGE_REFERENCE_RESULT_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
                    "../../benchmark-results/image/phase0-2026-07-21/diffusers/\
                     bf16-smoke-text-checkpoints-v1",
                )
            });
        let report: serde_json::Value =
            serde_json::from_slice(&std::fs::read(oracle_root.join("result.json")).unwrap())
                .unwrap();
        assert_eq!(report["request"]["prompt"], PROMPT);

        let output = encoder
            .encode_row_with_observer(
                &tokens.input_ids[0],
                |_| Ok(()),
                |checkpoint, actual| {
                    let name = if checkpoint == config.text_encoder.num_hidden_layers {
                        "text_encoder_final_norm".to_string()
                    } else {
                        format!("text_encoder_layer_{checkpoint:02}")
                    };
                    let record = &report["tensors"][&name];
                    assert_eq!(record["shape"], serde_json::json!([1, 53, 3584]));
                    let relative = record["path"].as_str().unwrap();
                    let encoded = std::fs::read(oracle_root.join(relative)).unwrap();
                    let expected = encoded
                        .chunks_exact(2)
                        .map(|bytes| {
                            bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32()
                        })
                        .collect::<Vec<_>>();
                    let whole = test_metrics(actual, &expected);
                    let token = 34 + 16;
                    let start = token * config.text_encoder.hidden_size;
                    let end = start + config.text_encoder.hidden_size;
                    let sensitive = test_metrics(&actual[start..end], &expected[start..end]);
                    eprintln!(
                        "{name}: all(max={:.6}, nrms={:.6}, cos={:.9}) retained_token16(max={:.6}, nrms={:.6}, cos={:.9})",
                        whole.0,
                        whole.1,
                        whole.2,
                        sensitive.0,
                        sensitive.1,
                        sensitive.2,
                    );
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(output.len(), 53 * 3_584);
    }

    fn test_metrics(actual: &[f32], expected: &[f32]) -> (f32, f32, f32) {
        assert_eq!(actual.len(), expected.len());
        let mut max_abs = 0.0f32;
        let mut squared_error = 0.0f64;
        let mut expected_squared = 0.0f64;
        let mut actual_squared = 0.0f64;
        let mut dot = 0.0f64;
        for (&actual, &expected) in actual.iter().zip(expected) {
            let difference = f64::from(actual) - f64::from(expected);
            max_abs = max_abs.max((actual - expected).abs());
            squared_error += difference * difference;
            expected_squared += f64::from(expected) * f64::from(expected);
            actual_squared += f64::from(actual) * f64::from(actual);
            dot += f64::from(actual) * f64::from(expected);
        }
        (
            max_abs,
            (squared_error / expected_squared.max(f64::MIN_POSITIVE)).sqrt() as f32,
            (dot / (actual_squared.sqrt() * expected_squared.sqrt()).max(f64::MIN_POSITIVE)) as f32,
        )
    }
}
