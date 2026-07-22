use std::collections::{BTreeMap, BTreeSet};

use half::bf16;
use xrt_kernels::{linear_bf16, rms_norm_rows, scaled_dot_product_attention, silu};
use xrt_safetensors::{SafeTensorDType, SafeTensorStore};

use crate::{ImageError, ImageModelBundle};

use super::{text_encoder::text_encoder_layout, QwenImageBundleConfig, QwenImageVisionConfig};

#[derive(Debug, Clone, PartialEq)]
pub struct QwenImageVisionInput {
    /// Processor-ordered patch rows. Each row contains
    /// `[channel, temporal, patch_height, patch_width]` values.
    pub pixel_values: Vec<f32>,
    pub grids: Vec<[usize; 3]>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QwenImageVisionEmbeddings {
    pub values: Vec<f32>,
    pub hidden_size: usize,
    pub image_token_counts: Vec<usize>,
    pub grids: Vec<[usize; 3]>,
}

#[derive(Debug)]
pub struct QwenImageCpuVisionEncoder {
    store: SafeTensorStore,
    config: QwenImageVisionConfig,
    auxiliary: BTreeMap<String, Vec<f32>>,
    emulate_bf16_compute: bool,
}

impl QwenImageCpuVisionEncoder {
    pub fn load(bundle: &ImageModelBundle) -> Result<Self, ImageError> {
        let config = QwenImageBundleConfig::load(bundle)?;
        Self::load_with_config(bundle, &config)
    }

    pub fn load_with_config(
        bundle: &ImageModelBundle,
        config: &QwenImageBundleConfig,
    ) -> Result<Self, ImageError> {
        let vision = config.text_encoder.vision.clone().ok_or_else(|| {
            ImageError::MissingComponent(
                "Qwen2.5-VL text encoder config has no vision_config".to_string(),
            )
        })?;
        let (component_root, layout) = text_encoder_layout(bundle)?;
        let store = SafeTensorStore::open_exact(component_root, layout).map_err(|error| {
            ImageError::CorruptComponent(format!(
                "Qwen Image vision encoder failed exact SafeTensors validation: {error}"
            ))
        })?;
        Self::from_store(store, vision)
    }

    pub fn from_store(
        store: SafeTensorStore,
        config: QwenImageVisionConfig,
    ) -> Result<Self, ImageError> {
        config.validate(config.out_hidden_size)?;
        let expected = expected_vision_tensors(&config)?;
        let mut auxiliary = BTreeMap::new();
        for tensor in &expected {
            let info = store.tensor_info(&tensor.name).ok_or_else(|| {
                ImageError::UnsupportedTensor(format!(
                    "missing Qwen2.5-VL vision tensor `{}`",
                    tensor.name
                ))
            })?;
            if info.shape != tensor.shape {
                return Err(ImageError::UnsupportedShape(format!(
                    "vision tensor `{}` has shape {:?}, expected {:?}",
                    tensor.name, info.shape, tensor.shape
                )));
            }
            if info.dtype != SafeTensorDType::Bf16 {
                return Err(ImageError::UnsupportedTensor(format!(
                    "vision tensor `{}` is {:?}, expected BF16",
                    tensor.name, info.dtype
                )));
            }
            if tensor.shape.len() == 1 {
                let view = store.require_tensor(&tensor.name).map_err(|error| {
                    ImageError::CorruptComponent(format!(
                        "failed to map vision tensor `{}`: {error}",
                        tensor.name
                    ))
                })?;
                auxiliary.insert(tensor.name.clone(), decode_bf16(view.data, &tensor.name)?);
            }
        }
        Ok(Self {
            store,
            config,
            auxiliary,
            emulate_bf16_compute: true,
        })
    }

    pub fn config(&self) -> &QwenImageVisionConfig {
        &self.config
    }

    pub fn encode(
        &self,
        input: &QwenImageVisionInput,
    ) -> Result<QwenImageVisionEmbeddings, ImageError> {
        self.encode_with_control(input, |_| Ok(()))
    }

    /// Execute the visual tower with a checkpoint before each block and once
    /// after the merger. The callback receives `0..=depth`.
    pub fn encode_with_control<F>(
        &self,
        input: &QwenImageVisionInput,
        mut checkpoint: F,
    ) -> Result<QwenImageVisionEmbeddings, ImageError>
    where
        F: FnMut(usize) -> Result<(), ImageError>,
    {
        let geometry = VisionGeometry::new(&self.config, input)?;
        let mut states = self.linear(
            "visual.patch_embed.proj",
            &input.pixel_values,
            geometry.patch_rows,
            geometry.patch_features,
            self.config.hidden_size,
            false,
        )?;
        self.round_activation(&mut states);
        states = reorder_groups(
            &states,
            &geometry.window_index,
            geometry.merge_unit,
            self.config.hidden_size,
        )?;
        let positions = reorder_position_groups(
            &geometry.positions,
            &geometry.window_index,
            geometry.merge_unit,
        )?;

        for layer in 0..self.config.depth {
            checkpoint(layer)?;
            let segments = if self.config.fullatt_block_indexes.contains(&layer) {
                &geometry.full_segments
            } else {
                &geometry.window_segments
            };
            states = self.block(layer, &states, &positions, segments)?;
        }
        checkpoint(self.config.depth)?;

        let mut normalized = states;
        rms_norm_rows(
            &mut normalized,
            geometry.patch_rows,
            self.config.hidden_size,
            Some(self.auxiliary("visual.merger.ln_q.weight")?),
            1e-6,
        )
        .map_err(map_kernel_error)?;
        self.round_activation(&mut normalized);
        let merge_width = checked_product(
            &[self.config.hidden_size, geometry.merge_unit],
            "vision merger width",
        )?;
        let merged_rows = geometry.patch_rows / geometry.merge_unit;
        let mut merged = self.linear(
            "visual.merger.mlp.0",
            &normalized,
            merged_rows,
            merge_width,
            merge_width,
            true,
        )?;
        merged
            .iter_mut()
            .for_each(|value| *value = gelu_exact(*value));
        self.round_activation(&mut merged);
        merged = self.linear(
            "visual.merger.mlp.2",
            &merged,
            merged_rows,
            merge_width,
            self.config.out_hidden_size,
            true,
        )?;
        self.round_activation(&mut merged);
        let values =
            reverse_reorder_rows(&merged, &geometry.window_index, self.config.out_hidden_size)?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err(ImageError::Numerical {
                component: "vision_encoder",
                step: self.config.depth,
            });
        }
        Ok(QwenImageVisionEmbeddings {
            values,
            hidden_size: self.config.out_hidden_size,
            image_token_counts: geometry.image_token_counts,
            grids: input.grids.clone(),
        })
    }

    fn block(
        &self,
        layer: usize,
        input: &[f32],
        positions: &[[usize; 2]],
        segments: &[usize],
    ) -> Result<Vec<f32>, ImageError> {
        let rows = positions.len();
        let hidden = self.config.hidden_size;
        let heads = self.config.num_heads;
        let head_dim = hidden / heads;
        let prefix = format!("visual.blocks.{layer}");

        let mut normalized = input.to_vec();
        rms_norm_rows(
            &mut normalized,
            rows,
            hidden,
            Some(self.auxiliary(&format!("{prefix}.norm1.weight"))?),
            1e-6,
        )
        .map_err(map_kernel_error)?;
        self.round_activation(&mut normalized);
        let qkv_width = checked_product(&[hidden, 3], "vision QKV width")?;
        let qkv = self.linear(
            &format!("{prefix}.attn.qkv"),
            &normalized,
            rows,
            hidden,
            qkv_width,
            true,
        )?;
        let state_values = checked_product(&[rows, hidden], "vision attention states")?;
        let mut query = vec![0.0; state_values];
        let mut key = vec![0.0; state_values];
        let mut value = vec![0.0; state_values];
        for row in 0..rows {
            let source = row * qkv_width;
            let destination = row * hidden;
            query[destination..destination + hidden].copy_from_slice(&qkv[source..source + hidden]);
            key[destination..destination + hidden]
                .copy_from_slice(&qkv[source + hidden..source + 2 * hidden]);
            value[destination..destination + hidden]
                .copy_from_slice(&qkv[source + 2 * hidden..source + 3 * hidden]);
        }
        apply_vision_rope(&mut query, &mut key, positions, heads, head_dim)?;
        self.round_activation(&mut query);
        self.round_activation(&mut key);
        let mut attended = segmented_attention(&query, &key, &value, segments, heads, head_dim)?;
        self.round_activation(&mut attended);
        let attention = self.linear(
            &format!("{prefix}.attn.proj"),
            &attended,
            rows,
            hidden,
            hidden,
            true,
        )?;
        let mut states = input
            .iter()
            .zip(attention)
            .map(|(residual, update)| residual + update)
            .collect::<Vec<_>>();
        self.round_activation(&mut states);

        let residual = states.clone();
        rms_norm_rows(
            &mut states,
            rows,
            hidden,
            Some(self.auxiliary(&format!("{prefix}.norm2.weight"))?),
            1e-6,
        )
        .map_err(map_kernel_error)?;
        self.round_activation(&mut states);
        let mut gate = self.linear(
            &format!("{prefix}.mlp.gate_proj"),
            &states,
            rows,
            hidden,
            self.config.intermediate_size,
            true,
        )?;
        let up = self.linear(
            &format!("{prefix}.mlp.up_proj"),
            &states,
            rows,
            hidden,
            self.config.intermediate_size,
            true,
        )?;
        for (gate, up) in gate.iter_mut().zip(up) {
            *gate = silu(*gate) * up;
        }
        self.round_activation(&mut gate);
        let update = self.linear(
            &format!("{prefix}.mlp.down_proj"),
            &gate,
            rows,
            self.config.intermediate_size,
            hidden,
            true,
        )?;
        let mut output = residual
            .into_iter()
            .zip(update)
            .map(|(residual, update)| residual + update)
            .collect::<Vec<_>>();
        self.round_activation(&mut output);
        Ok(output)
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
        let weight = self.store.require_tensor(&name).map_err(|error| {
            ImageError::CorruptComponent(format!("failed to map `{name}`: {error}"))
        })?;
        let flattened_input = if weight.info.shape.len() >= 2 {
            checked_product(&weight.info.shape[1..], "vision matrix input width")?
        } else {
            0
        };
        if weight.info.dtype != SafeTensorDType::Bf16
            || weight.info.shape.first().copied() != Some(output_features)
            || flattened_input != input_features
        {
            return Err(ImageError::UnsupportedShape(format!(
                "vision matrix `{name}` has {:?}/{:?}, expected [{output_features}, {input_features}] BF16",
                weight.info.shape, weight.info.dtype
            )));
        }
        let bias = has_bias
            .then(|| self.auxiliary(&format!("{prefix}.bias")))
            .transpose()?;
        let mut output = vec![0.0; checked_product(&[rows, output_features], "vision linear")?];
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

    fn auxiliary(&self, name: &str) -> Result<&[f32], ImageError> {
        self.auxiliary
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| ImageError::UnsupportedTensor(format!("missing vision tensor `{name}`")))
    }

    fn round_activation(&self, values: &mut [f32]) {
        if self.emulate_bf16_compute {
            values
                .iter_mut()
                .for_each(|value| *value = bf16::from_f32(*value).to_f32());
        }
    }
}

#[derive(Debug)]
struct VisionGeometry {
    patch_rows: usize,
    patch_features: usize,
    merge_unit: usize,
    image_token_counts: Vec<usize>,
    positions: Vec<[usize; 2]>,
    window_index: Vec<usize>,
    full_segments: Vec<usize>,
    window_segments: Vec<usize>,
}

impl VisionGeometry {
    fn new(
        config: &QwenImageVisionConfig,
        input: &QwenImageVisionInput,
    ) -> Result<Self, ImageError> {
        if input.grids.is_empty() || input.pixel_values.iter().any(|value| !value.is_finite()) {
            return Err(ImageError::InvalidRequest(
                "vision input requires finite pixels and at least one image grid".to_string(),
            ));
        }
        let merge = config.spatial_merge_size;
        let merge_unit = checked_product(&[merge, merge], "vision merge unit")?;
        let patch_features = checked_product(
            &[
                config.in_channels,
                config.temporal_patch_size,
                config.patch_size,
                config.patch_size,
            ],
            "vision patch features",
        )?;
        let mut patch_rows = 0usize;
        let mut image_token_counts = Vec::with_capacity(input.grids.len());
        let mut positions = Vec::new();
        let mut full_segments = Vec::new();
        for [temporal, height, width] in input.grids.iter().copied() {
            if temporal == 0
                || height == 0
                || width == 0
                || height % merge != 0
                || width % merge != 0
            {
                return Err(ImageError::UnsupportedShape(format!(
                    "invalid vision grid [{temporal}, {height}, {width}] for merge {merge}"
                )));
            }
            let rows = checked_product(&[temporal, height, width], "vision grid rows")?;
            patch_rows = patch_rows.checked_add(rows).ok_or_else(|| {
                ImageError::UnsupportedShape("vision patch row count overflow".to_string())
            })?;
            image_token_counts.push(rows / merge_unit);
            for _t in 0..temporal {
                full_segments.push(checked_product(
                    &[height, width],
                    "vision full-attention segment",
                )?);
                for group_h in 0..height / merge {
                    for group_w in 0..width / merge {
                        for local_h in 0..merge {
                            for local_w in 0..merge {
                                positions
                                    .push([group_h * merge + local_h, group_w * merge + local_w]);
                            }
                        }
                    }
                }
            }
        }
        let expected_pixels = checked_product(&[patch_rows, patch_features], "vision pixels")?;
        if input.pixel_values.len() != expected_pixels {
            return Err(ImageError::UnsupportedShape(format!(
                "vision processor produced {} values, expected {expected_pixels}",
                input.pixel_values.len()
            )));
        }
        let (window_index, window_segments) = window_geometry(config, &input.grids)?;
        if window_index.len() != patch_rows / merge_unit {
            return Err(ImageError::Internal(
                "vision window permutation changed the merged token count".to_string(),
            ));
        }
        Ok(Self {
            patch_rows,
            patch_features,
            merge_unit,
            image_token_counts,
            positions,
            window_index,
            full_segments,
            window_segments,
        })
    }
}

fn window_geometry(
    config: &QwenImageVisionConfig,
    grids: &[[usize; 3]],
) -> Result<(Vec<usize>, Vec<usize>), ImageError> {
    let merge = config.spatial_merge_size;
    let merge_unit = checked_product(&[merge, merge], "vision merge unit")?;
    let window = config.window_size / merge / config.patch_size;
    let mut index = Vec::new();
    let mut segments = Vec::new();
    let mut base = 0usize;
    for [temporal, height, width] in grids.iter().copied() {
        let llm_height = height / merge;
        let llm_width = width / merge;
        let windows_h = llm_height.div_ceil(window);
        let windows_w = llm_width.div_ceil(window);
        for t in 0..temporal {
            for window_h in 0..windows_h {
                for window_w in 0..windows_w {
                    let before = index.len();
                    for local_h in 0..window {
                        let row = window_h * window + local_h;
                        if row >= llm_height {
                            continue;
                        }
                        for local_w in 0..window {
                            let column = window_w * window + local_w;
                            if column < llm_width {
                                let row_offset = checked_product(
                                    &[t, llm_height],
                                    "vision temporal row offset",
                                )?
                                .checked_add(row)
                                .ok_or_else(|| {
                                    ImageError::UnsupportedShape(
                                        "vision row offset overflow".to_string(),
                                    )
                                })?;
                                let image_offset = checked_product(
                                    &[row_offset, llm_width],
                                    "vision image row offset",
                                )?
                                .checked_add(column)
                                .ok_or_else(|| {
                                    ImageError::UnsupportedShape(
                                        "vision image offset overflow".to_string(),
                                    )
                                })?;
                                index.push(base.checked_add(image_offset).ok_or_else(|| {
                                    ImageError::UnsupportedShape(
                                        "vision window index overflow".to_string(),
                                    )
                                })?);
                            }
                        }
                    }
                    let groups = index.len() - before;
                    if groups > 0 {
                        segments.push(checked_product(
                            &[groups, merge_unit],
                            "vision window segment",
                        )?);
                    }
                }
            }
        }
        base = base
            .checked_add(checked_product(
                &[temporal, llm_height, llm_width],
                "vision merged grid",
            )?)
            .ok_or_else(|| {
                ImageError::UnsupportedShape("vision window index overflow".to_string())
            })?;
    }
    let unique = index.iter().copied().collect::<BTreeSet<_>>();
    if unique.len() != index.len() || unique.iter().copied().ne(0..index.len()) {
        return Err(ImageError::Internal(
            "vision window indices are not a complete permutation".to_string(),
        ));
    }
    Ok((index, segments))
}

fn reorder_groups(
    input: &[f32],
    order: &[usize],
    group_size: usize,
    hidden: usize,
) -> Result<Vec<f32>, ImageError> {
    let group_width = checked_product(&[group_size, hidden], "vision reorder group")?;
    if input.len() != checked_product(&[order.len(), group_width], "vision reorder input")? {
        return Err(ImageError::UnsupportedShape(
            "vision reorder input geometry mismatch".to_string(),
        ));
    }
    let mut output = vec![0.0; input.len()];
    for (destination_group, source_group) in order.iter().copied().enumerate() {
        output[destination_group * group_width..(destination_group + 1) * group_width]
            .copy_from_slice(&input[source_group * group_width..(source_group + 1) * group_width]);
    }
    Ok(output)
}

fn reorder_position_groups(
    input: &[[usize; 2]],
    order: &[usize],
    group_size: usize,
) -> Result<Vec<[usize; 2]>, ImageError> {
    if input.len() != checked_product(&[order.len(), group_size], "vision position reorder input")?
    {
        return Err(ImageError::UnsupportedShape(
            "vision position reorder geometry mismatch".to_string(),
        ));
    }
    let mut output = vec![[0usize; 2]; input.len()];
    for (destination_group, source_group) in order.iter().copied().enumerate() {
        output[destination_group * group_size..(destination_group + 1) * group_size]
            .copy_from_slice(&input[source_group * group_size..(source_group + 1) * group_size]);
    }
    Ok(output)
}

fn reverse_reorder_rows(
    input: &[f32],
    window_index: &[usize],
    hidden: usize,
) -> Result<Vec<f32>, ImageError> {
    if input.len()
        != checked_product(
            &[window_index.len(), hidden],
            "vision merger reverse-order input",
        )?
    {
        return Err(ImageError::UnsupportedShape(
            "vision merger reverse-order geometry mismatch".to_string(),
        ));
    }
    let mut output = vec![0.0; input.len()];
    for (window_row, original_row) in window_index.iter().copied().enumerate() {
        output[original_row * hidden..(original_row + 1) * hidden]
            .copy_from_slice(&input[window_row * hidden..(window_row + 1) * hidden]);
    }
    Ok(output)
}

fn apply_vision_rope(
    query: &mut [f32],
    key: &mut [f32],
    positions: &[[usize; 2]],
    heads: usize,
    head_dim: usize,
) -> Result<(), ImageError> {
    let expected_values =
        checked_product(&[positions.len(), heads, head_dim], "vision RoPE values")?;
    if head_dim % 4 != 0 || query.len() != expected_values || key.len() != query.len() {
        return Err(ImageError::UnsupportedShape(
            "vision RoPE geometry mismatch".to_string(),
        ));
    }
    let half = head_dim / 2;
    let axis_width = half / 2;
    for (row, [height, width]) in positions.iter().copied().enumerate() {
        for head in 0..heads {
            let base = (row * heads + head) * head_dim;
            for pair in 0..half {
                let (position, frequency) = if pair < axis_width {
                    (height, pair)
                } else {
                    (width, pair - axis_width)
                };
                let exponent = -((2 * frequency) as f32) / half as f32;
                let angle = position as f32 * 10_000f32.powf(exponent);
                let (sin, cos) = angle.sin_cos();
                for values in [&mut *query, &mut *key] {
                    let left = values[base + pair];
                    let right = values[base + pair + half];
                    values[base + pair] = left * cos - right * sin;
                    values[base + pair + half] = left * sin + right * cos;
                }
            }
        }
    }
    Ok(())
}

fn segmented_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    segments: &[usize],
    heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>, ImageError> {
    let hidden = checked_product(&[heads, head_dim], "vision attention hidden size")?;
    let rows = segments.iter().try_fold(0usize, |total, length| {
        total.checked_add(*length).ok_or_else(|| {
            ImageError::UnsupportedShape("vision attention segment overflow".to_string())
        })
    })?;
    if segments.is_empty()
        || segments.contains(&0)
        || query.len() != checked_product(&[rows, hidden], "vision attention values")?
        || key.len() != query.len()
        || value.len() != query.len()
    {
        return Err(ImageError::UnsupportedShape(
            "vision segmented-attention geometry mismatch".to_string(),
        ));
    }
    let mut output = vec![0.0; query.len()];
    let mut row = 0usize;
    for length in segments.iter().copied() {
        let start = row * hidden;
        let end = (row + length) * hidden;
        scaled_dot_product_attention(
            &query[start..end],
            &key[start..end],
            &value[start..end],
            1,
            length,
            length,
            heads,
            head_dim,
            None,
            &mut output[start..end],
        )
        .map_err(map_kernel_error)?;
        row += length;
    }
    Ok(output)
}

#[derive(Debug)]
struct ExpectedVisionTensor {
    name: String,
    shape: Vec<usize>,
}

fn expected_vision_tensors(
    config: &QwenImageVisionConfig,
) -> Result<Vec<ExpectedVisionTensor>, ImageError> {
    config.validate(config.out_hidden_size)?;
    let patch_features = checked_product(
        &[
            config.in_channels,
            config.temporal_patch_size,
            config.patch_size,
            config.patch_size,
        ],
        "vision patch features",
    )?;
    let mut tensors = vec![ExpectedVisionTensor {
        name: "visual.patch_embed.proj.weight".to_string(),
        shape: vec![
            config.hidden_size,
            config.in_channels,
            config.temporal_patch_size,
            config.patch_size,
            config.patch_size,
        ],
    }];
    for layer in 0..config.depth {
        let prefix = format!("visual.blocks.{layer}");
        for normalization in ["norm1", "norm2"] {
            tensors.push(ExpectedVisionTensor {
                name: format!("{prefix}.{normalization}.weight"),
                shape: vec![config.hidden_size],
            });
        }
        let qkv_width = checked_product(&[config.hidden_size, 3], "vision QKV width")?;
        for (projection, output, input, bias) in [
            ("attn.qkv", qkv_width, config.hidden_size, true),
            ("attn.proj", config.hidden_size, config.hidden_size, true),
            (
                "mlp.gate_proj",
                config.intermediate_size,
                config.hidden_size,
                true,
            ),
            (
                "mlp.up_proj",
                config.intermediate_size,
                config.hidden_size,
                true,
            ),
            (
                "mlp.down_proj",
                config.hidden_size,
                config.intermediate_size,
                true,
            ),
        ] {
            tensors.push(ExpectedVisionTensor {
                name: format!("{prefix}.{projection}.weight"),
                shape: vec![output, input],
            });
            if bias {
                tensors.push(ExpectedVisionTensor {
                    name: format!("{prefix}.{projection}.bias"),
                    shape: vec![output],
                });
            }
        }
    }
    let merge_width = checked_product(
        &[
            config.hidden_size,
            config.spatial_merge_size,
            config.spatial_merge_size,
        ],
        "vision merger width",
    )?;
    tensors.extend([
        ExpectedVisionTensor {
            name: "visual.merger.ln_q.weight".to_string(),
            shape: vec![config.hidden_size],
        },
        ExpectedVisionTensor {
            name: "visual.merger.mlp.0.weight".to_string(),
            shape: vec![merge_width, merge_width],
        },
        ExpectedVisionTensor {
            name: "visual.merger.mlp.0.bias".to_string(),
            shape: vec![merge_width],
        },
        ExpectedVisionTensor {
            name: "visual.merger.mlp.2.weight".to_string(),
            shape: vec![config.out_hidden_size, merge_width],
        },
        ExpectedVisionTensor {
            name: "visual.merger.mlp.2.bias".to_string(),
            shape: vec![config.out_hidden_size],
        },
    ]);
    let mut names = BTreeSet::new();
    if tensors.iter().any(|tensor| !names.insert(&tensor.name)) {
        return Err(ImageError::Internal(
            "duplicate vision tensor schema entry".to_string(),
        ));
    }
    debug_assert_eq!(
        patch_features,
        checked_product(
            &[
                config.in_channels,
                config.temporal_patch_size,
                config.patch_size,
                config.patch_size,
            ],
            "vision patch features",
        )?
    );
    Ok(tensors)
}

fn decode_bf16(bytes: &[u8], name: &str) -> Result<Vec<f32>, ImageError> {
    if bytes.len() % 2 != 0 {
        return Err(ImageError::CorruptComponent(format!(
            "BF16 vision tensor `{name}` has odd byte length {}",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|encoded| bf16::from_bits(u16::from_le_bytes([encoded[0], encoded[1]])).to_f32())
        .collect())
}

fn gelu_exact(value: f32) -> f32 {
    value * 0.5 * (1.0 + erf_approx(value * std::f32::consts::FRAC_1_SQRT_2))
}

// Abramowitz-Stegun 7.1.26; max absolute error is approximately 1.5e-7.
fn erf_approx(value: f32) -> f32 {
    let sign = if value < 0.0 { -1.0 } else { 1.0 };
    let x = value.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let polynomial = (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_72)
        * t
        + 0.254_829_6)
        * t;
    sign * (1.0 - polynomial * (-x * x).exp())
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
    use xrt_safetensors::SafeTensorLayout;

    fn tiny_config() -> QwenImageVisionConfig {
        QwenImageVisionConfig {
            depth: 2,
            fullatt_block_indexes: vec![1],
            hidden_act: "silu".to_string(),
            hidden_size: 8,
            in_channels: 3,
            intermediate_size: 16,
            num_heads: 2,
            out_hidden_size: 6,
            patch_size: 2,
            spatial_merge_size: 2,
            temporal_patch_size: 2,
            window_size: 8,
        }
    }

    #[test]
    fn vision_schema_matches_qwen_parameter_families() {
        let tensors = expected_vision_tensors(&tiny_config()).unwrap();
        assert_eq!(tensors.len(), 30);
        assert!(tensors.iter().any(|tensor| {
            tensor.name == "visual.patch_embed.proj.weight" && tensor.shape == [8, 3, 2, 2, 2]
        }));
        assert!(tensors.iter().any(|tensor| {
            tensor.name == "visual.merger.mlp.2.weight" && tensor.shape == [6, 32]
        }));
    }

    #[test]
    fn window_geometry_is_a_complete_multi_image_permutation() {
        let (index, segments) = window_geometry(&tiny_config(), &[[1, 4, 8], [1, 8, 4]]).unwrap();
        assert_eq!(index.len(), 16);
        assert_eq!(segments.iter().sum::<usize>(), 64);
        let unique = index.iter().copied().collect::<BTreeSet<_>>();
        assert_eq!(unique, (0..16).collect());
    }

    #[test]
    fn exact_gelu_tracks_known_values() {
        assert!((gelu_exact(0.0) - 0.0).abs() < 1e-7);
        assert!((gelu_exact(1.0) - 0.841_344_7).abs() < 3e-7);
        assert!((gelu_exact(-1.0) + 0.158_655_26).abs() < 3e-7);
    }

    #[test]
    fn tiny_cpu_vision_encoder_matches_pinned_transformers_fixture() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../../tests/fixtures/qwen-image/vision-encoder-transformers-5.14.json"
        ))
        .unwrap();
        let config = tiny_config();
        let mut specifications = expected_vision_tensors(&config).unwrap();
        specifications.sort_by(|left, right| left.name.cmp(&right.name));
        let encoded = specifications
            .into_iter()
            .enumerate()
            .map(|(parameter_index, tensor)| {
                let length = tensor.shape.iter().product::<usize>();
                let bytes = (0..length)
                    .flat_map(|flat_index| {
                        let value = ((flat_index % 23) as f32 - 11.0) * 0.003
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
        let store = SafeTensorStore::open_exact(
            directory.path(),
            SafeTensorLayout::single("model.safetensors"),
        )
        .unwrap();
        let mut encoder = QwenImageCpuVisionEncoder::from_store(store, config).unwrap();
        // The tiny oracle executes F32 operations with BF16-rounded weights.
        // Production BF16 admission is covered separately by real-bundle tests.
        encoder.emulate_bf16_compute = false;
        let grids = fixture["grids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|grid| {
                let values = grid.as_array().unwrap();
                [
                    values[0].as_u64().unwrap() as usize,
                    values[1].as_u64().unwrap() as usize,
                    values[2].as_u64().unwrap() as usize,
                ]
            })
            .collect::<Vec<_>>();
        let pixel_values = (0..64 * 24)
            .map(|flat_index| ((flat_index % 29) as f32 - 14.0) * 0.01)
            .collect::<Vec<_>>();
        let mut checkpoints = Vec::new();
        let output = encoder
            .encode_with_control(
                &QwenImageVisionInput {
                    pixel_values,
                    grids,
                },
                |layer| {
                    checkpoints.push(layer);
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(checkpoints, [0, 1, 2]);
        assert_eq!(output.hidden_size, 6);
        assert_eq!(output.image_token_counts, [8, 8]);
        let expected = fixture["output"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect::<Vec<_>>();
        assert_eq!(output.values.len(), expected.len());
        for (index, (actual, expected)) in output.values.into_iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 4e-5,
                "vision hidden {index}: actual={actual}, expected={expected}, error={}",
                (actual - expected).abs()
            );
        }
    }
}
