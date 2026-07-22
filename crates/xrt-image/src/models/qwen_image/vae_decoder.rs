use std::collections::BTreeMap;

use half::{bf16, f16};
use xrt_core::XrtError;
use xrt_kernels::{
    causal_conv3d_ncthw, conv2d_ncthw, nearest_2x_ncthw, scaled_dot_product_attention,
    silu_inplace, vae_rms_norm_channels_ncthw,
};

use crate::ImageError;
use xrt_safetensors::{SafeTensorDType, SafeTensorStore};

use super::{expected_vae_tensors, validate_vae_safetensors, QwenImageVaeConfig};

/// Owned F32 tensors used by the scalar Qwen Image VAE reference executor.
/// Production SafeTensors/CUDA executors retain encoded storage and share the
/// same graph order without requiring this expanded representation.
pub type QwenImageVaeF32Weights = BTreeMap<String, Vec<f32>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QwenImageVaeTiling {
    pub tile_latent_height: usize,
    pub tile_latent_width: usize,
    pub stride_latent_height: usize,
    pub stride_latent_width: usize,
}

impl Default for QwenImageVaeTiling {
    fn default() -> Self {
        // Diffusers defaults: 256px tiles with a 192px stride at 8x VAE scale.
        Self {
            tile_latent_height: 32,
            tile_latent_width: 32,
            stride_latent_height: 24,
            stride_latent_width: 24,
        }
    }
}

/// Materialize only the decoder-side tensors from a validated BF16/F16 VAE
/// component. This is the portable CPU reference path; optimized backends may
/// retain encoded weights or upload them directly to the accelerator.
pub fn load_vae_decoder_f32_weights(
    store: &SafeTensorStore,
    config: &QwenImageVaeConfig,
) -> Result<QwenImageVaeF32Weights, ImageError> {
    validate_vae_safetensors(store, config)?;
    let mut output = BTreeMap::new();
    for tensor in expected_vae_tensors(config)?.into_iter().filter(|tensor| {
        tensor.name.starts_with("decoder.") || tensor.name.starts_with("post_quant_conv.")
    }) {
        let view = store.require_tensor(&tensor.name).map_err(|error| {
            ImageError::CorruptComponent(format!(
                "failed to map VAE tensor `{}`: {error}",
                tensor.name
            ))
        })?;
        let values = decode_float_tensor(view.info.dtype.clone(), view.data, &tensor.name)?;
        let expected = tensor.shape.iter().try_fold(1usize, |length, dimension| {
            length.checked_mul(*dimension).ok_or_else(|| {
                ImageError::UnsupportedShape(format!(
                    "VAE tensor `{}` element count overflows",
                    tensor.name
                ))
            })
        })?;
        if values.len() != expected {
            return Err(ImageError::CorruptComponent(format!(
                "VAE tensor `{}` decoded to {} values, expected {expected}",
                tensor.name,
                values.len()
            )));
        }
        output.insert(tensor.name, values);
    }
    Ok(output)
}

/// Materialize the encoder and posterior-projection tensors required by the
/// Edit-2511 reconstruction-conditioning path. The complete VAE store is
/// validated before any tensor is admitted.
pub fn load_vae_encoder_f32_weights(
    store: &SafeTensorStore,
    config: &QwenImageVaeConfig,
) -> Result<QwenImageVaeF32Weights, ImageError> {
    validate_vae_safetensors(store, config)?;
    let mut output = BTreeMap::new();
    for tensor in expected_vae_tensors(config)?.into_iter().filter(|tensor| {
        tensor.name.starts_with("encoder.") || tensor.name.starts_with("quant_conv.")
    }) {
        let view = store.require_tensor(&tensor.name).map_err(|error| {
            ImageError::CorruptComponent(format!(
                "failed to map VAE tensor `{}`: {error}",
                tensor.name
            ))
        })?;
        let values = decode_float_tensor(view.info.dtype.clone(), view.data, &tensor.name)?;
        let expected = tensor.shape.iter().try_fold(1usize, |length, dimension| {
            length.checked_mul(*dimension).ok_or_else(|| {
                ImageError::UnsupportedShape(format!(
                    "VAE tensor `{}` element count overflows",
                    tensor.name
                ))
            })
        })?;
        if values.len() != expected {
            return Err(ImageError::CorruptComponent(format!(
                "VAE tensor `{}` decoded to {} values, expected {expected}",
                tensor.name,
                values.len()
            )));
        }
        output.insert(tensor.name, values);
    }
    Ok(output)
}

fn decode_float_tensor(
    dtype: SafeTensorDType,
    bytes: &[u8],
    name: &str,
) -> Result<Vec<f32>, ImageError> {
    let width = match dtype {
        SafeTensorDType::Bf16 | SafeTensorDType::F16 => 2,
        SafeTensorDType::F32 => 4,
        other => {
            return Err(ImageError::UnsupportedTensor(format!(
                "VAE tensor `{name}` has unsupported execution dtype {other:?}"
            )))
        }
    };
    if bytes.len() % width != 0 {
        return Err(ImageError::CorruptComponent(format!(
            "VAE tensor `{name}` byte length {} is not divisible by dtype width {width}",
            bytes.len()
        )));
    }
    Ok(match dtype {
        SafeTensorDType::Bf16 => bytes
            .chunks_exact(2)
            .map(|encoded| bf16::from_bits(u16::from_le_bytes([encoded[0], encoded[1]])).to_f32())
            .collect(),
        SafeTensorDType::F16 => bytes
            .chunks_exact(2)
            .map(|encoded| f16::from_bits(u16::from_le_bytes([encoded[0], encoded[1]])).to_f32())
            .collect(),
        SafeTensorDType::F32 => bytes
            .chunks_exact(4)
            .map(|encoded| f32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]))
            .collect(),
        _ => unreachable!("dtype was rejected above"),
    })
}

/// Encode normalized source pixels through the Qwen Image VAE and return the
/// posterior mode used by the official Edit Plus pipeline. Input is NCTHW in
/// `[-1, 1]`; still-image execution requires temporal depth one.
#[allow(clippy::too_many_arguments)]
pub fn qwen_image_vae_encode_f32(
    config: &QwenImageVaeConfig,
    weights: &QwenImageVaeF32Weights,
    pixels: &[f32],
    batch: usize,
    depth: usize,
    height: usize,
    width: usize,
) -> Result<Vec<f32>, ImageError> {
    encode_f32_impl(
        config,
        weights,
        pixels,
        batch,
        depth,
        height,
        width,
        |_, _| Ok(()),
    )
}

/// Source VAE encode with a cooperative checkpoint after each graph stage.
#[allow(clippy::too_many_arguments)]
pub fn qwen_image_vae_encode_f32_with_control<F>(
    config: &QwenImageVaeConfig,
    weights: &QwenImageVaeF32Weights,
    pixels: &[f32],
    batch: usize,
    depth: usize,
    height: usize,
    width: usize,
    mut checkpoint: F,
) -> Result<Vec<f32>, ImageError>
where
    F: FnMut(usize) -> Result<(), ImageError>,
{
    let mut stage = 0usize;
    let encoded = encode_f32_impl(
        config,
        weights,
        pixels,
        batch,
        depth,
        height,
        width,
        |_, _| {
            checkpoint(stage)?;
            stage = stage.checked_add(1).ok_or_else(|| {
                ImageError::UnsupportedShape("VAE encoder stage count overflow".to_string())
            })?;
            Ok(())
        },
    )?;
    checkpoint(stage)?;
    Ok(encoded)
}

#[allow(clippy::too_many_arguments)]
fn encode_f32_impl<F>(
    config: &QwenImageVaeConfig,
    weights: &QwenImageVaeF32Weights,
    pixels: &[f32],
    batch: usize,
    depth: usize,
    height: usize,
    width: usize,
    mut checkpoint: F,
) -> Result<Vec<f32>, ImageError>
where
    F: FnMut(&str, &[f32]) -> Result<(), ImageError>,
{
    config.validate()?;
    if depth != 1 {
        return Err(ImageError::UnsupportedShape(format!(
            "still-image VAE encode requires temporal depth 1, found {depth}"
        )));
    }
    let scale_factor = config.scale_factor()?;
    let expected_pixels = checked_product(
        &[batch, config.input_channels, depth, height, width],
        "VAE source input",
    )?;
    if batch == 0
        || height == 0
        || width == 0
        || height % scale_factor != 0
        || width % scale_factor != 0
        || pixels.len() != expected_pixels
        || pixels.iter().any(|value| !value.is_finite())
    {
        return Err(ImageError::UnsupportedShape(format!(
            "invalid VAE source geometry or values: length={}, expected={expected_pixels}, shape=[{batch}, {}, {depth}, {height}, {width}], scale={scale_factor}",
            pixels.len(), config.input_channels
        )));
    }

    let channels = config
        .dim_mult
        .iter()
        .map(|multiplier| checked_mul(config.base_dim, *multiplier, "VAE encoder channels"))
        .collect::<Result<Vec<_>, _>>()?;
    let last = *channels.last().ok_or_else(|| {
        ImageError::UnsupportedShape("Qwen Image VAE has no encoder stages".to_string())
    })?;
    let mut shape = [depth, height, width];
    let mut current = conv3d(
        weights,
        "encoder.conv_in",
        pixels,
        batch,
        config.input_channels,
        config.base_dim,
        shape,
        [3, 3, 3],
        [1, 1, 1],
        [1, 1, 1],
    )?;
    checkpoint("encoder_conv_in", &current)?;

    let mut current_channels = config.base_dim;
    let mut down_index = 0usize;
    let mut scale = 1.0f32;
    for (stage, output_channels) in channels.iter().copied().enumerate() {
        for _ in 0..config.num_res_blocks {
            current = residual_block(
                weights,
                &format!("encoder.down_blocks.{down_index}"),
                &current,
                batch,
                current_channels,
                output_channels,
                shape,
            )?;
            current_channels = output_channels;
            checkpoint(&format!("encoder_down_block_{down_index}"), &current)?;
            down_index += 1;
            if config
                .attn_scales
                .iter()
                .any(|candidate| (*candidate - scale).abs() <= f32::EPSILON)
            {
                current = attention_block(
                    weights,
                    &format!("encoder.down_blocks.{down_index}"),
                    &current,
                    batch,
                    current_channels,
                    shape,
                )?;
                checkpoint(&format!("encoder_down_block_{down_index}"), &current)?;
                down_index += 1;
            }
        }
        if stage + 1 != channels.len() {
            current = conv2d(
                weights,
                &format!("encoder.down_blocks.{down_index}.resample.1"),
                &current,
                batch,
                current_channels,
                current_channels,
                shape,
                [3, 3],
                [2, 2],
                [0, 1, 0, 1],
            )?;
            shape[1] /= 2;
            shape[2] /= 2;
            checkpoint(&format!("encoder_down_block_{down_index}"), &current)?;
            down_index += 1;
            scale *= 0.5;
        }
    }
    if current_channels != last {
        return Err(ImageError::UnsupportedShape(format!(
            "VAE encoder tail received {current_channels} channels, expected {last}"
        )));
    }
    current = mid_block(weights, "encoder.mid_block", &current, batch, last, shape)?;
    checkpoint("encoder_mid_block", &current)?;
    vae_rms_norm_channels_ncthw(
        &mut current,
        batch,
        last,
        shape[0],
        shape[1],
        shape[2],
        require(weights, "encoder.norm_out.gamma")?,
        1e-12,
    )
    .map_err(map_kernel_error)?;
    silu_inplace(&mut current);
    current = conv3d(
        weights,
        "encoder.conv_out",
        &current,
        batch,
        last,
        config.z_dim * 2,
        shape,
        [3, 3, 3],
        [1, 1, 1],
        [1, 1, 1],
    )?;
    checkpoint("encoder_conv_out", &current)?;
    current = conv3d(
        weights,
        "quant_conv",
        &current,
        batch,
        config.z_dim * 2,
        config.z_dim * 2,
        shape,
        [1, 1, 1],
        [1, 1, 1],
        [0, 0, 0],
    )?;
    checkpoint("quant_conv", &current)?;

    let latent_plane = checked_product(&[shape[0], shape[1], shape[2]], "VAE latent plane")?;
    let mut mode =
        vec![0.0; checked_product(&[batch, config.z_dim, latent_plane], "VAE posterior mode")?];
    let posterior_channels = config.z_dim * 2;
    for batch_index in 0..batch {
        let source = batch_index * posterior_channels * latent_plane;
        let destination = batch_index * config.z_dim * latent_plane;
        let length = config.z_dim * latent_plane;
        mode[destination..destination + length].copy_from_slice(&current[source..source + length]);
    }
    if mode.iter().any(|value| !value.is_finite()) {
        return Err(ImageError::Numerical {
            component: "vae_encoder",
            step: 0,
        });
    }
    Ok(mode)
}

/// Decode one or more still-image latent tensors through the Qwen Image VAE
/// reference graph. Input and output use NCTHW layout with temporal depth one.
#[allow(clippy::too_many_arguments)]
pub fn qwen_image_vae_decode_f32(
    config: &QwenImageVaeConfig,
    weights: &QwenImageVaeF32Weights,
    latents: &[f32],
    batch: usize,
    depth: usize,
    latent_height: usize,
    latent_width: usize,
) -> Result<Vec<f32>, ImageError> {
    decode_f32_impl(
        config,
        weights,
        latents,
        batch,
        depth,
        latent_height,
        latent_width,
        |_, _| {},
    )
}

/// Bounded-memory overlap-blended VAE decode. At most the current and prior
/// decoded tile rows are retained, rather than a full grid of decoded tiles.
#[allow(clippy::too_many_arguments)]
pub fn qwen_image_vae_decode_tiled_f32(
    config: &QwenImageVaeConfig,
    weights: &QwenImageVaeF32Weights,
    latents: &[f32],
    batch: usize,
    depth: usize,
    latent_height: usize,
    latent_width: usize,
    tiling: QwenImageVaeTiling,
) -> Result<Vec<f32>, ImageError> {
    qwen_image_vae_decode_tiled_f32_with_control(
        config,
        weights,
        latents,
        batch,
        depth,
        latent_height,
        latent_width,
        tiling,
        |_| Ok(()),
    )
}

/// Tiled decode with a cooperative checkpoint before each tile and once after
/// the final tile. The callback receives the next tile's zero-based index.
#[allow(clippy::too_many_arguments)]
pub fn qwen_image_vae_decode_tiled_f32_with_control<F>(
    config: &QwenImageVaeConfig,
    weights: &QwenImageVaeF32Weights,
    latents: &[f32],
    batch: usize,
    depth: usize,
    latent_height: usize,
    latent_width: usize,
    tiling: QwenImageVaeTiling,
    mut checkpoint: F,
) -> Result<Vec<f32>, ImageError>
where
    F: FnMut(usize) -> Result<(), ImageError>,
{
    validate_tiling(tiling)?;
    if latent_height <= tiling.tile_latent_height && latent_width <= tiling.tile_latent_width {
        checkpoint(0)?;
        let decoded = qwen_image_vae_decode_f32(
            config,
            weights,
            latents,
            batch,
            depth,
            latent_height,
            latent_width,
        )?;
        checkpoint(1)?;
        return Ok(decoded);
    }
    let expected = checked_product(
        &[batch, config.z_dim, depth, latent_height, latent_width],
        "tiled VAE latent input",
    )?;
    if latents.len() != expected {
        return Err(ImageError::UnsupportedShape(format!(
            "tiled VAE latent length {} does not match expected {expected}",
            latents.len()
        )));
    }
    let scale = config.scale_factor()?;
    let output_height = checked_mul(latent_height, scale, "tiled VAE output height")?;
    let output_width = checked_mul(latent_width, scale, "tiled VAE output width")?;
    let mut output = vec![
        0.0;
        checked_product(
            &[
                batch,
                config.input_channels,
                depth,
                output_height,
                output_width
            ],
            "tiled VAE output",
        )?
    ];
    let blend_height = checked_mul(
        tiling.tile_latent_height - tiling.stride_latent_height,
        scale,
        "VAE vertical blend",
    )?;
    let blend_width = checked_mul(
        tiling.tile_latent_width - tiling.stride_latent_width,
        scale,
        "VAE horizontal blend",
    )?;
    let copy_height = checked_mul(
        tiling.stride_latent_height,
        scale,
        "VAE tile output stride height",
    )?;
    let copy_width = checked_mul(
        tiling.stride_latent_width,
        scale,
        "VAE tile output stride width",
    )?;
    let mut previous_row: Option<Vec<DecodedTile>> = None;
    let mut tile_index = 0usize;
    for latent_y in (0..latent_height).step_by(tiling.stride_latent_height) {
        let mut current_row = Vec::new();
        for latent_x in (0..latent_width).step_by(tiling.stride_latent_width) {
            checkpoint(tile_index)?;
            let tile_height = tiling.tile_latent_height.min(latent_height - latent_y);
            let tile_width = tiling.tile_latent_width.min(latent_width - latent_x);
            let tile_latents = extract_tile_ncthw(
                latents,
                batch,
                config.z_dim,
                depth,
                latent_height,
                latent_width,
                latent_y,
                latent_x,
                tile_height,
                tile_width,
            )?;
            let decoded = qwen_image_vae_decode_f32(
                config,
                weights,
                &tile_latents,
                batch,
                depth,
                tile_height,
                tile_width,
            )?;
            let mut tile = DecodedTile {
                values: decoded,
                height: checked_mul(tile_height, scale, "decoded tile height")?,
                width: checked_mul(tile_width, scale, "decoded tile width")?,
            };
            if let Some(above) = previous_row
                .as_ref()
                .and_then(|row| row.get(current_row.len()))
            {
                blend_vertical(
                    above,
                    &mut tile,
                    batch,
                    config.input_channels,
                    depth,
                    blend_height,
                );
            }
            if let Some(left) = current_row.last() {
                blend_horizontal(
                    left,
                    &mut tile,
                    batch,
                    config.input_channels,
                    depth,
                    blend_width,
                );
            }
            current_row.push(tile);
            tile_index = tile_index.checked_add(1).ok_or_else(|| {
                ImageError::UnsupportedShape("VAE tile count overflow".to_string())
            })?;
        }
        write_tile_row(
            &current_row,
            &mut output,
            batch,
            config.input_channels,
            depth,
            output_height,
            output_width,
            checked_mul(latent_y, scale, "tile row output offset")?,
            copy_height,
            copy_width,
        );
        previous_row = Some(current_row);
    }
    checkpoint(tile_index)?;
    Ok(output)
}

#[derive(Debug)]
struct DecodedTile {
    values: Vec<f32>,
    height: usize,
    width: usize,
}

fn validate_tiling(tiling: QwenImageVaeTiling) -> Result<(), ImageError> {
    if tiling.tile_latent_height == 0
        || tiling.tile_latent_width == 0
        || tiling.stride_latent_height == 0
        || tiling.stride_latent_width == 0
        || tiling.stride_latent_height > tiling.tile_latent_height
        || tiling.stride_latent_width > tiling.tile_latent_width
    {
        return Err(ImageError::InvalidRequest(format!(
            "invalid VAE tiling geometry: {tiling:?}"
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn extract_tile_ncthw(
    input: &[f32],
    batch: usize,
    channels: usize,
    depth: usize,
    height: usize,
    width: usize,
    offset_y: usize,
    offset_x: usize,
    tile_height: usize,
    tile_width: usize,
) -> Result<Vec<f32>, ImageError> {
    if offset_y + tile_height > height || offset_x + tile_width > width {
        return Err(ImageError::UnsupportedShape(
            "VAE tile exceeds latent bounds".to_string(),
        ));
    }
    let mut output = vec![
        0.0;
        checked_product(
            &[batch, channels, depth, tile_height, tile_width],
            "VAE latent tile",
        )?
    ];
    let input_plane = depth * height * width;
    let output_plane = depth * tile_height * tile_width;
    for batch_channel in 0..batch * channels {
        for temporal in 0..depth {
            for y in 0..tile_height {
                let source = batch_channel * input_plane
                    + temporal * height * width
                    + (offset_y + y) * width
                    + offset_x;
                let destination = batch_channel * output_plane
                    + temporal * tile_height * tile_width
                    + y * tile_width;
                output[destination..destination + tile_width]
                    .copy_from_slice(&input[source..source + tile_width]);
            }
        }
    }
    Ok(output)
}

fn blend_vertical(
    above: &DecodedTile,
    current: &mut DecodedTile,
    batch: usize,
    channels: usize,
    depth: usize,
    requested_extent: usize,
) {
    let extent = requested_extent.min(above.height).min(current.height);
    if extent == 0 {
        return;
    }
    let width = above.width.min(current.width);
    for batch_channel in 0..batch * channels {
        for temporal in 0..depth {
            for y in 0..extent {
                let alpha = y as f32 / extent as f32;
                for x in 0..width {
                    let above_index =
                        ((batch_channel * depth + temporal) * above.height + above.height - extent
                            + y)
                            * above.width
                            + x;
                    let current_index = ((batch_channel * depth + temporal) * current.height + y)
                        * current.width
                        + x;
                    current.values[current_index] = above.values[above_index] * (1.0 - alpha)
                        + current.values[current_index] * alpha;
                }
            }
        }
    }
}

fn blend_horizontal(
    left: &DecodedTile,
    current: &mut DecodedTile,
    batch: usize,
    channels: usize,
    depth: usize,
    requested_extent: usize,
) {
    let extent = requested_extent.min(left.width).min(current.width);
    if extent == 0 {
        return;
    }
    let height = left.height.min(current.height);
    for batch_channel in 0..batch * channels {
        for temporal in 0..depth {
            for y in 0..height {
                for x in 0..extent {
                    let alpha = x as f32 / extent as f32;
                    let left_index = ((batch_channel * depth + temporal) * left.height + y)
                        * left.width
                        + left.width
                        - extent
                        + x;
                    let current_index = ((batch_channel * depth + temporal) * current.height + y)
                        * current.width
                        + x;
                    current.values[current_index] = left.values[left_index] * (1.0 - alpha)
                        + current.values[current_index] * alpha;
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn write_tile_row(
    row: &[DecodedTile],
    output: &mut [f32],
    batch: usize,
    channels: usize,
    depth: usize,
    output_height: usize,
    output_width: usize,
    output_y: usize,
    copy_height: usize,
    copy_width: usize,
) {
    let output_plane = depth * output_height * output_width;
    for (column, tile) in row.iter().enumerate() {
        let output_x = column * copy_width;
        let height = copy_height
            .min(tile.height)
            .min(output_height.saturating_sub(output_y));
        let width = copy_width
            .min(tile.width)
            .min(output_width.saturating_sub(output_x));
        for batch_channel in 0..batch * channels {
            for temporal in 0..depth {
                for y in 0..height {
                    let source =
                        ((batch_channel * depth + temporal) * tile.height + y) * tile.width;
                    let destination = batch_channel * output_plane
                        + temporal * output_height * output_width
                        + (output_y + y) * output_width
                        + output_x;
                    output[destination..destination + width]
                        .copy_from_slice(&tile.values[source..source + width]);
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_f32_impl<F>(
    config: &QwenImageVaeConfig,
    weights: &QwenImageVaeF32Weights,
    latents: &[f32],
    batch: usize,
    depth: usize,
    latent_height: usize,
    latent_width: usize,
    mut checkpoint: F,
) -> Result<Vec<f32>, ImageError>
where
    F: FnMut(&str, &[f32]),
{
    config.validate()?;
    if depth != 1 {
        return Err(ImageError::UnsupportedShape(format!(
            "still-image VAE decode requires temporal depth 1, found {depth}"
        )));
    }
    let expected_latents = checked_product(
        &[batch, config.z_dim, depth, latent_height, latent_width],
        "VAE latent input",
    )?;
    if batch == 0
        || latent_height == 0
        || latent_width == 0
        || latents.len() != expected_latents
        || latents.iter().any(|value| !value.is_finite())
    {
        return Err(ImageError::UnsupportedShape(format!(
            "invalid VAE latent geometry or values: length={}, expected={expected_latents}, shape=[{batch}, {}, {depth}, {latent_height}, {latent_width}]",
            latents.len(), config.z_dim
        )));
    }

    let channels = config
        .dim_mult
        .iter()
        .map(|multiplier| checked_mul(config.base_dim, *multiplier, "VAE decoder channels"))
        .collect::<Result<Vec<_>, _>>()?;
    let first = channels[0];
    let last = *channels.last().ok_or_else(|| {
        ImageError::UnsupportedShape("Qwen Image VAE has no decoder stages".to_string())
    })?;
    let mut shape = [depth, latent_height, latent_width];
    let mut current = conv3d(
        weights,
        "post_quant_conv",
        latents,
        batch,
        config.z_dim,
        config.z_dim,
        shape,
        [1, 1, 1],
        [1, 1, 1],
        [0, 0, 0],
    )?;
    checkpoint("post_quant_conv", &current);
    current = conv3d(
        weights,
        "decoder.conv_in",
        &current,
        batch,
        config.z_dim,
        last,
        shape,
        [3, 3, 3],
        [1, 1, 1],
        [1, 1, 1],
    )?;
    checkpoint("decoder_conv_in", &current);
    current = mid_block(weights, "decoder.mid_block", &current, batch, last, shape)?;
    checkpoint("decoder_mid_block", &current);

    let mut decoder_dims = Vec::with_capacity(channels.len() + 1);
    decoder_dims.push(last);
    decoder_dims.extend(channels.iter().rev().copied());
    let mut current_channels = last;
    for stage in 0..channels.len() {
        let mut expected_input = decoder_dims[stage];
        if stage > 0 {
            if expected_input % 2 != 0 {
                return Err(ImageError::UnsupportedShape(format!(
                    "decoder stage {stage} input channels {expected_input} cannot be halved"
                )));
            }
            expected_input /= 2;
        }
        if current_channels != expected_input {
            return Err(ImageError::UnsupportedShape(format!(
                "decoder stage {stage} received {current_channels} channels, expected {expected_input}"
            )));
        }
        let output_channels = decoder_dims[stage + 1];
        for residual in 0..=config.num_res_blocks {
            current = residual_block(
                weights,
                &format!("decoder.up_blocks.{stage}.resnets.{residual}"),
                &current,
                batch,
                current_channels,
                output_channels,
                shape,
            )?;
            current_channels = output_channels;
        }
        if stage + 1 != channels.len() {
            let upsampled_height = checked_mul(shape[1], 2, "VAE upsample height")?;
            let upsampled_width = checked_mul(shape[2], 2, "VAE upsample width")?;
            let mut upsampled = vec![
                0.0;
                checked_product(
                    &[
                        batch,
                        current_channels,
                        shape[0],
                        upsampled_height,
                        upsampled_width,
                    ],
                    "VAE nearest upsample",
                )?
            ];
            nearest_2x_ncthw(
                &current,
                batch,
                current_channels,
                shape[0],
                shape[1],
                shape[2],
                &mut upsampled,
            )
            .map_err(map_kernel_error)?;
            shape[1] = upsampled_height;
            shape[2] = upsampled_width;
            let next_channels = current_channels / 2;
            current = conv2d(
                weights,
                &format!("decoder.up_blocks.{stage}.upsamplers.0.resample.1"),
                &upsampled,
                batch,
                current_channels,
                next_channels,
                shape,
                [3, 3],
                [1, 1],
                [1, 1, 1, 1],
            )?;
            current_channels = next_channels;
        }
        checkpoint(&format!("decoder_up_block_{stage}"), &current);
    }
    if current_channels != first {
        return Err(ImageError::UnsupportedShape(format!(
            "VAE decoder head received {current_channels} channels, expected {first}"
        )));
    }
    vae_rms_norm_channels_ncthw(
        &mut current,
        batch,
        current_channels,
        shape[0],
        shape[1],
        shape[2],
        require(weights, "decoder.norm_out.gamma")?,
        1e-12,
    )
    .map_err(map_kernel_error)?;
    silu_inplace(&mut current);
    current = conv3d(
        weights,
        "decoder.conv_out",
        &current,
        batch,
        current_channels,
        config.input_channels,
        shape,
        [3, 3, 3],
        [1, 1, 1],
        [1, 1, 1],
    )?;
    checkpoint("decoder_conv_out", &current);
    current
        .iter_mut()
        .for_each(|value| *value = value.clamp(-1.0, 1.0));
    if current.iter().any(|value| !value.is_finite()) {
        return Err(ImageError::Numerical {
            component: "vae_decoder",
            step: 0,
        });
    }
    Ok(current)
}

#[allow(clippy::too_many_arguments)]
fn residual_block(
    weights: &QwenImageVaeF32Weights,
    prefix: &str,
    input: &[f32],
    batch: usize,
    input_channels: usize,
    output_channels: usize,
    shape: [usize; 3],
) -> Result<Vec<f32>, ImageError> {
    let shortcut = if input_channels == output_channels {
        input.to_vec()
    } else {
        conv3d(
            weights,
            &format!("{prefix}.conv_shortcut"),
            input,
            batch,
            input_channels,
            output_channels,
            shape,
            [1, 1, 1],
            [1, 1, 1],
            [0, 0, 0],
        )?
    };
    let mut hidden = input.to_vec();
    vae_rms_norm_channels_ncthw(
        &mut hidden,
        batch,
        input_channels,
        shape[0],
        shape[1],
        shape[2],
        require(weights, &format!("{prefix}.norm1.gamma"))?,
        1e-12,
    )
    .map_err(map_kernel_error)?;
    silu_inplace(&mut hidden);
    hidden = conv3d(
        weights,
        &format!("{prefix}.conv1"),
        &hidden,
        batch,
        input_channels,
        output_channels,
        shape,
        [3, 3, 3],
        [1, 1, 1],
        [1, 1, 1],
    )?;
    vae_rms_norm_channels_ncthw(
        &mut hidden,
        batch,
        output_channels,
        shape[0],
        shape[1],
        shape[2],
        require(weights, &format!("{prefix}.norm2.gamma"))?,
        1e-12,
    )
    .map_err(map_kernel_error)?;
    silu_inplace(&mut hidden);
    let mut output = conv3d(
        weights,
        &format!("{prefix}.conv2"),
        &hidden,
        batch,
        output_channels,
        output_channels,
        shape,
        [3, 3, 3],
        [1, 1, 1],
        [1, 1, 1],
    )?;
    for (output, residual) in output.iter_mut().zip(shortcut) {
        *output += residual;
    }
    Ok(output)
}

fn mid_block(
    weights: &QwenImageVaeF32Weights,
    prefix: &str,
    input: &[f32],
    batch: usize,
    channels: usize,
    shape: [usize; 3],
) -> Result<Vec<f32>, ImageError> {
    let hidden = residual_block(
        weights,
        &format!("{prefix}.resnets.0"),
        input,
        batch,
        channels,
        channels,
        shape,
    )?;
    let hidden = attention_block(
        weights,
        &format!("{prefix}.attentions.0"),
        &hidden,
        batch,
        channels,
        shape,
    )?;
    residual_block(
        weights,
        &format!("{prefix}.resnets.1"),
        &hidden,
        batch,
        channels,
        channels,
        shape,
    )
}

fn attention_block(
    weights: &QwenImageVaeF32Weights,
    prefix: &str,
    input: &[f32],
    batch: usize,
    channels: usize,
    shape: [usize; 3],
) -> Result<Vec<f32>, ImageError> {
    let mut normalized = input.to_vec();
    vae_rms_norm_channels_ncthw(
        &mut normalized,
        batch,
        channels,
        shape[0],
        shape[1],
        shape[2],
        require(weights, &format!("{prefix}.norm.gamma"))?,
        1e-12,
    )
    .map_err(map_kernel_error)?;
    let qkv = conv2d(
        weights,
        &format!("{prefix}.to_qkv"),
        &normalized,
        batch,
        channels,
        checked_mul(channels, 3, "VAE qkv channels")?,
        shape,
        [1, 1],
        [1, 1],
        [0, 0, 0, 0],
    )?;
    let frames = checked_mul(batch, shape[0], "VAE attention frames")?;
    let sequence = checked_mul(shape[1], shape[2], "VAE attention sequence")?;
    let attention_len = checked_product(&[frames, sequence, channels], "VAE attention")?;
    let mut query = vec![0.0; attention_len];
    let mut key = vec![0.0; attention_len];
    let mut value = vec![0.0; attention_len];
    let qkv_plane = shape[0] * sequence;
    for batch_index in 0..batch {
        for temporal in 0..shape[0] {
            let frame = batch_index * shape[0] + temporal;
            for position in 0..sequence {
                for channel in 0..channels {
                    let destination = (frame * sequence + position) * channels + channel;
                    let source = |qkv_channel: usize| {
                        (batch_index * 3 * channels + qkv_channel) * qkv_plane
                            + temporal * sequence
                            + position
                    };
                    query[destination] = qkv[source(channel)];
                    key[destination] = qkv[source(channels + channel)];
                    value[destination] = qkv[source(2 * channels + channel)];
                }
            }
        }
    }
    let mut attended = vec![0.0; attention_len];
    scaled_dot_product_attention(
        &query,
        &key,
        &value,
        frames,
        sequence,
        sequence,
        1,
        channels,
        None,
        &mut attended,
    )
    .map_err(map_kernel_error)?;
    let mut attended_ncthw = vec![0.0; input.len()];
    for batch_index in 0..batch {
        for temporal in 0..shape[0] {
            let frame = batch_index * shape[0] + temporal;
            for position in 0..sequence {
                for channel in 0..channels {
                    let source = (frame * sequence + position) * channels + channel;
                    let destination = (batch_index * channels + channel) * qkv_plane
                        + temporal * sequence
                        + position;
                    attended_ncthw[destination] = attended[source];
                }
            }
        }
    }
    let mut output = conv2d(
        weights,
        &format!("{prefix}.proj"),
        &attended_ncthw,
        batch,
        channels,
        channels,
        shape,
        [1, 1],
        [1, 1],
        [0, 0, 0, 0],
    )?;
    for (output, residual) in output.iter_mut().zip(input) {
        *output += residual;
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn conv3d(
    weights: &QwenImageVaeF32Weights,
    prefix: &str,
    input: &[f32],
    batch: usize,
    input_channels: usize,
    output_channels: usize,
    input_shape: [usize; 3],
    kernel: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
) -> Result<Vec<f32>, ImageError> {
    let output_shape = [
        conv_output_dim(input_shape[0], kernel[0], stride[0], 2 * padding[0])?,
        conv_output_dim(input_shape[1], kernel[1], stride[1], 2 * padding[1])?,
        conv_output_dim(input_shape[2], kernel[2], stride[2], 2 * padding[2])?,
    ];
    let mut output = vec![
        0.0;
        checked_product(
            &[
                batch,
                output_channels,
                output_shape[0],
                output_shape[1],
                output_shape[2]
            ],
            "VAE Conv3D output",
        )?
    ];
    causal_conv3d_ncthw(
        input,
        batch,
        input_channels,
        input_shape[0],
        input_shape[1],
        input_shape[2],
        require(weights, &format!("{prefix}.weight"))?,
        output_channels,
        kernel,
        stride,
        padding,
        Some(require(weights, &format!("{prefix}.bias"))?),
        &mut output,
    )
    .map_err(map_kernel_error)?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn conv2d(
    weights: &QwenImageVaeF32Weights,
    prefix: &str,
    input: &[f32],
    batch: usize,
    input_channels: usize,
    output_channels: usize,
    input_shape: [usize; 3],
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 4],
) -> Result<Vec<f32>, ImageError> {
    let output_shape = [
        conv_output_dim(
            input_shape[1],
            kernel[0],
            stride[0],
            padding[0] + padding[1],
        )?,
        conv_output_dim(
            input_shape[2],
            kernel[1],
            stride[1],
            padding[2] + padding[3],
        )?,
    ];
    let mut output = vec![
        0.0;
        checked_product(
            &[
                batch,
                output_channels,
                input_shape[0],
                output_shape[0],
                output_shape[1]
            ],
            "VAE Conv2D output",
        )?
    ];
    conv2d_ncthw(
        input,
        batch,
        input_channels,
        input_shape[0],
        input_shape[1],
        input_shape[2],
        require(weights, &format!("{prefix}.weight"))?,
        output_channels,
        kernel,
        stride,
        padding,
        Some(require(weights, &format!("{prefix}.bias"))?),
        &mut output,
    )
    .map_err(map_kernel_error)?;
    Ok(output)
}

fn conv_output_dim(
    input: usize,
    kernel: usize,
    stride: usize,
    total_padding: usize,
) -> Result<usize, ImageError> {
    let padded = input.checked_add(total_padding).ok_or_else(|| {
        ImageError::UnsupportedShape("VAE convolution padded dimension overflow".to_string())
    })?;
    if stride == 0 || kernel == 0 || padded < kernel {
        return Err(ImageError::UnsupportedShape(format!(
            "invalid VAE convolution geometry: input={input}, kernel={kernel}, stride={stride}, padding={total_padding}"
        )));
    }
    Ok((padded - kernel) / stride + 1)
}

fn require<'a>(weights: &'a QwenImageVaeF32Weights, name: &str) -> Result<&'a [f32], ImageError> {
    weights
        .get(name)
        .map(Vec::as_slice)
        .ok_or_else(|| ImageError::UnsupportedTensor(format!("missing VAE tensor `{name}`")))
}

fn checked_mul(lhs: usize, rhs: usize, label: &str) -> Result<usize, ImageError> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| ImageError::UnsupportedShape(format!("{label} overflow")))
}

fn checked_product(values: &[usize], label: &str) -> Result<usize, ImageError> {
    values
        .iter()
        .try_fold(1usize, |product, value| checked_mul(product, *value, label))
}

fn map_kernel_error(error: XrtError) -> ImageError {
    match error {
        XrtError::Shape(message) | XrtError::InvalidTensor(message) => {
            ImageError::UnsupportedShape(message)
        }
        other => ImageError::Execution(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::qwen_image::expected_vae_tensors;

    fn fixture() -> serde_json::Value {
        serde_json::from_str(include_str!(
            "../../../../../tests/fixtures/qwen-image/vae-decoder-diffusers-0.39.json"
        ))
        .unwrap()
    }

    fn assert_close(actual: f32, expected: f32, tolerance: f32) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual}, expected={expected}, error={}",
            (actual - expected).abs()
        );
    }

    fn fixture_config_and_weights() -> (QwenImageVaeConfig, QwenImageVaeF32Weights) {
        let fixture = fixture();
        let config = QwenImageVaeConfig::from_json_bytes(
            serde_json::to_string(&fixture["config"])
                .unwrap()
                .as_bytes(),
        )
        .unwrap();
        let mut specifications = expected_vae_tensors(&config)
            .unwrap()
            .into_iter()
            .filter(|tensor| {
                tensor.name.starts_with("decoder.") || tensor.name.starts_with("post_quant_conv.")
            })
            .collect::<Vec<_>>();
        specifications.sort_by(|left, right| left.name.cmp(&right.name));
        let weights = specifications
            .into_iter()
            .enumerate()
            .map(|(parameter_index, tensor)| {
                let length = tensor.shape.iter().product::<usize>();
                let values = (0..length)
                    .map(|flat_index| {
                        if tensor.name.ends_with(".gamma") {
                            1.0 + ((flat_index % 7) as f32 - 3.0) * 0.01
                                + (parameter_index + 1) as f32 * 0.0001
                        } else if tensor.name.ends_with(".bias") {
                            ((flat_index % 11) as f32 - 5.0) * 0.003
                                + (parameter_index + 1) as f32 * 0.0001
                        } else {
                            ((flat_index % 17) as f32 - 8.0) * 0.004
                                + (parameter_index + 1) as f32 * 0.00005
                        }
                    })
                    .collect::<Vec<_>>();
                (tensor.name, values)
            })
            .collect::<BTreeMap<_, _>>();
        (config, weights)
    }

    fn fixture_encoder_config_and_weights() -> (QwenImageVaeConfig, QwenImageVaeF32Weights) {
        let fixture = fixture();
        let config = QwenImageVaeConfig::from_json_bytes(
            serde_json::to_string(&fixture["config"])
                .unwrap()
                .as_bytes(),
        )
        .unwrap();
        let mut specifications = expected_vae_tensors(&config)
            .unwrap()
            .into_iter()
            .filter(|tensor| {
                tensor.name.starts_with("encoder.") || tensor.name.starts_with("quant_conv.")
            })
            .collect::<Vec<_>>();
        specifications.sort_by(|left, right| left.name.cmp(&right.name));
        assert_eq!(specifications.len(), 40);
        let weights = specifications
            .into_iter()
            .enumerate()
            .map(|(parameter_index, tensor)| {
                let length = tensor.shape.iter().product::<usize>();
                let values = (0..length)
                    .map(|flat_index| {
                        if tensor.name.ends_with(".gamma") {
                            1.0 + ((flat_index % 7) as f32 - 3.0) * 0.01
                                + (parameter_index + 1) as f32 * 0.0001
                        } else if tensor.name.ends_with(".bias") {
                            ((flat_index % 11) as f32 - 5.0) * 0.003
                                + (parameter_index + 1) as f32 * 0.0001
                        } else {
                            ((flat_index % 17) as f32 - 8.0) * 0.004
                                + (parameter_index + 1) as f32 * 0.00005
                        }
                    })
                    .collect::<Vec<_>>();
                (tensor.name, values)
            })
            .collect::<BTreeMap<_, _>>();
        (config, weights)
    }

    #[test]
    fn tiny_encoder_mode_matches_pinned_diffusers_graph() {
        let fixture = fixture();
        let encoder_fixture = &fixture["encoder"];
        let (config, weights) = fixture_encoder_config_and_weights();
        let source_length = encoder_fixture["source_shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as usize)
            .product::<usize>();
        let source = (0..source_length)
            .map(|index| ((index % 23) as f32 - 11.0) * 0.04)
            .collect::<Vec<_>>();
        let output = qwen_image_vae_encode_f32(&config, &weights, &source, 1, 1, 8, 8).unwrap();
        let expected = encoder_fixture["output"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect::<Vec<_>>();
        assert_eq!(output.len(), expected.len());
        for (actual, expected) in output.into_iter().zip(expected) {
            assert_close(actual, expected, 3e-5);
        }
    }

    #[test]
    fn tiny_decoder_matches_pinned_diffusers_graph() {
        let fixture = fixture();
        let (config, weights) = fixture_config_and_weights();
        let latents = fixture["latent"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect::<Vec<_>>();
        let mut checkpoints = BTreeMap::<String, Vec<f32>>::new();
        let output = decode_f32_impl(&config, &weights, &latents, 1, 1, 2, 2, |name, values| {
            checkpoints.insert(name.to_string(), values.to_vec());
        })
        .unwrap();

        for (name, expected) in fixture["checkpoints"].as_object().unwrap() {
            let actual = &checkpoints[name];
            for (index, expected) in expected["sample_indices"]
                .as_array()
                .unwrap()
                .iter()
                .zip(expected["samples"].as_array().unwrap())
            {
                assert_close(
                    actual[index.as_u64().unwrap() as usize],
                    expected.as_f64().unwrap() as f32,
                    3e-5,
                );
            }
        }
        let expected = fixture["output"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect::<Vec<_>>();
        assert_eq!(output.len(), expected.len());
        for (actual, expected) in output.into_iter().zip(expected) {
            assert_close(actual, expected, 3e-5);
        }
    }

    #[test]
    fn tiled_decoder_matches_pinned_diffusers_overlap_blending() {
        let fixture = fixture();
        let (config, weights) = fixture_config_and_weights();
        let tiled = &fixture["tiled"];
        let latent_length = tiled["latent_shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as usize)
            .product::<usize>();
        let latents = (0..latent_length)
            .map(|index| ((index % 9) as f32 - 4.0) * 0.1)
            .collect::<Vec<_>>();
        let mut checkpoints = Vec::new();
        let output = qwen_image_vae_decode_tiled_f32_with_control(
            &config,
            &weights,
            &latents,
            1,
            1,
            3,
            3,
            QwenImageVaeTiling {
                tile_latent_height: 2,
                tile_latent_width: 2,
                stride_latent_height: 1,
                stride_latent_width: 1,
            },
            |tile| {
                checkpoints.push(tile);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(checkpoints, (0..=9).collect::<Vec<_>>());
        assert_eq!(output.len(), 108);
        for (index, expected) in tiled["output"]["sample_indices"]
            .as_array()
            .unwrap()
            .iter()
            .zip(tiled["output"]["samples"].as_array().unwrap())
        {
            assert_close(
                output[index.as_u64().unwrap() as usize],
                expected.as_f64().unwrap() as f32,
                4e-5,
            );
        }
    }

    #[test]
    fn still_decoder_rejects_video_depth() {
        let fixture = fixture();
        let config = QwenImageVaeConfig::from_json_bytes(
            serde_json::to_string(&fixture["config"])
                .unwrap()
                .as_bytes(),
        )
        .unwrap();
        let error = qwen_image_vae_decode_f32(&config, &BTreeMap::new(), &[0.0; 16], 1, 2, 2, 2)
            .unwrap_err();
        assert!(error.to_string().contains("temporal depth 1"));
    }

    #[test]
    fn float_tensor_decoder_handles_bf16_and_f16_little_endian() {
        let bf16_bytes = [bf16::from_f32(1.5), bf16::from_f32(-2.25)]
            .into_iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        let f16_bytes = [f16::from_f32(1.5), f16::from_f32(-2.25)]
            .into_iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(
            decode_float_tensor(SafeTensorDType::Bf16, &bf16_bytes, "test").unwrap(),
            [1.5, -2.25]
        );
        assert_eq!(
            decode_float_tensor(SafeTensorDType::F16, &f16_bytes, "test").unwrap(),
            [1.5, -2.25]
        );
    }
}
