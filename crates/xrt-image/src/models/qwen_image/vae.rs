use std::collections::BTreeSet;

use xrt_safetensors::{SafeTensorDType, SafeTensorLayout, SafeTensorStore};

use crate::{ComponentFormat, ComponentRole, ImageError, ImageModelBundle};

use super::{tensors::ExpectedTensor, QwenImageVaeConfig};

/// Derive the complete SafeTensors schema for the Diffusers
/// `AutoencoderKLQwenImage` graph from its component configuration.
pub fn expected_vae_tensors(
    config: &QwenImageVaeConfig,
) -> Result<Vec<ExpectedTensor>, ImageError> {
    config.validate()?;
    let mut tensors = Vec::new();
    let channels = config
        .dim_mult
        .iter()
        .map(|multiplier| checked_channels(config.base_dim, *multiplier))
        .collect::<Result<Vec<_>, _>>()?;
    let first = channels[0];
    let last = *channels.last().ok_or_else(|| {
        ImageError::UnsupportedShape("Qwen Image VAE has no channel stages".to_string())
    })?;

    push_conv3d(
        &mut tensors,
        "encoder.conv_in",
        config.input_channels,
        config.base_dim,
        [3, 3, 3],
    );
    let mut down_index = 0usize;
    let mut scale = 1.0f32;
    let mut in_channels = config.base_dim;
    for (stage, &out_channels) in channels.iter().enumerate() {
        for _ in 0..config.num_res_blocks {
            push_residual(
                &mut tensors,
                &format!("encoder.down_blocks.{down_index}"),
                in_channels,
                out_channels,
            );
            down_index += 1;
            if contains_scale(&config.attn_scales, scale) {
                push_attention(
                    &mut tensors,
                    &format!("encoder.down_blocks.{down_index}"),
                    out_channels,
                )?;
                down_index += 1;
            }
            in_channels = out_channels;
        }
        if stage + 1 != channels.len() {
            push_downsample(
                &mut tensors,
                &format!("encoder.down_blocks.{down_index}"),
                out_channels,
                config.temperal_downsample[stage],
            );
            down_index += 1;
            scale *= 0.5;
        }
    }
    push_mid_block(&mut tensors, "encoder.mid_block", last)?;
    push_norm(&mut tensors, "encoder.norm_out", last, false);
    push_conv3d(
        &mut tensors,
        "encoder.conv_out",
        last,
        checked_channels(config.z_dim, 2)?,
        [3, 3, 3],
    );

    push_conv3d(
        &mut tensors,
        "quant_conv",
        checked_channels(config.z_dim, 2)?,
        checked_channels(config.z_dim, 2)?,
        [1, 1, 1],
    );
    push_conv3d(
        &mut tensors,
        "post_quant_conv",
        config.z_dim,
        config.z_dim,
        [1, 1, 1],
    );

    push_conv3d(
        &mut tensors,
        "decoder.conv_in",
        config.z_dim,
        last,
        [3, 3, 3],
    );
    push_mid_block(&mut tensors, "decoder.mid_block", last)?;
    let mut decoder_dims = Vec::with_capacity(channels.len() + 1);
    decoder_dims.push(last);
    decoder_dims.extend(channels.iter().rev().copied());
    let temporal_upsample = config
        .temperal_downsample
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>();
    for stage in 0..channels.len() {
        let mut block_in = decoder_dims[stage];
        if stage > 0 {
            if block_in % 2 != 0 {
                return Err(ImageError::UnsupportedShape(format!(
                    "decoder stage {stage} input channels {block_in} cannot follow channel-halving upsample"
                )));
            }
            block_in /= 2;
        }
        let out_channels = decoder_dims[stage + 1];
        for residual in 0..=config.num_res_blocks {
            push_residual(
                &mut tensors,
                &format!("decoder.up_blocks.{stage}.resnets.{residual}"),
                block_in,
                out_channels,
            );
            block_in = out_channels;
        }
        if stage + 1 != channels.len() {
            push_upsample(
                &mut tensors,
                &format!("decoder.up_blocks.{stage}.upsamplers.0"),
                out_channels,
                temporal_upsample[stage],
            )?;
        }
    }
    push_norm(&mut tensors, "decoder.norm_out", first, false);
    push_conv3d(
        &mut tensors,
        "decoder.conv_out",
        first,
        config.input_channels,
        [3, 3, 3],
    );

    Ok(tensors)
}

/// Validate that a VAE store contains exactly the graph declared by the
/// component configuration. Unknown tensors are rejected to prevent a
/// superficially compatible but architecturally different checkpoint from
/// being admitted.
pub fn validate_vae_safetensors(
    store: &SafeTensorStore,
    config: &QwenImageVaeConfig,
) -> Result<(), ImageError> {
    let expected = expected_vae_tensors(config)?;
    let expected_names = expected
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect::<BTreeSet<_>>();
    for tensor in &expected {
        let info = store.tensor_info(&tensor.name).ok_or_else(|| {
            ImageError::UnsupportedTensor(format!("Qwen Image VAE is missing `{}`", tensor.name))
        })?;
        if !matches!(info.dtype, SafeTensorDType::Bf16 | SafeTensorDType::F16) {
            return Err(ImageError::UnsupportedTensor(format!(
                "Qwen Image VAE tensor `{}` has unsupported SafeTensors dtype {:?}",
                tensor.name, info.dtype
            )));
        }
        if info.shape != tensor.shape {
            return Err(ImageError::UnsupportedShape(format!(
                "Qwen Image VAE tensor `{}` has shape {:?}, expected {:?}",
                tensor.name, info.shape, tensor.shape
            )));
        }
    }
    if let Some(unknown) = store
        .tensor_names()
        .find(|name| !expected_names.contains(name))
    {
        return Err(ImageError::UnsupportedTensor(format!(
            "Qwen Image VAE contains unknown tensor `{unknown}`"
        )));
    }
    if store.tensor_count() != expected.len() {
        return Err(ImageError::UnsupportedTensor(format!(
            "Qwen Image VAE contains {} tensors, expected {}",
            store.tensor_count(),
            expected.len()
        )));
    }
    Ok(())
}

/// Open the exact SafeTensors files declared for the VAE role and validate
/// the complete Qwen Image schema before execution.
pub fn open_vae_safetensors(
    bundle: &ImageModelBundle,
    config: &QwenImageVaeConfig,
) -> Result<SafeTensorStore, ImageError> {
    let components = bundle
        .manifest()
        .components
        .iter()
        .filter(|component| component.role == ComponentRole::Vae)
        .collect::<Vec<_>>();
    let [component] = components.as_slice() else {
        return Err(ImageError::MissingComponent(format!(
            "expected exactly one VAE component, found {}",
            components.len()
        )));
    };
    if component.format != ComponentFormat::SafeTensors {
        return Err(ImageError::UnsupportedTensor(format!(
            "VAE component format `{}` is not safetensors",
            component.format.as_str()
        )));
    }
    let indexes = component
        .files
        .iter()
        .filter(|file| {
            file.path
                .to_ascii_lowercase()
                .ends_with(".safetensors.index.json")
        })
        .collect::<Vec<_>>();
    let tensor_files = component
        .files
        .iter()
        .filter(|file| file.path.to_ascii_lowercase().ends_with(".safetensors"))
        .map(|file| file.path.as_str())
        .collect::<Vec<_>>();
    let layout = match indexes.as_slice() {
        [] => {
            let [file] = tensor_files.as_slice() else {
                return Err(ImageError::MissingComponent(format!(
                    "unindexed VAE must declare exactly one SafeTensors file, found {}",
                    tensor_files.len()
                )));
            };
            SafeTensorLayout::single(*file)
        }
        [index] => SafeTensorLayout::indexed(index.path.as_str(), tensor_files),
        _ => {
            return Err(ImageError::CorruptComponent(format!(
                "VAE declares {} SafeTensors indexes",
                indexes.len()
            )))
        }
    };
    let store = SafeTensorStore::open_exact(bundle.root(), layout).map_err(|error| {
        ImageError::CorruptComponent(format!(
            "VAE SafeTensors component failed validation: {error}"
        ))
    })?;
    validate_vae_safetensors(&store, config)?;
    Ok(store)
}

fn checked_channels(base: usize, multiplier: usize) -> Result<usize, ImageError> {
    base.checked_mul(multiplier).ok_or_else(|| {
        ImageError::UnsupportedShape("Qwen Image VAE channel count overflow".to_string())
    })
}

fn contains_scale(scales: &[f32], expected: f32) -> bool {
    scales
        .iter()
        .any(|scale| (*scale - expected).abs() <= f32::EPSILON)
}

fn push_conv3d(
    tensors: &mut Vec<ExpectedTensor>,
    prefix: &str,
    input: usize,
    output: usize,
    kernel: [usize; 3],
) {
    tensors.push(ExpectedTensor::new(format!("{prefix}.bias"), vec![output]));
    tensors.push(ExpectedTensor::new(
        format!("{prefix}.weight"),
        vec![output, input, kernel[0], kernel[1], kernel[2]],
    ));
}

fn push_conv2d(
    tensors: &mut Vec<ExpectedTensor>,
    prefix: &str,
    input: usize,
    output: usize,
    kernel: usize,
) {
    tensors.push(ExpectedTensor::new(format!("{prefix}.bias"), vec![output]));
    tensors.push(ExpectedTensor::new(
        format!("{prefix}.weight"),
        vec![output, input, kernel, kernel],
    ));
}

fn push_norm(tensors: &mut Vec<ExpectedTensor>, prefix: &str, channels: usize, images: bool) {
    let shape = if images {
        vec![channels, 1, 1]
    } else {
        vec![channels, 1, 1, 1]
    };
    tensors.push(ExpectedTensor::new(format!("{prefix}.gamma"), shape));
}

fn push_residual(tensors: &mut Vec<ExpectedTensor>, prefix: &str, input: usize, output: usize) {
    push_conv3d(
        tensors,
        &format!("{prefix}.conv1"),
        input,
        output,
        [3, 3, 3],
    );
    push_conv3d(
        tensors,
        &format!("{prefix}.conv2"),
        output,
        output,
        [3, 3, 3],
    );
    if input != output {
        push_conv3d(
            tensors,
            &format!("{prefix}.conv_shortcut"),
            input,
            output,
            [1, 1, 1],
        );
    }
    push_norm(tensors, &format!("{prefix}.norm1"), input, false);
    push_norm(tensors, &format!("{prefix}.norm2"), output, false);
}

fn push_attention(
    tensors: &mut Vec<ExpectedTensor>,
    prefix: &str,
    channels: usize,
) -> Result<(), ImageError> {
    push_norm(tensors, &format!("{prefix}.norm"), channels, true);
    push_conv2d(
        tensors,
        &format!("{prefix}.to_qkv"),
        channels,
        checked_channels(channels, 3)?,
        1,
    );
    push_conv2d(tensors, &format!("{prefix}.proj"), channels, channels, 1);
    Ok(())
}

fn push_mid_block(
    tensors: &mut Vec<ExpectedTensor>,
    prefix: &str,
    channels: usize,
) -> Result<(), ImageError> {
    push_residual(tensors, &format!("{prefix}.resnets.0"), channels, channels);
    push_attention(tensors, &format!("{prefix}.attentions.0"), channels)?;
    push_residual(tensors, &format!("{prefix}.resnets.1"), channels, channels);
    Ok(())
}

fn push_downsample(
    tensors: &mut Vec<ExpectedTensor>,
    prefix: &str,
    channels: usize,
    temporal: bool,
) {
    push_conv2d(
        tensors,
        &format!("{prefix}.resample.1"),
        channels,
        channels,
        3,
    );
    if temporal {
        push_conv3d(
            tensors,
            &format!("{prefix}.time_conv"),
            channels,
            channels,
            [3, 1, 1],
        );
    }
}

fn push_upsample(
    tensors: &mut Vec<ExpectedTensor>,
    prefix: &str,
    channels: usize,
    temporal: bool,
) -> Result<(), ImageError> {
    if channels % 2 != 0 {
        return Err(ImageError::UnsupportedShape(format!(
            "VAE upsample channels {channels} cannot be halved"
        )));
    }
    push_conv2d(
        tensors,
        &format!("{prefix}.resample.1"),
        channels,
        channels / 2,
        3,
    );
    if temporal {
        push_conv3d(
            tensors,
            &format!("{prefix}.time_conv"),
            channels,
            checked_channels(channels, 2)?,
            [3, 1, 1],
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pinned_config() -> QwenImageVaeConfig {
        QwenImageVaeConfig::from_json_bytes(
            br#"{
                "_class_name":"AutoencoderKLQwenImage",
                "attn_scales":[],
                "base_dim":96,
                "dim_mult":[1,2,4,4],
                "dropout":0.0,
                "latents_mean":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                "latents_std":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                "num_res_blocks":2,
                "temperal_downsample":[false,true,true],
                "z_dim":16
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn pinned_2512_schema_matches_official_vae_graph() {
        let tensors = expected_vae_tensors(&pinned_config()).unwrap();
        assert_eq!(tensors.len(), 194);
        let by_name = tensors
            .iter()
            .map(|tensor| (tensor.name.as_str(), tensor.shape.as_slice()))
            .collect::<std::collections::BTreeMap<_, _>>();
        assert_eq!(
            by_name["encoder.down_blocks.5.time_conv.weight"],
            [192, 192, 3, 1, 1]
        );
        assert_eq!(
            by_name["decoder.up_blocks.1.resnets.0.conv_shortcut.weight"],
            [384, 192, 1, 1, 1]
        );
        assert_eq!(
            by_name["decoder.up_blocks.0.upsamplers.0.time_conv.weight"],
            [768, 384, 3, 1, 1]
        );
        assert_eq!(by_name["decoder.conv_out.weight"], [3, 96, 3, 3, 3]);
    }

    #[test]
    fn configured_encoder_attention_changes_flat_block_indices() {
        let mut config = pinned_config();
        config.attn_scales = vec![1.0];
        let tensors = expected_vae_tensors(&config).unwrap();
        assert!(tensors
            .iter()
            .any(|tensor| tensor.name == "encoder.down_blocks.1.to_qkv.weight"));
        assert!(tensors
            .iter()
            .any(|tensor| tensor.name == "encoder.down_blocks.4.resample.1.weight"));
    }
}
