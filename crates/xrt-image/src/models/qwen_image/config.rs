use std::{fs, path::Path};

use serde::Deserialize;

use crate::{
    scheduler::FlowMatchEulerConfig, ComponentRole, ImageCapability, ImageError, ImageModelBundle,
};

const MAX_COMPONENT_CONFIG_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct QwenImageTransformerConfig {
    #[serde(rename = "_class_name")]
    pub class_name: String,
    pub attention_head_dim: usize,
    pub axes_dims_rope: Vec<usize>,
    #[serde(default)]
    pub guidance_embeds: bool,
    pub in_channels: usize,
    pub joint_attention_dim: usize,
    pub num_attention_heads: usize,
    pub num_layers: usize,
    pub out_channels: usize,
    pub patch_size: usize,
    #[serde(default)]
    pub zero_cond_t: bool,
    #[serde(default)]
    pub use_additional_t_cond: bool,
    #[serde(default)]
    pub use_layer3d_rope: bool,
}

impl QwenImageTransformerConfig {
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, ImageError> {
        let config: Self = serde_json::from_slice(bytes).map_err(|error| {
            ImageError::Manifest(format!("invalid Qwen Image transformer config: {error}"))
        })?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), ImageError> {
        if self.class_name != "QwenImageTransformer2DModel" {
            return Err(ImageError::UnsupportedCapability(format!(
                "transformer class `{}` is not QwenImageTransformer2DModel",
                self.class_name
            )));
        }
        if self.num_layers == 0
            || self.num_attention_heads == 0
            || self.attention_head_dim == 0
            || self.in_channels == 0
            || self.out_channels == 0
            || self.joint_attention_dim == 0
            || self.patch_size == 0
        {
            return Err(ImageError::UnsupportedShape(
                "Qwen Image transformer dimensions must be positive".to_string(),
            ));
        }
        if self.axes_dims_rope.len() != 3
            || self
                .axes_dims_rope
                .iter()
                .any(|dimension| dimension % 2 != 0)
            || self.axes_dims_rope.iter().sum::<usize>() != self.attention_head_dim
        {
            return Err(ImageError::UnsupportedShape(format!(
                "RoPE axes {:?} must contain three even dimensions summing to head dimension {}",
                self.axes_dims_rope, self.attention_head_dim
            )));
        }
        let packed_channels = self
            .out_channels
            .checked_mul(self.patch_size)
            .and_then(|value| value.checked_mul(self.patch_size))
            .ok_or_else(|| ImageError::UnsupportedShape("packed channels overflow".to_string()))?;
        if packed_channels != self.in_channels {
            return Err(ImageError::UnsupportedShape(format!(
                "transformer in_channels {} must equal out_channels {} * patch_size {} squared",
                self.in_channels, self.out_channels, self.patch_size
            )));
        }
        self.inner_dim()?;
        Ok(())
    }

    pub fn inner_dim(&self) -> Result<usize, ImageError> {
        self.num_attention_heads
            .checked_mul(self.attention_head_dim)
            .ok_or_else(|| ImageError::UnsupportedShape("inner dimension overflow".to_string()))
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct QwenImageVaeConfig {
    #[serde(rename = "_class_name")]
    pub class_name: String,
    pub attn_scales: Vec<f32>,
    pub base_dim: usize,
    pub dim_mult: Vec<usize>,
    pub dropout: f32,
    #[serde(default = "default_vae_input_channels")]
    pub input_channels: usize,
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
    pub num_res_blocks: usize,
    // Upstream intentionally uses the historical `temperal` spelling.
    pub temperal_downsample: Vec<bool>,
    pub z_dim: usize,
}

impl QwenImageVaeConfig {
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, ImageError> {
        let config: Self = serde_json::from_slice(bytes).map_err(|error| {
            ImageError::Manifest(format!("invalid Qwen Image VAE config: {error}"))
        })?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), ImageError> {
        if self.class_name != "AutoencoderKLQwenImage" {
            return Err(ImageError::UnsupportedCapability(format!(
                "VAE class `{}` is not AutoencoderKLQwenImage",
                self.class_name
            )));
        }
        if self.base_dim == 0
            || self.z_dim == 0
            || self.input_channels == 0
            || self.num_res_blocks == 0
            || self.dim_mult.len() != self.temperal_downsample.len() + 1
            || self.dim_mult.contains(&0)
        {
            return Err(ImageError::UnsupportedShape(
                "invalid Qwen Image VAE block geometry".to_string(),
            ));
        }
        if self.latents_mean.len() != self.z_dim
            || self.latents_std.len() != self.z_dim
            || self.latents_mean.iter().any(|value| !value.is_finite())
            || self
                .latents_std
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(ImageError::UnsupportedShape(format!(
                "VAE latent mean/std must contain {} finite channels with positive std",
                self.z_dim
            )));
        }
        if !self.dropout.is_finite() || !(0.0..=1.0).contains(&self.dropout) {
            return Err(ImageError::Manifest(
                "VAE dropout must be finite in [0, 1]".to_string(),
            ));
        }
        self.scale_factor()?;
        Ok(())
    }

    pub fn scale_factor(&self) -> Result<usize, ImageError> {
        1usize
            .checked_shl(self.temperal_downsample.len() as u32)
            .ok_or_else(|| ImageError::UnsupportedShape("VAE scale factor overflow".to_string()))
    }
}

const fn default_vae_input_channels() -> usize {
    3
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct QwenTextCoreConfig {
    hidden_size: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    rope_scaling: QwenMropeConfig,
    vocab_size: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct QwenMropeConfig {
    mrope_section: [usize; 3],
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct QwenTextEnvelope {
    architectures: Vec<String>,
    dtype: String,
    hidden_size: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    model_type: String,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    rope_scaling: QwenMropeConfig,
    vocab_size: usize,
    text_config: QwenTextCoreConfig,
    #[serde(default)]
    image_token_id: Option<u32>,
    #[serde(default)]
    vision_start_token_id: Option<u32>,
    #[serde(default)]
    vision_end_token_id: Option<u32>,
    #[serde(default)]
    vision_config: Option<QwenImageVisionConfig>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct QwenImageVisionConfig {
    pub depth: usize,
    pub fullatt_block_indexes: Vec<usize>,
    pub hidden_act: String,
    pub hidden_size: usize,
    pub in_channels: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub out_hidden_size: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub window_size: usize,
}

impl QwenImageVisionConfig {
    pub fn validate(&self, text_hidden_size: usize) -> Result<(), ImageError> {
        if self.depth == 0
            || self.hidden_size == 0
            || self.in_channels == 0
            || self.intermediate_size == 0
            || self.num_heads == 0
            || self.patch_size == 0
            || self.spatial_merge_size == 0
            || self.temporal_patch_size == 0
            || self.window_size == 0
            || self.hidden_size % self.num_heads != 0
            || self.out_hidden_size != text_hidden_size
        {
            return Err(ImageError::UnsupportedShape(
                "invalid Qwen2.5-VL vision geometry".to_string(),
            ));
        }
        let window_stride = self
            .spatial_merge_size
            .checked_mul(self.patch_size)
            .ok_or_else(|| {
                ImageError::UnsupportedShape("Qwen2.5-VL vision window stride overflow".to_string())
            })?;
        if self.window_size % window_stride != 0 {
            return Err(ImageError::UnsupportedShape(
                "invalid Qwen2.5-VL vision window geometry".to_string(),
            ));
        }
        if self.hidden_act != "silu"
            || self.fullatt_block_indexes.is_empty()
            || self
                .fullatt_block_indexes
                .iter()
                .any(|index| *index >= self.depth)
        {
            return Err(ImageError::UnsupportedCapability(format!(
                "unsupported Qwen2.5-VL vision activation/full-attention layout: {}/ {:?}",
                self.hidden_act, self.fullatt_block_indexes
            )));
        }
        let mut indexes = self.fullatt_block_indexes.clone();
        indexes.sort_unstable();
        indexes.dedup();
        if indexes.len() != self.fullatt_block_indexes.len() {
            return Err(ImageError::UnsupportedShape(
                "vision full-attention block indexes must be unique".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct QwenImageTextConfig {
    pub architecture: String,
    pub dtype: String,
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub mrope_section: [usize; 3],
    pub vocab_size: usize,
    pub image_token_id: Option<u32>,
    pub vision_start_token_id: Option<u32>,
    pub vision_end_token_id: Option<u32>,
    pub vision: Option<QwenImageVisionConfig>,
}

impl QwenImageTextConfig {
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, ImageError> {
        let envelope: QwenTextEnvelope = serde_json::from_slice(bytes).map_err(|error| {
            ImageError::Manifest(format!("invalid Qwen2.5-VL text config: {error}"))
        })?;
        let architecture = envelope.architectures.first().cloned().ok_or_else(|| {
            ImageError::UnsupportedCapability("text encoder architecture is missing".to_string())
        })?;
        let core = &envelope.text_config;
        for (name, outer, inner) in [
            ("hidden_size", envelope.hidden_size, core.hidden_size),
            (
                "intermediate_size",
                envelope.intermediate_size,
                core.intermediate_size,
            ),
            (
                "max_position_embeddings",
                envelope.max_position_embeddings,
                core.max_position_embeddings,
            ),
            (
                "num_attention_heads",
                envelope.num_attention_heads,
                core.num_attention_heads,
            ),
            (
                "num_hidden_layers",
                envelope.num_hidden_layers,
                core.num_hidden_layers,
            ),
            (
                "num_key_value_heads",
                envelope.num_key_value_heads,
                core.num_key_value_heads,
            ),
            ("vocab_size", envelope.vocab_size, core.vocab_size),
        ] {
            if outer != inner || outer == 0 {
                return Err(ImageError::UnsupportedShape(format!(
                    "text encoder outer {name}={outer} does not match text_config {inner}"
                )));
            }
        }
        if (envelope.rms_norm_eps - core.rms_norm_eps).abs() > f32::EPSILON
            || (envelope.rope_theta - core.rope_theta).abs() > f32::EPSILON
            || envelope.rope_scaling != core.rope_scaling
            || !envelope.rms_norm_eps.is_finite()
            || envelope.rms_norm_eps <= 0.0
            || !envelope.rope_theta.is_finite()
            || envelope.rope_theta <= 0.0
        {
            return Err(ImageError::UnsupportedShape(
                "text encoder normalization or RoPE configuration is inconsistent".to_string(),
            ));
        }
        if envelope.hidden_size % envelope.num_attention_heads != 0 {
            return Err(ImageError::UnsupportedShape(
                "text encoder width is not divisible by its attention heads".to_string(),
            ));
        }
        let head_dim = envelope.hidden_size / envelope.num_attention_heads;
        let mrope_width = envelope
            .rope_scaling
            .mrope_section
            .iter()
            .try_fold(0usize, |total, section| total.checked_add(*section))
            .and_then(|total| total.checked_mul(2))
            .ok_or_else(|| {
                ImageError::UnsupportedShape("text mRoPE section width overflow".to_string())
            })?;
        if head_dim % 2 != 0 || mrope_width != head_dim {
            return Err(ImageError::UnsupportedShape(format!(
                "text mRoPE sections {:?} do not cover head dimension {head_dim}",
                envelope.rope_scaling.mrope_section
            )));
        }
        if architecture != "Qwen2_5_VLForConditionalGeneration"
            || envelope.model_type != "qwen2_5_vl"
        {
            return Err(ImageError::UnsupportedCapability(format!(
                "unsupported text encoder {architecture}/{}",
                envelope.model_type
            )));
        }
        if let Some(vision) = &envelope.vision_config {
            vision.validate(envelope.hidden_size)?;
            for (label, token) in [
                ("image_token_id", envelope.image_token_id),
                ("vision_start_token_id", envelope.vision_start_token_id),
                ("vision_end_token_id", envelope.vision_end_token_id),
            ] {
                let token = token.ok_or_else(|| {
                    ImageError::UnsupportedShape(format!(
                        "Qwen2.5-VL vision config requires {label}"
                    ))
                })?;
                if token as usize >= envelope.vocab_size {
                    return Err(ImageError::UnsupportedShape(format!(
                        "{label} {token} exceeds vocabulary {}",
                        envelope.vocab_size
                    )));
                }
            }
        }
        Ok(Self {
            architecture,
            dtype: envelope.dtype,
            model_type: envelope.model_type,
            hidden_size: envelope.hidden_size,
            intermediate_size: envelope.intermediate_size,
            max_position_embeddings: envelope.max_position_embeddings,
            num_attention_heads: envelope.num_attention_heads,
            num_hidden_layers: envelope.num_hidden_layers,
            num_key_value_heads: envelope.num_key_value_heads,
            rms_norm_eps: envelope.rms_norm_eps,
            rope_theta: envelope.rope_theta,
            mrope_section: envelope.rope_scaling.mrope_section,
            vocab_size: envelope.vocab_size,
            image_token_id: envelope.image_token_id,
            vision_start_token_id: envelope.vision_start_token_id,
            vision_end_token_id: envelope.vision_end_token_id,
            vision: envelope.vision_config,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct QwenImageBundleConfig {
    pub transformer: QwenImageTransformerConfig,
    pub vae: QwenImageVaeConfig,
    pub text_encoder: QwenImageTextConfig,
    pub scheduler: FlowMatchEulerConfig,
    pub max_sequence_length: usize,
}

impl QwenImageBundleConfig {
    pub fn load(bundle: &ImageModelBundle) -> Result<Self, ImageError> {
        let required_capability = match bundle.manifest().family.as_str() {
            "qwen-image" => ImageCapability::Generate,
            "qwen-image-edit" => ImageCapability::Edit,
            other => {
                return Err(ImageError::UnsupportedCapability(format!(
                    "bundle family `{other}` is not a supported Qwen Image family"
                )))
            }
        };
        if !bundle
            .manifest()
            .capabilities
            .contains(&required_capability)
        {
            return Err(ImageError::UnsupportedCapability(format!(
                "Qwen bundle `{}` does not advertise {}",
                bundle.manifest().family,
                required_capability.id()
            )));
        }
        if !matches!(
            bundle.manifest().quantization.as_str(),
            "BF16" | "Q8_0" | "Q6_K" | "Q5_K_M" | "Q4_K_M"
        ) {
            return Err(ImageError::UnsupportedQuantization(
                bundle.manifest().quantization.clone(),
            ));
        }
        let transformer = QwenImageTransformerConfig::from_json_bytes(&read_role_config(
            bundle,
            ComponentRole::Transformer,
            "config.json",
        )?)?;
        let vae = QwenImageVaeConfig::from_json_bytes(&read_role_config(
            bundle,
            ComponentRole::Vae,
            "config.json",
        )?)?;
        let text_encoder = QwenImageTextConfig::from_json_bytes(&read_role_config(
            bundle,
            ComponentRole::TextEncoder,
            "config.json",
        )?)?;
        let scheduler = FlowMatchEulerConfig::from_json_bytes(&read_role_config(
            bundle,
            ComponentRole::Scheduler,
            "scheduler_config.json",
        )?)?;
        if transformer.joint_attention_dim != text_encoder.hidden_size {
            return Err(ImageError::UnsupportedShape(format!(
                "transformer joint attention {} does not match text hidden size {}",
                transformer.joint_attention_dim, text_encoder.hidden_size
            )));
        }
        if transformer.out_channels != vae.z_dim {
            return Err(ImageError::UnsupportedShape(format!(
                "transformer latent channels {} do not match VAE z_dim {}",
                transformer.out_channels, vae.z_dim
            )));
        }
        let max_sequence_length = bundle.manifest().limits.max_sequence_length;
        if max_sequence_length == 0 || max_sequence_length > 1_024 {
            return Err(ImageError::UnsupportedShape(format!(
                "Qwen Image max_sequence_length must be in 1..=1024, found {max_sequence_length}"
            )));
        }
        Ok(Self {
            transformer,
            vae,
            text_encoder,
            scheduler,
            max_sequence_length,
        })
    }
}

fn read_role_config(
    bundle: &ImageModelBundle,
    role: ComponentRole,
    file_name: &str,
) -> Result<Vec<u8>, ImageError> {
    let matches = bundle
        .manifest()
        .components
        .iter()
        .filter(|component| component.role == role)
        .flat_map(|component| component.files.iter())
        .filter(|file| {
            Path::new(&file.path)
                .file_name()
                .is_some_and(|name| name == file_name)
        })
        .collect::<Vec<_>>();
    let [record] = matches.as_slice() else {
        return Err(ImageError::MissingComponent(format!(
            "expected exactly one {}/{} config, found {}",
            role.as_str(),
            file_name,
            matches.len()
        )));
    };
    if record.size_bytes > MAX_COMPONENT_CONFIG_BYTES {
        return Err(ImageError::Manifest(format!(
            "component config `{}` exceeds the byte limit",
            record.path
        )));
    }
    fs::read(bundle.root().join(&record.path)).map_err(ImageError::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_transformer_and_vae_configs_cross_validate() {
        let transformer = QwenImageTransformerConfig::from_json_bytes(
            br#"{
                "_class_name":"QwenImageTransformer2DModel",
                "attention_head_dim":128,
                "axes_dims_rope":[16,56,56],
                "guidance_embeds":false,
                "in_channels":64,
                "joint_attention_dim":3584,
                "num_attention_heads":24,
                "num_layers":60,
                "out_channels":16,
                "patch_size":2
            }"#,
        )
        .unwrap();
        let vae = QwenImageVaeConfig::from_json_bytes(
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
        .unwrap();
        assert_eq!(transformer.inner_dim().unwrap(), 3_072);
        assert_eq!(transformer.out_channels, vae.z_dim);
        assert_eq!(vae.scale_factor().unwrap(), 8);
    }

    #[test]
    fn transformer_rejects_inconsistent_packing() {
        let error = QwenImageTransformerConfig::from_json_bytes(
            br#"{
                "_class_name":"QwenImageTransformer2DModel",
                "attention_head_dim":128,
                "axes_dims_rope":[16,56,56],
                "in_channels":63,
                "joint_attention_dim":3584,
                "num_attention_heads":24,
                "num_layers":60,
                "out_channels":16,
                "patch_size":2
            }"#,
        )
        .unwrap_err();
        assert_eq!(error.kind(), crate::ImageErrorKind::UnsupportedShape);
    }
}
