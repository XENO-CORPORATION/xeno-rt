use std::collections::{BTreeMap, BTreeSet};

use half::bf16;
use xrt_kernels::linear_bf16;
use xrt_safetensors::{SafeTensorDType, SafeTensorStore};

use crate::{ImageError, ImageGenerationRequest};

use super::{transformer::QwenImageLinearOperator, QwenImageTransformerConfig};

pub const QWEN_IMAGE_2512_LIGHTNING_4STEP_BF16_FILE: &str =
    "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors";

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QwenImageDistilledProfile {
    pub steps: usize,
    pub true_cfg_scale: f32,
}

impl QwenImageDistilledProfile {
    pub fn validate_request(&self, request: &ImageGenerationRequest) -> Result<(), ImageError> {
        if request.steps != self.steps {
            return Err(ImageError::InvalidRequest(format!(
                "this distilled Qwen Image bundle requires exactly {} denoising steps, found {}",
                self.steps, request.steps
            )));
        }
        if request.true_cfg_scale != self.true_cfg_scale {
            return Err(ImageError::InvalidRequest(format!(
                "this distilled Qwen Image bundle requires true_cfg_scale={}, found {}",
                self.true_cfg_scale, request.true_cfg_scale
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
struct QwenImageLoraLayer {
    rank: usize,
    input_features: usize,
    output_features: usize,
    scale: f32,
}

#[derive(Debug)]
pub struct QwenImageLoraAdapter {
    store: SafeTensorStore,
    layers: BTreeMap<String, QwenImageLoraLayer>,
    profile: QwenImageDistilledProfile,
    cuda_bytes: u64,
}

impl QwenImageLoraAdapter {
    pub fn from_store(
        store: SafeTensorStore,
        config: &QwenImageTransformerConfig,
        profile: QwenImageDistilledProfile,
    ) -> Result<Self, ImageError> {
        config.validate()?;
        if profile.steps == 0 || !profile.true_cfg_scale.is_finite() || profile.true_cfg_scale < 0.0
        {
            return Err(ImageError::CorruptComponent(
                "distilled inference profile is invalid".to_string(),
            ));
        }

        let expected = expected_lora_layers(config)?;
        let mut expected_names = BTreeSet::new();
        let mut layers = BTreeMap::new();
        let mut cuda_bytes = 0u64;
        for (prefix, input_features, output_features) in expected {
            let alpha_name = format!("{prefix}.alpha");
            let down_name = format!("{prefix}.lora_down.weight");
            let up_name = format!("{prefix}.lora_up.weight");
            expected_names.extend([alpha_name.clone(), down_name.clone(), up_name.clone()]);

            let alpha = store.require_tensor(&alpha_name).map_err(|error| {
                ImageError::CorruptComponent(format!(
                    "failed to map Lightning tensor `{alpha_name}`: {error}"
                ))
            })?;
            if alpha.info.dtype != SafeTensorDType::Bf16 || !alpha.info.shape.is_empty() {
                return Err(ImageError::UnsupportedTensor(format!(
                    "Lightning alpha `{alpha_name}` must be a scalar BF16 tensor"
                )));
            }
            let alpha = decode_bf16_scalar(alpha.data, &alpha_name)?;

            let down = store.require_tensor(&down_name).map_err(|error| {
                ImageError::CorruptComponent(format!(
                    "failed to map Lightning tensor `{down_name}`: {error}"
                ))
            })?;
            let up = store.require_tensor(&up_name).map_err(|error| {
                ImageError::CorruptComponent(format!(
                    "failed to map Lightning tensor `{up_name}`: {error}"
                ))
            })?;
            if down.info.dtype != SafeTensorDType::Bf16
                || up.info.dtype != SafeTensorDType::Bf16
                || down.info.shape.len() != 2
                || up.info.shape.len() != 2
            {
                return Err(ImageError::UnsupportedTensor(format!(
                    "Lightning pair `{prefix}` must contain two rank-2 BF16 matrices"
                )));
            }
            let rank = down.info.shape[0];
            if rank == 0
                || down.info.shape != [rank, input_features]
                || up.info.shape != [output_features, rank]
            {
                return Err(ImageError::UnsupportedShape(format!(
                    "Lightning pair `{prefix}` has down/up shapes {:?}/{:?}, expected [{rank}, {input_features}]/[{output_features}, {rank}]",
                    down.info.shape, up.info.shape
                )));
            }
            if !alpha.is_finite() || alpha <= 0.0 {
                return Err(ImageError::CorruptComponent(format!(
                    "Lightning alpha `{alpha_name}` must be finite and positive"
                )));
            }
            let scale = alpha / rank as f32;
            cuda_bytes = cuda_bytes
                .checked_add(
                    u64::try_from(down.info.byte_len / 2 + up.info.byte_len / 2)
                        .unwrap_or(u64::MAX)
                        .checked_mul(4)
                        .ok_or_else(|| {
                            ImageError::Admission(
                                "Lightning CUDA byte estimate overflowed".to_string(),
                            )
                        })?,
                )
                .ok_or_else(|| {
                    ImageError::Admission("Lightning CUDA byte estimate overflowed".to_string())
                })?;
            layers.insert(
                prefix,
                QwenImageLoraLayer {
                    rank,
                    input_features,
                    output_features,
                    scale,
                },
            );
        }

        if let Some(unknown) = store
            .tensor_names()
            .find(|name| !expected_names.contains(*name))
        {
            return Err(ImageError::UnsupportedTensor(format!(
                "Lightning adapter contains unknown tensor `{unknown}`"
            )));
        }
        if store.tensor_count() != expected_names.len() {
            return Err(ImageError::UnsupportedTensor(format!(
                "Lightning adapter contains {} tensors, expected {}",
                store.tensor_count(),
                expected_names.len()
            )));
        }

        Ok(Self {
            store,
            layers,
            profile,
            cuda_bytes,
        })
    }

    pub const fn profile(&self) -> QwenImageDistilledProfile {
        self.profile
    }

    pub const fn cuda_bytes(&self) -> u64 {
        self.cuda_bytes
    }

    #[cfg(feature = "cuda")]
    pub(super) fn layer_names(&self) -> impl Iterator<Item = &str> {
        self.layers.keys().map(String::as_str)
    }

    pub(super) fn layer_view(
        &self,
        prefix: &str,
    ) -> Result<Option<QwenImageLoraView<'_>>, ImageError> {
        let Some(layer) = self.layers.get(prefix) else {
            return Ok(None);
        };
        let down_name = format!("{prefix}.lora_down.weight");
        let up_name = format!("{prefix}.lora_up.weight");
        let down = self.store.require_tensor(&down_name).map_err(|error| {
            ImageError::CorruptComponent(format!(
                "failed to map Lightning tensor `{down_name}`: {error}"
            ))
        })?;
        let up = self.store.require_tensor(&up_name).map_err(|error| {
            ImageError::CorruptComponent(format!(
                "failed to map Lightning tensor `{up_name}`: {error}"
            ))
        })?;
        Ok(Some(QwenImageLoraView {
            down: down.data,
            up: up.data,
            rank: layer.rank,
            input_features: layer.input_features,
            output_features: layer.output_features,
            scale: layer.scale,
        }))
    }

    pub(super) fn wrap_linear<'a, L>(
        &'a self,
        prefix: &str,
        base: L,
    ) -> Result<QwenImageLoraLinear<'a, L>, ImageError> {
        Ok(QwenImageLoraLinear {
            base,
            lora: self.layer_view(prefix)?,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct QwenImageLoraView<'a> {
    pub down: &'a [u8],
    pub up: &'a [u8],
    pub rank: usize,
    pub input_features: usize,
    pub output_features: usize,
    pub scale: f32,
}

pub(super) struct QwenImageLoraLinear<'a, L> {
    base: L,
    lora: Option<QwenImageLoraView<'a>>,
}

impl<L> QwenImageLoraLinear<'_, L> {
    pub(super) fn unadapted(base: L) -> Self {
        Self { base, lora: None }
    }
}

impl<L: QwenImageLinearOperator> QwenImageLinearOperator for QwenImageLoraLinear<'_, L> {
    fn forward(&self, input: &[f32], rows: usize) -> Result<Vec<f32>, ImageError> {
        let mut output = self.base.forward(input, rows)?;
        let Some(lora) = self.lora else {
            return Ok(output);
        };
        let mut hidden = vec![
            0.0f32;
            rows.checked_mul(lora.rank).ok_or_else(|| {
                ImageError::UnsupportedShape("Lightning down projection overflowed".to_string())
            })?
        ];
        linear_bf16(
            input,
            rows,
            lora.input_features,
            lora.down,
            lora.rank,
            None,
            &mut hidden,
        )
        .map_err(map_kernel_error)?;
        let mut delta = vec![
            0.0f32;
            rows.checked_mul(lora.output_features).ok_or_else(|| {
                ImageError::UnsupportedShape("Lightning up projection overflowed".to_string())
            })?
        ];
        linear_bf16(
            &hidden,
            rows,
            lora.rank,
            lora.up,
            lora.output_features,
            None,
            &mut delta,
        )
        .map_err(map_kernel_error)?;
        for (value, update) in output.iter_mut().zip(delta) {
            *value += lora.scale * update;
        }
        Ok(output)
    }
}

fn expected_lora_layers(
    config: &QwenImageTransformerConfig,
) -> Result<Vec<(String, usize, usize)>, ImageError> {
    let inner = config.inner_dim()?;
    let feed_forward = inner.checked_mul(4).ok_or_else(|| {
        ImageError::UnsupportedShape("Lightning feed-forward dimension overflowed".to_string())
    })?;
    let mut layers = Vec::with_capacity(config.num_layers.saturating_mul(12));
    for layer in 0..config.num_layers {
        let prefix = format!("transformer_blocks.{layer}");
        for projection in [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ] {
            layers.push((format!("{prefix}.{projection}"), inner, inner));
        }
        for stream in ["img", "txt"] {
            layers.push((
                format!("{prefix}.{stream}_mlp.net.0.proj"),
                inner,
                feed_forward,
            ));
            layers.push((format!("{prefix}.{stream}_mlp.net.2"), feed_forward, inner));
        }
    }
    Ok(layers)
}

fn decode_bf16_scalar(bytes: &[u8], name: &str) -> Result<f32, ImageError> {
    let [low, high] = bytes else {
        return Err(ImageError::CorruptComponent(format!(
            "BF16 scalar `{name}` must contain exactly two bytes"
        )));
    };
    Ok(bf16::from_bits(u16::from_le_bytes([*low, *high])).to_f32())
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

    fn config() -> QwenImageTransformerConfig {
        QwenImageTransformerConfig {
            class_name: "QwenImageTransformer2DModel".to_string(),
            attention_head_dim: 6,
            axes_dims_rope: vec![2, 2, 2],
            guidance_embeds: false,
            in_channels: 4,
            joint_attention_dim: 6,
            num_attention_heads: 1,
            num_layers: 1,
            out_channels: 1,
            patch_size: 2,
            zero_cond_t: false,
            use_additional_t_cond: false,
            use_layer3d_rope: false,
        }
    }

    fn write_adapter(path: &std::path::Path, omit_last: bool) {
        let rank = 2usize;
        let mut encoded = Vec::<(String, Vec<usize>, Vec<u8>)>::new();
        for (index, (prefix, input, output)) in expected_lora_layers(&config())
            .unwrap()
            .into_iter()
            .enumerate()
        {
            if omit_last && index == 11 {
                continue;
            }
            encoded.push((
                format!("{prefix}.alpha"),
                vec![],
                bf16::from_f32(rank as f32).to_bits().to_le_bytes().to_vec(),
            ));
            encoded.push((
                format!("{prefix}.lora_down.weight"),
                vec![rank, input],
                vec![0u8; rank * input * 2],
            ));
            encoded.push((
                format!("{prefix}.lora_up.weight"),
                vec![output, rank],
                vec![0u8; output * rank * 2],
            ));
        }
        let views = encoded
            .iter()
            .map(|(name, shape, bytes)| {
                (
                    name.clone(),
                    TensorView::new(Dtype::BF16, shape.clone(), bytes).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        safetensors::serialize_to_file(views, &None, path).unwrap();
    }

    #[test]
    fn validates_complete_lightning_schema_and_profile() {
        let directory = tempfile::tempdir().unwrap();
        let file = QWEN_IMAGE_2512_LIGHTNING_4STEP_BF16_FILE;
        write_adapter(&directory.path().join(file), false);
        let store =
            SafeTensorStore::open_exact(directory.path(), SafeTensorLayout::single(file)).unwrap();
        let adapter = QwenImageLoraAdapter::from_store(
            store,
            &config(),
            QwenImageDistilledProfile {
                steps: 4,
                true_cfg_scale: 1.0,
            },
        )
        .unwrap();
        assert_eq!(adapter.layers.len(), 12);
        assert_eq!(adapter.profile().steps, 4);
        assert!(adapter.cuda_bytes() > 0);
    }

    #[test]
    fn rejects_incomplete_lightning_schema() {
        let directory = tempfile::tempdir().unwrap();
        let file = QWEN_IMAGE_2512_LIGHTNING_4STEP_BF16_FILE;
        write_adapter(&directory.path().join(file), true);
        let store =
            SafeTensorStore::open_exact(directory.path(), SafeTensorLayout::single(file)).unwrap();
        let error = QwenImageLoraAdapter::from_store(
            store,
            &config(),
            QwenImageDistilledProfile {
                steps: 4,
                true_cfg_scale: 1.0,
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("failed to map Lightning tensor"));
    }

    #[test]
    fn distilled_profile_rejects_wrong_steps_or_cfg() {
        let profile = QwenImageDistilledProfile {
            steps: 4,
            true_cfg_scale: 1.0,
        };
        let request = ImageGenerationRequest {
            steps: 8,
            true_cfg_scale: 1.0,
            ..ImageGenerationRequest::default()
        };
        assert!(profile.validate_request(&request).is_err());
        let request = ImageGenerationRequest {
            steps: 4,
            true_cfg_scale: 4.0,
            ..ImageGenerationRequest::default()
        };
        assert!(profile.validate_request(&request).is_err());
    }
}
