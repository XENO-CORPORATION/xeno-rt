use std::collections::BTreeMap;

use half::bf16;
use xrt_safetensors::{SafeTensorDType, SafeTensorStore};

use crate::ImageError;

use super::{
    transformer_executor::{
        execute_transformer, execute_transformer_for_shapes, QwenImageTransformerWeights,
    },
    validate_transformer_safetensors, QwenImageBf16Linear, QwenImagePromptEmbeddings,
    QwenImageTransformerConfig,
};

/// Mmap-backed BF16 CPU reference executor for the Qwen Image denoiser.
/// Matrix weights remain encoded; only one-dimensional bias and normalization
/// tensors are materialized as F32.
#[derive(Debug)]
pub struct QwenImageBf16Transformer {
    config: QwenImageTransformerConfig,
    store: SafeTensorStore,
    auxiliary: BTreeMap<String, Vec<f32>>,
}

impl QwenImageBf16Transformer {
    pub fn from_store(
        store: SafeTensorStore,
        config: QwenImageTransformerConfig,
    ) -> Result<Self, ImageError> {
        validate_transformer_safetensors(&store, &config)?;
        if config.use_additional_t_cond || config.use_layer3d_rope {
            return Err(ImageError::UnsupportedCapability(
                "transformer executor does not support additional-timestep or Layer3D-RoPE conditioning"
                    .to_string(),
            ));
        }
        let mut auxiliary = BTreeMap::new();
        for name in store.tensor_names() {
            let info = store.tensor_info(name).ok_or_else(|| {
                ImageError::CorruptComponent(format!(
                    "transformer tensor `{name}` disappeared during load"
                ))
            })?;
            if info.shape.len() == 1 {
                let view = store.require_tensor(name).map_err(|error| {
                    ImageError::CorruptComponent(format!(
                        "failed to map transformer tensor `{name}`: {error}"
                    ))
                })?;
                auxiliary.insert(name.to_string(), decode_bf16(view.data, name)?);
            } else if info.dtype != SafeTensorDType::Bf16 {
                return Err(ImageError::UnsupportedTensor(format!(
                    "BF16 transformer executor cannot run matrix `{name}` with dtype {:?}",
                    info.dtype
                )));
            }
        }
        Ok(Self {
            config,
            store,
            auxiliary,
        })
    }

    pub fn config(&self) -> &QwenImageTransformerConfig {
        &self.config
    }

    pub fn tensor_store(&self) -> &SafeTensorStore {
        &self.store
    }

    /// Execute the complete generation denoiser graph for one scheduler
    /// timestep. Packed latents use `[batch, image_sequence, in_channels]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        packed_latents: &[f32],
        prompt: &QwenImagePromptEmbeddings,
        timestep: &[f32],
        frames: usize,
        patch_height: usize,
        patch_width: usize,
    ) -> Result<Vec<f32>, ImageError> {
        self.forward_with_control(
            packed_latents,
            prompt,
            timestep,
            frames,
            patch_height,
            patch_width,
            |_| Ok(()),
        )
    }

    /// Execute the complete denoiser while giving the caller a cancellation
    /// checkpoint before every transformer block and once after the final
    /// block. The callback receives the next block index in `0..=num_layers`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_control<F>(
        &self,
        packed_latents: &[f32],
        prompt: &QwenImagePromptEmbeddings,
        timestep: &[f32],
        frames: usize,
        patch_height: usize,
        patch_width: usize,
        checkpoint: F,
    ) -> Result<Vec<f32>, ImageError>
    where
        F: FnMut(usize) -> Result<(), ImageError>,
    {
        execute_transformer(
            self,
            packed_latents,
            prompt,
            timestep,
            frames,
            patch_height,
            patch_width,
            checkpoint,
        )
    }

    /// Execute the pinned Edit-2511 `zero_cond_t` graph. Shapes and packed
    /// latent sequences are ordered `[output, source_0, source_1, ...]`.
    pub fn forward_edit_with_control<F>(
        &self,
        packed_latents: &[f32],
        prompt: &QwenImagePromptEmbeddings,
        timestep: &[f32],
        image_shapes: &[[usize; 3]],
        checkpoint: F,
    ) -> Result<Vec<f32>, ImageError>
    where
        F: FnMut(usize) -> Result<(), ImageError>,
    {
        execute_transformer_for_shapes(
            self,
            packed_latents,
            prompt,
            timestep,
            image_shapes,
            checkpoint,
        )
    }

    fn linear(
        &self,
        prefix: &str,
        input_features: usize,
        output_features: usize,
    ) -> Result<QwenImageBf16Linear<'_>, ImageError> {
        let name = format!("{prefix}.weight");
        let weight = self.store.require_tensor(&name).map_err(|error| {
            ImageError::CorruptComponent(format!("failed to map `{name}`: {error}"))
        })?;
        if weight.info.dtype != SafeTensorDType::Bf16 {
            return Err(ImageError::UnsupportedTensor(format!(
                "transformer matrix `{name}` is {:?}, expected BF16",
                weight.info.dtype
            )));
        }
        Ok(QwenImageBf16Linear {
            weight_bytes: weight.data,
            bias: self.auxiliary(&format!("{prefix}.bias"))?,
            input_features,
            output_features,
        })
    }

    fn auxiliary(&self, name: &str) -> Result<&[f32], ImageError> {
        self.auxiliary.get(name).map(Vec::as_slice).ok_or_else(|| {
            ImageError::UnsupportedTensor(format!("missing auxiliary tensor `{name}`"))
        })
    }
}

impl QwenImageTransformerWeights for QwenImageBf16Transformer {
    type Linear<'a> = QwenImageBf16Linear<'a>;

    fn config(&self) -> &QwenImageTransformerConfig {
        &self.config
    }

    fn linear(
        &self,
        prefix: &str,
        input_features: usize,
        output_features: usize,
    ) -> Result<Self::Linear<'_>, ImageError> {
        QwenImageBf16Transformer::linear(self, prefix, input_features, output_features)
    }

    fn auxiliary(&self, name: &str) -> Result<&[f32], ImageError> {
        QwenImageBf16Transformer::auxiliary(self, name)
    }
}

fn decode_bf16(bytes: &[u8], name: &str) -> Result<Vec<f32>, ImageError> {
    if bytes.len() % 2 != 0 {
        return Err(ImageError::CorruptComponent(format!(
            "BF16 transformer tensor `{name}` has odd byte length {}",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|encoded| bf16::from_bits(u16::from_le_bytes([encoded[0], encoded[1]])).to_f32())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{Dtype, TensorView};
    use xrt_safetensors::SafeTensorLayout;

    use crate::models::qwen_image::expected_transformer_tensors;

    #[test]
    fn bf16_auxiliary_decode_is_little_endian() {
        let encoded = [bf16::from_f32(1.25), bf16::from_f32(-0.5)]
            .into_iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(decode_bf16(&encoded, "test").unwrap(), [1.25, -0.5]);
    }

    #[test]
    fn complete_bf16_graph_matches_pinned_diffusers_fixture() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../../tests/fixtures/qwen-image/operators-diffusers-0.39.json"
        ))
        .unwrap();
        let fixture = &fixture["full_transformer"];
        let config = QwenImageTransformerConfig::from_json_bytes(
            serde_json::to_string(&fixture["config"])
                .unwrap()
                .as_bytes(),
        )
        .unwrap();
        let mut specifications = expected_transformer_tensors(&config).unwrap();
        specifications.sort_by(|left, right| left.name.cmp(&right.name));
        assert_eq!(specifications.len(), 77);
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
        let store = SafeTensorStore::open_exact(
            directory.path(),
            SafeTensorLayout::single("model.safetensors"),
        )
        .unwrap();
        let transformer = QwenImageBf16Transformer::from_store(store, config).unwrap();
        let packed_latents = (0..16)
            .map(|index| ((index % 9) as f32 - 4.0) * 0.07)
            .collect::<Vec<_>>();
        let prompt = QwenImagePromptEmbeddings {
            embeddings: (0..16)
                .map(|index| ((index % 7) as f32 - 3.0) * 0.05)
                .collect(),
            attention_mask: vec![1, 0],
            retained_lengths: vec![1],
            batch_size: 1,
            sequence_length: 2,
            hidden_size: 8,
        };
        let output = transformer
            .forward(&packed_latents, &prompt, &[0.125], 1, 2, 2)
            .unwrap();
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
                "output {index}: actual={actual}, expected={expected}, error={}",
                (actual - expected).abs()
            );
        }
    }

    #[test]
    fn edit_zero_cond_graph_matches_pinned_diffusers_fixture() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../../tests/fixtures/qwen-image/operators-diffusers-0.39.json"
        ))
        .unwrap();
        let fixture = &fixture["edit_transformer"];
        let config = QwenImageTransformerConfig::from_json_bytes(
            serde_json::to_string(&fixture["config"])
                .unwrap()
                .as_bytes(),
        )
        .unwrap();
        let mut specifications = expected_transformer_tensors(&config).unwrap();
        specifications.sort_by(|left, right| left.name.cmp(&right.name));
        assert_eq!(specifications.len(), 77);
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
        let store = SafeTensorStore::open_exact(
            directory.path(),
            SafeTensorLayout::single("model.safetensors"),
        )
        .unwrap();
        let transformer = QwenImageBf16Transformer::from_store(store, config).unwrap();
        let packed_latents = (0..24)
            .map(|index| ((index % 9) as f32 - 4.0) * 0.07)
            .collect::<Vec<_>>();
        let prompt = QwenImagePromptEmbeddings {
            embeddings: (0..16)
                .map(|index| ((index % 7) as f32 - 3.0) * 0.05)
                .collect(),
            attention_mask: vec![1, 0],
            retained_lengths: vec![1],
            batch_size: 1,
            sequence_length: 2,
            hidden_size: 8,
        };
        let output = transformer
            .forward_edit_with_control(
                &packed_latents,
                &prompt,
                &[0.125],
                &[[1, 2, 2], [1, 1, 2]],
                |_| Ok(()),
            )
            .unwrap();
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
                "output {index}: actual={actual}, expected={expected}, error={}",
                (actual - expected).abs()
            );
        }
    }
}
