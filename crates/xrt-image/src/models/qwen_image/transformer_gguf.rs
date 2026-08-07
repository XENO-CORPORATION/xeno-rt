use std::collections::BTreeMap;

use half::{bf16, f16};
use xrt_core::DType;
use xrt_gguf::{GgufFile, QWEN_IMAGE_EDIT_TIMESTEP_ZERO_MARKER};

use crate::ImageError;

use super::lora::QwenImageLoraLinear;
use super::{
    transformer_executor::{
        execute_transformer, execute_transformer_for_shapes, QwenImageTransformerWeights,
    },
    validate_transformer_gguf, QwenImageDistilledProfile, QwenImageGgufLinear,
    QwenImageLoraAdapter, QwenImagePromptEmbeddings, QwenImageTransformerConfig,
};

/// Mmap-backed mixed-quantization CPU executor for a validated Qwen Image
/// transformer GGUF. Matrix tensors remain encoded and dispatch through the
/// same graph as the BF16 reference executor.
pub struct QwenImageGgufTransformer {
    config: QwenImageTransformerConfig,
    file: GgufFile,
    auxiliary: BTreeMap<String, Vec<f32>>,
    adapter: Option<QwenImageLoraAdapter>,
}

impl std::fmt::Debug for QwenImageGgufTransformer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("QwenImageGgufTransformer")
            .field("config", &self.config)
            .field("path", &self.file.path())
            .field("tensor_count", &self.file.tensor_infos().len())
            .field("has_adapter", &self.adapter.is_some())
            .finish_non_exhaustive()
    }
}

impl QwenImageGgufTransformer {
    pub fn from_file(
        file: GgufFile,
        config: QwenImageTransformerConfig,
        quantization: &str,
    ) -> Result<Self, ImageError> {
        Self::from_file_with_adapter(file, config, quantization, None)
    }

    pub fn from_file_with_adapter(
        file: GgufFile,
        config: QwenImageTransformerConfig,
        quantization: &str,
        adapter: Option<QwenImageLoraAdapter>,
    ) -> Result<Self, ImageError> {
        validate_transformer_gguf(&file, &config, quantization)?;
        if config.use_additional_t_cond || config.use_layer3d_rope {
            return Err(ImageError::UnsupportedCapability(
                "transformer executor does not support additional-timestep or Layer3D-RoPE conditioning"
                    .to_string(),
            ));
        }
        let mut auxiliary = BTreeMap::new();
        for info in file.tensor_infos().iter().filter(|info| {
            info.dimensions.len() == 1 && info.name != QWEN_IMAGE_EDIT_TIMESTEP_ZERO_MARKER
        }) {
            let data = file.tensor_data(&info.name).map_err(|error| {
                ImageError::CorruptComponent(format!(
                    "failed to map GGUF transformer tensor `{}`: {error}",
                    info.name
                ))
            })?;
            auxiliary.insert(
                info.name.clone(),
                decode_float_tensor(info.dtype, data, &info.name)?,
            );
        }
        Ok(Self {
            config,
            file,
            auxiliary,
            adapter,
        })
    }

    pub fn distilled_profile(&self) -> Option<QwenImageDistilledProfile> {
        self.adapter.as_ref().map(QwenImageLoraAdapter::profile)
    }

    pub fn config(&self) -> &QwenImageTransformerConfig {
        &self.config
    }

    pub fn gguf(&self) -> &GgufFile {
        &self.file
    }

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
    ) -> Result<QwenImageGgufLinear<'_>, ImageError> {
        let name = format!("{prefix}.weight");
        let info = self.file.require_tensor(&name).map_err(|error| {
            ImageError::CorruptComponent(format!("failed to resolve `{name}`: {error}"))
        })?;
        if info.dimensions != [input_features, output_features] {
            return Err(ImageError::UnsupportedShape(format!(
                "GGUF matrix `{name}` has dimensions {:?}, expected [{input_features}, {output_features}]",
                info.dimensions
            )));
        }
        let weight_bytes = self.file.tensor_data(&name).map_err(|error| {
            ImageError::CorruptComponent(format!("failed to map `{name}`: {error}"))
        })?;
        Ok(QwenImageGgufLinear {
            weight_bytes,
            dtype: info.dtype,
            bias: self.auxiliary(&format!("{prefix}.bias"))?,
            input_features,
            output_features,
        })
    }

    fn auxiliary(&self, name: &str) -> Result<&[f32], ImageError> {
        self.auxiliary.get(name).map(Vec::as_slice).ok_or_else(|| {
            ImageError::UnsupportedTensor(format!(
                "missing or non-floating GGUF auxiliary tensor `{name}`"
            ))
        })
    }
}

impl QwenImageTransformerWeights for QwenImageGgufTransformer {
    type Linear<'a> = QwenImageLoraLinear<'a, QwenImageGgufLinear<'a>>;

    fn config(&self) -> &QwenImageTransformerConfig {
        &self.config
    }

    fn linear(
        &self,
        prefix: &str,
        input_features: usize,
        output_features: usize,
    ) -> Result<Self::Linear<'_>, ImageError> {
        let base = QwenImageGgufTransformer::linear(self, prefix, input_features, output_features)?;
        match &self.adapter {
            Some(adapter) => adapter.wrap_linear(prefix, base),
            None => Ok(QwenImageLoraLinear::unadapted(base)),
        }
    }

    fn auxiliary(&self, name: &str) -> Result<&[f32], ImageError> {
        QwenImageGgufTransformer::auxiliary(self, name)
    }
}

fn decode_float_tensor(dtype: DType, bytes: &[u8], name: &str) -> Result<Vec<f32>, ImageError> {
    let values: Vec<f32> = match dtype {
        DType::F32 if bytes.len() % 4 == 0 => bytes
            .chunks_exact(4)
            .map(|encoded| f32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]))
            .collect(),
        DType::F16 if bytes.len() % 2 == 0 => bytes
            .chunks_exact(2)
            .map(|encoded| f16::from_bits(u16::from_le_bytes([encoded[0], encoded[1]])).to_f32())
            .collect(),
        DType::BF16 if bytes.len() % 2 == 0 => bytes
            .chunks_exact(2)
            .map(|encoded| bf16::from_bits(u16::from_le_bytes([encoded[0], encoded[1]])).to_f32())
            .collect(),
        DType::F32 | DType::F16 | DType::BF16 => {
            return Err(ImageError::CorruptComponent(format!(
                "GGUF auxiliary tensor `{name}` has invalid byte length {} for {dtype:?}",
                bytes.len()
            )))
        }
        _ => {
            return Err(ImageError::UnsupportedTensor(format!(
                "GGUF auxiliary tensor `{name}` cannot use {dtype:?}"
            )))
        }
    };
    if values.iter().any(|value| !value.is_finite()) {
        return Err(ImageError::CorruptComponent(format!(
            "GGUF auxiliary tensor `{name}` contains non-finite values"
        )));
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auxiliary_decoder_accepts_all_admitted_float_storage_types() {
        let f32_bytes = [1.25f32, -0.5]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let f16_bytes = [1.25f32, -0.5]
            .into_iter()
            .flat_map(|value| f16::from_f32(value).to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        let bf16_bytes = [1.25f32, -0.5]
            .into_iter()
            .flat_map(|value| bf16::from_f32(value).to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(
            decode_float_tensor(DType::F32, &f32_bytes, "f32").unwrap(),
            [1.25, -0.5]
        );
        assert_eq!(
            decode_float_tensor(DType::F16, &f16_bytes, "f16").unwrap(),
            [1.25, -0.5]
        );
        assert_eq!(
            decode_float_tensor(DType::BF16, &bf16_bytes, "bf16").unwrap(),
            [1.25, -0.5]
        );
    }
}
