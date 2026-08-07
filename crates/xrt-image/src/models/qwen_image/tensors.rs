use std::{collections::BTreeSet, path::Path};

use xrt_core::DType;
use xrt_gguf::{GgufCompatibility, GgufFile, QWEN_IMAGE_EDIT_TIMESTEP_ZERO_MARKER};
use xrt_safetensors::{SafeTensorDType, SafeTensorLayout, SafeTensorStore};

use crate::{ComponentFormat, ComponentRole, ImageError, ImageModelBundle};

use super::{
    QwenImageDistilledProfile, QwenImageLoraAdapter, QwenImageTransformerConfig,
    QWEN_IMAGE_2512_LIGHTNING_4STEP_BF16_FILE,
};

/// One configuration-derived tensor required by the Qwen Image transformer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectedTensor {
    pub name: String,
    pub shape: Vec<usize>,
}

impl ExpectedTensor {
    pub(crate) fn new(name: impl Into<String>, shape: impl Into<Vec<usize>>) -> Self {
        Self {
            name: name.into(),
            shape: shape.into(),
        }
    }
}

/// Derive the complete official Diffusers SafeTensors schema from the model
/// configuration. Dimensions are deliberately not hard-coded to the 2512
/// checkpoint, so a future compatible checkpoint must still prove its own
/// internal geometry before any weights are admitted.
pub fn expected_transformer_tensors(
    config: &QwenImageTransformerConfig,
) -> Result<Vec<ExpectedTensor>, ImageError> {
    config.validate()?;

    let inner = config.inner_dim()?;
    let feed_forward = inner.checked_mul(4).ok_or_else(|| {
        ImageError::UnsupportedShape("transformer feed-forward dimension overflow".to_string())
    })?;
    let modulation = inner.checked_mul(6).ok_or_else(|| {
        ImageError::UnsupportedShape("transformer modulation dimension overflow".to_string())
    })?;
    let output = config
        .out_channels
        .checked_mul(config.patch_size)
        .and_then(|value| value.checked_mul(config.patch_size))
        .ok_or_else(|| {
            ImageError::UnsupportedShape("transformer output dimension overflow".to_string())
        })?;

    let per_block = 32usize;
    let capacity = config
        .num_layers
        .checked_mul(per_block)
        .and_then(|value| value.checked_add(13))
        .ok_or_else(|| {
            ImageError::UnsupportedShape("transformer tensor count overflow".to_string())
        })?;
    let mut tensors = Vec::with_capacity(capacity);

    tensors.extend([
        ExpectedTensor::new("img_in.bias", vec![inner]),
        ExpectedTensor::new("img_in.weight", vec![inner, config.in_channels]),
        ExpectedTensor::new("norm_out.linear.bias", vec![inner * 2]),
        ExpectedTensor::new("norm_out.linear.weight", vec![inner * 2, inner]),
        ExpectedTensor::new("proj_out.bias", vec![output]),
        ExpectedTensor::new("proj_out.weight", vec![output, inner]),
        ExpectedTensor::new(
            "time_text_embed.timestep_embedder.linear_1.bias",
            vec![inner],
        ),
        ExpectedTensor::new(
            "time_text_embed.timestep_embedder.linear_1.weight",
            vec![inner, 256],
        ),
        ExpectedTensor::new(
            "time_text_embed.timestep_embedder.linear_2.bias",
            vec![inner],
        ),
        ExpectedTensor::new(
            "time_text_embed.timestep_embedder.linear_2.weight",
            vec![inner, inner],
        ),
        ExpectedTensor::new("txt_in.bias", vec![inner]),
        ExpectedTensor::new("txt_in.weight", vec![inner, config.joint_attention_dim]),
        ExpectedTensor::new("txt_norm.weight", vec![config.joint_attention_dim]),
    ]);

    for layer in 0..config.num_layers {
        let prefix = format!("transformer_blocks.{layer}");
        for projection in ["add_k_proj", "add_q_proj", "add_v_proj"] {
            tensors.push(ExpectedTensor::new(
                format!("{prefix}.attn.{projection}.bias"),
                vec![inner],
            ));
            tensors.push(ExpectedTensor::new(
                format!("{prefix}.attn.{projection}.weight"),
                vec![inner, inner],
            ));
        }
        for normalization in ["norm_added_k", "norm_added_q", "norm_k", "norm_q"] {
            tensors.push(ExpectedTensor::new(
                format!("{prefix}.attn.{normalization}.weight"),
                vec![config.attention_head_dim],
            ));
        }
        tensors.extend([
            ExpectedTensor::new(format!("{prefix}.attn.to_add_out.bias"), vec![inner]),
            ExpectedTensor::new(
                format!("{prefix}.attn.to_add_out.weight"),
                vec![inner, inner],
            ),
            ExpectedTensor::new(format!("{prefix}.attn.to_k.bias"), vec![inner]),
            ExpectedTensor::new(format!("{prefix}.attn.to_k.weight"), vec![inner, inner]),
            ExpectedTensor::new(format!("{prefix}.attn.to_out.0.bias"), vec![inner]),
            ExpectedTensor::new(format!("{prefix}.attn.to_out.0.weight"), vec![inner, inner]),
            ExpectedTensor::new(format!("{prefix}.attn.to_q.bias"), vec![inner]),
            ExpectedTensor::new(format!("{prefix}.attn.to_q.weight"), vec![inner, inner]),
            ExpectedTensor::new(format!("{prefix}.attn.to_v.bias"), vec![inner]),
            ExpectedTensor::new(format!("{prefix}.attn.to_v.weight"), vec![inner, inner]),
        ]);
        for stream in ["img", "txt"] {
            tensors.extend([
                ExpectedTensor::new(
                    format!("{prefix}.{stream}_mlp.net.0.proj.bias"),
                    vec![feed_forward],
                ),
                ExpectedTensor::new(
                    format!("{prefix}.{stream}_mlp.net.0.proj.weight"),
                    vec![feed_forward, inner],
                ),
                ExpectedTensor::new(format!("{prefix}.{stream}_mlp.net.2.bias"), vec![inner]),
                ExpectedTensor::new(
                    format!("{prefix}.{stream}_mlp.net.2.weight"),
                    vec![inner, feed_forward],
                ),
                ExpectedTensor::new(format!("{prefix}.{stream}_mod.1.bias"), vec![modulation]),
                ExpectedTensor::new(
                    format!("{prefix}.{stream}_mod.1.weight"),
                    vec![modulation, inner],
                ),
            ]);
        }
    }

    debug_assert_eq!(tensors.len(), capacity);
    Ok(tensors)
}

/// Validate the complete official BF16/F16 Diffusers transformer component
/// before accelerator allocation. Unknown names are rejected as firmly as
/// missing tensors because accepting either can hide an incompatible graph or
/// a partially converted checkpoint.
pub fn validate_transformer_safetensors(
    store: &SafeTensorStore,
    config: &QwenImageTransformerConfig,
) -> Result<(), ImageError> {
    let expected = expected_transformer_tensors(config)?;
    let expected_names = expected
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect::<BTreeSet<_>>();

    for tensor in &expected {
        let info = store.tensor_info(&tensor.name).ok_or_else(|| {
            ImageError::UnsupportedTensor(format!(
                "Qwen Image transformer is missing `{}`",
                tensor.name
            ))
        })?;
        if !matches!(info.dtype, SafeTensorDType::Bf16 | SafeTensorDType::F16) {
            return Err(ImageError::UnsupportedTensor(format!(
                "Qwen Image transformer tensor `{}` has unsupported SafeTensors dtype {:?}",
                tensor.name, info.dtype
            )));
        }
        if info.shape != tensor.shape {
            return Err(ImageError::UnsupportedShape(format!(
                "Qwen Image transformer tensor `{}` has shape {:?}, expected {:?}",
                tensor.name, info.shape, tensor.shape
            )));
        }
    }

    if let Some(unknown) = store
        .tensor_names()
        .find(|name| !expected_names.contains(name))
    {
        return Err(ImageError::UnsupportedTensor(format!(
            "Qwen Image transformer contains unknown tensor `{unknown}`"
        )));
    }
    if store.tensor_count() != expected.len() {
        return Err(ImageError::UnsupportedTensor(format!(
            "Qwen Image transformer contains {} tensors, expected {}",
            store.tensor_count(),
            expected.len()
        )));
    }
    Ok(())
}

/// Open the exact SafeTensors files declared for the transformer role and
/// validate their complete Qwen Image schema.
pub fn open_transformer_safetensors(
    bundle: &ImageModelBundle,
    config: &QwenImageTransformerConfig,
) -> Result<SafeTensorStore, ImageError> {
    let components = bundle
        .manifest()
        .components
        .iter()
        .filter(|component| component.role == ComponentRole::Transformer)
        .collect::<Vec<_>>();
    let [component] = components.as_slice() else {
        return Err(ImageError::MissingComponent(format!(
            "expected exactly one transformer component, found {}",
            components.len()
        )));
    };
    if component.format != ComponentFormat::SafeTensors {
        return Err(ImageError::UnsupportedTensor(format!(
            "transformer component format `{}` is not safetensors",
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
                    "unindexed transformer must declare exactly one SafeTensors file, found {}",
                    tensor_files.len()
                )));
            };
            SafeTensorLayout::single(*file)
        }
        [index] => SafeTensorLayout::indexed(index.path.as_str(), tensor_files),
        _ => {
            return Err(ImageError::CorruptComponent(format!(
                "transformer declares {} SafeTensors indexes",
                indexes.len()
            )))
        }
    };
    let store = SafeTensorStore::open_exact(bundle.root(), layout).map_err(|error| {
        ImageError::CorruptComponent(format!(
            "transformer SafeTensors component failed validation: {error}"
        ))
    })?;
    validate_transformer_safetensors(&store, config)?;
    Ok(store)
}

/// Open an optional, exact Qwen Image transformer adapter. The first admitted
/// adapter profile is intentionally narrow: the official BF16 rank-64
/// Qwen-Image-2512 Lightning V1.0 adapter with its 4-step, CFG-1 contract.
pub fn open_transformer_adapter(
    bundle: &ImageModelBundle,
    config: &QwenImageTransformerConfig,
) -> Result<Option<QwenImageLoraAdapter>, ImageError> {
    let components = bundle
        .manifest()
        .components
        .iter()
        .filter(|component| component.role == ComponentRole::TransformerAdapter)
        .collect::<Vec<_>>();
    let component = match components.as_slice() {
        [] => return Ok(None),
        [component] => *component,
        _ => {
            return Err(ImageError::MissingComponent(format!(
                "expected at most one transformer adapter component, found {}",
                components.len()
            )))
        }
    };
    if bundle.manifest().family != "qwen-image" {
        return Err(ImageError::UnsupportedCapability(
            "the admitted Qwen Image Lightning adapter is generation-only".to_string(),
        ));
    }
    if component.format != ComponentFormat::SafeTensors {
        return Err(ImageError::UnsupportedTensor(format!(
            "transformer adapter format `{}` is not safetensors",
            component.format.as_str()
        )));
    }
    let tensor_files = component
        .files
        .iter()
        .filter(|file| file.path.to_ascii_lowercase().ends_with(".safetensors"))
        .collect::<Vec<_>>();
    let [file] = tensor_files.as_slice() else {
        return Err(ImageError::MissingComponent(format!(
            "transformer adapter must declare exactly one SafeTensors file, found {}",
            tensor_files.len()
        )));
    };
    let file_name = Path::new(&file.path)
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| {
            ImageError::CorruptComponent(
                "transformer adapter path does not have a UTF-8 file name".to_string(),
            )
        })?;
    let profile = match file_name {
        QWEN_IMAGE_2512_LIGHTNING_4STEP_BF16_FILE => QwenImageDistilledProfile {
            steps: 4,
            true_cfg_scale: 1.0,
        },
        other => {
            return Err(ImageError::UnsupportedCapability(format!(
                "transformer adapter `{other}` has no admitted inference profile"
            )))
        }
    };
    let store =
        SafeTensorStore::open_exact(bundle.root(), SafeTensorLayout::single(file.path.as_str()))
            .map_err(|error| {
                ImageError::CorruptComponent(format!(
                    "transformer adapter SafeTensors component failed validation: {error}"
                ))
            })?;
    QwenImageLoraAdapter::from_store(store, config, profile).map(Some)
}

/// Validate a Qwen Image GGUF transformer against the complete configuration-
/// derived name and shape map plus the supported mixed-quantization policy.
pub fn validate_transformer_gguf(
    file: &GgufFile,
    config: &QwenImageTransformerConfig,
    quantization: &str,
) -> Result<(), ImageError> {
    if file.metadata_string("general.architecture") != Some("qwen_image") {
        return Err(ImageError::UnsupportedTensor(format!(
            "GGUF general.architecture is {:?}, expected qwen_image",
            file.metadata_string("general.architecture")
        )));
    }
    if file.metadata_usize("general.quantization_version") != Some(2) {
        return Err(ImageError::UnsupportedQuantization(format!(
            "GGUF general.quantization_version is {:?}, expected 2",
            file.metadata_usize("general.quantization_version")
        )));
    }

    let expected = expected_transformer_tensors(config)?;
    let expected_names = expected
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect::<BTreeSet<_>>();
    let has_timestep_zero_marker = file
        .tensor_info(QWEN_IMAGE_EDIT_TIMESTEP_ZERO_MARKER)
        .is_some();
    if has_timestep_zero_marker && !config.zero_cond_t {
        return Err(ImageError::UnsupportedTensor(format!(
            "Qwen Image GGUF transformer contains `{QWEN_IMAGE_EDIT_TIMESTEP_ZERO_MARKER}` but zero_cond_t is false"
        )));
    }
    let nominal = nominal_gguf_dtype(quantization)?;
    let mut saw_nominal = false;
    for tensor in &expected {
        let info = file.tensor_info(&tensor.name).ok_or_else(|| {
            ImageError::UnsupportedTensor(format!(
                "Qwen Image GGUF transformer is missing `{}`",
                tensor.name
            ))
        })?;
        let expected_dimensions = tensor.shape.iter().rev().copied().collect::<Vec<_>>();
        if info.dimensions != expected_dimensions {
            return Err(ImageError::UnsupportedShape(format!(
                "Qwen Image GGUF tensor `{}` has dimensions {:?}, expected GGML order {:?}",
                tensor.name, info.dimensions, expected_dimensions
            )));
        }
        if !gguf_tensor_dtype_allowed(quantization, tensor.shape.len(), info.dtype)? {
            return Err(ImageError::UnsupportedQuantization(format!(
                "Qwen Image {quantization} tensor `{}` uses disallowed dtype {:?}",
                tensor.name, info.dtype
            )));
        }
        saw_nominal |= info.dtype == nominal;
    }

    if let Some(unknown) = file.tensor_names().find(|name| {
        !(expected_names.contains(name)
            || config.zero_cond_t && *name == QWEN_IMAGE_EDIT_TIMESTEP_ZERO_MARKER)
    }) {
        return Err(ImageError::UnsupportedTensor(format!(
            "Qwen Image GGUF transformer contains unknown tensor `{unknown}`"
        )));
    }
    let expected_file_tensors = expected
        .len()
        .checked_add(usize::from(has_timestep_zero_marker))
        .ok_or_else(|| {
            ImageError::UnsupportedTensor("Qwen Image GGUF tensor count overflowed".to_string())
        })?;
    if file.tensor_infos().len() != expected_file_tensors {
        return Err(ImageError::UnsupportedTensor(format!(
            "Qwen Image GGUF transformer contains {} tensors, expected {} model tensors plus {} compatibility marker(s)",
            file.tensor_infos().len(),
            expected.len(),
            usize::from(has_timestep_zero_marker)
        )));
    }
    if !saw_nominal {
        return Err(ImageError::UnsupportedQuantization(format!(
            "Qwen Image GGUF labeled {quantization} contains no {nominal:?} tensors"
        )));
    }
    Ok(())
}

/// Open and validate the single GGUF file declared for the transformer role.
pub fn open_transformer_gguf(
    bundle: &ImageModelBundle,
    config: &QwenImageTransformerConfig,
) -> Result<GgufFile, ImageError> {
    let components = bundle
        .manifest()
        .components
        .iter()
        .filter(|component| component.role == ComponentRole::Transformer)
        .collect::<Vec<_>>();
    let [component] = components.as_slice() else {
        return Err(ImageError::MissingComponent(format!(
            "expected exactly one transformer component, found {}",
            components.len()
        )));
    };
    if component.format != ComponentFormat::Gguf {
        return Err(ImageError::UnsupportedTensor(format!(
            "transformer component format `{}` is not gguf",
            component.format.as_str()
        )));
    }
    let files = component
        .files
        .iter()
        .filter(|file| file.path.to_ascii_lowercase().ends_with(".gguf"))
        .collect::<Vec<_>>();
    let [record] = files.as_slice() else {
        return Err(ImageError::MissingComponent(format!(
            "GGUF transformer must declare exactly one .gguf file, found {}",
            files.len()
        )));
    };
    let path = bundle.root().join(&record.path);
    let file = if config.zero_cond_t {
        GgufFile::open_with_compatibility(path, GgufCompatibility::QwenImageEditTimestepZero)
    } else {
        GgufFile::open(path)
    }
    .map_err(|error| {
        ImageError::CorruptComponent(format!("transformer GGUF failed validation: {error}"))
    })?;
    validate_transformer_gguf(&file, config, &bundle.manifest().quantization)?;
    Ok(file)
}

fn nominal_gguf_dtype(quantization: &str) -> Result<DType, ImageError> {
    match quantization {
        "Q8_0" => Ok(DType::Q8_0),
        "Q6_K" => Ok(DType::Q6_K),
        "Q5_K_M" => Ok(DType::Q5_K),
        "Q4_K_M" => Ok(DType::Q4_K),
        other => Err(ImageError::UnsupportedQuantization(other.to_string())),
    }
}

fn gguf_dtype_allowed(quantization: &str, dtype: DType) -> Result<bool, ImageError> {
    let allowed = match quantization {
        "Q8_0" => matches!(dtype, DType::F32 | DType::F16 | DType::BF16 | DType::Q8_0),
        "Q6_K" => matches!(
            dtype,
            DType::F32 | DType::F16 | DType::BF16 | DType::Q8_0 | DType::Q6_K
        ),
        "Q5_K_M" => matches!(
            dtype,
            DType::F32 | DType::F16 | DType::BF16 | DType::Q8_0 | DType::Q6_K | DType::Q5_K
        ),
        "Q4_K_M" => matches!(
            dtype,
            DType::F32
                | DType::F16
                | DType::BF16
                | DType::Q8_0
                | DType::Q6_K
                | DType::Q5_K
                | DType::Q4_K
        ),
        other => return Err(ImageError::UnsupportedQuantization(other.to_string())),
    };
    Ok(allowed)
}

fn gguf_tensor_dtype_allowed(
    quantization: &str,
    rank: usize,
    dtype: DType,
) -> Result<bool, ImageError> {
    if rank == 1 {
        // Biases, modulation vectors, and normalization scales are decoded
        // directly to f32 by the native executor. Reject quantized auxiliary
        // tensors during admission instead of failing later during execution.
        return Ok(matches!(dtype, DType::F32 | DType::F16 | DType::BF16));
    }
    gguf_dtype_allowed(quantization, dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pinned_config() -> QwenImageTransformerConfig {
        QwenImageTransformerConfig::from_json_bytes(
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
        .unwrap()
    }

    #[test]
    fn pinned_2512_schema_has_every_official_tensor() {
        let tensors = expected_transformer_tensors(&pinned_config()).unwrap();
        assert_eq!(tensors.len(), 1_933);
        assert_eq!(
            tensors
                .iter()
                .find(|tensor| tensor.name == "norm_out.linear.weight")
                .unwrap()
                .shape,
            vec![6_144, 3_072]
        );
        assert_eq!(
            tensors
                .iter()
                .find(|tensor| { tensor.name == "transformer_blocks.59.txt_mlp.net.0.proj.weight" })
                .unwrap()
                .shape,
            vec![12_288, 3_072]
        );
    }

    #[test]
    fn schema_names_are_unique() {
        let tensors = expected_transformer_tensors(&pinned_config()).unwrap();
        let names = tensors
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(names.len(), tensors.len());
    }

    #[test]
    fn mixed_quantization_tiers_only_admit_equal_or_higher_precision_types() {
        assert!(gguf_dtype_allowed("Q4_K_M", DType::Q6_K).unwrap());
        assert!(gguf_dtype_allowed("Q5_K_M", DType::Q5_K).unwrap());
        assert!(!gguf_dtype_allowed("Q5_K_M", DType::Q4_K).unwrap());
        assert!(!gguf_dtype_allowed("Q8_0", DType::Q6_K).unwrap());
        assert!(!gguf_dtype_allowed("Q4_K_M", DType::Q4_0).unwrap());
    }

    #[test]
    fn quantized_auxiliary_tensors_are_rejected_during_admission() {
        assert!(gguf_tensor_dtype_allowed("Q4_K_M", 1, DType::F16).unwrap());
        assert!(!gguf_tensor_dtype_allowed("Q4_K_M", 1, DType::Q4_K).unwrap());
        assert!(gguf_tensor_dtype_allowed("Q4_K_M", 2, DType::Q4_K).unwrap());
    }
}
