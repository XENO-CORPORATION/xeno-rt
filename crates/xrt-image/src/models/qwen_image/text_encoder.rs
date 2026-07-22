use std::path::{Path, PathBuf};
#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use xrt_safetensors::HfModelBundle;
use xrt_safetensors::SafeTensorLayout;

#[cfg(feature = "cuda")]
use xrt_core::XrtError;
#[cfg(feature = "cuda")]
use xrt_runtime::{CudaResidentBackend, GpuResourceManager};

#[cfg(feature = "cuda")]
use super::QwenImageBundleConfig;
use super::QwenImageTokenBatch;
use crate::ImageError;
use crate::{BundleComponent, ComponentFormat, ComponentRole, ImageModelBundle};

/// Format-neutral prompt conditioning in row-major
/// `[batch, sequence, hidden]` order.
#[derive(Debug, Clone, PartialEq)]
pub struct QwenImagePromptEmbeddings {
    pub embeddings: Vec<f32>,
    pub attention_mask: Vec<u8>,
    pub retained_lengths: Vec<usize>,
    pub batch_size: usize,
    pub sequence_length: usize,
    pub hidden_size: usize,
}

impl QwenImagePromptEmbeddings {
    pub fn shape(&self) -> [usize; 3] {
        [self.batch_size, self.sequence_length, self.hidden_size]
    }
}

/// CUDA lifetime scope for the Qwen2.5-VL language backbone used by Qwen
/// Image. Drop this value after prompt encoding so a 24 GiB device can admit
/// the transformer component without retaining both weight sets.
#[cfg(feature = "cuda")]
pub struct QwenImageCudaTextEncoder {
    backend: CudaResidentBackend,
    hidden_size: usize,
    max_sequence_length: usize,
}

#[cfg(feature = "cuda")]
impl QwenImageCudaTextEncoder {
    pub fn load(
        bundle: &ImageModelBundle,
        resources: Arc<GpuResourceManager>,
    ) -> Result<Self, ImageError> {
        let config = QwenImageBundleConfig::load(bundle)?;
        Self::load_with_config(bundle, &config, resources)
    }

    pub fn load_with_config(
        bundle: &ImageModelBundle,
        config: &QwenImageBundleConfig,
        resources: Arc<GpuResourceManager>,
    ) -> Result<Self, ImageError> {
        let (component_root, layout) = text_encoder_layout(bundle)?;
        let model = HfModelBundle::open_exact(component_root, layout).map_err(|error| {
            ImageError::CorruptComponent(format!(
                "Qwen Image text encoder failed exact SafeTensors validation: {error}"
            ))
        })?;
        if model.config().hidden_size != config.text_encoder.hidden_size
            || model.config().vocab_size != config.text_encoder.vocab_size
            || model.config().num_hidden_layers != config.text_encoder.num_hidden_layers
        {
            return Err(ImageError::UnsupportedShape(
                "Qwen Image text encoder runtime geometry differs from the validated bundle config"
                    .to_string(),
            ));
        }
        let backend = CudaResidentBackend::from_hf_bundle_as_image_component(&model, resources)
            .map_err(map_cuda_load_error)?;
        Ok(Self {
            backend,
            hidden_size: config.text_encoder.hidden_size,
            max_sequence_length: config.max_sequence_length,
        })
    }

    pub fn encode_tokens(
        &self,
        tokens: &QwenImageTokenBatch,
    ) -> Result<QwenImagePromptEmbeddings, ImageError> {
        let valid_lengths = validate_token_batch(tokens, self.max_sequence_length)?;
        let mut encoded_rows = Vec::with_capacity(tokens.batch_size());
        for (row, valid_length) in tokens.input_ids.iter().zip(valid_lengths) {
            let encoded = self
                .backend
                .encode_standard_dense_hidden_states(&row[..valid_length])
                .map_err(map_cuda_execution_error)?;
            encoded_rows.push(encoded);
        }
        assemble_retained_embeddings(
            tokens,
            &encoded_rows,
            self.hidden_size,
            self.max_sequence_length,
        )
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

pub(super) fn text_encoder_layout(
    bundle: &ImageModelBundle,
) -> Result<(PathBuf, SafeTensorLayout), ImageError> {
    let components = bundle
        .manifest()
        .components
        .iter()
        .filter(|component| component.role == ComponentRole::TextEncoder)
        .collect::<Vec<_>>();
    let [component] = components.as_slice() else {
        return Err(ImageError::MissingComponent(format!(
            "expected exactly one text_encoder component, found {}",
            components.len()
        )));
    };
    if component.format != ComponentFormat::SafeTensors {
        return Err(ImageError::UnsupportedTensor(format!(
            "text_encoder component format `{}` is not safetensors",
            component.format.as_str()
        )));
    }
    text_encoder_layout_for_component(bundle.root(), component)
}

fn text_encoder_layout_for_component(
    bundle_root: &Path,
    component: &BundleComponent,
) -> Result<(PathBuf, SafeTensorLayout), ImageError> {
    let configs = component
        .files
        .iter()
        .filter(|file| {
            Path::new(&file.path)
                .file_name()
                .is_some_and(|name| name == "config.json")
        })
        .collect::<Vec<_>>();
    let [config] = configs.as_slice() else {
        return Err(ImageError::MissingComponent(format!(
            "text_encoder requires exactly one config.json, found {}",
            configs.len()
        )));
    };
    let relative_root = Path::new(&config.path)
        .parent()
        .unwrap_or_else(|| Path::new(""));
    if component.files.iter().any(|file| {
        Path::new(&file.path)
            .parent()
            .unwrap_or_else(|| Path::new(""))
            != relative_root
    }) {
        return Err(ImageError::CorruptComponent(
            "text_encoder files must share one component directory".to_string(),
        ));
    }

    let relative_to_component = |path: &str| -> Result<PathBuf, ImageError> {
        Path::new(path)
            .strip_prefix(relative_root)
            .map(Path::to_path_buf)
            .map_err(|_| {
                ImageError::CorruptComponent(format!(
                    "text_encoder file `{path}` is outside its component directory"
                ))
            })
    };
    let tensor_files = component
        .files
        .iter()
        .filter(|file| file.path.to_ascii_lowercase().ends_with(".safetensors"))
        .map(|file| relative_to_component(&file.path))
        .collect::<Result<Vec<_>, _>>()?;
    if tensor_files.is_empty() {
        return Err(ImageError::MissingComponent(
            "text_encoder declares no SafeTensors weights".to_string(),
        ));
    }
    let indexes = component
        .files
        .iter()
        .filter(|file| {
            file.path
                .to_ascii_lowercase()
                .ends_with(".safetensors.index.json")
        })
        .map(|file| relative_to_component(&file.path))
        .collect::<Result<Vec<_>, _>>()?;
    let layout = match indexes.as_slice() {
        [] if tensor_files.len() == 1 => SafeTensorLayout::single(tensor_files[0].clone()),
        [index] => SafeTensorLayout::indexed(index.clone(), tensor_files),
        [] => {
            return Err(ImageError::MissingComponent(
                "sharded text_encoder weights require one SafeTensors index".to_string(),
            ))
        }
        _ => {
            return Err(ImageError::CorruptComponent(format!(
                "text_encoder declares {} SafeTensors indexes",
                indexes.len()
            )))
        }
    };
    Ok((bundle_root.join(relative_root), layout))
}

pub(super) fn validate_token_batch(
    tokens: &QwenImageTokenBatch,
    max_sequence_length: usize,
) -> Result<Vec<usize>, ImageError> {
    if tokens.input_ids.is_empty() {
        return Err(ImageError::InvalidRequest(
            "Qwen Image token batch must not be empty".to_string(),
        ));
    }
    if tokens.attention_mask.len() != tokens.input_ids.len()
        || tokens.retained_lengths.len() != tokens.input_ids.len()
    {
        return Err(ImageError::InvalidRequest(
            "Qwen Image token batch row metadata does not match its batch size".to_string(),
        ));
    }
    let padded_length = tokens.input_ids[0].len();
    let retained_limit = max_sequence_length
        .checked_add(tokens.context_extension)
        .ok_or_else(|| {
            ImageError::UnsupportedShape("prompt retained length overflow".to_string())
        })?;
    let encoded_limit = retained_limit
        .checked_add(tokens.drop_tokens)
        .ok_or_else(|| {
            ImageError::UnsupportedShape("prompt encoded length overflow".to_string())
        })?;
    if padded_length > encoded_limit {
        return Err(ImageError::InputLimit(format!(
            "encoded prompt length {padded_length} exceeds Qwen Image limit {encoded_limit}"
        )));
    }

    let mut valid_lengths = Vec::with_capacity(tokens.input_ids.len());
    for (row_index, ((ids, mask), retained_length)) in tokens
        .input_ids
        .iter()
        .zip(&tokens.attention_mask)
        .zip(&tokens.retained_lengths)
        .enumerate()
    {
        if ids.len() != padded_length || mask.len() != padded_length {
            return Err(ImageError::InvalidRequest(format!(
                "Qwen Image token row {row_index} does not match padded length {padded_length}"
            )));
        }
        let valid_length = mask.iter().take_while(|value| **value == 1).count();
        if mask[..valid_length].iter().any(|value| *value != 1)
            || mask[valid_length..].iter().any(|value| *value != 0)
        {
            return Err(ImageError::InvalidRequest(format!(
                "Qwen Image attention mask row {row_index} is not right padded"
            )));
        }
        if tokens.drop_tokens == 0
            || valid_length <= tokens.drop_tokens
            || valid_length - tokens.drop_tokens != *retained_length
            || *retained_length > retained_limit
        {
            return Err(ImageError::InvalidRequest(format!(
                "Qwen Image retained length is inconsistent in row {row_index}"
            )));
        }
        valid_lengths.push(valid_length);
    }
    Ok(valid_lengths)
}

pub(super) fn assemble_retained_embeddings(
    tokens: &QwenImageTokenBatch,
    encoded_rows: &[Vec<f32>],
    hidden_size: usize,
    max_sequence_length: usize,
) -> Result<QwenImagePromptEmbeddings, ImageError> {
    if hidden_size == 0 {
        return Err(ImageError::UnsupportedShape(
            "Qwen Image text hidden size must be positive".to_string(),
        ));
    }
    let valid_lengths = validate_token_batch(tokens, max_sequence_length)?;
    if encoded_rows.len() != tokens.batch_size() {
        return Err(ImageError::Internal(
            "text encoder output batch size does not match token batch".to_string(),
        ));
    }
    let sequence_length = tokens.retained_lengths.iter().copied().max().unwrap_or(0);
    let embedding_len = tokens
        .batch_size()
        .checked_mul(sequence_length)
        .and_then(|value| value.checked_mul(hidden_size))
        .ok_or_else(|| {
            ImageError::UnsupportedShape("prompt embedding size overflow".to_string())
        })?;
    let mask_len = tokens
        .batch_size()
        .checked_mul(sequence_length)
        .ok_or_else(|| ImageError::UnsupportedShape("prompt mask size overflow".to_string()))?;
    let mut embeddings = vec![0.0f32; embedding_len];
    let mut attention_mask = vec![0u8; mask_len];

    for (row_index, ((encoded, valid_length), retained_length)) in encoded_rows
        .iter()
        .zip(valid_lengths)
        .zip(&tokens.retained_lengths)
        .enumerate()
    {
        let expected = valid_length.checked_mul(hidden_size).ok_or_else(|| {
            ImageError::UnsupportedShape("encoded text row size overflow".to_string())
        })?;
        if encoded.len() != expected {
            return Err(ImageError::Internal(format!(
                "text encoder row {row_index} has {} values, expected {expected}",
                encoded.len()
            )));
        }
        if encoded.iter().any(|value| !value.is_finite()) {
            return Err(ImageError::Numerical {
                component: "text_encoder",
                step: row_index,
            });
        }
        let source_start = tokens.drop_tokens.checked_mul(hidden_size).ok_or_else(|| {
            ImageError::UnsupportedShape("prompt drop offset overflow".to_string())
        })?;
        let source_end = valid_length.checked_mul(hidden_size).ok_or_else(|| {
            ImageError::UnsupportedShape("prompt source offset overflow".to_string())
        })?;
        let destination_start = row_index
            .checked_mul(sequence_length)
            .and_then(|value| value.checked_mul(hidden_size))
            .ok_or_else(|| {
                ImageError::UnsupportedShape("prompt destination offset overflow".to_string())
            })?;
        let destination_end = destination_start
            .checked_add(retained_length.checked_mul(hidden_size).ok_or_else(|| {
                ImageError::UnsupportedShape("prompt retained row size overflow".to_string())
            })?)
            .ok_or_else(|| {
                ImageError::UnsupportedShape("prompt destination end overflow".to_string())
            })?;
        embeddings[destination_start..destination_end]
            .copy_from_slice(&encoded[source_start..source_end]);
        let mask_start = row_index.checked_mul(sequence_length).ok_or_else(|| {
            ImageError::UnsupportedShape("prompt mask offset overflow".to_string())
        })?;
        attention_mask[mask_start..mask_start + retained_length].fill(1);
    }

    Ok(QwenImagePromptEmbeddings {
        embeddings,
        attention_mask,
        retained_lengths: tokens.retained_lengths.clone(),
        batch_size: tokens.batch_size(),
        sequence_length,
        hidden_size,
    })
}

#[cfg(feature = "cuda")]
fn map_cuda_load_error(error: XrtError) -> ImageError {
    match error {
        XrtError::Cuda(message)
            if ["memory", "vram", "budget", "allocation"]
                .iter()
                .any(|needle| message.to_ascii_lowercase().contains(needle)) =>
        {
            ImageError::InsufficientMemory(message)
        }
        XrtError::Cuda(message) | XrtError::Unsupported(message) => {
            ImageError::UnsupportedBackend(message)
        }
        XrtError::InvalidTensor(message)
        | XrtError::InvalidMetadata(message)
        | XrtError::InvalidFormat(message)
        | XrtError::Shape(message) => ImageError::UnsupportedTensor(message),
        other => ImageError::Execution(other.to_string()),
    }
}

#[cfg(feature = "cuda")]
fn map_cuda_execution_error(error: XrtError) -> ImageError {
    match error {
        XrtError::Cuda(message) => ImageError::Execution(format!("CUDA text encoder: {message}")),
        XrtError::Unsupported(message) => ImageError::UnsupportedBackend(message),
        other => ImageError::Execution(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token_batch() -> QwenImageTokenBatch {
        QwenImageTokenBatch {
            input_ids: vec![vec![1; 37], vec![2; 37]],
            attention_mask: vec![vec![1; 37], [vec![1; 36], vec![0]].concat()],
            retained_lengths: vec![3, 2],
            context_extension: 0,
            drop_tokens: 34,
        }
    }

    #[test]
    fn retained_embeddings_drop_template_states_and_right_pad() {
        let tokens = token_batch();
        let rows = tokens
            .retained_lengths
            .iter()
            .enumerate()
            .map(|(row, retained)| {
                let valid = retained + tokens.drop_tokens;
                (0..valid * 2)
                    .map(|index| (row * 1_000 + index) as f32)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let output = assemble_retained_embeddings(&tokens, &rows, 2, 512).unwrap();
        assert_eq!(output.shape(), [2, 3, 2]);
        assert_eq!(output.retained_lengths, vec![3, 2]);
        assert_eq!(output.attention_mask, vec![1, 1, 1, 1, 1, 0]);
        assert_eq!(output.embeddings[..6], rows[0][68..74]);
        assert_eq!(output.embeddings[6..10], rows[1][68..72]);
        assert_eq!(output.embeddings[10..12], [0.0, 0.0]);
    }

    #[test]
    fn token_batch_rejects_non_contiguous_attention_masks() {
        let mut tokens = token_batch();
        tokens.attention_mask[1][10] = 0;
        tokens.attention_mask[1][11] = 1;
        let error = validate_token_batch(&tokens, 512).unwrap_err();
        assert_eq!(error.kind(), crate::ImageErrorKind::InvalidRequest);
    }

    #[test]
    fn token_batch_admits_model_injected_visual_context_beyond_text_limit() {
        let mut tokens = token_batch();
        tokens.input_ids = vec![vec![1; 1_164]];
        tokens.attention_mask = vec![vec![1; 1_164]];
        tokens.retained_lengths = vec![1_100];
        tokens.context_extension = 3 * 196;
        tokens.drop_tokens = 64;

        assert_eq!(validate_token_batch(&tokens, 512).unwrap(), vec![1_164]);
        tokens.context_extension = 0;
        assert_eq!(
            validate_token_batch(&tokens, 512).unwrap_err().kind(),
            crate::ImageErrorKind::InputLimit
        );
    }
}
