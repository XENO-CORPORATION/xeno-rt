//! `xrt-embedding` — CPU-first, deterministic text embedding inference.
//!
//! This domain owns model/tokenizer integrity, task prefixes, pooling, dimensional
//! projection, and normalization. Consumers own indexing and retrieval policy.

use std::{
    fs::File,
    io::Read as _,
    path::{Path, PathBuf},
};

use ndarray::Array2;
use ort::{
    execution_providers::CPUExecutionProvider,
    session::{builder::GraphOptimizationLevel, Session},
};
use parking_lot::Mutex;
use serde::Serialize;
use sha2::{Digest, Sha256};
use tokenizers::{Tokenizer, TruncationDirection, TruncationParams, TruncationStrategy};

pub const MODEL_ID: &str = "nomic-ai/nomic-embed-text-v1.5";
pub const MODEL_REVISION: &str = "a15734e81021ea6c92b09050d2c7085001db8f36";
pub const MODEL_FILE: &str = "model_quantized.onnx";
pub const TOKENIZER_FILE: &str = "tokenizer.json";
pub const MODEL_SHA256: &str = "b4342336debaea79de872370664b0aaeb67dea4605513d00ee236ea871a81f27";
pub const TOKENIZER_SHA256: &str =
    "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66";
pub const MODEL_SIZE_BYTES: u64 = 137_296_292;
pub const TOKENIZER_SIZE_BYTES: u64 = 711_396;
#[cfg(windows)]
pub const ONNX_RUNTIME_FILE: &str = "onnxruntime.dll";
#[cfg(windows)]
pub const ONNX_RUNTIME_SIZE_BYTES: u64 = 11_567_648;
#[cfg(windows)]
pub const ONNX_RUNTIME_SHA256: &str =
    "52f8ebe8f08f369a44fed6d1cb680c7c89169795e1c2949ee25b88b538ef0948";
pub const NATIVE_DIMENSIONS: usize = 768;
pub const OUTPUT_DIMENSIONS: usize = 512;
pub const MAX_SEQUENCE_LENGTH: usize = 8192;
pub const MAX_BATCH_SIZE: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingTask {
    Query,
    Document,
}
impl EmbeddingTask {
    pub fn prefix(self) -> &'static str {
        match self {
            Self::Query => "search_query: ",
            Self::Document => "search_document: ",
        }
    }
}

#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    pub bundle_dir: PathBuf,
    pub max_sequence_length: usize,
    pub intra_threads: usize,
}

impl EmbeddingConfig {
    pub fn from_bundle_dir(bundle_dir: impl Into<PathBuf>) -> Self {
        Self {
            bundle_dir: bundle_dir.into(),
            max_sequence_length: MAX_SEQUENCE_LENGTH,
            intra_threads: 4,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingContract {
    pub model_id: &'static str,
    pub revision: &'static str,
    pub native_dimensions: usize,
    pub output_dimensions: usize,
    pub normalization: &'static str,
    pub pooling: &'static str,
    pub query_prefix: &'static str,
    pub document_prefix: &'static str,
}

pub fn contract() -> EmbeddingContract {
    EmbeddingContract {
        model_id: MODEL_ID,
        revision: MODEL_REVISION,
        native_dimensions: NATIVE_DIMENSIONS,
        output_dimensions: OUTPUT_DIMENSIONS,
        normalization: "layer_norm_768_then_truncate_512_then_l2",
        pooling: "attention_mask_mean",
        query_prefix: EmbeddingTask::Query.prefix(),
        document_prefix: EmbeddingTask::Document.prefix(),
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("embedding model bundle is invalid: {0}")]
    InvalidBundle(String),
    #[error("embedding request is invalid: {0}")]
    InvalidRequest(String),
    #[error("embedding tokenizer failed: {0}")]
    Tokenizer(String),
    #[error("embedding inference failed: {0}")]
    Inference(String),
}

pub struct EmbeddingRuntime {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    max_sequence_length: usize,
}

impl EmbeddingRuntime {
    pub fn load(config: EmbeddingConfig) -> Result<Self, EmbeddingError> {
        std::thread::Builder::new()
            .name("xrt-embedding-loader".to_string())
            .stack_size(32 * 1024 * 1024)
            .spawn(move || Self::load_on_current_thread(config))
            .map_err(|error| {
                EmbeddingError::Inference(format!("cannot start model loader: {error}"))
            })?
            .join()
            .map_err(|_| EmbeddingError::Inference("model loader panicked".to_string()))?
    }

    fn load_on_current_thread(config: EmbeddingConfig) -> Result<Self, EmbeddingError> {
        tracing::info!(bundle_dir = %config.bundle_dir.display(), "initializing embedding runtime");
        initialize_onnx_runtime()?;
        tracing::info!("verified and initialized the packaged ONNX Runtime companion");
        if !(1..=MAX_SEQUENCE_LENGTH).contains(&config.max_sequence_length) {
            return Err(EmbeddingError::InvalidBundle(format!(
                "max sequence length must be between 1 and {MAX_SEQUENCE_LENGTH}"
            )));
        }
        let model_path = config.bundle_dir.join(MODEL_FILE);
        let tokenizer_path = config.bundle_dir.join(TOKENIZER_FILE);
        verify_file(&model_path, MODEL_SHA256)?;
        verify_file(&tokenizer_path, TOKENIZER_SHA256)?;
        tracing::info!("verified embedding model and tokenizer integrity");

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|error| EmbeddingError::Tokenizer(error.to_string()))?;
        tracing::info!("loaded embedding tokenizer");
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: config.max_sequence_length,
                strategy: TruncationStrategy::LongestFirst,
                stride: 0,
                direction: TruncationDirection::Right,
            }))
            .map_err(|error| EmbeddingError::Tokenizer(error.to_string()))?;

        let session = Session::builder()
            .map_err(inference_error)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(inference_error)?
            .with_memory_pattern(true)
            .map_err(inference_error)?
            .with_intra_threads(config.intra_threads.max(1))
            .map_err(inference_error)?
            .with_execution_providers([CPUExecutionProvider::default().build()])
            .map_err(inference_error)?
            .commit_from_file(model_path)
            .map_err(inference_error)?;
        tracing::info!("loaded embedding ONNX session");

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            max_sequence_length: config.max_sequence_length,
        })
    }

    pub fn max_sequence_length(&self) -> usize {
        self.max_sequence_length
    }

    pub fn embed(
        &self,
        texts: &[String],
        task: EmbeddingTask,
    ) -> Result<(Vec<Vec<f32>>, usize), EmbeddingError> {
        if texts.is_empty() || texts.len() > MAX_BATCH_SIZE {
            return Err(EmbeddingError::InvalidRequest(format!(
                "batch size must be between 1 and {MAX_BATCH_SIZE}"
            )));
        }
        if texts.iter().any(|text| text.len() > 1_048_576) {
            return Err(EmbeddingError::InvalidRequest(
                "an input exceeds the 1 MiB UTF-8 limit".to_string(),
            ));
        }
        let prefixed = texts
            .iter()
            .map(|text| format!("{}{text}", task.prefix()))
            .collect::<Vec<_>>();
        let encodings = self
            .tokenizer
            .encode_batch(prefixed, true)
            .map_err(|error| EmbeddingError::Tokenizer(error.to_string()))?;
        let sequence_length = encodings
            .iter()
            .map(|encoding| encoding.len())
            .max()
            .unwrap_or(0);
        if sequence_length == 0 {
            return Err(EmbeddingError::InvalidRequest(
                "tokenizer produced an empty sequence".to_string(),
            ));
        }

        let batch = encodings.len();
        let mut input_ids = Array2::<i64>::zeros((batch, sequence_length));
        let mut token_type_ids = Array2::<i64>::zeros((batch, sequence_length));
        let mut attention_mask = Array2::<i64>::zeros((batch, sequence_length));
        let mut prompt_tokens = 0usize;
        for (row, encoding) in encodings.iter().enumerate() {
            prompt_tokens += encoding.len();
            for (column, token_id) in encoding.get_ids().iter().copied().enumerate() {
                input_ids[(row, column)] = i64::from(token_id);
                token_type_ids[(row, column)] = i64::from(encoding.get_type_ids()[column]);
                attention_mask[(row, column)] = 1;
            }
        }

        let session = self.session.lock();
        let outputs = session
            .run(
                ort::inputs! {
                    "input_ids" => input_ids.view(),
                    "token_type_ids" => token_type_ids.view(),
                    "attention_mask" => attention_mask.view()
                }
                .map_err(inference_error)?,
            )
            .map_err(inference_error)?;
        let output = outputs.get("last_hidden_state").ok_or_else(|| {
            EmbeddingError::Inference("model returned no last_hidden_state".to_string())
        })?;
        let (shape, values) = output
            .try_extract_raw_tensor::<f32>()
            .map_err(inference_error)?;
        let shape = shape
            .iter()
            .map(|value| *value as usize)
            .collect::<Vec<_>>();
        if shape.as_slice() != [batch, sequence_length, NATIVE_DIMENSIONS] {
            return Err(EmbeddingError::Inference(format!(
                "unexpected last_hidden_state shape {shape:?}"
            )));
        }
        let embeddings = pool_project_normalize(values, batch, sequence_length, &attention_mask)?;
        Ok((embeddings, prompt_tokens))
    }
}

#[cfg(windows)]
fn initialize_onnx_runtime() -> Result<(), EmbeddingError> {
    let executable = std::env::current_exe().map_err(|error| {
        EmbeddingError::InvalidBundle(format!("cannot resolve the XRT executable: {error}"))
    })?;
    let companion = executable
        .parent()
        .ok_or_else(|| {
            EmbeddingError::InvalidBundle("XRT executable has no parent directory".to_string())
        })?
        .join(ONNX_RUNTIME_FILE);
    let metadata = companion.metadata().map_err(|error| {
        EmbeddingError::InvalidBundle(format!(
            "required companion `{}` is unavailable: {error}",
            companion.display()
        ))
    })?;
    if metadata.len() != ONNX_RUNTIME_SIZE_BYTES {
        return Err(EmbeddingError::InvalidBundle(format!(
            "companion `{}` has {} bytes, expected {ONNX_RUNTIME_SIZE_BYTES}",
            companion.display(),
            metadata.len()
        )));
    }
    verify_file(&companion, ONNX_RUNTIME_SHA256)?;
    ort::init_from(companion.display().to_string())
        .with_name("xrt")
        .commit()
        .map_err(inference_error)?;
    Ok(())
}

#[cfg(not(windows))]
fn initialize_onnx_runtime() -> Result<(), EmbeddingError> {
    Ok(())
}

fn verify_file(path: &Path, expected_sha256: &str) -> Result<(), EmbeddingError> {
    let mut file = File::open(path).map_err(|error| {
        EmbeddingError::InvalidBundle(format!("cannot open `{}`: {error}", path.display()))
    })?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|error| {
            EmbeddingError::InvalidBundle(format!("cannot read `{}`: {error}", path.display()))
        })?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    let actual = format!("{:x}", digest.finalize());
    if actual != expected_sha256 {
        return Err(EmbeddingError::InvalidBundle(format!(
            "SHA-256 mismatch for `{}`: expected {expected_sha256}, found {actual}",
            path.display()
        )));
    }
    Ok(())
}

fn pool_project_normalize(
    hidden: &[f32],
    batch: usize,
    sequence_length: usize,
    attention_mask: &Array2<i64>,
) -> Result<Vec<Vec<f32>>, EmbeddingError> {
    let expected = batch
        .checked_mul(sequence_length)
        .and_then(|value| value.checked_mul(NATIVE_DIMENSIONS))
        .ok_or_else(|| EmbeddingError::Inference("hidden-state shape overflow".to_string()))?;
    if hidden.len() != expected {
        return Err(EmbeddingError::Inference(format!(
            "hidden-state length mismatch: expected {expected}, found {}",
            hidden.len()
        )));
    }
    let mut result = Vec::with_capacity(batch);
    for row in 0..batch {
        let mut pooled = vec![0.0f32; NATIVE_DIMENSIONS];
        let mut tokens = 0usize;
        for column in 0..sequence_length {
            if attention_mask[(row, column)] == 0 {
                continue;
            }
            tokens += 1;
            let start = (row * sequence_length + column) * NATIVE_DIMENSIONS;
            for (dimension, value) in pooled.iter_mut().enumerate() {
                *value += hidden[start + dimension];
            }
        }
        if tokens == 0 {
            return Err(EmbeddingError::Inference(
                "attention mask has no live tokens".to_string(),
            ));
        }
        let inverse_tokens = 1.0 / tokens as f32;
        for value in &mut pooled {
            *value *= inverse_tokens;
        }

        let mean = pooled.iter().copied().sum::<f32>() / NATIVE_DIMENSIONS as f32;
        let variance = pooled
            .iter()
            .map(|value| {
                let centered = *value - mean;
                centered * centered
            })
            .sum::<f32>()
            / NATIVE_DIMENSIONS as f32;
        let inverse_std = 1.0 / (variance + 1e-5).sqrt();
        let mut projected = pooled[..OUTPUT_DIMENSIONS]
            .iter()
            .map(|value| (*value - mean) * inverse_std)
            .collect::<Vec<_>>();
        let norm = projected
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        if !norm.is_finite() || norm <= f32::EPSILON {
            return Err(EmbeddingError::Inference(
                "embedding norm is invalid".to_string(),
            ));
        }
        for value in &mut projected {
            *value /= norm;
        }
        if projected.iter().any(|value| !value.is_finite()) {
            return Err(EmbeddingError::Inference(
                "embedding contains non-finite values".to_string(),
            ));
        }
        result.push(projected);
    }
    Ok(result)
}

fn inference_error(error: impl std::fmt::Display) -> EmbeddingError {
    EmbeddingError::Inference(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct ReleaseManifest {
        model_id: String,
        revision: String,
        contract: ReleaseContract,
        artifacts: Vec<ReleaseArtifact>,
    }

    #[derive(Deserialize)]
    struct ReleaseContract {
        native_dimensions: usize,
        output_dimensions: usize,
        pooling: String,
        normalization: String,
        query_prefix: String,
        document_prefix: String,
        max_sequence_length: usize,
        max_batch_size: usize,
    }

    #[derive(Deserialize)]
    struct ReleaseArtifact {
        path: String,
        size_bytes: u64,
        sha256: String,
    }

    #[cfg(windows)]
    #[derive(Deserialize)]
    struct RuntimeManifest {
        dll: RuntimeDll,
    }

    #[cfg(windows)]
    #[derive(Deserialize)]
    struct RuntimeDll {
        file_name: String,
        size_bytes: u64,
        sha256: String,
    }

    #[test]
    fn contract_is_locked_to_release_identity() {
        let contract = contract();
        assert_eq!(contract.output_dimensions, 512);
        assert_eq!(contract.revision, MODEL_REVISION);
        assert_eq!(contract.query_prefix, "search_query: ");
        assert_eq!(contract.document_prefix, "search_document: ");
    }

    #[test]
    fn release_manifest_matches_runtime_constants() {
        let manifest: ReleaseManifest = serde_json::from_str(include_str!(
            "../../../reference/embedding/nomic-embed-text-v1.5-a15734e.json"
        ))
        .unwrap();
        assert_eq!(manifest.model_id, MODEL_ID);
        assert_eq!(manifest.revision, MODEL_REVISION);
        assert_eq!(manifest.contract.native_dimensions, NATIVE_DIMENSIONS);
        assert_eq!(manifest.contract.output_dimensions, OUTPUT_DIMENSIONS);
        assert_eq!(manifest.contract.pooling, contract().pooling);
        assert_eq!(manifest.contract.normalization, contract().normalization);
        assert_eq!(
            manifest.contract.query_prefix,
            EmbeddingTask::Query.prefix()
        );
        assert_eq!(
            manifest.contract.document_prefix,
            EmbeddingTask::Document.prefix()
        );
        assert_eq!(manifest.contract.max_sequence_length, MAX_SEQUENCE_LENGTH);
        assert_eq!(manifest.contract.max_batch_size, MAX_BATCH_SIZE);
        let model = manifest
            .artifacts
            .iter()
            .find(|file| file.path == MODEL_FILE)
            .unwrap();
        assert_eq!(model.size_bytes, MODEL_SIZE_BYTES);
        assert_eq!(model.sha256, MODEL_SHA256);
        let tokenizer = manifest
            .artifacts
            .iter()
            .find(|file| file.path == TOKENIZER_FILE)
            .unwrap();
        assert_eq!(tokenizer.size_bytes, TOKENIZER_SIZE_BYTES);
        assert_eq!(tokenizer.sha256, TOKENIZER_SHA256);
    }

    #[cfg(windows)]
    #[test]
    fn onnx_runtime_manifest_matches_companion_constants() {
        let manifest: RuntimeManifest = serde_json::from_str(include_str!(
            "../../../reference/runtime/onnxruntime-1.20.0-windows-x64.json"
        ))
        .unwrap();
        assert_eq!(manifest.dll.file_name, ONNX_RUNTIME_FILE);
        assert_eq!(manifest.dll.size_bytes, ONNX_RUNTIME_SIZE_BYTES);
        assert_eq!(manifest.dll.sha256, ONNX_RUNTIME_SHA256);
    }

    #[test]
    fn pooling_ignores_padding_and_returns_unit_vectors() {
        let batch = 1;
        let sequence = 3;
        let mut hidden = vec![0.0; batch * sequence * NATIVE_DIMENSIONS];
        for dimension in 0..NATIVE_DIMENSIONS {
            hidden[dimension] = dimension as f32 / 100.0;
            hidden[NATIVE_DIMENSIONS + dimension] = 1.0 + dimension as f32 / 100.0;
            hidden[2 * NATIVE_DIMENSIONS + dimension] = 1000.0;
        }
        let mask = Array2::from_shape_vec((1, 3), vec![1, 1, 0]).unwrap();
        let vectors = pool_project_normalize(&hidden, batch, sequence, &mask).unwrap();
        assert_eq!(vectors[0].len(), OUTPUT_DIMENSIONS);
        let norm = vectors[0]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }
}
