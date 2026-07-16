#[path = "onnx/download.rs"]
pub mod download;
#[path = "onnx/registry.rs"]
pub mod registry;
#[path = "onnx/runtime.rs"]
pub mod runtime;

use std::path::PathBuf;

use thiserror::Error;

pub use self::download::{DownloadProgress, DownloadedArtifact, OnnxDownloader};
pub use self::registry::{
    ExecutionProviderKind, ModelArtifact, ModelDescriptor, ModelRegistry,
};
pub use self::runtime::{
    ExecutionDevice, GpuPreference, OnnxOptimizationLevel, OnnxRuntime,
    OnnxRuntimeOptions,
};

pub type Result<T> = std::result::Result<T, OnnxError>;

#[derive(Debug, Error)]
pub enum OnnxError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("onnx runtime error: {0}")]
    Ort(#[from] ort::Error),
    #[error("ONNX model file does not exist: {}", path.display())]
    ModelPathMissing { path: PathBuf },
    #[error("duplicate ONNX model id in registry: {0}")]
    DuplicateModel(String),
    #[error("ONNX model not found in registry: {0}")]
    ModelNotFound(String),
    #[error("invalid ONNX metadata: {0}")]
    InvalidMetadata(String),
    #[error("download failed: {0}")]
    Download(String),
    #[error(
        "size mismatch for {}: expected {expected} bytes, got {actual}",
        path.display()
    )]
    SizeMismatch {
        path: PathBuf,
        expected: u64,
        actual: u64,
    },
    #[error(
        "sha256 mismatch for {}: expected {expected}, got {actual}",
        path.display()
    )]
    ChecksumMismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },
}
