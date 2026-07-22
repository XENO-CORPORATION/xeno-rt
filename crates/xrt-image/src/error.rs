use std::io;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageErrorKind {
    InvalidRequest,
    UnsupportedCapability,
    UnsupportedQuantization,
    UnsupportedTensor,
    UnsupportedShape,
    UnsupportedBackend,
    MissingComponent,
    CorruptComponent,
    Checksum,
    Manifest,
    Admission,
    InsufficientMemory,
    Cancelled,
    Codec,
    InputLimit,
    Execution,
    Numerical,
    Internal,
}

#[derive(Debug, thiserror::Error)]
pub enum ImageError {
    #[error("invalid image request: {0}")]
    InvalidRequest(String),
    #[error("unsupported image capability: {0}")]
    UnsupportedCapability(String),
    #[error("unsupported image quantization: {0}")]
    UnsupportedQuantization(String),
    #[error("unsupported image tensor: {0}")]
    UnsupportedTensor(String),
    #[error("unsupported image shape: {0}")]
    UnsupportedShape(String),
    #[error("unsupported image backend: {0}")]
    UnsupportedBackend(String),
    #[error("missing image component: {0}")]
    MissingComponent(String),
    #[error("corrupt image component: {0}")]
    CorruptComponent(String),
    #[error("image component checksum failed: {0}")]
    Checksum(String),
    #[error("invalid image bundle manifest: {0}")]
    Manifest(String),
    #[error("image request admission failed: {0}")]
    Admission(String),
    #[error("insufficient memory for image request: {0}")]
    InsufficientMemory(String),
    #[error("image request cancelled")]
    Cancelled,
    #[error("image codec failed: {0}")]
    Codec(String),
    #[error("image input limit exceeded: {0}")]
    InputLimit(String),
    #[error("image execution failed: {0}")]
    Execution(String),
    #[error("non-finite image value in {component} at step {step}")]
    Numerical {
        component: &'static str,
        step: usize,
    },
    #[error("image runtime invariant failed: {0}")]
    Internal(String),
}

impl ImageError {
    pub const fn kind(&self) -> ImageErrorKind {
        match self {
            Self::InvalidRequest(_) => ImageErrorKind::InvalidRequest,
            Self::UnsupportedCapability(_) => ImageErrorKind::UnsupportedCapability,
            Self::UnsupportedQuantization(_) => ImageErrorKind::UnsupportedQuantization,
            Self::UnsupportedTensor(_) => ImageErrorKind::UnsupportedTensor,
            Self::UnsupportedShape(_) => ImageErrorKind::UnsupportedShape,
            Self::UnsupportedBackend(_) => ImageErrorKind::UnsupportedBackend,
            Self::MissingComponent(_) => ImageErrorKind::MissingComponent,
            Self::CorruptComponent(_) => ImageErrorKind::CorruptComponent,
            Self::Checksum(_) => ImageErrorKind::Checksum,
            Self::Manifest(_) => ImageErrorKind::Manifest,
            Self::Admission(_) => ImageErrorKind::Admission,
            Self::InsufficientMemory(_) => ImageErrorKind::InsufficientMemory,
            Self::Cancelled => ImageErrorKind::Cancelled,
            Self::Codec(_) => ImageErrorKind::Codec,
            Self::InputLimit(_) => ImageErrorKind::InputLimit,
            Self::Execution(_) => ImageErrorKind::Execution,
            Self::Numerical { .. } => ImageErrorKind::Numerical,
            Self::Internal(_) => ImageErrorKind::Internal,
        }
    }
}

impl From<io::Error> for ImageError {
    fn from(error: io::Error) -> Self {
        Self::CorruptComponent(error.to_string())
    }
}
