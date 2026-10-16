use crate::dirs::DirsError;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecretsErrorCode {
    UnprotectedStore,
    UnsupportedPlatform,
    KeyUnavailable,
    KeyMismatch,
    DecryptFailed,
    Malformed,
    InvalidArgument,
    /// An error from the `dirs` family (registry, layout, lock, I/O); the inner code is in the message prefix.
    Dirs,
}

impl SecretsErrorCode {
    pub fn as_str(self) -> &'static str {
        match self {
            SecretsErrorCode::UnprotectedStore => "unprotected-store",
            SecretsErrorCode::UnsupportedPlatform => "unsupported-platform",
            SecretsErrorCode::KeyUnavailable => "key-unavailable",
            SecretsErrorCode::KeyMismatch => "key-mismatch",
            SecretsErrorCode::DecryptFailed => "decrypt-failed",
            SecretsErrorCode::Malformed => "malformed",
            SecretsErrorCode::InvalidArgument => "invalid-argument",
            SecretsErrorCode::Dirs => "dirs",
        }
    }
}

/// Messages never contain a secret value or key; they may name the area and the secret's name (names are not secret).
#[derive(Debug, Clone)]
pub struct SecretsError {
    pub code: SecretsErrorCode,
    pub message: String,
    /// Set when `code` is `Dirs`: the underlying dirs code, e.g. `kind-not-used`, `layout-newer`.
    pub dirs_code: Option<&'static str>,
}

impl SecretsError {
    pub fn new(code: SecretsErrorCode, message: impl Into<String>) -> Self {
        SecretsError { code, message: message.into(), dirs_code: None }
    }
    /// The stable string callers compare against: the dirs code for a dirs error, otherwise this family's code.
    pub fn code_str(&self) -> &'static str {
        self.dirs_code.unwrap_or(self.code.as_str())
    }
}

impl fmt::Display for SecretsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code_str(), self.message)
    }
}

impl std::error::Error for SecretsError {}

impl From<DirsError> for SecretsError {
    fn from(e: DirsError) -> Self {
        SecretsError { code: SecretsErrorCode::Dirs, message: e.message, dirs_code: Some(e.code.as_str()) }
    }
}

impl From<std::io::Error> for SecretsError {
    fn from(e: std::io::Error) -> Self {
        SecretsError { code: SecretsErrorCode::Dirs, message: e.to_string(), dirs_code: Some("io") }
    }
}

pub type Result<T> = std::result::Result<T, SecretsError>;
