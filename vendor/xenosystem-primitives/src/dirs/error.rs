use std::fmt;

/// Stable codes shared with the TypeScript implementation (`DirsErrorCode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    HomeUnresolvable,
    UnregisteredArea,
    KindNotUsed,
    LocksOnly,
    InvalidSegment,
    LayoutNewer,
    LayoutInvalid,
    LockHeld,
    LockCorrupt,
    LockTimeout,
    MigrationRefused,
    InvalidArgument,
    Io,
}

impl ErrorCode {
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorCode::HomeUnresolvable => "home-unresolvable",
            ErrorCode::UnregisteredArea => "unregistered-area",
            ErrorCode::KindNotUsed => "kind-not-used",
            ErrorCode::LocksOnly => "locks-only",
            ErrorCode::InvalidSegment => "invalid-segment",
            ErrorCode::LayoutNewer => "layout-newer",
            ErrorCode::LayoutInvalid => "layout-invalid",
            ErrorCode::LockHeld => "lock-held",
            ErrorCode::LockCorrupt => "lock-corrupt",
            ErrorCode::LockTimeout => "lock-timeout",
            ErrorCode::MigrationRefused => "migration-refused",
            ErrorCode::InvalidArgument => "invalid-argument",
            ErrorCode::Io => "io",
        }
    }
}

#[derive(Debug, Clone)]
pub struct DirsError {
    pub code: ErrorCode,
    pub message: String,
}

impl DirsError {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        DirsError { code, message: message.into() }
    }
}

impl fmt::Display for DirsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code.as_str(), self.message)
    }
}

impl std::error::Error for DirsError {}

impl From<std::io::Error> for DirsError {
    fn from(e: std::io::Error) -> Self {
        DirsError::new(ErrorCode::Io, e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, DirsError>;
