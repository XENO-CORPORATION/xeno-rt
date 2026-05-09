//! `xrt-vision` — Image-domain task inference for the xeno-rt runtime.
//!
//! Wraps [`xeno_lib::ai_staging`] inference modules behind a stable async API
//! consumed by `xrt-server`. Each task module loads its ONNX model once,
//! caches the [`Session`] under a `parking_lot::Mutex`, and exposes a single
//! `run(...)` entry point that takes raw image bytes and returns a processed
//! result.
//!
//! ## Surface
//! - [`background_removal::remove_background`] — BiRefNet matte + alpha cutout.
//! - (Planned) `upscale`, `denoise`, `face_restore`, `colorize`, `inpaint`.
//!
//! ## Concurrency
//! ONNX `Session::run` is `&mut self`, so we serialise per-task with a mutex.
//! Multiple tasks can run concurrently because each holds its own session.
//! Heavy CPU/GPU work is wrapped in `tokio::task::spawn_blocking` by the
//! HTTP layer (`xrt-server`) so we never block the async executor.
//!
//! ## Model location
//! Defaults to `~/.xeno/models/{task}/...`. Overrideable per-call so the
//! orchestrator (or operator) can point at any local file. The server crate
//! is responsible for ensuring the model exists before invoking inference;
//! this crate returns a clear error if the file is missing.

pub mod background_removal;

#[derive(Debug, thiserror::Error)]
pub enum VisionError {
    /// Model file is missing on disk and we couldn't auto-download.
    #[error("model file not found at {path}: {message}")]
    ModelMissing { path: String, message: String },

    /// ONNX Runtime / inference itself failed.
    #[error("inference failed: {0}")]
    Inference(String),

    /// Input image bytes could not be decoded.
    #[error("invalid input image: {0}")]
    InvalidImage(String),

    /// Output image could not be encoded back to bytes.
    #[error("failed to encode output: {0}")]
    EncodeFailed(String),
}
