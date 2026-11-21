//! `xrt-video` — Video domain task and temporal inference for XENO RT.
//!
//! Provides video restoration, super-resolution (e.g. SeedVR2, BasicVSR++),
//! temporal sliding-window chunking, and frame-by-frame upscale pipelines.

pub mod temporal;
pub mod upscale;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum VideoError {
    #[error("model file not found at {path}: {message}")]
    ModelMissing { path: String, message: String },

    #[error("inference failed: {0}")]
    Inference(String),

    #[error("invalid input frame at index {index}: {message}")]
    InvalidFrame { index: usize, message: String },

    #[error("temporal processing failed: {0}")]
    Temporal(String),

    #[error("failed to encode output video/frame: {0}")]
    EncodeFailed(String),
}
