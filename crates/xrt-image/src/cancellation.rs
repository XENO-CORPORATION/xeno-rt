use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use serde::{Deserialize, Serialize};

use crate::ImageError;

#[derive(Clone, Debug, Default)]
pub struct ImageCancellation {
    cancelled: Arc<AtomicBool>,
}

impl ImageCancellation {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    pub fn check(&self) -> Result<(), ImageError> {
        if self.is_cancelled() {
            Err(ImageError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageProgressPhase {
    Admitted,
    PromptEncoding,
    SourceEncoding,
    Denoising,
    VaeDecode,
    Encoding,
    Complete,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageProgressEvent {
    pub output_index: usize,
    pub phase: ImageProgressPhase,
    pub step: Option<usize>,
    pub total_steps: Option<usize>,
}

/// An encoded, deterministic preview emitted at an admitted denoising
/// checkpoint. Preview bytes are kept behind an `Arc` so subscribers can
/// forward them without another full image copy.
#[derive(Debug, Clone)]
pub struct ImagePreviewEvent {
    pub output_index: usize,
    /// Number of denoising steps completed when this preview was decoded.
    pub step: usize,
    pub total_steps: usize,
    pub bytes: Arc<[u8]>,
    pub mime_type: String,
    pub width: u32,
    pub height: u32,
}

pub trait ImageProgressSink: Send + Sync {
    fn on_progress(&self, event: &ImageProgressEvent);

    /// Preview decoding stays opt-in because it adds material VAE and codec
    /// work. Progress-only subscribers retain their previous cost profile.
    fn wants_previews(&self) -> bool {
        false
    }

    fn on_preview(&self, _event: &ImagePreviewEvent) {}
}

impl<F> ImageProgressSink for F
where
    F: Fn(&ImageProgressEvent) + Send + Sync,
{
    fn on_progress(&self, event: &ImageProgressEvent) {
        self(event);
    }
}
