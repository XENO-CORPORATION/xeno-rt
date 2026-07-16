//! Background-removal task using BiRefNet.
//!
//! Inference pipeline:
//!   1. Decode incoming bytes to `image::DynamicImage`.
//!   2. `image_to_tensor`: resize to model input (1024×1024), normalize,
//!      pack into `Array4<f32>` of shape `[1, 3, H, W]`.
//!   3. `ModelSession::run`: BiRefNet → `Array2<f32>` mask in [0, 1].
//!   4. `apply_mask`: resize mask to original dimensions, refine edges via
//!      guided filter against the original RGB, alpha-multiply onto the
//!      original RGBA → `DynamicImage`.
//!   5. Encode result as PNG bytes (lossless, alpha-preserving).
//!
//! The session is cached behind a `Mutex` so successive calls reuse the
//! loaded weights. Cold start loads the ~928 MB ONNX model in 2-5s; warm
//! calls hit the GPU/CPU directly (1-3s on RTX-class GPU, 5-15s on CPU).

use std::path::PathBuf;
use std::sync::Arc;

use image::ImageEncoder;
use once_cell::sync::OnceCell;
use parking_lot::Mutex;

use crate::VisionError;

mod model;
mod postprocess;
mod preprocess;

use model::{load_model, BackgroundRemovalConfig, ModelSession};

// ─── Session cache ──────────────────────────────────────────────────────────
//
// One BiRefNet session, shared across all incoming requests. Wrapped in a
// `Mutex` because `Session::run` requires `&mut self`. If we ever want
// concurrent inference (multi-batch), spawn additional sessions on demand —
// not before, because each holds ~1 GB of GPU memory and the marginal
// throughput gain is small for a single-user creative tool.

static SESSION: OnceCell<Arc<Mutex<ModelSession>>> = OnceCell::new();

fn get_or_init_session(
    config: &BackgroundRemovalConfig,
) -> Result<Arc<Mutex<ModelSession>>, VisionError> {
    if let Some(existing) = SESSION.get() {
        return Ok(Arc::clone(existing));
    }
    if !config.model_path.exists() {
        return Err(VisionError::ModelMissing {
            path: config.model_path.display().to_string(),
            message: "BiRefNet ONNX file not present — download via the orchestrator before invoking inference".to_string(),
        });
    }
    let session = load_model(config)?;
    let arc: Arc<Mutex<ModelSession>> = Arc::new(Mutex::new(session));
    // First-write-wins. If two callers race, both `load_model` calls
    // succeed; the loser's session is dropped immediately.
    let _ = SESSION.set(Arc::clone(&arc));
    Ok(SESSION.get().map(Arc::clone).unwrap_or(arc))
}

// ─── Public API ─────────────────────────────────────────────────────────────

/// Configuration for a background-removal call. All fields default to safe
/// values; callers typically only override `model_path` if they want a
/// non-standard location.
#[derive(Debug, Clone)]
pub struct RemoveBackgroundOptions {
    /// Absolute path to the BiRefNet ONNX file.
    pub model_path: PathBuf,
    /// Try CUDA before falling back to CPU.
    pub use_gpu: bool,
    /// Confidence threshold for foreground pixels (0.0 ‒ 1.0).
    pub confidence_threshold: f32,
}

impl Default for RemoveBackgroundOptions {
    fn default() -> Self {
        // Default model location follows the locked Hub <-> apps convention
        // (`~/.xeno/models/birefnet-general/model.onnx`). Both Hub and Pixel
        // expect the same path so the file can be shared.
        Self {
            model_path: default_model_path(),
            use_gpu: true,
            confidence_threshold: 0.1,
        }
    }
}

fn default_model_path() -> PathBuf {
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join(".xeno")
        .join("models")
        .join("birefnet-general")
        .join("model.onnx")
}

/// Run BiRefNet background removal end-to-end.
///
/// Input: image bytes (PNG / JPEG / WebP — anything `image` decodes).
/// Output: PNG bytes with the background pixels alpha-cut.
///
/// Blocking — wrap in `tokio::task::spawn_blocking` from async contexts.
/// Inference takes ~1-3s on a recent NVIDIA GPU, ~5-15s on CPU at 1080p.
///
/// On the first call, loads the ONNX model into memory (cached for
/// subsequent calls). On a missing model file, returns
/// [`VisionError::ModelMissing`] immediately without attempting load.
pub fn remove_background(
    input_bytes: &[u8],
    opts: &RemoveBackgroundOptions,
) -> Result<Vec<u8>, VisionError> {
    // 1. Decode bytes to a workable image.
    let original = image::load_from_memory(input_bytes)
        .map_err(|e| VisionError::InvalidImage(e.to_string()))?;
    let (orig_w, orig_h) = (original.width(), original.height());

    // 2. Lazy-load the BiRefNet session (cached after first call).
    let lib_config = BackgroundRemovalConfig {
        model_path: opts.model_path.clone(),
        use_gpu: opts.use_gpu,
        gpu_device_id: 0,
        confidence_threshold: opts.confidence_threshold,
    };
    let session_arc = get_or_init_session(&lib_config)?;

    // 3. Run preprocess, inference, and postprocess. The mutex is held for
    //    inference because `Session::run` requires mutable access.
    let output_image = {
        let input = preprocess::image_to_tensor(&original, (1024, 1024));
        let mut session = session_arc.lock();
        let mask = session.run(&input)?;
        postprocess::apply_mask(
            &original,
            &mask,
            orig_w,
            orig_h,
            session.config().confidence_threshold,
        )
    };

    // 4. Encode PNG (lossless, alpha-preserving).
    let mut out_bytes = Vec::with_capacity(input_bytes.len());
    let rgba = output_image.to_rgba8();
    image::codecs::png::PngEncoder::new(&mut out_bytes)
        .write_image(
            rgba.as_raw(),
            orig_w,
            orig_h,
            image::ExtendedColorType::Rgba8,
        )
        .map_err(|e| VisionError::EncodeFailed(e.to_string()))?;

    Ok(out_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test — verifies the public API surface compiles and a missing
    /// model produces a clean ModelMissing error rather than a panic.
    #[test]
    fn missing_model_returns_clear_error() {
        let opts = RemoveBackgroundOptions {
            model_path: PathBuf::from("/definitely/does/not/exist.onnx"),
            ..Default::default()
        };
        // Tiny synthetic PNG so the decoder doesn't fail first.
        let mut png = Vec::new();
        let pixels = vec![255u8; 4 * 4 * 4];
        image::codecs::png::PngEncoder::new(&mut png)
            .write_image(&pixels, 4, 4, image::ExtendedColorType::Rgba8)
            .unwrap();
        let result = remove_background(&png, &opts);
        match result {
            Err(VisionError::ModelMissing { .. }) => {}
            other => panic!("expected ModelMissing, got {other:?}"),
        }
    }
}
