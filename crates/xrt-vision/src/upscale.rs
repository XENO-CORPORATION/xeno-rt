//! Image super-resolution and upscaling task.
//!
//! Inference pipeline:
//!   1. Decode incoming bytes to `image::DynamicImage`.
//!   2. If the image is large, split into padded tiles to preserve memory constraints.
//!   3. Run super-resolution model on tiles or full image.
//!   4. Stitch back seamlessly, removing overlapping tile padding.
//!   5. If alpha channel was present, upscale alpha channel cleanly.
//!   6. Encode result as PNG bytes.

mod model;
mod tiling;

use std::{
    io::Cursor,
    path::PathBuf,
};

use image::{DynamicImage, ImageBuffer, ImageFormat, Rgb};
use parking_lot::Mutex;

use self::model::{create_session, ModelSession, UpscaleConfig};
use crate::VisionError;

/// Cached upscale session.
static CACHED_SESSION: Mutex<Option<ModelSession>> = Mutex::new(None);

/// High-level upscale options passed from server / caller.
#[derive(Debug, Clone)]
pub struct UpscaleOptions {
    /// Path to the ONNX super-resolution model.
    pub model_path: Option<PathBuf>,
    /// Scale factor (e.g. 2, 4).
    pub scale_factor: u32,
    /// Use GPU if available.
    pub use_gpu: bool,
    /// GPU device ID (default 0).
    pub gpu_device_id: i32,
    /// Tile size for tiled inference (0 means auto or whole image if smaller than 512).
    pub tile_size: u32,
    /// Overlap padding between tiles in pixels.
    pub tile_pad: u32,
}

impl Default for UpscaleOptions {
    fn default() -> Self {
        Self {
            model_path: None,
            scale_factor: 4,
            use_gpu: true,
            gpu_device_id: 0,
            tile_size: 512,
            tile_pad: 16,
        }
    }
}

/// Default model path resolution: `~/.xeno/models/upscale/default.onnx`
pub fn default_upscale_model_path() -> PathBuf {
    if let Ok(override_path) = std::env::var("XRT_UPSCALE_MODEL_PATH") {
        let p = PathBuf::from(override_path);
        if !p.as_os_str().is_empty() {
            return p;
        }
    }

    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));

    home.join(".xeno")
        .join("models")
        .join("upscale")
        .join("real_esrgan_x4.onnx")
}

/// Run super-resolution on raw image bytes.
pub fn upscale_image(
    input_bytes: &[u8],
    options: Option<UpscaleOptions>,
) -> Result<Vec<u8>, VisionError> {
    let opts = options.unwrap_or_default();
    let model_path = opts.model_path.unwrap_or_else(default_upscale_model_path);

    let config = UpscaleConfig {
        model_path,
        use_gpu: opts.use_gpu,
        gpu_device_id: opts.gpu_device_id,
        scale_factor: if opts.scale_factor == 0 { 4 } else { opts.scale_factor },
        tile_size: if opts.tile_size == 0 { 512 } else { opts.tile_size },
        tile_pad: if opts.tile_pad == 0 { 16 } else { opts.tile_pad },
    };

    let decoded = image::load_from_memory(input_bytes)
        .map_err(|e| VisionError::InvalidImage(format!("failed to decode input image: {e}")))?;

    let upscaled = run_upscale_pipeline(&decoded, &config)?;

    let mut out_bytes = Vec::new();
    let mut cursor = Cursor::new(&mut out_bytes);
    upscaled
        .write_to(&mut cursor, ImageFormat::Png)
        .map_err(|e| VisionError::EncodeFailed(format!("failed to encode PNG: {e}")))?;

    Ok(out_bytes)
}

fn run_upscale_pipeline(
    img: &DynamicImage,
    config: &UpscaleConfig,
) -> Result<DynamicImage, VisionError> {
    let width = img.width();
    let height = img.height();
    let scale = config.scale_factor;
    let target_width = width * scale;
    let target_height = height * scale;

    let mut session_guard = CACHED_SESSION.lock();
    let session = match session_guard.as_mut() {
        Some(s) if s.config().model_path == config.model_path
            && s.config().use_gpu == config.use_gpu
            && s.config().gpu_device_id == config.gpu_device_id =>
        {
            s
        }
        _ => {
            let fresh = create_session(config)?;
            *session_guard = Some(fresh);
            session_guard.as_mut().unwrap()
        }
    };

    // If image fits comfortably in a single tile, run full image directly
    let tile_size = config.tile_size;
    let tile_pad = config.tile_pad;

    let output_rgb: ImageBuffer<Rgb<u8>, Vec<u8>> = if width <= tile_size && height <= tile_size {
        let tensor = tiling::image_to_tensor(img, 0, 0, width, height)?;
        let out_tensor = session.run(tensor)?;
        tiling::tensor_to_rgb_image(&out_tensor)?
    } else {
        // Tiled inference for large images
        let mut final_canvas = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(target_width, target_height);

        let mut y = 0;
        while y < height {
            let tile_h = (tile_size).min(height - y);
            let mut x = 0;
            while x < width {
                let tile_w = (tile_size).min(width - x);

                // Add padding for boundary smoothing
                let pad_x0 = x.saturating_sub(tile_pad);
                let pad_y0 = y.saturating_sub(tile_pad);
                let pad_x1 = (x + tile_w + tile_pad).min(width);
                let pad_y1 = (y + tile_h + tile_pad).min(height);

                let padded_w = pad_x1 - pad_x0;
                let padded_h = pad_y1 - pad_y0;

                let tensor = tiling::image_to_tensor(img, pad_x0, pad_y0, padded_w, padded_h)?;
                let out_tensor = session.run(tensor)?;
                let tile_out = tiling::tensor_to_rgb_image(&out_tensor)?;

                // Crop out the padding from the upscaled tile
                let crop_x = (x - pad_x0) * scale;
                let crop_y = (y - pad_y0) * scale;
                let crop_w = tile_w * scale;
                let crop_h = tile_h * scale;

                for dy in 0..crop_h {
                    for dx in 0..crop_w {
                        let px = tile_out.get_pixel(crop_x + dx, crop_y + dy);
                        let out_x = x * scale + dx;
                        let out_y = y * scale + dy;
                        if out_x < target_width && out_y < target_height {
                            final_canvas.put_pixel(out_x, out_y, *px);
                        }
                    }
                }

                x += tile_w;
            }
            y += tile_h;
        }

        final_canvas
    };

    let final_image = tiling::merge_alpha_if_present(img, output_rgb, target_width, target_height);
    Ok(final_image)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_model_returns_clear_error() {
        let fake_path = PathBuf::from("nonexistent_test_model.onnx");
        let opts = UpscaleOptions {
            model_path: Some(fake_path.clone()),
            ..Default::default()
        };
        // 1x1 red PNG
        let mut img = image::RgbImage::new(1, 1);
        img.put_pixel(0, 0, image::Rgb([255, 0, 0]));
        let mut bytes = Vec::new();
        DynamicImage::ImageRgb8(img)
            .write_to(&mut Cursor::new(&mut bytes), ImageFormat::Png)
            .unwrap();

        let res = upscale_image(&bytes, Some(opts));
        match res {
            Err(VisionError::ModelMissing { path, .. }) => {
                assert_eq!(path, fake_path.display().to_string());
            }
            other => panic!("expected ModelMissing error, got {:?}", other),
        }
    }
}
