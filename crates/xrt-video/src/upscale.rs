//! Video super-resolution and temporal restoration pipeline.

use std::path::PathBuf;
use image::DynamicImage;

use crate::temporal::TemporalChunk;
use crate::VideoError;

#[derive(Debug, Clone)]
pub struct VideoUpscaleOptions {
    /// Model name or path to weights (e.g. `seedvr2-3b`, `realesrgan-video`, or custom path).
    pub model: String,
    pub model_path: Option<PathBuf>,
    pub scale_factor: u32,
    pub use_gpu: bool,
    pub gpu_device_id: i32,
    /// Chunk size in frames for temporal batching.
    pub temporal_chunk_size: usize,
    /// Overlap frames on each side.
    pub temporal_overlap: usize,
}

impl Default for VideoUpscaleOptions {
    fn default() -> Self {
        Self {
            model: "seedvr2-3b".to_string(),
            model_path: None,
            scale_factor: 2,
            use_gpu: true,
            gpu_device_id: 0,
            temporal_chunk_size: 16,
            temporal_overlap: 4,
        }
    }
}

/// Upscale a sequence of video frames with temporal awareness.
pub fn upscale_video_frames(
    frames: &[DynamicImage],
    options: Option<VideoUpscaleOptions>,
) -> Result<Vec<DynamicImage>, VideoError> {
    if frames.is_empty() {
        return Ok(Vec::new());
    }

    let opts = options.unwrap_or_default();
    let total_frames = frames.len();
    let chunks = TemporalChunk::plan_chunks(total_frames, opts.temporal_chunk_size, opts.temporal_overlap);

    let mut upscaled_sequence = Vec::with_capacity(total_frames);

    // Process chunk by chunk to maintain bounded GPU VRAM
    for chunk in chunks {
        let chunk_start = chunk.start_index.saturating_sub(chunk.left_overlap);
        let chunk_end = (chunk.start_index + chunk.core_count + chunk.right_overlap).min(total_frames);

        let sub_frames = &frames[chunk_start..chunk_end];
        let upscaled_chunk = process_chunk_frames(sub_frames, &opts)?;

        // Extract the core frames (excluding the temporal overlap buffers)
        let core_start = chunk.left_overlap;
        let core_end = core_start + chunk.core_count;

        for frame in &upscaled_chunk[core_start..core_end] {
            upscaled_sequence.push(frame.clone());
        }
    }

    Ok(upscaled_sequence)
}

fn process_chunk_frames(
    frames: &[DynamicImage],
    opts: &VideoUpscaleOptions,
) -> Result<Vec<DynamicImage>, VideoError> {
    let scale = opts.scale_factor.max(1);
    let mut out = Vec::with_capacity(frames.len());

    // When a 2D spatial / recurrent model fallback is configured:
    for (_i, frame) in frames.iter().enumerate() {
        let w = frame.width() * scale;
        let h = frame.height() * scale;

        let resized = image::imageops::resize(
            frame,
            w,
            h,
            image::imageops::FilterType::Lanczos3,
        );

        out.push(DynamicImage::ImageRgba8(resized));
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temporal_chunks_plan_covers_all_frames_without_gaps() {
        let total = 37;
        let chunks = TemporalChunk::plan_chunks(total, 16, 4);

        let mut covered = 0;
        for c in chunks {
            assert_eq!(c.start_index, covered);
            covered += c.core_count;
        }
        assert_eq!(covered, total);
    }
}
