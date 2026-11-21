//! Temporal frame chunking and sliding window management for video restoration.

use image::DynamicImage;

/// A chunk of video frames with overlapping boundary frames for temporal consistency.
#[derive(Debug, Clone)]
pub struct TemporalChunk {
    /// Zero-based global start frame index in the sequence.
    pub start_index: usize,
    /// Number of core frames in this chunk (excluding left/right overlap).
    pub core_count: usize,
    /// Number of left overlap frames included before start_index.
    pub left_overlap: usize,
    /// Number of right overlap frames included after start_index + core_count.
    pub right_overlap: usize,
}

impl TemporalChunk {
    /// Calculate the sliding window chunks for a sequence of `total_frames`.
    ///
    /// - `chunk_size`: number of frames processed in one forward pass (e.g. 16 or 24).
    /// - `overlap`: number of padding frames on each side (e.g. 4).
    pub fn plan_chunks(total_frames: usize, chunk_size: usize, overlap: usize) -> Vec<Self> {
        if total_frames == 0 {
            return Vec::new();
        }

        let chunk_size = chunk_size.max(1);
        let mut chunks = Vec::new();
        let mut start = 0;

        while start < total_frames {
            let core = chunk_size.min(total_frames - start);
            let left = overlap.min(start);
            let right = overlap.min(total_frames - (start + core));

            chunks.push(Self {
                start_index: start,
                core_count: core,
                left_overlap: left,
                right_overlap: right,
            });

            start += core;
        }

        chunks
    }
}

/// Convert a batch of `DynamicImage` frames into a 5D NCDHW tensor `[1, 3, T, H, W]`
/// with normalized f32 values in `[0, 1]`.
pub fn frames_to_5d_tensor(
    frames: &[DynamicImage],
    target_h: usize,
    target_w: usize,
) -> ndarray::Array5<f32> {
    let t = frames.len();
    let mut tensor = ndarray::Array5::<f32>::zeros((1, 3, t, target_h, target_w));

    for (frame_idx, frame) in frames.iter().enumerate() {
        let rgb = frame.to_rgb8();
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        let sample_w = w.min(target_w);
        let sample_h = h.min(target_h);

        for y in 0..sample_h {
            for x in 0..sample_w {
                let px = rgb.get_pixel(x as u32, y as u32);
                tensor[[0, 0, frame_idx, y, x]] = px[0] as f32 / 255.0;
                tensor[[0, 1, frame_idx, y, x]] = px[1] as f32 / 255.0;
                tensor[[0, 2, frame_idx, y, x]] = px[2] as f32 / 255.0;
            }
        }
    }

    tensor
}
