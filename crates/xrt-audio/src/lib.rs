//! xeno-rt's audio-task crate: the signal frontend and ONNX inference for
//! audio-domain tasks, re-exposed by `xrt-server` as `/v1/audio/...`.
//!
//! # Why this crate is here and not in xeno-lib
//!
//! The ecosystem boundary is locked (root `CLAUDE.md`, interface table):
//! `xeno-lib` is media processing with **no AI**, and **all** model inference
//! lives in xeno-rt. xeno-lib's own README states it: *"AI inference lives in
//! xeno-rt - not here."*
//!
//! xeno-lib did once carry `transcribe/` and `audio_separate/`, now retired
//! into `src/ai_deprecated/`. They were not a head start. Its `audio_to_mel`
//! contained no FFT - it summed `s*s` per frame and multiplied that scalar by
//! a fixed sine curve - and its `decode_tokens` had no vocabulary, emitting
//! `"[50364]"` token ids as the transcript. Both were covered by tests that
//! asserted tensor *shape*, so they passed. That is the ecosystem's recurring
//! "fabricated success" defect: a result that reads as an answer and is not
//! one. Porting it would have moved the lie into the correct layer.
//!
//! # Status
//!
//! The frontend ([`stft`], [`mel`]) is implemented and tested against the
//! properties Whisper's weights depend on, and [`whisper`] runs the real
//! encoder/decoder over ONNX Runtime with long-form windowing. Both are proven
//! end to end against the published weights by `tests/whisper_e2e.rs`.
//!
//! Not built: Demucs separation, Whisper timestamp tokens, language detection,
//! and any `/v1/audio/*` route - so nothing outside this crate can reach any of
//! it yet. That last part is deliberate, per this repo's rule against
//! advertising unadmitted support.
//!
//! Nothing here fabricates a result when a model is missing; see
//! [`AudioError::ModelUnavailable`] and [`AudioError::ModelFileMissing`].

pub mod mel;
pub mod stft;
pub mod whisper;

use std::path::PathBuf;

/// Errors from any audio task.
#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    /// The model file is not present locally and could not be fetched.
    ///
    /// Kept distinct from every other failure on purpose: it is the only one
    /// a user can act on, and the action ("the model has not been published")
    /// is different from a genuine inference fault. Collapsing the two is how
    /// a distribution gap comes to look like a broken feature.
    #[error("model {name} is unavailable: {reason}")]
    ModelUnavailable { name: String, reason: String },

    #[error("model file not found at {0}")]
    ModelFileMissing(PathBuf),

    #[error("audio decode failed: {0}")]
    Decode(String),

    #[error("inference failed: {0}")]
    Inference(String),

    #[error("unsupported request: {0}")]
    Unsupported(String),
}

/// Convenience alias used across the crate.
pub type Result<T> = std::result::Result<T, AudioError>;

/// Down-mixes interleaved multi-channel audio to mono by averaging.
///
/// Whisper is a mono model; Demucs takes stereo. Averaging rather than taking
/// channel 0 matters for real material, where a hard-panned voice would
/// otherwise vanish entirely.
pub fn to_mono(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Linearly resamples to `target_rate`.
///
/// Adequate for Whisper's 16 kHz frontend, whose mel filters discard the
/// aliasing artefacts a linear resampler leaves above ~7 kHz. It is NOT
/// adequate for separation, whose output is listened to - that path resamples
/// with a windowed-sinc kernel instead.
pub fn resample_linear(samples: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    if source_rate == target_rate || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = target_rate as f64 / source_rate as f64;
    let new_len = ((samples.len() as f64) * ratio).round() as usize;
    let last = samples.len() - 1;
    (0..new_len)
        .map(|i| {
            let src = i as f64 / ratio;
            let i0 = (src.floor() as usize).min(last);
            let i1 = (i0 + 1).min(last);
            let frac = (src - i0 as f64) as f32;
            samples[i0] * (1.0 - frac) + samples[i1] * frac
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mono_averages_rather_than_taking_channel_zero() {
        // A signal hard-panned to the right channel must survive the downmix.
        let stereo = vec![0.0, 1.0, 0.0, 1.0];
        let mono = to_mono(&stereo, 2);
        assert_eq!(mono, vec![0.5, 0.5]);
        assert!(mono.iter().any(|&v| v != 0.0), "hard-panned audio was lost");
    }

    #[test]
    fn mono_passthrough_is_identity() {
        assert_eq!(to_mono(&[0.1, 0.2], 1), vec![0.1, 0.2]);
    }

    #[test]
    fn resample_preserves_duration() {
        let src = vec![0.0f32; 48_000]; // 1 s at 48 kHz
        let out = resample_linear(&src, 48_000, 16_000);
        assert_eq!(out.len(), 16_000, "1 second must stay 1 second");
    }

    #[test]
    fn resample_same_rate_is_identity() {
        let src = vec![0.1, 0.2, 0.3];
        assert_eq!(resample_linear(&src, 16_000, 16_000), src);
    }

    /// A resampler must preserve FREQUENCY, not merely sample count. A 1 kHz
    /// tone resampled 48k -> 16k must still be 1 kHz, which means it must land
    /// in the same mel band.
    #[test]
    fn resample_preserves_pitch() {
        let sr_in = 48_000usize;
        let tone: Vec<f32> = (0..sr_in)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / sr_in as f32).sin())
            .collect();
        let out = resample_linear(&tone, sr_in as u32, 16_000);

        let stft = stft::Stft::new(mel::WHISPER_N_FFT, mel::WHISPER_HOP);
        let power = stft.power(&out);
        let mid = &power[power.len() / 2];
        let peak = mid
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        // 1000 Hz at 16 kHz / 400-point FFT is bin 25.
        assert_eq!(peak, 25, "1 kHz did not survive the resample; peaked at bin {peak}");
    }

    #[test]
    fn empty_input_is_handled() {
        assert!(resample_linear(&[], 48_000, 16_000).is_empty());
        assert!(to_mono(&[], 2).is_empty());
    }
}
