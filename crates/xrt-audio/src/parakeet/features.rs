//! NeMo log-mel front end, as `AudioToMelSpectrogramPreprocessor` builds it for
//! the Parakeet models: pre-emphasis 0.97, a symmetric Hann window of 25 ms
//! centred in a 512-point FFT, 10 ms hop, centred with ZERO padding, power
//! spectrum, Slaney mel bands over 0-8 kHz, natural log with a 2^-24 guard, then
//! per-band mean/std normalisation over the clip.
//!
//! This is not Whisper's front end and sharing it would decode to noise: the
//! FFT size, window, padding, log base and normalisation all differ.

use rustfft::{num_complex::Complex, FftPlanner};

pub const SAMPLE_RATE: u32 = 16_000;
const N_FFT: usize = 512;
const WIN: usize = 400;
const HOP: usize = 160;
const PREEMPH: f32 = 0.97;
const LOG_GUARD: f32 = 5.960_464_5e-8; // 2^-24
const STD_FLOOR: f32 = 1e-5;

/// Mel features for one clip. `frames` is `samples / HOP`; NeMo drops the
/// partial frame the centred STFT would add.
pub struct Features {
    /// `[n_mels * frames]`, mel band major, ready for `[1, n_mels, frames]`.
    pub values: Vec<f32>,
    pub frames: usize,
}

pub fn log_mel(audio: &[f32], n_mels: usize) -> Features {
    let frames = audio.len() / HOP;
    if frames < 2 {
        return Features {
            values: Vec::new(),
            frames: 0,
        };
    }

    let mut x = Vec::with_capacity(audio.len() + N_FFT);
    x.resize(N_FFT / 2, 0.0f32);
    x.push(audio[0]);
    x.extend(audio.windows(2).map(|w| w[1] - PREEMPH * w[0]));
    x.resize(x.len() + N_FFT / 2, 0.0);

    let window = hann();
    let filters = mel_filters(n_mels);
    let bins = N_FFT / 2 + 1;
    let fft = FftPlanner::<f32>::new().plan_fft_forward(N_FFT);
    let mut buf = vec![Complex::new(0.0f32, 0.0); N_FFT];
    let mut values = vec![0.0f32; n_mels * frames];
    for t in 0..frames {
        let start = t * HOP;
        for (k, c) in buf.iter_mut().enumerate() {
            *c = Complex::new(x[start + k] * window[k], 0.0);
        }
        fft.process(&mut buf);
        for m in 0..n_mels {
            let row = &filters[m * bins..(m + 1) * bins];
            let energy: f32 = row.iter().zip(&buf).map(|(w, c)| w * c.norm_sqr()).sum();
            values[m * frames + t] = (energy + LOG_GUARD).ln();
        }
    }

    for band in values.chunks_exact_mut(frames) {
        let mean = band.iter().sum::<f32>() / frames as f32;
        let variance = band.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / (frames - 1) as f32;
        let std = variance.sqrt() + STD_FLOOR;
        for v in band.iter_mut() {
            *v = (*v - mean) / std;
        }
    }
    Features { values, frames }
}

/// Symmetric Hann of `WIN` samples, zero-padded to `N_FFT` on both sides.
fn hann() -> Vec<f32> {
    let mut window = vec![0.0f32; N_FFT];
    let offset = (N_FFT - WIN) / 2;
    for i in 0..WIN {
        let phase = 2.0 * std::f64::consts::PI * i as f64 / (WIN - 1) as f64;
        window[offset + i] = (0.5 - 0.5 * phase.cos()) as f32;
    }
    window
}

/// Slaney mel filterbank, `[n_mels * (N_FFT/2+1)]`, as librosa builds it with
/// `htk=False, norm="slaney"`.
fn mel_filters(n_mels: usize) -> Vec<f32> {
    let bins = N_FFT / 2 + 1;
    let (lo, hi) = (hz_to_mel(0.0), hz_to_mel(SAMPLE_RATE as f64 / 2.0));
    let points: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (n_mels + 1) as f64))
        .collect();
    let mut filters = vec![0.0f32; n_mels * bins];
    for m in 0..n_mels {
        let (left, centre, right) = (points[m], points[m + 1], points[m + 2]);
        let area = 2.0 / (right - left);
        for b in 0..bins {
            let f = b as f64 * SAMPLE_RATE as f64 / N_FFT as f64;
            let rise = (f - left) / (centre - left);
            let fall = (right - f) / (right - centre);
            filters[m * bins + b] = (rise.min(fall).max(0.0) * area) as f32;
        }
    }
    filters
}

fn hz_to_mel(f: f64) -> f64 {
    let (f_sp, break_hz) = (200.0 / 3.0, 1000.0);
    let step = (6.4f64).ln() / 27.0;
    if f >= break_hz {
        break_hz / f_sp + (f / break_hz).ln() / step
    } else {
        f / f_sp
    }
}

fn mel_to_hz(m: f64) -> f64 {
    let (f_sp, break_hz) = (200.0 / 3.0, 1000.0);
    let step = (6.4f64).ln() / 27.0;
    if m >= break_hz / f_sp {
        break_hz * (step * (m - break_hz / f_sp)).exp()
    } else {
        f_sp * m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_clip_shorter_than_two_hops_has_no_frames() {
        assert_eq!(log_mel(&vec![0.1; 300], 80).frames, 0);
        assert_eq!(log_mel(&[], 80).frames, 0);
    }

    #[test]
    fn frames_follow_the_clip_length_not_the_padded_stft() {
        let features = log_mel(&vec![0.1; 16_000], 128);
        assert_eq!(features.frames, 100);
        assert_eq!(features.values.len(), 12_800);
    }

    /// Normalisation is per band: every band ends with mean 0 and std ~1 on a
    /// signal with energy in all of them.
    #[test]
    fn every_band_is_normalised() {
        let noise: Vec<f32> = (0..32_000)
            .map(|i| ((i as f32 * 12.9898).sin() * 43_758.547).fract())
            .collect();
        let features = log_mel(&noise, 80);
        for band in features.values.chunks_exact(features.frames) {
            let mean = band.iter().sum::<f32>() / band.len() as f32;
            let var =
                band.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / (band.len() - 1) as f32;
            assert!(mean.abs() < 1e-3, "mean {mean}");
            assert!((var.sqrt() - 1.0).abs() < 1e-2, "std {}", var.sqrt());
        }
    }

    #[test]
    fn the_window_is_symmetric_and_centred() {
        let w = hann();
        assert_eq!(w.len(), N_FFT);
        assert_eq!(w[0], 0.0);
        assert_eq!(w[N_FFT - 1], 0.0);
        for i in 0..N_FFT / 2 {
            assert!((w[i] - w[N_FFT - 1 - i]).abs() < 1e-6);
        }
    }
}
