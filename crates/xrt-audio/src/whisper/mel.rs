//! Whisper log-mel front end, matching OpenAI's reference (and the
//! Hugging Face `WhisperFeatureExtractor`) exactly: 16 kHz audio padded or
//! cut to 30 s, STFT (n_fft 400, hop 160, periodic Hann, centred with
//! reflect padding, last frame dropped), power spectrum, 80 Slaney mel bands
//! (0–8 kHz, Slaney area normalisation), log10 floored at 1e-10, clamped to
//! 8 below the maximum, then `(x + 4) / 4`. Pinned against the reference by
//! `tests/whisper_mel.rs`.

use std::sync::OnceLock;

use rustfft::{num_complex::Complex, FftPlanner};

pub const SAMPLE_RATE: u32 = 16_000;
pub const N_FFT: usize = 400;
pub const HOP: usize = 160;
pub const N_MELS: usize = 80;
pub const N_SAMPLES: usize = 480_000; // 30 s
pub const N_FRAMES: usize = 3000;

/// `[N_MELS * N_FRAMES]`, row-major (mel band major), ready for `[1, 80, 3000]`.
pub fn log_mel(audio: &[f32]) -> Vec<f32> {
    let mut x = vec![0.0f32; N_SAMPLES];
    let n = audio.len().min(N_SAMPLES);
    x[..n].copy_from_slice(&audio[..n]);

    // Centre with reflect padding of n_fft/2 on each side.
    let pad = N_FFT / 2;
    let mut padded = Vec::with_capacity(N_SAMPLES + 2 * pad);
    for i in (1..=pad).rev() {
        padded.push(x[i]);
    }
    padded.extend_from_slice(&x);
    for i in 1..=pad {
        padded.push(x[N_SAMPLES - 1 - i]);
    }

    let window: Vec<f32> = (0..N_FFT)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64).cos() as f32)
        .collect();
    let fft = FftPlanner::<f32>::new().plan_fft_forward(N_FFT);
    let bins = N_FFT / 2 + 1;
    let filters = mel_filters();

    let mut mel = vec![0.0f32; N_MELS * N_FRAMES];
    let mut buf = vec![Complex::new(0.0f32, 0.0); N_FFT];
    let mut power = vec![0.0f32; bins];
    for t in 0..N_FRAMES {
        let start = t * HOP;
        for (k, c) in buf.iter_mut().enumerate() {
            *c = Complex::new(padded[start + k] * window[k], 0.0);
        }
        fft.process(&mut buf);
        for (b, p) in power.iter_mut().enumerate() {
            *p = buf[b].norm_sqr();
        }
        for m in 0..N_MELS {
            let row = &filters[m * bins..(m + 1) * bins];
            let v: f32 = row.iter().zip(&power).map(|(w, p)| w * p).sum();
            mel[m * N_FRAMES + t] = v;
        }
    }
    let mut max = f32::NEG_INFINITY;
    for v in mel.iter_mut() {
        *v = v.max(1e-10).log10();
        max = max.max(*v);
    }
    for v in mel.iter_mut() {
        *v = (v.max(max - 8.0) + 4.0) / 4.0;
    }
    mel
}

/// Slaney-scale mel filterbank, `[N_MELS * (N_FFT/2+1)]`, as librosa builds
/// it with `htk=False, norm="slaney"` (what Whisper's filters were made with).
fn mel_filters() -> &'static [f32] {
    static F: OnceLock<Vec<f32>> = OnceLock::new();
    F.get_or_init(|| {
        let bins = N_FFT / 2 + 1;
        let fft_freqs: Vec<f64> = (0..bins)
            .map(|i| i as f64 * SAMPLE_RATE as f64 / N_FFT as f64)
            .collect();
        let (min_mel, max_mel) = (hz_to_mel(0.0), hz_to_mel(SAMPLE_RATE as f64 / 2.0));
        let mel_pts: Vec<f64> = (0..N_MELS + 2)
            .map(|i| mel_to_hz(min_mel + (max_mel - min_mel) * i as f64 / (N_MELS + 1) as f64))
            .collect();
        let mut w = vec![0.0f32; N_MELS * bins];
        for m in 0..N_MELS {
            let (lo, c, hi) = (mel_pts[m], mel_pts[m + 1], mel_pts[m + 2]);
            let enorm = 2.0 / (hi - lo);
            for (b, &f) in fft_freqs.iter().enumerate() {
                let lower = (f - lo) / (c - lo);
                let upper = (hi - f) / (hi - c);
                let v = lower.min(upper).max(0.0) * enorm;
                w[m * bins + b] = v as f32;
            }
        }
        w
    })
}

fn hz_to_mel(f: f64) -> f64 {
    let (f_sp, min_log_hz) = (200.0 / 3.0, 1000.0);
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    if f >= min_log_hz {
        min_log_mel + (f / min_log_hz).ln() / logstep
    } else {
        f / f_sp
    }
}

fn mel_to_hz(m: f64) -> f64 {
    let (f_sp, min_log_hz) = (200.0 / 3.0, 1000.0);
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    if m >= min_log_mel {
        min_log_hz * (logstep * (m - min_log_mel)).exp()
    } else {
        f_sp * m
    }
}
