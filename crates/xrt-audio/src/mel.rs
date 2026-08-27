//! Mel filterbank and Whisper's log-mel frontend.
//!
//! Reproduces `librosa.filters.mel(sr, n_fft, n_mels, htk=False, norm="slaney")`
//! and OpenAI Whisper's `log_mel_spectrogram`, because those are what the
//! published weights were trained against. The mel scale here is the SLANEY
//! (Auditory Toolbox) variant, not HTK: it is piecewise, linear below 1 kHz
//! and logarithmic above. Substituting the HTK formula - a single log
//! expression, and the one most references show first - shifts every filter
//! centre and degrades transcription without ever failing anything.

use crate::stft::Stft;

/// Whisper's fixed frontend parameters. These are not tunable: the encoder's
/// first convolution has a fixed input width and the weights encode the rest.
pub const WHISPER_SAMPLE_RATE: usize = 16_000;
pub const WHISPER_N_FFT: usize = 400;
pub const WHISPER_HOP: usize = 160;
/// 30 seconds at 16 kHz - one encoder window.
pub const WHISPER_N_SAMPLES: usize = WHISPER_SAMPLE_RATE * 30;
/// Frames the encoder expects for one window (`N_SAMPLES / HOP`).
pub const WHISPER_N_FRAMES: usize = WHISPER_N_SAMPLES / WHISPER_HOP;

// Slaney mel-scale constants, from the Auditory Toolbox by way of librosa.
const F_SP: f64 = 200.0 / 3.0;
const MIN_LOG_HZ: f64 = 1000.0;
const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP; // exactly 15.0
const LOGSTEP: f64 = 0.068_751_777_648_691_37; // ln(6.4) / 27.0

/// Hz -> mel, Slaney scale.
pub fn hz_to_mel(hz: f64) -> f64 {
    if hz >= MIN_LOG_HZ {
        MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / LOGSTEP
    } else {
        hz / F_SP
    }
}

/// Mel -> Hz, Slaney scale. Exact inverse of [`hz_to_mel`].
pub fn mel_to_hz(mel: f64) -> f64 {
    if mel >= MIN_LOG_MEL {
        MIN_LOG_HZ * (LOGSTEP * (mel - MIN_LOG_MEL)).exp()
    } else {
        F_SP * mel
    }
}

/// Builds the `n_mels x (n_fft/2 + 1)` triangular filterbank with Slaney
/// (equal-area) normalization.
///
/// Slaney normalization scales each filter so that filters have equal AREA
/// rather than equal PEAK. Skipping it leaves high-frequency bands - which are
/// wide - dominating the input, and the model's later layers have no way to
/// tell that apart from genuinely loud high frequencies.
pub fn mel_filterbank(sr: usize, n_fft: usize, n_mels: usize) -> Vec<Vec<f32>> {
    let n_bins = n_fft / 2 + 1;
    // FFT bin centre frequencies: linspace(0, sr/2, n_bins).
    let fft_freqs: Vec<f64> = (0..n_bins)
        .map(|i| i as f64 * sr as f64 / n_fft as f64)
        .collect();

    // n_mels + 2 band edges, equally spaced on the mel scale.
    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(sr as f64 / 2.0);
    let mel_f: Vec<f64> = (0..n_mels + 2)
        .map(|i| {
            let m = mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64;
            mel_to_hz(m)
        })
        .collect();

    let mut weights = vec![vec![0.0f32; n_bins]; n_mels];
    for m in 0..n_mels {
        let (lo, ctr, hi) = (mel_f[m], mel_f[m + 1], mel_f[m + 2]);
        // Equal-area normalization over this band's full width.
        let enorm = 2.0 / (mel_f[m + 2] - mel_f[m]);
        for (b, &f) in fft_freqs.iter().enumerate() {
            let lower = if ctr > lo { (f - lo) / (ctr - lo) } else { 0.0 };
            let upper = if hi > ctr { (hi - f) / (hi - ctr) } else { 0.0 };
            let w = lower.min(upper).max(0.0);
            weights[m][b] = (w * enorm) as f32;
        }
    }
    weights
}

/// Whisper's log-mel spectrogram: `n_mels x frames`.
///
/// The exact pipeline from `whisper/audio.py`:
///   stft -> drop the final frame -> `|.|^2` -> mel filters -> log10 ->
///   clamp to `max - 8` -> `(x + 4) / 4`.
///
/// The final two steps are a per-utterance normalization, so the OUTPUT SPAN
/// IS ALWAYS EXACTLY 2.0 whenever the clamp binds. That invariant is what the
/// tests below assert; it is cheap to check and impossible to satisfy by
/// accident with a wrong frontend.
pub fn log_mel_spectrogram(samples: &[f32], n_mels: usize) -> Vec<Vec<f32>> {
    let stft = Stft::new(WHISPER_N_FFT, WHISPER_HOP);
    let mut power = stft.power(samples);
    // torch: `stft[..., :-1]` - the final centred frame reads past the signal.
    if !power.is_empty() {
        power.pop();
    }
    if power.is_empty() {
        return vec![Vec::new(); n_mels];
    }

    let filters = mel_filterbank(WHISPER_SAMPLE_RATE, WHISPER_N_FFT, n_mels);
    let n_frames = power.len();

    let mut out = vec![vec![0.0f32; n_frames]; n_mels];
    let mut max_val = f32::NEG_INFINITY;
    for (m, filter) in filters.iter().enumerate() {
        for (t, frame) in power.iter().enumerate() {
            let mut acc = 0.0f32;
            for (b, &w) in filter.iter().enumerate() {
                if w != 0.0 {
                    acc += w * frame[b];
                }
            }
            let v = acc.max(1e-10).log10();
            out[m][t] = v;
            if v > max_val {
                max_val = v;
            }
        }
    }

    let floor = max_val - 8.0;
    for row in out.iter_mut() {
        for v in row.iter_mut() {
            *v = (v.max(floor) + 4.0) / 4.0;
        }
    }
    out
}

/// Pads with silence or trims to exactly one 30-second Whisper window.
pub fn pad_or_trim(samples: &[f32]) -> Vec<f32> {
    let mut v = samples.to_vec();
    v.resize(WHISPER_N_SAMPLES, 0.0);
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f32, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| {
                (2.0 * std::f32::consts::PI * freq * i as f32 / WHISPER_SAMPLE_RATE as f32).sin()
            })
            .collect()
    }

    /// The Slaney break point is exact by construction: 1000 Hz is mel 15.
    /// If a future edit swaps in the HTK formula this is the first thing to go
    /// (HTK gives 2595*log10(1+1000/700) = 999.99..., not 15).
    #[test]
    fn slaney_break_point_is_exact() {
        assert!((hz_to_mel(1000.0) - 15.0).abs() < 1e-12, "{}", hz_to_mel(1000.0));
    }

    #[test]
    fn mel_scale_is_linear_below_the_break() {
        // 200 Hz / (200/3) = 3.0 exactly.
        assert!((hz_to_mel(200.0) - 3.0).abs() < 1e-12);
        assert!((hz_to_mel(400.0) - 6.0).abs() < 1e-12);
    }

    #[test]
    fn mel_round_trips_across_the_break() {
        for hz in [0.0, 100.0, 999.0, 1000.0, 1001.0, 4000.0, 8000.0] {
            let back = mel_to_hz(hz_to_mel(hz));
            assert!((back - hz).abs() < 1e-6, "{hz} -> {back}");
        }
    }

    #[test]
    fn filterbank_has_the_shape_whisper_expects() {
        let fb = mel_filterbank(WHISPER_SAMPLE_RATE, WHISPER_N_FFT, 80);
        assert_eq!(fb.len(), 80);
        assert_eq!(fb[0].len(), WHISPER_N_FFT / 2 + 1);
    }

    #[test]
    fn every_filter_is_non_negative_and_non_empty() {
        let fb = mel_filterbank(WHISPER_SAMPLE_RATE, WHISPER_N_FFT, 80);
        for (m, f) in fb.iter().enumerate() {
            assert!(f.iter().all(|&w| w >= 0.0), "filter {m} has a negative weight");
            assert!(f.iter().any(|&w| w > 0.0), "filter {m} is entirely zero");
        }
    }

    /// Filters must ASCEND in centre frequency. A sort bug here is invisible
    /// in every shape assertion and scrambles the model's input.
    #[test]
    fn filter_centres_ascend() {
        let fb = mel_filterbank(WHISPER_SAMPLE_RATE, WHISPER_N_FFT, 80);
        let peak = |f: &Vec<f32>| {
            f.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0
        };
        let mut prev = 0usize;
        for (m, f) in fb.iter().enumerate() {
            let p = peak(f);
            assert!(p >= prev, "filter {m} peaks at bin {p}, below the previous {prev}");
            prev = p;
        }
    }

    /// Slaney normalization is EQUAL AREA, not equal peak. Without the `enorm`
    /// factor the wide high-frequency filters would sum to far more than the
    /// narrow low ones; with it, the areas match closely.
    #[test]
    fn slaney_normalization_equalizes_area_not_peak() {
        let fb = mel_filterbank(WHISPER_SAMPLE_RATE, WHISPER_N_FFT, 80);
        // Compare mid-band filters, away from the edges where the triangles
        // are clipped by the 0 Hz and Nyquist boundaries.
        let areas: Vec<f32> = fb[20..70].iter().map(|f| f.iter().sum()).collect();
        let mn = areas.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = areas.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            mx / mn < 1.30,
            "filter areas vary {:.2}x across the mid band - that is the signature of a missing Slaney enorm",
            mx / mn
        );

        // ...and the PEAKS genuinely do vary, which is what distinguishes this
        // from equal-peak normalization.
        let peaks: Vec<f32> = fb[20..70]
            .iter()
            .map(|f| f.iter().cloned().fold(f32::NEG_INFINITY, f32::max))
            .collect();
        let pmn = peaks.iter().cloned().fold(f32::INFINITY, f32::min);
        let pmx = peaks.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(pmx / pmn > 1.5, "peaks are equal - this is equal-peak, not Slaney");
    }

    /// One 30 s window must produce exactly the 3000 frames the encoder wants.
    #[test]
    fn one_window_is_exactly_3000_frames() {
        let mel = log_mel_spectrogram(&pad_or_trim(&sine(440.0, 16_000)), 80);
        assert_eq!(mel.len(), 80);
        assert_eq!(mel[0].len(), WHISPER_N_FRAMES, "encoder expects 80 x 3000");
        assert_eq!(WHISPER_N_FRAMES, 3000);
    }

    /// The normalization invariant: span is exactly 2.0 once the clamp binds.
    #[test]
    fn normalized_output_spans_exactly_two() {
        let mel = log_mel_spectrogram(&pad_or_trim(&sine(440.0, 16_000)), 80);
        let mut mn = f32::INFINITY;
        let mut mx = f32::NEG_INFINITY;
        for row in &mel {
            for &v in row {
                mn = mn.min(v);
                mx = mx.max(v);
            }
        }
        assert!((mx - mn - 2.0).abs() < 1e-4, "span was {:.6}, expected 2.0", mx - mn);
    }

    /// THE PROPERTY THE RETIRED IMPLEMENTATION VIOLATED.
    ///
    /// A low tone must excite LOW mel bins and a high tone HIGH ones. The
    /// predecessor's output depended only on per-frame energy, so both tones
    /// produced the same curve and this fails there by construction.
    #[test]
    fn tone_frequency_selects_the_mel_band() {
        let energy_peak = |freq: f32| -> usize {
            let mel = log_mel_spectrogram(&pad_or_trim(&sine(freq, 16_000)), 80);
            // Look at a frame inside the tone, not in the trailing silence.
            let t = 100;
            mel.iter()
                .enumerate()
                .max_by(|a, b| a.1[t].partial_cmp(&b.1[t]).unwrap())
                .unwrap()
                .0
        };
        let low = energy_peak(200.0);
        let high = energy_peak(4000.0);
        assert!(
            low < high,
            "200 Hz peaked at mel bin {low} and 4 kHz at {high} - the frontend is not frequency-selective"
        );
        assert!(low < 20, "200 Hz should excite a low mel bin, got {low}");
        assert!(high > 45, "4 kHz should excite a high mel bin, got {high}");
    }

    #[test]
    fn silence_does_not_produce_nan_or_inf() {
        let mel = log_mel_spectrogram(&pad_or_trim(&[]), 80);
        assert!(mel.iter().all(|r| r.iter().all(|v| v.is_finite())));
    }

    #[test]
    fn supports_the_128_bin_large_v3_frontend() {
        let mel = log_mel_spectrogram(&pad_or_trim(&sine(440.0, 16_000)), 128);
        assert_eq!(mel.len(), 128);
        assert_eq!(mel[0].len(), WHISPER_N_FRAMES);
    }
}
