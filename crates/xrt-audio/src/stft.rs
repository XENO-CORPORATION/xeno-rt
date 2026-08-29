//! Short-time Fourier transform.
//!
//! Shared by every audio task in this crate: Whisper's log-mel frontend needs
//! magnitudes, Demucs needs the complex spectrogram and an inverse. Both are
//! built on the one implementation here so they cannot drift apart.
//!
//! THE PREDECESSOR HAD NO FFT. xeno-lib's retired `audio_to_mel` summed
//! `s*s` over each frame and multiplied that single scalar by `sin(bin/n * PI)`
//! to "distribute across mel bins". Every mel bin of a frame therefore carried
//! the same number scaled by a fixed curve, with no dependence on frequency
//! whatsoever - white noise and a pure tone of equal power produce identical
//! output. It is not an approximation of a spectrogram; it is unrelated to one.
//!
//! Conventions match `torch.stft(center=True, pad_mode="reflect",
//! onesided=True, normalized=False)`, because that is what OpenAI's Whisper
//! reference implementation calls and the model's weights were trained on its
//! output. A frontend that is merely "a reasonable spectrogram" produces
//! confident nonsense from a correct model.

use rustfft::{num_complex::Complex32, FftPlanner};

/// A periodic Hann window, matching `torch.hann_window(n)`.
///
/// PERIODIC, not symmetric - the divisor is `n`, not `n - 1`. SciPy's
/// `hann(n)` defaults to the symmetric form, so a window borrowed from there
/// is subtly wrong at every sample and produces spectral leakage that looks
/// like a slightly noisy model rather than a bug.
pub fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (2.0 * std::f64::consts::PI * i as f64) / n as f64;
            (0.5 - 0.5 * x.cos()) as f32
        })
        .collect()
}

/// Reflect-pads `samples` by `pad` on both ends, as `center=True` requires.
///
/// Reflection excludes the edge sample itself (`abcd` -> `cba|abcd|dcb`),
/// which is NumPy's `mode="reflect"` and torch's `pad_mode="reflect"`.
pub fn reflect_pad(samples: &[f32], pad: usize) -> Vec<f32> {
    if samples.is_empty() {
        return vec![0.0; pad * 2];
    }
    if samples.len() == 1 {
        // Reflection is undefined with a single sample; repeating it is the
        // only continuation that keeps the value and cannot index out of range.
        return vec![samples[0]; samples.len() + pad * 2];
    }
    let mut out = Vec::with_capacity(samples.len() + pad * 2);
    for i in (1..=pad).rev() {
        out.push(samples[i.min(samples.len() - 1)]);
    }
    out.extend_from_slice(samples);
    let last = samples.len() - 1;
    for i in 1..=pad {
        out.push(samples[last.saturating_sub(i)]);
    }
    out
}

/// A reusable STFT plan. Building the FFT plan is the expensive part, so it is
/// created once and reused across frames and across calls.
pub struct Stft {
    n_fft: usize,
    hop: usize,
    window: Vec<f32>,
    planner: parking_lot::Mutex<FftPlanner<f32>>,
}

impl Stft {
    pub fn new(n_fft: usize, hop: usize) -> Self {
        Self {
            n_fft,
            hop,
            window: hann_window(n_fft),
            planner: parking_lot::Mutex::new(FftPlanner::new()),
        }
    }

    /// Number of one-sided frequency bins produced per frame.
    pub fn n_bins(&self) -> usize {
        self.n_fft / 2 + 1
    }

    /// Computes the one-sided complex spectrogram.
    ///
    /// Returns `frames x n_bins`, centred (each frame `t` is centred on sample
    /// `t * hop` of the input).
    pub fn forward(&self, samples: &[f32]) -> Vec<Vec<Complex32>> {
        let padded = reflect_pad(samples, self.n_fft / 2);
        if padded.len() < self.n_fft {
            return Vec::new();
        }
        let n_frames = 1 + (padded.len() - self.n_fft) / self.hop;
        let fft = self.planner.lock().plan_fft_forward(self.n_fft);

        let mut out = Vec::with_capacity(n_frames);
        let mut buf = vec![Complex32::new(0.0, 0.0); self.n_fft];
        for f in 0..n_frames {
            let start = f * self.hop;
            for i in 0..self.n_fft {
                buf[i] = Complex32::new(padded[start + i] * self.window[i], 0.0);
            }
            fft.process(&mut buf);
            out.push(buf[..self.n_bins()].to_vec());
        }
        out
    }

    /// Power spectrogram: `|STFT|^2`, the quantity Whisper's mel filters expect.
    pub fn power(&self, samples: &[f32]) -> Vec<Vec<f32>> {
        self.forward(samples)
            .into_iter()
            .map(|frame| frame.into_iter().map(|c| c.norm_sqr()).collect())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: usize = 16_000;
    const N_FFT: usize = 400;
    const HOP: usize = 160;

    fn sine(freq: f32, len: usize, sr: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr as f32).sin())
            .collect()
    }

    #[test]
    fn hann_window_is_periodic_not_symmetric() {
        let w = hann_window(4);
        // Periodic: [0, 0.5, 1, 0.5]. Symmetric would be [0, 0.75, 0.75, 0].
        assert!((w[0] - 0.0).abs() < 1e-6, "w[0]={}", w[0]);
        assert!((w[1] - 0.5).abs() < 1e-6, "w[1]={}", w[1]);
        assert!((w[2] - 1.0).abs() < 1e-6, "w[2]={}", w[2]);
        assert!((w[3] - 0.5).abs() < 1e-6, "w[3]={}", w[3]);
    }

    #[test]
    fn reflect_pad_excludes_the_edge_sample() {
        // NumPy: np.pad([1,2,3,4], 2, mode="reflect") -> [3,2,1,2,3,4,3,2]
        let p = reflect_pad(&[1.0, 2.0, 3.0, 4.0], 2);
        assert_eq!(p, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]);
    }

    /// THE TEST THE PREDECESSOR COULD NOT HAVE PASSED.
    ///
    /// A bin-centred tone through a Hann window has an EXACT three-bin
    /// signature, and asserting it verifies the FFT and the window together.
    /// The Hann window splits a pure tone into amplitudes `0.25, 0.5, 0.25`
    /// over bins `k-1, k, k+1`; squaring gives powers `1/16, 1/4, 1/16`, so
    /// the shares are exactly **2/3 and 1/6 and 1/6**.
    ///
    /// The retired implementation spread every frame's scalar energy across
    /// all outputs by a fixed frequency-independent curve, so it fails this by
    /// construction - which is precisely why a shape assertion was the only
    /// thing it could be tested with.
    ///
    /// A NOTE ON HOW THIS TEST GOT ITS NUMBER. It first asserted ">90% in the
    /// peak bin" and failed at 66.7% against a correct implementation. 66.7%
    /// is 2/3 - the exact right answer - so the gate was wrong, not the code.
    /// Reading the value rather than trusting the boolean is what caught it;
    /// a threshold picked by intuition would have sent someone to "fix" a
    /// working FFT.
    #[test]
    fn a_pure_tone_has_the_exact_hann_three_bin_signature() {
        // Bin width = SR / N_FFT = 40 Hz. Bin 25 is exactly 1000 Hz.
        let bin = 25usize;
        let freq = (SR / N_FFT * bin) as f32; // 1000.0
        let stft = Stft::new(N_FFT, HOP);
        let power = stft.power(&sine(freq, SR, SR));

        let mid = &power[power.len() / 2];
        let total: f32 = mid.iter().sum();
        assert!(total > 0.0, "silent spectrum");

        let share = |b: usize| mid[b] / total;
        assert!(
            (share(bin) - 2.0 / 3.0).abs() < 0.01,
            "peak bin share {:.4}, expected 2/3 = 0.6667",
            share(bin)
        );
        assert!(
            (share(bin - 1) - 1.0 / 6.0).abs() < 0.01,
            "lower neighbour share {:.4}, expected 1/6 = 0.1667",
            share(bin - 1)
        );
        assert!(
            (share(bin + 1) - 1.0 / 6.0).abs() < 0.01,
            "upper neighbour share {:.4}, expected 1/6 = 0.1667",
            share(bin + 1)
        );

        // Those three bins account for essentially everything: outside the
        // main lobe a Hann window leaves under 0.5% total.
        let lobe: f32 = share(bin - 1) + share(bin) + share(bin + 1);
        assert!(
            lobe > 0.995,
            "main lobe held only {:.3} of the energy",
            lobe
        );
    }

    /// Two different tones must produce DIFFERENT spectra. This is the
    /// property the retired code violated most starkly: its output depended
    /// only on frame energy, so any two equal-amplitude signals were identical.
    #[test]
    fn different_frequencies_produce_different_spectra() {
        let stft = Stft::new(N_FFT, HOP);
        let a = stft.power(&sine(440.0, SR, SR));
        let b = stft.power(&sine(3000.0, SR, SR));

        let fa = &a[a.len() / 2];
        let fb = &b[b.len() / 2];
        let peak = |v: &Vec<f32>| {
            v.iter()
                .enumerate()
                .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
                .unwrap()
                .0
        };
        assert_ne!(
            peak(fa),
            peak(fb),
            "440 Hz and 3 kHz peaked in the same bin"
        );

        // ...and the total energies are comparable, so a test that only looked
        // at frame energy (as the predecessor effectively did) would see these
        // two signals as the same thing.
        let ea: f32 = fa.iter().sum();
        let eb: f32 = fb.iter().sum();
        let ratio = ea.max(eb) / ea.min(eb);
        assert!(
            ratio < 1.5,
            "energies differ by {ratio:.2}x - the point of this test is that they are similar while the spectra are not"
        );
    }

    #[test]
    fn frame_count_matches_the_centered_convention() {
        let stft = Stft::new(N_FFT, HOP);
        let n = SR; // 1 second
        let frames = stft.forward(&vec![0.0; n]).len();
        // center=True => padded length n + n_fft, so 1 + n/hop frames.
        assert_eq!(frames, 1 + n / HOP);
    }

    #[test]
    fn bin_count_is_one_sided() {
        let stft = Stft::new(N_FFT, HOP);
        assert_eq!(stft.n_bins(), N_FFT / 2 + 1);
        assert_eq!(stft.forward(&vec![0.0; SR])[0].len(), N_FFT / 2 + 1);
    }
}
