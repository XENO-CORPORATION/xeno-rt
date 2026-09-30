//! Take trimming applied after generation.
//!
//! ⚠️ There is deliberately NO time-stretching here. A pitch-preserving tempo
//! change (WSOLA) was built and removed on 2026-09-26: listeners heard it
//! smear and warble individual words, and narration pace must come from the
//! performance itself — reference choice, presets, and pauses at real
//! sentence boundaries — never from resampling speech.
//!
//! ⚠️ Breath softening and pause lengthening are NOT here any more. Both
//! were sound-based (a level/spectrum detector deciding what is "breath" and
//! what is "pause") and the 2026-09-27 stage audit measured both altering
//! words: 8 word endings ("st", "s") turned down, 2 words changed. They now
//! live in `wordsafe`, driven by the recognizer's word times, and cannot
//! touch a sample inside a word. Designs that failed, so they are not
//! rebuilt: a hissy-and-quiet frame detector (caught only the fricative half
//! of a breath and also turned down word-edge fricatives); a fixed 150 ms
//! guard around speech (left the loudest breath energy, which sits against
//! the next word); extending a pause by looping its own contents (repeated
//! the breath inside it — an audible "whoosh").

/// Cut a take to its speech: from the first spoken sound to the end of the
/// last word's natural release, discarding anything outside.
///
/// Why: after the last word Chatterbox often keeps generating a steady tone
/// (measured 2026-09-27: ~1 kHz at -30 dBFS for up to 2 s after "else.") —
/// upstream stops this with an attention-based "long tail" detector the ONNX
/// export cannot run. Speech-relative trimming keeps every word intact while
/// removing that tail: the edges are anchored on loud VOICED frames, then
/// extended outward only while the sound stays continuous with the word
/// (a consonant onset or release rises/decays smoothly; a separate hum or
/// breath begins after a dip or a rise), capped at 250 ms before the first
/// word and 300 ms after the last, with a short fade at each cut.
pub fn trim_to_speech(x: &[f32], rate: u32) -> Vec<f32> {
    let frame = ((0.01 * rate as f32) as usize).max(1);
    let n = x.len() / frame;
    if n < 4 {
        return x.to_vec();
    }
    let rms: Vec<f32> = (0..n)
        .map(|i| {
            let f = &x[i * frame..(i + 1) * frame];
            (f.iter().map(|s| s * s).sum::<f32>() / frame as f32).sqrt()
        })
        .collect();
    let mut sorted: Vec<f32> = rms.iter().copied().filter(|&r| r > 1e-4).collect();
    if sorted.is_empty() {
        return Vec::new();
    }
    sorted.sort_by(|a, b| a.total_cmp(b));
    let level = sorted[((sorted.len() - 1) as f32 * 0.95) as usize];
    let loud = level * 10f32.powf(-18.0 / 20.0);
    let dip = level * 10f32.powf(-45.0 / 20.0);
    let anchor = |i: usize| -> bool {
        if rms[i] < loud {
            return false;
        }
        let f = &x[i * frame..(i + 1) * frame];
        let e: f32 = f.iter().map(|s| s * s).sum();
        let d: f32 = f.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum();
        e > 0.0 && d / e <= 1.2
    };
    let (Some(first), Some(last)) = (
        (0..n).find(|&i| anchor(i)),
        (0..n).rev().find(|&i| anchor(i)),
    ) else {
        return Vec::new();
    };
    let extend = |edge: usize, step: isize, max: usize| -> usize {
        let mut len = 0usize;
        let mut prev = rms[edge];
        while len < max {
            let k = edge as isize + step * (len as isize + 1);
            if k < 0 || k as usize >= n {
                break;
            }
            let r = rms[k as usize];
            if r < dip || r > prev * 1.6 {
                break;
            }
            prev = r.min(prev);
            len += 1;
        }
        len
    };
    let start = (first - extend(first, -1, 25)) * frame;
    let end = ((last + extend(last, 1, 30) + 1) * frame).min(x.len());
    let mut y = x[start..end].to_vec();
    let (fi, fo) = (
        (0.005 * rate as f32) as usize,
        (0.03 * rate as f32) as usize,
    );
    let len = y.len();
    for (k, sample) in y.iter_mut().take(fi).enumerate() {
        *sample *= k as f32 / fi as f32;
    }
    for k in 0..fo.min(len) {
        y[len - 1 - k] *= k as f32 / fo as f32;
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 24_000;

    #[test]
    fn a_tone_after_the_last_word_is_cut() {
        // Speech, a natural 100 ms decay, a dip, then a steady 1 kHz hum.
        let mut x = tone(1.0, 180.0);
        let speech_end = x.len();
        x.extend(
            (0..(0.1 * SR as f32) as usize)
                .map(|i| 0.5 * (1.0 - i as f32 / 2400.0) * (i as f32 * 0.05).sin()),
        );
        x.extend(vec![0.0; (0.08 * SR as f32) as usize]);
        x.extend(tone(1.5, 1000.0).iter().map(|s| s * 0.06));
        let y = trim_to_speech(&x, SR);
        assert!(y.len() >= speech_end, "speech was cut");
        assert!(
            y.len() < speech_end + (0.35 * SR as f32) as usize,
            "hum kept: {} samples past speech",
            y.len() - speech_end
        );
    }

    #[test]
    fn a_word_initial_s_is_kept_by_speech_trimming() {
        // leading silence, 120 ms of "s" rising into the vowel, speech
        let mut x = vec![0.0; (0.3 * SR as f32) as usize];
        let s_start = x.len();
        x.extend((0..(0.12 * SR as f32) as usize).map(|i| {
            let n = ((i as u64).wrapping_mul(2654435761) % 2001) as f32 / 1000.0 - 1.0;
            n * 0.05 * (0.3 + 0.7 * i as f32 / 2880.0)
        }));
        x.extend(tone(1.0, 180.0));
        let y = trim_to_speech(&x, SR);
        let kept_before_vowel = y.len() - (x.len() - s_start - (0.12 * SR as f32) as usize);
        assert!(
            kept_before_vowel >= (0.1 * SR as f32) as usize,
            "the 's' onset was cut: {kept_before_vowel}"
        );
    }

    fn tone(secs: f32, hz: f32) -> Vec<f32> {
        (0..(secs * SR as f32) as usize)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * hz * i as f32 / SR as f32).sin())
            .collect()
    }
}
