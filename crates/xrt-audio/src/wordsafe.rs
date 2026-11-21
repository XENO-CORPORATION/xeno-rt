//! Word-safe delivery shaping: breath softening and pause lengthening that
//! can only touch audio BETWEEN words.
//!
//! Why this replaced the sound-based versions (removed 2026-09-28): the
//! 2026-09-27 stage audit found them altering words — breath softening turned
//! down 8 word endings ("st", "s") and pause expansion changed 2 words —
//! because a word-final fricative and an in-breath look alike to a level or
//! spectrum detector. Here the words' positions come from the recognizer, so
//! the question "is this sample part of a word?" is answered by what was
//! SAID, not by what the sample sounds like.
//!
//! Recognizer times are not sample-exact: measured against real takes, a word
//! end can sit at the START of its final "st" (the release of "chest." ran
//! 130 ms past it). So each word's protected span grows outward from its
//! recognized times while the sound stays continuous with it — until two
//! consecutive 10 ms frames fall 45 dB below the take's speech level (the
//! closure inside "st" is shorter than that; the pause after the word is
//! not) — with at least 40 ms kept on each side.
//!
//! Every sample inside a protected span is left bit-identical; the tests
//! assert exactly that.

use crate::whisper::Word;

const FRAME_S: f32 = 0.01;
const MIN_GUARD_S: f32 = 0.04;
const MAX_AFTER_S: f32 = 0.30;
const MAX_BEFORE_S: f32 = 0.15;
const DIP_DB: f32 = 45.0;

/// Sample ranges `[a, b)` that belong to words, merged and sorted.
pub fn protected_spans(x: &[f32], rate: u32, words: &[Word]) -> Vec<(usize, usize)> {
    let frame = ((FRAME_S * rate as f32) as usize).max(1);
    let n = x.len() / frame;
    if n == 0 {
        return if words.is_empty() {
            Vec::new()
        } else {
            vec![(0, x.len())]
        };
    }
    let rms: Vec<f32> = (0..n)
        .map(|i| {
            let f = &x[i * frame..(i + 1) * frame];
            (f.iter().map(|s| s * s).sum::<f32>() / frame as f32).sqrt()
        })
        .collect();
    let mut sorted: Vec<f32> = rms.iter().copied().filter(|&r| r > 1e-5).collect();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let level = sorted
        .get(((sorted.len().max(1) - 1) as f32 * 0.95) as usize)
        .copied()
        .unwrap_or(0.0);
    let dip = level * 10f32.powf(-DIP_DB / 20.0);
    let min_guard = (MIN_GUARD_S * rate as f32) as usize;

    // Walk from `from` in `step` direction while the sound is continuous;
    // returns the frame index of the last frame that was not part of a dip.
    let extend = |from: isize, step: isize, max_frames: isize| -> isize {
        let mut last_loud = from - step;
        let mut quiet_run = 0;
        let mut k = from;
        while (k - from).abs() < max_frames && k >= 0 && (k as usize) < n {
            if rms[k as usize] >= dip {
                last_loud = k;
                quiet_run = 0;
            } else {
                quiet_run += 1;
                if quiet_run >= 2 {
                    break;
                }
            }
            k += step;
        }
        last_loud
    };

    let mut spans: Vec<(usize, usize)> = words
        .iter()
        .map(|w| {
            let s = ((w.start.max(0.0) * rate as f32) as usize).min(x.len());
            let e = ((w.end.max(w.start) * rate as f32).ceil() as usize).clamp(s, x.len());
            let after = extend((e / frame) as isize, 1, (MAX_AFTER_S / FRAME_S) as isize);
            let before = extend(
                (s / frame) as isize - 1,
                -1,
                (MAX_BEFORE_S / FRAME_S) as isize,
            );
            let hi = (e + min_guard)
                .max(((after + 1).max(0) as usize) * frame)
                .min(x.len());
            let lo = s
                .saturating_sub(min_guard)
                .min((before.max(0) as usize) * frame);
            (lo, hi)
        })
        .collect();
    spans.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(spans.len());
    for (a, b) in spans {
        match merged.last_mut() {
            Some(last) if a <= last.1 + frame => last.1 = last.1.max(b),
            _ => merged.push((a, b)),
        }
    }
    merged
}

/// The complement of `spans` in `[0, len)`: the audio between words.
pub fn gaps(spans: &[(usize, usize)], len: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(spans.len() + 1);
    let mut at = 0;
    for &(a, b) in spans {
        if a > at {
            out.push((at, a));
        }
        at = at.max(b);
    }
    if at < len {
        out.push((at, len));
    }
    out
}

/// Lower everything between words by `reduction_db` — the model's in-breaths
/// live there — with 20 ms ramps placed inside each gap so the gain is back
/// at 1.0 on the first protected sample. Gaps shorter than 120 ms are left
/// alone (no breath fits; it is coarticulation). Returns the audio and how
/// many gaps were softened. The breath is kept, only quieter: deleting it
/// outright sounds edited.
pub fn soften_between_words(
    x: &[f32],
    rate: u32,
    spans: &[(usize, usize)],
    reduction_db: f32,
) -> (Vec<f32>, usize) {
    let mut y = x.to_vec();
    if reduction_db <= 0.0 || spans.is_empty() {
        return (y, 0);
    }
    let target = 10f32.powf(-reduction_db / 20.0);
    let min_gap = (0.12 * rate as f32) as usize;
    let ramp = (0.02 * rate as f32) as usize;
    let mut softened = 0;
    for (a, b) in gaps(spans, x.len()) {
        if b - a < min_gap {
            continue;
        }
        let r = ramp.min((b - a) / 2);
        for (k, v) in y[a..b].iter_mut().enumerate() {
            let from_left = if a == 0 { usize::MAX } else { k };
            let from_right = if b == x.len() {
                usize::MAX
            } else {
                b - a - 1 - k
            };
            let d = from_left.min(from_right);
            let g = if d >= r {
                target
            } else {
                1.0 - (1.0 - target) * (d + 1) as f32 / (r + 1) as f32
            };
            *v *= g;
        }
        softened += 1;
    }
    (y, softened)
}

/// Where to insert silence inside the gap `[a, b)`: the quietest 10 ms frame
/// of its first 60%. An in-breath almost always sits just before the next
/// phrase, so this lands the added silence BEFORE the breath rather than
/// splitting it (2026-09-27). `None` if the gap is too short to hold the
/// 5 ms fades.
pub fn insertion_point(x: &[f32], rate: u32, a: usize, b: usize) -> Option<usize> {
    let frame = ((FRAME_S * rate as f32) as usize).max(1);
    let fade = (0.005 * rate as f32) as usize;
    if b <= a || b - a < 2 * fade + frame {
        return None;
    }
    let lo = a + fade;
    let hi = (a + (b - a) * 6 / 10).max(lo + frame).min(b - fade);
    let mut best = (f32::INFINITY, None);
    let mut k = lo;
    while k + frame <= hi {
        let e: f32 = x[k..k + frame].iter().map(|v| v * v).sum();
        if e < best.0 {
            best = (e, Some(k + frame / 2));
        }
        k += frame;
    }
    best.1.or(Some((lo + hi) / 2))
}

/// Insert `extra` samples of silence at each `at`, with 5 ms fades into and
/// out of it. Callers place every `at` inside a gap, so the fades touch only
/// non-word audio. Returns the audio and, for each input sample index, its
/// new index (so later stages can map word spans).
pub fn insert_silence(x: &[f32], rate: u32, inserts: &[(usize, usize)]) -> Vec<f32> {
    let mut ins: Vec<(usize, usize)> = inserts
        .iter()
        .copied()
        .filter(|&(at, n)| n > 0 && at <= x.len())
        .collect();
    ins.sort_unstable();
    let fade = (0.005 * rate as f32) as usize;
    let mut out = Vec::with_capacity(x.len() + ins.iter().map(|i| i.1).sum::<usize>());
    let mut last = 0;
    for (at, extra) in ins {
        if at < last {
            continue;
        }
        out.extend_from_slice(&x[last..at]);
        let n = out.len();
        let f = fade.min(at - last);
        for k in 0..f {
            out[n - f + k] *= 1.0 - (k + 1) as f32 / (f + 1) as f32;
        }
        out.extend(std::iter::repeat(0.0).take(extra));
        let resume = out.len();
        let g = fade.min(x.len() - at);
        out.extend_from_slice(&x[at..at + g]);
        for k in 0..g {
            out[resume + k] *= (k + 1) as f32 / (g + 1) as f32;
        }
        last = at + g;
    }
    out.extend_from_slice(&x[last..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 24_000;

    fn tone(secs: f32, hz: f32, amp: f32) -> Vec<f32> {
        (0..(secs * SR as f32) as usize)
            .map(|i| amp * (2.0 * std::f32::consts::PI * hz * i as f32 / SR as f32).sin())
            .collect()
    }

    fn hiss(secs: f32, amp: f32) -> Vec<f32> {
        (0..(secs * SR as f32) as usize)
            .map(|i| amp * (((i as u64).wrapping_mul(2654435761) % 2001) as f32 / 1000.0 - 1.0))
            .collect()
    }

    fn silence(secs: f32) -> Vec<f32> {
        vec![0.0; (secs * SR as f32) as usize]
    }

    fn word(start: f32, end: f32) -> Word {
        Word {
            text: "w".into(),
            start,
            end,
            probability: 1.0,
        }
    }

    fn secs(x: &[f32]) -> f32 {
        x.len() as f32 / SR as f32
    }

    /// "…chest. [pause + breath] Speak…" where the recognizer's end time sits
    /// at the START of the final "st" and its start time AFTER the "s" onset —
    /// the real misalignments measured on 2026-09-28.
    struct TestTake {
        audio: Vec<f32>,
        words: Vec<Word>,
        st: (usize, usize),
        breath: (usize, usize),
        onset: (usize, usize),
    }

    fn take() -> TestTake {
        let mut x = tone(0.6, 180.0, 0.5);
        let vowel_end = secs(&x);
        let st0 = x.len();
        x.extend(hiss(0.06, 0.08)); // "s"
        x.extend(silence(0.015)); // stop closure
        x.extend(hiss(0.04, 0.06)); // "t" release
        let st = (st0, x.len());
        x.extend(silence(0.25));
        let b0 = x.len();
        x.extend(hiss(0.3, 0.05)); // in-breath
        let breath = (b0, x.len());
        x.extend(silence(0.06));
        let s0 = x.len();
        x.extend(hiss(0.1, 0.07)); // word-initial "s"
        let onset = (s0, x.len());
        let next_vowel = secs(&x);
        x.extend(tone(0.6, 180.0, 0.5));
        let words = vec![word(0.0, vowel_end), word(next_vowel - 0.02, secs(&x))];
        TestTake {
            audio: x,
            words,
            st,
            breath,
            onset,
        }
    }

    #[test]
    fn word_edges_the_recognizer_missed_are_protected_and_untouched() {
        let TestTake {
            audio: x,
            words,
            st,
            onset,
            ..
        } = take();
        let spans = protected_spans(&x, SR, &words);
        let inside = |(a, b): (usize, usize)| spans.iter().any(|&(s, e)| s <= a && b <= e);
        assert!(
            inside(st),
            "word-final st must be protected: {spans:?} vs {st:?}"
        );
        assert!(
            inside(onset),
            "word-initial s must be protected: {spans:?} vs {onset:?}"
        );
        let (y, n) = soften_between_words(&x, SR, &spans, 15.0);
        assert_eq!(n, 1);
        for &(a, b) in &spans {
            assert_eq!(&x[a..b], &y[a..b], "a protected sample changed");
        }
    }

    #[test]
    fn the_breath_between_words_is_softened() {
        let TestTake {
            audio: x,
            words,
            breath,
            ..
        } = take();
        let spans = protected_spans(&x, SR, &words);
        let (y, _) = soften_between_words(&x, SR, &spans, 15.0);
        let e = |s: &[f32]| s.iter().map(|v| v * v).sum::<f32>();
        let (a, b) = (breath.0 + SR as usize / 25, breath.1 - SR as usize / 25);
        let db = 10.0 * (e(&y[a..b]) / e(&x[a..b])).log10();
        assert!(db < -13.0, "breath lowered by only {db:.1} dB");
    }

    #[test]
    fn a_short_gap_between_words_is_left_alone() {
        let mut x = tone(0.5, 180.0, 0.5);
        x.extend(silence(0.08));
        let second = secs(&x);
        x.extend(tone(0.5, 180.0, 0.5));
        let spans = protected_spans(&x, SR, &[word(0.0, 0.5), word(second, secs(&x))]);
        assert_eq!(soften_between_words(&x, SR, &spans, 15.0), (x.clone(), 0));
    }

    #[test]
    fn silence_is_inserted_in_the_gap_before_the_breath_and_words_survive() {
        let TestTake {
            audio: x,
            words,
            breath,
            ..
        } = take();
        let spans = protected_spans(&x, SR, &words);
        let g = gaps(&spans, x.len());
        let (a, b) = g
            .iter()
            .copied()
            .find(|&(a, b)| a > 0 && b < x.len())
            .expect("a gap between the words");
        let at = insertion_point(&x, SR, a, b).unwrap();
        assert!(
            at >= a && at < breath.0,
            "inserted at {at}, gap {a}..{b}, breath starts {}",
            breath.0
        );
        let extra = SR as usize / 2;
        let y = insert_silence(&x, SR, &[(at, extra)]);
        assert_eq!(y.len(), x.len() + extra);
        let (w0, w1) = (spans[0], spans[1]);
        assert_eq!(&x[w0.0..w0.1], &y[w0.0..w0.1], "first word changed");
        assert_eq!(
            &x[w1.0..w1.1],
            &y[w1.0 + extra..w1.1 + extra],
            "second word changed"
        );
        // The breath appears exactly once: silence is inserted, never looped audio.
        let br = &x[breath.0..breath.1];
        assert_eq!(y.windows(br.len()).filter(|w| *w == br).count(), 1);
    }

    #[test]
    fn a_gap_too_short_for_the_fades_gets_no_insertion() {
        let x = tone(1.0, 180.0, 0.5);
        assert_eq!(insertion_point(&x, SR, 1000, 1100), None);
    }
}
