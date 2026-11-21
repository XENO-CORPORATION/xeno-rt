//! Minimal PCM utilities the TTS pipeline needs: WAV read/write, resampling,
//! silence trimming and stitching. Full audio decode/encode belongs to
//! `xeno-lib`; callers with MP3/MP4 references convert there first.

use crate::AudioError;

pub const SAMPLE_RATE: u32 = 24_000;

/// Decode a WAV (PCM 16/24/32-bit int or 32-bit float, any channel count) to
/// mono f32 at its native rate.
pub fn read_wav(bytes: &[u8]) -> Result<(Vec<f32>, u32), AudioError> {
    let bad = |m: &str| AudioError::InvalidReference(m.to_string());
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(bad(
            "not a RIFF/WAVE file (convert MP3/MP4 references to WAV first)",
        ));
    }
    let (mut fmt, mut data) = (None, None);
    let mut p = 12;
    while p + 8 <= bytes.len() {
        let id = &bytes[p..p + 4];
        let len = u32::from_le_bytes(bytes[p + 4..p + 8].try_into().unwrap()) as usize;
        if len > bytes.len() - p - 8 {
            return Err(bad("truncated WAV chunk"));
        }
        let body = &bytes[p + 8..p + 8 + len];
        match id {
            b"fmt " => fmt = Some(body),
            b"data" => data = Some(body),
            _ => {}
        }
        p += 8 + len + (len & 1);
    }
    let fmt = fmt.ok_or_else(|| bad("missing fmt chunk"))?;
    let data = data.ok_or_else(|| bad("missing data chunk"))?;
    if fmt.len() < 16 {
        return Err(bad("short fmt chunk"));
    }
    let mut format = u16::from_le_bytes([fmt[0], fmt[1]]);
    let channels = u16::from_le_bytes([fmt[2], fmt[3]]) as usize;
    let rate = u32::from_le_bytes(fmt[4..8].try_into().unwrap());
    let bits = u16::from_le_bytes([fmt[14], fmt[15]]);
    if format == 0xFFFE && fmt.len() >= 26 {
        format = u16::from_le_bytes([fmt[24], fmt[25]]); // WAVE_FORMAT_EXTENSIBLE subformat
    }
    if !(1..=32).contains(&channels) || !(8_000..=192_000).contains(&rate) {
        return Err(bad(
            "WAV must have 1..=32 channels and an 8000..=192000 Hz sample rate",
        ));
    }
    if !matches!((format, bits), (1, 16 | 24 | 32) | (3, 32)) {
        return Err(bad("unsupported WAV encoding"));
    }
    let frame = channels * (bits as usize / 8);
    if frame == 0 {
        return Err(bad("unsupported bit depth"));
    }
    if data.is_empty() || data.len() % frame != 0 {
        return Err(bad("empty or incomplete WAV sample frame"));
    }
    let decode = |s: &[u8]| -> Result<f32, AudioError> {
        Ok(match (format, bits) {
            (1, 16) => i16::from_le_bytes([s[0], s[1]]) as f32 / 32768.0,
            (1, 24) => (i32::from_le_bytes([0, s[0], s[1], s[2]]) >> 8) as f32 / 8_388_608.0,
            (1, 32) => i32::from_le_bytes(s.try_into().unwrap()) as f32 / 2_147_483_648.0,
            (3, 32) => f32::from_le_bytes(s.try_into().unwrap()),
            _ => {
                return Err(bad(&format!(
                    "unsupported WAV encoding (format {format}, {bits}-bit)"
                )))
            }
        })
    };
    let bps = bits as usize / 8;
    let mut mono = Vec::with_capacity(data.len() / frame);
    for f in data.chunks_exact(frame) {
        let mut acc = 0.0;
        for c in 0..channels {
            let sample = decode(&f[c * bps..(c + 1) * bps])?;
            if !sample.is_finite() || sample.abs() > 1_000.0 {
                return Err(bad(
                    "WAV samples must be finite and have a bounded amplitude",
                ));
            }
            acc += sample;
        }
        mono.push(acc / channels as f32);
    }
    // Float WAV permits headroom above full scale (including generated
    // intermediate takes). Normalize it instead of rejecting valid audio.
    let peak = mono.iter().fold(0.0f32, |peak, s| peak.max(s.abs()));
    if peak > 1.0 {
        mono.iter_mut().for_each(|s| *s /= peak);
    }
    Ok((mono, rate))
}

/// 32-bit float mono WAV.
pub fn write_wav(samples: &[f32], rate: u32) -> Vec<u8> {
    let data_len = (samples.len() * 4) as u32;
    let mut b = Vec::with_capacity(44 + data_len as usize);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data_len).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&3u16.to_le_bytes()); // IEEE float
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&rate.to_le_bytes());
    b.extend_from_slice(&(rate * 4).to_le_bytes());
    b.extend_from_slice(&4u16.to_le_bytes());
    b.extend_from_slice(&32u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data_len.to_le_bytes());
    for s in samples {
        b.extend_from_slice(&s.to_le_bytes());
    }
    b
}

/// Windowed-sinc resampler (Blackman window, 32 zero crossings). Good enough
/// for conditioning inputs; not a mastering-grade SRC.
pub fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    let ratio = to as f64 / from as f64;
    let cutoff = ratio.min(1.0) * 0.97;
    let half = 32.0 / cutoff;
    let n_out = (x.len() as f64 * ratio).round() as usize;
    let mut out = Vec::with_capacity(n_out);
    for i in 0..n_out {
        let center = i as f64 / ratio;
        let lo = (center - half).ceil().max(0.0) as usize;
        let hi = ((center + half).floor() as usize).min(x.len() - 1);
        let (mut acc, mut wsum) = (0.0f64, 0.0f64);
        for (j, &xj) in x.iter().enumerate().take(hi + 1).skip(lo) {
            let t = j as f64 - center;
            let sinc = if t.abs() < 1e-9 {
                1.0
            } else {
                let a = std::f64::consts::PI * t * cutoff;
                a.sin() / a
            };
            let w = 0.42
                + 0.5 * (std::f64::consts::PI * t / half).cos()
                + 0.08 * (2.0 * std::f64::consts::PI * t / half).cos();
            let k = sinc * w;
            acc += xj as f64 * k;
            wsum += k;
        }
        out.push(if wsum.abs() > 1e-12 {
            (acc / wsum) as f32
        } else {
            0.0
        });
    }
    out
}

/// Trim leading/trailing silence without clipping speech.
///
/// Speech starts and ends softly (breathy vowel onsets, decaying consonants),
/// well below a fixed sample threshold, so the edge is found on 10 ms frame
/// energy relative to the take's own speech level, and generous margins are
/// kept: 80 ms before the first speech frame, 250 ms after the last. An
/// earlier version cut 20 ms / 50 ms around the first/last sample above
/// 0.015 and audibly clipped soft first and last words.
pub fn trim_silence(x: &[f32], rate: u32, threshold: f32) -> Vec<f32> {
    let frame = ((0.01 * rate as f32) as usize).max(1);
    if x.len() < frame {
        return if x.iter().any(|s| s.abs() > threshold) {
            x.to_vec()
        } else {
            Vec::new()
        };
    }
    let rms: Vec<f32> = x
        .chunks(frame)
        .map(|f| (f.iter().map(|s| s * s).sum::<f32>() / f.len() as f32).sqrt())
        .collect();
    // Speech level from frames that carry sound: a take can be mostly
    // silence, and a percentile over ALL frames would then read as zero.
    let mut sorted: Vec<f32> = rms
        .iter()
        .copied()
        .filter(|&r| r > threshold * 0.5)
        .collect();
    if sorted.is_empty() {
        return Vec::new();
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let speech_level = sorted[((sorted.len() - 1) as f32 * 0.95) as usize];
    // 3% of speech level (~-30 dB), never below the caller's absolute floor.
    let floor = (0.03 * speech_level).max(threshold * 0.5);
    let first = rms.iter().position(|&r| r > floor);
    let last = rms.iter().rposition(|&r| r > floor);
    match (first, last) {
        (Some(a), Some(b)) if speech_level > threshold => {
            let a = (a * frame).saturating_sub((0.08 * rate as f32) as usize);
            let b = ((b + 1) * frame + (0.25 * rate as f32) as usize).min(x.len());
            x[a..b].to_vec()
        }
        _ => Vec::new(),
    }
}

/// Join pieces with the given silence between them; normalise the peak to
/// `peak`. Pieces must already be edge-faded (`prosody::trim_to_speech`
/// does it).
///
/// This used to apply its own 30 ms fade to every piece edge, on the
/// assumption that edges were silence. Since `trim_to_speech` cuts a take at
/// the first sound of its first word, that fade fell on word onsets ("s…")
/// and attenuated them; `stitch_leaves_the_pieces_untouched` pins it.
pub fn stitch(pieces: &[(Vec<f32>, f32)], rate: u32, peak: f32) -> Vec<f32> {
    let mut out = Vec::new();
    for (i, (p, gap_after)) in pieces.iter().enumerate() {
        out.extend_from_slice(p);
        if i + 1 < pieces.len() {
            out.extend(std::iter::repeat(0.0).take((gap_after * rate as f32) as usize));
        }
    }
    let m = out.iter().fold(0.0f32, |a, s| a.max(s.abs()));
    if m > 1e-6 {
        let g = peak / m;
        out.iter_mut().for_each(|s| *s *= g);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stitch_leaves_the_pieces_untouched() {
        let a: Vec<f32> = (0..2_400).map(|i| (i as f32 * 0.3).sin() * 0.4).collect();
        let b: Vec<f32> = (0..2_400).map(|i| (i as f32 * 0.2).cos() * 0.5).collect();
        let y = stitch(&[(a.clone(), 0.1), (b.clone(), 0.0)], 24_000, 0.5);
        let g = 0.5 / 0.5; // b's peak is 0.5
        assert_eq!(y.len(), a.len() + 2_400 + b.len());
        for (k, v) in a.iter().enumerate() {
            assert!((y[k] - v * g).abs() < 1e-6, "first piece changed at {k}");
        }
        for (k, v) in b.iter().enumerate() {
            assert!(
                (y[a.len() + 2_400 + k] - v * g).abs() < 1e-6,
                "second piece changed at {k}"
            );
        }
    }

    #[test]
    fn malformed_wav_cannot_reach_resampling() {
        for value in [f32::NAN, f32::INFINITY, f32::MAX] {
            assert!(read_wav(&write_wav(&[value], 24_000)).is_err());
        }
        for rate in [0, 1, 7_999, 192_001, u32::MAX / 4] {
            assert!(read_wav(&write_wav(&[0.1], rate)).is_err());
        }
        assert_eq!(
            read_wav(&write_wav(&[2.0, -1.0], 24_000)).unwrap().0,
            vec![1.0, -0.5]
        );
        let mut truncated = write_wav(&[0.1; 10], 24_000);
        truncated.pop();
        assert!(read_wav(&truncated).is_err());
        assert!(read_wav(&write_wav(&[], 24_000)).is_err());
    }

    #[test]
    fn wav_roundtrip() {
        let x: Vec<f32> = (0..480).map(|i| (i as f32 * 0.05).sin() * 0.5).collect();
        let (y, r) = read_wav(&write_wav(&x, 24_000)).unwrap();
        assert_eq!(r, 24_000);
        assert_eq!(x, y);
    }

    #[test]
    fn resample_preserves_a_tone() {
        let x: Vec<f32> = (0..48_000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48_000.0).sin())
            .collect();
        let y = resample(&x, 48_000, 24_000);
        assert_eq!(y.len(), 24_000);
        let crossings = y
            .windows(2)
            .filter(|w| (w[0] < 0.0) != (w[1] < 0.0))
            .count();
        assert!(
            (870..=890).contains(&crossings),
            "440 Hz ≈ 880 crossings/s, got {crossings}"
        );
    }

    #[test]
    fn trim_removes_silence() {
        let mut x = vec![0.0; 24_000];
        x.extend(vec![0.5; 2_400]);
        x.extend(vec![0.0; 24_000]);
        let y = trim_silence(&x, 24_000, 0.015);
        // 2400 samples of signal + 80 ms before + 250 ms after, on 10 ms frames.
        assert!((10_000..=11_000).contains(&y.len()), "{}", y.len());
        assert!(y.len() < x.len() / 4, "most of the silence must go");
    }

    #[test]
    fn trim_keeps_a_soft_final_word() {
        // A loud phrase, then a soft trailing word far below 0.015 per sample
        // but clearly speech relative to the take: it must not be cut.
        let mut x = vec![0.0; 2_400];
        x.extend((0..24_000).map(|i| 0.6 * ((i as f32) * 0.1).sin()));
        x.extend((0..4_800).map(|i| 0.012 * ((i as f32) * 0.1).sin()));
        x.extend(vec![0.0; 24_000]);
        let y = trim_silence(&x, 24_000, 0.015);
        assert!(
            y.len() >= 2_400 + 24_000 + 4_800 - 2_400,
            "soft ending was clipped: {}",
            y.len()
        );
    }
}
