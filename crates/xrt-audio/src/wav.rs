//! Minimal RIFF/WAVE reader for 16-bit PCM.
//!
//! ⚠️ THIS IS NOT A DECODER, and the distinction is the crate's architecture.
//! `RUNTIME_DOMAINS.md` gives `xeno-lib` "video and audio decode/encode" and
//! keeps model execution here; a WAV header is not a codec, it is a length
//! prefix in front of raw samples. Reading it costs forty lines and no
//! dependency, which is what makes accepting PCM a reasonable server contract.
//!
//! Anything COMPRESSED — mp3, aac, opus, flac — is genuinely decoding and does
//! not belong in this crate. A caller either decodes it (xeno-motion already
//! does, via WebCodecs) or asks `xeno-lib`. That boundary is why
//! [`WavError::UnsupportedFormat`] names the codec instead of trying to handle
//! it: a wrong answer here would quietly make xeno-rt a media library.
//!
//! It lives in the crate rather than in each example because it had been
//! copy-pasted into three of them, which is how two copies drift and the third
//! becomes the one nobody fixes.

/// Failure modes of the container read, kept separate from inference errors so
/// a caller can tell "you sent me the wrong thing" from "the model broke".
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum WavError {
    #[error("not a RIFF/WAVE file")]
    NotRiff,
    #[error("no data chunk")]
    NoData,
    #[error("no fmt chunk")]
    NoFmt,
    /// The payload is a real audio file this crate deliberately does not decode.
    #[error("{0} is not supported here - decode to 16-bit PCM WAV first (xeno-lib or the app owns decoding)")]
    UnsupportedFormat(&'static str),
    #[error("unsupported WAV encoding: {bits}-bit, format tag {tag}")]
    UnsupportedEncoding { bits: u16, tag: u16 },
}

/// Sniffs common compressed containers so the error names the format rather
/// than saying "not a RIFF file", which is true and useless.
fn sniff(b: &[u8]) -> Option<&'static str> {
    if b.len() < 12 {
        return None;
    }
    if &b[0..4] == b"fLaC" {
        return Some("FLAC");
    }
    if &b[0..4] == b"OggS" {
        return Some("Ogg (Vorbis/Opus)");
    }
    if &b[0..3] == b"ID3" || (b[0] == 0xFF && (b[1] & 0xE0) == 0xE0) {
        return Some("MP3");
    }
    if b.len() > 12 && &b[4..8] == b"ftyp" {
        return Some("MP4/M4A");
    }
    None
}

/// Reads 16-bit PCM WAV, returning interleaved samples in `[-1, 1]`, the sample
/// rate, and the channel count.
pub fn read_pcm16(b: &[u8]) -> Result<(Vec<f32>, u32, u16), WavError> {
    if let Some(fmt) = sniff(b) {
        return Err(WavError::UnsupportedFormat(fmt));
    }
    if b.len() < 12 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return Err(WavError::NotRiff);
    }

    let mut pos = 12usize;
    let (mut rate, mut channels, mut bits, mut tag) = (0u32, 0u16, 0u16, 0u16);
    let mut data: Option<&[u8]> = None;
    let mut saw_fmt = false;

    while pos + 8 <= b.len() {
        let id = &b[pos..pos + 4];
        let size = u32::from_le_bytes([b[pos + 4], b[pos + 5], b[pos + 6], b[pos + 7]]) as usize;
        let end = (pos + 8 + size).min(b.len());
        let body = &b[pos + 8..end];
        if id == b"fmt " && body.len() >= 16 {
            tag = u16::from_le_bytes([body[0], body[1]]);
            channels = u16::from_le_bytes([body[2], body[3]]);
            rate = u32::from_le_bytes([body[4], body[5], body[6], body[7]]);
            bits = u16::from_le_bytes([body[14], body[15]]);
            saw_fmt = true;
        } else if id == b"data" {
            data = Some(body);
        }
        // RIFF chunks are word-aligned; an odd size carries a pad byte.
        pos += 8 + size + (size & 1);
    }

    if !saw_fmt {
        return Err(WavError::NoFmt);
    }
    // 1 = PCM, 0xFFFE = WAVE_FORMAT_EXTENSIBLE (still PCM at 16 bits).
    if bits != 16 || (tag != 1 && tag != 0xFFFE) {
        return Err(WavError::UnsupportedEncoding { bits, tag });
    }
    let data = data.ok_or(WavError::NoData)?;

    Ok((
        data.chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
            .collect(),
        rate,
        channels.max(1),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav(rate: u32, channels: u16, samples: &[i16]) -> Vec<u8> {
        let data: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let mut v = Vec::new();
        v.extend_from_slice(b"RIFF");
        v.extend_from_slice(&(36u32 + data.len() as u32).to_le_bytes());
        v.extend_from_slice(b"WAVEfmt ");
        v.extend_from_slice(&16u32.to_le_bytes());
        v.extend_from_slice(&1u16.to_le_bytes()); // PCM
        v.extend_from_slice(&channels.to_le_bytes());
        v.extend_from_slice(&rate.to_le_bytes());
        v.extend_from_slice(&(rate * channels as u32 * 2).to_le_bytes());
        v.extend_from_slice(&(channels * 2).to_le_bytes());
        v.extend_from_slice(&16u16.to_le_bytes());
        v.extend_from_slice(b"data");
        v.extend_from_slice(&(data.len() as u32).to_le_bytes());
        v.extend_from_slice(&data);
        v
    }

    #[test]
    fn reads_mono_pcm16() {
        let (s, rate, ch) = read_pcm16(&wav(16_000, 1, &[0, 16384, -16384])).unwrap();
        assert_eq!((rate, ch), (16_000, 1));
        assert_eq!(s.len(), 3);
        assert!((s[1] - 0.5).abs() < 1e-4, "{}", s[1]);
        assert!((s[2] + 0.5).abs() < 1e-4, "{}", s[2]);
    }

    #[test]
    fn reports_channel_count_for_interleaved_stereo() {
        let (s, _, ch) = read_pcm16(&wav(48_000, 2, &[0, 0, 0, 0])).unwrap();
        assert_eq!(ch, 2);
        assert_eq!(s.len(), 4, "samples are interleaved, not per-channel");
    }

    /// A compressed file must be named, not merely rejected. "not a RIFF file"
    /// sends someone looking for a corrupt upload; "FLAC is not supported here"
    /// tells them exactly what to do.
    #[test]
    fn compressed_formats_are_named() {
        let mut flac = b"fLaC".to_vec();
        flac.extend_from_slice(&[0u8; 32]);
        assert_eq!(read_pcm16(&flac), Err(WavError::UnsupportedFormat("FLAC")));

        let mut mp4 = vec![0u8; 4];
        mp4.extend_from_slice(b"ftyp");
        mp4.extend_from_slice(&[0u8; 16]);
        assert_eq!(read_pcm16(&mp4), Err(WavError::UnsupportedFormat("MP4/M4A")));

        let mut mp3 = b"ID3".to_vec();
        mp3.extend_from_slice(&[0u8; 32]);
        assert_eq!(read_pcm16(&mp3), Err(WavError::UnsupportedFormat("MP3")));
    }

    #[test]
    fn rejects_non_pcm16_rather_than_misreading_it() {
        // 32-bit float WAV: same container, different sample encoding. Reading
        // it as i16 would produce loud noise rather than an error.
        let mut v = wav(16_000, 1, &[0, 0]);
        v[34] = 32; // bits per sample -> 32
        assert!(matches!(read_pcm16(&v), Err(WavError::UnsupportedEncoding { bits: 32, .. })));
    }

    #[test]
    fn garbage_is_rejected() {
        assert_eq!(read_pcm16(b"not audio at all"), Err(WavError::NotRiff));
        assert_eq!(read_pcm16(&[]), Err(WavError::NotRiff));
    }
}
