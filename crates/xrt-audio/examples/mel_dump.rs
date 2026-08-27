//! Dumps this crate's Whisper log-mel for a 16-bit PCM mono WAV.
//!
//! Exists so the frontend can be verified against the REAL trained weights
//! rather than against another implementation of itself: the output of this
//! example is fed straight into `encoder_model.onnx`, and if the mel is wrong
//! the transcript is wrong. That is a correctness check no unit test can make,
//! because only the model knows what its input is supposed to look like.
//!
//!     cargo run -p xrt-audio --example mel_dump -- in.wav out.f32
//!
//! Writes `n_mels * 3000` little-endian f32. The WAV parsing here is
//! deliberately minimal and lives in an example, not the crate: decoding
//! container formats is `xeno-lib`'s responsibility, never the runtime's.

use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: mel_dump <in.wav> <out.f32> [n_mels=80]");
        std::process::exit(2);
    }
    let n_mels: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(80);

    let bytes = std::fs::read(&args[1]).expect("read wav");
    let (samples, sample_rate, channels) = parse_wav(&bytes);
    eprintln!(
        "wav: {} frames, {} Hz, {} ch",
        samples.len() / channels.max(1) as usize,
        sample_rate,
        channels
    );

    let mono = xrt_audio::to_mono(&samples, channels as usize);
    let at16k = xrt_audio::resample_linear(&mono, sample_rate, 16_000);
    let mel = xrt_audio::mel::log_mel_spectrogram(&xrt_audio::mel::pad_or_trim(&at16k), n_mels);

    eprintln!("mel: {} x {}", mel.len(), mel[0].len());
    let mut out = std::io::BufWriter::new(std::fs::File::create(&args[2]).expect("create out"));
    for row in &mel {
        for &v in row {
            out.write_all(&v.to_le_bytes()).expect("write");
        }
    }
    out.flush().expect("flush");
}

/// Minimal RIFF/WAVE reader: 16-bit PCM only. Returns interleaved samples.
fn parse_wav(b: &[u8]) -> (Vec<f32>, u32, u16) {
    assert!(b.len() > 44 && &b[0..4] == b"RIFF" && &b[8..12] == b"WAVE", "not a WAV");
    let mut pos = 12usize;
    let (mut rate, mut channels, mut bits) = (16_000u32, 1u16, 16u16);
    let mut data: Option<&[u8]> = None;

    while pos + 8 <= b.len() {
        let id = &b[pos..pos + 4];
        let size = u32::from_le_bytes([b[pos + 4], b[pos + 5], b[pos + 6], b[pos + 7]]) as usize;
        let body = &b[pos + 8..(pos + 8 + size).min(b.len())];
        if id == b"fmt " && body.len() >= 16 {
            channels = u16::from_le_bytes([body[2], body[3]]);
            rate = u32::from_le_bytes([body[4], body[5], body[6], body[7]]);
            bits = u16::from_le_bytes([body[14], body[15]]);
        } else if id == b"data" {
            data = Some(body);
        }
        pos += 8 + size + (size & 1); // chunks are word-aligned
    }

    let data = data.expect("no data chunk");
    assert_eq!(bits, 16, "only 16-bit PCM is supported by this example");
    let samples = data
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
        .collect();
    (samples, rate, channels)
}
