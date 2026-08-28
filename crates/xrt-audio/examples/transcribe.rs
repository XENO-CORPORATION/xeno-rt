//! Transcribes a 16-bit PCM mono WAV with a local Whisper ONNX model.
//!
//!     ORT_DYLIB_PATH=<path to onnxruntime.dll/.so> \
//!     cargo run -p xrt-audio --example transcribe -- <model-dir> <in.wav>
//!
//! The model directory holds `encoder.onnx`, `decoder.onnx`, `config.json`,
//! `vocab.json` and `added_tokens.json`.
//!
//! This is the rung-3 check for the crate: it exercises the real sessions, the
//! real cache loop and the real tokenizer against a file, which is the only
//! thing that can tell you the adapter works. Unit tests cover the frontend;
//! they cannot tell you whether the decoder's cache is wired correctly, because
//! a mis-wired cache still produces fluent-looking text.

use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: transcribe <model-dir> <in.wav>");
        std::process::exit(2);
    }
    let dir = PathBuf::from(&args[1]);
    let wav = std::fs::read(&args[2]).expect("read wav");
    let (samples, rate, channels) = parse_wav(&wav);
    eprintln!(
        "audio: {:.2}s, {} Hz, {} ch",
        samples.len() as f32 / rate as f32 / channels as f32,
        rate,
        channels
    );

    let mono = xrt_audio::to_mono(&samples, channels as usize);

    let t0 = std::time::Instant::now();
    let mut model = match xrt_audio::whisper::WhisperModel::load(&dir) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("load failed: {e}");
            std::process::exit(1);
        }
    };
    eprintln!("model loaded in {:?}", t0.elapsed());

    let t1 = std::time::Instant::now();
    match model.transcribe(&mono, rate) {
        Ok(t) => {
            eprintln!("transcribed in {:?}", t1.elapsed());
            for s in &t.segments {
                println!("[{:>7.2} -> {:>7.2}]  {}", s.start, s.end, s.text);
            }
            println!("\nTEXT: {}", t.text);
        }
        Err(e) => {
            eprintln!("transcribe failed: {e}");
            std::process::exit(1);
        }
    }
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
        pos += 8 + size + (size & 1);
    }
    let data = data.expect("no data chunk");
    assert_eq!(bits, 16, "only 16-bit PCM is supported by this example");
    (
        data.chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
            .collect(),
        rate,
        channels,
    )
}
