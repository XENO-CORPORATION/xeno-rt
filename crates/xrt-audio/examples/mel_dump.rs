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
    let (samples, sample_rate, channels) = xrt_audio::wav::read_pcm16(&bytes).expect("read wav");
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


