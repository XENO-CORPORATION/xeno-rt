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
    // `--registry` exercises the path a real user takes: nothing on disk, the
    // model fetched and sha256-verified from updates.xenostudio.ai.
    let from_registry = args[1] == "--registry";
    let dir = PathBuf::from(&args[1]);
    let wav = std::fs::read(&args[2]).expect("read wav");
    let (samples, rate, channels) = xrt_audio::wav::read_pcm16(&wav).expect("read wav");
    eprintln!(
        "audio: {:.2}s, {} Hz, {} ch",
        samples.len() as f32 / rate as f32 / channels as f32,
        rate,
        channels
    );

    let mono = xrt_audio::to_mono(&samples, channels as usize);

    let t0 = std::time::Instant::now();
    let loaded = if from_registry {
        xrt_audio::whisper::WhisperModel::load_from_registry()
    } else {
        xrt_audio::whisper::WhisperModel::load(&dir)
    };
    let mut model = match loaded {
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


