//! Rung-3 gate: the real sessions, the real KV cache, the real tokenizer.
//!
//! The unit tests cover the frontend and can prove it correct. They cannot tell
//! you the ADAPTER works, because a mis-wired KV cache still produces fluent
//! English — just the wrong English. Only running a known file and comparing to
//! a known transcript separates those.
//!
//! Requires the ONNX Runtime shared library and a local model:
//!
//!     XRT_AUDIO_TEST_MODEL=<model-dir> \
//!     XRT_AUDIO_TEST_WAV=<jfk.wav> \
//!     ORT_DYLIB_PATH=<onnxruntime.dll|.so> \
//!     cargo test -p xrt-audio --test whisper_e2e -- --nocapture
//!
//! ⚠️ SKIPPING IS NOT PASSING, and this file is written so that cannot blur.
//! With no `XRT_AUDIO_TEST_MODEL` the test prints that it did not run and
//! returns; with the variable SET, every failure is a hard failure — a missing
//! model, a missing dylib or a wrong transcript all fail loudly. The dangerous
//! shape is a test that quietly degrades to green when its fixture is absent,
//! which is how this ecosystem has repeatedly shipped unreachable code.

use std::path::PathBuf;

fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var_os(key).map(PathBuf::from).filter(|p| !p.as_os_str().is_empty())
}

/// Whisper is free to punctuate and capitalise differently between runs and
/// model sizes; the WORDS are the contract. Compare on lowercased alphanumerics.
fn words(s: &str) -> Vec<String> {
    s.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .map(str::to_string)
        .collect()
}



const EXPECTED: &str = "and so my fellow americans ask not what your country can do for you \
                        ask what you can do for your country";

#[test]
fn transcribes_the_reference_sample() {
    let Some(model_dir) = env_path("XRT_AUDIO_TEST_MODEL") else {
        eprintln!(
            "SKIPPED: set XRT_AUDIO_TEST_MODEL, XRT_AUDIO_TEST_WAV and ORT_DYLIB_PATH to run \
             the end-to-end adapter gate. This did NOT pass — it did not run."
        );
        return;
    };
    let wav_path = env_path("XRT_AUDIO_TEST_WAV")
        .expect("XRT_AUDIO_TEST_MODEL is set, so XRT_AUDIO_TEST_WAV must be too");

    // From here on nothing is allowed to degrade to green.
    let bytes = std::fs::read(&wav_path).expect("read the reference wav");
    let (samples, rate, channels) = xrt_audio::wav::read_pcm16(&bytes).expect("read wav");
    let mono = xrt_audio::to_mono(&samples, channels as usize);

    let mut model = xrt_audio::whisper::WhisperModel::load(&model_dir)
        .expect("load the Whisper model (is ORT_DYLIB_PATH set?)");
    let out = model.transcribe(&mono, rate).expect("transcribe");

    let got = words(&out.text);
    let want = words(EXPECTED);
    let matched = got.iter().zip(want.iter()).filter(|(a, b)| a == b).count();
    eprintln!("transcript: {:?}", out.text);
    eprintln!("word match: {matched}/{}", want.len());

    assert!(
        matched * 10 >= want.len() * 9,
        "expected >=90% word match against the known reference, got {matched}/{}: {:?}",
        want.len(),
        out.text
    );

    // A segment must describe the audio it came from. Reporting the padded
    // 30-second encoder window for an 11-second clip would put every downstream
    // subtitle in the wrong place.
    assert_eq!(out.segments.len(), 1, "11 s is one window");
    let seg = &out.segments[0];
    assert!(seg.start.abs() < 0.01, "segment starts at {}", seg.start);
    assert!(
        (seg.end - 11.0).abs() < 0.5,
        "segment ends at {} — the 30 s padding leaked into the reported duration",
        seg.end
    );
}

/// A missing model must produce a clear, actionable error and never a fabricated
/// transcript. This one needs no fixture, so it always runs.
#[test]
fn a_missing_model_is_an_error_not_an_empty_transcript() {
    // `WhisperModel` holds ORT sessions and cannot derive Debug, so match
    // rather than `expect_err`.
    let msg = match xrt_audio::whisper::WhisperModel::load(std::path::Path::new(
        "definitely-not-a-model-directory",
    )) {
        Ok(_) => panic!("loading a nonexistent model directory must fail"),
        Err(e) => e.to_string(),
    };
    assert!(
        msg.contains("not found") || msg.contains("config.json") || msg.contains("unavailable"),
        "error should name what is missing, got: {msg}"
    );
}
