//! The Rust log-mel must reproduce the official Whisper feature extractor.
//! Golden values were produced by `transformers.WhisperFeatureExtractor` on a
//! deterministic 2.5 s signal (2026-09-28); no model files are needed.

#[test]
fn log_mel_matches_the_reference_extractor() {
    let golden: serde_json::Value =
        serde_json::from_str(include_str!("data/whisper_mel_golden.json")).unwrap();
    let sr = 16_000f64;
    let n = (2.5 * sr) as usize;
    let x: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f64 / sr;
            (0.3 * (2.0 * std::f64::consts::PI * 220.0 * t).sin()
                + 0.1 * (2.0 * std::f64::consts::PI * 1375.0 * t).sin() * (-t).exp())
                as f32
        })
        .collect();
    let mel = xrt_audio::whisper::mel::log_mel(&x);
    let frames = xrt_audio::whisper::mel::N_FRAMES;
    let mut worst = 0.0f64;
    for (k, col) in golden["values"].as_object().unwrap() {
        let t: usize = k.parse().unwrap();
        for (m, want) in col.as_array().unwrap().iter().enumerate() {
            let got = mel[m * frames + t] as f64;
            worst = worst.max((got - want.as_f64().unwrap()).abs());
        }
    }
    assert!(
        worst < 2e-3,
        "log-mel differs from the reference by up to {worst}"
    );
}
