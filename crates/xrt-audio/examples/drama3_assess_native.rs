//! Independent offline ASR assessment of retained WAVs. This is qualification,
//! not the native TTS graph, and does not establish perceptual or voice quality.
use serde::Deserialize;
use std::path::Path;
use xrt_audio::{whisper::WhisperModel, wordcheck};
#[derive(Deserialize)]
struct Corpus {
    seeds: Vec<u64>,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    id: String,
    spoken: String,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err(
            "usage: drama3_assess_native <corpus.json> <generated-directory> <new-report.json>"
                .into(),
        );
    }
    if std::fs::metadata(&args[0])?.len() > 1024 * 1024 {
        return Err("corpus exceeds 1 MiB".into());
    }
    let corpus: Corpus = serde_json::from_slice(&std::fs::read(&args[0])?)?;
    if corpus.cases.is_empty()
        || corpus.cases.len() > 32
        || corpus.seeds.is_empty()
        || corpus.seeds.len() > 8
    {
        return Err("corpus dimensions exceed qualification bounds".into());
    }
    let checker_digest = WhisperModel::offline_digest()?;
    let mut model = WhisperModel::load_offline(&checker_digest)?;
    let directory = Path::new(&args[1]);
    let mut checks = Vec::new();
    for case in corpus.cases {
        if case.id.is_empty()
            || case.id.len() > 64
            || !case
                .id
                .bytes()
                .all(|v| v.is_ascii_alphanumeric() || v == b'-')
        {
            return Err("invalid corpus id".into());
        }
        for seed in &corpus.seeds {
            let id = format!("{}-{seed}", case.id);
            let wav = directory.join(format!("{id}.wav"));
            if !wav.is_file() {
                let record = directory.join(format!("{id}.json"));
                let reason = if record.is_file() {
                    serde_json::from_slice::<serde_json::Value>(&std::fs::read(record)?)?
                } else {
                    serde_json::json!({"error":"missing_generation_receipt"})
                };
                checks
                    .push(serde_json::json!({"id":id,"passed":false,"generation_failure":reason}));
                continue;
            }
            if std::fs::metadata(&wav)?.len() > 16 * 1024 * 1024 {
                return Err("WAV exceeds 16 MiB".into());
            }
            let (samples, rate) = xrt_audio::audio::read_wav(&std::fs::read(&wav)?)?;
            if samples.is_empty() || rate != 44100 || samples.iter().any(|v| !v.is_finite()) {
                return Err("invalid native waveform".into());
            }
            let transcript = model.transcribe(&samples, rate)?;
            let check = wordcheck::check_text_exact(&case.spoken, &transcript.text);
            let peak = samples.iter().fold(0.0f32, |peak, v| peak.max(v.abs()));
            let rms = (samples.iter().map(|v| (*v as f64).powi(2)).sum::<f64>()
                / samples.len() as f64)
                .sqrt();
            let passed = check.error_rate == 0.0 && check.script_tokens > 0 && peak > 0.0;
            let record = serde_json::json!({"id":id,"passed":passed,"expected":case.spoken,"heard":transcript.text,"word_check":check,
                "duration_seconds":samples.len() as f64/rate as f64,"peak":peak,"rms":rms,
                "clipped_sample_fraction":samples.iter().filter(|v|v.abs()>=0.999).count() as f64/samples.len() as f64});
            println!("{record}");
            checks.push(record);
        }
    }
    let passed = checks.iter().all(|v| v["passed"] == true);
    let report = serde_json::json!({"admitted":false,"scope":"retained native English narration vs independent CPU Whisper-base; exact normalized words; no MOS, voice identity, word timing or multilingual claim","passed":passed,"checks":checks});
    use std::io::Write;
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[2])?
        .write_all(&serde_json::to_vec_pretty(&report)?)?;
    if !passed {
        return Err("one or more narration quality cases failed".into());
    }
    Ok(())
}
