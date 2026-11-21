//! The gates of `docs/AUDIO_ADMISSION_TRANSCRIPTION.md` for a Parakeet model.
//!
//! Usage: `parakeet_admission <model-dir> <wav-dir> <refs.json>`
//!
//! `refs.json` maps a wav file name to its reference text. Every gate starts as
//! `None`, and a `None` at the end fails the run, so a gate cannot pass by not
//! running. Peak memory is sampled by the caller.
use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use xrt_audio::{control::InferenceControl, parakeet::ParakeetModel};

/// Lowercase, keep letters and digits. Deliberately the same plain normaliser
/// as the whisper-base run, so the two numbers compare.
fn words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split_whitespace()
        .map(|word| {
            word.chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
        })
        .filter(|word| !word.is_empty())
        .collect()
}

fn edit_distance(a: &[String], b: &[String]) -> usize {
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, x) in a.iter().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, y) in b.iter().enumerate() {
            let next = (diagonal + usize::from(x != y))
                .min(row[j] + 1)
                .min(row[j + 1] + 1);
            diagonal = row[j + 1];
            row[j + 1] = next;
        }
    }
    row[b.len()]
}

fn gate(name: &str, measured: String, passed: bool) -> Option<bool> {
    println!(
        "{} {name}: {measured}",
        if passed { "PASS" } else { "FAIL" }
    );
    Some(passed)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [model_dir, wav_dir, refs] = &args[..] else {
        return Err("usage: parakeet_admission <model-dir> <wav-dir> <refs.json>".into());
    };
    let refs: BTreeMap<String, String> = serde_json::from_slice(&std::fs::read(refs)?)?;
    if refs.is_empty() {
        return Err("empty corpus".into());
    }
    let load = Instant::now();
    let mut model = ParakeetModel::load(Path::new(model_dir))?;
    println!("loaded in {:.2} s", load.elapsed().as_secs_f32());

    let control = InferenceControl::default();
    let (mut wrong, mut total, mut seconds, mut busy) = (0usize, 0usize, 0f32, 0f32);
    let mut first_result = None;
    let mut output = Vec::new();
    for (file, reference) in &refs {
        let (audio, rate) =
            xrt_audio::audio::read_wav(&std::fs::read(Path::new(wav_dir).join(file))?)?;
        let started = Instant::now();
        let transcript = model.transcribe_controlled(&audio, rate, &control)?;
        let elapsed = started.elapsed().as_secs_f32();
        first_result.get_or_insert(elapsed);
        busy += elapsed;
        seconds += audio.len() as f32 / rate as f32;
        let (want, got) = (words(reference), words(&transcript.text));
        wrong += edit_distance(&want, &got);
        total += want.len();
        output.push(transcript.text);
    }

    let (mut correctness, mut throughput, mut latency, mut determinism) = (None, None, None, None);
    let wer = 100.0 * wrong as f32 / total as f32;
    correctness = correctness.or(gate(
        "reference_correctness",
        format!("{wer:.2}% WER over {} clips", refs.len()),
        wer <= 10.0,
    ));
    let speed = seconds / busy;
    throughput = throughput.or(gate(
        "throughput",
        format!("{speed:.1}x realtime over {seconds:.0} s"),
        speed >= 3.0,
    ));
    let first = first_result.unwrap_or(f32::INFINITY) * 1000.0;
    latency = latency.or(gate(
        "first_segment_latency",
        format!("{first:.0} ms"),
        first <= 5000.0,
    ));

    let (file, _) = refs.iter().next().expect("corpus is not empty");
    let (audio, rate) = xrt_audio::audio::read_wav(&std::fs::read(Path::new(wav_dir).join(file))?)?;
    let repeat = model.transcribe_controlled(&audio, rate, &control)?;
    determinism = determinism.or(gate(
        "determinism",
        "same clip twice".into(),
        repeat.text == output[0],
    ));

    let gates = [correctness, throughput, latency, determinism];
    if gates.iter().any(|g| *g != Some(true)) {
        return Err("a gate failed or did not run".into());
    }
    println!("first clip: {}", output[0]);
    Ok(())
}
