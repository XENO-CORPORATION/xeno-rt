//! Reusable-engine development qualification, including A/B/A and cancellation.
use serde::Deserialize;
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};
use xrt_audio::{
    control::InferenceControl,
    drama3::{
        engine::{Budgets, Engine, NativeError, Request},
        prompt::Reference,
    },
};
use xrt_core::gpu_resource::GpuAllocationArena;
fn seed() -> u64 {
    42
}
fn frames() -> usize {
    128
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    id: String,
    text: String,
    #[serde(default = "seed")]
    seed: u64,
    #[serde(default = "frames")]
    max_frames: usize,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    reference: Option<Ref>,
    #[serde(default)]
    references: Vec<Ref>,
    cancel_after_frames: Option<usize>,
    expected_error: Option<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Ref {
    wav: PathBuf,
    transcript: String,
}
fn write(path: PathBuf, value: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)?
        .write_all(value)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(6..=7).contains(&args.len()) {
        return Err("usage: drama3_qualify_native <managed-model> <native-codec> <cases.json> <new-directory> <expert-MiB> <dense-MiB> [fp32-reference|bf16-w8a8]".into());
    }
    let out = Path::new(&args[3]);
    if out.exists() {
        return Err("refusing to overwrite qualification directory".into());
    }
    if std::fs::metadata(&args[2])?.len() > 1024 * 1024 {
        return Err("qualification cases exceed 1 MiB".into());
    }
    let cases: Vec<Case> = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    if cases.is_empty() || cases.len() > 32 {
        return Err("qualification requires 1..32 cases".into());
    }
    let mut ids = std::collections::HashSet::new();
    for case in &cases {
        if case.id.is_empty()
            || case.id.len() > 64
            || !case
                .id
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'-')
            || !ids.insert(case.id.clone())
        {
            return Err("invalid/duplicate case id".into());
        }
        if case
            .expected_error
            .as_deref()
            .is_some_and(|v| !matches!(v, "cancelled" | "frame_limit"))
        {
            return Err("unknown expected error".into());
        }
        if case.reference.is_some() && !case.references.is_empty() {
            return Err("use reference or references, not both".into());
        }
        if case.cancel_after_frames == Some(0) {
            return Err("cancel threshold must be positive".into());
        }
    }
    let numerics = match args.get(6).map(String::as_str).unwrap_or("fp32-reference") {
        "fp32-reference" => xrt_audio::drama3::numerics::Numerics::Fp32Reference,
        "bf16-w8a8" => xrt_audio::drama3::numerics::Numerics::Bf16W8a8,
        _ => return Err("unknown native numerics".into()),
    };
    let budgets = Budgets {
        numerics,
        expert_bytes: args[4]
            .parse::<u64>()?
            .checked_mul(1024 * 1024)
            .ok_or("expert budget overflow")?,
        codec_bytes: xrt_audio::drama3::codec::Codec::decoding_workspace_bytes(
            cases.iter().map(|c| c.max_frames).max().unwrap(),
        )?
        .max(2 * 1024 * 1024 * 1024),
        dense_bytes: args[5]
            .parse::<u64>()?
            .checked_mul(1024 * 1024)
            .ok_or("dense budget overflow")?,
        ..Budgets::default()
    };
    let arena = Arc::new(GpuAllocationArena::default());
    // Dedicated qualification process. Real hosts pass their existing shared
    // arena instead; this is not a second independent server memory broker.
    arena.configure_budget(22 * 1024 * 1024 * 1024)?;
    let engine = Engine::load(
        Path::new(&args[0]),
        Path::new(&args[1]),
        budgets,
        Some(arena.clone()),
    )?;
    std::fs::create_dir(out)?;
    write(
        out.join("load.json"),
        &serde_json::to_vec_pretty(
            &serde_json::json!({"admitted":false,"load_seconds":engine.load_seconds,"planned_host_bytes":engine.planned_host_bytes,"dense_resident_bytes":engine.dense_resident_bytes(),"numerics":numerics,"gpu_admission":arena.snapshot()}),
        )?,
    )?;
    let mut failures = Vec::new();
    for case in cases {
        let control = InferenceControl::default();
        let all_refs: Vec<_> = case
            .reference
            .iter()
            .chain(case.references.iter())
            .collect();
        let mut encoded = Vec::new();
        for reference in &all_refs {
            if std::fs::metadata(&reference.wav)?.len() > 16 * 1024 * 1024 {
                return Err("reference WAV exceeds 16 MiB".into());
            }
            let (samples, rate) = xrt_audio::audio::read_wav(&std::fs::read(&reference.wav)?)?;
            encoded.push(engine.encode_reference(&samples, rate, &control)?);
        }
        let refs: Vec<_> = all_refs
            .iter()
            .zip(&encoded)
            .map(|(r, e)| Reference {
                transcript: &r.transcript,
                frames: &e.frames,
            })
            .collect();
        let mut request = Request::plain(&case.text);
        request.seed = case.seed;
        request.max_frames = case.max_frames;
        if let Some(v) = case.temperature {
            request.sampling.temperature = v;
        }
        if let Some(v) = case.top_p {
            request.sampling.top_p = v;
        }
        if let Some(v) = case.top_k {
            request.sampling.top_k = v;
        }
        let sampling = request.sampling;
        request.references = &refs;
        let mut busy_checked = false;
        let result = std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    engine.synthesize(request, &control, |phase, count| {
                        if phase == "prefill" && count == 1 {
                            let other = InferenceControl::default();
                            busy_checked = matches!(
                                engine.synthesize(
                                    Request::plain("Concurrent probe."),
                                    &other,
                                    |_, _| {}
                                ),
                                Err(NativeError::Busy)
                            ) && matches!(
                                engine.encode_reference(&[0.0], 44100, &other),
                                Err(NativeError::Busy)
                            );
                        }
                        eprintln!(
                            "{}",
                            serde_json::json!({"id":case.id,"phase":phase,"completed":count})
                        );
                        if phase == "decode"
                            && case.cancel_after_frames.is_some_and(|limit| count >= limit)
                        {
                            control.cancel();
                        }
                    })
                })
                .join()
        })
        .map_err(|_| "native qualification worker panicked")?;
        if !busy_checked {
            return Err("native engine did not enforce exclusive inference ownership".into());
        }
        let report = match result {
            Ok(speech) => {
                if case.expected_error.is_some() {
                    failures.push(format!("{} expected an error but produced speech", case.id));
                }
                let wav =
                    xrt_audio::audio::write_wav(&speech.audio.samples, speech.audio.sample_rate);
                use sha2::{Digest, Sha256};
                let hash = format!("{:x}", Sha256::digest(&wav));
                write(out.join(format!("{}.wav", case.id)), &wav)?;
                write(
                    out.join(format!("{}.codes.json", case.id)),
                    &serde_json::to_vec(&speech.frames)?,
                )?;
                let duration = speech.audio.samples.len() as f64 / speech.audio.sample_rate as f64;
                serde_json::json!({"id":case.id,"admitted":false,"scope":"native reusable-engine development qualification","text":case.text,"seed":case.seed,"numerics":numerics,"sampling":{"temperature":sampling.temperature,"top_p":sampling.top_p,"top_k":sampling.top_k},"timings":speech.timings,"real_time_factor":speech.timings.seconds/duration,"duration":duration,"frames":speech.frames.len(),"clamped_residual_codes":speech.audio.clamped_residual_codes,"wav_sha256":hash,"projections":speech.projections})
            }
            Err(error) => {
                let kind = match error {
                    NativeError::Cancelled => "cancelled",
                    NativeError::FrameLimit => "frame_limit",
                    _ => "inference_error",
                };
                if case.expected_error.as_deref() != Some(kind) {
                    failures.push(format!("{} unexpected {kind}: {error}", case.id));
                }
                serde_json::json!({"id":case.id,"admitted":false,"error":kind,"partial_audio_discarded":true})
            }
        };
        write(
            out.join(format!("{}.json", case.id)),
            &serde_json::to_vec_pretty(&report)?,
        )?;
        println!("{report}");
    }
    drop(engine);
    let final_admission = arena.snapshot();
    if final_admission.allocated_bytes != 0 {
        return Err("native engine leaked GPU admission leases".into());
    }
    write(
        out.join("cleanup.json"),
        &serde_json::to_vec_pretty(&final_admission)?,
    )?;
    write(
        out.join("summary.json"),
        &serde_json::to_vec_pretty(
            &serde_json::json!({"admitted":false,"passed":failures.is_empty(),"failures":failures}),
        )?,
    )?;
    if !failures.is_empty() {
        return Err(format!("{} qualification cases failed", failures.len()).into());
    }
    Ok(())
}
