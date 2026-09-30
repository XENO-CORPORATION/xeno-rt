//! End-to-end check: `cargo run -p xrt-audio --release --example speak -- <ref.wav> <text-file> <out.wav> [cpu|cuda]`
//! Prints the per-chunk report so a defect is attributable to a chunk.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 3 {
        eprintln!("usage: speak <reference.wav> <text-file> <out.wav> [auto|cpu|cuda]");
        std::process::exit(2);
    }
    let (reference, rate) = xrt_audio::audio::read_wav(&std::fs::read(&a[0])?)?;
    let script = std::fs::read_to_string(&a[1])?;
    let mut opts = xrt_audio::SpeechOptions {
        device: match a.get(3).map(String::as_str) {
            Some("cpu") => xrt_audio::chatterbox::Device::Cpu,
            Some("cuda") => xrt_audio::chatterbox::Device::Cuda(0),
            _ => xrt_audio::chatterbox::Device::Auto,
        },
        ..Default::default()
    };
    // Delivery knobs for listening sweeps (XRT_TTS_EXAGGERATION / _CFG / _TEMPERATURE / _SEED).
    if let Ok(name) = std::env::var("XRT_TTS_PRESET") {
        xrt_audio::Preset::parse(&name)
            .ok_or("unknown XRT_TTS_PRESET")?
            .apply(&mut opts);
    }
    let env = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok());
    if let Some(v) = env("XRT_TTS_PAUSE_SCALE") {
        opts.pause_scale = v;
    }
    if let Some(v) = env("XRT_TTS_BREATH_DB") {
        opts.breath_reduction_db = v;
    }
    if let Some(v) = env("XRT_TTS_EXAGGERATION") {
        opts.exaggeration = v;
    }
    if let Some(v) = env("XRT_TTS_CFG") {
        opts.cfg_weight = v;
    }
    if let Some(v) = env("XRT_TTS_TEMPERATURE") {
        opts.temperature = v;
    }
    if let Some(v) = env("XRT_TTS_SEED") {
        opts.seed = v as u64;
    }
    eprintln!(
        "exaggeration {} cfg {} temperature {} seed {} pause_scale {}",
        opts.exaggeration, opts.cfg_weight, opts.temperature, opts.seed, opts.pause_scale
    );
    let t = std::time::Instant::now();
    let out = xrt_audio::speech::synthesize_with(
        &script,
        &reference,
        rate,
        &opts,
        &xrt_audio::speech::SignalValidator::default(),
        |c| {
            eprintln!(
                "  chunk {:2}: {:4} tok -> {:5.1}s  tries {}  wer {}  breaths {}  +pause {:.2}s{}",
                c.index,
                c.text_tokens,
                c.seconds,
                c.attempts,
                c.word_error_rate
                    .map_or("-".into(), |r| format!("{:.0}%", r * 100.0)),
                c.breaths_softened,
                c.pause_added_s,
                if c.best_effort { "  BEST-EFFORT" } else { "" }
            );
            for r in &c.rejected {
                eprintln!("      rejected: {r}");
            }
            for p in &c.word_problems {
                eprintln!("      kept: {p}");
            }
        },
    )?;
    std::fs::write(
        &a[2],
        xrt_audio::audio::write_wav(&out.samples, out.sample_rate),
    )?;
    let words = std::path::Path::new(&a[2]).with_extension("words.json");
    std::fs::write(&words, serde_json::to_vec_pretty(&out.words)?)?;
    eprintln!(
        "provider {} asr {:?}  audio {:.1}s  wall {:.1}s  chunks {}  words {} -> {}",
        out.provider,
        out.asr_provider,
        out.seconds,
        t.elapsed().as_secs_f32(),
        out.chunks.len(),
        out.words.len(),
        words.display()
    );
    Ok(())
}
