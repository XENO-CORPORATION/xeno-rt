//! Native reference WAV to acoustic codes; no Python inference dependency.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err(
            "usage: drama3_reference_probe <encoder-codec-dir> <44.1-kHz.wav> <new-prefix>".into(),
        );
    }
    let prefix = std::path::Path::new(&args[2]);
    for ext in ["codes.json", "json"] {
        if prefix.with_extension(ext).exists() {
            return Err("refusing to overwrite reference evidence".into());
        }
    }
    if std::fs::metadata(&args[1])?.len() > 16 * 1024 * 1024 {
        return Err("reference WAV exceeds 16 MiB".into());
    }
    let (samples, rate) = xrt_audio::audio::read_wav(&std::fs::read(&args[1])?)?;
    let codec = xrt_audio::drama3::codec::Codec::open(
        std::path::Path::new(&args[0]),
        2 * 1024 * 1024 * 1024,
    )?;
    let control = xrt_audio::control::InferenceControl::default();
    let start = std::time::Instant::now();
    let encoded = codec.encode(&samples, rate, &control)?;
    let report = serde_json::json!({"admitted":false,"scope":"native reference encoder","seconds":start.elapsed().as_secs_f64(),"frames":encoded.frames.len(),"original_samples":encoded.original_samples,"padded_samples":encoded.padded_samples,"sample_rate":encoded.sample_rate});
    use std::io::Write;
    for (path, bytes) in [
        (
            prefix.with_extension("codes.json"),
            serde_json::to_vec(&encoded.frames)?,
        ),
        (
            prefix.with_extension("json"),
            serde_json::to_vec_pretty(&report)?,
        ),
    ] {
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?
            .write_all(&bytes)?;
    }
    println!("{report}");
    Ok(())
}
