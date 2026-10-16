//! Decode explicit neural codes using the native tensor-only codec.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err(
            "usage: drama3_codec_probe <native-codec-dir> <codes.json> <new-output-prefix>".into(),
        );
    }
    let codes: Vec<Vec<u32>> = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let codec =
        xrt_audio::drama3::codec::Codec::open(std::path::Path::new(&args[0]), 1024 * 1024 * 1024)?;
    let control = xrt_audio::control::InferenceControl::default();
    let started = std::time::Instant::now();
    let out = codec.decode(&codes, &control)?;
    let prefix = std::path::Path::new(&args[2]);
    for ext in ["wav", "f32", "json"] {
        if prefix.with_extension(ext).exists() {
            return Err("refusing to overwrite codec evidence".into());
        }
    }
    std::fs::write(
        prefix.with_extension("wav"),
        xrt_audio::audio::write_wav(&out.samples, out.sample_rate),
    )?;
    let bytes: Vec<u8> = out.samples.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(prefix.with_extension("f32"), bytes)?;
    let report = serde_json::json!({"admitted":false,"scope":"native trained neural codec decoding","seconds":started.elapsed().as_secs_f64(),"samples":out.samples.len(),"sample_rate":out.sample_rate,"clamped_residual_codes":out.clamped_residual_codes,"peak":out.samples.iter().fold(0.0f32,|a,x|a.max(x.abs()))});
    std::fs::write(
        prefix.with_extension("json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{report}");
    Ok(())
}
