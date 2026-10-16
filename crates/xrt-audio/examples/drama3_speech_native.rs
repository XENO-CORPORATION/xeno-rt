//! Native development speech entrypoint. It does not register public support.
//! drama3_speech_native <model-dir> <codec-dir> <text> <new-output.wav> [expert-cache-MiB] [dense-budget-MiB]
use xrt_audio::drama3::{
    checkpoint::Checkpoint,
    codec::Codec,
    generate::{self, FinishReason},
    prompt::{self, JsonTokenizer},
    tower::Sampling,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(4..=6).contains(&args.len()) {
        return Err(
            "usage: drama3_speech_native <model> <codec> <text> <new-output.wav> [expert-cache-MiB] [dense-budget-MiB]"
                .into(),
        );
    }
    let model_root = std::path::Path::new(&args[0]);
    let _reader = xrt_hub::BundleReadLease::acquire_at(model_root)?;
    let output = std::path::Path::new(&args[3]);
    let receipt = output.with_extension("json");
    if output.exists() || receipt.exists() {
        return Err("refusing to overwrite retained speech evidence".into());
    }
    xrt_audio::installed::verify_managed(model_root)?;
    let checkpoint = Checkpoint::open(model_root)?;
    #[cfg(feature = "cuda")]
    let mut checkpoint = checkpoint;
    let cache = args
        .get(4)
        .map(|v| v.parse::<u64>())
        .transpose()?
        .unwrap_or(0);
    if cache > 0 {
        #[cfg(feature = "cuda")]
        {
            checkpoint.enable_expert_cache(
                0,
                cache
                    .checked_mul(1024 * 1024)
                    .ok_or("cache budget overflow")?,
            )?;
        }
        #[cfg(not(feature = "cuda"))]
        {
            return Err("native CUDA expert cache requires --features cuda".into());
        }
    }
    let dense = args
        .get(5)
        .map(|v| v.parse::<u64>())
        .transpose()?
        .unwrap_or(0);
    if dense > 0 {
        #[cfg(feature = "cuda")]
        {
            let bytes = checkpoint.enable_dense_cuda(
                0,
                dense
                    .checked_mul(1024 * 1024)
                    .ok_or("dense budget overflow")?,
            )?;
            eprintln!("{}", serde_json::json!({"dense_cuda_resident_bytes":bytes}));
        }
        #[cfg(not(feature = "cuda"))]
        {
            return Err("dense CUDA execution requires --features cuda".into());
        }
    }
    let tokenizer = JsonTokenizer::open(&model_root.join("tokenizer.json"))?;
    let prompt = prompt::build(&checkpoint.config, &tokenizer, &args[2], &[], &[], 8192)?;
    let codec = Codec::open(std::path::Path::new(&args[1]), 2 * 1024 * 1024 * 1024)?;
    let control = xrt_audio::control::InferenceControl::default();
    let start = std::time::Instant::now();
    let generated = generate::generate_with_progress(
        &checkpoint,
        &prompt,
        128,
        2 * 1024 * 1024 * 1024,
        Sampling::default(),
        42,
        &control,
        |phase, count| eprintln!("{}", serde_json::json!({"phase":phase,"completed":count})),
    )?;
    if generated.finish != FinishReason::Eos {
        return Err(
            "native speech exhausted its development frame budget; no truncated WAV delivered"
                .into(),
        );
    }
    if generated.frames.is_empty() {
        return Err("native speech produced no acoustic frames".into());
    }
    let codec_start = std::time::Instant::now();
    let out = codec.decode(&generated.frames, &control)?;
    let codec_seconds = codec_start.elapsed().as_secs_f64();
    let seconds = start.elapsed().as_secs_f64();
    let duration = out.samples.len() as f64 / out.sample_rate as f64;
    let bytes = xrt_audio::audio::write_wav(&out.samples, out.sample_rate);
    #[cfg(feature = "cuda")]
    let cache_stats = checkpoint.cache_stats()?.map(|s| serde_json::json!({"resident_bytes":s.resident_bytes,"peak_resident_bytes":s.peak_resident_bytes,"hits":s.hits,"misses":s.misses,"evictions":s.evictions,"workspace_bytes":s.workspace_bytes,"peak_workspace_bytes":s.peak_workspace_bytes,"upload_seconds":s.upload_seconds,"eviction_seconds":s.eviction_seconds,"execution_seconds":s.execution_seconds}));
    #[cfg(not(feature = "cuda"))]
    let cache_stats: Option<serde_json::Value> = None;
    #[cfg(feature = "cuda")]
    let dense_bytes = checkpoint.dense_resident_bytes();
    #[cfg(not(feature = "cuda"))]
    let dense_bytes = 0u64;
    let report = serde_json::json!({"admitted":false,
  "scope":"native development speech; quality, service and installed-tier qualification remain required","model":"fish-drama3-native-dev",
  "input":args[2],"seed":42,"backend":if cache>0||dense>0{"native-cpu-cuda-fp32-reference"}else{"native-cpu-fp32-reference"},
  "expert_cache_budget_mib":cache,"dense_budget_mib":dense,
  "dense_resident_bytes":dense_bytes,"expert_cache":cache_stats,"projection_profile":checkpoint.projection_profile()?,
  "prefill_seconds":generated.prefill_seconds,"decode_seconds":generated.decode_seconds,"codec_seconds":codec_seconds,
  "frames":generated.frames.len(),"sample_rate":out.sample_rate,"samples":out.samples.len(),"duration":duration,
  "seconds":seconds,"real_time_factor":seconds/duration,"clamped_residual_codes":out.clamped_residual_codes});
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    file.write_all(&bytes)?;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(receipt)?;
    file.write_all(&serde_json::to_vec_pretty(&report)?)?;
    println!("{report}");
    Ok(())
}
