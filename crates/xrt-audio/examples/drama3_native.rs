//! Native development entrypoint, never an admission claim.
//! drama3_native <checkpoint-dir> <text> <output.json> [max-frames] [cuda-cache-MiB]
use xrt_audio::drama3::{
    checkpoint::Checkpoint,
    generate::{self, FinishReason},
    prompt::{self, JsonTokenizer},
    tower::Sampling,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 3 || args.len() > 5 {
        return Err("usage: drama3_native <checkpoint> <text> <new-output.json> [max-frames] [cuda-cache-MiB]".into());
    }
    let root = std::path::Path::new(&args[0]);
    let _reader = xrt_hub::BundleReadLease::acquire_at(root)?;
    let output = std::path::Path::new(&args[2]);
    if output.exists() {
        return Err("refusing to overwrite retained evidence".into());
    }
    xrt_audio::installed::verify_managed(root)?;
    let checkpoint = Checkpoint::open(root)?;
    #[cfg(feature = "cuda")]
    let mut checkpoint = checkpoint;
    let cache = args
        .get(4)
        .map(|s| s.parse::<u64>())
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
            return Err("build with --features cuda for the native expert cache".into());
        }
    }
    let tokenizer = JsonTokenizer::open(&root.join("tokenizer.json"))?;
    let prompt = prompt::build(&checkpoint.config, &tokenizer, &args[1], &[], &[], 8192)?;
    let max = args
        .get(3)
        .map(|s| s.parse::<usize>())
        .transpose()?
        .unwrap_or(8);
    if max > 4096 {
        return Err("development frame budget exceeds 4096".into());
    }
    let control = xrt_audio::control::InferenceControl::default();
    let started = std::time::Instant::now();
    let generated = generate::generate(
        &checkpoint,
        &prompt,
        max,
        2 * 1024 * 1024 * 1024,
        Sampling::default(),
        42,
        &control,
    )?;
    let report = serde_json::json!({"admitted":false,"scope":"native backbone/tower code generation; neural codec and speech qualification still required",
  "checkpoint":root,"backend":if cache>0{"native-cpu-cuda-fp32-reference"}else{"native-cpu-fp32-reference"},
  "prompt_rows":prompt.len(),"width":prompt.width,"frames":generated.frames,
  "finish":if generated.finish==FinishReason::Eos{"eos"}else{"length"},"seconds":started.elapsed().as_secs_f64(),"seed":42});
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    file.write_all(&serde_json::to_vec_pretty(&report)?)?;
    println!(
        "Retained native code-generation evidence: {}",
        output.display()
    );
    Ok(())
}
