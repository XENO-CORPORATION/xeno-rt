//! Trained dense operators at explicitly rounded BF16 activation boundaries.
use xrt_audio::drama3::{checkpoint::Checkpoint, numerics::Numerics};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: drama3_precision_probe <managed-model> <new-report.json>".into());
    }
    let root = std::path::Path::new(&args[0]);
    let _reader = xrt_hub::BundleReadLease::acquire_at(root)?;
    xrt_audio::installed::verify_managed(root)?;
    let mut checkpoint = Checkpoint::open(root)?;
    checkpoint.numerics = Numerics::Bf16W8a8;
    let prefix = "audio_decoder_mtp_shared.layers.0";
    let mut checks = Vec::new();
    for suffix in [
        "hnorm",
        "enorm",
        "transformer_layer.attention_norm",
        "transformer_layer.attention.q_norm",
    ] {
        let name = format!("{prefix}.{suffix}.weight");
        let weight = checkpoint.vector(&name)?;
        let input: Vec<_> = (0..weight.len())
            .map(|i| checkpoint.numerics.round((i as f32 * 0.013).sin()))
            .collect();
        let output = checkpoint.norm(
            &input,
            &weight,
            checkpoint.config.audio_decoder_config.norm_eps,
            true,
        )?;
        checks.push(serde_json::json!({"op":"norm","name":name,"input":input,"output":output}));
    }
    for suffix in [
        "eh_proj",
        "transformer_layer.attention.wqkv",
        "transformer_layer.attention.wo",
        "transformer_layer.feed_forward.w1",
        "transformer_layer.feed_forward.w3",
        "transformer_layer.feed_forward.w2",
    ] {
        let name = format!("{prefix}.{suffix}.weight");
        let shape = &checkpoint
            .tensors
            .tensor_info(&name)
            .ok_or("missing tensor")?
            .shape;
        let input: Vec<_> = (0..shape[1])
            .map(|i| checkpoint.numerics.round((i as f32 * 0.013).sin()))
            .collect();
        let output = checkpoint.linear(&name, &input)?;
        checks.push(serde_json::json!({"op":"linear","name":name,"input":input,"output":output}));
    }
    use std::io::Write;
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[1])?
        .write_all(&serde_json::to_vec(&checks)?)?;
    Ok(())
}
