//! Two successive residual traces through every actual trained backbone layer.
use xrt_audio::drama3::{backbone::BackboneSession, checkpoint::Checkpoint};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(2..=3).contains(&args.len()) {
        return Err("usage: drama3_backbone_probe <managed-model> <new-trace.json>".into());
    }
    let root = std::path::Path::new(&args[0]);
    let _reader = xrt_hub::BundleReadLease::acquire_at(root)?;
    xrt_audio::installed::verify_managed(root)?;
    let mut checkpoint = Checkpoint::open(root)?;
    if args.get(2).is_some_and(|v| v == "bf16-w8a8") {
        checkpoint.numerics = xrt_audio::drama3::numerics::Numerics::Bf16W8a8;
    } else if args.len() == 3 {
        return Err("unsupported numerics; use bf16-w8a8 or omit it".into());
    }
    let mut session = BackboneSession::new(&checkpoint, 2, 2 * 1024 * 1024 * 1024)?;
    let control = xrt_audio::control::InferenceControl::default();
    let mut traces = Vec::new();
    for position in 0..2 {
        let mut input: Vec<_> = (0..checkpoint.config.text_config.hidden_size)
            .map(|i| {
                if position == 0 {
                    (i as f32 * 0.013).sin()
                } else {
                    (i as f32 * 0.017).cos()
                }
            })
            .collect();
        checkpoint.numerics.round_checked(&mut input)?;
        let mut layers = Vec::new();
        let hidden = session.step_traced(&input, &control, |layer, after, values| {
            layers.push(serde_json::json!({"layer":layer,"after":after,"values":values}));
        })?;
        traces.push(
            serde_json::json!({"position":position,"input":input,"layers":layers,"hidden":hidden}),
        );
    }
    let report = serde_json::json!({"admitted":false,"numerics":checkpoint.numerics,"scope":"actual trained native backbone layer traces","traces":traces});
    use std::io::Write;
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[1])?
        .write_all(&serde_json::to_vec(&report)?)?;
    Ok(())
}
