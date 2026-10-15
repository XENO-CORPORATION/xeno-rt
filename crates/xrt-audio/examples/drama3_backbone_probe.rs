//! Two successive residual traces through every actual trained backbone layer.
use xrt_audio::drama3::{backbone::BackboneSession, checkpoint::Checkpoint};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(2..=4).contains(&args.len()) {
        return Err("usage: drama3_backbone_probe <managed-model> <new-trace.json> [fp32-reference|bf16-w8a8] [operator-layer]".into());
    }
    let root = std::path::Path::new(&args[0]);
    let _reader = xrt_hub::BundleReadLease::acquire_at(root)?;
    xrt_audio::installed::verify_managed(root)?;
    let mut checkpoint = Checkpoint::open(root)?;
    if args.get(2).is_some_and(|v| v == "bf16-w8a8") {
        checkpoint.numerics = xrt_audio::drama3::numerics::Numerics::Bf16W8a8;
    } else if args.len() >= 3 && args[2] != "fp32-reference" {
        return Err("unsupported numerics; use bf16-w8a8 or omit it".into());
    }
    let audit_layer = args.get(3).map(|v| v.parse::<usize>()).transpose()?;
    if audit_layer.is_some_and(|layer| layer >= checkpoint.config.text_config.num_hidden_layers) {
        return Err("operator layer is outside the trained backbone".into());
    }
    if audit_layer.is_some() && !cfg!(feature = "drama3-diagnostics") {
        return Err("operator capture requires --features drama3-diagnostics".into());
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
        #[cfg(feature = "drama3-diagnostics")]
        let mut audit_error = None;
        let hidden = session.step_traced(&input, &control, |layer, after, values| {
            #[allow(unused_mut)]
            let mut record = serde_json::json!({"layer":layer,"after":after,"values":values});
            #[cfg(feature = "drama3-diagnostics")]
            if audit_layer == Some(layer) {
                if !after {
                    if let Err(e) = checkpoint.begin_operator_trace(16 * 1024 * 1024) {
                        audit_error = Some(e);
                    }
                } else {
                    match checkpoint.take_operator_trace() {
                        Ok(operators) => {
                            record["operators"] =
                                serde_json::to_value(operators).expect("finite diagnostic values")
                        }
                        Err(e) => audit_error = Some(e),
                    }
                }
            }
            layers.push(record);
        })?;
        #[cfg(feature = "drama3-diagnostics")]
        if let Some(e) = audit_error {
            return Err(e.into());
        }
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
