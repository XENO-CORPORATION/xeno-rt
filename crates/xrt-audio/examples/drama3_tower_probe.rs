//! Two successive teacher-forced traces through all trained codebook layers.
use xrt_audio::drama3::{checkpoint::Checkpoint, tower::TowerSession};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(2..=5).contains(&args.len()) {
        return Err("usage: drama3_tower_probe <managed-model> <new-trace.json> [fp32-reference|bf16-w8a8] [audit-layer|all] [dense-mib]".into());
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
    let audit_all = args.get(3).is_some_and(|v| v == "all");
    let audit_layer = args
        .get(3)
        .filter(|_| !audit_all)
        .map(|v| v.parse::<usize>())
        .transpose()?;
    let layer_count = checkpoint
        .config
        .audio_decoder_config
        .shared_enrichment_layers
        + checkpoint.config.audio_decoder_config.num_codebooks
        - 1;
    if audit_layer.is_some_and(|v| v >= layer_count) {
        return Err("audit layer is outside the trained tower".into());
    }
    if (audit_all || audit_layer.is_some()) && !cfg!(feature = "drama3-diagnostics") {
        return Err("operator capture requires --features drama3-diagnostics".into());
    }
    let dense_mib = args
        .get(4)
        .map(|v| v.parse::<u64>())
        .transpose()?
        .unwrap_or(0);
    if dense_mib > 8192 {
        return Err("dense budget exceeds 8192 MiB".into());
    }
    #[cfg(feature = "cuda")]
    if dense_mib > 0 {
        checkpoint.enable_dense_cuda(0, dense_mib * 1024 * 1024)?;
    }
    #[cfg(not(feature = "cuda"))]
    if dense_mib > 0 {
        return Err("dense CUDA probe requires cuda feature".into());
    }
    let mut tower = TowerSession::new(&checkpoint, 2, 2 * 1024 * 1024 * 1024)?;
    let mut traces = Vec::new();
    for position in 0..2 {
        let mut hidden: Vec<_> = (0..checkpoint.config.audio_decoder_config.dim)
            .map(|i| {
                if position == 0 {
                    (i as f32 * 0.013).sin()
                } else {
                    (i as f32 * 0.017).cos()
                }
            })
            .collect();
        checkpoint.numerics.round_checked(&mut hidden)?;
        let row = |offset: u32| {
            let mut row = vec![checkpoint.config.semantic_start_token_id + offset];
            row.extend((0..10).map(|i| offset + i));
            row
        };
        let frame = if position == 0 {
            vec![checkpoint.config.eos_token_id, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        } else {
            row(7)
        };
        let next = row(if position == 0 { 11 } else { 21 });
        let mut layers = Vec::new();
        #[cfg(feature = "drama3-diagnostics")]
        let mut audit_error = None;
        tower.prefill_traced(&hidden, position, &frame, &next, |layer, after, values| {
            #[allow(unused_mut)]
            let mut record = serde_json::json!({"layer":layer,"after":after,"values":values});
            #[cfg(feature = "drama3-diagnostics")]
            if audit_all || audit_layer == Some(layer) {
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
            layers.push(record)
        })?;
        #[cfg(feature = "drama3-diagnostics")]
        if let Some(e) = audit_error {
            return Err(e.into());
        }
        traces.push(serde_json::json!({"position":position,"hidden":hidden,"frame":frame,"next":next,"layers":layers}));
    }
    use std::io::Write;
    std::fs::OpenOptions::new().write(true).create_new(true).open(&args[1])?.write_all(&serde_json::to_vec(&serde_json::json!({"admitted":false,"numerics":checkpoint.numerics,"scope":"actual trained native teacher-forced tower traces","traces":traces}))?)?;
    Ok(())
}
