//! Two successive teacher-forced traces through all trained codebook layers.
use xrt_audio::drama3::{checkpoint::Checkpoint, tower::TowerSession};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(2..=3).contains(&args.len()) {
        return Err("usage: drama3_tower_probe <managed-model> <new-trace.json>".into());
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
        tower.prefill_traced(&hidden, position, &frame, &next, |layer, after, values| {
            layers.push(serde_json::json!({"layer":layer,"after":after,"values":values}))
        })?;
        traces.push(serde_json::json!({"position":position,"hidden":hidden,"frame":frame,"next":next,"layers":layers}));
    }
    use std::io::Write;
    std::fs::OpenOptions::new().write(true).create_new(true).open(&args[1])?.write_all(&serde_json::to_vec(&serde_json::json!({"admitted":false,"numerics":checkpoint.numerics,"scope":"actual trained native teacher-forced tower traces","traces":traces}))?)?;
    Ok(())
}
