//! Bounded synthetic, prompt, and teacher-forced traces through the trained backbone.
use xrt_audio::drama3::{
    backbone::BackboneSession,
    checkpoint::Checkpoint,
    prompt::{self, JsonTokenizer},
    tower::TowerSession,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(2..=8).contains(&args.len()) {
        return Err("usage: drama3_backbone_probe <managed-model> <new-trace.json> [fp32-reference|bf16-w8a8] [operator-layer|none] [prompt-text] [expert-MiB] [dense-MiB] [teacher-forced-codes.json]".into());
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
    let audit_layer = args
        .get(3)
        .filter(|v| *v != "none")
        .map(|v| v.parse::<usize>())
        .transpose()?;
    if audit_layer.is_some_and(|layer| layer >= checkpoint.config.text_config.num_hidden_layers) {
        return Err("operator layer is outside the trained backbone".into());
    }
    if audit_layer.is_some() && !cfg!(feature = "drama3-diagnostics") {
        return Err("operator capture requires --features drama3-diagnostics".into());
    }
    let expert_mib = args
        .get(5)
        .map(|v| v.parse::<u64>())
        .transpose()?
        .unwrap_or(0);
    let dense_mib = args
        .get(6)
        .map(|v| v.parse::<u64>())
        .transpose()?
        .unwrap_or(0);
    if expert_mib > 20480 || dense_mib > 8192 {
        return Err("probe GPU budgets exceed bounded profile".into());
    }
    #[cfg(feature = "cuda")]
    {
        if expert_mib > 0 {
            checkpoint.enable_expert_cache(0, expert_mib * 1024 * 1024)?;
        }
        if dense_mib > 0 {
            checkpoint.enable_dense_cuda(0, dense_mib * 1024 * 1024)?;
        }
    }
    #[cfg(not(feature = "cuda"))]
    if expert_mib > 0 || dense_mib > 0 {
        return Err("GPU probe requires the cuda feature".into());
    }
    let mut prompt = if let Some(text) = args.get(4) {
        if text.chars().count() > 128 {
            return Err("probe text exceeds 128 characters".into());
        }
        let tokenizer = JsonTokenizer::open(&root.join("tokenizer.json"))?;
        Some(prompt::build(
            &checkpoint.config,
            &tokenizer,
            text,
            &[],
            &[],
            64,
        )?)
    } else {
        None
    };
    let prompt_rows = prompt.as_ref().map_or(0, |p| p.len());
    let mut teacher_forced_frames = 0;
    let mut teacher_forced_sha256 = None;
    if let Some(path) = args.get(7) {
        use sha2::{Digest, Sha256};
        use std::io::Read;
        let mut bytes = Vec::new();
        std::fs::File::open(path)?
            .take(1024 * 1024 + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() > 1024 * 1024 {
            return Err("teacher-forced code fixture exceeds 1 MiB".into());
        }
        let frames: Vec<Vec<u32>> = serde_json::from_slice(&bytes)?;
        let config = &checkpoint.config;
        if frames.is_empty()
            || frames.len() > 1024
            || frames.iter().any(|f| {
                f.len() != config.audio_decoder_config.num_codebooks
                    || f.iter()
                        .any(|c| *c as usize >= config.audio_decoder_config.vocab_size)
            })
        {
            return Err("invalid teacher-forced code fixture".into());
        }
        let rows = prompt
            .as_mut()
            .ok_or("teacher forcing requires a text prompt")?;
        teacher_forced_frames = frames.len().min(8);
        if rows.len() + teacher_forced_frames > 64 {
            return Err("prompt and teacher-forced frames exceed 64 rows".into());
        }
        for frame in frames.iter().take(teacher_forced_frames) {
            rows.data.push(config.semantic_start_token_id + frame[0]);
            rows.data.extend_from_slice(frame);
        }
        teacher_forced_sha256 = Some(format!("{:x}", Sha256::digest(&bytes)));
    }
    let steps = prompt.as_ref().map_or(2, |p| p.len());
    let tower = TowerSession::new(&checkpoint, steps, 2 * 1024 * 1024 * 1024)?;
    let mut session = BackboneSession::new(&checkpoint, steps, 2 * 1024 * 1024 * 1024)?;
    let control = xrt_audio::control::InferenceControl::default();
    let mut traces = Vec::new();
    for position in 0..steps {
        let mut input: Vec<_> = if let Some(prompt) = &prompt {
            let frame = &prompt.data[position * prompt.width..][..prompt.width];
            tower.backbone_embedding(frame, &session.embedding(frame[0])?)?
        } else {
            (0..checkpoint.config.text_config.hidden_size)
                .map(|i| {
                    if position == 0 {
                        (i as f32 * 0.013).sin()
                    } else {
                        (i as f32 * 0.017).cos()
                    }
                })
                .collect()
        };
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
        let logits = session.semantic_logits(&hidden)?;
        let cfg = &checkpoint.config;
        let mut semantic_logits = logits
            [cfg.semantic_start_token_id as usize..=cfg.semantic_end_token_id as usize]
            .to_vec();
        semantic_logits.push(logits[cfg.eos_token_id as usize]);
        traces.push(
            serde_json::json!({"position":position,"input":input,"layers":layers,"hidden":hidden,"semantic_logits":semantic_logits}),
        );
    }
    let report = serde_json::json!({"admitted":false,"numerics":checkpoint.numerics,"scope":"actual trained native backbone layer traces","prompt_rows":prompt_rows,"teacher_forced_frames":teacher_forced_frames,"teacher_forced_sha256":teacher_forced_sha256,"traces":traces});
    use std::io::Write;
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[1])?
        .write_all(&serde_json::to_vec(&report)?)?;
    Ok(())
}
