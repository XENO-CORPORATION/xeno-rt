//! Native backbone/tower generation, retaining EOS vs exhausted-budget outcome.
use super::{
    backbone::BackboneSession,
    checkpoint::Checkpoint,
    prompt::PromptRows,
    tower::{Sampling, TowerSession},
};
use crate::{control::InferenceControl, sampling::Rng};
use xrt_core::{Result, XrtError};
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Eos,
    Length,
}
pub struct GeneratedCodes {
    pub frames: Vec<Vec<u32>>,
    pub finish: FinishReason,
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
}
pub fn generate(
    checkpoint: &Checkpoint,
    prompt: &PromptRows,
    max_new_frames: usize,
    state_budget: u64,
    sampling: Sampling,
    seed: u64,
    control: &InferenceControl,
) -> Result<GeneratedCodes> {
    generate_with_progress(
        checkpoint,
        prompt,
        max_new_frames,
        state_budget,
        sampling,
        seed,
        control,
        |_, _| {},
    )
}
#[allow(clippy::too_many_arguments)]
pub fn generate_with_progress(
    checkpoint: &Checkpoint,
    prompt: &PromptRows,
    max_new_frames: usize,
    state_budget: u64,
    sampling: Sampling,
    seed: u64,
    control: &InferenceControl,
    mut progress: impl FnMut(&str, usize),
) -> Result<GeneratedCodes> {
    let width = checkpoint.config.audio_decoder_config.num_codebooks + 1;
    if prompt.is_empty()
        || prompt.width != width
        || prompt.data.len() % width != 0
        || max_new_frames == 0
    {
        return Err(XrtError::Shape(
            "invalid Fish generation prompt/budget".into(),
        ));
    }
    let capacity = prompt
        .len()
        .checked_add(max_new_frames)
        .ok_or_else(|| XrtError::Shape("Fish generation context overflow".into()))?;
    let mut backbone = BackboneSession::new(checkpoint, capacity, state_budget)?;
    let mut tower = TowerSession::new(checkpoint, capacity, state_budget)?;
    let mut hidden = Vec::new();
    let start = std::time::Instant::now();
    let mut history = Vec::with_capacity(capacity);
    for (position, frame) in prompt.data.chunks_exact(width).enumerate() {
        control
            .check()
            .map_err(|e| XrtError::Runtime(e.to_string()))?;
        let base = backbone.embedding(frame[0])?;
        let embedding = tower.backbone_embedding(frame, &base)?;
        hidden = backbone.step(&embedding, control)?;
        progress("prefill", position + 1);
        history.push(frame[0]);
        if position + 1 < prompt.len() {
            tower.prefill(
                &hidden,
                position,
                frame,
                &prompt.data[(position + 1) * width..][..width],
            )?;
        }
    }
    let prefill_seconds = start.elapsed().as_secs_f64();
    let decode_start = std::time::Instant::now();
    let mut rng = Rng::new(seed);
    let mut frames = Vec::new();
    let mut previous = prompt.data[prompt.data.len() - width..].to_vec();
    for _ in 0..max_new_frames {
        control
            .check()
            .map_err(|e| XrtError::Runtime(e.to_string()))?;
        let logits = backbone.semantic_logits(&hidden)?;
        let frame = tower.decode(
            &logits,
            &hidden,
            backbone.position() - 1,
            &previous,
            &history,
            sampling,
            &mut rng,
        )?;
        if frame[0] == checkpoint.config.eos_token_id {
            return Ok(GeneratedCodes {
                frames,
                finish: FinishReason::Eos,
                prefill_seconds,
                decode_seconds: decode_start.elapsed().as_secs_f64(),
            });
        }
        frames.push(frame[1..].to_vec());
        progress("decode", frames.len());
        history.push(frame[0]);
        let base = backbone.embedding(frame[0])?;
        let embedding = tower.backbone_embedding(&frame, &base)?;
        hidden = backbone.step(&embedding, control)?;
        previous = frame;
    }
    Ok(GeneratedCodes {
        frames,
        finish: FinishReason::Length,
        prefill_seconds,
        decode_seconds: decode_start.elapsed().as_secs_f64(),
    })
}
