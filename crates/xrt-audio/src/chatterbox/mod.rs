//! Chatterbox Multilingual v3 adapter (four-graph ONNX layout).
//!
//! Graphs, read from the export itself (not the model card, whose tensor
//! names are wrong):
//! - `speech_encoder`   : `audio_values[1,S]@24k` → `audio_features[1,L,1024]`,
//!   `audio_tokens[1,T]`, `speaker_embeddings[1,192]`, `speaker_features[1,F,80]`
//! - `embed_tokens`     : `input_ids`, `position_ids`, `exaggeration[B]`,
//!   `text_conditioning[B]` → `inputs_embeds[B,N,1024]`
//! - `language_model`   : 0.5B Llama, 30 layers × 16 KV heads × 64, KV-cached
//! - `conditional_decoder_slim` : speech tokens + speaker → 24 kHz waveform
//!
//! The decode loop follows the calling convention proven against PyTorch
//! (xrt-audio spike `v3_probe.py`): CFG as batch-of-2 where the uncond row is
//! `text_conditioning = 0.0` (exactly the reference's `text_emb[1].zero_()`),
//! LM `position_ids` continuing from the KV-cache length, and speech-token
//! positions `step + 1`.

mod model;
pub(crate) use model::{check_ort_version, session};
mod onnx_initializer;

pub use model::{
    ChatterboxModel, CudaPrecision, Device, GenerateParams, Generated, ModelPaths, ModelVariant,
    VoiceConditioning,
};

/// The model's own speech-token budget (upstream `max_speech_tokens`).
pub fn model_max_speech_tokens() -> usize {
    model::MODEL_MAX_SPEECH_TOKENS
}
