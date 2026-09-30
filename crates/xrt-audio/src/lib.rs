//! `xrt-audio` — the audio domain of the XENO runtime.
//!
//! Scope: `xrt-audio` (see `docs/RUNTIME_DOMAINS.md`). The runtime owns model
//! loading and inference; consumer apps own timelines, tracks and editing.
//!
//! ## Surface
//! - [`speech::synthesize`] — text-to-speech with zero-shot voice cloning.
//!   Long scripts are split by [`chunking`] into pieces the model can read,
//!   generated one by one, and stitched into a single waveform.
//! - First adapter: [`chatterbox`] (Chatterbox Multilingual v3, MIT).
//!
//! ## What is enforced here, and why
//! Every rule below was measured, not assumed (xrt-audio spike, 2026-09-25/26):
//! - **Chunks stay inside the trained text-position range.** Chatterbox's
//!   learned text position table is trained to ~600 positions; beyond that it
//!   is noise and the model garbles from the first word (914 chars: 2/3 runs
//!   defective, 1439: 3/3, 1899: 3/3). The chunker caps by TOKENS, not
//!   characters, because characters-per-token differs by language.
//! - **A generation that hits the token budget is an error.** The upstream
//!   PyTorch wrapper hardcodes 1000 speech tokens (a silent 40 s cut); here the
//!   budget comes from the model, and reaching it is reported as truncation.
//! - **No "2 identical tokens" stop rule.** Upstream force-stops on two equal
//!   consecutive speech tokens; that fired on all 9 premature stops observed.
//!   Held vowels legitimately repeat a token.

pub mod audio;
pub mod chatterbox;
pub mod chunking;
pub mod control;
pub mod direction;
pub mod installed;
pub mod native;
pub mod prosody;
pub mod sampling;
pub mod speech;
pub mod tokenizer;
pub mod voices;
pub mod whisper;
pub mod wordcheck;
pub mod wordsafe;

pub use speech::{synthesize, ChunkReport, Preset, SpeechOptions, SpeechOutput};

#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    #[error("audio request cancelled")]
    Cancelled,
    /// A model file is missing on disk.
    #[error("model file not found at {path}: {message}")]
    ModelMissing { path: String, message: String },

    /// The loaded ONNX Runtime cannot run this model.
    #[error("ONNX Runtime incompatible: {0}")]
    RuntimeIncompatible(String),

    /// The request is invalid (empty text, unsupported language, bad option).
    #[error("invalid request: {0}")]
    InvalidRequest(String),

    /// The reference voice clip could not be used.
    #[error("invalid reference audio: {0}")]
    InvalidReference(String),

    /// The tokenizer file is missing or malformed.
    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    /// ONNX Runtime / inference itself failed.
    #[error("inference failed: {0}")]
    Inference(String),

    /// A chunk hit the model's speech-token budget without emitting STOP.
    /// Never returned as success: the audio would end mid-sentence.
    #[error("chunk {chunk} was truncated at the model's {limit}-token speech budget")]
    Truncated { chunk: usize, limit: usize },
}

impl From<ort::Error> for AudioError {
    fn from(e: ort::Error) -> Self {
        AudioError::Inference(e.to_string())
    }
}
