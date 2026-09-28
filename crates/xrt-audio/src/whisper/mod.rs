//! Speech recognition with word timings (Whisper, ONNX).
//!
//! Inside the TTS pipeline this is the ground truth for two jobs:
//! - **the word check**: every generated chunk is transcribed and compared
//!   with its script, and a chunk that dropped or repeated a phrase is
//!   regenerated (signal gates cannot see a fluent-but-wrong sentence);
//! - **word-safe post-processing**: breath softening and pause lengthening
//!   may only touch audio outside the words' timestamped spans, so no
//!   consonant can be damaged (the 2026-09-27 audit found sound-based
//!   detection altering word endings).
//!
//! Export: `onnx-community/whisper-small_timestamped` (encoder,
//! decoder_model, decoder_with_past_model), whose decoder exposes
//! cross-attention for word alignment. Word times use OpenAI's method:
//! alignment-head cross-attention, normalised per head, median-filtered,
//! then dynamic time warping from tokens to 20 ms encoder frames.

pub mod mel;
mod model;
mod tokenizer;

pub use model::{Recognizer, Word};
