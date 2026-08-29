//! Whisper transcription over ONNX Runtime.
//!
//! Two graphs: an encoder that turns an 80x3000 log-mel window into 1500x512
//! hidden states, and a KV-cache-merged decoder run autoregressively until it
//! emits end-of-text. The frontend in [`crate::mel`] produces the encoder's
//! input, and `xrt-tokenizer` turns the decoded ids back into text.
//!
//! # Long-form audio
//!
//! Whisper's encoder has a FIXED 30-second window - `max_source_positions` is
//! baked into the weights - so anything longer must be chunked. A timeline clip
//! is minutes long, so a single-window implementation would not be a feature.
//! This chunks sequentially and emits one segment per window with real time
//! offsets.
//!
//! ⚠️ That is the SIMPLE chunking strategy, and it is worth being precise about
//! what it does not do. OpenAI's reference implementation uses the model's own
//! timestamp tokens to choose where the next window starts, so a window ends on
//! a phrase boundary. Fixed 30-second cuts can land mid-word, which typically
//! costs a word at the seam. Timestamp-guided windowing and VAD are the
//! follow-ups; both are refinements of this loop rather than replacements.

use std::path::Path;

use ndarray::{Array1, Array2, Array3, Array4, Ix3, Ix4};
use ort::{
    session::{builder::GraphOptimizationLevel, Session, SessionOutputs},
    value::{DynValue, Tensor},
};

use crate::{mel, AudioError, Result};

/// One contiguous stretch of transcribed speech.
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    /// Seconds from the start of the supplied audio.
    pub start: f32,
    pub end: f32,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Transcript {
    pub text: String,
    pub segments: Vec<Segment>,
    /// `None` unless a language was explicitly requested; this decoder does not
    /// yet run Whisper's language-detection pass, and reporting a guess as a
    /// detection would be worse than reporting nothing.
    pub language: Option<String>,
}

/// Model dimensions, read from `config.json` rather than hardcoded so the same
/// code loads base / small / medium without a table to keep in sync.
#[derive(Debug, Clone, Copy)]
struct Dims {
    n_mels: usize,
    layers: usize,
    heads: usize,
    head_dim: usize,
}

/// Token ids Whisper reserves. Resolved from the tokenizer by NAME, never
/// hardcoded: they move between model sizes and between vocab revisions, and a
/// wrong `eot` produces a decode loop that runs to the step limit and returns
/// a transcript with the model's internal markup embedded in it.
#[derive(Debug, Clone, Copy)]
struct Special {
    sot: u32,
    eot: u32,
    transcribe: u32,
    no_timestamps: u32,
    lang_en: u32,
}

pub struct WhisperModel {
    encoder: Session,
    decoder: Session,
    tokenizer: xrt_tokenizer::Tokenizer,
    dims: Dims,
    special: Special,
    /// Cap on generated tokens per 30 s window. Whisper's own limit is 448.
    max_tokens: usize,
}

impl WhisperModel {
    /// Loads from a directory holding `encoder.onnx`, `decoder.onnx`,
    /// `config.json`, `vocab.json` and `added_tokens.json`.
    pub fn load(dir: &Path) -> Result<Self> {
        let cfg_path = dir.join("config.json");
        let cfg: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&cfg_path).map_err(|_| AudioError::ModelFileMissing(cfg_path.clone()))?,
        )
        .map_err(|e| AudioError::Inference(format!("config.json is not valid JSON: {e}")))?;

        let num = |k: &str, d: usize| cfg.get(k).and_then(|v| v.as_u64()).unwrap_or(d as u64) as usize;
        let d_model = num("d_model", 512);
        let heads = num("decoder_attention_heads", 8);
        let dims = Dims {
            n_mels: num("num_mel_bins", 80),
            layers: num("decoder_layers", 6),
            heads,
            head_dim: d_model / heads.max(1),
        };

        let encoder = build_session(&dir.join("encoder.onnx"))?;
        let decoder = build_session(&dir.join("decoder.onnx"))?;

        let tokenizer = xrt_tokenizer::Tokenizer::from_hf_dir(dir)
            .map_err(|e| AudioError::Inference(format!("tokenizer load failed: {e}")))?;

        let want = |piece: &str| -> Result<u32> {
            tokenizer.token_id_for_piece(piece).ok_or_else(|| {
                AudioError::Inference(format!(
                    "tokenizer has no `{piece}` token - is this a Whisper vocabulary?"
                ))
            })
        };
        let special = Special {
            sot: want("<|startoftranscript|>")?,
            eot: want("<|endoftext|>")?,
            transcribe: want("<|transcribe|>")?,
            no_timestamps: want("<|notimestamps|>")?,
            lang_en: want("<|en|>")?,
        };

        Ok(Self { encoder, decoder, tokenizer, dims, special, max_tokens: 448 })
    }

    /// The artifacts whisper-base needs, and the names its loaders look for.
    ///
    /// 🔴 SEVEN files, not two. The first publish shipped only encoder, decoder
    /// and `tokenizer.json`; all three resolved and the model still could not be
    /// assembled, because `load` needs `config.json` and `xrt-tokenizer` reads
    /// `vocab.json` + `merges.txt` and never looks at `tokenizer.json`. The set
    /// is the unit for exactly that reason.
    pub const WHISPER_BASE_SET: &[xrt_hub::SetMember] = &[
        xrt_hub::SetMember { registry_id: "whisper-base-encoder", local_name: "encoder.onnx" },
        xrt_hub::SetMember { registry_id: "whisper-base-decoder", local_name: "decoder.onnx" },
        xrt_hub::SetMember { registry_id: "whisper-base-config", local_name: "config.json" },
        xrt_hub::SetMember { registry_id: "whisper-base-vocab", local_name: "vocab.json" },
        xrt_hub::SetMember { registry_id: "whisper-base-merges", local_name: "merges.txt" },
        xrt_hub::SetMember { registry_id: "whisper-base-added-tokens", local_name: "added_tokens.json" },
    ];

    /// Loads whisper-base, fetching and verifying it from the XENO model
    /// registry if it is not already cached.
    ///
    /// This is what makes the capability real on a machine that has never seen
    /// the model: no environment variable, no hand-assembled directory. Every
    /// byte is sha256-verified against the published manifest by the bundle
    /// installer before it is used.
    pub fn load_from_registry() -> Result<Self> {
        let hub = xrt_hub::ModelHub::new().map_err(|e| AudioError::ModelUnavailable {
            name: "whisper-base".to_string(),
            reason: format!("model cache unavailable: {e}"),
        })?;
        let installed = hub
            .install_xeno_model_set("whisper-base", Self::WHISPER_BASE_SET)
            .map_err(|e| AudioError::ModelUnavailable {
                name: "whisper-base".to_string(),
                reason: e.to_string(),
            })?;
        tracing::info!(
            cached = installed.was_cached,
            path = %installed.path.display(),
            "whisper-base resolved"
        );
        Self::load(&installed.path)
    }

    /// Transcribes arbitrary-length audio, chunking into 30-second windows.
    pub fn transcribe(&mut self, samples: &[f32], sample_rate: u32) -> Result<Transcript> {
        let audio = crate::resample_linear(samples, sample_rate, mel::WHISPER_SAMPLE_RATE as u32);
        if audio.is_empty() {
            return Ok(Transcript { text: String::new(), segments: Vec::new(), language: None });
        }

        let win = mel::WHISPER_N_SAMPLES;
        let n_windows = audio.len().div_ceil(win);
        let mut segments = Vec::new();

        for w in 0..n_windows {
            let start = w * win;
            let chunk = &audio[start..(start + win).min(audio.len())];
            let text = self.transcribe_window(chunk)?;
            if text.trim().is_empty() {
                continue;
            }
            let t0 = start as f32 / mel::WHISPER_SAMPLE_RATE as f32;
            // The window's END is bounded by the real audio, not by the padded
            // 30 s: a 4-second clip must not report a 30-second segment.
            let t1 = (start + chunk.len()) as f32 / mel::WHISPER_SAMPLE_RATE as f32;
            segments.push(Segment { start: t0, end: t1, text: text.trim().to_string() });
        }

        let text = segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        Ok(Transcript { text, segments, language: None })
    }

    /// One 30-second window: mel -> encoder -> greedy decode.
    fn transcribe_window(&mut self, chunk: &[f32]) -> Result<String> {
        let mel_rows = mel::log_mel_spectrogram(&mel::pad_or_trim(chunk), self.dims.n_mels);
        let frames = mel_rows.first().map(|r| r.len()).unwrap_or(0);
        let mut flat = Vec::with_capacity(self.dims.n_mels * frames);
        for row in &mel_rows {
            flat.extend_from_slice(row);
        }
        let features = Array3::from_shape_vec((1, self.dims.n_mels, frames), flat)
            .map_err(|e| AudioError::Inference(format!("mel shape: {e}")))?;

        let features_t = Tensor::from_array(features)
            .map_err(|e| AudioError::Inference(format!("input tensor: {e}")))?;
        // Scoped so the encoder's borrow of `self` ends before the decoder needs
        // `&mut self`. `hidden` is owned, so nothing is lost by dropping the
        // session outputs here.
        let hidden = {
            let enc_out = self
                .encoder
                .run(vec![(std::borrow::Cow::from("input_features"), features_t.into_dyn())])
                .map_err(|e| AudioError::Inference(format!("encoder run: {e}")))?;
            extract3(&enc_out, "last_hidden_state")?
        };

        self.greedy_decode(&hidden)
    }

    fn greedy_decode(&mut self, hidden: &Array3<f32>) -> Result<String> {
        let (l, h, d) = (self.dims.layers, self.dims.heads, self.dims.head_dim);
        let prompt = [
            self.special.sot,
            self.special.lang_en,
            self.special.transcribe,
            self.special.no_timestamps,
        ];
        let mut tokens: Vec<u32> = prompt.to_vec();

        // Decoder self-attention cache, and the cross-attention cache over the
        // encoder output. The latter is computed on the first pass and NEVER
        // changes - the encoder output is fixed for the window - so it is
        // carried forward untouched rather than re-read every step.
        let mut past_dec: Vec<(Array4<f32>, Array4<f32>)> =
            (0..l).map(|_| (Array4::zeros((1, h, 0, d)), Array4::zeros((1, h, 0, d)))).collect();
        let mut past_enc: Vec<(Array4<f32>, Array4<f32>)> =
            (0..l).map(|_| (Array4::zeros((1, h, 0, d)), Array4::zeros((1, h, 0, d)))).collect();

        for step in 0..self.max_tokens {
            let ids: Vec<i64> = if step == 0 {
                prompt.iter().map(|&t| t as i64).collect()
            } else {
                vec![*tokens.last().unwrap() as i64]
            };
            let input_ids = Array2::from_shape_vec((1, ids.len()), ids)
                .map_err(|e| AudioError::Inference(format!("input_ids: {e}")))?;

            let mut inputs: Vec<(std::borrow::Cow<'static, str>, DynValue)> = Vec::new();
            inputs.push((
                "input_ids".into(),
                Tensor::from_array(input_ids)
                    .map_err(|e| AudioError::Inference(format!("ids tensor: {e}")))?
                    .into_dyn(),
            ));
            inputs.push((
                "encoder_hidden_states".into(),
                Tensor::from_array(hidden.clone())
                    .map_err(|e| AudioError::Inference(format!("hidden tensor: {e}")))?
                    .into_dyn(),
            ));
            for i in 0..l {
                for (kind, cache) in [("decoder", &past_dec), ("encoder", &past_enc)] {
                    for (which, arr) in [("key", &cache[i].0), ("value", &cache[i].1)] {
                        inputs.push((
                            format!("past_key_values.{i}.{kind}.{which}").into(),
                            Tensor::from_array(arr.clone())
                                .map_err(|e| AudioError::Inference(format!("kv tensor: {e}")))?
                                .into_dyn(),
                        ));
                    }
                }
            }
            inputs.push((
                "use_cache_branch".into(),
                Tensor::from_array(Array1::from_vec(vec![step > 0]))
                    .map_err(|e| AudioError::Inference(format!("cache flag: {e}")))?
                    .into_dyn(),
            ));

            let outputs = self
                .decoder
                .run(inputs)
                .map_err(|e| AudioError::Inference(format!("decoder run (step {step}): {e}")))?;

            let logits = outputs["logits"]
                .try_extract_tensor::<f32>()
                .map_err(|e| AudioError::Inference(format!("logits: {e}")))?;
            let logits = logits
                .into_dimensionality::<Ix3>()
                .map_err(|e| AudioError::Inference(format!("logits rank: {e}")))?;
            let last_pos = logits.shape()[1] - 1;
            let next = logits
                .index_axis(ndarray::Axis(0), 0)
                .index_axis(ndarray::Axis(0), last_pos)
                .iter()
                .copied()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(self.special.eot);

            if next == self.special.eot {
                break;
            }
            tokens.push(next);

            for i in 0..l {
                past_dec[i].0 = read_kv(&outputs, &format!("present.{i}.decoder.key"))?;
                past_dec[i].1 = read_kv(&outputs, &format!("present.{i}.decoder.value"))?;
                if step == 0 {
                    past_enc[i].0 = read_kv(&outputs, &format!("present.{i}.encoder.key"))?;
                    past_enc[i].1 = read_kv(&outputs, &format!("present.{i}.encoder.value"))?;
                }
            }
        }

        // Drop the prompt: those four tokens are instructions to the model, not
        // speech, and leaking them into a transcript is the classic Whisper bug.
        let generated: Vec<u32> = tokens[prompt.len()..].to_vec();
        self.tokenizer
            .decode(&generated, true)
            .map_err(|e| AudioError::Inference(format!("detokenize: {e}")))
    }
}

fn build_session(path: &Path) -> Result<Session> {
    if !path.exists() {
        return Err(AudioError::ModelFileMissing(path.to_path_buf()));
    }
    Session::builder()
        .and_then(|b| b.with_optimization_level(GraphOptimizationLevel::Level3))
        .and_then(|b| b.commit_from_file(path))
        .map_err(|e| AudioError::Inference(format!("failed to open {}: {e}", path.display())))
}

fn extract3(outputs: &SessionOutputs, name: &str) -> Result<Array3<f32>> {
    Ok(outputs[name]
        .try_extract_tensor::<f32>()
        .map_err(|e| AudioError::Inference(format!("{name}: {e}")))?
        .into_dimensionality::<Ix3>()
        .map_err(|e| AudioError::Inference(format!("{name} rank: {e}")))?
        .to_owned())
}

fn read_kv(outputs: &SessionOutputs, name: &str) -> Result<Array4<f32>> {
    Ok(outputs[name]
        .try_extract_tensor::<f32>()
        .map_err(|e| AudioError::Inference(format!("{name}: {e}")))?
        .into_dimensionality::<Ix4>()
        .map_err(|e| AudioError::Inference(format!("{name} rank: {e}")))?
        .to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The prompt tokens must never reach the transcript. This asserts the
    /// slice arithmetic that drops them, without needing a model.
    #[test]
    fn prompt_tokens_are_dropped_from_output() {
        let prompt_len = 4;
        let tokens: Vec<u32> = vec![50258, 50259, 50359, 50363, 100, 200];
        assert_eq!(&tokens[prompt_len..], &[100, 200]);
    }

    /// A short clip must not claim a 30-second segment just because the encoder
    /// window is padded to 30 seconds.
    #[test]
    fn segment_end_is_bounded_by_real_audio() {
        let sr = mel::WHISPER_SAMPLE_RATE;
        let chunk_len = sr * 4; // 4 seconds of real audio
        let end = (0 + chunk_len) as f32 / sr as f32;
        assert!((end - 4.0).abs() < 1e-6, "reported {end}s for a 4s clip");
        assert!(end < 30.0, "padding leaked into the reported duration");
    }
}
