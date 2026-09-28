use std::borrow::Cow;
use std::path::{Path, PathBuf};

use ndarray::{Array1, Array2, Array3};
use ort::execution_providers::{CPUExecutionProvider, CUDAExecutionProvider};
use ort::session::{builder::GraphOptimizationLevel, Session, SessionInputValue};
use ort::value::{DynValue, Tensor};

use crate::audio::{self, SAMPLE_RATE};
use crate::sampling::{self, Rng, SamplingParams};
use crate::tokenizer::{ChatterboxTokenizer, START_SPEECH, STOP_SPEECH};
use crate::AudioError;

const LAYERS: usize = 30;
const KV_HEADS: usize = 16;
const HEAD_DIM: usize = 64;
/// Speech-token vocabulary; ids at or above this are control tokens.
const SPEECH_VOCAB: i64 = 6561;
/// The model's own speech-position budget (`max_speech_tokens`). The upstream
/// PyTorch wrapper hardcodes 1000 instead, which silently cuts audio at 40 s.
pub const MODEL_MAX_SPEECH_TOKENS: usize = 4096;
/// The decoder conditions on the first 10 s of the reference (upstream
/// `DEC_COND_LEN`); the language model internally uses the first 6 s.
const REFERENCE_MAX_SECONDS: usize = 10;
/// ONNX Runtime minor version the v3 language_model requires (GroupQueryAttention
/// with 11 inputs; 1.20 accepts at most 9).
const MIN_ORT_MINOR: u32 = 23;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    /// CUDA when the loaded ONNX Runtime provides it, otherwise CPU. The
    /// provider actually used is reported, never assumed.
    Auto,
    Cpu,
    /// Fail rather than fall back if CUDA cannot be registered.
    Cuda(i32),
}

/// Matrix-multiply precision on CUDA.
///
/// ONNX Runtime's CUDA provider enables TF32 (10-bit mantissa) on Ampere+ by
/// default. For this model that is NOT acceptable: teacher-forced against CPU
/// on identical inputs (2026-09-26, RTX 4090, 305 steps), TF32 moved logits by
/// up to 0.029 and changed the min-p candidate set at 4 steps, which is enough
/// for an autoregressive take to fork onto a worse path; listeners heard the
/// forked CUDA takes degrade mid-sentence while the CPU take was clean. FP32
/// matched CPU to 1e-4 with zero candidate-set differences. So FP32 is the
/// default; TF32 remains available for throughput experiments only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CudaPrecision {
    Tf32,
    #[default]
    Fp32,
}

/// Which Chatterbox Multilingual export is loaded. Detected from the graphs
/// themselves, never from a directory name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelVariant {
    /// `t3_mtl23ls_v2` export: `embed_tokens` has no `text_conditioning`, the
    /// LM takes no `position_ids`, full `conditional_decoder`.
    V2,
    /// `t3_mtl23ls_v3` export (KitsuMate four_graph_fp32).
    V3,
}

/// How the CFG uncond row gets its text embedding removed.
enum Uncond {
    /// v3: the graph gates the text embedding itself.
    TextConditioning,
    /// v2: subtract the model's own `text_emb.weight` rows (additive embedding).
    Subtract {
        weight: Vec<f32>,
        rows: usize,
        dim: usize,
    },
}

#[derive(Debug, Clone)]
pub struct ModelPaths {
    pub speech_encoder: PathBuf,
    pub embed_tokens: PathBuf,
    pub language_model: PathBuf,
    pub decoder: PathBuf,
    pub tokenizer: PathBuf,
}

impl ModelPaths {
    /// The export's `four_graph_fp32` layout under a model directory.
    pub fn from_dir(dir: &Path) -> Self {
        let o = dir.join("onnx");
        Self {
            speech_encoder: o.join("speech_encoder.onnx"),
            embed_tokens: o.join("embed_tokens.onnx"),
            language_model: o.join("language_model.onnx"),
            decoder: if o.join("conditional_decoder_slim.onnx").is_file() {
                o.join("conditional_decoder_slim.onnx")
            } else {
                o.join("conditional_decoder.onnx")
            },
            tokenizer: dir.join("tokenizer.json"),
        }
    }
}

/// A voice prepared once from a reference clip and reused for every chunk.
#[derive(Debug, Clone)]
pub struct VoiceConditioning {
    /// Language-model prefix (speaker + prosody prompt), `[1, L, 1024]`.
    pub cond_emb: Array3<f32>,
    /// Decoder prompt speech tokens, `[1, T]`.
    pub prompt_tokens: Array2<i64>,
    /// Speaker x-vector, `[1, 192]`.
    pub speaker_embedding: Array2<f32>,
    /// Reference mel features, `[1, F, 80]`.
    pub speaker_features: Array3<f32>,
}

#[derive(Debug, Clone, Copy)]
pub struct GenerateParams {
    pub exaggeration: f32,
    pub sampling: SamplingParams,
    pub max_speech_tokens: usize,
    pub seed: u64,
}

#[derive(Debug, Clone)]
pub struct Generated {
    pub samples: Vec<f32>,
    pub speech_tokens: usize,
    /// The speech-token ids sent to the decoder (prompt excluded), so a take
    /// can be re-rendered or compared across backends.
    pub token_ids: Vec<i64>,
    /// False when the budget ran out before the model emitted STOP.
    pub stopped: bool,
}

pub struct ChatterboxModel {
    encoder: Session,
    embed: Session,
    lm: Session,
    decoder: Session,
    lm_output_names: Vec<String>,
    lm_takes_positions: bool,
    uncond: Uncond,
    pub variant: ModelVariant,
    pub tokenizer: ChatterboxTokenizer,
    /// Execution provider actually in use, e.g. "cuda:0" or "cpu (cuda unavailable: ...)".
    pub provider: String,
    pub ort_version: String,
}

impl std::fmt::Debug for ChatterboxModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatterboxModel")
            .field("variant", &self.variant)
            .field("provider", &self.provider)
            .field("ort_version", &self.ort_version)
            .finish()
    }
}

impl ChatterboxModel {
    pub fn load(paths: &ModelPaths, device: Device) -> Result<Self, AudioError> {
        Self::load_with(paths, device, CudaPrecision::default())
    }

    pub fn load_with(
        paths: &ModelPaths,
        device: Device,
        precision: CudaPrecision,
    ) -> Result<Self, AudioError> {
        for p in [
            &paths.speech_encoder,
            &paths.embed_tokens,
            &paths.language_model,
            &paths.decoder,
            &paths.tokenizer,
        ] {
            if !p.is_file() {
                return Err(AudioError::ModelMissing {
                    path: p.display().to_string(),
                    message: "Chatterbox Multilingual four-graph ONNX layout expected".into(),
                });
            }
        }
        let ort_version = check_ort_version()?;
        let tokenizer = ChatterboxTokenizer::from_file(&paths.tokenizer)?;

        let (encoder, provider) = session(&paths.speech_encoder, device, precision)?;
        // Every graph uses the provider the first one actually obtained, so a
        // partial CUDA registration cannot split the pipeline across devices.
        let device = if provider.starts_with("cuda") {
            device
        } else {
            Device::Cpu
        };
        let (embed, _) = session(&paths.embed_tokens, device, precision)?;
        let (lm, _) = session(&paths.language_model, device, precision)?;
        let (decoder, _) = session(&paths.decoder, device, precision)?;

        let lm_output_names = lm
            .outputs
            .iter()
            .map(|o| o.name.clone())
            .collect::<Vec<_>>();
        let expected = 1 + 2 * LAYERS;
        if lm_output_names.len() != expected || lm_output_names[0] != "logits" {
            return Err(AudioError::Inference(format!(
                "language_model has {} outputs (expected logits + {} present tensors); not a Chatterbox v3 export",
                lm_output_names.len(), 2 * LAYERS
            )));
        }
        let has_input = |s: &Session, n: &str| s.inputs.iter().any(|i| i.name == n);
        let lm_takes_positions = has_input(&lm, "position_ids");
        let (variant, uncond) = if has_input(&embed, "text_conditioning") {
            (ModelVariant::V3, Uncond::TextConditioning)
        } else {
            let w = super::onnx_initializer::read_f32(&paths.embed_tokens, "text_emb.weight")?;
            if w.dims.len() != 2 {
                return Err(AudioError::Inference(format!(
                    "text_emb.weight has dims {:?}",
                    w.dims
                )));
            }
            (
                ModelVariant::V2,
                Uncond::Subtract {
                    rows: w.dims[0],
                    dim: w.dims[1],
                    weight: w.data,
                },
            )
        };
        tracing::info!(provider = %provider, ort = %ort_version, ?variant, "loaded Chatterbox Multilingual");
        Ok(Self {
            encoder,
            embed,
            lm,
            decoder,
            lm_output_names,
            lm_takes_positions,
            uncond,
            variant,
            tokenizer,
            provider,
            ort_version,
        })
    }

    /// Condition on a reference voice. `samples` is mono audio at `rate`.
    pub fn prepare_voice(
        &self,
        samples: &[f32],
        rate: u32,
    ) -> Result<VoiceConditioning, AudioError> {
        let mut x = audio::resample(samples, rate, SAMPLE_RATE);
        x.truncate(REFERENCE_MAX_SECONDS * SAMPLE_RATE as usize);
        let secs = x.len() as f32 / SAMPLE_RATE as f32;
        if secs < 3.0 {
            return Err(AudioError::InvalidReference(format!(
                "reference is {secs:.1}s; at least 3 s of clean speech is needed (6-10 s recommended)"
            )));
        }
        let peak = x.iter().fold(0.0f32, |a, s| a.max(s.abs()));
        if peak < 1e-3 {
            return Err(AudioError::InvalidReference("reference is silent".into()));
        }
        let n = x.len();
        let out = self.encoder.run(ort::inputs![
            "audio_values" => Tensor::from_array(Array2::from_shape_vec((1, n), x).expect("shape"))?
        ]?)?;
        Ok(VoiceConditioning {
            cond_emb: to_array3(&out["audio_features"])?,
            prompt_tokens: to_array2_i64(&out["audio_tokens"])?,
            speaker_embedding: to_array2_f32(&out["speaker_embeddings"])?,
            speaker_features: to_array3(&out["speaker_features"])?,
        })
    }

    /// Generate one chunk. `text` must already be `punc_norm`-ed.
    pub fn generate(
        &self,
        text: &str,
        language: &str,
        voice: &VoiceConditioning,
        p: &GenerateParams,
    ) -> Result<Generated, AudioError> {
        let ids = self.tokenizer.encode_prompt(text, language)?;
        let n = ids.len();
        // Reference position scheme: text positions count from the first text
        // token; control tokens (EXAGGERATION, START_SPEECH) sit at 0.
        let pos: Vec<i64> = ids
            .iter()
            .enumerate()
            .map(|(i, &t)| {
                if t >= START_SPEECH || i == 0 {
                    0
                } else {
                    i as i64 - 1
                }
            })
            .collect();
        let ids_a = Array2::from_shape_vec((1, n), ids).expect("shape");
        let pos_a = Array2::from_shape_vec((1, n), pos).expect("shape");

        let use_cfg = p.sampling.cfg_weight > 0.0;
        let batch = if use_cfg { 2 } else { 1 };
        let cond_len = voice.cond_emb.shape()[1];
        let hidden = voice.cond_emb.shape()[2];

        // ---- prefill -------------------------------------------------------
        let e_c = self.embed(&ids_a, &pos_a, p.exaggeration, 1.0)?;
        let prefill_len = cond_len + n;
        let mut ie = Array3::<f32>::zeros((batch, prefill_len, hidden));
        for b in 0..batch {
            ie.slice_mut(ndarray::s![b, ..cond_len, ..])
                .assign(&voice.cond_emb.slice(ndarray::s![0, .., ..]));
        }
        ie.slice_mut(ndarray::s![0, cond_len.., ..])
            .assign(&e_c.slice(ndarray::s![0, .., ..]));
        if use_cfg {
            let e_u = self.uncond_embed(&ids_a, &pos_a, p.exaggeration, &e_c)?;
            ie.slice_mut(ndarray::s![1, cond_len.., ..])
                .assign(&e_u.slice(ndarray::s![0, .., ..]));
        }

        let mut past: Vec<DynValue> = (0..2 * LAYERS)
            .map(|_| {
                Tensor::from_array(ndarray::Array4::<f32>::zeros((
                    batch, KV_HEADS, 0, HEAD_DIM,
                )))
                .map(|t| t.into_dyn())
            })
            .collect::<Result<_, _>>()?;
        let mut past_len = 0usize;
        let mut history: Vec<i64> = vec![START_SPEECH];
        let mut rng = Rng::new(p.seed);
        let mut stopped = false;
        let mut embeds = ie;

        for step in 0..p.max_speech_tokens {
            let seq = embeds.shape()[1];
            let total = past_len + seq;
            let mask = Array2::<i64>::ones((batch, total));
            let lm_pos = Array2::from_shape_fn((batch, seq), |(_, t)| (past_len + t) as i64);

            let mut inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> =
                Vec::with_capacity(3 + 2 * LAYERS);
            inputs.push((
                "inputs_embeds".into(),
                Tensor::from_array(embeds)?.into_dyn().into(),
            ));
            inputs.push((
                "attention_mask".into(),
                Tensor::from_array(mask)?.into_dyn().into(),
            ));
            if self.lm_takes_positions {
                inputs.push((
                    "position_ids".into(),
                    Tensor::from_array(lm_pos)?.into_dyn().into(),
                ));
            }
            for (i, v) in past.drain(..).enumerate() {
                let kind = if i % 2 == 0 { "key" } else { "value" };
                inputs.push((format!("past_key_values.{}.{kind}", i / 2).into(), v.into()));
            }
            let mut out = self.lm.run(inputs)?;

            let (shape, data) = out["logits"].try_extract_raw_tensor::<f32>()?;
            let (lseq, vocab) = (shape[1] as usize, shape[2] as usize);
            let row = |b: usize| &data[(b * lseq + lseq - 1) * vocab..(b * lseq + lseq) * vocab];
            let logits = if use_cfg {
                sampling::apply_cfg(row(0), row(1), p.sampling.cfg_weight)
            } else {
                row(0).to_vec()
            };
            let next = sampling::sample(logits, &history, &p.sampling, &mut rng);

            // Carry the KV cache forward without copying it.
            for name in &self.lm_output_names[1..] {
                past.push(out.remove(name.as_str()).ok_or_else(|| {
                    AudioError::Inference(format!("language_model did not return `{name}`"))
                })?);
            }
            drop(out);
            past_len = total;

            history.push(next);
            if next == STOP_SPEECH {
                stopped = true;
                break;
            }
            let e = self.embed(
                &Array2::from_elem((1, 1), next),
                &Array2::from_elem((1, 1), step as i64 + 1),
                p.exaggeration,
                1.0,
            )?;
            embeds = if use_cfg {
                ndarray::concatenate(ndarray::Axis(0), &[e.view(), e.view()]).expect("same shape")
            } else {
                e
            };
        }

        let body: Vec<i64> = history[1..]
            .iter()
            .copied()
            .filter(|&t| (0..SPEECH_VOCAB).contains(&t))
            .collect();
        if body.is_empty() {
            return Ok(Generated {
                samples: Vec::new(),
                speech_tokens: 0,
                token_ids: Vec::new(),
                stopped,
            });
        }
        let samples = self.decode(&body, voice)?;
        Ok(Generated {
            samples,
            speech_tokens: body.len(),
            token_ids: body,
            stopped,
        })
    }

    /// Teacher-forced diagnostic: run the LM over `forced` speech tokens and
    /// return the CFG-combined logits at every step, without sampling. Two
    /// backends fed the same tokens must produce near-identical logits; this is
    /// how numeric divergence is separated from sampling luck.
    pub fn trace_logits(
        &self,
        text: &str,
        language: &str,
        voice: &VoiceConditioning,
        p: &GenerateParams,
        forced: &[i64],
    ) -> Result<Vec<Vec<f32>>, AudioError> {
        let ids = self.tokenizer.encode_prompt(text, language)?;
        let n = ids.len();
        let pos: Vec<i64> = ids
            .iter()
            .enumerate()
            .map(|(i, &t)| {
                if t >= START_SPEECH || i == 0 {
                    0
                } else {
                    i as i64 - 1
                }
            })
            .collect();
        let ids_a = Array2::from_shape_vec((1, n), ids).expect("shape");
        let pos_a = Array2::from_shape_vec((1, n), pos).expect("shape");
        let cond_len = voice.cond_emb.shape()[1];
        let hidden = voice.cond_emb.shape()[2];
        let e_c = self.embed(&ids_a, &pos_a, p.exaggeration, 1.0)?;
        let e_u = self.uncond_embed(&ids_a, &pos_a, p.exaggeration, &e_c)?;
        let mut embeds = Array3::<f32>::zeros((2, cond_len + n, hidden));
        for b in 0..2 {
            embeds
                .slice_mut(ndarray::s![b, ..cond_len, ..])
                .assign(&voice.cond_emb.slice(ndarray::s![0, .., ..]));
        }
        embeds
            .slice_mut(ndarray::s![0, cond_len.., ..])
            .assign(&e_c.slice(ndarray::s![0, .., ..]));
        embeds
            .slice_mut(ndarray::s![1, cond_len.., ..])
            .assign(&e_u.slice(ndarray::s![0, .., ..]));
        let mut past: Vec<DynValue> = (0..2 * LAYERS)
            .map(|_| {
                Tensor::from_array(ndarray::Array4::<f32>::zeros((2, KV_HEADS, 0, HEAD_DIM)))
                    .map(|t| t.into_dyn())
            })
            .collect::<Result<_, _>>()?;
        let mut past_len = 0usize;
        let mut trace = Vec::with_capacity(forced.len());
        for (step, &tok) in forced.iter().enumerate() {
            let seq = embeds.shape()[1];
            let total = past_len + seq;
            let mut inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> =
                Vec::with_capacity(3 + 2 * LAYERS);
            inputs.push((
                "inputs_embeds".into(),
                Tensor::from_array(embeds)?.into_dyn().into(),
            ));
            inputs.push((
                "attention_mask".into(),
                Tensor::from_array(Array2::<i64>::ones((2, total)))?
                    .into_dyn()
                    .into(),
            ));
            if self.lm_takes_positions {
                inputs.push((
                    "position_ids".into(),
                    Tensor::from_array(Array2::from_shape_fn((2, seq), |(_, t)| {
                        (past_len + t) as i64
                    }))?
                    .into_dyn()
                    .into(),
                ));
            }
            for (i, v) in past.drain(..).enumerate() {
                let kind = if i % 2 == 0 { "key" } else { "value" };
                inputs.push((format!("past_key_values.{}.{kind}", i / 2).into(), v.into()));
            }
            let mut out = self.lm.run(inputs)?;
            let (shape, data) = out["logits"].try_extract_raw_tensor::<f32>()?;
            let (lseq, vocab) = (shape[1] as usize, shape[2] as usize);
            let row = |b: usize| &data[(b * lseq + lseq - 1) * vocab..(b * lseq + lseq) * vocab];
            trace.push(sampling::apply_cfg(row(0), row(1), p.sampling.cfg_weight));
            for name in &self.lm_output_names[1..] {
                past.push(
                    out.remove(name.as_str())
                        .ok_or_else(|| AudioError::Inference(format!("missing `{name}`")))?,
                );
            }
            drop(out);
            past_len = total;
            let e = self.embed(
                &Array2::from_elem((1, 1), tok),
                &Array2::from_elem((1, 1), step as i64 + 1),
                p.exaggeration,
                1.0,
            )?;
            embeds =
                ndarray::concatenate(ndarray::Axis(0), &[e.view(), e.view()]).expect("same shape");
        }
        Ok(trace)
    }

    /// Render speech tokens to 24 kHz audio in `voice`.
    pub fn decode(&self, body: &[i64], voice: &VoiceConditioning) -> Result<Vec<f32>, AudioError> {
        let mut all: Vec<i64> = voice.prompt_tokens.iter().copied().collect();
        all.extend_from_slice(body);
        let len = all.len();
        let out = self.decoder.run(ort::inputs![
            "speech_tokens" => Tensor::from_array(Array2::from_shape_vec((1, len), all).expect("shape"))?,
            "speaker_embeddings" => Tensor::from_array(voice.speaker_embedding.clone())?,
            "speaker_features" => Tensor::from_array(voice.speaker_features.clone())?
        ]?)?;
        let (_, wav) = out["waveform"].try_extract_raw_tensor::<f32>()?;
        let mut wav = wav.to_vec();
        // Upstream parity (S3Gen.inference `trim_fade`): the vocoder's first
        // 40 ms carry spillover from the reference clip. Silence 20 ms, then
        // raised-cosine fade in over the next 20 ms.
        let n_trim = (SAMPLE_RATE / 50) as usize;
        for (k, s) in wav.iter_mut().take(2 * n_trim).enumerate() {
            let g = if k < n_trim {
                0.0
            } else {
                let t = (k - n_trim) as f32 / n_trim as f32;
                0.5 - 0.5 * (std::f32::consts::PI * t).cos()
            };
            *s *= g;
        }
        Ok(wav)
    }

    fn embed(
        &self,
        ids: &Array2<i64>,
        pos: &Array2<i64>,
        exaggeration: f32,
        text_conditioning: f32,
    ) -> Result<Array3<f32>, AudioError> {
        let out = match self.uncond {
            Uncond::TextConditioning => self.embed.run(ort::inputs![
                "input_ids" => Tensor::from_array(ids.clone())?,
                "position_ids" => Tensor::from_array(pos.clone())?,
                "exaggeration" => Tensor::from_array(Array1::from_vec(vec![exaggeration]))?,
                "text_conditioning" => Tensor::from_array(Array1::from_vec(vec![text_conditioning]))?
            ]?)?,
            Uncond::Subtract { .. } => self.embed.run(ort::inputs![
                "input_ids" => Tensor::from_array(ids.clone())?,
                "position_ids" => Tensor::from_array(pos.clone())?,
                "exaggeration" => Tensor::from_array(Array1::from_vec(vec![exaggeration]))?
            ]?)?,
        };
        to_array3(&out["inputs_embeds"])
    }

    /// CFG uncond prompt embedding: the reference zeroes ONLY the text
    /// embedding (`text_emb[1].zero_()`), keeping position and emotion.
    fn uncond_embed(
        &self,
        ids: &Array2<i64>,
        pos: &Array2<i64>,
        exaggeration: f32,
        cond: &Array3<f32>,
    ) -> Result<Array3<f32>, AudioError> {
        match &self.uncond {
            Uncond::TextConditioning => self.embed(ids, pos, exaggeration, 0.0),
            Uncond::Subtract { weight, rows, dim } => {
                let mut u = cond.clone();
                if u.shape()[2] != *dim {
                    return Err(AudioError::Inference(
                        "text_emb width does not match inputs_embeds".into(),
                    ));
                }
                // Only real text ids carry a text embedding; the EXAGGERATION
                // and START_SPEECH control ids are outside the table and keep
                // their (emotion / speech) contribution.
                for (k, &tok) in ids.row(0).iter().enumerate() {
                    if tok >= 0 && (tok as usize) < *rows {
                        let row = &weight[tok as usize * dim..(tok as usize + 1) * dim];
                        for (h, w) in row.iter().enumerate() {
                            u[[0, k, h]] -= w;
                        }
                    }
                }
                Ok(u)
            }
        }
    }
}

/// Refuse an ONNX Runtime that cannot run the v3 graphs, with a message that
/// names the fix, instead of the opaque op-schema error it would raise later.
pub(crate) fn check_ort_version() -> Result<String, AudioError> {
    // `ort` panics on a runtime older than its own API version; turn that into
    // an error the server can return.
    let info = std::panic::catch_unwind(ort::info).map_err(|_| {
        AudioError::RuntimeIncompatible(format!(
            "could not load ONNX Runtime (set ORT_DYLIB_PATH to an onnxruntime >= 1.{MIN_ORT_MINOR} library)"
        ))
    })?;
    let version = info
        .split("git-branch=rel-")
        .nth(1)
        .and_then(|s| s.split(|c: char| c == ',' || c.is_whitespace()).next())
        .unwrap_or("unknown")
        .to_string();
    let minor = version
        .split('.')
        .nth(1)
        .and_then(|m| m.parse::<u32>().ok());
    match minor {
        Some(m) if m >= MIN_ORT_MINOR => Ok(version),
        Some(_) => Err(AudioError::RuntimeIncompatible(format!(
            "ONNX Runtime {version} is loaded; Chatterbox v3 needs >= 1.{MIN_ORT_MINOR} \
             (its GroupQueryAttention op takes 11 inputs). See reference/runtime/onnxruntime-1.23.0-windows-x64.json"
        ))),
        None => {
            tracing::warn!(%info, "could not parse ONNX Runtime version; continuing");
            Ok(version)
        }
    }
}

pub(crate) fn session(
    path: &Path,
    device: Device,
    precision: CudaPrecision,
) -> Result<(Session, String), AudioError> {
    let builder = || -> Result<_, AudioError> {
        Ok(Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4),
            )?)
    };
    let cuda_id = match device {
        Device::Cpu => None,
        Device::Auto => Some(0),
        Device::Cuda(id) => Some(id),
    };
    let mut note = String::new();
    if let Some(id) = cuda_id {
        let attempt = builder()?
            .with_execution_providers([CUDAExecutionProvider::default()
                .with_device_id(id)
                .with_tf32(precision == CudaPrecision::Tf32)
                .build()
                .error_on_failure()])
            .and_then(|b| b.commit_from_file(path));
        match attempt {
            Ok(s) => return Ok((s, format!("cuda:{id}"))),
            Err(e) if device == Device::Auto => note = format!(" (cuda unavailable: {e})"),
            Err(e) => {
                return Err(AudioError::Inference(format!(
                    "CUDA requested but unavailable: {e}"
                )))
            }
        }
    }
    let s = builder()?
        .with_execution_providers([CPUExecutionProvider::default().build()])?
        .commit_from_file(path)
        .map_err(|e| AudioError::Inference(format!("failed to load `{}`: {e}", path.display())))?;
    Ok((s, format!("cpu{note}")))
}

fn to_array3(v: &DynValue) -> Result<Array3<f32>, AudioError> {
    let (s, d) = v.try_extract_raw_tensor::<f32>()?;
    if s.len() != 3 {
        return Err(AudioError::Inference(format!(
            "expected rank-3 tensor, got {s:?}"
        )));
    }
    Ok(
        Array3::from_shape_vec((s[0] as usize, s[1] as usize, s[2] as usize), d.to_vec())
            .expect("shape"),
    )
}

fn to_array2_f32(v: &DynValue) -> Result<Array2<f32>, AudioError> {
    let (s, d) = v.try_extract_raw_tensor::<f32>()?;
    if s.len() != 2 {
        return Err(AudioError::Inference(format!(
            "expected rank-2 tensor, got {s:?}"
        )));
    }
    Ok(Array2::from_shape_vec((s[0] as usize, s[1] as usize), d.to_vec()).expect("shape"))
}

fn to_array2_i64(v: &DynValue) -> Result<Array2<i64>, AudioError> {
    let (s, d) = v.try_extract_raw_tensor::<i64>()?;
    let (r, c) = match s.len() {
        1 => (1, s[0] as usize),
        2 => (s[0] as usize, s[1] as usize),
        _ => {
            return Err(AudioError::Inference(format!(
                "expected rank-2 token tensor, got {s:?}"
            )))
        }
    };
    Ok(Array2::from_shape_vec((r, c), d.to_vec()).expect("shape"))
}
