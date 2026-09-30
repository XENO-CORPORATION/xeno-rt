//! Long-form text-to-speech:
//! chunk → generate → trim → signal gate → word check → re-roll → shape → stitch.
//!
//! This is the "proper way" the quality products use (ElevenLabs Projects,
//! the Chatterbox audiobook forks): the model never sees more text than it was
//! trained on, every piece is checked the moment it is produced, and a piece
//! that fails is regenerated with a new seed instead of shipped.
//!
//! Two gates, because they catch different failures:
//! - the SIGNAL gate ([`SignalValidator`]) — silence, runaway length,
//!   truncation, internal holes — is cheap and runs first;
//! - the WORD check ([`crate::wordcheck`]) transcribes the take and compares
//!   it with the script, catching the fluent-but-wrong take no signal
//!   measure can see: a dropped phrase, a repeat, a garbled stretch.
//!
//! The word check's alignment then drives delivery shaping: breaths between
//! words are softened and pauses at sentence and paragraph ends lengthened,
//! touching only audio outside the words' spans ([`crate::wordsafe`]).

use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::Mutex;
use serde::Serialize;

use crate::audio::{self, SAMPLE_RATE};
use crate::chatterbox::{ChatterboxModel, Device, GenerateParams, ModelPaths, VoiceConditioning};
use crate::chunking::{chunk_script, Chunk, ChunkLimits};
use crate::sampling::SamplingParams;
use crate::tokenizer::punc_norm;
use crate::whisper::{Recognizer, Word};
use crate::wordcheck::{self, TimedWord, WordCheckLimits};
use crate::AudioError;

#[derive(Debug, Clone)]
pub struct SpeechOptions {
    pub model_dir: PathBuf,
    pub device: Device,
    /// BCP-47-ish language code, e.g. "en".
    pub language: String,
    /// Emotion / intensity. 0.5 is neutral; higher reads more dramatic.
    pub exaggeration: f32,
    /// Guidance weight. Lower (~0.3) gives slower, more deliberate delivery.
    pub cfg_weight: f32,
    pub temperature: f32,
    pub repetition_penalty: f32,
    pub min_p: f32,
    /// 1.0 for the multilingual model (upstream default).
    pub top_p: f32,
    pub seed: u64,
    pub chunk_limits: ChunkLimits,
    /// Pause at a sentence end, seconds — between chunks, and (when the word
    /// check runs) inside a chunk, where the model's own pause is lengthened
    /// to at least this. Never shortened.
    pub sentence_pause: f32,
    /// Pause at a paragraph end, seconds (same rules).
    pub paragraph_pause: f32,
    /// Generations per chunk before giving up.
    pub max_attempts: u32,
    pub max_speech_tokens: usize,
    /// Multiply the pauses the model placed at clause and sentence ends
    /// (after the sentence/paragraph minimums). 1.0 = unchanged. Needs the
    /// word check: pauses are found from word times, never from the sound.
    pub pause_scale: f32,
    /// Lower model-generated breaths — everything between words — by this
    /// many dB. Needs the word check, for the same reason. 0 = off.
    pub breath_reduction_db: f32,
    /// Transcribe every take and re-roll one that dropped, repeated or
    /// garbled words. Also what enables pause and breath shaping.
    pub word_check: bool,
    /// `$XRT_AUDIO_ASR_DIR`, else `~/.xeno/models/whisper-small-timestamped`.
    pub asr_dir: PathBuf,
    /// Proper nouns in the script, matched more loosely (recognizers spell
    /// names creatively: "Ma'at" → "mart").
    pub names: Vec<String>,
    pub word_limits: WordCheckLimits,
    /// Deliver a take that passed the signal gate but failed the word check
    /// after every attempt (the best one, with its problems reported),
    /// instead of failing the request.
    pub accept_best_effort: bool,
    pub direction: Option<crate::direction::DirectionPlan>,
}

/// Named delivery presets.
///
/// Measured on v3 (2026-09-26, same paragraph, seed 42, GPU): the lower
/// `cfg_weight`, the more often the model rambles — 0.5 passed first try,
/// 0.35 needed 4 attempts, 0.3 needed 3, 0.2 failed every attempt. Resemble's
/// "cfg 0.3 for drama" advice sits at that edge. So presets keep the model in
/// its reliable band (cfg ≥ 0.45) and get documentary pacing from exact,
/// deterministic cadence — the silence placed BETWEEN chunks — which cannot
/// make the model ramble and never touches a word. Pace itself comes from the
/// reference voice: a calm,
/// explanatory reference reads slower and steadier than a theatrical one
/// (measured: pitch range 5.6-6.6 st vs 9.9 st, and first-try success).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preset {
    /// Upstream defaults.
    Neutral,
    /// Documentary narration: weight, slower pace, longer beats.
    Documentary,
    /// Stronger emotion for dramatic reads; closer to the stability edge.
    Dramatic,
}

impl Preset {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "neutral" | "default" => Some(Preset::Neutral),
            "documentary" | "narration" => Some(Preset::Documentary),
            "dramatic" => Some(Preset::Dramatic),
            _ => None,
        }
    }

    /// Apply this preset's delivery settings on top of `opts`.
    pub fn apply(self, opts: &mut SpeechOptions) {
        // In-chunk pause expansion is NOT set by any preset: the 2026-09-27
        // stage audit showed it alters words. Presets shape only the model
        // settings and the between-chunk silence.
        let (ex, cfg, sentence, paragraph) = match self {
            Preset::Neutral => (0.5, 0.5, 0.28, 0.63),
            Preset::Documentary => (0.6, 0.5, 0.5, 0.9),
            Preset::Dramatic => (0.7, 0.45, 0.55, 1.0),
        };
        opts.exaggeration = ex;
        opts.cfg_weight = cfg;
        opts.sentence_pause = sentence;
        opts.paragraph_pause = paragraph;
    }
}

impl Default for SpeechOptions {
    fn default() -> Self {
        Self {
            model_dir: default_model_dir(),
            device: Device::Auto,
            language: "en".into(),
            exaggeration: 0.5,
            cfg_weight: 0.5,
            temperature: 0.8,
            // Upstream multilingual `generate` default, and what the PyTorch
            // reference the user approved was rendered with.
            repetition_penalty: 2.0,
            min_p: 0.05,
            top_p: 1.0,
            seed: 42,
            chunk_limits: ChunkLimits::default(),
            sentence_pause: 0.28,
            paragraph_pause: 0.63,
            max_attempts: 4,
            max_speech_tokens: crate::chatterbox::model_max_speech_tokens(),
            pause_scale: 1.0,
            // The model breathes ~4x as often as the speaker (8.8/min vs
            // 2.1/min, measured 2026-09-27). 15 dB keeps each breath audible
            // as a human pause while taking the "whoosh" out of it. Word-safe:
            // applied only between recognized words.
            breath_reduction_db: 15.0,
            word_check: true,
            asr_dir: Recognizer::default_dir(),
            names: Vec::new(),
            word_limits: WordCheckLimits::default(),
            accept_best_effort: false,
            direction: None,
        }
    }
}

/// `$XRT_AUDIO_MODEL_DIR`, else `~/.xeno/models/chatterbox-multilingual-v3`.
pub fn default_model_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("XRT_AUDIO_MODEL_DIR") {
        return PathBuf::from(d);
    }
    if let Ok(dir) = crate::installed::model_dir("chatterbox-multilingual-v3") {
        return dir;
    }
    let home = std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .unwrap_or_default();
    PathBuf::from(home)
        .join(".xeno")
        .join("models")
        .join("chatterbox-multilingual-v3")
}

#[derive(Debug, Clone, Serialize)]
pub struct ChunkReport {
    pub index: usize,
    pub text: String,
    pub text_tokens: usize,
    pub speech_tokens: usize,
    pub seconds: f32,
    pub attempts: u32,
    /// Why earlier attempts were rejected, in order.
    pub rejected: Vec<String>,
    /// Word mismatches the delivered take still has (tolerated isolated
    /// ones, or all of them when delivered best-effort). Empty = verbatim.
    pub word_problems: Vec<String>,
    /// Mismatches per script word; `None` when the word check did not run.
    pub word_error_rate: Option<f32>,
    /// True when no attempt passed the word check and the best one was
    /// delivered because `accept_best_effort` was set.
    pub best_effort: bool,
    /// Breaths (gaps between words) softened.
    pub breaths_softened: usize,
    /// Seconds of silence added inside the chunk.
    pub pause_added_s: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct SpeechOutput {
    #[serde(skip)]
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub seconds: f32,
    pub provider: String,
    /// The recognizer's execution provider, when the word check ran.
    pub asr_provider: Option<String>,
    pub chunks: Vec<ChunkReport>,
    /// Every script word with its time in the final audio (empty when the
    /// word check is off). Ready for captions and for editors that cut on
    /// words.
    pub words: Vec<TimedWord>,
    /// Validated controls used for this render, if requested.
    pub direction: Option<crate::direction::DirectionPlan>,
}

/// Decide whether a generated chunk is acceptable. `Err(reason)` re-rolls it.
pub trait ChunkValidator: Send + Sync {
    fn check(&self, chunk: &Chunk, samples: &[f32]) -> Result<(), String>;
}

/// Measured signal gates (xrt-audio spike + first Rust end-to-end run,
/// 2026-09-26). A line rendered as near-silence, runaway babble, a skipped
/// passage and a repeated tail all failed at least one of these.
///
/// ⚠️ Signal gates are necessary, not sufficient: a chunk that sounds like
/// fluent speech but says the wrong words passes them. Word-level checking
/// (transcription) closes that gap.
#[derive(Debug, Clone)]
pub struct SignalValidator {
    pub min_peak: f32,
    pub min_rms: f32,
    /// Characters per second outside this range means the audio cannot be the text.
    pub min_cps: f32,
    pub max_cps: f32,
    /// Seconds of audio per TEXT TOKEN above which a chunk is rejected. Clean
    /// chunks measured 0.086–0.099 s/token; the two defective CUDA chunks
    /// (a 5.5 s hole + skipped sentences; a gibberish repeat) measured 0.167
    /// and 0.127. The character gate above missed both.
    pub max_seconds_per_token: f32,
    /// Longest allowed stretch quieter than `quiet_fraction` of the chunk's
    /// speech level (95th-percentile 20 ms frame RMS). Relative, because the
    /// model fills a lost thread with breaths loud enough to beat any absolute
    /// sample threshold: the 5.5 s hole had zero samples under 0.015 for >1.2 s.
    pub max_quiet_gap_s: f32,
    pub quiet_fraction: f32,
}

impl Default for SignalValidator {
    fn default() -> Self {
        Self {
            min_peak: 0.15,
            min_rms: 0.02,
            min_cps: 6.0,
            max_cps: 30.0,
            max_seconds_per_token: 0.118,
            // Clean chunks peaked at 0.68 s; the defective one at 3.0 s.
            max_quiet_gap_s: 1.5,
            quiet_fraction: 0.08,
        }
    }
}

/// Longest run of 20 ms frames quieter than `fraction` of the chunk's
/// 95th-percentile frame RMS, in seconds.
pub fn longest_quiet_gap(x: &[f32], rate: u32, fraction: f32) -> f32 {
    let frame = (0.02 * rate as f32) as usize;
    if frame == 0 || x.len() < frame {
        return 0.0;
    }
    let rms: Vec<f32> = x
        .chunks_exact(frame)
        .map(|f| (f.iter().map(|s| s * s).sum::<f32>() / frame as f32).sqrt())
        .collect();
    let mut sorted = rms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let reference = sorted[((sorted.len() - 1) as f32 * 0.95) as usize];
    let floor = fraction * reference;
    let (mut best, mut run) = (0usize, 0usize);
    for r in rms {
        run = if r < floor { run + 1 } else { 0 };
        best = best.max(run);
    }
    best as f32 * 0.02
}

impl ChunkValidator for SignalValidator {
    fn check(&self, chunk: &Chunk, x: &[f32]) -> Result<(), String> {
        if x.is_empty() {
            return Err("no audio".into());
        }
        let peak = x.iter().fold(0.0f32, |a, s| a.max(s.abs()));
        let rms = (x.iter().map(|s| s * s).sum::<f32>() / x.len() as f32).sqrt();
        if peak < self.min_peak || rms < self.min_rms {
            return Err(format!("silent (peak {peak:.3}, rms {rms:.4})"));
        }
        let secs = x.len() as f32 / SAMPLE_RATE as f32;
        let cps = chunk.text.chars().count() as f32 / secs;
        if cps < self.min_cps {
            return Err(format!("too long for its text ({cps:.1} chars/s)"));
        }
        if cps > self.max_cps {
            return Err(format!("too short for its text ({cps:.1} chars/s)"));
        }
        if chunk.tokens > 0 {
            let spt = secs / chunk.tokens as f32;
            if spt > self.max_seconds_per_token {
                return Err(format!(
                    "{secs:.1}s is too long for {} text tokens ({spt:.3} s/token): skipped or repeated passage",
                    chunk.tokens
                ));
            }
        }
        let gap = longest_quiet_gap(x, SAMPLE_RATE, self.quiet_fraction);
        if gap > self.max_quiet_gap_s {
            return Err(format!(
                "{gap:.1}s hole mid-chunk: the model lost its place"
            ));
        }
        Ok(())
    }
}

/// Instance-owned cache. The server owns one runtime; library callers can
/// explicitly reuse one or let the convenience functions release it on return.
#[derive(Default)]
pub struct AudioRuntime {
    model: Mutex<Option<(PathBuf, Device, Arc<ChatterboxModel>)>>,
    recognizer: Mutex<Option<(PathBuf, Device, Arc<Recognizer>)>>,
    execution: Mutex<()>,
    base: Mutex<Option<crate::whisper::WhisperModel>>,
}

impl AudioRuntime {
    pub fn unload(&self) -> Result<(), AudioError> {
        let _execution = self
            .execution
            .try_lock()
            .ok_or_else(|| AudioError::InvalidRequest("audio runtime is busy".into()))?;
        self.recognizer.lock().take();
        self.model.lock().take();
        self.base.lock().take();
        Ok(())
    }

    pub fn transcribe_base_controlled(
        &self,
        samples: &[f32],
        rate: u32,
        control: &crate::control::InferenceControl,
    ) -> Result<crate::whisper::Transcript, AudioError> {
        control.check()?;
        let _execution = self
            .execution
            .try_lock()
            .ok_or_else(|| AudioError::InvalidRequest("audio runtime is busy".into()))?;
        let mut slot = self.base.lock();
        let result = (|| {
            if slot.is_none() {
                let model = match std::env::var_os("XENO_RT_WHISPER_DIR") {
                    Some(dir) => crate::whisper::WhisperModel::load(std::path::Path::new(&dir))?,
                    None => crate::whisper::WhisperModel::load_from_registry()?,
                };
                *slot = Some(model);
            }
            control.check()?;
            slot.as_mut()
                .expect("loaded")
                .transcribe_controlled(samples, rate, control)
        })();
        if result.is_err() {
            slot.take();
        }
        result
    }

    pub fn status(&self) -> (Option<String>, Option<String>) {
        (
            self.model
                .try_lock()
                .and_then(|slot| slot.as_ref().map(|(_, _, m)| m.provider.clone())),
            self.recognizer
                .try_lock()
                .and_then(|slot| slot.as_ref().map(|(_, _, m)| m.provider.clone())),
        )
    }

    fn model(
        &self,
        dir: &std::path::Path,
        device: Device,
    ) -> Result<Arc<ChatterboxModel>, AudioError> {
        let mut slot = self.model.lock();
        if let Some((path, current_device, model)) = slot.as_ref() {
            if path == dir && *current_device == device {
                return Ok(model.clone());
            }
        }
        // Release the old model BEFORE allocating a replacement.
        slot.take();
        crate::installed::verify_managed(dir)?;
        let model = Arc::new(ChatterboxModel::load(&ModelPaths::from_dir(dir), device)?);
        *slot = Some((dir.to_path_buf(), device, model.clone()));
        Ok(model)
    }

    fn recognizer(
        &self,
        dir: &std::path::Path,
        device: Device,
    ) -> Result<Arc<Recognizer>, AudioError> {
        let mut slot = self.recognizer.lock();
        if let Some((path, current_device, model)) = slot.as_ref() {
            if path == dir && *current_device == device {
                return Ok(model.clone());
            }
        }
        slot.take();
        crate::installed::verify_managed(dir)?;
        let model = Arc::new(Recognizer::load(dir, device)?);
        *slot = Some((dir.to_path_buf(), device, model.clone()));
        Ok(model)
    }
}

/// Synthesize `script` in the voice of `reference` (mono samples at `ref_rate`).
pub fn synthesize(
    script: &str,
    reference: &[f32],
    ref_rate: u32,
    opts: &SpeechOptions,
) -> Result<SpeechOutput, AudioError> {
    synthesize_with(
        script,
        reference,
        ref_rate,
        opts,
        &SignalValidator::default(),
        |_| {},
    )
}

/// As [`synthesize`], with a custom validator and a per-chunk progress callback.
pub fn synthesize_with(
    script: &str,
    reference: &[f32],
    ref_rate: u32,
    opts: &SpeechOptions,
    validator: &dyn ChunkValidator,
    progress: impl FnMut(&ChunkReport),
) -> Result<SpeechOutput, AudioError> {
    AudioRuntime::default().synthesize_controlled(
        script,
        reference,
        ref_rate,
        opts,
        validator,
        &crate::control::InferenceControl::default(),
        progress,
    )
}

impl AudioRuntime {
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_controlled(
        &self,
        script: &str,
        reference: &[f32],
        ref_rate: u32,
        opts: &SpeechOptions,
        validator: &dyn ChunkValidator,
        control: &crate::control::InferenceControl,
        mut progress: impl FnMut(&ChunkReport),
    ) -> Result<SpeechOutput, AudioError> {
        control.check()?;
        let _execution = self
            .execution
            .try_lock()
            .ok_or_else(|| AudioError::InvalidRequest("audio runtime is busy".into()))?;
        let result = (|| {
            validate_options(opts)?;
            if script.trim().is_empty() {
                return Err(AudioError::InvalidRequest("input text is empty".into()));
            }
            let model = self.model(&opts.model_dir, opts.device)?;
            if model.variant != crate::chatterbox::ModelVariant::V3 {
                return Err(AudioError::RuntimeIncompatible(
            "speech pipeline requires Chatterbox Multilingual v3; v2 exports are not admitted"
                .into(),
        ));
            }
            let asr = if opts.word_check {
                // Same device as the voice model: its sessions fell back to CPU if
                // CUDA was unavailable, and Auto resolves the same way here.
                Some(self.recognizer(&opts.asr_dir, opts.device)?)
            } else {
                None
            };
            control.check()?;
            let chunks = chunk_script(script, &opts.language, &model.tokenizer, opts.chunk_limits)?;
            if let Some(plan) = &opts.direction {
                if !opts.word_check {
                    return Err(AudioError::InvalidRequest(
                        "direction requires word_check".into(),
                    ));
                }
                plan.validate(script, &chunks)?;
            }
            let voice: VoiceConditioning =
                model.prepare_voice_controlled(reference, ref_rate, control)?;

            let mut pieces: Vec<(Vec<f32>, f32)> = Vec::with_capacity(chunks.len());
            let mut piece_words: Vec<Vec<TimedWord>> = Vec::with_capacity(chunks.len());
            let mut reports = Vec::with_capacity(chunks.len());
            let full_direction = opts.direction.as_ref();
            let mut word_offset = 0usize;
            for (i, chunk) in chunks.iter().enumerate() {
                let chunk_word_count = chunk.text.split_whitespace().count();
                let mut directed = opts.clone();
                if let Some(plan) = &opts.direction {
                    let control = &plan.chunks[i];
                    directed.exaggeration = control.exaggeration;
                    directed.sentence_pause = control.sentence_pause;
                    directed.paragraph_pause = control.paragraph_pause;
                    let mut local = plan.clone();
                    local.pauses.retain(|p| {
                        p.after_word > word_offset && p.after_word < word_offset + chunk_word_count
                    });
                    for p in &mut local.pauses {
                        p.after_word -= word_offset;
                    }
                    directed.direction = Some(local);
                }
                let opts = &directed;
                let text = punc_norm(&chunk.text);
                let mut rejected = Vec::new();
                let mut accepted: Option<Take> = None;
                let mut best_failed: Option<(Take, f32)> = None;
                for attempt in 0..opts.max_attempts {
                    control.check()?;
                    let params = GenerateParams {
                        exaggeration: opts.exaggeration,
                        sampling: SamplingParams {
                            cfg_weight: opts.cfg_weight,
                            temperature: opts.temperature,
                            repetition_penalty: opts.repetition_penalty,
                            min_p: opts.min_p,
                            top_p: opts.top_p,
                        },
                        max_speech_tokens: opts.max_speech_tokens,
                        seed: opts
                            .seed
                            .wrapping_add(1_000_003 * i as u64)
                            .wrapping_add(7_919 * attempt as u64),
                    };
                    let g = model.generate_controlled(
                        &text,
                        &opts.language,
                        &voice,
                        &params,
                        control,
                    )?;
                    if !g.stopped {
                        rejected.push(format!(
                            "truncated at the {}-token budget",
                            opts.max_speech_tokens
                        ));
                        continue;
                    }
                    let x = crate::prosody::trim_to_speech(&g.samples, SAMPLE_RATE);
                    if let Err(why) = validator.check(chunk, &x) {
                        tracing::warn!(chunk = i, attempt = attempt + 1, %why, "chunk rejected (signal)");
                        rejected.push(why);
                        continue;
                    }
                    let mut take = Take {
                        samples: x,
                        speech_tokens: g.speech_tokens,
                        attempts: attempt + 1,
                        heard: Vec::new(),
                        check: None,
                    };
                    if let Some(asr) = &asr {
                        let heard = asr.transcribe_long_controlled(
                            &take.samples,
                            SAMPLE_RATE,
                            &opts.language,
                            control,
                        )?;
                        // Compare against the text as written (numbers and names in
                        // the caller's spelling), not the model's normalised prompt.
                        let c =
                            wordcheck::check(&chunk.text, &heard, &opts.names, opts.word_limits);
                        take.heard = heard;
                        if let Some(why) = c.rejected.clone() {
                            tracing::warn!(chunk = i, attempt = attempt + 1, %why, "chunk rejected (words)");
                            rejected.push(why);
                            let rate = c.error_rate;
                            take.check = Some(c);
                            if best_failed.as_ref().map_or(true, |(_, r)| rate < *r) {
                                best_failed = Some((take, rate));
                            }
                            continue;
                        }
                        take.check = Some(c);
                    }
                    accepted = Some(take);
                    break;
                }
                let mut best_effort = false;
                let take = match accepted {
                    Some(t) => t,
                    None if opts.accept_best_effort && best_failed.is_some() => {
                        best_effort = true;
                        best_failed.take().expect("checked").0
                    }
                    None => {
                        if rejected.iter().all(|r| r.starts_with("truncated")) {
                            return Err(AudioError::Truncated {
                                chunk: i,
                                limit: opts.max_speech_tokens,
                            });
                        }
                        return Err(AudioError::Inference(format!(
                            "chunk {i} failed {} attempts: {}",
                            opts.max_attempts,
                            rejected.join("; ")
                        )));
                    }
                };

                // Delivery shaping, only where the word times say there is no word.
                let (samples, words, breaths, added) = match &take.check {
                    Some(c) if c.problems.is_empty() && !best_effort => {
                        shape(&take.samples, &take.heard, chunk, c, opts)
                    }
                    Some(c) => (
                        take.samples.clone(),
                        wordcheck::script_word_times(&chunk.text, c, &take.heard),
                        0,
                        0.0,
                    ),
                    None => (take.samples.clone(), Vec::new(), 0, 0.0),
                };
                let report = ChunkReport {
                    index: i,
                    text: chunk.text.clone(),
                    text_tokens: chunk.tokens,
                    speech_tokens: take.speech_tokens,
                    seconds: samples.len() as f32 / SAMPLE_RATE as f32,
                    attempts: take.attempts,
                    rejected,
                    word_problems: take
                        .check
                        .as_ref()
                        .map(|c| c.problems.clone())
                        .unwrap_or_default(),
                    word_error_rate: take.check.as_ref().map(|c| c.error_rate),
                    best_effort,
                    breaths_softened: breaths,
                    pause_added_s: added,
                };
                progress(&report);
                reports.push(report);
                let mut gap = if chunk.ends_paragraph {
                    opts.paragraph_pause
                } else {
                    opts.sentence_pause
                };
                word_offset += chunk_word_count;
                if let Some(pause) = full_direction
                    .and_then(|plan| plan.pauses.iter().find(|p| p.after_word == word_offset))
                {
                    gap = gap.max(pause.seconds);
                }
                pieces.push((samples, gap));
                piece_words.push(words);
            }

            control.check()?;
            let samples = audio::stitch(&pieces, SAMPLE_RATE, 0.985);
            // Word times in the stitched audio: each piece starts where the previous
            // one ended plus its gap (stitch concatenates exactly that).
            let mut words = Vec::new();
            let mut offset = 0.0f32;
            for ((p, gap), w) in pieces.iter().zip(piece_words) {
                words.extend(w.into_iter().map(|mut w| {
                    w.start += offset;
                    w.end += offset;
                    w
                }));
                offset += p.len() as f32 / SAMPLE_RATE as f32 + gap;
            }
            Ok(SpeechOutput {
                seconds: samples.len() as f32 / SAMPLE_RATE as f32,
                samples,
                sample_rate: SAMPLE_RATE,
                provider: model.provider.clone(),
                asr_provider: asr.map(|a| a.provider.clone()),
                chunks: reports,
                words,
                direction: opts.direction.clone(),
            })
        })();
        if result.is_err() {
            // Failed/cancelled initialization and native runs cannot strand model
            // allocations. Active local handles have unwound before this point.
            self.recognizer.lock().take();
            self.model.lock().take();
        }
        result
    }
}

struct Take {
    samples: Vec<f32>,
    speech_tokens: usize,
    attempts: u32,
    heard: Vec<Word>,
    check: Option<wordcheck::WordCheck>,
}

/// Soften breaths and lengthen pauses in one accepted take, using its word
/// alignment. Returns the audio, its script words with times, the number of
/// breaths softened and the seconds of silence added.
fn shape(
    x: &[f32],
    heard: &[Word],
    chunk: &Chunk,
    check: &wordcheck::WordCheck,
    opts: &SpeechOptions,
) -> (Vec<f32>, Vec<TimedWord>, usize, f32) {
    let rate = SAMPLE_RATE;
    let spans = crate::wordsafe::protected_spans(x, rate, heard);
    let (y, breaths) =
        crate::wordsafe::soften_between_words(x, rate, &spans, opts.breath_reduction_db);
    let mut words = wordcheck::script_word_times(&chunk.text, check, heard);

    // Where each part (paragraph) of the chunk ends, in script words.
    let mut part_ends = Vec::new();
    let mut n = 0;
    for p in &chunk.parts {
        n += p.split_whitespace().count();
        part_ends.push(n);
    }
    let mut covered = 0usize;
    let mut inserts: Vec<(usize, usize)> = Vec::new();
    for k in 0..words.len().saturating_sub(1) {
        covered += words[k].text.split_whitespace().count();
        let (w, next) = (&words[k], &words[k + 1]);
        if !(w.heard && next.heard) {
            continue; // an interpolated time is not precise enough to cut at
        }
        let last = w
            .text
            .trim_end_matches(['"', '\'', '\u{201d}', '\u{2019}', ')']);
        let directed_pause = opts
            .direction
            .as_ref()
            .and_then(|plan| plan.pauses.iter().find(|p| p.after_word == covered));
        let want_min = if let Some(pause) = directed_pause {
            pause.seconds
        } else if part_ends.contains(&covered) {
            opts.paragraph_pause
        } else if last.ends_with(['.', '!', '?']) {
            opts.sentence_pause
        } else if last.ends_with([',', ';', ':']) || last.ends_with('\u{2014}') {
            0.0
        } else {
            continue; // never inside a phrase
        };
        // The gap as the protected spans see it (they reach past the
        // recognized times to the words' real edges): the one lying between
        // the middle of this word and the middle of the next. None when the
        // words' spans touch, i.e. there is no audio between them to use.
        let mid = |w: &TimedWord| ((w.start + w.end) * 0.5 * rate as f32) as usize;
        let (lo, hi) = (mid(w), mid(next));
        let Some(&(ga, gb)) = crate::wordsafe::gaps(&spans, y.len())
            .iter()
            .find(|&&(ga, gb)| ga >= lo && gb <= hi)
        else {
            continue;
        };
        let current = (next.start - w.end).max(0.0);
        let target = (current * opts.pause_scale)
            .max(want_min)
            .min(current + 2.0);
        let extra = ((target - current) * rate as f32) as usize;
        if extra == 0 {
            continue;
        }
        if let Some(at) = crate::wordsafe::insertion_point(&y, rate, ga, gb) {
            inserts.push((at, extra));
        }
    }
    let added: usize = inserts.iter().map(|i| i.1).sum();
    let y = crate::wordsafe::insert_silence(&y, rate, &inserts);
    // Shift word times past each insertion.
    let mut sorted = inserts.clone();
    sorted.sort_unstable();
    for w in words.iter_mut() {
        let shift = |t: f32| {
            let s = (t * rate as f32) as usize;
            let before: usize = sorted.iter().filter(|(at, _)| *at <= s).map(|i| i.1).sum();
            t + before as f32 / rate as f32
        };
        w.start = shift(w.start);
        w.end = shift(w.end);
    }
    (y, words, breaths, added as f32 / rate as f32)
}

pub fn validate_options(o: &SpeechOptions) -> Result<(), AudioError> {
    let bad = |m: String| Err(AudioError::InvalidRequest(m));
    if !(0.0..=2.0).contains(&o.exaggeration) {
        return bad(format!("exaggeration {} outside 0..=2", o.exaggeration));
    }
    if !(0.0..=3.0).contains(&o.cfg_weight) {
        return bad(format!("cfg_weight {} outside 0..=3", o.cfg_weight));
    }
    if !(0.0..=2.0).contains(&o.temperature) {
        return bad(format!("temperature {} outside 0..=2", o.temperature));
    }
    if !(1.0..=5.0).contains(&o.repetition_penalty) {
        return bad(format!(
            "repetition_penalty {} outside 1..=5",
            o.repetition_penalty
        ));
    }
    if !(0.0..1.0).contains(&o.min_p) || !(0.0..=1.0).contains(&o.top_p) || o.top_p == 0.0 {
        return bad("min_p must be in [0,1) and top_p in (0,1]".into());
    }
    if !(0.0..=5.0).contains(&o.sentence_pause) || !(0.0..=5.0).contains(&o.paragraph_pause) {
        return bad("pauses must be between 0 and 5 seconds".into());
    }
    if o.max_attempts == 0 || o.max_attempts > 10 {
        return bad("max_attempts must be 1..=10".into());
    }
    if o.names.len() > 128
        || o.names
            .iter()
            .any(|name| name.len() > 128 || name.trim().is_empty())
    {
        return bad("names must contain at most 128 nonempty entries of at most 128 bytes".into());
    }
    if let Device::Cuda(id) = o.device {
        if id < 0 {
            return bad("CUDA device index must be nonnegative".into());
        }
    }
    let limits = o.chunk_limits;
    if limits.min_tokens == 0
        || limits.min_tokens > limits.target_tokens
        || limits.target_tokens > limits.max_tokens
        || limits.max_tokens > 480
    {
        return bad("chunk limits must satisfy 0 < min <= target <= max <= 480".into());
    }
    if !(0.0..=1.0).contains(&o.word_limits.max_error_rate) {
        return bad("word error rate must be finite and in 0..=1".into());
    }
    let cap = crate::chatterbox::model_max_speech_tokens();
    if o.max_speech_tokens == 0 || o.max_speech_tokens > cap {
        return bad(format!("max_speech_tokens must be 1..={cap}"));
    }
    if !(0.0..=40.0).contains(&o.breath_reduction_db) {
        return bad(format!(
            "breath_reduction_db {} outside 0..=40",
            o.breath_reduction_db
        ));
    }
    if !(1.0..=3.0).contains(&o.pause_scale) {
        return bad(format!("pause_scale {} outside 1..=3", o.pause_scale));
    }
    // Shaping without word times would have to guess what is a word from
    // the sound, which is exactly what the 2026-09-27 audit found damaging.
    if !o.word_check && (o.pause_scale != 1.0 || o.breath_reduction_db > 0.0) {
        return bad(
            "pause_scale and breath_reduction_db need the word check (word_check: true)".into(),
        );
    }
    crate::tokenizer::ChatterboxTokenizer::check_language(&o.language)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(tokens: usize, chars: usize) -> Chunk {
        Chunk::new("x".repeat(chars), tokens, false)
    }

    /// Speech-like signal: bursts at full level separated by short pauses.
    fn speech(seconds: f32, hole_at: Option<(f32, f32)>) -> Vec<f32> {
        let n = (seconds * SAMPLE_RATE as f32) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / SAMPLE_RATE as f32;
                if let Some((a, b)) = hole_at {
                    if t >= a && t < b {
                        // breath-level noise: above 0.015 but far below speech
                        return 0.02 * ((i as f32 * 1.7).sin());
                    }
                }
                let syllable = (t * 4.0).fract() < 0.8;
                if syllable {
                    0.5 * (t * 2.0 * std::f32::consts::PI * 180.0).sin()
                } else {
                    0.0
                }
            })
            .collect()
    }

    #[test]
    fn paragraph_cadence_is_applied_in_the_composed_shaper() {
        let rate = SAMPLE_RATE as usize;
        let mut x = vec![0.3; rate];
        x.extend(vec![0.0; rate / 2]);
        x.extend(vec![0.3; rate]);
        let heard = vec![
            Word {
                text: "First.".into(),
                start: 0.0,
                end: 1.0,
                probability: 1.0,
            },
            Word {
                text: "Second.".into(),
                start: 1.5,
                end: 2.5,
                probability: 1.0,
            },
        ];
        let chunk = Chunk {
            text: "First. Second.".into(),
            tokens: 10,
            ends_paragraph: true,
            parts: vec!["First.".into(), "Second.".into()],
        };
        let check = wordcheck::check(&chunk.text, &heard, &[], WordCheckLimits::default());
        let opts = SpeechOptions {
            paragraph_pause: 1.0,
            breath_reduction_db: 0.0,
            ..Default::default()
        };
        let (out, words, _, added) = shape(&x, &heard, &chunk, &check, &opts);
        assert!((added - 0.5).abs() < 0.001);
        assert_eq!(out.len(), x.len() + rate / 2);
        assert_eq!(&out[..rate], &x[..rate]);
        assert_eq!(&out[2 * rate..], &x[rate + rate / 2..]);
        assert!((words[1].start - 2.0).abs() < 0.001);
    }

    #[test]
    fn invalid_options_fail_before_loading_models() {
        let mut opts = SpeechOptions {
            temperature: f32::NAN,
            ..Default::default()
        };
        assert!(validate_options(&opts).is_err());
        opts.temperature = 0.8;
        opts.device = Device::Cuda(-1);
        assert!(validate_options(&opts).is_err());
        opts.device = Device::Cpu;
        opts.chunk_limits.max_tokens = 900;
        assert!(validate_options(&opts).is_err());
    }

    #[test]
    fn accepts_a_clean_chunk() {
        // 328 tokens / 520 chars in 28.2 s — the real clean chunk 0 on CPU.
        assert_eq!(
            SignalValidator::default().check(&chunk(328, 520), &speech(28.2, None)),
            Ok(())
        );
    }

    #[test]
    fn rejects_the_skipped_passage_shape() {
        // Real defective CUDA chunk 0: 328 tokens in 54.9 s with a 5.5 s breathy hole.
        let r =
            SignalValidator::default().check(&chunk(328, 520), &speech(54.9, Some((18.3, 23.8))));
        assert!(r.is_err(), "{r:?}");
    }

    #[test]
    fn rejects_a_breathy_hole_even_at_normal_length() {
        let x = speech(28.0, Some((10.0, 13.0)));
        let r = SignalValidator::default().check(&chunk(328, 520), &x);
        assert!(r.as_ref().is_err_and(|e| e.contains("hole")), "{r:?}");
    }

    #[test]
    fn rejects_the_repeated_tail_shape() {
        // Real defective CUDA chunk 3: 316 tokens in 40.2 s, no hole.
        let r = SignalValidator::default().check(&chunk(316, 500), &speech(40.2, None));
        assert!(r.as_ref().is_err_and(|e| e.contains("s/token")), "{r:?}");
    }
}
