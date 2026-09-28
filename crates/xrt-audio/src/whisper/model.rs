use std::borrow::Cow;
use std::path::{Path, PathBuf};

use ndarray::{Array2, Array3};
use ort::session::{Session, SessionInputValue};
use ort::value::{DynValue, Tensor};
use serde::Serialize;

use super::mel::{self, N_FRAMES, N_MELS, SAMPLE_RATE};
use super::tokenizer::WhisperTokenizer;
use crate::chatterbox::{CudaPrecision, Device};
use crate::AudioError;

const LAYERS: usize = 12;
/// One encoder frame is 20 ms (1500 frames per 30 s window).
const FRAME_S: f32 = 0.02;
const MAX_TOKENS: usize = 220;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Word {
    pub text: String,
    pub start: f32,
    pub end: f32,
    /// Mean token probability.
    pub probability: f32,
}

pub struct Recognizer {
    encoder: Session,
    decoder: Session,
    decoder_past: Session,
    tokenizer: WhisperTokenizer,
    /// (layer, head) pairs whose cross-attention tracks the audio position.
    alignment_heads: Vec<(usize, usize)>,
    suppress: Vec<i64>,
    sot: i64,
    eot: i64,
    no_timestamps: i64,
    transcribe: i64,
    pub provider: String,
}

impl Recognizer {
    /// `$XRT_AUDIO_ASR_DIR`, else `~/.xeno/models/whisper-small-timestamped`.
    pub fn default_dir() -> PathBuf {
        if let Some(d) = std::env::var_os("XRT_AUDIO_ASR_DIR") {
            return PathBuf::from(d);
        }
        let home = std::env::var_os("USERPROFILE")
            .or_else(|| std::env::var_os("HOME"))
            .unwrap_or_default();
        PathBuf::from(home)
            .join(".xeno")
            .join("models")
            .join("whisper-small-timestamped")
    }

    pub fn load(dir: &Path, device: Device) -> Result<Self, AudioError> {
        let o = dir.join("onnx");
        let files = [
            o.join("encoder_model.onnx"),
            o.join("decoder_model.onnx"),
            o.join("decoder_with_past_model.onnx"),
            dir.join("tokenizer.json"),
            dir.join("generation_config.json"),
        ];
        for p in &files {
            if !p.is_file() {
                return Err(AudioError::ModelMissing {
                    path: p.display().to_string(),
                    message: "Whisper timestamped ONNX export expected (onnx-community/whisper-small_timestamped)".into(),
                });
            }
        }
        crate::chatterbox::check_ort_version()?;
        let (encoder, provider) =
            crate::chatterbox::session(&files[0], device, CudaPrecision::Fp32)?;
        let device = if provider.starts_with("cuda") {
            device
        } else {
            Device::Cpu
        };
        let (decoder, _) = crate::chatterbox::session(&files[1], device, CudaPrecision::Fp32)?;
        let (decoder_past, _) = crate::chatterbox::session(&files[2], device, CudaPrecision::Fp32)?;
        let tokenizer = WhisperTokenizer::from_file(&files[3])?;

        let gen: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&files[4])
                .map_err(|e| AudioError::Tokenizer(e.to_string()))?,
        )
        .map_err(|e| AudioError::Tokenizer(e.to_string()))?;
        let alignment_heads = gen["alignment_heads"]
            .as_array()
            .ok_or_else(|| {
                AudioError::Tokenizer("generation_config has no alignment_heads".into())
            })?
            .iter()
            .filter_map(|p| Some((p[0].as_u64()? as usize, p[1].as_u64()? as usize)))
            .collect::<Vec<_>>();
        if alignment_heads.is_empty()
            || alignment_heads
                .iter()
                .any(|&(layer, head)| layer >= LAYERS || head >= 12)
        {
            return Err(AudioError::Tokenizer(
                "invalid or empty Whisper alignment heads".into(),
            ));
        }
        let mut suppress: Vec<i64> = gen["suppress_tokens"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|v| v.as_i64())
            .collect();
        suppress.sort_unstable();
        let no_timestamps = tokenizer.special("<|notimestamps|>")?;
        Ok(Self {
            sot: tokenizer.special("<|startoftranscript|>")?,
            eot: tokenizer.special("<|endoftext|>")?,
            transcribe: tokenizer.special("<|transcribe|>")?,
            no_timestamps,
            encoder,
            decoder,
            decoder_past,
            tokenizer,
            alignment_heads,
            suppress,
            provider,
        })
    }

    /// Transcribe audio of any length. Whisper sees 30 s at a time, so longer
    /// audio is cut at the quietest 10 ms between 20 and 28 s into each
    /// window and the windows' words are joined with their times offset.
    /// The quietest point is only a boundary estimate, not proof of silence;
    /// transcript comparison must still reject missing words across the cut.
    pub fn transcribe_long(
        &self,
        audio: &[f32],
        rate: u32,
        language: &str,
    ) -> Result<Vec<Word>, AudioError> {
        validate_audio(audio, rate)?;
        let window = (28.0 * rate as f32) as usize;
        let frame = ((0.01 * rate as f32) as usize).max(1);
        let mut words = Vec::new();
        let mut start = 0usize;
        while start < audio.len() {
            let rest = audio.len() - start;
            let end = if rest as f32 <= 29.5 * rate as f32 {
                audio.len()
            } else {
                let lo = start + (20.0 * rate as f32) as usize;
                let hi = start + window;
                let mut best = (f32::INFINITY, hi);
                let mut k = lo;
                while k + frame <= hi {
                    let e: f32 = audio[k..k + frame].iter().map(|v| v * v).sum();
                    if e < best.0 {
                        best = (e, k + frame / 2);
                    }
                    k += frame;
                }
                best.1
            };
            let offset = start as f32 / rate as f32;
            for mut w in self.transcribe(&audio[start..end], rate, language)? {
                w.start += offset;
                w.end += offset;
                words.push(w);
            }
            start = end;
        }
        Ok(words)
    }

    /// Transcribe up to 30 s of mono audio and return words with times.
    /// `language` is an ISO code such as "en".
    pub fn transcribe(
        &self,
        audio: &[f32],
        rate: u32,
        language: &str,
    ) -> Result<Vec<Word>, AudioError> {
        validate_audio(audio, rate)?;
        let x = crate::audio::resample(audio, rate, SAMPLE_RATE);
        let seconds = x.len() as f32 / SAMPLE_RATE as f32;
        if seconds > 30.0 {
            return Err(AudioError::InvalidRequest(format!(
                "recognizer window is 30 s; got {seconds:.1} s"
            )));
        }
        let lang = self
            .tokenizer
            .special(&format!("<|{}|>", language.to_ascii_lowercase()))?;
        let feats = mel::log_mel(&x);
        let enc = self.encoder.run(ort::inputs![
            "input_features" => Tensor::from_array(Array3::from_shape_vec((1, N_MELS, N_FRAMES), feats).expect("shape"))?
        ]?)?;
        let (eshape, edata) = enc["last_hidden_state"].try_extract_raw_tensor::<f32>()?;
        let hidden = Array3::from_shape_vec(
            (eshape[0] as usize, eshape[1] as usize, eshape[2] as usize),
            edata.to_vec(),
        )
        .expect("shape");
        drop(enc);

        // ---- greedy decode, text only ---------------------------------------
        let prompt = vec![self.sot, lang, self.transcribe, self.no_timestamps];
        let mut out = self.decoder.run(ort::inputs![
            "input_ids" => Tensor::from_array(Array2::from_shape_vec((1, prompt.len()), prompt.clone()).expect("shape"))?,
            "encoder_hidden_states" => Tensor::from_array(hidden.clone())?
        ]?)?;
        let mut tokens: Vec<i64> = Vec::new();
        let mut probs: Vec<f32> = Vec::new();
        // Encoder K/V come from the first pass and are reused unchanged;
        // decoder K/V are replaced every step.
        let mut enc_kv: Vec<DynValue> = Vec::with_capacity(2 * LAYERS);
        let mut dec_kv: Vec<DynValue> = Vec::with_capacity(2 * LAYERS);
        let mut first = true;
        loop {
            let (lshape, logits) = out["logits"].try_extract_raw_tensor::<f32>()?;
            let (seq, vocab) = (lshape[1] as usize, lshape[2] as usize);
            let mut last = logits[(seq - 1) * vocab..seq * vocab].to_vec();
            // Suppress configured tokens, every timestamp/special token, and
            // on the first step the blank and end-of-text tokens.
            for &t in &self.suppress {
                if (t as usize) < vocab {
                    last[t as usize] = f32::NEG_INFINITY;
                }
            }
            for v in last.iter_mut().skip((self.sot as usize).min(vocab)) {
                *v = f32::NEG_INFINITY;
            }
            if first {
                last[220] = f32::NEG_INFINITY;
                last[self.eot as usize] = f32::NEG_INFINITY;
            }
            let (best, p) = argmax_softmax(&last);

            dec_kv.clear();
            for l in 0..LAYERS {
                for part in ["decoder.key", "decoder.value"] {
                    let name = format!("present.{l}.{part}");
                    dec_kv.push(out.remove(name.as_str()).ok_or_else(|| {
                        AudioError::Inference(format!("decoder did not return `{name}`"))
                    })?);
                }
                if first {
                    for part in ["encoder.key", "encoder.value"] {
                        let name = format!("present.{l}.{part}");
                        enc_kv.push(out.remove(name.as_str()).ok_or_else(|| {
                            AudioError::Inference(format!("decoder did not return `{name}`"))
                        })?);
                    }
                }
            }
            drop(out);

            if best == self.eot {
                break;
            }
            if tokens.len() >= MAX_TOKENS {
                return Err(AudioError::Inference(
                    "Whisper token budget exhausted before end-of-text; transcript is incomplete"
                        .into(),
                ));
            }
            tokens.push(best);
            probs.push(p);

            let mut inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> =
                Vec::with_capacity(1 + 4 * LAYERS);
            inputs.push((
                "input_ids".into(),
                Tensor::from_array(Array2::from_elem((1, 1), best))?
                    .into_dyn()
                    .into(),
            ));
            let mut dk = dec_kv.drain(..);
            for l in 0..LAYERS {
                inputs.push((
                    format!("past_key_values.{l}.decoder.key").into(),
                    dk.next().expect("dk").into(),
                ));
                inputs.push((
                    format!("past_key_values.{l}.decoder.value").into(),
                    dk.next().expect("dv").into(),
                ));
                inputs.push((
                    format!("past_key_values.{l}.encoder.key").into(),
                    enc_kv[2 * l].view().into(),
                ));
                inputs.push((
                    format!("past_key_values.{l}.encoder.value").into(),
                    enc_kv[2 * l + 1].view().into(),
                ));
            }
            drop(dk);
            out = self.decoder_past.run(inputs)?;
            first = false;
        }
        drop(enc_kv);

        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        // ---- word timings: teacher-forced pass with cross-attention --------
        let mut forced = prompt.clone();
        forced.extend(&tokens);
        forced.push(self.eot);
        let out = self.decoder.run(ort::inputs![
            "input_ids" => Tensor::from_array(Array2::from_shape_vec((1, forced.len()), forced.clone()).expect("shape"))?,
            "encoder_hidden_states" => Tensor::from_array(hidden)?
        ]?)?;
        // OpenAI `find_alignment`, step for step:
        //  1. alignment-head cross-attention over the frames that hold audio
        //     (the export already applies softmax over frames),
        //  2. normalise each frame across ALL rows (torch.std_mean(dim=-2)),
        //  3. width-7 median filter along time,
        //  4. mean over heads, drop the SOT prompt rows and the EOT row,
        //  5. DTW on the negated matrix,
        //  6. a token's time is the frame where the path first REACHES it.
        let n_frames = ((seconds / FRAME_S).ceil() as usize).clamp(1, 1500);
        // Position p's attention predicts token p+1, so the row for the last
        // prompt token (<|notimestamps|>) holds the FIRST text token's timing,
        // and the last text token's row holds EOT's (= end of the last word).
        // OpenAI slices `matrix[len(sot_sequence):-1]` for exactly this; an
        // earlier version started one row late and put every word one token
        // behind (start == reference end, measured to the millisecond).
        let text0 = prompt.len() - 1;
        let n_text = tokens.len() + 1;
        let n_rows = forced.len();
        let mut matrix = vec![vec![0.0f32; n_frames]; n_text];
        for &(layer, head) in &self.alignment_heads {
            let (s, a) = out[format!("cross_attentions.{layer}").as_str()]
                .try_extract_raw_tensor::<f32>()?;
            let (heads, q, k) = (s[1] as usize, s[2] as usize, s[3] as usize);
            if head >= heads || q != n_rows || k < n_frames {
                return Err(AudioError::Inference(format!(
                    "unexpected cross-attention shape {s:?}"
                )));
            }
            let mut m: Vec<Vec<f32>> = (0..n_rows)
                .map(|r| {
                    let base = (head * q + r) * k;
                    a[base..base + n_frames].to_vec()
                })
                .collect();
            for f in 0..n_frames {
                let mean = m.iter().map(|r| r[f]).sum::<f32>() / n_rows as f32;
                let sd = (m.iter().map(|r| (r[f] - mean).powi(2)).sum::<f32>() / n_rows as f32)
                    .sqrt()
                    .max(1e-8);
                for r in m.iter_mut() {
                    r[f] = (r[f] - mean) / sd;
                }
            }
            for (i, row) in matrix.iter_mut().enumerate() {
                let filtered = median7(&m[text0 + i]);
                for (acc, v) in row.iter_mut().zip(filtered) {
                    *acc += v / self.alignment_heads.len() as f32;
                }
            }
        }
        let path = dtw(&matrix);
        // Frame at which the path first enters each row ("jumps"): row k is
        // the start of text token k; row `tokens.len()` is the end of the last.
        let mut token_start = vec![seconds; n_text];
        let mut prev_row = usize::MAX;
        for &(i, j) in &path {
            if i != prev_row {
                token_start[i] = j as f32 * FRAME_S;
                prev_row = i;
            }
        }

        // ---- group tokens into words (a word starts at a leading space) ----
        // A word ends where its first TRAILING punctuation token starts, not
        // at the next word: decoding without timestamp tokens, the "," / "."
        // row absorbs the pause that follows it, so ending at the next word
        // would put the whole pause inside the word (measured: "beast,"
        // ending 600 ms late) and word-safe processing could never reach it.
        let is_punct = |t: i64| {
            let d = self.tokenizer.decode(&[t]);
            let d = d.trim();
            !d.is_empty()
                && d.chars().all(|c| {
                    c.is_ascii_punctuation()
                        || "\u{2014}\u{2013}\u{2026}\u{2019}\u{201d}".contains(c)
                })
        };
        let mut words: Vec<Word> = Vec::new();
        let mut begin = 0usize;
        for i in 1..=tokens.len() {
            let boundary =
                i == tokens.len() || self.tokenizer.piece(tokens[i]).starts_with('\u{120}');
            if !boundary {
                continue;
            }
            let span = &tokens[begin..i];
            let mut core = span.len();
            while core > 1 && is_punct(span[core - 1]) {
                core -= 1;
            }
            let start = token_start[begin];
            words.push(Word {
                text: self.tokenizer.decode(span).trim().to_string(),
                start,
                end: token_start[begin + core].min(seconds).max(start),
                probability: probs[begin..i].iter().sum::<f32>() / (i - begin) as f32,
            });
            begin = i;
        }
        words.retain(|w| !w.text.is_empty());
        Ok(words)
    }
}

fn validate_audio(audio: &[f32], rate: u32) -> Result<(), AudioError> {
    if !(8_000..=192_000).contains(&rate)
        || audio.is_empty()
        || audio.iter().any(|s| !s.is_finite() || s.abs() > 1000.0)
    {
        return Err(AudioError::InvalidRequest(
            "recognition requires finite nonempty audio at 8000..=192000 Hz".into(),
        ));
    }
    Ok(())
}

fn argmax_softmax(l: &[f32]) -> (i64, f32) {
    let (mut bi, mut bv) = (0usize, f32::NEG_INFINITY);
    for (i, &v) in l.iter().enumerate() {
        if v > bv {
            bv = v;
            bi = i;
        }
    }
    let z: f32 = l
        .iter()
        .map(|v| if v.is_finite() { (v - bv).exp() } else { 0.0 })
        .sum();
    (bi as i64, 1.0 / z)
}

fn median7(x: &[f32]) -> Vec<f32> {
    let n = x.len();
    (0..n)
        .map(|i| {
            let mut w: Vec<f32> = (0..7)
                .map(|k| {
                    let j = i as isize + k as isize - 3;
                    // reflect padding, as scipy/OpenAI
                    let j = if j < 0 {
                        -j
                    } else if j as usize >= n {
                        2 * (n as isize - 1) - j
                    } else {
                        j
                    };
                    x[j.clamp(0, n as isize - 1) as usize]
                })
                .collect();
            w.sort_by(|a, b| a.total_cmp(b));
            w[3]
        })
        .collect()
}

/// Monotonic DTW over `-weights` (tokens × frames). Returns the path as
/// (token, frame) pairs from start to end.
fn dtw(w: &[Vec<f32>]) -> Vec<(usize, usize)> {
    let (n, m) = (w.len(), w[0].len());
    let inf = f32::INFINITY;
    let mut cost = vec![vec![inf; m + 1]; n + 1];
    let mut trace = vec![vec![0u8; m + 1]; n + 1];
    cost[0][0] = 0.0;
    for i in 1..=n {
        for j in 1..=m {
            let (c0, c1, c2) = (cost[i - 1][j - 1], cost[i - 1][j], cost[i][j - 1]);
            let (c, t) = if c0 <= c1 && c0 <= c2 {
                (c0, 0)
            } else if c1 <= c2 {
                (c1, 1)
            } else {
                (c2, 2)
            };
            cost[i][j] = -w[i - 1][j - 1] + c;
            trace[i][j] = t;
        }
    }
    let (mut i, mut j) = (n, m);
    let mut path = Vec::new();
    while i > 0 && j > 0 {
        path.push((i - 1, j - 1));
        match trace[i][j] {
            0 => {
                i -= 1;
                j -= 1;
            }
            1 => i -= 1,
            _ => j -= 1,
        }
    }
    path.reverse();
    path
}
