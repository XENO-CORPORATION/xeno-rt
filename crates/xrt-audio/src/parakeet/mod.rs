//! Parakeet TDT transcription over ONNX Runtime, for dictation.
//!
//! Three graphs: an encoder over the whole clip, a prediction network holding
//! the text so far, and a joiner that scores one encoder frame against that
//! text. The joiner's output is two heads glued together: the vocabulary, then
//! five duration bins saying how many frames to skip. Following those skips is
//! what makes TDT fast: one transducer step per spoken token, not per frame.
//!
//! Language is the model's business. The multilingual model recognises and
//! transcribes in one pass, with no language token to choose, so there is no
//! Whisper-style detection step to run and nothing for a caller to name.

pub mod features;

use std::{borrow::Cow, path::Path};

use ndarray::{Array1, Array2, Array3, Ix3};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::{DynValue, Tensor, ValueType},
};

use crate::{
    control::InferenceControl,
    whisper::{Segment, Transcript},
    AudioError,
};

type Result<T> = std::result::Result<T, AudioError>;

/// Output heads after the vocabulary: skip 0, 1, 2, 3 or 4 frames.
const DURATION_BINS: usize = 5;
/// A frame may emit this many tokens before the decoder is made to move on.
/// Without it a model stuck on one frame never reaches the end of the clip.
const MAX_TOKENS_PER_FRAME: usize = 5;
/// Bursts of work on a key press, so more threads only add contention.
const MAX_THREADS: usize = 4;
/// Encoder frames per second: 10 ms features subsampled 8 times.
const SUBSAMPLING: usize = 8;

/// The published dictation models.
pub const PARAKEET_MODELS: &[ParakeetVariant] = &[
    ParakeetVariant {
        id: "parakeet-tdt-0.6b-v3",
        set: &[
            xrt_hub::SetMember {
                registry_id: "parakeet-tdt-0.6b-v3-encoder",
                local_name: "encoder.int8.onnx",
            },
            xrt_hub::SetMember {
                registry_id: "parakeet-tdt-0.6b-v3-decoder",
                local_name: "decoder.int8.onnx",
            },
            xrt_hub::SetMember {
                registry_id: "parakeet-tdt-0.6b-v3-joiner",
                local_name: "joiner.int8.onnx",
            },
            xrt_hub::SetMember {
                registry_id: "parakeet-tdt-0.6b-v3-tokens",
                local_name: "tokens.txt",
            },
        ],
    },
    ParakeetVariant {
        id: "parakeet-tdt-110m-en",
        set: &[
            xrt_hub::SetMember {
                registry_id: "parakeet-tdt-110m-en-encoder",
                local_name: "encoder.int8.onnx",
            },
            xrt_hub::SetMember {
                registry_id: "parakeet-tdt-110m-en-decoder",
                local_name: "decoder.int8.onnx",
            },
            xrt_hub::SetMember {
                registry_id: "parakeet-tdt-110m-en-joiner",
                local_name: "joiner.int8.onnx",
            },
            xrt_hub::SetMember {
                registry_id: "parakeet-tdt-110m-en-tokens",
                local_name: "tokens.txt",
            },
        ],
    },
];

pub struct ParakeetVariant {
    pub id: &'static str,
    pub set: &'static [xrt_hub::SetMember],
}

impl ParakeetVariant {
    /// Whether this model can run without touching the network, which is what
    /// decides whether a person has to be asked about a download first.
    pub fn installed(&self) -> bool {
        match std::env::var_os("XRT_AUDIO_PARAKEET_DIR") {
            Some(dir) => Path::new(&dir).join(self.id).join("tokens.txt").is_file(),
            None => xrt_hub::ModelHub::new()
                .ok()
                .and_then(|hub| hub.xeno_model_set_if_installed(self.id))
                .is_some(),
        }
    }
}

pub fn variant(id: &str) -> Option<&'static ParakeetVariant> {
    PARAKEET_MODELS.iter().find(|model| model.id == id)
}

pub struct ParakeetModel {
    encoder: Session,
    decoder: Session,
    joiner: Session,
    pieces: Vec<String>,
    n_mels: usize,
    state_shape: (usize, usize, usize),
}

/// The prediction network's output and the recurrent state that produced it.
struct Prediction {
    output: Array3<f32>,
    state: [Array3<f32>; 2],
}

impl ParakeetModel {
    pub fn load(dir: &Path) -> Result<Self> {
        crate::chatterbox::check_ort_version()?;
        let tokens_path = dir.join("tokens.txt");
        let tokens =
            std::fs::read_to_string(&tokens_path).map_err(|_| AudioError::ModelMissing {
                path: tokens_path.display().to_string(),
                message: "Parakeet tokens.txt is required".into(),
            })?;
        let pieces = parse_tokens(&tokens)?;

        let encoder = build_session(&dir.join("encoder.int8.onnx"))?;
        let decoder = build_session(&dir.join("decoder.int8.onnx"))?;
        let joiner = build_session(&dir.join("joiner.int8.onnx"))?;

        let n_mels = input_dims(&encoder, 0)?
            .get(1)
            .copied()
            .filter(|&bands| bands == 80 || bands == 128)
            .ok_or_else(|| AudioError::Inference("unexpected Parakeet mel size".into()))?
            as usize;
        let state = input_dims(&decoder, 2)?;
        let [layers, _, hidden] = state[..] else {
            return Err(AudioError::Inference(
                "unexpected Parakeet state rank".into(),
            ));
        };
        if decoder.inputs.len() != 4
            || decoder.outputs.len() != 4
            || !(1..=8).contains(&layers)
            || !(1..=4096).contains(&hidden)
        {
            return Err(AudioError::Inference(
                "unexpected Parakeet prediction network".into(),
            ));
        }
        Ok(Self {
            encoder,
            decoder,
            joiner,
            pieces,
            n_mels,
            state_shape: (layers as usize, 1, hidden as usize),
        })
    }

    /// Fetches the set from the XENO model registry if it is not cached. Only
    /// reached once the person has agreed to the download.
    pub fn load_from_registry(model: &ParakeetVariant) -> Result<Self> {
        let hub = xrt_hub::ModelHub::new()
            .map_err(|e| AudioError::Inference(format!("model cache unavailable: {e}")))?;
        let installed = hub
            .install_xeno_model_set(model.id, model.set)
            .map_err(|e| AudioError::ModelMissing {
                path: model.id.into(),
                message: e.to_string(),
            })?;
        Self::load(&installed.path)
    }

    pub fn transcribe_controlled(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        control: &InferenceControl,
    ) -> Result<Transcript> {
        control.check()?;
        if !(8000..=192_000).contains(&sample_rate) || samples.iter().any(|s| !s.is_finite()) {
            return Err(AudioError::InvalidRequest(
                "invalid transcription audio".into(),
            ));
        }
        let audio = crate::audio::resample(samples, sample_rate, features::SAMPLE_RATE);
        let mel = features::log_mel(&audio, self.n_mels);
        let duration = audio.len() as f32 / features::SAMPLE_RATE as f32;
        if mel.frames == 0 {
            return Ok(Transcript {
                text: String::new(),
                segments: Vec::new(),
                language: None,
            });
        }
        let (encoded, frames) = self.encode(mel, control)?;
        let tokens = self.decode(&encoded, frames, control)?;
        let text = detokenize(&self.pieces, &tokens);
        let segments = if text.is_empty() {
            Vec::new()
        } else {
            vec![Segment {
                start: 0.0,
                end: duration,
                text: text.clone(),
            }]
        };
        Ok(Transcript {
            text,
            segments,
            language: None,
        })
    }

    fn encode(
        &self,
        mel: features::Features,
        control: &InferenceControl,
    ) -> Result<(Array3<f32>, usize)> {
        let input = Array3::from_shape_vec((1, self.n_mels, mel.frames), mel.values)
            .map_err(|e| AudioError::Inference(e.to_string()))?;
        let inputs: Vec<(Cow<'static, str>, DynValue)> = vec![
            (self.encoder.inputs[0].name.clone().into(), tensor(input)?),
            (
                self.encoder.inputs[1].name.clone().into(),
                tensor(Array1::from_vec(vec![mel.frames as i64]))?,
            ),
        ];
        let outputs = control.run(&self.encoder, inputs)?;
        let encoded = owned3(&outputs[self.encoder.outputs[0].name.as_str()], "encoder")?;
        // The graph reports its own length; a shorter value marks padding frames.
        let length = outputs[self.encoder.outputs[1].name.as_str()]
            .try_extract_tensor::<i64>()
            .map_err(|e| AudioError::Inference(format!("encoded length: {e}")))?
            .iter()
            .next()
            .copied()
            .unwrap_or(0);
        let frames = usize::try_from(length).unwrap_or(0).min(encoded.shape()[2]);
        let expected = mel.frames.div_ceil(SUBSAMPLING) + 1;
        if frames > expected {
            return Err(AudioError::Inference(
                "encoder returned more frames than the audio holds".into(),
            ));
        }
        Ok((encoded, frames))
    }

    /// Greedy TDT. A blank with a skip of zero still moves one frame, so the
    /// loop always reaches the end of the clip.
    fn decode(
        &self,
        encoded: &Array3<f32>,
        frames: usize,
        control: &InferenceControl,
    ) -> Result<Vec<usize>> {
        let vocabulary = self.pieces.len();
        let blank = vocabulary - 1;
        let (layers, batch, hidden) = self.state_shape;
        let empty = Array3::<f32>::zeros((layers, batch, hidden));
        let mut prediction = self.predict(blank, [empty.clone(), empty], control)?;

        let mut tokens = Vec::new();
        let mut frame = 0;
        let mut emitted_here = 0;
        while frame < frames {
            let scores = self.join(encoded, frame, &prediction.output, control)?;
            if scores.len() != vocabulary + DURATION_BINS {
                return Err(AudioError::Inference(
                    "joiner output does not match the vocabulary".into(),
                ));
            }
            let (token_scores, duration_scores) = scores.split_at(vocabulary);
            let token = argmax(token_scores);
            let mut skip = argmax(duration_scores);
            if token != blank {
                tokens.push(token);
                emitted_here += 1;
                prediction = self.predict(token, prediction.state, control)?;
            }
            if skip == 0 && (token == blank || emitted_here >= MAX_TOKENS_PER_FRAME) {
                skip = 1;
            }
            if skip > 0 {
                emitted_here = 0;
            }
            frame += skip;
        }
        Ok(tokens)
    }

    fn predict(
        &self,
        token: usize,
        state: [Array3<f32>; 2],
        control: &InferenceControl,
    ) -> Result<Prediction> {
        let [first, second] = state;
        let inputs: Vec<(Cow<'static, str>, DynValue)> = vec![
            (
                self.decoder.inputs[0].name.clone().into(),
                tensor(Array2::from_elem((1, 1), token as i32))?,
            ),
            (
                self.decoder.inputs[1].name.clone().into(),
                tensor(Array1::from_vec(vec![1i32]))?,
            ),
            (self.decoder.inputs[2].name.clone().into(), tensor(first)?),
            (self.decoder.inputs[3].name.clone().into(), tensor(second)?),
        ];
        let outputs = control.run(&self.decoder, inputs)?;
        Ok(Prediction {
            output: owned3(
                &outputs[self.decoder.outputs[0].name.as_str()],
                "prediction",
            )?,
            state: [
                owned3(&outputs[self.decoder.outputs[2].name.as_str()], "state")?,
                owned3(&outputs[self.decoder.outputs[3].name.as_str()], "state")?,
            ],
        })
    }

    fn join(
        &self,
        encoded: &Array3<f32>,
        frame: usize,
        prediction: &Array3<f32>,
        control: &InferenceControl,
    ) -> Result<Vec<f32>> {
        let depth = encoded.shape()[1];
        let column = encoded.slice(ndarray::s![0, .., frame]).to_owned();
        let column = column
            .into_shape_with_order((1, depth, 1))
            .map_err(|e| AudioError::Inference(e.to_string()))?;
        let inputs: Vec<(Cow<'static, str>, DynValue)> = vec![
            (self.joiner.inputs[0].name.clone().into(), tensor(column)?),
            (
                self.joiner.inputs[1].name.clone().into(),
                tensor(prediction.clone())?,
            ),
        ];
        let outputs = control.run(&self.joiner, inputs)?;
        let scores = outputs[self.joiner.outputs[0].name.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|e| AudioError::Inference(format!("joiner: {e}")))?;
        Ok(scores.iter().copied().collect())
    }
}

fn build_session(path: &Path) -> Result<Session> {
    if !path.exists() {
        return Err(AudioError::ModelMissing {
            path: path.display().to_string(),
            message: "Parakeet ONNX graph is required".into(),
        });
    }
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(MAX_THREADS);
    Session::builder()
        .and_then(|b| b.with_optimization_level(GraphOptimizationLevel::Level3))
        .and_then(|b| b.with_intra_threads(threads))
        .and_then(|b| b.commit_from_file(path))
        .map_err(|e| AudioError::Inference(format!("failed to open {}: {e}", path.display())))
}

fn input_dims(session: &Session, index: usize) -> Result<Vec<i64>> {
    match session.inputs.get(index).map(|input| &input.input_type) {
        Some(ValueType::Tensor { dimensions, .. }) => Ok(dimensions.clone()),
        _ => Err(AudioError::Inference(
            "unexpected Parakeet graph input".into(),
        )),
    }
}

fn tensor<T, D>(array: ndarray::Array<T, D>) -> Result<DynValue>
where
    T: ort::tensor::PrimitiveTensorElementType + Clone + std::fmt::Debug + 'static,
    D: ndarray::Dimension + 'static,
{
    Tensor::from_array(array)
        .map(|value| value.into_dyn())
        .map_err(|e| AudioError::Inference(format!("input tensor: {e}")))
}

fn owned3(value: &DynValue, what: &str) -> Result<Array3<f32>> {
    Ok(value
        .try_extract_tensor::<f32>()
        .map_err(|e| AudioError::Inference(format!("{what}: {e}")))?
        .into_dimensionality::<Ix3>()
        .map_err(|e| AudioError::Inference(format!("{what} rank: {e}")))?
        .to_owned())
}

fn argmax(scores: &[f32]) -> usize {
    let mut best = 0;
    for (index, score) in scores.iter().enumerate() {
        if *score > scores[best] {
            best = index;
        }
    }
    best
}

/// `tokens.txt` holds `<piece> <id>` per line, ids in order, blank last.
fn parse_tokens(text: &str) -> Result<Vec<String>> {
    let mut pieces = Vec::new();
    for (expected, line) in text.lines().enumerate() {
        let parsed = line
            .rsplit_once(' ')
            .and_then(|(piece, id)| Some((piece, id.parse::<usize>().ok()?)));
        let Some((piece, _)) = parsed.filter(|(_, id)| *id == expected) else {
            return Err(AudioError::Inference(format!(
                "invalid tokens.txt at line {}",
                expected + 1
            )));
        };
        pieces.push(piece.to_string());
    }
    if pieces.len() < 2 || pieces.last().map(String::as_str) != Some("<blk>") {
        return Err(AudioError::Inference(
            "tokens.txt must end with <blk>".into(),
        ));
    }
    Ok(pieces)
}

fn detokenize(pieces: &[String], tokens: &[usize]) -> String {
    let mut text = String::new();
    for &token in tokens {
        let piece = pieces[token].as_str();
        if piece.starts_with('<') && piece.ends_with('>') {
            continue;
        }
        text.push_str(piece);
    }
    text.replace('\u{2581}', " ").trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_are_read_in_order_and_end_with_blank() {
        let pieces = parse_tokens("<unk> 0\n\u{2581}hi 1\n<blk> 2\n").unwrap();
        assert_eq!(pieces, ["<unk>", "\u{2581}hi", "<blk>"]);
        assert!(parse_tokens("a 0\nb 2\n<blk> 3\n").is_err(), "gap in ids");
        assert!(parse_tokens("a 0\nb 1\n").is_err(), "no blank");
    }

    #[test]
    fn control_tokens_never_reach_the_text() {
        let pieces: Vec<String> = [
            "<unk>",
            "<|nospeech|>",
            "\u{2581}Ask",
            "\u{2581}not",
            ".",
            "<blk>",
        ]
        .map(String::from)
        .to_vec();
        assert_eq!(detokenize(&pieces, &[0, 2, 1, 3, 4]), "Ask not.");
    }

    #[test]
    fn argmax_picks_the_first_of_equal_scores() {
        assert_eq!(argmax(&[0.0, 2.0, 2.0, -1.0]), 1);
    }

    #[test]
    fn both_models_are_published_as_full_sets() {
        for model in PARAKEET_MODELS {
            let names: Vec<_> = model.set.iter().map(|m| m.local_name).collect();
            assert_eq!(
                names,
                [
                    "encoder.int8.onnx",
                    "decoder.int8.onnx",
                    "joiner.int8.onnx",
                    "tokens.txt"
                ],
                "{}",
                model.id
            );
        }
        assert!(variant("parakeet-tdt-110m-en").is_some());
        assert!(variant("whisper-base").is_none());
    }
}
