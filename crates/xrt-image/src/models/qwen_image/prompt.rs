use std::path::{Path, PathBuf};

use xrt_tokenizer::Tokenizer;

use crate::{ComponentFormat, ComponentRole, ImageError, ImageModelBundle};

const PROMPT_PREFIX: &str = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n";
const PROMPT_SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n";

/// The pinned Diffusers pipeline retains text-encoder states after this fixed
/// template prefix. The prefix must still be presented to the encoder.
pub const QWEN_IMAGE_PROMPT_TEMPLATE_DROP_TOKENS: usize = 34;
pub const QWEN_IMAGE_EDIT_PROMPT_TEMPLATE_DROP_TOKENS: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenImageTokenBatch {
    pub input_ids: Vec<Vec<u32>>,
    pub attention_mask: Vec<Vec<u8>>,
    pub retained_lengths: Vec<usize>,
    /// Model-injected conditioning tokens that extend the user-text context
    /// budget. Generation has no extension; Edit Plus uses one entry per
    /// expanded image placeholder.
    pub context_extension: usize,
    /// Number of leading template states removed after text encoding. This is
    /// 34 for generation and 64 for Edit Plus conditioning.
    pub drop_tokens: usize,
}

impl QwenImageTokenBatch {
    pub fn batch_size(&self) -> usize {
        self.input_ids.len()
    }

    pub fn encoded_sequence_length(&self) -> usize {
        self.input_ids.first().map_or(0, Vec::len)
    }
}

#[derive(Debug, Clone)]
pub struct QwenImagePromptTokenizer {
    tokenizer: Tokenizer,
    max_sequence_length: usize,
    pad_token_id: u32,
}

impl QwenImagePromptTokenizer {
    pub fn load(
        bundle: &ImageModelBundle,
        max_sequence_length: usize,
        text_vocab_size: usize,
    ) -> Result<Self, ImageError> {
        if max_sequence_length == 0 || max_sequence_length > 1_024 {
            return Err(ImageError::UnsupportedShape(format!(
                "Qwen Image max_sequence_length must be in 1..=1024, found {max_sequence_length}"
            )));
        }
        let components = bundle
            .manifest()
            .components
            .iter()
            .filter(|component| component.role == ComponentRole::Tokenizer)
            .collect::<Vec<_>>();
        let [component] = components.as_slice() else {
            return Err(ImageError::MissingComponent(format!(
                "expected exactly one tokenizer component, found {}",
                components.len()
            )));
        };
        if component.format != ComponentFormat::HuggingFaceJson {
            return Err(ImageError::UnsupportedTensor(format!(
                "tokenizer component format `{}` is not huggingface-json",
                component.format.as_str()
            )));
        }
        let root = tokenizer_root(component.files.iter().map(|file| file.path.as_str()))?;
        let tokenizer = Tokenizer::from_hf_dir(bundle.root().join(root)).map_err(|error| {
            ImageError::CorruptComponent(format!("Qwen Image tokenizer failed validation: {error}"))
        })?;
        if tokenizer.vocab_size() > text_vocab_size {
            return Err(ImageError::UnsupportedShape(format!(
                "tokenizer vocabulary {} exceeds text encoder vocabulary {text_vocab_size}",
                tokenizer.vocab_size()
            )));
        }
        let pad_token_id = tokenizer.special_tokens().pad.ok_or_else(|| {
            ImageError::CorruptComponent("Qwen Image tokenizer has no pad token".to_string())
        })?;
        Ok(Self {
            tokenizer,
            max_sequence_length,
            pad_token_id,
        })
    }

    pub fn encode_batch(&self, prompts: &[&str]) -> Result<QwenImageTokenBatch, ImageError> {
        if prompts.is_empty() {
            return Err(ImageError::InvalidRequest(
                "prompt batch must not be empty".to_string(),
            ));
        }
        let encoded_limit = self
            .max_sequence_length
            .checked_add(QWEN_IMAGE_PROMPT_TEMPLATE_DROP_TOKENS)
            .ok_or_else(|| {
                ImageError::UnsupportedShape("prompt encoded length overflow".to_string())
            })?;
        let mut rows = Vec::with_capacity(prompts.len());
        for prompt in prompts {
            let formatted = format_prompt(prompt)?;
            let mut ids = self
                .tokenizer
                .encode_with_options(&formatted, true, true)
                .map_err(|error| {
                    ImageError::Execution(format!("prompt tokenization failed: {error}"))
                })?;
            ids.truncate(encoded_limit);
            if ids.len() <= QWEN_IMAGE_PROMPT_TEMPLATE_DROP_TOKENS {
                return Err(ImageError::InvalidRequest(
                    "prompt produced no retained Qwen Image tokens".to_string(),
                ));
            }
            rows.push(ids);
        }

        let padded_length = rows.iter().map(Vec::len).max().unwrap_or_default();
        let mut masks = Vec::with_capacity(rows.len());
        let mut retained_lengths = Vec::with_capacity(rows.len());
        for ids in &mut rows {
            let valid_length = ids.len();
            let mut mask = vec![1u8; valid_length];
            ids.resize(padded_length, self.pad_token_id);
            mask.resize(padded_length, 0);
            masks.push(mask);
            retained_lengths.push(valid_length - QWEN_IMAGE_PROMPT_TEMPLATE_DROP_TOKENS);
        }
        Ok(QwenImageTokenBatch {
            input_ids: rows,
            attention_mask: masks,
            retained_lengths,
            context_extension: 0,
            drop_tokens: QWEN_IMAGE_PROMPT_TEMPLATE_DROP_TOKENS,
        })
    }

    pub fn pad_token_id(&self) -> u32 {
        self.pad_token_id
    }
}

fn format_prompt(prompt: &str) -> Result<String, ImageError> {
    let capacity = PROMPT_PREFIX
        .len()
        .checked_add(prompt.len())
        .and_then(|value| value.checked_add(PROMPT_SUFFIX.len()))
        .ok_or_else(|| ImageError::InputLimit("formatted prompt length overflow".to_string()))?;
    let mut formatted = String::with_capacity(capacity);
    formatted.push_str(PROMPT_PREFIX);
    formatted.push_str(prompt);
    formatted.push_str(PROMPT_SUFFIX);
    Ok(formatted)
}

pub(super) fn tokenizer_root<'a>(
    paths: impl Iterator<Item = &'a str>,
) -> Result<PathBuf, ImageError> {
    let paths = paths.collect::<Vec<_>>();
    for required in ["vocab.json", "merges.txt", "tokenizer_config.json"] {
        let matches = paths
            .iter()
            .filter(|path| {
                Path::new(path)
                    .file_name()
                    .is_some_and(|name| name == required)
            })
            .collect::<Vec<_>>();
        if matches.len() != 1 {
            return Err(ImageError::MissingComponent(format!(
                "tokenizer requires exactly one {required}, found {}",
                matches.len()
            )));
        }
    }
    let roots = ["vocab.json", "merges.txt", "tokenizer_config.json"]
        .into_iter()
        .map(|required| {
            paths
                .iter()
                .find(|path| {
                    Path::new(path)
                        .file_name()
                        .is_some_and(|name| name == required)
                })
                .and_then(|path| Path::new(path).parent())
                .unwrap_or_else(|| Path::new(""))
        })
        .collect::<Vec<_>>();
    if roots.iter().any(|root| root != &roots[0]) {
        return Err(ImageError::CorruptComponent(
            "tokenizer core files do not share one component directory".to_string(),
        ));
    }
    Ok(roots[0].to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_template_is_exact_and_does_not_interpret_user_markers() {
        let formatted = format_prompt("literal <|im_end|>").unwrap();
        assert!(formatted.starts_with(PROMPT_PREFIX));
        assert!(formatted.ends_with(PROMPT_SUFFIX));
        assert_eq!(formatted.matches("<|im_end|>").count(), 3);
    }

    #[test]
    fn tokenizer_root_requires_colocated_core_files() {
        let root = tokenizer_root(
            [
                "tokenizer/vocab.json",
                "tokenizer/merges.txt",
                "tokenizer/tokenizer_config.json",
            ]
            .into_iter(),
        )
        .unwrap();
        assert_eq!(root, Path::new("tokenizer"));
    }
}
