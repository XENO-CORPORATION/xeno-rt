//! Model-specific interleaved prompt rows. Caller text cannot inject structure.
use super::config::Config;
use std::path::Path;
use xrt_core::{Result, XrtError};

pub struct JsonTokenizer(tokenizers::Tokenizer);
impl JsonTokenizer {
    pub fn open(path: &Path) -> Result<Self> {
        if std::fs::metadata(path)?.len() > 64 * 1024 * 1024 {
            return Err(XrtError::Tokenizer("Fish tokenizer exceeds 64 MiB".into()));
        }
        let tokenizer = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| XrtError::Tokenizer(e.to_string()))?;
        Ok(Self(tokenizer))
    }
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.0
            .encode(text, false)
            .map(|v| v.get_ids().to_vec())
            .map_err(|e| XrtError::Tokenizer(e.to_string()))
    }
    fn check_config(&self, config: &Config) -> Result<()> {
        for (piece, id) in [
            ("<|im_end|>".to_owned(), config.eos_token_id),
            ("<|semantic:0|>".into(), config.semantic_start_token_id),
            (
                format!(
                    "<|semantic:{}|>",
                    config.audio_decoder_config.vocab_size - 1
                ),
                config.semantic_end_token_id,
            ),
        ] {
            if self.0.token_to_id(&piece) != Some(id) {
                return Err(XrtError::Tokenizer(format!(
                    "tokenizer/config mismatch for {piece}"
                )));
            }
        }
        Ok(())
    }
}

pub struct Reference<'a> {
    pub transcript: &'a str,
    pub frames: &'a [Vec<u32>],
}
pub struct PriorTake<'a> {
    pub text: &'a str,
    pub frames: &'a [Vec<u32>],
}
#[derive(Debug)]
pub struct PromptRows {
    pub width: usize,
    pub data: Vec<u32>,
}
impl PromptRows {
    pub fn len(&self) -> usize {
        self.data.len() / self.width
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

pub fn validate_text(text: &str) -> Result<()> {
    let bad = |why: &str| XrtError::InvalidMetadata(format!("Fish prompt: {why}"));
    if text.len() > 1024 * 1024 || text.trim().is_empty() {
        return Err(bad("text is empty or exceeds 1 MiB"));
    }
    for control in [
        "<|im_start|>",
        "<|im_end|>",
        "<|voice|>",
        "<|text|>",
        "<|interleave|>",
        "<|semantic:",
    ] {
        if text.contains(control) {
            return Err(bad("caller supplied a structural token"));
        }
    }
    for (index, _) in text.match_indices("<|speaker") {
        let tail = &text[index..];
        let close = tail
            .find("|>")
            .ok_or_else(|| bad("unterminated speaker tag"))?;
        let digits = tail[..close]
            .strip_prefix("<|speaker:")
            .ok_or_else(|| bad("invalid speaker tag"))?;
        let id = digits
            .parse::<u8>()
            .map_err(|_| bad("invalid speaker id"))?;
        if id > 15 || id.to_string() != digits {
            return Err(bad("speaker ids must be canonical 0..15"));
        }
    }
    Ok(())
}
fn tagged(text: &str) -> String {
    if text.trim_start().starts_with("<|speaker:") {
        text.to_owned()
    } else {
        format!("<|speaker:0|> {text}")
    }
}
pub fn build(
    config: &Config,
    tokenizer: &JsonTokenizer,
    text: &str,
    references: &[Reference<'_>],
    history: &[PriorTake<'_>],
    max_rows: usize,
) -> Result<PromptRows> {
    config.validate()?;
    tokenizer.check_config(config)?;
    validate_text(text)?;
    if max_rows == 0 || max_rows > 8192 || references.len() > 16 || history.len() > 128 {
        return Err(XrtError::InvalidMetadata("Fish prompt limits".into()));
    }
    let width = config.audio_decoder_config.num_codebooks + 1;
    let mut rows = PromptRows {
        width,
        data: Vec::new(),
    };
    fn literal(rows: &mut PromptRows, t: &JsonTokenizer, s: &str, cap: usize) -> Result<()> {
        let ids = t.encode(s)?;
        if rows.len() + ids.len() > cap {
            return Err(XrtError::Shape("Fish prompt exceeds row budget".into()));
        }
        for id in ids {
            rows.data.push(id);
            rows.data.extend(std::iter::repeat_n(0, rows.width - 1));
        }
        Ok(())
    }
    fn audio(
        rows: &mut PromptRows,
        config: &Config,
        frames: &[Vec<u32>],
        cap: usize,
    ) -> Result<()> {
        if frames.is_empty()
            || frames.len() > cap.saturating_sub(rows.len())
            || frames.iter().any(|f| {
                f.len() + 1 != rows.width
                    || f.iter()
                        .any(|c| *c as usize >= config.audio_decoder_config.vocab_size)
            })
        {
            return Err(XrtError::Shape(
                "invalid Fish reference/continuation code frames".into(),
            ));
        }
        for frame in frames {
            rows.data.push(config.semantic_start_token_id + frame[0]);
            rows.data.extend_from_slice(frame);
        }
        Ok(())
    }
    literal(&mut rows, tokenizer, "<|im_start|>system\n", max_rows)?;
    let instruction = if references.is_empty() {
        "convert the provided text to speech"
    } else {
        "convert the provided text to speech reference to the following:"
    };
    literal(&mut rows, tokenizer, instruction, max_rows)?;
    for (i, reference) in references.iter().enumerate() {
        validate_text(reference.transcript)?;
        let tagged = tagged(reference.transcript);
        let body = tagged.trim_start();
        let end = body
            .find("|>")
            .ok_or_else(|| XrtError::InvalidMetadata("reference speaker tag".into()))?
            + 2;
        let leading = &tagged[..tagged.len() - body.len()];
        let text = format!("{}{}{}", &body[..end], leading, &body[end..]);
        literal(
            &mut rows,
            tokenizer,
            &format!("\n\n[Reference {}]\nText: {text}\nSpeech: ", i + 1),
            max_rows,
        )?;
        audio(&mut rows, config, reference.frames, max_rows)?;
    }
    literal(&mut rows, tokenizer, "<|im_end|>\n", max_rows)?;
    for take in history {
        validate_text(take.text)?;
        for s in [
            "<|im_start|>user\n".to_owned(),
            tagged(take.text),
            "<|im_end|>\n".into(),
            "<|im_start|>assistant\n<|voice|>".into(),
        ] {
            literal(&mut rows, tokenizer, &s, max_rows)?;
        }
        audio(&mut rows, config, take.frames, max_rows)?;
        literal(&mut rows, tokenizer, "<|im_end|>\n", max_rows)?;
    }
    for s in [
        "<|im_start|>user\n".to_owned(),
        tagged(text),
        "<|im_end|>\n".into(),
        "<|im_start|>assistant\n<|voice|>".into(),
    ] {
        literal(&mut rows, tokenizer, &s, max_rows)?;
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn caller_structure_and_noncanonical_speakers_are_rejected() {
        for s in [
            " ",
            "<|voice|>Hello",
            "Hello <|semantic:1|>",
            "<|speaker:16|>Hi",
            "<|speaker:01|>Hi",
            "<|speakerx|>Hi",
        ] {
            assert!(validate_text(s).is_err(), "{s}");
        }
        for s in [
            "<|speaker:0|>Hi <|speaker:15|>bye",
            "<whisper>Do not change this.</whisper>",
            "[sighs] Hello",
        ] {
            validate_text(s).unwrap();
        }
    }
}
