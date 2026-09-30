//! Validated narration controls. The director returns indices and numbers,
//! never replacement prose, so it cannot silently rewrite a script.
use crate::{chunking::Chunk, AudioError};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DirectionPlan {
    pub chunks: Vec<ChunkDirection>,
    /// Optional pauses after a 1-based whitespace-word count in the ORIGINAL
    /// script. Must be at punctuation or a paragraph boundary. Never a cut
    /// inside a word; applied only when alignment finds a protected gap.
    #[serde(default)]
    pub pauses: Vec<PauseDirection>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ChunkDirection {
    pub index: usize,
    pub exaggeration: f32,
    pub sentence_pause: f32,
    pub paragraph_pause: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PauseDirection {
    pub after_word: usize,
    pub seconds: f32,
}

impl DirectionPlan {
    pub fn validate(&self, script: &str, chunks: &[Chunk]) -> Result<(), AudioError> {
        let bad = |s: &str| AudioError::InvalidRequest(format!("direction: {s}"));
        if self.chunks.len() != chunks.len() {
            return Err(bad(
                "one ordered control entry is required for every generated chunk",
            ));
        }
        for (index, c) in self.chunks.iter().enumerate() {
            if c.index != index
                || !(0.3..=0.9).contains(&c.exaggeration)
                || !(0.0..=2.0).contains(&c.sentence_pause)
                || !(0.0..=3.0).contains(&c.paragraph_pause)
            {
                return Err(bad("invalid index or control range"));
            }
        }
        let words: Vec<_> = script.split_whitespace().collect();
        if self.pauses.len() > words.len() {
            return Err(bad("too many pause entries"));
        }
        let mut previous = 0;
        for p in &self.pauses {
            if p.after_word <= previous
                || p.after_word >= words.len()
                || !(0.0..=3.0).contains(&p.seconds)
            {
                return Err(bad(
                    "pauses must have increasing unique in-range word counts and 0..=3 seconds",
                ));
            }
            let last =
                words[p.after_word - 1].trim_end_matches(['\"', '\'', '\u{201d}', '\u{2019}', ')']);
            if !last.ends_with(['.', ',', ';', ':', '!', '?']) {
                return Err(bad(
                    "pause must follow punctuation, not split a phrase arbitrarily",
                ));
            }
            previous = p.after_word;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn plan() -> DirectionPlan {
        DirectionPlan {
            chunks: vec![ChunkDirection {
                index: 0,
                exaggeration: 0.7,
                sentence_pause: 0.6,
                paragraph_pause: 1.0,
            }],
            pauses: vec![PauseDirection {
                after_word: 2,
                seconds: 1.2,
            }],
        }
    }
    #[test]
    fn plan_has_no_replacement_text_and_requires_complete_ordered_controls() {
        let chunks = vec![Chunk::new("First sentence. Second sentence.", 20, true)];
        let mut p = plan();
        assert!(p.validate(&chunks[0].text, &chunks).is_ok());
        p.chunks[0].index = 1;
        assert!(p.validate(&chunks[0].text, &chunks).is_err());
        assert!(
            serde_json::from_str::<DirectionPlan>(r#"{"chunks":[],"text":"rewrite"}"#).is_err()
        );
        p.chunks.clear();
        assert!(p.validate(&chunks[0].text, &chunks).is_err());
    }
    #[test]
    fn invalid_pauses_and_nonfinite_controls_are_refused() {
        let chunks = vec![Chunk::new("First sentence. Second sentence.", 20, true)];
        for index in [0, 1, 4, usize::MAX] {
            let mut p = plan();
            p.pauses[0].after_word = index;
            assert!(p.validate(&chunks[0].text, &chunks).is_err());
        }
        let mut p = plan();
        p.pauses.push(p.pauses[0].clone());
        assert!(p.validate(&chunks[0].text, &chunks).is_err());
        p = plan();
        p.chunks[0].exaggeration = f32::NAN;
        assert!(p.validate(&chunks[0].text, &chunks).is_err());
    }
}
