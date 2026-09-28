//! Chatterbox grapheme tokenizer.
//!
//! Reads the model's Hugging Face `tokenizer.json` (a character-level BPE with
//! 2454 pieces and ~265 merges) and reproduces the reference `MTLTokenizer`
//! exactly. Pinned by golden IDs generated from the reference implementation
//! (`tests/data/tokenizer_golden.json`).
//!
//! 🔴 The word-boundary bug this module exists to prevent: the shipped v3
//! `tokenizer.json` normalizer is bare NFKD and marks `[SPACE]` as not
//! normalized, so a naive HF load emits NO space tokens and the model runs
//! words together — heard as mumbling (ASR WER 50% → 0% once fixed). The
//! reference replaces `' '` with `[SPACE]` in Python before encoding. We do
//! the same here, in code, so the behaviour does not depend on which
//! tokenizer.json variant is on disk.

use std::collections::HashMap;
use std::path::Path;

use unicode_normalization::UnicodeNormalization;

use crate::AudioError;

pub const SPACE: &str = "[SPACE]";
pub const START_TEXT: i64 = 255;
pub const STOP_TEXT: i64 = 0;
pub const START_SPEECH: i64 = 6561;
pub const STOP_SPEECH: i64 = 6562;
pub const EXAGGERATION: i64 = 6563;

/// Languages the multilingual checkpoint was trained on.
pub const SUPPORTED_LANGUAGES: &[(&str, &str)] = &[
    ("ar", "Arabic"),
    ("da", "Danish"),
    ("de", "German"),
    ("el", "Greek"),
    ("en", "English"),
    ("es", "Spanish"),
    ("fi", "Finnish"),
    ("fr", "French"),
    ("he", "Hebrew"),
    ("hi", "Hindi"),
    ("it", "Italian"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("ms", "Malay"),
    ("nl", "Dutch"),
    ("no", "Norwegian"),
    ("pl", "Polish"),
    ("pt", "Portuguese"),
    ("ru", "Russian"),
    ("sv", "Swedish"),
    ("sw", "Swahili"),
    ("tr", "Turkish"),
    ("zh", "Chinese"),
];

/// Languages whose reference pipeline applies script conversion we have not
/// ported (Cangjie for zh, kana for ja, diacritics for he, jamo for ko, stress
/// for ru). Refused rather than silently producing different tokens.
const NEEDS_SCRIPT_CONVERSION: &[&str] = &["zh", "ja", "he", "ko", "ru"];

#[derive(Debug)]
pub struct ChatterboxTokenizer {
    vocab: HashMap<String, i64>,
    merges: HashMap<(String, String), usize>,
    /// Added tokens matched verbatim before BPE (e.g. `[SPACE]`, `[en]`).
    added: Vec<(String, i64)>,
    unk: i64,
}

impl ChatterboxTokenizer {
    pub fn from_file(path: &Path) -> Result<Self, AudioError> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| AudioError::Tokenizer(format!("cannot read {}: {e}", path.display())))?;
        Self::from_json(&raw)
    }

    pub fn from_json(raw: &str) -> Result<Self, AudioError> {
        let v: serde_json::Value =
            serde_json::from_str(raw).map_err(|e| AudioError::Tokenizer(e.to_string()))?;
        let model = &v["model"];
        if model["type"].as_str() != Some("BPE") {
            return Err(AudioError::Tokenizer(
                "expected a BPE tokenizer model".into(),
            ));
        }
        let vocab: HashMap<String, i64> = model["vocab"]
            .as_object()
            .ok_or_else(|| AudioError::Tokenizer("model.vocab missing".into()))?
            .iter()
            .filter_map(|(k, id)| id.as_i64().map(|id| (k.clone(), id)))
            .collect();

        let mut merges = HashMap::new();
        for (rank, m) in model["merges"]
            .as_array()
            .ok_or_else(|| AudioError::Tokenizer("model.merges missing".into()))?
            .iter()
            .enumerate()
        {
            let pair = match m {
                serde_json::Value::String(s) => {
                    let mut it = s.splitn(2, ' ');
                    (
                        it.next().unwrap_or("").to_string(),
                        it.next().unwrap_or("").to_string(),
                    )
                }
                serde_json::Value::Array(a) if a.len() == 2 => (
                    a[0].as_str().unwrap_or("").to_string(),
                    a[1].as_str().unwrap_or("").to_string(),
                ),
                _ => return Err(AudioError::Tokenizer(format!("malformed merge #{rank}"))),
            };
            merges.entry(pair).or_insert(rank);
        }

        let mut added: Vec<(String, i64)> = v["added_tokens"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|t| Some((t["content"].as_str()?.to_string(), t["id"].as_i64()?)))
                    .collect()
            })
            .unwrap_or_default();
        // Longest first so "[SPACE]" never loses to a shorter prefix token.
        added.sort_by_key(|entry| std::cmp::Reverse(entry.0.len()));

        let unk_name = model["unk_token"].as_str().unwrap_or("[UNK]");
        let unk = vocab
            .get(unk_name)
            .copied()
            .or_else(|| added.iter().find(|(c, _)| c == unk_name).map(|(_, i)| *i))
            .unwrap_or(1);

        if !added.iter().any(|(c, _)| c == SPACE) {
            return Err(AudioError::Tokenizer(format!(
                "tokenizer has no {SPACE} token"
            )));
        }
        Ok(Self {
            vocab,
            merges,
            added,
            unk,
        })
    }

    /// Validate a language code against what this port reproduces exactly.
    pub fn check_language(language: &str) -> Result<(), AudioError> {
        let lang = language.to_ascii_lowercase();
        if !SUPPORTED_LANGUAGES.iter().any(|(c, _)| *c == lang) {
            return Err(AudioError::InvalidRequest(format!(
                "unsupported language `{language}`"
            )));
        }
        if NEEDS_SCRIPT_CONVERSION.contains(&lang.as_str()) {
            return Err(AudioError::InvalidRequest(format!(
                "language `{language}` needs script conversion that xrt-audio does not implement yet"
            )));
        }
        Ok(())
    }

    /// Text tokens exactly as the reference `MTLTokenizer.encode` produces them
    /// (lowercase → NFKD → `[lang]` prefix → spaces as `[SPACE]` → BPE), with
    /// no template tokens. Input is expected to be [`punc_norm`]-ed already.
    pub fn encode_text(&self, text: &str, language: &str) -> Result<Vec<i64>, AudioError> {
        Self::check_language(language)?;
        let prepared: String = text.to_lowercase().nfkd().collect();
        let prepared =
            format!("[{}]{}", language.to_ascii_lowercase(), prepared).replace(' ', SPACE);

        let mut out = Vec::new();
        let mut rest = prepared.as_str();
        let mut word = String::new();
        while !rest.is_empty() {
            if let Some((content, id)) = self
                .added
                .iter()
                .find(|(c, _)| rest.starts_with(c.as_str()))
            {
                self.flush_word(&mut word, &mut out);
                out.push(*id);
                rest = &rest[content.len()..];
                continue;
            }
            let ch = rest.chars().next().expect("non-empty");
            // The HF `Whitespace` pre-tokenizer splits on \w+|[^\w\s]+ runs.
            // Spaces are already `[SPACE]`; other whitespace separates words.
            if ch.is_whitespace() {
                self.flush_word(&mut word, &mut out);
            } else if !word.is_empty()
                && is_word_char(ch) != word.chars().last().map(is_word_char).unwrap_or(false)
            {
                self.flush_word(&mut word, &mut out);
                word.push(ch);
            } else {
                word.push(ch);
            }
            rest = &rest[ch.len_utf8()..];
        }
        self.flush_word(&mut word, &mut out);
        Ok(out)
    }

    /// Full model input: `[EXAGGERATION] [START] text [STOP] [START_SPEECH] [START_SPEECH]`,
    /// the v3 tokenizer's post-processing template.
    pub fn encode_prompt(&self, text: &str, language: &str) -> Result<Vec<i64>, AudioError> {
        let mut ids = vec![EXAGGERATION, START_TEXT];
        ids.extend(self.encode_text(text, language)?);
        ids.extend([STOP_TEXT, START_SPEECH, START_SPEECH]);
        Ok(ids)
    }

    /// Number of text tokens a piece of text will occupy — what the chunker
    /// budgets against, since the model's limit is in positions, not chars.
    pub fn count(&self, text: &str, language: &str) -> Result<usize, AudioError> {
        Ok(self.encode_text(&punc_norm(text), language)?.len())
    }

    fn flush_word(&self, word: &mut String, out: &mut Vec<i64>) {
        if word.is_empty() {
            return;
        }
        let mut parts: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        loop {
            let best = parts
                .windows(2)
                .enumerate()
                .filter_map(|(i, w)| {
                    self.merges
                        .get(&(w[0].clone(), w[1].clone()))
                        .map(|r| (*r, i))
                })
                .min();
            let Some((_, i)) = best else { break };
            let merged = format!("{}{}", parts[i], parts[i + 1]);
            parts.splice(i..i + 2, [merged]);
        }
        for p in parts {
            out.push(self.vocab.get(&p).copied().unwrap_or(self.unk));
        }
        word.clear();
    }
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || unicode_is_mark(c)
}

/// Combining marks count as word characters in the regex `\w` class, which
/// matters after NFKD splits "é" into "e" + U+0301.
fn unicode_is_mark(c: char) -> bool {
    matches!(c as u32, 0x0300..=0x036F | 0x1AB0..=0x1AFF | 0x1DC0..=0x1DFF | 0x20D0..=0x20FF | 0xFE20..=0xFE2F)
}

/// The reference `punc_norm`: capitalise, collapse whitespace, map uncommon
/// punctuation onto what the model saw in training, ensure a sentence ender.
pub fn punc_norm(text: &str) -> String {
    if text.is_empty() {
        return "You need to add some text for me to talk.".to_string();
    }
    let mut t: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if let Some(first) = t.chars().next() {
        if first.is_lowercase() {
            t = first.to_uppercase().collect::<String>() + &t[first.len_utf8()..];
        }
    }
    for (old, new) in [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ] {
        t = t.replace(old, new);
    }
    let t = t.trim_end_matches(' ').to_string();
    const ENDERS: [&str; 10] = [".", "!", "?", "-", ",", "、", "，", "。", "？", "！"];
    if ENDERS.iter().any(|e| t.ends_with(e)) {
        t
    } else {
        t + "."
    }
}
