//! Whisper token decoding (GPT-2 byte-level BPE). Only decoding is needed:
//! the recognizer emits ids, and the pipeline needs their text.

use std::collections::HashMap;
use std::path::Path;

use crate::AudioError;

pub struct WhisperTokenizer {
    /// id -> byte-level token string (e.g. "Ġhello").
    pieces: HashMap<i64, String>,
    special: HashMap<String, i64>,
    byte_of: HashMap<char, u8>,
}

impl WhisperTokenizer {
    pub fn from_file(path: &Path) -> Result<Self, AudioError> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| AudioError::Tokenizer(format!("{}: {e}", path.display())))?;
        let v: serde_json::Value =
            serde_json::from_str(&raw).map_err(|e| AudioError::Tokenizer(e.to_string()))?;
        let mut pieces = HashMap::new();
        for (tok, id) in v["model"]["vocab"]
            .as_object()
            .ok_or_else(|| AudioError::Tokenizer("no vocab".into()))?
        {
            if let Some(id) = id.as_i64() {
                pieces.insert(id, tok.clone());
            }
        }
        let mut special = HashMap::new();
        for t in v["added_tokens"].as_array().into_iter().flatten() {
            if let (Some(c), Some(id)) = (t["content"].as_str(), t["id"].as_i64()) {
                special.insert(c.to_string(), id);
                pieces.entry(id).or_insert_with(|| c.to_string());
            }
        }
        let byte_of = bytes_to_unicode()
            .into_iter()
            .map(|(b, c)| (c, b))
            .collect();
        Ok(Self {
            pieces,
            special,
            byte_of,
        })
    }

    pub fn special(&self, s: &str) -> Result<i64, AudioError> {
        self.special
            .get(s)
            .copied()
            .ok_or_else(|| AudioError::Tokenizer(format!("token `{s}` not in vocabulary")))
    }

    /// The raw piece for one id (still byte-level encoded).
    pub fn piece(&self, id: i64) -> &str {
        self.pieces.get(&id).map(String::as_str).unwrap_or("")
    }

    /// Decode text ids to a string.
    pub fn decode(&self, ids: &[i64]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            for ch in self.piece(id).chars() {
                match self.byte_of.get(&ch) {
                    Some(b) => bytes.push(*b),
                    None => bytes.extend(ch.to_string().as_bytes()),
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// GPT-2's reversible byte <-> printable-unicode map.
fn bytes_to_unicode() -> Vec<(u8, char)> {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32)
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    bs.into_iter()
        .zip(cs)
        .map(|(b, c)| (b as u8, char::from_u32(c).unwrap()))
        .collect()
}
