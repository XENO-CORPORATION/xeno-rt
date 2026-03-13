use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use xrt_core::{Result, XrtError};
use xrt_gguf::GgufFile;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenizerKind {
    Piece,
    Bpe,
    Gpt2Bpe,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpecialTokens {
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    pub unk: Option<u32>,
    pub pad: Option<u32>,
    pub add_bos: bool,
    pub add_eos: bool,
}

#[derive(Debug, Clone)]
pub struct Tokenizer {
    vocab: Vec<String>,
    vocab_map: HashMap<String, u32>,
    scores: Vec<f32>,
    merges: HashMap<(String, String), usize>,
    kind: TokenizerKind,
    special: SpecialTokens,
    special_by_piece: HashMap<String, u32>,
    special_ids: HashSet<u32>,
    max_piece_chars: usize,
}

impl Tokenizer {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let tokens = gguf
            .metadata_array("tokenizer.ggml.tokens")
            .ok_or_else(|| {
                XrtError::InvalidMetadata(
                    "missing tokenizer.ggml.tokens array in GGUF metadata".to_string(),
                )
            })?
            .as_strings()
            .ok_or_else(|| {
                XrtError::InvalidMetadata(
                    "tokenizer.ggml.tokens must be an array of strings".to_string(),
                )
            })?;

        let vocab: Vec<String> = tokens.into_iter().map(ToOwned::to_owned).collect();
        let vocab_map: HashMap<String, u32> = vocab
            .iter()
            .enumerate()
            .map(|(index, token)| (token.clone(), index as u32))
            .collect();

        let scores = gguf
            .metadata_array("tokenizer.ggml.scores")
            .and_then(|array| array.as_f32_vec())
            .unwrap_or_else(|| vec![0.0; vocab.len()]);
        if scores.len() != vocab.len() {
            return Err(XrtError::InvalidMetadata(format!(
                "tokenizer.ggml.scores length {} does not match vocab size {}",
                scores.len(),
                vocab.len()
            )));
        }

        let mut merges = HashMap::new();
        if let Some(array) = gguf.metadata_array("tokenizer.ggml.merges") {
            for (rank, merge) in array
                .as_strings()
                .ok_or_else(|| {
                    XrtError::InvalidMetadata(
                        "tokenizer.ggml.merges must be an array of strings".to_string(),
                    )
                })?
                .into_iter()
                .enumerate()
            {
                if let Some((left, right)) = merge.split_once(' ') {
                    merges.insert((left.to_string(), right.to_string()), rank);
                }
            }
        }

        let special = SpecialTokens {
            bos: gguf
                .metadata_usize("tokenizer.ggml.bos_token_id")
                .map(|value| value as u32),
            eos: gguf
                .metadata_usize("tokenizer.ggml.eos_token_id")
                .map(|value| value as u32),
            unk: gguf
                .metadata_usize("tokenizer.ggml.unknown_token_id")
                .map(|value| value as u32),
            pad: gguf
                .metadata_usize("tokenizer.ggml.padding_token_id")
                .map(|value| value as u32),
            add_bos: gguf
                .metadata_bool("tokenizer.ggml.add_bos_token")
                .unwrap_or(true),
            add_eos: gguf
                .metadata_bool("tokenizer.ggml.add_eos_token")
                .unwrap_or(false),
        };

        let mut special_by_piece = HashMap::new();
        let mut special_ids = HashSet::new();
        for id in [special.bos, special.eos, special.unk, special.pad]
            .into_iter()
            .flatten()
        {
            if let Some(piece) = vocab.get(id as usize) {
                special_by_piece.insert(piece.clone(), id);
                special_ids.insert(id);
            }
        }

        let tokenizer_model = gguf.metadata_string("tokenizer.ggml.model").unwrap_or("");
        let kind = if tokenizer_model == "gpt2" {
            TokenizerKind::Gpt2Bpe
        } else if merges.is_empty() {
            TokenizerKind::Piece
        } else {
            TokenizerKind::Bpe
        };
        let max_piece_chars = vocab
            .iter()
            .map(|token| token.chars().count())
            .max()
            .unwrap_or(1);

        Ok(Self {
            vocab,
            vocab_map,
            scores,
            merges,
            kind,
            special,
            special_by_piece,
            special_ids,
            max_piece_chars,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn special_tokens(&self) -> &SpecialTokens {
        &self.special
    }

    pub fn token_to_piece(&self, token: u32) -> Option<&str> {
        self.vocab.get(token as usize).map(String::as_str)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.encode_with_options(text, true, true)
    }

    pub fn encode_with_options(
        &self,
        text: &str,
        add_special: bool,
        allow_special: bool,
    ) -> Result<Vec<u32>> {
        let mut output = Vec::new();
        if add_special && self.special.add_bos {
            if let Some(bos) = self.special.bos {
                output.push(bos);
            }
        }

        let mut position = 0usize;
        while position < text.len() {
            if allow_special {
                if let Some((piece, token_id)) = self.match_special_prefix(&text[position..]) {
                    output.push(token_id);
                    position += piece.len();
                    continue;
                }
            }

            let next_boundary = if allow_special {
                self.next_special_boundary(text, position)
                    .unwrap_or(text.len())
            } else {
                text.len()
            };
            let segment = &text[position..next_boundary];
            output.extend(self.encode_segment(segment)?);
            position = next_boundary;
        }

        if add_special && self.special.add_eos {
            if let Some(eos) = self.special.eos {
                output.push(eos);
            }
        }

        Ok(output)
    }

    pub fn decode(&self, tokens: &[u32], skip_special: bool) -> Result<String> {
        if self.kind == TokenizerKind::Gpt2Bpe {
            return self.decode_gpt2(tokens, skip_special);
        }

        let mut output = String::new();
        let mut pending_bytes = Vec::new();

        for token in tokens {
            if skip_special && self.special_ids.contains(token) {
                continue;
            }

            let piece = self.vocab.get(*token as usize).ok_or_else(|| {
                XrtError::Tokenizer(format!("token id {token} is out of vocabulary"))
            })?;

            if let Some(byte) = parse_byte_token(piece) {
                pending_bytes.push(byte);
                continue;
            }

            if !pending_bytes.is_empty() {
                output.push_str(std::str::from_utf8(&pending_bytes)?);
                pending_bytes.clear();
            }
            output.push_str(&piece.replace('▁', " "));
        }

        if !pending_bytes.is_empty() {
            output.push_str(std::str::from_utf8(&pending_bytes)?);
        }

        Ok(output)
    }

    fn decode_gpt2(&self, tokens: &[u32], skip_special: bool) -> Result<String> {
        let mut bytes = Vec::new();
        for token in tokens {
            if skip_special && self.special_ids.contains(token) {
                continue;
            }
            let piece = self.vocab.get(*token as usize).ok_or_else(|| {
                XrtError::Tokenizer(format!("token id {token} is out of vocabulary"))
            })?;
            for ch in piece.chars() {
                if let Some(byte) = unicode_to_byte(ch) {
                    bytes.push(byte);
                }
            }
        }
        String::from_utf8(bytes).map_err(|e| XrtError::Tokenizer(format!("invalid utf8 in decode: {e}")))
    }

    fn encode_segment(&self, segment: &str) -> Result<Vec<u32>> {
        match self.kind {
            TokenizerKind::Piece => self.encode_piece_segment(segment),
            TokenizerKind::Bpe => self.encode_bpe_segment(segment),
            TokenizerKind::Gpt2Bpe => self.encode_gpt2_bpe_segment(segment),
        }
    }

    fn encode_piece_segment(&self, segment: &str) -> Result<Vec<u32>> {
        let normalized = normalize_piece_segment(segment);
        let positions = char_positions(&normalized);
        let mut best_score = vec![f32::NEG_INFINITY; positions.len()];
        let mut best_next = vec![None::<usize>; positions.len()];
        best_score[positions.len() - 1] = 0.0;

        for index in (0..positions.len() - 1).rev() {
            let max_end = (index + self.max_piece_chars + 1).min(positions.len());
            for next in index + 1..max_end {
                let piece = &normalized[positions[index]..positions[next]];
                let Some(token) = self.vocab_map.get(piece) else {
                    continue;
                };
                let score = self.scores[*token as usize] + best_score[next];
                if score > best_score[index] {
                    best_score[index] = score;
                    best_next[index] = Some(next);
                }
            }
        }

        let mut output = Vec::new();
        let mut index = 0usize;
        while index < positions.len() - 1 {
            if let Some(next) = best_next[index] {
                let piece = &normalized[positions[index]..positions[next]];
                let token = self.vocab_map[piece];
                output.push(token);
                index = next;
                continue;
            }

            let end = positions[index + 1];
            let piece = &normalized[positions[index]..end];
            output.extend(self.fallback_piece(piece)?);
            index += 1;
        }

        Ok(output)
    }

    fn encode_bpe_segment(&self, segment: &str) -> Result<Vec<u32>> {
        let normalized = normalize_piece_segment(segment);
        let mut pieces: Vec<String> = normalized.chars().map(|ch| ch.to_string()).collect();

        loop {
            let mut best_pair: Option<(usize, usize)> = None;
            for index in 0..pieces.len().saturating_sub(1) {
                let pair = (pieces[index].clone(), pieces[index + 1].clone());
                let Some(rank) = self.merges.get(&pair).copied() else {
                    continue;
                };
                let merged = format!("{}{}", pair.0, pair.1);
                if !self.vocab_map.contains_key(&merged) {
                    continue;
                }
                match best_pair {
                    Some((_, current_rank)) if current_rank <= rank => {}
                    _ => best_pair = Some((index, rank)),
                }
            }

            let Some((index, _)) = best_pair else {
                break;
            };
            let merged = format!("{}{}", pieces[index], pieces[index + 1]);
            pieces.splice(index..=index + 1, [merged]);
        }

        let mut output = Vec::new();
        for piece in pieces {
            if let Some(token) = self.vocab_map.get(&piece) {
                output.push(*token);
            } else {
                output.extend(self.fallback_piece(&piece)?);
            }
        }
        Ok(output)
    }

    fn encode_gpt2_bpe_segment(&self, segment: &str) -> Result<Vec<u32>> {
        // GPT-2 BPE: convert bytes to unicode chars, then run BPE merges
        let unicode_str: String = segment.as_bytes().iter().map(|&b| byte_to_unicode(b)).collect();

        let mut pieces: Vec<String> = unicode_str.chars().map(|ch| ch.to_string()).collect();
        if pieces.is_empty() {
            return Ok(Vec::new());
        }

        loop {
            let mut best_pair: Option<(usize, usize)> = None;
            for index in 0..pieces.len().saturating_sub(1) {
                let pair = (pieces[index].clone(), pieces[index + 1].clone());
                let Some(rank) = self.merges.get(&pair).copied() else {
                    continue;
                };
                let merged = format!("{}{}", pair.0, pair.1);
                if !self.vocab_map.contains_key(&merged) {
                    continue;
                }
                match best_pair {
                    Some((_, current_rank)) if current_rank <= rank => {}
                    _ => best_pair = Some((index, rank)),
                }
            }

            let Some((index, _)) = best_pair else {
                break;
            };
            let merged = format!("{}{}", pieces[index], pieces[index + 1]);
            pieces.splice(index..=index + 1, [merged]);
        }

        let mut output = Vec::new();
        for piece in pieces {
            if let Some(token) = self.vocab_map.get(&piece) {
                output.push(*token);
            } else if let Some(unk) = self.special.unk {
                output.push(unk);
            } else {
                return Err(XrtError::Tokenizer(format!(
                    "unknown token piece: {piece:?}"
                )));
            }
        }
        Ok(output)
    }

    fn fallback_piece(&self, piece: &str) -> Result<Vec<u32>> {
        if let Some(token) = self.vocab_map.get(piece) {
            return Ok(vec![*token]);
        }
        let mut output = Vec::new();
        for byte in piece.as_bytes() {
            let token = self
                .byte_fallback(*byte)
                .or(self.special.unk)
                .ok_or_else(|| {
                    XrtError::Tokenizer(format!("unable to encode byte 0x{byte:02x}"))
                })?;
            output.push(token);
        }
        Ok(output)
    }

    fn byte_fallback(&self, byte: u8) -> Option<u32> {
        let key = format!("<0x{byte:02X}>");
        self.vocab_map.get(&key).copied()
    }

    fn match_special_prefix<'a>(&'a self, input: &'a str) -> Option<(&'a str, u32)> {
        self.special_by_piece
            .iter()
            .filter_map(|(piece, token)| {
                input.starts_with(piece).then_some((piece.as_str(), *token))
            })
            .max_by_key(|(piece, _)| piece.len())
    }

    fn next_special_boundary(&self, text: &str, start: usize) -> Option<usize> {
        self.special_by_piece
            .keys()
            .filter_map(|piece| text[start..].find(piece).map(|offset| start + offset))
            .min()
    }
}

fn normalize_piece_segment(segment: &str) -> String {
    let mut normalized = String::with_capacity(segment.len() + 1);
    normalized.push('▁');
    for ch in segment.chars() {
        if ch == ' ' {
            normalized.push('▁');
        } else {
            normalized.push(ch);
        }
    }
    normalized
}

fn char_positions(input: &str) -> Vec<usize> {
    input
        .char_indices()
        .map(|(index, _)| index)
        .chain(std::iter::once(input.len()))
        .collect()
}

fn parse_byte_token(piece: &str) -> Option<u8> {
    if piece.len() != 6 || !piece.starts_with("<0x") || !piece.ends_with('>') {
        return None;
    }
    u8::from_str_radix(&piece[3..5], 16).ok()
}

/// GPT-2 byte-to-unicode mapping: maps each byte to a printable unicode character.
/// Bytes 33-126, 161-172, 174-255 map to their codepoint directly.
/// Remaining bytes (0-32, 127-160, 173) map to codepoints starting at 256.
fn byte_to_unicode(byte: u8) -> char {
    gpt2_byte_table()[byte as usize]
}

/// Reverse GPT-2 unicode-to-byte mapping for decoding.
fn unicode_to_byte(ch: char) -> Option<u8> {
    let cp = ch as u32;
    match cp {
        33..=126 | 161..=172 | 174..=255 => Some(cp as u8),
        256..=323 => gpt2_reverse_table().get(&cp).copied(),
        _ => None,
    }
}

fn gpt2_byte_table() -> &'static [char; 256] {
    static TABLE: std::sync::OnceLock<[char; 256]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = ['\0'; 256];
        let mut next = 256u32;
        for b in 0u16..256 {
            let byte = b as u8;
            match byte {
                33..=126 | 161..=172 | 174..=255 => {
                    table[b as usize] = char::from(byte);
                }
                _ => {
                    table[b as usize] = char::from_u32(next).unwrap();
                    next += 1;
                }
            }
        }
        table
    })
}

fn gpt2_reverse_table() -> &'static HashMap<u32, u8> {
    static TABLE: std::sync::OnceLock<HashMap<u32, u8>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut map = HashMap::new();
        let mut next = 256u32;
        for b in 0u8..=255 {
            match b {
                33..=126 | 161..=172 | 174..=255 => {}
                _ => {
                    map.insert(next, b);
                    next += 1;
                }
            }
        }
        map
    })
}
