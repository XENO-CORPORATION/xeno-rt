pub mod chat_template;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    path::Path,
};
use xrt_core::{Result, XrtError};
use xrt_gguf::GgufFile;

pub use chat_template::{
    apply_chat_template, apply_chat_template_with_thinking, ChatMessage, CHATML_TEMPLATE,
};

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
    chat_template: Option<String>,
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

        for (index, piece) in vocab.iter().enumerate() {
            if looks_like_special_piece(piece) {
                special_by_piece
                    .entry(piece.clone())
                    .or_insert(index as u32);
                special_ids.insert(index as u32);
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

        let chat_template = gguf
            .metadata_string("tokenizer.chat_template")
            .map(|s| s.to_owned());

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
            chat_template,
        })
    }

    pub fn from_hf_dir(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        if !root.is_dir() {
            return Err(XrtError::Tokenizer(format!(
                "Hugging Face tokenizer path must be a directory, got `{}`",
                root.display()
            )));
        }
        let vocab_path = root.join("vocab.json");
        let vocab_bytes = read_bounded_tokenizer_file(&vocab_path, 64 * 1024 * 1024)?;
        let base_vocab: HashMap<String, u32> =
            serde_json::from_slice(&vocab_bytes).map_err(|err| {
                XrtError::Tokenizer(format!(
                    "failed to parse Hugging Face vocab `{}`: {err}",
                    vocab_path.display()
                ))
            })?;
        if base_vocab.is_empty() {
            return Err(XrtError::Tokenizer(format!(
                "Hugging Face vocab `{}` is empty",
                vocab_path.display()
            )));
        }

        let tokenizer_config = read_optional_json(&root.join("tokenizer_config.json"))?
            .unwrap_or_else(|| Value::Object(Default::default()));
        let model_config = read_optional_json(&root.join("config.json"))?
            .unwrap_or_else(|| Value::Object(Default::default()));
        let added_tokens = read_optional_json(&root.join("added_tokens.json"))?;

        let mut tokens_by_id = BTreeMap::new();
        for (piece, id) in base_vocab {
            insert_hf_token(&mut tokens_by_id, id, piece, "vocab.json")?;
        }
        if let Some(added_tokens) = added_tokens {
            let object = added_tokens.as_object().ok_or_else(|| {
                XrtError::Tokenizer("added_tokens.json must be a JSON object".to_string())
            })?;
            for (piece, id) in object {
                let id = json_u32(id, "added_tokens.json token id")?;
                insert_hf_token(&mut tokens_by_id, id, piece.clone(), "added_tokens.json")?;
            }
        }

        let mut explicitly_special = HashSet::new();
        if let Some(decoder) = tokenizer_config
            .get("added_tokens_decoder")
            .and_then(Value::as_object)
        {
            for (id, record) in decoder {
                let id = id.parse::<u32>().map_err(|_| {
                    XrtError::Tokenizer(format!(
                        "tokenizer_config.json added_tokens_decoder key `{id}` is not a u32"
                    ))
                })?;
                let record = record.as_object().ok_or_else(|| {
                    XrtError::Tokenizer(format!(
                        "tokenizer_config.json added token {id} must be an object"
                    ))
                })?;
                let piece = record
                    .get("content")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        XrtError::Tokenizer(format!(
                            "tokenizer_config.json added token {id} is missing content"
                        ))
                    })?
                    .to_string();
                insert_hf_token(
                    &mut tokens_by_id,
                    id,
                    piece,
                    "tokenizer_config.json added_tokens_decoder",
                )?;
                if record
                    .get("special")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    explicitly_special.insert(id);
                }
            }
        }

        let maximum_id = tokens_by_id.keys().next_back().copied().ok_or_else(|| {
            XrtError::Tokenizer("Hugging Face tokenizer has no token ids".to_string())
        })?;
        let vocab_len = (maximum_id as usize).checked_add(1).ok_or_else(|| {
            XrtError::Tokenizer("Hugging Face tokenizer vocab size overflows usize".to_string())
        })?;
        let mut vocab = vec![None; vocab_len];
        for (id, piece) in tokens_by_id {
            vocab[id as usize] = Some(piece);
        }
        let vocab = vocab
            .into_iter()
            .enumerate()
            .map(|(id, piece)| {
                piece.ok_or_else(|| {
                    XrtError::Tokenizer(format!(
                        "Hugging Face tokenizer vocabulary is missing token id {id}"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if let Some(expected) = model_config.get("vocab_size") {
            let expected = json_u32(expected, "config.json vocab_size")? as usize;
            if vocab.len() > expected {
                return Err(XrtError::Tokenizer(format!(
                    "loaded tokenizer size {} exceeds config.json vocab_size {expected}",
                    vocab.len()
                )));
            }
        }
        let mut vocab_map = HashMap::with_capacity(vocab.len());
        for (id, piece) in vocab.iter().enumerate() {
            if let Some(previous) = vocab_map.insert(piece.clone(), id as u32) {
                if previous != id as u32 {
                    return Err(XrtError::Tokenizer(format!(
                        "Hugging Face tokenizer piece `{piece}` maps to both {previous} and {id}"
                    )));
                }
            }
        }

        let merges_path = root.join("merges.txt");
        let merges_text =
            String::from_utf8(read_bounded_tokenizer_file(&merges_path, 64 * 1024 * 1024)?)
                .map_err(|err| {
                    XrtError::Tokenizer(format!(
                        "Hugging Face merges `{}` are not UTF-8: {err}",
                        merges_path.display()
                    ))
                })?;
        let mut merges = HashMap::new();
        for line in merges_text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut parts = line.split_whitespace();
            let left = parts.next().ok_or_else(|| {
                XrtError::Tokenizer(format!(
                    "invalid empty merge line in `{}`",
                    merges_path.display()
                ))
            })?;
            let right = parts.next().ok_or_else(|| {
                XrtError::Tokenizer(format!(
                    "merge line `{line}` in `{}` is missing its right token",
                    merges_path.display()
                ))
            })?;
            if parts.next().is_some() {
                return Err(XrtError::Tokenizer(format!(
                    "merge line `{line}` in `{}` has more than two tokens",
                    merges_path.display()
                )));
            }
            let rank = merges.len();
            if merges
                .insert((left.to_string(), right.to_string()), rank)
                .is_some()
            {
                return Err(XrtError::Tokenizer(format!(
                    "duplicate merge `{left} {right}` in `{}`",
                    merges_path.display()
                )));
            }
        }
        if merges.is_empty() {
            return Err(XrtError::Tokenizer(format!(
                "Hugging Face merges `{}` contain no BPE rules",
                merges_path.display()
            )));
        }

        let special = SpecialTokens {
            bos: hf_special_id(
                &model_config,
                &tokenizer_config,
                "bos_token_id",
                "bos_token",
                &vocab_map,
            )?,
            eos: hf_special_id(
                &model_config,
                &tokenizer_config,
                "eos_token_id",
                "eos_token",
                &vocab_map,
            )?,
            unk: hf_special_id(
                &model_config,
                &tokenizer_config,
                "unk_token_id",
                "unk_token",
                &vocab_map,
            )?,
            pad: hf_special_id(
                &model_config,
                &tokenizer_config,
                "pad_token_id",
                "pad_token",
                &vocab_map,
            )?,
            add_bos: tokenizer_config
                .get("add_bos_token")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            add_eos: tokenizer_config
                .get("add_eos_token")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        };
        let mut special_by_piece = HashMap::new();
        let mut special_ids = explicitly_special;
        for id in [special.bos, special.eos, special.unk, special.pad]
            .into_iter()
            .flatten()
        {
            special_ids.insert(id);
        }
        for (id, piece) in vocab.iter().enumerate() {
            let id = id as u32;
            if special_ids.contains(&id) || looks_like_special_piece(piece) {
                special_ids.insert(id);
                special_by_piece.insert(piece.clone(), id);
            }
        }
        let max_piece_chars = vocab
            .iter()
            .map(|token| token.chars().count())
            .max()
            .unwrap_or(1);
        let chat_template = hf_chat_template(&tokenizer_config)?;

        Ok(Self {
            scores: vec![0.0; vocab.len()],
            vocab,
            vocab_map,
            merges,
            kind: TokenizerKind::Gpt2Bpe,
            special,
            special_by_piece,
            special_ids,
            max_piece_chars,
            chat_template,
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

    pub fn token_id_for_piece(&self, piece: &str) -> Option<u32> {
        self.vocab_map.get(piece).copied()
    }

    /// Returns the raw Jinja2 chat template from GGUF metadata, if present.
    pub fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }

    /// Formats chat messages using the model's chat template (or ChatML fallback).
    pub fn format_chat(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String> {
        self.format_chat_with_thinking(messages, add_generation_prompt, None)
    }

    /// Formats chat messages while optionally selecting a model-native
    /// thinking mode. `None` preserves the model template's default.
    pub fn format_chat_with_thinking(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        enable_thinking: Option<bool>,
    ) -> Result<String> {
        let template = self.chat_template.as_deref().unwrap_or(CHATML_TEMPLATE);
        let bos = self
            .special
            .bos
            .and_then(|id| self.vocab.get(id as usize))
            .map(|s| s.as_str())
            .unwrap_or("");
        let eos = self
            .special
            .eos
            .and_then(|id| self.vocab.get(id as usize))
            .map(|s| s.as_str())
            .unwrap_or("");
        apply_chat_template_with_thinking(
            template,
            messages,
            bos,
            eos,
            add_generation_prompt,
            enable_thinking,
        )
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
            return self.decode_gpt2(tokens, skip_special, false);
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

    pub fn decode_lossy(&self, tokens: &[u32], skip_special: bool) -> Result<String> {
        if self.kind == TokenizerKind::Gpt2Bpe {
            return self.decode_gpt2(tokens, skip_special, true);
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
                output.push_str(&String::from_utf8_lossy(&pending_bytes));
                pending_bytes.clear();
            }
            output.push_str(&piece.replace('\u{2581}', " "));
        }

        if !pending_bytes.is_empty() {
            output.push_str(&String::from_utf8_lossy(&pending_bytes));
        }

        Ok(output)
    }

    fn decode_gpt2(&self, tokens: &[u32], skip_special: bool, lossy: bool) -> Result<String> {
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
        if lossy {
            Ok(String::from_utf8_lossy(&bytes).into_owned())
        } else {
            String::from_utf8(bytes)
                .map_err(|e| XrtError::Tokenizer(format!("invalid utf8 in decode: {e}")))
        }
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
        let unicode_str: String = segment
            .as_bytes()
            .iter()
            .map(|&b| byte_to_unicode(b))
            .collect();

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

fn read_bounded_tokenizer_file(path: &Path, max_bytes: u64) -> Result<Vec<u8>> {
    let length = fs::metadata(path)?.len();
    if length > max_bytes {
        return Err(XrtError::Tokenizer(format!(
            "tokenizer file `{}` is {length} bytes, above the {max_bytes}-byte limit",
            path.display()
        )));
    }
    Ok(fs::read(path)?)
}

fn read_optional_json(path: &Path) -> Result<Option<Value>> {
    if !path.is_file() {
        return Ok(None);
    }
    let bytes = read_bounded_tokenizer_file(path, 64 * 1024 * 1024)?;
    serde_json::from_slice(&bytes).map(Some).map_err(|err| {
        XrtError::Tokenizer(format!(
            "failed to parse tokenizer JSON `{}`: {err}",
            path.display()
        ))
    })
}

fn insert_hf_token(
    tokens: &mut BTreeMap<u32, String>,
    id: u32,
    piece: String,
    source: &str,
) -> Result<()> {
    if let Some(existing) = tokens.get(&id) {
        if existing != &piece {
            return Err(XrtError::Tokenizer(format!(
                "Hugging Face token id {id} is `{existing}` in an earlier source but `{piece}` in {source}"
            )));
        }
        return Ok(());
    }
    tokens.insert(id, piece);
    Ok(())
}

fn json_u32(value: &Value, field: &str) -> Result<u32> {
    let value = value
        .as_u64()
        .ok_or_else(|| XrtError::Tokenizer(format!("{field} must be an unsigned integer")))?;
    u32::try_from(value).map_err(|_| XrtError::Tokenizer(format!("{field} exceeds u32: {value}")))
}

fn hf_special_id(
    model_config: &Value,
    tokenizer_config: &Value,
    id_key: &str,
    piece_key: &str,
    vocab: &HashMap<String, u32>,
) -> Result<Option<u32>> {
    if let Some(value) = model_config.get(id_key) {
        let value = match value {
            Value::Array(values) => values.first(),
            value => Some(value),
        };
        if let Some(value) = value {
            return json_u32(value, &format!("config.json {id_key}")).map(Some);
        }
    }
    let Some(value) = tokenizer_config.get(piece_key) else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let piece = match value {
        Value::String(piece) => piece.as_str(),
        Value::Object(object) => {
            object
                .get("content")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    XrtError::Tokenizer(format!(
                        "tokenizer_config.json {piece_key} object is missing content"
                    ))
                })?
        }
        _ => {
            return Err(XrtError::Tokenizer(format!(
                "tokenizer_config.json {piece_key} must be a string, object, or null"
            )))
        }
    };
    vocab.get(piece).copied().map(Some).ok_or_else(|| {
        XrtError::Tokenizer(format!(
            "tokenizer_config.json {piece_key} `{piece}` is absent from the vocabulary"
        ))
    })
}

fn hf_chat_template(tokenizer_config: &Value) -> Result<Option<String>> {
    let Some(value) = tokenizer_config.get("chat_template") else {
        return Ok(None);
    };
    match value {
        Value::Null => Ok(None),
        Value::String(template) => Ok(Some(template.clone())),
        Value::Array(templates) => {
            let mut fallback = None;
            for template in templates {
                let object = template.as_object().ok_or_else(|| {
                    XrtError::Tokenizer(
                        "tokenizer_config.json chat_template entries must be objects".to_string(),
                    )
                })?;
                let name = object.get("name").and_then(Value::as_str).unwrap_or("");
                let template = object
                    .get("template")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        XrtError::Tokenizer(
                            "tokenizer_config.json chat_template entry is missing template"
                                .to_string(),
                        )
                    })?;
                if name == "default" {
                    return Ok(Some(template.to_string()));
                }
                fallback.get_or_insert_with(|| template.to_string());
            }
            Ok(fallback)
        }
        _ => Err(XrtError::Tokenizer(
            "tokenizer_config.json chat_template must be a string, array, or null".to_string(),
        )),
    }
}

fn looks_like_special_piece(piece: &str) -> bool {
    let trimmed = piece.trim();
    if trimmed.is_empty() {
        return false;
    }

    if trimmed.starts_with("<|") && trimmed.ends_with("|>") {
        return true;
    }

    if trimmed.starts_with('<') && trimmed.ends_with('>') && !trimmed.contains(' ') {
        let inner = trimmed
            .trim_start_matches('<')
            .trim_end_matches('>')
            .to_ascii_lowercase();
        return [
            "im_",
            "text",
            "think",
            "tool",
            "assistant",
            "user",
            "system",
            "response",
            "start",
            "end",
            "eot",
            "eom",
            "eos",
            "bos",
            "pad",
            "mask",
            "gmask",
            "sop",
            "eop",
            "vision",
            "image",
            "audio",
            "video",
        ]
        .iter()
        .any(|needle| inner.contains(needle));
    }

    false
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

#[cfg(test)]
mod tests {
    use super::*;

    fn write_hf_tokenizer(root: &Path, vocab: &str) {
        fs::write(root.join("vocab.json"), vocab).unwrap();
        fs::write(root.join("merges.txt"), "#version: 0.2\nh i\n").unwrap();
        fs::write(
            root.join("config.json"),
            r#"{"vocab_size":4,"eos_token_id":3}"#,
        )
        .unwrap();
        fs::write(
            root.join("tokenizer_config.json"),
            r#"{
                "add_bos_token": false,
                "add_eos_token": false,
                "eos_token": "<|end|>",
                "chat_template": "{{ messages }}",
                "added_tokens_decoder": {
                    "3": {"content":"<|end|>","special":true}
                }
            }"#,
        )
        .unwrap();
    }

    #[test]
    fn hf_bpe_loader_preserves_ids_merges_and_special_tokens() {
        let directory = tempfile::tempdir().unwrap();
        write_hf_tokenizer(directory.path(), r#"{"h":0,"i":1,"hi":2}"#);
        let tokenizer = Tokenizer::from_hf_dir(directory.path()).unwrap();
        assert_eq!(tokenizer.vocab_size(), 4);
        assert_eq!(tokenizer.chat_template(), Some("{{ messages }}"));
        assert_eq!(
            tokenizer
                .encode_with_options("hi<|end|>", false, true)
                .unwrap(),
            vec![2, 3]
        );
        assert_eq!(tokenizer.decode(&[2, 3], true).unwrap(), "hi");
    }

    #[test]
    fn hf_bpe_loader_rejects_sparse_token_ids() {
        let directory = tempfile::tempdir().unwrap();
        write_hf_tokenizer(directory.path(), r#"{"h":0,"hi":2}"#);
        let error = Tokenizer::from_hf_dir(directory.path()).unwrap_err();
        assert!(error.to_string().contains("missing token id 1"));
    }

    #[test]
    #[ignore = "requires XRT_REAL_HF_MODEL_DIR and XRT_REAL_GGUF"]
    fn real_hf_tokenizer_matches_the_equivalent_gguf_tokenizer() {
        let hf_root = std::env::var_os("XRT_REAL_HF_MODEL_DIR")
            .map(std::path::PathBuf::from)
            .expect("XRT_REAL_HF_MODEL_DIR must point to the Hugging Face model directory");
        let gguf_path = std::env::var_os("XRT_REAL_GGUF")
            .map(std::path::PathBuf::from)
            .expect("XRT_REAL_GGUF must point to the equivalent GGUF model");
        let hf = Tokenizer::from_hf_dir(hf_root).unwrap();
        let gguf_file = GgufFile::open(gguf_path).unwrap();
        let gguf = Tokenizer::from_gguf(&gguf_file).unwrap();
        assert!(hf.vocab_size() <= gguf.vocab_size());
        for prompt in [
            "Hello",
            "Hello world! This is a tokenizer parity test.",
            "<|im_start|>system\nYou are concise.<|im_end|>\n<|im_start|>user\nHello<|im_end|>",
            "fn main() { println!(\"hello\"); }",
        ] {
            assert_eq!(
                hf.encode_with_options(prompt, false, true).unwrap(),
                gguf.encode_with_options(prompt, false, true).unwrap(),
                "token mismatch for prompt {prompt:?}"
            );
        }
    }
}
