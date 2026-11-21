//! Split a script into pieces the model can read reliably.
//!
//! Measured (Chatterbox v3, ASR-scored, 3 seeds per length):
//! - < ~120 chars: unstable — silent, duplicated or hallucinated lines.
//! - 390–570 chars: 0 failures.
//! - 914+: garbles, because the learned text-position table is only trained to
//!   ~600 positions (neighbour cosine ≈ 0 from ~650 on).
//!
//! So the budget is in TOKENS (what the model's positions count), with a
//! target, a hard ceiling kept well inside the trained range, and a floor that
//! merges short fragments into a neighbour. Boundaries prefer, in order:
//! paragraph, sentence, clause, word. A paragraph break is remembered so the
//! stitcher can give it a longer pause.

use crate::tokenizer::ChatterboxTokenizer;
use crate::AudioError;

#[derive(Debug, Clone, Copy)]
pub struct ChunkLimits {
    /// Aim for this many text tokens per chunk.
    pub target_tokens: usize,
    /// Never exceed this. Kept below the ~600 trained positions.
    pub max_tokens: usize,
    /// Chunks shorter than this are merged into a neighbour when possible.
    pub min_tokens: usize,
}

impl Default for ChunkLimits {
    fn default() -> Self {
        // ≈ 350 tokens ≈ 560 English chars (the measured sweet spot);
        // 480 ≈ 770 chars, still ~120 positions inside the trained range.
        Self {
            target_tokens: 350,
            max_tokens: 480,
            min_tokens: 75,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Chunk {
    pub text: String,
    pub tokens: usize,
    /// True when this chunk ends a paragraph in the source.
    pub ends_paragraph: bool,
    /// The paragraphs (or paragraph pieces) this chunk holds, in order;
    /// `text == parts.join(" ")`. Short paragraphs are packed into one chunk,
    /// so a paragraph break can fall INSIDE a chunk, and the pipeline needs
    /// to know where to give it a paragraph pause.
    pub parts: Vec<String>,
}

impl Chunk {
    /// A single-part chunk (tests, tools).
    pub fn new(text: impl Into<String>, tokens: usize, ends_paragraph: bool) -> Self {
        let text = text.into();
        Self {
            parts: vec![text.clone()],
            text,
            tokens,
            ends_paragraph,
        }
    }
}

/// Split `script` into chunks. Every chunk is ≤ `max_tokens`; a single
/// sentence longer than that is split at a clause, then a word, boundary.
pub fn chunk_script(
    script: &str,
    language: &str,
    tok: &ChatterboxTokenizer,
    limits: ChunkLimits,
) -> Result<Vec<Chunk>, AudioError> {
    if limits.min_tokens > limits.target_tokens || limits.target_tokens > limits.max_tokens {
        return Err(AudioError::InvalidRequest(
            "chunk limits must satisfy min <= target <= max".into(),
        ));
    }
    let count = |s: &str| tok.count(s, language);

    // 1. Units: sentences, each tagged with whether it closes a paragraph.
    let mut units: Vec<(String, bool)> = Vec::new();
    for para in split_paragraphs(script) {
        let para = para.split_whitespace().collect::<Vec<_>>().join(" ");
        let sentences = split_sentences(&para);
        let n = sentences.len();
        for (i, s) in sentences.into_iter().enumerate() {
            for piece in split_to_fit(&s, limits.max_tokens, &count)? {
                units.push((piece, false));
            }
            if i + 1 == n {
                if let Some(last) = units.last_mut() {
                    last.1 = true;
                }
            }
        }
    }
    if units.is_empty() {
        return Err(AudioError::InvalidRequest("script contains no text".into()));
    }

    // 2. Greedy pack toward the target; close early at a paragraph end once
    //    past the floor, so chunk edges land on natural pauses.
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut cur = String::new();
    let mut parts: Vec<String> = Vec::new();
    let mut cur_para = false;
    // Append `unit` to the open parts: a new part after a paragraph end.
    let push_part = |parts: &mut Vec<String>, unit: &str, new_para: bool| match parts.last_mut() {
        Some(last) if !new_para => {
            last.push(' ');
            last.push_str(unit);
        }
        _ => parts.push(unit.to_string()),
    };
    for (unit, ends_para) in units {
        let candidate = if cur.is_empty() {
            unit.clone()
        } else {
            format!("{cur} {unit}")
        };
        let n = count(&candidate)?;
        if !cur.is_empty() && n > limits.target_tokens {
            let t = count(&cur)?;
            chunks.push(Chunk {
                text: std::mem::take(&mut cur),
                tokens: t,
                ends_paragraph: cur_para,
                parts: std::mem::take(&mut parts),
            });
            cur = unit.clone();
            parts.push(unit);
        } else {
            push_part(&mut parts, &unit, cur_para);
            cur = candidate;
        }
        cur_para = ends_para;
        if ends_para && count(&cur)? >= limits.target_tokens * 3 / 4 {
            let t = count(&cur)?;
            chunks.push(Chunk {
                text: std::mem::take(&mut cur),
                tokens: t,
                ends_paragraph: true,
                parts: std::mem::take(&mut parts),
            });
        }
    }
    if !cur.is_empty() {
        let t = count(&cur)?;
        chunks.push(Chunk {
            text: cur,
            tokens: t,
            ends_paragraph: true,
            parts,
        });
    }

    // 3. Merge any chunk under the floor into its smaller neighbour, if the
    //    result still fits under the ceiling.
    let mut i = 0;
    while i < chunks.len() {
        if chunks[i].tokens < limits.min_tokens && chunks.len() > 1 {
            let left = (i > 0).then(|| chunks[i - 1].tokens);
            let right = chunks.get(i + 1).map(|c| c.tokens);
            let into_left = match (left, right) {
                (Some(l), Some(r)) => l <= r,
                (Some(_), None) => true,
                _ => false,
            };
            let (a, b) = if into_left { (i - 1, i) } else { (i, i + 1) };
            let text = format!("{} {}", chunks[a].text, chunks[b].text);
            let n = count(&text)?;
            if n <= limits.max_tokens {
                let ends_paragraph = chunks[b].ends_paragraph;
                let mut parts = chunks[a].parts.clone();
                let mut rest = chunks[b].parts.clone().into_iter();
                if !chunks[a].ends_paragraph {
                    if let (Some(last), Some(first)) = (parts.last_mut(), rest.next()) {
                        last.push(' ');
                        last.push_str(&first);
                    }
                }
                parts.extend(rest);
                chunks[a] = Chunk {
                    text,
                    tokens: n,
                    ends_paragraph,
                    parts,
                };
                chunks.remove(b);
                i = a;
                continue;
            }
        }
        i += 1;
    }
    Ok(chunks)
}

/// Paragraphs are separated by a line holding nothing but whitespace, in any
/// line-ending convention. Splitting on a literal "\n\n" missed every break
/// in a Windows-authored script ("\r\n\r\n") and so gave it no paragraph
/// pauses at all — found 2026-09-28 on a real script file.
fn split_paragraphs(script: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for line in script.lines() {
        if line.trim().is_empty() {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
        } else {
            if !cur.is_empty() {
                cur.push(' ');
            }
            cur.push_str(line.trim());
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Split on sentence enders followed by whitespace, keeping the ender and any
/// closing quote with the sentence.
fn split_sentences(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i < chars.len() {
        if matches!(chars[i], '.' | '!' | '?' | '。' | '！' | '？') {
            let mut j = i + 1;
            while j < chars.len()
                && matches!(chars[j], '"' | '\'' | '”' | '’' | ')' | '.' | '!' | '?')
            {
                j += 1;
            }
            if j >= chars.len() || chars[j].is_whitespace() {
                let s: String = chars[start..j]
                    .iter()
                    .collect::<String>()
                    .trim()
                    .to_string();
                if !s.is_empty() {
                    out.push(s);
                }
                start = j;
                i = j;
                continue;
            }
        }
        i += 1;
    }
    let tail: String = chars[start..].iter().collect::<String>().trim().to_string();
    if !tail.is_empty() {
        out.push(tail);
    }
    out
}

/// Split one over-long sentence at clause boundaries, then words, until each
/// piece fits.
fn split_to_fit(
    s: &str,
    max: usize,
    count: &dyn Fn(&str) -> Result<usize, AudioError>,
) -> Result<Vec<String>, AudioError> {
    if count(s)? <= max {
        return Ok(vec![s.to_string()]);
    }
    for sep in [", ", "; ", ": ", " — ", " - "] {
        let parts: Vec<&str> = s.split(sep).collect();
        if parts.len() > 1 {
            let mut out: Vec<String> = Vec::new();
            let mut cur = String::new();
            for (k, p) in parts.iter().enumerate() {
                let piece = if k + 1 < parts.len() {
                    format!("{p}{}", sep.trim_end())
                } else {
                    p.to_string()
                };
                let cand = if cur.is_empty() {
                    piece.clone()
                } else {
                    format!("{cur} {piece}")
                };
                if !cur.is_empty() && count(&cand)? > max {
                    out.push(std::mem::take(&mut cur));
                    cur = piece;
                } else {
                    cur = cand;
                }
            }
            if !cur.is_empty() {
                out.push(cur);
            }
            if out.len() > 1 {
                let mut fitted = Vec::new();
                for o in out {
                    fitted.extend(split_to_fit(&o, max, count)?);
                }
                return Ok(fitted);
            }
        }
    }
    // Last resort: word boundary, halving.
    let words: Vec<&str> = s.split(' ').collect();
    if words.len() < 2 {
        return Err(AudioError::InvalidRequest(format!(
            "a single word exceeds the {max}-token chunk limit"
        )));
    }
    let mid = words.len() / 2;
    let mut out = split_to_fit(&words[..mid].join(" "), max, count)?;
    out.extend(split_to_fit(&words[mid..].join(" "), max, count)?);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sentences_keep_closing_quotes() {
        let s =
            split_sentences("He said \"I have not stolen.\" Then the heart faces the scales. Done");
        assert_eq!(
            s,
            vec![
                "He said \"I have not stolen.\"",
                "Then the heart faces the scales.",
                "Done"
            ]
        );
    }

    #[test]
    fn paragraphs_split_in_any_line_ending() {
        let want = vec![
            "One two.".to_string(),
            "Three four.".to_string(),
            "Five.".to_string(),
        ];
        assert_eq!(split_paragraphs("One two.\n\nThree four.\n\nFive."), want);
        assert_eq!(
            split_paragraphs("One two.\r\n\r\nThree four.\r\n  \r\nFive.\r\n"),
            want
        );
        assert_eq!(
            split_paragraphs("One\r\ntwo.\n\n\n\nThree four.\n\nFive."),
            want
        );
    }

    #[test]
    fn decimals_do_not_split() {
        assert_eq!(
            split_sentences("Version 1.5 shipped. Yes."),
            vec!["Version 1.5 shipped.", "Yes."]
        );
    }
}
