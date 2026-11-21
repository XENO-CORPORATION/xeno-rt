//! Chunk limits are the fix for the measured long-text collapse, so they are
//! pinned against the real Egypt-style narration shape: every chunk must stay
//! under the ceiling, nothing may be dropped or reordered, and fragments below
//! the floor must be merged.

use xrt_audio::chunking::{chunk_script, ChunkLimits};
use xrt_audio::tokenizer::ChatterboxTokenizer;

fn tokenizer() -> Option<ChatterboxTokenizer> {
    let dir = std::env::var_os("XRT_AUDIO_MODEL_DIR")?;
    Some(
        ChatterboxTokenizer::from_file(&std::path::Path::new(&dir).join("tokenizer.json")).unwrap(),
    )
}

const SCRIPT: &str = "A balance beam tilts as a jackal-headed attendant steadies it, and a human heart hangs above a waiting beast.

At Osiris's court, the dead person must speak while the evidence remains inside his own chest.

He says what he has never done.

\"I have not stolen.\" \"I have not killed.\" Then the heart faces the scales.

Egyptians did not confess their sins in this scene.

The familiar name, the Negative Confession, sends us toward a bowed sinner admitting private failures.

But Spell One Hundred and Twenty-Five stages something else.

It shows a person claiming that they lived in a way worthy of the world's order.

The Egyptians called that order Ma'at.

Ma'at meant truth, justice, balance, and the proper shape of life.

A feather could carry all that weight.";

fn paragraphs() -> impl Iterator<Item = &'static str> {
    SCRIPT
        .split("\n\n")
        .map(str::trim)
        .filter(|p| !p.is_empty())
}

fn words(s: &str) -> Vec<String> {
    s.split_whitespace().map(str::to_string).collect()
}

#[test]
fn respects_limits_and_keeps_every_word_in_order() {
    let Some(tok) = tokenizer() else {
        eprintln!("SKIPPED: XRT_AUDIO_MODEL_DIR not set");
        return;
    };
    let limits = ChunkLimits {
        target_tokens: 120,
        max_tokens: 180,
        min_tokens: 40,
    };
    let chunks = chunk_script(SCRIPT, "en", &tok, limits).unwrap();
    assert!(
        chunks.len() >= 3,
        "expected several chunks, got {}",
        chunks.len()
    );
    for c in &chunks {
        assert!(
            c.tokens <= limits.max_tokens,
            "chunk over ceiling: {} tokens",
            c.tokens
        );
        assert_eq!(
            c.tokens,
            tok.count(&c.text, "en").unwrap(),
            "reported token count is real"
        );
    }
    let joined: Vec<String> = chunks.iter().flat_map(|c| words(&c.text)).collect();
    assert_eq!(
        joined,
        words(SCRIPT),
        "no word dropped, duplicated or reordered"
    );
    assert!(chunks.last().unwrap().ends_paragraph);
    for c in &chunks {
        assert_eq!(c.parts.join(" "), c.text, "parts reassemble the chunk");
    }
    // Parts rejoin into exactly the source paragraphs: a chunk that does not
    // end its paragraph is continued by the next chunk's first part. So no
    // paragraph break is lost by packing, and none is invented by splitting.
    let paragraphs: Vec<String> = paragraphs()
        .map(|p| p.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect();
    let mut rebuilt: Vec<String> = Vec::new();
    let mut continues = false;
    for c in &chunks {
        for (k, part) in c.parts.iter().enumerate() {
            match rebuilt.last_mut() {
                Some(last) if k == 0 && continues => {
                    last.push(' ');
                    last.push_str(part);
                }
                _ => rebuilt.push(part.clone()),
            }
        }
        continues = !c.ends_paragraph;
    }
    assert_eq!(rebuilt, paragraphs);
}

#[test]
fn packed_paragraphs_stay_separate_parts() {
    let Some(tok) = tokenizer() else {
        eprintln!("SKIPPED: XRT_AUDIO_MODEL_DIR not set");
        return;
    };
    let chunks = chunk_script(SCRIPT, "en", &tok, ChunkLimits::default()).unwrap();
    let parts: usize = chunks.iter().map(|c| c.parts.len()).sum();
    assert_eq!(
        parts,
        paragraphs().count(),
        "one part per packed paragraph: {:?}",
        chunks.iter().map(|c| c.parts.len()).collect::<Vec<_>>()
    );
}

#[test]
fn short_fragments_are_merged() {
    let Some(tok) = tokenizer() else {
        eprintln!("SKIPPED: XRT_AUDIO_MODEL_DIR not set");
        return;
    };
    // The script is eleven one-line paragraphs, several of them a handful of
    // words: exactly the pieces that went silent or duplicated in testing.
    let limits = ChunkLimits::default();
    let chunks = chunk_script(SCRIPT, "en", &tok, limits).unwrap();
    let sizes: Vec<usize> = chunks.iter().map(|c| c.tokens).collect();
    assert!(
        chunks.len() <= 2,
        "one-line paragraphs must be packed, got {sizes:?}"
    );
    assert!(
        chunks.iter().all(|c| c.tokens >= limits.min_tokens),
        "fragment under the floor: {sizes:?}"
    );
}

#[test]
fn an_overlong_sentence_is_split_under_the_ceiling() {
    let Some(tok) = tokenizer() else {
        eprintln!("SKIPPED: XRT_AUDIO_MODEL_DIR not set");
        return;
    };
    let long = std::iter::repeat("the weighing of the heart, before the forty-two assessors")
        .take(30)
        .collect::<Vec<_>>()
        .join(", ")
        + ".";
    let limits = ChunkLimits::default();
    let chunks = chunk_script(&long, "en", &tok, limits).unwrap();
    assert!(chunks.len() > 1);
    assert!(chunks.iter().all(|c| c.tokens <= limits.max_tokens));
    let joined: Vec<String> = chunks.iter().flat_map(|c| words(&c.text)).collect();
    assert_eq!(joined, words(&long));
}
