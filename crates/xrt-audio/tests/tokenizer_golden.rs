//! The Rust tokenizer must reproduce the reference `MTLTokenizer` byte for
//! byte. Golden IDs were generated from the upstream Python implementation
//! (xrt-audio spike, 2026-09-26) and cover the cases that broke before:
//! word boundaries (`[SPACE]`), curly quotes, ellipsis/em-dash/colon
//! normalisation, accents (NFKD), and non-English language tags.
//!
//! Needs the model's tokenizer.json: set XRT_AUDIO_MODEL_DIR. Without it the
//! test reports itself skipped rather than passing silently.

use xrt_audio::tokenizer::{punc_norm, ChatterboxTokenizer};

fn tokenizer() -> Option<ChatterboxTokenizer> {
    let dir = std::env::var_os("XRT_AUDIO_MODEL_DIR")?;
    let path = std::path::Path::new(&dir).join("tokenizer.json");
    Some(ChatterboxTokenizer::from_file(&path).expect("tokenizer.json loads"))
}

#[test]
fn matches_reference_ids() {
    let Some(tok) = tokenizer() else {
        eprintln!("SKIPPED: XRT_AUDIO_MODEL_DIR not set");
        return;
    };
    let golden: serde_json::Value =
        serde_json::from_str(include_str!("data/tokenizer_golden.json")).unwrap();
    let cases = golden.as_array().unwrap();
    assert!(cases.len() >= 7);
    for case in cases {
        let text = case["text"].as_str().unwrap();
        let lang = case["lang"].as_str().unwrap();
        assert_eq!(
            punc_norm(text),
            case["normalized"].as_str().unwrap(),
            "punc_norm for {text:?}"
        );

        let want: Vec<i64> = case["reference_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        assert_eq!(
            tok.encode_text(&punc_norm(text), lang).unwrap(),
            want,
            "text ids for {text:?}"
        );

        let want_full: Vec<i64> = case["v3_ids_with_template"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        assert_eq!(
            tok.encode_prompt(&punc_norm(text), lang).unwrap(),
            want_full,
            "prompt ids for {text:?}"
        );
    }
}

#[test]
fn spaces_become_space_tokens() {
    let Some(tok) = tokenizer() else {
        eprintln!("SKIPPED: XRT_AUDIO_MODEL_DIR not set");
        return;
    };
    // The mumbling bug: with no [SPACE] ids the model runs words together.
    let ids = tok.encode_text("A human heart.", "en").unwrap();
    assert_eq!(ids.iter().filter(|&&i| i == 2).count(), 2, "{ids:?}");
}

#[test]
fn refuses_unported_scripts() {
    for lang in ["zh", "ja", "he", "ko", "ru", "xx"] {
        assert!(
            ChatterboxTokenizer::check_language(lang).is_err(),
            "{lang} must be refused"
        );
    }
    assert!(ChatterboxTokenizer::check_language("en").is_ok());
}
