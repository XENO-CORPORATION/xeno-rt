mod common;

use xrt_gguf::GgufFile;
use xrt_tokenizer::Tokenizer;

#[test]
fn encodes_and_decodes_with_expected_pieces() {
    let (fixture, spec) = common::build_tokenizer_fixture().expect("fixture should be created");
    let gguf = GgufFile::open(fixture.path()).expect("GGUF should parse");
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("tokenizer should load");

    assert_eq!(tokenizer.vocab_size(), spec.tokens.len());
    assert_eq!(
        tokenizer.token_to_piece(spec.hello_id),
        Some(spec.tokens[spec.hello_id as usize].as_str())
    );
    assert_eq!(tokenizer.special_tokens().unk, Some(spec.unk_id));

    let encoded = tokenizer
        .encode_with_options("hello world!", false, false)
        .expect("encoding should succeed");
    assert_eq!(encoded, vec![spec.hello_id, spec.world_id, spec.bang_id]);

    let decoded = tokenizer
        .decode(&encoded, true)
        .expect("decoding should succeed");
    assert_eq!(decoded, " hello world!");
    assert_eq!(decoded.trim_start(), "hello world!");
}

#[test]
fn handles_special_tokens_correctly() {
    let (fixture, spec) = common::build_tokenizer_fixture().expect("fixture should be created");
    let gguf = GgufFile::open(fixture.path()).expect("GGUF should parse");
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("tokenizer should load");

    let encoded = tokenizer.encode("hello").expect("encoding should succeed");
    assert_eq!(encoded, vec![spec.bos_id, spec.hello_id, spec.eos_id]);

    let explicit = tokenizer
        .encode_with_options("<s>hello</s>", false, true)
        .expect("special-token encoding should succeed");
    assert_eq!(explicit, vec![spec.bos_id, spec.hello_id, spec.eos_id]);

    let literal = tokenizer
        .encode_with_options("<s>", false, false)
        .expect("literal encoding should succeed");
    assert_eq!(literal, vec![0xE2u32, 0x96, 0x81, spec.bos_id]);
    assert_ne!(literal, vec![spec.bos_id]);

    let literal_text = tokenizer
        .decode(&literal, false)
        .expect("literal decoding should succeed");
    assert_eq!(literal_text, format!("{}<s>", common::SPM_SPACE));

    let with_specials = tokenizer
        .decode(&encoded, false)
        .expect("special decoding should succeed");
    assert_eq!(with_specials, "<s> hello</s>");
}
