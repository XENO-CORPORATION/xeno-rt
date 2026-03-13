mod common;

use xrt_core::{DType, XrtError};
use xrt_gguf::GgufFile;

fn build_minimal_valid_gguf() -> common::GgufFixture {
    common::build_minimal_valid_gguf_fixture().expect("GGUF fixture should be created")
}

#[test]
fn parses_header_metadata_and_tensor_info() {
    let fixture = build_minimal_valid_gguf();
    let gguf = GgufFile::open(fixture.path()).expect("fixture should parse");

    assert_eq!(gguf.header().version, 3);
    assert_eq!(gguf.header().tensor_count, 2);
    assert_eq!(gguf.header().metadata_kv_count, 8);
    assert_eq!(gguf.alignment(), 32);
    assert_eq!(gguf.metadata_string("general.architecture"), Some("llama"));
    assert_eq!(gguf.metadata_string("general.name"), Some("test"));
    assert_eq!(gguf.metadata_string("tokenizer.ggml.model"), Some("llama"));
    assert_eq!(gguf.metadata_usize("general.alignment"), Some(32));

    let tokens = gguf
        .metadata_array("tokenizer.ggml.tokens")
        .expect("tokens metadata should exist")
        .as_strings()
        .expect("tokens metadata should be an array of strings");
    assert_eq!(
        tokens,
        vec!["<unk>", &format!("{}test", common::SPM_SPACE), "!"]
    );

    let tensor_names = gguf.tensor_names().collect::<Vec<_>>();
    assert_eq!(tensor_names, vec!["tok_embeddings.weight", "output.weight"]);

    let embeddings = gguf
        .require_tensor("tok_embeddings.weight")
        .expect("tensor should exist");
    assert_eq!(embeddings.name, "tok_embeddings.weight");
    assert_eq!(embeddings.dimensions, vec![4, 2]);
    assert_eq!(embeddings.strides, vec![1, 4]);
    assert_eq!(embeddings.dtype, DType::F32);
    assert_eq!(embeddings.offset, 0);
    assert_eq!(embeddings.nbytes, 32);
    assert_eq!(embeddings.row_len(), 4);
    assert_eq!(embeddings.rows(), 2);
    assert_eq!(embeddings.numel(), 8);

    let output = gguf
        .require_tensor("output.weight")
        .expect("tensor should exist");
    assert_eq!(output.dimensions, vec![4, 1]);
    assert_eq!(output.strides, vec![1, 4]);
    assert_eq!(output.dtype, DType::F32);
    assert_eq!(output.offset, 32);
    assert_eq!(output.nbytes, 16);

    let data = gguf
        .tensor_data("tok_embeddings.weight")
        .expect("tensor bytes should be accessible");
    let values = data
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("f32 chunk")))
        .collect::<Vec<_>>();
    assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    let view = gguf
        .tensor_view("output.weight")
        .expect("tensor view should be valid");
    assert_eq!(view.shape, &[4, 1]);
    assert_eq!(view.stride, &[1, 4]);
    assert_eq!(view.dtype, DType::F32);
    assert!(view.is_contiguous());
}

#[test]
fn rejects_bad_magic() {
    let mut bytes = build_minimal_valid_gguf().bytes;
    bytes[..4].copy_from_slice(&0u32.to_le_bytes());
    let fixture = common::write_raw_gguf(bytes).expect("fixture should be written");
    let error = GgufFile::open(fixture.path()).expect_err("bad magic should fail");

    match error {
        XrtError::InvalidFormat(message) => {
            assert!(message.contains("invalid GGUF magic"));
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn rejects_truncated_files() {
    let bytes = build_minimal_valid_gguf().bytes[..16].to_vec();
    let fixture = common::write_raw_gguf(bytes).expect("fixture should be written");
    let error = GgufFile::open(fixture.path()).expect_err("truncated file should fail");

    match error {
        XrtError::InvalidFormat(message) => {
            assert!(message.contains("unexpected EOF"));
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn rejects_unsupported_versions() {
    let mut bytes = build_minimal_valid_gguf().bytes;
    bytes[4..8].copy_from_slice(&4u32.to_le_bytes());
    let fixture = common::write_raw_gguf(bytes).expect("fixture should be written");
    let error = GgufFile::open(fixture.path()).expect_err("version 4 should fail");

    match error {
        XrtError::Unsupported(message) => {
            assert!(message.contains("GGUF version 4"));
        }
        other => panic!("unexpected error: {other}"),
    }
}
