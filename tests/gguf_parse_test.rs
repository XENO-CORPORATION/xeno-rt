mod common;

use xrt_core::{DType, XrtError};
use xrt_gguf::{GgufCompatibility, GgufFile, QWEN_IMAGE_EDIT_TIMESTEP_ZERO_MARKER};

fn build_minimal_valid_gguf() -> common::GgufFixture {
    common::build_minimal_valid_gguf_fixture().expect("GGUF fixture should be created")
}

fn build_zero_sized_tensor_gguf(
    architecture: &str,
    name: &str,
    dimensions: Vec<usize>,
    dtype: DType,
) -> common::GgufFixture {
    common::build_gguf_fixture(
        3,
        vec![(
            "general.architecture".to_string(),
            common::MetadataValueSpec::String(architecture.to_string()),
        )],
        vec![common::TensorSpec {
            name: name.to_string(),
            dimensions,
            dtype,
            data: Vec::new(),
        }],
    )
    .expect("zero-sized GGUF fixture should be created")
}

#[test]
fn parses_header_metadata_and_tensor_info() {
    let fixture = build_minimal_valid_gguf();
    let gguf = GgufFile::open(fixture.path()).expect("fixture should parse");

    assert_eq!(gguf.header().version, 3);
    assert_eq!(gguf.header().tensor_count, 2);
    assert_eq!(gguf.header().metadata_kv_count, 11);
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
    let bools = gguf
        .metadata_array("test.bool_array")
        .expect("bool array should exist")
        .as_bool_vec()
        .expect("bool array should parse");
    assert_eq!(bools, vec![true, false, true]);
    let ints = gguf
        .metadata_array("test.int_array")
        .expect("int array should exist")
        .as_i32_vec()
        .expect("int array should parse");
    assert_eq!(ints, vec![8, 8, 1]);
    let uints = gguf
        .metadata_array("test.uint_array")
        .expect("uint array should exist")
        .as_u32_vec()
        .expect("uint array should parse");
    assert_eq!(uints, vec![2, 4, 8]);

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
    let error = GgufFile::open(fixture.path())
        .err()
        .expect("bad magic should fail");

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
    let error = GgufFile::open(fixture.path())
        .err()
        .expect("truncated file should fail");

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
    let error = GgufFile::open(fixture.path())
        .err()
        .expect("version 4 should fail");

    match error {
        XrtError::Unsupported(message) => {
            assert!(message.contains("GGUF version 4"));
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn strict_open_rejects_qwen_image_edit_zero_timestep_marker() {
    let fixture = build_zero_sized_tensor_gguf(
        "qwen_image",
        QWEN_IMAGE_EDIT_TIMESTEP_ZERO_MARKER,
        vec![0],
        DType::F32,
    );
    let error = GgufFile::open(fixture.path())
        .err()
        .expect("strict parsing must reject zero-sized tensors");

    match error {
        XrtError::InvalidTensor(message) => {
            assert!(message.contains("zero-sized dimension"));
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn opt_in_accepts_only_the_exact_qwen_image_edit_zero_timestep_marker() {
    let fixture = build_zero_sized_tensor_gguf(
        "qwen_image",
        QWEN_IMAGE_EDIT_TIMESTEP_ZERO_MARKER,
        vec![0],
        DType::F32,
    );
    let gguf = GgufFile::open_with_compatibility(
        fixture.path(),
        GgufCompatibility::QwenImageEditTimestepZero,
    )
    .expect("the exact Qwen Image Edit marker should parse with explicit compatibility");
    let marker = gguf
        .require_tensor(QWEN_IMAGE_EDIT_TIMESTEP_ZERO_MARKER)
        .expect("marker should remain visible to the adapter");
    assert_eq!(marker.dimensions, [0]);
    assert_eq!(marker.dtype, DType::F32);
    assert_eq!(marker.nbytes, 0);
    assert!(gguf
        .tensor_data(QWEN_IMAGE_EDIT_TIMESTEP_ZERO_MARKER)
        .unwrap()
        .is_empty());

    for (architecture, name, dimensions, dtype) in [
        ("qwen_image", "model.weight", vec![0], DType::F32),
        (
            "llama",
            QWEN_IMAGE_EDIT_TIMESTEP_ZERO_MARKER,
            vec![0],
            DType::F32,
        ),
        (
            "qwen_image",
            QWEN_IMAGE_EDIT_TIMESTEP_ZERO_MARKER,
            vec![0],
            DType::F16,
        ),
        (
            "qwen_image",
            QWEN_IMAGE_EDIT_TIMESTEP_ZERO_MARKER,
            vec![0, 1],
            DType::F32,
        ),
    ] {
        let malformed = build_zero_sized_tensor_gguf(architecture, name, dimensions, dtype);
        assert!(GgufFile::open_with_compatibility(
            malformed.path(),
            GgufCompatibility::QwenImageEditTimestepZero,
        )
        .is_err());
    }
}
