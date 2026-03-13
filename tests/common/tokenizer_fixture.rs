use std::{
    fs, io,
    path::{Path, PathBuf},
};

use tempfile::TempDir;
use xrt_core::{align_up, DType};

const GGUF_MAGIC: u32 = 0x4647_5547;
const GGUF_ALIGNMENT: usize = 32;

pub const SPM_SPACE: char = '\u{2581}';

#[derive(Debug)]
pub struct GgufFixture {
    _dir: TempDir,
    path: PathBuf,
}

impl GgufFixture {
    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[derive(Debug, Clone)]
pub struct TokenizerFixtureSpec {
    pub hello_id: u32,
    pub world_id: u32,
    pub bang_id: u32,
}

#[derive(Debug, Clone)]
struct TensorSpec {
    name: String,
    dimensions: Vec<usize>,
    dtype: DType,
    data: Vec<u8>,
}

#[derive(Debug, Clone)]
enum MetadataValueSpec {
    Bool(bool),
    Float32Array(Vec<f32>),
    String(String),
    StringArray(Vec<String>),
    UInt32(u32),
}

pub fn build_bpe_tokenizer_fixture() -> io::Result<(GgufFixture, TokenizerFixtureSpec)> {
    let mut tokens: Vec<String> = (0u8..=u8::MAX).map(byte_token).collect();
    let bos_id = tokens.len() as u32;
    tokens.push("<s>".to_string());
    let eos_id = tokens.len() as u32;
    tokens.push("</s>".to_string());
    let unk_id = tokens.len() as u32;
    tokens.push("<unk>".to_string());
    let hello_id = tokens.len() as u32;
    tokens.push(format!("{SPM_SPACE}hello"));
    let world_id = tokens.len() as u32;
    tokens.push(format!("{SPM_SPACE}world"));
    let bang_id = tokens.len() as u32;
    tokens.push("!".to_string());
    tokens.extend([
        format!("{SPM_SPACE}h"),
        format!("{SPM_SPACE}he"),
        format!("{SPM_SPACE}hel"),
        format!("{SPM_SPACE}hell"),
        format!("{SPM_SPACE}w"),
        format!("{SPM_SPACE}wo"),
        format!("{SPM_SPACE}wor"),
        format!("{SPM_SPACE}worl"),
    ]);

    let mut scores = vec![0.0; tokens.len()];
    scores[hello_id as usize] = 20.0;
    scores[world_id as usize] = 18.0;
    scores[bang_id as usize] = 5.0;

    let metadata = vec![
        (
            "general.architecture".to_string(),
            MetadataValueSpec::String("llama".to_string()),
        ),
        (
            "general.name".to_string(),
            MetadataValueSpec::String("tokenizer-bpe-bench".to_string()),
        ),
        (
            "general.alignment".to_string(),
            MetadataValueSpec::UInt32(GGUF_ALIGNMENT as u32),
        ),
        (
            "tokenizer.ggml.model".to_string(),
            MetadataValueSpec::String("llama".to_string()),
        ),
        (
            "tokenizer.ggml.tokens".to_string(),
            MetadataValueSpec::StringArray(tokens),
        ),
        (
            "tokenizer.ggml.scores".to_string(),
            MetadataValueSpec::Float32Array(scores),
        ),
        (
            "tokenizer.ggml.merges".to_string(),
            MetadataValueSpec::StringArray(vec![
                format!("{SPM_SPACE} h"),
                format!("{SPM_SPACE}h e"),
                format!("{SPM_SPACE}he l"),
                format!("{SPM_SPACE}hel l"),
                format!("{SPM_SPACE}hell o"),
                format!("{SPM_SPACE} w"),
                format!("{SPM_SPACE}w o"),
                format!("{SPM_SPACE}wo r"),
                format!("{SPM_SPACE}wor l"),
                format!("{SPM_SPACE}worl d"),
            ]),
        ),
        (
            "tokenizer.ggml.bos_token_id".to_string(),
            MetadataValueSpec::UInt32(bos_id),
        ),
        (
            "tokenizer.ggml.eos_token_id".to_string(),
            MetadataValueSpec::UInt32(eos_id),
        ),
        (
            "tokenizer.ggml.unknown_token_id".to_string(),
            MetadataValueSpec::UInt32(unk_id),
        ),
        (
            "tokenizer.ggml.add_bos_token".to_string(),
            MetadataValueSpec::Bool(true),
        ),
        (
            "tokenizer.ggml.add_eos_token".to_string(),
            MetadataValueSpec::Bool(true),
        ),
    ];

    let fixture = build_gguf_fixture(
        metadata,
        vec![TensorSpec {
            name: "tok_embeddings.weight".to_string(),
            dimensions: vec![4, 1],
            dtype: DType::F32,
            data: f32_tensor_bytes(&[0.1, 0.2, 0.3, 0.4]),
        }],
    )?;

    Ok((
        fixture,
        TokenizerFixtureSpec {
            hello_id,
            world_id,
            bang_id,
        },
    ))
}

fn byte_token(byte: u8) -> String {
    format!("<0x{byte:02X}>")
}

fn f32_tensor_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>()
}

fn build_gguf_fixture(
    metadata: Vec<(String, MetadataValueSpec)>,
    tensors: Vec<TensorSpec>,
) -> io::Result<GgufFixture> {
    let mut bytes = Vec::new();
    write_u32(&mut bytes, GGUF_MAGIC);
    write_u32(&mut bytes, 3);
    write_u64(&mut bytes, tensors.len() as u64);
    write_u64(&mut bytes, metadata.len() as u64);

    for (key, value) in &metadata {
        write_string(&mut bytes, key);
        write_metadata_value(&mut bytes, value);
    }

    let mut offsets = Vec::with_capacity(tensors.len());
    let mut next_offset = 0usize;
    for tensor in &tensors {
        let expected = tensor
            .dtype
            .storage_len(&tensor.dimensions)
            .map_err(io::Error::other)?;
        assert_eq!(tensor.data.len(), expected);
        let offset = align_up(next_offset, GGUF_ALIGNMENT).map_err(io::Error::other)?;
        offsets.push(offset);
        next_offset = offset + tensor.data.len();
    }

    for (tensor, offset) in tensors.iter().zip(offsets.iter().copied()) {
        write_string(&mut bytes, &tensor.name);
        write_u32(&mut bytes, tensor.dimensions.len() as u32);
        for dim in &tensor.dimensions {
            write_u64(&mut bytes, *dim as u64);
        }
        write_i32(&mut bytes, tensor.dtype.ggml_type_id());
        write_u64(&mut bytes, offset as u64);
    }

    let data_offset = align_up(bytes.len(), GGUF_ALIGNMENT).map_err(io::Error::other)?;
    bytes.resize(data_offset, 0);

    for (tensor, offset) in tensors.into_iter().zip(offsets.into_iter()) {
        let start = data_offset + offset;
        if bytes.len() < start {
            bytes.resize(start, 0);
        }
        bytes.extend_from_slice(&tensor.data);
    }

    write_raw_gguf(bytes)
}

fn write_raw_gguf(bytes: Vec<u8>) -> io::Result<GgufFixture> {
    let dir = TempDir::new()?;
    let path = dir.path().join("fixture.gguf");
    fs::write(&path, bytes)?;
    Ok(GgufFixture { _dir: dir, path })
}

fn write_metadata_value(bytes: &mut Vec<u8>, value: &MetadataValueSpec) {
    match value {
        MetadataValueSpec::Bool(value) => {
            write_u32(bytes, 7);
            bytes.push(u8::from(*value));
        }
        MetadataValueSpec::Float32Array(values) => {
            write_u32(bytes, 9);
            write_u32(bytes, 6);
            write_u64(bytes, values.len() as u64);
            for value in values {
                write_u32(bytes, value.to_bits());
            }
        }
        MetadataValueSpec::String(value) => {
            write_u32(bytes, 8);
            write_string(bytes, value);
        }
        MetadataValueSpec::StringArray(values) => {
            write_u32(bytes, 9);
            write_u32(bytes, 8);
            write_u64(bytes, values.len() as u64);
            for value in values {
                write_string(bytes, value);
            }
        }
        MetadataValueSpec::UInt32(value) => {
            write_u32(bytes, 4);
            write_u32(bytes, *value);
        }
    }
}

fn write_string(bytes: &mut Vec<u8>, value: &str) {
    write_u64(bytes, value.len() as u64);
    bytes.extend_from_slice(value.as_bytes());
}

fn write_i32(bytes: &mut Vec<u8>, value: i32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn write_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn write_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}
