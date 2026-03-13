#![allow(dead_code)]

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
    pub bytes: Vec<u8>,
}

impl GgufFixture {
    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[derive(Debug, Clone)]
pub struct TensorSpec {
    pub name: String,
    pub dimensions: Vec<usize>,
    pub dtype: DType,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone)]
pub enum MetadataValueSpec {
    Bool(bool),
    Float32(f32),
    Float32Array(Vec<f32>),
    String(String),
    StringArray(Vec<String>),
    UInt32(u32),
}

#[derive(Debug, Clone)]
pub struct TokenizerFixtureSpec {
    pub tokens: Vec<String>,
    pub scores: Vec<f32>,
    pub bos_id: u32,
    pub eos_id: u32,
    pub unk_id: u32,
    pub hello_id: u32,
    pub world_id: u32,
    pub bang_id: u32,
}

#[derive(Debug, Clone)]
pub struct SyntheticLlamaSpec {
    pub model_name: String,
    pub vocab_size: usize,
    pub context_length: usize,
    pub embedding_length: usize,
    pub feed_forward_length: usize,
    pub block_count: usize,
    pub attention_head_count: usize,
    pub attention_head_count_kv: usize,
    pub rope_dimension_count: usize,
    pub rms_norm_eps: f32,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub unk_token_id: u32,
    pub seed: u64,
}

impl SyntheticLlamaSpec {
    pub fn tiny() -> Self {
        Self {
            model_name: "synthetic-tiny-llama".to_string(),
            vocab_size: 256,
            context_length: 32,
            embedding_length: 64,
            feed_forward_length: 128,
            block_count: 2,
            attention_head_count: 4,
            attention_head_count_kv: 4,
            rope_dimension_count: 16,
            rms_norm_eps: 1e-5,
            rope_freq_base: 10_000.0,
            rope_freq_scale: 1.0,
            bos_token_id: 0,
            eos_token_id: 1,
            unk_token_id: 2,
            seed: 0x5EED_1234_ABCD_EF01,
        }
    }
}

pub fn build_minimal_valid_gguf_fixture() -> io::Result<GgufFixture> {
    let metadata = vec![
        (
            "general.architecture".to_string(),
            MetadataValueSpec::String("llama".to_string()),
        ),
        (
            "general.name".to_string(),
            MetadataValueSpec::String("test".to_string()),
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
            MetadataValueSpec::StringArray(vec![
                "<unk>".to_string(),
                format!("{SPM_SPACE}test"),
                "!".to_string(),
            ]),
        ),
        (
            "tokenizer.ggml.scores".to_string(),
            MetadataValueSpec::Float32Array(vec![0.0, 4.0, 1.0]),
        ),
        (
            "tokenizer.ggml.bos_token_id".to_string(),
            MetadataValueSpec::UInt32(0),
        ),
        (
            "tokenizer.ggml.add_bos_token".to_string(),
            MetadataValueSpec::Bool(true),
        ),
    ];

    let tensors = vec![
        TensorSpec {
            name: "tok_embeddings.weight".to_string(),
            dimensions: vec![4, 2],
            dtype: DType::F32,
            data: f32_tensor_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        },
        TensorSpec {
            name: "output.weight".to_string(),
            dimensions: vec![4, 1],
            dtype: DType::F32,
            data: f32_tensor_bytes(&[0.5, 1.5, 2.5, 3.5]),
        },
    ];

    build_gguf_fixture(3, metadata, tensors)
}

pub fn build_tokenizer_fixture() -> io::Result<(GgufFixture, TokenizerFixtureSpec)> {
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
            MetadataValueSpec::String("tokenizer-test".to_string()),
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
            MetadataValueSpec::StringArray(tokens.clone()),
        ),
        (
            "tokenizer.ggml.scores".to_string(),
            MetadataValueSpec::Float32Array(scores.clone()),
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
        3,
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
            tokens,
            scores,
            bos_id,
            eos_id,
            unk_id,
            hello_id,
            world_id,
            bang_id,
        },
    ))
}

pub fn build_bpe_tokenizer_fixture() -> io::Result<(GgufFixture, TokenizerFixtureSpec)> {
    let (base_fixture, mut spec) = build_tokenizer_fixture()?;
    let mut tokens = spec.tokens.clone();
    let mut scores = spec.scores.clone();

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
    scores.extend([12.0, 14.0, 16.0, 18.0, 11.0, 13.0, 15.0, 17.0]);

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
            MetadataValueSpec::StringArray(tokens.clone()),
        ),
        (
            "tokenizer.ggml.scores".to_string(),
            MetadataValueSpec::Float32Array(scores.clone()),
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
            MetadataValueSpec::UInt32(spec.bos_id),
        ),
        (
            "tokenizer.ggml.eos_token_id".to_string(),
            MetadataValueSpec::UInt32(spec.eos_id),
        ),
        (
            "tokenizer.ggml.unknown_token_id".to_string(),
            MetadataValueSpec::UInt32(spec.unk_id),
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
        3,
        metadata,
        vec![TensorSpec {
            name: "tok_embeddings.weight".to_string(),
            dimensions: vec![4, 1],
            dtype: DType::F32,
            data: f32_tensor_bytes(&[0.1, 0.2, 0.3, 0.4]),
        }],
    )?;

    drop(base_fixture);
    spec.tokens = tokens;
    spec.scores = scores;
    Ok((fixture, spec))
}

pub fn build_synthetic_llama_fixture(spec: SyntheticLlamaSpec) -> io::Result<GgufFixture> {
    if spec.vocab_size == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "synthetic llama vocab size must be non-zero",
        ));
    }
    if spec.embedding_length == 0
        || spec.feed_forward_length == 0
        || spec.block_count == 0
        || spec.attention_head_count == 0
        || spec.attention_head_count_kv == 0
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "synthetic llama dimensions must be non-zero",
        ));
    }
    if spec.embedding_length % spec.attention_head_count != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "embedding length must be divisible by attention head count",
        ));
    }
    if spec.attention_head_count % spec.attention_head_count_kv != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "attention head count must be divisible by KV head count",
        ));
    }
    for token_id in [spec.bos_token_id, spec.eos_token_id, spec.unk_token_id] {
        if token_id as usize >= spec.vocab_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "token id {token_id} is out of range for vocab {}",
                    spec.vocab_size
                ),
            ));
        }
    }

    let tokens = synthetic_tokens(&spec);
    let scores = vec![0.0; spec.vocab_size];
    let metadata = vec![
        (
            "general.architecture".to_string(),
            MetadataValueSpec::String("llama".to_string()),
        ),
        (
            "general.name".to_string(),
            MetadataValueSpec::String(spec.model_name.clone()),
        ),
        (
            "general.alignment".to_string(),
            MetadataValueSpec::UInt32(GGUF_ALIGNMENT as u32),
        ),
        (
            "llama.vocab_size".to_string(),
            MetadataValueSpec::UInt32(spec.vocab_size as u32),
        ),
        (
            "llama.context_length".to_string(),
            MetadataValueSpec::UInt32(spec.context_length as u32),
        ),
        (
            "llama.embedding_length".to_string(),
            MetadataValueSpec::UInt32(spec.embedding_length as u32),
        ),
        (
            "llama.feed_forward_length".to_string(),
            MetadataValueSpec::UInt32(spec.feed_forward_length as u32),
        ),
        (
            "llama.block_count".to_string(),
            MetadataValueSpec::UInt32(spec.block_count as u32),
        ),
        (
            "llama.attention.head_count".to_string(),
            MetadataValueSpec::UInt32(spec.attention_head_count as u32),
        ),
        (
            "llama.attention.head_count_kv".to_string(),
            MetadataValueSpec::UInt32(spec.attention_head_count_kv as u32),
        ),
        (
            "llama.rope.dimension_count".to_string(),
            MetadataValueSpec::UInt32(spec.rope_dimension_count as u32),
        ),
        (
            "llama.attention.layer_norm_rms_epsilon".to_string(),
            MetadataValueSpec::Float32(spec.rms_norm_eps),
        ),
        (
            "llama.rope.freq_base".to_string(),
            MetadataValueSpec::Float32(spec.rope_freq_base),
        ),
        (
            "llama.rope.scale_linear".to_string(),
            MetadataValueSpec::Float32(spec.rope_freq_scale),
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
            "tokenizer.ggml.bos_token_id".to_string(),
            MetadataValueSpec::UInt32(spec.bos_token_id),
        ),
        (
            "tokenizer.ggml.eos_token_id".to_string(),
            MetadataValueSpec::UInt32(spec.eos_token_id),
        ),
        (
            "tokenizer.ggml.unknown_token_id".to_string(),
            MetadataValueSpec::UInt32(spec.unk_token_id),
        ),
        (
            "tokenizer.ggml.add_bos_token".to_string(),
            MetadataValueSpec::Bool(true),
        ),
        (
            "tokenizer.ggml.add_eos_token".to_string(),
            MetadataValueSpec::Bool(false),
        ),
    ];

    build_gguf_fixture(3, metadata, synthetic_llama_tensors(&spec))
}

pub fn write_raw_gguf(bytes: Vec<u8>) -> io::Result<GgufFixture> {
    let dir = TempDir::new()?;
    let path = dir.path().join("fixture.gguf");
    fs::write(&path, &bytes)?;
    Ok(GgufFixture {
        _dir: dir,
        path,
        bytes,
    })
}

pub fn f32_tensor_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<u8>>()
}

pub fn byte_token(byte: u8) -> String {
    format!("<0x{byte:02X}>")
}

fn build_gguf_fixture(
    version: u32,
    metadata: Vec<(String, MetadataValueSpec)>,
    tensors: Vec<TensorSpec>,
) -> io::Result<GgufFixture> {
    let mut bytes = Vec::new();
    write_u32(&mut bytes, GGUF_MAGIC);
    write_u32(&mut bytes, version);
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
        assert_eq!(
            tensor.data.len(),
            expected,
            "tensor {} data does not match {} bytes",
            tensor.name,
            expected
        );
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

fn write_metadata_value(bytes: &mut Vec<u8>, value: &MetadataValueSpec) {
    match value {
        MetadataValueSpec::Bool(value) => {
            write_u32(bytes, 7);
            bytes.push(u8::from(*value));
        }
        MetadataValueSpec::Float32(value) => {
            write_u32(bytes, 6);
            write_u32(bytes, value.to_bits());
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

fn synthetic_tokens(spec: &SyntheticLlamaSpec) -> Vec<String> {
    let mut tokens = (0..spec.vocab_size)
        .map(|index| format!("{SPM_SPACE}tok{index:03}"))
        .collect::<Vec<_>>();
    tokens[spec.bos_token_id as usize] = "<s>".to_string();
    tokens[spec.eos_token_id as usize] = "</s>".to_string();
    tokens[spec.unk_token_id as usize] = "<unk>".to_string();
    tokens
}

fn synthetic_llama_tensors(spec: &SyntheticLlamaSpec) -> Vec<TensorSpec> {
    let head_dim = spec.embedding_length / spec.attention_head_count;
    let kv_width = spec.attention_head_count_kv * head_dim;
    let mut seed = spec.seed;
    let mut tensors = Vec::new();

    tensors.push(random_f32_tensor(
        "token_embd.weight",
        vec![spec.embedding_length, spec.vocab_size],
        &mut seed,
    ));

    for layer in 0..spec.block_count {
        tensors.push(random_f32_tensor(
            format!("blk.{layer}.attn_norm.weight"),
            vec![spec.embedding_length],
            &mut seed,
        ));
        tensors.push(random_f32_tensor(
            format!("blk.{layer}.attn_q.weight"),
            vec![spec.embedding_length, spec.embedding_length],
            &mut seed,
        ));
        tensors.push(random_f32_tensor(
            format!("blk.{layer}.attn_k.weight"),
            vec![spec.embedding_length, kv_width],
            &mut seed,
        ));
        tensors.push(random_f32_tensor(
            format!("blk.{layer}.attn_v.weight"),
            vec![spec.embedding_length, kv_width],
            &mut seed,
        ));
        tensors.push(random_f32_tensor(
            format!("blk.{layer}.attn_output.weight"),
            vec![spec.embedding_length, spec.embedding_length],
            &mut seed,
        ));
        tensors.push(random_f32_tensor(
            format!("blk.{layer}.ffn_norm.weight"),
            vec![spec.embedding_length],
            &mut seed,
        ));
        tensors.push(random_f32_tensor(
            format!("blk.{layer}.ffn_gate.weight"),
            vec![spec.embedding_length, spec.feed_forward_length],
            &mut seed,
        ));
        tensors.push(random_f32_tensor(
            format!("blk.{layer}.ffn_up.weight"),
            vec![spec.embedding_length, spec.feed_forward_length],
            &mut seed,
        ));
        tensors.push(random_f32_tensor(
            format!("blk.{layer}.ffn_down.weight"),
            vec![spec.feed_forward_length, spec.embedding_length],
            &mut seed,
        ));
    }

    tensors.push(random_f32_tensor(
        "output_norm.weight",
        vec![spec.embedding_length],
        &mut seed,
    ));
    tensors.push(random_f32_tensor(
        "output.weight",
        vec![spec.embedding_length, spec.vocab_size],
        &mut seed,
    ));

    tensors
}

fn random_f32_tensor(
    name: impl Into<String>,
    dimensions: Vec<usize>,
    seed: &mut u64,
) -> TensorSpec {
    TensorSpec {
        name: name.into(),
        data: random_f32_bytes(seed, product(&dimensions)),
        dimensions,
        dtype: DType::F32,
    }
}

fn random_f32_bytes(seed: &mut u64, count: usize) -> Vec<u8> {
    let values = (0..count)
        .map(|_| next_random_f32(seed))
        .collect::<Vec<_>>();
    f32_tensor_bytes(&values)
}

fn next_random_f32(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    let mantissa = (*seed >> 40) as u32;
    let unit = mantissa as f32 / ((1u32 << 24) as f32);
    (unit - 0.5) * 0.08
}

fn product(dimensions: &[usize]) -> usize {
    dimensions.iter().copied().product()
}
