#![allow(dead_code)]

use std::{
    fs, io,
    path::{Path, PathBuf},
};

use half::{bf16, f16};
use tempfile::TempDir;
use xrt_core::{align_up, DType};

const GGUF_MAGIC: u32 = 0x4655_4747;
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
    BoolArray(Vec<bool>),
    Float32(f32),
    Float32Array(Vec<f32>),
    Int32Array(Vec<i32>),
    String(String),
    StringArray(Vec<String>),
    UInt32(u32),
    UInt32Array(Vec<u32>),
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
        (
            "test.bool_array".to_string(),
            MetadataValueSpec::BoolArray(vec![true, false, true]),
        ),
        (
            "test.int_array".to_string(),
            MetadataValueSpec::Int32Array(vec![8, 8, 1]),
        ),
        (
            "test.uint_array".to_string(),
            MetadataValueSpec::UInt32Array(vec![2, 4, 8]),
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
    build_synthetic_llama_fixture_with_architecture(spec, "llama", "llama")
}

pub fn build_synthetic_llama_fixture_with_architecture(
    spec: SyntheticLlamaSpec,
    architecture: &str,
    metadata_prefix: &str,
) -> io::Result<GgufFixture> {
    let tensors = synthetic_llama_tensors(&spec);
    build_synthetic_llama_fixture_with_tensors(spec, architecture, metadata_prefix, tensors)
}

pub fn build_synthetic_q8_0_single_layer_llama_fixture(
    spec: SyntheticLlamaSpec,
) -> io::Result<GgufFixture> {
    if spec.block_count != 1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Q8_0 CUDA synthetic fixture requires exactly one layer",
        ));
    }
    build_synthetic_q8_0_llama_fixture(spec)
}

pub fn build_synthetic_q8_0_llama_fixture(spec: SyntheticLlamaSpec) -> io::Result<GgufFixture> {
    let tensors = synthetic_q8_0_llama_tensors(&spec);
    build_synthetic_llama_fixture_with_tensors(spec, "llama", "llama", tensors)
}

pub fn build_synthetic_f16_llama_fixture(spec: SyntheticLlamaSpec) -> io::Result<GgufFixture> {
    let tensors = synthetic_float_llama_tensors(&spec, DType::F16);
    build_synthetic_llama_fixture_with_tensors(spec, "llama", "llama", tensors)
}

pub fn build_synthetic_bf16_llama_fixture(spec: SyntheticLlamaSpec) -> io::Result<GgufFixture> {
    let tensors = synthetic_float_llama_tensors(&spec, DType::BF16);
    build_synthetic_llama_fixture_with_tensors(spec, "llama", "llama", tensors)
}

pub fn build_synthetic_q8_0_tied_output_llama_fixture(
    spec: SyntheticLlamaSpec,
) -> io::Result<GgufFixture> {
    let tensors = synthetic_q8_0_llama_tensors(&spec)
        .into_iter()
        .filter(|tensor| tensor.name != "output.weight")
        .collect();
    build_synthetic_llama_fixture_with_tensors(spec, "llama", "llama", tensors)
}

pub fn build_synthetic_q4_0_llama_fixture(spec: SyntheticLlamaSpec) -> io::Result<GgufFixture> {
    let tensors = synthetic_q4_0_llama_tensors(&spec);
    build_synthetic_llama_fixture_with_tensors(spec, "llama", "llama", tensors)
}

pub fn build_synthetic_q4_k_llama_fixture(spec: SyntheticLlamaSpec) -> io::Result<GgufFixture> {
    let tensors = synthetic_q4_k_llama_tensors(&spec);
    build_synthetic_llama_fixture_with_tensors(spec, "llama", "llama", tensors)
}

pub fn build_synthetic_q5_k_llama_fixture(spec: SyntheticLlamaSpec) -> io::Result<GgufFixture> {
    let tensors = synthetic_q5_k_llama_tensors(&spec);
    build_synthetic_llama_fixture_with_tensors(spec, "llama", "llama", tensors)
}

pub fn build_synthetic_q6_k_llama_fixture(spec: SyntheticLlamaSpec) -> io::Result<GgufFixture> {
    let tensors = synthetic_q6_k_llama_tensors(&spec);
    build_synthetic_llama_fixture_with_tensors(spec, "llama", "llama", tensors)
}

pub fn build_synthetic_qwen35_hybrid_fixture() -> io::Result<(GgufFixture, SyntheticLlamaSpec)> {
    build_synthetic_qwen35_hybrid_fixture_with_context(32)
}

pub fn build_synthetic_qwen35_mtp_fixture() -> io::Result<(GgufFixture, SyntheticLlamaSpec)> {
    let trunk = SyntheticLlamaSpec {
        model_name: "synthetic-tiny-qwen35-mtp".to_string(),
        vocab_size: 32,
        context_length: 32,
        embedding_length: 8,
        feed_forward_length: 16,
        block_count: 4,
        attention_head_count: 2,
        attention_head_count_kv: 1,
        rope_dimension_count: 4,
        rms_norm_eps: 1e-5,
        rope_freq_base: 10_000.0,
        rope_freq_scale: 1.0,
        bos_token_id: 0,
        eos_token_id: 1,
        unk_token_id: 2,
        seed: 0x5135_5EED_A11C_E004,
    };
    let mut tensors = synthetic_qwen35_hybrid_tensors(&trunk);
    let layer = trunk.block_count;
    let dim = trunk.embedding_length;
    let head_dim = dim / trunk.attention_head_count;
    let q_width = trunk.attention_head_count * head_dim;
    let kv_width = trunk.attention_head_count_kv * head_dim;
    let mut seed = trunk.seed ^ 0x4D54_5001;
    for (name, dimensions) in [
        (format!("blk.{layer}.attn_norm.weight"), vec![dim]),
        (format!("blk.{layer}.attn_q.weight"), vec![dim, q_width * 2]),
        (format!("blk.{layer}.attn_k.weight"), vec![dim, kv_width]),
        (format!("blk.{layer}.attn_v.weight"), vec![dim, kv_width]),
        (
            format!("blk.{layer}.attn_output.weight"),
            vec![q_width, dim],
        ),
        (format!("blk.{layer}.attn_q_norm.weight"), vec![head_dim]),
        (format!("blk.{layer}.attn_k_norm.weight"), vec![head_dim]),
        (format!("blk.{layer}.post_attention_norm.weight"), vec![dim]),
        (
            format!("blk.{layer}.ffn_gate.weight"),
            vec![dim, trunk.feed_forward_length],
        ),
        (
            format!("blk.{layer}.ffn_up.weight"),
            vec![dim, trunk.feed_forward_length],
        ),
        (
            format!("blk.{layer}.ffn_down.weight"),
            vec![trunk.feed_forward_length, dim],
        ),
        (
            format!("blk.{layer}.nextn.eh_proj.weight"),
            vec![dim * 2, dim],
        ),
        (format!("blk.{layer}.nextn.enorm.weight"), vec![dim]),
        (format!("blk.{layer}.nextn.hnorm.weight"), vec![dim]),
        (
            format!("blk.{layer}.nextn.shared_head_norm.weight"),
            vec![dim],
        ),
    ] {
        tensors.push(random_f32_tensor(name, dimensions, &mut seed));
    }
    let metadata = vec![
        (
            "qwen35.ssm.conv_kernel".to_string(),
            MetadataValueSpec::UInt32(4),
        ),
        (
            "qwen35.ssm.state_size".to_string(),
            MetadataValueSpec::UInt32(4),
        ),
        (
            "qwen35.ssm.group_count".to_string(),
            MetadataValueSpec::UInt32(1),
        ),
        (
            "qwen35.ssm.inner_size".to_string(),
            MetadataValueSpec::UInt32(8),
        ),
        (
            "qwen35.ssm.time_step_rank".to_string(),
            MetadataValueSpec::UInt32(2),
        ),
        (
            "qwen35.nextn_predict_layers".to_string(),
            MetadataValueSpec::UInt32(1),
        ),
    ];
    let mut physical = trunk.clone();
    physical.block_count += 1;
    let fixture = build_synthetic_llama_fixture_with_tensors_and_metadata(
        physical, "qwen35", "qwen35", tensors, metadata,
    )?;
    Ok((fixture, trunk))
}

pub fn build_synthetic_qwen35_hybrid_long_fixture() -> io::Result<(GgufFixture, SyntheticLlamaSpec)>
{
    build_synthetic_qwen35_hybrid_fixture_with_context(256)
}

pub fn build_synthetic_qwen35_hybrid_moe_fixture() -> io::Result<(GgufFixture, SyntheticLlamaSpec)>
{
    let spec = SyntheticLlamaSpec {
        model_name: "synthetic-tiny-qwen35-hybrid-moe".to_string(),
        vocab_size: 32,
        context_length: 32,
        embedding_length: 8,
        feed_forward_length: 16,
        block_count: 4,
        attention_head_count: 2,
        attention_head_count_kv: 1,
        rope_dimension_count: 4,
        rms_norm_eps: 1e-5,
        rope_freq_base: 10_000.0,
        rope_freq_scale: 1.0,
        bos_token_id: 0,
        eos_token_id: 1,
        unk_token_id: 2,
        seed: 0x5135_5EED_A11C_E003,
    };
    let expert_count = 4;
    let tensors = synthetic_qwen35_hybrid_moe_tensors(&spec, expert_count);
    let metadata = vec![
        (
            "qwen3_5_moe.expert_count".to_string(),
            MetadataValueSpec::UInt32(expert_count as u32),
        ),
        (
            "qwen3_5_moe.expert_used_count".to_string(),
            MetadataValueSpec::UInt32(2),
        ),
        (
            "qwen3_5_moe.expert_feed_forward_length".to_string(),
            MetadataValueSpec::UInt32(spec.feed_forward_length as u32),
        ),
        (
            "qwen3_5_moe.ssm.conv_kernel".to_string(),
            MetadataValueSpec::UInt32(4),
        ),
        (
            "qwen3_5_moe.ssm.state_size".to_string(),
            MetadataValueSpec::UInt32(4),
        ),
        (
            "qwen3_5_moe.ssm.group_count".to_string(),
            MetadataValueSpec::UInt32(1),
        ),
        (
            "qwen3_5_moe.ssm.inner_size".to_string(),
            MetadataValueSpec::UInt32(8),
        ),
        (
            "qwen3_5_moe.ssm.time_step_rank".to_string(),
            MetadataValueSpec::UInt32(2),
        ),
    ];
    let fixture = build_synthetic_llama_fixture_with_tensors_and_metadata(
        spec.clone(),
        "qwen3_5_moe",
        "qwen3_5_moe",
        tensors,
        metadata,
    )?;
    Ok((fixture, spec))
}

fn build_synthetic_qwen35_hybrid_fixture_with_context(
    context_length: usize,
) -> io::Result<(GgufFixture, SyntheticLlamaSpec)> {
    let spec = SyntheticLlamaSpec {
        model_name: "synthetic-tiny-qwen35-hybrid".to_string(),
        vocab_size: 32,
        context_length,
        embedding_length: 8,
        feed_forward_length: 16,
        block_count: 4,
        attention_head_count: 2,
        attention_head_count_kv: 1,
        rope_dimension_count: 4,
        rms_norm_eps: 1e-5,
        rope_freq_base: 10_000.0,
        rope_freq_scale: 1.0,
        bos_token_id: 0,
        eos_token_id: 1,
        unk_token_id: 2,
        seed: 0x5135_5EED_A11C_E001,
    };
    let tensors = synthetic_qwen35_hybrid_tensors(&spec);
    let metadata = vec![
        (
            "qwen3_5.ssm.conv_kernel".to_string(),
            MetadataValueSpec::UInt32(4),
        ),
        (
            "qwen3_5.ssm.state_size".to_string(),
            MetadataValueSpec::UInt32(4),
        ),
        (
            "qwen3_5.ssm.group_count".to_string(),
            MetadataValueSpec::UInt32(1),
        ),
        (
            "qwen3_5.ssm.inner_size".to_string(),
            MetadataValueSpec::UInt32(8),
        ),
        (
            "qwen3_5.ssm.time_step_rank".to_string(),
            MetadataValueSpec::UInt32(2),
        ),
    ];
    let fixture = build_synthetic_llama_fixture_with_tensors_and_metadata(
        spec.clone(),
        "qwen3_5",
        "qwen3_5",
        tensors,
        metadata,
    )?;
    Ok((fixture, spec))
}

pub fn build_synthetic_qwen3_moe_fixture() -> io::Result<(GgufFixture, SyntheticLlamaSpec)> {
    let spec = SyntheticLlamaSpec {
        model_name: "synthetic-tiny-qwen3-moe".to_string(),
        vocab_size: 32,
        context_length: 32,
        embedding_length: 8,
        feed_forward_length: 16,
        block_count: 2,
        attention_head_count: 2,
        attention_head_count_kv: 1,
        rope_dimension_count: 4,
        rms_norm_eps: 1e-5,
        rope_freq_base: 10_000.0,
        rope_freq_scale: 1.0,
        bos_token_id: 0,
        eos_token_id: 1,
        unk_token_id: 2,
        seed: 0x03E0_5EED_A11C_E002,
    };
    let expert_count = 4;
    let tensors = synthetic_qwen3_moe_tensors(&spec, expert_count);
    let metadata = vec![
        (
            "qwen3.expert_count".to_string(),
            MetadataValueSpec::UInt32(expert_count as u32),
        ),
        (
            "qwen3.expert_used_count".to_string(),
            MetadataValueSpec::UInt32(2),
        ),
    ];
    let fixture = build_synthetic_llama_fixture_with_tensors_and_metadata(
        spec.clone(),
        "qwen3",
        "qwen3",
        tensors,
        metadata,
    )?;
    Ok((fixture, spec))
}

/// Larger deterministic MoE used only to exercise the grouped CPU threshold.
pub fn build_synthetic_qwen3_moe_benchmark_fixture() -> io::Result<(GgufFixture, SyntheticLlamaSpec)>
{
    let spec = SyntheticLlamaSpec {
        model_name: "synthetic-benchmark-qwen3-moe".to_string(),
        vocab_size: 64,
        context_length: 32,
        embedding_length: 128,
        feed_forward_length: 256,
        block_count: 2,
        attention_head_count: 8,
        attention_head_count_kv: 2,
        rope_dimension_count: 16,
        rms_norm_eps: 1e-5,
        rope_freq_base: 10_000.0,
        rope_freq_scale: 1.0,
        bos_token_id: 0,
        eos_token_id: 1,
        unk_token_id: 2,
        seed: 0x03E0_5EED_BA7C_0002,
    };
    let expert_count = 8;
    let tensors = synthetic_qwen3_moe_tensors(&spec, expert_count);
    let metadata = vec![
        (
            "qwen3.expert_count".to_string(),
            MetadataValueSpec::UInt32(expert_count as u32),
        ),
        (
            "qwen3.expert_used_count".to_string(),
            MetadataValueSpec::UInt32(2),
        ),
    ];
    let fixture = build_synthetic_llama_fixture_with_tensors_and_metadata(
        spec.clone(),
        "qwen3",
        "qwen3",
        tensors,
        metadata,
    )?;
    Ok((fixture, spec))
}

pub fn build_synthetic_qwen3moe_packed_fixture() -> io::Result<(GgufFixture, SyntheticLlamaSpec)> {
    let spec = SyntheticLlamaSpec {
        model_name: "synthetic-packed-qwen3moe".to_string(),
        vocab_size: 32,
        context_length: 32,
        embedding_length: 8,
        feed_forward_length: 16,
        block_count: 2,
        attention_head_count: 2,
        attention_head_count_kv: 1,
        rope_dimension_count: 4,
        rms_norm_eps: 1e-5,
        rope_freq_base: 10_000.0,
        rope_freq_scale: 1.0,
        bos_token_id: 0,
        eos_token_id: 1,
        unk_token_id: 2,
        seed: 0x03E0_5EED_AC7E_0002,
    };
    let expert_count = 4;
    let tensors = synthetic_qwen3_moe_tensors_with_layout(&spec, expert_count, true);
    let metadata = vec![
        (
            "qwen3moe.expert_count".to_string(),
            MetadataValueSpec::UInt32(expert_count as u32),
        ),
        (
            "qwen3moe.expert_used_count".to_string(),
            MetadataValueSpec::UInt32(2),
        ),
        (
            "qwen3moe.expert_feed_forward_length".to_string(),
            MetadataValueSpec::UInt32(spec.feed_forward_length as u32),
        ),
    ];
    let fixture = build_synthetic_llama_fixture_with_tensors_and_metadata(
        spec.clone(),
        "qwen3moe",
        "qwen3moe",
        tensors,
        metadata,
    )?;
    Ok((fixture, spec))
}

pub fn build_synthetic_qwen3moe_shared_expert_fixture(
) -> io::Result<(GgufFixture, SyntheticLlamaSpec)> {
    let spec = SyntheticLlamaSpec {
        model_name: "synthetic-shared-expert-qwen3moe".to_string(),
        vocab_size: 32,
        context_length: 32,
        embedding_length: 8,
        feed_forward_length: 16,
        block_count: 2,
        attention_head_count: 2,
        attention_head_count_kv: 1,
        rope_dimension_count: 4,
        rms_norm_eps: 1e-5,
        rope_freq_base: 10_000.0,
        rope_freq_scale: 1.0,
        bos_token_id: 0,
        eos_token_id: 1,
        unk_token_id: 2,
        seed: 0x03E0_5EED_5A4E_0002,
    };
    let expert_count = 4;
    let shared_intermediate = 12;
    let mut tensors = synthetic_qwen3_moe_tensors_with_layout(&spec, expert_count, true);
    let mut seed = spec.seed ^ 0x5A4E_D000;
    for layer in 0..spec.block_count {
        tensors.push(random_f32_tensor(
            format!("blk.{layer}.ffn_gate_inp_shexp.weight"),
            vec![spec.embedding_length],
            &mut seed,
        ));
        tensors.push(random_f32_tensor(
            format!("blk.{layer}.ffn_gate_shexp.weight"),
            vec![spec.embedding_length, shared_intermediate],
            &mut seed,
        ));
        tensors.push(random_f32_tensor(
            format!("blk.{layer}.ffn_up_shexp.weight"),
            vec![spec.embedding_length, shared_intermediate],
            &mut seed,
        ));
        tensors.push(random_f32_tensor(
            format!("blk.{layer}.ffn_down_shexp.weight"),
            vec![shared_intermediate, spec.embedding_length],
            &mut seed,
        ));
    }
    let metadata = vec![
        (
            "qwen3moe.expert_count".to_string(),
            MetadataValueSpec::UInt32(expert_count as u32),
        ),
        (
            "qwen3moe.expert_used_count".to_string(),
            MetadataValueSpec::UInt32(2),
        ),
        (
            "qwen3moe.expert_feed_forward_length".to_string(),
            MetadataValueSpec::UInt32(spec.feed_forward_length as u32),
        ),
        (
            "qwen3moe.expert_shared_feed_forward_length".to_string(),
            MetadataValueSpec::UInt32(shared_intermediate as u32),
        ),
    ];
    let fixture = build_synthetic_llama_fixture_with_tensors_and_metadata(
        spec.clone(),
        "qwen3moe",
        "qwen3moe",
        tensors,
        metadata,
    )?;
    Ok((fixture, spec))
}

fn build_synthetic_llama_fixture_with_tensors(
    spec: SyntheticLlamaSpec,
    architecture: &str,
    metadata_prefix: &str,
    tensors: Vec<TensorSpec>,
) -> io::Result<GgufFixture> {
    build_synthetic_llama_fixture_with_tensors_and_metadata(
        spec,
        architecture,
        metadata_prefix,
        tensors,
        Vec::new(),
    )
}

fn build_synthetic_llama_fixture_with_tensors_and_metadata(
    spec: SyntheticLlamaSpec,
    architecture: &str,
    metadata_prefix: &str,
    tensors: Vec<TensorSpec>,
    extra_metadata: Vec<(String, MetadataValueSpec)>,
) -> io::Result<GgufFixture> {
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
    let mut metadata = vec![
        (
            "general.architecture".to_string(),
            MetadataValueSpec::String(architecture.to_string()),
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
            format!("{metadata_prefix}.vocab_size"),
            MetadataValueSpec::UInt32(spec.vocab_size as u32),
        ),
        (
            format!("{metadata_prefix}.context_length"),
            MetadataValueSpec::UInt32(spec.context_length as u32),
        ),
        (
            format!("{metadata_prefix}.embedding_length"),
            MetadataValueSpec::UInt32(spec.embedding_length as u32),
        ),
        (
            format!("{metadata_prefix}.feed_forward_length"),
            MetadataValueSpec::UInt32(spec.feed_forward_length as u32),
        ),
        (
            format!("{metadata_prefix}.block_count"),
            MetadataValueSpec::UInt32(spec.block_count as u32),
        ),
        (
            format!("{metadata_prefix}.attention.head_count"),
            MetadataValueSpec::UInt32(spec.attention_head_count as u32),
        ),
        (
            format!("{metadata_prefix}.attention.head_count_kv"),
            MetadataValueSpec::UInt32(spec.attention_head_count_kv as u32),
        ),
        (
            format!("{metadata_prefix}.rope.dimension_count"),
            MetadataValueSpec::UInt32(spec.rope_dimension_count as u32),
        ),
        (
            format!("{metadata_prefix}.attention.layer_norm_rms_epsilon"),
            MetadataValueSpec::Float32(spec.rms_norm_eps),
        ),
        (
            format!("{metadata_prefix}.rope.freq_base"),
            MetadataValueSpec::Float32(spec.rope_freq_base),
        ),
        (
            format!("{metadata_prefix}.rope.scale_linear"),
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
    metadata.extend(extra_metadata);

    build_gguf_fixture(3, metadata, tensors)
}

pub fn build_synthetic_gemma4_fixture() -> io::Result<GgufFixture> {
    let spec = SyntheticLlamaSpec {
        model_name: "synthetic-tiny-gemma4".to_string(),
        vocab_size: 32,
        context_length: 16,
        embedding_length: 8,
        feed_forward_length: 16,
        block_count: 2,
        attention_head_count: 2,
        attention_head_count_kv: 2,
        rope_dimension_count: 4,
        rms_norm_eps: 1e-5,
        rope_freq_base: 1_000_000.0,
        rope_freq_scale: 1.0,
        bos_token_id: 0,
        eos_token_id: 1,
        unk_token_id: 2,
        seed: 0x6E44_A400_D00D_0001,
    };
    let tokens = synthetic_tokens(&spec);
    let scores = vec![0.0; spec.vocab_size];
    let metadata = vec![
        (
            "general.architecture".to_string(),
            MetadataValueSpec::String("gemma4".to_string()),
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
            "gemma4.context_length".to_string(),
            MetadataValueSpec::UInt32(spec.context_length as u32),
        ),
        (
            "gemma4.embedding_length".to_string(),
            MetadataValueSpec::UInt32(spec.embedding_length as u32),
        ),
        (
            "gemma4.feed_forward_length".to_string(),
            MetadataValueSpec::UInt32(spec.feed_forward_length as u32),
        ),
        (
            "gemma4.block_count".to_string(),
            MetadataValueSpec::UInt32(spec.block_count as u32),
        ),
        (
            "gemma4.attention.head_count".to_string(),
            MetadataValueSpec::UInt32(spec.attention_head_count as u32),
        ),
        (
            "gemma4.attention.head_count_kv".to_string(),
            MetadataValueSpec::Int32Array(vec![2, 1]),
        ),
        (
            "gemma4.attention.key_length".to_string(),
            MetadataValueSpec::UInt32(4),
        ),
        (
            "gemma4.attention.value_length".to_string(),
            MetadataValueSpec::UInt32(4),
        ),
        (
            "gemma4.attention.key_length_swa".to_string(),
            MetadataValueSpec::UInt32(2),
        ),
        (
            "gemma4.attention.value_length_swa".to_string(),
            MetadataValueSpec::UInt32(2),
        ),
        (
            "gemma4.rope.dimension_count".to_string(),
            MetadataValueSpec::UInt32(4),
        ),
        (
            "gemma4.rope.dimension_count_swa".to_string(),
            MetadataValueSpec::UInt32(2),
        ),
        (
            "gemma4.rope.freq_base".to_string(),
            MetadataValueSpec::Float32(spec.rope_freq_base),
        ),
        (
            "gemma4.rope.freq_base_swa".to_string(),
            MetadataValueSpec::Float32(10_000.0),
        ),
        (
            "gemma4.attention.layer_norm_rms_epsilon".to_string(),
            MetadataValueSpec::Float32(spec.rms_norm_eps),
        ),
        (
            "gemma4.attention.sliding_window".to_string(),
            MetadataValueSpec::UInt32(4),
        ),
        (
            "gemma4.attention.shared_kv_layers".to_string(),
            MetadataValueSpec::UInt32(0),
        ),
        (
            "gemma4.attention.sliding_window_pattern".to_string(),
            MetadataValueSpec::BoolArray(vec![true, false]),
        ),
        (
            "gemma4.final_logit_softcapping".to_string(),
            MetadataValueSpec::Float32(30.0),
        ),
        (
            "tokenizer.ggml.model".to_string(),
            MetadataValueSpec::String("gemma4".to_string()),
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
            MetadataValueSpec::Bool(false),
        ),
    ];

    build_gguf_fixture(3, metadata, synthetic_gemma4_tensors(&spec))
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

pub fn build_gguf_fixture(
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
        MetadataValueSpec::BoolArray(values) => {
            write_u32(bytes, 9);
            write_u32(bytes, 7);
            write_u64(bytes, values.len() as u64);
            for value in values {
                bytes.push(u8::from(*value));
            }
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
        MetadataValueSpec::Int32Array(values) => {
            write_u32(bytes, 9);
            write_u32(bytes, 5);
            write_u64(bytes, values.len() as u64);
            for value in values {
                write_i32(bytes, *value);
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
        MetadataValueSpec::UInt32Array(values) => {
            write_u32(bytes, 9);
            write_u32(bytes, 4);
            write_u64(bytes, values.len() as u64);
            for value in values {
                write_u32(bytes, *value);
            }
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

    tensors.push(random_q8_0_tensor(
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

fn synthetic_qwen35_hybrid_tensors(spec: &SyntheticLlamaSpec) -> Vec<TensorSpec> {
    let head_dim = spec.embedding_length / spec.attention_head_count;
    let q_width = spec.attention_head_count * head_dim;
    let kv_width = spec.attention_head_count_kv * head_dim;
    let state_size = 4;
    let group_count = 1;
    let inner_size = 8;
    let dt_rank = 2;
    let conv_kernel = 4;
    let conv_channels = state_size * group_count * 2 + inner_size;
    let head_v_dim = inner_size / dt_rank;
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
        if layer % 4 != 3 {
            tensors.push(random_f32_tensor(
                format!("blk.{layer}.attn_qkv.weight"),
                vec![spec.embedding_length, conv_channels],
                &mut seed,
            ));
            tensors.push(random_f32_tensor(
                format!("blk.{layer}.attn_gate.weight"),
                vec![spec.embedding_length, inner_size],
                &mut seed,
            ));
            tensors.push(random_f32_tensor(
                format!("blk.{layer}.ssm_alpha.weight"),
                vec![spec.embedding_length, dt_rank],
                &mut seed,
            ));
            tensors.push(random_f32_tensor(
                format!("blk.{layer}.ssm_beta.weight"),
                vec![spec.embedding_length, dt_rank],
                &mut seed,
            ));
            tensors.push(random_f32_tensor(
                format!("blk.{layer}.ssm_a"),
                vec![dt_rank],
                &mut seed,
            ));
            tensors.push(random_f32_tensor(
                format!("blk.{layer}.ssm_dt.bias"),
                vec![dt_rank],
                &mut seed,
            ));
            tensors.push(random_f32_tensor(
                format!("blk.{layer}.ssm_norm.weight"),
                vec![head_v_dim],
                &mut seed,
            ));
            tensors.push(random_f32_tensor(
                format!("blk.{layer}.ssm_out.weight"),
                vec![inner_size, spec.embedding_length],
                &mut seed,
            ));
            tensors.push(random_f32_tensor(
                format!("blk.{layer}.ssm_conv1d.weight"),
                vec![conv_kernel, conv_channels],
                &mut seed,
            ));
        } else {
            tensors.push(random_f32_tensor(
                format!("blk.{layer}.attn_q.weight"),
                vec![spec.embedding_length, q_width * 2],
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
                vec![q_width, spec.embedding_length],
                &mut seed,
            ));
            tensors.push(random_f32_tensor(
                format!("blk.{layer}.attn_q_norm.weight"),
                vec![head_dim],
                &mut seed,
            ));
            tensors.push(random_f32_tensor(
                format!("blk.{layer}.attn_k_norm.weight"),
                vec![head_dim],
                &mut seed,
            ));
        }
        tensors.push(random_f32_tensor(
            format!("blk.{layer}.post_attention_norm.weight"),
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

fn synthetic_qwen35_hybrid_moe_tensors(
    spec: &SyntheticLlamaSpec,
    expert_count: usize,
) -> Vec<TensorSpec> {
    let mut tensors = synthetic_qwen35_hybrid_tensors(spec);
    tensors.retain(|tensor| {
        !tensor.name.ends_with(".ffn_gate.weight")
            && !tensor.name.ends_with(".ffn_up.weight")
            && !tensor.name.ends_with(".ffn_down.weight")
    });
    let mut seed = spec.seed ^ 0x4D4F_455F_5133_3500;
    for layer in 0..spec.block_count {
        tensors.push(random_f32_tensor(
            format!("blk.{layer}.ffn_gate_inp.weight"),
            vec![spec.embedding_length, expert_count],
            &mut seed,
        ));
        tensors.push(random_f32_tensor(
            format!("blk.{layer}.ffn_gate_exps.weight"),
            vec![
                spec.embedding_length,
                spec.feed_forward_length,
                expert_count,
            ],
            &mut seed,
        ));
        tensors.push(random_f32_tensor(
            format!("blk.{layer}.ffn_up_exps.weight"),
            vec![
                spec.embedding_length,
                spec.feed_forward_length,
                expert_count,
            ],
            &mut seed,
        ));
        tensors.push(random_f32_tensor(
            format!("blk.{layer}.ffn_down_exps.weight"),
            vec![
                spec.feed_forward_length,
                spec.embedding_length,
                expert_count,
            ],
            &mut seed,
        ));
    }
    tensors
}

fn synthetic_qwen3_moe_tensors(spec: &SyntheticLlamaSpec, expert_count: usize) -> Vec<TensorSpec> {
    synthetic_qwen3_moe_tensors_with_layout(spec, expert_count, false)
}

fn synthetic_qwen3_moe_tensors_with_layout(
    spec: &SyntheticLlamaSpec,
    expert_count: usize,
    packed: bool,
) -> Vec<TensorSpec> {
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
            format!("blk.{layer}.ffn_gate_inp.weight"),
            vec![spec.embedding_length, expert_count],
            &mut seed,
        ));
        if packed {
            tensors.push(random_f32_tensor(
                format!("blk.{layer}.ffn_gate_exps.weight"),
                vec![
                    spec.embedding_length,
                    spec.feed_forward_length,
                    expert_count,
                ],
                &mut seed,
            ));
            tensors.push(random_f32_tensor(
                format!("blk.{layer}.ffn_up_exps.weight"),
                vec![
                    spec.embedding_length,
                    spec.feed_forward_length,
                    expert_count,
                ],
                &mut seed,
            ));
            tensors.push(random_f32_tensor(
                format!("blk.{layer}.ffn_down_exps.weight"),
                vec![
                    spec.feed_forward_length,
                    spec.embedding_length,
                    expert_count,
                ],
                &mut seed,
            ));
        } else {
            for expert in 0..expert_count {
                tensors.push(random_f32_tensor(
                    format!("blk.{layer}.ffn_gate.{expert}.weight"),
                    vec![spec.embedding_length, spec.feed_forward_length],
                    &mut seed,
                ));
                tensors.push(random_f32_tensor(
                    format!("blk.{layer}.ffn_up.{expert}.weight"),
                    vec![spec.embedding_length, spec.feed_forward_length],
                    &mut seed,
                ));
                tensors.push(random_f32_tensor(
                    format!("blk.{layer}.ffn_down.{expert}.weight"),
                    vec![spec.feed_forward_length, spec.embedding_length],
                    &mut seed,
                ));
            }
        }
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

fn synthetic_q8_0_llama_tensors(spec: &SyntheticLlamaSpec) -> Vec<TensorSpec> {
    let head_dim = spec.embedding_length / spec.attention_head_count;
    let kv_width = spec.attention_head_count_kv * head_dim;
    let mut seed = spec.seed;
    let mut tensors = Vec::new();

    tensors.push(random_q8_0_tensor(
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
        tensors.push(random_q8_0_tensor(
            format!("blk.{layer}.attn_q.weight"),
            vec![spec.embedding_length, spec.embedding_length],
            &mut seed,
        ));
        tensors.push(random_q8_0_tensor(
            format!("blk.{layer}.attn_k.weight"),
            vec![spec.embedding_length, kv_width],
            &mut seed,
        ));
        tensors.push(random_q8_0_tensor(
            format!("blk.{layer}.attn_v.weight"),
            vec![spec.embedding_length, kv_width],
            &mut seed,
        ));
        tensors.push(random_f32_tensor(
            format!("blk.{layer}.attn_q.bias"),
            vec![spec.embedding_length],
            &mut seed,
        ));
        tensors.push(random_f32_tensor(
            format!("blk.{layer}.attn_k.bias"),
            vec![kv_width],
            &mut seed,
        ));
        tensors.push(random_f32_tensor(
            format!("blk.{layer}.attn_v.bias"),
            vec![kv_width],
            &mut seed,
        ));
        tensors.push(random_q8_0_tensor(
            format!("blk.{layer}.attn_output.weight"),
            vec![spec.embedding_length, spec.embedding_length],
            &mut seed,
        ));
        tensors.push(random_f32_tensor(
            format!("blk.{layer}.ffn_norm.weight"),
            vec![spec.embedding_length],
            &mut seed,
        ));
        tensors.push(random_q8_0_tensor(
            format!("blk.{layer}.ffn_gate.weight"),
            vec![spec.embedding_length, spec.feed_forward_length],
            &mut seed,
        ));
        tensors.push(random_q8_0_tensor(
            format!("blk.{layer}.ffn_up.weight"),
            vec![spec.embedding_length, spec.feed_forward_length],
            &mut seed,
        ));
        tensors.push(random_q8_0_tensor(
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
    tensors.push(random_q8_0_tensor(
        "output.weight",
        vec![spec.embedding_length, spec.vocab_size],
        &mut seed,
    ));

    tensors
}

fn synthetic_float_llama_tensors(spec: &SyntheticLlamaSpec, dtype: DType) -> Vec<TensorSpec> {
    assert!(matches!(dtype, DType::F32 | DType::F16 | DType::BF16));
    let head_dim = spec.embedding_length / spec.attention_head_count;
    let kv_width = spec.attention_head_count_kv * head_dim;
    let mut seed = spec.seed;
    let mut tensors = Vec::new();

    tensors.push(random_float_tensor(
        "token_embd.weight",
        vec![spec.embedding_length, spec.vocab_size],
        dtype,
        &mut seed,
    ));
    for layer in 0..spec.block_count {
        tensors.push(random_float_tensor(
            format!("blk.{layer}.attn_norm.weight"),
            vec![spec.embedding_length],
            dtype,
            &mut seed,
        ));
        tensors.push(random_float_tensor(
            format!("blk.{layer}.attn_q.weight"),
            vec![spec.embedding_length, spec.embedding_length],
            dtype,
            &mut seed,
        ));
        tensors.push(random_float_tensor(
            format!("blk.{layer}.attn_k.weight"),
            vec![spec.embedding_length, kv_width],
            dtype,
            &mut seed,
        ));
        tensors.push(random_float_tensor(
            format!("blk.{layer}.attn_v.weight"),
            vec![spec.embedding_length, kv_width],
            dtype,
            &mut seed,
        ));
        tensors.push(random_float_tensor(
            format!("blk.{layer}.attn_output.weight"),
            vec![spec.embedding_length, spec.embedding_length],
            dtype,
            &mut seed,
        ));
        tensors.push(random_float_tensor(
            format!("blk.{layer}.ffn_norm.weight"),
            vec![spec.embedding_length],
            dtype,
            &mut seed,
        ));
        tensors.push(random_float_tensor(
            format!("blk.{layer}.ffn_gate.weight"),
            vec![spec.embedding_length, spec.feed_forward_length],
            dtype,
            &mut seed,
        ));
        tensors.push(random_float_tensor(
            format!("blk.{layer}.ffn_up.weight"),
            vec![spec.embedding_length, spec.feed_forward_length],
            dtype,
            &mut seed,
        ));
        tensors.push(random_float_tensor(
            format!("blk.{layer}.ffn_down.weight"),
            vec![spec.feed_forward_length, spec.embedding_length],
            dtype,
            &mut seed,
        ));
    }
    tensors.push(random_float_tensor(
        "output_norm.weight",
        vec![spec.embedding_length],
        dtype,
        &mut seed,
    ));
    tensors.push(random_float_tensor(
        "output.weight",
        vec![spec.embedding_length, spec.vocab_size],
        dtype,
        &mut seed,
    ));

    tensors
}

fn synthetic_q4_0_llama_tensors(spec: &SyntheticLlamaSpec) -> Vec<TensorSpec> {
    let mut tensors = synthetic_q8_0_llama_tensors(spec);
    for tensor in &mut tensors {
        if tensor.dtype == DType::Q8_0 {
            let cols = tensor.dimensions.first().copied().unwrap_or_default();
            let rows = if tensor.dimensions.len() <= 1 {
                1
            } else {
                tensor.dimensions[1..].iter().copied().product()
            };
            let values = q8_0_to_f32_rows(&tensor.data, rows, cols);
            tensor.data = q4_0_tensor_bytes(&values, rows, cols);
            tensor.dtype = DType::Q4_0;
        }
    }
    tensors
}

fn synthetic_q4_k_llama_tensors(spec: &SyntheticLlamaSpec) -> Vec<TensorSpec> {
    let mut tensors = synthetic_q8_0_llama_tensors(spec);
    for tensor in &mut tensors {
        if tensor.dtype == DType::Q8_0 {
            let cols = tensor.dimensions.first().copied().unwrap_or_default();
            let rows = if tensor.dimensions.len() <= 1 {
                1
            } else {
                tensor.dimensions[1..].iter().copied().product()
            };
            let values = q8_0_to_f32_rows(&tensor.data, rows, cols);
            tensor.data = q4_k_tensor_bytes(&values, rows, cols);
            tensor.dtype = DType::Q4_K;
        }
    }
    tensors
}

fn synthetic_q5_k_llama_tensors(spec: &SyntheticLlamaSpec) -> Vec<TensorSpec> {
    let mut tensors = synthetic_q8_0_llama_tensors(spec);
    for tensor in &mut tensors {
        if tensor.dtype == DType::Q8_0 {
            let cols = tensor.dimensions.first().copied().unwrap_or_default();
            let rows = if tensor.dimensions.len() <= 1 {
                1
            } else {
                tensor.dimensions[1..].iter().copied().product()
            };
            let values = q8_0_to_f32_rows(&tensor.data, rows, cols);
            tensor.data = q5_k_tensor_bytes(&values, rows, cols);
            tensor.dtype = DType::Q5_K;
        }
    }
    tensors
}

fn synthetic_q6_k_llama_tensors(spec: &SyntheticLlamaSpec) -> Vec<TensorSpec> {
    let mut tensors = synthetic_q8_0_llama_tensors(spec);
    for tensor in &mut tensors {
        if tensor.dtype == DType::Q8_0 {
            let cols = tensor.dimensions.first().copied().unwrap_or_default();
            let rows = if tensor.dimensions.len() <= 1 {
                1
            } else {
                tensor.dimensions[1..].iter().copied().product()
            };
            let values = q8_0_to_f32_rows(&tensor.data, rows, cols);
            tensor.data = q6_k_tensor_bytes(&values, rows, cols);
            tensor.dtype = DType::Q6_K;
        }
    }
    tensors
}

fn synthetic_gemma4_tensors(spec: &SyntheticLlamaSpec) -> Vec<TensorSpec> {
    let mut seed = spec.seed;
    let mut tensors = Vec::new();

    tensors.push(random_f32_tensor(
        "token_embd.weight",
        vec![spec.embedding_length, spec.vocab_size],
        &mut seed,
    ));

    let layer_shapes = [
        (true, 2usize, 2usize, 2usize, true),
        (false, 2usize, 1usize, 4usize, false),
    ];

    for (layer, (_is_swa, heads, kv_heads, head_dim, has_v)) in layer_shapes.into_iter().enumerate()
    {
        let q_width = heads * head_dim;
        let kv_width = kv_heads * head_dim;

        tensors.push(random_f32_tensor(
            format!("blk.{layer}.attn_norm.weight"),
            vec![spec.embedding_length],
            &mut seed,
        ));
        tensors.push(random_f32_tensor(
            format!("blk.{layer}.attn_q.weight"),
            vec![spec.embedding_length, q_width],
            &mut seed,
        ));
        tensors.push(random_f32_tensor(
            format!("blk.{layer}.attn_k.weight"),
            vec![spec.embedding_length, kv_width],
            &mut seed,
        ));
        if has_v {
            tensors.push(random_f32_tensor(
                format!("blk.{layer}.attn_v.weight"),
                vec![spec.embedding_length, kv_width],
                &mut seed,
            ));
        }
        tensors.push(random_f32_tensor(
            format!("blk.{layer}.attn_output.weight"),
            vec![q_width, spec.embedding_length],
            &mut seed,
        ));
        tensors.push(random_f32_tensor(
            format!("blk.{layer}.attn_q_norm.weight"),
            vec![head_dim],
            &mut seed,
        ));
        tensors.push(random_f32_tensor(
            format!("blk.{layer}.attn_k_norm.weight"),
            vec![head_dim],
            &mut seed,
        ));
        tensors.push(random_f32_tensor(
            format!("blk.{layer}.post_attention_norm.weight"),
            vec![spec.embedding_length],
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
        tensors.push(random_f32_tensor(
            format!("blk.{layer}.post_ffw_norm.weight"),
            vec![spec.embedding_length],
            &mut seed,
        ));
        tensors.push(TensorSpec {
            name: format!("blk.{layer}.layer_output_scale.weight"),
            data: f32_tensor_bytes(&[1.0]),
            dimensions: vec![1],
            dtype: DType::F32,
        });
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

fn random_float_tensor(
    name: impl Into<String>,
    dimensions: Vec<usize>,
    dtype: DType,
    seed: &mut u64,
) -> TensorSpec {
    let values = (0..product(&dimensions))
        .map(|_| next_random_f32(seed))
        .collect::<Vec<_>>();
    TensorSpec {
        name: name.into(),
        data: float_tensor_bytes(&values, dtype),
        dimensions,
        dtype,
    }
}

fn random_q8_0_tensor(
    name: impl Into<String>,
    dimensions: Vec<usize>,
    seed: &mut u64,
) -> TensorSpec {
    let cols = dimensions.first().copied().unwrap_or_default();
    let rows = if dimensions.len() <= 1 {
        1
    } else {
        dimensions[1..].iter().copied().product()
    };
    let values = (0..product(&dimensions))
        .map(|_| next_random_f32(seed))
        .collect::<Vec<_>>();
    TensorSpec {
        name: name.into(),
        data: q8_0_tensor_bytes(&values, rows, cols),
        dimensions,
        dtype: DType::Q8_0,
    }
}

fn float_tensor_bytes(values: &[f32], dtype: DType) -> Vec<u8> {
    match dtype {
        DType::F32 => f32_tensor_bytes(values),
        DType::F16 => values
            .iter()
            .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
            .collect(),
        DType::BF16 => values
            .iter()
            .flat_map(|value| bf16::from_f32(*value).to_bits().to_le_bytes())
            .collect(),
        other => panic!("unsupported synthetic float dtype {other:?}"),
    }
}

fn q8_0_tensor_bytes(values: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(values.len(), rows * cols);
    assert_eq!(cols % 32, 0);
    let mut bytes = Vec::with_capacity(rows * (cols / 32) * DType::Q8_0.block_bytes());
    for row in values.chunks_exact(cols) {
        for block in row.chunks_exact(32) {
            let max_abs = block.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
            bytes.extend_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
            bytes.extend(block.iter().map(|value| {
                let quant = (value / scale).round().clamp(-127.0, 127.0);
                quant as i8 as u8
            }));
        }
    }
    bytes
}

fn q8_0_to_f32_rows(bytes: &[u8], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(cols % 32, 0);
    let blocks_per_row = cols / 32;
    let mut values = vec![0.0; rows * cols];
    for row in 0..rows {
        for block in 0..blocks_per_row {
            let block_offset = (row * blocks_per_row + block) * DType::Q8_0.block_bytes();
            let scale = f16::from_bits(u16::from_le_bytes([
                bytes[block_offset],
                bytes[block_offset + 1],
            ]))
            .to_f32();
            for lane in 0..32 {
                values[row * cols + block * 32 + lane] =
                    scale * bytes[block_offset + 2 + lane] as i8 as f32;
            }
        }
    }
    values
}

fn q4_0_tensor_bytes(values: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(values.len(), rows * cols);
    assert_eq!(cols % 32, 0);
    let mut bytes = Vec::with_capacity(rows * (cols / 32) * DType::Q4_0.block_bytes());
    for row in values.chunks_exact(cols) {
        for block in row.chunks_exact(32) {
            let max_abs = block.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 7.0 };
            bytes.extend_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
            for lane in 0..16 {
                let low = ((block[lane] / scale).round() as i32 + 8).clamp(0, 15) as u8;
                let high = ((block[lane + 16] / scale).round() as i32 + 8).clamp(0, 15) as u8;
                bytes.push(low | (high << 4));
            }
        }
    }
    bytes
}

fn q4_k_tensor_bytes(values: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(values.len(), rows * cols);
    assert_eq!(cols % 256, 0);
    let mut bytes = Vec::with_capacity(rows * (cols / 256) * DType::Q4_K.block_bytes());
    for row in values.chunks_exact(cols) {
        for block in row.chunks_exact(256) {
            let max_abs = block.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 7.0 };
            bytes.extend_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
            bytes.extend_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
            // ponytail: all 8 groups use scale=1,min=8; enough for CUDA/CPU parity fixtures.
            bytes.extend_from_slice(&[1, 1, 1, 1, 8, 8, 8, 8, 0x81, 0x81, 0x81, 0x81]);
            for group in 0..4 {
                for lane in 0..32 {
                    let low =
                        ((block[group * 64 + lane] / scale).round() as i32 + 8).clamp(0, 15) as u8;
                    let high = ((block[group * 64 + 32 + lane] / scale).round() as i32 + 8)
                        .clamp(0, 15) as u8;
                    bytes.push(low | (high << 4));
                }
            }
        }
    }
    bytes
}

fn q5_k_tensor_bytes(values: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(values.len(), rows * cols);
    assert_eq!(cols % 256, 0);
    let mut bytes = Vec::with_capacity(rows * (cols / 256) * DType::Q5_K.block_bytes());
    for row in values.chunks_exact(cols) {
        for block in row.chunks_exact(256) {
            let max_abs = block.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 15.0 };
            let mut qh = [0u8; 32];
            let mut qs = [0u8; 128];
            for group in 0..4 {
                for lane in 0..32 {
                    let low = q5_quant(block[group * 64 + lane], scale);
                    let high = q5_quant(block[group * 64 + 32 + lane], scale);
                    qs[group * 32 + lane] = (low & 0x0f) | ((high & 0x0f) << 4);
                    if low >= 16 {
                        qh[lane] |= 1 << (group * 2);
                    }
                    if high >= 16 {
                        qh[lane] |= 1 << (group * 2 + 1);
                    }
                }
            }
            bytes.extend_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
            bytes.extend_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
            // ponytail: all 8 groups use scale=1,min=16; enough for CUDA/CPU parity fixtures.
            bytes.extend_from_slice(&[1, 1, 1, 1, 0x50, 0x50, 0x50, 0x50, 1, 1, 1, 1]);
            bytes.extend_from_slice(&qh);
            bytes.extend_from_slice(&qs);
        }
    }
    bytes
}

fn q5_quant(value: f32, scale: f32) -> u8 {
    ((value / scale).round() as i32 + 16).clamp(0, 31) as u8
}

fn q6_k_tensor_bytes(values: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(values.len(), rows * cols);
    assert_eq!(cols % 256, 0);
    let mut bytes = Vec::with_capacity(rows * (cols / 256) * DType::Q6_K.block_bytes());
    for row in values.chunks_exact(cols) {
        for block in row.chunks_exact(256) {
            let max_abs = block.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 31.0 };
            let mut ql = [0u8; 128];
            let mut qh = [0u8; 64];
            for group in 0..2 {
                for lane in 0..32 {
                    let base = group * 128;
                    let q1 = q6_quant(block[base + lane], scale);
                    let q2 = q6_quant(block[base + 32 + lane], scale);
                    let q3 = q6_quant(block[base + 64 + lane], scale);
                    let q4 = q6_quant(block[base + 96 + lane], scale);
                    ql[group * 64 + lane] = (q1 & 0x0f) | ((q3 & 0x0f) << 4);
                    ql[group * 64 + 32 + lane] = (q2 & 0x0f) | ((q4 & 0x0f) << 4);
                    qh[group * 32 + lane] = ((q1 >> 4) & 0x03)
                        | (((q2 >> 4) & 0x03) << 2)
                        | (((q3 >> 4) & 0x03) << 4)
                        | (((q4 >> 4) & 0x03) << 6);
                }
            }
            bytes.extend_from_slice(&ql);
            bytes.extend_from_slice(&qh);
            bytes.extend_from_slice(&[1u8; 16]);
            bytes.extend_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
        }
    }
    bytes
}

fn q6_quant(value: f32, scale: f32) -> u8 {
    ((value / scale).round() as i32 + 32).clamp(0, 63) as u8
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
