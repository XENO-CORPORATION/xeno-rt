mod common;

use std::sync::Arc;
use xrt_core::KvCache;
use xrt_gguf::GgufFile;
use xrt_models::{LlamaConfig, LlamaModel};
use xrt_runtime::PagedKvCache;

#[test]
fn qwen2_architecture_loads_qwen2_metadata_prefix() {
    let fixture = common::build_synthetic_llama_fixture_with_architecture(
        common::SyntheticLlamaSpec::tiny(),
        "qwen2",
        "qwen2",
    )
    .expect("fixture should be created");
    let gguf = Arc::new(GgufFile::open(fixture.path()).expect("GGUF should parse"));

    let model = LlamaModel::from_gguf(gguf).expect("qwen2 should load");

    assert_eq!(model.config().architecture, "qwen2");
    assert_eq!(model.config().context_length, 32);
}

#[test]
fn qwen3_next_architecture_accepts_qwen35_metadata_prefix() {
    let fixture = common::build_synthetic_llama_fixture_with_architecture(
        common::SyntheticLlamaSpec::tiny(),
        "qwen3_next",
        "qwen35",
    )
    .expect("fixture should be created");
    let gguf = Arc::new(GgufFile::open(fixture.path()).expect("GGUF should parse"));

    let model = LlamaModel::from_gguf(gguf).expect("qwen3_next alias should load");

    assert_eq!(model.config().architecture, "qwen3_next");
    assert_eq!(model.config().context_length, 32);
}

#[test]
fn qwen35_mtp_artifact_separates_target_trunk_from_appended_predictor() {
    let (fixture, trunk) =
        common::build_synthetic_qwen35_mtp_fixture().expect("fixture should be created");
    let gguf = Arc::new(GgufFile::open(fixture.path()).expect("GGUF should parse"));

    let model = LlamaModel::from_gguf(gguf).expect("Qwen35 MTP target trunk should load");
    let config = model.config();

    assert_eq!(config.block_count, trunk.block_count);
    assert_eq!(config.total_block_count, trunk.block_count + 1);
    assert_eq!(config.nextn_predict_layers, 1);
    assert!(config.has_nextn_predictor());
    assert_eq!(
        config.nextn_layer_range(),
        trunk.block_count..trunk.block_count + 1
    );
}

#[test]
fn qwen35_uses_explicit_attention_dimensions_when_hidden_size_is_not_head_aligned() {
    use common::MetadataValueSpec::{Float32, String as MetadataString, StringArray, UInt32};

    let fixture = common::build_gguf_fixture(
        3,
        vec![
            (
                "general.architecture".to_string(),
                MetadataString("qwen35".to_string()),
            ),
            (
                "tokenizer.ggml.tokens".to_string(),
                StringArray(vec!["<unk>".to_string()]),
            ),
            ("qwen35.context_length".to_string(), UInt32(262_144)),
            ("qwen35.embedding_length".to_string(), UInt32(5_120)),
            ("qwen35.feed_forward_length".to_string(), UInt32(17_408)),
            ("qwen35.block_count".to_string(), UInt32(65)),
            ("qwen35.nextn_predict_layers".to_string(), UInt32(1)),
            ("qwen35.attention.head_count".to_string(), UInt32(24)),
            ("qwen35.attention.head_count_kv".to_string(), UInt32(4)),
            ("qwen35.attention.key_length".to_string(), UInt32(256)),
            ("qwen35.attention.value_length".to_string(), UInt32(256)),
            ("qwen35.rope.dimension_count".to_string(), UInt32(64)),
            (
                "qwen35.attention.layer_norm_rms_epsilon".to_string(),
                Float32(1e-6),
            ),
        ],
        Vec::new(),
    )
    .expect("fixture should be created");
    let gguf = GgufFile::open(fixture.path()).expect("GGUF should parse");

    let config = LlamaConfig::from_gguf(&gguf).expect("explicit Qwen head geometry should load");
    assert_eq!(config.head_dim(), 256);
    assert_eq!(config.q_width(), 6_144);
    assert_eq!(config.kv_width(), 1_024);
    assert_eq!(config.block_count, 64);
}

#[test]
fn qwen3_omni_architecture_reports_native_stack_requirement() {
    let fixture = common::build_synthetic_llama_fixture_with_architecture(
        common::SyntheticLlamaSpec::tiny(),
        "qwen3_omni_moe",
        "qwen3_omni_moe",
    )
    .expect("fixture should be created");
    let gguf = Arc::new(GgufFile::open(fixture.path()).expect("GGUF should parse"));

    let error = LlamaModel::from_gguf(gguf)
        .err()
        .expect("qwen3 omni should be unsupported");
    let message = error.to_string();
    assert!(message.contains("native thinker/vision/audio modules"));
}

#[test]
fn glm_vision_architecture_reports_multimodal_requirement() {
    let fixture = common::build_synthetic_llama_fixture_with_architecture(
        common::SyntheticLlamaSpec::tiny(),
        "glm46v",
        "glm46v",
    )
    .expect("fixture should be created");
    let gguf = Arc::new(GgufFile::open(fixture.path()).expect("GGUF should parse"));

    let error = LlamaModel::from_gguf(gguf)
        .err()
        .expect("glm vision should be unsupported");
    let message = error.to_string();
    assert!(message.contains("native multimodal stack"));
}

#[test]
fn gemma4_architecture_loads_and_runs_missing_v_fallback() {
    let fixture = common::build_synthetic_gemma4_fixture().expect("fixture should be created");
    let gguf = Arc::new(GgufFile::open(fixture.path()).expect("GGUF should parse"));

    let model = LlamaModel::from_gguf(gguf).expect("gemma4 should load");
    assert_eq!(model.config().architecture, "gemma4");
    assert!(model.config().is_gemma4());
    assert_eq!(model.config().q_width(), 8);
    assert_eq!(model.config().kv_width(), 4);

    let mut cache = PagedKvCache::new(
        model.config().block_count,
        model.config().kv_width(),
        model.config().context_length,
    );
    let mut logits = Vec::new();
    model
        .forward_token(0, 0, &mut cache, &mut logits)
        .expect("gemma4 forward should run");

    assert_eq!(logits.len(), model.config().vocab_size);
    assert!(logits.iter().all(|value| value.is_finite()));
    assert_eq!(cache.len(0), 1);
    assert_eq!(cache.len(1), 1);
}
