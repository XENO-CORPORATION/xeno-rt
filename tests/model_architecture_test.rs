mod common;

use std::sync::Arc;
use xrt_gguf::GgufFile;
use xrt_models::LlamaModel;

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
