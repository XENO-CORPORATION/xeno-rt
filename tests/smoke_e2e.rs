mod common;

use std::{cmp::Ordering, sync::Arc, time::Instant};
use xrt_core::KvCache;
use xrt_gguf::GgufFile;
use xrt_models::LlamaModel;
use xrt_runtime::PagedKvCache;
use xrt_tokenizer::Tokenizer;

#[test]
fn synthetic_llama_pipeline_handles_one_token() {
    run_synthetic_smoke(1);
}

#[test]
#[ignore = "smoke test"]
fn synthetic_llama_pipeline_runs_eight_tokens() {
    run_synthetic_smoke(8);
}

fn run_synthetic_smoke(token_count: usize) {
    let spec = common::SyntheticLlamaSpec::tiny();
    let fixture =
        common::build_synthetic_llama_fixture(spec.clone()).expect("fixture should be created");
    let gguf = Arc::new(GgufFile::open(fixture.path()).expect("GGUF should parse"));
    let tokenizer = Tokenizer::from_gguf(gguf.as_ref()).expect("tokenizer should load");
    let model = LlamaModel::from_gguf(gguf).expect("model should load");
    let mut cache = PagedKvCache::new(model.config().block_count, model.config().kv_width(), 4);
    let mut current = tokenizer
        .special_tokens()
        .bos
        .expect("synthetic tokenizer should have a BOS token");
    let mut sampled = Vec::with_capacity(token_count);

    assert_eq!(tokenizer.vocab_size(), spec.vocab_size);
    assert_eq!(model.config().block_count, 2);
    assert_eq!(model.config().embedding_length, 64);
    assert_eq!(model.config().attention_head_count, 4);

    let started = Instant::now();
    for position in 0..token_count {
        let logits = model
            .forward_token(current, position, &mut cache)
            .expect("forward pass should succeed");
        assert_eq!(logits.len(), spec.vocab_size);
        assert!(logits.iter().all(|value| value.is_finite()));

        let next = logits
            .iter()
            .enumerate()
            .max_by(|(_, lhs), (_, rhs)| lhs.partial_cmp(rhs).unwrap_or(Ordering::Equal))
            .map(|(index, _)| index as u32)
            .expect("logits should not be empty");
        assert!(next < spec.vocab_size as u32);
        assert!(tokenizer.token_to_piece(next).is_some());

        sampled.push(next);
        for layer in 0..model.config().block_count {
            assert_eq!(cache.len(layer), position + 1);
        }
        current = next;
    }

    let elapsed = started.elapsed();
    let tokens_per_second = token_count as f64 / elapsed.as_secs_f64().max(f64::EPSILON);
    println!(
        "synthetic llama smoke: {token_count} tokens in {:?} ({tokens_per_second:.2} tok/s) -> {:?}",
        elapsed, sampled
    );
}
