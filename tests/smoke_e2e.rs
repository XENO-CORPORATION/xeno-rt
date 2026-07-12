mod common;

use std::{cmp::Ordering, sync::Arc, time::Instant};
use xrt_core::KvCache;
use xrt_gguf::GgufFile;
use xrt_models::LlamaModel;
use xrt_runtime::{
    BackendDecodeBatchItem, BackendKind, BackendSession, GenerateRequest, KvCacheMode,
    PagedKvCache, RequestScheduler, Runtime, SchedulerConfig, SessionPolicy,
};
use xrt_tokenizer::Tokenizer;

#[cfg(feature = "cuda")]
static CUDA_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn synthetic_llama_pipeline_handles_one_token() {
    run_synthetic_smoke(1);
}

#[test]
fn generate_stream_reports_generated_token_count() {
    let spec = common::SyntheticLlamaSpec::tiny();
    let fixture = common::build_synthetic_llama_fixture(spec).expect("fixture should be created");
    let runtime =
        Runtime::load_with_backend(fixture.path(), BackendKind::Cpu).expect("runtime should load");
    let mut session = runtime.new_session();
    let request = GenerateRequest {
        prompt: "hello".to_string(),
        max_tokens: 1,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        ..Default::default()
    };

    let mut pieces = 0usize;
    let generated = session
        .generate_stream(&request, |_| pieces += 1)
        .expect("generation should succeed");

    assert!(generated <= request.max_tokens);
    assert!(pieces <= generated);
}

#[test]
fn repeated_cpu_prompt_reuses_an_immutable_prefix_snapshot() {
    let spec = common::SyntheticLlamaSpec::tiny();
    let fixture = common::build_synthetic_llama_fixture(spec).expect("fixture should be created");
    let runtime =
        Runtime::load_with_backend(fixture.path(), BackendKind::Cpu).expect("runtime should load");
    let request = GenerateRequest {
        prompt: "hello world".to_string(),
        max_tokens: 3,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(17),
        ..Default::default()
    };

    let first = runtime
        .new_session()
        .generate(&request)
        .expect("initial generation should populate the prefix cache");
    let after_first = runtime.prefix_cache_status();
    assert_eq!(after_first.lookups, 1);
    assert_eq!(after_first.misses, 1);
    assert_eq!(after_first.hits, 0);
    assert_eq!(after_first.inserts, 1);
    assert_eq!(after_first.entries, 1);

    let second = runtime
        .new_session()
        .generate(&request)
        .expect("repeated generation should attach the cached prefix");
    assert_eq!(second, first);
    let after_second = runtime.prefix_cache_status();
    assert_eq!(after_second.lookups, 2);
    assert_eq!(after_second.hits, 1);
    assert_eq!(after_second.misses, 1);
    assert!(after_second.prefill_tokens_saved >= 8);
    assert_eq!(after_second.hit_rate, 0.5);
}

#[test]
fn scheduled_chunked_prefill_matches_unscheduled_generation() {
    let spec = common::SyntheticLlamaSpec::tiny();
    let fixture = common::build_synthetic_llama_fixture(spec).expect("fixture should be created");
    let expected_runtime =
        Runtime::load_with_backend(fixture.path(), BackendKind::Cpu).expect("runtime should load");
    let scheduled_runtime =
        Runtime::load_with_backend(fixture.path(), BackendKind::Cpu).expect("runtime should load");
    let request = GenerateRequest {
        prompt: "hello".to_string(),
        max_tokens: 4,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(7),
        ..Default::default()
    };

    let expected = expected_runtime
        .new_session()
        .generate(&request)
        .expect("unscheduled generation should succeed");
    let scheduler = Arc::new(RequestScheduler::new(
        SchedulerConfig::new(2, 2, 4)
            .unwrap()
            .with_execution_policy(1, 2)
            .unwrap(),
    ));
    let actual = scheduled_runtime
        .new_session()
        .generate_scheduled(&request, &scheduler)
        .expect("scheduled generation should succeed");

    assert_eq!(actual, expected);
    let status = scheduler.status();
    assert_eq!(status.active_execution_phase, None);
    assert!(status.completed_prefill_turns >= 2);
}

#[test]
fn gpu_resource_status_tracks_active_sessions() {
    let spec = common::SyntheticLlamaSpec::tiny();
    let fixture = common::build_synthetic_llama_fixture(spec).expect("fixture should be created");
    let runtime =
        Runtime::load_with_backend(fixture.path(), BackendKind::Cpu).expect("runtime should load");

    let initial_status = runtime.gpu_resource_status();
    assert_eq!(initial_status.active_sessions, 0);
    assert_eq!(initial_status.free_vram_bytes, None);
    assert_eq!(initial_status.requested_kv_cache_mode, None);
    {
        let adaptive = runtime
            .clone()
            .new_session_with_cache_mode(KvCacheMode::AgentAdaptive);
        let status = adaptive.gpu_resource_status();
        assert_eq!(status.requested_kv_cache_mode, Some("agent_adaptive"));
        assert_eq!(status.kv_cache_mode, Some("agent_adaptive"));
    }
    assert_eq!(runtime.gpu_resource_status().active_sessions, 0);
    {
        let first = runtime.new_session();
        assert_eq!(runtime.gpu_resource_status().active_sessions, 1);
        assert_eq!(first.gpu_resource_status().active_sessions, 1);
        {
            let second = runtime.new_session();
            assert_eq!(runtime.gpu_resource_status().active_sessions, 2);
            assert_eq!(second.gpu_resource_status().active_sessions, 2);
        }
        assert_eq!(runtime.gpu_resource_status().active_sessions, 1);
        assert_eq!(first.gpu_resource_status().active_sessions, 1);
    }
    assert_eq!(runtime.gpu_resource_status().active_sessions, 0);
}

#[test]
fn synthetic_float_fixtures_decode_on_cpu() {
    let spec = common::SyntheticLlamaSpec::tiny();

    let f16 = common::build_synthetic_f16_llama_fixture(spec.clone())
        .expect("F16 fixture should be created");
    assert_cpu_runtime_decodes_one_token(f16.path(), &spec);

    let bf16 = common::build_synthetic_bf16_llama_fixture(spec.clone())
        .expect("BF16 fixture should be created");
    assert_cpu_runtime_decodes_one_token(bf16.path(), &spec);
}

#[test]
#[ignore = "smoke test"]
fn synthetic_llama_pipeline_runs_eight_tokens() {
    run_synthetic_smoke(8);
}

fn assert_cpu_runtime_decodes_one_token(
    fixture_path: &std::path::Path,
    spec: &common::SyntheticLlamaSpec,
) {
    let runtime = Runtime::load_with_backend(fixture_path, BackendKind::Cpu)
        .expect("CPU runtime should load");
    let mut session = runtime
        .backend()
        .new_session(KvCacheMode::F32, runtime.backend().config().context_length);
    let mut logits = Vec::new();
    runtime
        .backend()
        .forward_token(spec.bos_token_id, 0, &mut session, &mut logits)
        .expect("CPU token should decode");

    assert_eq!(logits.len(), spec.vocab_size);
    assert!(logits.iter().all(|value| value.is_finite()));
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
    assert_cpu_backend_matches_direct_logits(fixture.path(), &tokenizer, &model, current);

    let started = Instant::now();
    for position in 0..token_count {
        let mut logits = Vec::new();
        model
            .forward_token(current, position, &mut cache, &mut logits)
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

fn assert_cpu_backend_matches_direct_logits(
    fixture_path: &std::path::Path,
    tokenizer: &Tokenizer,
    model: &LlamaModel,
    token: u32,
) {
    let runtime = Runtime::load_with_backend(fixture_path, BackendKind::Cpu)
        .expect("runtime should load CPU backend");
    let auto_runtime = Runtime::load_with_backend(fixture_path, BackendKind::Auto)
        .expect("runtime should load auto backend");
    assert_eq!(auto_runtime.active_backend(), BackendKind::Cpu);
    let mut direct_cache =
        PagedKvCache::new(model.config().block_count, model.config().kv_width(), 4);
    let mut backend_session = BackendSession::new_cpu(
        KvCacheMode::F32,
        runtime.backend().config().block_count,
        runtime.backend().config().kv_width(),
        4,
    );
    let mut direct_logits = Vec::new();
    let mut backend_logits = Vec::new();

    model
        .forward_token(token, 0, &mut direct_cache, &mut direct_logits)
        .expect("direct model forward should succeed");
    runtime
        .backend()
        .forward_token(token, 0, &mut backend_session, &mut backend_logits)
        .expect("backend forward should succeed");

    assert_eq!(direct_logits.len(), tokenizer.vocab_size());
    assert_eq!(direct_logits, backend_logits);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA-capable device and driver"]
fn cuda_q8_0_runtime_matches_cpu_logits() {
    // ponytail: cudarc/driver state is process-global enough for these smoke tests; serialize them.
    let _guard = CUDA_TEST_LOCK
        .lock()
        .expect("CUDA test lock should not be poisoned");
    let spec = common::SyntheticLlamaSpec::tiny();
    let fixture = common::build_synthetic_q8_0_llama_fixture(spec.clone())
        .expect("Q8_0 fixture should be created");
    let cpu_runtime = Runtime::load_with_backend(fixture.path(), BackendKind::Cpu)
        .expect("CPU runtime should load");
    let cuda_runtime = Runtime::load_with_backend(fixture.path(), BackendKind::Auto)
        .expect("auto runtime should load");
    assert_eq!(cuda_runtime.requested_backend(), BackendKind::Auto);
    assert_eq!(cuda_runtime.active_backend(), BackendKind::CudaResident);

    let status = cuda_runtime.gpu_resource_status();
    assert!(status.cuda_available);
    assert!(status.resident_q8_0_probe_available);
    assert!(status.resident_q8_0_layer0_probe_available);

    let mut cpu_session = cpu_runtime.backend().new_session(
        KvCacheMode::F32,
        cpu_runtime.backend().config().context_length,
    );
    let mut cuda_session = cuda_runtime.backend().new_session(
        KvCacheMode::F32,
        cuda_runtime.backend().config().context_length,
    );

    for (position, token) in [spec.bos_token_id, 3].into_iter().enumerate() {
        let mut cpu_logits = Vec::new();
        let mut cuda_logits = Vec::new();
        cpu_runtime
            .backend()
            .forward_token(token, position, &mut cpu_session, &mut cpu_logits)
            .expect("CPU token should decode");
        cuda_runtime
            .backend()
            .forward_token(token, position, &mut cuda_session, &mut cuda_logits)
            .expect("CUDA token should decode");

        assert_eq!(cpu_logits.len(), spec.vocab_size);
        assert_eq!(cuda_logits.len(), spec.vocab_size);
        assert_close(&cuda_logits, &cpu_logits, 1e-2);
        assert_eq!(
            cuda_session.cuda_graph_capture_status(),
            Some("captured"),
            "standard dense F32 decode should capture on token 0 and replay on token 1"
        );
    }

    // A one-token page forces KV growth before position 1. Growth must discard the
    // old pointer-bound graph and capture a replacement without changing logits.
    let mut cpu_growth_session = cpu_runtime.backend().new_session(KvCacheMode::F32, 1);
    let mut cuda_growth_session = cuda_runtime.backend().new_session(KvCacheMode::F32, 1);
    for (position, token) in [spec.bos_token_id, 3].into_iter().enumerate() {
        let mut cpu_logits = Vec::new();
        let mut cuda_logits = Vec::new();
        cpu_runtime
            .backend()
            .forward_token(token, position, &mut cpu_growth_session, &mut cpu_logits)
            .expect("CPU growth token should decode");
        cuda_runtime
            .backend()
            .forward_token(token, position, &mut cuda_growth_session, &mut cuda_logits)
            .expect("CUDA growth token should decode");

        assert_close(&cuda_logits, &cpu_logits, 1e-2);
        assert_eq!(
            cuda_growth_session.cuda_graph_capture_status(),
            Some("captured"),
            "KV growth should recapture a valid graph for the new capacity; last error: {:?}",
            cuda_growth_session.cuda_graph_last_error()
        );
    }

    let tokens = [spec.bos_token_id, 3];
    let mut cpu_session = cpu_runtime.backend().new_session(
        KvCacheMode::F32,
        cpu_runtime.backend().config().context_length,
    );
    let mut cuda_session = cuda_runtime.backend().new_session(
        KvCacheMode::F32,
        cuda_runtime.backend().config().context_length,
    );
    let cpu_logits = cpu_runtime
        .backend()
        .forward_batch(&tokens, 0, &mut cpu_session)
        .expect("CPU batch should decode");
    let cuda_logits = cuda_runtime
        .backend()
        .forward_batch(&tokens, 0, &mut cuda_session)
        .expect("CUDA batch should decode");
    assert_close(&cuda_logits, &cpu_logits, 1e-2);

    let mut cpu_session = cpu_runtime.backend().new_session(
        KvCacheMode::F32,
        cpu_runtime.backend().config().context_length,
    );
    let mut cuda_session = cuda_runtime.backend().new_session(
        KvCacheMode::F32,
        cuda_runtime.backend().config().context_length,
    );
    let cpu_all_logits = cpu_runtime
        .backend()
        .forward_batch_all_logits(&tokens, 0, &mut cpu_session)
        .expect("CPU all-logits batch should decode");
    let cuda_all_logits = cuda_runtime
        .backend()
        .forward_batch_all_logits(&tokens, 0, &mut cuda_session)
        .expect("CUDA all-logits batch should decode");
    assert_eq!(cuda_all_logits.len(), tokens.len() * spec.vocab_size);
    assert_close(&cuda_all_logits, &cpu_all_logits, 1e-2);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA-capable device and driver"]
fn cuda_repeated_prompt_reuses_immutable_prefix_kv() {
    let _guard = CUDA_TEST_LOCK
        .lock()
        .expect("CUDA test lock should not be poisoned");
    let spec = common::SyntheticLlamaSpec::tiny();
    let fixture =
        common::build_synthetic_q8_0_llama_fixture(spec).expect("Q8_0 fixture should be created");
    let runtime = Runtime::load_with_backend(fixture.path(), BackendKind::CudaResident)
        .expect("CUDA runtime should load");
    let request = GenerateRequest {
        prompt: "hello world".to_string(),
        max_tokens: 3,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(29),
        ..Default::default()
    };

    for (index, mode) in [
        KvCacheMode::F32,
        KvCacheMode::Q8,
        KvCacheMode::KeyQ4ValueQ8,
        KvCacheMode::AgentAdaptive,
    ]
    .into_iter()
    .enumerate()
    {
        let first = runtime
            .new_session_with_cache_mode(mode)
            .generate(&request)
            .expect("initial CUDA generation should populate the prefix cache");
        let after_first = runtime.prefix_cache_status();
        assert_eq!(after_first.lookups, (index * 2 + 1) as u64);
        assert_eq!(after_first.misses, (index + 1) as u64);
        assert_eq!(after_first.hits, index as u64);
        assert_eq!(after_first.entries, index + 1);
        assert!(after_first.resident_bytes > 0);

        let second = runtime
            .new_session_with_cache_mode(mode)
            .generate(&request)
            .expect("repeated CUDA generation should materialize the cached prefix");
        assert_eq!(second, first);
        let after_second = runtime.prefix_cache_status();
        assert_eq!(after_second.lookups, (index * 2 + 2) as u64);
        assert_eq!(after_second.hits, (index + 1) as u64);
        assert_eq!(after_second.misses, (index + 1) as u64);
        assert!(after_second.prefill_tokens_saved >= ((index + 1) * 8) as u64);
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA-capable device and driver"]
fn cuda_multi_sequence_decode_graph_matches_cpu_logits() {
    let _guard = CUDA_TEST_LOCK
        .lock()
        .expect("CUDA test lock should not be poisoned");
    let spec = common::SyntheticLlamaSpec::tiny();
    let fixture = common::build_synthetic_q8_0_llama_fixture(spec.clone())
        .expect("Q8_0 fixture should be created");
    let cpu_runtime = Runtime::load_with_backend(fixture.path(), BackendKind::Cpu)
        .expect("CPU runtime should load");
    let cuda_runtime = Runtime::load_with_backend(fixture.path(), BackendKind::Auto)
        .expect("CUDA runtime should load");
    assert_eq!(cuda_runtime.active_backend(), BackendKind::CudaResident);
    assert!(
        cuda_runtime
            .backend()
            .supports_multi_sequence_decode_batch(),
        "standard dense CUDA runtime should advertise decode batching"
    );

    let mut cpu_sessions = (0..2)
        .map(|_| {
            cpu_runtime.backend().new_session(
                KvCacheMode::F32,
                cpu_runtime.backend().config().context_length,
            )
        })
        .collect::<Vec<_>>();
    let mut cuda_sessions = (0..2)
        .map(|_| {
            cuda_runtime.backend().new_session(
                KvCacheMode::F32,
                cuda_runtime.backend().config().context_length,
            )
        })
        .collect::<Vec<_>>();

    for (position, tokens) in [[spec.bos_token_id, 3], [3, 4]].into_iter().enumerate() {
        let mut expected = Vec::with_capacity(tokens.len());
        for (session, token) in cpu_sessions.iter_mut().zip(tokens) {
            let mut logits = Vec::new();
            cpu_runtime
                .backend()
                .forward_token(token, position, session, &mut logits)
                .expect("CPU sequence token should decode");
            expected.push(logits);
        }

        let mut batch = cuda_sessions
            .drain(..)
            .zip(tokens)
            .enumerate()
            .map(|(index, (session, token))| {
                BackendDecodeBatchItem::new(index as u64 + 1, token, position, session)
            })
            .collect::<Vec<_>>();
        let execution = cuda_runtime
            .backend()
            .forward_token_batch(&mut batch)
            .expect("CUDA decode batch should execute");
        assert_eq!(
            execution.fused,
            position > 0,
            "the first batch should warm/capture and the second should replay one shared graph"
        );

        for (item, expected_logits) in batch.iter().zip(&expected) {
            assert_close(item.output_logits(), expected_logits, 1e-2);
            let expected_capture = if position == 0 {
                Some("captured")
            } else {
                Some("batch-captured")
            };
            assert_eq!(item.session().cuda_graph_capture_status(), expected_capture);
        }
        cuda_sessions = batch.into_iter().map(|item| item.into_parts().1).collect();
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA-capable device and driver"]
fn cuda_q8_0_tied_output_runtime_matches_cpu_logits() {
    // ponytail: cudarc/driver state is process-global enough for these smoke tests; serialize them.
    let _guard = CUDA_TEST_LOCK
        .lock()
        .expect("CUDA test lock should not be poisoned");
    let spec = common::SyntheticLlamaSpec::tiny();
    let fixture = common::build_synthetic_q8_0_tied_output_llama_fixture(spec.clone())
        .expect("Q8_0 tied-output fixture should be created");
    let cpu_runtime = Runtime::load_with_backend(fixture.path(), BackendKind::Cpu)
        .expect("CPU runtime should load");
    let cuda_runtime = Runtime::load_with_backend(fixture.path(), BackendKind::CudaResident)
        .expect("CUDA runtime should load");

    let status = cuda_runtime.gpu_resource_status();
    assert!(status.cuda_available);
    assert!(status.resident_q8_0_probe_available);
    assert!(status.resident_q8_0_layer0_probe_available);

    let mut cpu_session = cpu_runtime.backend().new_session(
        KvCacheMode::F32,
        cpu_runtime.backend().config().context_length,
    );
    let mut cuda_session = cuda_runtime.backend().new_session(
        KvCacheMode::F32,
        cuda_runtime.backend().config().context_length,
    );

    for (position, token) in [spec.bos_token_id, 3].into_iter().enumerate() {
        let mut cpu_logits = Vec::new();
        let mut cuda_logits = Vec::new();
        cpu_runtime
            .backend()
            .forward_token(token, position, &mut cpu_session, &mut cpu_logits)
            .expect("CPU token should decode");
        cuda_runtime
            .backend()
            .forward_token(token, position, &mut cuda_session, &mut cuda_logits)
            .expect("CUDA token should decode");

        assert_close(&cuda_logits, &cpu_logits, 1e-2);
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA-capable device and driver"]
fn cuda_q8_0_quantized_kv_modes_decode() {
    // ponytail: hardware-only smoke; compile it in safe checks, run manually when GPU execution is approved.
    let _guard = CUDA_TEST_LOCK
        .lock()
        .expect("CUDA test lock should not be poisoned");
    let spec = common::SyntheticLlamaSpec::tiny();
    let fixture = common::build_synthetic_q8_0_llama_fixture(spec.clone())
        .expect("Q8_0 fixture should be created");
    let cpu_runtime = Runtime::load_with_backend(fixture.path(), BackendKind::Cpu)
        .expect("CPU runtime should load");
    let cuda_runtime = Runtime::load_with_backend(fixture.path(), BackendKind::CudaResident)
        .expect("CUDA runtime should load");

    for (requested, effective) in [
        (KvCacheMode::Q8, KvCacheMode::Q8),
        (KvCacheMode::KeyQ4ValueQ8, KvCacheMode::KeyQ4ValueQ8),
        (KvCacheMode::AgentAdaptive, KvCacheMode::AgentAdaptive),
    ] {
        let status_session = cuda_runtime.clone().new_session_with_cache_mode(requested);
        let status = status_session.gpu_resource_status();
        assert_eq!(status.requested_kv_cache_mode, Some(requested.as_str()));
        assert_eq!(status.kv_cache_mode, Some(effective.as_str()));
        drop(status_session);

        let mut cpu_session = cpu_runtime
            .backend()
            .new_session(requested, cpu_runtime.backend().config().context_length);
        let mut cuda_session = cuda_runtime
            .backend()
            .new_session(requested, cuda_runtime.backend().config().context_length);
        if requested == KvCacheMode::AgentAdaptive {
            let policy = SessionPolicy {
                recent_window_tokens: 1,
                ..SessionPolicy::agent_adaptive()
            };
            cpu_session.configure_policy(policy.clone(), 0, &[]);
            cuda_session.configure_policy(policy, 0, &[]);
        }
        for (position, token) in [spec.bos_token_id, 3, 4, 5].into_iter().enumerate() {
            let mut cpu_logits = Vec::new();
            let mut logits = Vec::new();
            cpu_runtime
                .backend()
                .forward_token(token, position, &mut cpu_session, &mut cpu_logits)
                .expect("CPU token should decode with quantized KV");
            cuda_runtime
                .backend()
                .forward_token(token, position, &mut cuda_session, &mut logits)
                .expect("CUDA token should decode with quantized KV");
            assert_eq!(logits.len(), spec.vocab_size);
            assert!(logits.iter().all(|value| value.is_finite()));
            assert_close(&logits, &cpu_logits, 2e-2);
        }
        assert!(cuda_session.cuda_kv_allocated_bytes() > 0);
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA-capable device and driver"]
fn cuda_gemma4_f32_runtime_matches_cpu_logits() {
    let _guard = CUDA_TEST_LOCK
        .lock()
        .expect("CUDA test lock should not be poisoned");
    let fixture =
        common::build_synthetic_gemma4_fixture().expect("Gemma4 fixture should be created");
    let cpu_runtime = Runtime::load_with_backend(fixture.path(), BackendKind::Cpu)
        .expect("CPU Gemma4 runtime should load");
    let cuda_runtime = Runtime::load_with_backend(fixture.path(), BackendKind::Auto)
        .expect("CUDA Gemma4 runtime should load");
    assert_eq!(cuda_runtime.active_backend(), BackendKind::CudaResident);
    assert!(
        cuda_runtime
            .gpu_resource_status()
            .resident_dense_quant_decode_available
    );

    let tokens = [0u32, 3, 4, 5, 6];
    let mut cpu_session = cpu_runtime.backend().new_session(
        KvCacheMode::F32,
        cpu_runtime.backend().config().context_length,
    );
    let mut cuda_session = cuda_runtime.backend().new_session(
        KvCacheMode::F32,
        cuda_runtime.backend().config().context_length,
    );
    for (position, token) in tokens.into_iter().enumerate() {
        let mut cpu_logits = Vec::new();
        let mut cuda_logits = Vec::new();
        cpu_runtime
            .backend()
            .forward_token(token, position, &mut cpu_session, &mut cpu_logits)
            .expect("CPU Gemma4 token should decode");
        cuda_runtime
            .backend()
            .forward_token(token, position, &mut cuda_session, &mut cuda_logits)
            .expect("CUDA Gemma4 token should decode");
        assert_eq!(cuda_logits.len(), 32);
        assert_close(&cuda_logits, &cpu_logits, 5e-2);
    }

    let mut cpu_session = cpu_runtime.backend().new_session(
        KvCacheMode::F32,
        cpu_runtime.backend().config().context_length,
    );
    let mut cuda_session = cuda_runtime.backend().new_session(
        KvCacheMode::F32,
        cuda_runtime.backend().config().context_length,
    );
    let cpu_logits = cpu_runtime
        .backend()
        .forward_batch(&tokens, 0, &mut cpu_session)
        .expect("CPU Gemma4 batch should decode");
    let cuda_logits = cuda_runtime
        .backend()
        .forward_batch(&tokens, 0, &mut cuda_session)
        .expect("CUDA Gemma4 batch should decode");
    assert_close(&cuda_logits, &cpu_logits, 5e-2);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA-capable device and driver"]
fn cuda_gemma4_quantized_kv_runtime_matches_cpu_logits() {
    let _guard = CUDA_TEST_LOCK
        .lock()
        .expect("CUDA test lock should not be poisoned");
    let fixture =
        common::build_synthetic_gemma4_fixture().expect("Gemma4 fixture should be created");
    let cpu_runtime = Runtime::load_with_backend(fixture.path(), BackendKind::Cpu)
        .expect("CPU Gemma4 runtime should load");
    let cuda_runtime = Runtime::load_with_backend(fixture.path(), BackendKind::CudaResident)
        .expect("CUDA Gemma4 runtime should load");

    let tokens = [0u32, 3, 4, 5, 6];
    for (cache_mode, tolerance) in [
        (KvCacheMode::Q8, 8e-2),
        (KvCacheMode::KeyQ4ValueQ8, 4e-1),
        (KvCacheMode::AgentAdaptive, 4e-1),
    ] {
        let mut cpu_session = cpu_runtime.backend().new_session(cache_mode, 2);
        let mut cuda_session = cuda_runtime.backend().new_session(cache_mode, 2);
        if cache_mode == KvCacheMode::AgentAdaptive {
            let policy = SessionPolicy {
                recent_window_tokens: 1,
                ..SessionPolicy::agent_adaptive()
            };
            cpu_session.configure_policy(policy.clone(), 0, &[]);
            cuda_session.configure_policy(policy, 0, &[]);
        }
        for (position, token) in tokens.into_iter().enumerate() {
            let mut cpu_logits = Vec::new();
            let mut cuda_logits = Vec::new();
            cpu_runtime
                .backend()
                .forward_token(token, position, &mut cpu_session, &mut cpu_logits)
                .expect("CPU Gemma4 quantized-KV token should decode");
            cuda_runtime
                .backend()
                .forward_token(token, position, &mut cuda_session, &mut cuda_logits)
                .expect("CUDA Gemma4 quantized-KV token should decode");
            assert_eq!(cuda_logits.len(), 32);
            assert_close(&cuda_logits, &cpu_logits, tolerance);
        }
        assert!(cuda_session.cuda_kv_allocated_bytes() > 0);
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA-capable device and driver"]
fn cuda_f16_runtime_matches_cpu_logits() {
    let _guard = CUDA_TEST_LOCK
        .lock()
        .expect("CUDA test lock should not be poisoned");
    let spec = common::SyntheticLlamaSpec::tiny();
    let fixture = common::build_synthetic_f16_llama_fixture(spec.clone())
        .expect("F16 fixture should be created");

    assert_cuda_fixture_matches_cpu_logits(fixture.path(), &spec, 3e-2);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA-capable device and driver"]
fn cuda_bf16_runtime_matches_cpu_logits() {
    let _guard = CUDA_TEST_LOCK
        .lock()
        .expect("CUDA test lock should not be poisoned");
    let spec = common::SyntheticLlamaSpec::tiny();
    let fixture = common::build_synthetic_bf16_llama_fixture(spec.clone())
        .expect("BF16 fixture should be created");

    assert_cuda_fixture_matches_cpu_logits(fixture.path(), &spec, 5e-2);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA-capable device and driver"]
fn cuda_q4_0_runtime_matches_cpu_logits() {
    let _guard = CUDA_TEST_LOCK
        .lock()
        .expect("CUDA test lock should not be poisoned");
    let spec = common::SyntheticLlamaSpec::tiny();
    let fixture = common::build_synthetic_q4_0_llama_fixture(spec.clone())
        .expect("Q4_0 fixture should be created");
    let cpu_runtime = Runtime::load_with_backend(fixture.path(), BackendKind::Cpu)
        .expect("CPU runtime should load");
    let cuda_runtime = Runtime::load_with_backend(fixture.path(), BackendKind::Auto)
        .expect("auto runtime should load");
    assert_eq!(cuda_runtime.active_backend(), BackendKind::CudaResident);
    let status = cuda_runtime.gpu_resource_status();
    assert!(status.resident_dense_quant_decode_available);
    assert!(!status.resident_q8_0_probe_available);

    let mut cpu_session = cpu_runtime.backend().new_session(
        KvCacheMode::F32,
        cpu_runtime.backend().config().context_length,
    );
    let mut cuda_session = cuda_runtime.backend().new_session(
        KvCacheMode::F32,
        cuda_runtime.backend().config().context_length,
    );

    for (position, token) in [spec.bos_token_id, 3].into_iter().enumerate() {
        let mut cpu_logits = Vec::new();
        let mut cuda_logits = Vec::new();
        cpu_runtime
            .backend()
            .forward_token(token, position, &mut cpu_session, &mut cpu_logits)
            .expect("CPU token should decode");
        cuda_runtime
            .backend()
            .forward_token(token, position, &mut cuda_session, &mut cuda_logits)
            .expect("CUDA token should decode");

        assert_eq!(cuda_logits.len(), spec.vocab_size);
        assert_close(&cuda_logits, &cpu_logits, 1e-2);
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA-capable device and driver"]
fn cuda_q4_k_runtime_matches_cpu_logits() {
    let _guard = CUDA_TEST_LOCK
        .lock()
        .expect("CUDA test lock should not be poisoned");
    let mut spec = common::SyntheticLlamaSpec::tiny();
    spec.embedding_length = 256;
    spec.feed_forward_length = 256;
    spec.rope_dimension_count = 64;
    let fixture = common::build_synthetic_q4_k_llama_fixture(spec.clone())
        .expect("Q4_K fixture should be created");
    let cpu_runtime = Runtime::load_with_backend(fixture.path(), BackendKind::Cpu)
        .expect("CPU runtime should load");
    let cuda_runtime = Runtime::load_with_backend(fixture.path(), BackendKind::Auto)
        .expect("auto runtime should load");
    assert_eq!(cuda_runtime.active_backend(), BackendKind::CudaResident);
    let status = cuda_runtime.gpu_resource_status();
    assert!(status.resident_dense_quant_decode_available);
    assert!(!status.resident_q8_0_probe_available);
    assert!(!status.resident_q8_0_layer0_probe_available);

    let mut cpu_session = cpu_runtime.backend().new_session(
        KvCacheMode::F32,
        cpu_runtime.backend().config().context_length,
    );
    let mut cuda_session = cuda_runtime.backend().new_session(
        KvCacheMode::F32,
        cuda_runtime.backend().config().context_length,
    );

    for (position, token) in [spec.bos_token_id, 3].into_iter().enumerate() {
        let mut cpu_logits = Vec::new();
        let mut cuda_logits = Vec::new();
        cpu_runtime
            .backend()
            .forward_token(token, position, &mut cpu_session, &mut cpu_logits)
            .expect("CPU token should decode");
        cuda_runtime
            .backend()
            .forward_token(token, position, &mut cuda_session, &mut cuda_logits)
            .expect("CUDA token should decode");

        assert_eq!(cuda_logits.len(), spec.vocab_size);
        assert_k_quant_logits_close(&cuda_logits, &cpu_logits);
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA-capable device and driver"]
fn cuda_q5_k_runtime_matches_cpu_logits() {
    let _guard = CUDA_TEST_LOCK
        .lock()
        .expect("CUDA test lock should not be poisoned");
    let mut spec = common::SyntheticLlamaSpec::tiny();
    spec.embedding_length = 256;
    spec.feed_forward_length = 256;
    spec.rope_dimension_count = 64;
    let fixture = common::build_synthetic_q5_k_llama_fixture(spec.clone())
        .expect("Q5_K fixture should be created");
    let cpu_runtime = Runtime::load_with_backend(fixture.path(), BackendKind::Cpu)
        .expect("CPU runtime should load");
    let cuda_runtime = Runtime::load_with_backend(fixture.path(), BackendKind::Auto)
        .expect("auto runtime should load");
    assert_eq!(cuda_runtime.active_backend(), BackendKind::CudaResident);
    assert!(
        cuda_runtime
            .gpu_resource_status()
            .resident_dense_quant_decode_available
    );

    let mut cpu_session = cpu_runtime.backend().new_session(
        KvCacheMode::F32,
        cpu_runtime.backend().config().context_length,
    );
    let mut cuda_session = cuda_runtime.backend().new_session(
        KvCacheMode::F32,
        cuda_runtime.backend().config().context_length,
    );

    for (position, token) in [spec.bos_token_id, 3].into_iter().enumerate() {
        let mut cpu_logits = Vec::new();
        let mut cuda_logits = Vec::new();
        cpu_runtime
            .backend()
            .forward_token(token, position, &mut cpu_session, &mut cpu_logits)
            .expect("CPU token should decode");
        cuda_runtime
            .backend()
            .forward_token(token, position, &mut cuda_session, &mut cuda_logits)
            .expect("CUDA token should decode");

        assert_eq!(cuda_logits.len(), spec.vocab_size);
        assert_k_quant_logits_close(&cuda_logits, &cpu_logits);
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA-capable device and driver"]
fn cuda_q6_k_runtime_matches_cpu_logits() {
    let _guard = CUDA_TEST_LOCK
        .lock()
        .expect("CUDA test lock should not be poisoned");
    let mut spec = common::SyntheticLlamaSpec::tiny();
    spec.embedding_length = 256;
    spec.feed_forward_length = 256;
    spec.rope_dimension_count = 64;
    let fixture = common::build_synthetic_q6_k_llama_fixture(spec.clone())
        .expect("Q6_K fixture should be created");
    let cpu_runtime = Runtime::load_with_backend(fixture.path(), BackendKind::Cpu)
        .expect("CPU runtime should load");
    let cuda_runtime = Runtime::load_with_backend(fixture.path(), BackendKind::Auto)
        .expect("auto runtime should load");
    assert_eq!(cuda_runtime.active_backend(), BackendKind::CudaResident);
    assert!(
        cuda_runtime
            .gpu_resource_status()
            .resident_dense_quant_decode_available
    );

    let mut cpu_session = cpu_runtime.backend().new_session(
        KvCacheMode::F32,
        cpu_runtime.backend().config().context_length,
    );
    let mut cuda_session = cuda_runtime.backend().new_session(
        KvCacheMode::F32,
        cuda_runtime.backend().config().context_length,
    );

    for (position, token) in [spec.bos_token_id, 3].into_iter().enumerate() {
        let mut cpu_logits = Vec::new();
        let mut cuda_logits = Vec::new();
        cpu_runtime
            .backend()
            .forward_token(token, position, &mut cpu_session, &mut cpu_logits)
            .expect("CPU token should decode");
        cuda_runtime
            .backend()
            .forward_token(token, position, &mut cuda_session, &mut cuda_logits)
            .expect("CUDA token should decode");

        assert_eq!(cuda_logits.len(), spec.vocab_size);
        assert_k_quant_logits_close(&cuda_logits, &cpu_logits);
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires XRT_REAL_GGUF plus a CUDA-capable device and driver"]
fn cuda_real_model_first_token_logits_choose_same_top_token_as_cpu() {
    let _guard = CUDA_TEST_LOCK
        .lock()
        .expect("CUDA test lock should not be poisoned");
    let Some(model_path) = std::env::var_os("XRT_REAL_GGUF").map(std::path::PathBuf::from) else {
        eprintln!("set XRT_REAL_GGUF to run real-model CUDA parity");
        return;
    };
    let cpu_runtime =
        Runtime::load_with_backend(&model_path, BackendKind::Cpu).expect("CPU runtime should load");
    let cuda_runtime = Runtime::load_with_backend(&model_path, BackendKind::CudaResident)
        .expect("CUDA runtime should load");
    assert!(
        cuda_runtime
            .gpu_resource_status()
            .resident_dense_quant_decode_available
    );

    let prompt_tokens = cpu_runtime
        .tokenizer()
        .encode_with_options("Hello", true, true)
        .expect("prompt should tokenize");
    let token = *prompt_tokens.first().expect("prompt should have a token");
    let config = cpu_runtime.backend().config();
    eprintln!(
        "real CUDA parity model: architecture={}, layers={}, embedding={}, ffn={}, heads={}, kv_heads={}, head_dim={}, vocab={}, token={token}",
        config.architecture,
        config.block_count,
        config.embedding_length,
        config.feed_forward_length,
        config.attention_head_count,
        config.attention_head_count_kv,
        config.head_dim(),
        config.vocab_size,
    );

    for (label, n_layers) in [("zero-layer", 0), ("one-layer", 1)] {
        let mut cpu_draft_session = cpu_runtime.backend().new_session(KvCacheMode::F32, 1);
        let mut cuda_draft_session = cuda_runtime.backend().new_session(KvCacheMode::F32, 1);
        let mut cpu_draft_logits = Vec::new();
        let mut cuda_draft_logits = Vec::new();
        cpu_runtime
            .backend()
            .forward_draft(
                token,
                0,
                n_layers,
                &mut cpu_draft_session,
                &mut cpu_draft_logits,
            )
            .expect("CPU draft should decode");
        cuda_runtime
            .backend()
            .forward_draft(
                token,
                0,
                n_layers,
                &mut cuda_draft_session,
                &mut cuda_draft_logits,
            )
            .expect("CUDA draft should decode");
        let (cuda_top, cpu_top) =
            report_real_model_logit_parity(label, &cuda_draft_logits, &cpu_draft_logits);
        assert_real_model_top_logit_close(
            label,
            &cuda_draft_logits,
            &cpu_draft_logits,
            cuda_top,
            cpu_top,
        );
    }

    let mut cpu_session = cpu_runtime.backend().new_session(KvCacheMode::F32, 1);
    let mut cuda_session = cuda_runtime.backend().new_session(KvCacheMode::F32, 1);
    let mut cpu_logits = Vec::new();
    let mut cuda_logits = Vec::new();
    cpu_runtime
        .backend()
        .forward_token(token, 0, &mut cpu_session, &mut cpu_logits)
        .expect("CPU token should decode");
    cuda_runtime
        .backend()
        .forward_token(token, 0, &mut cuda_session, &mut cuda_logits)
        .expect("CUDA token should decode");

    assert_eq!(cuda_logits.len(), cpu_logits.len());
    let (cuda_top, cpu_top) =
        report_real_model_logit_parity("full-model", &cuda_logits, &cpu_logits);
    assert_real_model_top_logit_close("full-model", &cuda_logits, &cpu_logits, cuda_top, cpu_top);

    let cache_modes = if config.is_gemma4() {
        vec![
            KvCacheMode::F32,
            KvCacheMode::Q8,
            KvCacheMode::KeyQ4ValueQ8,
            KvCacheMode::AgentAdaptive,
        ]
    } else {
        vec![KvCacheMode::Q8, KvCacheMode::KeyQ4ValueQ8]
    };
    let mut mismatches = Vec::new();
    let mut sequential_tokens = Vec::new();
    for cache_mode in cache_modes {
        let mut cpu_session = cpu_runtime.backend().new_session(cache_mode, 1);
        let mut cuda_session = cuda_runtime.backend().new_session(cache_mode, 1);
        if cache_mode == KvCacheMode::AgentAdaptive {
            let policy = SessionPolicy {
                recent_window_tokens: 1,
                ..SessionPolicy::agent_adaptive()
            };
            cpu_session.configure_policy(policy.clone(), 0, &[]);
            cuda_session.configure_policy(policy, 0, &[]);
        }
        let mut input_token = token;
        for position in 0..4 {
            if cache_mode == KvCacheMode::F32 {
                sequential_tokens.push(input_token);
            }
            let mut cpu_logits = Vec::new();
            let mut cuda_logits = Vec::new();
            cpu_runtime
                .backend()
                .forward_token(input_token, position, &mut cpu_session, &mut cpu_logits)
                .expect("CPU real-model parity token should decode");
            cuda_runtime
                .backend()
                .forward_token(input_token, position, &mut cuda_session, &mut cuda_logits)
                .expect("CUDA real-model parity token should decode");
            let label = format!("{cache_mode:?}-position-{position}");
            let (cuda_top, cpu_top) =
                report_real_model_logit_parity(&label, &cuda_logits, &cpu_logits);
            let top_score_delta = (cuda_logits[cpu_top] - cpu_logits[cpu_top]).abs();
            // Four-bit key-cache error compounds the known CPU Q8 activation
            // quantization drift. Mixed hot/cold attention adds a second bounded
            // numerical path, but exact greedy top-token agreement stays mandatory.
            let max_top_score_delta = match cache_mode {
                KvCacheMode::AgentAdaptive => 4.0,
                KvCacheMode::KeyQ4ValueQ8 => 2.0,
                KvCacheMode::F32 | KvCacheMode::Q8 => 1.0,
            };
            if cuda_top != cpu_top || top_score_delta > max_top_score_delta {
                mismatches.push(format!(
                    "{cache_mode:?} position {position}: CUDA {cuda_top}, CPU {cpu_top}, top score delta {top_score_delta}, limit {max_top_score_delta}"
                ));
            }
            input_token = cpu_top as u32;
        }
    }
    assert!(
        mismatches.is_empty(),
        "real-model sequential top-token mismatches: {}",
        mismatches.join("; ")
    );

    if config.is_gemma4()
        && std::env::var("XRT_REAL_GGUF_LAYER_DIAGNOSTICS").is_ok_and(|value| value == "1")
    {
        run_gemma4_layer_diagnostics(
            &cpu_runtime,
            &cuda_runtime,
            &sequential_tokens,
            config.block_count,
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires XRT_REAL_HF_MODEL_DIR, XRT_REAL_GGUF, and a CUDA-capable device"]
fn cuda_real_safetensors_qwen2_matches_equivalent_gguf_top_tokens() {
    run_real_hf_qwen2_cuda_parity(
        "XRT_REAL_HF_MODEL_DIR",
        "XRT_REAL_GGUF",
        "SafeTensors",
        "safetensors",
        "Hello",
        1.0,
        1,
        None,
        None,
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires XRT_REAL_AWQ_MODEL_DIR, XRT_REAL_AWQ_GGUF, and a CUDA-capable device"]
fn cuda_real_autoawq_qwen2_matches_equivalent_gguf_top_tokens() {
    run_real_hf_qwen2_cuda_parity(
        "XRT_REAL_AWQ_MODEL_DIR",
        "XRT_REAL_AWQ_GGUF",
        "AutoAWQ",
        "autoawq",
        "The capital of France is",
        // AWQ4 and GGUF Q8 are independently quantized checkpoints. Exact greedy
        // semantics remain mandatory while this bound catches gross score drift.
        5.0,
        2,
        None,
        Some(" Paris"),
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires XRT_REAL_GPTQ_MODEL_DIR, XRT_REAL_GPTQ_GGUF, and a CUDA-capable device"]
fn cuda_real_gptq_v1_qwen2_matches_equivalent_gguf_semantics() {
    run_real_hf_qwen2_cuda_parity(
        "XRT_REAL_GPTQ_MODEL_DIR",
        "XRT_REAL_GPTQ_GGUF",
        "GPTQ v1",
        "gptq-v1",
        "The capital of France is",
        // GPTQ4 and GGUF Q8 are independently quantized checkpoints. Exact
        // greedy semantics remain mandatory while this bound catches gross drift.
        5.0,
        2,
        // A full-model draft from only the prompt's first BPE token is highly
        // quantization-sensitive. Keep exact draft parity through layer one,
        // then use full-prompt greedy generation as the semantic gate.
        Some(1),
        Some(" Paris"),
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires XRT_REAL_COMPRESSED_TENSORS_MODEL_DIR, XRT_REAL_DENSE_HF_MODEL_DIR, and a CUDA-capable device"]
fn cuda_real_compressed_tensors_qwen2_matches_dense_bf16_semantics() {
    let _guard = CUDA_TEST_LOCK
        .lock()
        .expect("CUDA test lock should not be poisoned");
    let compressed_path = std::env::var_os("XRT_REAL_COMPRESSED_TENSORS_MODEL_DIR")
        .map(std::path::PathBuf::from)
        .expect("XRT_REAL_COMPRESSED_TENSORS_MODEL_DIR is required");
    let dense_path = std::env::var_os("XRT_REAL_DENSE_HF_MODEL_DIR")
        .map(std::path::PathBuf::from)
        .expect("XRT_REAL_DENSE_HF_MODEL_DIR is required");
    let prompt = "The capital of France is";
    let total_start = Instant::now();

    let stage_start = Instant::now();
    eprintln!("compressed-tensors parity: loading official dense BF16 CUDA runtime");
    let dense_runtime = Runtime::load_with_backend(&dense_path, BackendKind::CudaResident)
        .expect("official dense BF16 CUDA runtime should load");
    eprintln!(
        "compressed-tensors parity: dense runtime loaded in {:.3}s, resident_bytes={}",
        stage_start.elapsed().as_secs_f64(),
        dense_runtime.gpu_resource_status().model_weight_bytes
    );
    let stage_start = Instant::now();
    eprintln!("compressed-tensors parity: loading W4A16 CUDA runtime");
    let compressed_runtime =
        Runtime::load_with_backend(&compressed_path, BackendKind::CudaResident)
            .expect("compressed-tensors W4A16 CUDA runtime should load");
    eprintln!(
        "compressed-tensors parity: W4A16 runtime loaded in {:.3}s, resident_bytes={}",
        stage_start.elapsed().as_secs_f64(),
        compressed_runtime.gpu_resource_status().model_weight_bytes
    );

    for runtime in [&dense_runtime, &compressed_runtime] {
        assert_eq!(runtime.active_backend(), BackendKind::CudaResident);
        assert_eq!(runtime.model_architecture(), "qwen2");
        assert!(runtime.cpu_model().is_none());
        assert!(
            runtime
                .gpu_resource_status()
                .resident_dense_quant_decode_available
        );
    }

    let dense_tokens = dense_runtime
        .tokenizer()
        .encode_with_options(prompt, true, true)
        .expect("dense prompt should tokenize");
    let compressed_tokens = compressed_runtime
        .tokenizer()
        .encode_with_options(prompt, true, true)
        .expect("compressed prompt should tokenize");
    assert_eq!(compressed_tokens, dense_tokens);
    let token = *dense_tokens.first().expect("prompt should contain a token");
    let block_count = dense_runtime.backend().config().block_count;
    assert_eq!(
        compressed_runtime.backend().config().block_count,
        block_count
    );

    for (label, layer_count) in [
        ("compressed-tensors-zero-layer", 0),
        ("compressed-tensors-one-layer", 1),
        ("compressed-tensors-full-model", block_count),
    ] {
        let stage_start = Instant::now();
        eprintln!("compressed-tensors parity: running {label}");
        let mut dense_session = dense_runtime.backend().new_session(KvCacheMode::F32, 1);
        let mut compressed_session = compressed_runtime
            .backend()
            .new_session(KvCacheMode::F32, 1);
        let mut dense_logits = Vec::new();
        let mut compressed_logits = Vec::new();
        dense_runtime
            .backend()
            .forward_draft(token, 0, layer_count, &mut dense_session, &mut dense_logits)
            .expect("dense BF16 CUDA draft should decode");
        compressed_runtime
            .backend()
            .forward_draft(
                token,
                0,
                layer_count,
                &mut compressed_session,
                &mut compressed_logits,
            )
            .expect("compressed-tensors CUDA draft should decode");

        let (compressed_top, dense_top) =
            report_real_model_logit_parity(label, &compressed_logits, &dense_logits);
        if layer_count <= 1 {
            assert_real_model_top_logit_close_with_limit(
                label,
                &compressed_logits,
                &dense_logits,
                compressed_top,
                dense_top,
                5.0,
            );
            assert_real_model_top_k_overlap(label, &compressed_logits, &dense_logits, 5, 2);
        } else {
            eprintln!(
                "compressed-tensors parity: {label} is diagnostic-only for the first BPE token"
            );
        }
        assert!(compressed_logits.iter().all(|value| value.is_finite()));
        eprintln!(
            "compressed-tensors parity: {label} passed in {:.3}s",
            stage_start.elapsed().as_secs_f64()
        );
    }

    let request = GenerateRequest {
        prompt: prompt.to_string(),
        max_tokens: 1,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(17),
        ..Default::default()
    };
    let dense_text = dense_runtime
        .new_session()
        .generate(&request)
        .expect("dense BF16 CUDA generation should succeed");
    let compressed_text = compressed_runtime
        .new_session()
        .generate(&request)
        .expect("compressed-tensors CUDA generation should succeed");
    assert_eq!(
        compressed_text, dense_text,
        "compressed-tensors one-token generated text parity"
    );
    assert_eq!(
        compressed_text, " Paris",
        "known compressed-tensors one-token semantic output"
    );
    eprintln!(
        "compressed-tensors parity: complete in {:.3}s, generated={compressed_text:?}",
        total_start.elapsed().as_secs_f64()
    );
}

#[cfg(feature = "cuda")]
fn run_real_hf_qwen2_cuda_parity(
    hf_environment: &str,
    gguf_environment: &str,
    format_label: &str,
    test_label: &str,
    prompt: &str,
    max_top_score_delta: f32,
    minimum_top5_overlap: usize,
    strict_draft_layer_limit: Option<usize>,
    expected_generated_text: Option<&str>,
) {
    let _guard = CUDA_TEST_LOCK
        .lock()
        .expect("CUDA test lock should not be poisoned");
    let hf_path = std::env::var_os(hf_environment)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| panic!("{hf_environment} is required"));
    let gguf_path = std::env::var_os(gguf_environment)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| panic!("{gguf_environment} is required"));

    let total_start = Instant::now();
    let stage_start = Instant::now();
    eprintln!("{format_label} parity: loading equivalent GGUF CPU runtime");
    let cpu_runtime = Runtime::load_with_backend(&gguf_path, BackendKind::Cpu)
        .expect("equivalent GGUF CPU runtime should load");
    eprintln!(
        "{format_label} parity: GGUF CPU runtime loaded in {:.3}s",
        stage_start.elapsed().as_secs_f64()
    );
    let stage_start = Instant::now();
    eprintln!("{format_label} parity: loading Hugging Face CUDA runtime");
    let cuda_runtime = Runtime::load_with_backend(&hf_path, BackendKind::CudaResident)
        .expect("Hugging Face CUDA runtime should load");
    eprintln!(
        "{format_label} parity: CUDA runtime loaded in {:.3}s, resident_bytes={}",
        stage_start.elapsed().as_secs_f64(),
        cuda_runtime.gpu_resource_status().model_weight_bytes
    );
    assert_eq!(cuda_runtime.active_backend(), BackendKind::CudaResident);
    assert_eq!(cuda_runtime.model_architecture(), "qwen2");
    assert!(cuda_runtime.cpu_model().is_none());
    assert!(
        cuda_runtime
            .gpu_resource_status()
            .resident_dense_quant_decode_available
    );

    let cpu_tokens = cpu_runtime
        .tokenizer()
        .encode_with_options(prompt, true, true)
        .expect("GGUF prompt should tokenize");
    let hf_tokens = cuda_runtime
        .tokenizer()
        .encode_with_options(prompt, true, true)
        .expect("HF prompt should tokenize");
    assert_eq!(hf_tokens, cpu_tokens);
    let token = *cpu_tokens.first().expect("prompt should contain a token");
    let block_count = cpu_runtime.backend().config().block_count;
    assert_eq!(cuda_runtime.backend().config().block_count, block_count);

    for (label, layer_count) in [
        (format!("{test_label}-zero-layer"), 0),
        (format!("{test_label}-one-layer"), 1),
        (format!("{test_label}-full-model"), block_count),
    ] {
        let stage_start = Instant::now();
        eprintln!("{format_label} parity: running {label}");
        let mut cpu_session = cpu_runtime.backend().new_session(KvCacheMode::F32, 1);
        let mut cuda_session = cuda_runtime.backend().new_session(KvCacheMode::F32, 1);
        let mut cpu_logits = Vec::new();
        let mut cuda_logits = Vec::new();
        cpu_runtime
            .backend()
            .forward_draft(token, 0, layer_count, &mut cpu_session, &mut cpu_logits)
            .expect("GGUF CPU draft should decode");
        cuda_runtime
            .backend()
            .forward_draft(token, 0, layer_count, &mut cuda_session, &mut cuda_logits)
            .expect("Hugging Face CUDA draft should decode");

        let (cuda_top, cpu_top) = report_real_model_logit_parity(&label, &cuda_logits, &cpu_logits);
        let strict_draft = strict_draft_layer_limit.map_or(true, |limit| layer_count <= limit);
        if strict_draft {
            assert_real_model_top_logit_close_with_limit(
                &label,
                &cuda_logits,
                &cpu_logits,
                cuda_top,
                cpu_top,
                max_top_score_delta,
            );
            assert_real_model_top_k_overlap(
                &label,
                &cuda_logits,
                &cpu_logits,
                5,
                minimum_top5_overlap,
            );
        } else {
            eprintln!(
                "{format_label} parity: {label} is diagnostic-only across independently quantized checkpoints"
            );
        }
        assert!(cuda_logits.iter().all(|value| value.is_finite()));
        eprintln!(
            "{format_label} parity: {label} passed in {:.3}s",
            stage_start.elapsed().as_secs_f64()
        );
    }
    let request = GenerateRequest {
        prompt: prompt.to_string(),
        max_tokens: 1,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(17),
        ..Default::default()
    };
    let cpu_text = cpu_runtime
        .new_session()
        .generate(&request)
        .expect("GGUF CPU generation should succeed");
    let cuda_text = cuda_runtime
        .new_session()
        .generate(&request)
        .expect("Hugging Face CUDA generation should succeed");
    assert_eq!(cuda_text, cpu_text, "one-token generated text parity");
    if let Some(expected) = expected_generated_text {
        assert_eq!(cuda_text, expected, "known one-token semantic output");
    }
    eprintln!(
        "{format_label} parity: complete in {:.3}s, generated={cuda_text:?}",
        total_start.elapsed().as_secs_f64(),
    );
}

#[cfg(feature = "cuda")]
fn run_gemma4_layer_diagnostics(
    cpu_runtime: &std::sync::Arc<Runtime>,
    cuda_runtime: &std::sync::Arc<Runtime>,
    tokens: &[u32],
    block_count: usize,
) {
    assert_eq!(tokens.len(), 4, "Gemma4 diagnostic token count");
    run_gemma4_layer0_stage_diagnostics(cpu_runtime, cuda_runtime, tokens);

    let mut layer_counts = vec![0, 1, 2, 4, 8, 12, 16, 24, 32, 40, block_count];
    layer_counts.retain(|count| *count <= block_count);
    layer_counts.sort_unstable();
    layer_counts.dedup();

    for layer_count in layer_counts {
        let mut cpu_session = cpu_runtime
            .backend()
            .new_session(KvCacheMode::F32, tokens.len());
        let mut cuda_session = cuda_runtime
            .backend()
            .new_session(KvCacheMode::F32, tokens.len());
        let mut cpu_logits = Vec::new();
        let mut cuda_logits = Vec::new();
        for (position, token) in tokens.iter().copied().enumerate() {
            cpu_runtime
                .backend()
                .forward_draft(
                    token,
                    position,
                    layer_count,
                    &mut cpu_session,
                    &mut cpu_logits,
                )
                .expect("CPU Gemma4 layer diagnostic should decode");
            cuda_runtime
                .backend()
                .forward_draft(
                    token,
                    position,
                    layer_count,
                    &mut cuda_session,
                    &mut cuda_logits,
                )
                .expect("CUDA Gemma4 layer diagnostic should decode");
        }
        let label = format!("gemma4-{layer_count}-layers-position-3");
        report_real_model_logit_parity(&label, &cuda_logits, &cpu_logits);
    }
}

#[cfg(feature = "cuda")]
fn run_gemma4_layer0_stage_diagnostics(
    cpu_runtime: &std::sync::Arc<Runtime>,
    cuda_runtime: &std::sync::Arc<Runtime>,
    tokens: &[u32],
) {
    let mut cpu_session = cpu_runtime
        .backend()
        .new_session(KvCacheMode::F32, tokens.len());
    let mut cuda_session = cuda_runtime
        .backend()
        .new_session(KvCacheMode::F32, tokens.len());
    let mut cpu_logits = Vec::new();
    let mut cuda_logits = Vec::new();

    for (position, token) in tokens.iter().copied().take(3).enumerate() {
        cpu_runtime
            .backend()
            .forward_draft(token, position, 1, &mut cpu_session, &mut cpu_logits)
            .expect("CPU Gemma4 layer-0 trace prefix should decode");
        cuda_runtime
            .backend()
            .forward_draft(token, position, 1, &mut cuda_session, &mut cuda_logits)
            .expect("CUDA Gemma4 layer-0 trace prefix should decode");
    }

    let position = 3;
    let token = tokens[position];
    let cpu_trace = cpu_runtime
        .backend()
        .gemma4_layer0_trace(token, position, &mut cpu_session)
        .expect("CPU Gemma4 layer-0 trace should run")
        .expect("CPU Gemma4 layer-0 trace should be available");
    let cuda_trace = cuda_runtime
        .backend()
        .gemma4_layer0_trace(token, position, &mut cuda_session)
        .expect("CUDA Gemma4 layer-0 trace should run")
        .expect("CUDA Gemma4 layer-0 trace should be available");

    assert_eq!(cuda_trace.layer_index, cpu_trace.layer_index);
    assert_eq!(cuda_trace.position, cpu_trace.position);
    assert_eq!(cuda_trace.stages.len(), cpu_trace.stages.len());
    for (cuda_stage, cpu_stage) in cuda_trace.stages.iter().zip(&cpu_trace.stages) {
        assert_eq!(cuda_stage.name, cpu_stage.name);
        report_gemma4_trace_stage_parity(cuda_stage.name, &cuda_stage.values, &cpu_stage.values);
        if cuda_stage.name.ends_with("_projection_float_reference") {
            assert_close(&cuda_stage.values, &cpu_stage.values, 1e-3);
        }
    }
}

#[cfg(feature = "cuda")]
fn report_gemma4_trace_stage_parity(label: &str, cuda: &[f32], cpu: &[f32]) {
    assert_eq!(cuda.len(), cpu.len(), "Gemma4 trace stage {label}");
    if cuda.is_empty() {
        eprintln!("real CUDA Gemma4 layer-0 trace {label}: len=0");
        return;
    }
    let mut max_delta = 0.0f32;
    let mut max_index = 0usize;
    let mut sum_squared_delta = 0.0f64;
    for (index, (&cuda_value, &cpu_value)) in cuda.iter().zip(cpu).enumerate() {
        let delta = (cuda_value - cpu_value).abs();
        assert!(
            delta.is_finite(),
            "Gemma4 trace stage {label} has non-finite delta at {index}: CPU {cpu_value}, CUDA {cuda_value}"
        );
        if delta > max_delta {
            max_delta = delta;
            max_index = index;
        }
        sum_squared_delta += f64::from(delta) * f64::from(delta);
    }
    let rms_delta = (sum_squared_delta / cuda.len() as f64).sqrt();
    eprintln!(
        "real CUDA Gemma4 layer-0 trace {label}: len={}, max_delta={max_delta} at {max_index}, cpu_at_max={} cuda_at_max={}, rms_delta={rms_delta}",
        cuda.len(), cpu[max_index], cuda[max_index]
    );
}

#[cfg(feature = "cuda")]
fn assert_cuda_fixture_matches_cpu_logits(
    fixture_path: &std::path::Path,
    spec: &common::SyntheticLlamaSpec,
    tolerance: f32,
) {
    let cpu_runtime = Runtime::load_with_backend(fixture_path, BackendKind::Cpu)
        .expect("CPU runtime should load");
    let cuda_runtime = Runtime::load_with_backend(fixture_path, BackendKind::Auto)
        .expect("auto runtime should load");
    assert_eq!(cuda_runtime.active_backend(), BackendKind::CudaResident);
    assert!(
        cuda_runtime
            .gpu_resource_status()
            .resident_dense_quant_decode_available
    );

    let mut cpu_session = cpu_runtime.backend().new_session(
        KvCacheMode::F32,
        cpu_runtime.backend().config().context_length,
    );
    let mut cuda_session = cuda_runtime.backend().new_session(
        KvCacheMode::F32,
        cuda_runtime.backend().config().context_length,
    );

    for (position, token) in [spec.bos_token_id, 3].into_iter().enumerate() {
        let mut cpu_logits = Vec::new();
        let mut cuda_logits = Vec::new();
        cpu_runtime
            .backend()
            .forward_token(token, position, &mut cpu_session, &mut cpu_logits)
            .expect("CPU token should decode");
        cuda_runtime
            .backend()
            .forward_token(token, position, &mut cuda_session, &mut cuda_logits)
            .expect("CUDA token should decode");

        assert_eq!(cuda_logits.len(), spec.vocab_size);
        assert_close(&cuda_logits, &cpu_logits, tolerance);
    }
}

#[cfg(feature = "cuda")]
fn assert_k_quant_logits_close(actual: &[f32], expected: &[f32]) {
    // CPU K-quant SIMD quantizes activations to Q8_0; CUDA consumes resident F32 activations.
    assert_close(actual, expected, 5e-2);
}

#[cfg(feature = "cuda")]
fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    let mut max_delta = 0.0f32;
    let mut max_index = 0usize;
    for (idx, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let delta = (actual - expected).abs();
        assert!(
            delta.is_finite(),
            "value {idx} has a non-finite delta: actual={actual}, expected={expected}"
        );
        if delta > max_delta {
            max_delta = delta;
            max_index = idx;
        }
    }
    assert!(
        max_delta <= tolerance,
        "maximum delta at value {max_index}: actual={}, expected={}, delta={max_delta}, tolerance={tolerance}",
        actual[max_index],
        expected[max_index]
    );
}

#[cfg(feature = "cuda")]
fn report_real_model_logit_parity(label: &str, cuda: &[f32], cpu: &[f32]) -> (usize, usize) {
    assert_eq!(cuda.len(), cpu.len());
    let cuda_top = argmax(cuda);
    let cpu_top = argmax(cpu);
    let (max_index, max_delta) = cuda
        .iter()
        .zip(cpu)
        .enumerate()
        .map(|(index, (cuda, cpu))| (index, (cuda - cpu).abs()))
        .max_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
        .expect("logits must not be empty");
    eprintln!(
        "real CUDA parity {label}: max_delta={max_delta} at {max_index}, cpu_at_max={} cuda_at_max={}, cpu_top={cpu_top} cpu_score={} cuda_at_cpu_top={}, cuda_top={cuda_top} cuda_score={} cpu_at_cuda_top={}",
        cpu[max_index],
        cuda[max_index],
        cpu[cpu_top],
        cuda[cpu_top],
        cuda[cuda_top],
        cpu[cuda_top],
    );
    eprintln!(
        "real CUDA parity {label}: cpu_top5={:?} cuda_top5={:?}",
        top_k_scores(cpu, 5),
        top_k_scores(cuda, 5),
    );
    (cuda_top, cpu_top)
}

#[cfg(feature = "cuda")]
fn top_k_scores(values: &[f32], count: usize) -> Vec<(usize, f32)> {
    let mut indices = (0..values.len()).collect::<Vec<_>>();
    indices.sort_unstable_by(|left, right| values[*right].total_cmp(&values[*left]));
    indices.truncate(count.min(indices.len()));
    indices
        .into_iter()
        .map(|index| (index, values[index]))
        .collect()
}

#[cfg(feature = "cuda")]
fn assert_real_model_top_k_overlap(
    label: &str,
    cuda: &[f32],
    cpu: &[f32],
    count: usize,
    minimum_overlap: usize,
) {
    let cuda_top = top_k_scores(cuda, count);
    let cpu_top = top_k_scores(cpu, count);
    let overlap = cpu_top
        .iter()
        .filter(|(cpu_index, _)| {
            cuda_top
                .iter()
                .any(|(cuda_index, _)| cuda_index == cpu_index)
        })
        .count();
    assert!(
        overlap >= minimum_overlap,
        "real CUDA parity {label} top-{count} overlap {overlap} is below {minimum_overlap}"
    );
}

#[cfg(feature = "cuda")]
fn assert_real_model_top_logit_close(
    label: &str,
    cuda: &[f32],
    cpu: &[f32],
    cuda_top: usize,
    cpu_top: usize,
) {
    assert_real_model_top_logit_close_with_limit(label, cuda, cpu, cuda_top, cpu_top, 1.0);
}

#[cfg(feature = "cuda")]
fn assert_real_model_top_logit_close_with_limit(
    label: &str,
    cuda: &[f32],
    cpu: &[f32],
    cuda_top: usize,
    cpu_top: usize,
    max_top_score_delta: f32,
) {
    assert_eq!(cuda_top, cpu_top, "real CUDA parity {label} top token");
    let top_score_delta = (cuda[cpu_top] - cpu[cpu_top]).abs();
    assert!(
        top_score_delta <= max_top_score_delta,
        "real CUDA parity {label} top score delta {top_score_delta} exceeds {max_top_score_delta}"
    );
}

#[cfg(feature = "cuda")]
fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
        .map(|(idx, _)| idx)
        .expect("logits must not be empty")
}
