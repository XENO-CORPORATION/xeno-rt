mod common;

use std::{cmp::Ordering, sync::Arc, time::Instant};
use xrt_core::KvCache;
use xrt_gguf::GgufFile;
use xrt_models::LlamaModel;
use xrt_runtime::{
    BackendKind, BackendSession, GenerateRequest, KvCacheMode, PagedKvCache, Runtime, SessionPolicy,
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
        report_real_model_logit_parity(label, &cuda_draft_logits, &cpu_draft_logits);
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
    report_real_model_logit_parity("full-model", &cuda_logits, &cpu_logits);
    assert_eq!(argmax(&cuda_logits), argmax(&cpu_logits));

    let mut mismatches = Vec::new();
    for cache_mode in [KvCacheMode::Q8, KvCacheMode::KeyQ4ValueQ8] {
        let mut cpu_session = cpu_runtime.backend().new_session(cache_mode, 1);
        let mut cuda_session = cuda_runtime.backend().new_session(cache_mode, 1);
        let mut input_token = token;
        for position in 0..4 {
            let mut cpu_logits = Vec::new();
            let mut cuda_logits = Vec::new();
            cpu_runtime
                .backend()
                .forward_token(input_token, position, &mut cpu_session, &mut cpu_logits)
                .expect("CPU quantized-KV token should decode");
            cuda_runtime
                .backend()
                .forward_token(input_token, position, &mut cuda_session, &mut cuda_logits)
                .expect("CUDA quantized-KV token should decode");
            let label = format!("{cache_mode:?}-position-{position}");
            let (cuda_top, cpu_top) =
                report_real_model_logit_parity(&label, &cuda_logits, &cpu_logits);
            if cuda_top != cpu_top {
                mismatches.push(format!(
                    "{cache_mode:?} position {position}: CUDA {cuda_top}, CPU {cpu_top}"
                ));
            }
            input_token = cpu_top as u32;
        }
    }
    assert!(
        mismatches.is_empty(),
        "real-model quantized KV top-token mismatches: {}",
        mismatches.join("; ")
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
        "real CUDA parity {label}: max_delta={max_delta} at {max_index}, cpu_top={cpu_top} cpu_score={} cuda_at_cpu_top={}, cuda_top={cuda_top} cuda_score={} cpu_at_cuda_top={}",
        cpu[cpu_top],
        cuda[cpu_top],
        cuda[cuda_top],
        cpu[cuda_top],
    );
    (cuda_top, cpu_top)
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
