mod common;

use sha2::{Digest, Sha256};
#[cfg(feature = "cuda")]
use std::ops::ControlFlow;
use xrt_runtime::{
    BackendKind, BackendSession, CausalLmBackend, GenerateRequest, KvCacheMode, Runtime,
    SessionKvCache,
};
#[cfg(feature = "cuda")]
use xrt_runtime::{RequestScheduler, SchedulerConfig};

const SYNTHETIC_HYBRID_FIXTURE_SHA256: &str =
    "05ac8a03af75c09915a8120a032fab8708587bcaae6415956f44850ff20cb970";
const SYNTHETIC_HYBRID_LONG_FIXTURE_SHA256: &str =
    "55e026055d0984947d2d9f805eb1d78264d1ced9bcda421351b8ed41ccc89ee0";

fn prepared_session(backend: &dyn CausalLmBackend) -> BackendSession {
    let mut session = backend.new_session(KvCacheMode::F32, 4);
    backend
        .prepare_session_state(&mut session)
        .expect("hybrid session state should allocate before token zero");
    session
}

#[test]
fn synthetic_hybrid_fixture_sha256_is_pinned() {
    let (fixture, _) =
        common::build_synthetic_qwen35_hybrid_fixture().expect("fixture should build");
    let digest = format!("{:x}", Sha256::digest(&fixture.bytes));
    assert_eq!(digest, SYNTHETIC_HYBRID_FIXTURE_SHA256);
}

#[test]
fn synthetic_hybrid_long_fixture_sha256_is_pinned() {
    let (fixture, _) =
        common::build_synthetic_qwen35_hybrid_long_fixture().expect("fixture should build");
    let digest = format!("{:x}", Sha256::digest(&fixture.bytes));
    assert_eq!(digest, SYNTHETIC_HYBRID_LONG_FIXTURE_SHA256);
}

#[test]
fn experimental_mtp_opt_in_falls_back_cleanly_without_an_admitted_backend_head() {
    let (fixture, _) = common::build_synthetic_qwen35_mtp_fixture().expect("fixture should build");
    let runtime =
        Runtime::load_with_backend(fixture.path(), BackendKind::Cpu).expect("runtime should load");
    let request = GenerateRequest {
        prompt: "hello world".to_string(),
        max_tokens: 4,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.1,
        seed: Some(36),
        ..Default::default()
    };

    let mut reference = runtime.new_session_with_cache_mode(KvCacheMode::F32);
    reference.set_ngram_speculation_enabled(false);
    reference.set_mtp_speculation_enabled(false);
    let expected = reference.generate(&request).expect("reference should run");

    runtime.clear_prefix_cache();
    let mut mtp = runtime.new_session_with_cache_mode(KvCacheMode::F32);
    mtp.set_ngram_speculation_enabled(false);
    mtp.set_mtp_speculation_enabled(true);
    mtp.set_mtp_max_draft_tokens(0);
    assert_eq!(mtp.mtp_max_draft_tokens(), 1);
    mtp.set_mtp_max_draft_tokens(99);
    assert_eq!(mtp.mtp_max_draft_tokens(), 3);
    mtp.set_mtp_max_draft_tokens(1);
    let actual = mtp.generate(&request).expect("fallback should run");

    assert_eq!(actual, expected);
    assert_eq!(mtp.speculative_decode_stats().verification_batches, 0);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_mtp_drafting_preserves_target_greedy_output_and_transaction_boundaries() {
    let (fixture, _) = common::build_synthetic_qwen35_mtp_fixture().expect("fixture should build");
    let runtime = Runtime::load_with_backend(fixture.path(), BackendKind::CudaResident)
        .expect("CUDA should load");
    let request = GenerateRequest {
        prompt: "hello world".to_string(),
        max_tokens: 8,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.1,
        seed: Some(36),
        ..Default::default()
    };

    let mut reference = runtime.new_session_with_cache_mode(KvCacheMode::F32);
    reference.set_ngram_speculation_enabled(false);
    reference.set_mtp_speculation_enabled(false);
    let expected = reference.generate(&request).expect("reference should run");
    let reference_state = reference
        .recurrent_state_snapshot()
        .expect("reference state should snapshot")
        .expect("hybrid state should exist");

    runtime.clear_prefix_cache();
    let mut mtp = runtime.new_session_with_cache_mode(KvCacheMode::F32);
    mtp.set_ngram_speculation_enabled(false);
    mtp.set_mtp_speculation_enabled(true);
    let actual = mtp.generate(&request).expect("MTP decode should run");
    let stats = mtp.speculative_decode_stats();
    let mtp_state = mtp
        .recurrent_state_snapshot()
        .expect("MTP state should snapshot")
        .expect("hybrid state should exist");

    assert_eq!(actual, expected);
    assert!(stats.verification_batches > 0);
    assert!(stats.drafted_tokens > 0);
    assert_eq!(
        stats.accepted_tokens.saturating_add(stats.rejected_tokens),
        stats.drafted_tokens
    );
    assert_eq!(mtp_state, reference_state);
}

fn run_tokens(
    backend: &dyn CausalLmBackend,
    session: &mut BackendSession,
    tokens: &[u32],
    start_position: usize,
) -> Vec<Vec<f32>> {
    tokens
        .iter()
        .enumerate()
        .map(|(offset, &token)| {
            let mut logits = Vec::new();
            backend
                .forward_token(token, start_position + offset, session, &mut logits)
                .expect("hybrid token forward should succeed");
            logits
        })
        .collect()
}

#[test]
fn two_hybrid_sessions_are_isolated_when_interleaved_and_reset() {
    let (fixture, _) =
        common::build_synthetic_qwen35_hybrid_fixture().expect("fixture should build");
    let runtime =
        Runtime::load_with_backend(fixture.path(), BackendKind::Cpu).expect("runtime should load");
    let backend = runtime.backend();
    let tokens_a = [3, 4, 5, 6];
    let tokens_b = [7, 8, 9, 10];

    let mut isolated_a = prepared_session(backend);
    let expected_a = run_tokens(backend, &mut isolated_a, &tokens_a, 0);
    let mut isolated_b = prepared_session(backend);
    let expected_b = run_tokens(backend, &mut isolated_b, &tokens_b, 0);

    let mut interleaved_a = prepared_session(backend);
    let mut interleaved_b = prepared_session(backend);
    assert!(interleaved_a.recurrent_state_allocated_bytes() > 0);
    assert!(interleaved_b.recurrent_state_allocated_bytes() > 0);

    for position in 0..tokens_a.len() {
        let actual_a = run_tokens(
            backend,
            &mut interleaved_a,
            &tokens_a[position..=position],
            position,
        );
        assert_eq!(actual_a[0], expected_a[position]);

        let actual_b = run_tokens(
            backend,
            &mut interleaved_b,
            &tokens_b[position..=position],
            position,
        );
        assert_eq!(actual_b[0], expected_b[position]);
    }

    let b_before_reset = backend
        .save_state(&interleaved_b)
        .expect("session B should be snapshotable")
        .expect("session B should retain recurrent state");
    interleaved_a.clear();
    let reset_snapshot = backend
        .save_state(&interleaved_a)
        .expect("reset state should be snapshotable")
        .expect("hybrid reset should retain allocated state");
    assert_eq!(reset_snapshot.position(), 0);
    assert_eq!(
        backend
            .save_state(&interleaved_b)
            .expect("session B should remain snapshotable")
            .expect("session B should retain recurrent state"),
        b_before_reset
    );
    assert_eq!(
        run_tokens(backend, &mut interleaved_a, &tokens_a[..1], 0)[0],
        expected_a[0]
    );
}

#[test]
fn hybrid_snapshot_rolls_back_with_the_same_kv_boundary() {
    let (fixture, _) =
        common::build_synthetic_qwen35_hybrid_fixture().expect("fixture should build");
    let runtime =
        Runtime::load_with_backend(fixture.path(), BackendKind::Cpu).expect("runtime should load");
    let backend = runtime.backend();
    let mut session = prepared_session(backend);

    run_tokens(backend, &mut session, &[3, 4], 0);
    let checkpoint = backend
        .save_state(&session)
        .expect("snapshot should succeed")
        .expect("hybrid session should have recurrent state");
    assert_eq!(checkpoint.position(), 2);

    let first = backend
        .forward_batch_all_logits(&[5, 6], 2, &mut session)
        .expect("first speculative branch should run");
    session.truncate(2).expect("CPU KV rollback should succeed");
    backend
        .restore_state(&mut session, Some(&checkpoint), 2)
        .expect("recurrent rollback should match the KV boundary");
    let replay = backend
        .forward_batch_all_logits(&[5, 6], 2, &mut session)
        .expect("replayed branch should run");

    assert_eq!(replay, first);
    let after_replay = backend
        .save_state(&session)
        .expect("snapshot should succeed")
        .expect("hybrid session should have recurrent state");
    assert_eq!(after_replay.position(), 4);
    assert!(backend
        .restore_state(&mut session, Some(&checkpoint), 1)
        .is_err());
    assert_eq!(
        backend
            .save_state(&session)
            .expect("state should remain snapshotable")
            .expect("hybrid state should remain present"),
        after_replay
    );
}

#[test]
fn injected_kv_failure_does_not_commit_recurrent_state() {
    let (fixture, spec) =
        common::build_synthetic_qwen35_hybrid_fixture().expect("fixture should build");
    let runtime =
        Runtime::load_with_backend(fixture.path(), BackendKind::Cpu).expect("runtime should load");
    let backend = runtime.backend();
    let mut session = prepared_session(backend);
    run_tokens(backend, &mut session, &[3], 0);
    let before = backend
        .save_state(&session)
        .expect("snapshot should succeed")
        .expect("hybrid session should have recurrent state");

    if let BackendSession::Cpu { cache, .. } = &mut session {
        *cache = SessionKvCache::new(KvCacheMode::F32, spec.block_count, 1, 4);
    } else {
        panic!("CPU runtime returned a non-CPU backend session");
    }

    let mut logits = Vec::new();
    let error = backend
        .forward_token(4, 1, &mut session, &mut logits)
        .expect_err("wrong KV geometry should fail after recurrent layers execute");
    assert!(
        error.to_string().contains("width") || error.to_string().contains("cache"),
        "unexpected injected failure: {error}"
    );
    let after = backend
        .save_state(&session)
        .expect("rolled-back state should remain snapshotable")
        .expect("hybrid session should have recurrent state");
    assert_eq!(after, before);
}

fn assert_repeated_hybrid_prefix_matches(runtime: &std::sync::Arc<Runtime>) {
    let request = GenerateRequest {
        prompt: "hello world".to_string(),
        max_tokens: 3,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(35),
        ..Default::default()
    };
    let first = runtime
        .new_session_with_cache_mode(KvCacheMode::F32)
        .generate(&request)
        .expect("initial hybrid generation should populate the prefix cache");
    let after_first = runtime.prefix_cache_status();
    assert_eq!(after_first.misses, 1);
    assert_eq!(after_first.inserts, 1);
    assert_eq!(after_first.entries, 1);
    let second = runtime
        .new_session_with_cache_mode(KvCacheMode::F32)
        .generate(&request)
        .expect("repeated hybrid generation should restore KV and recurrent state");
    assert_eq!(second, first);
    let after_second = runtime.prefix_cache_status();
    assert_eq!(after_second.hits, 1);
    assert!(after_second.prefill_tokens_saved >= 8);
}

#[test]
fn repeated_cpu_hybrid_prompt_restores_kv_and_recurrent_prefix_state() {
    let (fixture, _) =
        common::build_synthetic_qwen35_hybrid_fixture().expect("fixture should build");
    let runtime =
        Runtime::load_with_backend(fixture.path(), BackendKind::Cpu).expect("runtime should load");
    assert_repeated_hybrid_prefix_matches(&runtime);
}

#[test]
fn cpu_hybrid_status_reports_session_ownership_and_capability_limits() {
    let (fixture, _) =
        common::build_synthetic_qwen35_hybrid_fixture().expect("fixture should build");
    let runtime =
        Runtime::load_with_backend(fixture.path(), BackendKind::Cpu).expect("runtime should load");
    let status = runtime
        .hybrid_state_status()
        .expect("Qwen3.5 runtime should report hybrid state");

    assert_eq!(status.owner, "session");
    assert_eq!(status.backend, "cpu");
    assert_eq!(status.recurrent_layers, 3);
    assert_eq!(status.full_attention_layers, 1);
    assert!(status.durable_snapshot_bytes > 0);
    assert_eq!(
        status.bytes_per_session,
        status.durable_snapshot_bytes.saturating_mul(2)
    );
    assert!(status.prefix_cache_supported);
    assert!(!status.shared_f32_kv_page_cow_supported);
    assert!(!status.quantized_kv_page_cow_supported);
    assert!(!status.speculative_rollback_supported);
    assert!(!status.speculative_decoding_enabled);
    assert!(status.speculative_decoding_disabled_reason.is_some());
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA device"]
fn repeated_cuda_hybrid_prompt_restores_kv_and_recurrent_prefix_state() {
    let (fixture, _) =
        common::build_synthetic_qwen35_hybrid_fixture().expect("fixture should build");
    let runtime = Runtime::load_with_backend(fixture.path(), BackendKind::CudaResident)
        .expect("Qwen3.5 CUDA runtime should load");
    assert_repeated_hybrid_prefix_matches(&runtime);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA device"]
fn cuda_hybrid_ngram_speculation_matches_eager_reference() {
    let (fixture, _) =
        common::build_synthetic_qwen35_hybrid_long_fixture().expect("fixture should build");
    let runtime = Runtime::load_with_backend(fixture.path(), BackendKind::CudaResident)
        .expect("Qwen3.5 CUDA runtime should load");
    // The pinned fixture produces a deterministic repeating decode after this
    // prompt. Once the trigram recurs without overlap, prompt lookup drafts a
    // suffix that the target accepts in full.
    let prompt = (3..32)
        .map(|token| format!("tok003 tok003 tok{token:03}"))
        .collect::<Vec<_>>()
        .join(" ");
    let request = GenerateRequest {
        prompt,
        max_tokens: 12,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(35),
        ..Default::default()
    };
    let mut reference = runtime.new_session_with_cache_mode(KvCacheMode::F32);
    reference.set_ngram_speculation_enabled(false);
    let expected = reference
        .generate(&request)
        .expect("non-speculative CUDA hybrid reference should execute");
    assert_eq!(reference.speculative_decode_stats().verification_batches, 0);

    runtime.clear_prefix_cache();
    let mut speculative = runtime.new_session_with_cache_mode(KvCacheMode::F32);
    assert!(speculative.ngram_speculation_enabled());
    let actual = speculative
        .generate(&request)
        .expect("speculative CUDA hybrid generation should execute");
    let stats = speculative.speculative_decode_stats();

    assert_eq!(actual, expected);
    assert!(stats.verification_batches > 0);
    assert!(stats.drafted_tokens > 0);
    assert_eq!(
        stats.accepted_tokens.saturating_add(stats.rejected_tokens),
        stats.drafted_tokens
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA device"]
fn cuda_hybrid_ngram_rejection_rolls_back_to_eager_reference() {
    let (fixture, _) =
        common::build_synthetic_qwen35_hybrid_fixture().expect("fixture should build");
    let runtime = Runtime::load_with_backend(fixture.path(), BackendKind::CudaResident)
        .expect("Qwen3.5 CUDA runtime should load");
    let request = GenerateRequest {
        prompt: "tok008 tok009 tok008 tok003 tok008 tok009".to_string(),
        max_tokens: 8,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(35),
        ..Default::default()
    };

    let mut reference = runtime.new_session_with_cache_mode(KvCacheMode::F32);
    reference.set_ngram_speculation_enabled(false);
    let expected = reference
        .generate(&request)
        .expect("non-speculative CUDA hybrid reference should execute");
    runtime.clear_prefix_cache();

    let mut speculative = runtime.new_session_with_cache_mode(KvCacheMode::F32);
    let actual = speculative
        .generate(&request)
        .expect("speculative CUDA hybrid generation should execute");
    let stats = speculative.speculative_decode_stats();

    assert_eq!(actual, expected);
    assert!(stats.verification_batches > 0);
    assert!(stats.rejected_tokens > 0);
    assert!(stats.rollback_count > 0);
    assert_eq!(
        stats.accepted_tokens.saturating_add(stats.rejected_tokens),
        stats.drafted_tokens
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA device"]
fn cuda_hybrid_ngram_cancellation_restores_the_emitted_boundary() {
    let (fixture, _) =
        common::build_synthetic_qwen35_hybrid_long_fixture().expect("fixture should build");
    let runtime = Runtime::load_with_backend(fixture.path(), BackendKind::CudaResident)
        .expect("Qwen3.5 CUDA runtime should load");
    let prompt = (3..32)
        .map(|token| format!("tok003 tok003 tok{token:03}"))
        .collect::<Vec<_>>()
        .join(" ");
    let request = GenerateRequest {
        prompt,
        max_tokens: 12,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(35),
        ..Default::default()
    };
    let cancel_after = 8usize;

    let mut reference = runtime.new_session_with_cache_mode(KvCacheMode::F32);
    reference.set_ngram_speculation_enabled(false);
    let mut expected = String::new();
    let mut reference_pieces = 0usize;
    let reference_generated = reference
        .generate_stream_with_control(&request, |piece| {
            expected.push_str(piece);
            reference_pieces += 1;
            if reference_pieces == cancel_after {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        })
        .expect("non-speculative cancelled reference should execute");
    let expected_state = reference
        .recurrent_state_snapshot()
        .expect("reference state snapshot should succeed")
        .expect("Qwen3.5 reference state should exist");
    runtime.clear_prefix_cache();

    let mut speculative = runtime.new_session_with_cache_mode(KvCacheMode::F32);
    let mut actual = String::new();
    let mut speculative_pieces = 0usize;
    let actual_generated = speculative
        .generate_stream_with_control(&request, |piece| {
            actual.push_str(piece);
            speculative_pieces += 1;
            if speculative_pieces == cancel_after {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        })
        .expect("speculative cancelled generation should execute");
    let actual_state = speculative
        .recurrent_state_snapshot()
        .expect("speculative state snapshot should succeed")
        .expect("Qwen3.5 speculative state should exist");
    let stats = speculative.speculative_decode_stats();

    assert_eq!(actual, expected);
    assert_eq!(actual_generated, reference_generated);
    assert_eq!(speculative_pieces, reference_pieces);
    assert_eq!(actual_state, expected_state);
    assert!(stats.verification_batches > 0);
    assert!(stats.accepted_tokens > 0);
    assert!(stats.rollback_count > 0);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA device"]
fn concurrent_cuda_hybrid_prefix_forks_are_isolated_and_eviction_safe() {
    use std::sync::{Arc, Barrier};

    let (fixture, _) =
        common::build_synthetic_qwen35_hybrid_fixture().expect("fixture should build");
    let runtime = Runtime::load_with_backend(fixture.path(), BackendKind::CudaResident)
        .expect("Qwen3.5 CUDA runtime should load");
    let request = GenerateRequest {
        prompt: "hello world".to_string(),
        max_tokens: 3,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(35),
        ..Default::default()
    };
    let expected = runtime
        .new_session_with_cache_mode(KvCacheMode::F32)
        .generate(&request)
        .expect("warm generation should insert a hybrid prefix");
    let inserted = runtime.prefix_cache_status();
    assert_eq!(inserted.inserts, 1);
    assert!(inserted.device_resident_bytes > 0);
    assert!(inserted.host_resident_bytes > 0);
    assert_eq!(
        inserted.resident_bytes,
        inserted
            .device_resident_bytes
            .saturating_add(inserted.host_resident_bytes)
    );

    let barrier = Arc::new(Barrier::new(3));
    let run = |runtime: Arc<Runtime>, barrier: Arc<Barrier>, request: GenerateRequest| {
        std::thread::spawn(move || {
            barrier.wait();
            runtime
                .new_session_with_cache_mode(KvCacheMode::F32)
                .generate(&request)
                .expect("concurrent prefix fork should execute")
        })
    };
    let first = run(Arc::clone(&runtime), Arc::clone(&barrier), request.clone());
    let second = run(Arc::clone(&runtime), Arc::clone(&barrier), request);
    barrier.wait();
    assert_eq!(first.join().expect("first fork should not panic"), expected);
    assert_eq!(
        second.join().expect("second fork should not panic"),
        expected
    );
    let after = runtime.prefix_cache_status();
    assert_eq!(after.hits, 2);
    assert!(after.prefill_tokens_saved >= 16);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA device"]
fn cuda_hybrid_scheduler_charges_only_prefix_device_bytes_and_clear_releases_both_tiers() {
    use std::sync::Arc;

    let (fixture, _) =
        common::build_synthetic_qwen35_hybrid_fixture().expect("fixture should build");
    let runtime = Runtime::load_with_backend(fixture.path(), BackendKind::CudaResident)
        .expect("Qwen3.5 CUDA runtime should load");
    let request = GenerateRequest {
        prompt: "hello world".to_string(),
        max_tokens: 2,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(71),
        ..Default::default()
    };
    runtime
        .new_session_with_cache_mode(KvCacheMode::F32)
        .generate(&request)
        .expect("warm generation should insert a hybrid prefix");
    let prefix = runtime.prefix_cache_status();
    assert!(prefix.device_resident_bytes > 0);
    assert!(prefix.host_resident_bytes > 0);

    let scheduler = Arc::new(RequestScheduler::new(
        SchedulerConfig::new(2, 2, 4).expect("scheduler config should validate"),
    ));
    runtime
        .new_session_with_cache_mode(KvCacheMode::F32)
        .generate_scheduled(&request, &scheduler)
        .expect("scheduled prefix reuse should succeed");
    assert_eq!(
        scheduler.status().kv_external_reserved_bytes,
        runtime.prefix_cache_status().device_resident_bytes
    );
    assert!(
        scheduler.status().kv_external_reserved_bytes
            < runtime.prefix_cache_status().resident_bytes,
        "host recurrent snapshot bytes must not be charged against the device KV budget"
    );

    runtime.clear_prefix_cache();
    let cleared = runtime.prefix_cache_status();
    assert_eq!(cleared.entries, 0);
    assert_eq!(cleared.resident_bytes, 0);
    assert_eq!(cleared.device_resident_bytes, 0);
    assert_eq!(cleared.host_resident_bytes, 0);
}

#[cfg(feature = "cuda")]
fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length mismatch");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        assert!(
            error <= tolerance,
            "{label} value {index} differs: CUDA={actual}, CPU={expected}, abs_error={error}, tolerance={tolerance}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA device"]
fn cuda_qwen35_matches_cpu_outputs_and_state_for_128_steps() {
    let (fixture, _) =
        common::build_synthetic_qwen35_hybrid_long_fixture().expect("fixture should build");
    let cpu = Runtime::load_with_backend(fixture.path(), BackendKind::Cpu)
        .expect("CPU runtime should load");
    let cuda = Runtime::load_with_backend(fixture.path(), BackendKind::CudaResident)
        .expect("Qwen3.5 CUDA runtime should load");
    let tokens = (0..128)
        .map(|index| 3 + (index % 29) as u32)
        .collect::<Vec<_>>();
    let mut cpu_session = prepared_session(cpu.backend());
    let mut cuda_session = prepared_session(cuda.backend());

    let expected = cpu
        .backend()
        .forward_batch_all_logits(&tokens, 0, &mut cpu_session)
        .expect("CPU Qwen3.5 reference should execute");
    let actual = cuda
        .backend()
        .forward_batch_all_logits(&tokens, 0, &mut cuda_session)
        .expect("CUDA Qwen3.5 recurrent path should execute");
    assert_close(&actual, &expected, 2e-3, "128-step final logits");
    for step in 0..tokens.len() {
        let offset = step * 32;
        let expected_top = expected[offset..offset + 32]
            .iter()
            .enumerate()
            .max_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
            .map(|(index, _)| index);
        let actual_top = actual[offset..offset + 32]
            .iter()
            .enumerate()
            .max_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
            .map(|(index, _)| index);
        assert_eq!(actual_top, expected_top, "top token differs at step {step}");
    }

    let cpu_state = cpu
        .backend()
        .save_state(&cpu_session)
        .expect("CPU state snapshot should succeed")
        .expect("CPU hybrid state should exist");
    let cuda_state = cuda
        .backend()
        .save_state(&cuda_session)
        .expect("CUDA state snapshot should succeed")
        .expect("CUDA hybrid state should exist");
    assert_eq!(cpu_state.position(), 128);
    assert_eq!(cuda_state.position(), 128);
    assert_eq!(cuda_state.descriptor(), cpu_state.descriptor());
    for (layer, (actual, expected)) in cuda_state
        .layers()
        .iter()
        .zip(cpu_state.layers())
        .enumerate()
    {
        match (actual, expected) {
            (Some(actual), Some(expected)) => {
                assert_close(
                    actual.conv_state_f32(),
                    expected.conv_state_f32(),
                    5e-4,
                    &format!("layer {layer} convolution state"),
                );
                assert_close(
                    actual.recurrent_state_f32(),
                    expected.recurrent_state_f32(),
                    2e-3,
                    &format!("layer {layer} recurrent state"),
                );
            }
            (None, None) => {}
            _ => panic!("layer {layer} recurrent-state presence differs"),
        }
    }

    let resources = cuda.gpu_resource_status();
    assert_eq!(
        resources.arena_allocations.recurrent_state_bytes,
        cuda_session.recurrent_state_allocated_bytes()
    );
    assert!(resources.arena_allocations.recurrent_state_bytes > 0);
    assert!(
        resources.arena_allocations.graph_bytes > 0,
        "captured graph executables must retain a central-arena admission lease"
    );
    assert_eq!(
        cuda_session.cuda_graph_capture_status(),
        Some("captured"),
        "the alternating recurrent pointer generations should retain an exact graph"
    );
    assert_eq!(cuda_session.cuda_graph_last_error(), None);
    let hybrid_status = cuda
        .hybrid_state_status()
        .expect("CUDA Qwen3.5 should report hybrid state");
    assert_eq!(hybrid_status.backend, "cuda-resident");
    assert_eq!(
        hybrid_status.bytes_per_session,
        hybrid_status.durable_snapshot_bytes.saturating_mul(3)
    );
    assert!(hybrid_status.shared_f32_kv_page_cow_supported);
    assert!(!hybrid_status.quantized_kv_page_cow_supported);
    assert!(hybrid_status.speculative_rollback_supported);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA device"]
fn cuda_qwen35_sessions_are_isolated_when_executed_concurrently_and_reset() {
    use std::sync::{Arc, Barrier};

    let (fixture, _) =
        common::build_synthetic_qwen35_hybrid_fixture().expect("fixture should build");
    let runtime = Runtime::load_with_backend(fixture.path(), BackendKind::CudaResident)
        .expect("Qwen3.5 CUDA runtime should load");
    let tokens_a = vec![3, 4, 5, 6, 7, 8, 9, 10];
    let tokens_b = vec![11, 12, 13, 14, 15, 16, 17, 18];

    let mut isolated_a = prepared_session(runtime.backend());
    let expected_a = run_tokens(runtime.backend(), &mut isolated_a, &tokens_a, 0);
    let mut isolated_b = prepared_session(runtime.backend());
    let expected_b = run_tokens(runtime.backend(), &mut isolated_b, &tokens_b, 0);

    let barrier = Arc::new(Barrier::new(3));
    let spawn_run = |runtime: Arc<Runtime>, barrier: Arc<Barrier>, tokens: Vec<u32>| {
        std::thread::spawn(move || {
            let mut session = prepared_session(runtime.backend());
            barrier.wait();
            let logits = run_tokens(runtime.backend(), &mut session, &tokens, 0);
            let state = runtime
                .backend()
                .save_state(&session)
                .expect("concurrent CUDA state snapshot should succeed")
                .expect("concurrent CUDA state should exist");
            (logits, state)
        })
    };
    let thread_a = spawn_run(Arc::clone(&runtime), Arc::clone(&barrier), tokens_a.clone());
    let thread_b = spawn_run(Arc::clone(&runtime), Arc::clone(&barrier), tokens_b.clone());
    barrier.wait();
    let (actual_a, state_a) = thread_a.join().expect("session A thread should not panic");
    let (actual_b, state_b) = thread_b.join().expect("session B thread should not panic");
    assert_eq!(state_a.position(), tokens_a.len() as u64);
    assert_eq!(state_b.position(), tokens_b.len() as u64);
    for (step, (actual, expected)) in actual_a.iter().zip(expected_a.iter()).enumerate() {
        assert_close(actual, expected, 1e-6, &format!("session A step {step}"));
    }
    for (step, (actual, expected)) in actual_b.iter().zip(expected_b.iter()).enumerate() {
        assert_close(actual, expected, 1e-6, &format!("session B step {step}"));
    }

    let mut session_a = prepared_session(runtime.backend());
    let mut session_b = prepared_session(runtime.backend());
    run_tokens(runtime.backend(), &mut session_a, &tokens_a[..4], 0);
    run_tokens(runtime.backend(), &mut session_b, &tokens_b[..4], 0);
    let b_before_reset = runtime
        .backend()
        .save_state(&session_b)
        .expect("session B snapshot should succeed")
        .expect("session B state should exist");
    session_a.clear();
    runtime
        .backend()
        .prepare_session_state(&mut session_a)
        .expect("reset CUDA state should zero during fallible preparation");
    let reset_state = runtime
        .backend()
        .save_state(&session_a)
        .expect("reset CUDA snapshot should succeed")
        .expect("reset CUDA state should exist");
    assert_eq!(reset_state.position(), 0);
    for layer in reset_state.layers().iter().flatten() {
        assert!(layer.conv_state_f32().iter().all(|value| *value == 0.0));
        assert!(layer
            .recurrent_state_f32()
            .iter()
            .all(|value| *value == 0.0));
    }
    assert_eq!(
        runtime
            .backend()
            .save_state(&session_b)
            .expect("session B snapshot after reset should succeed")
            .expect("session B state should exist"),
        b_before_reset
    );
    let restarted = run_tokens(runtime.backend(), &mut session_a, &tokens_a[..1], 0);
    assert_close(
        &restarted[0],
        &expected_a[0],
        1e-6,
        "reset session first token",
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA device"]
fn cuda_qwen35_snapshot_restore_and_forward_failure_do_not_publish_pending_state() {
    use xrt_runtime::backend::CudaLayerKvStore;

    let (fixture, _) =
        common::build_synthetic_qwen35_hybrid_fixture().expect("fixture should build");
    let runtime = Runtime::load_with_backend(fixture.path(), BackendKind::CudaResident)
        .expect("Qwen3.5 CUDA runtime should load");
    let backend = runtime.backend();
    let mut session = prepared_session(backend);
    run_tokens(backend, &mut session, &[3, 4], 0);
    let checkpoint = backend
        .save_state(&session)
        .expect("CUDA checkpoint should succeed")
        .expect("CUDA state should exist");

    let first = backend
        .forward_batch_all_logits(&[5, 6], 2, &mut session)
        .expect("first CUDA branch should execute");
    session
        .truncate(2)
        .expect("CUDA KV rollback should succeed");
    backend
        .restore_state(&mut session, Some(&checkpoint), 2)
        .expect("CUDA recurrent rollback should succeed");
    let replay = backend
        .forward_batch_all_logits(&[5, 6], 2, &mut session)
        .expect("replayed CUDA branch should execute");
    assert_close(&replay, &first, 1e-6, "restored CUDA branch");

    let mut failed_session = prepared_session(backend);
    run_tokens(backend, &mut failed_session, &[3], 0);
    let before_failure = backend
        .save_state(&failed_session)
        .expect("pre-failure CUDA snapshot should succeed")
        .expect("pre-failure CUDA state should exist");
    if let BackendSession::Cuda {
        device,
        layer_caches,
        ..
    } = &mut failed_session
    {
        let capacity = match &layer_caches[3] {
            CudaLayerKvStore::F32(cache) => cache.capacity(),
            _ => panic!("Qwen3.5 test session should use f32 KV"),
        };
        let mut invalid = device
            .alloc_paged_layer_kv_cache(capacity, 1, 4)
            .expect("invalid test cache should allocate");
        let key = device
            .upload_f32(&[0.0])
            .expect("invalid test key should upload");
        let value = device
            .upload_f32(&[0.0])
            .expect("invalid test value should upload");
        device
            .append_layer_kv(&mut invalid, &key, &value)
            .expect("invalid test cache should reach position one");
        layer_caches[3] = CudaLayerKvStore::F32(invalid);
    } else {
        panic!("CUDA runtime returned a non-CUDA session");
    }
    let mut logits = Vec::new();
    let error = backend
        .forward_token(4, 1, &mut failed_session, &mut logits)
        .expect_err("invalid full-attention KV width should fail after recurrent kernels");
    assert!(
        error.to_string().contains("width") || error.to_string().contains("length"),
        "unexpected injected CUDA error: {error}"
    );
    let after_failure = backend
        .save_state(&failed_session)
        .expect("failed CUDA transaction should remain snapshotable")
        .expect("failed CUDA transaction should retain state");
    assert_eq!(after_failure, before_failure);
}
