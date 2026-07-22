#[cfg(feature = "cuda")]
#[path = "../tests/common/mod.rs"]
mod common;

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};
#[cfg(feature = "cuda")]
use xrt_gguf::GgufFile;
#[cfg(feature = "cuda")]
use xrt_models::LlamaModel;
#[cfg(feature = "cuda")]
use xrt_runtime::{
    BackendKind, BackendSession, CausalLmBackend, CudaGraphMode, CudaResidentBackend,
    GenerateRequest, GpuResourceConfig, KvCacheMode, Runtime,
};

#[cfg(feature = "cuda")]
const WARM_TOKENS: usize = 2;
#[cfg(feature = "cuda")]
const MEASURED_TOKENS: usize = 126;

#[cfg(feature = "cuda")]
fn backend(path: &std::path::Path, graph_mode: CudaGraphMode) -> CudaResidentBackend {
    let gguf = Arc::new(GgufFile::open(path).expect("benchmark GGUF should open"));
    let model =
        Arc::new(LlamaModel::from_gguf(Arc::clone(&gguf)).expect("benchmark model should load"));
    CudaResidentBackend::new(
        model,
        gguf.as_ref(),
        GpuResourceConfig {
            cuda_graph_mode: graph_mode,
            ..GpuResourceConfig::default()
        },
    )
    .expect("benchmark CUDA backend should load")
}

#[cfg(feature = "cuda")]
fn prepared_session(backend: &CudaResidentBackend) -> BackendSession {
    let mut session = backend.new_session(KvCacheMode::F32, 32);
    backend
        .prepare_session_state(&mut session)
        .expect("benchmark recurrent state should prepare");
    session
        .prepare_for_total_len(WARM_TOKENS + MEASURED_TOKENS)
        .expect("benchmark KV should preallocate");
    let mut logits = Vec::new();
    for position in 0..WARM_TOKENS {
        backend
            .forward_token(
                3 + (position % 29) as u32,
                position,
                &mut session,
                &mut logits,
            )
            .expect("benchmark graph warmup token should execute");
    }
    session
}

#[cfg(feature = "cuda")]
fn run_measured(backend: &CudaResidentBackend, session: &mut BackendSession) -> Vec<f32> {
    let mut logits = Vec::new();
    for position in WARM_TOKENS..WARM_TOKENS + MEASURED_TOKENS {
        backend
            .forward_token(3 + (position % 29) as u32, position, session, &mut logits)
            .expect("benchmark token should execute");
    }
    logits
}

#[cfg(feature = "cuda")]
fn benchmark_qwen35_cuda_graph(c: &mut Criterion) {
    let (fixture, _) = common::build_synthetic_qwen35_hybrid_long_fixture()
        .expect("benchmark fixture should build");
    let eager = backend(fixture.path(), CudaGraphMode::Disabled);
    let graph = backend(fixture.path(), CudaGraphMode::Enabled);

    let mut eager_parity = prepared_session(&eager);
    let eager_logits = run_measured(&eager, &mut eager_parity);
    let mut graph_parity = prepared_session(&graph);
    assert_eq!(graph_parity.cuda_graph_capture_status(), Some("captured"));
    let graph_logits = run_measured(&graph, &mut graph_parity);
    assert_eq!(graph_logits.len(), eager_logits.len());
    for (&actual, &expected) in graph_logits.iter().zip(&eager_logits) {
        assert!((actual - expected).abs() <= 2e-3);
    }

    let mut group = c.benchmark_group("qwen35_cuda/decode");
    group.throughput(Throughput::Elements(MEASURED_TOKENS as u64));
    for (name, backend) in [("eager", &eager), ("graph", &graph)] {
        group.bench_function(name, |b| {
            b.iter_batched(
                || prepared_session(backend),
                |mut session| black_box(run_measured(backend, &mut session)),
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

#[cfg(feature = "cuda")]
fn benchmark_qwen35_prefix_ttft(c: &mut Criterion) {
    let (fixture, _) =
        common::build_synthetic_qwen35_hybrid_fixture().expect("benchmark fixture should build");
    let runtime = Runtime::load_with_backend(fixture.path(), BackendKind::CudaResident)
        .expect("benchmark CUDA runtime should load");
    let request = GenerateRequest {
        prompt: "hello world".to_string(),
        max_tokens: 1,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(35),
        ..Default::default()
    };

    let mut group = c.benchmark_group("qwen35_cuda/prefix_ttft");
    group.bench_function("cold_prefill", |b| {
        b.iter_batched(
            || runtime.clear_prefix_cache(),
            |_| {
                black_box(
                    runtime
                        .new_session_with_cache_mode(KvCacheMode::F32)
                        .generate(&request)
                        .expect("cold prefix benchmark should execute"),
                )
            },
            BatchSize::SmallInput,
        );
    });
    group.bench_function("prefix_hit", |b| {
        b.iter_batched(
            || {
                runtime.clear_prefix_cache();
                runtime
                    .new_session_with_cache_mode(KvCacheMode::F32)
                    .generate(&request)
                    .expect("prefix benchmark setup should insert")
            },
            |_| {
                black_box(
                    runtime
                        .new_session_with_cache_mode(KvCacheMode::F32)
                        .generate(&request)
                        .expect("prefix hit benchmark should execute"),
                )
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

#[cfg(feature = "cuda")]
fn benchmark_qwen35_fast_checkpoint(c: &mut Criterion) {
    let (fixture, _) = common::build_synthetic_qwen35_hybrid_long_fixture()
        .expect("benchmark fixture should build");
    let backend = backend(fixture.path(), CudaGraphMode::Disabled);
    let mut session = prepared_session(&backend);
    let journal_payload_bytes = session.recurrent_state_allocated_bytes() / 3;
    assert!(journal_payload_bytes > 0);

    let mut group = c.benchmark_group("qwen35_cuda/fast_checkpoint");
    group.throughput(Throughput::Bytes(journal_payload_bytes));
    group.bench_function("begin_commit", |b| {
        b.iter(|| {
            session
                .begin_fast_recurrent_checkpoint(WARM_TOKENS)
                .expect("benchmark checkpoint should start");
            session
                .commit_fast_recurrent_checkpoint()
                .expect("benchmark checkpoint should commit");
            session
                .synchronize_cuda()
                .expect("checkpoint timing must include CUDA completion");
        });
    });

    group.throughput(Throughput::Bytes(journal_payload_bytes.saturating_mul(2)));
    group.bench_function("begin_rollback", |b| {
        b.iter(|| {
            session
                .begin_fast_recurrent_checkpoint(WARM_TOKENS)
                .expect("benchmark checkpoint should start");
            session
                .rollback_fast_recurrent_checkpoint(WARM_TOKENS)
                .expect("benchmark checkpoint should roll back");
            session
                .synchronize_cuda()
                .expect("rollback timing must include CUDA completion");
            black_box(session.recurrent_state_allocated_bytes());
        });
    });
    group.finish();
}

#[cfg(feature = "cuda")]
criterion_group!(
    qwen35_cuda_benches,
    benchmark_qwen35_cuda_graph,
    benchmark_qwen35_prefix_ttft,
    benchmark_qwen35_fast_checkpoint
);
#[cfg(feature = "cuda")]
criterion_main!(qwen35_cuda_benches);

#[cfg(not(feature = "cuda"))]
fn main() {}
