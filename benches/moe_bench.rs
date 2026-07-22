#[path = "../tests/common/mod.rs"]
mod common;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::sync::Arc;
use xrt_models::{group_route_slot_by_expert, route_top_k, MoeRoutingRow};
use xrt_runtime::{
    BackendKind, CudaGraphMode, GpuResourceConfig, KvCacheMode, MoeAcceleration,
    MoePlacementPolicy, MoeRuntimeConfig, Runtime,
};

fn load_cuda_moe_graph_variant(
    model_path: &std::path::Path,
    acceleration: MoeAcceleration,
    budget: u64,
    graph_mode: CudaGraphMode,
) -> Arc<Runtime> {
    Runtime::load_with_backend_configs(
        model_path,
        BackendKind::CudaResident,
        MoeRuntimeConfig {
            acceleration,
            gpu_expert_budget_bytes: Some(budget),
            ..MoeRuntimeConfig::default()
        },
        GpuResourceConfig {
            cuda_graph_mode: graph_mode,
            ..GpuResourceConfig::default()
        },
    )
    .expect("CUDA MoE graph benchmark runtime should load")
}

fn benchmark_router(c: &mut Criterion) {
    let mut group = c.benchmark_group("moe/router");
    for &(expert_count, top_k) in &[(8usize, 2usize), (64, 8), (256, 8)] {
        let logits = (0..expert_count)
            .map(|expert| {
                let value = ((expert * 37 + 11) % 101) as f32;
                (value - 50.0) * 0.03125
            })
            .collect::<Vec<_>>();
        let mut row = MoeRoutingRow::default();
        group.throughput(Throughput::Elements(expert_count as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("experts_{expert_count}"), format!("top_{top_k}")),
            &(expert_count, top_k),
            |b, _| {
                b.iter(|| {
                    route_top_k(black_box(&logits), top_k, black_box(&mut row))
                        .expect("benchmark route should succeed");
                    black_box(row);
                });
            },
        );
    }
    group.finish();
}

fn benchmark_synthetic_moe_decode(c: &mut Criterion) {
    let (fixture, _) = common::build_synthetic_qwen3_moe_fixture().expect("fixture should build");
    let runtime =
        Runtime::load_with_backend(fixture.path(), BackendKind::Cpu).expect("runtime should load");
    let backend = runtime.backend();
    let mut session = backend.new_session(KvCacheMode::F32, 4);
    backend
        .prepare_session_state(&mut session)
        .expect("session preparation should succeed");
    let mut logits = Vec::new();

    let mut group = c.benchmark_group("moe/end_to_end");
    group.throughput(Throughput::Elements(1));
    group.bench_function("tiny_qwen3_moe_decode_token", |b| {
        b.iter(|| {
            session.clear();
            backend
                .forward_token(black_box(3), 0, &mut session, &mut logits)
                .expect("benchmark forward should succeed");
            black_box(&logits);
        });
    });
    group.finish();
}

fn benchmark_grouping(c: &mut Criterion) {
    const TOKENS: usize = 128;
    const EXPERTS: usize = 64;
    const TOP_K: usize = 8;
    let mut routes = [MoeRoutingRow::default(); TOKENS];
    let mut logits = [0.0f32; EXPERTS];
    for (token, route) in routes.iter_mut().enumerate() {
        for (expert, logit) in logits.iter_mut().enumerate() {
            *logit = ((token * 17 + expert * 29) % 101) as f32;
        }
        route_top_k(&logits, TOP_K, route).expect("route should build");
    }
    let mut counts = [0usize; EXPERTS];
    let mut offsets = [0usize; EXPERTS + 1];
    let mut cursors = [0usize; EXPERTS];
    let mut token_indices = [0usize; TOKENS];

    let mut group = c.benchmark_group("moe/dispatch");
    group.throughput(Throughput::Elements(TOKENS as u64));
    group.bench_function("group_128_tokens_64_experts_top8", |b| {
        let mut slot = 0usize;
        b.iter(|| {
            group_route_slot_by_expert(
                black_box(&routes),
                slot,
                EXPERTS,
                black_box(&mut counts),
                black_box(&mut offsets),
                black_box(&mut cursors),
                black_box(&mut token_indices),
            )
            .expect("grouping should succeed");
            slot = (slot + 1) % TOP_K;
        });
    });
    group.finish();
}

fn benchmark_synthetic_moe_prefill(c: &mut Criterion) {
    let (fixture, _) =
        common::build_synthetic_qwen3_moe_benchmark_fixture().expect("fixture should build");
    let legacy = Runtime::load_with_backend(fixture.path(), BackendKind::Cpu)
        .expect("legacy runtime should load");
    let optimized = Runtime::load_with_backend_and_moe_config(
        fixture.path(),
        BackendKind::Cpu,
        MoeRuntimeConfig::optimized_cpu(),
    )
    .expect("optimized runtime should load");
    let tokens = (0..16).map(|index| 3 + index % 12).collect::<Vec<_>>();

    let mut group = c.benchmark_group("moe/end_to_end_prefill");
    group.throughput(Throughput::Elements(tokens.len() as u64));
    for (name, runtime) in [("legacy", legacy), ("optimized_grouped", optimized)] {
        let backend = runtime.backend();
        let mut session = backend.new_session(KvCacheMode::F32, 4);
        group.bench_function(name, |b| {
            b.iter(|| {
                session.clear();
                black_box(
                    backend
                        .forward_batch_all_logits(black_box(&tokens), 0, &mut session)
                        .expect("benchmark prefill should succeed"),
                );
            });
        });
    }
    group.finish();
}

fn benchmark_synthetic_hybrid_moe_decode(c: &mut Criterion) {
    if !cfg!(feature = "cuda") {
        return;
    }
    let (fixture, _) = common::build_synthetic_qwen3_moe_fixture().expect("fixture should build");
    let cpu = Runtime::load_with_backend_and_moe_config(
        fixture.path(),
        BackendKind::Cpu,
        MoeRuntimeConfig::optimized_cpu(),
    )
    .expect("optimized CPU runtime should load");
    let hybrid = Runtime::load_with_backend_and_moe_config(
        fixture.path(),
        BackendKind::CudaResident,
        MoeRuntimeConfig {
            acceleration: MoeAcceleration::Hybrid,
            gpu_expert_budget_bytes: Some(4 * 1024),
            ..MoeRuntimeConfig::default()
        },
    )
    .expect("hybrid CUDA runtime should load");

    let mut group = c.benchmark_group("moe/exact_tier_decode");
    group.throughput(Throughput::Elements(1));
    for (name, runtime) in [("cpu_grouped", cpu), ("cuda_hybrid_forced_split", hybrid)] {
        let backend = runtime.backend();
        let mut session = backend.new_session(KvCacheMode::F32, 4);
        let mut logits = Vec::new();
        group.bench_function(name, |b| {
            b.iter(|| {
                session.clear();
                backend
                    .forward_token(black_box(3), 0, &mut session, &mut logits)
                    .expect("tier benchmark forward should succeed");
                black_box(&logits);
            });
        });
    }
    group.finish();
}

fn benchmark_synthetic_layerwise_moe_prefill(c: &mut Criterion) {
    if !cfg!(feature = "cuda") {
        return;
    }
    let (fixture, _) =
        common::build_synthetic_qwen3_moe_benchmark_fixture().expect("fixture should build");
    let eager = Runtime::load_with_backend_and_moe_config(
        fixture.path(),
        BackendKind::CudaResident,
        MoeRuntimeConfig {
            acceleration: MoeAcceleration::Hybrid,
            gpu_expert_budget_bytes: Some(1024 * 1024),
            ..MoeRuntimeConfig::default()
        },
    )
    .expect("eager hybrid runtime should load");
    let layerwise = Runtime::load_with_backend_and_moe_config(
        fixture.path(),
        BackendKind::CudaResident,
        MoeRuntimeConfig {
            acceleration: MoeAcceleration::Hybrid,
            gpu_expert_budget_bytes: Some(1024 * 1024),
            layerwise_prefill: true,
            ..MoeRuntimeConfig::default()
        },
    )
    .expect("layerwise hybrid runtime should load");
    let tokens = (0..16).map(|index| 3 + index % 24).collect::<Vec<_>>();

    let mut group = c.benchmark_group("moe/layerwise_prefill");
    group.sample_size(20);
    group.throughput(Throughput::Elements(tokens.len() as u64));
    for (name, runtime) in [
        ("eager_token_major", eager),
        ("layerwise_double_buffered", layerwise),
    ] {
        let backend = runtime.backend();
        let mut session = backend.new_session(KvCacheMode::F32, 4);
        group.bench_function(name, |b| {
            b.iter(|| {
                session.clear();
                black_box(
                    backend
                        .forward_batch_all_logits(black_box(&tokens), 0, &mut session)
                        .expect("layerwise benchmark prefill should succeed"),
                );
            });
        });
    }
    group.finish();
}

fn benchmark_synthetic_adaptive_moe_placement(c: &mut Criterion) {
    if !cfg!(feature = "cuda") {
        return;
    }
    let (fixture, _) =
        common::build_synthetic_qwen3_moe_benchmark_fixture().expect("fixture should build");
    let uniform = Runtime::load_with_backend_and_moe_config(
        fixture.path(),
        BackendKind::CudaResident,
        MoeRuntimeConfig {
            acceleration: MoeAcceleration::Hybrid,
            gpu_expert_budget_bytes: Some(1024 * 1024),
            ..MoeRuntimeConfig::default()
        },
    )
    .expect("uniform hybrid runtime should load");
    let adaptive = Runtime::load_with_backend_and_moe_config(
        fixture.path(),
        BackendKind::CudaResident,
        MoeRuntimeConfig {
            acceleration: MoeAcceleration::Hybrid,
            gpu_expert_budget_bytes: Some(1024 * 1024),
            placement: MoePlacementPolicy::Adaptive,
            placement_update_tokens: 16,
            ..MoeRuntimeConfig::default()
        },
    )
    .expect("adaptive hybrid runtime should load");
    let tokens = (0..16).map(|index| 3 + index % 24).collect::<Vec<_>>();
    let mut observation = adaptive.backend().new_session(KvCacheMode::F32, 4);
    adaptive
        .backend()
        .forward_batch_all_logits(&tokens, 0, &mut observation)
        .expect("adaptive observation batch should execute");
    let update_started = std::time::Instant::now();
    adaptive
        .backend()
        .prepare_request()
        .expect("adaptive placement update should succeed");
    let update_wall_micros = update_started.elapsed().as_micros();
    let adaptive_status = adaptive.moe_status();
    eprintln!(
        "adaptive placement diagnostic: generation={}, moves={}, upload_bytes={}, runtime_update_micros={}, wall_update_micros={}",
        adaptive_status.placement_generation,
        adaptive_status.placement_moves,
        adaptive_status.placement_upload_bytes,
        adaptive_status.placement_last_update_micros,
        update_wall_micros
    );

    let mut group = c.benchmark_group("moe/adaptive_placement");
    group.sample_size(20);
    group.throughput(Throughput::Elements(tokens.len() as u64));
    for (name, runtime) in [
        ("uniform_static", uniform),
        ("adaptive_after_bounded_update", adaptive),
    ] {
        let backend = runtime.backend();
        let mut session = backend.new_session(KvCacheMode::F32, 4);
        group.bench_function(name, |b| {
            b.iter(|| {
                session.clear();
                black_box(
                    backend
                        .forward_batch_all_logits(black_box(&tokens), 0, &mut session)
                        .expect("adaptive benchmark prefill should succeed"),
                );
            });
        });
    }
    group.finish();
}

fn benchmark_synthetic_moe_expert_graphs(c: &mut Criterion) {
    if !cfg!(feature = "cuda") {
        return;
    }
    let (fixture, _) = common::build_synthetic_qwen3_moe_fixture().expect("fixture should build");
    let variants = [
        (
            "gpu_eager",
            load_cuda_moe_graph_variant(
                fixture.path(),
                MoeAcceleration::Gpu,
                64 * 1024,
                CudaGraphMode::Disabled,
            ),
        ),
        (
            "gpu_graph",
            load_cuda_moe_graph_variant(
                fixture.path(),
                MoeAcceleration::Gpu,
                64 * 1024,
                CudaGraphMode::Enabled,
            ),
        ),
        (
            "hybrid_eager",
            load_cuda_moe_graph_variant(
                fixture.path(),
                MoeAcceleration::Hybrid,
                4 * 1024,
                CudaGraphMode::Disabled,
            ),
        ),
        (
            "hybrid_graph",
            load_cuda_moe_graph_variant(
                fixture.path(),
                MoeAcceleration::Hybrid,
                4 * 1024,
                CudaGraphMode::Enabled,
            ),
        ),
    ];

    let mut group = c.benchmark_group("moe/expert_graph_decode");
    group.sample_size(20);
    group.throughput(Throughput::Elements(1));
    for (name, runtime) in variants {
        let backend = runtime.backend();
        let mut session = backend.new_session(KvCacheMode::F32, 4);
        let mut logits = Vec::new();
        backend
            .forward_token(3, 0, &mut session, &mut logits)
            .expect("graph benchmark warm token should execute");
        session.clear();
        group.bench_function(name, |b| {
            b.iter(|| {
                session.clear();
                backend
                    .forward_token(black_box(3), 0, &mut session, &mut logits)
                    .expect("graph benchmark token should execute");
                black_box(&logits);
            });
        });
        let status = runtime.moe_status();
        eprintln!(
            "{name}: graph_captures={}, graph_replays={}, graph_fallbacks={}, graph_bytes={}",
            status.graph_captures,
            status.graph_replays,
            status.graph_fallbacks,
            runtime.gpu_resource_status().arena_allocations.graph_bytes
        );
    }
    group.finish();
}

criterion_group!(
    moe_benches,
    benchmark_router,
    benchmark_grouping,
    benchmark_synthetic_moe_decode,
    benchmark_synthetic_moe_prefill,
    benchmark_synthetic_hybrid_moe_decode,
    benchmark_synthetic_layerwise_moe_prefill,
    benchmark_synthetic_adaptive_moe_placement,
    benchmark_synthetic_moe_expert_graphs
);
criterion_main!(moe_benches);
