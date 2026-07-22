use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use xrt_models::{DeltaNetState, DeltaNetStateDescriptor};

fn qwen35_reference_descriptor() -> DeltaNetStateDescriptor {
    let recurrent_layers = (0..36).map(|layer| layer % 4 != 3).collect::<Vec<_>>();
    DeltaNetStateDescriptor::from_geometry(
        "qwen3.5-reference-36-layer",
        4,
        128,
        16,
        2048,
        16,
        &recurrent_layers,
    )
    .expect("benchmark geometry should be valid")
}

fn benchmark_recurrent_state(c: &mut Criterion) {
    let descriptor = qwen35_reference_descriptor();
    let state_bytes = DeltaNetState::try_new(descriptor.clone())
        .expect("benchmark state should allocate")
        .allocated_bytes();
    let mut group = c.benchmark_group("hybrid_state");

    let mut reset_state =
        DeltaNetState::try_new(descriptor.clone()).expect("benchmark state should allocate");
    group.throughput(Throughput::Bytes(state_bytes));
    group.bench_function("reset_36_layer", |b| {
        b.iter(|| {
            reset_state.clear();
            black_box(reset_state.position());
        });
    });

    let snapshot_state =
        DeltaNetState::try_new(descriptor.clone()).expect("benchmark state should allocate");
    group.throughput(Throughput::Bytes(state_bytes / 2));
    group.bench_function("durable_snapshot_36_layer", |b| {
        b.iter(|| {
            black_box(
                snapshot_state
                    .snapshot()
                    .expect("benchmark snapshot should succeed"),
            );
        });
    });

    let mut restore_state =
        DeltaNetState::try_new(descriptor.clone()).expect("benchmark state should allocate");
    let snapshot = restore_state
        .snapshot()
        .expect("benchmark snapshot should succeed");
    group.throughput(Throughput::Bytes(state_bytes / 2));
    group.bench_function("durable_restore_36_layer", |b| {
        b.iter(|| {
            restore_state
                .restore(black_box(&snapshot))
                .expect("benchmark restore should succeed");
        });
    });

    let mut checkpoint_state =
        DeltaNetState::try_new(descriptor).expect("benchmark state should allocate");
    group.throughput(Throughput::Elements(27));
    group.bench_function("token_checkpoint_commit_36_layer", |b| {
        b.iter(|| {
            let position = checkpoint_state.position();
            checkpoint_state
                .begin_token(position)
                .expect("benchmark transaction should start")
                .commit()
                .expect("benchmark transaction should commit");
            black_box(checkpoint_state.position());
        });
    });

    group.finish();
}

criterion_group!(hybrid_state_benches, benchmark_recurrent_state);
criterion_main!(hybrid_state_benches);
