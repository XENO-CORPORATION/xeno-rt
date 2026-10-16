use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use xrt_kernels::cpu::{
    fp8::{BlockFp8Matrix, E4M3FN},
    matmul::matvec,
};

fn execution(c: &mut Criterion) {
    let mut group = c.benchmark_group("fp8_expert_decode");
    group.sample_size(20);
    for (rows, cols) in [(512, 2048), (2048, 512)] {
        let bytes: Vec<_> = (0..rows * cols)
            .map(|i| ((i * 13) % 126) as u8 | if i % 3 == 0 { 128 } else { 0 })
            .collect();
        let scale_cols = cols / 128;
        let scales = vec![0.01; rows / 128 * scale_cols];
        let matrix = BlockFp8Matrix::new(&bytes, &scales, rows, cols, 128, 128).unwrap();
        let dense: Vec<_> = (0..rows * cols)
            .map(|i| {
                E4M3FN[bytes[i] as usize] * scales[(i / cols / 128) * scale_cols + (i % cols) / 128]
            })
            .collect();
        let input: Vec<_> = (0..cols).map(|i| (i as f32 * 0.01).sin()).collect();
        let mut output = vec![0.0; rows];
        group.throughput(Throughput::Elements((rows * cols) as u64));
        group.bench_with_input(
            BenchmarkId::new("fp8_f32", format!("{rows}x{cols}")),
            &(),
            |b, _| {
                b.iter(|| {
                    matrix
                        .matvec(black_box(&input), black_box(&mut output))
                        .unwrap()
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("fp8_w8a8", format!("{rows}x{cols}")),
            &(),
            |b, _| {
                b.iter(|| {
                    matrix
                        .matvec_w8a8(black_box(&input), black_box(&mut output))
                        .unwrap()
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("expanded_f32", format!("{rows}x{cols}")),
            &(),
            |b, _| {
                b.iter(|| {
                    matvec(
                        black_box(&dense),
                        rows,
                        cols,
                        black_box(&input),
                        black_box(&mut output),
                    )
                })
            },
        );
    }
    group.finish();
}
fn delta_execution(c: &mut Criterion) {
    use xrt_kernels::cpu::delta_state::{GatedDeltaState, HeadExpansion};
    let mut group = c.benchmark_group("drama3_delta_decode");
    group.sample_size(20);
    let q: Vec<f32> = (0..16 * 128).map(|i| (i as f32 * 0.013).sin()).collect();
    let k: Vec<f32> = (0..16 * 128).map(|i| (i as f32 * 0.017).cos()).collect();
    let v = vec![0.1; 32 * 128];
    let decay = vec![-0.1; 32];
    let beta = vec![0.3; 32];
    for ordered in [false, true] {
        let mut state = GatedDeltaState::new(
            16,
            32,
            128,
            128,
            HeadExpansion::RepeatInterleave,
            8 * 1024 * 1024,
        )
        .unwrap();
        let mut out = vec![0.0; 32 * 128];
        group.bench_function(
            if ordered {
                "ordered_fp32"
            } else {
                "fused_reference"
            },
            |b| {
                b.iter(|| {
                    if ordered {
                        state.step_ordered(black_box(&q), &k, &v, &decay, &beta, &mut out)
                    } else {
                        state.step(black_box(&q), &k, &v, &decay, &beta, &mut out)
                    }
                    .unwrap();
                    black_box(&out);
                })
            },
        );
    }
    group.finish();
}
criterion_group!(benches, execution, delta_execution);
criterion_main!(benches);
