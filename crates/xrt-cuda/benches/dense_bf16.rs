//! Warm encoded dense projection; excludes weight upload and model orchestration.
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use xrt_cuda::{CudaDevice, GpuDenseWorkspace, GpuOrderedDeltaState};
use xrt_kernels::cpu::delta_state::HeadExpansion;
fn rounded(value: f32) -> f32 {
    let b = value.to_bits();
    f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff0000)
}
fn dense(c: &mut Criterion) {
    let device = CudaDevice::new(0).expect("requires NVIDIA driver and CUDA-12 cuBLAS");
    let baseline = device.allocation_stats().current_bytes;
    let mut group = c.benchmark_group("warm_dense_bf16_activation");
    group.sample_size(20);
    for (rows, cols) in [(8192usize, 2048usize), (2048, 8192)] {
        let bytes: Vec<u8> = (0..rows * cols)
            .flat_map(|i| {
                ((rounded((i as f32 * 0.037).sin() * 0.1).to_bits() >> 16) as u16).to_le_bytes()
            })
            .collect();
        let matrix = device
            .upload_dense_bf16_matrix(&bytes, rows, cols, bytes.len() as u64)
            .unwrap();
        let input: Vec<_> = (0..cols)
            .map(|i| rounded((i as f32 * 0.031).cos()))
            .collect();
        let mut workspace = GpuDenseWorkspace::new(device.clone(), 4 * 1024 * 1024);
        let expected = workspace.matvec_bf16(&matrix, &input).unwrap();
        let actual = workspace.matvec_bf16_activation(&matrix, &input).unwrap();
        for (a, b) in actual.iter().zip(&expected) {
            assert!((a - b).abs() <= 0.016 * b.abs().max(1.0));
        }
        group.bench_with_input(
            BenchmarkId::new("custom_fp32_then_bf16", format!("{rows}x{cols}")),
            &(),
            |b, _| {
                b.iter(|| {
                    let mut output = workspace.matvec_bf16(&matrix, black_box(&input)).unwrap();
                    for v in &mut output {
                        *v = rounded(*v);
                    }
                    black_box(output)
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("cublas_bf16", format!("{rows}x{cols}")),
            &(),
            |b, _| {
                b.iter(|| {
                    black_box(
                        workspace
                            .matvec_bf16_activation(&matrix, black_box(&input))
                            .unwrap(),
                    )
                })
            },
        );
        workspace.clear();
    }
    group.finish();
    device.synchronize().unwrap();
    assert_eq!(device.allocation_stats().current_bytes, baseline);
}
fn normalization(c: &mut Criterion) {
    let device = CudaDevice::new(0).unwrap();
    let baseline = device.allocation_stats().current_bytes;
    let mut group = c.benchmark_group("bf16_normalization_roundtrip");
    group.sample_size(20);
    for n in [128usize, 2048] {
        let input: Vec<f32> = (0..n).map(|i| rounded((i as f32 * 0.013).sin())).collect();
        let weight = vec![0.0; n];
        let mut workspace = GpuDenseWorkspace::new(device.clone(), 4 * 1024 * 1024);
        group.bench_function(BenchmarkId::new("cuda", n), |b| {
            b.iter(|| {
                black_box(
                    workspace
                        .rmsnorm_bf16(black_box(&input), &weight, 1e-6, true, 1)
                        .unwrap(),
                )
            })
        });
        workspace.clear();
    }
    group.finish();
    device.synchronize().unwrap();
    assert_eq!(device.allocation_stats().current_bytes, baseline);
}
fn attention(c: &mut Criterion) {
    let device = CudaDevice::new(0).unwrap();
    let baseline = device.allocation_stats().current_bytes;
    let mut group = c.benchmark_group("bf16_attention_roundtrip");
    group.sample_size(20);
    for (heads, dim) in [(16usize, 256usize), (32, 128)] {
        for rows in [34usize, 257] {
            let q: Vec<_> = (0..heads * dim)
                .map(|i| rounded((i as f32 * 0.013).sin()))
                .collect();
            let k: Vec<_> = (0..rows * 2 * dim)
                .map(|i| rounded((i as f32 * 0.017).cos()))
                .collect();
            let v: Vec<_> = (0..k.len())
                .map(|i| rounded((i as f32 * 0.019).sin()))
                .collect();
            let mut workspace = GpuDenseWorkspace::new(device.clone(), 40 * 1024 * 1024);
            workspace.attention_bf16(&q, &k, &v, heads, 2, dim).unwrap();
            group.bench_function(format!("h{heads}_d{dim}_rows{rows}"), |b| {
                b.iter(|| {
                    black_box(
                        workspace
                            .attention_bf16(black_box(&q), &k, &v, heads, 2, dim)
                            .unwrap(),
                    )
                })
            });
            workspace.clear();
        }
    }
    group.finish();
    device.synchronize().unwrap();
    assert_eq!(device.allocation_stats().current_bytes, baseline);
}
fn recurrent_and_router(c: &mut Criterion) {
    let device = CudaDevice::new(0).unwrap();
    let baseline = device.allocation_stats().current_bytes;
    {
        let mut state = GpuOrderedDeltaState::new(
            device.clone(),
            16,
            32,
            HeadExpansion::RepeatInterleave,
            16 * 1024 * 1024,
        )
        .unwrap();
        let input: Vec<_> = (0..8192)
            .map(|i| rounded((i as f32 * 0.013).sin()))
            .collect();
        let controls = vec![0.0; 32];
        let mut group = c.benchmark_group("bf16_ordered_delta_roundtrip");
        group.sample_size(20);
        group.bench_function("key16_value32", |b| {
            b.iter(|| {
                black_box(
                    state
                        .step(
                            black_box(&input),
                            &controls,
                            &controls,
                            &controls,
                            &controls,
                        )
                        .unwrap(),
                )
            })
        });
        group.finish();
    }
    {
        let mut workspace = GpuDenseWorkspace::new(device.clone(), 4 * 1024 * 1024);
        let input: Vec<_> = (0..256)
            .map(|i| rounded((i as f32 * 0.013).sin()))
            .collect();
        let mut group = c.benchmark_group("bf16_router_roundtrip");
        group.sample_size(20);
        group.bench_function("experts256", |b| {
            b.iter(|| black_box(workspace.router_softmax(black_box(&input)).unwrap()))
        });
        group.finish();
        workspace.clear();
    }
    device.synchronize().unwrap();
    assert_eq!(device.allocation_stats().current_bytes, baseline);
}
criterion_group!(
    benches,
    dense,
    normalization,
    attention,
    recurrent_and_router
);
criterion_main!(benches);
